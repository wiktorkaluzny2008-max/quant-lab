"""Stage 3 models: baselines, small fixed-setting classifiers, calibration, importance.

Design rules (see QUANT_SPEC.md Stage 3):
* Everything that learns (imputer medians, scaler, models, calibrator) is fit on TRAINING
  data only.
* Settings are small, fixed and regularised; ``model_settings(name)`` exposes them for the
  trial log.
* Calibration: model on the first ~80% of the training window, Platt (sigmoid) on the last
  ~20% after a 1-day gap, base model frozen with ``FrozenEstimator``.  No
  ``CalibratedClassifierCV`` and no shuffled CV anywhere.
* A position taken at the close of day t earns ``next_return`` of day t (return t -> t+1).
"""
from __future__ import annotations

import copy

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.frozen import FrozenEstimator
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from quant import features as F

MODEL_NAMES = ("logreg", "rf", "lgbm")
SEED = 42
PROB_EPS = 1e-6
MIN_CAL_POINTS = 500          # spec: calibration slice must have at least 500 days
MIN_ISOTONIC_POINTS = 1000    # spec: isotonic only with >= 1000 calibration points
CAL_FRACTION = 0.20
CAL_GAP = 1

# --------------------------------------------------------------------------- settings
SETTINGS = {
    "logreg": dict(l1_ratio=0.0, C=0.1,  # l1_ratio=0 == pure L2 (penalty= is deprecated in sklearn 1.8+)
                   solver="lbfgs", max_iter=1000, random_state=SEED),
    "rf": dict(n_estimators=300, max_depth=4, min_samples_leaf=100, max_features="sqrt",
               random_state=SEED, n_jobs=1),  # n_jobs=1: bit-exact reproducibility,
    "lgbm": dict(num_leaves=7, max_depth=3, min_child_samples=100, learning_rate=0.02,
                 n_estimators=200, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                 reg_lambda=10.0, random_state=SEED, n_jobs=1, deterministic=True,
                 force_row_wise=True, verbose=-1),
}


def model_settings(name: str, seed: int | None = None) -> dict:
    """Copy of the fixed settings for ``name`` (for the trial log); ``seed`` overrides the RNG."""
    if name not in SETTINGS:
        raise ValueError(f"unknown model {name!r}; expected one of {MODEL_NAMES}")
    s = copy.deepcopy(SETTINGS[name])
    if seed is not None:
        s["random_state"] = seed
    return s


# --------------------------------------------------------------------------- baselines
def annualisation(ticker: str) -> float:
    """Trading periods per year: 365 for BTC-USD (daily, 7 days a week), 252 otherwise."""
    return 365.0 if ticker == F.BTC else 252.0


def climatology_prob(y_train) -> float:
    """Constant P(up) = up-day frequency of the training window."""
    y = np.asarray(y_train, dtype="float64")
    y = y[~np.isnan(y)]
    if len(y) == 0:
        raise ValueError("climatology_prob needs at least one training label")
    return float(y.mean())


def buy_and_hold_positions(index) -> pd.Series:
    return pd.Series(1.0, index=index, name="buy_and_hold")


def always_up_positions(index) -> pd.Series:
    """Always predict up == always long; same positions as buy and hold."""
    return pd.Series(1.0, index=index, name="always_up")


def trend_positions(close: pd.Series, window: int = 200) -> pd.Series:
    """Long (1) if close_t > mean(close_{t-window+1..t}), else flat (0).  Uses data through t
    only; during the warm-up (< window observations) the rule is flat."""
    ma = close.rolling(window, min_periods=window).mean()
    return (close > ma).astype("float64").rename("trend_200")


def training_sigma(train_returns, ann: float = 252.0) -> float:
    """Long-run annualised volatility of the training window (sigma_target)."""
    r = pd.Series(np.asarray(train_returns, dtype="float64")).dropna()
    if len(r) < 2:
        raise ValueError("training_sigma needs at least 2 returns")
    return float(r.std(ddof=1) * np.sqrt(ann))


def vol_targeted_weights(returns: pd.Series, sigma_target: float, ann: float = 252.0,
                         window: int = 21, max_weight: float = 1.0) -> pd.Series:
    """w_t = min(1, sigma_target / sigma_hat_t), sigma_hat = annualised ``window``-day
    realised vol of ``returns`` through t.  ``returns`` must be the *realised* return
    series (return of day t = close_t/close_{t-1}-1), so w_t uses nothing after t.
    Warm-up (< window returns) and sigma_hat == 0: weight 0 for warm-up (unknown risk ->
    stay out), ``max_weight`` for zero vol."""
    sig = returns.rolling(window, min_periods=window).std(ddof=1) * np.sqrt(ann)
    with np.errstate(divide="ignore", invalid="ignore"):
        w = (sigma_target / sig).clip(upper=max_weight)
    w = w.where(sig > 0, max_weight)
    w = w.where(sig.notna(), 0.0)
    return w.astype("float64").rename("vol_targeted")


def apply_positions(positions: pd.Series, next_return: pd.Series) -> pd.Series:
    """Strategy return of day t's position = position_t * next_return_t (return t -> t+1)."""
    return (positions.reindex(next_return.index) * next_return).rename("strategy_return")


# --------------------------------------------------------------------------- preprocessing
class MedianMissingImputer(BaseEstimator, TransformerMixin):
    """Fit on TRAINING data only.

    * ``medians_``: per-column training median (all-NaN column -> 0.0).
    * ``flag_columns_``: columns that have >= 1 NaN *in the training data*.  The output
      column set is therefore decided deterministically at fit time: original columns
      (NaN -> training median) followed by ``<col>__missing`` indicator columns for
      ``flag_columns_`` (1.0 where the value is NaN).  A column that was complete in
      training but is NaN at predict time is median-filled with no flag (the column set
      never changes after fit).
    """

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.columns_ = list(X.columns)
        self.medians_ = X.median().fillna(0.0)
        self.flag_columns_ = [c for c in self.columns_ if X[c].isna().any()]
        self.n_features_in_ = len(self.columns_)
        return self

    def get_feature_names_out(self, input_features=None):
        return np.array(self.columns_ + [f"{c}__missing" for c in self.flag_columns_])

    def transform(self, X):
        X = pd.DataFrame(X)
        if list(X.columns) != self.columns_:
            X = X[self.columns_]
        out = X.fillna(self.medians_)
        for c in self.flag_columns_:
            out[f"{c}__missing"] = X[c].isna().astype("float64")
        return out


def build_model(name: str, seed: int | None = None):
    """Unfitted estimator.  logreg: impute -> scale -> L2 LR; rf: impute -> RF;
    lgbm: LightGBM on raw features (NaN kept)."""
    s = model_settings(name, seed)
    if name == "logreg":
        return Pipeline([("imputer", MedianMissingImputer()), ("scaler", StandardScaler()),
                         ("clf", LogisticRegression(**s))])
    if name == "rf":
        return Pipeline([("imputer", MedianMissingImputer()),
                         ("clf", RandomForestClassifier(**s))])
    if name == "lgbm":
        from lightgbm import LGBMClassifier  # lazy: needs libomp on macOS
        return LGBMClassifier(**s)
    raise ValueError(f"unknown model {name!r}; expected one of {MODEL_NAMES}")


# --------------------------------------------------------------------------- calibration
class CalibratedModel:
    """Base model (frozen) + 1-D sigmoid/isotonic calibrator on logit(raw P(up))."""

    def __init__(self, name, base, frozen_base, calibrator, method, fit_range, cal_range,
                 gap, train_index):
        self.name = name
        self.base_model = base
        self.frozen_base = frozen_base
        self.calibrator = calibrator
        self.method = method
        self.fit_range = fit_range      # (start, stop) positions, stop exclusive
        self.cal_range = cal_range
        self.gap = gap
        self.train_index = train_index  # labels of every row used (fit + calibration)
        self.settings = model_settings(name)

    def predict_raw(self, X) -> np.ndarray:
        p = self.frozen_base.predict_proba(X)[:, 1]
        return np.clip(p, PROB_EPS, 1 - PROB_EPS)

    def predict_up(self, X) -> np.ndarray:
        p = self.predict_raw(X)
        if self.method == "sigmoid":
            z = np.log(p / (1 - p)).reshape(-1, 1)
            q = self.calibrator.predict_proba(z)[:, 1]
        else:
            q = self.calibrator.predict(p)
        return np.clip(q, PROB_EPS, 1 - PROB_EPS)


def fit_calibrated(model_name, X_train, y_train, method="sigmoid", seed=None,
                   cal_fraction=CAL_FRACTION, gap=CAL_GAP,
                   min_cal=MIN_CAL_POINTS) -> CalibratedModel:
    """Chronological fit/calibrate split of ONE training window (rows must be in time order).

    n rows -> calibration = last n_cal = round(cal_fraction*n) rows; ``gap`` rows before it
    are dropped; the base model is fit on everything before the gap.  Raises ValueError if
    n_cal < ``min_cal`` (500; i.e. window < 2500 rows at 20%) or if isotonic is requested
    with < 1000 calibration points.  No fallback is attempted on purpose.
    """
    if method not in ("sigmoid", "isotonic"):
        raise ValueError("method must be 'sigmoid' or 'isotonic'")
    X = pd.DataFrame(X_train)
    y = np.asarray(y_train, dtype="float64")
    if len(X) != len(y):
        raise ValueError("X_train and y_train differ in length")
    if np.isnan(y).any():
        raise ValueError("y_train contains NaN")
    n = len(X)
    n_cal = int(round(cal_fraction * n))
    if n_cal < min_cal:
        raise ValueError(f"training window of {n} rows gives {n_cal} calibration rows; "
                         f"need at least {min_cal} (window >= {int(np.ceil(min_cal / cal_fraction))})")
    if method == "isotonic" and n_cal < MIN_ISOTONIC_POINTS:
        raise ValueError(f"isotonic needs >= {MIN_ISOTONIC_POINTS} calibration points, got {n_cal}")
    fit_stop = n - n_cal - gap
    if fit_stop < 1:
        raise ValueError("no rows left to fit the base model")
    yi = y.astype(int)
    if len(np.unique(yi[:fit_stop])) < 2 or len(np.unique(yi[n - n_cal:])) < 2:
        raise ValueError("both the fit and the calibration slice need both classes")

    base = build_model(model_name, seed)
    base.fit(X.iloc[:fit_stop], yi[:fit_stop])
    frozen = FrozenEstimator(base)

    Xc, yc = X.iloc[n - n_cal:], yi[n - n_cal:]
    p = np.clip(frozen.predict_proba(Xc)[:, 1], PROB_EPS, 1 - PROB_EPS)
    if method == "sigmoid":
        cal = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)  # ~unpenalised Platt
        cal.fit(np.log(p / (1 - p)).reshape(-1, 1), yc)
    else:
        cal = IsotonicRegression(y_min=PROB_EPS, y_max=1 - PROB_EPS, out_of_bounds="clip")
        cal.fit(p, yc)
    used = X.index[:fit_stop].append(X.index[n - n_cal:])
    return CalibratedModel(model_name, base, frozen, cal, method, (0, fit_stop),
                           (n - n_cal, n), gap, used)


class EnsembleModel:
    """Equal-weight mean of the calibrated probabilities of the member models."""

    name = "ensemble"

    def __init__(self, members: dict):
        if not members:
            raise ValueError("ensemble needs at least one member")
        self.members = dict(members)
        self.train_index = next(iter(self.members.values())).train_index
        self.settings = {k: m.settings for k, m in self.members.items()}

    @classmethod
    def fit(cls, X_train, y_train, names=MODEL_NAMES, **kw):
        return cls({n: fit_calibrated(n, X_train, y_train, **kw) for n in names})

    def predict_up(self, X) -> np.ndarray:
        return np.mean([m.predict_up(X) for m in self.members.values()], axis=0)


# --------------------------------------------------------------------------- importance
def group_permutation_importance(model, X_eval, y_eval, groups=None, n_repeats=10,
                                 seed=0, metric="log_loss") -> pd.DataFrame:
    """Out-of-sample permutation importance per FEATURE GROUP.

    All columns of a group are permuted together (same row permutation, so within-group
    structure is kept) on the evaluation set; importance = increase in log loss
    (``metric='log_loss'``) or decrease in AUC (``'auc'``) versus the unpermuted score.
    Positive = the group helps.  ``X_eval`` must be data the model was NOT trained or
    calibrated on: if ``model.train_index`` overlaps ``X_eval.index`` a ValueError is raised.
    Never uses LightGBM gain.  Returns a DataFrame (one row per group) sorted descending.
    """
    X = pd.DataFrame(X_eval)
    y = np.asarray(y_eval, dtype="int64")
    if metric not in ("log_loss", "auc"):
        raise ValueError("metric must be 'log_loss' or 'auc'")
    ti = getattr(model, "train_index", None)
    if ti is not None and len(X.index.intersection(ti)) > 0:
        raise ValueError("X_eval overlaps the model's training/calibration rows; "
                         "importance must be computed out of sample")
    groups = F.feature_groups_for(X.columns) if groups is None else groups

    def score(p):
        if metric == "log_loss":
            return log_loss(y, np.clip(p, PROB_EPS, 1 - PROB_EPS), labels=[0, 1])
        return roc_auc_score(y, p)

    sign = 1.0 if metric == "log_loss" else -1.0
    base = score(model.predict_up(X))
    rng = np.random.default_rng(seed)
    rows = []
    for g in sorted(groups):
        cols = [c for c in groups[g] if c in X.columns]
        if not cols:
            continue
        vals = []
        for _ in range(n_repeats):
            Xp = X.copy()
            perm = rng.permutation(len(X))
            Xp[cols] = X[cols].to_numpy()[perm]
            vals.append(sign * (score(model.predict_up(Xp)) - base))
        rows.append({"group": g, "n_columns": len(cols), "importance_mean": float(np.mean(vals)),
                     "importance_std": float(np.std(vals, ddof=1)) if n_repeats > 1 else 0.0})
    out = pd.DataFrame(rows, columns=["group", "n_columns", "importance_mean", "importance_std"])
    return out.sort_values("importance_mean", ascending=False, ignore_index=True)
