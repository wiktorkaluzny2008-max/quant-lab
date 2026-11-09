"""Tests for quant.models (synthetic data, no network)."""
import pathlib

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import brier_score_loss, log_loss

from quant import models as M


def _lgbm_available():
    try:
        import lightgbm  # noqa: F401
        return True
    except OSError:  # libomp missing on macOS
        return False


needs_lgbm = pytest.mark.skipif(not _lgbm_available(), reason="lightgbm cannot load (libomp missing)")
ALL_MODELS = ["logreg", "rf", pytest.param("lgbm", marks=needs_lgbm)]


def make_data(n=3200, seed=0, signal=1.5, nan_col=False):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2005-01-03", periods=n)
    X = pd.DataFrame({
        "ret_lag_1": rng.normal(size=n),      # group returns (informative)
        "ret_lag_2": rng.normal(size=n),      # group returns (informative)
        "vol_10": rng.normal(size=n),         # group volatility (noise)
        "vol_20": rng.normal(size=n),         # group volatility (noise)
        "rsi_14": rng.normal(size=n),         # group trend (noise)
    }, index=idx)
    if nan_col:
        X.loc[X.index[:300], "rsi_14"] = np.nan
    z = signal * X["ret_lag_1"] + 0.5 * signal * X["ret_lag_2"]
    y = (rng.random(n) < 1 / (1 + np.exp(-z))).astype(float)
    return X, pd.Series(y, index=idx)


# --------------------------------------------------------------------------- preprocessing
def test_imputer_fit_on_train_only():
    Xtr, _ = make_data(600, 1, nan_col=True)
    Xte, _ = make_data(200, 2)
    imp = M.MedianMissingImputer().fit(Xtr)
    med, flags = imp.medians_.copy(), list(imp.flag_columns_)
    imp.transform(Xte * 1000 + 7)  # transforming wild test data must not change fitted state
    assert imp.medians_.equals(med) and imp.flag_columns_ == flags
    # refit on identical train after perturbing test: identical statistics
    imp2 = M.MedianMissingImputer().fit(Xtr)
    assert imp2.medians_.equals(med)
    np.testing.assert_allclose(med["ret_lag_1"], Xtr["ret_lag_1"].median())


def test_missing_flags_and_column_set():
    Xtr, _ = make_data(600, 1, nan_col=True)
    imp = M.MedianMissingImputer().fit(Xtr)
    assert imp.flag_columns_ == ["rsi_14"]
    Xte = Xtr.iloc[:50].copy()
    Xte.loc[Xte.index[0], "ret_lag_1"] = np.nan   # complete in training -> filled, no flag
    out = imp.transform(Xte)
    assert list(out.columns) == list(Xtr.columns) + ["rsi_14__missing"]
    assert not out.isna().any().any()
    assert out.iloc[0]["ret_lag_1"] == Xtr["ret_lag_1"].median()
    assert out["rsi_14__missing"].iloc[0] == 1.0  # first 300 rows NaN in train
    assert set(out["rsi_14__missing"]) <= {0.0, 1.0}
    assert imp.transform(Xtr)["rsi_14"].iloc[0] == Xtr["rsi_14"].median()


def test_scaler_fit_on_train_only():
    Xtr, ytr = make_data(600, 1)
    Xte, _ = make_data(200, 2)
    pipe = M.build_model("logreg").fit(Xtr, ytr.astype(int))
    mean, scale = pipe["scaler"].mean_.copy(), pipe["scaler"].scale_.copy()
    pipe.predict_proba(Xte * 50)
    np.testing.assert_array_equal(pipe["scaler"].mean_, mean)
    np.testing.assert_array_equal(pipe["scaler"].scale_, scale)
    np.testing.assert_allclose(mean, Xtr.mean().to_numpy())


def test_logreg_handles_nan_and_lgbm_keeps_nan():
    X, y = make_data(800, 3, nan_col=True)
    M.build_model("logreg").fit(X, y.astype(int)).predict_proba(X)
    M.build_model("rf").fit(X, y.astype(int)).predict_proba(X)
    if _lgbm_available():   # LightGBM is a bare estimator (no imputer): it receives the NaNs
        M.build_model("lgbm").fit(X, y.astype(int)).predict_proba(X)


# --------------------------------------------------------------------------- settings
def test_settings_within_spec():
    lr, rf, gb = (M.model_settings(n) for n in M.MODEL_NAMES)
    assert lr["l1_ratio"] == 0.0 and 0.05 <= lr["C"] <= 1 and lr["solver"] == "lbfgs"
    assert lr["max_iter"] >= 500
    assert rf["min_samples_leaf"] >= 100 and rf["max_depth"] <= 6 and rf["n_estimators"] >= 200
    assert rf["n_jobs"] <= 4 and rf["random_state"] is not None
    assert gb["num_leaves"] <= 15 and gb["min_child_samples"] >= 100
    assert gb["learning_rate"] <= 0.05 and gb["subsample"] < 1 and gb["verbose"] == -1
    assert gb["deterministic"] is True and gb["random_state"] is not None
    # settings are copies, not shared state
    M.model_settings("rf")["min_samples_leaf"] = 1
    assert M.model_settings("rf")["min_samples_leaf"] >= 100
    # the built estimators really use them
    assert M.build_model("rf")["clf"].min_samples_leaf >= 100
    with pytest.raises(ValueError):
        M.model_settings("nope")


@needs_lgbm
def test_lgbm_estimator_uses_settings():
    m = M.build_model("lgbm")
    assert m.num_leaves <= 15 and m.min_child_samples >= 100


# --------------------------------------------------------------------------- calibration
@pytest.mark.parametrize("name", ALL_MODELS)
def test_calibration_split_is_chronological_with_gap(name):
    X, y = make_data(3000, 4)
    cm = M.fit_calibrated(name, X, y)
    (f0, f1), (c0, c1) = cm.fit_range, cm.cal_range
    n = len(X)
    assert f0 == 0 and c1 == n
    assert c0 - f1 == 1 == cm.gap                  # exactly one dropped row between
    assert c1 - c0 == round(0.2 * n) >= 500
    fit_idx, cal_idx = set(X.index[f0:f1]), set(X.index[c0:c1])
    assert not fit_idx & cal_idx
    assert max(fit_idx) < min(cal_idx)             # calibration strictly later
    assert X.index[f1] not in fit_idx | cal_idx    # the gap row is in neither
    assert len(cm.train_index) == f1 + (c1 - c0)
    # the base model was trained on the early slice only
    clf = cm.base_model[-1] if hasattr(cm.base_model, "steps") else cm.base_model
    if name == "logreg":
        scaler = cm.base_model["scaler"]
        np.testing.assert_allclose(scaler.mean_, X.iloc[:f1].mean().to_numpy())


def test_refuses_short_window_and_isotonic_small():
    X, y = make_data(2000, 5)           # 20% = 400 < 500
    with pytest.raises(ValueError, match="500"):
        M.fit_calibrated("logreg", X, y)
    X, y = make_data(3000, 5)           # 600 cal points < 1000
    with pytest.raises(ValueError, match="isotonic"):
        M.fit_calibrated("logreg", X, y, method="isotonic")
    with pytest.raises(ValueError):
        M.fit_calibrated("logreg", X, y, method="bogus")


def test_isotonic_allowed_with_enough_points():
    X, y = make_data(5200, 6)
    cm = M.fit_calibrated("logreg", X, y, method="isotonic")
    p = cm.predict_up(X.iloc[-200:])
    assert ((p > 0) & (p < 1)).all()


@pytest.mark.parametrize("name", ALL_MODELS)
def test_calibrated_probabilities_in_open_unit_interval(name):
    X, y = make_data(3000, 7, nan_col=True)
    cm = M.fit_calibrated(name, X, y)
    p = cm.predict_up(make_data(500, 8, nan_col=True)[0])
    assert p.shape == (500,) and ((p > 0) & (p < 1)).all()


def test_calibration_improves_miscalibrated_model():
    # random forest with huge leaves is under-confident -> sigmoid should fix it
    X, y = make_data(4000, 9, signal=3.0)
    Xt, yt = make_data(6000, 10, signal=3.0)
    cm = M.fit_calibrated("rf", X, y)
    raw, cal = cm.predict_raw(Xt), cm.predict_up(Xt)
    assert log_loss(yt, cal) < log_loss(yt, raw)
    assert brier_score_loss(yt, cal) < brier_score_loss(yt, raw)


def test_ensemble_is_mean_of_members():
    X, y = make_data(3000, 11)
    ens = M.EnsembleModel.fit(X, y, names=("logreg", "rf"))
    Xt = make_data(300, 12)[0]
    manual = np.mean([m.predict_up(Xt) for m in ens.members.values()], axis=0)
    np.testing.assert_allclose(ens.predict_up(Xt), manual)
    assert set(ens.settings) == {"logreg", "rf"}


@needs_lgbm
def test_ensemble_default_has_three_members():
    X, y = make_data(3000, 11)
    ens = M.EnsembleModel.fit(X, y)
    assert set(ens.members) == set(M.MODEL_NAMES)


@pytest.mark.parametrize("name", ["logreg", "rf"])
def test_determinism(name):
    X, y = make_data(3000, 13)
    Xt = make_data(200, 14)[0]
    a = M.fit_calibrated(name, X, y).predict_up(Xt)
    b = M.fit_calibrated(name, X, y).predict_up(Xt)
    np.testing.assert_array_equal(a, b)


def test_no_shuffled_cv_or_calibratedclassifiercv():
    src = pathlib.Path(M.__file__).read_text()
    # import / call level check on the real source
    import ast
    names = {n.id for n in ast.walk(ast.parse(src)) if isinstance(n, ast.Name)}
    names |= {n.attr for n in ast.walk(ast.parse(src)) if isinstance(n, ast.Attribute)}
    names |= {a.name for n in ast.walk(ast.parse(src)) if isinstance(n, ast.ImportFrom)
              for a in n.names}
    assert "CalibratedClassifierCV" not in names
    assert not {"KFold", "StratifiedKFold", "cross_val_score", "GridSearchCV", "train_test_split"} & names
    assert "FrozenEstimator" in names


# --------------------------------------------------------------------------- baselines
def test_climatology():
    assert M.climatology_prob([1, 0, 1, 1]) == 0.75
    with pytest.raises(ValueError):
        M.climatology_prob([])


def _prices(n=400, seed=0):
    rng = np.random.default_rng(seed)
    return pd.Series(100 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, n))),
                     index=pd.bdate_range("2020-01-01", periods=n))


def test_positions_use_only_past_data():
    close = _prices()
    ret = close.pct_change()
    full_t = M.trend_positions(close)
    full_v = M.vol_targeted_weights(ret, sigma_target=0.15)
    for cut in (250, 300, 399):
        d = close.index[cut]
        t = M.trend_positions(close.loc[:d])
        v = M.vol_targeted_weights(ret.loc[:d], sigma_target=0.15)
        pd.testing.assert_series_equal(t, full_t.loc[:d])
        pd.testing.assert_series_equal(v, full_v.loc[:d])
    assert (full_t.iloc[:199] == 0).all()


def test_trend_rule_definition():
    close = pd.Series(np.arange(1.0, 301.0), index=pd.bdate_range("2020-01-01", periods=300))
    pos = M.trend_positions(close)
    assert pos.iloc[250] == 1.0                     # rising -> above its MA
    assert M.trend_positions(close[::-1].set_axis(close.index)).iloc[250] == 0.0


def test_vol_targeted_weight_hand_computed():
    r = pd.Series([0.01, -0.01, 0.02, 0.0], index=pd.bdate_range("2020-01-01", periods=4))
    w = M.vol_targeted_weights(r, sigma_target=0.01, ann=1.0, window=3)
    assert w.iloc[:2].tolist() == [0.0, 0.0]        # warm-up
    sig = 0.015275252316519468                      # sd([.01,-.01,.02], ddof=1)
    assert w.iloc[2] == pytest.approx(0.01 / sig)
    sig2 = np.std([-0.01, 0.02, 0.0], ddof=1)
    assert w.iloc[3] == pytest.approx(0.01 / sig2)
    assert M.vol_targeted_weights(r, 1.0, 1.0, 3).iloc[2:].tolist() == [1.0, 1.0]  # capped


def test_annualisation_and_training_sigma():
    assert M.annualisation("BTC-USD") == 365 and M.annualisation("SPY") == 252
    r = np.array([0.01, -0.01, 0.02, 0.0])
    assert M.training_sigma(r, 252) == pytest.approx(np.std(r, ddof=1) * np.sqrt(252))


def test_position_applies_to_next_return():
    pos = pd.Series([1.0, 0.0, 0.5])
    nxt = pd.Series([0.1, 0.2, -0.1])
    assert M.apply_positions(pos, nxt).tolist() == [0.1, 0.0, -0.05]
    assert M.buy_and_hold_positions(range(3)).eq(1).all() and M.always_up_positions(range(3)).eq(1).all()


# --------------------------------------------------------------------------- importance
def test_group_permutation_importance():
    X, y = make_data(3000, 15, signal=2.0)
    cm = M.fit_calibrated("logreg", X, y)
    Xo, yo = make_data(3000, 16, signal=2.0)
    Xo.index = pd.bdate_range("2030-01-01", periods=len(Xo))   # genuinely out of sample
    yo.index = Xo.index
    imp = M.group_permutation_importance(cm, Xo, yo, n_repeats=5, seed=1).set_index("group")
    assert imp.loc["returns", "importance_mean"] > 0.05
    for g in ("volatility", "trend"):
        assert abs(imp.loc[g, "importance_mean"]) < 0.01
    assert imp.index[0] == "returns" and imp.loc["returns", "n_columns"] == 2
    again = M.group_permutation_importance(cm, Xo, yo, n_repeats=5, seed=1).set_index("group")
    pd.testing.assert_frame_equal(imp, again)
    auc = M.group_permutation_importance(cm, Xo, yo, n_repeats=3, metric="auc").set_index("group")
    assert auc.loc["returns", "importance_mean"] > 0.05


def test_importance_rejects_training_data():
    X, y = make_data(3000, 17)
    cm = M.fit_calibrated("logreg", X, y)
    with pytest.raises(ValueError, match="out of sample"):
        M.group_permutation_importance(cm, X, y)


def test_fit_calibrated_min_cal_configurable():
    """Rolling 4y window (1461 rows -> 292 calibration rows) needs min_cal=250, not 500."""
    X, y = make_data(n=1461)
    with pytest.raises(ValueError, match="at least 500"):
        M.fit_calibrated("logreg", X, y)
    cm = M.fit_calibrated("logreg", X, y, min_cal=250)
    assert cm.cal_range[1] - cm.cal_range[0] == 292
    p = cm.predict_up(X.iloc[-50:])
    assert ((p > 0) & (p < 1)).all()


def test_platt_calibrator_is_fit_on_calibration_slice_only():
    """Independently refit the sigmoid from the frozen base model's probabilities on the
    calibration slice [c0:c1]; it must equal cm.calibrator and differ from a fit on the fit slice."""
    from sklearn.linear_model import LogisticRegression
    X, y = make_data(n=900, seed=5)
    cm = M.fit_calibrated("logreg", X, y, min_cal=50)
    (f0, f1), (c0, c1) = cm.fit_range, cm.cal_range
    assert f1 < c0 and c0 - f1 == cm.gap          # fit slice, then the gap, then calibration

    def platt(a, b):
        p = np.clip(cm.frozen_base.predict_proba(X.iloc[a:b])[:, 1], M.PROB_EPS, 1 - M.PROB_EPS)
        return LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000).fit(
            np.log(p / (1 - p)).reshape(-1, 1), y.to_numpy()[a:b].astype(int))

    ref = platt(c0, c1)
    np.testing.assert_allclose(cm.calibrator.coef_, ref.coef_, rtol=1e-6)
    np.testing.assert_allclose(cm.calibrator.intercept_, ref.intercept_, rtol=1e-6)
    wrong = platt(0, c1 - c0)                      # a calibrator fit on (part of) the fit slice
    assert not np.allclose(cm.calibrator.coef_, wrong.coef_, rtol=1e-3)
