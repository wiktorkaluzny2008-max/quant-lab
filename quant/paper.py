"""Stage 4 paper-trading ledger logic (FAKE money; this module never talks to a broker).

One ledger row = one asset at the CLOSE of one decision day ``D``:

* the calibrated probability ``p_up`` for D -> next day, the training ``base_rate``, the margin
  ``m`` and the INTENDED position for the next day (``position``), with its trade and cost;
* the mark-to-market of the position decided at the PREVIOUS close (``prev_position``) using the
  realised return D-1 -> D (``asset_return``), costs charged on position changes, the equity
  and P&L after that;
* parallel buy-and-hold and vol-targeted buy-and-hold equity, and the running LIVE Brier score,
  Brier skill (vs climatology = the training base rate) and hit rate next to the backtest's
  values for the same configuration.

Rules that make it safe (each one has tests):

* NO LOOKAHEAD.  The decision for D uses only dataset rows <= D.  Training uses rows
  ``<= pos(D) - 2`` (1-day gap, exactly like the walk-forward), Platt calibration on the last
  20% of that window (``models.fit_calibrated``), vol targeting through D, adaptive margin from
  days strictly before D.
* CATCH-UP.  Every missed trading day (every calendar day for BTC) is processed in order with
  the information of that day; the result equals running day by day.  The model is retrained on
  the first processed day of each calendar month (the same monthly blocks as the walk-forward),
  so the model used on a day does not depend on when the program was run.
* IDEMPOTENT.  Nothing is written when there is no new day; no wall-clock time is stored.
* HOLD-OUT.  The ledger starts at the first as-of date (never inside the pinned 24-month
  hold-out period); live metrics use realised dates strictly after the hold-out end; backtest
  reference numbers use predictions on or before the pinned cutoff only (the Stage 3 results
  contain nothing later).  The live MODEL is trained on all rows available at the time, which
  includes the hold-out calendar period, for FUTURE predictions only.
* Paper trading is unleveraged: 0 <= position <= 1.
"""
from __future__ import annotations

import copy
import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from quant import backtest as B
from quant import features as F
from quant import metrics as M
from quant import models as MD
from quant import trials as T

LEDGER_COLS = [
    "asset", "date", "config_id", "strategy_mode",
    "p_up", "base_rate", "margin_m", "vol_ratio",
    "position", "prev_position", "trade", "cost_rate", "cost", "cost_usd",
    "asset_return", "gross_return", "net_return", "equity", "pnl", "cum_pnl",
    "n_trades", "cum_cost_usd",
    "bh_position", "bh_equity", "vtbh_position", "vtbh_equity",
    "outcome_up", "live_n", "live_brier", "live_brier_skill", "live_hit_rate",
    "bt_brier", "bt_brier_skill", "bt_hit_rate", "bt_n",
]
STATE_VERSION = 1
DEFAULT_SIZED_SCALE = 0.02
MIN_SCALE_POINTS = 50
BT = B.BTConfig()        # min_train_days=2500, min_cal=500, vol_window=21, adaptive_min_days=500
TRADE_TOL = 1e-12
MAX_RATIO = 1e6         # sigma_target/sigma_hat is capped here so a flat price series stays finite


def _jf(x):
    """NaN -> None for the JSON state file."""
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else float(x)


def _nf(x):
    return float("nan") if x is None else float(x)


class PaperError(Exception):
    """A real error (bad config, corrupt ledger, hold-out violation): the run exits non-zero."""


class HoldoutError(PaperError):
    """Something would touch the hold-out period for a metric or a retroactive trade."""


class SkipAsset(Exception):
    """Stale/missing/insufficient data for one asset: say so clearly and skip its new trade."""


# =========================================================================== settings
@dataclass(frozen=True)
class PaperSettings:
    strategy_mode: str
    strategy: dict
    sized_lambda: float
    sized_band: float
    capital: dict
    costs: dict

    @classmethod
    def from_config(cls, cfg) -> "PaperSettings":
        p = cfg["paper_trading"]
        s = dict(p["strategy"])
        if s["window"] != "expanding":
            raise PaperError("paper trading supports only the 'expanding' window "
                             f"(leverage_strategy.window is {s['window']!r})")
        if s["model"] != "ensemble" and s["model"] not in MD.MODEL_NAMES:
            raise PaperError(f"unknown model {s['model']!r} in leverage_strategy")
        if s["mode"] != "sized" and s["m_setting"] != T.ADAPTIVE:
            try:
                float(s["m_setting"])
            except ValueError as e:
                raise PaperError(f"leverage_strategy.m_setting must be 'adaptive' or a number; "
                                 f"got {s['m_setting']!r}") from e
        return cls(p["strategy_mode"], s, float(p["sized_lambda"]), float(p["sized_band"]),
                   dict(p["starting_capital"]), dict(cfg["costs"]))

    @property
    def model_names(self) -> list:
        return list(MD.MODEL_NAMES) if self.strategy["model"] == "ensemble" else [self.strategy["model"]]

    @property
    def vol_targeted(self) -> bool:
        return bool(self.strategy["vol_targeted"])

    def config_id(self, ticker) -> str:
        s = self.strategy
        if self.strategy_mode == "sized":
            vt = "vt" if self.vol_targeted else "raw"
            return (f"{ticker}|{s['window']}|{s['model']}|sized|{vt}|"
                    f"lambda={self.sized_lambda:g},band={self.sized_band:g}")
        m = T.ADAPTIVE if s["m_setting"] == T.ADAPTIVE else float(s["m_setting"])
        return T.make_config_id(ticker, s["window"], s["model"], self.strategy_mode,
                                self.vol_targeted, m)

    def snapshot(self) -> dict:
        return {"strategy_mode": self.strategy_mode, "strategy": self.strategy,
                "sized_lambda": self.sized_lambda, "sized_band": self.sized_band,
                "capital": self.capital, "costs": self.costs}


# =========================================================================== hold-out guards
def holdout_end(cfg) -> pd.Timestamp:
    """Last day of the hold-out period = pinned cutoff + holdout_months."""
    return pd.Timestamp(cfg["holdout_cutoff_date"]) + pd.DateOffset(months=int(cfg["holdout_months"]))


def assert_live_dates(dates, start, hold_end, what="live metric"):
    """Every date a LIVE metric is computed over must be strictly after the hold-out end and on
    or after the ledger start."""
    d = pd.DatetimeIndex(dates)
    if len(d) == 0:
        return
    if d.min() <= hold_end:
        raise HoldoutError(f"{what}: date {d.min().date()} is inside the hold-out period "
                           f"(ends {hold_end.date()}); refusing to compute a metric over it")
    if start is not None and d.min() < pd.Timestamp(start):
        raise HoldoutError(f"{what}: date {d.min().date()} is before the ledger start "
                           f"{pd.Timestamp(start).date()}; no retroactive metrics")


def assert_dev_dates(dates, cutoff, what="backtest reference"):
    """Backtest reference numbers may only use dates on or before the pinned cutoff."""
    d = pd.DatetimeIndex(dates)
    if len(d) and d.max() > pd.Timestamp(cutoff):
        raise HoldoutError(f"{what}: date {d.max().date()} is after the pinned hold-out cutoff "
                           f"{pd.Timestamp(cutoff).date()}")


# =========================================================================== backtest reference
@dataclass
class BacktestHistory:
    """Stage 3 out-of-sample days of the chosen model/window (dev period only)."""
    dates: pd.DatetimeIndex
    p: np.ndarray
    base: np.ndarray
    nr: np.ndarray
    w: np.ndarray            # vol-targeting weights min(1, sigma_target/sigma_hat) of those days
    y: np.ndarray
    reference: dict          # bt_brier, bt_brier_skill, bt_hit_rate, bt_n


def verify_trial_log(config_id, results_dir):
    """The paper config must be a logged Stage 3 trial (canonical id or alias)."""
    idx = T.load_index(results_dir)
    if len(idx) == 0:
        raise PaperError(f"Stage 3 trial log not found in {results_dir} "
                         "(run `python -m quant.backtest` first)")
    if config_id not in T.all_logged_config_ids(idx):
        raise PaperError(f"config ID {config_id!r} is not in the Stage 3 trial log; paper trading "
                         "only reuses an already-logged configuration")


def load_backtest_history(ticker, settings: PaperSettings, results_dir, cutoff) -> BacktestHistory:
    rd = Path(results_dir) / ticker
    s = settings.strategy
    try:
        pred = pd.read_parquet(rd / "predictions.parquet")
        base = pd.read_parquet(rd / "baselines.parquet")
    except (FileNotFoundError, OSError) as e:
        raise PaperError(f"{ticker}: Stage 3 results not found in {rd} "
                         "(run `python -m quant.backtest` first)") from e
    pred = pred[(pred["window"] == s["window"]) & (pred["model"] == s["model"])]
    pred = pred.sort_values("date").reset_index(drop=True)
    assert_dev_dates(pred["date"], cutoff)
    w = base[(base["window"] == s["window"]) & (base["name"] == "vol_targeted_bh")]
    w = w.sort_values("date").reset_index(drop=True)
    if len(pred) == 0 or not np.array_equal(pred["date"].to_numpy(), w["date"].to_numpy()):
        raise PaperError(f"{ticker}: Stage 3 predictions/baselines for {s['window']}/{s['model']} "
                         "are missing or misaligned")
    dates = pd.DatetimeIndex(pred["date"]).astype("datetime64[ns]")
    assert_dev_dates(dates, cutoff)
    p, b = pred["p_up"].to_numpy(float), pred["base_rate"].to_numpy(float)
    y, nr = pred["y"].to_numpy(float), pred["next_return"].to_numpy(float)
    ref = dict(bt_brier=M.brier(y, p), bt_brier_skill=M.brier_skill(y, p, b),
               bt_hit_rate=float(np.mean((p >= b) == (y > 0.5))), bt_n=int(len(y)))
    return BacktestHistory(dates, p, b, nr, w["position"].to_numpy(float), y, ref)


# =========================================================================== model / decision
def month_key(ts) -> int:
    ts = pd.Timestamp(ts)
    return ts.year * 12 + ts.month


def _annual(ticker) -> float:
    return MD.annualisation(ticker)


def fit_live(ds: pd.DataFrame, i: int, settings: PaperSettings, ticker):
    """Fit the live model for decision position ``i`` using rows 0..i-2 only (1-day gap).
    Returns (model, info)."""
    hi = i - 1
    y = ds["target"].to_numpy(float)
    nr = ds["next_return"].to_numpy(float)
    keep = ~np.isnan(y[:hi])
    if hi < 1 or keep.sum() < BT.min_train_days:
        raise SkipAsset(f"only {int(keep.sum()) if hi >= 1 else 0} labelled training rows are "
                        f"available; at least {BT.min_train_days} are needed to train")
    feats = F.feature_columns(ds)
    Xtr = ds[feats].iloc[:hi][keep]
    ytr = y[:hi][keep]
    try:
        members = {m: MD.fit_calibrated(m, Xtr, ytr, min_cal=BT.min_cal)
                   for m in settings.model_names}
    except ValueError as e:
        raise SkipAsset(f"model could not be trained ({e})") from e
    model = (MD.EnsembleModel(members) if settings.strategy["model"] == "ensemble"
             else members[settings.strategy["model"]])
    info = dict(n_train=int(len(ytr)), train_last_row=str(ds.index[hi - 1].date()),
                base_rate=MD.climatology_prob(ytr),
                sigma_target=MD.training_sigma(nr[:hi][keep], _annual(ticker)))
    return model, info


def vol_ratio_at(ds: pd.DataFrame, i: int, sigma_target: float, ann: float) -> float:
    """sigma_target / sigma_hat_D (UNCAPPED) with sigma_hat = 21-day realised vol through D.
    Reuses ``models.vol_targeted_weights`` (0 during warm-up) on rows <= D only."""
    realised = B.realised_returns(ds.iloc[: i + 1])
    r = MD.vol_targeted_weights(realised, sigma_target, ann, BT.vol_window, max_weight=np.inf)
    return float(min(r.iloc[-1], MAX_RATIO))


def _history_arrays(bt: BacktestHistory, live: list, r_last, today):
    """Past out-of-sample days strictly before the current day D: the Stage 3 OOS days, then the
    live decision days (each one's realised next-day return is the next row's asset_return; the
    last live row's is today's return ``r_last``).  Returns p, base, nr, w and the REALISED dates
    of the live part (for the hold-out guard)."""
    p, base, w, nr = [bt.p], [bt.base], [bt.w], [bt.nr]
    live_rd = pd.DatetimeIndex([])
    if live:
        p.append(np.array([r["p_up"] for r in live], float))
        base.append(np.array([r["base_rate"] for r in live], float))
        w.append(np.minimum(1.0, np.array([r["vol_ratio"] for r in live], float)))
        nr.append(np.array([r["asset_return"] for r in live[1:]] + [r_last], float))
        live_rd = pd.DatetimeIndex([pd.Timestamp(r["date"]) for r in live[1:]] + [today])
    return np.concatenate(p), np.concatenate(base), np.concatenate(nr), np.concatenate(w), live_rd


def choose_margin(settings: PaperSettings, p, base, nr, w, cost):
    """(m, used_fallback).  Adaptive m is chosen by ``backtest.adaptive_positions`` (same code as
    Stage 3) from days strictly before today only: today's block is a dummy row appended after
    the history, so nothing at or after today can influence the choice."""
    s = settings.strategy
    mode = settings.strategy_mode
    if mode == "sized":
        return float("nan"), False
    if s["m_setting"] != T.ADAPTIVE:
        return float(s["m_setting"]), False
    n = len(p)
    if n == 0:
        return float(T.M_FALLBACK[mode]), True
    vt = settings.vol_targeted
    P, Bs = np.r_[p, 0.5], np.r_[base, 0.5]
    NR = np.r_[nr, 0.0]
    W = np.r_[w, 1.0] if vt else None
    _, log = B.adaptive_positions(mode, P, Bs, NR, W, [0, n], cost, BT.adaptive_min_days)
    return float(log[-1]["m"]), bool(log[-1]["used_fallback"])


def sized_scale(p, base) -> float:
    """scale in position = clip(lambda*(p-base)/scale*...): std of (p - base) over past OOS days."""
    d = np.asarray(p, float) - np.asarray(base, float)
    if len(d) < MIN_SCALE_POINTS:
        return DEFAULT_SIZED_SCALE
    sd = float(np.std(d, ddof=1))
    return sd if sd > 1e-9 else DEFAULT_SIZED_SCALE


def decide_position(settings: PaperSettings, p, base, m, ratio, scale, prev_pos) -> float:
    """Intended position (0..1) for the next day, from information at today's close."""
    vt = settings.vol_targeted
    if settings.strategy_mode == "sized":
        k = ratio if vt else 1.0
        target = float(np.clip(settings.sized_lambda * (p - base) / scale * k, 0.0, 1.0))
        return prev_pos if abs(target - prev_pos) < settings.sized_band else target
    w = min(1.0, ratio) if vt else 1.0
    sig = float(B.SIGNALS[settings.strategy_mode](np.array([p]), np.array([base]), m)[0])
    return float(np.clip(sig * w, 0.0, 1.0))


def running_live_metrics(p, base, y):
    """Cumulative live metrics for aligned arrays (prediction made the day before each outcome).
    Returns n, brier, brier skill vs climatology (the base rate at prediction time), hit rate."""
    p, base, y = (np.asarray(a, float) for a in (p, base, y))
    n = np.arange(1, len(p) + 1, dtype=float)
    sq = np.cumsum((p - y) ** 2)
    clim = np.cumsum((base - y) ** 2)
    hit = np.cumsum(((p >= base) == (y > 0.5)).astype(float))
    with np.errstate(divide="ignore", invalid="ignore"):
        skill = np.where(clim > 0, 1.0 - sq / clim, np.nan)
    return n, sq / n, skill, hit / n


# =========================================================================== per-asset engine
@dataclass
class AssetResult:
    rows: pd.DataFrame
    state: dict
    model: object = None          # set when a (re)fit happened and the file must be saved
    messages: list = field(default_factory=list)
    n_retrains: int = 0


def _row_dict(r) -> dict:
    return r if isinstance(r, dict) else r.to_dict()


def process_asset(ticker, ds: pd.DataFrame, settings: PaperSettings, ledger_a: pd.DataFrame,
                  state_a, bt: BacktestHistory, hold_end, cutoff, model_loader=None,
                  max_date=None) -> AssetResult:
    """Process every not-yet-recorded decision day of ``ds`` (dataset rows, sorted by date).

    ``ds`` must already be truncated to the data usable at the as-of date.  ``ledger_a`` is this
    asset's existing ledger (may be empty); ``state_a`` its state dict (None when new).
    ``model_loader()`` returns the persisted live model of ``state_a`` (None -> refit)."""
    if not ds.index.is_monotonic_increasing or ds.index.has_duplicates:
        raise PaperError(f"{ticker}: dataset index must be strictly increasing dates")
    dates = ds.index
    if max_date is not None:
        dates_ok = dates <= pd.Timestamp(max_date)
        ds = ds[dates_ok]
        dates = ds.index
    if len(ds) == 0:
        raise SkipAsset("no data rows")
    ann = _annual(ticker)
    cost_rate = float(settings.costs[ticker])
    cap = float(settings.capital[ticker])
    config_id = settings.config_id(ticker)
    feats = F.feature_columns(ds)
    nr_all = ds["next_return"].to_numpy(float)

    prior = [_row_dict(r) for r in ledger_a.sort_values("date").to_dict("records")]
    if (state_a is None) != (len(prior) == 0):
        raise PaperError(f"{ticker}: state file and ledger disagree (one is empty, the other is "
                         "not); restore data/paper from a backup or delete it to restart")
    if prior:
        last = pd.Timestamp(prior[-1]["date"])
        if last > dates[-1]:       # as-of earlier than the ledger end: nothing to do, never rewrite
            return AssetResult(rows=pd.DataFrame(columns=LEDGER_COLS), state=copy.deepcopy(state_a))
        if last not in dates:
            raise PaperError(f"{ticker}: last ledger date {last.date()} is not in the dataset")
        todo = list(range(dates.get_loc(last) + 1, len(dates)))
    else:
        todo = [len(dates) - 1]
        first = dates[-1]
        if first < hold_end:
            raise HoldoutError(
                f"{ticker}: the first live decision date {first.date()} is inside the hold-out "
                f"period (ends {hold_end.date()}); the live ledger may only start on or after "
                "that day (no retroactive paper trades)")
    out = AssetResult(rows=pd.DataFrame(columns=LEDGER_COLS), state=copy.deepcopy(state_a) if state_a else {})
    if not todo:
        return out
    state = out.state
    model = None
    new = []
    live = list(prior)
    prev = live[-1] if live else None

    for i in todo:
        D = dates[i]
        if prev is not None:
            if pd.Timestamp(prev["date"]) != dates[i - 1]:
                raise PaperError(f"{ticker}: ledger row {pd.Timestamp(prev['date']).date()} is not "
                                 f"the day before {D.date()} in the data (gap)")
            r = float(nr_all[i - 1])
            if not np.isfinite(r):
                raise SkipAsset(f"no realised return for {D.date()}")
        else:
            r = float("nan")
        # ---- retrain on the first processed day of a new calendar month
        if not state or state["train_key"] != month_key(D):
            model, info = fit_live(ds, i, settings, ticker)
            p_h, b_h, nr_h, w_h, live_rd = _history_arrays(bt, live, r, D)
            assert_dev_dates(bt.dates, cutoff, "adaptive-margin history (backtest part)")
            assert_live_dates(live_rd, None, hold_end, "adaptive-margin history (live part)")
            m, fb = choose_margin(settings, p_h, b_h, nr_h, w_h, cost_rate)
            scale = sized_scale(p_h, b_h) if settings.strategy_mode == "sized" else float("nan")
            state.update(train_key=month_key(D), train_date=str(D.date()), margin_m=_jf(m),
                         m_used_fallback=fb, sized_scale=_jf(scale), **info)
            state.setdefault("first_date", str(D.date()))
            state.setdefault("config_id", config_id)
            state.setdefault("backtest_reference", dict(bt.reference))
            out.model, out.n_retrains = model, out.n_retrains + 1
        elif model is None:
            model = model_loader() if model_loader else None
            if model is None:     # persisted model missing/unreadable: refit deterministically
                t0 = dates.get_loc(pd.Timestamp(state["train_date"]))
                model, _ = fit_live(ds, t0, settings, ticker)
                out.model = model
        # ---- today's prediction and intended position (information at close D only)
        p = float(model.predict_up(ds[feats].iloc[[i]])[0])
        base = float(state["base_rate"])
        ratio = vol_ratio_at(ds, i, float(state["sigma_target"]), ann)
        m = _nf(state["margin_m"])
        prev_pos = float(prev["position"]) if prev is not None else 0.0
        pos = decide_position(settings, p, base, m, ratio, _nf(state["sized_scale"]), prev_pos)
        # ---- mark yesterday's position to market, charge costs on the position change
        ret = r if prev is not None else float("nan")
        gross = (prev_pos * ret + 0.0) if prev is not None else 0.0   # + 0.0: no -0.0
        trade = abs(pos - prev_pos)
        cost = cost_rate * trade
        net = gross - cost
        prev_eq = float(prev["equity"]) if prev is not None else cap
        eq = prev_eq * (1.0 + net)
        w_cap = min(1.0, ratio)
        bh_prev = float(prev["bh_position"]) if prev is not None else 0.0
        vt_prev = float(prev["vtbh_position"]) if prev is not None else 0.0
        bh_eq0 = float(prev["bh_equity"]) if prev is not None else cap
        vt_eq0 = float(prev["vtbh_equity"]) if prev is not None else cap
        g_bh = bh_prev * ret if prev is not None else 0.0
        g_vt = vt_prev * ret if prev is not None else 0.0
        row = dict(
            asset=ticker, date=D, config_id=config_id, strategy_mode=settings.strategy_mode,
            p_up=p, base_rate=base, margin_m=m, vol_ratio=ratio,
            position=pos, prev_position=prev_pos, trade=trade, cost_rate=cost_rate, cost=cost,
            cost_usd=cost * prev_eq,
            asset_return=ret, gross_return=gross, net_return=net, equity=eq, pnl=eq - prev_eq,
            cum_pnl=eq - cap,
            n_trades=int(prev["n_trades"] if prev is not None else 0) + int(trade > TRADE_TOL),
            cum_cost_usd=float(prev["cum_cost_usd"] if prev is not None else 0.0) + cost * prev_eq,
            bh_position=1.0, bh_equity=bh_eq0 * (1.0 + g_bh - cost_rate * abs(1.0 - bh_prev)),
            vtbh_position=w_cap,
            vtbh_equity=vt_eq0 * (1.0 + g_vt - cost_rate * abs(w_cap - vt_prev)),
            outcome_up=(float(ret > 0) if prev is not None else float("nan")),
            live_n=0, live_brier=float("nan"), live_brier_skill=float("nan"),
            live_hit_rate=float("nan"),
            **state["backtest_reference"])
        new.append(row)
        live.append(row)
        prev = row

    df = pd.DataFrame(new, columns=LEDGER_COLS)
    _fill_live_metrics(df, prior, hold_end, start=(pd.Timestamp(prior[0]["date"]) if prior
                                                  else pd.Timestamp(df["date"].iloc[0])))
    out.rows = df
    out.state = state
    return out


def _fill_live_metrics(df: pd.DataFrame, prior: list, hold_end, start):
    """Running live Brier / skill / hit rate for the NEW rows (the prediction of row j-1 against
    the outcome realised at row j), over live dates only."""
    allr = df if not prior else pd.concat([pd.DataFrame(prior, columns=LEDGER_COLS), df],
                                          ignore_index=True)
    dts = pd.to_datetime(allr["date"])
    p = allr["p_up"].shift(1).to_numpy(float)
    b = allr["base_rate"].shift(1).to_numpy(float)
    y = allr["outcome_up"].to_numpy(float)
    ok = np.isfinite(y) & np.isfinite(p)
    assert_live_dates(dts[ok], start, hold_end, "live Brier/hit rate")
    n, br, sk, hr = (np.zeros(len(allr)), np.full(len(allr), np.nan),
                     np.full(len(allr), np.nan), np.full(len(allr), np.nan))
    if ok.any():
        n[ok], br[ok], sk[ok], hr[ok] = running_live_metrics(p[ok], b[ok], y[ok])
    k = len(prior)
    df["live_n"] = n[k:].astype("int64")
    df["live_brier"], df["live_brier_skill"], df["live_hit_rate"] = br[k:], sk[k:], hr[k:]


# =========================================================================== files
@dataclass(frozen=True)
class PaperFiles:
    root: Path

    @classmethod
    def for_config(cls, cfg):
        return cls(Path(cfg["data_dir"]) / "paper")

    @property
    def ledger(self) -> Path:
        return self.root / "ledger.parquet"

    @property
    def state(self) -> Path:
        return self.root / "state.json"

    def model(self, ticker) -> Path:
        return self.root / "models" / f"{ticker}.joblib"


def _replace_atomic(write, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    try:
        write(tmp)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def _coerce(df: pd.DataFrame) -> pd.DataFrame:
    df = df[LEDGER_COLS].copy()
    df["date"] = pd.to_datetime(df["date"]).astype("datetime64[ns]")
    for c in ("asset", "config_id", "strategy_mode"):
        df[c] = df[c].astype(str)
    for c in ("live_n", "n_trades", "bt_n"):
        df[c] = df[c].astype("int64")
    for c in LEDGER_COLS:
        if c not in ("asset", "date", "config_id", "strategy_mode", "live_n", "n_trades", "bt_n"):
            df[c] = df[c].astype("float64")
    return df


def load_ledger(files: PaperFiles) -> pd.DataFrame:
    if not files.ledger.exists():
        return _coerce(pd.DataFrame({c: [] for c in LEDGER_COLS}))
    try:
        df = pd.read_parquet(files.ledger)
    except Exception as e:
        raise PaperError(f"ledger {files.ledger} could not be read ({e})") from e
    if list(df.columns) != LEDGER_COLS:
        raise PaperError(f"ledger {files.ledger} has unexpected columns; expected {LEDGER_COLS}")
    if df.duplicated(["asset", "date"]).any():
        raise PaperError(f"ledger {files.ledger} has duplicate (asset, date) rows")
    return _coerce(df)


def save_ledger(files: PaperFiles, df: pd.DataFrame):
    df = _coerce(df).sort_values(["asset", "date"], kind="mergesort").reset_index(drop=True)
    _replace_atomic(lambda p: df.to_parquet(p, index=False), files.ledger)


def load_state(files: PaperFiles) -> dict:
    if not files.state.exists():
        return {}
    try:
        st = json.loads(files.state.read_text())
    except (OSError, ValueError) as e:
        raise PaperError(f"state file {files.state} could not be read ({e})") from e
    if st.get("version") != STATE_VERSION:
        raise PaperError(f"state file {files.state} has an unsupported version")
    return st


def save_state(files: PaperFiles, state: dict):
    text = json.dumps(state, indent=2, sort_keys=True) + "\n"
    _replace_atomic(lambda p: p.write_text(text), files.state)


def save_model(files: PaperFiles, ticker, model):
    import joblib
    _replace_atomic(lambda p: joblib.dump(model, p), files.model(ticker))


def load_model(files: PaperFiles, ticker):
    """The persisted live model, or None when missing/unreadable (caller refits)."""
    import joblib
    try:
        return joblib.load(files.model(ticker))
    except Exception:
        return None
