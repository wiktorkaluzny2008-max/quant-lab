"""Stage 3 walk-forward backtest, baselines, strategies, costs, significance and result files.

Run:  ``python -m quant.backtest [--assets SPY QQQ ...] [--models logreg rf lgbm]``

HOLD-OUT SAFETY
---------------
The last ``holdout_months`` (24) of every dataset are cut off by ``split_holdout`` the moment
the dataset is loaded (``load_dev``) and are never touched again: nothing here saves, scores,
fits on or prints them.  Every run function takes only the DEV frame plus the cutoff date and
raises if the frame contains a later date.  Because the label of the last dev row (its
``next_return``) is a price move that ends INSIDE the hold-out, ``split_holdout`` blanks that
one label (the row stays, its target/next_return become NaN and the row is never scored).

WALK-FORWARD (expanding window; BTC also a rolling 4-year window)
-----------------------------------------------------------------
* Retrain at the first row of each calendar month; the frozen model predicts every row of that
  month.
* Row j has features known at close j and a target realised at close j+1 (row j+1).  For a block
  whose first row has position s (the first prediction date t), training uses rows
  ``j <= s-2`` ONLY.  Their targets are realised at row j+1 <= s-1, i.e. before t, and row
  s-1 (whose target is realised exactly at t) is left out: that is the 1-day gap.
  Rows with NaN target are dropped.
* Each model is fit by ``models.fit_calibrated`` (80% fit / 1-day gap / 20% Platt calibration)
  and the equal-weight ``ensemble`` averages the calibrated members that were requested.
* Min initial training window ``min_train_days`` rows (2500 so that the 20% calibration slice
  has the spec's 500 days).  Rolling 4y variant: the last 1461 rows (before the gap);
  decision: ``min_cal=250`` for it (292 calibration rows at 20% of 1461 cannot reach 500).
* Recorded per row: date, model, calibrated p_up, training base rate (up-frequency of that
  block's whole training window = the climatology forecast), y, next_return.

CONVENTIONS
-----------
* Position decided at close t earns ``next_return[t]``.  Cost = cost * |position change|, the
  first entry from 0 is charged.  One-day delay variant: position decided at t is held from
  t+1 (earns next_return[t+1]).
* Both strategy modes use p >= base_rate - m => long; ``long_flat`` and ``default_long``
  differ only in the m grid and the adaptive fallback (spec).  Hence m=0.01/0.02 appear in
  both grids as identical return streams.  The code DEDUPLICATES identical streams (see
  ``trials.py``): each distinct stream is one trial, the others are listed as aliases
  (360 configurations -> 270 distinct trials in the full run).
* Vol targeting: w_t = min(1, sigma_target / sigma_hat_t), sigma_hat = 21-day realised vol
  through t, sigma_target = vol of the block's training window (past data only).
* HOLD-OUT is PINNED by the date ``holdout_cutoff_date`` in config.yaml (never re-derived from
  new data).  Appending newer rows to the dataset changes nothing.
* Adaptive m: at each block start, choose m from that mode's grid maximising net Sharpe
  (headline cost) of the fixed-m strategy over all OOS days STRICTLY BEFORE the block;
  fall back to M_FALLBACK until >= 500 such days exist.  Chosen m logged per block.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from quant import features as F
from quant import metrics as M
from quant import models as MD
from quant import trials as T

LOW_COST_ETF = 0.0001
STRESS_COST_BTC = 0.0060
SPA_REFERENCE = "buy_and_hold"
BASELINES = ("buy_and_hold", "always_up", "vol_targeted_bh", "trend_200")


@dataclass(frozen=True)
class BTConfig:
    holdout_months: int = 24
    min_train_days: int = 2500
    min_cal: int = 500
    rolling_window: int = 1461          # 4 years of calendar days (BTC trades daily)
    rolling_min_cal: int = 250
    vol_window: int = 21
    adaptive_min_days: int = 500
    n_random: int = 1000
    importance_repeats: int = 3
    spa_reps: int = 1000
    seed: int = 0


# =========================================================================== hold-out
def holdout_cutoff(df: pd.DataFrame, months: int) -> pd.Timestamp:
    return pd.Timestamp(df.index.max()) - pd.DateOffset(months=months)


def split_holdout(df: pd.DataFrame, months: int, cutoff=None):
    """(dev, holdout) split (dev: date <= cutoff).  ``cutoff`` is the PINNED date from config;
    without it the cutoff is ``last date - months`` (only used by tests / one-off pinning).

    The last dev row's label ends inside the hold-out, so its target/next_return are set to NaN
    (that day can never be scored or trained on)."""
    if not df.index.is_monotonic_increasing or df.index.has_duplicates:
        raise ValueError("dataset index must be strictly increasing dates")
    cutoff = holdout_cutoff(df, months) if cutoff is None else pd.Timestamp(cutoff)
    dev = df.loc[df.index <= cutoff].copy()
    hold = df.loc[df.index > cutoff].copy()
    if len(hold) and len(dev):
        for c in F.TARGET_COLS:
            dev.iloc[-1, dev.columns.get_loc(c)] = np.nan
    return dev, hold


def load_dev(ticker, cfg, bt: BTConfig | None = None):
    """Load the dataset and IMMEDIATELY discard the hold-out.  Returns (dev, cutoff)."""
    cutoff = pd.Timestamp(cfg["holdout_cutoff_date"])   # pinned in config.yaml; never changes
    df = F.load_dataset(ticker, cfg)
    dev, _ = split_holdout(df, int(cfg["holdout_months"]), cutoff)
    del df, _
    if len(dev) == 0:
        raise ValueError(f"{ticker}: no data on or before the pinned hold-out cutoff {cutoff.date()}")
    return dev, cutoff


def _assert_dev(dev: pd.DataFrame, cutoff):
    if cutoff is None:
        raise ValueError("cutoff date is required")
    if len(dev) == 0:
        raise ValueError("empty dev frame")
    if pd.Timestamp(dev.index.max()) > pd.Timestamp(cutoff):
        raise AssertionError(f"dev frame contains hold-out dates: max {dev.index.max()} > cutoff {cutoff}")


# =========================================================================== requirements
def require_models(models):
    bad = [m for m in models if m not in MD.MODEL_NAMES]
    if bad:
        raise ValueError(f"unknown model(s) {bad}; expected from {MD.MODEL_NAMES}")
    if "lgbm" in models:
        try:
            import lightgbm  # noqa: F401
        except Exception as e:   # OSError (libomp), ImportError, or anything else a broken install raises
            raise RuntimeError(
                "model 'lgbm' was requested but LightGBM cannot be imported "
                f"({type(e).__name__}: {str(e).splitlines()[0][:120]}). On macOS this usually "
                "means libomp is missing. Run `sh scripts/fix_lightgbm_macos.sh` or run with --models logreg rf.") from e


# =========================================================================== walk-forward
@dataclass
class WalkForward:
    window: str
    preds: pd.DataFrame        # long: window, model, date, block, p_up, p_raw (diagnostic only), base_rate, y, next_return
    retrains: pd.DataFrame     # one row per retrain
    importance: pd.DataFrame   # per block/model/group
    ensemble_members: list


def month_blocks(index: pd.DatetimeIndex):
    """(start, stop) row positions of each calendar-month block."""
    key = index.year * 12 + index.month
    starts = np.flatnonzero(np.r_[True, key[1:] != key[:-1]])
    stops = np.r_[starts[1:], len(index)]
    return list(zip(starts.tolist(), stops.tolist()))


def training_slice(s: int, window: str, bt: BTConfig):
    """Row positions [lo, hi) used to train the block starting at row s (1-day gap: hi = s-1,
    i.e. rows <= s-2).  Returns None when the rolling window does not fit."""
    hi = s - 1
    if hi < 1:          # no row at all before the gap (guards negative slice bounds)
        return None
    if window == "expanding":
        return 0, hi
    lo = hi - bt.rolling_window
    return (lo, hi) if lo >= 0 else None


def walk_forward(dev, models, bt: BTConfig, cutoff, window="expanding", ann=252.0,
                 importance=True, first_date=None) -> WalkForward:
    """Walk-forward over every calendar-month block of ``dev``.

    ``first_date`` (default None = all blocks, the Stage 3 behaviour) skips blocks that END before
    that date (used by the Stage 6 hold-out harness, quant/holdout.py, to predict only the
    blocks that contain hold-out days; every block it does predict is trained exactly as
    before: expanding window, rows <= s-2)."""
    _assert_dev(dev, cutoff)
    require_models(models)
    if window not in T.WINDOWS:
        raise ValueError(f"unknown window {window!r}")
    feats = F.feature_columns(dev)
    X = dev[feats]
    y = dev["target"].to_numpy(dtype="float64")
    nr = dev["next_return"].to_numpy(dtype="float64")
    dates = dev.index
    min_cal = bt.min_cal if window == "expanding" else bt.rolling_min_cal
    pred_rows, ret_rows, imp_rows = [], [], []
    use_ens = len(models) > 1
    names = [*models, *(["ensemble"] if use_ens else [])]
    for s, e in month_blocks(dates):
        if first_date is not None and dates[e - 1] < pd.Timestamp(first_date):
            continue
        sl = training_slice(s, window, bt)
        if sl is None:
            continue
        lo, hi = sl
        ytr = y[lo:hi]
        keep = ~np.isnan(ytr)
        if window == "expanding" and keep.sum() < bt.min_train_days:
            continue
        Xtr, ytr = X.iloc[lo:hi][keep], ytr[keep]
        mask = ~(np.isnan(y[s:e]) | np.isnan(nr[s:e]))
        if not mask.any():
            continue
        Xb = X.iloc[s:e][mask]
        yb, nrb, db = y[s:e][mask], nr[s:e][mask], dates[s:e][mask]
        base = MD.climatology_prob(ytr)
        members = {m: MD.fit_calibrated(m, Xtr, ytr, min_cal=min_cal) for m in models}
        fitted = dict(members)
        if use_ens:
            fitted["ensemble"] = MD.EnsembleModel(members)
        slopes = {m: (float(members[m].calibrator.coef_[0, 0]) if members[m].method == "sigmoid"
                      else float("nan")) for m in models}
        for name in names:
            p = fitted[name].predict_up(Xb)
            raw = (np.mean([members[m].predict_raw(Xb) for m in models], axis=0) if name == "ensemble"
                   else members[name].predict_raw(Xb))
            pred_rows.append(pd.DataFrame({
                "window": window, "model": name, "date": db, "block": dates[s], "p_up": p,
                "p_raw": raw,
                "base_rate": base, "y": yb, "next_return": nrb}))
            if importance and bt.importance_repeats > 0:
                imp = MD.group_permutation_importance(
                    fitted[name], Xb, yb, n_repeats=bt.importance_repeats, seed=bt.seed + s)
                imp = imp.assign(window=window, model=name, block=dates[s], n_rows=len(yb))
                imp_rows.append(imp)
        ret_rows.append(dict(
            window=window, block=dates[s], train_start=dates[lo], train_end=dates[hi - 1],
            n_train=int(len(ytr)), base_rate=base,
            sigma_target=MD.training_sigma(nr[lo:hi][keep], ann), n_oos=int(len(yb)),
            **{f"platt_slope_{m}": v for m, v in slopes.items()}))
    if not pred_rows:
        raise ValueError(f"no walk-forward block could be trained (window={window}, "
                         f"{len(dev)} dev rows, min_train_days={bt.min_train_days})")
    preds = pd.concat(pred_rows, ignore_index=True)
    imp_df = (pd.concat(imp_rows, ignore_index=True) if imp_rows else
              pd.DataFrame(columns=["group", "n_columns", "importance_mean", "importance_std",
                                    "window", "model", "block", "n_rows"]))
    return WalkForward(window, preds, pd.DataFrame(ret_rows), imp_df,
                       list(models) if use_ens else [])


def aggregate_importance(blocks: pd.DataFrame) -> pd.DataFrame:
    """Combine per-block out-of-sample importances: row-weighted mean over blocks, plus the
    standard error across blocks."""
    cols = ["window", "model", "group", "n_columns", "importance_mean", "importance_se",
            "n_blocks", "n_rows"]
    if blocks.empty:
        return pd.DataFrame(columns=cols)
    out = []
    for (w, m, g), d in blocks.groupby(["window", "model", "group"], sort=True):
        wts = d["n_rows"].to_numpy(float)
        v = d["importance_mean"].to_numpy(float)
        se = float(np.std(v, ddof=1) / np.sqrt(len(v))) if len(v) > 1 else float("nan")
        out.append(dict(window=w, model=m, group=g, n_columns=int(d["n_columns"].iloc[0]),
                        importance_mean=float(np.average(v, weights=wts)), importance_se=se,
                        n_blocks=int(len(d)), n_rows=int(wts.sum())))
    return pd.DataFrame(out, columns=cols)


# =========================================================================== baselines
def rebased_close(dev: pd.DataFrame) -> pd.Series:
    """Price level rebuilt from next_return (scale-free; enough for a moving-average rule).
    close[0]=1, close[j+1]=close[j]*(1+next_return[j])."""
    nr = dev["next_return"].to_numpy(dtype="float64")
    growth = np.r_[1.0, np.cumprod(1.0 + np.nan_to_num(nr[:-1], nan=0.0))]
    return pd.Series(growth, index=dev.index, name="close")


def realised_returns(dev: pd.DataFrame) -> pd.Series:
    """Return earned over (t-1 -> t): known at close t.  First row NaN."""
    return dev["next_return"].shift(1).rename("realised")


def vol_weights(dev, retrains: pd.DataFrame, oos_dates, ann, vol_window) -> pd.Series:
    """Vol-targeting weights on the OOS dates.  Block b uses its own sigma_target (training
    window vol, data before the block) and sigma_hat through the day itself."""
    realised = realised_returns(dev)
    w = pd.Series(np.nan, index=oos_dates, dtype="float64", name="vol_targeted")
    oos_dates = pd.DatetimeIndex(oos_dates)
    ends = list(retrains["block"].iloc[1:]) + [pd.Timestamp.max]
    for (_, r), end in zip(retrains.iterrows(), ends):
        sel = oos_dates[(oos_dates >= r["block"]) & (oos_dates < end)]
        if len(sel) == 0:
            continue
        wts = MD.vol_targeted_weights(realised.loc[:sel[-1]], float(r["sigma_target"]), ann,
                                      int(vol_window))
        w.loc[sel] = wts.loc[sel].to_numpy()
    return w


def baseline_positions(dev, retrains, oos_dates, ann, bt: BTConfig) -> dict:
    idx = pd.DatetimeIndex(oos_dates)
    trend = MD.trend_positions(rebased_close(dev), 200).reindex(idx)
    return {
        "buy_and_hold": MD.buy_and_hold_positions(idx),
        "always_up": MD.always_up_positions(idx),   # identical to buy and hold by construction
        "vol_targeted_bh": vol_weights(dev, retrains, idx, ann, bt.vol_window),
        "trend_200": trend,
    }


# =========================================================================== strategies
def long_flat_signal(p, base, m):
    """Long (1) when p >= base_rate - m, else flat (0)."""
    return (np.asarray(p) >= np.asarray(base) - m).astype("float64")


def default_long_signal(p, base, m):
    """100% long unless p < base_rate - m (clearly below the base rate), then flat."""
    return 1.0 - (np.asarray(p) < np.asarray(base) - m).astype("float64")


SIGNALS = {"long_flat": long_flat_signal, "default_long": default_long_signal}


def _fixed_position(mode, p, base, m, w):
    pos = SIGNALS[mode](p, base, m)
    return pos if w is None else pos * w


def adaptive_positions(mode, p, base, nr, w, block_starts, cost, min_days):
    """Positions using, in each block, the m chosen ONLY from OOS days before the block.

    ``block_starts`` are row positions (in the OOS arrays) of block starts.  Returns
    (positions, log rows).  net returns of each fixed-m strategy are causal (day t uses
    position t-1 and t), so their prefix [:k] is exactly what was knowable at row k."""
    grid = T.M_GRID[mode]
    nets = {m: M.net_returns(nr, _fixed_position(mode, p, base, m, w), cost) for m in grid}
    n = len(p)
    pos = np.empty(n)
    log = []
    stops = list(block_starts[1:]) + [n]
    for k, stop in zip(block_starts, stops):
        fallback = k < min_days
        chosen = T.M_FALLBACK[mode]
        if not fallback:
            scores = [M.sharpe(nets[m][:k], 1) for m in grid]
            scores = [s if np.isfinite(s) else -np.inf for s in scores]
            if max(scores) > -np.inf:
                chosen = grid[int(np.argmax(scores))]
            else:
                fallback = True
        sl = slice(k, stop)
        pos[sl] = _fixed_position(mode, p[sl], base[sl], chosen, None if w is None else w[sl])
        log.append(dict(block_row=k, m=float(chosen), n_past_days=int(k), used_fallback=bool(fallback)))
    return pos, log


def apply_delay(pos):
    """One-day execution delay: decided at close t, held from t+1 (first day flat)."""
    pos = np.asarray(pos, dtype="float64")
    return np.r_[0.0, pos[:-1]]


def evaluate(nr, pos, cost, ppy, delay=False) -> dict:
    """All trading metrics of a position stream (decided at close t, earning nr[t])."""
    nr = np.asarray(nr, dtype="float64")
    pos = apply_delay(pos) if delay else np.asarray(pos, dtype="float64")
    net = M.net_returns(nr, pos, cost)
    gross = pos * nr
    sr_d = M.sharpe(net, 1)
    return dict(
        n_days=int(len(nr)), total_return=M.total_return(net), cagr=M.cagr(net, ppy),
        ann_vol=M.ann_vol(net, ppy), sharpe=M.sharpe(net, ppy),
        sharpe_se=M.sharpe_standard_error(sr_d, len(net), ppy), sortino=M.sortino(net, ppy),
        max_drawdown=M.max_drawdown(net), pct_invested=M.pct_time_invested(pos),
        n_trades=M.n_trades(pos), turnover=M.turnover(pos),
        break_even_cost=M.break_even_cost(gross, pos), tstat=M.tstat_mean(net),
        mean_net_daily=float(net.mean()), mean_gross_daily=float(gross.mean()))


def cost_scenarios(ticker, headline: float) -> dict:
    """name -> (cost, delay).  ETFs: headline + 0.01%; BTC: headline + 0.60% stress; both
    also with the one-day execution delay at the headline cost."""
    sc = {"headline": (headline, False)}
    if ticker == F.BTC:
        sc["stress_cost"] = (STRESS_COST_BTC, False)
    else:
        sc["low_cost"] = (LOW_COST_ETF, False)
    sc["delay_1d"] = (headline, True)
    return sc


# =========================================================================== per-asset run
@dataclass
class AssetRun:
    ticker: str
    cutoff: pd.Timestamp
    models: list
    ppy: int
    headline_cost: float
    bt: BTConfig
    preds: pd.DataFrame
    retrains: pd.DataFrame
    importance_blocks: pd.DataFrame
    baselines: pd.DataFrame          # window, name, date, position, gross_return, net_return
    trial_returns: pd.DataFrame
    trial_index: pd.DataFrame
    trading_metrics: pd.DataFrame
    adaptive_m: pd.DataFrame
    prediction_metrics: pd.DataFrame
    raw_metrics: pd.DataFrame
    reliability: pd.DataFrame
    deciles: pd.DataFrame
    ensemble_members: list
    dev_first: pd.Timestamp
    dev_last: pd.Timestamp
    n_dev_rows: int


def _model_settings_json(model, models, ens_members):
    if model == "ensemble":
        s = {m: MD.model_settings(m) for m in ens_members}
    else:
        s = MD.model_settings(model)
    return json.dumps(s, sort_keys=True, default=str)


SKILL_NOTE = ("brier/log_loss skill are vs the TRAINING-window climatology and can be positive "
              "from base-rate drift alone; read with AUC and with brier_skill_vs_eval_base "
              "(skill vs the evaluation-window base rate)")


def _skill_vs_eval_base(y, p):
    """Brier skill vs a constant forecast equal to the evaluation-window up frequency
    (diagnostic; uses the evaluation labels on purpose, so it is NOT an out-of-sample yardstick)."""
    return M.brier_skill(y, p, float(np.mean(y)))


def prediction_metrics_tables(preds: pd.DataFrame):
    rows, rel, dec = [], [], []
    for (w, m), d in preds.groupby(["window", "model"], sort=False):
        d = d.sort_values("date")
        y, p, base = d["y"].to_numpy(), d["p_up"].to_numpy(), d["base_rate"].to_numpy()
        pred_up = (p >= base).astype(float)
        acc = M.accuracy_vs_base_rate(y, pred_up)
        pt = M.pesaran_timmermann(y, pred_up)
        rows.append(dict(
            window=w, model=m, n=int(len(y)), accuracy=acc["accuracy"],
            base_rate_oos=acc["base_rate"], accuracy_edge=acc["edge"],
            majority_rate=acc["majority_rate"], auc=M.roc_auc(y, p), log_loss=M.log_loss(y, p),
            brier=M.brier(y, p), brier_skill=M.brier_skill(y, p, base),
            log_loss_skill=M.log_loss_skill(y, p, base), pt_statistic=pt["statistic"],
            pt_pvalue=pt["pvalue"], share_up_days=pt["share_up_days"],
            share_up_predictions=pt["share_up_predictions"], hit_rate=pt["hit_rate"],
            mean_p=float(p.mean()), mean_base_rate=float(base.mean()),
            call_rate=float(np.mean(p >= base)),
            call_rate_flag=bool(np.mean(p >= base) < 0.10 or np.mean(p >= base) > 0.90),
            brier_skill_vs_eval_base=_skill_vs_eval_base(y, p), note=SKILL_NOTE))
        rel.append(M.reliability_curve(p, y).reset_index().assign(window=w, model=m))
        dec.append(M.hit_rate_by_confidence_decile(p, y, base).reset_index().assign(window=w, model=m))
    # climatology: the constant training base rate as the probability forecast
    for w, d in preds[preds["model"] == preds["model"].iloc[0]].groupby("window", sort=False):
        y, base = d["y"].to_numpy(), d["base_rate"].to_numpy()
        rows.append(dict(
            window=w, model="climatology", n=int(len(y)), accuracy=float(np.mean(y == 1)),   # always-up accuracy
            base_rate_oos=float(y.mean()), accuracy_edge=np.nan, majority_rate=np.nan, auc=0.5,
            log_loss=M.log_loss(y, base), brier=M.brier(y, base), brier_skill=0.0,
            log_loss_skill=0.0, pt_statistic=np.nan, pt_pvalue=np.nan,
            share_up_days=float(y.mean()), share_up_predictions=1.0, hit_rate=float(y.mean()),
            mean_p=float(base.mean()), mean_base_rate=float(base.mean()), call_rate=1.0,
            call_rate_flag=True, brier_skill_vs_eval_base=_skill_vs_eval_base(y, base),
            note=SKILL_NOTE))
    return (pd.DataFrame(rows), pd.concat(rel, ignore_index=True), pd.concat(dec, ignore_index=True))


def raw_prediction_metrics(preds: pd.DataFrame) -> pd.DataFrame:
    """DIAGNOSTIC ONLY: the same headline metrics on RAW (uncalibrated) probabilities.  The
    up/down call is p_raw >= base_rate (raw probabilities are not calibrated, so accuracy is
    only indicative).  Raw predictions never drive a strategy."""
    rows = []
    for (w, m), d in preds.groupby(["window", "model"], sort=False):
        d = d.sort_values("date")
        y, p, base = d["y"].to_numpy(), d["p_raw"].to_numpy(), d["base_rate"].to_numpy()
        acc = M.accuracy_vs_base_rate(y, (p >= base).astype(float))
        rows.append(dict(window=w, model=m, n=int(len(y)), auc=M.roc_auc(y, p),
                         brier_skill=M.brier_skill(y, p, base),
                         log_loss_skill=M.log_loss_skill(y, p, base), accuracy=acc["accuracy"],
                         base_rate_oos=acc["base_rate"], accuracy_edge=acc["edge"]))
    return pd.DataFrame(rows)


def run_asset(ticker, dev, cutoff, models, bt: BTConfig, headline_cost, windows=None,
              importance=True) -> AssetRun:
    """Everything for one asset from the DEV frame only."""
    _assert_dev(dev, cutoff)
    require_models(models)
    ppy = M.periods_per_year(ticker)
    ann = float(ppy)
    if windows is None:
        windows = ["expanding"] + (["rolling4y"] if ticker == F.BTC else [])
    scen = cost_scenarios(ticker, headline_cost)
    all_preds, all_ret, all_imp, base_rows, trial_ret, idx_rows, trade_rows, ad_rows = ([] for _ in range(8))
    streams = {}   # (window, position bytes) -> canonical index row (dedupe identical streams)
    ens_members = list(models) if len(models) > 1 else []
    call_info = {}
    for window in windows:
        wf = walk_forward(dev, models, bt, cutoff, window, ann, importance=(importance and window == "expanding"))
        all_preds.append(wf.preds)
        all_ret.append(wf.retrains)
        all_imp.append(wf.importance)
        retr = wf.retrains
        names = [*models, *(["ensemble"] if ens_members else [])]
        d0 = wf.preds[wf.preds["model"] == names[0]].sort_values("date")
        oos_dates = pd.DatetimeIndex(d0["date"])
        nr = d0["next_return"].to_numpy()
        block_ids = d0["block"].to_numpy()
        block_starts = np.flatnonzero(np.r_[True, block_ids[1:] != block_ids[:-1]]).tolist()

        # ---- baselines (not trials)
        bpos = baseline_positions(dev, retr, oos_dates, ann, bt)
        for bname, pos in bpos.items():
            pos_a = pos.to_numpy(dtype="float64")
            pos_a = np.nan_to_num(pos_a, nan=0.0)   # trend warm-up (never in practice) = flat
            net = M.net_returns(nr, pos_a, headline_cost)
            base_rows.append(pd.DataFrame({"window": window, "name": bname, "date": oos_dates,
                                           "position": pos_a, "gross_return": pos_a * nr,
                                           "net_return": net}))
            for sname, (cost, delay) in scen.items():
                trade_rows.append(dict(name=bname, kind="baseline", window=window, model="",
                                       mode="", vol_targeted=bname == "vol_targeted_bh",
                                       m_setting="", scenario=sname, cost=cost, delay=delay,
                                       **evaluate(nr, pos_a, cost, ppy, delay)))
        w_vt = bpos["vol_targeted_bh"].to_numpy(dtype="float64")

        # ---- strategies (each configuration is a trial)
        for model in names:
            dm = wf.preds[wf.preds["model"] == model].sort_values("date")
            if not np.array_equal(dm["date"].to_numpy(), d0["date"].to_numpy()):
                raise AssertionError("models do not share the same OOS dates")
            p, base = dm["p_up"].to_numpy(), dm["base_rate"].to_numpy()
            cr = float(np.mean(p >= base))
            call_info = dict(call_rate=cr, mean_p=float(p.mean()), mean_base_rate=float(base.mean()),
                             call_rate_flag=bool(cr < 0.10 or cr > 0.90))
            mjson = _model_settings_json(model, models, ens_members)
            for mode in T.MODES:
                for vt in (False, True):
                    w = w_vt if vt else None
                    for m in T.m_settings(mode):
                        if m == T.ADAPTIVE:
                            pos, log = adaptive_positions(mode, p, base, nr, w, block_starts,
                                                          headline_cost, bt.adaptive_min_days)
                            cid = T.make_config_id(ticker, window, model, mode, vt, m)
                            for lg in log:
                                ad_rows.append(dict(
                                    config_id=cid, window=window, model=model, mode=mode,
                                    vol_targeted=vt, block=oos_dates[lg["block_row"]],
                                    **{k: v for k, v in lg.items() if k != "block_row"}))
                        else:
                            pos = _fixed_position(mode, p, base, m, w)
                        net = M.net_returns(nr, pos, headline_cost)
                        cid = T.make_config_id(ticker, window, model, mode, vt, m)
                        key = (window, np.ascontiguousarray(pos, dtype="float64").tobytes())
                        canon = streams.get(key)
                        if canon is None:
                            row = T.build_index_row(ticker, window, model, mode, vt, m,
                                                    headline_cost, mjson, oos_dates, net, ppy)
                            streams[key] = row
                            idx_rows.append(row)
                            trial_ret.append(pd.DataFrame({
                                "config_id": cid, "date": oos_dates, "net_return": net,
                                "gross_return": pos * nr, "position": pos}))
                            dup = ""
                        else:
                            al = json.loads(canon["aliases"]) + [cid]
                            canon["aliases"] = json.dumps(sorted(al))
                            dup = canon["config_id"]
                        for sname, (cost, delay) in scen.items():
                            trade_rows.append(dict(
                                name=cid, kind="trial", window=window, model=model, mode=mode,
                                vol_targeted=vt, m_setting=T.m_label(m), scenario=sname,
                                cost=cost, delay=delay, duplicate_of=dup, **call_info,
                                **evaluate(nr, pos, cost, ppy, delay)))

    preds = pd.concat(all_preds, ignore_index=True)
    pm, rel, dec = prediction_metrics_tables(preds)
    return AssetRun(
        ticker=ticker, cutoff=pd.Timestamp(cutoff), models=list(models), ppy=ppy,
        headline_cost=float(headline_cost), bt=bt, preds=preds,
        retrains=pd.concat(all_ret, ignore_index=True),
        importance_blocks=pd.concat([d for d in all_imp if len(d)], ignore_index=True)
        if any(len(d) for d in all_imp) else all_imp[0],
        baselines=pd.concat(base_rows, ignore_index=True),
        trial_returns=pd.concat(trial_ret, ignore_index=True),
        trial_index=pd.DataFrame(idx_rows, columns=T.INDEX_COLS),
        trading_metrics=pd.DataFrame(trade_rows),
        adaptive_m=pd.DataFrame(ad_rows), prediction_metrics=pm, raw_metrics=raw_prediction_metrics(preds), reliability=rel, deciles=dec,
        ensemble_members=ens_members, dev_first=pd.Timestamp(dev.index.min()),
        dev_last=pd.Timestamp(dev.index.max()), n_dev_rows=int(len(dev)))


# =========================================================================== significance
SIG_COLS = ["test", "window", "subject", "reference", "value", "statistic", "pvalue", "n", "details"]


def _sig(test, window, subject, reference="", value=np.nan, statistic=np.nan, pvalue=np.nan,
         n=0, details=None):
    return dict(test=test, window=window, subject=subject, reference=reference, value=value,
                statistic=statistic, pvalue=pvalue, n=int(n),
                details=json.dumps(details or {}, sort_keys=True, default=float))


def _psr(net):
    sr = M.sharpe(net, 1)
    if not np.isfinite(sr):
        return float("nan"), sr
    from scipy import stats
    return M.probabilistic_sharpe_ratio(sr, 0.0, len(net), float(stats.skew(net)),
                                        float(stats.kurtosis(net, fisher=False))), sr


def compute_significance(run: AssetRun, n_trials: int, trial_sr: np.ndarray):
    """Significance tests for one asset.  Returns (significance_df, random_strategies_df).

    ``n_trials`` / ``trial_sr`` are the GLOBAL trial count (distinct return streams over all
    assets in the log) and the per-day trial Sharpe vector.

    Rows (``test``):
    * random_percentile: Sharpe percentile among ``n_random`` (1000) random circular shifts of the
      position series (same time in market, gross).  The 1000 random Sharpes are saved in
      random_strategies.parquet.  CAUTION: for vol-targeted trials the shift also scrambles the
      volatility alignment, so their percentile mostly reflects vol targeting, not model timing.
    * psr: PSR vs a Sharpe of 0.
    * dsr: Deflated Sharpe Ratio, two scopes.  scope='global' pools every trial of every asset
      (N = distinct streams in the log) with the variance of ALL trial Sharpes; scope='asset' uses
      only this asset's trials (N, variance and SR0 are shown in ``details``).  The effective
      number of INDEPENDENT trials is far smaller than N (many strategies are near-copies of each
      other), so both are conservative in N but the pooled variance mixes assets with very
      different Sharpe levels; read the two together.  DSR tests against zero, not against
      buy-and-hold, so any long-biased strategy on a rising asset scores high.
    * psr_excess_vt / psr_excess_bh and dsr_excess_vt / dsr_excess_bh: the same PSR/DSR on the
      EXCESS daily return over vol-targeted buy-and-hold (or buy-and-hold) of the same window, which
      separates the vol-targeting effect from model timing.  The excess DSR is asset-scoped: N =
      this asset's distinct trials, variance = variance of the trials' excess Sharpes.
    * tstat_vs_3: t-stat of the mean net return vs the Harvey-Liu-Zhu t>3 hurdle.  NOTE t>3 is a
      test against ZERO mean return.  Baselines (buy_and_hold, vol_targeted_bh) get a row too
      (details.baseline=true) so a strategy's t can be read next to the drift it merely inherits.
    * diebold_mariano: log loss; subject=model, reference=logreg|climatology; statistic > 0 means
      the SUBJECT has the higher loss.
    * spa: Hansen SPA of every strategy of the asset vs buy-and-hold (reference='buy_and_hold') and
      vs vol-targeted buy-and-hold (reference='vol_targeted_bh').
    """
    bt = run.bt
    rows, rand = [], []
    nr_by_window = {}
    for w, d in run.preds.groupby("window"):
        dd = d[d["model"] == d["model"].iloc[0]].sort_values("date")
        nr_by_window[w] = dd.set_index("date")["next_return"]
    tr = run.trial_returns
    idx = run.trial_index.set_index("config_id")
    bl = run.baselines
    base_net = {(w, n): g.sort_values("date").set_index("date")["net_return"]
                for (w, n), g in bl.groupby(["window", "name"])}
    asset_sr = run.trial_index["daily_sharpe"].to_numpy(dtype=float)
    n_asset = int(len(run.trial_index))
    # excess returns over both benchmarks, for every distinct trial
    excess = {}
    for cid, d in tr.groupby("config_id", sort=True):
        w = idx.loc[cid, "window"]
        d = d.sort_values("date").set_index("date")["net_return"]
        for key, bn in (("vt", "vol_targeted_bh"), ("bh", "buy_and_hold")):
            excess[(key, cid)] = (d - base_net[(w, bn)].reindex(d.index)).to_numpy()
    ex_sr = {k: np.array([M.sharpe(excess[(k, c)], 1) for c in sorted(idx.index)]) for k in ("vt", "bh")}
    for cid, d in tr.groupby("config_id", sort=True):
        w = idx.loc[cid, "window"]
        d = d.sort_values("date")
        net = d["net_return"].to_numpy()
        nr = nr_by_window[w].loc[d["date"]].to_numpy()
        pos = d["position"].to_numpy()
        n = len(net)
        rp = M.random_strategy_percentile(nr, pos, n=bt.n_random, seed=bt.seed)
        rows.append(_sig("random_percentile", w, cid, value=rp["percentile"],
                         statistic=rp["strategy_sharpe"], n=n,
                         details={"method": rp["method"], "n_random": bt.n_random,
                                  "vol_targeted": bool(idx.loc[cid, "vol_targeted"])}))
        rand.append(pd.DataFrame({"config_id": cid, "sim": np.arange(len(rp["random_sharpes"]), dtype="int32"),
                                  "sharpe_daily": rp["random_sharpes"].astype("float32")}))
        psr, sr = _psr(net)
        rows.append(_sig("psr", w, cid, reference="SR*=0", value=psr, statistic=sr, n=n))
        for scope, ts, nt in (("global", trial_sr, n_trials), ("asset", asset_sr, n_asset)):
            dsr = M.deflated_sharpe_ratio(net, trial_sharpes=ts, n_trials=nt)
            rows.append(_sig("dsr", w, cid, reference=scope, value=dsr["dsr"], statistic=dsr["sr"], n=n,
                             details={"scope": scope, "sr0": dsr["sr0"], "n_trials": dsr["n_trials"],
                                      "sr_var": dsr["sr_var"]}))
        for key in ("vt", "bh"):
            ex = excess[(key, cid)]
            psr_e, sr_e = _psr(ex)
            rows.append(_sig(f"psr_excess_{key}", w, cid, reference="SR*=0", value=psr_e,
                             statistic=sr_e, n=n))
            dsr = M.deflated_sharpe_ratio(ex, trial_sharpes=ex_sr[key], n_trials=n_asset)
            rows.append(_sig(f"dsr_excess_{key}", w, cid, reference="asset", value=dsr["dsr"],
                             statistic=dsr["sr"], n=n,
                             details={"scope": "asset", "sr0": dsr["sr0"], "n_trials": dsr["n_trials"],
                                      "sr_var": dsr["sr_var"]}))
        th = M.t_hurdle(net, 3.0)
        rows.append(_sig("tstat_vs_3", w, cid, reference="t>3 (test vs zero)", value=3.0,
                         statistic=th["tstat"], n=n, details={"passes": th["passes"]}))
    for (w, bn), ser in sorted(base_net.items()):
        if bn in ("buy_and_hold", "vol_targeted_bh"):
            th = M.t_hurdle(ser.to_numpy(), 3.0)
            rows.append(_sig("tstat_vs_3", w, bn, reference="t>3 (test vs zero)", value=3.0,
                             statistic=th["tstat"], n=len(ser),
                             details={"passes": th["passes"], "baseline": True}))
    # ---- Diebold-Mariano on log loss
    for w, d in run.preds.groupby("window"):
        piv_p = d.pivot(index="date", columns="model", values="p_up").sort_index()
        first = d[d["model"] == d["model"].iloc[0]].sort_values("date").set_index("date")
        y, base = first["y"], first["base_rate"]
        ll = {m: M.log_loss_series(y.loc[piv_p.index].to_numpy(), piv_p[m].to_numpy())
              for m in piv_p.columns}
        ll["climatology"] = M.log_loss_series(y.loc[piv_p.index].to_numpy(),
                                              base.loc[piv_p.index].to_numpy())
        for m in piv_p.columns:
            for ref in ("logreg", "climatology"):
                if ref == m or ref not in ll:
                    continue
                r = M.diebold_mariano(ll[m], ll[ref], h=1)
                rows.append(_sig("diebold_mariano", w, m, reference=ref, value=r["mean_diff"],
                                 statistic=r["statistic"], pvalue=r["pvalue"], n=len(piv_p)))
    # ---- Hansen SPA vs both benchmarks (expanding-window dates; the rolling variant, which
    # starts earlier, is cut to the same dates)
    strat = None
    for bn in ("buy_and_hold", "vol_targeted_bh"):
        bench = base_net[("expanding", bn)]
        strat = tr.pivot(index="date", columns="config_id", values="net_return").reindex(bench.index)
        strat = strat.dropna(axis=1, how="any")
        spa = M.spa_test(bench, strat, reps=bt.spa_reps, block_size=10, seed=bt.seed)
        rows.append(_sig("spa", "all", "all_strategies", reference=bn,
                         value=spa["best_excess_mean"], pvalue=spa["pvalue"], n=len(bench),
                         details={"best_strategy": spa["best_strategy"],
                                  "pvalue_lower": spa["pvalue_lower"],
                                  "pvalue_upper": spa["pvalue_upper"],
                                  "n_strategies": int(strat.shape[1])}))
    return (pd.DataFrame(rows, columns=SIG_COLS),
            pd.concat(rand, ignore_index=True) if rand else pd.DataFrame(columns=["config_id", "sim", "sharpe_daily"]))


# =========================================================================== output
def config_hash(run: AssetRun) -> str:
    blob = dict(bt=asdict(run.bt), cost=run.headline_cost, models=run.models,
                settings={m: MD.model_settings(m) for m in run.models},
                grids=dict(T.M_GRID), fallback=dict(T.M_FALLBACK), ppy=run.ppy,
                scenarios={k: list(v) for k, v in cost_scenarios(run.ticker, run.headline_cost).items()})
    return hashlib.sha256(json.dumps(blob, sort_keys=True, default=str).encode()).hexdigest()


def neg_slopes(run: AssetRun) -> dict:
    out = {}
    for w, d in run.retrains.groupby("window"):
        for c in d.columns:
            if c.startswith("platt_slope_"):
                out[f"{w}/{c[len('platt_slope_'):]}"] = dict(
                    negative=int((d[c] < 0).sum()), blocks=int(d[c].notna().sum()),
                    mean_slope=float(d[c].mean()))
    return out


def call_rate_flags(run: AssetRun) -> list:
    pm = run.prediction_metrics
    pm = pm[pm["model"] != "climatology"]
    return [f"{r.window}/{r.model}: call rate {r.call_rate:.3f}" for r in pm.itertuples()
            if r.call_rate_flag]


def metadata(run: AssetRun, n_trials_global: int) -> dict:
    wins = {}
    for w, d in run.retrains.groupby("window"):
        pw = run.preds[run.preds["window"] == w]
        wins[w] = dict(n_retrains=int(len(d)), first_oos_date=str(pw["date"].min().date()),
                       last_oos_date=str(pw["date"].max().date()),
                       n_oos_days=int(pw[pw["model"] == pw["model"].iloc[0]].shape[0]))
    return dict(
        ticker=run.ticker, holdout_cutoff_date=str(run.cutoff.date()),
        holdout_cutoff_pinned=True, holdout_months=run.bt.holdout_months,
        call_rate_flags=call_rate_flags(run), skill_note=SKILL_NOTE,
        dsr_note=("global DSR pools all assets' distinct streams; asset DSR uses this asset only; "
                  "effective independent trials << N; DSR/t>3 test against zero, not vs buy-and-hold"),
        dev_first_date=str(run.dev_first.date()), dev_last_date=str(run.dev_last.date()),
        n_dev_rows=run.n_dev_rows, windows=wins, models=run.models,
        ensemble_members=run.ensemble_members, headline_cost=run.headline_cost,
        cost_scenarios={k: dict(cost=v[0], delay=v[1])
                        for k, v in cost_scenarios(run.ticker, run.headline_cost).items()},
        periods_per_year=run.ppy, settings=asdict(run.bt),
        model_settings={m: MD.model_settings(m) for m in run.models},
        m_grid={k: list(v) for k, v in T.M_GRID.items()}, m_fallback=dict(T.M_FALLBACK),
        n_trials_this_asset=int(len(run.trial_index)),
        n_configs_before_dedupe=int(len(T.all_logged_config_ids(run.trial_index))),
        negative_platt_slope_blocks=neg_slopes(run), n_blocks=int(run.retrains.groupby("window").size().max()), n_trials_global=int(n_trials_global),
        config_hash=config_hash(run),
        notes=("hold-out excluded: last dev row label blanked; training uses rows <= s-2 of "
               "each block start s (1-day gap); rolling4y uses min_cal=%d" % run.bt.rolling_min_cal))


def _sorted(df, cols):
    return df.sort_values(cols, kind="mergesort").reset_index(drop=True)


def write_asset_outputs(run: AssetRun, results_dir, significance: pd.DataFrame, n_trials_global: int,
                        random_df: pd.DataFrame | None = None):
    d = Path(results_dir) / run.ticker
    d.mkdir(parents=True, exist_ok=True)
    _sorted(run.preds, ["window", "model", "date"]).to_parquet(d / "predictions.parquet", index=False)
    _sorted(run.prediction_metrics, ["window", "model"]).to_parquet(d / "prediction_metrics.parquet", index=False)
    _sorted(run.raw_metrics, ["window", "model"]).to_parquet(d / "raw_prediction_metrics.parquet", index=False)
    _sorted(run.reliability, ["window", "model", "bin"]).to_parquet(d / "prediction_reliability.parquet", index=False)
    _sorted(run.deciles, ["window", "model", "decile"]).to_parquet(d / "prediction_deciles.parquet", index=False)
    _sorted(run.trading_metrics, ["kind", "window", "name", "scenario"]).to_parquet(d / "trading_metrics.parquet", index=False)
    _sorted(run.baselines, ["window", "name", "date"]).to_parquet(d / "baselines.parquet", index=False)
    _sorted(significance, ["test", "window", "subject", "reference"]).to_parquet(d / "significance.parquet", index=False)
    if random_df is not None:
        _sorted(random_df, ["config_id", "sim"]).to_parquet(d / "random_strategies.parquet", index=False)
    agg = aggregate_importance(run.importance_blocks)
    agg.to_parquet(d / "importance.parquet", index=False)
    (_sorted(run.adaptive_m, ["config_id", "block"]) if len(run.adaptive_m) else run.adaptive_m
     ).to_parquet(d / "adaptive_m.parquet", index=False)
    run.retrains.to_parquet(d / "retrains.parquet", index=False)
    with open(d / "metadata.json", "w") as f:
        json.dump(metadata(run, n_trials_global), f, indent=2, sort_keys=True)
    return d


def run_and_save(runs: list, results_dir):
    """Log all runs' trials (replacing each ticker's previous rows), then compute significance
    with the GLOBAL trial count and write every asset's files."""
    for r in runs:
        T.replace_asset(r.ticker, r.trial_index, r.trial_returns, results_dir)
    n_trials = T.count_trials(results_dir)
    sr = T.trial_sharpes(results_dir)
    out = {}
    for r in runs:
        sig, rnd = compute_significance(r, n_trials, sr)
        out[r.ticker] = write_asset_outputs(r, results_dir, sig, n_trials, rnd)
    return out


# =========================================================================== CLI
HINT = "(run `python -m quant.data` then `python -m quant.features` to build the datasets)"


def _err(msg, hint=True):
    print(f"ERROR: {msg}" + (f" {HINT}" if hint else ""), file=sys.stderr)
    return 2


def main(argv=None, bt_overrides=None):
    from quant.config import load_config

    ap = argparse.ArgumentParser(description="Walk-forward backtest on the DEV portion (hold-out excluded).")
    ap.add_argument("--assets", nargs="*", default=None)
    ap.add_argument("--models", nargs="*", default=list(MD.MODEL_NAMES))
    ap.add_argument("--config", default=None)
    ap.add_argument("--results-dir", default=None)
    args = ap.parse_args(argv)
    try:
        cfg = load_config(args.config)
    except (OSError, ValueError, KeyError) as e:
        return _err(f"cannot load config: {e}", hint=False)
    try:
        require_models(args.models)
    except (RuntimeError, ValueError) as e:
        return _err(e, hint=False)
    tickers = args.assets or [a["ticker"] for a in cfg["assets"]]
    unknown = [t for t in tickers if t not in cfg["costs"]]
    if unknown:
        return _err(f"unknown asset(s) {unknown}", hint=False)
    results_dir = Path(args.results_dir) if args.results_dir else cfg["data_dir"] / "results"
    try:
        bt = BTConfig(holdout_months=int(cfg["holdout_months"]), **(bt_overrides or {}))
    except TypeError as e:
        return _err(f"bad backtest settings: {e}", hint=False)
    runs = []
    t0 = time.time()
    print(f"hold-out cutoff PINNED at {cfg['holdout_cutoff_date']} (config.yaml); never change it "
          "after Stage 3", flush=True)
    for tk in tickers:
        t1 = time.time()
        try:
            dev, cutoff = load_dev(tk, cfg, bt)
            print(f"[{tk}] dev rows {len(dev)} ({dev.index.min().date()}..{dev.index.max().date()}), "
                  f"hold-out cutoff {cutoff.date()} (hold-out excluded)", flush=True)
            runs.append(run_asset(tk, dev, cutoff, args.models, bt, cfg["costs"][tk]))
        except (OSError, ValueError, KeyError) as e:
            return _err(f"{tk}: {e}")
        print(f"[{tk}] walk-forward + strategies done in {time.time() - t1:.0f}s", flush=True)
    try:
        run_and_save(runs, results_dir)
    except (OSError, ValueError) as e:
        return _err(f"saving results: {e}", hint=False)
    print(f"done: {len(runs)} asset(s), {T.count_trials(results_dir)} trials in log, "
          f"{time.time() - t0:.0f}s total; results in {results_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
