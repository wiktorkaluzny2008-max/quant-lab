"""Leverage simulator (QUANT_SPEC.md "Leverage simulator").  WHAT-IF ONLY, FAKE MONEY.

This module only *re-uses already saved out-of-sample results* (``baselines.parquet`` and
``trials.parquet`` in ``<data_dir>/results``); it never imports or runs a model.  Its results are
NOT trials: nothing is written to the trial log, nothing is selected by them, and the hold-out is
never touched (inputs are the saved OOS results, whose last date is <= the hold-out cutoff, and
the Kelly estimate uses data strictly before the first OOS date).  Paper trading stays unlevered.

Series simulated per asset (``SERIES``):
  buy_and_hold      position 1 every day
  vol_targeted_bh   the saved vol-targeted buy-and-hold weights
  chosen_strategy   ONE logged trial config named in ``config.yaml: leverage_strategy`` (default:
                    ensemble, long_flat, vol_targeted, adaptive m, expanding window).  It is fixed
                    by config, NOT picked by Sharpe, so leverage results can never select
                    anything.  If that model is not in the log (e.g. LightGBM missing) the model
                    falls back to ensemble -> logreg -> any other logged model, other settings
                    unchanged; the model actually used and the reason are stored.

Daily-rebalanced leverage ``L`` (the equity is re-levered to L every day):

    r_L,t = L * g_t - (L - 1) * borrow_rate / days_per_year - cost * L * |pos_t - pos_{t-1}|

``g_t`` = saved gross return of the series (position x asset return), ``pos`` its position
(entry from 0 on the first day, like ``metrics.net_returns``), ``cost`` the headline cost from
config, ``days_per_year`` 365 for BTC-USD (trades_weekends) else 252.  The borrow charge applies
to the whole borrowed fraction (L - 1) of equity every day, also on days the strategy is flat
(conservative: a real broker would let you de-lever).  L = 1 reproduces the saved net returns
exactly.  No financing on cash is credited.

Wipe-out: if equity (start 1.0) ever falls below 20% of its start, ``wiped_out`` is set, the date
is stored and the path STOPS there: equity is frozen at that day's level (floored at 0), later
returns are 0.  Metrics are computed on that frozen path.

Kelly / growth curve (MacLean, Thorp & Ziemba 2010): ``g(L) = L*mu - L^2*sigma^2/2`` with
``L* = mu/sigma^2`` and ``g(2 L*) = 0``.  mu, sigma^2 are annualised arithmetic mean and variance
of the underlying asset's DAILY returns over dates strictly BEFORE the first OOS date (the
pre-OOS part of the dev data), so the curve is not in-sample for the displayed OOS period.
Uncertainty: se(mu) = sigma / sqrt(years of history); band = mu +/- 1.96 se divided by sigma^2
(the lower end floored at 0).  sigma^2 is treated as known; Kelly is very sensitive to mu.
The risk-free rate is taken as 0 (as in the Sharpe ratios); ``L_star_net_borrow`` uses
``(mu - borrow_rate)/sigma^2`` as a more conservative figure.

Outputs per ticker in ``<data_dir>/results/<ticker>/``: ``leverage.parquet``,
``leverage_equity.parquet`` (long, equity normalised to 1.0), ``kelly.parquet``, ``crashes.json``.
Re-running overwrites them with identical contents (idempotent).

CLI:  ``python -m quant.leverage [--assets SPY QQQ ...] [--config path]``
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from quant import metrics as M
from quant.config import load_config

LEVERAGES = (1.0, 1.5, 2.0, 3.0)
WIPEOUT_LEVEL = 0.20
KELLY_Z = 1.96
SERIES = ("buy_and_hold", "vol_targeted_bh", "chosen_strategy")
BASELINE_WINDOW = "expanding"

# Crash windows (inclusive calendar dates) stored as metadata for the dashboard.
CRASHES_ETF = [
    ("dotcom_2000_2002", "Dot-com crash", "2000-03-10", "2002-10-09"),
    ("gfc_2008", "Financial crisis", "2007-10-09", "2009-03-09"),
    ("covid_2020", "COVID crash (Mar 2020)", "2020-02-19", "2020-03-23"),
    ("bear_2022", "2022 bear market", "2022-01-03", "2022-10-12"),
]
CRASHES_BTC = [
    ("bear_2018", "2018 crypto bear market", "2017-12-17", "2018-12-15"),
    ("covid_2020", "COVID crash (Mar 2020)", "2020-02-12", "2020-03-13"),
    ("bear_2022", "2022 crypto bear market", "2021-11-10", "2022-12-31"),
]


class MissingResultsError(RuntimeError):
    """Saved baseline/trial results needed by the leverage simulator are missing."""


def days_per_year(ticker, cfg=None) -> int:
    """365 for assets that trade weekends (BTC-USD), 252 otherwise."""
    if cfg is not None:
        for a in cfg["assets"]:
            if a["ticker"] == ticker:
                return 365 if a["trades_weekends"] else 252
    return M.periods_per_year(ticker)


# --------------------------------------------------------------------------- simulation
def leveraged_returns(gross, position, cost, L, borrow_rate, dpy) -> np.ndarray:
    """r_L,t = L*g_t - (L-1)*borrow/dpy - cost*L*|dpos_t| (entry from 0).  No wipe-out logic."""
    g = np.asarray(gross, dtype="float64")
    pos = np.asarray(position, dtype="float64")
    if g.shape != pos.shape:
        raise ValueError("gross and position must have the same length")
    dpos = np.abs(np.diff(np.concatenate([[0.0], pos])))
    return L * g - (L - 1.0) * borrow_rate / dpy - cost * L * dpos


def apply_wipeout(r, level=WIPEOUT_LEVEL):
    """Stop the path the first day equity < ``level``: returns (path_returns, wipe_idx or None).
    On that day equity is floored at 0; afterwards returns are 0 (equity frozen)."""
    r = np.array(r, dtype="float64")
    eq = np.cumprod(1.0 + r)
    bad = np.flatnonzero(eq < level)
    if bad.size == 0:
        return r, None
    k = int(bad[0])
    r[k] = max(r[k], -1.0)
    r[k + 1:] = 0.0
    return r, k


def longest_underwater(equity) -> int:
    """Longest run (trading days) between a peak and the next time equity regains it, counted as
    consecutive days with equity < running peak.  An unrecovered drawdown counts up to the end.
    Equity is prepended with the 1.0 starting value."""
    peak, run, best = 1.0, 0, 0
    for e in equity:
        if e >= peak:
            peak, run = e, 0
        else:
            run += 1
            best = max(best, run)
    return best


def _worst_month(r, dates):
    s = pd.Series(r, index=pd.DatetimeIndex(dates))
    m = (1.0 + s).groupby([s.index.year, s.index.month]).prod() - 1.0
    k = m.idxmin()
    return float(m.min()), f"{k[0]:04d}-{k[1]:02d}"


def path_metrics(r, dates, ppy) -> dict:
    """Metrics of a (post-wipe-out) return path."""
    r = np.asarray(r, dtype="float64")
    eq = np.cumprod(1.0 + r)
    wd = int(np.argmin(r))
    wm, wm_label = _worst_month(r, dates)
    return dict(
        cagr=M.cagr(r, ppy), max_drawdown=M.max_drawdown(r),
        worst_day=float(r[wd]), worst_day_date=pd.Timestamp(dates[wd]),
        worst_month=wm, worst_month_label=wm_label,
        longest_recovery_days=int(longest_underwater(eq)),
        unrecovered=bool(eq[-1] < np.max(np.concatenate([[1.0], eq]))),
        sharpe=M.sharpe(r, ppy), final_equity=float(eq[-1]), n_days=int(r.size))


def simulate_series(dates, gross, position, cost, borrow_rate, dpy, ppy=None, leverages=LEVERAGES):
    """Simulate every L.  Returns (metrics rows list[dict], {L: (path_returns, equity)})."""
    ppy = dpy if ppy is None else ppy
    rows, paths = [], {}
    for L in leverages:
        raw = leveraged_returns(gross, position, cost, L, borrow_rate, dpy)
        r, k = apply_wipeout(raw)
        row = dict(L=float(L), **path_metrics(r, dates, ppy))
        row["wiped_out"] = k is not None
        row["wipe_date"] = pd.Timestamp(dates[k]) if k is not None else pd.NaT
        rows.append(row)
        paths[float(L)] = (r, np.cumprod(1.0 + r))
    return rows, paths


# --------------------------------------------------------------------------- Kelly
def kelly_estimate(returns, ppy, borrow_rate=0.0, z=KELLY_Z) -> dict:
    """Annualised mu, sigma^2, Kelly L* = mu/sigma^2 and its band from daily ``returns``."""
    r = np.asarray(returns, dtype="float64")
    r = r[np.isfinite(r)]
    if r.size < 30:
        raise ValueError(f"need at least 30 return observations for Kelly, got {r.size}")
    mu = float(r.mean() * ppy)
    var = float(np.var(r, ddof=1) * ppy)
    years = r.size / ppy
    se_mu = float(np.sqrt(var) / np.sqrt(years))
    return dict(mu=mu, sigma2=var, sigma=float(np.sqrt(var)), n_days=int(r.size), years=float(years),
                se_mu=se_mu, L_star=mu / var, L_star_lo=max(0.0, (mu - z * se_mu) / var),
                L_star_hi=max(0.0, (mu + z * se_mu) / var),
                L_star_net_borrow=(mu - borrow_rate) / var)


def growth_curve(L, mu, sigma2):
    """g(L) = L*mu - L^2*sigma2/2 (annual log-growth approximation)."""
    L = np.asarray(L, dtype="float64")
    return L * mu - L ** 2 * sigma2 / 2.0


def kelly_table(ticker, est: dict, borrow_rate, train_start, train_end) -> pd.DataFrame:
    top = float(min(10.0, max(4.0, 2.5 * max(est["L_star"], 0.0))))
    grid = np.round(np.arange(0.0, top + 1e-9, 0.05), 10)
    se, var = est["se_mu"], est["sigma2"]
    df = pd.DataFrame(dict(
        ticker=ticker, L=grid, g=growth_curve(grid, est["mu"], var),
        g_lo=growth_curve(grid, est["mu"] - KELLY_Z * se, var),
        g_hi=growth_curve(grid, est["mu"] + KELLY_Z * se, var)))
    for k in ("mu", "sigma", "sigma2", "se_mu", "n_days", "years", "L_star", "L_star_lo",
              "L_star_hi", "L_star_net_borrow"):
        df[k] = est[k]
    df["borrow_rate"] = float(borrow_rate)
    df["train_start"] = pd.Timestamp(train_start)
    df["train_end"] = pd.Timestamp(train_end)
    return df


# --------------------------------------------------------------------------- loading
def _results_dir(cfg) -> Path:
    return Path(cfg["data_dir"]) / "results"


def resolve_chosen(index: pd.DataFrame, ticker, spec: dict):
    """(config_id, model_used, note) for the chosen strategy, with the model fallback."""
    sub = index[(index["ticker"] == ticker) & (index["window"] == spec["window"])
                & (index["mode"] == spec["mode"])
                & (index["vol_targeted"] == bool(spec["vol_targeted"]))
                & (index["m_setting"] == str(spec["m_setting"]))]
    avail = list(sub["model"].unique())
    order = [spec["model"], "ensemble", "logreg"] + sorted(avail)
    for mdl in order:
        if mdl in avail:
            cid = sub.loc[sub["model"] == mdl, "config_id"].iloc[0]
            note = "" if mdl == spec["model"] else (
                f"requested model '{spec['model']}' not in the trial log; used '{mdl}'")
            return cid, mdl, note
    raise MissingResultsError(
        f"{ticker}: no logged trial matches leverage_strategy {spec}; run quant.backtest first")


def load_inputs(ticker, cfg):
    """{series: DataFrame(date, position, gross_return)} + info dict, from saved OOS results."""
    rd = _results_dir(cfg)
    bp, tp, ip = rd / ticker / "baselines.parquet", rd / "trials.parquet", rd / "trials_index.parquet"
    for p in (bp, tp, ip):
        if not p.exists():
            raise MissingResultsError(
                f"{ticker}: missing saved results file {p}; run `python -m quant.backtest "
                f"--assets {ticker}` first (leverage only re-uses saved OOS results)")
    base = pd.read_parquet(bp)
    base = base[base["window"] == BASELINE_WINDOW]
    out = {}
    for name in ("buy_and_hold", "vol_targeted_bh"):
        b = base[base["name"] == name].sort_values("date")
        if b.empty:
            raise MissingResultsError(f"{ticker}: baseline '{name}' ({BASELINE_WINDOW}) not in {bp}")
        out[name] = b[["date", "position", "gross_return"]].reset_index(drop=True)
    idx = pd.read_parquet(ip)
    cid, mdl, note = resolve_chosen(idx, ticker, cfg["leverage_strategy"])
    t = pd.read_parquet(tp, filters=[("config_id", "==", cid)])
    if t.empty:
        raise MissingResultsError(f"{ticker}: trial {cid} has no rows in {tp}")
    out["chosen_strategy"] = t.sort_values("date")[["date", "position", "gross_return"]].reset_index(drop=True)
    for k, d in out.items():
        if d[["position", "gross_return"]].isna().any().any():
            raise ValueError(f"{ticker}/{k}: NaN in saved position/gross_return")
    meta_p = rd / ticker / "metadata.json"
    cutoff = None
    if meta_p.exists():
        cutoff = pd.Timestamp(json.loads(meta_p.read_text())["holdout_cutoff_date"])
        for k, d in out.items():
            if d["date"].max() > cutoff:
                raise AssertionError(f"{ticker}/{k}: results contain hold-out dates "
                                     f"({d['date'].max()} > {cutoff})")
    return out, dict(config_id=cid, chosen_model=mdl, chosen_note=note, cutoff=cutoff)


def load_preoos_returns(ticker, cfg, first_oos_date, cutoff=None) -> pd.Series:
    """Daily close-to-close returns of the underlying with date STRICTLY BEFORE the first OOS date
    (hence in the dev portion; never hold-out).  The file is truncated right after loading."""
    p = Path(cfg["data_dir"]) / f"{ticker}.parquet"
    if not p.exists():
        raise MissingResultsError(f"{ticker}: price file {p} missing; run the data update first")
    px = pd.read_parquet(p)["Close"].astype("float64")
    first = pd.Timestamp(first_oos_date)
    px = px[px.index < first]
    if cutoff is not None and len(px) and px.index.max() > pd.Timestamp(cutoff):
        raise AssertionError("pre-OOS returns contain hold-out dates")
    return px.pct_change().dropna()


# --------------------------------------------------------------------------- crashes
def crash_windows(ticker, cfg=None):
    is_btc = days_per_year(ticker, cfg) == 365
    return CRASHES_BTC if is_btc else CRASHES_ETF


def crash_table(ticker, cfg, dates, paths_by_series) -> dict:
    """Window metadata + the max leveraged drawdown inside each window (OOS days only)."""
    dates = pd.DatetimeIndex(dates)
    out = []
    for key, label, a, b in crash_windows(ticker, cfg):
        a, b = pd.Timestamp(a), pd.Timestamp(b)
        sel = np.flatnonzero((dates >= a) & (dates <= b))
        o_start, o_end = max(a, dates[0]), min(b, dates[-1])
        cover = max(0.0, ((o_end - o_start).days + 1) / ((b - a).days + 1)) if o_end >= o_start else 0.0
        entry = dict(key=key, label=label, start=str(a.date()), end=str(b.date()),
                     n_oos_days=int(sel.size), calendar_coverage=round(float(cover), 6),
                     fully_covered=bool(dates[0] <= a and dates[-1] >= b), max_drawdown={})
        for s, per_L in paths_by_series.items():
            entry["max_drawdown"][s] = {}
            for L, (r, _) in per_L.items():
                if sel.size == 0:
                    v = None
                else:
                    eq = np.concatenate([[1.0], np.cumprod(1.0 + r[sel])])
                    v = float(np.min(eq / np.maximum.accumulate(eq) - 1.0))
                entry["max_drawdown"][s][f"{L:g}"] = v
        out.append(entry)
    return dict(ticker=ticker, oos_start=str(dates[0].date()), oos_end=str(dates[-1].date()),
                windows=out)


# --------------------------------------------------------------------------- orchestration
def _atomic_parquet(df, path: Path):
    tmp = path.with_suffix(".tmp")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def _atomic_text(text, path: Path):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def run_asset(ticker, cfg) -> dict:
    """Simulate one asset from saved results and write the four output files."""
    inputs, info = load_inputs(ticker, cfg)
    dpy = days_per_year(ticker, cfg)
    cost = float(cfg["costs"][ticker])
    borrow = float(cfg["borrow_rate"])
    base_dates = inputs["buy_and_hold"]["date"]
    for k, d in inputs.items():
        if not pd.DatetimeIndex(d["date"]).equals(pd.DatetimeIndex(base_dates)):
            raise ValueError(f"{ticker}: series '{k}' dates differ from buy_and_hold dates")
    dates = pd.DatetimeIndex(base_dates)
    rows, eq_parts, paths_by_series = [], [], {}
    for s in SERIES:
        d = inputs[s]
        srows, paths = simulate_series(dates, d["gross_return"].to_numpy(), d["position"].to_numpy(),
                                       cost, borrow, dpy)
        paths_by_series[s] = paths
        for row in srows:
            rows.append(dict(ticker=ticker, series=s,
                             config_id=info["config_id"] if s == "chosen_strategy" else s,
                             cost=cost, borrow_rate=borrow, days_per_year=dpy, **row))
            eq_parts.append(pd.DataFrame(dict(ticker=ticker, series=s, L=row["L"], date=dates,
                                              equity=paths[row["L"]][1])))
    lev = pd.DataFrame(rows)
    eq = pd.concat(eq_parts, ignore_index=True)
    pre = load_preoos_returns(ticker, cfg, dates[0], info["cutoff"])
    est = kelly_estimate(pre.to_numpy(), dpy, borrow)
    kel = kelly_table(ticker, est, borrow, pre.index[0], pre.index[-1])
    crashes = crash_table(ticker, cfg, dates, paths_by_series)
    crashes.update(chosen_config_id=info["config_id"], chosen_model=info["chosen_model"],
                   chosen_note=info["chosen_note"], borrow_rate=borrow, days_per_year=dpy,
                   cost=cost, what_if_only=True)
    out = _results_dir(cfg) / ticker
    out.mkdir(parents=True, exist_ok=True)
    _atomic_parquet(lev, out / "leverage.parquet")
    _atomic_parquet(eq, out / "leverage_equity.parquet")
    _atomic_parquet(kel, out / "kelly.parquet")
    _atomic_text(json.dumps(crashes, indent=2, sort_keys=True, default=str) + "\n",
                 out / "crashes.json")
    return dict(ticker=ticker, leverage=lev, equity=eq, kelly=kel, kelly_est=est, crashes=crashes,
                info=info)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Leverage what-if simulator (saved OOS results only).")
    ap.add_argument("--config", default=None)
    ap.add_argument("--assets", nargs="*", default=None)
    args = ap.parse_args(argv)
    try:
        cfg = load_config(args.config)
        tickers = args.assets or [a["ticker"] for a in cfg["assets"]]
        unknown = [t for t in tickers if t not in cfg["costs"]]
        if unknown:
            raise ValueError(f"unknown asset(s) {unknown}; config has {sorted(cfg['costs'])}")
        results = [run_asset(t, cfg) for t in tickers]
    except (MissingResultsError, ValueError, FileNotFoundError) as e:
        print(f"ERROR: {e.args[0] if e.args else e}", file=sys.stderr)
        raise SystemExit(2)
    for r in results:
        est, lev = r["kelly_est"], r["leverage"]
        print(f"\n{r['ticker']}  (chosen: {r['info']['config_id']}"
              f"{'; ' + r['info']['chosen_note'] if r['info']['chosen_note'] else ''})")
        print(f"  Kelly L* = {est['L_star']:.2f}  (95% band {est['L_star_lo']:.2f} - "
              f"{est['L_star_hi']:.2f}; mu={est['mu']:.3f}, sigma={est['sigma']:.3f}, "
              f"{est['n_days']} pre-OOS days)")
        print(lev[["series", "L", "cagr", "max_drawdown", "wiped_out"]]
              .to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    return results


if __name__ == "__main__":
    main()
