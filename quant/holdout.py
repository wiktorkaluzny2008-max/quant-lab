"""Stage 6 one-shot hold-out evaluation harness (FAKE money, research only, no broker).

Run:
    uv run python -m quant.holdout --dry-run [--pseudo-cutoff YYYY-MM-DD] [--assets SPY ...] [--out-dir DIR]
    uv run python -m quant.holdout --run --confirm-single-final-evaluation
    uv run python -m quant.holdout --significance-summary
    uv run python -m quant.holdout --verify-preregistration

WHAT IT DOES
------------
Evaluates the ONE pinned strategy (config.yaml ``leverage_strategy``: expanding window, ensemble of
logreg+rf+lgbm, long_flat, vol-targeted, adaptive margin m) on the final hold-out
(decision days from the pinned cutoff, outcomes realised through cutoff + holdout_months), plus
baselines (buy and hold, vol-targeted buy and hold, always up, trend_200, climatology) on the
same days, at headline and stress costs.  NOTHING is tuned: it reuses ``backtest.walk_forward``,
``backtest.adaptive_positions`` and ``backtest.evaluate`` unchanged.

WINDOW CONVENTION
-----------------
Stage 3 blanked the label of the last dev row (its next-day return is realised inside the
hold-out).  The hold-out therefore scores decision rows with date >= cutoff: dev and hold-out
partition all labelled rows with no overlap and no lost day.  The last row of the data (date =
end) has no label (blanked on load), so outcomes are realised through ``end``.
Each series starts the hold-out in cash and pays for its first entry (same for baselines).

NO LOOKAHEAD / CAUSALITY
------------------------
The walk-forward retrains monthly on rows <= s-2 (1-day gap) exactly as in Stage 3/4.  The
adaptive margin m of every block uses only out-of-sample days strictly before the block: the
saved Stage 3 history (dates < cutoff) followed by the hold-out days as they are realised.
Data after ``end`` is never loaded (parquet filter on load).

ONE-SHOT GUARD (--run)
----------------------
* requires ``--confirm-single-final-evaluation``;
* refuses if ``<data>/results/holdout/HOLDOUT_DONE.json`` exists (any status) or outputs exist;
* refuses (BEFORE the marker, so the one shot is not burnt) if the data do not cover the whole
  hold-out;
* writes the marker FIRST (status "started": timestamp, SHA-256 of quant/*.py, config.yaml,
  uv.lock, every Stage 3/4 result file, hold-out range and row counts); the marker is created
  atomically and exclusively (``os.link``), then updated to "done" or "failed";
* a crash leaves status "failed" (or "started" if the process was killed) and later runs
  REFUSE.  Manual step to allow a re-run: read the failure, rename the marker
  (e.g. HOLDOUT_DONE.json -> HOLDOUT_FAILED_<date>.json) AND write the reason in PROGRESS.md.
  A re-run after a failure is, by definition, no longer a single evaluation: say so in the report.
* the freeze manifest is re-verified after the run (code, config, lock, results, ledger);
* `--run` refuses `--config` (only PROJECT_ROOT/config.yaml is allowed) and asserts the pinned values
  (cutoff 2024-11-06, 24 months, the declared strategy, the project's own data/ folder) and that the
  SHA-256 of config.yaml equals PREREGISTERED_SHA256 (pre-registration, see PROGRESS.md);
* ALL preflight (lightgbm import, history, frames, coverage, strategy lookups, pre-registration)
  happens BEFORE the marker: a preflight failure does not burn the shot.

INTERPRETATION RULE (pre-registered, see PROGRESS.md "Stage 6 pre-registration")
-------------------------------------------------------------------------------
The hold-out can only DOWNGRADE evidence, never upgrade it.  ``hold_out_flags`` lists failure
signals; ``final_verdict`` maps (development significance numbers, flags) to one of
"no evidence of an edge" (default) / "weak evidence" / "evidence".  "weak evidence" comes ONLY from
tests against a baseline (SPA, excess DSR, excess t); the strategy's own t > 3 and the global DSR are
necessary for "evidence" but never grounds for "weak evidence".  ``headline_verdict`` = the WEAKEST
of the per-asset verdicts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from quant import backtest as B
from quant import config as CONFIG
from quant import features as F
from quant import metrics as M
from quant import models as MD
from quant import paper as P
from quant import trials as T

BT = B.BTConfig()                  # tests monkeypatch this with small settings
MARKER_NAME = "HOLDOUT_DONE.json"
HOLDOUT_DIRNAME = "holdout"
OUTPUT_FILES = ("holdout_metrics.parquet", "holdout_predictions.parquet", "holdout_returns.parquet",
                "holdout_summary.json", "holdout_results.md")
Z95 = 1.959963984540054
BASELINES = B.BASELINES            # buy_and_hold, always_up, vol_targeted_bh, trend_200
REFERENCES = ("buy_and_hold", "vol_targeted_bh")
STRATEGY = "strategy"

# ---- pre-registration (declared BEFORE the hold-out was looked at; tests monkeypatch these) -------
PROJECT_ROOT = CONFIG.PROJECT_ROOT
DATA_DIR = PROJECT_ROOT / "data"
PINNED_CUTOFF = "2024-11-06"
PINNED_MONTHS = 24
PINNED_STRATEGY = {"window": "expanding", "model": "ensemble", "mode": "long_flat",
                   "vol_targeted": True, "m_setting": T.ADAPTIVE}
PREREGISTERED_SHA256 = {"config.yaml": "16509415a26b53a40475e6ee7f3a4d2a2aeae26267a6ceaf0845d62b0c416918"}
# combined SHA-256 of quant/*.py EXCLUDING holdout.py (see code_sha256); holdout.py's own hash is in the run marker
PREREGISTERED_CODE_SHA256 = "123064d95c59dab6a2a442284176a673defabde3b37252341dd08f2fa75627ba"
VERDICT_NONE, VERDICT_WEAK, VERDICT_EVIDENCE = "no evidence of an edge", "weak evidence", "evidence"
VERDICT_ORDER = (VERDICT_NONE, VERDICT_WEAK, VERDICT_EVIDENCE)      # weakest first
VERDICT_CAVEATS = (
    "Verdict caveat 1: the strategy's Deflated Sharpe is GLOBAL (270 trials) but the excess-over-vol-targeted-B&H "
    "Deflated Sharpe is ASSET-scoped (about 54 to 106 trials), so the two are not equally strict.",
    "Verdict caveat 2: Hansen's SPA is a FAMILY-level test (is the best of all tried strategies better than the "
    "baseline?); it is not specific to the chosen strategy.",
    "Verdict caveat 3: t > 3 is a test against ZERO, which buy and hold also passes; it is necessary for "
    "\"evidence\" but never grounds for \"weak evidence\".",
    "Headline rule (pre-registered): the headline verdict is the WEAKEST of the per-asset verdicts "
    "(no evidence of an edge < weak evidence < evidence).")


class HoldoutRefusal(Exception):
    """The harness refuses to run (guard, coverage, bad arguments).  Nothing was started."""


class FreezeViolation(Exception):
    """A frozen file changed during the run."""


# =========================================================================== window
@dataclass(frozen=True)
class Window:
    cutoff: pd.Timestamp       # first decision date scored; history is strictly before
    end: pd.Timestamp          # last realised-return date; no later row is ever loaded
    pseudo: bool


def real_window(cfg) -> Window:
    c = pd.Timestamp(cfg["holdout_cutoff_date"])
    return Window(c, c + pd.DateOffset(months=int(cfg["holdout_months"])), False)


def pseudo_window(cfg, pseudo_cutoff=None) -> Window:
    """A pseudo hold-out lying ENTIRELY inside development data (end <= the PINNED real cutoff).
    The guard uses the module constants, never a (possibly edited or foreign) config value."""
    real = pd.Timestamp(PINNED_CUTOFF)
    months = int(PINNED_MONTHS)
    pc = real - pd.DateOffset(months=months) if pseudo_cutoff is None else pd.Timestamp(pseudo_cutoff)
    end = pc + pd.DateOffset(months=months)
    if end > real:
        raise HoldoutRefusal(
            f"dry run refused: pseudo-cutoff {pc.date()} + {months} months = {end.date()} is beyond "
            f"the real hold-out cutoff {real.date()}; the pseudo hold-out must lie inside development "
            f"data (latest allowed pseudo-cutoff {(real - pd.DateOffset(months=months)).date()})")
    return Window(pc, end, True)


def resolves_to(a, b) -> bool:
    return Path(a).resolve() == Path(b).resolve()


def assert_pinned_cfg(cfg):
    """Refuse unless the config carries exactly the pre-registered values (run_final only)."""
    if str(cfg["holdout_cutoff_date"]) != PINNED_CUTOFF:
        raise HoldoutRefusal(f"holdout_cutoff_date {cfg['holdout_cutoff_date']!r} != pinned {PINNED_CUTOFF}")
    if int(cfg["holdout_months"]) != PINNED_MONTHS:
        raise HoldoutRefusal(f"holdout_months {cfg['holdout_months']!r} != pinned {PINNED_MONTHS}")
    s = pinned_strategy(cfg)
    want = {**PINNED_STRATEGY, "m_setting": str(PINNED_STRATEGY["m_setting"])}
    if {k: s.get(k) for k in want} != want or set(s) != set(want):
        raise HoldoutRefusal(f"leverage_strategy {s} != pre-registered strategy {want}")
    if not resolves_to(cfg["data_dir"], DATA_DIR):
        raise HoldoutRefusal(f"data_dir {cfg['data_dir']} is not the project's data folder {DATA_DIR}")
    if not resolves_to(cfg["project_root"], PROJECT_ROOT):
        raise HoldoutRefusal(f"project_root {cfg['project_root']} is not {PROJECT_ROOT}")


def live_config_sha256(cfg=None) -> str:
    root = Path(cfg["project_root"]) if cfg is not None else PROJECT_ROOT
    return sha256_file(root / "config.yaml")


def code_sha256(cfg=None) -> str:
    """Combined SHA-256 of every quant/*.py file EXCEPT holdout.py (sorted by name; digest of the
    name -> file-hash map).  holdout.py's own hash is recorded in the run marker."""
    root = Path(cfg["project_root"]) if cfg is not None else PROJECT_ROOT
    files = {p.name: sha256_file(p) for p in sorted((root / "quant").glob("*.py")) if p.name != "holdout.py"}
    return manifest_digest(files)


def verify_preregistration(cfg=None) -> dict:
    """Refuse when the live SHA-256 of config.yaml or the combined code hash differs from the
    pre-registered constants."""
    live, want = live_config_sha256(cfg), PREREGISTERED_SHA256["config.yaml"]
    if live != want:
        raise HoldoutRefusal(f"config.yaml changed since pre-registration: live SHA-256 {live} != "
                             f"pre-registered {want}.  The hold-out is not evaluated with an edited config.")
    code, want_code = code_sha256(cfg), PREREGISTERED_CODE_SHA256
    if code != want_code:
        raise HoldoutRefusal(f"quant/*.py (excluding holdout.py) changed since pre-registration: live combined "
                             f"SHA-256 {code} != pre-registered {want_code}.")
    return {"config.yaml": live, "code_sha256": code}


def assert_no_rows_after(frame: pd.DataFrame, end, what="frame"):
    if len(frame) and pd.Timestamp(frame.index.max()) > pd.Timestamp(end):
        raise HoldoutRefusal(f"{what} contains rows after the allowed end {pd.Timestamp(end).date()} "
                             f"(max {frame.index.max().date()})")


def _features_path(ticker, cfg) -> Path:
    return Path(cfg["data_dir"]) / "features" / f"{ticker}.parquet"


def read_dates(ticker, cfg, end) -> pd.DatetimeIndex:
    """Row dates only (no labels, no features) up to ``end``."""
    df = pd.read_parquet(_features_path(ticker, cfg), columns=[],
                         filters=[("Date", "<=", pd.Timestamp(end))])
    return pd.DatetimeIndex(df.index)


def load_frame(ticker, cfg, end) -> pd.DataFrame:
    """Dataset rows with date <= end ONLY (parquet filter: later rows are never read).  The last
    row's label ends after ``end`` (or is unknown), so it is blanked."""
    df = pd.read_parquet(_features_path(ticker, cfg), filters=[("Date", "<=", pd.Timestamp(end))])
    assert_no_rows_after(df, end, f"{ticker} dataset")
    if len(df) == 0:
        raise HoldoutRefusal(f"{ticker}: no data on or before {pd.Timestamp(end).date()}")
    df = df.copy()
    for c in F.TARGET_COLS:
        df.iloc[-1, df.columns.get_loc(c)] = np.nan
    return df


def expected_dates(ticker, start_exclusive, end) -> pd.DatetimeIndex:
    """Dates that must have a row after ``start_exclusive`` up to ``end``: every calendar day for
    BTC, every NYSE session for the ETFs."""
    s, e = pd.Timestamp(start_exclusive) + pd.Timedelta(days=1), pd.Timestamp(end)
    if ticker == F.BTC:
        return pd.date_range(s, e, freq="D")
    ses = F._sessions()
    return ses[(ses >= s) & (ses <= e)]


def check_coverage(ticker, dates: pd.DatetimeIndex, w: Window) -> dict:
    """Raise HoldoutRefusal unless the data cover the whole window; return the row counts."""
    need = expected_dates(ticker, w.cutoff, w.end)
    missing = need.difference(dates)
    if len(need) == 0 or len(missing):
        raise HoldoutRefusal(
            f"{ticker}: the data do not cover the whole hold-out {w.cutoff.date()}..{w.end.date()}: "
            f"{len(missing)} of {len(need)} required days are missing"
            + (f" (first missing {missing[0].date()}, last data row {dates.max().date()})" if len(missing) else ""))
    inwin = dates[dates >= w.cutoff]
    return dict(first_decision_date=str(inwin.min().date()), last_data_date=str(dates.max().date()),
                n_rows_total=int(len(dates)), n_rows_in_window=int(len(inwin)),
                n_scorable_rows=int(len(inwin) - 1))


def scored_dates(ticker, cutoff, end) -> pd.DatetimeIndex:
    """Every decision date that must be scored: all NYSE sessions (ETFs) / calendar days (BTC) in
    [cutoff, end] except the last one (its next-day return is unknown, label blanked)."""
    full = expected_dates(ticker, pd.Timestamp(cutoff) - pd.Timedelta(days=1), end)
    return full[:-1]


def last_date_before(ticker, cutoff) -> pd.Timestamp:
    c = pd.Timestamp(cutoff)
    if ticker == F.BTC:
        return c - pd.Timedelta(days=1)
    ses = F._sessions()
    return ses[ses < c].max()


def hits_from_accuracy(acc, n) -> int:
    """Number of correct calls from accuracy*n.  ROUND, never truncate (0.29*100 = 28.999...)."""
    return int(round(float(acc) * int(n)))


def check_frame_shape(ticker, frame: pd.DataFrame, w: Window):
    missing = [c for c in F.TARGET_COLS if c not in frame.columns]
    if missing:
        raise HoldoutRefusal(f"{ticker}: dataset lacks columns {missing}")
    exp = expected_dates(ticker, w.cutoff, w.end)
    if len(exp) and pd.Timestamp(frame.index.max()) != exp.max():
        raise HoldoutRefusal(f"{ticker}: last data row {frame.index.max().date()} != last expected "
                             f"{exp.max().date()}")


def check_history_contiguity(ticker, hist: "History", w: Window):
    """The saved history must end on the last expected date before the cutoff AND contain every
    expected calendar day (NYSE session / calendar day) from its first to its last date: no gap, no overlap."""
    if not len(hist.dates):
        return
    last = last_date_before(ticker, w.cutoff)
    if hist.dates.max() != last:
        raise AssertionError(f"{ticker}: history ends {hist.dates.max().date()} but the last expected date "
                             f"before the cutoff is {last.date()} (gap/overlap)")
    exp = expected_dates(ticker, hist.dates.min() - pd.Timedelta(days=1), last)
    if not hist.dates.equals(exp):
        miss, extra = exp.difference(hist.dates), hist.dates.difference(exp)
        raise AssertionError(f"{ticker}: history is not contiguous: {len(miss)} expected day(s) missing"
                             + (f" (first {miss[0].date()})" if len(miss) else "")
                             + f", {len(extra)} unexpected day(s) (gap/overlap)")


# =========================================================================== stage 3 history
@dataclass
class History:
    dates: pd.DatetimeIndex
    p: np.ndarray
    base: np.ndarray
    nr: np.ndarray
    w: np.ndarray
    block: pd.DatetimeIndex


def load_history(ticker, strategy, results_dir, before, real_cutoff) -> History:
    """Saved Stage 3 out-of-sample days of the pinned model/window with date < ``before``."""
    rd = Path(results_dir) / ticker
    try:
        pred = pd.read_parquet(rd / "predictions.parquet")
        base = pd.read_parquet(rd / "baselines.parquet")
    except (FileNotFoundError, OSError) as e:
        raise HoldoutRefusal(f"{ticker}: Stage 3 results not found in {rd}") from e
    pred = pred[(pred["window"] == strategy["window"]) & (pred["model"] == strategy["model"])]
    pred = pred[pred["date"] < pd.Timestamp(before)].sort_values("date").reset_index(drop=True)
    w = base[(base["window"] == strategy["window"]) & (base["name"] == "vol_targeted_bh")]
    w = w[w["date"] < pd.Timestamp(before)].sort_values("date").reset_index(drop=True)
    P.assert_dev_dates(pred["date"], real_cutoff, "Stage 3 history")
    if not np.array_equal(pred["date"].to_numpy(), w["date"].to_numpy()):
        raise HoldoutRefusal(f"{ticker}: Stage 3 predictions and baselines are misaligned")
    return History(pd.DatetimeIndex(pred["date"]), pred["p_up"].to_numpy(float),
                   pred["base_rate"].to_numpy(float), pred["next_return"].to_numpy(float),
                   w["position"].to_numpy(float), pd.DatetimeIndex(pred["block"]))


# =========================================================================== statistics
def wilson_interval(k, n, z=Z95):
    """Wilson score interval for a binomial proportion k/n."""
    if n <= 0:
        return float("nan"), float("nan")
    ph = k / n
    den = 1 + z * z / n
    centre = (ph + z * z / (2 * n)) / den
    half = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / den
    return centre - half, centre + half


def sharpe_difference(net_a, net_b, ppy) -> dict:
    """SR_a - SR_b (annualised) with the Jobson-Korkie (1981) standard error as corrected by
    Memmel (2003): var = [2(1-rho) + 0.5 (SRa^2 + SRb^2 - 2 SRa SRb rho^2)] / n  (per-period SRs)."""
    a, b = np.asarray(net_a, float), np.asarray(net_b, float)
    n = len(a)
    out = dict(diff=float("nan"), se=float("nan"), z=float("nan"), lo=float("nan"), hi=float("nan"))
    sa, sb = M.sharpe(a, 1), M.sharpe(b, 1)
    if n < 3 or not (np.isfinite(sa) and np.isfinite(sb)):
        return out
    rho = float(np.corrcoef(a, b)[0, 1])
    v = (2 * (1 - rho) + 0.5 * (sa ** 2 + sb ** 2 - 2 * sa * sb * rho ** 2)) / n
    d = (sa - sb) * math.sqrt(ppy)
    if v < 0:
        return dict(out, diff=d)
    se = math.sqrt(v) * math.sqrt(ppy)
    if se <= 0:
        return dict(out, diff=d, se=se)
    return dict(diff=d, se=se, z=d / se, lo=d - Z95 * se, hi=d + Z95 * se)


def power_note(ppy, n) -> dict:
    """Why 24 months cannot prove an edge: Lo (2002) iid SE of an annualised Sharpe at SR~0 is
    sqrt(ppy/n) (about 0.71 for 24 months); the smallest true Sharpe detected with 80% power by a
    one-sided 5% test is (1.645 + 0.842) * SE."""
    se = math.sqrt(ppy / n) if n > 0 else float("nan")
    return dict(n_days=int(n), sharpe_se_at_zero=se, min_detectable_sharpe_80pct_power=(1.6449 + 0.8416) * se,
                note=("The hold-out's job is to catch failure, not to prove success: with "
                      f"{n} days the Sharpe standard error is about {se:.2f}, so only a true Sharpe above "
                      f"about {(1.6449 + 0.8416) * se:.1f} would be detected reliably (80% power, one-sided 5%).")
                if n > 0 else "no days")


def scenario_series(nr, pos, cost, delay):
    pos_eff = B.apply_delay(pos) if delay else np.asarray(pos, float)
    return pos_eff, M.net_returns(nr, pos_eff, cost)


# =========================================================================== evaluation core
@dataclass
class AssetEval:
    ticker: str
    config_id: str
    ppy: int
    dates: pd.DatetimeIndex
    prior_position: float
    trading: pd.DataFrame
    prediction: pd.DataFrame
    predictions: pd.DataFrame
    returns: pd.DataFrame
    power: dict


def pinned_strategy(cfg) -> dict:
    s = dict(cfg["leverage_strategy"])
    if s["window"] != "expanding":
        raise HoldoutRefusal("the hold-out harness supports only the expanding window")
    if s["mode"] not in B.SIGNALS:
        raise HoldoutRefusal(f"unsupported strategy mode {s['mode']!r}")
    if s["model"] != "ensemble" and s["model"] not in MD.MODEL_NAMES:
        raise HoldoutRefusal(f"unknown model {s['model']!r}")
    return s


def strategy_models(strategy) -> list:
    return list(MD.MODEL_NAMES) if strategy["model"] == "ensemble" else [strategy["model"]]


def strategy_config_id(ticker, s) -> str:
    m = T.ADAPTIVE if s["m_setting"] == T.ADAPTIVE else float(s["m_setting"])
    return T.make_config_id(ticker, s["window"], s["model"], s["mode"], bool(s["vol_targeted"]), m)


def evaluate_asset(ticker, frame, hist: History, strategy, headline_cost, w: Window, bt=None) -> AssetEval:
    """The evaluation core.  ``frame`` = dataset rows <= w.end (last label blanked); ``hist`` = the
    saved Stage 3 out-of-sample days strictly before w.cutoff."""
    bt = bt or BT
    assert_no_rows_after(frame, w.end, f"{ticker} frame")
    ppy = M.periods_per_year(ticker)
    ann = float(ppy)
    models = strategy_models(strategy)
    wf = B.walk_forward(frame, models, bt, w.end, "expanding", ann, importance=False, first_date=w.cutoff)
    preds = wf.preds[wf.preds["date"] >= w.cutoff].copy()
    d0 = preds[preds["model"] == strategy["model"]].sort_values("date")
    if len(d0) == 0:
        raise ValueError(f"{ticker}: no scorable hold-out day")
    dates = pd.DatetimeIndex(d0["date"])
    if len(hist.dates) and hist.dates.max() >= dates.min():
        raise AssertionError("history overlaps the evaluated window")
    exp = scored_dates(ticker, w.cutoff, w.end)
    if not (len(dates) == len(exp) and dates.min() == exp.min() and dates.max() == exp.max()
            and np.array_equal(dates.to_numpy(), exp.to_numpy())):
        raise AssertionError(
            f"{ticker}: scored window mismatch: got {len(dates)} days {dates.min().date()}..{dates.max().date()}, "
            f"expected {len(exp)} days {exp.min().date()}..{exp.max().date()} (a skipped or extra block?)")
    check_history_contiguity(ticker, hist, w)
    nr = d0["next_return"].to_numpy(float)
    bpos = B.baseline_positions(frame, wf.retrains, dates, ann, bt)
    w_new = bpos["vol_targeted_bh"].to_numpy(float)
    vt = bool(strategy["vol_targeted"])
    nh = len(hist.dates)
    p = np.r_[hist.p, d0["p_up"].to_numpy(float)]
    base = np.r_[hist.base, d0["base_rate"].to_numpy(float)]
    nr_all = np.r_[hist.nr, nr]
    w_all = np.r_[hist.w, w_new]
    ids = np.r_[hist.block.to_numpy("datetime64[ns]"), pd.DatetimeIndex(d0["block"]).to_numpy("datetime64[ns]")]
    starts = np.flatnonzero(np.r_[True, ids[1:] != ids[:-1]]).tolist()
    mode = strategy["mode"]
    margin = np.full(len(p), np.nan)
    if strategy["m_setting"] == T.ADAPTIVE:
        pos_all, log = B.adaptive_positions(mode, p, base, nr_all, w_all if vt else None, starts,
                                            headline_cost, bt.adaptive_min_days)
        stops = starts[1:] + [len(p)]
        for lg, stop in zip(log, stops):
            margin[lg["block_row"]:stop] = lg["m"]
    else:
        m = float(strategy["m_setting"])
        pos_all = B._fixed_position(mode, p, base, m, w_all if vt else None)
        margin[:] = m
    pos = pos_all[nh:]
    prior = float(pos_all[nh - 1]) if nh else 0.0
    positions = {STRATEGY: pos}
    for name in BASELINES:
        positions[name] = np.nan_to_num(bpos[name].to_numpy(float), nan=0.0)

    # ---- trading metrics, all scenarios
    scen = B.cost_scenarios(ticker, headline_cost)
    cid = strategy_config_id(ticker, strategy)
    trows, rrows = [], []
    for sname, (cost, delay) in scen.items():
        nets, effs = {}, {}
        for subj, ps in positions.items():
            effs[subj], nets[subj] = scenario_series(nr, ps, cost, delay)
        for subj, ps in positions.items():
            ev = B.evaluate(nr, ps, cost, ppy, delay)
            row = dict(asset=ticker, kind="trading", subject=subj, config_id=cid if subj == STRATEGY else subj,
                       scenario=sname, cost=cost, delay=delay, **ev,
                       sharpe_ci_low=ev["sharpe"] - Z95 * ev["sharpe_se"],
                       sharpe_ci_high=ev["sharpe"] + Z95 * ev["sharpe_se"])
            for ref in REFERENCES:
                sd = (sharpe_difference(nets[subj], nets[ref], ppy) if subj != ref else
                      dict(diff=np.nan, se=np.nan, z=np.nan, lo=np.nan, hi=np.nan))
                row.update({f"sharpe_diff_vs_{ref}": sd["diff"], f"sharpe_diff_se_vs_{ref}": sd["se"],
                            f"sharpe_diff_z_vs_{ref}": sd["z"], f"sharpe_diff_ci_low_vs_{ref}": sd["lo"],
                            f"sharpe_diff_ci_high_vs_{ref}": sd["hi"]})
            trows.append(row)
            rrows.append(pd.DataFrame({"asset": ticker, "subject": subj, "scenario": sname, "date": dates,
                                       "position": effs[subj], "gross_return": effs[subj] * nr,
                                       "net_return": nets[subj]}))

    # ---- prediction metrics (all models + climatology), then Wilson CI and eval-base skills
    pm, _, _ = B.prediction_metrics_tables(preds)
    extra = []
    for _, r in pm.iterrows():
        k = hits_from_accuracy(r["accuracy"], r["n"])
        lo, hi = wilson_interval(k, int(r["n"]))
        if r["model"] == "climatology":
            d = d0
            ll_eval = M.log_loss_skill(d["y"].to_numpy(), d["base_rate"].to_numpy(), float(d["y"].mean()))
        else:
            d = preds[preds["model"] == r["model"]]
            ll_eval = M.log_loss_skill(d["y"].to_numpy(), d["p_up"].to_numpy(), float(d["y"].mean()))
        q = (d["base_rate"] if r["model"] == "climatology" else d["p_up"]).to_numpy(float)
        yy = d["y"].to_numpy(float)
        dl = (q - yy) ** 2 - (yy.mean() - yy) ** 2          # >0: worse than the eval-window base rate
        dse = float(dl.std(ddof=1) / math.sqrt(len(dl))) if len(dl) > 1 else float("nan")
        extra.append(dict(hits=k, accuracy_ci_low=lo, accuracy_ci_high=hi, log_loss_skill_vs_eval_base=ll_eval,
                          brier_loss_diff_vs_eval_base=float(dl.mean()), brier_loss_diff_se=dse))
    pm = pd.concat([pm.reset_index(drop=True), pd.DataFrame(extra)], axis=1)
    pm.insert(0, "kind", "prediction")
    pm.insert(0, "asset", ticker)
    pm["is_strategy_model"] = pm["model"] == strategy["model"]

    pr = preds.assign(asset=ticker, position=np.nan, vol_weight=np.nan, margin_m=np.nan)
    sel = (pr["model"] == strategy["model"]).to_numpy()
    order = np.argsort(pr.loc[sel, "date"].to_numpy(), kind="stable")
    idx_sel = np.flatnonzero(sel)[order]
    pr.iloc[idx_sel, pr.columns.get_loc("position")] = pos
    pr.iloc[idx_sel, pr.columns.get_loc("vol_weight")] = w_new
    pr.iloc[idx_sel, pr.columns.get_loc("margin_m")] = margin[nh:]
    return AssetEval(ticker, cid, ppy, dates, prior, pd.DataFrame(trows), pm, pr.reset_index(drop=True),
                     pd.concat(rrows, ignore_index=True), power_note(ppy, len(dates)))


# =========================================================================== interpretation rule
def _isnum(x) -> bool:
    return x is not None and isinstance(x, (int, float, np.floating)) and bool(np.isfinite(x))


def hold_out_flags(a: dict) -> dict:
    """Pre-registered hold-out FAILURE SIGNALS for one asset (flags only, no verdict).

    The hold-out can only downgrade evidence.  A positive profit or Sharpe difference is never
    sufficient and is not a flag.  Failure signals (at the headline cost):
      1. Sharpe difference vs vol-targeted buy and hold: its 95% CI lies entirely below 0;
      2. Pesaran-Timmermann statistic < 0 with p < 0.05 (significantly NEGATIVE directional skill);
      3. the strategy model's Brier score is significantly WORSE than the evaluation-window base rate:
         the 95% CI of the mean paired squared-error difference (model minus constant base rate) lies
         entirely above 0.
    ``a`` keys: sharpe_diff_vs_vol_targeted_bh_ci [lo, hi], pt_statistic, pt_pvalue, brier_loss_diff_ci [lo, hi].
    A missing/NaN input cannot raise a flag (it is listed as 'undetermined')."""
    reasons, undetermined = [], []
    sd = a.get("sharpe_diff_vs_vol_targeted_bh_ci") or [float("nan")] * 2
    f1 = _isnum(sd[1]) and sd[1] < 0
    if f1:
        reasons.append(f"Sharpe difference vs vol-targeted B&H has its 95% CI entirely below 0 [{sd[0]:.2f}, {sd[1]:.2f}]")
    elif not _isnum(sd[1]):
        undetermined.append("sharpe_diff_vs_vol_targeted_bh")
    pts, ptp = a.get("pt_statistic"), a.get("pt_pvalue")
    f2 = _isnum(pts) and _isnum(ptp) and pts < 0 and ptp < 0.05
    if f2:
        reasons.append(f"Pesaran-Timmermann shows negative skill (statistic {pts:.2f}, p {ptp:.3f})")
    elif not (_isnum(pts) and _isnum(ptp)):
        undetermined.append("pesaran_timmermann")
    bd = a.get("brier_loss_diff_ci") or [float("nan")] * 2
    f3 = _isnum(bd[0]) and bd[0] > 0
    if f3:
        reasons.append(f"Brier score significantly worse than the evaluation-window base rate "
                       f"(95% CI of the loss difference [{bd[0]:.4f}, {bd[1]:.4f}] above 0)")
    elif not _isnum(bd[0]):
        undetermined.append("brier_vs_eval_base")
    return dict(sharpe_vs_vt_failure=bool(f1), pt_negative_skill=bool(f2), brier_worse_than_base=bool(f3),
                any_failure=bool(f1 or f2 or f3), reasons=reasons, undetermined=undetermined)


def final_verdict(sn: dict, flags: dict) -> tuple:
    """(verdict, reasons) from the saved development numbers ``sn`` (one asset of summary_numbers) and
    the hold-out flags.  Default "no evidence of an edge".

    BASELINE tests (against a baseline; the only possible grounds for "weak evidence"):
      B1 SPA p < 0.05 vs BOTH buy and hold and vol-targeted B&H;
      B2 excess-over-vol-targeted-B&H Deflated Sharpe (asset-scoped) >= 0.95;
      B3 t > 3 for the excess return over vol-targeted B&H.
    NECESSARY conditions for "evidence" (never grounds for "weak evidence"; buy and hold passes t > 3):
      N1 t > 3 for the strategy itself;  N2 global (270-trial) Deflated Sharpe >= 0.95.
    "evidence" = B1, B2, B3, N1 and N2 all cleared AND no hold-out failure signal;
    "weak evidence" = at least one baseline test cleared, not "evidence", AND no failure signal;
    otherwise (and ALWAYS when any hold-out failure signal fired) "no evidence of an edge".
    The hold-out never upgrades: only the failure flags enter."""
    def lt(x, thr):
        return _isnum(x) and x < thr

    def ge(x, thr):
        return _isnum(x) and x >= thr

    def gt(x, thr):
        return _isnum(x) and x > thr

    def fm(x, d=3):
        return f"{x:.{d}f}" if _isnum(x) else "n/a"
    spa = [sn.get("spa_pvalue_vs_buy_and_hold"), sn.get("spa_pvalue_vs_vol_targeted_bh")]
    dsr_g, dsr_x = sn.get("dsr_global"), sn.get("dsr_excess_over_vol_targeted_bh")
    t_s, t_x = sn.get("tstat"), sn.get("tstat_excess_over_vol_targeted_bh")
    baseline = [
        ("baseline test: SPA p < 0.05 vs both baselines (family-level test, not specific to this strategy)",
         all(lt(x, 0.05) for x in spa), f"p = {fm(spa[0])} / {fm(spa[1])}"),
        ("baseline test: excess-over-vol-targeted-B&H DSR >= 0.95 (asset-scoped, about 54 to 106 trials)",
         ge(dsr_x, 0.95), f"DSR {fm(dsr_x)}"),
        ("baseline test: t > 3 for the excess over vol-targeted B&H", gt(t_x, 3), f"t = {fm(t_x, 2)}")]
    necessary = [
        ("necessary condition: t > 3 for the strategy (a test against zero; buy and hold also passes it)",
         gt(t_s, 3), f"t = {fm(t_s, 2)}"),
        ("necessary condition: global Deflated Sharpe >= 0.95 (270 trials)", ge(dsr_g, 0.95), f"DSR {fm(dsr_g)}")]
    reasons = [f"{'CLEARED' if ok else 'not cleared'}: {name} ({val})" for name, ok, val in baseline + necessary]
    fail = bool(flags.get("any_failure"))
    if fail:
        reasons.append("hold-out failure signal(s): " + "; ".join(flags.get("reasons", [])))
    n_base = sum(ok for _, ok, _ in baseline)
    all_ok = n_base == len(baseline) and all(ok for _, ok, _ in necessary)
    if fail:
        verdict = VERDICT_NONE
        if n_base:
            reasons.append("a hold-out failure signal downgrades the verdict to the default")
    elif all_ok:
        verdict = VERDICT_EVIDENCE
    elif n_base:
        verdict = VERDICT_WEAK
    else:
        verdict = VERDICT_NONE
        reasons.append("no baseline test cleared: the strategy's own t > 3 / global DSR alone never give "
                       "\"weak evidence\"")
    return verdict, reasons


def headline_verdict(per_asset: dict) -> str:
    """The headline verdict = the WEAKEST per-asset verdict (none < weak < evidence); "no evidence of
    an edge" when there is no verdict at all.  Pre-registered aggregation."""
    vs = list(per_asset.values())
    bad = [v for v in vs if v not in VERDICT_ORDER]
    if bad:
        raise ValueError(f"unknown verdict(s) {bad}")
    return min(vs, key=VERDICT_ORDER.index) if vs else VERDICT_NONE


# =========================================================================== all assets
@dataclass
class HoldoutResult:
    window: Window
    strategy: dict
    assets: dict
    metrics: pd.DataFrame
    predictions: pd.DataFrame
    returns: pd.DataFrame
    summary: dict
    markdown: str
    dev_numbers: dict = None
    earlier_attempts: list = field(default_factory=list)


def earlier_failed_attempts(cfg) -> list:
    hd = holdout_dir(cfg)
    return sorted(p.name for p in hd.glob("HOLDOUT_FAILED_*.json")) if hd.exists() else []


def evaluate_all(cfg, w: Window, results_dir=None, assets=None, bt=None, frames=None, hists=None,
                 dev_numbers=None) -> HoldoutResult:
    strategy = pinned_strategy(cfg)
    rd = Path(results_dir) if results_dir else Path(cfg["data_dir"]) / "results"
    real_cut = pd.Timestamp(PINNED_CUTOFF)
    if w.pseudo and w.end > real_cut:
        raise HoldoutRefusal(f"a pseudo window may not end after the pinned cutoff {real_cut.date()}")
    tickers = list(assets) if assets else [a["ticker"] for a in cfg["assets"]]
    evs = {}
    for t in tickers:
        frame = frames[t] if frames else load_frame(t, cfg, w.end)
        hist = hists[t] if hists else load_history(t, strategy, rd, w.cutoff, real_cut)
        evs[t] = evaluate_asset(t, frame, hist, strategy, float(cfg["costs"][t]), w, bt)
    metrics = pd.concat([pd.concat([e.trading for e in evs.values()], ignore_index=True),
                         pd.concat([e.prediction for e in evs.values()], ignore_index=True)], ignore_index=True)
    preds = pd.concat([e.predictions for e in evs.values()], ignore_index=True)
    rets = pd.concat([e.returns for e in evs.values()], ignore_index=True)
    summary = build_summary(cfg, w, strategy, evs, dev_numbers)
    res = HoldoutResult(w, strategy, evs, metrics, preds, rets, summary, "", dev_numbers,
                        [] if w.pseudo else earlier_failed_attempts(cfg))
    res.markdown = render_markdown(res)
    return res


def build_summary(cfg, w, strategy, evs, dev_numbers=None) -> dict:
    head = {}
    for t, e in evs.items():
        tr = e.trading[e.trading["scenario"] == "headline"].set_index("subject")
        pm = e.prediction.set_index("model")
        sm = pm.loc[strategy["model"]]
        head[t] = dict(
            config_id=e.config_id, n_days=int(len(e.dates)), first_decision_date=str(e.dates.min().date()),
            last_decision_date=str(e.dates.max().date()), prior_position=e.prior_position,
            strategy_sharpe=float(tr.loc[STRATEGY, "sharpe"]),
            strategy_sharpe_ci=[float(tr.loc[STRATEGY, "sharpe_ci_low"]), float(tr.loc[STRATEGY, "sharpe_ci_high"])],
            sharpe_diff_vs_buy_and_hold=float(tr.loc[STRATEGY, "sharpe_diff_vs_buy_and_hold"]),
            sharpe_diff_vs_vol_targeted_bh=float(tr.loc[STRATEGY, "sharpe_diff_vs_vol_targeted_bh"]),
            accuracy=float(sm["accuracy"]), accuracy_ci=[float(sm["accuracy_ci_low"]), float(sm["accuracy_ci_high"])],
            auc=float(sm["auc"]), brier_skill=float(sm["brier_skill"]),
            brier_skill_vs_eval_base=float(sm["brier_skill_vs_eval_base"]),
            pt_statistic=float(sm["pt_statistic"]), pt_pvalue=float(sm["pt_pvalue"]),
            sharpe_diff_vs_vol_targeted_bh_ci=[float(tr.loc[STRATEGY, "sharpe_diff_ci_low_vs_vol_targeted_bh"]),
                                               float(tr.loc[STRATEGY, "sharpe_diff_ci_high_vs_vol_targeted_bh"])],
            brier_loss_diff_ci=[float(sm["brier_loss_diff_vs_eval_base"] - Z95 * sm["brier_loss_diff_se"]),
                                float(sm["brier_loss_diff_vs_eval_base"] + Z95 * sm["brier_loss_diff_se"])],
            pct_invested=float(tr.loc[STRATEGY, "pct_invested"]),
            power=e.power)
        head[t]["hold_out_flags"] = hold_out_flags(head[t])
        if dev_numbers and t in dev_numbers.get("assets", {}):
            v, why = final_verdict(dev_numbers["assets"][t], head[t]["hold_out_flags"])
            head[t]["verdict"], head[t]["verdict_reasons"] = v, why
    per_asset = {t: h["verdict"] for t, h in head.items() if "verdict" in h}
    return dict(
        per_asset_verdicts=per_asset, headline_verdict=headline_verdict(per_asset) if per_asset else None,
        dry_run=w.pseudo, cutoff=str(w.cutoff.date()), end=str(w.end.date()),
        holdout_months=int(cfg["holdout_months"]), pinned_strategy=strategy,
        scoring_convention=("decision rows with date >= cutoff; outcomes realised through end; each series "
                            "starts in cash and pays its first entry"),
        costs={t: float(cfg["costs"][t]) for t in evs}, assets=head,
        notes=["The hold-out catches failure; it does not prove success. It can only DOWNGRADE evidence; a "
               "positive hold-out profit or Sharpe difference alone is never sufficient.",
               "ETF stress is ONLY the 1-day delay: backtest.cost_scenarios has no ETF cost above the headline "
               "(low_cost is a sensitivity below it) and NO higher-cost ETF scenario was pre-registered, so none "
               "is shown (adding one now would be a new unlogged setting). BTC also has the 0.60% stress cost.",
               *VERDICT_CAVEATS,
               "sigma_target of vol targeting is the training-window volatility at each monthly retrain "
               "(identical for the strategy and the baseline, as in Stage 3/4)."])


def _f(x, d=2, pct=False):
    if x is None or not np.isfinite(x):
        return "n/a"
    return f"{x * 100:.{d}f}%" if pct else f"{x:.{d}f}"


def render_markdown(res: HoldoutResult) -> str:
    w = res.window
    title = ("DRY RUN on a PSEUDO hold-out (development data only; NOT the real hold-out)" if w.pseudo
             else "FINAL HOLD-OUT RESULTS (single evaluation)")
    out = [f"# {title}", ""]
    if res.earlier_attempts:
        out += [f"**ATTEMPT {len(res.earlier_attempts) + 1}: NOT a single evaluation.** Earlier attempt file(s): "
                + ", ".join(f"`{n}`" for n in res.earlier_attempts)
                + ". The reasons are recorded in PROGRESS.md; this result must not be presented as one-shot.", ""]
    out += [f"Decision days from {w.cutoff.date()}, outcomes realised through {w.end.date()}. "
           f"Strategy: `{json.dumps(res.strategy, sort_keys=True)}`. Net of costs; Sharpe = sqrt(252) ETFs, "
           "sqrt(365) BTC; 95% CI from Lo's iid standard error.", ""]
    for t, e in res.assets.items():
        tr = e.trading
        out += [f"## {t} ({len(e.dates)} days, {e.dates.min().date()} to {e.dates.max().date()})", "",
                "### Trading at headline cost", "",
                "| Series | Sharpe [95% CI] | CAGR | Max DD | % invested | Trades | Turnover | Sharpe diff vs B&H (SE) | Sharpe diff vs VT B&H (SE) |",
                "|---|---|---|---|---|---|---|---|---|"]
        h = tr[tr["scenario"] == "headline"]
        for _, r in h.iterrows():
            out.append(f"| {r['subject']} | {_f(r['sharpe'])} [{_f(r['sharpe_ci_low'])}, {_f(r['sharpe_ci_high'])}] | "
                       f"{_f(r['cagr'], 1, True)} | {_f(r['max_drawdown'], 1, True)} | {_f(r['pct_invested'], 0, True)} | "
                       f"{int(r['n_trades'])} | {_f(r['turnover'], 3)} | "
                       f"{_f(r['sharpe_diff_vs_buy_and_hold'])} ({_f(r['sharpe_diff_se_vs_buy_and_hold'])}) | "
                       f"{_f(r['sharpe_diff_vs_vol_targeted_bh'])} ({_f(r['sharpe_diff_se_vs_vol_targeted_bh'])}) |")
        out += ["", "### Stress and sensitivity scenarios", "",
                "| Scenario | Cost | Delay | Series | Sharpe [95% CI] | CAGR | Sharpe diff vs B&H (SE) | Sharpe diff vs VT B&H (SE) |",
                "|---|---|---|---|---|---|---|---|"]
        for _, r in tr[(tr["scenario"] != "headline") & tr["subject"].isin([STRATEGY, *REFERENCES])].iterrows():
            out.append(f"| {r['scenario']} | {r['cost'] * 100:.2f}% | {bool(r['delay'])} | {r['subject']} | "
                       f"{_f(r['sharpe'])} [{_f(r['sharpe_ci_low'])}, {_f(r['sharpe_ci_high'])}] | {_f(r['cagr'], 1, True)} | "
                       f"{_f(r['sharpe_diff_vs_buy_and_hold'])} ({_f(r['sharpe_diff_se_vs_buy_and_hold'])}) | "
                       f"{_f(r['sharpe_diff_vs_vol_targeted_bh'])} ({_f(r['sharpe_diff_se_vs_vol_targeted_bh'])}) |")
        btc = t == F.BTC
        cols = [("Model", None), ("Accuracy [Wilson 95%]", None), ("Up-day share", None), ("AUC", None)]
        bs_eval = ("Brier skill (EVAL base; read first)" if btc else "Brier skill (eval base)", "brier_skill_vs_eval_base")
        bs_train = ("Brier skill (train base)", "brier_skill")
        head_cols = [c[0] for c in cols] + [x[0] for x in ((bs_eval, bs_train) if btc else (bs_train, bs_eval))] \
            + ["Log-loss skill (train base)", "PT stat", "PT p", "Call rate"]
        out += ["", "### Prediction quality", "", "| " + " | ".join(head_cols) + " |",
                "|" + "---|" * len(head_cols)]
        for _, r in e.prediction.iterrows():
            name = ("climatology (= always-up accuracy)" if r["model"] == "climatology" else r["model"])
            skills = [f"{_f(r[bs_eval[1]], 4)}", f"{_f(r[bs_train[1]], 4)}"] if btc else \
                [f"{_f(r[bs_train[1]], 4)}", f"{_f(r[bs_eval[1]], 4)}"]
            out.append(f"| {name} | {_f(r['accuracy'], 1, True)} [{_f(r['accuracy_ci_low'], 1, True)}, "
                       f"{_f(r['accuracy_ci_high'], 1, True)}] | {_f(r['base_rate_oos'], 1, True)} | {_f(r['auc'], 3)} | "
                       f"{skills[0]} | {skills[1]} | {_f(r['log_loss_skill'], 4)} | "
                       f"{_f(r['pt_statistic'])} | {_f(r['pt_pvalue'], 3)} | {_f(r['call_rate'], 2)} |")
        out += ["", f"Power: {e.power['note']}", ""]
        sm = res.summary["assets"][t]
        if btc:
            inv = sm["pct_invested"]
            out += [f"BTC note: the strategy was invested on {_f(inv, 0, True)} of days"
                    + (" (mostly in cash), so its comparison with buy and hold is mostly a comparison with "
                       "doing nothing; " if inv < 0.5 else "; ")
                    + "read BTC's Brier skill against the evaluation-window base rate FIRST (the training base "
                    "rate can be far from the hold-out's up-day share).", ""]
        fl = sm["hold_out_flags"]
        out += ["### Hold-out failure signals (pre-registered; flags only)", "",
                f"- Sharpe difference vs vol-targeted B&H, 95% CI entirely below 0: **{fl['sharpe_vs_vt_failure']}**",
                f"- Pesaran-Timmermann negative skill (stat < 0, p < 0.05): **{fl['pt_negative_skill']}**",
                f"- Brier significantly worse than the evaluation-window base rate: **{fl['brier_worse_than_base']}**"]
        if fl["undetermined"]:
            out.append(f"- undetermined (not computable): {', '.join(fl['undetermined'])}")
        out.append("")
        if "verdict" in sm:
            bt_ = res.dev_numbers["assets"][t].get("baseline_tstats", {})
            out += ["### Verdict (pre-registered rule; the hold-out can only downgrade)", "",
                    f"**{sm['verdict']}**", ""] + [f"- {x}" for x in sm["verdict_reasons"]]
            if bt_:
                out += [f"- Development t-statistics of the baselines: "
                        + ", ".join(f"{k} t = {_f(v)} ({'clears' if v > 3 else 'does not clear'} t > 3)"
                                    for k, v in bt_.items())
                        + ". Buy-and-hold style baselines can also clear t > 3, so t > 3 alone is not an edge "
                        "(it is only a necessary condition, never grounds for weak evidence)."]
            out.append("")
    if res.summary.get("per_asset_verdicts"):
        out += ["## Headline verdict (pre-registered: the WEAKEST per-asset verdict)", "",
                "| Asset | Verdict |", "|---|---|"]
        out += [f"| {t} | {v} |" for t, v in res.summary["per_asset_verdicts"].items()]
        out += ["", f"**Headline: {res.summary['headline_verdict']}**", ""]
    out += ["## Notes", ""] + [f"- {n}" for n in res.summary["notes"]]
    return "\n".join(out) + "\n"


# =========================================================================== outputs
def _atomic_write(path: Path, writer):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    try:
        writer(tmp)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def write_outputs(res: HoldoutResult, out_dir) -> list:
    d = Path(out_dir)
    _atomic_write(d / "holdout_metrics.parquet", lambda p: res.metrics.to_parquet(p, index=False))
    _atomic_write(d / "holdout_predictions.parquet", lambda p: res.predictions.to_parquet(p, index=False))
    _atomic_write(d / "holdout_returns.parquet", lambda p: res.returns.to_parquet(p, index=False))
    _atomic_write(d / "holdout_summary.json",
                  lambda p: p.write_text(json.dumps(res.summary, indent=2, sort_keys=True, default=float) + "\n"))
    _atomic_write(d / "holdout_results.md", lambda p: p.write_text(res.markdown))
    return [d / n for n in OUTPUT_FILES]


# =========================================================================== freeze manifest
def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_manifest(cfg) -> dict:
    """SHA-256 of quant/*.py, config.yaml, uv.lock, pyproject.toml, every Stage 3 result file
    (data/results/**, except the hold-out folder), the live ledger (data/paper/**, except run.log)
    and the feature datasets (inputs).  Keys: 'project/...' or 'data/...'."""
    root, data = Path(cfg["project_root"]), Path(cfg["data_dir"])
    files = {}
    for p in sorted((root / "quant").glob("*.py")):
        files[f"project/quant/{p.name}"] = p
    for n in ("config.yaml", "uv.lock", "pyproject.toml"):
        if (root / n).exists():
            files[f"project/{n}"] = root / n
    for sub in ("results", "paper", "features"):
        base = data / sub
        if base.exists():
            for p in sorted(base.rglob("*")):
                rel = p.relative_to(data)
                if not p.is_file() or p.name in (".DS_Store", "run.log") or p.suffix == ".tmp":
                    continue
                if rel.parts[:2] == ("results", HOLDOUT_DIRNAME):
                    continue
                files[f"data/{rel.as_posix()}"] = p
    return {k: sha256_file(p) for k, p in files.items()}


def diff_manifests(before: dict, after: dict) -> list:
    return sorted([f"changed: {k}" for k in before if k in after and before[k] != after[k]]
                  + [f"removed: {k}" for k in before if k not in after]
                  + [f"added: {k}" for k in after if k not in before])


def manifest_digest(m: dict) -> str:
    return hashlib.sha256(json.dumps(m, sort_keys=True).encode()).hexdigest()


# =========================================================================== one-shot marker
def holdout_dir(cfg) -> Path:
    return Path(cfg["data_dir"]) / "results" / HOLDOUT_DIRNAME


def marker_path(cfg) -> Path:
    return holdout_dir(cfg) / MARKER_NAME


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def create_marker_exclusive(path: Path, payload: dict):
    """Write the complete marker to a temp file, then hard-link it into place: the link fails if the
    marker exists, so creation is atomic AND exclusive (two racing runs cannot both start)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        with open(tmp, "w") as f:
            f.write(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
            f.flush()
            os.fsync(f.fileno())
        os.link(tmp, path)
    except FileExistsError as e:
        raise HoldoutRefusal(f"{path} already exists") from e
    finally:
        if tmp.exists():
            tmp.unlink()


def update_marker(path: Path, **fields):
    cur = json.loads(path.read_text())
    cur.update(fields)
    _atomic_write(path, lambda p: p.write_text(json.dumps(cur, indent=2, sort_keys=True, default=str) + "\n"))


MANUAL_STEP = ("To allow a re-run you must do it by hand: read the failure in the marker, rename it "
               "(e.g. mv HOLDOUT_DONE.json HOLDOUT_FAILED_<date>.json), and write the reason in PROGRESS.md. "
               "A re-run is then NOT a single evaluation and the report must say so.")


def refuse_if_already_run(cfg):
    mp = marker_path(cfg)
    if mp.exists():
        try:
            st = json.loads(mp.read_text()).get("status", "unreadable")
        except (OSError, ValueError):
            st = "unreadable"
        raise HoldoutRefusal(f"the hold-out was already evaluated or attempted ({mp}, status={st!r}). "
                             + MANUAL_STEP)
    hd = holdout_dir(cfg)
    present = [n for n in OUTPUT_FILES if (hd / n).exists()]
    if present:
        raise HoldoutRefusal(f"hold-out outputs already exist without a marker: {present}. " + MANUAL_STEP)


def preflight(cfg) -> dict:
    """EVERYTHING that can fail for a non-statistical reason, BEFORE the marker is written, so a
    failure here does not burn the one shot: pinned config values, pre-registered config SHA, model
    imports (lightgbm), coverage, dataset shape, Stage 3 history, strategy lookups (summary_numbers)
    and the freeze manifest."""
    assert_pinned_cfg(cfg)
    verify_preregistration(cfg)
    w = real_window(cfg)
    strategy = pinned_strategy(cfg)
    B.require_models(strategy_models(strategy))
    rd = Path(cfg["data_dir"]) / "results"
    plan, frames, hists = {}, {}, {}
    for t in [a["ticker"] for a in cfg["assets"]]:
        plan[t] = check_coverage(t, read_dates(t, cfg, w.end), w)
        frames[t] = load_frame(t, cfg, w.end)
        check_frame_shape(t, frames[t], w)
        hists[t] = load_history(t, strategy, rd, w.cutoff, pd.Timestamp(PINNED_CUTOFF))
        check_history_contiguity(t, hists[t], w)
        strategy_config_id(t, strategy)
    return dict(w=w, strategy=strategy, plan=plan, frames=frames, hists=hists,
                dev_numbers=summary_numbers(cfg), before=build_manifest(cfg))


def run_final(cfg, confirm: bool, bt=None, out=print) -> HoldoutResult:
    """The one real evaluation.  Raises HoldoutRefusal before anything is written when a guard
    trips; after the marker is written any failure leaves status 'failed'."""
    if not confirm:
        raise HoldoutRefusal("--run requires --confirm-single-final-evaluation (the hold-out may be "
                             "evaluated exactly once)")
    refuse_if_already_run(cfg)
    pf = preflight(cfg)
    w, strategy, plan, before = pf["w"], pf["strategy"], pf["plan"], pf["before"]
    mp = marker_path(cfg)
    create_marker_exclusive(mp, dict(
        status="started", started_utc=_now(), holdout_cutoff=str(w.cutoff.date()), holdout_end=str(w.end.date()),
        holdout_months=int(cfg["holdout_months"]), pinned_strategy=strategy, per_asset=plan,
        preregistered_sha256=PREREGISTERED_SHA256, preregistered_code_sha256=PREREGISTERED_CODE_SHA256,
        holdout_py_sha256=(sha256_file(Path(cfg["project_root"]) / "quant" / "holdout.py")
                           if (Path(cfg["project_root"]) / "quant" / "holdout.py").exists() else None),
        earlier_failed_attempts=earlier_failed_attempts(cfg),
        manifest=before, manifest_sha256=manifest_digest(before), outputs=[]))
    try:
        res = evaluate_all(cfg, w, bt=bt, frames=pf["frames"], hists=pf["hists"], dev_numbers=pf["dev_numbers"])
        write_outputs(res, holdout_dir(cfg))
        diffs = diff_manifests(before, build_manifest(cfg))
        if diffs:
            raise FreezeViolation("frozen files changed during the run: " + "; ".join(diffs[:10]))
        update_marker(mp, status="done", finished_utc=_now(), freeze_verified=True,
                      outputs={n: sha256_file(holdout_dir(cfg) / n) for n in OUTPUT_FILES})
    except BaseException as e:
        try:
            update_marker(mp, status="failed", finished_utc=_now(), error=f"{type(e).__name__}: {e}",
                          traceback=traceback.format_exc(), manual_step=MANUAL_STEP)
        finally:
            raise
    return res


def run_dry(cfg, pseudo_cutoff=None, assets=None, out_dir=None, bt=None) -> tuple:
    """Pseudo hold-out inside development data; writes only to a scratch dir (never the real folder)."""
    w = pseudo_window(cfg, pseudo_cutoff)
    out = Path(out_dir) if out_dir else Path(tempfile.mkdtemp(prefix="holdout_dryrun_"))
    real = holdout_dir(cfg).resolve()
    if out.resolve() == real or real in out.resolve().parents:
        raise HoldoutRefusal("a dry run may never write into the real hold-out folder")
    res = evaluate_all(cfg, w, assets=assets, bt=bt, dev_numbers=summary_numbers(cfg))
    write_outputs(res, out)
    return res, out


# =========================================================================== significance summary
def summary_numbers(cfg=None) -> dict:
    """Read-only numbers for the top of GAUNTLET_REPORT.md, taken from the SAVED Stage 3 results:
    trial counts, the chosen strategy's PSR/DSR (asset scope and global/pooled scope), the t-statistic
    vs the Harvey-Liu-Zhu t > 3 hurdle, and Hansen's SPA p-values."""
    from quant import dash_data as DD
    cfg = DD._cfg(cfg)
    idx = DD.load_trial_index(cfg)
    s = pinned_strategy(cfg)
    out = dict(n_trials_global=int(len(idx)),
               n_configs_incl_aliases=int(len(T.all_logged_config_ids(idx))), assets={})
    for t in DD.tickers(cfg):
        res = DD.load_asset_results(t, cfg)
        cid = strategy_config_id(t, s)
        canon = DD.canonical_config_id(idx, cid)
        sig = res.significance
        rows = sig[(sig["subject"] == canon) & (sig["window"] == s["window"])]

        def one(test, ref=None):
            r = rows[rows["test"] == test]
            if ref is not None:
                r = r[r["reference"] == ref]
            return r.iloc[0] if len(r) else None

        def val(test, ref=None, col="value"):
            r = one(test, ref)
            return float("nan") if r is None else float(r[col])
        da, dg = one("dsr", "asset"), one("dsr", "global")
        ts = one("tstat_vs_3")
        spa = sig[sig["test"] == "spa"].set_index("reference")
        base_t = {b: float(sig[(sig["test"] == "tstat_vs_3") & (sig["subject"] == b)
                               & (sig["window"] == s["window"])].iloc[0]["statistic"])
                  for b in REFERENCES}
        out["assets"][t] = dict(
            config_id=cid, canonical_id=canon, n_oos_days=int(rows.iloc[0]["n"]),
            daily_sharpe=val("psr", col="statistic"), psr=val("psr"),
            dsr_asset=float(da["value"]), n_trials_asset=int(json.loads(da["details"])["n_trials"]),
            dsr_global=float(dg["value"]), n_trials_global=int(json.loads(dg["details"])["n_trials"]),
            dsr_excess_over_vol_targeted_bh=val("dsr_excess_vt"), psr_excess_over_vol_targeted_bh=val("psr_excess_vt"),
            # t of the mean excess return = daily excess Sharpe * sqrt(n)
            tstat_excess_over_vol_targeted_bh=(float("nan") if one("psr_excess_vt") is None else
                                               val("psr_excess_vt", col="statistic")
                                               * math.sqrt(float(one("psr_excess_vt")["n"]))),
            tstat=float(ts["statistic"]), clears_t_gt_3=bool(json.loads(ts["details"])["passes"]),
            baseline_tstats=base_t,
            spa_pvalue_vs_buy_and_hold=float(spa.loc["buy_and_hold", "pvalue"]),
            spa_pvalue_vs_vol_targeted_bh=float(spa.loc["vol_targeted_bh", "pvalue"]))
    return out


# =========================================================================== CLI
def _err(msg):
    print(f"ERROR: {msg}", file=sys.stderr)
    return 2


def main(argv=None) -> int:
    from quant.config import load_config
    ap = argparse.ArgumentParser(description="Stage 6 one-shot hold-out evaluation harness.")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--dry-run", action="store_true", help="evaluate a PSEUDO hold-out inside development data")
    g.add_argument("--run", action="store_true", help="the real, single final evaluation")
    g.add_argument("--significance-summary", action="store_true", help="print the saved significance numbers")
    g.add_argument("--verify-preregistration", action="store_true",
                   help="check config.yaml and quant/*.py (excl. holdout.py) against the pre-registered SHA-256s")
    ap.add_argument("--confirm-single-final-evaluation", action="store_true")
    ap.add_argument("--pseudo-cutoff", default=None)
    ap.add_argument("--assets", nargs="*", default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--config", default=None)
    args = ap.parse_args(argv)
    if not args.run and args.confirm_single_final_evaluation:
        return _err("--confirm-single-final-evaluation only applies to --run")
    if not args.dry_run and (args.pseudo_cutoff or args.out_dir or args.assets):
        return _err("--pseudo-cutoff, --assets and --out-dir only apply to --dry-run")
    if args.run and args.config is not None:
        cp = Path(args.config)
        cp = cp if cp.is_absolute() else PROJECT_ROOT / cp
        if not resolves_to(cp, PROJECT_ROOT / "config.yaml"):
            return _err("--run refuses --config: the single final evaluation only uses "
                        f"{PROJECT_ROOT / 'config.yaml'}")
    try:
        if args.verify_preregistration:
            ok = verify_preregistration()
            print(json.dumps({"config.yaml": ok["config.yaml"], "code_sha256_excl_holdout": ok["code_sha256"],
                              "holdout.py": sha256_file(Path(__file__)), "matches_preregistered": True,
                              **{n: (sha256_file(PROJECT_ROOT / n) if (PROJECT_ROOT / n).exists() else None)
                                 for n in ("uv.lock", "pyproject.toml")}}, indent=2))
            return 0
        cfg = load_config(args.config)
        if args.significance_summary:
            print(json.dumps(summary_numbers(cfg), indent=2, sort_keys=True))
            return 0
        if args.dry_run:
            res, out = run_dry(cfg, args.pseudo_cutoff, args.assets, args.out_dir)
            print(res.markdown)
            print(f"dry-run outputs written to {out}")
            return 0
        res = run_final(cfg, args.confirm_single_final_evaluation)
        print(res.markdown)
        print(f"hold-out results saved to {holdout_dir(cfg)}")
        return 0
    except HoldoutRefusal as e:
        return _err(e)
    except (OSError, ValueError, KeyError, FreezeViolation) as e:
        print(f"ERROR: {type(e).__name__}: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
