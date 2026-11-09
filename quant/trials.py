"""Trial log (QUANT_SPEC.md Stage 3, "Trial log").

Every configuration that has a tunable setting or a choice in it is a TRIAL: asset x window x
model x strategy mode x vol-targeted x margin setting (a fixed ``m`` from the grid, or the
``adaptive`` rule).  Fixed baselines (climatology, buy and hold, always-up, vol-targeted buy
and hold, trend_200) are NOT trials: nothing is selected among them.

Files (in ``<data_dir>/results/``):

* ``trials.parquet``       long format: config_id, date, net_return, gross_return, position
                           (daily out-of-sample returns at the HEADLINE cost).
* ``trials_index.parquet`` one row per config_id with every setting (including the model
                           settings from ``models.model_settings``), n_oos_days and the
                           annualised Sharpe, plus the per-day Sharpe used by the Deflated
                           Sharpe Ratio.

DEDUPLICATION: configurations whose position (hence return) streams are identical (e.g.
``long_flat`` and ``default_long`` are the same rule at equal m) are logged ONCE.  The canonical
config_id is the first one generated (order: window, model, mode [long_flat before
default_long], raw before vol-targeted, grid m ascending, adaptive last); the others are listed
in its ``aliases`` column (JSON list).  The trial count N of the Deflated Sharpe Ratio is the
number of DISTINCT streams = rows of the index.

Logging is idempotent: ``replace_asset`` removes every existing row of that ticker and writes
the new rows, then rewrites both files sorted by (config_id, date).  Other tickers' rows are
untouched, so running ``--assets SPY`` after a full run neither duplicates nor drops anything,
and running the same thing twice gives byte-identical contents.  The number of trials is the
number of rows of the index (``count_trials``), summed over all assets currently in the log.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from quant import metrics as M

MODES = ("long_flat", "default_long")
# Fixed grids of the threshold margin m (probability points below the training base rate).
M_GRID = {"long_flat": (0.0, 0.005, 0.01, 0.02), "default_long": (0.01, 0.02, 0.03)}
# Margin used by the adaptive rule until enough past out-of-sample days exist.
M_FALLBACK = {"long_flat": 0.0, "default_long": 0.02}
ADAPTIVE = "adaptive"
WINDOWS = ("expanding", "rolling4y")

RETURN_COLS = ["config_id", "date", "net_return", "gross_return", "position"]
INDEX_COLS = ["config_id", "ticker", "window", "model", "mode", "vol_targeted", "m_setting",
              "m_fixed", "cost", "model_settings", "n_oos_days", "start", "end",
              "ann_sharpe", "daily_sharpe", "duplicate_of", "aliases"]


def m_label(m) -> str:
    return ADAPTIVE if m == ADAPTIVE else f"{float(m):g}"


def make_config_id(ticker, window, model, mode, vol_targeted, m_setting) -> str:
    vt = "vt" if vol_targeted else "raw"
    return f"{ticker}|{window}|{model}|{mode}|{vt}|m={m_label(m_setting)}"


def m_settings(mode) -> list:
    """All margin settings that are trials for ``mode``: the fixed grid, then 'adaptive'."""
    return [*M_GRID[mode], ADAPTIVE]


def expected_config_ids(ticker, windows, models) -> list:
    """Every config_id a complete run over ``windows`` x ``models`` must log."""
    return [make_config_id(ticker, w, mdl, mode, vt, m)
            for w in windows for mdl in models for mode in MODES
            for vt in (False, True) for m in m_settings(mode)]


# --------------------------------------------------------------------------- paths / io
def default_results_dir() -> Path:
    from quant.config import load_config
    return load_config()["data_dir"] / "results"


def _paths(results_dir):
    d = Path(results_dir) if results_dir is not None else default_results_dir()
    return d, d / "trials.parquet", d / "trials_index.parquet"


def _atomic_write(df: pd.DataFrame, path: Path):
    tmp = path.with_suffix(".tmp")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def load_index(results_dir=None) -> pd.DataFrame:
    _, _, ip = _paths(results_dir)
    if not ip.exists():
        return pd.DataFrame(columns=INDEX_COLS)
    return pd.read_parquet(ip)


def load_trials(results_dir=None) -> pd.DataFrame:
    _, tp, _ = _paths(results_dir)
    if not tp.exists():
        return pd.DataFrame(columns=RETURN_COLS)
    return pd.read_parquet(tp)


def replace_asset(ticker, index_df: pd.DataFrame, returns_df: pd.DataFrame, results_dir=None):
    """Replace ALL rows of ``ticker`` in the log with the given ones (idempotent)."""
    d, tp, ip = _paths(results_dir)
    d.mkdir(parents=True, exist_ok=True)
    if index_df["config_id"].duplicated().any():
        raise ValueError("duplicate config_id in the new trial index")
    if set(index_df["config_id"]) != set(returns_df["config_id"]):
        raise ValueError("trial index and trial returns cover different config_ids")
    old_idx = load_index(d)
    old_ret = load_trials(d)
    keep_ids = set(old_idx.loc[old_idx["ticker"] != ticker, "config_id"])
    if set(index_df["config_id"]) & keep_ids:
        raise ValueError("a config_id of the new run already belongs to another ticker")
    idx = pd.concat([old_idx[old_idx["ticker"] != ticker], index_df[INDEX_COLS]],
                    ignore_index=True).sort_values("config_id", kind="mergesort")
    ret = pd.concat([old_ret[old_ret["config_id"].isin(keep_ids)], returns_df[RETURN_COLS]],
                    ignore_index=True).sort_values(["config_id", "date"], kind="mergesort")
    _atomic_write(idx.reset_index(drop=True), ip)
    _atomic_write(ret.reset_index(drop=True), tp)
    return len(idx)


def all_logged_config_ids(index: pd.DataFrame) -> set:
    """Canonical ids plus every alias (configs whose return stream equals a canonical one)."""
    out = set(index["config_id"])
    for a in index["aliases"]:
        out |= set(json.loads(a)) if a else set()
    return out


def count_trials(results_dir=None) -> int:
    """Number of configurations in the log (the N of the Deflated Sharpe Ratio)."""
    return int(len(load_index(results_dir)))


def trial_sharpes(results_dir=None) -> np.ndarray:
    """PER-DAY (not annualised) Sharpe of every logged trial, in config_id order; NaN for
    trials with zero variance (kept, so the vector length equals ``count_trials``)."""
    idx = load_index(results_dir).sort_values("config_id")
    return idx["daily_sharpe"].to_numpy(dtype=float)


def trial_sharpe_variance(results_dir=None) -> float:
    s = trial_sharpes(results_dir)
    s = s[np.isfinite(s)]
    return float(np.var(s, ddof=1)) if s.size > 1 else float("nan")


def build_index_row(ticker, window, model, mode, vol_targeted, m_setting, cost, model_settings_json,
                    dates, net, ppy, aliases=()) -> dict:
    return dict(
        config_id=make_config_id(ticker, window, model, mode, vol_targeted, m_setting),
        ticker=ticker, window=window, model=model, mode=mode, vol_targeted=bool(vol_targeted),
        m_setting=m_label(m_setting),
        m_fixed=float("nan") if m_setting == ADAPTIVE else float(m_setting),
        cost=float(cost), model_settings=model_settings_json, n_oos_days=int(len(net)),
        start=pd.Timestamp(dates[0]), end=pd.Timestamp(dates[-1]),
        ann_sharpe=M.sharpe(net, ppy), daily_sharpe=M.sharpe(net, 1),
        duplicate_of="", aliases=json.dumps(sorted(aliases)))
