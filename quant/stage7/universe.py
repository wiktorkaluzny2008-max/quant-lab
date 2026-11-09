"""Stage 7a - multi-asset universe (ETFs + BTC/ETH): download, load, eligibility.

Reuses the Stage 1 data layer (quant.data) unchanged. One parquet per ticker in
``data_dir``; each asset keeps its OWN calendar (no forward-fill, no back-filling).
"""
import logging

import pandas as pd

from quant import data as D
from quant.stage7.config import dev_cutoff, load_stage7_config

log = logging.getLogger(__name__)


class UniverseUpdateError(RuntimeError):
    """One or more universe tickers failed to download/validate (message lists all)."""


def _assets(cfg, tickers=None):
    ua = cfg["universe"]
    if tickers is None:
        return list(ua)
    by = {a["ticker"]: a for a in ua}
    unknown = [t for t in tickers if t not in by]
    if unknown:
        raise KeyError(f"not in the Stage 7 universe: {unknown}")
    return [by[t] for t in tickers]


def update_universe(cfg=None, now=None, warnings=None):
    """Download/refresh every universe ticker (Stage 1 rules). Returns {ticker: frame}.

    Tries ALL tickers even if some fail, then raises UniverseUpdateError naming each
    failed ticker. Stale-data warnings are appended to ``warnings`` if given."""
    cfg = cfg or load_stage7_config()
    out, failures = {}, []
    for a in cfg["universe"]:
        t, tw = a["ticker"], a["trades_weekends"]
        try:
            out[t] = D.update_ticker(t, a["start"], cfg["data_dir"], tw, now)
        except (D.DataDownloadError, D.DataValidationError) as e:
            log.error("%s: universe update failed: %s", t, e)
            failures.append(f"{t}: {e}")
            continue
        w = D.check_staleness(out[t], t, tw, now)
        if w and warnings is not None:
            warnings.append(w)
    if failures:
        raise UniverseUpdateError(f"{len(failures)} universe ticker(s) failed: " + " | ".join(failures))
    return out


def load_universe(cfg=None, tickers=None, include_holdout=False):
    """Read stored parquet files (no network).

    DEFAULT RETURNS DEV ROWS ONLY (index <= the pinned hold-out cutoff). Rows after
    the cutoff are returned only with include_holdout=True, which is for the
    Stage 7 hold-out evaluation ONLY."""
    cfg = cfg or load_stage7_config()
    cut = dev_cutoff(cfg)
    out = {}
    for a in _assets(cfg, tickers):
        df = pd.read_parquet(D._cache_path(cfg["data_dir"], a["ticker"]))
        out[a["ticker"]] = df if include_holdout else df[df.index <= cut]
    return out


def warmup_for(ticker, cfg):
    a = next(u for u in cfg["universe"] if u["ticker"] == ticker)
    return int(cfg["warmup_days"]["crypto" if a["group"] == "crypto" else "etf"])


def eligible_from(df, ticker, cfg):
    """Date of row number warmup_days (0-based), i.e. first date + warm-up rows.
    NaT if the asset has not that many rows yet."""
    w = warmup_for(ticker, cfg)
    return df.index[w] if len(df) > w else pd.NaT


def available_assets(date, cfg=None, data=None):
    """Tickers whose eligible_from <= date (an asset enters only after its warm-up)."""
    cfg = cfg or load_stage7_config()
    data = data if data is not None else load_universe(cfg)
    date = pd.Timestamp(date)
    out = []
    for a in cfg["universe"]:
        t = a["ticker"]
        if t in data:
            e = eligible_from(data[t], t, cfg)
            if pd.notna(e) and e <= date:
                out.append(t)
    return out


def start_date_table(cfg=None, data=None):
    """One row per asset; dev rows only (rows after the cutoff are never touched)."""
    cfg = cfg or load_stage7_config()
    data = data if data is not None else load_universe(cfg)
    cut = dev_cutoff(cfg)
    rows = []
    for a in cfg["universe"]:
        t = a["ticker"]
        if t not in data:
            continue
        df = data[t][data[t].index <= cut]
        rows.append({"ticker": t, "name": a["name"], "group": a["group"],
                     "first_date": df.index[0] if len(df) else pd.NaT,
                     "eligible_from": eligible_from(df, t, cfg),
                     "rows": len(df),
                     "last_dev_date": df.index[-1] if len(df) else pd.NaT})
    return pd.DataFrame(rows, columns=["ticker", "name", "group", "first_date",
                                       "eligible_from", "rows", "last_dev_date"])


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        cfg = load_stage7_config()
        warns = []
        update_universe(cfg, warnings=warns)
    except (D.DataDownloadError, D.DataValidationError, UniverseUpdateError) as e:
        log.error("%s", e)
        print(f"ERROR: {e}")
        raise SystemExit(1)
    if warns:
        bar = "!" * 70
        print(f"\n{bar}\nWARNING: {len(warns)} stale data warning(s)\n{bar}")
        for w in warns:
            print(f"  * {w}")
        print(bar)
    with pd.option_context("display.width", 200, "display.max_rows", 100):
        print(start_date_table(cfg).to_string(index=False))


if __name__ == "__main__":
    main()
