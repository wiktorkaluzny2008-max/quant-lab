"""Stage 7d - FRED daily macro series (public CSV, no API key).

LOOKAHEAD RULE: FRED dates are OBSERVATION dates. A value observed on date t is treated as known
only from the NEXT NYSE session strictly after t (see availability_date). Everything that reads
macro data for a target date must go through asof_macro / macro_panel, never a plain reindex.

HOLD-OUT RULE: load_fred returns only dev observations (<= pinned cutoff) by default. Later
observations are for the Stage 7 hold-out evaluation only (include_holdout=True).

These helpers are used by NEW Stage 7 outputs only; Stages 3-6 are not touched.
Run: uv run python -m quant.stage7.macro
"""
import json
import logging
import sys
import time
import urllib.request
from datetime import datetime, timedelta, timezone
from io import StringIO

import numpy as np
import pandas as pd

from quant.data import _write_atomic, _xnys
from quant.stage7.config import FRED_IDS, dev_cutoff, load_stage7_config

log = logging.getLogger(__name__)

REFETCH_DAYS = 28            # re-download this far before the last stored date
STALE_DAYS = 7               # FRED daily series lag a few days; warn beyond this
TIMEOUT_S = 30
RETRIES = 3
# NOTE: FRED silently drops requests with browser-like or custom User-Agents (connection hangs/resets),
# so we keep urllib's plain default "Python-urllib/x.y" agent on purpose.


class FredDownloadError(RuntimeError):
    """A FRED download or parse failed (message names the series)."""


class FredValidationError(ValueError):
    """Downloaded/merged FRED data is unusable."""


def _fred_dir(cfg):
    return cfg["data_dir"] / "fred"


def _check_ids(ids):
    bad = [s for s in ids if s not in FRED_IDS]
    if bad:
        raise ValueError(f"unknown FRED series {bad}; allowed: {list(FRED_IDS)}")


def _now_utc(now=None):
    if now is None:
        return datetime.now(timezone.utc)
    now = pd.Timestamp(now)
    return (now.tz_localize("UTC") if now.tzinfo is None else now.tz_convert("UTC")).to_pydatetime()


# ----------------------------------------------------------------------------- download / parse
def _fetch_text(url: str) -> str:
    """The single network call (mock this in tests)."""
    last = None
    for attempt in range(RETRIES):
        try:
            with urllib.request.urlopen(urllib.request.Request(url), timeout=TIMEOUT_S) as r:
                return r.read().decode("utf-8")
        except Exception as e:      # noqa: BLE001 - retried, then re-raised below
            last = e
            time.sleep(2 * (attempt + 1))
    raise last


def parse_fred_csv(text: str, series: str):
    """CSV text -> (Series named 'value' indexed by observation_date, number of missing dropped)."""
    try:
        raw = pd.read_csv(StringIO(text), dtype=str)
    except Exception as e:
        raise FredDownloadError(f"{series}: could not parse CSV: {e}") from e
    cols = list(raw.columns)
    if len(cols) != 2 or cols[1] != series or cols[0] not in ("observation_date", "DATE"):
        raise FredDownloadError(f"{series}: unexpected CSV columns {cols}")
    dates = pd.to_datetime(raw[cols[0]], errors="coerce")
    if dates.isna().any():
        raise FredDownloadError(f"{series}: unparseable date(s) in CSV")
    v = raw[cols[1]].str.strip()
    miss = v.isna() | (v == "") | (v == ".")
    vals = pd.to_numeric(v.where(~miss), errors="coerce")
    if (vals.isna() & ~miss).any():
        raise FredDownloadError(f"{series}: non-numeric value(s) in CSV")
    out = pd.Series(vals.to_numpy(dtype="float64"), index=pd.DatetimeIndex(dates, name="observation_date"),
                    name="value")
    n_missing = int(miss.sum())
    return out[~miss.to_numpy()], n_missing


def download_series(series: str, start: str, cfg) -> tuple:
    _check_ids([series])
    url = f"{cfg['fred']['base_url']}?id={series}&cosd={start}"
    try:
        text = _fetch_text(url)
    except Exception as e:
        raise FredDownloadError(f"{series}: download failed ({type(e).__name__}: {e}) from {url}") from e
    s, n_missing = parse_fred_csv(text, series)
    return s, n_missing, url


# ----------------------------------------------------------------------------- validation
def validate_series(s: pd.Series, series: str, cfg, now=None):
    if len(s) == 0:
        raise FredValidationError(f"{series}: no observations")
    if not s.index.is_unique:
        raise FredValidationError(f"{series}: duplicate dates")
    if not s.index.is_monotonic_increasing:
        raise FredValidationError(f"{series}: dates are not sorted")
    if not np.isfinite(s.to_numpy(dtype="float64")).all():
        raise FredValidationError(f"{series}: non-finite values")
    spec = cfg["fred"]["series"][series]
    if (s < spec["min"]).any() or (s > spec["max"]).any():
        bad = s[(s < spec["min"]) | (s > spec["max"])]
        raise FredValidationError(
            f"{series}: {len(bad)} value(s) outside plausible range [{spec['min']}, {spec['max']}], "
            f"first on {bad.index[0].date()}")
    today = pd.Timestamp(_now_utc(now).date())
    if s.index.max() > today:
        raise FredValidationError(f"{series}: future date {s.index.max().date()}")


def check_staleness(s: pd.Series, series: str, now=None):
    """Return a warning string if the last observation is older than STALE_DAYS, else None."""
    today = pd.Timestamp(_now_utc(now).date())
    last = s.index.max()
    if (today - last).days > STALE_DAYS:
        msg = f"{series}: STALE DATA - last observation {last.date()} is {(today - last).days} days old"
        log.warning("%s", msg)
        return msg
    return None


# ----------------------------------------------------------------------------- update (network)
def update_series(series: str, cfg, now=None) -> dict:
    """Create/refresh data/fred/{series}.parquet; returns the status dict for that series."""
    _check_ids([series])
    path = _fred_dir(cfg) / f"{series}.parquet"
    start = pd.Timestamp(cfg["fred"]["start"])
    old = None
    if path.exists():
        old = pd.read_parquet(path)["value"]
        if len(old):
            start = max(start, old.index.max() - timedelta(days=REFETCH_DAYS))
    new, n_missing, url = download_series(series, start.strftime("%Y-%m-%d"), cfg)
    if old is not None and len(old):
        # keep='last': newly downloaded values win over stored ones for overlapping dates
        merged = pd.concat([old, new])
        merged = merged[~merged.index.duplicated(keep="last")].sort_index()
    else:
        merged = new.sort_index()
    validate_series(merged, series, cfg, now)
    merged = merged.astype("float64").rename("value")
    merged.index.name = "observation_date"
    _write_atomic(merged.to_frame(), path)
    warn = check_staleness(merged, series, now)
    return {
        "series": series, "first_date": str(merged.index.min().date()), "last_date": str(merged.index.max().date()),
        "rows": int(len(merged)), "missing_dropped_in_last_download": n_missing,
        "last_download_window_start": start.strftime("%Y-%m-%d"), "source_url": url,
        "downloaded_at": _now_utc(now).isoformat(), "stale_warning": warn,
    }


def update_all(cfg=None, now=None, series=None) -> dict:
    cfg = cfg or load_stage7_config()
    ids = list(series) if series else list(cfg["fred"]["series"])
    _check_ids(ids)
    sp = _fred_dir(cfg) / "status.json"
    status = json.loads(sp.read_text()) if sp.exists() else {}
    for sid in ids:
        status[sid] = update_series(sid, cfg, now)
        sp.parent.mkdir(parents=True, exist_ok=True)
        tmp = sp.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(status, indent=2))
        tmp.replace(sp)
    return {k: status[k] for k in ids}


# ----------------------------------------------------------------------------- reading (no network)
def load_fred(series=None, cfg=None, include_holdout=False):
    """Stored observations, no network. One id -> Series; None/list -> dict of Series.

    DEFAULT: DEV observations only (observation_date <= pinned hold-out cutoff).
    include_holdout=True is for the Stage 7 hold-out evaluation ONLY.
    """
    cfg = cfg or load_stage7_config()
    single = isinstance(series, str)
    ids = [series] if single else (list(series) if series else list(cfg["fred"]["series"]))
    _check_ids(ids)
    out = {}
    for sid in ids:
        p = _fred_dir(cfg) / f"{sid}.parquet"
        if not p.exists():
            raise FileNotFoundError(f"{p} not found; run: uv run python -m quant.stage7.macro")
        s = pd.read_parquet(p)["value"]
        if not include_holdout:
            s = s[s.index <= dev_cutoff(cfg)]
        out[sid] = s
    return out[series] if single else out


# ----------------------------------------------------------------------------- lookahead rule
def availability_date(obs_dates) -> pd.DatetimeIndex:
    """First NYSE session strictly after each observation date (tz-naive, normalised)."""
    obs = pd.DatetimeIndex(obs_dates).normalize()
    sessions = _xnys().sessions
    sessions = sessions.tz_localize(None) if sessions.tz is not None else sessions
    pos = sessions.searchsorted(obs, side="right")
    out = []
    for o, p in zip(obs, pos):
        # beyond the calendar's last session: fall back to the next weekday
        out.append(sessions[p] if p < len(sessions) else o + pd.offsets.BDay(1))
    return pd.DatetimeIndex(out, name="available_from")


def _asof_one(target_index, s: pd.Series) -> pd.Series:
    tgt = pd.DatetimeIndex(target_index)
    if tgt.tz is not None:
        tgt = tgt.tz_localize(None)
    s = s.dropna().sort_index()
    if len(s) == 0:
        return pd.Series(np.nan, index=target_index, name=s.name, dtype="float64")
    av = pd.Series(s.to_numpy(dtype="float64"), index=availability_date(s.index))
    av = av[~av.index.duplicated(keep="last")].sort_index()
    pos = av.index.searchsorted(tgt, side="right") - 1      # latest availability <= d
    vals = np.where(pos >= 0, av.to_numpy()[np.clip(pos, 0, None)], np.nan)
    return pd.Series(vals, index=target_index, name=s.name, dtype="float64")


def asof_macro(target_index, data):
    """Backward as-of join: for each target date d the latest observation with availability <= d.

    Weekends/holidays carry the last known value; NaN before the first availability. `data` is a
    Series/one-column DataFrame (returns Series) or a dict of them (returns a wide DataFrame).
    """
    if isinstance(data, dict):
        return pd.DataFrame({k: _asof_one(target_index, _as_series(v, k)) for k, v in data.items()},
                            index=target_index)
    return _asof_one(target_index, _as_series(data, None))


def _as_series(v, name):
    if isinstance(v, pd.DataFrame):
        v = v["value"] if "value" in v.columns else v.iloc[:, 0]
    return v.rename(name if name is not None else v.name)


def macro_panel(target_index, cfg=None, include_holdout=False, series=None) -> pd.DataFrame:
    """Wide frame (one column per series) of causal as-of values on target_index."""
    cfg = cfg or load_stage7_config()
    return asof_macro(target_index, load_fred(series, cfg, include_holdout))


# ----------------------------------------------------------------------------- Stage 7 helpers
def risk_free_daily(target_index, cfg=None, trading_days=252) -> pd.Series:
    """DTB3 (annual percent) -> daily fraction, causal. trading_days: 252 (ETFs) or 365 (crypto)."""
    if trading_days not in (252, 365):
        raise ValueError("trading_days must be 252 or 365")
    cfg = cfg or load_stage7_config()
    s = asof_macro(target_index, load_fred("DTB3", cfg)) / 100.0 / trading_days
    return s.rename("risk_free_daily")


def borrow_rate_annual(target_index, cfg=None) -> pd.Series:
    """Leverage borrowing rate = DFF/100 + borrow_spread (annual fraction), causal."""
    cfg = cfg or load_stage7_config()
    s = asof_macro(target_index, load_fred("DFF", cfg)) / 100.0 + cfg["fred"]["borrow_spread"]
    return s.rename("borrow_rate_annual")


def regime_context(date, cfg=None, data=None, min_history=250) -> list:
    """Short plain-English macro notes for the 'Why?' text, using ONLY data known on `date`.

    `data` (optional dict of Series) lets callers/tests pass series directly; otherwise dev data.
    """
    cfg = cfg or load_stage7_config()
    d = pd.Timestamp(date).normalize()
    if data is None:
        data = load_fred(["T10Y2Y", "BAMLH0A0HYM2"], cfg, include_holdout=d > dev_cutoff(cfg))
    notes = []

    def known(sid):
        s = data[sid].dropna()
        return s[availability_date(s.index) <= d] if len(s) else s

    t = known("T10Y2Y") if "T10Y2Y" in data else pd.Series(dtype=float)
    if len(t):
        v = t.iloc[-1]
        notes.append(f"The yield curve is inverted (10y minus 2y = {v:.2f} points), historically a recession warning."
                     if v < 0 else f"The yield curve is not inverted (10y minus 2y = {v:.2f} points).")
    h = known("BAMLH0A0HYM2") if "BAMLH0A0HYM2" in data else pd.Series(dtype=float)
    if len(h) >= min_history:
        v = h.iloc[-1]
        pct = float((h <= v).mean() * 100)          # expanding history up to the known date
        level = "high" if pct >= 80 else "low" if pct <= 20 else "middling"
        notes.append(f"Junk-bond credit spreads are {level} ({v:.2f} points, higher than {pct:.0f}% of "
                     f"readings since {h.index[0].year}).")
    return notes


# ----------------------------------------------------------------------------- CLI
def main(argv=None) -> int:
    try:
        cfg = load_stage7_config()
        st = update_all(cfg)
    except Exception as e:
        print(f"ERROR: {e}")
        return 1
    print(f"{'series':<14}{'first':<12}{'last':<12}{'rows':>7}{'missing dropped':>17}")
    for k, v in st.items():
        print(f"{k:<14}{v['first_date']:<12}{v['last_date']:<12}{v['rows']:>7}{v['missing_dropped_in_last_download']:>17}")
    for v in st.values():
        if v["stale_warning"]:
            print("WARNING:", v["stale_warning"])
    late = [k for k, v in st.items() if v["first_date"] > cfg["fred"]["start"][:4] + "-12-31"]
    for k in late:
        print(f"NOTE: {k} starts {st[k]['first_date']}, later than requested start {cfg['fred']['start']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
