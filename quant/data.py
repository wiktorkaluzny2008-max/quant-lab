"""Stage 1 - data download, caching and validation.

One parquet per ticker in ``data_dir``. Prices are never forward-filled or
reindexed onto a common calendar: each ticker keeps its own trading days.
"""
import json
import logging
import os
import tempfile
from datetime import timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf

from quant.config import load_config

log = logging.getLogger(__name__)

COLUMNS = ["Open", "High", "Low", "Close", "Volume"]
PRICE_COLUMNS = ["Open", "High", "Low", "Close"]
REFETCH_DAYS = 5          # re-fetch this many calendar days before the last stored date
CALENDAR_START = "1990-01-01"   # XNYS calendar lower bound (library default is only ~20y back)
_CAL = None


def _xnys():
    """XNYS calendar (cached). It already treats 2001-09-11..14, Sandy etc. as closed."""
    global _CAL
    if _CAL is None:
        import warnings
        import exchange_calendars as xc
        with warnings.catch_warnings():
            # exchange_calendars 4.x calls Series.values on tz-aware data internally
            # (third-party pandas deprecation); not ours to fix.
            warnings.filterwarnings("ignore", message="Series.values returning")
            _CAL = xc.get_calendar("XNYS", start=CALENDAR_START)
    return _CAL


class DataDownloadError(RuntimeError):
    """yfinance failed or returned nothing."""


class DataValidationError(ValueError):
    """Downloaded/merged data is unusable."""


def _download(ticker, start, end=None):
    """Thin wrapper around yfinance (the single network call; mocked in tests)."""
    return yf.download(ticker, start=start, end=end, auto_adjust=True,
                       progress=False, actions=False, multi_level_index=False)


def _normalize(df, ticker):
    """Flatten columns, keep OHLCV, tz-naive normalized date index."""
    if df is None or len(df) == 0:
        raise DataDownloadError(f"{ticker}: yfinance returned no data")
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    missing = [c for c in COLUMNS if c not in df.columns]
    if missing:
        raise DataDownloadError(f"{ticker}: yfinance response missing columns {missing}")
    df = df[COLUMNS].astype("float64")
    idx = pd.DatetimeIndex(pd.to_datetime(df.index))
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    df.index = pd.DatetimeIndex(idx.normalize(), freq=None)
    df.index.name = "Date"
    df = df.dropna(how="all")
    if df.empty:
        raise DataDownloadError(f"{ticker}: yfinance returned only empty rows")
    return df


def validate(df, ticker, trades_weekends=False, extra=False):
    """Validate one price frame. Logs problems; raises DataValidationError.

    ``extra=True`` is for volatility-index feature series (^VIX, ^VIX3M, ^VXN,
    ^RVX). Decisions, from inspecting real Yahoo data (2026-11): the indices have
    zero volume (volume is never validated) and have no missing NYSE sessions
    after their first date, but ^VIX carries stale rows on a few US holidays
    (e.g. 2026-05-25, 2026-09-07) and each index starts on its own date. So for
    extras: duplicates, unsorted index, NaN and non-positive values are still
    fatal; NYSE sessions missing inside the series' own range are only logged as
    WARNINGs (an index may legitimately have holes; nothing is forward-filled);
    rows on non-session days are logged and kept. Assets (ETFs, BTC) stay strict.
    """
    problems = []
    soft = []
    if df.empty:
        problems.append("no rows")
    else:
        if df.index.has_duplicates:
            problems.append(f"{int(df.index.duplicated().sum())} duplicate dates")
        if not df.index.is_monotonic_increasing:
            problems.append("index not sorted")
        if df[COLUMNS].isna().any().any():
            problems.append(f"{int(df[COLUMNS].isna().sum().sum())} NaN values")
        bad = (df[PRICE_COLUMNS] <= 0).any(axis=1).sum()
        if bad:
            problems.append(f"{int(bad)} rows with zero/negative prices")
        if df.index.is_monotonic_increasing and not df.index.has_duplicates:
            if trades_weekends:
                gaps = pd.Series(df.index[1:] - df.index[:-1], index=df.index[1:])
                for end, gap in gaps[gaps > timedelta(days=1)].items():
                    start = end - gap
                    problems.append(f"gap of {gap.days} days between "
                                    f"{start.date()} and {end.date()}")
            else:
                cal = _xnys()
                lo = max(df.index.min(), cal.first_session.tz_localize(None))
                hi = min(df.index.max(), cal.last_session.tz_localize(None))
                if lo <= hi:
                    expected = cal.sessions_in_range(lo, hi).tz_localize(None)
                    for d in expected.difference(df.index):
                        (soft if extra else problems).append(f"missing trading session {d.date()}")
                    if extra:
                        for d in df.index.difference(expected):
                            if lo <= d <= hi:
                                soft.append(f"row on non-session day {d.date()} (kept)")
    for p in soft:
        log.warning("%s: validation note: %s", ticker, p)
    for p in problems:
        log.error("%s: validation problem: %s", ticker, p)
    if problems:
        raise DataValidationError(f"{ticker}: " + "; ".join(problems[:10]))
    return df


FOMC_FIRST_YEAR, FOMC_LAST_YEAR = 1999, 2027
FOMC_PATH = Path(__file__).resolve().parent.parent / "data" / "fomc_dates.csv"


def load_fomc_dates(path=None):
    """Scheduled FOMC announcement (statement) dates, hard-coded in git.

    Returns a sorted tz-naive DatetimeIndex. Validates: parseable, unique, sorted,
    weekdays only, 6-9 per calendar year (8 scheduled; 2020 had 7 because the March
    meeting was cancelled), and full coverage of 1999..2027."""
    p = Path(path) if path else FOMC_PATH
    if not p.exists():
        raise FileNotFoundError(f"FOMC dates file not found: {p}")
    raw = pd.read_csv(p, dtype=str)
    if "date" not in raw.columns:
        raise DataValidationError(f"{p}: missing 'date' column")
    try:
        dates = pd.DatetimeIndex(pd.to_datetime(raw["date"], format="%Y-%m-%d"))
    except (ValueError, TypeError) as e:
        raise DataValidationError(f"{p}: unparseable date: {e}") from e
    problems = []
    if dates.has_duplicates:
        problems.append("duplicate dates")
    if not dates.is_monotonic_increasing:
        problems.append("dates not sorted")
    wk = dates[dates.day_of_week >= 5]
    if len(wk):
        problems.append("weekend dates: " + ", ".join(str(d.date()) for d in wk))
    counts = pd.Series(dates.year).value_counts()
    for y in range(FOMC_FIRST_YEAR, FOMC_LAST_YEAR + 1):
        n = int(counts.get(y, 0))
        if not 6 <= n <= 9:
            problems.append(f"{y}: {n} dates (expected 6-9)")
    extra_years = sorted(set(counts.index) - set(range(FOMC_FIRST_YEAR, FOMC_LAST_YEAR + 1)))
    if extra_years:
        problems.append(f"years outside {FOMC_FIRST_YEAR}..{FOMC_LAST_YEAR}: {extra_years}")
    if problems:
        raise DataValidationError(f"{p}: " + "; ".join(problems))
    return dates


def _cache_path(data_dir, ticker):
    return Path(data_dir) / f"{ticker}.parquet"


def _write_atomic(df, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    os.close(fd)
    try:
        df.to_parquet(tmp)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def _now_utc(now=None):
    ts = pd.Timestamp.now("UTC") if now is None else pd.Timestamp(now)
    return ts.tz_localize("UTC") if ts.tz is None else ts.tz_convert("UTC")


def last_completed_session(now=None):
    """Most recent XNYS session (tz-naive date) whose close is <= now."""
    now = _now_utc(now)
    cal = _xnys()
    sessions = cal.sessions_in_range((now - timedelta(days=15)).tz_localize(None).normalize(),
                                     now.tz_localize(None).normalize())
    for s in reversed(list(sessions)):
        if cal.session_close(s) <= now:
            return s.tz_localize(None) if s.tz is not None else s
    raise RuntimeError("no completed session found in the last 15 days")


def _drop_incomplete_bar(df, ticker, trades_weekends, now=None):
    """Drop a bar for a period that has not finished (never store partial bars)."""
    now = _now_utc(now)
    if trades_weekends:
        today = pd.Timestamp(now.date())
        if (df.index == today).any():
            log.info("%s: dropping incomplete bar for today (%s UTC)", ticker, today.date())
            df = df[df.index != today]
        return df
    cal = _xnys()
    recent = df.index[df.index >= now.tz_localize(None).normalize() - timedelta(days=1)]
    drop = []
    for d in recent:
        if cal.is_session(d) and cal.session_close(d) > now:
            drop.append(d)
    if drop:
        log.info("%s: dropping incomplete bar(s) for unclosed session(s) %s", ticker,
                 ", ".join(str(d.date()) for d in drop))
        df = df[~df.index.isin(drop)]
    return df


def check_staleness(df, ticker, trades_weekends=False, now=None):
    """Return a warning message if the latest stored date is older than the most
    recent completed day (XNYS session close for stocks, yesterday UTC for BTC),
    else None. Also logs the warning."""
    now = _now_utc(now)
    if trades_weekends:
        expected = pd.Timestamp(now.date()) - timedelta(days=1)
        label = "yesterday (UTC)"
    else:
        expected = last_completed_session(now)
        label = "most recent completed trading session"
    latest = df.index.max() if len(df) else None
    if latest is not None and latest >= expected:
        return None
    msg = (f"{ticker}: STALE DATA - latest stored date is "
           f"{latest.date() if latest is not None else 'none'}, but {label} is "
           f"{expected.date()}")
    log.warning("%s", msg)
    return msg


def update_ticker(ticker, start, data_dir, trades_weekends=False, now=None, extra=False):
    """Create or incrementally update one ticker's parquet; return the frame."""
    path = _cache_path(data_dir, ticker)
    old = None
    fetch_from = pd.Timestamp(start)
    if path.exists():
        old = pd.read_parquet(path)
        if len(old):
            fetch_from = max(fetch_from, old.index.max() - timedelta(days=REFETCH_DAYS))
    try:
        raw = _download(ticker, fetch_from.strftime("%Y-%m-%d"))
    except DataDownloadError:
        raise
    except Exception as e:
        raise DataDownloadError(f"{ticker}: yfinance download failed: {e}") from e
    new = _normalize(raw, ticker)
    if old is not None and len(old):
        df = pd.concat([old[old.index < new.index.min()], new])
    else:
        df = new
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df = _drop_incomplete_bar(df, ticker, trades_weekends, now)
    validate(df, ticker, trades_weekends, extra=extra)
    _write_atomic(df, path)
    log.info("%s: %d rows, %s to %s", ticker, len(df),
             df.index[0].date(), df.index[-1].date())
    return df


def update_all(config=None, now=None, warnings=None):
    """Update all configured assets plus extra tickers. Returns {ticker: frame}.

    Staleness warnings are appended to the ``warnings`` list if one is passed."""
    cfg = config or load_config()
    data_dir = cfg["data_dir"]
    out = {}
    warns = warnings if warnings is not None else []
    for a in cfg["assets"]:
        tw = bool(a["trades_weekends"])
        out[a["ticker"]] = update_ticker(a["ticker"], a["start"], data_dir, tw, now)
        w = check_staleness(out[a["ticker"]], a["ticker"], tw, now)
        if w:
            warns.append(w)
    first_start = min(a["start"] for a in cfg["assets"])
    skipped = []
    for e in cfg.get("extra_tickers") or []:
        if isinstance(e, str):
            e = {"ticker": e}
        t = e["ticker"]
        start = e.get("start") or first_start
        try:
            out[t] = update_ticker(t, str(start), data_dir, trades_weekends=False,
                                   now=now, extra=True)
        except (DataDownloadError, DataValidationError) as err:
            if not e.get("optional"):
                raise
            msg = f"{t}: OPTIONAL extra skipped ({err}); downstream must use its documented fallback"
            log.warning("%s", msg)
            warns.append(msg)
            skipped.append({"ticker": t, "reason": str(err)})
            continue
        w = check_staleness(out[t], t, False, now)
        if w:
            warns.append(w)
    _write_extras_status(data_dir, skipped, [t for t in out if t not in {a["ticker"] for a in cfg["assets"]}])
    return out


def _write_extras_status(data_dir, skipped, loaded):
    """Record which extras were skipped (e.g. ^RVX) so downstream can fall back."""
    path = Path(data_dir) / "extras_status.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"loaded": sorted(loaded), "skipped": skipped}, indent=2))


def load_prices(ticker, config=None, now=None, check=True):
    """Read a stored ticker (no network). Logs a warning if the data is stale."""
    cfg = config or load_config()
    df = pd.read_parquet(_cache_path(cfg["data_dir"], ticker))
    if check:
        tw = any(a["ticker"] == ticker and a["trades_weekends"] for a in cfg["assets"])
        check_staleness(df, ticker, tw, now)
    return df


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        warns = []
        update_all(warnings=warns)
    except (DataDownloadError, DataValidationError) as e:
        log.error("%s", e)
        raise SystemExit(1)
    if warns:
        bar = "!" * 70
        print(f"\n{bar}\nWARNING: {len(warns)} stale data warning(s)\n{bar}")
        for w in warns:
            print(f"  * {w}")
        print(bar)


if __name__ == "__main__":
    main()
