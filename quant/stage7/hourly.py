"""Stage 7c - hourly crypto data (ccxt public endpoints, no key) and daily features built from it.

Storage: ``data/hourly/{TICKER}.parquet`` (TICKER e.g. ``BTC-USD``), index = bar OPEN time
(tz-naive UTC, named ``Datetime``), columns Open High Low Close Volume (float64), plus
``data/hourly/status.json`` (exchange, quote, first/last bar, gap list per asset).

IMPORTANT data notes
--------------------
* Prices are USDT (Binance) or USD (Kraken/Coinbase) quotes, NOT the USD prices Yahoo uses for the
  daily BTC-USD series. USDT trades within a few tenths of a percent of USD, so hourly-based
  features are close to, but not identical with, a USD-based calculation.
* One exchange per file. We never splice two exchanges together (their prices and volumes differ);
  if the stored exchange differs from the one in use, ``HourlyExchangeMixError`` is raised.
* A bar opening at ``t`` is complete only when ``t + 1h <= now``. Incomplete bars are never stored.
* Gaps (exchange maintenance outages are real): up to 48 missing hours are WARNINGS, logged and
  recorded in status.json and never filled or interpolated; more than 48 missing hours is a
  ``DataValidationError``. Inspected on real Binance data (2017-08 to 2026-11): BTC and ETH each
  have 28 gaps, 27 of them 1-10 missing hours and ONE of 33 hours (2018-02-08 00:00 to 02-09
  10:00, an exchange outage). The originally planned 24h limit would have rejected the whole
  history for that single known outage, so the limit is 48h (still catches multi-day holes).

Alignment to the Yahoo daily close (the lookahead rule)
-------------------------------------------------------
Yahoo's BTC-USD daily bar dated D covers [D 00:00, D+1 00:00) UTC and is known at D+1 00:00 UTC.
The daily feature row dated D therefore uses ONLY hourly bars whose OPEN time is in
[D 00:00, D 23:00] (the 23:00 bar closes exactly at D+1 00:00). A bar opening at D+1 00:00 never
enters row D. Row D is the same "as of the daily close" information set as the Yahoo bar D.

Daily features (``daily_features``)
-----------------------------------
* ``rv_hourly``: sqrt(sum of the 24 squared hourly log returns of day D). The 00:00 bar's return
  is ln(Close/Open) of that bar (so day D needs no data from D-1); the others are
  ln(Close_h / Close_{h-1}). NaN unless all 24 bars of day D exist. Not annualised.
* ``ret_6h``: ln(Close of the 23:00 bar / Close of the 17:00 bar) (the last 6 hours before the
  close). ``ret_24h``: ln(Close of D's 23:00 bar / Close of D-1's 23:00 bar). NaN if a bar is missing.
* ``vol_concentration``: Herfindahl index sum(share_h^2) of the day's hourly volume shares
  (1/24 = perfectly even, 1 = all volume in one hour). NaN if < 24 bars or zero volume.
* ``har_rv_forecast``: HAR-RV (Corsi 2009) in log form, on the daily realised volatility above:
  log RV_{t+1} = b0 + b1 log RV_t + b2 log mean(RV_{t-4..t}) + b3 log mean(RV_{t-21..t}).
  Row D holds the forecast of RV on day D+1. Coefficients come from OLS on an EXPANDING window of
  pairs (x_s, y_{s+1}) whose TARGET day s+1 is <= R, where R is the latest refit day <= D. Refit
  days are the days whose ordinal date number is divisible by 21 (a calendar rule that does not
  depend on where the data frame starts, so it is itself causal). R <= D means every target used
  was already known at the close of D. Fewer than 250 training pairs -> NaN. The forecast is
  exp(prediction), i.e. the conditional MEDIAN of RV, not the mean (no variance/smearing
  correction). Means are of volatilities (not variances).

Everything is causal: appending or changing bars after day D never changes any row <= D that was
already computable (a day with missing bars is NaN until its bars exist; that is "not yet known",
not lookahead).
"""
import json
import logging
import time

import numpy as np
import pandas as pd

from quant.data import COLUMNS, DataValidationError, _now_utc, _write_atomic
from quant.stage7.config import dev_cutoff, load_stage7_config

log = logging.getLogger(__name__)

HOUR = pd.Timedelta(hours=1)
REFETCH_HOURS = 6                 # re-fetch this many hours before the last stored bar
MAX_GAP_WARN_HOURS = 48           # up to this many MISSING hours: warning; more: error (see docstring)
HAR_MIN_PAIRS = 250
HAR_REFIT_DAYS = 21
FEATURE_COLUMNS = ["rv_hourly", "ret_6h", "ret_24h", "vol_concentration", "har_rv_forecast"]


class HourlyDownloadError(RuntimeError):
    """A ccxt call failed (names exchange + symbol)."""


class HourlyExchangeMixError(HourlyDownloadError):
    """The stored file came from another exchange than the one now in use."""


# ----------------------------------------------------------------------------- network (thin)
_EXCHANGES = {}


def _get_exchange(exchange_id):
    if exchange_id not in _EXCHANGES:
        import ccxt
        _EXCHANGES[exchange_id] = getattr(ccxt, exchange_id)({"enableRateLimit": False})
    return _EXCHANGES[exchange_id]


def _rate_limit_ms(exchange_id):
    return float(_get_exchange(exchange_id).rateLimit)


def _fetch_page(exchange_id, symbol, since_ms, limit):
    """The single network call (mocked in tests): one page of [ms, o, h, l, c, v] rows."""
    return _get_exchange(exchange_id).fetch_ohlcv(symbol, timeframe="1h", since=since_ms, limit=limit)


def _symbol(cfg, exchange_id, base):
    return f"{base}/{cfg['hourly']['quote'][exchange_id]}"


def _fetch_all(exchange_id, symbol, since, limit, now, sleep=time.sleep):
    """Page forward from ``since`` until no new bars arrive or we pass ``now``."""
    since_ms = int(since.value // 10**6)
    now_ms = int(now.tz_convert("UTC").tz_localize(None).value // 10**6)
    rows = []
    try:
        pause = _rate_limit_ms(exchange_id) / 1000.0
        while since_ms <= now_ms:
            page = _fetch_page(exchange_id, symbol, since_ms, limit)
            if not page:
                break
            rows.extend(page)
            nxt = int(page[-1][0]) + 3_600_000
            if nxt <= since_ms:
                break
            since_ms = nxt
            sleep(pause)
    except HourlyDownloadError:
        raise
    except Exception as e:
        raise HourlyDownloadError(f"{exchange_id} {symbol}: download failed: {type(e).__name__}: {e}") from e
    return rows


def _to_frame(rows, exchange_id, symbol):
    if not rows:
        raise HourlyDownloadError(f"{exchange_id} {symbol}: returned no hourly bars")
    arr = np.array([r[:6] for r in rows], dtype="float64")
    idx = pd.DatetimeIndex(pd.to_datetime(arr[:, 0], unit="ms")).as_unit("ns")
    df = pd.DataFrame(arr[:, 1:6], index=idx, columns=COLUMNS)
    df.index.name = "Datetime"
    return df


# ----------------------------------------------------------------------------- storage helpers
def hourly_dir(cfg):
    return cfg["data_dir"] / "hourly"


def _path(cfg, ticker):
    return hourly_dir(cfg) / f"{ticker}.parquet"


def _status_path(cfg):
    return hourly_dir(cfg) / "status.json"


def read_status(cfg):
    p = _status_path(cfg)
    return json.loads(p.read_text()) if p.exists() else {"assets": {}}


def _write_status(cfg, status):
    p = _status_path(cfg)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(status, indent=2, sort_keys=True))


def drop_incomplete(df, ticker, now=None):
    """Drop bars with open + 1h > now (never store a partial bar)."""
    now = _now_utc(now).tz_localize(None)
    keep = df.index + HOUR <= now
    if (~keep).any():
        log.info("%s: dropping %d incomplete hourly bar(s)", ticker, int((~keep).sum()))
    return df[keep]


def find_gaps(df):
    """List of gaps: bars missing between two consecutive stored bars (never filled)."""
    if len(df) < 2:
        return []
    d = df.index[1:] - df.index[:-1]
    out = []
    for prev, cur, step in zip(df.index[:-1][d > HOUR], df.index[1:][d > HOUR], d[d > HOUR]):
        out.append({"after": str(prev), "before": str(cur),
                    "missing_hours": int(step / HOUR) - 1})
    return out


def validate_hourly(df, ticker):
    """Raise DataValidationError on a broken frame; return the gap list (warnings logged)."""
    problems = []
    if df.empty:
        problems.append("no rows")
    else:
        if df.index.has_duplicates:
            problems.append(f"{int(df.index.duplicated().sum())} duplicate timestamps")
        if not df.index.is_monotonic_increasing:
            problems.append("index not sorted")
        if df[COLUMNS].isna().any().any():
            problems.append(f"{int(df[COLUMNS].isna().sum().sum())} NaN values")
        px = df[["Open", "High", "Low", "Close"]]
        n = int((px <= 0).any(axis=1).sum())
        if n:
            problems.append(f"{n} bars with zero/negative prices")
        n = int((df["High"] < df[["Open", "Close"]].max(axis=1)).sum())
        if n:
            problems.append(f"{n} bars with High below Open/Close")
        n = int((df["Low"] > df[["Open", "Close"]].min(axis=1)).sum())
        if n:
            problems.append(f"{n} bars with Low above Open/Close")
        n = int((df["Volume"] < 0).sum())
        if n:
            problems.append(f"{n} bars with negative volume")
    gaps = []
    if not problems:
        gaps = find_gaps(df)
        for g in gaps:
            if g["missing_hours"] > MAX_GAP_WARN_HOURS:
                problems.append(f"gap of {g['missing_hours']} missing hours between "
                                f"{g['after']} and {g['before']}")
            else:
                log.warning("%s: hourly gap of %d missing hour(s) between %s and %s (kept, not filled)",
                            ticker, g["missing_hours"], g["after"], g["before"])
    for p in problems:
        log.error("%s: hourly validation problem: %s", ticker, p)
    if problems:
        raise DataValidationError(f"{ticker}: " + "; ".join(problems[:10]))
    return gaps


def check_staleness_hourly(df, ticker, tolerance_hours=2, now=None):
    """Warning message if the latest stored bar is older than the most recent completed bar
    minus the tolerance, else None (also logged)."""
    now = _now_utc(now).tz_localize(None)
    expected = now.floor("h") - HOUR           # open time of the most recent completed bar
    latest = df.index.max() if len(df) else None
    if latest is not None and latest >= expected - pd.Timedelta(hours=tolerance_hours):
        return None
    msg = (f"{ticker}: STALE HOURLY DATA - latest stored bar opened {latest}, but the most recent "
           f"completed bar opened {expected} (tolerance {tolerance_hours}h)")
    log.warning("%s", msg)
    return msg


# ----------------------------------------------------------------------------- update
def update_asset(ticker, cfg=None, now=None, sleep=time.sleep, warnings=None):
    """Create or incrementally update one asset's hourly parquet; return the frame."""
    cfg = cfg or load_stage7_config()
    hc = cfg["hourly"]
    base = hc["assets"][ticker]
    now_ts = _now_utc(now)
    path = _path(cfg, ticker)
    status = read_status(cfg)
    stored_ex = status["assets"].get(ticker, {}).get("exchange")
    old = pd.read_parquet(path) if path.exists() else None
    start = pd.Timestamp(hc["start"])
    fetch_from = start
    if old is not None and len(old):
        fetch_from = max(start, old.index.max() - REFETCH_HOURS * HOUR)
    candidates = [hc["exchange"]] + [e for e in hc.get("fallbacks", []) if e != hc["exchange"]]
    errors, new, used = [], None, None
    for ex in candidates:
        if stored_ex and ex != stored_ex:
            raise HourlyExchangeMixError(
                f"{ticker}: stored hourly data comes from {stored_ex!r} but {ex!r} would be used now"
                f" ({'; '.join(errors) or 'no earlier error'}); refusing to mix exchanges in one file."
                f" Delete {path} and status.json entry deliberately to start over on the new exchange.")
        sym = _symbol(cfg, ex, base)
        try:
            rows = _fetch_all(ex, sym, fetch_from, int(hc["page_limit"]), now_ts, sleep)
            new = _to_frame(rows, ex, sym)
            used, used_sym = ex, sym
            break
        except HourlyDownloadError as e:
            log.warning("%s: %s failed (%s)", ticker, ex, e)
            errors.append(str(e))
    if new is None:
        raise HourlyDownloadError(f"{ticker}: all exchanges failed: " + " | ".join(errors))
    if used != hc["exchange"]:
        log.warning("%s: primary exchange %s unavailable; using fallback %s", ticker, hc["exchange"], used)
    df = new if old is None or not len(old) else pd.concat([old, new])
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df = drop_incomplete(df, ticker, now_ts)
    validate_hourly(df, ticker)
    gaps = find_gaps(df)
    if df.index[0] > start + pd.Timedelta(days=2):
        log.warning("%s: %s history starts %s, later than the configured start %s (recorded as the actual start)",
                    ticker, used, df.index[0], start.date())
    _write_atomic(df, path)
    status["assets"][ticker] = {
        "exchange": used, "symbol": used_sym, "quote": hc["quote"][used],
        "first_bar": str(df.index[0]), "last_bar": str(df.index[-1]), "bars": int(len(df)),
        "gaps": gaps, "updated_at_utc": str(now_ts)}
    _write_status(cfg, status)
    w = check_staleness_hourly(df, ticker, hc["stale_tolerance_hours"], now_ts)
    if w and warnings is not None:
        warnings.append(w)
    log.info("%s: %d hourly bars, %s to %s (%s)", ticker, len(df), df.index[0], df.index[-1], used)
    return df


def update_all(cfg=None, now=None, sleep=time.sleep, warnings=None):
    cfg = cfg or load_stage7_config()
    return {t: update_asset(t, cfg, now, sleep, warnings) for t in cfg["hourly"]["assets"]}


def load_hourly(ticker, cfg=None, include_holdout=False):
    """Read stored hourly bars (no network).

    DEFAULT returns only bars that OPENED on or before the pinned hold-out cutoff day (open time <
    cutoff day + 1 day, i.e. through 23:00 UTC of the cutoff day). ``include_holdout=True`` returns
    everything and is for Stage 7 hold-out evaluation ONLY."""
    cfg = cfg or load_stage7_config()
    df = pd.read_parquet(_path(cfg, ticker))
    if include_holdout:
        return df
    return df[df.index < dev_cutoff(cfg) + pd.Timedelta(days=1)]


# ----------------------------------------------------------------------------- features
def _har_forecast(rv, min_pairs=HAR_MIN_PAIRS, refit_days=HAR_REFIT_DAYS):
    """Causal log-HAR-RV forecast, see module docstring. ``rv``: daily Series, full daily index."""
    lrv = np.log(rv.where(rv > 0))
    f1 = lrv
    f2 = np.log(rv.where(rv > 0).rolling(5, min_periods=5).mean())
    f3 = np.log(rv.where(rv > 0).rolling(22, min_periods=22).mean())
    X = np.column_stack([np.ones(len(rv)), f1.to_numpy(), f2.to_numpy(), f3.to_numpy()])
    y_next = lrv.shift(-1).to_numpy()          # target of pair s lives on day s+1 (pair is used only
    ords = np.array([d.toordinal() for d in rv.index])   # once s+1 <= R, see below)
    xok = np.isfinite(X).all(axis=1)
    pair_ok = xok & np.isfinite(y_next)
    out = np.full(len(rv), np.nan)
    cache = {}
    for i in range(len(rv)):
        if not xok[i]:
            continue
        back = int(ords[i] % refit_days)       # R = i - back (rows are consecutive days)
        r = i - back
        if r not in cache:
            # pairs with target day <= R: pair index s satisfies s + 1 <= r
            mask = pair_ok[:max(r, 0)] if r > 0 else np.zeros(0, bool)
            if mask.sum() < min_pairs:
                cache[r] = None
            else:
                Xs, ys = X[:len(mask)][mask], y_next[:len(mask)][mask]
                cache[r] = np.linalg.lstsq(Xs, ys, rcond=None)[0]
        beta = cache[r]
        if beta is not None:
            out[i] = np.exp(X[i] @ beta)
    return pd.Series(out, index=rv.index)


def daily_features(hourly_df):
    """Daily features from hourly bars; see the module docstring for the exact definitions and the
    alignment rule. Index: normalized tz-naive dates from the first to the last day with a bar."""
    cols = pd.Index(FEATURE_COLUMNS)
    if len(hourly_df) == 0:
        return pd.DataFrame(columns=cols, dtype="float64")
    h = hourly_df.sort_index()
    h = h[~h.index.duplicated(keep="last")]
    day = h.index.normalize()
    hour = h.index.hour
    days = pd.date_range(day.min(), day.max(), freq="D")

    def mat(col):
        s = pd.Series(h[col].to_numpy(), index=pd.MultiIndex.from_arrays([day, hour]))
        return s.unstack().reindex(index=days, columns=range(24))

    C, O, V = mat("Close"), mat("Open"), mat("Volume")
    lc = np.log(C.to_numpy())
    r = np.empty_like(lc)
    r[:, 0] = np.log(C[0].to_numpy() / O[0].to_numpy())
    r[:, 1:] = lc[:, 1:] - lc[:, :-1]
    rv = pd.Series(np.sqrt((r ** 2).sum(axis=1)), index=days)       # NaN if any of 24 missing
    ret_6h = pd.Series(lc[:, 23] - lc[:, 17], index=days)
    c23 = pd.Series(lc[:, 23], index=days)
    ret_24h = c23 - c23.shift(1)
    v = V.to_numpy()
    tot = v.sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        conc = np.where(tot > 0, ((v / tot[:, None]) ** 2).sum(axis=1), np.nan)
    out = pd.DataFrame({"rv_hourly": rv, "ret_6h": ret_6h, "ret_24h": ret_24h,
                        "vol_concentration": conc}, index=days)
    out["har_rv_forecast"] = _har_forecast(rv)
    out.index.name = "Date"
    return out[FEATURE_COLUMNS]


# ----------------------------------------------------------------------------- CLI
def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        warns = []
        cfg = load_stage7_config()
        frames = update_all(cfg, warnings=warns)
    except (HourlyDownloadError, DataValidationError, FileNotFoundError, ValueError) as e:
        print(f"ERROR: {e}")
        raise SystemExit(1)
    status = read_status(cfg)
    for t, df in frames.items():
        s = status["assets"][t]
        big = max([g["missing_hours"] for g in s["gaps"]], default=0)
        print(f"{t}: exchange={s['exchange']} ({s['symbol']}) first={df.index[0]} last={df.index[-1]} "
              f"bars={len(df)} gaps={len(s['gaps'])} (largest {big}h)")
    for w in warns:
        print(f"WARNING: {w}")


if __name__ == "__main__":
    main()
