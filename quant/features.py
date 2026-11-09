"""Stage 2 (spec v2) - causal feature engineering.

Every feature in row ``t`` of an asset's frame uses only information known at the
close of that asset's bar ``t``, EXCEPT the calendar features, which describe day
t+1 and are computed from the calendar / fixed FOMC schedule only (see below).
Only trailing pandas ops are used (rolling/ewm/diff/shift with positive lags); there
is no centered window, no ``shift(-n)`` in features and no global scaling.
Prices are never forward-filled.

Conventions
-----------
* Each asset keeps its own calendar (stocks: NYSE sessions; BTC: every UTC day).
  Windows count that asset's own bars.
* Returns are LOG returns. ``ret_lag_1`` is the latest return (close[t-1] -> close[t]),
  ``ret_lag_k`` the return realised k-1 bars earlier. ``r5`` = sum of the last 5 daily
  log returns.
* Annualisation: sqrt(365) for BTC, sqrt(252) for ETFs, everywhere (volatilities and
  the realised variance inside the VRP; variances use the squared factor).
* Volume change ``vol_chg_20`` = Volume / mean(Volume, 20) - 1; 0.0 if the trailing
  mean volume is 0 (missing volume = "no information").
* Missing values: Stage 2 keeps NaN (no imputation). ``build_dataset`` drops warm-up
  rows using ``CORE_EXCLUDED`` (below) so rows before ^VIX3M (2007) / ^VXN (2001)
  exist survive with NaN in vix_ts / vix_backwardation / vrp.

Target
------
``next_return`` = Close[t+1] / Close[t] - 1 on the asset's OWN calendar (BTC: next
calendar day, stocks: next row = next session); ``target`` = 1.0 if next_return > 0
else 0.0 (a zero return is "not up" -> 0). The final row has NaN for both but is
KEPT for prediction. Targets never enter features.

Papers / formulas
-----------------
* Yang-Zhang volatility: Yang & Zhang (2000), "Drift-independent volatility
  estimation based on high, low, open, and close prices", J. Business 73(3);
  see RESEARCH_REPORT.md "Volatility predicts risk...".
* Short-term reversal R5: Baltussen, van Bekkum & Da (2019), JFE (RESEARCH_REPORT.md).
* Variance risk premium: Drechsler & Yaron (2011), VRP = (IV/100)^2 - RV_21 with
  RV_21 = (ann/21) * sum of last 21 squared daily log returns (RESEARCH_REPORT.md).
* VIX term structure / backwardation: Fassas & Hourvouliades (2019).
* Trend horizons: Moskowitz, Ooi & Pedersen (2012) (ETFs); Liu, Tsyvinski & Wu (2022)
  (BTC 1-4 weeks).
* IWM uses IWM's OWN prices, never ^RUT (Lo & MacKinlay 1990, nonsynchronous trading).

Calendar features (describe day t+1) - why they are NOT lookahead
-----------------------------------------------------------------
``dow_next``, ``is_weekend_next`` and ``pre_fomc`` depend only on (a) the NYSE
holiday calendar and (b) the FOMC *scheduled* announcement dates, both published
long before day t. They use no prices and no price-row presence, so they cannot leak
the future (the next session is taken from the exchange calendar, not from the next
data row, which would be unknown on the final row). Unscheduled FOMC meetings are
excluded from the file precisely because they were not knowable in advance. If the
next session lies beyond the calendar (> 2027-11) a business-day fallback is used;
if it lies beyond the last FOMC date in the file ``pre_fomc`` is NaN.

BTC cross-asset (alignment)
---------------------------
A BTC bar dated t closes at 00:00 UTC of t+1, after the US close (16:00 ET) of date
t. So SPY/QQQ/IWM/VIX values dated <= t are all known when BTC's bar t closes:
backward as-of join (``searchsorted``), weekends/holidays carry the last stock value
and ``stale`` = 1 whenever ANY carried-over source date differs from t.
``btc_spy_corr_60``: for each SPY session s with previous session p, pair
SPY ln(C_s/C_p) with BTC ln(C_s/C_p) using BTC's closes dated s and p (same
intervals as the SPY return; BTC's UTC close is a few hours after the US close, a
small documented mismatch). The 60-session rolling Pearson correlation of these pairs
(min 60 valid pairs) is then as-of joined onto BTC dates, so it only uses sessions
<= t and BTC closes <= t.
"""
import argparse
import json
import logging
import sys

import numpy as np
import pandas as pd

from quant.config import load_config, required_extra_tickers
from quant.data import _xnys, load_fomc_dates, load_prices

log = logging.getLogger(__name__)

RET_LAGS = (1, 2, 3, 5, 10, 20)
VOL_WINDOWS = (10, 20, 60)
YZ_WINDOW = 21
RV_WINDOW = 21
MA_WINDOWS = (50, 200)
RSI_PERIOD = 14
TREND_ETF = (63, 126, 252)
TREND_BTC = (7, 14, 21, 28)
NEW_HIGH_WINDOW = 20
CORR_WINDOW = 60
SPREAD_DAYS = 5
ETFS = ("SPY", "QQQ", "IWM")
BTC = "BTC-USD"
STOCK_SOURCES = ("SPY", "QQQ", "IWM")
# implied-vol index used in the VRP; ^RVX is optional (Yahoo does not serve it) and
# falls back to ^VIX when it is not in the price dict (loaded only if available).
VOL_INDEX = {"SPY": "^VIX", "QQQ": "^VXN", "IWM": "^RVX"}
VOL_INDEX_FALLBACK = "^VIX"

TARGET_COLS = ["target", "next_return"]

# group -> every column any asset may have (each column in exactly one group)
feature_groups = {
    "returns": [f"ret_lag_{k}" for k in RET_LAGS] + ["overnight_gap", "intraday_ret"],
    "reversal": ["r5"],
    "volatility": [f"vol_{w}" for w in VOL_WINDOWS] + ["yz_vol_21", "vol_ratio_10_60",
                                                        "vol_ratio_20_60", "vol_chg_20"],
    "vix_stress": ["vix_level", "vix_chg", "vix_ts", "vix_backwardation", "vrp"],
    "trend": [f"trend_{k}" for k in TREND_ETF + TREND_BTC] + ["new_high_20"]
             + [f"dist_ma_{w}" for w in MA_WINDOWS] + ["rsi_14"],
    "calendar": ["dow_next", "is_weekend_next", "pre_fomc"],
    "cross_asset": ["spy_ret", "qqq_ret", "iwm_ret", "stale", "btc_spy_corr_60",
                    "iwm_spy_spread_5"],
}

# Columns that may legitimately be NaN in the dataset because their SOURCE series start
# late (^VIX3M 2007, ^VXN 2001). They never cause a warm-up row to be dropped.
CORE_EXCLUDED = frozenset({"vix_ts", "vix_backwardation", "vrp"})

MIN_ROWS = max(max(TREND_ETF), max(MA_WINDOWS), max(VOL_WINDOWS), YZ_WINDOW, RSI_PERIOD) + 1
HINT = "run `python -m quant.data` to download/update the price data"


def feature_columns(df):
    return [c for c in df.columns if c not in TARGET_COLS]


def feature_groups_for(columns) -> dict:
    """``feature_groups`` restricted to ``columns`` (empty groups omitted)."""
    cols = set(columns)
    out = {g: [c for c in v if c in cols] for g, v in feature_groups.items()}
    return {g: v for g, v in out.items() if v}


def core_columns(columns):
    """Explicit set of columns whose NaN marks a warm-up row (everything but CORE_EXCLUDED)."""
    return [c for c in columns if c not in CORE_EXCLUDED]


# --------------------------------------------------------------------------- helpers
def _asof_pos(src_index: pd.DatetimeIndex, idx: pd.DatetimeIndex) -> np.ndarray:
    """Position of the last source date <= idx (-1 if none): backward as-of."""
    return src_index.searchsorted(idx, side="right") - 1


def _asof_values(src: pd.Series, idx: pd.DatetimeIndex) -> np.ndarray:
    pos = _asof_pos(src.index, idx)
    out = np.full(len(idx), np.nan)
    ok = pos >= 0
    out[ok] = src.to_numpy(dtype="float64")[pos[ok]]
    return out


def _asof_dates(src_index: pd.DatetimeIndex, idx: pd.DatetimeIndex) -> np.ndarray:
    """As-of source date as int64 ns (min int where no source date exists)."""
    pos = _asof_pos(src_index, idx)
    v = src_index.asi8.copy()
    return np.where(pos >= 0, v[np.clip(pos, 0, None)], np.iinfo("int64").min)


_SESSIONS = None


def _sessions() -> pd.DatetimeIndex:
    global _SESSIONS
    if _SESSIONS is None:
        s = _xnys().sessions
        _SESSIONS = pd.DatetimeIndex(s.tz_localize(None) if s.tz is not None else s).astype("datetime64[ns]")
    return _SESSIONS


def next_session_dates(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Next NYSE session strictly after each date (calendar only, no price data)."""
    s = _sessions()
    pos = s.searchsorted(idx, side="right")
    ok = pos < len(s)
    out = (idx + pd.offsets.BDay(1)).astype("datetime64[ns]").to_numpy().copy()   # fallback
    out[ok] = s.to_numpy()[pos[ok]]
    return pd.DatetimeIndex(out)


def yang_zhang_vol(df: pd.DataFrame, n: int, ann: float) -> pd.Series:
    """Rolling n-day Yang-Zhang volatility, annualised by sqrt(ann).

    Yang & Zhang (2000): s^2 = s_o^2 + k s_c^2 + (1-k) s_rs^2 with
    s_o^2 = sample var of overnight ln(O_t/C_{t-1}), s_c^2 = sample var of
    ln(C_t/O_t), s_rs^2 = mean of Rogers-Satchell terms
    ln(H/C)ln(H/O) + ln(L/C)ln(L/O), and k = 0.34 / (1.34 + (n+1)/(n-1))."""
    o, h, l, c = (df[x].astype("float64") for x in ("Open", "High", "Low", "Close"))
    ov = np.log(o / c.shift(1))
    oc = np.log(c / o)
    rs = np.log(h / c) * np.log(h / o) + np.log(l / c) * np.log(l / o)
    k = 0.34 / (1.34 + (n + 1) / (n - 1))
    var = ov.rolling(n).var() + k * oc.rolling(n).var() + (1 - k) * rs.rolling(n).mean()
    return np.sqrt(ann * var.clip(lower=0.0))


def _rsi(close: pd.Series, period=RSI_PERIOD) -> pd.Series:
    d = close.diff()
    gain = d.clip(lower=0.0)
    loss = (-d).clip(lower=0.0)
    kw = dict(alpha=1.0 / period, adjust=False, min_periods=period, ignore_na=False)
    ag = gain.ewm(**kw).mean()
    al = loss.ewm(**kw).mean()
    rsi = 100.0 - 100.0 / (1.0 + ag / al.where(al > 0))
    rsi = rsi.where(~((al == 0) & (ag > 0)), 100.0)
    rsi = rsi.where(~((al == 0) & (ag == 0)), 50.0)
    return rsi.where(ag.notna() & al.notna())


def _ann(ticker) -> float:
    return 365.0 if ticker == BTC else 252.0


def _own_features(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Features from the asset's own OHLCV (no other series)."""
    ann = _ann(ticker)
    close = df["Close"].astype("float64")
    lr = np.log(close).diff()
    f = pd.DataFrame(index=df.index)
    for k in RET_LAGS:
        f[f"ret_lag_{k}"] = lr.shift(k - 1)
    f["r5"] = lr.rolling(5).sum()
    if ticker != BTC:
        f["overnight_gap"] = np.log(df["Open"].astype("float64") / close.shift(1))
        f["intraday_ret"] = np.log(close / df["Open"].astype("float64"))
    for w in VOL_WINDOWS:
        f[f"vol_{w}"] = lr.rolling(w).std() * np.sqrt(ann)
    f["yz_vol_21"] = yang_zhang_vol(df, YZ_WINDOW, ann)
    f["vol_ratio_10_60"] = f["vol_10"] / f["vol_60"].where(f["vol_60"] > 0)
    f["vol_ratio_20_60"] = f["vol_20"] / f["vol_60"].where(f["vol_60"] > 0)
    for k in (TREND_BTC if ticker == BTC else TREND_ETF):
        f[f"trend_{k}"] = np.log(close / close.shift(k))
    if ticker == BTC:
        hi = close.rolling(NEW_HIGH_WINDOW).max()
        f["new_high_20"] = (close >= hi).astype("float64").where(hi.notna())
    for w in MA_WINDOWS:
        f[f"dist_ma_{w}"] = close / close.rolling(w).mean() - 1.0
    f["rsi_14"] = _rsi(close)
    vol = df["Volume"].astype("float64")
    vm = vol.rolling(20).mean()
    vchg = vol / vm.where(vm > 0) - 1.0
    f["vol_chg_20"] = vchg.where(~((vm <= 0) & vm.notna()), 0.0)
    return f


def _calendar_features(ticker, idx, cfg, fomc_dates=None) -> pd.DataFrame:
    f = pd.DataFrame(index=idx)
    if ticker == BTC:
        nxt = idx + pd.Timedelta(days=1)
        f["dow_next"] = nxt.day_of_week.astype("float64")
        f["is_weekend_next"] = (nxt.day_of_week >= 5).astype("float64")
        return f
    nxt = next_session_dates(idx)
    f["dow_next"] = nxt.day_of_week.astype("float64")
    fomc = fomc_dates if fomc_dates is not None else _load_fomc(cfg)
    flag = nxt.isin(fomc).astype("float64")
    flag = np.where(nxt > fomc.max(), np.nan, flag)
    f["pre_fomc"] = flag
    return f


def _load_fomc(cfg):
    path = cfg.get("fomc_path")
    try:
        return load_fomc_dates(path)
    except FileNotFoundError as e:
        raise ValueError(f"FOMC dates file not found ({e}); data/fomc_dates.csv is required "
                         "for SPY/QQQ/IWM calendar features") from e


def _vol_index_for(ticker, px):
    pref = VOL_INDEX[ticker]
    if pref in px:
        return pref
    log.info("%s unavailable for %s: using %s for the VRP", pref, ticker, VOL_INDEX_FALLBACK)
    return VOL_INDEX_FALLBACK


def _vix_features(ticker, px, idx, f) -> pd.DataFrame:
    vix = px["^VIX"]["Close"].astype("float64")
    v3 = px["^VIX3M"]["Close"].astype("float64")
    out = pd.DataFrame(index=idx)
    out["vix_level"] = _asof_values(vix, idx)
    out["vix_chg"] = _asof_values(vix.pct_change(), idx)
    ts = _asof_values(v3, idx)
    out["vix_ts"] = out["vix_level"] / ts
    out["vix_backwardation"] = (out["vix_level"] > ts).astype("float64").where(out["vix_ts"].notna())
    if ticker in ETFS:
        iv = px[_vol_index_for(ticker, px)]["Close"].astype("float64")
        lr = np.log(px[ticker]["Close"].astype("float64")).diff()
        rv21 = (252.0 / RV_WINDOW) * (lr ** 2).rolling(RV_WINDOW).sum()
        out["vrp"] = (_asof_values(iv, idx) / 100.0) ** 2 - rv21
    return out


def _log_ret(px, t):
    return np.log(px[t]["Close"].astype("float64")).diff()


def _btc_cross(px, idx) -> pd.DataFrame:
    out = pd.DataFrame(index=idx)
    stale = np.zeros(len(idx), dtype=bool)
    for t in STOCK_SOURCES:
        out[f"{t.lower()}_ret"] = _asof_values(_log_ret(px, t), idx)
        stale |= _asof_dates(px[t].index, idx) != idx.asi8
    stale |= _asof_dates(px["^VIX"].index, idx) != idx.asi8
    out["stale"] = stale.astype("float64")
    # BTC-SPY correlation over identical intervals (SPY session p -> s), see module doc
    sess = px["SPY"].index
    spy_lr = _log_ret(px, "SPY")
    btc_c = px[BTC]["Close"].astype("float64")
    c_s = pd.Series(_asof_values(btc_c, sess), index=sess)
    btc_lr = np.log(c_s).diff()
    both = pd.concat([spy_lr, btc_lr], axis=1, keys=["s", "b"])
    corr = both["s"].rolling(CORR_WINDOW, min_periods=CORR_WINDOW).corr(both["b"])
    out["btc_spy_corr_60"] = _asof_values(corr, idx)
    return out


def _iwm_cross(px, idx) -> pd.DataFrame:
    out = pd.DataFrame(index=idx)
    r5 = {t: _log_ret(px, t).rolling(SPREAD_DAYS).sum() for t in ("IWM", "SPY")}
    out["iwm_spy_spread_5"] = _asof_values(r5["IWM"], idx) - _asof_values(r5["SPY"], idx)
    return out


# --------------------------------------------------------------------------- loading
def _skipped_extras(cfg):
    p = cfg["data_dir"] / "extras_status.json"
    try:
        return {s["ticker"] for s in json.loads(p.read_text()).get("skipped", [])}
    except (OSError, ValueError, KeyError, TypeError):
        return set()


def _load_all(cfg, prices):
    if prices is not None:
        return prices
    out = {}
    names = [a["ticker"] for a in cfg["assets"]] + required_extra_tickers(cfg)
    for t in names:
        try:
            out[t] = load_prices(t, cfg, check=False)
        except FileNotFoundError as e:
            raise ValueError(f"Price file for {t} not found ({e.filename or e}); {HINT}") from e
    skipped = _skipped_extras(cfg)
    for e in cfg.get("extra_tickers") or []:
        if isinstance(e, dict) and e.get("optional") and e["ticker"] not in skipped:
            try:
                out[e["ticker"]] = load_prices(e["ticker"], cfg, check=False)
            except FileNotFoundError:
                log.info("optional %s has no parquet; skipped", e["ticker"])
    return out


def _check_prices(cfg, px, ticker):
    needed = [a["ticker"] for a in cfg["assets"]] + required_extra_tickers(cfg)
    for t in needed:
        if t not in px:
            raise ValueError(f"Prices for required ticker {t} are missing; {HINT}")
    if len(px[ticker]) == 0:
        raise ValueError(f"Price frame for {ticker} is empty; {HINT}")
    n = len(px[ticker])
    if n < MIN_ROWS:
        raise ValueError(f"{ticker} has only {n} rows; need at least {MIN_ROWS} "
                         f"(longest window {MIN_ROWS - 1} + 1) to build features")


def build_features(ticker, config=None, prices=None, fomc_dates=None) -> pd.DataFrame:
    """Causal feature frame for ``ticker`` (all rows incl. warm-up NaNs and the last row).

    ``prices``: optional {ticker: OHLCV frame} (assets + extras); loaded from
    ``data_dir`` when omitted. ``fomc_dates``: optional DatetimeIndex (default:
    data/fomc_dates.csv, or cfg["fomc_path"]). No target columns here."""
    cfg = config or load_config()
    tickers = [a["ticker"] for a in cfg["assets"]]
    if ticker not in tickers:
        raise KeyError(f"{ticker} is not a configured asset")
    px = _load_all(cfg, prices)
    _check_prices(cfg, px, ticker)
    df = px[ticker]
    idx = df.index
    parts = [_own_features(df, ticker),
             _vix_features(ticker, px, idx, None),
             _calendar_features(ticker, idx, cfg, fomc_dates)]
    if ticker == BTC:
        parts.append(_btc_cross(px, idx))
    elif ticker == "IWM":
        parts.append(_iwm_cross(px, idx))
    f = pd.concat(parts, axis=1)
    order = [c for g in feature_groups.values() for c in g if c in f.columns]
    assert set(order) == set(f.columns), set(f.columns) ^ set(order)
    return f[order]


def make_target(close: pd.Series):
    """(next_return, target) on the series' own calendar; last row NaN."""
    nxt = close.shift(-1) / close - 1.0
    return nxt, (nxt > 0).astype("float64").where(nxt.notna())


def build_dataset(ticker, config=None, prices=None, fomc_dates=None) -> pd.DataFrame:
    """Features + ``target`` + ``next_return``.

    Warm-up rows (NaN in any ``core_columns``) are dropped; columns in
    ``CORE_EXCLUDED`` may stay NaN. The final prediction row is never dropped."""
    cfg = config or load_config()
    px = _load_all(cfg, prices)
    f = build_features(ticker, cfg, px, fomc_dates)
    nxt, tgt = make_target(px[ticker]["Close"].astype("float64"))
    d = f.copy()
    d["next_return"] = nxt
    d["target"] = tgt
    d = d[[*f.columns, "target", "next_return"]]
    keep = d[core_columns(f.columns)].notna().all(axis=1)
    keep.iloc[-1] = True
    d = d[keep]
    if len(d) < 2:
        raise ValueError(f"{ticker}: dataset would have only {len(d)} row(s) after dropping "
                         "warm-up/NaN-feature rows (constant prices or too little data?)")
    return d


def features_path(ticker, config=None):
    cfg = config or load_config()
    return cfg["data_dir"] / "features" / f"{ticker}.parquet"


def save_dataset(ticker, df, config=None):
    p = features_path(ticker, config)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(p)
    return p


def load_dataset(ticker, config=None) -> pd.DataFrame:
    return pd.read_parquet(features_path(ticker, config))


def main(argv=None):
    ap = argparse.ArgumentParser(description="Build and save feature datasets.")
    ap.add_argument("--config", default=None)
    ap.add_argument("tickers", nargs="*")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        cfg = load_config(args.config)
        px = _load_all(cfg, None)
        out = [(t, build_dataset(t, cfg, px)) for t in
               (args.tickers or [a["ticker"] for a in cfg["assets"]])]
    except (ValueError, KeyError, FileNotFoundError) as e:
        print(f"ERROR: {e.args[0] if e.args else e}", file=sys.stderr)
        raise SystemExit(1)
    for t, d in out:
        p = save_dataset(t, d, cfg)
        nan = d.isna().sum()
        nan = nan[nan > 0].to_dict()
        print(f"{t}: {len(d)} rows x {len(d.columns)} cols, {d.index[0].date()} -> "
              f"{d.index[-1].date()}, saved {p}")
        print(f"   NaN counts (non-zero only): {nan}")


if __name__ == "__main__":
    main()
