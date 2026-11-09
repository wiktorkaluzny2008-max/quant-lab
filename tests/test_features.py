import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from quant import features as F
from quant.config import load_config
from quant.data import _xnys

CFG = {
    "assets": [
        {"name": "S", "ticker": "SPY", "start": "2000-01-01", "trades_weekends": False},
        {"name": "Q", "ticker": "QQQ", "start": "2000-01-01", "trades_weekends": False},
        {"name": "I", "ticker": "IWM", "start": "2000-01-01", "trades_weekends": False},
        {"name": "B", "ticker": "BTC-USD", "start": "2000-01-01", "trades_weekends": True},
    ],
    "extra_tickers": ["^VIX", "^VIX3M", "^VXN"],
}
TICKERS = ["SPY", "QQQ", "IWM", "BTC-USD"]
SERIES = TICKERS + ["^VIX", "^VIX3M", "^VXN"]
FIELDS = ["Open", "High", "Low", "Close", "Volume"]


def sessions(start, n):
    s = _xnys().sessions
    s = s[s >= pd.Timestamp(start)][:n]
    return pd.DatetimeIndex(s, name="Date")


def _ohlcv_from(c, idx, rng, vol=True):
    c = np.asarray(c, dtype=float)
    o = c * np.exp(rng.normal(0, 0.004, len(c)))
    hi = np.maximum(o, c) * np.exp(np.abs(rng.normal(0, 0.004, len(c))))
    lo = np.minimum(o, c) * np.exp(-np.abs(rng.normal(0, 0.004, len(c))))
    v = rng.integers(100, 1000, len(c)).astype(float) if vol else np.zeros(len(c))
    return pd.DataFrame({"Open": o, "High": hi, "Low": lo, "Close": c, "Volume": v},
                        index=pd.DatetimeIndex(idx, name="Date"))


def _walk(idx, rng, start=100.0, sd=0.01):
    return start * np.exp(np.cumsum(rng.normal(0, sd, len(idx))))


def make_prices(seed=0, n=900):
    rng = np.random.default_rng(seed)
    sess = sessions("2019-01-02", n)
    alld = pd.date_range("2019-01-02", sess[-1] + pd.Timedelta(days=2))
    px = {t: _ohlcv_from(_walk(sess, rng), sess, rng) for t in ("SPY", "QQQ", "IWM")}
    px["BTC-USD"] = _ohlcv_from(_walk(alld, rng, 5000.0, 0.03), alld, rng)
    px["^VIX"] = _ohlcv_from(_walk(sess, rng, 20.0, 0.03), sess, rng, vol=False)
    px["^VIX3M"] = _ohlcv_from(_walk(sess[400:], rng, 22.0, 0.02), sess[400:], rng, vol=False)
    px["^VXN"] = _ohlcv_from(_walk(sess[300:], rng, 25.0, 0.03), sess[300:], rng, vol=False)
    return px


def truncate(px, cut):
    return {k: v[v.index <= cut] for k, v in px.items()}


def all_features(px):
    return {t: F.build_features(t, CFG, px) for t in TICKERS}


# ------------------------------------------------------------ leakage tests
@pytest.mark.parametrize("seed", [0, 1])
@pytest.mark.parametrize("frac", [0.35, 0.6, 0.85, 0.98])
@pytest.mark.parametrize("ticker", TICKERS)
def test_truncation_identical(seed, frac, ticker):
    """REQUIRED: features on data truncated at a random date equal the full-data ones
    (ALL inputs - own OHLCV, other assets, VIX/VIX3M/VXN - truncated; FOMC fixed)."""
    px = make_prices(seed)
    rng = np.random.default_rng(seed * 100 + int(frac * 100))
    days = px["SPY"].index
    cut = days[int(len(days) * frac) - int(rng.integers(0, 5))]
    full = F.build_features(ticker, CFG, px)
    part = F.build_features(ticker, CFG, truncate(px, cut))
    assert len(part) > 0 and part.index[-1] <= cut
    assert_frame_equal(full.loc[:part.index[-1]], part)


def test_appended_wild_future_leaves_past_unchanged():
    px = make_prices(3)
    base = all_features(px)
    end = max(v.index[-1] for v in px.values())
    wild = {}
    for k, v in px.items():
        fut = pd.date_range(end + pd.Timedelta(days=1), periods=30, freq="D" if k == "BTC-USD" else "B")
        extra = pd.DataFrame({c: np.linspace(1e6, 1e-3, 30) for c in v.columns}, index=fut)
        extra.index.name = "Date"
        wild[k] = pd.concat([v, extra])
    new = all_features(wild)
    for t in TICKERS:
        assert_frame_equal(new[t].loc[:base[t].index[-1]], base[t], check_freq=False)


@pytest.mark.parametrize("field", FIELDS)
@pytest.mark.parametrize("changed", SERIES)
def test_perturbing_one_future_field_changes_no_earlier_row(changed, field):
    px = make_prices(4)
    base = all_features(px)
    d = px[changed].index[len(px[changed]) // 2 + 37]
    px2 = {k: v.copy() for k, v in px.items()}
    px2[changed].loc[d, field] *= 7.0
    new = all_features(px2)
    before = d - pd.Timedelta(days=1)
    for t in TICKERS:
        assert_frame_equal(new[t].loc[:before], base[t].loc[:before])


def test_no_feature_equals_or_derives_from_target():
    px = make_prices(11, 1800)
    for t in TICKERS:
        d = F.build_dataset(t, CFG, px).iloc[:-1]
        cols = F.feature_columns(d)
        assert not {"target", "next_return"} & set(cols)
        for c in cols:
            x = d[c].fillna(0)
            if x.std() > 0:
                assert abs(np.corrcoef(x, d["target"])[0, 1]) < 0.3, c
                assert abs(np.corrcoef(x, d["next_return"])[0, 1]) < 0.3, c
        # a shifted copy of next_return would equal it exactly on some lag
        for c in cols:
            for lag in (-1, 0, 1):
                assert not np.allclose(d[c].shift(lag).fillna(-9), d["next_return"].fillna(-9)), c


# ------------------------------------------------------------ hand-computed features
def _frame(c, idx, o=None, h=None, l=None, v=1.0):
    c = np.asarray(c, float)
    return pd.DataFrame({"Open": c if o is None else o, "High": c if h is None else h,
                         "Low": c if l is None else l, "Close": c, "Volume": v},
                        index=pd.DatetimeIndex(idx, name="Date"))


def test_returns_reversal_overnight_intraday_hand():
    c = np.array([100, 102, 101, 105, 104, 103, 107.0])
    o = np.array([100, 101, 103, 100, 106, 104, 102.0])
    idx = sessions("2021-03-01", len(c))
    f = F._own_features(_frame(c, idx, o=o), "SPY")
    lr = np.log(c[1:] / c[:-1])
    assert f["ret_lag_1"].iloc[6] == pytest.approx(lr[5])
    assert f["ret_lag_3"].iloc[6] == pytest.approx(lr[3])
    assert f["r5"].iloc[6] == pytest.approx(lr[1:6].sum())
    assert np.isnan(f["r5"].iloc[3]) and not np.isnan(f["r5"].iloc[5])
    assert f["overnight_gap"].iloc[3] == pytest.approx(np.log(100 / 101))
    assert f["intraday_ret"].iloc[3] == pytest.approx(np.log(105 / 100))
    assert np.isnan(f["overnight_gap"].iloc[0])
    b = F._own_features(_frame(c, idx, o=o), "BTC-USD")
    assert "overnight_gap" not in b and "intraday_ret" not in b and "r5" in b


def test_close_vol_annualisation_etf_vs_btc():
    px = make_prices(1)
    s = F._own_features(px["SPY"], "SPY")
    b = F._own_features(px["BTC-USD"], "BTC-USD")
    lr = np.log(px["SPY"]["Close"]).diff()
    assert s["vol_20"].iloc[100] == pytest.approx(lr.iloc[81:101].std(ddof=1) * np.sqrt(252))
    lb = np.log(px["BTC-USD"]["Close"]).diff()
    assert b["vol_20"].iloc[100] == pytest.approx(lb.iloc[81:101].std(ddof=1) * np.sqrt(365))
    assert s["vol_ratio_20_60"].iloc[100] == pytest.approx(s["vol_20"].iloc[100] / s["vol_60"].iloc[100])


def _yz_independent(O, H, L, C, i, n, ann):
    """Plain-loop Yang-Zhang for the n bars ending at i (needs C[i-n])."""
    ov = [np.log(O[j] / C[j - 1]) for j in range(i - n + 1, i + 1)]
    oc = [np.log(C[j] / O[j]) for j in range(i - n + 1, i + 1)]
    rs = [np.log(H[j] / C[j]) * np.log(H[j] / O[j]) + np.log(L[j] / C[j]) * np.log(L[j] / O[j])
          for j in range(i - n + 1, i + 1)]
    mo, mc = sum(ov) / n, sum(oc) / n
    so = sum((x - mo) ** 2 for x in ov) / (n - 1)
    sc = sum((x - mc) ** 2 for x in oc) / (n - 1)
    k = 0.34 / (1.34 + (n + 1) / (n - 1))
    return float(np.sqrt(ann * (so + k * sc + (1 - k) * sum(rs) / n)))


@pytest.mark.parametrize("ticker,ann", [("SPY", 252), ("BTC-USD", 365)])
def test_yang_zhang_matches_independent_formula(ticker, ann):
    px = make_prices(2)[ticker]
    f = F._own_features(px, ticker)
    O, H, L, C = (px[x].to_numpy() for x in ("Open", "High", "Low", "Close"))
    for i in (21, 60, 300):
        assert f["yz_vol_21"].iloc[i] == pytest.approx(_yz_independent(O, H, L, C, i, 21, ann), rel=1e-9)
    assert np.isnan(f["yz_vol_21"].iloc[20])


def test_yang_zhang_vs_close_to_close_same_order_of_magnitude():
    px = make_prices(5, 1500)["SPY"]
    f = F._own_features(px, "SPY")
    ratio = (f["yz_vol_21"] / f["vol_20"]).dropna()
    assert 0.5 < ratio.median() < 2.0


def test_trend_ma_new_high_hand():
    n = 300
    c = 100 * np.exp(np.cumsum(np.sin(np.arange(n) / 7.0) * 0.01 + 0.001))
    idx = sessions("2020-01-02", n)
    s = pd.Series(c, index=idx)
    f = F._own_features(_frame(c, idx), "SPY")
    for k in (63, 126, 252):
        assert f[f"trend_{k}"].iloc[280] == pytest.approx(np.log(c[280] / c[280 - k]))
        assert np.isnan(f[f"trend_{k}"].iloc[k - 1]) and not np.isnan(f[f"trend_{k}"].iloc[k])
    assert f["dist_ma_50"].iloc[280] == pytest.approx(c[280] / c[231:281].mean() - 1)
    assert f["dist_ma_200"].iloc[280] == pytest.approx(c[280] / c[81:281].mean() - 1)
    assert not any(x in f for x in ("mom_20", "mom_50", "mom_200", "dist_ma_20", "zscore_20", "new_high_20"))
    didx = pd.date_range("2020-01-01", periods=n)
    b = F._own_features(_frame(c, didx), "BTC-USD")
    for k in (7, 14, 21, 28):
        assert b[f"trend_{k}"].iloc[100] == pytest.approx(np.log(c[100] / c[100 - k]))
    assert "trend_63" not in b
    exp = [c[i] >= c[i - 19:i + 1].max() for i in range(19, n)]
    assert b["new_high_20"].iloc[19:].tolist() == [float(x) for x in exp]
    assert b["new_high_20"].iloc[:19].isna().all()
    assert s.iloc[0] == c[0]


def test_new_high_flag_explicit():
    c = np.r_[np.linspace(100, 119, 20), 110.0, 130.0, 125.0]
    b = F._own_features(_frame(c, pd.date_range("2020-01-01", periods=len(c))), "BTC-USD")
    assert b["new_high_20"].iloc[19] == 1.0      # 119 is the max of the first 20
    assert b["new_high_20"].iloc[20] == 0.0      # 110 < 119
    assert b["new_high_20"].iloc[21] == 1.0      # 130 new high
    assert b["new_high_20"].iloc[22] == 0.0


def test_rsi_and_volume_guard():
    c = np.arange(1, 31, dtype=float)
    idx = sessions("2022-01-03", 30)
    f = F._own_features(_frame(c, idx), "SPY")
    assert f["rsi_14"].iloc[-1] == 100.0 and np.isnan(f["rsi_14"].iloc[12])
    assert "zscore_20" not in f
    px = make_prices(10)
    px["SPY"]["Volume"] = 0.0
    g = F.build_features("SPY", CFG, px)["vol_chg_20"].dropna()
    assert len(g) > 0 and (g == 0.0).all()


def test_vix_termstructure_backwardation_and_vrp_hand():
    px = make_prices(6)
    f = F.build_features("SPY", CFG, px)
    t = px["SPY"].index[500]
    vix, v3 = px["^VIX"]["Close"][t], px["^VIX3M"]["Close"][t]
    assert f.loc[t, "vix_level"] == pytest.approx(vix)
    assert f.loc[t, "vix_chg"] == pytest.approx(vix / px["^VIX"]["Close"].shift(1)[t] - 1)
    assert f.loc[t, "vix_ts"] == pytest.approx(vix / v3)
    assert f.loc[t, "vix_backwardation"] == float(vix > v3)
    lr = np.log(px["SPY"]["Close"]).diff()
    rv = 252 / 21 * (lr.loc[:t].iloc[-21:] ** 2).sum()
    assert f.loc[t, "vrp"] == pytest.approx((vix / 100) ** 2 - rv)
    # NaN term structure and flag before VIX3M exists; vrp for QQQ uses VXN (NaN before it)
    early = px["SPY"].index[250]
    assert np.isnan(f.loc[early, "vix_ts"]) and np.isnan(f.loc[early, "vix_backwardation"])
    q = F.build_features("QQQ", CFG, px)
    assert np.isnan(q.loc[px["QQQ"].index[50], "vrp"])
    qv = px["^VXN"]["Close"][t]
    rvq = 252 / 21 * (np.log(px["QQQ"]["Close"]).diff().loc[:t].iloc[-21:] ** 2).sum()
    assert q.loc[t, "vrp"] == pytest.approx((qv / 100) ** 2 - rvq)
    assert "vrp" not in F.build_features("BTC-USD", CFG, px)


def test_backwardation_flag_both_signs():
    px = make_prices(6)
    t = px["SPY"].index[600]
    px["^VIX"].loc[t, "Close"] = 50.0
    px["^VIX3M"].loc[t, "Close"] = 30.0
    f = F.build_features("SPY", CFG, px)
    assert f.loc[t, "vix_backwardation"] == 1.0 and f.loc[t, "vix_ts"] == pytest.approx(50 / 30)
    px["^VIX"].loc[t, "Close"] = 15.0
    assert F.build_features("SPY", CFG, px).loc[t, "vix_backwardation"] == 0.0


def test_iwm_vrp_uses_rvx_when_available_else_vix():
    px = make_prices(7)
    t = px["IWM"].index[500]
    lr = np.log(px["IWM"]["Close"]).diff()
    rv = 252 / 21 * (lr.loc[:t].iloc[-21:] ** 2).sum()
    f = F.build_features("IWM", CFG, px)       # no ^RVX -> ^VIX fallback
    assert f.loc[t, "vrp"] == pytest.approx((px["^VIX"]["Close"][t] / 100) ** 2 - rv)
    rng = np.random.default_rng(1)
    px["^RVX"] = _ohlcv_from(_walk(px["IWM"].index, rng, 25.0, 0.02), px["IWM"].index, rng, vol=False)
    g = F.build_features("IWM", CFG, px)
    assert g.loc[t, "vrp"] == pytest.approx((px["^RVX"]["Close"][t] / 100) ** 2 - rv)


def test_optional_rvx_respects_status_file(tmp_path):
    cfg = dict(CFG, data_dir=tmp_path, extra_tickers=CFG["extra_tickers"] + [
        {"ticker": "^RVX", "start": None, "optional": True}])
    px = make_prices(8)
    for t, d in px.items():
        d.to_parquet(tmp_path / f"{t}.parquet")
    px["IWM"].to_parquet(tmp_path / "^RVX.parquet")      # a stale file that must be ignored
    (tmp_path / "extras_status.json").write_text(
        '{"loaded": [], "skipped": [{"ticker": "^RVX", "reason": "no data"}]}')
    assert "^RVX" not in F._load_all(cfg, None)
    (tmp_path / "extras_status.json").write_text('{"loaded": ["^RVX"], "skipped": []}')
    assert "^RVX" in F._load_all(cfg, None)


# ------------------------------------------------------------ calendar
def _mini_sessions_prices(n=300):
    return make_prices(0, n)


def test_dow_next_across_friday_and_holiday():
    idx = pd.DatetimeIndex(["2020-11-19", "2020-11-20", "2020-11-24", "2020-11-25",
                            "2021-04-01", "2021-12-30", "2022-01-14"])   # Thu Fri Tue Wed | Thu(GoodFri next) ...
    nxt = F.next_session_dates(idx)
    assert nxt[0] == pd.Timestamp("2020-11-20")
    assert nxt[1] == pd.Timestamp("2020-11-23") and nxt[1].day_of_week == 0     # Friday -> Monday
    assert nxt[3] == pd.Timestamp("2020-11-27") and nxt[3].day_of_week == 4     # Thanksgiving skipped
    assert nxt[4] == pd.Timestamp("2021-04-05")                                  # Good Friday skipped
    assert nxt[6] == pd.Timestamp("2022-01-18")                                  # MLK skipped
    px = make_prices(0)
    f = F.build_features("SPY", CFG, px)
    assert f.loc["2020-11-25", "dow_next"] == 4.0
    assert f.loc["2020-11-20", "dow_next"] == 0.0
    assert f.loc["2020-11-19", "dow_next"] == 4.0


def test_btc_weekend_flags():
    px = make_prices(0)
    f = F.build_features("BTC-USD", CFG, px)
    assert f.loc["2020-11-20", ["dow_next", "is_weekend_next"]].tolist() == [5.0, 1.0]   # Fri -> Sat
    assert f.loc["2020-11-21", ["dow_next", "is_weekend_next"]].tolist() == [6.0, 1.0]
    assert f.loc["2020-11-22", ["dow_next", "is_weekend_next"]].tolist() == [0.0, 0.0]   # Sun -> Mon
    assert f.loc["2020-11-18", ["dow_next", "is_weekend_next"]].tolist() == [3.0, 0.0]
    assert "pre_fomc" not in f and "is_weekend_next" not in F.build_features("SPY", CFG, px)


@pytest.mark.parametrize("t", ["SPY", "QQQ", "IWM"])
def test_pre_fomc_flag_t_plus_one_logic(t):
    px = make_prices(0)
    f = F.build_features(t, CFG, px)
    # scheduled FOMC announcement 2020-01-29 (Wed) and 2020-12-16 (Wed)
    assert f.loc["2020-01-28", "pre_fomc"] == 1.0
    assert f.loc["2020-01-29", "pre_fomc"] == 0.0
    assert f.loc["2020-01-27", "pre_fomc"] == 0.0
    assert f.loc["2020-12-15", "pre_fomc"] == 1.0
    # next session after Friday before a Monday FOMC would be flagged via custom dates
    custom = pd.DatetimeIndex(["2020-11-23", "2030-01-01"])
    g = F.build_features(t, CFG, px, fomc_dates=custom)
    assert g.loc["2020-11-20", "pre_fomc"] == 1.0 and g.loc["2020-11-19", "pre_fomc"] == 0.0


def test_calendar_last_row_beyond_data():
    px = make_prices(0)
    last = px["SPY"].index[-1]
    f = F.build_features("SPY", CFG, px)
    nxt = F.next_session_dates(pd.DatetimeIndex([last]))[0]
    assert f["dow_next"].iloc[-1] == nxt.day_of_week and not np.isnan(f["pre_fomc"].iloc[-1])
    # last row sits right before an FOMC day: data ends 2020-01-28
    short = truncate(px, pd.Timestamp("2020-01-28"))
    assert F.build_features("SPY", CFG, short)["pre_fomc"].iloc[-1] == 1.0
    d = F.build_dataset("SPY", CFG, short, ) if len(short["SPY"]) > 260 else None
    # FOMC date beyond last known FOMC date -> NaN (schedule coverage ended)
    g = F.build_features("SPY", CFG, px, fomc_dates=pd.DatetimeIndex(["2019-03-20"]))
    assert g["pre_fomc"].isna().sum() > 0 and g["pre_fomc"].iloc[-1] != g["pre_fomc"].iloc[-1]
    assert d is None or d["pre_fomc"].iloc[-1] == 1.0


# ------------------------------------------------------------ cross-asset
def test_btc_stale_flag_and_asof_values():
    px = make_prices(0)
    f = F.build_features("BTC-USD", CFG, px)
    spy_lr = np.log(px["SPY"]["Close"]).diff()
    # normal Wednesday: not stale, value from the same date
    assert f.loc["2020-11-18", "stale"] == 0.0
    assert f.loc["2020-11-18", "spy_ret"] == pytest.approx(spy_lr["2020-11-18"])
    # Fri not stale, Sat/Sun stale and carry Friday's values; Monday fresh
    assert f.loc["2020-11-20", "stale"] == 0.0
    for d in ("2020-11-21", "2020-11-22"):
        assert f.loc[d, "stale"] == 1.0
        assert f.loc[d, "spy_ret"] == pytest.approx(spy_lr["2020-11-20"])
        assert f.loc[d, "iwm_ret"] == pytest.approx(np.log(px["IWM"]["Close"]).diff()["2020-11-20"])
        assert f.loc[d, "vix_level"] == pytest.approx(px["^VIX"]["Close"]["2020-11-20"])
    assert f.loc["2020-11-23", "stale"] == 0.0
    # holiday (Thanksgiving 2020-11-26): stale, carries Wednesday
    assert f.loc["2020-11-26", "stale"] == 1.0
    assert f.loc["2020-11-26", "spy_ret"] == pytest.approx(spy_lr["2020-11-25"])
    assert f.loc["2020-11-27", "stale"] == 0.0
    # a BTC row after Good Friday 2021-04-02 and Easter Monday trading
    assert f.loc["2021-04-02", "stale"] == 1.0 and f.loc["2021-04-05", "stale"] == 0.0


def test_stock_assets_have_no_btc_cross_asset_or_xret():
    px = make_prices(0)
    for t in ("SPY", "QQQ"):
        f = F.build_features(t, CFG, px)
        assert not [c for c in f if c.startswith("xret_") or c in feature_cross_only()], t
        assert "cross_asset" not in F.feature_groups_for(f.columns)
    iwm = F.build_features("IWM", CFG, px)
    assert F.feature_groups_for(iwm.columns)["cross_asset"] == ["iwm_spy_spread_5"]
    assert not hasattr(F, "cross_na")
    import inspect
    assert "cross_na" not in inspect.signature(F.build_dataset).parameters


def feature_cross_only():
    return ["spy_ret", "qqq_ret", "iwm_ret", "stale", "btc_spy_corr_60", "iwm_spy_spread_5"]


def test_iwm_spy_spread_hand_and_uses_iwm_own_prices():
    px = make_prices(0)
    f = F.build_features("IWM", CFG, px)
    t = px["IWM"].index[400]
    i = px["IWM"]["Close"]
    s = px["SPY"]["Close"]
    exp = np.log(i[t] / i.shift(5)[t]) - np.log(s[t] / s.shift(5)[t])
    assert f.loc[t, "iwm_spy_spread_5"] == pytest.approx(exp)
    px2 = {k: v.copy() for k, v in px.items()}
    px2["^RUT"] = px2["IWM"] * 3          # an index frame must be ignored entirely
    assert_frame_equal(F.build_features("IWM", CFG, px2), f)


def test_btc_spy_corr_alignment_and_causality():
    px = make_prices(0, 700)
    f = F.build_features("BTC-USD", CFG, px)
    sess = px["SPY"].index
    t = sess[400]
    win = sess[341:401]       # 60 SPY returns ending at t (pairs p->s)
    sp = np.log(px["SPY"]["Close"]).diff()[win].to_numpy()
    bc = px["BTC-USD"]["Close"]
    bt = np.array([np.log(bc[s] / bc[sess[sess.get_loc(s) - 1]]) for s in win])
    assert f.loc[t, "btc_spy_corr_60"] == pytest.approx(np.corrcoef(sp, bt)[0, 1])
    # weekend after t carries the Friday value
    fri = sess[sess.day_of_week == 4][50]
    assert f.loc[fri + pd.Timedelta(days=2), "btc_spy_corr_60"] == pytest.approx(f.loc[fri, "btc_spy_corr_60"])
    # causal: changing a future BTC/SPY close leaves earlier correlations unchanged
    d = sess[500]
    for tk in ("BTC-USD", "SPY"):
        px2 = {k: v.copy() for k, v in px.items()}
        px2[tk].loc[d, "Close"] *= 3
        g = F.build_features("BTC-USD", CFG, px2)
        assert_frame_equal(g.loc[:d - pd.Timedelta(days=1)], f.loc[:d - pd.Timedelta(days=1)])
    assert f["btc_spy_corr_60"].iloc[:60].isna().all()


# ------------------------------------------------------------ groups
def test_feature_groups_complete_and_exclusive():
    allcols = [c for v in F.feature_groups.values() for c in v]
    assert len(allcols) == len(set(allcols))
    assert set(F.feature_groups) == {"returns", "reversal", "volatility", "vix_stress",
                                     "trend", "calendar", "cross_asset"}
    px = make_prices(0)
    for t in TICKERS:
        cols = F.build_features(t, CFG, px).columns
        grouped = F.feature_groups_for(cols)
        flat = [c for v in grouped.values() for c in v]
        assert sorted(flat) == sorted(cols)           # every column in exactly one group
        d = F.build_dataset(t, CFG, px)
        assert sorted(F.feature_columns(d)) == sorted(cols)


# ------------------------------------------------------------ dataset / target
def test_target_alignment_and_final_row():
    px = make_prices(8)
    for t in TICKERS:
        D = F.build_dataset(t, CFG, px)
        c = px[t]["Close"]
        for i in (0, 10, 100, len(D) - 2):
            pos = c.index.get_loc(D.index[i])
            exp = c.iloc[pos + 1] / c.iloc[pos] - 1
            assert D["next_return"].iloc[i] == pytest.approx(exp)
            assert D["target"].iloc[i] == (1.0 if exp > 0 else 0.0)
        assert D.index[-1] == c.index[-1]
        assert np.isnan(D["target"].iloc[-1]) and np.isnan(D["next_return"].iloc[-1])
        assert D["target"].iloc[:-1].isin([0.0, 1.0]).all()
    b = F.build_dataset("BTC-USD", CFG, px)
    assert (b.index.to_series().diff().dropna() == pd.Timedelta(days=1)).all()


def test_zero_return_maps_to_target_zero_flat_close():
    c = pd.Series([10.0, 10.0, 10.0, 11.0, 11.0, 10.0])
    nxt, tgt = F.make_target(c)
    assert nxt.iloc[:5].tolist() == [0.0, 0.0, pytest.approx(0.1), 0.0, pytest.approx(-1 / 11)]
    assert tgt.iloc[:5].tolist() == [0.0, 0.0, 1.0, 0.0, 0.0]
    assert np.isnan(tgt.iloc[-1])
    # in a full dataset: a plateau of equal closes inside a random walk
    rng = np.random.default_rng(3)
    sess = sessions("2019-01-02", 900)
    c = _walk(sess, rng)
    c[500:504] = c[500]
    px = make_prices(3)
    px["SPY"] = _ohlcv_from(c, sess, rng)
    D = F.build_dataset("SPY", CFG, px)
    for pos in (500, 501, 502):
        t = sess[pos]
        assert D.loc[t, "next_return"] == 0.0 and D.loc[t, "target"] == 0.0
    assert D.loc[sess[503], "next_return"] != 0.0


def test_dataset_nan_policy_and_core_columns():
    px = make_prices(12)
    for t in TICKERS:
        d = F.build_dataset(t, CFG, px)
        feats = F.feature_columns(d)
        core = F.core_columns(feats)
        assert set(F.CORE_EXCLUDED) & set(feats) == set(F.CORE_EXCLUDED) & set(feats)
        assert d[core].iloc[:-1].notna().all().all()
        assert d[core].iloc[-1].notna().all()
        assert d.index[-1] == px[t].index[-1]
    s = F.build_dataset("SPY", CFG, px)
    assert s["vix_ts"].isna().any() and s["vix_backwardation"].isna().any()   # VIX3M starts late, rows survive
    assert (s["vix_ts"].isna() == s["vix_backwardation"].isna()).all()
    q = F.build_dataset("QQQ", CFG, px)
    assert q["vrp"].isna().any()


def test_dataset_row_survives_when_only_late_source_missing():
    px = make_prices(12)
    full = F.build_dataset("SPY", CFG, px)
    px2 = dict(px)
    px2["^VIX3M"] = px["^VIX3M"].iloc[-50:]
    short = F.build_dataset("SPY", CFG, px2)
    assert len(short) == len(full)       # dropping ^VIX3M history drops no rows


def test_save_load_roundtrip(tmp_path):
    cfg = dict(CFG, data_dir=tmp_path)
    px = make_prices(13)
    d = F.build_dataset("QQQ", cfg, px)
    p = F.save_dataset("QQQ", d, cfg)
    assert p == tmp_path / "features" / "QQQ.parquet"
    assert_frame_equal(F.load_dataset("QQQ", cfg), d, check_freq=False)


def test_real_data_truncation_if_present():
    cfg = load_config()
    names = [a["ticker"] for a in cfg["assets"]] + F.required_extra_tickers(cfg)
    if not all((cfg["data_dir"] / f"{t}.parquet").exists() for t in names):
        pytest.skip("real data missing")
    px = F._load_all(cfg, None)
    for t in TICKERS:
        full = F.build_features(t, cfg, px)
        cut = full.index[len(full) // 2 + 123]
        part = F.build_features(t, cfg, truncate(px, cut))
        assert_frame_equal(full.loc[:part.index[-1]], part)


# ------------------------------------------------------------ errors
def test_error_missing_ticker_in_prices():
    for missing, pat in (("^VIX", r"\^VIX.*python -m quant.data"), ("^VIX3M", r"\^VIX3M"),
                         ("^VXN", r"\^VXN"), ("QQQ", "QQQ"), ("IWM", "IWM")):
        px = make_prices(20)
        del px[missing]
        with pytest.raises(ValueError, match=pat):
            F.build_features("SPY", CFG, px)
    with pytest.raises(KeyError):
        F.build_features("XYZ", CFG, make_prices(20))


def test_error_empty_or_short_frame():
    px = make_prices(21)
    px["SPY"] = px["SPY"].iloc[:0]
    with pytest.raises(ValueError, match="empty"):
        F.build_features("SPY", CFG, px)
    px = make_prices(21)
    px["SPY"] = px["SPY"].iloc[: F.MIN_ROWS - 1]
    with pytest.raises(ValueError, match="only .* rows"):
        F.build_features("SPY", CFG, px)
    px["SPY"] = make_prices(21)["SPY"].iloc[: F.MIN_ROWS]
    assert len(F.build_features("SPY", CFG, px)) == F.MIN_ROWS


def test_error_dataset_fewer_than_two_rows():
    px = make_prices(22)
    px["SPY"] = px["SPY"].iloc[:F.MIN_ROWS]
    with pytest.raises(ValueError, match="only"):
        F.build_dataset("SPY", CFG, px)


def test_constant_prices_raise_clear_error():
    px = make_prices(23)
    px["SPY"].iloc[:, :4] = 100.0
    with pytest.raises(ValueError, match="dataset would have only"):
        F.build_dataset("SPY", CFG, px)


def test_error_missing_parquet(tmp_path):
    cfg = dict(CFG, data_dir=tmp_path)
    with pytest.raises(ValueError, match="python -m quant.data"):
        F.build_features("SPY", cfg)


def test_error_missing_fomc_file(tmp_path):
    cfg = dict(CFG, fomc_path=tmp_path / "nope.csv")
    with pytest.raises(ValueError, match="FOMC"):
        F.build_features("SPY", cfg, make_prices(24))
    # BTC does not need the FOMC file
    assert len(F.build_features("BTC-USD", cfg, make_prices(24))) > 0


def test_cli_clean_error_exit_1(tmp_path, capsys):
    cfgp = tmp_path / "c.yaml"
    cfgp.write_text("assets:\n  - {name: S, ticker: SPY, start: '2000-01-01', trades_weekends: false}\n"
                    "extra_tickers: []\ndata_dir: " + str(tmp_path) + "\ncosts: {SPY: 0.0005}\n"
                    "threshold: 0.55\nstarting_capital: 1\nholdout_months: 12\n")
    with pytest.raises(SystemExit) as e:
        F.main(["--config", str(cfgp)])
    assert e.value.code == 1
    assert "ERROR" in capsys.readouterr().err
