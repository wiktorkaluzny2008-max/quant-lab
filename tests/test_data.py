import pandas as pd
import pytest

from quant import data
from quant.data import DataDownloadError, DataValidationError


def make(dates, price=100.0):
    idx = pd.DatetimeIndex(dates)
    n = len(idx)
    return pd.DataFrame({"Open": price, "High": price + 1, "Low": price - 1,
                         "Close": [price + i for i in range(n)],
                         "Volume": 1000.0}, index=idx)


def bdays(a, b):
    return pd.bdate_range(a, b)


@pytest.fixture
def calls(monkeypatch):
    log = []
    store = {"frame": None, "error": None}

    def fake(ticker, start, end=None):
        log.append((ticker, start))
        if store["error"]:
            raise store["error"]
        f = store["frame"]
        naive = f.index.tz_localize(None) if f.index.tz is not None else f.index
        return f[naive >= pd.Timestamp(start)]
    monkeypatch.setattr(data, "_download", fake)
    log_obj = type("C", (), {})()
    log_obj.calls, log_obj.store = log, store
    return log_obj


def test_full_download(calls, tmp_path):
    calls.store["frame"] = make(bdays("2020-01-01", "2020-03-31"))
    df = data.update_ticker("SPY", "2020-01-01", tmp_path)
    assert (tmp_path / "SPY.parquet").exists()
    assert list(df.columns) == data.COLUMNS
    assert df.index.tz is None
    assert len(pd.read_parquet(tmp_path / "SPY.parquet")) == len(df)
    assert not list(tmp_path.glob("*.tmp"))


def test_multiindex_and_tz_flattened(calls, tmp_path):
    f = make(bdays("2020-01-01", "2020-02-28"))
    f.index = f.index.tz_localize("America/New_York")
    f.columns = pd.MultiIndex.from_product([f.columns, ["SPY"]])
    calls.store["frame"] = f
    df = data.update_ticker("SPY", "2020-01-01", tmp_path)
    assert df.index.tz is None and list(df.columns) == data.COLUMNS


def test_incremental_fetches_only_missing(calls, tmp_path):
    calls.store["frame"] = make(bdays("2020-01-01", "2020-03-31"))
    data.update_ticker("SPY", "2020-01-01", tmp_path)
    calls.store["frame"] = make(bdays("2020-01-01", "2020-04-30"))
    df = data.update_ticker("SPY", "2020-01-01", tmp_path)
    start = pd.Timestamp(calls.calls[-1][1])
    assert start > pd.Timestamp("2020-03-15")
    assert start <= pd.Timestamp("2020-03-31")
    assert df.index.max() == pd.Timestamp("2020-04-30")
    assert not df.index.has_duplicates and df.index.is_monotonic_increasing


def test_idempotent_rerun(calls, tmp_path):
    calls.store["frame"] = make(bdays("2020-01-01", "2020-03-31"))
    a = data.update_ticker("SPY", "2020-01-01", tmp_path)
    b = data.update_ticker("SPY", "2020-01-01", tmp_path)
    pd.testing.assert_frame_equal(a, b)
    pd.testing.assert_frame_equal(pd.read_parquet(tmp_path / "SPY.parquet"), b)


def test_revisions_overwrite_recent_rows(calls, tmp_path):
    f = make(bdays("2020-01-01", "2020-03-31"))
    calls.store["frame"] = f
    data.update_ticker("SPY", "2020-01-01", tmp_path)
    g = f.copy()
    g.iloc[-1, g.columns.get_loc("Close")] = 999.0
    calls.store["frame"] = g
    df = data.update_ticker("SPY", "2020-01-01", tmp_path)
    assert df["Close"].iloc[-1] == 999.0


def test_failure_keeps_old_cache(calls, tmp_path):
    calls.store["frame"] = make(bdays("2020-01-01", "2020-03-31"))
    data.update_ticker("SPY", "2020-01-01", tmp_path)
    before = (tmp_path / "SPY.parquet").read_bytes()
    calls.store["error"] = RuntimeError("network down")
    with pytest.raises(DataDownloadError):
        data.update_ticker("SPY", "2020-01-01", tmp_path)
    calls.store["error"] = None
    calls.store["frame"] = make(bdays("2020-01-01", "2020-03-31")).iloc[0:0]
    with pytest.raises(DataDownloadError):
        data.update_ticker("SPY", "2020-01-01", tmp_path)
    assert (tmp_path / "SPY.parquet").read_bytes() == before
    assert not list(tmp_path.glob("*.tmp"))


def test_validation_failure_keeps_cache_untouched(calls, tmp_path):
    calls.store["frame"] = make(bdays("2020-01-01", "2020-03-31"))
    data.update_ticker("SPY", "2020-01-01", tmp_path)
    before = (tmp_path / "SPY.parquet").read_bytes()
    bad = make(bdays("2020-01-01", "2020-04-30"))
    bad.iloc[-1, bad.columns.get_loc("Close")] = 0.0
    calls.store["frame"] = bad
    with pytest.raises(DataValidationError):
        data.update_ticker("SPY", "2020-01-01", tmp_path)
    assert (tmp_path / "SPY.parquet").read_bytes() == before


def test_validate_duplicates():
    df = make(bdays("2020-01-01", "2020-01-10"))
    df = pd.concat([df, df.iloc[[3]]])
    with pytest.raises(DataValidationError, match="duplicate"):
        data.validate(df, "X")


def test_validate_unsorted():
    df = make(bdays("2020-01-01", "2020-01-10")).iloc[::-1]
    with pytest.raises(DataValidationError, match="sorted"):
        data.validate(df, "X")


@pytest.mark.parametrize("bad", [0.0, -5.0, float("nan")])
def test_validate_bad_prices(bad):
    df = make(bdays("2020-01-01", "2020-01-10"))
    df.iloc[4, df.columns.get_loc("Close")] = bad
    with pytest.raises(DataValidationError):
        data.validate(df, "X")


def test_validate_stock_gap():
    dates = list(bdays("2020-01-02", "2020-01-10")) + list(bdays("2020-01-27", "2020-02-07"))
    with pytest.raises(DataValidationError, match="missing trading session"):
        data.validate(make(dates), "SPY")


@pytest.mark.parametrize("missing", ["2020-01-08", "2020-01-17"])  # a Wednesday, a Friday
def test_single_missing_session_flagged(missing):
    dates = [d for d in bdays("2020-01-02", "2020-01-31")
             if d not in (pd.Timestamp(missing), pd.Timestamp("2020-01-20"))]  # 20th = MLK
    with pytest.raises(DataValidationError, match=missing):
        data.validate(make(dates), "SPY")


def test_stock_real_holidays_ok():
    # Thanksgiving 2020-11-26 and Good Friday 2021-04-02 are legitimately absent
    cal = data._xnys()
    for a, b, hol in [("2020-11-16", "2020-11-30", "2020-11-26"),
                      ("2021-03-29", "2021-04-09", "2021-04-02")]:
        days = bdays(a, b)
        assert pd.Timestamp(hol) in days
        data.validate(make([d for d in days if d != pd.Timestamp(hol)]), "SPY")
        assert pd.Timestamp(hol) not in cal.sessions_in_range(a, b)


def test_stock_2001_closure_and_old_start_ok():
    # 9/11-9/14/2001 are calendar closures; frame built from true sessions validates
    sessions = data._xnys().sessions_in_range("1999-01-04", "2001-10-05").tz_localize(None)
    assert pd.Timestamp("2001-09-12") not in sessions
    data.validate(make(sessions), "SPY")


def test_btc_partial_today_bar_dropped(calls, tmp_path):
    today = pd.Timestamp(pd.Timestamp.now("UTC").date())
    calls.store["frame"] = make(pd.date_range(today - pd.Timedelta(days=10), today))
    df = data.update_ticker("BTC-USD", str(today.date() - pd.Timedelta(days=10)),
                            tmp_path, trades_weekends=True)
    assert df.index.max() == today - pd.Timedelta(days=1)
    assert pd.read_parquet(tmp_path / "BTC-USD.parquet").index.max() == today - pd.Timedelta(days=1)


def test_btc_weekend_gap_flagged_but_stock_not():
    weekdays = bdays("2020-01-01", "2020-02-28")
    with pytest.raises(DataValidationError, match="gap"):
        data.validate(make(weekdays), "BTC-USD", trades_weekends=True)
    data.validate(make(weekdays), "SPY", trades_weekends=False)
    data.validate(make(pd.date_range("2020-01-01", "2020-02-28")), "BTC-USD", True)


def test_update_all(calls, tmp_path):
    calls.store["frame"] = make(pd.date_range("2020-01-01", "2020-02-28"))
    cfg = {"data_dir": tmp_path,
           "assets": [{"name": "B", "ticker": "BTC-USD", "start": "2020-01-01",
                       "trades_weekends": True}],
           "extra_tickers": []}
    out = data.update_all(cfg)
    assert set(out) == {"BTC-USD"} and (tmp_path / "BTC-USD.parquet").exists()


# --- staleness and incomplete bars (deterministic `now`) ---
# Wed 2024-01-10: session closes 16:00 ET = 21:00 UTC (EST)
BEFORE_CLOSE = pd.Timestamp("2024-01-10 20:59", tz="UTC")
AFTER_CLOSE = pd.Timestamp("2024-01-10 21:01", tz="UTC")


def test_last_completed_session():
    assert data.last_completed_session(BEFORE_CLOSE) == pd.Timestamp("2024-01-09")
    assert data.last_completed_session(AFTER_CLOSE) == pd.Timestamp("2024-01-10")
    # Saturday -> Friday
    assert data.last_completed_session(pd.Timestamp("2024-01-13 12:00", tz="UTC")) \
        == pd.Timestamp("2024-01-12")


def test_stale_stock_warns(caplog):
    df = make(bdays("2023-12-01", "2024-01-08"))
    with caplog.at_level("WARNING"):
        msg = data.check_staleness(df, "SPY", False, now=AFTER_CLOSE)
    assert msg and "SPY" in msg and "2024-01-10" in msg and "2024-01-08" in msg
    assert any("STALE" in r.message for r in caplog.records)


def test_stock_fresh_before_and_after_close():
    df = make(bdays("2023-12-01", "2024-01-09"))
    assert data.check_staleness(df, "SPY", False, now=BEFORE_CLOSE) is None
    assert data.check_staleness(df, "SPY", False, now=AFTER_CLOSE) is not None


def test_stock_weekend_not_stale():
    df = make(bdays("2024-01-01", "2024-01-12"))
    assert data.check_staleness(df, "SPY", False,
                                now=pd.Timestamp("2024-01-14 12:00", tz="UTC")) is None


def test_btc_staleness():
    now = pd.Timestamp("2024-01-10 08:00", tz="UTC")
    ok = make(pd.date_range("2023-12-01", "2024-01-09"))
    stale = make(pd.date_range("2023-12-01", "2024-01-08"))
    assert data.check_staleness(ok, "BTC-USD", True, now=now) is None
    assert "BTC-USD" in data.check_staleness(stale, "BTC-USD", True, now=now)


def test_stock_unclosed_session_bar_dropped(calls, tmp_path):
    calls.store["frame"] = make(bdays("2023-12-01", "2024-01-10"))
    df = data.update_ticker("SPY", "2023-12-01", tmp_path, now=BEFORE_CLOSE)
    assert df.index.max() == pd.Timestamp("2024-01-09")
    assert pd.read_parquet(tmp_path / "SPY.parquet").index.max() == pd.Timestamp("2024-01-09")


def test_stock_closed_session_bar_kept(calls, tmp_path):
    calls.store["frame"] = make(bdays("2023-12-01", "2024-01-10"))
    df = data.update_ticker("SPY", "2023-12-01", tmp_path, now=AFTER_CLOSE)
    assert df.index.max() == pd.Timestamp("2024-01-10")


def test_vix_style_stock_rule_applies_to_extra_ticker(calls, tmp_path):
    calls.store["frame"] = make(bdays("2023-12-01", "2024-01-10"))
    cfg = {"data_dir": tmp_path, "assets": [{"name": "S", "ticker": "SPY",
           "start": "2023-12-01", "trades_weekends": False}], "extra_tickers": ["^VIX"]}
    warns = []
    out = data.update_all(cfg, now=BEFORE_CLOSE, warnings=warns)
    assert out["^VIX"].index.max() == pd.Timestamp("2024-01-09")
    assert warns == []


def test_update_all_collects_stale_warnings(calls, tmp_path):
    calls.store["frame"] = make(bdays("2023-12-01", "2024-01-05"))
    cfg = {"data_dir": tmp_path, "assets": [{"name": "S", "ticker": "SPY",
           "start": "2023-12-01", "trades_weekends": False}], "extra_tickers": []}
    warns = []
    data.update_all(cfg, now=AFTER_CLOSE, warnings=warns)
    assert len(warns) == 1 and "SPY" in warns[0]


def test_btc_drop_today_with_injected_now(calls, tmp_path):
    now = pd.Timestamp("2024-01-10 08:00", tz="UTC")
    calls.store["frame"] = make(pd.date_range("2023-12-20", "2024-01-10"))
    df = data.update_ticker("BTC-USD", "2023-12-20", tmp_path, True, now=now)
    assert df.index.max() == pd.Timestamp("2024-01-09")


# --- extras (vol indices), optional skip, FOMC dates ---
def _cfg(tmp_path, extras):
    return {"data_dir": tmp_path, "assets": [{"name": "S", "ticker": "SPY",
            "start": "2023-12-01", "trades_weekends": False}], "extra_tickers": extras}


def test_extra_validation_hard_failures(calls, tmp_path):
    f = make(bdays("2023-12-01", "2024-01-05"))
    f.iloc[3, 3] = float("nan")
    with pytest.raises(DataValidationError):
        data.validate(f, "^VXN", extra=True)
    g = make(bdays("2023-12-01", "2024-01-05"))
    g.iloc[3, 0] = 0
    with pytest.raises(DataValidationError):
        data.validate(g, "^VXN", extra=True)
    dup = pd.concat([f.dropna(), f.dropna().iloc[:1]])
    with pytest.raises(DataValidationError, match="duplicate"):
        data.validate(dup, "^VXN", extra=True)
    uns = make(bdays("2023-12-01", "2024-01-05")).iloc[::-1]
    with pytest.raises(DataValidationError, match="sorted"):
        data.validate(uns, "^VXN", extra=True)


def test_extra_session_gap_is_soft_but_asset_gap_is_hard(caplog):
    d = bdays("2023-12-01", "2024-01-05").delete(5)
    df = make(d)
    data.validate(df, "^VXN", extra=True)          # warns only
    assert "missing trading session" in caplog.text
    with pytest.raises(DataValidationError):
        data.validate(df, "IWM")


def test_extra_zero_volume_and_holiday_row_ok(caplog):
    d = bdays("2023-12-01", "2024-01-05")      # includes Mon 2023-12-25 (NYSE holiday)
    df = make(d)
    df["Volume"] = 0.0
    data.validate(df, "^VIX", extra=True)
    assert "non-session day 2023-12-25" in caplog.text


def test_update_all_extras_with_per_ticker_start(calls, tmp_path):
    calls.store["frame"] = make(bdays("2023-01-02", "2024-01-09"))
    cfg = _cfg(tmp_path, [{"ticker": "^VIX", "start": None, "optional": False},
                          {"ticker": "^VIX3M", "start": "2023-12-15", "optional": False}])
    out = data.update_all(cfg, now=BEFORE_CLOSE)
    assert {"SPY", "^VIX", "^VIX3M"} == set(out)
    assert ("^VIX3M", "2023-12-15") in calls.calls
    assert out["^VIX3M"].index.min() == pd.Timestamp("2023-12-15")
    assert ("^VIX", "2023-12-01") in calls.calls     # default = earliest asset start


def test_optional_extra_skipped_and_recorded(calls, tmp_path, monkeypatch, caplog):
    orig = data._download

    def fake(ticker, start, end=None):
        if ticker == "^RVX":
            raise DataDownloadError("^RVX: yfinance returned no data")
        return orig(ticker, start, end)
    calls.store["frame"] = make(bdays("2023-12-01", "2024-01-09"))
    monkeypatch.setattr(data, "_download", lambda t, s, end=None: (
        (_ for _ in ()).throw(DataDownloadError("^RVX: yfinance returned no data"))
        if t == "^RVX" else calls_fake(t, s)))
    fr = calls.store["frame"]

    def calls_fake(t, s):
        return fr[fr.index >= pd.Timestamp(s)]
    cfg = _cfg(tmp_path, [{"ticker": "^VIX", "start": None, "optional": False},
                          {"ticker": "^RVX", "start": None, "optional": True}])
    warns = []
    out = data.update_all(cfg, now=BEFORE_CLOSE, warnings=warns)
    assert "^RVX" not in out and "^VIX" in out and "SPY" in out
    assert any("^RVX" in w and "skipped" in w for w in warns)
    assert "OPTIONAL extra skipped" in caplog.text
    import json
    st = json.loads((tmp_path / "extras_status.json").read_text())
    assert st["skipped"][0]["ticker"] == "^RVX" and "^VIX" in st["loaded"]


def test_non_optional_extra_failure_is_fatal(calls, tmp_path, monkeypatch):
    fr = make(bdays("2023-12-01", "2024-01-09"))

    def fake(t, s, end=None):
        if t == "^VXN":
            raise DataDownloadError("boom")
        return fr[fr.index >= pd.Timestamp(s)]
    monkeypatch.setattr(data, "_download", fake)
    with pytest.raises(DataDownloadError):
        data.update_all(_cfg(tmp_path, [{"ticker": "^VXN", "start": None, "optional": False}]),
                        now=BEFORE_CLOSE)


def test_iwm_is_strict_like_other_etfs(calls, tmp_path):
    calls.store["frame"] = make(bdays("2023-12-01", "2024-01-09").delete(7))
    with pytest.raises(DataValidationError, match="missing trading session"):
        data.update_ticker("IWM", "2023-12-01", tmp_path, now=BEFORE_CLOSE)


def test_fomc_dates_real_file():
    d = data.load_fomc_dates()
    assert d.is_monotonic_increasing and not d.has_duplicates
    assert (d.day_of_week < 5).all()
    assert d.min().year == 1999 and d.max().year == 2027
    for unscheduled in ["2020-03-15", "2001-09-17", "2025-08-22", "2019-10-04", "2008-01-22"]:
        assert pd.Timestamp(unscheduled) not in d
    for scheduled in ["2008-12-16", "2022-06-15", "2024-09-18", "2020-04-29"]:
        assert pd.Timestamp(scheduled) in d
    assert d.year.value_counts().loc[2020] == 7      # March 2020 meeting was cancelled


@pytest.mark.parametrize("rows,msg", [
    (["2000-01-02"], "weekend"),
    (["2000-02-02", "2000-02-02"], "duplicate"),
    (["2000-03-01", "2000-02-02"], "sorted"),
])
def test_fomc_validation_failures(tmp_path, rows, msg):
    p = tmp_path / "f.csv"
    p.write_text("date,notes\n" + "\n".join(f"{r}," for r in rows) + "\n")
    with pytest.raises(DataValidationError, match=msg):
        data.load_fomc_dates(p)


def test_fomc_unparseable_and_coverage(tmp_path):
    p = tmp_path / "f.csv"
    p.write_text("date,notes\nnot-a-date,\n")
    with pytest.raises(DataValidationError, match="unparseable"):
        data.load_fomc_dates(p)
    p.write_text("date,notes\n2000-02-02,\n")
    with pytest.raises(DataValidationError, match="expected 6-9"):
        data.load_fomc_dates(p)
