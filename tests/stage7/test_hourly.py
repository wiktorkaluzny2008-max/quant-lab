"""Offline tests for Stage 7c (hourly data + daily features). The ccxt call is mocked."""
import logging
import math

import numpy as np
import pandas as pd
import pytest

from quant.data import DataValidationError
from quant.stage7 import hourly as H

NOW = pd.Timestamp("2024-03-10 12:30", tz="UTC")


def make_bars(start, n, seed=0, price=100.0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=n, freq="h").as_unit("ns")
    close = price * np.exp(np.cumsum(rng.normal(0, 0.003, n)))
    open_ = np.concatenate([[price], close[:-1]])
    hi = np.maximum(open_, close) * (1 + rng.uniform(0, 0.002, n))
    lo = np.minimum(open_, close) * (1 - rng.uniform(0, 0.002, n))
    df = pd.DataFrame({"Open": open_, "High": hi, "Low": lo, "Close": close,
                       "Volume": rng.uniform(1, 10, n)}, index=idx)
    df.index.name = "Datetime"
    return df


def rows_of(df):
    return [[int(t.value // 10**6), *map(float, r)] for t, r in zip(df.index, df.to_numpy())]


@pytest.fixture
def cfg(tmp_path):
    return {"data_dir": tmp_path, "holdout_cutoff_date": "2024-11-06",
            "hourly": {"start": "2024-01-01", "exchange": "binance", "fallbacks": ["kraken", "coinbase"],
                       "quote": {"binance": "USDT", "kraken": "USD", "coinbase": "USD"},
                       "assets": {"BTC-USD": "BTC"}, "page_limit": 100, "stale_tolerance_hours": 2}}


class FakeNet:
    """Mocks H._fetch_page / H._rate_limit_ms from a dict {exchange: frame | Exception}."""
    def __init__(self, monkeypatch, data, page_cap=None):
        self.data, self.calls, self.page_cap = data, [], page_cap
        monkeypatch.setattr(H, "_fetch_page", self.fetch)
        monkeypatch.setattr(H, "_rate_limit_ms", lambda ex: 50.0)

    def fetch(self, ex, symbol, since_ms, limit):
        self.calls.append((ex, symbol, since_ms, limit))
        d = self.data[ex]
        if isinstance(d, Exception):
            raise d
        cap = min(limit, self.page_cap or limit)
        sub = d[d.index >= pd.Timestamp(since_ms, unit="ms")].iloc[:cap]
        return rows_of(sub)


def run_update(cfg, **kw):
    sleeps = []
    df = H.update_asset("BTC-USD", cfg, now=NOW, sleep=sleeps.append, **kw)
    return df, sleeps


# ------------------------------------------------------------------ download / storage
def test_paging_dedupe_and_storage(cfg, monkeypatch):
    src = make_bars("2024-01-01", 450)
    net = FakeNet(monkeypatch, {"binance": src}, page_cap=100)
    df, sleeps = run_update(cfg)
    assert len(net.calls) >= 5 and all(c[1] == "BTC/USDT" and c[3] == 100 for c in net.calls)
    assert sleeps and all(s == pytest.approx(0.05) for s in sleeps)       # rate limit respected
    assert df.index.is_unique and df.index.name == "Datetime" and df.index.tz is None
    assert list(df.columns) == ["Open", "High", "Low", "Close", "Volume"]
    assert all(df[c].dtype == "float64" for c in df.columns)
    pd.testing.assert_frame_equal(df, src, check_freq=False)
    stored = pd.read_parquet(cfg["data_dir"] / "hourly" / "BTC-USD.parquet")
    pd.testing.assert_frame_equal(stored, src, check_freq=False)
    st = H.read_status(cfg)["assets"]["BTC-USD"]
    assert st["exchange"] == "binance" and st["quote"] == "USDT" and st["bars"] == 450
    assert st["first_bar"] == "2024-01-01 00:00:00" and st["gaps"] == []


def test_incremental_update_dedupes_overlap(cfg, monkeypatch):
    src = make_bars("2024-01-01", 450)
    net = FakeNet(monkeypatch, {"binance": src.iloc[:300]})
    run_update(cfg)
    net.data["binance"] = src
    net.calls.clear()
    df, _ = run_update(cfg)
    assert net.calls[0][2] == int((src.index[299] - H.REFETCH_HOURS * H.HOUR).value // 10**6)
    assert df.index.is_unique and len(df) == 450
    pd.testing.assert_frame_equal(df, src, check_freq=False)


def test_incomplete_last_bar_dropped(cfg, monkeypatch):
    # NOW is 12:30 -> the 12:00 bar is still open; the 11:00 bar closed at 12:00
    src = make_bars("2024-03-09", 40)      # up to 2024-03-10 15:00
    FakeNet(monkeypatch, {"binance": src})
    df, _ = run_update(cfg)
    assert df.index[-1] == pd.Timestamp("2024-03-10 11:00")
    assert (df.index + H.HOUR <= NOW.tz_localize(None)).all()
    bar_at_close = H.drop_incomplete(src, "X", now=pd.Timestamp("2024-03-10 12:00", tz="UTC"))
    assert bar_at_close.index[-1] == pd.Timestamp("2024-03-10 11:00")   # t+1h <= now is complete


def test_fallback_used_when_primary_raises(cfg, monkeypatch, caplog):
    src = make_bars("2024-01-01", 200)
    net = FakeNet(monkeypatch, {"binance": RuntimeError("451 blocked"), "kraken": src})
    with caplog.at_level(logging.WARNING):
        df, _ = run_update(cfg)
    assert len(df) == 200
    assert net.calls[0][0] == "binance"
    assert any(c[0] == "kraken" and c[1] == "BTC/USD" for c in net.calls)
    st = H.read_status(cfg)["assets"]["BTC-USD"]
    assert st["exchange"] == "kraken" and st["quote"] == "USD"
    assert "fallback kraken" in caplog.text


def test_fallback_with_short_history_warns_and_records_actual_start(cfg, monkeypatch, caplog):
    src = make_bars("2024-02-20", 200)
    FakeNet(monkeypatch, {"binance": RuntimeError("down"), "kraken": src})
    with caplog.at_level(logging.WARNING):
        run_update(cfg)
    assert "later than the configured start" in caplog.text
    assert H.read_status(cfg)["assets"]["BTC-USD"]["first_bar"] == "2024-02-20 00:00:00"


def test_all_exchanges_failing_names_exchange_and_symbol(cfg, monkeypatch):
    FakeNet(monkeypatch, {e: RuntimeError("boom") for e in ("binance", "kraken", "coinbase")})
    with pytest.raises(H.HourlyDownloadError) as e:
        run_update(cfg)
    msg = str(e.value)
    assert "binance BTC/USDT" in msg and "kraken BTC/USD" in msg and "coinbase BTC/USD" in msg


def test_exchange_mixing_refused(cfg, monkeypatch):
    src = make_bars("2024-01-01", 200)
    net = FakeNet(monkeypatch, {"binance": src, "kraken": src})
    run_update(cfg)
    before = pd.read_parquet(cfg["data_dir"] / "hourly" / "BTC-USD.parquet")
    net.data["binance"] = RuntimeError("blocked")
    with pytest.raises(H.HourlyExchangeMixError, match="refusing to mix"):
        run_update(cfg)
    pd.testing.assert_frame_equal(pd.read_parquet(cfg["data_dir"] / "hourly" / "BTC-USD.parquet"), before)
    assert not any(c[0] == "kraken" for c in net.calls)         # never even fetched from the other one


# ------------------------------------------------------------------ validation
def bad(df, fn):
    d = df.copy()
    fn(d)
    return d


@pytest.mark.parametrize("name,fn,msg", [
    ("nan", lambda d: d.__setitem__(("Close"), d["Close"].mask(d.index == d.index[5])), "NaN"),
    ("zero", lambda d: d.__setitem__("Low", d["Low"].mask(d.index == d.index[5], 0.0)), "zero/negative"),
    ("high", lambda d: d.__setitem__("High", d["High"].mask(d.index == d.index[5], 1.0)), "High below"),
    ("low", lambda d: d.__setitem__("Low", d["Low"].mask(d.index == d.index[5], 1e6)), "Low above"),
    ("vol", lambda d: d.__setitem__("Volume", d["Volume"].mask(d.index == d.index[5], -1.0)), "negative volume"),
])
def test_validation_failures(name, fn, msg):
    with pytest.raises(DataValidationError, match=msg):
        H.validate_hourly(bad(make_bars("2024-01-01", 50), fn), "T")


def test_validation_duplicates_and_unsorted():
    df = make_bars("2024-01-01", 50)
    with pytest.raises(DataValidationError, match="duplicate"):
        H.validate_hourly(pd.concat([df, df.iloc[[3]]]).sort_index(), "T")
    with pytest.raises(DataValidationError, match="not sorted"):
        H.validate_hourly(df.iloc[::-1], "T")
    with pytest.raises(DataValidationError, match="no rows"):
        H.validate_hourly(df.iloc[:0], "T")


def test_gap_up_to_limit_warns_longer_errors(caplog):
    df = make_bars("2024-01-01", 200)
    ok = df.drop(df.index[50:50 + H.MAX_GAP_WARN_HOURS])           # exactly the limit missing
    with caplog.at_level(logging.WARNING):
        gaps = H.validate_hourly(ok, "T")
    assert gaps == [{"after": str(df.index[49]), "before": str(df.index[50 + H.MAX_GAP_WARN_HOURS]),
                     "missing_hours": H.MAX_GAP_WARN_HOURS}]
    assert "hourly gap" in caplog.text
    too_long = df.drop(df.index[50:50 + H.MAX_GAP_WARN_HOURS + 1])
    with pytest.raises(DataValidationError, match=f"gap of {H.MAX_GAP_WARN_HOURS + 1} missing hours"):
        H.validate_hourly(too_long, "T")


def test_gap_recorded_in_status_never_filled(cfg, monkeypatch):
    src = make_bars("2024-01-01", 300)
    src = src.drop(src.index[100:105])
    FakeNet(monkeypatch, {"binance": src})
    df, _ = run_update(cfg)
    assert len(df) == 295
    assert H.read_status(cfg)["assets"]["BTC-USD"]["gaps"][0]["missing_hours"] == 5


def test_staleness(caplog):
    df = make_bars("2024-03-09", 36)                  # last bar opens 2024-03-10 11:00
    assert H.check_staleness_hourly(df, "T", 2, NOW) is None
    old = df.iloc[:-4]                                # last bar 07:00; expected 11:00; 4h > 2h
    with caplog.at_level(logging.WARNING):
        msg = H.check_staleness_hourly(old, "T", 2, NOW)
    assert "STALE HOURLY" in msg and "STALE HOURLY" in caplog.text
    assert H.check_staleness_hourly(df.iloc[:-2], "T", 2, NOW) is None     # within tolerance


def test_load_hourly_default_is_dev_only(cfg):
    df = make_bars("2024-11-06 20:00", 12)            # 20:00 on the cutoff day .. 07:00 after it
    H._write_status(cfg, {"assets": {}})
    df.to_parquet(cfg["data_dir"] / "hourly" / "BTC-USD.parquet")
    dev = H.load_hourly("BTC-USD", cfg)
    assert dev.index.max() == pd.Timestamp("2024-11-06 23:00")
    assert len(dev) == 4
    assert (dev.index < pd.Timestamp("2024-11-07")).all()
    full = H.load_hourly("BTC-USD", cfg, include_holdout=True)
    assert full.index.max() == pd.Timestamp("2024-11-07 07:00") and len(full) == 12


# ------------------------------------------------------------------ features
def day_frame(day, n=24, seed=1, closes=None, opens=None, vols=None):
    idx = pd.date_range(day, periods=n, freq="h")
    rng = np.random.default_rng(seed)
    c = np.asarray(closes if closes is not None else 100 * np.exp(np.cumsum(rng.normal(0, .01, n))), float)
    o = np.asarray(opens if opens is not None else np.r_[100.0, c[:-1]], float)
    v = np.asarray(vols if vols is not None else rng.uniform(1, 5, n), float)
    return pd.DataFrame({"Open": o, "High": np.maximum(o, c) * 1.001, "Low": np.minimum(o, c) * .999,
                         "Close": c, "Volume": v}, index=idx)


def test_features_hand_computed_tiny_day():
    closes = [100.0 + h for h in range(24)]                       # 100..123
    opens = [99.0] + closes[:-1]
    vols = [0.0] * 23 + [4.0]
    prev = day_frame("2024-01-01", closes=[200.0] * 24, opens=[200.0] * 24)
    cur = day_frame("2024-01-02", closes=closes, opens=opens, vols=vols)
    f = H.daily_features(pd.concat([prev, cur]))
    d = pd.Timestamp("2024-01-02")
    rets = [math.log(closes[0] / 99.0)] + [math.log(closes[h] / closes[h - 1]) for h in range(1, 24)]
    assert f.loc[d, "rv_hourly"] == pytest.approx(math.sqrt(sum(r * r for r in rets)))
    assert f.loc[d, "ret_6h"] == pytest.approx(math.log(123.0 / 117.0))
    assert f.loc[d, "ret_24h"] == pytest.approx(math.log(123.0 / 200.0))
    assert f.loc[d, "vol_concentration"] == pytest.approx(1.0)       # all volume in one hour
    assert f.loc[pd.Timestamp("2024-01-01"), "rv_hourly"] == pytest.approx(0.0)
    assert math.isnan(f.loc[pd.Timestamp("2024-01-01"), "ret_24h"])  # no previous day
    even = day_frame("2024-01-03", vols=[2.0] * 24)
    assert H.daily_features(even)["vol_concentration"].iloc[0] == pytest.approx(1 / 24)


def test_features_nan_when_bars_missing_or_no_volume():
    d1 = day_frame("2024-01-01")
    d2 = day_frame("2024-01-02")
    d3 = day_frame("2024-01-03", vols=[0.0] * 24)
    full = pd.concat([d1, d2.drop(d2.index[[10]]), d3])
    f = H.daily_features(full)
    r2 = f.loc[pd.Timestamp("2024-01-02")]
    assert math.isnan(r2.rv_hourly) and math.isnan(r2.vol_concentration)
    assert not math.isnan(r2.ret_6h) and not math.isnan(r2.ret_24h)      # those bars exist
    assert math.isnan(f.loc[pd.Timestamp("2024-01-03"), "vol_concentration"])
    g = H.daily_features(pd.concat([d1, d2.drop(d2.index[[17]])]))
    assert math.isnan(g.loc[pd.Timestamp("2024-01-02"), "ret_6h"])
    g = H.daily_features(pd.concat([d1, d2.drop(d2.index[[23]])]))
    assert math.isnan(g.loc[pd.Timestamp("2024-01-02"), "ret_24h"])


def test_alignment_next_day_midnight_bar_never_enters_row_d():
    d = day_frame("2024-01-02")
    base = H.daily_features(d)
    nxt = day_frame("2024-01-03", n=1, closes=[1e6], opens=[1e6], vols=[1e9])   # 00:00 bar of D+1
    f = H.daily_features(pd.concat([d, nxt]))
    pd.testing.assert_series_equal(f.loc[pd.Timestamp("2024-01-02")], base.loc[pd.Timestamp("2024-01-02")])
    # row D uses exactly 00:00..23:00: changing any one of them changes rv; the 24th hour of D is 23:00
    for h in (0, 12, 23):
        m = d.copy()
        m.iloc[h, m.columns.get_loc("Close")] *= 1.05
        assert H.daily_features(m).rv_hourly.iloc[0] != base.rv_hourly.iloc[0]
    m = d.copy()
    m.iloc[23, m.columns.get_loc("Volume")] *= 3
    assert H.daily_features(m).vol_concentration.iloc[0] != base.vol_concentration.iloc[0]


def long_hourly(days, seed=3):
    """Synthetic hourly bars with volatility clustering, `days` full days."""
    rng = np.random.default_rng(seed)
    n = days * 24
    vol = np.exp(np.repeat(np.cumsum(rng.normal(0, .05, days)) * .3 - 5, 24) + rng.normal(0, .05, n))
    r = rng.normal(0, 1, n) * vol
    c = 100 * np.exp(np.cumsum(r))
    o = np.r_[100.0, c[:-1]]
    idx = pd.date_range("2022-01-01", periods=n, freq="h")
    return pd.DataFrame({"Open": o, "High": np.maximum(o, c) * 1.001, "Low": np.minimum(o, c) * .999,
                         "Close": c, "Volume": rng.uniform(1, 9, n)}, index=idx)


def har_reference(rv, D, min_pairs=250):
    """Independent recomputation with plain loops and the normal equations."""
    r = D - pd.Timedelta(days=D.toordinal() % 21)
    vals = rv.to_dict()

    def x(s):
        w5 = [vals.get(s - pd.Timedelta(days=k)) for k in range(5)]
        w22 = [vals.get(s - pd.Timedelta(days=k)) for k in range(22)]
        if any(v is None or not v > 0 for v in w22):
            return None
        return [1.0, math.log(vals[s]), math.log(sum(w5) / 5), math.log(sum(w22) / 22)]

    A, b = [], []
    for s in rv.index:
        t = s + pd.Timedelta(days=1)
        if t > r or t not in vals:
            continue
        xs = x(s)
        if xs is not None and vals[t] > 0:
            A.append(xs)
            b.append(math.log(vals[t]))
    if len(A) < min_pairs or x(D) is None:
        return float("nan"), len(A)
    A, b = np.array(A), np.array(b)
    beta = np.linalg.solve(A.T @ A, A.T @ b)
    return float(np.exp(np.array(x(D)) @ beta)), len(A)


def test_har_matches_independent_ols_and_nan_before_250_pairs():
    h = long_hourly(420)
    f = H.daily_features(h)
    rv = f["rv_hourly"]
    checked = 0
    for D in [f.index[i] for i in (30, 100, 200, 260, 270, 300, 380, 419)]:
        exp, npairs = har_reference(rv, D)
        got = f.loc[D, "har_rv_forecast"]
        if math.isnan(exp):
            assert math.isnan(got), (D, npairs)
        else:
            assert got == pytest.approx(exp, rel=1e-9)
            checked += 1
    assert checked >= 3
    early = f["har_rv_forecast"].iloc[:260]                       # 22-day warm-up + 250 pairs needed
    assert early.isna().all()
    assert f["har_rv_forecast"].iloc[-1] > 0
    # coefficients are piecewise constant between refits: same R -> same beta (forecast changes only via x)


def test_har_uses_only_targets_known_by_refit_day():
    """Changing rv strictly after the refit day R must not change the coefficients, so the forecast
    for a row D (with unchanged x_D) is unchanged."""
    h = long_hourly(420)
    f = H.daily_features(h)
    D = f.index[380]
    R = D - pd.Timedelta(days=D.toordinal() % 21)
    m = h.copy()
    cut = D + pd.Timedelta(days=1)                                 # mutate everything after day D
    m.loc[m.index >= cut, "Close"] *= np.exp(np.random.default_rng(5).normal(0, .1, (m.index >= cut).sum()))
    g = H.daily_features(m)
    assert g.loc[D, "har_rv_forecast"] == f.loc[D, "har_rv_forecast"]
    assert R <= D


def test_no_lookahead_append_and_mutate_future_bars():
    full = long_hourly(420)
    cut = pd.Timestamp("2022-11-01")                               # day boundary
    past = full[full.index < cut]
    base = H.daily_features(past)
    last_complete = cut - pd.Timedelta(days=1)
    f_full = H.daily_features(full)
    # appended future bars: every row up to the last complete day is bit-identical
    a = f_full.loc[:last_complete]
    b = base.loc[:last_complete]
    pd.testing.assert_frame_equal(a, b, check_exact=True)
    assert base.har_rv_forecast.notna().sum() > 10             # test is not vacuous
    # mutated future bars (wild prices/volumes): past rows still bit-identical
    rng = np.random.default_rng(9)
    m = full.copy()
    fut = m.index >= cut
    for c in ("Open", "Close", "High", "Low"):
        m.loc[fut, c] = m.loc[fut, c] * np.exp(rng.normal(0, .5, fut.sum()))
    m.loc[fut, "Volume"] = rng.uniform(0, 1e6, fut.sum())
    f_mut = H.daily_features(m)
    pd.testing.assert_frame_equal(f_mut.loc[:last_complete], b, check_exact=True)
    assert not f_mut.loc[cut:].equals(f_full.loc[cut:])        # the mutation really did change the future
    # a partially present last day stays NaN (not yet known) and never alters earlier rows
    part = full[full.index < cut + pd.Timedelta(hours=7)]
    fp = H.daily_features(part)
    assert math.isnan(fp.loc[cut, "rv_hourly"])
    pd.testing.assert_frame_equal(fp.loc[:last_complete], b, check_exact=True)


def test_features_empty_input():
    assert list(H.daily_features(make_bars("2024-01-01", 5).iloc[:0]).columns) == H.FEATURE_COLUMNS


def test_main_error_line_and_exit_code(monkeypatch, capsys):
    monkeypatch.setattr(H, "update_all", lambda *a, **k: (_ for _ in ()).throw(
        H.HourlyDownloadError("binance BTC/USDT: boom")))
    with pytest.raises(SystemExit) as e:
        H.main()
    assert e.value.code == 1
    out = capsys.readouterr().out.strip().splitlines()
    assert len(out) == 1 and out[0].startswith("ERROR: binance BTC/USDT")
