import logging

import pandas as pd
import pytest

from quant import data
from quant.data import DataDownloadError
from quant.stage7 import universe as U
from quant.stage7.config import load_stage7_config

NOW = pd.Timestamp("2022-01-10 23:00", tz="UTC")     # Monday after close
CUT = "2021-06-30"


def make(dates, price=100.0):
    idx = pd.DatetimeIndex(dates)
    return pd.DataFrame({"Open": price, "High": price + 1, "Low": price - 1,
                         "Close": [price + i for i in range(len(idx))], "Volume": 1000.0}, index=idx)


def _cfg(tmp_path, tickers=("SPY", "XLRE", "BTC-USD", "ETH-USD")):
    cfg = load_stage7_config()
    cfg["universe"] = [a for a in cfg["universe"] if a["ticker"] in tickers]
    cfg["data_dir"] = tmp_path
    cfg["holdout_cutoff_date"] = CUT
    return cfg


@pytest.fixture
def world(monkeypatch):
    frames, bad = {}, {}

    def fake(ticker, start, end=None):
        if ticker in bad:
            raise bad[ticker]
        f = frames[ticker]
        return f[f.index >= pd.Timestamp(start)]
    monkeypatch.setattr(data, "_download", fake)
    w = type("W", (), {})()
    w.frames, w.bad = frames, bad
    # 2019-01-02.. for SPY (all weekdays, XNYS holidays removed), XLRE short, crypto daily
    cal = data._xnys()
    sess = cal.sessions_in_range("2019-01-02", "2022-01-10").tz_localize(None)
    frames["SPY"] = make(sess)
    frames["XLRE"] = make(sess[sess >= "2021-03-01"])        # ~ 200 rows < 252 warm-up
    frames["BTC-USD"] = make(pd.date_range("2019-01-01", "2022-01-09"))
    frames["ETH-USD"] = make(pd.date_range("2019-06-01", "2022-01-09"))
    return w


def test_load_universe_dev_only_by_default(world, tmp_path):
    cfg = _cfg(tmp_path)
    U.update_universe(cfg, now=NOW)
    dev = U.load_universe(cfg)
    assert set(dev) == {"SPY", "XLRE", "BTC-USD", "ETH-USD"}
    for t, df in dev.items():
        assert df.index.max() == pd.Timestamp(CUT)
    full = U.load_universe(cfg, include_holdout=True)
    assert full["SPY"].index.max() == pd.Timestamp("2022-01-10")
    assert full["BTC-USD"].index.max() == pd.Timestamp("2022-01-09")
    assert len(full["BTC-USD"]) > len(dev["BTC-USD"])
    assert list(U.load_universe(cfg, tickers=["SPY"])) == ["SPY"]
    with pytest.raises(KeyError):
        U.load_universe(cfg, tickers=["NOPE"])


def test_eligible_from_arithmetic(world, tmp_path):
    cfg = _cfg(tmp_path)
    spy = world.frames["SPY"]
    assert U.eligible_from(spy, "SPY", cfg) == spy.index[252]
    btc = world.frames["BTC-USD"]
    assert U.eligible_from(btc, "BTC-USD", cfg) == pd.Timestamp("2019-01-01") + pd.Timedelta(days=365)
    exactly = make(pd.bdate_range("2020-01-01", periods=252))
    assert pd.isna(U.eligible_from(exactly, "SPY", cfg))          # needs row #252 to exist
    assert U.eligible_from(make(pd.bdate_range("2020-01-01", periods=253)), "SPY", cfg) \
        == pd.bdate_range("2020-01-01", periods=253)[252]
    short = world.frames["XLRE"].iloc[:200]
    assert pd.isna(U.eligible_from(short, "XLRE", cfg))


def test_available_assets_no_backfilling(world, tmp_path):
    cfg = _cfg(tmp_path)
    U.update_universe(cfg, now=NOW)
    dev = U.load_universe(cfg)
    assert U.available_assets("2019-06-01", cfg, dev) == []
    e = {t: U.eligible_from(dev[t], t, cfg) for t in ("SPY", "BTC-USD", "ETH-USD")}
    assert e["BTC-USD"] == pd.Timestamp("2020-01-01")       # first row 2019-01-01 + 365 rows
    one = pd.Timedelta(days=1)
    for t, d in e.items():
        assert t not in U.available_assets(d - one, cfg, dev)     # not a day early
        assert t in U.available_assets(d, cfg, dev)               # in on the eligible day
    assert U.available_assets(e["ETH-USD"] - one, cfg, dev) == ["SPY", "BTC-USD"]
    assert U.available_assets(e["ETH-USD"], cfg, dev) == ["SPY", "BTC-USD", "ETH-USD"]
    assert e["ETH-USD"] == pd.Timestamp("2020-05-31")       # 2019-06-01 + 365 rows
    # XLRE has fewer than 252 dev rows -> never available
    assert "XLRE" not in U.available_assets(CUT, cfg, dev)
    assert U.available_assets(CUT, cfg, dev) == ["SPY", "BTC-USD", "ETH-USD"]


def test_failing_ticker_named_others_still_update(world, tmp_path):
    cfg = _cfg(tmp_path)
    world.bad["XLRE"] = RuntimeError("boom")
    with pytest.raises(U.UniverseUpdateError) as ei:
        U.update_universe(cfg, now=NOW)
    msg = str(ei.value)
    assert "XLRE" in msg and "boom" in msg and "1 universe ticker" in msg
    for t in ("SPY", "BTC-USD", "ETH-USD"):
        assert (tmp_path / f"{t}.parquet").exists()
    assert not (tmp_path / "XLRE.parquet").exists()


def test_all_failures_listed(world, tmp_path):
    cfg = _cfg(tmp_path)
    world.bad["SPY"] = DataDownloadError("SPY: nothing")
    world.bad["ETH-USD"] = RuntimeError("x")
    with pytest.raises(U.UniverseUpdateError) as ei:
        U.update_universe(cfg, now=NOW)
    assert "SPY" in str(ei.value) and "ETH-USD" in str(ei.value) and "2 universe" in str(ei.value)


def test_validation_gap_reported_with_ticker(world, tmp_path):
    cfg = _cfg(tmp_path)
    f = world.frames["SPY"]
    world.frames["SPY"] = f.drop(f.index[300])
    with pytest.raises(U.UniverseUpdateError, match="SPY.*missing trading session"):
        U.update_universe(cfg, now=NOW)


def test_staleness_warning_propagates(world, tmp_path):
    cfg = _cfg(tmp_path, tickers=("SPY",))
    world.frames["SPY"] = world.frames["SPY"][lambda d: d.index <= "2022-01-05"]
    warns = []
    U.update_universe(cfg, now=NOW, warnings=warns)
    assert len(warns) == 1 and "SPY: STALE DATA" in warns[0] and "2022-01-05" in warns[0]
    warns2 = []
    world.frames["SPY"] = make(data._xnys().sessions_in_range("2019-01-02", "2022-01-10").tz_localize(None))
    U.update_universe(cfg, now=NOW, warnings=warns2)
    assert warns2 == []


def test_start_date_table(world, tmp_path):
    cfg = _cfg(tmp_path)
    U.update_universe(cfg, now=NOW)
    tab = U.start_date_table(cfg)
    assert list(tab.columns) == ["ticker", "name", "group", "first_date", "eligible_from",
                                 "rows", "last_dev_date"]
    assert list(tab.ticker) == ["SPY", "XLRE", "BTC-USD", "ETH-USD"]
    r = tab.set_index("ticker")
    assert r.loc["SPY", "group"] == "us_equity" and r.loc["BTC-USD", "group"] == "crypto"
    assert r.loc["SPY", "first_date"] == pd.Timestamp("2019-01-02")
    assert (r["last_dev_date"] == pd.Timestamp(CUT)).all()
    dev = U.load_universe(cfg)
    for t in r.index:
        assert r.loc[t, "rows"] == len(dev[t])
    assert r.loc["SPY", "eligible_from"] == dev["SPY"].index[252]
    assert pd.isna(r.loc["XLRE", "eligible_from"])
    # passing full (incl. hold-out) data must still not leak post-cutoff rows
    full = U.load_universe(cfg, include_holdout=True)
    tab2 = U.start_date_table(cfg, full)
    assert (tab2["last_dev_date"] == pd.Timestamp(CUT)).all()
    assert list(tab2["rows"]) == list(tab["rows"])


def test_config_rejects_duplicates_and_wrong_group(tmp_path):
    import yaml
    base = yaml.safe_load((load_stage7_config()["project_root"] / "config_stage7.yaml").read_text())

    def write(mut):
        c = yaml.safe_load(yaml.safe_dump(base))
        mut(c)
        p = tmp_path / "c.yaml"
        p.write_text(yaml.safe_dump(c))
        return p
    with pytest.raises(ValueError, match="duplicate ticker SPY"):
        load_stage7_config(write(lambda c: c["universe"].append(dict(c["universe"][0]))))
    with pytest.raises(ValueError, match="unknown group"):
        load_stage7_config(write(lambda c: c["universe"][0].update(group="stocks")))
    with pytest.raises(ValueError, match="trades_weekends"):
        load_stage7_config(write(lambda c: c["universe"][0].update(trades_weekends=True)))


def test_real_config_universe_shape():
    cfg = load_stage7_config()
    assert len(cfg["universe"]) == 30
    assert U.warmup_for("SPY", cfg) == 252 and U.warmup_for("ETH-USD", cfg) == 365


def test_main_exit_code_one_line_error(world, tmp_path, monkeypatch, capsys):
    cfg = _cfg(tmp_path)
    world.bad["SPY"] = RuntimeError("down")
    monkeypatch.setattr(U, "load_stage7_config", lambda: cfg)
    with pytest.raises(SystemExit) as ei:
        U.main()
    assert ei.value.code == 1
    out = capsys.readouterr().out
    assert out.startswith("ERROR:") and "SPY" in out and out.count("\n") == 1
