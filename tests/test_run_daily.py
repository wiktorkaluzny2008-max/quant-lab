"""Stage 4 daily run (quant/run_daily.py): idempotency, catch-up, stale data, CLI errors."""
import hashlib

import numpy as np
import pandas as pd
import pytest
import yaml
from pandas.testing import assert_frame_equal

from quant import paper as P
from quant import run_daily as R
from tests.synth import FAST_BT, TICKERS, make_cfg, make_results, truncate_world, world

PX = world()


@pytest.fixture(autouse=True)
def fast_models(monkeypatch):
    monkeypatch.setattr(P, "BT", FAST_BT)


def new_world(tmp_path, **paper):
    cfg = make_cfg(tmp_path, **paper)
    make_results(cfg)
    return cfg


def go(cfg, asof, px=PX, **kw):
    return R.run(cfg, asof=pd.Timestamp(asof), prices=px, update=None, out=lambda m: None, **kw)


def ledger(cfg):
    return P.load_ledger(P.PaperFiles.for_config(cfg))


def digest(root):
    return {str(p.relative_to(root)): (hashlib.sha256(p.read_bytes()).hexdigest(), p.stat().st_mtime_ns)
            for p in sorted(root.rglob("*")) if p.is_file()}


# --------------------------------------------------------------------------- idempotent
def test_second_run_same_day_leaves_every_file_byte_identical(tmp_path):
    cfg = new_world(tmp_path)
    r1 = go(cfg, "2021-10-15")
    root = tmp_path / "paper"
    first = digest(root)
    assert r1.wrote and {"ledger.parquet", "state.json"} <= set(first)
    assert any(k.startswith("models/") for k in first)
    r2 = go(cfg, "2021-10-15")
    assert not r2.wrote and r2.new_rows is None
    assert digest(root) == first                       # same bytes AND same mtimes: never rewritten
    assert len(ledger(cfg)) == 4
    assert not list(root.rglob("*.tmp"))


def test_idempotent_after_catch_up_and_with_the_real_clock_path(tmp_path):
    cfg = new_world(tmp_path)
    for now in ("2021-10-15 21:00:00", "2021-10-15 23:30:00"):     # --asof omitted: today by the clock
        r = R.run(cfg, prices=truncate_world(PX, "2021-10-15"), update=None, now=now, out=lambda m: None)
        if now.endswith("21:00:00"):
            first = digest(tmp_path / "paper")
    assert digest(tmp_path / "paper") == first
    go(cfg, "2021-11-10")
    again = digest(tmp_path / "paper")
    go(cfg, "2021-11-10")
    assert digest(tmp_path / "paper") == again
    L = ledger(cfg)
    assert not L.duplicated(["asset", "date"]).any()


def test_no_wall_clock_in_ledger_or_state(tmp_path):
    cfg = new_world(tmp_path)
    go(cfg, "2021-10-15", now="2031-01-01 00:00:00")        # a far-future real clock changes nothing
    st = P.load_state(P.PaperFiles.for_config(cfg))
    assert set(st) == {"version", "start_date", "settings", "holdout_cutoff", "holdout_end", "assets"}
    assert st["start_date"] == "2021-10-15"
    assert not any(k in c for c in P.LEDGER_COLS for k in ("time", "stamp", "updated", "created"))
    ref = digest(tmp_path / "paper")
    cfg2 = new_world(tmp_path / "other")
    go(cfg2, "2021-10-15", now="2021-10-15 20:00:00")
    a = {k: v[0] for k, v in ref.items()}
    b = {k: v[0] for k, v in digest(tmp_path / "other" / "paper").items()}
    assert a == b                                          # same inputs -> same bytes, any clock


def test_rerun_with_earlier_asof_does_nothing(tmp_path):
    cfg = new_world(tmp_path)
    go(cfg, "2021-10-15"); go(cfg, "2021-10-29")
    snap = digest(tmp_path / "paper")
    r = go(cfg, "2021-10-20")                           # inside the recorded range
    assert not r.wrote
    assert digest(tmp_path / "paper") == snap


# --------------------------------------------------------------------------- catch-up
def test_catch_up_equals_day_by_day(tmp_path):
    a = new_world(tmp_path / "daily")
    b = new_world(tmp_path / "catchup")
    days = pd.bdate_range("2021-10-01", "2021-11-12")
    for d in days:
        go(a, d)
    go(b, "2021-10-01")
    go(b, "2021-11-12")                                  # six weeks missed, one run
    La, Lb = ledger(a), ledger(b)
    assert len(La) > 100
    assert_frame_equal(La, Lb)
    sa = P.load_state(P.PaperFiles.for_config(a)); sb = P.load_state(P.PaperFiles.for_config(b))
    assert sa == sb
    # the catch-up run recorded every missed trading day, BTC every calendar day
    spy = Lb[Lb.asset == "SPY"]["date"]
    sess = PX["SPY"].index
    assert list(spy) == list(sess[(sess >= "2021-10-01") & (sess <= "2021-11-12")])
    btc = Lb[Lb.asset == "BTC-USD"]["date"]
    assert (btc.diff().dropna() == pd.Timedelta(days=1)).all()


def test_catch_up_with_a_weekend_and_holiday_gap_marks_across_it(tmp_path):
    cfg = new_world(tmp_path)
    go(cfg, "2021-11-19")                                # Friday
    go(cfg, "2021-11-30")                                # Thanksgiving + weekend missed
    L = ledger(cfg)
    spy = L[L.asset == "SPY"].reset_index(drop=True)
    assert pd.Timestamp("2021-11-25") not in set(spy["date"])           # holiday
    i = spy.index[spy["date"] == pd.Timestamp("2021-11-22")][0]
    close = PX["SPY"]["Close"]
    assert spy.loc[i, "asset_return"] == pytest.approx(close["2021-11-22"] / close["2021-11-19"] - 1)


# --------------------------------------------------------------------------- stale / missing data
def test_stale_asset_is_skipped_with_a_clear_message_and_others_proceed(tmp_path):
    cfg = new_world(tmp_path)
    px = dict(PX)
    px["BTC-USD"] = PX["BTC-USD"][PX["BTC-USD"].index <= "2021-10-08"]
    px["IWM"] = PX["IWM"][PX["IWM"].index <= "2021-10-12"]
    lines = []
    r = R.run(cfg, asof=pd.Timestamp("2021-10-15"), prices=px, update=None, out=lines.append)
    assert r.exit_code == 0
    assert set(r.skipped) == {"BTC-USD", "IWM"}
    text = "\n".join(lines)
    assert "SKIPPED BTC-USD: STALE DATA" in text and "2021-10-08" in text
    assert "SKIPPED IWM: STALE DATA" in text
    L = ledger(cfg)
    assert set(L["asset"]) == {"SPY", "QQQ"}
    # a later run with fresh data starts the skipped assets then; nothing is invented for the gap
    go(cfg, "2021-10-22")
    L = ledger(cfg)
    assert L[L.asset == "BTC-USD"]["date"].min() == pd.Timestamp("2021-10-21")
    assert L[L.asset == "IWM"]["date"].min() == pd.Timestamp("2021-10-22")


def test_stale_asset_with_existing_ledger_is_not_extended(tmp_path):
    cfg = new_world(tmp_path)
    go(cfg, "2021-10-15")
    px = dict(PX)
    px["SPY"] = PX["SPY"][PX["SPY"].index <= "2021-10-18"]
    r = go(cfg, "2021-10-22", px)
    assert "SPY" in r.skipped
    L = ledger(cfg)
    assert L[L.asset == "SPY"]["date"].max() == pd.Timestamp("2021-10-15")
    assert L[L.asset == "QQQ"]["date"].max() == pd.Timestamp("2021-10-22")
    go(cfg, "2021-10-22")                                # data is back: every missed day is filled
    spy = ledger(cfg).query("asset == 'SPY'")["date"]
    assert spy.max() == pd.Timestamp("2021-10-22") and spy.is_unique


def test_missing_price_file_skips_everything_with_exit_zero(tmp_path, monkeypatch, capsys):
    cfg = new_world(tmp_path)                            # no price parquet files at all
    monkeypatch.setattr(R, "load_config", lambda path=None: cfg)
    code = R.main(["--asof", "2021-10-15"], update=None)
    out = capsys.readouterr()
    assert code == 0
    assert "SKIPPED SPY" in out.out and "Traceback" not in out.out + out.err
    assert not (tmp_path / "paper").exists()


def test_download_failure_is_a_warning_and_stored_data_is_used(tmp_path):
    cfg = new_world(tmp_path)
    lines = []

    def boom(c, now, warns):
        raise R.D.DataDownloadError("SPY: network down")
    r = R.run(cfg, asof=pd.Timestamp("2021-10-15"), prices=None, update=boom, out=lines.append)
    assert r.exit_code == 0                              # no stored prices either: skipped, not crashed
    assert any("data update failed" in m for m in lines)
    assert set(r.skipped) == set(TICKERS)


def test_stale_input_series_is_not_silently_carried_forward(tmp_path):
    cfg = new_world(tmp_path)
    px = dict(PX)
    px["^VIX"] = PX["^VIX"][PX["^VIX"].index <= "2021-10-14"]           # VIX missing for 10-15
    lines = []
    r = R.run(cfg, asof=pd.Timestamp("2021-10-15"), prices=px, update=None, out=lines.append)
    assert set(r.skipped) == set(TICKERS)
    assert all("STALE INPUT DATA" in v and "^VIX" in v for v in r.skipped.values())
    assert not (tmp_path / "paper").exists()
    # cross-asset dependency: a stale SPY stops IWM and BTC but not QQQ
    px = dict(PX)
    px["SPY"] = PX["SPY"][PX["SPY"].index <= "2021-10-13"]
    r = go(cfg, "2021-10-15", px)
    assert set(r.skipped) == {"SPY", "IWM", "BTC-USD"}
    assert "SPY" in r.skipped["IWM"] and set(ledger(cfg)["asset"]) == {"QQQ"}


# --------------------------------------------------------------------------- dry run / subsets
def test_dry_run_writes_nothing(tmp_path):
    cfg = new_world(tmp_path)
    lines = []
    r = R.run(cfg, asof=pd.Timestamp("2021-10-15"), prices=PX, update=None, dry_run=True, out=lines.append)
    assert not r.wrote and r.new_rows is not None and len(r.new_rows) == 4
    assert not (tmp_path / "paper").exists()
    assert any("nothing was written" in m for m in lines)


def test_assets_subset_and_later_start_for_other_assets(tmp_path):
    cfg = new_world(tmp_path)
    go(cfg, "2021-10-15", assets=["SPY"])
    assert set(ledger(cfg)["asset"]) == {"SPY"}
    go(cfg, "2021-10-22")
    L = ledger(cfg)
    assert L[L.asset == "SPY"]["date"].min() == pd.Timestamp("2021-10-15")
    assert L[L.asset == "QQQ"]["date"].min() == pd.Timestamp("2021-10-22")     # no retroactive rows
    assert P.load_state(P.PaperFiles.for_config(cfg))["start_date"] == "2021-10-15"


# --------------------------------------------------------------------------- real errors
def cli(cfg, monkeypatch, argv, capsys, **kw):
    monkeypatch.setattr(R, "load_config", lambda path=None: cfg)
    code = R.main(argv, prices=PX, update=None, **kw)
    cap = capsys.readouterr()
    return code, cap.out, cap.err


@pytest.mark.parametrize("argv,needle", [
    (["--asof", "not-a-date"], "--asof must be a date"),
    (["--asof", "2021-8-3"], "--asof must be a date"),
    (["--asof", "2021-08-13"], "hold-out"),
    (["--asof", "2021-10-15", "--assets", "XYZ"], "unknown asset"),
])
def test_cli_real_errors_are_one_clean_line_with_exit_2(tmp_path, monkeypatch, capsys, argv, needle):
    cfg = new_world(tmp_path)
    code, out, err = cli(cfg, monkeypatch, argv, capsys)
    assert code == 2
    assert err.startswith("ERROR:") and needle in err
    assert "Traceback" not in err + out and err.count("\n") == 1
    assert not (tmp_path / "paper").exists()


def test_cli_success_exit_zero_and_prints_positions(tmp_path, monkeypatch, capsys):
    cfg = new_world(tmp_path)
    code, out, err = cli(cfg, monkeypatch, ["--asof", "2021-10-15"], capsys)
    assert code == 0 and err == ""
    for t in TICKERS:
        assert t in out
    assert "position" in out and "base rate" in out


def test_changing_the_strategy_after_the_ledger_started_is_refused(tmp_path, monkeypatch, capsys):
    cfg = new_world(tmp_path)
    go(cfg, "2021-10-15")
    snap = digest(tmp_path / "paper")
    cfg2 = make_cfg(tmp_path, strategy_mode="default_long")
    code, out, err = cli(cfg2, monkeypatch, ["--asof", "2021-10-22"], capsys)
    assert code == 2 and "differ" in err and "Traceback" not in err
    assert digest(tmp_path / "paper") == snap


def test_asof_before_the_ledger_start_is_refused(tmp_path, monkeypatch, capsys):
    cfg = new_world(tmp_path)
    go(cfg, "2021-10-15")
    code, out, err = cli(cfg, monkeypatch, ["--asof", "2021-10-04"], capsys)
    assert code == 2 and "before the ledger start" in err


def test_corrupt_or_foreign_ledger_is_a_clean_error(tmp_path, monkeypatch, capsys):
    cfg = new_world(tmp_path)
    go(cfg, "2021-10-15")
    (tmp_path / "paper" / "ledger.parquet").write_bytes(b"not parquet")
    code, out, err = cli(cfg, monkeypatch, ["--asof", "2021-10-22"], capsys)
    assert code == 2 and err.startswith("ERROR:") and "ledger" in err and "Traceback" not in err
    (tmp_path / "paper" / "ledger.parquet").unlink()
    code, out, err = cli(cfg, monkeypatch, ["--asof", "2021-10-22"], capsys)
    assert code == 2 and ("disagree" in err or "no entry" in err) and "Traceback" not in err


def test_corrupt_state_file_is_a_clean_error(tmp_path, monkeypatch, capsys):
    cfg = new_world(tmp_path)
    go(cfg, "2021-10-15")
    (tmp_path / "paper" / "state.json").write_text("{broken")
    code, out, err = cli(cfg, monkeypatch, ["--asof", "2021-10-22"], capsys)
    assert code == 2 and err.startswith("ERROR:") and "state file" in err


def test_missing_stage3_results_is_a_clean_error(tmp_path, monkeypatch, capsys):
    cfg = make_cfg(tmp_path)                             # no results written
    code, out, err = cli(cfg, monkeypatch, ["--asof", "2021-10-15"], capsys)
    assert code == 2 and "trial log" in err and "Traceback" not in err


def test_unexpected_failure_has_no_traceback(tmp_path, monkeypatch, capsys):
    cfg = new_world(tmp_path)
    monkeypatch.setattr(R, "run", lambda *a, **k: (_ for _ in ()).throw(ZeroDivisionError("boom")))
    code, out, err = cli(cfg, monkeypatch, ["--asof", "2021-10-15"], capsys)
    assert code == 2 and err.startswith("ERROR: unexpected failure") and "Traceback" not in err


def test_never_connects_to_a_broker_or_network(tmp_path, monkeypatch):
    """run() with injected prices and update=None makes no download; a download attempt would
    hit this sentinel."""
    cfg = new_world(tmp_path)
    monkeypatch.setattr(R.D, "_download", lambda *a, **k: (_ for _ in ()).throw(AssertionError("network")))
    go(cfg, "2021-10-15")


# --------------------------------------------------------------------------- hold-out never scored
def test_holdout_period_returns_never_enter_a_live_metric(tmp_path, monkeypatch):
    """Garbage prices inside the hold-out calendar period (after the cutoff, before the live start)
    may change the model (it trains on them for FUTURE predictions) but cannot change which dates
    the live metrics use: every live metric date is > the hold-out end."""
    cfg = new_world(tmp_path)
    seen = []
    real = P.assert_live_dates
    monkeypatch.setattr(P, "assert_live_dates",
                        lambda d, s, h, what="x": (seen.append(pd.DatetimeIndex(d)), real(d, s, h, what))[1])
    go(cfg, "2021-10-01"); go(cfg, "2021-10-29")
    hold_end = P.holdout_end(cfg)
    assert seen and all(len(d) == 0 or d.min() > hold_end for d in seen)
    L = ledger(cfg)
    assert (L["date"] >= hold_end).all()
    assert L.loc[L["live_n"] > 0, "date"].min() > hold_end
