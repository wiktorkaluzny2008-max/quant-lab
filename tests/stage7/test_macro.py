"""Offline tests for quant.stage7.macro (network function is mocked)."""
import json
import logging

import numpy as np
import pandas as pd
import pytest

from quant.stage7 import macro as M
from quant.stage7.config import load_stage7_config

NOW = "2024-11-20"


@pytest.fixture
def cfg(tmp_path):
    c = load_stage7_config()
    c["data_dir"] = tmp_path
    return c


def csv(series, rows):
    return "observation_date," + series + "\n" + "\n".join(f"{d},{v}" for d, v in rows) + "\n"


def mock_net(monkeypatch, texts):
    calls = []

    def fake(url):
        calls.append(url)
        sid = url.split("id=")[1].split("&")[0]
        t = texts[sid]
        if isinstance(t, Exception):
            raise t
        return t
    monkeypatch.setattr(M, "_fetch_text", fake)
    return calls


def S(rows, name="value"):
    idx = pd.DatetimeIndex([pd.Timestamp(d) for d, _ in rows], name="observation_date")
    return pd.Series([float(v) for _, v in rows], index=idx, name=name)


def store(cfg, sid, rows):
    s = S(rows)
    M._write_atomic(s.to_frame(), M._fred_dir(cfg) / f"{sid}.parquet")


# ---------------------------------------------------------------- parsing / download
def test_parse_missing_dots_and_blank():
    s, n = M.parse_fred_csv("observation_date,DFF\n2024-01-01,5.3\n2024-01-02,.\n2024-01-03,\n2024-01-04,5.4\n", "DFF")
    assert n == 2 and list(s.values) == [5.3, 5.4]
    assert s.index.name == "observation_date" and s.dtype == "float64" and s.name == "value"


def test_bad_csv_names_series():
    with pytest.raises(M.FredDownloadError, match="DFF"):
        M.parse_fred_csv("<html>not csv at all</html>", "DFF")
    with pytest.raises(M.FredDownloadError, match="DFF.*non-numeric"):
        M.parse_fred_csv("observation_date,DFF\n2024-01-01,abc\n", "DFF")
    with pytest.raises(M.FredDownloadError, match="DFF.*date"):
        M.parse_fred_csv("observation_date,DFF\nnot-a-date,5\n", "DFF")


def test_http_error_names_series(cfg, monkeypatch):
    mock_net(monkeypatch, {"DFF": OSError("HTTP 503")})
    with pytest.raises(M.FredDownloadError, match="DFF.*503"):
        M.update_series("DFF", cfg, now=NOW)


def test_url_format(cfg, monkeypatch):
    calls = mock_net(monkeypatch, {"DFF": csv("DFF", [("2024-11-18", 4.6)])})
    M.update_series("DFF", cfg, now=NOW)
    assert calls == [f"{cfg['fred']['base_url']}?id=DFF&cosd=1990-01-01"]


def test_unknown_series_rejected(cfg):
    with pytest.raises(ValueError, match="unknown FRED series"):
        M.load_fred("CPIAUCSL", cfg)
    with pytest.raises(ValueError, match="unknown FRED series"):
        M.download_series("CPIAUCSL", "1990-01-01", cfg)
    with pytest.raises(ValueError, match="unknown FRED series"):
        M.update_all(cfg, series=["UNRATE"])


# ---------------------------------------------------------------- validation
@pytest.mark.parametrize("rows,msg", [
    ([], "no observations"),
    ([("2024-11-01", 4.0), ("2024-11-01", 4.1)], "duplicate"),
    ([("2024-11-04", 4.0), ("2024-11-01", 4.1)], "not sorted"),
    ([("2024-11-01", 4.0), ("2024-11-04", 99.0)], "outside plausible range"),
    ([("2024-11-01", 4.0), ("2024-11-04", -5.0)], "outside plausible range"),
    ([("2024-11-01", 4.0), ("2024-12-25", 4.1)], "future date"),
])
def test_validation_failures(cfg, rows, msg):
    s = S(rows) if rows else pd.Series([], dtype="float64", index=pd.DatetimeIndex([]))
    with pytest.raises(M.FredValidationError, match=msg):
        M.validate_series(s, "DFF", cfg, now=NOW)


def test_validation_nonfinite(cfg):
    s = S([("2024-11-01", 4.0), ("2024-11-04", np.inf)])
    with pytest.raises(M.FredValidationError, match="non-finite"):
        M.validate_series(s, "DFF", cfg, now=NOW)


def test_validation_passes(cfg):
    M.validate_series(S([("2024-11-01", 4.0), ("2024-11-04", 4.1)]), "DFF", cfg, now=NOW)


def test_validation_failure_does_not_write(cfg, monkeypatch):
    mock_net(monkeypatch, {"DFF": csv("DFF", [("2024-11-01", 99)])})
    with pytest.raises(M.FredValidationError):
        M.update_series("DFF", cfg, now=NOW)
    assert not (M._fred_dir(cfg) / "DFF.parquet").exists()


# ---------------------------------------------------------------- staleness
def test_staleness_warning(caplog):
    s = S([("2024-11-01", 4.0)])
    with caplog.at_level(logging.WARNING):
        assert "STALE" in M.check_staleness(s, "DFF", now="2024-11-20")
    assert any("STALE" in r.message for r in caplog.records)
    assert M.check_staleness(s, "DFF", now="2024-11-08") is None      # 7 days: fine


def test_stale_series_still_stored(cfg, monkeypatch):
    mock_net(monkeypatch, {"DFF": csv("DFF", [("2024-10-01", 4.6)])})
    st = M.update_series("DFF", cfg, now=NOW)
    assert st["stale_warning"] and (M._fred_dir(cfg) / "DFF.parquet").exists()


# ---------------------------------------------------------------- storage / incremental / status
def test_update_stores_and_status(cfg, monkeypatch):
    mock_net(monkeypatch, {"DFF": csv("DFF", [("2024-11-14", 4.6), ("2024-11-15", "."), ("2024-11-18", 4.7)])})
    M.update_all(cfg, now=NOW, series=["DFF"])
    df = pd.read_parquet(M._fred_dir(cfg) / "DFF.parquet")
    assert list(df.columns) == ["value"] and df["value"].dtype == "float64"
    assert df.index.name == "observation_date" and df.index.tz is None and len(df) == 2
    st = json.loads((M._fred_dir(cfg) / "status.json").read_text())["DFF"]
    assert st["rows"] == 2 and st["missing_dropped_in_last_download"] == 1
    assert st["first_date"] == "2024-11-14" and st["last_date"] == "2024-11-18"
    assert st["source_url"].startswith(cfg["fred"]["base_url"]) and st["downloaded_at"]


def test_incremental_merge_keep_last(cfg, monkeypatch):
    store(cfg, "DFF", [("2024-09-02", 5.0), ("2024-10-28", 4.5), ("2024-11-01", 4.6)])
    calls = mock_net(monkeypatch, {"DFF": csv("DFF", [("2024-11-01", 4.65), ("2024-11-04", 4.7)])})
    st = M.update_series("DFF", cfg, now=NOW)
    assert "cosd=2024-10-04" in calls[0]                    # last stored date - 28 days
    s = pd.read_parquet(M._fred_dir(cfg) / "DFF.parquet")["value"]
    assert list(s.index.strftime("%Y-%m-%d")) == ["2024-09-02", "2024-10-28", "2024-11-01", "2024-11-04"]
    assert s.loc["2024-11-01"] == 4.65 and st["rows"] == 4 and st["first_date"] == "2024-09-02"


def test_late_start_is_recorded_not_an_error(cfg, monkeypatch):
    mock_net(monkeypatch, {"BAMLH0A0HYM2": csv("BAMLH0A0HYM2", [("2023-11-20", 3.5), ("2024-11-18", 3.0)])})
    assert M.update_series("BAMLH0A0HYM2", cfg, now=NOW)["first_date"] == "2023-11-20"


# ---------------------------------------------------------------- availability / as-of
def test_availability_weekend_and_holiday():
    av = M.availability_date(pd.to_datetime(["2024-11-01", "2024-11-04", "2024-11-27", "2024-11-28", "2024-11-02"]))
    assert list(av.strftime("%Y-%m-%d")) == [
        "2024-11-04",   # Friday -> Monday
        "2024-11-05",   # Monday -> Tuesday
        "2024-11-29",   # day before Thanksgiving -> day after the holiday
        "2024-11-29",   # holiday-dated -> next session
        "2024-11-04",   # Saturday-dated -> Monday
    ]


def test_asof_etf_calendar():
    s = S([("2024-10-31", 1.0), ("2024-11-01", 2.0)])        # Thu, Fri
    idx = pd.bdate_range("2024-10-31", "2024-11-06")
    out = M.asof_macro(idx, s)
    assert np.isnan(out["2024-10-31"])                       # nothing known yet
    assert out["2024-11-01"] == 1.0                          # Thu value known Fri
    assert out["2024-11-04"] == 2.0                          # Fri value first visible Monday
    assert out["2024-11-05"] == 2.0                          # carried


def test_asof_crypto_7day_calendar():
    s = S([("2024-10-31", 1.0), ("2024-11-01", 2.0)])
    idx = pd.date_range("2024-11-01", "2024-11-05", freq="D")  # Fri..Tue
    out = M.asof_macro(idx, s)
    assert out["2024-11-01"] == 1.0
    assert out["2024-11-02"] == 1.0 and out["2024-11-03"] == 1.0   # Sat/Sun: Friday's value NOT yet known
    assert out["2024-11-04"] == 2.0                                # Monday sees Friday's


def test_asof_holiday_carry():
    s = S([("2024-11-26", 1.0), ("2024-11-27", 2.0)])
    idx = pd.date_range("2024-11-27", "2024-11-30", freq="D")
    out = M.asof_macro(idx, s)
    assert out["2024-11-28"] == 1.0           # Thanksgiving: Wed value not visible until Fri
    assert out["2024-11-29"] == 2.0


def test_asof_dict_returns_wide_frame():
    out = M.asof_macro(pd.bdate_range("2024-11-04", "2024-11-06"),
                       {"A": S([("2024-11-01", 1.0)]), "B": S([("2024-11-04", 7.0)])})
    assert list(out.columns) == ["A", "B"]
    assert out["A"].tolist() == [1.0] * 3 and np.isnan(out["B"].iloc[0]) and out["B"].iloc[1] == 7.0


# ---------------------------------------------------------------- panel, leakage, dev-only
def _store_all(cfg, rows, ids=None):
    for sid in (ids or cfg["fred"]["series"]):
        store(cfg, sid, rows)


BASE = [("2024-10-28", 1.0), ("2024-10-29", 1.1), ("2024-10-30", 1.2), ("2024-10-31", 1.3), ("2024-11-01", 1.4)]


def test_load_fred_dev_only_default(cfg):
    store(cfg, "DFF", BASE + [("2024-11-06", 2.0), ("2024-11-07", 3.0), ("2025-02-03", 4.0)])
    dev = M.load_fred("DFF", cfg)
    assert dev.index.max() == pd.Timestamp("2024-11-06")
    full = M.load_fred("DFF", cfg, include_holdout=True)
    assert full.index.max() == pd.Timestamp("2025-02-03") and len(full) == len(dev) + 2
    _store_all(cfg, BASE, ids=[k for k in cfg["fred"]["series"] if k != "DFF"])
    assert set(M.load_fred(cfg=cfg)) == set(cfg["fred"]["series"])


def test_cutoff_boundary_not_visible_to_dev_rows(cfg):
    # observation dated exactly at the cutoff (Wed 2024-11-06) becomes available Thu 2024-11-07
    store(cfg, "DFF", BASE + [("2024-11-06", 9.0)])
    idx = pd.bdate_range("2024-11-01", "2024-11-06")        # dev target dates only
    out = M.macro_panel(idx, cfg, series=["DFF"])
    assert 9.0 not in out["DFF"].values
    assert out["DFF"].iloc[-1] == 1.4
    nxt = M.macro_panel(pd.DatetimeIndex(["2024-11-07"]), cfg, series=["DFF"])
    assert nxt["DFF"].iloc[0] == 9.0                         # visible only from the next session


def test_macro_panel_ignores_holdout_unless_asked(cfg):
    store(cfg, "DFF", BASE + [("2025-01-02", 8.0)])
    idx = pd.DatetimeIndex(["2025-01-06"])
    assert M.macro_panel(idx, cfg, series=["DFF"])["DFF"].iloc[0] == 1.4
    assert M.macro_panel(idx, cfg, include_holdout=True, series=["DFF"])["DFF"].iloc[0] == 8.0


def test_panel_leakage_future_mutation_changes_nothing_earlier(cfg):
    idx = pd.date_range("2024-10-28", "2024-11-12", freq="D")
    rows = BASE + [("2024-11-04", 1.5), ("2024-11-05", 1.6)]
    store(cfg, "DFF", rows)
    a = M.macro_panel(idx, cfg, include_holdout=True, series=["DFF"])
    # mutate a value and append future observations
    store(cfg, "DFF", rows[:-1] + [("2024-11-05", 99.0), ("2024-11-08", 5.0), ("2024-11-11", 6.0)])
    b = M.macro_panel(idx, cfg, include_holdout=True, series=["DFF"])
    # 2024-11-05 obs is first visible 2024-11-06; everything before that must be identical
    cut = pd.Timestamp("2024-11-06")
    pd.testing.assert_frame_equal(a[a.index < cut], b[b.index < cut])
    assert not a.equals(b)


def test_panel_all_six_columns(cfg):
    _store_all(cfg, BASE)
    out = M.macro_panel(pd.bdate_range("2024-11-01", "2024-11-05"), cfg)
    assert list(out.columns) == list(cfg["fred"]["series"])


# ---------------------------------------------------------------- helpers
def test_risk_free_and_borrow(cfg):
    store(cfg, "DTB3", [("2024-10-31", 5.04), ("2024-11-01", 5.04)])
    store(cfg, "DFF", [("2024-10-31", 4.58), ("2024-11-01", 4.58)])
    idx = pd.DatetimeIndex(["2024-11-04"])
    rf252 = M.risk_free_daily(idx, cfg, 252)
    assert rf252.iloc[0] == pytest.approx(0.0504 / 252)
    assert M.risk_free_daily(idx, cfg, 365).iloc[0] == pytest.approx(0.0504 / 365)
    assert M.borrow_rate_annual(idx, cfg).iloc[0] == pytest.approx(0.0458 + 0.015)
    with pytest.raises(ValueError):
        M.risk_free_daily(idx, cfg, 360)


def test_risk_free_causal(cfg):
    store(cfg, "DTB3", [("2024-10-31", 5.0), ("2024-11-01", 6.0)])
    idx = pd.DatetimeIndex(["2024-11-01", "2024-11-02"])      # Fri, Sat
    rf = M.risk_free_daily(idx, cfg, 365)
    assert rf.iloc[0] == pytest.approx(0.05 / 365) and rf.iloc[1] == pytest.approx(0.05 / 365)


def _regime_data(last_hy=9.0, t=0.5):
    idx = pd.bdate_range("2020-01-01", periods=400)
    hy = pd.Series(np.linspace(3, 6, 400), index=idx)
    hy.iloc[-1] = last_hy
    return {"T10Y2Y": pd.Series(t, index=idx), "BAMLH0A0HYM2": hy}


def test_regime_context_text(cfg):
    d = _regime_data()
    end = d["T10Y2Y"].index[-1] + pd.Timedelta(days=3)
    txt = " ".join(M.regime_context(end, cfg, data=d))
    assert "not inverted" in txt and "high" in txt
    d2 = _regime_data(t=-0.4)
    assert "is inverted" in " ".join(M.regime_context(end, cfg, data=d2))


def test_regime_context_uses_only_known_data(cfg):
    d = _regime_data()
    date = d["T10Y2Y"].index[-3]
    before = M.regime_context(date, cfg, data=d)
    d2 = {k: v.copy() for k, v in d.items()}
    for v in d2.values():
        v.iloc[-2:] = -4.0 if v is d2["T10Y2Y"] else 29.0   # change data after `date`
    d2["T10Y2Y"].iloc[-2:] = -4.0
    assert M.regime_context(date, cfg, data=d2) == before
    # the observation dated the day before `date` is not yet known either (available only after)
    d3 = {k: v.copy() for k, v in d.items()}
    d3["T10Y2Y"].loc[d["T10Y2Y"].index[-4]] = -1.0           # dated previous session -> visible on `date`
    assert M.regime_context(date, cfg, data=d3) != before


def test_regime_context_short_history_skips_percentile(cfg):
    idx = pd.bdate_range("2024-01-01", periods=10)
    out = M.regime_context(idx[-1] + pd.Timedelta(days=1), cfg,
                           data={"T10Y2Y": pd.Series(0.3, index=idx), "BAMLH0A0HYM2": pd.Series(3.0, index=idx)})
    assert len(out) == 1 and "yield curve" in out[0]


# ---------------------------------------------------------------- CLI
def test_main_error_exit(monkeypatch, capsys):
    monkeypatch.setattr(M, "update_all", lambda cfg=None: (_ for _ in ()).throw(M.FredDownloadError("DFF: boom")))
    assert M.main() == 1
    assert capsys.readouterr().out.startswith("ERROR: DFF: boom")
