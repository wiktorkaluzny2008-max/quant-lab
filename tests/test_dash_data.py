"""Stage 5 data-access layer (quant/dash_data.py)."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quant import dash_data as DD
from quant import metrics as M
from quant import paper as P
from quant import trials as T
from quant.config import load_config
from tests import dash_world as W
from tests.synth import FAST_BT, TICKERS, make_cfg, make_results, truncate_world

REAL = load_config()
HAVE_REAL = (REAL["data_dir"] / "paper" / "ledger.parquet").exists() and (
    REAL["data_dir"] / "results" / "SPY" / "predictions.parquet").exists()
real_only = pytest.mark.skipif(not HAVE_REAL, reason="real data/results + data/paper not present")
CUT = pd.Timestamp(REAL["holdout_cutoff_date"])


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    return W.build(tmp_path_factory.mktemp("ddworld"))


def preds_frame(n=600, seed=0, edge=0.0):
    rng = np.random.default_rng(seed)
    d = pd.bdate_range("2015-01-01", periods=n)
    y = (rng.random(n) < 0.55).astype(float)
    p = np.clip(0.55 + edge * (y - 0.55) + rng.normal(0, 0.02, n), 0.05, 0.95)
    return pd.DataFrame({"date": d, "p_up": p, "base_rate": 0.55, "y": y})


# =========================================================================== helpers use metrics.py
def test_rolling_auc_matches_metrics_roc_auc_on_each_window():
    df = preds_frame(400, 1, edge=0.3)
    r = DD.rolling_auc(df, window=100)
    assert r.iloc[:99].isna().all() and r.notna().sum() == len(df) - 99
    for i in (99, 200, 399):
        sl = slice(i - 99, i + 1)
        assert r.iloc[i] == pytest.approx(M.roc_auc(df["y"].to_numpy()[sl], df["p_up"].to_numpy()[sl]))
    assert r.index.equals(pd.DatetimeIndex(df["date"]))


def test_rolling_auc_is_nan_for_single_class_windows():
    df = preds_frame(150, 2)
    df["y"] = 1.0
    assert DD.rolling_auc(df, window=50).isna().all()


def test_rolling_brier_skill_matches_metrics_brier_skill():
    df = preds_frame(500, 3, edge=0.2)
    r = DD.rolling_brier_skill(df, window=120)
    for i in (119, 300, 499):
        sl = slice(i - 119, i + 1)
        want = M.brier_skill(df["y"].to_numpy()[sl], df["p_up"].to_numpy()[sl], df["base_rate"].to_numpy()[sl])
        assert r.iloc[i] == pytest.approx(want)
    assert r.iloc[:119].isna().all()


def test_hit_rate_deciles_and_confusion_matrix():
    df = preds_frame(1000, 4, edge=0.3)
    dec = M.hit_rate_by_confidence_decile(df["p_up"], df["y"], df["base_rate"]).reset_index()
    assert list(dec["decile"]) == list(range(1, 11)) and dec["n"].sum() == 1000
    cm = DD.confusion_matrix(df)
    assert cm.to_numpy().sum() == 1000
    said_up = (df["p_up"] >= df["base_rate"]).to_numpy()
    up = (df["y"] > 0.5).to_numpy()
    assert cm.loc["market went up", "model said up"] == int((said_up & up).sum())
    assert cm.loc["market went down", "model said down"] == int((~said_up & ~up).sum())
    hit = (cm.loc["market went up", "model said up"] + cm.loc["market went down", "model said down"]) / 1000
    assert hit == pytest.approx(M.hit_rate_by_confidence_decile(df["p_up"], df["y"], df["base_rate"])
                                .eval("n * hit_rate").sum() / 1000)


def test_sharpe_band_is_plus_minus_two_standard_errors():
    b = DD.sharpe_band({"sharpe": 0.8, "sharpe_se": 0.25})
    assert (b["low"], b["high"], b["k"]) == (pytest.approx(0.3), pytest.approx(1.3), 2.0)
    assert DD.sharpe_band({"sharpe": 0.8, "sharpe_se": 0.25}, k=1)["high"] == pytest.approx(1.05)


def test_drawdown_series_matches_metrics_max_drawdown():
    r = np.random.default_rng(5).normal(0.0005, 0.01, 800)
    dd = DD.drawdown_series(r)
    assert (dd <= 1e-12).all()
    assert dd.min() == pytest.approx(M.max_drawdown(r))
    eq = M.equity_curve(r)
    assert dd.iloc[100] == pytest.approx(eq[100] / max(1.0, eq[:101].max()) - 1)


# =========================================================================== loading + hold-out guard
def test_load_asset_results_and_tables(world):
    res = DD.load_asset_results("SPY", world)
    assert "predictions" in res.tables and "baselines" in res.tables
    assert len(res.predictions) > 100 and res.get("significance", ("a",)).empty
    with pytest.raises(AttributeError):
        res.significance
    pr = DD.model_predictions(res, "ensemble")
    assert pr["date"].is_monotonic_increasing and set(pr["model"]) == {"ensemble"}


def test_missing_results_raise_a_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="python -m quant.backtest"):
        DD.load_asset_results("SPY", make_cfg(tmp_path))


def test_holdout_dates_in_any_dev_series_are_refused(tmp_path):
    cfg = make_cfg(tmp_path)
    make_results(cfg)
    f = cfg["data_dir"] / "results" / "SPY" / "predictions.parquet"
    df = pd.read_parquet(f)
    df.loc[df.index[-1], "date"] = pd.Timestamp(cfg["holdout_cutoff_date"]) + pd.Timedelta(days=1)
    df.to_parquet(f)
    with pytest.raises(P.HoldoutError, match="after the pinned hold-out cutoff"):
        DD.load_asset_results("SPY", cfg)
    DD.load_asset_results("QQQ", cfg)                       # other assets are unaffected
    with pytest.raises(P.HoldoutError):
        DD.assert_dev(pd.DataFrame({"date": [pd.Timestamp(cfg["holdout_cutoff_date"]) + pd.Timedelta(days=1)]}),
                      "date", cfg)
    DD.assert_dev(pd.DataFrame({"date": [pd.Timestamp(cfg["holdout_cutoff_date"])]}), "date", cfg)   # equal is fine


def test_metadata_with_a_holdout_date_is_refused(tmp_path):
    cfg = make_cfg(tmp_path)
    make_results(cfg)
    (cfg["data_dir"] / "results" / "SPY" / "metadata.json").write_text(json.dumps({"dev_last_date": "2030-01-01"}))
    with pytest.raises(P.HoldoutError):
        DD.load_asset_results("SPY", cfg)


@real_only
def test_nothing_after_the_cutoff_in_any_real_dev_series():
    for t in TICKERS:
        res = DD.load_asset_results(t)
        for name, col in DD.DATE_COLUMNS.items():
            if name in res.tables:
                assert pd.to_datetime(res.tables[name][col]).max() <= CUT, (t, name)
        assert pd.Timestamp(res.metadata["dev_last_date"]) <= CUT
    tr = pd.read_parquet(REAL["data_dir"] / "results" / "trials.parquet", columns=["date"])
    assert tr["date"].max() <= CUT
    # equity curves and trial returns served to the pages obey the same rule
    cid = DD.strategy_config_id("SPY")
    assert DD.equity_curves("SPY", cid).index.max() <= CUT
    assert DD.load_trial_returns(cid)["date"].max() <= CUT
    assert DD.leverage_equity_curves(DD.load_asset_results("SPY")).index.max() <= CUT


# =========================================================================== real results cross-checks
@real_only
def test_real_hit_rate_deciles_equal_the_saved_ones():
    res = DD.load_asset_results("SPY")
    pr = DD.model_predictions(res, "ensemble")
    mine = M.hit_rate_by_confidence_decile(pr["p_up"].to_numpy(float), pr["y"].to_numpy(float),
                                            pr["base_rate"].to_numpy(float)).reset_index()
    saved = res.deciles
    saved = saved[(saved["model"] == "ensemble") & (saved["window"] == "expanding")].reset_index(drop=True)
    for c in ("decile", "n", "mean_confidence", "hit_rate"):
        np.testing.assert_allclose(mine[c].to_numpy(float), saved[c].to_numpy(float))


@real_only
def test_real_rolling_metrics_end_value_consistency():
    res = DD.load_asset_results("BTC-USD")
    pr = DD.model_predictions(res, "ensemble")
    auc = DD.rolling_auc(pr, 250)
    assert auc.iloc[-1] == pytest.approx(M.roc_auc(pr["y"].to_numpy()[-250:], pr["p_up"].to_numpy()[-250:]))
    bs = DD.rolling_brier_skill(pr, 250)
    assert bs.iloc[-1] == pytest.approx(M.brier_skill(pr["y"].to_numpy()[-250:], pr["p_up"].to_numpy()[-250:],
                                                      pr["base_rate"].to_numpy()[-250:]))
    # confusion matrix hit rate equals the saved prediction metric
    cm = DD.confusion_matrix(pr)
    pm = res.prediction_metrics
    pm = pm[(pm["model"] == "ensemble") & (pm["window"] == "expanding")].iloc[0]
    assert cm.to_numpy().sum() == pm["n"]
    assert (cm.loc["market went up", "model said up"] + cm.loc["market went down", "model said down"]) / pm["n"] \
        == pytest.approx(pm["hit_rate"])


@real_only
@pytest.mark.parametrize("ticker", TICKERS)
def test_real_equity_curve_matches_saved_trading_metrics(ticker):
    res = DD.load_asset_results(ticker)
    cid = DD.strategy_config_id(ticker)
    eq = DD.equity_curves(ticker, cid)
    row = DD.trading_row(res, cid)
    assert eq["strategy"].iloc[-1] - 1 == pytest.approx(row["total_return"], rel=1e-9)
    for name in ("buy_and_hold", "vol_targeted_bh"):
        b = DD.trading_row(res, name)
        assert eq[name].iloc[-1] - 1 == pytest.approx(b["total_return"], rel=1e-9)
    assert {"dd_strategy", "dd_buy_and_hold", "dd_vol_targeted_bh"} <= set(eq.columns)
    assert eq["dd_strategy"].min() == pytest.approx(row["max_drawdown"])
    # the saved Sharpe and its standard error are reproduced from the trial's net returns
    net = DD.load_trial_returns(cid)["net_return"].to_numpy()
    ppy = M.periods_per_year(ticker)
    assert M.sharpe(net, ppy) == pytest.approx(row["sharpe"])
    assert M.sharpe_standard_error(M.sharpe(net, 1), len(net), ppy) == pytest.approx(row["sharpe_se"])
    band = DD.sharpe_band(row)
    assert band["low"] == pytest.approx(row["sharpe"] - 2 * row["sharpe_se"])


@real_only
def test_real_leverage_tables():
    res = DD.load_asset_results("SPY")
    t = DD.leverage_table(res, "chosen_strategy")
    assert list(t["L"]) == [1.0, 1.5, 2.0, 3.0] and "wiped_out" in t.columns
    le = DD.leverage_equity_curves(res, "buy_and_hold")
    assert list(le.columns) == [1.0, 1.5, 2.0, 3.0]
    k = res.kelly
    assert {"L", "g", "L_star"} <= set(k.columns)


@real_only
def test_real_metrics_table_has_every_model_and_the_baselines():
    res = DD.load_asset_results("QQQ")
    tab = DD.metrics_table(res, mode="long_flat", vol_targeted=True, m_setting="adaptive")
    assert {"logreg", "rf", "lgbm", "ensemble"} <= set(tab["model"])
    assert {"buy_and_hold", "vol_targeted_bh", "always_up", "trend_200"} <= set(tab["name"])
    cid = DD.strategy_config_id("QQQ")
    assert DD.random_percentile(res, cid) == pytest.approx(
        res.significance.query("test == 'random_percentile' and subject == @cid").value.iloc[0])


@real_only
def test_real_trials_summary():
    s = DD.trials_summary()
    assert s["n_trials_global"] == T.count_trials(REAL["data_dir"] / "results") == sum(s["per_asset"].values())
    for t in TICKERS:
        a = s["assets"][t]
        assert a["config_id"] == DD.strategy_config_id(t)
        assert 0.0 <= a["dsr_global"] <= a["dsr_asset"] + 1e-12 <= 1.0 + 1e-12   # more trials -> lower DSR
        assert a["n_trials_global"] == s["n_trials_global"] and a["n_trials_asset"] == s["per_asset"][t]


@real_only
def test_alias_config_ids_resolve_to_the_logged_trial():
    idx = DD.load_trial_index()
    row = idx[idx["aliases"].map(lambda a: len(json.loads(a)) > 0)].iloc[0]
    alias = json.loads(row["aliases"])[0]
    assert DD.canonical_config_id(idx, alias) == row["config_id"]
    assert DD.canonical_config_id(idx, row["config_id"]) == row["config_id"]
    with pytest.raises(KeyError):
        DD.canonical_config_id(idx, "SPY|nope")
    assert len(DD.load_trial_returns(alias)) == row["n_oos_days"]


# =========================================================================== ledger / freshness / refresh
def digest(root: Path):
    return {str(p.relative_to(root)): (hashlib.sha256(p.read_bytes()).hexdigest(), p.stat().st_mtime_ns)
            for p in sorted(root.rglob("*")) if p.is_file()}


def test_ledger_loading_and_freshness_are_read_only(world):
    cfg = world
    before = digest(cfg["data_dir"])
    led = DD.load_ledger(cfg)
    assert set(led["asset"]) == set(TICKERS) and DD.ledger_for("SPY", cfg)["date"].is_monotonic_increasing
    assert DD.load_state(cfg)["assets"]["SPY"]["config_id"] == DD.strategy_config_id("SPY", cfg)
    DD.data_freshness(cfg, now="2021-10-15 23:00:00")
    assert digest(cfg["data_dir"]) == before


def test_freshness_reports_stale_prices_and_ledger(world, tmp_path):
    cfg = dict(world)
    # no price files in the synthetic data dir -> "no price data stored"; ledger is current on the as-of day
    f = DD.data_freshness(cfg, now="2021-10-15 23:00:00")
    assert f["SPY"]["last_ledger_date"] == pd.Timestamp("2021-10-15")
    assert "no price data stored" in f["SPY"]["warning"]
    assert f["BTC-USD"]["expected_date"] == pd.Timestamp("2021-10-14")       # BTC: yesterday (UTC)
    # a later "now": the ledger is behind -> STALE warning that points at the Refresh button
    f2 = DD.data_freshness(cfg, now="2021-10-20 23:00:00")
    assert f2["SPY"]["stale"] and "STALE DATA" in f2["SPY"]["warning"] and "Refresh data" in f2["SPY"]["warning"]
    # real price file next to the ledger
    px = world["data_dir"]
    pd.DataFrame({"Close": [1.0, 2.0]}, index=pd.to_datetime(["2021-10-14", "2021-10-15"])).to_parquet(px / "SPY.parquet")
    assert DD.latest_price_date("SPY", cfg) == pd.Timestamp("2021-10-15")
    assert DD.data_freshness(cfg, now="2021-10-16 12:00:00")["SPY"]["warning"] is None
    (px / "SPY.parquet").unlink()


def test_refresh_data_runs_the_stage4_update_and_is_idempotent(tmp_path):
    cfg = W.build(tmp_path, asof="2021-10-14")
    P_old = P.BT
    P.BT = FAST_BT
    try:
        kw = dict(asof=pd.Timestamp("2021-10-15"), prices=truncate_world(W.PX, "2021-10-15"), update=None)
        first = DD.refresh_data(cfg, **kw)
        assert first.ok and "new paper-trading row" in first.message
        assert DD.ledger_for("SPY", cfg)["date"].max() == pd.Timestamp("2021-10-15")
        before = digest(cfg["data_dir"])
        second = DD.refresh_data(cfg, **kw)
        assert second.ok and "Already up to date" in second.message
        assert digest(cfg["data_dir"]) == before
    finally:
        P.BT = P_old


def test_refresh_data_never_returns_a_traceback(tmp_path):
    cfg = make_cfg(tmp_path)

    def boom(cfg, **kw):
        raise ZeroDivisionError("secret internals")

    def paper_error(cfg, **kw):
        raise P.PaperError("ledger settings differ")
    out = DD.refresh_data(cfg, run=boom)
    assert not out.ok and "Traceback" not in out.message and "unexpectedly" in out.message
    out = DD.refresh_data(cfg, run=paper_error)
    assert not out.ok and out.message == "Refresh failed: ledger settings differ"

    class Res:
        messages, skipped, wrote, new_rows = ["x"], {"SPY": "STALE DATA"}, False, None
    out = DD.refresh_data(cfg, run=lambda cfg, **kw: Res())
    assert not out.ok and "skipped" in out.message and "SPY" in out.message     # partial = warning, not green
    # a real run against a directory without results fails cleanly
    out = DD.refresh_data(cfg, prices={}, update=None)
    assert isinstance(out.ok, bool) and "Traceback" not in out.message


def test_refresh_data_swallows_prints_from_the_run(tmp_path, capsys):
    cfg = make_cfg(tmp_path)

    def noisy(cfg, **kw):
        print("this goes to the logs")

        class R:
            messages, skipped, wrote, new_rows = [], {}, False, None
        return R()
    out = DD.refresh_data(cfg, run=noisy)
    assert out.ok and capsys.readouterr().out == ""
