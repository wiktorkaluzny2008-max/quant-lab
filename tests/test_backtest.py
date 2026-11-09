"""Tests for quant.backtest (synthetic data, no network)."""
import json
import sys

import numpy as np
import pandas as pd
import pytest

from quant import backtest as B
from quant import features as F
from quant import metrics as M
from quant import models as MD
from quant import trials as T

FAST = dict(spa_reps=100, n_random=100, importance_repeats=1)
BT = B.BTConfig(min_train_days=300, min_cal=50, rolling_window=250, rolling_min_cal=40,
                adaptive_min_days=60, n_random=30, importance_repeats=1, spa_reps=50,
                holdout_months=6)


def _lgbm_available():
    try:
        import lightgbm  # noqa: F401
        return True
    except (OSError, ImportError):
        return False


needs_lgbm = pytest.mark.skipif(not _lgbm_available(), reason="lightgbm cannot load (libomp missing)")


def make_df(n=800, seed=0, start="2010-01-04", freq="B"):
    """Dataset-shaped frame: features, target, next_return; last row has NaN labels."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=n, freq=freq, name="Date")
    X = pd.DataFrame({"ret_lag_1": rng.normal(size=n), "ret_lag_2": rng.normal(size=n),
                      "vol_10": rng.normal(size=n), "vol_20": rng.normal(size=n),
                      "rsi_14": rng.normal(size=n)}, index=idx)
    nr = 0.0004 + 0.008 * rng.normal(size=n) + 0.002 * X["ret_lag_1"]
    df = X.copy()
    df["target"] = (nr > 0).astype(float)
    df["next_return"] = nr
    df.iloc[-1, df.columns.get_loc("target")] = np.nan
    df.iloc[-1, df.columns.get_loc("next_return")] = np.nan
    return df


def garbage(part, seed):
    """Replace rows by nonsense (valid 0/1 labels, wild features/returns)."""
    rng = np.random.default_rng(seed)
    g = part.copy()
    g.iloc[:, :] = rng.normal(size=g.shape) * 20 + 30     # nonzero mean: shifts any future-based median
    g["target"] = rng.integers(0, 2, len(g)).astype(float)
    g["next_return"] = rng.normal(0, 0.1, len(g))
    return g


@pytest.fixture(scope="module")
def small_rf():
    mp = pytest.MonkeyPatch()
    s = dict(MD.SETTINGS["rf"], n_estimators=15)
    mp.setitem(MD.SETTINGS, "rf", s)
    yield
    mp.undo()


@pytest.fixture(scope="module")
def dev_cut(small_rf):
    df = make_df(800)
    dev, hold = B.split_holdout(df, BT.holdout_months)
    return df, dev, B.holdout_cutoff(df, BT.holdout_months), hold


@pytest.fixture(scope="module")
def run_rf(dev_cut):
    _, dev, cutoff, _ = dev_cut
    return B.run_asset("SPY", dev, cutoff, ["logreg", "rf"], BT, 0.0005)


# ------------------------------------------------------------------ hold-out
def test_split_holdout_cutoff_and_label_blanking(dev_cut):
    df, dev, cutoff, hold = dev_cut
    assert cutoff == df.index.max() - pd.DateOffset(months=6)
    assert dev.index.max() <= cutoff < hold.index.min()
    assert len(dev) + len(hold) == len(df)
    assert np.isnan(dev["target"].iloc[-1]) and np.isnan(dev["next_return"].iloc[-1])
    assert dev[["target", "next_return"]].iloc[:-1].notna().all().all()
    # the original frame is not modified
    assert df.loc[dev.index[-1], "next_return"] == df.loc[dev.index[-1], "next_return"]


def test_holdout_garbage_changes_nothing_and_no_late_dates(dev_cut, small_rf):
    df, dev, cutoff, _ = dev_cut
    res1 = B.run_asset("SPY", dev, cutoff, ["logreg"], BT, 0.0005, importance=False)
    g = df.copy()
    late = g.index > cutoff
    g.loc[late, :] = garbage(g.loc[late], 99)
    dev_g, _ = B.split_holdout(g, BT.holdout_months)
    pd.testing.assert_frame_equal(dev, dev_g)
    res2 = B.run_asset("SPY", dev_g, cutoff, ["logreg"], BT, 0.0005, importance=False)
    pd.testing.assert_frame_equal(res1.trial_returns, res2.trial_returns)
    pd.testing.assert_frame_equal(res1.preds, res2.preds)
    pd.testing.assert_frame_equal(res1.trading_metrics, res2.trading_metrics)
    for frame in (res1.preds, res1.trial_returns, res1.baselines):
        assert frame["date"].max() <= cutoff
    assert res1.preds["date"].max() < dev.index[-1]   # the blanked last row is never scored


def test_run_functions_reject_holdout_dates(dev_cut):
    df, dev, cutoff, _ = dev_cut
    with pytest.raises(AssertionError):
        B.run_asset("SPY", df, cutoff, ["logreg"], BT, 0.0005)
    with pytest.raises(AssertionError):
        B.walk_forward(df, ["logreg"], BT, cutoff)


def test_load_dev_never_returns_holdout(monkeypatch):
    df = make_df(500)
    monkeypatch.setattr(B.F, "load_dataset", lambda t, c=None: df)
    pinned = df.index[-100]
    dev, cutoff = B.load_dev("SPY", {"holdout_months": 6, "holdout_cutoff_date": str(pinned.date())})
    assert cutoff == pinned and dev.index.max() <= cutoff
    assert len(dev) == len(df) - 99 and np.isnan(dev["next_return"].iloc[-1])
    with pytest.raises(ValueError, match="no data"):
        B.load_dev("SPY", {"holdout_months": 6, "holdout_cutoff_date": "1990-01-01"})


# ------------------------------------------------------------------ walk-forward
def with_early_nans(df):
    """NaN in one feature for the early rows, so training windows need imputation."""
    df = df.copy()
    df.iloc[:150, df.columns.get_loc("rsi_14")] = np.nan
    return df


def test_walk_forward_truncation_and_garbage_future(small_rf):
    df = with_early_nans(make_df(700, seed=3))
    k = 450
    while df.index[k].day < 8 or df.index[k].day > 20:   # mid-month so a block crosses d
        k += 1
    d = df.index[k]
    full = B.walk_forward(df, ["logreg", "rf"], BT, df.index.max(), importance=False).preds
    trunc_df = df.iloc[:k + 1]
    trunc = B.walk_forward(trunc_df, ["logreg", "rf"], BT, trunc_df.index.max(), importance=False).preds
    garb = df.copy()
    garb.iloc[k + 1:, :] = garbage(garb.iloc[k + 1:], 1)
    garbled = B.walk_forward(garb, ["logreg", "rf"], BT, garb.index.max(), importance=False).preds
    ref = full[full["date"] <= d].reset_index(drop=True)
    assert len(ref) > 100 and ref["block"].nunique() > 3
    # d is not the last day of its month => the retrain month crosses d
    assert (full["date"] > d).any() and (full[full["date"] > d]["block"] == ref["block"].iloc[-1]).any()
    for other in (trunc, garbled):
        o = other[other["date"] <= d].reset_index(drop=True)
        pd.testing.assert_frame_equal(ref, o)


def test_run_asset_truncation_incl_adaptive_and_vol_targeting(small_rf):
    df = with_early_nans(make_df(700, seed=4))
    k = 520
    d = df.index[k]
    a = B.run_asset("SPY", df, df.index.max(), ["logreg"], BT, 0.0005, importance=False)
    garb = df.copy()
    garb.iloc[k + 1:, :] = garbage(garb.iloc[k + 1:], 5)
    b = B.run_asset("SPY", garb.iloc[:k + 1], df.index[k], ["logreg"], BT, 0.0005, importance=False)
    fa = a.trial_returns[a.trial_returns["date"] <= d].reset_index(drop=True)
    fb = b.trial_returns[b.trial_returns["date"] <= d].reset_index(drop=True)
    # dedupe can alias more streams in the shorter frame, so compare the configs both runs log
    common = sorted(set(fa["config_id"]) & set(fb["config_id"]))
    fa, fb = (f[f["config_id"].isin(common)].reset_index(drop=True) for f in (fa, fb))
    assert len(common) >= 6 and len(fa) > 1000
    pd.testing.assert_frame_equal(fa, fb)
    ba = a.baselines[a.baselines["date"] <= d].reset_index(drop=True)
    bb = b.baselines[b.baselines["date"] <= d].reset_index(drop=True)
    pd.testing.assert_frame_equal(ba, bb)


def test_one_day_gap_and_expanding_window(monkeypatch, small_rf):
    df = make_df(600, seed=7)
    seen = []
    orig = MD.fit_calibrated

    def spy(name, X, y, **kw):
        seen.append((name, X.index.copy(), np.asarray(y).copy(), kw))
        return orig(name, X, y, **kw)

    monkeypatch.setattr(B.MD, "fit_calibrated", spy)
    wf = B.walk_forward(df, ["logreg"], BT, df.index.max(), importance=False)
    pos = {d: i for i, d in enumerate(df.index)}
    assert len(seen) == len(wf.retrains)
    for (name, tr_idx, y, kw), (_, r) in zip(seen, wf.retrains.iterrows()):
        s = pos[r["block"]]
        last = tr_idx.max()
        # training rows are positions <= s-2, so their targets (realised at j+1) are
        # realised at or before row s-1, strictly before the first prediction date
        assert pos[last] <= s - 2
        assert df.index[pos[last] + 1] < r["block"]
        assert df.index[s - 1] not in tr_idx          # the row realised exactly at t is excluded
        assert tr_idx.is_monotonic_increasing and tr_idx.min() == df.index[0]   # expanding
        assert not np.isnan(y).any() and len(tr_idx) >= BT.min_train_days
        assert kw["min_cal"] == BT.min_cal
        assert r["train_end"] == tr_idx.max()


def test_monthly_retrain_boundaries_and_frozen_model(monkeypatch, small_rf):
    df = make_df(600, seed=8)
    calls = []
    orig = MD.fit_calibrated
    monkeypatch.setattr(B.MD, "fit_calibrated",
                        lambda *a, **k: (calls.append(1), orig(*a, **k))[1])
    wf = B.walk_forward(df, ["logreg"], BT, df.index.max(), importance=False)
    p = wf.preds
    assert len(calls) == len(wf.retrains) == p["block"].nunique()
    for blk, d in p.groupby("block"):
        first_of_month = df.index[(df.index.year == blk.year) & (df.index.month == blk.month)].min()
        assert blk == first_of_month                      # retrain on first row of the month
        assert (d["date"].dt.month == blk.month).all() and (d["date"].dt.year == blk.year).all()
        assert d["base_rate"].nunique() == 1             # one frozen model/base rate per block
        assert d["date"].min() >= blk
    # every month after the first trainable one is a block (no month skipped)
    months = p["date"].dt.to_period("M").unique()
    assert len(months) == p["block"].nunique()
    assert set(p.columns) >= {"date", "model", "p_up", "base_rate", "y", "next_return"}
    assert p["p_up"].between(0, 1).all()


def test_base_rate_is_training_up_frequency(small_rf):
    df = make_df(600, seed=9)
    wf = B.walk_forward(df, ["logreg"], BT, df.index.max(), importance=False)
    pos = {d: i for i, d in enumerate(df.index)}
    for _, r in wf.retrains.iterrows():
        s = pos[r["block"]]
        assert r["base_rate"] == pytest.approx(df["target"].iloc[:s - 1].mean())
        assert r["n_train"] == s - 1


def test_rolling_window_btc_variant(small_rf):
    df = make_df(700, seed=10, freq="D")
    res = B.run_asset(F.BTC, df, df.index.max(), ["logreg"], BT, 0.0015, importance=False)
    assert set(res.preds["window"]) == {"expanding", "rolling4y"}
    assert set(res.trial_index["window"]) == {"expanding", "rolling4y"}
    assert T.all_logged_config_ids(res.trial_index) == set(
        T.expected_config_ids(F.BTC, ["expanding", "rolling4y"], ["logreg"]))
    rt = res.retrains[res.retrains["window"] == "rolling4y"]
    assert (rt["n_train"] == BT.rolling_window).all()      # fixed-length window
    ex = res.retrains[res.retrains["window"] == "expanding"]
    assert ex["n_train"].is_monotonic_increasing
    # rolling variant starts earlier than the expanding one needs 300 rows? no: 250 < 300
    assert res.preds[res.preds["window"] == "rolling4y"]["date"].min() < \
        res.preds[res.preds["window"] == "expanding"]["date"].min()
    # ETFs have no rolling variant
    spy = B.run_asset("SPY", make_df(500, seed=1), make_df(500, seed=1).index.max(), ["logreg"], BT,
                      0.0005, importance=False)
    assert set(spy.trial_index["window"]) == {"expanding"}


def test_min_train_days_respected(small_rf):
    df = make_df(500, seed=11)
    wf = B.walk_forward(df, ["logreg"], BT, df.index.max(), importance=False)
    assert wf.retrains["n_train"].min() >= BT.min_train_days
    with pytest.raises(ValueError):
        B.walk_forward(df.iloc[:200], ["logreg"], BT, df.index.max(), importance=False)


# ------------------------------------------------------------------ strategies
def test_threshold_logic_by_hand():
    p = np.array([0.50, 0.535, 0.54, 0.56, 0.52])
    base = np.full(5, 0.54)
    np.testing.assert_array_equal(B.long_flat_signal(p, base, 0.0), [0, 0, 1, 1, 0])
    np.testing.assert_array_equal(B.long_flat_signal(p, base, 0.01), [0, 1, 1, 1, 0])
    np.testing.assert_array_equal(B.long_flat_signal(p, base, 0.02), [0, 1, 1, 1, 1])
    np.testing.assert_array_equal(B.default_long_signal(p, base, 0.03), [0, 1, 1, 1, 1])
    np.testing.assert_array_equal(B.default_long_signal(p, base, 0.05), [1, 1, 1, 1, 1])
    np.testing.assert_array_equal(B.default_long_signal(p, base, 0.01), [0, 1, 1, 1, 0])
    # thresholds are relative to the (per-row) training base rate, not absolute
    np.testing.assert_array_equal(B.long_flat_signal(np.array([0.5, 0.5]), np.array([0.49, 0.52]), 0.0), [1, 0])
    # grids from the spec
    assert T.M_GRID["long_flat"] == (0.0, 0.005, 0.01, 0.02)
    assert T.M_GRID["default_long"] == (0.01, 0.02, 0.03)
    assert T.M_FALLBACK == {"long_flat": 0.0, "default_long": 0.02}


def test_fixed_vol_targeted_position_is_signal_times_weight():
    p = np.array([0.6, 0.4, 0.6]); base = np.full(3, 0.5); w = np.array([0.5, 1.0, 0.25])
    np.testing.assert_allclose(B._fixed_position("long_flat", p, base, 0.0, w), [0.5, 0.0, 0.25])


def _adaptive_case(n_first=20, extra=None):
    """Two blocks: in block 1 days alternate p=.485 (return +1%) / p=.51 (return 0)."""
    n = 2 * n_first
    p = np.tile([0.485, 0.51], n // 2)
    base = np.full(n, 0.5)
    nr = np.tile([0.01, 0.0], n // 2)
    if extra is not None:
        p, nr = p.copy(), nr.copy()
        p[n_first:], nr[n_first:] = extra
    return p, base, nr, [0, n_first]


def test_adaptive_m_by_hand_and_fallback():
    p, base, nr, starts = _adaptive_case()
    pos, log = B.adaptive_positions("long_flat", p, base, nr, None, starts, 0.0, min_days=10)
    assert log[0]["used_fallback"] and log[0]["m"] == 0.0 and log[0]["n_past_days"] == 0
    # only m=0.02 (threshold .48) is long on the +1% days; the others have zero variance
    assert not log[1]["used_fallback"] and log[1]["m"] == 0.02
    np.testing.assert_array_equal(pos[20:], 1.0)
    np.testing.assert_array_equal(pos[:20], B.long_flat_signal(p[:20], base[:20], 0.0))
    # until >= min_days past days exist the fallback is used
    _, log2 = B.adaptive_positions("long_flat", p, base, nr, None, starts, 0.0, min_days=21)
    assert log2[1]["used_fallback"] and log2[1]["m"] == 0.0
    _, log3 = B.adaptive_positions("default_long", p, base, nr, None, starts, 0.0, min_days=21)
    assert log3[1]["m"] == T.M_FALLBACK["default_long"]


def test_adaptive_m_uses_only_past_data():
    n_first = 40
    p, base, nr, starts = _adaptive_case(n_first=n_first)
    starts = [0, 20, 40, 60]
    base_log = B.adaptive_positions("long_flat", p, base, nr, None, starts, 0.0005, 5)[1]
    # change everything from block 3 (row 60) onwards: earlier choices (rows 0, 20, 40, 60) stay
    rng = np.random.default_rng(0)
    p2, nr2 = p.copy(), nr.copy()
    p2[60:] = rng.uniform(0, 1, 20)
    nr2[60:] = rng.normal(0, 0.05, 20)
    log2 = B.adaptive_positions("long_flat", p2, base, nr2, None, starts, 0.0005, 5)[1]
    assert [l["m"] for l in base_log] == [l["m"] for l in log2]    # the choice AT row 60 used rows < 60
    # but changing a PAST row (row 10) can change later choices
    p3 = p.copy(); nr3 = nr.copy(); nr3[:20] = -0.02
    log3 = B.adaptive_positions("long_flat", p3, base, nr3, None, starts, 0.0005, 5)[1]
    assert [l["m"] for l in log3] != [l["m"] for l in base_log]
    # changing the last block's data never changes any earlier block's position
    pos_a = B.adaptive_positions("long_flat", p, base, nr, None, starts, 0.0005, 5)[0]
    pos_b = B.adaptive_positions("long_flat", p2, base, nr2, None, starts, 0.0005, 5)[0]
    np.testing.assert_array_equal(pos_a[:60], pos_b[:60])


def test_adaptive_logged_per_month_in_run(run_rf):
    am = run_rf.adaptive_m
    assert {"config_id", "block", "m", "used_fallback", "n_past_days"} <= set(am.columns)
    n_blocks = run_rf.preds["block"].nunique()
    assert len(am) == n_blocks * 3 * 2 * 2     # models (logreg, rf, ensemble) x modes x vol-targeted
    first = am[am["block"] == am["block"].min()]
    assert first["used_fallback"].all()
    assert set(am[am["mode"] == "long_flat"]["m"]) <= set(T.M_GRID["long_flat"])
    assert set(am[am["mode"] == "default_long"]["m"]) <= set(T.M_GRID["default_long"])


# ------------------------------------------------------------------ costs
def test_costs_hand_computed_incl_delay_and_break_even():
    nr = np.array([0.01, -0.02, 0.03, 0.01])
    pos = np.array([1.0, 1.0, 0.0, 1.0])
    r = B.evaluate(nr, pos, 0.001, 252)
    net = np.array([0.009, -0.02, -0.001, 0.009])        # trades: 1,0,1,1
    assert r["mean_net_daily"] == pytest.approx(net.mean())
    assert r["n_trades"] == 3 and r["turnover"] == pytest.approx(0.75)
    assert r["pct_invested"] == 0.75
    assert r["total_return"] == pytest.approx(np.prod(1 + net) - 1)
    assert r["sharpe"] == pytest.approx(M.sharpe(net, 252))
    d = B.evaluate(nr, pos, 0.001, 252, delay=True)     # held from t+1: [0,1,1,0]
    dnet = np.array([0.0, -0.02 - 0.001, 0.03, -0.001])  # trades 0,1,0,1
    assert d["mean_net_daily"] == pytest.approx(dnet.mean())
    np.testing.assert_array_equal(B.apply_delay(pos), [0, 1, 1, 0])
    # break-even: gross mean / mean turnover
    nr2 = np.array([0.02, 0.01, 0.03, -0.01]); pos2 = np.array([1.0, 0.0, 1.0, 1.0])
    r2 = B.evaluate(nr2, pos2, 0.0, 252)
    assert r2["break_even_cost"] == pytest.approx(0.01 / 0.75)
    # at the break-even cost the net mean is zero
    r3 = B.evaluate(nr2, pos2, r2["break_even_cost"], 252)
    assert r3["mean_net_daily"] == pytest.approx(0.0, abs=1e-15)


def test_annualisation_sqrt365_btc_vs_sqrt252():
    assert M.periods_per_year("BTC-USD") == 365 and M.periods_per_year("SPY") == 252
    nr = np.random.default_rng(0).normal(0.001, 0.01, 100); pos = np.ones(100)
    a = B.evaluate(nr, pos, 0.0, 365)["sharpe"]; b = B.evaluate(nr, pos, 0.0, 252)["sharpe"]
    assert a / b == pytest.approx(np.sqrt(365 / 252))


def test_cost_scenarios():
    s = B.cost_scenarios("SPY", 0.0005)
    assert s == {"headline": (0.0005, False), "low_cost": (0.0001, False), "delay_1d": (0.0005, True)}
    b = B.cost_scenarios(F.BTC, 0.0015)
    assert b == {"headline": (0.0015, False), "stress_cost": (0.006, False), "delay_1d": (0.0015, True)}


# ------------------------------------------------------------------ baselines / vol targeting
def test_vol_weights_capped_and_no_future_vol():
    df = make_df(400, seed=2)
    retr = pd.DataFrame({"block": [df.index[100]], "sigma_target": [0.13]})
    oos = df.index[100:300]
    w = B.vol_weights(df, retr, oos, 252.0, 21)
    assert (w <= 1.0).all() and (w >= 0).all()
    assert (w < 1).any() and (w == 1).any()
    # by hand for one date
    realised = df["next_return"].shift(1)
    t = oos[50]
    sig = realised.loc[:t].iloc[-21:].std(ddof=1) * np.sqrt(252)
    assert w.loc[t] == pytest.approx(min(1.0, 0.13 / sig))
    # garbage after d: weights up to d unchanged
    d = oos[120]
    g = df.copy()
    g.loc[g.index > d, "next_return"] = 0.5
    w2 = B.vol_weights(g, retr, oos, 252.0, 21)
    np.testing.assert_array_equal(w.loc[:d].to_numpy(), w2.loc[:d].to_numpy())
    # sigma_hat uses return through t only: changing next_return AT t (the future move t->t+1)
    # cannot change w_t
    g2 = df.copy(); g2.loc[t, "next_return"] = 0.9
    assert B.vol_weights(g2, retr, oos, 252.0, 21).loc[t] == w.loc[t]


def test_baselines_present_and_always_up_equals_buy_and_hold(run_rf):
    b = run_rf.baselines
    assert set(b["name"]) == set(B.BASELINES)
    bh = b[b["name"] == "buy_and_hold"].reset_index(drop=True)
    au = b[b["name"] == "always_up"].reset_index(drop=True)
    pd.testing.assert_series_equal(bh["net_return"], au["net_return"], check_names=False)
    assert (bh["position"] == 1).all()
    assert (b[b["name"] == "vol_targeted_bh"]["position"] <= 1).all()
    assert set(b[b["name"] == "trend_200"]["position"]) <= {0.0, 1.0}
    tm = run_rf.trading_metrics
    assert set(tm[tm["kind"] == "baseline"]["name"]) == set(B.BASELINES)
    # climatology is a probability-metrics baseline
    pm = run_rf.prediction_metrics
    assert "climatology" in set(pm["model"])
    assert pm[pm["model"] == "climatology"]["brier_skill"].iloc[0] == 0.0


def test_trend_uses_only_past_prices():
    df = make_df(500, seed=12)
    pos = MD.trend_positions(B.rebased_close(df), 200)
    df2 = df.copy(); df2.iloc[400:, df2.columns.get_loc("next_return")] = 0.3
    pos2 = MD.trend_positions(B.rebased_close(df2), 200)
    # close at row j depends on next_return rows < j, so rows <= 400 unchanged
    np.testing.assert_array_equal(pos.iloc[:401].to_numpy(), pos2.iloc[:401].to_numpy())


# ------------------------------------------------------------------ trial log / results
def test_trial_log_complete_unique_and_dsr_count(run_rf, tmp_path):
    exp = set(T.expected_config_ids("SPY", ["expanding"], ["logreg", "rf", "ensemble"]))
    idx = run_rf.trial_index
    assert T.all_logged_config_ids(idx) == exp and idx["config_id"].is_unique
    assert set(run_rf.trial_returns["config_id"]) == set(idx["config_id"])
    assert len(idx) < len(exp)          # deduplicated
    assert list(run_rf.trial_returns.columns) == T.RETURN_COLS
    assert (idx["n_oos_days"] == run_rf.trial_returns.groupby("config_id").size()
            .reindex(idx["config_id"]).to_numpy()).all()
    s0 = json.loads(idx.loc[idx["model"] == "logreg", "model_settings"].iloc[0])
    assert s0 == json.loads(json.dumps(MD.model_settings("logreg"), default=str))
    assert set(json.loads(idx.loc[idx["model"] == "ensemble", "model_settings"].iloc[0])) == {"logreg", "rf"}
    B.run_and_save([run_rf], tmp_path)
    n = T.count_trials(tmp_path)
    assert n == len(idx)
    sig = pd.read_parquet(tmp_path / "SPY" / "significance.parquet")
    dsr = sig[sig["test"] == "dsr"]
    g = dsr[dsr["reference"] == "global"]
    assert len(g) == n and len(dsr) == 2 * n
    assert all(json.loads(x)["n_trials"] == n and json.loads(x)["scope"] == "global" for x in g["details"])
    a = dsr[dsr["reference"] == "asset"]
    assert all(json.loads(x)["scope"] == "asset" and json.loads(x)["n_trials"] == len(idx)
               and "sr0" in json.loads(x) and "sr_var" in json.loads(x) for x in a["details"])
    assert len(T.trial_sharpes(tmp_path)) == n


def test_result_files_columns_and_content(run_rf, tmp_path):
    B.run_and_save([run_rf], tmp_path)
    d = tmp_path / "SPY"
    for f in ["predictions", "prediction_metrics", "prediction_reliability", "prediction_deciles",
              "trading_metrics", "significance", "importance", "baselines", "adaptive_m", "retrains"]:
        assert (d / f"{f}.parquet").exists(), f
    pr = pd.read_parquet(d / "predictions.parquet")
    assert {"date", "model", "p_up", "base_rate", "y", "next_return", "window"} <= set(pr.columns)
    pm = pd.read_parquet(d / "prediction_metrics.parquet")
    assert {"auc", "brier_skill", "log_loss_skill", "accuracy", "base_rate_oos", "pt_pvalue",
            "share_up_days", "share_up_predictions"} <= set(pm.columns)
    assert {"logreg", "rf", "ensemble", "climatology"} == set(pm["model"])
    rel = pd.read_parquet(d / "prediction_reliability.parquet")
    assert {"mean_p", "obs_freq", "count"} <= set(rel.columns)
    tm = pd.read_parquet(d / "trading_metrics.parquet")
    assert {"sharpe", "cagr", "max_drawdown", "break_even_cost", "turnover", "scenario"} <= set(tm.columns)
    assert set(tm["scenario"]) == {"headline", "low_cost", "delay_1d"}
    assert len(tm[tm["kind"] == "trial"]) == 3 * len(tm[tm["kind"] == "trial"]["name"].unique())
    sig = pd.read_parquet(d / "significance.parquet")
    assert {"random_percentile", "psr", "dsr", "tstat_vs_3", "diebold_mariano", "spa", "psr_excess_vt",
            "psr_excess_bh", "dsr_excess_vt", "dsr_excess_bh"} == set(sig["test"])
    ts = sig[sig["test"] == "tstat_vs_3"]
    base_t = ts[ts["details"].map(lambda x: json.loads(x).get("baseline", False))]
    assert set(base_t["subject"]) == {"buy_and_hold", "vol_targeted_bh"}
    rs = pd.read_parquet(d / "random_strategies.parquet")
    assert set(rs["config_id"]) == set(pd.read_parquet(tmp_path / "trials_index.parquet")["config_id"])
    assert rs.groupby("config_id").size().eq(BT.n_random).all()
    dm = sig[sig["test"] == "diebold_mariano"]
    assert {("rf", "logreg"), ("rf", "climatology"), ("logreg", "climatology"),
            ("ensemble", "logreg"), ("ensemble", "climatology")} == set(zip(dm["subject"], dm["reference"]))
    spa = sig[sig["test"] == "spa"]
    assert set(spa["reference"]) == {"buy_and_hold", "vol_targeted_bh"} and len(spa) == 2
    assert spa["pvalue"].between(0, 1).all()
    assert {"call_rate", "mean_p", "mean_base_rate", "brier_skill_vs_eval_base", "call_rate_flag",
            "note"} <= set(pm.columns)
    assert {"call_rate", "call_rate_flag"} <= set(tm.columns)
    imp = pd.read_parquet(d / "importance.parquet")
    assert {"group", "importance_mean", "importance_se", "model", "n_blocks"} <= set(imp.columns)
    assert {"returns", "volatility", "trend"} == set(imp["group"])
    meta = json.loads((d / "metadata.json").read_text())
    assert meta["holdout_cutoff_date"] == str(run_rf.cutoff.date())
    assert len(meta["config_hash"]) == 64 and meta["windows"]["expanding"]["n_retrains"] > 3
    assert max(pr["date"]) <= run_rf.cutoff


def test_prediction_metrics_hand_values(run_rf):
    pm = run_rf.prediction_metrics.set_index(["window", "model"])
    p = run_rf.preds[(run_rf.preds["model"] == "logreg")]
    y, pp, base = p["y"].to_numpy(), p["p_up"].to_numpy(), p["base_rate"].to_numpy()
    row = pm.loc[("expanding", "logreg")]
    assert row["brier_skill"] == pytest.approx(1 - np.mean((pp - y) ** 2) / np.mean((base - y) ** 2))
    assert row["auc"] == pytest.approx(M.roc_auc(y, pp))
    assert row["share_up_days"] == pytest.approx(y.mean())
    assert row["share_up_predictions"] == pytest.approx(np.mean(pp >= base))


# ------------------------------------------------------------------ requirements / CLI
def test_lgbm_unavailable_fails_clearly(monkeypatch):
    monkeypatch.setitem(sys.modules, "lightgbm", None)
    with pytest.raises(RuntimeError, match="lgbm"):
        B.require_models(["logreg", "lgbm"])
    B.require_models(["logreg", "rf"])
    with pytest.raises(ValueError):
        B.require_models(["svm"])


@needs_lgbm
def test_lgbm_runs(small_rf):
    df = make_df(500, seed=13)
    wf = B.walk_forward(df, ["logreg", "lgbm"], BT, df.index.max(), importance=False)
    assert set(wf.preds["model"]) == {"logreg", "lgbm", "ensemble"}


def _cfg_file(tmp_path, cutoff, **over):
    import yaml
    from quant.config import load_config
    cfg = {k: v for k, v in load_config().items() if k != "project_root"}
    cfg["data_dir"] = "data"
    cfg["holdout_cutoff_date"] = str(pd.Timestamp(cutoff).date())
    cfg.update(over)
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.safe_dump(cfg, default_flow_style=False))
    return p


def test_cli_errors_are_clean_not_tracebacks(tmp_path, capsys, monkeypatch):
    # nonexistent config
    assert B.main(["--config", str(tmp_path / "nope.yaml")]) == 2
    assert capsys.readouterr().err.startswith("ERROR:")
    # bad config (missing keys)
    bad = tmp_path / "bad.yaml"
    bad.write_text("assets: []\n")
    assert B.main(["--config", str(bad)]) == 2
    assert "ERROR:" in capsys.readouterr().err
    # missing features file
    cfgp = _cfg_file(tmp_path, "2024-11-06", data_dir=str(tmp_path / "empty"))
    assert B.main(["--assets", "SPY", "--models", "logreg", "--config", str(cfgp),
                   "--results-dir", str(tmp_path / "o")]) == 2
    err = capsys.readouterr().err
    assert err.startswith("ERROR:") and "quant.features" in err and "Traceback" not in err


def test_malformed_yaml_clean_error_for_both_clis(tmp_path, capsys):
    from quant import leverage as L
    from quant.config import load_config
    bad = tmp_path / "bad.yaml"
    bad.write_text("foo: [\n")
    with pytest.raises(ValueError, match="not valid YAML"):
        load_config(bad)
    assert B.main(["--config", str(bad)]) == 2
    err = capsys.readouterr().err
    assert err.startswith("ERROR:") and "Traceback" not in err
    with pytest.raises(SystemExit) as ei:
        L.main(["--config", str(bad)])
    assert ei.value.code == 2
    err = capsys.readouterr().err
    assert err.startswith("ERROR:") and "Traceback" not in err


def test_lightgbm_any_import_failure_is_clean_error(monkeypatch, capsys):
    import builtins
    real = builtins.__import__

    def boom(name, *a, **k):
        if name == "lightgbm":
            raise RuntimeError("weird broken install")
        return real(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", boom)
    monkeypatch.delitem(sys.modules, "lightgbm", raising=False)
    with pytest.raises(RuntimeError, match="fix_lightgbm_macos"):
        B.require_models(["lgbm"])
    assert B.main(["--models", "lgbm"]) == 2
    err = capsys.readouterr().err
    assert err.startswith("ERROR:") and "fix_lightgbm_macos.sh" in err and "Traceback" not in err


def test_pinned_cutoff_ignores_appended_rows(small_rf):
    df = make_df(800, seed=31)
    cutoff = df.index[-120]
    extra = make_df(60, seed=32, start=str((df.index[-1] + pd.Timedelta(days=1)).date()))
    both = pd.concat([df, extra])
    dev1, _ = B.split_holdout(df, 24, cutoff)
    dev2, _ = B.split_holdout(both, 24, cutoff)
    pd.testing.assert_frame_equal(dev1, dev2)
    r1 = B.run_asset("SPY", dev1, cutoff, ["logreg"], BT, 0.0005, importance=False)
    r2 = B.run_asset("SPY", dev2, cutoff, ["logreg"], BT, 0.0005, importance=False)
    pd.testing.assert_frame_equal(r1.trial_returns, r2.trial_returns)
    pd.testing.assert_frame_equal(r1.preds, r2.preds)


def test_cli_main_hold_out_untouched_idempotent_and_lgbm_error(tmp_path, monkeypatch, small_rf, capsys):
    base = make_df(3400, seed=21)
    holdout_cut = B.holdout_cutoff(base, 24)
    seen = {}

    def fake_load(ticker, cfg=None):
        seen["calls"] = seen.get("calls", 0) + 1
        return seen["frame"].copy()

    monkeypatch.setattr(B.F, "load_dataset", fake_load)
    out1, out2 = tmp_path / "r1", tmp_path / "r2"
    cfgp = _cfg_file(tmp_path, holdout_cut)
    seen["frame"] = base
    assert B.main(["--assets", "SPY", "--models", "logreg", "--results-dir", str(out1), "--config", str(cfgp)], bt_overrides=FAST) == 0
    g = base.copy()
    late = g.index > holdout_cut
    g.loc[late, :] = garbage(g.loc[late], 3)
    seen["frame"] = g
    assert B.main(["--assets", "SPY", "--models", "logreg", "--results-dir", str(out2), "--config", str(cfgp)], bt_overrides=FAST) == 0
    for f in ["trials.parquet", "trials_index.parquet", "SPY/predictions.parquet",
              "SPY/trading_metrics.parquet", "SPY/prediction_metrics.parquet"]:
        a, b = pd.read_parquet(out1 / f), pd.read_parquet(out2 / f)
        pd.testing.assert_frame_equal(a, b)
    tr = pd.read_parquet(out1 / "trials.parquet")
    assert tr["date"].max() <= holdout_cut
    # idempotent re-run: identical contents, no duplicates
    before = pd.read_parquet(out1 / "trials.parquet")
    seen["frame"] = base
    B.main(["--assets", "SPY", "--models", "logreg", "--results-dir", str(out1), "--config", str(cfgp)], bt_overrides=FAST)
    after = pd.read_parquet(out1 / "trials.parquet")
    pd.testing.assert_frame_equal(before, after)
    assert not after.duplicated(["config_id", "date"]).any()
    assert T.count_trials(out1) < 18
    assert len(T.all_logged_config_ids(T.load_index(out1))) == 18
    # lgbm requested but unavailable: clear failure, exit code 2, nothing written
    monkeypatch.setitem(sys.modules, "lightgbm", None)
    assert B.main(["--assets", "SPY", "--models", "logreg", "lgbm",
                   "--results-dir", str(tmp_path / "r3"), "--config", str(cfgp)], bt_overrides=FAST) == 2
    assert "lgbm" in capsys.readouterr().err
    assert not (tmp_path / "r3").exists()


# ------------------------------------------------------------------ dedupe / raw diagnostics
def test_duplicate_streams_logged_once_with_aliases(run_rf):
    idx = run_rf.trial_index.set_index("config_id")
    # long_flat and default_long are the same rule at equal m => default_long m=0.01/0.02 are aliases
    for m in ("0.01", "0.02"):
        for vt in ("raw", "vt"):
            canon = f"SPY|expanding|logreg|long_flat|{vt}|m={m}"
            alias = f"SPY|expanding|logreg|default_long|{vt}|m={m}"
            assert canon in idx.index and alias not in idx.index
            assert alias in json.loads(idx.loc[canon, "aliases"])
    tm = run_rf.trading_metrics
    assert (tm[tm["name"] == "SPY|expanding|logreg|default_long|raw|m=0.01"]["duplicate_of"]
            == "SPY|expanding|logreg|long_flat|raw|m=0.01").all()
    tr = run_rf.trial_returns
    # every logged stream is distinct
    wide = tr.pivot(index="date", columns="config_id", values="position")
    assert not wide.T.duplicated().any()


def test_raw_predictions_slopes_and_metadata(run_rf, tmp_path):
    p = run_rf.preds
    assert p["p_raw"].between(0, 1).all() and (p["p_raw"] != p["p_up"]).any()
    rt = run_rf.retrains
    assert {"platt_slope_logreg", "platt_slope_rf"} <= set(rt.columns) and rt["platt_slope_logreg"].notna().all()
    rm = run_rf.raw_metrics.set_index("model")
    row = rm.loc["logreg"]
    d = p[p["model"] == "logreg"]
    assert row["auc"] == pytest.approx(M.roc_auc(d["y"], d["p_raw"]))
    assert {"auc", "brier_skill", "log_loss_skill", "accuracy"} <= set(rm.columns)
    B.run_and_save([run_rf], tmp_path)
    assert (tmp_path / "SPY" / "raw_prediction_metrics.parquet").exists()
    meta = json.loads((tmp_path / "SPY" / "metadata.json").read_text())
    ns = meta["negative_platt_slope_blocks"]["expanding/logreg"]
    assert ns["negative"] == int((rt["platt_slope_logreg"] < 0).sum()) and ns["blocks"] == len(rt)
    # strategies use calibrated p only: trial positions do not depend on p_raw
    assert meta["n_trials_global"] == T.count_trials(tmp_path)
