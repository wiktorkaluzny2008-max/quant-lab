"""Stage 4 ledger logic (quant/paper.py): synthetic data only, never the network."""
import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from quant import backtest as B
from quant import metrics as M
from quant import models as MD
from quant import paper as P
from quant import run_daily as R
from quant.config import load_config
from tests.synth import (CUTOFF, FAST_BT, HOLDOUT_MONTHS, TICKERS, make_cfg, make_results,
                         truncate_world, world)

PX = world()
HOLD_END = pd.Timestamp(CUTOFF) + pd.DateOffset(months=HOLDOUT_MONTHS)    # 2021-09-30


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


@pytest.fixture(scope="module")
def baseline(tmp_path_factory):
    """One shared run: start 2021-10-01, then catch up to 2021-11-12."""
    mp = pytest.MonkeyPatch()
    mp.setattr(P, "BT", FAST_BT)
    tmp = tmp_path_factory.mktemp("base")
    cfg = make_cfg(tmp)
    make_results(cfg)
    go(cfg, "2021-10-01")
    go(cfg, "2021-11-12")
    yield cfg, ledger(cfg)
    mp.undo()


# --------------------------------------------------------------------------- schema
def test_ledger_schema(baseline):
    cfg, L = baseline
    assert list(L.columns) == P.LEDGER_COLS
    for c in ("asset", "date", "config_id", "p_up", "base_rate", "position", "trade", "cost",
              "equity", "pnl", "gross_return", "net_return", "bh_equity", "vtbh_equity",
              "live_brier", "live_brier_skill", "live_hit_rate", "bt_brier", "bt_brier_skill",
              "bt_hit_rate"):
        assert c in L.columns
    assert not L.duplicated(["asset", "date"]).any()
    assert str(L["date"].dtype).startswith("datetime64")
    assert L["position"].between(0, 1).all()                       # unleveraged
    assert L["p_up"].between(0, 1).all()
    assert set(L["asset"]) == set(TICKERS)
    for t, g in L.groupby("asset"):
        assert g["config_id"].nunique() == 1
        assert g["config_id"].iloc[0].startswith(f"{t}|expanding|logreg|long_flat|vt|m=adaptive")
        assert g["date"].is_monotonic_increasing


def test_config_id_is_in_the_stage3_trial_log(baseline):
    cfg, L = baseline
    from quant import trials as T
    logged = T.all_logged_config_ids(T.load_index(cfg["data_dir"] / "results"))
    assert set(L["config_id"]) <= logged


def test_missing_trial_log_entry_is_a_clean_error(tmp_path):
    cfg = make_cfg(tmp_path)
    make_results(cfg, config_modes=("default_long",))             # long_flat ids are absent
    with pytest.raises(P.PaperError, match="not in the Stage 3 trial log"):
        go(cfg, "2021-10-01")
    assert not (tmp_path / "paper").exists()


def test_daily_rows_follow_every_trading_day_and_btc_every_calendar_day(baseline):
    cfg, L = baseline
    spy = L[L.asset == "SPY"]["date"]
    assert (spy.diff().dropna().dt.days.between(1, 4)).all()
    sessions = PX["SPY"].index
    exp = sessions[(sessions >= spy.iloc[0]) & (sessions <= spy.iloc[-1])]
    assert list(spy) == list(exp)
    btc = L[L.asset == "BTC-USD"]["date"]
    assert (btc.diff().dropna() == pd.Timedelta(days=1)).all()


# --------------------------------------------------------------------------- arithmetic
def test_mark_to_market_uses_realised_return_and_previous_position(baseline):
    cfg, L = baseline
    for t in TICKERS:
        g = L[L.asset == t].reset_index(drop=True)
        close = PX[t]["Close"].astype(float)
        # asset_return_j = close_j / close_{j-1} - 1 (the dataset returns come from this series)
        exp = close.pct_change().reindex(g["date"]).to_numpy()
        np.testing.assert_allclose(g["asset_return"].to_numpy()[1:], exp[1:], rtol=1e-9, atol=1e-12)
        assert np.isnan(g["asset_return"].iloc[0]) and g["gross_return"].iloc[0] == 0.0
        np.testing.assert_allclose(g["gross_return"].to_numpy()[1:],
                                   g["prev_position"].to_numpy()[1:] * g["asset_return"].to_numpy()[1:])
        # yesterday's intended position is today's prev_position
        np.testing.assert_allclose(g["prev_position"].to_numpy()[1:], g["position"].to_numpy()[:-1])
        assert g["prev_position"].iloc[0] == 0.0               # the ledger starts in cash


def test_cost_accounting_and_equity_chain(baseline):
    cfg, L = baseline
    for t in TICKERS:
        g = L[L.asset == t].reset_index(drop=True)
        rate = cfg["costs"][t]
        assert rate == (0.0015 if t == "BTC-USD" else 0.0005)
        np.testing.assert_allclose(g["trade"], (g["position"] - g["prev_position"]).abs())
        np.testing.assert_allclose(g["cost"], rate * g["trade"])
        np.testing.assert_allclose(g["net_return"], g["gross_return"] - g["cost"])
        eq = 10000.0 * np.cumprod(1.0 + g["net_return"].to_numpy())
        np.testing.assert_allclose(g["equity"], eq, rtol=1e-12)
        prev_eq = np.r_[10000.0, eq[:-1]]
        np.testing.assert_allclose(g["pnl"], eq - prev_eq, atol=1e-9)
        np.testing.assert_allclose(g["cum_pnl"], eq - 10000.0, atol=1e-9)
        np.testing.assert_allclose(g["cost_usd"], g["cost"] * prev_eq, atol=1e-9)
        np.testing.assert_allclose(g["cum_cost_usd"], np.cumsum(g["cost_usd"]), atol=1e-9)
        assert list(g["n_trades"]) == list(np.cumsum(g["trade"] > 1e-12))
        # entering from cash costs the same as any other trade
        assert g["cost"].iloc[0] == pytest.approx(rate * g["position"].iloc[0])


def test_buy_and_hold_and_vol_targeted_baselines(baseline):
    cfg, L = baseline
    for t in TICKERS:
        g = L[L.asset == t].reset_index(drop=True)
        rate = cfg["costs"][t]
        r = g["asset_return"].fillna(0.0).to_numpy()
        bh = 10000.0 * np.cumprod(1.0 + np.r_[0.0, r[1:]] - rate * np.r_[1.0, np.zeros(len(g) - 1)])
        np.testing.assert_allclose(g["bh_equity"], bh, rtol=1e-12)
        w = g["vtbh_position"].to_numpy()
        assert ((w >= 0) & (w <= 1)).all()
        prev_w = np.r_[0.0, w[:-1]]
        net = prev_w * np.r_[0.0, r[1:]] - rate * np.abs(w - prev_w)
        np.testing.assert_allclose(g["vtbh_equity"], 10000.0 * np.cumprod(1.0 + net), rtol=1e-12)


def test_custom_starting_capital_and_asset_costs(tmp_path):
    cfg = new_world(tmp_path, starting_capital={"SPY": 5000, "QQQ": 20000})
    go(cfg, "2021-10-08")
    L = ledger(cfg)
    caps = {"SPY": 5000.0, "QQQ": 20000.0, "IWM": 10000.0, "BTC-USD": 10000.0}
    for t, g in L.groupby("asset"):
        g = g.reset_index(drop=True)
        assert g["equity"].iloc[0] == pytest.approx(caps[t] * (1 - g["cost"].iloc[0]))


def test_signal_matches_stage3_rule(baseline):
    """long_flat: position = [p >= base - m] * min(1, vol_ratio)  (reusing backtest.SIGNALS)."""
    cfg, L = baseline
    sig = (L["p_up"] >= L["base_rate"] - L["margin_m"]).astype(float)
    np.testing.assert_allclose(L["position"], sig * np.minimum(1.0, L["vol_ratio"]), atol=1e-12)


# --------------------------------------------------------------------------- live vs backtest metrics
def test_live_metrics_equal_metrics_module_and_sit_next_to_backtest_values(baseline):
    cfg, L = baseline
    for t in TICKERS:
        g = L[L.asset == t].reset_index(drop=True)
        last = g.iloc[-1]
        p_prev, b_prev = g["p_up"].to_numpy()[:-1], g["base_rate"].to_numpy()[:-1]
        y = (g["asset_return"].to_numpy()[1:] > 0).astype(float)
        assert last["live_n"] == len(y)
        assert last["live_brier"] == pytest.approx(M.brier(y, p_prev))
        assert last["live_brier_skill"] == pytest.approx(M.brier_skill(y, p_prev, b_prev))
        assert last["live_hit_rate"] == pytest.approx(np.mean((p_prev >= b_prev) == (y > 0.5)))
        assert g["live_n"].iloc[0] == 0 and np.isnan(g["live_brier"].iloc[0])
        # backtest reference = same config's saved predictions, constant over the ledger
        pred = pd.read_parquet(cfg["data_dir"] / "results" / t / "predictions.parquet")
        assert last["bt_brier"] == pytest.approx(M.brier(pred["y"], pred["p_up"]))
        assert last["bt_brier_skill"] == pytest.approx(
            M.brier_skill(pred["y"], pred["p_up"], pred["base_rate"]))
        assert last["bt_n"] == len(pred)
        assert g["bt_brier"].nunique() == 1


# --------------------------------------------------------------------------- training window
def test_live_training_uses_only_labels_known_with_a_one_day_gap(tmp_path, monkeypatch):
    cfg = new_world(tmp_path)
    seen = []
    real = MD.fit_calibrated

    def spy(name, X, y, **kw):
        seen.append((X.index.max(), len(X), np.isnan(np.asarray(y, float)).any(), kw))
        return real(name, X, y, **kw)
    monkeypatch.setattr(MD, "fit_calibrated", spy)
    go(cfg, "2021-10-15")
    prices = R._truncate_prices(cfg, PX, pd.Timestamp("2021-10-16", tz="UTC"))
    for t in TICKERS:
        ds = R.F.build_dataset(t, cfg, prices)
        st = P.load_state(P.PaperFiles.for_config(cfg))["assets"][t]
        i = ds.index.get_loc(pd.Timestamp(st["train_date"]))
        # last training row is two rows before the decision day: row i-1's label is only realised
        # at the decision day's close, so it is left out (the walk-forward's 1-day gap)
        assert pd.Timestamp(st["train_last_row"]) == ds.index[i - 2]
        assert st["n_train"] == int(ds["target"].iloc[: i - 1].notna().sum())
    assert seen and all(not nan for _, _, nan, _ in seen)
    assert all(kw.get("min_cal") == FAST_BT.min_cal for *_, kw in seen)


def test_live_model_trains_on_all_available_rows_including_holdout_calendar(tmp_path):
    cfg = new_world(tmp_path)
    go(cfg, "2021-10-15")
    st = P.load_state(P.PaperFiles.for_config(cfg))["assets"]["SPY"]
    assert pd.Timestamp(st["train_last_row"]) > HOLD_END - pd.Timedelta(days=5)
    assert st["n_train"] > 300


def test_insufficient_training_data_skips_with_message(tmp_path, monkeypatch):
    monkeypatch.setattr(P, "BT", B.BTConfig(min_train_days=5000, min_cal=30, adaptive_min_days=60))
    cfg = new_world(tmp_path)
    r = go(cfg, "2021-10-15")
    assert set(r.skipped) == set(TICKERS)
    assert all("training rows" in v for v in r.skipped.values())
    assert not (tmp_path / "paper").exists()


# --------------------------------------------------------------------------- monthly retrain
def test_retrain_only_on_first_processed_day_of_a_new_calendar_month(tmp_path, monkeypatch):
    cfg = new_world(tmp_path)
    calls = []
    real = P.fit_live

    def counting(ds, i, settings, ticker):
        calls.append((ticker, ds.index[i]))
        return real(ds, i, settings, ticker)
    monkeypatch.setattr(P, "fit_live", counting)
    go(cfg, "2021-10-18")                       # first run: trains (Oct)
    go(cfg, "2021-10-29")                       # same month: no retrain
    spy_calls = [d for t, d in calls if t == "SPY"]
    assert spy_calls == [pd.Timestamp("2021-10-18")]
    go(cfg, "2021-11-10")                       # new month: exactly one retrain, on Nov 1 (first session)
    spy_calls = [d for t, d in calls if t == "SPY"]
    assert spy_calls == [pd.Timestamp("2021-10-18"), pd.Timestamp("2021-11-01")]
    btc = [d for t, d in calls if t == "BTC-USD"]
    assert btc == [pd.Timestamp("2021-10-17"), pd.Timestamp("2021-11-01")]
    st = P.load_state(P.PaperFiles.for_config(cfg))["assets"]["SPY"]
    assert st["train_date"] == "2021-11-01"
    # a second run the same day never retrains
    n = len(calls)
    go(cfg, "2021-11-10")
    assert len(calls) == n


def test_missing_model_file_is_refit_deterministically(tmp_path):
    cfg_a = new_world(tmp_path / "a")
    cfg_b = new_world(tmp_path / "b")
    go(cfg_a, "2021-10-18"); go(cfg_b, "2021-10-18")
    files = P.PaperFiles.for_config(cfg_b)
    for t in TICKERS:
        files.model(t).unlink()
    go(cfg_a, "2021-10-26"); go(cfg_b, "2021-10-26")
    assert_frame_equal(ledger(cfg_a), ledger(cfg_b))


# --------------------------------------------------------------------------- no lookahead
def garbage_after(px, last_session, seed=0):
    """Same history up to ``last_session`` (BTC: the day before), nonsense afterwards."""
    rng = np.random.default_rng(seed)
    out = {}
    for k, v in px.items():
        cut = pd.Timestamp(last_session)
        v = v.copy()
        late = v.index > cut
        v.loc[late, :] = rng.uniform(0.01, 1e6, size=(int(late.sum()), v.shape[1]))
        out[k] = v
    return out


@pytest.mark.parametrize("asof", ["2021-10-15", "2021-10-20"])
def test_garbage_after_the_asof_date_changes_nothing(tmp_path, asof):
    """Position (and everything else) decided for the as-of date is identical when every price
    series after that date is replaced by nonsense."""
    a = new_world(tmp_path / "clean")
    b = new_world(tmp_path / "dirty")
    clean = truncate_world(PX, asof)
    dirty = garbage_after(PX, asof)
    go(a, asof, clean)
    go(b, asof, dirty)
    La, Lb = ledger(a), ledger(b)
    assert len(La) > 0
    assert_frame_equal(La, Lb)
    assert (La["date"] <= pd.Timestamp(asof)).all()


def test_garbage_after_asof_in_the_middle_of_a_catch_up(tmp_path):
    a = new_world(tmp_path / "clean")
    b = new_world(tmp_path / "dirty")
    go(a, "2021-10-01", truncate_world(PX, "2021-10-01"))
    go(b, "2021-10-01", garbage_after(PX, "2021-10-01"))
    go(a, "2021-10-12", truncate_world(PX, "2021-10-12"))
    go(b, "2021-10-12", garbage_after(PX, "2021-10-12", seed=1))
    assert_frame_equal(ledger(a), ledger(b))


def test_process_asset_ignores_dataset_rows_after_max_date(tmp_path):
    cfg = new_world(tmp_path)
    settings = P.PaperSettings.from_config(cfg)
    prices = R._truncate_prices(cfg, PX, pd.Timestamp("2021-12-31", tz="UTC"))
    ds = R.F.build_dataset("SPY", cfg, prices)
    bt = P.load_backtest_history("SPY", settings, cfg["data_dir"] / "results", CUTOFF)
    D = pd.Timestamp("2021-10-20")
    clean = P.process_asset("SPY", ds, settings, pd.DataFrame(columns=P.LEDGER_COLS), None, bt,
                            HOLD_END, CUTOFF, max_date=D)
    wild = ds.copy()
    later = wild.index > D
    rng = np.random.default_rng(3)
    wild.loc[later, :] = rng.normal(size=(int(later.sum()), wild.shape[1])) * 50 + 30
    dirty = P.process_asset("SPY", wild, settings, pd.DataFrame(columns=P.LEDGER_COLS), None, bt,
                            HOLD_END, CUTOFF, max_date=D)
    assert_frame_equal(clean.rows, dirty.rows)
    assert clean.rows["date"].iloc[-1] == D


# --------------------------------------------------------------------------- adaptive margin
def test_adaptive_margin_matches_backtest_logic_and_uses_only_past_days():
    rng = np.random.default_rng(5)
    n = 300
    p = 0.5 + 0.03 * rng.normal(size=n + 40)
    base = np.full(n + 40, 0.52)
    nr = 0.0004 + 0.01 * rng.normal(size=n + 40)
    w = rng.uniform(0.3, 1, n + 40)
    cfgs = load_config()
    s = P.PaperSettings.from_config(cfgs)
    mm, fb = P.choose_margin(s, p[:n], base[:n], nr[:n], w[:n], 0.0005)
    _, log = B.adaptive_positions("long_flat", p, base, nr, w, [0, n], 0.0005, FAST_BT.adaptive_min_days)
    assert (mm, fb) == (log[-1]["m"], log[-1]["used_fallback"])
    assert not fb and mm in (0.0, 0.005, 0.01, 0.02)
    # nothing after the history can matter: the choice is a function of the first n days only
    mm2, _ = P.choose_margin(s, p[:n], base[:n], np.r_[nr[:n - 1], nr[n - 1]], w[:n], 0.0005)
    assert mm2 == mm
    # fallback until enough past days exist
    m0, fb0 = P.choose_margin(s, p[:30], base[:30], nr[:30], w[:30], 0.0005)
    assert fb0 and m0 == 0.0
    assert P.choose_margin(s, p[:0], base[:0], nr[:0], w[:0], 0.0005) == (0.0, True)


def test_adaptive_margin_history_ends_with_the_realised_return_into_today(tmp_path, monkeypatch):
    """Regression test (Stage 4 Leakage auditor, round 1, gap D1): the last element of the
    adaptive-margin history is the return realised from the previous decision day to TODAY
    (known at today's close), never tomorrow's return, and never the wrong sign."""
    cfg = new_world(tmp_path)
    seen = []
    real = P._history_arrays

    def spy(bt, live, r_last, today):
        out = real(bt, live, r_last, today)
        seen.append((today, r_last, out[2].copy(), len(live)))
        return out
    monkeypatch.setattr(P, "_history_arrays", spy)
    go(cfg, "2021-10-18")                       # first run: no live history yet
    go(cfg, "2021-11-10")                       # catch-up over a month change: retrain on Nov 1
    prices = R._truncate_prices(cfg, PX, pd.Timestamp("2021-12-31", tz="UTC"))
    ds = R.F.build_dataset("SPY", cfg, prices)
    nxt = ds["next_return"]
    checked = 0
    for today, r_last, nr_hist, n_live in seen:
        if n_live == 0 or today not in ds.index:     # SPY rows only (the ETF calendar)
            continue
        i = ds.index.get_loc(today)
        expected = float(nxt.iloc[i - 1])            # return over (D-1 -> D)
        if np.isclose(r_last, expected) or np.isclose(nr_hist[-1], expected):
            assert r_last == pytest.approx(expected)
            assert nr_hist[-1] == pytest.approx(expected)
            assert expected != pytest.approx(float(nxt.iloc[i]))   # not tomorrow's return
            assert expected != pytest.approx(-float(nxt.iloc[i - 1])) or expected == 0
            checked += 1
    assert checked >= 1, "no SPY call with a live history was checked"


def test_fixed_m_setting(tmp_path):
    cfg = make_cfg(tmp_path)
    cfg["paper_trading"]["strategy"]["m_setting"] = "0.01"
    s = P.PaperSettings.from_config(cfg)
    assert P.choose_margin(s, np.ones(3), np.ones(3), np.ones(3), np.ones(3), 0.0) == (0.01, False)
    assert s.config_id("SPY") == "SPY|expanding|logreg|long_flat|vt|m=0.01"


def test_pre_start_history_is_the_stage3_dev_period_and_live_part_is_after_the_holdout(tmp_path, monkeypatch):
    """The adaptive-margin history = Stage 3 OOS days (<= cutoff) + live days; a recorder on the
    guards proves no hold-out date is ever used."""
    cfg = new_world(tmp_path)
    live_calls, dev_calls = [], []
    real_live, real_dev = P.assert_live_dates, P.assert_dev_dates
    monkeypatch.setattr(P, "assert_live_dates",
                        lambda d, s, h, what="x": (live_calls.append((what, pd.DatetimeIndex(d))), real_live(d, s, h, what))[1])
    monkeypatch.setattr(P, "assert_dev_dates",
                        lambda d, c, what="x": (dev_calls.append((what, pd.DatetimeIndex(d))), real_dev(d, c, what))[1])
    go(cfg, "2021-10-01")
    go(cfg, "2021-11-12")                       # includes a retrain with live history
    assert live_calls and dev_calls
    for what, d in live_calls:
        if len(d):
            assert d.min() > HOLD_END, what
    for what, d in dev_calls:
        assert d.max() <= pd.Timestamp(CUTOFF), what
    kinds = {w for w, _ in live_calls}
    assert "live Brier/hit rate" in kinds and "adaptive-margin history (live part)" in kinds


# --------------------------------------------------------------------------- hold-out safety
def test_first_decision_date_inside_holdout_is_refused(tmp_path):
    cfg = new_world(tmp_path)
    with pytest.raises(P.HoldoutError, match="hold-out"):
        go(cfg, "2021-08-13")
    assert not (tmp_path / "paper").exists()


def test_asof_after_holdout_end_starts_the_ledger_at_that_date(baseline):
    cfg, L = baseline
    start = L.groupby("asset")["date"].min()
    assert (start >= HOLD_END).all()
    assert start["SPY"] == pd.Timestamp("2021-10-01")
    assert start["BTC-USD"] == pd.Timestamp("2021-09-30")          # BTC: yesterday (UTC) of 10-01
    state = P.load_state(P.PaperFiles.for_config(cfg))
    assert state["start_date"] == "2021-10-01"
    assert state["holdout_end"] == str(HOLD_END.date())
    # nothing earlier than the start date and no realised return on or before the hold-out end
    assert L.loc[L["asset_return"].notna(), "date"].min() > HOLD_END


def test_live_metric_guard_rejects_holdout_dates():
    with pytest.raises(P.HoldoutError):
        P.assert_live_dates(pd.DatetimeIndex(["2021-09-30"]), None, HOLD_END)
    with pytest.raises(P.HoldoutError):
        P.assert_live_dates(pd.DatetimeIndex(["2021-10-05"]), "2021-10-06", HOLD_END)
    P.assert_live_dates(pd.DatetimeIndex(["2021-10-05"]), "2021-10-01", HOLD_END)
    with pytest.raises(P.HoldoutError):
        P.assert_dev_dates(pd.DatetimeIndex(["2021-04-01"]), CUTOFF)


def test_holdout_dated_ledger_row_cannot_enter_a_metric(tmp_path):
    cfg = new_world(tmp_path)
    go(cfg, "2021-10-01")
    files = P.PaperFiles.for_config(cfg)
    L = P.load_ledger(files)
    L.loc[(L.asset == "SPY"), "date"] = pd.Timestamp("2021-06-01")      # tamper: a hold-out day
    P.save_ledger(files, L)
    with pytest.raises(P.PaperError):
        go(cfg, "2021-10-05")


def test_backtest_reference_refuses_predictions_after_the_cutoff(tmp_path):
    cfg = new_world(tmp_path)
    rd = cfg["data_dir"] / "results" / "SPY"
    pred = pd.read_parquet(rd / "predictions.parquet")
    extra = pred.tail(3).copy()
    extra["date"] = pd.to_datetime(["2021-05-03", "2021-05-04", "2021-05-05"])
    pd.concat([pred, extra]).to_parquet(rd / "predictions.parquet", index=False)
    s = P.PaperSettings.from_config(cfg)
    with pytest.raises(P.HoldoutError):
        P.load_backtest_history("SPY", s, cfg["data_dir"] / "results", CUTOFF)


# --------------------------------------------------------------------------- sized mode
def sized_settings(tmp_path, lam=0.25, band=0.1, vt=True):
    cfg = make_cfg(tmp_path, strategy_mode="sized", sized_lambda=lam, sized_band=band)
    cfg["paper_trading"]["strategy"]["vol_targeted"] = vt
    return P.PaperSettings.from_config(cfg)


def test_sized_position_formula_and_clipping(tmp_path):
    s = sized_settings(tmp_path, lam=0.5, band=0.0)
    f = P.decide_position
    # lambda*(p-base)/scale * ratio
    assert f(s, 0.56, 0.54, np.nan, 1.0, 0.04, 0.0) == pytest.approx(0.5 * 0.02 / 0.04)
    assert f(s, 0.56, 0.54, np.nan, 0.5, 0.04, 0.0) == pytest.approx(0.5 * 0.02 / 0.04 * 0.5)
    assert f(s, 0.90, 0.54, np.nan, 5.0, 0.01, 0.0) == 1.0          # clipped, never levered
    assert f(s, 0.40, 0.54, np.nan, 1.0, 0.04, 0.0) == 0.0          # long-only
    nov = sized_settings(tmp_path, lam=0.5, band=0.0, vt=False)
    assert f(nov, 0.56, 0.54, np.nan, 7.0, 0.04, 0.0) == pytest.approx(0.25)   # ratio ignored


def test_sized_no_trade_band(tmp_path):
    s = sized_settings(tmp_path, lam=0.5, band=0.1, vt=False)
    f = P.decide_position
    # target = 0.5 * (p - 0.54) / 0.02, so p - base = 0.02 -> 0.5, 0.014 -> 0.35, 0.012 -> 0.30
    assert f(s, 0.554, 0.54, np.nan, 1.0, 0.02, 0.30) == 0.30            # target 0.35, diff 0.05 < band: hold
    assert f(s, 0.560, 0.54, np.nan, 1.0, 0.02, 0.30) == pytest.approx(0.50)   # diff 0.20: trade
    assert f(s, 0.552, 0.54, np.nan, 1.0, 0.02, 0.20) == pytest.approx(0.30)   # diff 0.10 = band: trade
    assert f(s, 0.545, 0.54, np.nan, 1.0, 0.02, 0.20) == 0.20            # target 0.125: hold
    assert f(s, 0.30, 0.54, np.nan, 1.0, 0.02, 0.05) == 0.05             # wants 0, only 0.05 away: hold


def test_sized_mode_end_to_end_band_and_config_id(tmp_path):
    cfg = make_cfg(tmp_path, strategy_mode="sized", sized_lambda=0.5, sized_band=0.1)
    make_results(cfg)
    go(cfg, "2021-10-01"); go(cfg, "2021-11-12")
    L = ledger(cfg)
    assert (L["strategy_mode"] == "sized").all()
    assert L["config_id"].str.contains(r"\|sized\|vt\|lambda=0.5,band=0.1").all()
    assert L["position"].between(0, 1).all()
    first = L.groupby("asset").head(1).index
    moves = L.loc[(L["trade"] > 1e-12) & ~L.index.isin(first), "trade"]
    assert (moves >= 0.1 - 1e-9).all()                                   # band respected
    assert L["margin_m"].isna().all()                                    # no m in sized mode
    st = P.load_state(P.PaperFiles.for_config(cfg))["assets"]["SPY"]
    assert st["sized_scale"] > 0


# --------------------------------------------------------------------------- default_long mode
def test_default_long_mode(tmp_path):
    cfg = make_cfg(tmp_path, strategy_mode="default_long")
    make_results(cfg)
    go(cfg, "2021-10-01"); go(cfg, "2021-10-20")
    L = ledger(cfg)
    assert L["config_id"].str.contains(r"\|default_long\|vt\|m=adaptive").all()
    w = np.minimum(1.0, L["vol_ratio"])
    sig = 1.0 - (L["p_up"] < L["base_rate"] - L["margin_m"]).astype(float)
    np.testing.assert_allclose(L["position"], sig * w, atol=1e-12)


def test_multiple_runs_never_touch_data_results(tmp_path):
    cfg = new_world(tmp_path)
    rd = cfg["data_dir"] / "results"
    before = {p: p.read_bytes() for p in rd.rglob("*") if p.is_file()}
    go(cfg, "2021-10-01"); go(cfg, "2021-11-12")
    after = {p: p.read_bytes() for p in rd.rglob("*") if p.is_file()}
    assert before == after


def test_running_live_metrics_helper():
    p = np.array([0.6, 0.4, 0.55]); b = np.array([0.5, 0.5, 0.5]); y = np.array([1.0, 0.0, 0.0])
    n, br, sk, hr = P.running_live_metrics(p, b, y)
    assert list(n) == [1, 2, 3]
    assert br[-1] == pytest.approx(M.brier(y, p))
    assert sk[-1] == pytest.approx(M.brier_skill(y, p, b))
    assert hr[-1] == pytest.approx(2 / 3)


def test_ensemble_of_all_three_models_end_to_end(tmp_path):
    """The pinned Stage 3 config (ensemble of logreg + rf + lgbm) works end to end."""
    pytest.importorskip("lightgbm")
    cfg = make_cfg(tmp_path)
    cfg["paper_trading"]["strategy"]["model"] = "ensemble"
    make_results(cfg, models=("ensemble",))
    go(cfg, "2021-10-15"); go(cfg, "2021-10-20")
    L = ledger(cfg)
    assert L["config_id"].str.contains(r"\|expanding\|ensemble\|long_flat\|vt\|m=adaptive").all()
    model = P.load_model(P.PaperFiles.for_config(cfg), "SPY")
    assert isinstance(model, MD.EnsembleModel) and set(model.members) == set(MD.MODEL_NAMES)
    X = R.F.build_dataset("SPY", cfg, R._truncate_prices(cfg, PX, pd.Timestamp("2021-10-21", tz="UTC")))
    feats = R.F.feature_columns(X)
    row = L[(L.asset == "SPY")].iloc[-1]
    assert model.predict_up(X.loc[[row["date"]], feats])[0] == pytest.approx(row["p_up"])
