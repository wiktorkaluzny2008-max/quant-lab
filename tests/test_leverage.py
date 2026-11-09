"""Tests for quant/leverage.py on synthetic data (no models, no network)."""
import json
import math

import numpy as np
import pandas as pd
import pytest

from quant import leverage as LV
from quant import metrics as M
from quant import trials as T
from quant.config import load_config


# ------------------------------------------------------------------ formula
def test_hand_computed_leveraged_returns():
    g = [0.01, -0.02, 0.005]
    pos = [1.0, 0.5, 0.5]            # dpos: 1, 0.5, 0
    L, b, dpy, c = 2.0, 0.06, 252, 0.001
    r = LV.leveraged_returns(g, pos, c, L, b, dpy)
    borrow = 1.0 * 0.06 / 252
    exp = [2 * 0.01 - borrow - c * 2 * 1.0,
           2 * -0.02 - borrow - c * 2 * 0.5,
           2 * 0.005 - borrow - 0.0]
    assert np.allclose(r, exp, atol=1e-15, rtol=0)


def test_l1_equals_unlevered_net_returns_exactly():
    rng = np.random.default_rng(0)
    ret = rng.normal(0, 0.01, 200)
    pos = rng.choice([0.0, 0.5, 1.0], 200)
    net = M.net_returns(ret, pos, 0.0005)
    r1 = LV.leveraged_returns(pos * ret, pos, 0.0005, 1.0, 0.06, 252)
    assert np.array_equal(r1, net) or np.allclose(r1, net, atol=1e-16, rtol=0)


def test_borrow_days_btc_365_etf_252():
    cfg = load_config()
    assert LV.days_per_year("BTC-USD", cfg) == 365
    for t in ("SPY", "QQQ", "IWM"):
        assert LV.days_per_year(t, cfg) == 252
    z = np.zeros(3)
    r_btc = LV.leveraged_returns(z, np.zeros(3), 0.0, 3.0, 0.06, LV.days_per_year("BTC-USD", cfg))
    r_etf = LV.leveraged_returns(z, np.zeros(3), 0.0, 3.0, 0.06, LV.days_per_year("SPY", cfg))
    assert np.allclose(r_btc, -2 * 0.06 / 365) and np.allclose(r_etf, -2 * 0.06 / 252)


# ------------------------------------------------------------------ wipe-out
def _crash():
    r = np.full(30, 0.0005)
    r[10] = -0.30
    r[11] = -0.20
    r[12:] = 0.01
    return r


def test_wiped_out_at_3x_not_at_1x():
    dates = pd.bdate_range("2020-01-01", periods=30)
    r = _crash()
    rows, paths = LV.simulate_series(dates, r, np.ones(30), 0.0, 0.0, 252)
    by = {x["L"]: x for x in rows}
    assert not by[1.0]["wiped_out"] and pd.isna(by[1.0]["wipe_date"])
    assert by[3.0]["wiped_out"]
    rl, eq = paths[3.0]
    # day 10: 1 - 0.9 -> equity 0.1 < 0.2 -> wiped that day, then frozen
    assert by[3.0]["wipe_date"] == dates[10]
    assert np.all(rl[11:] == 0.0) and np.all(eq[10:] == eq[10])
    assert by[3.0]["final_equity"] == pytest.approx(eq[10])
    assert eq[10] >= 0.0


def test_wipeout_floor_at_zero():
    r, k = LV.apply_wipeout([0.01, -1.7, 0.5])
    assert k == 1 and r[1] == -1.0 and r[2] == 0.0
    assert np.cumprod(1 + r)[-1] == 0.0


# ------------------------------------------------------------------ metrics vs plain loop
def test_metrics_vs_plain_loop():
    rng = np.random.default_rng(3)
    dates = pd.bdate_range("2021-01-04", periods=300)
    r = rng.normal(0.0005, 0.012, 300)
    m = LV.path_metrics(r, dates, 252)
    # plain loops
    eq, peak, run, best = 1.0, 1.0, 0, 0
    for x in r:
        eq *= 1 + x
        if eq >= peak:
            peak, run = eq, 0
        else:
            run += 1
            best = max(best, run)
    assert m["longest_recovery_days"] == best
    assert m["worst_day"] == min(r)
    assert m["worst_day_date"] == dates[int(np.argmin(r))]
    months = {}
    for d, x in zip(dates, r):
        months[(d.year, d.month)] = months.get((d.year, d.month), 1.0) * (1 + x)
    k = min(months, key=months.get)
    assert m["worst_month"] == pytest.approx(months[k] - 1)
    assert m["worst_month_label"] == f"{k[0]:04d}-{k[1]:02d}"
    # max drawdown
    e, pk, dd = 1.0, 1.0, 0.0
    for x in r:
        e *= 1 + x
        pk = max(pk, e)
        dd = min(dd, e / pk - 1)
    assert m["max_drawdown"] == pytest.approx(dd)
    assert m["cagr"] == pytest.approx(M.cagr(r, 252))


def test_longest_underwater_known_and_unrecovered():
    assert LV.longest_underwater([1.1, 1.0, 0.9, 1.1, 1.2, 1.1, 1.1]) == 2
    assert LV.longest_underwater([1.1, 1.0, 0.9, 1.1]) == 2          # 1.0, 0.9 below 1.1
    assert LV.longest_underwater([0.9, 0.95, 0.99]) == 3             # never recovers 1.0
    m = LV.path_metrics(np.array([-0.1, 0.01, 0.01]), pd.bdate_range("2020-01-01", periods=3), 252)
    assert m["unrecovered"] and m["longest_recovery_days"] == 3


# ------------------------------------------------------------------ Kelly
def test_kelly_hand_computed_and_growth_curve():
    r = np.array([0.01, -0.01, 0.02, 0.0] * 50)       # 200 days
    est = LV.kelly_estimate(r, 252)
    mu, var = r.mean() * 252, np.var(r, ddof=1) * 252
    assert est["mu"] == pytest.approx(mu) and est["sigma2"] == pytest.approx(var)
    assert est["L_star"] == pytest.approx(mu / var)
    se = math.sqrt(var) / math.sqrt(200 / 252)
    assert est["se_mu"] == pytest.approx(se)
    assert est["L_star_hi"] == pytest.approx((mu + 1.96 * se) / var)
    assert est["L_star_lo"] == pytest.approx(max(0.0, (mu - 1.96 * se) / var))
    Ls = est["L_star"]
    grid = np.linspace(0, 3 * Ls, 301)
    g = LV.growth_curve(grid, mu, var)
    assert grid[np.argmax(g)] == pytest.approx(Ls, abs=3 * Ls / 300)
    assert LV.growth_curve(Ls, mu, var) == pytest.approx(Ls * mu / 2)
    assert LV.growth_curve(2 * Ls, mu, var) == pytest.approx(0.0, abs=1e-12)
    assert LV.growth_curve(3 * Ls, mu, var) < 0


def test_kelly_needs_enough_data():
    with pytest.raises(ValueError, match="at least 30"):
        LV.kelly_estimate(np.zeros(5), 252)


# ------------------------------------------------------------------ crash windows
def test_crash_windows_btc_vs_etf():
    cfg = load_config()
    assert {k for k, *_ in LV.crash_windows("BTC-USD", cfg)} == {"bear_2018", "covid_2020", "bear_2022"}
    assert {k for k, *_ in LV.crash_windows("SPY", cfg)} == {"dotcom_2000_2002", "gfc_2008",
                                                             "covid_2020", "bear_2022"}


# ------------------------------------------------------------------ end to end on fake results
def _fake_results(tmp_path, ticker="SPY", with_ensemble=True, models=("logreg",), cutoff="2022-12-31"):
    rng = np.random.default_rng(1)
    # pre-OOS price history 2015-2019, OOS 2020-2022
    allidx = pd.bdate_range("2015-01-01", "2022-12-30")
    close = pd.Series(100 * np.cumprod(1 + rng.normal(0.0004, 0.01, len(allidx))), index=allidx)
    (tmp_path / "data").mkdir()
    pd.DataFrame({"Close": close}).rename_axis("Date").to_parquet(tmp_path / f"data/{ticker}.parquet")
    oos = pd.bdate_range("2020-01-01", "2022-12-30")
    r = close.pct_change().reindex(oos)
    w = pd.Series(np.clip(0.01 / (r.abs().rolling(5, min_periods=1).mean() + 1e-4) / 3, 0, 1), index=oos)
    pos = (rng.random(len(oos)) > 0.4).astype(float)
    rd = tmp_path / "data/results"
    (rd / ticker).mkdir(parents=True)
    base = []
    for name, p in (("buy_and_hold", np.ones(len(oos))), ("vol_targeted_bh", w.to_numpy())):
        gross = p * r.to_numpy()
        base.append(pd.DataFrame(dict(window="expanding", name=name, date=oos, position=p,
                                      gross_return=gross,
                                      net_return=M.net_returns(r.to_numpy(), p, 0.0005))))
    pd.concat(base).to_parquet(rd / ticker / "baselines.parquet")
    (rd / ticker / "metadata.json").write_text(json.dumps({"holdout_cutoff_date": cutoff}))
    idx_rows, ret_rows = [], []
    for mdl in (("ensemble",) if with_ensemble else ()) + tuple(models):
        cid = T.make_config_id(ticker, "expanding", mdl, "long_flat", True, "adaptive")
        net = M.net_returns(r.to_numpy(), pos, 0.0005)
        idx_rows.append(T.build_index_row(ticker, "expanding", mdl, "long_flat", True, "adaptive",
                                          0.0005, "{}", oos, net, 252))
        ret_rows.append(pd.DataFrame(dict(config_id=cid, date=oos, net_return=net,
                                          gross_return=pos * r.to_numpy(), position=pos)))
    T.replace_asset(ticker, pd.DataFrame(idx_rows), pd.concat(ret_rows), rd)
    cfg = load_config()
    cfg["data_dir"] = tmp_path / "data"
    return cfg, oos, r, pos


def test_end_to_end_files_and_content(tmp_path):
    cfg, oos, r, pos = _fake_results(tmp_path)
    res = LV.run_asset("SPY", cfg)
    out = tmp_path / "data/results/SPY"
    for f in ("leverage.parquet", "leverage_equity.parquet", "kelly.parquet", "crashes.json"):
        assert (out / f).exists()
    lev = pd.read_parquet(out / "leverage.parquet")
    assert set(lev["series"]) == set(LV.SERIES) and set(lev["L"]) == {1.0, 1.5, 2.0, 3.0}
    assert len(lev) == 12
    # L=1 equals unlevered net returns exactly (chosen strategy)
    eq = pd.read_parquet(out / "leverage_equity.parquet")
    e1 = eq[(eq.series == "chosen_strategy") & (eq.L == 1.0)].sort_values("date")["equity"].to_numpy()
    net = M.net_returns(r.to_numpy(), pos, 0.0005)
    assert np.allclose(np.diff(np.r_[1.0, e1]) / np.r_[1.0, e1[:-1]], net, atol=1e-14, rtol=0)
    # no hold-out dates: OOS ends before the cutoff; Kelly strictly before first OOS date
    assert eq["date"].max() <= pd.Timestamp("2022-12-31")
    kel = pd.read_parquet(out / "kelly.parquet")
    assert kel["train_end"].iloc[0] < oos[0] and kel["train_start"].iloc[0] >= pd.Timestamp("2015-01-01")
    assert kel["L_star"].iloc[0] == pytest.approx(res["kelly_est"]["L_star"])
    cr = json.loads((out / "crashes.json").read_text())
    names = {w["key"]: w for w in cr["windows"]}
    assert names["covid_2020"]["fully_covered"] and not names["gfc_2008"]["fully_covered"]
    assert names["gfc_2008"]["max_drawdown"]["buy_and_hold"]["1"] is None
    dd1 = names["covid_2020"]["max_drawdown"]["buy_and_hold"]["1"]
    dd3 = names["covid_2020"]["max_drawdown"]["buy_and_hold"]["3"]
    assert dd3 < dd1 <= 0
    assert cr["what_if_only"] is True and cr["chosen_model"] == "ensemble"


def test_borrow_cost_applied_in_run(tmp_path):
    cfg, oos, r, pos = _fake_results(tmp_path)
    cfg["borrow_rate"] = 0.0
    a = LV.run_asset("SPY", cfg)["leverage"]
    cfg["borrow_rate"] = 0.10
    b = LV.run_asset("SPY", cfg)["leverage"]
    ka = a[(a.series == "buy_and_hold")].set_index("L")["cagr"]
    kb = b[(b.series == "buy_and_hold")].set_index("L")["cagr"]
    assert ka[1.0] == pytest.approx(kb[1.0])          # no borrowing at 1x
    assert kb[3.0] < ka[3.0]


def test_fallback_when_requested_model_missing(tmp_path):
    cfg, *_ = _fake_results(tmp_path, with_ensemble=False, models=("logreg", "rf"))
    res = LV.run_asset("SPY", cfg)
    assert res["info"]["chosen_model"] == "logreg" and "ensemble" in res["info"]["chosen_note"]
    assert json.loads((tmp_path / "data/results/SPY/crashes.json").read_text())["chosen_model"] == "logreg"


def test_idempotent_and_does_not_touch_trial_log(tmp_path):
    cfg, *_ = _fake_results(tmp_path)
    rd = tmp_path / "data/results"
    trials_before = [(rd / f).read_bytes() for f in ("trials.parquet", "trials_index.parquet")]
    LV.run_asset("SPY", cfg)
    first = {f: (rd / "SPY" / f).read_bytes() for f in
             ("leverage.parquet", "leverage_equity.parquet", "kelly.parquet", "crashes.json")}
    LV.run_asset("SPY", cfg)
    second = {f: (rd / "SPY" / f).read_bytes() for f in first}
    assert first == second
    assert trials_before == [(rd / f).read_bytes() for f in ("trials.parquet", "trials_index.parquet")]
    assert T.count_trials(rd) == 2


def test_results_with_holdout_dates_rejected(tmp_path):
    cfg, *_ = _fake_results(tmp_path, cutoff="2022-06-30")
    with pytest.raises(AssertionError, match="hold-out"):
        LV.run_asset("SPY", cfg)


def test_missing_results_clear_error(tmp_path):
    cfg = load_config()
    cfg["data_dir"] = tmp_path
    with pytest.raises(LV.MissingResultsError, match="missing saved results"):
        LV.run_asset("SPY", cfg)


def test_cli_missing_results_exit_code(tmp_path, capsys, monkeypatch):
    import yaml
    raw = {k: v for k, v in load_config().items() if k not in ("project_root",)}
    raw["data_dir"] = str(tmp_path)
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump(raw))
    with pytest.raises(SystemExit) as ei:
        LV.main(["--config", str(p), "--assets", "SPY"])
    assert ei.value.code == 2
    assert "ERROR" in capsys.readouterr().err
