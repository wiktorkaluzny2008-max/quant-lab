import math

import numpy as np
import pandas as pd
import pytest
from scipy import stats
from sklearn.metrics import roc_auc_score
from statsmodels.tsa.stattools import diebold_mariano_test

from quant import metrics as m


# ---------------- prediction ----------------
def test_accuracy_vs_base_rate_hand():
    y = [1, 1, 0, 1]
    r = m.accuracy_vs_base_rate(y, [1, 0, 0, 1])
    assert r["accuracy"] == 0.75 and r["base_rate"] == 0.75 and r["edge"] == 0.0
    r = m.accuracy_vs_base_rate(y, [1, 1, 1, 1])  # always up == base rate
    assert r["accuracy"] == r["base_rate"] == 0.75
    assert np.isnan(m.accuracy_vs_base_rate([], [])["accuracy"])


def test_auc_matches_sklearn_and_single_class_nan():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 300)
    p = np.clip(0.5 + 0.1 * (y - 0.5) + rng.normal(0, 0.2, 300), 0, 1)
    assert m.roc_auc(y, p) == pytest.approx(roc_auc_score(y, p))
    assert np.isnan(m.roc_auc([1, 1, 1], [0.2, 0.5, 0.9]))


def test_log_loss_and_brier_hand():
    y, p = np.array([1, 0]), np.array([0.8, 0.4])
    assert m.log_loss(y, p) == pytest.approx(-(math.log(0.8) + math.log(0.6)) / 2)
    assert m.brier(y, p) == pytest.approx((0.04 + 0.16) / 2)
    assert np.isfinite(m.log_loss([1, 0], [0.0, 1.0]))  # clipped, no inf


def test_skill_scores():
    rng = np.random.default_rng(1)
    y = (rng.random(500) < 0.55).astype(float)
    assert m.brier_skill(y, np.full(500, 0.55), 0.55) == pytest.approx(0.0)
    assert m.log_loss_skill(y, np.full(500, 0.55), 0.55) == pytest.approx(0.0)
    bad = np.where(y == 1, 0.1, 0.9)
    assert m.brier_skill(y, bad, 0.55) < 0 and m.log_loss_skill(y, bad, 0.55) < 0
    good = np.where(y == 1, 0.8, 0.2)
    assert m.brier_skill(y, good, 0.55) > 0
    # per-row climatology equals scalar when constant
    assert m.brier_skill(y, bad, np.full(500, 0.55)) == pytest.approx(m.brier_skill(y, bad, 0.55))
    with pytest.raises(ValueError):
        m.brier_skill(y, bad, np.ones(3))


def test_reliability_curve_hand():
    p = np.array([0.05, 0.15, 0.12, 0.95, 1.0])
    y = np.array([0, 1, 0, 1, 1])
    df = m.reliability_curve(p, y, bins=10)
    assert df["count"].tolist() == [1, 2, 0, 0, 0, 0, 0, 0, 0, 2]
    assert df.loc[1, "mean_p"] == pytest.approx(0.135) and df.loc[1, "obs_freq"] == 0.5
    assert df.loc[9, "obs_freq"] == 1.0 and np.isnan(df.loc[2, "mean_p"])


def test_hit_rate_by_confidence_decile():
    rng = np.random.default_rng(2)
    n = 5000
    y = rng.integers(0, 2, n).astype(float)
    p = np.clip(0.5 + (y - 0.5) * rng.random(n) * 0.4 * rng.integers(0, 2, n), 0.01, 0.99)
    df = m.hit_rate_by_confidence_decile(p, y, 0.5)
    assert len(df) == 10 and df["n"].sum() == n
    assert df["mean_confidence"].is_monotonic_increasing
    # constant predictions still split into deciles without error
    assert len(m.hit_rate_by_confidence_decile(np.full(100, 0.5), np.ones(100), 0.5)) == 10


def test_pesaran_timmermann_hand_computed():
    y = np.array([1, 1, 1, 0, 0, 1, 0, 1, 1, 0])
    z = np.array([1, 1, 0, 0, 0, 1, 1, 1, 1, 0])
    n = 10
    py, pz, ph = 0.6, 0.6, 0.8  # hand counts: y up 6, z up 6, matches 8
    ps = py * pz + (1 - py) * (1 - pz)  # 0.52
    v = ps * (1 - ps) / n
    w = ((2 * py - 1) ** 2 * pz * (1 - pz) + (2 * pz - 1) ** 2 * py * (1 - py)) / n
    stat = (ph - ps) / math.sqrt(v - w)
    r = m.pesaran_timmermann(y, z)
    assert r["statistic"] == pytest.approx(stat)
    assert r["pvalue"] == pytest.approx(2 * stats.norm.sf(abs(stat)))
    assert r["share_up_days"] == 0.6 and r["share_up_predictions"] == 0.6


def test_pesaran_timmermann_random_and_perfect():
    rng = np.random.default_rng(3)
    y = rng.integers(0, 2, 1000)
    assert m.pesaran_timmermann(y, rng.integers(0, 2, 1000))["pvalue"] > 0.05
    assert m.pesaran_timmermann(y, y)["pvalue"] < 1e-10


def test_pesaran_timmermann_constant_predictions_nan():
    r = m.pesaran_timmermann([1, 0, 1, 1], [1, 1, 1, 1])
    assert np.isnan(r["pvalue"]) and r["share_up_predictions"] == 1.0


# ---------------- trading ----------------
def test_periods_per_year():
    assert m.periods_per_year("BTC-USD") == 365 and m.periods_per_year("SPY") == 252


def test_total_return_cagr_vol_hand():
    r = np.array([0.1, -0.1, 0.05])
    assert m.total_return(r) == pytest.approx(1.1 * 0.9 * 1.05 - 1)
    assert m.cagr(r, 3) == pytest.approx(1.1 * 0.9 * 1.05 - 1)  # exactly one "year"
    assert m.cagr(r, 6) == pytest.approx((1.1 * 0.9 * 1.05) ** 2 - 1)
    assert m.ann_vol(r, 252) == pytest.approx(np.std(r, ddof=1) * math.sqrt(252))
    assert m.cagr([-1.0, 0.1], 252) == -1.0


def test_sharpe_sortino_loops():
    rng = np.random.default_rng(4)
    r = rng.normal(0.001, 0.01, 400)
    mu = sum(r) / len(r)
    sd = math.sqrt(sum((x - mu) ** 2 for x in r) / (len(r) - 1))
    assert m.sharpe(r, 252) == pytest.approx(mu / sd * math.sqrt(252))
    dd = math.sqrt(sum(min(x, 0) ** 2 for x in r) / len(r))
    assert m.sortino(r, 252) == pytest.approx(mu / dd * math.sqrt(252))
    assert np.isnan(m.sharpe(np.zeros(10), 252)) and np.isnan(m.sharpe([0.1], 252))
    assert np.isnan(m.sortino([0.1, 0.2], 252))


def test_max_drawdown_loop():
    rng = np.random.default_rng(5)
    r = rng.normal(0, 0.02, 300)
    eq, peak, worst = 1.0, 1.0, 0.0
    for x in r:
        eq *= 1 + x
        peak = max(peak, eq)
        worst = min(worst, eq / peak - 1)
    assert m.max_drawdown(r) == pytest.approx(worst)
    assert m.max_drawdown([0.1, 0.1]) == 0.0
    assert m.max_drawdown([-0.5, 0.0]) == pytest.approx(-0.5)  # drawdown from the starting equity


def test_position_stats_hand():
    pos = np.array([0, 1, 1, 0, 0, 1, 0.5])
    assert m.pct_time_invested(pos) == pytest.approx(4 / 7)
    assert m.n_trades(pos) == 4  # 0->1, 1->0, 0->1, 1->.5
    assert m.turnover(pos) == pytest.approx((0 + 1 + 0 + 1 + 0 + 1 + 0.5) / 7)
    assert m.n_trades(np.ones(5)) == 1  # buy and hold: one entry
    assert m.n_trades(np.zeros(5)) == 0


def test_net_returns_and_break_even():
    r = np.array([0.01, 0.02, -0.01, 0.03])
    pos = np.array([1, 1, 0, 1.0])
    gross = pos * r
    be = m.break_even_cost(gross, pos)
    assert be == pytest.approx(gross.mean() / m.turnover(pos))
    # at the break-even cost the mean net return is zero
    assert m.net_returns(r, pos, be).mean() == pytest.approx(0.0, abs=1e-15)
    assert np.isnan(m.break_even_cost(np.zeros(4), np.zeros(4)))


def test_sharpe_standard_error():
    # iid Lo (2002): sqrt((1+SR^2/2)/n); 252 days, daily SR=0.0630 -> annual ~1.0
    sr_d = 1.0 / math.sqrt(252)
    se = m.sharpe_standard_error(sr_d, 252, 252)
    assert se == pytest.approx(math.sqrt((1 + sr_d ** 2 / 2) / 252) * math.sqrt(252))
    assert 0.95 < se < 1.1  # research report: ~1.0 for one year
    se2 = m.sharpe_standard_error(sr_d, 504, 252)
    assert 0.7 < se2 < 0.75  # ~0.71 for two years
    # normal skew/kurt: (1 + SR^2/2)/(n-1) vs iid (1+SR^2/2)/n -> nearly identical
    assert m.sharpe_standard_error(sr_d, 252, 252, 0.0, 3.0) == pytest.approx(se, rel=5e-3)
    # negative skew widens the SE
    assert m.sharpe_standard_error(sr_d, 252, 252, -1.0, 6.0) > se
    assert np.isnan(m.sharpe_standard_error(0.1, 1))


def test_tstat_and_hurdle():
    r = np.array([0.01, 0.02, 0.0, 0.03])
    assert m.tstat_mean(r) == pytest.approx(stats.ttest_1samp(r, 0).statistic)
    assert np.isnan(m.tstat_mean(np.zeros(5)))
    rng = np.random.default_rng(6)
    big = rng.normal(0.01, 0.01, 100)
    assert m.t_hurdle(big)["passes"] and not m.t_hurdle(r)["passes"]
    assert not m.t_hurdle(np.zeros(5))["passes"]


# ---------------- significance ----------------
def test_random_strategy_percentile_random_position():
    rng = np.random.default_rng(7)
    r = rng.normal(0, 0.01, 1500)
    pcts_shift, pcts_iid = [], []
    for s in range(30):
        pos = (np.random.default_rng(100 + s).random(1500) < 0.5).astype(float)
        pcts_shift.append(m.random_strategy_percentile(r, pos, 300, seed=s)["percentile"])
        pcts_iid.append(m.random_strategy_percentile(r, pos, 300, seed=s, method="iid")["percentile"])
    assert 35 < np.mean(pcts_shift) < 65 and 35 < np.mean(pcts_iid) < 65


def test_random_strategy_percentile_perfect_foresight_and_properties():
    rng = np.random.default_rng(8)
    r = rng.normal(0, 0.01, 800)
    pos = (r > 0).astype(float)
    for method in ("shift", "iid"):
        out = m.random_strategy_percentile(r, pos, 500, seed=1, method=method)
        assert out["percentile"] > 99
    a = m.random_strategy_percentile(r, pos, 50, seed=3)
    b = m.random_strategy_percentile(r, pos, 50, seed=3)
    assert np.array_equal(a["random_sharpes"], b["random_sharpes"])  # seeded
    with pytest.raises(ValueError):
        m.random_strategy_percentile(r, pos, 10, method="bogus")


def test_random_strategy_percentile_all_flat_nan():
    r = np.random.default_rng(9).normal(0, 0.01, 100)
    assert np.isnan(m.random_strategy_percentile(r, np.zeros(100), 20)["percentile"])


def test_psr_hand_and_report_example():
    sr, srb, n = 0.1, 0.0, 101
    expected = stats.norm.cdf((0.1 - 0) * 10 / math.sqrt(1 - 0 + 0.5 * 0.01))
    assert m.probabilistic_sharpe_ratio(sr, srb, n) == pytest.approx(expected)
    assert m.probabilistic_sharpe_ratio(0.1, 0.1, n) == pytest.approx(0.5)
    # report: annual SR 1 over 5 years (1260 days) -> PSR vs 0 about 0.99
    sd = 1 / math.sqrt(252)
    assert m.probabilistic_sharpe_ratio(sd, 0, 1260) > 0.98
    # negative skew lowers it
    assert m.probabilistic_sharpe_ratio(sd, 0, 1260, -2.0, 8.0) < m.probabilistic_sharpe_ratio(sd, 0, 1260)
    assert np.isnan(m.probabilistic_sharpe_ratio(10.0, 0, 100, 20.0, 3.0))  # non-positive variance


def test_deflated_sharpe_report_example_and_inputs():
    # report: annual SR 1, 5y, N=100, trial-Sharpe daily std 0.02 -> DSR about 0.67
    sr0 = m.expected_max_sharpe(100, 0.02 ** 2)
    g = stats.norm.ppf
    manual = 0.02 * ((1 - m.EULER_GAMMA) * g(1 - 1 / 100) + m.EULER_GAMMA * g(1 - 1 / (100 * math.e)))
    assert sr0 == pytest.approx(manual)
    p = m.probabilistic_sharpe_ratio(1 / math.sqrt(252), sr0, 1260)
    assert 0.6 < p < 0.75
    rng = np.random.default_rng(10)
    r = rng.normal(0.0008, 0.01, 1260)
    out = m.deflated_sharpe_ratio(r, n_trials=100, sr_var=0.02 ** 2)
    assert out["sr0"] == pytest.approx(sr0) and 0 <= out["dsr"] <= 1
    # more trials -> lower DSR; N=1 -> equals PSR vs 0
    d10 = m.deflated_sharpe_ratio(r, n_trials=10, sr_var=1e-4)["dsr"]
    d1000 = m.deflated_sharpe_ratio(r, n_trials=1000, sr_var=1e-4)["dsr"]
    assert d1000 < d10
    one = m.deflated_sharpe_ratio(r, n_trials=1, sr_var=1e-4)
    assert one["dsr"] == pytest.approx(m.probabilistic_sharpe_ratio(
        m.sharpe(r), 0, r.size, stats.skew(r), stats.kurtosis(r, fisher=False)))
    ts = rng.normal(0, 0.02, 50)
    o2 = m.deflated_sharpe_ratio(r, trial_sharpes=ts)
    assert o2["n_trials"] == 50 and o2["sr_var"] == pytest.approx(np.var(ts, ddof=1))
    with pytest.raises(ValueError):
        m.deflated_sharpe_ratio(r)
    with pytest.raises(ValueError):
        m.deflated_sharpe_ratio(r, trial_sharpes=[0.1])
    assert np.isnan(m.deflated_sharpe_ratio(np.zeros(50), n_trials=5, sr_var=1e-4)["dsr"])


def _dm_loop(a, b, h):
    d = [x - y for x, y in zip(a, b)]
    T = len(d)
    mu = sum(d) / T
    gam = lambda k: sum((d[t] - mu) * (d[t - k] - mu) for t in range(k, T)) / T
    lrv = gam(0) + 2 * sum(gam(k) for k in range(1, h))
    dm = mu / math.sqrt(lrv / T)
    return dm * math.sqrt((T + 1 - 2 * h + h * (h - 1) / T) / T)


@pytest.mark.parametrize("h", [1, 3])
def test_diebold_mariano_vs_loop_and_statsmodels(h):
    rng = np.random.default_rng(11)
    a = rng.exponential(1.0, 200) + 0.1
    b = rng.exponential(1.0, 200)
    r = m.diebold_mariano(a, b, h=h)
    assert r["statistic"] == pytest.approx(_dm_loop(a, b, h))
    assert r["pvalue"] == pytest.approx(2 * stats.t.sf(abs(r["statistic"]), 199))
    sm = diebold_mariano_test(np.zeros(200), a, b, lags=h - 1, harvey_adj=True,
                              horizon=h, criterion=lambda y, f: f)
    if h == 1:  # statsmodels' HAC small-sample conventions can differ for h>1
        assert r["statistic"] == pytest.approx(sm.statistic, rel=1e-6)
        assert r["pvalue"] == pytest.approx(sm.pvalue, rel=1e-6)


def test_diebold_mariano_sign_and_edge():
    rng = np.random.default_rng(12)
    good = rng.exponential(1.0, 300)
    bad = good + rng.exponential(0.3, 300)
    r = m.diebold_mariano(bad, good)  # A worse -> positive
    assert r["statistic"] > 0 and r["pvalue"] < 0.01
    assert m.diebold_mariano(good, bad)["statistic"] < 0
    assert m.diebold_mariano(good, good + rng.normal(0, 1, 300))["pvalue"] > 0.01
    assert np.isnan(m.diebold_mariano(good, good)["statistic"])
    with pytest.raises(ValueError):
        m.diebold_mariano(good, good[:-1])


def test_spa_noise_vs_real_edge():
    rng = np.random.default_rng(13)
    n = 750
    idx = pd.bdate_range("2020-01-01", periods=n)
    bench = pd.Series(rng.normal(0.0003, 0.01, n), index=idx)
    noise = pd.DataFrame(rng.normal(0.0003, 0.01, (n, 8)), index=idx,
                         columns=[f"s{i}" for i in range(8)])
    assert m.spa_test(bench, noise, reps=500, seed=1)["pvalue"] > 0.2
    real = noise.copy()
    real["edge"] = bench + 0.002 + rng.normal(0, 0.003, n)
    out = m.spa_test(bench, real, reps=500, seed=1)
    assert out["pvalue"] < 0.05 and out["best_strategy"] == "edge"
    assert out["pvalue"] == m.spa_test(bench, real, reps=500, seed=1)["pvalue"]  # seeded
    with pytest.raises(ValueError):
        m.spa_test(bench[:5], noise[:5])
