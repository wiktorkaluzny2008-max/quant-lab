"""Stage 6 hold-out harness (quant/holdout.py): equivalence with Stage 3, leakage, one-shot guard,
metric maths recomputed independently, freeze manifest, dashboard isolation.

Synthetic worlds live in tmp dirs; the REAL data tests only ever use a PSEUDO hold-out inside the
development period (the real hold-out rows are never read)."""
import dataclasses
import json
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats
from statsmodels.stats.proportion import proportion_confint

from quant import backtest as B
from quant import dash_data as DD
from quant import features as F
from quant import holdout as H
from quant import metrics as M
from quant import paper as P
from quant import trials as T
from quant.config import load_config
from tests.test_backtest import make_df

SMALL_BT = B.BTConfig(min_train_days=300, min_cal=50, adaptive_min_days=60, n_random=30,
                      importance_repeats=1, spa_reps=50, holdout_months=6)
CUTOFF = pd.Timestamp("2021-03-15")
END = CUTOFF + pd.DateOffset(months=6)          # 2021-09-15
TICKERS = ["SPY", "BTC-USD"]
COSTS = {"SPY": 0.0005, "BTC-USD": 0.0015}
STRAT = {"window": "expanding", "model": "logreg", "mode": "long_flat", "vol_targeted": True,
         "m_setting": "adaptive"}
ROOT = Path(__file__).resolve().parent.parent


def world_frame(ticker, seed, last="2021-12-31"):
    if ticker == F.BTC:
        idx = pd.date_range("2018-01-02", last, freq="D")
    else:
        s = F._sessions()
        idx = s[(s >= pd.Timestamp("2018-01-02")) & (s <= pd.Timestamp(last))]
    df = make_df(len(idx), seed)
    df.index = pd.DatetimeIndex(idx, name="Date")
    return df


@pytest.fixture(scope="module")
def base_tree(tmp_path_factory):
    """Synthetic project: features files, Stage 3 results (dev only), fake project files."""
    tmp = tmp_path_factory.mktemp("holdout_world")
    data = tmp / "data"
    (data / "features").mkdir(parents=True)
    proj = tmp / "proj"
    (proj / "quant").mkdir(parents=True)
    for n, body in (("a.py", "x = 1\n"), ("b.py", "y = 2\n")):
        (proj / "quant" / n).write_text(body)
    (proj / "config.yaml").write_text("a: 1\n")
    (proj / "uv.lock").write_text("lock\n")
    mp = pytest.MonkeyPatch()
    for i, t in enumerate(TICKERS):
        df = world_frame(t, i)
        df.to_parquet(data / "features" / f"{t}.parquet")
        dev, _ = B.split_holdout(df, 6, CUTOFF)
        run = B.run_asset(t, dev, CUTOFF, ["logreg"], SMALL_BT, COSTS[t], windows=["expanding"],
                          importance=False)
        rd = data / "results"
        (rd / t).mkdir(parents=True, exist_ok=True)
        run.preds.to_parquet(rd / t / "predictions.parquet", index=False)
        run.baselines.to_parquet(rd / t / "baselines.parquet", index=False)
        T.replace_asset(t, run.trial_index, run.trial_returns, rd)
    (data / "paper").mkdir()
    (data / "paper" / "ledger.parquet").write_bytes(b"ledger")
    (data / "paper" / "run.log").write_text("log")
    mp.undo()
    return tmp


def make_cfg(base_tree, tmp_path):
    root = tmp_path / "world"
    shutil.copytree(base_tree, root)
    cfg = load_config()
    cfg.update(data_dir=root / "data", project_root=root / "proj", holdout_cutoff_date=str(CUTOFF.date()),
               holdout_months=6, leverage_strategy=dict(STRAT),
               assets=[{"name": t, "ticker": t, "start": "2018-01-02", "trades_weekends": t == F.BTC}
                       for t in TICKERS], costs=dict(COSTS))
    return cfg


@pytest.fixture(autouse=True)
def never_load_the_real_config_in_the_cli(monkeypatch):
    """SAFETY: H.main() reads quant.config.load_config at call time.  A test that reaches the CLI
    without an explicit synthetic config must fail, never start a run on the REAL project."""
    def refuse(path=None):
        raise AssertionError("a test reached the CLI with the REAL config; patch load_config")
    monkeypatch.setattr("quant.config.load_config", refuse)


@pytest.fixture(autouse=True)
def never_run_final_on_the_real_project(monkeypatch):
    """SAFETY: run_final may be called by a test only with a synthetic config (project and data folder
    inside a tmp dir).  The real config object, or any config pointing at the real project/data folder,
    fails the test BEFORE run_final's own code runs."""
    orig = H.run_final

    def guarded(cfg, *a, **k):
        assert cfg is not REAL, "a test called run_final with the REAL config object"
        assert not H.resolves_to(cfg["project_root"], ROOT), "a test called run_final on the REAL project root"
        assert not H.resolves_to(cfg["data_dir"], ROOT / "data"), "a test called run_final on the REAL data folder"
        return orig(cfg, *a, **k)
    monkeypatch.setattr(H, "run_final", guarded)


def fake_dev(sig=False):
    """Stand-in for summary_numbers (the synthetic world has no significance tables)."""
    a = dict(spa_pvalue_vs_buy_and_hold=0.9, spa_pvalue_vs_vol_targeted_bh=0.8, dsr_global=0.02,
             dsr_excess_over_vol_targeted_bh=0.001, tstat=3.2, tstat_excess_over_vol_targeted_bh=-0.5,
             baseline_tstats={"buy_and_hold": 3.5, "vol_targeted_bh": 2.5})
    if sig:
        a.update(spa_pvalue_vs_buy_and_hold=0.01, spa_pvalue_vs_vol_targeted_bh=0.01, dsr_global=0.99,
                 dsr_excess_over_vol_targeted_bh=0.99, tstat=4.0, tstat_excess_over_vol_targeted_bh=3.5)
    return {"n_trials_global": 270, "assets": {t: dict(a) for t in TICKERS}}


@pytest.fixture
def cfg(base_tree, tmp_path, monkeypatch):
    monkeypatch.setattr(H, "BT", SMALL_BT)
    c = make_cfg(base_tree, tmp_path)
    monkeypatch.setattr("quant.config.load_config", lambda path=None: c)
    # the pre-registered constants point at the synthetic world (the real ones are tested separately)
    monkeypatch.setattr(H, "PROJECT_ROOT", Path(c["project_root"]))
    monkeypatch.setattr(H, "DATA_DIR", Path(c["data_dir"]))
    monkeypatch.setattr(H, "PINNED_CUTOFF", str(CUTOFF.date()))
    monkeypatch.setattr(H, "PINNED_MONTHS", 6)
    monkeypatch.setattr(H, "PINNED_STRATEGY", dict(STRAT))
    monkeypatch.setattr(H, "PREREGISTERED_SHA256", {"config.yaml": H.sha256_file(Path(c["project_root"]) / "config.yaml")})
    monkeypatch.setattr(H, "PREREGISTERED_CODE_SHA256", H.code_sha256(c))
    monkeypatch.setattr(H, "summary_numbers", lambda cfg=None: fake_dev())
    return c


def append_garbage(cfg, ticker, n=30):
    """Append wild rows AFTER the hold-out end to the features file."""
    p = Path(cfg["data_dir"]) / "features" / f"{ticker}.parquet"
    df = pd.read_parquet(p)
    rng = np.random.default_rng(5)
    g = df[df.index > H.real_window(cfg).end].copy()
    g.iloc[:, :] = rng.normal(size=g.shape) * 50
    g["target"] = rng.integers(0, 2, len(g)).astype(float)
    g["next_return"] = rng.normal(0, 0.2, len(g))
    pd.concat([df[df.index <= H.real_window(cfg).end], g]).to_parquet(p)


# =========================================================================== stage 3 equivalence
def assert_matches_stage3(cfg, ev, ticker, cutoff, strategy, tol=1e-9):
    """Predictions, positions and net returns of the harness equal the saved Stage 3 results for the
    same dates (the pseudo hold-out is a replay of the walk-forward on dates >= cutoff)."""
    rd = Path(cfg["data_dir"]) / "results"
    cutoff = pd.Timestamp(cutoff)
    pr = pd.read_parquet(rd / ticker / "predictions.parquet")
    pr = pr[(pr["window"] == "expanding") & (pr["model"] == strategy["model"]) & (pr["date"] >= cutoff)]
    pr = pr.sort_values("date").reset_index(drop=True)
    mine = ev.predictions[ev.predictions["model"] == strategy["model"]].sort_values("date").reset_index(drop=True)
    assert len(pr) > 100
    assert (mine["date"].to_numpy() == pr["date"].to_numpy()).all()
    for c in ("p_up", "base_rate", "y", "next_return"):
        assert np.abs(mine[c].to_numpy() - pr[c].to_numpy()).max() < tol, c
    idx = T.load_index(rd)
    canon = DD.canonical_config_id(idx, H.strategy_config_id(ticker, strategy))
    tr = T.load_trials(rd)
    tr = tr[(tr["config_id"] == canon) & (tr["date"] >= cutoff)].sort_values("date").reset_index(drop=True)
    assert (tr["date"].to_numpy() == pr["date"].to_numpy()).all()
    assert np.abs(mine["position"].to_numpy() - tr["position"].to_numpy()).max() < tol
    # net returns: identical to Stage 3 except day 1, where the harness starts in cash and pays for the
    # whole entry while Stage 3 pays only the change from the previous day's position.
    r = ev.returns[(ev.returns["subject"] == H.STRATEGY) & (ev.returns["scenario"] == "headline")]
    r = r.sort_values("date").reset_index(drop=True)
    net, saved = r["net_return"].to_numpy(), tr["net_return"].to_numpy()
    assert np.abs(net[1:] - saved[1:]).max() < tol
    pos0, cost = tr["position"].iloc[0], ev.trading["cost"].iloc[0]
    assert net[0] - saved[0] == pytest.approx(-cost * (abs(pos0) - abs(pos0 - ev.prior_position)), abs=tol)
    # baselines
    bl = pd.read_parquet(rd / ticker / "baselines.parquet")
    bl = bl[(bl["window"] == "expanding") & (bl["date"] >= cutoff)]
    for name in ("buy_and_hold", "vol_targeted_bh", "trend_200", "always_up"):
        b = bl[bl["name"] == name].sort_values("date")
        m = r_for(ev, name)
        assert (m["date"].to_numpy() == b["date"].to_numpy()).all()
        assert np.abs(m["position"].to_numpy() - b["position"].to_numpy()).max() < tol, name


def r_for(ev, name, scenario="headline"):
    r = ev.returns[(ev.returns["subject"] == name) & (ev.returns["scenario"] == scenario)]
    return r.sort_values("date").reset_index(drop=True)


@pytest.mark.parametrize("ticker", TICKERS)
def test_pseudo_holdout_replays_stage3_synthetic(cfg, ticker):
    pc = CUTOFF - pd.DateOffset(months=6)
    w = H.pseudo_window(cfg, pc)
    res = H.evaluate_all(cfg, w, assets=[ticker])
    ev = res.assets[ticker]
    assert ev.dates.min() >= pc and ev.dates.min() < pc + pd.Timedelta(days=5)
    assert_matches_stage3(cfg, ev, ticker, pc, STRAT)


REAL = load_config()
HAVE_REAL = ((REAL["data_dir"] / "results" / "SPY" / "predictions.parquet").exists()
             and (REAL["data_dir"] / "features" / "SPY.parquet").exists())


def _lgbm_ok():
    try:
        import lightgbm  # noqa: F401
        return True
    except Exception:
        return False


@pytest.mark.skipif(not (HAVE_REAL and _lgbm_ok()), reason="real data/results or lightgbm not available")
@pytest.mark.parametrize("ticker", ["SPY", "BTC-USD"])
def test_pseudo_holdout_replays_stage3_on_real_data(ticker, monkeypatch):
    """Pseudo hold-out = the last 24 months of DEVELOPMENT data (pseudo-cutoff 2022-11-06, data loaded
    only up to the real cutoff).  Predictions, positions and net returns equal the saved Stage 3
    results within 1e-9: models trained on the same rows give the same numbers, so no bug hides here.
    The only intended difference is day 1's entry cost (see assert_matches_stage3)."""
    monkeypatch.setattr(H, "BT", B.BTConfig())
    cfg = load_config()
    w = H.pseudo_window(cfg)
    assert w.cutoff == pd.Timestamp("2022-11-06") and w.end == pd.Timestamp("2024-11-06")
    res = H.evaluate_all(cfg, w, assets=[ticker])
    assert res.assets[ticker].dates.max() <= pd.Timestamp("2024-11-05")
    assert_matches_stage3(cfg, res.assets[ticker], ticker, w.cutoff, cfg["leverage_strategy"])


def test_walk_forward_first_date_default_unchanged_and_subset():
    df = make_df(900, 3)
    dev = df.iloc[:-1]
    cut = dev.index.max()
    full = B.walk_forward(dev, ["logreg"], SMALL_BT, cut, importance=False)
    part = B.walk_forward(dev, ["logreg"], SMALL_BT, cut, importance=False, first_date=dev.index[600])
    assert len(part.preds) < len(full.preds)
    merged = full.preds.merge(part.preds, on=["model", "date"], suffixes=("_f", "_p"))
    assert len(merged) == len(part.preds) and (merged["p_up_f"] == merged["p_up_p"]).all()
    assert part.preds["date"].max() == full.preds["date"].max()
    # every skipped block ends before first_date
    assert full.preds[~full.preds["date"].isin(part.preds["date"])]["date"].max() < dev.index[600]


# =========================================================================== leakage
def _eval(frame, hist, w):
    return H.evaluate_asset("SPY", frame, hist, STRAT, COSTS["SPY"], w, SMALL_BT)


def test_future_rows_and_changed_future_prices_do_not_change_earlier_output(cfg):
    df = pd.read_parquet(Path(cfg["data_dir"]) / "features" / "SPY.parquet")
    hist = H.load_history("SPY", STRAT, Path(cfg["data_dir"]) / "results", CUTOFF, CUTOFF)
    e1 = pd.Timestamp("2021-07-30")
    e2 = pd.Timestamp("2021-09-15")
    fa = df[df.index <= e1].copy()
    for c in F.TARGET_COLS:
        fa.iloc[-1, fa.columns.get_loc(c)] = np.nan
    fb = df[df.index <= e2].copy()
    rng = np.random.default_rng(1)
    late = fb.index > e1
    fb.loc[late, :] = rng.normal(size=(late.sum(), fb.shape[1])) * 30      # garbage future (incl. labels)
    fb.loc[late, "target"] = rng.integers(0, 2, late.sum()).astype(float)
    for c in F.TARGET_COLS:                     # like load_frame: the last row's label is unknown
        fb.iloc[-1, fb.columns.get_loc(c)] = np.nan
    a = _eval(fa, hist, H.Window(CUTOFF, e1, False))
    b = _eval(fb, hist, H.Window(CUTOFF, e2, False))
    pa = a.predictions[a.predictions["model"] == "logreg"].set_index("date")
    pb = b.predictions[b.predictions["model"] == "logreg"].set_index("date")
    common = pa.index[pa.index < e1]
    assert len(common) > 40
    for c in ("p_up", "base_rate", "position", "vol_weight", "margin_m"):
        assert np.array_equal(pa.loc[common, c].to_numpy(), pb.loc[common, c].to_numpy()), c


def test_rows_after_the_end_in_the_file_are_never_used(cfg):
    w = H.real_window(cfg)
    clean = H.evaluate_all(cfg, w, assets=["SPY", "BTC-USD"])
    for t in TICKERS:
        append_garbage(cfg, t)
    dirty = H.evaluate_all(cfg, w, assets=["SPY", "BTC-USD"])
    pd.testing.assert_frame_equal(clean.metrics, dirty.metrics)
    pd.testing.assert_frame_equal(clean.predictions, dirty.predictions)
    pd.testing.assert_frame_equal(clean.returns, dirty.returns)


def test_load_frame_truncates_and_blanks_last_label(cfg):
    append_garbage(cfg, "SPY")
    fr = H.load_frame("SPY", cfg, END)
    assert fr.index.max() <= END
    assert fr[F.TARGET_COLS].iloc[-1].isna().all() and fr[F.TARGET_COLS].iloc[:-1].notna().all().all()
    assert H.read_dates("SPY", cfg, END).max() == fr.index.max()
    with pytest.raises(H.HoldoutRefusal, match="after the allowed end"):
        H.assert_no_rows_after(pd.read_parquet(Path(cfg["data_dir"]) / "features" / "SPY.parquet"), END)


def test_scoring_starts_on_the_cutoff_row_and_history_is_strictly_before(cfg):
    w = H.real_window(cfg)
    ev = H.evaluate_all(cfg, w, assets=["BTC-USD"]).assets["BTC-USD"]
    assert ev.dates.min() == CUTOFF                    # the cutoff row (blank in dev) is scored here
    assert ev.dates.max() == END - pd.Timedelta(days=1)   # last row (end) has no label
    assert len(ev.dates) == (END - CUTOFF).days
    h = H.load_history("BTC-USD", STRAT, Path(cfg["data_dir"]) / "results", CUTOFF, CUTOFF)
    assert h.dates.max() == CUTOFF - pd.Timedelta(days=1)
    with pytest.raises(P.HoldoutError):
        H.load_history("BTC-USD", STRAT, Path(cfg["data_dir"]) / "results", END, CUTOFF - pd.Timedelta(days=30))


# =========================================================================== metric maths (independent)
def test_wilson_matches_statsmodels_and_edges():
    for k, n in ((0, 10), (10, 10), (250, 500), (3, 40), (321, 502)):
        lo, hi = H.wilson_interval(k, n)
        e = proportion_confint(k, n, alpha=0.05, method="wilson")
        assert (lo, hi) == pytest.approx(e, abs=1e-9)
    assert all(np.isnan(H.wilson_interval(1, 0)))


@pytest.fixture
def result(cfg):
    return H.evaluate_all(cfg, H.real_window(cfg))


def test_sharpe_ci_recomputed_from_the_returns_table(result):
    for t, ev in result.assets.items():
        ppy = M.periods_per_year(t)
        for subj in (H.STRATEGY, "buy_and_hold", "vol_targeted_bh"):
            r = r_for(ev, subj)["net_return"].to_numpy()
            sr_d = r.mean() / r.std(ddof=1)
            se = np.sqrt((1 + sr_d ** 2 / 2) / len(r)) * np.sqrt(ppy)
            row = ev.trading[(ev.trading["subject"] == subj) & (ev.trading["scenario"] == "headline")].iloc[0]
            assert row["sharpe"] == pytest.approx(sr_d * np.sqrt(ppy), rel=1e-10)
            assert row["sharpe_ci_low"] == pytest.approx(sr_d * np.sqrt(ppy) - 1.959964 * se, rel=1e-6)
            assert row["sharpe_ci_high"] == pytest.approx(sr_d * np.sqrt(ppy) + 1.959964 * se, rel=1e-6)
    assert np.sqrt(365 / 365) == 1.0 and M.periods_per_year("BTC-USD") == 365 and M.periods_per_year("SPY") == 252


def test_prediction_metrics_pt_brier_accuracy_recomputed(result):
    for t, ev in result.assets.items():
        d = ev.predictions[ev.predictions["model"] == "logreg"]
        y, p, b = d["y"].to_numpy(), d["p_up"].to_numpy(), d["base_rate"].to_numpy()
        row = ev.prediction[ev.prediction["model"] == "logreg"].iloc[0]
        n = len(y)
        z = (p >= b).astype(float)
        assert row["accuracy"] == pytest.approx(np.mean(z == y))
        # Pesaran-Timmermann (1992) by hand, with the base rates (share of up days and of up calls)
        py, pz = y.mean(), z.mean()
        phat = np.mean(z == y)
        pstar = py * pz + (1 - py) * (1 - pz)
        v = pstar * (1 - pstar) / n
        wv = ((2 * py - 1) ** 2 * pz * (1 - pz) + (2 * pz - 1) ** 2 * py * (1 - py)) / n   # statsmodels/PT form
        s = (phat - pstar) / np.sqrt(v - wv)
        assert row["pt_statistic"] == pytest.approx(s, rel=1e-9)
        assert row["pt_pvalue"] == pytest.approx(2 * (1 - stats.norm.cdf(abs(s))), rel=1e-6)
        assert row["share_up_days"] == pytest.approx(py) and row["share_up_predictions"] == pytest.approx(pz)
        # Brier skill vs climatology: training base rate (per row) and the evaluation-window base rate
        assert row["brier_skill"] == pytest.approx(1 - np.mean((p - y) ** 2) / np.mean((b - y) ** 2))
        assert row["brier_skill_vs_eval_base"] == pytest.approx(1 - np.mean((p - y) ** 2) / np.mean((py - y) ** 2))
        ll = lambda q: -np.mean(y * np.log(q) + (1 - y) * np.log(1 - q))
        assert row["log_loss_skill"] == pytest.approx(1 - ll(p) / ll(b))
        assert row["log_loss_skill_vs_eval_base"] == pytest.approx(1 - ll(p) / ll(np.full(n, py)))
        lo, hi = proportion_confint(int(round(row["accuracy"] * n)), n, alpha=0.05, method="wilson")
        assert (row["accuracy_ci_low"], row["accuracy_ci_high"]) == pytest.approx((lo, hi), abs=1e-9)
        assert row["auc"] == pytest.approx(stats.mannwhitneyu(p[y == 1], p[y == 0]).statistic / (y.sum() * (n - y.sum())))
        clim = ev.prediction[ev.prediction["model"] == "climatology"].iloc[0]
        assert clim["brier_skill"] == 0.0 and clim["accuracy"] == pytest.approx(py)


def test_sharpe_difference_matches_independent_formula():
    rng = np.random.default_rng(0)
    a = rng.normal(0.0006, 0.01, 500)
    b = 0.7 * a + rng.normal(0.0002, 0.006, 500)
    out = H.sharpe_difference(a, b, 252)
    C = np.cov(a, b, ddof=1)
    sa, sb = a.mean() / np.sqrt(C[0, 0]), b.mean() / np.sqrt(C[1, 1])
    rho = C[0, 1] / np.sqrt(C[0, 0] * C[1, 1])
    var = (2 - 2 * rho + 0.5 * (sa ** 2 + sb ** 2 - 2 * sa * sb * rho ** 2)) / 500
    assert out["diff"] == pytest.approx((sa - sb) * np.sqrt(252))
    assert out["se"] == pytest.approx(np.sqrt(var * 252))
    assert out["z"] == pytest.approx(out["diff"] / out["se"])
    assert out["lo"] == pytest.approx(out["diff"] - 1.959964 * out["se"], rel=1e-6)
    same = H.sharpe_difference(a, a, 252)
    assert same["diff"] == 0 and same["se"] == pytest.approx(0, abs=1e-12) or np.isnan(same["z"])
    assert np.isnan(H.sharpe_difference(a, np.zeros(500), 252)["diff"])
    # the table's strategy-vs-B&H numbers are the same function applied to the returns table
    # (checked in test_trading_table_diffs below)


def test_trading_table_diffs_and_scenarios(result):
    for t, ev in result.assets.items():
        ppy = M.periods_per_year(t)
        scen = B.cost_scenarios(t, COSTS[t])
        assert set(ev.trading["scenario"]) == set(scen)
        assert {"stress_cost" if t == F.BTC else "low_cost", "delay_1d", "headline"} == set(scen)
        for sname, (cost, delay) in scen.items():
            s = r_for(ev, H.STRATEGY, sname)
            for ref in H.REFERENCES:
                rr = r_for(ev, ref, sname)
                exp = H.sharpe_difference(s["net_return"], rr["net_return"], ppy)
                row = ev.trading[(ev.trading["subject"] == H.STRATEGY) & (ev.trading["scenario"] == sname)].iloc[0]
                assert row[f"sharpe_diff_vs_{ref}"] == pytest.approx(exp["diff"])
                assert row[f"sharpe_diff_se_vs_{ref}"] == pytest.approx(exp["se"])
                assert row["cost"] == cost and bool(row["delay"]) == delay
            if delay:
                h = r_for(ev, H.STRATEGY, "headline")
                assert s["position"].iloc[0] == 0 and np.array_equal(s["position"].to_numpy()[1:], h["position"].to_numpy()[:-1])
        # net = position*return - cost*|dposition| (fresh start in cash)
        h = r_for(ev, H.STRATEGY)
        nr = ev.predictions[ev.predictions["model"] == "logreg"].sort_values("date")["next_return"].to_numpy()
        pos = h["position"].to_numpy()
        assert np.allclose(h["net_return"], pos * nr - COSTS[t] * np.abs(np.diff(np.r_[0.0, pos])))
        assert (r_for(ev, "always_up")["position"] == 1).all()


def test_power_note_numbers():
    p = H.power_note(252, 504)
    assert p["sharpe_se_at_zero"] == pytest.approx(np.sqrt(0.5))
    assert p["min_detectable_sharpe_80pct_power"] == pytest.approx(2.4865 * np.sqrt(0.5), rel=1e-3)
    assert H.power_note(365, 730)["sharpe_se_at_zero"] == pytest.approx(np.sqrt(0.5))
    assert "catch failure" in p["note"] and "0.71" in p["note"]


def test_outputs_have_the_documented_content(result, tmp_path):
    files = H.write_outputs(result, tmp_path / "o")
    assert [f.name for f in files] == list(H.OUTPUT_FILES) and all(f.exists() for f in files)
    m = pd.read_parquet(tmp_path / "o" / "holdout_metrics.parquet")
    assert set(m["kind"]) == {"trading", "prediction"} and set(m["asset"]) == set(TICKERS)
    s = json.loads((tmp_path / "o" / "holdout_summary.json").read_text())
    assert s["cutoff"] == "2021-03-15" and s["end"] == "2021-09-15" and not s["dry_run"]
    assert "power" in s["assets"]["SPY"]
    md = (tmp_path / "o" / "holdout_results.md").read_text()
    assert "FINAL HOLD-OUT RESULTS" in md and "Wilson" in md and "catch failure" in md


# =========================================================================== guard: window / dry run
def test_pseudo_window_defaults_and_refusal(cfg):
    w = H.pseudo_window(cfg)
    assert w.pseudo and w.cutoff == CUTOFF - pd.DateOffset(months=6) and w.end == CUTOFF
    with pytest.raises(H.HoldoutRefusal, match="beyond the real hold-out cutoff"):
        H.pseudo_window(cfg, CUTOFF - pd.Timedelta(days=100))
    H.pseudo_window(cfg, CUTOFF - pd.DateOffset(months=6))      # boundary allowed


def test_dry_run_never_writes_the_real_folder_and_refuses_real_dir(cfg, tmp_path):
    res, out = H.run_dry(cfg, assets=["SPY"], out_dir=tmp_path / "scratch")
    assert res.window.pseudo and not H.holdout_dir(cfg).exists()
    assert (out / "holdout_results.md").exists()
    res2, out2 = H.run_dry(cfg, assets=["SPY"])                     # default: a fresh temp dir
    assert str(out2).startswith(__import__("tempfile").gettempdir()) or "holdout_dryrun_" in str(out2)
    shutil.rmtree(out2)
    for bad in (H.holdout_dir(cfg), H.holdout_dir(cfg) / "x"):
        with pytest.raises(H.HoldoutRefusal, match="never write"):
            H.run_dry(cfg, assets=["SPY"], out_dir=bad)
    assert not H.holdout_dir(cfg).exists()
    assert res.metrics["asset"].unique().tolist() == ["SPY"]
    assert res.assets["SPY"].dates.max() < CUTOFF                   # never beyond the pseudo end


def test_dry_run_loads_nothing_beyond_the_pseudo_end(cfg, monkeypatch):
    seen = []
    orig = H.load_frame

    def spy(t, c, end):
        f = orig(t, c, end)
        seen.append((t, end, f.index.max()))
        return f
    monkeypatch.setattr(H, "load_frame", spy)
    H.run_dry(cfg, assets=["SPY", "BTC-USD"], out_dir=cfg["data_dir"] / "scratch")
    assert seen and all(m <= e <= CUTOFF for _, e, m in seen)


# =========================================================================== guard: one shot
def marker(cfg):
    return json.loads(H.marker_path(cfg).read_text())


def test_run_requires_confirmation_and_writes_nothing(cfg):
    with pytest.raises(H.HoldoutRefusal, match="confirm-single-final-evaluation"):
        H.run_final(cfg, confirm=False)
    assert not H.holdout_dir(cfg).exists()
    assert H.main(["--run"]) == 2
    assert H.main(["--dry-run", "--confirm-single-final-evaluation"]) == 2
    assert H.main(["--run", "--confirm-single-final-evaluation", "--pseudo-cutoff", "2020-01-01"]) == 2


def test_successful_run_writes_marker_first_then_done_and_blocks_rerun(cfg, monkeypatch):
    seen = {}
    orig = H.evaluate_all

    def spy(*a, **k):
        seen["marker"] = marker(cfg)               # the marker already exists when evaluation starts
        assert not any((H.holdout_dir(cfg) / n).exists() for n in H.OUTPUT_FILES)
        return orig(*a, **k)
    monkeypatch.setattr(H, "evaluate_all", spy)
    res = H.run_final(cfg, confirm=True)
    m0 = seen["marker"]
    assert m0["status"] == "started" and m0["holdout_cutoff"] == "2021-03-15" and m0["holdout_end"] == "2021-09-15"
    assert m0["started_utc"] and set(m0["per_asset"]) == set(TICKERS)
    assert m0["per_asset"]["BTC-USD"]["n_rows_in_window"] == (END - CUTOFF).days + 1
    assert m0["per_asset"]["BTC-USD"]["first_decision_date"] == "2021-03-15"
    keys = set(m0["manifest"])
    assert {"project/quant/a.py", "project/config.yaml", "project/uv.lock",
            "data/results/trials.parquet", "data/results/SPY/predictions.parquet",
            "data/paper/ledger.parquet", "data/features/SPY.parquet"} <= keys
    assert m0["manifest_sha256"] == H.manifest_digest(m0["manifest"])
    done = marker(cfg)
    assert done["status"] == "done" and done["freeze_verified"] and set(done["outputs"]) == set(H.OUTPUT_FILES)
    assert all((H.holdout_dir(cfg) / n).exists() for n in H.OUTPUT_FILES)
    assert not res.window.pseudo
    with pytest.raises(H.HoldoutRefusal, match="already evaluated or attempted") as e:
        H.run_final(cfg, confirm=True)
    assert "by hand" in str(e.value) and "PROGRESS.md" in str(e.value)


def test_cli_rerun_after_done_exit_code_2(cfg, monkeypatch, capsys):
    H.run_final(cfg, confirm=True)
    assert H.main(["--run", "--confirm-single-final-evaluation"]) == 2
    assert "ERROR:" in capsys.readouterr().err


def test_crash_midway_leaves_failed_marker_and_blocks_rerun(cfg, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("model exploded")
    real_eval = H.evaluate_asset
    monkeypatch.setattr(H, "evaluate_asset", boom)
    with pytest.raises(RuntimeError, match="model exploded"):
        H.run_final(cfg, confirm=True)
    m = marker(cfg)
    assert m["status"] == "failed" and "model exploded" in m["error"] and "Traceback" in m["traceback"]
    assert "PROGRESS.md" in m["manual_step"]
    assert not any((H.holdout_dir(cfg) / n).exists() for n in H.OUTPUT_FILES)
    monkeypatch.setattr(H, "evaluate_asset", real_eval)
    with pytest.raises(H.HoldoutRefusal, match="status='failed'"):
        H.run_final(cfg, confirm=True)             # silently re-running is refused
    # the documented manual step re-opens it (and only that)
    H.marker_path(cfg).rename(H.holdout_dir(cfg) / "HOLDOUT_FAILED_manual.json")
    res = H.run_final(cfg, confirm=True)
    assert marker(cfg)["status"] == "done"
    assert marker(cfg)["earlier_failed_attempts"] == ["HOLDOUT_FAILED_manual.json"]
    assert "ATTEMPT 2: NOT a single evaluation" in res.markdown and "HOLDOUT_FAILED_manual.json" in res.markdown


def test_killed_run_marker_stays_started_and_blocks(cfg, monkeypatch):
    def kill(*a, **k):
        raise KeyboardInterrupt
    monkeypatch.setattr(H, "evaluate_all", kill)
    with pytest.raises(KeyboardInterrupt):
        H.run_final(cfg, confirm=True)
    assert marker(cfg)["status"] == "failed"
    H.update_marker(H.marker_path(cfg), status="started")      # simulate kill -9 before the update
    with pytest.raises(H.HoldoutRefusal, match="status='started'"):
        H.run_final(cfg, confirm=True)


def test_outputs_without_marker_block_a_run(cfg):
    (H.holdout_dir(cfg)).mkdir(parents=True)
    (H.holdout_dir(cfg) / "holdout_metrics.parquet").write_bytes(b"x")
    with pytest.raises(H.HoldoutRefusal, match="without a marker"):
        H.run_final(cfg, confirm=True)
    assert not H.marker_path(cfg).exists()


def test_marker_creation_is_exclusive_and_atomic(tmp_path):
    p = tmp_path / "h" / H.MARKER_NAME
    H.create_marker_exclusive(p, {"status": "started"})
    assert json.loads(p.read_text()) == {"status": "started"}
    with pytest.raises(H.HoldoutRefusal):
        H.create_marker_exclusive(p, {"status": "other"})
    assert json.loads(p.read_text())["status"] == "started"
    assert [x.name for x in p.parent.iterdir()] == [H.MARKER_NAME]      # no temp files left


def test_run_refuses_if_data_do_not_cover_the_holdout_and_burns_nothing(cfg):
    p = Path(cfg["data_dir"]) / "features" / "BTC-USD.parquet"
    df = pd.read_parquet(p)
    df[df.index < END - pd.Timedelta(days=5)].to_parquet(p)               # data stop 5 days early
    with pytest.raises(H.HoldoutRefusal, match="do not cover the whole hold-out"):
        H.run_final(cfg, confirm=True)
    assert not H.holdout_dir(cfg).exists()                                 # no marker: the shot is not used
    # a hole in the middle is caught as well
    cfg2_df = df.drop(pd.Timestamp("2021-06-10"))
    cfg2_df = pd.concat([cfg2_df, pd.read_parquet(Path(cfg["data_dir"]) / "features" / "SPY.parquet").iloc[:0]])
    s = Path(cfg["data_dir"]) / "features" / "SPY.parquet"
    spy = pd.read_parquet(s)
    spy.drop(pd.Timestamp("2021-06-10")).to_parquet(s)
    with pytest.raises(H.HoldoutRefusal, match="SPY: the data do not cover"):
        H.check_coverage("SPY", H.read_dates("SPY", cfg, END), H.real_window(cfg))
    assert H.check_coverage("BTC-USD", pd.DatetimeIndex(pd.date_range("2018-01-01", END)), H.real_window(cfg))["n_scorable_rows"] == (END - CUTOFF).days


def test_freeze_violation_during_run_marks_failed(cfg, monkeypatch):
    orig = H.write_outputs

    def tamper(res, d):
        out = orig(res, d)
        (Path(cfg["data_dir"]) / "results" / "SPY" / "predictions.parquet").write_bytes(b"tampered")
        return out
    monkeypatch.setattr(H, "write_outputs", tamper)
    with pytest.raises(H.FreezeViolation, match="predictions.parquet"):
        H.run_final(cfg, confirm=True)
    m = marker(cfg)
    assert m["status"] == "failed" and "FreezeViolation" in m["error"]


def test_run_leaves_frozen_inputs_untouched(cfg):
    before = H.build_manifest(cfg)
    H.run_final(cfg, confirm=True)
    assert H.diff_manifests(before, H.build_manifest(cfg)) == []


# =========================================================================== freeze manifest
def test_manifest_contents_exclusions_and_diff(cfg):
    m = H.build_manifest(cfg)
    root, data = Path(cfg["project_root"]), Path(cfg["data_dir"])
    assert m["project/quant/a.py"] == H.sha256_file(root / "quant" / "a.py")
    assert "data/paper/run.log" not in m
    (data / "results" / "holdout").mkdir()
    (data / "results" / "holdout" / "holdout_x.parquet").write_bytes(b"1")
    (data / "results" / "SPY" / "stale.tmp").write_bytes(b"1")
    assert H.build_manifest(cfg) == m                           # hold-out folder, run.log, *.tmp ignored
    (data / "paper" / "run.log").write_text("more log")
    assert H.build_manifest(cfg) == m
    # every kind of change is reported
    (root / "quant" / "a.py").write_text("x = 2\n")
    (data / "results" / "SPY" / "new.parquet").write_bytes(b"n")
    (data / "paper" / "ledger.parquet").unlink()
    (root / "uv.lock").write_text("other\n")
    d = H.diff_manifests(m, H.build_manifest(cfg))
    assert d == sorted(["changed: project/quant/a.py", "added: data/results/SPY/new.parquet",
                        "removed: data/paper/ledger.parquet", "changed: project/uv.lock"])
    assert H.diff_manifests(m, m) == []
    assert H.manifest_digest(m) != H.manifest_digest({**m, "project/uv.lock": "x"})


def test_real_manifest_covers_code_config_lock_results_and_ledger():
    if not HAVE_REAL:
        pytest.skip("real data not present")
    m = H.build_manifest(REAL)
    assert "project/quant/holdout.py" in m and "project/config.yaml" in m and "project/uv.lock" in m
    assert "data/results/trials.parquet" in m and "data/results/SPY/significance.parquet" in m
    assert all(not k.startswith("data/results/holdout") for k in m)
    if (REAL["data_dir"] / "paper" / "ledger.parquet").exists():
        assert "data/paper/ledger.parquet" in m and "data/paper/state.json" in m


# =========================================================================== dashboard isolation
def test_dashboard_loaders_ignore_the_holdout_folder(cfg):
    DD_cfg = cfg
    rd = Path(cfg["data_dir"]) / "results"
    sig0 = DD.data_signature(DD_cfg)
    res0 = DD.load_asset_results("SPY", DD_cfg)
    idx0 = DD.load_trial_index(DD_cfg)
    H.run_final(cfg, confirm=True)
    hd = rd / "holdout"
    assert hd.exists()
    (hd / "SPY").mkdir()
    (hd / "SPY" / "predictions.parquet").write_bytes(b"not even parquet")
    assert DD.data_signature(DD_cfg) == sig0
    res1 = DD.load_asset_results("SPY", DD_cfg)
    pd.testing.assert_frame_equal(res0.predictions, res1.predictions)
    pd.testing.assert_frame_equal(idx0, DD.load_trial_index(DD_cfg))
    assert DD.tickers(DD_cfg) == TICKERS
    assert pd.read_parquet(hd / "holdout_predictions.parquet")["date"].max() > CUTOFF   # hold-out dates exist there
    assert res1.predictions["date"].max() <= CUTOFF                                     # but not in the dev loaders


def test_dev_loader_guard_raises_on_a_holdout_date(cfg):
    p = Path(cfg["data_dir"]) / "results" / "SPY" / "predictions.parquet"
    df = pd.read_parquet(p)
    late = df.iloc[[-1]].copy()
    late["date"] = CUTOFF + pd.Timedelta(days=3)
    pd.concat([df, late]).to_parquet(p, index=False)
    with pytest.raises(P.HoldoutError):
        DD.load_asset_results("SPY", cfg)


def test_no_module_enumerates_the_results_folder():
    pat = re.compile(r"\.(glob|rglob|iterdir|walk)\(|os\.listdir|os\.scandir")
    hits = []
    for f in [*(ROOT / "quant").glob("*.py"), *(ROOT / "app").glob("*.py")]:
        if f.name == "holdout.py":                  # the manifest builder enumerates on purpose
            continue
        for i, line in enumerate(f.read_text().splitlines(), 1):
            if pat.search(line) and "#" not in line.split(pat.search(line).group(0))[0]:
                hits.append(f"{f.name}:{i}")
    assert hits == [], f"loaders must read named per-asset files, not enumerate folders: {hits}"


# =========================================================================== significance summary
@pytest.mark.skipif(not HAVE_REAL, reason="real results not present")
def test_summary_numbers_match_saved_results_and_dashboard_summary():
    s = H.summary_numbers(REAL)
    assert s["n_trials_global"] == 270
    ts = DD.trials_summary(REAL)
    assert ts["n_trials_global"] == s["n_trials_global"]
    for t, a in s["assets"].items():
        e = ts["assets"][t]
        assert a["dsr_asset"] == pytest.approx(e["dsr_asset"]) and a["dsr_global"] == pytest.approx(e["dsr_global"])
        assert a["psr"] == pytest.approx(e["psr"]) and a["n_trials_global"] == 270
        sig = pd.read_parquet(REAL["data_dir"] / "results" / t / "significance.parquet")
        c = a["canonical_id"]
        r = sig[(sig["subject"] == c) & (sig["test"] == "tstat_vs_3")].iloc[0]
        assert a["tstat"] == pytest.approx(r["statistic"]) and a["clears_t_gt_3"] == (r["statistic"] > 3)
        spa = sig[sig["test"] == "spa"].set_index("reference")["pvalue"]
        assert a["spa_pvalue_vs_buy_and_hold"] == spa["buy_and_hold"]
        assert a["spa_pvalue_vs_vol_targeted_bh"] == spa["vol_targeted_bh"]
        assert set(a["baseline_tstats"]) == {"buy_and_hold", "vol_targeted_bh"}
    spy = s["assets"]["SPY"]
    assert spy["dsr_asset"] == pytest.approx(0.993299, abs=1e-5) and spy["dsr_global"] == pytest.approx(0.016965, abs=1e-5)
    assert spy["tstat"] == pytest.approx(3.235262, abs=1e-5) and spy["clears_t_gt_3"]
    assert spy["spa_pvalue_vs_buy_and_hold"] == pytest.approx(0.998)
    # the dashboard badge reads the same pooled DSR
    from quant import explain as E
    assert E.evidence_badge("SPY", REAL)["inputs"]["dsr_global"] == pytest.approx(spy["dsr_global"])


def test_significance_summary_cli_prints_json(capsys, monkeypatch):
    if not HAVE_REAL:
        pytest.skip("real results not present")
    monkeypatch.setattr("quant.config.load_config", lambda path=None: REAL)
    assert H.main(["--significance-summary"]) == 0
    assert json.loads(capsys.readouterr().out)["n_trials_global"] == 270


# =========================================================================== round-1 fixes
# ---- 1. config / pre-registration guards
def test_run_refuses_a_foreign_config_flag(cfg, tmp_path):
    other = tmp_path / "other.yaml"
    other.write_text("x: 1\n")
    assert H.main(["--run", "--confirm-single-final-evaluation", "--config", str(other)]) == 2
    assert not H.holdout_dir(cfg).exists()
    # the project's own config file is accepted (absolute or relative to the project root)
    assert H.main(["--run", "--confirm-single-final-evaluation", "--config",
                   str(Path(cfg["project_root"]) / "config.yaml")]) == 0


@pytest.mark.parametrize("change,match", [
    ({"holdout_cutoff_date": "2024-01-01"}, "holdout_cutoff_date"),
    ({"holdout_months": 12}, "holdout_months"),
    ({"leverage_strategy": {**STRAT, "model": "rf"}}, "pre-registered strategy"),
    ({"leverage_strategy": {**STRAT, "m_setting": "0.01"}}, "pre-registered strategy"),
    ({"leverage_strategy": {**STRAT, "vol_targeted": False}}, "pre-registered strategy"),
    ({"data_dir": "/somewhere/else"}, "not the project's data"),
    ({"project_root": "/somewhere/else"}, "project_root"),
    ({"leverage_strategy": {**STRAT, "extra_key": 1}}, "pre-registered strategy"),
    ({"leverage_strategy": {k: v for k, v in STRAT.items() if k != "vol_targeted"}}, "pre-registered strategy|vol_targeted"),
])
def test_run_final_asserts_the_pinned_values_and_burns_nothing(cfg, change, match):
    cfg.update({k: (Path(v) if k == "data_dir" else v) for k, v in change.items()})
    with pytest.raises(H.HoldoutRefusal, match=match):
        H.run_final(cfg, confirm=True)
    assert not H.holdout_dir(cfg).exists()


def test_preregistered_config_hash_is_checked_before_the_marker(cfg, capsys):
    assert H.main(["--verify-preregistration"]) == 0
    assert json.loads(capsys.readouterr().out)["matches_preregistered"] is True
    (Path(cfg["project_root"]) / "config.yaml").write_text("a: 2\n")        # edited after pre-registration
    with pytest.raises(H.HoldoutRefusal, match="changed since pre-registration"):
        H.run_final(cfg, confirm=True)
    assert not H.holdout_dir(cfg).exists()
    assert H.main(["--verify-preregistration"]) == 2


def test_preregistered_code_hash_is_checked_before_the_marker(cfg, capsys):
    root = Path(cfg["project_root"])
    assert H.main(["--verify-preregistration"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["code_sha256_excl_holdout"] == H.PREREGISTERED_CODE_SHA256 == H.code_sha256(cfg)
    h0 = H.code_sha256(cfg)
    (root / "quant" / "holdout.py").write_text("# the harness itself is excluded from the code hash\n")
    assert H.code_sha256(cfg) == h0
    (root / "quant" / "a.py").write_text("x = 99\n")                      # any other quant file changes it
    assert H.code_sha256(cfg) != h0
    with pytest.raises(H.HoldoutRefusal, match=r"quant/\*\.py .* changed since pre-registration"):
        H.run_final(cfg, confirm=True)
    assert not H.holdout_dir(cfg).exists()
    assert H.main(["--verify-preregistration"]) == 2
    (root / "quant" / "a.py").write_text("x = 1\n")
    (root / "quant" / "new.py").write_text("z = 3\n")                      # a new file changes it too
    assert H.code_sha256(cfg) != h0


def test_marker_records_the_code_hashes(cfg):
    (Path(cfg["project_root"]) / "quant" / "holdout.py").write_text("# stand-in\n")
    H.run_final(cfg, confirm=True)
    m = marker(cfg)
    assert m["preregistered_code_sha256"] == H.code_sha256(cfg)
    assert m["holdout_py_sha256"] == H.sha256_file(Path(cfg["project_root"]) / "quant" / "holdout.py")


def test_real_config_matches_the_preregistered_hash_and_pins():
    assert H.sha256_file(ROOT / "config.yaml") == H.PREREGISTERED_SHA256["config.yaml"]
    assert H.PINNED_CUTOFF == REAL["holdout_cutoff_date"] == "2024-11-06"
    assert H.PINNED_MONTHS == REAL["holdout_months"] == 24
    H.assert_pinned_cfg(REAL)                                    # the real config satisfies every pin
    assert H.code_sha256() == H.PREREGISTERED_CODE_SHA256        # quant/*.py (excl. holdout.py) as pre-registered
    assert H.verify_preregistration()["code_sha256"] == H.PREREGISTERED_CODE_SHA256


def test_dry_run_cannot_reach_holdout_rows_even_with_an_edited_config(cfg):
    c2 = dict(cfg, holdout_cutoff_date="2030-01-01", holdout_months=48)
    with pytest.raises(H.HoldoutRefusal, match="beyond the real hold-out cutoff"):
        H.pseudo_window(c2, CUTOFF - pd.Timedelta(days=30))
    w = H.pseudo_window(c2)                                      # default uses the PINNED constants
    assert w.end == CUTOFF
    with pytest.raises(H.HoldoutRefusal, match="may not end after the pinned cutoff"):
        H.evaluate_all(c2, H.Window(CUTOFF - pd.Timedelta(days=30), CUTOFF + pd.Timedelta(days=30), True))


# ---- 2. interpretation rule
def test_hold_out_flags_only_flag_failures():
    ok = dict(sharpe_diff_vs_vol_targeted_bh_ci=[-0.5, 1.5], pt_statistic=1.2, pt_pvalue=0.03,
              brier_loss_diff_ci=[-0.01, 0.002])
    f = H.hold_out_flags(ok)
    assert not f["any_failure"] and f["reasons"] == [] and f["undetermined"] == []
    assert H.hold_out_flags(dict(ok, sharpe_diff_vs_vol_targeted_bh_ci=[-2.0, -0.1]))["sharpe_vs_vt_failure"]
    assert not H.hold_out_flags(dict(ok, sharpe_diff_vs_vol_targeted_bh_ci=[-2.0, 0.0]))["sharpe_vs_vt_failure"]
    assert H.hold_out_flags(dict(ok, pt_statistic=-2.0, pt_pvalue=0.01))["pt_negative_skill"]
    assert not H.hold_out_flags(dict(ok, pt_statistic=-2.0, pt_pvalue=0.2))["pt_negative_skill"]
    assert not H.hold_out_flags(dict(ok, pt_statistic=2.0, pt_pvalue=0.01))["pt_negative_skill"]   # positive skill is not a flag
    assert H.hold_out_flags(dict(ok, brier_loss_diff_ci=[0.001, 0.01]))["brier_worse_than_base"]
    assert not H.hold_out_flags(dict(ok, brier_loss_diff_ci=[-0.001, 0.01]))["brier_worse_than_base"]
    nan = H.hold_out_flags({})
    assert not nan["any_failure"] and len(nan["undetermined"]) == 3
    # good hold-out numbers never create a flag, however good (the hold-out cannot upgrade)
    great = H.hold_out_flags(dict(sharpe_diff_vs_vol_targeted_bh_ci=[3.0, 5.0], pt_statistic=9.0, pt_pvalue=0.0,
                                  brier_loss_diff_ci=[-0.5, -0.4]))
    assert not great["any_failure"]


def _flags(**kw):
    ok = dict(sharpe_diff_vs_vol_targeted_bh_ci=[-0.5, 1.5], pt_statistic=1.2, pt_pvalue=0.03,
              brier_loss_diff_ci=[-0.01, 0.002])
    return H.hold_out_flags(dict(ok, **kw))


NOFAIL = H.hold_out_flags({})
FAIL = dict(NOFAIL, any_failure=True, reasons=["x"])
NONE_, WEAK_, EVID_ = "no evidence of an edge", "weak evidence", "evidence"


def _v(over=None, base="good", flags=NOFAIL):
    d = fake_dev(sig=(base == "good"))["assets"]["SPY"]
    return H.final_verdict(dict(d, **(over or {})), flags)[0]


def test_final_verdict_table():
    dev = fake_dev()["assets"]["SPY"]
    good = fake_dev(sig=True)["assets"]["SPY"]
    # the strategy's own t > 3 alone (SPY/QQQ in real life) is NOT weak evidence any more
    v, why = H.final_verdict(dev, NOFAIL)
    assert v == NONE_ and sum(r.startswith("CLEARED") for r in why) == 1
    assert any("no baseline test cleared" in r for r in why)
    assert H.final_verdict(good, NOFAIL)[0] == EVID_
    v, why = H.final_verdict(good, FAIL)                          # a failure signal forces the default
    assert v == NONE_ and any("downgrades" in r for r in why)
    assert H.final_verdict(dev, FAIL)[0] == NONE_
    assert H.final_verdict(dict(dev, tstat=1.0), NOFAIL)[0] == NONE_
    # each of the five tests missing/NaN: the baseline ones leave "weak evidence", the necessary ones too
    # (the other baseline tests are still cleared), none of them can keep "evidence"
    for k in ("spa_pvalue_vs_buy_and_hold", "spa_pvalue_vs_vol_targeted_bh", "dsr_global",
              "dsr_excess_over_vol_targeted_bh", "tstat", "tstat_excess_over_vol_targeted_bh"):
        assert _v({k: float("nan")}) == WEAK_, k
    # only necessary conditions cleared (all baseline tests failed): default, never weak
    only_nec = dict(good, spa_pvalue_vs_buy_and_hold=0.9, dsr_excess_over_vol_targeted_bh=0.1,
                    tstat_excess_over_vol_targeted_bh=-1.0)
    assert H.final_verdict(only_nec, NOFAIL)[0] == NONE_
    # one baseline test alone is enough for "weak evidence", even when both necessary conditions fail
    for k, bad in (("spa_pvalue_vs_buy_and_hold", 0.9), ("dsr_excess_over_vol_targeted_bh", 0.1),
                   ("tstat_excess_over_vol_targeted_bh", 1.0)):
        keep = {"spa_pvalue_vs_buy_and_hold", "dsr_excess_over_vol_targeted_bh", "tstat_excess_over_vol_targeted_bh"} - {k}
        base_fail = {x: (0.9 if x.startswith("spa") else 0.1 if x.startswith("dsr") else -1.0) for x in keep}
        one = dict(good, tstat=1.0, dsr_global=0.1, **{**base_fail, "spa_pvalue_vs_vol_targeted_bh": 0.9})
        one[k] = {"spa_pvalue_vs_buy_and_hold": 0.01, "dsr_excess_over_vol_targeted_bh": 0.99,
                  "tstat_excess_over_vol_targeted_bh": 3.5}[k]
        if k.startswith("spa"):
            one["spa_pvalue_vs_vol_targeted_bh"] = 0.01
        assert H.final_verdict(one, NOFAIL)[0] == WEAK_, k
        assert H.final_verdict(one, FAIL)[0] == NONE_, k
    assert _v({"spa_pvalue_vs_vol_targeted_bh": 0.2}) == WEAK_          # SPA must clear BOTH baselines
    assert _v({"dsr_global": 0.949}) == WEAK_
    assert _v({"tstat": 3.0}) == WEAK_                                   # strictly > 3
    assert _v({"dsr_global": 0.95}) == EVID_                             # >= 0.95


@pytest.mark.parametrize("over,expected", [
    # SPA: strictly below 0.05 for BOTH baselines (0.05 and 0.10 are not enough)
    ({"spa_pvalue_vs_buy_and_hold": 0.05}, WEAK_), ({"spa_pvalue_vs_vol_targeted_bh": 0.05}, WEAK_),
    ({"spa_pvalue_vs_buy_and_hold": 0.0499}, EVID_), ({"spa_pvalue_vs_vol_targeted_bh": 0.0499}, EVID_),
    ({"spa_pvalue_vs_buy_and_hold": 0.10}, WEAK_), ({"spa_pvalue_vs_vol_targeted_bh": 0.10}, WEAK_),
    ({"spa_pvalue_vs_buy_and_hold": 0.07}, WEAK_),
    # excess t: strictly > 3
    ({"tstat_excess_over_vol_targeted_bh": 3.0}, WEAK_), ({"tstat_excess_over_vol_targeted_bh": 3.0001}, EVID_),
    ({"tstat_excess_over_vol_targeted_bh": 2.9999}, WEAK_),
    # the strategy's own t: strictly > 3 (necessary condition: failing it drops "evidence" to "weak evidence")
    ({"tstat": 3.0}, WEAK_), ({"tstat": 3.0001}, EVID_), ({"tstat": 2.9999}, WEAK_),
    # DSR: >= 0.95, global and excess
    ({"dsr_global": 0.95}, EVID_), ({"dsr_global": 0.9499}, WEAK_),
    ({"dsr_excess_over_vol_targeted_bh": 0.95}, EVID_), ({"dsr_excess_over_vol_targeted_bh": 0.9499}, WEAK_),
])
def test_final_verdict_threshold_boundaries(over, expected):
    assert _v(over) == expected


@pytest.mark.parametrize("over,expected", [
    # with ONLY that baseline test cleared (others failing), the exact boundary decides weak vs default
    ({"spa_pvalue_vs_buy_and_hold": 0.05, "spa_pvalue_vs_vol_targeted_bh": 0.05}, NONE_),
    ({"spa_pvalue_vs_buy_and_hold": 0.0499, "spa_pvalue_vs_vol_targeted_bh": 0.0499}, WEAK_),
    ({"spa_pvalue_vs_buy_and_hold": 0.10, "spa_pvalue_vs_vol_targeted_bh": 0.10}, NONE_),
    ({"dsr_excess_over_vol_targeted_bh": 0.95}, WEAK_), ({"dsr_excess_over_vol_targeted_bh": 0.9499}, NONE_),
    ({"tstat_excess_over_vol_targeted_bh": 3.0}, NONE_), ({"tstat_excess_over_vol_targeted_bh": 3.0001}, WEAK_),
])
def test_weak_evidence_boundaries_one_baseline_test_at_a_time(over, expected):
    fail_all = dict(spa_pvalue_vs_buy_and_hold=0.9, spa_pvalue_vs_vol_targeted_bh=0.9,
                    dsr_excess_over_vol_targeted_bh=0.1, tstat_excess_over_vol_targeted_bh=-1.0)
    assert _v({**fail_all, **over}) == expected
    assert _v({**fail_all, **over}, flags=FAIL) == NONE_              # a failure signal always wins


def test_strategy_t_and_global_dsr_alone_never_give_weak_evidence():
    fail_all = dict(spa_pvalue_vs_buy_and_hold=0.9, spa_pvalue_vs_vol_targeted_bh=0.9,
                    dsr_excess_over_vol_targeted_bh=0.1, tstat_excess_over_vol_targeted_bh=-1.0)
    for extra in ({"tstat": 9.0}, {"dsr_global": 0.999}, {"tstat": 9.0, "dsr_global": 0.999}):
        v, why = H.final_verdict(dict(fake_dev(sig=True)["assets"]["SPY"], **fail_all, **extra), NOFAIL)
        assert v == NONE_
    # and a missing necessary condition can never be rescued by clearing every baseline test
    assert _v({"tstat": float("nan")}) == WEAK_ and _v({"dsr_global": float("nan")}) == WEAK_


def test_failure_reason_line_text_and_single_flag_forces_default():
    good = fake_dev(sig=True)["assets"]["SPY"]
    v, why = H.final_verdict(good, FAIL)
    assert v == NONE_ and "hold-out failure signal(s): x" in why
    assert "a hold-out failure signal downgrades the verdict to the default" in why
    multi = dict(NOFAIL, any_failure=True, reasons=["a", "b"])
    assert "hold-out failure signal(s): a; b" in H.final_verdict(good, multi)[1]
    assert not any("failure signal" in r for r in H.final_verdict(good, NOFAIL)[1])
    # a failure without any cleared baseline test: default, and no "downgrades" claim
    v, why = H.final_verdict(fake_dev()["assets"]["SPY"], FAIL)
    assert v == NONE_ and not any("downgrades" in r for r in why)
    # any_failure alone drives the verdict (not the reasons list)
    assert H.final_verdict(good, dict(NOFAIL, any_failure=True))[0] == NONE_
    assert H.final_verdict(good, dict(NOFAIL, reasons=["x"]))[0] == EVID_


def test_the_three_real_flags_each_force_the_default_through_hold_out_flags():
    good = fake_dev(sig=True)["assets"]["SPY"]
    cases = {"sharpe_vs_vt_failure": dict(sharpe_diff_vs_vol_targeted_bh_ci=[-2.0, -0.1]),
             "pt_negative_skill": dict(pt_statistic=-2.0, pt_pvalue=0.01),
             "brier_worse_than_base": dict(brier_loss_diff_ci=[0.001, 0.01])}
    assert H.final_verdict(good, _flags())[0] == EVID_
    for name, kw in cases.items():
        f = _flags(**kw)
        assert f[name] and f["any_failure"] and sum(bool(f[n]) for n in cases) == 1, name
        v, why = H.final_verdict(good, f)
        assert v == NONE_ and any(r.startswith("hold-out failure signal(s): ") for r in why), name
        assert len(f["reasons"]) == 1 and f["reasons"][0] in why[-2], name


@pytest.mark.parametrize("kw,name,expected", [
    (dict(sharpe_diff_vs_vol_targeted_bh_ci=[-2.0, 0.0]), "sharpe_vs_vt_failure", False),      # upper bound exactly 0
    (dict(sharpe_diff_vs_vol_targeted_bh_ci=[-2.0, -1e-9]), "sharpe_vs_vt_failure", True),
    (dict(pt_statistic=0.0, pt_pvalue=0.01), "pt_negative_skill", False),                      # statistic exactly 0
    (dict(pt_statistic=-1e-9, pt_pvalue=0.01), "pt_negative_skill", True),
    (dict(pt_statistic=-2.0, pt_pvalue=0.05), "pt_negative_skill", False),                     # p exactly 0.05
    (dict(pt_statistic=-2.0, pt_pvalue=0.0499), "pt_negative_skill", True),
    (dict(brier_loss_diff_ci=[0.0, 0.01]), "brier_worse_than_base", False),                    # lower bound exactly 0
    (dict(brier_loss_diff_ci=[1e-9, 0.01]), "brier_worse_than_base", True),
])
def test_hold_out_flag_boundaries(kw, name, expected):
    f = _flags(**kw)
    assert f[name] is expected and f["any_failure"] is expected
    assert (len(f["reasons"]) == 1) is expected


def test_hold_out_flags_any_not_all_and_reason_text():
    f = _flags(pt_statistic=-2.0, pt_pvalue=0.01, brier_loss_diff_ci=[0.001, 0.01])
    assert f["pt_negative_skill"] and f["brier_worse_than_base"] and not f["sharpe_vs_vt_failure"]
    assert f["any_failure"] is True and len(f["reasons"]) == 2                  # any, not all
    assert f["reasons"][0].startswith("Pesaran-Timmermann shows negative skill (statistic -2.00, p 0.010)")
    assert f["reasons"][1].startswith("Brier score significantly worse than the evaluation-window base rate")
    s = _flags(sharpe_diff_vs_vol_targeted_bh_ci=[-2.0, -0.1])
    assert s["reasons"] == ["Sharpe difference vs vol-targeted B&H has its 95% CI entirely below 0 [-2.00, -0.10]"]
    allf = _flags(sharpe_diff_vs_vol_targeted_bh_ci=[-2.0, -0.1], pt_statistic=-2.0, pt_pvalue=0.01,
                  brier_loss_diff_ci=[0.001, 0.01])
    assert allf["any_failure"] and len(allf["reasons"]) == 3


def test_headline_verdict_is_the_weakest_per_asset_verdict():
    hv = H.headline_verdict
    assert hv({"A": EVID_, "B": EVID_, "C": EVID_, "D": EVID_}) == EVID_
    assert hv({"A": EVID_, "B": WEAK_, "C": EVID_, "D": EVID_}) == WEAK_
    assert hv({"A": EVID_, "B": WEAK_, "C": NONE_, "D": EVID_}) == NONE_
    assert hv({"A": WEAK_, "B": WEAK_}) == WEAK_
    assert hv({"A": NONE_}) == NONE_ and hv({}) == NONE_
    assert hv({"A": NONE_, "B": EVID_}) == hv({"B": EVID_, "A": NONE_}) == NONE_      # order does not matter
    with pytest.raises(ValueError):
        hv({"A": "great"})


def test_report_prints_flags_verdict_reasons_headline_and_caveats(cfg):
    res = H.evaluate_all(cfg, H.real_window(cfg), dev_numbers=fake_dev())
    md = res.markdown
    assert "Hold-out failure signals" in md and "Verdict (pre-registered rule" in md
    assert "CLEARED: necessary condition: t > 3 for the strategy" in md
    assert "so t > 3 alone is not an edge" in md and "buy_and_hold t = 3.50 (clears t > 3)" in md
    s = res.summary
    assert s["assets"]["SPY"]["verdict"] == NONE_                   # t > 3 alone: no longer "weak evidence"
    assert s["per_asset_verdicts"] == {t: NONE_ for t in TICKERS} and s["headline_verdict"] == NONE_
    assert "**Headline: no evidence of an edge**" in md and "| SPY | no evidence of an edge |" in md
    assert set(s["assets"]["SPY"]["hold_out_flags"]) >= {"any_failure", "reasons"}
    assert "can only DOWNGRADE" in md
    assert "NO higher-cost ETF scenario was pre-registered" in md and "ONLY the 1-day delay" in md
    # disclosures: scope of the DSRs, SPA is family-level, t > 3 is a test against zero
    assert "ASSET-scoped" in md and "GLOBAL (270 trials)" in md and "FAMILY-level" in md and "test against ZERO" in md
    assert "WEAKEST of the per-asset verdicts" in md
    assert "asset-scoped, about 54 to 106 trials" in " ".join(s["assets"]["SPY"]["verdict_reasons"])
    # a mixed world: SPY clears everything, BTC only the baseline excess-t test -> headline = weakest
    mixed = fake_dev(sig=True)
    mixed["assets"]["BTC-USD"].update(spa_pvalue_vs_buy_and_hold=0.9, dsr_excess_over_vol_targeted_bh=0.1, tstat=1.0)
    r2 = H.evaluate_all(cfg, H.real_window(cfg), dev_numbers=mixed)
    hv = r2.summary["per_asset_verdicts"]
    assert hv["BTC-USD"] == WEAK_ and r2.summary["headline_verdict"] == H.headline_verdict(hv)
    assert "**Headline: weak evidence**" in r2.markdown or hv["SPY"] == NONE_


# ---- 3. report honesty
def test_report_labels_attempts_climatology_and_btc_cash(cfg):
    w = H.real_window(cfg)
    first = H.evaluate_all(cfg, w, dev_numbers=fake_dev())
    assert "ATTEMPT" not in first.markdown and first.earlier_attempts == []
    hd = H.holdout_dir(cfg)
    hd.mkdir(parents=True)
    (hd / "HOLDOUT_FAILED_2026-11-10.json").write_text("{}")
    (hd / "HOLDOUT_FAILED_2026-11-11.json").write_text("{}")
    res = H.evaluate_all(cfg, w, dev_numbers=fake_dev())
    assert "ATTEMPT 3: NOT a single evaluation" in res.markdown
    assert "HOLDOUT_FAILED_2026-11-10.json" in res.markdown and "HOLDOUT_FAILED_2026-11-11.json" in res.markdown
    assert "climatology (= always-up accuracy)" in res.markdown
    # BTC: % invested comes from the numbers, and the eval-base Brier skill column comes first
    inv = res.summary["assets"]["BTC-USD"]["pct_invested"]
    assert f"invested on {inv * 100:.0f}% of days" in res.markdown
    assert ("mostly in cash" in res.markdown) == (inv < 0.5)
    btc_part = res.markdown.split("## BTC-USD")[1]
    hdr = [l for l in btc_part.splitlines() if l.startswith("| Model")][0]
    assert hdr.index("EVAL base") < hdr.index("train base")
    spy_hdr = [l for l in res.markdown.split("## SPY")[1].splitlines() if l.startswith("| Model")][0]
    assert spy_hdr.index("train base") < spy_hdr.index("eval base")
    # dry runs never claim attempts
    hd.joinpath("HOLDOUT_FAILED_x.json").write_text("{}")
    dry = H.evaluate_all(cfg, H.pseudo_window(cfg), assets=["SPY"])
    assert dry.earlier_attempts == [] and "ATTEMPT" not in dry.markdown


# ---- 4. scored window asserted
def test_scored_window_exact_for_etf_and_btc(cfg, result):
    for t, ev in result.assets.items():
        exp = H.scored_dates(t, CUTOFF, END)
        assert ev.dates.equals(exp) and ev.dates.min() == CUTOFF and len(ev.dates) == len(exp)
    assert result.assets["BTC-USD"].dates.max() == END - pd.Timedelta(days=1)
    assert result.assets["SPY"].dates.max() == F._sessions()[F._sessions() <= END][-2]
    assert H.last_date_before("SPY", CUTOFF) == pd.Timestamp("2021-03-12")           # Monday cutoff -> Friday
    assert H.last_date_before("BTC-USD", CUTOFF) == CUTOFF - pd.Timedelta(days=1)


def test_skipped_block_is_an_error(cfg, monkeypatch):
    w = H.real_window(cfg)
    hist = H.load_history("SPY", STRAT, Path(cfg["data_dir"]) / "results", CUTOFF, CUTOFF)
    fr = H.load_frame("SPY", cfg, END)
    _eval(fr, hist, w)                                                              # control: passes
    holey = fr.drop(fr.index[(fr.index >= "2021-05-03") & (fr.index < "2021-06-01")])   # a month of rows vanishes
    with pytest.raises(AssertionError, match="scored window mismatch"):
        _eval(holey, hist, w)
    orig = B.walk_forward

    def skip_first_block(*a, **k):                                                  # a whole retrain block skipped
        wf = orig(*a, **k)
        wf.preds = wf.preds[wf.preds["date"] >= CUTOFF + pd.Timedelta(days=40)]
        return wf
    monkeypatch.setattr(B, "walk_forward", skip_first_block)
    with pytest.raises(AssertionError, match="scored window mismatch"):
        _eval(fr, hist, w)


def test_history_gap_is_an_error_and_unblanked_last_label_too(cfg):
    w = H.real_window(cfg)
    hist = H.load_history("SPY", STRAT, Path(cfg["data_dir"]) / "results", CUTOFF, CUTOFF)
    fr = H.load_frame("SPY", cfg, END)
    short = dataclasses.replace(hist, dates=hist.dates[:-3], p=hist.p[:-3], base=hist.base[:-3], nr=hist.nr[:-3],
                                w=hist.w[:-3], block=hist.block[:-3])
    with pytest.raises(AssertionError, match="gap/overlap"):
        _eval(fr, short, w)
    unblanked = pd.read_parquet(Path(cfg["data_dir"]) / "features" / "SPY.parquet")
    unblanked = unblanked[unblanked.index <= END]          # last label still present -> one extra scored day
    with pytest.raises(AssertionError, match="scored window mismatch"):
        _eval(unblanked, hist, w)


# ---- 5. preflight before the marker
def test_lightgbm_import_failure_does_not_burn_the_shot(cfg, monkeypatch):
    s = {**STRAT, "model": "ensemble"}
    cfg["leverage_strategy"] = dict(s)
    monkeypatch.setattr(H, "PINNED_STRATEGY", dict(s))
    monkeypatch.setitem(sys.modules, "lightgbm", None)                  # import lightgbm -> ImportError
    with pytest.raises(RuntimeError, match="LightGBM cannot be imported"):
        H.run_final(cfg, confirm=True)
    assert not H.holdout_dir(cfg).exists()                             # no marker


def test_other_preflight_failures_do_not_burn_the_shot(cfg, monkeypatch):
    rd = Path(cfg["data_dir"]) / "results"
    saved = (rd / "SPY" / "predictions.parquet").read_bytes()
    (rd / "SPY" / "predictions.parquet").unlink()                       # history missing
    with pytest.raises(H.HoldoutRefusal, match="Stage 3 results not found"):
        H.run_final(cfg, confirm=True)
    assert not H.holdout_dir(cfg).exists()
    (rd / "SPY" / "predictions.parquet").write_bytes(saved)
    monkeypatch.setattr(H, "summary_numbers", lambda cfg=None: (_ for _ in ()).throw(KeyError("no significance")))
    with pytest.raises(KeyError):                                       # strategy lookup / dev numbers
        H.run_final(cfg, confirm=True)
    assert not H.holdout_dir(cfg).exists()
    p = Path(cfg["data_dir"]) / "features" / "SPY.parquet"              # dataset shape: no label columns
    df = pd.read_parquet(p)
    monkeypatch.setattr(H, "summary_numbers", lambda cfg=None: fake_dev())
    df.drop(columns=list(F.TARGET_COLS)).to_parquet(p)
    with pytest.raises((H.HoldoutRefusal, KeyError)):
        H.run_final(cfg, confirm=True)
    assert not H.holdout_dir(cfg).exists()


# ---- 6. survivors
def test_btc_coverage_with_a_missing_weekend_day_is_refused(cfg):
    w = H.real_window(cfg)
    dates = pd.date_range("2018-01-02", END, freq="D")
    sat = pd.Timestamp("2021-06-12")
    assert sat.day_of_week == 5
    assert H.check_coverage("BTC-USD", dates, w)["n_scorable_rows"] == (END - CUTOFF).days
    with pytest.raises(H.HoldoutRefusal, match="1 of .* required days are missing"):
        H.check_coverage("BTC-USD", dates.drop(sat), w)
    p = Path(cfg["data_dir"]) / "features" / "BTC-USD.parquet"
    df = pd.read_parquet(p)
    df.drop(sat).to_parquet(p)
    with pytest.raises(H.HoldoutRefusal, match="BTC-USD: the data do not cover"):
        H.run_final(cfg, confirm=True)
    assert not H.holdout_dir(cfg).exists()


def test_hits_are_rounded_not_truncated(result):
    bad = [(k, n) for n in range(1, 400) for k in range(n + 1) if int(k / n * n) != k]
    assert len(bad) > 10                                               # truncation really would be wrong
    for k, n in bad:
        assert H.hits_from_accuracy(k / n, n) == k
        lo, hi = H.wilson_interval(H.hits_from_accuracy(k / n, n), n)
        assert (lo, hi) == pytest.approx(proportion_confint(k, n, alpha=0.05, method="wilson"), abs=1e-9)
    for ev in result.assets.values():
        for _, r in ev.prediction.iterrows():
            assert r["hits"] == round(r["accuracy"] * r["n"]) and abs(r["hits"] - r["accuracy"] * r["n"]) < 1e-6


def test_climatology_eval_base_log_loss_skill_uses_the_evaluation_y_mean(result):
    ll = lambda y, q: -np.mean(y * np.log(q) + (1 - y) * np.log(1 - q))
    for t, ev in result.assets.items():
        d = ev.predictions[ev.predictions["model"] == "logreg"].sort_values("date")
        y, b = d["y"].to_numpy(), d["base_rate"].to_numpy()
        clim = ev.prediction[ev.prediction["model"] == "climatology"].iloc[0]
        py = y.mean()
        assert clim["log_loss_skill_vs_eval_base"] == pytest.approx(1 - ll(y, b) / ll(y, np.full(len(y), py)))
        assert clim["log_loss_skill_vs_eval_base"] <= 1e-12            # the in-sample mean is the log-loss optimum
        assert abs(b.mean() - py) > 1e-6                               # so training base != eval base here
        assert clim["brier_loss_diff_vs_eval_base"] == pytest.approx(np.mean((b - y) ** 2) - np.mean((py - y) ** 2))
        row = ev.prediction[ev.prediction["model"] == "logreg"].iloc[0]
        p = d["p_up"].to_numpy()
        dl = (p - y) ** 2 - (py - y) ** 2
        assert row["brier_loss_diff_vs_eval_base"] == pytest.approx(dl.mean())
        assert row["brier_loss_diff_se"] == pytest.approx(dl.std(ddof=1) / np.sqrt(len(dl)))


def test_pinned_strategy_refuses_non_expanding_and_bad_values(cfg):
    for bad, match in (({"window": "rolling"}, "only the expanding window"),
                       ({"mode": "nope"}, "unsupported strategy mode"),
                       ({"model": "xgb"}, "unknown model")):
        with pytest.raises(H.HoldoutRefusal, match=match):
            H.pinned_strategy(dict(cfg, leverage_strategy={**STRAT, **bad}))
    assert H.pinned_strategy(cfg) == STRAT


# =========================================================================== round-2 fixes
# ---- preflight really runs the contiguity and shape checks (a failure there must not burn the shot)
def test_preflight_history_with_a_mid_history_gap_writes_no_marker(cfg):
    rd = Path(cfg["data_dir"]) / "results"
    for name in ("predictions.parquet", "baselines.parquet"):        # same day removed from both (stay aligned)
        df = pd.read_parquet(rd / "SPY" / name)
        gap = sorted(df["date"].unique())[200]
        df[df["date"] != gap].to_parquet(rd / "SPY" / name, index=False)
    with pytest.raises(AssertionError, match="not contiguous"):
        H.run_final(cfg, confirm=True)
    assert not H.holdout_dir(cfg).exists()                          # no marker, no outputs: the shot is kept


def test_preflight_history_ending_early_writes_no_marker(cfg):
    rd = Path(cfg["data_dir"]) / "results"
    for name in ("predictions.parquet", "baselines.parquet"):
        df = pd.read_parquet(rd / "SPY" / name)
        last = df["date"].max()
        df[df["date"] != last].to_parquet(rd / "SPY" / name, index=False)
    with pytest.raises(AssertionError, match="gap/overlap"):
        H.run_final(cfg, confirm=True)
    assert not H.holdout_dir(cfg).exists()


def test_preflight_frame_missing_label_columns_writes_no_marker(cfg, monkeypatch):
    orig = H.load_frame
    monkeypatch.setattr(H, "load_frame", lambda t, c, end: orig(t, c, end).drop(columns=list(F.TARGET_COLS)))
    with pytest.raises(H.HoldoutRefusal, match="dataset lacks columns"):
        H.run_final(cfg, confirm=True)
    assert not H.holdout_dir(cfg).exists()


def test_preflight_frame_with_wrong_last_row_writes_no_marker(cfg, monkeypatch):
    orig = H.load_frame
    monkeypatch.setattr(H, "load_frame", lambda t, c, end: orig(t, c, end).iloc[:-1])
    with pytest.raises(H.HoldoutRefusal, match="last data row .* != last expected"):
        H.run_final(cfg, confirm=True)
    assert not H.holdout_dir(cfg).exists()


def test_preflight_calls_the_contiguity_and_shape_checks_for_every_asset(cfg, monkeypatch):
    """Removing either call from preflight() would let a bad history/frame reach the marker."""
    seen = {"hist": [], "shape": []}
    oh, osh = H.check_history_contiguity, H.check_frame_shape
    monkeypatch.setattr(H, "check_history_contiguity", lambda t, h, w: (seen["hist"].append(t), oh(t, h, w))[1])
    monkeypatch.setattr(H, "check_frame_shape", lambda t, f, w: (seen["shape"].append(t), osh(t, f, w))[1])
    H.preflight(cfg)
    assert seen == {"hist": TICKERS, "shape": TICKERS}
    assert not H.holdout_dir(cfg).exists()


def test_check_history_contiguity_unit(cfg):
    w = H.real_window(cfg)
    rd = Path(cfg["data_dir"]) / "results"
    for t in TICKERS:
        h = H.load_history(t, STRAT, rd, CUTOFF, CUTOFF)
        H.check_history_contiguity(t, h, w)                          # control passes
        keep = np.ones(len(h.dates), bool)
        keep[len(keep) // 2] = False                                 # one day vanishes from the middle
        holey = dataclasses.replace(h, dates=h.dates[keep], p=h.p[keep], base=h.base[keep], nr=h.nr[keep],
                                    w=h.w[keep], block=h.block[keep])
        with pytest.raises(AssertionError, match="not contiguous: 1 expected day"):
            H.check_history_contiguity(t, holey, w)
        extra = dataclasses.replace(h, dates=h.dates.insert(3, h.dates[3] + pd.Timedelta(hours=12)))
        with pytest.raises(AssertionError, match="unexpected"):
            H.check_history_contiguity(t, extra, w)


@pytest.mark.skipif(not HAVE_REAL, reason="real data/results not present")
def test_real_stage3_histories_are_contiguous_to_the_cutoff():
    """Reads only the saved DEVELOPMENT results (dates before the cutoff)."""
    w = H.real_window(REAL)
    s = H.pinned_strategy(REAL)
    for t in [a["ticker"] for a in REAL["assets"]]:
        h = H.load_history(t, s, REAL["data_dir"] / "results", w.cutoff, pd.Timestamp(H.PINNED_CUTOFF))
        H.check_history_contiguity(t, h, w)
        assert h.dates.max() == H.last_date_before(t, w.cutoff)


# ---- exact scored-dates equality
def test_scored_dates_must_match_exactly_not_just_in_length_and_ends(cfg, monkeypatch):
    w = H.real_window(cfg)
    hist = H.load_history("BTC-USD", STRAT, Path(cfg["data_dir"]) / "results", CUTOFF, CUTOFF)
    fr = H.load_frame("BTC-USD", cfg, END)
    ev = H.evaluate_asset("BTC-USD", fr, hist, STRAT, COSTS["BTC-USD"], w, SMALL_BT)           # control
    exp = H.scored_dates("BTC-USD", CUTOFF, END)
    assert ev.dates.equals(exp)
    shifted = exp + pd.Timedelta(days=1)                                                       # same length, shifted window
    monkeypatch.setattr(H, "scored_dates", lambda *a: shifted)
    with pytest.raises(AssertionError, match="scored window mismatch"):
        H.evaluate_asset("BTC-USD", fr, hist, STRAT, COSTS["BTC-USD"], w, SMALL_BT)
    inner = exp.delete(50).insert(50, exp[50] + pd.Timedelta(hours=12))       # same length, SAME first and last date
    assert len(inner) == len(exp) and inner[0] == exp[0] and inner[-1] == exp[-1] and not inner.equals(exp)
    monkeypatch.setattr(H, "scored_dates", lambda *a: inner)
    with pytest.raises(AssertionError, match="scored window mismatch"):
        H.evaluate_asset("BTC-USD", fr, hist, STRAT, COSTS["BTC-USD"], w, SMALL_BT)


# ---- adaptive margin m: causal on a synthetic case where m is KNOWN to change
def _adaptive_world():
    rng = np.random.default_rng(11)
    n, block = 1080, 90
    base = np.full(n, 0.5)
    deficit = rng.uniform(0.0, 0.05, n)                  # p = base - deficit
    p = base - deficit
    nr = rng.normal(0, 0.004, n)
    half = 360
    # first half: only days with deficit < 0.005 pay (best m ~ 0.005); second half: every day pays (best m = 0.02)
    nr[:half] += np.where(deficit[:half] < 0.005, 0.01, -0.01)
    nr[half:] += 0.02
    starts = list(range(0, n, block))
    return p, base, nr, starts


def test_adaptive_margin_is_causal_and_changes_over_time_synthetic():
    p, base, nr, starts = _adaptive_world()
    cost, min_days = 0.0005, 90
    pos, log = B.adaptive_positions("long_flat", p, base, nr, None, starts, cost, min_days)
    ms = [lg["m"] for lg in log]
    assert len(set(ms)) >= 3, ms                                # the chosen margin really changes
    assert log[0]["used_fallback"] and not log[-1]["used_fallback"]
    k = starts[7]
    p2, nr2 = p.copy(), nr.copy()
    rng = np.random.default_rng(3)
    p2[k:] = 0.5 - rng.uniform(0, 0.05, len(p) - k)             # a different future: signals AND outcomes
    nr2[k:] = -nr[k:] * 3 + rng.normal(0, 0.02, len(p) - k)
    pos2, log2 = B.adaptive_positions("long_flat", p2, base, nr2, None, starts, cost, min_days)
    assert np.array_equal(pos[:k], pos2[:k])
    assert [lg["m"] for lg in log[:8]] == [lg["m"] for lg in log2[:8]]       # blocks 0..7 use only rows < start
    assert [lg["m"] for lg in log] != [lg["m"] for lg in log2]               # the later margins DO react to the new future
    # sensitivity control: changing the PAST changes later margins (so the comparison above can detect a peek)
    nr3 = nr.copy()
    nr3[: starts[4]] = -nr3[: starts[4]]
    _, log3 = B.adaptive_positions("long_flat", p, base, nr3, None, starts, cost, min_days)
    assert [lg["m"] for lg in log3[5:]] != [lg["m"] for lg in log[5:]]


def test_harness_margin_depends_only_on_history_plus_earlier_holdout_days(cfg):
    """Through evaluate_asset: the margin and position of every day before the cut are unchanged when the
    hold-out days AFTER the cut are replaced by garbage."""
    df = pd.read_parquet(Path(cfg["data_dir"]) / "features" / "BTC-USD.parquet")
    hist = H.load_history("BTC-USD", STRAT, Path(cfg["data_dir"]) / "results", CUTOFF, CUTOFF)
    e1, e2 = pd.Timestamp("2021-07-15"), END
    fa = df[df.index <= e1].copy()
    for c in F.TARGET_COLS:
        fa.iloc[-1, fa.columns.get_loc(c)] = np.nan
    fb = df[df.index <= e2].copy()
    late = fb.index > e1
    rng = np.random.default_rng(2)
    fb.loc[late, :] = rng.normal(size=(late.sum(), fb.shape[1])) * 30
    fb.loc[late, "target"] = rng.integers(0, 2, late.sum()).astype(float)
    for c in F.TARGET_COLS:
        fb.iloc[-1, fb.columns.get_loc(c)] = np.nan
    a = H.evaluate_asset("BTC-USD", fa, hist, STRAT, COSTS["BTC-USD"], H.Window(CUTOFF, e1, False), SMALL_BT)
    b = H.evaluate_asset("BTC-USD", fb, hist, STRAT, COSTS["BTC-USD"], H.Window(CUTOFF, e2, False), SMALL_BT)
    pa = a.predictions[a.predictions["model"] == "logreg"].set_index("date")
    pb = b.predictions[b.predictions["model"] == "logreg"].set_index("date")
    common = pa.index[pa.index < e1 - pd.Timedelta(days=1)]
    assert len(common) > 60
    assert np.array_equal(pa.loc[common, "margin_m"].to_numpy(), pb.loc[common, "margin_m"].to_numpy())
    assert np.array_equal(pa.loc[common, "position"].to_numpy(), pb.loc[common, "position"].to_numpy())


# ---- run_final may never be reached with the real config
def _holdout_folder_state():
    """File names + SHA-256 of data/results/holdout (empty tuple while the real run has not happened)."""
    import hashlib
    d = ROOT / "data" / "results" / "holdout"
    if not d.exists():
        return ()
    return tuple(sorted((f.name, hashlib.sha256(f.read_bytes()).hexdigest()) for f in d.iterdir() if f.is_file()))


def test_guard_blocks_run_final_with_the_real_config(monkeypatch):
    before = _holdout_folder_state()
    with pytest.raises(AssertionError, match="REAL config object"):
        H.run_final(REAL, confirm=False)
    with pytest.raises(AssertionError, match="REAL config object"):
        H.run_final(REAL, confirm=True)
    clone = dict(REAL)                                                     # a copy still points at the real folders
    with pytest.raises(AssertionError, match="REAL project root"):
        H.run_final(clone, confirm=True)
    clone2 = dict(REAL, project_root=Path("/nonexistent/synthetic"))
    with pytest.raises(AssertionError, match="REAL data folder"):
        H.run_final(clone2, confirm=True)
    # the CLI path: a patched load_config that returns the real config is stopped as well
    monkeypatch.setattr("quant.config.load_config", lambda path=None: REAL)
    with pytest.raises(AssertionError, match="REAL config object"):
        H.main(["--run", "--confirm-single-final-evaluation"])
    # nothing was created, changed or removed in the real hold-out folder (absent before the real run;
    # after the single real run it must be byte-identical)
    assert _holdout_folder_state() == before


def test_guard_lets_a_synthetic_config_through(cfg):
    with pytest.raises(H.HoldoutRefusal, match="confirm-single-final-evaluation"):
        H.run_final(cfg, confirm=False)                                    # reaches the real run_final, refuses itself
