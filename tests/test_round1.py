"""Stage 5, round-1 critic fixes: logic-layer tests (explain.py, dash_data.py, ui.py).

Most tests here exist to kill a specific surviving mutant reported by the Code+test critic (a swapped
q25/q75, a wrong Sharpe-SE multiplier, a forward-looking rolling window, ...), so each assertion is
chosen to fail when exactly that thing is changed.
"""
import ast
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from app import ui
from quant import dash_data as DD
from quant import explain as E
from quant import features as F
from quant import metrics as M
from quant import paper as P
from tests import dash_world as W
from tests.synth import TICKERS

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    return W.build(tmp_path_factory.mktemp("r1world"), extra_days=("2021-10-14",))


# =========================================================================== wording: q25 / q75 / shares
FACTS = dict(column="r5", value=0.0, n_history=100, history_start="2010", below=50.0, above=40.0,
             q25=-0.01, q75=0.02, share_equal=10.0)


def test_middle_half_sentence_uses_q25_first_and_q75_second():
    s = E.feature_sentence("SPY", "r5", FACTS)
    assert "between -1.0% and +2.0%" in s          # swapping q25/q75 or using other quantiles breaks this
    assert "higher than 50% of days since 2010" in s


def test_lower_than_branch_uses_the_share_above():
    s = E.feature_sentence("SPY", "r5", {**FACTS, "below": 30.0, "above": 60.0})
    assert "lower than 60% of days since 2010" in s and "30%" not in s


@pytest.mark.parametrize("share,text", [(100.0, "every day"), (99.97, "almost every day"),
                                        (99.6, "99.6% of days"), (99.4, "99% of days"), (37.2, "37% of days")])
def test_share_phrase_never_rounds_below_100_up_to_100(share, text):
    assert E.share_phrase(share) == text


def test_sentence_says_almost_every_day_instead_of_100_percent():
    s = E.feature_sentence("SPY", "r5", {**FACTS, "below": 10.0, "above": 99.97})
    assert "lower than almost every day since 2010" in s and "100%" not in s
    s = E.feature_sentence("SPY", "r5", {**FACTS, "below": 10.0, "above": 99.6})
    assert "lower than 99.6% of days" in s


def test_flag_sentence_uses_the_right_share_in_both_branches():
    f = {**FACTS, "share_equal": 30.0}
    on = E.feature_sentence("SPY", "new_high_20", {**f, "value": 1.0})
    off = E.feature_sentence("SPY", "new_high_20", {**f, "value": 0.0})
    assert "true on 30% of days since 2010" in on
    assert "not the case today" in off and "true of 70% of days since 2010" in off


def test_bitcoin_weekend_volume_gets_a_caveat_but_weekdays_and_other_assets_do_not():
    facts = {**FACTS, "column": "vol_chg_20"}
    sat, mon = pd.Timestamp("2024-03-02"), pd.Timestamp("2024-03-04")
    assert sat.day_name() == "Saturday"
    assert "weekend" in E.feature_sentence("BTC-USD", "vol_chg_20", facts, sat).lower()
    assert "weekend" in E.feature_sentence("BTC-USD", "vol_chg_20", facts, pd.Timestamp("2024-03-03")).lower()
    assert "weekend" not in E.feature_sentence("BTC-USD", "vol_chg_20", facts, mon).lower()
    assert "weekend" not in E.feature_sentence("SPY", "vol_chg_20", facts, sat).lower()


def test_ordinal_covers_the_teens_and_the_endings():
    got = {n: E.ordinal(n) for n in (1, 2, 3, 4, 11, 12, 13, 21, 22, 23, 91, 100, 101, 111)}
    assert got == {1: "1st", 2: "2nd", 3: "3rd", 4: "4th", 11: "11th", 12: "12th", 13: "13th", 21: "21st",
                   22: "22nd", 23: "23rd", 91: "91st", 100: "100th", 101: "101st", 111: "111th"}


@pytest.mark.parametrize("pct,word", [(23, "23rd"), (91, "91st"), (1, "1st"), (2, "2nd"), (3, "3rd"), (11, "11th"),
                                      (12, "12th"), (13, "13th"), (52, "52nd")])
def test_gauge_annotation_uses_correct_ordinals(pct, word):
    bounds = [{"zone": z, "lo": -0.05, "hi": 0.05 if i == 3 else -0.05, "reachable": i == 3} for i, z in enumerate(E.ZONES)]
    m = SimpleNamespace(bounds=bounds, needle=0.0, percentile=float(pct), zone="Hold / buy", axis_lo=-0.05, axis_hi=0.05)
    fig = ui.gauge_figure(m)
    assert any(f"about the {word} percentile of past days" in a.text for a in fig.layout.annotations)
    assert "23th" not in str([a.text for a in fig.layout.annotations])


# =========================================================================== feature history excludes the decision day
def _dev_dataset(n=400, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2012-01-02", periods=n)
    return pd.DataFrame({"r5": rng.normal(0, 0.01, n)}, index=idx)


def test_feature_history_excludes_the_decision_day_itself_inside_the_dev_period():
    ds = _dev_dataset()
    cutoff = ds.index[-1] + pd.Timedelta(days=30)             # the whole dataset is "development"
    d = ds.index[200]                                          # decision day well inside the dev period
    base = E.feature_facts(ds, "r5", d, cutoff)
    ds2 = ds.copy()
    ds2.loc[d, "r5"] = 5.0                                     # perturb ONLY the row at d
    pert = E.feature_facts(ds2, "r5", d, cutoff)
    assert base["n_history"] == pert["n_history"] == 200       # rows strictly before d
    assert base["q25"] == pert["q25"] and base["q75"] == pert["q75"]
    assert pert["value"] == 5.0 and pert["below"] == 100.0     # d is compared with the history, not part of it
    ds3 = ds.copy()
    ds3.loc[ds3.index[199], "r5"] = 5.0                        # the day BEFORE d is part of the history
    assert E.feature_facts(ds3, "r5", d, cutoff)["q75"] != base["q75"] or \
        E.feature_facts(ds3, "r5", d, cutoff)["below"] != base["below"]


# =========================================================================== trust stats exclude the decision day
def _preds(n=300, seed=1):
    rng = np.random.default_rng(seed)
    d = pd.bdate_range("2015-01-01", periods=n)
    y = (rng.random(n) < 0.55).astype(float)
    return pd.DataFrame({"date": d, "p_up": np.clip(0.55 + rng.normal(0, 0.03, n), 0.05, 0.95),
                         "base_rate": 0.55, "y": y})


def test_confidence_bucket_ignores_days_at_or_after_the_decision_day():
    pr = _preds()
    d = pr["date"].iloc[200]
    base = E.confidence_bucket(pr[pr["date"] < d], 0.58, 0.55)
    junk = pr.copy()
    junk.loc[junk["date"] >= d, ["p_up", "y"]] = [0.99, 0.0]    # garbage at the decision day and after
    assert E.confidence_bucket(junk, 0.58, 0.55, asof=d) == base
    assert E.confidence_bucket(junk, 0.58, 0.55) != base         # (so the filter is what protects it)


def test_explain_today_trust_statistics_never_include_the_decision_day(world):
    cfg = world
    t = "SPY"
    ds = W.dataset(cfg, t)
    model = E.load_live_model_ex(t, cfg)[0]
    settings = DD.paper_settings(cfg)
    preds = DD.model_predictions(DD.load_asset_results(t, cfg), "ensemble")
    cutoff = pd.Timestamp(cfg["holdout_cutoff_date"])
    d = DD.ledger_for(t, cfg)["date"].iloc[-1]
    hist = np.array([-0.02, -0.01, 0.0, 0.01, 0.02])
    meter = E.meter_from_inputs(settings, 0.55, 0.54, 0.005, 1.0, 0.02, 0.0, hist)
    badge = E.evidence_badge(t, cfg)
    base = E.explain_today(t, ds, d, model, meter, preds, badge, cutoff)
    extra = pd.DataFrame({"date": [d, d + pd.Timedelta(days=1)], "p_up": [0.99, 0.99], "base_rate": [0.5, 0.5],
                          "y": [0.0, 0.0]})
    for col in preds.columns:
        if col not in extra:
            extra[col] = preds[col].iloc[-1]
    w = E.explain_today(t, ds, d, model, meter, pd.concat([preds, extra], ignore_index=True), badge, cutoff)
    assert w.trust["bucket"] == base.trust["bucket"] and w.trust["text"] == base.trust["text"]


# =========================================================================== conclusion consistent with p vs base rate
def test_conclusion_is_consistent_with_the_probability_versus_the_base_rate(world):
    cfg = world
    for t in TICKERS:
        rep = E.today_report(t, cfg, ds=W.dataset(cfg, t))
        c, m = rep["why"].conclusion, rep["meter"]
        assert "lean slightly up" not in c                       # old wording that could contradict p < base
        if m.gap < 0:
            assert "below the usual rate" in c and "not a bullish reading" in c
        elif m.gap > 0:
            assert "above the usual rate" in c


# =========================================================================== evidence badge: SKILL conservative
def test_skill_uses_the_smaller_of_the_two_skills_and_reports_both():
    kw = dict(pt_pvalue=0.01, pt_statistic=2.5, brier_skill=0.0059, brier_skill_eval_base=-0.0028, sharpe=1.2,
              sharpe_vt=0.9, psr_excess_vt=0.99, dsr_excess_vt=0.97, dsr_global=0.96)
    b = E.badge_from_numbers(**kw)
    sk = b["checks"]["SKILL"]
    assert not sk["passed"] and sk["value"] == pytest.approx(-0.0028)
    assert sk["vs_training_base"] == 0.0059 and sk["vs_eval_base"] == -0.0028
    assert b["level"] == "weak"                                   # drift-only skill can no longer reach green
    ok = E.badge_from_numbers(**{**kw, "brier_skill_eval_base": 0.001})
    assert ok["checks"]["SKILL"]["passed"] and ok["level"] == "evidence"
    assert "0.0059" in " ".join(b["reasons"]) and "-0.0028" in " ".join(b["reasons"])     # real values shown


def test_every_check_shows_its_actual_value_in_the_reasons():
    kw = dict(pt_pvalue=0.34, pt_statistic=0.95, brier_skill=0.01, brier_skill_eval_base=0.008, sharpe=1.2,
              sharpe_vt=0.9, psr_excess_vt=0.41, dsr_excess_vt=0.37, dsr_global=0.002)
    txt = " | ".join(E.badge_from_numbers(**kw)["reasons"])
    for want in ("p = 0.340", "statistic = 0.95", "strategy Sharpe 1.20", "0.90", "0.410", "0.370", "0.002"):
        assert want in txt, want


def test_badge_rule_declares_the_round1_conservative_change_and_psr_x():
    low = E.BADGE_RULE.lower()
    assert "more conservative" in low and "deliberately conservative" in low and "psr_x" in low and "dsr_g" in low
    assert "round-1" not in low and "critic" not in low


def test_amber_and_green_badges_are_reachable_with_injected_inputs():
    amber = E.badge_from_numbers(pt_pvalue=0.01, pt_statistic=2.0, brier_skill=0.01, brier_skill_eval_base=0.01,
                                 sharpe=1.2, sharpe_vt=0.9, psr_excess_vt=0.5, dsr_excess_vt=0.5, dsr_global=0.5)
    green = E.badge_from_numbers(pt_pvalue=0.01, pt_statistic=2.0, brier_skill=0.01, brier_skill_eval_base=0.01,
                                 sharpe=1.2, sharpe_vt=0.9, psr_excess_vt=0.99, dsr_excess_vt=0.99, dsr_global=0.99)
    assert (amber["level"], amber["color"], amber["faded"], amber["note"]) == ("weak", "amber", False, None)
    assert (green["level"], green["color"], green["faded"], green["note"]) == ("evidence", "green", False, None)


# =========================================================================== live model: the real reason
def test_load_live_model_reports_a_missing_file(world):
    cfg = world
    path = P.PaperFiles.for_config(cfg).model("SPY")
    backup = path.read_bytes()
    path.unlink()
    try:
        model, reason = E.load_live_model_ex("SPY", cfg)
        assert model is None and "missing" in reason
        assert E.load_live_model_ex("SPY", cfg)[0] is None
    finally:
        path.write_bytes(backup)


def test_load_live_model_reports_a_corrupt_file_not_a_missing_one(world):
    cfg = world
    path = P.PaperFiles.for_config(cfg).model("QQQ")
    backup = path.read_bytes()
    path.write_bytes(b"this is not a pickle")
    try:
        model, reason = E.load_live_model_ex("QQQ", cfg)
        assert model is None and "could not be read" in reason and "missing" not in reason
    finally:
        path.write_bytes(backup)


def test_load_live_model_reports_libomp_with_the_fix(world, monkeypatch):
    import joblib

    def boom(*a, **k):
        raise OSError("dlopen(lib_lightgbm.dylib): Library not loaded: @rpath/libomp.dylib")
    monkeypatch.setattr(joblib, "load", boom)
    model, reason = E.load_live_model_ex("SPY", world)
    assert model is None and "libomp" in reason and "brew install libomp" in reason
    assert "fix_lightgbm_macos.sh" in reason and "PROGRESS.md" in reason
    assert "file is missing" not in reason
    rep = E.today_report("SPY", world, ds=W.dataset(world, "SPY"))
    assert "libomp" in rep["warning"] and "saved live model is missing" not in rep["warning"]
    assert rep["agree"] is None and rep["why"] is None


# =========================================================================== dash_data
def test_confusion_matrix_counts_a_tie_as_said_up():
    df = pd.DataFrame({"date": pd.bdate_range("2020-01-01", periods=4), "p_up": [0.55, 0.55, 0.5, 0.6],
                       "base_rate": [0.55, 0.55, 0.55, 0.55], "y": [1.0, 0.0, 0.0, 1.0]})
    cm = DD.confusion_matrix(df)
    # p == base_rate (rows 0, 1) is "said up"; row 2 is below -> said down; row 3 above -> said up
    assert cm.loc["market went up", "model said up"] == 2 and cm.loc["market went down", "model said up"] == 1
    assert cm.loc["market went down", "model said down"] == 1 and cm.loc["market went up", "model said down"] == 0


def test_rolling_auc_is_backward_looking_on_noisy_non_separable_data():
    rng = np.random.default_rng(7)
    n, w = 500, 100
    y = (rng.random(n) < 0.5).astype(float)
    p = np.clip(0.5 + 0.02 * (y - 0.5) + rng.normal(0, 0.05, n), 0.01, 0.99)   # weak signal, heavy overlap
    df = pd.DataFrame({"date": pd.bdate_range("2015-01-01", periods=n), "p_up": p, "base_rate": 0.5, "y": y})
    r = DD.rolling_auc(df, window=w)
    i = 300
    want = M.roc_auc(y[i - w + 1: i + 1], p[i - w + 1: i + 1])
    assert 0.3 < want < 0.7 and r.iloc[i] == pytest.approx(want)               # genuinely noisy, not separable
    later = df.copy()
    later.loc[later.index > i, ["p_up", "y"]] = (rng.random((n - i - 1, 2)) > 0.5).astype(float)
    assert DD.rolling_auc(later, window=w).iloc[i] == pytest.approx(r.iloc[i])  # the future cannot change it
    outside = df.copy()
    outside.loc[i - w, ["p_up", "y"]] = [0.99, 0.0]                               # just outside the window
    assert DD.rolling_auc(outside, window=w).iloc[i] == pytest.approx(r.iloc[i])
    inside = df.copy()
    inside.loc[i - w + 1, ["p_up", "y"]] = [0.99, 0.0]                            # first day inside the window
    assert DD.rolling_auc(inside, window=w).iloc[i] != pytest.approx(r.iloc[i])
    assert r.iloc[i] != pytest.approx(M.roc_auc(y[i: i + w], p[i: i + w]))        # not the forward window


def test_rolling_brier_skill_eval_matches_skill_against_the_windows_own_base_rate():
    rng = np.random.default_rng(3)
    n, w = 400, 120
    y = (rng.random(n) < 0.6).astype(float)
    p = np.clip(0.58 + rng.normal(0, 0.03, n), 0.05, 0.95)
    df = pd.DataFrame({"date": pd.bdate_range("2015-01-01", periods=n), "p_up": p, "base_rate": 0.5, "y": y})
    r = DD.rolling_brier_skill_eval(df, w)
    assert r.iloc[: w - 1].isna().all()
    for i in (w - 1, 250, n - 1):
        sl = slice(i - w + 1, i + 1)
        assert r.iloc[i] == pytest.approx(M.brier_skill(y[sl], p[sl], float(np.mean(y[sl]))))


def test_percentile_in_the_meter_is_the_mid_rank_with_ties():
    st_ = P.PaperSettings(strategy_mode="long_flat", strategy={"window": "expanding", "model": "ensemble", "mode": "long_flat",
                                                               "vol_targeted": True, "m_setting": "adaptive"},
                          sized_lambda=0.25, sized_band=0.1, capital={}, costs={})
    g = 0.56 - 0.55
    hist = np.array([0.0, g, g, g, 0.02])
    r = E.meter_from_inputs(st_, 0.56, 0.55, 0.0, 1.0, 0.02, 0.0, hist)       # today's gap ties with 3 of 5
    assert r.percentile == pytest.approx(20.0 + 0.5 * 60.0)                  # share below + half the share equal
    assert r.percentile != 20.0 and r.percentile != 80.0


def test_data_signature_changes_with_the_saved_models_and_the_ledger(world):
    cfg = world
    s0 = DD.data_signature(cfg)
    model = P.PaperFiles.for_config(cfg).model("SPY")
    old = model.stat().st_mtime_ns
    import os
    os.utime(model, ns=(old + 10_000_000_000, old + 10_000_000_000))
    try:
        s1 = DD.data_signature(cfg)
        assert s1 != s0
        assert [a != b for a, b in zip(s0, s1)].count(True) == 1
    finally:
        os.utime(model, ns=(old, old))
    assert DD.data_signature(cfg) == s0


@pytest.mark.parametrize("exc", [ValueError("bad value"), OSError("disk full"), KeyError("missing key"),
                                 RuntimeError("network down"), ZeroDivisionError("oops")])
def test_refresh_data_never_raises_for_any_exception(exc):
    def boom(*a, **k):
        raise exc
    out = DD.refresh_data(DD._cfg(), run=boom)
    assert not out.ok and out.message.startswith("Refresh failed") and "Traceback" not in out.message
    if not isinstance(exc, ZeroDivisionError):
        assert str(exc.args[0]) in out.message


def test_price_staleness_comparison_direction(world):
    cfg = world
    px = world["data_dir"] / "SPY.parquet"
    now = "2021-10-15 23:00:00"                                        # SPY expects 2021-10-15

    def warn(last):
        pd.DataFrame({"Close": [1.0, 2.0]}, index=pd.to_datetime([pd.Timestamp(last) - pd.Timedelta(days=1), last])
                     ).to_parquet(px)
        return DD.data_freshness(cfg, now=now)["SPY"]["warning"]
    try:
        assert warn("2021-10-15") is None                              # prices at the expected day: fresh
        assert warn("2021-10-18") is None                              # prices AHEAD of expected are not stale
        w = warn("2021-10-14")                                          # one day behind: stale
        assert w and "STALE DATA: prices end 2021-10-14" in w
    finally:
        px.unlink()


# =========================================================================== ui drawing details
def test_sharpe_band_uses_k_equal_two_by_default():
    t = pd.DataFrame({"label": ["a", "b"], "sharpe": [1.0, 0.5], "sharpe_se": [0.5, 0.25], "kind": ["trial", "baseline"]})
    fig = ui.sharpe_band_fig(t)
    arr = sorted(np.asarray(fig.data[0].error_x.array, dtype=float))
    assert arr == pytest.approx([0.5, 1.0])                              # 2 * se, not 1 * se or 3 * se
    assert "2 standard errors" in fig.layout.xaxis.title.text


def test_leverage_chart_plots_the_selected_leverage_not_always_1x():
    idx = pd.bdate_range("2020-01-01", periods=5)
    eq = {"buy_and_hold": pd.DataFrame({1.0: np.linspace(1, 2, 5), 2.0: np.linspace(1, 5, 5)}, index=idx),
          "chosen_strategy": pd.DataFrame({1.0: np.linspace(1, 1.5, 5), 2.0: np.linspace(1, 3, 5)}, index=idx)}
    for L in (1.0, 2.0):
        fig = ui.leverage_equity_fig(eq, L)
        for tr, s in zip(fig.data, ("buy_and_hold", "chosen_strategy")):
            assert np.allclose(np.asarray(tr.y, dtype=float), eq[s][L].to_numpy())
        assert f"{L:g}x leveraged equity" in fig.layout.title.text


def test_wiped_out_line_is_drawn_at_twenty_percent():
    idx = pd.bdate_range("2020-01-01", periods=3)
    fig = ui.leverage_equity_fig({"buy_and_hold": pd.DataFrame({1.0: [1.0, 0.9, 0.8]}, index=idx)}, 1.0)
    lines = [s for s in fig.layout.shapes if s.type == "line"]
    assert len(lines) == 1 and lines[0].y0 == lines[0].y1 == 0.2
    assert "20%" in fig.layout.annotations[-1].text


def test_growth_chart_marks_kelly_and_twice_kelly_on_the_gross_curve():
    L = np.round(np.linspace(0, 4, 41), 2)
    mu, s2 = 0.08, 0.04
    g = L * mu - L * L * s2 / 2
    k = pd.DataFrame({"L": L, "g": g, "g_lo": g - 0.02, "g_hi": g + 0.02, "mu": mu, "sigma2": s2,
                      "L_star": mu / s2, "L_star_lo": 1.0, "L_star_hi": 3.0})
    fig = ui.growth_fig(k)
    names = {t.name: t for t in fig.data}
    assert names["Kelly leverage"].x[0] == pytest.approx(2.0)
    two = names["2x Kelly: growth about zero"]
    assert two.x[0] == pytest.approx(4.0) and two.y[0] == 0.0           # the 2x Kelly marker exists, at growth zero
    assert "doubles" not in str([t.name for t in fig.data]).lower()


def test_decile_chart_always_up_line_uses_the_real_base_rate_not_one_half():
    dec = pd.DataFrame({"decile": range(1, 11), "hit_rate": 0.5, "n": 20, "mean_confidence": 0.01})
    fig = ui.decile_fig(dec, base_hit=0.573)
    ys = sorted(s.y0 for s in fig.layout.shapes)
    assert ys == pytest.approx([0.5, 0.573])


def test_zone_legend_crosses_out_neutral_and_not_reduce_when_neutral_has_no_width():
    edges = {"Sell / step aside": (0, 10), "Reduce": (10, 20), "Neutral": (20, 20), "Hold / buy": (20, 90),
             "Strong buy": (90, 100)}
    bounds = [{"zone": z, "lo": lo, "hi": hi, "reachable": hi > lo} for z, (lo, hi) in edges.items()]
    html_ = ui.zone_legend_html(bounds)
    chips = re.findall(r"<span[^>]*>([^<]+)</span>", html_)
    assert chips == list(E.ZONES)                                          # order and names intact
    styled = {m.group(2): "line-through" in m.group(1) for m in re.finditer(r"<span style='([^']*)'>([^<]+)</span>", html_)}
    assert styled["Neutral"] and not styled["Reduce"] and not styled["Hold / buy"]


def test_paper_chart_uses_date_only_labels():
    led = pd.DataFrame({"asset": "SPY", "date": pd.to_datetime(["2026-11-05", "2026-11-06"]),
                        "equity": [100.0, 101.0], "bh_equity": [100.0, 100.5], "vtbh_equity": [100.0, 100.2]})
    fig = ui.paper_equity_fig(led, "SPY")
    assert fig.layout.xaxis.tickformat == "%Y-%m-%d" and fig.layout.xaxis.hoverformat == "%Y-%m-%d"


def test_strategy_series_is_not_labelled_chosen_unless_asked():
    assert ui.SERIES_LABEL["strategy"] != ui.SERIES_LABEL["chosen_strategy"]
    assert "config.yaml" in ui.SERIES_LABEL["chosen_strategy"]
    idx = pd.bdate_range("2020-01-01", periods=3)
    cur = pd.DataFrame({"strategy": [1, 1.1, 1.2], "buy_and_hold": [1, 1, 1], "vol_targeted_bh": [1, 1, 1]}, index=idx)
    assert ui.equity_fig(cur).data[0].name == "Selected model's strategy"
    assert ui.equity_fig(cur, strategy_label=ui.SERIES_LABEL["chosen_strategy"]).data[0].name.startswith("Chosen strategy")


def test_trials_histogram_names_the_conservative_bar_and_the_trial_count():
    fig = ui.trials_hist_fig(pd.Series([0.1, 0.2, 0.3]), 1.4, 270)
    ann = " ".join(a.text for a in fig.layout.annotations)
    assert "conservative" in ann and "270" in ann and "1.40" in ann
    assert fig.layout.margin.b >= 56                                      # x-axis title not clipped


# =========================================================================== hygiene
def _unused_imports(path: Path) -> list:
    src = path.read_text()
    tree = ast.parse(src)
    names = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            names += [(a.asname or a.name).split(".")[0] for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module != "__future__":
            names += [a.asname or a.name for a in node.names]
    used = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} | {
        n.value.id for n in ast.walk(tree) if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name)}
    return [n for n in names if n not in used]


@pytest.mark.parametrize("f", ["quant/explain.py", "quant/dash_data.py", "app/ui.py", "app/dashboard.py"])
def test_no_unused_imports(f):
    assert _unused_imports(ROOT / f) == []


def test_theory_cards_and_panels_have_no_ignored_parameters():
    src = (ROOT / "app" / "dashboard.py").read_text()
    assert "def theory_card(entry: dict, expanded=False, heading=None)" in src
    assert "def overfit_panel(sig)" in src and "summ, idx = c_trials(sig)\n    st.markdown(\n        f\"**What it looks like" not in src
    assert "chosen" not in re.search(r"def trials_hist_fig\(([^)]*)\)", (ROOT / "app" / "ui.py").read_text()).group(1)


# =========================================================================== theory.yaml disclosures
TH = E.load_theory()


def test_dead_or_unverified_links_are_annotated_not_removed():
    notes = {s["url"]: s["link_note"] for b in ("feature_groups", "models", "reality_check") for e in TH[b].values()
             for s in e["sources"] if s.get("link_note")}
    assert notes == {
        "https://content.govdelivery.com/accounts/USIMF/bulletins/30520df": "(link no longer works)",
        "https://en.wikipedia.org/wiki/Purged_cross-validation": "(link no longer works)"}
    assert "(link no longer works)" in E.source_line(TH["feature_groups"]["cross_asset"]["sources"][0])


def test_strong_buy_rule_and_all_badge_conditions_are_documented():
    meter = " ".join(str(TH["models"]["signal_meter"]["plain_english"]).split())
    assert "Strong buy" in meter and "no \"strong buy\" action is invented" in meter and "needle is today's gap" in meter
    assert "percentile of today's gap" in meter and "small print" in meter
    badge = " ".join(str(TH["models"]["evidence_badge"]["plain_english"]).split())
    for want in ("Pesaran-Timmermann", "Brier skill", "BOTH", "PSR of the excess return", "Deflated Sharpe",
                 "this asset's trials", "all trials", "volatility-targeted buy-and-hold"):
        assert want in badge, want
    assert "SKILL" in TH["models"]["evidence_badge"]["formula"] and "DSR_G" in TH["models"]["evidence_badge"]["formula"]
    assert "evaluation window" in " ".join(str(TH["models"]["brier_skill"]["plain_english"]).split())


def test_btc_cost_card_and_holdout_power_wording():
    costs = " ".join(str(TH["reality_check"]["edges_negative_after_costs"]["plain_english"]).split())
    assert "1- to 60-minute" in costs and "limited" in costs
    assert "0.71" in str(TH["reality_check"]["holdout_power"]["plain_english"])
