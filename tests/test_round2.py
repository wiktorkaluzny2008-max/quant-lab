"""Stage 5, round-2 critic fixes.

Sections: A honesty wording, B dashboard cosmetics, C leakage, D robustness/dead code, E mutation killers.
Page tests run the real app (AppTest) on the real data or on a private COPY of data/ (the real files are
never modified).  Pure helpers of ``app/dashboard.py`` are tested through ``load_dashboard()`` which runs
the file without its final ``main()`` call.
"""
import copy
import json
import os
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from app import ui
from quant import dash_data as DD
from quant import explain as E
from quant import metrics as M
from quant import paper as P
from quant import run_daily as R
from tests import dash_world as W

ROOT = Path(__file__).resolve().parents[1]
APP = str(ROOT / "app" / "dashboard.py")
CFG = DD._cfg()
TICKERS = DD.tickers(CFG)
FOOTER = "Research and education only, not financial advice."
CUT = pd.Timestamp(CFG["holdout_cutoff_date"])


# --------------------------------------------------------------------------- helpers
def run_page(page: str, **state) -> AppTest:
    at = AppTest.from_file(APP, default_timeout=300)
    at.session_state["page"] = page
    for k, v in state.items():
        at.session_state[k] = v
    return at.run()


def clear():
    st.cache_data.clear()
    st.cache_resource.clear()


@pytest.fixture(autouse=True)
def fresh_caches():
    clear()
    yield
    clear()


@pytest.fixture
def tmpdata(tmp_path, monkeypatch):
    """The app pointed at a private copy of data/."""
    cfg = copy.deepcopy(CFG)
    dd = tmp_path / "data"
    shutil.copytree(CFG["data_dir"], dd)
    cfg["data_dir"] = dd
    monkeypatch.setattr(DD, "_cfg", lambda c=None: c if c is not None else cfg)
    return cfg


def blob(at) -> str:
    out = []
    for kind in ("markdown", "caption", "info", "warning", "success", "error", "title", "header", "subheader", "code"):
        out += [str(e.value) for e in at.get(kind)]
    for m in at.metric:
        out += [str(m.label), str(m.value)]
    return "\n".join(out)


def messages(at) -> str:
    return "\n".join(str(x.value) for x in list(at.error) + list(at.warning) + list(at.info))


def no_crash(at):
    assert not at.exception, [e.value for e in at.exception]
    assert "Traceback" not in blob(at)
    assert any(FOOTER == str(c.value) for c in at.caption)


def metric(at, label):
    return next(m for m in at.metric if m.label == label)


def specs(at) -> list:
    return [json.loads(e.proto.spec) for e in at.get("plotly_chart")]


_DASH = {}


def load_dashboard() -> dict:
    """Namespace of app/dashboard.py with the final ``main()`` call removed (so its helpers can be unit-tested)."""
    if not _DASH:
        src = (ROOT / "app" / "dashboard.py").read_text()
        assert src.rstrip().endswith("main()")
        ns = {"__name__": "dash_ns", "__file__": APP}
        exec(compile(src.rstrip()[: -len("main()")], APP, "exec"), ns)
        _DASH.update(ns)
    return _DASH


# =========================================================================== A. statistical honesty
def test_random_percentile_is_labelled_gross_and_shows_the_vol_targeted_baseline_computed_from_data():
    expect = {"SPY": 98.2, "QQQ": 98.1, "IWM": 72.1, "BTC-USD": 66.2}     # the critic's reference values
    for t in TICKERS:
        res = DD.load_asset_results(t)
        vp = DD.vt_baseline_percentile(res)
        assert vp == pytest.approx(expect[t], abs=0.05), t
        at = run_page("Backtest", bt_asset=t)
        no_crash(at)
        assert metric(at, "Vol-targeted buy and hold percentile (same test)").value == f"{vp:.0f}"
        text = blob(at)
        assert "GROSS returns" in text and "mostly reflects vol targeting" in text
        assert f"scores {vp:.1f} in the same test" in text
        gl = "\n".join(str(m.value) for m in at.markdown)
        assert "GROSS returns (before trading costs)" in gl


def test_baseline_percentile_is_the_saved_percentile_recipe():
    res = DD.load_asset_results("SPY")
    b = res.baselines
    b = b[(b["name"] == "vol_targeted_bh") & (b["window"] == "expanding")].sort_values("date")
    nr = DD.model_predictions(res).set_index("date")["next_return"].reindex(b["date"]).to_numpy()
    want = M.random_strategy_percentile(nr, b["position"].to_numpy(), n=1000, seed=0)["percentile"]
    assert DD.vt_baseline_percentile(res) == pytest.approx(want)
    assert DD.vt_baseline_percentile(res, "no_such_baseline") is None


def test_trials_page_explains_why_the_two_dsr_columns_differ():
    at = run_page("Trials")
    no_crash(at)
    md = "\n".join(str(m.value) for m in at.markdown)
    assert "Why do the two DSR columns differ?" in md and "The pooled bar is the stricter reading" in md
    summ = DD.trials_summary()
    spy = summ["assets"]["SPY"]
    ns = pd.Series({"asset": spy["sr0_asset"] * np.sqrt(252), "global": spy["sr0_global"] * np.sqrt(252)})
    assert f"about {ns['asset']:.2f} a year" in md and f"about {ns['global']:.2f} a year" in md
    assert 0.18 < ns["asset"] < 0.20 and 1.39 < ns["global"] < 1.41              # ~0.19 and ~1.40 as the critic said
    df = next(d for d in at.dataframe if "Luck bar, own trials" in d.value.columns)
    row = df.value.set_index("Asset")
    assert row.loc["SPY", "Luck bar, own trials"] == f"{ns['asset']:.2f}"
    assert row.loc["SPY", "Luck bar, all trials"] == f"{ns['global']:.2f}"
    btc = summ["assets"]["BTC-USD"]
    assert row.loc["BTC-USD", "Luck bar, own trials"] == f"{btc['sr0_asset'] * np.sqrt(365):.2f}"   # 365 for Bitcoin


def test_reality_check_states_that_the_report_figures_need_more_in_this_projects_samples():
    at = run_page("Reality check")
    no_crash(at)
    text = blob(at)
    assert "figures are the research report's own" in text and "samples are shorter, so the bar is higher" in text
    df = next(d for d in at.dataframe if "Annual Sharpe needed for t = 3" in d.value.columns).value.set_index("Asset")
    spy_y = len(DD.model_predictions(DD.load_asset_results("SPY"))) / 252
    btc_y = len(DD.model_predictions(DD.load_asset_results("BTC-USD"))) / 365
    assert float(df.loc["SPY", "Out-of-sample years"]) == pytest.approx(spy_y, abs=0.05)
    assert df.loc["SPY", "Annual Sharpe needed for t = 3"] == f"{3 / np.sqrt(spy_y):.2f}"
    assert df.loc["BTC-USD", "Annual Sharpe needed for t = 3"] == f"{3 / np.sqrt(btc_y):.2f}"
    assert df.loc["SPY", "Annual Sharpe needed for t = 3"] == "0.78" and df.loc["BTC-USD", "Annual Sharpe needed for t = 3"] == "1.83"


def test_sharpe_needed_helper():
    assert DD.sharpe_needed_for_t(9.0) == pytest.approx(1.0) and DD.sharpe_needed_for_t(4.0, 2.0) == pytest.approx(1.0)
    assert np.isnan(DD.sharpe_needed_for_t(0.0))


def test_break_even_display_is_not_over_precise():
    be = load_dashboard()["be_text"]
    row = lambda v, **k: {"break_even_cost": v, "kind": "trial", "name": "x", **k}      # noqa: E731
    assert be(row(0.11117)) == "11.1%"                 # the cosmetic "11.117%" of vol-targeted trials
    assert be(row(0.01)) == "1.0%"                     # 1% and above: one decimal
    assert be(row(0.0099)) == "0.990%"                 # below 1%: three decimals (small numbers matter)
    assert be(row(0.00123)) == "0.123%"


# =========================================================================== B. cosmetics
def test_diagnostics_metrics_use_a_3_plus_2_layout_with_full_labels():
    at = run_page("Model diagnostics")
    no_crash(at)
    labels = [m.label for m in at.metric]
    for want in ("AUC (0.5 = coin flip)", "Brier skill vs training base rate", "Brier skill vs evaluation base rate",
                 "Pesaran-Timmermann p-value", "Said 'up' on"):
        assert want in labels
    cols = [c for c in at.columns if any(m.label in labels for m in c.metric)]
    assert len(cols) == 5 and max(len(c.metric) for c in cols) == 1
    # first three in one row of three columns, the other two in a second row
    proto_widths = {round(c.proto.weight, 3) for c in cols}
    assert len(proto_widths) == 1 and abs(proto_widths.pop() - 1 / 3) < 0.01


def test_importance_axis_title_is_short_and_ticks_have_no_micro_prefix():
    imp = pd.DataFrame({"group": ["a", "b"], "importance_mean": [0.00021, -0.0001], "importance_se": [0.0001, 0.0001]})
    fig = ui.importance_fig(imp)
    assert len(fig.layout.xaxis.title.text) <= 50
    assert fig.layout.xaxis.exponentformat == "none" and fig.layout.xaxis.tickformat == ".4f"


def test_importance_bar_colours_green_for_positive_red_otherwise():
    imp = pd.DataFrame({"group": ["neg", "zero", "pos"], "importance_mean": [-0.01, 0.0, 0.02], "importance_se": [0.001] * 3})
    fig = ui.importance_fig(imp)
    by_group = dict(zip(fig.data[0].y, fig.data[0].marker.color))
    assert by_group["pos"] == "#5BBF7A" and by_group["neg"] == "#E5534B" and by_group["zero"] == "#E5534B"
    assert list(fig.data[0].x) == sorted(imp["importance_mean"])           # sorted ascending, bars keep their values


def test_trials_histogram_annotation_sits_beside_the_line_with_room_and_line_is_at_the_luck_bar():
    fig = ui.trials_hist_fig(pd.Series([0.1, 0.2, 0.3, 0.5, 0.9]), 1.4, 270)
    ann = fig.layout.annotations[0]
    assert ann.xanchor == "left" and "<br>" in ann.text                   # two short lines, to the right of the line
    assert fig.layout.xaxis.range[1] > 1.4 + 0.3                           # the label has room
    line = [s for s in fig.layout.shapes if s.type == "line"][0]
    assert line.x0 == 1.4 and line.x1 == 1.4                               # luck_line_x


def test_quantpedia_link_has_no_unverified_note_and_the_link_answers():
    th = E.load_theory()
    srcs = [s for b in ("feature_groups", "models", "reality_check") for e in th[b].values() for s in e["sources"]]
    q = [s for s in srcs if "quantpedia.com" in s["url"]]
    assert q and all("link_note" not in s for s in q)
    assert "could not be verified" not in (ROOT / "quant" / "theory.yaml").read_text()


def test_today_says_needle_is_todays_gap_and_sits_in_the_paper_account_zone():
    at = run_page("Today")
    no_crash(at)
    caps = [str(c.value) for c in at.caption]
    md = [str(m.value) for m in at.markdown]
    assert sum("The needle is today's gap:" in c for c in md) == 1                    # the page intro
    assert sum("always inside the zone of the paper-trading position" in c for c in md) == len(TICKERS)
    assert not any("RELATIVE ranking" in c for c in caps)


def test_how_it_works_badge_text_has_no_round_one_wording_or_hyphen_artefact():
    at = run_page("How it works")
    no_crash(at)
    everything = blob(at) + "\n".join(str(d.value.to_csv()) for d in at.dataframe)
    for bad in ("round-1", "critic-driven", "evaluation- window", "evaluation-window base rate- "):
        assert bad not in everything and bad not in E.BADGE_RULE
    th = E.load_theory()
    txt = " ".join(str(th["models"]["evidence_badge"]["plain_english"]).split())
    assert "round-1" not in txt and "- window" not in txt and "base rate of the evaluation window" in txt
    for e in th["models"].values():
        assert "round-1" not in str(e) and "critic" not in str(e)


def test_why_history_phrase_names_the_years():
    rep = E.today_report("SPY", CFG)
    sentences = " ".join(s["sentence"] for s in rep["why"].saw)
    assert re.search(r"from \d{4} to 2024", sentences) and " since 20" not in sentences and " since 19" not in sentences
    f = E.feature_sentence("SPY", "r5", dict(column="r5", value=0.0, n_history=100, history_start="2000", history_end="2024",
                                              below=50.0, above=40.0, q25=-0.01, q75=0.01, share_equal=0.0))
    assert "higher than 50% of days from 2000 to 2024" in f


# =========================================================================== C. leakage
def test_rolling_brier_skill_is_backward_looking():
    rng = np.random.default_rng(3)
    n, w = 400, 60
    base = np.full(n, 0.55)
    y = (rng.random(n) < 0.55).astype(float)
    p = np.clip(base + rng.normal(0, 0.03, n), 0.05, 0.95)
    df = pd.DataFrame({"date": pd.bdate_range("2015-01-01", periods=n), "p_up": p, "y": y, "base_rate": base})
    r = DD.rolling_brier_skill(df, w)
    for i in (w - 1, 150, 300, n - 1):
        want = M.brier_skill(y[i - w + 1: i + 1], p[i - w + 1: i + 1], base[i - w + 1: i + 1])
        assert r.iloc[i] == pytest.approx(want)                           # equals metrics.brier_skill on the window ending at i
    assert r.iloc[: w - 1].isna().all()
    cut = 200
    df2 = df.copy()
    df2.loc[cut + 1:, ["p_up", "y", "base_rate"]] = [0.9, 1.0, 0.1]        # change every later row
    r2 = DD.rolling_brier_skill(df2, w)
    pd.testing.assert_series_equal(r.iloc[: cut + 1], r2.iloc[: cut + 1])  # nothing before/at the cut moved
    assert not np.allclose(r.iloc[cut + 1:].to_numpy(), r2.iloc[cut + 1:].to_numpy(), equal_nan=True)
    for fn in (DD.rolling_brier_skill_eval, DD.rolling_auc):
        a, b = fn(df, w), fn(df2, w)
        pd.testing.assert_series_equal(a.iloc[: cut + 1], b.iloc[: cut + 1])


@pytest.fixture(scope="module")
def world2(tmp_path_factory):
    return W.build(tmp_path_factory.mktemp("r2world"), extra_days=("2021-10-14",))


def test_today_report_refuses_the_live_model_for_a_day_that_is_not_the_latest(world2):
    cfg = world2
    for t in W.TICKERS:
        led = DD.ledger_for(t, cfg)
        assert len(led) >= 2
        old, last = led["date"].iloc[-2], led["date"].iloc[-1]
        ok = E.today_report(t, cfg, ds=W.dataset(cfg, t))
        assert ok["agree"] is not None and ok["why"] is not None and ok["date"] == str(last.date())
        past = E.today_report(t, cfg, asof=old, ds=W.dataset(cfg, t, asof=old))
        assert past["agree"] is None and past["why"] is None
        assert "look ahead" in past["warning"] and str(last.date()) in past["warning"]
        same = E.today_report(t, cfg, asof=last, ds=W.dataset(cfg, t))          # asking for the latest day explicitly is fine
        assert same["agree"] is not None
        model, reason = E.load_live_model_ex(t, cfg, asof=old)
        assert model is None and "look ahead" in reason


def test_live_model_refused_when_trained_after_the_decision_day(world2, monkeypatch):
    cfg = world2
    t = W.TICKERS[0]
    last = DD.ledger_for(t, cfg)["date"].iloc[-1]
    state = DD.load_state(cfg)
    state["assets"][t]["train_date"] = str((last + pd.Timedelta(days=3)).date())
    monkeypatch.setattr(DD, "load_state", lambda c=None: state)
    model, reason = E.load_live_model_ex(t, cfg, asof=last)
    assert model is None and "trained on" in reason and "after the decision day" in reason
    state["assets"][t]["train_date"] = str(last.date())                       # trained ON the decision day: fine
    assert E.load_live_model_ex(t, cfg, asof=last)[0] is not None
    assert E.live_model_guard(t, cfg) is None


def test_confidence_bucket_does_not_fail_on_an_empty_or_short_frame():
    empty = pd.DataFrame({"date": pd.to_datetime([]), "p_up": [], "y": [], "base_rate": []})
    tb = E.confidence_bucket(empty, 0.55, 0.53)
    assert tb["available"] is False and tb["n_total"] == 0 and tb["decile"] is None and tb["n"] == 0
    assert np.isnan(tb["hit_rate"]) and np.isnan(tb["overall_hit_rate"])
    few = pd.DataFrame({"date": pd.bdate_range("2020-01-01", periods=5), "p_up": [0.5] * 5, "y": [1.0, 0, 1, 1, 0],
                        "base_rate": [0.5] * 5})
    assert E.confidence_bucket(few, 0.55, 0.5)["available"] is False
    ok = pd.DataFrame({"date": pd.bdate_range("2020-01-01", periods=100), "p_up": np.linspace(0.4, 0.6, 100),
                       "y": np.tile([0.0, 1.0], 50), "base_rate": [0.5] * 100})
    assert E.confidence_bucket(ok, 0.55, 0.5)["available"] is True
    # asof before every prediction -> empty again, still no failure
    assert E.confidence_bucket(ok, 0.55, 0.5, asof="2019-01-01")["available"] is False


# =========================================================================== D. robustness
def test_libomp_condition_is_tested_before_the_generic_os_error():
    ef = load_dashboard()["explain_failure"]
    level, msg = ef(OSError("dlopen(lib_lightgbm.dylib): Library not loaded: libomp.dylib"))
    assert level == "error" and msg.startswith("LightGBM could not load") and "brew install libomp" in msg
    level, msg = ef(ImportError("lightgbm is not importable"))
    assert msg.startswith("LightGBM could not load")
    level, msg = ef(OSError("disk read error"))
    assert "could not be read" in msg and "LightGBM" not in msg
    level, msg = ef(FileNotFoundError("results/x.parquet"))
    assert level == "warning" and "saved results are missing" in msg


@pytest.mark.parametrize("exc", [LookupError("x"), KeyError("k"), IndexError("i")])
def test_lookup_family_is_one_expected_data_is_missing_message(exc):
    level, msg = load_dashboard()["explain_failure"](exc)
    assert level == "warning" and msg.startswith("Some expected data is missing")


def test_value_error_and_unknown_exceptions_keep_their_own_messages():
    ef = load_dashboard()["explain_failure"]
    assert "empty or inconsistent" in ef(ValueError("v"))[1]
    assert ef(RuntimeError("odd"))[0] == "error" and "unexpected" in ef(RuntimeError("odd"))[1]
    assert "Blocked" in ef(P.HoldoutError("late"))[1] and "damaged" in ef(P.PaperError("bad"))[1]


def test_missing_state_file_with_intact_ledger_says_so(tmpdata):
    (tmpdata["data_dir"] / "paper" / "state.json").unlink()
    assert not DD.ledger_for("SPY", tmpdata).empty
    at = run_page("Today")
    no_crash(at)
    text = messages(at)
    assert "state file missing" in text
    assert "no paper-trading decision yet" not in text and "('assets')" not in text
    with pytest.raises(P.PaperError, match="state file missing"):
        E.today_meter("SPY", tmpdata)


def test_a_really_empty_ledger_still_says_no_decision_yet(tmpdata):
    led = DD.load_ledger(tmpdata)
    led.iloc[0:0].to_parquet(tmpdata["data_dir"] / "paper" / "ledger.parquet", index=False)
    at = run_page("Today")
    no_crash(at)
    assert sum("no paper-trading decision yet" in str(i.value) for i in at.info) == len(TICKERS)
    with pytest.raises(E.NoDecisionError):
        E.today_meter("SPY", tmpdata)
    assert issubclass(E.NoDecisionError, LookupError)


def test_empty_predictions_file_gives_a_clear_message_not_bin_edges(tmpdata):
    p = tmpdata["data_dir"] / "results" / "SPY" / "predictions.parquet"
    pd.read_parquet(p).iloc[0:0].to_parquet(p, index=False)
    at = run_page("Today")
    no_crash(at)
    text = messages(at)
    assert "Bin edges" not in text and "predictions of the ensemble model are empty" in text
    assert "SPY" in text


def test_refresh_with_skipped_assets_is_not_ok_and_shows_amber(monkeypatch):
    class Res:
        messages, skipped, wrote, new_rows = [], {"SPY": "STALE DATA - old"}, False, None
    out = DD.refresh_data(CFG, run=lambda cfg, **kw: Res())
    assert out.ok is False and "Only partly refreshed" in out.message and "SPY" in out.message
    monkeypatch.setattr(R, "run", lambda cfg, **kw: Res())
    at = AppTest.from_file(APP, default_timeout=300)
    at.session_state["page"] = "Today"
    at.run()
    at.sidebar.button(key="refresh").click().run()
    assert any("Only partly refreshed" in str(w.value) for w in at.sidebar.warning)
    assert not list(at.sidebar.success)


def test_repeated_identical_errors_are_shown_once_per_page(tmpdata):
    (tmpdata["data_dir"] / "paper" / "state.json").write_text("{ not json")
    at = run_page("Today")
    no_crash(at)
    errs = [str(e.value) for e in at.error]
    assert sum("damaged or unreadable" in e for e in errs) == 1
    assert sum("Same problem as in the card above" in str(c.value) for c in at.caption) == len(TICKERS) - 1


def test_dead_code_is_gone():
    assert not hasattr(E, "load_live_model") and not hasattr(DD, "hit_rate_deciles")
    assert not hasattr(DD.AssetResults, "available") and not hasattr(ui, "ACCENT") and not hasattr(ui, "LEV_COLORS")
    assert not hasattr(E.MeterReading, "as_dict") and not hasattr(E.Why, "as_dict")


# =========================================================================== E. mutation killers: meter + badge
BASE = dict(pt_pvalue=0.01, pt_statistic=2.0, brier_skill=0.01, brier_skill_eval_base=0.01, sharpe=1.2, sharpe_vt=0.9,
            psr_excess_vt=0.99, dsr_excess_vt=0.99, dsr_global=0.99)


def badge(**kw):
    return E.badge_from_numbers(**{**BASE, **kw})


def passed(b, k):
    return b["checks"][k]["passed"]


def test_badge_boundaries_skill_pt_beats_and_probability_thresholds():
    assert badge()["level"] == "evidence"
    # SKILL: strictly greater than zero, on BOTH reference rates
    assert not passed(badge(brier_skill=0.0, brier_skill_eval_base=0.0), "SKILL")
    assert not passed(badge(brier_skill=0.0), "SKILL") and not passed(badge(brier_skill_eval_base=0.0), "SKILL")
    assert not passed(badge(brier_skill=0.01, brier_skill_eval_base=-1e-9), "SKILL")
    assert passed(badge(brier_skill=1e-9, brier_skill_eval_base=1e-9), "SKILL")
    assert badge(brier_skill=0.0, brier_skill_eval_base=0.0)["level"] != "evidence"
    # PT: p strictly below 0.05 and statistic strictly above 0
    assert not passed(badge(pt_statistic=0.0), "PT") and passed(badge(pt_statistic=1e-9), "PT")
    assert not passed(badge(pt_pvalue=0.05), "PT") and passed(badge(pt_pvalue=0.0499), "PT")
    assert not passed(badge(pt_statistic=-3.0, pt_pvalue=0.001), "PT")
    # BEATS: strictly greater Sharpe
    assert not passed(badge(sharpe=0.9), "BEATS") and passed(badge(sharpe=0.9001), "BEATS")
    # PSR / DSR: 0.95 counts (>=), just below does not
    for k, name in (("PSR_X", "psr_excess_vt"), ("DSR_X", "dsr_excess_vt"), ("DSR_G", "dsr_global")):
        assert passed(badge(**{name: 0.95}), k), k
        assert not passed(badge(**{name: 0.9499999}), k), k
    assert badge(dsr_global=0.95)["level"] == "evidence" and badge(dsr_excess_vt=0.95)["level"] == "evidence"
    assert badge(dsr_global=0.94)["level"] == "weak"


def test_badge_levels_amber_and_grey_rules():
    assert badge(pt_statistic=0.0)["level"] == "weak"                      # BEATS and PSR_X
    assert badge(pt_statistic=0.0, psr_excess_vt=0.9)["level"] == "none"   # neither PT nor PSR_X
    assert badge(psr_excess_vt=0.9, dsr_excess_vt=0.9)["level"] == "weak"  # BEATS and PT
    assert badge(sharpe=0.9)["level"] == "none"                            # no BEATS -> never amber
    assert badge(sharpe=0.5, psr_excess_vt=0.99, pt_pvalue=0.001)["level"] == "none"


@pytest.mark.parametrize("name", list(BASE))
def test_any_nan_or_missing_number_can_never_raise_the_badge(name):
    for bad in (float("nan"), None, "x"):
        b = badge(**{name: bad})
        assert b["checks"] is not None
        assert b["level"] != "evidence" or name == "psr_excess_vt"        # PSR_X only matters for amber
        if name == "psr_excess_vt":
            assert not passed(b, "PSR_X")
    assert not passed(badge(brier_skill_eval_base=float("nan")), "SKILL")      # one NaN skill is enough to fail it
    assert not passed(badge(brier_skill=float("nan")), "SKILL")
    allnan = E.badge_from_numbers(**{k: float("nan") for k in BASE})
    assert allnan["level"] == "none" and allnan["faded"] and allnan["note"] == E.FADED_NOTE


def test_badge_ui_colours_and_labels():
    for level, colour in (("none", "grey"), ("weak", "amber"), ("evidence", "green")):
        b = {"none": badge(sharpe=0.5), "weak": badge(pt_statistic=0.0), "evidence": badge()}[level]
        assert b["color"] == colour
        h = ui.badge_html(b)
        assert ui.BADGE_COLORS[colour] in h and f"evidence badge ({colour})" in h and b["label"] in h
    assert len({ui.BADGE_COLORS[c] for c in ("grey", "amber", "green")}) == 3
    assert badge()["faded"] is False and badge()["note"] is None and badge(sharpe=0.5)["faded"] is True


@pytest.mark.parametrize("mode,pos,want", [
    ("long_flat", 0.0, 0), ("long_flat", 1e-12, 0), ("long_flat", 2e-12, 3), ("long_flat", 1.0, 3), ("long_flat", 0.5, 3),
    ("long_flat", float("nan"), 0), ("default_long", 1.0, 3), ("default_long", 0.0, 0),
    ("sized", 0.0, 0), ("sized", 0.24999, 1), ("sized", 0.25, 2),
    ("sized", 0.49999, 2), ("sized", 0.5, 3), ("sized", 0.99, 3),
    ("sized", 1.0 - 2e-9, 3), ("sized", 1.0 - 1e-10, 4), ("sized", 1.0, 4),
    ("sized", 1e-9, 1), ("sized", 1.0 - 1e-9, 4), ("sized", float("nan"), 0)])
def test_classify_zone_thresholds(mode, pos, want):
    assert E.classify_zone(mode, pos) == want


def test_classify_zone_sized_without_sizing_information_is_hold_when_invested():
    assert E.classify_zone("sized", 0.3, cuts_known=False) == 3 and E.classify_zone("sized", 1.0, cuts_known=False) == 3
    assert E.classify_zone("sized", 0.0, cuts_known=False) == 0


def test_classify_zone_constants_are_pinned():
    assert not hasattr(E, "STRONG_PCTL")                # no invented 'top decile' rule any more
    assert E.SIZED_CUTS == (0.25, 0.5, 1.0) and E.ZERO == 1e-12 and E.FULL == 1.0 - 1e-9


def pb(h, g):
    return 100.0 * float(np.mean(np.asarray(h) < g))


def sized_settings(vt=True):
    return P.PaperSettings(strategy_mode="sized", strategy={"window": "expanding", "model": "ensemble", "mode": "sized",
                           "vol_targeted": vt, "m_setting": "adaptive"}, sized_lambda=0.25, sized_band=0.1, capital={}, costs={})


def binary_settings(mode="long_flat"):
    return P.PaperSettings(strategy_mode=mode, strategy={"window": "expanding", "model": "ensemble", "mode": mode,
                           "vol_targeted": True, "m_setting": "adaptive"}, sized_lambda=0.25, sized_band=0.1, capital={}, costs={})


def test_zone_edges_for_sized_mode_are_the_position_cuts_in_gap_space():
    h = np.linspace(-0.1, 0.1, 2001)
    ratio, scale = 2.0, 0.04
    edges, lo, hi = E.zone_edges(sized_settings(True), h, 0.01, None, ratio, scale)
    cuts = [c * scale / (0.25 * ratio) for c in (0.25, 0.5, 1.0)]
    assert edges[1:5] == pytest.approx([0.0] + cuts) and edges[0] == lo and edges[5] == hi and lo < 0 and hi > cuts[-1]
    edges1, _, _ = E.zone_edges(sized_settings(False), h, 0.01, None, ratio, scale)      # vol ratio ignored when not vol-targeted
    assert edges1[2:5] == pytest.approx([c * scale / 0.25 for c in (0.25, 0.5, 1.0)])
    flat, _, _ = E.zone_edges(sized_settings(True), h, 0.01, None, float("nan"), scale)  # no usable sizing information
    assert flat[1] == 0.0 and flat[2] == flat[3] == 0.0 and flat[4] == flat[5]


def test_zone_edges_for_binary_modes_use_minus_m_and_have_no_strong_zone():
    h = np.linspace(-0.1, 0.1, 2001)
    for mode in ("long_flat", "default_long"):
        e, lo, hi = E.zone_edges(binary_settings(mode), h, 0.0, 0.02, 1.0, 0.02)
        assert e[1] == e[2] == e[3] == -0.02 and e[4] == e[5] == hi and e[0] == lo
        m = E.meter_from_inputs(binary_settings(mode), 0.5, 0.5, 0.02, 1.0, 0.02, 0.0, h)
        assert [x["reachable"] for x in m.bounds] == [True, False, False, True, False]
        assert m.zones_used == ("Sell / step aside", "Hold / buy") and m.threshold_gap == -0.02


def test_meter_needle_is_todays_gap_and_sits_in_its_zone_arc():
    h = np.linspace(-0.05, 0.05, 1001)
    m = E.meter_from_inputs(binary_settings(), 0.60, 0.50, 0.01, 1.0, 0.02, 0.0, h)         # gap +0.10
    assert m.zone == "Hold / buy" and m.percentile > 99 and m.needle == pytest.approx(0.10) and not m.needle_clamped
    low = E.meter_from_inputs(binary_settings(), 0.40, 0.50, 0.01, 1.0, 0.02, 0.0, h)       # gap -0.10 -> step aside
    assert low.zone == "Sell / step aside" and low.position == 0.0 and low.needle == pytest.approx(-0.10)
    assert low.needle < low.bounds[0]["hi"] == -0.01
    hold = E.meter_from_inputs(binary_settings(), 0.505, 0.50, 0.01, 1.0, 0.02, 0.0, h)
    assert hold.zone == "Hold / buy" and hold.bounds[3]["lo"] <= hold.needle <= hold.bounds[3]["hi"]
    nohist = E.meter_from_inputs(binary_settings(), 0.52, 0.50, 0.01, 1.0, 0.02, 0.0, [])
    assert np.isnan(nohist.percentile) and nohist.bounds[3]["lo"] <= nohist.needle <= nohist.bounds[3]["hi"]
    forced = E.meter_from_inputs(binary_settings(), 0.60, 0.50, 0.01, 1.0, 0.02, 0.0, h, position=0.0)
    assert forced.position == 0.0 and forced.zone == "Sell / step aside" and forced.position_matches_rule is False
    assert low.position_matches_rule is True


def test_ordinals_round_not_truncate_and_use_the_right_suffix():
    assert E.ordinal(22.6) == "23rd" and E.ordinal(22.4) == "22nd" and E.ordinal(0.6) == "1st"
    assert [E.ordinal(x) for x in (1, 2, 3, 4, 11, 12, 13, 21, 22, 23, 91, 100, 101, 111, 112)] == [
        "1st", "2nd", "3rd", "4th", "11th", "12th", "13th", "21st", "22nd", "23rd", "91st", "100th", "101st", "111th", "112th"]
    # ordinal(10) is "10th" with either lower range bound (10 or 11): that mutant is equivalent
    assert E.ordinal(10) == "10th" and E.ordinal(20) == "20th" and E.ordinal(14) == "14th"


class Stub:
    def __init__(self, p):
        self.p = p

    def predict_up(self, x):
        return np.array([self.p])


class StubEns:
    def __init__(self, **ps):
        self.members = {k: Stub(v) for k, v in ps.items()}


def test_models_agree_direction_and_tie_rules():
    x = pd.DataFrame({"x": [0.0]})
    tie = E.models_agree(StubEns(a=0.6, b=0.4), x, 0.5, None)            # 1 vs 1: counts as "above" (>=)
    assert tie["direction"] == "above" and tie["n_agree"] == 1 and tie["above_strategy_threshold"] is None
    exact = E.models_agree(StubEns(a=0.5, b=0.5, c=0.4), x, 0.5, 0.02)    # p == base leans above
    assert exact["direction"] == "above" and exact["n_agree"] == 2 and exact["n_above"] == 2
    assert exact["above_strategy_threshold"] == {"a": True, "b": True, "c": False}     # 0.4 < 0.5 - 0.02
    assert E.models_agree(StubEns(a=0.49, b=0.49, c=0.6), x, 0.5, 0.02)["direction"] == "below"
    assert E.models_agree(StubEns(a=0.49, b=0.49, c=0.6), x, 0.5, 0.02)["above_strategy_threshold"] == {
        "a": True, "b": True, "c": True}                                  # 0.49 >= 0.50 - 0.02
    with pytest.raises(ValueError):
        E.models_agree(object(), x, 0.5)


def test_feature_sentence_picks_the_nearer_tail_and_ties_say_higher():
    f = dict(column="r5", value=0.0, n_history=100, history_start="2010", history_end="2024", q25=-0.01, q75=0.01, share_equal=0.0)
    assert "higher than 40% of days" in E.feature_sentence("SPY", "r5", {**f, "below": 40.0, "above": 40.0})   # tie -> higher
    assert "higher than 41% of days" in E.feature_sentence("SPY", "r5", {**f, "below": 41.0, "above": 39.0})
    assert "lower than 41% of days" in E.feature_sentence("SPY", "r5", {**f, "below": 39.0, "above": 41.0})


def make_preds(conf, y=None, base=0.5):
    n = len(conf)
    p = base + np.asarray(conf, float)
    return pd.DataFrame({"date": pd.bdate_range("2015-01-01", periods=n), "p_up": p,
                         "y": np.ones(n) if y is None else y, "base_rate": np.full(n, base)})


def test_confidence_bucket_edges_and_overall_hit_rate():
    conf = np.arange(1, 101) / 1000.0
    df = make_preds(conf)
    top3 = df.iloc[29]                                   # the largest confidence of decile 3 (rank 30)
    tb = E.confidence_bucket(df, float(top3["p_up"]), 0.5)
    assert tb["decile"] == 3 and tb["conf_hi"] == pytest.approx(0.030)             # <= at the edge, not <
    assert E.confidence_bucket(df, float(df.iloc[30]["p_up"]), 0.5)["decile"] == 4
    big = E.confidence_bucket(df, 0.99, 0.5)
    assert big["decile"] == 10 and big["n"] == 10                                    # beyond every edge -> top decile
    # overall hit rate counts p == base as an "up" call
    y = np.r_[np.ones(10), np.zeros(90)]
    tied = make_preds(np.r_[np.zeros(10), np.arange(1, 91) / 1000.0], y=y)
    tied.loc[10:, "p_up"] = 0.5 - np.arange(1, 91) / 1000.0                          # the rest say "down" and were down
    tb = E.confidence_bucket(tied, 0.51, 0.5)
    assert tb["overall_hit_rate"] == pytest.approx(1.0)                              # p == base (10 rows) -> up -> right
    assert tb["always_up_rate"] == pytest.approx(0.1)
    asof = df["date"].iloc[60]                                                       # only strictly earlier predictions
    assert E.confidence_bucket(df, 0.55, 0.5, asof=asof)["n_total"] == 60


def test_conclusion_text_depends_on_the_zone_and_the_gap_sign(world2):
    cfg = world2
    t = W.TICKERS[0]
    ds = W.dataset(cfg, t)
    model = E.load_live_model_ex(t, cfg)[0]
    preds = DD.model_predictions(DD.load_asset_results(t, cfg), "ensemble")
    settings = DD.paper_settings(cfg)
    d = DD.ledger_for(t, cfg)["date"].iloc[-1]
    h = np.random.default_rng(7).normal(0.0, 0.012, 1500)
    bad = E.evidence_badge(t, cfg)

    def concl(p, base):
        m = E.meter_from_inputs(settings, p, base, 0.01, 1.0, 0.02, 0.0, h)
        return m, E.explain_today(t, ds, d, model, m, preds, bad, CUT).conclusion

    m0, c0 = concl(0.40, 0.55)
    assert m0.zone_index == 0 and "stepping aside means cash, not a short" in c0 and "long only" not in c0
    assert "below the usual rate, so this is not a bullish reading" in c0
    m1, c1 = concl(0.60, 0.55)
    assert m1.zone_index != 0 and "long only, never leveraged" in c1 and "stepping aside" not in c1
    assert "above the usual rate" in c1 and "not a bullish" not in c1
    assert "exactly the usual rate" in concl(0.55, 0.55)[1]


# =========================================================================== E. mutation killers: data layer
@pytest.fixture
def results_copy(tmp_path):
    """A private cfg whose data dir holds only copies of the SPY results (cheap, safe to corrupt)."""
    cfg = copy.deepcopy(CFG)
    dd = tmp_path / "data"
    (dd / "results").mkdir(parents=True)
    shutil.copytree(CFG["data_dir"] / "results" / "SPY", dd / "results" / "SPY")
    for f in ("trials_index.parquet",):
        shutil.copy(CFG["data_dir"] / "results" / f, dd / "results" / f)
    cfg["data_dir"] = dd
    return cfg


def test_metadata_dev_last_date_after_the_cutoff_is_blocked_even_slightly(results_copy):
    p = results_copy["data_dir"] / "results" / "SPY" / "metadata.json"
    meta = json.loads(p.read_text())
    assert pd.Timestamp(meta["dev_last_date"]) <= CUT
    assert DD.load_asset_results("SPY", results_copy).ticker == "SPY"
    for late in (CUT + pd.Timedelta(days=1), CUT + pd.Timedelta(days=30), CUT + pd.Timedelta(days=401)):
        meta["dev_last_date"] = str(late.date())
        p.write_text(json.dumps(meta))
        with pytest.raises(P.HoldoutError, match="dev_last_date"):
            DD.load_asset_results("SPY", results_copy)
    meta["dev_last_date"] = str(CUT.date())                       # exactly on the cutoff is allowed
    p.write_text(json.dumps(meta))
    DD.load_asset_results("SPY", results_copy)


def test_a_prediction_row_after_the_cutoff_is_blocked(results_copy):
    p = results_copy["data_dir"] / "results" / "SPY" / "predictions.parquet"
    df = pd.read_parquet(p)
    late = df.iloc[[-1]].copy()
    late["date"] = CUT + pd.Timedelta(days=1)
    pd.concat([df, late]).to_parquet(p, index=False)
    with pytest.raises(P.HoldoutError):
        DD.load_asset_results("SPY", results_copy)
    ok = df.iloc[[-1]].copy()
    ok["date"] = CUT
    pd.concat([df.iloc[:-1], ok]).to_parquet(p, index=False)
    DD.load_asset_results("SPY", results_copy)


def test_trial_returns_after_the_cutoff_are_blocked(results_copy):
    cid = DD.strategy_config_id("SPY", results_copy)
    tr = pd.read_parquet(CFG["data_dir"] / "results" / "trials.parquet", filters=[("config_id", "==", cid)])
    assert tr["date"].max() <= CUT
    out = results_copy["data_dir"] / "results" / "trials.parquet"
    tr.to_parquet(out, index=False)
    assert DD.load_trial_returns(cid, results_copy)["date"].max() <= CUT
    late = tr.iloc[[-1]].copy()
    late["date"] = CUT + pd.Timedelta(days=1)
    pd.concat([tr, late]).to_parquet(out, index=False)
    with pytest.raises(P.HoldoutError):
        DD.load_trial_returns(cid, results_copy)
    late["date"] = CUT + pd.Timedelta(days=400)
    pd.concat([tr, late]).to_parquet(out, index=False)
    with pytest.raises(P.HoldoutError):
        DD.load_trial_returns(cid, results_copy)


def test_data_signature_changes_with_every_file_the_pages_depend_on(tmpdata):
    cfg = tmpdata
    dd = cfg["data_dir"]
    rd = dd / "results"
    paths = [dd / "paper" / "ledger.parquet", dd / "paper" / "state.json", rd / "trials_index.parquet", rd / "trials.parquet",
             rd / "SPY" / "predictions.parquet", rd / "SPY" / "trading_metrics.parquet",
             dd / "paper" / "models" / "SPY.joblib", dd / "SPY.parquet", dd / "BTC-USD.parquet"]
    base = DD.data_signature(cfg)
    assert DD.data_signature(cfg) == base                                   # stable while nothing changes
    for p in paths:
        assert p.exists(), p
        st_ = p.stat()
        os.utime(p, ns=(st_.st_atime_ns, st_.st_mtime_ns + 5_000_000_000))
        new = DD.data_signature(cfg)
        assert new != base, f"{p.name} is not part of the cache key"
        base = new
    ledger = dd / "paper" / "ledger.parquet"
    before = DD.data_signature(cfg)
    ledger.write_bytes(ledger.read_bytes())                                # rewritten = a daily run happened
    assert DD.data_signature(cfg) != before
    ledger.unlink()                                                         # a vanished file changes it too
    assert DD.data_signature(cfg) != before


def test_the_equity_curve_follows_the_selected_training_window(monkeypatch):
    from quant import trials as T
    seen = {}
    real = ui.equity_fig
    monkeypatch.setattr(ui, "equity_fig", lambda cur, **k: (seen.__setitem__(win[0], cur), real(cur, **k))[1])
    win = [None]
    for w in ("expanding", "rolling4y"):
        win[0] = w
        clear()
        no_crash(run_page("Backtest", bt_asset="BTC-USD", bt_window=w))
    for w, cur in seen.items():
        cid = T.make_config_id("BTC-USD", w, "ensemble", "long_flat", True, T.ADAPTIVE)
        pd.testing.assert_frame_equal(cur, DD.equity_curves("BTC-USD", cid, window=w))
    a, b = seen["expanding"], seen["rolling4y"]
    common = a.index.intersection(b.index)
    assert len(common) > 100 and not np.allclose(a.loc[common, "strategy"], b.loc[common, "strategy"])


def test_oos_caption_names_the_first_and_last_day_of_the_equity_curve():
    at = run_page("Backtest", bt_asset="SPY")
    no_crash(at)
    from quant import trials as T
    cid = T.make_config_id("SPY", "expanding", "ensemble", "long_flat", True, T.ADAPTIVE)
    cur = DD.equity_curves("SPY", cid)
    want = f"Out of sample from {cur.index.min().date()} to {cur.index.max().date()}, after costs."
    assert want in [str(c.value) for c in at.caption]
    assert cur.index.max() <= CUT


def test_window_choices_list_expanding_first():
    for page, key in (("Backtest", "bt_window"), ("Model diagnostics", "dg_window")):
        at = run_page(page, **{"bt_asset" if page == "Backtest" else "dg_asset": "BTC-USD"})
        no_crash(at)
        assert list(at.selectbox(key=key).options) == ["expanding", "rolling4y"], page
        assert at.selectbox(key=key).value == "expanding"


def test_one_failing_today_card_does_not_take_the_page_down(monkeypatch):
    real = E.today_report

    def flaky(ticker, *a, **k):
        if ticker == "SPY":
            raise RuntimeError("something odd in SPY")        # not a PaperError / OSError / ValueError
        return real(ticker, *a, **k)
    monkeypatch.setattr(E, "today_report", flaky)
    at = run_page("Today")
    no_crash(at)
    assert any("something odd in SPY" in str(e.value) for e in at.error)
    gauges = [g for g in specs(at) if g["data"][0]["type"] == "indicator"]
    assert len(gauges) == len(TICKERS) - 1                       # the other three cards still draw their meters


def test_a_card_failure_with_a_lookup_error_is_an_expected_data_message_not_no_decision(monkeypatch):
    def boom(*a, **k):
        raise KeyError("assets")
    monkeypatch.setattr(E, "today_report", boom)
    at = run_page("Today")
    no_crash(at)
    text = messages(at)
    assert "Some expected data is missing" in text and "no paper-trading decision yet" not in text


# ---- paper page
def write_ledger(cfg, edit):
    files = P.PaperFiles.for_config(cfg)
    led = P.load_ledger(files)
    led = edit(led)
    P.save_ledger(files, led)


def test_paper_page_one_day_table_columns_are_not_swapped(tmpdata):
    def edit(led):
        led.loc[led["asset"] == "SPY", ["equity", "bh_equity", "vtbh_equity"]] = [10101.0, 10202.0, 10303.0]
        return led
    write_ledger(tmpdata, edit)
    at = run_page("Paper trading")
    no_crash(at)
    one = [d.value for d in at.dataframe if "Date" in d.value.columns and "Paper strategy ($)" in d.value.columns]
    assert len(one) == len(TICKERS)                                          # one single-day table per asset
    row = next(d for d in one if d["Paper strategy ($)"].iloc[0] == "10,101.00").iloc[0]
    assert row["Buy and hold ($)"] == "10,202.00" and row["Vol-targeted buy and hold ($)"] == "10,303.00"
    pl = next(d.value for d in at.dataframe if "Costs paid ($)" in d.value.columns).set_index("Asset")
    assert pl.loc["SPY", "Strategy ($)"] == "+101.00" and pl.loc["SPY", "Buy and hold ($)"] == "+202.00"
    assert pl.loc["SPY", "Vol-targeted buy and hold ($)"] == "+303.00"
    pos = next(d.value for d in at.dataframe if "Account value ($)" in d.value.columns)
    assert "10,101.00" in set(pos["Account value ($)"])


def test_paper_page_asset_without_rows_gets_a_note_but_no_table(tmpdata):
    write_ledger(tmpdata, lambda led: led[led["asset"] != "BTC-USD"])
    at = run_page("Paper trading")
    no_crash(at)
    infos = [str(i.value) for i in at.info]
    assert any(i.startswith("BTC-USD: only 0 day(s)") for i in infos)
    assert sum(" only 1 day(s)" in i for i in infos) == len(TICKERS) - 1
    one = [d.value for d in at.dataframe if "Paper strategy ($)" in d.value.columns and "Date" in d.value.columns]
    assert len(one) == len(TICKERS) - 1                                      # no single-day table for the empty asset


def test_paper_equity_figure_maps_each_series_to_its_own_column():
    led = pd.DataFrame({"asset": "SPY", "date": pd.to_datetime(["2026-11-05", "2026-11-06", "2026-11-09"]),
                        "equity": [1.0, 2.0, 3.0], "bh_equity": [10.0, 20.0, 30.0], "vtbh_equity": [100.0, 200.0, 300.0]})
    other = led.assign(asset="QQQ", equity=-1.0, bh_equity=-1.0, vtbh_equity=-1.0)
    fig = ui.paper_equity_fig(pd.concat([led.iloc[::-1], other]), "SPY")      # unsorted input, another asset mixed in
    got = {t.name: list(t.y) for t in fig.data}
    assert got == {"Paper strategy": [1.0, 2.0, 3.0], "Buy and hold": [10.0, 20.0, 30.0],
                   "Vol-targeted buy and hold": [100.0, 200.0, 300.0]}
    assert list(fig.data[0].x) == list(led["date"])


def test_leverage_table_shows_the_right_wipe_dates():
    res = DD.load_asset_results("BTC-USD")
    at = run_page("Backtest", bt_asset="BTC-USD", lev_L=3.0)
    no_crash(at)
    sel = next(d.value for d in at.dataframe if "Wiped out?" in d.value.columns)
    lab = {v: k for k, v in ui.SERIES_LABEL.items()}
    checked = 0
    for _, r in sel.iterrows():
        s = lab[r["Series"]]
        lv = DD.leverage_table(res, s)
        row = lv[lv["L"] == 3.0].iloc[0]
        want = f"YES, on {pd.Timestamp(row['wipe_date']).date()}" if row["wiped_out"] else "no"
        assert r["Wiped out?"] == want, s
        checked += int(row["wiped_out"])
    assert checked == 2 and set(sel["Leverage"]) == {"3x"}
    assert "YES, on 2022-05-17" in set(sel["Wiped out?"]) and "YES, on 2022-06-10" in set(sel["Wiped out?"])


# ---- trials page numbers
def test_trials_luck_bar_uses_exactly_the_logged_trial_count(monkeypatch):
    seen = {}
    real_h, real_c = ui.trials_hist_fig, ui.luck_curve_fig
    monkeypatch.setattr(ui, "trials_hist_fig", lambda s, luck, n=None, **k: (seen.update(hist=(luck, n)), real_h(s, luck, n, **k))[1])
    monkeypatch.setattr(ui, "luck_curve_fig", lambda ns, sh, marks, **k: (seen.update(marks=marks), real_c(ns, sh, marks, **k))[1])
    at = run_page("Trials")
    no_crash(at)
    summ = DD.trials_summary()
    n, var = summ["n_trials_global"], summ["daily_sharpe_variance"]
    want = M.expected_max_sharpe(n, var) * np.sqrt(252)
    assert M.expected_max_sharpe(n + 1, var) * np.sqrt(252) != pytest.approx(want, rel=1e-9)     # n+1 would show
    assert seen["hist"][0] == pytest.approx(want, rel=1e-12) and seen["hist"][1] == n
    assert seen["marks"] == {f"{n} tried": (n, pytest.approx(want, rel=1e-12))}
    md = "\n".join(str(m.value) for m in at.markdown)
    assert f"**{want:.2f}**" in md and f"would be about {M.expected_max_sharpe(n, var) * np.sqrt(365):.2f}" not in md
    assert f"about {M.expected_max_sharpe(n, var) * np.sqrt(365):.2f}" in blob(at)                    # Bitcoin 365-day bar
    assert metric(at, "Distinct configurations tried").value == str(n)


def test_reality_check_hold_out_power_table_values():
    at = run_page("Reality check")
    no_crash(at)
    df = next(d.value for d in at.dataframe if "Years of daily data" in d.value.columns)
    assert list(df["Years of daily data"]) == [1, 2, 5, 10]
    se = list(df["Standard error of the yearly Sharpe"])
    assert se == [f"{1 / np.sqrt(y):.2f}" for y in (1, 2, 5, 10)] == ["1.00", "0.71", "0.45", "0.32"]    # 252 days a year
    assert list(df["Standard error of the hit rate (points)"]) == ["3.1", "2.2", "1.4", "1.0"]       # 100*sqrt(0.25/n)


# ---- ui details
def test_sharpe_chart_draws_the_zero_line_and_reliability_uses_the_right_columns():
    t = pd.DataFrame({"label": ["a", "b"], "sharpe": [1.0, -0.5], "sharpe_se": [0.5, 0.25], "kind": ["trial", "baseline"]})
    fig = ui.sharpe_band_fig(t)
    zero = [s for s in fig.layout.shapes if s.type == "line"]
    assert len(zero) == 1 and zero[0].x0 == 0 and zero[0].x1 == 0 and zero[0].line.dash == "dot"
    assert list(fig.data[0].marker.color) == [ui.C_BH, ui.C_STRAT]        # baseline grey, trial blue (input reversed)
    rel = pd.DataFrame({"bin": [0, 1, 2], "lo": [0.4, 0.5, 0.6], "hi": [0.5, 0.6, 0.7], "mean_p": [0.45, 0.55, 0.66],
                        "obs_freq": [0.50, 0.52, 0.70], "count": [10, 0, 30]})
    f = ui.reliability_fig(rel)
    model = next(t_ for t_ in f.data if t_.name == "model")
    assert list(model.x) == [0.45, 0.66] and list(model.y) == [0.50, 0.70]            # empty bin left out
    bars = next(t_ for t_ in f.data if t_.name == "days in bin")
    assert list(bars.x) == pytest.approx([0.45, 0.55, 0.65]) and list(bars.y) == [10, 0, 30]
    diag = next(t_ for t_ in f.data if t_.name == "perfect calibration")
    assert list(diag.x) == [0, 1] and list(diag.y) == [0, 1]


def test_setup_note_text_is_pinned():
    note = load_dashboard()["SETUP_NOTE"]
    assert note == ("On a fresh Mac, LightGBM needs the libomp library: run `brew install libomp` (or "
                    "`sh scripts/fix_lightgbm_macos.sh`) and restart. See 'Setup on a fresh Mac' in PROGRESS.md.")
    assert "Setup on a fresh Mac" in (ROOT / "PROGRESS.md").read_text()


def test_break_even_text_rules():
    be = load_dashboard()["be_text"]
    ns = load_dashboard()
    base = lambda v, name, kind="baseline": {"break_even_cost": v, "kind": kind, "name": name}      # noqa: E731
    assert be(base(0.5, "buy_and_hold")) == ns["NO_BREAK_EVEN"] == "n/a (no trades/costless)"
    assert be(base(0.5, "always_up")) == ns["NO_BREAK_EVEN"]
    assert be(base(0.0123, "trend_200")) == "1.2%"                                   # other baselines show their number
    assert be(base(0.0021, "vol_targeted_bh")) == "0.210%"
    assert be(base(0.0021, "buy_and_hold", kind="trial")) == "0.210%"                # only BASELINE buy-and-hold is n/a
    assert be({"break_even_cost": -0.001, "kind": "trial", "name": "x"}) == ns["LOSES_ANYWAY"] == "n/a (loses even at zero cost)"
    assert be({"break_even_cost": 0.0, "kind": "trial", "name": "x"}) == "0.000%"
    for bad in (float("nan"), float("inf"), None):
        assert be({"break_even_cost": bad, "kind": "trial", "name": "x"}) == "n/a"


# ---- live model reasons
@pytest.fixture
def world_copy(world2, tmp_path):
    cfg = copy.deepcopy(world2)
    dd = tmp_path / "w"
    shutil.copytree(world2["data_dir"], dd)
    cfg["data_dir"] = dd
    return cfg


def test_load_live_model_ex_returns_the_model_and_no_reason(world_copy):
    model, reason = E.load_live_model_ex("SPY", world_copy)
    assert reason is None and set(model.members) == {"logreg", "rf", "lgbm"}


def test_load_live_model_ex_reasons(world_copy, monkeypatch):
    import joblib
    path = P.PaperFiles.for_config(world_copy).model("SPY")
    path.write_bytes(b"not a joblib file")
    m, why = E.load_live_model_ex("SPY", world_copy)
    assert m is None and why.startswith("the saved live model could not be read") and "libomp" not in why
    for text in ("dlopen libomp.dylib failed", "cannot load lib_lightgbm.dylib", "OpenMP runtime missing", "LightGBM broke"):
        monkeypatch.setattr(joblib, "load", lambda p, t=text: (_ for _ in ()).throw(OSError(t)))
        m, why = E.load_live_model_ex("SPY", world_copy)
        assert m is None and why.startswith(E.LIBOMP_HINT) and "Details:" in why, text
    path.unlink()
    m, why = E.load_live_model_ex("SPY", world_copy)
    assert m is None and "model file is missing (SPY.joblib)" in why and "Refresh data" in why
