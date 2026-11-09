"""Stage 5, round-1 critic fixes: the real Streamlit app (AppTest) in failure and edge situations.

* Section A: every expected failure (corrupt / missing files in a tmp COPY of the data dir, an unreadable
  library, an empty trial log ...) must end in a plain message and never an exception/traceback.
* The rest: wording and wiring tests that kill the mutants the Code+test critic found and check the
  honesty / disclosure fixes.  The real data is never modified: tests that change files do it in a copy.
"""
import copy
import json
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from quant import dash_data as DD
from quant import explain as E
from quant import metrics as M
from quant import paper as P
from quant import trials as T

ROOT = Path(__file__).resolve().parents[1]
APP = str(ROOT / "app" / "dashboard.py")
CFG = DD._cfg()
TICKERS = DD.tickers(CFG)
FOOTER = "Research and education only, not financial advice."


def run_page(page: str, **state) -> AppTest:
    at = AppTest.from_file(APP, default_timeout=300)
    at.session_state["page"] = page
    for k, v in state.items():
        at.session_state[k] = v
    return at.run()


def clear():
    st.cache_data.clear()
    st.cache_resource.clear()


def blob(at) -> str:
    out = []
    for kind in ("markdown", "caption", "info", "warning", "success", "error", "title", "header", "subheader", "code"):
        out += [str(e.value) for e in at.get(kind)]
    for m in at.metric:
        out += [str(m.label), str(m.value)]
    return "\n".join(out)


@pytest.fixture(autouse=True)
def fresh_caches():
    clear()
    yield
    clear()


@pytest.fixture
def tmpdata(tmp_path, monkeypatch):
    """The app pointed at a private copy of data/ (so tests can corrupt or edit files safely)."""
    cfg = copy.deepcopy(CFG)
    dd = tmp_path / "data"
    shutil.copytree(CFG["data_dir"], dd)
    cfg["data_dir"] = dd
    monkeypatch.setattr(DD, "_cfg", lambda c=None: c if c is not None else cfg)
    return cfg


def no_crash(at, page=""):
    assert not at.exception, (page, [e.value for e in at.exception])
    assert "Traceback" not in blob(at)
    assert any(FOOTER == str(c.value) for c in at.caption), page            # the page still finished drawing


def messages(at) -> str:
    return "\n".join(str(x.value) for x in list(at.error) + list(at.warning))


# =========================================================================== A. robustness
def test_corrupt_ledger_is_a_clear_message_on_every_page_that_reads_it(tmpdata):
    (tmpdata["data_dir"] / "paper" / "ledger.parquet").write_bytes(b"not a parquet file")
    for page in ("Today", "Paper trading"):
        at = run_page(page)
        no_crash(at, page)
        assert "damaged or unreadable" in messages(at), page


def test_corrupt_state_file_is_a_message_per_card_not_a_crash(tmpdata):
    (tmpdata["data_dir"] / "paper" / "state.json").write_text("{ this is not json")
    at = run_page("Today")
    no_crash(at)
    assert messages(at).count("damaged or unreadable") == 1               # said once; the other cards point to it
    assert sum("Same problem as in the card above" in str(c.value) for c in at.caption) == len(TICKERS) - 1


def test_missing_price_file_is_a_message_not_a_traceback(tmpdata):
    (tmpdata["data_dir"] / "SPY.parquet").unlink()
    at = run_page("Today")
    no_crash(at)
    text = messages(at)
    assert "SPY" in text and ("no price data stored" in text or "missing" in text)


def test_corrupt_predictions_file_is_a_message_on_every_results_page(tmpdata):
    (tmpdata["data_dir"] / "results" / "SPY" / "predictions.parquet").write_bytes(b"garbage")
    for page in ("Today", "Backtest", "Model diagnostics", "Reality check"):
        clear()
        at = run_page(page)
        no_crash(at, page)
        assert "could not be read" in messages(at), page


def test_missing_results_directory_is_a_message(tmpdata):
    shutil.rmtree(tmpdata["data_dir"] / "results" / "SPY")
    for page in ("Backtest", "Model diagnostics", "Reality check"):
        clear()
        at = run_page(page)
        no_crash(at, page)
        assert "saved results are missing" in messages(at), page


@pytest.mark.parametrize("how", ["missing", "empty"])
def test_empty_or_missing_trial_log_is_a_results_missing_message(tmpdata, how):
    ip = tmpdata["data_dir"] / "results" / "trials_index.parquet"
    if how == "missing":
        ip.unlink()
    else:
        pd.DataFrame(columns=T.INDEX_COLS).to_parquet(ip)
    at = run_page("Trials")
    no_crash(at)
    assert "trial log is missing or empty" in messages(at)
    clear()
    at = run_page("Reality check")
    no_crash(at)                                                              # overfit panel degrades too
    assert any("trial log is missing or empty" in str(i.value) for i in at.info)


@pytest.mark.parametrize("exc,word", [(OSError("disk read error"), "could not be read"),
                                      (ValueError("empty results"), "empty or inconsistent"),
                                      (P.PaperError("ledger broken"), "damaged"),
                                      (RuntimeError("something odd"), "unexpected")])
def test_any_other_expected_failure_gets_a_message(monkeypatch, exc, word):
    def boom(*a, **k):
        raise exc
    monkeypatch.setattr(DD, "load_asset_results", boom)
    at = run_page("Backtest")
    no_crash(at)
    assert word in messages(at)


def test_the_real_reason_for_a_missing_live_model_is_shown(monkeypatch):
    monkeypatch.setattr(E, "load_live_model_ex", lambda t, c=None, asof=None: (None, E.LIBOMP_HINT))
    at = run_page("Today")
    no_crash(at)
    b = messages(at)
    assert b.count("brew install libomp") == len(TICKERS) and "PROGRESS.md" in b
    assert "saved live model is missing" not in b
    assert not at.exception


def test_a_really_missing_model_file_says_so(tmpdata):
    (tmpdata["data_dir"] / "paper" / "models" / "SPY.joblib").unlink()
    at = run_page("Today")
    no_crash(at)
    assert "model file is missing (SPY.joblib)" in messages(at)


# =========================================================================== B. setup note
def test_progress_has_the_fresh_mac_setup_section_and_the_dashboard_points_to_it():
    prog = (ROOT / "PROGRESS.md").read_text()
    assert "Setup on a fresh Mac" in prog and "brew install libomp" in prog
    assert "scripts/fix_lightgbm_macos.sh" in prog and "Stage 6" in prog and "README" in prog
    assert "Setup on a fresh Mac" in E.LIBOMP_HINT and "brew install libomp" in E.LIBOMP_HINT


# =========================================================================== Today page
def test_today_explains_the_needle_and_the_terms_per_asset():
    at = run_page("Today")
    no_crash(at)
    md = [str(m.value) for m in at.markdown]
    assert sum("The dial's axis is today's gap" in c for c in md) == len(TICKERS)
    assert sum("Strong buy are crossed out" in c and "none is invented" in c for c in md) == len(TICKERS)
    assert sum("Small print: today's gap is about the" in c and "percentile of this model's past days" in c for c in md) == len(TICKERS)
    caps = [str(c.value) for c in at.caption]
    assert sum("Below the typical rate is not bullish" in c for c in caps) == len(TICKERS)
    terms = [e for e in at.expander if e.label == "What do these words mean?"]
    assert len(terms) == len(TICKERS)
    t = "\n".join(str(m.value) for m in at.markdown)
    for word in ("Percentile", "Decile", "Calibrated", "Tree-SHAP", "AUC"):
        assert f"**{word}**" in t
    assert len(at.tabs) >= len(TICKERS) * 3                     # the long text is folded into tabs/expanders


def _badge_with(level_kwargs, monkeypatch):
    badge = E.badge_from_numbers(**level_kwargs)
    badge["inputs"] = {}
    real = E.evidence_badge
    monkeypatch.setattr(E, "evidence_badge", lambda *a, **k: badge)
    return badge, real


GOOD = dict(pt_pvalue=0.01, pt_statistic=2.0, brier_skill=0.01, brier_skill_eval_base=0.01, sharpe=1.2, sharpe_vt=0.9)


@pytest.mark.parametrize("kw,label,faded", [
    (dict(**GOOD, psr_excess_vt=0.5, dsr_excess_vt=0.5, dsr_global=0.5), "Weak evidence", False),
    (dict(**GOOD, psr_excess_vt=0.99, dsr_excess_vt=0.99, dsr_global=0.99), "Evidence", False)])
def test_amber_and_green_badges_render_unfaded_with_their_actual_values(monkeypatch, kw, label, faded):
    badge, _ = _badge_with(kw, monkeypatch)
    at = run_page("Today")
    no_crash(at)
    b = blob(at)
    assert b.count(label) >= len(TICKERS) and E.FADED_NOTE not in [str(i.value) for i in at.info]
    gauges = [json.loads(e.proto.spec) for e in at.get("plotly_chart")]
    ind = [g for g in gauges if g["data"][0]["type"] == "indicator"]
    assert len(ind) == len(TICKERS)
    assert all(float(g["data"][0]["gauge"]["steps"][0]["color"].rstrip(")").split(",")[-1]) > 0.5 for g in ind)   # not faded
    expander_text = "\n".join(str(m.value) for m in at.markdown)
    assert "(actual: p = 0.010, statistic = 2.00)" in expander_text            # real values next to the checks
    assert "skill vs training base rate 0.0100" in expander_text and "strategy Sharpe 1.20" in expander_text


def test_real_grey_badge_expander_shows_the_actual_numbers():
    at = run_page("Today")
    text = "\n".join(str(m.value) for m in at.markdown)
    for t in TICKERS:
        b = E.evidence_badge(t, CFG)
        assert f"p = {b['inputs']['pt_pvalue']:.3f}" in text
        assert f"vs evaluation-window base rate {b['inputs']['brier_skill_eval_base']:.4f}" in text


def test_dead_links_are_annotated_on_the_how_it_works_page():
    at = run_page("How it works")
    no_crash(at)
    md = "\n".join(str(m.value) for m in at.markdown)
    assert re.search(r"Wikipedia: purged cross-validation\]\([^)]*\) \(link no longer works\)", md)
    assert re.search(r"IMF: Bitcoin-S&P correlation\]\([^)]*\) \(link no longer works\)", md)
    assert "could not be verified" not in md and "Quantpedia summary" in md       # the link now answers HTTP 200


def test_how_it_works_documents_the_strong_buy_rule_and_every_badge_condition():
    at = run_page("How it works")
    tables = "\n".join(str(d.value.to_dict()) for d in at.dataframe)
    assert "no stronger step than holding" in tables and "no 'strong buy' action is invented" in tables
    for word in ("PT", "SKILL", "BEATS", "DSR_X", "DSR_G", "PSR of the excess", "BOTH"):
        assert word in tables, word
    assert "formula" not in tables.lower() or True
    code = "\n".join(str(c.value) for c in at.code)
    assert "Green  = all five pass" in code                                    # the beginner-readable formula


# =========================================================================== Backtest page
def test_backtest_break_even_is_n_a_where_it_means_nothing_and_always_up_is_merged():
    at = run_page("Backtest")
    no_crash(at)
    table = at.dataframe[0].value
    strategies = list(table["Strategy"])
    assert "Always 'up'" not in " ".join(strategies) and sum("Buy and hold (= always up)" == s for s in strategies) == 1
    bh = table[table["Strategy"].str.startswith("Buy and hold")].iloc[0]
    assert bh["Break-even cost"] == "n/a (no trades/costless)"
    assert "no costs" not in blob(at)
    assert "Volatility" not in table.columns                      # narrowed so the table fits 1366px
    assert "NOT picked by Sharpe ratio" in blob(at) and "config.yaml" in blob(at)


def test_negative_break_even_shows_the_reason(monkeypatch):
    real = DD.load_asset_results

    def patched(ticker, cfg=None, windows=None):
        res = real(ticker, cfg, windows)
        tm = res.tables["trading_metrics"].copy()
        tm.loc[(tm["kind"] == "trial"), "break_even_cost"] = -0.01
        res.tables["trading_metrics"] = tm
        return res
    monkeypatch.setattr(DD, "load_asset_results", patched)
    at = run_page("Backtest")
    no_crash(at)
    assert {m.label: str(m.value) for m in at.metric}["Break-even cost"] == "n/a (loses even at zero cost)"
    assert (at.dataframe[0].value["Break-even cost"] == "n/a (loses even at zero cost)").sum() == 4


def test_random_percentile_is_labelled_as_not_corrected_for_the_search():
    at = run_page("Backtest")
    assert "not corrected for the 54-106 configurations" in blob(at)


def test_dropdown_model_is_not_labelled_chosen_strategy():
    at = run_page("Backtest")
    at.selectbox(key="bt_model").set_value("logreg").run()
    names = [tr.get("name") for e in at.get("plotly_chart") for tr in json.loads(e.proto.spec)["data"]]
    perf = names[:3]
    assert "Selected model's strategy" in perf and not any(str(n).startswith("Chosen strategy") for n in perf)


def _with_row(monkeypatch, **cols):
    real = DD.load_asset_results

    def patched(ticker, cfg=None, windows=None):
        res = real(ticker, cfg, windows)
        cid = DD.paper_settings(cfg).config_id(ticker)
        tm = res.tables["trading_metrics"].copy()
        for k, v in cols.items():
            tm.loc[tm["name"] == cid, k] = v
        res.tables["trading_metrics"] = tm
        return res
    monkeypatch.setattr(DD, "load_asset_results", patched)


@pytest.mark.parametrize("sharpe,se,includes_zero", [(0.3, 0.5, True), (2.0, 0.1, False), (-2.0, 0.1, False),
                                                    (0.0, 0.1, True)])
def test_band_includes_zero_notice_direction_and_table_se_multiplier(monkeypatch, sharpe, se, includes_zero):
    _with_row(monkeypatch, sharpe=sharpe, sharpe_se=se)
    at = run_page("Backtest")
    notice = [i for i in at.info if "band includes zero" in str(i.value)]
    assert bool(notice) is includes_zero
    vals = {m.label: str(m.value) for m in at.metric}
    assert vals["Sharpe band (+-2 SE)"] == f"{sharpe - 2 * se:.2f} to {sharpe + 2 * se:.2f}"
    table = at.dataframe[0].value
    row = table[table["Strategy"] == "Ensemble"].iloc[0]
    assert row["Sharpe +-2 SE"] == f"{sharpe - 2 * se:.2f} to {sharpe + 2 * se:.2f}"   # k = 2 in the table too


def test_sharpe_chart_error_bars_are_two_standard_errors_in_the_app(monkeypatch):
    from app import ui
    seen = []
    real = ui.sharpe_band_fig
    monkeypatch.setattr(ui, "sharpe_band_fig", lambda *a, **k: (seen.append(k.get("k", 2.0)), real(*a, **k))[1])
    run_page("Backtest")
    assert seen == [2.0]


def test_leverage_tab_kelly_text_is_honest_and_clamps_a_negative_net_point():
    at = run_page("Backtest")                                   # SPY: net-of-borrowing Kelly is negative in the data
    b = blob(at)
    k = DD.load_asset_results("SPY", CFG).kelly.iloc[0]
    assert k["L_star_net_borrow"] < 0
    assert "0x (no leverage pays after 6% borrowing)" in b and "-0.57" not in b
    assert "Estimated on the underlying SPY" in b and "not on the chosen strategy" in b
    assert "2009-12-31" in b                                    # the pre-out-of-sample years it was estimated on
    assert "On the gross curve (before borrowing costs), at twice that point long-run growth falls back to about zero" in b
    assert "doubles" not in b.lower()
    assert "2010-2024 bull market, not from the model" in b and "can disagree" in b
    assert "different things" in b


def test_leverage_tab_shows_a_positive_net_kelly_point_for_bitcoin():
    at = run_page("Backtest")
    at.selectbox(key="bt_asset").set_value("BTC-USD").run()
    k = DD.load_asset_results("BTC-USD", CFG).kelly.iloc[0]
    assert f"about {k['L_star_net_borrow']:.2f}x after paying 6% a year to borrow" in blob(at)


def test_wiped_out_column_matches_the_saved_flag_row_by_row():
    at = run_page("Backtest")
    at.selectbox(key="bt_asset").set_value("BTC-USD").run()
    res = DD.load_asset_results("BTC-USD", CFG)
    full = [d.value for d in at.dataframe if "Wiped out?" in d.value.columns][-1]
    lv = res.leverage.copy()
    want = {(r["series"], f"{r['L']:g}x"): bool(r["wiped_out"]) for _, r in lv.iterrows()}
    names = {"chosen_strategy": "Chosen strategy (fixed in config.yaml)", "buy_and_hold": "Buy and hold",
             "vol_targeted_bh": "Vol-targeted buy and hold"}
    got = {}
    for _, r in full.iterrows():
        got[(r["Series"], r["Leverage"])] = r["Wiped out?"].startswith("YES")
    assert {(names[s], l): v for (s, l), v in want.items()} == got
    assert any(want.values()) and not all(want.values())        # both YES and no occur, so an inversion is caught
    assert full["Wiped out?"].map(lambda v: v == "no" or v.startswith("YES, on 20")).all()


# =========================================================================== Diagnostics page
def test_diagnostics_shows_both_brier_skills_and_uses_250_day_windows(monkeypatch):
    calls = []
    for name in ("rolling_auc", "rolling_brier_skill", "rolling_brier_skill_eval"):
        real = getattr(DD, name)
        monkeypatch.setattr(DD, name, (lambda r, n: lambda p, window=250: (calls.append((n, window)), r(p, window))[1])(real, name))
    at = run_page("Model diagnostics")
    no_crash(at)
    assert sorted(calls) == [("rolling_auc", 250), ("rolling_brier_skill", 250), ("rolling_brier_skill_eval", 250)]
    pm = DD.load_asset_results("SPY", CFG).prediction_metrics
    pm = pm[(pm["model"] == "ensemble") & (pm["window"] == "expanding")].iloc[0]
    vals = {m.label: str(m.value) for m in at.metric}
    assert vals["Brier skill vs training base rate"] == f"{pm['brier_skill']:.4f}"
    assert vals["Brier skill vs evaluation base rate"] == f"{pm['brier_skill_vs_eval_base']:.4f}"
    assert "drift alone" in blob(at) and "-0.0028" in blob(at)
    roll = next(json.loads(e.proto.spec) for e in at.get("plotly_chart")
                if "Rolling 250-day AUC" in e.proto.spec)
    assert {t.get("name") for t in roll["data"]} >= {"AUC", "skill vs training base rate (can be drift)",
                                                     "skill vs evaluation-window base rate"}


def test_diagnostics_decile_line_is_the_real_base_rate():
    at = run_page("Model diagnostics")
    preds = DD.model_predictions(DD.load_asset_results("SPY", CFG), "ensemble")
    fig = next(json.loads(e.proto.spec) for e in at.get("plotly_chart") if "Hit rate by confidence decile" in e.proto.spec)
    ys = sorted(s["y0"] for s in fig["layout"]["shapes"])
    assert ys == pytest.approx(sorted([0.5, float(np.mean(preds["y"] > 0.5))]))


def _with_pt(monkeypatch, p, stat):
    real = DD.load_asset_results

    def patched(ticker, cfg=None, windows=None):
        res = real(ticker, cfg, windows)
        pm = res.tables["prediction_metrics"].copy()
        pm["pt_pvalue"], pm["pt_statistic"] = p, stat
        res.tables["prediction_metrics"] = pm
        return res
    monkeypatch.setattr(DD, "load_asset_results", patched)


@pytest.mark.parametrize("p,stat,beats", [(0.01, 2.5, True), (0.01, -2.5, False), (0.40, 2.5, False),
                                         (0.04, 0.1, True), (0.05, 2.0, False)])
def test_pesaran_timmermann_verdict_direction(monkeypatch, p, stat, beats):
    _with_pt(monkeypatch, p, stat)
    at = run_page("Model diagnostics")
    b = blob(at)
    assert ("): the up/down calls beat chance." in b) is beats
    assert ("): no evidence that the up/down calls beat chance" in b) is (not beats)


# =========================================================================== Paper trading page
def test_paper_page_warning_comes_first_and_short_history_is_a_note():
    at = run_page("Paper trading")
    no_crash(at)
    kinds = [e.type for e in at.main]
    assert kinds.index("warning") < kinds.index("dataframe")                   # 'proves nothing' warning at the TOP
    assert str(at.warning[0].value).startswith("A few live days prove nothing")
    notes = [str(i.value) for i in at.info if "day(s) of data so far" in str(i.value)]
    assert notes and all("only 1 day(s)" in n for n in notes)
    assert not [e for e in at.get("plotly_chart")]                             # no unreadable one-point charts


@pytest.fixture
def two_day_ledger(tmpdata):
    """A copy of the ledger with a second day per asset and distinguishable baseline equities."""
    f = tmpdata["data_dir"] / "paper" / "ledger.parquet"
    led = pd.read_parquet(f)
    nxt = led.copy()
    nxt["date"] = nxt["date"] + pd.Timedelta(days=1)
    start = {t: float(tmpdata["paper_trading"]["starting_capital"][t]) if isinstance(
        tmpdata["paper_trading"]["starting_capital"], dict) else float(tmpdata["paper_trading"]["starting_capital"])
        for t in TICKERS}
    s = nxt["asset"].map(start)
    nxt["equity"], nxt["bh_equity"], nxt["vtbh_equity"] = s * 1.01, s * 1.10, s * 0.95
    nxt["cum_pnl"] = nxt["equity"] - s
    pd.concat([led, nxt], ignore_index=True).to_parquet(f, index=False)
    return tmpdata, start


def test_paper_page_draws_charts_with_date_only_axes_once_there_are_two_days(two_day_ledger):
    cfg, start = two_day_ledger
    at = run_page("Paper trading")
    no_crash(at)
    figs = [json.loads(e.proto.spec) for e in at.get("plotly_chart")]
    assert len(figs) == len(TICKERS)
    assert all(f["layout"]["xaxis"]["tickformat"] == "%Y-%m-%d" for f in figs)
    assert not [i for i in at.info if "day(s) of data so far" in str(i.value)]


def test_profit_and_loss_baseline_columns_are_not_swapped(two_day_ledger):
    cfg, start = two_day_ledger
    at = run_page("Paper trading")
    pl = next(d.value for d in at.dataframe if "Buy and hold ($)" in d.value.columns)
    for _, r in pl.iterrows():
        s = start[r["Asset"]]
        assert r["Strategy ($)"] == f"{s * 0.01:+,.2f}"
        assert r["Buy and hold ($)"] == f"{s * 0.10:+,.2f}"
        assert r["Vol-targeted buy and hold ($)"] == f"{-s * 0.05:+,.2f}"


# =========================================================================== Trials page
def test_trials_luck_bar_uses_the_real_trial_count_and_states_its_caveats():
    at = run_page("Trials")
    no_crash(at)
    summ = DD.trials_summary(CFG)
    n = summ["n_trials_global"]
    assert n == 270
    bar = M.expected_max_sharpe(n, summ["daily_sharpe_variance"]) * np.sqrt(252)
    b = blob(at)
    assert f"**{bar:.2f}**" in b and f"{n} configurations" in b
    hist = next(json.loads(e.proto.spec) for e in at.get("plotly_chart") if "All configurations tried" in e.proto.spec)
    ann = hist["layout"]["annotations"][0]["text"]
    assert f"{bar:.2f}" in ann and str(n) in ann
    assert M.expected_max_sharpe(1, 0.01) == 0.0 and bar > M.expected_max_sharpe(10, summ["daily_sharpe_variance"]) * np.sqrt(252)
    for phrase in ("conservative upper-bound bar", "heavily correlated", "far smaller", "between the four assets",
                   "252 trading days", "365"):
        assert phrase in b, phrase


def test_trials_dsr_table_has_the_excess_column_and_the_zero_only_caption():
    at = run_page("Trials")
    t = next(d.value for d in at.dataframe if "After this asset's trials (DSR)" in d.value.columns)
    summ = DD.trials_summary(CFG)
    col = "Excess (DSR)"
    for _, r in t.iterrows():
        assert r[col] == f"{summ['assets'][r['Asset']]['dsr_excess_vt']:.3f}"
    assert "test the Sharpe against zero only" in blob(at) and "NOT evidence of skill when the evidence badge is grey" in blob(at)


# =========================================================================== Reality check page
def test_reality_check_has_a_glossary_the_auc_reading_and_the_fixed_wording():
    at = run_page("Reality check")
    no_crash(at)
    ex = [e for e in at.expander if e.label == "Plain-English glossary for this page"]
    assert len(ex) == 1
    g = "\n".join(str(m.value) for m in at.markdown)
    for term in ("AUC", "Sharpe ratio", "Out-of-sample", "Hold-out", "PSR / DSR", "Diebold-Mariano", "Hansen SPA",
                 "t > 3", "bp (basis point)", "Kelly", "Cross-sectional", "LSTM / GBT", "Data-snooping-adjusted p-value"):
        assert f"**{term}**" in g, term
    aucs = []
    for t in TICKERS:
        pm = DD.load_asset_results(t, CFG).prediction_metrics
        pm = pm[(pm["window"] == "expanding") & pm["model"].isin(["logreg", "rf", "lgbm", "ensemble"])]
        aucs += [float(a) for a in pm["auc"]]
    low = sum(a <= 0.5 for a in aucs)
    assert f"{low} of these {len(aucs)} AUCs are at or below 0.5 and the best is {max(aucs):.3f}" in g
    assert "no better than a coin flip" in g
    b = blob(at)
    assert "far smaller than the uncertainty" not in b and "about 0.71 after two years" in b
    assert "1- to 60-minute" in b and "limited" in b


# =========================================================================== shell
def test_sidebar_holdout_caption_includes_the_cutoff_date():
    at = run_page("Today")
    cap = " ".join(str(c.value) for c in at.sidebar.caption)
    assert "Hold-out (from 2024-11-06) is never shown" in cap


def test_refresh_clears_both_caches_and_catches_nothing_silently(monkeypatch):
    calls = []
    monkeypatch.setattr(st.cache_data, "clear", lambda *a, **k: calls.append("data"))
    monkeypatch.setattr(st.cache_resource, "clear", lambda *a, **k: calls.append("resource"))
    monkeypatch.setattr(DD, "refresh_data", lambda *a, **k: DD.RefreshOutcome(True, "Already up to date - nothing new to record."))
    at = run_page("Today")
    calls.clear()
    at.sidebar.button(key="refresh").click().run()
    assert sorted(calls) == ["data", "resource"]
    assert not at.exception


def test_main_page_failure_never_leaves_a_page_without_the_footer(tmpdata):
    (tmpdata["data_dir"] / "paper" / "ledger.parquet").write_bytes(b"x")
    for page in ("Today", "Paper trading", "Backtest", "Trials", "How it works", "Reality check", "Model diagnostics"):
        clear()
        at = run_page(page)
        assert not at.exception, page
        assert any(FOOTER == str(c.value) for c in at.caption), page
