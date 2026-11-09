"""Stage 5 dashboard tests: the real Streamlit app run headlessly with ``streamlit.testing.v1.AppTest``
on the REAL saved results and paper ledger (nothing is faked except where a test says so, and nothing
is skipped).  The app is read-only; a test checksums data/paper and data/results around all the page runs.
"""
import hashlib
import json
import re
from pathlib import Path

import pandas as pd
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from quant import dash_data as DD
from quant import explain as E
from quant import paper as P

ROOT = Path(__file__).resolve().parents[1]
APP = str(ROOT / "app" / "dashboard.py")
PAGES = ["Today", "Backtest", "Model diagnostics", "Paper trading", "Trials", "How it works", "Reality check"]
FOOTER = "Research and education only, not financial advice."
LAST_DEV_DAY = pd.Timestamp("2024-11-05")        # last development day: the cutoff 2024-11-06 minus one
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}")
CFG = DD._cfg()
TICKERS = DD.tickers(CFG)


def tree_hash(*dirs) -> str:
    h = hashlib.sha256()
    for d in dirs:
        for p in sorted(Path(d).rglob("*")):
            if p.is_file():
                h.update(str(p).encode())
                h.update(p.read_bytes())
    return h.hexdigest()


def run_page(page: str, **state) -> AppTest:
    at = AppTest.from_file(APP, default_timeout=300)
    at.session_state["page"] = page
    for k, v in state.items():
        at.session_state[k] = v
    return at.run()


@pytest.fixture(scope="module")
def pages():
    """Every page run once on the real data, with a before/after checksum of everything the app could write."""
    paths = (ROOT / CFG["data_dir"] / "paper", ROOT / CFG["data_dir"] / "results")
    before = tree_hash(*paths)
    out = {p: run_page(p) for p in PAGES}
    out["_hash_before"], out["_hash_after"] = before, tree_hash(*paths)
    return out


def texts(at: AppTest) -> list:
    """All visible text of a page: markdown, captions, notices, metrics, headings, code."""
    out = []
    for kind in ("markdown", "caption", "info", "warning", "success", "error", "title", "header", "subheader", "code"):
        out += [str(e.value) for e in at.get(kind)]
    for m in at.metric:
        out += [str(m.label), str(m.value)]
    return out


def blob(at: AppTest) -> str:
    return "\n".join(texts(at))


def figures(at: AppTest) -> list:
    return [json.loads(e.proto.spec) for e in at.get("plotly_chart")]


def figure_dates(fig: dict) -> list:
    """Every date in the x or y values of a Plotly figure's traces (ISO strings)."""
    out = []
    for tr in fig["data"]:
        for ax in ("x", "y"):
            v = tr.get(ax)
            if isinstance(v, list):
                out += [pd.Timestamp(s) for s in v if isinstance(s, str) and DATE_RE.match(s)]
    return out


def ledger_expected(ticker):
    led = DD.ledger_for(ticker, CFG)
    return led.iloc[-1]


# --------------------------------------------------------------------------- the app runs
@pytest.mark.parametrize("page", PAGES)
def test_page_renders_without_exception(pages, page):
    at = pages[page]
    assert not at.exception, [e.value for e in at.exception]
    assert not at.error, [e.value for e in at.error]
    assert at.sidebar.radio[0].value == page


@pytest.mark.parametrize("page", PAGES)
def test_footer_disclaimer_on_every_page(pages, page):
    assert any(FOOTER == str(c.value).strip() for c in pages[page].caption)


def test_app_is_read_only(pages):
    assert pages["_hash_before"] == pages["_hash_after"]


def test_refresh_button_exists_on_every_page(pages):
    for p in PAGES:
        assert [b.label for b in pages[p].sidebar.button] == ["Refresh data"]


def test_theme_is_dark_and_wide_layout():
    cfg = (ROOT / ".streamlit" / "config.toml").read_text()
    assert 'base = "dark"' in cfg
    assert 'layout="wide"' in (ROOT / "app" / "dashboard.py").read_text()


# --------------------------------------------------------------------------- Today
def test_today_has_a_gauge_per_asset(pages):
    figs = figures(pages["Today"])
    gauges = [f for f in figs if f["data"][0]["type"] == "indicator"]
    assert len(gauges) == len(TICKERS) == 4
    for g in gauges:
        ind = g["data"][0]
        assert ind["gauge"]["shape"] == "angular"                # a semicircle
        lo, hi = ind["gauge"]["axis"]["range"]                    # the gap axis in percentage points
        assert lo < 0 < hi and lo <= ind["value"] <= hi
        assert ind["gauge"]["threshold"]["value"] == ind["value"]  # the needle


def test_gauge_has_five_zone_colours_and_is_faded_when_grey(pages):
    rep = E.today_report(TICKERS[0], CFG)
    from app import ui
    f_normal = ui.gauge_figure(rep["meter"], faded=False)
    f_faded = ui.gauge_figure(rep["meter"], faded=True)
    assert len(E.ZONES) == 5 and len(ui.ZONE_COLORS) == 5
    a = lambda f: float(f.data[0].gauge.steps[0].color.rstrip(")").split(",")[-1])    # noqa: E731
    assert a(f_faded) < a(f_normal)
    # the legend names all five zones
    legend = ui.zone_legend_html(rep["meter"].bounds)
    assert all(z in legend for z in E.ZONES)


def test_disclaimer_always_shown_with_each_meter(pages):
    at = pages["Today"]
    n_meters = len([f for f in figures(at) if f["data"][0]["type"] == "indicator"])
    n_disc = sum(1 for w in at.warning if str(w.value) == E.DISCLAIMER)
    assert n_disc >= n_meters == 4


def test_probability_is_shown_next_to_base_rate(pages):
    at = pages["Today"]
    hits = [m.value for m in at.markdown if re.search(r"\*\*\d+\.\d% vs \d+\.\d% typical\*\*", str(m.value))]
    assert len(hits) == len(TICKERS)
    for t, h in zip(TICKERS, hits):
        r = ledger_expected(t)
        assert f"{r['p_up'] * 100:.1f}% vs {r['base_rate'] * 100:.1f}% typical" in h


def test_meter_zone_equals_ledger_position(pages):
    at = pages["Today"]
    lines = [str(m.value) for m in at.markdown if "Meter reads:" in str(m.value)]
    assert len(lines) == len(TICKERS)
    for t in TICKERS:
        line = next(x for x in lines if x.startswith(f"{t} ·"))
        zone = re.search(r"Meter reads: \*\*(.+?)\*\*", line).group(1)
        pos = float(re.search(r"paper position: \*\*(\d+)% invested", line).group(1))
        r = ledger_expected(t)
        assert pos == round(r["position"] * 100)
        if r["position"] == 0:
            assert zone == "Sell / step aside"
        else:
            assert zone == "Hold / buy"                       # long/flat has no stronger step than holding
        # and it is what explain.py (which uses paper.decide_position) says
        assert zone == E.today_meter(t, CFG).zone


def test_gauge_drawing_names_the_same_zone_as_the_text(pages):
    at = pages["Today"]
    gauges = [f for f in figures(at) if f["data"][0]["type"] == "indicator"]
    for t, g in zip(TICKERS, gauges):
        zone = E.today_meter(t, CFG).zone
        assert any(zone in a["text"] for a in g["layout"]["annotations"])


def test_evidence_badge_shown_per_asset_with_grey_note(pages):
    at = pages["Today"]
    b = blob(at)
    for t in TICKERS:
        badge = E.evidence_badge(t, CFG)
        assert badge["label"] in b
    greys = sum(1 for t in TICKERS if E.evidence_badge(t, CFG)["level"] == "none")
    notes = [i for i in at.info if str(i.value) == E.FADED_NOTE]      # the exact wording
    assert len(notes) == greys
    assert E.FADED_NOTE == "This model hasn't shown a reliable edge — treat this as a learning tool."
    assert greys >= 1, "the real data has grey badges (this test must see the faded note)"
    for t in TICKERS:                         # faded gauge exactly when grey
        assert E.evidence_badge(t, CFG)["faded"] == (E.evidence_badge(t, CFG)["level"] == "none")


def test_models_agree_count_shown(pages):
    at = pages["Today"]
    lines = [str(m.value) for m in at.markdown if str(m.value).startswith("**Models agree:**")]
    assert len(lines) == len(TICKERS)
    for t, ln in zip(TICKERS, lines):
        ag = E.today_report(t, CFG)["agree"]
        assert f"{ag['n_agree']} of {ag['n_models']}" in ln


def test_why_has_four_parts_for_every_asset(pages):
    at = pages["Today"]
    ms = [str(m.value) for m in at.markdown]
    for part in ("1. What the model saw", "2. The theory behind it", "3. How much to trust it", "4. The conclusion"):
        assert sum(1 for m in ms if m == f"**{part}**") == len(TICKERS), part
    assert sum(1 for m in ms if m == "#### Why?") == len(TICKERS)


def test_learn_more_expanders_and_working_theory_links(pages):
    at = pages["Today"]
    labels = [e.label for e in at.expander]
    assert sum(1 for l in labels if l.startswith("Learn more:")) >= len(TICKERS)
    theory = E.load_theory()
    allowed = {s["url"] for g in theory["feature_groups"].values() for s in g["sources"]}
    urls = set()
    for m in at.markdown:
        urls |= set(re.findall(r"\]\((https://[^)\s]+)\)", str(m.value)))
    assert urls, "no source links shown"
    assert urls <= allowed                    # every link shown comes from theory.yaml
    report = (ROOT / "RESEARCH_REPORT.md").read_text()
    assert all(u in report for u in urls)     # ... and from the research report
    assert any(str(m.value).startswith("- ") and "([source](https://" in str(m.value) for m in at.markdown)


def test_position_size_update_time_shown(pages):
    b = blob(pages["Today"])
    for t in TICKERS:
        r = ledger_expected(t)
        assert f"Decision for {r['date'].date()}" in b
    assert "Paper ledger last written" in b


def test_stale_data_warning_is_shown(monkeypatch):
    st.cache_data.clear()
    fake = {t: dict(ticker=t, stale=True, warning=f"{DD.STALE_NOTE}: prices end 2020-01-01 but the latest completed day is 2026-11-06")
            for t in TICKERS}
    monkeypatch.setattr(DD, "data_freshness", lambda *a, **k: fake)
    try:
        at = run_page("Today")
        assert not at.exception
        stale = [w for w in at.warning if DD.STALE_NOTE in str(w.value)]
        assert len(stale) == len(TICKERS)
    finally:
        st.cache_data.clear()


def test_missing_ledger_row_is_a_message_not_a_crash(monkeypatch):
    st.cache_data.clear()
    def boom(*a, **k):
        raise E.NoDecisionError("ledger is empty")
    monkeypatch.setattr(E, "today_report", boom)
    try:
        at = run_page("Today")
        assert not at.exception
        assert any("no paper-trading decision yet" in str(i.value) for i in at.info)
        assert any(FOOTER == str(c.value) for c in at.caption)
    finally:
        st.cache_data.clear()


# --------------------------------------------------------------------------- Backtest
def test_backtest_has_curves_baselines_table_and_selectors(pages):
    at = pages["Backtest"]
    assert [s.label for s in at.selectbox][:3] == ["Asset", "Model", "Strategy mode"]
    names = [tr.get("name") for f in figures(at) for tr in f["data"]]
    assert "Buy and hold" in names and "Vol-targeted buy and hold" in names
    assert any(str(n).startswith("Chosen strategy") for n in names)      # round 1: label now says it is config-fixed
    assert any("Drawdown" in str(f["layout"].get("title", {}).get("text", "")) for f in figures(at))
    table = at.dataframe[0].value
    rows = " | ".join(table["Strategy"])
    for expect in ("Logistic regression", "Random forest", "LightGBM", "Ensemble", "Buy and hold",
                   "Vol-targeted buy and hold", "Trend rule"):
        assert expect in rows
    assert {"Sharpe", "Sharpe +-2 SE", "Break-even cost"} <= set(table.columns)
    labels = [m.label for m in at.metric]
    assert "Sharpe band (+-2 SE)" in labels and "Break-even cost" in labels and "Random-strategy percentile" in labels


def test_backtest_numbers_trace_to_saved_files(pages):
    at = pages["Backtest"]
    res = DD.load_asset_results("SPY", CFG)
    cid = DD.strategy_config_id("SPY", CFG)
    row = DD.trading_row(res, cid)
    band = DD.sharpe_band(row)
    vals = {m.label: str(m.value) for m in at.metric}
    assert vals["Sharpe (per year)"] == f"{band['sharpe']:.2f}"
    assert vals["Sharpe band (+-2 SE)"] == f"{band['low']:.2f} to {band['high']:.2f}"
    pr = DD.random_percentile(res, cid, CFG)
    assert vals["Random-strategy percentile"] == f"{pr:.0f}"


def test_backtest_selectors_change_the_result_without_crashing():
    at = run_page("Backtest")
    at.selectbox(key="bt_asset").set_value("BTC-USD").run()
    assert not at.exception and not at.error
    at.selectbox(key="bt_model").set_value("logreg").run()
    at.selectbox(key="bt_mode").set_value("default_long").run()
    at.selectbox(key="bt_window").set_value("rolling4y").run()
    assert not at.exception and not at.error, [e.value for e in at.exception]
    assert max(d for f in figures(at) for d in figure_dates(f)) <= LAST_DEV_DAY


def test_leverage_tab_selector_curves_and_notes(pages):
    at = pages["Backtest"]
    radio = at.radio(key="lev_L")
    assert radio.options == ["1x", "1.5x", "2x", "3x"]
    b = blob(at)
    assert "Leverage is never used to pick a strategy" in b
    assert "small edge plus leverage usually ends badly" in b
    assert "Kelly point" in b
    lev = [f for f in figures(at) if "leveraged equity" in str(f["layout"]["title"]["text"])]
    assert len(lev) == 1 and lev[0]["layout"]["yaxis"]["type"] == "log"
    growth = [f for f in figures(at) if "More leverage" in str(f["layout"]["title"]["text"])]
    assert len(growth) == 1
    assert any(tr.get("name") == "Kelly leverage" for tr in growth[0]["data"])      # the Kelly point is marked
    tabs = [t.label for t in at.tabs]
    assert "Leverage (what-if)" in tabs


def test_leverage_selector_updates_chart_and_wipeout_table():
    at = run_page("Backtest")
    at.selectbox(key="bt_asset").set_value("BTC-USD").run()
    for L in (1.0, 1.5, 2.0, 3.0):
        at.radio(key="lev_L").set_value(L).run()
        assert not at.exception, [e.value for e in at.exception]
        lev = next(f for f in figures(at) if "leveraged equity" in str(f["layout"]["title"]["text"]))
        assert f"{L:g}x leveraged equity" in lev["layout"]["title"]["text"]
        shapes = lev["layout"].get("shapes", [])
        assert any(s.get("fillcolor", "").startswith("rgba(229,83,75") for s in shapes), "crash windows are shaded"
        sel = [d.value for d in at.dataframe if "Wiped out?" in d.value.columns][0]
        assert set(sel["Leverage"]) == {f"{L:g}x"}
    wiped = [d.value for d in at.dataframe if "Wiped out?" in d.value.columns]
    full = wiped[-1]
    assert full["Wiped out?"].str.startswith("YES").any()    # Bitcoin buy-and-hold at 1.5x+ was wiped out in 2022


# --------------------------------------------------------------------------- Model diagnostics
def test_diagnostics_elements(pages):
    at = pages["Model diagnostics"]
    titles = [str(f["layout"].get("title", {}).get("text", "")) for f in figures(at)]
    for key in ("Reliability", "Hit rate by confidence decile", "Which signal groups", "Confusion matrix"):
        assert any(key in t for t in titles), key
    roll = next(f for f in figures(at) if f["layout"].get("annotations") and any("Rolling 250-day AUC" in a["text"] for a in f["layout"]["annotations"]))
    assert roll
    b = blob(at)
    assert "Pesaran-Timmermann" in b and "6.5 percentage" in b
    labels = [m.label for m in at.metric]
    assert "Pesaran-Timmermann p-value" in labels and "AUC (0.5 = coin flip)" in labels
    rel = next(f for f in figures(at) if "Reliability" in str(f["layout"]["title"]["text"]))
    assert any(tr.get("name") == "days in bin" for tr in rel["data"])             # bin counts are drawn


def test_diagnostics_numbers_trace_to_saved_files(pages):
    at = pages["Model diagnostics"]
    pm = DD.load_asset_results("SPY", CFG).prediction_metrics
    pm = pm[(pm["model"] == "ensemble") & (pm["window"] == "expanding")].iloc[0]
    vals = {m.label: str(m.value) for m in at.metric}
    assert vals["Pesaran-Timmermann p-value"] == f"{pm['pt_pvalue']:.3f}"
    assert vals["AUC (0.5 = coin flip)"] == f"{pm['auc']:.3f}"


def test_diagnostics_all_assets_models_windows_render():
    at = run_page("Model diagnostics")
    for t in TICKERS:
        at.selectbox(key="dg_asset").set_value(t).run()
        for m in ("logreg", "rf", "lgbm", "ensemble"):
            at.selectbox(key="dg_model").set_value(m).run()
            assert not at.exception and not at.error, (t, m)
    at.selectbox(key="dg_asset").set_value("BTC-USD").run()
    at.selectbox(key="dg_window").set_value("rolling4y").run()
    assert not at.exception and not at.error


# --------------------------------------------------------------------------- Paper trading
def test_paper_page_content(pages):
    at = pages["Paper trading"]
    b = blob(at)
    assert "A few live days prove nothing" in b
    for h in ("Open positions", "Equity curves", "Profit and loss against both baselines", "Trade log",
              "Live accuracy against the backtest"):
        assert h in b
    # round 1: the real ledger has ONE day per asset, so no unreadable one-point chart; a note says so
    one_day = [str(i.value) for i in at.info if "day(s) of data so far" in str(i.value)]
    counts = DD.load_ledger(CFG).groupby("asset").size()
    assert len(one_day) == int((counts < 2).sum())
    pos = at.dataframe[0].value
    assert len(pos) == len(TICKERS)
    led = DD.load_ledger(CFG)
    for f in figures(at):                       # live dates come only from the ledger
        assert set(figure_dates(f)) <= set(pd.DatetimeIndex(led["date"]))
    lb = [d.value for d in at.dataframe if "Live Brier score" in d.value.columns][0]
    assert {"Live Brier score", "Backtest Brier score"} <= set(lb.columns)


# --------------------------------------------------------------------------- Trials
def test_trials_page_counts_and_plain_language(pages):
    at = pages["Trials"]
    vals = {m.label: str(m.value) for m in at.metric}
    idx = DD.load_trial_index(CFG)
    n_alias = sum(len(json.loads(a)) for a in idx["aliases"] if a)
    assert vals["Distinct configurations tried"] == "270" == str(len(idx))
    assert vals["Including identical duplicates (aliases)"] == "360" == str(len(idx) + n_alias)
    b = blob(at)
    assert "the more you try, the higher a Sharpe ratio you get by luck alone" in b
    assert "Deflated Sharpe Ratio" in b
    t = next(d.value for d in at.dataframe if "After this asset's trials (DSR)" in d.value.columns)
    assert len(t) == 4
    summ = DD.trials_summary(CFG)
    spy = t[t["Asset"] == "SPY"].iloc[0]
    assert spy["After all 270 trials (DSR)"] == f"{summ['assets']['SPY']['dsr_global']:.3f}"


# --------------------------------------------------------------------------- How it works / Reality check
def test_how_it_works_covers_every_signal_and_model(pages):
    at = pages["How it works"]
    th = E.load_theory()
    labels = " | ".join(e.label for e in at.expander)
    for g in th["feature_groups"].values():
        assert g["title"] in labels
    for k in ("logistic_regression", "random_forest", "lightgbm", "ensemble", "walk_forward", "calibration"):
        assert th["models"][k]["title"] in labels
    code = [str(c.value) for c in at.code]
    assert any(th["feature_groups"]["reversal"]["formula"] in c for c in code)       # formulas shown
    b = blob(at)
    assert "How to read the meter and the evidence badge" in b
    assert all(z in "\n".join(str(d.value.to_dict()) for d in at.dataframe) for z in E.ZONES)
    assert "No proven edge" in "\n".join(str(d.value.to_dict()) for d in at.dataframe)
    links = [m for m in at.markdown if "](https://" in str(m.value)]
    assert links


def test_reality_check_has_every_fact(pages):
    at = pages["Reality check"]
    th = E.load_theory()["reality_check"]
    heads = [str(s.value) for s in at.subheader]
    for k in ("realistic_auc", "cross_sectional_vs_index", "edges_negative_after_costs", "post_publication_decay",
              "holdout_power", "overfitting", "significance_tests"):
        assert th[k]["title"] in heads
    b = blob(at)
    assert "0.51" in b and "58%" in b and "AUC" in b
    assert "after costs" in b.lower() or "after trading costs" in b.lower()
    assert "hold-out" in b.lower()
    assert "overfitting" in b.lower()
    assert any("Pesaran-Timmermann" in t for t in texts(at))
    own = next(d.value for d in at.dataframe if "Logistic regression AUC" in d.value.columns)
    assert len(own) == len(TICKERS)


# --------------------------------------------------------------------------- hold-out
def test_no_holdout_dates_in_any_plotted_dev_series(pages):
    """Every date in every figure of the development pages is <= 2024-11-05."""
    seen = 0
    for p in ("Backtest", "Model diagnostics", "Trials", "Reality check", "How it works"):
        for f in figures(pages[p]):
            ds = figure_dates(f)
            seen += len(ds)
            if ds:
                assert max(ds) <= LAST_DEV_DAY, (p, max(ds))
    assert seen > 10_000          # the equity / rolling / leverage series really were scanned


def test_saved_dev_series_end_before_the_cutoff():
    assert pd.Timestamp(CFG["holdout_cutoff_date"]) - LAST_DEV_DAY == pd.Timedelta(days=1)
    for t in TICKERS:
        res = DD.load_asset_results(t, CFG)
        for name, col in (("predictions", "date"), ("baselines", "date"), ("leverage_equity", "date")):
            assert pd.Timestamp(res.tables[name][col].max()) <= LAST_DEV_DAY, (t, name)
        cid = DD.strategy_config_id(t, CFG)
        assert DD.equity_curves(t, cid, CFG).index.max() <= LAST_DEV_DAY
        assert DD.rolling_auc(DD.model_predictions(res, "ensemble"), 250).index.max() <= LAST_DEV_DAY
        assert DD.leverage_equity_curves(res).index.max() <= LAST_DEV_DAY


def test_dashboard_refuses_holdout_dates_with_a_message(monkeypatch):
    st.cache_data.clear()
    st.cache_resource.clear()
    def bad(*a, **k):
        raise P.HoldoutError("series reached 2025-01-02")
    monkeypatch.setattr(DD, "load_asset_results", bad)
    try:
        at = run_page("Model diagnostics")
        assert not at.exception
        assert any("hold-out" in str(e.value) for e in at.error)
        assert any(FOOTER == str(c.value) for c in at.caption)
    finally:
        st.cache_data.clear()
        st.cache_resource.clear()


def test_missing_results_are_a_warning_not_a_crash(monkeypatch):
    st.cache_data.clear()
    st.cache_resource.clear()
    def missing(*a, **k):
        raise FileNotFoundError("SPY: no saved results")
    monkeypatch.setattr(DD, "load_asset_results", missing)
    try:
        for page in ("Backtest", "Reality check"):
            at = run_page(page)
            assert not at.exception, page
            assert any("saved results are missing" in str(w.value) for w in at.warning), page
    finally:
        st.cache_data.clear()
        st.cache_resource.clear()


# --------------------------------------------------------------------------- Refresh button
def test_refresh_handler_returns_a_message_and_never_touches_holdout():
    before = tree_hash(ROOT / CFG["data_dir"] / "paper", ROOT / CFG["data_dir"] / "results")
    calls = []

    class Res:
        wrote, new_rows, skipped, messages = False, None, {}, []

    def fake_run(cfg, out=None, **kw):
        calls.append(cfg["holdout_cutoff_date"])
        return Res()

    out = DD.refresh_data(CFG, run=fake_run)
    assert out.ok and out.message and "Traceback" not in out.message
    assert calls == [CFG["holdout_cutoff_date"]]        # the pinned cutoff is passed through unchanged
    boom = DD.refresh_data(CFG, run=lambda *a, **k: (_ for _ in ()).throw(RuntimeError("network down")))
    assert not boom.ok and "network down" in boom.message and "Traceback" not in boom.message
    assert before == tree_hash(ROOT / CFG["data_dir"] / "paper", ROOT / CFG["data_dir"] / "results")


def test_refresh_button_shows_the_message(monkeypatch):
    calls = []

    def fake(*a, **k):
        calls.append(1)
        return DD.RefreshOutcome(True, "Already up to date - nothing new to record.")
    monkeypatch.setattr(DD, "refresh_data", fake)
    at = run_page("Today")
    at.sidebar.button(key="refresh").click().run()
    assert calls == [1]
    assert not at.exception
    assert any("Already up to date" in str(s.value) for s in at.sidebar.success)


def test_refresh_failure_is_a_warning_without_traceback(monkeypatch):
    monkeypatch.setattr(DD, "refresh_data", lambda *a, **k: DD.RefreshOutcome(False, "Refresh failed: no internet"))
    at = run_page("Today")
    at.sidebar.button(key="refresh").click().run()
    assert not at.exception
    assert any("Refresh failed: no internet" in str(w.value) for w in at.sidebar.warning)
    assert "Traceback" not in blob(at)
