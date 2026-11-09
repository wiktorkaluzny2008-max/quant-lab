"""Quant Lab dashboard (Stage 5).  Run with:  uv run streamlit run app/dashboard.py

Seven pages: Today, Backtest, Model diagnostics, Paper trading, Trials, How it works, Reality check.
Every chart is drawn from results that Stages 3 and 4 already saved (data/results, data/paper); the
only thing that writes anything is the "Refresh data" button, which runs the Stage 4 daily update.
Formulas and text come from quant/dash_data.py, quant/explain.py and quant/theory.yaml -- this file
only lays them out.  The 24-month hold-out is never loaded (dash_data refuses hold-out dates).
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:               # `streamlit run app/dashboard.py` only puts app/ on the path
    sys.path.insert(0, str(ROOT))

import numpy as np                           # noqa: E402
import pandas as pd                          # noqa: E402
import pyarrow as pa                         # noqa: E402
import streamlit as st                       # noqa: E402

from app import ui                           # noqa: E402
from quant import dash_data as DD            # noqa: E402
from quant import explain as E               # noqa: E402
from quant import metrics as M               # noqa: E402
from quant import paper as P                 # noqa: E402
from quant import trials as T                # noqa: E402

PAGES = ["Today", "Backtest", "Model diagnostics", "Paper trading", "Trials", "How it works", "Reality check"]
FOOTER = "Research and education only, not financial advice."
MODEL_LABEL = {"ensemble": "Ensemble (average of the three)", "logreg": "Logistic regression",
               "rf": "Random forest", "lgbm": "LightGBM"}
MODE_LABEL = {"long_flat": "Long / flat (hold, or step aside)", "default_long": "Default long (step aside rarely)"}
BASE_LABEL = {"always_up": "Always 'up'", "buy_and_hold": "Buy and hold (= always up)",
              "trend_200": "Trend rule (200-day)", "vol_targeted_bh": "Vol-targeted buy and hold"}
LEVS = [1.0, 1.5, 2.0, 3.0]
SETUP_NOTE = ("On a fresh Mac, LightGBM needs the libomp library: run `brew install libomp` (or "
              "`sh scripts/fix_lightgbm_macos.sh`) and restart. See 'Setup on a fresh Mac' in PROGRESS.md.")
NO_BREAK_EVEN = "n/a (no trades/costless)"
LOSES_ANYWAY = "n/a (loses even at zero cost)"
BE_BIG = 0.01            # break-even costs of 1% or more are shown with one decimal ("11.1%", not "11.117%")
BE_SMALL_DIGITS = 3

st.set_page_config(page_title="Quant Lab", page_icon=":bar_chart:", layout="wide",
                   initial_sidebar_state="expanded")
st.markdown("""<style>
.block-container{padding-top:2.2rem;padding-bottom:1.5rem;max-width:1500px}
h1,h2,h3{letter-spacing:-0.01em}
[data-testid="stMetricValue"]{font-size:1.5rem}
</style>""", unsafe_allow_html=True)


# =========================================================================== cached loaders
def data_signature() -> tuple:
    """Cache key that changes after a daily run (see ``dash_data.data_signature``)."""
    return DD.data_signature()


@st.cache_data(show_spinner="Building today's report...", ttl=900)
def c_report(ticker: str, sig: tuple) -> dict:
    return E.today_report(ticker)


@st.cache_data(show_spinner=False, ttl=300)
def c_fresh(sig: tuple) -> dict:
    return DD.data_freshness()


@st.cache_resource(show_spinner=False, ttl=900)
def c_results(ticker: str, sig: tuple):
    return DD.load_asset_results(ticker)


@st.cache_data(show_spinner=False, ttl=900)
def c_ledger(sig: tuple) -> pd.DataFrame:
    return DD.load_ledger()


@st.cache_data(show_spinner=False, ttl=900)
def c_equity(ticker: str, config_id: str, window: str, sig: tuple) -> pd.DataFrame:
    return DD.equity_curves(ticker, config_id, window=window)


@st.cache_data(show_spinner=False, ttl=900)
def c_vt_percentile(ticker: str, window: str, sig: tuple):
    return DD.vt_baseline_percentile(DD.load_asset_results(ticker), "vol_targeted_bh", window)


@st.cache_data(show_spinner=False, ttl=900)
def c_trials(sig: tuple):
    idx = DD.load_trial_index()
    return DD.trials_summary(), idx


@st.cache_data(show_spinner=False, ttl=900)
def c_theory() -> dict:
    return E.load_theory()


def pct(x, digits=1, signed=False) -> str:
    if x is None or not np.isfinite(x):
        return "n/a"
    return f"{x * 100:+.{digits}f}%" if signed else f"{x * 100:.{digits}f}%"


def num(x, digits=2) -> str:
    return "n/a" if x is None or not np.isfinite(x) else f"{x:.{digits}f}"


def show(fig, key: str):
    st.plotly_chart(fig, width="stretch", key=key)


def src_links(sources) -> str:
    return "  \n".join(f"- [{s['title']}]({s['url']})" + (f" {s['link_note']}" if s.get("link_note") else "")
                      for s in (sources or []))


def theory_card(entry: dict, expanded=False, heading=None):
    """One theory entry: plain English, formula, evidence, caveats and working source links."""
    with st.expander(heading or entry["title"], expanded=expanded):
        st.markdown(" ".join(str(entry["plain_english"]).split()))
        if entry.get("formula"):
            st.markdown("**Formula**")
            st.code(str(entry["formula"]), language=None)
        if entry.get("evidence"):
            st.markdown("**What the research found:** " + " ".join(str(entry["evidence"]).split()))
        if entry.get("caveats"):
            st.markdown("**Caveats:** " + " ".join(str(entry["caveats"]).split()))
        if entry.get("source_note"):
            st.caption(" ".join(str(entry["source_note"]).split()))
        if entry.get("sources"):
            st.markdown("**Sources**  \n" + src_links(entry["sources"]))


def wide_headers(rows) -> dict:
    """Column config giving every column enough width for its header (st.dataframe clips long headers)."""
    cols = list(rows[0]) if rows else []
    return {c: st.column_config.Column(width="medium" if len(c) > 12 else "small") for c in cols}


def glossary(items: dict, label="Plain-English glossary"):
    with st.expander(label):
        for k, v in items.items():
            st.markdown(f"**{k}**: {v}")


# =========================================================================== failures -> plain messages
def is_config_error(e: Exception) -> bool:
    """True for the errors ``quant.config.load_config`` raises: file missing, not valid YAML, missing keys."""
    msg = str(e)
    return (isinstance(e, (FileNotFoundError, ValueError)) and
            (msg.startswith("Config file") or msg.startswith("Config missing") or msg.startswith("Config '")
             or msg.startswith("Asset #") or "config.yaml" in msg.lower()))


def explain_failure(e: Exception) -> tuple:
    """(level, message) for any expected failure: damaged or missing files, a library that cannot
    load, empty results.  Never a traceback."""
    if isinstance(e, P.HoldoutError):
        return "error", f"Blocked: a saved series reached into the hold-out period. {e}"
    if isinstance(e, P.PaperError):
        return "error", (f"The paper-trading files look damaged or unreadable: {e}. Nothing was changed. Restore "
                         "them from a backup, or move data/paper aside and run `uv run python -m quant.run_daily` "
                         "to rebuild the ledger from scratch.")
    if is_config_error(e):
        return "error", (f"config.yaml could not be used: {e}. The dashboard needs it to know which assets and "
                         "folders to show. Restore config.yaml from a backup (or fix the line it names) and reload "
                         "the page. Nothing was changed.")
    if "libomp" in str(e) or "lightgbm" in str(e).lower():       # before the generic OSError branch: this is how it arrives
        return "error", f"LightGBM could not load. {SETUP_NOTE}"
    if isinstance(e, FileNotFoundError):
        return "warning", (f"Some saved results are missing: {e}. Run `uv run python -m quant.run_daily` (prices and "
                           "paper ledger), `uv run python -m quant.backtest` and `uv run python -m quant.leverage` "
                           "(saved results) to create it.")
    if isinstance(e, (pa.ArrowException, OSError)):
        return "error", (f"A saved data file could not be read ({type(e).__name__}: {e}). It may be damaged or "
                         "half-written; re-create it with the matching `uv run python -m quant...` command.")
    if isinstance(e, LookupError):
        return "warning", f"Some expected data is missing: {e}"
    if isinstance(e, ValueError):
        return "warning", f"Some saved results look empty or inconsistent: {e}"
    return "error", f"Something unexpected went wrong ({type(e).__name__}: {e}). Nothing was changed."


def show_failure(e: Exception):
    level, msg = explain_failure(e)
    (st.error if level == "error" else st.warning)(msg)


TODAY_TERMS = {
    "Gap": "today's probability of an up day minus how often the asset usually rises. 0 = a perfectly typical day; "
           "the needle on the dial is this number, in percentage points.",
    "Percentile": "where today's gap ranks among this model's own past gaps: 90 means higher than about 90% of them. "
                  "Shown as small print; it does not decide the zone.",
    "Decile": "one tenth of the days, sorted from least to most confident.",
    "Calibrated": "the raw model score has been adjusted so that when it says 60%, about 60 in 100 such days really were up days.",
    "Tree-SHAP": "an exact way to split a tree model's answer into how much each signal pushed it up or down.",
    "AUC": "the chance the model gives a higher probability to a random up day than to a random down day; 0.5 is a coin flip.",
}


# =========================================================================== page 1: Today
def page_today():
    sig = data_signature()
    st.title("Today")
    st.markdown(
        "A **signal meter** for each asset. The needle is today's gap: the model's probability of an up day "
        "minus how often the asset usually rises. The coloured zones begin where the paper-trading account "
        "(fake money) would change what it does, so the needle always sits in the zone of the paper position. "
        "Most days the honest answer is *'about the same as usual'*.")
    fresh = c_fresh(sig)
    names = DD.asset_names()
    ledger_file = Path(DD._cfg()["data_dir"]) / "paper" / "ledger.parquet"
    written = (datetime.fromtimestamp(ledger_file.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
               if ledger_file.exists() else "never")
    st.caption(f"Paper ledger last written {written}. Use each card's other tabs for the reasons, the evidence "
               "badge details and plain-English terms.")
    tickers = DD.tickers()
    cols_pair = None
    seen: set = set()                  # failure messages already shown on this page (shown once, not per card)
    for i, t in enumerate(tickers):
        if i % 2 == 0:
            cols_pair = st.columns(2, gap="large")
        with cols_pair[i % 2]:
            asset_card(t, names.get(t, t), fresh.get(t), written, sig, seen)


def show_failure_once(e: Exception, seen: set):
    """Like ``show_failure`` but a message identical to one already shown on this page becomes a short pointer."""
    level, msg = explain_failure(e)
    if msg in seen:
        st.caption("Same problem as in the card above (see its message).")
        return
    seen.add(msg)
    (st.error if level == "error" else st.warning)(msg)


def asset_card(ticker: str, name: str, fresh: dict | None, written: str, sig: tuple, seen: set | None = None):
    seen = set() if seen is None else seen
    with st.container(border=True):
        st.subheader(f"{name} ({ticker})")
        if fresh and fresh.get("warning"):
            st.warning(f"{ticker}: {fresh['warning']}")
        try:
            rep = c_report(ticker, sig)
        except E.NoDecisionError as e:
            st.info(f"{ticker}: no paper-trading decision yet ({e}). Press 'Refresh data' in the sidebar.")
            return
        except Exception as e:      # noqa: BLE001 - any failure stays inside THIS card; the other cards still draw
            show_failure_once(e, seen)
            return
        m, badge = rep["meter"], rep["badge"]
        faded = bool(rep["faded"])
        tab_meter, tab_why, tab_terms = st.tabs(["Meter", "Why?", "Badge and terms"])
        with tab_meter:
            show(ui.gauge_figure(m, faded=faded), f"gauge_{ticker}")
            st.markdown(ui.zone_legend_html(m.bounds, faded), unsafe_allow_html=True)
            st.markdown(f"**{m.probability_text}**")
            st.markdown(f"{ticker} · Meter reads: **{m.zone}** · paper position: **{rep['position'] * 100:.0f}% invested**"
                        f" (fake money, long only)")
            st.markdown(ui.badge_html(badge), unsafe_allow_html=True)
            if faded:
                st.info(rep["faded_note"])
            ag = rep.get("agree")
            if not ag and rep.get("warning"):
                st.warning(rep["warning"])
            st.warning(E.DISCLAIMER, icon=":material/info:")
        with tab_why:
            if rep.get("why") is not None:
                why_block(rep["why"], ticker)
            else:
                st.info("'Why?' is unavailable for this asset: see the note on the Meter tab.")
        with tab_terms:
            st.markdown("**How to read the dial**")
            for line in E.meter_explanation(m):
                st.markdown(f"- {line}")
            st.caption(f"{ticker}: {m.probability_text} is the model's chance of an up day next to how often "
                       f"{ticker} rises on a typical day. Below the typical rate is not bullish.")
            if m.caption:
                st.caption(m.caption + " " + m.sell_means)
            if ag:
                st.markdown(f"**Models agree:** {ag['n_agree']} of {ag['n_models']} lean "
                            f"{ag['direction']} the usual rate.")
                st.caption(ag["text"] + " Each model is asked: is your probability at or above the usual rate?")
            st.caption(f"Decision for {rep['date']} (made after that day's close). Paper ledger last written {written}.")
            with st.expander("Why this evidence badge?"):
                st.markdown(f"**{badge['label']}.** " + "".join(f"\n- {r}" for r in badge["reasons"]))
                st.caption("A badge of 'No proven edge' does not mean the model is proven useless: a few years of "
                           "daily data have little statistical power. It means the backtest did not show a "
                           "reliable edge.")
            glossary(TODAY_TERMS, "What do these words mean?")


def why_block(why, ticker: str):
    st.markdown("#### Why?")
    st.markdown("**1. What the model saw**")
    for s in why.saw:
        st.markdown(f"- {s['sentence']}")
    st.markdown("**2. The theory behind it**")
    for t in why.theory:
        link = (f" ([source]({t['url']}){' ' + t['link_note'] if t.get('link_note') else ''})" if t["url"] else "")
        st.markdown(f"- {t['text']}{link}")
    for t in why.theory:
        lm = t["learn_more"]
        title = c_theory()["feature_groups"][t["group"]]["title"]
        theory_card(dict(title=title, plain_english=lm["plain_english"], formula=lm["formula"],
                         evidence=lm["evidence"], caveats=lm["caveats"], sources=lm["sources"]),
                    heading=f"Learn more: {title}")
    st.markdown("**3. How much to trust it**")
    tr = why.trust
    st.markdown(f"- {tr['text']}")
    st.markdown(f"- {tr['caveat']}")
    if tr.get("signal_caveat"):
        st.markdown(f"- Caveat for the lead signal: {tr['signal_caveat']}")
    if tr.get("badge_note"):
        st.markdown(f"- {tr['badge_note']}")
    st.markdown("**4. The conclusion**")
    st.markdown(f"- {why.conclusion}")
    st.caption(why.method_note)


# =========================================================================== page 2: Backtest
def page_backtest():
    sig = data_signature()
    st.title("Backtest")
    st.markdown(
        "How the strategy would have done **out of sample** (each day predicted by a model that had only seen the "
        "past, retrained monthly), after trading costs. Saved development results only: the final 24-month "
        "hold-out (from 2024-11-06) is untouched.")
    names = DD.asset_names()
    tickers = DD.tickers()
    paper = DD.paper_settings()
    c1, c2, c3, c4, c5 = st.columns([1.2, 1.4, 1.5, 1, 1])
    ticker = c1.selectbox("Asset", tickers, format_func=lambda t: f"{names[t]} ({t})", key="bt_asset")
    res = c_results(ticker, sig)
    windows = sorted(res.trading_metrics["window"].unique(), key=lambda w: (w != "expanding", w))
    model = c2.selectbox("Model", list(MODEL_LABEL), format_func=MODEL_LABEL.get, key="bt_model")
    mode = c3.selectbox("Strategy mode", list(MODE_LABEL), format_func=MODE_LABEL.get, key="bt_mode")
    vt = c4.selectbox("Size by volatility", [True, False], format_func=lambda v: "Yes (vol-targeted)" if v else "No (all-in)",
                      key="bt_vt")
    window = c5.selectbox("Training window", windows, key="bt_window",
                          help="'expanding' uses all past data; 'rolling4y' only the last 4 years (Bitcoin only).")
    tm = res.trading_metrics
    ms = sorted({str(x) for x in tm[(tm["kind"] == "trial") & (tm["mode"] == mode)
                                    & (tm["vol_targeted"] == vt)]["m_setting"]},
                key=lambda s: (s != "adaptive", s))
    m_set = st.selectbox("Step-aside margin (how far below the usual rate before stepping aside)", ms, key="bt_m",
                         help="'adaptive' picks the margin inside each training window only, never from the test period.")
    cid = T.make_config_id(ticker, window, model, mode, vt, T.ADAPTIVE if m_set == T.ADAPTIVE else float(m_set))
    row = DD.trading_row(res, cid, window)
    glossary({
        "Equity curve": "what 1 dollar invested at the start would have become, after trading costs.",
        "Drawdown": "how far a line is below its previous best level - the pain of the worst stretch.",
        "Sharpe ratio": "return per unit of risk, per year. Around 0.4-0.9 is typical for simply owning stocks.",
        "Standard error band": "the Sharpe you see is an estimate; plus or minus 2 standard errors is the range "
                               "that is plausible. If the band includes 0, you cannot tell it from luck.",
        "Break-even cost": "the trading cost per trade at which the strategy's profit would be wiped out. "
                           "Bigger is safer; the real cost used here is 0.05% (ETFs) / 0.15% (Bitcoin). Shown as n/a "
                           "when it means nothing (no trades, or the strategy loses even with zero costs).",
        "Random-strategy percentile": "we also built 1,000 random strategies that are invested the same share of "
                                      "the time. 50 = no better than random timing, 95 = better than 95% of them. "
                                      "It is computed on GROSS returns (before trading costs), and for volatility-"
                                      "targeted series it mostly reflects the vol targeting, not the model's timing: "
                                      "plain vol-targeted buy and hold ranks high too (shown next to it). It is also "
                                      "NOT corrected for the 54-106 configurations tried per asset: the best of many "
                                      "tries ranks high by luck.",
        "Vol-targeted buy and hold": "own the asset, but hold less when it is jumpy. The research says this simple "
                                     "sizing rule is the best-evidenced edge, so it is the hard benchmark."})
    fixed_cid = DD.strategy_config_id(ticker)
    is_fixed = cid == fixed_cid
    strat_label = ui.SERIES_LABEL["chosen_strategy"] if is_fixed else ui.SERIES_LABEL["strategy"]
    st.caption("The 'Chosen strategy' in this project is the one fixed in config.yaml (`leverage_strategy`) before "
               "results were examined; it is NOT picked by Sharpe ratio. Any other dropdown combination is shown as "
               f"'Selected model's strategy'. The fixed one for {ticker} is: {fixed_cid}.")
    tabs = st.tabs(["Performance", "Leverage (what-if)"])
    with tabs[0]:
        if row is None:
            st.warning("This combination was not part of the saved backtest (no saved result).")
        else:
            band = DD.sharpe_band(row)
            pr = DD.random_percentile(res, cid)
            vp = c_vt_percentile(ticker, window, sig)
            k1, k2, k3 = st.columns(3)
            k1.metric("Sharpe (per year)", num(band["sharpe"]), help="Return per unit of risk.")
            k2.metric("Sharpe band (+-2 SE)", f"{band['low']:.2f} to {band['high']:.2f}")
            k3.metric("Return per year (CAGR)", pct(row["cagr"]))
            k4, k5, k6 = st.columns(3)
            k4.metric("Break-even cost", be_text(row))
            k5.metric("Random-strategy percentile", "n/a" if pr is None else f"{pr:.0f}",
                      help="Gross returns; not corrected for the 54-106 configurations tried per asset.")
            k6.metric("Vol-targeted buy and hold percentile (same test)", "n/a" if vp is None else f"{vp:.0f}",
                      help="The same random-strategy test applied to plain vol-targeted buy and hold (no model).")
            if pr is not None:
                vp_txt = "" if vp is None else (f" For comparison, plain vol-targeted buy and hold (no model at all) "
                                                f"scores {vp:.1f} in the same test, against {pr:.1f} here.")
                st.caption("The random-strategy percentile is computed on GROSS returns (before trading costs). For "
                           "vol-targeted series it mostly reflects vol targeting, not the model's timing, because "
                           "shifting the positions also scrambles the volatility alignment." + vp_txt +
                           " It is also not corrected for the 54-106 configurations tried per asset (see the Trials "
                           "page): the best of many tries ranks high by luck alone.")
            if band["low"] <= 0 <= band["high"]:
                st.info("The Sharpe band includes zero: this result cannot be told apart from luck.")
        try:
            cur = c_equity(ticker, cid, window, sig)
            st.caption(f"Out of sample from {cur.index.min().date()} to {cur.index.max().date()}, after costs.")
            a, b = st.columns(2)
            with a:
                show(ui.equity_fig(cur, strategy_label=strat_label), "bt_equity")
            with b:
                show(ui.drawdown_fig(cur, strategy_label=strat_label), "bt_dd")
        except (KeyError, FileNotFoundError) as e:
            st.warning(f"Equity curve unavailable: {e}")
        table = metrics_display(res, window, mode, vt, m_set)
        st.markdown("##### All models and baselines (same mode, sizing and margin)")
        shown = table.drop(columns=["_kind", "_sharpe", "_se"])
        widths = {c: "small" for c in shown.columns}
        widths.update({"Strategy": "medium", "Break-even cost": "medium", "Sharpe +-2 SE": "medium"})
        st.dataframe(shown, hide_index=True, width="stretch",
                     column_config={c: st.column_config.TextColumn(width=w) if c != "Trades" else
                                    st.column_config.NumberColumn(width=w) for c, w in widths.items()})
        st.caption("The last three rows are baselines: they need no model at all. A model only matters if it beats them after costs. "
                   "'Always up' is the same as buy and hold (one purchase, held), so it is listed once.")
        sb = pd.DataFrame({"label": table["Strategy"], "sharpe": table["_sharpe"], "sharpe_se": table["_se"],
                           "kind": table["_kind"]})
        show(ui.sharpe_band_fig(sb), "bt_sharpe")
    with tabs[1]:
        leverage_tab(ticker, res, sig)


def be_text(r) -> str:
    """Break-even cost for display: n/a (with the reason) where the number would mean nothing."""
    v = r["break_even_cost"]
    if r.get("kind") == "baseline" and r.get("name") in ("buy_and_hold", "always_up"):
        return NO_BREAK_EVEN
    if v is None or not np.isfinite(v):
        return "n/a"
    return LOSES_ANYWAY if v < 0 else pct(v, BE_SMALL_DIGITS if v < BE_BIG else 1)


def metrics_display(res, window, mode, vt, m_set) -> pd.DataFrame:
    t = DD.metrics_table(res, window=window, mode=mode, vol_targeted=vt, m_setting=m_set)
    t = t[t["name"] != "always_up"]            # identical to buy and hold; listed once
    rows = []
    for _, r in t.iterrows():
        label = (MODEL_LABEL.get(r["model"], r["model"]).split(" (")[0] if r["kind"] == "trial"
                 else BASE_LABEL.get(r["name"], r["name"]))
        se = float(r["sharpe_se"]) if np.isfinite(r["sharpe_se"]) else float("nan")
        rows.append({"Strategy": label, "Sharpe": num(r["sharpe"]),
                     "Sharpe +-2 SE": f"{r['sharpe'] - 2 * se:.2f} to {r['sharpe'] + 2 * se:.2f}" if np.isfinite(se) else "n/a",
                     "Return/yr": pct(r["cagr"]),
                     "Worst drop": pct(r["max_drawdown"]), "Invested": pct(r["pct_invested"], 0),
                     "Trades": int(r["n_trades"]), "Break-even cost": be_text(r),
                     "_kind": r["kind"], "_sharpe": float(r["sharpe"]), "_se": se})
    return pd.DataFrame(rows)


def leverage_tab(ticker: str, res, sig: tuple):
    st.markdown(
        "**What-if only, with fake money.** Leverage means borrowing to hold more than you own. Here L = 2 means "
        "twice your money invested, rebalanced daily, paying a borrowing rate (6% a year) on the borrowed part. "
        "**Leverage is never used to pick a strategy** and paper trading stays unleveraged.")
    if res.get("leverage").empty:
        st.warning("No saved leverage results for this asset (run `uv run python -m quant.leverage`).")
        return
    st.caption("Series: the 'Chosen strategy' is the one fixed in config.yaml, shown with buy and hold and "
               "vol-targeted buy and hold, over the out-of-sample period.")
    L = st.radio("Leverage", LEVS, index=0, horizontal=True, format_func=lambda v: f"{v:g}x", key="lev_L")
    eq = {s: DD.leverage_equity_curves(res, s) for s in ("buy_and_hold", "vol_targeted_bh", "chosen_strategy")}
    windows = res.crashes.get("windows", [])
    show(ui.leverage_equity_fig(eq, L, windows), "lev_equity")
    out_of_range = [w["label"] for w in windows if not w.get("n_oos_days")]
    if out_of_range:
        st.caption("Crashes before the out-of-sample period are not simulated: " + ", ".join(out_of_range) + ".")
    parts = []
    for s in ("chosen_strategy", "buy_and_hold", "vol_targeted_bh"):
        t = DD.leverage_table(res, s)
        t.insert(0, "Series", ui.SERIES_LABEL[s])
        parts.append(t)
    lv = pd.concat(parts, ignore_index=True)
    disp = pd.DataFrame({
        "Series": lv["Series"], "Leverage": lv["L"].map(lambda v: f"{v:g}x"),
        "Return / year": lv["cagr"].map(pct), "Worst drawdown": lv["max_drawdown"].map(pct),
        "Worst day": lv["worst_day"].map(pct), "Worst month": lv["worst_month"].map(pct),
        "Days to recover": lv["longest_recovery_days"].map(lambda v: f"{int(v)}" if np.isfinite(v) else "n/a"),
        "Wiped out?": [("YES, on " + str(pd.Timestamp(d).date())) if w else "no"
                       for w, d in zip(lv["wiped_out"], lv["wipe_date"])]})
    st.markdown(f"##### Drawdowns and 'wiped out' (selected: {L:g}x highlighted below)")
    sel = disp[disp["Leverage"] == f"{L:g}x"]
    st.dataframe(sel, hide_index=True, width="stretch")
    with st.expander("Show every leverage level"):
        st.dataframe(disp, hide_index=True, width="stretch")
    st.caption("'Wiped out' = the account fell below 20% of its starting value; a broker would have force-sold "
               "you well before that.")
    k = res.get("kelly")
    if not k.empty:
        show(ui.growth_fig(k), "lev_growth")
        r = k.iloc[0]
        net = float(r["L_star_net_borrow"])
        net_txt = (f"about {net:.2f}x after paying {r['borrow_rate'] * 100:.0f}% a year to borrow" if net > 0 else
                   f"0x (no leverage pays after {r['borrow_rate'] * 100:.0f}% borrowing)")
        st.markdown(
            f"**Kelly point.** Estimated on the underlying {ticker} itself (not on the chosen strategy), using only "
            f"the years before the out-of-sample period ({pd.Timestamp(r['train_start']).date()} to "
            f"{pd.Timestamp(r['train_end']).date()}), the growth curve peaks at about **{r['L_star']:.2f}x** "
            f"(plausible range {r['L_star_lo']:.2f}x to {r['L_star_hi']:.2f}x) before borrowing costs, and at "
            f"{net_txt}. On the gross curve (before borrowing costs), at twice that point long-run growth falls "
            f"back to about zero.")
        st.caption("The Kelly point and the table above are different things: the table is for the chosen strategy "
                   "over the later out-of-sample period, the Kelly curve is for the underlying asset over earlier "
                   "years, so they can disagree. In particular, leverage gains on SPY and QQQ in the table come "
                   "from the asset's 2010-2024 bull market, not from the model.")
    st.markdown(
        "##### Why a small edge plus leverage usually ends badly\n"
        "A strategy that is right 52% of the time gains a tiny amount per day, but leverage multiplies the "
        "losses and the swings by the same factor, and the swings cost you more than the average gain adds "
        "(the 'volatility drag', about half of volatility squared times L squared). Past the Kelly point more "
        "leverage means *less* long-run growth, and the estimate of that point is itself very uncertain. "
        "A crash then hits a bigger position, a broker may force a sale at the bottom, and a 100% loss can "
        "never be recovered. So a small, uncertain edge is not a reason to borrow.")


# =========================================================================== page 3: diagnostics
def page_diagnostics():
    sig = data_signature()
    st.title("Model diagnostics")
    st.markdown("Is the model's *probability* honest, and does it add information? All numbers are out of sample "
                "(development period only).")
    names = DD.asset_names()
    c1, c2, c3 = st.columns(3)
    ticker = c1.selectbox("Asset", DD.tickers(), format_func=lambda t: f"{names[t]} ({t})", key="dg_asset")
    res = c_results(ticker, sig)
    model = c2.selectbox("Model", list(MODEL_LABEL), format_func=MODEL_LABEL.get, key="dg_model")
    windows = sorted(res.predictions["window"].unique(), key=lambda w: (w != "expanding", w))
    window = c3.selectbox("Training window", windows, key="dg_window")
    preds = DD.model_predictions(res, model, window)
    if preds.empty:
        st.warning("No saved predictions for this choice.")
        return
    st.caption(f"{len(preds)} out-of-sample days from {preds['date'].min().date()} to {preds['date'].max().date()}.")
    pm = res.prediction_metrics
    pm = pm[(pm["window"] == window) & (pm["model"] == model)]
    if not pm.empty:
        r = pm.iloc[0]
        k1, k2, k3 = st.columns(3)             # 3 + 2 (not 5 in a row): the labels stay readable at 1366 px
        k4, k5, _ = st.columns(3)
        k1.metric("AUC (0.5 = coin flip)", num(r["auc"], 3))
        k2.metric("Brier skill vs training base rate", num(r["brier_skill"], 4),
                  help="Can look positive from base-rate drift alone; read it with the next number.")
        ev = r.get("brier_skill_vs_eval_base", float("nan"))
        k3.metric("Brier skill vs evaluation base rate", num(ev, 4),
                  help="Against the up-day share of the evaluation period itself. This one cannot be flattered by drift.")
        k4.metric("Pesaran-Timmermann p-value", num(r["pt_pvalue"], 3))
        k5.metric("Said 'up' on", pct(r["share_up_predictions"], 0),
                  help=f"The market actually went up on {pct(r['share_up_days'], 0)} of days.")
        st.caption("Why two Brier skills? The first compares with the up-day share of the TRAINING years. If up days "
                   "became more common later, it can look positive from that drift alone (Bitcoin: +0.0059 against "
                   "the training rate, but -0.0028 against the evaluation period's own rate). The evidence badge "
                   "uses the smaller of the two.")
        pv = r["pt_pvalue"]
        verdict = ("no evidence that the up/down calls beat chance" if not (pv < 0.05 and r["pt_statistic"] > 0)
                   else "the up/down calls beat chance")
        st.markdown(f"**Pesaran-Timmermann test:** p = {num(pv, 3)} (statistic {num(r['pt_statistic'], 2)}): {verdict}. "
                    "It asks whether the up/down calls beat chance *given* how often the market and the model each "
                    "say 'up' (a model that always says 'up' can look accurate with zero skill). "
                    "A p-value under 0.05 with a positive statistic would count as evidence.")
    glossary({"AUC": "the chance the model gives a higher probability to a random up day than to a random down day. 0.5 is a coin flip; realistic daily models reach 0.51-0.55.",
              "Brier skill": "how much better the probabilities are than just quoting an up-rate. Zero or below = no better. "
                             "Shown against the TRAINING up-rate (can look positive from drift) and against the "
                             "EVALUATION period's own up-rate (cannot).",
              "Reliability curve": "if the model says 60% on 100 days, about 60 of them should be up days. Points near the diagonal mean honest probabilities.",
              "Permutation importance": "scramble one signal group and see how much worse the predictions get. Near zero = the model is not really using it."})
    rel = res.reliability
    rel = rel[(rel["window"] == window) & (rel["model"] == model)]
    dec = res.deciles
    dec = dec[(dec["window"] == window) & (dec["model"] == model)]
    a, b = st.columns(2)
    with a:
        if rel.empty:
            st.info("No saved reliability curve.")
        else:
            show(ui.reliability_fig(rel), "dg_rel")
    with b:
        if dec.empty:
            st.info("No saved decile table.")
        else:
            show(ui.decile_fig(dec, float(np.mean(preds["y"] > 0.5))), "dg_dec")
    auc, sk = DD.rolling_auc(preds, 250), DD.rolling_brier_skill(preds, 250)
    ske = DD.rolling_brier_skill_eval(preds, 250)
    c, d = st.columns(2)
    with c:
        show(ui.rolling_fig(auc.dropna(), sk.dropna(), ske.dropna()), "dg_roll")
        st.caption("Why 250 days and not 60? Accuracy over 60 days has a standard error of about 6.5 percentage "
                   "points, so a 60-day number moving between 45% and 60% is mostly noise.")
    with d:
        imp = res.get("importance")
        imp = imp[(imp["window"] == window) & (imp["model"] == model)] if not imp.empty else imp
        if imp.empty:
            st.info("No saved permutation importance for this choice (saved for the expanding window only).")
        else:
            show(ui.importance_fig(imp), "dg_imp")
    e1, e2 = st.columns([1, 1])
    with e1:
        show(ui.confusion_fig(DD.confusion_matrix(preds)), "dg_cm")
    with e2:
        st.markdown("**Reading the confusion matrix.** Rows are what the market did, columns what the model said "
                    "(it says 'up' when its probability is at or above the usual up-rate). A model with no skill "
                    "puts the same share of up days in both columns.")


# =========================================================================== page 4: paper trading
def page_paper():
    sig = data_signature()
    st.title("Paper trading")
    st.warning("A few live days prove nothing. One week of results, up or down, is well inside what luck alone "
               "produces: it takes years of daily data to tell a real edge from chance. Use this page to check "
               "that the machinery works and to watch for decay over many months, not to judge the strategy.")
    st.markdown("A pretend account with **fake money** that follows the strategy live, one decision per day. "
                "It starts on the first day the project ran, so there is very little history yet.")
    led = c_ledger(sig)
    if led.empty:
        st.info("The paper-trading ledger is empty. Press 'Refresh data' in the sidebar to run the daily update.")
        return
    names = DD.asset_names()
    last = led.sort_values("date").groupby("asset").tail(1).set_index("asset")
    counts = led.groupby("asset").size()
    st.markdown("##### Open positions")
    rows = []
    for t in DD.tickers():
        if t in last.index:
            r = last.loc[t]
            rows.append({"Asset": f"{names[t]} ({t})", "Decision for": str(r["date"].date()),
                         "Position": f"{r['position'] * 100:.0f}% invested",
                         "Model probability": pct(r["p_up"]), "Usual up-rate": pct(r["base_rate"]),
                         "Account value ($)": f"{r['equity']:,.2f}", "Profit/loss ($)": f"{r['cum_pnl']:+,.2f}"})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", column_config=wide_headers(rows))
    st.markdown("##### Equity curves")
    cols = st.columns(2)
    for i, t in enumerate(DD.tickers()):
        with cols[i % 2]:
            n = int(counts.get(t, 0))
            if n < 2:
                st.info(f"{t}: only {n} day(s) of data so far - a line needs at least 2 days. It will draw itself "
                        "as the daily updates add days.")
                if n:
                    r = last.loc[t]
                    one = [{"Date": str(r["date"].date()), "Paper strategy ($)": f"{r['equity']:,.2f}",
                            "Buy and hold ($)": f"{r['bh_equity']:,.2f}",
                            "Vol-targeted buy and hold ($)": f"{r['vtbh_equity']:,.2f}"}]
                    st.dataframe(pd.DataFrame(one), hide_index=True, width="stretch", column_config=wide_headers(one))
            else:
                show(ui.paper_equity_fig(led, t), f"pp_eq_{t}")
    st.markdown("##### Profit and loss against both baselines")
    pl = []
    cap = DD._cfg()["paper_trading"]["starting_capital"]
    for t in DD.tickers():
        if t in last.index:
            r = last.loc[t]
            start = float(cap[t]) if isinstance(cap, dict) else float(cap)
            pl.append({"Asset": t, "Strategy ($)": f"{r['equity'] - start:+,.2f}",
                       "Buy and hold ($)": f"{r['bh_equity'] - start:+,.2f}",
                       "Vol-targeted buy and hold ($)": f"{r['vtbh_equity'] - start:+,.2f}",
                       "Costs paid ($)": f"{r['cum_cost_usd']:,.2f}", "Trades so far": int(r["n_trades"])})
    st.dataframe(pd.DataFrame(pl), hide_index=True, width="stretch", column_config=wide_headers(pl))
    st.markdown("##### Trade log")
    tr = led[led["trade"].abs() > P.TRADE_TOL].sort_values(["date", "asset"])
    if tr.empty:
        st.info("No trades yet.")
    else:
        st.dataframe(pd.DataFrame({
            "Date": tr["date"].dt.date, "Asset": tr["asset"],
            "Position before": tr["prev_position"].map(lambda v: f"{v * 100:.0f}%"),
            "Position after": tr["position"].map(lambda v: f"{v * 100:.0f}%"),
            "Cost ($)": tr["cost_usd"].map(lambda v: f"{v:.2f}")}), hide_index=True, width="stretch")
    st.markdown("##### Live accuracy against the backtest")
    lb = []
    for t in DD.tickers():
        if t in last.index:
            r = last.loc[t]
            lb.append({"Asset": t, "Live days scored": int(r["live_n"]), "Live Brier score": num(r["live_brier"], 4),
                       "Backtest Brier score": num(r["bt_brier"], 4), "Live hit rate": pct(r["live_hit_rate"]),
                       "Backtest hit rate": pct(r["bt_hit_rate"]), "Backtest days": int(r["bt_n"])})
    st.dataframe(pd.DataFrame(lb), hide_index=True, width="stretch", column_config=wide_headers(lb))
    st.caption("The Brier score is the average squared error of the probabilities (lower is better, 0.25 = always "
               "saying 50%). It can only be scored once the day being predicted has finished.")


# =========================================================================== page 5: trials
def luck_bar_ann(ticker: str, entry: dict, scope: str) -> float:
    """Annualised luck bar (the DSR's SR0) of one asset's chosen strategy, ``scope`` = 'asset' or 'global'
    (252 days a year; 365 for Bitcoin, which trades every day)."""
    sr0 = entry.get(f"sr0_{scope}")
    if sr0 is None or not np.isfinite(sr0):
        return float("nan")
    weekend = {a["ticker"]: bool(a["trades_weekends"]) for a in DD._cfg()["assets"]}.get(ticker, False)
    return float(sr0 * np.sqrt(365 if weekend else 252))


def page_trials():
    sig = data_signature()
    st.title("Trials")
    st.markdown("Every model, setting and strategy variation we tried was logged. This matters because **the more "
                "you try, the higher a Sharpe ratio you get by luck alone**.")
    summ, idx = c_trials(sig)
    if summ["n_trials_global"] < 1 or idx.empty:
        st.warning("The trial log is missing or empty, so the Trials page cannot be drawn. Run "
                   "`uv run python -m quant.backtest` to create the saved results (data/results/trials_index.parquet).")
        return
    n_alias = int(sum(len(json.loads(a)) for a in idx["aliases"] if a))
    k1, k2, k3 = st.columns(3)
    k1.metric("Distinct configurations tried", summ["n_trials_global"])
    k2.metric("Including identical duplicates (aliases)", summ["n_trials_global"] + n_alias,
              help="Some settings produced exactly the same trades as another one; they are stored once with the other names as aliases.")
    k3.metric("Best annualised Sharpe among them", num(summ["annual_sharpe_best"]))
    var = summ["daily_sharpe_variance"]
    luck_daily = M.expected_max_sharpe(summ["n_trials_global"], var)
    luck_ann = luck_daily * np.sqrt(252)
    st.markdown(
        f"**What this means.** If every one of the {summ['n_trials_global']} configurations were pure noise and "
        f"completely independent, the best of them would be expected to show an annualised Sharpe of about "
        f"**{luck_ann:.2f}** just by luck. Treat {luck_ann:.2f} as a **conservative upper-bound bar**: it assumes "
        f"{summ['n_trials_global']} independent trials, but the trials are heavily correlated (many differ only "
        f"slightly), so the number of effectively independent trials is far smaller and the true luck bar is lower. "
        f"The spread behind it also mixes differences between the four assets with true noise, which pushes the bar "
        f"up. The best we actually found is {summ['annual_sharpe_best']:.2f} and the typical one "
        f"{summ['annual_sharpe_median']:.2f}. The **Deflated Sharpe Ratio (DSR)** is the probability that the real "
        "Sharpe beats that luck bar: close to 1 means convincing, close to 0 means 'what you would expect from "
        "trying this many things'.")
    st.caption(f"Annualisation: the luck bar uses 252 trading days a year (the ETF convention). The Sharpe ratios of "
               f"the configurations in the histogram and tables are annualised per asset (Bitcoin trades every "
               f"day, so its own figures use 365 days); the equivalent Bitcoin luck bar would be about "
               f"{luck_daily * np.sqrt(365):.2f}.")
    a, b = st.columns(2)
    with a:
        show(ui.trials_hist_fig(idx["ann_sharpe"].dropna(), luck_ann, summ["n_trials_global"]), "tr_hist")
    with b:
        ns = np.unique(np.round(np.logspace(0, 3, 60)).astype(int))
        curve = [M.expected_max_sharpe(int(n), var) * np.sqrt(252) for n in ns]
        marks = {f"{summ['n_trials_global']} tried": (summ["n_trials_global"], luck_ann)}
        show(ui.luck_curve_fig(ns, curve, marks), "tr_luck")
    st.markdown("##### The chosen strategy of each asset, before and after allowing for the search")
    rows = []
    for t, e in summ["assets"].items():
        if "psr" in e:
            rows.append({"Asset": t, "Tried (this asset)": e["n_trials_asset"],
                         "Daily Sharpe": num(e.get("daily_sharpe"), 3),
                         "Looks real? (PSR, vs zero)": num(e["psr"], 3),
                         "After this asset's trials (DSR)": num(e.get("dsr_asset"), 3),
                         f"After all {summ['n_trials_global']} trials (DSR)": num(e.get("dsr_global"), 3),
                         "Excess (DSR)": num(e.get("dsr_excess_vt"), 3),
                         "Luck bar, own trials": num(luck_bar_ann(t, e, "asset"), 2),
                         "Luck bar, all trials": num(luck_bar_ann(t, e, "global"), 2)})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", column_config=wide_headers(rows))
    spy = summ["assets"].get("SPY", {})
    st.markdown(
        "**Why do the two DSR columns differ?** *After this asset's trials* compares the strategy only with that "
        "asset's own configurations. Those are mostly near-copies of each other (small changes of margin, sizing or "
        "model), so their Sharpe ratios barely differ and the luck bar they imply is low"
        + (f" (SPY: about {luck_bar_ann('SPY', spy, 'asset'):.2f} a year)" if np.isfinite(luck_bar_ann("SPY", spy, "asset")) else "")
        + f". *After all {summ['n_trials_global']} trials* pools every asset: the spread of those Sharpe ratios also "
        "contains real differences between a stock index and Bitcoin, so the luck bar is much higher"
        + (f" (about {luck_bar_ann('SPY', spy, 'global'):.2f} a year)" if np.isfinite(luck_bar_ann("SPY", spy, "global")) else "")
        + " and the pooled DSR is near 0. **The pooled bar is the stricter reading**, and it is the one the evidence "
        "badge uses (DSR_G); the per-asset bar is the more lenient one because near-copies understate how many "
        "genuinely different ideas were tried. Neither is exact: the true number of independent trials lies "
        "somewhere in between.")
    st.caption("The main DSR columns test the Sharpe against zero only. Stocks and Bitcoin mostly rose over the "
               "test years, so they can be high without any skill, and they are NOT evidence of skill when the "
               "evidence badge is grey. The last column, Excess (DSR), asks the harder question: does the strategy beat simply "
               "owning the asset with volatility-targeted sizing?")
    st.markdown(
        "**Reading the table.** *PSR* asks whether the Sharpe is above zero given how short and bumpy the data are; "
        "stocks mostly rise, so it is often high. *DSR* then raises the bar to what the best of many tries would "
        "reach by luck. A strategy can pass the first and fail the second, which is what 'overfitting by search' looks "
        "like. Also note that being above zero is not the same as beating buy-and-hold: see the Backtest page.")
    glossary({"Sharpe ratio": "average return divided by its ups and downs, per year.",
              "PSR": "probability that the true Sharpe is above a benchmark, allowing for short samples and fat tails.",
              "DSR": "PSR with the benchmark raised to 'the best result luck alone would give from N tries'.",
              "Alias": "a setting that produced exactly the same daily returns as another, so it is stored once."})


# =========================================================================== page 6: how it works
def page_how():
    th = c_theory()
    st.title("How it works")
    st.markdown("A beginner's tour. Nothing here predicts the market; it explains what the project measures and "
                "why most of the time the honest result is 'not much'.")
    st.header("1. The idea in one minute")
    st.markdown(
        "Each day the project asks: *what is the probability that tomorrow is an up day?* It builds some numbers "
        "from past prices (**signals**), feeds them to a few simple **models**, and turns the probability into a "
        "paper-trading position (hold or step aside). Everything is judged **out of sample**: on days the model "
        "had not seen when it was trained.")
    st.header("2. The signals (what the model looks at)")
    for g, e in th["feature_groups"].items():
        theory_card(e)
    st.header("3. The models")
    for k in ("logistic_regression", "random_forest", "lightgbm", "ensemble"):
        theory_card(th["models"][k])
    st.header("4. How we test honestly")
    for k in ("walk_forward", "calibration", "climatology_baseline", "brier_skill", "volatility_targeting"):
        theory_card(th["models"][k])
    st.header("5. How to read the meter and the evidence badge")
    theory_card(th["models"]["signal_meter"], expanded=True)
    zt = pd.DataFrame({"Zone": list(E.ZONES), "Meaning": [
        "The strategy steps aside (cash). Never a short sale. Used when today's gap (probability minus the usual "
        "up-rate) is below the strategy's tolerance line, minus m.",
        "Reduce the position (used only by the optional 'sized' mode; crossed out on the dial otherwise).",
        "Half-way state (used only by the optional 'sized' mode; crossed out on the dial otherwise).",
        "The strategy holds. Hold does not mean bullish: it can show while the probability is slightly below usual.",
        "Fully invested. Exists only in the optional 'sized' mode. The current long/flat strategy has no stronger "
        "step than holding, so this zone is crossed out and no 'strong buy' action is invented. The needle is "
        "today's gap on the same axis as the zone edges, so it always sits in the zone of the paper position; its "
        "percentile among past days is shown as small print."]})
    st.dataframe(zt, hide_index=True, width="stretch")
    theory_card(th["models"]["evidence_badge"], expanded=True)
    bt = pd.DataFrame({"Badge": [v[0] for v in E.BADGES.values()], "Colour": [v[1] for v in E.BADGES.values()],
                       "What it means": ["Anything else, or any number missing. The backtest did not show a reliable edge. "
                                         "The meter is drawn faded. This is the default and the most common honest answer.",
                                         "The strategy beat vol-targeted buy and hold after costs (BEATS), AND either the up/down "
                                         "calls beat chance (PT) or its excess return over that benchmark is convincing (PSR of "
                                         "the excess >= 0.95).",
                                         "All five pass: PT (up/down calls beat chance, p < 0.05, positive), SKILL (Brier skill "
                                         "above zero against BOTH the training and the evaluation base rate; the smaller is "
                                         "used), BEATS (net Sharpe above vol-targeted buy and hold), DSR_X (excess return still "
                                         "convincing after this asset's trials, >= 0.95) and DSR_G (same after all 270 trials)."]})
    st.dataframe(bt, hide_index=True, width="stretch")
    with st.expander("The exact badge rule"):
        st.code(E.BADGE_RULE, language=None)
    st.caption(f"The meter is only ever shown with this disclaimer: {E.DISCLAIMER}")


# =========================================================================== page 7: reality check
def page_reality():
    sig = data_signature()
    th = c_theory()["reality_check"]
    st.title("Reality check")
    st.markdown("Predicting daily market direction is one of the hardest problems in finance. Here is what the "
                "research says, in plain language, and what this project's own numbers look like next to it.")
    glossary(REALITY_TERMS, "Plain-English glossary for this page")
    order = ["realistic_auc", "cross_sectional_vs_index", "edges_negative_after_costs", "post_publication_decay",
             "holdout_power", "overfitting", "significance_tests", "leverage_kelly"]
    for k in order:
        e = th[k]
        st.subheader(e["title"])
        st.markdown(" ".join(str(e["plain_english"]).split()))
        if e.get("evidence"):
            st.markdown("**Evidence:** " + " ".join(str(e["evidence"]).split()))
        if e.get("caveats"):
            st.markdown("**Caveats:** " + " ".join(str(e["caveats"]).split()))
        if e.get("sources"):
            with st.expander("Sources"):
                st.markdown(src_links(e["sources"]))
        if k == "realistic_auc":
            own_auc_table(sig)
        if k == "significance_tests":
            own_sample_hurdle(sig)
        if k == "holdout_power":
            holdout_power_table()
        if k == "overfitting":
            overfit_panel(sig)


REALITY_TERMS = {
    "AUC": "the chance the model ranks a random up day above a random down day. 0.5 is a coin flip.",
    "Sharpe ratio": "average return divided by how much it bounces around, per year.",
    "Out-of-sample": "tested on days the model had not seen when it was trained.",
    "Hold-out": "the final stretch of data (here the last 24 months) kept locked away until the very end.",
    "PSR / DSR": "Probabilistic / Deflated Sharpe Ratio: the chance the true Sharpe beats a benchmark; the deflated "
                 "version raises the benchmark to what luck alone would give from many tries.",
    "Diebold-Mariano": "a test of whether one forecaster is really more accurate than another.",
    "Hansen SPA": "a test of whether the best of many strategies beats a benchmark by more than luck.",
    "t > 3": "a stricter bar than the usual t > 2, because trying many things makes lucky results common.",
    "bp (basis point)": "one hundredth of a percent; 30 bp = 0.30%.",
    "Kelly": "the bet size (or leverage) that maximises long-run growth; bigger bets than that reduce growth.",
    "Cross-sectional": "ranking many stocks against each other, rather than timing one index up or down.",
    "LSTM / GBT": "a neural-network type and gradient-boosted trees: the more flexible models used in some studies.",
    "Data-snooping-adjusted p-value": "a p-value corrected for the fact that many strategies were tried.",
}


def own_auc_table(sig):
    rows = []
    for t in DD.tickers():
        pm = c_results(t, sig).prediction_metrics
        pm = pm[(pm["window"] == "expanding") & (pm["model"].isin(["logreg", "rf", "lgbm", "ensemble"]))]
        r = {"Asset": t}
        for _, x in pm.iterrows():
            r[MODEL_LABEL[x["model"]].split(" (")[0] + " AUC"] = num(x["auc"], 3)
        rows.append(r)
    st.markdown("**This project's own out-of-sample AUC** (saved backtest, 0.5 = coin flip; research range 0.51-0.55):")
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    vals = [float(v) for r in rows for k, v in r.items() if k != "Asset" and v != "n/a"]
    if vals:
        low = sum(v <= 0.5 for v in vals)
        st.markdown(
            f"**In plain words:** {low} of these {len(vals)} AUCs are at or below 0.5 and the best is {max(vals):.3f}. "
            + ("Mostly at or below 0.5 means the models were no better than a coin flip at ranking up days above "
               "down days, which is the honest, expected result for daily direction." if low > len(vals) / 2 else
               "Values this close to 0.5 are barely better than a coin flip, which is the honest, expected result "
               "for daily direction."))


def own_sample_hurdle(sig):
    """The report's 'needs Sharpe above ~0.6 (SPY) / ~0.87 (BTC)' is the REPORT's figure for its own sample
    lengths; this project's out-of-sample periods are shorter, so the bar (t = 3 -> Sharpe = 3 / sqrt(years)) is higher."""
    rows = []
    for t in DD.tickers():
        res = c_results(t, sig)
        yrs = DD.oos_sample_years(res)
        rows.append({"Asset": t, "Out-of-sample years": num(yrs, 1),
                     "Annual Sharpe needed for t = 3": num(DD.sharpe_needed_for_t(yrs), 2)})
    st.markdown("**Caution on the numbers above.** The 'about 0.6 (SPY) / 0.87 (BTC)' figures are the research "
                "report's own, for its sample lengths. This project's out-of-sample samples are shorter, so the bar "
                "is higher (needed Sharpe = 3 divided by the square root of the years):")
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")


def holdout_power_table():
    rows = []
    for yrs in (1, 2, 5, 10):
        n = 252 * yrs
        rows.append({"Years of daily data": yrs, "Standard error of the yearly Sharpe": num(M.sharpe_standard_error(0.0, n, 252), 2),
                     "Standard error of the hit rate (points)": num(100 * np.sqrt(0.25 / n), 1)})
    st.markdown("**How uncertain a short test is** (same formulas the dashboard uses for its Sharpe bands):")
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption("A true yearly Sharpe of 0.5 is small next to the standard error (about 0.71 after two years), so a "
               "1-2 year hold-out can show that a strategy has failed but cannot prove it works.")


def overfit_panel(sig):
    summ, _ = c_trials(sig)
    if summ["n_trials_global"] < 1:
        st.info("The trial log is missing or empty, so the overfitting numbers cannot be shown.")
        return
    st.markdown(
        f"**What it looks like here.** We tried {summ['n_trials_global']} configurations. Their annualised Sharpe ratios "
        f"range up to {summ['annual_sharpe_best']:.2f} (median {summ['annual_sharpe_median']:.2f}). Picking the best "
        "one and calling it 'the strategy' is exactly how overfitting happens. See the Trials page for the luck bar.")


# =========================================================================== shell
def sidebar() -> str:
    st.sidebar.title("Quant Lab")
    st.sidebar.caption("Daily market-direction research with fake money.")
    page = st.sidebar.radio("Page", PAGES, key="page")
    st.sidebar.divider()
    if st.sidebar.button("Refresh data", key="refresh", help="Download new prices and catch up the paper-trading ledger."):
        with st.spinner("Updating prices and the paper-trading ledger..."):
            out = DD.refresh_data()
        st.session_state["refresh_result"] = (out.ok, out.message)
        st.cache_data.clear()
        st.cache_resource.clear()
    res = st.session_state.get("refresh_result")
    if res:
        (st.sidebar.success if res[0] else st.sidebar.warning)(res[1])
    try:
        cut = f"Hold-out (from {DD.cutoff().date()}) is never shown"
    except Exception:      # noqa: BLE001 - a broken or missing config.yaml must not break the sidebar
        cut = "Hold-out cutoff date unavailable (config.yaml could not be read); the hold-out is never shown"
    st.sidebar.caption(f"{cut}: only saved development results and the live paper ledger.")
    return page


def footer():
    st.divider()
    st.caption(FOOTER)


def run_page(fn):
    """Run one page; any expected failure becomes a plain message (never a traceback)."""
    try:
        fn()
    except Exception as e:      # noqa: BLE001 - see explain_failure: every kind gets a clear message
        show_failure(e)


def safe_sidebar() -> str:
    """The sidebar inside the same failure handler as the pages: whatever goes wrong becomes a plain message
    (never a traceback) and the page choice still works."""
    try:
        return sidebar()
    except Exception as e:      # noqa: BLE001 - see explain_failure
        show_failure(e)
        page = st.session_state.get("page", PAGES[0])
        return page if page in PAGES else PAGES[0]


def main():
    page = safe_sidebar()
    fn = {"Today": page_today, "Backtest": page_backtest, "Model diagnostics": page_diagnostics,
          "Paper trading": page_paper, "Trials": page_trials, "How it works": page_how,
          "Reality check": page_reality}[page]
    run_page(fn)
    footer()


main()
