"""Post-cap fixes to Stage 5 (approved by Wiktor after the 3-round cap).

A. A broken or missing config.yaml gives a plain message on all 7 pages (never a traceback).
B. The untested round-3 guards: vt_baseline_percentile, refresh_data with an unreadable config, and the
   "Why?" conclusion clauses ("nudged it down", the percentile phrase).
C. The meter invariant: the needle is TODAY'S GAP on an axis whose zone edges are the active strategy's real
   thresholds, so its zone ALWAYS equals the zone of the paper position (property tests over many random,
   boundary and clamped cases for long_flat, default_long and sized), plus the simplified Today card.
"""
import dataclasses
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from app import ui
from quant import config as C
from quant import dash_data as DD
from quant import explain as E
from quant import metrics as M
from quant import paper as P
from tests import dash_world as W
from tests.synth import TICKERS as SYN_TICKERS

ROOT = Path(__file__).resolve().parents[1]
APP = str(ROOT / "app" / "dashboard.py")
PAGES = ["Today", "Backtest", "Model diagnostics", "Paper trading", "Trials", "How it works", "Reality check"]
FOOTER = "Research and education only, not financial advice."
CFG = DD._cfg()
TICKERS = DD.tickers(CFG)


def blob(at) -> str:
    out = []
    for kind in ("markdown", "caption", "info", "warning", "success", "error", "title", "header", "subheader", "code"):
        out += [str(e.value) for e in at.get(kind)]
    for m in at.metric:
        out += [str(m.label), str(m.value)]
    return "\n".join(out)


@pytest.fixture(autouse=True)
def fresh_caches():
    st.cache_data.clear()
    st.cache_resource.clear()
    yield
    st.cache_data.clear()
    st.cache_resource.clear()


# =========================================================================== A. broken / missing config.yaml
@pytest.fixture(params=["broken_yaml", "deleted"])
def bad_config(request, tmp_path, monkeypatch):
    """The project root points at a folder whose config.yaml is invalid YAML (``foo: [``) or absent."""
    if request.param == "broken_yaml":
        (tmp_path / "config.yaml").write_text("foo: [\n")
    monkeypatch.setattr(C, "PROJECT_ROOT", tmp_path)
    return request.param


@pytest.mark.parametrize("page", PAGES)
def test_broken_or_missing_config_is_a_plain_message_on_every_page(bad_config, page):
    at = AppTest.from_file(APP, default_timeout=300)
    at.session_state["page"] = page
    at.run()
    assert not at.exception, (page, bad_config, [e.value for e in at.exception])
    text = blob(at)
    assert "Traceback" not in text
    assert any(FOOTER == str(c.value) for c in at.caption), page                       # the page still finished
    side = "\n".join(str(c.value) for c in at.sidebar.caption)
    assert "Hold-out cutoff date unavailable" in side and "config.yaml could not be read" in side
    assert [b.label for b in at.sidebar.button] == ["Refresh data"]                  # the sidebar still works
    if page != "How it works":                                                         # that page needs no config
        msgs = "\n".join(str(x.value) for x in list(at.error) + list(at.warning))
        assert "config.yaml could not be used" in msgs and "Restore config.yaml" in msgs, (page, msgs)


def test_a_failure_inside_sidebar_itself_is_caught_by_the_same_handler(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("sidebar exploded")
    monkeypatch.setattr(st.sidebar, "title", boom)
    at = AppTest.from_file(APP, default_timeout=300)
    at.session_state["page"] = "How it works"
    at.run()
    assert not at.exception and "Traceback" not in blob(at)
    assert any("Something unexpected went wrong (RuntimeError: sidebar exploded)" in str(e.value) for e in at.error)
    assert any(FOOTER == str(c.value) for c in at.caption)                              # the page still drew


def test_config_error_detector_only_claims_config_problems():
    import importlib.util
    spec = importlib.util.spec_from_file_location("dash_ns2", APP)
    src = Path(APP).read_text().rstrip()
    ns = {"__name__": "dash_ns2", "__file__": APP}
    exec(compile(src[: -len("main()")], APP, "exec"), ns)
    f = ns["is_config_error"]
    assert f(FileNotFoundError("Config file not found: /x/config.yaml"))
    assert f(ValueError("Config file /x/config.yaml is not valid YAML: boom"))
    assert f(ValueError("Config missing required key(s): assets"))
    assert not f(FileNotFoundError("data/SPY/predictions.parquet"))
    assert not f(ValueError("no saved predictions"))
    assert not f(RuntimeError("Config file not found"))


# =========================================================================== B. the untested round-3 guards
def _res(baselines, predictions):
    return DD.AssetResults(ticker="X", tables={"baselines": baselines, "predictions": predictions})


def _preds(dates, nr):
    return pd.DataFrame({"model": "ensemble", "window": "expanding", "date": dates, "p_up": 0.5, "base_rate": 0.5,
                         "y": 1.0, "next_return": nr})


def _base(dates, pos, name="vol_targeted_bh"):
    return pd.DataFrame({"name": name, "window": "expanding", "date": dates, "position": pos})


DAYS = pd.bdate_range("2020-01-01", periods=60)
RNG = np.random.default_rng(11)


def test_vt_baseline_percentile_returns_a_number_for_a_healthy_input():
    nr = RNG.normal(0.0005, 0.01, len(DAYS))
    v = DD.vt_baseline_percentile(_res(_base(DAYS, RNG.uniform(0.2, 1, len(DAYS))), _preds(DAYS, nr)))
    assert isinstance(v, float) and 0.0 <= v <= 100.0


@pytest.mark.parametrize("n", [0, 1, 2])
def test_vt_baseline_percentile_needs_at_least_three_days(n):
    d = DAYS[:n]
    assert DD.vt_baseline_percentile(_res(_base(d, np.ones(n)), _preds(d, np.full(n, 0.01)))) is None


def test_vt_baseline_percentile_three_days_is_enough():
    d = DAYS[:3]
    v = DD.vt_baseline_percentile(_res(_base(d, [0.2, 0.9, 0.5]), _preds(d, [0.01, -0.02, 0.03])))
    assert v is not None and 0.0 <= v <= 100.0       # computed, not refused by the length guard


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_vt_baseline_percentile_refuses_non_finite_returns(bad):
    nr = RNG.normal(0, 0.01, len(DAYS))
    nr[10] = bad
    assert DD.vt_baseline_percentile(_res(_base(DAYS, np.ones(len(DAYS)) * 0.5), _preds(DAYS, nr))) is None


def test_vt_baseline_percentile_refuses_baseline_days_without_a_return():
    nr = RNG.normal(0, 0.01, len(DAYS))
    short = _preds(DAYS[:-5], nr[:-5])                  # five baseline days have no matching prediction -> NaN
    assert DD.vt_baseline_percentile(_res(_base(DAYS, np.full(len(DAYS), 0.5)), short)) is None


def test_vt_baseline_percentile_empty_baselines_or_predictions_is_none():
    nr = RNG.normal(0, 0.01, len(DAYS))
    full_b, full_p = _base(DAYS, np.full(len(DAYS), 0.5)), _preds(DAYS, nr)
    assert DD.vt_baseline_percentile(_res(full_b.iloc[0:0], full_p)) is None
    assert DD.vt_baseline_percentile(_res(full_b, full_p.iloc[0:0])) is None
    assert DD.vt_baseline_percentile(_res(full_b.iloc[0:0], full_p.iloc[0:0])) is None
    assert DD.vt_baseline_percentile(_res(full_b, full_p), name="no_such_baseline") is None


def test_vt_baseline_percentile_nan_percentile_becomes_none():
    nr = RNG.normal(0, 0.01, len(DAYS))
    flat = _res(_base(DAYS, np.zeros(len(DAYS))), _preds(DAYS, nr))            # never invested: no variance -> NaN
    assert np.isnan(M.random_strategy_percentile(nr, np.zeros(len(DAYS)), n=50, seed=0)["percentile"])
    assert DD.vt_baseline_percentile(flat) is None


def test_refresh_data_with_an_unreadable_config_is_ok_false_and_never_raises(tmp_path, monkeypatch):
    (tmp_path / "config.yaml").write_text("foo: [\n")
    monkeypatch.setattr(C, "PROJECT_ROOT", tmp_path)
    called = []
    out = DD.refresh_data(run=lambda *a, **k: called.append(1))
    assert out.ok is False and out.message.startswith("Could not read config.yaml") and "not valid YAML" in out.message
    assert not called                                                   # nothing was run
    (tmp_path / "config.yaml").unlink()
    out = DD.refresh_data(run=lambda *a, **k: called.append(1))
    assert out.ok is False and "Could not read config.yaml" in out.message and "not found" in out.message
    assert not called


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    return W.build(tmp_path_factory.mktemp("postcapworld"), extra_days=("2021-10-14",))


def _why_with(world, monkeypatch, signs, percentile):
    """explain_today for SPY with the signal groups pushed in the given directions and a chosen percentile."""
    cfg = world
    t = "SPY"
    ds = W.dataset(cfg, t)
    d = DD.ledger_for(t, cfg).iloc[-1]["date"]
    model = E.load_live_model_ex(t, cfg)[0]
    preds = DD.model_predictions(DD.load_asset_results(t, cfg), "ensemble")
    meter = dataclasses.replace(E.today_meter(t, cfg), percentile=percentile)
    real = E.group_effects

    def fake(contrib, p):
        g = real(contrib, p)
        sign = np.array([signs[i % len(signs)] for i in range(len(g))], dtype=float)
        g["combined"] = g["abs"] * sign
        g["points"] = g["abs"] * sign
        return g
    monkeypatch.setattr(E, "group_effects", fake)
    why = E.explain_today(t, ds, d, model, meter, preds, E.evidence_badge(t, cfg), pd.Timestamp(cfg["holdout_cutoff_date"]))
    return why, meter


def test_why_conclusion_says_nudged_it_down_when_a_group_pushed_down(world, monkeypatch):
    why, _ = _why_with(world, monkeypatch, [-1], 37.0)
    assert " nudged it down" in why.conclusion and "nudged the probability up" not in why.conclusion
    assert all(s["direction"] == "down" for s in why.saw)


def test_why_conclusion_has_both_clauses_when_groups_disagree(world, monkeypatch):
    why, _ = _why_with(world, monkeypatch, [1, -1], 37.0)
    c = why.conclusion
    assert "nudged the probability up" in c and "nudged it down" in c and c.index("up") < c.index("down")
    assert "; " in c


def test_why_conclusion_has_only_the_up_clause_when_all_pushed_up(world, monkeypatch):
    why, _ = _why_with(world, monkeypatch, [1], 37.0)
    assert "nudged the probability up" in why.conclusion and "nudged it down" not in why.conclusion


@pytest.mark.parametrize("pct,word", [(37.0, "37th"), (1.0, "1st"), (92.4, "92nd"), (13.0, "13th")])
def test_why_conclusion_states_the_percentile_phrase(world, monkeypatch, pct, word):
    why, meter = _why_with(world, monkeypatch, [1], pct)
    assert f", which is the {word} percentile of the gaps this model has produced" in why.conclusion
    assert meter.probability_text in why.conclusion


def test_why_conclusion_has_no_percentile_phrase_without_history(world, monkeypatch):
    why, _ = _why_with(world, monkeypatch, [1], float("nan"))
    assert "percentile" not in why.conclusion and "nan" not in why.conclusion.lower()


# =========================================================================== C. the meter invariant
def settings(mode, vt=True, lam=0.25, band=0.1):
    return P.PaperSettings(
        strategy_mode=mode,
        strategy={"window": "expanding", "model": "ensemble", "mode": mode, "vol_targeted": vt, "m_setting": "adaptive"},
        sized_lambda=lam, sized_band=band, capital={}, costs={})


def oracle_zone(st_, pos, ratio, scale) -> int:
    """Zone of a position, written independently of ``classify_zone`` (the test's own reading of the spec)."""
    if not np.isfinite(pos) or pos <= 1e-12:
        return 0
    if st_.strategy_mode != "sized":
        return 3                                           # long/flat and default long: hold, nothing stronger
    k = ratio if st_.vol_targeted else 1.0
    if not (np.isfinite(k) and k > 0 and np.isfinite(scale) and scale > 0):
        return 3
    return 4 if pos >= 1 - 1e-9 else 1 if pos < 0.25 else 2 if pos < 0.5 else 3


def edges_of(r):
    return [b["lo"] for b in r.bounds] + [r.bounds[-1]["hi"]]


def check_invariant(r, st_, pos, ratio, scale):
    want = oracle_zone(st_, pos, ratio, scale)
    assert r.zone_index == want and r.zone == E.ZONES[want]
    e = edges_of(r)
    assert all(b >= a for a, b in zip(e, e[1:]))                                    # arcs are ordered
    assert e[0] == r.axis_lo and e[-1] == r.axis_hi
    assert r.axis_lo <= r.needle <= r.axis_hi
    assert E.zone_at(e, r.needle) == want, (st_.strategy_mode, r.needle, e, want)    # THE INVARIANT
    assert r.bounds[want]["reachable"]                                              # and it is a zone the strategy has
    if st_.strategy_mode != "sized":
        assert r.zones_used == ("Sell / step aside", "Hold / buy")                  # nothing invented for binary modes


def make_hist(rng):
    kind = rng.integers(0, 6)
    if kind == 0:
        return np.array([])
    if kind == 1:
        return rng.normal(0, 0.01, 1)
    if kind == 2:
        return np.full(7, 0.003)                                                    # all ties
    if kind == 3:
        return np.round(rng.normal(0, 0.012, 300), 3)                               # many ties
    if kind == 4:
        return np.concatenate([rng.normal(-0.03, 0.005, 400), [np.nan, np.inf]])    # shifted, with junk
    return rng.normal(0, 0.012, 1500)


@pytest.mark.parametrize("mode", ["long_flat", "default_long", "sized"])
@pytest.mark.parametrize("vt", [True, False])
def test_invariant_needle_zone_equals_paper_position_zone_over_random_cases(mode, vt):
    rng = np.random.default_rng(1234 + 7 * (mode == "sized") + 3 * vt)
    st_ = settings(mode, vt)
    n_clamped = 0
    for _ in range(1500):
        base = float(rng.uniform(0.45, 0.62))
        m = float(rng.choice([0.0, 1e-4, 0.005, 0.0123, 0.02, 0.05])) if mode != "sized" else float(rng.choice([np.nan, 0.01]))
        ratio = float(rng.choice([0.0, 0.3, 1.0, 2.5, np.nan]))
        scale = float(rng.choice([0.005, 0.02, 0.05, np.nan]))
        prev = float(rng.choice([0.0, 0.05, 0.35, 0.95, 1.0]))
        how = rng.integers(0, 4)
        if how == 0:
            p = base + float(rng.uniform(-0.12, 0.12))
        elif how == 1 and mode != "sized":
            p = float(np.nextafter(base - m, rng.choice([-np.inf, 0.0, np.inf])))     # on or 1 ulp from the edge
        elif how == 2 and mode == "sized" and np.isfinite(scale) and np.isfinite(ratio) and ratio > 0:
            k = ratio if vt else 1.0
            p = base + float(rng.choice([0.0, 0.25, 0.5, 1.0])) * scale / (0.25 * k) + float(rng.choice([-1e-9, 0, 1e-9]))
        else:
            p = base + float(rng.choice([-0.06, -0.02, 0.0, 0.02, 0.2]))
        p = min(max(p, 0.0), 1.0)
        hist = make_hist(rng)
        pos = P.decide_position(st_, p, base, m, ratio, scale, prev)
        r = E.meter_from_inputs(st_, p, base, m, ratio, scale, prev, hist)
        assert (r.position == pos) or (np.isnan(r.position) and np.isnan(pos))
        check_invariant(r, st_, pos, ratio, scale)
        n_clamped += r.needle_clamped
        sized_to_zero = vt and not ratio > 1e-12            # vol-targeting can shrink a held signal to nothing (ratio 0)
        if mode != "sized" and abs((p - base) + m) > 1e-9 and not sized_to_zero:
            assert not r.needle_clamped and r.needle == pytest.approx(p - base)       # binary: needle IS today's gap
        # the history only changes the small percentile, never the zone or the needle's zone
        r0 = E.meter_from_inputs(st_, p, base, m, ratio, scale, prev, [])
        assert r0.zone_index == r.zone_index
    assert n_clamped >= 0


@pytest.mark.parametrize("mode", ["long_flat", "default_long", "sized"])
def test_invariant_holds_for_any_recorded_ledger_position(mode):
    """The Today card passes the position the ledger recorded; the needle must sit in THAT position's zone."""
    rng = np.random.default_rng(99)
    st_ = settings(mode, True)
    for _ in range(600):
        base, p = float(rng.uniform(0.48, 0.58)), float(rng.uniform(0.40, 0.70))
        pos = float(rng.choice([0.0, 1e-13, 2e-12, 0.1, 0.2499, 0.25, 0.3, 0.5, 0.75, 0.999999999, 1.0, 0.4]))
        ratio, scale, m = float(rng.choice([0.5, 1.0, 2.0])), 0.02, (0.01 if mode != "sized" else np.nan)
        r = E.meter_from_inputs(st_, p, base, m, ratio, scale, 0.0, make_hist(rng), position=pos)
        check_invariant(r, st_, pos, ratio, scale)


@pytest.mark.parametrize("mode", ["long_flat", "default_long"])
@pytest.mark.parametrize("m", [0.0, 0.01, 0.0123])
def test_exact_boundary_of_the_binary_rule(mode, m):
    st_ = settings(mode, vt=False)
    base = 0.55
    thr = base - m
    on = E.meter_from_inputs(st_, thr, base, m, 1.0, 0.02, 0.0, HIST_FIXED)                     # p == base - m: still holds
    assert on.position == 1.0 and on.zone == "Hold / buy" and E.zone_at(edges_of(on), on.needle) == 3
    just_below = float(np.nextafter(thr, -np.inf))
    off = E.meter_from_inputs(st_, just_below, base, m, 1.0, 0.02, 0.0, HIST_FIXED)             # one ulp lower: steps aside
    assert off.position == 0.0 and off.zone == "Sell / step aside"
    assert E.zone_at(edges_of(off), off.needle) == 0 and off.needle < off.bounds[0]["hi"]
    just_above = float(np.nextafter(thr, np.inf))
    assert E.meter_from_inputs(st_, just_above, base, m, 1.0, 0.02, 0.0, HIST_FIXED).zone == "Hold / buy"


HIST_FIXED = np.random.default_rng(5).normal(0, 0.01, 400)


def test_the_bug_that_motivated_the_redesign_low_percentile_but_holding():
    """QQQ-like: a low percentile (6th) yet the strategy holds (gap inside the tolerance).  The old needle sat in
    the red zone while the label said Hold / buy; now the needle is in the Hold arc and the percentile is small print."""
    st_ = settings("long_flat", vt=True)
    hist = np.random.default_rng(2).normal(0.0, 0.012, 1500)
    base, m = 0.56, 0.0125
    p = base - 0.0105                                       # gap -1.05 pts: below ~19% of history, inside the margin
    r = E.meter_from_inputs(st_, p, base, m, 1.0, 0.02, 0.0, hist)
    assert r.percentile < 25 and r.position == 1.0
    assert r.zone == "Hold / buy" and E.zone_at(edges_of(r), r.needle) == 3
    deep = E.meter_from_inputs(st_, base - 0.0136, base, m, 1.0, 0.02, 0.0, hist)    # now beyond the margin
    assert deep.zone == "Sell / step aside" and E.zone_at(edges_of(deep), deep.needle) == 0


@pytest.mark.parametrize("vt,ratio", [(True, 2.0), (True, 0.5), (False, 1.0)])
def test_sized_mode_cuts_are_where_the_target_position_crosses_the_levels(vt, ratio):
    st_ = settings("sized", vt)
    scale, base = 0.04, 0.54
    k = ratio if vt else 1.0
    cuts = [c * scale / (0.25 * k) for c in (0.25, 0.5, 1.0)]
    for gap, want in [(-0.01, 0), (-1e-9, 0), (1e-6, 1), (cuts[0] - 1e-6, 1), (cuts[0] + 1e-6, 2),
                      (cuts[1] - 1e-6, 2), (cuts[1] + 1e-6, 3), (cuts[2] - 1e-6, 3), (cuts[2] + 1e-6, 4), (cuts[2] * 3, 4)]:
        r = E.meter_from_inputs(st_, base + gap, base, np.nan, ratio, scale, 0.0, HIST_FIXED, position=None)
        pos = P.decide_position(st_, base + gap, base, np.nan, ratio, scale, 0.0)
        # prev_pos = 0 and a 0.1 no-trade band: targets below 0.1 keep the old (zero) position
        check_invariant(r, st_, pos, ratio, scale)
        if abs(pos - 0.0) > 1e-12 or want == 0:
            assert r.zone_index == want
    assert [b["zone"] for b in r.bounds] == list(E.ZONES) and all(b["reachable"] for b in r.bounds)
    edges = edges_of(r)
    assert edges[1:5] == pytest.approx([0.0] + cuts)


def test_a_needle_clamped_by_the_no_trade_band_stays_in_the_position_zone():
    st_ = settings("sized", False)
    # target position 0.08 (inside the 0.1 band around the held 0) -> the account stays at 0 = Sell, gap alone says Reduce
    scale, base = 0.04, 0.54
    gap = 0.08 * scale / 0.25 * 1.0
    r = E.meter_from_inputs(st_, base + gap, base, np.nan, 1.0, scale, 0.0, HIST_FIXED)
    assert r.position == 0.0 and r.zone == "Sell / step aside" and r.needle_clamped
    assert r.needle < 0.0 < r.gap and E.zone_at(edges_of(r), r.needle) == 0
    assert "no-trade band" in " ".join(E.meter_explanation(r))
    # holding 0.35 while the target falls to 0.30 stays 'Neutral' for both
    r2 = E.meter_from_inputs(st_, base + 0.30 * scale / 0.25, base, np.nan, 1.0, scale, 0.35, HIST_FIXED)
    assert r2.position == 0.35 and r2.zone == "Neutral" and not r2.needle_clamped


def test_axis_always_contains_the_needle_every_threshold_and_has_positive_width():
    rng = np.random.default_rng(3)
    for mode in ("long_flat", "sized"):
        st_ = settings(mode)
        for _ in range(300):
            p = float(rng.uniform(0.2, 0.9))
            r = E.meter_from_inputs(st_, p, 0.54, 0.02 if mode != "sized" else np.nan, 1.5, 0.03, 0.0, make_hist(rng))
            e = edges_of(r)
            assert r.axis_hi > r.axis_lo and r.axis_lo <= min(r.gap, r.needle) and max(r.gap, r.needle) <= r.axis_hi
            assert all(r.axis_lo <= x <= r.axis_hi for x in e)


def test_zone_at_is_total_and_never_returns_a_zero_width_zone():
    e = [-1.0, 0.0, 0.0, 0.0, 1.0, 1.0]
    assert [E.zone_at(e, v) for v in (-5, -1, -0.5, -1e-12, 0.0, 0.5, 1.0, 9)] == [0, 0, 0, 0, 3, 3, 3, 3]
    assert E.zone_at([0, 0, 0, 0, 0, 0], 3.0) == 0


def test_gauge_axis_and_needle_are_in_percentage_points_of_the_gap():
    st_ = settings("long_flat")
    r = E.meter_from_inputs(st_, 0.5305, 0.54, 0.02, 1.0, 0.02, 0.0, HIST_FIXED)
    fig = ui.gauge_figure(r)
    g = fig.data[0].gauge
    assert tuple(g.axis.range) == pytest.approx((r.axis_lo * 100, r.axis_hi * 100))
    assert fig.data[0].value == pytest.approx(r.needle * 100) == pytest.approx(g.threshold.value)
    steps = [(s.range[0], s.range[1]) for s in g.steps]
    assert np.ravel(steps) == pytest.approx([r.bounds[0]["lo"] * 100, -2.0, -2.0, r.axis_hi * 100])  # two real zones only
    assert "0" in list(g.axis.ticktext) and "-2.0" in list(g.axis.ticktext)                   # the strategy's own line
    assert any("percentile of past days" in a.text for a in fig.layout.annotations)
    assert tuple(r.zones_used) == ("Sell / step aside", "Hold / buy")


def test_legend_crosses_out_the_zones_the_strategy_does_not_have():
    r = E.meter_from_inputs(settings("long_flat"), 0.54, 0.54, 0.01, 1.0, 0.02, 0.0, HIST_FIXED)
    html_ = ui.zone_legend_html(r.bounds)
    import re
    styled = {m.group(2): "line-through" in m.group(1) for m in re.finditer(r"<span style='([^']*)'>([^<]+)</span>", html_)}
    assert styled == {"Sell / step aside": False, "Reduce": True, "Neutral": True, "Hold / buy": False, "Strong buy": True}


def test_meter_explanation_describes_each_mode():
    b = E.meter_from_inputs(settings("long_flat"), 0.54, 0.54, 0.0125, 1.0, 0.02, 0.0, HIST_FIXED)
    txt = " ".join(E.meter_explanation(b))
    assert "-1.2 points" in txt or "-1.3 points" in txt
    assert "Strong buy are crossed out" in txt and "none is invented" in txt and "Small print" in txt
    s = E.meter_from_inputs(settings("sized"), 0.54, 0.54, np.nan, 1.0, 0.02, 0.0, HIST_FIXED)
    assert "0%, 25%, 50% and 100% invested" in " ".join(E.meter_explanation(s))
    flat = E.meter_from_inputs(settings("sized"), 0.6, 0.54, np.nan, np.nan, 0.02, 0.0, HIST_FIXED, position=0.4)
    assert "no usable sizing information" in " ".join(E.meter_explanation(flat)) and flat.zone == "Hold / buy"
    nohist = E.meter_from_inputs(settings("long_flat"), 0.54, 0.54, 0.01, 1.0, 0.02, 0.0, [])
    assert "Small print" not in " ".join(E.meter_explanation(nohist))


# --------------------------------------------------------------------------- the real data, four assets
def test_real_assets_needle_zone_equals_paper_position_zone():
    for t in TICKERS:
        r = E.today_meter(t, CFG)
        row = DD.ledger_for(t, CFG).iloc[-1]
        want = 0 if row["position"] <= 1e-12 else 3                      # the paper strategy is long/flat
        assert r.zone_index == want and E.zone_at(edges_of(r), r.needle) == want, t
        assert r.position == row["position"] and r.zones_used == ("Sell / step aside", "Hold / buy")
        assert r.axis_lo < r.needle < r.axis_hi and (not r.needle_clamped or abs(r.gap + r.margin_m) < 1e-9)


# --------------------------------------------------------------------------- the simplified Today card
@pytest.fixture(scope="module")
def today():
    st.cache_data.clear()
    at = AppTest.from_file(APP, default_timeout=300)
    at.session_state["page"] = "Today"
    return at.run()


def test_today_card_meter_tab_holds_only_the_meter_probability_position_badge_and_disclaimer(today):
    assert not today.exception
    cards = [today.tabs[3 * i:3 * i + 3] for i in range(len(TICKERS))]
    assert [t.label for c in cards for t in c] == ["Meter", "Why?", "Badge and terms"] * len(TICKERS)
    for t, (meter, why, terms) in zip(TICKERS, cards):
        md = [str(m.value) for m in meter.markdown]
        assert len(meter.get("plotly_chart")) == 1
        assert any(m.startswith("<span") and "Sell / step aside" in m for m in md)               # zone legend chips
        assert sum(bool(__import__("re").fullmatch(r"\*\*\d+\.\d% vs \d+\.\d% typical\*\*", m)) for m in md) == 1
        assert sum("Meter reads:" in m and "paper position:" in m for m in md) == 1
        assert sum("evidence badge" in m for m in md) == 1
        assert [str(w.value) for w in meter.warning if str(w.value) == E.DISCLAIMER] == [E.DISCLAIMER]
        assert len(meter.get("caption")) == 0                                                    # no explanation text left
        long_text = "\n".join(md)
        for gone in ("The dial's axis", "Models agree", "needle", "Decision for", "Why this evidence badge"):
            assert gone not in long_text, (t, gone)
        tm = "\n".join(str(m.value) for m in terms.markdown) + "\n".join(str(c.value) for c in terms.caption)
        for kept in ("How to read the dial", "Small print: today's gap", "Models agree:", "Decision for",
                     "Below the typical rate is not bullish", "Sell means step aside or reduce - never short."):
            assert kept in tm, (t, kept)
        assert any(e.label == "Why this evidence badge?" for e in today.expander)
        assert "1. What the model saw" in "\n".join(str(m.value) for m in why.markdown)
    assert sum(1 for e in today.expander if e.label == "What do these words mean?") == len(TICKERS)


def test_today_meter_needle_is_in_the_paper_position_zone_in_the_drawn_figure(today):
    figs = [json.loads(e.proto.spec) for e in today.get("plotly_chart")]
    gauges = [f for f in figs if f["data"][0]["type"] == "indicator"]
    assert len(gauges) == len(TICKERS)
    for t, g in zip(TICKERS, gauges):
        ind = g["data"][0]["gauge"]
        needle = g["data"][0]["value"]
        row = DD.ledger_for(t, CFG).iloc[-1]
        steps = [s["range"] for s in ind["steps"]]
        hold = steps[-1]                                                  # the last drawn arc is Hold / buy
        in_hold = hold[0] <= needle <= hold[1]
        assert in_hold == (row["position"] > 1e-12), (t, needle, steps, row["position"])
        assert len(steps) == 2                                            # only the zones the strategy really has
