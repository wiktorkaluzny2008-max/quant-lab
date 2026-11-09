"""Stage 5 logic layer: signal meter, evidence badge, models-agree, "Why?" text, theory.yaml."""
import hashlib
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from quant import explain as E
from quant import dash_data as DD
from quant import features as F
from quant import models as MD
from quant import paper as P
from quant import trials as T
from quant.config import PROJECT_ROOT, load_config
from tests import dash_world as W
from tests.synth import TICKERS, make_cfg, make_results

REAL = load_config()
HAVE_REAL = (REAL["data_dir"] / "paper" / "ledger.parquet").exists() and (
    REAL["data_dir"] / "results" / "SPY" / "predictions.parquet").exists()
real_only = pytest.mark.skipif(not HAVE_REAL, reason="real data/results + data/paper not present")
REPORT = (PROJECT_ROOT / "RESEARCH_REPORT.md").read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    return W.build(tmp_path_factory.mktemp("dashworld"), extra_days=("2021-10-14",))


def settings_for(mode, vt=True):
    return P.PaperSettings(
        strategy_mode=mode,
        strategy={"window": "expanding", "model": "ensemble", "mode": mode, "vol_targeted": vt,
                  "m_setting": "adaptive"},
        sized_lambda=0.25, sized_band=0.1, capital={}, costs={})


# =========================================================================== percentile maths
def test_percentile_maths_known_values():
    h = [1.0, 2.0, 3.0, 4.0]
    assert E.percentile_below(h, 3.0) == 50.0           # 1, 2 are below 3
    assert E.percentile_above(h, 3.0) == 25.0           # only 4 is above 3
    assert E.percentile_mid(h, 3.0) == 50.0 + 12.5      # 50% below + half of the 25% equal
    assert E.percentile_below(h, 0.0) == 0.0 and E.percentile_below(h, 9.0) == 100.0
    assert E.percentile_mid(h, 9.0) == 100.0 and E.percentile_mid(h, 0.0) == 0.0
    assert E.percentile_mid([2.0, 2.0, 2.0], 2.0) == 50.0
    assert np.isnan(E.percentile_mid([], 1.0)) and np.isnan(E.percentile_below([np.nan], 1.0))


def test_percentile_is_monotone_and_matches_a_brute_force_rank():
    rng = np.random.default_rng(1)
    h = rng.normal(0, 0.01, 500)
    xs = np.linspace(-0.04, 0.04, 81)
    mids = [E.percentile_mid(h, x) for x in xs]
    assert all(b >= a for a, b in zip(mids, mids[1:]))
    x = 0.0123
    assert E.percentile_below(h, x) == pytest.approx(100.0 * sum(v < x for v in h) / len(h))


def test_ordinal():
    assert [E.ordinal(x) for x in (1, 2, 3, 4, 11, 12, 13, 21, 22, 23, 91)] == [
        "1st", "2nd", "3rd", "4th", "11th", "12th", "13th", "21st", "22nd", "23rd", "91st"]


# =========================================================================== meter vs position
HIST = np.random.default_rng(7).normal(0.0, 0.012, 1500)


@pytest.mark.parametrize("mode", ["long_flat", "default_long", "sized"])
@pytest.mark.parametrize("vt", [True, False])
def test_meter_zone_agrees_with_decide_position_over_a_probability_sweep(mode, vt):
    st = settings_for(mode, vt)
    base = 0.54
    n = 0
    for m in (0.0, 0.005, 0.02):
        for ratio in (0.0, 0.4, 1.0, 2.5):
            for prev in (0.0, 0.35, 1.0):
                for p in np.round(np.arange(0.30, 0.80, 0.0025), 6):
                    r = E.meter_from_inputs(st, p, base, m, ratio, 0.02, prev, HIST)
                    pos = P.decide_position(st, p, base, m, ratio, 0.02, prev)
                    n += 1
                    assert r.position == pos and r.position_matches_rule
                    # the two invariants
                    assert (r.zone_index == 0) == (pos <= 1e-12)
                    if pos >= 1 - 1e-9:
                        assert r.zone in ("Hold / buy", "Strong buy")
                    if mode != "sized" and pos > 1e-12:
                        assert r.zone == "Hold / buy"                    # binary strategies have no 'Strong buy'
                    # the needle (today's gap) is in the arc of the paper position; arcs are ordered
                    assert E.zone_at([x["lo"] for x in r.bounds] + [r.bounds[-1]["hi"]], r.needle) == r.zone_index
                    edges = [x["lo"] for x in r.bounds] + [r.bounds[-1]["hi"]]
                    assert edges[0] == r.axis_lo and edges[-1] == r.axis_hi
                    assert all(y >= x for x, y in zip(edges, edges[1:]))
    assert n == 7200


@pytest.mark.parametrize("mode", ["long_flat", "default_long"])
def test_binary_modes_zone_is_exactly_the_strategy_rule(mode):
    """No vol-targeting: position 1 iff p >= base - m; 'Sell' iff p < base - m (boundary included in hold)."""
    st = settings_for(mode, vt=False)
    base, m = 0.55, 0.01
    thr = base - m
    for p in (thr - 1e-6, thr, thr + 1e-6, 0.30, 0.9):
        r = E.meter_from_inputs(st, p, base, m, 1.0, 0.02, 0.0, HIST)
        assert (r.zone == "Sell / step aside") == (not (p >= thr))
        assert r.threshold_gap == -m


def test_vol_targeting_changes_the_size_not_the_zone():
    st = settings_for("long_flat", vt=True)
    hi = E.meter_from_inputs(st, 0.60, 0.54, 0.0, 2.5, 0.02, 0.0, HIST)     # calm: full size
    lo = E.meter_from_inputs(st, 0.60, 0.54, 0.0, 0.4, 0.02, 0.0, HIST)     # turbulent: 40% size
    assert hi.position == 1.0 and lo.position == pytest.approx(0.4)
    assert hi.zone == lo.zone == "Hold / buy"


def test_binary_arcs_come_from_the_strategy_threshold_and_move_with_m():
    st = settings_for("long_flat")
    a = E.meter_from_inputs(st, 0.54, 0.54, 0.0, 1.0, 0.02, 0.0, HIST)
    b = E.meter_from_inputs(st, 0.54, 0.54, 0.02, 1.0, 0.02, 0.0, HIST)
    assert a.bounds[0]["hi"] == pytest.approx(-0.0) and b.bounds[0]["hi"] == pytest.approx(-0.02)
    assert b.bounds[0]["hi"] < a.bounds[0]["hi"]
    for r in (a, b):                        # Reduce / Neutral / Strong buy have zero width for the binary modes
        assert not r.bounds[1]["reachable"] and not r.bounds[2]["reachable"] and not r.bounds[4]["reachable"]


def test_a_binary_strategy_never_shows_strong_buy_even_for_the_top_reading():
    st = settings_for("long_flat", vt=False)
    top = E.meter_from_inputs(st, 0.60, 0.54, 0.0, 1.0, 0.02, 0.0, HIST)      # gap +6 pts, top of history
    assert top.zone == "Hold / buy" and top.percentile > 90
    neg = np.random.default_rng(3).normal(-0.03, 0.005, 800)
    r = E.meter_from_inputs(st, 0.53, 0.54, 0.02, 1.0, 0.02, 0.0, neg)        # gap -1 pt, above ~97% of history
    assert r.percentile > 90 and r.gap < 0 and r.position == 1.0 and r.zone == "Hold / buy"
    assert "Hold does not mean bullish" in r.caption
    # the exact bug that motivated the redesign: low percentile, yet the strategy holds -> needle in the hold arc
    pos_hist = np.random.default_rng(5).normal(0.0, 0.01, 800)
    q = E.meter_from_inputs(st, 0.5305, 0.54, 0.02, 1.0, 0.02, 0.0, pos_hist)  # gap -0.95 pt, near the 17th percentile
    assert q.zone == "Hold / buy" and q.percentile < 25 and q.needle >= q.bounds[3]["lo"]


def test_sized_zones_follow_the_position_levels():
    st = settings_for("sized", vt=False)
    seen = set()
    for p in np.arange(0.50, 0.70, 0.001):
        r = E.meter_from_inputs(st, p, 0.54, np.nan, 1.0, 0.02, 0.0, HIST)
        pos = r.position
        want = 0 if pos <= 1e-12 else 1 if pos < 0.25 else 2 if pos < 0.5 else 3 if pos < 1 - 1e-9 else 4
        assert r.zone_index == want
        seen.add(r.zone)
    assert seen == set(E.ZONES)          # every zone is reachable in sized mode


def test_meter_without_history_still_returns_a_consistent_reading():
    r = E.meter_from_inputs(settings_for("long_flat"), 0.60, 0.54, 0.0, 1.0, 0.02, 0.0, np.array([]))
    assert r.history_n == 0 and np.isnan(r.percentile) and r.needle == 50.0 or r.zone_index in (3, 4)
    assert r.zone in ("Hold / buy", "Strong buy")


# =========================================================================== evidence badge table
GREEN = dict(pt_pvalue=0.01, pt_statistic=2.5, brier_skill=0.01, brier_skill_eval_base=0.008, sharpe=1.2, sharpe_vt=0.9,
             psr_excess_vt=0.99, dsr_excess_vt=0.97, dsr_global=0.96)


def badge(**kw):
    return E.badge_from_numbers(**{**GREEN, **kw})


def test_evidence_requires_every_condition():
    assert badge()["level"] == "evidence" and badge()["color"] == "green" and not badge()["faded"]
    assert badge()["label"] == "Evidence"
    breakers = {"PT p-value": dict(pt_pvalue=0.20), "PT sign": dict(pt_statistic=-2.5),
                "skill": dict(brier_skill=-0.001), "skill_eval": dict(brier_skill_eval_base=-0.001), "beats": dict(sharpe=0.8),
                "dsr_excess": dict(dsr_excess_vt=0.90), "dsr_global": dict(dsr_global=0.50)}
    for name, kw in breakers.items():
        assert badge(**kw)["level"] != "evidence", name


@pytest.mark.parametrize("kw,level", [
    (dict(), "evidence"),
    # amber: beats vol-targeted B&H and (PT or PSR of the excess)
    (dict(dsr_global=0.2), "weak"),
    (dict(dsr_excess_vt=0.5, dsr_global=0.2), "weak"),
    (dict(brier_skill=-0.01), "weak"),
    (dict(brier_skill_eval_base=-0.0028), "weak"),                              # drift-only skill (BTC): not green
    (dict(brier_skill=0.0059, brier_skill_eval_base=-0.0028), "weak"),
    (dict(pt_pvalue=0.5, pt_statistic=0.1, dsr_global=0.3), "weak"),            # PSR_X alone is enough
    (dict(pt_pvalue=0.5, pt_statistic=0.1, psr_excess_vt=0.5), "none"),         # nothing significant
    # grey whenever it does not beat vol-targeted buy-and-hold, however good the rest looks
    (dict(sharpe=0.9), "none"),                                                 # equal Sharpe is not "beats"
    (dict(sharpe=0.5), "none"),
    # boundaries: p must be strictly below 0.05, probabilities may equal 0.95
    (dict(pt_pvalue=0.05, psr_excess_vt=0.5, dsr_global=0.2), "none"),
    (dict(dsr_excess_vt=0.95, dsr_global=0.95), "evidence"),
])
def test_badge_rule_table(kw, level):
    assert badge(**kw)["level"] == level


@pytest.mark.parametrize("field", list(GREEN))
def test_missing_or_nan_number_never_helps(field):
    check = {"pt_pvalue": "PT", "pt_statistic": "PT", "brier_skill": "SKILL", "brier_skill_eval_base": "SKILL",
             "sharpe": "BEATS",
             "sharpe_vt": "BEATS", "psr_excess_vt": "PSR_X", "dsr_excess_vt": "DSR_X",
             "dsr_global": "DSR_G"}[field]
    for bad in (np.nan, None):
        b = badge(**{field: bad})
        assert not b["checks"][check]["passed"]         # an unknown number never passes its check
        if check != "PSR_X":                            # PSR_X is not part of the green rule
            assert b["level"] != "evidence"


def test_all_missing_is_grey_with_the_exact_note():
    b = E.badge_from_numbers(**{k: np.nan for k in GREEN})
    assert (b["level"], b["label"], b["color"]) == ("none", "No proven edge", "grey")
    assert b["faded"] and b["note"] == "This model hasn't shown a reliable edge — treat this as a learning tool."
    assert E.FADED_NOTE == b["note"]
    assert badge()["note"] is None


def test_rule_text_is_declared_in_code():
    low = E.BADGE_RULE.lower()
    assert "pesaran" in low and "deflated" in low and "vol-targeted buy-and-hold" in low
    assert "more conservative" in low and "evaluation window" in low and "psr_x" in low     # round-1 change declared
    assert badge()["rule"] == E.BADGE_RULE


# =========================================================================== badge from saved files
def write_results(cfg, ticker="SPY", model="logreg", mode="long_flat", vt=True, m="adaptive", *,
                  pt_p=0.01, pt_s=2.0, skill=0.01, skill_eval=0.008, sr=1.2, sr_vt=0.9, psr_x=0.99, dsr_x=0.97, dsr_g=0.96,
                  alias=None):
    """Fake saved files (tests the file lookups, not the maths)."""
    rd = cfg["data_dir"] / "results" / ticker
    make_results(cfg, models=(model,), config_modes=("long_flat", "default_long"))
    cid = T.make_config_id(ticker, "expanding", model, mode, vt, m)
    pd.DataFrame([dict(window="expanding", model=model, pt_pvalue=pt_p, pt_statistic=pt_s,
                       brier_skill=skill, brier_skill_vs_eval_base=skill_eval),
                  dict(window="expanding", model="rf", pt_pvalue=0.9, pt_statistic=-1.0, brier_skill=-1.0,
                       brier_skill_vs_eval_base=-1.0)]
                 ).to_parquet(rd / "prediction_metrics.parquet")
    tm = [dict(name=cid, kind="trial", window="expanding", scenario="headline", sharpe=sr),
          dict(name=cid, kind="trial", window="expanding", scenario="low_cost", sharpe=99.0),
          dict(name="vol_targeted_bh", kind="baseline", window="expanding", scenario="headline", sharpe=sr_vt),
          dict(name="buy_and_hold", kind="baseline", window="expanding", scenario="headline", sharpe=77.0)]
    pd.DataFrame(tm).to_parquet(rd / "trading_metrics.parquet")
    sig = [dict(test="psr_excess_vt", window="expanding", subject=cid, reference="SR*=0", value=psr_x),
           dict(test="dsr_excess_vt", window="expanding", subject=cid, reference="asset", value=dsr_x),
           dict(test="dsr", window="expanding", subject=cid, reference="asset", value=0.123),
           dict(test="dsr", window="expanding", subject=cid, reference="global", value=dsr_g),
           dict(test="dsr_excess_bh", window="expanding", subject=cid, reference="asset", value=0.0)]
    pd.DataFrame(sig).to_parquet(rd / "significance.parquet")
    return cid


def lookup_cfg(tmp_path, **kw):
    cfg = make_cfg(tmp_path)
    cfg["leverage_strategy"]["model"] = kw.pop("model", "logreg")
    write_results(cfg, **kw)
    return cfg


def test_evidence_badge_reads_the_right_saved_rows(tmp_path):
    b = E.evidence_badge("SPY", lookup_cfg(tmp_path))
    assert b["level"] == "evidence" and b["color"] == "green"
    i = b["inputs"]
    assert (i["pt_pvalue"], i["pt_statistic"], i["brier_skill"], i["brier_skill_eval_base"]) == (0.01, 2.0, 0.01, 0.008)
    assert (i["sharpe"], i["sharpe_vt"]) == (1.2, 0.9)                  # headline scenario, vol-targeted baseline
    assert (i["psr_excess_vt"], i["dsr_excess_vt"], i["dsr_global"]) == (0.99, 0.97, 0.96)   # global DSR, not 0.123


def test_evidence_badge_is_grey_when_it_does_not_beat_vol_targeted_buy_and_hold(tmp_path):
    b = E.evidence_badge("SPY", lookup_cfg(tmp_path, sr=0.8))
    assert b["level"] == "none" and b["faded"] and b["failed_checks"] == ["BEATS"]


def test_evidence_badge_falls_back_to_grey_without_files(tmp_path):
    cfg = make_cfg(tmp_path)
    make_results(cfg)                                    # predictions only: no significance / metrics
    b = E.evidence_badge("SPY", cfg)
    assert b["level"] == "none" and b["faded"] and b["note"] == E.FADED_NOTE
    b = E.evidence_badge("SPY", make_cfg(tmp_path / "empty"))
    assert b["level"] == "none" and "unavailable" in b["reasons"][0]


def test_evidence_badge_for_an_alias_config_and_other_models(tmp_path):
    cfg = lookup_cfg(tmp_path)
    assert E.evidence_badge("SPY", cfg, model="rf")["level"] == "none"   # no logged rf trial -> grey, no crash


@real_only
def test_real_badges_follow_the_rule_applied_independently():
    """Re-derive each real asset's badge straight from the parquet files and compare."""
    cfg = REAL
    ps = DD.paper_settings(cfg)
    idx = pd.read_parquet(cfg["data_dir"] / "results" / "trials_index.parquet")
    for t in TICKERS:
        rd = cfg["data_dir"] / "results" / t
        cid = ps.config_id(t)
        canon = cid if cid in set(idx.config_id) else idx.loc[idx.aliases.str.contains(cid, regex=False), "config_id"].iloc[0]
        sig = pd.read_parquet(rd / "significance.parquet")
        sig = sig[(sig.subject == canon) & (sig.window == "expanding")]
        pm = pd.read_parquet(rd / "prediction_metrics.parquet")
        pm = pm[(pm.window == "expanding") & (pm.model == "ensemble")].iloc[0]
        tm = pd.read_parquet(rd / "trading_metrics.parquet")
        tm = tm[(tm.window == "expanding") & (tm.scenario == "headline")]
        sr = tm[tm.name == cid].iloc[0].sharpe
        sv = tm[tm.name == "vol_targeted_bh"].iloc[0].sharpe
        v = lambda test, ref=None: sig[(sig.test == test) & ((sig.reference == ref) if ref else True)].value.iloc[0]
        pt = pm.pt_pvalue < 0.05 and pm.pt_statistic > 0
        sk = min(pm.brier_skill, pm.brier_skill_vs_eval_base) > 0
        beats = sr > sv
        dx, dg, px = v("dsr_excess_vt") >= 0.95, v("dsr", "global") >= 0.95, v("psr_excess_vt") >= 0.95
        want = "evidence" if (pt and sk and beats and dx and dg) else "weak" if beats and (pt or px) else "none"
        got = E.evidence_badge(t, cfg)
        assert got["level"] == want, t
        assert got["color"] == {"none": "grey", "weak": "amber", "evidence": "green"}[want]
        assert got["inputs"]["dsr_global"] == pytest.approx(v("dsr", "global"))


# =========================================================================== models agree
class Stub:
    def __init__(self, p):
        self.p = p

    def predict_up(self, X):
        return np.array([self.p])


class StubEnsemble:
    def __init__(self, **ps):
        self.members = {k: Stub(v) for k, v in ps.items()}


@pytest.mark.parametrize("ps,n,direction,unanimous", [
    (dict(logreg=0.56, rf=0.57, lgbm=0.60), 3, "above", True),
    (dict(logreg=0.50, rf=0.51, lgbm=0.52), 3, "below", True),
    (dict(logreg=0.56, rf=0.57, lgbm=0.50), 2, "above", False),
    (dict(logreg=0.56, rf=0.50, lgbm=0.50), 2, "below", False),
    (dict(logreg=0.55, rf=0.55, lgbm=0.549), 2, "above", False),        # p == base counts as "above"
])
def test_models_agree_counts(ps, n, direction, unanimous):
    a = E.models_agree(StubEnsemble(**ps), pd.DataFrame({"x": [0.0]}), 0.55, 0.01)
    assert (a["n_agree"], a["direction"], a["unanimous"]) == (n, direction, unanimous)
    assert a["n_models"] == 3 and a["n_above"] + a["n_below"] == 3
    assert a["ensemble_p"] == pytest.approx(np.mean(list(ps.values())))
    assert f"{n} of 3 models lean {direction}" in a["text"]


def test_models_agree_on_the_saved_live_ensemble_is_deterministic_and_matches_the_ledger(world):
    cfg = world
    for t in TICKERS:
        ds = W.dataset(cfg, t)
        model = E.load_live_model_ex(t, cfg)[0]
        row = DD.ledger_for(t, cfg).iloc[-1]
        x = ds[F.feature_columns(ds)].loc[[row["date"]]]
        a1 = E.models_agree(model, x, row["base_rate"], row["margin_m"])
        a2 = E.models_agree(model, x, row["base_rate"], row["margin_m"])
        assert a1 == a2
        assert a1["ensemble_p"] == pytest.approx(row["p_up"], abs=1e-9)      # members reproduce the ledger probability
        assert set(a1["probabilities"]) == set(MD.MODEL_NAMES)


# =========================================================================== why: contributions
def test_feature_contributions_reproduce_the_member_scores(world):
    cfg = world
    ds = W.dataset(cfg, "SPY")
    model = E.load_live_model_ex("SPY", cfg)[0]
    d = DD.ledger_for("SPY", cfg)["date"].iloc[-1]
    x = ds[F.feature_columns(ds)].loc[[d]]
    c = E.feature_contributions(model, x)
    assert list(c["feature"]) == list(x.columns) and c["group"].notna().all()
    lr = model.members["logreg"]
    a = lr.calibrator.coef_[0][0]
    raw = lr.predict_raw(x)[0]
    clf = lr.base_model.named_steps["clf"]
    # intercept + sum(coef*z) == raw logit  ->  effects (x Platt slope) sum to a*(logit - intercept)
    assert c["logreg"].sum() == pytest.approx(a * (np.log(raw / (1 - raw)) - clf.intercept_[0]), abs=1e-6)
    lg = model.members["lgbm"]
    a2 = lg.calibrator.coef_[0][0]
    margin = float(np.asarray(lg.base_model.predict(x[list(lg.base_model.feature_name_)], pred_contrib=True))[0].sum())
    rawl = lg.predict_raw(x)[0]
    assert margin == pytest.approx(np.log(rawl / (1 - rawl)), abs=1e-6)       # SHAP values + bias == raw logit
    bias = float(np.asarray(lg.base_model.predict(x[list(lg.base_model.feature_name_)], pred_contrib=True))[0][-1])
    assert c["lgbm"].sum() == pytest.approx(a2 * (margin - bias), abs=1e-6)
    assert np.allclose(c["combined"], (c["logreg"] + c["lgbm"]) / 2)


def test_every_feature_has_a_sentence_template():
    for g, cols in F.feature_groups.items():
        for col in cols:
            for t in ("SPY", "BTC-USD"):
                desc, kind = E._spec(col, t)
                assert not desc.startswith("the feature '"), col
                assert kind != "num", col


# =========================================================================== why: numbers
NUM = re.compile(r"[-+]?\d+(?:\.\d+)?")


def fmt(kind, x):
    return {"pct": lambda v: f"{v * 100:+.1f}", "vol": lambda v: f"{v * 100:.1f}", "ratio": lambda v: f"{v:.2f}",
            "level": lambda v: f"{v:.1f}", "rsi": lambda v: f"{v:.0f}", "corr": lambda v: f"{v:+.2f}",
            "var": lambda v: f"{v:+.3f}"}[kind](x)


def test_every_number_in_a_why_sentence_equals_an_independently_computed_feature_value(world):
    cfg = world
    cutoff = pd.Timestamp(cfg["holdout_cutoff_date"])
    checked = 0
    for t in TICKERS:
        ds = W.dataset(cfg, t)
        rep = E.today_report(t, cfg, ds=ds)
        why, d = rep["why"], pd.Timestamp(rep["date"])
        assert d in ds.index and len(why.saw) >= 3
        for item in why.saw:
            col = item["feature"]
            desc, kind = E._spec(col, t)
            v = float(ds.loc[d, col])                                       # independent value
            h = ds.loc[(ds.index < d) & (ds.index <= cutoff), col].dropna().to_numpy()
            f = item["facts"]
            assert f["value"] == v and f["n_history"] == len(h)
            below, above = 100 * np.mean(h < v), 100 * np.mean(h > v)
            assert f["below"] == pytest.approx(below) and f["above"] == pytest.approx(above)
            assert f["q25"] == pytest.approx(np.quantile(h, 0.25)) and f["q75"] == pytest.approx(np.quantile(h, 0.75))
            year = str(ds.loc[ds.index <= cutoff, col].dropna().index[0].year)
            assert f["history_start"] == year
            end_year = str(ds.loc[(ds.index < d) & (ds.index <= cutoff), col].dropna().index[-1].year)
            assert f["history_end"] == end_year
            body = item["sentence"].replace(desc[0].upper() + desc[1:], "").replace(
                f"(signal group: {item['group_title']})", "")
            tokens = [x.replace("+", "") for x in NUM.findall(body)]
            if kind in ("flag", "dow"):
                share = 100 * np.mean(h == v)
                allowed = {f"{share:.0f}", f"{100 - share:.0f}", year, end_year}
            else:
                allowed = {fmt(kind, v).replace("+", ""), fmt(kind, np.quantile(h, 0.25)).replace("+", ""),
                           fmt(kind, np.quantile(h, 0.75)).replace("+", ""), year, end_year,
                           f"{(below if above <= below else above):.0f}"}
            allowed.add(f"{abs(item['points']):.2f}")
            assert {x.replace("+", "") for x in tokens} <= {x.replace("+", "") for x in allowed}, (t, col, body)
            if kind not in ("flag", "dow"):
                assert fmt(kind, v).replace("+", "") in [x.replace("+", "") for x in tokens]
            checked += 1
    assert checked >= 12


def test_trust_numbers_match_the_saved_out_of_sample_predictions(world):
    cfg = world
    for t in TICKERS:
        rep = E.today_report(t, cfg, ds=W.dataset(cfg, t))
        tb = rep["why"].trust["bucket"]
        pr = pd.read_parquet(cfg["data_dir"] / "results" / t / "predictions.parquet")
        pr = pr[pr.model == "ensemble"]
        conf = (pr.p_up - pr.base_rate).abs()
        inside = pr[(conf >= tb["conf_lo"] - 1e-12) & (conf <= tb["conf_hi"] + 1e-12)]
        hit = ((inside.p_up >= inside.base_rate) == (inside.y > 0.5)).mean()
        assert tb["n"] == len(inside) and tb["hit_rate"] == pytest.approx(hit)
        assert tb["overall_hit_rate"] == pytest.approx(((pr.p_up >= pr.base_rate) == (pr.y > 0.5)).mean())
        assert tb["always_up_rate"] == pytest.approx((pr.y > 0.5).mean())
        txt = rep["why"].trust["text"]
        assert f"{tb['hit_rate'] * 100:.1f}%" in txt and f"over {tb['n']} days" in txt


def test_why_has_the_four_parts_theory_links_and_the_meter_zone(world):
    cfg = world
    th = E.load_theory()
    for t in TICKERS:
        rep = E.today_report(t, cfg, ds=W.dataset(cfg, t))
        why, m = rep["why"], rep["meter"]
        assert 3 <= len(why.saw) <= 4 and len(why.theory) == len(why.saw)
        for item, tt in zip(why.saw, why.theory):
            assert tt["group"] == item["group"]
            assert tt["text"] == " ".join(th["feature_groups"][item["group"]]["short"].split())
            src = th["feature_groups"][item["group"]]["sources"][0]
            assert tt["url"] == src["url"] and src["url"] in tt["source"]
        assert m.zone in why.conclusion and m.probability_text in why.conclusion
        assert f"{m.position * 100:.0f}%" in why.conclusion
        assert "never claim" not in why.text and "not why the market will move" in why.text
        assert why.text.count("1. What the model saw") == 1 and "4. The conclusion" in why.text
        assert "short" not in why.conclusion.lower() or "not a short" in why.conclusion.lower()


# =========================================================================== no lookahead
def test_why_ignores_everything_after_the_decision_date(world):
    cfg = world
    t = "SPY"
    ds = W.dataset(cfg, t)
    model = E.load_live_model_ex(t, cfg)[0]
    settings = DD.paper_settings(cfg)
    res = DD.load_asset_results(t, cfg)
    preds = DD.model_predictions(res, "ensemble")
    cutoff = pd.Timestamp(cfg["holdout_cutoff_date"])
    d = ds.index[-10]                                         # a decision date well before the end
    meter = E.meter_from_inputs(settings, 0.55, 0.54, 0.005, 1.0, 0.02, 0.0, HIST)
    badge_ = E.evidence_badge(t, cfg)
    base = E.explain_today(t, ds[ds.index <= d], d, model, meter, preds, badge_, cutoff)
    # same dataset with the future present and then replaced by garbage
    junk = ds.copy()
    later = junk.index > d
    rng = np.random.default_rng(0)
    junk.loc[later, F.feature_columns(junk)] = rng.normal(100.0, 50.0, (later.sum(), len(F.feature_columns(junk))))
    junk.loc[later, ["target", "next_return"]] = 9.9
    extra = junk.tail(5).copy()
    extra.index = pd.date_range(ds.index[-1] + pd.Timedelta(days=1), periods=5)
    junk = pd.concat([junk, extra])
    for variant in (ds, junk):
        w = E.explain_today(t, variant, d, model, meter, preds, badge_, cutoff)
        assert w.text == base.text
        assert [s["facts"] for s in w.saw] == [s["facts"] for s in base.saw]


def test_history_before_the_cutoff_only_uses_rows_before_the_decision_date(world):
    cfg = world
    ds = W.dataset(cfg, "SPY")
    cutoff = pd.Timestamp(cfg["holdout_cutoff_date"])
    d = ds.index[ds.index <= cutoff][-5]                       # decision date inside the dev period
    f1 = E.feature_facts(ds, "r5", d, cutoff)
    ds2 = ds.copy()
    ds2.loc[(ds2.index > d), "r5"] = 123.0                      # rows after d (still <= cutoff) must not matter
    assert E.feature_facts(ds2, "r5", d, cutoff) == f1


def test_meter_history_ignores_ledger_rows_from_the_future(tmp_path):
    early = W.build(tmp_path / "a", asof="2021-10-14")
    late = W.build(tmp_path / "b", asof="2021-10-15", extra_days=("2021-10-14",))
    for t in TICKERS:
        day = DD.ledger_for(t, early)["date"].max()        # BTC's last complete day is the day before
        m1 = E.today_meter(t, early)
        m2 = E.today_meter(t, late, asof=day)
        assert vars(m1).keys() == vars(m2).keys()
        for k_, v_ in vars(m1).items():
            assert (v_ == vars(m2)[k_]) or (v_ != v_ and vars(m2)[k_] != vars(m2)[k_]), k_
        h1 = E.gap_history(t, early, asof=day)
        h2 = E.gap_history(t, late, asof=day)
        assert np.array_equal(h1, h2)
        h3 = E.gap_history(t, late, asof=DD.ledger_for(t, late)["date"].max())
        assert len(h3) == len(h1) + 1


# =========================================================================== today_report / hold-out / read-only
def test_today_report_always_carries_the_disclaimer_and_the_grey_note(world):
    cfg = world
    for t in TICKERS:
        rep = E.today_report(t, cfg, ds=W.dataset(cfg, t))
        assert rep["disclaimer"] == E.DISCLAIMER and "not financial advice" in rep["disclaimer"]
        assert rep["badge"]["level"] == "none" and rep["faded"]        # fake results have no significance files
        assert rep["faded_note"] == "This model hasn't shown a reliable edge — treat this as a learning tool."
        assert rep["meter"].sell_means.startswith("Sell means step aside")


def test_meter_agrees_with_the_ledger_position_in_the_synthetic_world(world):
    cfg = world
    for t in TICKERS:
        m = E.today_meter(t, cfg)
        pos = DD.ledger_for(t, cfg).iloc[-1]["position"]
        assert m.position == pos and m.position_matches_rule
        assert (m.zone == "Sell / step aside") == (pos <= 1e-12)


def digest(root: Path):
    return {str(p.relative_to(root)): (hashlib.sha256(p.read_bytes()).hexdigest(), p.stat().st_mtime_ns)
            for p in sorted(root.rglob("*")) if p.is_file()}


def test_explain_is_read_only_in_the_synthetic_world(world):
    cfg = world
    before = digest(cfg["data_dir"])
    for t in TICKERS:
        E.today_report(t, cfg, ds=W.dataset(cfg, t))
    DD.data_freshness(cfg)
    assert digest(cfg["data_dir"]) == before


@real_only
def test_real_today_reports_are_read_only_and_agree_with_the_ledger():
    root = REAL["data_dir"] / "paper"
    before = digest(root)
    for t in TICKERS:
        rep = E.today_report(t, REAL)
        m, row = rep["meter"], DD.ledger_for(t, REAL).iloc[-1]
        assert m.position == row["position"] and m.position_matches_rule
        assert (m.zone == "Sell / step aside") == (row["position"] <= 1e-12)
        if row["position"] >= 1 - 1e-9:
            assert m.zone in ("Hold / buy", "Strong buy")
        assert rep["agree"]["matches_ledger"] and rep["agree"]["n_models"] == 3
        assert rep["disclaimer"] == E.DISCLAIMER
        assert pd.Timestamp(rep["date"]) == row["date"]
        # the 'Why?' numbers are computed for the decision day, from development-period history only
        for s in rep["why"].saw:
            assert s["facts"]["history_start"] is not None
    assert digest(root) == before                      # idempotent AND nothing written (bytes + mtimes)


@real_only
def test_real_gap_history_contains_no_holdout_dates():
    cut = pd.Timestamp(REAL["holdout_cutoff_date"])
    for t in TICKERS:
        res = DD.load_asset_results(t, REAL)
        pr = DD.model_predictions(res, "ensemble")
        assert pr["date"].max() <= cut
        led = DD.ledger_for(t, REAL)
        assert led["date"].min() >= P.holdout_end(REAL)     # live ledger starts on/after the hold-out end
        h = E.gap_history(t, REAL, asof=led["date"].iloc[-1])
        assert len(h) == len(pr) + (len(led) - 1)


# =========================================================================== theory.yaml
TH = E.load_theory()
ENTRY_KEYS = {"title", "plain_english", "formula", "evidence", "caveats", "sources", "report_section"}
URLS = set(re.findall(r"https?://[^\s)>\]\"']+", REPORT))


def all_entries():
    for block in ("feature_groups", "models", "reality_check"):
        for name, e in TH[block].items():
            yield block, name, e


def test_every_feature_group_has_a_theory_entry():
    assert set(TH["feature_groups"]) == set(F.feature_groups)
    for g, e in TH["feature_groups"].items():
        assert ENTRY_KEYS <= set(e), g
        assert e["short"].strip() and e["short_name"].strip() and e["plain_english"].strip(), g
        assert e["evidence"].strip() and e["caveats"].strip() and e["formula"].strip(), g


def test_how_it_works_and_reality_check_entries_exist():
    assert {"logistic_regression", "random_forest", "lightgbm", "ensemble", "walk_forward", "calibration",
            "climatology_baseline", "brier_skill", "signal_meter", "evidence_badge"} <= set(TH["models"])
    assert {"realistic_auc", "cross_sectional_vs_index", "edges_negative_after_costs",
            "post_publication_decay", "holdout_power", "overfitting", "significance_tests"} <= set(TH["reality_check"])


def test_every_entry_has_the_schema_and_sources_with_https_urls():
    n = 0
    for block, name, e in all_entries():
        assert ENTRY_KEYS <= set(e), (block, name)
        assert e["plain_english"].strip() and e["title"].strip() and e["report_section"].strip()
        assert isinstance(e["sources"], list)
        if not e["sources"]:
            assert e.get("source_note"), f"{block}.{name}: no sources, so it must say so in source_note"
        for s in e["sources"]:
            assert s["title"].strip() and s["authors"].strip() and "year" in s, (block, name)
            assert s["url"].startswith("https://"), (block, name, s["url"])
            n += 1
    assert n > 40


def test_every_source_url_appears_in_the_research_report():
    for block, name, e in all_entries():
        for s in e["sources"]:
            assert s["url"] in REPORT, f"{block}.{name}: {s['url']} is not in RESEARCH_REPORT.md"
            assert s["url"] in URLS or s["url"].rstrip("/") in REPORT


def test_report_section_names_a_real_heading_of_the_report():
    heads = [h.strip("# ").strip() for h in re.findall(r"(?m)^#{1,4} .*$", REPORT)]
    for block, name, e in all_entries():
        first = re.split(r" > |; ", e["report_section"])[0]
        assert any(first.lower().replace("-", " ") in h.lower().replace("-", " ").replace("–", " ") or
                   h.lower().replace("-", " ").startswith(first.lower().replace("-", " ")[:25]) for h in heads), \
            (block, name, first)


@pytest.mark.parametrize("entry,phrases", [
    (("reality_check", "realistic_auc"), ["51-56%", "0.51-0.55", "0.51-0.53"]),
    (("reality_check", "cross_sectional_vs_index"), ["0.45%", "54.3%", "5.8"]),
    (("reality_check", "edges_negative_after_costs"), ["6.7", "1992-2001", "2010-2015", "39%", "30 bp"]),
    (("reality_check", "post_publication_decay"), ["97", "26%", "58%", "7,846", "0.341", "0.908"]),
    (("reality_check", "holdout_power"), ["252 days", "0.71", "0.45", "3.1 percentage points", "11 years"]),
    (("reality_check", "overfitting"), ["0.99", "100 trials", "0.67", "10.6%", "96.3%"]),
    (("reality_check", "significance_tests"), ["t > 3.0", "0.6", "0.87"]),
    (("reality_check", "leverage_kelly"), ["4.8", "75%"]),
    (("feature_groups", "reversal"), ["16 of 20", "0.67", "+0.05"]),
    (("feature_groups", "trend"), ["3.77", "1880", "67 markets"]),
    (("feature_groups", "volatility"), ["4.86%", "0.42", "0.51", "103"]),
    (("feature_groups", "calendar"), ["0.64", "50 bp", "1994-2011"]),
    (("feature_groups", "vix_stress"), ["1.5-4%", "2010-2017"]),
    (("feature_groups", "cross_asset"), ["0.01", "0.36", "2017-19", "2020-21"]),
    (("models", "logistic_regression"), ["51.99%", "50.51%"]),
    (("models", "random_forest"), ["1.90", "50.51%"]),
    (("models", "ensemble"), ["highest directional accuracy"]),
])
def test_numbers_quoted_in_theory_appear_in_the_report(entry, phrases):
    e = TH[entry[0]][entry[1]]
    text = " ".join(str(v) for v in e.values() if isinstance(v, str))
    flat = REPORT.replace("−", "-").replace("–", "-")
    for ph in phrases:
        assert ph in text.replace("−", "-"), f"{entry}: '{ph}' missing from theory entry"
        assert ph in flat, f"'{ph}' is not in RESEARCH_REPORT.md"


def test_theory_does_not_name_a_paper_without_a_link_or_a_note():
    """Every 'Author Year' named in an entry's prose must have its URL among the entry's sources, or
    the entry must say there is none (no invented citations)."""
    for block, name, e in all_entries():
        titles = " ".join(s["title"] for s in e["sources"])
        for m in re.finditer(r"\(([A-Z][\w&.,' -]+ (?:et al\. )?\d{4})\)", e["plain_english"] + e["short"] if "short" in e else e["plain_english"]):
            who = m.group(1).split()[0].rstrip(",")
            assert who in titles or e.get("source_note"), (block, name, m.group(1))
