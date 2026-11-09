"""Stage 5 logic layer: signal meter, evidence badge, "models agree" count and "Why?" text.

Templates only -- no external AI service, no API key, no randomness.  Everything is deterministic
and READ-ONLY (nothing under data/ is ever written by this module).

The UI builder calls :func:`today_report` (one dict per asset) or the pieces below:

* :func:`meter_from_inputs` / :func:`today_meter`      the five-zone signal meter
* :func:`evidence_badge`                               grey / amber / green badge from saved results
* :func:`models_agree`                                 how many of logreg / rf / lgbm lean the same way
* :func:`explain_today`                                the four-part "Why?" explanation
* :func:`load_theory`                                  quant/theory.yaml

==========================================================================================
THE METER  (declared before any result was looked at; see PROGRESS.md for the reasons)
==========================================================================================
gap_t   = calibrated p_t - base_rate_t              (positive = more bullish than a typical day)
needle  = percentile of gap_t among the gaps of the model's past out-of-sample history
          (Stage 3 walk-forward predictions, development period only, then the live ledger rows
          before today).  Mid-rank percentile: 100*(share below + half the share equal).

The zone is a function of the position the ACTIVE STRATEGY takes (``paper.decide_position``), so
the meter can never disagree with the paper-trading account:

  long_flat / default_long   (the same rule, "long iff p >= base_rate - m", with different m grids):
      position == 0                -> "Sell / step aside"   (gap < -m)
      position  > 0 (signal on)    -> "Hold / buy"          (gap >= -m)
      "Reduce", "Neutral" and "Strong buy" exist on the dial but have ZERO width for these two modes:
      a binary strategy has no partial-conviction state and no stronger "buy" than holding, and we do
      not invent a trading action the strategy does not have.  Vol-targeting scales the SIZE of a held
      position (position = w <= 1) but never changes the zone (signal on is signal on).
  sized  (position = clip(lambda*(p-base)/scale * sigma_target/sigma_hat, 0, 1), no-trade band):
      position == 0 -> Sell / step aside;   0 < position < 0.25 -> Reduce;   0.25 <= position < 0.5
      -> Neutral;   0.5 <= position < 1 -> Hold / buy;   position == 1 (fully invested) -> Strong buy

THE DIAL IS DRAWN IN GAP SPACE (gap = p - base_rate, in fractions; the page shows percentage points).
Its zone edges are the strategy's real thresholds on that axis: gap = -m for the binary modes; the gaps
where the sized target reaches 0, 0.25, 0.5 and 1 for the sized mode.  The needle is TODAY'S GAP on the
same axis, so it falls in the arc of the paper position by construction (it is only moved when the
no-trade band keeps an older position: ``needle_clamped``).  The percentile of today's gap among past
out-of-sample gaps is kept as a small secondary number (``percentile``): it says how unusual the day is,
not what the strategy does.

THE INVARIANT (tested over thousands of random and boundary cases):  ``zone_at(edges, needle)`` ==
zone of the position the strategy actually holds, for every mode, margin, history and input.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from quant import dash_data as DD
from quant import features as F
from quant import metrics as M
from quant import paper as P
from quant import trials as T

THEORY_PATH = Path(__file__).resolve().parent / "theory.yaml"

DISCLAIMER = ("Research and education only - not financial advice. Paper trading uses fake money; "
              "this project never connects to a broker or places real trades.")
FADED_NOTE = "This model hasn't shown a reliable edge — treat this as a learning tool."
ZONES = ("Sell / step aside", "Reduce", "Neutral", "Hold / buy", "Strong buy")
SIZED_CUTS = (0.25, 0.5, 1.0)      # sized-mode position levels separating Reduce/Neutral/Hold/Strong
ZERO = 1e-12                       # |position| below this is "no position" (paper.TRADE_TOL)
FULL = 1.0 - 1e-9                  # position at/above this is "fully invested"



class NoDecisionError(LookupError):
    """There is no paper-trading decision to explain yet (empty ledger, or no row for the date asked)."""


# --------------------------------------------------------------------------- theory
_THEORY_CACHE: dict = {}


def load_theory(path=None) -> dict:
    """Parsed quant/theory.yaml (blocks: feature_groups, models, reality_check)."""
    p = Path(path) if path else THEORY_PATH
    key = str(p)
    if key not in _THEORY_CACHE:
        _THEORY_CACHE[key] = yaml.safe_load(p.read_text(encoding="utf-8"))
    return _THEORY_CACHE[key]


def source_line(src: dict) -> str:
    """"Author Year: topic - URL" (the title already carries author and year); a link the check
    found dead or unverifiable carries its ``link_note`` ("(link no longer works)")."""
    note = f" {src['link_note']}" if src.get("link_note") else ""
    return f"{src['title']} - {src['url']}{note}"


def ordinal(x: float) -> str:
    n = int(round(x))
    suf = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suf}"


# =========================================================================== percentile maths
def percentile_below(hist, x) -> float:
    """Share (0-100) of ``hist`` strictly below ``x``.  NaN for an empty history."""
    h = np.asarray(hist, dtype=float)
    h = h[np.isfinite(h)]
    return float("nan") if h.size == 0 else float(100.0 * np.mean(h < x))


def percentile_above(hist, x) -> float:
    h = np.asarray(hist, dtype=float)
    h = h[np.isfinite(h)]
    return float("nan") if h.size == 0 else float(100.0 * np.mean(h > x))


def percentile_mid(hist, x) -> float:
    """Mid-rank percentile (0-100): share below plus half the share equal.  NaN if no history."""
    h = np.asarray(hist, dtype=float)
    h = h[np.isfinite(h)]
    if h.size == 0:
        return float("nan")
    return float(100.0 * (np.mean(h < x) + 0.5 * np.mean(h == x)))


# =========================================================================== the meter
@dataclass
class MeterReading:
    zone: str
    zone_index: int
    position: float
    p_up: float
    base_rate: float
    gap: float
    percentile: float            # SECONDARY: mid-rank percentile of today's gap in the past (NaN if no history)
    needle: float                # today's gap (fraction) on the dial axis, moved into its zone's arc if needed
    needle_clamped: bool
    bounds: list                 # [{"zone", "lo", "hi", "reachable"}], gap axis (fractions)
    axis_lo: float
    axis_hi: float
    strategy_mode: str
    margin_m: float | None
    threshold_gap: float | None  # binary modes: -m (position > 0 iff gap >= threshold_gap)
    history_n: int
    probability_text: str        # "52.1% vs 54.4% typical"
    position_matches_rule: bool | None = None   # recomputed decide_position == ledger position?
    caption: str = ""
    sell_means: str = "Sell means step aside or reduce - never short."
    zones_used: tuple = ()       # names of the zones that really exist for this strategy
    cuts_known: bool = True


def probability_text(p: float, base: float) -> str:
    return f"{p * 100:.1f}% vs {base * 100:.1f}% typical"


def classify_zone(strategy_mode: str, position: float, cuts_known: bool = True) -> int:
    """Zone index (0-4) of the position the strategy takes (see module docstring).  A binary strategy
    only has Sell (position 0) and Hold (position > 0); a non-finite position counts as 'no position'."""
    if not np.isfinite(position) or position <= ZERO:
        return 0
    if strategy_mode != "sized" or not cuts_known:
        return 3
    if position >= FULL:
        return 4
    if position < SIZED_CUTS[0]:
        return 1
    if position < SIZED_CUTS[1]:
        return 2
    return 3


def zone_at(edges, value: float) -> int:
    """Zone index of a point on the gap axis: arc i is [edges[i], edges[i+1]); zero-width arcs are never
    returned; beyond either end the nearest real arc is returned."""
    e = [float(x) for x in edges]
    real = [i for i in range(5) if e[i + 1] > e[i]]
    if not real:
        return 0
    if value < e[real[0]]:
        return real[0]
    for i in real:
        if e[i] <= value < e[i + 1]:
            return i
    return real[-1]


def _cuts_usable(settings: P.PaperSettings, ratio, scale) -> bool:
    k = ratio if settings.vol_targeted else 1.0
    return bool(k is not None and np.isfinite(k) and k > 0 and scale is not None and np.isfinite(scale) and scale > 0)


def zone_edges(settings: P.PaperSettings, hist, gap, m, ratio, scale) -> tuple:
    """(edges, axis_lo, axis_hi): the 6 zone edges on the GAP axis (fractions) and the dial range.

    edges[i]..edges[i+1] is zone i.  The axis spans the 1st-99th percentile of past gaps plus today's gap
    and every real threshold, with some padding, so the needle and all thresholds are always on the dial."""
    if settings.strategy_mode == "sized" and _cuts_usable(settings, ratio, scale):
        k = ratio if settings.vol_targeted else 1.0
        inner = [0.0] + [c * float(scale) / (settings.sized_lambda * float(k)) for c in SIZED_CUTS]
        kind = "sized"
    elif settings.strategy_mode == "sized":
        inner, kind = [0.0], "flat"                        # no sizing information: only the step-aside line
    else:
        t = -float(m) if m is not None and np.isfinite(m) else 0.0
        inner, kind = [t], "binary"
    h = np.asarray(hist, dtype=float)
    h = h[np.isfinite(h)]
    pts = list(inner) + ([float(gap)] if np.isfinite(gap) else [])
    pts += [float(np.percentile(h, 1)), float(np.percentile(h, 99))] if h.size else [-0.05, 0.05]
    lo, hi = min(pts), max(pts)
    pad = max(0.1 * (hi - lo), 0.005)
    lo, hi = lo - pad, hi + pad
    if kind == "sized":
        edges = [lo] + inner + [hi]
    else:
        edges = [lo, inner[0], inner[0], inner[0], hi, hi]
    edges = [float(e) for e in np.maximum.accumulate(edges)]
    return edges, edges[0], edges[-1]


def _place_needle(edges, zi: int, gap: float) -> float:
    """``gap`` moved into the arc of zone ``zi`` (half-open: the upper edge belongs to the next zone,
    except for the last real arc)."""
    lo, hi = edges[zi], edges[zi + 1]
    if not hi > lo:
        return lo
    last_real = not any(edges[j + 1] > edges[j] for j in range(zi + 1, 5))
    top = hi if last_real else float(np.nextafter(hi, -np.inf))
    return float(min(max(gap, lo), top))


def meter_from_inputs(settings: P.PaperSettings, p, base, m, ratio, scale, prev_pos, hist,
                      position=None) -> MeterReading:
    """The meter for one set of strategy inputs.

    ``position`` defaults to ``paper.decide_position(...)`` -- exactly what the paper account
    would hold -- so the zone agrees with the paper-trading position by construction.  Pass the
    ledger's recorded position to read the zone of what is actually held."""
    rule_pos = P.decide_position(settings, float(p), float(base), m, ratio, scale, prev_pos)
    pos = rule_pos if position is None else float(position)
    gap = float(p) - float(base)
    hist = np.asarray(hist, dtype=float)
    hist = hist[np.isfinite(hist)]
    pct = percentile_mid(hist, gap)
    usable = settings.strategy_mode != "sized" or _cuts_usable(settings, ratio, scale)
    zi = classify_zone(settings.strategy_mode, pos, usable)
    edges, a_lo, a_hi = zone_edges(settings, hist, gap, m, ratio, scale)
    needle = _place_needle(edges, zi, gap)
    binary = settings.strategy_mode != "sized"
    mm = float(m) if binary and m is not None and np.isfinite(m) else None
    bounds = [{"zone": ZONES[i], "lo": edges[i], "hi": edges[i + 1], "reachable": edges[i + 1] > edges[i]}
              for i in range(5)]
    return MeterReading(
        zone=ZONES[zi], zone_index=zi, position=pos, p_up=float(p), base_rate=float(base), gap=gap,
        percentile=pct, needle=float(needle), needle_clamped=bool(abs(needle - gap) > 1e-12),
        bounds=bounds, axis_lo=a_lo, axis_hi=a_hi, strategy_mode=settings.strategy_mode,
        margin_m=mm, threshold_gap=(-mm if mm is not None else None), history_n=int(len(hist)),
        probability_text=probability_text(float(p), float(base)),
        position_matches_rule=bool(abs(pos - rule_pos) < 1e-9),
        caption=_caption(ZONES[zi], gap, mm, settings.strategy_mode),
        zones_used=tuple(b["zone"] for b in bounds if b["reachable"]), cuts_known=usable)


def _caption(zone: str, gap: float, m, mode: str) -> str:
    """One plain sentence under the meter explaining the zone in terms of the strategy's rule."""
    if mode == "sized":
        return (f"Sized mode: the position grows with how far the probability is above the base rate; "
                f"today's gap is {gap * 100:+.1f} percentage points.")
    if m is None:
        return ""
    if zone == "Sell / step aside":
        return (f"The probability is {abs(gap) * 100:.1f} points below the base rate, beyond the strategy's "
                f"{m * 100:.1f}-point tolerance, so it steps aside (cash, not a short).")
    if gap < 0:
        return (f"The probability is {abs(gap) * 100:.1f} points below the base rate, but inside the "
                f"strategy's {m * 100:.1f}-point tolerance, so it keeps holding. Hold does not mean bullish.")
    return f"The probability is {gap * 100:.1f} points above the base rate, so the strategy holds."


def meter_explanation(m: MeterReading) -> list:
    """Plain-English bullet points explaining how to read ONE dial (shown on the 'Badge and terms' tab)."""
    out = ["The dial's axis is today's gap: the model's probability of an up day minus how often the asset "
           "usually rises, in percentage points (0 = exactly typical). The needle is today's gap, so it is "
           "always inside the zone of the paper-trading position."]
    if m.strategy_mode == "sized":
        if m.cuts_known:
            out.append("The zone edges are where this 'sized' strategy's target position reaches 0%, 25%, 50% and "
                       "100% invested, so Reduce, Neutral and Strong buy (fully invested) are all real zones here.")
        else:
            out.append("The 'sized' strategy has no usable sizing information today, so only the step-aside line "
                       "(gap 0) is drawn and any held position shows as Hold / buy.")
    else:
        t = m.threshold_gap
        edge = "the 'step aside' line" if t is None else f"the 'step aside' line at {t * 100:+.1f} points"
        out.append(f"This strategy has only two states: it holds unless the gap falls below {edge} (its tolerance "
                   "margin m, which adapts using past data only), and then it steps aside into cash. So Reduce, "
                   "Neutral and Strong buy are crossed out: the strategy has no such action, and none is invented "
                   "for the dial. Hold / buy means 'the strategy is holding', not 'bullish'.")
    if np.isfinite(m.percentile):
        out.append(f"Small print: today's gap is about the {ordinal(m.percentile)} percentile of this model's past "
                   "days (higher = more positive than usual). It tells you how unusual the day is; it does not "
                   "decide the zone.")
    if m.needle_clamped:
        out.append("The paper account kept an earlier position because of its no-trade band, so the needle is "
                   "drawn at the edge of that position's zone instead of at the exact gap.")
    return out


def gap_history(ticker, cfg=None, asof=None, model=None, window="expanding") -> np.ndarray:
    """Past out-of-sample gaps ``p_up - base_rate`` of the active model for ``ticker``: the Stage 3
    walk-forward predictions (development period only; the loader refuses hold-out dates) followed
    by the live ledger rows dated strictly before ``asof``.  Nothing after ``asof`` is used."""
    cfg = DD._cfg(cfg)
    model = model or DD.paper_settings(cfg).strategy["model"]
    res = DD.load_asset_results(ticker, cfg)
    pr = DD.model_predictions(res, model, window)
    gaps = [(pr["p_up"] - pr["base_rate"]).to_numpy(float)]
    led = DD.ledger_for(ticker, cfg)
    if asof is not None:
        led = led[led["date"] < pd.Timestamp(asof)]
    if len(led):
        gaps.append((led["p_up"] - led["base_rate"]).to_numpy(float))
    return np.concatenate(gaps)


def today_meter(ticker, cfg=None, asof=None) -> MeterReading:
    """The meter for the latest ledger row (or the row dated ``asof``) of ``ticker``."""
    cfg = DD._cfg(cfg)
    settings = DD.paper_settings(cfg)
    led = DD.ledger_for(ticker, cfg)
    if led.empty:
        raise NoDecisionError(f"{ticker}: the paper-trading ledger has no row yet (run the daily update)")
    row = _ledger_row(led, asof)
    st = asset_state(ticker, cfg)
    scale = st.get("sized_scale")
    scale = float("nan") if scale is None else float(scale)
    hist = gap_history(ticker, cfg, asof=row["date"])
    r = meter_from_inputs(settings, row["p_up"], row["base_rate"], row["margin_m"], row["vol_ratio"],
                          scale, row["prev_position"], hist, position=row["position"])
    return r


def asset_state(ticker, cfg=None) -> dict:
    """The paper-trading state entry of ``ticker``.  A missing state file (or entry) is a damaged-files
    problem (``PaperError``: "state file missing"), NOT "no decision yet"."""
    cfg = DD._cfg(cfg)
    files = P.PaperFiles.for_config(cfg)
    st = DD.load_state(cfg).get("assets", {}).get(ticker)
    if st is None:
        raise P.PaperError(f"state file missing or without an entry for {ticker} ({files.state}) although the "
                           "ledger has rows; the ledger and state files must be kept together")
    return st


def _ledger_row(led: pd.DataFrame, asof=None):
    if asof is None:
        return led.iloc[-1]
    sub = led[led["date"] == pd.Timestamp(asof)]
    if sub.empty:
        raise NoDecisionError(f"no ledger row for {pd.Timestamp(asof).date()}")
    return sub.iloc[0]


# =========================================================================== evidence badge
BADGE_RULE = """\
Evidence badge rule (fixed in code BEFORE the per-asset results were examined; not tuned afterwards).
Inputs come from the saved development-period backtest (data/results/<ASSET>/), window = expanding,
costs = the headline cost (0.05% ETFs / 0.15% BTC), for the active model and strategy configuration:

  PT      Pesaran-Timmermann: p-value < 0.05 AND statistic > 0 (the up/down calls beat chance given
          how often the market and the model say "up"; two-sided p, so a negative statistic never counts)
  SKILL   Brier skill > 0 against BOTH reference rates, i.e. the MORE CONSERVATIVE (smaller) of
          (a) the skill vs the training-window base rate and (b) the skill vs the base rate of the
          evaluation window itself.  (a) alone can look positive just because the base rate drifted
          (a deliberately conservative choice; Bitcoin: +0.0059 vs the training rate but
          -0.0028 vs the evaluation rate)
  BEATS   net Sharpe of the strategy > net Sharpe of volatility-targeted buy-and-hold (same costs)
  PSR_X   probabilistic Sharpe ratio of the strategy's EXCESS return over vol-targeted buy-and-hold >= 0.95
  DSR_X   deflated Sharpe ratio of that excess return (trial count of this asset) >= 0.95
  DSR_G   deflated Sharpe ratio of the strategy (trial count of ALL assets, the most conservative) >= 0.95

  Green  "Evidence"        : PT and SKILL and BEATS and DSR_X and DSR_G  (all five)
  Amber  "Weak evidence"   : not green, but BEATS and (PT or PSR_X)   [PSR_X is used for amber only]
  Grey   "No proven edge"  : everything else, including any missing or undefined number
                             (a number that cannot be computed counts as "not shown")
Grey => the meter is drawn faded with FADED_NOTE.  Grey does not mean "proven useless": a few years of
daily data have little statistical power.
"""
PT_ALPHA = 0.05
PROB_MIN = 0.95
BADGES = {"none": ("No proven edge", "grey"), "weak": ("Weak evidence", "amber"),
          "evidence": ("Evidence", "green")}


def _num(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return float("nan")
    return v


def badge_from_numbers(*, pt_pvalue, pt_statistic, brier_skill, brier_skill_eval_base, sharpe, sharpe_vt,
                       psr_excess_vt, dsr_excess_vt, dsr_global) -> dict:
    """The pure rule (see ``BADGE_RULE``).  Any NaN/None makes its condition fail.  SKILL uses the
    smaller of the skill vs the training base rate and the skill vs the evaluation-window base rate."""
    pt_p, pt_s, bs_tr, bs_ev = _num(pt_pvalue), _num(pt_statistic), _num(brier_skill), _num(brier_skill_eval_base)
    bs = min(bs_tr, bs_ev) if np.isfinite(bs_tr) and np.isfinite(bs_ev) else float("nan")
    sr, sv, px, dx, dg = (_num(v) for v in (sharpe, sharpe_vt, psr_excess_vt, dsr_excess_vt, dsr_global))

    def ok(cond, *vals):
        return bool(all(np.isfinite(v) for v in vals) and cond)
    checks = {
        "PT": dict(passed=ok(pt_p < PT_ALPHA and pt_s > 0, pt_p, pt_s), pvalue=pt_p, statistic=pt_s,
                   needed="p < 0.05 and statistic > 0"),
        "SKILL": dict(passed=ok(bs > 0, bs), value=bs, vs_training_base=bs_tr, vs_eval_base=bs_ev,
                      needed="Brier skill > 0 vs BOTH the training base rate and the evaluation-window base rate"),
        "BEATS": dict(passed=ok(sr > sv, sr, sv), sharpe=sr, sharpe_vol_targeted_bh=sv,
                      needed="net Sharpe > vol-targeted buy-and-hold net Sharpe"),
        "PSR_X": dict(passed=ok(px >= PROB_MIN, px), value=px, needed="PSR of excess return >= 0.95"),
        "DSR_X": dict(passed=ok(dx >= PROB_MIN, dx), value=dx, needed="DSR of excess return >= 0.95"),
        "DSR_G": dict(passed=ok(dg >= PROB_MIN, dg), value=dg, needed="DSR (all trials) >= 0.95"),
    }
    p = {k: v["passed"] for k, v in checks.items()}
    if p["PT"] and p["SKILL"] and p["BEATS"] and p["DSR_X"] and p["DSR_G"]:
        level = "evidence"
    elif p["BEATS"] and (p["PT"] or p["PSR_X"]):
        level = "weak"
    else:
        level = "none"
    return _badge_dict(level, checks)


def _f(x, d=3) -> str:
    return "n/a" if x is None or not np.isfinite(x) else f"{x:.{d}f}"


def check_actual(key: str, c: dict) -> str:
    """The real numbers behind one badge check, in words (shown next to the check on the Today page)."""
    if key == "PT":
        return f"p = {_f(c['pvalue'])}, statistic = {_f(c['statistic'], 2)}"
    if key == "SKILL":
        return (f"skill vs training base rate {_f(c['vs_training_base'], 4)}, "
                f"vs evaluation-window base rate {_f(c['vs_eval_base'], 4)}; the smaller is used")
    if key == "BEATS":
        return f"strategy Sharpe {_f(c['sharpe'], 2)} vs vol-targeted buy-and-hold {_f(c['sharpe_vol_targeted_bh'], 2)}"
    return f"{_f(c['value'])}"


def _badge_dict(level, checks, extra_reason=None) -> dict:
    label, color = BADGES[level]
    failed = [k for k, v in checks.items() if not v["passed"]]
    reasons = [f"{k}: {v['needed']} - {'passed' if v['passed'] else 'not met'} (actual: {check_actual(k, v)})"
               for k, v in checks.items()]
    if extra_reason:
        reasons.insert(0, extra_reason)
    return {"level": level, "label": label, "color": color, "faded": level == "none",
            "note": FADED_NOTE if level == "none" else None, "checks": checks,
            "failed_checks": failed, "reasons": reasons, "rule": BADGE_RULE}


def evidence_badge(ticker, cfg=None, model=None, mode=None, vol_targeted=None, m_setting=None,
                   window="expanding") -> dict:
    """Badge for one asset / model / strategy configuration from the SAVED significance results.
    Defaults to the active paper-trading strategy.  Anything missing -> grey."""
    cfg = DD._cfg(cfg)
    st = DD.paper_settings(cfg)
    s = st.strategy
    model = model or s["model"]
    mode = mode or st.strategy_mode
    vol_targeted = st.vol_targeted if vol_targeted is None else bool(vol_targeted)
    m_setting = s["m_setting"] if m_setting is None else m_setting
    nan = float("nan")
    numbers = dict(pt_pvalue=nan, pt_statistic=nan, brier_skill=nan, brier_skill_eval_base=nan, sharpe=nan, sharpe_vt=nan,
                   psr_excess_vt=nan, dsr_excess_vt=nan, dsr_global=nan)
    missing = None
    try:
        res = DD.load_asset_results(ticker, cfg)
        mval = T.ADAPTIVE if str(m_setting) == T.ADAPTIVE else float(m_setting)
        if mode == "sized":
            raise KeyError("the 'sized' strategy mode has no logged backtest trial")
        cid = T.make_config_id(ticker, window, model, mode, vol_targeted, mval)
        canon = DD.canonical_config_id(DD.load_trial_index(cfg), cid)
        pm = res.get("prediction_metrics", ("window", "model"))
        pm = pm[(pm["window"] == window) & (pm["model"] == model)]
        if not pm.empty:
            numbers.update(pt_pvalue=pm.iloc[0]["pt_pvalue"], pt_statistic=pm.iloc[0]["pt_statistic"],
                           brier_skill=pm.iloc[0]["brier_skill"],
                           brier_skill_eval_base=(pm.iloc[0]["brier_skill_vs_eval_base"]
                                                  if "brier_skill_vs_eval_base" in pm.columns else nan))
        tr, vt = DD.trading_row(res, cid, window), DD.trading_row(res, "vol_targeted_bh", window)
        if tr is not None:
            numbers["sharpe"] = tr["sharpe"]
        if vt is not None:
            numbers["sharpe_vt"] = vt["sharpe"]
        sig = res.get("significance", ("subject", "window", "test", "reference", "value"))
        sig = sig[(sig["subject"] == canon) & (sig["window"] == window)]

        def val(test, ref=None):
            r = sig[sig["test"] == test]
            if ref is not None:
                r = r[r["reference"] == ref]
            return r.iloc[0]["value"] if len(r) else nan
        numbers.update(psr_excess_vt=val("psr_excess_vt"), dsr_excess_vt=val("dsr_excess_vt"),
                       dsr_global=val("dsr", "global"))
    except (FileNotFoundError, KeyError) as e:
        missing = f"saved results unavailable for this configuration ({e.args[0] if e.args else e})"
    b = badge_from_numbers(**numbers)
    if missing:
        b = _badge_dict("none", b["checks"], missing)
    b["inputs"] = {k: (None if not np.isfinite(_num(v)) else float(v)) for k, v in numbers.items()}
    b["config"] = dict(ticker=ticker, model=model, mode=mode, vol_targeted=vol_targeted,
                       m_setting=str(m_setting), window=window)
    return b


# =========================================================================== models agree
LIBOMP_HINT = ("lightgbm could not load (on a Mac this usually means libomp is missing). Fix: run "
               "`brew install libomp` or `sh scripts/fix_lightgbm_macos.sh`, then restart the dashboard "
               "(see 'Setup on a fresh Mac' in PROGRESS.md).")


def live_model_guard(ticker, cfg=None, asof=None) -> str | None:
    """Why the saved live model must NOT be used for the decision dated ``asof`` (``None`` = fine).
    The file holds the model trained for the LATEST ledger day, so it may only explain that day: for an
    older day it was trained on data that did not exist yet (look-ahead), and a model trained after
    ``asof`` is the same thing."""
    if asof is None:
        return None
    cfg = DD._cfg(cfg)
    d = pd.Timestamp(asof)
    led = DD.ledger_for(ticker, cfg)
    if len(led) and d != pd.Timestamp(led["date"].max()):
        return (f"the saved live model belongs to the latest decision ({pd.Timestamp(led['date'].max()).date()}), "
                f"not to {d.date()}; using it would look ahead")
    td = DD.load_state(cfg).get("assets", {}).get(ticker, {}).get("train_date")
    if td is not None and pd.Timestamp(td) > d:
        return f"the saved live model was trained on {td}, after the decision day {d.date()}; using it would look ahead"
    return None


def load_live_model_ex(ticker, cfg=None, asof=None):
    """``(model, reason)``: the persisted live ensemble (data/paper/models/<T>.joblib; read-only) or
    ``(None, plain-language reason)`` saying why it could not be loaded.  With ``asof`` (the decision
    day being explained) a model that is not the one for that day is refused (``live_model_guard``)."""
    import joblib
    guard = live_model_guard(ticker, cfg, asof)
    if guard:
        return None, guard
    p = P.PaperFiles.for_config(DD._cfg(cfg)).model(ticker)
    if not Path(p).exists():
        return None, f"the saved live model file is missing ({Path(p).name}); press Refresh data to rebuild it"
    try:
        return joblib.load(p), None
    except Exception as e:      # noqa: BLE001 - report the real reason
        text = f"{type(e).__name__}: {e}"
        if "libomp" in text or "lib_lightgbm" in text or "OpenMP" in text or "lightgbm" in text.lower():
            return None, LIBOMP_HINT + f" Details: {text.splitlines()[0][:160]}"
        return None, f"the saved live model could not be read ({text.splitlines()[0][:200]})"


def models_agree(model, x_row: pd.DataFrame, base_rate: float, margin_m=None) -> dict:
    """How many of the member models lean the same way on the feature row ``x_row`` (one row).

    A member "leans up" if its calibrated probability is >= the base rate (more bullish than a
    typical day), "down" otherwise.  ``n_agree`` is the size of the larger camp (2 or 3 of 3).
    Deterministic and read-only: it only calls ``predict_up`` on the saved members."""
    members = getattr(model, "members", None)
    if not members:
        raise ValueError("model has no ensemble members")
    probs = {name: float(m.predict_up(x_row)[0]) for name, m in members.items()}
    up = [n for n, p in probs.items() if p >= base_rate]
    down = [n for n in probs if n not in up]
    camp, direction = (up, "above") if len(up) >= len(down) else (down, "below")
    total = len(probs)
    thr = None if margin_m is None else float(base_rate) - float(margin_m)
    return {
        "probabilities": probs, "base_rate": float(base_rate), "n_models": total,
        "n_above": len(up), "n_below": len(down), "direction": direction, "n_agree": len(camp),
        "unanimous": len(camp) == total,
        "above_strategy_threshold": ({n: p >= thr for n, p in probs.items()} if thr is not None else None),
        "ensemble_p": float(np.mean(list(probs.values()))),
        "text": (f"{len(camp)} of {total} models lean {direction} the usual base rate "
                 f"({', '.join(sorted(camp))})."),
    }


# =========================================================================== feature wording
DOW = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")


def _spec(name: str, ticker: str):
    """(description, kind) of a feature for sentences.  kind: pct | vol | ratio | level | rsi |
    corr | var | flag | dow | num."""
    unit = "day" if ticker == F.BTC else "trading day"
    if name.startswith("ret_lag_"):
        k = int(name.rsplit("_", 1)[1])
        d = {1: f"{ticker}'s latest daily move", 2: f"{ticker}'s move the day before"}.get(
            k, f"{ticker}'s daily move {k - 1} {unit}s earlier")
        return d, "pct"
    if name.startswith("trend_"):
        return f"{ticker}'s price change over the last {name.split('_')[1]} days", "pct"
    if name.startswith("dist_ma_"):
        return f"{ticker}'s distance from its {name.split('_')[2]}-day average price", "pct"
    if name in ("vol_10", "vol_20", "vol_60"):
        return f"{name.split('_')[1]}-day volatility (annualised)", "vol"
    table = {
        "overnight_gap": ("the overnight gap (yesterday's close to today's open)", "pct"),
        "intraday_ret": ("today's open-to-close move", "pct"),
        "r5": (f"{ticker}'s last five daily moves added together", "pct"),
        "yz_vol_21": ("21-day Yang-Zhang volatility (annualised)", "vol"),
        "vol_ratio_10_60": ("10-day volatility relative to 60-day volatility", "ratio"),
        "vol_ratio_20_60": ("20-day volatility relative to 60-day volatility", "ratio"),
        "vol_chg_20": ("trading volume versus its 20-day average", "pct"),
        "vix_level": ("the VIX", "level"),
        "vix_chg": ("the VIX's latest daily change", "pct"),
        "vix_ts": ("the VIX divided by the 3-month VIX (VIX3M)", "ratio"),
        "vix_backwardation": ("the VIX is above the 3-month VIX (backwardation: near-term fear exceeds longer-term fear)", "flag"),
        "vrp": ("the variance risk premium (options-implied minus recently realised variance)", "var"),
        "rsi_14": ("the 14-day RSI", "rsi"),
        "new_high_20": (f"{ticker} is at a 20-day high", "flag"),
        "dow_next": ("the weekday being predicted", "dow"),
        "is_weekend_next": ("the predicted day is a weekend day", "flag"),
        "pre_fomc": ("the predicted day is a scheduled Fed (FOMC) announcement day", "flag"),
        "spy_ret": ("the S&P 500's latest daily move", "pct"),
        "qqq_ret": ("the Nasdaq-100's latest daily move", "pct"),
        "iwm_ret": ("the Russell 2000's latest daily move", "pct"),
        "stale": ("stock-market inputs are carried over from an earlier day (stale)", "flag"),
        "btc_spy_corr_60": ("the 60-day correlation between Bitcoin and the S&P 500", "corr"),
        "iwm_spy_spread_5": ("the Russell 2000 minus the S&P 500 over five days", "pct"),
    }
    return table.get(name, (f"the feature '{name}'", "num"))


def format_value(kind: str, x: float) -> str:
    if kind == "pct":
        return f"{x * 100:+.1f}%"
    if kind == "vol":
        return f"{x * 100:.1f}%"
    if kind == "ratio":
        return f"{x:.2f}"
    if kind == "level":
        return f"{x:.1f}"
    if kind == "rsi":
        return f"{x:.0f}"
    if kind == "corr":
        return f"{x:+.2f}"
    if kind == "var":
        return f"{x:+.3f}"
    if kind == "dow":
        return DOW[int(round(x)) % 7]
    return f"{x:.3g}"


def feature_facts(ds: pd.DataFrame, column: str, decision_date, cutoff) -> dict:
    """Today's value of ``column`` and where it sits in the asset's own history.  The history is the
    development period only (rows strictly before the decision date and on or before the pinned
    hold-out cutoff), so nothing after the decision date -- and nothing from the hold-out -- is used."""
    d = pd.Timestamp(decision_date)
    value = float(ds.loc[d, column])
    hs = ds.loc[(ds.index < d) & (ds.index <= pd.Timestamp(cutoff)), column].dropna()
    hist = hs.to_numpy(float)
    facts = {"column": column, "value": value, "n_history": int(len(hist)),
             "history_start": (str(hs.index[0].year) if len(hist) else None),
             "history_end": (str(hs.index[-1].year) if len(hist) else None)}
    if len(hist):
        facts.update(below=percentile_below(hist, value), above=percentile_above(hist, value),
                     q25=float(np.quantile(hist, 0.25)), q75=float(np.quantile(hist, 0.75)),
                     share_equal=float(100.0 * np.mean(hist == value)))
    return facts


def share_phrase(share: float) -> str:
    """"37% of days" / "99.5% of days" / "almost every day" -- never rounds a share below 100% up to 100%."""
    if share >= 100.0:
        return "every day"
    if share >= 99.95:
        return "almost every day"
    if share >= 99.5:
        return f"{share:.1f}% of days"
    return f"{share:.0f}% of days"


def feature_sentence(ticker, column, facts, date=None) -> str:
    """One sentence with today's actual number and what is normal for this asset.  ``date`` (the
    decision day) only adds a caveat for Bitcoin weekend volume."""
    desc, kind = _spec(column, ticker)
    v = facts["value"]
    yr = facts.get("history_start")
    end = facts.get("history_end")
    since = (f" from {yr} to {end}" if (yr and end) else f" since {yr}" if yr else "")
    if facts["n_history"] == 0:
        return f"{desc[0].upper() + desc[1:]} is {format_value(kind, v)} (no earlier history to compare with)."
    if kind == "flag":
        if v >= 0.5:
            share = facts["share_equal"]
            return f"{desc[0].upper() + desc[1:]} today - that was true on {share:.0f}% of days{since}."
        share = facts["share_equal"]
        return (f"{desc[0].upper() + desc[1:]} is not the case today - that was true of "
                f"{100 - share:.0f}% of days{since}.")
    if kind == "dow":
        return (f"The predicted day is a {format_value(kind, v)}; {facts['share_equal']:.0f}% of days"
                f"{since} were followed by a {format_value(kind, v)}.")
    if facts["above"] <= facts["below"]:
        where = f"higher than {share_phrase(facts['below'])}{since}"
    else:
        where = f"lower than {share_phrase(facts['above'])}{since}"
    normal = (f"the middle half of days sit between {format_value(kind, facts['q25'])} and "
              f"{format_value(kind, facts['q75'])}")
    out = f"{desc[0].upper() + desc[1:]} is {format_value(kind, v)}, {where} ({normal})."
    if (column == "vol_chg_20" and ticker == F.BTC and date is not None
            and pd.Timestamp(date).day_of_week >= 5):
        out += (" Note: Bitcoin trades on weekends but volume is usually lower on Saturdays and Sundays, "
                "so a weekend bar looks quiet against a weekday-heavy average.")
    return out


# =========================================================================== group contributions
def _feature_of(col: str) -> str:
    return col[: -len("__missing")] if col.endswith("__missing") else col


def feature_contributions(model, x_row: pd.DataFrame) -> pd.DataFrame:
    """What moved the model today, per feature, in calibrated log-odds units relative to an
    average training day.

    * logistic regression: coefficient x standardised value (imputer -> scaler -> coef), times the
      slope A of its Platt calibrator (A can be negative: then the calibrator flips the sign);
    * LightGBM: ``pred_contrib`` (exact tree-SHAP) per feature, times its Platt slope;
    * the random forest has no cheap exact decomposition and is not decomposed (stated in the text).

    ``combined`` is the plain mean of the two decomposed members.  Returns one row per feature with
    columns ``feature, group, logreg, lgbm, combined``."""
    groups = {c: g for g, cols in F.feature_groups.items() for c in cols}
    members = model.members
    cols = list(x_row.columns)
    out = pd.DataFrame({"feature": cols})
    out["group"] = out["feature"].map(groups)
    out["logreg"] = 0.0
    out["lgbm"] = 0.0
    n_used = 0
    cm = members.get("logreg")
    if cm is not None:
        pipe = cm.base_model
        imp, sc, clf = (pipe.named_steps[k] for k in ("imputer", "scaler", "clf"))
        z = sc.transform(imp.transform(x_row))[0]
        a = float(cm.calibrator.coef_[0][0])
        eff = a * clf.coef_[0] * z
        s = pd.Series(eff, index=[_feature_of(c) for c in imp.get_feature_names_out()]).groupby(level=0).sum()
        out["logreg"] = out["feature"].map(s).fillna(0.0)
        n_used += 1
    cl = members.get("lgbm")
    if cl is not None:
        b = cl.base_model
        xl = x_row[list(b.feature_name_)]
        contrib = np.asarray(b.predict(xl, pred_contrib=True))[0]
        a = float(cl.calibrator.coef_[0][0])
        s = pd.Series(a * contrib[:-1], index=list(b.feature_name_))
        out["lgbm"] = out["feature"].map(s).fillna(0.0)
        n_used += 1
    if n_used == 0:
        raise ValueError("neither the logistic regression nor the LightGBM member is available")
    cols_used = [c for c in ("logreg", "lgbm") if (c == "logreg" and cm is not None) or (c == "lgbm" and cl is not None)]
    out["combined"] = out[cols_used].mean(axis=1)
    return out


def group_effects(contrib: pd.DataFrame, p: float) -> pd.DataFrame:
    """Sum the per-feature effects by feature group; ``points`` = approximate change of the
    probability in percentage points (log-odds effect x p(1-p) x 100, a local linearisation)."""
    g = contrib.groupby("group")[["logreg", "lgbm", "combined"]].sum()
    g["points"] = g["combined"] * p * (1.0 - p) * 100.0
    g["abs"] = g["combined"].abs()
    return g.sort_values("abs", ascending=False)


# =========================================================================== trust (hit rate)
def confidence_bucket(preds: pd.DataFrame, p: float, base: float, n_bins=10, asof=None) -> dict:
    """Out-of-sample hit rate of the up/down call in the confidence decile that today's
    ``|p - base_rate|`` falls in (deciles from ``metrics.hit_rate_by_confidence_decile``).
    With ``asof`` only predictions dated strictly before it are used, so trust statistics can never
    include the decision day or later."""
    if asof is not None:
        preds = preds[pd.to_datetime(preds["date"]) < pd.Timestamp(asof)]
    pp, yy, bb = (preds[c].to_numpy(float) for c in ("p_up", "y", "base_rate"))
    if len(pp) < n_bins:         # empty (or too short) history: say so instead of failing in qcut
        nan = float("nan")
        return {"decile": None, "n_bins": n_bins, "n": 0, "hit_rate": nan, "conf_lo": nan, "conf_hi": nan,
                "overall_hit_rate": (float(np.mean((pp >= bb) == (yy > 0.5))) if len(pp) else nan),
                "always_up_rate": (float(np.mean(yy > 0.5)) if len(pp) else nan), "n_total": int(len(pp)),
                "today_confidence": float(abs(p - base)), "available": False}
    conf = np.abs(pp - bb)
    dec = pd.qcut(pd.Series(conf).rank(method="first"), n_bins, labels=False) + 1
    hi = pd.Series(conf).groupby(dec.to_numpy()).max()
    lo = pd.Series(conf).groupby(dec.to_numpy()).min()
    today = abs(p - base)
    bucket = int(next((d for d in hi.index if today <= hi[d]), n_bins))
    tab = M.hit_rate_by_confidence_decile(pp, yy, bb, n_bins)
    return {"decile": bucket, "n_bins": n_bins, "n": int(tab.loc[bucket, "n"]),
            "hit_rate": float(tab.loc[bucket, "hit_rate"]),
            "conf_lo": float(lo[bucket]), "conf_hi": float(hi[bucket]),
            "overall_hit_rate": float(np.mean((pp >= bb) == (yy > 0.5))),
            "always_up_rate": float(np.mean(yy > 0.5)), "n_total": int(len(pp)),
            "today_confidence": float(today), "available": True}


# =========================================================================== the "Why?" text
@dataclass
class Why:
    ticker: str
    date: str
    saw: list                    # part 1: [{group, direction, points, sentence, feature, facts}]
    theory: list                 # part 2: [{group, text, source, url, learn_more}]
    trust: dict                  # part 3
    conclusion: str              # part 4
    method_note: str
    text: str = ""


METHOD_NOTE = ("What moved the model today: the logistic regression (coefficient x value) and LightGBM "
               "(tree-SHAP contributions) members, summed by feature group and averaged; the random "
               "forest is not decomposed. This describes what moved the model, not why the market will move.")


def explain_today(ticker, ds: pd.DataFrame, decision_date, model, meter: MeterReading, preds: pd.DataFrame,
                  badge: dict, cutoff, theory=None, top_n: int = 4) -> Why:
    """The four-part "Why?" explanation for ``decision_date`` (templates only).

    ``ds`` is the feature dataset; ONLY rows up to and including ``decision_date`` are used (rows
    after it are dropped first, a test appends garbage to prove it).  ``preds`` are the saved
    out-of-sample predictions of the active model (dev only)."""
    theory = theory or load_theory()
    d = pd.Timestamp(decision_date)
    ds = ds[ds.index <= d]
    if d not in ds.index:
        raise LookupError(f"{ticker}: no feature row for {d.date()}")
    feats = F.feature_columns(ds)
    x_row = ds[feats].loc[[d]]
    contrib = feature_contributions(model, x_row)
    ge = group_effects(contrib, meter.p_up)
    top = ge.head(top_n)
    top = top[top["abs"] > 1e-12]
    saw, theory_items = [], []
    for g, r in top.iterrows():
        sub = contrib[contrib["group"] == g].copy()
        sub["abs"] = sub["combined"].abs()
        lead = sub.sort_values("abs", ascending=False).iloc[0]["feature"]
        facts = feature_facts(ds, lead, d, cutoff)
        up = r["combined"] > 0
        gl = theory["feature_groups"][g]
        sent = feature_sentence(ticker, lead, facts, d)
        push = (f"This nudged the model's probability {'up' if up else 'down'} by about "
                f"{abs(r['points']):.2f} percentage points (signal group: {gl['title']}).")
        saw.append({"group": g, "group_title": gl["title"], "short_name": gl["short_name"], "direction": "up" if up else "down",
                    "points": float(r["points"]), "feature": lead, "facts": facts,
                    "sentence": f"{sent} {push}"})
        src = gl["sources"][0] if gl.get("sources") else None
        theory_items.append({
            "group": g, "text": " ".join(gl["short"].split()),
            "source": (None if src is None else source_line(src)), "url": (None if src is None else src["url"]),
            "link_note": (None if src is None else src.get("link_note")),
            "learn_more": {k: gl.get(k) for k in ("plain_english", "formula", "evidence", "caveats", "sources")}})
    tb = confidence_bucket(preds, meter.p_up, meter.base_rate, asof=d)
    rc = theory["reality_check"]["realistic_auc"]
    top_caveat = ""
    if saw:
        top_caveat = " ".join(str(theory["feature_groups"][saw[0]["group"]]["caveats"]).split())
    if not tb["available"]:
        trust_text = (f"There are too few saved out-of-sample predictions ({tb['n_total']}; at least {tb['n_bins']} are "
                      "needed) to say how often this model's calls were right at today's confidence level.")
    else:
        trust_text = (
            f"Out of sample, when this model was as confident as today (decile {tb['decile']} of {tb['n_bins']}: "
            f"{tb['conf_lo'] * 100:.2f} to {tb['conf_hi'] * 100:.2f} percentage points from the base rate), its "
            f"up/down call was right {tb['hit_rate'] * 100:.1f}% of the time over {tb['n']} days; across all days it "
            f"was right {tb['overall_hit_rate'] * 100:.1f}% and always guessing 'up' would have been right "
            f"{tb['always_up_rate'] * 100:.1f}%.")
    trust = {"bucket": tb, "text": trust_text,
             "caveat": " ".join(rc["plain_english"].split()),
             "signal_caveat": top_caveat,
             "badge_label": badge["label"], "badge_note": badge.get("note")}
    ups = [s["short_name"] for s in saw if s["direction"] == "up"]
    downs = [s["short_name"] for s in saw if s["direction"] == "down"]
    parts = []
    if ups:
        parts.append(f"{_join(ups)} nudged the probability up")
    if downs:
        parts.append(f"{_join(downs)} nudged it down")
    pct_txt = ("" if not np.isfinite(meter.percentile) else
               f", which is the {ordinal(meter.percentile)} percentile of the gaps this model has produced")
    if meter.gap < 0:
        net = "below the usual rate, so this is not a bullish reading"
    elif meter.gap > 0:
        net = "above the usual rate"
    else:
        net = "exactly the usual rate"
    conclusion = (
        f"{_cap('; '.join(parts)) if parts else 'No feature group moved the model noticeably'} "
        f"(compared with an average day for this model); in total today's probability is "
        f"{meter.probability_text}{pct_txt}, {net}. The meter reads "
        f"{meter.zone} and the paper account holds {meter.position * 100:.0f}% "
        f"({'stepping aside means cash, not a short' if meter.zone_index == 0 else 'long only, never leveraged'}).")
    why = Why(ticker=ticker, date=str(d.date()), saw=saw, theory=theory_items, trust=trust,
              conclusion=conclusion, method_note=METHOD_NOTE)
    why.text = render_text(why)
    return why


def _cap(t: str) -> str:
    return t[:1].upper() + t[1:]


def _join(items):
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def render_text(w: Why) -> str:
    lines = ["1. What the model saw"]
    lines += [f"   - {s['sentence']}" for s in w.saw]
    lines += ["2. The theory behind it"]
    for t in w.theory:
        lines.append(f"   - {t['text']}" + (f" [{t['source']}]" if t["source"] else "")
                     + (f" {t['link_note']}" if t.get("link_note") else ""))
    lines += ["3. How much to trust it", f"   - {w.trust['text']}",
              f"   - {w.trust['caveat']}"]
    if w.trust.get("signal_caveat"):
        lines.append(f"   - Caveat for the lead signal: {w.trust['signal_caveat']}")
    if w.trust.get("badge_note"):
        lines.append(f"   - {w.trust['badge_note']}")
    lines += ["4. The conclusion", f"   - {w.conclusion}", f"({w.method_note})"]
    return "\n".join(lines)


# =========================================================================== orchestration
def load_dataset_asof(ticker, cfg=None, asof=None) -> pd.DataFrame:
    """Feature dataset rebuilt (in memory, read-only) from the stored prices and cut at ``asof``
    (default: the latest ledger date).  Features are causal, so rows up to ``asof`` do not depend
    on anything later."""
    cfg = DD._cfg(cfg)
    if asof is None:
        led = DD.ledger_for(ticker, cfg)
        if led.empty:
            raise NoDecisionError(f"{ticker}: ledger is empty")
        asof = led["date"].max()
    ds = F.build_dataset(ticker, cfg)
    return ds[ds.index <= pd.Timestamp(asof)]


def today_report(ticker, cfg=None, asof=None, ds=None, model=None) -> dict:
    """Everything the "Today" card of one asset needs (a plain dict of JSON-friendly values plus
    the ``Why`` object).  Read-only; deterministic."""
    cfg = DD._cfg(cfg)
    meter = today_meter(ticker, cfg, asof)
    led = DD.ledger_for(ticker, cfg)
    row = _ledger_row(led, asof)
    badge = evidence_badge(ticker, cfg)
    ds = load_dataset_asof(ticker, cfg, row["date"]) if ds is None else ds
    reason = None
    if model is None:
        model, reason = load_live_model_ex(ticker, cfg, asof=row["date"])
    out = {"ticker": ticker, "date": str(pd.Timestamp(row["date"]).date()), "meter": meter, "badge": badge,
           "disclaimer": DISCLAIMER, "position": float(row["position"]),
           "faded": badge["faded"], "faded_note": badge["note"], "agree": None, "why": None}
    if model is None:
        out["warning"] = f"'Models agree' and 'Why?' are unavailable: {reason.rstrip('.')}."
        return out
    feats = F.feature_columns(ds)
    x_row = ds[feats].loc[[pd.Timestamp(row["date"])]]
    out["agree"] = models_agree(model, x_row, float(row["base_rate"]), row["margin_m"])
    out["agree"]["matches_ledger"] = bool(abs(out["agree"]["ensemble_p"] - float(row["p_up"])) < 1e-6)
    settings = DD.paper_settings(cfg)
    res = DD.load_asset_results(ticker, cfg)
    preds = DD.model_predictions(res, settings.strategy["model"])
    if preds.empty:
        raise ValueError(f"{ticker}: the saved out-of-sample predictions of the {settings.strategy['model']} model "
                         "are empty (re-run `uv run python -m quant.backtest`)")
    out["why"] = explain_today(ticker, ds, row["date"], model, meter, preds, badge, DD.cutoff(cfg))
    return out
