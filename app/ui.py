"""Drawing helpers for the Quant Lab dashboard (Plotly figures + small HTML snippets).

No Streamlit import and no data loading here: every function takes data that the page got from
``quant.dash_data`` / ``quant.explain`` (saved results) and returns a Plotly figure or a string.
That keeps the page code short and lets the figures be tested without a browser.
"""
from __future__ import annotations

import html

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

BG = "rgba(0,0,0,0)"
GRID = "rgba(160,170,190,0.15)"
TEXT = "#D8DEEA"
C_STRAT, C_BH, C_VT = "#4FB3FF", "#9AA4B2", "#F2A93B"
ZONE_COLORS = ["#E5534B", "#E8903E", "#A9B0BC", "#5BBF7A", "#2FA05A"]   # Sell .. Strong buy
BADGE_COLORS = {"grey": "#8A93A3", "amber": "#E3A72F", "green": "#3FB950"}
SERIES_LABEL = {"strategy": "Selected model's strategy", "buy_and_hold": "Buy and hold",
                "vol_targeted_bh": "Vol-targeted buy and hold",
                "chosen_strategy": "Chosen strategy (fixed in config.yaml)"}
SERIES_COLOR = {"strategy": C_STRAT, "chosen_strategy": C_STRAT, "buy_and_hold": C_BH,
                "vol_targeted_bh": C_VT}
LEGEND_BELOW = dict(orientation="h", yanchor="top", y=-0.2, x=0)


def rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def style(fig: go.Figure, height: int = 330, title: str | None = None, legend_below: bool = True) -> go.Figure:
    """Common dark, compact layout (readable at ~1366x768)."""
    fig.update_layout(
        template="plotly_dark", paper_bgcolor=BG, plot_bgcolor=BG, height=height,
        margin=dict(l=48, r=16, t=38 if title else 14, b=56), font=dict(color=TEXT, size=12),
        title=dict(text=title, x=0, font=dict(size=14)) if title else None,
        legend=LEGEND_BELOW if legend_below else dict(),
        hoverlabel=dict(bgcolor="#1B2130"))
    fig.update_xaxes(gridcolor=GRID, zeroline=False)
    fig.update_yaxes(gridcolor=GRID, zeroline=False)
    return fig


# ----------------------------------------------------------------------------- the signal meter
def gauge_ticks(meter) -> tuple:
    """(tick positions, labels) on the gap axis in percentage points: 0 (the usual rate) and the real
    thresholds of the strategy's zones, thinned so labels never overlap."""
    span = (meter.axis_hi - meter.axis_lo) * 100.0
    cand = {0.0}
    for b in meter.bounds[:-1]:
        if b["hi"] > b["lo"] and np.isfinite(b["hi"]) and b["hi"] < meter.axis_hi:
            cand.add(round(b["hi"] * 100.0, 3))
    vals = []
    for v in sorted(cand):
        if meter.axis_lo * 100.0 < v < meter.axis_hi * 100.0 and all(abs(v - u) > 0.1 * span for u in vals):
            vals.append(v)
    return vals, ["0" if abs(v) < 1e-9 else f"{v:+.1f}" for v in vals]


def gauge_figure(meter, faded: bool = False, height: int = 250) -> go.Figure:
    """Semicircle gauge drawn in GAP space (today's probability minus the typical up-day rate, in
    percentage points).  The zone edges are the active strategy's real thresholds, and the needle is
    today's gap, so it always sits in the zone of the paper-trading position.  Zones with zero width
    (e.g. Reduce / Neutral / Strong buy for the hold-or-step-aside strategies) are not drawn.
    ``faded`` draws everything dim (grey evidence badge)."""
    alpha = 0.28 if faded else 0.92
    steps = [dict(range=[b["lo"] * 100.0, b["hi"] * 100.0], color=rgba(ZONE_COLORS[i], alpha), line=dict(width=0))
             for i, b in enumerate(meter.bounds) if b["hi"] > b["lo"]]
    needle_color = "#7C8596" if faded else "#FFFFFF"
    ticks, labels = gauge_ticks(meter)
    nv = float(meter.needle) * 100.0
    fig = go.Figure(go.Indicator(
        mode="gauge", value=nv,
        gauge=dict(
            shape="angular",
            axis=dict(range=[meter.axis_lo * 100.0, meter.axis_hi * 100.0], tickmode="array", tickvals=ticks,
                      ticktext=labels, tickcolor=TEXT, tickfont=dict(size=10)),
            bar=dict(color=BG, thickness=0.0), bgcolor=BG, borderwidth=0, steps=steps,
            threshold=dict(line=dict(color=needle_color, width=6), thickness=0.9, value=nv)),
        domain=dict(x=[0, 1], y=[0, 1])))
    from quant.explain import ordinal
    pct = ("" if not np.isfinite(meter.percentile) else
           f"about the {ordinal(meter.percentile)} percentile of past days")
    fig.add_annotation(x=0.5, y=0.12, xref="paper", yref="paper", showarrow=False,
                       text=f"<b>{html.escape(meter.zone)}</b>", font=dict(size=20, color=needle_color))
    fig.add_annotation(x=0.5, y=-0.02, xref="paper", yref="paper", showarrow=False, text=pct,
                       font=dict(size=11, color="#9AA4B2"))
    fig.update_layout(template="plotly_dark", paper_bgcolor=BG, plot_bgcolor=BG, height=height,
                      margin=dict(l=24, r=24, t=28, b=22), font=dict(color=TEXT))
    return fig


def zone_legend_html(bounds, faded: bool = False) -> str:
    """Coloured chips naming the five zones (a zone this strategy does not have is crossed out)."""
    a = 0.4 if faded else 1.0
    from quant.explain import ZONES
    chips = []
    for i, z in enumerate(ZONES):
        reach = bounds[i]["hi"] > bounds[i]["lo"]
        deco = "" if reach else "text-decoration:line-through;opacity:.45;"
        chips.append(f"<span style='display:inline-block;margin:2px 6px 2px 0;padding:1px 8px;border-radius:9px;"
                     f"font-size:12px;background:{rgba(ZONE_COLORS[i], 0.25 * a + 0.05)};"
                     f"border:1px solid {rgba(ZONE_COLORS[i], 0.8 * a)};{deco}'>{html.escape(z)}</span>")
    return "".join(chips)


def badge_html(badge: dict) -> str:
    c = BADGE_COLORS[badge["color"]]
    return (f"<span style='display:inline-block;padding:3px 12px;border-radius:12px;font-weight:600;"
            f"font-size:13px;color:#10141C;background:{c};'>{html.escape(badge['label'])}</span>"
            f"&nbsp;<span style='font-size:12px;color:#9AA4B2'>evidence badge ({badge['color']})</span>")


# ----------------------------------------------------------------------------- backtest charts
def equity_fig(curves: pd.DataFrame, title="Growth of 1 (out of sample, after costs)", log=False,
               height=320, strategy_label: str | None = None) -> go.Figure:
    fig = go.Figure()
    for c in ("strategy", "buy_and_hold", "vol_targeted_bh"):
        if c in curves:
            fig.add_scatter(x=curves.index, y=curves[c],
                            name=strategy_label if (c == "strategy" and strategy_label) else SERIES_LABEL[c], mode="lines",
                            line=dict(color=SERIES_COLOR[c], width=2.2 if c == "strategy" else 1.5))
    fig.update_yaxes(type="log" if log else "linear", title="Value of 1 invested")
    return style(fig, height, title)


def drawdown_fig(curves: pd.DataFrame, height=260, strategy_label: str | None = None) -> go.Figure:
    fig = go.Figure()
    for c in ("strategy", "buy_and_hold", "vol_targeted_bh"):
        col = f"dd_{c}"
        if col in curves:
            fig.add_scatter(x=curves.index, y=curves[col],
                            name=strategy_label if (c == "strategy" and strategy_label) else SERIES_LABEL[c], mode="lines",
                            line=dict(color=SERIES_COLOR[c], width=1.4))
    fig.update_yaxes(title="Drop from the previous peak", tickformat=".0%")
    return style(fig, height, "Drawdown: how far below its best level each line was")


def sharpe_band_fig(table: pd.DataFrame, k=2.0, height=300) -> go.Figure:
    """Annualised Sharpe with a +-k standard-error band for every row (``label``, ``sharpe``, ``sharpe_se``)."""
    t = table.dropna(subset=["sharpe", "sharpe_se"]).iloc[::-1]
    colors = [C_BH if k_ == "baseline" else C_STRAT for k_ in t["kind"]]
    fig = go.Figure(go.Scatter(
        x=t["sharpe"], y=t["label"], mode="markers", marker=dict(size=9, color=colors),
        error_x=dict(type="data", array=k * t["sharpe_se"], color="#8A93A3", thickness=1.5),
        hovertemplate="%{y}<br>Sharpe %{x:.2f}<extra></extra>"))
    fig.add_vline(x=0, line=dict(color="#E5534B", dash="dot"))
    fig.update_xaxes(title=f"Annualised Sharpe ratio (bars = plus/minus {k:g} standard errors)")
    return style(fig, height, "Sharpe ratio with its uncertainty band", legend_below=False)


# ----------------------------------------------------------------------------- leverage charts
def add_crash_bands(fig, windows, x_min, x_max):
    """Shade the historical crash windows that overlap the plotted dates."""
    for w in windows or []:
        if not w.get("n_oos_days"):
            continue
        a, b = pd.Timestamp(w["start"]), pd.Timestamp(w["end"])
        if b < x_min or a > x_max:
            continue
        fig.add_vrect(x0=max(a, x_min), x1=min(b, x_max), fillcolor="rgba(229,83,75,0.16)", line_width=0,
                      annotation_text=w["label"], annotation_position="top left",
                      annotation_font=dict(size=10, color="#F0A39E"))


def leverage_equity_fig(eq_by_series: dict, L: float, windows=None, height=340) -> go.Figure:
    """Equity at leverage ``L`` for each series, log scale, crash windows shaded."""
    fig = go.Figure()
    xmin = xmax = None
    for s, eq in eq_by_series.items():
        if L not in eq.columns:
            continue
        y = eq[L].dropna()
        fig.add_scatter(x=y.index, y=y, name=SERIES_LABEL[s], mode="lines",
                        line=dict(color=SERIES_COLOR[s], width=2.2 if s == "chosen_strategy" else 1.4))
        xmin = y.index.min() if xmin is None else min(xmin, y.index.min())
        xmax = y.index.max() if xmax is None else max(xmax, y.index.max())
    if xmin is not None:
        add_crash_bands(fig, windows, xmin, xmax)
    fig.add_hline(y=0.2, line=dict(color="#E5534B", dash="dot"), annotation_text="wiped-out line (20%)",
                  annotation_font=dict(size=10, color="#F0A39E"), annotation_position="bottom right")
    fig.update_yaxes(type="log", title="Value of 1 (log scale)")
    return style(fig, height, f"{L:g}x leveraged equity (fake money, daily rebalanced, borrowing cost included)")


def growth_fig(kelly: pd.DataFrame, height=320) -> go.Figure:
    """Long-run growth rate g(L) = L*mu - L^2*sigma^2/2 with the Kelly point and its uncertainty band."""
    k = kelly.sort_values("L")
    fig = go.Figure()
    fig.add_scatter(x=k["L"], y=k["g_hi"], mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip")
    fig.add_scatter(x=k["L"], y=k["g_lo"], mode="lines", line=dict(width=0), fill="tonexty",
                    fillcolor="rgba(79,179,255,0.18)", name="uncertainty band", hoverinfo="skip")
    fig.add_scatter(x=k["L"], y=k["g"], mode="lines", name="growth rate g(L)", line=dict(color=C_STRAT, width=2.5))
    r = k.iloc[0]
    ls = float(r["L_star"])
    gl = lambda L: L * r["mu"] - L * L * r["sigma2"] / 2.0   # noqa: E731 - same formula as the saved curve
    fig.add_vrect(x0=float(r["L_star_lo"]), x1=min(float(r["L_star_hi"]), float(k["L"].max())),
                  fillcolor="rgba(91,191,122,0.12)", line_width=0)
    fig.add_scatter(x=[ls], y=[gl(ls)], mode="markers+text", name="Kelly leverage",
                    marker=dict(size=14, color="#5BBF7A", symbol="star"),
                    text=[f"Kelly {ls:.2f}x"], textposition="top center")
    if 2 * ls <= float(k["L"].max()):
        fig.add_scatter(x=[2 * ls], y=[0.0], mode="markers+text", name="2x Kelly: growth about zero",
                        marker=dict(size=11, color="#E5534B", symbol="x"),
                        text=["2x Kelly"], textposition="bottom center")
    fig.add_hline(y=0, line=dict(color="#8A93A3", dash="dot"))
    fig.update_xaxes(title="Leverage L (times your own money)")
    top = max(float(k["g"].max()) * 2.5, 0.05)      # keep the curve readable; the wide band is clipped
    fig.update_yaxes(title="Long-run growth per year", tickformat=".0%",
                     range=[min(float(k["g"].min()) * 1.05, -0.05), top])
    return style(fig, height, "More leverage is not always more growth")


# ----------------------------------------------------------------------------- diagnostics charts
def reliability_fig(rel: pd.DataFrame, height=340) -> go.Figure:
    """Reliability curve (predicted vs actual frequency) with the bin counts underneath."""
    r = rel.sort_values("bin")
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.72, 0.28], vertical_spacing=0.06)
    fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", name="perfect calibration",
                    line=dict(color="#8A93A3", dash="dash"), row=1, col=1)
    ok = r[r["count"] > 0]
    fig.add_scatter(x=ok["mean_p"], y=ok["obs_freq"], mode="lines+markers", name="model",
                    line=dict(color=C_STRAT), marker=dict(size=np.clip(np.sqrt(ok["count"]) / 2 + 5, 6, 22)),
                    customdata=ok["count"], hovertemplate="said %{x:.1%}, happened %{y:.1%}<br>%{customdata} days<extra></extra>",
                    row=1, col=1)
    fig.add_bar(x=(r["lo"] + r["hi"]) / 2, y=r["count"], name="days in bin", marker_color="#5B6578",
                width=0.08, row=2, col=1)
    fig.update_xaxes(range=[0, 1], tickformat=".0%", row=2, col=1, title="Probability the model gave")
    fig.update_yaxes(range=[0, 1], tickformat=".0%", title="How often it went up", row=1, col=1)
    fig.update_yaxes(title="Days", row=2, col=1)
    return style(fig, height, "Reliability: when the model says X%, how often does the market go up?")


def decile_fig(dec: pd.DataFrame, base_hit: float, height=300) -> go.Figure:
    fig = go.Figure(go.Bar(x=dec["decile"], y=dec["hit_rate"], marker_color=C_STRAT, name="hit rate",
                           customdata=np.c_[dec["n"], dec["mean_confidence"] * 100],
                           hovertemplate="decile %{x}: right %{y:.1%} of %{customdata[0]} days<br>"
                                         "average confidence %{customdata[1]:.2f} points<extra></extra>"))
    fig.add_hline(y=0.5, line=dict(color="#8A93A3", dash="dot"), annotation_text="coin flip 50%",
                  annotation_font=dict(size=10))
    fig.add_hline(y=base_hit, line=dict(color=C_VT, dash="dot"), annotation_text="always guessing 'up'",
                  annotation_position="top left", annotation_font=dict(size=10, color=C_VT))
    fig.update_xaxes(title="Confidence decile (1 = least sure, 10 = most sure)", dtick=1)
    fig.update_yaxes(title="Up/down call was right", tickformat=".0%", range=[0.3, 0.75])
    return style(fig, height, "Hit rate by confidence decile", legend_below=False)


def rolling_fig(auc: pd.Series, skill: pd.Series, skill_eval: pd.Series | None = None, height=340) -> go.Figure:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                        subplot_titles=("Rolling 250-day AUC (0.5 = coin flip)",
                                        "Rolling 250-day Brier skill (0 = no better than quoting a base rate)"))
    fig.add_scatter(x=auc.index, y=auc, mode="lines", name="AUC", line=dict(color=C_STRAT), row=1, col=1)
    fig.add_hline(y=0.5, line=dict(color="#8A93A3", dash="dot"), row=1, col=1)
    fig.add_scatter(x=skill.index, y=skill, mode="lines", name="skill vs training base rate (can be drift)",
                    line=dict(color=C_VT), row=2, col=1)
    if skill_eval is not None:
        fig.add_scatter(x=skill_eval.index, y=skill_eval, mode="lines", name="skill vs evaluation-window base rate",
                        line=dict(color="#5BBF7A", dash="dash"), row=2, col=1)
    fig.add_hline(y=0.0, line=dict(color="#8A93A3", dash="dot"), row=2, col=1)
    fig.update_annotations(font=dict(size=12))
    return style(fig, height, legend_below=skill_eval is not None)


def importance_fig(imp: pd.DataFrame, height=300) -> go.Figure:
    t = imp.sort_values("importance_mean")
    colors = ["#5BBF7A" if v > 0 else "#E5534B" for v in t["importance_mean"]]
    fig = go.Figure(go.Bar(x=t["importance_mean"], y=t["group"], orientation="h", marker_color=colors,
                           error_x=dict(type="data", array=t["importance_se"], color="#8A93A3")))
    fig.add_vline(x=0, line=dict(color="#8A93A3"))
    fig.update_xaxes(title="Extra log loss when scrambled (bars = 1 SE)", tickformat=".4f", exponentformat="none",
                     nticks=6)
    return style(fig, height, "Which signal groups does the model rely on?", legend_below=False)


def confusion_fig(cm: pd.DataFrame, height=260) -> go.Figure:
    z = cm.to_numpy()
    share = z / z.sum() if z.sum() else z
    text = [[f"{z[i, j]}<br>{share[i, j]:.0%}" for j in range(2)] for i in range(2)]
    fig = go.Figure(go.Heatmap(z=z, x=list(cm.columns), y=list(cm.index), text=text, texttemplate="%{text}",
                               colorscale=[[0, "#1B2130"], [1, "#2E6FA8"]], showscale=False))
    fig.update_yaxes(autorange="reversed")
    return style(fig, height, "Confusion matrix (days)", legend_below=False)


# ----------------------------------------------------------------------------- trials charts
def trials_hist_fig(ann_sharpes, best_luck: float | None, n_trials: int | None = None, height=300) -> go.Figure:
    fig = go.Figure(go.Histogram(x=ann_sharpes, nbinsx=30, marker_color="#5B6578", name="configurations"))
    if best_luck is not None and np.isfinite(best_luck):
        fig.add_vline(x=best_luck, line=dict(color=C_VT, dash="dash"),
                      annotation_text=(f"conservative luck bar ~{best_luck:.2f}"
                                       + (f"<br>({n_trials} independent trials)" if n_trials else "")),
                      annotation_font=dict(size=10, color=C_VT), annotation_position="top right")
        vals = np.asarray(ann_sharpes, dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size:
            lo, hi = min(float(vals.min()), best_luck), max(float(vals.max()), best_luck)
            fig.update_xaxes(range=[lo - 0.05 * (hi - lo), hi + 0.45 * (hi - lo)])  # room for the label beside the line
    fig.update_xaxes(title="Annualised Sharpe ratio of each configuration tried")
    fig.update_yaxes(title="Number of configurations")
    return style(fig, height, "All configurations tried (out of sample, after costs)", legend_below=False)


def luck_curve_fig(ns, sharpes_ann, marks: dict, height=300) -> go.Figure:
    fig = go.Figure(go.Scatter(x=ns, y=sharpes_ann, mode="lines", line=dict(color=C_VT, width=2.5),
                               name="best Sharpe from luck"))
    for label, (n, s) in marks.items():
        fig.add_scatter(x=[n], y=[s], mode="markers+text", text=[label], textposition="top left",
                        marker=dict(size=10, color=C_STRAT), showlegend=False)
    fig.update_xaxes(title="Configurations tried (all with zero real edge)", type="log")
    fig.update_yaxes(title="Best Sharpe from luck (annualised)")
    return style(fig, height, "The more you try, the better the best one looks by luck", legend_below=False)


def paper_equity_fig(led: pd.DataFrame, ticker: str, height=300) -> go.Figure:
    t = led[led["asset"] == ticker].sort_values("date")
    fig = go.Figure()
    for col, name, color in (("equity", "Paper strategy", C_STRAT), ("bh_equity", "Buy and hold", C_BH),
                             ("vtbh_equity", "Vol-targeted buy and hold", C_VT)):
        fig.add_scatter(x=t["date"], y=t[col], name=name, mode="lines+markers", line=dict(color=color, width=2),
                        marker=dict(size=8))
    fig.update_yaxes(title="Fake dollars")
    fig.update_xaxes(tickformat="%Y-%m-%d", hoverformat="%Y-%m-%d")
    return style(fig, height, f"{ticker}: paper account (fake money)")
