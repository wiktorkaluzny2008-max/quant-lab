"""Stage 5 data-access layer for the dashboard (no UI code, no Streamlit import).

Everything here is a plain function: given the config (``cfg``, default ``load_config()``) it reads
files that Stages 3 and 4 already saved and returns pandas objects / small dicts the pages can draw.
Pages should wrap these in ``st.cache_data`` (the functions are pure and take only hashable-ish
arguments).  Nothing in this module writes anything, except :func:`refresh_data`, which calls the
Stage 4 daily run (that is the point of the "Refresh data" button).

Hold-out rule: every saved *development* series (predictions, baselines, trial returns, leverage
equity, retrain blocks ...) is checked on load; a date after the pinned cutoff
(``holdout_cutoff_date`` in config.yaml) raises ``paper.HoldoutError`` -- nothing from the
24-month hold-out can reach a chart.  Live numbers come only from the paper-trading ledger.

Formulas are NOT re-implemented here: they call ``quant.metrics`` / ``quant.paper`` /
``quant.backtest`` (rolling AUC uses ``metrics.roc_auc``, the hit-rate deciles use
``metrics.hit_rate_by_confidence_decile``, the Sharpe band uses ``metrics.sharpe_standard_error``).
"""
from __future__ import annotations

import io
import json
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from quant import metrics as M
from quant import paper as P
from quant import trials as T
from quant.config import load_config

#: attribute name -> file name under data/results/<TICKER>/
RESULT_FILES = {
    "predictions": "predictions.parquet",
    "prediction_metrics": "prediction_metrics.parquet",
    "raw_prediction_metrics": "raw_prediction_metrics.parquet",
    "trading_metrics": "trading_metrics.parquet",
    "significance": "significance.parquet",
    "importance": "importance.parquet",
    "reliability": "prediction_reliability.parquet",
    "deciles": "prediction_deciles.parquet",
    "leverage": "leverage.parquet",
    "leverage_equity": "leverage_equity.parquet",
    "kelly": "kelly.parquet",
    "retrains": "retrains.parquet",
    "baselines": "baselines.parquet",
    "adaptive_m": "adaptive_m.parquet",
    "random_strategies": "random_strategies.parquet",
}
#: dev-series date column per table (checked against the hold-out cutoff on load)
DATE_COLUMNS = {"predictions": "date", "baselines": "date", "leverage_equity": "date",
                "retrains": "block", "adaptive_m": "block"}
HEADLINE = "headline"
STALE_NOTE = "STALE DATA"


# =========================================================================== config helpers
def _cfg(cfg=None) -> dict:
    return cfg if cfg is not None else load_config()


def results_dir(cfg=None) -> Path:
    return Path(_cfg(cfg)["data_dir"]) / "results"


def cutoff(cfg=None) -> pd.Timestamp:
    return pd.Timestamp(_cfg(cfg)["holdout_cutoff_date"])


def tickers(cfg=None) -> list:
    return [a["ticker"] for a in _cfg(cfg)["assets"]]


def asset_names(cfg=None) -> dict:
    """ticker -> display name (e.g. ``SPY`` -> ``S&P 500``)."""
    return {a["ticker"]: a["name"] for a in _cfg(cfg)["assets"]}


def paper_settings(cfg=None) -> P.PaperSettings:
    return P.PaperSettings.from_config(_cfg(cfg))


def assert_dev(df: pd.DataFrame, column: str, cfg=None, what="results"):
    """Raise ``paper.HoldoutError`` when ``df[column]`` contains a date after the pinned cutoff."""
    if len(df) and column in df.columns:
        P.assert_dev_dates(pd.DatetimeIndex(df[column]), cutoff(cfg), what)


# =========================================================================== saved results
@dataclass
class AssetResults:
    """Everything Stage 3 saved for one asset (development period only)."""
    ticker: str
    tables: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)
    crashes: dict = field(default_factory=dict)

    def __getattr__(self, name):
        try:
            return self.__dict__["tables"][name]
        except KeyError:
            raise AttributeError(name) from None

    def get(self, name, columns=()) -> pd.DataFrame:
        """The table ``name`` or an empty frame with ``columns`` when that file was not saved."""
        df = self.tables.get(name)
        return df if df is not None else pd.DataFrame(columns=list(columns))


def load_asset_results(ticker, cfg=None, windows=None) -> AssetResults:
    """Load the saved Stage 3 result tables of ``ticker`` (each as a DataFrame attribute:
    ``res.predictions``, ``res.metrics`` ...).  Missing optional files are simply absent; a missing
    ``predictions`` file raises ``FileNotFoundError`` with a hint.  Dev-only guard on every series."""
    cfg = _cfg(cfg)
    rd = results_dir(cfg) / ticker
    if not (rd / RESULT_FILES["predictions"]).exists():
        raise FileNotFoundError(f"{ticker}: no saved results in {rd} (run `python -m quant.backtest` first)")
    res = AssetResults(ticker)
    for name, fn in RESULT_FILES.items():
        p = rd / fn
        if p.exists():
            df = pd.read_parquet(p)
            if name in DATE_COLUMNS:
                assert_dev(df, DATE_COLUMNS[name], cfg, f"{ticker}/{fn}")
            res.tables[name] = df
    for key, fn in (("metadata", "metadata.json"), ("crashes", "crashes.json")):
        p = rd / fn
        if p.exists():
            setattr(res, key, json.loads(p.read_text()))
    last = res.metadata.get("dev_last_date")
    if last is not None and pd.Timestamp(last) > cutoff(cfg):
        raise P.HoldoutError(f"{ticker}: metadata dev_last_date {last} is after the hold-out cutoff")
    return res


def load_trial_index(cfg=None) -> pd.DataFrame:
    return T.load_index(results_dir(cfg))


def load_trial_returns(config_id, cfg=None) -> pd.DataFrame:
    """Out-of-sample daily returns of one logged trial (dev only).  ``config_id`` may be an alias."""
    cfg = _cfg(cfg)
    cid = canonical_config_id(load_trial_index(cfg), config_id)
    p = results_dir(cfg) / "trials.parquet"
    if not p.exists():
        raise FileNotFoundError(f"trial log {p} not found (run `python -m quant.backtest` first)")
    df = pd.read_parquet(p, filters=[("config_id", "==", cid)]).sort_values("date")
    if df.empty:
        raise KeyError(f"config {config_id!r} has no rows in the trial log")
    assert_dev(df, "date", cfg, f"trial {cid}")
    return df.reset_index(drop=True)


def canonical_config_id(index: pd.DataFrame, config_id: str) -> str:
    """The logged (canonical) id of a configuration: identical return streams are stored once with
    the other ids as ``aliases``.  Raises ``KeyError`` if the id was never logged."""
    if config_id in set(index["config_id"]):
        return config_id
    for cid, al in zip(index["config_id"], index["aliases"]):
        if al and config_id in json.loads(al):
            return cid
    raise KeyError(f"config ID {config_id!r} is not in the trial log")


def trials_summary(cfg=None) -> dict:
    """Numbers for the Trials page: how many configurations were tried (global and per asset), the
    spread of their Sharpe ratios, and the Deflated Sharpe Ratio of the active strategy per asset
    (both with the per-asset and with the global trial count) next to its plain PSR."""
    cfg = _cfg(cfg)
    idx = load_trial_index(cfg)
    out = {"n_trials_global": int(len(idx)), "per_asset": {}, "assets": {}}
    ann = idx["ann_sharpe"].to_numpy(float)
    ann = ann[np.isfinite(ann)]
    out["annual_sharpe_best"] = float(ann.max()) if ann.size else float("nan")
    out["annual_sharpe_median"] = float(np.median(ann)) if ann.size else float("nan")
    out["daily_sharpe_variance"] = T.trial_sharpe_variance(results_dir(cfg))
    for t in tickers(cfg):
        sub = idx[idx["ticker"] == t]
        out["per_asset"][t] = int(len(sub))
        entry = {"n_trials_asset": int(len(sub))}
        try:
            res = load_asset_results(t, cfg)
            cid = paper_settings(cfg).config_id(t)
            canon = canonical_config_id(idx, cid)
            sig = res.significance
            rows = sig[sig["subject"] == canon]
            for test in ("psr", "dsr"):
                for _, r in rows[rows["test"] == test].iterrows():
                    scope = "global" if r["reference"] == "global" else ("asset" if test == "dsr" else "")
                    entry[f"{test}{'_' + scope if scope else ''}"] = float(r["value"])
                    entry["daily_sharpe"] = float(r["statistic"])
                    if test == "dsr":
                        det = json.loads(r["details"])
                        entry[f"n_trials_{scope}"] = int(det.get("n_trials", 0))
                        entry[f"sr0_{scope}"] = float(det.get("sr0", float("nan")))    # daily luck bar
            ex = rows[rows["test"] == "dsr_excess_vt"]
            if len(ex):
                entry["dsr_excess_vt"] = float(ex.iloc[0]["value"])
            entry["config_id"] = cid
        except (FileNotFoundError, KeyError):
            pass
        out["assets"][t] = entry
    return out


def _mtime(p: Path) -> int:
    try:
        return p.stat().st_mtime_ns
    except OSError:
        return 0


def data_signature(cfg=None) -> tuple:
    """Changes whenever the ledger, state, saved live models (data/paper/models/*.joblib) or result
    files change on disk, so the dashboard's cached loaders refresh after a daily run (from the button
    or from the command line)."""
    cfg = _cfg(cfg)
    data = Path(cfg["data_dir"])
    rd = results_dir(cfg)
    paths = [data / "paper" / "ledger.parquet", data / "paper" / "state.json",
             rd / "trials_index.parquet", rd / "trials.parquet"]
    for t in tickers(cfg):
        paths += [rd / t / "predictions.parquet", rd / t / "trading_metrics.parquet",
                  data / "paper" / "models" / f"{t}.joblib", data / f"{t}.parquet"]
    return tuple(_mtime(p) for p in paths)


# =========================================================================== ledger / freshness
def load_ledger(cfg=None) -> pd.DataFrame:
    """The whole live paper-trading ledger (read-only)."""
    return P.load_ledger(P.PaperFiles.for_config(_cfg(cfg)))


def ledger_for(ticker, cfg=None) -> pd.DataFrame:
    led = load_ledger(cfg)
    return led[led["asset"] == ticker].sort_values("date").reset_index(drop=True)


def load_state(cfg=None) -> dict:
    return P.load_state(P.PaperFiles.for_config(_cfg(cfg)))


def latest_price_date(ticker, cfg=None):
    p = Path(_cfg(cfg)["data_dir"]) / f"{ticker}.parquet"
    if not p.exists():
        return None
    idx = pd.read_parquet(p, columns=["Close"]).index
    return pd.Timestamp(idx.max()) if len(idx) else None


def data_freshness(cfg=None, now=None) -> dict:
    """Per asset: last price date, last ledger date, the latest date that should exist now, and a
    plain-language ``warning`` (``None`` when fresh).  ``stale`` is True if the stored prices lag
    the most recent completed day or the ledger lags the stored prices."""
    from quant import run_daily as R
    cfg = _cfg(cfg)
    led = load_ledger(cfg)
    out = {}
    for a in cfg["assets"]:
        t = a["ticker"]
        expected = R.usable_last_date(bool(a["trades_weekends"]), now)
        px = latest_price_date(t, cfg)
        lt = led[led["asset"] == t]
        last_led = pd.Timestamp(lt["date"].max()) if len(lt) else None
        msgs = []
        if px is None:
            msgs.append("no price data stored")
        elif px < expected:
            msgs.append(f"{STALE_NOTE}: prices end {px.date()} but the latest completed day is "
                        f"{expected.date()}")
        if last_led is None:
            msgs.append("no paper-trading decision recorded yet")
        elif last_led < expected:
            msgs.append(f"{STALE_NOTE}: the latest recorded decision is for {last_led.date()}; "
                        f"{expected.date()} is available - press Refresh data")
        out[t] = dict(ticker=t, last_price_date=px, last_ledger_date=last_led, expected_date=expected,
                      stale=bool(msgs), warning="; ".join(msgs) if msgs else None)
    return out


@dataclass
class RefreshOutcome:
    ok: bool
    message: str
    lines: list = field(default_factory=list)


def refresh_data(cfg=None, *, run=None, **run_kw) -> RefreshOutcome:
    """The "Refresh data" button: runs the Stage 4 daily update (download new prices, catch up the
    paper-trading ledger).  Never raises and never returns a traceback: failures come back as
    ``ok=False`` with a one-line message.  ``run`` / ``run_kw`` exist so tests can inject a fake
    run or synthetic prices; the default is ``quant.run_daily.run`` (idempotent: running it twice
    in a day writes nothing the second time)."""
    from quant import run_daily as R
    run = run or R.run
    try:
        cfg = _cfg(cfg)
    except Exception as e:      # noqa: BLE001 - one clean line
        return RefreshOutcome(False, f"Could not read config.yaml: {e}")
    buf = io.StringIO()
    try:
        with redirect_stdout(buf), redirect_stderr(buf):
            res = run(cfg, out=lambda m: None, **run_kw)
    except (P.PaperError, RuntimeError, ValueError, KeyError, OSError) as e:
        msg = e.args[0] if isinstance(e, KeyError) and e.args else str(e)
        return RefreshOutcome(False, f"Refresh failed: {msg}")
    except Exception as e:      # noqa: BLE001 - no tracebacks in the UI
        return RefreshOutcome(False, f"Refresh failed unexpectedly ({type(e).__name__}: {e})")
    lines = list(getattr(res, "messages", []))
    skipped = dict(getattr(res, "skipped", {}) or {})
    if skipped:
        why = "; ".join(f"{t}: {r}" for t, r in skipped.items())
        n_new = 0 if getattr(res, "new_rows", None) is None else len(res.new_rows)
        return RefreshOutcome(False, f"Only partly refreshed: {len(skipped)} asset(s) were skipped - {why} "
                                     f"({n_new} new paper-trading row(s) recorded for the others).", lines)
    if getattr(res, "wrote", False):
        n = 0 if res.new_rows is None else len(res.new_rows)
        return RefreshOutcome(True, f"Refreshed: {n} new paper-trading row(s) recorded.", lines)
    return RefreshOutcome(True, "Already up to date - nothing new to record.", lines)


# =========================================================================== prediction helpers
def model_predictions(res: AssetResults, model="ensemble", window="expanding") -> pd.DataFrame:
    """Out-of-sample predictions of one model/window, sorted by date (dev only)."""
    d = res.predictions
    d = d[(d["model"] == model) & (d["window"] == window)].sort_values("date")
    return d.reset_index(drop=True)


def rolling_auc(preds: pd.DataFrame, window: int = 250) -> pd.Series:
    """Rolling ``window``-day ROC AUC of ``p_up`` vs ``y`` (``metrics.roc_auc``), indexed by date.
    NaN while the window is not full or contains one class only."""
    p, y = preds["p_up"].to_numpy(float), preds["y"].to_numpy(float)
    out = np.full(len(p), np.nan)
    for i in range(window - 1, len(p)):
        ys = y[i - window + 1: i + 1]
        if 0 < ys.sum() < window:
            out[i] = M.roc_auc(ys, p[i - window + 1: i + 1])
    return pd.Series(out, index=pd.DatetimeIndex(preds["date"]), name=f"auc_{window}")


def rolling_brier_skill(preds: pd.DataFrame, window: int = 250) -> pd.Series:
    """Rolling ``window``-day Brier skill vs the (per-day) training base rate:
    ``1 - mean((p-y)^2) / mean((base-y)^2)`` -- the same quantity as ``metrics.brier_skill`` applied
    to each window (a test checks that)."""
    p, y, b = (preds[c].to_numpy(float) for c in ("p_up", "y", "base_rate"))
    s = pd.Series((p - y) ** 2).rolling(window, min_periods=window).mean()
    c = pd.Series((b - y) ** 2).rolling(window, min_periods=window).mean()
    out = (1.0 - s / c.where(c > 0)).to_numpy()
    return pd.Series(out, index=pd.DatetimeIndex(preds["date"]), name=f"brier_skill_{window}")


def rolling_brier_skill_eval(preds: pd.DataFrame, window: int = 250) -> pd.Series:
    """Rolling ``window``-day Brier skill against the base rate of the SAME window (the share of up
    days inside it): ``metrics.brier_skill(y, p, mean(y))`` per window.  Unlike the skill against the
    training base rate it cannot look positive from base-rate drift alone."""
    p, y = preds["p_up"].to_numpy(float), preds["y"].to_numpy(float)
    s = pd.Series((p - y) ** 2).rolling(window, min_periods=window).mean()
    ybar = pd.Series(y).rolling(window, min_periods=window).mean()
    # mean((ybar_w - y)^2) over the window = ybar_w * (1 - ybar_w) exactly (y is 0/1)
    c = ybar * (1.0 - ybar)
    out = (1.0 - s / c.where(c > 0)).to_numpy()
    return pd.Series(out, index=pd.DatetimeIndex(preds["date"]), name=f"brier_skill_eval_{window}")


def confusion_matrix(preds: pd.DataFrame) -> pd.DataFrame:
    """2x2 counts: rows = what happened (down/up), columns = what the model said (down/up, i.e.
    p < base_rate / p >= base_rate).  Also carries ``share`` columns for the pages."""
    said_up = preds["p_up"].to_numpy(float) >= preds["base_rate"].to_numpy(float)
    was_up = preds["y"].to_numpy(float) > 0.5
    cm = pd.DataFrame({"model said down": [int((~said_up & ~was_up).sum()), int((~said_up & was_up).sum())],
                       "model said up": [int((said_up & ~was_up).sum()), int((said_up & was_up).sum())]},
                      index=["market went down", "market went up"])
    return cm


def sharpe_band(trading_row, k: float = 2.0) -> dict:
    """Annualised Sharpe with a +-``k`` standard-error band, from a saved ``trading_metrics`` row
    (``sharpe`` and ``sharpe_se`` as computed by ``metrics.sharpe_standard_error``)."""
    sr, se = float(trading_row["sharpe"]), float(trading_row["sharpe_se"])
    return dict(sharpe=sr, se=se, low=sr - k * se, high=sr + k * se, k=k)


def drawdown_series(returns) -> pd.Series:
    """Drawdown from the running peak of the equity curve (<= 0), same convention as
    ``metrics.max_drawdown`` (whose minimum it equals)."""
    r = pd.Series(returns, dtype="float64")
    eq = pd.Series(M.equity_curve(r.to_numpy()), index=r.index)
    peak = np.maximum.accumulate(np.r_[1.0, eq.to_numpy()])[1:]
    return (eq / peak - 1.0).rename("drawdown")


# =========================================================================== backtest page
def strategy_config_id(ticker, cfg=None) -> str:
    """Config ID of the active paper-trading strategy (``paper.PaperSettings.config_id``)."""
    return paper_settings(cfg).config_id(ticker)


def equity_curves(ticker, config_id, cfg=None, window="expanding") -> pd.DataFrame:
    """Out-of-sample equity (start 1.0) of one logged strategy NET of headline costs next to
    buy-and-hold and vol-targeted buy-and-hold (saved ``baselines.parquet`` net returns), plus
    their drawdowns.  Columns: ``strategy, buy_and_hold, vol_targeted_bh`` and
    ``dd_<same>``.  Indexed by date; dev only."""
    cfg = _cfg(cfg)
    res = load_asset_results(ticker, cfg)
    base = res.baselines
    base = base[base["window"] == window]
    tr = load_trial_returns(config_id, cfg).set_index("date")["net_return"]
    cols = {"strategy": tr}
    for name in ("buy_and_hold", "vol_targeted_bh"):
        b = base[base["name"] == name].sort_values("date").set_index("date")["net_return"]
        cols[name] = b
    rets = pd.DataFrame(cols).dropna()
    out = pd.DataFrame(index=rets.index)
    for c in rets:
        out[c] = M.equity_curve(rets[c].to_numpy())
        out[f"dd_{c}"] = drawdown_series(rets[c].to_numpy()).to_numpy()
    return out


def trading_row(res: AssetResults, name, window="expanding", scenario=HEADLINE):
    """One ``trading_metrics`` row (a trial config ID or a baseline name); ``None`` if absent."""
    tm = res.get("trading_metrics", ("name", "window", "scenario"))
    sub = tm[(tm["name"] == name) & (tm["window"] == window) & (tm["scenario"] == scenario)]
    return None if sub.empty else sub.iloc[0]


def metrics_table(res: AssetResults, window="expanding", mode=None, vol_targeted=None,
                  m_setting=None, scenario=HEADLINE) -> pd.DataFrame:
    """Metrics table comparing every model (for the chosen mode/vol-target/m) and the baselines."""
    tm = res.trading_metrics
    tm = tm[(tm["window"] == window) & (tm["scenario"] == scenario)]
    base = tm[tm["kind"] == "baseline"]
    tri = tm[tm["kind"] == "trial"]
    if mode is not None:
        tri = tri[tri["mode"] == mode]
    if vol_targeted is not None:
        tri = tri[tri["vol_targeted"] == bool(vol_targeted)]
    if m_setting is not None:
        tri = tri[tri["m_setting"] == str(m_setting)]
    return pd.concat([tri, base], ignore_index=True)


def random_percentile(res: AssetResults, config_id, cfg=None) -> float | None:
    """Percentile (0-100) of the strategy among the saved 1,000 random same-exposure strategies."""
    canon = canonical_config_id(load_trial_index(cfg), config_id)
    sig = res.significance
    r = sig[(sig["test"] == "random_percentile") & (sig["subject"] == canon)]
    return None if r.empty else float(r.iloc[0]["value"])


def oos_sample_years(res: AssetResults, cfg=None, model="ensemble", window="expanding") -> float:
    """Length of the out-of-sample period in years (days / 365 for assets that trade weekends, else / 252)."""
    n = len(model_predictions(res, model, window))
    weekend = {a["ticker"]: bool(a["trades_weekends"]) for a in _cfg(cfg)["assets"]}.get(res.ticker, False)
    return n / (365.0 if weekend else 252.0)


def sharpe_needed_for_t(years: float, t: float = 3.0) -> float:
    """Annual Sharpe at which the mean-return t-statistic reaches ``t`` after ``years`` years (t = SR * sqrt(years))."""
    return float(t / np.sqrt(years)) if years > 0 else float("nan")


def vt_baseline_percentile(res: AssetResults, name="vol_targeted_bh", window="expanding", cfg=None,
                           model="ensemble") -> float | None:
    """Random-strategy percentile (0-100) of a BASELINE (default: vol-targeted buy-and-hold), computed
    exactly like the saved ``random_percentile`` of the strategies (same 1,000 circular shifts, same seed,
    GROSS returns of the asset), so the two numbers can be read side by side.  ``None`` if not computable.
    Vol-targeted series mostly rank high because the shift scrambles the volatility alignment."""
    from quant.backtest import BTConfig
    bt = BTConfig()
    b = res.baselines
    b = b[(b["name"] == name) & (b["window"] == window)].sort_values("date")
    p = model_predictions(res, model, window).set_index("date")["next_return"]
    if b.empty or p.empty:
        return None
    nr = p.reindex(b["date"]).to_numpy(float)
    if not np.isfinite(nr).all() or len(nr) < 3:
        return None
    v = M.random_strategy_percentile(nr, b["position"].to_numpy(float), n=bt.n_random, seed=bt.seed)["percentile"]
    return float(v) if np.isfinite(v) else None


# =========================================================================== leverage page
def leverage_table(res: AssetResults, series="chosen_strategy") -> pd.DataFrame:
    """Saved leverage results (1x/1.5x/2x/3x) of one series, with the "wiped out" flag."""
    lv = res.leverage
    return lv[lv["series"] == series].sort_values("L").reset_index(drop=True)


def leverage_equity_curves(res: AssetResults, series="chosen_strategy") -> pd.DataFrame:
    """Leveraged equity curves (columns = L), indexed by date; dev only."""
    le = res.leverage_equity
    le = le[le["series"] == series]
    return le.pivot(index="date", columns="L", values="equity").sort_index()
