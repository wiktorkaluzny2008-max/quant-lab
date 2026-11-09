"""Pure metric / significance functions for Stage 3 (numpy/pandas in, numbers out).

Conventions
-----------
* ``y`` is a 0/1 array (1 = next-day return > 0); ``p`` is P(up).
* ``returns`` are SIMPLE per-period returns (net of costs unless stated).
* ``position`` is the exposure that EARNS ``returns[t]`` (already lagged by the
  caller: decided at close t-1). Longs are > 0, flat = 0, shorts < 0.
* Annualisation is passed as ``periods_per_year`` (365 for BTC-USD, 252 ETFs);
  see :func:`periods_per_year`.
* Undefined quantities (zero variance, empty input, no trades) return NaN
  rather than raising; invalid inputs (length mismatch, probabilities outside
  [0,1], missing arguments) raise ValueError.
* Sharpe-type inputs to PSR/DSR are PER-PERIOD (daily), not annualised.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score
from statsmodels.stats.diagnostic import pesaran_timmermann as _sm_pt

EULER_GAMMA = 0.5772156649015329
_EPS = 1e-15


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def periods_per_year(asset: str) -> int:
    """365 for BTC-USD (trades every day), 252 for ETFs."""
    return 365 if asset.upper().startswith("BTC") else 252


def _arr(x, name="x") -> np.ndarray:
    a = np.asarray(x, dtype=float).ravel()
    return a


def _pair(a, b, na="a", nb="b"):
    a, b = _arr(a), _arr(b)
    if a.shape != b.shape:
        raise ValueError(f"{na} and {nb} must have the same length ({a.size} vs {b.size})")
    return a, b


def _clim(p_clim, n) -> np.ndarray:
    c = np.asarray(p_clim, dtype=float)
    if c.ndim == 0:
        return np.full(n, float(c))
    c = c.ravel()
    if c.size != n:
        raise ValueError("p_clim must be a scalar or have the same length as p")
    return c


def _clip(p):
    return np.clip(np.asarray(p, dtype=float), _EPS, 1 - _EPS)


# --------------------------------------------------------------------------
# prediction metrics
# --------------------------------------------------------------------------
def accuracy_vs_base_rate(y, pred_up) -> dict:
    """Accuracy of up/down calls versus the base rate (share of up days).

    ``edge`` = accuracy - base_rate (the spec's comparison). Note that "always
    up" scores exactly ``base_rate``, while the honest no-skill yardstick for a
    model that is sometimes down is ``majority_rate = max(b, 1-b)``; both are
    returned. NaN if empty.
    """
    y, pu = _pair(y, pred_up, "y", "pred_up")
    if y.size == 0:
        return dict(accuracy=np.nan, base_rate=np.nan, edge=np.nan, majority_rate=np.nan)
    acc = float(np.mean((pu > 0.5) == (y > 0.5)))
    b = float(np.mean(y > 0.5))
    return dict(accuracy=acc, base_rate=b, edge=acc - b, majority_rate=max(b, 1 - b))


def roc_auc(y, p) -> float:
    """ROC AUC (sklearn). NaN if y has a single class."""
    y, p = _pair(y, p, "y", "p")
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, p))


def log_loss(y, p) -> float:
    """Mean binary cross-entropy; p clipped to [1e-15, 1-1e-15]. NaN if empty."""
    y, p = _pair(y, p, "y", "p")
    if y.size == 0:
        return float("nan")
    p = _clip(p)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def log_loss_series(y, p) -> np.ndarray:
    """Per-observation log loss (input to :func:`diebold_mariano`)."""
    y, p = _pair(y, p, "y", "p")
    p = _clip(p)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def brier(y, p) -> float:
    y, p = _pair(y, p, "y", "p")
    if y.size == 0:
        return float("nan")
    return float(np.mean((p - y) ** 2))


def brier_skill(y, p, p_clim) -> float:
    """1 - Brier_model / Brier_climatology.

    ``p_clim`` is the TRAINING-window base rate (scalar or per-row array, e.g.
    the expanding-window base rate at each walk-forward step) -- never the
    evaluation base rate. 0 for the climatology itself, <0 if worse. NaN if
    the climatology Brier is 0.
    """
    y, p = _pair(y, p, "y", "p")
    bc = brier(y, _clim(p_clim, y.size))
    if not np.isfinite(bc) or bc == 0:
        return float("nan")
    return float(1 - brier(y, p) / bc)


def log_loss_skill(y, p, p_clim) -> float:
    """1 - LogLoss_model / LogLoss_climatology (same conventions as brier_skill)."""
    y, p = _pair(y, p, "y", "p")
    lc = log_loss(y, _clim(p_clim, y.size))
    if not np.isfinite(lc) or lc == 0:
        return float("nan")
    return float(1 - log_loss(y, p) / lc)


def reliability_curve(p, y, bins: int = 10) -> pd.DataFrame:
    """Equal-width probability bins on [0,1]; columns mean_p, obs_freq, count.

    Empty bins are kept with count 0 and NaN mean_p/obs_freq so bin positions
    stay stable.
    """
    p, y = _pair(p, y, "p", "y")
    if bins < 1:
        raise ValueError("bins must be >= 1")
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1], right=False), 0, bins - 1)
    rows = []
    for b in range(bins):
        m = idx == b
        c = int(m.sum())
        rows.append(dict(bin=b, lo=edges[b], hi=edges[b + 1],
                         mean_p=float(p[m].mean()) if c else np.nan,
                         obs_freq=float(y[m].mean()) if c else np.nan, count=c))
    return pd.DataFrame(rows).set_index("bin")


def hit_rate_by_confidence_decile(p, y, base_rate: float, n_bins: int = 10) -> pd.DataFrame:
    """Hit rate of the up/down call (up iff p >= base_rate) by confidence decile.

    confidence = |p - base_rate|; decile 1 = least confident. Ties are broken
    by order (rank method="first") so constant predictions still split evenly.
    """
    p, y = _pair(p, y, "p", "y")
    if p.size < n_bins:
        raise ValueError("need at least n_bins observations")
    conf = np.abs(p - base_rate)
    pred = p >= base_rate
    hit = pred == (y > 0.5)
    dec = pd.qcut(pd.Series(conf).rank(method="first"), n_bins, labels=False) + 1
    df = pd.DataFrame(dict(decile=dec, conf=conf, hit=hit.astype(float)))
    g = df.groupby("decile")
    return pd.DataFrame(dict(n=g.size(), mean_confidence=g.conf.mean(), hit_rate=g.hit.mean()))


def pesaran_timmermann(y, pred_up) -> dict:
    """Pesaran & Timmermann (1992), "A simple nonparametric test of predictive
    performance", JBES 10(4). Uses statsmodels' implementation (present in the
    installed 0.15.0); verified against a hand computation in the tests.

    S_n = (p̂ - p̂*) / sqrt(v̂ - ŵ), p̂* = p̂_y p̂_z + (1-p̂_y)(1-p̂_z).
    Returns statistic, pvalue (two-sided), share_up_days, share_up_predictions,
    hit_rate. Statistic/pvalue are NaN when undefined (e.g. constant
    predictions or constant outcomes: non-positive variance).
    """
    y, pu = _pair(y, pred_up, "y", "pred_up")
    out = dict(statistic=np.nan, pvalue=np.nan, share_up_days=float(np.mean(y > 0.5)) if y.size else np.nan,
               share_up_predictions=float(np.mean(pu > 0.5)) if y.size else np.nan,
               hit_rate=float(np.mean((pu > 0.5) == (y > 0.5))) if y.size else np.nan)
    if y.size < 2:
        return out
    try:
        r = _sm_pt((y > 0.5).astype(float), (pu > 0.5).astype(float))
    except ValueError:
        return out
    out["statistic"], out["pvalue"] = float(r.statistic), float(r.pvalue)
    return out


# --------------------------------------------------------------------------
# trading metrics
# --------------------------------------------------------------------------
def equity_curve(returns) -> np.ndarray:
    return np.cumprod(1.0 + _arr(returns))


def total_return(returns) -> float:
    r = _arr(returns)
    return float(np.prod(1 + r) - 1) if r.size else float("nan")


def cagr(returns, periods_per_year: int) -> float:
    r = _arr(returns)
    if r.size == 0:
        return float("nan")
    g = np.prod(1 + r)
    if g <= 0:
        return -1.0
    return float(g ** (periods_per_year / r.size) - 1)


def ann_vol(returns, periods_per_year: int) -> float:
    r = _arr(returns)
    if r.size < 2:
        return float("nan")
    return float(np.std(r, ddof=1) * math.sqrt(periods_per_year))


def sharpe(returns, periods_per_year: int = 1) -> float:
    """mean/std(ddof=1) * sqrt(ppy); rf = 0 (spec uses raw returns). NaN if std == 0.
    ``periods_per_year=1`` gives the per-period Sharpe used by PSR/DSR."""
    r = _arr(returns)
    if r.size < 2:
        return float("nan")
    sd = np.std(r, ddof=1)
    if sd < 1e-14:
        return float("nan")
    return float(r.mean() / sd * math.sqrt(periods_per_year))


def sortino(returns, periods_per_year: int) -> float:
    """mean / downside deviation * sqrt(ppy); downside dev = sqrt(mean(min(r,0)^2))
    over ALL periods, target 0. NaN if there are no negative returns."""
    r = _arr(returns)
    if r.size == 0:
        return float("nan")
    dd = math.sqrt(np.mean(np.minimum(r, 0.0) ** 2))
    if dd == 0:
        return float("nan")
    return float(r.mean() / dd * math.sqrt(periods_per_year))


def max_drawdown(returns) -> float:
    """Worst peak-to-trough fall of the equity curve (start equity 1.0), as a
    non-positive fraction. 0 if never below a prior peak."""
    r = _arr(returns)
    if r.size == 0:
        return float("nan")
    eq = np.concatenate([[1.0], np.cumprod(1 + r)])
    peak = np.maximum.accumulate(eq)
    return float(np.min(eq / peak - 1))


def pct_time_invested(position) -> float:
    pos = _arr(position)
    return float(np.mean(np.abs(pos) > 1e-12)) if pos.size else float("nan")


def _trades(position) -> np.ndarray:
    pos = _arr(position)
    return np.abs(np.diff(np.concatenate([[0.0], pos])))


def n_trades(position, tol: float = 1e-12) -> int:
    """Number of days on which the position changes (entering from 0 at the
    start counts, so a buy-and-hold position has 1 trade)."""
    return int(np.sum(_trades(position) > tol))


def turnover(position) -> float:
    """MEAN daily |Δposition| (initial entry from 0 included)."""
    t = _trades(position)
    return float(t.mean()) if t.size else float("nan")


def net_returns(asset_returns, position, cost: float) -> np.ndarray:
    """position*r - cost*|Δposition| (cost as a fraction, e.g. 0.0005)."""
    r, pos = _pair(asset_returns, position, "returns", "position")
    return pos * r - cost * _trades(pos)


def break_even_cost(gross_returns, position) -> float:
    """gross mean daily return / mean daily turnover: the per-unit-turnover
    cost at which net mean return is zero. NaN if no trading."""
    g, pos = _pair(gross_returns, position, "gross_returns", "position")
    t = turnover(pos)
    if not np.isfinite(t) or t == 0:
        return float("nan")
    return float(g.mean() / t)


def sharpe_standard_error(sr: float, n: int, periods_per_year: int = 1,
                          skew: float | None = None, kurt: float | None = None) -> float:
    """Standard error of an (annualised) Sharpe ratio estimate.

    ``sr`` is the PER-PERIOD Sharpe. Result is annualised by sqrt(ppy).
    * skew/kurt None: Lo (2002, "The Statistics of Sharpe Ratios", FAJ) iid
      result  se = sqrt((1 + SR^2/2) / n).
    * with raw skew γ3 and raw kurtosis γ4 (normal = 3): the non-normal
      variant (Mertens 2002; Bailey & Lopez de Prado 2012, the PSR
      denominator)  se = sqrt((1 - γ3 SR + (γ4-1)/4 SR^2) / (n-1)).
    Note Lo's iid correction ignores autocorrelation of returns.
    """
    if n < 2 or not np.isfinite(sr):
        return float("nan")
    if skew is None or kurt is None:
        v = (1 + 0.5 * sr ** 2) / n
    else:
        v = (1 - skew * sr + (kurt - 1) / 4 * sr ** 2) / (n - 1)
    if v < 0:
        return float("nan")
    return float(math.sqrt(v) * math.sqrt(periods_per_year))


def tstat_mean(returns) -> float:
    """t-statistic of the mean return (H0: mean = 0), iid standard error."""
    r = _arr(returns)
    if r.size < 2:
        return float("nan")
    sd = np.std(r, ddof=1)
    if sd < 1e-14:
        return float("nan")
    return float(r.mean() / (sd / math.sqrt(r.size)))


def t_hurdle(returns, hurdle: float = 3.0) -> dict:
    """Compare the mean-return t-stat with the Harvey, Liu & Zhu (2016) t > 3
    hurdle ("...and the Cross-Section of Expected Returns", RFS 29(1))."""
    t = tstat_mean(returns)
    return dict(tstat=t, hurdle=hurdle, passes=bool(np.isfinite(t) and t > hurdle))


# --------------------------------------------------------------------------
# significance
# --------------------------------------------------------------------------
def random_strategy_percentile(returns_asset, position, n: int = 1000, seed: int = 0,
                               method: str = "shift") -> dict:
    """Percentile of the strategy's Sharpe among ``n`` random strategies with the
    same time in market, applied to the same asset returns (gross, no costs;
    pass costless inputs on both sides).

    method="shift" (PRIMARY): random circular shift of the position series.
      Preserves exposure, number and length of position blocks, and the
      autocorrelation of positions, destroying only the alignment with
      returns -- the right null for "is the timing better than random timing
      with the same behaviour". Shifts of 0 are excluded.
    method="iid": random permutation of the positions (same fraction/values,
      but blocks are shattered, so far more trades); a cruder, more generous null.
    Percentile = 100*(P[random < s] + 0.5 P[random == s]). Returns dict with
    percentile, strategy_sharpe, random_sharpes (per-period Sharpe).
    NaN percentile if the strategy has zero variance (e.g. all-flat).
    """
    r, pos = _pair(returns_asset, position, "returns_asset", "position")
    if method not in ("shift", "iid"):
        raise ValueError("method must be 'shift' or 'iid'")
    if r.size < 3:
        raise ValueError("need at least 3 observations")
    rng = np.random.default_rng(seed)
    s = sharpe(pos * r)
    sims = np.empty(n)
    for i in range(n):
        if method == "shift":
            rp = np.roll(pos, int(rng.integers(1, r.size)))
        else:
            rp = rng.permutation(pos)
        sims[i] = sharpe(rp * r)
    valid = sims[np.isfinite(sims)]
    if not np.isfinite(s) or valid.size == 0:
        pct = float("nan")
    else:
        pct = 100.0 * (np.mean(valid < s) + 0.5 * np.mean(valid == s))
    return dict(percentile=float(pct), strategy_sharpe=s, random_sharpes=sims, method=method)


def probabilistic_sharpe_ratio(sr: float, sr_benchmark: float, n: int,
                               skew: float = 0.0, kurt: float = 3.0) -> float:
    """PSR(SR*) = Φ( (SR - SR*) sqrt(n-1) / sqrt(1 - γ3 SR + (γ4-1)/4 SR²) ).

    Bailey & Lopez de Prado (2012), "The Sharpe Ratio Efficient Frontier";
    Bernardini & de Castro (2021, arXiv 2101.07217). ``sr``/``sr_benchmark`` per
    period (daily), n = number of periods, kurt RAW (normal = 3). NaN if the
    variance term is non-positive.
    """
    if n < 2 or not np.isfinite(sr):
        return float("nan")
    v = 1 - skew * sr + (kurt - 1) / 4 * sr ** 2
    if v <= 0:
        return float("nan")
    return float(stats.norm.cdf((sr - sr_benchmark) * math.sqrt(n - 1) / math.sqrt(v)))


def expected_max_sharpe(n_trials: int, sr_var: float) -> float:
    """SR0 = sqrt(Var[SR_hat]) * [ (1-γ_E) Φ⁻¹(1-1/N) + γ_E Φ⁻¹(1-1/(N e)) ]
    -- Bailey & Lopez de Prado (2014), "The Deflated Sharpe Ratio", eq. for the
    expected maximum of N iid null Sharpes. N = 1 gives 0 (no selection)."""
    if n_trials < 1:
        raise ValueError("n_trials must be >= 1")
    if sr_var < 0:
        raise ValueError("sr_var must be >= 0")
    if n_trials == 1:
        return 0.0
    N = float(n_trials)
    return float(math.sqrt(sr_var) * ((1 - EULER_GAMMA) * stats.norm.ppf(1 - 1 / N)
                                      + EULER_GAMMA * stats.norm.ppf(1 - 1 / (N * math.e))))


def deflated_sharpe_ratio(returns, trial_sharpes=None, n_trials: int | None = None,
                          sr_var: float | None = None) -> dict:
    """DSR = PSR(SR0) where SR0 = expected max Sharpe across N null trials
    (Bailey & Lopez de Prado 2014).

    Provide EITHER ``trial_sharpes`` (per-period/daily Sharpes of all trials;
    N = len and Var = sample variance, ddof=1) OR ``n_trials`` together with
    ``sr_var`` (variance of daily trial Sharpes). ``n_trials`` may also be
    given with ``trial_sharpes`` to override N (e.g. if only a subset of
    Sharpes was stored). Strategy Sharpe, skew and raw kurtosis come from
    ``returns``. Returns dict(dsr, sr0, sr, n, n_trials, sr_var). NaN dsr if
    the strategy Sharpe is undefined.
    """
    r = _arr(returns)
    if trial_sharpes is not None:
        ts = _arr(trial_sharpes)
        ts = ts[np.isfinite(ts)]
        if ts.size < 2:
            raise ValueError("need at least 2 finite trial Sharpes to estimate their variance")
        var = float(np.var(ts, ddof=1))
        N = int(n_trials) if n_trials is not None else int(ts.size)
    else:
        if n_trials is None or sr_var is None:
            raise ValueError("give trial_sharpes, or both n_trials and sr_var")
        N, var = int(n_trials), float(sr_var)
    sr0 = expected_max_sharpe(N, var)
    sr = sharpe(r)
    if not np.isfinite(sr):
        return dict(dsr=float("nan"), sr0=sr0, sr=sr, n=r.size, n_trials=N, sr_var=var)
    g3 = float(stats.skew(r))
    g4 = float(stats.kurtosis(r, fisher=False))
    d = probabilistic_sharpe_ratio(sr, sr0, r.size, g3, g4)
    return dict(dsr=d, sr0=sr0, sr=sr, n=r.size, n_trials=N, sr_var=var)


def diebold_mariano(loss_a, loss_b, h: int = 1) -> dict:
    """Diebold & Mariano (1995) test with the Harvey, Leybourne & Newbold
    (1997) small-sample correction.

    d_t = loss_a - loss_b. DM = mean(d)/sqrt(LRV/T) with
    LRV = γ0 + 2 Σ_{k=1}^{h-1} γ_k (rectangular, h-1 lags, autocovariances /T).
    HLN: DM* = DM * sqrt((T + 1 - 2h + h(h-1)/T)/T), compared with Student t,
    T-1 df. Two-sided p-value. Statistic > 0 means A has HIGHER loss (B is
    better). Equivalent to statsmodels' ``diebold_mariano_test(...,
    harvey_adj=True, lags=h-1)`` (verified in tests). NaN if d is constant.
    """
    a, b = _pair(loss_a, loss_b, "loss_a", "loss_b")
    if h < 1:
        raise ValueError("h must be >= 1")
    T = a.size
    if T <= h:
        raise ValueError("need more observations than the horizon")
    d = a - b
    dm = d - d.mean()
    lrv = float(np.dot(dm, dm) / T)
    for k in range(1, h):
        lrv += 2 * float(np.dot(dm[k:], dm[:-k]) / T)
    if lrv <= 1e-24:
        return dict(statistic=float("nan"), pvalue=float("nan"), mean_diff=float(d.mean()))
    dm_stat = d.mean() / math.sqrt(lrv / T)
    adj = (T + 1 - 2 * h + h * (h - 1) / T) / T
    if adj <= 0:
        return dict(statistic=float("nan"), pvalue=float("nan"), mean_diff=float(d.mean()))
    stat = dm_stat * math.sqrt(adj)
    pval = 2 * stats.t.sf(abs(stat), T - 1)
    return dict(statistic=float(stat), pvalue=float(pval), mean_diff=float(d.mean()))


def spa_test(benchmark_returns, strategy_returns: pd.DataFrame, reps: int = 1000,
             block_size: int = 10, seed: int = 0) -> dict:
    """Hansen (2005), "A Test for Superior Predictive Ability", JBES 23(4), via
    arch.bootstrap.SPA with the stationary bootstrap (mean block ``block_size``)
    and a fixed seed. H0: no strategy beats the benchmark. Loss = -return, so
    a small p-value means at least one strategy has higher mean return than the
    benchmark after accounting for the number tried. Returns the *consistent*
    p-value (``pvalue``), plus lower/upper bounds and the best strategy name.
    Inputs are aligned on index when pandas; rows with NaN are dropped.
    """
    from arch.bootstrap import SPA

    if not isinstance(strategy_returns, pd.DataFrame):
        strategy_returns = pd.DataFrame(np.asarray(strategy_returns))
    bench = pd.Series(np.asarray(benchmark_returns, dtype=float).ravel()
                      if not isinstance(benchmark_returns, pd.Series) else benchmark_returns)
    if not isinstance(benchmark_returns, pd.Series):
        bench.index = strategy_returns.index
    df = pd.concat([bench.rename("__bench__"), strategy_returns], axis=1).dropna()
    if len(df) < 20:
        raise ValueError("need at least 20 aligned observations")
    spa = SPA(-df["__bench__"].to_numpy(), -df.drop(columns="__bench__").to_numpy(),
              block_size=block_size, reps=reps, bootstrap="stationary", seed=seed)
    spa.compute()
    pv = spa.pvalues
    means = df.drop(columns="__bench__").mean() - df["__bench__"].mean()
    return dict(pvalue=float(pv["consistent"]), pvalue_lower=float(pv["lower"]),
                pvalue_upper=float(pv["upper"]), best_strategy=str(means.idxmax()),
                best_excess_mean=float(means.max()))
