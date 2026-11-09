"""Config loader. Paths resolve relative to the project root."""
from pathlib import Path

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REQUIRED_KEYS = ["assets", "extra_tickers", "data_dir", "costs", "threshold",
                 "starting_capital", "holdout_months", "holdout_cutoff_date"]
ASSET_KEYS = ["name", "ticker", "start", "trades_weekends"]
# Optional keys and their defaults (leverage simulator, quant/leverage.py).
DEFAULT_BORROW_RATE = 0.06
DEFAULT_LEVERAGE_STRATEGY = {"window": "expanding", "model": "ensemble", "mode": "long_flat",
                             "vol_targeted": True, "m_setting": "adaptive"}
# Paper trading (quant/paper.py). strategy_mode: long_flat (default) | default_long | sized.
STRATEGY_MODES = ("long_flat", "default_long", "sized")
PAPER_KEYS = {"starting_capital", "strategy_mode", "sized_lambda", "sized_band"}
DEFAULT_SIZED_LAMBDA = 0.25
DEFAULT_SIZED_BAND = 0.1
SIZED_LAMBDA_RANGE = (0.25, 0.5)     # spec: fractional Kelly lambda ~ 0.25-0.5


def _num(x, what, positive=False):
    if isinstance(x, bool) or not isinstance(x, (int, float)) or x != x:
        raise ValueError(f"Config {what} must be a number; got {x!r}")
    if positive and x <= 0:
        raise ValueError(f"Config {what} must be positive; got {x!r}")
    return float(x)


def _normalize_paper(cfg, tickers) -> dict:
    """Validate the optional ``paper_trading`` section and fill defaults.

    The strategy itself is NOT configured here: paper trading reuses the already-pinned Stage 3
    ``leverage_strategy`` (window/model/vol_targeted/m_setting), with ``mode`` = strategy_mode."""
    raw = cfg.get("paper_trading") or {}
    if not isinstance(raw, dict):
        raise ValueError("Config 'paper_trading' must be a mapping")
    given_strategy = raw.get("strategy")        # only accepted when it equals the derived one
    bad = set(raw) - PAPER_KEYS - {"strategy"}
    if bad:
        raise ValueError(f"Config 'paper_trading' has unknown key(s): {', '.join(sorted(bad))}")
    cap = raw.get("starting_capital", cfg["starting_capital"])
    if isinstance(cap, dict):
        extra = set(cap) - set(tickers)
        if extra:
            raise ValueError(f"Config 'paper_trading.starting_capital' has unknown ticker(s): "
                             f"{', '.join(sorted(map(str, extra)))}")
        default = cfg["starting_capital"]
        caps = {t: _num(cap.get(t, default), f"'paper_trading.starting_capital.{t}'", True)
                for t in tickers}
    else:
        caps = {t: _num(cap, "'paper_trading.starting_capital'", True) for t in tickers}
    mode = raw.get("strategy_mode", "long_flat")
    if mode not in STRATEGY_MODES:
        raise ValueError(f"Config 'paper_trading.strategy_mode' must be one of "
                         f"{', '.join(STRATEGY_MODES)}; got {mode!r}")
    lam = _num(raw.get("sized_lambda", DEFAULT_SIZED_LAMBDA), "'paper_trading.sized_lambda'")
    lo, hi = SIZED_LAMBDA_RANGE
    if not lo <= lam <= hi:
        raise ValueError(f"Config 'paper_trading.sized_lambda' must be between {lo} and {hi}; got {lam}")
    band = _num(raw.get("sized_band", DEFAULT_SIZED_BAND), "'paper_trading.sized_band'")
    if not 0 <= band < 1:
        raise ValueError(f"Config 'paper_trading.sized_band' must be in [0, 1); got {band}")
    strat = dict(cfg["leverage_strategy"])
    strat["mode"] = mode
    if given_strategy is not None and given_strategy != strat:
        raise ValueError("Config 'paper_trading.strategy' is derived from 'leverage_strategy' "
                         "(the pinned Stage 3 choice) and strategy_mode; do not set it")
    return {"starting_capital": caps, "strategy_mode": mode, "sized_lambda": lam,
            "sized_band": band, "strategy": strat}


def _normalize_extras(extras) -> list:
    """Each extra becomes {"ticker", "start" (str|None), "optional" (bool)}.
    Bare strings are accepted (start None = earliest asset start, not optional)."""
    out = []
    for i, e in enumerate(extras or []):
        if isinstance(e, str):
            e = {"ticker": e}
        if not isinstance(e, dict) or "ticker" not in e:
            raise ValueError(f"Extra ticker #{i} must be a string or a mapping with 'ticker'")
        start = e.get("start")
        out.append({"ticker": e["ticker"], "start": None if start is None else str(start),
                    "optional": bool(e.get("optional", False))})
    return out


def load_config(path=None) -> dict:
    p = Path(path) if path else PROJECT_ROOT / "config.yaml"
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p}")
    with open(p) as f:
        try:
            cfg = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ValueError(f"Config file {p} is not valid YAML: {e}") from e
    if not isinstance(cfg, dict):
        raise ValueError(f"Config file {p} must contain a YAML mapping")
    missing = [k for k in REQUIRED_KEYS if k not in cfg]
    if missing:
        raise ValueError(f"Config missing required key(s): {', '.join(missing)}")
    if not isinstance(cfg["assets"], list) or not cfg["assets"]:
        raise ValueError("Config 'assets' must be a non-empty list")
    for i, a in enumerate(cfg["assets"]):
        miss = [k for k in ASSET_KEYS if not isinstance(a, dict) or k not in a]
        if miss:
            raise ValueError(f"Asset #{i} missing key(s): {', '.join(miss)}")
        a["start"] = str(a["start"])
    tickers = [a["ticker"] for a in cfg["assets"]]
    costs = cfg["costs"]
    if not isinstance(costs, dict) or set(costs) != set(tickers):
        raise ValueError(f"Config 'costs' keys must match asset tickers {tickers}; "
                         f"got {sorted(costs) if isinstance(costs, dict) else costs!r}")
    br = cfg.get("borrow_rate", DEFAULT_BORROW_RATE)
    if isinstance(br, bool) or not isinstance(br, (int, float)) or br < 0:
        raise ValueError(f"Config 'borrow_rate' must be a non-negative number; got {br!r}")
    cfg["borrow_rate"] = float(br)
    ls = {**DEFAULT_LEVERAGE_STRATEGY, **(cfg.get("leverage_strategy") or {})}
    bad = set(ls) - set(DEFAULT_LEVERAGE_STRATEGY)
    if bad:
        raise ValueError(f"Config 'leverage_strategy' has unknown key(s): {', '.join(sorted(bad))}")
    ls["m_setting"] = str(ls["m_setting"])
    cfg["leverage_strategy"] = ls
    try:
        cfg["holdout_cutoff_date"] = str(pd.Timestamp(str(cfg["holdout_cutoff_date"])).date())
    except (ValueError, TypeError) as e:
        raise ValueError(f"Config 'holdout_cutoff_date' must be a date (YYYY-MM-DD); got "
                         f"{cfg['holdout_cutoff_date']!r}") from e
    if not isinstance(cfg["holdout_months"], int) or cfg["holdout_months"] < 1:
        raise ValueError("Config 'holdout_months' must be a positive integer")
    cfg["paper_trading"] = _normalize_paper(cfg, tickers)
    cfg["extra_tickers"] = _normalize_extras(cfg["extra_tickers"])
    data_dir = Path(cfg["data_dir"])
    cfg["data_dir"] = data_dir if data_dir.is_absolute() else PROJECT_ROOT / data_dir
    cfg["project_root"] = PROJECT_ROOT
    return cfg


def required_extra_tickers(cfg) -> list:
    """Names of non-optional extra tickers (accepts normalized dicts or bare strings)."""
    return [e if isinstance(e, str) else e["ticker"]
            for e in cfg.get("extra_tickers") or []
            if isinstance(e, str) or not e.get("optional")]
