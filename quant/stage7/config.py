"""Loader for config_stage7.yaml (merged with the main config for the pinned hold-out cutoff)."""
from pathlib import Path

import pandas as pd
import yaml

from quant.config import PROJECT_ROOT, load_config

GROUPS = ("us_equity", "us_sector", "international", "bonds", "real_assets", "dollar", "crypto")
UNIVERSE_KEYS = ("ticker", "name", "group", "start", "trades_weekends")
FRED_IDS = ("DTB3", "DFF", "DGS10", "T10Y2Y", "BAMLH0A0HYM2", "T10YIE")


def load_stage7_config(path=None, main_config=None) -> dict:
    p = Path(path) if path else PROJECT_ROOT / "config_stage7.yaml"
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    if not p.exists():
        raise FileNotFoundError(f"Stage 7 config file not found: {p}")
    with open(p) as f:
        try:
            cfg = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ValueError(f"Stage 7 config {p} is not valid YAML: {e}") from e
    if not isinstance(cfg, dict):
        raise ValueError(f"Stage 7 config {p} must contain a YAML mapping")
    for k in ("data_dir", "warmup_days", "costs", "universe", "hourly", "fred"):
        if k not in cfg:
            raise ValueError(f"Stage 7 config missing required key: {k}")
    seen = set()
    for i, a in enumerate(cfg["universe"]):
        miss = [k for k in UNIVERSE_KEYS if not isinstance(a, dict) or k not in a]
        if miss:
            raise ValueError(f"Stage 7 universe entry #{i} missing key(s): {', '.join(miss)}")
        if a["group"] not in GROUPS:
            raise ValueError(f"Stage 7 universe {a['ticker']}: unknown group {a['group']!r}")
        if a["ticker"] in seen:
            raise ValueError(f"Stage 7 universe: duplicate ticker {a['ticker']}")
        seen.add(a["ticker"])
        a["start"] = str(a["start"])
        a["trades_weekends"] = bool(a["trades_weekends"])
        if a["trades_weekends"] != (a["group"] == "crypto"):
            raise ValueError(f"Stage 7 universe {a['ticker']}: trades_weekends must be true exactly for crypto")
    unknown = set(cfg["fred"]["series"]) - set(FRED_IDS)
    if unknown:
        raise ValueError(f"Stage 7 config: unexpected FRED series {sorted(unknown)}")
    for h in cfg["hourly"]["assets"]:
        if h not in seen:
            raise ValueError(f"Stage 7 hourly asset {h} is not in the universe")
    main = main_config or load_config()
    cfg["holdout_cutoff_date"] = main["holdout_cutoff_date"]     # single source of truth
    d = Path(cfg["data_dir"])
    cfg["data_dir"] = d if d.is_absolute() else PROJECT_ROOT / d
    cfg["project_root"] = PROJECT_ROOT
    return cfg


def cost_for(cfg, ticker: str, stress: bool = False) -> float:
    a = next(u for u in cfg["universe"] if u["ticker"] == ticker)
    if a["group"] == "crypto":
        return cfg["costs"]["crypto_stress" if stress else "crypto"]
    return cfg["costs"]["etf"]


def dev_cutoff(cfg) -> pd.Timestamp:
    """Last DEV day (inclusive). Everything later is the final hold-out and must not be used."""
    return pd.Timestamp(cfg["holdout_cutoff_date"])
