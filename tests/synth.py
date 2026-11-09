"""Synthetic worlds for the Stage 4 tests (no network, tiny models)."""
import json
import numpy as np
import pandas as pd

from quant import backtest as B
from quant import paper as P
from quant import trials as T
from quant.config import load_config
from tests.test_features import make_prices

TICKERS = ["SPY", "QQQ", "IWM", "BTC-USD"]
CUTOFF = "2021-03-31"
HOLDOUT_MONTHS = 6          # hold-out end = 2021-09-30
FAST_BT = B.BTConfig(min_train_days=150, min_cal=30, adaptive_min_days=60)
STRAT = {"window": "expanding", "model": "logreg", "mode": "long_flat",
         "vol_targeted": True, "m_setting": "adaptive"}


def make_cfg(tmp_path, **paper):
    """Real config.yaml with data_dir moved into tmp and a small synthetic calendar."""
    cfg = load_config()
    cfg["data_dir"] = tmp_path
    cfg["holdout_cutoff_date"] = CUTOFF
    cfg["holdout_months"] = HOLDOUT_MONTHS
    cfg["extra_tickers"] = [{"ticker": t, "start": None, "optional": False}
                            for t in ("^VIX", "^VIX3M", "^VXN")]
    cfg["leverage_strategy"] = dict(STRAT)
    from quant.config import _normalize_paper
    cfg["paper_trading"] = _normalize_paper({**cfg, "paper_trading": dict(paper)}, TICKERS)
    return cfg


def make_results(cfg, seed=0, models=("logreg",), config_modes=("long_flat", "default_long")):
    """Fake Stage 3 result files under <data_dir>/results (dev period only)."""
    rd = cfg["data_dir"] / "results"
    rng = np.random.default_rng(seed)
    rows = []
    for t in TICKERS:
        d = pd.bdate_range("2020-06-01", CUTOFF)
        n = len(d)
        nr = rng.normal(0.0004, 0.01, n)
        y = (nr > 0).astype(float)
        p = np.clip(0.54 + rng.normal(0, 0.02, n), 0.05, 0.95)
        pred = pd.concat([pd.DataFrame({"window": "expanding", "model": m, "date": d, "block": d[0],
                                        "p_up": p, "p_raw": p, "base_rate": 0.54, "y": y,
                                        "next_return": nr}) for m in models], ignore_index=True)
        base = pd.DataFrame({"window": "expanding", "name": "vol_targeted_bh", "date": d,
                             "position": np.clip(rng.uniform(0.3, 1.0, n), 0, 1),
                             "gross_return": 0.0, "net_return": 0.0})
        (rd / t).mkdir(parents=True, exist_ok=True)
        pred.to_parquet(rd / t / "predictions.parquet", index=False)
        base.to_parquet(rd / t / "baselines.parquet", index=False)
        for model in models:
            for mode in config_modes:
                for vt in (False, True):
                    for m in ("adaptive", 0.0):
                        rows.append({"config_id": T.make_config_id(t, "expanding", model, mode, vt, m),
                                     "aliases": "[]"})
    pd.DataFrame(rows).to_parquet(rd / "trials_index.parquet", index=False)
    return rd


def world(seed=0, n=900):
    return make_prices(seed, n)


def truncate_world(px, last_session):
    """Prices as they would have been seen at the close of ``last_session``."""
    out = {}
    for k, v in px.items():
        out[k] = v[v.index <= pd.Timestamp(last_session)]
    return out
