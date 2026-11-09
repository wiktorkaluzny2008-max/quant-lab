"""Synthetic paper-trading world for the Stage 5 logic-layer tests (no network, tiny models).

``build(tmp_path, asof)`` writes fake Stage 3 results (ensemble predictions), runs the real Stage 4
``run_daily.run`` on the synthetic prices and returns ``(cfg, ds_by_ticker)``.
"""
import numpy as np
import pandas as pd

from quant import features as F
from quant import paper as P
from quant import run_daily as R
from tests.synth import FAST_BT, TICKERS, make_cfg, make_results, truncate_world, world

PX = world()
ASOF = "2021-10-15"


def build(tmp_path, asof=ASOF, mode="long_flat", model="ensemble", extra_days=()):
    """Config + fake results + a live ledger/state/models written by ``run_daily.run``."""
    cfg = make_cfg(tmp_path, strategy_mode=mode)
    cfg["leverage_strategy"]["model"] = model
    cfg["paper_trading"] = __import__("quant.config", fromlist=["_normalize_paper"])._normalize_paper(
        {**cfg, "paper_trading": {"strategy_mode": mode}}, TICKERS)
    make_results(cfg, models=(model,), config_modes=("long_flat", "default_long"))
    old = P.BT
    P.BT = FAST_BT
    try:
        for day in (*extra_days, asof):
            R.run(cfg, asof=pd.Timestamp(day), prices=truncate_world(PX, day), update=None,
                  out=lambda m: None)
    finally:
        P.BT = old
    return cfg


def dataset(cfg, ticker, asof=ASOF):
    return F.build_dataset(ticker, cfg, truncate_world(PX, asof))
