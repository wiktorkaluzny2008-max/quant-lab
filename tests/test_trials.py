"""Tests for quant.trials (trial log). Synthetic, no network."""
import numpy as np
import pandas as pd
import pytest

from quant import trials as T


def _frames(ticker, n_cfg=3, n=30, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n)
    ids = [T.make_config_id(ticker, "expanding", "logreg", "long_flat", False, m)
           for m in T.M_GRID["long_flat"][:n_cfg]]
    idx, ret = [], []
    for cid in ids:
        net = rng.normal(0.0005, 0.01, n)
        row = T.build_index_row(ticker, "expanding", "logreg", "long_flat", False,
                                float(cid.split("m=")[1]), 0.0005, "{}", dates, net, 252)
        assert row["config_id"] == cid
        idx.append(row)
        ret.append(pd.DataFrame({"config_id": cid, "date": dates, "net_return": net,
                                 "gross_return": net, "position": 1.0}))
    return pd.DataFrame(idx, columns=T.INDEX_COLS), pd.concat(ret, ignore_index=True)


def test_config_ids_are_unique_and_enumerated():
    ids = T.expected_config_ids("SPY", ["expanding"], ["logreg", "rf", "ensemble"])
    assert len(ids) == len(set(ids)) == 3 * 2 * (4 + 3 + 2)  # models x vt x (m grids + adaptive)
    ids2 = T.expected_config_ids("BTC-USD", ["expanding", "rolling4y"], ["logreg"])
    assert len(ids2) == 2 * 18
    assert any(i.endswith("|vt|m=adaptive") for i in ids)
    assert T.make_config_id("SPY", "expanding", "rf", "default_long", True, 0.02) == \
        "SPY|expanding|rf|default_long|vt|m=0.02"


def test_replace_asset_idempotent_and_keeps_other_tickers(tmp_path):
    i1, r1 = _frames("SPY")
    i2, r2 = _frames("QQQ", n_cfg=2, seed=1)
    T.replace_asset("SPY", i1, r1, tmp_path)
    T.replace_asset("QQQ", i2, r2, tmp_path)
    assert T.count_trials(tmp_path) == 5
    a = T.load_trials(tmp_path)
    T.replace_asset("SPY", i1, r1, tmp_path)           # same run again
    T.replace_asset("QQQ", i2, r2, tmp_path)
    pd.testing.assert_frame_equal(a, T.load_trials(tmp_path))
    assert T.count_trials(tmp_path) == 5
    assert not T.load_trials(tmp_path).duplicated(["config_id", "date"]).any()
    # re-running SPY with fewer configs replaces ALL of SPY's old rows, QQQ untouched
    i3, r3 = _frames("SPY", n_cfg=1)
    T.replace_asset("SPY", i3, r3, tmp_path)
    assert T.count_trials(tmp_path) == 3
    idx = T.load_index(tmp_path)
    assert (idx["ticker"] == "QQQ").sum() == 2


def test_columns_and_sharpe_vector(tmp_path):
    i1, r1 = _frames("SPY")
    T.replace_asset("SPY", i1, r1, tmp_path)
    assert list(T.load_trials(tmp_path).columns) == T.RETURN_COLS
    assert list(T.load_index(tmp_path).columns) == T.INDEX_COLS
    s = T.trial_sharpes(tmp_path)
    assert len(s) == T.count_trials(tmp_path) == 3
    assert np.isfinite(T.trial_sharpe_variance(tmp_path))


def test_mismatched_inputs_rejected(tmp_path):
    i1, r1 = _frames("SPY")
    with pytest.raises(ValueError):
        T.replace_asset("SPY", i1, r1[r1["config_id"] != r1["config_id"].iloc[0]], tmp_path)
    with pytest.raises(ValueError):
        T.replace_asset("SPY", pd.concat([i1, i1]), r1, tmp_path)
    i2, r2 = _frames("QQQ")
    T.replace_asset("QQQ", i2, r2, tmp_path)
    clash = i2.copy()   # QQQ ids pushed in as SPY
    with pytest.raises(ValueError):
        T.replace_asset("SPY", clash, r2, tmp_path)


def test_empty_log(tmp_path):
    assert T.count_trials(tmp_path) == 0
    assert len(T.trial_sharpes(tmp_path)) == 0
