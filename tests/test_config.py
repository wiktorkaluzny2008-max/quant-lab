import pytest
import yaml

from quant.config import PROJECT_ROOT, load_config


def test_default_load():
    cfg = load_config()
    assert [a["ticker"] for a in cfg["assets"]] == ["SPY", "QQQ", "IWM", "BTC-USD"]
    assert cfg["assets"][3]["trades_weekends"] is True
    assert cfg["assets"][0]["start"] == "1999-01-01"
    assert cfg["data_dir"] == PROJECT_ROOT / "data"
    assert cfg["threshold"] == 0.55 and cfg["holdout_months"] == 24
    assert cfg["holdout_cutoff_date"] == "2024-11-06"


def test_missing_key(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump({"assets": [{"name": "x"}]}))
    with pytest.raises(ValueError, match="missing required"):
        load_config(p)


def test_bad_asset(tmp_path):
    cfg = load_config()
    cfg = {k: v for k, v in cfg.items() if k != "project_root"}
    cfg["data_dir"] = "data"
    cfg["assets"] = [{"name": "x"}]
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump(cfg, default_flow_style=False))
    with pytest.raises(ValueError, match="Asset #0 missing"):
        load_config(p)


def test_cost_keys_match_tickers():
    assert set(load_config()["costs"]) == {"SPY", "QQQ", "IWM", "BTC-USD"}


def test_cost_keys_mismatch(tmp_path):
    cfg = load_config()
    cfg = {k: v for k, v in cfg.items() if k != "project_root"}
    cfg["data_dir"] = "data"
    cfg["costs"] = {"spy": 0.0005, "qqq": 0.0005, "btc": 0.0015}
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump(cfg, default_flow_style=False))
    with pytest.raises(ValueError, match="costs"):
        load_config(p)


def test_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "nope.yaml")


def test_iwm_asset_and_cost():
    cfg = load_config()
    iwm = next(a for a in cfg["assets"] if a["ticker"] == "IWM")
    assert iwm["start"] == "2000-05-26" and iwm["trades_weekends"] is False
    assert cfg["costs"]["IWM"] == 0.0005


def test_extras_config():
    ex = {e["ticker"]: e for e in load_config()["extra_tickers"]}
    assert set(ex) == {"^VIX", "^VIX3M", "^VXN", "^RVX"}
    assert ex["^VIX3M"]["start"] == "2007-01-01"
    assert ex["^RVX"]["optional"] is True and not ex["^VIX"]["optional"]


def test_bare_string_extras_normalized(tmp_path):
    cfg = {k: v for k, v in load_config().items() if k != "project_root"}
    cfg["data_dir"] = "data"
    cfg["extra_tickers"] = ["^VIX"]
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump(cfg))
    assert load_config(p)["extra_tickers"] == [{"ticker": "^VIX", "start": None, "optional": False}]


def test_borrow_rate_and_leverage_strategy_defaults(tmp_path):
    cfg = load_config()
    assert cfg["borrow_rate"] == 0.06
    assert cfg["leverage_strategy"] == {"window": "expanding", "model": "ensemble",
                                        "mode": "long_flat", "vol_targeted": True,
                                        "m_setting": "adaptive"}
    raw = {k: v for k, v in cfg.items() if k not in ("project_root", "borrow_rate", "leverage_strategy")}
    raw["data_dir"] = "data"
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump(raw))
    assert load_config(p)["borrow_rate"] == 0.06     # key is optional


def test_bad_borrow_rate(tmp_path):
    raw = {k: v for k, v in load_config().items() if k not in ("project_root", "leverage_strategy")}
    raw["data_dir"] = "data"
    raw["borrow_rate"] = -0.01
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="borrow_rate"):
        load_config(p)


def _write_cfg(tmp_path, **over):
    cfg = {k: v for k, v in load_config().items() if k not in ("project_root",)}
    cfg["data_dir"] = "data"
    cfg.update(over)
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump(cfg, default_flow_style=False))
    return p


def test_pinned_cutoff_required_and_validated(tmp_path):
    p = _write_cfg(tmp_path)
    assert load_config(p)["holdout_cutoff_date"] == "2024-11-06"
    cfg = {k: v for k, v in load_config().items() if k not in ("project_root", "holdout_cutoff_date")}
    cfg["data_dir"] = "data"
    q = tmp_path / "d.yaml"
    q.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match="missing required"):
        load_config(q)
    with pytest.raises(ValueError, match="holdout_cutoff_date"):
        load_config(_write_cfg(tmp_path, holdout_cutoff_date="not-a-date"))


# ---------------------------------------------------------------- paper_trading (Stage 4)
def _cfg_with_paper(tmp_path, paper):
    cfg = load_config()
    cfg = {k: v for k, v in cfg.items() if k not in ("project_root", "paper_trading")}
    cfg["data_dir"] = "data"
    cfg["extra_tickers"] = [{"ticker": e["ticker"], "start": e["start"], "optional": e["optional"]}
                            for e in cfg["extra_tickers"]]
    if paper is not None:
        cfg["paper_trading"] = paper
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump(cfg, default_flow_style=False))
    return p


def test_paper_defaults_reuse_the_pinned_leverage_strategy(tmp_path):
    cfg = load_config(_cfg_with_paper(tmp_path, None))
    pt = cfg["paper_trading"]
    assert pt["strategy_mode"] == "long_flat" and pt["sized_band"] == 0.1
    assert pt["starting_capital"] == {t: 10000.0 for t in ("SPY", "QQQ", "IWM", "BTC-USD")}
    assert pt["strategy"] == {**cfg["leverage_strategy"], "mode": "long_flat"}
    assert pt["strategy"] == {"window": "expanding", "model": "ensemble", "mode": "long_flat",
                              "vol_targeted": True, "m_setting": "adaptive"}


def test_paper_config_in_repo_config_yaml():
    pt = load_config()["paper_trading"]
    assert pt["strategy_mode"] == "long_flat" and pt["sized_lambda"] == 0.25


def test_paper_per_asset_capital_and_modes(tmp_path):
    cfg = load_config(_cfg_with_paper(tmp_path, {"starting_capital": {"SPY": 5000},
                                                 "strategy_mode": "sized", "sized_lambda": 0.5}))
    pt = cfg["paper_trading"]
    assert pt["starting_capital"]["SPY"] == 5000.0 and pt["starting_capital"]["BTC-USD"] == 10000.0
    assert pt["strategy"]["mode"] == "sized"
    cfg = load_config(_cfg_with_paper(tmp_path, {"strategy_mode": "default_long"}))
    assert cfg["paper_trading"]["strategy"]["mode"] == "default_long"


@pytest.mark.parametrize("paper,needle", [
    ({"strategy_mode": "yolo"}, "strategy_mode"),
    ({"sized_lambda": 0.9}, "sized_lambda"),
    ({"sized_lambda": 0.1}, "sized_lambda"),
    ({"sized_band": 1.5}, "sized_band"),
    ({"sized_band": -0.1}, "sized_band"),
    ({"starting_capital": 0}, "starting_capital"),
    ({"starting_capital": -5}, "starting_capital"),
    ({"starting_capital": "lots"}, "starting_capital"),
    ({"starting_capital": {"NOPE": 100}}, "unknown ticker"),
    ({"leverage": 2}, "unknown key"),
    ("long_flat", "mapping"),
])
def test_paper_config_validation(tmp_path, paper, needle):
    with pytest.raises(ValueError, match=needle):
        load_config(_cfg_with_paper(tmp_path, paper))
