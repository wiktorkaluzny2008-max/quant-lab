"""Stage 4 daily run (FAKE money, no broker):

    uv run python -m quant.run_daily [--asof YYYY-MM-DD] [--assets SPY BTC-USD] [--dry-run]

update data -> (retrain if a new calendar month started) -> predict the NEXT day -> record the
intended position at today's close in ``data/paper/ledger.parquet`` -> mark the previous
positions to market (realised return, costs on position changes).

Safe to run twice (the second run writes nothing) and to run after missed days (every missed
trading day, every calendar day for BTC, is processed in order with only the information of
that day).  No scheduled job is created by this program.

Exit code 0 = the run behaved correctly (including "an asset was skipped because its data is
stale or missing"; that is reported on screen).  Exit code 2 = a real error, reported as a
single ``ERROR:`` line (bad config, corrupt ledger, would-be hold-out violation, ...).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from datetime import timedelta

import pandas as pd

from quant import backtest as B
from quant import data as D
from quant import features as F
from quant import paper as P
from quant.config import load_config

WARN_BAR = "!" * 70


@dataclass
class RunResult:
    exit_code: int = 0
    messages: list = field(default_factory=list)     # lines printed (also returned for tests)
    skipped: dict = field(default_factory=dict)      # ticker -> reason
    new_rows: pd.DataFrame = None
    wrote: bool = False


def usable_last_date(trades_weekends: bool, now) -> pd.Timestamp:
    """Latest bar that is complete at ``now``: yesterday (UTC) for BTC, last closed NYSE session
    for ETFs and the VIX family."""
    now = D._now_utc(now)
    if trades_weekends:
        return pd.Timestamp(now.date()) - timedelta(days=1)
    return pd.Timestamp(D.last_completed_session(now))


def resolve_times(asof, now=None):
    """(asof date, effective 'now').  Without --asof: the real clock.  With --asof: the end of
    that day, but never later than the real clock (the future cannot be used)."""
    real = D._now_utc(now)
    if asof is None:
        return pd.Timestamp(real.date()), real
    a = pd.Timestamp(asof).normalize()
    end = (a + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)).tz_localize("UTC")
    return a, min(real, end)


def _truncate_prices(cfg, px, now_eff):
    """Cut every price frame at the last bar that is complete at the as-of time, BEFORE any
    feature is built, so nothing after the as-of date can reach a feature or a label."""
    out = {}
    btc = {a["ticker"] for a in cfg["assets"] if a["trades_weekends"]}
    for t, df in px.items():
        cut = usable_last_date(t in btc, now_eff)
        out[t] = df[df.index <= cut]
    return out


# Series (besides the asset itself and the required extras) whose features it uses.
INPUTS = {"IWM": ("SPY",), "BTC-USD": ("SPY", "QQQ", "IWM")}


def stale_inputs(cfg, prices, ticker, etf_expected) -> list:
    """Input series (required extras such as ^VIX, plus the assets this one's cross-asset features
    read) whose latest bar is older than the last completed NYSE session.  The as-of join would
    silently carry them forward, so a stale one must stop the new trade instead."""
    need = [e["ticker"] for e in cfg["extra_tickers"] if not e["optional"]] + list(INPUTS.get(ticker, ()))
    out = []
    for n in need:
        df = prices.get(n)
        last = None if df is None or len(df) == 0 else df.index.max()
        if last is None or last < etf_expected:
            out.append(f"{n} (latest {'none' if last is None else last.date()}, need "
                       f"{etf_expected.date()})")
    return out


def _fmt_row(r) -> str:
    pos = r["position"]
    state = "flat" if pos < 1e-9 else ("fully invested" if pos > 1 - 1e-9 else f"{pos:.0%} invested")
    brier = "" if r["live_n"] == 0 else (f"  live Brier {r['live_brier']:.4f} (backtest {r['bt_brier']:.4f}), "
                                       f"hit rate {r['live_hit_rate']:.1%} (backtest {r['bt_hit_rate']:.1%})")
    return (f"{r['asset']:8s} {pd.Timestamp(r['date']).date()}  P(up) {r['p_up']:.3f} vs base rate "
            f"{r['base_rate']:.3f} -> position {pos:.2f} ({state}); trades {int(r['n_trades'])}, "
            f"equity ${r['equity']:,.2f} (B&H ${r['bh_equity']:,.2f}, vol-targeted B&H "
            f"${r['vtbh_equity']:,.2f}){brier}")


def run(cfg, *, asof=None, assets=None, dry_run=False, update="default", now=None, prices=None,
        out=print) -> RunResult:
    res = RunResult()

    def say(msg):
        res.messages.append(msg)
        out(msg)

    settings = P.PaperSettings.from_config(cfg)
    B.require_models(settings.model_names)
    all_tickers = [a["ticker"] for a in cfg["assets"]]
    tickers = list(assets) if assets else all_tickers
    unknown = [t for t in tickers if t not in all_tickers]
    if unknown:
        raise P.PaperError(f"unknown asset(s) {unknown}; configured: {all_tickers}")
    asof_d, now_eff = resolve_times(asof, now)
    hold_end = P.holdout_end(cfg)
    cutoff = pd.Timestamp(cfg["holdout_cutoff_date"])
    results_dir = cfg["data_dir"] / "results"
    files = P.PaperFiles.for_config(cfg)

    ledger = P.load_ledger(files)
    state = P.load_state(files)
    snap = json.loads(json.dumps(settings.snapshot()))
    if state:
        if state.get("settings") != snap:
            raise P.PaperError(
                "the paper-trading settings (strategy, mode, capital, costs) differ from the ones "
                f"the ledger in {files.root} was started with; changing them would mix two "
                "strategies in one ledger. Restore config.yaml, or move data/paper away to start "
                "a new ledger")
        if asof_d < pd.Timestamp(state["start_date"]):
            raise P.PaperError(f"--asof {asof_d.date()} is before the ledger start "
                               f"{state['start_date']}; no retroactive paper trades")
    new_state = json.loads(json.dumps(state)) if state else {
        "version": P.STATE_VERSION, "start_date": str(asof_d.date()), "settings": snap,
        "holdout_cutoff": str(cutoff.date()), "holdout_end": str(hold_end.date()), "assets": {}}

    # ---- 1. update data (the existing incremental update); never in a dry run
    if prices is None:
        if update == "default":
            update = lambda c, n, w: D.update_all(c, now=None, warnings=w)   # noqa: E731
        if update is not None and not dry_run:
            warns = []
            try:
                update(cfg, now_eff, warns)
            except (D.DataDownloadError, D.DataValidationError) as e:
                say(f"WARNING: data update failed ({e}); using the data already stored.")
            for w in warns:
                say(f"WARNING: {w}")
        elif dry_run:
            say("dry run: not downloading new data and writing nothing under data/paper")
        try:
            prices = F._load_all(cfg, None)
        except (ValueError, FileNotFoundError, OSError) as e:
            for t in tickers:
                res.skipped[t] = f"price data missing ({e})"
                say(f"SKIPPED {t}: price data missing ({e}); no new trade recorded.")
            return res
    prices = _truncate_prices(cfg, prices, now_eff)

    # ---- 2. per asset: dataset, catch-up processing
    ledger_new, models_new, any_new = [], {}, False
    for t in tickers:
        tw = next(a["trades_weekends"] for a in cfg["assets"] if a["ticker"] == t)
        try:
            expected = usable_last_date(bool(tw), now_eff)
            px_t = prices.get(t)
            if px_t is None or len(px_t) == 0:
                raise P.SkipAsset(f"no price data stored for {t}")
            last_px = px_t.index.max()
            if last_px < expected:
                raise P.SkipAsset(f"STALE DATA - latest stored date is {last_px.date()}, but the "
                                  f"most recent completed day is {expected.date()}")
            bad = stale_inputs(cfg, prices, t, usable_last_date(False, now_eff))
            if bad:
                raise P.SkipAsset("STALE INPUT DATA - " + "; ".join(bad))
            try:
                ds = F.build_dataset(t, cfg, prices)
            except (ValueError, KeyError) as e:
                raise P.SkipAsset(f"features could not be built ({e})") from e
            if ds.index.max() < expected:
                raise P.SkipAsset(f"features are only available through {ds.index.max().date()} "
                                  f"(a required input series is stale); most recent completed "
                                  f"day is {expected.date()}")
            if ds[F.core_columns(F.feature_columns(ds))].iloc[-1].isna().any():
                raise P.SkipAsset(f"today's ({ds.index[-1].date()}) features are incomplete; "
                                  "not predicting from imputed values")
            led_t = ledger[ledger["asset"] == t]
            st_t = new_state["assets"].get(t)
            if st_t is None and len(led_t):
                raise P.PaperError(f"{t}: ledger has rows but the state file has no entry")
            if not len(led_t) and settings.strategy_mode != "sized":
                # a brand-new ledger for this asset: the config must be a logged Stage 3 trial
                P.verify_trial_log(settings.config_id(t), results_dir)
            bt = P.load_backtest_history(t, settings, results_dir, cutoff)
            r = P.process_asset(t, ds, settings, led_t, st_t, bt, hold_end, cutoff,
                                model_loader=lambda t=t: P.load_model(files, t))
        except P.SkipAsset as e:
            res.skipped[t] = str(e)
            say(f"SKIPPED {t}: {e}. No new trade recorded for it; will retry on the next run.")
            continue
        if len(r.rows):
            any_new = True
            ledger_new.append(r.rows)
            new_state["assets"][t] = r.state
            if r.model is not None:
                models_new[t] = r.model
            for _, row in r.rows.iterrows():
                if len(r.rows) <= 5 or row["date"] == r.rows["date"].iloc[-1]:
                    say(_fmt_row(row))
            if len(r.rows) > 5:
                say(f"{t}: caught up {len(r.rows)} days "
                    f"({pd.Timestamp(r.rows['date'].iloc[0]).date()} to "
                    f"{pd.Timestamp(r.rows['date'].iloc[-1]).date()}), {r.n_retrains} retrain(s)")
        else:
            have = led_t["date"].max()
            say(f"{t}: ledger already up to date (through {pd.Timestamp(have).date()}); nothing to do.")

    # ---- 3. write (only when something is new); models first, state last
    if any_new and not dry_run:
        for t, m in models_new.items():
            P.save_model(files, t, m)
        P.save_ledger(files, pd.concat([ledger] + ledger_new, ignore_index=True)
                      if len(ledger) else pd.concat(ledger_new, ignore_index=True))
        P.save_state(files, new_state)
        res.wrote = True
    elif any_new and dry_run:
        say("dry run: nothing was written")
    res.new_rows = pd.concat(ledger_new, ignore_index=True) if ledger_new else None
    if res.skipped:
        say(f"{WARN_BAR}\nWARNING: {len(res.skipped)} asset(s) skipped (see SKIPPED lines above)\n{WARN_BAR}")
    return res


def main(argv=None, **run_kw) -> int:
    ap = argparse.ArgumentParser(description="Paper-trading daily run (fake money, no broker).")
    ap.add_argument("--asof", default=None, help="decide as of this date (YYYY-MM-DD); default today")
    ap.add_argument("--assets", nargs="*", default=None, help="subset of configured tickers")
    ap.add_argument("--dry-run", action="store_true", help="compute and print, write nothing")
    ap.add_argument("--config", default=None)
    args = ap.parse_args(argv)

    def err(msg):
        print(f"ERROR: {msg}", file=sys.stderr)
        return 2
    asof = None
    if args.asof:
        try:
            asof = pd.Timestamp(args.asof)
            if len(args.asof) != 10 or pd.isna(asof):
                raise ValueError
        except (ValueError, TypeError):
            return err(f"--asof must be a date like 2026-11-06; got {args.asof!r}")
    try:
        cfg = load_config(args.config)
    except (OSError, ValueError, KeyError) as e:
        return err(f"cannot load config: {e}")
    try:
        run(cfg, asof=asof, assets=args.assets, dry_run=args.dry_run, **run_kw)
    except (P.PaperError, RuntimeError, ValueError, KeyError, OSError) as e:
        return err(e.args[0] if isinstance(e, KeyError) and e.args else e)
    except Exception as e:      # noqa: BLE001 - a clean one-line error, never a traceback
        if os.environ.get("QUANT_DEBUG"):
            raise
        return err(f"unexpected failure: {type(e).__name__}: {e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
