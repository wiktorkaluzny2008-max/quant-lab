# Quant Lab

A personal research and learning project: daily direction models and honest backtests for the S&P 500 (`SPY`), Nasdaq-100 (`QQQ`), Russell 2000 (`IWM`) and Bitcoin (`BTC-USD`), a Streamlit dashboard, and a paper-trading ledger with **fake money**. It never connects to a broker and never places real trades.

## Research stance (honest summary)

The literature says next-day direction of indices and Bitcoin is barely predictable (accuracy about 51-56%, AUC about 0.51-0.55). This project is built to test that claim without fooling itself: no lookahead, small fixed models, walk-forward validation, transaction costs, baselines that include climatology and volatility-targeted buy and hold, a trial log with the Deflated Sharpe Ratio, and a 24-month hold-out evaluated exactly once. A strategy that loses to buy and hold, reported honestly, is a valid result. See `QUANT_SPEC.md` (the build spec), `RESEARCH_REPORT.md` (the evidence), `PROGRESS.md` (decisions, in beginner language) and `GAUNTLET_REPORT.md` (independent critic verdicts per stage).

**Not financial advice.** This is research and education only. Nothing here is a recommendation to buy or sell anything. There is no broker integration.

## Setup

Python **3.12** and [uv](https://docs.astral.sh/uv/):

```
uv sync
```

### Setup on a fresh Mac (LightGBM needs OpenMP)
LightGBM's macOS wheel needs the OpenMP library (`libomp`). On a clean checkout without it, `import lightgbm` fails, some tests fail or error, and the dashboard's Today page says "lightgbm could not load". Do ONE of these, then restart:
1. `brew install libomp` (the normal fix; needs Homebrew), OR
2. `sh scripts/fix_lightgbm_macos.sh` (no Homebrew: points the venv's lightgbm at the libomp that scikit-learn already ships inside `.venv`; re-run after any reinstall; undo with `uv sync --reinstall-package lightgbm`).

Then check with `uv run python -c "import lightgbm"` and `uv run pytest -W error -q`.

## How to run

```
uv run python -m quant.data                          # download / incrementally update prices
uv run python -m quant.features                      # build the feature datasets
uv run python -m quant.backtest --models logreg rf lgbm   # walk-forward backtest, all assets (~20 min), writes data/results/
uv run python -m quant.leverage                      # what-if leverage tables from the saved results
uv run python -m quant.run_daily                     # update data, predict, update the paper-trading ledger
uv run python -m quant.run_daily --dry-run           # show what it would do, write nothing
uv run streamlit run app/dashboard.py                # the dashboard (usually http://localhost:8501)
uv run pytest -W error -q                            # tests
```

Add `--assets SPY` or `--models logreg rf` to the backtest to run a subset. Errors print one `ERROR:` line.

### Daily paper trading update, and scheduling it yourself
`uv run python -m quant.run_daily` is safe to run as often as you like (idempotent) and catches up every missed day. **This project does not schedule anything.** If you want it automatic, either use cron (`crontab -e`), e.g. Tuesday to Saturday at 06:30 local time:
```
30 6 * * 2-6 cd /path/to/quant-lab && $HOME/.local/bin/uv run python -m quant.run_daily >> data/paper/run.log 2>&1
```
or launchd (also runs a missed job when the Mac wakes): save a plist as `~/Library/LaunchAgents/com.quantlab.rundaily.plist` with `ProgramArguments` = `uv run python -m quant.run_daily`, `WorkingDirectory` = the project folder, and a `StartCalendarInterval` of 06:30; load it with `launchctl load ...` and undo with `launchctl unload ...`. The full plist and the caveats (laptop must be awake for cron, Desktop folders may need Full Disk Access, `uv` needs an absolute path, internet is needed to refresh prices, BTC rows lag one UTC day by design) are in `PROGRESS.md`, section "Stage 4 built".

## Changing assets and settings (`config.yaml`)

- `assets`, `costs` (keys must match the tickers), `starting_capital`, `borrow_rate`, `paper_trading` (strategy mode, sized-mode parameters) live in `config.yaml`.
- **`holdout_cutoff_date` is pinned. Never change it.** Everything after it is the final hold-out; re-deriving it from "latest date minus 24 months" would let new data leak into development.
- `leverage_strategy` is the pinned "chosen strategy" (not picked by Sharpe). Paper trading and the hold-out evaluation both use it.
- **Paper-trading settings are frozen once the ledger has started**: changing them is refused (it would mix two strategies). To restart, move `data/paper/` away yourself.
- Adding or removing an asset means re-running data, features and the backtest, and it changes the trial count (the Deflated Sharpe Ratio's N).

## Project structure

```
config.yaml            assets, costs, pinned hold-out cutoff, strategy, paper trading
quant/data.py          download + incremental update + validation
quant/features.py      causal feature engineering (no lookahead)
quant/models.py        baselines, logistic regression / random forest / LightGBM, Platt calibration
quant/backtest.py      walk-forward validation, strategies, costs, significance tests
quant/metrics.py       prediction, trading and significance metrics
quant/trials.py        trial log (every configuration tried)
quant/leverage.py      what-if leverage simulator
quant/paper.py         paper trading ledger logic;  quant/run_daily.py is its one-command runner
quant/explain.py, dash_data.py, theory.yaml   dashboard logic, explanations, theory text
quant/holdout.py       Stage 6 one-shot hold-out harness
app/dashboard.py       Streamlit app (7 pages)
tests/                 pytest suite (synthetic data, no network)
data/                  parquet files (git-ignored): prices, features, results/, paper/
```

## The hold-out rule

The 24 months after the pinned cutoff (2024-11-06) are kept untouched until Stage 6 and are evaluated exactly once, with the strategy and the verdict rule declared beforehand and nothing tuned (the results are filled in below after the run):

```
uv run python -m quant.holdout --dry-run                 # rehearsal on a PSEUDO hold-out inside development data
uv run python -m quant.holdout --verify-preregistration  # config.yaml and quant/*.py (excl. holdout.py) still match the pre-registered SHA-256s
uv run python -m quant.holdout --significance-summary    # print the saved development significance numbers (read-only)
uv run python -m quant.holdout --run --confirm-single-final-evaluation   # the single real evaluation
```

The real run writes `data/results/holdout/HOLDOUT_DONE.json` first and refuses to run again (a failed run also leaves the marker; re-running needs a documented manual step, see `quant/holdout.py`). Results go to `data/results/holdout/` (`holdout_metrics.parquet`, `holdout_predictions.parquet`, `holdout_returns.parquet`, `holdout_summary.json`, `holdout_results.md`), which the dashboard and the other loaders never read. Before and after the run, SHA-256 checksums of the freeze manifest (`quant/*.py`, `config.yaml`, `uv.lock`, `pyproject.toml`, the Stage 3 results, the feature datasets and the paper ledger) are compared to prove nothing changed. The verdict rule (default "no evidence of an edge"; "weak evidence" only from tests against a baseline; headline = the weakest per-asset verdict) is written in `PROGRESS.md` before the run.

The hold-out's job is to catch failure, not to prove success: even 24 months give a Sharpe standard error of about 0.7.

### HOLD-OUT RESULTS (evaluated once, decision days 2024-11-06 to 2026-11-05)

Pinned strategy: ensemble (logistic regression + random forest + LightGBM), long/flat, volatility-targeted, adaptive margin, expanding window, headline costs (0.05% ETFs, 0.15% BTC). Sharpe with a 95% confidence interval (the interval is about plus or minus 1.4, so none of these differences is statistically distinguishable from zero).

| Asset | Days | Strategy Sharpe [95% CI] | Buy and hold | Vol-targeted B&H | Accuracy | AUC | Brier skill (eval base) | Verdict |
|---|---|---|---|---|---|---|---|---|
| SPY | 501 | 0.60 [-0.79, 1.99] | 0.81 | 0.76 | 52.3% | 0.486 | -0.0014 | no evidence of an edge |
| QQQ | 501 | 0.54 [-0.85, 1.93] | 0.72 | 0.68 | 50.3% | 0.437 | -0.0077 (flag: significantly worse) | no evidence of an edge |
| IWM | 501 | 0.68 [-0.71, 2.07] | 0.50 | 0.42 | 49.7% | 0.533 | +0.0015 | no evidence of an edge |
| BTC-USD | 730 | -0.13 [-1.52, 1.26] | 0.22 | 0.22 | 50.5% | 0.486 | -0.0024 | no evidence of an edge |

**Headline: no evidence of an edge.** The strategy did not beat buy-and-hold on SPY, QQQ or BTC-USD. On IWM it was ahead (+0.18 vs buy-and-hold, +0.26 vs vol-targeted, standard error about 0.5) only because it was invested 67% of the time in a weak year; that gap is far inside the noise and it reverses with a one-day execution delay (Sharpe 0.03). Full tables, stress scenarios and the verdict reasons: `data/results/holdout/holdout_results.md`. The development-period numbers (Deflated Sharpe, SPA, t-statistics) are summarised at the top of `GAUNTLET_REPORT.md`. This is research and education, not financial advice.
