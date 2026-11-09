# PROGRESS

## Stage 1 — Data: plan
- T1: Scaffold (pyproject.toml via uv, config.yaml, .gitignore, quant/ package, tests/, README stub).
- T2: quant/data.py — download, incremental update, parquet cache, validation, clear errors. Tests mock yfinance (no network in tests).
- Then: Gauntlet (Code + test critic only for stage 1), fix, re-run, record in GAUNTLET_REPORT.md.

## Decisions (beginner-friendly)
- Installed `uv` with the official installer (approved by Wiktor); it fetches Python 3.11+ itself since system Python is 3.9.
- Tests never hit the network: Yahoo rate-limits (HTTP 429), and tests must be repeatable.
- Gauntlet round 1 FAIL: gap check too loose. Fix: compare against a real NYSE trading calendar (exchange_calendars package, small dependency) so any single missing trading day is caught. Also: drop BTC's still-forming bar for today (UTC), since a partial daily bar would leak a half-finished price into later stages.
- Gauntlet round 2 PASS. Open non-blocking ideas for Wiktor: pin a Python version; add a "data is stale" check before Stage 4 predictions; avoid training on today's partial stock bar.
- Follow-ups done (requested by Wiktor): staleness warning, no incomplete stock/VIX bars, Python 3.12 pin, cost keys = tickers, pandas warning filtered (it comes from the exchange_calendars library, not our code). Round 3 critic PASS.

## Stage 2 — Features: plan
- T3: quant/features.py — per-asset features known at close of day t, target (next-day up/down) + next-day return stored separately from features.
- T4: tests/test_features.py — required truncation test (features up to a random date identical on truncated data), plus fake-future-data test, as-of cross-asset join test, target alignment test.
- Gauntlet: Leakage auditor + Code + test critic (per spec table). Max 3 rounds.
- Beginner note: "lookahead" means accidentally using tomorrow's information to predict tomorrow; it makes backtests look great and real trading fail.
- Stage 2 built (features.py, 88 tests green). Open decision for Stage 3 (not needed now): SPY/QQQ's `xret_BTC-USD` feature is NaN before BTC data starts (2014-09). Default `cross_na="keep"` keeps full history; LightGBM handles NaN, logistic regression/random forest need rows dropped or imputed.
- BTC-to-stock rule: a stock on day t only sees BTC bars dated before t, because BTC's day-t bar closes at 00:00 UTC (after the 16:00 ET stock close).
- Stage 2 round 1: Code+test FAIL. Decisions: momentum windows become 20/50/200 (matches the spec's wording); clear ValueErrors for empty/short/missing-VIX input.
- Stage 2 round 2: both critics PASS. Deferred ideas: add a flat-close zero-return target test and tidy test leftovers; Stage 3 must decide how to handle NaN `xret_BTC-USD` for SPY/QQQ (keep for LightGBM, drop/impute for the others).

## Spec v2 (Wiktor replaced QUANT_SPEC.md and added RESEARCH_REPORT.md)
What changed that affects finished work:
- Stage 1: add IWM, ^VIX3M, ^VXN, ^RVX and data/fomc_dates.csv (the "Stage 1 follow-ups" section). Staleness, incomplete-bar rule, Python 3.12 and ticker cost keys are already done.
- Stage 2: the feature list was rewritten (reversal R5, overnight/intraday returns, Yang-Zhang vol, trend horizons per asset, VIX term structure, variance risk premium, FOMC flag, stale flag for BTC cross-asset values, BTC-SPY correlation, IWM-SPY spread, feature_groups). The earlier Stage 2 PASS was against the OLD feature list, so Stage 2 is reopened as "Stage 2 (spec v2)".
- Later stages (3-6) now include climatology/vol-targeted baselines, trial log, calibration, PT/PSR/DSR/SPA tests, leverage simulator, richer dashboard, 24-month hold-out.
Plan now: (1) Stage 1 follow-ups + Code+test critic; (2) Stage 2 rebuild to spec v2 + Leakage auditor + Code+test critic. Stopping there; Stage 3+ are separate goals.
- Stage 1 follow-ups (spec v2) done, critic PASS. ^RVX unavailable on Yahoo -> IWM uses ^VIX for its variance risk premium (decision per spec). FOMC dates verified against the Fed's site.
- Stage 2 v2 design decisions: SPY/QQQ get no cross-asset features (spec lists cross-asset only for BTC and the IWM-SPY spread), so the old NaN `xret_BTC-USD` problem disappears. One of RSI(14) / z-score: keep RSI(14). Missing-value rule (Stage 3, not Stage 2): LightGBM keeps NaN; logistic regression/random forest impute with the training-window median plus a missing flag.
- Stage 2 (spec v2) done, both critics PASS in round 1. Before Stage 3: config.yaml still says holdout_months: 12 but spec v2 says 24 (Stage 3 must change it). Deferred test gaps G1/G2 listed in GAUNTLET_REPORT.md.
- config.yaml holdout_months set to 24 (spec v2), approved by Wiktor.

## Stage 3 — Models, validation and metrics: plan
- T5 quant/metrics.py: prediction metrics (accuracy vs base rate, AUC, log loss, Brier skill vs climatology, reliability bins, hit rate by confidence decile, Pesaran-Timmermann), trading metrics (CAGR, vol with sqrt(365)/sqrt(252), Sharpe, Sortino, max DD, % invested, trades, turnover, break-even cost), significance (random strategies percentile, PSR, DSR, Diebold-Mariano, Hansen SPA via arch, t-stat vs 3).
- T6 quant/models.py: baselines (climatology, buy&hold, vol-targeted B&H, always up, 200-day trend), logistic/RF/LightGBM with small fixed settings, equal-weight ensemble, training-median imputation + missing flags, Platt calibration on the last ~20% of each training window (FrozenEstimator), group permutation importance.
- T7 quant/trials.py + quant/backtest.py: walk-forward expanding window, monthly retrain, 1-day gap, BTC rolling-4y variant, strategies (long/flat with base-rate-relative threshold, default-long-step-aside, vol-targeted versions), costs (headline + sensitivities + one-day delay), trial log, 24-month hold-out EXCLUDED from all Stage 3 data.
- T8 quant/leverage.py + results runner: leverage simulator (1x/1.5x/2x/3x, borrowing cost, wiped-out flag, Kelly curve); `python -m quant.backtest` writes data/results/ for all 4 assets.
- Gauntlet: Leakage auditor + Statistical honesty critic + Code+test critic. Max 3 rounds.
- Beginner note: the hold-out is the last 24 months of data, locked away until Stage 6. Its job is to catch failure, not to prove success: even 24 months gives a Sharpe standard error of about 0.7, so a good hold-out result alone proves little, but a bad one is informative.
- Stage 3 environment: LightGBM could not load on this Mac (no Homebrew libomp). With Wiktor's approval I pointed the venv's lightgbm library at the libomp bundled with scikit-learn (install_name_tool + ad-hoc codesign, .venv only). Reproduce with `scripts/fix_lightgbm_macos.sh`; undo with `uv sync --reinstall-package lightgbm`.
- Stage 3 decisions (Wiktor): (1) keep sigmoid calibration exactly as in the spec, and also report RAW (uncalibrated) prediction metrics, because the Platt slope can go negative and flip a signal (found in the SPY smoke run: all OOS AUCs ~0.485); (2) dedupe trial configs that produce identical return streams (long_flat and default_long are the same rule at equal m).
- Beginner note: AUC below 0.5 means the model ranks days slightly WORSE than a coin flip. With ~3,700 test days that is within noise (standard error about 0.01), but the negative calibration slope shows how a calibration step fitted on a small, noisy slice can flip a weak signal.
- Stage 3 round 1: all three critics FAIL (details in GAUNTLET_REPORT.md). Fix round 2 scope: calibration-data test; per-asset DSR + excess-over-vol-targeted-B&H significance, SPA vs vol-targeted B&H, baseline t-stats; pinned hold-out cutoff date 2024-11-06; call-rate/base-rate disclosure; clean CLI errors; Stage 3 documentation (run commands, file schema, decisions).

## Stage 3 built (quant/trials.py, quant/backtest.py; round-2 fixes included)

### How to run it (in this order)
```
uv sync
sh scripts/fix_lightgbm_macos.sh      # only on this Mac: lets LightGBM find its OpenMP library
python -m quant.data                  # download/update prices
python -m quant.features              # build the feature datasets
python -m quant.backtest --models logreg rf lgbm   # all 4 assets, ~20 minutes
python -m quant.leverage              # what-if leverage tables from the saved results
```
`--assets SPY` / `--models logreg rf` run a subset. Errors (missing data, bad config) print one `ERROR:` line and exit code 2.

### Settings and decisions (plain words)
- **Hold-out is pinned by date.** `holdout_cutoff_date: "2024-11-06"` in `config.yaml` (= last data date minus 24 months, on the day it was pinned). Data after it is never loaded into any result. Never change it after Stage 3, otherwise newly downloaded days would leak into "development" data. New rows added to the dataset change nothing (tested). The last dev day has its label blanked (its next-day return ends inside the hold-out), so one day is lost.
- **What the hold-out is for:** to catch failure, not to prove success. Even 24 months gives a Sharpe standard error of about 0.7, so a good number proves little while a bad one is informative. It is evaluated once, in Stage 6.
- **Walk-forward:** expanding window, retrain on the first row of every month, training rows end 2 rows before the first predicted day (1-day gap). Minimum 2500 training rows (so the 20% calibration slice has 500 days). 21-day window for realised volatility in vol-targeting.
- **Calibration** is Platt (sigmoid) exactly as in the spec. Raw (uncalibrated) predictions are saved too, as diagnostics only. The fitted Platt slope can be negative (it flips the model's ranking); counted per model in `metadata.json`.
- **Strategies:** long/flat and default-long, each raw and vol-targeted, margins m on fixed grids (long_flat 0/0.005/0.01/0.02, default_long 0.01/0.02/0.03) plus an adaptive m. **Adaptive m uses PRIOR OUT-OF-SAMPLE days** (strictly before the block, causal), a deliberate deviation from the spec wording "chosen only on training data": the training window has no out-of-sample strategy returns to choose from. It falls back to m=0 (long_flat) / 0.02 (default_long) until 500 past OOS days exist.
- **Dedupe:** long_flat and default_long are the same rule at equal m, so identical return streams are logged once (aliases listed). 360 configurations collapse to **270 distinct trials**. That 270 is the N in the Deflated Sharpe Ratio. The effective number of independent trials is far smaller (many are near-copies), so the DSR is conservative in N.
- **BTC rolling 4-year window** uses `min_cal=250` instead of the spec's >=500 calibration days (292 calibration rows at 20% of 1461; 500 is arithmetically impossible). **BTC expanding-window OOS is only 981 days (2022-03 to 2024-11)**, thin and a different period from the rolling variant (2019-05 on), so the two BTC windows are NOT comparable (buy&hold Sharpe 0.64 vs 1.06 on their own dates). The SPA test for BTC uses the expanding dates.
- **Probability skill caveat:** Brier/log-loss skill are measured against the TRAINING-window base rate. In a drifting market (BTC especially) the calibrated probability and the whole-window base rate can sit at different levels, which can make skill positive (BTC expanding: +0.005) with AUC below 0.5. Read skill together with AUC and with the extra column `brier_skill_vs_eval_base` (skill versus the evaluation window's own up frequency; a diagnostic that uses the test labels). `call_rate` (share of days with p >= base rate) is saved; rates below 10% or above 90% are flagged (`call_rate_flag`, and listed in `metadata.json`).
- **Significance:** Sharpe standard errors and the SPA test are large with a few thousand daily returns, so "close to buy&hold" cannot be told from "equal". `t > 3` and the DSR test against ZERO return, not against buy&hold, so a long-biased strategy on a rising asset scores high; that is why excess-over-vol-targeted-B&H and excess-over-B&H PSR/DSR rows and a second SPA (vs vol-targeted B&H) exist. For vol-targeted trials the random-strategy percentile mostly reflects vol targeting, because shuffling also destroys the vol alignment.
- **Stage 6 rule:** the strategy to evaluate on the hold-out must be declared BEFORE looking, not picked as the best-Sharpe one of the 270 trials. The leverage simulator uses the fixed `leverage_strategy` in `config.yaml` (expanding / ensemble / long_flat / vol-targeted / adaptive m) for that reason.
- `threshold: 0.55` in `config.yaml` is unused (strategies are relative to the base rate).

### Per-asset result files (`data/results/<ticker>/`)
| File | What is in it |
|---|---|
| `predictions.parquet` | one row per window, model, day: calibrated `p_up`, raw `p_raw`, training `base_rate`, outcome `y`, `next_return` |
| `prediction_metrics.parquet` | per window and model: accuracy, AUC, log loss, Brier and skills, Pesaran-Timmermann, call rate, mean p, skill vs evaluation base rate, note |
| `raw_prediction_metrics.parquet` | AUC, Brier skill, log-loss skill, accuracy on RAW probabilities (diagnostic) |
| `prediction_reliability.parquet` | reliability curve bins with counts |
| `prediction_deciles.parquet` | hit rate by confidence decile |
| `trading_metrics.parquet` | one row per baseline/trial config and cost scenario (headline, low/stress, 1-day delay): return, Sharpe, drawdown, turnover, break-even cost, call rate, `duplicate_of` |
| `baselines.parquet` | daily position and returns of buy&hold, always-up, vol-targeted B&H, trend_200 |
| `significance.parquet` | long table of tests (`test`, `window`, `subject`, `reference`, `value`, `statistic`, `pvalue`, `n`, `details`); **`details` is a JSON string** (e.g. DSR scope, SR0, N, variance) |
| `random_strategies.parquet` | the 1000 random-strategy daily Sharpes per trial (`config_id`, `sim`, `sharpe_daily`) |
| `importance.parquet` | out-of-sample group permutation importance, aggregated over blocks |
| `adaptive_m.parquet` | the m chosen in every block by the adaptive rule |
| `retrains.parquet` | one row per retrain: dates, training size, base rate, sigma_target, Platt slopes |
| `metadata.json` | dates, pinned cutoff, settings, model settings, config hash, trial counts, negative-slope counts, call-rate flags, notes |
| `leverage.parquet`, `leverage_equity.parquet`, `kelly.parquet`, `crashes.json` | leverage what-if outputs (from `quant.leverage`) |

Trial files (`data/results/`): `trials.parquet` (config_id, date, net_return, gross_return, position; headline cost, canonical configs only) and `trials_index.parquet` (one row per distinct stream: all settings incl. model settings, n_oos_days, Sharpe, `aliases`, `duplicate_of`).
- Stage 3 round 2: Leakage PASS, Stat honesty PASS, Code+test FAIL (malformed YAML traceback). Round 3 (last allowed): fix YAML error, catch any lightgbm import failure cleanly, close the imputer test gap with NaN-carrying test frames, add the five disclosure nits to the docs, then re-run the full gauntlet.

### Stage 3 built: extra reading notes and column lists (round 3)
- **Platt slope flips.** The Platt calibrator often learns a NEGATIVE slope (it turns the model's ranking upside down because the recent calibration slice disagreed with the fit slice). Negative-slope blocks: SPY logreg 116/179, SPY rf 90/179, QQQ logreg 115/177, QQQ rf 107/177, IWM logreg 91/162, BTC expanding logreg 12/33 (rf and lgbm 0/33). As a result the calibrated AUC can be LOWER than the raw AUC (for example SPY and IWM). The raw numbers are in `raw_prediction_metrics.parquet`; the calibration stays as the spec says.
- **BTC expanding is rarely invested in default-long mode.** Calibrated p averages about 0.49 while the whole-window training base rate is about 0.536 (4 to 5 points higher), so "p >= base_rate - m" is false most days: the call rate is only 1% to 7%, and the non-adaptive default_long strategies are invested about 2% to 32% of days (ensemble about 6% to 23%). That is an artefact of the offset between the calibrated probability and the base rate, not a market view. The rolling 4-year window is invested 33% to 59%.
- **Ten adaptive-m trials are coincidental aliases.** In 10 cases (BTC 6, QQQ 2, IWM 2) the adaptive default_long rule chose m=0.02 in every block, partly because the m=0.02 fallback applies until 500 prior out-of-sample days exist, so its stream equals the fixed long_flat m=0.02 trial; they are deduped like the long_flat/default_long copies (90 aliases in total, 270 distinct trials).
- **BTC per-asset DSR mixes two windows.** The BTC "asset" scope DSR pools the expanding window (981 days) and the rolling 4-year window (2016 days), which are different periods and not comparable (see above).
- **Diebold-Mariano sign:** a POSITIVE statistic means the subject model has the HIGHER log loss (is WORSE) than the reference. Example: IWM rf vs climatology statistic 2.46, p = 0.014 means rf is worse than just predicting the base rate (uncorrected for the 35 Diebold-Mariano tests run across assets and models, so treat p = 0.014 as suggestive, not conclusive).

Column lists the table above only summarised:
- `predictions.parquet`: `window, model, date, block, p_up, p_raw, base_rate, y, next_return` (`block` = first date of the month whose retrain produced the prediction).
- `kelly.parquet`: `ticker, L, g, g_lo, g_hi, mu, sigma, sigma2, se_mu, n_days, years, L_star, L_star_lo, L_star_hi, L_star_net_borrow, borrow_rate, train_start, train_end`.
- `leverage.parquet`: `ticker, series, config_id, cost, borrow_rate, days_per_year, L, cagr, max_drawdown, worst_day, worst_day_date, worst_month, worst_month_label, longest_recovery_days, unrecovered, sharpe, final_equity, n_days, wiped_out, wipe_date`.
- `leverage_equity.parquet`: `ticker, series, L, date, equity`.
- `metadata.json` keys: `ticker, holdout_cutoff_date, holdout_cutoff_pinned, holdout_months, call_rate_flags, skill_note, dsr_note, dev_first_date, dev_last_date, n_dev_rows, windows, models, ensemble_members, headline_cost, cost_scenarios, periods_per_year, settings, model_settings, m_grid, m_fallback, n_trials_this_asset, n_configs_before_dedupe, negative_platt_slope_blocks, n_blocks, n_trials_global, config_hash, notes`.
- Stage 3 round 3: Leakage PASS, Code+test PASS, Stat honesty FAIL on a wording error (aliases "all BTC"; actually BTC 6, QQQ 2, IWM 2). Wording fixed; no 4th round run because the spec caps rounds at 3. Needs Wiktor's call.

## Stage 3 closed (2026-11-08)
In plain words: Stage 3 is finished. One reviewer (Statistical honesty) needed extra checks because of wording mistakes in my notes, never because of wrong numbers. You chose to accept it as it is, and the report says so openly. Stage 4 (paper trading) has not started.


## Stage 4 built (quant/paper.py, quant/run_daily.py; paper trading with FAKE money, no broker)

### Decisions (with reasons)
- **Nothing new is selected.** Paper trading reuses the already-pinned Stage 3 configuration `leverage_strategy` in `config.yaml` (expanding window, ensemble, long_flat, vol-targeted, adaptive m). Reason: picking a "best" strategy now would be a new unlogged search. Its config ID is looked up in the Stage 3 trial log (`data/results/trials_index.parquet`) at the start of each asset's ledger and stored in every row, e.g. `SPY|expanding|ensemble|long_flat|vt|m=adaptive`, so live results can be compared with the backtest of exactly the same configuration. If the ID is missing from the log, the run stops with an `ERROR:` (it never trades an unlogged config).
- **Unleveraged**: position is always between 0 and 1.
- **`paper_trading` section in `config.yaml`**: `starting_capital` (10000 each; a number or a ticker -> amount map), `strategy_mode` = `long_flat` (default) | `default_long` | `sized`, `sized_lambda` (0.25 to 0.5, default 0.25 = the most cautious fractional Kelly) and `sized_band` (0.1). The strategy itself is derived from `leverage_strategy` with `mode = strategy_mode`; setting it separately is refused. Changing the settings after the ledger has started is refused (it would mix two strategies in one ledger); to restart, move `data/paper/` away yourself.
- **Sized mode**: `position = clip(lambda * (p - base_rate) / scale * sigma_target / sigma_hat, 0, 1)` and the position only changes if the new target differs from the current one by at least the band (0.1). The spec does not define `scale`; I use the standard deviation of `p - base_rate` over past out-of-sample days (Stage 3 days plus live days), fixed at each monthly retrain (0.02 if fewer than 50 days). Sized mode is not a Stage 3 trial, so its `config_id` is `<ticker>|expanding|ensemble|sized|vt|lambda=..,band=..` and is not in the trial log (the probability metrics shown next to it still come from the same model's backtest).
- **Retraining rule**: the live model is retrained on the first processed day of each new calendar month (exactly the walk-forward's monthly blocks) instead of "30 days since last time", so the model used on a given day does not depend on when you happened to run the program. That is what makes catch-up equal day-by-day. Training uses rows up to two rows before the decision day (1-day gap), labels known at that time, Platt calibration on the last 20% (`models.fit_calibrated`, unchanged).
- **Adaptive m** uses `backtest.adaptive_positions` itself on past out-of-sample days strictly before the month: the Stage 3 out-of-sample days (read from `data/results`, all on or before the cutoff) followed by the live days. Junction caveat: the hold-out calendar period lies between those two segments and is skipped, but one cost term at the junction (live day 1 versus the last backtest day) is treated as a normal position change; this is negligible and noted for honesty. Replaying this code on the Stage 3 predictions reproduces `trials.parquet` positions of the chosen config exactly (checked for all 4 assets).
- **Costs**: 0.05% ETF, 0.15% BTC of the traded fraction, charged on every position change. The ledger starts in cash, so entering the first position costs the same as any trade. Buy-and-hold and vol-targeted buy-and-hold start at the same time and pay costs too.
- **Stale data**: if an asset's data (or a required input such as `^VIX`, or SPY for IWM, SPY/QQQ/IWM for BTC) is older than the most recent completed day, the asset is SKIPPED with a clear message and no row is written; the next run catches up. A failed download is a warning (stored data is used, then judged stale or not). Both end in exit code 0. Exit code 2 is for real errors (bad config, corrupt ledger or state, config id not in the trial log, hold-out violation, changed settings); they print one `ERROR:` line, never a traceback.

### How to run
```
export PATH=$HOME/.local/bin:$PATH
uv run python -m quant.run_daily                       # update data, catch up, write ledger
uv run python -m quant.run_daily --dry-run             # show what it would do, write nothing (no download either)
uv run python -m quant.run_daily --asof 2026-11-13     # decide as of a date (never earlier than the ledger start)
uv run python -m quant.run_daily --assets SPY BTC-USD  # subset
```
Running it twice on the same day changes no byte under `data/paper/` (it only writes when there is a new day, and stores no wall-clock time). After missed days it processes every missed trading day in order (BTC: every calendar day), each with only the information available on that day.
The first run starts the ledger at that day (SPY/QQQ/IWM: last completed session; BTC: yesterday UTC). There are no retroactive paper trades or metrics for earlier dates, and starting inside the 24-month hold-out period (before 2026-11-06 here) is refused.

### Files in `data/paper/` (git-ignored like the rest of `data/`)
- `ledger.parquet`: one row per asset per decision day, sorted by (asset, date).
- `state.json`: ledger start date, the settings snapshot, and per asset: config ID, first date, last retrain (train date, last training row, rows, base rate, sigma_target, margin m, sized scale) and the backtest reference numbers. Deterministic JSON, no timestamps.
- `models/<ticker>.joblib`: the live model, written only at retrains. If one is missing or unreadable the model is refit deterministically from the stored train date.

### Ledger schema (row = the close of `date`)
| Column | Meaning |
|---|---|
| `asset`, `date`, `config_id`, `strategy_mode` | which asset/day and the Stage 3 trial-log ID |
| `p_up`, `base_rate`, `margin_m`, `vol_ratio` | calibrated P(up) for the next day; training base rate used; margin m (blank in sized mode); sigma_target / sigma_hat (uncapped) |
| `position`, `prev_position`, `trade` | INTENDED position for the next day (0 to 1); position decided at the previous close; absolute change |
| `cost_rate`, `cost`, `cost_usd` | 0.0005 or 0.0015; cost as a return fraction; in dollars |
| `asset_return`, `gross_return`, `net_return` | realised return (previous close to this close); `prev_position * asset_return`; minus cost |
| `equity`, `pnl`, `cum_pnl`, `n_trades`, `cum_cost_usd` | account value after the day; its change; since start; trades so far; costs so far |
| `bh_position`, `bh_equity`, `vtbh_position`, `vtbh_equity` | parallel buy-and-hold and vol-targeted buy-and-hold over the same period |
| `outcome_up` | did the asset rise today (the outcome of the previous row's prediction) |
| `live_n`, `live_brier`, `live_brier_skill`, `live_hit_rate` | running LIVE numbers: Brier of yesterday's p_up against today's outcome; skill vs climatology (the base rate used); share of days where `p_up >= base_rate` matched the direction |
| `bt_brier`, `bt_brier_skill`, `bt_hit_rate`, `bt_n` | the backtest's values for the same model and window (from `data/results/<ticker>/predictions.parquet`, dev period only), fixed at ledger start so decay is visible next to them |

### Hold-out and the live model (important)
- No metric is ever computed over the 24-month hold-out. Live metrics use only realised dates after the hold-out end (cutoff `2024-11-06` + 24 months = `2026-11-06`), enforced by `assert_live_dates`; backtest reference numbers and the adaptive-m history use only Stage 3 dates on or before the cutoff, enforced by `assert_dev_dates`. Nothing is ever written to `data/results/`.
- The live MODEL is retrained on all data available at the time, which includes the hold-out calendar period as training data, for FUTURE predictions only. This does not evaluate the hold-out. Stage 6 still evaluates the hold-out through the walk-forward as the spec says, with the strategy declared beforehand.
- Beginner note: a first row has no result yet (it only decides tomorrow's position). Live Brier/hit rate need several weeks before they mean anything: with 20 days the hit rate has a standard error of about 11 percentage points, so early differences from the backtest are noise.

### Scheduling it on a Mac laptop later (NOT installed; your decision)
The program is safe to run as often as you like. Easiest, cron (`crontab -e`), e.g. Tuesday to Saturday at 06:30 local time (after the US close and after the UTC day for BTC has ended; for Europe/Warsaw that is well after the US close):
```
30 6 * * 2-6 cd /Users/wiktorkaluzny/Desktop/quant-lab && $HOME/.local/bin/uv run python -m quant.run_daily >> data/paper/run.log 2>&1
```
(`data/paper/run.log` is a plain log of the program's text output, not part of the ledger; do not include it when comparing files.) Or launchd, which also runs a missed job when the Mac wakes: save as `~/Library/LaunchAgents/com.quantlab.rundaily.plist` and load it with `launchctl load ~/Library/LaunchAgents/com.quantlab.rundaily.plist`:
```
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>com.quantlab.rundaily</string>
  <key>ProgramArguments</key><array>
    <string>/Users/wiktorkaluzny/.local/bin/uv</string><string>run</string>
    <string>python</string><string>-m</string><string>quant.run_daily</string></array>
  <key>WorkingDirectory</key><string>/Users/wiktorkaluzny/Desktop/quant-lab</string>
  <key>StartCalendarInterval</key><dict><key>Hour</key><integer>6</integer><key>Minute</key><integer>30</integer></dict>
  <key>StandardOutPath</key><string>/Users/wiktorkaluzny/Desktop/quant-lab/data/paper/run.log</string>
  <key>StandardErrorPath</key><string>/Users/wiktorkaluzny/Desktop/quant-lab/data/paper/run.log</string>
</dict></plist>
```
Undo with `launchctl unload ~/Library/LaunchAgents/com.quantlab.rundaily.plist` (or `crontab -e` and delete the line). Caveats: the laptop must be awake (cron does NOT run a missed job after sleep; launchd runs it on wake); if the Mac is off for days nothing is lost, because the next run catches up every missed day; the project is on the Desktop, which macOS may protect (cron/launchd may need Full Disk Access, or move the folder out of Desktop/Documents); `uv`'s path must be absolute in both; the run needs internet to refresh prices, otherwise it prints a warning and skips assets whose stored data is stale; a laptop running at 06:30 sees BTC's UTC-yesterday bar as the latest complete one, so BTC rows are one day behind the calendar date by design.

### Tests
`tests/test_paper.py` and `tests/test_run_daily.py` (synthetic prices and fake Stage 3 result files in tmp dirs, never the network) plus `paper_trading` cases in `tests/test_config.py`: schema, mark-to-market and cost arithmetic, equity chain, baselines, live vs `metrics` Brier/skill/hit rate, one-day training gap, monthly retrain trigger, refit when the model file is missing, adaptive m equals `backtest.adaptive_positions`, sized formula and band, default_long, no lookahead (garbage after the as-of date changes nothing, also mid catch-up), byte-identical re-run, catch-up equals day-by-day, stale/missing data skip, dry run, hold-out guards (recorded dates), clean `ERROR:` exits, `data/results/` untouched.

## Stage 5 logic layer built (2026-11-08) - no UI yet

What this is: all the non-screen code the dashboard needs, fully tested, so the page builder only has to draw things. Three new files: `quant/theory.yaml` (the explanations), `quant/explain.py` (meter, badge, "models agree", "Why?" text), `quant/dash_data.py` (loading saved results + chart helpers + the Refresh button). Tests: `tests/test_explain.py`, `tests/test_dash_data.py` (+ helper `tests/dash_world.py`). Nothing in `data/` is written by any of it except the Refresh button, which just calls the Stage 4 daily run.

### Decisions and reasons
- **Theory text comes only from `RESEARCH_REPORT.md`.** Every source link in `theory.yaml` is checked by a test to appear in the report; where the report gives no link (e.g. the IWM-SPY spread feature, the climatology baseline) the entry says so instead of citing something. Paper "titles" are descriptive labels written from the report's own sentences (the report names authors and years, not full titles).
- **One theory entry per real feature group** (`returns, reversal, volatility, vix_stress, trend, calendar, cross_asset`; the spec's "VIX/stress" group is called `vix_stress` in `features.py`), plus entries for the models, walk-forward, calibration, climatology, Brier skill, vol targeting, the meter and the badge, and the Reality-check facts (AUC 0.51-0.55, stock-ranking vs index timing, post-2010 negative edges, 58% post-publication decay, why a 1-2 year hold-out can't prove an edge, overfitting, how to read the significance tests, Kelly/leverage).
- **Everything is read-only and deterministic.** Explanations are templates; there is no AI service, no random numbers, and a test checksums `data/paper` before and after to prove nothing is written.
- **Hold-out is never touched.** Every saved development series is checked on load (a date after 2024-11-06 raises an error). "What is normal" in the Why? sentences uses only history up to the cutoff and strictly before the decision day. A test appends garbage rows after the decision date and shows the output does not change.

### The signal meter - how a zone is decided
- gap = calibrated probability minus the base rate (positive = more bullish than a typical day). The needle is the percentile of today's gap among the model's past out-of-sample gaps (Stage 3 walk-forward predictions, dev period only, then the live ledger rows before today).
- The zone comes from the **position the paper strategy actually takes** (`paper.decide_position`), so it cannot disagree with the ledger:
  - long_flat / default_long (these are the same rule, "hold if p >= base rate - m"; they differ only in which margins m were searched): position 0 -> **Sell / step aside** (p is more than m below the base rate). Position above 0 -> **Hold / buy**, or **Strong buy** when the reading is in the top 10% of the model's own history AND above the base rate. Volatility targeting changes the size (e.g. 84% invested) but never the zone.
  - **Reduce** and **Neutral** exist on the dial but have zero width for those two modes, because a hold/step-aside rule has no half-way state. They are used by the optional `sized` mode (position 0 / under 25% / 25-50% / 50-99% / fully invested -> Sell / Reduce / Neutral / Hold / Strong buy).
  - "Sell" always means cash, never a short.
- The dial arcs are drawn at the percentile of the strategy's own threshold (gap = -m), so they move when the strategy's margin changes. The needle drawn is clamped into the arc of its zone so the picture cannot contradict the position.
- Beginner warning shown in the caption: with margin m, "Hold / buy" can show while the probability is slightly BELOW the base rate (the strategy tolerates up to m points below). Hold does not mean bullish.
- A test sweeps probability, margin, volatility ratio and previous position for every mode (7,200 points each) and asserts: position 0 <=> Sell zone; fully invested -> Hold/Strong buy; zone equals `decide_position`.

### The evidence badge - rule written before looking at per-asset results
Inputs: saved dev-period backtest, expanding window, headline costs, for the active model/config (`explain.BADGE_RULE` holds the same text).
- PT: Pesaran-Timmermann p < 0.05 and statistic > 0.  SKILL: Brier skill vs climatology > 0.  BEATS: net Sharpe > vol-targeted buy-and-hold net Sharpe.  PSR_X: probabilistic Sharpe of the excess return over vol-targeted buy-and-hold >= 0.95.  DSR_X: deflated Sharpe of that excess (this asset's trial count) >= 0.95.  DSR_G: deflated Sharpe of the strategy using all 270 trials >= 0.95.
- **Green "Evidence"** needs PT, SKILL, BEATS, DSR_X and DSR_G all true. **Amber "Weak evidence"**: BEATS and (PT or PSR_X). **Grey "No proven edge"**: everything else, and any missing/NaN number counts as "not shown". Grey -> meter drawn faded with the note "This model hasn't shown a reliable edge — treat this as a learning tool."
- Why these conditions: direction skill (PT, SKILL), beating the cheap sizing-only benchmark after costs (BEATS - the report says vol targeting may capture most achievable gains), and surviving the search (DSR) are the three separate questions in the report. The DSR of the strategy alone can be high just because the market rises (SPY's is 0.99 against zero), which is why the excess-over-vol-targeted-buy-and-hold versions are required for green.
- Result today: all four assets get **grey "No proven edge"** (for the exact reasons, see the final report; none beats vol-targeted buy-and-hold, and no Pesaran-Timmermann test is significant). The rule was not adjusted after seeing this.

### "Models agree"
The three members of the saved live ensemble (`data/paper/models/<ASSET>.joblib`: logistic regression, random forest, LightGBM, each already calibrated) are asked for today's probability on today's feature row. A member "leans above" if its probability >= the base rate, otherwise "below"; the count is the size of the bigger camp (2 or 3 of 3). The average of the three equals the ledger probability (checked, `matches_ledger`).

### The "Why?" text
- Part 1: per feature the effect is (logistic coefficient x standardised value) and LightGBM `pred_contrib` (tree-SHAP), each multiplied by that member's Platt-calibration slope (the slope can be negative, which flips the sign - ignoring it would mislabel up and down), averaged and summed by feature group. The top 3-4 groups each get one sentence about their most influential feature: today's number, the share of days since the start of history that were higher or lower, and the middle-half range. The random forest has no cheap exact breakdown and is not decomposed (stated in the text). Sizes are in approximate percentage points of probability.
- Part 2: the group's short theory sentence and a source link from `theory.yaml`; "Learn more" data (plain English, formula, evidence, caveats, all sources) is returned too.
- Part 3: hit rate of the up/down call in the same confidence decile (distance of p from the base rate) out of sample for this asset, next to the overall hit rate and the always-up rate, plus the research caveat and the badge note.
- Part 4: one sentence tying the groups to the meter zone and the paper position. It only describes what moved the model.

### Tests (456 in total: 346 existing + 110 new, all pass with `-W error`)
Meter/position sweeps for every mode; percentile maths; badge rule table (grey fallback, green needs every condition, boundaries, NaN); badge lookup from saved files and an independent re-derivation of every real asset's badge; models-agree counts; contribution sums reproduce the member scores; every number in each Why? sentence recomputed independently from the features; no-lookahead (garbage after the decision date changes nothing); theory.yaml schema, https links, every URL and key number present in `RESEARCH_REPORT.md`; hold-out guard; read-only checksums; rolling AUC / Brier skill / deciles / confusion matrix / drawdown / equity curves checked against `metrics.py` and the saved numbers; Refresh button (success, idempotent, no tracebacks).

## Stage 5 dashboard built (2026-11-08)

What this is: the screen on top of the logic layer. One Streamlit app, seven pages, every chart drawn from results that earlier stages already saved. Nothing is computed from the 24-month hold-out.

**How to open it:** `uv run streamlit run app/dashboard.py` (then open the address it prints, normally http://localhost:8501). Pick a page in the left sidebar. "Refresh data" in the sidebar downloads new prices and catches up the paper-trading ledger (the same code as `python -m quant.run_daily`); it shows a one-line message and never a traceback.

### The seven pages and the real data each one shows
1. **Today** - for SPY, QQQ, IWM and BTC: the semicircle signal meter (five zones), "52.7% vs 54.4% typical" style probability, models-agree count, evidence badge, the faded meter plus the exact "This model hasn't shown a reliable edge" note when the badge is grey, the disclaimer under every meter, the four-part "Why?" with an expandable "Learn more" and source link per signal, position size, last update, stale-data warnings. Data: `data/paper/ledger.parquet`, `state.json`, the saved live models, `data/results/<ASSET>/` (for the badge and the percentile history), `quant/theory.yaml`.
2. **Backtest** - pick asset, model, strategy mode (plus vol-sizing, margin and window). Equity curve against buy-and-hold and vol-targeted buy-and-hold, drawdown chart, table of all models and baselines, Sharpe with a plus/minus 2 standard-error band, break-even cost, random-strategy percentile. A **Leverage** tab with a 1x/1.5x/2x/3x selector: log-scale leveraged equity with crashes shaded, drawdown and "wiped out" table, growth-vs-leverage curve with the Kelly point, and a plain note on why a small edge plus leverage usually ends badly. Data: `trading_metrics`, `baselines`, `significance`, `leverage`, `leverage_equity`, `kelly`, `crashes.json`, `trials.parquet`.
3. **Model diagnostics** - reliability curve with bin counts, hit rate by confidence decile, rolling 250-day AUC and Brier skill (with the note that 60-day accuracy has a standard error of about 6.5 points), group permutation importance, Pesaran-Timmermann p-value, confusion matrix. Data: `prediction_reliability`, `prediction_deciles`, `predictions`, `importance`, `prediction_metrics`.
4. **Paper trading** - equity curves, open positions, trade log, profit/loss against both baselines, live vs backtest Brier score and hit rate, and the honest note that a few live days prove nothing. Data: the ledger only.
5. **Trials** - 270 distinct configurations (360 counting identical duplicates), the luck bar (the best Sharpe expected from pure noise after that many tries), and each asset's PSR versus Deflated Sharpe Ratio. Data: `trials_index.parquet`, `significance`.
6. **How it works** - every signal, model, the walk-forward / calibration / climatology / Brier-skill / vol-targeting ideas, and how to read the meter and the badge, from `theory.yaml` (which comes from `RESEARCH_REPORT.md`).
7. **Reality check** - the research facts in plain language (AUC 0.51-0.55, stock-ranking vs index timing, edges negative after costs, about 58% post-publication decay, why a 1-2 year hold-out cannot prove an edge, overfitting, how to read the significance tests, Kelly), each with sources, next to this project's own saved AUCs and trial counts.

### Decisions and reasons
- **Reuse, not re-implement.** All formulas and wording come from `quant/dash_data.py`, `quant/explain.py` and `quant/theory.yaml`. `app/dashboard.py` only lays them out; `app/ui.py` only draws Plotly figures. One cleanup in `explain.py`: a dead, truncated duplicate of `feature_facts`/`feature_sentence` (Python used the later, correct copy) was deleted; behaviour is unchanged and its 84 tests still pass.
- **Sidebar radio instead of Streamlit's multipage folder.** One file, one shared header/footer, and the disclaimer footer cannot be forgotten on a page. Simple to test too.
- **Caching with automatic invalidation.** Loaders use `st.cache_data` (results use `st.cache_resource`), keyed on the modification times of the ledger and result files, so a daily run from the command line is picked up on the next click; the Refresh button also clears the caches. A 15-minute expiry is the backstop.
- **Hold-out.** Only saved development series and the live ledger are loaded. A saved series that contains a date after 2024-11-06 shows a red "blocked" message instead of a chart. Tests scan every date in every plotted series (all at or before 2024-11-05) and checksum `data/` before and after running the app (read-only).
- **Meter and ledger agree.** The zone text on Today comes from the position `paper.decide_position` gives, and a test compares it with the ledger for every asset.
- **Wide layout, dark theme** (`.streamlit/config.toml`), checked at 1366x768.
- **No scheduled job** was created.

### Known limits
- The ledger has only one day per asset so far, so the Paper trading page shows an "only 1 day(s) of data so far" note instead of charts (round 1), and the live Brier score reads "n/a" until predicted days have finished.
- Today's meter needs the saved live models in `data/paper/models/`; if they are missing the card says so and still shows the meter.
- Two source links in `RESEARCH_REPORT.md` return 404 (IMF govdelivery bulletin 30520df, Wikipedia Purged_cross-validation) and one (quantpedia) could not be verified; they are shown with "(link no longer works)" / "(could not be verified)" next to them (round 1).
- Hold-out results are deliberately absent until Stage 6.

### Tests
`tests/test_dashboard.py`: 49 tests running the real app with `streamlit.testing` on the real data (every page, footers, gauges, disclaimer, probability next to base rate, zone equals ledger, badge and faded note, Why? parts and links, leverage selector, refresh button, stale/missing/hold-out messages, no hold-out dates in plotted series, read-only checksum). Total suite: 505 tests, all pass with `-W error`.

### Setup on a fresh Mac
LightGBM's macOS wheel needs the OpenMP library (`libomp`). On a clean checkout without it, `import lightgbm` fails, 7 tests fail and 13 error, and the dashboard's Today page says "lightgbm could not load" (with this same fix) instead of showing "Models agree" and "Why?". Do ONE of these, then restart:
1. `brew install libomp` (the normal fix; needs Homebrew), OR
2. `sh scripts/fix_lightgbm_macos.sh` (no Homebrew: points the venv's lightgbm at the libomp that scikit-learn already ships inside `.venv`; re-run after any reinstall; undo with `uv sync --reinstall-package lightgbm`).
Then check with `uv run python -c "import lightgbm"` and `uv run pytest -W error -q`. **Stage 6's README must include this section** (it is the first thing a new user on a Mac will hit). `pyproject.toml` / `uv.lock` were not changed.

### Round-1 fixes (Stage 5 dashboard, after the round-1 critics; 2026-11-08)
Round 1: Leakage PASS (with test-strength notes), Statistical honesty FAIL, Dashboard FAIL, Code+test FAIL. Everything below is for correctness, honesty, robustness, clarity and test strength; the strategy, the badge thresholds and the saved results were NOT tuned or touched.
- **A. Robustness.** `run_page` now turns every failure into a plain message: damaged ledger/state (`PaperError`), missing price/result files, unreadable parquet (`pyarrow` errors), `OSError`, `ValueError`, an empty/missing trial log (Trials and the Reality-check overfitting panel say "results missing" instead of raising), and a catch-all. Each Today card handles its own failure so one bad asset does not hide the others. `explain.load_live_model_ex` returns the REAL reason (file missing / file unreadable / lightgbm could not load -> libomp fix with a pointer to "Setup on a fresh Mac"); the Today card shows that reason, not "saved live model is missing".
- **B. Clean checkout.** "Setup on a fresh Mac" section above; the dashboard message points to it.
- **C. Test strength / dead code.** ~100 new tests (`tests/test_round1.py`, `tests/test_dashboard_round1.py`) aimed at each surviving mutant (q25/q75, shares, decision day excluded from history, +-2 SE in table and chart, wiped-out flag, leverage chart L, 250-day windows, baseline columns, staleness direction, trial count, cache clearing, broad exception catch, >= in the confusion matrix, decile base-rate line, sidebar date, zone legend, Kelly text/marker, 20% line, PT verdict, "band includes zero", noisy rolling AUC that a forward window would change, mid-rank percentile) plus AMBER, GREEN and wiped-out paths. Dead imports/parameters removed; the data signature now includes the saved live models and price files (moved to `dash_data.data_signature`).
- **D. Honesty.** (D1) Brier skill is shown against the training base rate AND the evaluation-window base rate everywhere (Diagnostics, rolling chart, glossary, theory); the badge SKILL check uses the smaller of the two - a round-1, critic-driven, deliberately CONSERVATIVE change (declared in `BADGE_RULE`, `theory.yaml` and here; all four real badges are still grey and are re-derived independently in a test). (D2) the Trials luck bar is labelled a conservative upper bound that assumes 270 independent trials, with the correlated-trials and between-asset-dispersion caveats and a clear statement of the sqrt(252) vs sqrt(365) annualisation. (D3) Leverage tab: negative net-of-borrowing Kelly shows "0x (no leverage pays after 6% borrowing)", says which series/years Kelly uses, explains why Kelly and the table can disagree (2010-2024 bull market), and "growth falls to about zero at twice Kelly" is stated for the gross curve only. (D4) DSR table gets the excess-over-vol-targeted column and a "against zero only" caption. (D5) n/a for meaningless break-even costs; "Always up" merged into "Buy and hold"; the "Chosen strategy" is explained as config.yaml-fixed and an arbitrary dropdown model is labelled "Selected model's strategy". (D6) Strong-buy rule, all badge conditions, actual values per check, random-percentile multiple-testing note, the "proves nothing" warning at the top of Paper trading, hold-out-power wording (SE ~0.71 at 2 years), BTC cost qualifier (1-60 minute horizons).
- **E. Dashboard items.** Ordinal bug (23rd, 91st ...), dead-link annotations in `theory.yaml` ("(link no longer works)" for the IMF and Wikipedia links, "(could not be verified)" for quantpedia; `RESEARCH_REPORT.md` untouched), a "only N day(s) of data so far" note instead of one-point charts and date-only axes, a narrower Backtest table, a taller chart margin so the Trials axis title is not clipped, a Reality-check glossary and a plain reading of the project's own AUC table, Today page folded into three tabs (Meter / Why? / Badge and terms) with one-line needle explanation and a terms glossary, a beginner-readable badge formula, "almost every day" instead of "100%", a Bitcoin weekend-volume caveat, and a "Why?" conclusion that agrees with p versus the base rate.
- **F. Leakage latent note.** `confidence_bucket(..., asof=)` and `explain_today` only use predictions dated strictly before the decision day.
- **Not fixed / by design:** the three source links stay in the report (not edited); the sandbox cannot re-check the quantpedia link.


### Round-2 fixes (Stage 5 dashboard, after the round-2 critics; 2026-11-08)
Round 2 came back with FAILs on honesty, the dashboard, and test strength. Nothing about the strategy, the badge thresholds or the saved results changed, and the hold-out (from 2024-11-06) is still untouched. In plain words:
- **A. Honesty wording.** (1) The "Random-strategy percentile" now says it uses GROSS returns (before costs) and, for volatility-targeted series, mostly shows the vol targeting rather than the model's timing. Right next to it is the same test for plain vol-targeted buy and hold, computed from the saved data (SPY 98.2, QQQ 98.1, IWM 72.1, Bitcoin 66.2, against 85.4/93.4 for the model on SPY/QQQ). (2) The Trials page explains why the per-asset DSR (luck bar about 0.19 a year for SPY) differs from the pooled DSR (bar about 1.40) and says the pooled bar is the stricter reading; both luck bars are now table columns. (3) The Reality check page says the "needs a Sharpe above about 0.6 (SPY) / 0.87 (Bitcoin)" figure is the research report's own, and shows what THIS project's shorter out-of-sample samples need (3 / sqrt(years): SPY about 14.8 years -> 0.78, Bitcoin about 2.7 years -> 1.83). A break-even cost of 1% or more is shown with one decimal ("11.1%", not "11.117%").
- **B. Cosmetics.** The Diagnostics numbers are in a 3 + 2 grid so their labels are not cut off at 1366 px; the permutation-importance axis title is short and its ticks are plain numbers (no micro-unit letters); the wide table headers ("Sharpe +-2 SE", "Vol-targeted buy and hold ($)") get wider columns; the Trials histogram label sits beside the line with room; the Quantpedia link no longer says "(could not be verified)" (it answers HTTP 200 when re-checked with curl); Today says the needle is a relative ranking while the label is what the paper account does; the badge text lost the "round-1 critic-driven" wording and the "evaluation- window" typo; the "Why?" history phrase now reads "from 2000 to 2024" instead of "since 2000".
- **C. Leakage.** A new test proves the rolling Brier skill only looks backwards (it equals `metrics.brier_skill` on the window ending that day, and changing later rows changes nothing earlier). `today_report` / `load_live_model_ex` now refuse the saved live model when the day asked for is not the latest ledger day, or the model was trained after that day, and say so instead of silently using it. `confidence_bucket` returns an "unavailable" answer on an empty or too short history instead of crashing.
- **D. Robustness and dead code.** The libomp/lightgbm message is checked before the generic file-error message; a missing `state.json` with an intact ledger says "state file missing" (only a really empty ledger says "no paper-trading decision yet"); an empty predictions file gives a clear message instead of "Bin edges must be unique"; a Refresh where some assets were skipped is reported as a warning (amber, `ok=False`), not green; identical error messages appear once per page; any unexpected error stays inside its own Today card. Removed unused code: `explain.load_live_model`, `Why.as_dict`, `MeterReading.as_dict`, `dash_data.hit_rate_deciles`, `AssetResults.available`, `ui.ACCENT`, `ui.LEV_COLORS`.
- **E. Tests.** `tests/test_round2.py` (91 tests) pins the boundaries the critic's mutants slipped through: badge limits (skill > 0, PT > 0, 0.95 exactly, NaN), meter zone cuts, models-agree ties, confidence-bucket edges, ordinals, hold-out guards on metadata and trial returns, the cache key (ledger included), the equity curve and training-window choice, one failing card, paper tables, wipe dates, the luck-bar trial count, hold-out power numbers, chart details and live-model reasons. Some of them were hand-mutated to prove they fail (see the report); the lower-bound mutant in `ordinal` (10 vs 11) is equivalent and is only noted.


### Post-cap fixes (Stage 5 dashboard; approved by Wiktor after the 3-round cap; 2026-11-08)
Two things were fixed after the last critic round. Nothing about the strategy, the paper ledger, the badge rules or the saved results changed, and the hold-out (from 2024-11-06) is still untouched. In plain words:
- **1. A broken or missing `config.yaml` no longer shows a stack trace.** Before, the sidebar read the hold-out date from the config outside the page's safety net, so every page ended in a red Streamlit exception box. Now the sidebar sits inside the same safety net: you get one clear message ("config.yaml could not be used ... restore it from a backup"), the sidebar says the cutoff date is unavailable, and the page still finishes with its footer. New tests break the file two ways (`foo: [` and a deleted file) on all 7 pages, and also test the three round-3 guards that had no test: the "how does plain vol-targeted buy and hold rank" number (`vt_baseline_percentile` gives nothing for under 3 days, non-finite returns, empty inputs or an undefined percentile), the Refresh button with an unreadable config (it reports a failure, never raises), and the "nudged it down" and percentile phrases of the Why? conclusion.
- **2. The meter's needle can no longer disagree with the label.** Before, the needle showed how today's gap ranked among past days (for example the 6th percentile, in the red part of the dial) while the label showed what the paper account does (Hold / buy), because the account only steps aside when the gap is below minus its tolerance margin m. Now the dial is drawn on the gap itself (probability minus the usual up-day rate, in percentage points). The zone edges are the strategy's real thresholds on that same axis (the step-aside line at minus m, which moves with the adaptive margin), and the needle is today's gap. So the needle is always inside the zone of the paper position. The percentile survives as small print ("about the Nth percentile of past days"). The long/flat strategy only has two states, hold or step aside, so Reduce, Neutral and **Strong buy are crossed out for it: there is no such trading action, so none was invented** (they are real zones only in the optional 'sized' mode, where the edges are the points where the target position reaches 25%, 50% and 100%). The rule "the needle's zone equals the paper position's zone" is a unit-tested invariant (thousands of random and boundary cases for long_flat, default_long and sized, with and without vol-targeting, including a needle that is moved because of the no-trade band).
- **3. Simpler Today cards.** Each card now shows only the dial, the "53.7% vs 54.4% typical" line, the paper position and the evidence badge (plus the required disclaimer). The long explanations moved, unchanged in substance, to the "Badge and terms" tab (how to read the dial, models agree, badge reasons, glossary, decision date); the "Why?" tab is unchanged. The How-it-works zone table, the glossary and `theory.yaml` describe the new dial.


## Stage 6 Phase A: one-shot hold-out harness built and rehearsed (2026-11-09)

In plain words: the machine that will evaluate the hold-out exactly once is built and tested, and it was rehearsed on a PSEUDO hold-out made from the last 24 months of DEVELOPMENT data. The real hold-out (decision days from 2024-11-06, outcomes through 2026-11-06) has NOT been evaluated and no result file exists under `data/results/holdout/`.

### What `quant/holdout.py` does
- **Same strategy, nothing tuned.** It evaluates the strategy pinned in `config.yaml` (`leverage_strategy`: expanding window, ensemble of logreg+rf+lgbm, long/flat, vol-targeted, adaptive margin m) with the same code as Stage 3/4: `backtest.walk_forward` (monthly retrain, expanding window, 1-day gap, Platt calibration), `backtest.adaptive_positions`, `backtest.evaluate`, and the headline and stress costs of `backtest.cost_scenarios` (ETFs: 0.05%, 0.01% and a one-day delay; BTC: 0.15%, 0.60% stress and a one-day delay). The only change to existing code is an optional `first_date` argument of `walk_forward` (default None = identical behaviour) that skips blocks lying completely before the hold-out.
- **Adaptive m uses only what was known.** For each month the margin m is chosen from out-of-sample days strictly before that month: the saved Stage 3 history (dates before the cutoff) followed by the hold-out days as they happen.
- **Scoring convention.** Stage 3 blanked one label (the cutoff day's next-day return belongs to the hold-out). The hold-out therefore scores decision days from the cutoff day on, outcomes realised through the end date: no day is lost or counted twice. Every series (strategy and baselines) starts in cash and pays for its first entry.
- **Baselines on the same days:** buy and hold, vol-targeted buy and hold, always up, trend_200 (positions), climatology (training base rate, as the probability yardstick). Vol targeting uses the training-window volatility at each monthly retrain, identical for strategy and baseline.
- **Numbers per asset:** net Sharpe with a 95% interval (Lo's iid standard error; sqrt(252) for ETFs, sqrt(365) for BTC), CAGR, max drawdown, % invested, trades, turnover, accuracy with a Wilson 95% interval, AUC, Brier and log-loss skill versus climatology (training base rate AND the evaluation-window base rate), Pesaran-Timmermann statistic and p-value, and the difference of Sharpe ratios versus buy and hold and versus vol-targeted buy and hold with its standard error (Jobson-Korkie, Memmel's correction). All at headline and stress costs. A power note says why 24 months (Sharpe standard error about 0.7) can only expose failure.
- **`summary_numbers()`** (also `--significance-summary`) reads the saved Stage 3 results: 270 distinct trials, the chosen strategy's Deflated Sharpe Ratio per asset and with the pooled (all-assets, 270-trial) bar, Hansen SPA p-values, and whether t > 3 is cleared. Checked against the dashboard's own numbers in a test.

### Why the hold-out catches failure instead of proving success
It is one 24-month sample. The Sharpe standard error is about 0.7, so a good result is also what luck produces fairly often, while a clearly bad result (strategy far below the baselines, no directional skill) is informative. We can fail it, we cannot prove an edge with it. It is also evaluated only once, because looking twice and adjusting would turn it into just another tuned dataset.

### The one-shot rule (how it is enforced)
- `--run` needs `--confirm-single-final-evaluation`, refuses if `data/results/holdout/HOLDOUT_DONE.json` exists (any status) or if outputs exist without a marker, and refuses BEFORE creating the marker if the data do not cover every required day of the hold-out (so a data problem does not burn the one shot).
- The marker is written first, atomically and exclusively (temp file then hard link), with the timestamp, SHA-256 of `quant/*.py`, `config.yaml`, `uv.lock`, `pyproject.toml`, every Stage 3 result file, the live ledger (`data/paper/`, except `run.log`) and the feature datasets, plus the exact date range and row counts per asset. Then the outputs are written, the checksums are verified again, and the marker becomes `done`.
- Any crash (including Ctrl-C) makes the marker `failed` with the error; a killed process leaves `started`. Both block later runs. **Manual step to re-run:** read the failure, rename the marker (for example `mv HOLDOUT_DONE.json HOLDOUT_FAILED_<date>.json`) and write the reason here in PROGRESS.md. A re-run is then not a single evaluation and the report must say so.
- Isolation: results sit in `data/results/holdout/`; the dashboard loaders read named per-asset files, never list the results folder (a test scans for that), and dev loaders still raise on any hold-out date.

### Rehearsal and test evidence
- Dry run (`--dry-run`): pseudo-cutoff 2022-11-06, data loaded only up to 2024-11-06 (the real cutoff; the stored row of that day has its label blanked, exactly like Stage 3's dev frame), output into a temp dir, never the real folder. It refuses a pseudo-cutoff whose window would reach the real hold-out.
- Equivalence on REAL data (SPY and BTC-USD): the harness's predictions, base rates, positions, baseline positions and net returns equal the saved Stage 3 results for the same dates within 1e-9. The single intended difference is day 1's entry cost (a fresh account pays the whole entry, Stage 3 only the change from the previous day), and the test asserts that exact difference. No unexplained difference.
- Tests in `tests/test_holdout.py`: leakage (garbage after the end, changed future prices), one-shot guard (marker blocks re-run, failed status, confirm flag, coverage refusal, freeze violation, dry run never touches the real folder), metric maths recomputed independently (Wilson via statsmodels, Lo interval, Pesaran-Timmermann by hand, Brier/log-loss skill, Memmel formula), freeze manifest, dashboard isolation, summary numbers. A safety fixture makes any test that reaches the CLI with the real config fail instead of starting a real run.
- Incident (recorded for honesty): during development one of my own tests called the CLI with the real config and, through a test bug, wrote a `failed` marker into the real `data/results/holdout/` BEFORE any hold-out data was loaded or evaluated (the failure came from a test assertion ahead of `evaluate_all`). I removed that file by hand, the folder is gone again, and the safety fixture above prevents a repeat. No hold-out outcome was read, scored or saved.


### Stage 6 pre-registration (written 2026-11-09, BEFORE any hold-out outcome was read)

Beginner note: "pre-registration" means writing down the rules and fingerprints now, so nobody (including me) can quietly change them after seeing the hold-out result. A SHA-256 is a fingerprint of a file: if even one character changes, the fingerprint changes.

Fingerprints (SHA-256) of the files that define the experiment:
- `config.yaml`: `16509415a26b53a40475e6ee7f3a4d2a2aeae26267a6ceaf0845d62b0c416918` (this exact value is the constant `PREREGISTERED_SHA256["config.yaml"]` in `quant/holdout.py`; `--run` refuses to start if the live file differs)
- `uv.lock`: `084fa6a9a21ba350f2e968eebc5dce077c2aff57bec9117db0c80b7d4dfbb2ab` (recorded here and in the run marker; the freeze manifest compares it before and after)
- `pyproject.toml`: `4f21d856c606a55548a2bdeae55605586147922440373f705dc8af13b5c69f0e` (same)
- Check any time with `uv run python -m quant.holdout --verify-preregistration`.

Pinned values the real run asserts: cutoff 2024-11-06, 24 months, strategy expanding / ensemble / long_flat / vol-targeted / adaptive m, data in the project's own `data/` folder, and no `--config` option (`--run --config` is refused).

**Interpretation rule (pre-registered).** The hold-out can only DOWNGRADE evidence, never upgrade it. A positive hold-out profit or a positive Sharpe difference alone is never sufficient.
- Hold-out failure signals (headline cost, per asset): (1) the Sharpe difference vs vol-targeted buy and hold has its 95% interval entirely below 0; (2) Pesaran-Timmermann statistic < 0 with p < 0.05 (significantly negative directional skill); (3) the Brier score of the strategy model is significantly worse than the evaluation-window base rate (95% interval of the mean paired squared-error difference entirely above 0). The report prints these flags only.
- One-line verdict per asset, default **"no evidence of an edge"**. It becomes **"evidence"** only if ALL of these hold and no failure signal fired: SPA p < 0.05 against both baselines (development), pooled Deflated Sharpe (270 trials) >= 0.95, excess-over-vol-targeted-B&H Deflated Sharpe >= 0.95, t > 3 for the strategy, and t > 3 for its excess over vol-targeted B&H. **[SUPERSEDED in round 2, before any hold-out result existed: "weak evidence" now needs a cleared BASELINE test; see "Stage 6 round-2 fixes".]** Any failure signal forces the default. Buy and hold also clears t > 3, so t > 3 alone is never an edge; the report prints the baselines' t-statistics next to it.
- ETF stress is only the 1-day delay. No higher-cost ETF scenario was pre-registered, so none is added (it would be a new unlogged setting).


## Stage 6 round-1 fixes (2026-11-09)

In plain words: an independent critic found holes that could let the hold-out be evaluated more than once or be misread. All are closed; the hold-out is still untouched (no result folder exists).

1. **No way around the one-shot guard.** `--run --config other.yaml` used to let a different config (another cutoff, another strategy) slip through. Now `--run` refuses `--config` unless it is the project's own `config.yaml`, and `run_final` asserts the pinned cutoff, 24 months, the declared strategy and the project's `data/` folder. The dry run's "stay inside development data" guard now uses the pinned constant, not a config value, so an edited config cannot make a dry run read hold-out rows. `config.yaml`'s fingerprint is checked against the pre-registered value (see above) before the marker is written.
2. **Interpretation rule written down and coded.** `hold_out_flags()` lists the failure signals and `final_verdict()` maps development tests plus flags to the one-line verdict; both are pure functions with table tests, and the report prints the flags, the verdict and every reason, plus the baselines' t-statistics.
3. **Report honesty.** An attempt after a renamed `HOLDOUT_FAILED_*.json` prints "ATTEMPT N+1: NOT a single evaluation" and lists the files. The climatology accuracy row is labelled "always-up accuracy". The BTC section states the share of days invested (computed, not hardcoded) and says the comparison with buy and hold is then mostly a comparison with doing nothing; BTC's Brier skill table leads with the evaluation-window base rate. The ETF stress note says it is only the 1-day delay.
4. **Scored window asserted.** `evaluate_asset` now checks that the scored dates are exactly all NYSE sessions (ETFs) or all calendar days (BTC) from the cutoff to the day before the last row, and that the Stage 3 history ends on the last expected date before the cutoff (no gap, no overlap). Tests delete a month of rows or skip a whole retrain block and expect an error. (This also caught one of my own older tests that fed a frame whose last label was not blanked.)
5. **Preflight before the marker.** The LightGBM import, the Stage 3 histories, the dataset shape, coverage, the strategy lookups, the development significance numbers, the config fingerprint and the freeze manifest are all done BEFORE the marker is written. A failure there no longer burns the one shot (tested, including a simulated LightGBM import failure).
6. **More tests.** BTC with a missing weekend day is refused; the Wilson hit count uses rounding (the code already did, now in `hits_from_accuracy`, tested on every case where truncation would be wrong); climatology's evaluation-base log-loss skill uses the evaluation-window mean of y; a non-expanding window is refused.
7. README sentence reworded. 8. No new ETF scenario was added.


## Stage 6 round-2 fixes (2026-11-09)

In plain words: the round-2 critics found that the "weak evidence" label could be earned too easily, that the headline had no written rule, and that several important checks were not pinned by tests. Everything below was written and coded BEFORE any hold-out outcome was read; the real hold-out is still untouched (`data/results/holdout/` does not exist, `--run` was never started, `config.yaml` is unchanged).

Beginner note: a "baseline test" asks "is the strategy better than a simple alternative (buy and hold, or buy and hold with the same volatility targeting)?". A test "against zero" only asks "did it make money at all?", and in 2010-2024 even plain buy and hold passed that, so it says almost nothing about skill.

### A. Verdict rule, re-registered (supersedes the earlier "five tests" wording)
Default verdict per asset: **"no evidence of an edge"**.
- **Baseline tests** (the ONLY possible grounds for "weak evidence"): (B1) Hansen SPA p < 0.05 against BOTH buy and hold and vol-targeted buy and hold; (B2) excess-over-vol-targeted-B&H Deflated Sharpe >= 0.95 (asset-scoped); (B3) t > 3 for the excess return over vol-targeted B&H.
- **Necessary conditions** (never grounds for "weak evidence"): (N1) the strategy's own t > 3; (N2) the global Deflated Sharpe (270 trials) >= 0.95.
- **"evidence"** = B1, B2, B3, N1 and N2 ALL cleared and no hold-out failure signal.
- **"weak evidence"** = at least one baseline test (B1 to B3) cleared, not "evidence", and no hold-out failure signal.
- Otherwise, and ALWAYS when any hold-out failure signal fired (one flag is enough, "any" not "all"): **"no evidence of an edge"**.
- Thresholds are exact: `p < 0.05` (0.05 itself fails), `t > 3` (3.0 itself fails), `DSR >= 0.95` (0.95 itself passes). A missing or NaN number counts as not cleared.
- The test that used to pin "t > 3 only gives weak evidence" now pins the opposite: t > 3 only gives "no evidence of an edge" (real-life consequence: SPY and QQQ, whose own t is above 3 but whose baseline tests all fail, would read "no evidence of an edge" rather than "weak evidence").

**Disclosures (printed in the report notes and in the per-asset reasons):**
1. The excess DSR is asset-scoped (about 54 to 106 trials), the strategy DSR is global (270 trials), so the two bars are not equally strict.
2. SPA is a family-level test (is the best of everything tried better than the baseline?), not a test of the chosen strategy specifically.
3. t > 3 is a test against zero, which buy and hold also passes.

**Headline aggregation (registered as one line):** the headline verdict is the WEAKEST of the four per-asset verdicts, in the order no evidence of an edge < weak evidence < evidence. `headline_verdict(per_asset)` is a pure function (tested); the report prints the per-asset verdicts and the headline, and `holdout_summary.json` has `per_asset_verdicts` and `headline_verdict`.

### B. Test gaps closed (table-driven)
Boundary tables for `final_verdict` (SPA exactly 0.05 and 0.10, excess t exactly 3, strategy t exactly 3, DSR exactly 0.95 for both DSRs, one baseline test at a time) and `hold_out_flags` (Sharpe CI upper bound exactly 0, PT statistic exactly 0, PT p exactly 0.05, Brier lower bound exactly 0); Brier-only, PT-only and Sharpe-only failures each force the default; "any, not all" flags; the exact "hold-out failure signal(s): ..." reason line. Preflight tests: a history with a mid-history gap, a history ending early, a frame without label columns and a frame with a wrong last row each raise BEFORE the marker (no marker file), and a spy test proves `preflight` calls `check_history_contiguity` and `check_frame_shape` for every asset (deleting either call makes tests fail; I checked by hand-made mutants). `leverage_strategy` with an extra or missing key, and a wrong `project_root`, are refused. The scored-dates test now has a same-length shifted window and a same-length window with identical first and last dates but one wrong interior date, both refused (so the full `array_equal` check is pinned).

### C. Leakage notes
- `check_history_contiguity` now checks the WHOLE saved history against the expected calendar (every NYSE session / every calendar day between its first and last date, and the last date equals the last expected day before the cutoff). The real saved histories pass (tested on development results only).
- Adaptive-m test: a synthetic case where the chosen margin is known to change (0.0, then 0.005, then 0.02) shows that replacing the future leaves earlier margins and positions identical, that the later margins DO react to the new future, and that changing the past changes later margins (so the test can detect a peek). A harness-level test on the synthetic BTC world checks the same through `evaluate_asset`.
- **Code pre-registered:** combined SHA-256 of `quant/*.py` EXCLUDING `holdout.py` = `123064d95c59dab6a2a442284176a673defabde3b37252341dd08f2fa75627ba` (constant `PREREGISTERED_CODE_SHA256` in `quant/holdout.py`; computed last, as the sorted file-name to file-hash map hashed again). `--run` (preflight) and `--verify-preregistration` refuse if it differs. `holdout.py`'s own hash is written into the run marker (`holdout_py_sha256`) and also printed by `--verify-preregistration`. No other `quant/*.py` file was edited in this round. `config.yaml` is unchanged (same SHA-256 `16509415...6918`).

### D. README
Mentions `--verify-preregistration` and `--significance-summary`; the hash list says the manifest covers `quant/*.py`, `config.yaml`, `uv.lock`, `pyproject.toml`, Stage 3 results, features and the paper ledger; "evaluated once" is now future/neutral tense with the results placeholder to be filled after the run.

### E. Safety fixture
An autouse fixture wraps `run_final` so a test that passes the real config object (or any config pointing at the real project or data folder) fails with an AssertionError before anything runs; a guard test proves it (also through the CLI).

### Rehearsal (pseudo hold-out only, NOT a result)
See the dry-run numbers in the report of this round; they come from development data only and say nothing about the real hold-out.
- holdout.py SHA-256 immediately before the real run: ee456502ffdeab3ae71f68d0556aaa0c3050d611d7d11302511139bebefb40e7

### Stage 6 complete (2026-11-09)
The hold-out (decision days 2024-11-06 to 2026-11-05) was evaluated once with the pinned ensemble / long-flat / vol-targeted / adaptive-margin strategy. Result: no evidence of an edge on any asset (strategy Sharpe SPY 0.60, QQQ 0.54, IWM 0.68, BTC-USD -0.13 against buy-and-hold 0.81 / 0.72 / 0.50 / 0.22; every 95% interval is about plus or minus 1.4, so a 24-month hold-out can only catch failure, not prove success). Nothing was changed after the run. Where things are: results in `data/results/holdout/` (see `holdout_results.md`), summary at the top of `GAUNTLET_REPORT.md`, how to use everything in `README.md`. If you ever want to scrap the paper-trading ledger and start again, that is a Stage 4 decision, not a hold-out one: the hold-out must never be re-run to look for a better answer.
