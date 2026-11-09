# Quant Lab — Build Spec

Build a personal quant research tool for **S&P 500, Nasdaq-100, Russell 2000 small caps and Bitcoin** (extended to a ~30-asset portfolio in Stage 7) that does three things:
1. **Research + backtesting** — daily ML/statistical models tested honestly against buy-and-hold.
2. **Daily prediction dashboard** — today's signal for each asset with a probability.
3. **Paper trading** — simulated trades with fake money, updated each day.

This is a research and learning project. It must **never connect to a broker or place real trades.**

**Research basis.** `RESEARCH_REPORT.md` in this folder summarises what the academic literature says works for daily direction prediction, with formulas and paper links. This spec already includes its recommendations. Builders should look up exact formulas there (Yang-Zhang volatility, Pesaran-Timmermann, PSR/DSR, HAR-RV, etc.) and cite the paper in a code comment when implementing one.

**What to expect.** Careful studies find next-day direction for indices and Bitcoin is barely predictable: about 51–56% accuracy, AUC about 0.51–0.55. If a model shows much more than that, suspect a bug before believing a discovery. The best-evidenced improvement is **volatility targeting** (holding less when markets are turbulent), which is about sizing, not direction.

---

## How to run this build

Put this file and `RESEARCH_REPORT.md` in a folder called `quant-lab`, open Claude Code there, switch to auto mode so it can run unattended, and start with **one** of these:

**Option A — whole build in one go**
```
/goal Every stage in QUANT_SPEC.md is built using the Gauntlet Loop it describes; `uv run pytest` passes with 0 failures; `uv run python -m quant.run_daily` exits with code 0; `uv run streamlit run app/dashboard.py` starts without errors; and GAUNTLET_REPORT.md shows a final PASS from every critic for stages 1 to 7. Print the test summary and the final critic verdicts at the end.
```

**Option B — one stage at a time** (slower, but you can check each stage before moving on)
```
/goal Stage <N> of QUANT_SPEC.md is built using the Gauntlet Loop it describes; `uv run pytest` passes with 0 failures; and GAUNTLET_REPORT.md shows a final PASS from every critic for stage <N>. Print the test summary and the critic verdicts at the end.
```
Replace `<N>` with 1, then 2, and so on. Review the result before starting the next goal.

**If Stage 1 was built from an earlier version of this spec**, first build "Stage 1 follow-ups" below, then continue with Stage 2.

While a goal is active, don't stop to ask for approval between steps. Make a sensible decision, write it down in `PROGRESS.md`, and keep going. Only stop and ask me if something can't be undone or needs money, an account or a password.

---

## The Gauntlet Loop (how every stage is built)

You are the **lead agent**. For each stage:

1. **Plan.** Break the stage into small, independently checkable tasks. Write them in `PROGRESS.md`.
2. **Build.** Give each task to a **builder** subagent. Builders write code and tests for their task only.
3. **Gauntlet.** Send the finished stage to the **critic** subagents below. Each critic starts fresh, has not seen the builder's reasoning, reads the actual code, runs the actual commands, and returns:
   - a verdict: **PASS** or **FAIL**
   - for every FAIL: the exact problem, the evidence (file and line, command output or failing test) and what would fix it
4. **Fix.** Send every FAIL back to a builder with the critic's evidence. Then run the full gauntlet again, not just the critic that failed.
5. **Record.** After each round, append to `GAUNTLET_REPORT.md`: stage, round number, each critic's verdict and a one-line reason.

**Stop rules**
- A stage is done when **every critic returns PASS in the same round** and all tests pass.
- Maximum **3 rounds per stage.** If a critic still fails after round 3, record the open issue clearly in `GAUNTLET_REPORT.md` as "unresolved", explain why, and move on. Never fake a PASS.
- Don't run critics in parallel with builders on the same files.

### The critics

| Critic | Checks | Runs on |
|---|---|---|
| **Leakage auditor** | No lookahead anywhere: features only use data known at the close of day *t*; as-of joins are correct; no target information in features; train/test split respects time; scalers, calibrators, imputers, HAR-RV fits and thresholds are fit on training data only; calibration never uses shuffled k-fold; BTC rows with stale weekend/holiday cross-asset values are flagged. Tries to break it, e.g. by appending fake future data and checking that past features and predictions don't change. | Stages 2, 3, 4 |
| **Statistical honesty critic** | Baselines present (including the **climatology** baseline and **volatility-targeted buy-and-hold**); transaction costs applied, including the BTC stress case; hold-out untouched until Stage 6; **trial log complete** (every configuration tried is in it); thresholds chosen only on training data; **Pesaran-Timmermann computed with base rates**; Brier skill and log loss reported relative to climatology; metrics computed correctly (recompute Sharpe, max drawdown and PSR independently on a sample; BTC annualised with √365, ETFs with √252). Leverage simulator includes borrowing costs and leveraged trading costs, and is never used to pick a strategy. Flags overfitting signs, such as huge in-sample vs out-of-sample gaps or accuracy far above ~56%. | Stages 3, 4, 6 |
| **Code + test critic** | Code runs from a clean checkout with `uv sync`, tests are meaningful (not just "it runs"), errors are handled clearly, no secrets or hardcoded paths, data validation works, the daily run is idempotent and catches up after missed days. | Every stage |
| **Dashboard critic** | Starts the dashboard, loads every page, checks that charts render with real data, numbers match the saved results files, probabilities are shown next to the base rate, the meter's zone matches the paper-trading position and its evidence badge matches the saved significance results, every number quoted in a "Why?" explanation matches the feature data for that day, theory text matches `RESEARCH_REPORT.md` with working source links, layout is readable on a laptop screen, the disclaimer is visible, and a beginner could understand the "Today" and "Reality check" pages. | Stage 5 |

### The quality bar (what critics judge against)

- **Correctness and honesty come first.** A strategy that loses to buy-and-hold, reported honestly, is a PASS. A strategy that "wins" because of leakage, missing costs, an unlogged search or a tuned hold-out is a FAIL.
- **Critics must never fail work because returns are low.** Pushing the builder to "improve performance" round after round is exactly how overfitting happens. Critics judge the process, not the profit.
- Every number shown in the dashboard or summary can be traced back to a saved results file and reproduced by re-running the code.

---

## Tech stack
- Python **3.12**, pinned in `pyproject.toml` and `.python-version`; environment managed with **uv**
- pandas, numpy, yfinance, scikit-learn, lightgbm, statsmodels, pyarrow, **arch** (for Hansen's SPA test), exchange_calendars, and in Stage 7 **ccxt** (free hourly crypto data)
- **Streamlit + Plotly** for the dashboard
- pytest for tests
- No PyTorch/TensorFlow. The research shows deep learning adds at most about 1 percentage point of accuracy on this kind of problem, often not significantly.
- Keep total disk use around 1 GB including the environment

## Assets and data (in `config.yaml`, easy to change)
| Name | Ticker | Notes |
|---|---|---|
| S&P 500 | `SPY` | ETF, adjusted close includes dividends |
| Nasdaq-100 | `QQQ` | ETF, adjusted close includes dividends |
| Russell 2000 | `IWM` | small-cap ETF, adjusted close includes dividends |
| Bitcoin | `BTC-USD` | trades 7 days a week |

Extra feature sources: `^VIX`, `^VIX3M` (starts 2007), `^VXN` (Nasdaq-100 volatility index) and `^RVX` (Russell 2000 volatility index; if Yahoo doesn't serve it, log that in `PROGRESS.md` and use `^VIX` for IWM instead). Use ETFs rather than raw indices because they're actually tradable and give an honest buy-and-hold benchmark.

For Stages 1–6, **don't add** Treasury yields, the dollar index, credit spreads, gold or Google Trends as features. The research found no evidence they help at a daily horizon, and Trends data gets revised, which hides lookahead. Stage 7 adds a wider ETF universe, hourly crypto data and FRED macro data in a controlled way. Never add Google Trends.

## Project structure
```
quant-lab/
  QUANT_SPEC.md        # this file
  RESEARCH_REPORT.md   # literature review: formulas, evidence, sources
  PROGRESS.md          # plan, decisions, what was done
  GAUNTLET_REPORT.md   # critic verdicts per stage and round
  config.yaml          # assets, dates, costs, thresholds, starting capital
  data/                # parquet files, gitignored; under 10 MB until Stage 7, about 50 MB after
    fomc_dates.csv     # scheduled FOMC announcement dates (hard-coded)
    results/           # backtest outputs + trials.parquet
  quant/
    data.py            # download + incremental update + caching
    features.py        # feature engineering
    models.py          # baselines + ML models + calibration
    backtest.py        # walk-forward validation + strategy simulation
    metrics.py         # prediction, trading and significance metrics
    trials.py          # trial log: every configuration tried
    paper.py           # paper trading ledger
    explain.py         # turns today's model inputs into plain-English reasoning
    theory.yaml        # one entry per feature group: the theory, evidence and paper links
    run_daily.py       # one command: update data -> predict -> update paper trades
    universe.py        # Stage 7: multi-asset ETF + crypto universe
    portfolio.py       # Stage 7: trend-following, sector rotation, pooled ML ranking
    hourly.py          # Stage 7: hourly crypto data via ccxt + intraday features
    macro.py           # Stage 7: FRED downloads with publication lags
  app/dashboard.py     # Streamlit app
  tests/
  README.md
```

---

## Stage 1 — Data
- Download full daily history (SPY/QQQ since 1999, IWM since 2000, BTC since 2014, `^VIX`, `^VIX3M`, `^VXN`, `^RVX`) via yfinance, store as parquet in `data/`.
- Later runs only download missing days (incremental update).
- Handle calendars correctly using the NYSE calendar (`exchange_calendars`): stock ETFs trade weekdays only and skip US holidays; BTC trades every day. Never forward-fill prices into the target.
- Never store an incomplete bar: drop today's SPY/QQQ/IWM bar if the NYSE session hasn't closed yet, and today's BTC bar until the UTC day has ended.
- Validate data: no duplicate dates, no gaps beyond holidays, no zero/negative prices. Log any problems.
- If yfinance fails, show a clear error. **Flag stale data**: if the latest stored date is older than the most recent completed trading day (NYSE for ETFs, yesterday UTC for BTC), show a clear warning. Never silently use stale data.
- Cost keys in `config.yaml` use the tickers (`SPY`, `QQQ`, `BTC-USD`).
- Add `data/fomc_dates.csv` with **scheduled** FOMC announcement dates since 1999 (unscheduled meetings weren't knowable in advance, so leave them out).

### Stage 1 follow-ups (only if Stage 1 was built from an earlier spec)
Bring the existing Stage 1 up to the list above: pin Python 3.12 and re-sync; add `IWM`, `^VIX3M`, `^VXN`, `^RVX` and `fomc_dates.csv`; add the stale-data warning and the incomplete-bar rule with tests; rename cost keys to tickers; fix any deprecation warnings. Run the Code + test critic on the changes and record it in `GAUNTLET_REPORT.md` as "Stage 1 follow-ups".

## Stage 2 — Features (no lookahead!)
For each asset, features at day *t* may only use information known at the close of day *t*. The target is about day *t+1*.

**Core features**
- Lagged log returns (1, 2, 3, 5, 10, 20 days)
- **Short-term reversal** `R5` = sum of the last 5 daily log returns. This is the best-supported daily equity signal in the research (daily index returns have tended to reverse since about 2000).
- **Overnight gap** `ln(Open_t / Close_{t−1})` and **intraday return** `ln(Close_t / Open_t)` (ETFs)
- Volatility: close-to-close rolling volatility (10, 20, 60 days), **Yang-Zhang 21-day volatility** from OHLC, and volatility ratios
- **Trend** (documented momentum horizons per asset): `ln(C_t / C_{t−k})` for k = 63, 126, 252 trading days (ETFs) and k = 7, 14, 21, 28 days plus a "20-day new high" flag (BTC); distance from the 50- and 200-day moving averages
- **One** of RSI(14) or the 20-day z-score, not both (they repeat information already in returns and moving averages). Don't add MACD or Bollinger bands.
- Volume change vs 20-day average

**Volatility / stress features** (slow regime gauges, the strongest published vol-based signals)
- VIX level and 1-day change
- **VIX term structure** `VIX / VIX3M` and a backwardation flag `1{VIX > VIX3M}`
- **Variance risk premium**: `(VIX/100)² − RV_21` for SPY and `(VXN/100)² − RV_21` for QQQ, `(RVX/100)² − RV_21` for IWM, where `RV_21` is the asset's own annualised 21-day realised variance

**Calendar features (describe the predicted day t+1)**
- Day of week of t+1; for BTC, `is_weekend(t+1)`. These are known in advance, so they're not lookahead.
- Pre-FOMC flag: t+1 is a scheduled FOMC announcement day (SPY/QQQ/IWM only)
- Don't add pre-holiday or turn-of-month features. They haven't been significant for the S&P 500 in recent decades.

**Cross-asset features**
- For BTC: the latest SPY/QQQ/IWM return and VIX values *as of* that date (as-of join, never future data), plus a **`stale` flag** when those values were carried over from a Friday or holiday
- Rolling 60-day BTC–SPY return correlation
- For IWM: the IWM−SPY 5-day return spread (small caps vs large caps), known at the close of t

**Note on IWM for the leakage auditor and honesty critic:** small-cap *indices* show day-to-day autocorrelation partly because many small stocks trade rarely, so the index price updates late (Lo & MacKinlay 1990). The IWM ETF trades continuously, so features and targets must use IWM's own prices, never the `^RUT` index, or the backtest will show a pattern nobody could trade.

**Conventions**
- Annualise with √365 for BTC and √252 for ETFs everywhere.
- Add a `feature_groups` mapping (returns, reversal, volatility, VIX/stress, trend, calendar, cross-asset). It's used for importance and dashboard explanations.
- Missing values: `^VIX3M` starts in 2007. LightGBM can handle NaN; for logistic regression and random forest, impute with the training-window median and add a missing flag. Document the rule in `PROGRESS.md`.

**Target:** sign of the next day's return (up/down). Also store the next-day return itself for the backtest.

**Required test:** compute features on the full dataset, then on the dataset truncated at a random date. Features up to that date must be identical. This catches lookahead bugs.

## Stage 3 — Models, validation and metrics

### Baselines (models must beat these to mean anything)
- **Climatology:** a constant probability equal to the training window's up-day frequency. This is the honest yardstick for probability metrics, because stocks rise on slightly more than half of days.
- Buy and hold
- **Volatility-targeted buy and hold:** weight `w_t = min(1, σ_target / σ̂_t)` with σ̂ from data through day t (default σ_target = the asset's long-run training-window volatility). This is the best-evidenced edge in the literature (Moreira & Muir 2017; Harvey et al. 2018) and may capture most of the achievable improvement.
- Always predict "up"
- Simple trend rule (long if price > 200-day MA)

### Models
Use small, fixed, regularised settings. Flexible models overfit daily index data; in one careful study plain logistic regression beat random forest.
- **Logistic regression** with L2 penalty, scaled features, modest `C`. This is the model everything else has to beat.
- **Random forest** with large `min_samples_leaf` (e.g. ≥ 100)
- **LightGBM** with `num_leaves ≤ 15`, `min_data_in_leaf ≥ 100`
- **Equal-weight ensemble:** average of the three calibrated probabilities (forecast averaging is the most robust approach in the research)

Any tuning happens only inside training windows and every setting tried is logged as a trial.

### Calibration
Inside each walk-forward step: fit the model on the first ~80% of the training window, then fit **sigmoid (Platt) calibration** on the last ~20% (at least 500 days) after a 1-day gap, using `FrozenEstimator`. Never use `CalibratedClassifierCV` with shuffled `cv=5`. Use isotonic only if there are ≥ 1,000 calibration points.

### Validation
- Walk-forward with an expanding window, retraining monthly. Never shuffle time series.
- Leave a 1-day gap between training data and prediction (the target uses the next day).
- Also run a **rolling 4-year window** variant for BTC, logged as a trial (Bitcoin's market structure changed a lot).
- Keep the **last 24 months as a final hold-out**, untouched until Stage 6, and evaluated **once**. Write in `PROGRESS.md` that its job is to catch failure, not to prove success: a 12-month hold-out has a Sharpe standard error of about 1.0, and even 24 months (about 0.7) is mainly useful for showing that a strategy *isn't* working.

### Trial log
Write every configuration tried (model × feature set × threshold × strategy mode × window) to `data/results/trials.parquet` with a config ID and its out-of-sample daily net returns. This is needed for the Deflated Sharpe Ratio and makes overfitting visible.

### Strategies
- **Long/flat (default):** thresholds are relative to the training-window base rate, not an absolute 0.55. Go long when `p ≥ base_rate − m`, flat otherwise, with `m` chosen only on training data. Report m = 0 as a reference.
- **Default long, step aside:** hold 100% unless the probability falls clearly below the base rate. With an upward-drifting asset, being flat is costly, so this trades much less.
- **Volatility-targeted versions** of both, using the same `σ_target / σ̂_t` scaling as the baseline.
- Optional **long/short** in config.

### Costs
Subtract costs on every position change. Headline: 0.05% for SPY/QQQ/IWM, 0.15% for BTC. Sensitivities: 0.01% for ETFs, and a **0.60% BTC stress case** (what a small retail account actually pays at Coinbase's lowest tier). Also report **break-even cost** = gross mean return ÷ mean turnover, and a **one-day execution delay** sensitivity (decide at close t, earn the return of t+2).

### Metrics
- **Prediction:** accuracy vs base rate; ROC AUC; log loss and **Brier skill score relative to climatology** (`1 − Brier_model / Brier_climatology`); reliability curve with bin counts; **hit rate by confidence decile**; **Pesaran-Timmermann** p-value (statsmodels `pesaran_timmermann`), showing the share of up days and the share of "up" predictions.
- **Trading:** total return, CAGR, annualised volatility, Sharpe, Sortino, max drawdown, % time invested, number of trades, turnover, break-even cost.
- **Significance:**
  - Keep the 1,000 random strategies with the same time-in-market and report the percentile.
  - **Probabilistic Sharpe Ratio** and **Deflated Sharpe Ratio** using the trial count from the trial log (formulas in `RESEARCH_REPORT.md`).
  - **Diebold-Mariano** on log loss: each model vs logistic regression and vs climatology (check that the installed statsmodels has `diebold_mariano_test`; if not, implement it with the Harvey-Leybourne-Newbold correction).
  - **Hansen's SPA test** via `arch.bootstrap.SPA`: do any strategies beat buy-and-hold after accounting for trying several?
  - t-statistic of mean net return, compared with the **t > 3** hurdle (Harvey, Liu & Zhu 2016).
- **Feature importance:** out-of-sample permutation importance computed per **feature group**, not LightGBM gain importance (which is biased and splits credit among similar features).

Save all results to `data/results/` so the dashboard loads instantly.

### Leverage simulator (what-if only, fake money)
A tool for seeing what leverage does, not a way to find a better strategy.
- For buy-and-hold, vol-targeted buy-and-hold and the chosen strategy of each asset, simulate daily-rebalanced leverage L = 1×, 1.5×, 2× and 3×:
  `r_L,t = L·r_t − (L − 1)·borrow_rate/days_per_year − costs on L × turnover`
  with `borrow_rate` in config (default 6% a year; 365 days for BTC, 252 for ETFs). Leveraged ETFs and margin rebalance daily, which is where "volatility drag" comes from.
- Report per L: CAGR, max drawdown, worst day, worst month, longest time to recover, and a **"wiped out" flag** if equity ever falls below 20% of its starting value (a broker would have force-sold long before that).
- Show the **growth curve** `g(L) ≈ L·μ − L²·σ²/2` estimated from training data, with the **Kelly leverage** `L* = μ/σ²` and its uncertainty band. This shows that past a point, more leverage means *lower* long-run growth, and that 2× Kelly gives roughly zero growth (MacLean, Thorp & Ziemba 2010; formulas in `RESEARCH_REPORT.md`).
- Highlight the historical crashes on the chart (e.g. 2008, March 2020, 2022 for BTC) so the leveraged drawdowns are easy to see.
- Leverage results are **not** trials, are never used to choose a strategy, and are never applied to the hold-out selection. Paper trading stays unleveraged.

### Optional experiments (only after everything above passes; each logged as a trial)
- Dead-zone training label: drop days with `|r_{t+1}| < 0.1–0.25·σ_t` from **training only**, never from evaluation or P&L
- 5-day-direction target (theory suggests intermediate horizons are more predictable; needs a 5-day purge and multi-day holding)
- HAR-RV next-day volatility forecast (statsmodels OLS on each training window) as a feature and for sizing
- **Don't add:** triple-barrier labels, hidden Markov regimes, deep learning, combinatorial purged CV

## Stage 4 — Paper trading
- Fake starting capital per asset (default $10,000 each, configurable).
- `run_daily.py`: update data → retrain if a month has passed → predict next day → record intended position at today's close in a ledger (parquet) → mark yesterday's positions to market.
- The ledger records the **calibrated probability, the base rate used, the position size, and the config ID** from the trial log, so live results can be compared with the backtest of exactly the same configuration.
- Strategy mode is set in config: long/flat (default), default-long step aside, or **sized mode**: `position = clip(λ·(p − base_rate)/scale · σ_target/σ̂_t, 0, 1)` with λ ≈ 0.25–0.5 (fractional Kelly) and a no-trade band of 0.1 so it doesn't churn.
- Track positions, trades, costs, equity and P&L per asset, plus buy-and-hold and vol-targeted buy-and-hold over the same period.
- Track a running **live Brier score and hit rate** vs the backtest's, so decay is visible as it happens.
- Must be safe to run twice in one day (idempotent) and must catch up correctly after missed days.
- Don't set up any scheduled or automatic job. Instead, write in `PROGRESS.md` how I could schedule it on my laptop later, and I'll decide.

## Stage 5 — Dashboard (Streamlit)

### The signal meter
Each asset gets a semicircle gauge (Plotly indicator) with five zones: **Sell / step aside · Reduce · Neutral · Hold / buy · Strong buy**.
- The needle is driven by how far today's calibrated probability is from the base rate, expressed as a **percentile of that gap over the past out-of-sample history** (so "Strong buy" means "one of the most positive readings this model has given", not a fixed number).
- Zone boundaries come from the same thresholds the active strategy uses, so the meter always agrees with the paper-trading position.
- "Sell" means **reduce or step aside**, not short-selling, because the project is long-only by default.
- Under the meter, show: the probability vs base rate, how many of the 3 models agree, and an **evidence badge** from the backtest: "No proven edge" (grey), "Weak evidence" (amber) or "Evidence" (green), based on the Pesaran-Timmermann p-value, Deflated Sharpe Ratio and whether it beat vol-targeted buy-and-hold after costs. If the badge is grey, the meter is drawn faded with the note "This model hasn't shown a reliable edge — treat this as a learning tool."
- The meter is never shown without the disclaimer.

### "Why?" explanations
Every prediction comes with reasoning a beginner can follow, generated by `quant/explain.py` from templates (no external AI service or API key):
1. **What the model saw:** for the top 3–4 feature groups pushing the probability up or down (logistic-regression coefficient × value, or LightGBM `pred_contrib`, aggregated by group), one sentence with today's actual number and what's normal. Example: *"SPY fell 3.1% over the last 5 days, more than 90% of 5-day periods since 2000."*
2. **The theory behind it:** from `quant/theory.yaml`, one or two plain-language sentences on why that signal might matter, plus the paper and link. Example: *"Short-term reversal: since about 2000, daily index moves have tended to partly bounce back, possibly because big funds rebalance after sharp moves (Baltussen, van Bekkum & Da 2019)."*
3. **How much to trust it:** how often this signal has worked out of sample for this asset (hit rate in that confidence bucket), and the caveat from the research (e.g. "fragile after trading costs").
4. **The conclusion:** one sentence linking the above to the meter reading, e.g. *"Reversal and calm volatility point slightly up, the VIX term structure is neutral, so the model leans to Hold / buy."*

`theory.yaml` has one entry per feature group (returns, reversal, volatility, VIX/stress, trend, calendar, cross-asset) with: a plain-English explanation, the formula, the evidence and its caveats, and source links, all taken from `RESEARCH_REPORT.md`. Explanations must only describe what moved the model, never claim to know why the market will move.

### Pages
1. **Today** — per asset: the **signal meter**, the **"Why?" explanation** (expandable "Learn more" for each theory), position size, last update time and any stale-data warning. Probability is always shown **next to the base rate** (e.g. "52% vs 54% typical"), so a below-average day isn't shown as bullish.
2. **Backtest** — pick asset, model and strategy mode. Equity curve vs buy-and-hold **and vol-targeted buy-and-hold**, drawdown chart, metrics table comparing all models and baselines, Sharpe with a ±2 standard-error band, break-even cost, random-strategy percentile. A **Leverage** tab with a 1×/1.5×/2×/3× selector: leveraged equity curves (log scale) with crashes highlighted, the drawdown and "wiped out" table, and the growth-vs-leverage curve with the Kelly point marked. A plain-language note explains why a small edge plus leverage usually ends badly.
3. **Model diagnostics** — reliability curve with bin counts, hit rate by confidence decile, **rolling 250-day AUC and Brier skill score** (60-day accuracy has a standard error of about 6.5 percentage points, so it's mostly noise), group permutation importance, Pesaran-Timmermann p-value, confusion matrix.
4. **Paper trading** — equity curves, open positions, trade log, P&L vs both baselines, live vs backtest Brier score.
5. **Trials** — how many configurations were tried, and the resulting Deflated Sharpe Ratio, explained in plain language: the more you try, the higher a Sharpe you get by luck alone.
6. **How it works** — a beginner's guide built from `theory.yaml` and `RESEARCH_REPORT.md`: each signal and model explained with its theory and formula, how walk-forward testing works, what calibration means, and how to read the meter and evidence badge.
7. **Reality check** — plain-language evidence from `RESEARCH_REPORT.md`: realistic AUC is 0.51–0.55; famous ML papers rank hundreds of stocks against each other, which is easier than timing one index; those edges went negative after costs after about 2010; published signals lose about 58% of their returns after publication; why a 1–2 year hold-out can't prove an edge; what overfitting looks like and how to read the significance tests.

Requirements:
- Clean dark theme, readable on a laptop screen
- A "Refresh data" button that runs the daily update
- A disclaimer in the footer: research and education only, not financial advice

## Stage 6 — Wrap-up
- Run the hold-out evaluation **once** and save the results. Don't change any model or setting afterwards.
- Report hold-out results with **confidence intervals** for Sharpe and accuracy, Pesaran-Timmermann p-value, Brier skill vs climatology, and comparison against both buy-and-hold and vol-targeted buy-and-hold, at headline and stress costs.
- README with setup (`uv sync`), how to run the backtest, the daily update and the dashboard, and how to change assets or settings.
- All tests pass.
- A short summary at the top of `GAUNTLET_REPORT.md`: total trial count, the Deflated Sharpe Ratio of the chosen strategy, the SPA p-value, whether it clears t > 3, any unresolved critic issues, and a one-line verdict: **"no evidence of an edge"**, **"weak evidence"** or **"evidence"**.

## Stage 7 — Multi-asset portfolio and extra data
Built only after Stage 6 is complete. Nothing from Stages 1–6 is changed except where noted; the four-asset models and their results stay as they are. This stage adds what real quant funds rely on most: **breadth** (many small edges averaged together) plus two free extra data sources.

### 7a — Universe (free ETFs + two cryptos, in `config.yaml`)
| Group | Tickers |
|---|---|
| US equity | SPY, QQQ, IWM |
| US sectors | XLK, XLF, XLE, XLV, XLI, XLY, XLP, XLU, XLB (since 1998); XLRE (2015), XLC (2018) |
| International equity | EFA, EEM, EWJ, VGK |
| Bonds | SHY, IEF, TLT, TIP, LQD, HYG |
| Real assets | GLD, DBC, VNQ |
| Dollar | UUP |
| Crypto | BTC-USD, ETH-USD |

- Each asset enters the backtest only from its own start date plus warm-up (no back-filling).
- **Survivorship note** for `PROGRESS.md` and the Reality check page: this list was chosen today, so it only contains ETFs and coins that still exist. That bias is much smaller than for individual stocks, which is why the universe uses ETFs and not today's S&P 500 members. Don't add more cryptos for the same reason.
- `data/` may now grow to about 50 MB.

### 7b — Portfolio strategies (rebalanced weekly or monthly, long-only by default)
Baselines:
- Equal weight across available assets; **inverse-volatility weight** (each asset gets the same risk); **60/40** SPY/IEF.

Strategies:
1. **Time-series momentum / trend-following** (Moskowitz, Ooi & Pedersen 2012; Hurst, Ooi & Pedersen "A Century of Evidence on Trend-Following"): for each asset, signal = sign of the 1-, 3- and 12-month returns averaged; hold the asset only when the signal is positive (long/flat; long/short as an optional config); size each position by inverse volatility; scale the whole portfolio to a **10% annual volatility target**.
2. **Sector rotation** (industry momentum, Moskowitz & Grinblatt 1999): rank the sector ETFs by 12-month return skipping the latest month; hold the top 3 equally weighted; rebalance monthly.
3. **Pooled ML ranking** (the setup behind Krauss et al. 2017, applied to ETFs): one logistic regression and one LightGBM trained across all assets on the Stage 2 feature families, predicting whether each asset beats the cross-sectional median over the next **21 trading days**; hold the top 5 with inverse-volatility weights. Purge 21 days between training and test.

Rules:
- Same walk-forward, calibration, trial log, cost rules (0.05% ETFs, 0.15% crypto, plus the 0.60% crypto stress case), PSR/DSR, SPA test and t > 3 hurdle as Stage 3.
- Max weight per asset 25%; crypto in total at most 10%.
- Hold-out: the same final 24 months. These portfolio strategies were **pre-registered in this spec on 2026-11-07, before the Stage 6 hold-out was run**, and must be built exactly as written; don't change them because of Stage 6 results. Their hold-out is evaluated **once** at the end of this stage.
- **Honesty note:** the Stage 6 hold-out results for SPY, QQQ, IWM and BTC are already known, so for the portfolio this is a **partially seen hold-out**. Label it that way in `GAUNTLET_REPORT.md` and on the dashboard, and record the pre-registration date.
- **Forward test (the truly unseen data):** from the day Stage 7 is finished, the portfolio paper-trading ledger is the real out-of-sample test. Show its results separately on the Portfolio page, with the start date and number of days, and never use them to change a strategy.
- Metrics also include: average pairwise correlation of holdings, contribution to return and risk by asset group, turnover.

### 7c — Hourly crypto data (`ccxt`, free public data, no account or API key)
- Download hourly OHLCV for BTC/USDT and ETH/USDT from 2017 via `ccxt` public endpoints (exchange set in config; default Binance, fallback Kraken or Coinbase if blocked). Incremental updates, same validation and stale-data rules as Stage 1. Note in `PROGRESS.md` that USDT prices differ slightly from USD.
- Align hourly bars to the Yahoo daily close (00:00 UTC). Daily features may only use hours up to that close.
- New daily features for BTC and ETH: **realised volatility from hourly returns** (a much better volatility estimate than daily data), the return over the last 6 and 24 hours before the close, hourly volume concentration, and a **HAR-RV** forecast fitted on hourly-based realised volatility (Corsi 2009). Test them in the BTC model and the portfolio's volatility targeting, logged as trials.
- Don't use Yahoo's intraday data for ETFs: it only goes back about 2 years.

### 7d — Macro data from FRED (free CSV downloads, no API key)
- Series: 3-month T-bill (`DTB3`), fed funds (`DFF`), 10-year yield (`DGS10`), 10y–2y term spread (`T10Y2Y`), high-yield credit spread (`BAMLH0A0HYM2`), 10-year breakeven inflation (`T10YIE`).
- **Lookahead rule:** FRED dates are observation dates; treat a daily value as known only from the **next** trading day. Don't use monthly series like CPI (they're published weeks later and revised).
- Uses:
  1. **Risk-free rate** for Sharpe ratios and the **borrowing rate** in the leverage simulator (fed funds + 1.5%), replacing the fixed 6%. Re-running Stage 3–6 metrics with this is allowed and logged.
  2. **Regime context** in the "Why?" explanations (e.g. "the yield curve is inverted").
  3. Macro features in the portfolio and four-asset models **as an experiment only**, logged as trials. Welch & Goyal (2008) found these predictors fail out of sample, so the expected result is no improvement.

### 7e — Dashboard additions
- **Portfolio** page: today's target weights with the meter and "Why?" explanation per asset (which trend signals are on or off, and why), weights over time as a heat map, equity curves vs the three baselines, drawdowns, contribution by asset group, and a correlation matrix.
- **Macro** panel on the Today page with the current rates and spreads, labelled as context, not a prediction.
- `run_daily.py` also updates the universe, hourly crypto and FRED data, and the portfolio paper-trading ledger (unleveraged, $100,000 fake capital).
- The Reality check page adds why breadth helps (many weak, uncorrelated edges average into a steadier result) and the survivorship note.

### 7f — Wrap-up
Add a Stage 7 section to the `GAUNTLET_REPORT.md` summary: trial count, Deflated Sharpe and SPA p-value for each portfolio strategy, hold-out results vs the equal-weight, inverse-vol and 60/40 baselines, and the same one-line verdict ("no evidence of an edge" / "weak evidence" / "evidence"). Update the README.

All four critics run on Stage 7. The leakage auditor pays particular attention to the hourly-to-daily alignment, FRED publication lags, assets entering the universe late, and the 21-day purge.

## Ground rules
- Be honest in results. If nothing beats buy-and-hold or vol-targeted buy-and-hold, say so clearly. Don't tune the hold-out period.
- Don't add complexity (deep learning, sentiment scraping, extra data sources, etc.) unless I ask.
- Log key decisions briefly in `PROGRESS.md` with the reason. I'm learning how quant research works, so write them so a beginner can follow, and link to the relevant section of `RESEARCH_REPORT.md` where it helps.
