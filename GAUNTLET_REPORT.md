# GAUNTLET REPORT

## Summary (final; written after the single hold-out evaluation)
**Verdict: no evidence of an edge.**

| Item | Result |
|---|---|
| Total trials | 270 distinct configurations (360 logged config ids including identical duplicates); per asset: SPY 56, QQQ 54, IWM 54, BTC-USD 106 |
| Deflated Sharpe Ratio of the chosen strategy (global, 270 trials) | SPY 0.017, QQQ 0.057, IWM 0.0003, BTC-USD 0.000 (needs 0.95). Asset-scoped DSR is higher (SPY 0.993, QQQ 0.999, IWM 0.730, BTC-USD 0.000) but is a test against zero; the excess-over-vol-targeted-buy-and-hold DSR is 0.000 / 0.001 / 0.0001 / 0.011 |
| Hansen SPA p-value (vs buy-and-hold / vs vol-targeted buy-and-hold) | SPY 0.998 / 0.894, QQQ 0.940 / 0.707, IWM 0.976 / 0.709, BTC-USD 0.933 / 0.937 (nothing near 0.05) |
| Clears t > 3? | Strategy alone: SPY 3.24 and QQQ 3.76 yes, IWM 1.70 and BTC-USD -0.88 no. But buy-and-hold also clears it (SPY 3.25, QQQ 3.64) and the strategy's excess t over vol-targeted buy-and-hold is negative for all four (-2.01, -0.83, -1.35, -1.42), so t > 3 is not evidence of an edge |
| Hold-out (2024-11-06 to 2026-11-05, evaluated once) | Strategy Sharpe SPY 0.60, QQQ 0.54, IWM 0.68, BTC-USD -0.13 vs buy-and-hold 0.81 / 0.72 / 0.50 / 0.22; all 95% CIs about plus or minus 1.4 (a 24-month hold-out can only catch failure). QQQ raised a pre-registered failure flag (Brier significantly worse than the base rate). No model, feature or setting was changed after the run |
| Unresolved critic issues | Stage 3: four minor documentation items (see Stage 3 round 4). Stage 5: test gap for the drawn gauge needle when the sized-mode no-trade band clamps it, a few cosmetic dashboard items, two dead source links copied from RESEARCH_REPORT.md (see Stage 5 post-cap fixes). Stage 6: the BTC-USD table in `holdout_results.md` lists the Brier-skill columns eval-base first, the ETF tables train-base first (values are correct; presentation only); QQQ's Brier failure flag is one of 12 pre-registered flag tests with no multiplicity adjustment; non-blocking items from round 3 (no end-to-end test of failure flags reaching the verdict in `build_summary`, a traceback instead of a clean error when lightgbm cannot import during preflight or a marker file holds non-dict JSON, holdout.py and uv.lock only recorded, not pinned) |

Honest reading: the hold-out did not change the verdict. The pre-registered rule already gave "no evidence of an edge" from the development numbers, and the hold-out could only keep or downgrade it. Nothing beat vol-targeted buy-and-hold significantly in development or in the hold-out (IWM was ahead as a point estimate, 0.68 vs 0.42, but the difference is far inside the noise and reverses with a one-day delay).

## Stage 1 — Data
Critics for this stage: Code + test critic (per spec table).

### Round 1
| Critic | Verdict | Reason |
|---|---|---|
| Code + test | FAIL | Stock gap validation only flags gaps > 5 calendar days, so a single missing trading day (e.g. one Wednesday) goes undetected. Everything else verified (20 tests pass, real download works, idempotent, failure keeps cache). |

### Round 2
| Critic | Verdict | Reason |
|---|---|---|
| Code + test | PASS | 160/160 randomly deleted trading days caught; 24 tests pass from a clean checkout; real download OK; idempotent; failures keep cache; BTC partial bar dropped. |

Non-blocking notes from the critic (not failures, deferred): (1) series that merely end early (stale last date) are not flagged by validation; (2) a stock bar fetched while the market is open is stored but heals on the next run; (3) no .python-version pin, so uv picks the newest Python.

**Stage 1 final: PASS from every critic assigned to stage 1 (Code + test) in round 2.**

## Stage 1 follow-ups (requested by Wiktor before Stage 2)
Changes: stale-data warning (NYSE last completed session / yesterday UTC for BTC); stocks and ^VIX drop today's bar until the session has closed; Python pinned to 3.12; cost keys renamed to SPY/QQQ/BTC-USD (validated against asset tickers); pandas deprecation warning from exchange_calendars suppressed by a narrow filter; 12 new tests.

### Round 3 (re-run of the Code + test critic)
| Critic | Verdict | Reason |
|---|---|---|
| Code + test | PASS | 36 tests pass with no warnings (also under -W error); close-time, early-close, DST and holiday probes all correct; clean `uv sync` runs on Python 3.12.16; earlier guarantees re-verified. |

Non-blocking notes (deferred): no unit tests pin early-close or DST cases; ^VIX is treated as final at 16:00 ET though CBOE finalises about 16:15 ET (self-heals on the next run's 5-day refetch).

**Stage 1 final: PASS from every critic assigned to stage 1.**

## Stage 2 — Features
Critics for this stage: Leakage auditor, Code + test critic.

### Round 1
| Critic | Verdict | Reason |
|---|---|---|
| Leakage auditor | PASS | 0 mismatches over 86 truncation dates on real data; appended/perturbed future data changed no past row; 13 of 15 injected lookahead bugs caught by tests (2 survivors were harmless). |
| Code + test | FAIL | (1) Momentum windows were 20/60 but the spec says 20/50/200. (2) Short, empty or missing-VIX input gave silent 1-row results or bare KeyError/IndexError instead of clear errors. |

### Round 2
| Critic | Verdict | Reason |
|---|---|---|
| Leakage auditor | PASS | 151 asset/date truncation runs on real data matched exactly (incl. holidays/weekends); 39,229 changed-row checks after single-bar perturbations had 0 violations; appended wild future rows changed nothing; BTC bar dated d reaches stocks only on d+1; mutation tests caught every real lookahead bug injected. |
| Code + test | PASS | All spec features present with correct windows and independently recomputed on real data; round-1 FAILs fixed (momentum 20/50/200, clear ValueErrors, CLI exit 1); 95 tests pass under -W error from a clean checkout. |

Non-blocking notes (deferred, no code changed after the PASS round): the direct "zero next-day return gives target 0" test was removed during the fix (mutation `> 0` -> `>= 0` survives; a flat-close test would close this); a few cosmetic test leftovers (misleading test name, dead setup code, a vacuous assertion); the official VIX close (~16:15 ET) is slightly after the 16:00 stock close, a negligible timing convention.

**Stage 2 final: PASS from every critic assigned to stage 2 (Leakage auditor, Code + test) in the same round (round 2).**

## Stage 1 follow-ups (spec v2: IWM, ^VIX3M, ^VXN, ^RVX, FOMC dates)
### Round 1
| Critic | Verdict | Reason |
|---|---|---|
| Code + test | PASS | 109 tests pass under -W error from a clean checkout (1 skip: real-data test, expected); real download OK for IWM/^VIX3M/^VXN; ^RVX is not served by Yahoo (bare `RVX` is frozen at 2021) so it is skipped with a warning and IWM falls back to ^VIX; FOMC file checked date by date against federalreserve.gov, no wrong date; deleted single trading days still caught for IWM/SPY/BTC. |

Non-blocking: extras accept missing NYSE sessions as a warning only (a large hole in an index feature series would not fail); a stale ^RVX.parquet would survive an optional skip (none exists).

## Stage 2 (spec v2) — Features, rebuilt to the new feature list
Critics: Leakage auditor, Code + test critic. (The earlier Stage 2 PASS above was against the old feature list and is superseded.)

### Round 1
| Critic | Verdict | Reason |
|---|---|---|
| Leakage auditor | PASS | 70 truncation dates (weekends, holidays, around FOMC) matched exactly for all 4 assets with all 7 input series truncated; wild future rows and single-bar perturbations changed no past row; BTC cross-asset values and stale flag matched an independent loop on 410 dates; pre_fomc uses t+1 correctly; 26 of 30 injected lookahead bugs caught by tests. |
| Code + test | PASS | Every spec feature present per asset; Yang-Zhang, VRP, trend, correlation etc. independently recomputed on real data to ~1e-9; ^RVX fallback driven by the status file; clear errors; 155 tests pass under -W error from a clean checkout; saving idempotent. |

Non-blocking gaps (deferred, no code changed after PASS): (G1) no test for a partial-stale BTC day (e.g. VIX present but stocks missing; real data has such days and the code handles them); (G2) bfill on vix_level survives the mutation test because test VIX starts with the stocks; two weak/tautological assertions in tests; `vrp` added to the warm-up exclusion set (QQQ's ^VXN starts 2001); IWM's VRP uses ^VIX as a weaker proxy because ^RVX is unavailable.

**Stage 2 (spec v2) final: PASS from every critic assigned to stage 2 in the same round (round 1).**

## Stage 3 — Models, validation, metrics, leverage simulator
Critics: Leakage auditor, Statistical honesty critic, Code + test critic.

### Round 1
| Critic | Verdict | Reason |
|---|---|---|
| Leakage auditor | FAIL (test gap only) | No leakage found: truncation/garbage-future runs bit-identical, 1-day gap and calibration split verified on real data, hold-out garbage left every result file byte-identical, 17 of 18 injected bugs caught. One mutation survived: calibrating on the fit slice instead of the later slice passes the suite (the test checks recorded ranges, not the data the calibrator saw). |
| Statistical honesty | FAIL | All numbers independently reproduced. But: (F1) the Deflated Sharpe pools Sharpe variance across all 4 assets (dominated by BTC) without disclosure or per-asset sensitivity, and compares to 0 not to vol-targeted B&H; (F2) the hold-out cutoff is derived from the last data date, so it drifts with each daily update (must be pinned to 2024-11-06); (F3) the threshold compares calibrated p with the whole-window base rate, so BTC models call "up" on only ~1-7% of days and BTC Brier skill looks positive from base-rate drift; undisclosed. |
| Code + test | FAIL | (F1) `quant.backtest` prints raw tracebacks for missing feature files and bad configs (leverage CLI is clean); (F2) Stage 3 decisions, run commands and the results-file schema are not documented in PROGRESS.md. 265 tests pass, outputs byte-reproducible, all deliverables present. |

### Round 2
| Critic | Verdict | Reason |
|---|---|---|
| Leakage auditor | PASS | Round-1 gap closed (calibrating on the fit slice is now killed). A full re-run with garbage written over every price/feature after 2024-11-06 reproduced all 70 result files byte-for-byte; appending new rows after the cutoff also changes nothing. 1-day gap and calibration split verified on real data. One non-blocking test gap: "imputer fit on all data" at the walk_forward call site survives because the synthetic test frames contain no NaN. |
| Statistical honesty | PASS | Every metric recomputed independently from the saved daily returns with zero mismatches (228/456 rows per asset, PSR/DSR to 1e-16, SPA, PT, DM, leverage, Kelly). F1/F2/F3 verified as fixed in the data, not just in wording. Five disclosure nits (slope-flip rates, BTC default_long mostly flat, adaptive aliases, BTC windows pooled in per-asset DSR, DM sign label). |
| Code + test | FAIL | A malformed YAML config (`foo: [`) still prints a raw traceback from both CLIs because `load_config` does not catch `yaml.YAMLError`. Everything else passed (269 tests, byte-identical re-runs, all error paths, docs table accurate). |

### Round 3 (final allowed round)
| Critic | Verdict | Reason |
|---|---|---|
| Leakage auditor | PASS | Previous survivor mutation now killed; 19 of 19 injected leakage bugs killed; real-data truncation and garbage-future runs bit-identical to the saved predictions for SPY, BTC expanding and BTC rolling4y (all 4 models); garbage written over everything after 2024-11-06 reproduced all result files byte-for-byte; a full re-run reproduces all 83 data/results files exactly. |
| Statistical honesty | **FAIL** | Every statistic re-verified with zero mismatches (2,220 trading-metric comparisons, ~1,200 PSR/DSR checks, SPA, PT, DM, leverage, Kelly). Sole defect: PROGRESS.md said the ten adaptive-m alias trials are "all BTC"; the data show BTC 6, QQQ 2, IWM 2, and the m=0.02 fallback partly explains them. A documentation sentence, not a number or a method. |
| Code + test | PASS | 271 tests pass under -W error; the round-2 YAML defect is fixed in both CLIs; every error path (including any lightgbm import failure) prints a clean ERROR and exits 2; results idempotent and reproducible; schema tables match the real files. Nits: non-numeric `costs` values are not validated in config.py; `--models` with no values gives a confusing message. |

**Stage 3 status (as of round 3; see the final confirmation at the end of this stage for the later PASS): NOT a clean final PASS.** The spec allows at most 3 rounds and the Statistical honesty critic failed round 3 on the wording above. The sentence has since been corrected in PROGRESS.md (and a "35 uncorrected DM tests" caveat added), but no fourth gauntlet round has been run, so the corrected text has not been re-checked by the critic. Recorded as: UNRESOLVED-BY-RULE, fix applied, awaiting Wiktor's decision.

Non-blocking items deferred from all rounds: validate `costs` values as non-negative numbers; reject empty `--models`; climatology rows carry a trivially true call_rate_flag.

### Re-check of round 3's Statistical honesty FAIL (requested by Wiktor; beyond the 3-round cap)
| Critic | Verdict | Reason |
|---|---|---|
| Statistical honesty (re-check) | FAIL, then fixed | The corrected alias sentence (BTC 6, QQQ 2, IWM 2; 90 aliases; 270 distinct) was verified correct. But the Diebold-Mariano caveat I had added said "28 tests"; the data hold 35 (SPY 7, QQQ 7, IWM 7, BTC 14 across two windows). I corrected "28" to "35" in PROGRESS.md and this report and re-counted the rows myself (35). All other spot-checked facts agreed with the saved data; no result files changed; 271 tests pass. |

The only wrong statements the critics found in this stage were documentation sentences; no number in any result file was wrong in any round. The corrected DM count was verified by me directly, not by a further critic round.

### Final critic confirmation of the Diebold-Mariano count (requested by Wiktor; beyond the 3-round cap)
| Critic | Verdict | Reason |
|---|---|---|
| Statistical honesty (fresh re-check) | **PASS** | Counted the `diebold_mariano` rows itself in each `significance.parquet`: SPY 7, QQQ 7, IWM 7, BTC-USD 14, total 35. Every mention of the DM test count in PROGRESS.md and GAUNTLET_REPORT.md says 35; no stray "28" remains. No files edited. |

**Stage 3 final verdicts (latest verdict from each critic):**
| Critic | Latest verdict | Where |
|---|---|---|
| Leakage auditor | PASS | Round 3 |
| Statistical honesty | PASS | Final DM-count confirmation above (it failed round 3 and the first re-check, both on documentation sentences only) |
| Code + test | PASS | Round 3 |

Honest caveat: reaching this took more than the spec's 3 rounds (two extra Statistical honesty re-checks, both run at Wiktor's request). Earlier rounds' FAILs stay recorded above. Update to the earlier status line: the corrected text HAS now been re-checked by a critic, so "UNRESOLVED-BY-RULE" is superseded by this PASS.

### Decision on Stage 3 closure (Wiktor, 2026-11-08)
Wiktor chose to **accept Stage 3 as it stands**: the latest verdict from each critic is PASS (Leakage r3, Code + test r3, Statistical honesty on the final Diebold-Mariano re-check). No full-stage Statistical honesty PASS exists; that critic failed round 3 and the first re-check on documentation wording only, and no fourth full round was run. This is an accepted exception to the 3-round rule, not a clean 3-round PASS.

### Stage 3, round 4 (beyond the 3-round cap, requested by me)
| Critic | Verdict | Reason |
|---|---|---|
| Statistical honesty (fresh, full review) | **PASS** | Recomputed independently from saved files: prediction metrics (AUC, Brier skill, log-loss skill, PT), all 35 Diebold-Mariano rows (zero mismatches; sign convention as documented), Hansen SPA p-values, the 270 distinct trials / 90 aliases, global DSR (N=270) and PSR on 24 sampled trials plus 5 excess trials, 480 trading-metric rows (max error 0), 1000 random strategies, and the leverage/Kelly series. No wrong result number, no method error, no edge claimed. Hold-out not evaluated (all results end on or before 2024-11-05). |

Minor findings from this round, recorded as **unresolved** (documentation only; none changes a number; not fixed, per instruction not to loop):
1. GAUNTLET_REPORT.md round-3 Leakage row says a re-run reproduces "all 83 data/results files"; disk holds 70 (17 per asset x 4 + trials.parquet + trials_index.parquet). The round-2 row already says 70.
2. PROGRESS.md "Negative-slope blocks" list is a subset (omits e.g. SPY lgbm 65/179, QQQ lgbm 105/177, IWM rf 55/162, IWM lgbm 73/162, BTC rolling4y logreg 36/67); full counts are in metadata.json. The explanation given for the flips is a plausible, untested guess.
3. Call-rate warnings name BTC only; QQQ long/flat call rates are 0.90-0.95 (near buy-and-hold), so their t values (up to ~3.9) are inherited drift.
4. Multiple-testing caveat covers the 35 DM tests but not the 20 uncorrected PT p-values; QQQ per-asset DSR exceeds 0.95 for all 54 trials while the excess-over-vol-targeted DSR is about 0; three IWM DM p-values below 0.05 are all in the "model worse" direction.

**Stage 3 final state:** every critic's latest verdict is PASS, and the Statistical honesty critic has now also passed a full-stage review (round 4, beyond the cap, with the four documentation items above left unresolved).

## Stage 4 — Paper trading

### Round 1
| Critic | Verdict | Reason |
|---|---|---|
| Leakage auditor | **PASS** | No leak found. Garbage after the as-of date, a one-shot catch-up vs 44 daily runs, and truncated-vs-full feature builds all gave identical ledgers/features; 17 of 19 injected leakage mutants killed. Hold-out untouched (all results end on or before 2024-11-05; first live decision 2026-11-06 or later). |
| Statistical honesty | **PASS** | Ledger positions, costs, equity, buy-and-hold and vol-targeted baselines recomputed by hand and matched; config IDs exist in the Stage 3 trial log; backtest reference Brier/skill/hit rate match `prediction_metrics.parquet` to 1e-12; paper trading unleveraged and uses the pinned strategy; no live evidence claimed. |
| Code + test | **PASS** | 345 tests pass under -W error; `quant.run_daily` exits 0 twice with `data/paper/` checksums unchanged; `--dry-run` writes nothing; 9 failure paths give clean `ERROR:` and no traceback; scheduling instructions checked; no scheduled job or broker code. 38 of 47 mutants killed. |

**Stage 4 round 1: PASS from every critic in the same round.** No FAILs, so no rebuild round was needed.

Non-blocking notes from the critics, NOT fixed (the code was not changed after review):
1. Test gap: no test pins `r_last` (the realised D-1 to D return) into the adaptive-margin history (`paper.py` ~273, 413). Mutants feeding tomorrow's return or a flipped sign survive all tests; on real data such a bug would change `margin_m` in 21 of 618 catch-up rows and `position` in 2. The shipped code is correct (the Leakage auditor traced it); only the test is missing.
2. Other surviving mutants: ledger `base_rate` is not asserted equal to the training base rate; `--dry-run` not asserted to skip the download; the NaN-features skip and the duplicate-row check on ledger load are untested; ties (`>=` vs `>`) untested.
3. `default_long` mode cannot be used for BTC (that exact config is not in the trial log); it fails cleanly. `state.json` `start_date` (2026-11-08) differs from the first ledger decision dates (2026-11-06/07). Live-metric Sharpe SE caveat not needed yet because no live Sharpe is computed.
4. The live model trains on hold-out calendar rows as training data for future predictions only; no hold-out metric is computed (disclosed in PROGRESS.md).

### Stage 4 follow-up (requested by Wiktor, after the round-1 PASS)
Non-blocking note 1 above is now closed: added `test_adaptive_margin_history_ends_with_the_realised_return_into_today` in tests/test_paper.py (test-only change; no production code changed). It spies on `_history_arrays` across a month-change catch-up and asserts that the last adaptive-margin history element is the realised (D-1 -> D) return, not tomorrow's and not sign-flipped. I confirmed it fails against both mutants (tomorrow's return; flipped sign) and passes on the real code. `uv run pytest -W error`: 346 passed. This follow-up was verified by me, not by a new critic round. Other round-1 notes (2-4) remain open.

## Stage 5 — Dashboard

### Round 1
| Critic | Verdict | Reason |
|---|---|---|
| Leakage auditor | **PASS** (with notes) | No leakage in the meter percentile, the "Why?" numbers, the trust hit rates, models-agree or the diagnostics: 30 garbage future price bars plus fake future ledger rows left every meter/Why?/trust output bit-identical; 14 of 19 injected mutants killed. Notes: three test-strength gaps (decision day inside its own "normal range"; a forward-looking rolling AUC; mid-rank percentile spec fidelity), none a live defect. |
| Statistical honesty | **FAIL** | Every displayed number recomputed and matched; all four badges correctly grey. Defects are display/wording: Brier skill shown only vs the drift-prone training base rate (BTC +0.0059 reads as skill, eval-base skill is -0.0028); Trials "luck bar 1.40" stated as fact without the correlated-trials caveat; Kelly text contradicts the leverage table without explanation; Trials DSR table (vs zero) can read as evidence beside grey badges; meaningless break-even values and duplicate baseline rows; disclosure gaps (Strong-buy rule, badge conditions, random-strategy percentile not multiple-testing corrected, "proves nothing" note placement, hold-out-power and BTC-cost wording). |
| Dashboard | **FAIL** | No correctness or honesty defects; all numbers matched. Defects: "23th/91th" ordinal bug; two dead source links (IMF govdelivery 404, Wikipedia Purged_cross-validation 404) shown as in RESEARCH_REPORT.md, quantpedia link timed out; Paper-trading charts unreadable with one data point; Backtest table clipped at 1366px and meaningless baseline break-even values; clipped Trials axis title; Reality-check page jargon without glossary and no plain reading of the project's own AUC table; Today page terms unexplained and high needle beside below-base-rate probability confusing; badge "formula" text not beginner readable; "lower than 100% of days" overstated and BTC weekend volume caveat missing. |
| Code + test | **FAIL** | 505 tests pass and no data is written by the dashboard. Defects: corrupt ledger/state, missing price file, corrupt predictions or missing results give raw tracebacks (run_page catches too little; Trials crashes on an empty index); a clean checkout fails 7 tests + 13 errors where lightgbm cannot load libomp (the fix is not documented) and the dashboard then wrongly says the model is missing; 23 of 67 mutants survive (e.g. Sharpe-SE multiplier in table/chart, wiped-out flag inverted, leverage chart always 1x, 60-day rolling window instead of 250, baseline columns swapped, refresh not clearing caches); dead imports/params; amber/green/wiped-out dashboard paths never exercised. |

Round 1 result: **not all critics passed.** All FAILs go back to a builder; the full gauntlet re-runs in round 2.

### Round 2
| Critic | Verdict | Reason |
|---|---|---|
| Leakage auditor | **PASS** (with notes) | No leakage. One test-strength survivor: nothing pins `DD.rolling_brier_skill` as backward-looking. Latent limitation: `today_report` / `load_live_model` are not guarded when `asof` is not the latest ledger date. |
| Statistical honesty | **FAIL** | All numbers verified correct. Two medium wording/disclosure defects: (1) the Backtest "random-strategy percentile" is computed on gross returns and, for vol-targeted series, mostly reflects vol targeting rather than model timing (vol-targeted buy-and-hold scores 98.2 SPY / 98.1 QQQ / 72.1 IWM / 66.2 BTC vs model 85.4 / 93.4); (2) the Trials page per-asset DSR column (e.g. SPY 0.993) is not reconciled with the stricter pooled DSR (0.017, luck bar 1.40). Minor: the Reality-check "needs Sharpe above ~0.6 / 0.87" is the report's figure, not this project's sample sizes (SPY ~0.78, BTC ~1.8). |
| Dashboard | **PASS** (8 cosmetic notes) | Metric labels truncated at 1366px, clipped chart axis titles/headers, crowded Trials histogram annotation, stale "could not be verified" Quantpedia annotation, meter-needle vs zone-label confusion, stray wording ("round-1 critic-driven change", "evaluation- window"), "Why?" history phrase should say 2000 to 2024. |
| Code + test | **FAIL** | 615 tests pass under -W error; 7 pages load; 26 corruption cases and the refresh handler give clean messages with no traceback; libomp fix verified; no secrets, scheduled jobs or broker code. But 45 of 208 mutants survive, including two hold-out guards (`holdout_load`, `trial_assert_dev`), the cache key (`data_sig_ledger`), `equity_window`, per-card error handling (`card_exc_narrow`, `fail_lookup`), badge boundaries and meter-zone thresholds. Defects (non-crashing): `explain_failure` checks OSError before the libomp test; missing `state.json` shows a misleading "no paper-trading decision yet ('assets')"; empty predictions shows a cryptic bin-edges message; `refresh_data` returns ok=True when assets were skipped; dead code (`explain.load_live_model`, `Why.as_dict`, `MeterReading.as_dict`, `dash_data.hit_rate_deciles`, `AssetResults.available`, `ui.ACCENT`, `ui.LEV_COLORS`). |

Round 2 result: **not all critics passed.** All FAILs, plus the Dashboard and Leakage notes, go to a builder in one fix list; round 3 (the last allowed) follows.

### Round 3 (final round allowed)
Before round 3 a builder applied the round-2 fix list (91 new tests in tests/test_round2.py). `uv run pytest -W error -q`: 707 passed, 0 failed (verified by me and by the Code + test critic).

| Critic | Verdict | Reason |
|---|---|---|
| Leakage auditor | **PASS** | No look-ahead. 5 garbage future bars per asset plus fake later ledger rows changed no meter, "Why?" or models-agree output; `live_model_guard` refuses non-latest days; appended 2024-12 predictions raise `HoldoutError`; no hold-out metric is computed. 15 of 17 injected mutants killed; the 2 survivors are redundant defensive cuts (equivalent). |
| Statistical honesty | **PASS** | Every displayed number recomputed with 0 mismatches (290 trading-metric rows, luck bar 1.40, per-asset vs pooled DSR, vt-baseline percentiles 98.2/98.1/72.1/66.2, needed-Sharpe table, badges all grey because BEATS fails for every asset). All round-1/2 disclosure defects are now disclosed. |
| Dashboard | **PASS** | All 7 pages show real data at 1366px and 420px, no console errors, prior clipping/wording notes fixed. Cosmetic notes only. |
| Code + test | **FAIL** (narrow) | 707 tests pass, 7 pages load, data/ checksum unchanged, 70 of 88 mutants killed, all round-2 important-logic survivors now killed. Remaining issues below. |

Round 3 result: **not every critic passed in the same round. The 3-round cap is reached; the remaining issues are recorded as unresolved (not fixed, per the rules).**

Unresolved after round 3:
1. (Code + test, the FAIL reason) A broken or missing `config.yaml` shows a Streamlit exception box with a stack trace on all 7 pages: the error is raised in `sidebar()` at `DD.cutoff()` (app/dashboard.py ~945), outside `run_page`. The message is clear; only the traceback box is wrong. Small fix: wrap `sidebar()` in the same failure handler.
2. (Code + test) Untested guards in round-3 code: `vt_baseline_percentile` (`len(nr)<3`, non-finite guard, empty-input `None`, NaN-percentile `None`) and `refresh_data` with an unreadable config returning `ok=False`.
3. (Code + test) "Why?" text: the "nudged it down" clause and the percentile phrase are not pinned by tests.
4. (Code + test) Minor surviving mutants: meter edge cases (`bounds_strong_max`, `bounds_unusable`, `needle_nan`, `clamped_flag`), `equity_dropna_off`, message-level cosmetics (`fail_value_level`, `refresh_nnew`, `refresh_keyerr`, `nodecision_asof`); two are equivalent (`badge_nan_ok`, `ordinal_teen`).
5. (Leakage, notes) Two redundant `ds.index <= d` cuts in `explain_today` / `load_dataset_asof` are untested; `live_model_guard` gives no protection if called directly with `asof=None` (the dashboard always passes it). Stage 6 must state that the live ledger is not the hold-out evaluation.
6. (Statistical honesty, notes) The "mostly reflects vol targeting" caption does not hold for BTC rolling-4y (vt baseline 38.4 vs model 59.5); the "Why two Brier skills?" caption cites only BTC (QQQ drifts too); Kelly caption is generic; log-return percentages shown as percent; hindsight nature of the evaluation-base Brier skill is not explained.
7. (Dashboard, cosmetic) Backtest metric label "Vol-targeted buy and hold percentile" truncated at 1366px; drawdown y-axis title overlaps chart title; wide tables scroll horizontally without a cue (Trials "Excess (DSR)" column, Paper tables, Backtest models table); truncated Backtest dropdown labels; blank "Trades so far" column in the Paper P&L table; crowded log ticks on the Leverage chart; struck-through unused zone chips; two source links still 404 (labelled "link no longer works", as in RESEARCH_REPORT.md).

## Stage 5 post-cap fixes (approved by me)
Requested by Wiktor after the 3-round cap. A builder (1) made a broken or missing `config.yaml` show a clean message instead of a traceback (`safe_sidebar`, `is_config_error`, cutoff-caption fallback), with tests for the untested round-3 guards (`vt_baseline_percentile`, `refresh_data` with an unreadable config, the "nudged it down" clause and percentile phrase); (2) redrew the Today meter in gap space (probability minus the usual up-day rate) with zone edges taken from each asset's real strategy thresholds (`zone_at`, `zone_edges`, `meter_explanation`), so the needle is always inside the zone of the actual paper position; the percentile is now small print; no Strong-buy zone is invented for long_flat/default_long (zero-width zones are crossed out in the legend); cards show only the meter, the "p vs typical" line, the position, the badge, the grey-badge note and the disclaimer, with the long explanations moved to "Badge and terms". `paper.py` and every ledger value are unchanged. `uv run pytest -W error -q`: 770 passed, 0 failed (verified by me and by the Code + test critic).

| Critic | Verdict | Reason |
|---|---|---|
| Code + test | **PASS** (one minor test gap) | Config cases (`foo: [`, empty, top-level list, missing key, deleted theory.yaml, deleted results/ledger) give clean messages on all 7 pages; own independent oracle, 36,000 random and boundary cases over long_flat/default_long/sized x vol-targeting on/off: 0 zone violations; on real data needle zone equals the ledger position's zone for all four assets; data/ and ledger byte-identical; 32 of 34 mutants killed (1 equivalent). Survivor: `gauge_figure` draws `meter.gap` instead of `meter.needle` undetected, which matters only when the sized-mode no-trade band clamps the needle (not the current mode). |
| Dashboard | **PASS** (cosmetic notes) | Read from the live Plotly gauges: SPY needle in Sell (position 0%), QQQ/IWM/BTC-USD in Hold/buy (84%/100%/100%); the QQQ mismatch is fixed; cards show only the intended items; Why? and Badge-and-terms intact; no console errors; no clipping at 1366px or 420px; other six pages unchanged. |

Not fixed (recorded as unresolved):
1. Test gap: drawn gauge needle not pinned to `meter.needle` when clamped (sized mode only).
2. Cosmetic: the "0 / typical" tick is unlabelled on the SPY and QQQ gauges and the gap value is not printed on the dial; the active zone chip is not visually distinct; the dial subtitle repeats the label; the How-it-works badge block shows the constant name `FADED_NOTE` (quant/explain.py ~388, pre-existing); the sidebar is open by default at 420px.
3. `is_config_error` also matches any FileNotFoundError/ValueError whose text mentions config.yaml (wording only).

## Stage 6 — Wrap-up
Critics for this stage: Leakage auditor, Statistical honesty critic, Code + test critic. The real hold-out can be evaluated only once, so the gauntlet is run in two parts: the one-shot harness (`quant/holdout.py`) is reviewed BEFORE the real run on a pseudo hold-out made from the last 24 months of development data ("pre-run" rounds), then the real run happens once, then the results, report and README are reviewed ("post-run" round). Critics never run `--run` and never read hold-out rows.

### Round 1 (pre-run review of the harness)
| Critic | Verdict | Reason |
|---|---|---|
| Leakage auditor | **PASS** (4 hardening notes) | No leakage by code reading and mutants: training slice ends 2 rows before each block, calibration/imputers/base rate/sigma_target/adaptive-m history use only earlier data, the cutoff label never enters training, window definition double-counts and skips nothing; equivalence with saved Stage 3 results holds. Notes: no assertion on the scored dates, marker written before preflight (a lightgbm import problem would burn the shot), `--config` could sidestep the guard, features cannot be re-verified without reading hold-out rows. |
| Statistical honesty | **FAIL** (narrow) | All statistics correct (Sharpe CI, Wilson, PT with base rates, Brier skill, Jobson-Korkie/Memmel SE, power note, pinned config ID, costs). Defects: (1) `--config` with `--run` could bypass the one-shot guard (different data_dir/cutoff/strategy); (2) no pre-registered interpretation rule or verdict mapping before seeing hold-out numbers; (3) repeat runs only blocked by hand-editable file state, and the report did not flag a repeat attempt. |
| Code + test | **PASS** (test gaps) | Full-run guard scenarios in copies (KeyboardInterrupt, manifest tamper of quant files/config/uv.lock/results/features/ledger, markers in every state, short data does not burn the shot) all behave; 46 of 53 mutants killed; README commands work. Gaps: BTC weekend-hole coverage, Wilson hit-count rounding, climatology eval-base log-loss skill, non-expanding-window refusal. |

Round 1 result: not every critic passed. All findings go to a builder (done: `--run` refuses `--config`, pinned-constant assertions, config.yaml SHA-256 pre-registered in code and PROGRESS.md, interpretation rule `hold_out_flags()` / `final_verdict()` pre-registered, repeat-attempt flag in the report, scored-window and contiguity assertions, preflight before the marker, new tests; 829 tests pass), then the harness is re-reviewed in round 2.

### Round 2 (pre-run review of the harness after the round-1 fixes)
| Critic | Verdict | Reason |
|---|---|---|
| Leakage auditor | **PASS** (notes) | Round-1 fixes hold under mutation (dropped/shifted months, history gaps, unblanked last label all raise; gap 0/+3, full-sample sigma/base rate caught by the real-data replay). Full `run_final` rehearsed on a pseudo window in a copy: completes with freeze verified. `--config` bypass closed; verdict code feeds nothing back. Notes: history contiguity only checks the last date (a mid-history gap is not detected), only config.yaml SHA pre-registered (not the code), adaptive-m peek caught by the BTC replay only. |
| Statistical honesty | **FAIL** (narrow) | Round-1 defects 1 and 3 fixed; recomputations match. (D1) the pre-registered "weak evidence" band is gameable: passing the strategy's own t > 3 (which buy-and-hold also passes, SPY 3.25, QQQ 3.64) alone gives "weak evidence", so SPY/QQQ would read "weak evidence" although every baseline-relative measure (excess t negative, SPA p 0.7-1.0, DSR far below 0.95) points away from an edge; (D2) the excess DSR is asset-scoped (54-106 trials) while the strategy DSR is global (270) and SPA is family-level: not disclosed in PROGRESS.md/report reasons; (D3) no pre-registered rule for turning four per-asset verdicts into the one-line headline. |
| Code + test | **FAIL** (narrow) | Harness code correct; all guard, tamper, KeyboardInterrupt and preflight scenarios pass in copies. But 16 of 56 mutants survive, mostly the pre-registered thresholds: `t_x >= 3`, SPA 0.05/0.10 and `<=`, PT `<=`, Brier `>= 0`, `any_failure` combinations, the failure-signal reason line; preflight does not test contiguity/shape (a failure would burn the shot); `leverage_strategy` extra keys and `project_root` pins untested. |

Round 2 result: not every critic passed. Fix list to a builder, then round 3 (the last allowed round).

### Round 3 (final pre-run review of the harness, after the round-2 fixes; 883 tests pass)
| Critic | Verdict | Reason |
|---|---|---|
| Leakage auditor | **PASS** (notes) | 15 mutants (adaptive-m peeks incl. +1, gap 0/+3, full-sample sigma/base rate, dropped month, mid-history gap, cutoff-day history, unblanked last label) all caught; `run_final` rehearsed end to end on a pseudo window; real-run path scores exactly [2024-11-06, 2026-11-05]; no new selection path from the verdict code. Notes: holdout.py not pre-registered (its hash is recorded in the marker; also written in PROGRESS.md before the run), preflight does not catch an extra non-session row or a missing cutoff-day row (the lead checked the real data by dates only: both clean, 502 sessions per ETF and 731 days for BTC in [2024-11-06, 2026-11-06]), +1 peek caught by the synthetic test only. |
| Statistical honesty | **PASS** | D1-D3 fixed: "weak evidence" now needs a baseline test; scopes disclosed (excess DSR asset-scoped 54-106 trials vs global 270; SPA family-level; t>3 also passed by buy-and-hold); headline = weakest asset. Applied to the development numbers the pre-registered rule gives "no evidence of an edge" for all four assets, so the hold-out can only keep or confirm it. All dry-run metrics recomputed (PT differs ~1e-3, a variance-formula variant). |
| Code + test | **PASS** (narrow) | 114 mutants of holdout.py: 103 killed (all thresholds, flags, preflight, pins, marker, manifest); full-run scenarios (success, KeyboardInterrupt, 11 tamper types, markers in every state, preflight failures not burning the shot) pass; README accurate. Open (non-blocking): no end-to-end test proves hold-out failure flags reach the real verdict in `build_summary` (lead verified by reading: `holdout.py` ~708-712 passes the flags to `final_verdict`); a non-dict JSON marker raises an AttributeError (still refuses); a lightgbm import failure in preflight shows a traceback (shot not burnt); uv.lock/pyproject only compared before/after, holdout.py not pinned. |

**Pre-run result: PASS from every critic in the same round (round 3).** holdout.py SHA-256 before the run: `ee456502ffdeab3ae71f68d0556aaa0c3050d611d7d11302511139bebefb40e7`.
The real hold-out is now run once.


### Hold-out run (after the round-3 PASS)
`quant.holdout --run --confirm-single-final-evaluation` ran once: marker status `done`, freeze manifest re-verified (no code, config, result or data file changed). Outputs: `data/results/holdout/`. Per-asset and headline verdict: no evidence of an edge. The round-3 PASS from every critic was recorded before the run. 
**Post-run documentation check (Statistical honesty critic, not a new gauntlet round; results frozen): PASS.** Every number in the README results section and in this Summary was recomputed from the saved hold-out files and matched (Sharpe, Lo CIs, Wilson, PT, Brier skills, Jobson-Korkie/Memmel SEs, the QQQ paired Brier CI [0.0004, 0.0034], DSRs, SPA p-values, t-statistics). The suspected BTC Brier "swap" is a deliberate column reorder in the markdown (eval base first for BTC), not an error. The verdict rule applied by hand gives the same per-asset and headline verdicts. Freeze verified: marker `done`, all hashes match PROGRESS.md, no `HOLDOUT_FAILED_*` files. One overclaim in the Summary ("nothing beat vol-targeted buy-and-hold") was corrected to "significantly".

Test-only follow-up after the run (2026-11-09, lead): `test_guard_blocks_run_final_with_the_real_config` asserted that `data/results/holdout/` did not exist, true only before the real run; it now asserts the folder is byte-identical before and after the test. No production code, config or result file changed (`--verify-preregistration` still matches). Final suite: `uv run pytest -W error -q` 883 passed, 0 failed.


## Stage 7 data (7a universe, 7c hourly crypto, 7d FRED) — 2026-11-09
Scope of this goal: DATA only (no strategies, backtests, trials or hold-out evaluation). New code lives in `quant/stage7/` with `config_stage7.yaml` so the Stage 6 pre-registered hashes of `config.yaml` and `quant/*.py` stay valid. **Pre-registration note (as stated by Wiktor, a claim, not verifiable from this repo):** the Stage 7 strategies were written on 2026-11-07 in a separate planning chat; the spec reached the repo on 2026-11-09, after Stage 6 ran. For the portfolio the Stage 6 hold-out is therefore a **partially seen hold-out**.

### Round 1 (final round; every critic PASS in the same round)
Builders: three in parallel (7a, 7c, 7d), then the lead ran the full suite (954 passed) before the gauntlet. Critics were fresh agents that read the code and ran commands.
- **Leakage auditor: PASS.** Appended/mutated future hourly bars, D+1 00:00 bars and future FRED observations at several cut dates: every earlier row of rv_hourly, ret_6h, ret_24h, vol_concentration, har_rv_forecast, asof_macro and regime_context was bit-identical. HAR-RV recomputed independently (full float precision). FRED availability = next NYSE session verified over weekends and holidays (Memorial Day, July 4th, Christmas, New Year) and for the 7-day crypto calendar. Dev-only defaults and the cutoff boundary hold; no default path touches post-cutoff rows.
- **Statistical honesty critic: PASS.** Hold-out untouched (`--verify-preregistration` passes; Stage 6 files and results unchanged). Documentation matches the code (48h gap limit, short BAMLH0A0HYM2 and T10YIE histories, warm-ups, USDT note, survivorship note, pre-registration wording). Bad inputs are rejected, not repaired. Realised vol, 6h/24h returns, Herfindahl, HAR-RV and the risk-free/borrowing arithmetic recomputed independently and matched.
- **Code + test critic: PASS.** `uv lock --check` and `uv sync --frozen` consistent; 954 passed (883 old + 71 new), 0 failed. Mutation testing: 43 mutations, 39 killed. Atomic writes, idempotent re-runs, one-line ERROR + exit 1 on the three CLIs, no secrets or absolute paths.
- **Dashboard critic: PASS.** No dashboard file changed; all 7 pages load with 0 exceptions; disclaimer visible; none of the new files are picked up; `run_daily --dry-run` exit 0.

### Non-blocking observations (recorded, deliberately NOT fixed so the PASSed code stays the code that was audited)
- Mutation survivors (4, non-critical): the "high/low/middling" credit-spread wording boundary; the 6-hour overlap re-fetch and `keep="last"` in the hourly incremental update are not asserted on revised bars; `config.cost_for(..., stress=True)` is untested and has no caller yet (the portfolio goal will use it).
- `regime_context` loads hold-out FRED data if asked for a post-cutoff date (by design, for live use); `update_universe` returns full frames including hold-out rows (research code should go through `load_universe`); `daily_features` inherits the dev-only default from `load_hourly`.
- `hourly.update_asset`: a transient Binance outage on existing data raises the "refusing to mix exchanges" message (misleading wording, original cause is included); `hourly` `status.json` is written non-atomically; `universe.main()` does not catch config-file errors (traceback instead of one ERROR line).
- BAMLH0A0HYM2 history starts 2023-11-10 (about one year of development data); the "since 2023" wording of its regime note rests on roughly one year.
- Refresh behaviour: newly downloaded values replace stored ones for overlapping dates (FRED and hourly), and `update_universe` re-fetched the last days of SPY/QQQ/IWM/BTC-USD, rewriting those four price files (not covered by the freeze manifest; Stage 3-6 results were not recomputed).
Status of every issue above: none blocks a critic's PASS; they stay open as minor "unresolved" items for the later Stage 7 goals.

### Start dates (development rows; `eligible_from` = first date + 252 rows for ETFs, 365 for crypto)
| Ticker | Group | First date | Eligible from | Rows (dev) |
|---|---|---|---|---|
| SPY | us_equity | 1999-01-04 | 2000-01-03 | 6504 |
| QQQ | us_equity | 1999-03-10 | 2000-03-08 | 6459 |
| IWM | us_equity | 2000-05-26 | 2001-05-29 | 6151 |
| XLK, XLF, XLE, XLV, XLI, XLY, XLP, XLU, XLB | us_sector | 1998-12-22 | 1999-12-22 | 6511 each |
| XLRE | us_sector | 2015-10-08 | 2016-10-07 | 2286 |
| XLC | us_sector | 2018-06-19 | 2019-06-20 | 1608 |
| EFA | international | 2001-08-27 | 2002-09-03 | 5836 |
| EEM | international | 2003-04-14 | 2004-04-14 | 5430 |
| EWJ | international | 1999-01-04 | 2000-01-03 | 6504 |
| VGK | international | 2005-03-10 | 2006-03-10 | 4950 |
| SHY, IEF, TLT, LQD | bonds | 2002-07-30 | 2003-07-30 | 5608 each |
| TIP | bonds | 2003-12-05 | 2004-12-07 | 5266 |
| HYG | bonds | 2007-04-11 | 2008-04-10 | 4426 |
| GLD | real_assets | 2004-11-18 | 2005-11-17 | 5026 |
| DBC | real_assets | 2006-02-06 | 2007-02-07 | 4721 |
| VNQ | real_assets | 2004-09-29 | 2005-09-28 | 5062 |
| UUP | dollar | 2007-03-01 | 2008-02-29 | 4454 |
| BTC-USD | crypto | 2014-09-17 | 2015-09-17 | 3704 |
| ETH-USD | crypto | 2017-11-09 | 2018-11-09 | 2555 |
| Hourly BTC/USDT, ETH/USDT (Binance) | crypto | 2017-08-17 04:00 UTC | n/a | 63,204 bars each (to 2024-11-06 23:00) |
| FRED DTB3 / DGS10 / T10Y2Y / DFF | macro | 1990-01-02 (DFF 1990-01-01) | next NYSE session after the observation | 8720 / 8720 / 8720 / 12729 |
| FRED T10YIE | macro | 2003-01-02 | same | 5468 |
| FRED BAMLH0A0HYM2 | macro | 2023-11-10 | same | 260 |

Final check: `uv run pytest -W error -q` = **954 passed, 0 failed** (re-run after the last code change; only documentation was edited afterwards).
