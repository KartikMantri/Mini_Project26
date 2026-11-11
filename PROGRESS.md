# Progress Log — Decoding the Game

**Purpose of this file:** a running, append-only log of every real decision made on this
project — not just what changed, but *why*, what it affects downstream, and what's left.
Read this top-to-bottom before doing anything if you're picking this project up cold
(a new Claude session, or Kartik coming back after a break). Newest entries are at the top.

Phase-specific methodology/results live in `docxxs/PhaseN_*.md` — this file is the
connective narrative across phases, including decisions still being debated.

---

## Current snapshot (as of 2026-11-12)

| Phase | Status |
|---|---|
| 0 — Setup & Foundation | ✅ Done |
| 1 — Data Collection | ✅ Scaled: 10,493 calls across 200 tickers (2005–2025); full 12,789-call set with corrected dates in `data/calls_all.csv` |
| 2 — Signal Extraction | ✅ Rebuilt with FinBERT: `data/features_finbert.csv` (12,716 calls × 18 features). Phase 3 still reads the old VADER `data/features_expanded.csv` until it is switched over |
| 3 — Type Classifier | ✅ Re-run on FinBERT (`train_classifier_finbert.py`): reaction AUC 0.611 on the 2018–2025 holdout (VADER 0.588) |
| 4 — Market-Reaction Test | ✅ Done on the FinBERT score (`market_reaction_finbert.py`): holdout drift +1.62% top vs bottom (p 0.002), +1.31% with controls (p 0.013) |
| 5 — Trading Backtest | ✅ Done (`backtest_finbert.py`): long-short +6.9%/year after costs, Sharpe 0.74, p 0.032, on the 2018–2025 holdout |
| 6 — Dashboard & Report | 🔄 Script ready (`dashboard.py`), still reads the old VADER outputs |

**Immediate next step:** Phase 6, switch the dashboard to the FinBERT outputs and write the report. The VADER pipeline is kept only as a comparison baseline.

**2026-11-12 — PARA price fix, Phases 3–4 re-run, Phase 5 done (Kartik):**
- **PARA prices are not Paramount's.** The series in `data/prices.csv` starts 2021-02-12 at
  ~83,000, falls to ~6, has 22% zero-change days and a 0.05 correlation with the S&P 500 (no
  sign of the March 2021 Archegos crash). A scan of all 200 tickers found no other bad series
  (AIG, UAL, VRTX, LNG big moves are real). `event_returns.BAD_PRICE_TICKERS` now returns no
  prices for PARA; Phase 3 blanks the returns of its 77 calls (their text still feeds the
  change-since-last-call features); Phase 4 drops them.
- **Phase 3 change:** pre_ar, pre_vol and imm_ar are capped at the 1st/99th percentile of each
  fit's training rows (`TrainClipper`), so no future data sets the caps. Transcripts are now read
  as plain Python strings; the Arrow-string `.tolist()` ran out of memory.
- **Phase 3 after the fix:** same pick (LR on tone + market, validation AUC 0.640). Holdout
  reaction AUC 0.611 (was 0.613), drift top − bottom +2.25% (was +1.93%).
- **Phase 4 after the fix (holdout):** immediate +3.94% (t 13.2); drift **+1.62%** (t 3.4,
  p 0.002); with controls **+1.31%** (p 0.013); no-language model −0.36% (p 0.53). Validation
  2012–17 drift **+0.85% (p 0.031)**, was +0.70% (p 0.072). VADER on the same calls: +1.40%
  (p 0.059), +1.29% with controls (p 0.099).
- **Phase 5 (`backtest_finbert.py`), method:** at each call, LONG if its score is in the top 20%
  of the past 365 days' scores, SHORT if in the bottom 20% (no look-ahead, unlike Phase 4's
  within-quarter ranks). Enter at day +2's close, exit at day +60's. Daily equal-weight long leg
  minus short leg. 10 bps per buy or sell, 0.5%/year short borrow. Nothing was tuned on the
  backtest. The old `backtest.py` is kept but not used: it compounded overlapping trades.
- **Phase 5, holdout 2018 – May 2025:** **+6.9%/year after costs, Sharpe 0.74, max drawdown
  −14.1%, t 2.14 (p 0.032)**; before costs +9.4%, Sharpe 1.01. Beta to the S&P 500 0.10.
  Break-even cost 46 bps per trade. 2,315 trades, 54% profitable after costs. No-language score
  −4.7%/year; Velu's VADER score +5.9%/year, Sharpe 0.56 (p 0.11). S&P 500 over the same dates
  +12.9%/year, Sharpe 0.64, drawdown −33.9%.
- **Phase 5 caveats:** 2020 alone gave +24% (holdout ≈ +4.4%/year without it). Validation
  2012–17 is weaker: +3.1%/year, Sharpe 0.50, p 0.24 after costs (p 0.035 before), and there
  the short side lost money. Survivorship bias (today's 200 tickers).

**2026-11-11 — Phase 4 on the FinBERT score (Kartik):**
- **Setup:** `market_reaction_finbert.py` reads `data/predictions_finbert.csv`. It uses the
  standard event-study tests with the same definitions as Velu's VADER Phase 4 (branch
  `vadar-phase4`, not merged), so the results compare directly: within-quarter quintiles,
  Newey-West t across quarters, drift regressions with quarter fixed effects and errors clustered
  by quarter, controls for the immediate reaction and pre-call return/volatility. Velu's scores
  were run through the same script as a check: it reproduces his holdout drift (+1.26% vs. his
  reported +1.20%, both t 1.8).
- **Note:** Velu's branch also changes Phase 3 (adds a `text_delta` input), so his Phase 4 baseline
  is that model, not the 0.588-AUC one.
- **Holdout 2018–2025:** immediate top − bottom **+3.93%** (t 13.5). Drift top − bottom
  **+1.55%** (t 3.4, p 0.002); **+1.27%** with controls (p 0.018). 28% of the total move comes
  after the reaction window. The language-free model's drift is +0.13% (p 0.78).
- **vs VADER, same 4,822 holdout calls:** drift +1.75% (p 0.005) vs +1.26% (p 0.075); with
  controls +1.41% (p 0.021) vs +1.12% (p 0.147). So FinBERT makes the under-reaction significant
  on the holdout, where VADER's wasn't. Over 2012–2025 on the same calls the two are close (+1.42%
  vs +1.37%). On the validation years alone, FinBERT's drift is weaker (+0.70%, p 0.07).
- **Pandemic:** unlike Velu's run, FinBERT's drift stays positive in 2020 and 2021 and stays
  significant when those years are dropped. The result doesn't depend on that period.
- **Verdict:** the market prices most of the language signal immediately, and a smaller,
  statistically significant part (~1.3–1.6% top vs bottom) drifts in over the next 60 trading
  days: evidence of under-reaction. Results: `docxxs/Phase4_finbert_results.md`.

**2026-11-11 — Phase 3 re-run on FinBERT features (Kartik):**
- **Setup:** `train_classifier_finbert.py`, which uses Velu's walk-forward method (LR + XGBoost, inputs chosen
  greedily on 2012–2017 only, holdout 2018–2025) as the main result, and Kartik's fixed split (train
  before 2021-10-06 = 61-trading-day gap, test 2022–2025 once) as a check. 12,712 calls (4
  unparseable dropped). Velu's VADER `train_classifier.py` kept unchanged as the baseline.
- **Chosen on validation:** LR on FinBERT tone (8) + market (2). The greedy search added tone first
  (+0.136 AUC), then market; TF-IDF text added nothing once tone was in.
- **Holdout 2018–2025:** reaction AUC **0.613** (Velu VADER 0.588), GOOD/BAD AUC 0.535 (0.528),
  top-minus-bottom 20% drift **+1.93%** (+1.30%), rank-corr t 2.9. On the same 4,833 calls as Velu:
  0.611 vs 0.588 reaction AUC, +2.28% vs +1.30% drift spread.
- **Fixed-split check (2022–2025):** reaction AUC 0.613, drift spread +2.29% (no-language model
  +0.94%), so the result holds under both setups.
- **Results:** `docxxs/Phase3_finbert_results.md`; predictions `data/predictions_finbert.csv`.
  Phase 4 (`market_reaction.py`) still reads Velu's `data/predictions.csv` and needs switching.

**2026-11-11 — Phase 2 FinBERT full run done, checks complete (Kartik):**
- **Run:** all 12,716 non-stub calls, 198 tickers → `data/features_finbert.csv` (22 columns).
  About 1.1 s per call in practice (≈ 4 h, not the 2.1 h the trial suggested); one laptop standby
  pause, survived thanks to per-ticker checkpoints in `data/finbert_cache/`.
- **Data problems found:** 4 transcripts (AMD 2006, MSFT 2006, PEP 2009, PLD 2010) have no line
  breaks between speakers, so every FinBERT feature is blank and `log_word_count` = 0. They should be
  dropped in Phase 3. About 330 calls have no analyst/executive roles (Q&A features blank) and 27
  have a failed prepared/Q&A split.
- **Correlations** (`analyze_features_finbert.py`, heatmap `results/feature_correlation.png`):
  only one pair above 0.85, `neg_share`–`tone_p10` (−0.97), which is a near-duplicate. The tone block
  splits into a *level* group (the three section tones, `neg_share`, `tone_p10`) and a *spread*
  measure (`tone_std`, unrelated to the level). `lm_uncertainty_pct`–`lm_weak_modal_pct` is 0.61
  (moderate). `tone_gap` correlates 0.77 with `prep_net_tone` (partly mechanical, since it is computed from it), so
  that hypothesis failed. `delta_tone`, `specific_fls_share` and `numeric_density` stay weakly
  related to tone (≤ 0.32), as hoped.
- **Length artifacts:** none. Every feature has |r| < 0.12 with `log_word_count`.
- **GOOD vs BAD t-tests (pre-2018 only, report only):** executive-answer tone (p 0.001), prepared
  tone (p 0.005), `tone_p10` (p 0.006), weak modals (p 0.02) and `neg_share` (p 0.03) differ in
  the expected direction (GOOD calls sound slightly more positive). The effects are tiny
  (Cohen's d ≤ 0.08), and only executive-answer tone survives a Bonferroni correction. VADER
  showed nothing here, so this is an improvement, but expect only a small gain in Phase 3 AUC.
- **Downstream:** Phase 3 must switch inputs to `features_finbert.csv`, drop the 4 unparseable rows,
  and handle blanks (the walk-forward models need imputation or NaN-aware XGBoost).

**2026-11-11 — Phase 2 rebuild with FinBERT started (Kartik), 50-call trial done:**
- **Why:** Velu's 2026-11-07 check found whole-document VADER compound at +1.0 on 99.5% of calls,
  and research favours FinBERT on financial text. 16 of the 31 Phase 2 features came from VADER
  (10 section-level `vader_*` plus 6 paragraph-level `sent_compound_*` / `pct_*_sents`).
- **Label definition checked (`collect_data.py`):** ±2% abnormal return over trading days +2 → +60,
  NEUTRAL dropped. `docxxs/features.md` wrongly says ±1% and +2 → +62.
- **Full call set relabelled (`build_call_dataset.py` → `data/calls_all.csv`, gitignored):**
  12,789 calls, including the 2,296 Phase 1 NEUTRAL calls, which a real strategy couldn't have
  excluded in advance. Returns use Velu's corrected event dates (`event_returns.py`): `imm_ar`
  (day −1 → +1), `drift_ar` (+2 → +60), `label` = GOOD if `drift_ar` > 0 (6,817 GOOD / 5,972 BAD).
  All 10,493 old calls are included, and 97.7% keep their old label. 73 stub transcripts
  (< 1,000 words, mostly empty 2007 files) are flagged `is_stub` and skipped.
  `master.csv` and `features_expanded.csv` are unchanged, so Phase 3 still runs as before.
- **New features (`extract_features_finbert.py`):** 18 per call, scored per sentence with
  `yiyanghkust/finbert-tone` and `yiyanghkust/finbert-fls` (label orders read from each config:
  tone 0 Neutral / 1 Positive / 2 Negative; FLS 0 Not / 1 Non-specific / 2 Specific). Tone level
  (prepared, executive answers, analyst questions), bad moments (`neg_share`, `tone_p10`,
  `tone_std`), contrast (`tone_gap`, `delta_tone` vs the same ticker's previous call),
  specificity (`specific_fls_share`, `fls_density`), full Loughran-McDonald dictionary
  (uncertainty, weak/strong modal, negative), `numeric_density`, `qa_directness` (executive
  answers only), `log_word_count`, and `fog_index`, kept as the one readability measure since
  `flesch_ease` (r = −0.92) and `avg_sentence_length` (r = 0.78) duplicate it.
- **Text handling:** Q&A split at the 2nd "Operator:" (works for ~94–97% of calls). In Q&A, the first
  speaker after an Operator turn who didn't speak in the prepared remarks is the analyst. Checked
  against 76 calls that mark turns "Q -" / "A -": 96.1% of Q&A sentences tagged correctly.
- **50-call trial (RTX 4050):** 0.59 s per call, so ~2.1 h for all 12,716 non-stub calls. No
  unexpected blanks (only `delta_tone` on each ticker's first call). Early sanity: prepared
  remarks are more positive than executive answers (mean `tone_gap` +0.18), and analyst questions
  are close to neutral (mean −0.02). Two parsing bugs were found and fixed in the trial: long
  "Name - Title" speaker labels were missed, and a single stray "Q –" label switched a whole call
  to markup mode.
- **Downstream:** Phase 3 needs updating to read `data/features_finbert.csv` and the 12,789-call
  set; to agree with Velu.

**2026-11-07 — Phase 3 done (Velu):**
- **Date bug found and fixed.** Scripts compared the call's full timestamp against daily price
  dates, so a 9 a.m. call got "day 0" = the next day. Most calls (≈70%) are before or during
  market hours, so the immediate window missed the real reaction day. Day 0 is now the call's
  calendar date (`event_returns.py`; same fix in `collect_data.py`, `market_reaction.py`,
  `backtest.py`). Checked: average absolute abnormal return is 1.1% on normal days, 2.6% on
  day 0, 2.3% on day +1. Re-labelling with corrected dates keeps 97.7% of GOOD/BAD labels the same.
  `features_expanded.csv`'s `label` column still holds the old labels; Phase 3 recomputes
  labels from `prices.csv` and does not use it.
- **First finding: the original setup had no signal.** All 31 features → GOOD/BAD (drift)
  scored AUC ≈ 0.51 on a date split. Whole-document VADER compound is +1.0 on 99.5% of calls
  (3 columns dropped).
- **New design (two-step):** train on the *immediate* reaction (strong signal), use its
  probability as the GOOD/BAD score, and test against the *drift* in Phase 4. Training directly
  on drift was tried and ranks drift worse.
- **New inputs:** change vs. the firm's previous call (`delta`), TF-IDF text scores, previous-call
  similarity, pre-call return/volatility. `feature_study.py` picked delta + text + lm + market by
  greedy selection on 2012–2017 only.
- **Results (2018–2025 holdout, never used for choosing):** reaction AUC 0.588; score vs. drift
  rank-corr +0.056 (t 3.3); top-20% minus bottom-20% drift +1.30%. No-language baseline: +0.02%.
- **Phase 4 (old script, quick run):** high-confidence GOOD drift +1.02% vs. BAD +0.21%, p = 0.02
  (plain t-test; overlapping windows mean the true p-value is somewhat larger).
- **Phase 5 not trusted yet:** `backtest.py` chains thousands of overlapping 20-day trades as if
  sequential and annualises Sharpe as ~12 trades/year, giving a meaningless −99.8% drawdown.
  Outputs from that run were deleted. Needs a daily calendar-time long/short portfolio.
- **Also:** Phase 4/5 scripts crash on Windows consoles (they print → and ≥); run with
  `PYTHONIOENCODING=utf-8` until they are rewritten.

**2026-11-03 — Phase 3 split changed to train/validation/test, first validation run done:**
- **Why a validation set:** the script originally had only train (78.1%) and test (21.9%). Any adjustment made after seeing the test score would let the test set help choose the model, and Phases 4–5 rely on these predictions. Validation is now used for comparing and tuning, and the test set runs once.
- **Why gaps:** a label uses prices up to 61 trading days after the call (60 from the labeling window, plus 1 because the date lookup moves timed calls to the next trading day). Calls just before a boundary would leak later prices into training, so they're left out.
- **Split (10,493 calls, 2005-10-13 → 2025-05-15):** train 6,113 (58.3%, to 2018-10-02); validation 1,783 (17.0%, 2019-01-09 → 2021-10-05); test 2,296 (21.9%, 2022-01-06 → 2025-05-15); unused gaps 301 (2.9%).
- **`train_classifier.py` changes:** default run trains and scores on validation only and saves nothing. `--final` retrains on all 8,043 pre-test calls, scores the test set, and saves models, `data/predictions.csv` and `docxxs/Phase3_results.md`.
- **First validation results (no tuning):** Logistic Regression 50.4% accuracy / 0.519 AUC, predicting GOOD for 93% of calls; XGBoost 51.2% / 0.511, predicting GOOD for 70%. Always guessing the larger class gets 50.1%. With a rough AUC margin of error of ±0.03, neither is clearly better than chance.
- **Benchmark correction:** only 47.0% of test calls are GOOD, so the test accuracy to beat is 53.0% (always guessing BAD), not the 53.7% GOOD share across the whole dataset.
- **Downstream effect:** Phases 4–5 need `data/predictions.csv`, which now exists only after `--final`. If the signal stays this weak, Phase 4 may show no meaningful difference, which is still a valid finding.
- **Superseded 2026-11-07:** the walk-forward design above (2026-11-07 entry) replaced this fixed split and the `--final` flag. Its ideas carry over: inputs are chosen on validation years (2012–2017) only, a held-out period (2018–2025) is never used for choosing, and an embargo gap keeps later prices out of training. These AUC 0.51–0.52 results matched the walk-forward rerun of the original 31 features (≈0.51).

**2026-11-03 — Local data regenerated and verified, unused data files removed:**
- **Verified Velu's scaled dataset on Kartik's machine:** regenerated `data/master.csv` with `python collect_data.py --start 2005-01-01 --end 2025-12-31` (no `--force`, so it reused the committed `data/prices.csv`). Result: 10,493 rows, 5,638 GOOD / 4,855 BAD; every row matches `data/features_expanded.csv` on ticker + call date, and every label agrees.
- **Regeneration gotcha:** a first run with the script's default dates (2021-01-01 → 2024-06-30) produced only 2,317 rows, and the script reuses an existing `raw_transcripts.csv`, so that file had to be deleted before re-running with the full window. Steps are now in `README.md`.
- **Removed `features.csv` from the repo root:** the old 509-row, 17-feature prototype output, superseded by `data/features_expanded.csv` and not read by any script for Phases 3–6. Older entries below mention it as history; it can be recovered from Git (last changed in `6b94ea8`).
- **Removed `extract_features.py`:** the old single-process FinBERT script that produced `features.csv`, replaced by `extract_features_fast.py` + `_feat_worker.py`. No script imports it, and it can be recovered from Git (last changed in `6b94ea8`). Older entries below mention it as history.
- **Deleted local `data/raw_transcripts.csv` and `data/labeled_all.csv`:** both gitignored, so this only affects Kartik's machine. No Phase 3–6 script reads them, and `collect_data.py` can rebuild them.
- **Downstream effect:** Phases 3 and 5 need only committed files. Phase 4 and the dashboard also need local `data/master.csv`, which every collaborator must generate or be sent.

**2026-10-19 — Data Scaling & Fast Multi-Worker Feature Extraction Complete:**
- **Data Scaling (Priority 1 from Velu's Proposal):** Expanded universe from 45 to 200 tickers and window to 2005–2025. Collected 10,493 transcripts, labeled via market abnormal return: 5,638 GOOD / 4,855 BAD in `data/master.csv`.
- **Parallel Feature Pipeline:** Built `extract_features_fast.py` and `_feat_worker.py` utilizing 7 concurrent background worker processes, processing all 10,493 transcripts in ~3.5 hours.
- **Expanded Feature Set (35 columns):** Replaced slow FinBERT with high-throughput VADER (prepared remarks, Q&A, and paragraph distributions), Loughran-McDonald dictionary percentages (uncertainty, weak/strong modals, litigious), Fog/Flesch readability indices, numeric density, forward-looking statements, and Q&A interaction metrics.
- **Complete Mapping in `docxxs/features.md`:** Documented all 35 columns with actual min/mean/max/null metrics calculated across the entire 10,493-row dataset.
- **Repository & Teammate Access Architecture:** Excluded >100MB raw files (`raw_transcripts.csv`, `master.csv`, `labeled_all.csv`) from Git tracking to comply with GitHub limits. Committed `data/features_expanded.csv` (3.98 MB) and `data/prices.csv` (18.02 MB) so teammates can immediately train models in Phase 3 without requiring raw data downloads or hours of feature extraction.

**2026-09-28 — NaN-handling decision locked:** Kartik chose to **leave the 4 NaN rows in
`features.csv` as-is** rather than drop them now — deferred to Phase 3, where XGBoost can
handle missing values natively. Note for whoever builds the Logistic Regression baseline first:
scikit-learn's `LogisticRegression` cannot take `NaN` input, so those 4 rows (or at least their
`finbert_qa_*`/`qa_directness` columns) will need dropping or imputing *at that step specifically*
— this was deliberately not pre-solved in `features.csv` itself.

**2026-09-28 — Sanity-check pass on `features.csv`, limitations found:**
- Directional check: AAPL's known-outcome calls (2021-01-27, 2022-10-27, 2024-02-01 — all
  previously validated as BAD events in Phase 1) show correctly as BAD in `features.csv`.
- Distribution check: all 17 features fall in sane ranges, no broken/impossible values.
- **Correlation-with-label check (the important one): a Welch's t-test on all 17 features
  between GOOD and BAD groups found no statistically significant separation on any single
  feature** — best p-value is 0.10 (`finbert_qa_mean`), everything else p > 0.28,
  `numeric_density` is essentially identical between groups. `finbert_qa_mean` even runs
  slightly counter-intuitive (higher for BAD than GOOD).
- **This does not mean the pipeline is broken** — t-tests only catch linear/univariate
  separation, and Phase 3's classifier (especially XGBoost) can pick up interactions a t-test
  can't. But it's a legitimate, known-hard-problem signal (executives tend to sound calm
  regardless of actual results) rather than a given the classifier will work well. Treated as
  an open empirical question for Phase 3, not assumed either way.
- Full limitations list (8 items, each with a possible mitigation) written into
  `docxxs/Phase2_Signal_Extraction.md`, covering: weak group separation, counter-intuitive
  direction, the 4 unsplit-Q&A rows, LM's blindness to negation, incomplete dilution fix from
  chunking, FinBERT only spot-checked on AAPL so far, naive specificity regex, and discarding
  FinBERT's neutral-class probability.
- Added `docxxs/phase2_flowchart.svg` (mirrors Phase 1's flowchart) showing the full pipeline:
  `master.csv` → section split → chunking → FinBERT/LM/fog/specificity → `features.csv` →
  Phase 3.
- **Downstream effect:** Phase 3 should not assume strong accuracy is guaranteed going in —
  a modest or even weak result is itself a valid, reportable finding for this project, not a
  failure, given the honest empirical picture from this sanity check.

**2026-09-27 — Phase 2 built, run, and a real bug found and fixed:**
- Wrote `extract_features.py` per the locked Option D design (8 FinBERT + 5 LM + 1 readability
  + 3 specificity = 17 features/call).
- First run hit two real bugs, both fixed and verified against live output before accepting
  results:
  1. `pysentiment2`'s `LM.get_score()` only exposes Positive/Negative/Polarity/Subjectivity —
     it silently doesn't have Uncertainty/Modal/Litigious. Fixed by reading the bundled LM
     dictionary CSV directly and building those category word-sets ourselves (`Modal==1`
     strong, `Modal==3` weak, confirmed against the real CSV contents).
  2. The "first question" keyword marker for the prepared-remarks/Q&A split failed on 31/509
     (6%) transcripts (different phrasing, e.g. Cisco just announces the caller's name with no
     "first question" text), and was found to also silently mis-split some already-"successful"
     rows when an analyst said "my first question is..." mid-Q&A. Replaced with a structural
     rule — split at the **2nd `"Operator:"` turn** — which dropped failures to 4/509 (0.8%),
     all genuine edge cases (no Operator-tagged structure, including one row where the
     transcript tagged `TMO` is actually a PPD earnings call — a source-dataset labeling issue,
     not our bug).
- Final `features.csv`: 509 rows, 255 GOOD / 254 BAD (matches Phase 1 exactly), 4 rows with
  `NaN` Q&A features (decision on how to handle these — drop/impute/leave-as-is for
  XGBoost — still open).
- Full feature reference written to `docxxs/features.md`.
- **Downstream effect:** Phase 3 trains directly on `features.csv` + `label`; the NaN-handling
  decision needs to be made before or during Phase 3's training step, not deferred further.

---

## Decision log (newest first)

### 2026-09-15 — Phase 2 design discussion (in progress, not finalized)
- Confirmed via `master.csv`: transcripts average ~9,400 words (range 1,168–31,216) — 20-80x
  longer than FinBERT's 512-token limit, so chunking is mandatory, not optional.
- Identified the **dilution problem**: averaging FinBERT sentiment across ~20+ mostly-neutral
  chunks per call washes out the 1-2 chunks that actually carry signal (worked example: a
  chunk with 0.58 negative-sentiment probability gets buried inside a whole-call average of
  ~0.06). Decided against a single mean-sentiment feature for this reason.
- Decided (in principle) to compute **mean / min / max / std** of chunk-level sentiment
  instead of a single mean, so outlier moments survive into the feature set even when the
  overall average looks calm.
- Confirmed empirically that the "first question" text marker reliably separates prepared
  remarks from Q&A: present in 49/50 sampled transcripts (98%).
- Discussed whether to hand-assign different weights to different sections (boilerplate vs.
  prepared remarks vs. Q&A vs. Q&A follow-ups) before averaging. **Decided against
  hand-picking weights.** Reasoning: any weight we choose ourselves (e.g. "Q&A = 3x
  prepared remarks") is a guess baked permanently into the feature before any evidence
  supports it — if wrong, it silently makes the feature worse with no way to catch it later.
  Instead: keep prepared-remarks and Q&A features as **separate columns**, drop pure
  boilerplate (that one's safe — it's noise removal, not an assumption about importance),
  and let **Phase 3's classifier learn the actual weighting** from labeled outcomes (its
  logistic regression coefficients / XGBoost feature importances *are* the answer to "how
  much does each part matter", discovered from data instead of assumed).
- Decided against trying to detect Q&A *follow-up* questions specifically (vs. first
  answers) in this pass — text-pattern detection for that is much less reliable than the
  "first question" marker; flagged as a possible future refinement, not part of the first
  build.
- **Downstream effect:** this shapes `features.csv`'s schema directly — likely 8 FinBERT
  columns (4 stats × 2 sections) instead of 4, plus it means Phase 3 should be read closely
  once trained, since its feature importances will retroactively validate (or contradict)
  the "Q&A matters more" assumption embedded in this design.
- **Remaining work:** final go/no-go on Option C vs Option D from Kartik, then build and run
  `extract_features.py`.

### 2026-09-14 — Repo structure flattened
- Moved all project files from the nested `Mini_project26/` subfolder up to the repo root
  (`git mv`, preserving file history as renames, not delete+add).
- Fixed two now-stale references that assumed the nested layout: the folder-tree diagrams in
  `README.md` and `docxxs/Phase0_Setup_and_Foundation.md` (now show `.` as root instead of
  `Mini_Project26/`), and a stale "gitignored — regenerate locally" comment on `data/` left
  over from before Phase 1's data was actually committed.
- Verified via `git diff` that only those two `.md` files changed content — every other file
  was a pure rename with zero content diff.
- Committed (`361fd3d`) and pushed to `origin/master`.
- **Downstream effect:** any doc or script referencing paths relative to a `Mini_project26/`
  prefix would now be wrong — checked, none currently do (the `docxxs/*.md` links like
  `../requirements.txt` were already correct, since `docxxs/` was and still is one level
  below the root files either way).

### 2026-09-13/14 — Phase 1 pipeline executed for real, data committed
- Found `data/` contained only a `.gitkeep` despite `docxxs/Phase1_Data_Collection.md`
  claiming specific real results — flagged this discrepancy to Kartik rather than assuming
  the docs were accurate. Confirmed with him this needed a real re-run, not just a docs fix.
- Hit two environment blockers before the pipeline would run at all (see also the
  `env-windows-ssl-avast` memory entry for full technical detail, kept out of this repo file
  since it's machine-specific, not project-specific):
  - This machine has two separate Python installs; bare `pip` was installing into the wrong
    one. Fix: use `python -m pip install ...`.
  - Avast's HTTPS interception broke SSL verification for every external request (Hugging
    Face, PyPI, Yahoo Finance). Fixed via `pip-system-certs` (for `requests`/`urllib3`-based
    calls) plus manually exporting the Windows Root CA store to a PEM file and pointing
    `CURL_CA_BUNDLE`/`SSL_CERT_FILE` at it (needed separately for `yfinance`, which uses
    `curl_cffi`'s own independent SSL stack, unaffected by the first fix).
- Ran `python collect_data.py --force` for real. Result **exactly matched** what the
  (previously unverified) docs had claimed: 630 transcripts fetched, 255 GOOD / 254 BAD /
  121 NEUTRAL labeled, 509 rows in `master.csv` — confirming the earlier claims were
  genuine, just never actually committed.
- Spot-checked several real AAPL rows against known real-world events (Feb-2021 growth-stock
  rotation, Oct-2022 guidance/China concerns, Feb-2024 China iPhone demand miss) — labels
  matched what actually happened, so the labeling logic is behaving sensibly, not just
  mechanically correct.
- Removed the `data/*.csv` exclusion from `.gitignore` — deliberate, per project scope: the
  brief calls for committing the real dataset despite the size cost (~95MB across 4 files),
  since later phases need to build directly on `master.csv` without requiring a re-run.
  Clarified for Kartik that this ~95MB is *not* the same as the ~5GB Hugging Face dataset
  cache his machine downloaded — that cache is the full 33,362-transcript universe (every
  S&P 500 company, 2005–2025), lives outside the repo entirely, and is a one-time cost —
  Phase 2 onward reads `transcript_text` straight out of `master.csv` and never touches
  Hugging Face again.
- Committed and pushed (originally `af8bc12`, later amended — see next entry).
- **Downstream effect:** `data/master.csv` is now the fixed, real input every later phase
  builds on. No further changes expected to this file unless the labeling threshold or
  ticker universe changes.

### 2026-09-14 — Commit attribution corrected
- Kartik pointed out he'd already asked, before this session, for commits to never include
  a "Co-Authored-By: Claude" line — this had been added anyway (default session behavior).
- Amended the Phase 1 data commit to remove the attribution line and force-pushed it (safe:
  it was the branch tip, pushed only minutes earlier, nothing built on top of it yet).
  Hash changed from `af8bc12` to `18766bf`.
- Saved as a standing instruction so this doesn't recur in future sessions.
- **Downstream effect:** none on project content — commit hash changed, message content did
  not otherwise change.

### (earlier, before this session) — Phase 0 complete
- 45-ticker, sector-diversified universe fixed in `tickers.csv`; date window
  2021-01-01 → 2024-06-30 chosen. Full rationale in `docxxs/Phase0_Setup_and_Foundation.md`.

---

## Remaining work, by phase

- **Phase 2:** ✅ Done — `extract_features.py` written and run for real, `features.csv`
  produced (509 rows × 17 features), sanity-checked, limitations documented with mitigations
  in `docxxs/Phase2_Signal_Extraction.md`, flowchart added, all committed (`6b94ea8`, `3a7195f`).
  The 4 NaN Q&A rows are deliberately left unresolved in the data itself — see the NaN-handling
  decision above; Phase 3 handles them at training time. (That prototype was later superseded
  by `data/features_expanded.csv` — 10,493 rows × 35 features — and `features.csv` was removed
  on 2026-11-03.)
- **Phase 3:** train Logistic Regression baseline then XGBoost on `data/features_expanded.csv`,
  using a date-based (not random) train/test split; handle missing values for the Logistic
  Regression step specifically (drop or impute — sklearn can't take NaN, XGBoost can);
  evaluate accuracy/AUC/confusion matrix; inspect feature importances (this is also where
  Phase 2's "no hand-picked section weights" decision and the weak-univariate-signal finding
  both get empirically tested — a modest result here is a valid finding, not a failure);
  output `predictions.csv`.
- **Phase 4:** compare immediate (day 0→+1) vs. drift (day +2→+20) market reaction windows
  against Phase 3's predicted confidence, via t-test.
- **Phase 5:** long-GOOD/short-BAD backtest with a confidence filter, transaction costs,
  Sharpe ratio, equity curve.
- **Phase 6:** React + Recharts dashboard, final written report.
