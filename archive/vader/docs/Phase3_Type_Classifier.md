# Phase 3 — Type Classifier (Good / Bad)

**Status:** ✅ Complete — results in [`Phase3_results.md`](Phase3_results.md), feature
evidence in [`Phase3_feature_study.md`](Phase3_feature_study.md).

## Goal
A model that scores each call as GOOD or BAD from language alone (no future prices), with
an honest out-of-sample measure of how well it works.

## Approach
1. **Corrected event dates.** Day 0 is the call's calendar date (time of day dropped), so
   before-open calls are no longer shifted one day late (`event_returns.py`).
2. **Walk-forward testing.** Each year is predicted by a model trained only on earlier years,
   with a gap so no training target overlaps the test year. This gives out-of-sample
   predictions for every year from 2010 onward instead of one fixed split.
3. **Two-step design.** The model is trained to predict the *immediate* reaction
   (day −1 → +1), which measures what news the language carries. Its probability is the
   GOOD/BAD score. Phase 4 tests whether that score also predicts the *drift* (day +2 → +60),
   i.e. whether the market under-reacted. Training straight on the drift label was tried
   and did worse (the drift label is very noisy).
4. **Inputs chosen by evidence, not guesswork.** `feature_study.py` scores every feature
   group, then adds groups one at a time and keeps one only if it raises validation AUC
   (2012–2017) by at least 0.002. The 2018–2025 holdout years were not used for choosing.

## Inputs
| Group | What it is | Kept? |
|---|---|---|
| delta | change in each Phase 2 feature vs. the firm's previous call | ✅ biggest single gain |
| text | TF-IDF + logistic-regression score on the transcript words (full, prepared, Q&A) | ✅ |
| lm | Loughran-McDonald positive / negative / uncertainty / net | ✅ |
| market | 3-month abnormal return and volatility before the call | ✅ |
| sentiment, style, qa | Phase 2 VADER, readability, Q&A dynamics (raw levels) | ❌ no gain once the above are in |
| similarity | word overlap with the previous call | ❌ no gain |
| — | whole-document VADER compound (3 columns) | ❌ dropped: +1.0 on 99.5% of calls |

Final model: Logistic Regression on 37 inputs (it beat XGBoost on this data).

## Results (holdout years 2018–2025)
- Immediate reaction: **AUC 0.588**.
- Later drift: rank correlation +0.056 (t = 3.3); the 20% highest-scored calls drift
  **+1.30%** more than the 20% lowest over ~3 months.
- A model without language (market + reaction only) gets +0.02% on the same comparison,
  so the drift signal comes from the language.
- GOOD/BAD accuracy is 51.7% vs. 51.0% for always guessing the majority. The score is a
  ranking signal: it is useful at the extremes, not as a yes/no call on every firm.

## How to run
```bash
python train_classifier.py              # ~30 s using committed data/text_scores.csv
python train_classifier.py --rebuild-text   # needs local data/master.csv, ~6 min
python feature_study.py                 # optional, slow: re-runs the input selection
```

## Outputs
- `data/predictions.csv` — out-of-sample score per call, 2012–2025 (`confidence_good`,
  `predicted`, plus every comparison model's score)
- `data/text_scores.csv` — out-of-sample text scores (committed; lets teammates run without master.csv)
- `models/final_model.joblib` — final model refit on all data

## Earlier attempt (2026-11-03, Kartik)
A fixed split (train to 2018, validation 2019–2021, test 2022+, with 61-trading-day gaps) on the
original 31 features gave validation AUC 0.519 (Logistic Regression) and 0.511 (XGBoost), no
better than chance. That result led to the current design; its validation/holdout separation and
leakage gap are kept in the walk-forward setup. Details in `PROGRESS.md`.

## Next
→ Phase 4 — Market-Reaction Test
