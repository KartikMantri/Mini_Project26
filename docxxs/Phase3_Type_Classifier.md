# Phase 3 — Type Classifier (Good / Bad)

**Status:** 🟡 In progress — first validation run done; final test run not done yet

## Goal
A trained model that predicts Good/Bad from language alone — no future price data — and a
measured accuracy that proves the signal is real.

## Plan
1. Split by date (train on older calls, validate on 2019–2021, test on 2022 onward — mimics
   real forecasting), leaving a 61-trading-day gap before each boundary because labels look
   ahead that far. Tune against validation; run the test set once with `--final`.
2. Baseline **Logistic Regression**, then **XGBoost**.
3. Evaluate: accuracy, AUC, confusion matrix.
4. Read feature importances — which language cues drive "Bad"?
5. Output per call: predicted label + confidence `P(Good)`.

## Input / Output
- **Input:** `data/features_expanded.csv` (Phase 2), which already includes `label`.
- **Output:** trained models in `models/` + `data/predictions.csv` — `[ticker, call_date, label, lr_predicted, lr_confidence_good, xgb_predicted, xgb_confidence_good]` (the `xgb_*` columns are only written if XGBoost is installed).

## Data split
All 10,493 calls run from 2005-10-13 to 2025-05-15.

| Set | Dates of calls | Calls | Share | GOOD |
|---|---|---|---|---|
| Train | 2005-10-13 → 2018-10-02 | 6,113 | 58.3% | 56.9% |
| Gap 1 (unused) | Oct–Dec 2018 | 147 | 1.4% | — |
| Validation | 2019-01-09 → 2021-10-05 | 1,783 | 17.0% | 49.9% |
| Gap 2 (unused) | Oct–Dec 2021 | 154 | 1.5% | — |
| Test | 2022-01-06 → 2025-05-15 | 2,296 | 21.9% | 47.0% |

**Why the gaps:** a label uses prices up to 61 trading days after the call, so a call from
December 2021 has a label that depends on March 2022 prices. Training on it would leak
information from the following period into the model.

The `--final` run retrains on all 8,043 calls before Gap 2 (2005-10-13 → 2021-10-05), then
scores the test set.

## First validation results
Fixed model settings, no tuning. Always guessing the larger class gets 50.1% accuracy.

| Model | Accuracy | AUC | Calls predicted GOOD |
|---|---|---|---|
| Logistic Regression | 50.4% | 0.519 | 93% |
| XGBoost | 51.2% | 0.511 | 70% |

The signal so far is very weak: with a rough AUC margin of error of ±0.03, neither model is
clearly better than chance or than the other. These are validation scores, not final results.

## TBD
- [ ] Try one small improvement round on validation only: simpler XGBoost settings and
      balanced GOOD/BAD class weighting. Don't keep tuning past this, to avoid fitting noise.
- [ ] Run `python train_classifier.py --final` once the model is settled. Run it only once.
- [ ] Compare the test score with always guessing the larger class (53.0% on the test set).
- [ ] Review the feature importances and record what they show. Treat the Logistic Regression
      readability weights with care, because those three features overlap heavily.
- [ ] Update the warning at `dashboard.py:232` to say `python train_classifier.py --final`.
- [ ] Write up the result, including a weak one as a valid finding.

## Next
→ Phase 4 — Market-Reaction Test
