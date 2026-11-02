# Phase 3 — Type Classifier (Good / Bad)

**Status:** 🔄 Ready to train (`train_classifier.py`)

## Goal
A trained model that predicts Good/Bad from language alone — no future price data — and a
measured accuracy that proves the signal is real.

## Plan
1. Split by date (train on older calls, test on newer — mimics real forecasting).
2. Baseline **Logistic Regression**, then **XGBoost**.
3. Evaluate: accuracy, AUC, confusion matrix.
4. Read feature importances — which language cues drive "Bad"?
5. Output per call: predicted label + confidence `P(Good)`.

## Input / Output
- **Input:** `data/features_expanded.csv` (Phase 2), which already includes `label`.
- **Output:** trained models in `models/` + `data/predictions.csv` — `[ticker, call_date, label, predicted, confidence_good]`.

## Next
→ Phase 4 — Market-Reaction Test
