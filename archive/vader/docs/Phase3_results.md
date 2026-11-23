# Phase 3 — Classifier Results

All numbers are **out-of-sample**: each year is predicted by a model trained only on
earlier years (walk-forward), with a gap so no training target overlaps the test year.
Inputs were chosen on the validation years 2012–2017 only
(`feature_study.py`); the holdout years were not used while choosing.
Regenerate with `python train_classifier.py`.

**Final model:** Logistic Regression trained to predict the immediate reaction
(day −1 → +1) from: delta, text, lm, market (37 inputs). Its probability is the GOOD/BAD score.
**GOOD/BAD:** GOOD if the drift (day +2 → +60 abnormal return) was positive.

Columns: *Reaction AUC* — how well the score predicts the immediate reaction;
*GOOD/BAD AUC / accuracy* — against the drift label (accuracy uses a 0.5 cut-off);
*Drift rank-corr* — average yearly Spearman correlation between score and drift, with its
t-statistic across years; *Drift top-bottom 20%* — average drift of the 20% highest-scored
calls minus the 20% lowest, within each year.

## Holdout years 2018–2025 (untouched while choosing)

| Model | Reaction AUC | GOOD/BAD AUC | GOOD/BAD accuracy | Drift rank-corr (t) | Drift top-bottom 20% |
|---|---|---|---|---|---|
| **Final** — reaction-trained, Logistic Regression | 0.5875 | 0.5281 | 0.517 (majority 0.510) | +0.0561 (+3.3) | +1.30% |
| Reaction-trained, XGBoost | 0.5813 | 0.5236 | 0.514 (majority 0.510) | +0.0509 (+2.4) | +1.43% |
| Trained straight on GOOD/BAD (text + market + reaction) | — | 0.5305 | 0.492 (majority 0.510) | +0.0579 (+1.9) | +1.10% |
| No language — market + reaction only | — | 0.5179 | 0.489 (majority 0.510) | +0.0248 (+1.0) | +0.02% |

## Validation years 2012–2017 (used to choose inputs)

| Model | Reaction AUC | GOOD/BAD AUC | GOOD/BAD accuracy | Drift rank-corr (t) | Drift top-bottom 20% |
|---|---|---|---|---|---|
| **Final** — reaction-trained, Logistic Regression | 0.5953 | 0.5168 | 0.520 (majority 0.575) | +0.0370 (+1.2) | +1.39% |
| Reaction-trained, XGBoost | 0.5697 | 0.5122 | 0.516 (majority 0.575) | +0.0351 (+1.2) | +0.71% |
| Trained straight on GOOD/BAD (text + market + reaction) | — | 0.5191 | 0.576 (majority 0.575) | +0.0155 (+1.4) | +0.64% |
| No language — market + reaction only | — | 0.5216 | 0.577 (majority 0.575) | +0.0173 (+1.0) | +0.37% |

## Final model — GOOD/BAD confusion matrix (holdout)
```
             pred BAD  pred GOOD
  actual BAD     1029      1434
  actual GOOD     899      1471
```

## Largest coefficients (final model, standardised inputs)

| Input | Coefficient | Direction |
|---|---|---|
| `text_full` | +0.205 | → reaction up |
| `d_forward_looking_count` | -0.134 | → reaction down |
| `d_fog_index` | -0.128 | → reaction down |
| `d_avg_sentence_length` | +0.107 | → reaction up |
| `d_vader_neg` | +0.106 | → reaction up |
| `d_vader_qa_neg` | -0.100 | → reaction down |
| `d_lm_net_sentiment` | +0.079 | → reaction up |
| `d_sent_compound_mean` | +0.074 | → reaction up |
| `d_vader_qa_pos` | +0.071 | → reaction up |
| `d_qa_ratio` | -0.071 | → reaction down |
| `d_sent_compound_max` | -0.070 | → reaction down |
| `pre_ar` | -0.067 | → reaction down |

## Reading these numbers
- Short-term stock moves are mostly noise, so AUC 0.55–0.60 is a strong result for
  this kind of data. An AUC far above that would point to leaked future information.
- GOOD/BAD accuracy stays close to the always-majority rate: the score is a *ranking*
  signal, so its value shows in the top-vs-bottom drift spread, which Phase 4 tests.
- Compare the final row with the *no language* row: the gap is what language adds.
- Why these inputs: `docxxs/Phase3_feature_study.md`.