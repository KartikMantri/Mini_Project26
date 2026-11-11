# Phase 3 (v2) — Classifier Results on FinBERT Features

Same walk-forward method as [`Phase3_results.md`](Phase3_results.md) (Velu, VADER),
with the VADER features replaced by the 18 FinBERT/LM features
([`features_finbert.md`](features_finbert.md)) and the full call set (12,712 calls, old NEUTRAL calls included).
Regenerate with `python train_classifier_finbert.py`.

**Final model:** LR trained to predict the immediate reaction (day −1 → +1)
from: tone, market (10 inputs), chosen greedily on the
validation years 2012–2017 only. Its probability is the GOOD/BAD score.
**GOOD/BAD:** GOOD if the drift (day +2 → +60 abnormal return) was positive.

Columns as in `Phase3_results.md`. Short-term returns are mostly noise: AUC 0.55–0.60 on
the reaction is a strong result; GOOD/BAD AUC barely above 0.5 is normal, and the
top-minus-bottom 20% drift spread is what Phase 4 tests.

## Main: walk-forward, holdout 2018–2025 (untouched while choosing)

| Model | Reaction AUC | GOOD/BAD AUC | GOOD/BAD accuracy | Drift rank-corr (t) | Drift top-bottom 20% |
|---|---|---|---|---|---|
| **Final** — reaction-trained, LR, greedy pick | 0.6132 | 0.5350 | 0.519 (majority 0.502) | +0.0625 (+2.9) | +1.93% |
| Reaction-trained, XGB, its own greedy pick | 0.6168 | 0.5337 | 0.520 (majority 0.502) | +0.0624 (+2.9) | +1.76% |
| Reaction-trained, FinBERT/LM features only (no text, no market) | 0.6234 | 0.5244 | 0.512 (majority 0.502) | +0.0430 (+4.1) | +1.04% |
| Trained straight on GOOD/BAD (tone + text + market + reaction) | — | 0.5150 | 0.495 (majority 0.502) | +0.0252 (+1.1) | +0.69% |
| No language — market + reaction only | — | 0.5146 | 0.497 (majority 0.502) | -0.0007 (-0.0) | +0.19% |

## Check: Kartik's fixed split, test 2022–2025

Trained once on 9,798 calls before 2021-10-06 (a 61-trading-day gap
before 2022, so no training label uses 2022 prices), tested once on 2022–2025. Same inputs
as the walk-forward models.

| Model | Reaction AUC | GOOD/BAD AUC | GOOD/BAD accuracy | Drift rank-corr (t) | Drift top-bottom 20% |
|---|---|---|---|---|---|
| **Final** — reaction-trained, LR, greedy pick | 0.6131 | 0.5291 | 0.511 (majority 0.521) | +0.0805 (+1.9) | +2.29% |
| Reaction-trained, FinBERT/LM features only (no text, no market) | 0.6176 | 0.5129 | 0.497 (majority 0.521) | +0.0410 (+2.4) | +1.22% |
| No language — market + reaction only | — | 0.5111 | 0.478 (majority 0.521) | -0.0031 (-0.1) | +0.94% |

For reference, the walk-forward models on the same 2022–2025 calls:

| Model | Reaction AUC | GOOD/BAD AUC | GOOD/BAD accuracy | Drift rank-corr (t) | Drift top-bottom 20% |
|---|---|---|---|---|---|
| **Final** — reaction-trained, LR, greedy pick | 0.6150 | 0.5271 | 0.510 (majority 0.521) | +0.0749 (+1.8) | +2.04% |
| Reaction-trained, XGB, its own greedy pick | 0.6172 | 0.5251 | 0.507 (majority 0.521) | +0.0759 (+1.9) | +2.21% |
| Reaction-trained, FinBERT/LM features only (no text, no market) | 0.6188 | 0.5145 | 0.501 (majority 0.521) | +0.0446 (+2.6) | +1.01% |
| Trained straight on GOOD/BAD (tone + text + market + reaction) | — | 0.4983 | 0.476 (majority 0.521) | +0.0229 (+0.6) | +0.60% |
| No language — market + reaction only | — | 0.5173 | 0.478 (majority 0.521) | +0.0172 (+1.2) | +1.04% |

## FinBERT vs Velu's VADER run, on the same 4,833 holdout calls

Only calls present in both runs (Velu's set excludes the old NEUTRAL calls). Velu's
score is `score_reaction_lr` from his `data/predictions.csv`.

| Model | Reaction AUC | GOOD/BAD AUC | GOOD/BAD accuracy | Drift rank-corr (t) | Drift top-bottom 20% |
|---|---|---|---|---|---|
| Velu — VADER features, 2018–2025 | 0.5875 | 0.5281 | 0.517 (majority 0.510) | +0.0561 (+3.3) | +1.30% |
| Ours — FinBERT features, 2018–2025 | 0.6114 | 0.5330 | 0.517 (majority 0.510) | +0.0611 (+2.8) | +2.28% |
| Velu — VADER features, 2022–2025 | 0.5912 | 0.5211 | 0.518 (majority 0.531) | +0.0675 (+2.7) | +1.50% |
| Ours — FinBERT features, 2022–2025 | 0.6137 | 0.5244 | 0.510 (majority 0.531) | +0.0736 (+1.8) | +2.19% |

## Validation years 2012–2017 (used to choose inputs)

| Model | Reaction AUC | GOOD/BAD AUC | GOOD/BAD accuracy | Drift rank-corr (t) | Drift top-bottom 20% |
|---|---|---|---|---|---|
| **Final** — reaction-trained, LR, greedy pick | 0.6403 | 0.5174 | 0.524 (majority 0.567) | +0.0323 (+1.9) | +1.05% |
| Reaction-trained, XGB, its own greedy pick | 0.6338 | 0.5180 | 0.526 (majority 0.567) | +0.0410 (+2.7) | +0.82% |
| Reaction-trained, FinBERT/LM features only (no text, no market) | 0.6264 | 0.5178 | 0.527 (majority 0.567) | +0.0306 (+2.7) | +1.00% |
| Trained straight on GOOD/BAD (tone + text + market + reaction) | — | 0.5142 | 0.568 (majority 0.567) | +0.0214 (+1.1) | +0.58% |
| No language — market + reaction only | — | 0.5105 | 0.567 (majority 0.567) | +0.0266 (+1.7) | +0.60% |

## Greedy selection log (validation reaction AUC)

| Model | Step | Valid AUC | Gain | |
|---|---|---|---|---|
| LR | +tone | 0.6357 | +0.1357 | kept |
| LR | +market | 0.6403 | +0.0046 | kept |
| LR | +text | 0.6398 | -0.0005 | stop |
| XGB | +tone | 0.6220 | +0.1220 | kept |
| XGB | +market | 0.6291 | +0.0071 | kept |
| XGB | +lm | 0.6338 | +0.0047 | kept |
| XGB | +fls | 0.6331 | -0.0007 | stop |
