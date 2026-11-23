# Phase 3 — Classifier Results on FinBERT Features

Walk-forward classifier on the 18 FinBERT/LM features
([`features_finbert.md`](features_finbert.md)) and the full call set (12,712 calls, old NEUTRAL calls included).
Regenerate with `python phase3_train_classifier.py`.

**Final model:** LR trained to predict the immediate reaction (day −1 → +1)
from: tone, market (10 inputs), chosen greedily on the
validation years 2012–2017 only. Its probability is the GOOD/BAD score.
**GOOD/BAD:** GOOD if the drift (day +2 → +60 abnormal return) was positive.

Short-term returns are mostly noise: AUC 0.55–0.60 on
the reaction is a strong result; GOOD/BAD AUC barely above 0.5 is normal, and the
top-minus-bottom 20% drift spread is what Phase 4 tests.

## Main: walk-forward, holdout 2018–2025 (untouched while choosing)

| Model | Reaction AUC | GOOD/BAD AUC | GOOD/BAD accuracy | Drift rank-corr (t) | Drift top-bottom 20% |
|---|---|---|---|---|---|
| **Final** — reaction-trained, LR, greedy pick | 0.6112 | 0.5371 | 0.522 (majority 0.501) | +0.0674 (+3.0) | +2.25% |
| Reaction-trained, XGB, its own greedy pick | 0.6182 | 0.5330 | 0.520 (majority 0.501) | +0.0599 (+2.8) | +1.89% |
| Reaction-trained, FinBERT/LM features only (no text, no market) | 0.6241 | 0.5250 | 0.513 (majority 0.501) | +0.0438 (+4.3) | +1.16% |
| Trained straight on GOOD/BAD (tone + text + market + reaction) | — | 0.5105 | 0.499 (majority 0.501) | +0.0150 (+0.7) | -0.28% |
| No language — market + reaction only | — | 0.5130 | 0.499 (majority 0.501) | -0.0031 (-0.2) | -0.69% |

## Check: Kartik's fixed split, test 2022–2025

Trained once on 9,736 calls before 2021-10-06 (a 61-trading-day gap
before 2022, so no training label uses 2022 prices), tested once on 2022–2025. Same inputs
as the walk-forward models.

| Model | Reaction AUC | GOOD/BAD AUC | GOOD/BAD accuracy | Drift rank-corr (t) | Drift top-bottom 20% |
|---|---|---|---|---|---|
| **Final** — reaction-trained, LR, greedy pick | 0.6106 | 0.5323 | 0.513 (majority 0.520) | +0.0899 (+1.9) | +2.93% |
| Reaction-trained, FinBERT/LM features only (no text, no market) | 0.6186 | 0.5137 | 0.497 (majority 0.520) | +0.0421 (+2.5) | +1.06% |
| No language — market + reaction only | — | 0.5038 | 0.480 (majority 0.520) | -0.0162 (-0.4) | +0.57% |

For reference, the walk-forward models on the same 2022–2025 calls:

| Model | Reaction AUC | GOOD/BAD AUC | GOOD/BAD accuracy | Drift rank-corr (t) | Drift top-bottom 20% |
|---|---|---|---|---|---|
| **Final** — reaction-trained, LR, greedy pick | 0.6131 | 0.5297 | 0.512 (majority 0.520) | +0.0818 (+1.9) | +2.53% |
| Reaction-trained, XGB, its own greedy pick | 0.6218 | 0.5263 | 0.510 (majority 0.520) | +0.0747 (+2.0) | +2.42% |
| Reaction-trained, FinBERT/LM features only (no text, no market) | 0.6196 | 0.5153 | 0.503 (majority 0.520) | +0.0454 (+2.7) | +1.18% |
| Trained straight on GOOD/BAD (tone + text + market + reaction) | — | 0.5073 | 0.480 (majority 0.520) | +0.0465 (+1.1) | +0.72% |
| No language — market + reaction only | — | 0.5134 | 0.480 (majority 0.520) | +0.0177 (+1.5) | +0.61% |

## Earlier VADER baseline (past result)

An earlier version of this model on VADER features (Velu's pipeline) reached a holdout
reaction AUC of 0.588 against 0.611 here, on the same calls. That code and its data are
kept in [`archive/vader/`](../archive/vader/) and are no longer part of the pipeline.

## Validation years 2012–2017 (used to choose inputs)

| Model | Reaction AUC | GOOD/BAD AUC | GOOD/BAD accuracy | Drift rank-corr (t) | Drift top-bottom 20% |
|---|---|---|---|---|---|
| **Final** — reaction-trained, LR, greedy pick | 0.6401 | 0.5194 | 0.524 (majority 0.571) | +0.0391 (+2.3) | +1.19% |
| Reaction-trained, XGB, its own greedy pick | 0.6325 | 0.5166 | 0.528 (majority 0.571) | +0.0366 (+2.3) | +0.74% |
| Reaction-trained, FinBERT/LM features only (no text, no market) | 0.6264 | 0.5169 | 0.528 (majority 0.571) | +0.0285 (+2.3) | +0.93% |
| Trained straight on GOOD/BAD (tone + text + market + reaction) | — | 0.5189 | 0.571 (majority 0.571) | +0.0282 (+1.6) | +0.81% |
| No language — market + reaction only | — | 0.5100 | 0.571 (majority 0.571) | +0.0163 (+1.2) | +0.20% |

## Greedy selection log (validation reaction AUC)

| Model | Step | Valid AUC | Gain | |
|---|---|---|---|---|
| LR | +tone | 0.6357 | +0.1357 | kept |
| LR | +market | 0.6401 | +0.0044 | kept |
| LR | +text | 0.6401 | -0.0001 | stop |
| XGB | +tone | 0.6220 | +0.1220 | kept |
| XGB | +market | 0.6296 | +0.0076 | kept |
| XGB | +lm | 0.6325 | +0.0029 | kept |
| XGB | +fls | 0.6328 | +0.0003 | stop |
