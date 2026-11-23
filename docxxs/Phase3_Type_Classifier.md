# Phase 3 — Type Classifier (Good / Bad score)

**Status:** ✅ Complete. Results: [`Phase3_finbert_results.md`](Phase3_finbert_results.md).
Script: [`phase3_train_classifier.py`](../phase3_train_classifier.py).

## Goal
One score per call, between 0 and 1, that says how likely the firm is a GOOD type, computed
only from information available on the day of the call.

## Input
- `data/features_finbert.csv` (Phase 2): the 18 language features.
- `data/calls_all.csv` and `data/prices.csv` (Phase 1): returns, used for targets and the two
  market inputs.

## Procedure
1. **Target: the immediate reaction.** The model learns whether the stock beat the S&P 500 from
   day −1 to +1. That reaction is mostly caused by the call, so it is a cleaner teacher than the
   3-month GOOD/BAD label (training on GOOD/BAD directly gave AUC 0.511 vs 0.537). The model
   also never sees the 3-month outcome, so if its score predicts the drift in Phase 4, the market
   must have been slow.
2. **Inputs chosen on validation years only.** Feature groups are added one at a time and kept
   only if the 2012–2017 validation AUC rises by at least 0.002. Two groups survived:
   - **FinBERT tone (8):** script, executive-answer and analyst tone, negative share, worst-10%
     tone, tone spread, tone gap, delta tone
   - **Market (2):** pre-call abnormal return and pre-call volatility

   Forward-looking, dictionary, style, change-since-last-call and a word-frequency (TF-IDF)
   text model added nothing.
3. **Cleaning, from training rows only:** extremes capped at the 1st/99th percentile, blanks
   filled with the median, inputs standardised to z-scores.
4. **Model: logistic regression** (L2 penalty, C = 0.05). It is simple, it overfits less on noisy
   returns, and its weights are readable. XGBoost was tried: AUC 0.618 vs 0.611, no real gain.
5. **Walk-forward training.** Every year from 2010 to 2025 a fresh model is trained only on
   earlier calls (with a 7-day embargo) and scores the next year, so every score is
   out-of-sample.
6. **Check:** a single fixed split (train before 2022, test 2022–2025) gives the same AUC (0.611).

## What it learned (2024 model)
Largest weights: analyst tone +0.287, delta tone +0.247, script tone +0.136. A very mixed call
(high tone spread) and a stock that already ran up before the call both lower the score. The
model trusts the signals that are hardest to fake: the analysts and the change in tone.

## Example
Starbucks, 30 January 2024: delta tone was −3.04 standard deviations; with the other inputs the
total was −0.764, so score = 1 / (1 + e^0.764) = **0.317**, one of the lowest of that quarter.

## Results (holdout 2018–2025)
| Check | AUC |
|---|---|
| Reaction, holdout 2018–2025 | **0.611** |
| Reaction, validation 2012–2017 | 0.640 |
| Earlier VADER-feature version (archived) | 0.588 |
| XGBoost instead | 0.618 |
| Fixed split 2022–2025 | 0.611 |

AUC = chance a random up-reacting call is ranked above a random down-reacting one (0.5 = coin flip).

## How to run
```bash
python phase3_train_classifier.py                 # uses the committed text-score cache
python phase3_train_classifier.py --rebuild-text  # needs local data/calls_all.csv
```

## Output
`data/predictions_finbert.csv`: one out-of-sample score per call, 2012–2025 (`score_final`, plus
each comparison model's score).

## Next
→ [Phase 4 — Market-Reaction Test](Phase4_Market_Reaction_Test.md)
