# Phase 4 — Market-Reaction Test (Event Study)

**Status:** ✅ Done on the FinBERT score (2026-11-11). Results:
[`Phase4_finbert_results.md`](Phase4_finbert_results.md). Script:
[`market_reaction_finbert.py`](../market_reaction_finbert.py).

## Goal
Check whether the market fully prices what an earnings call's language reveals. This is the
scientific core of the project.

## Method
1. **Score:** the Phase 3 FinBERT model's out-of-sample probability that the stock reacts up
   (`data/predictions_finbert.csv`).
2. **Immediate window** (day −1 → +1): does the market react to the language at all?
3. **Drift window** (day +2 → +60): does the score keep predicting returns afterwards?
4. Calls are ranked by score within each calendar quarter; returns are abnormal (stock minus
   S&P 500), winsorised at 1% / 99%.
5. **Tests:** top-20% minus bottom-20% spread per quarter, with Newey-West t across quarters.
   Drift regressions on the score rank with quarter fixed effects and errors clustered by
   quarter, controlling for the immediate reaction and the pre-call return/volatility.
6. ★ **The make-or-break test:** if the score predicts the drift after these controls, the
   market under-reacted (an inefficiency). If drift ≈ 0, the market is efficient (also a valid
   finding).

## Result (holdout 2018–2025)
- Immediate top − bottom **+3.93%** (t 13.5): the market reacts strongly to the language.
- Drift top − bottom **+1.55%** (p 0.002), **+1.27%** with controls (p 0.018): a smaller part
  of the signal is priced in only over the following weeks, so there is evidence of
  under-reaction.
- A model without language shows no drift (+0.13%, p 0.78).
- Velu's VADER version (branch `vadar-phase4`), run through the same tests on the same calls:
  +1.26% (p 0.075), +1.12% with controls (p 0.147), not significant on the holdout.

## Next
→ Phase 5 — Trading Backtest: is the drift large enough to trade after costs?
