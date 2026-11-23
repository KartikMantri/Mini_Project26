# Phase 4 — Market-Reaction Test (Event Study)

**Status:** ✅ Done on the FinBERT score (2026-11-11; re-run 2026-11-12 after the PARA price
fix). Results:
[`Phase4_finbert_results.md`](Phase4_finbert_results.md). Script:
[`phase4_market_reaction_test.py`](../phase4_market_reaction_test.py).

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
- Immediate top − bottom **+3.94%** (t 13.2): the market reacts strongly to the language.
- Drift top − bottom **+1.62%** (p 0.002), **+1.31%** with controls (p 0.013): a smaller part
  of the signal (29% of the total move) is priced in only over the following weeks, so there
  is evidence of under-reaction.
- A model without language shows no drift (−0.36%, p 0.53).
- Validation years 2012–2017: drift +0.85% (p 0.031).
- Velu's earlier VADER version (past result; code in [`archive/vader/`](../archive/vader/)), run through the same tests on the same calls:
  +1.40% (p 0.059), +1.29% with controls (p 0.099), not significant at 5% on the holdout.
- PARA calls are excluded: its series in `data/prices.csv` is not Paramount's price
  (see `phase1_abnormal_returns.py`).

## Next
→ Phase 5 — Trading Backtest: is the drift large enough to trade after costs? Yes, modestly:
see [`Phase5_Trading_Backtest.md`](Phase5_Trading_Backtest.md).
