# Phase 4 — Market-Reaction Test (Event Study)

**Status:** ✅ Complete — results in [`Phase4_results.md`](Phase4_results.md)

## Result in one paragraph
The Phase 3 score lines up strongly with the immediate reaction: the highest-scored 20% of
calls beat the lowest 20% by +3.9% over day −1 → +1 (t ≈ 10, 2018–2025). A smaller part keeps
coming afterwards: +1.3% over day +2 → +60 across 2012–2025 (t = 2.9, p = 0.005; p = 0.03 after
controlling for the immediate reaction and pre-call momentum/volatility). On the 2018–2025
holdout alone the drift is +1.2% (p = 0.09), so the evidence of under-reaction is **modest,
not conclusive**. The drift comes mostly from low-scored calls, which keep falling after the
reaction; a language-free model shows no drift (+0.3%, p = 0.64).

**Pandemic check (robustness):** the weak holdout number comes from 2020–2021. In those years
the score still predicted the reaction better than chance (AUC 0.55–0.58 vs. a median of 0.60
in other years) but the drift vanished. Without 2020–2021 calls, the holdout drift is +1.75%
(p = 0.003, also with controls). These exclusions were chosen after seeing the yearly numbers,
so they explain the holdout result rather than replace it.

## Method (as built)
- Calls ranked by score within each calendar quarter; quintiles 1 (lowest) to 5 (highest).
- Abnormal returns (stock − S&P 500), winsorized at 1%/99%; event day = call date.
- Top-minus-bottom spreads tested across quarters with Newey-West t-statistics.
- Regressions of drift on score rank + immediate reaction + pre-call return/volatility,
  quarter fixed effects, standard errors clustered by quarter.
- Outputs: `data/market_reaction.csv` (per call), `data/car_paths.csv` (day-by-day average
  path per quintile, for the dashboard), `docxxs/Phase4_results.md`.

## Original plan

## Goal
Check whether the market reacts to the signal the way it "should" — the scientific core of
the project.

## Plan
1. **Immediate window** (day 0 → +1): how much did the market price instantly?
2. **Drift window** (day +2 → +20): how much did it keep moving afterward?
3. Compare GOOD vs BAD groups' average abnormal return in each window, with a t-test.
4. ★ **The make-or-break test:** does the signal predict the drift? If yes → the market
   under-reacted (an inefficiency). If drift ≈ 0 → market is efficient (also a valid finding).

## Input / Output
- **Input:** `abnormal_return` (Phase 1) + predicted `label` (Phase 3).
- **Output:** immediate-vs-drift comparison table + significance results.

## Next
→ Phase 5 — Trading Backtest
