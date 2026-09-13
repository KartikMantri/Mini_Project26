# Phase 4 — Market-Reaction Test (Event Study)

**Status:** ⬜ Not started

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
