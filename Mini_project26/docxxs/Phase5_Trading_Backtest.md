# Phase 5 — Trading Backtest

**Status:** ⬜ Not started

## Goal
Turn the finding into a tangible strategy with real numbers.

## Plan
1. Rule: predicted **GOOD → go long**, **BAD → go short**, hold ~20 trading days.
2. Confidence filter: only trade when `P(Good) > 0.65` or `< 0.35`.
3. Subtract transaction cost (~0.1% per trade).
4. Compute total return, hit-rate, Sharpe ratio; plot an equity curve.
5. Compare against a baseline ("always long").

## Input / Output
- **Input:** predicted `label` + `confidence` (Phase 3), price data (Phase 1).
- **Output:** backtest metrics + equity-curve chart.

## Next
→ Phase 6 — Dashboard & Report
