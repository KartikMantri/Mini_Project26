# Phase 5 — Trading Backtest

**Status:** ✅ Done on the FinBERT score (2026-11-12). Script:
[`backtest_finbert.py`](../backtest_finbert.py). Full tables:
[`Phase5_finbert_results.md`](Phase5_finbert_results.md). Chart:
[`phase5_equity_curve.png`](phase5_equity_curve.png).

## Goal
Phase 4 showed that calls the model likes keep beating calls it dislikes for about three months.
Phase 5 asks whether an investor could have made money from that, with realistic timing and
trading costs.

## Method
1. **Signal, no look-ahead.** At each call, compare its Phase 3 score with the scores of all
   calls in the previous 365 days. Top 20% → **long**, bottom 20% → **short**, rest → no trade.
   Trading starts once 200 past calls exist (April 2012). Phase 4 ranked within each quarter,
   which uses calls later in the same quarter; a trader cannot do that.
2. **Timing:** enter at the close of day +2, exit at the close of day +60 (Phase 4's drift
   window). A stock whose prices stop early exits at its last price.
3. **Portfolio:** every trading day, equal weight across open longs and across open shorts;
   daily return = long leg − short leg ($1 long and $1 short per $1 of capital). If a leg is
   empty, the S&P 500 stands in for it. On average 36 longs and 36 shorts are open.
4. **Costs:** 10 bps (0.10%) per buy or sell, charged on the weight traded, plus 0.5% a year to
   borrow the shorted shares. A sensitivity table covers 0–30 bps and the break-even cost.
5. **Measures:** annual return (daily mean × 252), volatility (daily std × √252), Sharpe ratio
   (return ÷ volatility), max drawdown, hit rate, Newey-West t on daily returns (10 lags).
6. **Benchmarks:** S&P 500 buy-and-hold, long-only top 20% minus the S&P 500, the no-language
   score, and Velu's VADER score under the same rules.
7. **Nothing was tuned on the backtest:** the 20% cut-off and the +2 → +60 hold come from
   Phase 4; costs are assumptions.

## Result (holdout 2018 – May 2025)

| Strategy | Annual return | Volatility | Sharpe | Max drawdown | p |
|---|---|---|---|---|---|
| **FinBERT long-short, after costs** | **+6.9%** | 9.3% | **0.74** | −14.1% | 0.032 |
| FinBERT long-short, before costs | +9.4% | 9.3% | 1.01 | −13.0% | 0.004 |
| No-language score | −4.7% | 10.8% | −0.43 | −43.7% | 0.23 |
| Velu's VADER score | +5.9% | 10.5% | 0.56 | −20.4% | 0.11 |
| S&P 500 buy-and-hold | +12.9% | 20.2% | 0.64 | −33.9% | — |

- Profitable after costs and significant at 5%. Beta to the S&P 500 is 0.10: a separate source
  of return, not a bet on the market.
- Break-even cost 46 bps per trade, against the 10 bps assumed.
- 2,315 trades, 54% profitable after costs, average +0.86% per trade after costs.
- The language matters: the no-language score loses money.

## Caveats
- 2020 alone gave +24%; without it the holdout averages about +4.4% a year.
- Validation 2012–2017 is weaker: +3.1% a year after costs, Sharpe 0.50, not significant
  (p 0.24); the long side made the money and the short side lost.
- Survivorship bias: the universe is today's 200 tickers. Trades fill at the closing price with
  no market impact; rebalancing between equal weights is not charged.
- The old [`backtest.py`](../backtest.py) is superseded: it compounded thousands of
  overlapping trades as if they ran one after another.

## Next
→ Phase 6 — Dashboard & Report
