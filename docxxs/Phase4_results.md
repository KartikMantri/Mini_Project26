# Phase 4 — Market-Reaction Test

Does the market fully price what an earnings call's language reveals?
The Phase 3 score (trained only on the *immediate* reaction, always out-of-sample) is
compared with the immediate window (day −1 → +1) and the drift window (day +2 → +60).
If the score also predicts the drift, the market under-reacted to the language.

Method notes: calls are ranked by score within each calendar quarter; returns are
abnormal (stock minus S&P 500) and winsorized at 1%/99%; spreads are tested across
quarters with Newey-West (1 lag) t-statistics; regressions use quarter fixed
effects with standard errors clustered by quarter. Regenerate with
`python market_reaction.py`.

## Answer (holdout years 2018–2025)

- **Immediate:** top-20% calls beat bottom-20% calls by **+3.86%** in the
  day −1 → +1 window (t = 10.2). The market does react to what the language says.
- **Drift:** they keep beating them by **+1.20%** over day +2 → +60
  (t = 1.8, p = 0.088, 30 quarters).
- **After controlling for the immediate reaction and pre-call momentum/volatility**, the
  score still predicts drift: highest-vs-lowest ranked call +1.15%
  (t = 1.5, p = 0.155).
- Of the total top-minus-bottom move (immediate + drift), **24%** came *after* the reaction window.

Over the **full out-of-sample period 2012–2025** (54 quarters): drift top − bottom
+1.34% (t = 2.9, p = 0.005); with controls +1.30%
(t = 2.2, p = 0.030).

**Verdict:** the market prices most of the language signal immediately. A smaller part drifts in over the following weeks: significant over 2012–2025, but not on the 2018–2025 holdout alone. Evidence of **under-reaction is modest**, not conclusive.

**Pandemic check:** without 2020–2021 calls, the holdout drift is +1.75%
(p = 0.003; with controls p = 0.003). The weaker holdout result comes from that period — see *Robustness* below.

## Holdout years 2018–2025

### Average abnormal return by score quintile

| Quintile | Calls | Immediate (−1 → +1) | Drift (+2 → +60) |
|---|---|---|---|
| 1 (lowest score) | 974 | -1.83% | -0.92% |
| 2 | 961 | -0.05% | -0.17% |
| 3 | 956 | -0.05% | +0.85% |
| 4 | 961 | +0.70% | +0.73% |
| 5 (highest score) | 970 | +2.02% | +0.25% |
| **Top − bottom (Immediate)** | 30 quarters | **+3.86%** (t 10.2, p 0.000) |  |
| **Top − bottom (Drift)** | 30 quarters |  | **+1.20%** (t 1.8, p 0.088) |

Language-free model (market + reaction only), same drift test: top − bottom +0.29% (t 0.5, p 0.644).

### Regressions: drift (+2 → +60) on score rank

Coefficient on *score rank* = drift of the highest-ranked call minus the lowest.

| Model | Score rank | Immediate reaction | Pre-call return | Pre-call volatility | Calls | Quarters |
|---|---|---|---|---|---|---|
| (1) score only | +1.60%** (t 2.3) |  |  |  | 4822 | 30 |
| (2) + immediate reaction | +1.32%* (t 1.8) | +0.067* (t 1.7) |  |  | 4822 | 30 |
| (3) + pre-call return, volatility | +1.15% (t 1.5) | +0.066* (t 1.7) | -0.035 (t -1.0) | +0.347 (t 0.5) | 4819 | 30 |

Immediate reaction on score rank: +4.26%*** (t 11.6).

`*` p < 0.10, `**` p < 0.05, `***` p < 0.01.

## Validation years 2012–2017

### Average abnormal return by score quintile

| Quintile | Calls | Immediate (−1 → +1) | Drift (+2 → +60) |
|---|---|---|---|
| 1 (lowest score) | 660 | -1.02% | +0.86% |
| 2 | 648 | -0.15% | +1.24% |
| 3 | 648 | +0.73% | +1.70% |
| 4 | 648 | +0.53% | +2.13% |
| 5 (highest score) | 655 | +1.60% | +2.40% |
| **Top − bottom (Immediate)** | 24 quarters | **+2.61%** (t 10.5, p 0.000) |  |
| **Top − bottom (Drift)** | 24 quarters |  | **+1.51%** (t 2.5, p 0.022) |

Language-free model (market + reaction only), same drift test: top − bottom +0.28% (t 0.4, p 0.663).

### Regressions: drift (+2 → +60) on score rank

Coefficient on *score rank* = drift of the highest-ranked call minus the lowest.

| Model | Score rank | Immediate reaction | Pre-call return | Pre-call volatility | Calls | Quarters |
|---|---|---|---|---|---|---|
| (1) score only | +1.90%** (t 2.4) |  |  |  | 3259 | 24 |
| (2) + immediate reaction | +1.73%** (t 2.1) | +0.059 (t 1.2) |  |  | 3259 | 24 |
| (3) + pre-call return, volatility | +1.47%* (t 1.7) | +0.046 (t 1.0) | +0.001 (t 0.0) | +1.419 (t 1.7) | 3258 | 24 |

Immediate reaction on score rank: +2.97%*** (t 9.5).

`*` p < 0.10, `**` p < 0.05, `***` p < 0.01.

## All years 2012–2025

### Average abnormal return by score quintile

| Quintile | Calls | Immediate (−1 → +1) | Drift (+2 → +60) |
|---|---|---|---|
| 1 (lowest score) | 1634 | -1.50% | -0.20% |
| 2 | 1609 | -0.09% | +0.40% |
| 3 | 1604 | +0.27% | +1.20% |
| 4 | 1609 | +0.63% | +1.30% |
| 5 (highest score) | 1625 | +1.85% | +1.13% |
| **Top − bottom (Immediate)** | 54 quarters | **+3.30%** (t 12.5, p 0.000) |  |
| **Top − bottom (Drift)** | 54 quarters |  | **+1.34%** (t 2.9, p 0.005) |

Language-free model (market + reaction only), same drift test: top − bottom +0.28% (t 0.6, p 0.535).

### Regressions: drift (+2 → +60) on score rank

Coefficient on *score rank* = drift of the highest-ranked call minus the lowest.

| Model | Score rank | Immediate reaction | Pre-call return | Pre-call volatility | Calls | Quarters |
|---|---|---|---|---|---|---|
| (1) score only | +1.73%*** (t 3.3) |  |  |  | 8081 | 54 |
| (2) + immediate reaction | +1.49%*** (t 2.7) | +0.066** (t 2.2) |  |  | 8081 | 54 |
| (3) + pre-call return, volatility | +1.30%** (t 2.2) | +0.065** (t 2.2) | -0.022 (t -0.8) | +0.682 (t 1.3) | 8077 | 54 |

Immediate reaction on score rank: +3.73%*** (t 14.2).

`*` p < 0.10, `**` p < 0.05, `***` p < 0.01.

## Robustness: holdout without the pandemic period

Top 20% minus bottom 20% of calls by score. *Controls* = drift regression with the
immediate reaction and pre-call return/volatility.

| Sample | Calls | Quarters | Immediate | Drift (+2 → +60) | Drift with controls |
|---|---|---|---|---|---|
| Holdout 2018–2025, all calls (main result) | 4822 | 30 | +3.86% (t 10.2) | +1.20%* (p 0.088) | +1.15% (p 0.155) |
| Without the crash window (calls Dec 2019 – Jun 2020) | 4467 | 28 | +3.92% (t 9.8) | +1.10% (p 0.132) | +1.03% (p 0.211) |
| Without all of 2020 | 4144 | 26 | +4.05% (t 9.9) | +1.44%** (p 0.026) | +1.65%*** (p 0.007) |
| Without 2020–2021 | 3493 | 22 | +4.54% (t 15.8) | +1.75%*** (p 0.003) | +1.81%*** (p 0.003) |

- Removing only the crash months does not change the result; removing all of 2020, or
  2020–2021, makes the drift clearly significant. The weak stretch is the whole
  stimulus / meme-stock / retail-trading period, not just the March 2020 crash.
- **Caveat:** these exclusions were chosen *after* seeing that 2020–2021 were weak years.
  Dropping weak years always improves a result, so this is a robustness check that
  explains the holdout number; the main result above stays the headline.

## Year by year (2012–2025)

*Reaction AUC*: how well the score predicted the immediate reaction that year (0.5 =
guessing). Spreads are top 20% minus bottom 20% of calls by score.

| Year | Calls | Reaction AUC | Immediate spread | Drift spread |
|---|---|---|---|---|
| 2012 | 526 | 0.591 | +2.70% | +2.38% |
| 2013 | 526 | 0.597 | +1.92% | -0.36% |
| 2014 | 534 | 0.628 | +2.69% | +0.88% |
| 2015 | 548 | 0.599 | +2.59% | +0.04% |
| 2016 | 557 | 0.608 | +3.65% | +0.69% |
| 2017 | 568 | 0.584 | +2.23% | +5.42% |
| 2018 | 597 | 0.625 | +3.76% | +0.66% |
| 2019 | 600 | 0.645 | +4.32% | +3.69% |
| 2020 ⚠️ pandemic | 678 | 0.580 | +2.72% | -0.42% |
| 2021 ⚠️ pandemic | 651 | 0.551 | +1.41% | -0.14% |
| 2022 | 659 | 0.596 | +4.21% | +0.40% |
| 2023 | 674 | 0.627 | +4.79% | +1.57% |
| 2024 | 660 | 0.616 | +5.34% | +2.62% |
| 2025 | 303 | 0.597 | +4.40% | +1.47% |

In 2020–2021 the score still predicted the reaction better than chance (AUC 0.55–0.58), but less sharply than in other years (median 0.60), and the later drift disappeared (-0.28% on average vs. +1.62%). Prices were
driven mainly by market-wide forces (lockdowns, stimulus, meme-stock trading) rather
than company-specific news, so the language signal weakened.


## Cumulative abnormal return path (holdout, from close before the call)

| Day | Q1 (lowest) | Q3 | Q5 (highest) | Q5 − Q1 |
|---|---|---|---|---|
| +0 | -1.19% | -0.01% | +1.14% | +2.33% |
| +1 | -1.92% | -0.06% | +2.10% | +4.02% |
| +5 | -1.91% | +0.08% | +2.35% | +4.26% |
| +10 | -2.08% | +0.30% | +2.13% | +4.21% |
| +20 | -1.88% | +0.66% | +2.41% | +4.29% |
| +40 | -2.22% | +0.39% | +2.26% | +4.48% |
| +60 | -2.73% | +0.92% | +2.43% | +5.16% |

Full paths for every day and quintile: `data/car_paths.csv`.

## Caveats
- Drift windows of calls in the same quarter overlap, so calls are not independent;
  the tests account for this by working across quarters and clustering by quarter.
- The 200 tickers are today's S&P 500 members (survivorship bias), which may flatter
  long-run returns but affects both high- and low-score groups.
- No trading costs here; Phase 5 tests whether the drift is large enough to trade.