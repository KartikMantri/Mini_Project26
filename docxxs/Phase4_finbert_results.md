# Phase 4 (v2) — Market-Reaction Test on the FinBERT Score

Does the market fully price what an earnings call's language reveals? The Phase 3
FinBERT score ([`Phase3_finbert_results.md`](Phase3_finbert_results.md); trained only on the
immediate reaction, always out-of-sample) is compared with the immediate window
(day −1 → +1) and the drift window (day +2 → +60). If the score also predicts the drift,
the market under-reacted to the language.

Method: calls are ranked by score within each calendar quarter; returns are abnormal
(stock minus S&P 500) and winsorised at 1% / 99%; spreads are tested across quarters with
Newey-West (1 lag) t-statistics; regressions use quarter fixed effects with errors clustered
by quarter. These are the same definitions as Velu's VADER Phase 4, so the numbers compare
directly. Regenerate with `python market_reaction_finbert.py`.

## Answer (holdout years 2018–2025)

- **Immediate:** top-20% calls beat bottom-20% calls by **+3.94%** over day −1 → +1 (t 13.2). The market reacts to what the language says.
- **Drift:** they keep beating them by **+1.62%** over day +2 → +60 (t 3.4, p 0.002, 30 quarters).
- **With controls** (immediate reaction, pre-call return and volatility), highest- vs lowest-ranked call: **+1.31%** (t 2.7, p 0.013).
- Of the total top-minus-bottom move, **29%** came after the reaction window.

Over 2012–2025 (54 quarters): drift top − bottom +1.28% (t 4.0, p 0.000); with controls +1.13% (t 3.3, p 0.002).

## FinBERT vs Velu's VADER Phase 4

Velu's scores are `score_reaction_lr` from his `vadar-phase4` branch (his Phase 3 with
the extra `text_delta` input), run through **this** script. His own report gave
holdout drift +1.20% (t 1.8) and +1.15% with controls; small differences below come from
implementation details.

**Each run on its own calls** (ours includes the old NEUTRAL calls):

| Run | Calls | Immediate top − bottom | Drift top − bottom | Drift with controls (3) |
|---|---|---|---|---|
| FinBERT, holdout 2018–2025 | 5,748 | +3.94% (t 13.2) | +1.62%\*\*\* (t 3.4, p 0.002) | +1.31%\*\* (t 2.7, p 0.013) |
| VADER (Velu), holdout 2018–2025 | 4,807 | +3.86% (t 10.1) | +1.40%\* (t 2.0, p 0.059) | +1.29%\* (t 1.7, p 0.099) |
| FinBERT, 2012–2025 | 9,848 | +3.86% (t 19.6) | +1.28%\*\*\* (t 4.0, p 0.000) | +1.13%\*\*\* (t 3.3, p 0.002) |
| VADER (Velu), 2012–2025 | 8,066 | +3.32% (t 12.5) | +1.43%\*\*\* (t 3.0, p 0.004) | +1.36%\*\* (t 2.4, p 0.021) |

**Same calls only** (calls in both runs):

| Run | Calls | Immediate top − bottom | Drift top − bottom | Drift with controls (3) |
|---|---|---|---|---|
| FinBERT, holdout 2018–2025 | 4,807 | +3.90% (t 12.8) | +1.75%\*\*\* (t 3.0, p 0.006) | +1.48%\*\* (t 2.6, p 0.013) |
| VADER (Velu), holdout 2018–2025 | 4,807 | +3.86% (t 10.1) | +1.40%\* (t 2.0, p 0.059) | +1.29%\* (t 1.7, p 0.099) |
| FinBERT, 2012–2025 | 8,056 | +3.85% (t 18.2) | +1.44%\*\*\* (t 3.6, p 0.001) | +1.32%\*\*\* (t 3.2, p 0.002) |
| VADER (Velu), 2012–2025 | 8,056 | +3.33% (t 12.8) | +1.45%\*\*\* (t 3.1, p 0.003) | +1.37%\*\* (t 2.4, p 0.020) |

## Holdout years 2018–2025

### Average abnormal return by score quintile

| Quintile | Calls | Immediate (−1 → +1) | Drift (+2 → +60) |
|---|---|---|---|
| 1 (lowest score) | 1,152 | -1.94% | -0.53% |
| 2 | 1,145 | -0.46% | -0.02% |
| 3 | 1,144 | +0.33% | +0.25% |
| 4 | 1,145 | +1.07% | +0.24% |
| 5 (highest score) | 1,162 | +2.00% | +1.10% |

- **Top − bottom, immediate:** **+3.94%** \*\*\* (t 13.2, p 0.000, 30 quarters)
- **Top − bottom, drift:** **+1.62%** \*\*\* (t 3.4, p 0.002, 30 quarters)
- Language-free model (market + reaction only), drift: **-0.36%**  (t -0.6, p 0.533, 30 quarters)

### Regressions: drift (+2 → +60) on score rank

Coefficient on *score rank* = drift of the highest-ranked call minus the lowest. Quarter fixed effects; errors clustered by quarter.

| Model | Score rank | Immediate reaction | Pre-call return | Pre-call volatility | Calls | Quarters |
|---|---|---|---|---|---|---|
| (1) score only | +1.91%\*\*\* (t 3.6) |  |  |  | 5,748 | 30 |
| (2) + immediate reaction | +1.65%\*\*\* (t 3.1) | +0.053 (t 1.7) |  |  | 5,748 | 30 |
| (3) + pre-call return, volatility | +1.31%\*\* (t 2.7) | +0.056\* (t 1.8) | -0.035 (t -1.1) | +1.110 (t 1.3) | 5,747 | 30 |

Immediate reaction on score rank: +4.80%\*\*\* (t 16.8).

## Validation years 2012–2017

### Average abnormal return by score quintile

| Quintile | Calls | Immediate (−1 → +1) | Drift (+2 → +60) |
|---|---|---|---|
| 1 (lowest score) | 825 | -1.72% | +0.72% |
| 2 | 813 | -0.29% | +0.83% |
| 3 | 818 | +0.29% | +1.52% |
| 4 | 813 | +0.96% | +2.03% |
| 5 (highest score) | 831 | +2.02% | +1.60% |

- **Top − bottom, immediate:** **+3.75%** \*\*\* (t 16.5, p 0.000, 24 quarters)
- **Top − bottom, drift:** **+0.85%** \*\* (t 2.3, p 0.031, 24 quarters)
- Language-free model (market + reaction only), drift: **+0.21%**  (t 0.4, p 0.691, 24 quarters)

### Regressions: drift (+2 → +60) on score rank

Coefficient on *score rank* = drift of the highest-ranked call minus the lowest. Quarter fixed effects; errors clustered by quarter.

| Model | Score rank | Immediate reaction | Pre-call return | Pre-call volatility | Calls | Quarters |
|---|---|---|---|---|---|---|
| (1) score only | +1.50%\*\*\* (t 3.0) |  |  |  | 4,100 | 24 |
| (2) + immediate reaction | +1.27%\*\* (t 2.5) | +0.053 (t 1.4) |  |  | 4,100 | 24 |
| (3) + pre-call return, volatility | +0.85%\* (t 1.8) | +0.050 (t 1.4) | +0.004 (t 0.1) | +1.361\* (t 1.8) | 4,099 | 24 |

Immediate reaction on score rank: +4.46%\*\*\* (t 16.7).

## All years 2012–2025

### Average abnormal return by score quintile

| Quintile | Calls | Immediate (−1 → +1) | Drift (+2 → +60) |
|---|---|---|---|
| 1 (lowest score) | 1,977 | -1.85% | -0.01% |
| 2 | 1,958 | -0.39% | +0.33% |
| 3 | 1,962 | +0.31% | +0.78% |
| 4 | 1,958 | +1.03% | +0.99% |
| 5 (highest score) | 1,993 | +2.01% | +1.31% |

- **Top − bottom, immediate:** **+3.86%** \*\*\* (t 19.6, p 0.000, 54 quarters)
- **Top − bottom, drift:** **+1.28%** \*\*\* (t 4.0, p 0.000, 54 quarters)
- Language-free model (market + reaction only), drift: **-0.11%**  (t -0.3, p 0.791, 54 quarters)

### Regressions: drift (+2 → +60) on score rank

Coefficient on *score rank* = drift of the highest-ranked call minus the lowest. Quarter fixed effects; errors clustered by quarter.

| Model | Score rank | Immediate reaction | Pre-call return | Pre-call volatility | Calls | Quarters |
|---|---|---|---|---|---|---|
| (1) score only | +1.74%\*\*\* (t 4.7) |  |  |  | 9,848 | 54 |
| (2) + immediate reaction | +1.49%\*\*\* (t 3.9) | +0.053\*\* (t 2.2) |  |  | 9,848 | 54 |
| (3) + pre-call return, volatility | +1.13%\*\*\* (t 3.3) | +0.055\*\* (t 2.3) | -0.022 (t -0.9) | +1.211\*\* (t 2.1) | 9,846 | 54 |

Immediate reaction on score rank: +4.66%\*\*\* (t 23.3).

## Robustness: holdout without the pandemic period

Same exclusions as Velu's check. **Post-hoc:** these were chosen after seeing that
2020–2021 were weak years in his run, so they explain the main result; they don't replace it.

| Run | Calls | Immediate top − bottom | Drift top − bottom | Drift with controls (3) |
|---|---|---|---|---|
| Holdout 2018–2025, all calls (main result) | 5,748 | +3.94% (t 13.2) | +1.62%\*\*\* (t 3.4, p 0.002) | +1.31%\*\* (t 2.7, p 0.013) |
| Without the crash window (calls Dec 2019 – Jun 2020) | 5,346 | +4.07% (t 14.4) | +1.50%\*\*\* (t 3.0, p 0.006) | +1.16%\*\* (t 2.3, p 0.030) |
| Without all of 2020 | 4,973 | +4.11% (t 13.6) | +1.50%\*\*\* (t 2.8, p 0.009) | +1.27%\*\* (t 2.3, p 0.028) |
| Without 2020–2021 | 4,195 | +4.44% (t 17.8) | +1.49%\*\*\* (t 3.1, p 0.006) | +0.97%\* (t 1.7, p 0.097) |

## Year by year (2012–2025)

| Year | Calls | Reaction AUC | Immediate spread | Drift spread |
|---|---|---|---|---|
| 2012 | 638 | 0.635 | +4.02% | +1.15% |
| 2013 | 660 | 0.661 | +4.56% | -0.23% |
| 2014 | 683 | 0.616 | +2.79% | -0.38% |
| 2015 | 691 | 0.652 | +3.69% | +0.79% |
| 2016 | 702 | 0.661 | +4.39% | +1.33% |
| 2017 | 726 | 0.630 | +3.04% | +2.44% |
| 2018 | 732 | 0.632 | +4.13% | +0.16% |
| 2019 | 757 | 0.653 | +4.25% | +1.37% |
| 2020 | 775 | 0.589 | +2.89% | +2.41% |
| 2021 | 778 | 0.597 | +2.27% | +1.54% |
| 2022 | 782 | 0.585 | +4.45% | +0.55% |
| 2023 | 782 | 0.618 | +4.23% | +2.77% |
| 2024 | 784 | 0.652 | +4.94% | +2.23% |
| 2025 | 358 | 0.606 | +4.84% | +2.18% |

## Cumulative abnormal return path (holdout, from the close before the call)

| Day | Q1 (lowest) | Q3 | Q5 (highest) | Q5 − Q1 |
|---|---|---|---|---|
| +0 | -1.25% | +0.21% | +1.21% | +2.46% |
| +1 | -2.07% | +0.31% | +2.04% | +4.11% |
| +5 | -1.96% | +0.31% | +2.22% | +4.18% |
| +10 | -2.19% | +0.40% | +2.24% | +4.43% |
| +20 | -2.20% | +0.51% | +2.81% | +5.01% |
| +40 | -2.44% | +0.62% | +2.90% | +5.34% |
| +60 | -2.54% | +0.67% | +3.27% | +5.81% |

Full paths: `data/car_paths_finbert.csv`.

`*` p < 0.10, `**` p < 0.05, `***` p < 0.01.

## Caveats
- Drift windows of calls in the same quarter overlap, so calls are not independent; the
  tests work across quarters and cluster by quarter to account for this.
- The 200 tickers are today's S&P 500 members (survivorship bias); it affects high- and
  low-score groups alike but may flatter long-run returns.
- No trading costs here; Phase 5 tests whether the drift is large enough to trade.
