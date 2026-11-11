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

- **Immediate:** top-20% calls beat bottom-20% calls by **+3.93%** over day −1 → +1 (t 13.5). The market reacts to what the language says.
- **Drift:** they keep beating them by **+1.55%** over day +2 → +60 (t 3.4, p 0.002, 30 quarters).
- **With controls** (immediate reaction, pre-call return and volatility), highest- vs lowest-ranked call: **+1.27%** (t 2.5, p 0.018).
- Of the total top-minus-bottom move, **28%** came after the reaction window.

Over 2012–2025 (54 quarters): drift top − bottom +1.17% (t 3.7, p 0.000); with controls +1.10% (t 3.1, p 0.003).

## FinBERT vs Velu's VADER Phase 4

Velu's scores are `score_reaction_lr` from his `vadar-phase4` branch (his Phase 3 with
the extra `text_delta` input), run through **this** script. His own report gave
holdout drift +1.20% (t 1.8) and +1.15% with controls; small differences below come from
implementation details.

**Each run on its own calls** (ours includes the old NEUTRAL calls):

| Run | Calls | Immediate top − bottom | Drift top − bottom | Drift with controls (3) |
|---|---|---|---|---|
| FinBERT, holdout 2018–2025 | 5,766 | +3.93% (t 13.5) | +1.55%\*\*\* (t 3.4, p 0.002) | +1.27%\*\* (t 2.5, p 0.018) |
| VADER (Velu), holdout 2018–2025 | 4,822 | +3.86% (t 10.2) | +1.26%\* (t 1.8, p 0.075) | +1.12% (t 1.5, p 0.147) |
| FinBERT, 2012–2025 | 9,866 | +3.83% (t 19.4) | +1.17%\*\*\* (t 3.7, p 0.000) | +1.10%\*\*\* (t 3.1, p 0.003) |
| VADER (Velu), 2012–2025 | 8,081 | +3.32% (t 12.6) | +1.35%\*\*\* (t 2.9, p 0.005) | +1.27%\*\* (t 2.2, p 0.031) |

**Same calls only** (calls in both runs):

| Run | Calls | Immediate top − bottom | Drift top − bottom | Drift with controls (3) |
|---|---|---|---|---|
| FinBERT, holdout 2018–2025 | 4,822 | +3.90% (t 13.5) | +1.75%\*\*\* (t 3.0, p 0.005) | +1.41%\*\* (t 2.4, p 0.021) |
| VADER (Velu), holdout 2018–2025 | 4,822 | +3.86% (t 10.2) | +1.26%\* (t 1.8, p 0.075) | +1.12% (t 1.5, p 0.147) |
| FinBERT, 2012–2025 | 8,071 | +3.87% (t 18.8) | +1.42%\*\*\* (t 3.5, p 0.001) | +1.26%\*\*\* (t 3.0, p 0.004) |
| VADER (Velu), 2012–2025 | 8,071 | +3.33% (t 12.9) | +1.37%\*\*\* (t 3.0, p 0.004) | +1.28%\*\* (t 2.2, p 0.029) |

## Holdout years 2018–2025

### Average abnormal return by score quintile

| Quintile | Calls | Immediate (−1 → +1) | Drift (+2 → +60) |
|---|---|---|---|
| 1 (lowest score) | 1,160 | -1.94% | -0.42% |
| 2 | 1,148 | -0.48% | -0.25% |
| 3 | 1,144 | +0.29% | +0.26% |
| 4 | 1,148 | +1.14% | +0.11% |
| 5 (highest score) | 1,166 | +2.00% | +1.14% |

- **Top − bottom, immediate:** **+3.93%** \*\*\* (t 13.5, p 0.000, 30 quarters)
- **Top − bottom, drift:** **+1.55%** \*\*\* (t 3.4, p 0.002, 30 quarters)
- Language-free model (market + reaction only), drift: **+0.13%**  (t 0.3, p 0.784, 30 quarters)

### Regressions: drift (+2 → +60) on score rank

Coefficient on *score rank* = drift of the highest-ranked call minus the lowest. Quarter fixed effects; errors clustered by quarter.

| Model | Score rank | Immediate reaction | Pre-call return | Pre-call volatility | Calls | Quarters |
|---|---|---|---|---|---|---|
| (1) score only | +1.78%\*\*\* (t 3.4) |  |  |  | 5,766 | 30 |
| (2) + immediate reaction | +1.52%\*\*\* (t 2.8) | +0.054 (t 1.7) |  |  | 5,766 | 30 |
| (3) + pre-call return, volatility | +1.27%\*\* (t 2.5) | +0.056\* (t 1.8) | -0.030 (t -0.9) | +0.891 (t 1.1) | 5,763 | 30 |

Immediate reaction on score rank: +4.84%\*\*\* (t 17.0).

## Validation years 2012–2017

### Average abnormal return by score quintile

| Quintile | Calls | Immediate (−1 → +1) | Drift (+2 → +60) |
|---|---|---|---|
| 1 (lowest score) | 825 | -1.70% | +0.77% |
| 2 | 813 | -0.36% | +0.87% |
| 3 | 818 | +0.29% | +1.57% |
| 4 | 813 | +1.02% | +1.97% |
| 5 (highest score) | 831 | +2.01% | +1.50% |

- **Top − bottom, immediate:** **+3.71%** \*\*\* (t 15.3, p 0.000, 24 quarters)
- **Top − bottom, drift:** **+0.70%** \* (t 1.9, p 0.072, 24 quarters)
- Language-free model (market + reaction only), drift: **+0.61%**  (t 1.2, p 0.245, 24 quarters)

### Regressions: drift (+2 → +60) on score rank

Coefficient on *score rank* = drift of the highest-ranked call minus the lowest. Quarter fixed effects; errors clustered by quarter.

| Model | Score rank | Immediate reaction | Pre-call return | Pre-call volatility | Calls | Quarters |
|---|---|---|---|---|---|---|
| (1) score only | +1.38%\*\*\* (t 2.9) |  |  |  | 4,100 | 24 |
| (2) + immediate reaction | +1.13%\*\* (t 2.3) | +0.056 (t 1.5) |  |  | 4,100 | 24 |
| (3) + pre-call return, volatility | +0.82%\* (t 1.7) | +0.051 (t 1.4) | +0.004 (t 0.1) | +1.369\* (t 1.8) | 4,099 | 24 |

Immediate reaction on score rank: +4.44%\*\*\* (t 16.9).

## All years 2012–2025

### Average abnormal return by score quintile

| Quintile | Calls | Immediate (−1 → +1) | Drift (+2 → +60) |
|---|---|---|---|
| 1 (lowest score) | 1,985 | -1.84% | +0.08% |
| 2 | 1,961 | -0.43% | +0.21% |
| 3 | 1,962 | +0.29% | +0.80% |
| 4 | 1,961 | +1.09% | +0.88% |
| 5 (highest score) | 1,997 | +2.00% | +1.29% |

- **Top − bottom, immediate:** **+3.83%** \*\*\* (t 19.4, p 0.000, 54 quarters)
- **Top − bottom, drift:** **+1.17%** \*\*\* (t 3.7, p 0.000, 54 quarters)
- Language-free model (market + reaction only), drift: **+0.35%**  (t 1.0, p 0.339, 54 quarters)

### Regressions: drift (+2 → +60) on score rank

Coefficient on *score rank* = drift of the highest-ranked call minus the lowest. Quarter fixed effects; errors clustered by quarter.

| Model | Score rank | Immediate reaction | Pre-call return | Pre-call volatility | Calls | Quarters |
|---|---|---|---|---|---|---|
| (1) score only | +1.61%\*\*\* (t 4.5) |  |  |  | 9,866 | 54 |
| (2) + immediate reaction | +1.36%\*\*\* (t 3.6) | +0.055\*\* (t 2.2) |  |  | 9,866 | 54 |
| (3) + pre-call return, volatility | +1.10%\*\*\* (t 3.1) | +0.055\*\* (t 2.3) | -0.019 (t -0.8) | +1.083\* (t 1.9) | 9,862 | 54 |

Immediate reaction on score rank: +4.67%\*\*\* (t 23.4).

## Robustness: holdout without the pandemic period

Same exclusions as Velu's check. **Post-hoc:** these were chosen after seeing that
2020–2021 were weak years in his run, so they explain the main result; they don't replace it.

| Run | Calls | Immediate top − bottom | Drift top − bottom | Drift with controls (3) |
|---|---|---|---|---|
| Holdout 2018–2025, all calls (main result) | 5,766 | +3.93% (t 13.5) | +1.55%\*\*\* (t 3.4, p 0.002) | +1.27%\*\* (t 2.5, p 0.018) |
| Without the crash window (calls Dec 2019 – Jun 2020) | 5,364 | +4.06% (t 14.8) | +1.37%\*\*\* (t 2.9, p 0.008) | +1.06%\*\* (t 2.1, p 0.045) |
| Without all of 2020 | 4,991 | +4.08% (t 13.7) | +1.45%\*\*\* (t 2.8, p 0.009) | +1.19%\*\* (t 2.2, p 0.037) |
| Without 2020–2021 | 4,209 | +4.41% (t 18.4) | +1.44%\*\*\* (t 3.2, p 0.005) | +0.86% (t 1.6, p 0.132) |

## Year by year (2012–2025)

| Year | Calls | Reaction AUC | Immediate spread | Drift spread |
|---|---|---|---|---|
| 2012 | 638 | 0.634 | +3.95% | +0.93% |
| 2013 | 660 | 0.661 | +4.63% | -0.33% |
| 2014 | 683 | 0.613 | +2.81% | -0.72% |
| 2015 | 691 | 0.656 | +3.59% | +1.03% |
| 2016 | 702 | 0.659 | +4.36% | +0.75% |
| 2017 | 726 | 0.632 | +2.93% | +2.54% |
| 2018 | 732 | 0.633 | +4.02% | -0.07% |
| 2019 | 757 | 0.653 | +4.33% | +1.34% |
| 2020 | 775 | 0.591 | +3.00% | +2.23% |
| 2021 | 782 | 0.600 | +2.23% | +1.49% |
| 2022 | 786 | 0.590 | +4.16% | +0.82% |
| 2023 | 786 | 0.620 | +4.34% | +2.88% |
| 2024 | 788 | 0.652 | +5.11% | +1.92% |
| 2025 | 360 | 0.606 | +4.59% | +2.02% |

## Cumulative abnormal return path (holdout, from the close before the call)

| Day | Q1 (lowest) | Q3 | Q5 (highest) | Q5 − Q1 |
|---|---|---|---|---|
| +0 | -1.23% | +0.22% | +1.22% | +2.45% |
| +1 | -2.06% | +0.27% | +2.04% | +4.09% |
| +5 | -1.95% | +0.22% | +2.19% | +4.14% |
| +10 | -2.16% | +0.31% | +2.20% | +4.36% |
| +20 | -2.08% | +0.41% | +2.76% | +4.85% |
| +40 | -2.32% | +0.57% | +2.77% | +5.09% |
| +60 | -2.43% | +0.56% | +3.22% | +5.65% |

Full paths: `data/car_paths_finbert.csv`.

`*` p < 0.10, `**` p < 0.05, `***` p < 0.01.

## Caveats
- Drift windows of calls in the same quarter overlap, so calls are not independent; the
  tests work across quarters and cluster by quarter to account for this.
- The 200 tickers are today's S&P 500 members (survivorship bias); it affects high- and
  low-score groups alike but may flatter long-run returns.
- No trading costs here; Phase 5 tests whether the drift is large enough to trade.
