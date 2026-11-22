# Phase 6 — Dashboard & Report

**Status:** ✅ Dashboard done (2026-11-22). Strategy report:
[`Strategy_Report.pdf`](Strategy_Report.pdf).

## Run it
```bash
pip install -r requirements.txt
streamlit run dashboard.py
```
It reads only files committed in `data/`, so no raw transcripts or `master.csv` are needed.

## Pages
| Page | What it shows |
|---|---|
| Overview | Headline numbers (reaction AUC, drift gap, strategy return, alpha), pipeline, growth of 100 |
| Call explorer | Any call: score, percentile vs the past year's calls, trade signal, language profile, price path vs the S&P 500, the Strategy A trade |
| Language | Any of the 18 features: GOOD vs BAD distribution (pre-2018 only) with Welch t-test and effect size; correlation heatmap |
| Classifier | Out-of-sample reaction AUC by year; model comparison; returns by score decile |
| Market reaction | Quintile returns, cumulative abnormal return paths, drift gap per quarter, fixed-effects regressions; holdout, validation or all years |
| Strategies | Strategies A–G, no-language and VADER versions, buy every call and the S&P 500: growth of 100, drawdowns, risk vs return, scorecard with beta and alpha, year by year |
| Method | Rules, limitations, links to the reports |

Headline numbers are recomputed from the data files when the app starts. The theme is set in
[`.streamlit/config.toml`](../.streamlit/config.toml).

## The three numbers that define success
1. **Classifier AUC:** 0.611 on the 2018–2025 holdout (VADER version 0.588).
2. **Drift predictability:** top − bottom quintile +1.62% over days +2 → +60 (p 0.002),
   +1.31% with controls.
3. **Backtest:** market-neutral +6.9%/yr after costs (Sharpe 0.74); tilted 2:1 +12.9%/yr
   (Sharpe 0.95, alpha +5.6%/yr, p 0.033).
