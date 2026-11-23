# Decision: Scaling Up Training Data
**Decision taken by:** Velu  
**Status:** ✅ Implemented & Verified (10,493 calls collected and features extracted)

---

## The Problem

The current dataset (`features.csv`) contains only **509 instances** (255 GOOD / 254 BAD), with
~504 usable for training after a date-based train/test split. This is **too few** to reliably
train a classifier — especially XGBoost, which benefits from thousands of examples to learn
interaction effects across 17 features without overfitting.

**Consequences of training on ~504 rows:**
- High variance in cross-validation scores (results change significantly across folds)
- Risk of overfitting — the model memorises noise instead of learning real patterns
- Feature-importance estimates become unstable and unreliable
- Phase 4 & 5 (market-reaction test, backtest) inherit weak/noisy predictions

---

## What We Need

More earnings call transcripts with:
1. **Same structure** — full-text transcripts with identifiable Prepared Remarks / Q&A sections
2. **Associated stock prices** — so we can compute the same abnormal-return GOOD/BAD labels
3. **Ideally S&P 500 / large-cap US companies** — to match our existing data distribution

**Target:** scale from ~509 to **2,000–5,000+ labeled instances**.

---

## Recommended Data Sources

### Strategy A — Expand the Existing Pipeline (Easiest, Highest Compatibility)

Our `collect_data.py` already pulls from `kurry/sp500_earnings_transcripts` on Hugging Face,
which contains **33,000+ transcripts** across 685 companies (2005–2025). We're currently using
only **45 tickers x 14 quarters (2021-01 to 2024-06)**, which is a tiny fraction.

**Two levers to scale up without changing code:**

| Lever | How | Estimated New Rows |
|---|---|---|
| **Extend the date window** | Widen `--start 2018-01-01 --end 2024-06-30` (or even 2015) | +300-500 per direction |
| **Add more tickers** | Expand `tickers.csv` from 45 to 100-200 S&P 500 tickers | 2x-4x current count |
| **Both combined** | Wider window + more tickers | **2,000-5,000+ rows** |

> This is the **recommended first move** -- it uses the same pipeline, same labeling logic,
> same feature extraction, so there is zero schema-mismatch risk. Just edit `tickers.csv`
> and change the `--start`/`--end` arguments.

### Strategy B — Additional Hugging Face Datasets

| Dataset | Size | What It Offers |
|---|---|---|
| **Bose345/sp500_earnings_transcripts** | 33,000+ transcripts, 685 companies | Same S&P 500 universe, full raw text + structured speaker-by-speaker content |
| **Rogersurf/earnings-call-transcripts** | ~9,000+ transcripts | Cleaned Parquet format, ticker/date/quarter metadata, ready for NLP |
| **Aiera/aiera-transcript-sentiment** | Pre-segmented and labeled | Comes with sentiment labels -- could be used for transfer learning |

### Strategy C — Kaggle Datasets

| Dataset | Notes |
|---|---|
| **Earnings Call Transcripts - NASDAQ (2016-2020)** | Select NASDAQ companies, useful for cross-market validation |
| **Scraped Motley Fool Earnings Call Transcripts** | Dates, exchanges, quarters, tickers + raw text |
| **Top 50 Companies Earnings Call Transcripts (AlphaSpread)** | Major companies, good for quick expansion |
| **Earning Call Transcripts US Companies (Seeking Alpha)** | Broad coverage, good for LLM/NLP pipelines |

### Strategy D — Academic / Institutional Sources

| Source | Access | Notes |
|---|---|---|
| **DataverseNL** (Tilburg University) | Free, DOI: 10.34894/tje0d0 | Earnings transcripts paired with stock prices and sector indices |
| **WRDS (Wharton)** - Capital IQ Transcripts | University library access required | Gold-standard academic dataset, massive coverage |
| **Financial Modeling Prep API** | Free tier available | Programmatic access, can pull transcripts for specific tickers |
| **Finnhub API** | Free developer tier | Earnings call transcripts endpoint |

### Strategy E — Data Augmentation (If External Data Is Still Insufficient)

| Technique | How It Works | Risk Level |
|---|---|---|
| **SMOTE / ADASYN** | Synthetic oversampling in feature space -- generates new rows by interpolating between existing feature vectors | Low |
| **Back-Translation** | Translate transcript text to French/German then back to English, re-extract features | Medium |
| **LLM Paraphrasing** | Prompt GPT/Claude to rephrase transcript sections while preserving meaning, then re-extract features | Medium |
| **FinBERT Contextual Substitution** | Mask non-critical words in transcripts, use FinBERT to fill in contextually appropriate synonyms | Low-Medium |
| **Noise Injection** | Add small Gaussian noise to numeric features in features.csv | Low |

**Warning:** Text-level augmentation (back-translation, LLM paraphrasing) should **never alter
numbers, percentages, ticker symbols, or directional words** (grew/shrank, profit/loss). Always
validate augmented samples via semantic similarity (cosine > 0.85) before including them.

---

## Recommended Action Plan

```
Priority 1 (immediate, no new code needed):
  -> Expand tickers.csv from 45 to 150+ S&P 500 tickers
  -> Extend date window: --start 2017-01-01 --end 2024-06-30
  -> Re-run: python collect_data.py --force
  -> Re-run: python extract_features.py
  -> Expected result: 2,000-4,000+ labeled instances

Priority 2 (if Priority 1 still < 2,000 rows):
  -> Pull additional transcripts from Bose345 or Rogersurf HF datasets
  -> Merge with existing master.csv (deduplicate by ticker+date)
  -> Re-label via yfinance prices, re-extract features

Priority 3 (complementary, can run in parallel):
  -> Apply SMOTE on features.csv for the training split only
  -> Never apply augmentation to test/validation data
```

---

## What This Changes Downstream

- **Phase 3 (Classifier):** More training data means more stable cross-validation, better
  generalization, more trustworthy feature importances
- **Phase 4 (Market Reaction):** Larger sample means more statistical power for t-tests
- **Phase 5 (Backtest):** More predictions means smoother equity curve, more realistic Sharpe

---

## Implementation Outcome (2026-10-19)

- **Priority 1 Executed Successfully:**
  - `tickers.csv` expanded to **200 S&P 500 tickers**.
  - Date window widened from **2005-01-01 to 2025-12-31**.
  - `collect_data.py` produced `data/master.csv` with **10,493 labeled transcripts** (5,638 GOOD, 4,855 BAD).
  - Parallel feature extraction via `extract_features_fast.py` (7 workers) completed successfully, outputting **10,493 rows and 35 features** to `data/features_expanded.csv`.
  - Feature distributions and null-value analyses documented in [`features.md`](features.md).
