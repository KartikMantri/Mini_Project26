# Phase 1 — Data Collection

**Status:** ✅ Complete
**Full report:** [`Phase1_Data_Collection_Report.pdf`](Phase1_Data_Collection_Report.pdf)
**Flowchart:** [`phase1_flowchart.svg`](phase1_flowchart.svg) / [`.png`](phase1_flowchart.png)
**Script:** [`collect_data.py`](../collect_data.py)

## Goal
One clean table — `master.csv` — with one row per earnings call, its transcript, and a
Good/Bad label describing how the stock performed relative to the market afterward.

## Pipeline

```
tickers.csv ──┬──> Fetch Transcripts (Hugging Face) ──┐
              │                                        ├──> master.csv
              └──> Fetch Prices (yfinance) ──> Label ──┘
```

See the [full flowchart](phase1_flowchart.svg) for every component and formula.

## Data sources

| Data | Source | Notes |
|---|---|---|
| Transcripts | [`kurry/sp500_earnings_transcripts`](https://huggingface.co/datasets/kurry/sp500_earnings_transcripts) (Hugging Face) | 33,362 transcripts, 2005–2025, free |
| Prices | `yfinance` | 45 tickers + `^GSPC` (S&P 500 benchmark) |

## Labeling method

Label = **abnormal return**: the stock's return minus the S&P 500's return, over trading
days **+2 to +60** after the call.

```
stock_ret        = ( P[t+60] / P[t+2] ) − 1
market_ret       = ( M[t+60] / M[t+2] ) − 1
abnormal_return  = stock_ret − market_ret

label = GOOD     if abnormal_return >  +0.02
        BAD      if abnormal_return <  −0.02
        NEUTRAL  otherwise (excluded from master.csv)
```

This was chosen over an EPS-beat/miss label (fully explained in the PDF report, Section 7.1)
because it's fully automatable from price data alone, with no missing-data risk. It also
inherently discounts broad market moves (crashes, COVID, geopolitical shocks) that have
nothing to do with a specific call — see Section 7.4 of the report for the worked example.

## Results (2021-01-01 → 2024-06-30)

| Metric | Value |
|---|---|
| Transcripts fetched | 630 |
| GOOD-labeled calls | 255 |
| BAD-labeled calls | 254 |
| NEUTRAL (excluded) | 121 |
| Rows in `master.csv` | 509 |

## `master.csv` schema

| Column | Type | Description |
|---|---|---|
| `ticker` | string | Stock symbol |
| `call_date` | datetime | Earnings call date |
| `transcript_text` | string | Full call transcript (untouched, feeds Phase 2) |
| `abnormal_return` | float | Stock return minus S&P 500 return, days +2→+60 |
| `label` | string | `GOOD` or `BAD` (target variable for Phase 3) |

## How to reproduce

```bash
pip install -r requirements.txt
python collect_data.py
```

Outputs land in `data/`: `raw_transcripts.csv`, `prices.csv`, `labeled_all.csv`, `master.csv`.

## Known limitations
- Assumes each stock's beta ≈ 1 relative to the market (see report Section 7.4 / 11).
- ±2% threshold was tuned for this specific ticker/date combination.

## Next
→ Phase 2 — Signal Extraction *(not yet started)*
