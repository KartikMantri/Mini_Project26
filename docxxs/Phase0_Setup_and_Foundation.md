# Phase 0 — Setup & Foundation

**Status:** ✅ Complete

## Goal
A clean environment and a decided scope, so no time is lost later.

## What was done

1. **Environment.** Python environment with dependencies pinned in [`requirements.txt`](../requirements.txt).
2. **Scope decision.** 45 S&P 500 companies, sector-diversified, continuously listed across the
   full date range — see [`tickers.csv`](../tickers.csv).
   - Date range: **2021-01-01 → 2024-06-30**
   - Rationale for this window and for the 45-ticker count is documented in
     [Phase 1's report](Phase1_Data_Collection_Report.pdf), Section 5.
3. **Ticker universe.** Sector breakdown:

   | Sector | Count |
   |---|---|
   | Technology | 10 |
   | Financials | 8 |
   | Healthcare | 8 |
   | Consumer Discretionary | 5 |
   | Consumer Staples | 6 |
   | Industrials | 5 |
   | Energy | 2 |
   | Communication Services | 1 |

4. **Repo folder structure** (this repository):
   ```
   .
   ├── tickers.csv          # ticker + sector, read at runtime
   ├── collect_data.py       # Phase 1 pipeline script
   ├── requirements.txt
   ├── data/                 # pipeline outputs land here (committed — real dataset)
   └── docxxs/               # phase-wise documentation (this folder)
   ```

## Corrections made during setup

Two assumptions in the original plan were checked and corrected before building the pipeline:

- The **MAEC dataset** (originally assumed to cover 2021–2023) actually spans
  **Feb 2015 – Jun 2018** — it could not be used for this project's chosen window.
- The **Financial Modeling Prep (FMP)** transcript API is no longer free — it now requires
  the paid Ultimate tier.

Both are documented in full in [Phase 1's report](Phase1_Data_Collection_Report.pdf), Section 4.1.

## Next
→ [Phase 1 — Data Collection](Phase1_Data_Collection.md)
