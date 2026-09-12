# Decoding the Game

**Detecting Information Asymmetry in Earnings Calls with NLP**

A mini project (Semester 5) that teaches a model to read earnings-call language, classifies
each firm as secretly *Good* or *Bad*, then tests whether the stock market already prices
that signal — or misses it (a tradeable inefficiency).

> **Core pipeline:** `transcript → language features → Good/Bad classifier → market-reaction test → backtest → dashboard + report`

## Team

| Role | Name |
|---|---|
| Instructor | Assistant Prof. Girish Sharma |
| Student | Kartik Mantri (24UCS246) |
| Student | Velu Kala (24DCS012) |
| Institute | The LNM Institute of Information Technology (LNMIIT) |

## Project status

| Phase | Description | Status |
|---|---|---|
| 0 | Setup & Foundation | ✅ Complete |
| 1 | Data Collection | ✅ Complete |
| 2 | Signal Extraction (Features) | ⬜ Not started |
| 3 | Type Classifier (Good/Bad) | ⬜ Not started |
| 4 | Market-Reaction Test | ⬜ Not started |
| 5 | Trading Backtest | ⬜ Not started |
| 6 | Dashboard & Report | ⬜ Not started |

Full documentation for each phase lives in [`docxxs/`](docxxs/).

## Repository structure

```
Mini_Project26/
├── README.md
├── requirements.txt
├── tickers.csv              # 45-ticker universe (ticker, sector)
├── collect_data.py           # Phase 1 pipeline script
├── data/                     # pipeline outputs (gitignored — regenerate locally)
└── docxxs/                   # phase-wise documentation
    ├── Phase0_Setup_and_Foundation.md
    ├── Phase1_Data_Collection.md
    ├── Phase1_Data_Collection_Report.pdf
    ├── phase1_flowchart.svg
    ├── phase1_flowchart.png
    ├── Phase2_Signal_Extraction.md
    ├── Phase3_Type_Classifier.md
    ├── Phase4_Market_Reaction_Test.md
    ├── Phase5_Trading_Backtest.md
    └── Phase6_Dashboard_and_Report.md
```

## Quick start

```bash
pip install -r requirements.txt
python collect_data.py
```

This regenerates `data/master.csv` — the labeled dataset that every later phase builds on.
See [`docxxs/Phase1_Data_Collection.md`](docxxs/Phase1_Data_Collection.md) for the full
methodology, schema, and results.

## Tech stack

`pandas` · `numpy` · `yfinance` · `datasets` (Hugging Face) · `transformers` (FinBERT, Phase 2)
· `scikit-learn` / `xgboost` (Phase 3) · `statsmodels` (Phase 4) · React + Recharts (Phase 6)
