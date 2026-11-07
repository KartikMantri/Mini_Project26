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

## Project Status

| Phase | Description | Status | Output / Details |
|---|---|---|---|
| 0 | Setup & Foundation | ✅ Complete | Repository structure, universe definition, environment |
| 1 | Data Collection | ✅ Complete | 10,493 calls collected (200 tickers, 2005–2025), labeled via abnormal returns |
| 2 | Signal Extraction (Features) | ✅ Complete | 35 language features extracted per call → [`data/features_expanded.csv`](data/features_expanded.csv) |
| 3 | Type Classifier (Good/Bad) | ✅ Complete | Walk-forward model, reaction AUC 0.59 on 2018–2025 → [`docxxs/Phase3_results.md`](docxxs/Phase3_results.md) |
| 4 | Market-Reaction Test | 🔄 Script ready | Event study & Welch's t-test (`market_reaction.py`) |
| 5 | Trading Backtest | 🔄 Script ready | Long/short portfolio simulation & Sharpe ratio (`backtest.py`) |
| 6 | Dashboard & Report | 🔄 Script ready | Interactive Streamlit analytics UI (`dashboard.py`) |

Full documentation for each phase lives in [`docxxs/`](docxxs/).

---

## Data Access & Teammate Setup

### For Collaborators (Starting Phase 3 Immediately)
The core modeling dataset is **already committed to this repository**:
- **[`data/features_expanded.csv`](data/features_expanded.csv)** (~4 MB): Contains all 10,493 earnings calls, 35 engineered features (VADER sentiments, Loughran-McDonald lexicon, Fog/Flesch readability, specificity, Q&A dynamics), and the binary `label` (`GOOD` / `BAD`).
- **[`data/prices.csv`](data/prices.csv)** (~18 MB): Historical stock prices used for market return benchmarks.

**You do NOT need to download gigabytes of raw transcripts or re-run the 7-worker NLP extraction.** Simply pull the repo and run Phase 3:
```bash
pip install -r requirements.txt
python train_classifier.py
```

Phase 4 (`market_reaction.py`) and the dashboard (`dashboard.py`) also read `data/master.csv`, which is not in Git — generate it locally using the steps below before running them.

### Why Raw Transcript Files Are Excluded From Git
Raw transcripts and intermediate files exceed GitHub's strict 100 MB per-file limit:
- `data/master.csv` (~582 MB) — needed locally for Phase 4 and the dashboard
- `data/raw_transcripts.csv` (~741 MB) — intermediate; no later phase reads it, safe to delete
- `data/labeled_all.csv` (~709 MB) — intermediate; no later phase reads it, safe to delete

These are excluded via `.gitignore`. To re-generate them locally:
```bash
# 1. Download transcripts from Hugging Face & label returns using the committed data/prices.csv
python collect_data.py --start 2005-01-01 --end 2025-12-31

# 2. Optional: only if the features need to change, re-extract using 7 parallel worker processes
python extract_features_fast.py --workers 7
```

Notes for step 1:
- **Always pass the dates.** The script defaults to 2021-01-01 → 2024-06-30, which gives only 2,317 rows in `master.csv` instead of 10,493.
- **Don't use `--force`** if you want to reproduce the committed dataset. It re-downloads `prices.csv`, which can change the labels.
- **Delete any partial `data/raw_transcripts.csv` first.** The script reuses an existing file, even one covering a narrower date range.

---

## Repository Structure

```
.
├── README.md                           # Project overview & quick start
├── PROGRESS.md                         # Detailed chronological decision log
├── requirements.txt                    # Python package dependencies
├── tickers.csv                         # 200-ticker S&P universe (ticker, sector)
│
├── collect_data.py                     # Phase 1: Transcript scraping & return labeling
├── extract_features_fast.py            # Phase 2: Multi-process feature extraction orchestrator
├── _feat_worker.py                     # Phase 2: Parallel worker for NLP feature extraction
├── event_returns.py                    # Shared event-window return helpers (corrected event dates)
├── train_classifier.py                 # Phase 3: Walk-forward classifier (TF-IDF text + features)
├── feature_study.py                    # Phase 3: Which feature groups help (greedy selection)
├── market_reaction.py                  # Phase 4: Market-reaction statistical tests
├── backtest.py                         # Phase 5: Trading strategy backtest simulation
├── dashboard.py                        # Phase 6: Streamlit interactive dashboard
│
├── data/
│   ├── features_expanded.csv           # 10,493 rows x 35 columns (Phase 2 output, committed)
│   ├── prices.csv                      # Historical daily price data (committed)
│   └── (master, raw_transcripts, labeled_all are gitignored due to >100MB size; generated locally)
│
└── docxxs/                             # Phase-wise documentation & decisions
    ├── features.md                     # Comprehensive dictionary for all 35 features
    ├── Decision_Velu_Data_Scaling.md   # Documentation on scaling to 10k+ rows
    ├── Phase0_Setup_and_Foundation.md
    ├── Phase1_Data_Collection.md
    ├── Phase1_Data_Collection_Report.pdf
    ├── Phase2_Signal_Extraction.md
    ├── Phase3_Type_Classifier.md
    ├── Phase4_Market_Reaction_Test.md
    ├── Phase5_Trading_Backtest.md
    └── Phase6_Dashboard_and_Report.md
```

## Tech Stack

`pandas` · `numpy` · `yfinance` · `datasets` (Hugging Face) · `nltk` (VADER) · `pysentiment2` (Loughran-McDonald) · `textstat` · `scikit-learn` · `xgboost` · `streamlit`
