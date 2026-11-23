# Decoding the Game

**Detecting Information Asymmetry in Earnings Calls with NLP**

A mini project (Semester 5) that teaches a model to read earnings-call language, scores each
firm as secretly *Good* or *Bad*, then tests whether the stock market already prices that signal
or is slow to absorb it, and whether that delay can be traded after costs.

> **Pipeline:** `transcript → FinBERT language features → classifier score → market-reaction test → backtest → dashboard + report`

## Team

| Role | Name |
|---|---|
| Instructor | Assistant Prof. Girish Sharma |
| Student | Kartik Mantri (24UCS246) |
| Student | Velu Kala (24DCS012) |
| Institute | The LNM Institute of Information Technology (LNMIIT) |

## Results at a glance

| Phase | What | Result | Details |
|---|---|---|---|
| 0 | Setup & rules | 200 S&P 500 stocks; no look-ahead; 2018–2025 holdout locked | [`Phase0`](docxxs/Phase0_Setup_and_Foundation.md) |
| 1 | Data | 12,789 calls, 198 companies, 2005–2025, labelled by drift abnormal return | [`Phase1`](docxxs/Phase1_Data_Collection.md) |
| 2 | Language features | 18 FinBERT / Loughran-McDonald features for 12,716 calls | [`Phase2`](docxxs/Phase2_Signal_Extraction.md) |
| 3 | Classifier | Walk-forward logistic regression, reaction AUC **0.611** on 2018–2025 | [`Phase3`](docxxs/Phase3_Type_Classifier.md) |
| 4 | Market test | Drift top − bottom **+1.62%** (p 0.002), +1.31% with controls (p 0.013) | [`Phase4`](docxxs/Phase4_Market_Reaction_Test.md) |
| 5 | Backtest | Market-neutral A +6.9%/yr (Sharpe 0.74); **chosen Strategy D +13.3%/yr** (Sharpe 0.93, alpha +6.0%, p 0.045) | [`Phase5`](docxxs/Phase5_Trading_Backtest.md), [`Strategy report`](docxxs/Strategy_Report.pdf) |
| 6 | Dashboard | Streamlit app on the committed results | [`Phase6`](docxxs/Phase6_Dashboard_and_Report.md) |

Also: an interactive [journey page](journey/) and a [presentation](presentation/Decoding_the_Game.pptx).
New to the project? Start with [`docxxs/project_flow_diagram.png`](docxxs/project_flow_diagram.png).

## Code, phase by phase

| File | Phase | What it does |
|---|---|---|
| `phase1_collect_transcripts_and_prices.py` | 1 | Downloads transcripts (Hugging Face) and daily prices (Yahoo Finance) |
| `phase1_abnormal_returns.py` | 1 (shared) | Abnormal-return windows around each call (pre-call, immediate, drift); used by Phases 1–5 |
| `phase1_build_labelled_calls.py` | 1 | Builds `data/calls_all.csv`: every call with its returns and GOOD/BAD label |
| `phase2_extract_finbert_features.py` | 2 | Splits each call, scores every sentence with FinBERT, writes 18 features → `data/features_finbert.csv` |
| `phase2_check_features.py` | 2 | Correlation heatmap, length checks, GOOD vs BAD t-tests (pre-2018 only) |
| `phase3_train_classifier.py` | 3 | Picks inputs on 2012–2017, trains walk-forward, writes one out-of-sample score per call → `data/predictions_finbert.csv` |
| `phase4_market_reaction_test.py` | 4 | Quintile spreads, Newey–West tests, regressions with controls, placebo → `data/market_reaction_finbert.csv` |
| `backtest_finbert.py` | 5 | Strategy A: market-neutral long-short backtest with costs |
| `backtest_finbert_2to1.py` | 5 | Strategies B, C, D, F, G: 2:1 tilted long-short (cut-offs set by `--long-pct` / `--short-pct`) |
| `backtest_finbert_longonly.py` | 5 | Strategy E: long-only, plus "buy every call" |
| `phase6_dashboard.py` | 6 | Streamlit dashboard (7 pages) |

Documentation for each phase is in [`docxxs/`](docxxs/); the decision log is [`PROGRESS.md`](PROGRESS.md).

## Run it

Everything from Phase 3 on runs from committed files:
```bash
pip install -r requirements.txt
python phase3_train_classifier.py            # uses the committed text-score cache
python phase4_market_reaction_test.py
python backtest_finbert.py                   # Strategy A
python backtest_finbert_2to1.py --long-pct 0.15 --short-pct 0.15   # Strategy D (+ B, C in the same file)
python backtest_finbert_longonly.py          # Strategy E
streamlit run phase6_dashboard.py
```

Phases 1 and 2 need the raw transcripts, which are too large for Git (> 100 MB per file) and are
rebuilt locally:
```bash
python phase1_build_labelled_calls.py --start 2005-01-01 --end 2025-12-31   # → data/calls_all.csv
python phase2_extract_finbert_features.py                                   # → data/features_finbert.csv (slow on CPU)
```

## Earlier VADER pipeline (archived)

The project first used VADER, a general-purpose sentiment tool, for its language features. FinBERT
replaced it because it reads financial text correctly; on the same calls the VADER version reached
a reaction AUC of 0.588 against FinBERT's 0.611. The old code, data and docs are kept in
[`archive/vader/`](archive/vader/) and are not part of the pipeline.

## Tech stack

`pandas` · `numpy` · `yfinance` · `datasets` (Hugging Face) · `transformers` + `torch` (FinBERT) · `pysentiment2` (Loughran-McDonald) · `textstat` · `scikit-learn` · `xgboost` · `scipy` · `matplotlib` · `plotly` · `streamlit`
