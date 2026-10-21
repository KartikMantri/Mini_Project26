# Phase 2 — Signal Extraction (Features)

**Status:** ✅ `data/features_expanded.csv` generated (10,493 rows, 35 columns) — see [features.md](features.md) for the full feature reference and dataset statistics.

## Goal
Transform raw earnings call transcripts into structured numerical feature vectors describing linguistic style, management sentiment, uncertainty, complexity, and Q&A dynamics.

## Dataset Scaling & Architecture Evolution
The initial prototype extracted 17 FinBERT-based features across 509 calls in `features.csv`. When the dataset was expanded to **10,493 calls** (200 tickers, 2005–2025) per the data scaling plan ([`Decision_Velu_Data_Scaling.md`](Decision_Velu_Data_Scaling.md)), single-process FinBERT inference proved too computationally slow (~50+ hours on CPU). 

To scale efficiently while improving signal diversity, the pipeline transitioned to:
- **Fast multi-process orchestrator:** [`extract_features_fast.py`](../extract_features_fast.py) leveraging 7 parallel background workers running [`_feat_worker.py`](../_feat_worker.py).
- **VADER Sentiment Analysis:** Fast, robust lexical sentiment computed separately across Prepared Remarks, Q&A sections, and paragraph samples.
- **Expanded Feature Set (35 columns):** Incorporating Loughran-McDonald financial dictionary percentages, Fog/Flesch readability indices, numeric specificity metrics, and Q&A directness indicators.

## Design Decisions

- **Section Split:** Each transcript is split into **Prepared Remarks** vs **Q&A** at the 2nd `"Operator:"` turn. The operator opens every call (1st turn), remains silent through prepared remarks, and speaks again to introduce the first Q&A caller (2nd turn). Transcripts with fewer than 2 operator turns fall back to treating the full text as prepared remarks with `NaN` for Q&A-specific metrics.
- **Sentiment Scoring & Aggregation:**
  - `vader_prepared_pos`, `vader_prepared_neg`, `vader_prepared_compound`: Sentiment metrics on prepared remarks.
  - `vader_qa_pos`, `vader_qa_neg`, `vader_qa_compound`: Sentiment metrics on Q&A responses (256 rows with no split Q&A contain `NaN`, handled by downstream tree models).
  - Paragraph-level compound stats: `sent_compound_mean`, `sent_compound_min`, `sent_compound_max`, `sent_compound_std`, `pct_pos_sents`, `pct_neg_sents`.
- **Financial Lexicon (Loughran-McDonald):**
  - Proportions of uncertainty words (`lm_uncertainty_pct`), weak modals (`lm_weak_modal_pct`), strong modals (`lm_strong_modal_pct`), and litigious terms (`lm_litigious_pct`).
- **Readability & Complexity:**
  - `fog_index`: Gunning Fog index for complexity.
  - `flesch_ease`: Flesch Reading Ease score.
  - `avg_sentence_length`: Average sentence length in words.
- **Specificity & Q&A Dynamics:**
  - `numeric_density`: Fraction of words containing digits (concrete disclosures vs vague generalities).
  - `forward_looking_count`: Frequency of guidance and forward-looking statements.
  - `qa_directness`, `qa_turn_count`, `qa_ratio`: Measures how directly management answers analysts and the balance between remarks and questioning.

## Feature Schema Summary

| Group | Columns | Count | Description |
|---|---|---|---|
| Identifiers & Targets | `ticker`, `call_date`, `label`, `abnormal_return` | 4 | Primary keys and target labels |
| VADER Prepared | `vader_prepared_{pos,neg,compound}` | 3 | Tone of management's opening script |
| VADER Q&A | `vader_qa_{pos,neg,compound}` | 3 | Tone of unscripted executive answers |
| Paragraph Sentiment | `sent_compound_{mean,min,max,std}`, `pct_{pos,neg}_sents` | 6 | Micro-level sentiment distribution |
| Loughran-McDonald | `lm_{uncertainty,weak_modal,strong_modal,litigious}_pct` | 4 | Financial-domain dictionary frequencies |
| Readability | `fog_index`, `flesch_ease`, `avg_sentence_length` | 3 | Text complexity and readability |
| Specificity & Structure | `numeric_density`, `word_count`, `forward_looking_{count,pct,density}` | 5 | Precision of corporate disclosure |
| Q&A Dynamics | `qa_directness`, `qa_turn_count`, `qa_ratio` | 3 | Executive responsiveness and interaction balance |

*For complete descriptive statistics (mean, min, max, null counts) on all 35 columns, see [`docxxs/features.md`](features.md).*

## Input / Output
- **Input:** `data/master.csv` (10,493 calls)
- **Output:** `data/features_expanded.csv` (10,493 rows, 35 columns, 3.98 MB) — committed directly to the repo for Phase 3.

## Next Step
→ **Phase 3 — Type Classifier:** Train Logistic Regression baseline and XGBoost models on `data/features_expanded.csv` using date-based train/test splits.
