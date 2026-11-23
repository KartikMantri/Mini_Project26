# Phase 2 — Feature Reference

Full reference for all **35 columns** in `data/features_expanded.csv`, produced by
[`_feat_worker.py`](./_feat_worker.py) (called by [`extract_features_fast.py`](../extract_features_fast.py))
from `data/master.csv`.

> **Dataset summary:** 10,493 labeled earnings calls (GOOD: 5,638 | BAD: 4,855)
> across ~200 S&P 500 tickers, 2005–2025.

Each transcript is split into two sections — **prepared remarks** and **Q&A** — by
detecting the second `Operator:` speaker turn (which introduces the first analyst question).
All sentiment features are computed separately per section.

---

## 1. Identity Columns (2)

| Column | Type | Description |
|--------|------|-------------|
| `ticker` | string | Stock ticker symbol (e.g. `AAPL`) — join key back to `master.csv` |
| `call_date` | datetime | Date/time of earnings call — used for date-based train/test split |

---

## 2. VADER Sentiment — Whole Transcript (4)

VADER (Valence Aware Dictionary and sEntiment Reasoner) scores the entire transcript
as a single string. Range: compound [-1, +1]; pos/neg/neu [0, 1] summing to 1.

| Column | Mean | Min | Max | Nulls | Description |
|--------|------|-----|-----|-------|-------------|
| `vader_compound` | 0.9952 | -0.5574 | 1.0 | 0 | Normalised compound score for the full transcript |
| `vader_pos` | 0.1455 | 0.0 | 0.246 | 0 | Fraction of text with positive valence |
| `vader_neg` | 0.0224 | 0.0 | 0.079 | 0 | Fraction of text with negative valence |
| `vader_neu` | 0.8322 | 0.754 | 1.0 | 0 | Fraction of text with neutral valence |

---

## 3. VADER Sentiment — Prepared Remarks Section (3)

VADER applied to the prepared-remarks portion only (before the first analyst question).

| Column | Mean | Min | Max | Nulls | Description |
|--------|------|-----|-----|-------|-------------|
| `vader_prepared_compound` | 0.9914 | -0.6486 | 1.0 | 0 | Compound sentiment of prepared remarks |
| `vader_prepared_pos` | 0.1446 | 0.0 | 0.265 | 0 | Positive fraction of prepared remarks |
| `vader_prepared_neg` | 0.0236 | 0.0 | 0.092 | 0 | Negative fraction of prepared remarks |

---

## 4. VADER Sentiment — Q&A Section (3)

VADER applied to the Q&A portion only (from the first analyst question onward).
**256 nulls** = calls with no detectable Q&A section.

| Column | Mean | Min | Max | Nulls | Description |
|--------|------|-----|-----|-------|-------------|
| `vader_qa_compound` | 0.9968 | -0.9042 | 1.0 | 256 | Compound sentiment of Q&A section |
| `vader_qa_pos` | 0.1459 | 0.0 | 0.5 | 256 | Positive fraction of Q&A |
| `vader_qa_neg` | 0.0215 | 0.0 | 0.076 | 256 | Negative fraction of Q&A |

---

## 5. Paragraph-Level Sentiment Distribution (6)

Rather than scoring every sentence individually (430+ VADER calls per transcript),
up to 60 paragraphs are sampled evenly and scored. These statistics capture **how
sentiment varies throughout the call** — a flat-positive call vs. one with dips.
**46 nulls** = transcripts with no parseable paragraphs.

| Column | Mean | Min | Max | Nulls | Description |
|--------|------|-----|-----|-------|-------------|
| `sent_compound_mean` | 0.5563 | -0.5574 | 1.0 | 46 | Average paragraph-level compound sentiment |
| `sent_compound_std` | 0.3842 | 0.0 | 0.615 | 46 | Std-dev of paragraph sentiment (tone variability) |
| `sent_compound_min` | -0.4440 | -0.9962 | 1.0 | 46 | Most negative paragraph in the call |
| `sent_compound_max` | 0.9976 | -0.5574 | 1.0 | 46 | Most positive paragraph in the call |
| `pct_positive_sents` | 0.6043 | 0.0 | 1.0 | 46 | Fraction of paragraphs with compound > 0.5 |
| `pct_negative_sents` | 0.0145 | 0.0 | 1.0 | 46 | Fraction of paragraphs with compound < -0.5 |

---

## 6. Loughran–McDonald Financial Lexicon (4)

Word-level counts against the Loughran–McDonald (LM) financial sentiment dictionary —
designed specifically for corporate finance text (avoids false positives from general-purpose
dictionaries, e.g. "liability" is negative in finance but not in everyday English).

```
lm_<category>_pct = (count of transcript words in that LM category) / total_word_count
```

| Column | Mean | Min | Max | Nulls | Description |
|--------|------|-----|-----|-------|-------------|
| `lm_positive_pct` | 0.0162 | 0.0 | 0.052 | 0 | Density of LM positive words (e.g. "achieve", "growth", "record") |
| `lm_negative_pct` | 0.0048 | 0.0 | 0.0226 | 0 | Density of LM negative words (e.g. "decline", "loss", "impairment") |
| `lm_uncertainty_pct` | 0.0102 | 0.0 | 0.0214 | 0 | Density of hedging/uncertainty words (e.g. "approximately", "may", "uncertain") |
| `lm_net_sentiment` | 0.0114 | -0.0122 | 0.0465 | 0 | `lm_positive_pct − lm_negative_pct` — net financial tone |

---

## 7. Readability (3)

Standard readability indices applied to the full transcript text.

| Column | Mean | Min | Max | Nulls | Description |
|--------|------|-----|-----|-------|-------------|
| `fog_index` | 12.34 | 8.38 | 136.5 | 0 | Gunning Fog Index — estimated years of schooling to read the text. Higher = denser language. Formula: `0.4 × [(words/sentences) + 100 × (complex_words/words)]` where complex = 3+ syllables |
| `flesch_ease` | 56.21 | -277.4 | 72.2 | 0 | Flesch Reading Ease score — higher = easier to read (0–100 scale). Negative outliers = very complex/unusual transcripts |
| `avg_sentence_length` | 17.96 | 1.0 | 317.0 | 0 | Mean words per sentence (`textstat.words_per_sentence`) |

> Finance rationale: executives may use denser, more hedged language when delivering
> bad news and plainer language for good news.

---

## 8. Structural / Specificity (5)

Custom heuristics for whether the speaker is being **concrete and committal** vs. **vague**.

| Column | Mean | Min | Max | Nulls | Description |
|--------|------|-----|-----|-------|-------------|
| `numeric_density` | 0.0218 | 0.0 | 0.154 | 0 | Fraction of words containing a digit — proxy for data-backed, specific language |
| `word_count` | 9,488 | 1 | 42,971 | 0 | Total word count of the transcript |
| `sentence_count` | 516 | 1 | 2,598 | 0 | Number of sentences (regex-based splitter) |
| `forward_looking_count` | 65.2 | 0 | 264 | 0 | Raw count of forward-looking phrases (e.g. "expect", "guidance", "outlook", "next quarter") |
| `forward_looking_density` | 0.0069 | 0.0 | 0.0219 | 0 | `forward_looking_count / word_count` — normalised guidance density |

---

## 9. Q&A Features (3)

Structural analysis of the analyst question-and-answer section.
**256 nulls** for `qa_directness` = calls with no detectable Q&A turns.

| Column | Mean | Min | Max | Nulls | Description |
|--------|------|-----|-----|-------|-------------|
| `qa_directness` | 0.3348 | 0.0 | 1.0 | 256 | Fraction of answer-turns containing a number — proxy for how concretely executives respond to analyst questions |
| `qa_turn_count` | 76.9 | 0 | 277 | 0 | Total number of speaker turns in the Q&A section |
| `qa_ratio` | 4.085 | 0.0 | 6150.4 | 0 | `len(Q&A) / len(prepared remarks)` — relative length of Q&A vs. prepared section |

---

## 10. Target & Return Columns (2)

| Column | Type | Values | Description |
|--------|------|--------|-------------|
| `label` | string | `GOOD` / `BAD` | Target variable. `GOOD` = abnormal return > +1% in [+2, +62] trading day window. `BAD` = abnormal return < -1%. `NEUTRAL` rows excluded from model training. |
| `abnormal_return` | float | [-1.0, +1.0] | Stock return minus S&P 500 return over the [+2, +62] window after the call date |

---

## Feature Groups Summary

| Group | # Features | Columns |
|-------|-----------|---------|
| VADER — whole transcript | 4 | `vader_compound`, `vader_pos`, `vader_neg`, `vader_neu` |
| VADER — prepared remarks | 3 | `vader_prepared_*` |
| VADER — Q&A | 3 | `vader_qa_*` |
| Paragraph-level sentiment | 6 | `sent_compound_*`, `pct_*_sents` |
| Loughran–McDonald lexicon | 4 | `lm_*` |
| Readability | 3 | `fog_index`, `flesch_ease`, `avg_sentence_length` |
| Structural / specificity | 5 | `numeric_density`, `word_count`, `sentence_count`, `forward_looking_*` |
| Q&A structure | 3 | `qa_directness`, `qa_turn_count`, `qa_ratio` |
| **Total features** | **31** | |
| Identity | 2 | `ticker`, `call_date` |
| Target | 2 | `label`, `abnormal_return` |
| **Total columns** | **35** | |

---

## Null Handling

| Column(s) | Null count | Cause | Treatment in Phase 3 |
|-----------|-----------|-------|----------------------|
| `vader_qa_*`, `qa_directness` | 256 | No Q&A section found | Imputed with column median before training |
| `sent_compound_*`, `pct_*_sents` | 46 | Transcript has no newline-separated paragraphs | Imputed with column median before training |
| All other feature columns | 0 | — | No imputation needed |

---

*Generated from `data/features_expanded.csv` — 10,493 rows, 35 columns.*
*See [`Phase2_Signal_Extraction.md`](Phase2_Signal_Extraction.md) for design rationale.*
