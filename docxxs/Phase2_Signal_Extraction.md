# Phase 2 — Signal Extraction (Features)

**Status:** ✅ `features.csv` generated (509 rows, 17 features) — see [features.md](features.md) for the full feature reference

## Goal
Turn each transcript from text into a row of numbers describing *how* it was said.

## Design decisions (locked 2026-09-27)

- **Section split:** each transcript is split into **prepared remarks** vs **Q&A** at the **2nd
  `"Operator:"` turn** — the operator opens every call (1st turn), then stays silent through
  prepared remarks and speaks again only to introduce the first Q&A caller (2nd turn). This
  structural rule replaced an earlier "first question" keyword marker after real-data testing
  found the keyword approach both under- and over-fired: it missed calls that transition to
  Q&A with different phrasing (e.g. Cisco just says `"Operator: Meta Marshall from Morgan
  Stanley."` with no "first question" anywhere), and it occasionally false-positived on
  analysts saying "my first question is..." mid-Q&A, silently shifting the split point by
  thousands of characters in either direction on some "successful" rows. The structural rule
  cut split failures from 31/509 (6%) to 4/509 (0.8%) — the remaining 4 are genuine edge cases
  (fewer than 2 `"Operator:"` turns, e.g. one row mislabeled in the source dataset: a
  transcript tagged `TMO` is actually a PPD earnings call). Those 4 rows fall back to
  treating the whole transcript as "prepared remarks" with `NaN` Q&A features.
- **Chunking:** FinBERT has a 512-token limit; transcripts average ~9,400 words (range
  1,168–31,216), so each section is split into ≤512-token chunks before scoring.
- **FinBERT aggregation — Option D (split, 8 features):** each chunk is reduced to one scalar,
  **net sentiment = P(positive) − P(negative)** (range −1 to +1), rather than keeping all
  three FinBERT class probabilities — this keeps both positive and negative extremes visible
  in a single number. Mean / min / max / std of that scalar is computed **separately** for
  prepared remarks and Q&A (not combined into one whole-transcript average) → 4 stats × 2
  sections = 8 columns. Rejected the single whole-transcript average because averaging across
  ~20+ mostly-neutral chunks dilutes the 1-2 chunks that actually carry signal (e.g. a
  0.58-negative chunk gets buried in a ~0.06 whole-call average).
- **No hand-picked section weights:** prepared-remarks and Q&A features are kept as separate
  columns rather than combined with an assumed weighting (e.g. "Q&A = 3x prepared remarks").
  Phase 3's classifier learns the actual weighting from labeled outcomes instead.
- **Q&A follow-ups not split out:** detecting follow-up vs. first-answer questions specifically
  was considered and rejected for this pass (marker not reliable enough) — flagged as a
  possible future refinement.

## Feature schema

| # | Group | Columns | Method |
|---|---|---|---|
| 1–8 | FinBERT sentiment | `finbert_{prepared,qa}_{mean,min,max,std}` | Chunk each section (≤510 tokens), score each chunk as net sentiment = P(positive) − P(negative), aggregate per section |
| 9–13 | Loughran–McDonald % | `lm_uncertainty_pct`, `lm_weak_modal_pct`, `lm_strong_modal_pct`, `lm_negative_pct`, `lm_litigious_pct` | `pysentiment2` dictionary word-counts over the whole transcript |
| 14 | Readability | `fog_index` | `textstat`, whole transcript |
| 15–17 | Specificity | `numeric_density`, `forward_looking_count`, `qa_directness` | Regex/heuristic, whole transcript |

Plus `ticker`, `call_date`, `label` carried through unchanged for the join into Phase 3.

## Input / Output
- **Input:** `data/master.csv` (`transcript_text` column) from Phase 1.
- **Output:** `features.csv` — one numeric feature row per call, joined on `ticker` + `call_date`.

## Dependencies
Added to `requirements.txt`: `transformers`, `torch`, `pysentiment2`, `textstat`, `nltk`.

## Script
[`extract_features.py`](../extract_features.py) — resumable (skips ticker/call_date pairs
already in `features.csv` on rerun), checkpoints every 20 rows.

## Next
→ Run `extract_features.py`, review `features.csv`, then Phase 3 — Type Classifier
