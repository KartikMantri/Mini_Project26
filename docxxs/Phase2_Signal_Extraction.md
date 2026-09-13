# Phase 2 — Signal Extraction (Features)

**Status:** ⬜ Not started

## Goal
Turn each transcript from text into a row of numbers describing *how* it was said.

## Planned features
1. **Loughran–McDonald dictionary counts** (via `pysentiment2`): % uncertainty, % weak-modal,
   % strong-modal, % negative, % litigious.
2. **FinBERT sentiment** (`ProsusAI/finbert`): positive / negative / neutral probability.
3. **Specificity:** numeric density, count of forward-looking statements, directness of
   Q&A answers.
4. **Readability:** Fog index (`textstat`).

## Input / Output
- **Input:** `data/master.csv` (`transcript_text` column) from Phase 1.
- **Output:** `features.csv` — one numeric feature row per call, joined on `ticker` + `call_date`.

## Next
→ Phase 3 — Type Classifier
