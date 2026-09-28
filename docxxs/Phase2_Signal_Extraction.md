# Phase 2 — Signal Extraction (Features)

**Status:** ✅ `features.csv` generated (509 rows, 17 features) — see [features.md](features.md) for the full feature reference
**Flowchart:** [`phase2_flowchart.svg`](phase2_flowchart.svg)

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

## Known limitations (found during the sanity-check pass, 2026-09-28)

| # | Limitation | Evidence | Possible solution |
|---|---|---|---|
| 1 | **No statistically significant separation between GOOD and BAD on any single feature.** A Welch's t-test on all 17 features between the two label groups found the best p-value is 0.10 (`finbert_qa_mean`); every other feature is p > 0.28. `numeric_density` is essentially identical between groups (diff ≈ 0.00001). | Full t-test results in the 2026-09-28 sanity-check session (see PROGRESS.md decision log). | This is a univariate/linear test only — it can't see interactions or nonlinear patterns, which is exactly what XGBoost is for. Don't treat this as a verdict; treat it as a reason to (a) actually run Phase 3's classifier before concluding the features are unpredictive, (b) engineer interaction/delta features (e.g. `finbert_qa_mean - finbert_prepared_mean`, a "tone shift into Q&A" signal) that a plain group-mean test would miss, and (c) consider that executives are trained to sound calm and confident regardless of actual results — raw tone may be a genuinely weak predictor on its own, and language features may need to be combined with (not replace) hard financial metrics like EPS beat/miss in a future iteration. |
| 2 | **The direction is sometimes counter-intuitive.** `finbert_qa_mean` is very slightly *higher* on average for BAD calls (0.223) than GOOD calls (0.207) — the opposite of naive expectation. | Same t-test pass. | Investigate whether this reverses when controlling for company (some executives are just more upbeat speakers regardless of quarter quality) — e.g. normalize each call's sentiment against that same company's own historical average tone, rather than using the raw absolute score. |
| 3 | **4/509 rows (0.8%) still have no detectable Q&A section** (MCD 2022-04-28, TMO 2021-04-28, WMT 2021-02-20, XOM 2024-04-26) — the structural "2nd Operator: turn" rule found nothing because these transcripts don't use `"Operator:"` speaker tags consistently, or (for TMO) the transcript is actually mislabeled at the source (it's a PPD earnings call, not Thermo Fisher). | Found and confirmed by re-reading `data/master.csv` directly during the split-logic fix. | Drop these 4 rows before training (simplest, and they're a data-quality issue, not a modeling one), or hand-inspect each transcript's actual structure and add a one-off fallback marker per case. The TMO/PPD mismatch specifically should be flagged as a possible symptom of other undetected mismatches in the source Hugging Face dataset — worth a wider spot-check (see #6). |
| 4 | **The Loughran-McDonald dictionary can't see negation.** "We did **not** see a decline" still counts "decline" as a negative-category hit, because LM is pure word-counting with no grammar awareness. | Documented behavior of dictionary-based sentiment generally (see [features.md](features.md)). | Either accept this as a known, bounded source of noise (FinBERT is the feature that's supposed to catch what LM can't, since it does understand negation) and weight LM features accordingly, or add a lightweight negation-flip preprocessing pass (e.g. flip a word's polarity if a negation word appears within ~3 tokens before it). |
| 5 | **Chunk-level dilution isn't fully solved by min/max/std.** Aggregating with mean/min/max/std helps outlier moments survive (the original reason for choosing this over a single average), but within a long section with many chunks, `mean` can still average out real signal, and `min`/`max` only capture the single most extreme chunk, not "how many chunks were bad." | Inherent to any chunk-then-aggregate approach on documents far longer than a model's token limit. | Consider adding a count-based feature (e.g. "% of chunks with net sentiment below -0.2") in a future iteration, which captures breadth of negative tone rather than just its peak. |
| 6 | **FinBERT output has only been spot-checked on a handful of AAPL calls**, not systematically across sectors/companies. | Directional sanity check so far only covered AAPL. | Before fully trusting `features.csv` for Phase 3, spot-check a small random sample (5-10 calls) from other sectors (e.g. energy, healthcare) against known real-world outcomes, the same way Phase 1's labels were validated. |
| 7 | **Specificity heuristics are naive regex, not context-aware.** `numeric_density` counts *any* digit-containing word (including dates like "2021" or ordinal quarter labels like "Q1"), `forward_looking_count` is a raw substring match with no grammar check, and `qa_directness` only checks "does the answer contain any number" — an analyst could get a throwaway number unrelated to their actual question and still count as "direct." | Design-time review of the regex logic in `extract_features.py`. | Accept as a cheap first-pass heuristic (consistent with the project's stated preference for simple, transparent signals per [[project-decoding-the-game]]) and revisit only if Phase 3's feature importances show these columns matter — no point refining a feature the classifier ends up ignoring. |
| 8 | **FinBERT's neutral-class probability is discarded.** Reducing each chunk to `P(positive) - P(negative)` means a chunk that's 90% neutral/5%/5% looks identical to one that's 50%/45%/5% in the net-sentiment number, even though the second is a much more decisive, information-dense chunk. | Design choice made when locking Option D (see Design decisions above). | If Phase 3's accuracy is disappointingly low, consider adding `finbert_{prepared,qa}_neutral_mean` back as an extra pair of columns — "how much of the call was low-conviction filler" is a distinct signal from directional tone. |

## Next
→ Decide how to handle limitation #3's 4 NaN rows, run Phase 3 — Type Classifier, and treat limitation #1
as an open empirical question the classifier itself needs to answer.
