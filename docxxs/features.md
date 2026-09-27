# Phase 2 — Feature Reference

Full reference for the 17 numeric feature columns in `features.csv`, produced by
[`extract_features.py`](../extract_features.py) from `data/master.csv`. See
[Phase2_Signal_Extraction.md](Phase2_Signal_Extraction.md) for the design rationale behind
these choices.

Each transcript is split into two sections — **prepared remarks** and **Q&A** — using the
"first question" text marker, before any feature is computed.

## FinBERT sentiment (8 features)

Each chunk (≤510 tokens, whole sentences packed together so none are cut mid-sentence) is
scored by FinBERT (`ProsusAI/finbert`) and reduced to one scalar:

```
net_sentiment = P(positive) - P(negative)      # range -1 to +1
```

These 8 columns are the mean/min/max/std of that scalar, computed **separately** per section
(not combined into one whole-transcript average — averaging across ~20+ mostly-neutral chunks
would dilute the 1-2 chunks that actually carry signal).

| Feature | Meaning | Formula |
|---|---|---|
| `finbert_prepared_mean` | Average tone of the prepared-remarks section | mean of net-sentiment across prepared-remarks chunks |
| `finbert_prepared_min` | Most negative moment in prepared remarks | min of net-sentiment across prepared-remarks chunks |
| `finbert_prepared_max` | Most positive moment in prepared remarks | max of net-sentiment across prepared-remarks chunks |
| `finbert_prepared_std` | How much tone swings within prepared remarks | std-dev of net-sentiment across prepared-remarks chunks |
| `finbert_qa_mean` | Average tone across analyst Q&A | mean of net-sentiment across Q&A chunks |
| `finbert_qa_min` | Most negative moment in Q&A (e.g. a defensive/hedged answer) | min of net-sentiment across Q&A chunks |
| `finbert_qa_max` | Most positive moment in Q&A | max of net-sentiment across Q&A chunks |
| `finbert_qa_std` | How much tone swings within Q&A | std-dev of net-sentiment across Q&A chunks |

If the "first question" marker isn't found (~2% of calls), the Q&A section is empty and all
four `finbert_qa_*` columns are `NaN` for that row.

## Loughran–McDonald dictionary (5 features)

Whole-transcript word counting — no model, just checking each word against the LM finance
sentiment dictionary (~4,000 words, manually classified from real 10-K filings, built to avoid
false positives that general-purpose dictionaries make on financial text, e.g. "liability").

```
lm_<category>_pct = (count of transcript words in that LM category) / (total word count)
```

| Feature | Meaning | LM category |
|---|---|---|
| `lm_uncertainty_pct` | How much hedging language ("approximately," "may," "uncertain") | `Uncertainty` |
| `lm_weak_modal_pct` | Low-confidence forward language ("could," "might," "apparently") | `Modal == 3` (weak) |
| `lm_strong_modal_pct` | High-confidence, assertive language ("definitely," "clearly," "always") | `Modal == 1` (strong) |
| `lm_negative_pct` | Density of finance-specific negative words ("decline," "loss," "impairment") | `Negative` |
| `lm_litigious_pct` | Density of legal/litigation language ("lawsuit," "plaintiff," "settlement") | `Litigious` |

Note: `pysentiment2`'s `LM.get_score()` only exposes `Positive`/`Negative`/`Polarity`/
`Subjectivity` — `extract_features.py` reads the bundled LM dictionary CSV directly
(`load_lm_categories`) to get the `Uncertainty`, `Litigious`, and `Modal` categories, since the
package doesn't surface them itself.

## Readability (1 feature)

| Feature | Meaning | Formula |
|---|---|---|
| `fog_index` | Estimated years of schooling needed to parse the text in one read; a proxy for whether language got denser/more complex than usual | `0.4 × [(words/sentences) + 100 × (complex_words/words)]`, where `complex_words` = words with 3+ syllables (`textstat.gunning_fog`) |

Finance rationale: some research finds executives use denser, more hedged language when
delivering bad news and plainer language for good news — this is a cheap, transparent proxy
for that.

## Specificity (3 features)

Custom heuristics for "is the speaker being concrete and committal, or vague and hedgy?"

| Feature | Meaning | Formula |
|---|---|---|
| `numeric_density` | How concrete/data-backed the language is (numbers vs. vague description) | `(words containing a digit) / total_words`, over the whole transcript |
| `forward_looking_count` | Volume of future-oriented statements (predictions, guidance, promises) | raw count of matches for phrases like "expect," "guidance," "outlook," "going forward," "next quarter" |
| `qa_directness` | How often analysts get a concrete number back in Q&A vs. a dodge | `(Q&A answer-turns containing a number) / (total Q&A answer-turns)` — `NaN` if no Q&A section was found |

## Non-feature columns carried through

| Column | Purpose |
|---|---|
| `ticker` | Join key back to `master.csv` / `prices.csv` |
| `call_date` | Join key back to `master.csv` / `prices.csv` |
| `label` | `GOOD` or `BAD` — the target variable Phase 3's classifier predicts |
