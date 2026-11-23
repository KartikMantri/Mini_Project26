# Phase 2 — Signal Extraction (Language Features)

**Status:** ✅ Complete (2026-11-11). Output: `data/features_finbert.csv` (12,716 calls × 22 columns).
Full feature reference: [`features_finbert.md`](features_finbert.md).

## Goal
Turn each earnings-call transcript into numbers that describe *how* management and analysts
spoke, using only the call's own text and the same company's earlier calls (no look-ahead).

## Input
`data/calls_all.csv` from Phase 1: 12,789 calls with full transcript text and speaker names.

## Procedure
Script: [`phase2_extract_finbert_features.py`](../phase2_extract_finbert_features.py).

1. **Speaker turns.** A line starting `Name:` starts a turn; Operator lines are ignored.
2. **Script vs live Q&A.** The call is split at the second `Operator:` line into prepared
   remarks (can be polished in advance) and the Q&A (live, harder to manage).
3. **Analyst vs executive.** After each Operator turn, the first speaker who did not appear in
   the prepared remarks is the analyst asking the question.
4. **FinBERT tone.** Every sentence is scored by
   [`yiyanghkust/finbert-tone`](https://huggingface.co/yiyanghkust/finbert-tone), a BERT model
   trained on financial text. Net tone = P(positive) − P(negative), from −1 to +1.
5. **Forward-looking statements.** A second model,
   [`yiyanghkust/finbert-fls`](https://huggingface.co/yiyanghkust/finbert-fls), marks each
   sentence as specific, non-specific or not forward-looking.
6. **Aggregation into 18 features**, in five groups:

| Group | Features | What it captures |
|---|---|---|
| Tone level (3) | `prep_net_tone`, `qa_exec_net_tone`, `analyst_tone` | how positive the script, the answers and the questions sound |
| Bad moments (3) | `neg_share`, `tone_p10`, `tone_std` | share of negative sentences, the worst 10%, how mixed the call is |
| Contrast (2) | `tone_gap`, `delta_tone` | script vs live Q&A; change in tone since the company's previous call |
| Specificity (2) | `specific_fls_share`, `fls_density` | how much, and how specifically, management talks about the future |
| Dictionary & style (8) | Loughran-McDonald uncertainty / modal / negative word shares, numbers, directness, length, Fog index | hedging, precision, readability |

7. **Checks** ([`phase2_check_features.py`](../phase2_check_features.py)): correlation heatmap
   (`results/feature_correlation.png`), duplicate measures removed, length checks, and GOOD vs BAD
   t-tests on calls before 2018 only, so the 2018–2025 holdout stays untouched.

## Key decisions
- **FinBERT, not a general sentiment tool.** The first version used VADER, built for social
  media. It misreads finance words such as *liability* or *tax*, and its whole-call score was
  +1.0 on 99.5% of calls. FinBERT improved every later result (reaction AUC 0.611 vs 0.588).
  The VADER pipeline is archived in [`archive/vader/`](../archive/vader/).
- **`delta_tone`:** some CEOs sound cheerful every quarter, so a drop from their own normal is
  the real news. It compares only with the previous call, so there is no look-ahead.
- **Stub transcripts** under 1,000 words (73 calls) are skipped.

## Example
Starbucks, 30 January 2024: `prep_net_tone` +0.489 (upbeat script), `analyst_tone` −0.046,
`delta_tone` −0.231 (a large fall since the previous quarter), `tone_std` 0.624 (a mixed call).

## Output
`data/features_finbert.csv`: 12,716 rows × 22 columns (ticker, call date, label, drift return and
the 18 features).

## Next
→ [Phase 3 — Type Classifier](Phase3_Type_Classifier.md)
