# Phase 2 (v2) — FinBERT Feature Reference

> **Status: complete (2026-11-11).** Full run over all 12,716 calls done; statistics, correlation
> checks and GOOD vs BAD tests are at the end of this file. Checks script:
> [`analyze_features_finbert.py`](../analyze_features_finbert.py).

Reference for all **22 columns** in `data/features_finbert.csv`, produced by
[`extract_features_finbert.py`](../extract_features_finbert.py) from `data/calls_all.csv`
(built by [`build_call_dataset.py`](../build_call_dataset.py)). Replaces the VADER-based
[`features.md`](features.md) / `data/features_expanded.csv`.

> **Dataset:** 12,789 earnings calls across 198 S&P 500 tickers, 2005-10 → 2025-05, including the
> 2,296 calls Phase 1 labeled NEUTRAL. 73 stub transcripts (< 1,000 words) are skipped, leaving
> 12,716 rows.

---

## 1. Identity Columns (2)

| Column | Type | Description |
|--------|------|-------------|
| `ticker` | string | Stock ticker symbol (e.g. `AAPL`) |
| `call_date` | datetime | Date/time of the earnings call |

---

## 2. Tone Level — FinBERT (3)

Model: [`yiyanghkust/finbert-tone`](https://huggingface.co/yiyanghkust/finbert-tone), trained on
financial text including earnings calls. Every sentence is scored; **net tone = P(positive) −
P(negative)**, range [−1, +1]. Label order read from the model config: 0 Neutral, 1 Positive,
2 Negative. The model is very confident, so most sentences score close to −1, 0 or +1.

| Column | Description |
|--------|-------------|
| `prep_net_tone` | Mean net tone over prepared-remarks sentences |
| `qa_exec_net_tone` | Mean net tone over executive answer sentences (falls back to all Q&A sentences if speaker roles can't be tagged) |
| `analyst_tone` | Mean net tone over analyst question sentences |

---

## 3. Bad-Moment Detection — FinBERT, Whole Call (3)

| Column | Description |
|--------|-------------|
| `neg_share` | Share of sentences with P(negative) > 0.5 |
| `tone_p10` | 10th percentile of sentence net tone (not the minimum, which grows more extreme with call length) |
| `tone_std` | Standard deviation of sentence net tone |

---

## 4. Contrast — FinBERT (2)

| Column | Description |
|--------|-------------|
| `tone_gap` | `prep_net_tone` − mean net tone of all Q&A sentences. Positive = scripted remarks sound better than live answers |
| `delta_tone` | Whole-call mean net tone minus the **same ticker's previous call**. Blank for each ticker's first call |

---

## 5. Specificity — FinBERT-FLS, Whole Call (2)

Model: [`yiyanghkust/finbert-fls`](https://huggingface.co/yiyanghkust/finbert-fls). Label order
from the config: 0 Not FLS, 1 Non-specific FLS, 2 Specific FLS. Each sentence gets its most
likely class.

| Column | Description |
|--------|-------------|
| `specific_fls_share` | Specific FLS / (Specific + Non-specific FLS). Blank if a call has no forward-looking sentences |
| `fls_density` | (Specific + Non-specific FLS) / all sentences |

Known quirk: the model sometimes calls scheduling statements ("our next call is in April")
Specific FLS, and vague optimism ("we remain optimistic about the long term") Not FLS.

---

## 6. Dictionary and Style — Whole Call (8)

Computed on everything spoken, with Operator turns excluded. Loughran-McDonald word lists come from
the full dictionary bundled with `pysentiment2` (`LM.csv`), not the short hand-typed lists the
VADER pipeline used.

| Column | Description |
|--------|-------------|
| `lm_uncertainty_pct` | Share of words in the LM Uncertainty list (297 words) |
| `lm_weak_modal_pct` | Share of LM weak modal words (`Modal == 3`, 27 words, e.g. "may", "could") |
| `lm_strong_modal_pct` | Share of LM strong modal words (`Modal == 1`, 19 words, e.g. "will", "must") |
| `lm_negative_pct` | Share of words in the LM Negative list (2,355 words) |
| `numeric_density` | Share of words containing a digit |
| `qa_directness` | Share of executive answer turns that contain a number. Blank if roles can't be tagged |
| `log_word_count` | Natural log of the number of words spoken (length control) |
| `fog_index` | Gunning Fog readability index. Kept as the single readability measure: `flesch_ease` (r = −0.92) and `avg_sentence_length` (r = 0.78) duplicated it in the old feature set |

---

## 7. Target Columns (2) — never use as model inputs

| Column | Description |
|--------|-------------|
| `label` | `GOOD` if `abnormal_return` > 0, else `BAD` |
| `abnormal_return` | Stock return minus S&P 500 return over trading days +2 → +60, with day 0 = the call's calendar date (`event_returns.py`) |

This differs from Phase 1's label (±2% threshold, NEUTRAL calls dropped, day 0 one trading day late
for timed calls). For the 10,493 calls in both, 97.7% of labels agree.

---

## Text Handling

1. **Speaker turns.** A line starting `Name:` begins a turn. A label counts as a speaker if it
   starts with a capital letter, has no `?` or `!`, and is either short (≤ 6 words) or in
   `Name - Title, Company` form (≤ 25 words). Header lists (`Executives:`, `Analysts:`, …) and
   Operator turns are not scored.
2. **Prepared remarks vs Q&A.** Split at the 2nd `Operator:` (the existing rule; correct for
   roughly 94–97% of calls). If fewer than two `Operator:` appear, the whole call is prepared
   remarks. If the split leaves under 2% of the text as prepared remarks (e.g. a transcript starting
   `Operator: Operator:`), it is treated as failed and the section features are blank.
3. **Analyst vs executive.** After each Operator turn, the first speaker who did **not** speak in
   the prepared remarks is the analyst; later turns by that person are analyst turns, and other
   speakers are executives. This stops investor-relations staff who introduce the next questioner
   from being tagged as analysts.
4. **Transcripts' own `Q -` / `A -` markup.** Some older transcripts mark every turn `Q - Name` /
   `A - Name`. That markup is used instead of rule 3, but **only when a call has at least 3 `Q -`
   and at least 3 `A -` labels**. A single stray label (one Costco call has one `Q – Laura
   Champine`) would otherwise switch a whole call to markup mode and leave every other turn
   untagged.
5. **Sentences.** Split on sentence-ending punctuation and line breaks. Sentences under 4 words are
   dropped. Each is scored with `max_length = 128` tokens.

**Tagging accuracy:** on the 76 calls with full `Q -` / `A -` markup, rule 3 (run with the markup
removed) agrees with the markup on **96.1%** of Q&A sentences (12,488 of 12,989).

---

## 50-Call Trial (2026-11-11, RTX 4050 Laptop GPU)

50 calls: 5 tickers × 10 consecutive calls (seed 42), so `delta_tone` could be checked.

- **Speed:** 20,423 sentences (median 392 per call) in 29.6 s → **0.59 s per call**, about
  **2.1 hours** for all 12,716 calls.
- **Coverage:** Q&A split worked on 50 of 50 calls; analyst/executive roles tagged on 50 of 50.
- **Blanks:** only `delta_tone`, on each ticker's first call (5), as expected.

| Feature | Mean | Min | Max | Nulls |
|---|---|---|---|---|
| `prep_net_tone` | 0.279 | 0.022 | 0.587 | 0 |
| `qa_exec_net_tone` | 0.153 | −0.018 | 0.326 | 0 |
| `analyst_tone` | −0.018 | −0.165 | 0.122 | 0 |
| `neg_share` | 0.092 | 0.020 | 0.167 | 0 |
| `tone_p10` | −0.396 | −1.000 | −0.008 | 0 |
| `tone_std` | 0.547 | 0.404 | 0.671 | 0 |
| `tone_gap` | 0.181 | −0.044 | 0.412 | 0 |
| `delta_tone` | 0.004 | −0.186 | 0.178 | 5 |
| `specific_fls_share` | 0.649 | 0.222 | 0.895 | 0 |
| `fls_density` | 0.181 | 0.063 | 0.264 | 0 |
| `lm_uncertainty_pct` | 0.0083 | 0.0052 | 0.0151 | 0 |
| `lm_weak_modal_pct` | 0.0038 | 0.0017 | 0.0067 | 0 |
| `lm_strong_modal_pct` | 0.0054 | 0.0024 | 0.0092 | 0 |
| `lm_negative_pct` | 0.0087 | 0.0053 | 0.0158 | 0 |
| `numeric_density` | 0.026 | 0.015 | 0.043 | 0 |
| `qa_directness` | 0.459 | 0.147 | 0.842 | 0 |
| `log_word_count` | 9.03 | 8.48 | 9.47 | 0 |
| `fog_index` | 12.15 | 9.35 | 15.11 | 0 |

Early sanity: prepared remarks sound more positive than live answers (mean `tone_gap` +0.18), and
analyst questions are close to neutral.

**Bugs found and fixed in the trial:**
- Long `Name - Title` speaker labels were not recognised, so executives' speech was merged into the
  previous turn.
- A single stray `Q –` label switched a whole call to markup mode (now needs ≥ 3 `Q -` and ≥ 3 `A -`).

---

## Full Run (2026-11-11, RTX 4050 Laptop GPU)

All 12,716 non-stub calls, 198 tickers, 2005-10-31 → 2025-05-15. About 1.1 s per call in practice
(the 0.59 s trial estimate was optimistic); the laptop also paused once in standby, which the
per-ticker checkpoints survived.

| Feature | Mean | Min | Max | Nulls |
|---|---|---|---|---|
| `prep_net_tone` | 0.349 | −0.997 | 0.813 | 27 |
| `qa_exec_net_tone` | 0.215 | −0.245 | 0.740 | 330 |
| `analyst_tone` | 0.011 | −0.427 | 0.618 | 330 |
| `neg_share` | 0.090 | 0.000 | 0.354 | 4 |
| `tone_p10` | −0.357 | −1.000 | −0.0001 | 4 |
| `tone_std` | 0.567 | 0.316 | 0.804 | 4 |
| `tone_gap` | 0.194 | −1.166 | 0.736 | 332 |
| `delta_tone` | 0.001 | −0.487 | 0.434 | 204 |
| `specific_fls_share` | 0.648 | 0.134 | 1.000 | 4 |
| `fls_density` | 0.157 | 0.025 | 0.368 | 4 |
| `lm_uncertainty_pct` | 0.0089 | 0 | 0.0228 | 0 |
| `lm_weak_modal_pct` | 0.0041 | 0 | 0.0104 | 0 |
| `lm_strong_modal_pct` | 0.0053 | 0 | 0.0149 | 0 |
| `lm_negative_pct` | 0.0089 | 0 | 0.0350 | 0 |
| `numeric_density` | 0.023 | 0 | 0.157 | 0 |
| `qa_directness` | 0.453 | 0 | 1.000 | 328 |
| `log_word_count` | 9.07 | 0 | 10.65 | 0 |
| `fog_index` | 12.52 | 8.52 | 19.40 | 4 |

**Where the blanks come from:**
- **4 unparseable transcripts** (AMD 2006-01-19, MSFT 2006-01-27, PEP 2009-02-13, PLD 2010-07-30):
  the source text has no line breaks between speakers, so no turns are found. Every FinBERT feature
  is blank and the word-based features are 0 (`log_word_count` = 0). Drop these 4 rows in Phase 3.
- **~330 calls without analyst/executive roles** (`qa_exec_net_tone`, `analyst_tone`,
  `qa_directness` blank), mostly calls with no detectable Q&A.
- **27 calls with a failed prepared/Q&A split** (`prep_net_tone`, `tone_gap` blank).
- **`delta_tone`**: each ticker's first call (198), plus calls following an unparseable one.

---

## Correlation Checks (Step 5)

Spearman correlation over all 12,716 calls (labels not used). Heatmap:
[`results/feature_correlation.png`](../results/feature_correlation.png).

**Only one pair has |r| > 0.85: `neg_share` vs `tone_p10`, r = −0.97.** They carry almost the
same information (a call with many negative sentences has a very low 10th percentile). Next
strongest: `prep_net_tone`–`tone_gap` 0.77, `neg_share`–`tone_std` 0.73, `tone_p10`–`tone_std`
−0.68, `neg_share`–`lm_negative_pct` 0.62.

| Hypothesis | Result | Verdict |
|---|---|---|
| Tone block (features 1–6) strongly inter-correlated | Median \|r\| 0.53. The three level features correlate 0.33–0.55 with each other and ~0.5 with `neg_share`/`tone_p10`. But `tone_std` is unrelated to the level features (r −0.00, 0.06, −0.28); it moves with `neg_share` (0.73) instead | **Partly held** — two sub-blocks: *level* and *spread* |
| `lm_uncertainty_pct` vs `lm_weak_modal_pct` high | r = 0.61 | **Held, but moderate** — not a duplicate |
| `tone_gap` weak vs tone block | max \|r\| 0.77 with `prep_net_tone` | **Did not hold.** Partly mechanical: `tone_gap` = `prep_net_tone` − Q&A tone, and prepared tone varies more than Q&A tone |
| `delta_tone` weak vs tone block | max \|r\| 0.31 (`qa_exec_net_tone`) | **Held** |
| `specific_fls_share` weak vs tone block | max \|r\| 0.32 (`qa_exec_net_tone`) | **Held** |
| `numeric_density` weak vs tone block | max \|r\| 0.11 | **Held** |

FinBERT tone also agrees with the dictionary: `lm_negative_pct` correlates 0.62 with `neg_share`
and −0.47 with `prep_net_tone`, which is a useful sanity check that both measure negativity.

### Length artifacts

Every feature's |r| with `log_word_count` is **below 0.12** (largest: `tone_std` 0.12,
`neg_share` 0.10, `tone_p10` −0.10). No length artifacts: using shares, means and the 10th
percentile instead of counts and the minimum worked.

---

## GOOD vs BAD — Welch t-test (report only, not used to choose features)

Calls **before 2018-01-01 only** (3,903 GOOD / 3,035 BAD), so the Phase 3 holdout years
(2018–2025) are not looked at.

| Feature | Mean GOOD | Mean BAD | t | p |
|---|---|---|---|---|
| `qa_exec_net_tone` | 0.1950 | 0.1861 | 3.23 | **0.001** |
| `prep_net_tone` | 0.3266 | 0.3155 | 2.83 | **0.005** |
| `tone_p10` | −0.4047 | −0.4306 | 2.76 | **0.006** |
| `lm_weak_modal_pct` | 0.0041 | 0.0040 | 2.26 | **0.024** |
| `neg_share` | 0.0974 | 0.1000 | −2.16 | **0.031** |
| `delta_tone` | 0.0022 | −0.0004 | 1.38 | 0.168 |
| `tone_gap` | 0.1909 | 0.1864 | 1.37 | 0.170 |
| `lm_uncertainty_pct` | 0.0092 | 0.0091 | 0.99 | 0.324 |
| `analyst_tone` | −0.0050 | −0.0065 | 0.79 | 0.429 |
| `qa_directness` | 0.4311 | 0.4338 | −0.72 | 0.471 |
| `lm_negative_pct` | 0.0093 | 0.0093 | −0.51 | 0.610 |
| `numeric_density` | 0.0243 | 0.0243 | −0.45 | 0.653 |
| `fog_index` | 12.53 | 12.51 | 0.41 | 0.685 |
| `tone_std` | 0.5698 | 0.5704 | −0.40 | 0.689 |
| `lm_strong_modal_pct` | 0.0054 | 0.0055 | −0.35 | 0.726 |
| `log_word_count` | 9.081 | 9.084 | −0.33 | 0.739 |
| `specific_fls_share` | 0.6516 | 0.6506 | 0.33 | 0.740 |
| `fls_density` | 0.1587 | 0.1586 | 0.04 | 0.965 |

**Reading it honestly:**
- The FinBERT tone features point the expected way: GOOD calls sound slightly more positive, in
  both the prepared remarks and the executives' live answers, and have fewer negative moments.
  The old VADER whole-call compound could not show this at all (it was +1.0 on 99.5% of calls).
- The differences are **tiny** (Cohen's d ≈ 0.08 at most). With 18 tests, only
  `qa_exec_net_tone` passes a Bonferroni cut-off (0.05 / 18 = 0.0028).
- Expect a small improvement in Phase 3, not a large one. Tone is a weak signal for +2 → +60 day
  drift, as the literature also finds.
