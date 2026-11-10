# Phase 3 — Feature Study

Which inputs actually help? Every number is out-of-sample (walk-forward by year).
Groups are chosen on the **validation years 2012–2017** only;
the **holdout years 2018–2025** show how the choice holds up on new data.

Feature groups:

- **sentiment** (13): vader_pos, vader_neg, vader_neu, vader_prepared_pos, vader_prepared_neg, vader_qa_pos, vader_qa_neg, sent_compound_mean, sent_compound_std, sent_compound_min, sent_compound_max, pct_positive_sents, pct_negative_sents
- **lm** (4): lm_positive_pct, lm_negative_pct, lm_uncertainty_pct, lm_net_sentiment
- **style** (8): fog_index, flesch_ease, avg_sentence_length, numeric_density, word_count, sentence_count, forward_looking_count, forward_looking_density
- **qa** (3): qa_directness, qa_turn_count, qa_ratio
- **delta** (28): d_vader_pos, d_vader_neg, d_vader_neu, d_vader_prepared_pos, d_vader_prepared_neg, d_vader_qa_pos, d_vader_qa_neg, d_sent_compound_mean, d_sent_compound_std, d_sent_compound_min, d_sent_compound_max, d_pct_positive_sents, d_pct_negative_sents, d_lm_positive_pct, d_lm_negative_pct, d_lm_uncertainty_pct, d_lm_net_sentiment, d_fog_index, d_flesch_ease, d_avg_sentence_length, d_numeric_density, d_word_count, d_sentence_count, d_forward_looking_count, d_forward_looking_density, d_qa_directness, d_qa_turn_count, d_qa_ratio
- **similarity** (3): sim_prev_full, sim_prev_prepared, sim_prev_qa
- **text** (3): text_full, text_prepared, text_qa
- **text_drift** (3): text_drift_full, text_drift_prepared, text_drift_qa
- **market** (2): pre_ar, pre_vol
- **reaction** (1): imm_ar

## Reaction (day -1 to +1 up?)

### Each group on its own

| Group | # cols | LR valid AUC | XGB valid AUC | LR holdout AUC | XGB holdout AUC |
|---|---|---|---|---|---|
| sentiment | 13 | 0.5353 | 0.5229 | 0.5294 | 0.5322 |
| lm | 4 | 0.5580 | 0.5309 | 0.5427 | 0.5298 |
| style | 8 | 0.5176 | 0.5162 | 0.5183 | 0.5073 |
| qa | 3 | 0.5026 | 0.5177 | 0.5114 | 0.5099 |
| delta | 28 | 0.5735 | 0.5435 | 0.5666 | 0.5571 |
| similarity | 3 | 0.4969 | 0.4977 | 0.4992 | 0.5075 |
| text | 3 | 0.5557 | 0.5318 | 0.5746 | 0.5597 |
| market | 2 | 0.5111 | 0.5259 | 0.4765 | 0.5022 |

### All groups together vs. greedy selection

| Model | Groups | # cols | Valid AUC | Holdout AUC | Holdout accuracy | Holdout drift rank-corr (t) |
|---|---|---|---|---|---|---|
| lr — all groups | sentiment, lm, style, qa, delta, similarity, text, market | 64 | 0.5912 | 0.5883 | 0.564 | +0.0622 (+3.5) |
| lr — greedy pick | delta, text, lm, market | 37 | 0.5953 | 0.5875 | 0.567 | +0.0565 (+3.3) |
| xgb — all groups | sentiment, lm, style, qa, delta, similarity, text, market | 64 | 0.5749 | 0.5863 | 0.558 | +0.0500 (+3.0) |
| xgb — greedy pick | delta, market, text, lm | 37 | 0.5737 | 0.5807 | 0.555 | +0.0495 (+2.3) |

### Greedy selection steps (validation AUC)

| Model | Step | Valid AUC | Gain | Decision |
|---|---|---|---|---|
| lr | +delta | 0.5735 | +0.0735 | kept |
| lr | +text | 0.5856 | +0.0121 | kept |
| lr | +lm | 0.5902 | +0.0046 | kept |
| lr | +market | 0.5953 | +0.0052 | kept |
| lr | +similarity | 0.5943 | -0.0010 | stop |
| xgb | +delta | 0.5435 | +0.0435 | kept |
| xgb | +market | 0.5544 | +0.0109 | kept |
| xgb | +text | 0.5618 | +0.0075 | kept |
| xgb | +lm | 0.5737 | +0.0119 | kept |
| xgb | +style | 0.5740 | +0.0003 | stop |

## Type (drift day +2 to +60 up?)

### Each group on its own

| Group | # cols | LR valid AUC | XGB valid AUC | LR holdout AUC | XGB holdout AUC |
|---|---|---|---|---|---|
| sentiment | 13 | 0.5017 | 0.5155 | 0.5161 | 0.5085 |
| lm | 4 | 0.5274 | 0.5037 | 0.5228 | 0.5175 |
| style | 8 | 0.4886 | 0.5221 | 0.5249 | 0.5047 |
| qa | 3 | 0.4814 | 0.4965 | 0.5341 | 0.5191 |
| delta | 28 | 0.4887 | 0.4973 | 0.5136 | 0.5029 |
| similarity | 3 | 0.4923 | 0.4834 | 0.5239 | 0.5113 |
| text | 3 | 0.5126 | 0.4836 | 0.5363 | 0.5123 |
| text_drift | 3 | 0.4942 | 0.5214 | 0.5132 | 0.5029 |
| market | 2 | 0.5014 | 0.5027 | 0.5190 | 0.5025 |
| reaction | 1 | 0.5183 | 0.5168 | 0.5217 | 0.5083 |

### All groups together vs. greedy selection

| Model | Groups | # cols | Valid AUC | Holdout AUC | Holdout accuracy | Holdout drift rank-corr (t) |
|---|---|---|---|---|---|---|
| lr — all groups | sentiment, lm, style, qa, delta, similarity, text, text_drift, market, reaction | 68 | 0.5106 | 0.5262 | 0.514 | +0.0487 (+1.7) |
| lr — greedy pick | lm, reaction | 5 | 0.5357 | 0.5212 | 0.492 | +0.0215 (+1.3) |
| xgb — all groups | sentiment, lm, style, qa, delta, similarity, text, text_drift, market, reaction | 68 | 0.5180 | 0.5167 | 0.503 | +0.0393 (+1.3) |
| xgb — greedy pick | style, reaction, lm, market | 15 | 0.5343 | 0.5070 | 0.495 | +0.0082 (+0.3) |

### Greedy selection steps (validation AUC)

| Model | Step | Valid AUC | Gain | Decision |
|---|---|---|---|---|
| lr | +lm | 0.5274 | +0.0274 | kept |
| lr | +reaction | 0.5357 | +0.0083 | kept |
| lr | +market | 0.5327 | -0.0030 | stop |
| xgb | +style | 0.5221 | +0.0221 | kept |
| xgb | +reaction | 0.5268 | +0.0047 | kept |
| xgb | +lm | 0.5304 | +0.0037 | kept |
| xgb | +market | 0.5343 | +0.0039 | kept |
| xgb | +qa | 0.5311 | -0.0033 | stop |

---
Regenerate with `python feature_study.py`. The chosen groups are copied into
`FINAL` in `train_classifier.py`.

## Later addition: `text_delta` (2026-11-07)

A TF-IDF model on the *change* in wording vs. the firm's previous call (this call's TF-IDF
minus the previous call's), trained walk-forward on the immediate reaction. Added to the
greedy pick as one more step, judged by the same rule:

| Inputs | Valid AUC | Gain | Holdout AUC |
|---|---|---|---|
| delta, text, lm, market (greedy pick) | 0.5953 | — | 0.5875 |
| + text_delta | 0.5974 | +0.0021 (above the 0.002 bar) | 0.5962 |

Other ideas tested the same way and **not** kept (holdout AUC with the greedy pick):
stronger/weaker regularisation (0.587–0.589), averaging Logistic Regression with XGBoost
(0.588), training only on moves above 1–2% (0.588–0.590), and features scaled against the
firm's own history (0.572–0.583).

The tables above predate `text_delta`; `text_delta` is now in `CANDIDATES`, so a rerun of
this script includes it from the start.
