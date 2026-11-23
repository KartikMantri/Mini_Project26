# Archive: the earlier VADER pipeline

The project's first version (Velu's pipeline) built its language features with **VADER**, a
general-purpose sentiment tool. It was replaced by FinBERT, which reads financial text correctly
and improved every result. These files are kept for reference only and are **not part of the
current pipeline**.

| Comparison on the same holdout calls (2018–2025) | VADER (here) | FinBERT (current) |
|---|---|---|
| Phase 3 reaction AUC | 0.588 | 0.611 |
| Phase 4 drift gap, top − bottom 20% | +1.40% (p 0.059) | +1.62% (p 0.002) |
| Phase 5 market-neutral, after costs | +5.9%/yr, Sharpe 0.56 | +6.9%/yr, Sharpe 0.74 |

## Contents
| File | Phase | What it was |
|---|---|---|
| `extract_features_fast.py`, `_feat_worker.py` | 2 | Parallel VADER / dictionary / readability feature extraction → `data/features_expanded.csv` |
| `train_classifier.py` | 3 | Walk-forward classifier on the VADER features → `data/predictions.csv`, `data/text_scores.csv` |
| `feature_study.py` | 3 | Greedy feature-group selection for that classifier |
| `market_reaction.py` | 4 | First version of the market-reaction test |
| `backtest.py` | 5 | First backtest (compounded overlapping trades; superseded) |
| `data/` | | The VADER-era outputs listed above |
| `docs/` | | The VADER-era Phase 2/3 docs, the feature reference and the data-scaling decision |
| `feature_correlation_old_nonvader.png` | 2 | Old feature correlation chart |

The scripts were written for the repository root and use the old module names
(`event_returns`, `collect_data`); they will not run from this folder without changes.
