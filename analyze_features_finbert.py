"""
Phase 2 (v2) — checks on the FinBERT features
==============================================
Runs after extract_features_finbert.py. Report only: nothing here chooses features.

  1. Spearman correlation heatmap of the 18 features -> results/feature_correlation.png,
     and every pair with |r| > 0.85
  2. The design hypotheses: tone block inter-correlated; uncertainty vs weak modals high;
     contrast/specificity features only weakly tied to the tone block
  3. Correlation of every feature with log_word_count (length artifacts)
  4. Welch t-test of each feature, GOOD vs BAD, on calls before HOLDOUT_START only, so the
     Phase 3 holdout years are not looked at
  5. Summary statistics (mean/min/max/nulls) for docxxs/features_finbert.md

Usage:
    python analyze_features_finbert.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from extract_features_finbert import FEATURES

FEATURES_CSV  = Path("data/features_finbert.csv")
HEATMAP_PNG   = Path("results/feature_correlation.png")
HOLDOUT_START = "2018-01-01"   # Phase 3 holdout years (2018-2025) stay unseen
HIGH_R        = 0.85

TONE_BLOCK     = ["prep_net_tone", "qa_exec_net_tone", "analyst_tone", "neg_share", "tone_p10", "tone_std"]
SHOULD_BE_WEAK = ["tone_gap", "delta_tone", "specific_fls_share", "numeric_density"]


def heatmap(corr, n_calls):
    HEATMAP_PNG.parent.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(FEATURES)), FEATURES, rotation=60, ha="right")
    ax.set_yticks(range(len(FEATURES)), FEATURES)
    for i in range(len(FEATURES)):
        for j in range(len(FEATURES)):
            v = corr.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7.5,
                    color="white" if abs(v) > 0.6 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Spearman r")
    ax.set_title(f"Spearman correlation: 18 FinBERT-era features ({n_calls:,} calls)")
    fig.tight_layout()
    fig.savefig(HEATMAP_PNG, dpi=150)
    print(f"saved {HEATMAP_PNG}")


def main():
    df = pd.read_csv(FEATURES_CSV, parse_dates=["call_date"])
    print(f"{len(df):,} calls x {len(FEATURES)} features, "
          f"{df['call_date'].min().date()} to {df['call_date'].max().date()}")

    # ---- 5. summary statistics ----
    print("\n=== Summary statistics ===")
    summary = pd.DataFrame({"mean": df[FEATURES].mean(), "min": df[FEATURES].min(),
                            "max": df[FEATURES].max(), "nulls": df[FEATURES].isna().sum()})
    print(summary.to_string(float_format=lambda v: f"{v:.4f}"))

    # ---- 1. correlation ----
    corr = df[FEATURES].corr(method="spearman")
    heatmap(corr, len(df))
    pairs = sorted(((FEATURES[i], FEATURES[j], corr.iloc[i, j])
                    for i in range(len(FEATURES)) for j in range(i + 1, len(FEATURES))),
                   key=lambda p: -abs(p[2]))
    print(f"\n=== Pairs with |r| > {HIGH_R} ===")
    high = [p for p in pairs if abs(p[2]) > HIGH_R]
    for a, b, r in high:
        print(f"  {a:<22} {b:<22} {r:+.3f}")
    if not high:
        print("  none")
    print("  strongest five overall:")
    for a, b, r in pairs[:5]:
        print(f"    {a:<22} {b:<22} {r:+.3f}")

    # ---- 2. hypotheses ----
    print("\n=== Hypotheses ===")
    block = corr.loc[TONE_BLOCK, TONE_BLOCK].abs().values[np.triu_indices(len(TONE_BLOCK), 1)]
    print(f"tone block |r|: median {np.median(block):.2f}, min {block.min():.2f}, max {block.max():.2f}")
    print(corr.loc[TONE_BLOCK, TONE_BLOCK].round(2).to_string())
    r_unc = corr.loc["lm_uncertainty_pct", "lm_weak_modal_pct"]
    print(f"\nlm_uncertainty_pct vs lm_weak_modal_pct: {r_unc:+.3f}")
    print("\nlargest |r| of each 'should be weak' feature with the tone block:")
    for f in SHOULD_BE_WEAK:
        row = corr.loc[f, TONE_BLOCK].abs()
        print(f"  {f:<20} max |r| {row.max():.2f} (with {row.idxmax()})")

    # ---- 3. length artifacts ----
    print("\n=== Correlation with log_word_count ===")
    length = corr["log_word_count"].drop("log_word_count").sort_values(key=abs, ascending=False)
    for f, r in length.items():
        flag = "  <- possible length artifact" if abs(r) > 0.3 else ""
        print(f"  {f:<22} {r:+.3f}{flag}")

    # ---- 4. GOOD vs BAD, pre-holdout only ----
    early = df[df["call_date"] < HOLDOUT_START]
    good, bad = early[early["label"] == "GOOD"], early[early["label"] == "BAD"]
    print(f"\n=== Welch t-test, GOOD vs BAD, calls before {HOLDOUT_START} "
          f"({len(good):,} GOOD / {len(bad):,} BAD) — report only ===")
    rows = []
    for f in FEATURES:
        g, b = good[f].dropna(), bad[f].dropna()
        t, p = stats.ttest_ind(g, b, equal_var=False)
        rows.append({"feature": f, "mean_GOOD": g.mean(), "mean_BAD": b.mean(), "t": t, "p": p})
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    main()
