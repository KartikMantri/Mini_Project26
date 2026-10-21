"""
Phase 4 — Market-Reaction Test
================================
Compares immediate (day 0→+1) vs drift (day +2→+60) abnormal returns
against Phase 3's predicted confidence, via Welch's t-test.

Reads: data/predictions.csv, data/prices.csv, data/master.csv
Writes: data/market_reaction.csv, docxxs/Phase4_results.md

Usage:
    python market_reaction.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

PREDICTIONS_CSV     = Path("data/predictions.csv")
PRICES_CSV          = Path("data/prices.csv")
MASTER_CSV          = Path("data/master.csv")
MARKET_REACTION_CSV = Path("data/market_reaction.csv")
RESULTS_MD          = Path("docxxs/Phase4_results.md")

MARKET_INDEX  = "^GSPC"
IMMEDIATE_WIN = (0, 1)   # day 0 → day +1
DRIFT_WIN     = (2, 60)  # day +2 → day +60  (matches Phase 1 labelling window)


def price_at_offset(series, call_date, offset):
    idx = series.index.searchsorted(call_date)
    target = idx + offset
    if target >= len(series):
        return None
    return series.iloc[target]


def compute_return(ticker, call_date, prices, window):
    if ticker not in prices.columns or MARKET_INDEX not in prices.columns:
        return None
    stock  = prices[ticker].dropna()
    market = prices[MARKET_INDEX].dropna()
    s0 = price_at_offset(stock,  call_date, window[0])
    s1 = price_at_offset(stock,  call_date, window[1])
    m0 = price_at_offset(market, call_date, window[0])
    m1 = price_at_offset(market, call_date, window[1])
    if None in (s0, s1, m0, m1) or s0 == 0 or m0 == 0:
        return None
    return (s1 / s0 - 1) - (m1 / m0 - 1)


def ttest(group_a, group_b, label_a="High conf GOOD", label_b="High conf BAD"):
    a = np.array(group_a, dtype=float)
    b = np.array(group_b, dtype=float)
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]
    t, p = stats.ttest_ind(a, b, equal_var=False)
    print(f"  {label_a}: n={len(a)}, mean={a.mean()*100:.2f}%")
    print(f"  {label_b}: n={len(b)}, mean={b.mean()*100:.2f}%")
    print(f"  t={t:.3f}, p={p:.4f} {'*** SIGNIFICANT' if p < 0.05 else '(not significant)'}")
    return t, p, a.mean(), b.mean(), len(a), len(b)


def main():
    print("[Phase4] Loading data ...")
    preds  = pd.read_csv(PREDICTIONS_CSV, parse_dates=["call_date"])
    prices = pd.read_csv(PRICES_CSV, index_col=0, parse_dates=True)
    master = pd.read_csv(MASTER_CSV, parse_dates=["call_date"])

    # Use XGBoost predictions if available, else LR
    conf_col = "xgb_confidence_good" if "xgb_confidence_good" in preds.columns \
               else "lr_confidence_good"
    pred_col = "xgb_predicted" if "xgb_predicted" in preds.columns else "lr_predicted"
    print(f"[Phase4] Using predictions from: {pred_col}")

    # Compute immediate and drift returns for each call
    records = []
    for _, row in preds.iterrows():
        imm  = compute_return(row["ticker"], row["call_date"], prices, IMMEDIATE_WIN)
        drift = compute_return(row["ticker"], row["call_date"], prices, DRIFT_WIN)
        records.append({
            "ticker":          row["ticker"],
            "call_date":       row["call_date"],
            "actual_label":    row["label"],
            "predicted":       row[pred_col],
            "confidence_good": row[conf_col],
            "immediate_ar":    imm,
            "drift_ar":        drift,
        })

    df = pd.DataFrame(records)
    df.to_csv(MARKET_REACTION_CSV, index=False)
    print(f"[Phase4] Saved {len(df)} rows -> {MARKET_REACTION_CSV}")

    # High-confidence subsets (top/bottom 30% by confidence)
    hi_thresh = df["confidence_good"].quantile(0.70)
    lo_thresh = df["confidence_good"].quantile(0.30)
    hi_good = df[df["confidence_good"] >= hi_thresh]
    hi_bad  = df[df["confidence_good"] <= lo_thresh]

    results = {}
    print("\n--- IMMEDIATE window (day 0→+1) ---")
    t, p, ma, mb, na, nb = ttest(
        hi_good["immediate_ar"], hi_bad["immediate_ar"],
        "High-conf GOOD", "High-conf BAD")
    results["imm"] = dict(t=t, p=p, mean_good=ma, mean_bad=mb, n_good=na, n_bad=nb)

    print("\n--- DRIFT window (day +2→+60) ---")
    t, p, ma, mb, na, nb = ttest(
        hi_good["drift_ar"], hi_bad["drift_ar"],
        "High-conf GOOD", "High-conf BAD")
    results["drift"] = dict(t=t, p=p, mean_good=ma, mean_bad=mb, n_good=na, n_bad=nb)

    # Write results markdown
    lines = [
        "# Phase 4 — Market-Reaction Test",
        "",
        f"**Classifier used:** {pred_col}  ",
        f"**Confidence threshold:** top/bottom 30% (≥{hi_thresh:.3f} / ≤{lo_thresh:.3f})",
        "",
        "## Immediate Window (day 0 → +1)",
        "",
        f"| Group | N | Mean Abnormal Return |",
        f"|-------|---|---------------------|",
        f"| High-conf GOOD | {results['imm']['n_good']} | {results['imm']['mean_good']*100:.2f}% |",
        f"| High-conf BAD  | {results['imm']['n_bad']}  | {results['imm']['mean_bad']*100:.2f}%  |",
        f"",
        f"Welch's t-test: t={results['imm']['t']:.3f}, **p={results['imm']['p']:.4f}**  ",
        f"{'✅ Statistically significant (p<0.05)' if results['imm']['p']<0.05 else '❌ Not significant (p≥0.05)'}",
        "",
        "## Drift Window (day +2 → +60)",
        "",
        f"| Group | N | Mean Abnormal Return |",
        f"|-------|---|---------------------|",
        f"| High-conf GOOD | {results['drift']['n_good']} | {results['drift']['mean_good']*100:.2f}% |",
        f"| High-conf BAD  | {results['drift']['n_bad']}  | {results['drift']['mean_bad']*100:.2f}%  |",
        f"",
        f"Welch's t-test: t={results['drift']['t']:.3f}, **p={results['drift']['p']:.4f}**  ",
        f"{'✅ Statistically significant (p<0.05)' if results['drift']['p']<0.05 else '❌ Not significant (p≥0.05)'}",
        "",
        "---",
        "See `data/market_reaction.csv` for full per-call results.",
    ]
    RESULTS_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"[Phase4] Saved results -> {RESULTS_MD}")
    print("\n[Phase4] DONE. Next: python backtest.py")


if __name__ == "__main__":
    main()
