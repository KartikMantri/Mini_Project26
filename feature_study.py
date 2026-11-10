"""
Phase 3 — Feature study: which inputs actually improve prediction?
===================================================================
1. Scores every feature group on its own.
2. Greedy forward selection: start with nothing, add the group that raises validation
   AUC the most, and stop once no group adds at least MIN_GAIN.
3. Re-checks the chosen set on the holdout years, which played no part in choosing.

Selection uses only the validation years (2012-2017), so the holdout numbers
(2018-2025) are a fair estimate of how the chosen model does on new data.

Writes docxxs/Phase3_feature_study.md. Run after train_classifier.py has produced
data/text_scores.csv (or with master.csv available).

Usage:
    python feature_study.py
"""

from pathlib import Path

import numpy as np

from train_classifier import (GROUPS, HOLDOUT_YEARS, VALID_YEARS, attach_text_scores,
                              build_dataset, columns_for, evaluate, run_tabular)

OUT_MD   = Path("docxxs/Phase3_feature_study.md")
MIN_GAIN = 0.002   # smallest validation-AUC gain worth an extra group

CANDIDATES = {
    "y_imm":   ["sentiment", "lm", "style", "qa", "delta", "similarity", "text", "text_delta",
                "market"],
    "y_drift": ["sentiment", "lm", "style", "qa", "delta", "similarity", "text",
                "text_drift", "text_delta", "market", "reaction"],
}
TARGET_NAME = {"y_imm": "Reaction (day -1 to +1 up?)", "y_drift": "Type (drift day +2 to +60 up?)"}


def score(df, target, groups, kind, cache):
    key = (target, tuple(sorted(groups)), kind)
    if key not in cache:
        p = run_tabular(df, target, list(groups), kind)
        cache[key] = (evaluate(df, p, target, VALID_YEARS),
                      evaluate(df, p, target, HOLDOUT_YEARS))
    return cache[key]


def greedy(df, target, kind, cache, log):
    chosen, best = [], 0.5
    while True:
        trials = []
        for g in CANDIDATES[target]:
            if g in chosen:
                continue
            v, _ = score(df, target, chosen + [g], kind, cache)
            trials.append((v["auc"], g))
        if not trials:
            break
        auc, g = max(trials)
        log.append(f"| {kind} | +{g} | {auc:.4f} | {auc - best:+.4f} | "
                   f"{'kept' if auc - best >= MIN_GAIN else 'stop'} |")
        if auc - best < MIN_GAIN:
            break
        chosen.append(g)
        best = auc
    return chosen


def main():
    df = attach_text_scores(build_dataset())
    cache, lines = {}, [
        "# Phase 3 — Feature Study",
        "",
        "Which inputs actually help? Every number is out-of-sample (walk-forward by year).",
        f"Groups are chosen on the **validation years {VALID_YEARS[0]}–{VALID_YEARS[1]}** only;",
        f"the **holdout years {HOLDOUT_YEARS[0]}–{HOLDOUT_YEARS[1]}** show how the choice holds up on new data.",
        "",
        "Feature groups:",
        "",
    ]
    for g, cols in GROUPS.items():
        lines.append(f"- **{g}** ({len(cols)}): {', '.join(cols)}")

    for target in ("y_imm", "y_drift"):
        print(f"\n=== {TARGET_NAME[target]} ===")
        lines += ["", f"## {TARGET_NAME[target]}", "",
                  "### Each group on its own", "",
                  "| Group | # cols | LR valid AUC | XGB valid AUC | LR holdout AUC | XGB holdout AUC |",
                  "|---|---|---|---|---|---|"]
        for g in CANDIDATES[target]:
            (lv, lh), (xv, xh) = (score(df, target, [g], k, cache) for k in ("lr", "xgb"))
            print(f"  {g:<11s} LR {lv['auc']:.4f}/{lh['auc']:.4f}  XGB {xv['auc']:.4f}/{xh['auc']:.4f}")
            lines.append(f"| {g} | {len(GROUPS[g])} | {lv['auc']:.4f} | {xv['auc']:.4f} | "
                         f"{lh['auc']:.4f} | {xh['auc']:.4f} |")

        everything = CANDIDATES[target]
        lines += ["", "### All groups together vs. greedy selection", "",
                  "| Model | Groups | # cols | Valid AUC | Holdout AUC | Holdout accuracy | "
                  "Holdout drift rank-corr (t) |", "|---|---|---|---|---|---|---|"]
        log = []
        for kind in ("lr", "xgb"):
            chosen = greedy(df, target, kind, cache, log)
            for name, groups in (("all groups", everything), ("greedy pick", chosen)):
                if not groups:
                    continue
                v, h = score(df, target, groups, kind, cache)
                print(f"  {kind} {name:<11s} valid {v['auc']:.4f} holdout {h['auc']:.4f} "
                      f"[{', '.join(groups)}]")
                lines.append(f"| {kind} — {name} | {', '.join(groups)} | "
                             f"{len(columns_for(groups))} | {v['auc']:.4f} | {h['auc']:.4f} | "
                             f"{h['acc']:.3f} | {h['drift_ic']:+.4f} ({h['drift_ic_t']:+.1f}) |")
        lines += ["", "### Greedy selection steps (validation AUC)", "",
                  "| Model | Step | Valid AUC | Gain | Decision |", "|---|---|---|---|---|", *log]

    lines += ["", "---",
              "Regenerate with `python feature_study.py`. The chosen groups are copied into",
              "`FINAL_GROUPS` in `train_classifier.py`."]
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nsaved -> {OUT_MD}")


if __name__ == "__main__":
    main()
