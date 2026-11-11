"""
Phase 3 (v2) — Type Classifier on the FinBERT features
=======================================================
Same method as train_classifier.py (Velu's walk-forward), with the VADER features replaced by
the 18 FinBERT/LM features in data/features_finbert.csv and the full 12,712-call set.

Two evaluation setups:
  MAIN   walk-forward: each year is predicted by a model trained only on earlier years.
         Inputs and model type are chosen greedily on the validation years 2012-2017 only;
         2018-2025 is the holdout, looked at once at the end.
  CHECK  Kartik's fixed split: train once on calls before 2022 (leaving a 61-trading-day gap
         so no training label uses 2022 prices), test once on 2022-2025.

The main model learns to predict the IMMEDIATE reaction (day -1 -> +1); its probability is the
GOOD/BAD score. GOOD/BAD = drift (day +2 -> +60 abnormal return) > 0. The comparison rows include
a model trained straight on GOOD/BAD.

Outputs:
  data/text_scores_finbert.csv        out-of-sample TF-IDF text scores (cache)
  data/predictions_finbert.csv        out-of-sample predictions, 2012-2025
  docxxs/Phase3_finbert_results.md    results, incl. comparison with Velu's VADER run

Usage:
    python train_classifier_finbert.py
    python train_classifier_finbert.py --rebuild-text
"""

import argparse

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline

from event_returns import BAD_PRICE_TICKERS, PRE_WIN, EventReturns, load_prices
from extract_features_finbert import FEATURES
from train_classifier import (HOLDOUT_YEARS, TEXT_FIRST_TEST_YEAR, FIRST_TEST_YEAR,
                              VALID_YEARS, TABLE_HEAD, TextModel, _split_sections,
                              evaluate, fmt_row, make_tabular, prev_call_similarity,
                              walk_forward)
from pathlib import Path

FEATURES_CSV    = Path("data/features_finbert.csv")
CALLS_CSV       = Path("data/calls_all.csv")
TEXT_SCORES_CSV = Path("data/text_scores_finbert.csv")
PREDICTIONS_CSV = Path("data/predictions_finbert.csv")
VELU_PRED_CSV   = Path("data/predictions.csv")          # Velu's VADER run (baseline)
CACHE_DIR       = Path("data/_cache")
RESULTS_MD      = Path("docxxs/Phase3_finbert_results.md")

MIN_GAIN        = 0.002          # same rule as feature_study.py
FIXED_TEST_START = "2022-01-01"  # Kartik's split
LABEL_GAP_DAYS   = 61            # trading days

DELTA_SRC = [f for f in FEATURES if f != "delta_tone"]   # delta_tone is already a change

GROUPS = {
    "tone":       ["prep_net_tone", "qa_exec_net_tone", "analyst_tone", "neg_share",
                   "tone_p10", "tone_std", "tone_gap", "delta_tone"],
    "fls":        ["specific_fls_share", "fls_density"],
    "lm":         ["lm_uncertainty_pct", "lm_weak_modal_pct", "lm_strong_modal_pct",
                   "lm_negative_pct"],
    "style":      ["numeric_density", "qa_directness", "log_word_count", "fog_index"],
    "delta":      ["d_" + c for c in DELTA_SRC],
    "similarity": ["sim_prev_full", "sim_prev_prepared", "sim_prev_qa"],
    "text":       ["text_full", "text_prepared", "text_qa"],
    "text_drift": ["text_drift_full", "text_drift_prepared", "text_drift_qa"],
    "market":     ["pre_ar", "pre_vol"],
    "reaction":   ["imm_ar"],
}
FINBERT_GROUPS = ["tone", "fls", "lm", "style", "delta"]
CANDIDATES = ["tone", "fls", "lm", "style", "delta", "similarity", "text", "market"]


CLIP_COLS = ["pre_ar", "pre_vol", "imm_ar"]   # return inputs with fat tails
CLIP_Q    = (0.01, 0.99)


def cols_for(groups):
    return [c for g in groups for c in GROUPS[g]]


class TrainClipper(BaseEstimator, TransformerMixin):
    """Caps the given columns at the 1st / 99th percentile of the TRAINING rows only,
    so a few extreme returns cannot dominate the fit and no future data sets the caps."""

    def __init__(self, idx=()):
        self.idx = idx

    def fit(self, X, y=None):
        idx = list(self.idx)
        self.lo_ = np.nanquantile(X[:, idx], CLIP_Q[0], axis=0) if idx else []
        self.hi_ = np.nanquantile(X[:, idx], CLIP_Q[1], axis=0) if idx else []
        return self

    def transform(self, X):
        X = np.array(X, dtype=float, copy=True)
        for j, lo, hi in zip(self.idx, self.lo_, self.hi_):
            X[:, j] = np.clip(X[:, j], lo, hi)   # NaN stays NaN
        return X


def make_model(kind, cols):
    idx = tuple(i for i, c in enumerate(cols) if c in CLIP_COLS)
    return Pipeline([("clip", TrainClipper(idx)), ("model", make_tabular(kind))])


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def build_dataset():
    df = pd.read_csv(FEATURES_CSV, parse_dates=["call_date"])
    bad = df["log_word_count"] <= 0           # 4 transcripts with no line breaks
    print(f"[data] dropping {bad.sum()} unparseable transcripts")
    df = df[~bad]

    calls = pd.read_csv(CALLS_CSV, usecols=["ticker", "call_date", "imm_ar", "drift_ar"],
                        parse_dates=["call_date"])
    df = df.merge(calls, on=["ticker", "call_date"], how="left", validate="one_to_one")
    assert np.allclose(df["drift_ar"], df["abnormal_return"], equal_nan=True)
    # calls_all.csv was built from the bad PARA prices: keep the calls (their text feeds
    # the change-since-last-call features) but never use them as labels or score them.
    bad_px = df["ticker"].isin(BAD_PRICE_TICKERS)
    print(f"[data] blanking returns for {bad_px.sum()} calls with bad prices "
          f"({', '.join(sorted(BAD_PRICE_TICKERS))})")
    df.loc[bad_px, ["imm_ar", "drift_ar"]] = np.nan

    er = EventReturns(load_prices())
    keys = list(zip(df["ticker"], df["call_date"]))
    df["pre_ar"]  = [er.abnormal(t, d, PRE_WIN) for t, d in keys]
    df["pre_vol"] = [er.volatility(t, d, PRE_WIN) for t, d in keys]

    df = df.sort_values(["ticker", "call_date"])
    prev = df.groupby("ticker")[DELTA_SRC].shift(1)
    for c in DELTA_SRC:
        df["d_" + c] = df[c] - prev[c]

    df = df.sort_values(["call_date", "ticker"]).reset_index(drop=True)
    df["row_id"]  = np.arange(len(df))
    df["year"]    = df["call_date"].dt.year
    df["y_imm"]   = (df["imm_ar"] > 0).astype(float).where(df["imm_ar"].notna())
    df["y_drift"] = (df["drift_ar"] > 0).astype(float).where(df["drift_ar"].notna())
    return df


# ---------------------------------------------------------------------------
# Text scores (same TF-IDF model as train_classifier.py, on calls_all.csv)
# ---------------------------------------------------------------------------

def hash_texts(df, min_global_df=5):
    print("[text] reading transcripts ...")
    # dtype=object keeps plain Python strings; converting pandas' Arrow strings with
    # .tolist() needs a second full copy of ~600 MB of text and runs out of memory
    txt = pd.read_csv(CALLS_CSV, usecols=["ticker", "call_date", "transcript_text"],
                      parse_dates=["call_date"], dtype={"transcript_text": object})
    txt = df[["ticker", "call_date"]].merge(txt, on=["ticker", "call_date"], how="left")
    texts = txt["transcript_text"].fillna("").tolist()
    del txt
    vec = HashingVectorizer(n_features=2**21, ngram_range=(1, 2), stop_words="english",
                            token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b",
                            alternate_sign=False, norm=None, dtype=np.float32)
    parts = {"full": [], "prepared": [], "qa": []}
    for i in range(0, len(texts), 500):
        batch = texts[i:i + 500]
        secs = [_split_sections(t) for t in batch]
        parts["full"].append(vec.transform(batch))
        parts["prepared"].append(vec.transform([p for p, _ in secs]))
        parts["qa"].append(vec.transform([q for _, q in secs]))
        print(f"[text]   hashed {min(i + 500, len(texts))}/{len(texts)}", end="\r")
    print()
    out = {}
    for s, mats in parts.items():
        X = sp.vstack(mats).tocsr()
        X = X[:, np.where(X.getnnz(axis=0) >= min_global_df)[0]]
        X.data = 1 + np.log(X.data)
        out[s] = X
    return out


def attach_text_scores(df, rebuild):
    if TEXT_SCORES_CSV.exists() and not rebuild:
        ts = pd.read_csv(TEXT_SCORES_CSV, parse_dates=["call_date"])
        if (len(ts) == len(df) and (ts["ticker"].values == df["ticker"].values).all()
                and (ts["call_date"].values == df["call_date"].values).all()):
            return pd.concat([df, ts.drop(columns=["ticker", "call_date"])], axis=1)
        print("[text] cached scores out of date, rebuilding")
    counts = hash_texts(df)
    ts = df[["ticker", "call_date"]].copy()
    for s, X in counts.items():
        ts[f"sim_prev_{s}"] = prev_call_similarity(df, X)
    for s, X in counts.items():
        for target, name in (("y_imm", f"text_{s}"), ("y_drift", f"text_drift_{s}")):
            print(f"[text] walk-forward {name} ...")
            ts[name] = walk_forward(df, target, TEXT_FIRST_TEST_YEAR,
                                    fit=lambda tr, y, X=X: TextModel().fit(X[tr], y),
                                    predict=lambda m, te, X=X: m.predict_proba(X[te]))
    ts.to_csv(TEXT_SCORES_CSV, index=False)
    print(f"[text] saved -> {TEXT_SCORES_CSV}")
    return pd.concat([df, ts.drop(columns=["ticker", "call_date"])], axis=1)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

def uses_text(groups):
    return any(g.startswith("text") for g in groups)


def run_wf(df, target, groups, kind):
    cols = cols_for(groups)
    X = df[cols].values.astype(float)
    mask = (df["year"] >= TEXT_FIRST_TEST_YEAR).values if uses_text(groups) else None
    return walk_forward(df, target, FIRST_TEST_YEAR,
                        fit=lambda tr, y: make_model(kind, cols).fit(X[tr], y),
                        predict=lambda m, te: m.predict_proba(X[te])[:, 1],
                        train_mask=mask)


def run_fixed(df, target, groups, kind):
    """Kartik's split: fit once on calls before the gap, predict 2022 onward."""
    days = load_prices().index
    test_start = pd.Timestamp(FIXED_TEST_START)
    cutoff = days[days.searchsorted(test_start) - LABEL_GAP_DAYS]
    cols = cols_for(groups)
    X = df[cols].values.astype(float)
    tr = (df["call_date"] < cutoff) & df[target].notna()
    if uses_text(groups):
        tr &= df["year"] >= TEXT_FIRST_TEST_YEAR
    te = (df["call_date"] >= test_start).values
    model = make_model(kind, cols).fit(X[tr.values], df.loc[tr, target].astype(int))
    proba = np.full(len(df), np.nan)
    proba[te] = model.predict_proba(X[te])[:, 1]
    return proba, int(tr.sum()), cutoff


def greedy(df, target, kind, cache, log):
    def val_auc(groups):
        key = (target, tuple(sorted(groups)), kind)
        if key not in cache:
            cache[key] = evaluate(df, run_wf(df, target, groups, kind), target, VALID_YEARS)["auc"]
        return cache[key]

    chosen, best = [], 0.5
    while True:
        trials = [(val_auc(chosen + [g]), g) for g in CANDIDATES if g not in chosen]
        if not trials:
            break
        auc, g = max(trials)
        kept = auc - best >= MIN_GAIN
        log.append(f"| {kind.upper()} | +{g} | {auc:.4f} | {auc - best:+.4f} | "
                   f"{'kept' if kept else 'stop'} |")
        print(f"[greedy {kind}] +{g:<10} valid reaction AUC {auc:.4f} "
              f"({'kept' if kept else 'stop'})")
        if not kept:
            break
        chosen.append(g)
        best = auc
    return chosen, best


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

PERIODS = {"valid": VALID_YEARS, "holdout": HOLDOUT_YEARS, "late": (2022, 2025)}


def evaluate_all(df, p, target):
    return {per: {"imm": evaluate(df, p, "y_imm", yrs) if target == "y_imm" else None,
                  "drift": evaluate(df, p, "y_drift", yrs)}
            for per, yrs in PERIODS.items()}


def velu_baseline(df, ours):
    """Velu's VADER predictions vs ours on the calls both have (holdout + 2022+)."""
    v = pd.read_csv(VELU_PRED_CSV, parse_dates=["call_date"])
    m = df[["ticker", "call_date", "year", "imm_ar", "drift_ar", "y_imm", "y_drift"]].copy()
    m["ours"] = ours
    m = m.merge(v[["ticker", "call_date", "score_reaction_lr"]], on=["ticker", "call_date"])
    m = m.dropna(subset=["ours", "score_reaction_lr"]).reset_index(drop=True)
    out = {}
    for per, yrs in (("holdout", HOLDOUT_YEARS), ("late", (2022, 2025))):
        out[per] = {name: {"imm": evaluate(m, m[col].values, "y_imm", yrs),
                           "drift": evaluate(m, m[col].values, "y_drift", yrs)}
                    for name, col in (("velu", "score_reaction_lr"), ("ours", "ours"))}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild-text", action="store_true")
    args = ap.parse_args()

    df = attach_text_scores(build_dataset(), args.rebuild_text)
    print(f"[Phase3] {len(df):,} calls, {df['y_imm'].notna().sum():,} with reaction")

    # ---- 1. choose inputs + model type on validation years only ----
    cache, log = {}, []
    picks = {k: greedy(df, "y_imm", k, cache, log) for k in ("lr", "xgb")}
    final_kind = max(picks, key=lambda k: picks[k][1])
    final_groups = picks[final_kind][0]
    print(f"[Phase3] final: {final_kind.upper()} on {final_groups} "
          f"(valid reaction AUC {picks[final_kind][1]:.4f})")

    # ---- 2. models to report ----
    other = "xgb" if final_kind == "lr" else "lr"
    models = {
        "final":        ("y_imm", final_groups, final_kind,
                         f"**Final** — reaction-trained, {final_kind.upper()}, greedy pick"),
        "other_kind":   ("y_imm", picks[other][0], other,
                         f"Reaction-trained, {other.upper()}, its own greedy pick"),
        "finbert_only": ("y_imm", FINBERT_GROUPS, final_kind,
                         "Reaction-trained, FinBERT/LM features only (no text, no market)"),
        "direct":       ("y_drift", ["tone", "text_drift", "market", "reaction"], "lr",
                         "Trained straight on GOOD/BAD (tone + text + market + reaction)"),
        "nolang":       ("y_drift", ["market", "reaction"], "lr",
                         "No language — market + reaction only"),
    }
    results, probas = {}, {}
    for key, (target, groups, kind, _) in models.items():
        print(f"[Phase3] walk-forward {key} ({kind}: {', '.join(groups)}) ...")
        probas[key] = p = run_wf(df, target, groups, kind)
        results[key] = evaluate_all(df, p, target)

    # ---- 3. Kartik's fixed split, same final config ----
    fixed = {}
    for key in ("final", "finbert_only", "nolang"):
        target, groups, kind, _ = models[key]
        p, n_train, cutoff = run_fixed(df, target, groups, kind)
        fixed[key] = {"imm": evaluate(df, p, "y_imm", (2022, 2025)) if target == "y_imm" else None,
                      "drift": evaluate(df, p, "y_drift", (2022, 2025))}
    fixed_info = (n_train, cutoff)

    # ---- 4. Velu's VADER run on the same calls ----
    base = velu_baseline(df, probas["final"])

    # ---- outputs ----
    keep = df["year"] >= VALID_YEARS[0]
    pred = df.loc[keep, ["ticker", "call_date", "imm_ar", "drift_ar", "pre_ar"]].copy()
    pred["label"] = np.where(pred["drift_ar"] > 0, "GOOD", "BAD")
    for key in models:
        pred[f"score_{key}"] = probas[key][keep.values]
    pred["confidence_good"] = pred["score_final"]
    pred["predicted"] = np.where(pred["confidence_good"] > 0.5, "GOOD", "BAD")
    pred = pred[pred["confidence_good"].notna()]
    pred.to_csv(PREDICTIONS_CSV, index=False)
    print(f"[Phase3] saved {len(pred):,} predictions -> {PREDICTIONS_CSV}")

    write_results(df, models, results, fixed, fixed_info, base, log, final_kind, final_groups)


def write_results(df, models, results, fixed, fixed_info, base, log, final_kind, final_groups):
    def table(period):
        rows = [TABLE_HEAD]
        for key, (*_, desc) in models.items():
            r = results[key][period]
            rows.append(fmt_row(desc, r["imm"], r["drift"]))
        return "\n".join(rows)

    n_train, cutoff = fixed_info
    fixed_rows = [TABLE_HEAD] + [fmt_row(models[k][3], fixed[k]["imm"], fixed[k]["drift"])
                                 for k in fixed]
    cmp_rows = [TABLE_HEAD]
    for per, label in (("holdout", "2018–2025"), ("late", "2022–2025")):
        for name, desc in (("velu", "Velu — VADER features"), ("ours", "Ours — FinBERT features")):
            r = base[per][name]
            cmp_rows.append(fmt_row(f"{desc}, {label}", r["imm"], r["drift"]))
    n_cmp = base["holdout"]["ours"]["drift"]["n"]

    lines = [
        "# Phase 3 (v2) — Classifier Results on FinBERT Features",
        "",
        "Same walk-forward method as [`Phase3_results.md`](Phase3_results.md) (Velu, VADER),",
        "with the VADER features replaced by the 18 FinBERT/LM features",
        "([`features_finbert.md`](features_finbert.md)) and the full call set "
        f"({len(df):,} calls, old NEUTRAL calls included).",
        "Regenerate with `python train_classifier_finbert.py`.",
        "",
        f"**Final model:** {final_kind.upper()} trained to predict the immediate reaction (day −1 → +1)",
        f"from: {', '.join(final_groups)} ({len(cols_for(final_groups))} inputs), chosen greedily on the",
        f"validation years {VALID_YEARS[0]}–{VALID_YEARS[1]} only. Its probability is the GOOD/BAD score.",
        "**GOOD/BAD:** GOOD if the drift (day +2 → +60 abnormal return) was positive.",
        "",
        "Columns as in `Phase3_results.md`. Short-term returns are mostly noise: AUC 0.55–0.60 on",
        "the reaction is a strong result; GOOD/BAD AUC barely above 0.5 is normal, and the",
        "top-minus-bottom 20% drift spread is what Phase 4 tests.",
        "",
        f"## Main: walk-forward, holdout {HOLDOUT_YEARS[0]}–{HOLDOUT_YEARS[1]} (untouched while choosing)",
        "",
        table("holdout"),
        "",
        "## Check: Kartik's fixed split, test 2022–2025",
        "",
        f"Trained once on {n_train:,} calls before {cutoff.date()} (a {LABEL_GAP_DAYS}-trading-day gap",
        "before 2022, so no training label uses 2022 prices), tested once on 2022–2025. Same inputs",
        "as the walk-forward models.",
        "",
        "\n".join(fixed_rows),
        "",
        "For reference, the walk-forward models on the same 2022–2025 calls:",
        "",
        table("late"),
        "",
        f"## FinBERT vs Velu's VADER run, on the same {n_cmp:,} holdout calls",
        "",
        "Only calls present in both runs (Velu's set excludes the old NEUTRAL calls). Velu's",
        "score is `score_reaction_lr` from his `data/predictions.csv`.",
        "",
        "\n".join(cmp_rows),
        "",
        f"## Validation years {VALID_YEARS[0]}–{VALID_YEARS[1]} (used to choose inputs)",
        "",
        table("valid"),
        "",
        "## Greedy selection log (validation reaction AUC)",
        "",
        "| Model | Step | Valid AUC | Gain | |",
        "|---|---|---|---|---|",
        *log,
        "",
    ]
    RESULTS_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"[Phase3] saved results -> {RESULTS_MD}")


if __name__ == "__main__":
    main()
