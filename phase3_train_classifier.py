"""
Phase 3 — Type Classifier: one out-of-sample score per call
===========================================================
Walk-forward logistic regression on the 18 FinBERT/LM features in data/features_finbert.csv
(plus pre-call market inputs), over the full 12,712-call set.

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
  docxxs/Phase3_finbert_results.md    results

Usage:
    python phase3_train_classifier.py
    python phase3_train_classifier.py --rebuild-text
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import spearmanr
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, normalize

from phase1_abnormal_returns import BAD_PRICE_TICKERS, PRE_WIN, EventReturns, load_prices
from phase2_extract_finbert_features import FEATURES

warnings.filterwarnings("ignore")

FEATURES_CSV    = Path("data/features_finbert.csv")
CALLS_CSV       = Path("data/calls_all.csv")
TEXT_SCORES_CSV = Path("data/text_scores_finbert.csv")
PREDICTIONS_CSV = Path("data/predictions_finbert.csv")
CACHE_DIR       = Path("data/_cache")
RESULTS_MD      = Path("docxxs/Phase3_finbert_results.md")

MIN_GAIN        = 0.002          # a group is kept only if it adds this much validation AUC
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



# Walk-forward settings
TEXT_FIRST_TEST_YEAR = 2008     # first year with out-of-sample text scores
FIRST_TEST_YEAR      = 2010     # first year predicted by the main models
VALID_YEARS          = (2012, 2017)
HOLDOUT_YEARS        = (2018, 2025)
# A training row is only usable once its target is known: ~1 week for the reaction,
# ~3 months (60 trading days) for the drift.
EMBARGO_DAYS = {"y_imm": 7, "y_drift": 100}


# ---------------------------------------------------------------------------
# Shared building blocks (walk-forward, models, metrics)
# ---------------------------------------------------------------------------

def walk_forward(df, target, first_year, fit, predict, train_mask=None):
    """For each year Y >= first_year: fit on rows whose target was already known
    before Jan 1 of Y, predict year Y. Returns an out-of-sample probability array."""
    proba = np.full(len(df), np.nan)
    embargo = pd.Timedelta(days=EMBARGO_DAYS[target])
    has_y = df[target].notna().values
    base = has_y if train_mask is None else has_y & train_mask
    for year in range(first_year, df["year"].max() + 1):
        cutoff = pd.Timestamp(f"{year}-01-01") - embargo
        tr = np.where(base & (df["call_date"] < cutoff).values)[0]
        te = np.where((df["year"] == year).values)[0]
        if len(tr) < 200 or len(te) == 0:
            continue
        model = fit(tr, df[target].values[tr].astype(int))
        proba[te] = predict(model, te)
    return proba


def make_tabular(kind):
    """'lr': median fill + z-score + L2 logistic regression (C=0.05); 'xgb': small XGBoost."""
    if kind == "lr":
        return Pipeline([("imp", SimpleImputer(strategy="median")),
                         ("sc", StandardScaler()),
                         ("clf", LogisticRegression(C=0.05, max_iter=3000))])
    from xgboost import XGBClassifier
    return XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.03,
                         subsample=0.8, colsample_bytree=0.8, min_child_weight=20,
                         eval_metric="logloss", random_state=42, n_jobs=-1)


def _split_sections(text):
    """Same rule as Phase 2: Q&A starts at the 2nd 'Operator:' turn."""
    first = text.find("Operator:")
    second = text.find("Operator:", first + 1) if first >= 0 else -1
    if second < 0:
        return text, ""
    return text[:second], text[second:]


class TextModel:
    """TF-IDF (vocabulary + idf learned from training rows only) + logistic regression."""

    def __init__(self, C=1.0, min_df=10, max_df=0.5, max_features=50_000):
        self.C, self.min_df, self.max_df, self.max_features = C, min_df, max_df, max_features

    def _tfidf(self, X):
        return normalize(X[:, self.keep].multiply(self.idf).tocsr())

    def fit(self, X, y):
        n = X.shape[0]
        dfreq = X.getnnz(axis=0)
        keep = np.where((dfreq >= self.min_df) & (dfreq <= self.max_df * n))[0]
        if len(keep) > self.max_features:
            keep = keep[np.argsort(-dfreq[keep])[:self.max_features]]
        self.keep = keep
        self.idf = (np.log((1 + n) / (1 + dfreq[keep])) + 1).astype(np.float32)
        self.clf = LogisticRegression(C=self.C, max_iter=3000).fit(self._tfidf(X), y)
        return self

    def predict_proba(self, X):
        return self.clf.predict_proba(self._tfidf(X))[:, 1]


def prev_call_similarity(df, X):
    """Cosine similarity between each call and the same firm's previous call."""
    Xn = normalize(X)
    order = df.sort_values(["ticker", "call_date"])
    prev_row = order.groupby("ticker")["row_id"].shift(1)
    sim = np.full(len(df), np.nan)
    has = prev_row.notna().values
    cur = order["row_id"].values[has]
    prv = prev_row.values[has].astype(int)
    sim[cur] = np.asarray(Xn[cur].multiply(Xn[prv]).sum(axis=1)).ravel()
    return sim


def year_mask(df, years):
    return ((df["year"] >= years[0]) & (df["year"] <= years[1])).values


def evaluate(df, proba, target, years):
    """AUC / accuracy for the target, plus how well the score ranks the later drift."""
    m = year_mask(df, years) & ~np.isnan(proba) & df[target].notna().values
    y = df[target].values[m].astype(int)
    p = proba[m]
    res = {"n": int(m.sum()), "auc": roc_auc_score(y, p),
           "acc": accuracy_score(y, p > 0.5), "base_rate": max(y.mean(), 1 - y.mean())}

    # Rank correlation with drift, computed year by year (avoids pooling regimes)
    sub = pd.DataFrame({"p": p, "drift": df["drift_ar"].values[m],
                        "year": df["year"].values[m]}).dropna()
    ics = sub.groupby("year").apply(lambda g: spearmanr(g["p"], g["drift"])[0])
    res["drift_ic"] = ics.mean()
    res["drift_ic_t"] = ics.mean() / (ics.std(ddof=1) / np.sqrt(len(ics)))
    sub["q"] = sub.groupby("year")["p"].transform(
        lambda s: pd.qcut(s.rank(method="first"), 5, labels=False))
    qm = sub.groupby("q")["drift"].mean()
    res["drift_q5_q1"] = qm.iloc[-1] - qm.iloc[0]
    res["cm"] = confusion_matrix(y, p > 0.5)
    return res


TABLE_HEAD = ("| Model | Reaction AUC | GOOD/BAD AUC | GOOD/BAD accuracy | "
              "Drift rank-corr (t) | Drift top-bottom 20% |\n|---|---|---|---|---|---|")


def fmt_row(name, r_imm, r_drift):
    imm = f"{r_imm['auc']:.4f}" if r_imm else "—"
    return (f"| {name} | {imm} | {r_drift['auc']:.4f} | {r_drift['acc']:.3f} "
            f"(majority {r_drift['base_rate']:.3f}) | "
            f"{r_drift['drift_ic']:+.4f} ({r_drift['drift_ic_t']:+.1f}) | "
            f"{r_drift['drift_q5_q1']*100:+.2f}% |")


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
# Text scores (TF-IDF word-frequency model on calls_all.csv)
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

    write_results(df, models, results, fixed, fixed_info, log, final_kind, final_groups)


def write_results(df, models, results, fixed, fixed_info, log, final_kind, final_groups):
    def table(period):
        rows = [TABLE_HEAD]
        for key, (*_, desc) in models.items():
            r = results[key][period]
            rows.append(fmt_row(desc, r["imm"], r["drift"]))
        return "\n".join(rows)

    n_train, cutoff = fixed_info
    fixed_rows = [TABLE_HEAD] + [fmt_row(models[k][3], fixed[k]["imm"], fixed[k]["drift"])
                                 for k in fixed]

    lines = [
        "# Phase 3 — Classifier Results on FinBERT Features",
        "",
        "Walk-forward classifier on the 18 FinBERT/LM features",
        "([`features_finbert.md`](features_finbert.md)) and the full call set "
        f"({len(df):,} calls, old NEUTRAL calls included).",
        "Regenerate with `python phase3_train_classifier.py`.",
        "",
        f"**Final model:** {final_kind.upper()} trained to predict the immediate reaction (day −1 → +1)",
        f"from: {', '.join(final_groups)} ({len(cols_for(final_groups))} inputs), chosen greedily on the",
        f"validation years {VALID_YEARS[0]}–{VALID_YEARS[1]} only. Its probability is the GOOD/BAD score.",
        "**GOOD/BAD:** GOOD if the drift (day +2 → +60 abnormal return) was positive.",
        "",
        "Short-term returns are mostly noise: AUC 0.55–0.60 on",
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
        "## Earlier VADER baseline (past result)",
        "",
        "An earlier version of this model on VADER features (Velu's pipeline) reached a holdout",
        "reaction AUC of 0.588 against 0.611 here, on the same calls. That code and its data are",
        "kept in [`archive/vader/`](../archive/vader/) and are no longer part of the pipeline.",
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
