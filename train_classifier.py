"""
Phase 3 — Type Classifier (Good / Bad)
========================================
Everything is trained and tested walk-forward: each year is predicted by a model that
only saw earlier years, so every prediction is out-of-sample.

The main model learns to read the call: it predicts the IMMEDIATE price reaction
(day -1 -> +1) from the call's language. Its score P(reaction up) is the firm's
GOOD/BAD "type" score. GOOD/BAD itself is defined by the DRIFT after the call
(day +2 -> +60), and Phase 4 tests whether the score predicts that drift, i.e. whether
the market under-reacted to the language.

Why not train straight on the drift label? We tried (feature_study.py and the
comparison rows in Phase3_results.md): the drift label is so noisy that a model trained
on it ranks the later drift worse than the reaction-trained score does.

Inputs used (feature groups — see feature_study.py for which ones earn their place):
  sentiment / lm / style / qa   hand-built Phase 2 features (data/features_expanded.csv)
  delta                         change in those features vs. the firm's previous call
  similarity                    word-overlap with the firm's previous call ("Lazy Prices")
  text                          TF-IDF + logistic-regression score on the transcript words
  market                        3-month return and volatility before the call
  reaction                      the immediate reaction itself (type model only)

Outputs:
  data/text_scores.csv        out-of-sample text scores (committed, so teammates without
                              data/master.csv can still run this script)
  data/predictions.csv        out-of-sample predictions for every test year
  models/*.joblib             final models refit on all data
  docxxs/Phase3_results.md    metrics

Usage:
    python train_classifier.py
    python train_classifier.py --rebuild-text    # recompute text scores from master.csv
"""

import argparse
import hashlib
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import spearmanr
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, normalize

from event_returns import (DRIFT_WIN, IMMEDIATE_WIN, PRE_WIN, EventReturns,
                           load_prices)

warnings.filterwarnings("ignore")

FEATURES_CSV    = Path("data/features_expanded.csv")
MASTER_CSV      = Path("data/master.csv")
TEXT_SCORES_CSV = Path("data/text_scores.csv")
PREDICTIONS_CSV = Path("data/predictions.csv")
CACHE_DIR       = Path("data/_cache")
MODELS_DIR      = Path("models")
RESULTS_MD      = Path("docxxs/Phase3_results.md")

# ---------------------------------------------------------------------------
# Feature groups
# ---------------------------------------------------------------------------

# Whole-document VADER saturates at +1.0 on 99.5% of calls (it is built for short
# texts), so these three carry no information and are left out.
SATURATED = ["vader_compound", "vader_prepared_compound", "vader_qa_compound"]

HAND_GROUPS = {
    "sentiment": ["vader_pos", "vader_neg", "vader_neu",
                  "vader_prepared_pos", "vader_prepared_neg",
                  "vader_qa_pos", "vader_qa_neg",
                  "sent_compound_mean", "sent_compound_std",
                  "sent_compound_min", "sent_compound_max",
                  "pct_positive_sents", "pct_negative_sents"],
    "lm":        ["lm_positive_pct", "lm_negative_pct",
                  "lm_uncertainty_pct", "lm_net_sentiment"],
    "style":     ["fog_index", "flesch_ease", "avg_sentence_length",
                  "numeric_density", "word_count", "sentence_count",
                  "forward_looking_count", "forward_looking_density"],
    "qa":        ["qa_directness", "qa_turn_count", "qa_ratio"],
}
HAND_COLS = [c for cols in HAND_GROUPS.values() for c in cols]

GROUPS = {
    **HAND_GROUPS,
    "delta":      ["d_" + c for c in HAND_COLS],
    "similarity": ["sim_prev_full", "sim_prev_prepared", "sim_prev_qa"],
    "text":       ["text_full", "text_prepared", "text_qa"],
    "text_drift": ["text_drift_full", "text_drift_prepared", "text_drift_qa"],
    "market":     ["pre_ar", "pre_vol"],
    "reaction":   ["imm_ar"],          # only allowed for the drift (type) model
}

# Final configuration, chosen by feature_study.py on the 2012-2017 validation years
# (the 2018-2025 years were not used for choosing). See docxxs/Phase3_feature_study.md.
FINAL_GROUPS = ["delta", "text", "lm", "market"]   # greedy pick, Logistic Regression
FINAL_KIND   = "lr"

# Models trained and reported. The first one is the final model; the rest are there
# for comparison. (name: target, groups, model kind, description)
MODELS = {
    "reaction_lr":  ("y_imm", FINAL_GROUPS, "lr",
                     "**Final** — reaction-trained, Logistic Regression"),
    "reaction_xgb": ("y_imm", FINAL_GROUPS, "xgb",
                     "Reaction-trained, XGBoost"),
    "direct_lr":    ("y_drift", ["text", "market", "reaction"], "lr",
                     "Trained straight on GOOD/BAD (text + market + reaction)"),
    "nolang_lr":    ("y_drift", ["market", "reaction"], "lr",
                     "No language — market + reaction only"),
}
FINAL_MODEL = "reaction_lr"

# Walk-forward settings
TEXT_FIRST_TEST_YEAR = 2008     # first year with out-of-sample text scores
FIRST_TEST_YEAR      = 2010     # first year predicted by the main models
VALID_YEARS          = (2012, 2017)
HOLDOUT_YEARS        = (2018, 2025)
# A training row is only usable once its target is known: ~1 week for the reaction,
# ~3 months (60 trading days) for the drift.
EMBARGO_DAYS = {"y_imm": 7, "y_drift": 100}


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def build_dataset():
    """Hand features + corrected event returns + targets, one row per call."""
    df = pd.read_csv(FEATURES_CSV, parse_dates=["call_date"])
    df = df.drop(columns=SATURATED + ["abnormal_return"], errors="ignore")
    df = df.rename(columns={"label": "label_phase1"})
    df["row_id"] = np.arange(len(df))     # row order of features_expanded / master

    er = EventReturns(load_prices())
    keys = list(zip(df["ticker"], df["call_date"]))
    df["imm_ar"]   = [er.abnormal(t, d, IMMEDIATE_WIN) for t, d in keys]
    df["drift_ar"] = [er.abnormal(t, d, DRIFT_WIN) for t, d in keys]
    df["pre_ar"]   = [er.abnormal(t, d, PRE_WIN) for t, d in keys]
    df["pre_vol"]  = [er.volatility(t, d, PRE_WIN) for t, d in keys]

    # Change vs. the same firm's previous call
    df = df.sort_values(["ticker", "call_date"])
    prev = df.groupby("ticker")[HAND_COLS].shift(1)
    for c in HAND_COLS:
        df["d_" + c] = df[c] - prev[c]
    df = df.sort_values("row_id").reset_index(drop=True)

    df["year"]    = df["call_date"].dt.year
    df["y_imm"]   = (df["imm_ar"] > 0).astype(float).where(df["imm_ar"].notna())
    df["y_drift"] = (df["drift_ar"] > 0).astype(float).where(df["drift_ar"].notna())
    df["label"]   = np.where(df["drift_ar"] > 0, "GOOD", "BAD")
    return df


# ---------------------------------------------------------------------------
# Text: hashed word counts -> TF-IDF + logistic regression (fitted per fold)
# ---------------------------------------------------------------------------

def _split_sections(text):
    """Same rule as Phase 2: Q&A starts at the 2nd 'Operator:' turn."""
    first = text.find("Operator:")
    second = text.find("Operator:", first + 1) if first >= 0 else -1
    if second < 0:
        return text, ""
    return text[:second], text[second:]


def _cache_key(df):
    h = hashlib.md5()
    h.update(pd.util.hash_pandas_object(df[["ticker", "call_date"]], index=False).values)
    return h.hexdigest()[:12]


def load_text_counts(df, min_global_df=5):
    """Sublinear term-frequency matrices for full / prepared / qa text, row-aligned
    with df.row_id. Cached under data/_cache/ because hashing takes a few minutes.

    Columns appearing in fewer than `min_global_df` documents are dropped up front to
    save memory. This uses no labels, and each fold later applies its own stricter
    min_df on training rows only, so no test information reaches the model."""
    key = _cache_key(df)
    paths = {s: CACHE_DIR / f"text_{s}_{key}.npz" for s in ("full", "prepared", "qa")}
    if all(p.exists() for p in paths.values()):
        return {s: sp.load_npz(p) for s, p in paths.items()}

    print("[text] hashing transcripts (one-off, cached afterwards) ...")
    vec = HashingVectorizer(n_features=2**21, ngram_range=(1, 2), stop_words="english",
                            token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b",
                            alternate_sign=False, norm=None, dtype=np.float32)
    parts = {s: [] for s in paths}
    n_seen = 0
    for chunk in pd.read_csv(MASTER_CSV, usecols=["ticker", "call_date", "transcript_text"],
                             parse_dates=["call_date"], chunksize=500):
        expect = df.iloc[n_seen:n_seen + len(chunk)]
        if not ((chunk["ticker"].values == expect["ticker"].values).all()
                and (chunk["call_date"].values == expect["call_date"].values).all()):
            raise SystemExit("master.csv rows do not line up with features_expanded.csv")
        texts = chunk["transcript_text"].fillna("").tolist()
        secs = [_split_sections(t) for t in texts]
        parts["full"].append(vec.transform(texts))
        parts["prepared"].append(vec.transform([p for p, _ in secs]))
        parts["qa"].append(vec.transform([q for _, q in secs]))
        n_seen += len(chunk)
        print(f"[text]   {n_seen}/{len(df)}", end="\r")
    print()

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    out = {}
    for s, mats in parts.items():
        X = sp.vstack(mats).tocsr()
        X = X[:, np.where(X.getnnz(axis=0) >= min_global_df)[0]]
        X.data = 1 + np.log(X.data)          # sublinear tf
        sp.save_npz(paths[s], X)
        out[s] = X
    return out


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


def build_text_scores(df):
    """Out-of-sample text scores (walk-forward by year) + previous-call similarity."""
    counts = load_text_counts(df)
    out = df[["ticker", "call_date"]].copy()
    for s, X in counts.items():
        out[f"sim_prev_{s}"] = prev_call_similarity(df, X)

    for s, X in counts.items():
        for target, name in (("y_imm", f"text_{s}"), ("y_drift", f"text_drift_{s}")):
            print(f"[text] walk-forward {name} ...")
            out[name] = walk_forward(
                df, target, TEXT_FIRST_TEST_YEAR,
                fit=lambda tr, y: TextModel().fit(X[tr], y),
                predict=lambda m, te: m.predict_proba(X[te]))
    out.to_csv(TEXT_SCORES_CSV, index=False)
    print(f"[text] saved -> {TEXT_SCORES_CSV}")
    return out


def attach_text_scores(df, rebuild=False):
    have_cache = TEXT_SCORES_CSV.exists()
    if rebuild or not have_cache:
        if not MASTER_CSV.exists():
            if not have_cache:
                raise SystemExit("Need data/master.csv (or a committed data/text_scores.csv).")
            print("[text] master.csv not found -- using committed text_scores.csv")
        else:
            ts = build_text_scores(df)
            return pd.concat([df, ts.drop(columns=["ticker", "call_date"])], axis=1)
    ts = pd.read_csv(TEXT_SCORES_CSV, parse_dates=["call_date"])
    if not ((ts["ticker"].values == df["ticker"].values).all()
            and (ts["call_date"].values == df["call_date"].values).all()):
        raise SystemExit("text_scores.csv is out of date -- rerun with --rebuild-text")
    return pd.concat([df, ts.drop(columns=["ticker", "call_date"])], axis=1)


# ---------------------------------------------------------------------------
# Walk-forward training
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
    if kind == "lr":
        return Pipeline([("imp", SimpleImputer(strategy="median")),
                         ("sc", StandardScaler()),
                         ("clf", LogisticRegression(C=0.05, max_iter=3000))])
    from xgboost import XGBClassifier
    return XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.03,
                         subsample=0.8, colsample_bytree=0.8, min_child_weight=20,
                         eval_metric="logloss", random_state=42, n_jobs=-1)


def columns_for(groups):
    return [c for g in groups for c in GROUPS[g]]


def run_tabular(df, target, groups, kind):
    cols = columns_for(groups)
    X = df[cols].values.astype(float)
    uses_text = any(g.startswith("text") for g in groups)
    # text scores only exist out-of-sample from TEXT_FIRST_TEST_YEAR on
    mask = (df["year"] >= TEXT_FIRST_TEST_YEAR).values if uses_text else None
    return walk_forward(
        df, target, FIRST_TEST_YEAR,
        fit=lambda tr, y: make_tabular(kind).fit(X[tr], y),
        predict=lambda m, te: m.predict_proba(X[te])[:, 1],
        train_mask=mask)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

PERIODS = {"valid": VALID_YEARS, "holdout": HOLDOUT_YEARS,
           "all": (VALID_YEARS[0], HOLDOUT_YEARS[1])}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild-text", action="store_true",
                        help="recompute data/text_scores.csv from data/master.csv")
    args, _ = parser.parse_known_args()

    print("[Phase3] building dataset ...")
    df = build_dataset()
    df = attach_text_scores(df, rebuild=args.rebuild_text)
    print(f"[Phase3] {len(df)} calls, {df['y_imm'].notna().sum()} with reaction, "
          f"{df['y_drift'].notna().sum()} with drift")

    results, probas = {}, {}
    for key, (target, groups, kind, _desc) in MODELS.items():
        print(f"[Phase3] walk-forward {key} ({', '.join(groups)}) ...")
        probas[key] = p = run_tabular(df, target, groups, kind)
        results[key] = {
            per: {"imm": evaluate(df, p, "y_imm", yrs) if target == "y_imm" else None,
                  "drift": evaluate(df, p, "y_drift", yrs)}
            for per, yrs in PERIODS.items()}
        h = results[key]["holdout"]
        imm = f"reaction AUC {h['imm']['auc']:.4f}  " if h["imm"] else ""
        print(f"           holdout: {imm}GOOD/BAD AUC {h['drift']['auc']:.4f}  "
              f"drift rank-corr {h['drift']['drift_ic']:+.4f} (t {h['drift']['drift_ic_t']:+.1f})")

    # ---- predictions.csv (out-of-sample, validation + holdout years) ----
    keep = year_mask(df, PERIODS["all"])
    pred = df.loc[keep, ["ticker", "call_date", "imm_ar", "drift_ar", "pre_ar", "label"]].copy()
    for key in MODELS:
        pred[f"score_{key}"] = probas[key][keep]
    # names read by market_reaction.py / backtest.py / dashboard.py
    pred["confidence_good"] = probas[FINAL_MODEL][keep]
    pred["predicted"] = np.where(pred["confidence_good"] > 0.5, "GOOD", "BAD")
    pred["xgb_confidence_good"] = pred["confidence_good"]
    pred["xgb_predicted"] = pred["predicted"]
    pred = pred[pred["confidence_good"].notna()]
    pred.to_csv(PREDICTIONS_CSV, index=False)
    print(f"[Phase3] saved {len(pred)} out-of-sample predictions -> {PREDICTIONS_CSV}")

    # ---- final model refit on all data, for scoring new calls ----
    MODELS_DIR.mkdir(exist_ok=True)
    target, groups, kind, _ = MODELS[FINAL_MODEL]
    cols = columns_for(groups)
    ok = df[target].notna() & (df["year"] >= TEXT_FIRST_TEST_YEAR)
    model = make_tabular(kind).fit(df.loc[ok, cols].values, df.loc[ok, target].astype(int))
    joblib.dump({"model": model, "columns": cols}, MODELS_DIR / "final_model.joblib")
    coefs = None
    if kind == "lr":
        coefs = pd.Series(model.named_steps["clf"].coef_[0], index=cols)

    write_results(results, coefs)
    print("[Phase3] DONE. Next: python market_reaction.py")


def write_results(results, coefs):
    def table(period):
        rows = [TABLE_HEAD]
        for key, (*_, desc) in MODELS.items():
            r = results[key][period]
            rows.append(fmt_row(desc, r["imm"], r["drift"]))
        return "\n".join(rows)

    cm = results[FINAL_MODEL]["holdout"]["drift"]["cm"]
    lines = [
        "# Phase 3 — Classifier Results",
        "",
        "All numbers are **out-of-sample**: each year is predicted by a model trained only on",
        "earlier years (walk-forward), with a gap so no training target overlaps the test year.",
        f"Inputs were chosen on the validation years {VALID_YEARS[0]}–{VALID_YEARS[1]} only",
        "(`feature_study.py`); the holdout years were not used while choosing.",
        "Regenerate with `python train_classifier.py`.",
        "",
        "**Final model:** Logistic Regression trained to predict the immediate reaction",
        f"(day −1 → +1) from: {', '.join(FINAL_GROUPS)} "
        f"({len(columns_for(FINAL_GROUPS))} inputs). Its probability is the GOOD/BAD score.",
        "**GOOD/BAD:** GOOD if the drift (day +2 → +60 abnormal return) was positive.",
        "",
        "Columns: *Reaction AUC* — how well the score predicts the immediate reaction;",
        "*GOOD/BAD AUC / accuracy* — against the drift label (accuracy uses a 0.5 cut-off);",
        "*Drift rank-corr* — average yearly Spearman correlation between score and drift, with its",
        "t-statistic across years; *Drift top-bottom 20%* — average drift of the 20% highest-scored",
        "calls minus the 20% lowest, within each year.",
        "",
        f"## Holdout years {HOLDOUT_YEARS[0]}–{HOLDOUT_YEARS[1]} (untouched while choosing)",
        "",
        table("holdout"),
        "",
        f"## Validation years {VALID_YEARS[0]}–{VALID_YEARS[1]} (used to choose inputs)",
        "",
        table("valid"),
        "",
        "## Final model — GOOD/BAD confusion matrix (holdout)",
        "```",
        "             pred BAD  pred GOOD",
        f"  actual BAD   {cm[0,0]:6d}    {cm[0,1]:6d}",
        f"  actual GOOD  {cm[1,0]:6d}    {cm[1,1]:6d}",
        "```",
        "",
    ]
    if coefs is not None:
        top = coefs.reindex(coefs.abs().sort_values(ascending=False).index).head(12)
        lines += ["## Largest coefficients (final model, standardised inputs)", "",
                  "| Input | Coefficient | Direction |", "|---|---|---|"]
        lines += [f"| `{c}` | {v:+.3f} | {'→ reaction up' if v > 0 else '→ reaction down'} |"
                  for c, v in top.items()]
        lines.append("")
    lines += [
        "## Reading these numbers",
        "- Short-term stock moves are mostly noise, so AUC 0.55–0.60 is a strong result for",
        "  this kind of data. An AUC far above that would point to leaked future information.",
        "- GOOD/BAD accuracy stays close to the always-majority rate: the score is a *ranking*",
        "  signal, so its value shows in the top-vs-bottom drift spread, which Phase 4 tests.",
        "- Compare the final row with the *no language* row: the gap is what language adds.",
        "- Why these inputs: `docxxs/Phase3_feature_study.md`.",
    ]
    RESULTS_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"[Phase3] saved results -> {RESULTS_MD}")


if __name__ == "__main__":
    main()
