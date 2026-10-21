"""
Phase 3 — Type Classifier (Good / Bad)
========================================
Trains Logistic Regression (baseline) and XGBoost on features_expanded.csv.
Uses a date-based train/test split to mimic real forecasting.

Outputs:
  models/lr_model.joblib     — trained Logistic Regression
  models/xgb_model.joblib    — trained XGBoost
  data/predictions.csv       — [ticker, call_date, label, predicted, confidence_good]
  docxxs/Phase3_results.md   — accuracy / AUC / confusion matrix / feature importances

Usage:
    python train_classifier.py
    python train_classifier.py --test-date 2022-01-01   # custom split date
"""

import argparse
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

warnings.filterwarnings("ignore")

FEATURES_CSV   = Path("data/features_expanded.csv")
MASTER_CSV     = Path("data/master.csv")
PREDICTIONS_CSV = Path("data/predictions.csv")
MODELS_DIR     = Path("models")
RESULTS_MD     = Path("docxxs/Phase3_results.md")

FEATURE_COLS = [
    "vader_compound", "vader_pos", "vader_neg", "vader_neu",
    "vader_prepared_compound", "vader_prepared_pos", "vader_prepared_neg",
    "vader_qa_compound", "vader_qa_pos", "vader_qa_neg",
    "sent_compound_mean", "sent_compound_std", "sent_compound_min", "sent_compound_max",
    "pct_positive_sents", "pct_negative_sents",
    "lm_positive_pct", "lm_negative_pct", "lm_uncertainty_pct", "lm_net_sentiment",
    "fog_index", "flesch_ease", "avg_sentence_length",
    "numeric_density", "word_count", "sentence_count",
    "forward_looking_count", "forward_looking_density",
    "qa_directness", "qa_turn_count", "qa_ratio",
]


def load_data():
    df = pd.read_csv(FEATURES_CSV, parse_dates=["call_date"])
    # Keep only GOOD/BAD
    df = df[df["label"].isin(["GOOD", "BAD"])].copy()
    df["y"] = (df["label"] == "GOOD").astype(int)
    return df


def date_split(df, test_date):
    train = df[df["call_date"] < pd.Timestamp(test_date)]
    test  = df[df["call_date"] >= pd.Timestamp(test_date)]
    return train, test


def get_xy(df):
    # Only keep feature cols that exist in this dataset
    cols = [c for c in FEATURE_COLS if c in df.columns]
    X = df[cols].copy()
    y = df["y"].values
    return X, y, cols


def train_lr(X_train, y_train):
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("clf",     LogisticRegression(max_iter=1000, random_state=42, C=1.0)),
    ])
    pipe.fit(X_train, y_train)
    return pipe


def train_xgb(X_train, y_train):
    try:
        from xgboost import XGBClassifier
    except ImportError:
        print("[Phase3] xgboost not installed — skipping XGB. Run: pip install xgboost")
        return None

    # Fill NaN for XGBoost (it CAN handle them but safer to impute for consistency)
    from sklearn.pipeline import Pipeline as Pipe
    from sklearn.impute import SimpleImputer
    pipe = Pipe([
        ("imputer", SimpleImputer(strategy="median")),
        ("clf", XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            use_label_encoder=False, eval_metric="logloss",
            random_state=42, n_jobs=-1,
        )),
    ])
    pipe.fit(X_train, y_train)
    return pipe


def evaluate(name, pipe, X_test, y_test, feat_cols):
    y_pred  = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]
    acc  = accuracy_score(y_test, y_pred)
    auc  = roc_auc_score(y_test, y_proba)
    cm   = confusion_matrix(y_test, y_pred)
    rep  = classification_report(y_test, y_pred, target_names=["BAD","GOOD"])

    print(f"\n{'='*50}")
    print(f"  {name}")
    print(f"{'='*50}")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  AUC      : {auc:.4f}")
    print(f"\nConfusion Matrix (rows=actual, cols=predicted):")
    print(f"         BAD   GOOD")
    print(f"  BAD  {cm[0,0]:5d}  {cm[0,1]:5d}")
    print(f"  GOOD {cm[1,0]:5d}  {cm[1,1]:5d}")
    print(f"\nClassification Report:\n{rep}")

    # Feature importances
    importances = None
    try:
        clf = pipe.named_steps["clf"]
        if hasattr(clf, "coef_"):
            importances = pd.Series(np.abs(clf.coef_[0]), index=feat_cols).sort_values(ascending=False)
            print(f"\nTop 10 features (|coef|):")
        elif hasattr(clf, "feature_importances_"):
            importances = pd.Series(clf.feature_importances_, index=feat_cols).sort_values(ascending=False)
            print(f"\nTop 10 features (importance):")
        if importances is not None:
            print(importances.head(10).to_string())
    except Exception:
        pass

    return acc, auc, cm, rep, importances, y_pred, y_proba


def save_predictions(df_test, y_pred_lr, y_proba_lr, y_pred_xgb, y_proba_xgb):
    out = df_test[["ticker", "call_date", "label"]].copy()
    out["lr_predicted"]       = ["GOOD" if p == 1 else "BAD" for p in y_pred_lr]
    out["lr_confidence_good"] = y_proba_lr
    if y_pred_xgb is not None:
        out["xgb_predicted"]       = ["GOOD" if p == 1 else "BAD" for p in y_pred_xgb]
        out["xgb_confidence_good"] = y_proba_xgb
    out.to_csv(PREDICTIONS_CSV, index=False)
    print(f"\n[Phase3] Saved predictions -> {PREDICTIONS_CSV} ({len(out)} rows)")


def save_results_md(split_date, train_n, test_n, feat_cols,
                    lr_acc, lr_auc, lr_cm, lr_rep, lr_imp,
                    xgb_acc, xgb_auc, xgb_cm, xgb_rep, xgb_imp):
    lines = [
        "# Phase 3 — Classifier Results",
        "",
        f"**Train/test split date:** {split_date}  ",
        f"**Training rows:** {train_n}  ",
        f"**Test rows:** {test_n}  ",
        f"**Features used:** {len(feat_cols)}",
        "",
        "---",
        "",
        "## Logistic Regression (Baseline)",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Accuracy | {lr_acc:.4f} |",
        f"| AUC | {lr_auc:.4f} |",
        "",
        "**Confusion Matrix:**",
        "```",
        f"         BAD   GOOD",
        f"  BAD  {lr_cm[0,0]:5d}  {lr_cm[0,1]:5d}",
        f"  GOOD {lr_cm[1,0]:5d}  {lr_cm[1,1]:5d}",
        "```",
        "",
    ]
    if lr_imp is not None:
        lines += ["**Top 10 Features (|coefficient|):**", "```"]
        for feat, val in lr_imp.head(10).items():
            lines.append(f"  {feat:<35s} {val:.4f}")
        lines += ["```", ""]

    if xgb_acc is not None:
        lines += [
            "---",
            "",
            "## XGBoost",
            "",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Accuracy | {xgb_acc:.4f} |",
            f"| AUC | {xgb_auc:.4f} |",
            "",
            "**Confusion Matrix:**",
            "```",
            f"         BAD   GOOD",
            f"  BAD  {xgb_cm[0,0]:5d}  {xgb_cm[0,1]:5d}",
            f"  GOOD {xgb_cm[1,0]:5d}  {xgb_cm[1,1]:5d}",
            "```",
            "",
        ]
        if xgb_imp is not None:
            lines += ["**Top 10 Features (importance):**", "```"]
            for feat, val in xgb_imp.head(10).items():
                lines.append(f"  {feat:<35s} {val:.4f}")
            lines += ["```", ""]

    lines += [
        "---",
        "",
        "## Notes",
        "- Date-based split: train on older calls, test on newer (no data leakage).",
        "- NaN features imputed with column median before training.",
        "- A modest accuracy result is a *valid finding* — executives tend to sound",
        "  similarly confident regardless of actual outcomes (see Phase 2 sanity-check).",
        "- See `data/predictions.csv` for per-call predicted label and confidence score.",
    ]

    RESULTS_MD.parent.mkdir(exist_ok=True)
    RESULTS_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"[Phase3] Saved results -> {RESULTS_MD}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-date", default="2022-01-01",
                        help="date boundary for train/test split (default: 2022-01-01)")
    args, _ = parser.parse_known_args()

    print(f"[Phase3] Loading {FEATURES_CSV} ...")
    df = load_data()
    print(f"[Phase3] {len(df)} labeled rows (GOOD+BAD)")

    train_df, test_df = date_split(df, args.test_date)
    print(f"[Phase3] Train: {len(train_df)} rows (before {args.test_date})")
    print(f"[Phase3] Test : {len(test_df)} rows (from {args.test_date})")

    if len(train_df) < 50 or len(test_df) < 10:
        print("ERROR: Not enough data for split. Adjust --test-date.")
        return

    X_train, y_train, feat_cols = get_xy(train_df)
    X_test,  y_test,  _        = get_xy(test_df)

    MODELS_DIR.mkdir(exist_ok=True)

    # --- Logistic Regression ---
    print("\n[Phase3] Training Logistic Regression ...")
    lr_pipe = train_lr(X_train, y_train)
    joblib.dump(lr_pipe, MODELS_DIR / "lr_model.joblib")
    lr_acc, lr_auc, lr_cm, lr_rep, lr_imp, lr_pred, lr_proba = evaluate(
        "Logistic Regression", lr_pipe, X_test, y_test, feat_cols)

    # --- XGBoost ---
    print("\n[Phase3] Training XGBoost ...")
    xgb_pipe = train_xgb(X_train, y_train)
    xgb_acc = xgb_auc = xgb_cm = xgb_rep = xgb_imp = xgb_pred = xgb_proba = None
    if xgb_pipe is not None:
        joblib.dump(xgb_pipe, MODELS_DIR / "xgb_model.joblib")
        xgb_acc, xgb_auc, xgb_cm, xgb_rep, xgb_imp, xgb_pred, xgb_proba = evaluate(
            "XGBoost", xgb_pipe, X_test, y_test, feat_cols)

    # --- Save outputs ---
    save_predictions(test_df, lr_pred, lr_proba, xgb_pred, xgb_proba)
    save_results_md(args.test_date, len(train_df), len(test_df), feat_cols,
                    lr_acc, lr_auc, lr_cm, lr_rep, lr_imp,
                    xgb_acc, xgb_auc, xgb_cm, xgb_rep, xgb_imp)

    print("\n[Phase3] DONE. Next: python market_reaction.py")


if __name__ == "__main__":
    main()
