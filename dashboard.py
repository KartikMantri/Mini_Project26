"""
Phase 6 — Interactive Dashboard
=================================
Streamlit dashboard visualising the full pipeline:
  - Dataset overview
  - Feature distributions (GOOD vs BAD)
  - Classifier performance (accuracy, AUC, confusion matrix)
  - Feature importances
  - Market-reaction test results
  - Backtest equity curve

Usage:
    streamlit run dashboard.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Decoding the Game — Earnings Call Classifier",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.main { background: #0f1117; }
.metric-card {
    background: linear-gradient(135deg, #1e2235 0%, #252b3b 100%);
    border: 1px solid #2e3654;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    text-align: center;
}
.metric-value { font-size: 2rem; font-weight: 700; color: #7c9ef8; }
.metric-label { font-size: 0.85rem; color: #9ba3c0; margin-top: 0.3rem; }
.section-header {
    font-size: 1.4rem; font-weight: 600; color: #e2e8f0;
    border-left: 4px solid #4f7cff; padding-left: 0.8rem;
    margin: 1.5rem 0 1rem 0;
}
</style>
""", unsafe_allow_html=True)

PALETTE = {"GOOD": "#4ade80", "BAD": "#f87171", "NEUTRAL": "#94a3b8"}
BG      = "#0f1117"
CARD_BG = "#1e2235"
GRID    = "#2e3654"


def load_data():
    data = {}
    for name, path in [
        ("features", "data/features_expanded.csv"),
        ("master",   "data/master.csv"),
        ("preds",    "data/predictions.csv"),
        ("market",   "data/market_reaction.csv"),
        ("equity",   "data/equity_curve.csv"),
        ("backtest", "data/backtest_results.csv"),
    ]:
        p = Path(path)
        if p.exists():
            data[name] = pd.read_csv(p, parse_dates=["call_date"] if "call_date" in
                                     pd.read_csv(p, nrows=0).columns else None)
        else:
            data[name] = None
    return data


def plotly_dark_layout(fig, title=None, height=400):
    fig.update_layout(
        height=height,
        paper_bgcolor=BG,
        plot_bgcolor=CARD_BG,
        font=dict(color="#e2e8f0", family="Inter"),
        title=dict(text=title, font=dict(size=16, color="#e2e8f0")) if title else {},
        xaxis=dict(gridcolor=GRID, linecolor=GRID),
        yaxis=dict(gridcolor=GRID, linecolor=GRID),
        margin=dict(l=40, r=20, t=40 if title else 20, b=40),
        legend=dict(bgcolor=CARD_BG, bordercolor=GRID),
    )
    return fig


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📊 Decoding the Game")
    st.markdown("*Earnings Call Signal Classifier*")
    st.markdown("---")
    page = st.radio("Navigate", [
        "🏠 Overview",
        "🔬 Feature Analysis",
        "🤖 Classifier Performance",
        "📉 Market Reaction",
        "💰 Backtest",
    ])
    st.markdown("---")
    st.markdown("**Dataset:** S&P 500 Earnings Calls  \n**Period:** 2005 – 2025  \n**Tickers:** ~200")


D = load_data()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1: Overview
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Overview":
    st.markdown("# 📈 Decoding the Game")
    st.markdown("### Predicting Earnings Call Outcomes from Language Alone")
    st.markdown("---")

    # Top-level metrics
    c1, c2, c3, c4, c5 = st.columns(5)
    if D["master"] is not None:
        m = D["master"]
        n_calls    = len(m)
        n_tickers  = m["ticker"].nunique()
        n_good     = (m["label"] == "GOOD").sum()
        n_bad      = (m["label"] == "BAD").sum()
        yr_range   = f"{m['call_date'].str[:4].min()}–{m['call_date'].str[:4].max()}" \
                     if "call_date" in m.columns else "–"
    else:
        n_calls = n_tickers = n_good = n_bad = "–"
        yr_range = "–"

    for col, val, lbl in zip(
        [c1, c2, c3, c4, c5],
        [n_calls, n_tickers, n_good, n_bad, yr_range],
        ["Labeled Calls", "Tickers", "GOOD Calls", "BAD Calls", "Date Range"]
    ):
        col.markdown(f"""
        <div class="metric-card">
          <div class="metric-value">{val:,}" if isinstance(val, int) else f"{val}</div>
          <div class="metric-label">{lbl}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("")
    st.markdown('<div class="section-header">Label Distribution Over Time</div>', unsafe_allow_html=True)

    if D["master"] is not None:
        m = D["master"].copy()
        m["year"] = m["call_date"].str[:4]
        yearly = m.groupby(["year", "label"]).size().reset_index(name="count")
        fig = px.bar(yearly, x="year", y="count", color="label",
                     color_discrete_map=PALETTE, barmode="stack",
                     labels={"year": "Year", "count": "Calls", "label": "Label"})
        plotly_dark_layout(fig, height=350)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-header">Pipeline Overview</div>', unsafe_allow_html=True)
    cols = st.columns(6)
    steps = [
        ("📥", "Phase 1", "Data Collection", "33k+ transcripts\n→ labeled dataset"),
        ("🔬", "Phase 2", "Feature Extraction", "VADER + LM lexicon\n+ readability"),
        ("🤖", "Phase 3", "Classifier", "LR baseline\n+ XGBoost"),
        ("📉", "Phase 4", "Market Reaction", "Immediate vs drift\nt-test"),
        ("💰", "Phase 5", "Backtest", "Long/Short strategy\nSharpe ratio"),
        ("📊", "Phase 6", "Dashboard", "This dashboard"),
    ]
    for col, (icon, ph, title, desc) in zip(cols, steps):
        col.markdown(f"""
        <div class="metric-card">
          <div style="font-size:1.8rem">{icon}</div>
          <div style="font-size:0.7rem;color:#4f7cff;font-weight:600">{ph}</div>
          <div style="font-size:0.9rem;font-weight:600;color:#e2e8f0">{title}</div>
          <div style="font-size:0.75rem;color:#9ba3c0;margin-top:0.3rem">{desc}</div>
        </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2: Feature Analysis
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔬 Feature Analysis":
    st.markdown("# 🔬 Feature Analysis")
    st.markdown("Distribution of extracted NLP features for GOOD vs BAD calls.")

    if D["features"] is None:
        st.warning("features_expanded.csv not found. Run `python extract_features_fast.py` first.")
    else:
        feat_df = D["features"]
        numeric_feats = [
            "vader_compound", "vader_prepared_compound", "vader_qa_compound",
            "sent_compound_mean", "lm_positive_pct", "lm_negative_pct",
            "lm_uncertainty_pct", "lm_net_sentiment", "fog_index",
            "numeric_density", "forward_looking_density", "qa_directness",
        ]
        numeric_feats = [f for f in numeric_feats if f in feat_df.columns]

        selected = st.selectbox("Select feature", numeric_feats)
        fig = go.Figure()
        for label, color in PALETTE.items():
            if label == "NEUTRAL":
                continue
            sub = feat_df[feat_df["label"] == label][selected].dropna()
            fig.add_trace(go.Histogram(
                x=sub, name=label, opacity=0.7,
                marker_color=color, nbinsx=50,
            ))
        fig.update_layout(barmode="overlay")
        plotly_dark_layout(fig, title=f"Distribution of {selected} by Label", height=400)
        st.plotly_chart(fig, use_container_width=True)

        # Correlation heatmap
        st.markdown('<div class="section-header">Feature Correlation Matrix</div>', unsafe_allow_html=True)
        corr_feats = [f for f in numeric_feats if f in feat_df.columns]
        corr = feat_df[corr_feats].corr()
        fig2 = px.imshow(corr, color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                         text_auto=".2f", aspect="auto")
        plotly_dark_layout(fig2, height=500)
        st.plotly_chart(fig2, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3: Classifier Performance
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🤖 Classifier Performance":
    st.markdown("# 🤖 Classifier Performance")

    if D["preds"] is None:
        st.warning("predictions.csv not found. Run `python train_classifier.py` first.")
    else:
        preds = D["preds"]
        col_pred = "xgb_predicted" if "xgb_predicted" in preds.columns else "lr_predicted"
        col_conf = "xgb_confidence_good" if "xgb_confidence_good" in preds.columns else "lr_confidence_good"

        correct = (preds["label"] == preds[col_pred]).mean()
        good_mask = preds["label"] == "GOOD"
        c1, c2, c3, c4 = st.columns(4)
        for col, val, lbl in zip(
            [c1, c2, c3, c4],
            [f"{correct:.1%}", len(preds),
             (preds["label"] == "GOOD").sum(),
             (preds["label"] == "BAD").sum()],
            ["Test Accuracy", "Test Samples", "GOOD in Test", "BAD in Test"]
        ):
            col.markdown(f"""
            <div class="metric-card">
              <div class="metric-value">{val}</div>
              <div class="metric-label">{lbl}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("")

        # Confusion matrix
        st.markdown('<div class="section-header">Confusion Matrix</div>', unsafe_allow_html=True)
        from sklearn.metrics import confusion_matrix
        labels_order = ["GOOD", "BAD"]
        cm = confusion_matrix(preds["label"], preds[col_pred], labels=labels_order)
        fig_cm = px.imshow(cm, x=labels_order, y=labels_order,
                           labels=dict(x="Predicted", y="Actual", color="Count"),
                           text_auto=True, color_continuous_scale="Blues")
        plotly_dark_layout(fig_cm, height=350)
        st.plotly_chart(fig_cm, use_container_width=True)

        # Confidence distribution
        st.markdown('<div class="section-header">Confidence Score Distribution</div>', unsafe_allow_html=True)
        fig_conf = go.Figure()
        for label, color in PALETTE.items():
            if label == "NEUTRAL":
                continue
            sub = preds[preds["label"] == label][col_conf].dropna()
            fig_conf.add_trace(go.Histogram(x=sub, name=label, opacity=0.7,
                                            marker_color=color, nbinsx=40))
        fig_conf.update_layout(barmode="overlay")
        plotly_dark_layout(fig_conf, title="P(GOOD) by Actual Label", height=350)
        st.plotly_chart(fig_conf, use_container_width=True)

        # Feature importances (if Phase3_results.md exists)
        results_md = Path("docxxs/Phase3_results.md")
        if results_md.exists():
            with st.expander("📄 Full Phase 3 Report"):
                st.markdown(results_md.read_text(encoding="utf-8"))


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4: Market Reaction
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📉 Market Reaction":
    st.markdown("# 📉 Market Reaction Test")
    st.markdown("Comparing abnormal returns for high-confidence GOOD vs BAD predictions.")

    if D["market"] is None:
        st.warning("market_reaction.csv not found. Run `python market_reaction.py` first.")
    else:
        mr = D["market"]
        hi = mr["confidence_good"].quantile(0.70)
        lo = mr["confidence_good"].quantile(0.30)
        hi_good = mr[mr["confidence_good"] >= hi]
        hi_bad  = mr[mr["confidence_good"] <= lo]

        for win, col_name, label in [
            ("Immediate (day 0→+1)", "immediate_ar", "Immediate"),
            ("Drift (day +2→+60)",   "drift_ar",     "Drift"),
        ]:
            st.markdown(f'<div class="section-header">{win} Abnormal Return</div>',
                        unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            with c1:
                fig = go.Figure()
                for grp, df_grp, color in [
                    ("High-conf GOOD", hi_good, PALETTE["GOOD"]),
                    ("High-conf BAD",  hi_bad,  PALETTE["BAD"]),
                ]:
                    vals = df_grp[col_name].dropna() * 100
                    fig.add_trace(go.Histogram(x=vals, name=grp, opacity=0.7,
                                               marker_color=color, nbinsx=50))
                fig.update_layout(barmode="overlay",
                                  xaxis_title="Abnormal Return (%)")
                plotly_dark_layout(fig, title=f"{label} Window Distribution", height=350)
                st.plotly_chart(fig, use_container_width=True)

            with c2:
                means = {
                    "High-conf GOOD": hi_good[col_name].mean() * 100,
                    "High-conf BAD":  hi_bad[col_name].mean()  * 100,
                }
                fig2 = go.Figure(go.Bar(
                    x=list(means.keys()), y=list(means.values()),
                    marker_color=[PALETTE["GOOD"], PALETTE["BAD"]],
                ))
                fig2.update_layout(yaxis_title="Mean Abnormal Return (%)")
                plotly_dark_layout(fig2, title="Mean Return by Group", height=350)
                st.plotly_chart(fig2, use_container_width=True)

        results_md = Path("docxxs/Phase4_results.md")
        if results_md.exists():
            with st.expander("📄 Full Phase 4 Report"):
                st.markdown(results_md.read_text(encoding="utf-8"))


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5: Backtest
# ══════════════════════════════════════════════════════════════════════════════
elif page == "💰 Backtest":
    st.markdown("# 💰 Trading Backtest")
    st.markdown("Long high-confidence GOOD / Short high-confidence BAD strategy.")

    if D["equity"] is None:
        st.warning("equity_curve.csv not found. Run `python backtest.py` first.")
    else:
        eq = D["equity"]
        bt = D["backtest"]

        # Summary metrics
        total_ret = eq["equity"].iloc[-1] - 1
        returns   = eq["net_return"].values
        win_rate  = (returns > 0).mean()
        roll_max  = np.maximum.accumulate(eq["equity"].values)
        max_dd    = ((eq["equity"].values - roll_max) / roll_max).min()
        trades_yr = 252 / 20
        sharpe    = returns.mean() / (returns.std() + 1e-9) * np.sqrt(trades_yr)

        c1, c2, c3, c4 = st.columns(4)
        for col, val, lbl in zip(
            [c1, c2, c3, c4],
            [f"{total_ret:.1%}", f"{win_rate:.1%}", f"{sharpe:.2f}", f"{max_dd:.1%}"],
            ["Total Return", "Win Rate", "Sharpe Ratio", "Max Drawdown"]
        ):
            col.markdown(f"""
            <div class="metric-card">
              <div class="metric-value">{val}</div>
              <div class="metric-label">{lbl}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("")

        # Equity curve
        st.markdown('<div class="section-header">Equity Curve</div>', unsafe_allow_html=True)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=eq["trade_num"], y=(eq["equity"] - 1) * 100,
            mode="lines", name="Strategy",
            line=dict(color="#4f7cff", width=2),
            fill="tozeroy", fillcolor="rgba(79,124,255,0.1)",
        ))
        fig.add_hline(y=0, line_dash="dash", line_color=GRID)
        fig.update_layout(xaxis_title="Trade #", yaxis_title="Cumulative Return (%)")
        plotly_dark_layout(fig, height=400)
        st.plotly_chart(fig, use_container_width=True)

        # Return distribution
        st.markdown('<div class="section-header">Per-Trade Return Distribution</div>',
                    unsafe_allow_html=True)
        fig2 = go.Figure(go.Histogram(
            x=returns * 100, nbinsx=50,
            marker_color="#4f7cff", opacity=0.8,
        ))
        fig2.add_vline(x=0, line_dash="dash", line_color=PALETTE["BAD"])
        fig2.update_layout(xaxis_title="Net Return per Trade (%)", yaxis_title="Count")
        plotly_dark_layout(fig2, height=350)
        st.plotly_chart(fig2, use_container_width=True)

        results_md = Path("docxxs/Phase5_results.md")
        if results_md.exists():
            with st.expander("📄 Full Phase 5 Report"):
                st.markdown(results_md.read_text(encoding="utf-8"))
