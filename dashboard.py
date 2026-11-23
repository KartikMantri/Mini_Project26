"""
Phase 6 — Dashboard
====================
A Streamlit dashboard over the FinBERT pipeline's committed outputs (no raw transcripts needed):

  Overview         headline numbers and the pipeline
  Call explorer    any call: its score, language profile, price path and trade
  Language         the 18 FinBERT / dictionary features, GOOD vs BAD, correlations
  Classifier       out-of-sample AUC by year and by model, score vs outcome
  Market reaction  the Phase 4 event study (quintiles, drift paths, quarterly gaps, regressions)
  Strategies       the Phase 5 backtests A-G against the S&P 500
  Method           rules, data and limitations

Reads (all in data/):
  features_finbert.csv, predictions_finbert.csv, market_reaction_finbert.csv,
  car_paths_finbert.csv, prices.csv, backtest_finbert_daily.csv, backtest_finbert_trades.csv,
  backtest_finbert_2to1_l15s15_daily.csv (+ _l5s5_, _l10s5_ if present)

Usage:
    streamlit run dashboard.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy import stats
from sklearn.metrics import roc_auc_score

from market_reaction_finbert import fe_ols, newey_west, spread_by_quarter

DATA = Path(__file__).parent / "data"
HOLDOUT, VALID, ALL = (2018, 2025), (2012, 2017), (2012, 2025)
PERIODS = {"Holdout 2018–25": HOLDOUT, "Validation 2012–17": VALID, "All 2012–25": ALL}

# ── palette ──────────────────────────────────────────────────────────────────
INK, MUTED, FAINT = "#d7dee8", "#7d8899", "rgba(255,255,255,0.06)"
UP, DOWN, BLUE, AMBER = "#3ddc97", "#ff6b6b", "#5aa9ff", "#ffc857"
QCOL = ["#ff6b6b", "#ff9f8a", "#8a94a6", "#8fc4ff", "#5aa9ff"]

st.set_page_config(page_title="Decoding the Game", page_icon="📈", layout="wide",
                   initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=JetBrains+Mono:wght@400;600&display=swap');
html, body, [class*="css"], .stMarkdown, p, li { font-family: 'Inter', sans-serif; }
.stApp {
  background:
    radial-gradient(1100px 520px at 85% -8%, rgba(61,220,151,0.10), transparent 60%),
    radial-gradient(900px 480px at -10% 105%, rgba(90,169,255,0.08), transparent 60%),
    repeating-linear-gradient(0deg, rgba(255,255,255,0.022) 0 1px, transparent 1px 44px),
    repeating-linear-gradient(90deg, rgba(255,255,255,0.022) 0 1px, transparent 1px 44px),
    #070b12;
}
[data-testid="stSidebar"] { background: rgba(9,14,22,0.92); border-right: 1px solid #141d2b; }
[data-testid="stHeader"] { background: transparent; }
.block-container { padding-top: 2.2rem; max-width: 1320px; }
h1, h2, h3 { font-weight: 500 !important; letter-spacing: -0.01em; color: #eef3f8 !important; }
.tape { overflow: hidden; white-space: nowrap; border-top: 1px solid #16202e;
        border-bottom: 1px solid #16202e; padding: 6px 0; margin: 0 0 1.4rem 0;
        font-family: 'JetBrains Mono', monospace; font-size: 12px; color: #7d8899; }
.tape span.track { display: inline-block; padding-left: 100%; animation: tape 48s linear infinite; }
.tape b { color: #d7dee8; font-weight: 600; } .tape i { font-style: normal; color: #3ddc97; }
@keyframes tape { from { transform: translateX(0); } to { transform: translateX(-100%); } }
.kpi { background: rgba(255,255,255,0.025); border: 1px solid #16202e; border-radius: 10px;
       padding: 14px 16px 12px; height: 100%; }
.kpi .l { font-size: 11px; letter-spacing: .12em; text-transform: uppercase; color: #7d8899; }
.kpi .v { font-family: 'JetBrains Mono', monospace; font-size: 26px; color: #eef3f8; margin-top: 4px; }
.kpi .d { font-family: 'JetBrains Mono', monospace; font-size: 12px; margin-top: 2px; color: #7d8899; }
.kpi .up { color: #3ddc97; } .kpi .dn { color: #ff6b6b; }
.eyebrow { font-family: 'JetBrains Mono', monospace; font-size: 11px; letter-spacing: .18em;
           color: #3ddc97; text-transform: uppercase; margin-bottom: 2px; }
.sub { color: #7d8899; font-size: 15px; margin-top: -6px; margin-bottom: 18px; }
.note { color: #7d8899; font-size: 13px; }
.pipe { display: flex; gap: 8px; flex-wrap: wrap; margin: 6px 0 4px; }
.pipe > div { flex: 1 1 120px; border: 1px solid #16202e; border-radius: 8px; padding: 10px 12px;
            background: rgba(255,255,255,0.02); }
.pipe .n { font-family: 'JetBrains Mono', monospace; color: #3ddc97; font-size: 11px; }
.pipe .t { color: #eef3f8; font-size: 14px; margin-top: 2px; }
.pipe .s { color: #7d8899; font-size: 12px; }
.brand { font-family: 'JetBrains Mono', monospace; font-size: 13px; letter-spacing: .2em; color: #3ddc97; }
div[data-testid="stDataFrame"] { border: 1px solid #16202e; border-radius: 8px; }
</style>""", unsafe_allow_html=True)


# ── helpers ──────────────────────────────────────────────────────────────────
def kpi(col, label, value, delta=None, good=None):
    cls = "" if good is None else ("up" if good else "dn")
    d = f'<div class="d {cls}">{delta}</div>' if delta else '<div class="d">&nbsp;</div>'
    col.markdown(f'<div class="kpi"><div class="l">{label}</div><div class="v">{value}</div>{d}</div>',
                 unsafe_allow_html=True)


def header(eyebrow, title, sub=None):
    st.markdown(f'<div class="eyebrow">{eyebrow}</div>', unsafe_allow_html=True)
    st.markdown(f"## {title}")
    if sub:
        st.markdown(f'<div class="sub">{sub}</div>', unsafe_allow_html=True)


def style(fig, height=360, legend=True):
    fig.update_layout(
        height=height, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color=INK, size=12),
        margin=dict(l=8, r=8, t=28, b=8), hovermode="x unified" if legend else "closest",
        showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, bgcolor="rgba(0,0,0,0)",
                    font=dict(size=11, color=MUTED)),
        hoverlabel=dict(bgcolor="#0d1420", bordercolor="#1d2838",
                        font=dict(family="JetBrains Mono, monospace", size=12, color=INK)))
    fig.update_xaxes(gridcolor=FAINT, zeroline=False, linecolor="#1d2838", tickfont=dict(color=MUTED))
    fig.update_yaxes(gridcolor=FAINT, zeroline=False, linecolor="#1d2838", tickfont=dict(color=MUTED))
    return fig


def pct(x, d=1, sign=True):
    return "—" if x is None or pd.isna(x) else (f"{x * 100:+.{d}f}%" if sign else f"{x * 100:.{d}f}%")


def in_years(obj, yrs, col=None):
    idx = obj.index if col is None else obj[col]
    y = idx.year if col is None else idx.dt.year
    return obj[(y >= yrs[0]) & (y <= yrs[1])]


def auc(y, s):
    m = ~(pd.isna(y) | pd.isna(s))
    return roc_auc_score(np.asarray(y)[m], np.asarray(s)[m]) if m.sum() > 50 else np.nan


# ── data ─────────────────────────────────────────────────────────────────────
FEATURES = {
    "prep_net_tone": "Prepared remarks tone", "qa_exec_net_tone": "Executives' Q&A tone",
    "analyst_tone": "Analysts' tone", "neg_share": "Share of negative sentences",
    "tone_p10": "10th-percentile sentence tone", "tone_std": "Spread of sentence tone",
    "tone_gap": "Scripted minus live tone", "delta_tone": "Tone change vs last call",
    "specific_fls_share": "Specific forward-looking share", "fls_density": "Forward-looking density",
    "lm_uncertainty_pct": "LM uncertainty words", "lm_weak_modal_pct": "LM weak modals",
    "lm_strong_modal_pct": "LM strong modals", "lm_negative_pct": "LM negative words",
    "numeric_density": "Numbers per word", "qa_directness": "Answers with numbers",
    "log_word_count": "Length (log words)", "fog_index": "Fog readability"}


@st.cache_data(show_spinner=False)
def load():
    f = pd.read_csv(DATA / "features_finbert.csv", parse_dates=["call_date"])
    p = pd.read_csv(DATA / "predictions_finbert.csv", parse_dates=["call_date"])
    p = p[p["ticker"] != "PARA"]                 # its price series is not Paramount's
    mr = pd.read_csv(DATA / "market_reaction_finbert.csv", parse_dates=["call_date"])
    car = pd.read_csv(DATA / "car_paths_finbert.csv")
    tr = pd.read_csv(DATA / "backtest_finbert_trades.csv",
                     parse_dates=["call_date", "entry_date", "exit_date"])
    for d in (p, mr, f):
        d["year"] = d["call_date"].dt.year
    return f, p, mr, car, tr


@st.cache_data(show_spinner=False)
def load_prices():
    return pd.read_csv(DATA / "prices.csv", index_col=0, parse_dates=True)


STRATS = {   # key: (label, colour, dash, file, column)
    "A": ("A · Market-neutral 1:1, 20/20", UP, "solid", "backtest_finbert_daily.csv", "ours_net"),
    "B": ("B · Tilted 2:1, 20/20", "#4dd0e1", "solid", "backtest_finbert_2to1_l15s15_daily.csv", "ours_2020"),
    "C": ("C · Tilted 2:1, 20/10", "#b18cff", "solid", "backtest_finbert_2to1_l15s15_daily.csv", "ours_2010"),
    "D": ("D · Tilted 2:1, 15/15 (our choice)", BLUE, "solid", "backtest_finbert_2to1_l15s15_daily.csv", "ours"),
    "E": ("E · Long-only, top 15%", AMBER, "solid", "backtest_finbert_2to1_l15s15_daily.csv", "longonly"),
    "F": ("F · Tilted 2:1, 5/5", "#ff8fab", "solid", "backtest_finbert_2to1_l5s5_daily.csv", "ours"),
    "G": ("G · Tilted 2:1, 10/5", "#a3e635", "solid", "backtest_finbert_2to1_l10s5_daily.csv", "ours"),
    "ALL": ("Buy every call (no score)", "#8a94a6", "dot", "backtest_finbert_2to1_l15s15_daily.csv", "all_calls"),
    "NOLANG": ("A with no-language score", "#5b6475", "dot", "backtest_finbert_daily.csv", "nolang_net"),
    "VADER": ("A with VADER score", "#f472b6", "dot", "backtest_finbert_daily.csv", "velu_net"),
    "SPX": ("S&P 500 (price index)", "#eef3f8", "dash", "backtest_finbert_daily.csv", "sp500"),
}


@st.cache_data(show_spinner=False)
def load_strategies():
    out, cache = {}, {}
    for k, (*_, fname, col) in STRATS.items():
        path = DATA / fname
        if not path.exists():
            continue
        if fname not in cache:
            cache[fname] = pd.read_csv(path, index_col=0, parse_dates=True)
        if col in cache[fname]:
            out[k] = cache[fname][col]
    return pd.DataFrame(out)


def strat_stats(r, sp):
    x = pd.concat([r, sp], axis=1, keys=["r", "sp"]).dropna()
    if len(x) < 60:
        return None
    w = (1 + x["r"]).cumprod()
    vol = x["r"].std() * np.sqrt(252)
    beta = np.cov(x["r"], x["sp"])[0, 1] / x["sp"].var()
    a_mean, a_t, a_p, _ = newey_west((x["r"] - beta * x["sp"]).values, 10)
    return {"Annual return": x["r"].mean() * 252, "Volatility": vol,
            "Sharpe": x["r"].mean() * 252 / vol, "Max drawdown": (w / w.cummax() - 1).min(),
            "Beta": beta, "Alpha / yr": a_mean * 252, "Alpha p": a_p, "100 →": 100 * w.iloc[-1]}


FEAT, PRED, MR, CAR, TRADES = load()
SR = load_strategies()


# ── computed headline numbers (live from the files) ──────────────────────────
@st.cache_data(show_spinner=False)
def headline():
    h = in_years(PRED, HOLDOUT, "call_date")
    r_auc = auc((h["imm_ar"] > 0).astype(float).where(h["imm_ar"].notna()), h["score_final"])
    mrh = in_years(MR, HOLDOUT, "call_date")
    g = spread_by_quarter(mrh, "drift_ar")
    gm, gt, gp, _ = newey_west(g.values, 1)
    sd = strat_stats(in_years(SR["D"], HOLDOUT), in_years(SR["SPX"], HOLDOUT))
    sa = strat_stats(in_years(SR["A"], HOLDOUT), in_years(SR["SPX"], HOLDOUT))
    return r_auc, gm, gp, (g > 0).mean(), sa, sd


R_AUC, GAP, GAP_P, GAP_POS, SA, SD = headline()


# ── sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="brand">DECODING THE GAME</div>', unsafe_allow_html=True)
    st.caption("Earnings-call language · market under-reaction")
    page = st.radio("Page", ["Overview", "Call explorer", "Language", "Classifier",
                             "Market reaction", "Strategies", "Method"],
                    label_visibility="collapsed")
    st.markdown("---")
    st.caption(f"{len(FEAT):,} calls · {FEAT['ticker'].nunique()} companies · "
               f"{FEAT['call_date'].min():%Y}–{FEAT['call_date'].max():%Y}")
    st.caption("Holdout 2018–2025 never used for any choice")
    st.caption("Kartik Mantri · Velu Kala · LNMIIT")

st.markdown(
    f'<div class="tape"><span class="track">'
    f'REACTION AUC <b>{R_AUC:.3f}</b> <i>▲</i> &nbsp;·&nbsp; DRIFT Q5−Q1 <b>{pct(GAP, 2)}</b> '
    f'<i>p {GAP_P:.3f}</i> &nbsp;·&nbsp; STRATEGY D <b>{pct(SD["Annual return"])}/yr</b> '
    f'SHARPE <b>{SD["Sharpe"]:.2f}</b> &nbsp;·&nbsp; ALPHA <b>{pct(SD["Alpha / yr"])}/yr</b> '
    f'<i>p {SD["Alpha p"]:.3f}</i> &nbsp;·&nbsp; MARKET-NEUTRAL A SHARPE <b>{SA["Sharpe"]:.2f}</b> '
    f'&nbsp;·&nbsp; S&P 500 MAX DD <b>−33.9%</b> &nbsp;·&nbsp; FINBERT · WALK-FORWARD · NO LOOK-AHEAD'
    f'</span></div>', unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
if page == "Overview":
    header("Phase 0 → 6", "Do stock prices under-react to how managers talk?",
           "We read 12,716 earnings calls with FinBERT, score each one out of sample, and test "
           "whether the score predicts returns after the market's first reaction.")
    c = st.columns(4)
    kpi(c[0], "Reaction AUC · holdout", f"{R_AUC:.3f}", "VADER version 0.588", True)
    kpi(c[1], "Drift Q5 − Q1 · days +2→+60", pct(GAP, 2),
        f"p {GAP_P:.3f} · {GAP_POS:.0%} of quarters > 0", True)
    kpi(c[2], "Strategy D · after costs", f"{pct(SD['Annual return'])}/yr",
        f"Sharpe {SD['Sharpe']:.2f} vs S&P 0.63", True)
    kpi(c[3], "Alpha beyond market risk", f"{pct(SD['Alpha / yr'])}/yr",
        f"p {SD['Alpha p']:.3f} · beta {SD['Beta']:.2f}", True)

    st.markdown("#### Pipeline")
    st.markdown("""<div class="pipe">
<div><div class="n">01</div><div class="t">Data</div><div class="s">12,789 calls, 198 firms</div></div>
<div><div class="n">02</div><div class="t">Language</div><div class="s">18 FinBERT / LM features</div></div>
<div><div class="n">03</div><div class="t">Classifier</div><div class="s">walk-forward score 0–1</div></div>
<div><div class="n">04</div><div class="t">Event study</div><div class="s">does score predict drift?</div></div>
<div><div class="n">05</div><div class="t">Backtest</div><div class="s">tradeable after costs?</div></div>
<div><div class="n">06</div><div class="t">Dashboard</div><div class="s">you are here</div></div>
</div>""", unsafe_allow_html=True)

    left, right = st.columns([1.35, 1])
    with left:
        st.markdown("#### Growth of 100 · holdout")
        fig = go.Figure()
        for k in ("A", "D", "E", "SPX"):
            if k in SR:
                r = in_years(SR[k], HOLDOUT).dropna()
                lab, colr, dash, *_ = STRATS[k]
                fig.add_trace(go.Scatter(x=r.index, y=100 * (1 + r).cumprod(), name=lab,
                                         line=dict(color=colr, width=2, dash=dash)))
        st.plotly_chart(style(fig, 330), config={"displayModeBar": False})
    with right:
        st.markdown("#### What we found")
        st.markdown(f"""
- **The language carries information.** The score predicts the stock's immediate reaction
  (AUC {R_AUC:.3f}); a model without language does not.
- **The market is slow.** Top-scored calls beat bottom-scored ones by a further
  **{pct(GAP, 2)}** over the next ~3 months, about 29% of the total move.
- **It survives controls** for the reaction itself, momentum and volatility (+1.31%, p 0.013).
- **It can be traded.** Market-neutral: {pct(SA['Annual return'])}/yr, Sharpe {SA['Sharpe']:.2f}.
  Strategy D (tilted 2:1, top/bottom 15%, our choice): {pct(SD['Annual return'])}/yr, Sharpe {SD['Sharpe']:.2f}, worst fall
  {pct(SD['Max drawdown'])} vs −33.9% for the S&P 500.
- **Caveats:** moderate statistics (t ≈ 2), survivorship bias, 2020 helps, 2012–17 weaker.
""")

# ═════════════════════════════════════════════════════════════════════════════
elif page == "Call explorer":
    header("Explore", "One earnings call, end to end",
           "Pick a company and a call to see its score, its language profile, what the stock did "
           "and whether the strategy traded it.")
    calls = PRED.merge(FEAT.drop(columns=["label", "abnormal_return", "year"]),
                       on=["ticker", "call_date"], how="left")
    tickers = sorted(calls["ticker"].unique())
    c1, c2 = st.columns([1, 2])
    tk = c1.selectbox("Company", tickers, index=tickers.index("SBUX") if "SBUX" in tickers else 0)
    sub = calls[calls["ticker"] == tk].sort_values("call_date", ascending=False)
    dates = [d.strftime("%Y-%m-%d") for d in sub["call_date"]]
    default = dates.index("2024-01-30") if tk == "SBUX" and "2024-01-30" in dates else 0
    ds = c2.selectbox("Call date", dates, index=default)
    row = sub.iloc[dates.index(ds)]
    day = row["call_date"].normalize()

    past = PRED[(PRED["call_date"] < day) & (PRED["call_date"] >= day - pd.Timedelta(days=365))]
    pctl = (past["score_final"] < row["score_final"]).mean() if len(past) > 200 else np.nan
    signal = ("LONG" if pctl >= 0.8 else "SHORT" if pctl <= 0.2 else "no trade") \
        if not pd.isna(pctl) else "—"

    c = st.columns(5)
    kpi(c[0], "Score", f"{row['score_final']:.3f}", f"P(reacts up)")
    kpi(c[1], "Vs past year's calls", "—" if pd.isna(pctl) else f"{pctl:.0%}",
        f"signal: {signal}", None if signal in ("no trade", "—") else signal == "LONG")
    kpi(c[2], "Pre-call · −61→−1", pct(row["pre_ar"]), "vs S&P 500",
        None if pd.isna(row["pre_ar"]) else row["pre_ar"] > 0)
    kpi(c[3], "Reaction · −1→+1", pct(row["imm_ar"]), "vs S&P 500",
        None if pd.isna(row["imm_ar"]) else row["imm_ar"] > 0)
    kpi(c[4], "Drift · +2→+60", pct(row["drift_ar"]),
        "GOOD" if row["drift_ar"] > 0 else "BAD" if row["drift_ar"] <= 0 else "pending",
        None if pd.isna(row["drift_ar"]) else row["drift_ar"] > 0)

    left, right = st.columns([1.3, 1])
    with left:
        st.markdown("#### Price path · stock vs S&P 500, rebased at day −1")
        px_ = load_prices()
        if tk in px_.columns:
            mkt = px_["^GSPC"].dropna()
            stock = px_[tk].reindex(mkt.index).ffill()
            i = mkt.index.searchsorted(day)
            lo, hi = max(i - 20, 0), min(i + 61, len(mkt) - 1)
            base = i - 1
            if base >= lo and not pd.isna(stock.iloc[base]):
                xs = np.arange(lo - i, hi - i + 1)
                s = 100 * stock.iloc[lo:hi + 1].values / stock.iloc[base]
                m = 100 * mkt.iloc[lo:hi + 1].values / mkt.iloc[base]
                fig = go.Figure()
                fig.add_vrect(x0=-1, x1=1, fillcolor="rgba(255,200,87,0.10)", line_width=0)
                fig.add_vrect(x0=2, x1=60, fillcolor="rgba(90,169,255,0.05)", line_width=0)
                fig.add_trace(go.Scatter(x=xs, y=s, name=tk, line=dict(color=UP, width=2.2)))
                fig.add_trace(go.Scatter(x=xs, y=m, name="S&P 500",
                                         line=dict(color="#eef3f8", width=1.4, dash="dash")))
                fig.update_xaxes(title_text="trading days from the call")
                st.plotly_chart(style(fig, 330), config={"displayModeBar": False})
                st.markdown('<div class="note">Amber: immediate window. Blue: drift window.</div>',
                            unsafe_allow_html=True)
        else:
            st.info("No price series for this ticker.")
    with right:
        st.markdown("#### Language profile · percentile among all calls")
        rows = []
        for k, lab in FEATURES.items():
            v = row.get(k)
            if pd.isna(v):
                continue
            rows.append((lab, (FEAT[k] < v).mean() * 100, v))
        prof = pd.DataFrame(rows, columns=["feature", "pct", "value"]).iloc[::-1]
        fig = go.Figure(go.Bar(
            x=prof["pct"] - 50, y=prof["feature"], orientation="h", base=50,
            marker_color=[BLUE if p >= 50 else AMBER for p in prof["pct"]],   # not good/bad
            customdata=prof["value"], hovertemplate="%{y}: %{customdata:.3f} · %{x:.0f}th pct<extra></extra>"))
        fig.add_vline(x=50, line_color="#2a3546")
        fig.update_xaxes(range=[0, 100], title_text="percentile (50 = typical call)")
        st.plotly_chart(style(fig, 470, legend=False), config={"displayModeBar": False})

    t = TRADES[(TRADES["strategy"] == "ours") & (TRADES["ticker"] == tk)
               & (TRADES["call_date"].dt.normalize() == day)]
    if len(t):
        t = t.iloc[0]
        st.markdown(
            f"**Strategy A traded this call:** {t['side']} from {t['entry_date']:%d %b %Y} to "
            f"{t['exit_date']:%d %b %Y} · stock {pct(t['stock_return'])}, S&P "
            f"{pct(t['market_return'])} · **{pct(t['net_excess_return'])} after costs**.")
    else:
        st.markdown('<div class="note">Strategy A did not trade this call (score in the middle 60%, '
                    'or before trading started in 2012).</div>', unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════════════
elif page == "Language":
    header("Phase 2", "Measuring how managers talk",
           "18 features per call from FinBERT (tone, forward-looking statements) and the "
           "Loughran–McDonald finance dictionary. GOOD = the stock beat the S&P 500 over days +2→+60.")
    pre = FEAT[FEAT["call_date"] < "2018-01-01"]
    feat = st.selectbox("Feature", list(FEATURES), format_func=lambda k: FEATURES[k],
                        index=list(FEATURES).index("qa_exec_net_tone"))
    g, b = pre.loc[pre["label"] == "GOOD", feat].dropna(), pre.loc[pre["label"] == "BAD", feat].dropna()
    t, p = stats.ttest_ind(g, b, equal_var=False)
    d = (g.mean() - b.mean()) / np.sqrt((g.var() + b.var()) / 2)
    c = st.columns(4)
    kpi(c[0], "Mean · GOOD calls", f"{g.mean():.4f}", f"n = {len(g):,}")
    kpi(c[1], "Mean · BAD calls", f"{b.mean():.4f}", f"n = {len(b):,}")
    kpi(c[2], "Welch t-test", f"t {t:+.2f}", f"p {p:.3f}", p < 0.05 if t > 0 else None)
    kpi(c[3], "Effect size", f"d {d:+.3f}", "tiny < 0.2 < small")
    left, right = st.columns([1.2, 1])
    with left:
        st.markdown("#### Distribution · calls before 2018 (holdout not inspected)")
        lo, hi = pre[feat].quantile([0.01, 0.99])
        fig = go.Figure()
        edges = np.linspace(lo, hi, 61)
        mids = (edges[:-1] + edges[1:]) / 2
        for lab, s, colr, fill in (("GOOD", g, UP, "rgba(61,220,151,0.10)"),
                                   ("BAD", b, DOWN, "rgba(255,107,107,0.08)")):
            h_, _ = np.histogram(s.clip(lo, hi), bins=edges)
            fig.add_trace(go.Scatter(x=mids, y=h_ / h_.sum(), name=lab, mode="lines",
                                     line=dict(color=colr, width=2, shape="hvh"),
                                     fill="tozeroy", fillcolor=fill))
        fig.update_yaxes(title_text="share of calls")
        st.plotly_chart(style(fig, 360), config={"displayModeBar": False})
    with right:
        st.markdown("#### Spearman correlation · all calls")
        corr = FEAT[list(FEATURES)].corr(method="spearman")
        fig = go.Figure(go.Heatmap(z=corr.values, x=list(corr.columns), y=list(corr.index),
                                   zmin=-1, zmax=1, colorscale=[[0, DOWN], [0.5, "#0d1420"], [1, BLUE]],
                                   hovertemplate="%{y} × %{x}: %{z:.2f}<extra></extra>",
                                   colorbar=dict(thickness=8, outlinewidth=0)))
        fig.update_xaxes(showticklabels=False)
        fig.update_yaxes(tickfont=dict(size=9))
        st.plotly_chart(style(fig, 360, legend=False), config={"displayModeBar": False})
    st.markdown('<div class="note">Each feature alone separates GOOD and BAD calls only slightly '
                '(Cohen\'s d ≤ 0.08); Phase 3 combines them. Only one near-duplicate pair: '
                'share of negative sentences vs 10th-percentile tone (ρ = −0.97).</div>',
                unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════════════
elif page == "Classifier":
    header("Phase 3", "Scoring each call, strictly out of sample",
           "Logistic regression on 8 FinBERT tone features + 2 market controls, refit every year on "
           "past calls only, trained to predict the immediate reaction (day −1→+1).")
    h = in_years(PRED, HOLDOUT, "call_date")
    y_imm = (h["imm_ar"] > 0).astype(float).where(h["imm_ar"].notna())
    y_dr = (h["drift_ar"] > 0).astype(float).where(h["drift_ar"].notna())
    models = {"score_final": "Final · LR, tone + market", "score_other_kind": "XGBoost",
              "score_finbert_only": "LR, all FinBERT/LM features",
              "score_direct": "Trained on GOOD/BAD directly", "score_nolang": "No language"}
    rows = [(lab, auc(y_imm, h[k]) if k not in ("score_direct", "score_nolang") else np.nan,
             auc(y_dr, h[k])) for k, lab in models.items() if k in h]
    c = st.columns(3)
    kpi(c[0], "Reaction AUC · holdout", f"{rows[0][1]:.3f}", "0.5 = coin flip", True)
    kpi(c[1], "GOOD/BAD AUC · holdout", f"{rows[0][2]:.3f}", "3-month drift is mostly noise")
    kpi(c[2], "Old VADER model", "0.588", "same calls, same method")
    left, right = st.columns([1.2, 1])
    with left:
        st.markdown("#### Reaction AUC by year")
        yrs = sorted(PRED["year"].unique())
        a = [auc((PRED.loc[PRED.year == y, "imm_ar"] > 0).astype(float),
                 PRED.loc[PRED.year == y, "score_final"]) for y in yrs]
        fig = go.Figure(go.Bar(x=yrs, y=a, marker_color=[BLUE if y >= 2018 else "#2c3e57" for y in yrs],
                               hovertemplate="%{x}: AUC %{y:.3f}<extra></extra>"))
        fig.add_hline(y=0.5, line_dash="dot", line_color=MUTED)
        fig.update_yaxes(range=[0.45, 0.7])
        st.plotly_chart(style(fig, 330, legend=False), config={"displayModeBar": False})
        st.markdown('<div class="note">Dark: validation years (used to choose inputs). '
                    'Blue: holdout.</div>', unsafe_allow_html=True)
    with right:
        st.markdown("#### Model comparison · holdout")
        cmp_ = pd.DataFrame(rows, columns=["Model", "Reaction AUC", "GOOD/BAD AUC"])
        for col in ("Reaction AUC", "GOOD/BAD AUC"):
            cmp_[col] = cmp_[col].map(lambda v: "—" if pd.isna(v) else f"{v:.3f}")
        st.dataframe(cmp_, hide_index=True)
        st.markdown('<div class="note">Strongest weights: analysts\' tone (+0.29) and the change in '
                    'tone since the firm\'s last call (+0.25).</div>', unsafe_allow_html=True)
    st.markdown("#### Score decile vs what happened · holdout")
    hh = h.dropna(subset=["imm_ar"]).copy()
    hh["decile"] = pd.qcut(hh["score_final"], 10, labels=False) + 1
    agg = hh.groupby("decile")[["imm_ar", "drift_ar"]].mean() * 100
    fig = go.Figure()
    fig.add_trace(go.Bar(x=agg.index, y=agg["imm_ar"], name="Immediate −1→+1", marker_color=AMBER))
    fig.add_trace(go.Bar(x=agg.index, y=agg["drift_ar"], name="Drift +2→+60", marker_color=BLUE))
    fig.update_xaxes(title_text="score decile (1 = most disliked)", dtick=1)
    fig.update_yaxes(title_text="avg abnormal return %")
    st.plotly_chart(style(fig, 320), config={"displayModeBar": False})

# ═════════════════════════════════════════════════════════════════════════════
elif page == "Market reaction":
    header("Phase 4", "Does the market price the language at once?",
           "Calls ranked by score within each quarter. If the top fifth keeps beating the bottom "
           "fifth after the first reaction, the market under-reacted.")
    per = st.segmented_control("Period", list(PERIODS), default="Holdout 2018–25", required=True)
    d = in_years(MR, PERIODS[per], "call_date")
    gi, gd = spread_by_quarter(d, "imm_ar"), spread_by_quarter(d, "drift_ar")
    mi, ti, pi, _ = newey_west(gi.values, 1)
    md, td, pdv, nq = newey_west(gd.values, 1)
    reg, n, G = fe_ols(d, "drift_ar", ["rank", "imm_ar", "pre_ar", "pre_vol"])
    c = st.columns(4)
    kpi(c[0], "Immediate Q5 − Q1", pct(mi, 2), f"t {ti:.1f}", mi > 0)
    kpi(c[1], "Drift Q5 − Q1", pct(md, 2), f"t {td:.1f} · p {pdv:.3f}", md > 0)
    kpi(c[2], "Drift with controls", pct(reg["rank"][0], 2),
        f"t {reg['rank'][1]:.1f} · p {reg['rank'][2]:.3f}", reg["rank"][0] > 0)
    kpi(c[3], "Quarters with gap > 0", f"{(gd > 0).sum()} / {nq}", f"{n:,} calls")
    left, right = st.columns(2)
    with left:
        st.markdown("#### Average return by score quintile")
        q = d.groupby("q")[["imm_ar", "drift_ar"]].mean() * 100
        fig = go.Figure()
        fig.add_trace(go.Bar(x=[f"Q{i}" for i in q.index], y=q["imm_ar"], name="Immediate",
                             marker_color=AMBER))
        fig.add_trace(go.Bar(x=[f"Q{i}" for i in q.index], y=q["drift_ar"], name="Drift",
                             marker_color=BLUE))
        st.plotly_chart(style(fig, 330), config={"displayModeBar": False})
    with right:
        st.markdown("#### Cumulative abnormal return · holdout")
        fig = go.Figure()
        for i, colr in zip(range(5, 0, -1), QCOL[::-1]):
            fig.add_trace(go.Scatter(x=CAR["day"], y=100 * CAR[f"Q{i}"], name=f"Q{i}",
                                     line=dict(color=colr, width=2.4 if i in (1, 5) else 1.3)))
        fig.add_vrect(x0=-1, x1=1, fillcolor="rgba(255,200,87,0.10)", line_width=0)
        fig.update_xaxes(title_text="trading days after the call")
        st.plotly_chart(style(fig, 330), config={"displayModeBar": False})
    st.markdown("#### Drift gap, quarter by quarter")
    fig = go.Figure(go.Bar(x=gd.index, y=100 * gd.values,
                           marker_color=[UP if v > 0 else DOWN for v in gd.values],
                           hovertemplate="%{x}: %{y:+.2f}%<extra></extra>"))
    fig.add_hline(y=100 * md, line_dash="dot", line_color=INK, annotation_position="top left",
                  annotation_text=f"mean {pct(md, 2)}", annotation_font_color=INK)
    st.plotly_chart(style(fig, 260, legend=False), config={"displayModeBar": False})
    st.markdown("#### Drift regression · quarter fixed effects, errors clustered by quarter")
    regs = []
    for name, xs in (("(1) score only", ["rank"]), ("(2) + immediate reaction", ["rank", "imm_ar"]),
                     ("(3) + pre-call return, volatility", ["rank", "imm_ar", "pre_ar", "pre_vol"])):
        r_, n_, _ = fe_ols(d, "drift_ar", xs)
        regs.append({"Model": name, "Score effect (top − bottom)": pct(r_["rank"][0], 2),
                     "t": round(r_["rank"][1], 2), "p": round(r_["rank"][2], 3), "Calls": n_})
    st.dataframe(pd.DataFrame(regs), hide_index=True)

# ═════════════════════════════════════════════════════════════════════════════
elif page == "Strategies":
    header("Phase 5", "Trading it · after costs, no look-ahead",
           "Signal from the past year's scores only. Enter day +2, exit day +60. 10 bps per trade, "
           "0.5%/yr short borrow. A is the pre-set main test; D is our chosen strategy; C–G are exploratory.")
    avail = [k for k in STRATS if k in SR]
    c1, c2 = st.columns([3, 1.2])
    pick = c1.multiselect("Strategies", avail, default=[k for k in ("A", "B", "D", "E", "SPX") if k in avail],
                          format_func=lambda k: STRATS[k][0])
    per = c2.segmented_control("Period", list(PERIODS), default="Holdout 2018–25", required=True)
    yrs = PERIODS[per]
    sp = in_years(SR["SPX"], yrs)
    if not pick:
        st.info("Pick at least one strategy.")
        st.stop()
    table = []
    for k in pick:
        s = strat_stats(in_years(SR[k], yrs), sp)
        if s:
            if k == "SPX":                      # alpha of the index against itself is meaningless
                s["Alpha / yr"], s["Alpha p"] = np.nan, np.nan
            table.append({"Strategy": STRATS[k][0], **s})
    tab = pd.DataFrame(table)

    best = tab.loc[tab["Sharpe"].idxmax()] if len(tab) else None
    c = st.columns(4)
    if best is not None:
        kpi(c[0], "Best Sharpe", f"{best['Sharpe']:.2f}", best["Strategy"].split(" · ")[0], True)
        top = tab.loc[tab["Annual return"].idxmax()]
        kpi(c[1], "Highest return", f"{pct(top['Annual return'])}/yr", top["Strategy"].split(" · ")[0], True)
        low = tab.loc[tab["Max drawdown"].idxmax()]
        kpi(c[2], "Shallowest worst fall", pct(low["Max drawdown"]), low["Strategy"].split(" · ")[0])
        spx = strat_stats(sp, sp)
        kpi(c[3], "S&P 500", f"{pct(spx['Annual return'])}/yr", f"Sharpe {spx['Sharpe']:.2f}")

    fig = go.Figure()
    for k in pick:
        r = in_years(SR[k], yrs).dropna()
        lab, colr, dash, *_ = STRATS[k]
        fig.add_trace(go.Scatter(x=r.index, y=100 * (1 + r).cumprod(), name=lab,
                                 line=dict(color=colr, width=2, dash=dash)))
    st.markdown("#### Growth of 100")
    st.plotly_chart(style(fig, 380), config={"displayModeBar": False})

    left, right = st.columns([1.4, 1])
    with left:
        st.markdown("#### Fall from previous peak")
        fig = go.Figure()
        for k in pick:
            r = in_years(SR[k], yrs).dropna()
            w = (1 + r).cumprod()
            lab, colr, dash, *_ = STRATS[k]
            fig.add_trace(go.Scatter(x=w.index, y=100 * (w / w.cummax() - 1), name=lab,
                                     line=dict(color=colr, width=1.4, dash=dash)))
        st.plotly_chart(style(fig, 300), config={"displayModeBar": False})
    with right:
        st.markdown("#### Risk vs return")
        fig = go.Figure()
        for sr_ in (0.5, 1.0):
            fig.add_trace(go.Scatter(x=[0, 25], y=[0, 25 * sr_], mode="lines", showlegend=False,
                                     line=dict(color="#1d2838", width=1), hoverinfo="skip"))
        for _, rw in tab.iterrows():
            k = next(k for k in pick if STRATS[k][0] == rw["Strategy"])
            fig.add_trace(go.Scatter(x=[rw["Volatility"] * 100], y=[rw["Annual return"] * 100],
                                     mode="markers+text", text=[k if k != "SPX" else "S&P"],
                                     textposition="top center", name=rw["Strategy"],
                                     marker=dict(size=13, color=STRATS[k][1],
                                                 line=dict(color="#070b12", width=2))))
        fig.update_xaxes(title_text="volatility %/yr", range=[0, 26])
        fig.update_yaxes(title_text="return %/yr")
        st.plotly_chart(style(fig, 300, legend=False), config={"displayModeBar": False})

    st.markdown("#### Scorecard")
    show = tab.copy()
    for col in ("Annual return", "Volatility", "Max drawdown"):
        show[col] = show[col] * 100
    show["Alpha / yr"] = show["Alpha / yr"].map(lambda v: "—" if pd.isna(v) else f"{v * 100:+.1f}%")
    show["Alpha p"] = show["Alpha p"].map(lambda v: "—" if pd.isna(v) else f"{v:.3f}")
    st.dataframe(show, hide_index=True, column_config={
        "Annual return": st.column_config.NumberColumn(format="%+.1f%%"),
        "Volatility": st.column_config.NumberColumn(format="%.1f%%"),
        "Sharpe": st.column_config.NumberColumn(format="%.2f"),
        "Max drawdown": st.column_config.NumberColumn(format="%.1f%%"),
        "Beta": st.column_config.NumberColumn(format="%.2f"),
        "100 →": st.column_config.NumberColumn(format="%.0f")})

    st.markdown("#### Year by year")
    yy = pd.DataFrame({STRATS[k][0].split(" · ")[0] if k != "SPX" else "S&P 500":
                       (1 + in_years(SR[k], yrs).dropna()).groupby(lambda x: x.year).prod() - 1
                       for k in pick})
    fig = go.Figure(go.Heatmap(z=100 * yy.T.values, x=[str(y) for y in yy.index], y=list(yy.columns),
                               colorscale=[[0, "#a8383c"], [0.5, "#0d1420"], [1, "#1f8a62"]], zmid=0,
                               zmin=-40, zmax=40,
                               text=np.vectorize(lambda v: "" if pd.isna(v) else f"{v:+.0f}")(100 * yy.T.values),
                               texttemplate="%{text}",
                               textfont=dict(family="JetBrains Mono", size=11, color="#eef3f8"),
                               hovertemplate="%{y} %{x}: %{z:+.1f}%<extra></extra>", showscale=False))
    st.plotly_chart(style(fig, 70 + 34 * len(yy.columns), legend=False), config={"displayModeBar": False})
    st.markdown('<div class="note">Raw returns after costs. The S&P 500 series excludes dividends; '
                'our stock prices include them, which flatters comparisons with the index by about '
                '1.5–2%/yr. Alpha = return beyond beta × S&P 500 (Newey–West p).</div>',
                unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════════════
else:
    header("Method", "How the numbers are made", "Everything is out of sample.")
    left, right = st.columns(2)
    with left:
        st.markdown("""
#### Rules
- **Abnormal return** = stock return − S&P 500 return over the same days.
- **Windows:** pre-call −61→−1 · immediate −1→+1 · drift +2→+60 trading days.
- **Labels:** GOOD if the drift abnormal return > 0.
- **Classifier:** refit every year on calls whose outcome was already known
  (7-day / 100-day embargo). Inputs chosen on 2012–17; 2018–25 used once.
- **Event study:** within-quarter quintiles, Newey–West t across quarters,
  quarter fixed effects with clustered errors.
- **Backtest:** cut-offs from the previous 365 days of scores only;
  10 bps per trade; 0.5%/yr borrow.
""")
    with right:
        st.markdown("""
#### Limitations
- **Survivorship:** today's large companies only.
- **Benchmark:** the S&P 500 series has no dividends.
- **Statistics:** t ≈ 2, not ≈ 3; several strategy variants were explored.
- **Periods:** 2020 contributes a lot; 2012–17 is weaker.
- **Trading:** close-price fills, no market impact.
- **Data fix:** PARA's price series is not Paramount's; its calls are excluded.

#### Reports
`docxxs/Phase3_finbert_results.md` · `Phase4_finbert_results.md` ·
`Phase5_finbert_results.md` · `Strategy_Report.pdf` · `project_flow_diagram.png`
""")
