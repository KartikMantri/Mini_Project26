"""
Phase 4 (v2) — Market-Reaction Test on the FinBERT Phase 3 score
==================================================================
Does the market fully price what an earnings call's language reveals?

The Phase 3 score (data/predictions_finbert.csv, trained only on the immediate reaction and
always out-of-sample) is compared with
  - the immediate window, day -1 -> +1: does the market react to the language at all?
  - the drift window, day +2 -> +60: does the score keep predicting returns afterwards?
If it predicts the drift, the market under-reacted to the language.

Tests (the standard event-study set, same definitions as Velu's VADER Phase 4 so the two
can be compared like for like):
  1. Quintiles: calls ranked by score within each calendar quarter, mean abnormal return per
     quintile. Top-minus-bottom spread per quarter, t-statistic across quarters with
     Newey-West (1 lag) errors, because consecutive quarters' 60-day windows overlap.
  2. Regressions of drift on the score's within-quarter rank (0 = lowest, 1 = highest, so the
     coefficient = top call minus bottom call), adding the immediate reaction and pre-call
     return/volatility as controls. Quarter fixed effects, errors clustered by quarter.
  3. A language-free model (market + reaction only) through the same drift test.
  4. Year by year, cumulative abnormal-return path, and a (post-hoc) pandemic check.
Returns are winsorised at 1% / 99%.

Reads:  data/predictions_finbert.csv, data/prices.csv,
        data/_cache/velu_phase4_predictions.csv (Velu's VADER scores, for comparison;
        from `git show origin/vadar-phase4:data/predictions.csv`)
Writes: data/market_reaction_finbert.csv, data/car_paths_finbert.csv,
        docxxs/Phase4_finbert_results.md

Usage:
    python market_reaction_finbert.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

from event_returns import BAD_PRICE_TICKERS, PRE_WIN, EventReturns, load_prices

OURS_CSV   = Path("data/predictions_finbert.csv")
VELU_CSV   = Path("data/_cache/velu_phase4_predictions.csv")
OUT_CSV    = Path("data/market_reaction_finbert.csv")
PATHS_CSV  = Path("data/car_paths_finbert.csv")
RESULTS_MD = Path("docxxs/Phase4_finbert_results.md")

PERIODS = {"holdout": (2018, 2025), "valid": (2012, 2017), "all": (2012, 2025)}
PERIOD_NAME = {"holdout": "Holdout years 2018–2025", "valid": "Validation years 2012–2017",
               "all": "All years 2012–2025"}
PATH_DAYS = 60
WINSOR = (0.01, 0.99)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load(csv, score_col, nolang_col, er):
    d = pd.read_csv(csv, parse_dates=["call_date"])
    d = d.rename(columns={score_col: "score", nolang_col: "nolang"})
    d = d[~d["ticker"].isin(BAD_PRICE_TICKERS)]   # their stored returns are not real
    d = d.dropna(subset=["score", "imm_ar", "drift_ar"]).copy()
    d["pre_vol"] = [er.volatility(t, c, PRE_WIN) for t, c in zip(d["ticker"], d["call_date"])]
    for c in ("imm_ar", "drift_ar", "pre_ar", "pre_vol"):
        lo, hi = d[c].quantile(WINSOR)
        d[c] = d[c].clip(lo, hi)
    d["year"] = d["call_date"].dt.year
    d["quarter"] = d["call_date"].dt.to_period("Q").astype(str)
    return d[["ticker", "call_date", "year", "quarter", "score", "nolang",
              "imm_ar", "drift_ar", "pre_ar", "pre_vol"]].reset_index(drop=True)


def add_ranks(d, col="score", prefix=""):
    g = d.groupby("quarter")[col]
    r = g.rank(method="first")
    n = g.transform("count")
    d[prefix + "rank"] = (r - 1) / (n - 1)
    d[prefix + "q"] = np.minimum((d[prefix + "rank"] * 5).astype(int), 4) + 1
    return d


def in_period(d, years):
    return d[(d["year"] >= years[0]) & (d["year"] <= years[1])].copy()


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def newey_west(x, lags=1):
    """Mean of a time series and its Newey-West t-statistic / two-sided p-value."""
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    T = len(x)
    e = x - x.mean()
    var = e @ e / T
    for k in range(1, lags + 1):
        var += 2 * (1 - k / (lags + 1)) * (e[k:] @ e[:-k]) / T
    t = x.mean() / np.sqrt(var / T)
    return x.mean(), t, 2 * stats.t.sf(abs(t), T - 1), T


def spread_by_quarter(d, ret, qcol="q"):
    top = d[d[qcol] == 5].groupby("quarter")[ret].mean()
    bot = d[d[qcol] == 1].groupby("quarter")[ret].mean()
    return (top - bot).dropna()


def fe_ols(d, y, xs):
    """OLS with quarter fixed effects and errors clustered by quarter."""
    d = d.dropna(subset=[y] + xs)
    Y = d[y] - d.groupby("quarter")[y].transform("mean")
    X = d[xs] - d.groupby("quarter")[xs].transform("mean")
    Xv, Yv = X.values, Y.values
    XtX_inv = np.linalg.inv(Xv.T @ Xv)
    beta = XtX_inv @ Xv.T @ Yv
    resid = Yv - Xv @ beta
    meat = np.zeros((len(xs), len(xs)))
    for _, idx in d.groupby("quarter").indices.items():
        s = Xv[idx].T @ resid[idx]
        meat += np.outer(s, s)
    G, N, K = d["quarter"].nunique(), len(d), len(xs) + d["quarter"].nunique()
    V = XtX_inv @ meat @ XtX_inv * G / (G - 1) * (N - 1) / (N - K)
    se = np.sqrt(np.diag(V))
    t = beta / se
    p = 2 * stats.t.sf(np.abs(t), G - 1)
    return {x: (b, tt, pp) for x, b, tt, pp in zip(xs, beta, t, p)}, N, G


def stars(p):
    # escaped so markdown shows the stars instead of reading them as bold/italic
    return "\\*\\*\\*" if p < 0.01 else "\\*\\*" if p < 0.05 else "\\*" if p < 0.10 else ""


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

REGS = [("(1) score only", ["rank"]),
        ("(2) + immediate reaction", ["rank", "imm_ar"]),
        ("(3) + pre-call return, volatility", ["rank", "imm_ar", "pre_ar", "pre_vol"])]


def analyse(d):
    d = add_ranks(d)
    out = {"n": len(d)}
    out["quint"] = d.groupby("q").agg(calls=("score", "size"), imm=("imm_ar", "mean"),
                                      drift=("drift_ar", "mean"))
    out["imm_spread"] = newey_west(spread_by_quarter(d, "imm_ar"))
    out["drift_spread"] = newey_west(spread_by_quarter(d, "drift_ar"))
    if d["nolang"].notna().any():
        dn = add_ranks(d.dropna(subset=["nolang"]).copy(), "nolang", "n_")
        out["nolang_spread"] = newey_west(spread_by_quarter(dn, "drift_ar", "n_q"))
    out["regs"] = [(name, *fe_ols(d, "drift_ar", xs)) for name, xs in REGS]
    out["imm_reg"] = fe_ols(d, "imm_ar", ["rank"])[0]["rank"]
    return out, d


def by_year(d):
    rows = []
    for y, g in d.groupby("year"):
        ok = g["imm_ar"].notna()
        auc = roc_auc_score(g.loc[ok, "imm_ar"] > 0, g.loc[ok, "score"])
        imm = spread_by_quarter(g, "imm_ar").mean()
        drift = spread_by_quarter(g, "drift_ar").mean()
        rows.append((y, len(g), auc, imm, drift))
    return rows


def car_paths(d, er):
    """Cumulative abnormal return from the close of day -1 to day 0..60, by quintile."""
    paths = {q: [] for q in range(1, 6)}
    for t, c, q in zip(d["ticker"], d["call_date"], d["q"]):
        hit = er._locate(t, c, -1, PATH_DAYS)
        if hit is None:
            continue
        stock, mkt, a, b = hit
        paths[q].append((stock[a + 1:b + 1] / stock[a] - 1) - (mkt[a + 1:b + 1] / mkt[a] - 1))
    out = pd.DataFrame({f"Q{q}": np.nanmean(np.vstack(p), axis=0) for q, p in paths.items()})
    out.index.name = "day"
    return out


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def pct(x):
    return f"{x * 100:+.2f}%"


def spread_txt(s):
    m, t, p, T = s
    return f"**{pct(m)}** {stars(p)} (t {t:.1f}, p {p:.3f}, {T} quarters)"


def section(name, res):
    q = res["quint"]
    L = [f"## {PERIOD_NAME[name]}", "",
         "### Average abnormal return by score quintile", "",
         "| Quintile | Calls | Immediate (−1 → +1) | Drift (+2 → +60) |", "|---|---|---|---|"]
    for i, r in q.iterrows():
        lab = {1: "1 (lowest score)", 5: "5 (highest score)"}.get(i, str(i))
        L.append(f"| {lab} | {int(r.calls):,} | {pct(r.imm)} | {pct(r.drift)} |")
    L += ["", f"- **Top − bottom, immediate:** {spread_txt(res['imm_spread'])}",
          f"- **Top − bottom, drift:** {spread_txt(res['drift_spread'])}"]
    if "nolang_spread" in res:
        L.append(f"- Language-free model (market + reaction only), drift: "
                 f"{spread_txt(res['nolang_spread'])}")
    L += ["", "### Regressions: drift (+2 → +60) on score rank", "",
          "Coefficient on *score rank* = drift of the highest-ranked call minus the lowest. "
          "Quarter fixed effects; errors clustered by quarter.", "",
          "| Model | Score rank | Immediate reaction | Pre-call return | Pre-call volatility | Calls | Quarters |",
          "|---|---|---|---|---|---|---|"]
    for name_, coefs, N, G in res["regs"]:
        cells = []
        for x in ("rank", "imm_ar", "pre_ar", "pre_vol"):
            if x in coefs:
                b, t, p = coefs[x]
                v = pct(b) if x == "rank" else f"{b:+.3f}"
                cells.append(f"{v}{stars(p)} (t {t:.1f})")
            else:
                cells.append("")
        L.append(f"| {name_} | " + " | ".join(cells) + f" | {N:,} | {G} |")
    b, t, p = res["imm_reg"]
    L += ["", f"Immediate reaction on score rank: {pct(b)}{stars(p)} (t {t:.1f}).", ""]
    return L


def headline_row(label, res):
    m_i, t_i, _, _ = res["imm_spread"]
    m_d, t_d, p_d, T = res["drift_spread"]
    b, t, p = res["regs"][2][1]["rank"]
    return (f"| {label} | {res['n']:,} | {pct(m_i)} (t {t_i:.1f}) | "
            f"{pct(m_d)}{stars(p_d)} (t {t_d:.1f}, p {p_d:.3f}) | "
            f"{pct(b)}{stars(p)} (t {t:.1f}, p {p:.3f}) |")


HEAD = ("| Run | Calls | Immediate top − bottom | Drift top − bottom | Drift with controls (3) |\n"
        "|---|---|---|---|---|")


def main():
    er = EventReturns(load_prices())
    ours = load(OURS_CSV, "score_final", "score_nolang", er)
    velu = load(VELU_CSV, "score_reaction_lr", "score_nolang_lr", er)
    print(f"[Phase4] ours {len(ours):,} calls, Velu {len(velu):,} calls")

    res, ranked = {}, {}
    for per, yrs in PERIODS.items():
        res[per], ranked[per] = analyse(in_period(ours, yrs))
        m, t, p, T = res[per]["drift_spread"]
        print(f"[Phase4] {per:8s} drift top-bottom {pct(m)} (t {t:.2f}, p {p:.3f})")

    # Velu's scores through the same code; and both on the calls they share
    vres = {per: analyse(in_period(velu, yrs))[0] for per, yrs in PERIODS.items()}
    keys = ["ticker", "call_date"]
    common = ours.merge(velu[keys], on=keys)
    v_common = velu.merge(ours[keys], on=keys)
    cres = {per: (analyse(in_period(common, yrs))[0], analyse(in_period(v_common, yrs))[0])
            for per, yrs in (("holdout", PERIODS["holdout"]), ("all", PERIODS["all"]))}

    # Year by year, CAR paths, post-hoc pandemic check
    years = by_year(ranked["all"])
    paths = car_paths(ranked["holdout"], er)
    paths.to_csv(PATHS_CSV)
    h = in_period(ours, PERIODS["holdout"])
    cd = h["call_date"]
    robust = [("Holdout 2018–2025, all calls (main result)", h),
              ("Without the crash window (calls Dec 2019 – Jun 2020)",
               h[(cd < "2019-12-01") | (cd >= "2020-07-01")]),
              ("Without all of 2020", h[h["year"] != 2020]),
              ("Without 2020–2021", h[~h["year"].isin([2020, 2021])])]
    robust = [(n, analyse(x.copy())[0]) for n, x in robust]

    out = ranked["all"][["ticker", "call_date", "quarter", "score", "rank", "q",
                         "imm_ar", "drift_ar", "pre_ar", "pre_vol"]]
    out.to_csv(OUT_CSV, index=False)

    # ---- markdown ----
    H = res["holdout"]
    m_i = H["imm_spread"][0]
    m_d, t_d, p_d, _ = H["drift_spread"]
    b3, t3, p3 = H["regs"][2][1]["rank"]
    share = m_d / (m_i + m_d)
    A = res["all"]
    L = [
        "# Phase 4 (v2) — Market-Reaction Test on the FinBERT Score",
        "",
        "Does the market fully price what an earnings call's language reveals? The Phase 3",
        "FinBERT score ([`Phase3_finbert_results.md`](Phase3_finbert_results.md); trained only on the",
        "immediate reaction, always out-of-sample) is compared with the immediate window",
        "(day −1 → +1) and the drift window (day +2 → +60). If the score also predicts the drift,",
        "the market under-reacted to the language.",
        "",
        "Method: calls are ranked by score within each calendar quarter; returns are abnormal",
        "(stock minus S&P 500) and winsorised at 1% / 99%; spreads are tested across quarters with",
        "Newey-West (1 lag) t-statistics; regressions use quarter fixed effects with errors clustered",
        "by quarter. These are the same definitions as Velu's VADER Phase 4, so the numbers compare",
        "directly. Regenerate with `python market_reaction_finbert.py`.",
        "",
        "## Answer (holdout years 2018–2025)",
        "",
        f"- **Immediate:** top-20% calls beat bottom-20% calls by **{pct(m_i)}** over day −1 → +1"
        f" (t {H['imm_spread'][1]:.1f}). The market reacts to what the language says.",
        f"- **Drift:** they keep beating them by **{pct(m_d)}** over day +2 → +60 "
        f"(t {t_d:.1f}, p {p_d:.3f}, {H['drift_spread'][3]} quarters).",
        f"- **With controls** (immediate reaction, pre-call return and volatility), highest- vs "
        f"lowest-ranked call: **{pct(b3)}** (t {t3:.1f}, p {p3:.3f}).",
        f"- Of the total top-minus-bottom move, **{share:.0%}** came after the reaction window.",
        "",
        f"Over 2012–2025 ({A['drift_spread'][3]} quarters): drift top − bottom "
        f"{pct(A['drift_spread'][0])} (t {A['drift_spread'][1]:.1f}, p {A['drift_spread'][2]:.3f}); "
        f"with controls {pct(A['regs'][2][1]['rank'][0])} "
        f"(t {A['regs'][2][1]['rank'][1]:.1f}, p {A['regs'][2][1]['rank'][2]:.3f}).",
        "",
        "## FinBERT vs Velu's VADER Phase 4",
        "",
        "Velu's scores are `score_reaction_lr` from his `vadar-phase4` branch (his Phase 3 with",
        "the extra `text_delta` input), run through **this** script. His own report gave",
        "holdout drift +1.20% (t 1.8) and +1.15% with controls; small differences below come from",
        "implementation details.",
        "",
        "**Each run on its own calls** (ours includes the old NEUTRAL calls):",
        "",
        HEAD,
        headline_row("FinBERT, holdout 2018–2025", res["holdout"]),
        headline_row("VADER (Velu), holdout 2018–2025", vres["holdout"]),
        headline_row("FinBERT, 2012–2025", res["all"]),
        headline_row("VADER (Velu), 2012–2025", vres["all"]),
        "",
        "**Same calls only** (calls in both runs):",
        "",
        HEAD,
        headline_row("FinBERT, holdout 2018–2025", cres["holdout"][0]),
        headline_row("VADER (Velu), holdout 2018–2025", cres["holdout"][1]),
        headline_row("FinBERT, 2012–2025", cres["all"][0]),
        headline_row("VADER (Velu), 2012–2025", cres["all"][1]),
        "",
    ]
    for per in ("holdout", "valid", "all"):
        L += section(per, res[per])
    L += ["## Robustness: holdout without the pandemic period", "",
          "Same exclusions as Velu's check. **Post-hoc:** these were chosen after seeing that",
          "2020–2021 were weak years in his run, so they explain the main result; they don't replace it.",
          "", HEAD]
    L += [headline_row(n, r) for n, r in robust]
    L += ["", "## Year by year (2012–2025)", "",
          "| Year | Calls | Reaction AUC | Immediate spread | Drift spread |", "|---|---|---|---|---|"]
    L += [f"| {y} | {n:,} | {auc:.3f} | {pct(i)} | {pct(dr)} |" for y, n, auc, i, dr in years]
    L += ["", "## Cumulative abnormal return path (holdout, from the close before the call)", "",
          "| Day | Q1 (lowest) | Q3 | Q5 (highest) | Q5 − Q1 |", "|---|---|---|---|---|"]
    for day in (0, 1, 5, 10, 20, 40, 60):
        r = paths.iloc[day]
        L.append(f"| +{day} | {pct(r.Q1)} | {pct(r.Q3)} | {pct(r.Q5)} | {pct(r.Q5 - r.Q1)} |")
    L += ["", f"Full paths: `{PATHS_CSV.as_posix()}`.", "",
          "`*` p < 0.10, `**` p < 0.05, `***` p < 0.01.", "",
          "## Caveats",
          "- Drift windows of calls in the same quarter overlap, so calls are not independent; the",
          "  tests work across quarters and cluster by quarter to account for this.",
          "- The 200 tickers are today's S&P 500 members (survivorship bias); it affects high- and",
          "  low-score groups alike but may flatter long-run returns.",
          "- No trading costs here; Phase 5 tests whether the drift is large enough to trade.", ""]
    RESULTS_MD.write_text("\n".join(L), encoding="utf-8")
    print(f"[Phase4] saved {RESULTS_MD}, {OUT_CSV}, {PATHS_CSV}")


if __name__ == "__main__":
    main()
