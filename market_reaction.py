"""
Phase 4 — Market-Reaction Test (event study)
==============================================
Does the market fully price what the call's language reveals, or does it miss part of it?

Uses Phase 3's out-of-sample score for each call (data/predictions.csv) and asks:
  1. Immediate window (day -1 -> +1): does the score line up with the instant reaction?
     (It should: the model was trained on it. This shows the market reacts to the language.)
  2. Drift window (day +2 -> +60): does the score ALSO predict what happens afterwards?
     If yes, the market under-reacted to the language at first: an inefficiency.

Tests:
  - Quintiles: calls ranked by score within each calendar quarter; average drift of the top
    20% minus the bottom 20%, with a t-test across quarters (Newey-West, so neighbouring
    quarters' overlapping windows don't inflate significance).
  - Regression of drift on the score's rank, controlling for the immediate reaction, the
    pre-call 3-month return and volatility, with quarter fixed effects and standard errors
    clustered by quarter. If the score stays significant after controlling for the immediate
    reaction, the language carries information the price had not yet absorbed.
  - Same tests for a language-free model, as a comparison.

Returns are winsorized at the 1st/99th percentile within each period so a few extreme moves
(drift reaches +344% in the data) don't drive the averages.

Reads:  data/predictions.csv, data/prices.csv
Writes: data/market_reaction.csv, data/car_paths.csv, docxxs/Phase4_results.md

Usage:
    python market_reaction.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

from event_returns import PRE_WIN, EventReturns, load_prices

PREDICTIONS_CSV     = Path("data/predictions.csv")
MARKET_REACTION_CSV = Path("data/market_reaction.csv")
CAR_PATHS_CSV       = Path("data/car_paths.csv")
RESULTS_MD          = Path("docxxs/Phase4_results.md")

SCORE_COL  = "confidence_good"      # final Phase 3 model
NOLANG_COL = "score_nolang_lr"      # language-free comparison model
PERIODS = {
    "holdout":    (2018, 2025),     # never used for choosing anything in Phase 3
    "validation": (2012, 2017),
    "all":        (2012, 2025),
}
# Robustness check: the 2018-2025 holdout without the pandemic period. These exclusions were
# chosen AFTER seeing that 2020-2021 were weak years, so they are reported as a robustness
# check next to the main result, never in place of it.
PANDEMIC_EXCLUSIONS = {
    "Holdout 2018–2025, all calls (main result)": None,
    "Without the crash window (calls Dec 2019 – Jun 2020)": ("2019-12-01", "2020-06-30"),
    "Without all of 2020": ("2020-01-01", "2020-12-31"),
    "Without 2020–2021": ("2020-01-01", "2021-12-31"),
}
CAR_DAYS = (-1, 60)                 # day-by-day path from the close before the call
NW_LAGS  = 1                        # 60-trading-day windows overlap into the next quarter


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------

def winsorize(s, q=0.01):
    lo, hi = s.quantile(q), s.quantile(1 - q)
    return s.clip(lo, hi)


def newey_west_t(x, lags=NW_LAGS):
    """Mean of a time series, with a Newey-West t-statistic and two-sided p-value."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    u = x - x.mean()
    var = u @ u / n
    for lag in range(1, lags + 1):
        w = 1 - lag / (lags + 1)
        var += 2 * w * (u[lag:] @ u[:-lag]) / n
    se = np.sqrt(var / n)
    t = x.mean() / se
    return x.mean(), t, 2 * stats.t.sf(abs(t), n - 1), n


def ols_fe_cluster(df, y_col, x_cols, group_col):
    """OLS with fixed effects for group_col (by demeaning) and standard errors
    clustered by the same groups. Returns a DataFrame of coef / se / t / p."""
    d = df[[y_col] + x_cols + [group_col]].dropna()
    g = d[group_col].values
    demean = lambda a: a - pd.DataFrame(a).groupby(g).transform("mean").values
    y = demean(d[[y_col]].values.astype(float)).ravel()
    X = demean(d[x_cols].values.astype(float))
    xtx_inv = np.linalg.inv(X.T @ X)
    beta = xtx_inv @ X.T @ y
    resid = y - X @ beta
    meat = np.zeros((len(x_cols), len(x_cols)))
    for grp in np.unique(g):
        m = g == grp
        s = X[m].T @ resid[m]
        meat += np.outer(s, s)
    n, k, n_groups = len(y), len(x_cols) + len(np.unique(g)), len(np.unique(g))
    corr = n_groups / (n_groups - 1) * (n - 1) / (n - k)
    se = np.sqrt(np.diag(corr * xtx_inv @ meat @ xtx_inv))
    t = beta / se
    p = 2 * stats.t.sf(np.abs(t), n_groups - 1)
    return pd.DataFrame({"coef": beta, "se": se, "t": t, "p": p}, index=x_cols), n, n_groups


def stars(p):
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load():
    df = pd.read_csv(PREDICTIONS_CSV, parse_dates=["call_date"])
    df = df.rename(columns={"imm_ar": "immediate_ar"})
    df = df.dropna(subset=[SCORE_COL, "immediate_ar", "drift_ar"]).reset_index(drop=True)
    er = EventReturns(load_prices())
    df["pre_vol"] = [er.volatility(t, d, PRE_WIN) for t, d in zip(df["ticker"], df["call_date"])]
    df["year"] = df["call_date"].dt.year
    df["quarter"] = df["call_date"].dt.to_period("Q").astype(str)

    # Rank within each quarter, centred: -0.5 = lowest score, +0.5 = highest. A regression
    # coefficient on it reads as "drift of the highest-scored call minus the lowest".
    df["score_rank"] = df.groupby("quarter")[SCORE_COL].rank(pct=True) - 0.5
    df["quintile"] = df.groupby("quarter")[SCORE_COL].transform(
        lambda s: pd.qcut(s.rank(method="first"), 5, labels=False) + 1)
    return df, er


def period_frame(df, years):
    sub = df[(df["year"] >= years[0]) & (df["year"] <= years[1])].copy()
    for c in ("immediate_ar", "drift_ar", "pre_ar"):
        sub[c] = winsorize(sub[c])
    return sub


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def quintile_test(sub, rank_col="quintile"):
    """Mean returns by quintile, plus the top-minus-bottom spread tested across quarters."""
    table = sub.groupby(rank_col)[["immediate_ar", "drift_ar"]].mean()
    table["n"] = sub.groupby(rank_col).size()
    out = {"table": table}
    for col in ("immediate_ar", "drift_ar"):
        by_q = sub.groupby(["quarter", rank_col])[col].mean().unstack()
        spread = (by_q[5] - by_q[1]).sort_index()
        out[col] = newey_west_t(spread.values)
    return out


def quintile_test_score(sub, score_col):
    """Top-minus-bottom 20% drift spread for any score column (ranked within quarter)."""
    q = sub.groupby("quarter")[score_col].transform(
        lambda s: pd.qcut(s.rank(method="first"), 5, labels=False) + 1)
    by_q = sub.assign(_q=q).groupby(["quarter", "_q"])["drift_ar"].mean().unstack()
    return newey_west_t((by_q[5] - by_q[1]).sort_index().values)


def regressions(sub):
    specs = {
        "(1) score only":                   ["score_rank"],
        "(2) + immediate reaction":         ["score_rank", "immediate_ar"],
        "(3) + pre-call return, volatility": ["score_rank", "immediate_ar", "pre_ar", "pre_vol"],
    }
    out = {}
    for name, cols in specs.items():
        out[name] = ols_fe_cluster(sub, "drift_ar", cols, "quarter")
    out["immediate"] = ols_fe_cluster(sub, "immediate_ar", ["score_rank"], "quarter")
    return out


def pandemic_robustness(df):
    """Main holdout tests repeated with pandemic-era calls removed."""
    hold = df[(df["year"] >= PERIODS["holdout"][0]) & (df["year"] <= PERIODS["holdout"][1])]
    rows = []
    for name, window in PANDEMIC_EXCLUSIONS.items():
        sub = hold if window is None else hold[~hold["call_date"].between(*window)]
        sub = sub.copy()
        for c in ("immediate_ar", "drift_ar", "pre_ar"):
            sub[c] = winsorize(sub[c])
        q = quintile_test(sub)
        reg = ols_fe_cluster(sub, "drift_ar", ["score_rank", "immediate_ar", "pre_ar", "pre_vol"],
                             "quarter")[0].loc["score_rank"]
        rows.append({"sample": name, "calls": len(sub), "quarters": q["drift_ar"][3],
                     "imm": q["immediate_ar"][0], "imm_t": q["immediate_ar"][1],
                     "drift": q["drift_ar"][0], "drift_p": q["drift_ar"][2],
                     "reg": reg["coef"], "reg_p": reg["p"]})
    return pd.DataFrame(rows)


def year_by_year(df):
    """Per year: how well the score predicts the reaction, and the top-minus-bottom spreads."""
    sub = period_frame(df, PERIODS["all"])
    rows = []
    for year, g in sub.groupby("year"):
        q = g.groupby("quintile")[["immediate_ar", "drift_ar"]].mean()
        rows.append({"year": year, "calls": len(g),
                     "auc": roc_auc_score(g["immediate_ar"] > 0, g[SCORE_COL]),
                     "imm": q.loc[5, "immediate_ar"] - q.loc[1, "immediate_ar"],
                     "drift": q.loc[5, "drift_ar"] - q.loc[1, "drift_ar"]})
    return pd.DataFrame(rows).set_index("year")


def car_paths(df, er):
    """Average cumulative abnormal return path, day -1 -> +60, by score quintile."""
    rows = []
    days = np.arange(CAR_DAYS[0], CAR_DAYS[1] + 1)
    paths = np.full((len(df), len(days)), np.nan)
    for i, (t, d) in enumerate(zip(df["ticker"], df["call_date"])):
        p = er.path(t, d, *CAR_DAYS)
        if p is not None:
            paths[i] = p
    for pname, years in PERIODS.items():
        m = ((df["year"] >= years[0]) & (df["year"] <= years[1])).values
        for q in range(1, 6):
            sel = m & (df["quintile"] == q).values
            # clip extreme paths like the winsorized tests do
            sub = paths[sel]
            lo, hi = np.nanpercentile(sub, 1, axis=0), np.nanpercentile(sub, 99, axis=0)
            mean = np.nanmean(np.clip(sub, lo, hi), axis=0)
            rows += [{"period": pname, "quintile": q, "day": int(dd), "car": float(v)}
                     for dd, v in zip(days, mean)]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def pct(x):
    return f"{x * 100:+.2f}%"


def write_report(results, car, yearly, robust):
    L = [
        "# Phase 4 — Market-Reaction Test",
        "",
        "Does the market fully price what an earnings call's language reveals?",
        "The Phase 3 score (trained only on the *immediate* reaction, always out-of-sample) is",
        "compared with the immediate window (day −1 → +1) and the drift window (day +2 → +60).",
        "If the score also predicts the drift, the market under-reacted to the language.",
        "",
        "Method notes: calls are ranked by score within each calendar quarter; returns are",
        "abnormal (stock minus S&P 500) and winsorized at 1%/99%; spreads are tested across",
        f"quarters with Newey-West ({NW_LAGS} lag) t-statistics; regressions use quarter fixed",
        "effects with standard errors clustered by quarter. Regenerate with",
        "`python market_reaction.py`.",
        "",
    ]
    main = results["holdout"]
    q = main["quint"]
    imm_m, imm_t, imm_p, nq = q["immediate_ar"]
    dr_m, dr_t, dr_p, _ = q["drift_ar"]
    reg = main["reg"]["(3) + pre-call return, volatility"][0].loc["score_rank"]
    L += [
        "## Answer (holdout years 2018–2025)",
        "",
        f"- **Immediate:** top-20% calls beat bottom-20% calls by **{pct(imm_m)}** in the",
        f"  day −1 → +1 window (t = {imm_t:.1f}). The market does react to what the language says.",
        f"- **Drift:** they keep beating them by **{pct(dr_m)}** over day +2 → +60",
        f"  (t = {dr_t:.1f}, p = {dr_p:.3f}, {nq} quarters).",
        f"- **After controlling for the immediate reaction and pre-call momentum/volatility**, the",
        f"  score still predicts drift: highest-vs-lowest ranked call {pct(reg['coef'])}",
        f"  (t = {reg['t']:.1f}, p = {reg['p']:.3f}).",
        f"- Of the total top-minus-bottom move (immediate + drift), "
        f"**{dr_m / (imm_m + dr_m) * 100:.0f}%** came *after* the reaction window.",
        "",
    ]
    full = results["all"]
    f_m, f_t, f_p, f_n = full["quint"]["drift_ar"]
    f_reg = full["reg"]["(3) + pre-call return, volatility"][0].loc["score_rank"]
    L += [
        f"Over the **full out-of-sample period 2012–2025** ({f_n} quarters): drift top − bottom",
        f"{pct(f_m)} (t = {f_t:.1f}, p = {f_p:.3f}); with controls {pct(f_reg['coef'])}",
        f"(t = {f_reg['t']:.1f}, p = {f_reg['p']:.3f}).",
        "",
    ]
    strong = lambda p_, r_: p_ < 0.05 and r_["p"] < 0.05
    if strong(dr_p, reg):
        verdict = ("the language signal is only partly priced at first: the market "
                   "**under-reacts**, and the rest arrives over the following weeks.")
    elif strong(f_p, f_reg) or f_p < 0.05:
        verdict = ("the market prices most of the language signal immediately. A smaller part "
                   "drifts in over the following weeks: significant over 2012–2025, but not on "
                   "the 2018–2025 holdout alone. Evidence of **under-reaction is modest**, "
                   "not conclusive.")
    else:
        verdict = ("the market appears to price the language signal immediately; there is no "
                   "reliable evidence of later drift (consistent with an **efficient market**).")
    L += [f"**Verdict:** {verdict}", ""]
    no_covid = robust.iloc[-1]
    L += [f"**Pandemic check:** without 2020–2021 calls, the holdout drift is {pct(no_covid['drift'])}",
          f"(p = {no_covid['drift_p']:.3f}; with controls p = {no_covid['reg_p']:.3f}). The weaker "
          "holdout result comes from that period — see *Robustness* below.", ""]

    for pname, years in PERIODS.items():
        r = results[pname]
        q = r["quint"]
        L += [f"## {pname.capitalize()} years {years[0]}–{years[1]}", "",
              "### Average abnormal return by score quintile", "",
              "| Quintile | Calls | Immediate (−1 → +1) | Drift (+2 → +60) |",
              "|---|---|---|---|"]
        for idx, row in q["table"].iterrows():
            name = {1: "1 (lowest score)", 5: "5 (highest score)"}.get(idx, str(idx))
            L.append(f"| {name} | {int(row['n'])} | {pct(row['immediate_ar'])} | "
                     f"{pct(row['drift_ar'])} |")
        for col, lab in (("immediate_ar", "Immediate"), ("drift_ar", "Drift")):
            m_, t_, p_, n_ = q[col]
            L.append(f"| **Top − bottom ({lab})** | {n_} quarters | "
                     + (f"**{pct(m_)}** (t {t_:.1f}, p {p_:.3f})" if col == "immediate_ar" else "")
                     + " | "
                     + (f"**{pct(m_)}** (t {t_:.1f}, p {p_:.3f})" if col == "drift_ar" else "")
                     + " |")
        nl = r.get("nolang")
        if nl is not None:
            L += ["", f"Language-free model (market + reaction only), same drift test: "
                      f"top − bottom {pct(nl[0])} (t {nl[1]:.1f}, p {nl[2]:.3f})."]

        L += ["", "### Regressions: drift (+2 → +60) on score rank", "",
              "Coefficient on *score rank* = drift of the highest-ranked call minus the lowest.", "",
              "| Model | Score rank | Immediate reaction | Pre-call return | Pre-call volatility | Calls | Quarters |",
              "|---|---|---|---|---|---|---|"]
        for name, (tab, n, g) in r["reg"].items():
            if name == "immediate":
                continue
            cell = lambda c, f=pct: (f"{f(tab.loc[c, 'coef'])}{stars(tab.loc[c, 'p'])} "
                                     f"(t {tab.loc[c, 't']:.1f})" if c in tab.index else "")
            num = lambda v: f"{v:+.3f}"
            L.append(f"| {name} | {cell('score_rank')} | "
                     f"{cell('immediate_ar', num)} | {cell('pre_ar', num)} | "
                     f"{cell('pre_vol', num)} | {n} | {g} |")
        tab, n, g = r["reg"]["immediate"]
        L += ["", f"Immediate reaction on score rank: {pct(tab.loc['score_rank', 'coef'])}"
                  f"{stars(tab.loc['score_rank', 'p'])} (t {tab.loc['score_rank', 't']:.1f}).",
              "", "`*` p < 0.10, `**` p < 0.05, `***` p < 0.01.", ""]

    L += ["## Robustness: holdout without the pandemic period", "",
          "Top 20% minus bottom 20% of calls by score. *Controls* = drift regression with the",
          "immediate reaction and pre-call return/volatility.", "",
          "| Sample | Calls | Quarters | Immediate | Drift (+2 → +60) | Drift with controls |",
          "|---|---|---|---|---|---|"]
    for _, r_ in robust.iterrows():
        L.append(f"| {r_['sample']} | {r_['calls']} | {r_['quarters']} | "
                 f"{pct(r_['imm'])} (t {r_['imm_t']:.1f}) | {pct(r_['drift'])}{stars(r_['drift_p'])} "
                 f"(p {r_['drift_p']:.3f}) | {pct(r_['reg'])}{stars(r_['reg_p'])} (p {r_['reg_p']:.3f}) |")
    L += ["",
          "- Removing only the crash months does not change the result; removing all of 2020, or",
          "  2020–2021, makes the drift clearly significant. The weak stretch is the whole",
          "  stimulus / meme-stock / retail-trading period, not just the March 2020 crash.",
          "- **Caveat:** these exclusions were chosen *after* seeing that 2020–2021 were weak years.",
          "  Dropping weak years always improves a result, so this is a robustness check that",
          "  explains the holdout number; the main result above stays the headline.",
          "",
          "## Year by year (2012–2025)", "",
          "*Reaction AUC*: how well the score predicted the immediate reaction that year (0.5 =",
          "guessing). Spreads are top 20% minus bottom 20% of calls by score.", "",
          "| Year | Calls | Reaction AUC | Immediate spread | Drift spread |",
          "|---|---|---|---|---|"]
    for y, r_ in yearly.iterrows():
        mark = " ⚠️ pandemic" if y in (2020, 2021) else ""
        L.append(f"| {y}{mark} | {int(r_['calls'])} | {r_['auc']:.3f} | {pct(r_['imm'])} | "
                 f"{pct(r_['drift'])} |")
    p_yrs = yearly.loc[[2020, 2021]]
    o_yrs = yearly.drop([2020, 2021])
    L += ["",
          f"In 2020–2021 the score still predicted the reaction better than chance (AUC "
          f"{p_yrs['auc'].min():.2f}–{p_yrs['auc'].max():.2f}), but less sharply than in other years "
          f"(median {o_yrs['auc'].median():.2f}), and the later drift disappeared "
          f"({pct(p_yrs['drift'].mean())} on average vs. {pct(o_yrs['drift'].mean())}). Prices were",
          "driven mainly by market-wide forces (lockdowns, stimulus, meme-stock trading) rather",
          "than company-specific news, so the language signal weakened.", ""]

    hold = car[car["period"] == "holdout"].pivot(index="day", columns="quintile", values="car")
    L += ["", "## Cumulative abnormal return path (holdout, from close before the call)", "",
          "| Day | Q1 (lowest) | Q3 | Q5 (highest) | Q5 − Q1 |", "|---|---|---|---|---|"]
    for d in (0, 1, 5, 10, 20, 40, 60):
        if d in hold.index:
            r_ = hold.loc[d]
            L.append(f"| {d:+d} | {pct(r_[1])} | {pct(r_[3])} | {pct(r_[5])} | "
                     f"{pct(r_[5] - r_[1])} |")
    L += ["", "Full paths for every day and quintile: `data/car_paths.csv`.", "",
          "## Caveats",
          "- Drift windows of calls in the same quarter overlap, so calls are not independent;",
          "  the tests account for this by working across quarters and clustering by quarter.",
          "- The 200 tickers are today's S&P 500 members (survivorship bias), which may flatter",
          "  long-run returns but affects both high- and low-score groups.",
          "- No trading costs here; Phase 5 tests whether the drift is large enough to trade.",
          ]
    RESULTS_MD.write_text("\n".join(L), encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("[Phase4] loading predictions and prices ...")
    df, er = load()
    print(f"[Phase4] {len(df)} calls with score, immediate and drift returns")

    results = {}
    for pname, years in PERIODS.items():
        sub = period_frame(df, years)
        res = {"quint": quintile_test(sub), "reg": regressions(sub)}
        if NOLANG_COL in sub.columns:
            res["nolang"] = quintile_test_score(sub, NOLANG_COL)
        results[pname] = res
        m, t, p, n = res["quint"]["drift_ar"]
        im, it, _, _ = res["quint"]["immediate_ar"]
        reg = res["reg"]["(3) + pre-call return, volatility"][0].loc["score_rank"]
        print(f"  {pname:<10s} immediate top-bottom {im*100:+.2f}% (t {it:.1f}) | "
              f"drift top-bottom {m*100:+.2f}% (t {t:.1f}, p {p:.3f}) | "
              f"drift coef w/ controls {reg['coef']*100:+.2f}% (t {reg['t']:.1f})")

    yearly = year_by_year(df)
    robust = pandemic_robustness(df)
    for _, r_ in robust.iterrows():
        print(f"  {r_['sample'][:52]:<52s} drift {r_['drift']*100:+.2f}% (p {r_['drift_p']:.3f}) "
              f"| w/ controls p {r_['reg_p']:.3f}".encode("ascii", "replace").decode())

    print("[Phase4] computing day-by-day return paths ...")
    car = car_paths(df, er)
    car.to_csv(CAR_PATHS_CSV, index=False)

    out_cols = ["ticker", "call_date", "year", "quarter", SCORE_COL, "score_rank", "quintile",
                "predicted", "label", "immediate_ar", "drift_ar", "pre_ar", "pre_vol"]
    df[[c for c in out_cols if c in df.columns]].to_csv(MARKET_REACTION_CSV, index=False)
    write_report(results, car, yearly, robust)
    print(f"[Phase4] saved {MARKET_REACTION_CSV}, {CAR_PATHS_CSV}, {RESULTS_MD}")
    print("[Phase4] DONE. Next: python backtest.py")


if __name__ == "__main__":
    main()
