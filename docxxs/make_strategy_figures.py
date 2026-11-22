"""
Figures for docxxs/Strategy_Report.md: all Phase 5 trading approaches on one chart.

Reads:  data/backtest_finbert_2to1_l15s15_daily.csv (holds every strategy's daily net return:
        written by `python backtest_finbert_2to1.py --long-pct 0.15 --short-pct 0.15`)
Writes: docxxs/fig_strategy_equity.png       growth of 100, holdout 2018-2025
        docxxs/fig_strategy_risk_return.png  annual return vs volatility, holdout

Usage:
    python docxxs/make_strategy_figures.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).parent
DAILY = HERE.parent / "data/backtest_finbert_2to1_l15s15_daily.csv"
# approach F: `python backtest_finbert_2to1.py --long-pct 0.05 --short-pct 0.05`
DAILY_F = HERE.parent / "data/backtest_finbert_2to1_l5s5_daily.csv"
# approach G: `python backtest_finbert_2to1.py --long-pct 0.10 --short-pct 0.05`
DAILY_G = HERE.parent / "data/backtest_finbert_2to1_l10s5_daily.csv"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

# key in the daily file -> (label, colour, line style)
STRATS = {
    "neutral":   ("A  Market-neutral 1:1, 20/20 (main)", "#1baf7a", "-"),
    "ours_2020": ("B  Tilted 2:1, 20/20", "#6250d6", "-"),
    "ours_2010": ("C  Tilted 2:1, 20/10", "#e87ba4", "-"),
    "ours":      ("D  Tilted 2:1, 15/15", "#2a78d6", "-"),
    "longonly":  ("E  Long-only, top 15%", "#eda100", "-"),
    "f_5_5":     ("F  Tilted 2:1, 5/5", "#e34948", "-"),
    "g_10_5":    ("G  Tilted 2:1, 10/5", "#008300", "-"),
    "all_calls": ("Buy every call (no score)", "#9a9893", "-"),
    "sp500":     ("S&P 500 (no dividends)", INK2, "--"),
}


def style(ax):
    ax.set_facecolor(SURFACE)
    ax.set_axisbelow(True)
    ax.grid(True, color=GRID, lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)


def main():
    d = pd.read_csv(DAILY, index_col=0, parse_dates=True)
    f = pd.read_csv(DAILY_F, index_col=0, parse_dates=True)["ours"].rename("f_5_5")
    g = pd.read_csv(DAILY_G, index_col=0, parse_dates=True)["ours"].rename("g_10_5")
    d = d.join(f, how="left").join(g, how="left")
    d = d[(d.index.year >= 2018) & (d.index.year <= 2025)]

    fig, ax = plt.subplots(figsize=(11, 6))
    fig.patch.set_facecolor(SURFACE)
    ends = []
    for k, (lab, col, ls) in STRATS.items():
        w = 100 * (1 + d[k].dropna()).cumprod()
        ax.plot(w.index, w.values, color=col, lw=2 if k != "all_calls" else 1.6, ls=ls, label=lab)
        ends.append([w.iloc[-1], w.index[-1]])
    ends.sort(key=lambda e: e[0])
    ends[0].append(ends[0][0])
    for i in range(1, len(ends)):
        ends[i].append(max(ends[i][0], ends[i - 1][2] + 8))
    for val, when, ypos in ends:
        ax.annotate(f"{val:.0f}", (when, val), xytext=(when + pd.Timedelta(days=30), ypos),
                    textcoords="data", va="center", fontsize=9, color=INK)
    ax.axhline(100, color=INK2, lw=0.8)
    ax.set_ylabel("Value of 100 invested (after costs)", color=INK)
    ax.set_title("Figure 1. Growth of 100 under each approach, holdout 2018 – May 2025",
                 loc="left", fontsize=12, color=INK)
    ax.legend(frameon=False, fontsize=9, loc="upper left", labelcolor=INK)
    ax.margins(x=0.06)
    style(ax)
    fig.savefig(HERE / "fig_strategy_equity.png", dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 6))
    fig.patch.set_facecolor(SURFACE)
    for sr, (tx, ty) in ((0.5, (24.5, 11.6)), (1.0, (18.8, 19.6))):   # reference Sharpe lines
        xs = np.linspace(0, 27, 50)
        ax.plot(xs, sr * xs, color=GRID, lw=1.2, zorder=0)
        ax.text(tx, ty, f"Sharpe {sr:g}", fontsize=8, color=INK2, ha="right", va="top")
    # label offsets (points) so the three close 2:1 points don't collide
    offs = {"ours_2020": (-10, -9, "right"), "ours_2010": (-10, 9, "right"),
            "ours": (-4, 42, "center"), "f_5_5": (0, -14, "center"), "g_10_5": (30, 26, "left")}
    leader = {"ours", "g_10_5"}                 # labels placed away from the dot get a line
    for k, (lab, col, _) in STRATS.items():
        r = d[k].dropna()
        ret, vol = r.mean() * 252 * 100, r.std() * np.sqrt(252) * 100
        ax.scatter(vol, ret, s=90, color=col, edgecolor="white", linewidth=2, zorder=3)
        dx, dy, ha = offs.get(k, (8, 0, "left"))
        ax.annotate(lab.replace(" (main)", ""), (vol, ret), xytext=(dx, dy),
                    textcoords="offset points", va="center", ha=ha, fontsize=8.5, color=INK,
                    arrowprops=dict(arrowstyle="-", color=INK2, lw=0.7, shrinkA=2, shrinkB=6)
                    if k in leader else None)
    ax.set_xlim(0, 27)
    ax.set_ylim(0, 21)
    ax.set_xlabel("Volatility (% a year) — risk", color=INK)
    ax.set_ylabel("Annual return (%) after costs", color=INK)
    ax.set_title("Figure 2. Risk vs return, holdout 2018 – May 2025", loc="left", fontsize=12,
                 color=INK)
    style(ax)
    fig.savefig(HERE / "fig_strategy_risk_return.png", dpi=150, bbox_inches="tight",
                facecolor=SURFACE)
    plt.close(fig)
    print("saved fig_strategy_equity.png, fig_strategy_risk_return.png")


if __name__ == "__main__":
    main()
