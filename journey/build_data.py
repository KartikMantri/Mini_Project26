"""Build journey/data.js: weekly growth-of-100 curves (holdout 2018-2025) for the journey page.

Run from the repo root:  python journey/build_data.py
Reads only committed backtest files in data/.
"""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = Path(__file__).resolve().parent / "data.js"

# key: (file, column) - same mapping as dashboard.py STRATS
SERIES = {
    "A": ("backtest_finbert_daily.csv", "ours_net"),
    "B": ("backtest_finbert_2to1_l15s15_daily.csv", "ours_2020"),
    "C": ("backtest_finbert_2to1_l15s15_daily.csv", "ours_2010"),
    "D": ("backtest_finbert_2to1_l15s15_daily.csv", "ours"),
    "E": ("backtest_finbert_2to1_l15s15_daily.csv", "longonly"),
    "F": ("backtest_finbert_2to1_l5s5_daily.csv", "ours"),
    "G": ("backtest_finbert_2to1_l10s5_daily.csv", "ours"),
    "ALL": ("backtest_finbert_2to1_l15s15_daily.csv", "all_calls"),
    "SPX": ("backtest_finbert_daily.csv", "sp500"),
}


def main():
    cache, cols = {}, {}
    for k, (f, c) in SERIES.items():
        if f not in cache:
            cache[f] = pd.read_csv(DATA / f, index_col=0, parse_dates=True)
        cols[k] = cache[f][c]
    df = pd.DataFrame(cols)
    df = df[df.index >= "2018-01-01"].fillna(0.0)
    wealth = 100 * (1 + df).cumprod()
    weekly = wealth.resample("W-FRI").last().dropna(how="all")
    out = {
        "dates": [d.strftime("%Y-%m-%d") for d in weekly.index],
        "series": {k: [round(float(v), 2) for v in weekly[k]] for k in weekly},
        "final": {k: round(float(wealth[k].iloc[-1]), 1) for k in wealth},
    }
    OUT.write_text("window.JOURNEY_DATA = " + json.dumps(out) + ";\n", encoding="utf-8")
    print("wrote", OUT, len(out["dates"]), "weeks; final:", out["final"])


if __name__ == "__main__":
    main()
