"""
Phase 5 — Trading Backtest
============================
Long-GOOD / short-BAD strategy using Phase 3 confidence scores.
Applies a confidence filter, transaction costs, and computes:
  - Equity curve
  - Sharpe ratio (annualised)
  - Max drawdown
  - Win rate

Reads: data/predictions.csv, data/prices.csv
Writes: data/backtest_results.csv, docxxs/Phase5_results.md

Usage:
    python backtest.py
    python backtest.py --confidence 0.65   # stricter filter
    python backtest.py --cost 0.001        # 10bps transaction cost
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

PREDICTIONS_CSV    = Path("data/predictions.csv")
PRICES_CSV         = Path("data/prices.csv")
BACKTEST_CSV       = Path("data/backtest_results.csv")
EQUITY_CSV         = Path("data/equity_curve.csv")
RESULTS_MD         = Path("docxxs/Phase5_results.md")

MARKET_INDEX = "^GSPC"
HOLD_DAYS    = 20    # hold position for +2 to +21 trading days after call


def price_at_offset(series, call_date, offset):
    idx = series.index.searchsorted(call_date)
    target = idx + offset
    if target >= len(series):
        return None
    return series.iloc[target]


def compute_trade_return(ticker, call_date, prices, entry_day=2, hold=HOLD_DAYS):
    """Return (entry_price, exit_price, stock_return, market_return, trade_return)."""
    if ticker not in prices.columns:
        return None
    stock  = prices[ticker].dropna()
    market = prices[MARKET_INDEX].dropna()

    entry_s = price_at_offset(stock,  call_date, entry_day)
    exit_s  = price_at_offset(stock,  call_date, entry_day + hold)
    entry_m = price_at_offset(market, call_date, entry_day)
    exit_m  = price_at_offset(market, call_date, entry_day + hold)

    if None in (entry_s, exit_s, entry_m, exit_m) or entry_s == 0 or entry_m == 0:
        return None

    stock_ret  = exit_s / entry_s - 1
    market_ret = exit_m / entry_m - 1
    return stock_ret, market_ret


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--confidence", type=float, default=0.60,
                        help="min confidence to take a trade (default 0.60)")
    parser.add_argument("--cost", type=float, default=0.001,
                        help="one-way transaction cost (default 0.001 = 10bps)")
    args, _ = parser.parse_known_args()

    print("[Phase5] Loading data ...")
    preds  = pd.read_csv(PREDICTIONS_CSV, parse_dates=["call_date"])
    prices = pd.read_csv(PRICES_CSV, index_col=0, parse_dates=True)

    conf_col = "xgb_confidence_good" if "xgb_confidence_good" in preds.columns \
               else "lr_confidence_good"
    pred_col = "xgb_predicted" if "xgb_predicted" in preds.columns else "lr_predicted"

    trades = []
    for _, row in preds.iterrows():
        conf = row[conf_col]
        pred = row[pred_col]

        # Only trade high-confidence predictions
        if pred == "GOOD" and conf < args.confidence:
            continue
        if pred == "BAD"  and conf > (1 - args.confidence):
            continue

        ret = compute_trade_return(row["ticker"], row["call_date"], prices)
        if ret is None:
            continue

        stock_ret, market_ret = ret
        direction = 1 if pred == "GOOD" else -1   # +1 = long, -1 = short
        gross_ret  = direction * stock_ret
        net_ret    = gross_ret - 2 * args.cost     # entry + exit cost

        trades.append({
            "call_date":       row["call_date"],
            "ticker":          row["ticker"],
            "actual_label":    row["label"],
            "predicted":       pred,
            "confidence":      conf,
            "direction":       "LONG" if direction == 1 else "SHORT",
            "stock_return":    stock_ret,
            "market_return":   market_ret,
            "gross_return":    gross_ret,
            "net_return":      net_ret,
        })

    df = pd.DataFrame(trades).sort_values("call_date").reset_index(drop=True)
    df.to_csv(BACKTEST_CSV, index=False)
    print(f"[Phase5] {len(df)} trades saved -> {BACKTEST_CSV}")

    if len(df) == 0:
        print("No trades passed the confidence filter.")
        return

    # ---- Performance metrics ----
    returns = df["net_return"].values
    cum_ret = (1 + returns).cumprod() - 1
    total_return = cum_ret[-1]
    win_rate = (returns > 0).mean()
    avg_return = returns.mean()

    # Annualised Sharpe (assume ~252 trading days / year, 1 trade per ~20 days)
    trades_per_year = 252 / HOLD_DAYS
    sharpe = (avg_return / (returns.std() + 1e-9)) * np.sqrt(trades_per_year)

    # Max drawdown
    cum_wealth = (1 + returns).cumprod()
    roll_max   = np.maximum.accumulate(cum_wealth)
    drawdown   = (cum_wealth - roll_max) / roll_max
    max_dd     = drawdown.min()

    # Equity curve CSV
    eq = pd.DataFrame({
        "trade_num":    np.arange(1, len(df) + 1),
        "call_date":    df["call_date"].values,
        "net_return":   returns,
        "equity":       cum_wealth,
    })
    eq.to_csv(EQUITY_CSV, index=False)

    print(f"\n{'='*50}")
    print(f"  Backtest Results (confidence ≥ {args.confidence})")
    print(f"{'='*50}")
    print(f"  Trades         : {len(df)}")
    print(f"  Win Rate       : {win_rate*100:.1f}%")
    print(f"  Total Return   : {total_return*100:.1f}%")
    print(f"  Avg per Trade  : {avg_return*100:.2f}%")
    print(f"  Sharpe Ratio   : {sharpe:.2f}")
    print(f"  Max Drawdown   : {max_dd*100:.1f}%")
    longs  = (df["direction"] == "LONG").sum()
    shorts = (df["direction"] == "SHORT").sum()
    print(f"  Longs/Shorts   : {longs} / {shorts}")

    # Save results MD
    lines = [
        "# Phase 5 — Trading Backtest",
        "",
        f"**Strategy:** Long high-confidence GOOD / Short high-confidence BAD  ",
        f"**Confidence threshold:** ≥ {args.confidence}  ",
        f"**Transaction cost:** {args.cost*100:.1f} bps (one-way)  ",
        f"**Hold period:** Day +2 → +{2+HOLD_DAYS} (trading days after earnings call)",
        "",
        "## Results",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Total Trades | {len(df)} |",
        f"| Long / Short | {longs} / {shorts} |",
        f"| Win Rate | {win_rate*100:.1f}% |",
        f"| Total Return | {total_return*100:.1f}% |",
        f"| Avg Return per Trade | {avg_return*100:.2f}% |",
        f"| Sharpe Ratio (annualised) | {sharpe:.2f} |",
        f"| Max Drawdown | {max_dd*100:.1f}% |",
        "",
        "---",
        "- Equity curve: `data/equity_curve.csv`",
        "- Per-trade results: `data/backtest_results.csv`",
        "> Note: Past backtest performance does not imply future results.",
    ]
    RESULTS_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"[Phase5] Saved results -> {RESULTS_MD}")
    print("\n[Phase5] DONE. Next: streamlit run dashboard.py")


if __name__ == "__main__":
    main()
