"""
Phase 5 (v2) — Trading backtest on the FinBERT score
=====================================================
Phase 4 showed that calls the Phase 3 model likes drift up relative to calls it dislikes over
days +2 -> +60. This script trades that, the way a real investor could have:

  SIGNAL   At each call, compare its score with the scores of all calls in the PAST 365 days
           (nothing later is used). Top 20% -> LONG, bottom 20% -> SHORT, rest -> no trade.
           Trading starts once 200 past calls are available.
  TIMING   Enter at the close of day +2, exit at the close of day +60 (Phase 4's drift window).
           If a stock's prices stop early (delisting), exit at its last price.
  PORTFOLIO Every trading day: equal weight within the long leg and within the short leg,
           return = long leg - short leg (market-neutral, $1 long and $1 short per $1 capital).
           If one leg is empty that day, the S&P 500 stands in for it.
  COSTS    COST_BPS per buy or sell, charged on the weight traded, plus BORROW per year on the
           short leg. Rebalancing between equal weights is not charged.

Benchmarks: the S&P 500 (price index, no dividends), long-only top 20% minus the S&P 500,
the no-language Phase 3 model, and Velu's VADER score under the same rules.
Holding period, cut-offs and costs were fixed from Phase 4 before looking at any backtest.

Reads:  data/predictions_finbert.csv, data/prices.csv,
        data/_cache/velu_phase4_predictions.csv (optional, Velu's VADER scores)
Writes: data/backtest_finbert_trades.csv   one row per trade
        data/backtest_finbert_daily.csv    daily returns of every strategy
        docxxs/Phase5_finbert_results.md   results
        docxxs/phase5_equity_curve.png     equity curve and drawdown, holdout years

Usage:
    python backtest_finbert.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from event_returns import BAD_PRICE_TICKERS, MARKET_INDEX, event_day, load_prices
from market_reaction_finbert import newey_west

PRED_CSV   = Path("data/predictions_finbert.csv")
VELU_CSV   = Path("data/_cache/velu_phase4_predictions.csv")
TRADES_CSV = Path("data/backtest_finbert_trades.csv")
DAILY_CSV  = Path("data/backtest_finbert_daily.csv")
RESULTS_MD = Path("docxxs/Phase5_finbert_results.md")
CHART_PNG  = Path("docxxs/phase5_equity_curve.png")

ENTRY_DAY, EXIT_DAY = 2, 60     # trading days after the call (Phase 4's drift window)
LOOKBACK_DAYS = 365             # cut-offs come from the past year's calls only
MIN_HISTORY   = 200             # past calls needed before the first trade
TOP           = 0.20            # top / bottom fraction traded
COST_BPS      = 10              # per buy or sell
BORROW        = 0.005           # per year, short leg
COST_GRID     = [0, 5, 10, 20, 30]
NW_LAGS       = 10              # positions overlap for ~58 days, so daily returns are correlated
DAYS_PER_YEAR = 252
PERIODS = {"holdout": (2018, 2025), "valid": (2012, 2017), "all": (2012, 2025)}
PERIOD_NAME = {"holdout": "Holdout 2018–2025", "valid": "Validation 2012–2017",
               "all": "All years 2012–2025"}


# ---------------------------------------------------------------------------
# Signal: real-time cut-offs
# ---------------------------------------------------------------------------

def assign_sides(calls, col):
    """+1 / -1 / 0 per call, from cut-offs over the previous LOOKBACK_DAYS of scores."""
    d = calls.dropna(subset=[col]).sort_values("call_date").reset_index(drop=True)
    day = d["call_date"].dt.normalize().values
    s = d[col].values
    side = np.zeros(len(d), int)
    lo_cut = np.full(len(d), np.nan)
    hi_cut = np.full(len(d), np.nan)
    for i in range(len(d)):
        a = np.searchsorted(day, day[i] - np.timedelta64(LOOKBACK_DAYS, "D"))
        b = np.searchsorted(day, day[i])            # strictly earlier calendar days
        if b - a < MIN_HISTORY:
            continue
        lo_cut[i], hi_cut[i] = np.quantile(s[a:b], [TOP, 1 - TOP])
        side[i] = 1 if s[i] >= hi_cut[i] else (-1 if s[i] <= lo_cut[i] else 0)
    d["score"], d["side"], d["cut_lo"], d["cut_hi"] = s, side, lo_cut, hi_cut
    return d[["ticker", "call_date", "score", "cut_lo", "cut_hi", "side"]]


# ---------------------------------------------------------------------------
# Prices
# ---------------------------------------------------------------------------

class Market:
    """Daily close prices on the S&P 500 trading calendar."""

    def __init__(self, prices):
        prices = prices[prices[MARKET_INDEX].notna()]
        self.dates = prices.index
        self.mkt = prices[MARKET_INDEX].values
        self.mkt_ret = np.r_[0.0, self.mkt[1:] / self.mkt[:-1] - 1]
        self.px = {}
        for t in prices.columns:
            if t == MARKET_INDEX or t in BAD_PRICE_TICKERS:
                continue
            p = prices[t]
            valid = np.where(p.notna().values & (p.values > 0))[0]
            if len(valid):
                # forward-fill gaps between the first and last real price only
                self.px[t] = (p.ffill().values, valid[0], valid[-1])

    def trade(self, ticker, call_date):
        """(entry index, exit index, stock price path) or None."""
        if ticker not in self.px:
            return None
        p, first, last = self.px[ticker]
        i = self.dates.searchsorted(event_day(call_date))
        e, x = i + ENTRY_DAY, min(i + EXIT_DAY, last)
        if e < first or e >= x:
            return None
        return e, x, p


# ---------------------------------------------------------------------------
# Portfolio
# ---------------------------------------------------------------------------

def build_trades(sides, m, name):
    rows, paths = [], []
    for t, c, sc, sd in zip(sides["ticker"], sides["call_date"], sides["score"], sides["side"]):
        if sd == 0:
            continue
        hit = m.trade(t, c)
        if hit is None:
            continue
        e, x, p = hit
        stock = p[x] / p[e] - 1
        mkt = m.mkt[x] / m.mkt[e] - 1
        rows.append({"strategy": name, "ticker": t, "call_date": c, "score": sc,
                     "side": "LONG" if sd > 0 else "SHORT",
                     "entry_date": m.dates[e].date(), "exit_date": m.dates[x].date(),
                     "hold_days": x - e, "stock_return": stock, "market_return": mkt,
                     "excess_return": sd * (stock - mkt),
                     "net_excess_return": sd * (stock - mkt) - 2 * COST_BPS / 1e4})
        paths.append((sd, e, x, p[e:x + 1]))
    return pd.DataFrame(rows), paths


def daily_portfolio(paths, m, long_only=False):
    """Daily gross return, turnover (weight traded) and short exposure, by date."""
    n = len(m.dates)
    leg_sum = {1: np.zeros(n), -1: np.zeros(n)}
    leg_n = {1: np.zeros(n), -1: np.zeros(n)}
    enters = {1: np.zeros(n), -1: np.zeros(n)}
    exits = {1: np.zeros(n), -1: np.zeros(n)}
    for sd, e, x, p in paths:
        if long_only and sd < 0:
            continue
        r = p[1:] / p[:-1] - 1                  # returns over days e+1 .. x
        leg_sum[sd][e + 1:x + 1] += r
        leg_n[sd][e + 1:x + 1] += 1
        enters[sd][e] += 1
        exits[sd][x] += 1

    mr = m.mkt_ret
    leg_ret, turnover = {}, np.zeros(n)
    for sd in (1, -1):
        cnt = leg_n[sd]
        leg_ret[sd] = np.where(cnt > 0, leg_sum[sd] / np.maximum(cnt, 1), mr)
        held_next = np.r_[cnt[1:], 0]             # positions held over the next day
        turnover += enters[sd] / np.maximum(held_next, 1) + exits[sd] / np.maximum(cnt, 1)
    active = (leg_n[1] > 0) | (leg_n[-1] > 0)
    if long_only:
        gross = np.where(leg_n[1] > 0, leg_ret[1] - mr, 0.0)
        short_exp = np.zeros(n)
    else:
        gross = np.where(active, leg_ret[1] - leg_ret[-1], 0.0)
        short_exp = (leg_n[-1] > 0).astype(float)
    return pd.DataFrame({"gross": gross, "turnover": turnover, "short_exp": short_exp,
                         "n_long": leg_n[1], "n_short": leg_n[-1]}, index=m.dates)


def net_returns(day, cost_bps=COST_BPS):
    return day["gross"] - cost_bps / 1e4 * day["turnover"] - BORROW / DAYS_PER_YEAR * day["short_exp"]


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def in_years(s, years):
    return s[(s.index.year >= years[0]) & (s.index.year <= years[1])]


def metrics(r):
    r = r.dropna()
    wealth = (1 + r).cumprod()
    mean, t, p, _ = newey_west(r.values, NW_LAGS)
    vol = r.std() * np.sqrt(DAYS_PER_YEAR)
    return {"days": len(r), "ann_return": mean * DAYS_PER_YEAR, "vol": vol,
            "sharpe": mean * DAYS_PER_YEAR / vol if vol > 0 else np.nan,
            "cagr": wealth.iloc[-1] ** (DAYS_PER_YEAR / len(r)) - 1,
            "max_dd": (wealth / wealth.cummax() - 1).min(), "t": t, "p": p}


def break_even_bps(day):
    """Cost per buy/sell at which the average net return is zero."""
    gross = day["gross"].mean() - BORROW / DAYS_PER_YEAR * day["short_exp"].mean()
    turn = day["turnover"].mean()
    return gross / turn * 1e4 if turn > 0 else np.nan


def by_year(r):
    return (1 + r).groupby(r.index.year).prod() - 1


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def pct(x, d=1):
    return "—" if pd.isna(x) else f"{x * 100:+.{d}f}%"


def stars(p):
    return "" if pd.isna(p) else " \\*\\*\\*" if p < 0.01 else " \\*\\*" if p < 0.05 else " \\*" if p < 0.10 else ""


METRIC_HEAD = ("| Strategy | Annual return | Volatility | Sharpe | Max drawdown | t (p) |\n"
               "|---|---|---|---|---|---|")


def metric_row(label, mt):
    return (f"| {label} | {pct(mt['ann_return'])} | {mt['vol'] * 100:.1f}% | {mt['sharpe']:.2f} | "
            f"{pct(mt['max_dd'])} | {mt['t']:.2f} ({mt['p']:.3f}){stars(mt['p'])} |")


def trade_stats(trades, years):
    y = pd.to_datetime(trades["entry_date"]).dt.year
    t = trades[(y >= years[0]) & (y <= years[1])]
    return {"n": len(t), "long": (t["side"] == "LONG").sum(), "short": (t["side"] == "SHORT").sum(),
            "hit": (t["net_excess_return"] > 0).mean(), "avg": t["net_excess_return"].mean(),
            "avg_long": t.loc[t.side == "LONG", "excess_return"].mean(),
            "avg_short": t.loc[t.side == "SHORT", "excess_return"].mean()}


def write_results(series, days, trades, sides):
    L = ["# Phase 5 (v2) — Trading Backtest on the FinBERT score", "",
         "Generated by [`backtest_finbert.py`](../backtest_finbert.py). Method: "
         "[`Phase5_Trading_Backtest.md`](Phase5_Trading_Backtest.md).", "",
         f"**Rule:** at each call, LONG if its score is in the top {TOP:.0%} of the scores of all "
         f"calls in the previous {LOOKBACK_DAYS} days, SHORT if in the bottom {TOP:.0%}. Enter at the "
         f"close of day +{ENTRY_DAY}, exit at the close of day +{EXIT_DAY}. Equal weight within each "
         "leg, long leg minus short leg every day.  ",
         f"**Costs:** {COST_BPS} bps per buy or sell, {BORROW:.1%} a year to borrow the short leg.  ",
         "**Returns** are daily, annualised as mean × 252; volatility = daily std × √252; Sharpe = "
         f"annual return / volatility; t from Newey-West ({NW_LAGS} lags) on daily returns. "
         "The S&P 500 is the price index (no dividends).", ""]

    labels = [("ours", "**Long-short, FinBERT score (net of costs)**"),
              ("ours_gross", "Long-short, FinBERT score, before costs"),
              ("ours_long", "Long-only top 20%, minus S&P 500 (net)"),
              ("nolang", "Long-short, no-language score (net)"),
              ("velu", "Long-short, Velu's VADER score (net)"),
              ("sp500", "S&P 500 buy-and-hold")]
    for per in ("holdout", "valid", "all"):
        yrs = PERIODS[per]
        L += [f"## {PERIOD_NAME[per]}", "", METRIC_HEAD]
        for key, lab in labels:
            if key in series:
                L.append(metric_row(lab, metrics(in_years(series[key], yrs))))
        ts = trade_stats(trades[trades.strategy == "ours"], yrs)
        L += ["", f"Trades (FinBERT long-short): {ts['n']:,} ({ts['long']:,} long, {ts['short']:,} "
              f"short). Hit rate after costs {ts['hit']:.1%}; average trade {pct(ts['avg'], 2)} after "
              f"costs (long side {pct(ts['avg_long'], 2)}, short side {pct(ts['avg_short'], 2)} "
              "before costs, both relative to the S&P 500).", ""]
        if per == "valid":
            first = sides.loc[sides["side"] != 0, "call_date"].min()
            L += [f"First trade signal: {first.date()} (needs {MIN_HISTORY} past calls), "
                  "so this period effectively starts in 2013.", ""]

    hd = days["ours"].loc[in_years(series["ours"], PERIODS["holdout"]).index]
    L += ["## Costs (holdout 2018–2025)", "",
          "| Cost per buy or sell | Annual return | Sharpe |", "|---|---|---|"]
    for c in COST_GRID:
        mt = metrics(net_returns(hd, c))
        L.append(f"| {c} bps | {pct(mt['ann_return'])} | {mt['sharpe']:.2f} |")
    L += ["", f"Break-even cost: **{break_even_bps(hd):.0f} bps** per buy or sell. Average "
          f"positions open: {hd['n_long'].mean():.0f} long, {hd['n_short'].mean():.0f} short. "
          f"Weight traded: {hd['turnover'].mean() * DAYS_PER_YEAR:.1f}× capital per year "
          "(both legs, buys + sells).", ""]

    yr = pd.DataFrame({"FinBERT long-short (net)": by_year(series["ours"]),
                       "No-language (net)": by_year(series["nolang"]),
                       "S&P 500": by_year(series["sp500"])})
    if "velu" in series:
        yr.insert(2, "Velu VADER (net)", by_year(series["velu"]))
    yr = yr.loc[(yr.index >= 2013) & (yr.index <= 2025)]
    L += ["## Year by year", "", "| Year | " + " | ".join(yr.columns) + " |",
          "|---" * (len(yr.columns) + 1) + "|"]
    end = series["ours"].index[-1]
    for y, row in yr.iterrows():
        lab = f"{y} (to {end:%d %b})" if y == end.year and end.month < 12 else str(y)
        L.append(f"| {lab} | " + " | ".join(pct(v) for v in row.values) + " |")
    L += ["", f"All series end on {end.date()}, the last new trade (the data's last calls are "
          "from May 2025); the S&P 500 is cut to the same dates."]
    L += ["", "## Notes", "",
          "- No setting was tuned on the backtest: the 20% cut-off and the day +2 → +60 hold "
          "come from Phase 4, costs are assumed.",
          "- Scores come from Phase 3's walk-forward models, so every score was made without "
          "seeing that call's future; the cut-offs use only earlier calls.",
          "- Universe: today's 200 tickers, so firms that disappeared before 2025 are missing "
          "(survivorship bias). PARA is excluded (bad prices, see `event_returns.py`).",
          "- Trades are filled at the closing price with no market impact; daily rebalancing "
          "between equal weights is not charged.",
          "", "Chart: [`phase5_equity_curve.png`](phase5_equity_curve.png). Trades: "
          "`data/backtest_finbert_trades.csv`. Daily returns: `data/backtest_finbert_daily.csv`."]
    RESULTS_MD.write_text("\n".join(L) + "\n", encoding="utf-8")


def plot(series):
    ink, ink2, grid, surface = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
    lines = [("ours", "FinBERT long-short (net)", "#2a78d6", "-"),
             ("velu", "Velu VADER long-short (net)", "#1baf7a", "-"),
             ("nolang", "No-language long-short (net)", "#eb6834", "-"),
             ("sp500", "S&P 500 buy-and-hold", ink2, "--")]
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(11, 7.5), sharex=True,
                                  gridspec_kw={"height_ratios": [3, 1.2], "hspace": 0.08})
    fig.patch.set_facecolor(surface)
    ends = []
    for key, lab, col, ls in lines:
        if key not in series:
            continue
        r = in_years(series[key], PERIODS["holdout"])
        w = 100 * (1 + r).cumprod()
        ax.plot(w.index, w.values, color=col, lw=2, ls=ls, label=lab)
        ends.append([w.iloc[-1], w.index[-1]])
    ends.sort(key=lambda e: e[0])
    ends[0].append(ends[0][0])
    for i in range(1, len(ends)):                 # keep end labels at least 7 units apart
        ends[i].append(max(ends[i][0], ends[i - 1][2] + 7))
    for val, when, ypos in ends:
        ax.annotate(f"{val:.0f}", (when, val), xytext=(when + pd.Timedelta(days=30), ypos),
                    textcoords="data", va="center", fontsize=9, color=ink)
    ax.axhline(100, color=ink2, lw=0.8)
    ax.set_ylabel("Value of 100 invested", color=ink)
    ax.set_title("Phase 5 backtest, holdout 2018–2025: what 100 grows to", loc="left",
                 fontsize=13, color=ink, pad=10)
    ax.legend(frameon=False, loc="upper left", fontsize=9, labelcolor=ink)

    r = in_years(series["ours"], PERIODS["holdout"])
    w = (1 + r).cumprod()
    dd = 100 * (w / w.cummax() - 1)
    ax2.fill_between(dd.index, dd.values, 0, color="#2a78d6", alpha=0.35, lw=0)
    ax2.plot(dd.index, dd.values, color="#2a78d6", lw=1.2)
    ax2.set_ylabel("Drawdown (%)", color=ink)
    ax2.set_title("FinBERT long-short: fall from its previous peak", loc="left", fontsize=10,
                  color=ink2)
    for a in (ax, ax2):
        a.set_facecolor(surface)
        a.grid(True, color=grid, lw=0.6)
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            a.spines[s].set_color(grid)
        a.tick_params(colors=ink2, labelsize=9)
    ax.margins(x=0.06)
    fig.savefig(CHART_PNG, dpi=130, bbox_inches="tight", facecolor=surface)
    plt.close(fig)


# ---------------------------------------------------------------------------

def run(calls, col, name, m, long_only_too=False):
    sides = assign_sides(calls, col)
    trades, paths = build_trades(sides, m, name)
    out = {name: daily_portfolio(paths, m)}
    if long_only_too:
        out[name + "_long"] = daily_portfolio(paths, m, long_only=True)
    return sides, trades, out


def main():
    print("[Phase5] loading ...")
    m = Market(load_prices())
    preds = pd.read_csv(PRED_CSV, parse_dates=["call_date"])
    preds = preds[~preds["ticker"].isin(BAD_PRICE_TICKERS)]

    days, all_trades = {}, []
    sides, tr, dd = run(preds, "score_final", "ours", m, long_only_too=True)
    days.update(dd); all_trades.append(tr)
    _, tr, dd = run(preds, "score_nolang", "nolang", m)
    days.update(dd); all_trades.append(tr)
    if VELU_CSV.exists():
        velu = pd.read_csv(VELU_CSV, parse_dates=["call_date"])
        velu = velu[~velu["ticker"].isin(BAD_PRICE_TICKERS)]
        _, tr, dd = run(velu, "score_reaction_lr", "velu", m)
        days.update(dd); all_trades.append(tr)
    else:
        print(f"[Phase5] {VELU_CSV} not found, skipping Velu's comparison")

    trades = pd.concat(all_trades, ignore_index=True)
    last_entry = trades.groupby("strategy")["entry_date"].max()

    def live_period(r, day, name):
        # from the first open position to the last new trade: no zero-return days before,
        # and no wind-down after the data's last call, when only a few positions remain
        held = (day["n_long"] + day["n_short"]) > 0
        end = pd.Timestamp(last_entry[name.split("_")[0]])
        return r[(r.index >= held.idxmax()) & (r.index <= end)]

    series = {k: live_period(net_returns(v), v, k) for k, v in days.items()}
    series["ours_gross"] = live_period(days["ours"]["gross"], days["ours"], "ours")
    sp = pd.Series(m.mkt_ret, index=m.dates)
    series["sp500"] = sp[(sp.index >= series["ours"].index[0]) & (sp.index <= series["ours"].index[-1])]

    trades.to_csv(TRADES_CSV, index=False)
    daily = pd.DataFrame({f"{k}_net" if k not in ("ours_gross", "sp500") else k: v
                          for k, v in series.items()})
    daily["ours_n_long"] = days["ours"]["n_long"]
    daily["ours_n_short"] = days["ours"]["n_short"]
    daily = daily[daily.index.year >= 2012]
    daily.to_csv(DAILY_CSV, index_label="date")
    print(f"[Phase5] {len(trades):,} trades -> {TRADES_CSV}, daily -> {DAILY_CSV}")

    for per in ("holdout", "valid"):
        mt = metrics(in_years(series["ours"], PERIODS[per]))
        print(f"[Phase5] {PERIOD_NAME[per]}: annual {pct(mt['ann_return'])}, "
              f"Sharpe {mt['sharpe']:.2f}, max DD {pct(mt['max_dd'])}, t {mt['t']:.2f}")
    write_results(series, days, trades, sides)
    plot(series)
    print(f"[Phase5] results -> {RESULTS_MD}, chart -> {CHART_PNG}")


if __name__ == "__main__":
    main()
