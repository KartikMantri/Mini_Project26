"""
Shared event-window return helpers (used by Phase 3; Phases 4-5 should switch to it).

Day 0 is the first trading day on or after the call's calendar date. The time of day is
ignored on purpose: most calls happen before or during market hours, so the price reacts
on the call date itself. (Comparing the full timestamp against daily price dates pushed
day 0 one day too late for those calls.)

Windows are (start, end) trading-day offsets, measured close-to-close. Example: the
immediate window (-1, 1) runs from the close before the call to the close of the next
trading day, which covers both before-open and after-close calls.
"""

import numpy as np
import pandas as pd

MARKET_INDEX = "^GSPC"

IMMEDIATE_WIN = (-1, 1)    # reaction to the call itself
DRIFT_WIN     = (2, 60)    # what happens afterwards (Phase 1 label window)
PRE_WIN       = (-61, -1)  # ~3 months before the call (momentum / volatility)

# Tickers whose price series in data/prices.csv is not the company's real price. Their
# returns come back as NaN everywhere. PARA: the series starts 2021-02-12 at ~83,000,
# falls to ~6, has 22% zero-change days and a 0.05 correlation with the S&P 500 (no sign
# of the March 2021 Archegos crash), so it is some other instrument under that symbol.
BAD_PRICE_TICKERS = {"PARA"}


def event_day(call_date):
    """Calendar date of the call, with the time of day dropped."""
    return pd.Timestamp(call_date).normalize()


class EventReturns:
    """Fast abnormal-return lookups (stock minus S&P 500) around event dates."""

    def __init__(self, prices):
        market = prices[MARKET_INDEX]
        self._data = {}
        for ticker in prices.columns:
            if ticker == MARKET_INDEX or ticker in BAD_PRICE_TICKERS:
                continue
            stock = prices[ticker].dropna()
            mkt = market.reindex(stock.index)
            ok = mkt.notna() & (stock > 0) & (mkt > 0)
            stock, mkt = stock[ok], mkt[ok]
            self._data[ticker] = (stock.index.values, stock.values, mkt.values)

    def _locate(self, ticker, call_date, start, end):
        if ticker not in self._data:
            return None
        dates, stock, mkt = self._data[ticker]
        i = dates.searchsorted(event_day(call_date).to_datetime64())
        a, b = i + start, i + end
        if a < 0 or b >= len(stock) or i >= len(stock):
            return None
        return stock, mkt, a, b

    def abnormal(self, ticker, call_date, window):
        """Stock return minus market return over the window."""
        hit = self._locate(ticker, call_date, *window)
        if hit is None:
            return np.nan
        stock, mkt, a, b = hit
        return (stock[b] / stock[a] - 1) - (mkt[b] / mkt[a] - 1)

    def raw(self, ticker, call_date, window):
        """(stock return, market return) over the window, or None."""
        hit = self._locate(ticker, call_date, *window)
        if hit is None:
            return None
        stock, mkt, a, b = hit
        return stock[b] / stock[a] - 1, mkt[b] / mkt[a] - 1

    def volatility(self, ticker, call_date, window):
        """Std of daily abnormal returns over the window."""
        hit = self._locate(ticker, call_date, *window)
        if hit is None:
            return np.nan
        stock, mkt, a, b = hit
        s = stock[a:b + 1]
        m = mkt[a:b + 1]
        return float(np.std(s[1:] / s[:-1] - m[1:] / m[:-1]))


def load_prices(path="data/prices.csv"):
    return pd.read_csv(path, index_col=0, parse_dates=True)
