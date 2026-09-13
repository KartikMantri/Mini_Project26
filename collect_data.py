"""
Decoding the Game -- Phase 1 Data Collection Pipeline
=======================================================
Pulls earnings-call transcripts (Hugging Face) + daily prices (yfinance)
for a fixed list of S&P 500 tickers, computes a Good/Bad label from
post-call abnormal returns, and builds master.csv.

Usage:
    pip install -r requirements.txt
    python collect_data.py
    python collect_data.py --start 2021-01-01 --end 2024-06-30 --threshold 0.02
    python collect_data.py --force        # re-download even if cached files exist

Outputs (in ./data/):
    raw_transcripts.csv   -- filtered transcripts, one row per call
    prices.csv            -- daily adjusted close, tickers + ^GSPC
    labeled_all.csv        -- transcripts + abnormal_return + label (GOOD/BAD/NEUTRAL)
    master.csv             -- same, but NEUTRAL rows dropped (feeds Phase 2/3)
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

TICKERS_FILE = Path("tickers.csv")  # Phase 0 deliverable -- ticker,sector columns
MARKET_INDEX = "^GSPC"


def load_tickers(path=TICKERS_FILE):
    """Reads the Phase 0 ticker universe. Falls back to a hardcoded default
    list (with a warning) if tickers.csv isn't found, so the script never
    hard-fails just because the file is missing."""
    if path.exists():
        df = pd.read_csv(path)
        return df["ticker"].tolist()

    print(f"[config] WARNING: {path} not found -- using built-in default ticker list. "
          f"Run with tickers.csv present for reproducibility.")
    return [
        "AAPL", "MSFT", "GOOGL", "ORCL", "ADBE", "CSCO", "IBM", "TXN", "QCOM", "INTC",
        "JPM", "BAC", "WFC", "GS", "MS", "AXP", "BLK", "SPGI",
        "JNJ", "UNH", "PFE", "MRK", "ABT", "TMO", "LLY", "ABBV",
        "HD", "MCD", "NKE", "SBUX", "TJX",
        "PG", "KO", "PEP", "WMT", "COST", "CL",
        "HON", "UPS", "CAT", "BA", "LMT",
        "XOM", "CVX",
        "DIS",
    ]

HF_DATASET_NAME = "kurry/sp500_earnings_transcripts"

DATA_DIR = Path("data")

# Abnormal-return label window, in TRADING DAYS after the call date.
LABEL_WINDOW = (2, 60)          # day +2 -> day +60, matches Phase 4's drift window
PRICE_BUFFER_DAYS = 120         # extra calendar days pulled past --end, so the
                                 # +60 trading-day window has data even for the
                                 # very last calls in range


# ---------------------------------------------------------------------------
# Step 1: Fetch transcripts
# ---------------------------------------------------------------------------

def fetch_transcripts(tickers, start, end, out_path, force=False):
    if out_path.exists() and not force:
        print(f"[transcripts] cached file found -> {out_path}")
        return pd.read_csv(out_path, parse_dates=["call_date"])

    try:
        from datasets import load_dataset
    except ImportError:
        sys.exit("Missing dependency: run `pip install datasets` first.")

    print(f"[transcripts] loading {HF_DATASET_NAME} from Hugging Face ...")
    ds = load_dataset(HF_DATASET_NAME, split="train")
    df = ds.to_pandas()

    # Expected columns: symbol, company_name, year, quarter, date, content, structured_content
    df = df.rename(columns={"symbol": "ticker", "date": "call_date", "content": "transcript_text"})
    df["call_date"] = pd.to_datetime(df["call_date"], errors="coerce")

    mask = (
        df["ticker"].isin(tickers)
        & df["call_date"].between(pd.Timestamp(start), pd.Timestamp(end))
    )
    filtered = df.loc[mask, ["ticker", "call_date", "transcript_text"]].dropna()
    filtered = filtered.sort_values(["ticker", "call_date"]).reset_index(drop=True)

    missing = set(tickers) - set(filtered["ticker"].unique())
    if missing:
        print(f"[transcripts] WARNING: no transcripts found for {sorted(missing)}")

    DATA_DIR.mkdir(exist_ok=True)
    filtered.to_csv(out_path, index=False)
    print(f"[transcripts] saved {len(filtered)} rows -> {out_path}")
    return filtered


# ---------------------------------------------------------------------------
# Step 2: Fetch prices
# ---------------------------------------------------------------------------

def fetch_prices(tickers, start, end, out_path, force=False):
    if out_path.exists() and not force:
        print(f"[prices] cached file found -> {out_path}")
        return pd.read_csv(out_path, index_col=0, parse_dates=True)

    try:
        import yfinance as yf
    except ImportError:
        sys.exit("Missing dependency: run `pip install yfinance` first.")

    buffered_end = pd.Timestamp(end) + pd.Timedelta(days=PRICE_BUFFER_DAYS)
    all_symbols = tickers + [MARKET_INDEX]

    print(f"[prices] downloading {len(all_symbols)} symbols, "
          f"{start} -> {buffered_end.date()} ...")
    raw = yf.download(
        all_symbols,
        start=start,
        end=str(buffered_end.date()),
        auto_adjust=True,
        progress=False,
        group_by="ticker",
        threads=True,
    )

    # Flatten yfinance's multi-index columns down to one 'Close' column per symbol
    closes = {}
    for sym in all_symbols:
        try:
            closes[sym] = raw[sym]["Close"]
        except (KeyError, TypeError):
            # single-ticker download shape fallback
            closes[sym] = raw["Close"] if "Close" in raw else None

    prices = pd.DataFrame(closes).dropna(how="all")
    prices.to_csv(out_path)
    print(f"[prices] saved {prices.shape[0]} trading days x {prices.shape[1]} symbols -> {out_path}")
    return prices


# ---------------------------------------------------------------------------
# Step 3: Compute abnormal-return labels
# ---------------------------------------------------------------------------

def _price_at_offset(price_series, call_date, offset_trading_days):
    """Trading-day offset lookup: nearest trading day on/after call_date, then +N rows."""
    idx = price_series.index.searchsorted(call_date)
    target_idx = idx + offset_trading_days
    if target_idx >= len(price_series):
        return None
    return price_series.iloc[target_idx]


def compute_abnormal_return(ticker, call_date, prices, window=LABEL_WINDOW):
    if ticker not in prices.columns or MARKET_INDEX not in prices.columns:
        return None

    stock = prices[ticker].dropna()
    market = prices[MARKET_INDEX].dropna()

    s0 = _price_at_offset(stock, call_date, window[0])
    s1 = _price_at_offset(stock, call_date, window[1])
    m0 = _price_at_offset(market, call_date, window[0])
    m1 = _price_at_offset(market, call_date, window[1])

    if None in (s0, s1, m0, m1) or s0 in (0, None) or m0 in (0, None):
        return None

    stock_ret = (s1 / s0) - 1
    market_ret = (m1 / m0) - 1
    return stock_ret - market_ret


def label_from_return(abnormal_return, threshold):
    if abnormal_return is None or pd.isna(abnormal_return):
        return None
    if abnormal_return > threshold:
        return "GOOD"
    if abnormal_return < -threshold:
        return "BAD"
    return "NEUTRAL"


def build_labeled_dataset(transcripts, prices, threshold):
    print("[labels] computing abnormal returns ...")
    abnormal_returns, labels = [], []

    for _, row in transcripts.iterrows():
        ar = compute_abnormal_return(row["ticker"], row["call_date"], prices)
        abnormal_returns.append(ar)
        labels.append(label_from_return(ar, threshold))

    out = transcripts.copy()
    out["abnormal_return"] = abnormal_returns
    out["label"] = labels

    dropped = out["label"].isna().sum()
    if dropped:
        print(f"[labels] dropped {dropped} rows (insufficient future price data)")
    out = out.dropna(subset=["label"]).reset_index(drop=True)
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Phase 1 data collection pipeline")
    parser.add_argument("--start", default="2021-01-01")
    parser.add_argument("--end", default="2024-06-30")
    parser.add_argument("--threshold", type=float, default=0.02,
                         help="abnormal-return cutoff for GOOD/BAD (default 0.02 = 2%%)")
    parser.add_argument("--force", action="store_true",
                         help="re-download even if cached CSVs exist")
    # parse_known_args (not parse_args) so this doesn't choke on Colab/Jupyter's
    # own injected arguments (e.g. "-f /root/.../kernel-....json")
    args, _unknown = parser.parse_known_args()

    DATA_DIR.mkdir(exist_ok=True)
    TICKERS = load_tickers()

    transcripts = fetch_transcripts(
        TICKERS, args.start, args.end, DATA_DIR / "raw_transcripts.csv", force=args.force
    )
    prices = fetch_prices(
        TICKERS, args.start, args.end, DATA_DIR / "prices.csv", force=args.force
    )
    labeled = build_labeled_dataset(transcripts, prices, args.threshold)

    labeled_all_path = DATA_DIR / "labeled_all.csv"
    labeled.to_csv(labeled_all_path, index=False)

    master = labeled[labeled["label"].isin(["GOOD", "BAD"])].reset_index(drop=True)
    master_path = DATA_DIR / "master.csv"
    master.to_csv(master_path, index=False)

    # ---- Summary report ----
    counts = labeled["label"].value_counts().reindex(["GOOD", "BAD", "NEUTRAL"], fill_value=0)
    print("\n" + "=" * 60)
    print(f"Fetched {len(transcripts)} transcripts across {transcripts['ticker'].nunique()} "
          f"tickers ({args.start} to {args.end})")
    print(f"Price data pulled for {prices.shape[1]} symbols "
          f"({len(TICKERS)} tickers + {MARKET_INDEX})")
    print(f"Labels computed: {len(labeled)} total")
    print(f"   GOOD:    {counts['GOOD']}")
    print(f"   BAD:     {counts['BAD']}")
    print(f"   NEUTRAL: {counts['NEUTRAL']}  (dropped from master.csv, kept in labeled_all.csv)")
    print(f"Saved: {master_path} ({len(master)} rows, GOOD+BAD only)")
    print(f"Saved: {labeled_all_path} ({len(labeled)} rows, all three classes)")
    print("=" * 60)

    if len(master):
        print("\nSample rows:")
        print(master[["ticker", "call_date", "label", "abnormal_return"]].head(5).to_string(index=False))


if __name__ == "__main__":
    main()
