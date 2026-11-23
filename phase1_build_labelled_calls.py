"""
Build the full call dataset with corrected event dates
=======================================================
Unlike master.csv, this keeps every call with enough price data -- including the calls
Phase 1 labeled NEUTRAL (|abnormal return| <= 2%) and dropped. A strategy can't know in
advance which calls will end up NEUTRAL, so leaving them out tests only on calls that
later moved a lot.

Returns use phase1_abnormal_returns.py (day 0 = call date), the same definitions as Phase 3:
  imm_ar    abnormal return, day -1 -> +1 (the immediate reaction)
  drift_ar  abnormal return, day +2 -> +60 (what happens afterwards)
  label     GOOD if drift_ar > 0, else BAD
  is_stub   True for transcripts under 1,000 words (empty or header-only files)

The older master.csv (±2% labels, NEUTRAL calls dropped) is not touched.

Usage:
    python phase1_build_labelled_calls.py
    python phase1_build_labelled_calls.py --start 2005-01-01 --end 2025-12-31

Output:
    data/calls_all.csv  (gitignored: >100MB)
"""

import argparse

import numpy as np

from phase1_collect_transcripts_and_prices import DATA_DIR, fetch_transcripts, load_tickers
from phase1_abnormal_returns import DRIFT_WIN, IMMEDIATE_WIN, EventReturns, load_prices

OUT_CSV    = DATA_DIR / "calls_all.csv"
STUB_WORDS = 1000


def main():
    parser = argparse.ArgumentParser(description="Full call dataset with corrected event dates")
    parser.add_argument("--start", default="2005-01-01")
    parser.add_argument("--end", default="2025-12-31")
    args, _unknown = parser.parse_known_args()

    calls = fetch_transcripts(load_tickers(), args.start, args.end, DATA_DIR / "raw_transcripts.csv")
    events = EventReturns(load_prices())

    print("[calls] computing abnormal returns with corrected event dates ...")
    keys = list(zip(calls["ticker"], calls["call_date"]))
    calls["imm_ar"]   = [events.abnormal(t, d, IMMEDIATE_WIN) for t, d in keys]
    calls["drift_ar"] = [events.abnormal(t, d, DRIFT_WIN) for t, d in keys]

    no_drift = calls["drift_ar"].isna()
    print(f"[calls] dropped {no_drift.sum()} calls without enough price data for the drift window")
    calls = calls[~no_drift].reset_index(drop=True)

    calls["label"]      = np.where(calls["drift_ar"] > 0, "GOOD", "BAD")
    calls["word_count"] = calls["transcript_text"].str.split().str.len()
    calls["is_stub"]    = calls["word_count"] < STUB_WORDS

    calls = calls[["ticker", "call_date", "imm_ar", "drift_ar", "label",
                   "word_count", "is_stub", "transcript_text"]]
    calls.to_csv(OUT_CSV, index=False)

    counts = calls["label"].value_counts()
    print("\n" + "=" * 60)
    print(f"Saved: {OUT_CSV} ({len(calls)} calls, {calls['ticker'].nunique()} tickers, "
          f"{calls['call_date'].min().date()} to {calls['call_date'].max().date()})")
    print(f"   GOOD: {counts.get('GOOD', 0)}   BAD: {counts.get('BAD', 0)}")
    print(f"   immediate reaction missing: {calls['imm_ar'].isna().sum()}")
    print(f"   stubs (< {STUB_WORDS:,} words): {calls['is_stub'].sum()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
