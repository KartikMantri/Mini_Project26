"""
Phase 2 (Expanded) — Fast Parallel Signal Extraction (Orchestrator)
====================================================================
Splits master.csv into N chunks, spawns one independent Python subprocess
per chunk (_feat_worker.py), then merges results.

Resume-aware: if out_NNN.csv already exists with expected rows, skip that chunk.

Usage:
    python extract_features_fast.py               # resume / run fresh
    python extract_features_fast.py --workers 4   # limit cores
    python extract_features_fast.py --force        # delete & redo everything
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

MASTER_CSV    = Path("data/master.csv")
FEATURES_CSV  = Path("data/features_expanded.csv")
CHUNK_DIR     = Path("data/_feat_chunks")
WORKER_SCRIPT = Path("_feat_worker.py")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force",   action="store_true",
                        help="delete existing output and redo everything")
    parser.add_argument("--workers", type=int,
                        default=max(1, os.cpu_count() - 1))
    args, _ = parser.parse_known_args()

    # ------------------------------------------------------------------ setup
    print(f"[features] loading {MASTER_CSV} ...")
    master = pd.read_csv(MASTER_CSV)
    n_total = len(master)
    print(f"[features] {n_total} rows | {args.workers} workers")

    if args.force:
        # Wipe everything
        if FEATURES_CSV.exists():
            FEATURES_CSV.unlink()
        if CHUNK_DIR.exists():
            for f in CHUNK_DIR.iterdir():
                f.unlink()
        print("[features] --force: cleared all previous output")

    # ------------------------------------------- create chunks (or reuse them)
    CHUNK_DIR.mkdir(parents=True, exist_ok=True)

    n_workers  = min(args.workers, n_total)
    chunk_size = max(1, n_total // n_workers)
    all_chunks = [master.iloc[i:i + chunk_size]
                  for i in range(0, n_total, chunk_size)]
    if len(all_chunks) > n_workers:
        all_chunks[-2] = pd.concat([all_chunks[-2], all_chunks[-1]], ignore_index=True)
        all_chunks = all_chunks[:-1]

    chunk_tasks  = []
    tasks_to_run = []   # only chunks that still need work

    for i, chunk in enumerate(all_chunks):
        chunk_in  = CHUNK_DIR / f"in_{i:03d}.csv"
        chunk_out = CHUNK_DIR / f"out_{i:03d}.csv"
        expected  = len(chunk)

        # Check if this chunk is already complete
        if chunk_out.exists():
            try:
                done = sum(1 for _ in open(chunk_out)) - 1   # subtract header
            except Exception:
                done = 0
            if done >= expected:
                print(f"  [worker {i}] already complete ({done} rows) — skipping")
                chunk_tasks.append((chunk_in, chunk_out, i, expected, True))
                continue
            else:
                # Partially done — filter out already-processed rows
                done_df  = pd.read_csv(chunk_out)
                done_key = set(zip(done_df["ticker"], done_df["call_date"].astype(str)))
                remaining = chunk[~chunk.apply(
                    lambda r: (r["ticker"], str(r["call_date"])) in done_key, axis=1
                )].reset_index(drop=True)
                print(f"  [worker {i}] partial ({done}/{expected}) — resuming {len(remaining)} rows")
                remaining.to_csv(chunk_in, index=False)
        else:
            # Fresh chunk
            chunk.to_csv(chunk_in, index=False)

        chunk_tasks.append((chunk_in, chunk_out, i, expected, False))
        tasks_to_run.append((chunk_in, chunk_out, i, expected))

    if not tasks_to_run:
        print("[features] all chunks already complete — merging ...")
    else:
        print(f"[features] {len(tasks_to_run)} chunk(s) need work — spawning workers ...")

    # -------------------------------------------- spawn workers for todo chunks
    t0        = time.time()
    procs     = []
    log_files = []

    for chunk_in, chunk_out, wid, expected in tasks_to_run:
        log_path = CHUNK_DIR / f"worker_{wid:03d}.log"
        log_f    = open(log_path, "w", encoding="utf-8")
        # If partial, tell worker to APPEND to existing out file
        append_flag = ["--append"] if chunk_out.exists() else []
        proc = subprocess.Popen(
            [sys.executable, str(WORKER_SCRIPT),
             str(chunk_in), str(chunk_out), str(wid)] + append_flag,
            stdout=log_f, stderr=log_f,
            cwd=str(Path.cwd()),
        )
        procs.append((proc, chunk_out, wid, expected))
        log_files.append(log_f)
        print(f"  [worker {wid}] PID {proc.pid} -- {len(pd.read_csv(chunk_in))} rows to go")

    # ------------------------------------------------- poll until all done
    if procs:
        print(f"\n[features] polling every 60s ...")
        done_workers = set()
        while len(done_workers) < len(procs):
            time.sleep(60)
            for proc, chunk_out, wid, expected in procs:
                if wid in done_workers:
                    continue
                ret      = proc.poll()
                rows_out = 0
                if chunk_out.exists():
                    try:
                        rows_out = sum(1 for _ in open(chunk_out)) - 1
                    except Exception:
                        pass
                elapsed = (time.time() - t0) / 60
                if ret is not None:
                    status = "DONE" if ret == 0 else f"ERROR(exit {ret})"
                    done_workers.add(wid)
                else:
                    status = "running"
                print(f"  [worker {wid}] {status} | rows out: {rows_out}/{expected} | {elapsed:.0f}min",
                      flush=True)

        for lf in log_files:
            lf.close()

    # ------------------------------------------------------------------ merge
    print("\n[features] merging all chunks ...")
    parts = []
    for chunk_in, chunk_out, wid, expected, was_done in chunk_tasks:
        if chunk_out.exists():
            df = pd.read_csv(chunk_out)
            parts.append(df)
            print(f"  chunk {wid}: {len(df)} rows")
        else:
            print(f"  WARNING: chunk {wid} output missing!")

    if not parts:
        print("ERROR: no output files — extraction failed.")
        sys.exit(1)

    merged = pd.concat(parts, ignore_index=True)
    merged.to_csv(FEATURES_CSV, index=False)

    # Clean up chunk files
    for f in CHUNK_DIR.iterdir():
        try:
            f.unlink()
        except Exception:
            pass
    try:
        CHUNK_DIR.rmdir()
    except OSError:
        pass

    elapsed = (time.time() - t0) / 60
    print(f"\n{'='*60}")
    print(f"Feature extraction complete!")
    print(f"  Total rows : {len(merged)}")
    print(f"  Features   : {len(merged.columns) - 4}")
    print(f"  Time       : {elapsed:.1f} min (this run)")
    print(f"  Saved      : {FEATURES_CSV}")
    print(f"{'='*60}")
    print("\nLabel distribution:")
    print(merged["label"].value_counts().to_string())


if __name__ == "__main__":
    main()
