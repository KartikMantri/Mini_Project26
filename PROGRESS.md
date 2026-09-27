# Progress Log — Decoding the Game

**Purpose of this file:** a running, append-only log of every real decision made on this
project — not just what changed, but *why*, what it affects downstream, and what's left.
Read this top-to-bottom before doing anything if you're picking this project up cold
(a new Claude session, or Kartik coming back after a break). Newest entries are at the top.

Phase-specific methodology/results live in `docxxs/PhaseN_*.md` — this file is the
connective narrative across phases, including decisions still being debated.

---

## Current snapshot (as of 2026-09-28)

| Phase | Status |
|---|---|
| 0 — Setup & Foundation | ✅ Done |
| 1 — Data Collection | ✅ Done, real data committed and pushed |
| 2 — Signal Extraction | ✅ `features.csv` generated (509 rows, 17 features), pending sanity-check + NaN-handling decision before commit |
| 3 — Type Classifier | ⬜ Not started (planned, see docxxs/Phase3_Type_Classifier.md) |
| 4 — Market-Reaction Test | ⬜ Not started |
| 5 — Trading Backtest | ⬜ Not started |
| 6 — Dashboard & Report | ⬜ Not started |

**Immediate next step:** sanity-check `features.csv` values against known real-world quarters
(same spot-check approach used for Phase 1), decide how to handle the 4 remaining `NaN` rows,
then start Phase 3 (date-based train/test split, Logistic Regression baseline → XGBoost).

**2026-09-27/28 — Phase 2 built, run, and a real bug found and fixed:**
- Wrote `extract_features.py` per the locked Option D design (8 FinBERT + 5 LM + 1 readability
  + 3 specificity = 17 features/call).
- First run hit two real bugs, both fixed and verified against live output before accepting
  results:
  1. `pysentiment2`'s `LM.get_score()` only exposes Positive/Negative/Polarity/Subjectivity —
     it silently doesn't have Uncertainty/Modal/Litigious. Fixed by reading the bundled LM
     dictionary CSV directly and building those category word-sets ourselves (`Modal==1`
     strong, `Modal==3` weak, confirmed against the real CSV contents).
  2. The "first question" keyword marker for the prepared-remarks/Q&A split failed on 31/509
     (6%) transcripts (different phrasing, e.g. Cisco just announces the caller's name with no
     "first question" text), and was found to also silently mis-split some already-"successful"
     rows when an analyst said "my first question is..." mid-Q&A. Replaced with a structural
     rule — split at the **2nd `"Operator:"` turn** — which dropped failures to 4/509 (0.8%),
     all genuine edge cases (no Operator-tagged structure, including one row where the
     transcript tagged `TMO` is actually a PPD earnings call — a source-dataset labeling issue,
     not our bug).
- Final `features.csv`: 509 rows, 255 GOOD / 254 BAD (matches Phase 1 exactly), 4 rows with
  `NaN` Q&A features (decision on how to handle these — drop/impute/leave-as-is for
  XGBoost — still open).
- Full feature reference written to `docxxs/features.md`.
- **Downstream effect:** Phase 3 trains directly on `features.csv` + `label`; the NaN-handling
  decision needs to be made before or during Phase 3's training step, not deferred further.

---

## Decision log (newest first)

### 2026-09-15 — Phase 2 design discussion (in progress, not finalized)
- Confirmed via `master.csv`: transcripts average ~9,400 words (range 1,168–31,216) — 20-80x
  longer than FinBERT's 512-token limit, so chunking is mandatory, not optional.
- Identified the **dilution problem**: averaging FinBERT sentiment across ~20+ mostly-neutral
  chunks per call washes out the 1-2 chunks that actually carry signal (worked example: a
  chunk with 0.58 negative-sentiment probability gets buried inside a whole-call average of
  ~0.06). Decided against a single mean-sentiment feature for this reason.
- Decided (in principle) to compute **mean / min / max / std** of chunk-level sentiment
  instead of a single mean, so outlier moments survive into the feature set even when the
  overall average looks calm.
- Confirmed empirically that the "first question" text marker reliably separates prepared
  remarks from Q&A: present in 49/50 sampled transcripts (98%).
- Discussed whether to hand-assign different weights to different sections (boilerplate vs.
  prepared remarks vs. Q&A vs. Q&A follow-ups) before averaging. **Decided against
  hand-picking weights.** Reasoning: any weight we choose ourselves (e.g. "Q&A = 3x
  prepared remarks") is a guess baked permanently into the feature before any evidence
  supports it — if wrong, it silently makes the feature worse with no way to catch it later.
  Instead: keep prepared-remarks and Q&A features as **separate columns**, drop pure
  boilerplate (that one's safe — it's noise removal, not an assumption about importance),
  and let **Phase 3's classifier learn the actual weighting** from labeled outcomes (its
  logistic regression coefficients / XGBoost feature importances *are* the answer to "how
  much does each part matter", discovered from data instead of assumed).
- Decided against trying to detect Q&A *follow-up* questions specifically (vs. first
  answers) in this pass — text-pattern detection for that is much less reliable than the
  "first question" marker; flagged as a possible future refinement, not part of the first
  build.
- **Downstream effect:** this shapes `features.csv`'s schema directly — likely 8 FinBERT
  columns (4 stats × 2 sections) instead of 4, plus it means Phase 3 should be read closely
  once trained, since its feature importances will retroactively validate (or contradict)
  the "Q&A matters more" assumption embedded in this design.
- **Remaining work:** final go/no-go on Option C vs Option D from Kartik, then build and run
  `extract_features.py`.

### 2026-09-14 — Repo structure flattened
- Moved all project files from the nested `Mini_project26/` subfolder up to the repo root
  (`git mv`, preserving file history as renames, not delete+add).
- Fixed two now-stale references that assumed the nested layout: the folder-tree diagrams in
  `README.md` and `docxxs/Phase0_Setup_and_Foundation.md` (now show `.` as root instead of
  `Mini_Project26/`), and a stale "gitignored — regenerate locally" comment on `data/` left
  over from before Phase 1's data was actually committed.
- Verified via `git diff` that only those two `.md` files changed content — every other file
  was a pure rename with zero content diff.
- Committed (`361fd3d`) and pushed to `origin/master`.
- **Downstream effect:** any doc or script referencing paths relative to a `Mini_project26/`
  prefix would now be wrong — checked, none currently do (the `docxxs/*.md` links like
  `../requirements.txt` were already correct, since `docxxs/` was and still is one level
  below the root files either way).

### 2026-09-13/14 — Phase 1 pipeline executed for real, data committed
- Found `data/` contained only a `.gitkeep` despite `docxxs/Phase1_Data_Collection.md`
  claiming specific real results — flagged this discrepancy to Kartik rather than assuming
  the docs were accurate. Confirmed with him this needed a real re-run, not just a docs fix.
- Hit two environment blockers before the pipeline would run at all (see also the
  `env-windows-ssl-avast` memory entry for full technical detail, kept out of this repo file
  since it's machine-specific, not project-specific):
  - This machine has two separate Python installs; bare `pip` was installing into the wrong
    one. Fix: use `python -m pip install ...`.
  - Avast's HTTPS interception broke SSL verification for every external request (Hugging
    Face, PyPI, Yahoo Finance). Fixed via `pip-system-certs` (for `requests`/`urllib3`-based
    calls) plus manually exporting the Windows Root CA store to a PEM file and pointing
    `CURL_CA_BUNDLE`/`SSL_CERT_FILE` at it (needed separately for `yfinance`, which uses
    `curl_cffi`'s own independent SSL stack, unaffected by the first fix).
- Ran `python collect_data.py --force` for real. Result **exactly matched** what the
  (previously unverified) docs had claimed: 630 transcripts fetched, 255 GOOD / 254 BAD /
  121 NEUTRAL labeled, 509 rows in `master.csv` — confirming the earlier claims were
  genuine, just never actually committed.
- Spot-checked several real AAPL rows against known real-world events (Feb-2021 growth-stock
  rotation, Oct-2022 guidance/China concerns, Feb-2024 China iPhone demand miss) — labels
  matched what actually happened, so the labeling logic is behaving sensibly, not just
  mechanically correct.
- Removed the `data/*.csv` exclusion from `.gitignore` — deliberate, per project scope: the
  brief calls for committing the real dataset despite the size cost (~95MB across 4 files),
  since later phases need to build directly on `master.csv` without requiring a re-run.
  Clarified for Kartik that this ~95MB is *not* the same as the ~5GB Hugging Face dataset
  cache his machine downloaded — that cache is the full 33,362-transcript universe (every
  S&P 500 company, 2005–2025), lives outside the repo entirely, and is a one-time cost —
  Phase 2 onward reads `transcript_text` straight out of `master.csv` and never touches
  Hugging Face again.
- Committed and pushed (originally `af8bc12`, later amended — see next entry).
- **Downstream effect:** `data/master.csv` is now the fixed, real input every later phase
  builds on. No further changes expected to this file unless the labeling threshold or
  ticker universe changes.

### 2026-09-14 — Commit attribution corrected
- Kartik pointed out he'd already asked, before this session, for commits to never include
  a "Co-Authored-By: Claude" line — this had been added anyway (default session behavior).
- Amended the Phase 1 data commit to remove the attribution line and force-pushed it (safe:
  it was the branch tip, pushed only minutes earlier, nothing built on top of it yet).
  Hash changed from `af8bc12` to `18766bf`.
- Saved as a standing instruction so this doesn't recur in future sessions.
- **Downstream effect:** none on project content — commit hash changed, message content did
  not otherwise change.

### (earlier, before this session) — Phase 0 complete
- 45-ticker, sector-diversified universe fixed in `tickers.csv`; date window
  2021-01-01 → 2024-06-30 chosen. Full rationale in `docxxs/Phase0_Setup_and_Foundation.md`.

---

## Remaining work, by phase

- **Phase 2:** lock the FinBERT chunking/weighting design (see open question above), install
  `pysentiment2`, `transformers`, `torch`, `textstat`, `nltk`; write and run
  `extract_features.py`; produce and review `features.csv`; write
  `docxxs/Phase2_Signal_Extraction.md` with real results; commit.
- **Phase 3:** train Logistic Regression baseline then XGBoost on `features.csv` + `label`,
  using a date-based (not random) train/test split; evaluate accuracy/AUC/confusion matrix;
  inspect feature importances (this is also where the Phase 2 weighting assumption gets
  empirically checked); output `predictions.csv`.
- **Phase 4:** compare immediate (day 0→+1) vs. drift (day +2→+20) market reaction windows
  against Phase 3's predicted confidence, via t-test.
- **Phase 5:** long-GOOD/short-BAD backtest with a confidence filter, transaction costs,
  Sharpe ratio, equity curve.
- **Phase 6:** React + Recharts dashboard, final written report.
