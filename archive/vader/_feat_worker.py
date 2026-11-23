"""
Feature extraction worker (OPTIMISED) — standalone subprocess.
Called by extract_features_fast.py; do NOT run directly.
Usage: python _feat_worker.py <chunk_in_csv> <chunk_out_csv> <worker_id>

Key optimisation vs previous version:
  - Removed per-sentence VADER loop (was ~430 calls/transcript = ~80% of runtime)
  - Replaced with fast regex-based sentence splitting + single-pass stats
  - Whole-text and section-level VADER still computed (3 calls vs 430+)
  - Expected speedup: 5-8x, from ~25 hrs to ~3-4 hrs total
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import textstat
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

LM_POSITIVE = {
    "achieve","achieved","achievements","attain","benefit","better","boost",
    "breakthrough","confident","delight","deliver","delivered","enable",
    "enhance","enjoy","exceed","exceeded","excellent","excited","exciting",
    "favorable","gain","gains","great","grew","growth","highest","improve",
    "improved","improvement","increase","increased","innovation","momentum",
    "opportunities","opportunity","optimistic","outperform","outperformed",
    "pleased","positive","profitable","profitability","progress","prosper",
    "record","recovery","remarkable","resilient","reward","solid","strength",
    "strong","stronger","strongest","succeed","success","successful","superior",
    "surpass","surpassed","thriving","upbeat","upturn",
}
LM_NEGATIVE = {
    "adverse","adversely","against","challenge","challenges","challenging",
    "closure","concern","concerns","contraction","decline","declined","declines",
    "declining","decreased","default","deficit","delay","delayed","deteriorate",
    "deterioration","difficult","difficulty","disappoint","disappointed",
    "disappointing","disappointment","disruption","downgrade","downturn","drop",
    "dropped","failure","fell","headwind","headwinds","hurt","impair","impairment",
    "inability","loss","losses","lower","lowered","miss","missed","negative",
    "negatively","obstacle","obstacles","pressure","pressured","problem",
    "recession","restructure","restructuring","risk","risks","setback","shortfall",
    "shrink","shrinkage","slack","slow","slowed","slowdown","slower","sluggish",
    "struggle","suffered","uncertain","uncertainty","underperform","unfavorable",
    "volatile","volatility","weak","weaken","weakness","worsen","worsening",
    "writedown","writeoff",
}
LM_UNCERTAINTY = {
    "almost","apparent","apparently","appear","appeared","appears","approximately",
    "assume","assumed","assumption","believe","believed","conceivable","conditional",
    "contingency","contingent","depend","dependent","depends","doubt","doubtful",
    "estimate","estimated","estimation","expect","expected","expose","exposed",
    "fluctuate","fluctuation","hopefully","if","imprecise","indefinite","indicate",
    "indication","likelihood","may","maybe","might","nearly","occasionally",
    "perhaps","possible","possibly","predict","predicted","prediction","preliminary",
    "presumably","probable","probably","random","reassess","reconsider","reexamine",
    "reinterpret","revise","revised","revision","roughly","seem","seemed","seems",
    "somewhat","speculate","speculation","suggest","suggested","susceptible",
    "tentative","uncertain","uncertainty","unclear","undecided","undefined",
    "undetermined","unforeseeable","unknown","unlikely","unplanned","unpredictable",
    "unproven","unquantifiable","unsettled","unspecified","untested","unusual",
    "variable","variability",
}
FORWARD_WORDS = [
    "expect","expects","expected","anticipate","anticipates","guidance","outlook",
    "going forward","next quarter","next year","plan to","planning to","will be",
    "forecast","project","projected",
]
FWD_PAT = re.compile("|".join(re.escape(w) for w in FORWARD_WORDS), re.IGNORECASE)

# Fast sentence splitter (regex-based, no NLTK download needed, ~10x faster)
SENT_END = re.compile(r'(?<=[.!?])\s+(?=[A-Z])')


def split_sentences_fast(text):
    """Fast regex sentence splitter — no NLTK required."""
    return SENT_END.split(text.strip())


def split_sections(text):
    matches = list(re.finditer(r"Operator:", text))
    if len(matches) < 2:
        return text, None
    return text[:matches[1].start()], text[matches[1].start():]


def word_pct(tokens, word_set):
    if not tokens:
        return 0.0
    return sum(1 for t in tokens if t in word_set) / len(tokens)


def extract_one(text, sid):
    """Extract all features. No per-sentence VADER — uses paragraph sampling."""
    prepared, qa = split_sections(text)
    lower_text = text.lower()
    words = lower_text.split()
    word_count = max(len(words), 1)

    # --- VADER (3 calls: whole, prepared, qa) ---
    ws = sid.polarity_scores(text)
    ps = sid.polarity_scores(prepared)
    feat = {
        "vader_compound":          ws["compound"],
        "vader_pos":               ws["pos"],
        "vader_neg":               ws["neg"],
        "vader_neu":               ws["neu"],
        "vader_prepared_compound": ps["compound"],
        "vader_prepared_pos":      ps["pos"],
        "vader_prepared_neg":      ps["neg"],
    }
    if qa:
        qs = sid.polarity_scores(qa)
        feat["vader_qa_compound"] = qs["compound"]
        feat["vader_qa_pos"]      = qs["pos"]
        feat["vader_qa_neg"]      = qs["neg"]
    else:
        feat["vader_qa_compound"] = float("nan")
        feat["vader_qa_pos"]      = float("nan")
        feat["vader_qa_neg"]      = float("nan")

    # --- Fast sentence-level sentiment stats via paragraph sampling ---
    # Split into paragraphs (newline-separated), score each paragraph once
    # This gives distribution stats without 430 individual VADER calls
    paragraphs = [p.strip() for p in text.split('\n') if len(p.strip()) > 30]
    if paragraphs:
        # Sample up to 60 paragraphs evenly for speed
        step = max(1, len(paragraphs) // 60)
        sample = paragraphs[::step][:60]
        arr = np.array([sid.polarity_scores(p)["compound"] for p in sample])
        feat["sent_compound_mean"] = arr.mean()
        feat["sent_compound_std"]  = arr.std()
        feat["sent_compound_min"]  = arr.min()
        feat["sent_compound_max"]  = arr.max()
        feat["pct_positive_sents"] = (arr > 0.5).mean()
        feat["pct_negative_sents"] = (arr < -0.5).mean()
    else:
        for k in ["sent_compound_mean","sent_compound_std","sent_compound_min",
                  "sent_compound_max","pct_positive_sents","pct_negative_sents"]:
            feat[k] = float("nan")

    # --- Fast sentence count via regex (no NLTK) ---
    sentences = SENT_END.split(text.strip())
    sentence_count = len(sentences)

    # --- Loughran-McDonald word lists ---
    tokens = re.findall(r"[a-z]+", lower_text)
    feat["lm_positive_pct"]    = word_pct(tokens, LM_POSITIVE)
    feat["lm_negative_pct"]    = word_pct(tokens, LM_NEGATIVE)
    feat["lm_uncertainty_pct"] = word_pct(tokens, LM_UNCERTAINTY)
    feat["lm_net_sentiment"]   = feat["lm_positive_pct"] - feat["lm_negative_pct"]

    # --- Readability ---
    feat["fog_index"]           = textstat.gunning_fog(text)
    feat["flesch_ease"]         = textstat.flesch_reading_ease(text)
    feat["avg_sentence_length"] = textstat.words_per_sentence(text)

    # --- Structural / specificity ---
    feat["numeric_density"]         = sum(1 for w in words if re.search(r"\d", w)) / word_count
    feat["word_count"]              = word_count
    feat["sentence_count"]          = sentence_count
    feat["forward_looking_count"]   = len(FWD_PAT.findall(text))
    feat["forward_looking_density"] = feat["forward_looking_count"] / word_count

    # --- Q&A features ---
    if qa:
        turns = re.findall(r"^[^:\n]+:\s*(.+)$", qa, flags=re.MULTILINE)
        if turns:
            feat["qa_directness"] = sum(1 for t in turns if re.search(r"\d", t)) / len(turns)
            feat["qa_turn_count"] = len(turns)
        else:
            feat["qa_directness"] = float("nan")
            feat["qa_turn_count"] = 0
        feat["qa_ratio"] = len(qa) / max(len(prepared), 1)
    else:
        feat["qa_directness"] = float("nan")
        feat["qa_turn_count"] = 0
        feat["qa_ratio"]      = 0.0

    return feat


def main():
    if len(sys.argv) < 4:
        print("Usage: python _feat_worker.py <in.csv> <out.csv> <worker_id>")
        sys.exit(1)

    chunk_in  = Path(sys.argv[1])
    chunk_out = Path(sys.argv[2])
    worker_id = sys.argv[3]
    append    = "--append" in sys.argv   # append to existing out file

    sid = SentimentIntensityAnalyzer()
    df  = pd.read_csv(chunk_in)
    n   = len(df)
    print(f"[worker {worker_id}] starting — {n} rows (append={append})", flush=True)

    id_cols    = ["ticker", "call_date"]
    label_cols = ["label", "abnormal_return"]
    results    = []
    header_written = append and chunk_out.exists()  # don't re-write header if appending

    for i, row in df.iterrows():
        feat = extract_one(row["transcript_text"], sid)
        feat["ticker"]          = row["ticker"]
        feat["call_date"]       = row["call_date"]
        feat["label"]           = row["label"]
        feat["abnormal_return"] = row["abnormal_return"]
        results.append(feat)

        # Checkpoint every 100 rows
        if len(results) % 100 == 0:
            batch = pd.DataFrame(results)
            fc    = [c for c in batch.columns if c not in id_cols + label_cols]
            batch = batch[id_cols + fc + label_cols]
            batch.to_csv(chunk_out, mode="a", header=not header_written, index=False)
            header_written = True
            results = []
            pct = (i + 1) / n * 100
            print(f"[worker {worker_id}] {i+1}/{n} ({pct:.0f}%)", flush=True)

    # Flush remainder
    if results:
        batch = pd.DataFrame(results)
        fc    = [c for c in batch.columns if c not in id_cols + label_cols]
        batch = batch[id_cols + fc + label_cols]
        batch.to_csv(chunk_out, mode="a", header=not header_written, index=False)

    print(f"[worker {worker_id}] DONE — {n} rows -> {chunk_out}", flush=True)


if __name__ == "__main__":
    main()
