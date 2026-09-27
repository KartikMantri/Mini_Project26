"""
Phase 2 — Signal Extraction.

Reads data/master.csv (Phase 1 output) and writes features.csv: one numeric
feature row per earnings call, per the locked design in
docxxs/Phase2_Signal_Extraction.md.

Resumable: re-running skips (ticker, call_date) pairs already written to
features.csv, so a crash partway through doesn't cost a full re-run.
"""

import re
import sys

import nltk
import numpy as np
import pandas as pd
import textstat
import torch
from pysentiment2 import LM
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MASTER_CSV = "data/master.csv"
FEATURES_CSV = "features.csv"
FINBERT_MODEL = "ProsusAI/finbert"
MAX_CHUNK_TOKENS = 510  # leaves room for [CLS]/[SEP]
CHECKPOINT_EVERY = 20

FORWARD_LOOKING_WORDS = [
    "expect", "expects", "expected", "anticipate", "anticipates",
    "guidance", "outlook", "going forward", "next quarter", "next year",
    "plan to", "planning to", "will be", "forecast", "project", "projected",
]


def ensure_nltk_punkt():
    for pkg in ("punkt", "punkt_tab"):
        try:
            nltk.data.find(f"tokenizers/{pkg}")
        except LookupError:
            nltk.download(pkg, quiet=True)


def split_sections(text: str) -> tuple[str, str | None]:
    """Split a transcript into (prepared_remarks, qa) using the 2nd
    "Operator:" turn as the Q&A boundary. The operator always opens the call
    (1st turn), then stays silent through prepared remarks and speaks again
    only to introduce the first Q&A caller (2nd turn) — this holds regardless
    of the exact phrasing used ("first question", "we'll go first to X",
    or just announcing the caller's name directly), unlike matching on the
    literal phrase "first question", which also false-positives on analysts
    saying "my first question is..." mid-Q&A. Returns qa=None if there
    aren't at least 2 "Operator:" turns (rare malformed/atypical transcripts)."""
    matches = list(re.finditer(r"Operator:", text))
    if len(matches) < 2:
        return text, None
    split_point = matches[1].start()
    return text[:split_point], text[split_point:]


def chunk_by_tokens(text: str, tokenizer, max_tokens: int) -> list[str]:
    """Pack whole sentences into chunks that stay under max_tokens."""
    sentences = nltk.sent_tokenize(text)
    chunks, current, current_len = [], [], 0
    for sentence in sentences:
        sent_len = len(tokenizer.encode(sentence, add_special_tokens=False))
        if sent_len > max_tokens:
            # pathological single sentence — truncate on its own
            if current:
                chunks.append(" ".join(current))
                current, current_len = [], 0
            chunks.append(sentence)
            continue
        if current_len + sent_len > max_tokens and current:
            chunks.append(" ".join(current))
            current, current_len = [], 0
        current.append(sentence)
        current_len += sent_len
    if current:
        chunks.append(" ".join(current))
    return chunks


def score_chunks_finbert(chunks: list[str], tokenizer, model, device) -> list[float]:
    """Return net sentiment (P(positive) - P(negative)) per chunk."""
    if not chunks:
        return []
    scores = []
    batch_size = 8
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        inputs = tokenizer(
            batch, return_tensors="pt", padding=True, truncation=True,
            max_length=512,
        ).to(device)
        with torch.no_grad():
            logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        # ProsusAI/finbert label order: 0=positive, 1=negative, 2=neutral
        scores.extend((probs[:, 0] - probs[:, 1]).tolist())
    return scores


def section_stats(scores: list[float]) -> dict:
    if not scores:
        return {"mean": np.nan, "min": np.nan, "max": np.nan, "std": np.nan}
    arr = np.array(scores)
    return {
        "mean": arr.mean(),
        "min": arr.min(),
        "max": arr.max(),
        "std": arr.std(),
    }


def load_lm_categories(lm: LM) -> dict:
    """pysentiment2's LM.get_score() only exposes Positive/Negative/Polarity/
    Subjectivity, but the bundled dictionary CSV also has Uncertainty,
    Litigious, and a single Modal column (1=strong, 2=moderate, 3=weak).
    Build our own word sets straight from that CSV for the categories the
    package doesn't surface."""
    data = pd.read_csv(lm.PATH)
    def word_set(mask):
        return set(data.loc[mask, "Word"].apply(lm.tokenize_first).dropna())
    return {
        "negative": word_set(data["Negative"] > 0),
        "uncertainty": word_set(data["Uncertainty"] > 0),
        "litigious": word_set(data["Litigious"] > 0),
        "weak_modal": word_set(data["Modal"] == 3),
        "strong_modal": word_set(data["Modal"] == 1),
    }


def lm_features(text: str, lm: LM, lm_categories: dict) -> dict:
    tokens = lm.tokenize(text)
    word_count = max(len(tokens), 1)
    def pct(category):
        word_set = lm_categories[category]
        return sum(1 for t in tokens if t in word_set) / word_count
    return {
        "lm_uncertainty_pct": pct("uncertainty"),
        "lm_weak_modal_pct": pct("weak_modal"),
        "lm_strong_modal_pct": pct("strong_modal"),
        "lm_negative_pct": pct("negative"),
        "lm_litigious_pct": pct("litigious"),
    }


def specificity_features(full_text: str, qa_text: str | None) -> dict:
    words = full_text.split()
    word_count = max(len(words), 1)
    numeric_density = sum(1 for w in words if re.search(r"\d", w)) / word_count

    forward_pattern = re.compile(
        "|".join(re.escape(w) for w in FORWARD_LOOKING_WORDS), re.IGNORECASE
    )
    forward_looking_count = len(forward_pattern.findall(full_text))

    if qa_text:
        answer_turns = re.findall(r"^[^:\n]+:\s*(.+)$", qa_text, flags=re.MULTILINE)
        if answer_turns:
            with_number = sum(1 for t in answer_turns if re.search(r"\d", t))
            qa_directness = with_number / len(answer_turns)
        else:
            qa_directness = np.nan
    else:
        qa_directness = np.nan

    return {
        "numeric_density": numeric_density,
        "forward_looking_count": forward_looking_count,
        "qa_directness": qa_directness,
    }


def extract_row_features(ticker, call_date, text, tokenizer, model, device, lm, lm_categories) -> dict:
    prepared, qa = split_sections(text)

    prepared_chunks = chunk_by_tokens(prepared, tokenizer, MAX_CHUNK_TOKENS)
    prepared_scores = score_chunks_finbert(prepared_chunks, tokenizer, model, device)
    prepared_stats = section_stats(prepared_scores)

    if qa:
        qa_chunks = chunk_by_tokens(qa, tokenizer, MAX_CHUNK_TOKENS)
        qa_scores = score_chunks_finbert(qa_chunks, tokenizer, model, device)
    else:
        qa_scores = []
    qa_stats = section_stats(qa_scores)

    row = {"ticker": ticker, "call_date": call_date}
    for stat_name, value in prepared_stats.items():
        row[f"finbert_prepared_{stat_name}"] = value
    for stat_name, value in qa_stats.items():
        row[f"finbert_qa_{stat_name}"] = value

    row.update(lm_features(text, lm, lm_categories))
    row["fog_index"] = textstat.gunning_fog(text)
    row.update(specificity_features(text, qa))
    return row


def load_checkpoint():
    try:
        existing = pd.read_csv(FEATURES_CSV)
        done = set(zip(existing["ticker"], existing["call_date"]))
        return existing, done
    except FileNotFoundError:
        return None, set()


def main():
    ensure_nltk_punkt()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", file=sys.stderr)

    print("Loading FinBERT...", file=sys.stderr)
    tokenizer = AutoTokenizer.from_pretrained(FINBERT_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(FINBERT_MODEL).to(device)
    model.eval()

    lm = LM()
    lm_categories = load_lm_categories(lm)

    master = pd.read_csv(MASTER_CSV)
    existing, done = load_checkpoint()
    rows_written = 0 if existing is None else len(existing)

    header_needed = existing is None
    for i, record in master.iterrows():
        key = (record["ticker"], record["call_date"])
        if key in done:
            continue

        row = extract_row_features(
            record["ticker"], record["call_date"], record["transcript_text"],
            tokenizer, model, device, lm, lm_categories,
        )
        row["label"] = record["label"]

        pd.DataFrame([row]).to_csv(
            FEATURES_CSV, mode="a", header=header_needed, index=False
        )
        header_needed = False
        rows_written += 1

        if rows_written % CHECKPOINT_EVERY == 0:
            print(f"[{rows_written}/{len(master)}] {record['ticker']} {record['call_date']}",
                  file=sys.stderr)

    print(f"Done. {rows_written} rows in {FEATURES_CSV}", file=sys.stderr)


if __name__ == "__main__":
    main()
