"""
Phase 2 (v2) — FinBERT features
================================
Rebuilds the call-level language features with FinBERT instead of VADER, scoring every
sentence (not 512-token chunks). One row per call, 18 features:

  Tone (finbert-tone; net tone per sentence = P(positive) - P(negative))
    prep_net_tone       mean net tone, prepared-remarks sentences
    qa_exec_net_tone    mean net tone, executive answers (whole-Q&A tone if roles can't be tagged)
    analyst_tone        mean net tone, analyst questions
    neg_share           share of sentences with P(negative) > 0.5        (whole call)
    tone_p10            10th percentile of sentence net tone               (whole call)
    tone_std            std of sentence net tone                           (whole call)
    tone_gap            prep_net_tone - mean Q&A net tone
    delta_tone          whole-call mean net tone minus the same ticker's previous call
  Forward-looking statements (finbert-fls, whole call)
    specific_fls_share  Specific / (Specific + Non-specific FLS)
    fls_density         (Specific + Non-specific FLS) / sentences
  Dictionary and style (whole call, Operator turns excluded)
    lm_uncertainty_pct, lm_weak_modal_pct, lm_strong_modal_pct, lm_negative_pct
                        Loughran-McDonald word shares (pysentiment2's bundled LM.csv)
    numeric_density     share of words containing a digit
    qa_directness       share of executive answer turns containing a number
    log_word_count      log of the number of words spoken (length control)
    fog_index           Gunning Fog readability index

Every feature uses only the call's own text, except delta_tone (same ticker's EARLIER call).

Text handling: prepared remarks vs Q&A split at the 2nd "Operator:" (the existing rule; a
split leaving under 2% of the text as prepared remarks counts as failed). In Q&A, the first
speaker after an Operator turn who did not speak in the prepared remarks is the analyst;
other speakers are executives. Transcripts that mark turns "Q -" / "A -" use that markup.

Input:   data/calls_all.csv   (build_call_dataset.py; stub transcripts are skipped)
Output:  data/features_finbert.csv  — ticker, call_date, 18 features, label, abnormal_return
Cache:   data/finbert_cache/<ticker>.parquet, so an interrupted run resumes where it stopped

Usage:
    python extract_features_finbert.py --sample 50    # trial: 50 calls, no cache, report only
    python extract_features_finbert.py                # full run (GPU if available, else CPU)
    python extract_features_finbert.py --data-dir /content/drive/MyDrive/data   # e.g. on Colab
"""

import argparse
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import textstat
import torch
from transformers import BertForSequenceClassification, BertTokenizer

TONE_MODEL = "yiyanghkust/finbert-tone"
FLS_MODEL  = "yiyanghkust/finbert-fls"
MAX_LEN    = 128      # tokens per sentence
MIN_WORDS  = 4        # shorter sentences are dropped
MIN_PREP_SHARE = 0.02 # a Q&A split leaving less than this as prepared remarks has failed

FEATURES = [
    "prep_net_tone", "qa_exec_net_tone", "analyst_tone",
    "neg_share", "tone_p10", "tone_std",
    "tone_gap", "delta_tone",
    "specific_fls_share", "fls_density",
    "lm_uncertainty_pct", "lm_weak_modal_pct", "lm_strong_modal_pct", "lm_negative_pct",
    "numeric_density", "qa_directness", "log_word_count", "fog_index",
]

TURN     = re.compile(r"(?m)^[ \t]*([^\n:]{1,150}):[ \t]*")
SENT_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"“(])|\n+")
HEADERS  = {"executives", "analysts", "participants", "corporate participants",
            "conference call participants", "company representatives", "presentation"}


# ---------------------------------------------------------------------------
# Text parsing
# ---------------------------------------------------------------------------

def speaker_key(name):
    """'Q - Edward White' / 'Greg Gordon - Citigroup' -> 'edward white' / 'greg gordon'."""
    name = re.sub(r"^[QA]\s*[-–]\s*", "", name.strip())
    return re.split(r"\s+[-–,]\s+|\s+of\s+|\s+with\s+", name)[0].strip().lower()


def looks_like_speaker(label):
    """A short name, or 'Name - Title, Company' (which can run long)."""
    label = label.strip()
    if not label[:1].isupper() or re.search(r"[?!]", label):
        return False
    words = len(label.split())
    return words <= 6 or (words <= 25 and bool(re.search(r"\s[-–]\s|,", label)))


def speaker_turns(text):
    """(raw name, key, start position, spoken text) for each speaker turn."""
    marks = [m for m in TURN.finditer(text) if looks_like_speaker(m.group(1))]
    turns = []
    for i, m in enumerate(marks):
        key = speaker_key(m.group(1))
        if key in HEADERS:
            continue
        end = marks[i + 1].start() if i + 1 < len(marks) else len(text)
        turns.append((m.group(1).strip(), key, m.start(), text[m.end():end]))
    return turns


def is_operator(key):
    return key.startswith("operator")


def parse_call(text):
    """Split one transcript into scored sentences plus the inputs for the count features."""
    ops = [m.start() for m in re.finditer(r"Operator:", text)]
    if len(ops) < 2:
        status, qa_start = "no_qa", None
    elif ops[1] / len(text) < MIN_PREP_SHARE:
        status, qa_start = "failed", None
    else:
        status, qa_start = "ok", ops[1]

    turns = speaker_turns(text)
    prep_speakers = {k for _, k, pos, _ in turns
                     if not is_operator(k) and (qa_start is None or pos < qa_start)}
    # Use the transcript's own "Q -" / "A -" markup only when it is used throughout
    n_q = sum(bool(re.match(r"Q\s*[-–]", raw)) for raw, *_ in turns)
    n_a = sum(bool(re.match(r"A\s*[-–]", raw)) for raw, *_ in turns)
    markup = n_q >= 3 and n_a >= 3

    sentences, answer_turns, spoken = [], [], []
    analyst, expect_analyst, roles_ok = None, False, False
    for raw, key, pos, body in turns:
        section = "qa" if qa_start is not None and pos >= qa_start else ("prep" if status != "failed" else "call")
        if is_operator(key):
            expect_analyst = section == "qa"
            continue
        spoken.append(body)

        role = None
        if section == "qa":
            if markup:
                role = "analyst" if re.match(r"Q\s*[-–]", raw) else "exec" if re.match(r"A\s*[-–]", raw) else None
            else:
                if expect_analyst and key not in prep_speakers:
                    analyst, expect_analyst = key, False
                if analyst is not None:
                    role = "analyst" if key == analyst else "exec"
            if role == "exec":
                answer_turns.append(body)
            roles_ok = roles_ok or role == "analyst"

        for s in SENT_END.split(body):
            s = " ".join(s.split())
            if len(s.split()) >= MIN_WORDS:
                sentences.append((section, role, s))

    return {"status": status, "roles_ok": roles_ok, "sentences": sentences,
            "answer_turns": answer_turns, "spoken": "\n".join(spoken)}


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class FinBert:
    def __init__(self, device, batch_size):
        self.device, self.batch_size = device, batch_size
        self.tok = BertTokenizer.from_pretrained(TONE_MODEL)
        if BertTokenizer.from_pretrained(FLS_MODEL).get_vocab() != self.tok.get_vocab():
            raise SystemExit("The two models use different vocabularies; this script assumes one shared tokenizer.")
        # use_safetensors=False: load the authors' original weights, not a community conversion
        self.tone = BertForSequenceClassification.from_pretrained(TONE_MODEL, use_safetensors=False).to(device).eval()
        self.fls  = BertForSequenceClassification.from_pretrained(FLS_MODEL, use_safetensors=False).to(device).eval()

        # Label positions come from each model's config, not assumed
        tone_ids = {v.lower(): k for k, v in self.tone.config.id2label.items()}
        fls_ids  = {v.lower(): k for k, v in self.fls.config.id2label.items()}
        self.pos, self.neg = tone_ids["positive"], tone_ids["negative"]
        self.spec, self.nonspec = fls_ids["specific fls"], fls_ids["non-specific fls"]
        print(f"[models] {TONE_MODEL} {self.tone.config.id2label} | {FLS_MODEL} {self.fls.config.id2label} | device {device}")

    @torch.inference_mode()
    def score(self, texts):
        """Per sentence: net tone, P(negative), and the FLS class index."""
        n = len(texts)
        net, pneg, fls = np.zeros(n, np.float32), np.zeros(n, np.float32), np.zeros(n, np.int8)
        order = np.argsort([len(t) for t in texts])   # similar lengths per batch -> less padding
        for start in range(0, n, self.batch_size):
            idx = order[start:start + self.batch_size]
            enc = self.tok([texts[i] for i in idx], padding=True, truncation=True,
                           max_length=MAX_LEN, return_tensors="pt").to(self.device)
            with torch.autocast(device_type=self.device, dtype=torch.float16, enabled=self.device == "cuda"):
                tone = torch.softmax(self.tone(**enc).logits.float(), dim=-1).cpu().numpy()
                cls = self.fls(**enc).logits.argmax(dim=-1).cpu().numpy()
            net[idx] = tone[:, self.pos] - tone[:, self.neg]
            pneg[idx] = tone[:, self.neg]
            fls[idx] = cls
        return net, pneg, fls


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------

def load_lm():
    import pysentiment2
    lm = pd.read_csv(Path(pysentiment2.__file__).parent / "static" / "LM.csv")
    words = lm["Word"].str.lower()
    return {
        "uncertainty": set(words[lm["Uncertainty"] > 0]),
        "weak_modal":  set(words[lm["Modal"] == 3]),
        "strong_modal": set(words[lm["Modal"] == 1]),
        "negative":    set(words[lm["Negative"] > 0]),
    }


def mean_or_nan(values):
    return float(np.mean(values)) if len(values) else np.nan


def call_features(parsed, net, pneg, fls, model, lm):
    sections = np.array([s for s, _, _ in parsed["sentences"]])
    roles = np.array([r if r else "" for _, r, _ in parsed["sentences"]])
    status, roles_ok = parsed["status"], parsed["roles_ok"]
    f = {}

    prep, qa = net[sections == "prep"], net[sections == "qa"]
    f["prep_net_tone"] = mean_or_nan(prep) if status != "failed" else np.nan
    if roles_ok:
        f["qa_exec_net_tone"] = mean_or_nan(net[(sections == "qa") & (roles == "exec")])
        f["analyst_tone"] = mean_or_nan(net[(sections == "qa") & (roles == "analyst")])
    else:
        f["qa_exec_net_tone"] = mean_or_nan(qa)          # fall back to whole-Q&A tone
        f["analyst_tone"] = np.nan

    has = len(net) > 0
    f["neg_share"] = float((pneg > 0.5).mean()) if has else np.nan
    f["tone_p10"]  = float(np.percentile(net, 10)) if has else np.nan
    f["tone_std"]  = float(net.std()) if has else np.nan
    f["tone_gap"]  = f["prep_net_tone"] - mean_or_nan(qa) if status == "ok" else np.nan
    f["_call_tone"] = mean_or_nan(net)                  # for delta_tone, computed across calls

    n_spec, n_non = int((fls == model.spec).sum()), int((fls == model.nonspec).sum())
    f["specific_fls_share"] = n_spec / (n_spec + n_non) if n_spec + n_non else np.nan
    f["fls_density"] = (n_spec + n_non) / len(fls) if has else np.nan

    spoken = parsed["spoken"]
    tokens = re.findall(r"[a-z]+", spoken.lower())
    n_tok = max(len(tokens), 1)
    for name, words in lm.items():
        f[f"lm_{name}_pct"] = sum(t in words for t in tokens) / n_tok
    words = spoken.split()
    f["numeric_density"] = sum(any(c.isdigit() for c in w) for w in words) / max(len(words), 1)
    answers = parsed["answer_turns"]
    f["qa_directness"] = (sum(bool(re.search(r"\d", a)) for a in answers) / len(answers)
                          if roles_ok and answers else np.nan)
    f["log_word_count"] = float(np.log(max(len(words), 1)))
    f["fog_index"] = textstat.gunning_fog(spoken) if words else np.nan

    f["_status"], f["_roles_ok"], f["_sentences"] = status, roles_ok, len(net)
    return f


def process_calls(calls, model, lm):
    """Features for a block of calls; their sentences are scored together for speed."""
    parsed = [parse_call(t) for t in calls["transcript_text"]]
    texts = [s for p in parsed for _, _, s in p["sentences"]]
    net, pneg, fls = model.score(texts)

    rows, start = [], 0
    for p in parsed:
        end = start + len(p["sentences"])
        rows.append(call_features(p, net[start:end], pneg[start:end], fls[start:end], model, lm))
        start = end
    out = pd.DataFrame(rows, index=calls.index)
    out.insert(0, "call_date", calls["call_date"])
    out.insert(0, "ticker", calls["ticker"])
    out["label"] = calls["label"]
    out["abnormal_return"] = calls["drift_ar"]
    return out


def add_delta_tone(df):
    df = df.sort_values(["ticker", "call_date"]).reset_index(drop=True)
    df["delta_tone"] = df["_call_tone"] - df.groupby("ticker")["_call_tone"].shift(1)
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def sample_calls(calls, n, seed=42):
    """Consecutive calls from a few tickers, so delta_tone can be checked too."""
    rng = np.random.default_rng(seed)
    per_ticker = 10
    picked = []
    for ticker in rng.permutation(calls["ticker"].unique()):
        rows = calls[calls["ticker"] == ticker]
        if len(rows) < per_ticker:
            continue
        start = rng.integers(0, len(rows) - per_ticker + 1)
        picked.append(rows.iloc[start:start + per_ticker])
        if sum(len(p) for p in picked) >= n:
            break
    return pd.concat(picked).head(n)


def report(df, elapsed, n_total):
    per_call = elapsed / len(df)
    print("\n" + "=" * 70)
    print(f"Sample: {len(df)} calls, {df['_sentences'].sum():,} sentences "
          f"(median {df['_sentences'].median():.0f} per call)")
    print(f"Time: {elapsed:.1f} s scoring -> {per_call:.2f} s per call")
    print(f"Projected full run ({n_total:,} calls): {per_call * n_total / 3600:.1f} hours")
    print(f"Q&A split: {df['_status'].value_counts().to_dict()} | roles tagged: {int(df['_roles_ok'].sum())} of {len(df)}")
    print("\nSanity check:")
    table = pd.DataFrame({"mean": df[FEATURES].mean(), "min": df[FEATURES].min(),
                          "max": df[FEATURES].max(), "nulls": df[FEATURES].isna().sum()})
    print(table.round(4).to_string())
    print("\nExample calls:")
    for i in np.linspace(0, len(df) - 1, 3).astype(int):
        r = df.iloc[i]
        print(f"\n  {r['ticker']} {r['call_date']} | {r['label']} (drift {r['abnormal_return']:+.1%}) | "
              f"{r['_sentences']} sentences, split {r['_status']}, roles {'yes' if r['_roles_ok'] else 'no'}")
        print("   " + ", ".join(f"{c}={r[c]:.3f}" for c in FEATURES))
    print("=" * 70)


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Phase 2 (v2): FinBERT features")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--sample", type=int, default=0, help="trial run on N calls: no cache, prints a report")
    parser.add_argument("--sample-out", default=None, help="optional CSV path for the trial run's features")
    parser.add_argument("--batch-size", type=int, default=None, help="sentences per batch (default 256 GPU / 32 CPU)")
    args, _unknown = parser.parse_known_args()

    data = Path(args.data_dir)
    cache_dir, out_csv = data / "finbert_cache", data / "features_finbert.csv"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    calls = pd.read_csv(data / "calls_all.csv", parse_dates=["call_date"])
    stubs = calls["is_stub"].astype(bool)
    print(f"[data] {len(calls):,} calls; skipping {stubs.sum()} stub transcripts")
    calls = calls[~stubs].sort_values(["ticker", "call_date"]).reset_index(drop=True)

    model = FinBert(device, args.batch_size or (256 if device == "cuda" else 32))
    lm = load_lm()

    if args.sample:
        chosen = sample_calls(calls, args.sample)
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        feats = process_calls(chosen, model, lm)
        if device == "cuda":
            torch.cuda.synchronize()
        elapsed = time.time() - t0
        feats = add_delta_tone(feats)
        report(feats, elapsed, len(calls))
        if args.sample_out:
            feats[["ticker", "call_date"] + FEATURES + ["label", "abnormal_return"]].to_csv(args.sample_out, index=False)
            print(f"saved {args.sample_out}")
        return

    cache_dir.mkdir(parents=True, exist_ok=True)
    tickers = calls["ticker"].unique()
    t0 = time.time()
    for i, ticker in enumerate(tickers, 1):
        path = cache_dir / f"{ticker}.parquet"
        if path.exists():
            continue
        block = calls[calls["ticker"] == ticker]
        tick = time.time()
        process_calls(block, model, lm).to_parquet(path, index=False)
        print(f"[{i}/{len(tickers)}] {ticker}: {len(block)} calls in {time.time() - tick:.0f} s "
              f"(total {(time.time() - t0) / 60:.0f} min)")

    feats = add_delta_tone(pd.concat([pd.read_parquet(cache_dir / f"{t}.parquet") for t in tickers]))
    feats[["ticker", "call_date"] + FEATURES + ["label", "abnormal_return"]].to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv} ({len(feats):,} calls x {len(FEATURES)} features)")


if __name__ == "__main__":
    main()
