"""
Draws docxxs/project_flow_diagram.png: every phase of the project, how they connect, the
methods and formulas used, a worked example, and expected vs current results.

Numbers are typed in by hand from the results docs (features_finbert.md,
Phase3_finbert_results.md, Phase4_finbert_results.md). Update them here when results change.

Usage:
    python docxxs/make_flow_diagram.py
"""

import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

OUT = Path(__file__).with_name("project_flow_diagram.png")

# ---------------------------------------------------------------------------
# Layout constants (inches)
# ---------------------------------------------------------------------------
W      = 28.0
MARGIN = 0.45
COLGAP = 0.28
FS     = 10.5                       # body font size
LH     = FS * 1.42 / 72             # line height
CHARW  = FS * 0.555 / 72            # average character width
NCOL   = 4
COLW   = (W - 2 * MARGIN - (NCOL - 1) * COLGAP) / NCOL
PAD    = 0.14

INK, MUTED, LINE, BG = "#1f2937", "#4b5563", "#d1d5db", "#ffffff"
STATUS = {"DONE": "#059669", "NEXT": "#d97706", "LATER": "#6b7280"}


def wrap(text, width_in, indent_in=0.0):
    chars = max(20, int((width_in - indent_in) / CHARW))
    return textwrap.wrap(text, chars, break_long_words=False) or [""]


# Items in a column: ("h", text) heading, ("p", text) paragraph, ("b", text) bullet,
# ("f", mathtext, height) formula, ("g", draw_fn, height) small chart.
def item_height(it, width):
    kind = it[0]
    if kind == "h":
        return LH * 1.35
    if kind == "p":
        return len(wrap(it[1], width)) * LH + 0.06
    if kind == "b":
        return len(wrap(it[1], width, 0.22)) * LH + 0.04
    return it[2] + 0.06


def column_height(items, width):
    return sum(item_height(it, width) for it in items) + 2 * PAD


def draw_column(ax, fig, H, items, x, ytop, width, color):
    y = ytop - PAD
    inner = width - 2 * PAD
    for it in items:
        kind, h = it[0], item_height(it, inner)
        if kind == "h":
            ax.text(x + PAD, y - 0.02, it[1], fontsize=FS + 0.8, fontweight="bold",
                    color=color, va="top")
        elif kind == "p":
            for i, line in enumerate(wrap(it[1], inner)):
                ax.text(x + PAD, y - i * LH, line, fontsize=FS, color=INK, va="top")
        elif kind == "b":
            for i, line in enumerate(wrap(it[1], inner, 0.22)):
                if i == 0:
                    ax.text(x + PAD, y, "•", fontsize=FS, color=color, va="top",
                            fontweight="bold")
                ax.text(x + PAD + 0.2, y - i * LH, line, fontsize=FS, color=INK, va="top")
        elif kind == "f":
            ax.add_patch(FancyBboxPatch((x + PAD, y - it[2]), inner, it[2] - 0.02,
                                        boxstyle="round,pad=0,rounding_size=0.06",
                                        fc="#f3f4f6", ec=LINE, lw=0.8))
            ax.text(x + PAD + 0.15, y - it[2] / 2, it[1], fontsize=FS + 2.2, color=INK,
                    va="center")
        elif kind == "g":
            gx, gy, gw, gh = x + PAD, y - it[2], inner, it[2] - 0.04
            if it[1] is g_quintiles:          # room for tick labels inside the slot
                gx, gy, gw, gh = gx + 0.55, gy + 0.45, gw - 0.6, gh - 0.5
            gax = fig.add_axes([gx / W, gy / H, gw / W, gh / H])
            it[1](gax)
        y -= h


# ---------------------------------------------------------------------------
# Small charts
# ---------------------------------------------------------------------------

def g_windows(ax):
    ax.set_xlim(-68, 78)
    ax.set_ylim(-1.4, 1.6)
    ax.axis("off")
    spans = [(-61, -1, "#93c5fd", "PRE-CALL  −61 → −1\nmomentum, volatility"),
             (-1, 1, "#f59e0b", "IMMEDIATE\n−1 → +1"),
             (2, 60, "#34d399", "DRIFT  +2 → +60\n(the GOOD/BAD label)")]
    for a, b, c, lab in spans:
        ax.add_patch(Rectangle((a, -0.25), b - a, 0.5, fc=c, ec="white", lw=1))
    ax.text(-31, 0.45, spans[0][3], ha="center", va="bottom", fontsize=8.5, color=INK)
    ax.annotate(spans[1][3], xy=(0, -0.28), xytext=(-14, -1.05), ha="center", va="center",
                fontsize=8.5, color="#92400e", fontweight="bold",
                arrowprops=dict(arrowstyle="-", color="#92400e", lw=0.8))
    ax.text(31, 0.45, spans[2][3], ha="center", va="bottom", fontsize=8.5, color=INK)
    ax.plot([0, 0], [-0.3, 1.2], color=INK, lw=1.2)
    ax.text(0, 1.3, "call = day 0", ha="center", va="bottom", fontsize=8.5, fontweight="bold",
            color=INK)
    for d in (-61, 60):
        ax.text(d, -0.4, f"{d:+d}", ha="center", va="top", fontsize=8, color=MUTED)
    ax.text(77, 0, "trading\ndays", fontsize=7.5, color=MUTED, va="center", ha="right")


def g_years(ax):
    ax.set_xlim(2001.2, 2026.2)
    ax.set_ylim(-0.2, 2.4)
    ax.axis("off")
    rows = [(1.45, "Walk-forward\n(main)", [(2005, 2012, "#e5e7eb", "train only"),
                                           (2012, 2018, "#93c5fd", "validation: choose"),
                                           (2018, 2026, "#fdba74", "holdout: test once")]),
            (0.25, "Kartik's split\n(check)", [(2005, 2021.75, "#a7f3d0", "train (until Oct 2021)"),
                                              (2021.75, 2022, "#ffffff", ""),
                                              (2022, 2026, "#fca5a5", "test 2022–25")])]
    for y, lab, segs in rows:
        ax.text(2004.6, y + 0.4, lab, ha="right", va="center", fontsize=8, color=INK)
        for a, b, c, t in segs:
            ax.add_patch(Rectangle((a, y), b - a, 0.8, fc=c, ec="white", lw=1.5))
            if t:
                ax.text((a + b) / 2, y + 0.4, t, ha="center", va="center", fontsize=8,
                        color=INK)
    for yr in (2005, 2012, 2018, 2022, 2025):
        ax.text(yr + 0.5, -0.12, str(yr), ha="center", va="top", fontsize=7.5, color=MUTED)


def g_quintiles(ax):
    imm   = [-1.94, -0.46, 0.33, 1.07, 2.00]
    drift = [-0.53, -0.02, 0.25, 0.24, 1.10]
    xs = range(1, 6)
    ax.bar([x - 0.19 for x in xs], imm, 0.36, color="#f59e0b", label="immediate (−1 → +1)")
    ax.bar([x + 0.19 for x in xs], drift, 0.36, color="#059669", label="drift (+2 → +60)")
    ax.axhline(0, color=INK, lw=0.8)
    ax.set_xticks(list(xs), ["Q1\nlowest", "Q2", "Q3", "Q4", "Q5\nhighest"], fontsize=8)
    ax.tick_params(axis="y", labelsize=8)
    ax.set_ylabel("avg abnormal return %", fontsize=8)
    for x, v in zip(xs, drift):
        ax.text(x + 0.19, v + (0.08 if v >= 0 else -0.08), f"{v:+.2f}", ha="center",
                va="bottom" if v >= 0 else "top", fontsize=7, color="#065f46")
    ax.legend(fontsize=7.5, frameon=False, loc="upper left")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_ylim(-2.6, 2.7)


# ---------------------------------------------------------------------------
# Content
# ---------------------------------------------------------------------------

PHASES = [
    dict(num="0", title="SETUP & UNIVERSE", tag="pick the companies, the years and the tools",
         status="DONE", color="#4b5563",
         what=[("p", "Before collecting anything we fix the playground: which companies, which "
                     "years, and what to compare against."),
               ("b", "200 large US companies (S&P 500 members) from every sector, listed in "
                     "tickers.csv."),
               ("b", "Years 2005–2025: long enough to include very different markets (2008 crisis, "
                     "COVID 2020, 2022 rate hikes)."),
               ("b", "Benchmark = the S&P 500 index (^GSPC). A stock is always compared with the "
                     "market, so a day when everything rises does not count as good news for one "
                     "company.")],
         how=[("b", "Python 3.12 with pandas, scikit-learn, XGBoost, PyTorch and Hugging Face "
                    "transformers (FinBERT runs on the laptop GPU)."),
              ("b", "GitHub repo. Files over 100 MB (raw transcripts) stay on each laptop and are "
                    "rebuilt by scripts."),
              ("b", "PROGRESS.md logs every decision and why; docxxs/ has one document per phase.")],
         example=[("p", "A row of tickers.csv:   AAPL, Information Technology"),
                  ("p", "Apple holds 4 earnings calls a year, so about 80 calls between 2005 and "
                        "2025. Across 200 companies that is ~13,000 calls.")],
         results=[("h", "Expected"),
                  ("p", "A fixed, documented list of companies and years."),
                  ("h", "Now"),
                  ("b", "200 tickers; 198 have transcripts."),
                  ("b", "Caveat (survivorship bias): these are TODAY'S S&P 500 members, so firms "
                        "that collapsed earlier are missing.")],
         out="tickers.csv  —  200 companies + sector"),

    dict(num="1", title="DATA COLLECTION & LABELS", tag="every call's transcript + what the stock did",
         status="DONE", color="#2563eb",
         what=[("p", "An earnings call is a ~1 hour conference call each quarter: the CEO and CFO "
                     "present results (PREPARED REMARKS), then answer analysts' questions (Q&A)."),
               ("p", "For every call we collect the transcript and the stock prices around it, then "
                     "label the call GOOD or BAD by what the stock did AFTERWARDS compared with the "
                     "market."),
               ("b", "Transcripts: a public Hugging Face dataset of earnings-call transcripts."),
               ("b", "Prices: daily closing prices from Yahoo Finance (data/prices.csv)."),
               ("b", "Day 0 = the call's date; days are counted in trading days.")],
         how=[("h", "Abnormal return (AR) = the stock's move beyond the market"),
              ("f", r"$AR = \left(\frac{P_{\mathrm{end}}}{P_{\mathrm{start}}} - 1\right) - "
                    r"\left(\frac{M_{\mathrm{end}}}{M_{\mathrm{start}}} - 1\right)$", 0.62),
              ("p", "P = stock price, M = S&P 500 level, at the start and end of a window."),
              ("h", "Three windows around every call"),
              ("g", g_windows, 1.45),
              ("h", "Label"),
              ("p", "GOOD if the drift AR (day +2 → +60) > 0, otherwise BAD. Transcripts under "
                    "1,000 words (empty stubs) are skipped.")],
         example=[("p", "A call on 1 February. From day +2 to day +60:"),
                  ("b", "Stock: 100 → 106   (+6.0%)"),
                  ("b", "S&P 500: 4,000 → 4,080   (+2.0%)"),
                  ("b", "AR = 6.0% − 2.0% = +4.0%   →   GOOD"),
                  ("p", "If the stock had risen only +1% while the market rose +2%, AR = −1% → BAD, "
                        "even though its price went up. We only care about beating the market."),
                  ("p", "The same formula gives the immediate reaction (day −1 → +1), which Phase 3 "
                        "learns from, and the pre-call return (day −61 → −1).")],
         results=[("h", "Expected"),
                  ("p", "Roughly half GOOD and half BAD; thousands of calls so the statistics are "
                        "reliable."),
                  ("h", "Now"),
                  ("b", "13,334 transcripts → 12,789 with prices → 12,716 usable (73 stubs)."),
                  ("b", "GOOD 6,817 (53%) / BAD 5,972 (47%)."),
                  ("b", "Saved as data/calls_all.csv (710 MB, local only; rebuilt by "
                        "build_call_dataset.py)."),
                  ("b", "Paramount (PARA): its price series is some other instrument, so its 77 "
                        "calls get no returns (excluded from training and tests). Fixed 12 Nov.")],
         out="data/calls_all.csv  —  12,789 calls: transcript text + returns (pre, immediate, drift) + GOOD/BAD"),

    dict(num="2", title="SIGNAL EXTRACTION (FinBERT FEATURES)", tag="turn 8,000 words into 18 numbers",
         status="DONE", color="#7c3aed",
         what=[("p", "A model cannot read 8,000 words directly, so every transcript becomes 18 "
                     "numbers that describe HOW management talks: positive or negative, confident or "
                     "hedging, specific or vague, simple or complex."),
               ("h", "Why FinBERT and not VADER?"),
               ("p", "Version 1 used VADER, a general sentiment tool built for tweets. On long "
                     "transcripts it gave +1.0 ('very positive') to 99.5% of calls, so it could not "
                     "tell calls apart."),
               ("p", "FinBERT is a BERT language model trained on financial text, including earnings "
                     "calls. It knows that 'margin compression' is bad news. It reads each sentence "
                     "and outputs three probabilities: positive, negative, neutral.")],
         how=[("h", "1 · Split the call"),
              ("p", "Prepared remarks | Q&A at the 2nd 'Operator:' line. In the Q&A, the first new "
                    "speaker after the operator is the analyst; the others are executives (96.1% "
                    "correct, checked on transcripts that mark Q/A)."),
              ("h", "2 · Score every sentence (≥ 4 words) with FinBERT"),
              ("f", r"$\mathrm{tone}_s = P(\mathrm{pos})_s - P(\mathrm{neg})_s$     range −1 … +1", 0.4),
              ("h", "3 · Combine sentences into call-level features"),
              ("f", r"prep_net_tone $= \frac{1}{N}\sum_{s=1}^{N} \mathrm{tone}_s$   (prepared sentences)", 0.62),
              ("b", "Same for executives' answers (qa_exec_net_tone) and analysts' questions "
                    "(analyst_tone)."),
              ("b", "neg_share = share of sentences with P(neg) > 0.5;  tone_p10 = 10th-percentile "
                    "tone;  tone_std = how much tone swings."),
              ("b", "tone_gap = prepared tone − Q&A tone;  delta_tone = this call's tone − the SAME "
                    "company's previous call."),
              ("b", "FinBERT-FLS: share of forward-looking sentences that are specific (numbers, "
                    "dates) rather than vague."),
              ("b", "Loughran-McDonald finance dictionary: % uncertainty / weak & strong modal / "
                    "negative words; % of numbers; length; Fog readability:"),
              ("f", r"Fog $= 0.4 \times \left(\frac{\mathrm{words}}{\mathrm{sentences}} + "
                    r"100 \times \frac{\mathrm{complex\ words}}{\mathrm{words}}\right)$", 0.62)],
         example=[("p", "Three sentences from one call (illustrative numbers):"),
                  ("b", "\"Revenue grew 12%, ahead of our guidance.\"   P(pos) 0.97, P(neg) 0.01 → "
                        "tone +0.96"),
                  ("b", "\"We saw continued softness in China.\"   P(pos) 0.02, P(neg) 0.93 → "
                        "tone −0.91"),
                  ("b", "\"Our next call is scheduled for April.\"   neutral → tone 0.00"),
                  ("p", "Average of the three = +0.02. A real call has ~400 sentences. Typical values: "
                        "prepared remarks +0.35 (scripted, upbeat), executives' answers +0.21, "
                        "analysts' questions ≈ 0.00 (neutral)."),
                  ("p", "delta_tone (real): Starbucks' Jan 2024 call sounded 0.23 less positive than "
                        "its previous call, one of the biggest drops that quarter."),
                  ("p", "Fog: 20 words per sentence and 15% complex words → 0.4 × (20 + 15) = 14, "
                        "i.e. needs ~14 years of schooling.")],
         results=[("h", "Expected"),
                  ("b", "Features that measure different things (no duplicates)."),
                  ("b", "Not simply measuring how long the call is."),
                  ("b", "GOOD calls should sound a little more positive."),
                  ("h", "Now"),
                  ("b", "12,716 calls × 18 features → data/features_finbert.csv (≈ 4 h on the RTX "
                        "4050 GPU)."),
                  ("b", "Only one near-duplicate pair: neg_share vs tone_p10 (r = −0.97)."),
                  ("b", "No length effect: every feature has |r| < 0.12 with call length."),
                  ("b", "GOOD calls do sound slightly more positive (executives' answers p = 0.001), "
                        "but the difference is tiny (effect size ≈ 0.08). One call tells you little; "
                        "many calls together do."),
                  ("b", "4 transcripts unreadable (no line breaks) → dropped in Phase 3.")],
         out="data/features_finbert.csv  —  12,716 calls × 18 numbers (+ label, abnormal return)"),

    dict(num="3", title="TYPE CLASSIFIER (GOOD / BAD SCORE)", tag="turn 18 numbers into one score from 0 to 1",
         status="DONE", color="#059669",
         what=[("p", "Teach a model to turn the 18 numbers into ONE score between 0 and 1 per call: how "
                     "likely the stock is to jump right after the call. High score = the call "
                     "'sounds good'."),
               ("h", "Why learn from the immediate reaction?"),
               ("p", "The 60-day drift is mostly noise (news, market swings), so a model trained "
                     "directly on GOOD/BAD learns little (we tried: AUC 0.515). The 1–2 day reaction "
                     "is a cleaner teacher: it shows how investors read the call. Phase 4 then checks "
                     "whether this score ALSO predicts the later drift."),
               ("h", "No cheating"),
               ("p", "Every score is out-of-sample: the call, and its whole year, were never in that "
                     "model's training data.")],
         how=[("h", "Logistic Regression (final model)"),
              ("f", r"$z = \frac{x - \mu}{\sigma}$    (μ, σ = mean and spread over training calls)", 0.6),
              ("f", r"score $= 1 \;/\; \left(1 + e^{-(b_0 + w_1 z_1 + w_2 z_2 + \ldots + w_{10} z_{10})}\right)$", 0.5),
              ("p", "Each feature is standardised (z = how unusual it is: 0 = typical, ±2 = rare), "
                    "multiplied by a learned weight w, summed, and squashed into 0–1. XGBoost (300 "
                    "small decision trees added together) was the comparison: same accuracy, so the "
                    "simpler LR wins."),
              ("h", "Walk-forward: never look into the future"),
              ("g", g_years, 1.5),
              ("b", "To score year Y, train only on calls before Y, with a gap so no training label "
                    "uses prices from year Y."),
              ("b", "Greedy choice of inputs: add the feature group that raises 2012–17 AUC the most; "
                    "stop when the gain < 0.002."),
              ("h", "AUC: how we grade it"),
              ("p", "Take one call whose stock went up and one that went down: AUC = the chance the "
                    "up-call got the higher score. 0.5 = coin flip, 1.0 = perfect.")],
         example=[("p", "Starbucks, 30 Jan 2024 (real numbers):"),
                  ("b", "delta_tone −0.23 → z = −3.0 (tone fell sharply) × w 0.25 = −0.75"),
                  ("b", "analyst_tone z = −0.7 (analysts negative) × w 0.29 = −0.19"),
                  ("b", "prep_net_tone z = +0.9 (upbeat script) × w 0.13 = +0.12"),
                  ("b", "the other 7 inputs: −0.01;  starting value b0 = +0.05"),
                  ("b", "total = −0.77  →  score = 1 / (1 + e^0.77) = 0.32  →  bottom 20%"),
                  ("p", "What happened: the stock drifted −8.3% vs the market ✓. But Rockwell scored "
                        "0.30 and drifted +2.1% ✗. Single calls are noisy; the edge only shows in "
                        "averages over many calls (Phase 4)."),
                  ("p", "Strongest weights: analyst_tone +0.29, delta_tone +0.25, prep_net_tone +0.13, "
                        "pre-call return −0.08 (stocks that already ran up react worse).")],
         results=[("h", "Expected"),
                  ("b", "Reaction AUC 0.55–0.65 is good for stock data. Above ~0.70 would suggest a "
                        "leak (future data sneaking in)."),
                  ("b", "GOOD/BAD AUC only just above 0.5 is normal."),
                  ("h", "Now (holdout 2018–2025, never used for choices)"),
                  ("b", "Chosen: LR on 8 FinBERT tone features + 2 market features."),
                  ("b", "Reaction AUC 0.611  (old VADER model: 0.588)."),
                  ("b", "GOOD/BAD AUC 0.537 (VADER 0.530). XGBoost 0.618 / 0.533: the same."),
                  ("b", "Kartik's fixed split (train to Oct 2021, test 2022–25 once): AUC 0.611, so the "
                        "result does not depend on the setup."),
                  ("b", "Velu's TF-IDF word model added nothing once FinBERT tone was in.")],
         out="data/predictions_finbert.csv  —  an out-of-sample score (0–1) for 9,934 calls, 2012–2025"),

    dict(num="4", title="MARKET-REACTION TEST (THE SCIENTIFIC CORE)", tag="does the market miss part of the signal?",
         status="DONE", color="#dc2626",
         what=[("p", "If the market were perfectly efficient, everything a call reveals would be in "
                     "the price within a day, and the score would NOT predict the drift (day +2 → +60)."),
               ("p", "If high-score stocks keep beating low-score stocks for weeks after the call, "
                     "investors under-reacted to the language. That is evidence of a market "
                     "inefficiency (and maybe a trading opportunity, Phase 5)."),
               ("h", "Two possible answers, both valid science"),
               ("b", "Drift gap ≈ 0 → the market is efficient."),
               ("b", "Drift gap > 0 and significant → under-reaction.")],
         how=[("h", "1 · Sort calls into 5 groups, every quarter"),
              ("p", "Within each calendar quarter, rank calls by score: Q1 = lowest 20% … Q5 = "
                    "highest 20%. Comparing inside a quarter removes market-wide booms and crashes."),
              ("f", r"gap$_q$ = mean drift(Q5) − mean drift(Q1)", 0.4),
              ("h", "2 · Is the average gap real or luck?"),
              ("f", r"$t = \frac{\mathrm{mean\ gap}}{\mathrm{SE}_{\mathrm{NW}}}$     p = chance of |t| this big if there were no effect", 0.62),
              ("p", "Newey-West standard error: corrects for neighbouring quarters' 60-day windows "
                    "overlapping. Rule of thumb: |t| > 2 → p < 0.05 (significant)."),
              ("h", "3 · Does it survive controls? (regression)"),
              ("f", r"drift$_i = a_q + b\cdot\mathrm{rank}_i + c\cdot\mathrm{imm}_i + "
                    r"d\cdot\mathrm{pre}_i + e\cdot\mathrm{vol}_i + \varepsilon_i$", 0.42),
              ("p", "rank = 0 for the lowest-scored call, 1 for the highest, so b = drift of the best "
                    "minus the worst call after removing the immediate reaction, momentum and "
                    "volatility. a_q = one constant per quarter; errors clustered by quarter; returns "
                    "capped at the 1st / 99th percentile."),
              ("h", "Holdout 2018–2025, average return by group"),
              ("g", g_quintiles, 2.4)],
         example=[("p", "2024 Q1: 196 calls (real):"),
                  ("b", "Q5: the 40 calls the model liked most (scores 0.64–0.83) → average drift "
                        "+0.74%"),
                  ("b", "Q1: the 39 calls it liked least (scores 0.30–0.49, incl. Starbucks −8.3%) → "
                        "average drift −0.99%"),
                  ("b", "gap = +0.74% − (−0.99%) = +1.73%"),
                  ("p", "Repeat for all 30 holdout quarters: 22 gaps positive, 8 negative, average "
                        "+1.62%, standard error 0.47%."),
                  ("b", "t = 1.62 / 0.47 ≈ 3.4  →  p = 0.002: a gap this big would appear by pure luck "
                        "only ~2 times in 1,000."),
                  ("p", "Coin analogy: 9 heads in 10 flips (p ≈ 0.02) makes you doubt the coin is "
                        "fair. Here, the 'coin' is the claim that language does not matter."),
                  ("p", "Individual stocks are still noisy: Abbott, the top-scored call that quarter "
                        "(0.83), drifted −8.3%.")],
         results=[("h", "Expected"),
                  ("p", "Efficient market: drift gap ≈ 0. Under-reaction: drift gap > 0 with p < 0.05, "
                        "also after controls."),
                  ("h", "Now (holdout 2018–2025)"),
                  ("b", "Immediate gap +3.94% (t 13.2): the market reacts strongly to the language."),
                  ("b", "Drift gap +1.62% (p 0.002); with controls +1.31% (p 0.013) ✓"),
                  ("b", "29% of the total top-vs-bottom move happens AFTER the call → under-reaction."),
                  ("b", "Model without language: drift gap −0.36% (p 0.53). The language is what "
                        "matters."),
                  ("b", "Velu's VADER score, same test on the same calls: +1.40% (p 0.059), with "
                        "controls p 0.099 → not significant at 5%."),
                  ("b", "Smaller but significant on 2012–17 (+0.85%, p 0.03); holds through 2020–21.")],
         out="data/market_reaction_finbert.csv  —  score rank, group (Q1–Q5) and returns for every call"),

    dict(num="5", title="TRADING BACKTEST", tag="could an investor actually make money from it?",
         status="DONE", color="#d97706",
         what=[("p", "Phase 4 shows the drift is statistically real. Phase 5 asks the investor's "
                     "question: after trading costs, could you actually make money from it, and how "
                     "risky would it be?"),
               ("p", "A 'real but not tradeable' answer would still be a valid finding: it would mean "
                     "costs keep the inefficiency alive.")],
         how=[("h", "Long–short portfolio, no peeking"),
              ("b", "At each call: compare its score with the past 365 days of scores only. Top 20% "
                    "→ BUY, bottom 20% → SELL SHORT. Enter day +2, exit day +60."),
              ("b", "Every day: equal weight in ~36 longs and ~36 shorts. Market moves cancel out, "
                    "so profit can only come from the gap."),
              ("f", r"$R_{day}$ = mean($R_{\mathrm{long}}$) − mean($R_{\mathrm{short}}$) − costs", 0.42),
              ("f", r"Sharpe $= \frac{\mathrm{mean}(R_{day}) \times 252}{\mathrm{std}(R_{day}) "
                    r"\times \sqrt{252}}$", 0.62),
              ("b", "Costs: 0.10% per buy or sell (spread + commission) + 0.5% a year to borrow the "
                    "shorted shares. Also tested 0–0.30%.")],
         example=[("p", "Starbucks, 30 Jan 2024 (real):"),
                  ("b", "score 0.317 < past-year bottom-20% cut-off → SHORT."),
                  ("b", "Sell short 1 Feb at \\$87.06, buy back 25 Apr at \\$82.40: stock −5.4%, "
                        "S&P +2.9%."),
                  ("b", "Gain vs market +8.3%; minus 2 × 0.10% costs → +8.1%."),
                  ("p", "One of 2,315 holdout trades: 54% made money after costs, average +0.86% "
                        "per trade.")],
         results=[("h", "Expected"),
                  ("b", "Positive return after costs. Sharpe ≥ 0.5 is respectable, ≥ 1 is strong."),
                  ("h", "Now (holdout 2018 – May 2025)"),
                  ("b", "+6.9% a year after costs, Sharpe 0.74, p 0.032 ✓  (before costs +9.4%, "
                        "Sharpe 1.01). 100 → 161."),
                  ("b", "Worst fall −14% (S&P 500: −34%); beta to the market 0.10."),
                  ("b", "Break-even cost 0.46% per trade, vs 0.10% assumed."),
                  ("b", "No-language score: −4.7% a year. Velu's VADER: +5.9%, Sharpe 0.56 (p 0.11)."),
                  ("b", "Caveats: 2020 alone +24%; 2012–17 only +3.1% a year (p 0.24).")],
         out="data/backtest_finbert_daily.csv + trades  —  daily returns, equity curve, Sharpe"),

    dict(num="6", title="DASHBOARD & REPORT", tag="show it and write it up",
         status="NEXT", color="#0891b2",
         what=[("p", "Present everything: an interactive dashboard to explore calls and results, and "
                     "the final written report.")],
         how=[("b", "Streamlit app (dashboard.py): look up any call → its tone features, its score and "
                    "what the stock did next."),
              ("b", "Charts: returns by group, cumulative return paths (car_paths_finbert.csv), "
                    "backtest equity curve."),
              ("b", "Report: question → data → method → results → limitations.")],
         example=[("p", "Pick 'SBUX 2024-01-30' → tone fell sharply vs last quarter, score 0.32 "
                        "(bottom 20%), stock −8.3% vs the market over the next 60 days.")],
         results=[("h", "Now"),
                  ("b", "dashboard.py exists but reads the old VADER files → switch it to the FinBERT "
                        "outputs of Phases 3–5.")],
         out=None),
]

PIPE = [("0", "Setup", "200 firms, 2005–25"), ("1", "Data", "12,789 calls + GOOD/BAD"),
        ("2", "Features", "18 FinBERT numbers / call"), ("3", "Classifier", "score 0–1 per call"),
        ("4", "Market test", "does the score predict drift?"), ("5", "Backtest", "can it be traded?"),
        ("6", "Dashboard", "show & report")]

SCORE = [("Phase 3 · reaction AUC (0.5 = coin flip)", "0.611", "0.588"),
         ("Phase 3 · GOOD/BAD AUC", "0.537", "0.530"),
         ("Phase 4 · drift gap, top − bottom 20%", "+1.62%  (p 0.002) ✓", "+1.40%  (p 0.059)"),
         ("Phase 4 · drift gap with controls", "+1.31%  (p 0.013) ✓", "+1.29%  (p 0.099)"),
         ("Phase 5 · long-short, after costs", "+6.9%/yr, Sharpe 0.74 ✓", "+5.9%/yr, Sharpe 0.56")]

GLOSSARY = [
    ("Abnormal return", "stock return minus the S&P 500 return over the same days."),
    ("Out-of-sample", "scored by a model that never saw that call (or its year) while training."),
    ("Look-ahead bias", "accidentally using future information; prevented by walk-forward training "
                        "and gaps."),
    ("Holdout", "years kept untouched until the very end (2018–2025): the honest test."),
    ("AUC", "chance an up-call scores higher than a down-call. 0.5 coin flip, 1.0 perfect."),
    ("Quintile", "one of 5 equal groups after sorting (Q1 lowest 20% … Q5 highest 20%)."),
    ("t-statistic", "how many standard errors a result is from zero; |t| > 2 ≈ significant."),
    ("p-value", "chance of a result this strong if there were truly no effect; < 0.05 = significant."),
    ("Sharpe ratio", "return per unit of risk (after costs); ≥ 0.5 decent, ≥ 1 strong."),
    ("FinBERT", "a BERT language model fine-tuned on financial text to label sentences positive / "
                "negative / neutral."),
]

ISSUES = [
    "Fixed 12 Nov: Paramount's (PARA) price series is not Paramount; its 77 calls are excluded and "
    "the market inputs are capped at 1% / 99% of each training set. Phases 3–5 re-run.",
    "~330 calls have no detectable Q&A speakers → those features are blank (filled with the median).",
    "neg_share and tone_p10 are near-duplicates (r = −0.97): decide whether to keep both.",
    "Survivorship bias: today's S&P 500 members only.",
    "Phase 5 profit leans on 2020 (+24%) and is weaker in 2012–17; costs assume close-price fills.",
    "Next: point the Phase 6 dashboard at the FinBERT outputs and write the report.",
]


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

COLS = [("what", "WHAT & WHY  (plain English)"), ("how", "HOW IT WORKS  (methods & formulas)"),
        ("example", "WORKED EXAMPLE"), ("results", "RESULTS  —  expected vs now")]
HEADER_H, COLHEAD_H, CONNECT_H = 0.62, 0.38, 0.8
TOP_H = 7.6


def phase_height(ph):
    return HEADER_H + COLHEAD_H + max(column_height(ph[k], COLW - 2 * PAD) for k, _ in COLS) + 0.15


def bottom_height():
    half = (W - 2 * MARGIN - 0.4) / 2
    gl = sum(len(wrap(b, half - 2.6)) * LH + 0.05 for _, b in GLOSSARY)
    iss = sum(len(wrap(t, half - 0.7)) for t in ISSUES) * LH + len(ISSUES) * 0.05
    return max(gl, iss) + 1.2


def main():
    heights = [phase_height(p) for p in PHASES]
    H = TOP_H + sum(heights) + CONNECT_H * (len(PHASES) - 1) + 0.5 + bottom_height() + 0.5
    fig = plt.figure(figsize=(W, H), facecolor=BG)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")

    # ---- title ----
    y = H - 0.45
    ax.text(MARGIN, y, "Decoding the Game — how the whole project works", fontsize=30,
            fontweight="bold", color=INK, va="top")
    ax.text(MARGIN, y - 0.72, "Detecting information asymmetry in earnings calls with NLP  ·  "
            "Kartik Mantri & Velu Kala, LNMIIT  ·  status as of 12 Nov 2026", fontsize=14,
            color=MUTED, va="top")
    y -= 1.35
    q = ("THE BIG QUESTION:  Every quarter, company managers talk to analysts for about an hour. Does HOW "
         "they talk (confident or nervous, specific or vague) carry information that the stock price "
         "takes weeks to absorb? If so, an AI that reads calls can spot it early. We measure the "
         "language with FinBERT (Phase 2), turn it into one score per call (Phase 3), and test whether "
         "that score predicts how the stock moves AFTER the call, once the first reaction is over "
         "(Phase 4). Phase 5 checks whether it could be traded.")
    lines = textwrap.wrap(q, int((W - 2 * MARGIN - 0.4) / (13 * 0.555 / 72)))
    bh = len(lines) * 13 * 1.45 / 72 + 0.3
    ax.add_patch(FancyBboxPatch((MARGIN, y - bh), W - 2 * MARGIN, bh,
                                boxstyle="round,pad=0,rounding_size=0.12", fc="#eff6ff",
                                ec="#93c5fd", lw=1.2))
    for i, l in enumerate(lines):
        ax.text(MARGIN + 0.2, y - 0.15 - i * 13 * 1.45 / 72, l, fontsize=13, color=INK, va="top")
    y -= bh + 0.35

    # ---- pipeline strip ----
    bw = (W - 2 * MARGIN - 6 * 0.45) / 7
    for i, (n, name, desc) in enumerate(PIPE):
        x = MARGIN + i * (bw + 0.45)
        ph = PHASES[i]
        ax.add_patch(FancyBboxPatch((x, y - 1.1), bw, 1.1, boxstyle="round,pad=0,rounding_size=0.1",
                                    fc=ph["color"], ec="none"))
        ax.text(x + bw / 2, y - 0.3, f"{n} · {name}", ha="center", va="center", fontsize=14,
                fontweight="bold", color="white")
        ax.text(x + bw / 2, y - 0.62, desc, ha="center", va="center", fontsize=10.5, color="white")
        ax.text(x + bw / 2, y - 0.9, ph["status"], ha="center", va="center", fontsize=9,
                fontweight="bold", color="white", alpha=0.9)
        if i < 6:
            ax.annotate("", xy=(x + bw + 0.42, y - 0.55), xytext=(x + bw + 0.03, y - 0.55),
                        arrowprops=dict(arrowstyle="-|>", color=INK, lw=2))
    y -= 1.45

    # ---- scoreboard ----
    sh = 0.42 * (len(SCORE) + 1) + 0.45
    ax.add_patch(FancyBboxPatch((MARGIN, y - sh), W - 2 * MARGIN, sh,
                                boxstyle="round,pad=0,rounding_size=0.12", fc="#f0fdf4",
                                ec="#86efac", lw=1.2))
    ax.text(MARGIN + 0.2, y - 0.15, "RESULTS SO FAR  —  2018–2025 holdout (never used for any "
            "choice)", fontsize=13, fontweight="bold", color="#065f46", va="top")
    cx = [MARGIN + 0.3, MARGIN + 9.5, MARGIN + 15.5]
    ry = y - 0.62
    for x, t in zip(cx, ("", "FinBERT (now)", "VADER (old, Velu)")):
        ax.text(x, ry, t, fontsize=11.5, fontweight="bold", color=INK, va="top")
    for name, a, b in SCORE:
        ry -= 0.42
        ax.text(cx[0], ry, name, fontsize=11.5, color=INK, va="top")
        ax.text(cx[1], ry, a, fontsize=11.5, color="#065f46", fontweight="bold", va="top")
        ax.text(cx[2], ry, b, fontsize=11.5, color=MUTED, va="top")
    ax.text(MARGIN + 21.3, y - 0.62, "In plain words: the AI's reading of the call\npredicts the "
            "stock's next 3 months a little but\nreliably — the market misses part of what\n"
            "managers' language reveals, and trading it\nearned ~7% a year after costs.",
            fontsize=11.5, color=INK, va="top", style="italic")
    y -= sh + 0.45

    # ---- phases ----
    for i, (ph, h) in enumerate(zip(PHASES, heights)):
        c = ph["color"]
        ax.add_patch(FancyBboxPatch((MARGIN, y - h), W - 2 * MARGIN, h,
                                    boxstyle="round,pad=0,rounding_size=0.15", fc="white", ec=c, lw=2))
        ax.add_patch(FancyBboxPatch((MARGIN, y - HEADER_H), W - 2 * MARGIN, HEADER_H,
                                    boxstyle="round,pad=0,rounding_size=0.15", fc=c, ec=c, lw=2))
        ax.text(MARGIN + 0.25, y - HEADER_H / 2, f"PHASE {ph['num']}  ·  {ph['title']}", fontsize=17,
                fontweight="bold", color="white", va="center")
        ax.text(MARGIN + 11.2, y - HEADER_H / 2, f"—  {ph['tag']}", fontsize=13, color="white",
                va="center", style="italic")
        ax.text(W - MARGIN - 0.25, y - HEADER_H / 2, f"STATUS: {ph['status']}", fontsize=13,
                fontweight="bold", color="white", va="center", ha="right",
                bbox=dict(boxstyle="round,pad=0.3", fc=STATUS[ph["status"]], ec="white", lw=1.2))
        top = y - HEADER_H - 0.08
        for j, (key, label) in enumerate(COLS):
            x = MARGIN + j * (COLW + COLGAP) + (0.05 if j == 0 else 0)
            ax.text(x + PAD, top - 0.05, label, fontsize=FS + 1, fontweight="bold", color=c,
                    va="top")
            ax.plot([x + PAD, x + COLW - PAD], [top - COLHEAD_H + 0.04] * 2, color=c, lw=1, alpha=0.5)
            draw_column(ax, fig, H, ph[key], x, top - COLHEAD_H, COLW, c)
            if j:
                ax.plot([x - COLGAP / 2] * 2, [y - h + 0.15, top - 0.05], color=LINE, lw=0.8)
        y -= h
        if ph["out"] and i < len(PHASES) - 1:
            ax.annotate("", xy=(W / 2 - 4, y - CONNECT_H + 0.02), xytext=(W / 2 - 4, y - 0.02),
                        arrowprops=dict(arrowstyle="-|>", color=INK, lw=2.5, mutation_scale=22))
            ax.text(W / 2 - 3.7, y - CONNECT_H / 2, f"passes on:  {ph['out']}", fontsize=11.5,
                    color=INK, va="center", fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.3", fc="#fefce8", ec="#facc15", lw=1))
            y -= CONNECT_H

    # ---- glossary & issues ----
    y -= 0.5
    bh = bottom_height()
    half = (W - 2 * MARGIN - 0.4) / 2
    for k, (title, col) in enumerate((("GLOSSARY  —  words you will meet", "#374151"),
                                      ("KNOWN ISSUES & NEXT STEPS", "#b91c1c"))):
        x = MARGIN + k * (half + 0.4)
        ax.add_patch(FancyBboxPatch((x, y - bh), half, bh, boxstyle="round,pad=0,rounding_size=0.12",
                                    fc="#f9fafb", ec=col, lw=1.4))
        ax.text(x + 0.2, y - 0.18, title, fontsize=14, fontweight="bold", color=col, va="top")
        yy = y - 0.75
        if k == 0:
            for a, b in GLOSSARY:
                ax.text(x + 0.25, yy, a, fontsize=FS, color=INK, va="top", fontweight="bold")
                for line in wrap(b, half - 2.6):
                    ax.text(x + 2.3, yy, line, fontsize=FS, color=INK, va="top")
                    yy -= LH
                yy -= 0.05
        else:
            for n, t in enumerate(ISSUES, 1):
                for i, line in enumerate(wrap(t, half - 0.7)):
                    if i == 0:
                        ax.text(x + 0.25, yy, f"{n}.", fontsize=FS, color=col, va="top",
                                fontweight="bold")
                    ax.text(x + 0.5, yy, line, fontsize=FS, color=INK, va="top")
                    yy -= LH
                yy -= 0.05

    fig.savefig(OUT, dpi=110, facecolor=BG)
    print(f"saved {OUT}  ({W:.0f} x {H:.1f} in)")


if __name__ == "__main__":
    main()
