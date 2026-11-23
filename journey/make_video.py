"""Make the narrated explainer video for Decoding the Game.

    python journey/make_video.py                 # everything: slides -> voice -> video
    python journey/make_video.py --stage slides  # only render slide PNGs (check them first)
    python journey/make_video.py --stage audio   # only synthesise narration
    python journey/make_video.py --stage video   # only assemble (needs slides + audio)

Needs: Google Chrome (slides), ffmpeg with libass (video, subtitles), and
`python -m pip install edge-tts` (Microsoft neural voice; needs internet, sends only the
narration text below). Output goes to journey/video/ (gitignored, the MP4 is large).

Every scene = one slide (HTML -> PNG through headless Chrome) + narration sentences.
Each sentence is voiced separately, so subtitle timing is exact.
"""
import argparse
import asyncio
import hashlib
import html
import json
import math
import re
import shutil
import subprocess
import sys
import wave
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "video"
BUILD = OUT / "build"
FINAL = OUT / "Decoding_the_Game_explained.mp4"
SRT_OUT = OUT / "Decoding_the_Game_explained.srt"

VOICE = "en-IN-NeerjaNeural"
RATE = "+4%"
FPS = 30
SR = 24000                      # edge-tts output rate (mono)
LEAD, GAP, TAIL = 0.6, 0.35, 0.9  # seconds of silence: before first, between, after last sentence

CHROME_CANDIDATES = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    "google-chrome", "chromium", "msedge",
]

PHASE_NAMES = ["Setup", "Data", "Features", "Classifier", "Market test", "Backtest", "Dashboard", "Returns"]

# ----------------------------------------------------------------------------- slide kit
CSS = """
*{box-sizing:border-box;margin:0;padding:0}
html,body{width:1920px;height:1080px;overflow:hidden}
html{background:radial-gradient(1100px 600px at 88% 8%,rgba(61,220,151,.08),transparent 60%),radial-gradient(1000px 700px at 6% 95%,rgba(77,163,255,.08),transparent 60%),linear-gradient(rgba(77,163,255,.045) 1px,transparent 1px) 0 0/60px 60px,linear-gradient(90deg,rgba(77,163,255,.045) 1px,transparent 1px) 0 0/60px 60px,#070b12}
body{background:transparent;color:#d7dee8;font-family:Inter,"Segoe UI",system-ui,sans-serif;position:relative}
.wrap{position:absolute;left:110px;right:110px;top:70px;bottom:170px;z-index:1;display:flex;flex-direction:column}
.top{display:flex;justify-content:space-between;align-items:center;margin-bottom:26px}
.kick{font:600 22px "JetBrains Mono",Consolas,monospace;letter-spacing:.16em;text-transform:uppercase;color:#3ddc97}
.kick.final{color:#ffb547}
.prog{display:flex;gap:10px;align-items:center}
.prog i{width:16px;height:16px;border-radius:50%;border:2px solid #2a3b55;display:block}
.prog i.on{background:#3ddc97;border-color:#3ddc97;box-shadow:0 0 12px #3ddc97}
.prog i.cur{transform:scale(1.4)}
.prog b{width:22px;height:2px;background:#1d2a3d;display:block}
h1{font-size:72px;line-height:1.08;letter-spacing:-.02em;font-weight:800;margin-bottom:30px}
h1 em{font-style:normal;background:linear-gradient(90deg,#3ddc97,#4da3ff);-webkit-background-clip:text;color:transparent}
.body{flex:1;display:flex;gap:56px;min-height:0}
.col{flex:1;min-width:0;display:flex;flex-direction:column;gap:20px}
ul.b{list-style:none;display:flex;flex-direction:column;gap:26px}
ul.b li{font-size:39px;line-height:1.35;padding-left:42px;position:relative;color:#c7d0dc}
ul.b li:before{content:"";position:absolute;left:4px;top:19px;width:14px;height:14px;border-radius:3px;background:#3ddc97;transform:rotate(45deg)}
ul.b li b,p b{color:#fff}
p.t{font-size:38px;line-height:1.45;color:#c7d0dc}
.f{font:500 34px/1.6 "JetBrains Mono",Consolas,monospace;background:#091019;border:1px solid #1d2a3d;border-left:4px solid #4da3ff;border-radius:14px;padding:22px 28px;color:#e8eef6;white-space:pre}
.box{border:1px solid #1d2a3d;background:rgba(18,27,42,.85);border-radius:16px;padding:26px 30px;font-size:34px;line-height:1.45;color:#c7d0dc}
.box.g{border-left:5px solid #3ddc97}.box.a{border-left:5px solid #ffb547}.box.bl{border-left:5px solid #4da3ff}
table{border-collapse:collapse;width:100%;font-size:33px}
th{font:600 23px "JetBrains Mono",monospace;letter-spacing:.06em;text-transform:uppercase;color:#8a97aa;text-align:left;padding:10px 14px;border-bottom:2px solid #2a3b55}
td{padding:13px 16px;border-bottom:1px solid #1d2a3d;color:#d7dee8}
td.n,th.n{text-align:right;font-family:"JetBrains Mono",monospace}
tr.hl td{background:rgba(61,220,151,.09)} tr.m td{color:#8a97aa}
.up{color:#3ddc97}.dn{color:#ff5d73}.am{color:#ffb547}.mu{color:#8a97aa}
code{font:500 .9em "JetBrains Mono",monospace;background:#121b2a;border:1px solid #1d2a3d;border-radius:6px;padding:1px 8px;color:#e8eef6}
.flow{display:flex;flex-wrap:wrap;gap:14px;align-items:center;font:500 31px "JetBrains Mono",monospace}
.flow span{border:1px solid #2a3b55;background:#0d1420;border-radius:10px;padding:12px 18px}
.flow span.hot{border-color:#3ddc97;color:#3ddc97}.flow span.bad{border-color:#ff5d73;color:#ff5d73}
.flow i{font-style:normal;color:#5b6a80}
.kpis{display:grid;grid-template-columns:repeat(4,1fr);gap:22px}
.kpi{border:1px solid #1d2a3d;background:rgba(18,27,42,.85);border-radius:18px;padding:26px}
.kpi .v{font:600 54px "JetBrains Mono",monospace;letter-spacing:-.02em}
.kpi .l{font-size:23px;color:#8a97aa;margin-top:6px}
.bars{display:flex;flex-direction:column;gap:18px}
.bar{display:grid;grid-template-columns:330px 1fr 170px;gap:20px;align-items:center;font:500 30px "JetBrains Mono",monospace}
.bar .t{height:30px;background:#121b2a;border-radius:6px;position:relative}
.bar .t i{position:absolute;top:0;bottom:0;border-radius:6px}
.bar .t s{position:absolute;left:50%;top:-6px;bottom:-6px;width:2px;background:#2a3b55}
.bar span:last-child{text-align:right}
.sub{font:600 27px "JetBrains Mono",monospace;color:#8a97aa;letter-spacing:.08em;text-transform:uppercase;margin:6px 0 4px}
"""

FONTS = ('<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800'
         '&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">')


def prog(k):
    """Mini road: 8 dots, the first k+1 lit, k current. k=None -> none."""
    out = []
    for i in range(8):
        cls = "on" if k is not None and i <= k else ""
        if k is not None and i == k:
            cls += " cur"
        out.append(f'<i class="{cls}"></i>')
    return '<div class="prog">' + "<b></b>".join(out) + "</div>"


def page(kicker, title, body, k=None, final=False):
    return (f'<!doctype html><html><head><meta charset="utf-8">{FONTS}<style>{CSS}</style></head><body>'
            f'<div class="wrap"><div class="top"><div class="kick{" final" if final else ""}">{kicker}</div>{prog(k)}</div>'
            f'<h1>{title}</h1><div class="body">{body}</div></div></body></html>')


def ul(items):
    return '<ul class="b">' + "".join(f"<li>{x}</li>" for x in items) + "</ul>"


def col(*parts):
    return '<div class="col">' + "".join(parts) + "</div>"


def tbl(head, rows, num_from=1, hl=None, muted=None):
    hl, muted = hl or set(), muted or set()
    h = "".join(f'<th class="{"n" if i >= num_from else ""}">{c}</th>' for i, c in enumerate(head))
    body = ""
    for r, row in enumerate(rows):
        cls = "hl" if r in hl else ("m" if r in muted else "")
        body += f'<tr class="{cls}">' + "".join(
            f'<td class="{"n" if i >= num_from else ""}">{c}</td>' for i, c in enumerate(row)) + "</tr>"
    return f"<table><tr>{h}</tr>{body}</table>"


def flow(items):
    parts = []
    for it in items:
        cls = ""
        if it.startswith("!"):
            cls, it = "hot", it[1:]
        elif it.startswith("-"):
            cls, it = "bad", it[1:]
        parts.append(f'<span class="{cls}">{it}</span>')
    return '<div class="flow">' + "<i>→</i>".join(parts) + "</div>"


def bars(rows, scale, fmt="{:+.3f}", label_w=None):
    out = []
    for name, v in rows:
        w = min(abs(v) / scale * 50, 50)
        left = 50 if v >= 0 else 50 - w
        colr = "#3ddc97" if v >= 0 else "#ff5d73"
        out.append(f'<div class="bar"><span>{name}</span><span class="t"><s></s>'
                   f'<i style="left:{left}%;width:{w}%;background:{colr}"></i></span><span>{fmt.format(v)}</span></div>')
    return '<div class="bars">' + "".join(out) + "</div>"


def road_svg(k):
    """Big winding road with 8 checkpoints; checkpoints <= k lit, k highlighted."""
    W, H = 1700, 560
    xs = [110 + i * (W - 220) / 7 for i in range(8)]
    ys = [H / 2 + (150 if i % 2 else -150) for i in range(8)]
    d = f"M {xs[0]} {ys[0]}"
    for i in range(1, 8):
        mx = (xs[i - 1] + xs[i]) / 2
        d += f" C {mx} {ys[i-1]}, {mx} {ys[i]}, {xs[i]} {ys[i]}"
    # lit part: path up to node k
    dl = f"M {xs[0]} {ys[0]}"
    for i in range(1, (k or 0) + 1):
        mx = (xs[i - 1] + xs[i]) / 2
        dl += f" C {mx} {ys[i-1]}, {mx} {ys[i]}, {xs[i]} {ys[i]}"
    g = (f'<svg width="{W}" height="{H}" viewBox="0 0 {W} {H}">'
         f'<defs><linearGradient id="lg" x1="0" x2="1"><stop offset="0" stop-color="#4da3ff"/>'
         f'<stop offset=".75" stop-color="#3ddc97"/><stop offset="1" stop-color="#ffb547"/></linearGradient>'
         f'<filter id="gl"><feGaussianBlur stdDeviation="5" result="b"/><feMerge><feMergeNode in="b"/>'
         f'<feMergeNode in="SourceGraphic"/></feMerge></filter></defs>'
         f'<path d="{d}" fill="none" stroke="#1d2a3d" stroke-width="16" stroke-linecap="round"/>'
         f'<path d="{d}" fill="none" stroke="#2a3b55" stroke-width="2" stroke-dasharray="3 14"/>')
    if k is not None and k > 0:
        g += f'<path d="{dl}" fill="none" stroke="url(#lg)" stroke-width="7" stroke-linecap="round" filter="url(#gl)"/>'
    for i in range(8):
        on = k is not None and i <= k
        cur = k == i
        last = i == 7
        r = 52 if cur else 40
        fill = ("#ffb547" if last else "#3ddc97") if on else "#0d1420"
        stroke = ("#ffb547" if last else "#3ddc97") if on else "#2a3b55"
        txt = "#04140c" if on else "#8a97aa"
        lab = "₹" if last else str(i)
        if cur:
            g += f'<circle cx="{xs[i]}" cy="{ys[i]}" r="{r+18}" fill="none" stroke="{stroke}" stroke-opacity=".5" stroke-width="2" stroke-dasharray="6 6"/>'
        g += (f'<circle cx="{xs[i]}" cy="{ys[i]}" r="{r}" fill="{fill}" stroke="{stroke}" stroke-width="3"'
              f'{chr(32) + "filter=" + chr(39) + "url(#gl)" + chr(39) if cur else ""}/>'
              f'<text x="{xs[i]}" y="{ys[i]+ (14 if cur else 11)}" text-anchor="middle" font-family="JetBrains Mono,Consolas,monospace" '
              f'font-weight="700" font-size="{38 if cur else 30}" fill="{txt}">{lab}</text>')
        ly = ys[i] + (r + 50 if i % 2 else -(r + 30))
        g += (f'<text x="{xs[i]}" y="{ly}" text-anchor="middle" font-family="Inter,Segoe UI,sans-serif" font-size="{30 if cur else 25}" '
              f'font-weight="{700 if cur else 500}" fill="{"#ffffff" if cur else "#8a97aa"}">{PHASE_NAMES[i]}</text>')
    return g + "</svg>"


def road_page(k, title, line):
    body = (f'<div class="col" style="align-items:center;justify-content:center;gap:10px">{road_svg(k)}'
            f'<p class="t" style="text-align:center;max-width:1400px">{line}</p></div>')
    kick = "Finish line" if k == 7 else ("The journey" if k is None else f"Checkpoint {k}")
    return page(kick, title, body, k, final=(k == 7))


def equity_svg():
    data = json.loads((HERE / "data.js").read_text(encoding="utf-8").split("=", 1)[1].rstrip().rstrip(";"))
    W, H, L, R, T, B = 1700, 640, 90, 150, 20, 50
    dates, ser = data["dates"], data["series"]
    show = [("SPX", "S&amp;P 500", "#eef3f8", "10 8"), ("A", "A · 1:1", "#3ddc97", ""),
            ("D", "D · 2:1, 15/15 (our choice)", "#4da3ff", ""), ("E", "E · long-only", "#ffb547", "")]
    lo, hi = 80, 340
    n = len(dates)
    x = lambda i: L + (W - L - R) * i / (n - 1)
    y = lambda v: T + (H - T - B) * (1 - (v - lo) / (hi - lo))
    g = f'<svg width="{W}" height="{H}" viewBox="0 0 {W} {H}">'
    for v in range(100, hi + 1, 50):
        g += (f'<line x1="{L}" x2="{W-R}" y1="{y(v)}" y2="{y(v)}" stroke="{"#2a3b55" if v == 100 else "#16202f"}" '
              f'{"stroke-dasharray=" + chr(39) + "6 6" + chr(39) if v == 100 else ""}/>'
              f'<text x="{L-14}" y="{y(v)+9}" text-anchor="end" fill="#5b6a80" font-size="24" font-family="JetBrains Mono,monospace">₹{v}</text>')
    last = ""
    for i, d in enumerate(dates):
        if d[:4] != last:
            last = d[:4]
            g += (f'<line x1="{x(i)}" x2="{x(i)}" y1="{T}" y2="{H-B}" stroke="#121b2a"/>'
                  f'<text x="{x(i)+6}" y="{H-12}" fill="#5b6a80" font-size="24" font-family="JetBrains Mono,monospace">{last}</text>')
    for key, lab, colr, dash in show:
        s = ser[key]
        pts = " ".join(f"{x(i):.1f},{y(v):.1f}" for i, v in enumerate(s))
        g += (f'<polyline points="{pts}" fill="none" stroke="{colr}" stroke-width="{3 if key == "SPX" else 4}" '
              f'{"stroke-dasharray=" + chr(39) + dash + chr(39) if dash else ""} stroke-linejoin="round"/>')
        g += (f'<circle cx="{x(n-1)}" cy="{y(s[-1])}" r="7" fill="{colr}"/>'
              f'<text x="{x(n-1)+16}" y="{y(s[-1])+9}" fill="{colr}" font-size="26" font-weight="700" '
              f'font-family="JetBrains Mono,monospace">₹{s[-1]:.0f}</text>')
    legend = "".join(f'<span style="margin-right:34px;color:{c}">━ {l}</span>' for _, l, c, _ in show)
    return g + "</svg>", f'<div style="font:500 26px JetBrains Mono,monospace">{legend}</div>'


# ----------------------------------------------------------------------------- the script
def scenes():
    S = []

    def add(html_, say):
        S.append((html_, say))

    # ---------- intro
    add(page("LNMIIT · Mini project · Semester 5", "Decoding the <em>Game</em>",
             col('<p class="t" style="font-size:40px;max-width:1500px">Detecting information asymmetry in earnings calls with NLP, and testing whether the stock market is slow to notice.</p>',
                 '<div class="kpis" style="margin-top:30px">'
                 '<div class="kpi"><div class="v">12,789</div><div class="l">earnings calls, 2005 to 2025</div></div>'
                 '<div class="kpi"><div class="v">0.611</div><div class="l">classifier AUC, unseen years</div></div>'
                 '<div class="kpi"><div class="v up">+1.62%</div><div class="l">slow-reaction gap, p 0.002</div></div>'
                 '<div class="kpi"><div class="v up">+13.3%</div><div class="l">per year, Strategy D, after costs</div></div></div>',
                 '<p class="t mu" style="margin-top:30px;font-size:28px">Kartik Mantri (24UCS246) · Velu Kala (24DCS012) · Supervisor: Asst. Prof. Girish Sharma</p>')),
        "Welcome. This video explains our mini project, Decoding the Game, from start to finish. "
        "It is the work of Kartik Mantri and Velu Kala at the LNM Institute of Information Technology, supervised by Assistant Professor Girish Sharma. "
        "For every phase we will cover what goes in, what happens inside, why each decision was made, a worked example, and what comes out. "
        "At the end, we look at the trading strategies and their returns.")

    add(page("The question", "Can words reveal what managers <em>know</em>?",
             col(ul(["Every quarter, companies hold an <b>earnings call</b>: a prepared script, then live analyst questions.",
                     "Managers know far more than outside investors.",
                     "<b>Q1.</b> Can a computer read the call and tell if the firm is really doing well or badly?",
                     "<b>Q2.</b> Does the market absorb that at once, or <b>react slowly</b>?",
                     "<b>Q3.</b> If it is slow, can a strategy profit <b>after costs</b>?"]))),
        "Every quarter, large companies hold an earnings call. Managers read a prepared script, and then analysts ask live questions. "
        "Managers know far more about the business than outside investors do. "
        "So our question has three parts. "
        "First, can a computer read the language of a call and tell whether the company is really doing well or badly? "
        "Second, if it can, does the stock market absorb that information immediately, or does it react slowly? "
        "And third, if the market is slow, can a trading strategy profit from it after realistic costs?")

    add(page("Game theory", "A <em>signalling game</em>",
             col(flow(["Nature: firm is good or bad", "Sender: management", "!Message: the call", "Receiver: the market", "Action: buy or sell"]),
                 '<div style="height:20px"></div>',
                 ul(["The type is <b>private</b>: only management knows it.",
                     "Talking is free, so the call is <b>cheap talk</b>: a bad firm can try to sound good.",
                     "We build a <b>better receiver</b>, a model that reads the words, and check whether the real market was slow."]))),
        "We frame this as a signalling game from game theory. "
        "Nature decides whether a firm is truly good or bad. That is private information, known only to management. "
        "Management sends a message, which is the earnings call. "
        "Talking is free, so in game theory terms it is cheap talk: a bad firm can try to sound like a good one. "
        "The receiver, the market, has to infer the firm's type from the words, and then act by buying or selling. "
        "Our project builds a better receiver, a model that reads the words, and then checks whether the real market was slow.")

    add(road_page(None, "Seven phases, one pipeline",
                  "transcript → language features → score → market test → backtest → dashboard"),
        "The project is a pipeline of phases, like checkpoints on a road. "
        "Phase zero sets the rules. Phase one collects transcripts and prices, and labels each call. "
        "Phase two turns language into numbers. Phase three trains a model that gives each call a score. "
        "Phase four tests whether the market under-reacts to that score. Phase five trades on it in a realistic backtest. "
        "Phase six puts everything into a dashboard, and the finish line is the returns of our strategies. "
        "The output of each phase is the input of the next.")

    # ---------- phase 0
    add(road_page(0, "Setup &amp; foundation", "Input: the idea and a list of stocks. Output: the universe, the rules, the plan."),
        "Checkpoint zero: setup and foundation.")

    add(page("Phase 0 · Input", "The universe: 200 large stocks",
             col(ul(["<code>tickers.csv</code>: 200 S&amp;P 500 companies, every sector.",
                     "<b>Why large companies?</b> Reliable prices, heavy trading, low costs.",
                     "Every one holds regular calls with full transcripts.",
                     "A signal in the <b>most-watched</b> stocks is stronger evidence than one in tiny, illiquid stocks."])), 0),
        "We chose a universe of 200 large S and P 500 companies across all sectors, saved in tickers dot csv. "
        "Why large companies? Their prices are reliable, they are heavily traded so trading costs are low, and every one of them holds regular earnings calls with full transcripts. "
        "If we find a signal in the most watched stocks in the world, that is stronger evidence than finding one in tiny, illiquid stocks.")

    add(page("Phase 0 · Procedure", "Three rules, fixed before any result",
             col('<div class="box g"><b>1. No look-ahead.</b> A feature may use only the call\'s own text, or the same company\'s <i>earlier</i> calls.</div>',
                 '<div class="box bl"><b>2. A locked holdout.</b> Choices made on 2012–2017. The years 2018–2025 were used once, at the end.</div>',
                 '<div class="box a"><b>3. Success defined in advance:</b> AUC above 0.5 · a significant drift gap · profit after costs.</div>'), 0),
        "Before running any model, we fixed three rules. "
        "Rule one: no look-ahead. Every feature may use only the call's own text, or the same company's earlier calls. "
        "Nothing from the future can leak in, because a backtest that peeks at the future always looks brilliant, and is always fake. "
        "Rule two: a holdout. The years 2018 to 2025 were locked away. All choices were made on 2012 to 2017, and the holdout was used only once, at the end. "
        "Rule three: success was defined in advance by three numbers. A classifier that beats a coin flip, a statistically significant slow-reaction gap, and a backtest that is still profitable after costs.")

    # ---------- phase 1
    add(road_page(1, "Data collection", "Input: transcripts and prices. Output: 12,789 labelled calls."),
        "Checkpoint one: data collection.")

    add(page("Phase 1 · Input", "Two raw sources",
             col(ul(["<b>Transcripts:</b> Hugging Face dataset <code>kurry/sp500_earnings_transcripts</code>, full text with speaker names.",
                     "<b>Prices:</b> daily adjusted close from Yahoo Finance (<code>yfinance</code>), for every stock.",
                     "<b>The market:</b> the S&amp;P 500 index <code>^GSPC</code>, the benchmark for every return."])), 1),
        "Phase one collects the raw data. "
        "Transcripts come from a public Hugging Face dataset of S and P 500 earnings call transcripts, which holds the full text of each call with speaker names. "
        "Daily adjusted closing prices come from Yahoo Finance through the y finance library, for every stock, and for the S and P 500 index itself, which we use as the market.")

    add(page("Phase 1 · Procedure", "Abnormal return",
             col('<div class="f">abnormal return = stock return − S&amp;P 500 return\n                  (over the same days)</div>',
                 '<div class="box bl"><b>Why subtract the market?</b> If the whole market rises 5% and the stock rises 5%, the company did nothing special. We keep only the part of the move that belongs to the company, which is the part its language could explain.</div>'), 1),
        "From prices, we compute the abnormal return. "
        "It is simply the stock's return minus the S and P 500's return, over the same days. "
        "Why subtract the market? If the whole market rises five percent in a month, a stock that rose five percent did nothing special. "
        "We only want the part of the move that belongs to the company, because that is the part the call's language could explain.")

    add(page("Phase 1 · Procedure", "Three windows around day 0",
             col('<div class="f">pre-call    day −61 → −1    how the stock did before   (control input)\n'
                 'immediate   day  −1 → +1    the first reaction         (model target)\n'
                 'drift       day  +2 → +60   the next ~3 months         (the slow part)</div>',
                 ul(["Day 0 = the call's calendar date; windows counted in <b>trading days</b>.",
                     "Drift starts at <b>+2</b> so it never overlaps the first reaction.",
                     "+60 ≈ one quarter, up to the next call."])), 1),
        "Day zero is the calendar date of the call, and we measure three windows in trading days. "
        "The pre-call window, day minus 61 to minus 1, tells us how the stock was doing before the call. It becomes a control input. "
        "The immediate window, day minus 1 to plus 1, is the market's first reaction to the call. "
        "The drift window, day plus 2 to plus 60, is roughly the next three months, up to the next call. "
        "The drift window starts at day plus 2 on purpose, so that it never overlaps the first reaction. "
        "Anything that happens in the drift window is the market still moving, after it has already had time to react.")

    add(page("Phase 1 · Procedure", "Labels: the firm's true type",
             col('<div class="f">GOOD  if drift abnormal return &gt; 0\nBAD   otherwise</div>',
                 ul(["First version: ±2% threshold, neutral calls dropped, 10,493 calls kept.",
                     "Final version keeps <b>all 12,789</b>: a trader can't know in advance which calls end up neutral.",
                     "The two labelings agree on <b>97.7%</b> of calls."])), 1),
        "Each call gets a label. If its drift abnormal return is above zero, it is GOOD. Otherwise it is BAD. "
        "This is the firm's true type, revealed only later. "
        "Our first version used a plus or minus two percent threshold and threw away the neutral calls in between, keeping 10,493 calls. "
        "We later kept every call, 12,789 in total, because a real trader cannot know in advance which calls will turn out neutral. "
        "Dropping them would quietly use future information. "
        "The two labelings agree on 97.7 percent of calls.")

    add(page("Phase 1 · Quality check", "Don't trust bad numbers",
             col(ul(["Every ticker's price series was scanned.",
                     "<b>Paramount (PARA)</b>: series starting near 83,000, almost no link to the market.",
                     "Its <b>77 calls</b> were removed from every return-based test.",
                     "No other ticker had the problem."])), 1),
        "We also checked the price data of every ticker. "
        "Paramount, ticker P A R A, had broken prices, with a series starting near 83,000 and almost no relationship with the market. "
        "Rather than trust bad numbers, we removed its 77 calls from every return-based test. No other ticker had this problem.")

    add(page("Phase 1 · Example", "Starbucks, 30 January 2024",
             col(tbl(["Window", "Days", "Abnormal return", "Meaning"],
                     [["Pre-call", "−61 → −1", '<span class="dn">−15.3%</span>', "lagged the market before the call"],
                      ["Drift", "+2 → +60", '<span class="dn">−8.3%</span>', "lagged the market again after"]], num_from=2),
                 '<div class="box a" style="margin-top:20px">Drift is below 0, so the call is labelled <b class="dn">BAD</b>. We follow this call through every phase.</div>'), 1),
        "Let's follow one call through the whole project: Starbucks, on the 30th of January 2024. "
        "In the three months before the call, Starbucks lagged the S and P 500 by 15.3 percent. "
        "In the three months after, from day plus 2 to plus 60, it lagged by another 8.3 percent. "
        "So its drift abnormal return is minus 8.3 percent, and the call is labelled BAD.")

    add(page("Phase 1 · Output", "What phase 1 hands over",
             col(ul(["<code>data/calls_all.csv</code>: <b>12,789 calls</b>, 198 companies, Oct 2005 → May 2025.",
                     "Columns: <code>pre_ar</code>, <code>imm_ar</code>, <code>drift_ar</code>, <code>label</code>.",
                     "<code>data/prices.csv</code>: daily prices for every stock and the S&amp;P 500.",
                     "Raw transcripts (&gt;500 MB) stay local: GitHub's limit is 100 MB per file."])), 1),
        "The output of phase one is calls all dot csv: 12,789 calls from 198 companies, from October 2005 to May 2025. "
        "Each call has its pre-call, immediate and drift abnormal returns, and its label. "
        "There is also prices dot csv, with the daily prices. "
        "The raw transcript files are over 500 megabytes, too big for GitHub, so they stay on the local machine.")

    # ---------- phase 2
    add(road_page(2, "Signal extraction", "Input: transcript text. Output: 18 language features per call."),
        "Checkpoint two: signal extraction.")

    add(page("Phase 2 · Why language", "Hard numbers vs soft words",
             col(ul(["<b>Hard information</b>: revenue, profit. Priced in within minutes.",
                     "<b>Soft information</b>: tone, hesitation, how analysts push back.",
                     "Soft information is harder to process, so it is exactly where a <b>slow reaction</b> could hide."])), 2),
        "Phase two turns language into numbers. "
        "Earnings figures like revenue and profit are hard information, and the market prices them in within minutes. "
        "Language is soft information: tone, hesitation, how analysts push back. "
        "It is harder to process, so it is exactly where a slow market reaction might hide.")

    add(page("Phase 2 · Procedure", "Who is speaking, and when",
             col(flow(["Transcript", "Speaker turns", "!Prepared remarks", "!Q&amp;A", "Analyst vs executive"]),
                 '<div style="height:14px"></div>',
                 ul(["A line <code>Name:</code> starts a turn; Operator lines are ignored.",
                     "Split at the 2nd <code>Operator:</code> → <b>script</b> vs <b>live Q&amp;A</b>.",
                     "<b>Why?</b> A script can be polished by lawyers. Live answers are much harder to manage.",
                     "Analyst = first speaker after an Operator turn who wasn't in the script."])), 2),
        "First, we split each transcript into speaker turns. A line that begins with a name and a colon starts a new turn, and the operator's lines are ignored. "
        "Then we split the call into two parts, at the second time the operator speaks: the prepared remarks, which are a script, and the question and answer session, which is live. "
        "Why separate them? A script can be polished by lawyers and investor relations. Live answers to unexpected questions are much harder to manage, so they may reveal more. "
        "Finally, we tell analysts from executives. After each operator turn, the first person who did not speak in the prepared remarks is the analyst asking the question.")

    add(page("Phase 2 · Procedure", "FinBERT scores every sentence",
             col('<div class="f">net tone = P(positive) − P(negative)        range −1 … +1</div>',
                 tbl(["Sentence (illustrative)", "P(pos)", "P(neg)", "Net"],
                     [["“Revenue grew to a record this quarter.”", "0.96", "0.01", '<span class="up">+0.95</span>'],
                      ["“We saw continued softness in traffic.”", "0.02", "0.93", '<span class="dn">−0.91</span>']]),
                 '<div class="box bl"><b>Why FinBERT, not VADER?</b> VADER is built for social media and misreads finance words like <i>liability</i> or <i>tax</i>. FinBERT is trained on financial text.</div>'), 2),
        "Every sentence is then scored by FinBERT, a BERT language model trained on financial text. "
        "For each sentence, it gives three probabilities: positive, negative and neutral. "
        "We define net tone as the probability of positive minus the probability of negative, which runs from minus 1 to plus 1. "
        "Why FinBERT? Our first version used VADER, a general sentiment tool built for social media. "
        "VADER misreads finance. Words like liability or tax are neutral in a financial report, but sound negative to a general tool. "
        "FinBERT understands financial context, and it improved every later result.")

    add(page("Phase 2 · Procedure", "18 features in 5 groups",
             col(tbl(["Group", "Features", "Captures"],
                     [["Tone level (3)", "script · exec answers · analysts", "how positive each part sounds"],
                      ["Bad moments (3)", "neg share · worst 10% · spread", "negative sentences, mixedness"],
                      ["Contrast (2)", "tone gap · <b>delta tone</b>", "script vs live; change vs last call"],
                      ["Specificity (2)", "forward-looking share · density", "how specific the outlook is"],
                      ["Dictionary &amp; style (8)", "uncertainty · modal · numbers · Fog …", "hedging, precision, readability"]],
                     num_from=9)), 2),
        "Sentence scores are then summarised into 18 features per call, in five groups. "
        "Tone level: the net tone of the prepared remarks, of executive answers, and of analyst questions. "
        "Bad moments: the share of clearly negative sentences, the tone of the worst ten percent of sentences, and how mixed the tone is. "
        "Contrast: the tone gap between the script and the live Q and A, and delta tone, the change in tone since the company's previous call. "
        "Specificity, from a second FinBERT model that detects forward-looking statements: how many there are, and how specific they are. "
        "And finally dictionary and style: Loughran McDonald word lists for uncertainty, negativity, and modal words like may or will, how often numbers are used, how directly executives answer, the length of the call, and its readability.")

    add(page("Phase 2 · Key decision", "Delta tone: change from your own normal",
             col('<div class="f">delta_tone = this call\'s tone − same company\'s PREVIOUS call tone</div>',
                 ul(["Some CEOs are cheerful <b>every</b> quarter: their absolute tone says little.",
                     "A <b>drop</b> from their own normal is the real news.",
                     "Uses only the past call, so <b>no look-ahead</b>."])), 2),
        "Delta tone deserves a special mention. Some chief executives are cheerful every single quarter, so their absolute tone tells you little. "
        "What matters is a change from their own normal. "
        "So delta tone compares each call only with the same company's previous call, which is in the past, so there is no look-ahead.")

    add(page("Phase 2 · Example", "Starbucks: what the words said",
             col(tbl(["Feature", "Value", "Reading"],
                     [["<code>prep_net_tone</code>", '<span class="up">+0.489</span>', "script still fairly upbeat"],
                      ["<code>analyst_tone</code>", '<span class="dn">−0.046</span>', "analysts slightly negative"],
                      ["<code>delta_tone</code>", '<span class="dn">−0.231</span>', "big drop vs last quarter"],
                      ["<code>tone_std</code>", "0.624", "a mixed call"]], num_from=1, hl={2}),
                 '<div class="box a">The script was positive, but the overall tone had <b>fallen sharply</b>.</div>'), 2),
        "For the Starbucks call, the prepared remarks still sounded fairly upbeat, with a net tone of plus 0.489. "
        "But the analysts' questions were slightly negative, at minus 0.046. The call was mixed. "
        "And most importantly, delta tone was minus 0.231: a big drop compared with the previous quarter. "
        "The script was positive, but the overall tone had fallen sharply.")

    add(page("Phase 2 · Output", "What phase 2 hands over",
             col(ul(["<code>data/features_finbert.csv</code>: <b>12,716 calls × 22 columns</b>.",
                     "73 stub transcripts under 1,000 words skipped.",
                     "Duplicate measures removed after a correlation check.",
                     "GOOD vs BAD t-tests run on <b>pre-2018 calls only</b>: the holdout stays untouched."])), 2),
        "The output of phase two is features finbert dot csv: 12,716 calls with 22 columns. "
        "73 stub transcripts under a thousand words were skipped. "
        "We also checked correlations and removed duplicate measures, and ran GOOD versus BAD t-tests on calls before 2018 only, so the holdout stayed untouched.")

    # ---------- phase 3
    add(road_page(3, "The type classifier", "Input: 10 numbers per call. Output: one score between 0 and 1."),
        "Checkpoint three: the classifier.")

    add(page("Phase 3 · Key decision", "What should the model learn?",
             col('<div class="f">target = did the stock beat the S&amp;P from day −1 to +1 ?   (1 / 0)</div>',
                 '<div class="box bl"><b>Reason 1, a cleaner teacher.</b> The 3-day reaction is mostly caused by the call. A 3-month return is mostly other news. Training on GOOD/BAD directly was worse: AUC 0.511 vs 0.537.</div>',
                 '<div class="box g"><b>Reason 2, an honest test.</b> The model never sees the 3-month outcome. If its score still predicts the drift, the market was slow.</div>'), 3),
        "Phase three trains the classifier that turns these features into one score per call. "
        "The first decision is what the model should predict. We train it to predict the immediate reaction: did the stock beat the market from day minus 1 to plus 1? "
        "Why not train directly on the three-month GOOD or BAD label? There are two reasons. "
        "First, the three-day reaction is mostly caused by the call itself, while a three-month return is mostly caused by other news. So the reaction is a cleaner teacher. "
        "We tested it: training on GOOD and BAD directly gave a worse result, an AUC of 0.511 against 0.537. "
        "Second, it keeps the main test honest. The model never sees the three-month outcome. "
        "So if its score still predicts the three-month drift, that cannot be because the model memorised it. It must be because the market was slow.")

    add(page("Phase 3 · Input", "Ten inputs, chosen on validation years",
             col(tbl(["#", "Inputs", "Group"],
                     [["1–3", "script tone · exec answer tone · <b>analyst tone</b>", "FinBERT tone"],
                      ["4–6", "negative share · worst-10% tone · tone spread", "FinBERT tone"],
                      ["7–8", "tone gap · <b>delta tone</b>", "FinBERT tone"],
                      ["9–10", "pre-call return · pre-call volatility", "Market"]], num_from=9),
                 '<div class="box a">Groups added one at a time on 2012–2017, kept only if AUC rose ≥ 0.002. Forward-looking, dictionary, style and word-frequency features added <b>nothing</b>.</div>'), 3),
        "Which inputs does it use? We grouped the features, and added groups one at a time, keeping a group only if it raised the validation AUC, on 2012 to 2017, by at least 0.002. "
        "Two groups survived: the eight FinBERT tone features, and two market inputs, the pre-call abnormal return and the pre-call volatility. "
        "The forward-looking, dictionary and style groups, and even a separate word-frequency model of the full transcript, added nothing, so they were left out. "
        "Fewer inputs means less chance of fitting noise.")

    add(page("Phase 3 · Procedure", "Clean the inputs, using training data only",
             col(flow(["Cap extremes at 1% / 99%", "Fill blanks with median", "!Standardise to z-scores"]),
                 '<div style="height:14px"></div>',
                 ul(["Caps stop one stock that ran up <b>146%</b> from dominating.",
                     "Blanks: e.g. a company's first call has no previous call for delta tone.",
                     "z-score = how many standard deviations from a typical call.",
                     "All statistics come from <b>training years only</b>, otherwise the future leaks in."])), 3),
        "Before training, the inputs are cleaned, using the training data only. "
        "Extreme values of pre-call return and volatility are capped at the 1st and 99th percentile, so one stock that ran up 146 percent cannot dominate. "
        "Missing values, like delta tone for a company's very first call, are filled with the median. "
        "Then each input becomes a z-score: how many standard deviations it is above or below a typical call. "
        "All of these statistics come from the training years only, because computing them on all years would leak future information.")

    add(page("Phase 3 · The model", "Logistic regression",
             col('<div class="f">total = b₀ + w₁·z₁ + w₂·z₂ + … + w₁₀·z₁₀\nscore = 1 / (1 + e^(−total))          → between 0 and 1</div>',
                 ul(["<b>Why simple?</b> The signal in returns is small and noisy; simple models overfit less.",
                     "The weights are <b>readable</b>: we can explain every score.",
                     "XGBoost tried: AUC 0.618 vs 0.611, no real gain, so we kept the transparent model."])), 3),
        "The model itself is logistic regression. "
        "It multiplies each z-score by a weight, adds them up with an intercept, and squashes the total through the sigmoid function, one over one plus e to the minus total, into a probability between zero and one. "
        "That probability is the call's score. "
        "Why such a simple model? Because the signal in stock returns is small and noisy, and simple models overfit less. "
        "Its weights are also readable, so we can explain exactly why a call scored low. "
        "We did try XGBoost, a powerful tree model. It reached an AUC of 0.618, against 0.611. That is no real difference, so we kept the simpler, transparent model.")

    add(page("Phase 3 · Training", "Walk-forward: never see the future",
             col('<div class="f">model for 2016 ← trained on calls up to ~Dec 2015 → scores 2016\n'
                 'model for 2017 ← trained on calls up to ~Dec 2016 → scores 2017\n'
                 '   …  a fresh model every year, 2010 → 2025</div>',
                 ul(["Weights found by the <b>L-BFGS</b> optimiser; L2 penalty <b>C = 0.05</b> keeps them small.",
                     "<b>7-day embargo</b>: training calls end a week before the cut-off.",
                     "Every score in the project is <b>out-of-sample</b>."])), 3),
        "Nobody types the weights in. The L-BFGS optimisation algorithm finds the weights that best fit past calls, with an L2 penalty, C equals 0.05, that keeps the weights small. "
        "And the model is trained walk-forward. Every year from 2010 to 2025, a fresh model is trained only on earlier calls, then used to score the next year. "
        "There is also a seven-day embargo: a training call must be at least seven days before the cut-off, so its reaction was already known. "
        "The result is that every single score in the project is out of sample, produced by a model that had never seen that call, or anything after it.")

    add(page("Phase 3 · What it learned", "Weights of the 2024 model",
             col(bars([("analyst_tone", .287), ("delta_tone", .247), ("prep_net_tone", .136), ("neg_share", .127),
                       ("pre_vol", .075), ("tone_p10", .035), ("tone_gap", .035), ("qa_exec_tone", -.028),
                       ("pre_ar", -.069), ("tone_std", -.085)], .3)), 3),
        "Here is what the 2024 model learned. "
        "The biggest weight, plus 0.287, is on analyst tone: when the analysts sound positive, the stock tends to react up. "
        "Second is delta tone, plus 0.247: improving tone compared with last quarter is good news. "
        "Prepared remarks tone is also positive, at plus 0.136. "
        "A very mixed call, with a high tone spread, is negative. "
        "And a stock that already ran up before the call tends to react worse, which is the small negative weight on pre-call return. "
        "Note that negative share and the worst ten percent tone are almost mirror images, so their two weights must be read together. "
        "In game theory terms, the model trusts the unscripted signals most. The analysts, and the change in tone, are the hardest parts of a call to fake.")

    add(page("Phase 3 · Example", "Starbucks → score 0.317",
             col(tbl(["Input", "z-score", "× weight", "= contribution"],
                     [["delta_tone", "−3.04", "+0.247", '<span class="dn">−0.752</span>'],
                      ["analyst_tone", "−0.66", "+0.287", '<span class="dn">−0.189</span>'],
                      ["prep_net_tone", "+0.94", "+0.136", '<span class="up">+0.128</span>'],
                      ["pre_ar", "−1.58", "−0.069", '<span class="up">+0.110</span>'],
                      ["tone_std", "+1.00", "−0.085", '<span class="dn">−0.084</span>'],
                      ["other 5 + intercept", "", "", "+0.023"],
                      ["<b>Total</b>", "", "", "<b>−0.764</b>"]], hl={6}),
                 '<div class="f">score = 1 / (1 + e^0.764) = 0.317</div>'), 3),
        "Now the Starbucks call. Its delta tone of minus 0.231 is minus 3.04 standard deviations, and times the weight of 0.247, that contributes minus 0.752. "
        "Analyst tone contributes minus 0.189. The upbeat script adds plus 0.128, the earlier price fall adds plus 0.110, and the mixed tone subtracts 0.084. "
        "The other five inputs subtract 0.030 together, and the intercept adds 0.053. "
        "The total is minus 0.764. Through the sigmoid, that gives a score of 0.317, one of the lowest of that quarter. "
        "The model's message is clear: whatever the script says, the tone fell three standard deviations, and that is a warning.")

    add(page("Phase 3 · Output", "How good is the score?",
             col(tbl(["Check", "AUC"],
                     [["<b>Reaction, holdout 2018–2025</b>", "<b>0.611</b>"],
                      ["Reaction, validation 2012–2017", "0.640"],
                      ["Same model on old VADER features", "0.588"],
                      ["XGBoost instead", "0.618"],
                      ["Fixed split 2022–2025", "0.611"],
                      ["GOOD / BAD (3-month)", "0.537"]], hl={0}),
                 '<div class="box bl">AUC = chance a random up-reacting call is ranked above a random down-reacting one. 0.5 = coin flip.</div>'), 3),
        "How good is the classifier? On the holdout years 2018 to 2025, its AUC for the immediate reaction is 0.611. "
        "AUC is the probability that the model ranks a randomly chosen up-reacting call above a randomly chosen down-reacting call. A coin flip gives 0.5. "
        "On the validation years, it was 0.640. "
        "The same model built on the old VADER features reached only 0.588. "
        "And a fixed train and test split gave the same 0.611, so the result does not depend on how we split the data. "
        "For stock returns, 0.61 is a real and useful signal. A value above 0.7 would actually make us suspicious of a leak. "
        "The output of this phase is predictions finbert dot csv, with one out-of-sample score for every call.")

    # ---------- phase 4
    add(road_page(4, "Market-reaction test", "Input: scores and returns. Output: does the market under-react?"),
        "Checkpoint four: the market-reaction test. This is the heart of the project.")

    add(page("Phase 4 · The hypothesis", "Efficient, or slow?",
             col('<div class="box">If markets were <b>perfectly efficient</b>, everything the call revealed is priced in by day +1. The score would predict the immediate reaction, and <b>nothing after</b>.</div>',
                 '<div class="box g">If the score <b>also</b> predicts days +2 → +60, the market <b>under-reacted</b>: it took weeks to fully believe what the call was saying.</div>'), 4),
        "If markets were perfectly efficient, everything the call revealed would be priced in by day plus 1. "
        "The score would then predict the immediate reaction, but nothing afterwards. "
        "If the score also predicts days plus 2 to plus 60, then the market under-reacted. It took weeks to fully believe what the call was saying.")

    add(page("Phase 4 · Procedure", "Quintiles, gaps and honest t-tests",
             col(ul(["Each quarter: rank calls by score → <b>5 equal groups</b> (quintiles).",
                     "<b>Why within the quarter?</b> Compares calls made under the same market conditions.",
                     "Gap = top quintile − bottom quintile, for each of <b>30 holdout quarters</b>.",
                     "t-test with <b>Newey–West</b> errors: allows neighbouring quarters to be related.",
                     "p-value below 0.05 = unlikely to be luck."]),
                 '<div class="f">drift gap = avg drift(top 20%) − avg drift(bottom 20%)</div>'), 4),
        "Here is the method. Within each calendar quarter, we rank all calls by score and split them into five equal groups, called quintiles. "
        "Ranking within the quarter matters, because it compares each call only with calls made at the same time, under the same market conditions. "
        "We average the abnormal returns of each quintile, and compute the gap between the top quintile and the bottom quintile, separately for each of the 30 holdout quarters. "
        "Then a t-test asks whether the average gap is really above zero. "
        "We use Newey West standard errors, which allow neighbouring quarters to be related, so that we do not overstate our confidence. "
        "A p-value tells us how likely a gap this big would be if there were truly no effect. Below 0.05 is the usual bar.")

    q = [("Q1 lowest", -1.94, -0.53), ("Q2", -0.46, -0.02), ("Q3", 0.33, 0.25), ("Q4", 1.07, 0.24), ("Q5 highest", 2.00, 1.10)]
    add(page("Phase 4 · Result", "Holdout 2018–2025, by score quintile",
             '<div class="col"><div class="sub">Immediate · day −1 → +1 · gap +3.94% (t 13.2)</div>'
             + bars([(a, b) for a, b, _ in q], 2.2, "{:+.2f}%") + "</div>"
             '<div class="col"><div class="sub">Drift · day +2 → +60 · gap +1.62% (p 0.002)</div>'
             + bars([(a, c) for a, _, c in q], 1.25, "{:+.2f}%") + "</div>", 4),
        "Here are the holdout results. In the immediate window, the lowest quintile lost 1.94 percent relative to the market, and the highest gained 2 percent. "
        "That is a gap of 3.94 percent, with a t statistic of 13. The market clearly reacts to the language. "
        "Now look at the drift window, after the reaction is over. "
        "The lowest quintile keeps drifting down, minus 0.53 percent, while the highest keeps drifting up, plus 1.10 percent. "
        "The average quarterly gap is plus 1.62 percent, with a p-value of 0.002. That is the slow reaction.")

    add(page("Phase 4 · Robustness", "Is it really the words?",
             col(tbl(["Test (holdout)", "Drift gap", "p"],
                     [["Score only", '<span class="up">+1.62%</span>', "0.002"],
                      ["+ controls (reaction, pre-return, volatility)", '<span class="up">+1.31%</span>', "0.013"],
                      ["Placebo: model with <b>no language</b>", '<span class="dn">−0.36%</span>', "0.53"],
                      ["VADER score instead", "+1.40%", "0.059"],
                      ["Validation 2012–2017", "+0.85%", "0.031"]], hl={0, 1}),
                 ul(["<b>29%</b> of the total move came after the reaction window.",
                     "Gap positive in <b>22 of 30</b> quarters. SBUX sat in the bottom quintile of Q1 2024."])), 4),
        "Could this just be something else in disguise? "
        "We ran a regression of drift on the score's rank, controlling for the immediate reaction, the pre-call return and the pre-call volatility, with quarter fixed effects and errors clustered by quarter. "
        "The gap shrinks only a little, to plus 1.31 percent, with a p-value of 0.013. "
        "As a placebo, we ran the same test on a model with no language at all, only market inputs. Its drift gap is minus 0.36 percent, which is not significant. "
        "So the slow drift comes from the words. "
        "Overall, about 29 percent of the total top-minus-bottom move came after the reaction window, and the gap was positive in 22 of the 30 quarters. "
        "And our Starbucks call? Its score put it in the bottom quintile for the first quarter of 2024, and its minus 8.3 percent drift is one of the data points in that bottom group.")

    # ---------- phase 5
    add(road_page(5, "Trading backtest", "Input: scores and daily prices. Output: returns after costs."),
        "Checkpoint five: the trading backtest. Can we actually trade it?")

    add(page("Phase 5 · Procedure", "Rules a real trader could follow",
             col(flow(["Call happens", "Score vs past 365 days", "!Top 20% → long", "-Bottom 20% → short", "Enter day +2", "Exit day +60"]),
                 '<div style="height:10px"></div>',
                 ul(["Cut-offs from the <b>past year's</b> scores only (≥ 200 calls): no look-ahead.",
                     "Enter on <b>day +2</b>: the first reaction is already gone; we only chase the drift.",
                     "Exit on <b>day +60</b>, the end of the drift window."])), 5),
        "Phase five asks the practical question: can we trade it? The rules are built so a real trader could follow them on the day. "
        "When a call happens, its score is compared with the scores of all calls in the past 365 days, requiring at least 200 past calls. "
        "If it is in the top 20 percent, we buy, going long. If it is in the bottom 20 percent, we sell short. "
        "We use only the past year's scores for the cut-offs, because using the whole sample's distribution would be look-ahead. "
        "We enter on day plus 2, because the immediate reaction is already gone and we are only trying to capture the drift. And we exit on day plus 60.")

    add(page("Phase 5 · Procedure", "Portfolio and costs",
             col(ul(["Each day, the money is split <b>equally</b> across open longs, and the same for shorts.",
                     "An empty side's money sits in the S&amp;P 500.",
                     "<b>10 bps</b> (0.10%) on every buy and sell · <b>0.5%/yr</b> to borrow shorted stock."]),
                 '<div class="f">Strategy A:  ₹100 long top 20%  +  ₹100 short bottom 20%\n'
                 'daily return = long leg − short leg − costs     (market-neutral)</div>'), 5),
        "Each day, the money on the long side is split equally across all open long positions, and the same for the short side. "
        "If a side has no positions, that money sits in the S and P 500. "
        "Costs are charged on every trade: 10 basis points, that is 0.1 percent, every time we buy or sell, plus 0.5 percent a year to borrow the stocks we short. "
        "In strategy A, the main test, we put 100 rupees long and 100 rupees short, so the market's ups and downs roughly cancel out. "
        "This is called market neutral, and it means any profit must come from the signal, not from the market rising.")

    add(page("Phase 5 · Measures", "How we judge a strategy",
             col(tbl(["Measure", "Meaning"],
                     [["Annual return", "growth per year, after costs"],
                      ["Sharpe ratio", "return ÷ volatility: return per unit of risk"],
                      ["Max drawdown", "worst fall from a previous peak"],
                      ["Beta", "how much it moves with the market"],
                      ["Alpha", "return left after removing beta × market"]], num_from=9),
                 '<div class="box a">Sharpe guide: &lt; 0.5 weak · 0.5–1 decent · &gt; 1 very good · &gt; 2 in a backtest usually means a mistake.</div>'), 5),
        "We judge strategies with a few measures. "
        "Annual return. "
        "The Sharpe ratio, which is annual return divided by annual volatility: return per unit of risk. Below 0.5 is weak, 0.5 to 1 is decent, above 1 is very good, and above 2 in a backtest usually means a mistake. "
        "Maximum drawdown, the worst fall from a previous peak. "
        "Beta, how much the strategy moves with the market. "
        "And alpha, the return left after removing beta times the market's return. It is the part the market cannot explain.")

    add(page("Phase 5 · Example", "The Starbucks trade",
             col(flow(["30 Jan 2024", "score 0.317", "bottom 2% of past year", "-SHORT", "day +2 → +60"]),
                 '<div style="height:10px"></div>',
                 ul(["Starbucks then lagged the market by <b class='dn'>8.3%</b>, so the short gained.",
                     "Minus 0.1% on entry, 0.1% on exit, and a little borrow fee.",
                     "One trade proves nothing; many trades lose. It works because, over <b>thousands</b> of trades, the average tilts our way."])), 5),
        "For Starbucks, a score of 0.317 placed it in the bottom 2 percent of the past year's calls. "
        "So the strategy shorted it on day plus 2, and closed the position on day plus 60. "
        "Starbucks lagged the market by 8.3 percent, so the short gained, minus 0.1 percent cost on entry and on exit, and a small borrowing fee. "
        "One trade proves nothing, though, and many trades lose. "
        "The strategy works only because, over thousands of trades, the average is tilted slightly in our favour.")

    add(page("Phase 5 · Output", "Strategy A, holdout, after costs",
             col('<div class="kpis">'
                 '<div class="kpi"><div class="v up">+6.9%</div><div class="l">per year</div></div>'
                 '<div class="kpi"><div class="v">0.74</div><div class="l">Sharpe (p 0.032)</div></div>'
                 '<div class="kpi"><div class="v dn">−14.1%</div><div class="l">worst fall</div></div>'
                 '<div class="kpi"><div class="v">₹161</div><div class="l">from ₹100</div></div></div>',
                 ul(["Alpha <b>+5.7%/yr</b>, independent of the market.",
                     "Break-even cost <b>46 bps</b>: costs could be 4–5× higher before the profit disappears.",
                     "The no-language and VADER scores did worse in the same backtest."])), 5),
        "Strategy A, on the holdout, after all costs, returned 6.9 percent a year, with a Sharpe ratio of 0.74 and a p-value of 0.032. "
        "Its worst fall was only minus 14.1 percent, and its alpha was 5.7 percent a year. 100 rupees became 161. "
        "The break-even cost is 46 basis points per trade, so costs could be four to five times higher than we assumed before the profit disappears. "
        "The same backtest using the no-language score, or the old VADER score, did worse.")

    # ---------- strategies
    add(road_page(7, "Returns &amp; strategies", "Same scores, seven ways to place the money."),
        "And now, the finish line: returns and strategies.")

    add(page("Strategies · Design", "Seven ways to place the money", col(
        tbl(["", "Long", "Short", "Split", "Idea"],
            [["<b>A</b>", "top 20%", "bottom 20%", "1 : 1", "market-neutral, pure test"],
             ["<b>B</b>", "top 20%", "bottom 20%", "2 : 1", "tilted long, same picks"],
             ["<b>C</b>", "top 20%", "bottom 10%", "2 : 1", "short only the worst"],
             ["<b>D</b>", "top 15%", "bottom 15%", "2 : 1", "<b>our choice</b>: pickier, still diversified"],
             ["<b>E</b>", "top 15%", "—", "long only", "no shorting, idle cash in S&amp;P"],
             ["<b>F</b>", "top 5%", "bottom 5%", "2 : 1", "very picky"],
             ["<b>G</b>", "top 10%", "bottom 5%", "2 : 1", "between D and F"]], num_from=9)), 7, final=True),
        "Next, we kept the same scores, and changed only how the money is placed. "
        "Strategy B tilts two to one: 100 rupees long and 50 rupees short, on the same top and bottom 20 percent. "
        "C is long the top 20 percent, and short only the bottom 10 percent. "
        "D, our chosen strategy, uses 15 percent on both sides: 100 rupees long the top 15 percent, and 50 rupees short the bottom 15 percent. "
        "E is long only: the top 15 percent, with idle money in the S and P 500. "
        "F uses 5 and 5 percent, and G uses 10 and 5 percent, both with the two to one split.")

    add(page("Strategies · Results", "Holdout 2018–2025, after costs", col(
        tbl(["Strategy", "Return/yr", "Sharpe", "Worst fall", "Alpha (p)", "₹100 →"],
            [["A · 1:1 20/20", "+6.9%", "0.74", "−14.1%", "+5.7% (0.076)", "161"],
             ["B · 2:1 20/20", "+12.9%", "0.95", "−19.3%", "+5.6% (0.034)", "241"],
             ["C · 2:1 20/10", "+13.4%", "0.97", "−21.0%", "+6.2% (0.027)", "250"],
             ["<b>D · 2:1 15/15</b>", "+13.3%", "0.93", "−20.2%", "+6.0% (0.045)", "246"],
             ["E · long-only 15%", "+18.1%", "0.81", "−37.5%", "+4.9% (0.10)", "315"],
             ["F · 2:1 5/5", "+13.1%", "0.76", "−27.8%", "+5.9% (0.18)", "234"],
             ["G · 2:1 10/5", "+13.3%", "0.84", "−22.6%", "+5.9% (0.13)", "242"],
             ["S&amp;P 500", "+12.8%", "0.63", "−33.9%", "—", "220"]], hl={3}, muted={7})), 7, final=True),
        "Here are the results, all after costs, on the holdout. "
        "Our chosen strategy, D, returns 13.3 percent a year, with a Sharpe of 0.93, a worst fall of minus 20.2 percent, and an alpha of 6 percent with a p-value of 0.045. "
        "B and C are very similar, at 12.9 and 13.4 percent, also with significant alphas. "
        "E, long only, has the highest raw return, 18.1 percent, but its worst fall is minus 37.5 percent, deeper than the market's. "
        "F and G, the pickier versions, return about 13 percent, but with deeper falls, and alphas that are no longer significant. "
        "The S and P 500 itself returned 12.8 percent, with a Sharpe of 0.63, and a worst fall of minus 33.9 percent.")

    svg, legend = equity_svg()
    add(page("Strategies · Growth of ₹100", "January 2018 → 2025, after costs",
             f'<div class="col" style="gap:10px">{legend}{svg}</div>', 7, final=True),
        "This chart shows how 100 rupees grew from January 2018. Notice the 2020 crash, and the 2022 bear market. "
        "The market-neutral strategy A, in green, barely notices them. "
        "Our chosen strategy D, in blue, ends at 246 rupees, above the S and P 500 at about 220, but with much shallower dips. "
        "The long-only strategy E, in orange, ends highest at 315, but falls hard with the market.")

    add(page("Strategies · Lessons", "What the variants taught us",
             col('<div class="box bl"><b>1. Tilting 2:1</b> lifts return from ~7% to ~13%, mostly via market exposure. Alpha stays near <b>6%/yr</b>: the signal is robust to sizing.</div>',
                 '<div class="box a"><b>2. Too picky doesn\'t help.</b> Top 5–10% means fewer stocks, noisier returns, deeper falls.</div>',
                 '<div class="box g"><b>3. Even long-only, the score adds value:</b> +4.8%/yr over buying every call (p 0.044).</div>',
                 '<div class="box"><b>Our choice: D</b> (+13.3%/yr, alpha +6.0%, ~27 stocks per side). <b>Check: B</b> (pre-set 20% cut-offs) gives almost the same. <b>Main scientific result: A</b>.</div>'), 7, final=True),
        "There are three lessons from the variants. "
        "One: tilting two to one raises the return from about 7 to about 13 percent, mostly by adding market exposure, while alpha stays near 6 percent a year. The signal is robust to how we size the bets. "
        "Two: being too picky does not help. With only the top 5 or 10 percent, there are fewer stocks, so returns get noisier. "
        "Three: even long only, the score adds value. Compared with simply buying every call, it adds 4.8 percent a year, with a p-value of 0.044. "
        "Our chosen strategy is D. It earns 13.3 percent a year with a significant alpha of 6 percent, and with 15 percent on both sides it trades only the stronger signals while still holding about 27 stocks per side. "
        "To be honest, the 15 percent cut-off was picked after seeing earlier results, so D is slightly flattered. "
        "But B, which keeps the pre-set 20 percent cut-offs, gives almost the same result, so D does not rest on a lucky cut-off. "
        "A remains the main scientific result, because its profit cannot come from the market rising.")

    add(page("Verdict", "Did we beat the market?", col(
        tbl(["Question", "Answer"],
            [["Higher raw return than the S&amp;P?", "B–G ≈ tie (+13% vs +12.8%) · E higher · A lower but market-free"],
             ["Better return per unit of risk?", '<b class="up">Yes</b>: Sharpe 0.74–0.97 vs 0.63'],
             ["Smaller crashes?", '<b class="up">Yes</b> for A–D and G: −14% to −23% vs −34%'],
             ["Return the market can't explain?", '<b class="up">Yes</b>: alpha ≈ 6%/yr, significant for B, C, D']], num_from=9)), 7, final=True),
        "So, did we beat the market? "
        "On raw return, the tilted strategies roughly tie the S and P 500, and long-only beats it. "
        "On risk-adjusted terms, yes, clearly. Every strategy has a higher Sharpe ratio than the market. "
        "Strategies A to D, and G, have much smaller crashes. "
        "And there is an alpha of about 6 percent a year, which is statistically significant for B, C and D.")

    add(page("Verdict", "A good finding, with honest limits",
             '<div class="col"><div class="sub">Why it holds up</div>'
             + ul(["Out-of-sample, every score", "Survives costs (break-even 46 bps)", "Survives controls",
                   "Disappears without language", "FinBERT beats VADER"]) + "</div>"
             '<div class="col"><div class="sub">Limits</div>'
             + ul(["8 test years, 198 large US stocks", "Assumed costs, no taxes", "Several variants tested: best is flattered",
                   "FinBERT used as is, not retrained", "Sharpe ≈ 1: good, not extraordinary"]) + "</div>", 7, final=True),
        "Is it a good finding? As research, yes. "
        "The signal is out of sample, it survives costs, it survives controls, it disappears without language, and FinBERT beats VADER. "
        "That is consistent with published work showing that markets under-react to soft information. "
        "But we must be honest about the limits. The test covers eight years and 198 large US stocks. "
        "Costs are assumed, not real trades, and there are no taxes. "
        "We tested several strategy variants, so the best one is slightly flattered. "
        "And FinBERT was used as is, not retrained on earnings calls. "
        "A Sharpe ratio near one is good, but not extraordinary. This is evidence, not a money machine.")

    add(page("Game theory", "Mapping the results to the game", col(
        tbl(["Game piece", "In our project"],
            [["Private type", "firm truly good or bad (seen only later)"],
             ["Cheap talk", "the call: free to send, can be spun"],
             ["Hard-to-fake signal", "live Q&amp;A and <b>analyst tone</b>: the biggest weight"],
             ["Bayesian receiver", "the classifier: score ≈ P(good | words)"],
             ["Receiver strategy", "threshold rules A–G: act when belief crosses a cut-off"],
             ["Slow updating", "the drift gap: the market updates too little at first"]], num_from=9),
        '<div class="box a">An <b>interpretation</b> using signalling and cheap-talk theory, not a formal game-theoretic estimation.</div>'), 7, final=True),
        "Back to game theory. The firm's type is private. The call is cheap talk: it is free to send, so managers can spin it. "
        "But some parts are harder to fake: the live answers, and especially the analysts' reaction. That is exactly where our model puts the most weight. "
        "The classifier acts as a Bayesian receiver. Its score is a belief, roughly the probability that the firm is good, given its words. "
        "Strategies A to G are threshold strategies for the receiver: act when the belief crosses a cut-off, and size the bet according to risk appetite. "
        "The drift gap shows the real market updating too little at first, so a sharper receiver profits. "
        "This is an interpretation of our results using signalling and cheap-talk theory, not a formal game-theoretic estimation.")

    # ---------- phase 6 + close
    add(road_page(6, "Dashboard &amp; report", "Input: committed result files. Output: an explorable dashboard."),
        "Finally, checkpoint six: the dashboard, where all of this can be explored.")

    add(page("Phase 6", "Everything in one place", col(
        '<div class="f">pip install -r requirements.txt\nstreamlit run phase6_dashboard.py</div>',
        tbl(["Page", "Shows"],
            [["Overview", "headline numbers, pipeline, growth of ₹100"],
             ["Call explorer", "any call (e.g. SBUX 2024-01-30): score, signal, price path"],
             ["Language · Classifier", "every feature; AUC by year; returns by decile"],
             ["Market reaction · Strategies", "quintiles, drift per quarter; A–G vs the S&amp;P"],
             ["Method", "rules and limitations"]], num_from=9)), 6),
        "Phase six brings everything together in a Streamlit dashboard. It reads only files committed in the repository, so anyone can run it with streamlit run dashboard dot py. "
        "It has seven pages: an overview; a call explorer, where you can open any call, like our Starbucks call, and see its score, signal and price path; "
        "a language page for every feature; the classifier's accuracy over time; the market reaction tests; all strategies; and the method and its limits. "
        "Alongside it, we wrote a strategy report, and the interactive journey page.")

    add(page("Summary", "The whole project in one breath", col(
        flow(["12,789 calls", "FinBERT features", "!score · AUC 0.611", "!drift gap +1.62%", "!alpha ≈ 6%/yr"]),
        '<div style="height:16px"></div>',
        ul(["Language predicts the market's reaction to earnings calls.",
            "The market reacts <b>too little at first</b>: about 29% of the move comes later.",
            "A simple, costed strategy earns ≈ 6%/yr of alpha, with a better Sharpe and smaller crashes than the S&amp;P 500."])), 7, final=True),
        "To finish, here is the whole pipeline in one breath. "
        "12,789 earnings calls were turned into FinBERT language features. "
        "A walk-forward logistic regression turned those into one score per call, with an AUC of 0.611 on unseen years. "
        "Calls with high scores kept beating calls with low scores by 1.62 percent over the next three months, so the market under-reacts to the language of earnings calls. "
        "And a simple strategy that trades on this earns about 6 percent a year of alpha after costs, with a better Sharpe ratio and smaller crashes than the S and P 500.")

    add(page("Thank you", "Decoding the <em>Game</em>",
             col('<p class="t" style="font-size:40px">Kartik Mantri (24UCS246) · Velu Kala (24DCS012)</p>',
                 '<p class="t mu">Supervisor: Asst. Prof. Girish Sharma · The LNM Institute of Information Technology</p>',
                 '<p class="t mu" style="margin-top:30px">Code and results: github.com/KartikMantri/Mini_Project26</p>')),
        "Thank you for watching. Decoding the Game, by Kartik Mantri and Velu Kala, at the LNM Institute of Information Technology.")
    return S


def sentences(text):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]


# ----------------------------------------------------------------------------- helpers
def run(cmd, **kw):
    r = subprocess.run(cmd, capture_output=True, text=True, **kw)
    if r.returncode != 0:
        sys.exit(f"command failed: {' '.join(map(str, cmd[:6]))} ...\n{r.stderr[-2000:]}")
    return r


def find_chrome():
    for c in CHROME_CANDIDATES:
        if Path(c).exists() or shutil.which(c):
            return c
    sys.exit("Chrome not found")


def key(text):
    return hashlib.sha1(f"{VOICE}|{RATE}|{text}".encode()).hexdigest()[:16]


# ----------------------------------------------------------------------------- stages
def stage_slides(S, only=None):
    d = BUILD / "slides"
    d.mkdir(parents=True, exist_ok=True)
    chrome = find_chrome()
    for i, (h, _) in enumerate(S):
        if only is not None and i not in only:
            continue
        hp, png = d / f"s{i:02d}.html", d / f"s{i:02d}.png"
        hp.write_text(h, encoding="utf-8")
        run([chrome, "--headless=new", "--disable-gpu", "--hide-scrollbars", "--force-device-scale-factor=1",
             "--window-size=1920,1080", "--virtual-time-budget=4000", f"--screenshot={png}", hp.as_uri()])
        print(f"slide {i:02d} ok")


async def _tts_all(texts):
    import edge_tts
    d = BUILD / "tts"
    d.mkdir(parents=True, exist_ok=True)
    sem = asyncio.Semaphore(4)

    async def one(t):
        mp3, wav = d / f"{key(t)}.mp3", d / f"{key(t)}.wav"
        if wav.exists():
            return
        async with sem:
            for attempt in range(4):
                try:
                    await edge_tts.Communicate(t, VOICE, rate=RATE).save(str(mp3))
                    break
                except Exception as e:  # network hiccup: retry
                    if attempt == 3:
                        raise
                    print("  retry:", type(e).__name__, e)
                    await asyncio.sleep(2 + attempt * 2)
        run(["ffmpeg", "-y", "-v", "error", "-i", str(mp3), "-ac", "1", "-ar", str(SR), "-c:a", "pcm_s16le", str(wav)])

    todo = sorted(set(texts))
    for i in range(0, len(todo), 20):
        await asyncio.gather(*(one(t) for t in todo[i:i + 20]))
        print(f"voice {min(i + 20, len(todo))}/{len(todo)}")


def stage_audio(S):
    try:
        import edge_tts  # noqa: F401
    except ImportError:
        sys.exit("edge-tts missing: run  python -m pip install edge-tts")
    asyncio.run(_tts_all([s for _, say in S for s in sentences(say)]))


def read_wav(p):
    with wave.open(str(p)) as w:
        assert w.getframerate() == SR and w.getnchannels() == 1
        return w.readframes(w.getnframes())


def srt_time(t):
    ms = int(round(t * 1000))
    return f"{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d},{ms % 1000:03d}"


def chunks(sentence, limit=78):
    words, out, cur = sentence.split(), [], ""
    for w in words:
        if cur and len(cur) + 1 + len(w) > limit:
            out.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    if cur:
        out.append(cur)
    return out


def stage_video(S):
    slides, tts, segs = BUILD / "slides", BUILD / "tts", BUILD / "segs"
    segs.mkdir(parents=True, exist_ok=True)
    silence = lambda sec: b"\x00\x00" * int(round(sec * SR))
    audio, subs, t0, seglist = bytearray(), [], 0.0, []
    for i, (_, say) in enumerate(S):
        pcm = bytearray(silence(LEAD))
        t = t0 + LEAD
        sents = sentences(say)
        for j, s in enumerate(sents):
            w = tts / f"{key(s)}.wav"
            if not w.exists():
                sys.exit(f"missing voice for scene {i}: run --stage audio")
            data = read_wav(w)
            dur = len(data) / 2 / SR
            parts = chunks(s)
            tot = sum(len(p) for p in parts)
            ts = t
            for p in parts:
                pd_ = dur * len(p) / tot
                subs.append((ts, ts + pd_, p))
                ts += pd_
            pcm += data
            t += dur
            if j < len(sents) - 1:
                pcm += silence(GAP)
                t += GAP
        pcm += silence(TAIL)
        frames = math.ceil(len(pcm) / 2 / SR * FPS)
        pcm += silence(frames / FPS - len(pcm) / 2 / SR)          # pad to a whole frame
        audio += pcm
        dur = frames / FPS
        seg = segs / f"seg{i:02d}.mp4"
        fo = max(dur - 0.35, 0)
        run(["ffmpeg", "-y", "-v", "error", "-loop", "1", "-framerate", str(FPS), "-i", str(slides / f"s{i:02d}.png"),
             "-vf", f"scale=1920:1080,format=yuv420p,fade=in:st=0:d=0.35,fade=out:st={fo:.3f}:d=0.35",
             "-frames:v", str(frames), "-c:v", "libx264", "-preset", "veryfast", "-tune", "stillimage", "-crf", "16",
             "-r", str(FPS), str(seg)])
        seglist.append(seg)
        t0 += dur
        print(f"scene {i:02d}: {dur:6.1f}s   total {t0/60:5.1f} min")

    wav_all = BUILD / "narration.wav"
    with wave.open(str(wav_all), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR); w.writeframes(bytes(audio))
    (BUILD / "segs.txt").write_text("".join(f"file '{p.as_posix()}'\n" for p in seglist), encoding="utf-8")
    run(["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", str(BUILD / "segs.txt"), "-c", "copy",
         str(BUILD / "video_only.mp4")])
    srt = "".join(f"{k+1}\n{srt_time(a)} --> {srt_time(b)}\n{txt}\n\n" for k, (a, b, txt) in enumerate(subs))
    (BUILD / "subs.srt").write_text(srt, encoding="utf-8")
    SRT_OUT.write_text(srt, encoding="utf-8")
    style = ("FontName=Segoe UI,FontSize=13,PrimaryColour=&H00FFFFFF,OutlineColour=&H90000000,"
             "BorderStyle=3,Outline=6,Shadow=0,MarginV=14,Alignment=2")
    run(["ffmpeg", "-y", "-v", "error", "-i", "video_only.mp4", "-i", "narration.wav",
         "-vf", f"subtitles=subs.srt:force_style='{style}'", "-c:v", "libx264", "-preset", "medium", "-crf", "20",
         "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "160k", "-ar", "48000", "-shortest", "-movflags", "+faststart",
         str(FINAL)], cwd=BUILD)
    print(f"\nwrote {FINAL}  ({t0/60:.1f} min)\nsubtitles {SRT_OUT}")


def main():
    global VOICE, RATE
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["all", "slides", "audio", "video"], default="all")
    ap.add_argument("--voice", default=VOICE, help="edge-tts voice, e.g. en-IN-PrabhatNeural")
    ap.add_argument("--rate", default=RATE, help='speaking rate, e.g. "-4%%" or "+5%%"')
    ap.add_argument("--only", default="", help="comma-separated scene numbers (slides stage only)")
    a = ap.parse_args()
    VOICE, RATE = a.voice, a.rate
    S = scenes()
    words = sum(len(say.split()) for _, say in S)
    print(f"{len(S)} scenes, {words} words of narration (~{words/150:.0f} min)")
    only = {int(x) for x in a.only.split(",") if x.strip()} or None
    if a.stage in ("all", "slides"):
        stage_slides(S, only)
    if a.stage in ("all", "audio"):
        stage_audio(S)
    if a.stage in ("all", "video"):
        stage_video(S)


if __name__ == "__main__":
    main()
