// Builds presentation/Decoding_the_Game.pptx
// Run:  node presentation/make_presentation.js   (needs: npm install pptxgenjs)
// Light, high-contrast palette chosen for projection in a lit room.

const fs = require("fs");
const path = require("path");
const pptxgen = require("pptxgenjs");
const { applyTheme } = require(process.env.APPLY_THEME || "./apply_theme.js");

const OUT = path.join(__dirname, "Decoding_the_Game.pptx");

// ------------------------------------------------------------------ theme
const HEX = {
  ink: "15202E",      // near-black navy: all body text
  navy: "1B3A63",     // titles, main shapes
  white: "FFFFFF",
  tint: "EDF2F8",     // card background
  green: "0B7A53",    // up / long / Strategy A
  red: "B42318",      // down / short
  blue: "1D5FB8",     // Strategy D (our choice)
  amber: "C25E00",    // Strategy E / highlight
  gray: "5B6573",     // S&P 500, captions
  purple: "6B3FA0",
  grid: "D5DCE6",
};
const THEME = {
  name: "Decoding the Game",
  headFontFace: "Calibri",
  bodyFontFace: "Calibri",
  colors: {
    dk1: HEX.ink, lt1: HEX.white, dk2: HEX.navy, lt2: HEX.tint,
    accent1: HEX.green, accent2: HEX.red, accent3: HEX.blue, accent4: HEX.amber,
    accent5: HEX.gray, accent6: HEX.purple, hlink: HEX.blue, folHlink: HEX.purple,
  },
};

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.33 x 7.5
pres.title = "Decoding the Game";
pres.author = "Kartik Mantri, Velu Kala";
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
const C = pres.SchemeColor;
const COL = { ink: C.text1, navy: C.text2, white: C.background1, tint: C.background2,
  green: C.accent1, red: C.accent2, blue: C.accent3, amber: C.accent4, gray: C.accent5, purple: C.accent6 };

const W = 13.333, MX = 0.6, CW = W - 2 * MX;   // content width 12.13
const TOP = 1.72;                               // content starts here

// ------------------------------------------------------------------ layouts
const footer = [
  { text: { text: "Decoding the Game  ·  Kartik Mantri & Velu Kala  ·  LNMIIT",
    options: { x: MX, y: 7.02, w: 8, h: 0.3, fontSize: 11, color: COL.gray, margin: 0 } } },
];
pres.defineSlideMaster({
  title: "CONTENT",
  background: { color: COL.white },
  objects: [
    ...footer,
    { placeholder: { options: { name: "kicker", type: "body", x: MX, y: 0.32, w: 8.2, h: 0.36,
      fontSize: 15, bold: true, color: COL.green, margin: 0, valign: "middle" }, text: "" } },
    { placeholder: { options: { name: "title", type: "title", x: MX, y: 0.7, w: CW, h: 0.85,
      fontSize: 34, bold: true, color: COL.navy, margin: 0, valign: "middle", align: "left" }, text: "" } },
  ],
  slideNumber: { x: 12.1, y: 7.0, w: 0.63, h: 0.3, fontSize: 11, color: COL.gray, align: "right" },
});
pres.defineSlideMaster({
  title: "DIVIDER",
  background: { color: COL.tint },
  objects: [
    ...footer,
    { placeholder: { options: { name: "kicker", type: "body", x: MX, y: 2.35, w: 8, h: 0.5,
      fontSize: 22, bold: true, color: COL.green, margin: 0 }, text: "" } },
    { placeholder: { options: { name: "title", type: "title", x: MX, y: 2.85, w: 8.2, h: 1.3,
      fontSize: 48, bold: true, color: COL.navy, margin: 0, valign: "top", align: "left" }, text: "" } },
    { placeholder: { options: { name: "body", type: "body", x: MX, y: 4.25, w: 7.6, h: 1.6,
      fontSize: 20, color: COL.ink, margin: 0, valign: "top" }, text: "" } },
  ],
  slideNumber: { x: 12.1, y: 7.0, w: 0.63, h: 0.3, fontSize: 11, color: COL.gray, align: "right" },
});
pres.defineSlideMaster({
  title: "TITLE",
  background: { color: COL.white },
  objects: [
    { placeholder: { options: { name: "kicker", type: "body", x: MX, y: 1.0, w: 10, h: 0.45,
      fontSize: 18, bold: true, color: COL.green, margin: 0 }, text: "" } },
    { placeholder: { options: { name: "title", type: "title", x: MX, y: 1.45, w: 11, h: 1.3,
      fontSize: 60, bold: true, color: COL.navy, margin: 0, align: "left" }, text: "" } },
    { placeholder: { options: { name: "body", type: "body", x: MX, y: 2.8, w: 10.5, h: 1.0,
      fontSize: 22, color: COL.ink, margin: 0, valign: "top" }, text: "" } },
  ],
});

// ------------------------------------------------------------------ helpers
let section = "";
function slide(master, kicker, title, notes) {
  const s = pres.addSlide({ masterName: master, sectionTitle: section });
  if (kicker) s.addText(kicker, { placeholder: "kicker" });
  if (title) s.addText(title, { placeholder: "title" });
  if (notes) s.addNotes(notes);
  return s;
}
function newSection(t) { section = t; pres.addSection({ title: t }); }

// phase tracker: the deck's motif (7 checkpoints, current one filled)
function tracker(s, cur) {
  const r = 0.34, gap = 0.16, n = 7;
  const x0 = W - MX - n * r - (n - 1) * gap;
  for (let i = 0; i < n; i++) {
    const x = x0 + i * (r + gap);
    if (i < n - 1) s.addShape(pres.shapes.LINE, { x: x + r, y: 0.5, w: gap, h: 0,
      line: { color: i < cur ? COL.green : COL.gray, width: 1.25 }, objectName: `track-line-${i}` });
    const done = cur !== null && i < cur, now = i === cur;
    s.addText(String(i), { shape: pres.shapes.OVAL, x, y: 0.5 - r / 2, w: r, h: r,
      fill: { color: now ? COL.navy : done ? COL.green : COL.white },
      line: { color: now ? COL.navy : done ? COL.green : COL.gray, width: 1.25 },
      color: now || done ? COL.white : COL.gray, fontSize: 11, bold: true, align: "center",
      valign: "middle", margin: 0, isTextBox: true, objectName: `track-${i}` });
  }
}

function txt(s, text, o) {
  s.addText(text, Object.assign({ isTextBox: true, fontSize: 18, color: COL.ink, margin: 0, valign: "top" }, o));
}
// runs: "plain **bold** plain"
function rich(str, base = {}) {
  const parts = str.split(/(\*\*[^*]+\*\*)/).filter(Boolean);
  return parts.map((p) => p.startsWith("**")
    ? { text: p.slice(2, -2), options: Object.assign({}, base, { bold: true }) }
    : { text: p, options: Object.assign({}, base) });
}
function bullets(s, items, o = {}) {
  const size = o.fontSize || 18;
  const runs = [];
  items.forEach((it, i) => {
    const r = rich(it, { fontSize: size, color: COL.ink });
    r[0].options = Object.assign({}, r[0].options, { bullet: { indent: 18 }, paraSpaceAfter: o.space ?? 9 });
    if (i < items.length - 1) r[r.length - 1].options.breakLine = true;
    runs.push(...r);
  });
  s.addText(runs, Object.assign({ isTextBox: true, margin: 0, valign: "top" }, o, { fontSize: size }));
}
// card: tinted rounded box, bold head + body
function card(s, x, y, w, h, head, body, o = {}) {
  const color = o.color || COL.navy;
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.08,
    fill: { color: o.fill || COL.tint }, line: { color: o.fill || COL.tint }, objectName: o.name || "card" });
  const runs = [{ text: head, options: { bold: true, color, fontSize: o.headSize || 19, breakLine: !!body, paraSpaceAfter: 6 } }];
  if (body) {
    const b = Array.isArray(body) ? body : [body];
    b.forEach((line, i) => {
      const r = rich(line, { fontSize: o.size || 16, color: COL.ink });
      if (i < b.length - 1) { r[r.length - 1].options.breakLine = true; r[r.length - 1].options.paraSpaceAfter = 4; }
      runs.push(...r);
    });
  }
  s.addText(runs, { isTextBox: true, x: x + 0.2, y: y + 0.15, w: w - 0.4, h: h - 0.3, margin: 0, valign: o.valign || "top" });
}
// numbered badge circle
function badge(s, x, y, label, color, d = 0.5) {
  s.addText(label, { shape: pres.shapes.OVAL, x, y, w: d, h: d, fill: { color }, line: { color },
    color: COL.white, bold: true, fontSize: d > 0.45 ? 16 : 12, align: "center", valign: "middle", margin: 0, isTextBox: true });
}
// horizontal flow of boxes with arrows
function flow(s, steps, y, o = {}) {
  const n = steps.length, h = o.h || 1.0, ag = o.arrow || 0.38;
  const x0 = o.x ?? MX, tw = o.w ?? CW;
  const bw = (tw - (n - 1) * ag) / n;
  steps.forEach((st, i) => {
    const x = x0 + i * (bw + ag);
    const kind = st.kind || "n";
    const fill = kind === "hi" ? COL.navy : kind === "up" ? COL.green : kind === "dn" ? COL.red : COL.tint;
    const fc = kind === "n" ? COL.ink : COL.white;
    const runs = [{ text: st.t, options: { bold: true, fontSize: o.size || 16, color: fc, breakLine: !!st.s } }];
    if (st.s) runs.push({ text: st.s, options: { fontSize: (o.size || 16) - 3, color: fc } });
    s.addText(runs, { shape: pres.shapes.ROUNDED_RECTANGLE, rectRadius: 0.08, x, y, w: bw, h,
      fill: { color: fill }, line: { color: fill }, align: "center", valign: "middle", margin: 4, isTextBox: true,
      objectName: `flow-${i}` });
    if (i < n - 1) s.addShape(pres.shapes.RIGHT_ARROW, { x: x + bw + 0.07, y: y + h / 2 - 0.13, w: ag - 0.14, h: 0.26,
      fill: { color: COL.gray }, line: { color: COL.gray }, objectName: `flow-arrow-${i}` });
  });
}
// table with navy header, zebra rows
function table(s, head, rows, o = {}) {
  const size = o.fontSize || 15;
  const hl = new Set(o.hl || []), mute = new Set(o.mute || []);
  const numFrom = o.numFrom ?? 99;
  const cell = (v, ri, ci) => {
    let color = COL.ink, bold = false, fill = ri % 2 ? COL.white : COL.tint;
    if (hl.has(ri)) { fill = "DCEFE6"; bold = true; }   // hex: highlight row (light green)
    if (mute.has(ri)) color = COL.gray;
    let text = String(v);
    if (text.startsWith("+") && ci >= numFrom && !o.noSign) color = COL.green;
    if (text.startsWith("−") && ci >= numFrom && !o.noSign) color = COL.red;
    if (/^\*\*[^*]+\*\*$/.test(text)) { text = text.slice(2, -2); bold = true; }
    else if (text.includes("**")) text = rich(text, { fontSize: size, color });
    return { text, options: { fontSize: size, color, bold, fill: { color: fill },
      align: ci >= numFrom ? "right" : "left", valign: "middle", margin: [3, 8, 3, 8] } };
  };
  const data = [head.map((h, ci) => ({ text: h, options: { bold: true, fontSize: size, color: COL.white,
    fill: { color: COL.navy }, align: ci >= numFrom ? "right" : "left", valign: "middle", margin: [3, 8, 3, 8] } }))]
    .concat(rows.map((r, ri) => r.map((v, ci) => cell(v, ri, ci))));
  s.addTable(data, { x: o.x ?? MX, y: o.y ?? TOP, w: o.w ?? CW, colW: o.colW, rowH: o.rowH || 0.42,
    border: { type: "solid", pt: 0.75, color: "FFFFFF" }, objectName: o.name || "table" });
}
function stat(s, x, y, w, value, label, color) {
  txt(s, value, { x, y, w, h: 0.9, fontSize: 44, bold: true, color: color || COL.navy, valign: "bottom" });
  txt(s, label, { x, y: y + 0.92, w, h: 0.6, fontSize: 15, color: COL.gray });
}
function formula(s, text, x, y, w, h, size = 18) {
  s.addText(text, { isTextBox: true, x, y, w, h, fontFace: "Courier New", fontSize: size, color: COL.navy, bold: true,
    fill: { color: COL.tint }, margin: [6, 14, 6, 14], valign: "middle", objectName: "formula" });
}
function callout(s, text, x, y, w, h, color) {
  s.addText(rich(text, { fontSize: 17, color: COL.white }), { shape: pres.shapes.ROUNDED_RECTANGLE, rectRadius: 0.08,
    x, y, w, h, fill: { color: color || COL.navy }, line: { color: color || COL.navy }, margin: [6, 14, 6, 14],
    valign: "middle", isTextBox: true, objectName: "callout" });
}
const chartBase = () => ({
  catAxisLabelColor: HEX.ink, valAxisLabelColor: HEX.ink, catAxisLabelFontFace: "+mn-lt", valAxisLabelFontFace: "+mn-lt",
  catAxisLabelFontSize: 14, valAxisLabelFontSize: 13, dataLabelFontFace: "+mn-lt", dataLabelFontSize: 14,
  dataLabelColor: HEX.ink, titleFontFace: "+mn-lt", titleFontSize: 16, titleColor: HEX.navy,
  valGridLine: { color: HEX.grid, size: 0.75 }, catGridLine: { style: "none" },
  catAxisLineColor: HEX.gray, valAxisLineShow: false, legendFontFace: "+mn-lt", legendFontSize: 14,
});

// ================================================================== SLIDES
newSection("Opening");
{
  const s = slide("TITLE", "LNMIIT  ·  Mini project", "Decoding the Game",
    "Welcome. This is our mini project, Decoding the Game, by Kartik Mantri and Velu Kala, supervised by Assistant Professor Girish Sharma. " +
    "We ask whether the language of earnings calls reveals what managers know, whether the stock market is slow to absorb it, and whether a trading strategy can profit from that after costs.");
  s.addText("Detecting information asymmetry in earnings calls with NLP, and testing whether the stock market is slow to notice",
    { placeholder: "body" });
  const ys = 4.15;
  [["12,789", "earnings calls, 2005–2025", COL.navy], ["0.611", "classifier AUC on unseen years", COL.navy],
   ["+1.62%", "slow-reaction gap (p 0.002)", COL.green], ["+13.3%", "per year, Strategy D, after costs", COL.green]]
    .forEach(([v, l, c], i) => {
      const x = MX + i * 3.08;
      s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: ys, w: 2.85, h: 1.55, rectRadius: 0.08,
        fill: { color: COL.tint }, line: { color: COL.tint }, objectName: `kpi-${i}` });
      txt(s, v, { x: x + 0.2, y: ys + 0.12, w: 2.5, h: 0.75, fontSize: 38, bold: true, color: c, valign: "bottom" });
      txt(s, l, { x: x + 0.2, y: ys + 0.9, w: 2.5, h: 0.55, fontSize: 14, color: COL.gray });
    });
  txt(s, [{ text: "Kartik Mantri (24UCS246)  ·  Velu Kala (24DCS012)", options: { bold: true, breakLine: true } },
    { text: "Supervisor: Asst. Prof. Girish Sharma  ·  The LNM Institute of Information Technology", options: {} }],
    { x: MX, y: 6.15, w: 11, h: 0.8, fontSize: 17, color: COL.ink });
}

// ---------- the problem (2 slides)
newSection("The problem");
{
  const s = slide("CONTENT", "The problem  ·  1 of 2", "Managers know more than investors do",
    "Every quarter, large companies hold an earnings call. Managers read a prepared script, then analysts ask live questions. " +
    "Managers know far more about the business than outside investors: this is information asymmetry. " +
    "The numbers in the call, like revenue and profit, are hard information and get priced in within minutes. " +
    "The language, the tone, hesitation, how analysts push back, is soft information. It is harder to process, so it is exactly where a slow reaction could hide.");
  tracker(s, null);
  // diagram: manager -> call -> investors, with a "knowledge gap"
  const y = 1.95;
  flow(s, [{ t: "Management", s: "knows the true state of the firm", kind: "hi" },
           { t: "Earnings call", s: "script + live Q&A", kind: "n" },
           { t: "Investors", s: "see only the words and numbers", kind: "n" }], y, { h: 1.25, size: 19, w: 8.0 });
  card(s, 9.05, y, 3.68, 1.25, "Information asymmetry", "one side knows more than the other", { color: COL.red, size: 15, headSize: 17 });
  card(s, MX, 3.6, 5.9, 2.95, "Hard information", ["Revenue, profit, guidance numbers", "Easy to compare with forecasts",
    "Priced in **within minutes** of the call"], { color: COL.navy, size: 19, headSize: 21 });
  card(s, MX + 6.23, 3.6, 5.9, 2.95, "Soft information: language", ["Tone, hesitation, vague answers",
    "How analysts react and push back", "Harder to process, so this is where a **slow market reaction** could hide"],
    { color: COL.green, size: 19, headSize: 21 });
}
{
  const s = slide("CONTENT", "The problem  ·  2 of 2", "Three questions this project answers",
    "Our question has three parts. First, can a computer read the call and tell whether the firm is really doing well or badly? " +
    "Second, if it can, does the market absorb that immediately, or react slowly? Third, if it is slow, can a strategy profit after realistic costs? " +
    "Each question is answered by a phase of the project, with a number that defines success, fixed before we looked at the results.");
  tracker(s, null);
  const qs = [
    ["Q1", "Can a computer read the call?", "Tell from the words whether the firm is really doing well or badly", "Classifier AUC above 0.5", "Phases 1–3"],
    ["Q2", "Is the market slow?", "Does the price keep moving in the direction of the words for weeks after the call?", "Significant drift gap", "Phase 4"],
    ["Q3", "Can we profit from it?", "Trade the signal with real rules, real timing and trading costs", "Positive return after costs", "Phase 5"],
  ];
  qs.forEach(([q, h, b, m, p], i) => {
    const x = MX + i * 4.1, y = TOP + 0.15, w = 3.85;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h: 4.85, rectRadius: 0.08, fill: { color: COL.tint },
      line: { color: COL.tint }, objectName: `q-card-${i}` });
    badge(s, x + 0.25, y + 0.25, q, [COL.navy, COL.green, COL.blue][i], 0.7);
    txt(s, h, { x: x + 0.25, y: y + 1.1, w: w - 0.5, h: 0.8, fontSize: 21, bold: true, color: COL.navy });
    txt(s, b, { x: x + 0.25, y: y + 1.95, w: w - 0.5, h: 1.3, fontSize: 17 });
    txt(s, [{ text: "Success test", options: { bold: true, color: COL.gray, fontSize: 13, breakLine: true } },
      { text: m, options: { bold: true, color: COL.green, fontSize: 18 } }], { x: x + 0.25, y: y + 3.35, w: w - 0.5, h: 0.8 });
    txt(s, p, { x: x + 0.25, y: y + 4.2, w: w - 0.5, h: 0.4, fontSize: 14, color: COL.gray });
  });
}

// ---------- what we will do (2 slides)
newSection("Our approach");
{
  const s = slide("CONTENT", "What we will do  ·  1 of 2", "A pipeline of seven phases",
    "The project is a pipeline. Phase zero fixes the rules. Phase one collects transcripts and prices and labels every call. " +
    "Phase two turns language into numbers. Phase three trains a model that gives every call a score. " +
    "Phase four tests whether the market under-reacts to that score. Phase five trades on it, and phase six puts it all in a dashboard. " +
    "The output of each phase is the input of the next.");
  tracker(s, null);
  const ph = [
    ["0", "Setup", "universe + rules"], ["1", "Data", "12,789 labelled calls"], ["2", "Language", "18 FinBERT features"],
    ["3", "Classifier", "one score per call"], ["4", "Market test", "is the market slow?"], ["5", "Backtest", "returns after costs"],
    ["6", "Dashboard", "explore everything"],
  ];
  const n = 7, ag = 0.22, bw = (CW - (n - 1) * ag) / n, y = 2.05;
  ph.forEach(([k, t, o], i) => {
    const x = MX + i * (bw + ag);
    const c = i === 3 || i === 4 ? COL.navy : COL.tint;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w: bw, h: 2.0, rectRadius: 0.08, fill: { color: c }, line: { color: c },
      objectName: `phase-box-${i}` });
    const fc = c === COL.navy ? COL.white : COL.ink;
    badge(s, x + bw / 2 - 0.25, y + 0.18, k, i === 3 || i === 4 ? COL.green : COL.navy);
    txt(s, t, { x: x + 0.08, y: y + 0.78, w: bw - 0.16, h: 0.45, fontSize: 18, bold: true, color: fc, align: "center" });
    txt(s, o, { x: x + 0.08, y: y + 1.22, w: bw - 0.16, h: 0.7, fontSize: 14, color: fc, align: "center" });
    if (i < n - 1) s.addShape(pres.shapes.RIGHT_ARROW, { x: x + bw + 0.03, y: y + 0.9, w: ag - 0.06, h: 0.22,
      fill: { color: COL.gray }, line: { color: COL.gray }, objectName: `phase-arrow-${i}` });
  });
  txt(s, "The heart of the project: phases 3 and 4", { x: MX + 3 * (bw + ag), y: y + 2.1, w: 2 * bw + ag, h: 0.35,
    fontSize: 13, bold: true, color: COL.navy, align: "center" });
  card(s, MX, 4.75, 3.9, 1.9, "In", "Earnings-call transcripts and daily stock prices for 200 large US companies", { size: 16 });
  card(s, MX + 4.12, 4.75, 3.9, 1.9, "Inside", "NLP (FinBERT) → logistic regression score → statistical test → trading rules", { size: 16 });
  card(s, MX + 8.24, 4.75, 3.89, 1.9, "Out", "Evidence that the market under-reacts, and strategies that earn ≈ 6% a year of alpha",
    { size: 16, color: COL.green });
}
{
  const s = slide("CONTENT", "What we will do  ·  2 of 2", "The idea: a signalling game",
    "We frame the problem as a signalling game. Nature decides if the firm is truly good or bad, which only management knows. " +
    "Management sends a message, the earnings call. Talking is free, so it is cheap talk: a bad firm can try to sound good. " +
    "The receiver, the market, infers the type and acts. We build a better receiver, a model that reads the words, and check whether the real market was slow.");
  tracker(s, null);
  flow(s, [{ t: "Nature", s: "firm is good or bad" }, { t: "Sender", s: "management" }, { t: "Message", s: "the call", kind: "hi" },
           { t: "Receiver", s: "the market" }, { t: "Action", s: "buy or sell" }], 1.95, { h: 1.15, size: 19 });
  card(s, MX, 3.45, 3.9, 2.4, "Private type", "Only management knows if the firm is truly doing well. The market learns it months later.",
    { size: 16 });
  card(s, MX + 4.12, 3.45, 3.9, 2.4, "Cheap talk", "Talking is free, so a bad firm can try to sound good. Some parts (live Q&A) are harder to fake.",
    { size: 16 });
  card(s, MX + 8.24, 3.45, 3.89, 2.4, "Our receiver", "A model that reads the words and gives a belief: how likely the firm is good.",
    { size: 16, color: COL.green });
  callout(s, "Plan: build a **better receiver** → check whether the **real market was slow** → **trade** the gap after costs",
    MX, 6.05, CW, 0.65, COL.navy);
}

// ---------- phase 0
newSection("Phase 0 · Setup");
{
  const s = slide("CONTENT", "Phase 0  ·  Setup & foundation", "The universe and three rules fixed in advance",
    "Phase zero. Input: the idea and a list of stocks. We chose 200 large S&P 500 companies across all sectors, saved in tickers.csv. " +
    "Large companies have reliable prices, heavy trading and low costs, and all hold regular calls. A signal in the most-watched stocks is stronger evidence. " +
    "Before running any model we fixed three rules: no look-ahead, a locked holdout of 2018 to 2025 used only once at the end, and success defined in advance by three numbers. " +
    "Output: the universe, the rules and the plan.");
  tracker(s, 0);
  card(s, MX, TOP + 0.1, 4.2, 4.95, "Input: the universe", ["**200 S&P 500 companies**, every sector (tickers.csv)",
    "Reliable prices, heavy trading, low trading costs", "All hold regular calls with full transcripts",
    "A signal in the **most-watched** stocks is stronger evidence than in tiny ones"], { size: 16 });
  const rules = [
    ["1", "No look-ahead", "A feature may use only the call's own text, or the same company's **earlier** calls. A backtest that peeks at the future always looks brilliant, and is always fake.", COL.green],
    ["2", "Locked holdout", "All choices made on **2012–2017**. The years **2018–2025** were used once, at the end.", COL.blue],
    ["3", "Success defined first", "AUC above 0.5  ·  significant drift gap  ·  profit after costs", COL.amber],
  ];
  rules.forEach(([n, h, b, c], i) => {
    const y = TOP + 0.1 + i * 1.7, x = MX + 4.5;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w: 7.63, h: 1.5, rectRadius: 0.08, fill: { color: COL.tint },
      line: { color: COL.tint }, objectName: `rule-${i}` });
    badge(s, x + 0.25, y + 0.45, n, c, 0.6);
    txt(s, [{ text: h, options: { bold: true, color: COL.navy, fontSize: 19, breakLine: true } },
      ...rich(b, { fontSize: 16, color: COL.ink })], { x: x + 1.05, y: y + 0.15, w: 6.4, h: 1.2, valign: "middle" });
  });
  txt(s, "Output →  the universe, the rules and the plan", { x: MX, y: 6.65, w: CW, h: 0.3, fontSize: 15, bold: true, color: COL.green });
}

// ---------- phase 1
newSection("Phase 1 · Data");
{
  const s = slide("CONTENT", "Phase 1  ·  Data collection  ·  Input & procedure", "Two raw sources, and the abnormal return",
    "Transcripts come from a public Hugging Face dataset of S&P 500 earnings calls, with full text and speaker names. " +
    "Daily adjusted prices come from Yahoo Finance via the yfinance library, for every stock and for the S&P 500 index. " +
    "From prices we compute the abnormal return: the stock's return minus the S&P 500's return over the same days. " +
    "If the whole market rises 5% and the stock rises 5%, the company did nothing special. We keep only the company's own part of the move, the part its language could explain.");
  tracker(s, 1);
  card(s, MX, TOP + 0.1, 3.9, 2.25, "Transcripts", ["Hugging Face dataset", "kurry/sp500_earnings_transcripts", "Full text with speaker names"], { size: 16 });
  card(s, MX + 4.12, TOP + 0.1, 3.9, 2.25, "Stock prices", ["Yahoo Finance (yfinance)", "Daily adjusted close", "Every stock in the universe"], { size: 16 });
  card(s, MX + 8.24, TOP + 0.1, 3.89, 2.25, "The market", ["S&P 500 index (^GSPC)", "The benchmark for every return"], { size: 16 });
  formula(s, "abnormal return  =  stock return  −  S&P 500 return   (same days)", MX, 4.3, CW, 0.75, 20);
  card(s, MX, 5.25, CW, 1.4, "Why subtract the market?",
    "If the whole market rises 5% and the stock rises 5%, the company did nothing special. We keep only the part of the move that belongs to the company, which is the part its words could explain.",
    { size: 16, color: COL.green });
}
{
  const s = slide("CONTENT", "Phase 1  ·  Data collection  ·  Procedure", "Three return windows around the call (day 0)",
    "Day zero is the call's date. Windows are counted in trading days. The pre-call window, day minus 61 to minus 1, shows how the stock did before; it becomes a control input. " +
    "The immediate window, day minus 1 to plus 1, is the first reaction, and it is the model's training target. " +
    "The drift window, day plus 2 to plus 60, is roughly the next quarter. It starts at plus 2 so it never overlaps the first reaction. " +
    "Each call is labelled GOOD if its drift abnormal return is above zero, otherwise BAD. That is the firm's true type, revealed later.");
  tracker(s, 1);
  // timeline diagram
  const y = 2.35, x0 = MX + 0.2, x1 = W - MX - 0.2;
  const pos = (d) => x0 + (d + 61) / 121 * (x1 - x0);
  s.addShape(pres.shapes.LINE, { x: x0, y: y + 0.9, w: x1 - x0, h: 0, line: { color: COL.ink, width: 2 }, objectName: "timeline" });
  const seg = (a, b, label, sub, color, row) => {
    const xa = pos(a), xb = pos(b);
    s.addText([{ text: label, options: { bold: true, fontSize: 17, color: COL.white, breakLine: true } },
      { text: sub, options: { fontSize: 14, color: COL.white } }],
      { shape: pres.shapes.ROUNDED_RECTANGLE, rectRadius: 0.06, x: xa, y: y + row, w: xb - xa, h: 0.75,
        fill: { color }, line: { color }, align: "center", valign: "middle", margin: 2, isTextBox: true, objectName: `win-${label}` });
  };
  seg(-61, -1.6, "Pre-call", "day −61 → −1", COL.gray, 0);
  seg(-1.4, 1.4, "", "", COL.navy, 0);
  txt(s, "Immediate −1 → +1", { x: pos(0) - 1.1, y: y - 0.42, w: 2.2, h: 0.35, fontSize: 15, bold: true, color: COL.navy, align: "center" });
  seg(2, 60, "Drift (the slow part)", "day +2 → +60  ≈ one quarter", COL.green, 0);
  [-61, -1, 0, 1, 2, 60].forEach((d) => {
    const x = pos(d);
    s.addShape(pres.shapes.LINE, { x, y: y + 0.82, w: 0, h: 0.16, line: { color: COL.ink, width: 1.5 }, objectName: `tick-${d}` });
  });
  txt(s, "−61", { x: pos(-61) - 0.3, y: y + 1.05, w: 0.6, h: 0.3, fontSize: 14, align: "center" });
  txt(s, "day 0 = call", { x: pos(0) - 0.7, y: y + 1.05, w: 1.4, h: 0.3, fontSize: 14, bold: true, align: "center" });
  txt(s, "+60", { x: pos(60) - 0.3, y: y + 1.05, w: 0.6, h: 0.3, fontSize: 14, align: "center" });
  table(s, ["Window", "Days", "Role in the project"], [
    ["Pre-call", "−61 → −1", "how the stock did before: a control input"],
    ["Immediate", "−1 → +1", "the market's first reaction: the classifier's target"],
    ["Drift", "+2 → +60", "after the reaction: the slow part we test and trade"],
  ], { y: 3.85, colW: [2.2, 2.0, 7.93], fontSize: 16, rowH: 0.44 });
  formula(s, "label  =  GOOD  if drift abnormal return > 0,   else  BAD", MX, 5.85, CW, 0.7, 19);
}
{
  const s = slide("CONTENT", "Phase 1  ·  Data collection  ·  Decisions, example & output", "12,789 labelled calls, checked for bad data",
    "Our first version used a plus-minus 2% threshold and dropped neutral calls, keeping 10,493. We then kept all 12,789, because a trader cannot know in advance which calls end up neutral; the two labelings agree on 97.7% of calls. " +
    "We scanned every price series: Paramount had broken prices starting near 83,000, so its 77 calls were removed from every return-based test. " +
    "Our running example is Starbucks on 30 January 2024: it lagged the market by 15.3% before the call and by 8.3% in the drift window, so it is labelled BAD. " +
    "Output: calls_all.csv, 12,789 calls from 198 companies, October 2005 to May 2025.");
  tracker(s, 1);
  card(s, MX, TOP + 0.1, 5.9, 2.35, "Decision: keep every call", ["First version: ±2% threshold, neutral calls dropped (10,493 kept)",
    "Final: **all 12,789** – a trader can't know in advance which calls end up neutral", "The two labelings agree on **97.7%** of calls"], { size: 15 });
  card(s, MX, TOP + 2.65, 5.9, 2.0, "Quality check: Paramount (PARA)", ["Price series started near 83,000, no link to the market",
    "Its **77 calls removed** from every return test. No other ticker affected"], { size: 15, color: COL.red });
  // example
  const ex = MX + 6.23;
  txt(s, "Running example: Starbucks, 30 Jan 2024", { x: ex, y: TOP + 0.1, w: 5.9, h: 0.4, fontSize: 18, bold: true, color: COL.navy });
  table(s, ["Window", "Days", "Abnormal return"], [["Pre-call", "−61 → −1", "−15.3%"], ["Drift", "+2 → +60", "−8.3%"]],
    { x: ex, y: TOP + 0.6, w: 5.9, colW: [1.9, 1.9, 2.1], numFrom: 2, fontSize: 16 });
  callout(s, "Drift is below 0, so the call is labelled **BAD**", ex, TOP + 2.0, 5.9, 0.6, COL.red);
  card(s, ex, TOP + 2.85, 5.9, 1.8, "Output → data/calls_all.csv", ["**12,789 calls**, 198 companies, Oct 2005 → May 2025",
    "Columns pre_ar, imm_ar, drift_ar, label  +  prices.csv"], { size: 15, color: COL.green });
}

// ---------- phase 2
newSection("Phase 2 · Language");
{
  const s = slide("CONTENT", "Phase 2  ·  Signal extraction  ·  Procedure", "Split the call, then let FinBERT read every sentence",
    "First, each transcript is split into speaker turns. A line starting with a name and a colon starts a turn, and operator lines are ignored. " +
    "The call is split at the second operator line into the prepared script and the live Q&A. A script can be polished by lawyers; live answers are harder to manage. " +
    "After each operator turn, the first person who did not speak in the script is the analyst. " +
    "Every sentence is scored by FinBERT, a BERT model trained on financial text. It gives positive, negative and neutral probabilities, and net tone is positive minus negative. " +
    "Why not VADER? VADER is built for social media and misreads words like liability or tax, which are neutral in finance.");
  tracker(s, 2);
  flow(s, [{ t: "Transcript" }, { t: "Speaker turns", s: "Name: starts a turn" }, { t: "Prepared script", s: "can be polished", kind: "hi" },
           { t: "Live Q&A", s: "hard to manage", kind: "up" }, { t: "Analyst vs exec", s: "who is asking" }], TOP + 0.1, { h: 1.05, size: 17 });
  formula(s, "net tone  =  P(positive)  −  P(negative)          range −1 … +1", MX, 3.15, CW, 0.65, 19);
  table(s, ["Sentence (illustrative)", "P(pos)", "P(neg)", "Net tone"], [
    ["“Revenue grew to a record this quarter.”", "0.96", "0.01", "+0.95"],
    ["“We saw continued softness in traffic.”", "0.02", "0.93", "−0.91"],
  ], { y: 4.0, w: 7.4, colW: [4.1, 1.0, 1.0, 1.3], numFrom: 1, fontSize: 15 });
  card(s, MX + 7.7, 4.0, 4.43, 2.6, "Why FinBERT, not VADER?", ["VADER is built for **social media**",
    "It misreads finance words like **liability** or **tax**", "FinBERT is trained on **financial text**: it improved every later result"],
    { size: 15, color: COL.green });
  txt(s, "Why split? Live answers to unexpected questions are much harder to spin than a script.", { x: MX, y: 5.45, w: 7.4, h: 0.9, fontSize: 16, color: COL.gray });
}
{
  const s = slide("CONTENT", "Phase 2  ·  Signal extraction  ·  Procedure", "18 language features per call, in five groups",
    "Sentence scores are summarised into 18 features per call. Tone level: the script, executive answers and analyst questions. " +
    "Bad moments: the share of negative sentences, the worst 10% of sentences and how mixed the tone is. " +
    "Contrast: the gap between script and Q&A, and delta tone, the change since the company's previous call. " +
    "Specificity, from a second FinBERT model that detects forward-looking statements. And dictionary and style features: Loughran-McDonald word lists, numbers, readability. " +
    "Delta tone matters most: some CEOs are cheerful every quarter, so a drop from their own normal is the real news. It uses only the past call, so there is no look-ahead.");
  tracker(s, 2);
  table(s, ["Group", "Features", "What it captures"], [
    ["Tone level (3)", "script · executive answers · analyst questions", "how positive each part sounds"],
    ["Bad moments (3)", "negative share · worst-10% tone · tone spread", "negative sentences, mixedness"],
    ["Contrast (2)", "tone gap · **delta tone**", "script vs live; change vs last call"],
    ["Specificity (2)", "forward-looking share · density", "how specific the outlook is"],
    ["Dictionary & style (8)", "uncertainty · modal · numbers · Fog …", "hedging, precision, readability"],
  ], { y: TOP + 0.05, w: 7.55, colW: [2.15, 3.15, 2.25], fontSize: 14, rowH: 0.62, hl: [2] });
  card(s, MX + 7.85, TOP + 0.05, 4.28, 3.0, "Key idea: delta tone", ["Some CEOs are cheerful **every** quarter, so absolute tone says little",
    "A **drop from their own normal** is the real news", "Uses only the previous call: **no look-ahead**"], { size: 15, color: COL.green });
  formula(s, "delta_tone = tone(this call)\n           − tone(previous call)", MX + 7.85, 4.95, 4.28, 1.0, 15);
  txt(s, "Output → data/features_finbert.csv: 12,716 calls × 22 columns (73 stub transcripts under 1,000 words skipped)",
    { x: MX, y: 6.15, w: CW, h: 0.6, fontSize: 16, bold: true, color: COL.green });
}
{
  const s = slide("CONTENT", "Phase 2  ·  Signal extraction  ·  Example & output", "Starbucks: the script was upbeat, but the tone fell sharply",
    "For Starbucks the prepared remarks still sounded fairly upbeat at plus 0.489. Analysts were slightly negative at minus 0.046. " +
    "The call was mixed, with a tone spread of 0.624. Most importantly, delta tone was minus 0.231, a big drop compared with the previous quarter. " +
    "Phase two's output is features_finbert.csv. Duplicate measures were removed after a correlation check, and GOOD versus BAD t-tests used only pre-2018 calls so the holdout stayed untouched.");
  tracker(s, 2);
  s.addChart(pres.charts.BAR, [{ name: "Starbucks", labels: ["prep_net_tone (script)", "analyst_tone", "delta_tone (vs last call)", "tone_std (mixedness)"],
    values: [0.489, -0.046, -0.231, 0.624] }], Object.assign(chartBase(), {
    x: MX, y: TOP, w: 7.2, h: 4.4, barDir: "bar", chartColors: [HEX.green, HEX.red, HEX.red, HEX.gray], invertedColors: [HEX.red],
    showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "+0.000;−0.000", showLegend: false,
    showTitle: true, title: "Starbucks call, 30 Jan 2024: FinBERT features", valAxisMinVal: -0.4, valAxisMaxVal: 0.8, valAxisMajorUnit: 0.2,
    valAxisLabelFormatCode: "0.0", catAxisOrientation: "maxMin", catAxisLabelPos: "low", barGapWidthPct: 60, objectName: "sbux-features" }));
  card(s, MX + 7.5, TOP, 4.63, 2.4, "Reading", ["Script still **fairly upbeat** (+0.489)", "Analysts **slightly negative** (−0.046)",
    "Tone **dropped 0.231** since last quarter", "A **mixed** call (spread 0.624)"], { size: 15 });
  card(s, MX + 7.5, TOP + 2.6, 4.63, 1.8, "Output → features_finbert.csv", ["12,716 calls × 22 columns", "Duplicates removed after a correlation check",
    "t-tests on **pre-2018 calls only**"], { size: 15, color: COL.green });
  callout(s, "Whatever the script said, the overall tone had **fallen sharply**: the model will see that in Phase 3",
    MX, 6.2, CW, 0.62, COL.navy);
}

// ---------- phase 3
newSection("Phase 3 · Classifier");
{
  const s = slide("CONTENT", "Phase 3  ·  Type classifier  ·  Key decision", "Train on the 3-day reaction, test on the 3-month drift",
    "Phase three turns the features into one score per call. The first decision: what should the model learn? " +
    "We train it to predict the immediate reaction: did the stock beat the S&P 500 from day minus 1 to plus 1? " +
    "Reason one: the 3-day reaction is mostly caused by the call, while a 3-month return is mostly other news, so it is a cleaner teacher. Training directly on GOOD/BAD gave a worse AUC, 0.511 against 0.537. " +
    "Reason two: the model never sees the 3-month outcome. If its score still predicts the drift, it cannot be memorisation; the market must have been slow.");
  tracker(s, 3);
  formula(s, "target = did the stock beat the S&P 500 from day −1 to +1?  (1 / 0)", MX, TOP + 0.05, CW, 0.7, 19);
  card(s, MX, 2.75, 5.9, 2.75, "Reason 1: a cleaner teacher", ["The 3-day reaction is **mostly caused by the call**",
    "A 3-month return is mostly other news", "Training on GOOD/BAD directly was worse: **AUC 0.511 vs 0.537**"], { size: 16 });
  card(s, MX + 6.23, 2.75, 5.9, 2.75, "Reason 2: an honest test", ["The model **never sees** the 3-month outcome",
    "If its score still predicts the drift, that can't be memorisation", "It must be because **the market was slow**"], { size: 16, color: COL.green });
  flow(s, [{ t: "Train on", s: "day −1 → +1 reaction", kind: "hi" }, { t: "Score every call", s: "out-of-sample" },
           { t: "Test on", s: "day +2 → +60 drift", kind: "up" }], 5.75, { h: 0.95, size: 17 });
}
{
  const s = slide("CONTENT", "Phase 3  ·  Type classifier  ·  Input & cleaning", "Ten inputs, cleaned with training data only",
    "Inputs were chosen on the validation years 2012 to 2017. Feature groups were added one at a time and kept only if AUC rose by at least 0.002. " +
    "Two groups survived: eight FinBERT tone features and two market inputs, pre-call return and pre-call volatility. Forward-looking, dictionary, style and even a word-frequency model added nothing. " +
    "Cleaning: extremes capped at the 1st and 99th percentile, so one stock that ran up 146% can't dominate; blanks filled with the median; every input standardised to a z-score. " +
    "All statistics come from the training years only, otherwise the future leaks in.");
  tracker(s, 3);
  table(s, ["#", "Input", "Group"], [
    ["1–3", "script tone · exec answer tone · **analyst tone**", "FinBERT tone"],
    ["4–6", "negative share · worst-10% tone · tone spread", "FinBERT tone"],
    ["7–8", "tone gap · **delta tone**", "FinBERT tone"],
    ["9–10", "pre-call return · pre-call volatility", "Market"],
  ], { y: TOP + 0.05, w: 7.0, colW: [0.8, 4.35, 1.85], fontSize: 15, rowH: 0.55 });
  card(s, MX, 4.6, 7.0, 2.0, "How they were chosen (on 2012–2017 only)", ["Groups added one at a time, kept only if AUC rose **≥ 0.002**",
    "Forward-looking, dictionary, style and word-frequency features added **nothing**: left out"], { size: 15 });
  const x = MX + 7.3, w = 4.83;
  txt(s, "Cleaning, in order", { x, y: TOP + 0.05, w, h: 0.4, fontSize: 19, bold: true, color: COL.navy });
  [["1", "Cap extremes at 1% / 99%", "one stock that ran up 146% can't dominate"],
   ["2", "Fill blanks with the median", "e.g. a first call has no delta tone"],
   ["3", "Standardise to z-scores", "standard deviations from a typical call"]].forEach(([n, h, b], i) => {
    const y = TOP + 0.6 + i * 1.2;
    badge(s, x, y + 0.1, n, COL.navy);
    txt(s, [{ text: h, options: { bold: true, fontSize: 17, breakLine: true } }, { text: b, options: { fontSize: 15, color: COL.gray } }],
      { x: x + 0.7, y, w: w - 0.7, h: 1.0 });
  });
  callout(s, "All statistics from **training years only**", x, 5.95, w, 0.65, COL.red);
}
{
  const s = slide("CONTENT", "Phase 3  ·  Type classifier  ·  The model", "Logistic regression, trained walk-forward",
    "The model is logistic regression. It multiplies each z-score by a weight, adds them with an intercept, and squashes the total through the sigmoid into a score between 0 and 1. " +
    "Why so simple? The signal in returns is small and noisy, and simple models overfit less. The weights are readable, so we can explain every score. XGBoost reached 0.618 against 0.611: no real gain. " +
    "Weights are found by the L-BFGS optimiser with an L2 penalty, C = 0.05. Training is walk-forward: every year from 2010 to 2025, a fresh model is trained only on earlier calls and scores the next year, with a 7-day embargo. " +
    "So every score in the project is out of sample.");
  tracker(s, 3);
  formula(s, "total = b₀ + w₁·z₁ + … + w₁₀·z₁₀\nscore = 1 / (1 + e^(−total))   ∈ (0, 1)",
    MX, TOP + 0.05, 7.2, 1.15, 18);
  bullets(s, ["**Why simple?** The signal in returns is small and noisy; simple models overfit less",
    "**Readable weights**: we can explain every single score",
    "**XGBoost tried**: AUC 0.618 vs 0.611, no real gain, so we kept the transparent model",
    "Weights found by **L-BFGS**, L2 penalty **C = 0.05** keeps them small"], { x: MX, y: 3.15, w: 7.2, h: 2.6, fontSize: 17 });
  // walk-forward diagram
  const x = MX + 7.55, w = 4.58;
  txt(s, "Walk-forward training", { x, y: TOP + 0.05, w, h: 0.4, fontSize: 19, bold: true, color: COL.navy });
  const yrs = ["2016", "2017", "2018", "2019"];
  yrs.forEach((yv, i) => {
    const y = TOP + 0.6 + i * 0.75, tw = 1.5 + i * 0.55;
    s.addShape(pres.shapes.RECTANGLE, { x, y, w: tw, h: 0.5, fill: { color: COL.navy }, line: { color: COL.navy }, objectName: `wf-train-${i}` });
    txt(s, "train: all earlier calls", { x: x + 0.08, y: y + 0.08, w: tw - 0.1, h: 0.36, fontSize: 12, color: COL.white, valign: "middle" });
    s.addText(`score ${yv}`, { shape: pres.shapes.RECTANGLE, x: x + tw + 0.08, y, w: 1.3, h: 0.5, fill: { color: COL.green },
      line: { color: COL.green }, fontSize: 14, bold: true, color: COL.white, align: "center", valign: "middle", margin: 0, isTextBox: true });
  });
  txt(s, "… a fresh model every year, 2010 → 2025", { x, y: TOP + 3.65, w, h: 0.35, fontSize: 14, color: COL.gray });
  callout(s, "**7-day embargo** + walk-forward = every score is **out-of-sample**", x, 5.85, w, 0.85, COL.green);
}
{
  const s = slide("CONTENT", "Phase 3  ·  Type classifier  ·  What it learned", "The model trusts the unscripted signals most",
    "Here are the weights of the 2024 model. The biggest, plus 0.287, is analyst tone: when analysts sound positive the stock tends to react up. " +
    "Second is delta tone, plus 0.247: improving tone versus last quarter is good news. Script tone is plus 0.136. " +
    "A mixed call with a high tone spread is negative, and a stock that already ran up tends to react worse. " +
    "Negative share and worst-10% tone are almost mirror images, so read their weights together. " +
    "In game-theory terms, the model trusts the signals that are hardest to fake: the analysts, and the change in tone.");
  tracker(s, 3);
  const lab = ["analyst_tone", "delta_tone", "prep_net_tone", "neg_share", "pre_vol", "tone_p10", "tone_gap", "qa_exec_tone", "pre_ar", "tone_std"];
  const val = [0.287, 0.247, 0.136, 0.127, 0.075, 0.035, 0.035, -0.028, -0.069, -0.085];
  s.addChart(pres.charts.BAR, [{ name: "Weight", labels: lab, values: val }], Object.assign(chartBase(), {
    x: MX, y: TOP - 0.05, w: 7.6, h: 5.2, barDir: "bar", chartColors: [HEX.green], invertedColors: [HEX.red],
    showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "+0.000;−0.000", showLegend: false,
    showTitle: true, title: "Weights of the 2024 model (per 1 standard deviation)", valAxisMinVal: -0.2, valAxisMaxVal: 0.4, valAxisMajorUnit: 0.1,
    valAxisLabelFormatCode: "0.0", catAxisOrientation: "maxMin", catAxisLabelPos: "low", barGapWidthPct: 45, catAxisLabelFontSize: 13, objectName: "weights" }));
  card(s, MX + 7.9, TOP, 4.23, 2.45, "Top two inputs", ["**Analyst tone +0.287**: positive analysts → stock reacts up",
    "**Delta tone +0.247**: improving tone vs last quarter"], { size: 15, color: COL.green });
  card(s, MX + 7.9, TOP + 2.65, 4.23, 2.5, "Game-theory reading", ["The script is **cheap talk**: easy to polish",
    "Analysts and tone **change** are hardest to fake, so they get the biggest weights"], { size: 15 });
}
{
  const s = slide("CONTENT", "Phase 3  ·  Type classifier  ·  Example", "Starbucks gets a score of 0.317",
    "Starbucks' delta tone of minus 0.231 is minus 3.04 standard deviations. Times the weight 0.247, that contributes minus 0.752. " +
    "Analyst tone contributes minus 0.189. The upbeat script adds 0.128, the earlier price fall adds 0.110, and the mixed tone subtracts 0.084. " +
    "The other five inputs and the intercept add 0.023. The total is minus 0.764, and the sigmoid gives a score of 0.317, one of the lowest of that quarter. " +
    "Whatever the script said, the tone fell three standard deviations, and that is a warning.");
  tracker(s, 3);
  table(s, ["Input", "z-score", "× weight", "= contribution"], [
    ["delta_tone", "−3.04", "+0.247", "−0.752"],
    ["analyst_tone", "−0.66", "+0.287", "−0.189"],
    ["prep_net_tone", "+0.94", "+0.136", "+0.128"],
    ["pre_ar (pre-call return)", "−1.58", "−0.069", "+0.110"],
    ["tone_std (mixedness)", "+1.00", "−0.085", "−0.084"],
    ["other 5 inputs + intercept", "", "", "+0.023"],
    ["**Total**", "", "", "**−0.764**"],
  ], { y: TOP + 0.05, w: 7.6, colW: [3.1, 1.4, 1.4, 1.7], numFrom: 1, fontSize: 16, rowH: 0.5, hl: [6], noSign: false });
  formula(s, "score = 1 / (1 + e^0.764) = 0.317", MX, 5.85, 7.6, 0.7, 20);
  const x = MX + 7.9;
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: TOP + 0.05, w: 4.23, h: 2.4, rectRadius: 0.08, fill: { color: COL.tint },
    line: { color: COL.tint }, objectName: "score-box" });
  txt(s, "0.317", { x: x + 0.25, y: TOP + 0.2, w: 3.8, h: 1.2, fontSize: 66, bold: true, color: COL.red, valign: "bottom" });
  txt(s, "one of the lowest scores of Q1 2024 (0.5 = neutral)", { x: x + 0.25, y: TOP + 1.45, w: 3.8, h: 0.8, fontSize: 16, color: COL.gray });
  card(s, x, TOP + 2.65, 4.23, 2.5, "The model's message", ["The biggest pull is **delta tone**: −3 standard deviations",
    "Whatever the script says, a fall this large is a **warning**"], { size: 15, color: COL.red });
}
{
  const s = slide("CONTENT", "Phase 3  ·  Type classifier  ·  Output", "The score beats a coin flip on unseen years",
    "On the holdout years 2018 to 2025, the AUC for the immediate reaction is 0.611; on validation it was 0.640. " +
    "AUC is the chance that a random up-reacting call is ranked above a random down-reacting one; 0.5 is a coin flip. " +
    "The same model on the old VADER features reached 0.588, XGBoost 0.618, and a fixed 2022–2025 split gave the same 0.611. " +
    "For stock returns, 0.61 is a real and useful signal; above 0.7 would make us suspect a leak. " +
    "Output: predictions_finbert.csv, one out-of-sample score for every call.");
  tracker(s, 3);
  s.addChart(pres.charts.BAR, [{ name: "AUC", labels: ["Holdout 2018–2025", "Validation 2012–2017", "Old VADER features", "XGBoost instead", "Fixed split 2022–25", "Coin flip"],
    values: [0.611, 0.640, 0.588, 0.618, 0.611, 0.5] }], Object.assign(chartBase(), {
    x: MX, y: TOP - 0.05, w: 7.6, h: 4.9, barDir: "col", chartColors: [HEX.navy, HEX.blue, HEX.gray, HEX.gray, HEX.blue, "B9C2CE"],
    showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.000", showLegend: false,
    showTitle: true, title: "Reaction AUC (higher is better)", valAxisMinVal: 0.45, valAxisMaxVal: 0.66,
    valAxisLabelFormatCode: "0.00", catAxisLabelFontSize: 12, barGapWidthPct: 50, objectName: "auc-chart" }));
  stat(s, MX + 7.95, TOP, 4.1, "0.611", "AUC on the locked 2018–2025 holdout", COL.navy);
  card(s, MX + 7.9, TOP + 1.75, 4.23, 2.1, "What is AUC?", ["Chance a random **up-reacting** call is ranked above a random **down-reacting** one",
    "0.5 = coin flip  ·  > 0.7 would suggest a leak"], { size: 15 });
  card(s, MX + 7.9, TOP + 4.05, 4.23, 1.15, "Output → predictions_finbert.csv", "one out-of-sample score per call", { size: 15, color: COL.green });
  txt(s, "FinBERT beats VADER: 0.611 vs 0.588", { x: MX, y: 6.6, w: 7.6, h: 0.35, fontSize: 14, color: COL.gray });
}

// ---------- phase 4
newSection("Phase 4 · Market test");
{
  const s = slide("CONTENT", "Phase 4  ·  Market-reaction test  ·  Hypothesis & method", "Efficient market, or a slow one?",
    "This is the heart of the project. If markets were perfectly efficient, everything the call revealed would be priced in by day plus 1; the score would predict the reaction and nothing after. " +
    "If the score also predicts days plus 2 to plus 60, the market under-reacted. " +
    "Method: within each calendar quarter, rank calls by score into five equal groups, quintiles. Ranking within the quarter compares calls made under the same market conditions. " +
    "The gap is top quintile minus bottom quintile, for each of the 30 holdout quarters. A t-test with Newey-West errors, which allow neighbouring quarters to be related, asks whether the average gap is above zero.");
  tracker(s, 4);
  card(s, MX, TOP + 0.05, 5.9, 2.1, "If the market is efficient", ["Everything is priced in by day +1",
    "The score predicts the reaction and **nothing after**"], { size: 16, color: COL.gray });
  card(s, MX + 6.23, TOP + 0.05, 5.9, 2.1, "If the market is slow (our hypothesis)", ["The score **also** predicts days +2 → +60",
    "The market took weeks to believe the call"], { size: 16, color: COL.green });
  flow(s, [{ t: "Each quarter", s: "rank calls by score" }, { t: "5 quintiles", s: "equal-sized groups" },
           { t: "Gap", s: "top 20% − bottom 20%", kind: "hi" }, { t: "30 quarters", s: "2018–2025 holdout" },
           { t: "t-test", s: "Newey–West errors", kind: "up" }], 4.15, { h: 1.05, size: 17 });
  formula(s, "drift gap  =  average drift(top 20%)  −  average drift(bottom 20%)", MX, 5.5, CW, 0.65, 19);
  txt(s, "Ranking within a quarter compares calls made under the same market conditions. p below 0.05 = unlikely to be luck.",
    { x: MX, y: 6.3, w: CW, h: 0.4, fontSize: 15, color: COL.gray });
}
{
  const s = slide("CONTENT", "Phase 4  ·  Market-reaction test  ·  Result", "High-score calls keep rising, low-score calls keep falling",
    "In the immediate window, the lowest quintile lost 1.94% against the market and the highest gained 2.00%, a gap of 3.94% with a t-statistic of 13. The market clearly reacts to the language. " +
    "In the drift window, after the reaction is over, the lowest quintile keeps drifting down, minus 0.53%, and the highest keeps drifting up, plus 1.10%. " +
    "The average quarterly gap is plus 1.62%, with a p-value of 0.002. That is the slow reaction.");
  tracker(s, 4);
  const labs = ["Q1 lowest", "Q2", "Q3", "Q4", "Q5 highest"];
  const mk = (vals, title, x, min, max) => s.addChart(pres.charts.BAR, [{ name: "AR", labels: labs, values: vals }], Object.assign(chartBase(), {
    x, y: TOP - 0.05, w: 5.95, h: 4.3, barDir: "col", chartColors: [HEX.green], invertedColors: [HEX.red],
    showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "+0.00\"%\";−0.00\"%\"", showLegend: false,
    showTitle: true, title, valAxisMinVal: min, valAxisMaxVal: max, valAxisLabelFormatCode: "0.0\"%\"", barGapWidthPct: 45, catAxisLabelPos: "low",
    objectName: title }));
  mk([-1.94, -0.46, 0.33, 1.07, 2.00], "Immediate reaction, day −1 → +1", MX, -2.5, 2.5);
  mk([-0.53, -0.02, 0.25, 0.24, 1.10], "Drift, day +2 → +60 (after the reaction)", MX + 6.18, -1.0, 1.5);
  [["+3.94%", "immediate gap (t 13.2): the market reacts to the words", MX],
   ["+1.62%", "drift gap (p 0.002): it keeps moving for weeks = under-reaction", MX + 6.18]].forEach(([v, l, x]) => {
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: 5.45, w: 5.95, h: 1.25, rectRadius: 0.08, fill: { color: COL.tint },
      line: { color: COL.tint }, objectName: "gap-box" });
    txt(s, v, { x: x + 0.25, y: 5.55, w: 2.0, h: 1.05, fontSize: 36, bold: true, color: COL.green, valign: "middle" });
    txt(s, l, { x: x + 2.3, y: 5.55, w: 3.5, h: 1.05, fontSize: 15, valign: "middle" });
  });
}
{
  const s = slide("CONTENT", "Phase 4  ·  Market-reaction test  ·  Robustness & output", "It really is the words",
    "Could it be something else in disguise? A regression of drift on the score's rank, controlling for the immediate reaction, pre-call return and volatility, with quarter fixed effects, still gives plus 1.31%, p 0.013. " +
    "As a placebo, the same test on a model with no language at all gives minus 0.36%, not significant. So the drift comes from the words. " +
    "VADER gives a weaker plus 1.40% with p 0.059, and the validation years give plus 0.85%, p 0.031. " +
    "About 29% of the total move came after the reaction window, and the gap was positive in 22 of 30 quarters. Starbucks sat in the bottom quintile of Q1 2024, and its minus 8.3% drift is one of those data points.");
  tracker(s, 4);
  table(s, ["Test (holdout 2018–2025)", "Drift gap", "p-value"], [
    ["Score only", "+1.62%", "0.002"],
    ["+ controls (reaction, pre-return, volatility)", "+1.31%", "0.013"],
    ["Placebo: model with **no language**", "−0.36%", "0.53"],
    ["VADER score instead", "+1.40%", "0.059"],
    ["Validation years 2012–2017", "+0.85%", "0.031"],
  ], { y: TOP + 0.05, w: 7.6, colW: [4.6, 1.5, 1.5], numFrom: 1, fontSize: 16, rowH: 0.55, hl: [0, 1] });
  stat(s, MX + 7.95, TOP, 2.0, "29%", "of the total move came after the reaction", COL.green);
  stat(s, MX + 10.15, TOP, 2.0, "22 / 30", "quarters with a positive gap", COL.navy);
  card(s, MX + 7.9, TOP + 1.85, 4.23, 2.0, "Placebo proves the point", ["With **no language**, the drift gap vanishes (−0.36%, p 0.53)",
    "So the slow drift comes from the **words**"], { size: 15, color: COL.green });
  card(s, MX + 7.9, TOP + 4.0, 4.23, 1.2, "Starbucks", "bottom quintile of Q1 2024; its −8.3% drift is one of those points", { size: 15, color: COL.red });
  callout(s, "Output → evidence of **under-reaction**: the market is slow to absorb what the call's language reveals", MX, 5.95, 7.6, 0.75, COL.navy);
}

// ---------- phase 5
newSection("Phase 5 · Backtest");
{
  const s = slide("CONTENT", "Phase 5  ·  Trading backtest  ·  Procedure", "Rules a real trader could follow on the day",
    "Phase five asks the practical question. When a call happens, its score is compared with all calls in the previous 365 days, with at least 200 past calls. " +
    "Top 20%: buy, go long. Bottom 20%: sell short. Using the whole sample's distribution would be look-ahead. " +
    "We enter at the close of day plus 2, because the immediate reaction is already gone, and exit at day plus 60, the end of the drift window. " +
    "Money is split equally across open positions on each side; an empty side's money sits in the S&P 500. Costs: 10 basis points on every buy and sell, and 0.5% a year to borrow shorted stock. " +
    "Strategy A, the main test, puts 100 long and 100 short, so it is market-neutral: any profit must come from the signal.");
  tracker(s, 5);
  flow(s, [{ t: "Call happens" }, { t: "Rank vs past 365 days", s: "≥ 200 earlier calls" }, { t: "Top 20%", s: "→ LONG", kind: "up" },
           { t: "Bottom 20%", s: "→ SHORT", kind: "dn" }, { t: "Enter day +2" }, { t: "Exit day +60", kind: "hi" }], TOP + 0.1, { h: 1.1, size: 16, arrow: 0.32 });
  card(s, MX, 3.2, 3.9, 2.55, "No look-ahead", ["Cut-offs from the **past year's** scores only",
    "Enter on **day +2**: the first reaction is gone, we only chase the drift"], { size: 15 });
  card(s, MX + 4.12, 3.2, 3.9, 2.55, "Portfolio", ["Equal weight across open positions, each side", "Empty side's money sits in the **S&P 500**"],
    { size: 15 });
  card(s, MX + 8.24, 3.2, 3.89, 2.55, "Realistic costs", ["**10 bps** (0.10%) on every buy and sell", "**0.5% a year** to borrow shorted stock"],
    { size: 15, color: COL.red });
  formula(s, "Strategy A:  ₹100 long top 20%  +  ₹100 short bottom 20%   →  market-neutral", MX, 6.0, CW, 0.7, 18);
}
{
  const s = slide("CONTENT", "Phase 5  ·  Trading backtest  ·  Measures & example", "How we judge a strategy, and the Starbucks trade",
    "We judge strategies by annual return; the Sharpe ratio, return divided by volatility, return per unit of risk; maximum drawdown, the worst fall from a peak; " +
    "beta, how much it moves with the market; and alpha, the return left after removing beta times the market. " +
    "Sharpe guide: below 0.5 weak, 0.5 to 1 decent, above 1 very good, above 2 in a backtest usually means a mistake. " +
    "For Starbucks, the score of 0.317 was in the bottom 2% of the past year, so the strategy shorted it on day plus 2 and closed on day plus 60. Starbucks lagged the market by 8.3%, so the short gained, less costs. " +
    "One trade proves nothing; over thousands of trades the average tilts our way.");
  tracker(s, 5);
  table(s, ["Measure", "Meaning"], [
    ["Annual return", "growth per year, after costs"],
    ["Sharpe ratio", "return ÷ volatility: return per unit of risk"],
    ["Max drawdown", "worst fall from a previous peak"],
    ["Beta", "how much it moves when the market moves 1%"],
    ["Alpha", "return left after removing beta × market"],
  ], { y: TOP + 0.05, w: 6.6, colW: [2.1, 4.5], fontSize: 16, rowH: 0.52 });
  txt(s, "Sharpe guide:  < 0.5 weak  ·  0.5–1 decent  ·  > 1 very good  ·  > 2 in a backtest = probably a mistake",
    { x: MX, y: 5.75, w: 6.6, h: 0.7, fontSize: 15, color: COL.gray });
  const x = MX + 6.95, w = 5.18;
  txt(s, "The Starbucks trade", { x, y: TOP + 0.05, w, h: 0.4, fontSize: 19, bold: true, color: COL.navy });
  [["30 Jan 2024", "call, score 0.317", COL.navy], ["Bottom 2%", "of the past year's calls", COL.red],
   ["SHORT", "day +2 → +60", COL.red], ["Stock lagged 8.3%", "the short gains (less 0.1% × 2 + borrow)", COL.green]]
    .forEach(([h, b, c], i) => {
      const y = TOP + 0.55 + i * 0.86;
      badge(s, x, y + 0.1, String(i + 1), c);
      txt(s, [{ text: h, options: { bold: true, fontSize: 17, color: c, breakLine: true } }, { text: b, options: { fontSize: 15 } }],
        { x: x + 0.7, y, w: w - 0.7, h: 0.85 });
    });
  callout(s, "One trade proves nothing. Over **thousands** of trades the average tilts our way", x, 5.8, w, 0.85, COL.navy);
}
{
  const s = slide("CONTENT", "Phase 5  ·  Trading backtest  ·  Output", "Strategy A: profitable after costs, with almost no market risk",
    "Strategy A on the holdout, after all costs, returned 6.9% a year with a Sharpe of 0.74, p 0.032. Its worst fall was only 14.1% and its alpha was 5.7% a year. 100 rupees became 161. " +
    "The break-even cost is 46 basis points per trade, so costs could be four to five times higher before the profit disappears. " +
    "The same backtest with the no-language score or the old VADER score did worse.");
  tracker(s, 5);
  [["+6.9%", "per year after costs", COL.green], ["0.74", "Sharpe ratio (p 0.032)", COL.navy], ["−14.1%", "worst fall (S&P: −33.9%)", COL.red],
   ["₹100 → ₹161", "Jan 2018 → May 2025", COL.navy]].forEach(([v, l, c], i) => {
    const x = MX + i * 3.08;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: TOP + 0.05, w: 2.88, h: 1.7, rectRadius: 0.08, fill: { color: COL.tint },
      line: { color: COL.tint }, objectName: `a-kpi-${i}` });
    txt(s, v, { x: x + 0.2, y: TOP + 0.15, w: 2.55, h: 0.9, fontSize: i === 3 ? 30 : 40, bold: true, color: c, valign: "bottom" });
    txt(s, l, { x: x + 0.2, y: TOP + 1.1, w: 2.55, h: 0.55, fontSize: 15, color: COL.gray });
  });
  card(s, MX, 3.75, 3.9, 2.3, "Alpha +5.7% a year", "Return the market cannot explain: beta is only 0.10", { size: 16, color: COL.green });
  card(s, MX + 4.12, 3.75, 3.9, 2.3, "Break-even cost 46 bps", "Costs could be 4–5× higher than assumed before the profit disappears", { size: 16 });
  card(s, MX + 8.24, 3.75, 3.89, 2.3, "Language matters", "The no-language and VADER scores did worse in the same backtest", { size: 16 });
  callout(s, "A is the **main scientific result**: its profit cannot come from the market rising", MX, 6.25, CW, 0.6, COL.navy);
}

// ---------- phase 6
newSection("Phase 6 · Dashboard");
{
  const s = slide("CONTENT", "Phase 6  ·  Dashboard & report", "Everything in one place",
    "Phase six brings everything together in a Streamlit dashboard. It reads only files committed in the repository, so anyone can run it. " +
    "It has seven pages: an overview; a call explorer where you can open any call, like Starbucks, and see its score, signal and price path; " +
    "the language features; the classifier's accuracy by year; the market reaction tests; all strategies against the S&P 500; and the method and limits. " +
    "Alongside it we wrote the strategy report and built an interactive journey page.");
  tracker(s, 6);
  formula(s, "pip install -r requirements.txt\nstreamlit run phase6_dashboard.py", MX, TOP + 0.05, 5.0, 1.0, 17);
  bullets(s, ["Input: **only committed result files**: no raw transcripts needed", "Headline numbers recomputed when the app starts",
    "Also delivered: **Strategy report**, interactive **journey page**, explainer video"], { x: MX, y: 3.05, w: 5.0, h: 3.2, fontSize: 17 });
  const pages = [["Overview", "headline numbers, pipeline, growth of ₹100"], ["Call explorer", "any call: score, signal, price path"],
    ["Language", "each feature, GOOD vs BAD t-tests"], ["Classifier", "AUC by year, model comparison"],
    ["Market reaction", "quintiles, drift per quarter"], ["Strategies", "A–G vs the S&P 500, drawdowns"], ["Method", "rules and limitations"]];
  pages.forEach(([h, b], i) => {
    const col = i % 2, row = Math.floor(i / 2);
    const x = MX + 5.35 + col * 3.44, y = TOP + 0.05 + row * 1.25;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w: 3.3, h: 1.1, rectRadius: 0.08, fill: { color: COL.tint },
      line: { color: COL.tint }, objectName: `dash-${i}` });
    badge(s, x + 0.15, y + 0.3, String(i + 1), COL.navy, 0.45);
    txt(s, [{ text: h, options: { bold: true, fontSize: 16, color: COL.navy, breakLine: true } }, { text: b, options: { fontSize: 13 } }],
      { x: x + 0.72, y: y + 0.1, w: 2.5, h: 0.9, valign: "middle" });
  });
}

// ---------- results
newSection("Results");
{
  const s = slide("DIVIDER", "The finish line", "Returns & strategies",
    "Now the finish line: returns and strategies. We keep exactly the same scores, and change only how the money is placed.");
  s.addText("Same scores, seven ways to place the money. All results are on the untouched 2018 – May 2025 holdout, after trading costs.",
    { placeholder: "body" });
  tracker(s, 7);
}
{
  const s = slide("CONTENT", "Backtest results  ·  Design", "Seven strategies: same scores, different money split",
    "A is market-neutral, 100 long and 100 short on the top and bottom 20%. It was fixed in advance from the Phase 4 results. " +
    "B tilts 2 to 1, 100 long and 50 short, on the same 20% cut-offs: only the money split changes. " +
    "C shorts only the bottom 10%, D uses 15% on both sides, E is long only the top 15%, F uses 5 and 5%, and G 10 and 5%. " +
    "C to G also change the cut-offs, which were picked after seeing earlier results, so they are exploratory. " +
    "Our chosen strategy is D, highlighted: 15% on both sides with the 2 to 1 split.");
  tracker(s, 7);
  table(s, ["", "Long side", "Short side", "Money per ₹100", "Beta", "Avg. positions", "Status"], [
    ["**A**  Market-neutral", "top 20%", "bottom 20%", "100 long / 100 short", "0.10", "36 / 36", "pre-set (main test)"],
    ["**B**  Tilted 2:1", "top 20%", "bottom 20%", "100 long / 50 short", "0.56", "36 / 36", "pre-set cut-offs"],
    ["**C**  Tilted, narrow short", "top 20%", "bottom 10%", "100 long / 50 short", "0.56", "36 / 18", "exploratory"],
    ["**D**  Tilted, 15 / 15", "top 15%", "bottom 15%", "100 long / 50 short", "0.57", "27 / 27", "our choice"],
    ["**E**  Long-only", "top 15%", "none", "100 long", "1.04", "27 / 0", "exploratory"],
    ["**F**  Tilted, very narrow", "top 5%", "bottom 5%", "100 long / 50 short", "0.59", "9 / 9", "exploratory"],
    ["**G**  Tilted, narrow short", "top 10%", "bottom 5%", "100 long / 50 short", "0.58", "18 / 9", "exploratory"],
  ], { y: TOP + 0.05, colW: [2.85, 1.35, 1.5, 2.3, 0.9, 1.5, 1.73], fontSize: 15, rowH: 0.5, hl: [3] });
  txt(s, "Every strategy uses the same signal, timing (day +2 → +60) and costs (10 bps per trade, 0.5%/yr borrow). Only the cut-offs and the money split change.",
    { x: MX, y: 5.95, w: CW, h: 0.7, fontSize: 16, color: COL.gray });
}
{
  const s = slide("CONTENT", "Backtest results  ·  Holdout 2018 – May 2025, after costs", "All strategies side by side",
    "Here are the results, all after costs on the holdout. A: 6.9% a year, Sharpe 0.74, worst fall 14.1%. " +
    "Our chosen strategy D: 13.3% a year, Sharpe 0.93, worst fall 20.2%, alpha 6.0% a year with p 0.045, and 100 rupees became 246. B and C are very similar, 12.9 and 13.4%. " +
    "E, long only, has the highest raw return, 18.1%, but its worst fall is 37.5%, deeper than the market's. " +
    "F and G return about 13% with deeper falls and alphas that are no longer significant. Buying every call returns 13.6% with almost no alpha, and the S&P 500 returned 12.8%, Sharpe 0.63, worst fall 33.9%.");
  tracker(s, 7);
  table(s, ["Strategy", "Return / yr", "Volatility", "Sharpe", "Worst fall", "Beta", "Alpha / yr (p)", "₹100 →"], [
    ["A  Market-neutral 20/20", "+6.9%", "9.3%", "0.74", "−14.1%", "0.10", "+5.7% (0.076)", "161"],
    ["B  Tilted 2:1, 20/20", "+12.9%", "13.6%", "0.95", "−19.3%", "0.56", "+5.6% (0.034)", "241"],
    ["C  Tilted 2:1, 20/10", "+13.4%", "13.8%", "0.97", "−21.0%", "0.56", "+6.2% (0.027)", "250"],
    ["D  Tilted 2:1, 15/15", "+13.3%", "14.2%", "0.93", "−20.2%", "0.57", "+6.0% (0.045)", "246"],
    ["E  Long-only 15%", "+18.1%", "22.5%", "0.81", "−37.5%", "1.04", "+4.9% (0.101)", "315"],
    ["F  Tilted 2:1, 5/5", "+13.1%", "17.2%", "0.76", "−27.8%", "0.59", "+5.9% (0.175)", "234"],
    ["G  Tilted 2:1, 10/5", "+13.3%", "15.7%", "0.84", "−22.6%", "0.58", "+5.9% (0.129)", "242"],
    ["Buy every call (no score)", "+13.6%", "20.2%", "0.67", "−38.9%", "0.96", "+1.0% (0.62)", "235"],
    ["S&P 500 buy-and-hold", "+12.8%", "20.2%", "0.63", "−33.9%", "1.00", "—", "220"],
  ], { y: TOP + 0.0, colW: [3.05, 1.3, 1.25, 1.05, 1.35, 0.9, 2.08, 1.15], numFrom: 1, fontSize: 15, rowH: 0.47, hl: [3], mute: [7, 8] });
  txt(s, "Alpha = return the market can't explain; p below 0.05 = statistically significant (Newey–West, 10 lags). Green row = our chosen strategy, D.",
    { x: MX, y: 6.5, w: CW, h: 0.4, fontSize: 13, color: COL.gray });
}
{
  // equity curves from journey/data.js
  const raw = fs.readFileSync(path.join(__dirname, "..", "journey", "data.js"), "utf8");
  const J = JSON.parse(raw.slice(raw.indexOf("{"), raw.lastIndexOf("}") + 1));
  const step = 4;
  const idx = J.dates.map((_, i) => i).filter((i) => i % step === 0 || i === J.dates.length - 1);
  const labels = idx.map((i, k) => (k === 0 || J.dates[i].slice(0, 4) !== J.dates[idx[k - 1]].slice(0, 4)) ? J.dates[i].slice(0, 4) : "");
  const ser = [["A", "A  market-neutral"], ["D", "D  tilted 15/15 (our choice)"], ["E", "E  long-only"], ["SPX", "S&P 500"]]
    .map(([k, n]) => ({ name: n, labels, values: idx.map((i) => Math.round(J.series[k][i] * 10) / 10) }));
  const s = slide("CONTENT", "Backtest results  ·  Growth of ₹100", "How ₹100 grew from January 2018",
    "This chart shows how 100 rupees grew from January 2018, after costs. Notice the 2020 crash and the 2022 bear market. " +
    "The market-neutral strategy A, in green, barely notices them. Our chosen strategy D, in blue, ends at 246, above the S&P 500 at about 220, with much shallower dips. " +
    "The long-only strategy E, in orange, ends highest at 315 but falls hard with the market.");
  tracker(s, 7);
  s.addChart(pres.charts.LINE, ser, Object.assign(chartBase(), {
    x: MX, y: TOP - 0.1, w: 8.75, h: 5.15, chartColors: [HEX.green, HEX.blue, HEX.amber, HEX.gray], lineSize: 2.5, lineDataSymbol: "none",
    showLegend: true, legendPos: "b", showTitle: false, valAxisMinVal: 50, valAxisMaxVal: 350, valAxisMajorUnit: 50,
    catAxisLabelFrequency: 1, catAxisMajorTickMark: "none", catAxisLabelFontSize: 13, valAxisLabelFormatCode: "₹0", objectName: "equity-chart" }));
  const x = MX + 9.0, w = 3.13;
  [["₹315", "E long-only", COL.amber], ["₹246", "D tilted 15/15", COL.blue], ["₹220", "S&P 500", COL.gray], ["₹161", "A market-neutral", COL.green]]
    .forEach(([v, l, c], i) => {
      const y = TOP + i * 1.05;
      txt(s, v, { x, y, w: 1.55, h: 0.8, fontSize: 30, bold: true, color: c, valign: "middle" });
      txt(s, l, { x: x + 1.55, y, w: w - 1.55, h: 0.8, fontSize: 15, valign: "middle" });
    });
  txt(s, "Final values at the end of the holdout. The chart's S&P line runs a few weeks past the last trade, so it ends near 223.",
    { x, y: TOP + 4.3, w, h: 0.9, fontSize: 12, color: COL.gray });
}
{
  const s = slide("CONTENT", "Backtest results  ·  Risk vs reward", "Every FinBERT strategy earns more per unit of risk",
    "Here is the Sharpe ratio of every strategy, return per unit of risk. Every FinBERT strategy beats the S&P 500's 0.63 and buying every call at 0.67. " +
    "C, B and D lead at about 0.93 to 0.97; our chosen D, in blue, is 0.93. On the right, the worst fall: A to D and G lose far less than the market in a crash, while E and buying every call fall as hard as the market.");
  tracker(s, 7);
  const labs = ["A", "B", "C", "D", "E", "F", "G", "All", "S&P"];
  s.addChart(pres.charts.BAR, [{ name: "Sharpe", labels: labs, values: [0.74, 0.95, 0.97, 0.93, 0.81, 0.76, 0.84, 0.67, 0.63] }],
    Object.assign(chartBase(), { x: MX, y: TOP - 0.05, w: 5.95, h: 4.2, barDir: "col",
      chartColors: [HEX.green, HEX.green, HEX.green, HEX.blue, HEX.green, HEX.green, HEX.green, HEX.gray, HEX.gray],
      showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.00", showLegend: false, showTitle: true,
      title: "Sharpe ratio (return per unit of risk)", valAxisMinVal: 0, valAxisMaxVal: 1.1, valAxisLabelFormatCode: "0.0",
      barGapWidthPct: 40, dataLabelFontSize: 13, objectName: "sharpe-chart" }));
  s.addChart(pres.charts.BAR, [{ name: "Worst fall", labels: labs, values: [-14.1, -19.3, -21.0, -20.2, -37.5, -27.8, -22.6, -38.9, -33.9] }],
    Object.assign(chartBase(), { x: MX + 6.18, y: TOP - 0.05, w: 5.95, h: 4.2, barDir: "col",
      chartColors: [HEX.red], invertedColors: [HEX.red], showValue: true, dataLabelPosition: "inEnd",
      dataLabelFormatCode: "0\"%\"", dataLabelColor: HEX.white, dataLabelFontSize: 12, showLegend: false, showTitle: true,
      title: "Worst fall from a peak (smaller is better)", valAxisMinVal: -45, valAxisMaxVal: 0, valAxisLabelFormatCode: "0\"%\"",
      barGapWidthPct: 40, objectName: "drawdown-chart" }));
  callout(s, "S&P 500: Sharpe **0.63**, worst fall **−33.9%**.  Strategy D: Sharpe **0.93**, worst fall **−20.2%**   ·   All = buy every call", MX, 6.05, CW, 0.7, COL.navy);
}
{
  const s = slide("CONTENT", "The winning result", "Strategy D: +13.3% a year, after costs",
    "Our chosen strategy is D: tilted 2 to 1, 100 rupees long the top 15% of calls and 50 rupees short the bottom 15%. " +
    "It returned 13.3% a year after costs, slightly above the S&P 500's 12.8%, but with about 30% less volatility, 14.2% against 20.2%, " +
    "and a worst fall about 40% smaller, 20.2% against 33.9%. Its Sharpe ratio is 0.93 against 0.63, and it earns an alpha of 6.0% a year " +
    "that is statistically significant, p 0.045. 100 rupees became 246, against 220 for the S&P 500.");
  tracker(s, 7);
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: MX, y: TOP + 0.05, w: 4.6, h: 4.95, rectRadius: 0.08, fill: { color: COL.navy },
    line: { color: COL.navy }, objectName: "hero-box" });
  txt(s, "+13.3%", { x: MX + 0.3, y: TOP + 0.35, w: 4.0, h: 1.3, fontSize: 72, bold: true, color: COL.white, valign: "bottom" });
  txt(s, "per year, after all trading costs", { x: MX + 0.3, y: TOP + 1.7, w: 4.0, h: 0.5, fontSize: 18, color: COL.white });
  txt(s, [{ text: "₹100 → ₹246", options: { bold: true, fontSize: 28, breakLine: true } },
    { text: "January 2018 → May 2025", options: { fontSize: 16 } }], { x: MX + 0.3, y: TOP + 2.55, w: 4.0, h: 1.0, color: COL.white });
  txt(s, "100 long top 15%  ·  50 short bottom 15%", { x: MX + 0.3, y: TOP + 3.95, w: 4.0, h: 0.8, fontSize: 16, color: COL.white });
  // comparison D vs S&P
  table(s, ["", "Strategy D", "S&P 500", "D is…"], [
    ["Return per year", "+13.3%", "+12.8%", "slightly higher"],
    ["Volatility", "14.2%", "20.2%", "≈ 30% calmer"],
    ["Sharpe ratio", "0.93", "0.63", "+48% better"],
    ["Worst fall", "−20.2%", "−33.9%", "≈ 40% smaller"],
    ["Alpha per year (p)", "+6.0% (0.045)", "—", "significant"],
    ["₹100 became", "₹246", "₹220", "+₹26"],
  ], { x: MX + 4.9, y: TOP + 0.05, w: 7.23, colW: [2.2, 1.75, 1.4, 1.88], numFrom: 1, fontSize: 16, rowH: 0.55, noSign: true });
  callout(s, "More than the market's return, with **less risk** and a **much smaller worst crash**", MX + 4.9, 5.85, 7.23, 1.0, COL.green);
}
{
  const s = slide("CONTENT", "The winning result  ·  Why D?", "Why we chose Strategy D",
    "Why D, and not the strategy with the biggest number? E has 18.1% and C has 13.4%. " +
    "First, D is balanced and selective: the same 15% cut-off on both sides, so one simple rule. It trades only the stronger signals, yet still holds about 27 stocks on each side, so it stays diversified. C, by contrast, shorts only about 18 stocks. " +
    "Second, D's alpha of 6.0% a year is statistically significant, p 0.045. E's is not, p 0.10: most of E's extra return is market risk, with a worst fall of 37.5%, deeper than the market. " +
    "Third, D does not go too narrow: F and G hold only about 9 stocks on the short side, which adds risk and loses significance. " +
    "Fourth, it is robust: B with the pre-set 20% cut-offs and C give almost the same result, so D does not hinge on a lucky cut-off. " +
    "To be honest, the 15% cut-off was chosen after seeing earlier results, so D's numbers are slightly flattered; B, which uses the pre-set cut-offs, confirms the result, and A remains the cleanest scientific test.");
  tracker(s, 7);
  const rs = [
    ["1", "Balanced and selective", "**Same 15% cut-off on both sides**: trades only stronger signals, yet ~27 stocks per side keep it diversified", COL.green],
    ["2", "Significant alpha", "**+6.0%/yr, p 0.045**. E's +18.1% is mostly market risk: alpha p 0.10, worst fall −37.5%", COL.blue],
    ["3", "Not too narrow", "F and G hold only ~9 stocks short: more risk, same return, alpha not significant", COL.amber],
    ["4", "Robust, honestly", "B (pre-set 20%) and C give ≈ the same: D doesn't hinge on a lucky cut-off. 15% was picked after earlier results", COL.navy],
  ];
  rs.forEach(([n, h, b, c], i) => {
    const col = i % 2, row = Math.floor(i / 2);
    const x = MX + col * 6.18, y = TOP + 0.05 + row * 1.85, w = 5.95;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h: 1.7, rectRadius: 0.08, fill: { color: COL.tint },
      line: { color: COL.tint }, objectName: `why-${i}` });
    badge(s, x + 0.25, y + 0.25, n, c, 0.6);
    txt(s, [{ text: h, options: { bold: true, fontSize: 19, color: COL.navy, breakLine: true } }, ...rich(b, { fontSize: 15 })],
      { x: x + 1.05, y: y + 0.15, w: w - 1.25, h: 1.45 });
  });
  table(s, ["", "A", "B", "D", "E"], [["Return / yr", "+6.9%", "+12.9%", "+13.3%", "+18.1%"],
    ["Role", "main scientific test", "pre-set cut-offs", "our choice", "most return, full risk"]],
    { y: 5.55, colW: [1.9, 2.4, 2.6, 2.4, 2.83], numFrom: 1, fontSize: 15, rowH: 0.4, noSign: false });
}
{
  const s = slide("CONTENT", "Verdict", "Did we beat the market?",
    "On raw return, the tilted strategies roughly tie the S&P 500 and long-only beats it. On a risk-adjusted basis, yes, clearly: every strategy has a higher Sharpe ratio than the market. " +
    "A to D and G have much smaller crashes. And there is an alpha of about 6% a year, statistically significant for B, C and D. " +
    "Even long-only, the score adds value: plus 4.8% a year over buying every call, p 0.044.");
  tracker(s, 7);
  const v = [
    ["Higher raw return than the S&P 500?", "Tie for B–G (≈ +13% vs +12.8%); E higher; A lower but market-free", COL.amber, "≈"],
    ["Better return per unit of risk?", "Yes: Sharpe 0.74–0.97 vs 0.63", COL.green, "✓"],
    ["Smaller crashes?", "Yes for A–D and G: −14% to −23% vs −34%", COL.green, "✓"],
    ["Return the market can't explain?", "Yes: alpha ≈ 6%/yr, significant for B, C, D", COL.green, "✓"],
    ["Does the score add value even long-only?", "Yes: +4.8%/yr over buying every call (p 0.044)", COL.green, "✓"],
  ];
  v.forEach(([q, a, c, m], i) => {
    const y = TOP + 0.05 + i * 0.98;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: MX, y, w: CW, h: 0.85, rectRadius: 0.08, fill: { color: COL.tint },
      line: { color: COL.tint }, objectName: `verdict-${i}` });
    badge(s, MX + 0.2, y + 0.15, m, c, 0.55);
    txt(s, q, { x: MX + 1.0, y, w: 5.2, h: 0.85, fontSize: 18, bold: true, color: COL.navy, valign: "middle" });
    txt(s, a, { x: MX + 6.3, y, w: 5.7, h: 0.85, fontSize: 17, valign: "middle" });
  });
}
{
  const s = slide("CONTENT", "Verdict", "A good finding, with honest limits",
    "Is it a good finding? As research, yes. The signal is out of sample, it survives costs, it survives controls, it disappears without language, and FinBERT beats VADER. " +
    "But the limits: the test covers about 7.4 years and 198 large US stocks; costs are assumed, with no taxes; several variants were tested so the best one is slightly flattered; " +
    "FinBERT was used as is, not retrained on earnings calls; t-statistics are around 2, not 3; and survivorship bias and 2020 matter. " +
    "A Sharpe near one is good, not extraordinary. This is evidence, not a money machine.");
  tracker(s, 7);
  card(s, MX, TOP + 0.05, 5.9, 4.4, "Why it holds up", ["Every score is **out-of-sample** (walk-forward)",
    "Survives costs: break-even **46 bps**", "Survives controls: **+1.31%**, p 0.013", "**Disappears without language** (placebo)",
    "FinBERT beats VADER at every stage"], { size: 19, color: COL.green });
  card(s, MX + 6.23, TOP + 0.05, 5.9, 4.4, "Honest limits", ["~7.4 test years, 198 large US stocks",
    "Assumed costs, no taxes; survivorship bias", "Several variants tested: the best is **flattered**",
    "FinBERT used as is, not retrained", "t-statistics ≈ 2, not 3; Sharpe ≈ 1 is good, not extraordinary"], { size: 19, color: COL.red });
  callout(s, "**Evidence**, not a money machine: markets under-react to the language of earnings calls", MX, 6.15, CW, 0.65, COL.navy);
}
{
  const s = slide("CONTENT", "Game theory", "Mapping the results back to the game",
    "Back to game theory. The firm's type is private. The call is cheap talk, free to send, so managers can spin it. " +
    "But some parts are harder to fake: the live answers and especially the analysts' reaction, which is exactly where our model puts the most weight. " +
    "The classifier acts as a Bayesian receiver: its score is a belief, roughly the probability the firm is good given its words. " +
    "Strategies A to G are threshold strategies: act when the belief crosses a cut-off, and size the bet by risk appetite. " +
    "The drift gap shows the real market updating too little at first, so a sharper receiver profits. This is an interpretation, not a formal equilibrium estimation.");
  tracker(s, 7);
  table(s, ["Game piece", "In our project"], [
    ["Private type", "firm truly good or bad (revealed only later, in the drift)"],
    ["Cheap talk", "the call: free to send, can be spun"],
    ["Hard-to-fake signal", "live Q&A and **analyst tone**: the biggest model weight"],
    ["Bayesian receiver", "the classifier: score ≈ P(good | words)"],
    ["Receiver strategy", "threshold rules A–G: act when belief crosses a cut-off"],
    ["Slow updating", "the drift gap: the market updates too little at first"],
  ], { y: TOP + 0.05, colW: [3.2, 8.93], fontSize: 17, rowH: 0.6 });
  txt(s, "An interpretation using signalling and cheap-talk theory, not a formal game-theoretic estimation (no equilibrium is solved).",
    { x: MX, y: 6.0, w: CW, h: 0.6, fontSize: 15, color: COL.gray });
}

// ---------- close
newSection("Close");
{
  const s = slide("CONTENT", "Summary", "The whole project in one line",
    "To finish: 12,789 earnings calls were turned into FinBERT language features. A walk-forward logistic regression turned them into one score per call, with an AUC of 0.611 on unseen years. " +
    "High-score calls kept beating low-score calls by 1.62% over the next three months, so the market under-reacts to the language of earnings calls. " +
    "A simple strategy that trades on this earns about 6% a year of alpha after costs; our chosen Strategy D earns 13.3% a year with a better Sharpe and smaller crashes than the S&P 500.");
  tracker(s, 7);
  flow(s, [{ t: "12,789 calls", s: "Phase 1" }, { t: "FinBERT features", s: "Phase 2" }, { t: "Score, AUC 0.611", s: "Phase 3", kind: "hi" },
           { t: "Drift gap +1.62%", s: "Phase 4", kind: "hi" }, { t: "D: +13.3%/yr", s: "Phase 5", kind: "up" }], TOP + 0.15, { h: 1.2, size: 18 });
  card(s, MX, 3.45, 3.9, 2.9, "Q1  Can a computer read it?", "**Yes.** Language predicts the market's reaction: AUC 0.611 on unseen years.", { size: 17 });
  card(s, MX + 4.12, 3.45, 3.9, 2.9, "Q2  Is the market slow?", "**Yes.** About 29% of the move comes later; drift gap +1.62% (p 0.002).", { size: 17, color: COL.green });
  card(s, MX + 8.24, 3.45, 3.89, 2.9, "Q3  Can we profit?", "**Yes, modestly.** ≈ 6%/yr alpha after costs; D beats the S&P's return with a much smaller worst crash.", { size: 17, color: COL.blue });
}
{
  const s = slide("TITLE", "Thank you", "Questions?",
    "Thank you. We are happy to take questions. The code and results are on GitHub, and the dashboard can be run with streamlit run phase6_dashboard.py.");
  s.addText("Decoding the Game  ·  Kartik Mantri (24UCS246)  ·  Velu Kala (24DCS012)", { placeholder: "body" });
  txt(s, [{ text: "Supervisor: Asst. Prof. Girish Sharma  ·  The LNM Institute of Information Technology", options: { breakLine: true } },
    { text: "Code and results: github.com/KartikMantri/Mini_Project26", options: { bold: true, color: COL.navy } }],
    { x: MX, y: 4.3, w: 11, h: 1.0, fontSize: 18, color: COL.ink });
  flow(s, [{ t: "Data" }, { t: "Language" }, { t: "Score" }, { t: "Market test", kind: "hi" }, { t: "Strategy", kind: "up" }], 5.7,
    { h: 0.75, size: 16, w: 9 });
}

(async () => {
  await pres.writeFile({ fileName: OUT });
  await applyTheme(OUT, THEME);
  console.log("wrote", OUT);
})();
