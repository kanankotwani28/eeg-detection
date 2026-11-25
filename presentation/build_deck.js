// Build the project presentation from the logged results (thesis/values.json, thesis/brain_values.json).
// Run: NODE_PATH=<node_modules with pptxgenjs, sharp, react, react-dom, react-icons> node presentation/build_deck.js
const path = require("path");
const fs = require("fs");
const pptxgen = require("pptxgenjs");
const sharp = require("sharp");
const React = require("react");
const ReactDOMServer = require("react-dom/server");
const Fa = require("react-icons/fa");
const { applyTheme } = require(process.env.PPTX_SKILL + "/scripts/apply_theme.js");

const ROOT = path.resolve(__dirname, "..");
const FIG = path.join(ROOT, "thesis", "figures");
const V = JSON.parse(fs.readFileSync(path.join(ROOT, "thesis", "values.json")));
const BV = JSON.parse(fs.readFileSync(path.join(ROOT, "thesis", "brain_values.json")));
const OUT = path.join(__dirname, "EEG_FL_DP_Presentation_Kanan_Kotwani.pptx");

// ------------------------------------------------------------------ values
const row = (e, m = "logreg", v = null, eps = "inf") =>
  V.summary.find(r => r.experiment === e && r.model === m && r.eps === eps && (v === null || r.variant === v));
const lira = (reg, eps) => V.lira.find(r => r.model === "logreg" && r.regime === reg && r.eps === eps);
const ext = s => V.external.find(r => r.source === s);
const comp = (l, m = "balanced_accuracy") => V.comparisons.find(c => c.comparison === l && c.metric === m);
const f3 = x => x.toFixed(3);
const f2 = x => x.toFixed(2);
const ci = (r, m = "balanced_accuracy") => `${f3(r[m])} [${f3(r[m + "_lo"])}–${f3(r[m + "_hi"])}]`;
const e01 = row("E01"), e02 = row("E02"), e03 = row("E03"), e16 = row("E16"), e22 = row("E22");
const e04 = row("E04", "eegnet", "T1"), e05 = row("E05", "logreg", "iid");
const e06 = row("E06", "logreg", "iid"), e06s = row("E06", "logreg", "label_skew"), e06d = row("E06", "logreg", "iid_dropout");
const e09 = { "1": row("E09", "logreg", null, "1"), "3": row("E09", "logreg", null, "3"), "8": row("E09", "logreg", null, "8"), inf: row("E09") };
const L = { c: lira("central", "inf"), inf: lira("federated", "inf"), "8": lira("federated", "8"), "3": lira("federated", "3"), "1": lira("federated", "1") };
const xc = ext("LR-central"), xf = ext("LR-FedAvg"), xd = ext("LR-FedAvg-DP-eps8");
const probe = V.external.find(r => r.kind === "probe");
const cAge = comp("Band power LR - age-only LR"), cFl = comp("LR FedAvg IID - LR central");
const cDp8 = comp("logreg DP eps=8 - non-DP"), cDp3 = comp("logreg DP eps=3 - non-DP");
const C1 = V.cohort.T1, C2 = V.cohort.T2, QC = V.qc, CC = V.c;

// ------------------------------------------------------------------ theme
const THEME = {
  name: "EEG Privacy Benchmark",
  headFontFace: "Cambria",
  bodyFontFace: "Calibri",
  colors: {
    dk1: "1D2B2E", lt1: "FFFFFF", dk2: "1F3B40", lt2: "EEF3F1",
    accent1: "2F5D62", accent2: "C4703F", accent3: "7FB0A8", accent4: "9B4D2B", accent5: "8A8F94", accent6: "E8C3AC",
    hlink: "2F5D62", folHlink: "9B4D2B",
  },
};
const HEX = THEME.colors;

async function icon(Comp, color = "FFFFFF", size = 256) {
  const svg = ReactDOMServer.renderToStaticMarkup(React.createElement(Comp, { color: "#" + color, size: String(size) }));
  const buf = await sharp(Buffer.from(svg)).png().toBuffer();
  return "image/png;base64," + buf.toString("base64");
}

async function fit(file, bx, by, bw, bh) {
  const m = await sharp(file).metadata();
  const r = Math.min(bw / m.width, bh / m.height);
  const w = m.width * r, h = m.height * r;
  return { path: file, x: bx + (bw - w) / 2, y: by + (bh - h) / 2, w, h };
}

(async () => {
  const pres = new pptxgen();
  pres.layout = "LAYOUT_WIDE"; // 13.333 x 7.5
  pres.author = "Kanan Kotwani";
  pres.title = "Privacy-Preserving Federated Learning for EEG-Based Classification of MDD";
  pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
  const C = pres.SchemeColor;
  const FOOT = "Kanan Kotwani  ·  BTECH/15140/24  ·  Minor Project (AI24300), BIT Mesra Off-Campus Patna";

  pres.defineSlideMaster({
    title: "TITLE_DARK",
    background: { color: HEX.dk2 },
    objects: [
      { placeholder: { options: { name: "title", type: "title", x: 0.7, y: 1.55, w: 8.6, h: 2.2, fontSize: 36, bold: true, color: C.background1, valign: "top", align: "left", fontFace: THEME.headFontFace }, text: "" } },
      { placeholder: { options: { name: "body", type: "body", x: 0.7, y: 4.05, w: 8.6, h: 2.6, fontSize: 16, color: C.background2, valign: "top" }, text: "" } },
    ],
  });
  pres.defineSlideMaster({
    title: "CONTENT",
    background: { color: HEX.lt1 },
    margin: [0.5, 0.6, 0.6, 0.6],
    objects: [
      { placeholder: { options: { name: "title", type: "title", x: 0.6, y: 0.35, w: 12.1, h: 0.85, fontSize: 32, bold: true, color: C.text2, valign: "middle", fontFace: THEME.headFontFace, margin: 0 }, text: "" } },
      { text: { text: FOOT, options: { x: 0.6, y: 7.02, w: 9.5, h: 0.3, fontSize: 10, color: C.accent5, margin: 0 } } },
    ],
    slideNumber: { x: 12.2, y: 7.0, w: 0.5, h: 0.3, fontSize: 10, color: HEX.accent5, align: "right" },
  });
  pres.defineSlideMaster({
    title: "CLOSING_DARK",
    background: { color: HEX.dk2 },
    objects: [
      { placeholder: { options: { name: "title", type: "title", x: 0.6, y: 0.35, w: 12.1, h: 0.85, fontSize: 32, bold: true, color: C.background1, valign: "middle", fontFace: THEME.headFontFace, margin: 0 }, text: "" } },
      { text: { text: FOOT, options: { x: 0.6, y: 7.02, w: 9.5, h: 0.3, fontSize: 10, color: C.accent3, margin: 0 } } },
    ],
    slideNumber: { x: 12.2, y: 7.0, w: 0.5, h: 0.3, fontSize: 10, color: HEX.accent3, align: "right" },
  });

  const caption = (s, text, x, y, w) =>
    s.addText(text, { x, y, w, h: 0.32, fontSize: 11, italic: true, color: C.accent5, margin: 0, isTextBox: true, objectName: "caption" });
  const card = (s, x, y, w, h, fill = C.background2, name = "card") =>
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, fill: { color: fill }, line: { color: fill }, rectRadius: 0.08, objectName: name });
  const stat = (s, x, y, w, big, small, color = C.accent1, fs = 40) => {
    s.addText(big, { x, y, w, h: 0.85, fontSize: fs, bold: true, color, fontFace: THEME.headFontFace, margin: 0, isTextBox: true, objectName: "stat value" });
    s.addText(small, { x, y: y + 0.85, w, h: 0.6, fontSize: 13, color: C.text1, margin: 0, valign: "top", isTextBox: true, objectName: "stat label" });
  };
  const circleIcon = async (s, Comp, x, y, d = 0.62, fill = C.accent1) => {
    s.addShape(pres.shapes.OVAL, { x, y, w: d, h: d, fill: { color: fill }, line: { color: fill }, objectName: "icon circle" });
    s.addImage({ data: await icon(Comp), x: x + d * 0.22, y: y + d * 0.22, w: d * 0.56, h: d * 0.56, objectName: "icon" });
  };

  // ================================================================ 1. Title
  pres.addSection({ title: "Introduction" });
  let s = pres.addSlide({ masterName: "TITLE_DARK", sectionTitle: "Introduction" });
  s.addText("Privacy-Preserving Federated Learning for EEG-Based Classification of Major Depressive Disorder", { placeholder: "title" });
  s.addText([
    { text: "A leakage-safe benchmark on TDBRAIN", options: { italic: true, breakLine: true, color: HEX.accent6 } },
    { text: " ", options: { breakLine: true, fontSize: 8 } },
    { text: "Kanan Kotwani  (BTECH/15140/24)", options: { bold: true, breakLine: true, color: HEX.lt1 } },
    { text: "Guide: Dr. Gyanendra Tiwary", options: { breakLine: true } },
    { text: "Department of Computer Science and Engineering, BIT Mesra, Off-Campus Patna", options: { breakLine: true } },
    { text: "Minor Project (PROJECT-I, AI24300)  ·  Monsoon Semester 2026" },
  ], { placeholder: "body" });
  s.addShape(pres.shapes.OVAL, { x: 9.95, y: 2.0, w: 2.9, h: 2.9, fill: { color: C.background1 }, line: { color: C.background1 }, objectName: "logo backdrop" });
  s.addImage({ path: path.join(FIG, "bit_logo.png"), x: 10.15, y: 2.2, w: 2.5, h: 2.5, objectName: "BIT Mesra logo" });
  s.addNotes("Introduce yourself and the title. One-line summary: I built a reproducible benchmark that measures how much accuracy EEG depression classifiers lose when trained federated and with participant-level differential privacy, how much privacy that actually buys, and how the models transfer to another data set. Stress early: this is a research benchmark, not a diagnostic tool.");

  // ================================================================ 2. Problem statement
  pres.addSection({ title: "Problem and data" });
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Problem and data" });
  s.addText("Problem statement", { placeholder: "title" });
  const probs = [
    [Fa.FaUserSecret, "Leakage inflates accuracy", "Many EEG studies split one person's segments across train and test, so models recognise people, not depression; reported accuracies often exceed 95%."],
    [Fa.FaLock, "EEG cannot simply be pooled", "EEG and psychiatric labels are sensitive health data. Federated learning keeps data on site, but shared updates can still leak."],
    [Fa.FaGlobeAsia, "Single-site results rarely transfer", "Hardware, reference, country and population differ between sites, so a model may fail on a new data set."],
  ];
  for (let i = 0; i < 3; i++) {
    const x = 0.6 + i * 4.1;
    card(s, x, 1.45, 3.85, 2.5);
    await circleIcon(s, probs[i][0], x + 0.25, 1.65);
    s.addText(probs[i][1], { x: x + 1.0, y: 1.65, w: 2.7, h: 0.62, fontSize: 17, bold: true, color: C.text2, valign: "middle", margin: 0, isTextBox: true });
    s.addText(probs[i][2], { x: x + 0.25, y: 2.4, w: 3.4, h: 1.2, fontSize: 15, color: C.text1, valign: "top", margin: 0, isTextBox: true });
  }
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 0.6, y: 4.2, w: 12.1, h: 2.55, fill: { color: C.accent1 }, line: { color: C.accent1 }, rectRadius: 0.08, objectName: "question card" });
  s.addText([
    { text: "Research question", options: { bold: true, fontSize: 18, color: HEX.accent6, breakLine: true } },
    { text: "On TDBRAIN, with evaluation strictly by participant, how much accuracy do EEG-based MDD classifiers lose when trained federated and with participant-level differential privacy, how much privacy does that buy against a membership-inference attack, and how well do the models transfer to an independent data set?", options: { fontSize: 20, color: HEX.lt1 } },
  ], { x: 0.95, y: 4.3, w: 11.4, h: 2.35, valign: "middle", margin: 0, isTextBox: true });
  s.addNotes("Three problems motivate the work. (1) Leakage: segment-level splits let the model memorise individuals (Brookshire et al., 2024). (2) Privacy: hospitals cannot pool EEG; FL helps, but gradients can leak (Zhu et al., 2019), so a formal guarantee is needed. (3) Transfer: single-site models often fail elsewhere. The research question combines all three.\n\nLikely question: why not just report accuracy? Because without subject-level splits and controls the number cannot be trusted.");

  // ================================================================ 3. Dataset
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Problem and data" });
  s.addText("Datasets", { placeholder: "title" });
  card(s, 0.6, 1.45, 4.3, 2.55);
  await circleIcon(s, Fa.FaDatabase, 0.85, 1.62);
  s.addText("TDBRAIN (training and CV)", { x: 1.6, y: 1.62, w: 3.2, h: 0.62, fontSize: 16, bold: true, color: C.text2, valign: "middle", margin: 0, isTextBox: true });
  s.addText([
    { text: "Brainclinics clinical archive, V3 release (BDF, BIDS)", options: { bullet: true, breakLine: true } },
    { text: "26 EEG channels, 500 Hz; 2 min eyes closed used", options: { bullet: true, breakLine: true } },
    { text: `${CC.first_session} participants; first session only`, options: { bullet: true, breakLine: true } },
    { text: "Access: ORCID login + data-use agreement", options: { bullet: true } },
  ], { x: 0.85, y: 2.35, w: 3.95, h: 1.6, fontSize: 13, color: C.text1, paraSpaceAfter: 3, margin: 0, valign: "top", isTextBox: true });
  card(s, 0.6, 4.2, 4.3, 2.55);
  await circleIcon(s, Fa.FaExternalLinkAlt, 0.85, 4.37, 0.62, C.accent2);
  s.addText("Mumtaz (external test only)", { x: 1.6, y: 4.37, w: 3.2, h: 0.62, fontSize: 16, bold: true, color: C.text2, valign: "middle", margin: 0, isTextBox: true });
  s.addText([
    { text: "Public figshare set: 34 MDD, 30 healthy", options: { bullet: true, breakLine: true } },
    { text: "19 channels, 256 Hz, linked-ear reference", options: { bullet: true, breakLine: true } },
    { text: `${xc.n} participants with eyes-closed data`, options: { bullet: true, breakLine: true } },
    { text: "Opened once, after all training", options: { bullet: true } },
  ], { x: 0.85, y: 5.1, w: 3.95, h: 1.6, fontSize: 13, color: C.text1, paraSpaceAfter: 3, margin: 0, valign: "top", isTextBox: true });
  // stats
  stat(s, 5.3, 1.45, 2.3, String(CC.T1_n), `T1: MDD vs healthy\n(${C1.MDD.n} vs ${C1.HEALTHY.n})`);
  stat(s, 7.7, 1.45, 2.3, String(CC.T2_n), `T2: MDD vs other\npatients (${C1.MDD.n} vs ${C2.OTHER.n})`, C.accent2);
  stat(s, 5.3, 3.15, 2.3, "22,170", "clean 4-s epochs\n(19 × 1,024 samples)");
  stat(s, 7.7, 3.15, 2.3, `${(100 * QC.n_excluded / QC.n_participants).toFixed(1)}%`, `recordings excluded\n(${QC.n_excluded} of ${QC.n_participants})`, C.accent2);
  s.addText(`Healthy group is ${(C1.MDD.age_mean - C1.HEALTHY.age_mean).toFixed(1)} years younger on average → age-only control (E22)`, { x: 5.3, y: 5.0, w: 4.7, h: 0.8, fontSize: 13, italic: true, color: C.accent4, margin: 0, valign: "top", isTextBox: true });
  const mont = await fit(path.join(FIG, "fig_montage.png"), 10.2, 1.5, 2.5, 3.6);
  s.addImage({ ...mont, objectName: "montage figure" });
  caption(s, "Thesis Fig. 4.1: 19 shared channels (filled)", 10.0, 5.25, 2.9);
  s.addNotes(`TDBRAIN V3 was obtained under its data-use agreement. Only first sessions, adults 18–65, and labelled participants (UNKNOWN and blinded excluded). T1 = MDD vs healthy (${CC.T1_n}), T2 = MDD vs other diagnosed patients (${CC.T2_n}). Preprocessing kept 22,170 clean 4-second epochs; only ${QC.n_excluded} recordings were excluded. Mumtaz is used only as a one-time external test. The 19 channels are those both data sets share — no channel is invented by interpolation.\n\nLikely question: why only 19 channels? To make the external test possible without fabricating channels.`);

  // ================================================================ 4. Pipeline
  pres.addSection({ title: "Method" });
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Method" });
  s.addText("Pipeline of work", { placeholder: "title" });
  const arch = await fit(path.join(FIG, "fig_architecture.png"), 0.6, 1.4, 7.2, 4.9);
  s.addImage({ ...arch, objectName: "architecture figure" });
  caption(s, "Thesis Fig. 3.1: six-layer organisation of the benchmark", 0.6, 6.4, 7.2);
  const steps = [
    ["Index", "First-session cohort, T1/T2 labels"],
    ["Split", "5-fold by participant, hashed; 4 clients (IID / label-skew)"],
    ["Preprocess", "19 ch, 1–40 Hz, CAR, 256 Hz, 4-s epochs, reject > 150 µV"],
    ["Model", "Band power + FAA → LR;  EEGNet-8,2 + GroupNorm"],
    ["Train", "Centralised · local · FedAvg · FedAvg + participant DP"],
    ["Evaluate", "Bootstrap CIs, LiRA audit, external test once"],
  ];
  for (let i = 0; i < steps.length; i++) {
    const y = 1.45 + i * 0.84;
    s.addShape(pres.shapes.OVAL, { x: 8.15, y, w: 0.56, h: 0.56, fill: { color: i === 5 ? C.accent2 : C.accent1 }, line: { color: i === 5 ? C.accent2 : C.accent1 }, objectName: "step number" });
    s.addText(String(i + 1), { x: 8.15, y, w: 0.56, h: 0.56, fontSize: 16, bold: true, color: C.background1, align: "center", valign: "middle", margin: 0, isTextBox: true });
    s.addText([{ text: steps[i][0], options: { bold: true, breakLine: true, color: HEX.dk2 } }, { text: steps[i][1], options: { fontSize: 13 } }],
      { x: 8.9, y: y - 0.08, w: 3.85, h: 0.76, fontSize: 15, color: C.text1, margin: 0, valign: "middle", isTextBox: true });
  }
  s.addNotes("Walk through the six steps. Key design rules: splits are made once and hashed so every experiment uses identical participants; nothing (scaler, threshold, C, early stopping) is fitted on test participants; preprocessing uses only each recording's own data; Mumtaz is read by one module only. Every run is logged to MLflow, and the report and dashboard are generated from those logs, so no number is typed by hand.\n\nLikely question: how do you know there is no leakage? Shuffled-label control (E01) gave chance, and automated tests assert no participant overlap.");

  // ================================================================ 5. Federated + DP method
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Method" });
  s.addText("Federated learning with participant-level privacy", { placeholder: "title" });
  const reg = await fit(path.join(FIG, "fig_regimes.png"), 0.6, 1.35, 6.3, 2.2);
  s.addImage({ ...reg, objectName: "regimes figure" });
  caption(s, "Thesis Fig. 3.4: the four training regimes", 0.6, 3.6, 6.3);
  const fld = await fit(path.join(FIG, "fig_fl_dp.png"), 0.6, 4.0, 6.3, 2.5);
  s.addImage({ ...fld, objectName: "FL DP figure" });
  caption(s, "Thesis Fig. 4.9: one federated round with participant-level DP", 0.6, 6.55, 6.3);
  card(s, 7.3, 1.45, 5.4, 5.3);
  s.addText([
    { text: "Participant-level DP-SGD", options: { bold: true, fontSize: 17, color: HEX.dk2, breakLine: true } },
    { text: "Each local step samples people (q = 0.25), not epochs", options: { bullet: true, breakLine: true } },
    { text: "Each person's gradient clipped to C = 1, Gaussian noise added", options: { bullet: true, breakLine: true } },
    { text: `ε ∈ {1, 3, 8} at δ = 10⁻⁵ via Opacus RDP accountant (σ = ${f2(e09["8"].sigma_mean)} for ε = 8)`, options: { bullet: true, breakLine: true } },
    { text: "Reported ε = max over clients", options: { bullet: true, breakLine: true } },
    { text: " ", options: { breakLine: true, fontSize: 8 } },
    { text: "Privacy audit (LiRA)", options: { bold: true, fontSize: 17, color: HEX.dk2, breakLine: true } },
    { text: "16 models per setting, each on a random half of participants", options: { bullet: true, breakLine: true } },
    { text: "Likelihood ratio of a person's confidence: member or not?", options: { bullet: true, breakLine: true } },
    { text: "Report attack AUC and TPR at 1% / 5% FPR", options: { bullet: true } },
  ], { x: 7.6, y: 1.65, w: 4.9, h: 4.95, fontSize: 14, color: C.text1, paraSpaceAfter: 4, margin: 0, valign: "top", isTextBox: true });
  s.addNotes("Why participant-level? One person contributes ~30 epochs; per-epoch DP would protect an epoch, not a person, and would report a misleadingly small ε. Here sampling, clipping and accounting are all per participant (McMahan et al., 2018). FedAvg is run with Flower's simulation engine; an in-process engine with identical client code was verified to give identical weights. LiRA (Carlini et al., 2022) tests the guarantee empirically.\n\nLikely question: what does ε = 8 mean? The probability of any output changes by at most a factor e^8 when one person is added or removed (with δ = 1e-5); smaller ε is stronger protection.");

  // ================================================================ 6. Scope
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Method" });
  s.addText("Scope", { placeholder: "title" });
  card(s, 0.6, 1.45, 5.9, 3.85);
  await circleIcon(s, Fa.FaCheck, 0.85, 1.62);
  s.addText("In scope", { x: 1.6, y: 1.62, w: 4.5, h: 0.62, fontSize: 18, bold: true, color: C.text2, valign: "middle", margin: 0, isTextBox: true });
  s.addText([
    { text: "Resting eyes-closed EEG; MDD vs healthy (primary) and MDD vs other patients", options: { bullet: true, breakLine: true } },
    { text: "Two model tracks under four training regimes", options: { bullet: true, breakLine: true } },
    { text: "Participant-level DP and an attack-based privacy audit", options: { bullet: true, breakLine: true } },
    { text: "One-time external test and a data-set identity probe", options: { bullet: true, breakLine: true } },
    { text: "Reproducible code, 42 automated tests, MLflow, dashboard", options: { bullet: true } },
  ], { x: 0.9, y: 2.4, w: 5.4, h: 2.8, fontSize: 15, color: C.text1, paraSpaceAfter: 8, margin: 0, valign: "top", isTextBox: true });
  card(s, 6.8, 1.45, 5.9, 3.85);
  await circleIcon(s, Fa.FaTimes, 7.05, 1.62, 0.62, C.accent2);
  s.addText("Out of scope", { x: 7.8, y: 1.62, w: 4.5, h: 0.62, fontSize: 18, bold: true, color: C.text2, valign: "middle", margin: 0, isTextBox: true });
  s.addText([
    { text: "Clinical diagnosis or treatment advice — this is not a diagnostic tool", options: { bullet: true, breakLine: true } },
    { text: "Real hospitals: the 4 clients are simulated partitions of one archive", options: { bullet: true, breakLine: true } },
    { text: "New data collection; secure aggregation", options: { bullet: true, breakLine: true } },
    { text: "Not yet run: full-scale EEGNet (only fold 0), E12 leakage demo, FedProx/FedBN, ablations", options: { bullet: true } },
  ], { x: 7.1, y: 2.4, w: 5.4, h: 2.8, fontSize: 15, color: C.text1, paraSpaceAfter: 8, margin: 0, valign: "top", isTextBox: true });
  s.addText("Experiments run on real TDBRAIN data", { x: 0.6, y: 5.5, w: 12.1, h: 0.4, fontSize: 16, bold: true, color: C.text2, margin: 0, isTextBox: true });
  const chips = [["E01", "shuffled-label control"], ["E02", "spectral baseline"], ["E03", "MDD vs other"], ["E04*", "EEGNet"], ["E05", "local only"], ["E06", "FedAvg"],
    ["E09", "FedAvg + DP"], ["E10", "LiRA audit"], ["E11", "external test"], ["E13", "data-set probe"], ["E16", "FAA only"], ["E22", "age only"]];
  for (let i = 0; i < chips.length; i++) {
    const x = 0.6 + i * 1.015;
    const fill = chips[i][0].endsWith("*") ? C.accent2 : C.accent1;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: 6.0, w: 0.93, h: 0.82, fill: { color: fill }, line: { color: fill }, rectRadius: 0.06, objectName: "experiment chip" });
    s.addText([{ text: chips[i][0], options: { bold: true, fontSize: 14, breakLine: true } }, { text: chips[i][1], options: { fontSize: 9 } }],
      { x, y: 6.0, w: 0.93, h: 0.82, color: C.background1, align: "center", valign: "middle", margin: 2, isTextBox: true });
  }
  s.addNotes("Be explicit about boundaries. Strengths: a complete benchmark with controls. Honest limits: clients are simulated; EEGNet ran on one fold only under a CPU budget; several optional experiments are implemented but not yet run. Future work: real second site such as AIIMS after ethics approval. The chips show the 12 experiments run on real data; E04 (orange, *) ran on fold 0 only.");

  // ================================================================ 7. Tech stack
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Method" });
  s.addText("Tech stack", { placeholder: "title" });
  const pk = V.packages || {};
  const tech = [
    [Fa.FaPython, "Python 3.11 + uv", "Locked, reproducible environment"],
    [Fa.FaBrain, "MNE-Python " + (pk.mne || ""), "Read BDF/EDF, filter, re-reference"],
    [Fa.FaCogs, "PyTorch " + (pk.torch || "").split("+")[0], "EEGNet, per-participant gradients"],
    [Fa.FaProjectDiagram, "Flower " + (pk.flwr || ""), "Federated simulation (FedAvg)"],
    [Fa.FaShieldAlt, "Opacus " + (pk.opacus || ""), "RDP privacy accountant"],
    [Fa.FaChartLine, "scikit-learn " + (pk["scikit-learn"] || ""), "Splits, logistic regression, metrics"],
    [Fa.FaClipboardList, "MLflow " + (pk.mlflow || ""), `Tracked ${V.n_runs} runs with hashes`],
    [Fa.FaDesktop, "Streamlit + Plotly", "Read-only 8-page dashboard"],
  ];
  for (let i = 0; i < tech.length; i++) {
    const col = i % 4, r = Math.floor(i / 4);
    const x = 0.6 + col * 3.05, y = 1.5 + r * 2.65;
    card(s, x, y, 2.85, 2.4);
    await circleIcon(s, tech[i][0], x + 0.25, y + 0.25, 0.7, i % 2 ? C.accent2 : C.accent1);
    s.addText(tech[i][1], { x: x + 0.25, y: y + 1.05, w: 2.45, h: 0.5, fontSize: 15, bold: true, color: C.text2, margin: 0, valign: "middle", isTextBox: true });
    s.addText(tech[i][2], { x: x + 0.25, y: y + 1.55, w: 2.45, h: 0.7, fontSize: 13, color: C.text1, margin: 0, valign: "top", isTextBox: true });
  }
  s.addNotes("Mention reproducibility: every MLflow run stores the config hash, split hash, participants-table hash and git commit. pytest covers splits, preprocessing, no-test-fitting, DP clipping/accounting, FL equivalence, LiRA, and Mumtaz isolation. Docker image and Makefile (make reproduce) are provided; data are never committed.");

  // ================================================================ 8. Results: accuracy
  pres.addSection({ title: "Results" });
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Results" });
  s.addText("Results: accuracy across regimes (T1, n = " + e02.n_participants + ")", { placeholder: "title" });
  const bars = [["Shuffled labels (E01)", e01], ["FAA only (E16)", e16], ["Age only (E22)", e22], ["Local-only site (E05)", e05],
    ["FedAvg + DP ε=8 (E09)", e09["8"]], ["FedAvg label-skew (E06)", e06s], ["FedAvg IID (E06)", e06], ["Centralised (E02)", e02]];
  s.addChart(pres.charts.BAR, [{ name: "Balanced accuracy", labels: bars.map(b => b[0]), values: bars.map(b => +b[1].balanced_accuracy.toFixed(3)) }], {
    x: 0.5, y: 1.35, w: 7.8, h: 5.25, barDir: "bar",
    chartColors: [HEX.accent1], showValue: true, dataLabelPosition: "outEnd", dataLabelFontSize: 12, dataLabelColor: HEX.dk1, dataLabelFormatCode: "0.000",
    dataLabelFontFace: "+mn-lt", catAxisLabelFontFace: "+mn-lt", valAxisLabelFontFace: "+mn-lt", titleFontFace: "+mn-lt",
    catAxisLabelFontSize: 12, valAxisLabelFontSize: 11, catAxisLabelColor: HEX.dk1, valAxisLabelColor: HEX.accent5,
    valAxisMinVal: 0.4, valAxisMaxVal: 0.8, valAxisMajorUnit: 0.1, valGridLine: { color: "E3E7E5", size: 0.5 }, catGridLine: { style: "none" },
    showLegend: false, showTitle: true, title: "Participant-level balanced accuracy (0.5 = chance)", titleFontSize: 13, titleColor: HEX.dk2,
    objectName: "accuracy chart",
  });
  caption(s, "Generated from logged MLflow runs (same values as thesis Table 6.1/6.2); 95% CIs in notes", 0.6, 6.62, 7.8);
  stat(s, 8.75, 1.45, 3.9, f3(e02.balanced_accuracy), `centralised balanced accuracy\nAUC ${f3(e02.auc)}, 95% CI ${f3(e02.balanced_accuracy_lo)}–${f3(e02.balanced_accuracy_hi)}`);
  card(s, 8.75, 3.25, 3.95, 3.45);
  s.addText([
    { text: `FedAvg costs only ${f3(-cFl.diff)} BA; label skew and client dropout hardly matter`, options: { bullet: true, breakLine: true } },
    { text: `Single site loses ${f3(e02.balanced_accuracy - e05.balanced_accuracy)} BA → federation pays off`, options: { bullet: true, breakLine: true } },
    { text: `Age alone reaches ${f3(e22.balanced_accuracy)}; band power adds +${f3(cAge.diff)}`, options: { bullet: true, breakLine: true } },
    { text: `FAA alone ≈ chance (${f3(e16.balanced_accuracy)}); T2 is harder (${f3(e03.balanced_accuracy)})`, options: { bullet: true } },
  ], { x: 9.0, y: 3.45, w: 3.5, h: 3.1, fontSize: 14, color: C.text1, paraSpaceAfter: 6, margin: 0, valign: "top", isTextBox: true });
  s.addNotes(`95% bootstrap CIs (2,000 resamples, participant level): Centralised ${ci(e02)}; FedAvg IID ${ci(e06)}; FedAvg label-skew ${ci(e06s)}; FedAvg dropout ${ci(e06d)}; Local-only ${ci(e05)}; DP ε=8 ${ci(e09["8"])}; Age only ${ci(e22)}; FAA only ${ci(e16)}; Shuffled ${ci(e01)}; T2 ${ci(e03)}.\nEEGNet (fold 0 only, preliminary): BA ${f3(e04.balanced_accuracy)}, AUC ${f3(e04.auc)} — not better than the features under this budget.\n\nLikely question: is 0.74 good? TDBRAIN curators treat >0.60 as success; but part of it is age (E22), so it is moderate separation, not a diagnostic test.`);

  // ================================================================ 9. Results: privacy
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Results" });
  s.addText("Results: privacy cost and privacy gained", { placeholder: "title" });
  const pv = await fit(path.join(FIG, "fig_privacy.png"), 0.6, 1.3, 8.2, 3.4);
  s.addImage({ ...pv, objectName: "privacy figure" });
  caption(s, "Thesis Fig. 6.4: accuracy (left) and LiRA attack AUC (right) vs participant-level ε", 0.6, 4.75, 8.2);
  stat(s, 9.2, 1.45, 3.5, `−${f3(-cDp8.diff)}`, `BA cost of DP at ε = 8\n(−${f3(-cDp3.diff)} at ε = 3)`, C.accent2);
  stat(s, 9.2, 3.05, 3.5, f3(L["8"].auc), `LiRA attack AUC at ε = 8\n(${f3(L.inf.auc)} without DP; 0.5 = chance)`);
  s.addTable([
    [{ text: "Setting", options: { bold: true, color: HEX.lt1, fill: { color: HEX.accent1 } } }, { text: "Attack AUC", options: { bold: true, color: HEX.lt1, fill: { color: HEX.accent1 } } }, { text: "TPR @ 1% FPR", options: { bold: true, color: HEX.lt1, fill: { color: HEX.accent1 } } }],
    ["Centralised, no DP", f3(L.c.auc), f3(L.c["tpr_at_fpr_0.01"])],
    ["Federated, no DP", f3(L.inf.auc), f3(L.inf["tpr_at_fpr_0.01"])],
    ["Federated, ε = 8", f3(L["8"].auc), f3(L["8"]["tpr_at_fpr_0.01"])],
    ["Federated, ε = 3", f3(L["3"].auc), f3(L["3"]["tpr_at_fpr_0.01"])],
    ["Federated, ε = 1", f3(L["1"].auc), f3(L["1"]["tpr_at_fpr_0.01"])],
  ], { x: 0.6, y: 5.15, w: 8.2, colW: [3.4, 2.4, 2.4], fontSize: 12, color: HEX.dk1, rowH: 0.27, margin: [1, 4, 1, 4], border: { type: "solid", pt: 0.5, color: "D5DDD9" }, objectName: "LiRA table" });
  s.addText(`Federation alone barely reduces leakage; DP brings the attack to chance. ${L.inf.n_nonmember_evals.toLocaleString("en-US")} non-member evaluations per setting.`, { x: 9.2, y: 4.75, w: 3.5, h: 1.9, fontSize: 14, italic: true, color: C.accent4, margin: 0, valign: "top", isTextBox: true });
  s.addNotes(`DP utility (BA): ε=1 ${ci(e09["1"])}, ε=3 ${ci(e09["3"])}, ε=8 ${ci(e09["8"])}, ε=∞ ${ci(e09.inf)} (same DP-ready features). Noise multipliers: σ=${f2(e09["1"].sigma_mean)} (ε=1), ${f2(e09["3"].sigma_mean)} (ε=3), ${f2(e09["8"].sigma_mean)} (ε=8).\nLiRA sanity check (random relabelling of non-members) gave AUC ≈ 0.5 everywhere.\n\nLikely question: why did DP LR use different features? A pooled scaler would leak information outside the DP accounting, so a per-participant transform is used at every ε including ∞ for a like-for-like comparison.`);

  // ================================================================ 10. Results: brain maps + external
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Results" });
  s.addText("Results: brain maps and external transfer", { placeholder: "title" });
  const tb = await fit(path.join(FIG, "fig_topo_bands.png"), 0.6, 1.35, 6.6, 5.0);
  s.addImage({ ...tb, objectName: "topography figure" });
  caption(s, `Thesis Fig. 4.6: relative band power, group means (MDD n = ${BV.n_mdd}, healthy n = ${BV.n_hc}) and difference`, 0.6, 6.45, 6.8);
  card(s, 7.55, 1.45, 5.15, 1.95);
  s.addText([
    { text: "Group-level description (not causal)", options: { bold: true, fontSize: 16, color: HEX.dk2, breakLine: true } },
    { text: `Occipital alpha: ${BV.alpha_rel_O_mdd.toFixed(1)}% MDD vs ${BV.alpha_rel_O_hc.toFixed(1)}% healthy`, options: { bullet: true, breakLine: true } },
    { text: `Frontal-midline theta: ${BV.theta_rel_Fz_mdd.toFixed(1)}% vs ${BV.theta_rel_Fz_hc.toFixed(1)}%`, options: { bullet: true, breakLine: true } },
    { text: `Beta: ${BV.beta_rel_mean_mdd.toFixed(1)}% vs ${BV.beta_rel_mean_hc.toFixed(1)}%  (age and medication confound)`, options: { bullet: true } },
  ], { x: 7.8, y: 1.6, w: 4.75, h: 1.75, fontSize: 14, color: C.text1, paraSpaceAfter: 4, margin: 0, valign: "top", isTextBox: true });
  stat(s, 7.6, 3.7, 2.3, f3(probe.auc), "TDBRAIN vs Mumtaz\nhealthy: probe AUC", C.accent2);
  stat(s, 10.0, 3.7, 2.7, `${f2(xc.auc)}–${f2(xd.auc)}`, `external AUC on\nMumtaz (n = ${xc.n})`, C.accent1, 34);
  s.addText("Data sets are almost perfectly separable from recording characteristics alone, so transfer fails — the hardest open problem.", { x: 7.6, y: 5.35, w: 5.1, h: 1.1, fontSize: 15, italic: true, color: C.accent4, margin: 0, valign: "top", isTextBox: true });
  s.addNotes(`External test (run once, threshold fixed from TDBRAIN): LR central BA ${ci(xc)}, AUC ${ci(xc, "auc")}; LR FedAvg AUC ${f3(xf.auc)}; LR FedAvg+DP ε=8 AUC ${f3(xd.auc)}. Thresholds labelled nearly everyone MDD (sensitivity ${f3(xc.sensitivity)}, specificity ${f3(xc.specificity)}), so the BA is not real transfer. Dataset-identity probe AUC ${ci(probe, "auc")}.\nTopomaps are descriptive group averages only — they explain nothing causally and are not biomarkers.`);

  // ================================================================ 11. Dashboard
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Results" });
  s.addText("Interactive results dashboard (Streamlit)", { placeholder: "title" });
  const d1 = await fit(path.join(__dirname, "dash_overview_crop.png"), 0.6, 1.4, 5.9, 4.95);
  s.addImage({ ...d1, objectName: "dashboard overview screenshot" });
  s.addShape(pres.shapes.RECTANGLE, { x: d1.x, y: d1.y, w: d1.w, h: d1.h, fill: { type: "none" }, line: { color: "C9D3CF", width: 1 }, objectName: "screenshot frame" });
  caption(s, "Dashboard: Overview page — study question and cohort flow (real TDBRAIN runs)", 0.6, 6.45, 6.0);
  const d2 = await fit(path.join(__dirname, "dash_performance_crop.png"), 6.8, 1.4, 5.9, 4.95);
  s.addImage({ ...d2, objectName: "dashboard performance screenshot" });
  s.addShape(pres.shapes.RECTANGLE, { x: d2.x, y: d2.y, w: d2.w, h: d2.h, fill: { type: "none" }, line: { color: "C9D3CF", width: 1 }, objectName: "screenshot frame" });
  caption(s, "Dashboard: Model performance page — BA and AUC with 95% CIs per experiment", 6.8, 6.45, 6.0);
  s.addNotes("The dashboard (make dashboard) has eight pages: overview, cohort and QC, model performance, federated learning, privacy, external test, explainability, reproducibility. It is read-only and reads only the exported results, so it cannot show anything that was not logged. Offer to demo it live if a laptop is available.");

  // ================================================================ 12. Conclusion
  pres.addSection({ title: "Conclusion" });
  s = pres.addSlide({ masterName: "CLOSING_DARK", sectionTitle: "Conclusion" });
  s.addText("Conclusion and future work", { placeholder: "title" });
  const concl = [
    ["Federation is cheap", `FedAvg within ${f3(-cFl.diff)} BA of centralised; robust to skew and dropout`],
    ["Privacy has a measurable price", `Participant-level DP costs ≈ ${f2(-cDp8.diff)}–${f2(-cDp3.diff)} BA and drives LiRA to chance`],
    ["Transfer is the open problem", `External AUC near chance; data sets separable (probe ${f2(probe.auc)})`],
  ];
  for (let i = 0; i < 3; i++) {
    const x = 0.6 + i * 4.1;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: 1.45, w: 3.85, h: 1.9, fill: { color: C.accent1 }, line: { color: C.accent1 }, rectRadius: 0.08, objectName: "finding card" });
    s.addText([{ text: concl[i][0], options: { bold: true, fontSize: 20, color: HEX.accent6, breakLine: true } }, { text: concl[i][1], options: { fontSize: 16, color: HEX.lt1 } }],
      { x: x + 0.3, y: 1.6, w: 3.3, h: 1.6, margin: 0, valign: "top", isTextBox: true });
  }
  s.addText([
    { text: "Limitations", options: { bold: true, fontSize: 17, color: HEX.accent6, breakLine: true } },
    { text: "Simulated clients; age confound; one seed for FL/DP; EEGNet preliminary (fold 0)", options: { bullet: true, breakLine: true } },
    { text: "Research benchmark — not a diagnostic tool", options: { bullet: true } },
  ], { x: 0.6, y: 3.75, w: 5.9, h: 2.9, fontSize: 16, color: C.background1, paraSpaceAfter: 6, margin: 0, valign: "top", isTextBox: true });
  s.addText([
    { text: "Future work", options: { bold: true, fontSize: 17, color: HEX.accent6, breakLine: true } },
    { text: "Full EEGNet runs on a GPU, with FedProx, FedBN, DP and LiRA", options: { bullet: true, breakLine: true } },
    { text: "Age-matched analysis; leakage demo (E12) and ablations", options: { bullet: true, breakLine: true } },
    { text: "Real second site (e.g. AIIMS) after ethics approval; secure aggregation", options: { bullet: true } },
  ], { x: 6.8, y: 3.75, w: 5.9, h: 2.9, fontSize: 16, color: C.background1, paraSpaceAfter: 6, margin: 0, valign: "top", isTextBox: true });
  s.addNotes("Close with the three answers to the research question, then the limitations stated plainly, then future work. Thank the guide and the panel and invite questions.\n\nPossible questions: Why logistic regression and not only deep learning? It sets a reference point; deep-first hides whether deep learning adds anything. Why simulated clients? No multi-site data yet; the code accepts real sites without changes. What would you do with more time? Full EEGNet runs and a real second site.");

  await pres.writeFile({ fileName: OUT });
  await applyTheme(OUT, THEME);
  console.log("wrote", OUT);
})().catch(e => { console.error(e); process.exit(1); });
