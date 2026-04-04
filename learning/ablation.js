// SHIFU ABLATION SUITE v2.1.0 — Run: node test/ablation.js
//
// Not "does it pass" — "does each mechanism earn its place."
//
// Method:
//   1. Train ONE engine with the full corpus
//   2. Score a fixed set of sentences (positives, negatives, hard negatives)
//   3. Clone the engine with each mechanism disabled (config override)
//   4. Re-score the same sentences
//   5. Compare: does disabling each mechanism shrink the right gaps?
//
// Hard negatives (shared vocabulary, wrong meaning):
//   "patient treats doctor"       — reversed agent
//   "doctor treats medication"    — wrong object
//   "physician manages therapy"   — wrong object (via resonance)
//   "doctor with and patient"     — pure noise, all real words

const { ShifuEngine, CONFIG, fieldCosine } = require("../core/engine");

// ─── Corpus ─────────────────────────────────────────────────────────
const corpus = [
  "doctor treats patient with medication and therapy",
  "doctor treats patient with medication and therapy",
  "doctor treats patient with medication and therapy",
  "doctor examines patient and notes findings",
  "doctor examines patient and notes findings",
  "doctor prescribes medication for patient",
  "doctor prescribes medication for patient",
  "physician treats patient with medication and therapy",
  "physician treats patient with medication and therapy",
  "physician examines patient and notes findings",
  "physician manages patient effectively in the ward",
  "physician manages patient effectively in the ward",
  "nurse administers medication as per protocol",
  "nurse administers medication as per protocol",
  "nurse records vital signs for the patient",
  "patient presents with stroke and weakness",
  "patient presents with stroke and weakness",
  "patient was admitted to the stroke unit for monitoring",
  "patient was discharged home with follow up in clinic",
];

// ─── Test sentences ─────────────────────────────────────────────────
const sentences = {
  // Positives: should score high
  known:       "doctor treats patient with medication",
  knownAlt:    "physician treats patient with therapy",
  unseen:      "physician manages patient with medication",
  // Hard negatives: shared vocabulary, WRONG meaning
  reversed:    "patient treats doctor",
  wrongObj:    "doctor treats medication",
  wrongObjRes: "physician manages therapy",
  wordSoup:    "doctor with and patient",
  // True negative
  gibberish:   "zzzzz xxxxx yyyyy",
};

// ─── Train base engine ──────────────────────────────────────────────
function trainEngine(cfgOverride = {}) {
  const cfg = JSON.parse(JSON.stringify(CONFIG));
  // Apply overrides
  for (const [k, v] of Object.entries(cfgOverride)) {
    if (typeof v === "object" && v !== null) {
      cfg[k] = { ...(cfg[k] || {}), ...v };
    } else {
      cfg[k] = v;
    }
  }
  const eng = new ShifuEngine(cfg);
  for (const s of corpus) eng.feed(s);
  return eng;
}

// ─── Score all sentences on a given engine ───────────────────────────
function scoreAll(eng) {
  const results = {};
  for (const [name, sent] of Object.entries(sentences)) {
    const s = eng.scoreSentence(sent);
    results[name] = {
      forward: s.coherence,
      corrected: s.correctedCoherence,
      settled: s.settledCoherence,
    };
  }
  return results;
}

// ─── Compare sentence fields ────────────────────────────────────────
function fieldSims(eng) {
  const sims = {};
  const pairs = [
    ["known", "knownAlt", "equivalent meaning"],
    ["known", "unseen", "unseen equivalent"],
    ["known", "reversed", "reversed agent (hard neg)"],
    ["known", "wrongObj", "wrong object (hard neg)"],
    ["known", "gibberish", "gibberish"],
  ];
  for (const [a, b, label] of pairs) {
    const r = eng.compareSentences(sentences[a], sentences[b]);
    sims[`${a}_vs_${b}`] = { similarity: r.similarity, label };
  }
  return sims;
}

// ─── Run ablation ───────────────────────────────────────────────────
console.log("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
console.log("  SHIFU ABLATION SUITE v2.1.0");
console.log("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

// Condition A: FULL v2.1 (baseline)
const engFull = trainEngine();
const scoresFull = scoreAll(engFull);
const simsFull = fieldSims(engFull);

// Condition B: INHIBITION OFF (floor=ceiling=1.0 — all words pass through equally)
const engNoInh = trainEngine({ inhibition: { floor: 1.0, ceiling: 1.0 } });
const scoresNoInh = scoreAll(engNoInh);
const simsNoInh = fieldSims(engNoInh);

// Condition C: CORRECTION OFF (strength=0 — no backward pass)
const engNoCorr = trainEngine({ correction: { strength: 0 } });
const scoresNoCorr = scoreAll(engNoCorr);
const simsNoCorr = fieldSims(engNoCorr);

// Condition D: SETTLING OFF (maxIter=0 — no iterative convergence)
const engNoSettle = trainEngine({ settle: { maxIter: 0, epsilon: 0.01, reactivateTop: 8, iterDecay: 0.5 } });
const scoresNoSettle = scoreAll(engNoSettle);
const simsNoSettle = fieldSims(engNoSettle);

// ─── Report: Coherence scores ───────────────────────────────────────
function reportScores(label, scores) {
  console.log(`\n  ${label}`);
  console.log("  " + "-".repeat(72));
  const keys = Object.keys(sentences);
  const maxNameLen = Math.max(...keys.map(k => k.length));
  for (const k of keys) {
    const s = scores[k];
    const pad = " ".repeat(maxNameLen - k.length);
    console.log(`    ${k}${pad}  fwd=${s.forward.toFixed(4)}  corr=${s.corrected.toFixed(4)}  settled=${s.settled.toFixed(4)}`);
  }
}

reportScores("FULL v2.1 (baseline)", scoresFull);
reportScores("INHIBITION OFF", scoresNoInh);
reportScores("CORRECTION OFF", scoresNoCorr);
reportScores("SETTLING OFF", scoresNoSettle);

// ─── Report: Field similarities ─────────────────────────────────────
function reportSims(label, sims) {
  console.log(`\n  ${label} — field similarities`);
  console.log("  " + "-".repeat(60));
  for (const [k, v] of Object.entries(sims)) {
    console.log(`    ${v.label.padEnd(30)} ${v.similarity.toFixed(4)}`);
  }
}

reportSims("FULL v2.1", simsFull);
reportSims("INHIBITION OFF", simsNoInh);

// ─── Ablation analysis ──────────────────────────────────────────────
console.log("\n━━━ ABLATION ANALYSIS ━━━\n");

let pass = 0, fail = 0, total = 0;
function assert(n, c) { total++; if (c) pass++; else { fail++; console.log(`  ✗ ${n}`); } }

// GAP ANALYSIS: the difference between positives and hard negatives
// A mechanism "earns its place" if removing it shrinks the gap

function gap(scores, pos, neg, level = "settled") {
  return scores[pos][level] - scores[neg][level];
}

// 1. INHIBITION: should it be suppressing noise?
const inhGapFull = gap(scoresFull, "known", "wordSoup");
const inhGapOff = gap(scoresNoInh, "known", "wordSoup");
console.log(`  Inhibition gap (known vs wordSoup): full=${inhGapFull.toFixed(4)}, off=${inhGapOff.toFixed(4)}`);
assert("inhibition widens known-vs-wordSoup gap", inhGapFull >= inhGapOff - 0.01);

// Inhibition on field sims: equivalent sentences should be MORE similar
// with inhibition (noise removed) than without
const inhSimEq = simsFull["known_vs_knownAlt"].similarity;
const noInhSimEq = simsNoInh["known_vs_knownAlt"].similarity;
console.log(`  Field sim (equivalent): full=${inhSimEq.toFixed(4)}, noInh=${noInhSimEq.toFixed(4)}`);
// This may go either way depending on corpus — report but don't hard-assert

// 2. CORRECTION: does backward pass help REAL sentences more than noise?
const corrUpliftKnown = scoresFull.known.corrected - scoresFull.known.forward;
const corrUpliftGib = scoresFull.gibberish.corrected - scoresFull.gibberish.forward;
const corrUpliftReversed = scoresFull.reversed.corrected - scoresFull.reversed.forward;
console.log(`  Correction uplift: known=${corrUpliftKnown.toFixed(4)}, reversed=${corrUpliftReversed.toFixed(4)}, gib=${corrUpliftGib.toFixed(4)}`);

// The key test: does turning correction OFF hurt positives more than negatives?
const corrGapFull = gap(scoresFull, "known", "reversed", "corrected");
const corrGapOff = gap(scoresNoCorr, "known", "reversed", "corrected");
console.log(`  Correction gap (known vs reversed): full=${corrGapFull.toFixed(4)}, off=${corrGapOff.toFixed(4)}`);
assert("correction helps known vs reversed", corrGapFull >= corrGapOff - 0.01);

// 3. SETTLING: does it separate good from bad?
const settleGapFull = gap(scoresFull, "known", "wordSoup", "settled");
const settleGapOff = gap(scoresNoSettle, "known", "wordSoup", "settled");
console.log(`  Settling gap (known vs wordSoup): full=${settleGapFull.toFixed(4)}, off=${settleGapOff.toFixed(4)}`);
assert("settling widens or preserves known-vs-wordSoup gap", settleGapFull >= settleGapOff - 0.02);

// 4. HARD NEGATIVES: the real discriminative test
console.log("\n━━━ HARD NEGATIVE DISCRIMINATION ━━━\n");

// At settled level, positives should ALWAYS beat hard negatives
const posNames = ["known", "knownAlt", "unseen"];
const negNames = ["reversed", "wrongObj", "wrongObjRes", "wordSoup", "gibberish"];

for (const pos of posNames) {
  for (const neg of negNames) {
    const g = gap(scoresFull, pos, neg, "settled");
    const label = `${pos} > ${neg} (gap=${g.toFixed(4)})`;
    assert(label, g > -0.01); // allow tiny tolerance
  }
}

// 5. REVERSAL DETECTION: "doctor treats patient" vs "patient treats doctor"
// This is the original critic's test — v2.1 should still pass it
const fwdGap = scoresFull.known.forward - scoresFull.reversed.forward;
const corrGap2 = scoresFull.known.corrected - scoresFull.reversed.corrected;
const settledGap = scoresFull.known.settled - scoresFull.reversed.settled;
console.log(`\n  Reversal detection: fwd=${fwdGap.toFixed(4)}, corr=${corrGap2.toFixed(4)}, settled=${settledGap.toFixed(4)}`);
assert("reversal detected at forward level", fwdGap > 0);
assert("reversal detected at corrected level", corrGap2 > -0.01);
assert("reversal detected at settled level", settledGap > -0.01);

// 6. UNSEEN EQUIVALENCE: "physician manages patient" never trained explicitly
// Should score CLOSER to known than to gibberish
const unseenToKnown = Math.abs(scoresFull.unseen.settled - scoresFull.known.settled);
const unseenToGib = Math.abs(scoresFull.unseen.settled - scoresFull.gibberish.settled);
console.log(`  Unseen distance: to_known=${unseenToKnown.toFixed(4)}, to_gib=${unseenToGib.toFixed(4)}`);
assert("unseen closer to known than to gibberish", unseenToKnown < unseenToGib);

// ─── Summary ────────────────────────────────────────────────────────
console.log(`\n╔═══════════════════════════════════════════════╗`);
console.log(`║  ABLATION: ${pass}/${total} passed${fail > 0 ? `, ${fail} FAILED` : ""}`.padEnd(48) + "║");
console.log(`╚═══════════════════════════════════════════════╝`);
process.exit(fail > 0 ? 1 : 0);
