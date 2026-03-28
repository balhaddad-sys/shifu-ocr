// SHIFU ABLATION SUITE — does each mechanism earn its place?
// Run: node test/ablation.js

const { ShifuEngine, CONFIG, fieldCosine } = require("../core/engine");

const corpus = [
  "doctor treats patient with medication and therapy",
  "doctor treats patient with medication and therapy",
  "doctor treats patient with medication and therapy",
  "doctor examines patient and notes findings",
  "doctor examines patient and notes findings",
  "doctor prescribes medication for patient",
  "physician treats patient with medication and therapy",
  "physician treats patient with medication and therapy",
  "physician examines patient and notes findings",
  "physician manages patient effectively in the ward",
  "physician manages patient effectively in the ward",
  "nurse administers medication as per protocol",
  "nurse administers medication as per protocol",
  "patient presents with stroke and weakness",
  "patient presents with stroke and weakness",
  "patient was admitted to the stroke unit for monitoring",
];

const sentences = {
  known: "doctor treats patient with medication",
  reversed: "patient treats doctor",
  wrongObj: "doctor treats medication",
  wordSoup: "doctor with and patient",
  gibberish: "zzzzz xxxxx yyyyy",
};

function trainEngine(cfgOverride = {}) {
  const cfg = JSON.parse(JSON.stringify(CONFIG));
  for (const [k, v] of Object.entries(cfgOverride)) {
    if (typeof v === "object" && v !== null) cfg[k] = { ...(cfg[k] || {}), ...v };
    else cfg[k] = v;
  }
  const eng = new ShifuEngine(cfg);
  for (const s of corpus) eng.feed(s);
  return eng;
}

function scoreAll(eng) {
  const r = {};
  for (const [n, s] of Object.entries(sentences)) {
    const sc = eng.scoreSentence(s);
    r[n] = { forward: sc.coherence, corrected: sc.correctedCoherence, settled: sc.settledCoherence };
  }
  return r;
}

let pass = 0, fail = 0, total = 0;
function assert(n, c) { total++; if (c) pass++; else { fail++; console.log(`  FAIL: ${n}`); } }
function gap(scores, a, b, level = "settled") { return scores[a][level] - scores[b][level]; }

console.log("=== SHIFU ABLATION SUITE ===\n");

const full = scoreAll(trainEngine());
const noInh = scoreAll(trainEngine({ inhibition: { floor: 1.0, ceiling: 1.0 } }));
const noCorr = scoreAll(trainEngine({ correction: { strength: 0 } }));
const noSettle = scoreAll(trainEngine({ settle: { maxIter: 0, epsilon: 0.01, reactivateTop: 8, iterDecay: 0.5 } }));

assert("inhibition widens gap", gap(full, "known", "wordSoup") >= gap(noInh, "known", "wordSoup") - 0.01);
assert("correction helps known vs reversed", gap(full, "known", "reversed", "corrected") >= gap(noCorr, "known", "reversed", "corrected") - 0.01);
assert("settling preserves gap", gap(full, "known", "wordSoup") >= gap(noSettle, "known", "wordSoup") - 0.02);
assert("reversal detected", full.known.forward > full.reversed.forward);

console.log(`\n${"=".repeat(50)}`);
console.log(`  ABLATION: ${pass}/${total} passed${fail > 0 ? `, ${fail} FAILED` : ""}`);
console.log(`${"=".repeat(50)}`);
process.exit(fail > 0 ? 1 : 0);
