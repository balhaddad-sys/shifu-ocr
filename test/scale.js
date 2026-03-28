// SHIFU SCALE TEST — real corpus, real claims
// Run: node test/scale.js

const { ShifuEngine, fieldCosine } = require("../core/engine");
const corpus = require("../data/medical_corpus");

let pass = 0, fail = 0, total = 0;
function assert(n, c) { total++; if (c) pass++; else { fail++; console.log(`  FAIL: ${n}`); } }
function S(n) { console.log(`\n--- ${n} ---`); }

console.log("=== SHIFU SCALE TEST ===\n");
const eng = new ShifuEngine();
const t0 = Date.now();
eng.feedBatch(corpus);
const stats = eng.stats();
console.log(`  Trained in ${Date.now()-t0}ms: ${stats.vocabulary} words, ${stats.sentences} sentences`);

S("1 SCALE INVARIANTS");
assert("vocabulary > 200", stats.vocabulary > 200);
assert("resonance pairs > 50", stats.resonancePairs > 50);

S("2 RESONANCE");
assert("doctor-physician found", (eng.res["doctor"]?.["physician"] || 0) > 0);
assert("treats-manages found", (eng.res["treats"]?.["manages"] || 0) > 0);

S("3 DIRECTIONALITY");
const scFwd = eng.scoreSentence("doctor treats patient with medication");
const scRev = eng.scoreSentence("patient treats doctor with medication");
const scGib = eng.scoreSentence("zzzzz xxxxx yyyyy qqqqq");
assert("correct > reversed", scFwd.correctedCoherence > scRev.correctedCoherence);
assert("known > gibberish", scFwd.settledCoherence > scGib.settledCoherence);

S("4 UNSEEN EQUIVALENCE");
const scUnseen = eng.scoreSentence("physician manages patient with therapy");
assert("unseen > gibberish", scUnseen.settledCoherence > scGib.settledCoherence);

S("5 SENTENCE COMPARISON");
const sim1 = eng.compareSentences("doctor treats patient", "physician treats patient");
const sim2 = eng.compareSentences("doctor treats patient", "zzzzz xxxxx");
assert("equivalent > gibberish", sim1.similarity > sim2.similarity);

S("6 ROBUSTNESS");
const real = ["Patient presents with acute onset left sided weakness.",
  "Doctor ordered urgent CT head to rule out hemorrhage.",
  "Nurse administered intravenous antibiotics as prescribed."];
for (const s of real) {
  assert(`real: "${s.slice(0,40)}..."`, eng.scoreSentence(s).settledCoherence > scGib.settledCoherence);
}

S("7 SERIALIZATION");
const json = eng.serialize();
const eng2 = ShifuEngine.deserialize(json);
assert("round-trip vocabulary", eng2.stats().vocabulary === stats.vocabulary);
assert("round-trip score", Math.abs(eng2.scoreSentence("doctor treats patient").settledCoherence - scFwd.settledCoherence) < 0.001);

S("8 PERFORMANCE");
const p0 = Date.now();
for (let i = 0; i < 100; i++) eng.scoreSentence("doctor treats patient with medication");
console.log(`  avg scoreSentence: ${((Date.now()-p0)/100).toFixed(1)}ms`);
assert("scoreSentence < 50ms", (Date.now()-p0)/100 < 50);

console.log(`\n${"=".repeat(50)}`);
console.log(`  SCALE: ${pass}/${total} passed${fail > 0 ? `, ${fail} FAILED` : ""}`);
console.log(`${"=".repeat(50)}`);
process.exit(fail > 0 ? 1 : 0);
