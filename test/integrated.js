// SHIFU INTEGRATED TEST — observe, learn, command
// Run: node test/integrated.js

const { ShifuEngine, ocrDist, fieldCosine } = require("../core/engine");
const corpus = require("../data/medical_corpus");

let pass = 0, fail = 0, total = 0;
function assert(n, c) { total++; if (c) pass++; else { fail++; console.log(`  FAIL: ${n}`); } }
function S(n) { console.log(`\n--- ${n} ---`); }

const eng = new ShifuEngine();
eng.feedBatch(corpus);
const stats = eng.stats();
console.log("=== SHIFU INTEGRATED TEST ===");
console.log(`  ${stats.vocabulary} words, ${stats.sentences} sentences`);

S("1 INFER");
assert("stroke is known", eng.infer("stroke").known === true);
assert("levetiracepam -> levetiracetam", eng.infer("levetiracepam").nearest === "levetiracetam");
assert("patlent -> patient", eng.infer("patlent").nearest === "patient");
assert("zzzzz has high distance", eng.infer("zzzzz").distance >= 3);

S("2 CHOOSE — good corrections");
const corrections = [
  { words: ["doctor","treats","patient","with","stok"], pos: 4, cands: ["stroke","stock"], expected: "stroke" },
  { words: ["patient","was","admitted","to","the","wrad"], pos: 5, cands: ["ward","wart"], expected: "ward" },
  { words: ["patient","presents","with","chest","paim"], pos: 4, cands: ["pain","palm"], expected: "pain" },
];
for (const { words, pos, cands, expected } of corrections) {
  const r = eng.choose(words, pos, cands);
  assert(`${words[pos]} -> ${expected}`, r.word === expected);
}

S("3 CHOOSE — bad corrections rejected");
const rejections = [
  { words: ["doctor","treats","patient"], pos: 2, cands: ["pleasant"], expected: "patient" },
  { words: ["nurse","records","vital","signs"], pos: 2, cands: ["vinyl"], expected: "vital" },
];
for (const { words, pos, cands, expected } of rejections) {
  const r = eng.choose(words, pos, cands);
  assert(`keep ${expected}`, r.word === expected);
}

S("4 CHOOSE — adversarial");
const adversarial = [
  { words: ["patient","admitted","with","chest","infoction"], pos: 4, cands: ["infection","infarction"], expected: "infection" },
  { words: ["doctor","prescribed","medication","for","seisure"], pos: 4, cands: ["seizure","suture"], expected: "seizure" },
  { words: ["patient","presents","with","bilateral","weskness"], pos: 4, cands: ["weakness","wetness"], expected: "weakness" },
];
for (const { words, pos, cands, expected } of adversarial) {
  const r = eng.choose(words, pos, cands);
  assert(`${words[pos]} -> ${expected}`, r.word === expected);
}

S("5 CHOOSE LINE");
const lineResult = eng.chooseLine(
  ["doctor","treets","patlent","with","medlcation"],
  { 1: ["treats","trees"], 2: ["patient","patent"], 4: ["medication","meditation"] }
);
assert("treats chosen", lineResult.words[1] === "treats");
assert("patient chosen", lineResult.words[2] === "patient");
assert("medication chosen", lineResult.words[4] === "medication");

S("6 ACCOMMODATION");
const sentResult = eng.accommodateSentence([
  "doctor treats patient with medication",
  "doctor treats patient with napkin",
]);
assert("medication beats napkin", sentResult.best === "doctor treats patient with medication");

S("7 WAVE FIELD");
const fieldA = eng.sentenceField("doctor treats patient with medication");
const fieldGib = eng.sentenceField("zzzzz xxxxx yyyyy qqqqq");
const simAGib = fieldCosine(fieldA, fieldGib);
assert("known > gibberish in field space", simAGib < 0.5);

S("8 RESONANCE");
const dpRes = eng.res["doctor"]?.["physician"] || 0;
assert("doctor-physician resonance", dpRes > 1);

S("9 EDGE CASES");
assert("empty candidates", eng.choose(["hello"], 0, []).word === "hello");
assert("infer empty", eng.infer("").word === "");
assert("chooseLine empty", eng.chooseLine(["doctor","treats","patient"], {}).corrected === "doctor treats patient");

console.log(`\n${"=".repeat(50)}`);
console.log(`  INTEGRATED: ${pass}/${total} passed${fail > 0 ? `, ${fail} FAILED` : ""}`);
console.log(`${"=".repeat(50)}`);
process.exit(fail > 0 ? 1 : 0);
