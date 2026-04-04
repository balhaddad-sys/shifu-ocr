// SHIFU SYSTEM TEST — everything flows through one field
// Run: node test/system.js

const shifu = require('../core/shifu');

let pass = 0, fail = 0, total = 0;
function test(name, got, expected) {
  total++;
  if (got === expected) { pass++; }
  else { fail++; console.log(`  FAIL: ${name}`); console.log(`    got:      "${got}"`); console.log(`    expected: "${expected}"`); }
}
function section(n) { console.log(`\n--- ${n} ---`); }

console.log("=== SHIFU SYSTEM TEST — one field, one system ===");
const s = shifu.stats();
console.log(`  ${s.vocabulary} words, ${s.sentences} sentences`);

section("1 process(): raw OCR text in, corrected text out");
let r;
r = shifu.process("doctor treats patient with stok");
test("stok->stroke", r.corrected, "doctor treats patient with stroke");

r = shifu.process("patient admitted with chest infoction");
test("infoction->infection", r.corrected, "patient admitted with chest infection");

r = shifu.process("doctor treets patlent with medlcation");
test("multi-error line", r.corrected, "doctor treats patient with medication");

r = shifu.process("doctor treats patient with medication");
test("clean text unchanged", r.corrected, "doctor treats patient with medication");
test("clean text: no decisions", r.decisions.length, 0);

section("2 processWithCandidates()");
r = shifu.processWithCandidates("doctor treats patient with stok", { 4: ["stroke", "stock", "stork"] });
test("with candidates: stok->stroke", r.corrected, "doctor treats patient with stroke");

section("3 defend known words");
r = shifu.processWithCandidates("doctor treats patient", { 2: ["pleasant"] });
test("keep patient", r.corrected, "doctor treats patient");

r = shifu.processWithCandidates("patient presents with stroke", { 3: ["stork"] });
test("keep stroke", r.corrected, "patient presents with stroke");

section("4 score()");
const goodScore = shifu.score("doctor treats patient with medication");
const gibScore = shifu.score("zzzzz xxxxx yyyyy qqqqq");
test("good > gibberish", goodScore.settled > gibScore.settled, true);

section("5 validate()");
const v = shifu.validate("doctor treats patient", "doctor treats pleasant");
test("patient->pleasant rejected", v.recommendation, "reject");

section("6 infer()");
const inf = shifu.infer("levetiracepam");
test("levetiracepam -> levetiracetam", inf.nearest, "levetiracetam");
test("stroke is known", shifu.infer("stroke").known, true);

section("7 processOCRLine()");
r = shifu.processOCRLine({
  text: "doctor treets patlent",
  characters: [
    {char:'d',confidence:0.9},{char:'o',confidence:0.8},{char:'c',confidence:0.9},
    {char:'t',confidence:0.9},{char:'o',confidence:0.8},{char:'r',confidence:0.9},
    {char:' ',confidence:1.0},
    {char:'t',confidence:0.8},{char:'r',confidence:0.7},{char:'e',confidence:0.3},
    {char:'e',confidence:0.3},{char:'t',confidence:0.8},{char:'s',confidence:0.9},
    {char:' ',confidence:1.0},
    {char:'p',confidence:0.8},{char:'a',confidence:0.7},{char:'t',confidence:0.3},
    {char:'l',confidence:0.4},{char:'e',confidence:0.8},{char:'n',confidence:0.8},
    {char:'t',confidence:0.9},
  ],
});
test("OCR: treets corrected", r.corrected.includes("treats"), true);
test("OCR: patlent corrected", r.corrected.includes("patient"), true);

section("8 processTable()");
const tableResult = shifu.processTable([
  ["12-3", "patient", "strok", "doctor"],
  ["20-1", "patient", "seisure", "physician"],
]);
test("table: strok->stroke", tableResult.corrected[0][2], "stroke");
test("table: seisure->seizure", tableResult.corrected[1][2], "seizure");

section("9 edge cases");
r = shifu.process("");
test("empty string", r.corrected, "");
r = shifu.process("zzzzz xxxxx yyyyy");
test("gibberish passes through", r.corrected, "zzzzz xxxxx yyyyy");
r = shifu.processOCRLine({ text: "", characters: [] });
test("empty OCR", r.corrected, "");

section("10 performance");
const t0 = Date.now();
for (let i = 0; i < 50; i++) shifu.process("doctor treats patient with medication");
const avgMs = (Date.now() - t0) / 50;
console.log(`  avg process(): ${avgMs.toFixed(1)}ms`);
test("process() < 100ms", avgMs < 100, true);

console.log(`\n${"=".repeat(50)}`);
console.log(`  SYSTEM: ${pass}/${total} passed${fail > 0 ? `, ${fail} FAILED` : ""}`);
if (fail === 0) console.log("  One field. One system. It works.");
console.log(`${"=".repeat(50)}`);
process.exit(fail > 0 ? 1 : 0);
