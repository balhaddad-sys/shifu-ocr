#!/usr/bin/env node
// ═══════════════════════════════════════════════════════════════════════
// SHIFU LEARNING VERIFICATION — Did it actually learn?
// ═══════════════════════════════════════════════════════════════════════
// Tests the engine BEFORE and AFTER feeding to prove learning happened.

const { ShifuEngine } = require('../core/engine');
const { createTeacher } = require('./index');

const C = {
  reset: '\x1b[0m', bright: '\x1b[1m', dim: '\x1b[2m',
  green: '\x1b[32m', red: '\x1b[31m', yellow: '\x1b[33m',
  cyan: '\x1b[36m', magenta: '\x1b[35m',
};

const PASS = `${C.green}PASS${C.reset}`;
const FAIL = `${C.red}FAIL${C.reset}`;

console.log('');
console.log(`${C.bright}${C.cyan}${'═'.repeat(70)}${C.reset}`);
console.log(`${C.bright}${C.cyan}  DID SHIFU ACTUALLY LEARN? — Verification Suite${C.reset}`);
console.log(`${C.bright}${C.cyan}${'═'.repeat(70)}${C.reset}`);

let passed = 0;
let failed = 0;

function test(name, result, detail) {
  if (result) {
    console.log(`  ${PASS}  ${name}${detail ? ` ${C.dim}(${detail})${C.reset}` : ''}`);
    passed++;
  } else {
    console.log(`  ${FAIL}  ${name}${detail ? ` ${C.dim}(${detail})${C.reset}` : ''}`);
    failed++;
  }
}

// ═══════════════════════════════════════════════════════════════════════
// TEST 1: BLANK ENGINE vs FED ENGINE — Vocabulary
// ═══════════════════════════════════════════════════════════════════════
console.log(`\n${C.bright}--- TEST 1: Vocabulary Growth ---${C.reset}`);

const blankEngine = new ShifuEngine();
const fedEngine = new ShifuEngine();
const teacher = createTeacher({ engine: fedEngine }, { restore: false, seedBuiltin: false });

// Feed the teacher
const medSentences = [
  "Patient presents with acute myocardial infarction and elevated troponin levels.",
  "Doctor prescribed metoprolol and lisinopril for hypertension management.",
  "Nurse administered morphine sulfate 4mg IV for chest pain relief.",
  "Laboratory results show hemoglobin 10.2 and platelet count 180 thousand.",
  "Chest X-ray revealed bilateral pulmonary infiltrates consistent with pneumonia.",
  "Patient started on piperacillin-tazobactam for hospital acquired infection.",
  "Echocardiogram showed ejection fraction of 35 percent with wall motion abnormality.",
  "Blood cultures grew methicillin resistant Staphylococcus aureus requiring vancomycin.",
];

const legalSentences = [
  "The plaintiff alleges breach of contract under the Uniform Commercial Code.",
  "Defendant filed motion to dismiss for failure to state a claim.",
  "The court granted summary judgment in favor of the respondent.",
  "Counsel submitted memorandum in support of preliminary injunction.",
];

const finSentences = [
  "Quarterly revenue increased 12 percent year over year to 4.2 billion dollars.",
  "EBITDA margin expanded 200 basis points to 28.5 percent.",
  "Earnings per share came in at 2.15 beating consensus estimates.",
];

teacher.teachCorpus(medSentences, 'medical');
teacher.teachCorpus(legalSentences, 'legal');
teacher.teachCorpus(finSentences, 'financial');

const blankVocab = Object.keys(blankEngine.wf).length;
const fedVocab = Object.keys(fedEngine.wf).length;

test('Blank engine has zero vocabulary', blankVocab === 0, `blank=${blankVocab}`);
test('Fed engine learned vocabulary', fedVocab > 50, `fed=${fedVocab} words`);

// ═══════════════════════════════════════════════════════════════════════
// TEST 2: WORD FREQUENCY — Does it know medical terms?
// ═══════════════════════════════════════════════════════════════════════
console.log(`\n${C.bright}--- TEST 2: Word Recognition ---${C.reset}`);

const medTerms = ['patient', 'doctor', 'morphine', 'troponin', 'hemoglobin', 'pneumonia', 'vancomycin'];
const legalTerms = ['plaintiff', 'defendant', 'court', 'judgment', 'contract'];
const finTerms = ['revenue', 'earnings', 'margin', 'billion'];

for (const term of medTerms) {
  const freq = fedEngine.wf[term] || 0;
  test(`Knows medical term "${term}"`, freq > 0, `freq=${freq}`);
}
for (const term of legalTerms) {
  const freq = fedEngine.wf[term] || 0;
  test(`Knows legal term "${term}"`, freq > 0, `freq=${freq}`);
}
for (const term of finTerms) {
  const freq = fedEngine.wf[term] || 0;
  test(`Knows financial term "${term}"`, freq > 0, `freq=${freq}`);
}

// ═══════════════════════════════════════════════════════════════════════
// TEST 3: CO-OCCURRENCE — Does it understand context?
// ═══════════════════════════════════════════════════════════════════════
console.log(`\n${C.bright}--- TEST 3: Contextual Understanding (Co-occurrence) ---${C.reset}`);

function hasCooccurrence(engine, word1, word2) {
  const co = engine.co[word1];
  return co && co[word2] && co[word2] > 0;
}

test('"patient" co-occurs with "doctor"', hasCooccurrence(fedEngine, 'patient', 'doctor') || hasCooccurrence(fedEngine, 'doctor', 'patient'));
test('"morphine" co-occurs with "pain"', hasCooccurrence(fedEngine, 'morphine', 'pain') || hasCooccurrence(fedEngine, 'pain', 'morphine'));
test('"chest" co-occurs with "ray"', hasCooccurrence(fedEngine, 'chest', 'ray') || hasCooccurrence(fedEngine, 'ray', 'chest'));
test('"plaintiff" co-occurs with "defendant"', hasCooccurrence(fedEngine, 'plaintiff', 'defendant') || hasCooccurrence(fedEngine, 'defendant', 'plaintiff'));
test('"revenue" co-occurs with "percent"', hasCooccurrence(fedEngine, 'revenue', 'percent') || hasCooccurrence(fedEngine, 'percent', 'revenue'));

// Blank engine should have NONE
test('Blank engine has no co-occurrences', !hasCooccurrence(blankEngine, 'patient', 'doctor'));

// ═══════════════════════════════════════════════════════════════════════
// TEST 4: SEQUENCE PREDICTION — Does it predict what comes next?
// ═══════════════════════════════════════════════════════════════════════
console.log(`\n${C.bright}--- TEST 4: Sequence Prediction (Next-word) ---${C.reset}`);

function topNextWords(engine, word, n = 5) {
  const nx = engine.nx[word];
  if (!nx) return [];
  return Object.entries(nx).sort((a, b) => b[1] - a[1]).slice(0, n).map(([w, c]) => ({ word: w, count: c }));
}

const afterPatient = topNextWords(fedEngine, 'patient');
test('"patient" has learned next-words', afterPatient.length > 0,
  afterPatient.map(w => w.word).join(', '));

const afterDoctor = topNextWords(fedEngine, 'doctor');
test('"doctor" has learned next-words', afterDoctor.length > 0,
  afterDoctor.map(w => w.word).join(', '));

const afterChest = topNextWords(fedEngine, 'chest');
test('"chest" has learned next-words', afterChest.length > 0,
  afterChest.map(w => w.word).join(', '));

const afterPlaintiff = topNextWords(fedEngine, 'plaintiff');
test('"plaintiff" has learned next-words', afterPlaintiff.length > 0,
  afterPlaintiff.map(w => w.word).join(', '));

// Blank engine
const blankAfterPatient = topNextWords(blankEngine, 'patient');
test('Blank engine cannot predict after "patient"', blankAfterPatient.length === 0);

// ═══════════════════════════════════════════════════════════════════════
// TEST 5: SIMILARITY — Can it find similar words?
// ═══════════════════════════════════════════════════════════════════════
console.log(`\n${C.bright}--- TEST 5: Word Similarity ---${C.reset}`);

function findSimilar(engine, word) {
  try {
    const results = engine._similarTo(word);
    return results.slice(0, 5);
  } catch(e) { return []; }
}

const simToPatient = findSimilar(fedEngine, 'patient');
test('"patient" has similar words', simToPatient.length > 0,
  simToPatient.map(s => `${s.word}:${s.sim.toFixed(2)}`).join(', '));

const simToDoctor = findSimilar(fedEngine, 'doctor');
test('"doctor" has similar words', simToDoctor.length > 0,
  simToDoctor.map(s => `${s.word}:${s.sim.toFixed(2)}`).join(', '));

// ═══════════════════════════════════════════════════════════════════════
// TEST 6: CALIBRATION PERSISTENCE — Are corrections saved?
// ═══════════════════════════════════════════════════════════════════════
console.log(`\n${C.bright}--- TEST 6: Calibration from Corrections ---${C.reset}`);

// Teach corrections
teacher.teachCorrection('tr0ponin', 'troponin', 'medical');
teacher.teachCorrection('m0rphine', 'morphine', 'medical');
teacher.teachCorrection('p1aintiff', 'plaintiff', 'legal');
teacher.teachCorrection('depreciati0n', 'depreciation', 'financial');

const calStore = teacher.model.calibrations;
test('Calibrations recorded', calStore.points.length >= 4, `${calStore.points.length} points`);

const medAcc = calStore.getDomainAccuracy('medical');
test('Medical domain has accuracy tracking', medAcc !== null,
  medAcc ? `accuracy=${(medAcc.accuracy * 100).toFixed(1)}%` : 'none');

const crossConfusions = calStore.getCrossDomainConfusions(1);
test('Cross-domain confusions detected', crossConfusions.length > 0,
  `${crossConfusions.length} confusion patterns found`);

// Check specific confusion pattern: 0/O should appear in multiple domains
const zeroOh = crossConfusions.find(c => c.pair === '0,o' || c.pair === 'o,0');
test('"0/O" confusion tracked across domains', !!zeroOh,
  zeroOh ? `count=${zeroOh.count}, domains=${Object.keys(zeroOh.domains).join('+')}` : 'not found');

// ═══════════════════════════════════════════════════════════════════════
// TEST 7: DOMAIN DETECTION — Can it identify text domains?
// ═══════════════════════════════════════════════════════════════════════
console.log(`\n${C.bright}--- TEST 7: Domain Auto-Detection ---${C.reset}`);

const medText = "Patient admitted with acute stroke and elevated blood pressure requiring medication.";
const legalText = "The defendant filed a motion to dismiss the plaintiff's breach of contract claim.";
const finText = "Revenue growth exceeded expectations with strong earnings and dividend yield.";

const detMed = teacher.autoDetect(medText);
test('Detects medical text', detMed[0]?.domain?.id === 'medical',
  `detected: ${detMed[0]?.domain?.id} (score: ${detMed[0]?.score})`);

const detLegal = teacher.autoDetect(legalText);
test('Detects legal text', detLegal[0]?.domain?.id === 'legal',
  `detected: ${detLegal[0]?.domain?.id} (score: ${detLegal[0]?.score})`);

const detFin = teacher.autoDetect(finText);
test('Detects financial text', detFin[0]?.domain?.id === 'financial',
  `detected: ${detFin[0]?.domain?.id} (score: ${detFin[0]?.score})`);

// ═══════════════════════════════════════════════════════════════════════
// TEST 8: OCR CORRECTION — Can it correct corrupted text?
// ═══════════════════════════════════════════════════════════════════════
console.log(`\n${C.bright}--- TEST 8: OCR Correction Capability ---${C.reset}`);

// Use the engine's candidate matching to find corrections
function findBestMatch(engine, corrupted) {
  const candidates = engine._candidates(corrupted.toLowerCase());
  if (candidates.length === 0) return null;

  // Sort by OCR distance (lower = more likely correction)
  let best = null;
  let bestDist = Infinity;
  for (const cand of candidates) {
    const chars = corrupted.toLowerCase().split('');
    const candChars = cand.split('');
    let dist = 0;
    const minLen = Math.min(chars.length, candChars.length);
    for (let i = 0; i < minLen; i++) {
      if (chars[i] !== candChars[i]) dist++;
    }
    dist += Math.abs(chars.length - candChars.length);
    if (dist < bestDist) {
      bestDist = dist;
      best = cand;
    }
  }
  return best;
}

const corrections = [
  { corrupted: 'pat1ent', expected: 'patient' },
  { corrupted: 'doct0r', expected: 'doctor' },
  { corrupted: 'tropon1n', expected: 'troponin' },
  { corrupted: 'p1aintiff', expected: 'plaintiff' },
  { corrupted: 'revenu3', expected: 'revenue' },
  { corrupted: 'pneum0nia', expected: 'pneumonia' },
];

for (const { corrupted, expected } of corrections) {
  const match = findBestMatch(fedEngine, corrupted);
  const correct = match === expected;
  test(`Corrects "${corrupted}" → "${expected}"`, correct,
    correct ? 'matched!' : `got: ${match || 'no match'}`);
}

// Same test on blank engine — should fail
const blankMatch = findBestMatch(blankEngine, 'pat1ent');
test('Blank engine CANNOT correct "pat1ent"', blankMatch === null,
  blankMatch ? `incorrectly matched: ${blankMatch}` : 'correctly returned null');

// ═══════════════════════════════════════════════════════════════════════
// TEST 9: WAVE PROPAGATION — Network connectivity
// ═══════════════════════════════════════════════════════════════════════
console.log(`\n${C.bright}--- TEST 9: Neural Network Connectivity ---${C.reset}`);

const totalWords = Object.keys(fedEngine.wf).length;
const totalCoLinks = Object.values(fedEngine.co).reduce((s, obj) => s + Object.keys(obj).length, 0);
const totalNxLinks = Object.values(fedEngine.nx).reduce((s, obj) => s + Object.keys(obj).length, 0);

test(`Network has ${totalWords} neurons (words)`, totalWords > 50);
test(`Network has ${totalCoLinks} co-occurrence synapses`, totalCoLinks > 100);
test(`Network has ${totalNxLinks} sequence connections`, totalNxLinks > 50);

// Check connectivity: how many words does "patient" connect to?
const patientConns = fedEngine.co['patient'] ? Object.keys(fedEngine.co['patient']).length : 0;
test(`"patient" connects to ${patientConns} other words`, patientConns > 3);

// ═══════════════════════════════════════════════════════════════════════
// TEST 10: CROSS-DOMAIN TRANSFER — Universal patterns
// ═══════════════════════════════════════════════════════════════════════
console.log(`\n${C.bright}--- TEST 10: Cross-Domain Knowledge ---${C.reset}`);

// Words that appear across domains
const crossDomainWords = ['the', 'with', 'for', 'percent', 'showed'];
for (const word of crossDomainWords) {
  const freq = fedEngine.wf[word] || 0;
  test(`Cross-domain word "${word}" reinforced`, freq >= 2, `freq=${freq} (seen across domains)`);
}

// Universal confusion profile
const activeConfusion = teacher.model.getActiveConfusionProfile();
const confusionPairs = Object.keys(activeConfusion).length;
test('Active confusion profile built', confusionPairs > 0, `${confusionPairs} confusion pairs`);

// ═══════════════════════════════════════════════════════════════════════
// SUMMARY
// ═══════════════════════════════════════════════════════════════════════
console.log('');
console.log(`${C.bright}${'═'.repeat(70)}${C.reset}`);
console.log(`${C.bright}  RESULTS: ${C.green}${passed} passed${C.reset}, ${failed > 0 ? C.red : C.green}${failed} failed${C.reset}  out of ${passed + failed} tests`);

if (failed === 0) {
  console.log(`\n${C.bright}${C.green}  YES — Shifu actually learned.${C.reset}`);
  console.log(`${C.dim}  The engine built vocabulary, co-occurrence networks, sequence predictions,`);
  console.log(`  cross-domain confusions, and calibration data — all persisted for future use.${C.reset}`);
} else {
  console.log(`\n${C.bright}${C.yellow}  Shifu learned, but ${failed} test(s) need attention.${C.reset}`);
}

console.log(`${C.bright}${'═'.repeat(70)}${C.reset}`);
console.log('');

process.exit(failed > 0 ? 1 : 0);
