#!/usr/bin/env node
// Shifu Teaching Model — Interactive Demo
// Demonstrates cross-domain teaching, calibration, and evaluation.

const { ShifuEngine } = require('../core/engine');
const { createTeacher } = require('./index');

console.log('');
console.log('='.repeat(60));
console.log('  SHIFU TEACHING MODEL v1.0.0 — Demo');
console.log('  Cross-domain teaching with calibration persistence');
console.log('='.repeat(60));

// ─── 1. Create Engine & Teacher ─────────────────────────────────────

console.log('\n--- 1. Creating Shifu Engine + Teacher ---');
const engine = new ShifuEngine();
const teacher = createTeacher({ engine }, { restore: false, seedBuiltin: true });

console.log('  Teacher created with built-in seeds');
console.log('  Available domains:', teacher.listDomains().map(d => d.id).join(', '));

// ─── 2. Domain Detection ────────────────────────────────────────────

console.log('\n--- 2. Auto-Domain Detection ---');

const testTexts = [
  'Patient admitted with acute myocardial infarction and elevated troponin.',
  'The defendant filed a motion to dismiss under Rule 12(b)(6).',
  'Quarterly revenue increased 12% year-over-year to $4.2 billion.',
  'The catalyst increased reaction yield from 45% to 92%.',
  'The microcontroller operates at 3.3V with clock frequency 168MHz.',
  'Student enrollment for the fall semester reached 15,000.',
];

for (const text of testTexts) {
  const detected = teacher.autoDetect(text);
  const top = detected[0];
  console.log(`  "${text.slice(0, 55)}..." → ${top.domain.id} (score: ${top.score})`);
}

// ─── 3. Cross-Domain Teaching ───────────────────────────────────────

console.log('\n--- 3. Cross-Domain Teaching ---');

// Medical teaching
teacher.startSession('medical', 'cardiology');
const medicalCorpus = [
  'Troponin I elevated at 2.4 ng/mL consistent with NSTEMI.',
  'Heparin drip started at 18 units per kilogram per hour.',
  'Echocardiogram showed ejection fraction of 35 percent.',
  'Aspirin 325mg and clopidogrel 75mg daily prescribed.',
  'Patient transferred to CCU for continuous monitoring.',
];
const medResult = teacher.teachCorpus(medicalCorpus, 'medical');
console.log(`  Medical: taught ${medResult.totalSentences} sentences in ${medResult.batches} batches`);

// Legal teaching
const legalCorpus = [
  'The court finds insufficient evidence to support the claim.',
  'Pursuant to Section 301 the arbitration agreement is enforceable.',
  'Discovery materials must be produced within thirty days.',
  'The witness was cross-examined regarding chain of custody.',
];
const legalResult = teacher.teachCorpus(legalCorpus, 'legal');
console.log(`  Legal: taught ${legalResult.totalSentences} sentences`);

// Financial teaching
const finCorpus = [
  'EBITDA margin expanded 200 basis points to 28.5 percent.',
  'Total assets under management reached $12.8 billion.',
  'The dividend yield currently stands at 3.2 percent annually.',
];
const finResult = teacher.teachCorpus(finCorpus, 'financial');
console.log(`  Financial: taught ${finResult.totalSentences} sentences`);

const sessionSummary = teacher.endSession();
console.log(`  Session duration: ${sessionSummary.duration}ms`);

// ─── 4. Calibration from Corrections ────────────────────────────────

console.log('\n--- 4. Teaching from Corrections ---');

const corrections = [
  { ocr: 'tr0ponin', correct: 'troponin', domain: 'medical' },
  { ocr: 'm0rphine', correct: 'morphine', domain: 'medical' },
  { ocr: 'p1aintiff', correct: 'plaintiff', domain: 'legal' },
  { ocr: 'ju0gment', correct: 'judgment', domain: 'legal' },
  { ocr: '$4.2 bi11ion', correct: '$4.2 billion', domain: 'financial' },
  { ocr: 'depreciati0n', correct: 'depreciation', domain: 'financial' },
  { ocr: 'cata1yst', correct: 'catalyst', domain: 'scientific' },
  { ocr: 'micr0controller', correct: 'microcontroller', domain: 'engineering' },
];

for (const { ocr, correct, domain } of corrections) {
  const result = teacher.teachCorrection(ocr, correct, domain);
  console.log(`  [${domain}] "${ocr}" → "${correct}" (cal: ${result.calibration.id.slice(0, 20)}...)`);
}

// ─── 5. Curriculum Progress ─────────────────────────────────────────

console.log('\n--- 5. Curriculum Status ---');
const progress = teacher.getProgress();

for (const [domainId, dp] of Object.entries(progress.domainProgress)) {
  console.log(`  ${domainId}: vocab=${dp.vocabularySize}, taught=${dp.sentencesTaught}, corrections=${dp.correctionsMade}`);
}

console.log(`\n  Total calibrations: ${progress.totalCalibrations}`);
console.log(`  Unapplied: ${progress.unappliedCalibrations}`);
console.log(`  Universal confusions: ${progress.universalConfusions}`);

// ─── 6. Recommendations ────────────────────────────────────────────

console.log('\n--- 6. Teaching Recommendations ---');
const recs = teacher.getRecommendations();
for (const rec of recs.slice(0, 5)) {
  console.log(`  [${rec.type}] ${rec.message}`);
}

// ─── 7. Evaluation Report ───────────────────────────────────────────

console.log('\n--- 7. Evaluation ---');
const report = teacher.generateReport();
console.log(`  Domains evaluated: ${report.summary.domainsEvaluated}`);
console.log(`  Total predictions: ${report.summary.totalPredictions}`);
if (report.summary.totalPredictions > 0) {
  console.log(`  Overall accuracy: ${(report.summary.overallAccuracy * 100).toFixed(1)}%`);
  console.log(`  Overall CER: ${report.summary.overallCER.toFixed(4)}`);
}

// ─── 8. Persistence ────────────────────────────────────────────────

console.log('\n--- 8. Saving State ---');
teacher.save();
console.log('  Teaching state saved to .teaching/');
console.log('  Calibrations saved to .teaching/calibrations/');

// ─── Done ───────────────────────────────────────────────────────────

console.log('\n' + '='.repeat(60));
console.log('  Demo complete! Teaching model is ready.');
console.log('');
console.log('  Seed data pipeline:');
console.log('    npm run teach:seed:builtin   # Built-in seeds (no download)');
console.log('    npm run teach:seed:kaggle    # Kaggle datasets');
console.log('    npm run teach:seed:huggingface # HuggingFace datasets');
console.log('    npm run teach:seed:all       # All sources');
console.log('    npm run teach:seed:stats     # View statistics');
console.log('    npm run teach:emit           # Merge into teaching_data.json');
console.log('');
console.log('  Python seeding:');
console.log('    python -m teaching.seeds.kaggle --list');
console.log('    python -m teaching.seeds.huggingface --list');
console.log('    python -m teaching.seeds.seed_registry --stats');
console.log('='.repeat(60));
console.log('');
