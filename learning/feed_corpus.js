#!/usr/bin/env node
/**
 * Shifu Corpus Feeder
 * 
 * Reads a corpus file (one sentence per line) and feeds it through
 * the ShifuEngine to build language understanding.
 * 
 * Usage:
 *   node learning/feed_corpus.js <corpus_file> [--passes N] [--state-dir DIR]
 */

const fs = require('fs');
const path = require('path');
const { createShifu, ShifuPersistence } = require('../index');

const args = process.argv.slice(2);
const corpusFile = args.find(a => !a.startsWith('--'));
const passesArg = args.indexOf('--passes');
const passes = passesArg >= 0 ? parseInt(args[passesArg + 1]) || 1 : 1;
const stateArg = args.indexOf('--state-dir');
const stateDir = stateArg >= 0 ? args[stateArg + 1] : undefined;

if (!corpusFile) {
  console.error('Usage: node learning/feed_corpus.js <corpus_file> [--passes N]');
  process.exit(1);
}

if (!fs.existsSync(corpusFile)) {
  console.error(`File not found: ${corpusFile}`);
  process.exit(1);
}

console.log(`\n${'='.repeat(60)}`);
console.log('  SHIFU CORPUS FEEDER');
console.log(`${'='.repeat(60)}`);

// Load or create engine
console.log('\nLoading engine...');
const persistence = new ShifuPersistence(stateDir);
const saved = persistence.load();
const shifu = createShifu({
  seed: true,
  loadTrained: true,
  ...(saved ? { seed: false, loadTrained: false, savedCoreState: saved.savedCoreState, savedLearningState: saved.savedLearningState } : {}),
});
const coreBefore = shifu.core.stats();
console.log(`  Vocabulary: ${coreBefore.vocabulary} words`);
console.log(`  Sentences seen: ${coreBefore.sentences}`);

// Read corpus
const raw = fs.readFileSync(corpusFile, 'utf-8');
const lines = raw.split('\n').map(l => l.trim()).filter(l => l.length > 5);
console.log(`\nCorpus: ${corpusFile}`);
console.log(`  Lines: ${lines.length.toLocaleString()}`);

// Feed in passes
const t0 = Date.now();
let totalSentences = 0;
let totalTokens = 0;

for (let pass = 1; pass <= passes; pass++) {
  let passSentences = 0;
  let passTokens = 0;
  
  // Shuffle lines each pass for better generalization
  const shuffled = [...lines];
  for (let i = shuffled.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
  }

  // Feed in batches of 500 for progress reporting
  const batchSize = 500;
  for (let i = 0; i < shuffled.length; i += batchSize) {
    const batch = shuffled.slice(i, i + batchSize);
    const result = shifu.core.feedBatch(batch);
    passSentences += result.sentences;
    passTokens += result.tokens;
    
    if ((i + batchSize) % 5000 === 0 || i + batchSize >= shuffled.length) {
      const pct = Math.min(100, Math.round((i + batchSize) / shuffled.length * 100));
      const elapsed = ((Date.now() - t0) / 1000).toFixed(0);
      process.stdout.write(
        `\r  Pass ${pass}/${passes}: ${pct}% (${passSentences.toLocaleString()} sentences, ${elapsed}s)`
      );
    }
  }
  
  totalSentences += passSentences;
  totalTokens += passTokens;
  console.log(''); // newline after progress

  // Compact every pass to keep memory reasonable
  if (pass < passes) {
    shifu.core.compact();
  }
}

// Final compact
shifu.core.compact();

// ── Phase 2: Learning Loop Training ─────────────────────────
// Simulate OCR errors + nurse corrections to train:
//   - Confusion profile (which chars get swapped)
//   - Ward vocabulary (learn new words from confirmations)
//   - Context chains (learn column associations)
//   - Clinical corrector (improve correction accuracy)

console.log(`\n${'='.repeat(60)}`);
console.log('  PHASE 2 — Learning Loop (Correction Simulation)');
console.log(`${'='.repeat(60)}`);

const { correctLine } = require('../clinical/corrector');

// OCR confusion table: what Shifu's topology engine typically confuses
const ocrSwaps = {
  'o': '0', '0': 'o', 'l': '1', '1': 'l', 'i': 't', 't': 'i',
  's': '5', '5': 's', 'b': '8', '8': 'b', 'g': '9', '9': 'g',
  'e': 'c', 'c': 'e', 'n': 'r', 'r': 'n', 'u': 'v', 'v': 'u',
  'h': 'b', 'd': 'o', 'a': 'o', 'm': 'rn',
};

function simulateOcrDamage(text, damageRate = 0.15) {
  // Apply realistic OCR-style corruption to text
  let result = '';
  for (const ch of text) {
    if (Math.random() < damageRate && ocrSwaps[ch.toLowerCase()]) {
      const swap = ocrSwaps[ch.toLowerCase()];
      result += ch === ch.toUpperCase() ? swap.toUpperCase() : swap;
    } else {
      result += ch;
    }
  }
  return result;
}

// Sample lines for correction training (use a subset — learning is expensive)
const learnSampleSize = Math.min(lines.length, 3000);
const learnSample = [];
for (let i = 0; i < learnSampleSize; i++) {
  learnSample.push(lines[Math.floor(Math.random() * lines.length)]);
}

let learnedCount = 0;
let correctionCount = 0;

for (let i = 0; i < learnSample.length; i++) {
  const truth = learnSample[i];
  const damaged = simulateOcrDamage(truth);

  if (damaged === truth) continue; // no damage, skip

  // Simulate a table row (medical context)
  const columns = ['Diagnosis', 'Notes', 'Medication', 'Patient'];
  const col = columns[i % columns.length];
  const ocrRow = { [col]: damaged };
  const confirmedRow = { [col]: truth };

  try {
    const result = shifu.learn(ocrRow, confirmedRow);
    if (result && result.accepted) {
      learnedCount++;
    }
    correctionCount++;
  } catch (e) {
    // Skip errors silently — some corrections may be rejected
  }

  if ((i + 1) % 500 === 0 || i === learnSample.length - 1) {
    const pct = Math.round((i + 1) / learnSample.length * 100);
    process.stdout.write(
      `\r  Learning: ${pct}% (${learnedCount} accepted / ${correctionCount} corrections)`
    );
  }
}
console.log(''); // newline

// Also feed confirmed text through ward vocabulary directly
console.log('  Building ward vocabulary...');
const vocabSample = learnSample.slice(0, 2000);
for (const line of vocabSample) {
  const words = line.split(/\s+/).filter(w => w.length > 2);
  for (const word of words) {
    shifu.learning.vocabulary.confirmWord(word);
  }
}

// Save
console.log('\nSaving state...');
persistence.save(shifu);

const coreAfter = shifu.core.stats();
const topConfusions = shifu.learning.confusion.getTopConfusions(10);
const learnedWords = shifu.learning.vocabulary.getLearnedWords(1);
const elapsed = ((Date.now() - t0) / 1000).toFixed(1);

console.log(`\n${'='.repeat(60)}`);
console.log('  FEEDING COMPLETE');
console.log(`${'='.repeat(60)}`);
console.log(`  Phase 1 — Language:`);
console.log(`    Sentences fed:  ${totalSentences.toLocaleString()}`);
console.log(`    Tokens fed:     ${totalTokens.toLocaleString()}`);
console.log(`    Vocabulary:     ${coreBefore.vocabulary} → ${coreAfter.vocabulary} words`);
console.log(`  Phase 2 — Learning Loop:`);
console.log(`    Corrections:    ${correctionCount} simulated`);
console.log(`    Accepted:       ${learnedCount}`);
console.log(`    Confusion pairs: ${topConfusions.length}`);
console.log(`    Learned words:  ${learnedWords.length}`);
console.log(`  Total time:       ${elapsed}s`);
console.log(`${'='.repeat(60)}\n`);

// Quick coherence test
const testSentences = [
  'The patient was admitted with chest pain',
  'Doctor prescribed medication for the infection',
  'akjsdh qwert zxcv bmnxc lkjhg',
];
console.log('Coherence test:');
for (const s of testSentences) {
  const score = shifu.scoreSentence(s);
  console.log(`  "${s.substring(0, 50)}..." → ${score.coherence.toFixed(3)}`);
}

// Correction test
console.log('\nCorrection test:');
const corrTests = [
  'Paiient adrnitted with chesi infecfion',
  'D0ct0r prescri8ed levetirace1am for seizvre',
  'P0tassium 4.5 within n0rmal range',
];
for (const s of corrTests) {
  const result = correctLine(s, {
    learningEngine: shifu.learning,
    coreEngine: shifu.core,
    ocrSource: true,
  });
  console.log(`  "${s}"`);
  console.log(`  → "${result.output}"  (conf: ${result.avgConfidence})`);
}
console.log('');
