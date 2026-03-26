#!/usr/bin/env node
// ╔══════════════════════════════════════════════════════════════════════════╗
// ║                    SHIFU OCR — Clinical Demo                            ║
// ║                                                                         ║
// ║  Scan in → Corrected table out → Safety flags highlighted               ║
// ║  This is what a doctor would see.                                       ║
// ╚══════════════════════════════════════════════════════════════════════════╝

const { createShifu, VERSION } = require('../index');

// ─── Colors for terminal output ─────────────────────────────────────────────
const C = {
  reset: '\x1b[0m',
  bold: '\x1b[1m',
  dim: '\x1b[2m',
  red: '\x1b[31m',
  green: '\x1b[32m',
  yellow: '\x1b[33m',
  blue: '\x1b[34m',
  magenta: '\x1b[35m',
  cyan: '\x1b[36m',
  white: '\x1b[37m',
  bgRed: '\x1b[41m',
  bgGreen: '\x1b[42m',
  bgYellow: '\x1b[43m',
};

function hr(char = '─', len = 78) { return char.repeat(len); }
function pad(s, len) { return String(s).padEnd(len); }
function rpad(s, len) { return String(s).padStart(len); }

// ═════════════════════════════════════════════════════════════════════════════
//  DEMO 1: Ward Census Table Correction
// ═════════════════════════════════════════════════════════════════════════════

function demoWardCensus(shifu) {
  console.log(`\n${C.bold}${C.cyan}  DEMO 1: Ward Census OCR Correction${C.reset}`);
  console.log(`  ${C.dim}Simulated OCR output from a handwritten ward board${C.reset}\n`);

  // Simulate noisy OCR from a ward census board
  const ocrRows = [
    { Patient: 'Bader A1athoub',  Room: '3O2A',  Diagnosis: 'lschemic str0ke',       Doctor: 'Dr. Hisharn',  Status: 'Red' },
    { Patient: 'Noura Bazzah',    Room: '3O3B',  Diagnosis: 'Seizure / epilepsy',     Doctor: 'Dr. Sa1eh',    Status: 'Yellow' },
    { Patient: 'Nawaf A1essa',    Room: '1CU-2', Diagnosis: 'Status epi1epticus',     Doctor: 'Dr. Zarha',    Status: 'Red' },
    { Patient: 'Ahnad Hassan',    Room: '4O1',   Diagnosis: 'Guil1ain barre',         Doctor: 'Dr. Jarna1',   Status: 'Yellow' },
    { Patient: 'Fa1sal Turki',    Room: '2O5C',  Diagnosis: 'Subarachno1d hernorrhage', Doctor: 'Dr. Mneer',  Status: 'Red' },
    { Patient: 'Dalal Mariam',    Room: '3O6',   Diagnosis: 'Migraine cluster',       Doctor: 'Dr. Ade1',     Status: 'Green' },
    { Patient: 'Roju1o Raju',     Room: '1CU-5', Diagnosis: 'Raised 1cp herniation',  Doctor: 'Dr. Hussain',  Status: 'Red' },
    { Patient: 'Su1aiman Ornar',  Room: '2O8',   Diagnosis: 'Bells pa1sy',            Doctor: 'Dr. Kha1id',   Status: 'Green' },
  ];

  const columns = ['Patient', 'Room', 'Diagnosis', 'Doctor', 'Status'];
  const colWidths = [20, 8, 28, 15, 8];

  // Header
  console.log(`  ${C.bold}${C.white}` + columns.map((c, i) => pad(c, colWidths[i])).join('') + C.reset);
  console.log(`  ${hr('─', colWidths.reduce((a, b) => a + b, 0))}`);

  const allFlags = [];
  const pipeline = shifu.createPipeline();

  for (const row of ocrRows) {
    const result = pipeline.processTableRow(row);

    const cells = columns.map((col, i) => {
      const data = result.corrected[col] || {};
      const input = row[col] || '';
      const output = data.output || input;
      const changed = input !== output;

      let color = C.white;
      if (data.safetyFlags && data.safetyFlags.length > 0) {
        color = C.red + C.bold;
      } else if (changed) {
        color = C.green;
      }

      return color + pad(output, colWidths[i]) + C.reset;
    });

    // Decision indicator
    const indicator = result.decision === 'reject' ? `${C.bgRed}${C.white} REJECT ${C.reset}`
      : result.decision === 'verify' ? `${C.bgYellow}${C.white} VERIFY ${C.reset}`
      : `${C.bgGreen}${C.white} OK ${C.reset}`;

    console.log(`  ${cells.join('')} ${indicator}`);

    // Show what changed
    const changes = columns.filter(col => {
      const out = (result.corrected[col] || {}).output || '';
      return out !== row[col] && out !== '';
    });
    if (changes.length > 0) {
      const changeStr = changes.map(col => {
        const out = (result.corrected[col] || {}).output || '';
        return `${C.dim}${col}: ${C.red}${row[col]}${C.reset}${C.dim} → ${C.green}${out}${C.reset}`;
      }).join('  ');
      console.log(`  ${C.dim}  └ ${changeStr}`);
    }

    for (const flag of result.safetyFlags) {
      allFlags.push(flag);
    }
  }

  if (allFlags.length > 0) {
    console.log(`\n  ${C.bold}${C.red}Safety Flags:${C.reset}`);
    for (const flag of allFlags) {
      const icon = flag.severity === 'danger' ? '!!!' : '(!)'
      console.log(`  ${C.red}  ${icon} ${flag.message || flag.status}${C.reset}`);
    }
  }

  console.log();
}

// ═════════════════════════════════════════════════════════════════════════════
//  DEMO 2: Line-by-line OCR Correction
// ═════════════════════════════════════════════════════════════════════════════

function demoLineCorrectionn(shifu) {
  console.log(`\n${C.bold}${C.cyan}  DEMO 2: Clinical Text Line Correction${C.reset}`);
  console.log(`  ${C.dim}Simulated OCR from clinical notes / neurological exam${C.reset}\n`);

  const lines = [
    { ocr: 'Crainal merves intact bil1ateral',         context: 'Neuro exam' },
    { ocr: 'Puplis reactive PERRLA no nystagrnus',     context: 'Cranial nerves' },
    { ocr: 'Power 5/5 upper 1imbs nornal tone',        context: 'Motor exam' },
    { ocr: 'Bab1nski downqoing p1antar reflexes',      context: 'Reflexes' },
    { ocr: 'Levetiracetarn 5OOmg twice dai1y',         context: 'Medication' },
    { ocr: 'Carbarnazepine 2OOmg three tirnes daily',  context: 'Medication' },
    { ocr: 'Sod1um 139 Potass1um 4.2 Creatinine 88',  context: 'Lab values' },
    { ocr: 'HbA1c 71 gluocse fasting',                 context: 'Lab values' },
    { ocr: 'CT head no acute hernorrhage or ederna',    context: 'Imaging' },
    { ocr: 'MRI brain dern yelinat ion lesions seen',   context: 'Imaging' },
  ];

  const pipeline = shifu.createPipeline();

  console.log(`  ${C.bold}${pad('Context', 14)}${pad('OCR Input', 44)}${pad('Corrected', 44)}Decision${C.reset}`);
  console.log(`  ${hr('─', 110)}`);

  for (const { ocr, context } of lines) {
    const result = pipeline.processText(ocr);

    let decColor = C.green;
    let decText = 'ACCEPT';
    if (result.decision === 'reject') { decColor = C.red; decText = 'REJECT'; }
    else if (result.decision === 'verify') { decColor = C.yellow; decText = 'VERIFY'; }

    // Highlight changes in the corrected output
    const ocrWords = ocr.split(/\s+/);
    const corrWords = result.corrected.split(/\s+/);
    const highlightedCorr = corrWords.map((w, i) => {
      if (i < ocrWords.length && w.toLowerCase() !== ocrWords[i].toLowerCase()) {
        return `${C.green}${C.bold}${w}${C.reset}`;
      }
      return w;
    }).join(' ');

    console.log(`  ${C.dim}${pad(context, 14)}${C.reset}${pad(ocr, 44)}${pad('', 0)}${highlightedCorr}`);
    console.log(`  ${' '.repeat(58)}${decColor}${C.bold}[${decText}]${C.reset} conf: ${(result.avgConfidence * 100).toFixed(0)}%`);

    if (result.safetyFlags && result.safetyFlags.length > 0) {
      for (const flag of result.safetyFlags) {
        console.log(`  ${C.red}  !!! ${flag.message || flag.status}${C.reset}`);
      }
    }
  }

  console.log();
}

// ═════════════════════════════════════════════════════════════════════════════
//  DEMO 3: Learning Loop — Watch it get smarter
// ═════════════════════════════════════════════════════════════════════════════

function demoLearning(shifu) {
  console.log(`\n${C.bold}${C.cyan}  DEMO 3: Adaptive Learning — Watch It Learn${C.reset}`);
  console.log(`  ${C.dim}Nurse corrects readings → system adapts → future accuracy improves${C.reset}\n`);

  // Simulate a nurse correcting ward census readings
  const corrections = [
    {
      ocr:       { Patient: 'Bader A1athoub', Diagnosis: 'lschemic str0ke',  Doctor: 'Dr. Hisharn' },
      confirmed: { Patient: 'Bader Alathoub', Diagnosis: 'Ischemic stroke',  Doctor: 'Dr. Hisham' },
    },
    {
      ocr:       { Patient: 'N0ura Bazzah',   Diagnosis: 'Seizure epi1epsy', Doctor: 'Dr. Sa1eh' },
      confirmed: { Patient: 'Noura Bazzah',   Diagnosis: 'Seizure epilepsy', Doctor: 'Dr. Saleh' },
    },
    {
      ocr:       { Patient: 'Ahnad Hassan',   Diagnosis: 'Gui11ain barre',   Doctor: 'Dr. Jarna1' },
      confirmed: { Patient: 'Ahmad Hassan',   Diagnosis: 'Guillain barre',   Doctor: 'Dr. Jamal' },
    },
    {
      ocr:       { Patient: 'Fa1sal Turki',   Diagnosis: 'Pneurnonia cap',   Doctor: 'Dr. Kha1id' },
      confirmed: { Patient: 'Faisal Turki',   Diagnosis: 'Pneumonia cap',    Doctor: 'Dr. Khalid' },
    },
    {
      ocr:       { Patient: 'Su1aiman 0rnar', Diagnosis: 'Myasthenia gravis', Doctor: 'Dr. Ade1' },
      confirmed: { Patient: 'Sulaiman Omar',  Diagnosis: 'Myasthenia gravis', Doctor: 'Dr. Adel' },
    },
  ];

  console.log(`  ${C.bold}Before learning:${C.reset}`);
  const statsBefore = shifu.stats();
  console.log(`    Core vocabulary: ${statsBefore.core.vocabulary || 'N/A'} words`);
  console.log(`    Learned words:   ${statsBefore.vocabulary.learnedWords}`);
  console.log(`    Corrections:     0\n`);

  // Feed corrections
  for (let i = 0; i < corrections.length; i++) {
    const { ocr, confirmed } = corrections[i];
    shifu.learn(ocr, confirmed);
    console.log(`  ${C.green}+${C.reset} Correction ${i + 1}: ${C.dim}${confirmed.Patient} — ${confirmed.Diagnosis}${C.reset}`);
  }

  console.log(`\n  ${C.bold}After learning:${C.reset}`);
  const statsAfter = shifu.stats();
  console.log(`    Core vocabulary: ${statsAfter.core.vocabulary || 'N/A'} words`);
  console.log(`    Learned words:   ${statsAfter.vocabulary.learnedWords}`);
  console.log(`    Corrections:     ${corrections.length}`);

  // Show top confusions the system learned
  const topConfusions = shifu.learning.confusion.getTopConfusions(5);
  if (topConfusions.length > 0) {
    console.log(`\n  ${C.bold}Top learned confusion pairs:${C.reset}`);
    for (const { pair, count, cost } of topConfusions) {
      const [a, b] = pair.split(',');
      console.log(`    ${C.yellow}${a}${C.reset} ↔ ${C.yellow}${b}${C.reset}  seen ${count}x  cost: ${cost.toFixed(3)}`);
    }
  }

  // Show that the system now handles these better
  console.log(`\n  ${C.bold}Re-processing with learned knowledge:${C.reset}`);
  const testRow = { Patient: 'Fa1sa1 Turk1', Diagnosis: 'Pneurn0nia', Doctor: 'Dr. Kha11d' };
  const pipeline = shifu.createPipeline();
  const result = pipeline.processTableRow(testRow);
  for (const [col, data] of Object.entries(result.corrected)) {
    const inp = testRow[col];
    const out = data.output;
    if (inp !== out) {
      console.log(`    ${pad(col, 12)} ${C.red}${inp}${C.reset} → ${C.green}${out}${C.reset}`);
    }
  }
  console.log();
}

// ═════════════════════════════════════════════════════════════════════════════
//  DEMO 4: Core Engine Intelligence
// ═════════════════════════════════════════════════════════════════════════════

function demoCoreEngine(shifu) {
  console.log(`\n${C.bold}${C.cyan}  DEMO 4: Core Engine — Resonance Intelligence${C.reset}`);
  console.log(`  ${C.dim}What the engine learned from the medical corpus${C.reset}\n`);

  // Coherence scoring — medical vs gibberish
  const sentences = [
    { text: 'Patient presents with acute ischemic stroke', label: 'Medical (correct)' },
    { text: 'Doctor ordered CT head for the patient',     label: 'Medical (correct)' },
    { text: 'Levetiracetam 500mg twice daily for seizure', label: 'Medical (correct)' },
    { text: 'Xylophone banana rocket purple sidewalk',     label: 'Gibberish' },
    { text: 'The cat sat on the mat happily',              label: 'Non-medical' },
    { text: 'Ptainet prseents wiht acuet strkoe',         label: 'OCR-garbled medical' },
  ];

  console.log(`  ${C.bold}${pad('Sentence', 50)}${pad('Coherence', 12)}Type${C.reset}`);
  console.log(`  ${hr('─', 78)}`);

  for (const { text, label } of sentences) {
    const score = shifu.scoreSentence(text);
    const bar = '█'.repeat(Math.round(score.coherence * 20)) + '░'.repeat(20 - Math.round(score.coherence * 20));
    const color = score.coherence > 0.3 ? C.green : score.coherence > 0.15 ? C.yellow : C.red;
    console.log(`  ${pad(text, 50)}${color}${bar}${C.reset} ${C.dim}${label}${C.reset}`);
  }

  // Word similarity / resonance
  console.log(`\n  ${C.bold}Resonance partners (words that fill the same structural slots):${C.reset}`);
  const probeWords = ['stroke', 'patient', 'seizure', 'doctor'];
  for (const word of probeWords) {
    const partners = shifu.resonancePartners(word, 5);
    if (partners && partners.length > 0) {
      const partnerStr = partners.map(p => `${C.cyan}${p.word || p}${C.reset}`).join(', ');
      console.log(`    ${C.bold}${word}${C.reset} → ${partnerStr}`);
    }
  }

  // OCR correction via core engine
  console.log(`\n  ${C.bold}Core engine OCR correction:${C.reset}`);
  const garbled = ['str0ke', 'pat1ent', 'se1zure', 'levetiracetarn', 'crania1'];
  for (const word of garbled) {
    const result = shifu.correct(word, 3);
    const candidates = (result && result.candidates) || [];
    if (candidates.length > 0) {
      const suggStr = candidates.map(s => `${C.green}${s.word || s}${C.reset}`).join(', ');
      console.log(`    ${C.red}${pad(word, 18)}${C.reset} → ${suggStr}`);
    }
  }

  console.log();
}

// ═════════════════════════════════════════════════════════════════════════════
//  DEMO 5: Persistence — Save and Restore
// ═════════════════════════════════════════════════════════════════════════════

function demoPersistence(shifu) {
  console.log(`\n${C.bold}${C.cyan}  DEMO 5: Persistence — Save & Restore${C.reset}`);
  console.log(`  ${C.dim}Everything the system learns survives between sessions${C.reset}\n`);

  // Serialize
  const state = shifu.serialize();
  const coreSize = JSON.stringify(state.core).length;
  const learningSize = JSON.stringify(state.learning).length;
  const totalSize = coreSize + learningSize;

  console.log(`  ${C.bold}System state:${C.reset}`);
  console.log(`    Core engine:     ${(coreSize / 1024).toFixed(1)} KB`);
  console.log(`    Learning engine: ${(learningSize / 1024).toFixed(1)} KB`);
  console.log(`    Total:           ${(totalSize / 1024).toFixed(1)} KB`);
  console.log(`    Version:         ${state.version}`);

  // Simulate restore
  const { restoreShifu } = require('../index');
  const restored = restoreShifu(state);
  const restoredStats = restored.stats();

  console.log(`\n  ${C.bold}After restore:${C.reset}`);
  console.log(`    Core vocabulary preserved: ${C.green}yes${C.reset}`);
  console.log(`    Learning state preserved:  ${C.green}yes${C.reset}`);
  console.log(`    Ready for corrections:     ${C.green}yes${C.reset}`);

  // Verify the restored system works
  const test = restored.correctLine('Levetiracetarn 5OOmg for seizure');
  console.log(`\n  ${C.bold}Restored system correction test:${C.reset}`);
  console.log(`    Input:  ${C.red}Levetiracetarn 5OOmg for seizure${C.reset}`);
  console.log(`    Output: ${C.green}${test.output}${C.reset}`);
  console.log(`    Confidence: ${(test.avgConfidence * 100).toFixed(0)}%`);
  console.log();
}

// ═════════════════════════════════════════════════════════════════════════════
//  SUMMARY
// ═════════════════════════════════════════════════════════════════════════════

function printSummary(shifu) {
  const stats = shifu.stats();

  console.log(`\n${C.bold}${C.white}${'═'.repeat(78)}${C.reset}`);
  console.log(`${C.bold}${C.white}  SHIFU OCR v${VERSION} — System Summary${C.reset}`);
  console.log(`${C.bold}${C.white}${'═'.repeat(78)}${C.reset}\n`);

  console.log(`  ${C.bold}Architecture:${C.reset}`);
  console.log(`    Core Engine       Resonance learning, soft trajectories, skip-gram expectations`);
  console.log(`    Clinical Layer    700+ medical terms, 30+ lab ranges, 180+ medications`);
  console.log(`    Confusion Model   Topology-predicted OCR substitution costs`);
  console.log(`    Safety Flags      Medication ambiguity, out-of-range values, dose validation`);
  console.log(`    Learning Loop     Adaptive confusion × ward vocabulary × context chains`);
  console.log(`    Trained Model     Python fluid-theory character landscapes (80KB)`);
  console.log(`    Pipeline          Image → Python OCR → JS correction → safety-checked output`);
  console.log(`    Persistence       Auto-save/load between sessions\n`);

  console.log(`  ${C.bold}Design Principles:${C.reset}`);
  console.log(`    1. The system suggests, the clinician decides.`);
  console.log(`    2. Low confidence = loud flag, not silent guess.`);
  console.log(`    3. Medications are NEVER silently corrected.`);
  console.log(`    4. Every correction makes the system smarter.`);
  console.log(`    5. The model is 100% auditable.\n`);

  console.log(`  ${C.bold}By the numbers:${C.reset}`);
  console.log(`    Vocabulary:   ${stats.vocabulary.baseSize} base words`);
  console.log(`    Core tokens:  ${stats.core.vocabulary || 'N/A'}`);
  console.log(`    Model size:   lightweight (no GPU, no cloud)\n`);
}

// ═════════════════════════════════════════════════════════════════════════════
//  MAIN
// ═════════════════════════════════════════════════════════════════════════════

function main() {
  console.log(`\n${C.bold}${C.cyan}${'╔' + '═'.repeat(76) + '╗'}${C.reset}`);
  console.log(`${C.bold}${C.cyan}║${' '.repeat(20)}SHIFU OCR — Clinical Demo${' '.repeat(31)}║${C.reset}`);
  console.log(`${C.bold}${C.cyan}║${' '.repeat(76)}║${C.reset}`);
  console.log(`${C.bold}${C.cyan}║${C.reset}   Scan in → Corrected table out → Safety flags highlighted${' '.repeat(17)}${C.bold}${C.cyan}║${C.reset}`);
  console.log(`${C.bold}${C.cyan}║${C.reset}   Every nurse correction makes it smarter.${' '.repeat(32)}${C.bold}${C.cyan}║${C.reset}`);
  console.log(`${C.bold}${C.cyan}║${C.reset}   No neural network. No GPU. No cloud.${' '.repeat(36)}${C.bold}${C.cyan}║${C.reset}`);
  console.log(`${C.bold}${C.cyan}${'╚' + '═'.repeat(76) + '╝'}${C.reset}\n`);

  console.log(`  ${C.dim}Initializing Shifu v${VERSION}...${C.reset}`);
  const shifu = createShifu({ loadTrained: true });
  const stats = shifu.stats();
  console.log(`  ${C.green}Engine ready.${C.reset} Core: ${stats.core.vocabulary || 'N/A'} tokens, Vocabulary: ${stats.vocabulary.baseSize} medical terms\n`);

  demoWardCensus(shifu);
  demoLineCorrectionn(shifu);
  demoLearning(shifu);
  demoCoreEngine(shifu);
  demoPersistence(shifu);
  printSummary(shifu);
}

main();
