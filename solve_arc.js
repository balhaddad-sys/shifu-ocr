// solve_arc.js — ARC solver using Shifu's full engine as a stone
//
// Phase 1: PRE-TRAIN — Feed ALL training tasks to build spatial grain.
//   The stone develops intuition for grid transformations.
// Phase 2: REACT — For each task, add task-specific examples,
//   then let the stone react to the test input.
//
// The stone's grain determines the response. No verification, no filtering.
// Just exposure and reaction.

const fs = require('fs');
const path = require('path');
const { ShifuEngine } = require('./core/engine');

const TASKS_DIR = path.join(__dirname, 'data', 'arc', 'training');

// ── Encode a grid transformation as sentences ──
// Each (input, output) pair generates multiple "sentences":
// - Row transitions: "i3 i0 i3 becomes o4 o0 o4"
// - Cell transitions with context: "center i3 north i0 south i3 east i0 west i3 becomes o4"

function encodePair(inp, out) {
  const sentences = [];
  const rows = inp.length;
  const cols = inp[0].length;

  // Row transitions
  for (let r = 0; r < rows; r++) {
    const inR = inp[r].map(v => 'c' + v).join(' ');
    const outR = out[r].map(v => 'c' + v).join(' ');
    if (inR !== outR) {
      sentences.push(inR + ' becomes ' + outR);
    }
  }

  // Cell transitions with spatial context
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      if (inp[r][c] === out[r][c]) continue; // only interesting changes
      const parts = ['cell c' + inp[r][c]];
      if (r > 0) parts.push('above c' + inp[r-1][c]);
      if (r < rows-1) parts.push('below c' + inp[r+1][c]);
      if (c > 0) parts.push('left c' + inp[r][c-1]);
      if (c < cols-1) parts.push('right c' + inp[r][c+1]);
      parts.push('becomes c' + out[r][c]);
      sentences.push(parts.join(' '));
    }
  }

  return sentences;
}

// ── Phase 1: Pre-train on ALL tasks ──
process.stderr.write('Phase 1: Pre-training on all tasks...\n');
const stone = new ShifuEngine();

const files = fs.readdirSync(TASKS_DIR).filter(f => f.endsWith('.json')).sort();
let totalSentences = 0;

for (const fname of files) {
  const task = JSON.parse(fs.readFileSync(path.join(TASKS_DIR, fname), 'utf8'));
  for (const ex of task.train) {
    if (ex.input.length !== ex.output.length) continue;
    if (ex.input[0].length !== ex.output[0].length) continue;
    const sentences = encodePair(ex.input, ex.output);
    for (const s of sentences) {
      stone.feed(s);
      totalSentences++;
    }
  }
}

process.stderr.write(`  Fed ${totalSentences} sentences, vocab: ${Object.keys(stone.wf).length} words\n`);

// ── Phase 2: React to each task ──
process.stderr.write('Phase 2: Reacting to tasks...\n');

let correct = 0;
let total = 0;
let attempted = 0;

for (const fname of files) {
  const task = JSON.parse(fs.readFileSync(path.join(TASKS_DIR, fname), 'utf8'));
  total++;

  const ex0 = task.train[0];
  if (ex0.input.length !== ex0.output.length) continue;
  if (ex0.input[0].length !== ex0.output[0].length) continue;
  const testIn = task.test[0].input;
  const testOut = task.test[0].output;
  if (testIn.length !== ex0.input.length) continue;
  if (testIn[0].length !== ex0.input[0].length) continue;

  attempted++;
  const rows = testIn.length;
  const cols = testIn[0].length;

  // For each cell: score candidate output values
  const predicted = [];
  for (let r = 0; r < rows; r++) {
    const row = [];
    for (let c = 0; c < cols; c++) {
      // Build context sentence for this cell
      const parts = ['cell c' + testIn[r][c]];
      if (r > 0) parts.push('above c' + testIn[r-1][c]);
      if (r < rows-1) parts.push('below c' + testIn[r+1][c]);
      if (c > 0) parts.push('left c' + testIn[r][c-1]);
      if (c < cols-1) parts.push('right c' + testIn[r][c+1]);
      parts.push('becomes');

      // Score each possible output color
      const candidates = new Set();
      for (let v = 0; v <= 9; v++) candidates.add(v);

      let bestVal = testIn[r][c];
      let bestCoherence = -1;

      for (const ov of candidates) {
        const sentence = [...parts, 'c' + ov].join(' ');
        const score = stone.scoreSentence(sentence);
        if (score.coherence > bestCoherence) {
          bestCoherence = score.coherence;
          bestVal = ov;
        }
      }

      row.push(bestVal);
    }
    predicted.push(row);
  }

  // Check
  let match = true;
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      if (predicted[r][c] !== testOut[r][c]) {
        match = false;
        break;
      }
    }
    if (!match) break;
  }

  if (match) {
    correct++;
    process.stderr.write(`  PASS ${fname}\n`);
  }
}

process.stderr.write(`\nShifu Stone ARC: ${correct}/${total} (${(100*correct/total).toFixed(1)}%) [${attempted} attempted]\n`);
console.log(JSON.stringify({ correct, total, attempted }));
