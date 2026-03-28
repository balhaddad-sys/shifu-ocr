// SHIFU — The Unified System
// OCR text in → the field reads it → corrected text out.

const { ShifuEngine, ocrDist, fieldCosine } = require('./engine');
const corpus = require('../data/medical_corpus');

let _engine = null;

function engine() {
  if (!_engine) {
    _engine = new ShifuEngine();
    _engine.feedBatch(corpus);
  }
  return _engine;
}

function generateCandidates(eng, word, maxDist, maxCandidates) {
  if (maxDist === undefined) maxDist = Math.max(word.length * 0.5, 2);
  if (maxCandidates === undefined) maxCandidates = 5;
  const candidates = [];
  for (const known of Object.keys(eng.wf)) {
    const lenDiff = Math.abs(known.length - word.length);
    if (lenDiff > Math.max(2, word.length * 0.4)) continue;
    if (known.length < Math.ceil(word.length * 0.6)) continue;
    const d = ocrDist(word, known);
    if (d <= maxDist && d > 0) candidates.push({ word: known, distance: d });
  }
  if (candidates.length === 0) return [];
  candidates.sort((a, b) => a.distance - b.distance);
  const bestDist = candidates[0].distance;
  const cutoff = bestDist * 1.5 + 0.5;
  return candidates.filter(c => c.distance <= cutoff).slice(0, maxCandidates).map(c => c.word);
}

function process(text) {
  const eng = engine();
  const words = (text.toLowerCase().match(/[a-z0-9.]+/g) || []).filter(w => w.length > 0);
  if (words.length === 0) return { original: text, corrected: '', words: [], decisions: [] };
  const candidatesMap = {};
  for (let i = 0; i < words.length; i++) {
    const w = words[i];
    if (/^[\d.]+$/.test(w)) continue;
    if (eng._meaningRichness(w) > 0) continue;
    const candidates = generateCandidates(eng, w);
    if (candidates.length > 0) candidatesMap[i] = candidates;
  }
  const result = eng.read(words.join(' '), candidatesMap);
  return { original: text, corrected: result.corrected, words: result.words, decisions: result.decisions, fieldSize: result.field.size };
}

function processWithCandidates(text, candidatesMap) {
  const result = engine().read(text, candidatesMap);
  return { original: text, corrected: result.corrected, words: result.words, decisions: result.decisions, fieldSize: result.field.size };
}

function score(text) {
  const result = engine().scoreSentence(text);
  return { text, coherence: result.coherence, corrected: result.correctedCoherence, settled: result.settledCoherence };
}

function compare(a, b) { return engine().compareSentences(a, b); }
function infer(word) { return engine().infer(word); }

function validate(original, corrected) {
  const eng = engine();
  const origScore = eng.scoreSentence(original);
  const corrScore = eng.scoreSentence(corrected);
  const delta = corrScore.correctedCoherence - origScore.correctedCoherence;
  return {
    original: { text: original, coherence: origScore.correctedCoherence },
    corrected: { text: corrected, coherence: corrScore.correctedCoherence },
    delta, recommendation: delta > 0.001 ? 'accept' : delta < -0.001 ? 'reject' : 'neutral',
  };
}

function processOCRLine(ocrResult) {
  const eng = engine();
  const rawText = ocrResult.text || '';
  const characters = ocrResult.characters || [];
  if (!rawText || characters.length === 0) return { original: rawText, corrected: rawText, decisions: [] };
  const words = rawText.split(/\s+/).filter(w => w.length > 0);
  const candidatesMap = {};
  let charIdx = 0;
  for (let wi = 0; wi < words.length; wi++) {
    const word = words[wi];
    const wordChars = [];
    while (charIdx < characters.length && wordChars.length < word.length) {
      const ch = characters[charIdx];
      if (ch && ch.char && ch.char.trim()) wordChars.push(ch);
      charIdx++;
    }
    while (charIdx < characters.length && characters[charIdx] && (!characters[charIdx].char || characters[charIdx].char.trim() === '')) charIdx++;
    const hasUncertain = wordChars.some(ch => (ch.confidence || 0) < 0.5);
    if (hasUncertain && eng._meaningRichness(word.toLowerCase()) === 0) {
      const cands = generateCandidates(eng, word.toLowerCase());
      if (cands.length > 0) candidatesMap[wi] = cands;
    }
  }
  const result = eng.read(words.join(' '), candidatesMap);
  return { original: rawText, corrected: result.corrected, words: result.words, decisions: result.decisions, fieldSize: result.field.size };
}

function processTable(rows) {
  const correctedRows = [];
  for (const row of rows) {
    const correctedRow = [];
    for (const cell of row) {
      if (!cell || !cell.trim()) { correctedRow.push(cell || ''); continue; }
      correctedRow.push(process(cell).corrected);
    }
    correctedRows.push(correctedRow);
  }
  return { original: rows, corrected: correctedRows };
}

function stats() { return engine().stats(); }

module.exports = { process, processWithCandidates, processOCRLine, processTable, score, compare, infer, validate, stats, engine };
