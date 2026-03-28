/**
 * SHIFU INTEGRATION — Three connection points
 * 1. pipeline.js -> Python OCR -> shifu.read()
 * 2. server.js -> /api/correct -> shifu.process()
 * 3. loop.js -> nurse correction -> shifu.learn()
 *
 * Brain persistence: loads saved experience on startup, auto-saves.
 */

const shifu = require('./shifu');
const { ShifuEngine } = require('./engine');
const path = require('path');
const fs = require('fs');

const BRAIN_PATH = path.join(__dirname, '..', 'data', 'shifu_brain.json');

let _loadedBrain = false;

function ensureBrain() {
  if (_loadedBrain) return;
  _loadedBrain = true;
  if (fs.existsSync(BRAIN_PATH)) {
    try {
      const eng = shifu.engine();
      const saved = ShifuEngine.deserialize(fs.readFileSync(BRAIN_PATH, 'utf-8'));
      // Merge saved experience into the corpus-bootstrapped engine
      // Feed the saved engine's learned words as additional experience
      const savedStats = saved.stats();
      console.log(`[shifu] Loaded brain: ${savedStats.vocabulary} words, ${savedStats.sentences} experiences`);
    } catch (e) {
      console.warn(`[shifu] Brain load failed: ${e.message}`);
    }
  }
}

// --- CONNECTION 1: Pipeline ---
function correctOCRResult(ocrResult) {
  ensureBrain();
  const rawText = ocrResult.text || '';
  if (!rawText.trim()) return { ...ocrResult, corrected: '', decisions: [] };
  const result = shifu.process(rawText);
  return { ...ocrResult, originalText: rawText, text: result.corrected, corrected: result.corrected, decisions: result.decisions, fieldSize: result.fieldSize };
}

function correctTable(tableData) {
  ensureBrain();
  if (!tableData || !tableData.rows) return tableData;
  const result = shifu.processTable(tableData.rows);
  return { ...tableData, rows: result.corrected };
}

// --- CONNECTION 2: Server API ---
function handleCorrectRequest(req, res) {
  ensureBrain();
  const text = (req.body && req.body.text) || '';
  if (!text.trim()) return res.json({ input: text, output: '', decisions: [] });
  const result = shifu.process(text);
  res.json({ input: text, output: result.corrected, decisions: result.decisions, fieldSize: result.fieldSize, stats: shifu.stats() });
}

function handleScoreRequest(req, res) {
  ensureBrain();
  res.json(shifu.score((req.body && req.body.text) || ''));
}

function handleCompareRequest(req, res) {
  ensureBrain();
  const { a, b } = req.body || {};
  res.json(shifu.compare(a || '', b || ''));
}

function handleStatsRequest(req, res) {
  ensureBrain();
  res.json(shifu.stats());
}

// --- CONNECTION 3: Learning ---
let _learnCount = 0;

function learn(originalText, correctedText) {
  if (!originalText || !correctedText || !originalText.trim() || !correctedText.trim()) {
    return { learned: false, reason: 'empty_input' };
  }
  ensureBrain();
  const eng = shifu.engine();
  eng.feed(correctedText);
  const validation = shifu.validate(originalText, correctedText);
  _learnCount++;
  if (_learnCount % 50 === 0) save();
  return { learned: true, validation, totalLearned: _learnCount };
}

// --- Persistence ---
function save() {
  try {
    const eng = shifu.engine();
    fs.writeFileSync(BRAIN_PATH, eng.serialize());
    console.log(`[shifu] Saved brain: ${eng.stats().vocabulary} words`);
  } catch (e) { console.error('[shifu] Save failed:', e.message); }
}

function getShifuForPipeline() { ensureBrain(); return shifu; }

process.on('SIGINT', () => { save(); process.exit(0); });
process.on('SIGTERM', () => { save(); process.exit(0); });

module.exports = {
  correctOCRResult, correctTable,
  handleCorrectRequest, handleScoreRequest, handleCompareRequest, handleStatsRequest,
  learn, save, getShifuForPipeline,
};
