/**
 * SHIFU INTEGRATION — Three connection points
 * 1. pipeline.js → calls Python OCR → gets raw text → calls shifu.read()
 * 2. server.js → /api/correct → calls shifu.process()
 * 3. loop.js → nurse correction → shifu.experience()
 */

const shifu = require('./shifu');
const path = require('path');
const fs = require('fs');

const BRAIN_PATH = path.join(__dirname, '..', 'data', 'shifu_brain.json');

function correctOCRResult(ocrResult) {
  const rawText = ocrResult.text || '';
  if (!rawText.trim()) return { ...ocrResult, corrected: '', decisions: [] };
  const result = shifu.process(rawText);
  return { ...ocrResult, originalText: rawText, text: result.corrected, corrected: result.corrected, decisions: result.decisions, fieldSize: result.fieldSize };
}

function correctTable(tableData) {
  if (!tableData || !tableData.rows) return tableData;
  const result = shifu.processTable(tableData.rows);
  return { ...tableData, rows: result.corrected };
}

function handleCorrectRequest(req, res) {
  const text = (req.body && req.body.text) || '';
  if (!text.trim()) return res.json({ input: text, output: '', decisions: [] });
  const result = shifu.process(text);
  res.json({ input: text, output: result.corrected, decisions: result.decisions, fieldSize: result.fieldSize, stats: shifu.stats() });
}

function handleScoreRequest(req, res) {
  const text = (req.body && req.body.text) || '';
  res.json(shifu.score(text));
}

function handleCompareRequest(req, res) {
  const { a, b } = req.body || {};
  res.json(shifu.compare(a || '', b || ''));
}

function handleStatsRequest(req, res) { res.json(shifu.stats()); }

function learn(originalText, correctedText) {
  const eng = shifu.engine();
  eng.feed(correctedText);
  const validation = shifu.validate(originalText, correctedText);
  return { learned: true, validation };
}

function save() {
  try {
    const eng = shifu.engine();
    fs.writeFileSync(BRAIN_PATH, eng.serialize());
    console.log('[shifu] Saved brain');
  } catch (e) { console.error('[shifu] Save failed:', e.message); }
}

module.exports = { correctOCRResult, correctTable, handleCorrectRequest, handleScoreRequest, handleCompareRequest, handleStatsRequest, learn, save };
