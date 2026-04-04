/**
 * SHIFU INTEGRATION GUIDE
 * ========================
 *
 * 3 edits to wire the unified field into the existing system:
 *
 * === CHANGE 1: pipeline.js ===
 *   const shifu = require('./core/integration');
 *   const corrected = shifu.correctOCRResult(ocrResult);
 *
 * === CHANGE 2: server.js ===
 *   app.post('/api/correct', shifu.handleCorrectRequest);
 *
 * === CHANGE 3: learning/loop.js ===
 *   const result = shifu.learn(original, corrected);
 *
 * KEEP: clinical/safety.js (domain rules, not learning)
 * KEEP: Python OCR pipeline (unchanged)
 * KEEP: UI (unchanged — same API interface)
 */
