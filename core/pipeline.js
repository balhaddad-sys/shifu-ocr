// Shifu Pipeline — End-to-end: Image → Python OCR → JS Correction → Output
//
// Bridges the Python fluid-theory OCR engine with the JS correction engine.
// Python does: image → raw characters (medium displacement)
// JS does: raw text → corrected text (resonance + clinical vocabulary + safety)

const { spawn } = require('child_process');
const path = require('path');
const fs = require('fs');

const PYTHON_SCRIPT = path.join(__dirname, '..', 'shifu_ocr', 'pipeline_worker.py');

class ShifuPipeline {
  constructor(shifu, options = {}) {
    this.shifu = shifu; // The full createShifu() system
    this.pythonCmd = options.python || 'python';
    this.modelPath = options.modelPath || path.join(__dirname, '..', 'shifu_ocr', 'trained_model.json');
    this.tempDir = options.tempDir || path.join(__dirname, '..', '.tmp');
    this._ensureTemp();
  }

  _ensureTemp() {
    if (!fs.existsSync(this.tempDir)) fs.mkdirSync(this.tempDir, { recursive: true });
  }

  /**
   * Process a single image file through the full pipeline.
   * Image → Python OCR → JS correction → safety-checked output.
   *
   * @param {string} imagePath - Path to the image file
   * @param {object} options - { format: 'line'|'table', columns: [...] }
   * @returns {Promise<object>} Full pipeline result
   */
  async processImage(imagePath, options = {}) {
    const format = options.format || 'line';

    // Step 1: Python OCR — extract raw text from image
    const ocrResult = await this._runPythonOCR(imagePath, format);

    // Step 2: JS Correction — clinical vocabulary + confusion model + safety
    if (format === 'table') {
      return this._processTableResult(ocrResult, options);
    }
    return this._processLineResult(ocrResult);
  }

  /**
   * Process multiple images (e.g., a ward census with one image per row).
   */
  async processImages(imagePaths, options = {}) {
    const results = [];
    for (const imgPath of imagePaths) {
      const result = await this.processImage(imgPath, options);
      results.push(result);
    }
    return results;
  }

  /**
   * Process raw OCR text (skip Python step — useful when OCR is done externally).
   */
  processText(rawText, options = {}) {
    if (options.correct === false) {
      const coherence = this.shifu.scoreSentence(rawText);
      return {
        input: rawText,
        corrected: rawText,
        safetyFlags: [],
        coherence: coherence.coherence,
        decision: 'accept',
        hasDangers: false,
        hasWarnings: false,
        avgConfidence: 1,
      };
    }

    const corrected = this.shifu.correctLine(rawText, options);
    const coherence = this.shifu.scoreSentence(corrected.output);
    const decision = this.shifu.assessConfidence(corrected);

    return {
      input: rawText,
      corrected: corrected.output,
      words: corrected.words,
      safetyFlags: corrected.safetyFlags,
      coherence: coherence.coherence,
      decision,
      hasDangers: corrected.hasDangers,
      hasWarnings: corrected.hasWarnings,
      avgConfidence: corrected.avgConfidence,
    };
  }

  /**
   * Process a table row (ward census format).
   * Each field gets column-aware correction.
   */
  processTableRow(row, options = {}) {
    // Standard corrector has all the guards (titles, alphanumeric, safety)
    const standard = this.shifu.correctTableRow(row, options);

    // Enhance with adaptive coherence scoring
    const adaptive = this.shifu.correctRowAdaptive(row, options);

    const merged = {};
    for (const [col, data] of Object.entries(standard.corrected)) {
      const adaptiveData = (adaptive.corrected || {})[col] || {};
      merged[col] = {
        input: data.input,
        output: data.output,
        coherence: adaptiveData.coherence,
        words: data.words || [],
        safetyFlags: data.safetyFlags || [],
      };
    }

    const allFlags = standard.safetyFlags || [];
    // Check all words across all columns for low-confidence / unknown
    const allWords = Object.values(merged).flatMap(d => d.words || []);
    const hasUncertain = allWords.some(w => w.flag === 'low_confidence' || w.flag === 'unknown' || w.flag === 'corrected_verify');

    const decision = allFlags.some(f => f.severity === 'danger') ? 'reject'
      : (allFlags.some(f => f.severity === 'warning' || f.severity === 'error') || hasUncertain) ? 'verify' : 'accept';

    return {
      corrected: merged,
      safetyFlags: allFlags,
      decision,
      hasDangers: standard.hasDangers,
      hasWarnings: standard.hasWarnings || hasUncertain,
    };
  }

  /**
   * Learn from a confirmed correction (nurse/doctor verified).
   */
  learnFromCorrection(ocrRow, confirmedRow) {
    return this.shifu.learn(ocrRow, confirmedRow);
  }

  /**
   * Check if the Python OCR backend is available.
   * Returns { available, version, error }.
   */
  async checkPythonOCR() {
    return new Promise((resolve) => {
      let proc;
      try {
        proc = spawn(this.pythonCmd, ['--version'], { timeout: 5000 });
      } catch (err) {
        resolve({ available: false, version: null, error: err.message });
        return;
      }
      let stdout = '';
      proc.stdout.on('data', d => { stdout += d.toString(); });
      proc.stderr.on('data', d => { stdout += d.toString(); }); // python --version writes to stderr on some systems
      proc.on('close', code => {
        if (code !== 0) {
          resolve({ available: false, version: null, error: `exit code ${code}` });
          return;
        }
        const version = stdout.trim();
        // Also check if the pipeline worker script exists
        const scriptExists = fs.existsSync(PYTHON_SCRIPT);
        resolve({
          available: scriptExists,
          version,
          scriptExists,
          error: scriptExists ? null : `Pipeline script not found: ${PYTHON_SCRIPT}`,
        });
      });
      proc.on('error', err => {
        resolve({ available: false, version: null, error: err.message });
      });
    });
  }

  // ─── Internal ─────────────────────────────────────────────────

  _processLineResult(ocrResult) {
    const lines = ocrResult.lines || [ocrResult.text || ''];
    // Pass ocrSource flag so the corrector knows this is OCR output and can be more aggressive
    const results = lines.map(line => line ? this.processText(line, { ocrSource: true }) : null).filter(Boolean);

    // Empty or failed OCR = verify (never accept nothing)
    if (results.length === 0 || ocrResult.fallback) {
      return { raw: ocrResult, lines: results, overallDecision: 'verify' };
    }

    const hasVerify = results.some(r => r.decision === 'verify');
    return {
      raw: ocrResult,
      lines: results,
      overallDecision: results.some(r => r.hasDangers) ? 'reject'
        : (results.some(r => r.hasWarnings) || hasVerify) ? 'verify' : 'accept',
    };
  }

  _processTableResult(ocrResult, options) {
    const rows = ocrResult.rows || [];
    const columns = options.columns || ocrResult.columns || [];
    const results = rows.map(row => {
      const rowObj = {};
      columns.forEach((col, i) => { rowObj[col] = row[i] || ''; });
      return this.processTableRow(rowObj);
    });
    if (results.length === 0 || ocrResult.fallback) {
      return { raw: ocrResult, columns, rows: results, overallDecision: 'verify' };
    }
    return {
      raw: ocrResult,
      columns,
      rows: results,
      overallDecision: results.some(r => r.hasDangers) ? 'reject'
        : results.some(r => r.hasWarnings) ? 'verify' : 'accept',
    };
  }

  _runPythonOCR(imagePath, format) {
    return new Promise((resolve) => {
      const args = [PYTHON_SCRIPT, '--image', imagePath, '--format', format, '--model', this.modelPath];
      let proc;
      try {
        proc = spawn(this.pythonCmd, args, { timeout: 60000 });
      } catch (err) {
        // Synchronous spawn failure (invalid executable, etc.) — fall back to JS-only mode
        resolve({ text: '', lines: [], rows: [], fallback: true, error: err.message });
        return;
      }

      let stdout = '';
      let stderr = '';
      proc.stdout.on('data', d => { stdout += d.toString(); });
      proc.stderr.on('data', d => { stderr += d.toString(); });

      proc.on('close', code => {
        if (code !== 0) {
          // If Python OCR fails, return empty result (JS-only mode)
          console.warn(`Python OCR exited ${code}: ${stderr.slice(0, 200)}`);
          resolve({ text: '', lines: [], rows: [], fallback: true, error: stderr.slice(0, 500) });
          return;
        }
        try {
          resolve(JSON.parse(stdout));
        } catch (e) {
          resolve({ text: stdout.trim(), lines: [stdout.trim()], rows: [] });
        }
      });

      proc.on('error', err => {
        console.warn(`Python OCR spawn failed: ${err.message}`);
        resolve({ text: '', lines: [], rows: [], fallback: true, error: err.message });
      });
    });
  }
}

module.exports = { ShifuPipeline };
