// Shifu Metrics System
// Tracks improvement over time. Answers: "Is the system getting smarter?"
//
// Three measurement dimensions:
//   1. Raw accuracy: what % of words are correct before correction?
//   2. Corrected accuracy: what % are correct AFTER correction?
//   3. Clinical safety: how many dangerous errors were caught?
//
// Plus: improvement curves, per-category breakdowns, comparison baselines.

const path = require('path');
const fs = require('fs');

class MetricsTracker {
  constructor(options = {}) {
    this.stateDir = options.stateDir || path.join(__dirname, '..', '.state');
    this.data = this._load();
  }

  /**
   * Record a single correction event with ground truth.
   * This is the fundamental measurement unit.
   *
   * @param {object} event
   * @param {string} event.ocrText - Raw OCR output
   * @param {string} event.correctedText - Shifu's correction
   * @param {string} event.groundTruth - Human-verified correct text
   * @param {object} event.meta - Optional metadata (column, source, etc.)
   */
  record(event) {
    const { ocrText, correctedText, groundTruth, meta = {} } = event;

    const ocrWords = (ocrText || '').split(/\s+/).filter(w => w.length > 0);
    const corrWords = (correctedText || '').split(/\s+/).filter(w => w.length > 0);
    const truthWords = (groundTruth || '').split(/\s+/).filter(w => w.length > 0);

    // LCS-based alignment for accurate scoring with insertions/deletions
    const rawCorrect = this._lcsCount(ocrWords, truthWords);
    const correctedCorrect = this._lcsCount(corrWords, truthWords);
    // Total = max tokens across all three (unmatched tokens in any set are errors)
    const total = Math.max(ocrWords.length, corrWords.length, truthWords.length);

    // Per-token flip tracking: compare each truth token's match status in ocr vs corrected
    const { fixes: rawToCorrectFlips, regressions: correctToWrongFlips } =
      this._countFlips(ocrWords, corrWords, truthWords);

    const entry = {
      timestamp: new Date().toISOString(),
      total,
      rawCorrect,
      correctedCorrect,
      rawAccuracy: total > 0 ? rawCorrect / total : 0,
      correctedAccuracy: total > 0 ? correctedCorrect / total : 0,
      improvement: total > 0 ? (correctedCorrect - rawCorrect) / total : 0,
      fixes: rawToCorrectFlips,
      regressions: correctToWrongFlips,
      column: meta.column || null,
      source: meta.source || null,
    };

    this.data.events.push(entry);
    this.data.totalEvents++;
    this.data.totalWords += total;
    this.data.totalRawCorrect += rawCorrect;
    this.data.totalCorrectedCorrect += correctedCorrect;
    this.data.totalFixes += rawToCorrectFlips;
    this.data.totalRegressions += correctToWrongFlips;

    // Per-column tracking
    if (meta.column) {
      if (!this.data.byColumn[meta.column]) {
        this.data.byColumn[meta.column] = { total: 0, rawCorrect: 0, correctedCorrect: 0, fixes: 0, regressions: 0 };
      }
      const col = this.data.byColumn[meta.column];
      col.total += total;
      col.rawCorrect += rawCorrect;
      col.correctedCorrect += correctedCorrect;
      col.fixes += rawToCorrectFlips;
      col.regressions += correctToWrongFlips;
    }

    this._save();
    return entry;
  }

  /**
   * Record a full ward census evaluation.
   */
  recordWardCensus(ocrRows, correctedRows, groundTruthRows) {
    const entries = [];
    for (let i = 0; i < Math.min(ocrRows.length, groundTruthRows.length); i++) {
      const ocrRow = ocrRows[i];
      const corrRow = correctedRows[i] || ocrRow;
      const truthRow = groundTruthRows[i];

      for (const col of Object.keys(truthRow)) {
        // Support both flat rows and pipeline-shaped rows (.corrected.Col.output)
        let corrText = '';
        if (corrRow.corrected && corrRow.corrected[col]) {
          corrText = typeof corrRow.corrected[col] === 'object' ? corrRow.corrected[col].output : corrRow.corrected[col];
        } else if (typeof corrRow[col] === 'object' && corrRow[col] !== null) {
          corrText = corrRow[col].output || '';
        } else {
          corrText = corrRow[col] || '';
        }

        const entry = this.record({
          ocrText: ocrRow[col] || '',
          correctedText: corrText,
          groundTruth: truthRow[col] || '',
          meta: { column: col, source: 'ward_census', row: i },
        });
        entries.push(entry);
      }
    }
    return entries;
  }

  /**
   * Get the full metrics summary.
   */
  summary() {
    const d = this.data;
    const rawAcc = d.totalWords > 0 ? d.totalRawCorrect / d.totalWords : 0;
    const corrAcc = d.totalWords > 0 ? d.totalCorrectedCorrect / d.totalWords : 0;

    return {
      // Overall
      totalEvents: d.totalEvents,
      totalWords: d.totalWords,
      rawAccuracy: rawAcc,
      correctedAccuracy: corrAcc,
      improvement: corrAcc - rawAcc,
      totalFixes: d.totalFixes,
      totalRegressions: d.totalRegressions,
      netFixes: d.totalFixes - d.totalRegressions,

      // Per-column breakdown
      byColumn: Object.fromEntries(
        Object.entries(d.byColumn).map(([col, data]) => [col, {
          total: data.total,
          rawAccuracy: data.total > 0 ? data.rawCorrect / data.total : 0,
          correctedAccuracy: data.total > 0 ? data.correctedCorrect / data.total : 0,
          improvement: data.total > 0 ? (data.correctedCorrect - data.rawCorrect) / data.total : 0,
          fixes: data.fixes,
          regressions: data.regressions,
        }])
      ),

      // Improvement over time (last 10 events)
      recentTrend: this._recentTrend(10),
    };
  }

  /**
   * Generate a human-readable report.
   */
  report() {
    const s = this.summary();
    const lines = [
      `Shifu OCR — Accuracy Report`,
      `${'═'.repeat(50)}`,
      ``,
      `Words evaluated:  ${s.totalWords}`,
      ``,
      `  Raw OCR accuracy:       ${(s.rawAccuracy * 100).toFixed(1)}%`,
      `  After correction:       ${(s.correctedAccuracy * 100).toFixed(1)}%`,
      `  Improvement:            ${s.improvement >= 0 ? '+' : ''}${(s.improvement * 100).toFixed(1)}%`,
      ``,
      `  Words fixed:            ${s.totalFixes}`,
      `  Words regressed:        ${s.totalRegressions}`,
      `  Net fixes:              ${s.netFixes}`,
    ];

    if (Object.keys(s.byColumn).length > 0) {
      lines.push(``, `Per-column:`);
      for (const [col, data] of Object.entries(s.byColumn)) {
        lines.push(
          `  ${col.padEnd(15)} raw: ${(data.rawAccuracy * 100).toFixed(0)}% → corrected: ${(data.correctedAccuracy * 100).toFixed(0)}% (${data.improvement >= 0 ? '+' : ''}${(data.improvement * 100).toFixed(0)}%, ${data.fixes} fixes, ${data.regressions} regressions)`
        );
      }
    }

    if (s.recentTrend.length >= 2) {
      const first = s.recentTrend[0].correctedAccuracy;
      const last = s.recentTrend[s.recentTrend.length - 1].correctedAccuracy;
      const delta = last - first;
      lines.push(``, `Trend (last ${s.recentTrend.length} events): ${delta >= 0 ? '+' : ''}${(delta * 100).toFixed(1)}%`);
    }

    return lines.join('\n');
  }

  /**
   * Reset all metrics.
   */
  reset() {
    this.data = this._fresh();
    this._save();
  }

  // ─── Internal ─────────────────────────────────────────────────

  /**
   * Count matching tokens between two word arrays using LCS.
   * Handles insertions/deletions correctly.
   */
  _lcsCount(wordsA, wordsB) {
    const a = wordsA.map(w => w.toLowerCase());
    const b = wordsB.map(w => w.toLowerCase());
    const m = a.length, n = b.length;
    const dp = Array.from({ length: m + 1 }, () => new Array(n + 1).fill(0));
    for (let i = 1; i <= m; i++) {
      for (let j = 1; j <= n; j++) {
        dp[i][j] = a[i - 1] === b[j - 1]
          ? dp[i - 1][j - 1] + 1
          : Math.max(dp[i - 1][j], dp[i][j - 1]);
      }
    }
    return dp[m][n];
  }

  /**
   * Count per-token fixes and regressions by building LCS match sets.
   * A "fix" = token matches truth after correction but not before.
   * A "regression" = token matched truth before but not after correction.
   */
  _countFlips(ocrWords, corrWords, truthWords) {
    const rawMatched = this._lcsMatchedSet(ocrWords, truthWords);
    const corrMatched = this._lcsMatchedSet(corrWords, truthWords);
    let fixes = 0, regressions = 0;
    for (let j = 0; j < truthWords.length; j++) {
      const inRaw = rawMatched.has(j);
      const inCorr = corrMatched.has(j);
      if (!inRaw && inCorr) fixes++;
      if (inRaw && !inCorr) regressions++;
    }
    return { fixes, regressions };
  }

  /**
   * Returns the set of truth-indices that are matched by wordsA via LCS backtrack.
   */
  _lcsMatchedSet(wordsA, truthWords) {
    const a = wordsA.map(w => w.toLowerCase());
    const b = truthWords.map(w => w.toLowerCase());
    const m = a.length, n = b.length;
    const dp = Array.from({ length: m + 1 }, () => new Array(n + 1).fill(0));
    for (let i = 1; i <= m; i++) {
      for (let j = 1; j <= n; j++) {
        dp[i][j] = a[i - 1] === b[j - 1]
          ? dp[i - 1][j - 1] + 1
          : Math.max(dp[i - 1][j], dp[i][j - 1]);
      }
    }
    // Backtrack to find which truth indices were matched
    const matched = new Set();
    let i = m, j = n;
    while (i > 0 && j > 0) {
      if (a[i - 1] === b[j - 1]) {
        matched.add(j - 1);
        i--; j--;
      } else if (dp[i - 1][j] >= dp[i][j - 1]) {
        i--;
      } else {
        j--;
      }
    }
    return matched;
  }

  _recentTrend(n) {
    return this.data.events.slice(-n).map(e => ({
      timestamp: e.timestamp,
      rawAccuracy: e.rawAccuracy,
      correctedAccuracy: e.correctedAccuracy,
      improvement: e.improvement,
    }));
  }

  _load() {
    const filePath = path.join(this.stateDir, 'metrics.json');
    if (!fs.existsSync(this.stateDir)) fs.mkdirSync(this.stateDir, { recursive: true });
    if (fs.existsSync(filePath)) {
      try { return { ...this._fresh(), ...JSON.parse(fs.readFileSync(filePath, 'utf-8')) }; }
      catch { return this._fresh(); }
    }
    return this._fresh();
  }

  _save() {
    const filePath = path.join(this.stateDir, 'metrics.json');
    if (!fs.existsSync(this.stateDir)) fs.mkdirSync(this.stateDir, { recursive: true });
    fs.writeFileSync(filePath, JSON.stringify(this.data, null, 2));
  }

  _fresh() {
    return {
      totalEvents: 0,
      totalWords: 0,
      totalRawCorrect: 0,
      totalCorrectedCorrect: 0,
      totalFixes: 0,
      totalRegressions: 0,
      byColumn: {},
      events: [],
    };
  }
}

module.exports = { MetricsTracker };
