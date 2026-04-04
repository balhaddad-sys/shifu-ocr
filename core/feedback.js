// Shifu Feedback Loop
// The system's self-awareness layer.
//
// predict() → evaluate() → learn() → reshape landscapes
//
// This closes the loop that was missing: the system now knows when it's wrong,
// tracks what it confuses, and reshapes itself from every correction.
//
// Three feedback channels:
//   1. Character-level: Python OCR landscapes reshape from corrections
//   2. Word-level: JS confusion model adapts substitution costs
//   3. Line-level: Clinical corrector learns ward-specific patterns
//
// All three feed the metrics system so improvement is measurable.

const path = require('path');
const fs = require('fs');

const METRICS_FILE = 'feedback_metrics.json';

class FeedbackLoop {
  constructor(shifu, options = {}) {
    this.shifu = shifu;
    this.stateDir = options.stateDir || path.join(__dirname, '..', '.state');
    this.metrics = this._loadMetrics();
  }

  /**
   * Process text and return a correction proposal.
   * Does NOT apply — the caller (nurse/doctor) must confirm or correct.
   */
  propose(ocrText, options = {}) {
    const corrected = this.shifu.correctLine(ocrText, options);
    const coherence = this.shifu.scoreSentence(corrected.output);
    const decision = this.shifu.assessConfidence(corrected);

    const proposal = {
      id: this._generateId(),
      timestamp: new Date().toISOString(),
      input: ocrText,
      proposed: corrected.output,
      words: corrected.words.map(w => ({
        original: w.original,
        proposed: w.corrected,
        confidence: w.confidence,
        flag: w.flag,
        candidates: (w.candidates || []).slice(0, 3),
      })),
      safetyFlags: corrected.safetyFlags,
      coherence: coherence.coherence,
      decision,
    };

    // Track the proposal for later evaluation and persist
    this.metrics.totalProposals++;
    this._saveMetrics();
    return proposal;
  }

  /**
   * Propose corrections for a table row (ward census).
   */
  proposeRow(ocrRow, options = {}) {
    const corrected = this.shifu.correctTableRow(ocrRow, options);

    const proposal = {
      id: this._generateId(),
      timestamp: new Date().toISOString(),
      columns: {},
      safetyFlags: corrected.safetyFlags,
      decision: null,
    };

    const allWords = [];
    for (const [col, data] of Object.entries(corrected.corrected)) {
      const words = (data.words || []).map(w => ({
        original: w.original,
        proposed: w.corrected,
        confidence: w.confidence,
        flag: w.flag,
      }));
      proposal.columns[col] = {
        input: data.input,
        proposed: data.output,
        words,
      };
      allWords.push(...words);
    }

    const hasUncertain = allWords.some(w => w.flag === 'low_confidence' || w.flag === 'unknown' || w.flag === 'corrected_verify');
    proposal.decision = corrected.hasDangers ? 'reject'
      : (corrected.hasWarnings || hasUncertain) ? 'verify' : 'accept';

    this.metrics.totalProposals++;
    this._saveMetrics();
    return proposal;
  }

  /**
   * Evaluate a proposal against ground truth.
   * This is THE critical method — it closes the feedback loop.
   *
   * @param {object} proposal - The proposal from propose() or proposeRow()
   * @param {string|object} confirmed - The confirmed text (string for lines, object for rows)
   * @returns {object} Evaluation result with per-word accuracy
   */
  evaluate(proposal, confirmed) {
    const evaluation = {
      proposalId: proposal.id,
      timestamp: new Date().toISOString(),
      words: [],
      correct: 0,
      total: 0,
      accuracy: 0,
      corrections: [],
      safetyHits: [],
    };

    if (typeof confirmed === 'string') {
      // Line-level evaluation
      const confirmedWords = confirmed.split(/\s+/).filter(w => w.length > 0);
      const proposedWords = proposal.words || [];

      this._alignAndScore(proposedWords, confirmedWords, evaluation, {});

      // Now learn from this evaluation
      const ocrRow = { text: proposal.input };
      const confirmedRow = { text: confirmed };
      this.shifu.learn(ocrRow, confirmedRow);

    } else if (typeof confirmed === 'object') {
      // Row-level evaluation
      for (const [col, confirmedText] of Object.entries(confirmed)) {
        const proposalCol = (proposal.columns || {})[col];
        if (!proposalCol) continue;

        // Skip null/undefined confirmed (no data for this column)
        if (confirmedText == null) continue;
        const confirmedWords = confirmedText.split(/\s+/).filter(w => w.length > 0);
        const proposedWords = proposalCol.words || [];

        if (confirmedWords.length === 0 && proposedWords.length > 0) {
          // Confirmed blank: every proposed word is extra
          for (const pw of proposedWords) {
            evaluation.total++;
            evaluation.words.push({
              column: col, original: pw.original, proposed: pw.proposed,
              confirmed: '', correct: false, flag: 'deleted',
            });
            evaluation.corrections.push({
              column: col, position: 0, original: pw.original,
              proposed: pw.proposed, confirmed: '', flag: 'deleted',
            });
          }
        } else {
          this._alignAndScore(proposedWords, confirmedWords, evaluation, { column: col });
        }
      }

      // Reconstruct OCR and confirmed rows for learning
      // Preserve blank confirmed cells as-is (don't fall back to OCR input)
      const ocrRow = {};
      const confirmedRow = {};
      for (const [col, data] of Object.entries(proposal.columns || {})) {
        ocrRow[col] = data.input;
        confirmedRow[col] = col in confirmed ? confirmed[col] : data.input;
      }
      this.shifu.learn(ocrRow, confirmedRow);
    }

    // Safety evaluation: did we catch real issues?
    for (const flag of (proposal.safetyFlags || [])) {
      evaluation.safetyHits.push({ flag: flag.status, severity: flag.severity, message: flag.message });
    }

    evaluation.accuracy = evaluation.total > 0 ? evaluation.correct / evaluation.total : 0;

    // Update metrics
    this._updateMetrics(evaluation);
    this._saveMetrics();

    return evaluation;
  }

  /**
   * Get the current performance metrics.
   */
  getMetrics() {
    const m = this.metrics;
    return {
      totalProposals: m.totalProposals,
      totalEvaluations: m.totalEvaluations,
      totalWords: m.totalWords,
      totalCorrect: m.totalCorrect,
      accuracy: m.totalWords > 0 ? m.totalCorrect / m.totalWords : 0,

      // Per-flag breakdown: how accurate is each correction type?
      byFlag: Object.fromEntries(
        Object.entries(m.byFlag).map(([flag, data]) => [flag, {
          total: data.total,
          correct: data.correct,
          accuracy: data.total > 0 ? data.correct / data.total : 0,
        }])
      ),

      // Improvement over time (windows of 50 evaluations)
      windows: m.windows,

      // What the system gets wrong most
      topErrors: this._topErrors(10),

      // Clinical safety stats
      safety: {
        totalFlags: m.safetyFlagsRaised,
        byType: m.safetyByType,
      },

      sessionStart: m.sessionStart,
      lastEvaluation: m.lastEvaluation,
    };
  }

  /**
   * Get a simple accuracy report string.
   */
  report() {
    const m = this.getMetrics();
    const lines = [
      `Shifu OCR — Performance Report`,
      `${'─'.repeat(50)}`,
      `Proposals:    ${m.totalProposals}`,
      `Evaluations:  ${m.totalEvaluations}`,
      `Words scored: ${m.totalWords}`,
      `Accuracy:     ${(m.accuracy * 100).toFixed(1)}%`,
      ``,
    ];

    if (Object.keys(m.byFlag).length > 0) {
      lines.push(`Per-flag accuracy:`);
      for (const [flag, data] of Object.entries(m.byFlag)) {
        lines.push(`  ${flag.padEnd(25)} ${(data.accuracy * 100).toFixed(1)}% (${data.correct}/${data.total})`);
      }
      lines.push(``);
    }

    if (m.topErrors.length > 0) {
      lines.push(`Top errors:`);
      for (const err of m.topErrors.slice(0, 5)) {
        lines.push(`  ${err.pattern} [${err.flag}] ${err.outcome} (${err.count}x)`);
      }
      lines.push(``);
    }

    if (m.windows.length >= 2) {
      const first = m.windows[0];
      const last = m.windows[m.windows.length - 1];
      const delta = last.accuracy - first.accuracy;
      lines.push(`Improvement: ${delta > 0 ? '+' : ''}${(delta * 100).toFixed(1)}% (window 1 → ${m.windows.length})`);
    }

    return lines.join('\n');
  }

  // ─── Internal ─────────────────────────────────────────────────

  /**
   * Align proposed and confirmed tokens using LCS, then score.
   * Handles insertions and deletions correctly instead of positional comparison.
   */
  _alignAndScore(proposedWords, confirmedWords, evaluation, extra) {
    const proposedTexts = proposedWords.map(w => w.proposed.toLowerCase());
    const confirmedTexts = confirmedWords.map(w => w.toLowerCase());

    // Needleman-Wunsch alignment (edit-distance with backtrace)
    // Unlike LCS, this produces substitution pairs for mismatched tokens
    const m = proposedTexts.length;
    const n = confirmedTexts.length;
    const dp = Array.from({ length: m + 1 }, () => new Array(n + 1).fill(0));
    for (let i = 0; i <= m; i++) dp[i][0] = i;
    for (let j = 0; j <= n; j++) dp[0][j] = j;
    for (let i = 1; i <= m; i++) {
      for (let j = 1; j <= n; j++) {
        const cost = proposedTexts[i - 1] === confirmedTexts[j - 1] ? 0 : 1;
        dp[i][j] = Math.min(
          dp[i - 1][j] + 1,      // deletion (extra proposed)
          dp[i][j - 1] + 1,      // insertion (missing from proposal)
          dp[i - 1][j - 1] + cost // match or substitution
        );
      }
    }

    // Backtrack to build alignment
    const aligned = [];
    let i = m, j = n;
    while (i > 0 && j > 0) {
      const cost = proposedTexts[i - 1] === confirmedTexts[j - 1] ? 0 : 1;
      if (dp[i][j] === dp[i - 1][j - 1] + cost) {
        aligned.push({ type: cost === 0 ? 'match' : 'substitution', pi: i - 1, ci: j - 1 });
        i--; j--;
      } else if (dp[i][j] === dp[i - 1][j] + 1) {
        aligned.push({ type: 'deletion', pi: i - 1, ci: -1 });
        i--;
      } else {
        aligned.push({ type: 'insertion', pi: -1, ci: j - 1 });
        j--;
      }
    }
    while (i > 0) { aligned.push({ type: 'deletion', pi: --i, ci: -1 }); }
    while (j > 0) { aligned.push({ type: 'insertion', pi: -1, ci: --j }); }
    aligned.reverse();

    // Score from alignment
    for (const a of aligned) {
      evaluation.total++;
      if (a.type === 'match') {
        const pw = proposedWords[a.pi];
        const cw = confirmedWords[a.ci];
        evaluation.correct++;
        evaluation.words.push({
          ...extra, original: pw.original, proposed: pw.proposed,
          confirmed: cw, correct: true, flag: pw.flag,
        });
      } else if (a.type === 'substitution') {
        // Wrong token — proposed doesn't match confirmed
        const pw = proposedWords[a.pi];
        const cw = confirmedWords[a.ci];
        evaluation.words.push({
          ...extra, original: pw.original, proposed: pw.proposed,
          confirmed: cw, correct: false, flag: pw.flag,
        });
        evaluation.corrections.push({
          ...extra, position: a.pi, original: pw.original,
          proposed: pw.proposed, confirmed: cw, flag: pw.flag,
        });
      } else if (a.type === 'deletion') {
        const pw = proposedWords[a.pi];
        evaluation.words.push({
          ...extra, original: pw.original, proposed: pw.proposed,
          confirmed: '(extra)', correct: false, flag: pw.flag,
        });
        evaluation.corrections.push({
          ...extra, position: a.pi, original: pw.original,
          proposed: pw.proposed, confirmed: '(extra)', flag: 'extra_token',
        });
      } else if (a.type === 'insertion') {
        const cw = confirmedWords[a.ci];
        evaluation.words.push({
          ...extra, original: '', proposed: '(missing)',
          confirmed: cw, correct: false, flag: 'missing_token',
        });
        evaluation.corrections.push({
          ...extra, position: a.ci, original: '', proposed: '(missing)',
          confirmed: cw, flag: 'missing_token',
        });
      }
    }
  }

  _updateMetrics(evaluation) {
    const m = this.metrics;
    m.totalEvaluations++;
    m.totalWords += evaluation.total;
    m.totalCorrect += evaluation.correct;
    m.lastEvaluation = evaluation.timestamp;

    // Per-flag breakdown
    for (const w of evaluation.words) {
      const flag = w.flag || 'unknown';
      if (!m.byFlag[flag]) m.byFlag[flag] = { total: 0, correct: 0 };
      m.byFlag[flag].total++;
      if (w.correct) m.byFlag[flag].correct++;
    }

    // Error tracking — anonymized keys to avoid persisting PHI
    for (const c of evaluation.corrections) {
      const key = `len${(c.original||'').length}|${c.flag || 'unknown'}|${c.correct ? 'hit' : 'miss'}`;
      m.errors[key] = (m.errors[key] || 0) + 1;
    }

    // Safety stats
    for (const s of evaluation.safetyHits) {
      m.safetyFlagsRaised++;
      m.safetyByType[s.flag] = (m.safetyByType[s.flag] || 0) + 1;
    }

    // Rolling accuracy windows (every 50 evaluations)
    m.windowBuffer.push(evaluation.accuracy);
    if (m.windowBuffer.length >= 50) {
      const windowAcc = m.windowBuffer.reduce((a, b) => a + b, 0) / m.windowBuffer.length;
      m.windows.push({
        windowIndex: m.windows.length,
        evaluations: m.windowBuffer.length,
        accuracy: windowAcc,
        timestamp: new Date().toISOString(),
      });
      m.windowBuffer = [];
    }
  }

  _topErrors(n = 10) {
    return Object.entries(this.metrics.errors)
      .map(([key, count]) => {
        const parts = key.split('|');
        return { pattern: parts[0], flag: parts[1], outcome: parts[2], count };
      })
      .sort((a, b) => b.count - a.count)
      .slice(0, n);
  }

  _loadMetrics() {
    const filePath = path.join(this.stateDir, METRICS_FILE);
    if (!fs.existsSync(this.stateDir)) fs.mkdirSync(this.stateDir, { recursive: true });
    if (fs.existsSync(filePath)) {
      try {
        const data = JSON.parse(fs.readFileSync(filePath, 'utf-8'));
        // Ensure all fields exist
        return { ...this._freshMetrics(), ...data };
      } catch { return this._freshMetrics(); }
    }
    return this._freshMetrics();
  }

  _saveMetrics() {
    const filePath = path.join(this.stateDir, METRICS_FILE);
    if (!fs.existsSync(this.stateDir)) fs.mkdirSync(this.stateDir, { recursive: true });
    fs.writeFileSync(filePath, JSON.stringify(this.metrics, null, 2));
  }

  _freshMetrics() {
    return {
      totalProposals: 0,
      totalEvaluations: 0,
      totalWords: 0,
      totalCorrect: 0,
      byFlag: {},
      errors: {},
      safetyFlagsRaised: 0,
      safetyByType: {},
      windows: [],
      windowBuffer: [],
      sessionStart: new Date().toISOString(),
      lastEvaluation: null,
    };
  }

  _generateId() {
    return `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
  }
}

module.exports = { FeedbackLoop };
