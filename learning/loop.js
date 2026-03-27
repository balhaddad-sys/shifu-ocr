// Shifu Learning Loop
// Three fluid landscapes that reshape with every nurse correction.
// Now integrated with ShifuEngine's resonance system.

const {
  VOCABULARY,
  createCandidateIndex,
  addWordToCandidateIndex,
  getIndexedCandidates,
} = require('../clinical/vocabulary');

function learningIncrement(weight = 1) {
  return Math.max(1, Math.round(Number(weight) || 1));
}

// =========================================================================
// 1. ADAPTIVE CONFUSION PROFILE
// =========================================================================

class AdaptiveConfusionProfile {
  constructor() {
    this.confusionCounts = {};
    this.totalCorrections = 0;
    // All keys are sorted so lookup always matches: getCost sorts the pair too
    this.baseCosts = {};
    const rawCosts = {
      'O,0': 0.1, 'o,0': 0.1, 'l,1': 0.2, 'I,1': 0.2, 'I,l': 0.1,
      '5,S': 0.3, '5,s': 0.3, '8,B': 0.3, '6,G': 0.4, '2,Z': 0.4,
      'r,n': 0.3, 'm,n': 0.4, 'u,v': 0.5, 'c,e': 0.5, 'h,b': 0.5,
      'D,O': 0.3, 'f,t': 0.4, 'E,F': 0.4, 'a,e': 0.4, 'a,o': 0.4,
      'd,o': 0.3, 'e,i': 0.4, 'u,n': 0.4, 'b,d': 0.4,
      // FLAIR engine systematic confusions (lowercase for weightedDistance)
      'i,t': 0.1, 'i,l': 0.1, '6,g': 0.1, '1,j': 0.2,
      'c,x': 0.2, 'g,q': 0.3, 'g,9': 0.2, '3,s': 0.2, '9,g': 0.2,
      // Additional FLAIR confusions from real screenshot analysis
      'x,s': 0.3, 'x,k': 0.3, 'x,t': 0.3,   // X over-predicted
      'u,o': 0.3, 'u,a': 0.3,                  // U over-predicted
      'g,d': 0.3, 'g,a': 0.3, 'g,o': 0.3,     // g over-predicted
      'n,r': 0.3, 'n,h': 0.3,                  // n low-confidence
      // MRI-RF model v2 confusions (no-punctuation, from real ward census)
      'q,b': 0.2, 'q,d': 0.2, 'q,o': 0.2, 'q,g': 0.2,  // Q over-predicted for round chars
      '4,a': 0.1, '4,e': 0.2, '4,o': 0.2,                // 4↔a/e/o (4D MRI systematic)
      '3,b': 0.2, '0,n': 0.2, '0,o': 0.1,                // 3↔B, 0↔N/O
      '6,e': 0.1, '6,b': 0.3,                             // 6↔e/b
      'z,2': 0.2, 'z,s': 0.3,                             // z↔2/s
      '9,g': 0.1, 'g,9': 0.1,                             // 9↔g loop+tail
      '2,z': 0.2, 'z,2': 0.2,                             // 2↔z angular
      'j,i': 0.2, 'j,l': 0.2, 'j,1': 0.2,                // j↔thin verticals
      'w,m': 0.3, 'm,w': 0.3,                             // w↔m wave patterns
      'v,u': 0.2,                                          // v↔u open curves
    };
    for (const [k, v] of Object.entries(rawCosts)) {
      const sorted = k.split(',').sort().join(',');
      // Keep the lowest cost if duplicates after sorting
      if (!(sorted in this.baseCosts) || v < this.baseCosts[sorted]) {
        this.baseCosts[sorted] = v;
      }
    }
  }

  /**
   * Record a correction event. Clinical weight makes the system learn
   * faster from critical corrections (medications, diagnoses).
   *
   * @param {string} ocrText - What OCR produced
   * @param {string} correctedText - What the human confirmed
   * @param {number} clinicalWeight - Learning rate multiplier (1.0 = normal, 3.0 = medication)
   */
  recordCorrection(ocrText, correctedText, clinicalWeight = 1.0) {
    const pairs = this._alignAndExtract(ocrText.toLowerCase(), correctedText.toLowerCase());
    // Clinical weight: medication corrections count 3x, diagnosis 2.5x, etc.
    const increment = Math.max(1, Math.round(clinicalWeight));
    for (const [ocrChar, correctChar] of pairs) {
      if (ocrChar !== correctChar) {
        const key = [ocrChar, correctChar].sort().join(',');
        this.confusionCounts[key] = (this.confusionCounts[key] || 0) + increment;
        // Track per-character accuracy for real confidence
        const errKey = `err:${key}`;
        this.confusionCounts[errKey] = (this.confusionCounts[errKey] || 0) + increment;
      } else {
        // Track correct matches too — needed for real confidence
        const okKey = `ok:${ocrChar}`;
        this.confusionCounts[okKey] = (this.confusionCounts[okKey] || 0) + increment;
      }
    }
    this.totalCorrections += increment;
  }

  /**
   * Dynamic confusion cost — derived from actual correction data.
   * Replaces static hardcoded costs as the system gains experience.
   *
   * The formula: start with base cost, then blend toward empirical cost
   * as more data arrives. Empirical cost = 1 - P(confusion), floored at 0.05
   * so known confusions always get cheap substitution.
   */
  getCost(char1, char2) {
    if (char1 === char2) return 0.0;
    // Case is free — same letter, same FLAIR response
    if (char1.toLowerCase() === char2.toLowerCase()) return 0.0;
    const key = [char1, char2].sort().join(',');
    const baseCost = this.baseCosts[key] ?? 1.0;
    const confusionCount = this.confusionCounts[key] || 0;

    if (confusionCount === 0 || this.totalCorrections < 5) return baseCost;

    // P(confusion) = how often this pair was confused / total corrections
    const pConfusion = confusionCount / this.totalCorrections;
    // Higher confusion probability → lower cost (easier substitution)
    const empiricalCost = Math.max(0.05, 1.0 - pConfusion * 5);

    // Blend: as experience grows, trust empirical data more
    const experience = Math.min(this.totalCorrections / 50, 1.0);
    return baseCost * (1 - experience) + empiricalCost * experience;
  }

  /**
   * Real confidence for a character based on observed accuracy.
   * Unlike hardcoded thresholds, this is grounded in actual data.
   */
  getCharConfidence(char) {
    const okKey = `ok:${char.toLowerCase()}`;
    const correct = this.confusionCounts[okKey] || 0;
    // Count all errors involving this character
    let errors = 0;
    for (const [key, count] of Object.entries(this.confusionCounts)) {
      if (key.startsWith('err:') && key.includes(char.toLowerCase())) {
        errors += count;
      }
    }
    const total = correct + errors;
    if (total === 0) return 0.5; // No data
    return correct / total;
  }

  weightedDistance(str1, str2) {
    const s1 = str1.toLowerCase(), s2 = str2.toLowerCase();
    if (s1.length === 0) return s2.length;
    if (s2.length === 0) return s1.length;
    const m = [];
    for (let i = 0; i <= s1.length; i++) m[i] = [i];
    for (let j = 0; j <= s2.length; j++) m[0][j] = j;
    for (let i = 1; i <= s1.length; i++) {
      for (let j = 1; j <= s2.length; j++) {
        const sub = this.getCost(s1[i - 1], s2[j - 1]);
        m[i][j] = Math.min(m[i-1][j] + 1, m[i][j-1] + 1, m[i-1][j-1] + sub);
      }
    }
    return m[s1.length][s2.length];
  }

  getTopConfusions(n = 10) {
    return Object.entries(this.confusionCounts)
      .filter(([key]) => !key.startsWith('ok:') && !key.startsWith('err:'))
      .sort((a, b) => b[1] - a[1]).slice(0, n)
      .map(([pair, count]) => ({ pair, count, cost: this.getCost(...pair.split(',')) }));
  }

  _alignAndExtract(s1, s2) {
    // Edit-distance backtrace alignment — produces correct substitution pairs
    // even when strings differ in length (insertions/deletions)
    const n = s1.length, m = s2.length;
    const dp = Array.from({ length: n + 1 }, () => new Array(m + 1).fill(0));
    for (let i = 0; i <= n; i++) dp[i][0] = i;
    for (let j = 0; j <= m; j++) dp[0][j] = j;
    for (let i = 1; i <= n; i++) {
      for (let j = 1; j <= m; j++) {
        const cost = s1[i - 1] === s2[j - 1] ? 0 : 1;
        dp[i][j] = Math.min(dp[i-1][j] + 1, dp[i][j-1] + 1, dp[i-1][j-1] + cost);
      }
    }
    // Backtrace: only emit substitution pairs (not insertions/deletions)
    const pairs = [];
    let i = n, j = m;
    while (i > 0 && j > 0) {
      const cost = s1[i - 1] === s2[j - 1] ? 0 : 1;
      if (dp[i][j] === dp[i-1][j-1] + cost) {
        // Substitution or match — this is a real character correspondence
        pairs.push([s1[i - 1], s2[j - 1]]);
        i--; j--;
      } else if (dp[i][j] === dp[i-1][j] + 1) {
        i--; // Deletion in s1 — skip, no pair
      } else {
        j--; // Insertion in s2 — skip, no pair
      }
    }
    return pairs.reverse();
  }

  toJSON() {
    return { confusionCounts: this.confusionCounts, totalCorrections: this.totalCorrections };
  }

  static fromJSON(data) {
    const p = new AdaptiveConfusionProfile();
    p.confusionCounts = data.confusionCounts || {};
    p.totalCorrections = data.totalCorrections || 0;
    return p;
  }
}

// =========================================================================
// 2. WARD VOCABULARY LANDSCAPE
// =========================================================================

class WardVocabulary {
  constructor(baseVocabulary = {}) {
    this.baseWords = new Map();
    this.frequency = {};
    this.categoryAssoc = {};
    this.totalReadings = 0;
    this._baseVocabulary = baseVocabulary;
    this._allWords = new Set();
    this._candidateIndex = createCandidateIndex([]);
    this.reloadBaseVocabulary(baseVocabulary);
  }

  _loadBaseWords(baseVocabulary = {}) {
    this.baseWords = new Map();
    for (const [category, words] of Object.entries(baseVocabulary)) {
      for (const w of words) {
        const lower = w.toLowerCase();
        this.baseWords.set(lower, category);
        for (const part of lower.split(/\s+/)) {
          if (part.length > 1 && !this.baseWords.has(part)) this.baseWords.set(part, category);
        }
      }
    }
  }

  _rebuildIndexes() {
    this._allWords = new Set(this.baseWords.keys());
    for (const word of Object.keys(this.frequency)) {
      this._allWords.add(word.toLowerCase());
    }
    this._candidateIndex = createCandidateIndex(this._allWords);
  }

  reloadBaseVocabulary(baseVocabulary = this._baseVocabulary) {
    this._baseVocabulary = baseVocabulary;
    this._loadBaseWords(baseVocabulary);
    this._rebuildIndexes();
  }

  confirmWord(word, category = null, weight = 1) {
    const lower = word.toLowerCase().trim();
    if (!lower) return;
    const increment = learningIncrement(weight);
    this.frequency[lower] = (this.frequency[lower] || 0) + increment;
    this.totalReadings += increment;
    if (!this._allWords.has(lower)) {
      this._allWords.add(lower);
      addWordToCandidateIndex(this._candidateIndex, lower);
    }
    if (category) {
      if (!this.categoryAssoc[lower]) this.categoryAssoc[lower] = {};
      this.categoryAssoc[lower][category] = (this.categoryAssoc[lower][category] || 0) + increment;
    }
  }

  confirmReading(fields, weightByColumn = {}) {
    for (const [col, text] of Object.entries(fields)) {
      if (!text || typeof text !== 'string') continue;
      const weight = weightByColumn[col.toLowerCase()] || weightByColumn[col] || 1;
      for (const word of text.split(/\s+/)) {
        if (word.length > 1) this.confirmWord(word, col.toLowerCase(), weight);
      }
    }
  }

  getFrequencyBoost(word) {
    const count = this.frequency[word.toLowerCase()] || 0;
    if (count === 0) return 0;
    return Math.log(1 + count) / Math.log(1 + 100);
  }

  isKnown(word) {
    const lower = word.toLowerCase();
    return this.baseWords.has(lower) || (this.frequency[lower] || 0) > 0;
  }

  getColumnWords(columnType) {
    const type = columnType.toLowerCase();
    const words = new Map();
    for (const [word, category] of this.baseWords) {
      if (this._categoryMatches(category, type)) {
        words.set(word, { base: true, frequency: this.frequency[word] || 0, boost: this.getFrequencyBoost(word) });
      }
    }
    for (const [word, categories] of Object.entries(this.categoryAssoc)) {
      for (const [cat] of Object.entries(categories)) {
        if (this._categoryMatches(cat, type)) {
          words.set(word, { base: this.baseWords.has(word), frequency: this.frequency[word] || 0, boost: this.getFrequencyBoost(word) });
        }
      }
    }
    return words;
  }

  getAllWords() {
    return new Set(this._allWords);
  }

  getCandidateWords(word, options = {}) {
    return getIndexedCandidates(word, this._candidateIndex, this._allWords, options);
  }

  getLearnedWords(minCount = 2) {
    const learned = [];
    for (const [word, count] of Object.entries(this.frequency)) {
      if (!this.baseWords.has(word) && count >= minCount) {
        learned.push({ word, count, categories: this.categoryAssoc[word] || {} });
      }
    }
    return learned.sort((a, b) => b.count - a.count);
  }

  _categoryMatches(category, columnType) {
    if (!category || !columnType) return true;
    const cat = category.toLowerCase(), col = columnType.toLowerCase();
    if (col.includes('patient') || col.includes('name')) return cat === 'names' || cat === 'patient';
    if (col.includes('diagnos')) return cat === 'diagnoses' || cat === 'diagnosis';
    if (col.includes('doctor')) return cat === 'names' || cat === 'doctor';
    if (col.includes('med') || col.includes('drug')) return cat === 'medications';
    return true;
  }

  toJSON() {
    return { frequency: this.frequency, categoryAssoc: this.categoryAssoc, totalReadings: this.totalReadings };
  }

  static fromJSON(data, baseVocabulary) {
    const v = new WardVocabulary(baseVocabulary);
    v.frequency = data.frequency || {};
    v.categoryAssoc = data.categoryAssoc || {};
    v.totalReadings = data.totalReadings || 0;
    v._rebuildIndexes();
    return v;
  }
}

// =========================================================================
// 3. CONTEXT CHAINS
// =========================================================================

class ContextChains {
  constructor() {
    this.chains = {};
    this.totalEvents = 0;
  }

  learn(fields, weightByColumn = {}) {
    const entries = Object.entries(fields)
      .filter(([, v]) => v && typeof v === 'string' && v.trim())
      .map(([k, v]) => [k.toLowerCase(), v.toLowerCase().trim()]);
    for (let i = 0; i < entries.length; i++) {
      for (let j = 0; j < entries.length; j++) {
        if (i === j) continue;
        const [ctxCol, ctxVal] = entries[i];
        const [tgtCol, tgtVal] = entries[j];
        const increment = learningIncrement(weightByColumn[tgtCol] || weightByColumn[tgtCol.toLowerCase()] || 1);
        // Store the full phrase for predict()
        const key = `${ctxCol}:${ctxVal}|${tgtCol}`;
        if (!this.chains[key]) this.chains[key] = {};
        this.chains[key][tgtVal] = (this.chains[key][tgtVal] || 0) + increment;
        // Also store individual words so getBoost() works on single candidates
        for (const word of tgtVal.split(/\s+/)) {
          if (word.length > 1) {
            this.chains[key][word] = (this.chains[key][word] || 0) + increment;
          }
        }
      }
    }
    this.totalEvents++;
  }

  getBoost(knownFields, targetColumn, candidateWord) {
    let totalBoost = 0;
    const candidate = candidateWord.toLowerCase();
    const target = targetColumn.toLowerCase();
    for (const [col, val] of Object.entries(knownFields)) {
      if (!val) continue;
      const key = `${col.toLowerCase()}:${val.toLowerCase()}|${target}`;
      const chain = this.chains[key];
      if (chain && chain[candidate]) {
        totalBoost += Math.log(1 + chain[candidate]) * 0.3;
      }
    }
    return totalBoost;
  }

  predict(knownFields, targetColumn, topK = 5) {
    const target = targetColumn.toLowerCase();
    const scores = {};
    for (const [col, val] of Object.entries(knownFields)) {
      if (!val) continue;
      const key = `${col.toLowerCase()}:${val.toLowerCase()}|${target}`;
      const chain = this.chains[key];
      if (!chain) continue;
      for (const [word, count] of Object.entries(chain)) {
        scores[word] = (scores[word] || 0) + count;
      }
    }
    return Object.entries(scores).sort((a, b) => b[1] - a[1]).slice(0, topK)
      .map(([word, count]) => ({ word, count }));
  }

  toJSON() { return { chains: this.chains, totalEvents: this.totalEvents }; }

  static fromJSON(data) {
    const c = new ContextChains();
    c.chains = data.chains || {};
    c.totalEvents = data.totalEvents || 0;
    return c;
  }
}

// =========================================================================
// 4. INTEGRATED LEARNING ENGINE (wraps all three + feeds ShifuEngine)
// =========================================================================

class ShifuLearningEngine {
  constructor(baseVocabulary = {}) {
    this.confusion = new AdaptiveConfusionProfile();
    this.vocabulary = new WardVocabulary(baseVocabulary);
    this.context = new ContextChains();
    this.correctionCount = 0;
    this.sessionStart = new Date().toISOString();
    this._baseVocabulary = baseVocabulary;

    // Correction history for rollback (ring buffer, keeps last N)
    this._history = [];
    this._historyLimit = 100;

    // Write lock — prevents concurrent learn() calls from interleaving
    this._locked = false;
  }

  correctRow(ocrRow, knownContext = {}) {
    // Lazy-require to avoid circular dependency
    const { runSafetyChecks } = require('../clinical/safety');

    const corrected = {};
    const allFlags = [];
    for (const [column, rawText] of Object.entries(ocrRow)) {
      if (!rawText || typeof rawText !== 'string') {
        corrected[column] = { input: rawText, output: rawText, words: [], safetyFlags: [] };
        continue;
      }
      const words = rawText.split(/\s+/).filter(w => w.length > 0);
      const correctedWords = [];
      for (const raw of words) {
        const result = this._correctWord(raw, column, knownContext);
        correctedWords.push(result);
      }
      const output = correctedWords.map(w => w.corrected).join(' ');
      const safetyFlags = runSafetyChecks(correctedWords);
      corrected[column] = { input: rawText, output, words: correctedWords, safetyFlags };
      knownContext[column] = output;
      for (const flag of safetyFlags) {
        allFlags.push({ column, ...flag });
      }
    }
    return {
      corrected,
      safetyFlags: allFlags,
      hasDangers: allFlags.some(f => f.severity === 'danger'),
      hasWarnings: allFlags.some(f => f.severity === 'warning' || f.severity === 'error'),
    };
  }

  /**
   * Learn from a confirmed reading.
   * Also feeds the ShifuEngine core if provided — this is the bridge
   * that makes nurse corrections strengthen the resonance system.
   *
   * Confidence gating: suspicious corrections are rejected to prevent
   * nurse typos from poisoning the model.
   */
  learn(ocrRow, confirmedRow, coreEngine = null) {
    // Write lock — prevent concurrent learn() calls from interleaving state updates
    if (this._locked) {
      console.warn('ShifuLearningEngine: learn() called while locked, skipping');
      return { accepted: false, reason: 'locked' };
    }
    this._locked = true;

    try {
      return this._learnInternal(ocrRow, confirmedRow, coreEngine);
    } finally {
      this._locked = false;
    }
  }

  _learnInternal(ocrRow, confirmedRow, coreEngine) {
    const { getLearningRate, requiresSafetyReview } = require('./clinical_weights');

    // Confidence gate: reject corrections that look like nurse typos
    const rejected = [];
    for (const [column, confirmedText] of Object.entries(confirmedRow)) {
      if (!confirmedText || typeof confirmedText !== 'string') continue;
      const ocrText = ocrRow[column] || '';
      if (!ocrText || ocrText === confirmedText) continue;

      // If confirmed text is very different from OCR AND not a known vocabulary word,
      // it might be a nurse typo rather than a real correction
      const confirmedWords = confirmedText.split(/\s+/).filter(w => w.length > 0);
      for (const w of confirmedWords) {
        const lower = w.toLowerCase();
        if (lower.length <= 2) continue; // skip short words
        const isKnown = this.vocabulary.isKnown(lower);
        const ocrWords = ocrText.toLowerCase().split(/\s+/);
        const isInOcr = ocrWords.some(ow => ow === lower);
        if (!isKnown && !isInOcr) {
          // Unknown word not in OCR — could be a typo. Check edit distance.
          const minDist = ocrWords.reduce((min, ow) =>
            Math.min(min, this.confusion.weightedDistance(lower, ow)), Infinity);
          // If the confirmed word is very far from all OCR words, flag it
          if (minDist > Math.max(lower.length * 0.8, 4)) {
            rejected.push({ column, word: w, reason: 'too_distant', distance: minDist });
          }
        }
      }
    }

    // Build a clean row excluding columns with rejected words
    const rejectedColumns = new Set(rejected.map(r => r.column));
    const cleanConfirmedRow = {};
    for (const [col, text] of Object.entries(confirmedRow)) {
      if (!rejectedColumns.has(col)) {
        cleanConfirmedRow[col] = text;
      }
    }

    // If ALL columns were rejected, this learn is fully rejected — no mutations
    if (Object.keys(cleanConfirmedRow).length === 0) {
      return { accepted: false, rejected, reason: 'all_columns_rejected' };
    }

    // Snapshot state before learning (for rollback)
    const snapshot = {
      id: this.correctionCount,
      timestamp: new Date().toISOString(),
      ocrRow: { ...ocrRow },
      confirmedRow: { ...confirmedRow },
      confusionSnapshot: JSON.parse(JSON.stringify(this.confusion.toJSON())),
      vocabFreqSnapshot: { ...this.vocabulary.frequency },
      vocabCategorySnapshot: JSON.parse(JSON.stringify(this.vocabulary.categoryAssoc)),
      vocabTotalReadings: this.vocabulary.totalReadings,
      contextSnapshot: JSON.parse(JSON.stringify(this.context.toJSON())),
      // Snapshot core engine state for full rollback
      coreSnapshot: coreEngine && typeof coreEngine.serialize === 'function'
        ? coreEngine.serialize() : null,
      rejected,
    };

    this.correctionCount++;
    const confirmedSentences = [];
    const columnWeights = {};

    for (const [column, confirmedText] of Object.entries(cleanConfirmedRow)) {
      if (!confirmedText || typeof confirmedText !== 'string') continue;
      columnWeights[column.toLowerCase()] = learningIncrement(
        getLearningRate(column, confirmedText, VOCABULARY)
      );
    }

    for (const [column, confirmedText] of Object.entries(confirmedRow)) {
      if (!confirmedText || typeof confirmedText !== 'string') continue;
      const ocrText = ocrRow[column] || '';
      if (ocrText && ocrText !== confirmedText) {
        const isSafety = requiresSafetyReview(column, confirmedText, VOCABULARY);
        const hasRejectedWords = rejected.some(r => r.column === column);

        if (!hasRejectedWords || isSafety) {
          const weight = columnWeights[column.toLowerCase()] || learningIncrement(
            getLearningRate(column, confirmedText, VOCABULARY)
          );
          this._recordAlignedCorrection(ocrText, confirmedText, weight);
        }
      }
      if (!rejectedColumns.has(column)) {
        confirmedSentences.push({ column, text: confirmedText });
      }
    }

    this.context.learn(cleanConfirmedRow, columnWeights);
    this.vocabulary.confirmReading(cleanConfirmedRow, columnWeights);

    if (coreEngine && typeof coreEngine.feed === 'function') {
      for (const { column, text } of confirmedSentences) {
        const repeat = columnWeights[column.toLowerCase()] || 1;
        for (let i = 0; i < repeat; i++) {
          coreEngine.feed(text);
        }
      }
    }

    // Save to history (ring buffer)
    this._history.push(snapshot);
    if (this._history.length > this._historyLimit) {
      this._history.shift();
    }

    return { accepted: true, rejected, id: snapshot.id };
  }

  /**
   * Undo the most recent learn() call.
   * Restores confusion profile, vocabulary, context, and core engine to pre-learn state.
   *
   * @param {object} coreEngine - The ShifuEngine to restore (optional but recommended)
   * @returns {object|null} The undone correction or null if nothing to undo
   */
  undo(coreEngine = null) {
    if (this._history.length === 0) return null;
    const snapshot = this._history.pop();

    // Restore confusion profile
    this.confusion.confusionCounts = snapshot.confusionSnapshot.confusionCounts;
    this.confusion.totalCorrections = snapshot.confusionSnapshot.totalCorrections;

    // Restore vocabulary (frequency, categories, totalReadings)
    this.vocabulary.frequency = { ...snapshot.vocabFreqSnapshot };
    this.vocabulary.categoryAssoc = JSON.parse(JSON.stringify(snapshot.vocabCategorySnapshot));
    this.vocabulary.totalReadings = snapshot.vocabTotalReadings;
    this.vocabulary.reloadBaseVocabulary(this._baseVocabulary);

    // Restore context chains
    const ctxData = snapshot.contextSnapshot;
    this.context.chains = ctxData.chains || {};
    this.context.totalEvents = ctxData.totalEvents || 0;

    // Restore core engine state (reverses feed() calls).
    // Note: coreSnapshot is stripped during serialization (too large for disk),
    // so undo after restore only rolls back confusion/vocab/context, not core vectors.
    if (coreEngine && snapshot.coreSnapshot && typeof coreEngine.constructor.deserialize === 'function') {
      const restored = coreEngine.constructor.deserialize(
        typeof snapshot.coreSnapshot === 'string' ? snapshot.coreSnapshot : JSON.stringify(snapshot.coreSnapshot)
      );
      // Copy all state from restored into the live engine
      for (const key of Object.keys(restored)) {
        if (key !== 'config' && key !== 'version') {
          coreEngine[key] = restored[key];
        }
      }
    }

    this.correctionCount = Math.max(0, this.correctionCount - 1);

    return {
      undone: snapshot.id,
      timestamp: snapshot.timestamp,
      columns: snapshot.ocrRow ? Object.keys(snapshot.ocrRow) : (snapshot.columns || []),
      coreRestored: !!snapshot.coreSnapshot,
    };
  }

  /**
   * Get the correction history (most recent first).
   */
  getHistory(n = 10) {
    return this._history.slice(-n).reverse().map(h => ({
      id: h.id,
      timestamp: h.timestamp,
      columns: h.ocrRow ? Object.keys(h.ocrRow) : (h.columns || []),
      rejectedCount: h.rejectedCount || (h.rejected ? h.rejected.length : 0),
    }));
  }

  refreshBaseVocabulary(baseVocabulary = this._baseVocabulary) {
    this._baseVocabulary = baseVocabulary;
    this.vocabulary.reloadBaseVocabulary(baseVocabulary);
  }

  _alignTokenPairs(ocrText, confirmedText) {
    const ocrWords = String(ocrText || '').toLowerCase().split(/\s+/).filter(Boolean);
    const confirmedWords = String(confirmedText || '').toLowerCase().split(/\s+/).filter(Boolean);
    if (ocrWords.length === 0 || confirmedWords.length === 0) return [];

    const m = ocrWords.length;
    const n = confirmedWords.length;
    const dp = Array.from({ length: m + 1 }, () => new Array(n + 1).fill(0));

    for (let i = 0; i <= m; i++) dp[i][0] = i;
    for (let j = 0; j <= n; j++) dp[0][j] = j;

    const substitutionCost = (a, b) => {
      if (a === b) return 0;
      const dist = this.confusion.weightedDistance(a, b);
      return Math.min(dist / Math.max(Math.max(a.length, b.length), 1), 2);
    };

    for (let i = 1; i <= m; i++) {
      for (let j = 1; j <= n; j++) {
        const sub = substitutionCost(ocrWords[i - 1], confirmedWords[j - 1]);
        dp[i][j] = Math.min(
          dp[i - 1][j] + 1,
          dp[i][j - 1] + 1,
          dp[i - 1][j - 1] + sub
        );
      }
    }

    const pairs = [];
    let i = m, j = n;
    while (i > 0 && j > 0) {
      const sub = substitutionCost(ocrWords[i - 1], confirmedWords[j - 1]);
      if (Math.abs(dp[i][j] - (dp[i - 1][j - 1] + sub)) < 1e-9) {
        pairs.push({ ocr: ocrWords[i - 1], confirmed: confirmedWords[j - 1] });
        i--;
        j--;
      } else if (dp[i][j] === dp[i - 1][j] + 1) {
        i--;
      } else {
        j--;
      }
    }

    return pairs.reverse();
  }

  _recordAlignedCorrection(ocrText, confirmedText, clinicalWeight) {
    const pairs = this._alignTokenPairs(ocrText, confirmedText);
    if (pairs.length === 0) {
      this.confusion.recordCorrection(ocrText, confirmedText, clinicalWeight);
      return;
    }
    for (const pair of pairs) {
      this.confusion.recordCorrection(pair.ocr, pair.confirmed, clinicalWeight);
    }
  }

  _correctWord(rawWord, columnType, knownContext) {
    const word = rawWord.trim();
    if (!word) return { original: word, corrected: word, confidence: 0, flag: 'empty', candidates: [] };
    const wordLower = word.toLowerCase();

    // ── Guards (mirror clinical/corrector.js) ──────────────────────
    if (/^[\d.,/\-:]+$/.test(word)) {
      return { original: word, corrected: word, confidence: 0.8, flag: 'number', candidates: [] };
    }
    if (/^(dr|mr|mrs|ms|prof)\.?$/i.test(word)) {
      return { original: word, corrected: word, confidence: 0.95, flag: 'title', candidates: [] };
    }
    if (/^\d+\.?\d*(mg|mcg|g|ml|units?|iu|%|mmol|meq)$/i.test(word)) {
      return { original: word, corrected: word, confidence: 0.8, flag: 'dosage', candidates: [] };
    }
    if (word.length <= 6 && /\d/.test(word) && /[a-zA-Z]/.test(word)) {
      if (/^\d/.test(word) || (word === word.toUpperCase() && word.length <= 5) || word.includes('-')) {
        return { original: word, corrected: word, confidence: 0.7, flag: 'room_code', candidates: [] };
      }
    }
    if (wordLower.length <= 3 && !this.vocabulary.isKnown(wordLower)) {
      return { original: word, corrected: word, confidence: 0.5, flag: 'short', candidates: [] };
    }
    // ── End guards ─────────────────────────────────────────────────

    if (this.vocabulary.isKnown(wordLower)) {
      const boost = this.vocabulary.getFrequencyBoost(wordLower);
      return { original: word, corrected: word, confidence: 0.9 + boost * 0.1, flag: 'exact' };
    }
    const columnWords = this.vocabulary.getColumnWords(columnType);
    const maxDist = Math.max(word.length * 0.5, 2.0);
    const candidateWords = this.vocabulary.getCandidateWords(wordLower, {
      maxLengthDiff: wordLower.length <= 4 ? 1 : 3,
      maxCandidates: 768,
    });
    let candidates = [];
    for (const vocabWord of candidateWords) {
      if (Math.abs(vocabWord.length - wordLower.length) > 3) continue;
      const dist = this.confusion.weightedDistance(wordLower, vocabWord);
      if (dist > maxDist) continue;
      const freqBoost = this.vocabulary.getFrequencyBoost(vocabWord);
      const colBoost = columnWords.has(vocabWord) ? 0.4 : 0;
      const chainBoost = this.context.getBoost(knownContext, columnType, vocabWord);
      const lenRatio = Math.min(wordLower.length, vocabWord.length) / Math.max(wordLower.length, vocabWord.length);
      const score = Math.max(0,
        1.0 - (dist - colBoost - chainBoost - freqBoost * 0.5) / Math.max(wordLower.length, 3)
      ) * lenRatio;
      candidates.push({ word: vocabWord, distance: dist, score, freqBoost, colBoost, chainBoost });
    }
    candidates.sort((a, b) => b.score - a.score);
    candidates = candidates.slice(0, 5);
    if (candidates.length === 0) {
      return { original: word, corrected: word, confidence: 0, flag: 'unknown', candidates: [] };
    }
    const top = candidates[0];
    const margin = candidates.length > 1 ? top.score - candidates[1].score : top.score;
    let flag;
    if (top.score > 0.8 && margin > 0.15) flag = 'high_confidence';
    else if (top.score > 0.5) flag = 'verify';
    else flag = 'low_confidence';
    // Edit-distance-aware policy (matches standard corrector):
    // - high_confidence: always apply
    // - verify with small edit distance (<=2): apply (obvious OCR typo)
    // - verify with large edit distance (>2): keep original (suspicious)
    // - low_confidence: always keep original
    const editDist = this._editDistance(wordLower, top.word);
    const isUncertain = flag === 'low_confidence' || (flag === 'verify' && editDist > 2);
    const correctedWord = isUncertain ? word : this._preserveCase(word, top.word);
    return {
      original: word, corrected: correctedWord,
      confidence: top.score, flag,
      candidates: candidates.slice(0, 3).map(c => ({
        word: c.word, distance: Math.round(c.distance * 100) / 100,
        boosts: { frequency: Math.round(c.freqBoost * 100) / 100, column: Math.round(c.colBoost * 100) / 100, context: Math.round(c.chainBoost * 100) / 100 },
      })),
    };
  }

  _editDistance(a, b) {
    if (a.length === 0) return b.length;
    if (b.length === 0) return a.length;
    const m = Array.from({ length: a.length + 1 }, (_, i) => [i]);
    for (let j = 1; j <= b.length; j++) m[0][j] = j;
    for (let i = 1; i <= a.length; i++)
      for (let j = 1; j <= b.length; j++)
        m[i][j] = Math.min(m[i-1][j] + 1, m[i][j-1] + 1, m[i-1][j-1] + (a[i-1] !== b[j-1] ? 1 : 0));
    return m[a.length][b.length];
  }

  _preserveCase(original, corrected) {
    if (!original || !corrected) return corrected;
    if (original === original.toUpperCase() && original.length > 1) return corrected.toUpperCase();
    if (original[0] === original[0].toUpperCase()) return corrected[0].toUpperCase() + corrected.slice(1);
    return corrected.toLowerCase();
  }

  getStats() {
    return {
      totalCorrections: this.correctionCount,
      confusionPairs: Object.keys(this.confusion.confusionCounts).length,
      topConfusions: this.confusion.getTopConfusions(5),
      vocabularySize: this.vocabulary._allWords.size,
      learnedWords: this.vocabulary.getLearnedWords(2).length,
      contextChains: Object.keys(this.context.chains).length,
    };
  }

  toJSON() {
    // Strip core snapshots (too large for disk) and raw patient data (PHI)
    // from history before serialization. Rollback snapshots (confusion, vocab,
    // context) are kept since they contain learned parameters, not patient data.
    const lightHistory = this._history.slice(-this._historyLimit).map(h => {
      const { coreSnapshot, ocrRow, confirmedRow, rejected, ...rest } = h;
      // Preserve column names so history is useful after restore
      rest.columns = ocrRow ? Object.keys(ocrRow) : (h.columns || []);
      // Strip rejected words (contain confirmed input = PHI), keep only count
      rest.rejectedCount = rejected ? rejected.length : 0;
      return rest;
    });
    return {
      confusion: this.confusion.toJSON(),
      vocabulary: this.vocabulary.toJSON(),
      context: this.context.toJSON(),
      correctionCount: this.correctionCount,
      history: lightHistory,
      version: '2.0.0',
    };
  }

  static fromJSON(data, baseVocabulary = {}) {
    const e = new ShifuLearningEngine(baseVocabulary);
    if (data.confusion) e.confusion = AdaptiveConfusionProfile.fromJSON(data.confusion);
    if (data.vocabulary) e.vocabulary = WardVocabulary.fromJSON(data.vocabulary, baseVocabulary);
    if (data.context) e.context = ContextChains.fromJSON(data.context);
    e.correctionCount = data.correctionCount || 0;
    e._history = Array.isArray(data.history) ? data.history : [];
    return e;
  }
}

module.exports = {
  AdaptiveConfusionProfile, WardVocabulary, ContextChains, ShifuLearningEngine,
};
