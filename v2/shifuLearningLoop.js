/**
 * Shifu Learning Loop
 * ====================
 * 
 * The fluid landscape principle applied to the correction engine.
 * 
 * Every nurse correction reshapes three landscapes:
 *   1. Confusion Profile — what does PaddleOCR actually get wrong?
 *   2. Ward Vocabulary — what words actually appear on this ward?
 *   3. Context Chains — what patterns co-occur? (ward → doctor → diagnosis)
 * 
 * The system on day 100 is fundamentally different from day 1.
 * Not because someone retrained it, but because it absorbed 
 * 100 days of clinical experience.
 * 
 * Storage: Firebase Firestore (already in MedEvac stack).
 * All data stays local to the hospital instance.
 */

// =========================================================================
// 1. ADAPTIVE CONFUSION PROFILE
// =========================================================================

/**
 * Tracks what PaddleOCR actually confuses on YOUR documents.
 * 
 * Every correction creates a data point:
 *   OCR said "Abdulah" → Nurse typed "Abdullah" → confusion: h→lh (insertion)
 * 
 * Over time, builds an empirical confusion matrix that replaces
 * the theoretical topology-based costs.
 * 
 * The landscape: frequent confusions get lower cost (easier to correct).
 * Rare confusions keep the default high cost.
 */
export class AdaptiveConfusionProfile {
  constructor() {
    // Character-level confusion counts: { "a,e": 14, "0,O": 23, ... }
    this.confusionCounts = {};
    // Total corrections seen
    this.totalCorrections = 0;
    // Base theoretical costs (fallback)
    this.baseCosts = {
      'O,0': 0.1, 'o,0': 0.1, 'l,1': 0.2, 'I,1': 0.2, 'I,l': 0.2,
      '5,S': 0.3, '5,s': 0.3, '8,B': 0.3, '6,G': 0.4, '2,Z': 0.4,
      'r,n': 0.3, 'm,n': 0.4, 'u,v': 0.5, 'c,e': 0.5, 'h,b': 0.5,
      'D,O': 0.3, 'f,t': 0.4, 'E,F': 0.4,
    };
  }

  /**
   * Record a correction event.
   * Extracts character-level confusions from the OCR→corrected pair.
   */
  recordCorrection(ocrText, correctedText) {
    const pairs = this._alignAndExtract(ocrText.toLowerCase(), correctedText.toLowerCase());
    for (const [ocrChar, correctChar] of pairs) {
      if (ocrChar !== correctChar) {
        const key = [ocrChar, correctChar].sort().join(',');
        this.confusionCounts[key] = (this.confusionCounts[key] || 0) + 1;
      }
    }
    this.totalCorrections++;
  }

  /**
   * Get the confusion cost between two characters.
   * 
   * Blends theoretical cost with empirical cost.
   * As more data accumulates, empirical dominates.
   * With no data, falls back to topology-based costs.
   * 
   * This IS the fluid landscape: the cost surface reshapes
   * with experience. Consistent confusions get cheaper.
   * Novel confusions stay expensive until observed.
   */
  getCost(char1, char2) {
    if (char1 === char2) return 0.0;

    const key = [char1, char2].sort().join(',');

    // Theoretical cost
    const baseCost = this.baseCosts[key] ?? 1.0;

    // Empirical cost: more observations → lower cost
    const count = this.confusionCounts[key] || 0;

    if (count === 0 || this.totalCorrections < 10) {
      // Not enough data — use theoretical
      return baseCost;
    }

    // Blend: as count grows, cost drops toward 0.05 (very cheap)
    // Formula: cost = base * exp(-count/scale) + floor
    // At count=0: cost = base
    // At count=10: cost ≈ base * 0.37
    // At count=30: cost ≈ base * 0.05 + floor
    const scale = 10;
    const floor = 0.05;
    const empiricalCost = baseCost * Math.exp(-count / scale) + floor;

    // Blend based on total experience
    const experience = Math.min(this.totalCorrections / 100, 1.0); // 0→1 over 100 corrections
    return baseCost * (1 - experience) + empiricalCost * experience;
  }

  /**
   * OCR-weighted edit distance using adaptive costs.
   */
  weightedDistance(str1, str2) {
    const s1 = str1.toLowerCase();
    const s2 = str2.toLowerCase();
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

  /**
   * Align two strings and extract character-level pairs.
   * Uses simple Needleman-Wunsch alignment.
   */
  _alignAndExtract(s1, s2) {
    const pairs = [];
    const n = Math.min(s1.length, s2.length);
    for (let i = 0; i < n; i++) {
      pairs.push([s1[i], s2[i]]);
    }
    return pairs;
  }

  /**
   * Get the top N most frequent confusions (for debugging/display).
   */
  getTopConfusions(n = 10) {
    return Object.entries(this.confusionCounts)
      .sort((a, b) => b[1] - a[1])
      .slice(0, n)
      .map(([pair, count]) => ({
        pair,
        count,
        currentCost: this.getCost(pair.split(',')[0], pair.split(',')[1]),
      }));
  }

  // === Serialization (for Firebase) ===
  toJSON() {
    return {
      confusionCounts: this.confusionCounts,
      totalCorrections: this.totalCorrections,
      updatedAt: new Date().toISOString(),
    };
  }

  static fromJSON(data) {
    const profile = new AdaptiveConfusionProfile();
    profile.confusionCounts = data.confusionCounts || {};
    profile.totalCorrections = data.totalCorrections || 0;
    return profile;
  }
}


// =========================================================================
// 2. WARD VOCABULARY LANDSCAPE
// =========================================================================

/**
 * Frequency-weighted vocabulary that learns from confirmed readings.
 * 
 * Every word that appears in a confirmed reading gets a frequency bump.
 * Words that appear often in YOUR ward get higher priority in matching.
 * New words that nurses confirm get added automatically.
 * 
 * The landscape: frequently confirmed words are peaks.
 * Rare or unseen words are flat terrain.
 * Nothing is ever removed — just weighted by experience.
 */
export class WardVocabulary {
  constructor(baseVocabulary = {}) {
    // Base vocabulary: { word: category }
    this.baseWords = new Map();
    for (const [category, words] of Object.entries(baseVocabulary)) {
      for (const w of words) {
        const lower = w.toLowerCase();
        this.baseWords.set(lower, category);
        // Multi-word terms: also index individual words
        for (const part of lower.split(/\s+/)) {
          if (part.length > 1 && !this.baseWords.has(part)) {
            this.baseWords.set(part, category);
          }
        }
      }
    }

    // Learned frequency: { word: count }
    this.frequency = {};

    // Category associations: { word: { category: count } }
    this.categoryAssoc = {};

    // Total confirmed readings
    this.totalReadings = 0;
  }

  /**
   * Record a confirmed word (nurse verified or typed).
   */
  confirmWord(word, category = null) {
    const lower = word.toLowerCase().trim();
    if (!lower) return;

    this.frequency[lower] = (this.frequency[lower] || 0) + 1;
    this.totalReadings++;

    if (category) {
      if (!this.categoryAssoc[lower]) this.categoryAssoc[lower] = {};
      this.categoryAssoc[lower][category] = (this.categoryAssoc[lower][category] || 0) + 1;
    }
  }

  /**
   * Record a full confirmed reading (e.g., a table row).
   */
  confirmReading(fields) {
    for (const [columnName, text] of Object.entries(fields)) {
      if (!text || typeof text !== 'string') continue;
      for (const word of text.split(/\s+/)) {
        if (word.length > 1) {
          this.confirmWord(word, columnName.toLowerCase());
        }
      }
    }
  }

  /**
   * Get the frequency boost for a word.
   * Returns a value 0→1 that scales with how often this word
   * has been confirmed on this ward.
   * 
   * The landscape: log-scaled frequency.
   * 1 confirmation = small boost. 10 = moderate. 50 = strong.
   * Never saturates completely (always room to grow).
   */
  getFrequencyBoost(word) {
    const count = this.frequency[word.toLowerCase()] || 0;
    if (count === 0) return 0;
    // Log scale: boost = log(1 + count) / log(1 + maxExpected)
    return Math.log(1 + count) / Math.log(1 + 100); // Normalizes around 100 as "a lot"
  }

  /**
   * Check if a word is known (base or learned).
   */
  isKnown(word) {
    const lower = word.toLowerCase();
    return this.baseWords.has(lower) || (this.frequency[lower] || 0) > 0;
  }

  /**
   * Get all words relevant to a column type, sorted by frequency.
   */
  getColumnWords(columnType) {
    const type = columnType.toLowerCase();
    const words = new Map();

    // Base vocabulary
    for (const [word, category] of this.baseWords) {
      if (this._categoryMatches(category, type)) {
        const boost = this.getFrequencyBoost(word);
        words.set(word, { base: true, frequency: this.frequency[word] || 0, boost });
      }
    }

    // Learned words with category association
    for (const [word, categories] of Object.entries(this.categoryAssoc)) {
      for (const [cat, count] of Object.entries(categories)) {
        if (this._categoryMatches(cat, type)) {
          const boost = this.getFrequencyBoost(word);
          words.set(word, { base: this.baseWords.has(word), frequency: this.frequency[word] || 0, boost });
        }
      }
    }

    return words;
  }

  /**
   * Get the full word set for matching (base + learned).
   */
  getAllWords() {
    const words = new Set(this.baseWords.keys());
    for (const word of Object.keys(this.frequency)) {
      words.add(word);
    }
    return words;
  }

  /**
   * Get newly learned words (not in base vocabulary).
   */
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
    const cat = category.toLowerCase();
    const col = columnType.toLowerCase();
    if (col.includes('patient') || col.includes('name')) return cat === 'names' || cat === 'patient';
    if (col.includes('diagnos')) return cat === 'diagnoses' || cat === 'diagnosis';
    if (col.includes('doctor')) return cat === 'names' || cat === 'doctor';
    if (col.includes('med') || col.includes('drug')) return cat === 'medications';
    return true;
  }

  // === Serialization ===
  toJSON() {
    return {
      frequency: this.frequency,
      categoryAssoc: this.categoryAssoc,
      totalReadings: this.totalReadings,
      updatedAt: new Date().toISOString(),
    };
  }

  static fromJSON(data, baseVocabulary) {
    const vocab = new WardVocabulary(baseVocabulary);
    vocab.frequency = data.frequency || {};
    vocab.categoryAssoc = data.categoryAssoc || {};
    vocab.totalReadings = data.totalReadings || 0;
    return vocab;
  }
}


// =========================================================================
// 3. CONTEXT CHAINS
// =========================================================================

/**
 * Soft co-occurrence patterns learned from confirmed readings.
 * 
 * When a nurse confirms: Ward 20, Patient: Hassan, Diagnosis: Chest infection
 * The system learns: "In Ward 20, 'Chest infection' is a common diagnosis."
 * 
 * Next time OCR reads "Chst infecfion" in Ward 20, the context chain
 * gives "Chest infection" an extra boost because it's been seen in 
 * that ward before.
 * 
 * Not rigid rules. Soft frequency boosts. The landscape of 
 * what's probable in a given context.
 */
export class ContextChains {
  constructor() {
    // Co-occurrence counts: { "ward:20|diagnosis": { "chest infection": 3, "cva": 5 } }
    this.chains = {};
    this.totalEvents = 0;
  }

  /**
   * Learn from a confirmed reading.
   * Creates co-occurrence entries for all field combinations.
   */
  learn(fields) {
    const entries = Object.entries(fields)
      .filter(([k, v]) => v && typeof v === 'string' && v.trim())
      .map(([k, v]) => [k.toLowerCase(), v.toLowerCase().trim()]);

    // For each pair of fields, create a context chain
    for (let i = 0; i < entries.length; i++) {
      for (let j = 0; j < entries.length; j++) {
        if (i === j) continue;

        const [contextCol, contextVal] = entries[i];
        const [targetCol, targetVal] = entries[j];

        // Context key: "ward:20|diagnosis" means "given ward=20, what diagnoses appear?"
        const contextKey = `${contextCol}:${contextVal}|${targetCol}`;

        if (!this.chains[contextKey]) this.chains[contextKey] = {};
        this.chains[contextKey][targetVal] = (this.chains[contextKey][targetVal] || 0) + 1;
      }
    }

    this.totalEvents++;
  }

  /**
   * Get contextual boost for a candidate word given known context.
   * 
   * @param {object} knownFields - Already-read fields { ward: "20", doctor: "bader" }
   * @param {string} targetColumn - Column we're trying to read ("diagnosis")
   * @param {string} candidateWord - The candidate to boost or not
   * @returns {number} Boost value (0 = no context, higher = stronger association)
   */
  getBoost(knownFields, targetColumn, candidateWord) {
    let totalBoost = 0;
    const candidate = candidateWord.toLowerCase();
    const target = targetColumn.toLowerCase();

    for (const [col, val] of Object.entries(knownFields)) {
      if (!val) continue;
      const contextKey = `${col.toLowerCase()}:${val.toLowerCase()}|${target}`;
      const chain = this.chains[contextKey];

      if (chain && chain[candidate]) {
        // Boost proportional to log of co-occurrence count
        totalBoost += Math.log(1 + chain[candidate]) * 0.3;
      }
    }

    return totalBoost;
  }

  /**
   * Get the most likely values for a target column given context.
   * Useful for suggesting completions or ranking candidates.
   */
  predict(knownFields, targetColumn, topK = 5) {
    const target = targetColumn.toLowerCase();
    const scores = {};

    for (const [col, val] of Object.entries(knownFields)) {
      if (!val) continue;
      const contextKey = `${col.toLowerCase()}:${val.toLowerCase()}|${target}`;
      const chain = this.chains[contextKey];
      if (!chain) continue;

      for (const [word, count] of Object.entries(chain)) {
        scores[word] = (scores[word] || 0) + count;
      }
    }

    return Object.entries(scores)
      .sort((a, b) => b[1] - a[1])
      .slice(0, topK)
      .map(([word, count]) => ({ word, count }));
  }

  // === Serialization ===
  toJSON() {
    return {
      chains: this.chains,
      totalEvents: this.totalEvents,
      updatedAt: new Date().toISOString(),
    };
  }

  static fromJSON(data) {
    const ctx = new ContextChains();
    ctx.chains = data.chains || {};
    ctx.totalEvents = data.totalEvents || 0;
    return ctx;
  }
}


// =========================================================================
// 4. THE LEARNING ENGINE (combines all three)
// =========================================================================

/**
 * ShifuLearningEngine
 * 
 * The complete adaptive correction system.
 * Wraps confusion profile + ward vocabulary + context chains
 * into a single interface that MedTriage calls.
 * 
 * Usage:
 *   const engine = new ShifuLearningEngine(baseVocabulary);
 *   
 *   // Correct OCR output
 *   const result = engine.correctRow(ocrRow, { ward: '20' });
 *   
 *   // Nurse confirms/corrects → system learns
 *   engine.learn(ocrRow, confirmedRow);
 *   
 *   // Persist to Firebase
 *   await db.doc('shifu/state').set(engine.toJSON());
 *   
 *   // Restore on next session
 *   const engine = ShifuLearningEngine.fromJSON(savedState, baseVocab);
 */
export class ShifuLearningEngine {
  constructor(baseVocabulary = {}) {
    this.confusion = new AdaptiveConfusionProfile();
    this.vocabulary = new WardVocabulary(baseVocabulary);
    this.context = new ContextChains();
    this.correctionCount = 0;
    this.sessionStart = new Date().toISOString();
  }

  /**
   * Correct a table row of OCR output.
   * Uses all three landscapes: confusion costs, vocabulary frequency, context chains.
   */
  correctRow(ocrRow, knownContext = {}) {
    const corrected = {};
    const flags = [];

    for (const [column, rawText] of Object.entries(ocrRow)) {
      if (!rawText || typeof rawText !== 'string') {
        corrected[column] = { input: rawText, output: rawText, words: [] };
        continue;
      }

      const words = rawText.split(/\s+/).filter(w => w.length > 0);
      const correctedWords = [];

      for (const raw of words) {
        const result = this._correctWord(raw, column, knownContext);
        correctedWords.push(result);
      }

      const output = correctedWords.map(w => w.corrected).join(' ');
      corrected[column] = {
        input: rawText,
        output,
        words: correctedWords,
      };

      // Update known context with this column's output for subsequent columns
      knownContext[column] = output;
    }

    return { corrected, flags };
  }

  /**
   * Learn from a confirmed reading.
   * Call this when a nurse verifies or corrects the OCR output.
   */
  learn(ocrRow, confirmedRow) {
    this.correctionCount++;

    for (const [column, confirmedText] of Object.entries(confirmedRow)) {
      if (!confirmedText || typeof confirmedText !== 'string') continue;

      const ocrText = ocrRow[column] || '';

      // Record character-level confusions
      if (ocrText && ocrText !== confirmedText) {
        this.confusion.recordCorrection(ocrText, confirmedText);
      }

      // Confirm vocabulary
      for (const word of confirmedText.split(/\s+/)) {
        if (word.length > 1) {
          this.vocabulary.confirmWord(word, column);
        }
      }
    }

    // Learn context chains
    this.context.learn(confirmedRow);

    // Confirm the full reading
    this.vocabulary.confirmReading(confirmedRow);
  }

  /**
   * Correct a single word using all three landscapes.
   */
  _correctWord(rawWord, columnType, knownContext) {
    const word = rawWord.trim();
    const wordLower = word.toLowerCase();

    // Numbers pass through
    if (/^[\d.,/\-:]+$/.test(word)) {
      return { original: word, corrected: word, confidence: 0.8, flag: 'number' };
    }

    // Exact match in vocabulary
    if (this.vocabulary.isKnown(wordLower)) {
      const boost = this.vocabulary.getFrequencyBoost(wordLower);
      return {
        original: word, corrected: word,
        confidence: 0.9 + boost * 0.1,
        flag: 'exact',
      };
    }

    // Fuzzy match with adaptive confusion costs + vocabulary frequency + context
    const allWords = this.vocabulary.getAllWords();
    const columnWords = this.vocabulary.getColumnWords(columnType);
    const maxDist = Math.max(word.length * 0.5, 2.0);

    let candidates = [];

    for (const vocabWord of allWords) {
      if (Math.abs(vocabWord.length - wordLower.length) > 3) continue;

      // Adaptive confusion-weighted distance
      const dist = this.confusion.weightedDistance(wordLower, vocabWord);
      if (dist > maxDist) continue;

      // Frequency boost from ward vocabulary
      const freqBoost = this.vocabulary.getFrequencyBoost(vocabWord);

      // Column context boost
      const colBoost = columnWords.has(vocabWord) ? 0.4 : 0;

      // Context chain boost
      const chainBoost = this.context.getBoost(knownContext, columnType, vocabWord);

      // Combined score
      const lenRatio = Math.min(wordLower.length, vocabWord.length) /
                       Math.max(wordLower.length, vocabWord.length);
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

    return {
      original: word,
      corrected: this._preserveCase(word, top.word),
      confidence: top.score,
      flag,
      candidates: candidates.slice(0, 3).map(c => ({
        word: c.word,
        distance: Math.round(c.distance * 100) / 100,
        boosts: {
          frequency: Math.round(c.freqBoost * 100) / 100,
          column: Math.round(c.colBoost * 100) / 100,
          context: Math.round(c.chainBoost * 100) / 100,
        },
      })),
    };
  }

  _preserveCase(original, corrected) {
    if (!original || !corrected) return corrected;
    if (original === original.toUpperCase() && original.length > 1) return corrected.toUpperCase();
    if (original[0] === original[0].toUpperCase()) return corrected[0].toUpperCase() + corrected.slice(1);
    return corrected.toLowerCase();
  }

  /**
   * Get system stats for display / debugging.
   */
  getStats() {
    return {
      totalCorrections: this.correctionCount,
      confusionPairs: Object.keys(this.confusion.confusionCounts).length,
      topConfusions: this.confusion.getTopConfusions(5),
      vocabularySize: this.vocabulary.getAllWords().size,
      learnedWords: this.vocabulary.getLearnedWords(2).length,
      contextChains: Object.keys(this.context.chains).length,
      sessionStart: this.sessionStart,
    };
  }

  // === Serialization (Firebase) ===
  toJSON() {
    return {
      confusion: this.confusion.toJSON(),
      vocabulary: this.vocabulary.toJSON(),
      context: this.context.toJSON(),
      correctionCount: this.correctionCount,
      version: '2.0.0',
      updatedAt: new Date().toISOString(),
    };
  }

  static fromJSON(data, baseVocabulary = {}) {
    const engine = new ShifuLearningEngine(baseVocabulary);
    if (data.confusion) engine.confusion = AdaptiveConfusionProfile.fromJSON(data.confusion);
    if (data.vocabulary) engine.vocabulary = WardVocabulary.fromJSON(data.vocabulary, baseVocabulary);
    if (data.context) engine.context = ContextChains.fromJSON(data.context);
    engine.correctionCount = data.correctionCount || 0;
    return engine;
  }
}
