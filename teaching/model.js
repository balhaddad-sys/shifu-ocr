// Shifu Teaching Model v1.0.0
// Cross-domain teaching engine with calibration-driven learning.
//
// The teaching model wraps ShifuEngine and extends it with:
// 1. Cross-domain knowledge transfer — learning in one domain improves others
// 2. Calibration persistence — every correction saved for future sessions
// 3. Progressive curriculum — structured learning paths across domains
// 4. Multi-source seeding — Kaggle, HuggingFace, custom corpora
//
// Philosophy: Teaching is not training. Training adjusts weights.
// Teaching builds understanding — connections between domains,
// patterns that transfer, confusions that are universal vs domain-specific.

const { DomainRegistry } = require('./domains');
const { CalibrationStore } = require('./calibration');

// ─── Teaching Configuration ─────────────────────────────────────────────

const TEACHING_CONFIG = {
  // Transfer learning
  transfer: {
    minCalibrations: 10,    // Minimum calibrations before transfer kicks in
    transferDecay: 0.85,    // How much knowledge decays during transfer
    maxTransferHops: 2,     // Maximum chain of domain transfers
    blendRate: 0.3,         // How much transferred knowledge blends with native
  },
  // Curriculum
  curriculum: {
    batchSize: 50,          // Sentences per learning batch
    warmupBatches: 3,       // Easy batches before difficulty increases
    difficultyRamp: 0.15,   // How fast difficulty increases per batch
    maxDifficulty: 0.9,     // Cap on difficulty
    reviewInterval: 5,      // Re-test after N batches
  },
  // Calibration
  calibration: {
    autoSaveInterval: 25,   // Save after every N calibrations
    compactAfterDays: 30,   // Compact old calibrations after N days
    highValueThreshold: 0.15, // Error threshold for high-value calibrations
  },
  // Seeding
  seeding: {
    maxSamplesPerSource: 5000,  // Max samples from each source
    shuffleSeed: 42,            // Reproducible shuffling
    validationSplit: 0.1,       // Hold out 10% for validation
  },
};

// ─── Teaching Session ───────────────────────────────────────────────────

class TeachingSession {
  constructor(domain, subdomain) {
    this.id = `session_${Date.now()}`;
    this.domain = domain;
    this.subdomain = subdomain || 'general';
    this.startedAt = Date.now();
    this.endedAt = null;
    this.batches = [];
    this.currentBatch = 0;
    this.difficulty = 0;
    this.metrics = {
      totalSentences: 0,
      correctPredictions: 0,
      corrections: 0,
      averageConfidence: 0,
      confusionPairs: {},
    };
  }

  recordBatchResult(batchResult) {
    this.batches.push(batchResult);
    this.currentBatch++;
    this.metrics.totalSentences += batchResult.sentences;
    this.metrics.correctPredictions += batchResult.correct;
    this.metrics.corrections += batchResult.corrections;

    // Update running average confidence
    const n = this.batches.length;
    this.metrics.averageConfidence =
      ((this.metrics.averageConfidence * (n - 1)) + batchResult.avgConfidence) / n;

    // Adjust difficulty based on performance
    const batchAccuracy = batchResult.correct / Math.max(batchResult.sentences, 1);
    if (batchAccuracy > 0.85) {
      this.difficulty = Math.min(TEACHING_CONFIG.curriculum.maxDifficulty,
        this.difficulty + TEACHING_CONFIG.curriculum.difficultyRamp);
    } else if (batchAccuracy < 0.5) {
      this.difficulty = Math.max(0, this.difficulty - TEACHING_CONFIG.curriculum.difficultyRamp);
    }
  }

  end() {
    this.endedAt = Date.now();
    return this.summary();
  }

  summary() {
    return {
      id: this.id,
      domain: this.domain,
      subdomain: this.subdomain,
      duration: (this.endedAt || Date.now()) - this.startedAt,
      batches: this.batches.length,
      difficulty: this.difficulty,
      accuracy: this.metrics.totalSentences > 0
        ? this.metrics.correctPredictions / this.metrics.totalSentences : 0,
      ...this.metrics,
    };
  }
}

// ─── Teaching Model ─────────────────────────────────────────────────────

class ShifuTeachingModel {
  constructor(shifuEngine, learningEngine, options = {}) {
    this.engine = shifuEngine;
    this.learning = learningEngine;
    this.config = { ...TEACHING_CONFIG, ...options };

    // Core components
    this.domains = new DomainRegistry();
    this.calibrations = new CalibrationStore(options.calibrationDir);

    // Session tracking
    this.sessions = [];
    this.activeSession = null;
    this._calibrationCount = 0;

    // Domain-specific learned knowledge
    this.domainKnowledge = {};

    // Cross-domain confusion model — universal confusions
    this.universalConfusions = {};

    // Load persisted calibrations
    this.calibrations.load();
    this._rebuildUniversalConfusions();
  }

  // ─── Domain Management ──────────────────────────────────────────────

  /** Activate a domain for teaching */
  activateDomain(domainId) {
    const domain = this.domains.activate(domainId);
    if (!this.domainKnowledge[domainId]) {
      this.domainKnowledge[domainId] = {
        vocabulary: new Set(),
        patterns: [],
        confusionOverrides: {},
        sentences: 0,
        corrections: 0,
      };
    }
    return domain;
  }

  /** Auto-detect and activate domains from text */
  autoDetect(text) {
    const detected = this.domains.detect(text);
    for (const { domain } of detected) {
      if (domain) this.activateDomain(domain.id);
    }
    return detected;
  }

  /** Register a custom domain */
  registerDomain(domainDef) {
    return this.domains.register(domainDef);
  }

  // ─── Teaching Sessions ──────────────────────────────────────────────

  /** Start a new teaching session */
  startSession(domainId, subdomain) {
    if (this.activeSession) this.endSession();
    this.activateDomain(domainId);
    this.activeSession = new TeachingSession(domainId, subdomain);
    this.sessions.push(this.activeSession);
    return this.activeSession;
  }

  /** End the current teaching session */
  endSession() {
    if (!this.activeSession) return null;
    const summary = this.activeSession.end();
    this.activeSession = null;
    this.calibrations.save();
    return summary;
  }

  // ─── Core Teaching Methods ──────────────────────────────────────────

  /**
   * Teach from a single sentence.
   * The engine processes the sentence, and we record calibration data.
   */
  teachSentence(sentence, domain, options = {}) {
    const domainId = domain || this._inferDomain(sentence);
    this.activateDomain(domainId);

    // Feed to engine
    const words = sentence.toLowerCase().split(/\s+/).filter(w => w.length > 0);
    if (words.length === 0) return null;

    this.engine.feed(sentence);

    // Track domain knowledge
    const dk = this.domainKnowledge[domainId];
    dk.sentences++;
    for (const word of words) {
      dk.vocabulary.add(word);
    }

    // If we have expected output, record calibration
    if (options.expected) {
      const cal = this.calibrations.record({
        domain: domainId,
        subdomain: options.subdomain,
        input: sentence,
        expected: options.expected,
        predicted: sentence, // Will be replaced with actual prediction
        confidence: options.confidence || 0.5,
      });
      this._checkAutoSave();
      return { calibration: cal, words: words.length };
    }

    return { words: words.length };
  }

  /**
   * Teach from a corpus of sentences.
   * Feeds in batches with difficulty progression.
   */
  teachCorpus(sentences, domain, options = {}) {
    const domainId = domain || 'general';
    const batchSize = options.batchSize || this.config.curriculum.batchSize;
    const results = [];

    for (let i = 0; i < sentences.length; i += batchSize) {
      const batch = sentences.slice(i, i + batchSize);
      const batchResult = this._teachBatch(batch, domainId, options);
      results.push(batchResult);

      if (this.activeSession) {
        this.activeSession.recordBatchResult(batchResult);
      }
    }

    return {
      domain: domainId,
      totalSentences: sentences.length,
      batches: results.length,
      results,
    };
  }

  _teachBatch(batch, domainId, options) {
    let correct = 0;
    let corrections = 0;
    let totalConfidence = 0;

    for (const sentence of batch) {
      const result = this.teachSentence(sentence, domainId, options);
      if (result) {
        if (result.calibration && result.calibration.error === 0) correct++;
        if (result.calibration && result.calibration.correction) corrections++;
        totalConfidence += (result.calibration?.confidence || 0.5);
      }
    }

    return {
      sentences: batch.length,
      correct,
      corrections,
      avgConfidence: batch.length > 0 ? totalConfidence / batch.length : 0,
    };
  }

  /**
   * Teach from a correction (human feedback).
   * This is the highest-value teaching signal.
   */
  teachCorrection(ocrText, correctedText, domain, options = {}) {
    const domainId = domain || this._inferDomain(correctedText);
    this.activateDomain(domainId);

    const clinicalWeight = options.weight || 1;

    // Record calibration
    const cal = this.calibrations.recordCorrection(
      domainId, ocrText, correctedText, options.confidence || 0, clinicalWeight
    );

    // Feed correction to learning engine if available
    if (this.learning && typeof this.learning.learnCorrection === 'function') {
      this.learning.learnCorrection(ocrText, correctedText, {
        clinicalWeight,
        column: options.column,
      });
    }

    // Update domain knowledge
    const dk = this.domainKnowledge[domainId];
    dk.corrections++;
    const correctedWords = correctedText.toLowerCase().split(/\s+/);
    for (const w of correctedWords) dk.vocabulary.add(w);

    // Cross-domain transfer: if this confusion is universal, apply to all active domains
    this._attemptTransfer(cal);

    this._checkAutoSave();

    return {
      calibration: cal,
      domain: domainId,
      transferred: cal._transferred || false,
    };
  }

  /**
   * Teach from seeded data (Kaggle, HuggingFace, etc.).
   * Expects pre-processed data from the Python seed pipeline.
   */
  teachFromSeed(seedData) {
    const { domain, source, sentences, pairs } = seedData;
    const domainId = domain || 'general';

    this.activateDomain(domainId);

    const results = { domain: domainId, source, sentences: 0, pairs: 0 };

    // Teach raw sentences for vocabulary and co-occurrence
    if (sentences && sentences.length > 0) {
      const corpusResult = this.teachCorpus(sentences, domainId);
      results.sentences = corpusResult.totalSentences;
    }

    // Teach correction pairs for confusion calibration
    if (pairs && pairs.length > 0) {
      for (const { input, expected, confidence } of pairs) {
        this.teachCorrection(input, expected, domainId, { confidence });
      }
      results.pairs = pairs.length;
    }

    return results;
  }

  // ─── Cross-Domain Transfer ──────────────────────────────────────────

  _attemptTransfer(calibrationPoint) {
    if (!calibrationPoint.correction) return;

    const { from, to } = calibrationPoint.correction;
    const sourceDomain = calibrationPoint.domain;

    // Extract character-level confusions
    const minLen = Math.min(from.length, to.length);
    const confusions = [];
    for (let i = 0; i < minLen; i++) {
      if (from[i] !== to[i]) {
        confusions.push([from[i].toLowerCase(), to[i].toLowerCase()].sort().join(','));
      }
    }

    if (confusions.length === 0) return;

    // Check if confusions are universal (appear in multiple domains)
    for (const pair of confusions) {
      const ledger = this.calibrations.confusionLedger[pair];
      if (ledger && Object.keys(ledger.domains).length >= 2) {
        // This is a universal confusion — update all active domains
        if (!this.universalConfusions[pair]) {
          this.universalConfusions[pair] = { count: 0, domains: new Set() };
        }
        this.universalConfusions[pair].count++;
        this.universalConfusions[pair].domains.add(sourceDomain);
      }
    }

    // Transfer to related domains
    const transferWeights = this.domains.get(sourceDomain)?.transferWeights || {};
    for (const [targetDomain, weight] of Object.entries(transferWeights)) {
      if (weight < 0.2) continue; // Too weak to transfer
      if (!this.domains.get(targetDomain)) continue;

      const decay = this.config.transfer.transferDecay;
      const transferStrength = weight * decay;

      // Record transfer in calibrations
      this.calibrations.recordTransfer(
        sourceDomain, targetDomain,
        from, to,
        transferStrength > 0.3
      );

      calibrationPoint._transferred = true;
    }
  }

  _rebuildUniversalConfusions() {
    const crossDomain = this.calibrations.getCrossDomainConfusions(2);
    for (const { pair, count, domains } of crossDomain) {
      this.universalConfusions[pair] = {
        count,
        domains: new Set(Object.keys(domains)),
      };
    }
  }

  // ─── Query & Analysis ──────────────────────────────────────────────

  /** Get the best confusion profile for current active domains */
  getActiveConfusionProfile() {
    const blended = this.domains.getBlendedConfusion();

    // Layer in universal confusions
    for (const [pair, data] of Object.entries(this.universalConfusions)) {
      if (data.count >= 3) {
        const baseCost = blended[pair] || 0.5;
        // Universal confusions get lower cost (more likely)
        blended[pair] = Math.min(baseCost, 0.1 + 0.05 * (1 / Math.log2(data.count + 2)));
      }
    }

    return blended;
  }

  /** Get vocabulary for active domains */
  getActiveVocabulary() {
    const vocab = new Set();
    for (const domainId of this.domains.activeDomains) {
      const dk = this.domainKnowledge[domainId];
      if (dk) {
        for (const word of dk.vocabulary) vocab.add(word);
      }
    }
    return vocab;
  }

  /** Get teaching progress report */
  getProgress() {
    const calSummary = this.calibrations.summary();
    const domainProgress = {};

    for (const [domainId, dk] of Object.entries(this.domainKnowledge)) {
      const acc = this.calibrations.getDomainAccuracy(domainId);
      domainProgress[domainId] = {
        vocabularySize: dk.vocabulary.size,
        sentencesTaught: dk.sentences,
        correctionsMade: dk.corrections,
        accuracy: acc?.accuracy || null,
        averageError: acc?.averageError || null,
      };
    }

    return {
      totalCalibrations: calSummary.totalCalibrations,
      unappliedCalibrations: calSummary.unapplied,
      activeDomains: [...this.domains.activeDomains],
      domainProgress,
      universalConfusions: Object.keys(this.universalConfusions).length,
      sessions: this.sessions.length,
      transferEffectiveness: calSummary.transferEffectiveness,
    };
  }

  /** Get recommendations for what to teach next */
  getRecommendations() {
    const recommendations = [];
    const accuracies = this.calibrations.getAllAccuracies();

    // Recommend domains with low accuracy
    for (const [domainId, acc] of Object.entries(accuracies)) {
      if (acc && acc.accuracy < 0.7 && acc.total >= 10) {
        recommendations.push({
          type: 'low_accuracy',
          domain: domainId,
          accuracy: acc.accuracy,
          message: `Domain "${domainId}" has ${(acc.accuracy * 100).toFixed(1)}% accuracy — needs more teaching`,
        });
      }
    }

    // Recommend high-value unapplied calibrations
    const highValue = this.calibrations.getHighValue(10);
    if (highValue.length > 0) {
      recommendations.push({
        type: 'high_value_pending',
        count: highValue.length,
        message: `${highValue.length} high-value calibrations waiting to be applied`,
      });
    }

    // Recommend domains not yet touched
    for (const domain of this.domains.list()) {
      if (!this.domainKnowledge[domain.id]) {
        recommendations.push({
          type: 'untouched_domain',
          domain: domain.id,
          priority: domain.priority,
          message: `Domain "${domain.label}" has not been taught yet`,
        });
      }
    }

    // Recommend cross-domain transfer opportunities
    const transfers = this.calibrations.getTransferEffectiveness();
    for (const [key, stats] of Object.entries(transfers)) {
      if (stats.rate > 0.6 && stats.total >= 5) {
        recommendations.push({
          type: 'strong_transfer',
          path: key,
          rate: stats.rate,
          message: `Transfer ${key} is ${(stats.rate * 100).toFixed(0)}% effective — leverage this`,
        });
      }
    }

    return recommendations.sort((a, b) => {
      const priority = { low_accuracy: 3, high_value_pending: 2, untouched_domain: 1, strong_transfer: 1 };
      return (priority[b.type] || 0) - (priority[a.type] || 0);
    });
  }

  // ─── Helpers ────────────────────────────────────────────────────────

  _inferDomain(text) {
    const detected = this.domains.detect(text);
    return detected[0]?.domain?.id || 'general';
  }

  _checkAutoSave() {
    this._calibrationCount++;
    if (this._calibrationCount % this.config.calibration.autoSaveInterval === 0) {
      this.calibrations.save();
    }
  }

  // ─── Persistence ────────────────────────────────────────────────────

  serialize() {
    return {
      version: '1.0.0',
      savedAt: new Date().toISOString(),
      domainKnowledge: Object.fromEntries(
        Object.entries(this.domainKnowledge).map(([id, dk]) => [id, {
          vocabulary: [...dk.vocabulary],
          patterns: dk.patterns,
          confusionOverrides: dk.confusionOverrides,
          sentences: dk.sentences,
          corrections: dk.corrections,
        }])
      ),
      universalConfusions: Object.fromEntries(
        Object.entries(this.universalConfusions).map(([pair, data]) => [pair, {
          count: data.count,
          domains: [...data.domains],
        }])
      ),
      domains: this.domains.serialize(),
      sessionHistory: this.sessions.map(s => s.summary()),
    };
  }

  restore(state) {
    if (state.domainKnowledge) {
      for (const [id, dk] of Object.entries(state.domainKnowledge)) {
        this.domainKnowledge[id] = {
          vocabulary: new Set(dk.vocabulary || []),
          patterns: dk.patterns || [],
          confusionOverrides: dk.confusionOverrides || {},
          sentences: dk.sentences || 0,
          corrections: dk.corrections || 0,
        };
      }
    }
    if (state.universalConfusions) {
      for (const [pair, data] of Object.entries(state.universalConfusions)) {
        this.universalConfusions[pair] = {
          count: data.count,
          domains: new Set(data.domains || []),
        };
      }
    }
    if (state.domains) {
      this.domains.restore(state.domains);
    }
  }

  save() {
    this.calibrations.save();
  }
}

module.exports = { ShifuTeachingModel, TeachingSession, TEACHING_CONFIG };
