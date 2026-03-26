// SHIFU OCR v2.0.0 — Unified Medical OCR Engine
//
// Architecture:
//   Core Engine (v2.0.0)  — Resonance learning, soft trajectories, skip-gram expectations
//   Clinical Layer         — Vocabulary, confusion model, safety flags, corrector
//   Learning Loop          — Adaptive confusion, ward vocabulary, context chains
//   Medical Corpus         — Pre-trained medical domain knowledge
//   Trained Model          — Python OCR character landscapes → JS confusion awareness
//   Pipeline               — Image → Python OCR → JS correction → safety-checked output
//   Persistence            — Auto-save/load learning state between sessions
//   V2 Modules             — Clinical corrector, confusion model, safety flags, vocabulary (Firebase-ready)
//   Training Pipeline      — Privacy shield, seed harvesting, bulk generation, PaddleOCR fine-tuning
//
// Python OCR Multi-Engine Ensemble (shifu_ocr/):
//   ensemble.py            — Orchestrator: all engines see every character simultaneously
//   engine.py              — Topology engine (static form: components, holes, symmetry)
//   fluid.py               — Fluid engine (probability landscapes shaped by experience)
//   perturbation.py        — Perturbation engine (MRI-style relaxation signatures)
//   theory_revision.py     — Theory-revision engine (auditable reasoning)
//   codefining.py          — Co-defining engine (bidirectional char↔word↔context)
//   coherence.py           — Coherence displacement (colored backgrounds)
//   displacement.py        — Formal medium displacement theory
//   photoreceptor.py       — Smart per-cell adaptive binarization
//   complete.py            — Full integrated pipeline
//
// The core engine learns LANGUAGE (what follows what, what substitutes for what).
// The clinical layer knows MEDICINE (vocabulary, lab ranges, medication safety).
// The learning loop ADAPTS (every nurse correction reshapes the system).
// The trained model bridges VISION (Python character landscapes → JS confusion patterns).
// Together: a medical OCR corrector that gets smarter with every use.

const { ShifuEngine, CONFIG, VERSION, ocrDist, levDist, cos, cosVec, IDX } = require('./core/engine');
const {
  VOCABULARY,
  buildWordSet,
  getColumnVocabulary,
  ensureVocabularySize,
} = require('./clinical/vocabulary');
const { CONFUSION_PAIRS, getConfusionCost, ocrWeightedDistance, fixDigraphs, DIGRAPH_CONFUSIONS, setAdaptiveProfile } = require('./clinical/confusion');
const { LAB_RANGES, checkLabRange, checkMedicationAmbiguity, checkDosePlausibility, runSafetyChecks } = require('./clinical/safety');
const { correctWord, correctLine, correctTableRow, assessConfidence } = require('./clinical/corrector');
const { ShifuLearningEngine, AdaptiveConfusionProfile, WardVocabulary, ContextChains } = require('./learning/loop');
const { MEDICAL_CORPUS, seedEngine, generateWardSentences, fullSeed } = require('./learning/corpus');
const { CLINICAL_WEIGHTS, getColumnWeight, getWordWeight, getLearningRate, requiresSafetyReview } = require('./learning/clinical_weights');
const { feedTrainedModelToEngine, loadTrainedModel, extractConfusionKnowledge } = require('./core/trainedLoader');
const { ShifuPipeline } = require('./core/pipeline');
const { ShifuPersistence, withAutoSave } = require('./core/persistence');
const { FeedbackLoop } = require('./core/feedback');
const { DocumentIngestor } = require('./core/ingest');
const { MetricsTracker } = require('./core/metrics');
const { extractRoles, compareRoles, structuralInvariance } = require('./core/invariance');
const { normalizeDigits, normalizeDose, normalizeLineNumeric } = require('./clinical/numeric');

/**
 * Create a fully initialized Shifu system.
 * Loads the trained model, seeds with medical corpus, and optionally restores saved state.
 *
 * @param {object} opts
 * @param {boolean} opts.seed - Pre-seed with medical corpus (default: true)
 * @param {boolean} opts.loadTrained - Load Python trained model data (default: true)
 * @param {boolean} opts.autoSave - Enable auto-save after corrections (default: false)
 * @param {string} opts.stateDir - Directory for persistence
 * @param {number} opts.vocabularyTargetSize - Expand the base vocabulary up to this many tokens
 * @param {object} opts.savedCoreState - Serialized core engine state to restore
 * @param {object} opts.savedLearningState - Serialized learning engine state to restore
 * @returns {object} The complete Shifu system
 */
function createShifu(opts = {}) {
  const {
    seed = true,
    loadTrained = true,
    autoSave = false,
    stateDir,
    vocabularyTargetSize = 0,
    savedCoreState = null,
    savedLearningState = null,
  } = opts;

  if (vocabularyTargetSize > 0) {
    ensureVocabularySize(vocabularyTargetSize);
  }

  // Core engine: language understanding with resonance
  let core;
  if (savedCoreState) {
    core = ShifuEngine.deserialize(
      typeof savedCoreState === 'string' ? savedCoreState : JSON.stringify(savedCoreState)
    );
  } else {
    core = new ShifuEngine();
  }

  // Learning engine: adaptive correction with three landscapes
  let learning;
  if (savedLearningState) {
    learning = ShifuLearningEngine.fromJSON(savedLearningState, VOCABULARY);
  } else {
    learning = new ShifuLearningEngine(VOCABULARY);
  }

  // Wire adaptive confusion profile into static module so ALL paths benefit from learning
  setAdaptiveProfile(learning.confusion);

  // Pre-seed with medical corpus if starting fresh
  if (seed && !savedCoreState) {
    fullSeed(core);
  }

  // Load trained model confusion knowledge into core engine
  if (loadTrained && !savedCoreState) {
    feedTrainedModelToEngine(core);
  }

  const shifu = {
    core,
    learning,

    // ── Correct OCR output ──────────────────────────────────────────
    correctLine(ocrText, options = {}) {
      return correctLine(ocrText, {
        learningEngine: learning,
        coreEngine: core,
        ...options,
      });
    },

    correctTableRow(row, options = {}) {
      return correctTableRow(row, {
        learningEngine: learning,
        coreEngine: core,
        ...options,
      });
    },

    correctRowAdaptive(ocrRow, knownContext = {}) {
      const result = learning.correctRow(ocrRow, knownContext);
      for (const [col, data] of Object.entries(result.corrected)) {
        if (data.output && typeof data.output === 'string') {
          const scored = core.scoreSentence(data.output);
          data.coherence = scored.coherence;
          data.meanSurprise = scored.meanSurprise;
        }
      }
      return result;
    },

    // ── Learn from corrections ──────────────────────────────────────
    learn(ocrRow, confirmedRow) {
      return learning.learn(ocrRow, confirmedRow, core);
    },

    undo() {
      return learning.undo(core);
    },

    expandVocabulary(targetSize = 100000) {
      const expansion = ensureVocabularySize(targetSize);
      learning.refreshBaseVocabulary(VOCABULARY);
      return expansion;
    },

    getHistory(n = 10) {
      return learning.getHistory(n);
    },

    // ── Core engine queries ─────────────────────────────────────────
    scoreSentence(text) { return core.scoreSentence(text); },
    compare(a, b, profile = 'meaning') { return core.compare(a, b, profile); },
    similar(word, k = 10) { return core.similar(word, k); },
    correct(garbled, k = 5) { return core.correct(garbled, k); },
    resonancePartners(word, k = 10) { return core.resonancePartners(word, k); },

    // ── Structural invariance ───────────────────────────────────────
    extractRoles(sentence) { return extractRoles(sentence, core); },
    compareStructure(sentA, sentB) { return structuralInvariance(sentA, sentB, core); },

    // ── Pipeline (image → corrected text) ───────────────────────────
    createPipeline(pipelineOpts = {}) {
      return new ShifuPipeline(shifu, pipelineOpts);
    },

    // ── Safety checks ───────────────────────────────────────────────
    checkLabRange,
    checkMedicationAmbiguity,
    checkDosePlausibility,
    assessConfidence,

    // ── System info ─────────────────────────────────────────────────
    stats() {
      return {
        core: core.stats(),
        learning: learning.getStats(),
        vocabulary: {
          baseSize: buildWordSet().size,
          learnedWords: learning.vocabulary.getLearnedWords(1).length,
        },
      };
    },

    serialize() {
      return {
        core: core.serialize(),
        learning: learning.toJSON(),
        version: VERSION,
        savedAt: new Date().toISOString(),
      };
    },
  };

  // Wrap with auto-save if requested
  if (autoSave) {
    return withAutoSave(shifu, { stateDir });
  }

  return shifu;
}

/**
 * Restore a Shifu system from serialized state.
 */
function restoreShifu(savedState, opts = {}) {
  return createShifu({
    seed: false,
    loadTrained: false,
    ...opts,
    savedCoreState: savedState.core,
    savedLearningState: savedState.learning,
  });
}

/**
 * Create a Shifu system that auto-loads from saved state if available.
 * Falls back to fresh initialization if no saved state found.
 */
function createOrRestore(opts = {}) {
  const persistence = new ShifuPersistence(opts.stateDir);
  const saved = persistence.load();
  if (saved) {
    // persistence.load() returns { savedCoreState, savedLearningState, meta }
    // restoreShifu() expects { core, learning } — bridge the key names
    const shifu = createShifu({
      seed: false,
      loadTrained: false,
      autoSave: true,
      ...opts,
      stateDir: opts.stateDir,
      savedCoreState: saved.savedCoreState,
      savedLearningState: saved.savedLearningState,
    });
    return shifu;
  }
  return createShifu({ autoSave: true, ...opts });
}

module.exports = {
  // Main API
  createShifu,
  restoreShifu,
  createOrRestore,

  // Core engine (for direct use)
  ShifuEngine, CONFIG, VERSION, IDX,

  // Clinical tools
  VOCABULARY, buildWordSet, getColumnVocabulary,
  ensureVocabularySize,
  CONFUSION_PAIRS, getConfusionCost, ocrWeightedDistance, fixDigraphs, setAdaptiveProfile,
  LAB_RANGES, checkLabRange, checkMedicationAmbiguity, checkDosePlausibility, runSafetyChecks,
  correctWord, correctLine, correctTableRow, assessConfidence,

  // Learning engine
  ShifuLearningEngine, AdaptiveConfusionProfile, WardVocabulary, ContextChains,

  // Corpus & trained model
  MEDICAL_CORPUS, seedEngine, generateWardSentences, fullSeed,
  loadTrainedModel, extractConfusionKnowledge, feedTrainedModelToEngine,

  // Pipeline & persistence
  ShifuPipeline,
  ShifuPersistence, withAutoSave,

  // Clinical weights
  CLINICAL_WEIGHTS, getColumnWeight, getWordWeight, getLearningRate, requiresSafetyReview,

  // Feedback, ingestion, metrics
  FeedbackLoop,
  DocumentIngestor,
  MetricsTracker,

  // Structural invariance
  extractRoles, compareRoles, structuralInvariance,

  // Numeric canonicalization
  normalizeDigits, normalizeDose, normalizeLineNumeric,

  // Utilities
  ocrDist, levDist, cos, cosVec,
};
