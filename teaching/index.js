// SHIFU TEACHING MODEL v1.0.0
//
// Cross-domain teaching engine with calibration persistence.
//
// Architecture:
//   Domains          — 12 knowledge domains with transfer weights
//   Calibrations     — Every correction saved for future learning sessions
//   Teaching Model   — Cross-domain transfer, progressive curriculum
//   Curriculum       — Spaced repetition, difficulty ramping, phase promotions
//   Evaluation       — CER, WER, ECE, F1, temporal stability tracking
//   Seeds            — Kaggle + HuggingFace data pipeline (Python)
//
// Usage:
//   const { createTeacher } = require('./teaching');
//   const teacher = createTeacher(shifu);
//   teacher.activateDomain('medical');
//   teacher.teachSentence('Patient admitted with acute MI.');
//   teacher.teachCorrection('m0rphine', 'morphine', 'medical');
//   console.log(teacher.getProgress());

const fs = require('fs');
const path = require('path');
const { ShifuTeachingModel, TEACHING_CONFIG } = require('./model');
const { DomainRegistry, DOMAINS } = require('./domains');
const { CalibrationStore, CalibrationPoint } = require('./calibration');
const { CurriculumManager, PHASES, CurriculumItem } = require('./curriculum');
const { CrossDomainEvaluator, charErrorRate, wordErrorRate } = require('./evaluation');

const TEACHING_STATE_DIR = path.join(__dirname, '..', '.teaching');
const TEACHING_STATE_FILE = 'teaching_state.json';

// ─── Factory ────────────────────────────────────────────────────────────

/**
 * Create a fully initialized teaching model.
 * Integrates with an existing Shifu system.
 *
 * @param {object} shifu - A Shifu system (from createShifu/restoreShifu)
 * @param {object} opts
 * @param {string} opts.stateDir - Where to persist teaching state
 * @param {boolean} opts.restore - Try to restore previous teaching state
 * @param {boolean} opts.seedBuiltin - Pre-seed with built-in domain corpora
 * @returns {ShifuTeacher}
 */
function createTeacher(shifu, opts = {}) {
  const stateDir = opts.stateDir || TEACHING_STATE_DIR;
  const engine = shifu.engine || shifu;
  const learning = shifu.learning || null;

  // Create the teaching model
  const model = new ShifuTeachingModel(engine, learning, {
    calibrationDir: path.join(stateDir, 'calibrations'),
  });

  // Create curriculum manager
  const curriculum = new CurriculumManager();

  // Create evaluator
  const evaluator = new CrossDomainEvaluator(model.calibrations);

  // Build the teacher wrapper
  const teacher = new ShifuTeacher(model, curriculum, evaluator, stateDir);

  // Restore previous state if available
  if (opts.restore !== false) {
    teacher.restore();
  }

  // Seed with built-in corpora
  if (opts.seedBuiltin !== false) {
    teacher.seedBuiltin();
  }

  return teacher;
}

/**
 * Restore a teaching model from saved state.
 */
function restoreTeacher(shifu, stateDir) {
  return createTeacher(shifu, { stateDir, restore: true, seedBuiltin: false });
}

// ─── Teacher Wrapper ────────────────────────────────────────────────────

class ShifuTeacher {
  constructor(model, curriculum, evaluator, stateDir) {
    this.model = model;
    this.curriculum = curriculum;
    this.evaluator = evaluator;
    this.stateDir = stateDir;
    this._seeded = false;
  }

  // ─── Domain Management ────────────────────────────────────────────

  /** Activate a domain */
  activateDomain(domainId) {
    const domain = this.model.activateDomain(domainId);
    this.curriculum.getCurriculum(domainId);
    return domain;
  }

  /** Auto-detect domains from text */
  autoDetect(text) {
    return this.model.autoDetect(text);
  }

  /** Register a custom domain */
  registerDomain(def) {
    return this.model.registerDomain(def);
  }

  /** List all domains */
  listDomains() {
    return this.model.domains.list();
  }

  /** Get active domains */
  getActiveDomains() {
    return this.model.domains.getActive();
  }

  // ─── Teaching ─────────────────────────────────────────────────────

  /** Teach a single sentence */
  teachSentence(sentence, domain, options) {
    const result = this.model.teachSentence(sentence, domain, options);
    if (domain) {
      this.curriculum.getCurriculum(domain).recordResult(true, 0.5);
    }
    return result;
  }

  /** Teach a corpus of sentences */
  teachCorpus(sentences, domain, options) {
    return this.model.teachCorpus(sentences, domain, options);
  }

  /** Teach from a human correction */
  teachCorrection(ocrText, correctedText, domain, options) {
    return this.model.teachCorrection(ocrText, correctedText, domain, options);
  }

  /** Teach from seed data (output of Python pipeline) */
  teachFromSeed(seedData) {
    return this.model.teachFromSeed(seedData);
  }

  /** Load and teach from a seed file */
  teachFromSeedFile(filePath) {
    const data = JSON.parse(fs.readFileSync(filePath, 'utf8'));
    const results = [];

    if (data.domains) {
      // Unified teaching data format
      for (const [domain, domainData] of Object.entries(data.domains)) {
        const r = this.teachFromSeed({
          domain,
          source: 'file',
          sentences: domainData.sentences || [],
          pairs: domainData.pairs || [],
        });
        results.push(r);
      }
    } else {
      // Single domain seed format
      const r = this.teachFromSeed({
        domain: data.domain || 'general',
        source: data.source || 'file',
        sentences: data.sentences || [],
        pairs: data.pairs || [],
      });
      results.push(r);
    }

    return results;
  }

  // ─── Teaching Sessions ────────────────────────────────────────────

  /** Start a structured teaching session */
  startSession(domainId, subdomain) {
    return this.model.startSession(domainId, subdomain);
  }

  /** End the current teaching session */
  endSession() {
    const summary = this.model.endSession();
    this.save();
    return summary;
  }

  // ─── Curriculum ───────────────────────────────────────────────────

  /** Get the curriculum for a domain */
  getCurriculum(domainId) {
    return this.curriculum.getCurriculum(domainId);
  }

  /** Generate curriculum from sentences */
  generateCurriculum(domainId, sentences) {
    return this.curriculum.generateFoundation(domainId, sentences);
  }

  /** Get next batch of curriculum items */
  getNextBatch(domainId, batchSize) {
    return this.curriculum.getCurriculum(domainId).getNextBatch(batchSize);
  }

  /** Try to promote a domain to the next phase */
  tryPromote(domainId) {
    return this.curriculum.getCurriculum(domainId).promote();
  }

  // ─── Evaluation ───────────────────────────────────────────────────

  /** Evaluate a domain */
  evaluateDomain(domainId) {
    return this.evaluator.evaluateDomain(domainId);
  }

  /** Evaluate all domains */
  evaluateAll() {
    return this.evaluator.evaluateAll();
  }

  /** Generate comprehensive report */
  generateReport() {
    return this.evaluator.generateReport();
  }

  // ─── Progress & Recommendations ───────────────────────────────────

  /** Get teaching progress */
  getProgress() {
    return {
      ...this.model.getProgress(),
      curriculum: this.curriculum.overallProgress(),
    };
  }

  /** Get recommendations */
  getRecommendations() {
    const modelRecs = this.model.getRecommendations();
    const currRecs = this.curriculum.getRecommendations();
    return [...modelRecs, ...currRecs].sort((a, b) => (b.priority || 0) - (a.priority || 0));
  }

  // ─── Built-in Seeding ─────────────────────────────────────────────

  seedBuiltin() {
    if (this._seeded) return;

    // Seed with domain-specific corpora from the built-in seed registry
    const builtinSeeds = {
      medical: [
        "Patient admitted with acute myocardial infarction requiring emergent catheterization.",
        "Nurse administered morphine sulfate 4mg IV for chest pain management.",
        "Laboratory results show elevated troponin I at 2.4 ng/mL.",
        "Physician ordered continuous cardiac monitoring and serial ECGs.",
        "Blood pressure 140/90 mmHg heart rate 98 bpm respiratory rate 22.",
        "Chest X-ray reveals bilateral pulmonary infiltrates consistent with pneumonia.",
        "Patient started on piperacillin-tazobactam 4.5g IV every 6 hours.",
        "Discharge summary completed with follow-up in cardiology clinic.",
      ],
      legal: [
        "The plaintiff alleges breach of contract under Section 2-207 of the UCC.",
        "Defendant filed a motion to dismiss for failure to state a claim.",
        "The court granted summary judgment in favor of the respondent.",
        "Counsel submitted memorandum of law in support of preliminary injunction.",
        "The arbitration clause in paragraph 14.3 governs dispute resolution.",
      ],
      financial: [
        "Quarterly revenue increased 12% year-over-year to $4.2 billion.",
        "The Federal Reserve raised interest rates by 25 basis points.",
        "Net income attributable to shareholders was $1.8 million.",
        "Earnings per share came in at $2.15 beating consensus estimates.",
        "The portfolio allocation shifted toward fixed-income securities.",
      ],
      scientific: [
        "The experiment demonstrated a statistically significant correlation with p value below 0.01.",
        "Spectroscopic analysis revealed absorption peaks at 254nm and 380nm.",
        "The catalyst increased reaction yield from 45% to 92% under mild conditions.",
        "Genome sequencing identified three novel single nucleotide polymorphisms.",
      ],
      engineering: [
        "The microcontroller operates at 3.3V with a clock frequency of 168MHz.",
        "Tensile strength of the alloy exceeded 450 MPa at room temperature.",
        "Load testing revealed maximum deflection of 2.3mm under 500N force.",
        "The firmware update resolved the I2C communication timeout issue.",
      ],
      general: [
        "The conference attracted over three thousand attendees from forty countries.",
        "Chapter twelve explores the historical context of the industrial revolution.",
        "The editorial board reviewed manuscripts from leading international scholars.",
        "Public transportation services were temporarily suspended during the storm.",
      ],
    };

    for (const [domain, sentences] of Object.entries(builtinSeeds)) {
      this.model.activateDomain(domain);
      this.model.teachCorpus(sentences, domain);
    }

    this._seeded = true;
  }

  // ─── Persistence ──────────────────────────────────────────────────

  save() {
    const dir = this.stateDir;
    if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });

    const state = {
      version: '1.0.0',
      savedAt: new Date().toISOString(),
      model: this.model.serialize(),
      curriculum: this.curriculum.serialize(),
      evaluator: this.evaluator.serialize(),
    };

    const tmpPath = path.join(dir, TEACHING_STATE_FILE + '.tmp');
    const finalPath = path.join(dir, TEACHING_STATE_FILE);
    fs.writeFileSync(tmpPath, JSON.stringify(state));
    fs.renameSync(tmpPath, finalPath);

    // Also save calibrations
    this.model.calibrations.save();
  }

  restore() {
    const statePath = path.join(this.stateDir, TEACHING_STATE_FILE);
    if (!fs.existsSync(statePath)) return false;

    try {
      const state = JSON.parse(fs.readFileSync(statePath, 'utf8'));

      if (state.model) this.model.restore(state.model);
      if (state.curriculum) this.curriculum.restore(state.curriculum);
      if (state.evaluator) this.evaluator.restore(state.evaluator);

      return true;
    } catch (e) {
      console.warn(`Failed to restore teaching state: ${e.message}`);
      return false;
    }
  }
}

// ─── Exports ────────────────────────────────────────────────────────────

module.exports = {
  // Factory
  createTeacher,
  restoreTeacher,

  // Classes
  ShifuTeacher,
  ShifuTeachingModel,
  DomainRegistry,
  CalibrationStore,
  CalibrationPoint,
  CurriculumManager,
  CrossDomainEvaluator,

  // Constants
  DOMAINS,
  PHASES,
  TEACHING_CONFIG,

  // Utilities
  charErrorRate,
  wordErrorRate,
};
