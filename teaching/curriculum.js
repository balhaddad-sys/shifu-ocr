// Shifu Curriculum Engine
// Manages progressive learning across domains with spaced repetition,
// difficulty ramping, and adaptive review scheduling.
//
// A curriculum is a structured sequence of teaching phases:
//   1. Foundation — core vocabulary and common patterns
//   2. Specialization — domain-specific terms and confusions
//   3. Transfer — cross-domain knowledge application
//   4. Mastery — edge cases, rare confusions, adversarial examples
//
// Each phase has milestones. Calibration data drives promotion decisions.

const { DOMAINS } = require('./domains');

// ─── Curriculum Phases ──────────────────────────────────────────────────

const PHASES = {
  foundation: {
    id: 'foundation',
    label: 'Foundation',
    order: 0,
    description: 'Core vocabulary, common OCR confusions, basic structure',
    promotionThreshold: 0.70, // 70% accuracy to advance
    minSentences: 100,
    focus: ['vocabulary', 'commonConfusions', 'structure'],
  },
  specialization: {
    id: 'specialization',
    label: 'Specialization',
    order: 1,
    description: 'Domain-specific terminology, field-specific confusions',
    promotionThreshold: 0.75,
    minSentences: 200,
    focus: ['domainVocabulary', 'domainConfusions', 'contextChains'],
  },
  transfer: {
    id: 'transfer',
    label: 'Transfer',
    order: 2,
    description: 'Cross-domain patterns, universal confusions, transfer learning',
    promotionThreshold: 0.80,
    minSentences: 150,
    focus: ['crossDomain', 'universalConfusions', 'blending'],
  },
  mastery: {
    id: 'mastery',
    label: 'Mastery',
    order: 3,
    description: 'Edge cases, adversarial examples, rare confusions',
    promotionThreshold: 0.90,
    minSentences: 100,
    focus: ['edgeCases', 'adversarial', 'rareConfusions'],
  },
};

// ─── Curriculum Item ────────────────────────────────────────────────────

class CurriculumItem {
  constructor({ text, domain, subdomain, difficulty, type, confusionTarget, expected }) {
    this.text = text;
    this.domain = domain;
    this.subdomain = subdomain || 'general';
    this.difficulty = difficulty || 0;
    this.type = type || 'sentence'; // sentence, pair, adversarial
    this.confusionTarget = confusionTarget || null; // Which confusion this tests
    this.expected = expected || null;
    this.attempts = 0;
    this.lastAttempt = null;
    this.lastCorrect = false;
    this.easeFactor = 2.5; // Spaced repetition ease factor
    this.interval = 1;     // Days until next review
    this.nextReview = null;
  }

  /** Update spaced repetition schedule after an attempt */
  recordAttempt(correct, confidence) {
    this.attempts++;
    this.lastAttempt = Date.now();
    this.lastCorrect = correct;

    // SM-2 algorithm adaptation
    const quality = correct ? (confidence > 0.8 ? 5 : confidence > 0.5 ? 4 : 3) : (confidence > 0.3 ? 2 : 1);

    if (quality >= 3) {
      // Correct — increase interval
      if (this.attempts === 1) this.interval = 1;
      else if (this.attempts === 2) this.interval = 3;
      else this.interval = Math.round(this.interval * this.easeFactor);
    } else {
      // Incorrect — reset interval
      this.interval = 1;
    }

    // Update ease factor
    this.easeFactor = Math.max(1.3,
      this.easeFactor + 0.1 - (5 - quality) * (0.08 + (5 - quality) * 0.02));

    this.nextReview = Date.now() + this.interval * 24 * 60 * 60 * 1000;
  }

  /** Is this item due for review? */
  isDue() {
    if (!this.nextReview) return true;
    return Date.now() >= this.nextReview;
  }
}

// ─── Domain Curriculum ──────────────────────────────────────────────────

class DomainCurriculum {
  constructor(domainId) {
    this.domainId = domainId;
    this.domain = DOMAINS[domainId] || null;
    this.phase = 'foundation';
    this.items = [];
    this.completedItems = [];
    this.phaseProgress = {
      foundation: { sentences: 0, correct: 0, total: 0 },
      specialization: { sentences: 0, correct: 0, total: 0 },
      transfer: { sentences: 0, correct: 0, total: 0 },
      mastery: { sentences: 0, correct: 0, total: 0 },
    };
    this.promotionHistory = [];
  }

  /** Get current phase config */
  getCurrentPhase() {
    return PHASES[this.phase];
  }

  /** Check if ready for phase promotion */
  canPromote() {
    const phase = PHASES[this.phase];
    const progress = this.phaseProgress[this.phase];
    if (progress.total < phase.minSentences) return false;
    const accuracy = progress.total > 0 ? progress.correct / progress.total : 0;
    return accuracy >= phase.promotionThreshold;
  }

  /** Promote to the next phase */
  promote() {
    if (!this.canPromote()) return false;
    const phaseOrder = ['foundation', 'specialization', 'transfer', 'mastery'];
    const currentIdx = phaseOrder.indexOf(this.phase);
    if (currentIdx >= phaseOrder.length - 1) return false;

    const oldPhase = this.phase;
    this.phase = phaseOrder[currentIdx + 1];
    this.promotionHistory.push({
      from: oldPhase,
      to: this.phase,
      timestamp: Date.now(),
      accuracy: this.phaseProgress[oldPhase].correct / Math.max(this.phaseProgress[oldPhase].total, 1),
    });
    return true;
  }

  /** Record a teaching result for this curriculum */
  recordResult(correct, confidence = 0.5) {
    const progress = this.phaseProgress[this.phase];
    progress.total++;
    progress.sentences++;
    if (correct) progress.correct++;
  }

  /** Add a curriculum item */
  addItem(item) {
    if (!(item instanceof CurriculumItem)) {
      item = new CurriculumItem(item);
    }
    this.items.push(item);
    return item;
  }

  /** Get items due for review (spaced repetition) */
  getDueItems(limit = 20) {
    return this.items
      .filter(item => item.isDue())
      .sort((a, b) => (a.nextReview || 0) - (b.nextReview || 0))
      .slice(0, limit);
  }

  /** Get next batch of items for current phase */
  getNextBatch(batchSize = 50) {
    const phase = PHASES[this.phase];
    const dueItems = this.getDueItems(Math.floor(batchSize / 3));

    // Mix: 1/3 review, 2/3 new
    const newItems = this.items
      .filter(item => item.attempts === 0 && item.difficulty <= (phase.order + 1) * 0.25)
      .slice(0, batchSize - dueItems.length);

    return [...dueItems, ...newItems];
  }

  /** Get progress summary */
  summary() {
    return {
      domain: this.domainId,
      phase: this.phase,
      phaseLabel: PHASES[this.phase].label,
      canPromote: this.canPromote(),
      progress: this.phaseProgress,
      totalItems: this.items.length,
      dueForReview: this.items.filter(i => i.isDue()).length,
      promotions: this.promotionHistory.length,
    };
  }

  /** Serialize for persistence */
  serialize() {
    return {
      domainId: this.domainId,
      phase: this.phase,
      phaseProgress: this.phaseProgress,
      promotionHistory: this.promotionHistory,
      items: this.items.map(item => ({
        text: item.text,
        domain: item.domain,
        subdomain: item.subdomain,
        difficulty: item.difficulty,
        type: item.type,
        confusionTarget: item.confusionTarget,
        expected: item.expected,
        attempts: item.attempts,
        lastAttempt: item.lastAttempt,
        lastCorrect: item.lastCorrect,
        easeFactor: item.easeFactor,
        interval: item.interval,
        nextReview: item.nextReview,
      })),
    };
  }

  /** Restore from saved state */
  static restore(state) {
    const curriculum = new DomainCurriculum(state.domainId);
    curriculum.phase = state.phase || 'foundation';
    curriculum.phaseProgress = state.phaseProgress || curriculum.phaseProgress;
    curriculum.promotionHistory = state.promotionHistory || [];
    if (state.items) {
      curriculum.items = state.items.map(i => Object.assign(new CurriculumItem(i), i));
    }
    return curriculum;
  }
}

// ─── Curriculum Manager ─────────────────────────────────────────────────

class CurriculumManager {
  constructor() {
    this.curricula = {};
  }

  /** Get or create curriculum for a domain */
  getCurriculum(domainId) {
    if (!this.curricula[domainId]) {
      this.curricula[domainId] = new DomainCurriculum(domainId);
    }
    return this.curricula[domainId];
  }

  /** Generate foundation curriculum items from seed data */
  generateFoundation(domainId, sentences) {
    const curriculum = this.getCurriculum(domainId);
    const domain = DOMAINS[domainId];

    for (let i = 0; i < sentences.length; i++) {
      const difficulty = i / sentences.length; // Linear difficulty ramp
      curriculum.addItem({
        text: sentences[i],
        domain: domainId,
        difficulty,
        type: 'sentence',
      });
    }

    // Add confusion-targeted items from domain confusion profile
    if (domain && domain.confusionProfile) {
      for (const [pair, cost] of Object.entries(domain.confusionProfile)) {
        const [a, b] = pair.split(',');
        // Generate adversarial items that test specific confusions
        curriculum.addItem({
          text: `[confusion_test] ${a} vs ${b}`,
          domain: domainId,
          difficulty: 1 - cost, // Lower cost = harder (more easily confused)
          type: 'adversarial',
          confusionTarget: pair,
        });
      }
    }

    return curriculum;
  }

  /** Generate cross-domain transfer items */
  generateTransferItems(fromDomain, toDomain, sentences) {
    const curriculum = this.getCurriculum(toDomain);
    const transferWeight = DOMAINS[fromDomain]?.transferWeights?.[toDomain] || 0.3;

    for (const sentence of sentences) {
      curriculum.addItem({
        text: sentence,
        domain: toDomain,
        subdomain: `transfer_from_${fromDomain}`,
        difficulty: 0.5 + (1 - transferWeight) * 0.3, // Weaker transfer = harder
        type: 'sentence',
      });
    }

    return curriculum;
  }

  /** Get overall progress across all domains */
  overallProgress() {
    const result = {};
    for (const [domainId, curriculum] of Object.entries(this.curricula)) {
      result[domainId] = curriculum.summary();
    }
    return result;
  }

  /** Get recommended next actions across all curricula */
  getRecommendations() {
    const recs = [];

    for (const [domainId, curriculum] of Object.entries(this.curricula)) {
      // Check for promotion opportunities
      if (curriculum.canPromote()) {
        recs.push({
          priority: 3,
          type: 'promote',
          domain: domainId,
          message: `"${domainId}" is ready for promotion to ${
            ['specialization', 'transfer', 'mastery'][
              ['foundation', 'specialization', 'transfer'].indexOf(curriculum.phase)
            ] || 'next phase'
          }`,
        });
      }

      // Check for review items
      const dueCount = curriculum.items.filter(i => i.isDue()).length;
      if (dueCount > 10) {
        recs.push({
          priority: 2,
          type: 'review',
          domain: domainId,
          count: dueCount,
          message: `${dueCount} items due for review in "${domainId}"`,
        });
      }

      // Check for stalled progress
      const progress = curriculum.phaseProgress[curriculum.phase];
      if (progress.total > 50 && progress.correct / progress.total < 0.5) {
        recs.push({
          priority: 4,
          type: 'struggling',
          domain: domainId,
          accuracy: progress.correct / progress.total,
          message: `"${domainId}" is struggling at ${(progress.correct / progress.total * 100).toFixed(0)}% — consider more foundation work`,
        });
      }
    }

    return recs.sort((a, b) => b.priority - a.priority);
  }

  /** Serialize all curricula */
  serialize() {
    const data = {};
    for (const [id, c] of Object.entries(this.curricula)) {
      data[id] = c.serialize();
    }
    return data;
  }

  /** Restore all curricula */
  restore(state) {
    for (const [id, data] of Object.entries(state)) {
      this.curricula[id] = DomainCurriculum.restore(data);
    }
  }
}

module.exports = { PHASES, CurriculumItem, DomainCurriculum, CurriculumManager };
