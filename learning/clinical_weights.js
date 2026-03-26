// Clinical-Weighted Learning
//
// Not all errors are equal. Getting a diagnosis wrong is worse than
// misspelling a name. Getting a medication wrong can kill.
//
// This module defines clinical severity weights that shape how aggressively
// the system learns from different types of corrections.
//
// The weights affect:
//   1. How much confusion cost drops for a given correction (learn faster from dangerous errors)
//   2. How many observations it takes before the system trusts a correction
//   3. Whether a correction triggers safety re-evaluation

const CLINICAL_WEIGHTS = {
  // Column-level weights: how critical is accuracy in this field?
  columns: {
    // Medications: errors can be lethal. Maximum learning weight.
    medication:  { weight: 3.0, minConfidence: 0.85, safetyOverride: true },
    med:         { weight: 3.0, minConfidence: 0.85, safetyOverride: true },
    drug:        { weight: 3.0, minConfidence: 0.85, safetyOverride: true },
    rx:          { weight: 3.0, minConfidence: 0.85, safetyOverride: true },
    dose:        { weight: 3.0, minConfidence: 0.85, safetyOverride: true },

    // Diagnoses: errors delay treatment. High weight.
    diagnosis:   { weight: 2.5, minConfidence: 0.7, safetyOverride: false },
    dx:          { weight: 2.5, minConfidence: 0.7, safetyOverride: false },

    // Lab values: wrong numbers change treatment. High weight.
    lab:         { weight: 2.5, minConfidence: 0.8, safetyOverride: true },
    result:      { weight: 2.5, minConfidence: 0.8, safetyOverride: true },

    // Status/triage: affects evacuation priority. Medium-high weight.
    status:      { weight: 2.0, minConfidence: 0.7, safetyOverride: false },
    triage:      { weight: 2.0, minConfidence: 0.7, safetyOverride: false },

    // Doctor names: important but not life-threatening. Medium weight.
    doctor:      { weight: 1.5, minConfidence: 0.6, safetyOverride: false },
    consultant:  { weight: 1.5, minConfidence: 0.6, safetyOverride: false },

    // Patient names: misidentification is serious. Medium weight.
    patient:     { weight: 1.5, minConfidence: 0.6, safetyOverride: false },
    name:        { weight: 1.5, minConfidence: 0.6, safetyOverride: false },

    // Room/bed: logistical, not clinical. Lower weight.
    room:        { weight: 1.0, minConfidence: 0.5, safetyOverride: false },
    bed:         { weight: 1.0, minConfidence: 0.5, safetyOverride: false },
    ward:        { weight: 1.0, minConfidence: 0.5, safetyOverride: false },
  },

  // Word-category weights: how critical is this type of word?
  categories: {
    medication:    3.0,
    lab_value:     2.5,
    diagnosis:     2.5,
    examination:   1.5,
    name:          1.5,
    general:       1.0,
  },

  // Default weight for unknown columns/categories
  defaultWeight: 1.0,
  defaultMinConfidence: 0.5,
};

/**
 * Get the clinical weight for a column type.
 */
function getColumnWeight(columnType) {
  if (!columnType) return { weight: CLINICAL_WEIGHTS.defaultWeight, minConfidence: CLINICAL_WEIGHTS.defaultMinConfidence, safetyOverride: false };
  const key = columnType.toLowerCase().replace(/\s+/g, '');
  for (const [pattern, config] of Object.entries(CLINICAL_WEIGHTS.columns)) {
    if (key.includes(pattern)) return config;
  }
  return { weight: CLINICAL_WEIGHTS.defaultWeight, minConfidence: CLINICAL_WEIGHTS.defaultMinConfidence, safetyOverride: false };
}

/**
 * Get the clinical weight for a word based on what vocabulary category it belongs to.
 */
function getWordWeight(word, vocabulary) {
  const lower = word.toLowerCase();
  if (!vocabulary) return CLINICAL_WEIGHTS.defaultWeight;

  // For multi-token strings, check each token individually as well
  const tokens = lower.split(/\s+/).filter(t => t.length > 0);

  // Check medication list first — highest priority
  if (vocabulary.medications && vocabulary.medications.some(m => {
    const ml = m.toLowerCase();
    return ml === lower || tokens.some(t => t === ml);
  })) {
    return CLINICAL_WEIGHTS.categories.medication;
  }
  if (vocabulary.labs && vocabulary.labs.some(l => {
    const ll = l.toLowerCase();
    return ll === lower || tokens.some(t => t === ll);
  })) {
    return CLINICAL_WEIGHTS.categories.lab_value;
  }
  if (vocabulary.diagnoses && vocabulary.diagnoses.some(d => {
    const dl = d.toLowerCase();
    return dl.includes(lower) || tokens.some(t => dl.includes(t) || t.includes(dl));
  })) {
    return CLINICAL_WEIGHTS.categories.diagnosis;
  }
  if (vocabulary.examination && vocabulary.examination.some(e => {
    const el = e.toLowerCase();
    return el === lower || tokens.some(t => t === el);
  })) {
    return CLINICAL_WEIGHTS.categories.examination;
  }
  if (vocabulary.names && vocabulary.names.some(n => {
    const nl = n.toLowerCase();
    return nl === lower || tokens.some(t => t === nl);
  })) {
    return CLINICAL_WEIGHTS.categories.name;
  }
  return CLINICAL_WEIGHTS.defaultWeight;
}

/**
 * Compute a weighted learning rate for a correction event.
 * Higher weight = system learns more aggressively from this correction.
 *
 * @param {string} columnType - The column this word came from
 * @param {string} word - The word being corrected
 * @param {object} vocabulary - The VOCABULARY object
 * @returns {number} Learning rate multiplier (1.0 = normal, 3.0 = medication-critical)
 */
function getLearningRate(columnType, word, vocabulary) {
  const colWeight = getColumnWeight(columnType).weight;
  const wordWeight = getWordWeight(word, vocabulary);
  // Take the max — if a medication appears in a "notes" column, it's still critical
  return Math.max(colWeight, wordWeight);
}

/**
 * Should this correction trigger a safety re-evaluation?
 * True for medications, lab values, and other safety-critical fields.
 */
function requiresSafetyReview(columnType, word, vocabulary) {
  const col = getColumnWeight(columnType);
  if (col.safetyOverride) return true;
  // Even outside safety columns, medication words always need review
  if (vocabulary && vocabulary.medications) {
    const lower = word.toLowerCase();
    const tokens = lower.split(/\s+/).filter(t => t.length > 0);
    if (vocabulary.medications.some(m => {
      const ml = m.toLowerCase();
      return ml === lower || tokens.some(t => t === ml);
    })) return true;
  }
  return false;
}

module.exports = {
  CLINICAL_WEIGHTS,
  getColumnWeight,
  getWordWeight,
  getLearningRate,
  requiresSafetyReview,
};
