/**
 * Shifu Firebase Integration
 * 
 * Wires the learning loop into MedTriage's existing Firebase stack.
 * 
 * Firestore structure:
 *   shifu/
 *     state/          → { confusion, vocabulary, context, correctionCount }
 *     corrections/    → individual correction events (audit trail)
 *     stats/          → daily aggregated stats
 * 
 * The state document is the serialized learning engine.
 * Load on app start, save after every correction batch.
 */

import { ShifuLearningEngine } from './shifuLearningLoop.js';
import { VOCABULARY } from './shifuVocabulary.js';

/**
 * Initialize or restore the Shifu engine from Firebase.
 * Call this once on app startup.
 * 
 * @param {object} db - Firestore instance
 * @returns {ShifuLearningEngine}
 */
export async function initShifu(db) {
  try {
    const doc = await db.collection('shifu').doc('state').get();

    if (doc.exists) {
      const data = doc.data();
      console.log(`[Shifu] Restored: ${data.correctionCount || 0} corrections, ` +
                  `${Object.keys(data.vocabulary?.frequency || {}).length} learned words`);
      return ShifuLearningEngine.fromJSON(data, VOCABULARY);
    }
  } catch (err) {
    console.warn('[Shifu] Could not restore state, starting fresh:', err.message);
  }

  console.log('[Shifu] Starting fresh engine');
  return new ShifuLearningEngine(VOCABULARY);
}

/**
 * Save the engine state to Firebase.
 * Call after correction batches (not after every single correction —
 * batch to avoid excessive writes).
 * 
 * @param {object} db - Firestore instance
 * @param {ShifuLearningEngine} engine
 */
export async function saveShifu(db, engine) {
  try {
    await db.collection('shifu').doc('state').set(engine.toJSON());
  } catch (err) {
    console.error('[Shifu] Failed to save state:', err.message);
  }
}

/**
 * Record a correction event for audit trail.
 * Every nurse correction is logged individually so you can
 * review what the system learned and why.
 * 
 * @param {object} db - Firestore instance
 * @param {object} event - { ocrRow, confirmedRow, ward, user, timestamp }
 */
export async function logCorrection(db, event) {
  try {
    await db.collection('shifu').doc('corrections').collection('events').add({
      ...event,
      timestamp: event.timestamp || new Date().toISOString(),
    });
  } catch (err) {
    console.error('[Shifu] Failed to log correction:', err.message);
  }
}

/**
 * Full correction + learning flow for MedTriage.
 * 
 * Call flow:
 *   1. PaddleOCR reads image → raw text
 *   2. ocrMedCorrector.js (Layer 1) → partially corrected
 *   3. shifuCorrectAndLearn() (Layer 2) → final output + flags
 *   4. Nurse reviews and confirms → learn()
 * 
 * @param {ShifuLearningEngine} engine
 * @param {object} ocrRow - { Room: "12-3", Patient: "Abdulah", ... }
 * @param {object} context - { ward: "20" } or any known fields
 * @returns {object} corrected row with flags
 */
export function shifuCorrect(engine, ocrRow, context = {}) {
  return engine.correctRow(ocrRow, context);
}

/**
 * Call when nurse confirms or edits the OCR result.
 * This is where learning happens.
 * 
 * @param {ShifuLearningEngine} engine
 * @param {object} db - Firestore instance
 * @param {object} ocrRow - Original OCR output
 * @param {object} confirmedRow - What the nurse confirmed/corrected
 */
export async function shifuLearn(engine, db, ocrRow, confirmedRow) {
  // Learn
  engine.learn(ocrRow, confirmedRow);

  // Save state (debounce in production — save every 10 corrections, not every 1)
  if (engine.correctionCount % 10 === 0) {
    await saveShifu(db, engine);
  }

  // Log for audit
  await logCorrection(db, { ocrRow, confirmedRow });
}

/**
 * Integration example for MedTriage patient card flow:
 * 
 * // On app startup:
 * const engine = await initShifu(db);
 * 
 * // When OCR reads a ward sheet:
 * const ocrRow = { Room: "12-3", Patient: "Abdulah", Diagnosis: "Chst infecfion", Doctor: "Bader" };
 * const result = shifuCorrect(engine, ocrRow, { ward: "20" });
 * // result.corrected.Patient.output = "Abdullah"
 * // result.corrected.Diagnosis.output = "Chest infection"
 * // Display to nurse for verification
 * 
 * // When nurse confirms (or edits and confirms):
 * const confirmedRow = { Room: "12-3", Patient: "Abdullah", Diagnosis: "Chest infection", Doctor: "Bader" };
 * await shifuLearn(engine, db, ocrRow, confirmedRow);
 * // Engine is now slightly smarter for next time
 * 
 * // On app close or periodically:
 * await saveShifu(db, engine);
 */
