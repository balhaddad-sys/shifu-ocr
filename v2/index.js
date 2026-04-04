/**
 * Shifu Clinical OCR Post-Processing
 * ====================================
 * 
 * Drop-in post-processing layer for ANY OCR engine.
 * Takes raw OCR text → returns corrected text with safety flags.
 * 
 * Usage:
 * 
 *   import { correctLine, correctTableRow, assessConfidence } from './shifu';
 * 
 *   // Single line correction
 *   const result = correctLine("Potasium 45", { columnType: "diagnosis" });
 *   // result.output = "Potassium 45"
 *   // result.safetyFlags = [{ status: 'OUT_OF_RANGE', message: '...suggest 4.5...' }]
 * 
 *   // Full table row
 *   const row = correctTableRow({
 *     Room: "12-3",
 *     Patient: "Abdulah",
 *     Diagnosis: "Chst infecfion",
 *     Doctor: "Bader",
 *   });
 *   // row.corrected.Patient.output = "Abdullah"
 *   // row.corrected.Diagnosis.output = "Chest infection"
 * 
 *   // Quick confidence check
 *   assessConfidence(result); // 'accept' | 'verify' | 'reject'
 * 
 * Modules:
 *   shifuConfusionModel.js  — OCR character confusion costs
 *   shifuVocabulary.js      — Clinical vocabulary (ward + neuro + meds + labs)
 *   shifuSafetyFlags.js     — Lab ranges, medication ambiguity, dose checks
 *   shifuClinicalCorrector.js — Main correction engine
 */

export {
  correctWord,
  correctLine,
  correctTableRow,
  assessConfidence,
} from './shifuClinicalCorrector.js';

export {
  ocrWeightedDistance,
  getConfusionCost,
  fixDigraphs,
  CONFUSION_PAIRS,
  DIGRAPH_CONFUSIONS,
} from './shifuConfusionModel.js';

export {
  VOCABULARY,
  buildWordSet,
  getColumnVocabulary,
} from './shifuVocabulary.js';

export {
  LAB_RANGES,
  checkLabRange,
  checkMedicationAmbiguity,
  checkDosePlausibility,
  runSafetyChecks,
} from './shifuSafetyFlags.js';

export {
  ShifuLearningEngine,
  AdaptiveConfusionProfile,
  WardVocabulary,
  ContextChains,
} from './shifuLearningLoop.js';

export {
  initShifu,
  saveShifu,
  shifuCorrect,
  shifuLearn,
} from './shifuFirebase.js';
