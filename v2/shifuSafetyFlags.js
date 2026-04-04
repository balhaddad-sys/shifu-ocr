/**
 * Shifu Safety Flags
 * 
 * Clinical validation layer. Catches dangerous OCR errors:
 * - Lab values outside expected ranges (missing decimal points)
 * - Ambiguity between two medications
 * - Implausible numerical values
 * 
 * DESIGN PRINCIPLE: Never silently correct. Always flag.
 * Low confidence = loud warning, not quiet guess.
 */

import { ocrWeightedDistance } from './shifuConfusionModel.js';
import { VOCABULARY } from './shifuVocabulary.js';

// === LAB VALUE RANGES ===
export const LAB_RANGES = {
  hba1c:       { min: 4.0,  max: 15.0,  unit: '%' },
  glucose:     { min: 1.0,  max: 50.0,  unit: 'mmol/L' },
  sodium:      { min: 100,  max: 180,   unit: 'mmol/L' },
  potassium:   { min: 1.5,  max: 9.0,   unit: 'mmol/L' },
  chloride:    { min: 70,   max: 130,   unit: 'mmol/L' },
  bicarbonate: { min: 5,    max: 45,    unit: 'mmol/L' },
  urea:        { min: 0.5,  max: 80,    unit: 'mmol/L' },
  creatinine:  { min: 20,   max: 2000,  unit: 'μmol/L' },
  hemoglobin:  { min: 3.0,  max: 25.0,  unit: 'g/dL' },
  wbc:         { min: 0.1,  max: 100.0, unit: '×10⁹/L' },
  platelets:   { min: 5,    max: 1500,  unit: '×10⁹/L' },
  inr:         { min: 0.5,  max: 10.0,  unit: '' },
  crp:         { min: 0.0,  max: 500.0, unit: 'mg/L' },
  esr:         { min: 0,    max: 150,   unit: 'mm/hr' },
  tsh:         { min: 0.01, max: 100.0, unit: 'mIU/L' },
  troponin:    { min: 0,    max: 50000, unit: 'ng/L' },
  lactate:     { min: 0.1,  max: 20.0,  unit: 'mmol/L' },
  albumin:     { min: 5,    max: 60,    unit: 'g/L' },
  bilirubin:   { min: 0,    max: 500,   unit: 'μmol/L' },
  alt:         { min: 0,    max: 5000,  unit: 'U/L' },
  ast:         { min: 0,    max: 5000,  unit: 'U/L' },
  alp:         { min: 0,    max: 2000,  unit: 'U/L' },
  calcium:     { min: 1.0,  max: 4.5,   unit: 'mmol/L' },
  magnesium:   { min: 0.3,  max: 3.0,   unit: 'mmol/L' },
  phosphate:   { min: 0.3,  max: 5.0,   unit: 'mmol/L' },
  ph:          { min: 6.8,  max: 7.8,   unit: '' },
};

/**
 * Check a numerical value against expected lab ranges.
 * Returns a safety flag object.
 * 
 * @param {string} labName - The lab test name (from preceding text)
 * @param {string|number} rawValue - The OCR-read value
 * @returns {object} { status, message, alternatives? }
 */
export function checkLabRange(labName, rawValue) {
  const key = labName.toLowerCase().replace(/\s+/g, '');
  const range = LAB_RANGES[key];

  if (!range) {
    return { status: 'unknown_lab', message: `No range defined for "${labName}"` };
  }

  const value = parseFloat(String(rawValue).replace(',', '.'));
  if (isNaN(value)) {
    return { status: 'parse_error', message: `Cannot parse "${rawValue}" as number` };
  }

  // In range
  if (value >= range.min && value <= range.max) {
    return {
      status: 'in_range',
      message: `${labName} ${value} ${range.unit} within [${range.min}-${range.max}]`,
      value,
    };
  }

  // Out of range — check for missing decimal point
  const rawStr = String(rawValue).replace(',', '.');
  const alternatives = [];

  for (let dp = 1; dp < rawStr.length; dp++) {
    if (rawStr[dp] === '.') continue;
    const candidate = parseFloat(rawStr.slice(0, dp) + '.' + rawStr.slice(dp));
    if (!isNaN(candidate) && candidate >= range.min && candidate <= range.max) {
      alternatives.push(candidate);
    }
  }

  if (alternatives.length > 0) {
    return {
      status: 'OUT_OF_RANGE',
      severity: 'warning',
      message: `${labName} ${value} OUTSIDE [${range.min}-${range.max}] ${range.unit}. ` +
               `Possible missing decimal: ${alternatives.join(', ')}. VERIFY.`,
      value,
      alternatives,
    };
  }

  return {
    status: 'OUT_OF_RANGE',
    severity: 'error',
    message: `${labName} ${value} OUTSIDE [${range.min}-${range.max}] ${range.unit}. Verify reading.`,
    value,
  };
}

/**
 * Check if two medication matches are dangerously close.
 * If OCR output is ambiguous between two medications,
 * flag as DANGER requiring human verification.
 * 
 * @param {string} ocrText - The raw OCR text
 * @param {Array<{word: string, distance: number}>} candidates - Top matches
 * @returns {object|null} Safety flag if dangerous, null if safe
 */
export function checkMedicationAmbiguity(ocrText, candidates) {
  if (candidates.length < 2) return null;

  const meds = new Set(VOCABULARY.medications.map(m => m.toLowerCase()));

  const top = candidates[0];
  const second = candidates[1];

  const topIsMed = meds.has(top.word.toLowerCase());
  const secondIsMed = meds.has(second.word.toLowerCase());

  // Both candidates are medications and distances are close
  if (topIsMed && secondIsMed) {
    const margin = second.distance - top.distance;
    if (margin < 1.5) {
      return {
        status: 'MEDICATION_AMBIGUITY',
        severity: 'danger',
        message: `OCR "${ocrText}" ambiguous between "${top.word}" and "${second.word}". ` +
                 `MUST VERIFY — medication confusion is clinically dangerous.`,
        candidates: [top, second],
      };
    }
  }

  return null;
}

/**
 * Validate a dose string against basic plausibility.
 * Catches OCR artifacts like "5000mg" when "500mg" was intended.
 */
export function checkDosePlausibility(medName, doseStr) {
  const match = doseStr.match(/^(\d+(?:\.\d+)?)\s*(mg|g|mcg|ml|units?|iu)?$/i);
  if (!match) return null;

  const value = parseFloat(match[1]);
  const unit = (match[2] || '').toLowerCase();

  // Basic sanity: most oral doses are < 2000mg
  if (unit === 'mg' && value > 5000) {
    return {
      status: 'IMPLAUSIBLE_DOSE',
      severity: 'warning',
      message: `${medName} ${doseStr} seems high. Verify: could be ${value / 10}${unit}?`,
    };
  }

  // Insulin: typical range 1-100 units
  if (medName.toLowerCase().includes('insulin') && value > 200) {
    return {
      status: 'IMPLAUSIBLE_DOSE',
      severity: 'warning',
      message: `Insulin ${value} units seems very high. Verify.`,
    };
  }

  return null;
}

/**
 * Run all safety checks on a sequence of words.
 * Looks for lab values after lab names, medication ambiguities, etc.
 * 
 * @param {Array<{original: string, corrected: string, candidates: Array}>} words
 * @returns {Array<object>} List of safety flags
 */
export function runSafetyChecks(words) {
  const flags = [];
  const labNames = new Set(Object.keys(LAB_RANGES));

  for (let i = 0; i < words.length; i++) {
    const w = words[i];

    // Check medication ambiguity
    if (w.candidates && w.candidates.length >= 2) {
      const medFlag = checkMedicationAmbiguity(w.original, w.candidates);
      if (medFlag) {
        flags.push({ position: i, ...medFlag });
      }
    }

    // Check if this is a number after a lab name
    if (/^[\d.,]+$/.test(w.corrected)) {
      // Look back for a lab name
      for (let j = i - 1; j >= Math.max(0, i - 3); j--) {
        const prevWord = words[j].corrected.toLowerCase().replace(/\s+/g, '');
        if (labNames.has(prevWord)) {
          const rangeFlag = checkLabRange(words[j].corrected, w.corrected);
          if (rangeFlag.status === 'OUT_OF_RANGE') {
            flags.push({ position: i, ...rangeFlag });
          }
          break;
        }
      }
    }

    // Check dose after medication
    if (i > 0 && /\d/.test(w.corrected)) {
      const prevMed = words[i - 1].corrected.toLowerCase();
      const meds = new Set(VOCABULARY.medications.map(m => m.toLowerCase()));
      if (meds.has(prevMed)) {
        const doseFlag = checkDosePlausibility(prevMed, w.corrected);
        if (doseFlag) {
          flags.push({ position: i, ...doseFlag });
        }
      }
    }
  }

  return flags;
}
