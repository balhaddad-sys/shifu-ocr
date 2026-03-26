// Shifu Safety Flags
// Clinical validation layer. Catches dangerous OCR errors.
// DESIGN PRINCIPLE: Never silently correct. Always flag.

const { ocrWeightedDistance } = require('./confusion');
const { VOCABULARY } = require('./vocabulary');

const LAB_RANGES = {
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

function checkLabRange(labName, rawValue) {
  const key = labName.toLowerCase().replace(/\s+/g, '');
  const range = LAB_RANGES[key];
  if (!range) return { status: 'unknown_lab', message: `No range for "${labName}"` };

  const value = parseFloat(String(rawValue).replace(',', '.'));
  if (isNaN(value)) return { status: 'parse_error', message: `Cannot parse "${rawValue}"` };

  if (value >= range.min && value <= range.max) {
    return { status: 'in_range', message: `${labName} ${value} ${range.unit} within range`, value };
  }

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
      status: 'OUT_OF_RANGE', severity: 'warning',
      message: `${labName} ${value} OUTSIDE [${range.min}-${range.max}] ${range.unit}. Possible: ${alternatives.join(', ')}. VERIFY.`,
      value, alternatives,
    };
  }

  return {
    status: 'OUT_OF_RANGE', severity: 'error',
    message: `${labName} ${value} OUTSIDE [${range.min}-${range.max}] ${range.unit}. Verify.`,
    value,
  };
}

function checkMedicationAmbiguity(ocrText, candidates) {
  if (candidates.length < 2) return null;
  const meds = new Set(VOCABULARY.medications.map(m => m.toLowerCase()));
  const top = candidates[0], second = candidates[1];
  if (meds.has(top.word.toLowerCase()) && meds.has(second.word.toLowerCase())) {
    const margin = second.distance - top.distance;
    if (margin < 1.5) {
      return {
        status: 'MEDICATION_AMBIGUITY', severity: 'danger',
        message: `"${ocrText}" ambiguous between "${top.word}" and "${second.word}". MUST VERIFY.`,
        candidates: [top, second],
      };
    }
  }
  return null;
}

function checkDosePlausibility(medName, doseStr) {
  // Normalize common OCR confusions in dose tokens: O→0, l→1, I→1
  const normalized = doseStr.replace(/O/g, '0').replace(/(?<=[0-9])l/g, '1').replace(/(?<=[0-9])I/g, '1');
  const match = normalized.match(/^(\d+(?:\.\d+)?)\s*(mg|g|mcg|ml|units?|iu)?$/i);
  if (!match) return null;
  const value = parseFloat(match[1]);
  const unit = (match[2] || '').toLowerCase();
  if (unit === 'mg' && value > 5000) {
    return { status: 'IMPLAUSIBLE_DOSE', severity: 'warning', message: `${medName} ${doseStr} seems high. Verify.` };
  }
  if (medName.toLowerCase().includes('insulin') && value > 200) {
    return { status: 'IMPLAUSIBLE_DOSE', severity: 'warning', message: `Insulin ${value} units very high. Verify.` };
  }
  return null;
}

function runSafetyChecks(words) {
  const flags = [];
  const labNames = new Set(Object.keys(LAB_RANGES));
  for (let i = 0; i < words.length; i++) {
    const w = words[i];
    if (w.candidates && w.candidates.length >= 2) {
      const medFlag = checkMedicationAmbiguity(w.original, w.candidates);
      if (medFlag) flags.push({ position: i, ...medFlag });
    }
    if (/^[\d.,]+$/.test(w.corrected)) {
      for (let j = i - 1; j >= Math.max(0, i - 3); j--) {
        const prevWord = words[j].corrected.toLowerCase().replace(/\s+/g, '');
        if (labNames.has(prevWord)) {
          const rangeFlag = checkLabRange(words[j].corrected, w.corrected);
          if (rangeFlag.status === 'OUT_OF_RANGE') flags.push({ position: i, ...rangeFlag });
          break;
        }
      }
    }
    if (i > 0 && /[\dO]/.test(w.corrected)) {
      const prevMed = words[i - 1].corrected.toLowerCase();
      const meds = new Set(VOCABULARY.medications.map(m => m.toLowerCase()));
      if (meds.has(prevMed)) {
        const doseFlag = checkDosePlausibility(prevMed, w.corrected);
        if (doseFlag) flags.push({ position: i, ...doseFlag });
      }
    }
  }
  return flags;
}

module.exports = { LAB_RANGES, checkLabRange, checkMedicationAmbiguity, checkDosePlausibility, runSafetyChecks };
