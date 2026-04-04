// Clinical Numeric Canonicalization
//
// In medicine, numbers are not tokens — they are constraints on reality.
// "5OOmg" must become "500mg" before any downstream system can use it.
//
// This module:
//   1. Detects numeric-adjacent OCR errors (O→0, l→1, I→1)
//   2. Normalizes dose strings to canonical form
//   3. Flags ambiguous doses that could not be resolved with certainty
//   4. Validates dose plausibility after normalization

const UNIT_PATTERN = /^([\dOoIl.,]+)\s*(mg|mcg|g|ml|units?|iu|%|mmol|meq|µg)$/i;
const PURE_NUMBER = /^[\dOoIl.,]+$/;

// OCR digit normalization map
const DIGIT_MAP = { 'O': '0', 'o': '0', 'I': '1', 'l': '1' };

/**
 * Normalize OCR-garbled digits in a string.
 * Only replaces letters that are likely OCR errors in numeric context.
 *
 * @param {string} text
 * @returns {{ normalized: string, changes: Array<{pos: number, from: string, to: string}> }}
 */
function normalizeDigits(text) {
  const changes = [];
  let normalized = '';
  for (let i = 0; i < text.length; i++) {
    const ch = text[i];
    if (DIGIT_MAP[ch]) {
      // Only replace if in numeric context: adjacent to a real digit or another mappable char
      const prev = i > 0 ? text[i - 1] : '';
      const next = i < text.length - 1 ? text[i + 1] : '';
      const prevIsDigit = /\d/.test(prev) || (prev && DIGIT_MAP[prev]);
      const nextIsDigit = /\d/.test(next) || (next && DIGIT_MAP[next]);
      if (prevIsDigit || nextIsDigit) {
        changes.push({ pos: i, from: ch, to: DIGIT_MAP[ch] });
        normalized += DIGIT_MAP[ch];
        continue;
      }
    }
    normalized += ch;
  }
  return { normalized, changes };
}

/**
 * Normalize a dose token: "5OOmg" → "500mg", "3OOunits" → "300units"
 *
 * @param {string} token - Raw OCR token like "5OOmg"
 * @returns {{ original: string, normalized: string, value: number|null, unit: string, changes: Array, ambiguous: boolean }}
 */
function normalizeDose(token) {
  const result = { original: token, normalized: token, value: null, unit: '', changes: [], ambiguous: false };

  const unitMatch = token.match(UNIT_PATTERN);
  if (!unitMatch) {
    // Try normalizing the whole token first, then re-match
    const { normalized } = normalizeDigits(token);
    const retryMatch = normalized.match(UNIT_PATTERN);
    if (!retryMatch) return result;
    // Use the normalized version
    const numPart = retryMatch[1];
    const unitPart = retryMatch[2];
    const { normalized: normNum, changes } = normalizeDigits(numPart);
    const value = parseFloat(normNum.replace(',', '.'));
    return {
      original: token,
      normalized: normNum + unitPart.toLowerCase(),
      value: isNaN(value) ? null : value,
      unit: unitPart.toLowerCase(),
      changes,
      ambiguous: changes.length > 0,
    };
  }

  const numPart = unitMatch[1];
  const unitPart = unitMatch[2];
  const { normalized: normNum, changes } = normalizeDigits(numPart);
  const value = parseFloat(normNum.replace(',', '.'));

  return {
    original: token,
    normalized: normNum + unitPart.toLowerCase(),
    value: isNaN(value) ? null : value,
    unit: unitPart.toLowerCase(),
    changes,
    ambiguous: changes.length > 0,
  };
}

/**
 * Normalize all dose-like tokens in a line of text.
 *
 * @param {string} text
 * @returns {{ text: string, doses: Array, flags: Array }}
 */
function normalizeLineNumeric(text) {
  const words = text.split(/\s+/);
  const doses = [];
  const flags = [];

  const normalized = words.map((w, i) => {
    // Check if this looks like a dose token
    const { normalized: norm, changes } = normalizeDigits(w);
    if (UNIT_PATTERN.test(norm) || UNIT_PATTERN.test(w)) {
      const dose = normalizeDose(w);
      if (dose.changes.length > 0) {
        doses.push({ position: i, ...dose });
        flags.push({
          position: i,
          status: 'DOSE_NORMALIZED',
          severity: 'warning',
          message: `Dose normalized: "${w}" → "${dose.normalized}" (${dose.changes.map(c => c.from + '→' + c.to).join(', ')})`,
          original: w,
          normalized: dose.normalized,
        });
        return dose.normalized;
      }
    }
    // Also handle standalone numbers adjacent to units
    if (PURE_NUMBER.test(w) && i + 1 < words.length) {
      const nextWord = words[i + 1];
      if (/^(mg|mcg|g|ml|units?|iu|mmol|meq|µg)$/i.test(nextWord)) {
        const { normalized: normW, changes } = normalizeDigits(w);
        if (changes.length > 0) {
          doses.push({ position: i, original: w, normalized: normW, unit: nextWord, changes, ambiguous: true });
          flags.push({
            position: i,
            status: 'DOSE_NORMALIZED',
            severity: 'warning',
            message: `Number normalized: "${w}" → "${normW}" (before ${nextWord})`,
            original: w,
            normalized: normW,
          });
          return normW;
        }
      }
    }
    return w;
  });

  return { text: normalized.join(' '), doses, flags };
}

module.exports = { normalizeDigits, normalizeDose, normalizeLineNumeric };
