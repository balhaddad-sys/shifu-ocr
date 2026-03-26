/**
 * Shifu OCR Confusion Model
 * 
 * Topology-predicted character substitution costs.
 * Characters with similar displacement signatures are more likely
 * to be confused by ANY OCR engine (Tesseract, Google Vision, etc).
 * 
 * Usage: plug into weighted edit distance for post-OCR correction.
 * Lower cost = more likely OCR confusion = cheaper substitution.
 */

// Confusion pairs with costs and reasoning
// Cost 0.0-0.2: Nearly identical topology, very likely confusion
// Cost 0.2-0.4: Similar topology, common confusion
// Cost 0.4-0.6: Partial similarity, occasional confusion
// Cost 1.0: Default (unrelated characters)
export const CONFUSION_PAIRS = {
  // Identical topology: 1 hole, symmetric → cost 0.1
  'O,0': 0.1,
  'o,0': 0.1,
  'D,O': 0.3,
  'D,0': 0.3,
  'Q,O': 0.3,

  // Identical topology: 0 holes, tall narrow vertical → cost 0.2
  'l,1': 0.2,
  'I,1': 0.2,
  'I,l': 0.2,
  'l,|': 0.2,

  // Similar curvature, 0 holes → cost 0.3
  '5,S': 0.3,
  '5,s': 0.3,
  '8,B': 0.3,
  '6,G': 0.4,
  '6,b': 0.4,
  '2,Z': 0.4,
  '2,z': 0.4,
  '9,g': 0.4,
  '9,q': 0.4,

  // Adjacent strokes merge at low resolution → cost 0.3
  'r,n': 0.3,
  'rn,m': 0.2, // digraph confusion
  'cl,d': 0.3,
  'vv,w': 0.3,

  // Similar structure → cost 0.4-0.5
  'm,n': 0.4,
  'u,v': 0.5,
  'u,n': 0.5,
  'c,e': 0.5,
  'h,b': 0.5,
  'p,q': 0.4,
  'f,t': 0.4,
  'i,j': 0.4,
  'V,W': 0.4,
  'E,F': 0.4,

  // Case confusions at low res
  'a,A': 0.3,
  'c,C': 0.2,
  'o,O': 0.1,
  's,S': 0.2,
  'v,V': 0.2,
  'w,W': 0.2,
  'x,X': 0.2,
  'z,Z': 0.2,
};

/**
 * Get the confusion cost between two characters.
 * Returns a value between 0.0 (identical) and 1.0 (unrelated).
 */
export function getConfusionCost(char1, char2) {
  if (char1 === char2) return 0.0;

  const key1 = `${char1},${char2}`;
  const key2 = `${char2},${char1}`;

  return CONFUSION_PAIRS[key1] ?? CONFUSION_PAIRS[key2] ?? 1.0;
}

/**
 * Weighted Levenshtein distance using OCR confusion costs.
 * Characters that are topologically similar cost less to substitute,
 * producing better "did you mean?" suggestions for OCR errors.
 */
export function ocrWeightedDistance(str1, str2) {
  const s1 = str1.toLowerCase();
  const s2 = str2.toLowerCase();

  if (s1.length === 0) return s2.length;
  if (s2.length === 0) return s1.length;

  const matrix = [];

  for (let i = 0; i <= s1.length; i++) {
    matrix[i] = [i];
  }
  for (let j = 0; j <= s2.length; j++) {
    matrix[0][j] = j;
  }

  for (let i = 1; i <= s1.length; i++) {
    for (let j = 1; j <= s2.length; j++) {
      const subCost = getConfusionCost(s1[i - 1], s2[j - 1]);
      matrix[i][j] = Math.min(
        matrix[i - 1][j] + 1,        // deletion
        matrix[i][j - 1] + 1,        // insertion
        matrix[i - 1][j - 1] + subCost // substitution
      );
    }
  }

  return matrix[s1.length][s2.length];
}

/**
 * Common OCR digraph confusions (multi-character).
 * These happen when adjacent characters merge or split.
 */
export const DIGRAPH_CONFUSIONS = [
  { ocr: 'rn', correct: 'm', context: 'Adjacent strokes merge' },
  { ocr: 'cl', correct: 'd', context: 'Strokes merge into closed form' },
  { ocr: 'vv', correct: 'w', context: 'Double-v merges' },
  { ocr: 'nn', correct: 'm', context: 'Adjacent humps merge' },
  { ocr: 'ii', correct: 'u', context: 'Adjacent verticals merge bottom' },
  { ocr: 'li', correct: 'h', context: 'l + i merge' },
  { ocr: 'ri', correct: 'n', context: 'r + i merge' },
];

/**
 * Apply digraph corrections to a raw OCR string.
 */
export function fixDigraphs(text) {
  let result = text;
  for (const { ocr, correct } of DIGRAPH_CONFUSIONS) {
    result = result.replace(new RegExp(ocr, 'g'), correct);
  }
  return result;
}
