// Shifu OCR Confusion Model
// Topology-predicted character substitution costs.
// Lower cost = more likely OCR confusion = cheaper substitution.

// ALL KEYS LOWERCASE + SORTED — ocrWeightedDistance lowercases inputs,
// and lookup uses sorted key for consistency with adaptive profile.
const CONFUSION_PAIRS = {
  // Classic OCR confusions
  '0,o': 0.1, 'd,o': 0.3, '0,d': 0.3, 'o,q': 0.3,
  '1,l': 0.2, '1,i': 0.2, 'i,l': 0.1, '|,l': 0.2,
  '5,s': 0.3, '8,b': 0.3, '6,g': 0.1, '6,b': 0.3,
  '2,z': 0.2, '9,g': 0.1, '9,q': 0.4,
  'n,r': 0.3, 'm,rn': 0.2, 'cl,d': 0.3, 'vv,w': 0.3,
  'm,n': 0.4, 'u,v': 0.3, 'n,u': 0.5, 'c,e': 0.5,
  'b,h': 0.5, 'p,q': 0.4, 'f,t': 0.4, 'i,j': 0.4,
  'v,w': 0.4, 'e,f': 0.4,
  'a,e': 0.4, 'a,o': 0.4, 'e,i': 0.4, 'f,l': 0.4,
  'e,s': 0.5, 'b,d': 0.4,
  // FLAIR engine systematic confusions
  'i,t': 0.1, 'g,q': 0.3,
  '1,j': 0.2, 'c,x': 0.2, '3,s': 0.2,
  'k,x': 0.3, 's,x': 0.3, 't,x': 0.3,
  'a,u': 0.3, 'o,u': 0.3,
  'a,g': 0.3, 'd,g': 0.3, 'g,o': 0.3,
  'h,n': 0.3,
  // 4D MRI systematic confusions
  'b,q': 0.2, 'd,q': 0.2, 'g,q': 0.2, 'o,q': 0.2,
  '4,a': 0.1, '4,e': 0.2, '4,o': 0.2,
  '0,n': 0.2, '3,b': 0.2, '6,e': 0.1,
  's,z': 0.3, 'm,w': 0.3,
  'i,l': 0.1, 'j,l': 0.2,
};

// Optional adaptive profile can be injected to use learned costs
let _adaptiveProfile = null;

function setAdaptiveProfile(profile) {
  _adaptiveProfile = profile;
}

function getConfusionCost(char1, char2) {
  if (char1 === char2) return 0.0;
  // CASE IS FREE: same letter, different case = same character.
  // FLAIR perturbation response is identical for 'a' and 'A'.
  // Case is spatial (baseline position), not shape.
  if (char1.toLowerCase() === char2.toLowerCase()) return 0.0;
  // Use adaptive profile if available — it blends learned data with base costs
  if (_adaptiveProfile && typeof _adaptiveProfile.getCost === 'function') {
    return _adaptiveProfile.getCost(char1, char2);
  }
  // Sorted key — consistent with adaptive profile's sorted lookup
  const key = [char1.toLowerCase(), char2.toLowerCase()].sort().join(',');
  return CONFUSION_PAIRS[key] ?? 1.0;
}

function ocrWeightedDistance(str1, str2) {
  const s1 = str1.toLowerCase();
  const s2 = str2.toLowerCase();
  if (s1.length === 0) return s2.length;
  if (s2.length === 0) return s1.length;
  const m = [];
  for (let i = 0; i <= s1.length; i++) m[i] = [i];
  for (let j = 0; j <= s2.length; j++) m[0][j] = j;
  for (let i = 1; i <= s1.length; i++) {
    for (let j = 1; j <= s2.length; j++) {
      const sub = getConfusionCost(s1[i - 1], s2[j - 1]);
      m[i][j] = Math.min(m[i-1][j] + 1, m[i][j-1] + 1, m[i-1][j-1] + sub);
    }
  }
  return m[s1.length][s2.length];
}

const DIGRAPH_CONFUSIONS = [
  { ocr: 'rn', correct: 'm', context: 'Adjacent strokes merge' },
  { ocr: 'cl', correct: 'd', context: 'Strokes merge into closed form' },
  { ocr: 'vv', correct: 'w', context: 'Double-v merges' },
  { ocr: 'nn', correct: 'm', context: 'Adjacent humps merge' },
  { ocr: 'ii', correct: 'u', context: 'Adjacent verticals merge bottom' },
  { ocr: 'li', correct: 'h', context: 'l + i merge' },
  { ocr: 'ri', correct: 'n', context: 'r + i merge' },
];

function fixDigraphs(text) {
  let result = text;
  for (const { ocr, correct } of DIGRAPH_CONFUSIONS) {
    result = result.replace(new RegExp(ocr, 'g'), correct);
  }
  return result;
}

module.exports = { CONFUSION_PAIRS, getConfusionCost, ocrWeightedDistance, DIGRAPH_CONFUSIONS, fixDigraphs, setAdaptiveProfile };
