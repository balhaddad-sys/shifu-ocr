// Shifu OCR Confusion Model
// Topology-predicted character substitution costs.
// Lower cost = more likely OCR confusion = cheaper substitution.

const CONFUSION_PAIRS = {
  'O,0': 0.1, 'o,0': 0.1, 'D,O': 0.3, 'D,0': 0.3, 'Q,O': 0.3,
  'l,1': 0.2, 'I,1': 0.2, 'I,l': 0.1, 'l,|': 0.2,
  '5,S': 0.3, '5,s': 0.3, '8,B': 0.3, '6,G': 0.4, '6,b': 0.4,
  '2,Z': 0.4, '2,z': 0.4, '9,g': 0.2, '9,q': 0.4,
  'r,n': 0.3, 'rn,m': 0.2, 'cl,d': 0.3, 'vv,w': 0.3,
  'm,n': 0.4, 'u,v': 0.5, 'u,n': 0.5, 'c,e': 0.5,
  'h,b': 0.5, 'p,q': 0.4, 'f,t': 0.4, 'i,j': 0.4,
  'V,W': 0.4, 'E,F': 0.4,
  'a,A': 0.3, 'c,C': 0.2, 'o,O': 0.1, 's,S': 0.2,
  'v,V': 0.2, 'w,W': 0.2, 'x,X': 0.2, 'z,Z': 0.2,
  'a,e': 0.4, 'a,o': 0.4, 'e,i': 0.4, 'f,l': 0.4,
  's,e': 0.5, 'b,d': 0.4, 'r,n': 0.3, 'd,o': 0.3,
  // FLAIR engine systematic confusions (shape-similar under perturbation)
  // These must be LOWERCASE since ocrWeightedDistance lowercases inputs
  'i,t': 0.1, 't,i': 0.1,   // vertical stroke: I/i → t
  'i,l': 0.1, 'l,i': 0.1,   // vertical stroke: i ↔ l
  '6,g': 0.1, 'g,6': 0.1,   // loop + tail
  '1,j': 0.2, 'j,1': 0.2,   // vertical stroke
  'c,x': 0.2, 'x,c': 0.2,   // crossing strokes
  'u,v': 0.3,                // open curves
  'g,q': 0.3, 'g,9': 0.2,   // descender + loop
  '3,s': 0.2,                // curved forms
  // Additional FLAIR confusions from real document analysis
  'x,s': 0.3, 'x,k': 0.3, 'x,t': 0.3,   // X over-predicted
  'u,o': 0.3, 'u,a': 0.3,                  // U over-predicted
  'g,d': 0.3, 'g,a': 0.3, 'g,o': 0.3,     // g over-predicted
  'n,r': 0.3, 'n,h': 0.3,                  // n low-confidence
  // MRI-RF model v2 confusions (from real ward census output)
  'q,b': 0.2, 'q,d': 0.2, 'q,o': 0.2, 'q,g': 0.2,  // Q for round chars
  '9,g': 0.1,                                          // 9↔g
  '2,z': 0.2,                                          // 2↔z
  'j,i': 0.2, 'j,l': 0.2, 'j,1': 0.2,                // j↔thin verticals
  'w,m': 0.3,                                          // w↔m
  'v,u': 0.2,                                          // v↔u
};

// Optional adaptive profile can be injected to use learned costs
let _adaptiveProfile = null;

function setAdaptiveProfile(profile) {
  _adaptiveProfile = profile;
}

function getConfusionCost(char1, char2) {
  if (char1 === char2) return 0.0;
  // Use adaptive profile if available — it blends learned data with base costs
  if (_adaptiveProfile && typeof _adaptiveProfile.getCost === 'function') {
    return _adaptiveProfile.getCost(char1, char2);
  }
  const key1 = `${char1},${char2}`;
  const key2 = `${char2},${char1}`;
  return CONFUSION_PAIRS[key1] ?? CONFUSION_PAIRS[key2] ?? 1.0;
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
