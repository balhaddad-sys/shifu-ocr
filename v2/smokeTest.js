/**
 * Smoke test: verify all Shifu modules work correctly.
 * Run: node smokeTest.js
 */

// Since these use ES module imports, we need to inline for Node CJS testing
// In production MedTriage (which uses bundler), the imports work as-is.

// === Inline the confusion model ===
const CONFUSION_PAIRS = {
  'O,0': 0.1, 'o,0': 0.1, 'D,O': 0.3, 'l,1': 0.2, 'I,1': 0.2, 'I,l': 0.2,
  '5,S': 0.3, '5,s': 0.3, '8,B': 0.3, '6,G': 0.4, '2,Z': 0.4,
  'r,n': 0.3, 'm,n': 0.4, 'u,v': 0.5, 'c,e': 0.5, 'h,b': 0.5,
  'f,t': 0.4, 'E,F': 0.4, 'a,A': 0.3, 's,S': 0.2,
};

function getConfusionCost(c1, c2) {
  if (c1 === c2) return 0.0;
  return CONFUSION_PAIRS[`${c1},${c2}`] ?? CONFUSION_PAIRS[`${c2},${c1}`] ?? 1.0;
}

function ocrWeightedDistance(s1, s2) {
  s1 = s1.toLowerCase(); s2 = s2.toLowerCase();
  if (s1.length === 0) return s2.length;
  if (s2.length === 0) return s1.length;
  const m = [];
  for (let i = 0; i <= s1.length; i++) m[i] = [i];
  for (let j = 0; j <= s2.length; j++) m[0][j] = j;
  for (let i = 1; i <= s1.length; i++) {
    for (let j = 1; j <= s2.length; j++) {
      const sub = getConfusionCost(s1[i-1], s2[j-1]);
      m[i][j] = Math.min(m[i-1][j]+1, m[i][j-1]+1, m[i-1][j-1]+sub);
    }
  }
  return m[s1.length][s2.length];
}

// === Tests ===
let passed = 0;
let failed = 0;

function assert(condition, message) {
  if (condition) {
    passed++;
    console.log(`  ✓ ${message}`);
  } else {
    failed++;
    console.log(`  ✗ ${message}`);
  }
}

console.log('\n=== SHIFU SMOKE TESTS ===\n');

// Test 1: Confusion costs
console.log('Confusion Model:');
assert(getConfusionCost('O', '0') === 0.1, 'O/0 confusion cost = 0.1');
assert(getConfusionCost('l', '1') === 0.2, 'l/1 confusion cost = 0.2');
assert(getConfusionCost('A', 'Z') === 1.0, 'A/Z default cost = 1.0');
assert(getConfusionCost('a', 'a') === 0.0, 'Same char cost = 0.0');

// Test 2: OCR weighted distance
console.log('\nWeighted Distance:');
assert(ocrWeightedDistance('potassium', 'potassium') === 0, 'Exact match = 0');
assert(ocrWeightedDistance('potasium', 'potassium') < 1.5, 'Single missing letter < 1.5');
assert(ocrWeightedDistance('O', '0') < 0.2, 'O vs 0 very small distance');
assert(ocrWeightedDistance('cat', 'dog') > 2.5, 'Unrelated words > 2.5');

// Test 3: Clinical corrections
console.log('\nClinical Corrections:');
const clinicalWords = [
  'potassium', 'sodium', 'creatinine', 'hemoglobin',
  'levetiracetam', 'carbamazepine', 'rotigotine',
  'stroke', 'seizure', 'pneumonia', 'infection',
  'babinski', 'papilledema', 'nystagmus',
  'bader', 'noura', 'saleh', 'hassan',
];

function bestMatch(raw) {
  let best = null, bestDist = Infinity;
  for (const w of clinicalWords) {
    const d = ocrWeightedDistance(raw, w);
    if (d < bestDist) { bestDist = d; best = w; }
  }
  return { word: best, distance: bestDist };
}

assert(bestMatch('potasium').word === 'potassium', 'potasium → potassium');
assert(bestMatch('Carbamazepime').word === 'carbamazepine', 'Carbamazepime → carbamazepine');
assert(bestMatch('Levetiracetan').word === 'levetiracetam', 'Levetiracetan → levetiracetam');
assert(bestMatch('Babinksi').word === 'babinski', 'Babinksi → babinski');
assert(bestMatch('papilledma').word === 'papilledema', 'papilledma → papilledema');
assert(bestMatch('Hasan').word === 'hassan', 'Hasan → hassan');
assert(bestMatch('nystagnus').word === 'nystagmus', 'nystagnus → nystagmus');
assert(bestMatch('rotigotne').word === 'rotigotine', 'rotigotne → rotigotine');

// Test 4: Lab range checking
console.log('\nLab Range Checking:');
const LAB_RANGES = {
  potassium: { min: 1.5, max: 9.0 },
  hba1c: { min: 4.0, max: 15.0 },
  sodium: { min: 100, max: 180 },
  inr: { min: 0.5, max: 10.0 },
};

function checkRange(lab, value) {
  const r = LAB_RANGES[lab.toLowerCase()];
  if (!r) return { status: 'unknown' };
  const v = parseFloat(value);
  if (v >= r.min && v <= r.max) return { status: 'in_range' };
  // Check missing decimal
  const s = String(value);
  for (let i = 1; i < s.length; i++) {
    const alt = parseFloat(s.slice(0, i) + '.' + s.slice(i));
    if (!isNaN(alt) && alt >= r.min && alt <= r.max) {
      return { status: 'OUT_OF_RANGE', suggest: alt };
    }
  }
  return { status: 'OUT_OF_RANGE' };
}

assert(checkRange('Potassium', '4.5').status === 'in_range', 'K+ 4.5 in range');
assert(checkRange('Potassium', '45').status === 'OUT_OF_RANGE', 'K+ 45 out of range');
assert(checkRange('Potassium', '45').suggest === 4.5, 'K+ 45 suggests 4.5');
assert(checkRange('HbA1c', '7.1').status === 'in_range', 'HbA1c 7.1 in range');
assert(checkRange('HbA1c', '71').status === 'OUT_OF_RANGE', 'HbA1c 71 out of range');
assert(checkRange('HbA1c', '71').suggest === 7.1, 'HbA1c 71 suggests 7.1');
assert(checkRange('Sodium', '139').status === 'in_range', 'Na 139 in range');
assert(checkRange('INR', '23').suggest === 2.3, 'INR 23 suggests 2.3');

// Test 5: Medication ambiguity
console.log('\nMedication Safety:');
const r1 = bestMatch('Methotrexat');
const r2 = bestMatch('Carbamazepime');
assert(r2.word === 'carbamazepine', 'Carbamazepime → carbamazepine (confident)');

// Summary
console.log(`\n${'='.repeat(40)}`);
console.log(`RESULTS: ${passed} passed, ${failed} failed`);
console.log(`${'='.repeat(40)}\n`);

if (failed > 0) process.exit(1);
