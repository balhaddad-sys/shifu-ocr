// Shifu Clinical Vocabulary
// Organized for ward census / evacuation workflow + neurology.
// Each word is categorized so context-aware boosting can work.

const VOCABULARY = {
  ward_structure: [
    'ward', 'bed', 'room', 'icu', 'er', 'nicu', 'picu', 'ccu',
    'male', 'female', 'active', 'chronic', 'list',
    'discharge', 'discharged', 'admitted', 'transfer', 'transferred',
    'evacuation', 'status', 'unassigned', 'assigned',
    'patient', 'name', 'doctor', 'nurse', 'consultant', 'diagnosis',
    'window', 'confusional', 'state', 'acute', 'exacerbation',
    'sheet', 'unit', 'floor', 'section', 'block',
    'triage', 'red', 'yellow', 'green', 'black',
    'ambulatory', 'wheelchair', 'stretcher', 'bedridden',
    'oxygen', 'ventilator', 'bipap', 'cpap', 'nasal cannula',
    'isolation', 'contact', 'droplet', 'airborne',
    'full code', 'dnr', 'comfort care', 'dni',
    'allergy', 'nkda', 'penicillin allergy', 'sulfa allergy',
  ],

  diagnoses: [
    // Neurology
    'cva', 'stroke', 'ischemic', 'hemorrhagic', 'tia',
    'seizure', 'epilepsy', 'status epilepticus',
    'meningitis', 'encephalitis', 'demyelination',
    'neuropathy', 'myopathy', 'myelopathy',
    'parkinson', 'parkinsonian', 'tremor',
    'migraine', 'headache', 'cluster',
    'guillain', 'barre', 'myasthenia', 'gravis',
    'mca', 'aca', 'pca', 'occlusion', 'thrombosis',
    'subarachnoid', 'subdural', 'epidural', 'hematoma',
    'hydrocephalus', 'raised icp',
    'multiple sclerosis', 'transverse myelitis',
    'bell palsy', 'trigeminal neuralgia',
    'brain tumor', 'glioblastoma', 'meningioma',
    'cerebral edema', 'herniation',
    // Respiratory
    'chest infection', 'pneumonia', 'cap', 'hap', 'vap',
    'copd', 'exacerbation', 'asthma', 'pleural effusion',
    'pulmonary embolism', 'respiratory failure',
    'pneumothorax', 'hemothorax', 'ards',
    // Cardiac
    'heart failure', 'lvf', 'hfref', 'hfpef',
    'mi', 'stemi', 'nstemi', 'acs',
    'atrial fibrillation', 'af', 'svt', 'vt',
    'endocarditis', 'pericarditis',
    'hypertension', 'hypotension', 'cardiomyopathy',
    // Renal
    'aki', 'ckd', 'dialysis', 'hypernatremia', 'hyponatremia',
    'hyperkalemia', 'hypokalemia', 'metabolic acidosis',
    'urosepsis', 'uti', 'pyelonephritis',
    // GI / Hepato
    'cholestitis', 'cholecystitis', 'cholangitis',
    'pancreatitis', 'gi bleed', 'upper gi bleed',
    'liver cirrhosis', 'hepatitis', 'ascites',
    'bowel obstruction', 'ileus',
    // Hematology
    'anemia', 'pancytopenia', 'dvt', 'pe',
    'dic', 'thrombocytopenia', 'dlc',
    // Infectious
    'sepsis', 'bacteremia', 'cellulitis', 'abscess',
    'covid', 'tuberculosis', 'malaria',
    // Endocrine
    'dka', 'hhs', 'diabetes', 'thyroid storm', 'addisonian crisis',
    'diabetes mellitus', 'hypothyroidism', 'hyperthyroidism',
    // Other
    'drug overdose', 'poisoning', 'fall', 'fracture',
    'delirium', 'altered mental status',
    'pressure injury', 'decubitus',
    'malnutrition', 'weight loss',
    'palliative', 'end of life',
  ],

  examination: [
    'mental status', 'cranial nerves', 'motor', 'sensory',
    'reflexes', 'coordination', 'gait', 'cerebellar',
    'examination', 'neurological', 'cardiovascular',
    'respiratory', 'abdominal', 'musculoskeletal',
    'intact', 'normal', 'abnormal',
    'positive', 'negative', 'present', 'absent',
    'mild', 'moderate', 'severe',
    'acute', 'chronic', 'subacute',
    'bilateral', 'unilateral', 'right', 'left',
    'upper', 'lower', 'proximal', 'distal',
    'pupils', 'reactive', 'perrla', 'nystagmus',
    'papilledema', 'ptosis', 'diplopia',
    'power', 'tone', 'babinski',
    'clonus', 'brisk', 'upgoing', 'downgoing',
    'proprioception', 'vibration', 'pinprick',
    'romberg', 'dysmetria', 'ataxia',
    'fasciculations', 'spasticity', 'rigidity',
    'weakness', 'hemiparesis', 'hemiplegia',
    'paraparesis', 'paraplegia', 'quadriparesis',
    'dysphasia', 'aphasia', 'dysarthria',
    'dysphagia', 'neglect', 'hemianopia',
  ],

  medications: [
    // Neurology - Anticonvulsants
    'levetiracetam', 'keppra', 'valproate', 'depakine',
    'carbamazepine', 'tegretol', 'phenytoin', 'dilantin',
    'lamotrigine', 'topiramate', 'lacosamide', 'vimpat',
    'gabapentin', 'pregabalin', 'lyrica',
    'phenobarbital', 'clobazam', 'brivaracetam',
    // Parkinson
    'levodopa', 'carbidopa', 'sinemet',
    'rotigotine', 'neupro', 'pramipexole', 'ropinirole',
    'rasagiline', 'selegiline', 'entacapone',
    'amantadine', 'trihexyphenidyl', 'benztropine',
    // Anticoagulants / antiplatelets
    'aspirin', 'clopidogrel', 'plavix',
    'warfarin', 'coumadin', 'heparin', 'enoxaparin', 'clexane',
    'rivaroxaban', 'xarelto', 'apixaban', 'eliquis',
    'dabigatran', 'pradaxa', 'ticagrelor', 'prasugrel',
    // Thrombolytics
    'alteplase', 'tenecteplase', 'tpa',
    // Cardiac
    'amlodipine', 'lisinopril', 'enalapril', 'ramipril',
    'bisoprolol', 'metoprolol', 'atenolol', 'propranolol',
    'atorvastatin', 'rosuvastatin', 'simvastatin',
    'amiodarone', 'digoxin', 'verapamil', 'diltiazem',
    'furosemide', 'lasix', 'spironolactone',
    'isosorbide', 'nitroglycerin', 'hydralazine',
    'losartan', 'valsartan', 'candesartan',
    // GI
    'omeprazole', 'pantoprazole', 'esomeprazole',
    'metoclopramide', 'ondansetron', 'domperidone',
    'lactulose', 'loperamide',
    // Steroids
    'prednisolone', 'methylprednisolone', 'dexamethasone',
    'hydrocortisone', 'fludrocortisone',
    // Pain / sedation
    'paracetamol', 'acetaminophen', 'ibuprofen', 'diclofenac',
    'morphine', 'tramadol', 'fentanyl', 'codeine',
    'diazepam', 'lorazepam', 'midazolam', 'clonazepam',
    // Antibiotics
    'amoxicillin', 'augmentin', 'ceftriaxone', 'cefuroxime',
    'meropenem', 'piperacillin', 'tazobactam', 'tazocin',
    'vancomycin', 'metronidazole', 'ciprofloxacin',
    'azithromycin', 'clarithromycin', 'doxycycline',
    'trimethoprim', 'nitrofurantoin',
    'levofloxacin', 'gentamicin', 'amikacin',
    'fluconazole', 'acyclovir', 'oseltamivir',
    // Endocrine
    'metformin', 'gliclazide', 'insulin',
    'levothyroxine', 'carbimazole',
    'glargine', 'lispro', 'aspart', 'detemir',
    // Other
    'mannitol', 'thiamine', 'pyridoxine',
    'potassium chloride', 'calcium gluconate',
    'iron', 'folic acid', 'vitamin d',
    'magnesium sulfate', 'sodium bicarbonate',
    'albumin', 'immunoglobulin', 'ivig',
    'plasma exchange', 'plasmapheresis',
  ],

  labs: [
    'hba1c', 'glucose', 'sodium', 'potassium', 'chloride',
    'bicarbonate', 'urea', 'creatinine', 'egfr',
    'hemoglobin', 'hematocrit', 'wbc', 'platelets',
    'neutrophils', 'lymphocytes', 'monocytes',
    'inr', 'pt', 'aptt', 'fibrinogen', 'ddimer',
    'crp', 'esr', 'procalcitonin',
    'alt', 'ast', 'alp', 'ggt', 'bilirubin', 'albumin',
    'tsh', 'free t4', 'free t3',
    'calcium', 'magnesium', 'phosphate',
    'lactate', 'ammonia', 'ck', 'troponin', 'bnp',
    'csf protein', 'csf glucose', 'csf wbc',
    'blood culture', 'urine culture', 'csf culture',
    'abg', 'vbg', 'ph', 'pco2', 'po2',
    // Common medical acronyms (short but valid)
    'mri', 'ct', 'ecg', 'eeg', 'emg', 'cxr', 'xr',
    'gcs', 'bp', 'hr', 'rr', 'spo2', 'bmi', 'bsa',
    'iv', 'im', 'sc', 'po', 'pr', 'ng', 'nj',
    'od', 'bd', 'tds', 'qds', 'prn', 'stat',
    'npo', 'npo', 'obs', 'rx', 'dx', 'hx', 'sx', 'tx', 'mx',
  ],

  names: [
    'bader', 'noura', 'bazzah', 'bazza', 'saleh',
    'hisham', 'alathoub', 'athoub', 'zahra',
    'nawaf', 'hassan', 'jamal', 'ahmad', 'alessa',
    'ali', 'hussain', 'adel', 'mneer', 'osama', 'waseem', 'najaf',
    'abdullatif', 'abdulhafez', 'abdullah', 'mohammad', 'mohammed',
    'mahmoud', 'noor', 'islam',
    'rojelo', 'abdolmohsen', 'raju',
    'khalid', 'fahad', 'sultan', 'faisal',
    'nasser', 'yousef', 'ibrahim', 'omar',
    'fatima', 'maryam', 'aisha', 'sara',
    'hamad', 'turki', 'saud', 'badr', 'dalal',
    'layla', 'mariam', 'huda', 'sulaiman',
  ],
};

const OCR_INDEX_EQUIVALENTS = {
  '0': 'o',
  '1': 'l',
  '3': 'e',
  '5': 's',
  '6': 'g',
  '8': 'b',
  i: 'l',
};

const LARGE_VOCAB_PREFIXES = [
  'neuro', 'cardio', 'pulmo', 'gastro', 'hepato', 'nephro', 'uro', 'dermato',
  'hemo', 'hemato', 'endo', 'ortho', 'psycho', 'ophthalmo', 'oto', 'infecto',
  'immuno', 'electro', 'angio', 'broncho', 'cyto', 'lympho', 'myelo', 'osteo',
  'vasculo', 'tachy', 'brady', 'hyper', 'hypo', 'peri', 'post', 'pre', 'anti',
  'micro', 'macro', 'inter', 'intra', 'sub', 'supra', 'trans',
];

const LARGE_VOCAB_SUFFIXES = [
  'itis', 'osis', 'emia', 'uria', 'algia', 'pathy', 'plegia', 'paresis', 'penia',
  'cytosis', 'ectomy', 'otomy', 'ostomy', 'graphy', 'gram', 'scopy', 'genic',
  'genicity', 'tropic', 'stasis', 'lysis', 'logy', 'metry', 'meter', 'phasia',
  'phagia', 'rrhea', 'rrhage', 'rrhagia', 'malacia', 'megaly', 'sclerosis',
  'spasm', 'toxicity', 'dysfunction', 'syndrome', 'insufficiency', 'disease',
];

let cachedWordSet = null;
let cachedCandidateIndex = null;

function collectWords(terms, target) {
  for (const term of terms) {
    const lower = String(term || '').toLowerCase().trim();
    if (!lower) continue;
    target.add(lower);
    for (const word of lower.split(/\s+/)) {
      if (word.length > 1) target.add(word);
    }
  }
}

function rebuildWordSet() {
  const words = new Set();
  for (const category of Object.values(VOCABULARY)) {
    collectWords(category, words);
  }
  return words;
}

function invalidateVocabularyCaches() {
  cachedWordSet = null;
  cachedCandidateIndex = null;
}

function getBaseWordSet() {
  if (!cachedWordSet) {
    cachedWordSet = rebuildWordSet();
  }
  return cachedWordSet;
}

function buildWordSet() {
  return new Set(getBaseWordSet());
}

function normalizeIndexWord(word) {
  return String(word || '')
    .toLowerCase()
    .trim()
    .replace(/[^a-z0-9]/g, '')
    .split('')
    .map(char => OCR_INDEX_EQUIVALENTS[char] || char)
    .join('');
}

function prefixKey(normalized) {
  return normalized.slice(0, 2);
}

function suffixKey(normalized) {
  return normalized.slice(-2);
}

function deletionKeys(normalized) {
  const keys = new Set();
  if (!normalized) return keys;
  for (let i = 0; i < normalized.length; i++) {
    keys.add(normalized.slice(0, i) + normalized.slice(i + 1));
  }
  return keys;
}

function addBucketValue(map, key, value) {
  if (!key && key !== 0) return;
  let bucket = map.get(key);
  if (!bucket) {
    bucket = new Set();
    map.set(key, bucket);
  }
  bucket.add(value);
}

function createCandidateIndex(words) {
  const index = {
    allWords: new Set(),
    normalized: new Map(),
    exact: new Map(),
    deletes: new Map(),
    byPrefixLength: new Map(),
    bySuffixLength: new Map(),
    byLength: new Map(),
  };

  for (const word of words) {
    addWordToCandidateIndex(index, word);
  }

  return index;
}

function addWordToCandidateIndex(index, word) {
  const lower = String(word || '').toLowerCase().trim();
  if (!lower || index.allWords.has(lower)) return;

  index.allWords.add(lower);

  const normalized = normalizeIndexWord(lower);
  index.normalized.set(lower, normalized);
  addBucketValue(index.byLength, lower.length, lower);

  if (!normalized) return;

  addBucketValue(index.exact, normalized, lower);
  addBucketValue(index.byPrefixLength, `${prefixKey(normalized)}:${lower.length}`, lower);
  addBucketValue(index.bySuffixLength, `${suffixKey(normalized)}:${lower.length}`, lower);

  for (const key of deletionKeys(normalized)) {
    addBucketValue(index.deletes, key, lower);
  }
}

function scoreCandidateShape(query, normalizedQuery, candidate, index) {
  const normalizedCandidate = index?.normalized?.get(candidate) || normalizeIndexWord(candidate);
  let score = 0;
  if (candidate[0] === query[0]) score += 3;
  if (candidate.slice(-1) === query.slice(-1)) score += 2;
  if (prefixKey(normalizedCandidate) === prefixKey(normalizedQuery)) score += 2;
  if (suffixKey(normalizedCandidate) === suffixKey(normalizedQuery)) score += 1;
  score -= Math.abs(candidate.length - query.length) * 0.75;
  return score;
}

function fallbackCandidateSubset(query, fallbackWords, maxLengthDiff) {
  const fallback = new Set();
  if (!fallbackWords) return fallback;
  for (const candidate of fallbackWords) {
    if (Math.abs(candidate.length - query.length) > maxLengthDiff + 1) continue;
    if (candidate[0] === query[0] || candidate.slice(-1) === query.slice(-1)) {
      fallback.add(candidate);
    }
  }
  return fallback.size > 0 ? fallback : new Set(fallbackWords);
}

function getIndexedCandidates(queryWord, index, fallbackWords = null, options = {}) {
  const query = String(queryWord || '').toLowerCase().trim();
  if (!query) return new Set();

  if (!index) {
    return fallbackCandidateSubset(query, fallbackWords, options.maxLengthDiff ?? 2);
  }

  const normalized = normalizeIndexWord(query);
  const maxLengthDiff = options.maxLengthDiff ?? (query.length <= 4 ? 1 : 2);
  const maxCandidates = options.maxCandidates ?? 512;
  const candidates = new Set();

  const addBucket = bucket => {
    if (!bucket) return;
    for (const value of bucket) candidates.add(value);
  };

  addBucket(index.exact.get(normalized));
  addBucket(index.deletes.get(normalized));
  for (const key of deletionKeys(normalized)) {
    addBucket(index.deletes.get(key));
  }

  for (let len = Math.max(1, query.length - maxLengthDiff); len <= query.length + maxLengthDiff; len++) {
    addBucket(index.byPrefixLength.get(`${prefixKey(normalized)}:${len}`));
    if (candidates.size < 96) {
      addBucket(index.bySuffixLength.get(`${suffixKey(normalized)}:${len}`));
    }
  }

  if (candidates.size < 32) {
    for (let len = Math.max(1, query.length - maxLengthDiff); len <= query.length + maxLengthDiff; len++) {
      addBucket(index.byLength.get(len));
    }
  }

  if (candidates.size === 0) {
    return fallbackCandidateSubset(query, fallbackWords, maxLengthDiff);
  }

  const filtered = Array.from(candidates)
    .filter(candidate => Math.abs(candidate.length - query.length) <= maxLengthDiff + 1)
    .sort((a, b) => scoreCandidateShape(query, normalized, b, index) - scoreCandidateShape(query, normalized, a, index))
    .slice(0, maxCandidates);

  return new Set(filtered);
}

function getBaseCandidateIndex() {
  if (!cachedCandidateIndex) {
    cachedCandidateIndex = createCandidateIndex(getBaseWordSet());
  }
  return cachedCandidateIndex;
}

function getSeedRoots() {
  const roots = new Set();
  for (const [category, terms] of Object.entries(VOCABULARY)) {
    if (category === 'generated_large') continue;
    for (const term of terms) {
      const lower = String(term || '').toLowerCase();
      for (const word of lower.split(/\s+/)) {
        if (/^[a-z]+$/.test(word) && word.length >= 3 && word.length <= 12) {
          roots.add(word);
        }
      }
    }
  }
  return Array.from(roots).sort();
}

function ensureVocabularySize(targetSize = 100000) {
  const target = Math.max(0, Number(targetSize) || 0);
  const currentSize = getBaseWordSet().size;
  if (!target || currentSize >= target) {
    return { targetSize: target, totalSize: currentSize, addedTerms: 0 };
  }

  if (!Array.isArray(VOCABULARY.generated_large)) {
    VOCABULARY.generated_large = [];
  }

  const generated = VOCABULARY.generated_large;
  const existingTerms = new Set(generated.map(term => term.toLowerCase()));
  const existingWords = new Set(getBaseWordSet());
  const roots = getSeedRoots();
  const initialGeneratedCount = generated.length;

  const tryAdd = term => {
    const lower = String(term || '').toLowerCase().trim();
    if (!/^[a-z]+$/.test(lower)) return false;
    if (lower.length < 4 || lower.length > 28) return false;
    if (existingWords.has(lower) || existingTerms.has(lower)) return false;
    generated.push(lower);
    existingTerms.add(lower);
    existingWords.add(lower);
    return existingWords.size < target;
  };

  const appendCombinations = builder => {
    for (const root of roots) {
      for (const part of builder(root)) {
        const shouldContinue = tryAdd(part);
        if (!shouldContinue && existingWords.size >= target) return true;
      }
    }
    return existingWords.size >= target;
  };

  appendCombinations(root => LARGE_VOCAB_PREFIXES.map(prefix => `${prefix}${root}`));
  if (existingWords.size < target) {
    appendCombinations(root => LARGE_VOCAB_SUFFIXES.map(suffix => `${root}${suffix}`));
  }
  if (existingWords.size < target) {
    appendCombinations(root => {
      const composites = [];
      for (const prefix of LARGE_VOCAB_PREFIXES) {
        for (const suffix of LARGE_VOCAB_SUFFIXES) {
          composites.push(`${prefix}${root}${suffix}`);
        }
      }
      return composites;
    });
  }

  invalidateVocabularyCaches();
  return {
    targetSize: target,
    totalSize: getBaseWordSet().size,
    addedTerms: generated.length - initialGeneratedCount,
  };
}

function getColumnVocabulary(columnType) {
  const type = columnType.toLowerCase();
  if (type.includes('patient') || type.includes('name')) {
    return new Set(VOCABULARY.names.map(w => w.toLowerCase()));
  }
  if (type.includes('diagnos') || type.includes('dx')) {
    const words = new Set();
    for (const term of VOCABULARY.diagnoses) {
      for (const w of term.toLowerCase().split(/\s+/)) {
        if (w.length > 1) words.add(w);
      }
    }
    return words;
  }
  if (type.includes('doctor') || type.includes('assign')) {
    return new Set(VOCABULARY.names.map(w => w.toLowerCase()));
  }
  if (type.includes('status') || type.includes('evacuation') || type.includes('triage')) {
    return new Set(VOCABULARY.ward_structure.map(w => w.toLowerCase()));
  }
  if (type.includes('room') || type.includes('ward') || type.includes('bed')) {
    return new Set(VOCABULARY.ward_structure.map(w => w.toLowerCase()));
  }
  if (type.includes('med') || type.includes('drug') || type.includes('rx')) {
    return new Set(VOCABULARY.medications.map(w => w.toLowerCase()));
  }
  if (type.includes('lab') || type.includes('result')) {
    return new Set(VOCABULARY.labs.map(w => w.toLowerCase()));
  }
  if (type.includes('exam') || type.includes('finding')) {
    return new Set(VOCABULARY.examination.map(w => w.toLowerCase()));
  }
  return new Set(getBaseWordSet());
}

module.exports = {
  VOCABULARY,
  buildWordSet,
  getBaseWordSet,
  getBaseCandidateIndex,
  createCandidateIndex,
  addWordToCandidateIndex,
  getIndexedCandidates,
  getColumnVocabulary,
  invalidateVocabularyCaches,
  ensureVocabularySize,
};
