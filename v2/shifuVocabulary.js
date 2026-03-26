/**
 * Shifu Clinical Vocabulary
 * 
 * Organized for ward census / evacuation workflow + neurology.
 * Each word is categorized so context-aware boosting can work:
 * if we're reading a "Diagnosis" column, diagnosis terms get priority.
 * 
 * Extend by adding to the relevant category array.
 */

export const VOCABULARY = {
  // === WARD CENSUS / EVACUATION ===
  ward_structure: [
    'ward', 'bed', 'room', 'icu', 'er', 'nicu', 'picu', 'ccu',
    'male', 'female', 'active', 'chronic', 'list',
    'discharge', 'discharged', 'admitted', 'transfer', 'transferred',
    'evacuation', 'status', 'unassigned', 'assigned',
    'patient', 'name', 'doctor', 'nurse', 'consultant',
  ],

  // === DIAGNOSES (general medicine + neuro) ===
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
    // Respiratory
    'chest infection', 'pneumonia', 'cap', 'hap',
    'copd', 'exacerbation', 'asthma', 'pleural effusion',
    'pulmonary embolism', 'respiratory failure',
    // Cardiac
    'heart failure', 'lvf', 'hfref', 'hfpef',
    'mi', 'stemi', 'nstemi', 'acs',
    'atrial fibrillation', 'af', 'svt', 'vt',
    'endocarditis', 'pericarditis',
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
    // Other
    'drug overdose', 'poisoning', 'fall', 'fracture',
    'delirium', 'altered mental status',
    'pressure injury', 'decubitus',
    'malnutrition', 'weight loss',
    'palliative', 'end of life',
  ],

  // === EXAMINATION TERMS ===
  examination: [
    'mental status', 'cranial nerves', 'motor', 'sensory',
    'reflexes', 'coordination', 'gait', 'cerebellar',
    'examination', 'neurological', 'cardiovascular',
    'respiratory', 'abdominal', 'musculoskeletal',
    // Findings
    'intact', 'normal', 'abnormal',
    'positive', 'negative', 'present', 'absent',
    'mild', 'moderate', 'severe',
    'acute', 'chronic', 'subacute',
    'bilateral', 'unilateral', 'right', 'left',
    'upper', 'lower', 'proximal', 'distal',
    // Neuro-specific
    'pupils', 'reactive', 'perrla', 'nystagmus',
    'papilledema', 'ptosis', 'diplopia',
    'power', 'tone', 'reflexes', 'babinski',
    'clonus', 'brisk', 'upgoing', 'downgoing',
    'proprioception', 'vibration', 'pinprick',
    'romberg', 'dysmetria', 'ataxia',
    'fasciculations', 'spasticity', 'rigidity',
  ],

  // === MEDICATIONS ===
  medications: [
    // Neurology
    'levetiracetam', 'keppra', 'valproate', 'depakine',
    'carbamazepine', 'tegretol', 'phenytoin', 'dilantin',
    'lamotrigine', 'topiramate', 'lacosamide', 'vimpat',
    'gabapentin', 'pregabalin', 'lyrica',
    'levodopa', 'carbidopa', 'sinemet',
    'rotigotine', 'neupro', 'pramipexole', 'ropinirole',
    'rasagiline', 'selegiline', 'entacapone',
    'amantadine', 'trihexyphenidyl', 'benztropine',
    // Anticoagulants / antiplatelets
    'aspirin', 'clopidogrel', 'plavix',
    'warfarin', 'coumadin', 'heparin', 'enoxaparin', 'clexane',
    'rivaroxaban', 'xarelto', 'apixaban', 'eliquis',
    'dabigatran', 'pradaxa',
    // Thrombolytics
    'alteplase', 'tenecteplase', 'tpa',
    // Cardiac
    'amlodipine', 'lisinopril', 'enalapril', 'ramipril',
    'bisoprolol', 'metoprolol', 'atenolol', 'propranolol',
    'atorvastatin', 'rosuvastatin', 'simvastatin',
    'amiodarone', 'digoxin', 'verapamil', 'diltiazem',
    'furosemide', 'lasix', 'spironolactone',
    'isosorbide', 'nitroglycerin', 'hydralazine',
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
    // Endocrine
    'metformin', 'gliclazide', 'insulin',
    'levothyroxine', 'carbimazole',
    // Other
    'mannitol', 'thiamine', 'pyridoxine',
    'potassium chloride', 'calcium gluconate',
    'iron', 'folic acid', 'vitamin d',
  ],

  // === LAB TESTS ===
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
  ],

  // === COMMON NAMES (Kuwait ward context) ===
  names: [
    'bader', 'noura', 'bazzah', 'bazza', 'saleh',
    'hisham', 'alathoub', 'athoub', 'zahra',
    'nawaf', 'hassan', 'jamal', 'ahmad', 'alessa',
    'ali', 'hussain', 'adel', 'mneer',
    'abdullatif', 'abdullah', 'mohammad', 'mohammed',
    'rojelo', 'abdolmohsen', 'raju',
    'khalid', 'fahad', 'sultan', 'faisal',
    'nasser', 'yousef', 'ibrahim', 'omar',
    'fatima', 'maryam', 'aisha', 'sara',
  ],
};

/**
 * Build a flat Set of all vocabulary words for quick lookup.
 */
export function buildWordSet() {
  const words = new Set();
  for (const category of Object.values(VOCABULARY)) {
    for (const term of category) {
      // Add the full term
      words.add(term.toLowerCase());
      // Also add individual words from multi-word terms
      for (const w of term.split(/\s+/)) {
        if (w.length > 1) words.add(w.toLowerCase());
      }
    }
  }
  return words;
}

/**
 * Get context-relevant vocabulary for a given column type.
 * Returns words that should get a matching boost in that column.
 */
export function getColumnVocabulary(columnType) {
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
  if (type.includes('status') || type.includes('evacuation')) {
    return new Set(VOCABULARY.ward_structure.map(w => w.toLowerCase()));
  }
  if (type.includes('room') || type.includes('ward') || type.includes('bed')) {
    return new Set(VOCABULARY.ward_structure.map(w => w.toLowerCase()));
  }
  if (type.includes('med') || type.includes('drug') || type.includes('rx')) {
    return new Set(VOCABULARY.medications.map(w => w.toLowerCase()));
  }

  // Default: return everything
  return buildWordSet();
}
