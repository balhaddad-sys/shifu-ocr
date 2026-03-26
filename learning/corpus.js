// Shifu Medical Corpus Seeder
// Pre-trains the core engine with medical domain knowledge.
// Feeds clinical sentences to build resonance between medical terms.

const MEDICAL_CORPUS = [
  // ─── NEUROLOGY ────────────────────────────────────────────────────
  // Stroke
  "Doctor ordered CT head for the acute stroke patient.",
  "Patient presents with acute ischemic stroke and left sided weakness.",
  "Physician examined the patient and noted bilateral weakness.",
  "Doctor prescribed levetiracetam for seizure prophylaxis.",
  "Patient admitted to stroke unit for monitoring and observation.",
  "CT head showed no evidence of hemorrhage or midline shift.",
  "MRI brain confirmed acute ischemic stroke in MCA territory.",
  "Aspirin and clopidogrel prescribed for dual antiplatelet therapy.",
  "Patient has history of hypertension and diabetes mellitus.",
  "Consultant reviewed the patient and recommended thrombolysis.",
  "Alteplase administered within the therapeutic window for stroke.",
  "Patient developed hemorrhagic transformation after thrombolysis.",
  "Doctor ordered repeat CT head to rule out hemorrhage.",
  "Patient transferred to ICU for close neurological monitoring.",
  "Nurse documented neurological observations every fifteen minutes.",

  // Seizure
  "Patient presented with new onset seizure and altered consciousness.",
  "Doctor prescribed levetiracetam for seizure control.",
  "Physician administered lorazepam for status epilepticus.",
  "EEG ordered to evaluate for epileptiform activity.",
  "Patient on valproate for epilepsy with good seizure control.",
  "Phenytoin level checked and found to be subtherapeutic.",
  "Doctor increased carbamazepine dose for breakthrough seizures.",
  "Lacosamide added as adjunctive therapy for focal seizures.",

  // Parkinson / Movement
  "Patient diagnosed with Parkinson disease and started on levodopa.",
  "Physician prescribed carbidopa levodopa for tremor and rigidity.",
  "Rotigotine patch applied for Parkinson disease management.",
  "Patient has essential tremor treated with propranolol.",
  "Doctor noted cogwheel rigidity and bradykinesia on examination.",

  // Headache / Neuralgia
  "Patient presents with severe migraine and photophobia.",
  "Doctor prescribed sumatriptan for acute migraine treatment.",
  "Patient has trigeminal neuralgia managed with carbamazepine.",

  // Neuromuscular
  "Patient diagnosed with Guillain Barre syndrome with ascending weakness.",
  "IVIG started for Guillain Barre syndrome treatment.",
  "Patient with myasthenia gravis on pyridostigmine.",
  "Plasma exchange initiated for myasthenic crisis.",
  "EMG and nerve conduction studies showed demyelinating neuropathy.",

  // ─── GENERAL MEDICINE ─────────────────────────────────────────────
  // Respiratory
  "Patient admitted with community acquired pneumonia and fever.",
  "Doctor prescribed ceftriaxone and azithromycin for pneumonia.",
  "Chest infection treated with augmentin and physiotherapy.",
  "Patient with COPD exacerbation requiring oxygen therapy.",
  "Nebulized salbutamol and ipratropium given for bronchospasm.",
  "Patient developed respiratory failure and required intubation.",
  "Pleural effusion drained with ultrasound guided thoracentesis.",

  // Cardiac
  "Patient admitted with acute coronary syndrome and chest pain.",
  "Troponin elevated consistent with non ST elevation MI.",
  "Doctor prescribed aspirin metoprolol and atorvastatin.",
  "Patient with atrial fibrillation on apixaban for anticoagulation.",
  "Heart failure managed with furosemide and spironolactone.",
  "ECG showed ST elevation in leads V1 through V4.",
  "Patient with hypertension on amlodipine and ramipril.",

  // Renal
  "Patient developed acute kidney injury with rising creatinine.",
  "Doctor ordered renal ultrasound for hydronephrosis.",
  "Potassium elevated at 6.2 requiring urgent treatment.",
  "Calcium gluconate and insulin given for hyperkalemia.",
  "Patient with chronic kidney disease on dialysis three times weekly.",
  "Sodium 128 consistent with hyponatremia fluid restrict ordered.",

  // GI / Hepato
  "Patient with upper GI bleed and hematemesis.",
  "Doctor ordered urgent endoscopy for GI bleeding.",
  "Pantoprazole infusion started for peptic ulcer bleed.",
  "Patient with liver cirrhosis and ascites requiring paracentesis.",
  "Lactulose prescribed for hepatic encephalopathy.",

  // Infectious
  "Blood cultures positive for gram positive cocci in clusters.",
  "Vancomycin started for suspected MRSA bacteremia.",
  "Patient with urosepsis started on meropenem.",
  "Cellulitis treated with flucloxacillin and elevation.",
  "COVID positive patient isolated with droplet precautions.",

  // Endocrine
  "Patient with diabetic ketoacidosis on insulin infusion.",
  "HbA1c 9.2 indicating poor glycemic control.",
  "Doctor adjusted insulin glargine dose for diabetes management.",
  "Patient with thyroid storm treated with propylthiouracil.",
  "Levothyroxine started for hypothyroidism.",

  // Hematology
  "Patient with DVT started on enoxaparin bridging to warfarin.",
  "INR 4.5 warfarin held and vitamin K administered.",
  "Hemoglobin 6.2 requiring packed red cell transfusion.",
  "Patient with pancytopenia bone marrow biopsy ordered.",

  // ─── WARD CENSUS PATTERNS ─────────────────────────────────────────
  "Ward 20 bed 12 patient Hassan admitted with stroke.",
  "Ward 20 bed 14 patient Abdullah diagnosis chest infection.",
  "Ward 18 bed 3 patient Fatima admitted with seizure.",
  "Ward 18 bed 7 patient Ahmad diagnosis diabetes mellitus.",
  "Doctor Bader assigned to ward 20 neurology patients.",
  "Doctor Noura covering ward 18 general medicine patients.",
  "Patient Khalid transferred from ER to ward 20 bed 5.",
  "Patient Maryam discharged from ward 18 bed 2.",
  "Triage red patient requires immediate transfer to ICU.",
  "Triage yellow patient stable for ward level care.",
  "Patient on stretcher transport cannot ambulate.",
  "Patient ambulatory no special transport needs.",
  "Isolation contact precautions for MRSA positive patient.",
  "Patient is full code for resuscitation.",
  "Patient is DNR comfort care measures only.",
  "Allergy NKDA no known drug allergies documented.",
  "Patient allergic to penicillin alternative antibiotic needed.",

  // ─── CLINICAL EXAMINATION ─────────────────────────────────────────
  "Neurological examination showed right sided hemiparesis.",
  "Cranial nerves examination pupils equal and reactive to light.",
  "Motor examination power four out of five in left upper limb.",
  "Sensory examination intact to pinprick and vibration.",
  "Reflexes examination showed brisk reflexes bilaterally.",
  "Babinski sign positive upgoing plantar on the right.",
  "Cerebellar examination showed dysmetria on finger nose test.",
  "Gait examination patient ataxic and unsteady requiring assistance.",
  "Mental status examination patient alert and oriented.",
  "Cardiovascular examination showed irregular pulse with murmur.",
  "Respiratory examination bilateral crackles at lung bases.",
  "Abdominal examination soft non tender no organomegaly.",

  // ─── LAB VALUES IN CONTEXT ────────────────────────────────────────
  "Potassium 4.5 within normal range no intervention needed.",
  "Sodium 135 within normal limits.",
  "Glucose 12.3 elevated will increase insulin dose.",
  "Hemoglobin 10.2 stable no transfusion required.",
  "WBC 15.4 elevated consistent with infection.",
  "Creatinine 180 elevated suggests acute kidney injury.",
  "CRP 85 elevated indicating active inflammation.",
  "Lactate 3.2 elevated concerning for sepsis.",
  "Troponin 450 elevated rule out myocardial infarction.",
  "INR 2.8 within therapeutic range for warfarin.",
  "HbA1c 7.2 moderate glycemic control.",
  "Albumin 28 low indicating malnutrition.",
  "Platelets 45 low thrombocytopenia workup needed.",
  "Bilirubin 35 elevated suggesting liver dysfunction.",
  "TSH 0.05 low consistent with hyperthyroidism.",

  // ─── MEDICATION PRESCRIBING PATTERNS ──────────────────────────────
  "Prescribed levetiracetam 500mg twice daily for seizure prophylaxis.",
  "Started metformin 500mg twice daily for diabetes.",
  "Furosemide 40mg once daily for fluid overload.",
  "Omeprazole 20mg once daily for gastric protection.",
  "Enoxaparin 40mg subcutaneous for DVT prophylaxis.",
  "Dexamethasone 4mg four times daily for cerebral edema.",
  "Vancomycin 1g intravenous twice daily for MRSA.",
  "Ceftriaxone 2g intravenous once daily for pneumonia.",
  "Insulin glargine 20 units at bedtime for diabetes.",
  "Paracetamol 1g four times daily for pain and fever.",
  "Morphine 5mg subcutaneous as needed for severe pain.",
  "Amlodipine 10mg once daily for hypertension.",
  "Atorvastatin 40mg once daily for hyperlipidemia.",
  "Aspirin 100mg once daily for stroke prevention.",
];

/**
 * Seed the core ShifuEngine with medical domain knowledge.
 * This pre-builds resonance between synonyms and related terms.
 */
function seedEngine(engine) {
  const result = engine.feedBatch(MEDICAL_CORPUS);
  return {
    ...result,
    corpus_size: MEDICAL_CORPUS.length,
  };
}

/**
 * Generate ward census training sentences.
 * Creates synthetic but realistic ward data to strengthen context chains.
 */
function generateWardSentences(count = 50) {
  const names = ['Hassan', 'Abdullah', 'Fatima', 'Ahmad', 'Maryam', 'Khalid',
    'Bader', 'Noura', 'Sara', 'Omar', 'Ibrahim', 'Zahra', 'Faisal', 'Huda'];
  const diagnoses = ['stroke', 'seizure', 'chest infection', 'pneumonia', 'heart failure',
    'diabetes', 'hypertension', 'COPD exacerbation', 'acute kidney injury', 'DVT',
    'atrial fibrillation', 'meningitis', 'Guillain Barre', 'sepsis', 'GI bleed'];
  const doctors = ['Bader', 'Noura', 'Hisham', 'Saleh', 'Adel', 'Hassan'];
  const wards = ['18', '20', '22', '24'];

  const sentences = [];
  const rand = (arr) => arr[Math.floor(Math.random() * arr.length)];

  for (let i = 0; i < count; i++) {
    const name = rand(names);
    const dx = rand(diagnoses);
    const doc = rand(doctors);
    const ward = rand(wards);
    const bed = Math.floor(Math.random() * 20) + 1;

    sentences.push(`Ward ${ward} bed ${bed} patient ${name} diagnosis ${dx} doctor ${doc}.`);
    sentences.push(`Doctor ${doc} prescribed medication for patient ${name} with ${dx}.`);
    sentences.push(`Patient ${name} admitted to ward ${ward} for ${dx} management.`);
  }
  return sentences;
}

/**
 * Full seeding: medical corpus + ward patterns.
 * This gives the engine a strong foundation before it sees real data.
 */
function fullSeed(engine, passes = 3) {
  let totalSentences = 0, totalTokens = 0;

  // Multiple passes over the medical corpus to build strong transitions
  for (let i = 0; i < passes; i++) {
    const r = engine.feedBatch(MEDICAL_CORPUS);
    totalSentences += r.sentences;
    totalTokens += r.tokens;
  }

  // Ward census patterns
  const wardSentences = generateWardSentences(100);
  const wardResult = engine.feedBatch(wardSentences);
  totalSentences += wardResult.sentences;
  totalTokens += wardResult.tokens;

  return { totalSentences, totalTokens };
}

module.exports = { MEDICAL_CORPUS, seedEngine, generateWardSentences, fullSeed };
