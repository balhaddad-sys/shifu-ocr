#!/usr/bin/env python3
"""
Shifu Language Teacher — Overnight Fluency Training
====================================================

Builds a massive language corpus and feeds it into ShifuEngine to teach
Shifu how English works: grammar, syntax, conversation, medical language,
and natural speech patterns.

Designed to run overnight. Generates corpus → feeds to Node.js engine.

Phases:
  1. Download public domain literature (Gutenberg)
  2. Generate medical language at scale
  3. Generate conversational patterns
  4. Generate syntactic drills (grammar structures)
  5. Generate clinical documentation patterns
  6. Generate numerical/lab patterns
  7. Feed everything through ShifuEngine via Node.js subprocess

Usage:
    python shifu_ocr/teach_language.py                  # Full overnight run
    python shifu_ocr/teach_language.py --generate-only   # Just build corpus
    python shifu_ocr/teach_language.py --feed-only        # Just feed existing corpus
    python shifu_ocr/teach_language.py --quick             # Quick test (10k sentences)
"""

import argparse
import os
import random
import subprocess
import sys
import time
import urllib.request
import urllib.error
import re
import json
from pathlib import Path


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
CORPUS_DIR = os.path.join(PROJECT_DIR, 'corpus_data')
CORPUS_FILE = os.path.join(CORPUS_DIR, 'shifu_language_corpus.txt')
FEEDER_SCRIPT = os.path.join(PROJECT_DIR, 'learning', 'feed_corpus.js')


# =============================================================================
# UTILITY
# =============================================================================

def pick(pool):
    """Pick a random element."""
    return random.choice(pool)

def pick_n(pool, n):
    """Pick n random elements (with replacement)."""
    return [random.choice(pool) for _ in range(n)]

def vary(sentence):
    """Slight variation: randomly capitalize or add trailing period."""
    if random.random() < 0.3:
        sentence = sentence.capitalize()
    if random.random() < 0.7 and not sentence.endswith('.'):
        sentence += '.'
    return sentence


# =============================================================================
# WORD POOLS (building blocks, not hardcoded sentences)
# =============================================================================

# ── People ──
FIRST_NAMES = [
    'Hassan', 'Abdullah', 'Fatima', 'Ahmad', 'Maryam', 'Khalid', 'Bader',
    'Noura', 'Sara', 'Omar', 'Ibrahim', 'Zahra', 'Faisal', 'Huda', 'Ali',
    'Mohammed', 'Saleh', 'Hisham', 'Adel', 'Reem', 'Dalal', 'Nasser',
    'Youssef', 'Layla', 'Khaled', 'Dana', 'Saud', 'Meshaal', 'Fahad',
    'Latifa', 'Hamad', 'Aisha', 'Turki', 'Nora', 'Jassim', 'Amina',
    'John', 'Sarah', 'Michael', 'Jennifer', 'David', 'Maria', 'Robert',
    'Lisa', 'James', 'Emily', 'William', 'Anna', 'Thomas', 'Susan',
]

TITLES = ['Doctor', 'Nurse', 'Professor', 'Consultant', 'Physician', 'Specialist']

# ── Medical ──
DIAGNOSES = [
    'stroke', 'seizure', 'pneumonia', 'chest infection', 'heart failure',
    'diabetes mellitus', 'hypertension', 'COPD exacerbation', 'acute kidney injury',
    'deep vein thrombosis', 'atrial fibrillation', 'meningitis', 'sepsis',
    'gastrointestinal bleed', 'pulmonary embolism', 'myocardial infarction',
    'urinary tract infection', 'cellulitis', 'diabetic ketoacidosis',
    'hepatic encephalopathy', 'pancreatitis', 'cholecystitis', 'appendicitis',
    'bowel obstruction', 'asthma exacerbation', 'anemia', 'thrombocytopenia',
    'hypoglycemia', 'hyperkalemia', 'hyponatremia', 'dehydration',
    'congestive heart failure', 'acute coronary syndrome', 'pleural effusion',
    'respiratory failure', 'renal failure', 'liver cirrhosis', 'ascites',
    'Guillain Barre syndrome', 'myasthenia gravis', 'Parkinson disease',
    'multiple sclerosis', 'epilepsy', 'migraine', 'vertigo', 'syncope',
]

MEDICATIONS = [
    'paracetamol', 'ibuprofen', 'aspirin', 'metformin', 'insulin',
    'amlodipine', 'ramipril', 'atorvastatin', 'omeprazole', 'pantoprazole',
    'furosemide', 'spironolactone', 'metoprolol', 'bisoprolol', 'warfarin',
    'apixaban', 'enoxaparin', 'heparin', 'clopidogrel', 'ticagrelor',
    'levetiracetam', 'phenytoin', 'carbamazepine', 'valproate', 'lacosamide',
    'amoxicillin', 'augmentin', 'ceftriaxone', 'meropenem', 'vancomycin',
    'azithromycin', 'ciprofloxacin', 'metronidazole', 'flucloxacillin',
    'salbutamol', 'ipratropium', 'prednisolone', 'dexamethasone',
    'morphine', 'tramadol', 'codeine', 'gabapentin', 'pregabalin',
    'sertraline', 'fluoxetine', 'diazepam', 'lorazepam', 'haloperidol',
    'levothyroxine', 'hydrocortisone', 'insulin glargine', 'insulin aspart',
    'lactulose', 'senna', 'ondansetron', 'cyclizine', 'domperidone',
    'citalopram', 'amitriptyline', 'propranolol', 'lisinopril', 'losartan',
]

BODY_PARTS = [
    'head', 'brain', 'chest', 'abdomen', 'heart', 'lungs', 'liver', 'kidneys',
    'spine', 'pelvis', 'arm', 'leg', 'hand', 'foot', 'neck', 'throat',
    'eyes', 'ears', 'left upper limb', 'right lower limb', 'bilateral legs',
]

SYMPTOMS = [
    'pain', 'fever', 'weakness', 'numbness', 'tingling', 'swelling',
    'shortness of breath', 'chest tightness', 'dizziness', 'nausea',
    'vomiting', 'diarrhea', 'constipation', 'headache', 'confusion',
    'altered consciousness', 'blurred vision', 'difficulty walking',
    'difficulty swallowing', 'weight loss', 'fatigue', 'cough',
    'blood in urine', 'blood in stool', 'loss of appetite', 'palpitations',
]

LAB_TESTS = [
    'potassium', 'sodium', 'glucose', 'hemoglobin', 'WBC', 'platelets',
    'creatinine', 'urea', 'CRP', 'lactate', 'troponin', 'INR',
    'HbA1c', 'albumin', 'bilirubin', 'TSH', 'calcium', 'magnesium',
    'phosphate', 'ALT', 'AST', 'alkaline phosphatase', 'GGT',
    'prothrombin time', 'APTT', 'D-dimer', 'ferritin', 'iron',
    'vitamin D', 'B12', 'folate', 'ammonia', 'blood gas', 'pH',
]

IMAGING = [
    'CT head', 'CT chest', 'CT abdomen', 'MRI brain', 'MRI spine',
    'chest X-ray', 'abdominal X-ray', 'ultrasound abdomen', 'ultrasound kidneys',
    'echocardiography', 'CT angiography', 'MRI with contrast',
    'CT pulmonary angiogram', 'Doppler ultrasound', 'PET scan',
]

PROCEDURES = [
    'blood transfusion', 'lumbar puncture', 'thoracentesis', 'paracentesis',
    'central line insertion', 'intubation', 'bronchoscopy', 'endoscopy',
    'colonoscopy', 'dialysis', 'cardioversion', 'defibrillation',
    'wound debridement', 'chest tube insertion', 'urinary catheter insertion',
]

# ── General English ──
ADJECTIVES = [
    'good', 'better', 'best', 'important', 'large', 'small', 'new', 'old',
    'interesting', 'beautiful', 'difficult', 'easy', 'fast', 'slow',
    'happy', 'sad', 'strong', 'weak', 'bright', 'dark', 'clear', 'simple',
    'complex', 'significant', 'effective', 'severe', 'mild', 'moderate',
    'stable', 'critical', 'essential', 'sufficient', 'appropriate',
]

ADVERBS = [
    'quickly', 'slowly', 'carefully', 'immediately', 'frequently',
    'always', 'never', 'sometimes', 'often', 'rarely', 'usually',
    'recently', 'currently', 'previously', 'subsequently', 'gradually',
    'significantly', 'approximately', 'particularly', 'effectively',
]

TRANSITION_WORDS = [
    'however', 'therefore', 'moreover', 'furthermore', 'additionally',
    'consequently', 'nevertheless', 'meanwhile', 'subsequently',
    'alternatively', 'similarly', 'accordingly', 'specifically',
]

PREPOSITIONS = [
    'in', 'on', 'at', 'by', 'for', 'with', 'from', 'to', 'into',
    'through', 'during', 'before', 'after', 'between', 'under', 'above',
    'about', 'around', 'without', 'within', 'across', 'along', 'behind',
]

CONJUNCTIONS = ['and', 'but', 'or', 'so', 'because', 'although', 'while', 'if', 'when', 'since', 'unless']

COMMON_NOUNS = [
    'time', 'year', 'people', 'way', 'day', 'man', 'woman', 'child',
    'world', 'life', 'hand', 'part', 'place', 'case', 'week', 'company',
    'system', 'program', 'question', 'work', 'government', 'number',
    'night', 'point', 'home', 'water', 'room', 'mother', 'area',
    'money', 'story', 'fact', 'month', 'lot', 'right', 'study',
    'book', 'eye', 'job', 'word', 'business', 'issue', 'side',
    'kind', 'head', 'house', 'service', 'friend', 'father', 'power',
    'hour', 'game', 'line', 'end', 'member', 'law', 'car', 'city',
    'community', 'name', 'president', 'team', 'minute', 'idea',
    'body', 'information', 'back', 'parent', 'face', 'others',
    'level', 'office', 'door', 'health', 'person', 'art', 'war',
    'history', 'party', 'result', 'change', 'morning', 'reason',
    'research', 'girl', 'guy', 'moment', 'air', 'teacher', 'force',
    'education', 'foot', 'boy', 'age', 'policy', 'process', 'music',
    'market', 'sense', 'product', 'effect', 'class', 'piece',
    'land', 'group', 'development', 'role', 'field', 'effort',
    'use', 'report', 'difference', 'interest', 'order', 'mind',
    'care', 'action', 'plan', 'form', 'value', 'need',
]

VERBS_PRESENT = [
    'is', 'are', 'has', 'have', 'does', 'goes', 'makes', 'takes',
    'comes', 'sees', 'knows', 'thinks', 'wants', 'gives', 'says',
    'finds', 'tells', 'asks', 'works', 'seems', 'feels', 'tries',
    'leaves', 'calls', 'needs', 'becomes', 'keeps', 'begins', 'shows',
    'hears', 'plays', 'runs', 'moves', 'lives', 'believes',
]

VERBS_PAST = [
    'was', 'were', 'had', 'did', 'went', 'made', 'took', 'came',
    'saw', 'knew', 'thought', 'wanted', 'gave', 'said', 'found',
    'told', 'asked', 'worked', 'seemed', 'felt', 'tried', 'left',
    'called', 'needed', 'became', 'kept', 'began', 'showed', 'heard',
    'played', 'ran', 'moved', 'lived', 'believed', 'brought',
    'happened', 'reached', 'remained', 'suggested', 'created',
    'received', 'appeared', 'considered', 'reported', 'developed',
]


# =============================================================================
# PHASE 1: PUBLIC DOMAIN LITERATURE (Gutenberg)
# =============================================================================

GUTENBERG_BOOKS = [
    # (ID, Title) — all public domain
    (1342, 'Pride and Prejudice'),
    (11, 'Alice in Wonderland'),
    (1661, 'Sherlock Holmes'),
    (84, 'Frankenstein'),
    (2701, 'Moby Dick'),
    (1952, 'The Yellow Wallpaper'),
    (345, 'Dracula'),
    (98, 'A Tale of Two Cities'),
    (2542, 'A Doll\'s House'),
    (1232, 'The Prince'),
    (76, 'Adventures of Huckleberry Finn'),
    (5200, 'Metamorphosis'),
    (844, 'The Importance of Being Earnest'),
    (174, 'The Picture of Dorian Gray'),
    (1080, 'A Modest Proposal'),
    (16328, 'Beowulf'),
    (43, 'The Strange Case of Dr Jekyll and Mr Hyde'),
    (514, 'Little Women'),
    (1400, 'Great Expectations'),
    (2591, 'Grimm Fairy Tales'),
]

def download_gutenberg(book_id, title, max_chars=500000):
    """Download a Gutenberg book as plain text."""
    urls = [
        f'https://www.gutenberg.org/files/{book_id}/{book_id}-0.txt',
        f'https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt',
        f'https://www.gutenberg.org/ebooks/{book_id}.txt.utf-8',
    ]

    for url in urls:
        try:
            req = urllib.request.Request(url, headers={
                'User-Agent': 'ShifuOCR/1.0 (research; bader@shifu.dev)',
                'Accept': 'text/plain',
            })
            with urllib.request.urlopen(req, timeout=15) as resp:
                text = resp.read().decode('utf-8', errors='replace')
                if len(text) > 1000:
                    # Strip Gutenberg header/footer
                    start = text.find('*** START')
                    if start > 0:
                        text = text[text.find('\n', start) + 1:]
                    end = text.find('*** END')
                    if end > 0:
                        text = text[:end]
                    return text[:max_chars]
        except Exception:
            continue
    return None


def extract_sentences(text, min_len=30, max_len=200):
    """Extract clean sentences from raw text."""
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    # Split on sentence boundaries
    raw = re.split(r'(?<=[.!?])\s+', text)
    sentences = []
    for s in raw:
        s = s.strip()
        # Filter: reasonable length, mostly letters, no weird chars
        if len(s) < min_len or len(s) > max_len:
            continue
        if sum(c.isalpha() or c.isspace() for c in s) / max(len(s), 1) < 0.7:
            continue
        # Skip lines with too many caps (headers, TOC)
        if sum(c.isupper() for c in s) / max(len(s), 1) > 0.5:
            continue
        sentences.append(s)
    return sentences


def download_literature(target_sentences=50000):
    """Phase 1: Download Gutenberg books and extract sentences."""
    print(f"\n{'='*60}")
    print("  PHASE 1 — Public Domain Literature")
    print(f"{'='*60}")

    all_sentences = []
    for book_id, title in GUTENBERG_BOOKS:
        if len(all_sentences) >= target_sentences:
            break
        print(f"  Downloading: {title} (#{book_id})...", end=' ', flush=True)
        text = download_gutenberg(book_id, title)
        if text:
            sents = extract_sentences(text)
            all_sentences.extend(sents)
            print(f"{len(sents)} sentences")
        else:
            print("FAILED (skipping)")
        time.sleep(1)  # Be polite

    print(f"\n  Literature total: {len(all_sentences)} sentences")
    return all_sentences


# =============================================================================
# PHASE 2: MEDICAL LANGUAGE GENERATION
# =============================================================================

def generate_medical_sentences(n=20000):
    """Phase 2: Generate medical sentences from word pools."""
    print(f"\n{'='*60}")
    print(f"  PHASE 2 — Medical Language ({n} sentences)")
    print(f"{'='*60}")

    sentences = []

    # Admission patterns
    for _ in range(n // 10):
        name = pick(FIRST_NAMES)
        dx = pick(DIAGNOSES)
        symptom = pick(SYMPTOMS)
        sentences.append(vary(f"Patient {name} was admitted with {dx} presenting with {symptom}"))
        sentences.append(vary(f"{pick(TITLES)} examined {name} and diagnosed {dx}"))
        sentences.append(vary(f"{name} presented to the emergency department with {symptom} and was found to have {dx}"))

    # Medication patterns
    for _ in range(n // 10):
        med = pick(MEDICATIONS)
        dx = pick(DIAGNOSES)
        dose = f"{random.choice([5,10,20,25,40,50,100,200,250,500,1000])}mg"
        freq = random.choice(['once daily', 'twice daily', 'three times daily', 'four times daily', 'as needed', 'at bedtime'])
        route = random.choice(['oral', 'intravenous', 'subcutaneous', 'intramuscular', 'topical'])
        sentences.append(vary(f"Prescribed {med} {dose} {freq} for {dx}"))
        sentences.append(vary(f"{pick(TITLES)} started {med} {route} for management of {dx}"))
        sentences.append(vary(f"Patient commenced on {med} {dose} {route} {freq}"))

    # Lab result patterns
    for _ in range(n // 10):
        test = pick(LAB_TESTS)
        val = round(random.uniform(0.1, 500), 1)
        status = random.choice(['elevated', 'low', 'within normal range', 'critically high', 'critically low', 'stable', 'improving', 'worsening'])
        action = random.choice(['no intervention needed', 'will repeat tomorrow', 'urgent treatment required', 'continue monitoring', 'dose adjustment needed', 'specialist consultation requested'])
        sentences.append(vary(f"{test} {val} {status} {action}"))
        sentences.append(vary(f"Lab results show {test} of {val} which is {status}"))

    # Imaging patterns
    for _ in range(n // 15):
        img = pick(IMAGING)
        finding = random.choice([
            'showed no acute abnormality', 'revealed consolidation in the right lower lobe',
            'demonstrated a mass lesion', 'was unremarkable', 'showed mild cardiomegaly',
            'confirmed the clinical diagnosis', 'showed bilateral pleural effusions',
            'demonstrated no evidence of pulmonary embolism', 'revealed a fracture',
            'showed diffuse cerebral edema', 'demonstrated acute ischemic changes',
        ])
        sentences.append(vary(f"{img} {finding}"))

    # Examination patterns
    for _ in range(n // 10):
        system = random.choice(['Neurological', 'Cardiovascular', 'Respiratory', 'Abdominal', 'Musculoskeletal'])
        finding = random.choice([
            'was within normal limits', 'showed no abnormalities',
            'revealed bilateral crackles at the bases', 'demonstrated reduced power in the left limbs',
            'showed irregular pulse', 'was soft and non-tender', 'showed brisk reflexes bilaterally',
            'demonstrated decreased air entry on the right', 'was unremarkable',
            'showed peripheral edema', 'revealed hepatomegaly', 'showed a positive Babinski sign',
        ])
        sentences.append(vary(f"{system} examination {finding}"))

    # Progress notes
    for _ in range(n // 10):
        name = pick(FIRST_NAMES)
        status = random.choice([
            'is clinically improving', 'remains stable', 'has deteriorated overnight',
            'is ready for discharge', 'requires continued monitoring', 'is responding well to treatment',
            'developed new symptoms overnight', 'is being transferred to ICU',
            'tolerated the procedure well', 'is ambulating independently',
        ])
        plan = random.choice([
            'continue current medications', 'plan for discharge tomorrow',
            'escalate antibiotics', 'request specialist review', 'repeat blood tests in the morning',
            'order additional imaging', 'start physiotherapy', 'adjust insulin regimen',
            'prepare for surgery', 'arrange follow-up in clinic',
        ])
        sentences.append(vary(f"Patient {name} {status}"))
        sentences.append(vary(f"Plan for {name} is to {plan}"))

    # Ward census
    for _ in range(n // 10):
        ward = random.choice(['18', '20', '22', '24', '26', '30', '32'])
        bed = random.randint(1, 20)
        name = pick(FIRST_NAMES)
        dx = pick(DIAGNOSES)
        doc = pick(FIRST_NAMES)
        sentences.append(vary(f"Ward {ward} bed {bed} patient {name} diagnosis {dx} doctor {doc}"))
        sentences.append(vary(f"Patient {name} in ward {ward} bed {bed} under the care of Doctor {doc}"))

    # Procedure documentation
    for _ in range(n // 15):
        proc = pick(PROCEDURES)
        name = pick(FIRST_NAMES)
        outcome = random.choice(['was performed successfully', 'completed without complications',
                                   'was tolerated well', 'resulted in improvement', 'was uneventful'])
        sentences.append(vary(f"{proc} for patient {name} {outcome}"))

    # Nursing documentation
    for _ in range(n // 10):
        name = pick(FIRST_NAMES)
        obs = random.choice([
            'vital signs stable', 'temperature 37.2 degrees', 'blood pressure 130 over 80',
            'heart rate 78 regular', 'oxygen saturation 97 percent on room air',
            'respiratory rate 18 breaths per minute', 'urine output adequate',
            'intake and output balanced', 'pain score 3 out of 10',
            'GCS 15 alert and oriented', 'wound dressing clean and dry',
            'IV site clean no signs of infection', 'fall risk assessment completed',
        ])
        sentences.append(vary(f"Nursing assessment for {name}: {obs}"))

    random.shuffle(sentences)
    print(f"  Generated: {len(sentences)} medical sentences")
    return sentences[:n]


# =============================================================================
# PHASE 3: CONVERSATIONAL PATTERNS
# =============================================================================

def generate_conversational(n=10000):
    """Phase 3: Generate conversational/dialogue patterns."""
    print(f"\n{'='*60}")
    print(f"  PHASE 3 — Conversational Patterns ({n})")
    print(f"{'='*60}")

    sentences = []

    # Question patterns (teaches question structure)
    q_templates = [
        "What is the {noun} of the {noun2}",
        "How does the {noun} affect the {noun2}",
        "Why did the {noun} change {adverb}",
        "When will the {noun} be {adjective}",
        "Where is the {noun} located",
        "Who is responsible for the {noun}",
        "Can you explain the {noun} to me",
        "What happened to the {noun} {adverb}",
        "How many {noun} are there in the {noun2}",
        "Is the {noun} {adjective} enough",
        "Does the patient have any {noun}",
        "What medication is the patient taking for the {noun}",
        "When was the last {noun} performed",
        "Has the {noun} been reviewed by the doctor",
        "Should we increase the {noun} dose",
    ]
    for _ in range(n // 5):
        tmpl = pick(q_templates)
        s = tmpl.format(
            noun=pick(COMMON_NOUNS), noun2=pick(COMMON_NOUNS),
            adjective=pick(ADJECTIVES), adverb=pick(ADVERBS),
        )
        sentences.append(s + '?')

    # Request patterns
    req_templates = [
        "Please check the {noun} for patient {name}",
        "Could you review the {noun} results",
        "I need help with the {noun} assessment",
        "Please document the {noun} findings",
        "Can we schedule a {noun} for tomorrow",
        "Please notify the doctor about the {noun}",
        "We need to prepare {noun} for the patient",
        "Please confirm the {noun} with the pharmacy",
        "I would like to request a {noun} consultation",
        "Please arrange for {noun} as soon as possible",
    ]
    for _ in range(n // 5):
        tmpl = pick(req_templates)
        noun = random.choice(COMMON_NOUNS + LAB_TESTS + IMAGING)
        s = tmpl.format(noun=noun, name=pick(FIRST_NAMES))
        sentences.append(vary(s))

    # Statement patterns (teaches declarative structure)
    stmt_templates = [
        "The {noun} is {adjective} and needs {noun2}",
        "We {verb_past} the {noun} {adverb}",
        "The {noun} {verb_present} {adverb} in this {noun2}",
        "{name} {verb_past} the {noun} before the {noun2}",
        "The {adjective} {noun} was {verb_past} by the team",
        "After the {noun} we noticed the {noun2} was {adjective}",
        "Both the {noun} and the {noun2} are {adjective}",
        "Despite the {adjective} {noun} the {noun2} remained {adjective}",
    ]
    for _ in range(n // 5):
        tmpl = pick(stmt_templates)
        s = tmpl.format(
            noun=pick(COMMON_NOUNS), noun2=pick(COMMON_NOUNS),
            adjective=pick(ADJECTIVES), adverb=pick(ADVERBS),
            verb_past=pick(VERBS_PAST), verb_present=pick(VERBS_PRESENT),
            name=pick(FIRST_NAMES),
        )
        sentences.append(vary(s))

    # Compound sentences with conjunctions
    for _ in range(n // 5):
        s1 = f"The {pick(COMMON_NOUNS)} {pick(VERBS_PAST)} {pick(ADVERBS)}"
        conj = pick(CONJUNCTIONS)
        s2 = f"the {pick(COMMON_NOUNS)} {pick(VERBS_PAST)} {pick(ADVERBS)}"
        sentences.append(vary(f"{s1} {conj} {s2}"))

    # Transition sentences
    for _ in range(n // 5):
        tw = pick(TRANSITION_WORDS)
        s1 = f"the {pick(COMMON_NOUNS)} was {pick(ADJECTIVES)}"
        s2 = f"the {pick(COMMON_NOUNS)} {pick(VERBS_PAST)} {pick(ADVERBS)}"
        sentences.append(vary(f"{tw} {s1} and {s2}"))

    random.shuffle(sentences)
    print(f"  Generated: {len(sentences)} conversational patterns")
    return sentences[:n]


# =============================================================================
# PHASE 4: SYNTACTIC DRILLS
# =============================================================================

def generate_syntax_drills(n=15000):
    """Phase 4: Generate sentences that drill specific grammar structures."""
    print(f"\n{'='*60}")
    print(f"  PHASE 4 — Syntactic Drills ({n})")
    print(f"{'='*60}")

    sentences = []

    # Subject-Verb-Object (SVO)
    for _ in range(n // 8):
        subj = random.choice(['The patient', 'The doctor', 'The nurse', f'{pick(FIRST_NAMES)}',
                              'The team', 'The consultant', 'The family', 'We', 'They'])
        verb = pick(VERBS_PAST)
        obj = f"the {pick(COMMON_NOUNS)}"
        sentences.append(vary(f"{subj} {verb} {obj}"))

    # Passive voice
    for _ in range(n // 8):
        obj = f"The {pick(COMMON_NOUNS)}"
        verb = random.choice(['was examined', 'was prescribed', 'was performed', 'was documented',
                              'was reviewed', 'was ordered', 'was started', 'was completed',
                              'was administered', 'was confirmed', 'was reported', 'was observed'])
        by = f"by {pick(TITLES)} {pick(FIRST_NAMES)}"
        sentences.append(vary(f"{obj} {verb} {by}"))

    # Prepositional phrases
    for _ in range(n // 8):
        sentences.append(vary(f"The {pick(COMMON_NOUNS)} {pick(PREPOSITIONS)} the {pick(COMMON_NOUNS)} was {pick(ADJECTIVES)}"))

    # Conditional (if/then)
    for _ in range(n // 8):
        condition = f"the {pick(COMMON_NOUNS)} is {pick(ADJECTIVES)}"
        result = f"{pick(VERBS_PRESENT)} the {pick(COMMON_NOUNS)} {pick(ADVERBS)}"
        sentences.append(vary(f"If {condition} then {result}"))

    # Temporal (before/after/during/while)
    for _ in range(n // 8):
        temporal = random.choice(['before', 'after', 'during', 'while'])
        event1 = f"the {pick(COMMON_NOUNS)} {pick(VERBS_PAST)}"
        event2 = f"the {pick(COMMON_NOUNS)} {pick(VERBS_PAST)} {pick(ADVERBS)}"
        sentences.append(vary(f"{temporal} {event1} {event2}"))

    # Relative clauses (who/which/that)
    for _ in range(n // 8):
        rel = random.choice(['who', 'which', 'that'])
        sentences.append(vary(f"The {pick(COMMON_NOUNS)} {rel} {pick(VERBS_PAST)} the {pick(COMMON_NOUNS)} was {pick(ADJECTIVES)}"))

    # Comparatives and superlatives
    for _ in range(n // 8):
        adj = pick(ADJECTIVES)
        sentences.append(vary(f"The {pick(COMMON_NOUNS)} is more {adj} than the {pick(COMMON_NOUNS)}"))
        sentences.append(vary(f"This is the most {adj} {pick(COMMON_NOUNS)} in the {pick(COMMON_NOUNS)}"))

    # Lists and enumerations
    for _ in range(n // 8):
        items = pick_n(COMMON_NOUNS, random.randint(3, 5))
        listed = ', '.join(items[:-1]) + f' and {items[-1]}'
        sentences.append(vary(f"The {pick(COMMON_NOUNS)} includes {listed}"))

    random.shuffle(sentences)
    print(f"  Generated: {len(sentences)} syntactic drills")
    return sentences[:n]


# =============================================================================
# PHASE 5: COMMON ENGLISH PHRASES
# =============================================================================

def generate_common_phrases(n=10000):
    """Phase 5: Generate common English usage patterns."""
    print(f"\n{'='*60}")
    print(f"  PHASE 5 — Common English Phrases ({n})")
    print(f"{'='*60}")

    sentences = []

    # Time expressions
    time_exprs = ['in the morning', 'at night', 'during the afternoon', 'early in the day',
                  'late at night', 'throughout the week', 'over the weekend', 'every day',
                  'once a week', 'twice daily', 'three times a day', 'as needed']

    for _ in range(n // 5):
        sentences.append(vary(f"The {pick(COMMON_NOUNS)} happens {pick(time_exprs)}"))
        sentences.append(vary(f"{pick(FIRST_NAMES)} {pick(VERBS_PRESENT)} the {pick(COMMON_NOUNS)} {pick(time_exprs)}"))

    # Quantity expressions
    quantities = ['a few', 'several', 'many', 'all', 'most', 'some', 'none of the',
                  'each of the', 'both', 'either', 'neither', 'any', 'every']
    for _ in range(n // 5):
        sentences.append(vary(f"{pick(quantities)} {pick(COMMON_NOUNS)} {pick(VERBS_PAST)} {pick(ADVERBS)}"))

    # Cause and effect
    for _ in range(n // 5):
        cause = f"the {pick(COMMON_NOUNS)} {pick(VERBS_PAST)}"
        effect = f"the {pick(COMMON_NOUNS)} became {pick(ADJECTIVES)}"
        connector = random.choice(['because', 'since', 'as a result of', 'due to', 'owing to', 'therefore', 'consequently'])
        sentences.append(vary(f"{connector} {cause} {effect}"))

    # Opinion/judgment
    for _ in range(n // 5):
        opinion = random.choice(['I think', 'It seems', 'It appears', 'We believe', 'It is likely',
                                  'The evidence suggests', 'Based on the findings', 'In my opinion'])
        sentences.append(vary(f"{opinion} the {pick(COMMON_NOUNS)} is {pick(ADJECTIVES)}"))

    # Polite/formal language
    for _ in range(n // 5):
        polite = random.choice([
            f"Thank you for your {pick(COMMON_NOUNS)} and {pick(COMMON_NOUNS)}",
            f"I appreciate the {pick(ADJECTIVES)} {pick(COMMON_NOUNS)} you provided",
            f"Would it be possible to review the {pick(COMMON_NOUNS)}",
            f"I would be grateful if you could check the {pick(COMMON_NOUNS)}",
            f"Please let me know if you need any additional {pick(COMMON_NOUNS)}",
            f"I am writing to inform you about the {pick(COMMON_NOUNS)}",
        ])
        sentences.append(vary(polite))

    random.shuffle(sentences)
    print(f"  Generated: {len(sentences)} common phrases")
    return sentences[:n]


# =============================================================================
# MAIN ORCHESTRATOR
# =============================================================================

def build_corpus(quick=False):
    """Build the complete language corpus."""
    os.makedirs(CORPUS_DIR, exist_ok=True)

    scale = 0.1 if quick else 1.0
    all_sentences = []

    # Phase 1: Literature
    lit = download_literature(target_sentences=int(50000 * scale))
    all_sentences.extend(lit)

    # Phase 2: Medical
    med = generate_medical_sentences(n=int(20000 * scale))
    all_sentences.extend(med)

    # Phase 3: Conversational
    conv = generate_conversational(n=int(10000 * scale))
    all_sentences.extend(conv)

    # Phase 4: Syntax drills
    syn = generate_syntax_drills(n=int(15000 * scale))
    all_sentences.extend(syn)

    # Phase 5: Common phrases
    phr = generate_common_phrases(n=int(10000 * scale))
    all_sentences.extend(phr)

    # Final shuffle
    random.shuffle(all_sentences)

    # Write corpus
    with open(CORPUS_FILE, 'w', encoding='utf-8') as f:
        for s in all_sentences:
            f.write(s.strip() + '\n')

    size_mb = os.path.getsize(CORPUS_FILE) / (1024 * 1024)
    print(f"\n{'='*60}")
    print(f"  CORPUS BUILT")
    print(f"{'='*60}")
    print(f"  Total sentences: {len(all_sentences):,}")
    print(f"  File: {CORPUS_FILE}")
    print(f"  Size: {size_mb:.1f} MB")
    print(f"{'='*60}")

    return all_sentences


def feed_corpus(passes=3):
    """Feed the corpus into the Node.js ShifuEngine."""
    if not os.path.exists(CORPUS_FILE):
        print(f"ERROR: Corpus file not found: {CORPUS_FILE}")
        print("Run with --generate-only first, or without --feed-only")
        sys.exit(1)

    if not os.path.exists(FEEDER_SCRIPT):
        print(f"ERROR: Feeder script not found: {FEEDER_SCRIPT}")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"  FEEDING CORPUS TO SHIFU ENGINE")
    print(f"  Passes: {passes}")
    print(f"{'='*60}")

    cmd = ['node', FEEDER_SCRIPT, CORPUS_FILE, '--passes', str(passes)]
    print(f"  Running: {' '.join(cmd)}")
    print()

    result = subprocess.run(cmd, cwd=PROJECT_DIR)

    if result.returncode != 0:
        print(f"\nERROR: Feeder script failed with exit code {result.returncode}")
        sys.exit(result.returncode)


def main():
    parser = argparse.ArgumentParser(description='Shifu Language Teacher')
    parser.add_argument('--generate-only', action='store_true',
                        help='Only generate corpus, do not feed')
    parser.add_argument('--feed-only', action='store_true',
                        help='Only feed existing corpus')
    parser.add_argument('--quick', action='store_true',
                        help='Quick mode (~10K sentences instead of ~100K)')
    parser.add_argument('--passes', type=int, default=3,
                        help='Number of feeding passes (default: 3)')
    args = parser.parse_args()

    t0 = time.time()

    print(f"\n{'#'*60}")
    print(f"  SHIFU LANGUAGE TEACHER — Overnight Fluency Training")
    print(f"{'#'*60}")

    if not args.feed_only:
        build_corpus(quick=args.quick)

    if not args.generate_only:
        feed_corpus(passes=args.passes)

    elapsed = time.time() - t0
    hours = int(elapsed // 3600)
    minutes = int((elapsed % 3600) // 60)
    print(f"\n{'#'*60}")
    print(f"  COMPLETE — Total time: {hours}h {minutes}m")
    print(f"{'#'*60}\n")


if __name__ == '__main__':
    main()
