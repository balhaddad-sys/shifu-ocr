"""
Shifu-OCR: Shared Utilities
============================
Mod 1: Break circular imports. All shared primitives live here.
"""

import re
import numpy as np
from PIL import Image

CONFUSIONS = {
    ('0', 'o'): 0.1, ('1', 'l'): 0.2, ('1', 'i'): 0.2, ('l', 'i'): 0.2,
    ('5', 's'): 0.3, ('2', 'z'): 0.4, ('8', 'b'): 0.3, ('6', 'g'): 0.4,
    ('m', 'n'): 0.4, ('u', 'v'): 0.5, ('c', 'e'): 0.5, ('r', 'n'): 0.3,
    ('d', 'o'): 0.3, ('p', 'q'): 0.4, ('h', 'b'): 0.5,
    ('i', 't'): 0.1, ('4', 'a'): 0.1, ('3', 'b'): 0.2, ('0', 'n'): 0.2,
    ('6', 'e'): 0.1, ('g', 'q'): 0.3, ('c', 'x'): 0.2, ('m', 'w'): 0.3,
}


def ocr_distance(s1, s2):
    """Edit distance with OCR-confusion-aware substitution costs."""
    s1, s2 = s1.lower(), s2.lower()
    if len(s1) < len(s2):
        return ocr_distance(s2, s1)
    if len(s2) == 0:
        return float(len(s1))
    prev = [float(x) for x in range(len(s2) + 1)]
    for i, c1 in enumerate(s1):
        curr = [float(i + 1)]
        for j, c2 in enumerate(s2):
            sub = 0.0 if c1 == c2 else CONFUSIONS.get(tuple(sorted([c1, c2])), 1.0)
            curr.append(min(prev[j + 1] + 1, curr[j] + 1, prev[j] + sub))
        prev = curr
    return prev[-1]


def normalize_char(region, size=(48, 48)):
    """Normalize a binary character region to standard size.
    Uses LANCZOS for edge preservation (macula principle)."""
    img = Image.fromarray((region * 255).astype(np.uint8)).resize(size, Image.LANCZOS)
    return (np.array(img) > 127).astype(np.uint8)


VOCAB = {
    'exam': ['cranial', 'nerves', 'motor', 'sensory', 'reflexes', 'coordination',
             'gait', 'cerebellar', 'examination', 'neurological', 'mental', 'status'],
    'findings': ['intact', 'normal', 'abnormal', 'positive', 'negative', 'absent',
                 'present', 'mild', 'moderate', 'severe', 'acute', 'chronic',
                 'bilateral', 'unilateral', 'right', 'left', 'upper', 'lower'],
    'cranial': ['pupil', 'pupils', 'reactive', 'perrla', 'extraocular', 'movements',
                'facial', 'symmetry', 'nystagmus', 'diplopia', 'ptosis', 'papilledema'],
    'motor': ['power', 'tone', 'bulk', 'strength', 'weakness', 'spasticity', 'rigidity',
              'flaccid', 'atrophy', 'fasciculations', 'tremor', 'hemiparesis', 'paraparesis'],
    'reflexes': ['biceps', 'triceps', 'brachioradialis', 'patellar', 'achilles',
                 'plantar', 'babinski', 'clonus', 'brisk', 'upgoing', 'downgoing'],
    'meds': ['levetiracetam', 'valproate', 'carbamazepine', 'phenytoin', 'lamotrigine',
             'topiramate', 'gabapentin', 'pregabalin', 'levodopa', 'carbidopa',
             'aspirin', 'clopidogrel', 'warfarin', 'heparin', 'rivaroxaban', 'apixaban',
             'prednisolone', 'dexamethasone', 'paracetamol', 'omeprazole', 'metformin',
             'insulin', 'atorvastatin', 'furosemide', 'amlodipine', 'metoprolol'],
    'diagnoses': ['cva', 'stroke', 'ischemic', 'hemorrhagic', 'tia', 'seizure',
                  'epilepsy', 'meningitis', 'encephalitis', 'neuropathy', 'myopathy',
                  'pneumonia', 'uti', 'dvt', 'aki', 'hypernatremia', 'hyponatremia',
                  'cholecystitis', 'exacerbation', 'lvf', 'cap', 'urosepsis', 'occlusion'],
    'ward': ['ward', 'bed', 'room', 'icu', 'discharge', 'admitted', 'chronic',
             'active', 'male', 'female', 'patient', 'doctor', 'assigned',
             'evacuation', 'status', 'unassigned'],
    'names': ['bader', 'noura', 'bazzah', 'saleh', 'hisham', 'alathoub', 'zahra',
              'nawaf', 'hassan', 'jamal', 'ahmad', 'alessa', 'ali', 'hussain',
              'adel', 'abdullatif', 'abdullah', 'mohammad', 'mohammed', 'mneer',
              'osama', 'waseem', 'najaf', 'mahmoud', 'noor', 'islam'],
}

CLINICAL_WORDS = set()
for _words in VOCAB.values():
    CLINICAL_WORDS.update(w.lower() for w in _words)


def match_word(raw_word, max_dist=None):
    """Find best clinical vocabulary match for a raw OCR word."""
    if max_dist is None:
        max_dist = max(len(raw_word) * 0.5, 2.0)
    raw_lower = raw_word.lower().strip()
    if not raw_lower:
        return raw_word, 0, 'empty'
    if raw_lower in CLINICAL_WORDS:
        return raw_word, 0, 'exact'
    if re.match(r'^[\d.,/\-]+$', raw_lower):
        return raw_word, 0, 'number'
    best = None
    best_dist = float('inf')
    for word in CLINICAL_WORDS:
        if abs(len(word) - len(raw_lower)) > 3:
            continue
        d = ocr_distance(raw_lower, word)
        if d < best_dist:
            best_dist = d
            best = word
    if best and best_dist <= max_dist:
        if raw_word[0].isupper() and len(best) > 0:
            corrected = best[0].upper() + best[1:]
        else:
            corrected = best
        return corrected, best_dist, 'corrected'
    return raw_word, best_dist if best else float('inf'), 'unknown'
