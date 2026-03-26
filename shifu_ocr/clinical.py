"""
Shifu-OCR: Clinical Context Engine
Post-processing with medical domain priors.
"""

import re
import numpy as np
from collections import defaultdict


def levenshtein(s1, s2):
    if len(s1) < len(s2): return levenshtein(s2, s1)
    if len(s2) == 0: return len(s1)
    prev = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr = [i + 1]
        for j, c2 in enumerate(s2):
            curr.append(min(prev[j+1]+1, curr[j]+1, prev[j]+(c1!=c2)))
        prev = curr
    return prev[-1]


# OCR confusion costs (topology-predicted)
CONFUSIONS = {
    ('0','o'):0.1, ('1','l'):0.2, ('1','i'):0.2, ('l','i'):0.2,
    ('5','s'):0.4, ('2','z'):0.4, ('8','b'):0.3, ('6','g'):0.4,
    ('m','n'):0.4, ('u','v'):0.5, ('c','e'):0.5, ('h','b'):0.5,
    ('r','n'):0.3, ('d','o'):0.3, ('p','q'):0.4,
}

def ocr_distance(s1, s2):
    s1, s2 = s1.lower(), s2.lower()
    if len(s1) < len(s2): return ocr_distance(s2, s1)
    if len(s2) == 0: return float(len(s1))
    prev = [float(x) for x in range(len(s2)+1)]
    for i, c1 in enumerate(s1):
        curr = [float(i+1)]
        for j, c2 in enumerate(s2):
            if c1 == c2:
                sub = 0.0
            else:
                pair = tuple(sorted([c1, c2]))
                sub = CONFUSIONS.get(pair, 1.0)
            curr.append(min(prev[j+1]+1, curr[j]+1, prev[j]+sub))
        prev = curr
    return prev[-1]


# Medical vocabulary (neurology-focused)
NEURO_VOCAB = {
    'exam_sections': [
        'mental status', 'cranial nerves', 'motor', 'sensory',
        'reflexes', 'coordination', 'gait', 'cerebellar',
    ],
    'cranial': [
        'intact','normal','pupil','pupils','reactive','perrla',
        'extraocular','movements','facial','symmetry','symmetric',
        'nystagmus','diplopia','ptosis','visual','fields','papilledema',
        'optic','trigeminal','abducens','oculomotor','vestibulocochlear',
        'glossopharyngeal','vagus','accessory','hypoglossal',
    ],
    'motor': [
        'power','tone','bulk','strength','weakness','spasticity',
        'rigidity','flaccid','atrophy','fasciculations','tremor',
        'mrc','grade','proximal','distal','upper','lower','limb','limbs',
        'hemiparesis','paraparesis','quadriparesis',
    ],
    'sensory': [
        'sensation','light','touch','pinprick','vibration',
        'proprioception','temperature','numbness','tingling',
        'paresthesia','dermatomal','glove','stocking','intact','reduced',
    ],
    'reflexes': [
        'biceps','triceps','brachioradialis','knee','ankle','patellar',
        'achilles','plantar','babinski','clonus','hyperreflexia',
        'hyporeflexia','areflexia','brisk','normal','absent','upgoing',
        'downgoing','hoffman',
    ],
    'medications': [
        'levetiracetam','valproate','carbamazepine','phenytoin',
        'lamotrigine','topiramate','lacosamide','gabapentin','pregabalin',
        'levodopa','carbidopa','rotigotine','pramipexole','ropinirole',
        'selegiline','rasagiline','entacapone','amantadine',
        'sumatriptan','propranolol','amitriptyline',
        'warfarin','heparin','enoxaparin','rivaroxaban','apixaban',
        'aspirin','clopidogrel','alteplase','tenecteplase',
        'prednisolone','methylprednisolone','dexamethasone',
        'paracetamol','ibuprofen','omeprazole','metoclopramide',
        'diazepam','lorazepam','midazolam','mannitol','furosemide',
        'amlodipine','metformin','insulin','atorvastatin',
    ],
    'general': [
        'patient','history','examination','diagnosis','treatment',
        'admitted','discharged','presents','complains','denies',
        'blood','pressure','heart','rate','respiratory','temperature',
        'oxygen','saturation','weight','height',
        'normal','abnormal','positive','negative','present','absent',
        'mild','moderate','severe','acute','chronic',
        'right','left','bilateral','unilateral',
    ],
}

LAB_RANGES = {
    'hba1c': (4.0, 15.0, '%'),
    'glucose': (1.0, 50.0, 'mmol/L'),
    'sodium': (100, 180, 'mmol/L'),
    'potassium': (1.5, 9.0, 'mmol/L'),
    'creatinine': (20, 2000, 'μmol/L'),
    'hemoglobin': (3.0, 25.0, 'g/dL'),
    'wbc': (0.1, 100.0, '×10⁹/L'),
    'platelets': (5, 1500, '×10⁹/L'),
    'inr': (0.5, 10.0, ''),
    'crp': (0.0, 500.0, 'mg/L'),
}

# Build flat word set
ALL_CLINICAL_WORDS = set()
for category_words in NEURO_VOCAB.values():
    if isinstance(category_words, list):
        if category_words and isinstance(category_words[0], str):
            ALL_CLINICAL_WORDS.update(w.lower() for w in category_words)
        else:
            for sublist in category_words:
                if isinstance(sublist, str):
                    ALL_CLINICAL_WORDS.add(sublist.lower())


class ClinicalPostProcessor:
    """Post-process OCR output using medical domain knowledge."""
    
    def __init__(self):
        self.current_section = None
        self.previous_words = []
    
    def process_word(self, ocr_word, max_dist=2.5):
        """
        Process a single OCR word against clinical vocabulary.
        Returns candidates ranked by combined visual + contextual score.
        """
        word = ocr_word.strip()
        if not word:
            return {'input': word, 'output': word, 'confidence': 0, 'flag': 'empty'}
        
        word_lower = word.lower()
        
        # Exact match
        if word_lower in ALL_CLINICAL_WORDS:
            self._update_context(word_lower)
            return {
                'input': word, 'output': word, 'confidence': 0.95,
                'flag': 'exact_match', 'candidates': []
            }
        
        # Number handling
        if re.match(r'^[\d.,/]+$', word):
            return self._process_number(word)
        
        # Fuzzy match
        context_words = self._get_context_words()
        candidates = []
        
        for vocab_word in ALL_CLINICAL_WORDS:
            dist = ocr_distance(word_lower, vocab_word)
            if dist <= max_dist:
                context_boost = 0.4 if vocab_word in context_words else 0.0
                len_ratio = min(len(word_lower), len(vocab_word)) / max(len(word_lower), len(vocab_word), 1)
                score = max(0, 1.0 - (dist - context_boost) / max(len(word_lower), 3)) * len_ratio
                candidates.append((vocab_word, dist, score, context_boost > 0))
        
        candidates.sort(key=lambda x: x[2], reverse=True)
        candidates = candidates[:5]
        
        if not candidates:
            return {
                'input': word, 'output': word, 'confidence': 0.0,
                'flag': 'unknown', 'candidates': []
            }
        
        top = candidates[0]
        margin = top[2] - candidates[1][2] if len(candidates) > 1 else top[2]
        
        # Safety flags
        is_med = top[0] in NEURO_VOCAB['medications']
        runner_is_med = len(candidates) > 1 and candidates[1][0] in NEURO_VOCAB['medications']
        
        if is_med and runner_is_med and margin < 0.2:
            flag = 'DANGER_med_ambiguity'
            confidence = min(top[2], 0.3)
        elif top[2] > 0.8 and margin > 0.15:
            flag = 'high_confidence'
            confidence = top[2]
        elif top[2] > 0.5:
            flag = 'verify'
            confidence = top[2]
        else:
            flag = 'low_confidence'
            confidence = top[2]
        
        if confidence > 0.5:
            self._update_context(top[0])
        
        return {
            'input': word, 'output': top[0], 'confidence': confidence,
            'flag': flag, 'context_used': top[3],
            'candidates': [(c[0], round(c[1], 2), round(c[2], 2)) for c in candidates[:3]]
        }
    
    def process_text(self, ocr_text):
        """Process a full OCR text string."""
        self.current_section = None
        self.previous_words = []
        
        words = ocr_text.split()
        results = []
        output_words = []
        
        for word in words:
            r = self.process_word(word)
            results.append(r)
            output_words.append(r['output'])
        
        return {
            'input': ocr_text,
            'output': ' '.join(output_words),
            'words': results,
            'avg_confidence': np.mean([r['confidence'] for r in results]) if results else 0,
            'flags': [r for r in results if r['flag'] not in ('exact_match', 'high_confidence', 'empty')],
        }
    
    def _process_number(self, word):
        """Check numerical values against known ranges."""
        try:
            value = float(word.replace(',', '.'))
        except ValueError:
            return {'input': word, 'output': word, 'confidence': 0.5, 'flag': 'number'}
        
        # Check recent context for lab names
        for prev in reversed(self.previous_words[-5:]):
            if prev in LAB_RANGES:
                lo, hi, unit = LAB_RANGES[prev]
                if lo <= value <= hi:
                    return {
                        'input': word, 'output': word, 'confidence': 0.9,
                        'flag': 'in_range',
                        'context': f'{prev}: {value} {unit} [{lo}-{hi}]'
                    }
                # Check missing decimal
                alts = []
                for dp in range(1, len(word)):
                    try:
                        alt = float(word[:dp] + '.' + word[dp:])
                        if lo <= alt <= hi:
                            alts.append(alt)
                    except:
                        pass
                if alts:
                    return {
                        'input': word, 'output': word, 'confidence': 0.3,
                        'flag': 'OUT_OF_RANGE',
                        'context': f'{prev}: {value} outside [{lo}-{hi}] {unit}',
                        'alternatives': alts,
                    }
                return {
                    'input': word, 'output': word, 'confidence': 0.4,
                    'flag': 'OUT_OF_RANGE',
                    'context': f'{prev}: {value} outside [{lo}-{hi}] {unit}',
                }
        
        return {'input': word, 'output': word, 'confidence': 0.7, 'flag': 'number'}
    
    def _update_context(self, word):
        self.previous_words.append(word)
        if len(self.previous_words) > 15:
            self.previous_words = self.previous_words[-15:]
        for section in NEURO_VOCAB['exam_sections']:
            if word in section.split():
                self.current_section = section
    
    def _get_context_words(self):
        words = set()
        if self.current_section:
            for key in ['cranial', 'motor', 'sensory', 'reflexes']:
                if key in self.current_section:
                    words.update(NEURO_VOCAB.get(key, []))
        words.update(NEURO_VOCAB['general'])
        return words
