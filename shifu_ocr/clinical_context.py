"""
MD-OCR Clinical Context Engine
================================

Domain-constrained post-processing for medical OCR.

THE PRINCIPLE:
A senior consultant reads a smudged prescription not by "hallucinating" 
the right answer, but by:
1. Generating multiple candidate readings
2. Filtering against domain knowledge (is this a real medication?)
3. Checking contextual consistency (does this dose make sense?)
4. FLAGGING UNCERTAINTY when evidence is weak

This module does the same thing. It NEVER silently overrides the OCR.
It ranks candidates, provides confidence, and demands human verification
when the evidence is ambiguous.

WHY THIS MATTERS IN MEDICINE:
"Methotrexate 15mg" and "Metoprolol 150mg" are both plausible in context.
A system that confidently picks one when the ink is smudged will eventually
kill someone. A system that says "I see two possibilities, please verify"
saves lives.

The strong prior makes the system SMARTER, not more CONFIDENT.
Smarter = better candidates. Not = less uncertainty.

Author: Bader & Claude — March 2026
"""

import numpy as np
from collections import defaultdict
import re


# =============================================================================
# LAYER 1: CLINICAL VOCABULARY — The "Gravity Wells"
# =============================================================================

class ClinicalVocabulary:
    """
    Domain-specific vocabulary organized by clinical context.
    
    This is the "medium" of medicine — the structured space through 
    which clinical language flows. Words don't appear randomly; they 
    follow predictable patterns determined by the clinical context.
    
    This is NOT a complete medical dictionary. It's a demonstration 
    of the principle with neurology-focused terms.
    """
    
    def __init__(self):
        # Neurology examination terms
        self.neuro_exam = {
            'sections': [
                'Mental Status', 'Cranial Nerves', 'Motor', 'Sensory',
                'Reflexes', 'Coordination', 'Gait', 'Cerebellar',
            ],
            'findings': {
                'cranial_nerves': [
                    'intact', 'normal', 'pupil', 'pupils', 'reactive', 'PERRLA',
                    'extraocular', 'movements', 'facial', 'symmetry', 'symmetric',
                    'asymmetric', 'palsy', 'nystagmus', 'diplopia', 'ptosis',
                    'visual', 'fields', 'fundoscopy', 'papilledema',
                    'optic', 'disc', 'trigeminal', 'abducens', 'oculomotor',
                    'trochlear', 'vestibulocochlear', 'glossopharyngeal',
                    'vagus', 'accessory', 'hypoglossal',
                ],
                'motor': [
                    'power', 'tone', 'bulk', 'strength', 'weakness',
                    'spasticity', 'rigidity', 'flaccid', 'atrophy',
                    'fasciculations', 'tremor', 'MRC', 'grade',
                    'proximal', 'distal', 'upper', 'lower', 'limb', 'limbs',
                    'hemiparesis', 'paraparesis', 'quadriparesis', 'monoparesis',
                ],
                'sensory': [
                    'sensation', 'light', 'touch', 'pinprick', 'vibration',
                    'proprioception', 'temperature', 'numbness', 'tingling',
                    'paresthesia', 'dysesthesia', 'dermatomal', 'glove',
                    'stocking', 'distribution', 'intact', 'reduced', 'absent',
                ],
                'reflexes': [
                    'biceps', 'triceps', 'brachioradialis', 'knee', 'ankle',
                    'patellar', 'Achilles', 'plantar', 'Babinski', 'clonus',
                    'hyperreflexia', 'hyporeflexia', 'areflexia',
                    'brisk', 'normal', 'absent', 'diminished', 'upgoing',
                    'downgoing', 'Hoffman', 'jaw', 'jerk',
                ],
                'coordination': [
                    'finger', 'nose', 'heel', 'shin', 'dysdiadochokinesia',
                    'dysmetria', 'intention', 'tremor', 'ataxia', 'Romberg',
                    'tandem', 'gait', 'cerebellar', 'past', 'pointing',
                ],
            },
        }
        
        # Common medications (neurology-focused)
        self.medications = {
            'antiepileptics': [
                'Levetiracetam', 'Valproate', 'Carbamazepine', 'Phenytoin',
                'Lamotrigine', 'Topiramate', 'Lacosamide', 'Brivaracetam',
                'Oxcarbazepine', 'Gabapentin', 'Pregabalin', 'Clobazam',
                'Clonazepam', 'Zonisamide', 'Perampanel',
            ],
            'parkinsonian': [
                'Levodopa', 'Carbidopa', 'Rotigotine', 'Pramipexole',
                'Ropinirole', 'Selegiline', 'Rasagiline', 'Entacapone',
                'Amantadine', 'Trihexyphenidyl', 'Benztropine',
            ],
            'headache': [
                'Sumatriptan', 'Rizatriptan', 'Topiramate', 'Propranolol',
                'Amitriptyline', 'Verapamil', 'Valproate', 'Botulinum',
                'Erenumab', 'Fremanezumab', 'Galcanezumab',
            ],
            'anticoagulants': [
                'Warfarin', 'Heparin', 'Enoxaparin', 'Rivaroxaban',
                'Apixaban', 'Dabigatran', 'Edoxaban',
            ],
            'thrombolytics': [
                'Alteplase', 'Tenecteplase',
            ],
            'antiplatelets': [
                'Aspirin', 'Clopidogrel', 'Ticagrelor', 'Dipyridamole',
            ],
            'steroids': [
                'Prednisolone', 'Methylprednisolone', 'Dexamethasone',
                'Hydrocortisone',
            ],
            'common': [
                'Paracetamol', 'Ibuprofen', 'Omeprazole', 'Metoclopramide',
                'Ondansetron', 'Diazepam', 'Lorazepam', 'Midazolam',
                'Phenobarbital', 'Mannitol', 'Furosemide', 'Amlodipine',
                'Lisinopril', 'Metformin', 'Insulin', 'Atorvastatin',
            ],
        }
        
        # Lab values with expected ranges
        self.lab_ranges = {
            'HbA1c': (4.0, 15.0, '%'),
            'Glucose': (1.0, 50.0, 'mmol/L'),
            'Sodium': (100, 180, 'mmol/L'),
            'Potassium': (1.5, 9.0, 'mmol/L'),
            'Creatinine': (20, 2000, 'μmol/L'),
            'Hemoglobin': (3.0, 25.0, 'g/dL'),
            'WBC': (0.1, 100.0, '×10⁹/L'),
            'Platelets': (5, 1500, '×10⁹/L'),
            'INR': (0.5, 10.0, ''),
            'CRP': (0.0, 500.0, 'mg/L'),
            'ESR': (0, 150, 'mm/hr'),
            'TSH': (0.01, 100.0, 'mIU/L'),
            'CSF Protein': (0.1, 10.0, 'g/L'),
            'CSF Glucose': (0.5, 10.0, 'mmol/L'),
            'CSF WBC': (0, 10000, 'cells/μL'),
        }
        
        # Build flat word list for fuzzy matching
        self._all_words = set()
        self._build_word_index()
    
    def _build_word_index(self):
        """Index all known clinical words for fast lookup."""
        for section_terms in self.neuro_exam['findings'].values():
            for term in section_terms:
                self._all_words.add(term.lower())
        
        for section in self.neuro_exam['sections']:
            for word in section.split():
                self._all_words.add(word.lower())
        
        for med_list in self.medications.values():
            for med in med_list:
                self._all_words.add(med.lower())
        
        for lab in self.lab_ranges:
            for word in lab.split():
                self._all_words.add(word.lower())
    
    def get_all_words(self):
        return self._all_words
    
    def get_medications_flat(self):
        meds = []
        for med_list in self.medications.values():
            meds.extend(med_list)
        return meds


# =============================================================================
# LAYER 2: EDIT DISTANCE WITH CLINICAL WEIGHTING
# =============================================================================

def levenshtein_distance(s1, s2):
    """Standard Levenshtein edit distance."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    
    prev_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        curr_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = prev_row[j + 1] + 1
            deletions = curr_row[j] + 1
            substitutions = prev_row[j] + (c1 != c2)
            curr_row.append(min(insertions, deletions, substitutions))
        prev_row = curr_row
    
    return prev_row[-1]


def ocr_weighted_distance(s1, s2):
    """
    Edit distance weighted by OCR confusion probability.
    
    Some substitutions are MORE LIKELY in OCR than others:
    - 'rn' ↔ 'm' (adjacent strokes merge)
    - 'l' ↔ '1' ↔ 'I' (similar vertical strokes)
    - 'O' ↔ '0' (same topology!)
    - 'cl' ↔ 'd' (strokes merge)
    - 'vv' ↔ 'w' (similar structure)
    
    These are NOT random — they're predicted by the medium displacement 
    theory. Characters with similar topological signatures are more 
    likely to be confused.
    """
    # Common OCR confusion pairs (cost < 1.0 for likely confusions)
    confusions = {
        ('r', 'n'): 0.3,   # rn ↔ m confusion
        ('l', '1'): 0.2,
        ('l', 'I'): 0.2,
        ('I', '1'): 0.2,
        ('O', '0'): 0.1,   # Same topology = very easy to confuse
        ('o', '0'): 0.1,
        ('S', '5'): 0.4,
        ('Z', '2'): 0.4,
        ('B', '8'): 0.3,
        ('G', '6'): 0.4,
        ('c', 'e'): 0.5,   # Similar with smudge
        ('u', 'v'): 0.5,
        ('h', 'b'): 0.5,
        ('q', 'g'): 0.4,
    }
    
    s1, s2 = s1.lower(), s2.lower()
    if len(s1) < len(s2):
        return ocr_weighted_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    
    prev_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr_row = [i + 1]
        for j, c2 in enumerate(s2):
            if c1 == c2:
                sub_cost = 0
            else:
                pair = tuple(sorted([c1, c2]))
                sub_cost = confusions.get(pair, 1.0)
            
            insertions = prev_row[j + 1] + 1
            deletions = curr_row[j] + 1
            substitutions = prev_row[j] + sub_cost
            curr_row.append(min(insertions, deletions, substitutions))
        prev_row = curr_row
    
    return prev_row[-1]


# =============================================================================
# LAYER 3: CONTEXTUAL CONSTRAINTS — "Gravity Wells"
# =============================================================================

class ClinicalContext:
    """
    Tracks the current clinical context to constrain interpretations.
    
    As the OCR reads through a document, context accumulates:
    - If we're in a "Cranial Nerves" section, expect CN-related terms
    - If we just read a medication name, expect a dose next
    - If we see "HbA1c:", expect a number in range 4.0-15.0
    
    This is the "medium of medicine" — the structured space that 
    constrains what words can appear where.
    """
    
    def __init__(self, vocab):
        self.vocab = vocab
        self.current_section = None
        self.previous_words = []
        self.expecting = None  # What type of token we expect next
    
    def update(self, word):
        """Update context based on a confirmed word."""
        self.previous_words.append(word)
        if len(self.previous_words) > 20:
            self.previous_words = self.previous_words[-20:]
        
        # Detect section headers
        word_lower = word.lower()
        for section in self.vocab.neuro_exam['sections']:
            if word_lower in section.lower().split():
                self.current_section = section.lower()
        
        # Detect what should come next
        if word_lower in [m.lower() for meds in self.vocab.medications.values() for m in meds]:
            self.expecting = 'dose'
        elif word_lower in self.vocab.lab_ranges or any(
                word_lower == lab.lower().split()[0] for lab in self.vocab.lab_ranges):
            self.expecting = 'lab_value'
        elif word_lower in ['power', 'strength', 'mrc', 'grade']:
            self.expecting = 'mrc_grade'
        elif word_lower in ['reflexes', 'reflex']:
            self.expecting = 'reflex_grade'
        else:
            self.expecting = None
    
    def get_expected_words(self):
        """Get the set of words that are contextually likely here."""
        candidates = set()
        
        # Section-specific terms
        if self.current_section:
            for section_key, terms in self.vocab.neuro_exam['findings'].items():
                if any(w in self.current_section for w in section_key.split('_')):
                    candidates.update(t.lower() for t in terms)
        
        # Always include common clinical terms
        candidates.update(t.lower() for t in self.vocab.neuro_exam['sections'])
        
        return candidates
    
    def get_value_constraints(self):
        """If we're expecting a numerical value, what range is plausible?"""
        if self.expecting == 'dose':
            return {'type': 'dose', 'note': 'Check: is this a plausible dose?'}
        
        if self.expecting == 'lab_value':
            # Find which lab
            for lab, (lo, hi, unit) in self.vocab.lab_ranges.items():
                if any(w.lower() in lab.lower() for w in self.previous_words[-3:]):
                    return {'type': 'lab', 'name': lab, 'min': lo, 'max': hi, 'unit': unit}
        
        if self.expecting == 'mrc_grade':
            return {'type': 'mrc', 'valid_values': [0, 1, 2, 3, 4, 5], 'format': 'X/5'}
        
        if self.expecting == 'reflex_grade':
            return {'type': 'reflex', 'valid_values': [0, 1, 2, 3, 4],
                    'labels': ['absent', 'diminished', 'normal', 'brisk', 'clonus']}
        
        return None


# =============================================================================
# LAYER 4: INTERPRETATION ENGINE — Ranking, not overriding
# =============================================================================

class ClinicalInterpreter:
    """
    The core interpretation engine.
    
    Given an OCR reading (possibly noisy), generate ranked candidates 
    with confidence scores and uncertainty flags.
    
    CRITICAL DESIGN PRINCIPLE:
    This system NEVER silently changes an OCR reading. It:
    1. Generates candidates (what could this word be?)
    2. Scores them (how likely is each, given context?)
    3. Flags uncertainty (is the top candidate clearly best, or ambiguous?)
    4. Requires human verification when confidence is low
    
    A wrong confident answer in medicine is worse than no answer.
    """
    
    def __init__(self):
        self.vocab = ClinicalVocabulary()
        self.context = ClinicalContext(self.vocab)
    
    def interpret_word(self, ocr_text, max_candidates=5, max_edit_distance=3):
        """
        Interpret a single word from OCR output.
        
        Returns a ranked list of candidates with:
        - The candidate word
        - Visual distance (how different it looks from OCR output)
        - Contextual score (how well it fits the clinical context)
        - Overall confidence
        - Uncertainty flag
        """
        ocr_lower = ocr_text.lower().strip()
        
        # If it's already a known clinical word, high confidence
        if ocr_lower in self.vocab.get_all_words():
            result = {
                'input': ocr_text,
                'top_candidate': ocr_text,
                'confidence': 0.95,
                'flag': 'ACCEPT',
                'candidates': [(ocr_text, 0.0, 1.0, 0.95)],
                'reasoning': 'Exact match in clinical vocabulary.',
            }
            self.context.update(ocr_text)
            return result
        
        # Generate candidates from clinical vocabulary
        candidates = []
        context_words = self.context.get_expected_words()
        all_words = self.vocab.get_all_words()
        
        for word in all_words:
            # OCR-weighted edit distance
            dist = ocr_weighted_distance(ocr_lower, word)
            
            if dist <= max_edit_distance:
                # Contextual boost: words expected in current context get lower effective distance
                context_boost = 0.0
                if word in context_words:
                    context_boost = 0.5
                
                # Length similarity bonus
                len_ratio = min(len(ocr_lower), len(word)) / max(len(ocr_lower), len(word), 1)
                
                effective_dist = dist - context_boost
                score = max(0, 1.0 - effective_dist / max(len(ocr_lower), 3)) * len_ratio
                
                candidates.append((word, dist, context_boost, score))
        
        # Sort by score (descending)
        candidates.sort(key=lambda x: x[3], reverse=True)
        candidates = candidates[:max_candidates]
        
        if not candidates:
            result = {
                'input': ocr_text,
                'top_candidate': ocr_text,
                'confidence': 0.0,
                'flag': 'UNKNOWN — not in clinical vocabulary',
                'candidates': [],
                'reasoning': f'No clinical vocabulary match within edit distance {max_edit_distance}.',
            }
            return result
        
        top = candidates[0]
        
        # Determine confidence and flag
        confidence = top[3]
        
        if len(candidates) >= 2:
            margin = top[3] - candidates[1][3]
        else:
            margin = top[3]
        
        # Flag logic — THIS IS THE SAFETY LAYER
        if confidence > 0.85 and margin > 0.2:
            flag = 'ACCEPT — high confidence'
        elif confidence > 0.6 and margin > 0.1:
            flag = 'LIKELY — verify if critical'
        elif confidence > 0.4:
            flag = '⚠ AMBIGUOUS — multiple plausible readings, VERIFY'
        else:
            flag = '⛔ LOW CONFIDENCE — manual verification REQUIRED'
        
        # Check for dangerous ambiguity (similar medications)
        if len(candidates) >= 2:
            all_meds = [m.lower() for meds in self.vocab.medications.values() for m in meds]
            top_is_med = top[0] in all_meds
            second_is_med = candidates[1][0] in all_meds
            
            if top_is_med and second_is_med and margin < 0.3:
                flag = '⛔ DANGER — ambiguous between two medications: MUST VERIFY'
                confidence = min(confidence, 0.3)  # Force low confidence
        
        result = {
            'input': ocr_text,
            'top_candidate': top[0],
            'confidence': confidence,
            'flag': flag,
            'candidates': candidates,
            'reasoning': self._build_reasoning(ocr_text, candidates),
        }
        
        # Update context only with high-confidence results
        if confidence > 0.6:
            self.context.update(top[0])
        
        return result
    
    def interpret_number(self, ocr_text):
        """
        Interpret a numerical value with range checking.
        
        If context tells us we're reading an HbA1c, and the OCR reads "71",
        we can note that this is outside the expected range (4.0-15.0) and 
        suggest it might be "7.1" (missing decimal point — common OCR error).
        
        But we FLAG this, we don't silently fix it.
        """
        constraints = self.context.get_value_constraints()
        
        # Try to parse the number
        try:
            value = float(ocr_text.replace(',', '.').strip())
        except ValueError:
            return {
                'input': ocr_text,
                'parsed_value': None,
                'flag': '⚠ Cannot parse as number',
                'constraints': constraints,
            }
        
        if constraints is None:
            return {
                'input': ocr_text,
                'parsed_value': value,
                'flag': 'ACCEPT — no range constraints active',
                'constraints': None,
            }
        
        if constraints['type'] == 'lab':
            lo, hi = constraints['min'], constraints['max']
            name = constraints['name']
            unit = constraints['unit']
            
            if lo <= value <= hi:
                return {
                    'input': ocr_text,
                    'parsed_value': value,
                    'flag': f'ACCEPT — {name} value {value} {unit} is within range [{lo}-{hi}]',
                    'constraints': constraints,
                }
            
            # Check if a decimal point might be missing
            alternatives = []
            for decimal_pos in range(1, len(ocr_text)):
                try:
                    alt = float(ocr_text[:decimal_pos] + '.' + ocr_text[decimal_pos:])
                    if lo <= alt <= hi:
                        alternatives.append(alt)
                except ValueError:
                    pass
            
            if alternatives:
                return {
                    'input': ocr_text,
                    'parsed_value': value,
                    'flag': f'⚠ {name} value {value} OUTSIDE range [{lo}-{hi}] {unit}. '
                            f'Possible missing decimal: {alternatives}. VERIFY.',
                    'alternatives': alternatives,
                    'constraints': constraints,
                }
            
            return {
                'input': ocr_text,
                'parsed_value': value,
                'flag': f'⛔ {name} value {value} OUTSIDE expected range [{lo}-{hi}] {unit}. '
                        f'Verify reading.',
                'constraints': constraints,
            }
        
        if constraints['type'] == 'mrc':
            valid = constraints['valid_values']
            if value in valid:
                return {
                    'input': ocr_text,
                    'parsed_value': int(value),
                    'flag': f'ACCEPT — MRC grade {int(value)}/5',
                    'constraints': constraints,
                }
            return {
                'input': ocr_text,
                'parsed_value': value,
                'flag': f'⚠ MRC grade should be {valid}, got {value}. VERIFY.',
                'constraints': constraints,
            }
        
        return {
            'input': ocr_text,
            'parsed_value': value,
            'flag': 'ACCEPT',
            'constraints': constraints,
        }
    
    def interpret_sequence(self, ocr_words):
        """
        Interpret a sequence of OCR words with accumulating context.
        
        Each word updates the context, which influences interpretation 
        of subsequent words. This is the "semantic flow" — the medium 
        of the sentence constraining what comes next.
        """
        self.context = ClinicalContext(self.vocab)  # Reset
        results = []
        
        for word in ocr_words:
            word = word.strip()
            if not word:
                continue
            
            # Is it a number?
            is_number = bool(re.match(r'^[\d.,]+$', word))
            
            if is_number:
                result = self.interpret_number(word)
            else:
                result = self.interpret_word(word)
            
            results.append(result)
        
        return results
    
    def _build_reasoning(self, ocr_text, candidates):
        """Human-readable explanation of the interpretation."""
        if not candidates:
            return f"'{ocr_text}' has no close matches in clinical vocabulary."
        
        top = candidates[0]
        lines = [f"OCR read: '{ocr_text}'"]
        lines.append(f"Best match: '{top[0]}' (edit distance: {top[1]:.1f}, "
                     f"context boost: {top[2]:.1f}, score: {top[3]:.2f})")
        
        if self.context.current_section:
            lines.append(f"Current section: {self.context.current_section}")
        
        if len(candidates) > 1:
            lines.append("Other candidates:")
            for c in candidates[1:]:
                lines.append(f"  '{c[0]}' (dist: {c[1]:.1f}, score: {c[3]:.2f})")
        
        return '\n'.join(lines)


# =============================================================================
# DEMO & TESTING
# =============================================================================

def demo_word_interpretation():
    """Show how the context engine handles various OCR errors."""
    print("=" * 70)
    print("DEMO 1: Word-Level Interpretation")
    print("=" * 70)
    print()
    
    interp = ClinicalInterpreter()
    
    test_cases = [
        # (OCR output, description of the error)
        ("Cranial", "Clean read"),
        ("Nerves", "Clean read — context should be 'cranial nerves' now"),
        ("Crainal", "Transposition error"),
        ("Merves", "Smudged N→M: should suggest 'Nerves' in CN context"),
        ("Papilledema", "Clean medical term"),
        ("Papilledma", "Missing letter"),
        ("Nystagnus", "Common OCR error: m→n"),
        ("Babinski", "Clean — expect reflex context"),
        ("upqoing", "Smudged: 'upgoing'"),
        ("Levetiracetan", "Medication with typo"),
        ("Rotigotine", "Clean medication"),
        ("Carbamazepime", "OCR error: n→m at end"),
    ]
    
    for ocr_text, description in test_cases:
        result = interp.interpret_word(ocr_text)
        
        flag_symbol = {'ACCEPT': '✓', 'LIKELY': '○'}.get(
            result['flag'].split('—')[0].strip().split()[0], '⚠')
        
        print(f"  OCR: '{ocr_text:20s}' → '{result['top_candidate']:20s}' "
              f"[{result['confidence']:.0%}] {result['flag']}")
        
        if result.get('candidates') and len(result['candidates']) > 1:
            c2 = result['candidates'][1]
            print(f"       {'':20s}    also: '{c2[0]}' (score: {c2[3]:.2f})")
        print()


def demo_sequence_interpretation():
    """Show contextual interpretation of a clinical sentence."""
    print("=" * 70)
    print("DEMO 2: Sequence Interpretation with Context Flow")
    print("=" * 70)
    print()
    
    # Simulated OCR output from a neurology note
    sequences = [
        {
            'description': 'Cranial nerve examination (some OCR errors)',
            'words': ['Cranial', 'Nerves', ':', 'Puplis', 'reactive',
                      'Extraocular', 'movments', 'intact', 'Facail', 'symmetry'],
        },
        {
            'description': 'Motor examination (messy handwriting)',
            'words': ['Motor', ':', 'Power', '5', 'upper', 'linbs',
                      'Tone', 'nornal', 'No', 'fasciculatlons'],
        },
        {
            'description': 'Medication list (critical — errors here are dangerous)',
            'words': ['Levetiracetam', '500mg', 'Rotigotine', '4mg',
                      'Carbamazepime', '200mg'],
        },
    ]
    
    for seq in sequences:
        print(f"  Scenario: {seq['description']}")
        print(f"  Input: {' '.join(seq['words'])}")
        print()
        
        interp = ClinicalInterpreter()
        results = interp.interpret_sequence(seq['words'])
        
        for r in results:
            if 'parsed_value' in r:
                # Numerical
                print(f"    '{r['input']:15s}' → {r.get('parsed_value', '?'):>8} {r['flag']}")
            else:
                top = r['top_candidate']
                conf = r['confidence']
                changed = '→' if top.lower() != r['input'].lower() else '✓'
                
                if conf > 0.6 and top.lower() != r['input'].lower():
                    print(f"    '{r['input']:15s}' {changed} '{top:15s}' [{conf:.0%}] {r['flag']}")
                elif conf <= 0.6 and r.get('candidates'):
                    print(f"    '{r['input']:15s}' ? [{conf:.0%}] {r['flag']}")
                else:
                    print(f"    '{r['input']:15s}' ✓ [{conf:.0%}]")
        
        print()


def demo_number_interpretation():
    """Show range-checking for lab values."""
    print("=" * 70)
    print("DEMO 3: Numerical Value Interpretation with Range Checking")
    print("=" * 70)
    print()
    
    interp = ClinicalInterpreter()
    
    # Simulate reading a lab report
    test_sequences = [
        (['HbA1c', '71'], 'Missing decimal: should be 7.1, not 71'),
        (['HbA1c', '7.1'], 'Correct reading'),
        (['Sodium', '139'], 'Normal value'),
        (['Sodium', '1139'], 'Extra digit — OCR artifact'),
        (['Potassium', '45'], 'Missing decimal: should be 4.5'),
        (['Potassium', '4.5'], 'Correct'),
        (['INR', '23'], 'Missing decimal: should be 2.3'),
        (['Hemoglobin', '12.5'], 'Normal'),
    ]
    
    for words, description in test_sequences:
        interp2 = ClinicalInterpreter()
        results = interp2.interpret_sequence(words)
        
        # Show the number result
        num_result = results[-1]  # Last result is the number
        print(f"  {description}")
        print(f"    Input: {' '.join(words)}")
        print(f"    {num_result['flag']}")
        if 'alternatives' in num_result:
            print(f"    Suggested alternatives: {num_result['alternatives']}")
        print()


def demo_medication_safety():
    """Show the safety layer for medication disambiguation."""
    print("=" * 70)
    print("DEMO 4: Medication Safety — The Dangerous Ambiguity Problem")
    print("=" * 70)
    print()
    print("  The most dangerous OCR error is confidently misreading")
    print("  one medication as another. The system must flag these.\n")
    
    interp = ClinicalInterpreter()
    
    # Deliberately ambiguous medication readings
    dangerous_cases = [
        ('Methotrexat', 'Missing final e'),
        ('Carbamazepime', 'n→m error — could be Carbamazepine'),
        ('Prednisolome', 'n→m error'),
        ('Levodops', 'a→s error'),
        ('Carbidops', 'a→s error'),
        ('Clopidoqrel', 'g→q error'),
        ('Valproafe', 't→f error'),
    ]
    
    for ocr_text, error_type in dangerous_cases:
        result = interp.interpret_word(ocr_text)
        
        print(f"  OCR: '{ocr_text}' ({error_type})")
        print(f"    → '{result['top_candidate']}' [{result['confidence']:.0%}]")
        print(f"    {result['flag']}")
        
        if len(result.get('candidates', [])) > 1:
            print(f"    Other candidates: ", end='')
            for c in result['candidates'][1:3]:
                print(f"'{c[0]}' ({c[3]:.2f}), ", end='')
            print()
        print()


def demo_context_flow():
    """Show how context accumulates and constrains interpretation."""
    print("=" * 70)
    print("DEMO 5: Context Flow — How the 'Medical Medium' Works")
    print("=" * 70)
    print()
    print("  The same ambiguous word gets different interpretations")
    print("  depending on what came before it.\n")
    
    # Same noisy word, different contexts
    print("  Context 1: Cranial Nerves section")
    interp1 = ClinicalInterpreter()
    r1 = interp1.interpret_sequence(['Cranial', 'Nerves', 'puplis'])
    last1 = r1[-1]
    print(f"    'puplis' → '{last1['top_candidate']}' [{last1['confidence']:.0%}]")
    if last1.get('candidates'):
        for c in last1['candidates'][:3]:
            print(f"      candidate: '{c[0]}' (dist={c[1]:.1f}, context_boost={c[2]:.1f}, score={c[3]:.2f})")
    
    print()
    print("  Context 2: Reflexes section")
    interp2 = ClinicalInterpreter()
    r2 = interp2.interpret_sequence(['Reflexes', 'biceps'])
    # Now add same-ish noisy word in reflex context
    r2b = interp2.interpret_word('brlsk')
    print(f"    'brlsk' → '{r2b['top_candidate']}' [{r2b['confidence']:.0%}]")
    if r2b.get('candidates'):
        for c in r2b['candidates'][:3]:
            print(f"      candidate: '{c[0]}' (dist={c[1]:.1f}, context_boost={c[2]:.1f}, score={c[3]:.2f})")
    
    print()


# =============================================================================
# MAIN
# =============================================================================

def main():
    print()
    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║       MD-OCR Clinical Context Engine — Proof of Concept            ║")
    print("║                                                                    ║")
    print("║  Principle: The 'medium of medicine' constrains interpretation.    ║")
    print("║  Safety:    NEVER silently override. ALWAYS flag uncertainty.       ║")
    print("║  Design:    Rank candidates. Check ranges. Require verification.   ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")
    print()
    
    demo_word_interpretation()
    demo_sequence_interpretation()
    demo_number_interpretation()
    demo_medication_safety()
    demo_context_flow()
    
    print("=" * 70)
    print("DESIGN PRINCIPLES — NON-NEGOTIABLE")
    print("=" * 70)
    print()
    print("  1. The system suggests, the clinician decides.")
    print("     Every interpretation is a CANDIDATE, not a fact.")
    print()
    print("  2. Low confidence = loud flag, not silent guess.")
    print("     Ambiguity between two medications is flagged as DANGER.")
    print()
    print("  3. Context makes the system smarter, not more confident.")
    print("     A strong prior narrows candidates. It doesn't eliminate")
    print("     the need for human verification.")
    print()
    print("  4. The cost function is asymmetric.")
    print("     A wrong confident answer is infinitely worse than")
    print("     'I can't read this, please verify.'")
    print()
    print("  5. This is a TOOL, not an oracle.")
    print("     It extends the clinician's capacity.")
    print("     It does not replace the clinician's judgment.")
    print()


if __name__ == '__main__':
    main()
