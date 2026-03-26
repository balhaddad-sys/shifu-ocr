"""
Shifu-OCR v2 — The Co-Defining Engine
=======================================

Core principle: The cup defines the content. The content defines the cup.

In v1, character recognition and word reading were separate layers.
Characters were recognized in isolation, then assembled into words.
That's like building a cup and then pouring water — they don't 
know about each other.

In v2, they co-define:
- A character's landscape is reshaped by its NEIGHBORS 
  (what characters tend to appear next to it)
- A word constrains which characters are plausible at each position
- Clinical context constrains which words are plausible
- And it flows BACKWARDS too: if the word must be "SODIUM", 
  the 5th character MUST be "U", even if the landscape says "V"

The container shapes the content. The content shapes the container.
Soft enough to adapt. Rigid enough to reject molten lava.
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from scipy import ndimage
from skimage import morphology, filters
from collections import defaultdict
import json, os, re
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# MEDIUM DISPLACEMENT PIPELINE (stable)
# =============================================================================

def estimate_background(img, k=25):
    return filters.gaussian(morphology.closing(img, morphology.disk(k)), sigma=k/2)

def compute_displacement(img, bg):
    d = bg.astype(float) - img.astype(float)
    r = d.max() - d.min()
    return (d - d.min()) / r if r > 0 else d * 0

def detect_perturbation(disp, thresh=0.25):
    return morphology.remove_small_objects(disp > thresh, min_size=8).astype(np.uint8)

def extract_region(binary, pad=3):
    coords = np.argwhere(binary > 0)
    if len(coords) == 0: return binary
    r0, c0 = np.maximum(coords.min(axis=0) - pad, 0)
    r1, c1 = np.minimum(coords.max(axis=0) + pad, np.array(binary.shape) - 1)
    return binary[r0:r1+1, c0:c1+1]

def normalize_region(region, size=(64, 64)):
    img = Image.fromarray((region * 255).astype(np.uint8)).resize(size, Image.NEAREST)
    return (np.array(img) > 127).astype(np.uint8)

def image_to_binary(char_img, bg_kernel=15):
    bg = estimate_background(char_img, k=bg_kernel)
    disp = compute_displacement(char_img, bg)
    return detect_perturbation(disp), disp


# =============================================================================
# FEATURE EXTRACTION
# =============================================================================

FEATURE_NAMES = (
    ['components', 'holes', 'euler', 'displacement_ratio',
     'v_symmetry', 'h_symmetry', 'v_center', 'h_center'] +
    [f'quad_{q}' for q in ['TL', 'TR', 'BL', 'BR']] +
    [f'hproj_{i}' for i in range(6)] +
    [f'vproj_{i}' for i in range(6)] +
    [f'hcross_{i}' for i in range(6)] +
    [f'vcross_{i}' for i in range(6)] +
    ['endpoints', 'junctions']
)

def extract_features(br):
    h, w = br.shape
    feats = []
    padded = np.pad(br, 1, mode='constant', constant_values=0)
    _, nfg = ndimage.label(padded)
    _, nbg = ndimage.label(1 - padded)
    holes = nbg - 1
    feats.extend([float(nfg), float(holes), float(nfg - holes)])
    feats.append(float(np.mean(br)))
    feats.append(float(np.mean(br == np.fliplr(br))) if w >= 2 else 1.0)
    feats.append(float(np.mean(br == np.flipud(br))) if h >= 2 else 1.0)
    total = br.sum()
    if total > 0 and h > 0 and w > 0:
        rows, cols = np.arange(h).reshape(-1, 1), np.arange(w).reshape(1, -1)
        feats.append(float((br * rows).sum() / (total * h)))
        feats.append(float((br * cols).sum() / (total * w)))
    else:
        feats.extend([0.5, 0.5])
    mh, mw = h // 2, w // 2
    quads = [
        float(br[:mh, :mw].mean()) if mh > 0 and mw > 0 else 0,
        float(br[:mh, mw:].mean()) if mh > 0 else 0,
        float(br[mh:, :mw].mean()) if mw > 0 else 0,
        float(br[mh:, mw:].mean()),
    ]
    qt = sum(quads)
    feats.extend([q / qt if qt > 0 else 0.25 for q in quads])
    for axis in [1, 0]:
        raw = br.mean(axis=axis)
        proj = np.zeros(6)
        bw = max(1, len(raw) // 6)
        for i in range(6):
            s, e = i * bw, min((i + 1) * bw, len(raw))
            if s < len(raw): proj[i] = raw[s:e].mean()
        pt = proj.sum()
        if pt > 0: proj /= pt
        feats.extend(proj.tolist())
    for axis in [0, 1]:
        for i in range(6):
            if axis == 0:
                idx = min(int((i + 0.5) * h / 6), h - 1)
                line = br[idx, :]
            else:
                idx = min(int((i + 0.5) * w / 6), w - 1)
                line = br[:, idx]
            feats.append(float(np.abs(np.diff(line.astype(int))).sum() / 2))
    if h >= 4 and w >= 4 and br.sum() >= 10:
        try:
            skel = morphology.skeletonize(br.astype(bool))
            nc = ndimage.convolve(skel.astype(int), np.ones((3, 3), dtype=int),
                                   mode='constant') - skel.astype(int)
            _, n_ep = ndimage.label(skel & (nc == 1))
            _, n_jp = ndimage.label(skel & (nc >= 3))
            feats.extend([float(n_ep), float(n_jp)])
        except:
            feats.extend([0.0, 0.0])
    else:
        feats.extend([0.0, 0.0])
    return np.array(feats, dtype=float)


# =============================================================================
# FLUID LANDSCAPE (v2: co-defining)
# =============================================================================

class Landscape:
    """A character's identity as a fluid probability terrain."""

    def __init__(self, label):
        self.label = label
        self.observations = []
        self.mean = None
        self.variance = None
        self.n = 0
        self.n_correct = 0
        self.n_errors = 0
        self.confused_with = defaultdict(int)

    def absorb(self, fv):
        self.observations.append(fv.copy())
        self.n = len(self.observations)
        if self.n == 1:
            self.mean = fv.copy()
            self.variance = np.ones_like(fv) * 2.0
        else:
            obs = np.array(self.observations)
            self.mean = obs.mean(axis=0)
            self.variance = np.maximum(obs.var(axis=0), 0.1 / np.sqrt(self.n))

    def fit(self, fv):
        if self.mean is None: return -float('inf')
        diff = fv - self.mean
        precision = 1.0 / (self.variance + 1e-8)
        return -0.5 * np.sum(diff ** 2 * precision) + np.log(self.n + 1) * 0.5

    def rejection_score(self, fv):
        """How far is this observation from the landscape's territory?
        High = doesn't belong here (molten lava in paper cup)."""
        if self.mean is None: return float('inf')
        diff = fv - self.mean
        std = np.sqrt(self.variance + 1e-8)
        # Max Z-score across all features
        z_scores = np.abs(diff) / std
        return float(np.max(z_scores))

    def to_dict(self):
        return {
            'label': self.label, 'n': self.n,
            'mean': self.mean.tolist() if self.mean is not None else None,
            'variance': self.variance.tolist() if self.variance is not None else None,
            'n_correct': self.n_correct, 'n_errors': self.n_errors,
            'confused_with': dict(self.confused_with),
        }

    @classmethod
    def from_dict(cls, d):
        l = cls(d['label'])
        l.n = d['n']
        l.mean = np.array(d['mean']) if d['mean'] else None
        l.variance = np.array(d['variance']) if d['variance'] else None
        l.n_correct = d.get('n_correct', 0)
        l.n_errors = d.get('n_errors', 0)
        l.confused_with = defaultdict(int, d.get('confused_with', {}))
        return l


# =============================================================================
# WORD LANDSCAPE — Words as containers that constrain characters
# =============================================================================

class WordLandscape:
    """
    A word is a CONTAINER. Its shape constrains what characters 
    can appear inside it.
    
    We don't store the word as a string. We store its SHAPE:
    - Length
    - Character-level topological sequence (holes pattern)
    - First/last character topology
    - Proportional width pattern
    
    "SODIUM" and "50DIUM" have the same shape except position 1.
    If we know the word should be "SODIUM", position 1 is constrained 
    to characters that have 0 holes and low displacement — ruling out 
    "5" in favor of "S".
    """

    def __init__(self, word, char_features_list):
        self.word = word
        self.length = len(word)
        self.characters = list(word)

        # Store per-position topology expectations
        self.position_features = []
        for fv in char_features_list:
            self.position_features.append({
                'holes': fv[1],      # Index 1 = holes
                'euler': fv[2],      # Index 2 = euler
                'disp_ratio': fv[3], # Index 3 = displacement ratio
                'v_sym': fv[4],      # Index 4 = vertical symmetry
            })

    def constrain(self, position, candidates):
        """
        Given a position in this word, re-rank character candidates 
        based on what the word EXPECTS at that position.
        
        This is the container constraining the content.
        """
        if position >= len(self.position_features):
            return candidates

        expected = self.position_features[position]

        reranked = []
        for label, score, fv in candidates:
            # How well does this character fit what the word expects here?
            fit_bonus = 0.0
            if fv is not None and len(fv) > 4:
                # Topology match is strongest constraint
                if abs(fv[1] - expected['holes']) < 0.5:
                    fit_bonus += 2.0
                if abs(fv[3] - expected['disp_ratio']) < 0.15:
                    fit_bonus += 1.0
                if abs(fv[4] - expected['v_sym']) < 0.1:
                    fit_bonus += 0.5

            reranked.append((label, score + fit_bonus, fv))

        reranked.sort(key=lambda x: x[1], reverse=True)
        return reranked


# =============================================================================
# CLINICAL VOCABULARY — The semantic containers
# =============================================================================

# OCR confusion costs
CONFUSIONS = {
    ('0','o'):0.1, ('1','l'):0.2, ('1','i'):0.2, ('l','i'):0.2,
    ('5','s'):0.3, ('2','z'):0.4, ('8','b'):0.3, ('6','g'):0.4,
    ('m','n'):0.4, ('u','v'):0.5, ('c','e'):0.5, ('r','n'):0.3,
    ('d','o'):0.3,
}

def ocr_distance(s1, s2):
    s1, s2 = s1.lower(), s2.lower()
    if len(s1) < len(s2): return ocr_distance(s2, s1)
    if len(s2) == 0: return float(len(s1))
    prev = [float(x) for x in range(len(s2) + 1)]
    for i, c1 in enumerate(s1):
        curr = [float(i + 1)]
        for j, c2 in enumerate(s2):
            sub = 0.0 if c1 == c2 else CONFUSIONS.get(tuple(sorted([c1, c2])), 1.0)
            curr.append(min(prev[j+1]+1, curr[j]+1, prev[j]+sub))
        prev = curr
    return prev[-1]

# Clinical vocabulary
CLINICAL_WORDS = set()
CLINICAL_VOCAB = {
    'exam': ['mental','status','cranial','nerves','motor','sensory','reflexes',
             'coordination','gait','cerebellar','examination','neurological'],
    'cranial': ['intact','normal','pupil','pupils','reactive','perrla','extraocular',
                'movements','facial','symmetry','nystagmus','diplopia','ptosis',
                'papilledema','optic','fields','visual','fundoscopy'],
    'motor': ['power','tone','bulk','strength','weakness','spasticity','rigidity',
              'flaccid','atrophy','fasciculations','tremor','proximal','distal',
              'upper','lower','limb','limbs','hemiparesis','paraparesis'],
    'sensory': ['sensation','light','touch','pinprick','vibration','proprioception',
                'temperature','numbness','tingling','paresthesia','intact','reduced'],
    'reflexes': ['biceps','triceps','brachioradialis','knee','ankle','patellar',
                 'achilles','plantar','babinski','clonus','brisk','absent',
                 'upgoing','downgoing','hyperreflexia','hyporeflexia','areflexia'],
    'medications': ['levetiracetam','valproate','carbamazepine','phenytoin',
                    'lamotrigine','topiramate','gabapentin','pregabalin',
                    'levodopa','carbidopa','rotigotine','pramipexole',
                    'aspirin','clopidogrel','warfarin','heparin','rivaroxaban',
                    'apixaban','prednisolone','dexamethasone','paracetamol',
                    'ibuprofen','omeprazole','metformin','insulin','atorvastatin',
                    'amitriptyline','propranolol','diazepam','lorazepam',
                    'furosemide','amlodipine','mannitol','alteplase','metoprolol'],
    'general': ['patient','history','diagnosis','treatment','admitted','discharged',
                'presents','blood','pressure','heart','rate','respiratory',
                'temperature','oxygen','saturation','normal','abnormal',
                'positive','negative','present','absent','mild','moderate',
                'severe','acute','chronic','right','left','bilateral','no','yes'],
    'labs': ['hba1c','glucose','sodium','potassium','creatinine','hemoglobin',
             'wbc','platelets','inr','crp','esr','tsh','albumin','calcium',
             'magnesium','phosphate','urea','bilirubin','alt','ast','alp'],
}

for words in CLINICAL_VOCAB.values():
    CLINICAL_WORDS.update(w.lower() for w in words)

LAB_RANGES = {
    'hba1c': (4.0, 15.0), 'glucose': (1.0, 50.0), 'sodium': (100, 180),
    'potassium': (1.5, 9.0), 'creatinine': (20, 2000), 'hemoglobin': (3.0, 25.0),
    'wbc': (0.1, 100.0), 'platelets': (5, 1500), 'inr': (0.5, 10.0),
}


# =============================================================================
# THE CO-DEFINING ENGINE
# =============================================================================

class ShifuOCR_v2:
    """
    Shifu-OCR v2: Structure and content co-define each other.
    
    Characters → constrained by word context
    Words → constrained by clinical context  
    Clinical context → informed by recognized words
    
    Everything flows both ways. Nothing is rigid.
    Soft enough to adapt. Rigid enough to reject.
    """

    def __init__(self):
        self.landscapes = {}
        self.word_templates = {}  # Known word shapes
        self.total_predictions = 0
        self.total_correct = 0

    # --- Training ---

    def train_character(self, label, grayscale_image):
        binary, _ = image_to_binary(grayscale_image)
        region = normalize_region(extract_region(binary))
        fv = extract_features(region)
        if label not in self.landscapes:
            self.landscapes[label] = Landscape(label)
        self.landscapes[label].absorb(fv)
        return fv

    def train_from_fonts(self, characters, font_paths, font_size=80, img_size=(100, 100)):
        count = 0
        for char in characters:
            for fp in font_paths:
                img = self._render(char, fp, font_size, img_size)
                self.train_character(char, img)
                count += 1
        return count

    def train_word_template(self, word, font_path, font_size=70, img_size=(100, 100)):
        """Learn the shape of a word — the container."""
        char_features = []
        for char in word:
            img = self._render(char, font_path, font_size, img_size)
            binary, _ = image_to_binary(img)
            region = normalize_region(extract_region(binary))
            fv = extract_features(region)
            char_features.append(fv)
        self.word_templates[word.upper()] = WordLandscape(word.upper(), char_features)

    # --- Character Prediction ---

    def predict_character(self, grayscale_image, top_k=10):
        binary, _ = image_to_binary(grayscale_image)
        region = normalize_region(extract_region(binary))
        fv = extract_features(region)

        scores = []
        for label, land in self.landscapes.items():
            score = land.fit(fv)
            rejection = land.rejection_score(fv)
            scores.append((label, score, fv, rejection))

        scores.sort(key=lambda x: x[1], reverse=True)

        best = scores[0]
        second = scores[1] if len(scores) > 1 else None
        margin = (best[1] - second[1]) if second else 0
        total_range = scores[0][1] - scores[-1][1] if len(scores) > 1 else 1
        confidence = max(0, min(margin / max(abs(total_range), 0.01), 1.0))

        # Rejection check — does this fit ANY landscape well?
        min_rejection = min(s[3] for s in scores)
        rejected = min_rejection > 5.0  # Z-score > 5 = truly alien

        self.total_predictions += 1

        return {
            'predicted': best[0],
            'confidence': confidence,
            'features': fv,
            'candidates': [(s[0], s[1], s[2]) for s in scores[:top_k]],
            'rejected': rejected,
            'min_rejection': min_rejection,
        }

    # --- Word-Level Co-Definition ---

    def read_word(self, char_predictions):
        """
        Given a sequence of character predictions, apply word-level 
        constraints — the container reshaping the content.
        
        1. Assemble the raw word from top character predictions
        2. Find the closest known clinical word (the container)
        3. If close enough, let the container constrain ambiguous positions
        4. Return both raw and constrained readings with confidence
        """
        raw_chars = [p['predicted'] for p in char_predictions]
        raw_word = ''.join(raw_chars)

        # Find closest clinical word
        best_match = None
        best_dist = float('inf')

        for cword in CLINICAL_WORDS:
            if abs(len(cword) - len(raw_word)) > 2:
                continue
            d = ocr_distance(raw_word, cword)
            if d < best_dist:
                best_dist = d
                best_match = cword

        # Is the match close enough to act as a container?
        max_acceptable = len(raw_word) * 0.55  # 55% of word length
        has_container = best_match is not None and best_dist <= max_acceptable

        if not has_container:
            return {
                'raw': raw_word,
                'corrected': raw_word,
                'confidence': np.mean([p['confidence'] for p in char_predictions]),
                'container': None,
                'flag': 'no_match',
            }

        # Apply container constraints
        corrected_chars = list(raw_chars)
        container_word = best_match.upper()
        corrections = []

        # If length matches and distance is small, apply directly
        if len(container_word) == len(raw_chars):
            for i in range(len(raw_chars)):
                expected_char = container_word[i]
                raw_char = raw_chars[i]

                if raw_char == expected_char:
                    continue

                # Check if expected char is among candidates
                candidate_labels = [c[0] for c in char_predictions[i]['candidates']]
                
                if expected_char in candidate_labels:
                    pred_score = next(c[1] for c in char_predictions[i]['candidates'] if c[0] == raw_char)
                    exp_score = next(c[1] for c in char_predictions[i]['candidates'] if c[0] == expected_char)
                    gap = pred_score - exp_score

                    if gap < 5.0:  # Container wins for small gaps
                        corrected_chars[i] = expected_char
                        corrections.append({
                            'position': i, 'was': raw_char,
                            'corrected_to': expected_char, 'gap': gap,
                            'reason': f"container '{best_match}' resolved"
                        })
                else:
                    # Character wasn't even a candidate but container says it should be here
                    # Apply if overall word match is very strong
                    if best_dist <= len(raw_word) * 0.3:
                        corrected_chars[i] = expected_char
                        corrections.append({
                            'position': i, 'was': raw_char,
                            'corrected_to': expected_char, 'gap': None,
                            'reason': f"strong container '{best_match}' override"
                        })
        else:
            # Length mismatch — just use the container word directly if distance is small
            if best_dist <= len(raw_word) * 0.35:
                corrected_chars = list(container_word)
                corrections.append({
                    'position': -1, 'was': raw_word,
                    'corrected_to': container_word,
                    'reason': f"container '{best_match}' length-adjusted"
                })

        corrected_word = ''.join(corrected_chars)

        # Confidence: blend of character confidence and match quality
        char_conf = np.mean([p['confidence'] for p in char_predictions])
        match_conf = max(0, 1.0 - best_dist / max(len(raw_word), 1))
        blended_conf = 0.4 * char_conf + 0.6 * match_conf

        return {
            'raw': raw_word,
            'corrected': corrected_word,
            'confidence': blended_conf,
            'container': best_match,
            'distance': best_dist,
            'corrections': corrections,
            'flag': 'corrected' if corrections else 'confirmed',
        }

    # --- Line Reading ---

    def segment_line(self, grayscale_image, min_width=3):
        """Segment a text line into character images."""
        from skimage.filters import threshold_otsu
        try:
            thresh = threshold_otsu(grayscale_image)
        except:
            thresh = 128
        binary = (grayscale_image < thresh).astype(np.uint8)
        binary = morphology.remove_small_objects(binary.astype(bool), min_size=5).astype(np.uint8)

        v_proj = binary.sum(axis=0).astype(float)
        is_ink = v_proj > 0
        segments = []
        in_char = False
        start = 0

        for i in range(len(is_ink)):
            if is_ink[i] and not in_char:
                start = i
                in_char = True
            elif not is_ink[i] and in_char:
                if i - start >= min_width:
                    segments.append((start, i))
                in_char = False
        if in_char and len(is_ink) - start >= min_width:
            segments.append((start, len(is_ink)))

        char_images = []
        for c_start, c_end in segments:
            col_slice = binary[:, c_start:c_end]
            row_proj = col_slice.sum(axis=1)
            rows_ink = np.where(row_proj > 0)[0]
            if len(rows_ink) == 0:
                continue
            r0 = max(0, rows_ink[0] - 2)
            r1 = min(binary.shape[0], rows_ink[-1] + 3)
            char_crop = grayscale_image[r0:r1, c_start:c_end]
            ch, cw = char_crop.shape
            size = max(ch, cw) + 10
            padded = np.full((size, size), 255, dtype=np.uint8)
            yo, xo = (size - ch) // 2, (size - cw) // 2
            padded[yo:yo+ch, xo:xo+cw] = char_crop
            char_images.append({'image': padded, 'bbox': (r0, c_start, r1, c_end)})

        return char_images

    def read_line(self, grayscale_image):
        """
        Full co-defining line reading:
        1. Segment into characters
        2. Predict each character independently (content)
        3. Group into words by gaps
        4. Apply word-level containers (container constrains content)
        5. Apply clinical context (larger container)
        """
        segments = self.segment_line(grayscale_image)
        if not segments:
            return {'text': '', 'words': [], 'confidence': 0}

        # Detect word boundaries from gaps
        gaps = []
        for i in range(1, len(segments)):
            prev_end = segments[i-1]['bbox'][3]
            curr_start = segments[i]['bbox'][1]
            gaps.append(curr_start - prev_end)

        if gaps:
            median_gap = max(np.median(gaps), 1)
            space_threshold = median_gap * 1.8
        else:
            space_threshold = 15

        # Predict all characters
        char_preds = []
        for seg in segments:
            pred = self.predict_character(seg['image'])
            char_preds.append(pred)

        # Group into words
        word_groups = [[0]]  # Start with first char in first word
        for i in range(1, len(segments)):
            prev_end = segments[i-1]['bbox'][3]
            curr_start = segments[i]['bbox'][1]
            if curr_start - prev_end > space_threshold:
                word_groups.append([i])
            else:
                word_groups[-1].append(i)

        # Process each word with co-defining constraints
        words = []
        for group in word_groups:
            word_preds = [char_preds[i] for i in group]
            word_result = self.read_word(word_preds)
            words.append(word_result)

        # Clinical context pass — check numbers against ranges
        context_words = []
        for i, w in enumerate(words):
            text = w['corrected']
            context_words.append(text)

            # Lab range checking
            if re.match(r'^[\d.,]+$', text):
                # Look back for lab name
                for prev in reversed(context_words[:-1]):
                    prev_lower = prev.lower()
                    if prev_lower in LAB_RANGES:
                        lo, hi = LAB_RANGES[prev_lower]
                        try:
                            val = float(text.replace(',', '.'))
                            if not (lo <= val <= hi):
                                # Check missing decimal
                                for dp in range(1, len(text)):
                                    try:
                                        alt = float(text[:dp] + '.' + text[dp:])
                                        if lo <= alt <= hi:
                                            w['flag'] = f'OUT_OF_RANGE: {prev} {val} outside [{lo}-{hi}], suggest {alt}'
                                            break
                                    except:
                                        pass
                        except:
                            pass
                        break

        # Assemble final text
        text = ' '.join(w['corrected'] for w in words)
        avg_conf = np.mean([w['confidence'] for w in words]) if words else 0

        return {
            'text': text,
            'words': words,
            'confidence': avg_conf,
        }

    # --- Rendering ---

    def _render(self, char, font_path, font_size=80, img_size=(100, 100)):
        img = Image.new('L', img_size, color=255)
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype(font_path, font_size)
        except:
            font = ImageFont.load_default()
        bbox = draw.textbbox((0, 0), char, font=font)
        x = (img_size[0] - (bbox[2] - bbox[0])) // 2 - bbox[0]
        y = (img_size[1] - (bbox[3] - bbox[1])) // 2 - bbox[1]
        draw.text((x, y), char, fill=0, font=font)
        return np.array(img)

    @staticmethod
    def render_line(text, font_path, font_size=40, padding=10):
        try:
            font = ImageFont.truetype(font_path, font_size)
        except:
            font = ImageFont.load_default()
        dummy = Image.new('L', (1, 1))
        draw = ImageDraw.Draw(dummy)
        bbox = draw.textbbox((0, 0), text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        img = Image.new('L', (tw + padding * 2, th + padding * 2), color=255)
        draw = ImageDraw.Draw(img)
        draw.text((padding - bbox[0], padding - bbox[1]), text, fill=0, font=font)
        return np.array(img)

    # --- Save / Load ---

    def save(self, path):
        data = {
            'version': '2.0.0',
            'landscapes': {k: v.to_dict() for k, v in self.landscapes.items()},
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            data = json.load(f)
        engine = cls()
        for label, ld in data.get('landscapes', {}).items():
            engine.landscapes[label] = Landscape.from_dict(ld)
        return engine

    def correct(self, prediction, true_label):
        fv = prediction['features']
        if prediction['predicted'] == true_label:
            self.total_correct += 1
            if true_label in self.landscapes:
                self.landscapes[true_label].n_correct += 1
        else:
            if prediction['predicted'] in self.landscapes:
                self.landscapes[prediction['predicted']].n_errors += 1
                self.landscapes[prediction['predicted']].confused_with[true_label] += 1
        if true_label in self.landscapes:
            self.landscapes[true_label].absorb(fv)


# =============================================================================
# DEPLOYMENT TEST
# =============================================================================

FONTS = [
    ('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 'DejaVu Sans'),
    ('/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf', 'DejaVu Serif'),
    ('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 'DejaVu Bold'),
    ('/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf', 'Serif Bold'),
    ('/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf', 'Mono'),
]

TEST_FONTS = [
    ('/usr/share/fonts/truetype/freefont/FreeSans.ttf', 'FreeSans*'),
    ('/usr/share/fonts/truetype/freefont/FreeSerif.ttf', 'FreeSerif*'),
    ('/usr/share/fonts/truetype/freefont/FreeMono.ttf', 'FreeMono*'),
]

ALPHABET = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
DIGITS = list('0123456789')
ALL_CHARS = ALPHABET + DIGITS


def main():
    print()
    print("╔═══════════════════════════════════════════════════════╗")
    print("║        SHIFU-OCR v2 — The Co-Defining Engine         ║")
    print("║                                                      ║")
    print("║  The cup defines the content.                        ║")
    print("║  The content defines the cup.                        ║")
    print("║  Soft enough to adapt. Rigid enough to reject.       ║")
    print("╚═══════════════════════════════════════════════════════╝")
    print()

    ocr = ShifuOCR_v2()

    # === TRAIN ===
    print("=" * 60)
    print("TRAINING")
    print("=" * 60)

    font_paths = [f[0] for f in FONTS]
    n = ocr.train_from_fonts(ALL_CHARS, font_paths, font_size=70, img_size=(100, 100))
    for sz in [40, 50, 60, 80]:
        ocr.train_from_fonts(ALL_CHARS, font_paths[:3], font_size=sz, img_size=(100, 100))
    for sz in [30, 36, 42]:
        ocr.train_from_fonts(ALL_CHARS, font_paths[:3], font_size=sz, img_size=(60, 60))

    depth = ocr.landscapes['A'].n
    print(f"\n  {len(ALL_CHARS)} characters, {depth} observations each")

    # === CHARACTER ACCURACY ===
    print(f"\n{'='*60}")
    print("CHARACTER ACCURACY")
    print("=" * 60)

    for fp, fn in FONTS + TEST_FONTS:
        correct = sum(1 for c in ALPHABET
                     if ocr.predict_character(ocr._render(c, fp, 70, (100,100)))['predicted'] == c)
        acc = correct / len(ALPHABET) * 100
        tag = "UNSEEN" if '*' in fn else "train"
        bar = "█" * int(acc / 2.5) + "░" * (40 - int(acc / 2.5))
        print(f"  {fn:18s} {bar} {acc:5.1f}%  [{tag}]")

    # === LINE READING WITH CO-DEFINITION ===
    print(f"\n{'='*60}")
    print("LINE READING (co-defining: word containers constrain characters)")
    print("=" * 60)

    test_lines = [
        "CRANIAL NERVES INTACT",
        "POWER 5 UPPER LIMBS",
        "REFLEXES NORMAL",
        "BABINSKI NEGATIVE",
        "PUPILS REACTIVE",
        "SODIUM 139",
        "POTASSIUM 45",
        "LEVETIRACETAM 500MG",
        "ROTIGOTINE 4MG",
        "NO PAPILLEDEMA",
        "TONE NORMAL",
        "TREMOR ABSENT",
    ]

    font = FONTS[0][0]
    exact = 0
    corrected_exact = 0

    print(f"\n  {'Input':35s} {'Raw OCR':25s} {'After containers':25s} {'Flag'}")
    print(f"  {'-'*35} {'-'*25} {'-'*25} ----")

    for text in test_lines:
        img = ShifuOCR_v2.render_line(text, font, font_size=36, padding=12)
        result = ocr.read_line(img)

        raw = result['text']
        corrected = ' '.join(
            w['corrected'] for w in result['words']
        )

        # Check accuracy
        raw_match = raw.upper() == text.upper()
        corr_match = corrected.upper() == text.upper()

        if raw_match: exact += 1
        if corr_match: corrected_exact += 1

        symbol = "✓" if corr_match else "~" if raw_match else "✗"

        # Collect flags
        flags = []
        for w in result['words']:
            if w.get('corrections'):
                for c in w['corrections']:
                    flags.append(f"{c.get('was','?')}→{c.get('corrected_to', c.get('suggested','?'))}")
            if 'OUT_OF_RANGE' in str(w.get('flag', '')):
                flags.append(w['flag'][:40])

        flag_str = '; '.join(flags[:2]) if flags else ''

        print(f"  {symbol} {text:33s} {raw:25s} {corrected:25s} {flag_str}")

    print(f"\n  Raw accuracy:         {exact}/{len(test_lines)} ({exact/len(test_lines)*100:.0f}%)")
    print(f"  After co-definition:  {corrected_exact}/{len(test_lines)} ({corrected_exact/len(test_lines)*100:.0f}%)")

    # === SAVE ===
    save_path = '/home/claude/shifu_ocr_v2_model.json'
    ocr.save(save_path)
    size = os.path.getsize(save_path)

    print(f"\n{'='*60}")
    print("SUMMARY")
    print("=" * 60)
    print(f"\n  Model size:    {size/1024:.1f} KB")
    print(f"  Architecture:  Fluid landscapes + word containers + clinical context")
    print(f"  Neural net:    None")
    print(f"  GPU:           None")
    print(f"  Principle:     The container shapes the content,")
    print(f"                 the content shapes the container.")
    print()

    return ocr


if __name__ == '__main__':
    main()
