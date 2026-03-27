"""
Shifu Accommodation
====================
The principle that connects everything Shifu does.

    accommodate(hypotheses, scorer) -> sharpest

The eye does not see with a fixed lens. It ACCOMMODATES.
"""

import numpy as np
from scipy import ndimage
from skimage import morphology


def _segment_vp(binary, grayscale, min_char_width=3):
    v_proj = binary.sum(axis=0).astype(float)
    is_ink = v_proj > 0
    segments = []; in_char = False; start = 0
    for i in range(len(is_ink)):
        if is_ink[i] and not in_char: start = i; in_char = True
        elif not is_ink[i] and in_char:
            if i - start >= min_char_width: segments.append((start, i))
            in_char = False
    if in_char and len(is_ink) - start >= min_char_width:
        segments.append((start, len(is_ink)))
    chars = []
    for c0, c1 in segments:
        rows = np.where(binary[:, c0:c1].sum(axis=1) > 0)[0]
        if len(rows) == 0: continue
        r0 = max(0, rows[0] - 2); r1 = min(binary.shape[0], rows[-1] + 3)
        crop = grayscale[r0:r1, c0:c1]
        ch, cw = crop.shape; size = max(ch, cw) + 10
        padded = np.full((size, size), 255, dtype=np.uint8)
        yo, xo = (size - ch) // 2, (size - cw) // 2
        padded[yo:yo + ch, xo:xo + cw] = crop
        chars.append({'image': padded, 'bbox': (r0, c0, r1, c1)})
    return chars


def _segment_cc(binary, grayscale):
    labeled, n = ndimage.label(binary)
    if n == 0: return []
    comps = []
    for i in range(1, n + 1):
        coords = np.argwhere(labeled == i)
        if len(coords) < 4: continue
        r0, c0 = coords.min(axis=0); r1, c1 = coords.max(axis=0)
        h, w = r1 - r0 + 1, c1 - c0 + 1
        if h < 4 or w < 2: continue
        if w / max(h, 1) > 8 or h / max(w, 1) > 10: continue
        comps.append({'r0': r0, 'c0': c0, 'r1': r1, 'c1': c1, 'area': int((labeled == i).sum())})
    comps.sort(key=lambda x: x['c0'])
    merged = []; used = set()
    for i, c in enumerate(comps):
        if i in used: continue
        r0, c0, r1, c1, area = c['r0'], c['c0'], c['r1'], c['c1'], c['area']
        for j in range(i + 1, len(comps)):
            if j in used: continue
            o = comps[j]
            if min(c1, o['c1']) - max(c0, o['c0']) < 0: continue
            if o['area'] < area * 0.3 or area < o['area'] * 0.3:
                r0, c0 = min(r0, o['r0']), min(c0, o['c0'])
                r1, c1 = max(r1, o['r1']), max(c1, o['c1'])
                area += o['area']; used.add(j)
        merged.append({'r0': r0, 'c0': c0, 'r1': r1, 'c1': c1})
    merged.sort(key=lambda x: x['c0'])
    chars = []
    for m in merged:
        r0p = max(0, m['r0'] - 2); r1p = min(grayscale.shape[0], m['r1'] + 3)
        c0p = max(0, m['c0'] - 1); c1p = min(grayscale.shape[1], m['c1'] + 2)
        crop = grayscale[r0p:r1p, c0p:c1p]
        ch, cw = crop.shape; size = max(ch, cw) + 10
        padded = np.full((size, size), 255, dtype=np.uint8)
        yo, xo = (size - ch) // 2, (size - cw) // 2
        padded[yo:yo + ch, xo:xo + cw] = crop
        chars.append({'image': padded, 'bbox': (r0p, c0p, r1p, c1p)})
    return chars


def accommodate_segmentation(ocr_engine, grayscale_image, min_char_width=3):
    from skimage.filters import threshold_otsu
    try: thresh = threshold_otsu(grayscale_image)
    except: thresh = 128
    binary = (grayscale_image < thresh).astype(np.uint8)
    binary = morphology.remove_small_objects(binary.astype(bool), min_size=5).astype(np.uint8)
    vp = _segment_vp(binary, grayscale_image, min_char_width)
    cc = _segment_cc(binary, grayscale_image)
    if not vp and not cc: return [], 'none', 0, 0
    if not vp: return cc, 'cc', 0, 0
    if not cc: return vp, 'vp', 0, 0
    def avg_conf(chars, n=5):
        sample = chars[max(0, (len(chars) - n) // 2):][:n]
        confs = [ocr_engine.predict_character(c['image'])['confidence'] for c in sample]
        return sum(confs) / max(len(confs), 1)
    vp_conf = avg_conf(vp); cc_conf = avg_conf(cc)
    if cc_conf > vp_conf + 0.05: return cc, 'cc', vp_conf, cc_conf
    return vp, 'vp', vp_conf, cc_conf


class ShifuAccommodation:
    def __init__(self, ocr_engine, language_engine=None):
        self.ocr = ocr_engine
        self.lang = language_engine

    def read_line_accommodated(self, grayscale_image, space_threshold=None):
        chars, method, vp_conf, cc_conf = accommodate_segmentation(self.ocr, grayscale_image)
        if not chars:
            return {'text': '', 'confidence': 0, 'method': 'none', 'accommodated': True}
        predictions = []
        for c in chars:
            pred = self.ocr.predict_character(c['image'])
            predictions.append({'char': pred['predicted'], 'confidence': pred['confidence'],
                                'candidates': pred.get('candidates', []), 'bbox': c['bbox']})
        if space_threshold is None:
            gaps = [chars[i]['bbox'][1] - chars[i-1]['bbox'][3] for i in range(1, len(chars))]
            space_threshold = np.median(gaps) * 2.0 if gaps else 20
        parts = []
        for i, p in enumerate(predictions):
            if i > 0 and chars[i]['bbox'][1] - chars[i-1]['bbox'][3] > space_threshold:
                parts.append(' ')
            parts.append(p['char'])
        raw_text = ''.join(parts)
        avg_conf = np.mean([p['confidence'] for p in predictions]) if predictions else 0
        result = {'text': raw_text, 'words': raw_text.split(), 'characters': predictions,
                  'confidence': float(avg_conf), 'segmentation_method': method,
                  'vp_confidence': float(vp_conf), 'cc_confidence': float(cc_conf), 'accommodated': True}
        if self.lang is not None and result['words']:
            coherence = self.lang.scoreSentence(raw_text)
            result['coherence'] = {'forward': coherence.get('coherence', 0),
                                   'corrected': coherence.get('correctedCoherence', 0),
                                   'settled': coherence.get('settledCoherence', 0)}
        return result

    def validate_correction(self, original, corrected):
        if self.lang is None: return 'accept'
        result = self.lang.accommodateSentence([original, corrected])
        if result['best'] == corrected and result['margin'] > 0.02: return 'accept'
        elif result['best'] == original and result['margin'] > 0.05: return 'reject'
        return 'review'

    def accommodate_page(self, grayscale_image):
        page_result = self.ocr.read_page(grayscale_image)
        if self.lang is None: return page_result
        lines = page_result.get('lines', [])
        needs_review = []
        for i, line_text in enumerate(lines):
            if not line_text.strip(): continue
            coherence = self.lang.scoreSentence(line_text)
            if coherence.get('settledCoherence', 0) < 0.2:
                needs_review.append({'line_index': i, 'text': line_text,
                                     'coherence': coherence.get('settledCoherence', 0)})
        page_result['needs_review'] = needs_review
        page_result['review_count'] = len(needs_review)
        return page_result
