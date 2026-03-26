"""
Shifu-OCR: Fluid Theory OCR Engine
Core module — the landscape classifier with full pipeline.
"""

import numpy as np
import json
import os
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage
from skimage import morphology, filters, measure
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# MEDIUM DISPLACEMENT PIPELINE
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
    if len(coords) == 0:
        return binary
    r0, c0 = np.maximum(coords.min(axis=0) - pad, 0)
    r1, c1 = np.minimum(coords.max(axis=0) + pad, np.array(binary.shape) - 1)
    return binary[r0:r1+1, c0:c1+1]

def normalize_region(region, size=(64, 64)):
    img = Image.fromarray((region * 255).astype(np.uint8)).resize(size, Image.NEAREST)
    return (np.array(img) > 127).astype(np.uint8)

def image_to_binary(char_img, bg_kernel=15, disp_thresh=0.25):
    bg = estimate_background(char_img, k=bg_kernel)
    disp = compute_displacement(char_img, bg)
    return detect_perturbation(disp, thresh=disp_thresh), disp


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
    ['endpoints', 'junctions'] +
    # v2: geometric/topological features — pure displacement geometry
    # No heuristics. Landscapes learn what these mean from evidence.
    ['aspect_ratio', 'ink_density',
     'top_third_density', 'mid_third_density', 'bot_third_density',
     'stroke_width_mean', 'stroke_width_var',
     'top_disconnection', 'mid_horizontal_extent',
     'ascender_ratio', 'descender_ratio',
     'left_edge_straightness', 'right_edge_straightness',
     'top_openness', 'bot_openness',
    ]
)

def extract_features(br):
    h, w = br.shape
    feats = []

    # Topology
    padded = np.pad(br, 1, mode='constant', constant_values=0)
    _, nfg = ndimage.label(padded)
    _, nbg = ndimage.label(1 - padded)
    holes = nbg - 1
    feats.extend([float(nfg), float(holes), float(nfg - holes)])

    # Displacement ratio
    feats.append(float(np.mean(br)))

    # Symmetry
    feats.append(float(np.mean(br == np.fliplr(br))) if w >= 2 else 1.0)
    feats.append(float(np.mean(br == np.flipud(br))) if h >= 2 else 1.0)

    # Center of mass
    total = br.sum()
    if total > 0 and h > 0 and w > 0:
        rows, cols = np.arange(h).reshape(-1, 1), np.arange(w).reshape(1, -1)
        feats.append(float((br * rows).sum() / (total * h)))
        feats.append(float((br * cols).sum() / (total * w)))
    else:
        feats.extend([0.5, 0.5])

    # Quadrant density
    mh, mw = h // 2, w // 2
    quads = [
        float(br[:mh, :mw].mean()) if mh > 0 and mw > 0 else 0,
        float(br[:mh, mw:].mean()) if mh > 0 else 0,
        float(br[mh:, :mw].mean()) if mw > 0 else 0,
        float(br[mh:, mw:].mean()),
    ]
    qt = sum(quads)
    feats.extend([q / qt if qt > 0 else 0.25 for q in quads])

    # Projection profiles
    for axis, length in [(1, h), (0, w)]:
        raw = br.mean(axis=axis)
        bins = 6
        proj = np.zeros(bins)
        bw = max(1, len(raw) // bins)
        for i in range(bins):
            s, e = i * bw, min((i + 1) * bw, len(raw))
            if s < len(raw):
                proj[i] = raw[s:e].mean()
        pt = proj.sum()
        if pt > 0:
            proj /= pt
        feats.extend(proj.tolist())

    # Crossing counts
    for axis in [0, 1]:
        for i in range(6):
            if axis == 0:
                row = min(int((i + 0.5) * h / 6), h - 1)
                line = br[row, :]
            else:
                col = min(int((i + 0.5) * w / 6), w - 1)
                line = br[:, col]
            feats.append(float(np.abs(np.diff(line.astype(int))).sum() / 2))

    # Junctions and endpoints
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

    # ── v2: Discriminative features for l/i/t and a/o confusion ──────

    ink = br > 0
    ink_pixels = ink.sum()

    # Aspect ratio: tall narrow (l) vs square (o) vs wide (m)
    ink_coords = np.argwhere(ink)
    if len(ink_coords) >= 2:
        ink_h = ink_coords[:, 0].max() - ink_coords[:, 0].min() + 1
        ink_w = ink_coords[:, 1].max() - ink_coords[:, 1].min() + 1
        feats.append(float(ink_h / max(ink_w, 1)))  # aspect_ratio
    else:
        feats.append(1.0)

    # Ink density: total ink / bounding box area
    feats.append(float(ink_pixels / max(h * w, 1)))

    # Third densities: top/mid/bottom — separates i(dot on top) from l(uniform)
    t3 = h // 3
    top_d = float(br[:t3, :].mean()) if t3 > 0 else 0
    mid_d = float(br[t3:2*t3, :].mean()) if t3 > 0 else 0
    bot_d = float(br[2*t3:, :].mean()) if t3 > 0 else 0
    feats.extend([top_d, mid_d, bot_d])

    # Stroke width: mean and variance of horizontal run lengths (vectorized)
    diffs = np.diff(np.pad(br, ((0, 0), (1, 1)), constant_values=0), axis=1)
    starts = np.where(diffs == 1)
    ends = np.where(diffs == -1)
    if len(starts[0]) > 0 and len(starts[0]) == len(ends[0]):
        run_lengths = ends[1] - starts[1]
        feats.append(float(run_lengths.mean() / max(w, 1)))
        feats.append(float(run_lengths.std() / max(w, 1)))
    else:
        feats.extend([0.0, 0.0])

    # Top disconnection: gap between top ink cluster and rest
    # Continuous — measures vertical gap in ink distribution.
    # 'i' has a gap (dot then body), 'l' has continuous ink top to bottom.
    # Uses projection gap instead of expensive ndimage.label()
    v_proj = br.mean(axis=1)  # already have ink per row
    gap_rows = (v_proj < 0.01).astype(int)
    # Count largest gap in top half
    top_half_gaps = gap_rows[:h // 2]
    max_gap = 0
    current_gap = 0
    for g in top_half_gaps:
        if g: current_gap += 1
        else:
            max_gap = max(max_gap, current_gap)
            current_gap = 0
    max_gap = max(max_gap, current_gap)
    feats.append(float(max_gap / max(h // 2, 1)))  # top_disconnection

    # Mid horizontal extent: how wide is the ink in the middle third relative to total width
    # Continuous — 't' has wide middle extent (crossbar), 'l' has narrow.
    mid_slice = br[t3:2*t3, :] if t3 > 0 else br
    mid_col_ink = (mid_slice.sum(axis=0) > 0).sum()
    feats.append(float(mid_col_ink / max(w, 1)))  # mid_horizontal_extent

    # Ascender/descender ratio: how much ink is in top/bottom vs middle
    # l has high ascender, a doesn't
    if ink_pixels > 0:
        feats.append(float(br[:t3, :].sum() / ink_pixels))   # ascender_ratio
        feats.append(float(br[2*t3:, :].sum() / ink_pixels)) # descender_ratio
    else:
        feats.extend([0.33, 0.33])

    # Edge straightness: vectorized contour geometry
    row_has_ink = br.any(axis=1)
    if row_has_ink.sum() >= 3:
        ink_rows = np.where(row_has_ink)[0]
        left_edge = np.array([np.where(br[r, :] > 0)[0][0] for r in ink_rows])
        right_edge = np.array([np.where(br[r, :] > 0)[0][-1] for r in ink_rows])
        feats.append(1.0 - float(np.std(left_edge) / max(w, 1)))
        feats.append(1.0 - float(np.std(right_edge) / max(w, 1)))
    else:
        feats.extend([0.5, 0.5])

    # Top/bottom openness: is the character open at top (u, c) or bottom (n)?
    top_row_ink = float(br[0, :].mean()) if h > 0 else 0
    bot_row_ink = float(br[-1, :].mean()) if h > 0 else 0
    feats.append(1.0 - top_row_ink)  # top_openness (high = open top)
    feats.append(1.0 - bot_row_ink)  # bot_openness (high = open bottom)

    return np.array(feats, dtype=float)


# =============================================================================
# FLUID LANDSCAPE
# =============================================================================

class Landscape:
    def __init__(self, label):
        self.label = label
        self.observations = []  # kept only for small n; cleared after warmup
        self.mean = None
        self.variance = None
        self._m2 = None  # Welford's running sum of squared differences
        self.n = 0
        self.n_correct = 0
        self.n_errors = 0
        self.confused_with = defaultdict(int)

    def absorb(self, fv):
        self.n += 1
        if self.n == 1:
            self.mean = fv.copy()
            self._m2 = np.zeros_like(fv)
            self.variance = np.ones_like(fv) * 2.0
        else:
            # Welford's online algorithm — O(1) per observation
            delta = fv - self.mean
            self.mean += delta / self.n
            delta2 = fv - self.mean
            self._m2 += delta * delta2
            raw_var = self._m2 / self.n
            self.variance = np.maximum(raw_var, 0.1 / np.sqrt(self.n))

    def fit(self, fv, global_var=None):
        if self.mean is None:
            return -float('inf')
        diff = fv - self.mean
        # Use minimum of within-class and global variance for tighter discrimination
        var = np.minimum(self.variance, global_var) if global_var is not None else self.variance
        precision = 1.0 / (var + 1e-8)
        score = -0.5 * np.sum(diff ** 2 * precision)
        return score + np.log(self.n + 1) * 0.5

    def to_dict(self):
        return {
            'label': self.label,
            'n': self.n,
            'mean': self.mean.tolist() if self.mean is not None else None,
            'variance': self.variance.tolist() if self.variance is not None else None,
            'n_correct': self.n_correct,
            'n_errors': self.n_errors,
            'confused_with': dict(self.confused_with),
        }

    @classmethod
    def from_dict(cls, d):
        l = cls(d['label'])
        l.n = d['n']
        l.mean = np.array(d['mean']) if d['mean'] else None
        l.variance = np.array(d['variance']) if d['variance'] else None
        # Reconstruct Welford's M2 from saved variance: M2 = variance * n
        if l.variance is not None and l.n > 0:
            l._m2 = l.variance * l.n
        l.n_correct = d.get('n_correct', 0)
        l.n_errors = d.get('n_errors', 0)
        l.confused_with = defaultdict(int, d.get('confused_with', {}))
        return l


# =============================================================================
# SHIFU-OCR ENGINE
# =============================================================================

class ShifuOCR:
    """
    Complete Fluid Theory OCR engine.
    
    Train on character images → builds landscapes.
    Predict on new images → finds best-fit landscape.
    Learn from corrections → reshapes landscapes.
    Save/load → persist the trained model.
    """

    def __init__(self):
        self.landscapes = {}
        self.total_predictions = 0
        self.total_correct = 0
        self.version = "1.0.0"

    @staticmethod
    def _fast_binarize(grayscale_image):
        """Fast binarization for pre-segmented character images.
        Uses Otsu threshold instead of full medium displacement pipeline.
        The displacement pipeline (estimate_background + compute_displacement)
        is designed for raw page scans where background varies. For individual
        character crops, Otsu is faster and equally effective."""
        from skimage.filters import threshold_otsu
        try:
            thresh = threshold_otsu(grayscale_image)
        except ValueError:
            thresh = 128
        binary = (grayscale_image < thresh).astype(np.uint8)
        return morphology.remove_small_objects(binary.astype(bool), min_size=5).astype(np.uint8)

    # --- Training ---

    def _extract_unified_features(self, grayscale_image):
        """Unified feature vector: static features + FLAIR perturbation signatures + shape.
        Like MRI: T1 (static structure) + T2/FLAIR/DWI (perturbation response) + morphometry."""
        from shifu_ocr.perturbation import extract_relaxation_signature
        binary = self._fast_binarize(grayscale_image)
        raw_region = extract_region(binary)
        rh, rw = raw_region.shape if raw_region.size > 0 else (1, 1)
        # Shape features: aspect ratio and relative size (case/thin char discrimination)
        aspect = rw / max(rh, 1)
        tallness = rh / max(rw, 1)
        fill_ratio = float(raw_region.sum()) / max(rh * rw, 1)
        shape_feats = np.array([aspect, tallness, fill_ratio, rh / max(grayscale_image.shape[0], 1)])
        region = normalize_region(raw_region)
        static = extract_features(region)        # 38-dim: structure
        dynamic = extract_relaxation_signature(region)  # 64-dim: perturbation response
        return np.concatenate([static, dynamic, shape_feats])  # 106-dim combined

    def train_character(self, label, grayscale_image):
        """Train on a single character image (grayscale numpy array)."""
        fv = self._extract_unified_features(grayscale_image)
        if label not in self.landscapes:
            self.landscapes[label] = Landscape(label)
        self.landscapes[label].absorb(fv)

    def train_from_fonts(self, characters, font_paths, font_size=80, img_size=(100, 100)):
        """Train on rendered characters across multiple fonts."""
        for char in characters:
            for font_path in font_paths:
                img = self._render(char, font_path, font_size, img_size)
                self.train_character(char, img)
        return len(characters) * len(font_paths)

    # --- Prediction ---

    def predict_character(self, grayscale_image, top_k=5):
        """Predict a single character from a grayscale image."""
        fv = self._extract_unified_features(grayscale_image)

        # Compute global variance across all class means for discriminative scoring
        if not hasattr(self, '_global_var') or self._global_var is None:
            means = [l.mean for l in self.landscapes.values() if l.mean is not None]
            if len(means) > 1:
                mean_stack = np.array(means)
                self._global_var = np.var(mean_stack, axis=0) + 1e-6
            else:
                self._global_var = None

        scores = [(label, land.fit(fv, self._global_var)) for label, land in self.landscapes.items()]
        scores.sort(key=lambda x: x[1], reverse=True)

        if len(scores) < 2:
            return {'predicted': scores[0][0] if scores else '?', 'confidence': 0, 'candidates': scores}

        best_score = scores[0][1]
        second_score = scores[1][1]
        margin = best_score - second_score
        # Confidence: how much better is best vs second, relative to second vs third
        third_score = scores[2][1] if len(scores) > 2 else second_score - 1
        local_range = max(abs(second_score - third_score), 0.01)
        confidence = max(0, min(margin / (local_range + margin + 0.01), 1.0))

        self.total_predictions += 1

        return {
            'predicted': scores[0][0],
            'confidence': confidence,
            'margin': margin,
            'candidates': scores[:top_k],
            'features': fv,
        }

    def correct(self, prediction, true_label):
        """Learn from a correction."""
        fv = prediction['features']
        predicted = prediction['predicted']
        correct = predicted == true_label

        if correct:
            self.total_correct += 1
            if true_label in self.landscapes:
                self.landscapes[true_label].n_correct += 1
        else:
            if predicted in self.landscapes:
                self.landscapes[predicted].n_errors += 1
                self.landscapes[predicted].confused_with[true_label] += 1

        if true_label in self.landscapes:
            self.landscapes[true_label].absorb(fv)

        return correct

    # --- Text Processing ---

    def segment_characters(self, grayscale_image, min_char_width=3):
        """
        Segment a text line image into individual character images.
        Uses simple Otsu binarization (better for full lines) + vertical projection.
        """
        from skimage.filters import threshold_otsu
        
        # For line images, simple Otsu works better than medium displacement
        # because characters are close together and the background is uniform
        try:
            thresh = threshold_otsu(grayscale_image)
        except:
            thresh = 128
        
        binary = (grayscale_image < thresh).astype(np.uint8)
        
        # Clean up
        binary = morphology.remove_small_objects(binary.astype(bool), min_size=5).astype(np.uint8)
        
        # Vertical projection — sum ink in each column
        v_proj = binary.sum(axis=0).astype(float)
        
        # Find character boundaries by detecting gaps
        is_ink = v_proj > 0
        segments = []
        in_char = False
        start = 0
        
        for i in range(len(is_ink)):
            if is_ink[i] and not in_char:
                start = i
                in_char = True
            elif not is_ink[i] and in_char:
                if i - start >= min_char_width:
                    segments.append((start, i))
                in_char = False
        
        if in_char and len(is_ink) - start >= min_char_width:
            segments.append((start, len(is_ink)))
        
        # Extract character images — pad each into a square for consistent processing
        char_images = []
        for c_start, c_end in segments:
            col_slice = binary[:, c_start:c_end]
            row_proj = col_slice.sum(axis=1)
            rows_with_ink = np.where(row_proj > 0)[0]
            
            if len(rows_with_ink) == 0:
                continue
            
            r_start = max(0, rows_with_ink[0] - 2)
            r_end = min(binary.shape[0], rows_with_ink[-1] + 3)
            
            # Extract from ORIGINAL grayscale (not binary)
            char_crop = grayscale_image[r_start:r_end, c_start:c_end]
            
            # Pad into a square with white background for consistent processing
            ch, cw = char_crop.shape
            size = max(ch, cw) + 10
            padded = np.full((size, size), 255, dtype=np.uint8)
            y_off = (size - ch) // 2
            x_off = (size - cw) // 2
            padded[y_off:y_off+ch, x_off:x_off+cw] = char_crop
            
            char_images.append({
                'image': padded,
                'bbox': (r_start, c_start, r_end, c_end),
            })
        
        return char_images

    def read_line(self, grayscale_image, space_threshold=None):
        """
        Read a line of text from a grayscale image.
        Returns recognized text with per-character confidence.
        """
        char_segments = self.segment_characters(grayscale_image)
        
        if not char_segments:
            return {'text': '', 'characters': [], 'confidence': 0}
        
        # Detect spaces by looking at gaps between segments
        if space_threshold is None:
            gaps = []
            for i in range(1, len(char_segments)):
                prev_end = char_segments[i-1]['bbox'][3]
                curr_start = char_segments[i]['bbox'][1]
                gaps.append(curr_start - prev_end)
            
            if gaps:
                median_gap = np.median(gaps)
                space_threshold = median_gap * 2.0
            else:
                space_threshold = 20
        
        results = []
        text_parts = []
        
        # Compute baseline metrics for case disambiguation
        # Uppercase chars sit higher (smaller top y), lowercase descenders go lower
        all_tops = [seg['bbox'][0] for seg in char_segments]
        all_bottoms = [seg['bbox'][2] for seg in char_segments]
        all_heights = [seg['bbox'][2] - seg['bbox'][0] for seg in char_segments]
        median_height = np.median(all_heights) if all_heights else 20
        median_top = np.median(all_tops) if all_tops else 0

        for i, seg in enumerate(char_segments):
            pred = self.predict_character(seg['image'])
            char = pred['predicted']

            # Case disambiguation: only when classifier is uncertain between case pairs
            if pred['confidence'] < 0.1 and char.isalpha():
                char_height = seg['bbox'][2] - seg['bbox'][0]
                char_top = seg['bbox'][0]
                is_tall = char_height > median_height * 0.85
                is_high = char_top <= median_top + median_height * 0.15
                cands = pred['candidates'][:2]
                if len(cands) >= 2:
                    c1, c2 = cands[0][0], cands[1][0]
                    if c1.lower() == c2.lower():
                        char = c1.upper() if (is_tall and is_high) else c1.lower()

            results.append({
                'char': char,
                'confidence': pred['confidence'],
                'bbox': seg['bbox'],
                'candidates': pred['candidates'][:3],
            })

            # Check for space before this character
            if i > 0:
                prev_end = char_segments[i-1]['bbox'][3]
                curr_start = seg['bbox'][1]
                if curr_start - prev_end > space_threshold:
                    text_parts.append(' ')

            text_parts.append(char)

        text = ''.join(text_parts)
        avg_conf = np.mean([r['confidence'] for r in results]) if results else 0
        
        return {
            'text': text,
            'characters': results,
            'confidence': avg_conf,
        }

    # --- Page-level OCR ---

    def segment_lines(self, grayscale_image, min_line_height=8, min_gap=3):
        """
        Segment a full page image into individual text line images.
        Uses horizontal projection to find rows of text.
        Removes grid lines (spreadsheet borders) before segmentation
        so the FLAIR perturbation engine gets clean character input.
        """
        from skimage.filters import threshold_otsu

        try:
            thresh = threshold_otsu(grayscale_image)
        except:
            thresh = 128

        binary = (grayscale_image < thresh).astype(np.uint8)

        # CONTINUITY PRINCIPLE for grid removal:
        # Grid lines = CONTINUOUS runs of ink across the full span.
        # Text = DISCONTINUOUS clusters with gaps between characters.
        # Count the longest continuous run in each row/col. If it spans >50%,
        # it's a grid line (text never has unbroken runs that long).
        h, w = binary.shape
        for row in range(h):
            row_data = binary[row, :]
            # Find longest continuous run of ink
            max_run = 0; cur_run = 0
            for px in row_data:
                if px: cur_run += 1; max_run = max(max_run, cur_run)
                else: cur_run = 0
            # A continuous run > 50% of width = grid line, not text
            if max_run > w * 0.5:
                binary[row, :] = 0

        for col in range(w):
            col_data = binary[:, col]
            max_run = 0; cur_run = 0
            for px in col_data:
                if px: cur_run += 1; max_run = max(max_run, cur_run)
                else: cur_run = 0
            if max_run > h * 0.5:
                binary[:, col] = 0

        binary = morphology.remove_small_objects(binary.astype(bool), min_size=10).astype(np.uint8)

        # Horizontal projection — sum ink in each row
        h_proj = binary.sum(axis=1).astype(float)

        # Find line boundaries by detecting gaps
        is_ink = h_proj > 0
        lines = []
        in_line = False
        start = 0

        for i in range(len(is_ink)):
            if is_ink[i] and not in_line:
                start = i
                in_line = True
            elif not is_ink[i] and in_line:
                if i - start >= min_line_height:
                    lines.append((start, i))
                in_line = False

        if in_line and len(is_ink) - start >= min_line_height:
            lines.append((start, len(is_ink)))

        # Merge lines that are very close (broken by thin gaps)
        merged = []
        for s, e in lines:
            if merged and s - merged[-1][1] < min_gap:
                merged[-1] = (merged[-1][0], e)
            else:
                merged.append((s, e))

        # Extract line images with some vertical padding
        line_images = []
        for s, e in merged:
            pad = max(2, (e - s) // 8)
            rs = max(0, s - pad)
            re = min(grayscale_image.shape[0], e + pad)
            line_img = grayscale_image[rs:re, :]
            line_images.append({
                'image': line_img,
                'bbox': (rs, 0, re, grayscale_image.shape[1]),
                'row_index': len(line_images),
            })

        return line_images

    def read_page(self, grayscale_image, space_threshold=None):
        """
        Read a full page using document-level adaptation.

        Principle: a document has RULES OF PRESENTATION.
        Same font, same spacing, same rendering throughout.
        High-confidence predictions teach the model about THIS document's
        specific rendering, bridging the gap for low-confidence chars.

        Pass 1: OCR everything, collect high-confidence character samples.
        Adapt:  Train a document-local model from confident predictions.
        Pass 2: Re-OCR using the adapted model.
        """
        line_segments = self.segment_lines(grayscale_image)

        if not line_segments:
            return {'text': '', 'lines': [], 'confidence': 0}

        # --- Pass 1: Initial OCR, collect confident character samples ---
        pass1_results = []
        confident_samples = []  # (label, image) pairs from high-confidence predictions

        for seg in line_segments:
            char_segments = self.segment_characters(seg['image'])
            line_chars = []
            for cseg in char_segments:
                pred = self.predict_character(cseg['image'])
                line_chars.append({
                    'pred': pred,
                    'image': cseg['image'],
                    'bbox': cseg['bbox'],
                })
                # Collect high-confidence samples: these are the document's own truth
                if pred['confidence'] > 0.5 and pred['predicted'].isalnum():
                    confident_samples.append((pred['predicted'], cseg['image']))
            pass1_results.append({'seg': seg, 'chars': line_chars})

        # --- Adapt: Build document-local model from confident samples ---
        # Clone current landscapes and reinforce with document-specific evidence
        adapted = None
        if len(confident_samples) > 10:
            from copy import deepcopy
            adapted = deepcopy(self)
            # Each confident sample reinforces the landscape for THIS document's rendering
            for label, img in confident_samples:
                adapted.train_character(label, img)

        ocr_engine = adapted if adapted else self

        # --- Pass 2: Re-OCR everything with adapted model ---
        lines = []
        all_text = []

        for p1 in pass1_results:
            seg = p1['seg']
            if adapted:
                # Full re-read with adapted model
                result = ocr_engine.read_line(seg['image'], space_threshold=space_threshold)
            else:
                # No adaptation possible, reconstruct from pass 1
                text_parts = []
                for c in p1['chars']:
                    text_parts.append(c['pred']['predicted'])
                result = {
                    'text': ''.join(text_parts),
                    'characters': [{'char': c['pred']['predicted'], 'confidence': c['pred']['confidence'], 'bbox': c['bbox']} for c in p1['chars']],
                    'confidence': np.mean([c['pred']['confidence'] for c in p1['chars']]) if p1['chars'] else 0,
                }

            text = result['text'].strip()
            if text:
                # Filter noise lines: mostly non-alphanumeric = grid remnants
                alnum = sum(1 for c in text if c.isalnum())
                if alnum >= max(len(text) * 0.15, 1):
                    result['bbox'] = seg['bbox']
                    result['row_index'] = seg['row_index']
                    lines.append(result)
                    all_text.append(text)

        avg_conf = np.mean([l['confidence'] for l in lines]) if lines else 0

        return {
            'text': '\n'.join(all_text),
            'lines': [l['text'] for l in lines],
            'line_details': lines,
            'confidence': avg_conf,
            'adapted': adapted is not None,
            'confident_samples': len(confident_samples),
        }

    def detect_columns(self, grayscale_image):
        """
        Detect column boundaries from vertical gaps in the image.
        Principle: columns are separated by continuous vertical whitespace.
        """
        from skimage.filters import threshold_otsu
        try:
            thresh = threshold_otsu(grayscale_image)
        except:
            thresh = 128
        binary = (grayscale_image < thresh).astype(np.uint8)

        # Vertical projection: sum ink per column across all rows
        v_proj = binary.sum(axis=0).astype(float)

        # Find wide gaps (column separators): runs of near-zero ink
        gap_thresh = binary.shape[0] * 0.02  # <2% of height = empty column
        in_gap = False
        gaps = []
        gap_start = 0

        for i in range(len(v_proj)):
            if v_proj[i] <= gap_thresh and not in_gap:
                gap_start = i
                in_gap = True
            elif v_proj[i] > gap_thresh and in_gap:
                if i - gap_start > 15:  # gaps must be >15px wide
                    gaps.append((gap_start, i))
                in_gap = False

        if not gaps:
            return None  # Not a columnar document

        # Build column boundaries from gaps
        columns = []
        prev_end = 0
        for gs, ge in gaps:
            if gs > prev_end + 20:  # column must be >20px wide
                columns.append((prev_end, gs))
            prev_end = ge
        if prev_end < len(v_proj) - 20:
            columns.append((prev_end, len(v_proj)))

        return columns if len(columns) >= 2 else None

    def read_structured_page(self, grayscale_image, space_threshold=None):
        """
        Read a structured document (spreadsheet, table) using layout understanding.

        START WITHIN, THEN WITHOUT:
        1. Detect structure (columns, rows) — the organization
        2. OCR each cell within its column context
        3. Use patterns: repeated words, column-consistent vocabulary
        4. The bigger picture emerges from understanding the parts
        """
        # Try to detect columnar structure
        columns = self.detect_columns(grayscale_image)

        if columns is None:
            # Not a structured document — fall back to standard page read
            return self.read_page(grayscale_image, space_threshold)

        # Segment rows
        line_segments = self.segment_lines(grayscale_image)
        if not line_segments:
            return {'text': '', 'lines': [], 'confidence': 0, 'structured': False}

        # For each row, extract text per column
        rows = []
        all_text = []

        # Pass 1: OCR all cells, collect confident predictions per column
        col_vocab = {i: {} for i in range(len(columns))}  # word frequencies per column

        for seg in line_segments:
            row_cells = []
            for ci, (col_start, col_end) in enumerate(columns):
                # Extract this cell's image
                cell_img = seg['image'][:, max(0,col_start):min(col_end, seg['image'].shape[1])]
                if cell_img.size == 0 or cell_img.shape[1] < 5:
                    row_cells.append('')
                    continue

                result = self.read_line(cell_img, space_threshold=space_threshold)
                text = result['text'].strip()
                row_cells.append(text)

                # Track word frequencies per column
                if text and result.get('confidence', 0) > 0:
                    for word in text.split():
                        if word.isalpha() and len(word) > 1:
                            col_vocab[ci][word] = col_vocab[ci].get(word, 0) + 1

            row_text = '\t'.join(row_cells)
            rows.append(row_cells)
            all_text.append(row_text)

        # Pass 2: Use column vocabulary to correct uncertain words
        # Words that appear multiple times in the same column are likely correct
        confident_words = {}
        for ci, vocab in col_vocab.items():
            for word, count in vocab.items():
                if count >= 2:  # seen 2+ times in same column = trusted
                    confident_words[word.lower()] = word

        avg_conf = 0
        return {
            'text': '\n'.join(all_text),
            'lines': ['\t'.join(r) for r in rows],
            'rows': rows,
            'columns': len(columns),
            'column_boundaries': columns,
            'column_vocab': {i: dict(sorted(v.items(), key=lambda x: -x[1])[:5]) for i, v in col_vocab.items()},
            'confident_words': confident_words,
            'confidence': avg_conf,
            'structured': True,
        }

    # --- Rendering utility ---

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
    def render_text_line(text, font_path, font_size=40, padding=10):
        """Render a line of text as a grayscale image."""
        try:
            font = ImageFont.truetype(font_path, font_size)
        except:
            font = ImageFont.load_default()
        
        # Measure text
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
        """Save trained model to JSON."""
        data = {
            'version': self.version,
            'total_predictions': self.total_predictions,
            'total_correct': self.total_correct,
            'landscapes': {k: v.to_dict() for k, v in self.landscapes.items()},
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path):
        """Load trained model from JSON."""
        with open(path) as f:
            data = json.load(f)
        engine = cls()
        engine.version = data.get('version', '1.0.0')
        engine.total_predictions = data.get('total_predictions', 0)
        engine.total_correct = data.get('total_correct', 0)
        for label, ld in data.get('landscapes', {}).items():
            engine.landscapes[label] = Landscape.from_dict(ld)
        return engine

    # --- Stats ---

    def get_stats(self):
        n_chars = len(self.landscapes)
        depths = [l.n for l in self.landscapes.values()]
        acc = self.total_correct / self.total_predictions * 100 if self.total_predictions > 0 else 0
        return {
            'characters': n_chars,
            'min_depth': min(depths) if depths else 0,
            'max_depth': max(depths) if depths else 0,
            'avg_depth': np.mean(depths) if depths else 0,
            'total_predictions': self.total_predictions,
            'total_correct': self.total_correct,
            'accuracy': acc,
        }
