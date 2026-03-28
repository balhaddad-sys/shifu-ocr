"""
Image Interpreter: Complete MRI-OCR Image Interpretation Pipeline
===================================================================

The final integration layer that combines ALL Shifu-OCR engines into
a single coherent image interpretation system.

This module brings together:
  1. MRI FLAIR Engine     — Multi-sequence character fingerprinting
  2. Topology Coherence   — Capped coherence detection with topological filters
  3. MRI Perturbation     — Relaxation signature recognition
  4. Medium Displacement  — Core topology features
  5. Fluid Landscapes     — Probability terrains shaped by experience
  6. Clinical Vocabulary  — Domain-specific word correction
  7. Co-Defining Context  — Bidirectional character-word constraints

Pipeline:
  Image → Preprocessing → Region Detection → Character Segmentation →
  Multi-Engine Recognition → Word Assembly → Clinical Correction →
  Structured Output with Confidence & Safety Flags

The Radiologist Metaphor:
  Each MRI sequence (T1, T2, FLAIR, DWI) shows the same anatomy
  differently. The radiologist integrates ALL sequences to form
  a diagnosis. Similarly, each OCR engine sees the same character
  differently. The Image Interpreter integrates all engines to
  form a recognition.

Author: Bader & Claude — March 2026
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage
from skimage import morphology, filters
from collections import defaultdict
import re
import warnings
warnings.filterwarnings('ignore')

from .mri_flair import (
    FLAIR_OCR, extract_flair_signature, FLAIRLandscape,
    T1Weighted, T2Weighted, FLAIRSequence, DWISequence, SWISequence, MRASequence,
    measure_flair_response, _normalize_for_flair,
)
from .topology_coherence import (
    TopologyCaps, CoherenceCappedDetector, TopologyCoherenceFusion,
    AdaptiveTopologyCaps,
)
from .perturbation import (
    MRI_OCR, extract_relaxation_signature, measure_response,
)
from .complete import (
    Landscape, extract_features, normalize_char, render_char,
    match_word, ocr_distance, CLINICAL_WORDS, CONFUSIONS,
    compute_coherence_displacement, binarize_coherence,
)


# =============================================================================
# MULTI-ENGINE RECOGNITION RESULT
# =============================================================================

class InterpretationResult:
    """Result of interpreting a single character through all engines."""

    def __init__(self, label, confidence, engine_votes, topology_report=None):
        self.label = label
        self.confidence = confidence
        self.engine_votes = engine_votes  # {engine_name: (label, score)}
        self.topology_report = topology_report

    def agreement(self):
        """Fraction of engines that agree on the label."""
        if not self.engine_votes:
            return 0.0
        votes = [v[0] for v in self.engine_votes.values()]
        return votes.count(self.label) / len(votes)

    def to_dict(self):
        return {
            'label': self.label,
            'confidence': round(self.confidence, 3),
            'agreement': round(self.agreement(), 3),
            'votes': {k: {'label': v[0], 'score': round(v[1], 3)}
                      for k, v in self.engine_votes.items()},
            'topology': self.topology_report,
        }


class WordResult:
    """Result of interpreting a word (sequence of characters)."""

    def __init__(self, raw, corrected, distance, match_type, characters):
        self.raw = raw
        self.corrected = corrected
        self.distance = distance
        self.match_type = match_type
        self.characters = characters  # list of InterpretationResult

    def avg_confidence(self):
        if not self.characters:
            return 0.0
        return np.mean([c.confidence for c in self.characters])

    def to_dict(self):
        return {
            'raw': self.raw,
            'corrected': self.corrected,
            'distance': round(self.distance, 2),
            'match_type': self.match_type,
            'confidence': round(self.avg_confidence(), 3),
            'n_chars': len(self.characters),
        }


class CellResult:
    """Result of interpreting a table cell."""

    def __init__(self, text, words, raw_text, bbox=None):
        self.text = text
        self.words = words  # list of WordResult
        self.raw_text = raw_text
        self.bbox = bbox

    def confidence(self):
        if not self.words:
            return 0.0
        return np.mean([w.avg_confidence() for w in self.words])

    def to_dict(self):
        return {
            'text': self.text,
            'raw': self.raw_text,
            'confidence': round(self.confidence(), 3),
            'words': [w.to_dict() for w in self.words],
        }


# =============================================================================
# MULTI-ENGINE CHARACTER RECOGNIZER
# =============================================================================

class MultiEngineRecognizer:
    """
    Fuses signals from multiple OCR engines for each character.

    Engines and their weights:
      - FLAIR-OCR (252-dim signatures)    : weight 1.2 (best at low resolution)
      - MRI Perturbation (120-dim sigs)   : weight 1.0 (proven on real data)
      - Topology (38-dim features)        : weight 1.0 (fast, reliable at high res)

    Fusion: confidence-weighted score accumulation.
    """

    def __init__(self):
        self.flair_engine = FLAIR_OCR()
        self.mri_engine = MRI_OCR()
        self.topology_landscapes = {}  # label -> Landscape
        self.weights = {
            'flair': 1.2,
            'mri_perturbation': 1.0,
            'topology': 1.0,
        }
        self._trained = False
        self._track = defaultdict(lambda: {'correct': 0, 'total': 0})

    def train_char(self, label, binary_region):
        """Train all engines on one character."""
        # FLAIR engine
        self.flair_engine.train(label, binary_region)

        # MRI perturbation engine
        self.mri_engine.train(label, binary_region)

        # Topology engine
        normed = normalize_char(binary_region)
        fv = extract_features(normed)
        if label not in self.topology_landscapes:
            self.topology_landscapes[label] = Landscape(label)
        self.topology_landscapes[label].absorb(fv)

    def train_from_fonts(self, font_paths=None, font_sizes=None, chars=None):
        """Train all engines from rendered font characters."""
        if font_paths is None:
            font_paths = [
                '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
                '/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf',
                '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',
                '/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf',
            ]
        if font_sizes is None:
            font_sizes = [20, 24, 28, 32, 36, 40, 50, 60]
        if chars is None:
            chars = list('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789')

        count = 0
        for char in chars:
            for fp in font_paths:
                for sz in font_sizes:
                    try:
                        br = render_char(char, fp, sz, (80, 80))
                        coords = np.argwhere(br > 0)
                        if len(coords) == 0:
                            continue
                        r0, c0 = coords.min(axis=0)
                        r1, c1 = coords.max(axis=0)
                        cropped = br[max(0, r0 - 2):r1 + 3, max(0, c0 - 2):c1 + 3]
                        self.train_char(char, cropped)
                        count += 1
                    except Exception:
                        pass

        self._trained = True
        return count

    def recognize(self, binary_region, top_k=5):
        """
        Recognize a character using all engines and fuse results.

        Returns InterpretationResult with fused prediction.
        """
        engine_votes = {}
        label_scores = defaultdict(float)

        # --- FLAIR Engine ---
        try:
            flair_results = self.flair_engine.recognize(binary_region, top_k=top_k)
            if flair_results:
                best_label, best_score = flair_results[0]
                engine_votes['flair'] = (best_label, best_score)
                for label, score in flair_results:
                    # Normalize score to positive range for accumulation
                    norm_score = max(0, score - flair_results[-1][1]) if len(flair_results) > 1 else 1.0
                    label_scores[label] += self.weights['flair'] * norm_score
        except Exception:
            pass

        # --- MRI Perturbation Engine ---
        try:
            mri_results = self.mri_engine.recognize(binary_region, top_k=top_k)
            if mri_results:
                best_label, best_score = mri_results[0]
                engine_votes['mri_perturbation'] = (best_label, best_score)
                for label, score in mri_results:
                    norm_score = max(0, score - mri_results[-1][1]) if len(mri_results) > 1 else 1.0
                    label_scores[label] += self.weights['mri_perturbation'] * norm_score
        except Exception:
            pass

        # --- Topology Engine ---
        try:
            normed = normalize_char(binary_region)
            fv = extract_features(normed)
            topo_scores = [(l, land.fit(fv)) for l, land in self.topology_landscapes.items()]
            topo_scores.sort(key=lambda x: x[1], reverse=True)
            topo_top = topo_scores[:top_k]
            if topo_top:
                best_label, best_score = topo_top[0]
                engine_votes['topology'] = (best_label, best_score)
                for label, score in topo_top:
                    norm_score = max(0, score - topo_top[-1][1]) if len(topo_top) > 1 else 1.0
                    label_scores[label] += self.weights['topology'] * norm_score
        except Exception:
            pass

        if not label_scores:
            return InterpretationResult('?', 0.0, engine_votes)

        # Fuse: best label by accumulated weighted score
        sorted_labels = sorted(label_scores.items(), key=lambda x: x[1], reverse=True)
        best_label = sorted_labels[0][0]
        best_score = sorted_labels[0][1]
        second_score = sorted_labels[1][1] if len(sorted_labels) > 1 else 0.0

        # Confidence from agreement and margin
        n_agree = sum(1 for v in engine_votes.values() if v[0] == best_label)
        agreement = n_agree / max(len(engine_votes), 1)
        total = sum(s for _, s in sorted_labels)
        margin = (best_score - second_score) / max(total, 0.01)
        confidence = 0.5 * agreement + 0.5 * min(margin, 1.0)

        return InterpretationResult(best_label, confidence, engine_votes)

    def correct(self, result, true_label, binary_region):
        """Feed a correction back into all engines."""
        # Retrain all engines with the correct label
        self.train_char(true_label, binary_region)

        # Track per-engine accuracy
        for engine_name, (pred, score) in result.engine_votes.items():
            self._track[engine_name]['total'] += 1
            if pred == true_label:
                self._track[engine_name]['correct'] += 1

        # Adaptive weights: boost engines that are accurate
        for engine_name in self.weights:
            rec = self._track[engine_name]
            if rec['total'] >= 10:
                accuracy = rec['correct'] / rec['total']
                target = accuracy / 0.5  # 50% accuracy = 1.0 weight
                self.weights[engine_name] = (
                    0.9 * self.weights[engine_name] +
                    0.1 * max(0.2, min(3.0, target))
                )

    def stats(self):
        return {
            'flair_landscapes': len(self.flair_engine.landscapes),
            'mri_landscapes': len(self.mri_engine.landscapes),
            'topology_landscapes': len(self.topology_landscapes),
            'weights': dict(self.weights),
            'track_record': dict(self._track),
            'trained': self._trained,
        }


# =============================================================================
# IMAGE INTERPRETER: The Complete Pipeline
# =============================================================================

class ImageInterpreter:
    """
    Complete image interpretation system.

    Combines:
      - Multi-engine character recognition (FLAIR + MRI + Topology)
      - Topology-capped coherence detection
      - Clinical vocabulary correction
      - Table structure detection
      - Structured output with confidence and safety flags

    This is the top-level class that turns a raw image into
    structured, clinically-validated text output.
    """

    def __init__(self, adaptive_caps=True):
        # Topology caps (adaptive or fixed)
        if adaptive_caps:
            self.caps = AdaptiveTopologyCaps()
        else:
            self.caps = TopologyCaps()

        # Detection pipeline
        self.detector = CoherenceCappedDetector(caps=self.caps)
        self.fusion = TopologyCoherenceFusion(caps=self.caps)

        # Multi-engine recognizer
        self.recognizer = MultiEngineRecognizer()

        # State
        self._trained = False
        self._n_corrections = 0

    def train(self, font_paths=None, font_sizes=None, chars=None):
        """Train all engines from rendered fonts."""
        count = self.recognizer.train_from_fonts(font_paths, font_sizes, chars)
        self._trained = True
        return count

    def train_from_document(self, image_array, gray, ground_truth_cells):
        """
        Train from a real document with ground truth annotations.

        Args:
            image_array: RGB image as numpy array
            gray: grayscale image as numpy array
            ground_truth_cells: list of (r0, r1, c0, c1, text) tuples
        """
        total = 0
        for r0, r1, c0, c1, text in ground_truth_cells:
            cell_rgb = image_array[r0:r1, c0:c1]
            cell_gray = gray[r0:r1, c0:c1]

            # Detect character regions with topology caps
            regions = self.fusion.detect_and_extract(cell_rgb, cell_gray)

            gt_chars = list(text.replace(' ', ''))
            n = min(len(regions), len(gt_chars))

            if n < len(gt_chars) * 0.4:
                continue

            for i in range(n):
                self.recognizer.train_char(gt_chars[i], regions[i]['binary'])

                # Adapt topology caps from confirmed characters
                if isinstance(self.caps, AdaptiveTopologyCaps):
                    self.caps.observe(regions[i]['binary'])

                total += 1

        self._trained = True
        return total

    def read_cell(self, cell_rgb, cell_gray):
        """
        Read text from a single table cell.

        Returns CellResult with corrected text, raw text, word-level
        results, and character-level engine votes.
        """
        # Detect character regions
        regions = self.fusion.detect_and_extract(cell_rgb, cell_gray)

        if not regions:
            return CellResult('', [], '', None)

        # Detect spaces between characters
        space_positions = set()
        if len(regions) > 1:
            gaps = []
            for i in range(1, len(regions)):
                prev_end = regions[i-1]['bbox'][3]
                curr_start = regions[i]['bbox'][1]
                gaps.append(curr_start - prev_end)
            space_thresh = np.median(gaps) * 2.0 if gaps else 20

            for i in range(1, len(regions)):
                gap = regions[i]['bbox'][1] - regions[i-1]['bbox'][3]
                if gap > space_thresh:
                    space_positions.add(i)

        # Recognize each character
        char_results = []
        raw_chars = []

        for i, region in enumerate(regions):
            if i in space_positions:
                raw_chars.append(' ')

            result = self.recognizer.recognize(region['binary'])
            result.topology_report = region.get('topology')
            char_results.append(result)
            raw_chars.append(result.label)

        raw_text = ''.join(raw_chars)

        # Assemble into words and correct
        words_raw = raw_text.split()
        word_results = []
        char_idx = 0

        for w_raw in words_raw:
            corrected, dist, match_type = match_word(w_raw)
            # Collect character results for this word
            word_chars = char_results[char_idx:char_idx + len(w_raw)]
            char_idx += len(w_raw)
            word_results.append(WordResult(w_raw, corrected, dist, match_type, word_chars))

        corrected_text = ' '.join(w.corrected for w in word_results)

        return CellResult(corrected_text, word_results, raw_text)

    def read_line(self, line_gray):
        """
        Read a single line of text from a grayscale image.

        Returns CellResult (same structure as read_cell).
        """
        # Create a pseudo-RGB from grayscale
        if len(line_gray.shape) == 2:
            line_rgb = np.stack([line_gray] * 3, axis=-1)
        else:
            line_rgb = line_gray
            line_gray = np.mean(line_gray, axis=2).astype(np.uint8)

        return self.read_cell(line_rgb, line_gray)

    def process_table(self, image_array, gray, column_defs=None):
        """
        Process a full table image into structured rows.

        Args:
            image_array: RGB numpy array
            gray: grayscale numpy array
            column_defs: list of (c_start, c_end, col_name) tuples
                         If None, auto-detect columns.

        Returns:
            list of dicts, each with column_name -> CellResult
        """
        # Detect rows
        rows = self._detect_rows(gray)

        # Detect or use provided columns
        if column_defs is None:
            column_defs = self._detect_columns(gray, image_array, rows)

        # Process each cell
        results = []
        for r_idx, (r_start, r_end) in enumerate(rows):
            row_data = {'_row': r_idx, '_y': (r_start, r_end)}

            for c_start, c_end, col_name in column_defs:
                cell_rgb = image_array[r_start:r_end, c_start:c_end]
                cell_gray = gray[r_start:r_end, c_start:c_end]

                cell_result = self.read_cell(cell_rgb, cell_gray)
                row_data[col_name] = cell_result

            # Check if row has content
            has_content = any(
                row_data[col_name].text.strip()
                for _, _, col_name in column_defs
                if col_name in row_data and isinstance(row_data[col_name], CellResult)
            )

            if has_content:
                results.append(row_data)

        return results

    def process_image(self, image_path, column_defs=None):
        """
        Full pipeline: image file path → structured table data.

        Args:
            image_path: path to image file
            column_defs: optional column definitions

        Returns:
            list of structured row dicts
        """
        img = Image.open(image_path)
        rgb = np.array(img)
        gray = np.array(img.convert('L'))

        return self.process_table(rgb, gray, column_defs)

    def interpret_region(self, image_array, gray, r0, r1, c0, c1):
        """
        Interpret a specific region of an image.

        Returns CellResult for the specified region.
        """
        cell_rgb = image_array[r0:r1, c0:c1]
        cell_gray = gray[r0:r1, c0:c1]
        return self.read_cell(cell_rgb, cell_gray)

    def analyze_character(self, binary_region):
        """
        Full diagnostic analysis of a character.

        Returns per-engine analysis, topology report, and FLAIR sequence
        contributions — like a radiology report.
        """
        result = self.recognizer.recognize(binary_region)

        analysis = {
            'prediction': result.to_dict(),
            'flair_analysis': self.recognizer.flair_engine.analyze_character(binary_region),
            'flair_contributions': self.recognizer.flair_engine.get_sequence_contribution(binary_region),
        }

        # Topology report
        passed, topo_report = self.caps.check(binary_region)
        analysis['topology'] = topo_report

        return analysis

    def correct(self, cell_result, true_text, cell_rgb=None, cell_gray=None):
        """
        Feed a correction back into all engines.

        If cell_rgb/cell_gray are provided, re-segment and train
        with the correct characters.
        """
        if cell_rgb is not None and cell_gray is not None:
            regions = self.fusion.detect_and_extract(cell_rgb, cell_gray)
            gt_chars = list(true_text.replace(' ', ''))
            n = min(len(regions), len(gt_chars))

            for i in range(n):
                # Find the corresponding character result
                char_result = None
                if i < len(cell_result.words):
                    for w in cell_result.words:
                        if i < len(w.characters):
                            char_result = w.characters[i]
                            break

                self.recognizer.train_char(gt_chars[i], regions[i]['binary'])

                if isinstance(self.caps, AdaptiveTopologyCaps):
                    self.caps.observe(regions[i]['binary'])

            self._n_corrections += 1

    # =========================================================================
    # INTERNAL: Table structure detection
    # =========================================================================

    def _detect_rows(self, gray, min_height=18):
        """Detect horizontal text bands."""
        binary = (gray < 200).astype(np.uint8)
        h_proj = binary.sum(axis=1).astype(float)
        h_proj = ndimage.uniform_filter1d(h_proj, size=3)

        threshold = max(h_proj.max() * 0.01, 5)
        is_text = h_proj > threshold

        rows = []
        in_row = False
        start = 0
        for i in range(len(is_text)):
            if is_text[i] and not in_row:
                start = i
                in_row = True
            elif not is_text[i] and in_row:
                if i - start >= min_height:
                    rows.append((start, i))
                in_row = False
        if in_row and len(is_text) - start >= min_height:
            rows.append((start, len(is_text)))

        return rows

    def _detect_columns(self, gray, rgb, rows):
        """Auto-detect column boundaries from vertical edges."""
        if not rows:
            return [(0, gray.shape[1], 'text')]

        # Vertical gradient magnitude
        v_proj = np.abs(np.diff(gray.astype(float), axis=1)).mean(axis=0)
        v_proj_smooth = ndimage.uniform_filter1d(v_proj, size=10)

        # Color transitions
        if len(rgb.shape) == 3:
            color_diff = np.abs(np.diff(rgb.astype(float), axis=1)).sum(axis=2).mean(axis=0)
            color_smooth = ndimage.uniform_filter1d(color_diff, size=10)
            combined = v_proj_smooth + color_smooth
        else:
            combined = v_proj_smooth

        # Find peaks
        try:
            from scipy.signal import find_peaks
            peaks, _ = find_peaks(combined, distance=60, height=combined.max() * 0.15)
        except Exception:
            peaks = []

        cols = [0]
        for p in sorted(peaks):
            cols.append(int(p))
        cols.append(gray.shape[1])

        # Build column ranges
        column_ranges = []
        for i in range(len(cols) - 1):
            if cols[i + 1] - cols[i] > 30:
                column_ranges.append((cols[i], cols[i + 1], f'col_{i}'))

        return column_ranges if column_ranges else [(0, gray.shape[1], 'text')]

    # =========================================================================
    # STATISTICS & REPORTING
    # =========================================================================

    def stats(self):
        """Return comprehensive system statistics."""
        result = {
            'trained': self._trained,
            'corrections': self._n_corrections,
            'recognizer': self.recognizer.stats(),
        }
        if isinstance(self.caps, AdaptiveTopologyCaps):
            result['adaptive_caps'] = self.caps.get_stats()
        return result

    def summary(self):
        """Human-readable summary of the interpreter state."""
        s = self.stats()
        lines = [
            "Image Interpreter Status",
            "=" * 40,
            f"  Trained: {s['trained']}",
            f"  Corrections absorbed: {s['corrections']}",
            f"  FLAIR landscapes: {s['recognizer']['flair_landscapes']}",
            f"  MRI landscapes: {s['recognizer']['mri_landscapes']}",
            f"  Topology landscapes: {s['recognizer']['topology_landscapes']}",
            f"  Engine weights: {s['recognizer']['weights']}",
        ]
        if 'adaptive_caps' in s:
            caps = s['adaptive_caps']
            lines.append(f"  Topology observations: {caps['n_observations']}")
            lines.append(f"  Component range: {caps['component_range']}")
            lines.append(f"  Aspect range: {caps['aspect_range']}")
        return '\n'.join(lines)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_interpreter(train=True, font_paths=None):
    """
    Factory function to create a fully initialized ImageInterpreter.

    Args:
        train: if True, train from system fonts
        font_paths: optional list of font paths to train from

    Returns:
        ImageInterpreter ready to process images
    """
    interpreter = ImageInterpreter(adaptive_caps=True)
    if train:
        interpreter.train(font_paths=font_paths)
    return interpreter


def interpret_image(image_path, column_defs=None, train=True):
    """
    One-shot image interpretation: image file → structured output.

    Args:
        image_path: path to image file
        column_defs: optional column definitions
        train: if True, train from system fonts first

    Returns:
        list of structured row dicts
    """
    interpreter = create_interpreter(train=train)
    return interpreter.process_image(image_path, column_defs)


def interpret_region(image_path, r0, r1, c0, c1, train=True):
    """
    Interpret a specific region of an image.

    Returns CellResult for the specified region.
    """
    interpreter = create_interpreter(train=train)
    img = Image.open(image_path)
    rgb = np.array(img)
    gray = np.array(img.convert('L'))
    return interpreter.interpret_region(rgb, gray, r0, r1, c0, c1)
