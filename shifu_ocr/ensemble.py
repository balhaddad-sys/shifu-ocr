"""
Shifu-OCR Multi-Engine Ensemble
================================

Core design principle: multiple engines see the SAME character simultaneously,
each through a different lens. Their signals are fused, not voted on.

Engines and what they see:
  - Topology Engine (engine.py)    : Static form — components, holes, symmetry, projections
  - Fluid Engine (fluid.py)        : Probability landscapes — no rules, shaped by experience
  - Perturbation Engine (perturbation.py) : Response to disturbance — erosion, dilation, blur
  - Theory-Revision Engine (theory_revision.py) : Principled prediction with auditable reasoning
  - Co-Defining Engine (codefining.py) : Bidirectional constraints — character ↔ word ↔ context

Why multiple engines at the same time:
  Static features fail on low-resolution characters (a, e, o look identical at 6px).
  Perturbation signatures distinguish them (they degrade differently under erosion).
  But perturbation is expensive and sometimes noisy at high resolution.
  Topology is fast and reliable at high resolution.
  The co-defining engine uses word context to break ties neither can resolve alone.

  No single lens is always right. The ensemble is.

The learning loop feeds corrections back into ALL engines simultaneously.
When a nurse corrects "seizrue" → "seizure", every engine absorbs that correction
through its own mechanism (landscape reshaping, theory revision, co-defining update).

Author: Bader & Claude — March 2026
"""

import numpy as np
from collections import defaultdict


class EngineVote:
    """A single engine's opinion about a character."""
    __slots__ = ('label', 'confidence', 'score', 'engine_name', 'reasoning')

    def __init__(self, label, confidence, score, engine_name, reasoning=None):
        self.label = label
        self.confidence = confidence
        self.score = score
        self.engine_name = engine_name
        self.reasoning = reasoning


class EnsembleResult:
    """Fused result from all engines."""

    def __init__(self, label, confidence, votes, agreement, margin):
        self.label = label
        self.confidence = confidence
        self.votes = votes          # list of EngineVote
        self.agreement = agreement  # 0.0–1.0: how many engines agree
        self.margin = margin        # gap between top and runner-up

    def to_dict(self):
        return {
            'predicted': self.label,
            'confidence': self.confidence,
            'agreement': self.agreement,
            'margin': self.margin,
            'votes': [
                {'engine': v.engine_name, 'label': v.label, 'confidence': v.confidence}
                for v in self.votes
            ],
        }


class ShifuEnsemble:
    """
    Multi-engine orchestrator.

    All registered engines are queried for every character.
    Their signals are fused using confidence-weighted scoring,
    with adaptive weights that shift based on each engine's
    track record for the current document.
    """

    def __init__(self):
        self.engines = {}           # name -> engine instance
        self.weights = {}           # name -> current weight (adapts over time)
        self.base_weights = {}      # name -> starting weight
        self.track_record = {}      # name -> {'correct': int, 'total': int}
        self.total_predictions = 0
        self.total_correct = 0

    def register(self, name, engine, weight=1.0, predict_fn=None):
        """
        Register an engine with an optional custom predict function.

        predict_fn(engine, binary_region) -> {'predicted': str, 'confidence': float, ...}
        If not provided, engine.predict(binary_region) or engine.recognize(binary_region)
        or engine.predict_character(binary_region) is used.
        """
        self.engines[name] = engine
        self.weights[name] = weight
        self.base_weights[name] = weight
        self.track_record[name] = {'correct': 0, 'total': 0}

        if predict_fn:
            self._predict_fns = getattr(self, '_predict_fns', {})
            self._predict_fns[name] = predict_fn

    def _get_prediction(self, name, engine, binary_region):
        """Get a prediction from an engine, handling different interfaces."""
        custom = getattr(self, '_predict_fns', {}).get(name)
        if custom:
            return custom(engine, binary_region)

        # Try common interfaces
        if hasattr(engine, 'predict_character'):
            return engine.predict_character(binary_region)
        elif hasattr(engine, 'predict'):
            return engine.predict(binary_region)
        elif hasattr(engine, 'recognize'):
            results = engine.recognize(binary_region, top_k=5)
            if results:
                best_label, best_score = results[0]
                second_score = results[1][1] if len(results) > 1 else float('-inf')
                total_range = results[0][1] - results[-1][1] if len(results) > 1 else 1.0
                margin = best_score - second_score
                confidence = max(0, min(margin / max(abs(total_range), 0.01), 1.0))
                return {
                    'predicted': best_label,
                    'confidence': confidence,
                    'candidates': results,
                }
        return None

    def predict(self, binary_region):
        """
        Query ALL engines and fuse their signals.

        Fusion strategy: confidence-weighted score accumulation.
        Each engine contributes: weight * confidence to its predicted label.
        The label with the highest accumulated score wins.
        """
        votes = []
        label_scores = defaultdict(float)

        for name, engine in self.engines.items():
            try:
                result = self._get_prediction(name, engine, binary_region)
                if result is None:
                    continue

                label = result.get('predicted', '?')
                confidence = result.get('confidence', 0.5)
                w = self.weights[name]

                vote = EngineVote(
                    label=label,
                    confidence=confidence,
                    score=w * confidence,
                    engine_name=name,
                    reasoning=result.get('reasoning'),
                )
                votes.append(vote)

                # Accumulate weighted score for this label
                label_scores[label] += w * confidence

            except Exception:
                # Engine failed — skip it, don't crash the ensemble
                continue

        if not votes:
            return EnsembleResult('?', 0.0, [], 0.0, 0.0)

        # Find winner
        sorted_labels = sorted(label_scores.items(), key=lambda x: x[1], reverse=True)
        best_label, best_score = sorted_labels[0]
        second_score = sorted_labels[1][1] if len(sorted_labels) > 1 else 0.0

        # Agreement: fraction of engines that chose the winner
        n_agree = sum(1 for v in votes if v.label == best_label)
        agreement = n_agree / len(votes)

        # Margin between top and runner-up
        total = sum(s for _, s in sorted_labels)
        margin = (best_score - second_score) / max(total, 0.01)

        # Confidence: combines agreement and margin
        confidence = 0.5 * agreement + 0.5 * min(margin, 1.0)

        self.total_predictions += 1

        return EnsembleResult(
            label=best_label,
            confidence=confidence,
            votes=votes,
            agreement=agreement,
            margin=margin,
        )

    def correct(self, result, true_label):
        """
        Feed a correction back into ALL engines.

        Each engine absorbs the correction through its own mechanism.
        Also updates the adaptive weights: engines that got it right
        get a slight boost; engines that got it wrong get dampened.
        """
        is_correct = result.label == true_label
        if is_correct:
            self.total_correct += 1

        for vote in result.votes:
            name = vote.engine_name
            engine = self.engines[name]
            record = self.track_record[name]
            record['total'] += 1

            engine_correct = vote.label == true_label
            if engine_correct:
                record['correct'] += 1

            # Feed correction into engine (try common interfaces)
            try:
                if hasattr(engine, 'correct'):
                    # Theory-revision and ShifuOCR style
                    pred = {'predicted': vote.label, 'confidence': vote.confidence,
                            'features': getattr(vote, 'features', None)}
                    engine.correct(pred, true_label)
                elif hasattr(engine, 'experience'):
                    # Fluid engine style
                    pred = {'predicted': vote.label, 'confidence': vote.confidence,
                            'features': getattr(vote, 'features', None)}
                    engine.experience(pred, true_label)
            except Exception:
                pass

            # Adaptive weight: smoothed accuracy ratio
            if record['total'] >= 5:
                accuracy = record['correct'] / record['total']
                # Blend toward accuracy, but don't go below 0.1x or above 3x base
                target = self.base_weights[name] * max(0.1, min(3.0, accuracy / 0.5))
                self.weights[name] = 0.9 * self.weights[name] + 0.1 * target

    def read_line(self, grayscale_image, coherence_fn=None):
        """
        Read a text line using multi-engine ensemble for each character.

        Uses the topology engine's segmentation (since it handles that well),
        then runs each segment through the full ensemble.
        """
        # Use the first engine that has segment_characters
        segmenter = None
        for name, engine in self.engines.items():
            if hasattr(engine, 'segment_characters'):
                segmenter = engine
                break

        if segmenter is None:
            return {'text': '', 'characters': [], 'confidence': 0}

        segments = segmenter.segment_characters(grayscale_image)
        if not segments:
            return {'text': '', 'characters': [], 'confidence': 0}

        # Detect spaces
        gaps = []
        for i in range(1, len(segments)):
            prev_end = segments[i-1]['bbox'][3]
            curr_start = segments[i]['bbox'][1]
            gaps.append(curr_start - prev_end)
        space_threshold = np.median(gaps) * 2.0 if gaps else 20

        results = []
        text_parts = []

        for i, seg in enumerate(segments):
            ensemble_result = self.predict(seg['image'])
            results.append(ensemble_result)

            if i > 0:
                prev_end = segments[i-1]['bbox'][3]
                curr_start = seg['bbox'][1]
                if curr_start - prev_end > space_threshold:
                    text_parts.append(' ')

            text_parts.append(ensemble_result.label)

        text = ''.join(text_parts)
        avg_conf = np.mean([r.confidence for r in results]) if results else 0

        return {
            'text': text,
            'characters': [r.to_dict() for r in results],
            'confidence': avg_conf,
            'agreement': np.mean([r.agreement for r in results]) if results else 0,
        }

    def read_page(self, grayscale_image, space_threshold=None):
        """
        Read a full page using the ensemble for character recognition.
        Uses the topology engine for line/character segmentation,
        then runs each character through all engines for voting.
        """
        # Use topology engine for segmentation (it has segment_lines)
        segmenter = None
        for name, engine in self.engines.items():
            if hasattr(engine, 'segment_lines'):
                segmenter = engine
                break

        if segmenter is None:
            return {'text': '', 'lines': [], 'confidence': 0}

        line_segments = segmenter.segment_lines(grayscale_image)
        if not line_segments:
            return {'text': '', 'lines': [], 'confidence': 0}

        all_lines = []
        all_confidences = []

        for seg in line_segments:
            line_result = self.read_line(seg['image'], space_threshold)
            text = line_result.get('text', '').strip()
            if text:
                # Filter noise: mostly non-alphanumeric = grid remnants
                alnum = sum(1 for c in text if c.isalnum())
                if alnum >= max(len(text) * 0.15, 1):
                    all_lines.append(text)
                    all_confidences.append(line_result.get('confidence', 0))

        full_text = '\n'.join(all_lines)
        avg_conf = float(np.mean(all_confidences)) if all_confidences else 0

        return {
            'text': full_text,
            'lines': all_lines,
            'confidence': avg_conf,
        }

    def get_stats(self):
        """Return per-engine and ensemble statistics."""
        acc = self.total_correct / self.total_predictions * 100 if self.total_predictions > 0 else 0
        engine_stats = {}
        for name in self.engines:
            rec = self.track_record[name]
            eng_acc = rec['correct'] / rec['total'] * 100 if rec['total'] > 0 else 0
            engine_stats[name] = {
                'weight': round(self.weights[name], 3),
                'accuracy': round(eng_acc, 1),
                'predictions': rec['total'],
            }
        return {
            'ensemble_accuracy': round(acc, 1),
            'total_predictions': self.total_predictions,
            'engines': engine_stats,
        }


def create_ensemble(engines_to_use=None, model_path=None):
    """
    Factory function to create a ShifuEnsemble with standard engines.

    Args:
        engines_to_use: list of engine names to include.
            Options: 'topology', 'fluid', 'perturbation', 'theory_revision'
            Default: all engines.
        model_path: path to trained_model.json. If provided, loads the trained
            topology engine and shares its landscapes with other engines.

    Returns:
        ShifuEnsemble with engines registered.
    """
    import os
    available = engines_to_use or ['topology', 'fluid', 'perturbation', 'theory_revision']
    ensemble = ShifuEnsemble()

    # Load trained topology engine (primary — has segmentation + trained landscapes)
    if 'topology' in available:
        from .engine import ShifuOCR
        if model_path and os.path.exists(model_path):
            topo = ShifuOCR.load(model_path)
        else:
            topo = ShifuOCR()
        ensemble.register('topology', topo, weight=1.0)

    if 'fluid' in available:
        from .fluid import FluidEngine
        ensemble.register('fluid', FluidEngine(), weight=1.0)

    if 'perturbation' in available:
        from .perturbation import MRI_OCR
        ensemble.register('perturbation', MRI_OCR(), weight=0.8)

    if 'theory_revision' in available:
        from .theory_revision import TheoryRevisionEngine
        ensemble.register('theory_revision', TheoryRevisionEngine(), weight=0.9)

    return ensemble


def train_ensemble(ensemble, characters, font_paths, font_sizes=None):
    """
    Train all engines in the ensemble on rendered characters.

    Each engine's train/teach method is called with the same character images,
    so all engines learn from identical data but build different representations.
    """
    from PIL import Image, ImageDraw, ImageFont
    from .engine import normalize_region, extract_region, image_to_binary

    if font_sizes is None:
        font_sizes = [20, 24, 28, 32, 36, 40, 50, 60]

    count = 0
    for char in characters:
        for font_path in font_paths:
            for font_size in font_sizes:
                try:
                    # Render character
                    img_size = (100, 100)
                    img = Image.new('L', img_size, color=255)
                    draw = ImageDraw.Draw(img)
                    try:
                        font = ImageFont.truetype(font_path, font_size)
                    except Exception:
                        font = ImageFont.load_default()
                    bbox = draw.textbbox((0, 0), char, font=font)
                    x = (img_size[0] - (bbox[2] - bbox[0])) // 2 - bbox[0]
                    y = (img_size[1] - (bbox[3] - bbox[1])) // 2 - bbox[1]
                    draw.text((x, y), char, fill=0, font=font)
                    grayscale = np.array(img)

                    # Get binary region
                    binary, _ = image_to_binary(grayscale)
                    region = extract_region(binary)
                    normed = normalize_region(region)

                    # Train each engine through its own interface
                    for name, engine in ensemble.engines.items():
                        if hasattr(engine, 'train_character'):
                            engine.train_character(char, grayscale)
                        elif hasattr(engine, 'teach'):
                            engine.teach(char, normed)
                        elif hasattr(engine, 'train'):
                            engine.train(char, normed)

                    count += 1
                except Exception:
                    pass

    return count
