"""
MRI FLAIR OCR Engine: Fluid-Attenuated Inversion Recovery for Character Recognition
=====================================================================================

In clinical MRI, FLAIR suppresses cerebrospinal fluid (CSF) signal to reveal
periventricular lesions that T1/T2 sequences miss. The key insight:
REMOVE THE OBVIOUS to reveal the subtle.

In MRI-FLAIR-OCR, we apply the same principle:
  1. INVERSION RECOVERY: Invert the image, then let it "recover" —
     the recovery rate reveals structural properties invisible in static view
  2. FLUID ATTENUATION: Suppress the background (the "fluid" of the image) —
     what remains is pure structure
  3. MULTI-SEQUENCE ACQUISITION: Like clinical MRI protocols (T1, T2, FLAIR, DWI),
     each sequence reveals different character properties

WHY THIS MATTERS FOR LOW-RESOLUTION OCR:
At 6 pixels, 'a', 'e', 'o', 's' are indistinguishable blobs.
But their FLAIR signatures differ:
  - 'a': After fluid attenuation, the gap persists (open topology)
  - 'e': After fluid attenuation, the crossbar creates a unique T2* decay
  - 'o': Uniform recovery (closed, symmetric topology)
  - 's': Non-uniform recovery (open, asymmetric topology)

The FLAIR sequence sees what static imaging cannot.

MRI Sequence Analogy:
  T1-weighted  → Erosion response (what survives thinning = "fat-bright")
  T2-weighted  → Dilation response (what fills in = "water-bright")
  FLAIR        → Background-suppressed structure (pure lesion detection)
  DWI          → Perturbation diffusion (how disturbance spreads)
  ADC map      → Apparent diffusion coefficient (quantitative diffusion)
  SWI          → Susceptibility-weighted (micro-structure sensitivity)
  MRA          → Angiography (connectivity/flow paths through the character)

Author: Bader & Claude — March 2026
"""

import numpy as np
from PIL import Image
from scipy import ndimage
from skimage import morphology, filters
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# MRI SEQUENCE DEFINITIONS
# Each sequence reveals a different property of the character
# =============================================================================

class MRISequence:
    """Base class for an MRI-like imaging sequence."""

    def __init__(self, name, description):
        self.name = name
        self.description = description

    def acquire(self, binary):
        """Apply this sequence to a binary character image.
        Returns a response image or feature vector."""
        raise NotImplementedError


class T1Weighted(MRISequence):
    """T1-weighted: Erosion cascade — what survives progressive thinning.
    Analogous to T1 where fat (dense tissue) appears bright.
    Dense strokes survive; thin strokes vanish first."""

    def __init__(self):
        super().__init__('T1w', 'Erosion cascade (dense structure persistence)')

    def acquire(self, binary):
        cascade = [binary.copy()]
        current = binary.copy()
        for radius in [1, 2, 3]:
            eroded = morphology.erosion(current, morphology.disk(radius))
            cascade.append(eroded.astype(np.uint8))
            current = eroded.astype(np.uint8)
        return cascade


class T2Weighted(MRISequence):
    """T2-weighted: Dilation cascade — how the character spreads.
    Analogous to T2 where water (fluid regions) appears bright.
    Gaps close, components merge, revealing connectivity."""

    def __init__(self):
        super().__init__('T2w', 'Dilation cascade (fluid spread)')

    def acquire(self, binary):
        cascade = [binary.copy()]
        current = binary.copy()
        for radius in [1, 2, 3]:
            dilated = morphology.dilation(current, morphology.disk(radius))
            cascade.append(dilated.astype(np.uint8))
            current = dilated.astype(np.uint8)
        return cascade


class FLAIRSequence(MRISequence):
    """FLAIR: Fluid-Attenuated Inversion Recovery.

    The defining sequence of this engine.

    Process:
    1. INVERSION: Flip the image (ink becomes background, background becomes ink)
    2. RECOVERY: Apply progressive morphological operations to "recover"
    3. ATTENUATION: Subtract the background model to suppress "fluid" (smooth regions)
    4. What remains = pure structural signal

    This reveals enclosed spaces, structural bridges, and topological features
    that are invisible in direct viewing."""

    def __init__(self):
        super().__init__('FLAIR', 'Fluid-attenuated inversion recovery')

    def acquire(self, binary):
        h, w = binary.shape
        if h < 3 or w < 3:
            return [binary, binary, binary, binary]

        # Step 1: INVERSION — flip ink and background
        inverted = 1 - binary

        # Step 2: Crop to character region for meaningful analysis
        coords = np.argwhere(binary > 0)
        if len(coords) < 2:
            return [binary, inverted, binary, binary]
        r0, c0 = np.maximum(coords.min(axis=0) - 1, 0)
        r1, c1 = np.minimum(coords.max(axis=0) + 2, [h, w])
        roi_inv = inverted[r0:r1, c0:c1]
        roi_orig = binary[r0:r1, c0:c1]

        # Step 3: RECOVERY — progressive opening reveals structure
        # Small opening: removes noise, preserves large enclosed spaces
        try:
            recovered_light = morphology.opening(roi_inv, morphology.disk(1)).astype(np.uint8)
        except Exception:
            recovered_light = roi_inv

        # Step 4: ATTENUATION — suppress the "fluid" (background regions)
        # The background of the inverted image = the character's ink
        # What remains after attenuation = enclosed background (holes, gaps)
        # This is the FLAIR signal: pure structural topology

        # Background model of the inverted ROI
        if roi_inv.shape[0] > 5 and roi_inv.shape[1] > 5:
            bg_model = ndimage.uniform_filter(roi_inv.astype(float), size=3)
            flair_signal = np.abs(roi_inv.astype(float) - bg_model)
            flair_binary = (flair_signal > 0.15).astype(np.uint8)
        else:
            flair_binary = roi_inv

        return [roi_orig, roi_inv, recovered_light, flair_binary]


class DWISequence(MRISequence):
    """Diffusion-Weighted Imaging: How perturbation spreads through the character.

    Applies a small local perturbation and measures how it diffuses.
    Restricted diffusion (like in acute stroke) = tight structure.
    Free diffusion = open structure."""

    def __init__(self):
        super().__init__('DWI', 'Diffusion-weighted (perturbation spread)')

    def acquire(self, binary):
        h, w = binary.shape
        if h < 3 or w < 3 or binary.sum() < 2:
            return [binary, np.zeros_like(binary)]

        # Distance transform = how far each background pixel is from ink
        # This IS the diffusion map: ink restricts diffusion
        dist_from_ink = ndimage.distance_transform_edt(1 - binary)

        # ADC map: apparent diffusion coefficient
        # Within the character, measure how "trapped" each pixel is
        dist_within = ndimage.distance_transform_edt(binary)

        # Normalize both
        if dist_from_ink.max() > 0:
            dist_from_ink = dist_from_ink / dist_from_ink.max()
        if dist_within.max() > 0:
            dist_within = dist_within / dist_within.max()

        # DWI signal: high where diffusion is restricted (inside tight structures)
        dwi = (dist_within > 0.2).astype(np.uint8)

        # ADC: continuous diffusion coefficient
        adc = dist_within

        return [dwi, adc]


class SWISequence(MRISequence):
    """Susceptibility-Weighted Imaging: Micro-structure sensitivity.

    Uses gradient-based edge detection at multiple scales to reveal
    fine structural details. Analogous to SWI's sensitivity to
    paramagnetic substances (iron, blood products)."""

    def __init__(self):
        super().__init__('SWI', 'Susceptibility-weighted (micro-structure)')

    def acquire(self, binary):
        h, w = binary.shape
        if h < 3 or w < 3:
            return [binary]

        img_f = binary.astype(float)

        # Multi-scale gradient (like phase/magnitude in SWI)
        gx = ndimage.sobel(img_f, axis=1)
        gy = ndimage.sobel(img_f, axis=0)
        magnitude = np.sqrt(gx**2 + gy**2)

        # Phase angle (direction of edge)
        phase = np.arctan2(gy, gx + 1e-10)

        # SWI = magnitude weighted by phase consistency
        # Consistent phase = strong structural edge
        # Inconsistent = noise
        phase_smooth = ndimage.uniform_filter(phase, size=3)
        phase_consistency = np.cos(phase - phase_smooth)
        swi = magnitude * np.maximum(phase_consistency, 0)

        if swi.max() > 0:
            swi = swi / swi.max()

        return [(swi > 0.2).astype(np.uint8), swi]


class MRASequence(MRISequence):
    """MR Angiography: Connectivity and flow paths through the character.

    Skeletonize to find the "vascular" network of the character,
    then analyze branching, endpoints, and flow paths.
    This reveals the character's connectivity graph."""

    def __init__(self):
        super().__init__('MRA', 'Angiography (connectivity paths)')

    def acquire(self, binary):
        if binary.sum() < 3:
            return [binary, np.zeros_like(binary)]

        # Skeletonize = find the vascular network
        try:
            skeleton = morphology.skeletonize(binary.astype(bool)).astype(np.uint8)
        except Exception:
            skeleton = binary

        # Find branch points and endpoints
        if skeleton.sum() > 0:
            kernel = np.array([[1, 1, 1],
                               [1, 0, 1],
                               [1, 1, 1]])
            neighbors = ndimage.convolve(skeleton, kernel, mode='constant', cval=0)

            # Endpoints: skeleton pixels with 1 neighbor
            endpoints = ((skeleton == 1) & (neighbors == 1)).astype(np.uint8)

            # Branch points: skeleton pixels with 3+ neighbors
            branches = ((skeleton == 1) & (neighbors >= 3)).astype(np.uint8)

            # Vessel map: skeleton with branch/endpoint annotations
            vessel_map = skeleton.copy()
            vessel_map[branches > 0] = 2  # branch points marked as 2
            vessel_map[endpoints > 0] = 3  # endpoints marked as 3
        else:
            vessel_map = skeleton

        return [skeleton, vessel_map]


# =============================================================================
# FLAIR FEATURE EXTRACTION
# Measure the response to each MRI sequence
# =============================================================================

def measure_flair_response(binary):
    """
    Measure a binary image's state with 12 features optimized for
    FLAIR-style analysis (more topological detail than standard measure_response).
    """
    if binary.size == 0 or binary.sum() == 0:
        return np.zeros(12)

    h, w = binary.shape

    # 0: Mass (ink fraction)
    mass = binary.sum() / binary.size

    # 1-2: Component and hole topology
    padded = np.pad(binary, 1, mode='constant', constant_values=0)
    _, n_fg = ndimage.label(padded)
    _, n_bg = ndimage.label(1 - padded)
    n_holes = n_bg - 1

    # 3-4: Center of mass (normalized)
    total = binary.sum()
    if total > 0 and h > 0 and w > 0:
        rows = np.arange(h).reshape(-1, 1)
        cols = np.arange(w).reshape(1, -1)
        vc = (binary * rows).sum() / (total * h)
        hc = (binary * cols).sum() / (total * w)
    else:
        vc, hc = 0.5, 0.5

    # 5: Vertical symmetry
    v_sym = np.mean(binary == np.fliplr(binary)) if w >= 2 else 1.0

    # 6: Horizontal symmetry
    h_sym = np.mean(binary == np.flipud(binary)) if h >= 2 else 1.0

    # 7: Compactness (perimeter^2 / area)
    edges = (np.abs(np.diff(binary.astype(int), axis=0)).sum() +
             np.abs(np.diff(binary.astype(int), axis=1)).sum())
    compactness = edges**2 / max(total, 1) / 100

    # 8: Stroke width estimate
    dist = ndimage.distance_transform_edt(binary)
    ink_mask = binary > 0
    stroke_width = dist[ink_mask].mean() if ink_mask.any() else 0.0
    diag = np.sqrt(h**2 + w**2)
    stroke_width = stroke_width / max(diag, 1.0)

    # 9: Euler number (components - holes)
    euler = float(n_fg - n_holes)

    # 10: Aspect ratio
    aspect = w / max(h, 1)

    # 11: Edge density (fraction of boundary pixels)
    edge_density = float(edges) / max(h * w, 1)

    return np.array([
        mass, float(n_fg), float(n_holes),
        vc, hc, v_sym, h_sym, compactness,
        stroke_width, euler, aspect, edge_density
    ])


def extract_flair_signature(binary):
    """
    THE COMPLETE FLAIR PROTOCOL.

    Like a clinical MRI study, we run multiple sequences and combine
    their results into a comprehensive diagnostic signature.

    Protocol:
      1. T1w (erosion cascade)      → 4 stages × 12 features = 48 dims
      2. T2w (dilation cascade)     → 4 stages × 12 features = 48 dims
      3. FLAIR (inversion recovery) → 4 stages × 12 features = 48 dims
      4. DWI (diffusion)            → 2 maps × 12 features   = 24 dims
      5. SWI (micro-structure)      → 2 maps × 12 features   = 24 dims
      6. MRA (connectivity)         → 2 maps × 12 features   = 24 dims
      7. Delta signatures           → 3 deltas × 12 features = 36 dims

    Total: 252-dimensional FLAIR signature.
    """
    if binary.sum() < 2:
        return np.zeros(252)

    # Normalize to standard size
    normed = _normalize_for_flair(binary, size=(32, 32))

    responses = []

    # === SEQUENCE 1: T1-weighted (erosion cascade) ===
    t1 = T1Weighted()
    t1_cascade = t1.acquire(normed)
    for stage in t1_cascade:
        responses.append(measure_flair_response(stage))

    # === SEQUENCE 2: T2-weighted (dilation cascade) ===
    t2 = T2Weighted()
    t2_cascade = t2.acquire(normed)
    for stage in t2_cascade:
        responses.append(measure_flair_response(stage))

    # === SEQUENCE 3: FLAIR (inversion recovery) ===
    flair = FLAIRSequence()
    flair_stages = flair.acquire(normed)
    for stage in flair_stages:
        responses.append(measure_flair_response(stage))

    # === SEQUENCE 4: DWI (diffusion) ===
    dwi = DWISequence()
    dwi_maps = dwi.acquire(normed)
    for m in dwi_maps:
        if m.dtype == float and m.max() <= 1.0:
            responses.append(measure_flair_response((m > 0.3).astype(np.uint8)))
        else:
            responses.append(measure_flair_response(m))

    # === SEQUENCE 5: SWI (susceptibility) ===
    swi = SWISequence()
    swi_maps = swi.acquire(normed)
    for m in swi_maps:
        if m.dtype == float or (hasattr(m, 'max') and isinstance(m.flat[0], (float, np.floating))):
            responses.append(measure_flair_response((m > 0.3).astype(np.uint8)))
        else:
            responses.append(measure_flair_response(m))

    # === SEQUENCE 6: MRA (angiography) ===
    mra = MRASequence()
    mra_maps = mra.acquire(normed)
    for m in mra_maps:
        responses.append(measure_flair_response((m > 0).astype(np.uint8)))

    # === SEQUENCE 7: Delta signatures ===
    # Rate of change between sequences reveals character dynamics
    baseline = responses[0]  # T1 original

    # T1 delta: how much changes under erosion
    t1_delta = responses[0] - responses[2]  # original vs eroded-2
    responses.append(t1_delta)

    # T2 delta: how much changes under dilation
    t2_delta = responses[4] - responses[6]  # T2 original vs dilated-2
    responses.append(t2_delta)

    # FLAIR delta: inversion recovery difference
    flair_delta = responses[8] - responses[11]  # FLAIR original vs attenuated
    responses.append(flair_delta)

    # Concatenate all responses into one signature
    signature = np.concatenate(responses)

    # Ensure exact dimensionality
    if len(signature) < 252:
        signature = np.concatenate([signature, np.zeros(252 - len(signature))])
    elif len(signature) > 252:
        signature = signature[:252]

    return signature


def _normalize_for_flair(region, size=(32, 32)):
    """Normalize binary region to standard size for FLAIR analysis."""
    img = Image.fromarray((region * 255).astype(np.uint8)).resize(size, Image.LANCZOS)
    return (np.array(img) > 127).astype(np.uint8)


# =============================================================================
# FLAIR LANDSCAPE: Fluid probability terrain shaped by experience
# =============================================================================

class FLAIRLandscape:
    """A character landscape built from FLAIR signatures.

    Each character is a probability terrain in 252-dimensional space.
    New observations flow into the landscape, reshaping it.
    Recognition = finding which landscape the observation fits best."""

    def __init__(self, label):
        self.label = label
        self.observations = []
        self.mean = None
        self.variance = None
        self.n = 0

    def absorb(self, signature):
        """Absorb a new FLAIR signature into this landscape."""
        self.observations.append(signature.copy())
        self.n = len(self.observations)
        obs = np.array(self.observations)
        self.mean = obs.mean(axis=0)
        if self.n >= 2:
            self.variance = np.maximum(obs.var(axis=0), 0.1 / np.sqrt(self.n))
        else:
            self.variance = np.ones_like(signature) * 2.0

    def fit(self, signature):
        """How well does this signature fit this landscape?
        Returns log-likelihood score."""
        if self.mean is None:
            return -float('inf')
        diff = signature - self.mean
        precision = 1.0 / (self.variance + 1e-8)
        return -0.5 * np.sum(diff**2 * precision) + np.log(self.n + 1) * 0.5

    def confidence(self, signature):
        """Return a 0-1 confidence score for this signature."""
        score = self.fit(signature)
        # Sigmoid-like normalization
        return 1.0 / (1.0 + np.exp(-score / 100))

    def to_dict(self):
        return {
            'label': self.label,
            'n': self.n,
            'mean': self.mean.tolist() if self.mean is not None else None,
            'variance': self.variance.tolist() if self.variance is not None else None,
        }


# =============================================================================
# FLAIR-OCR ENGINE
# =============================================================================

class FLAIR_OCR:
    """
    Character recognition via FLAIR (Fluid-Attenuated Inversion Recovery) signatures.

    Each character's identity is revealed not by how it LOOKS,
    but by how it RESPONDS to a complete MRI protocol:
    T1, T2, FLAIR, DWI, SWI, MRA sequences.

    252-dimensional signature per character.
    """

    def __init__(self):
        self.landscapes = {}
        self.sequences = {
            'T1w': T1Weighted(),
            'T2w': T2Weighted(),
            'FLAIR': FLAIRSequence(),
            'DWI': DWISequence(),
            'SWI': SWISequence(),
            'MRA': MRASequence(),
        }

    def train(self, label, binary_region):
        """Train on one character — absorb its FLAIR signature into the landscape."""
        sig = extract_flair_signature(binary_region)
        if label not in self.landscapes:
            self.landscapes[label] = FLAIRLandscape(label)
        self.landscapes[label].absorb(sig)

    def recognize(self, binary_region, top_k=5):
        """Recognize by finding best-fit FLAIR landscape."""
        sig = extract_flair_signature(binary_region)
        scores = [(label, land.fit(sig)) for label, land in self.landscapes.items()]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def predict(self, binary_region):
        """Predict with confidence — interface compatible with ensemble."""
        results = self.recognize(binary_region, top_k=5)
        if not results:
            return {'predicted': '?', 'confidence': 0.0}

        best_label, best_score = results[0]
        second_score = results[1][1] if len(results) > 1 else float('-inf')

        total_range = results[0][1] - results[-1][1] if len(results) > 1 else 1.0
        margin = best_score - second_score
        confidence = max(0, min(margin / max(abs(total_range), 0.01), 1.0))

        return {
            'predicted': best_label,
            'confidence': confidence,
            'score': best_score,
            'candidates': results,
        }

    def analyze_character(self, binary_region):
        """
        Full diagnostic analysis of a character — like a radiology report.

        Returns per-sequence measurements and the integrated diagnosis.
        """
        normed = _normalize_for_flair(binary_region, (32, 32))
        report = {}

        for seq_name, seq in self.sequences.items():
            stages = seq.acquire(normed)
            measurements = []
            for stage in stages:
                if isinstance(stage, np.ndarray):
                    if stage.dtype == float or (hasattr(stage, 'max') and
                            isinstance(stage.flat[0] if stage.size > 0 else 0, (float, np.floating))):
                        meas = measure_flair_response((stage > 0.3).astype(np.uint8))
                    else:
                        meas = measure_flair_response(stage)
                    measurements.append({
                        'mass': float(meas[0]),
                        'components': float(meas[1]),
                        'holes': float(meas[2]),
                        'v_center': float(meas[3]),
                        'h_center': float(meas[4]),
                        'v_symmetry': float(meas[5]),
                        'h_symmetry': float(meas[6]),
                        'compactness': float(meas[7]),
                        'stroke_width': float(meas[8]),
                        'euler': float(meas[9]),
                        'aspect': float(meas[10]),
                        'edge_density': float(meas[11]),
                    })
            report[seq_name] = measurements

        # Integrated diagnosis
        prediction = self.predict(binary_region)
        report['diagnosis'] = prediction

        return report

    def get_sequence_contribution(self, binary_region, true_label=None):
        """
        Ablation analysis: Which MRI sequence contributed most to the recognition?

        Like asking: "Was FLAIR or DWI more diagnostic for this case?"
        """
        full_sig = extract_flair_signature(binary_region)
        full_result = self.predict(binary_region)

        contributions = {}
        # Each sequence occupies a block of the 252-dim signature
        # T1: 0-47, T2: 48-95, FLAIR: 96-143, DWI: 144-167, SWI: 168-191, MRA: 192-215, Delta: 216-251
        blocks = {
            'T1w': (0, 48),
            'T2w': (48, 96),
            'FLAIR': (96, 144),
            'DWI': (144, 168),
            'SWI': (168, 192),
            'MRA': (192, 216),
            'Delta': (216, 252),
        }

        for seq_name, (start, end) in blocks.items():
            # Ablate this sequence (zero it out) and see how prediction changes
            ablated_sig = full_sig.copy()
            ablated_sig[start:end] = 0

            # Score with ablated signature
            ablated_scores = []
            for label, land in self.landscapes.items():
                ablated_scores.append((label, land.fit(ablated_sig)))
            ablated_scores.sort(key=lambda x: x[1], reverse=True)

            # Did ablation change the prediction?
            ablated_best = ablated_scores[0][0] if ablated_scores else '?'
            prediction_changed = ablated_best != full_result['predicted']

            # Score drop = importance
            full_score = full_result.get('score', 0)
            ablated_score = ablated_scores[0][1] if ablated_scores else 0
            score_drop = full_score - ablated_score

            contributions[seq_name] = {
                'score_drop': float(score_drop),
                'prediction_changed': prediction_changed,
                'ablated_prediction': ablated_best,
                'importance': abs(float(score_drop)),
            }

        # Rank by importance
        ranked = sorted(contributions.items(), key=lambda x: x[1]['importance'], reverse=True)
        contributions['_ranked'] = [name for name, _ in ranked]

        return contributions

    def stats(self):
        """Engine statistics."""
        return {
            'n_landscapes': len(self.landscapes),
            'signature_dims': 252,
            'sequences': list(self.sequences.keys()),
            'characters': {
                label: land.n for label, land in self.landscapes.items()
            },
        }
