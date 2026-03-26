"""
Fluid Theory OCR (FT-OCR)
==========================

No rules. No thresholds. No walls.

Each character is a LANDSCAPE — a probability terrain shaped by 
every observation the system has ever seen. When a new observation 
arrives, it flows into the landscape that fits most naturally.

When the system is wrong, it doesn't add a rule. It ABSORBS the 
new observation into the correct landscape, which subtly reshapes 
the terrain. The landscape gets smoother, wider, more refined — 
never rigid.

"Just because you haven't seen a pink cat doesn't mean it can't exist."

A character theory is not "endpoints MUST be 0." It's "endpoints 
TEND to cluster around 0, with this spread, and this confidence, 
based on what I've seen so far." When a serif 'O' shows up with 
2 endpoints, the landscape stretches to accommodate. It doesn't break.

This is the difference between:
  - "Remove fluid" (a rigid rule that breaks when context changes)
  - "Reduce preload" (a principle that adapts: diuretics, UF, 
    restriction — whatever fits the context)

Author: Bader & Claude — March 2026
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage
from skimage import morphology, filters
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# MEDIUM DISPLACEMENT PIPELINE (proven, stable)
# =============================================================================

def estimate_background(img, k=25):
    return filters.gaussian(morphology.closing(img, morphology.disk(k)), sigma=k/2)

def compute_displacement(img, bg):
    d = bg.astype(float) - img.astype(float)
    r = d.max() - d.min()
    return (d - d.min()) / r if r > 0 else d

def detect_perturbation(disp, thresh=0.25):
    return morphology.remove_small_objects(disp > thresh, min_size=8).astype(np.uint8)

def extract_region(binary, pad=2):
    coords = np.argwhere(binary > 0)
    if len(coords) == 0: return binary
    r0, c0 = np.maximum(coords.min(axis=0) - pad, 0)
    r1, c1 = np.minimum(coords.max(axis=0) + pad, np.array(binary.shape) - 1)
    return binary[r0:r1+1, c0:c1+1]

def normalize(region, size=(64,64)):
    img = Image.fromarray((region*255).astype(np.uint8)).resize(size, Image.NEAREST)
    return (np.array(img) > 127).astype(np.uint8)

def image_to_region(char_img):
    bg = estimate_background(char_img, k=15)
    return normalize(extract_region(detect_perturbation(compute_displacement(char_img, bg))))


# =============================================================================
# FEATURE EXTRACTION — The observation
# =============================================================================

def extract_features(br):
    """Extract a flat feature vector from a binary region.
    Every feature measures the medium's response."""
    h, w = br.shape
    feats = []
    
    # Topology
    padded = np.pad(br, 1, mode='constant', constant_values=0)
    _, nfg = ndimage.label(padded)
    _, nbg = ndimage.label(1 - padded)
    holes = nbg - 1
    feats.extend([nfg, holes, nfg - holes])  # components, holes, euler
    
    # Displacement ratio
    feats.append(float(np.mean(br)))
    
    # Symmetry
    feats.append(float(np.mean(br == np.fliplr(br))) if w >= 2 else 1.0)
    feats.append(float(np.mean(br == np.flipud(br))) if h >= 2 else 1.0)
    
    # Center of mass
    total = br.sum()
    if total > 0 and h > 0 and w > 0:
        rows, cols = np.arange(h).reshape(-1,1), np.arange(w).reshape(1,-1)
        feats.append(float((br*rows).sum() / (total*h)))
        feats.append(float((br*cols).sum() / (total*w)))
    else:
        feats.extend([0.5, 0.5])
    
    # Quadrant density (proportional)
    mh, mw = h//2, w//2
    quads = [
        float(br[:mh,:mw].mean()) if mh>0 and mw>0 else 0,
        float(br[:mh,mw:].mean()) if mh>0 else 0,
        float(br[mh:,:mw].mean()) if mw>0 else 0,
        float(br[mh:,mw:].mean()),
    ]
    qt = sum(quads)
    feats.extend([q/qt if qt > 0 else 0.25 for q in quads])
    
    # Projection profiles (6 bins each)
    for axis, length in [(1, h), (0, w)]:
        raw = br.mean(axis=axis)
        bins = 6
        proj = np.zeros(bins)
        bw = max(1, len(raw)//bins)
        for i in range(bins):
            s, e = i*bw, min((i+1)*bw, len(raw))
            if s < len(raw): proj[i] = raw[s:e].mean()
        pt = proj.sum()
        if pt > 0: proj /= pt
        feats.extend(proj.tolist())
    
    # Crossing counts (6 lines each direction)
    for axis in [0, 1]:
        for i in range(6):
            if axis == 0:
                row = min(int((i+0.5)*h/6), h-1)
                line = br[row, :]
            else:
                col = min(int((i+0.5)*w/6), w-1)
                line = br[:, col]
            feats.append(float(np.abs(np.diff(line.astype(int))).sum() / 2))
    
    # Junctions and endpoints
    if h >= 4 and w >= 4 and br.sum() >= 10:
        try:
            skel = morphology.skeletonize(br.astype(bool))
            nc = ndimage.convolve(skel.astype(int), np.ones((3,3), dtype=int),
                                   mode='constant') - skel.astype(int)
            _, n_ep = ndimage.label(skel & (nc == 1))
            _, n_jp = ndimage.label(skel & (nc >= 3))
            feats.extend([float(n_ep), float(n_jp)])
        except:
            feats.extend([0.0, 0.0])
    else:
        feats.extend([0.0, 0.0])
    
    return np.array(feats, dtype=float)


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


# =============================================================================
# THE FLUID LANDSCAPE — No rules. Just experience condensed into shape.
# =============================================================================

class Landscape:
    """
    A character's identity as a probability landscape.
    
    Not "what this character IS" but "where observations of this 
    character TEND to fall." 
    
    The landscape is defined by:
    - A center (mean of all observations)
    - A spread (variance — how much each feature varies)
    - A confidence (how many observations shaped this landscape)
    
    With 1 observation: the landscape is wide and uncertain.
    With 10 observations: it tightens around the consistent features 
    and stays wide on the variable ones.
    
    Features that vary a lot across fonts (like junction count) 
    automatically get low weight — the landscape is FLAT in that 
    dimension, meaning it doesn't discriminate there.
    
    Features that are consistent (like hole count) automatically get 
    high weight — the landscape is PEAKED in that dimension.
    
    The system discovers what matters by observing what's consistent.
    No human needs to specify weights. The data sculpts the landscape.
    """
    
    def __init__(self, label):
        self.label = label
        self.observations = []  # Raw feature vectors
        self.mean = None        # Center of the landscape
        self.variance = None    # Spread of the landscape
        self.n = 0              # Experience count
        
        # Track record
        self.n_correct = 0
        self.n_errors = 0
        self.confused_with = defaultdict(int)
    
    def absorb(self, feature_vector):
        """
        Absorb a new observation into the landscape.
        
        The landscape doesn't ADD A RULE. It RESHAPES.
        Like dropping a stone into water — the surface adjusts 
        everywhere, not just at the impact point.
        """
        self.observations.append(feature_vector.copy())
        self.n = len(self.observations)
        
        if self.n == 1:
            self.mean = feature_vector.copy()
            # Wide initial variance — we know almost nothing
            self.variance = np.ones_like(feature_vector) * 2.0
        else:
            obs_matrix = np.array(self.observations)
            self.mean = obs_matrix.mean(axis=0)
            
            if self.n >= 2:
                raw_var = obs_matrix.var(axis=0)
                # Floor: never let variance go to zero (pink cats exist)
                # The floor shrinks with experience but never disappears
                floor = 0.1 / np.sqrt(self.n)
                self.variance = np.maximum(raw_var, floor)
            else:
                self.variance = np.ones_like(feature_vector) * 1.0
    
    def fit(self, feature_vector):
        """
        How naturally does this observation fit into this landscape?
        
        Returns a score (higher = better fit).
        
        Uses Mahalanobis-like distance: features where the landscape 
        is NARROW (low variance = consistent) matter MORE.
        Features where the landscape is WIDE (high variance = variable) 
        matter LESS.
        
        This is the key: THE LANDSCAPE ITSELF DETERMINES WHAT MATTERS.
        Consistent features automatically discriminate.
        Variable features automatically get ignored.
        No hand-tuned weights needed.
        """
        if self.mean is None:
            return -float('inf')
        
        diff = feature_vector - self.mean
        
        # Precision = 1/variance (how much each feature matters)
        precision = 1.0 / (self.variance + 1e-8)
        
        # Weighted squared distance
        weighted_dist = np.sum(diff**2 * precision)
        
        # Log-likelihood (unnormalized Gaussian)
        # Higher = better fit
        score = -0.5 * weighted_dist
        
        # Confidence bonus: landscapes with more experience are more trustworthy
        # But only slightly — we don't want to overweight just because we've 
        # seen more of one character
        confidence = np.log(self.n + 1) * 0.5
        
        return score + confidence
    
    def get_profile(self):
        """Human-readable profile of what this landscape looks like."""
        if self.mean is None:
            return f"'{self.label}': no observations"
        
        lines = [f"'{self.label}' — {self.n} observations, "
                 f"{self.n_correct} correct, {self.n_errors} errors"]
        
        # Show the most peaked features (most consistent = most diagnostic)
        if self.variance is not None:
            precision = 1.0 / (self.variance + 1e-8)
            ranked = np.argsort(-precision)
            
            lines.append(f"  Most consistent features (the landscape peaks):")
            for idx in ranked[:5]:
                if idx < len(FEATURE_NAMES):
                    name = FEATURE_NAMES[idx]
                    lines.append(f"    {name}: mean={self.mean[idx]:.3f}, "
                               f"spread={np.sqrt(self.variance[idx]):.3f}")
            
            lines.append(f"  Most variable features (the landscape is flat):")
            for idx in ranked[-3:]:
                if idx < len(FEATURE_NAMES):
                    name = FEATURE_NAMES[idx]
                    lines.append(f"    {name}: mean={self.mean[idx]:.3f}, "
                               f"spread={np.sqrt(self.variance[idx]):.3f}")
        
        if self.confused_with:
            pairs = sorted(self.confused_with.items(), key=lambda x: -x[1])
            lines.append(f"  Confused with: " + 
                        ", ".join(f"'{k}'×{v}" for k, v in pairs[:5]))
        
        return '\n'.join(lines)


# =============================================================================
# THE FLUID ENGINE — Predict, collide, reshape
# =============================================================================

class FluidEngine:
    """
    The complete fluid learning engine.
    
    No rules. No thresholds. No rigid boundaries.
    
    Each character is a landscape. New observations reshape landscapes.
    Classification = which landscape does this observation fit into 
    most naturally?
    
    Learning = absorb the observation into the CORRECT landscape, 
    which subtly changes its shape for all future classifications.
    """
    
    def __init__(self):
        self.landscapes = {}
        self.total_predictions = 0
        self.total_correct = 0
        self.history = []  # Learning journey
    
    def teach(self, label, binary_region):
        """Show the system one example. Like reading the textbook."""
        fv = extract_features(binary_region)
        if label not in self.landscapes:
            self.landscapes[label] = Landscape(label)
        self.landscapes[label].absorb(fv)
        return fv
    
    def predict(self, binary_region):
        """
        Find which landscape this observation fits most naturally.
        
        Also returns the MARGIN — how much better the best fit is 
        than the second best. Large margin = confident. Small margin = 
        ambiguous, the observation sits between two landscapes.
        """
        fv = extract_features(binary_region)
        
        scores = []
        for label, landscape in self.landscapes.items():
            score = landscape.fit(fv)
            scores.append((score, label))
        
        scores.sort(reverse=True)  # Highest score = best fit
        
        best = scores[0]
        second = scores[1] if len(scores) > 1 else (float('-inf'), '?')
        
        margin = best[0] - second[0]
        total_range = scores[0][0] - scores[-1][0] if len(scores) > 1 else 1.0
        confidence = min(margin / max(abs(total_range), 0.01), 1.0)
        confidence = max(confidence, 0.0)
        
        self.total_predictions += 1
        
        return {
            'predicted': best[1],
            'confidence': confidence,
            'margin': margin,
            'features': fv,
            'top_5': [(label, score) for score, label in scores[:5]],
        }
    
    def experience(self, prediction, true_label):
        """
        Learn from reality.
        
        If correct: absorb into the correct landscape (reinforce).
        If wrong: absorb into the correct landscape (reshape).
        
        Either way, the landscape evolves. No rules added.
        Just experience absorbed into shape.
        """
        predicted = prediction['predicted']
        fv = prediction['features']
        correct = predicted == true_label
        
        if correct:
            self.total_correct += 1
            self.landscapes[true_label].n_correct += 1
        else:
            self.landscapes[predicted].n_errors += 1
            self.landscapes[predicted].confused_with[true_label] += 1
        
        # ABSORB the observation into the correct landscape
        if true_label in self.landscapes:
            self.landscapes[true_label].absorb(fv)
        
        # Track the journey
        entry = {
            'true': true_label,
            'predicted': predicted,
            'correct': correct,
            'confidence': prediction['confidence'],
            'margin': prediction['margin'],
        }
        
        if not correct:
            # What made us wrong? Find the features where the correct 
            # landscape and predicted landscape differ most
            true_land = self.landscapes.get(true_label)
            pred_land = self.landscapes.get(predicted)
            
            if true_land and pred_land and true_land.mean is not None and pred_land.mean is not None:
                diff = np.abs(true_land.mean - pred_land.mean)
                precision = 1.0 / (true_land.variance + pred_land.variance + 1e-8)
                diagnostic = diff * precision  # Features that differ AND are reliable
                
                top_idx = np.argsort(-diagnostic)[:3]
                entry['diagnostic_features'] = [
                    {
                        'feature': FEATURE_NAMES[i] if i < len(FEATURE_NAMES) else f'feat_{i}',
                        'observed': float(fv[i]),
                        'true_expects': float(true_land.mean[i]),
                        'pred_expects': float(pred_land.mean[i]),
                        'diagnostic_power': float(diagnostic[i]),
                    }
                    for i in top_idx
                ]
        
        self.history.append(entry)
        return entry
    
    def get_accuracy(self):
        if self.total_predictions == 0:
            return 0.0
        return self.total_correct / self.total_predictions * 100
    
    def get_landscape_summary(self, label):
        if label in self.landscapes:
            return self.landscapes[label].get_profile()
        return f"No landscape for '{label}'"


# =============================================================================
# IMAGE UTILITIES
# =============================================================================

FONTS = [
    ('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 'DejaVu Sans'),
    ('/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf', 'DejaVu Serif'),
    ('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 'DejaVu Bold'),
    ('/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf', 'Serif Bold'),
    ('/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf', 'DejaVu Mono'),
    ('/usr/share/fonts/truetype/dejavu/DejaVuSerif-BoldItalic.ttf', 'Serif BoldItalic'),
    ('/usr/share/fonts/truetype/freefont/FreeMono.ttf', 'FreeMono'),
    ('/usr/share/fonts/truetype/freefont/FreeSans.ttf', 'FreeSans'),
    ('/usr/share/fonts/truetype/freefont/FreeSerif.ttf', 'FreeSerif'),
    ('/usr/share/fonts/truetype/google-fonts/Poppins-Bold.ttf', 'Poppins Bold'),
]

ALPHABET = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')

def render_char(char, font_path, size=80, img_size=(100,100)):
    img = Image.new('L', img_size, color=255)
    draw = ImageDraw.Draw(img)
    try: font = ImageFont.truetype(font_path, size)
    except: font = ImageFont.load_default()
    bbox = draw.textbbox((0,0), char, font=font)
    x = (img_size[0]-(bbox[2]-bbox[0]))//2 - bbox[0]
    y = (img_size[1]-(bbox[3]-bbox[1]))//2 - bbox[1]
    draw.text((x,y), char, fill=0, font=font)
    return np.array(img)


# =============================================================================
# THE LEARNING JOURNEY
# =============================================================================

def run_journey():
    engine = FluidEngine()
    
    # =========================================================
    # TEXTBOOK PHASE
    # =========================================================
    print("=" * 70)
    print("THE TEXTBOOK — Learning principles from one font")
    print("=" * 70)
    
    for char in ALPHABET:
        img = render_char(char, FONTS[0][0])
        region = image_to_region(img)
        engine.teach(char, region)
    
    print(f"\n  Formed {len(engine.landscapes)} landscapes from {FONTS[0][1]}")
    print(f"  Each landscape: 1 observation, wide uncertainty")
    
    # Show initial landscapes for key characters
    for char in ['O', 'D', 'T', 'I']:
        print(f"\n  {engine.get_landscape_summary(char)}")
    
    # =========================================================
    # CLINICAL ROTATIONS — Experience reshapes the landscape
    # =========================================================
    
    rotation_accuracies = []
    cumulative_accuracy = []
    
    for rot_idx, (font_path, font_name) in enumerate(FONTS[1:], 1):
        print(f"\n{'='*70}")
        print(f"ROTATION {rot_idx}: {font_name}")
        print(f"{'='*70}")
        
        correct = 0
        errors = []
        
        for char in ALPHABET:
            img = render_char(char, font_path)
            region = image_to_region(img)
            
            prediction = engine.predict(region)
            result = engine.experience(prediction, char)
            
            if result['correct']:
                correct += 1
            else:
                errors.append(result)
        
        acc = correct / len(ALPHABET) * 100
        rotation_accuracies.append(acc)
        cum_acc = engine.get_accuracy()
        cumulative_accuracy.append(cum_acc)
        
        bar = "█" * int(acc / 2.5) + "░" * (40 - int(acc / 2.5))
        print(f"\n  This rotation:  {bar} {acc:.1f}%")
        print(f"  Cumulative:     {cum_acc:.1f}% over {engine.total_predictions} predictions")
        print(f"  Landscape depth: {engine.landscapes['A'].n} observations per character")
        
        if errors:
            print(f"\n  Errors ({len(errors)}):")
            for e in errors[:6]:
                line = f"    '{e['true']}' → '{e['predicted']}' (conf={e['confidence']:.2f})"
                if 'diagnostic_features' in e:
                    df = e['diagnostic_features'][0]
                    line += f"  key: {df['feature']} was {df['observed']:.2f}, " \
                            f"'{e['true']}' expects {df['true_expects']:.2f}"
                print(line)
    
    # =========================================================
    # THE LEARNING CURVE
    # =========================================================
    print(f"\n{'='*70}")
    print("THE LEARNING CURVE")
    print("=" * 70)
    
    print(f"\n  Per-rotation accuracy (how well it does on each new font):\n")
    for i, (acc, (_, fname)) in enumerate(zip(rotation_accuracies, FONTS[1:])):
        bar = "█" * int(acc / 2.5) + "░" * (40 - int(acc / 2.5))
        depth = i + 2  # Number of observations per landscape
        print(f"  R{i+1} {fname:18s} {bar} {acc:5.1f}% (depth={depth})")
    
    print(f"\n  First rotation:  {rotation_accuracies[0]:.1f}%")
    print(f"  Last rotation:   {rotation_accuracies[-1]:.1f}%")
    change = rotation_accuracies[-1] - rotation_accuracies[0]
    print(f"  Change:          {change:+.1f}%")
    
    # Trend: average of first 3 vs last 3
    if len(rotation_accuracies) >= 6:
        early = np.mean(rotation_accuracies[:3])
        late = np.mean(rotation_accuracies[-3:])
        print(f"\n  Early average (R1-R3):  {early:.1f}%")
        print(f"  Late average (R7-R9):   {late:.1f}%")
        print(f"  Trend:                  {late - early:+.1f}%")
    
    # =========================================================
    # EVOLVED LANDSCAPES
    # =========================================================
    print(f"\n{'='*70}")
    print("EVOLVED LANDSCAPES — What experience taught the system")
    print("=" * 70)
    
    for char in ['O', 'D', 'I', 'T', 'V', 'W', 'B']:
        print(f"\n  {engine.get_landscape_summary(char)}")
    
    # =========================================================
    # THE SELF-KNOWLEDGE — What the system knows it doesn't know
    # =========================================================
    print(f"\n{'='*70}")
    print("SELF-KNOWLEDGE — The system's awareness of its own uncertainty")
    print("=" * 70)
    
    # Find the most confused pairs
    confusion_pairs = defaultdict(int)
    for entry in engine.history:
        if not entry['correct']:
            pair = tuple(sorted([entry['true'], entry['predicted']]))
            confusion_pairs[pair] += 1
    
    print(f"\n  Most confused pairs (where the landscape boundaries overlap):")
    for pair, count in sorted(confusion_pairs.items(), key=lambda x: -x[1])[:10]:
        c1, c2 = pair
        l1, l2 = engine.landscapes.get(c1), engine.landscapes.get(c2)
        
        if l1 and l2 and l1.mean is not None and l2.mean is not None:
            # Find the most diagnostic feature between them
            diff = np.abs(l1.mean - l2.mean)
            precision = 1.0 / (l1.variance + l2.variance + 1e-8)
            diagnostic = diff * precision
            best_idx = np.argmax(diagnostic)
            best_feat = FEATURE_NAMES[best_idx] if best_idx < len(FEATURE_NAMES) else f'f{best_idx}'
            
            overlap = np.sum(np.minimum(
                1.0 / np.sqrt(2 * np.pi * (l1.variance + 1e-8)),
                1.0 / np.sqrt(2 * np.pi * (l2.variance + 1e-8))
            ))
            
            print(f"    '{c1}' ↔ '{c2}': confused {count}× | "
                  f"best separator: {best_feat}")
    
    # =========================================================
    # RE-TEST: Does the first font still work?
    # =========================================================
    print(f"\n{'='*70}")
    print("STABILITY CHECK — Can it still read the first font?")
    print("=" * 70)
    
    correct = 0
    for char in ALPHABET:
        img = render_char(char, FONTS[0][0])
        region = image_to_region(img)
        pred = engine.predict(region)
        if pred['predicted'] == char:
            correct += 1
    
    stability = correct / len(ALPHABET) * 100
    bar = "█" * int(stability / 2.5) + "░" * (40 - int(stability / 2.5))
    print(f"\n  {FONTS[0][1]:18s} {bar} {stability:.1f}%")
    
    if stability >= 90:
        print("  The landscape absorbed new fonts WITHOUT forgetting the old ones.")
    else:
        print("  Some catastrophic forgetting — the landscape shifted too far.")
    
    # =========================================================
    # FINAL SUMMARY
    # =========================================================
    print(f"\n{'='*70}")
    print("WHAT THE FLUID LANDSCAPE PROVES")
    print("=" * 70)
    print(f"""
  Architecture: No rules. No thresholds. No rigid boundaries.
  Each character = a probability landscape shaped by experience.
  
  Key properties:
    • Features that are CONSISTENT automatically matter more
      (the landscape peaks there)
    • Features that VARY automatically matter less
      (the landscape is flat there)
    • No hand-tuned weights — the data sculpts the landscape
    • New observations reshape, they don't overwrite
    • The system can never "break" from an unexpected input —
      it just stretches to accommodate

  Results:
    First rotation:  {rotation_accuracies[0]:.1f}%
    Last rotation:   {rotation_accuracies[-1]:.1f}%
    Stability:       {stability:.1f}% (re-test on original font)
    
  Total observations: {engine.landscapes['A'].n} per character
  Parameters learned: 0 (only means and variances from data)
  Rules: 0
  Neural network: None
""")
    
    return engine, rotation_accuracies


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║          FLUID THEORY OCR (FT-OCR)                     ║")
    print("║                                                        ║")
    print("║  No rules. No walls. Just landscapes shaped by         ║")
    print("║  experience. What's consistent peaks. What varies       ║")
    print("║  flattens. The system discovers what matters.          ║")
    print("║                                                        ║")
    print("║  'Just because you haven't seen a pink cat             ║")
    print("║   doesn't mean it can't exist.'                        ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print()
    
    engine, accuracies = run_journey()
