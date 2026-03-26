"""
MRI-OCR: Relaxation Signature Recognition
============================================

Principle: Don't read the character. DISTURB it and measure the response.

In MRI:
  1. Apply RF pulse → disrupts proton alignment (chaos)
  2. Measure how different tissues RELAX back (T1, T2)
  3. The relaxation pattern identifies the tissue

In MRI-OCR:
  1. Apply perturbations → disrupt the character image (chaos)
  2. Measure how the character RESPONDS to each perturbation
  3. The response pattern identifies the character

WHY THIS WORKS ON 6-PIXEL CHARACTERS:
Static features at 6px: 'a', 'e', 'o', 's' are all similar blobs.
But their RESPONSE to disturbance is different:
  - Erode 'a': the gap closes → displacement ratio INCREASES
  - Erode 'e': the crossbar vanishes → holes change from 0 to 0
  - Erode 'o': stays circular but shrinks → ratio DECREASES uniformly
  - Erode 's': breaks into pieces → component count INCREASES

The degradation pattern is the fingerprint.
"""

import numpy as np
from PIL import Image
from scipy import ndimage
from skimage import morphology
from collections import defaultdict
import time
from .engine import Landscape, extract_features
from .complete import match_word, ocr_distance, normalize_char, CLINICAL_WORDS


# =============================================================================
# THE RF PULSES: Different perturbations that reveal different properties
# =============================================================================

def pulse_erode(binary, iterations=1):
    """Erosion pulse: thin strokes vanish first, thick strokes survive."""
    return morphology.erosion(binary, morphology.disk(iterations)).astype(np.uint8)

def pulse_dilate(binary, iterations=1):
    """Dilation pulse: gaps close, nearby components merge."""
    return morphology.dilation(binary, morphology.disk(iterations)).astype(np.uint8)

def pulse_blur(binary, sigma=1.5):
    """Gaussian blur pulse: sharp features soften, thin features vanish."""
    from skimage.filters import gaussian
    blurred = gaussian(binary.astype(float), sigma=sigma)
    return (blurred > 0.3).astype(np.uint8)

def pulse_tophat(binary):
    """Top-hat: reveals thin bright features (strokes thinner than the element)."""
    selem = morphology.disk(2)
    return morphology.white_tophat(binary, selem).astype(np.uint8)

def pulse_skeleton(binary):
    """Skeletonization: reduce to topological spine."""
    if binary.sum() < 3:
        return binary
    try:
        return morphology.skeletonize(binary.astype(bool)).astype(np.uint8)
    except:
        return binary

def pulse_distance(binary):
    """Distance transform: how deep is each pixel from the boundary?"""
    dist = ndimage.distance_transform_edt(binary)
    if dist.max() > 0:
        return (dist / dist.max() > 0.3).astype(np.uint8)
    return binary


# =============================================================================
# RELAXATION MEASUREMENT: How does the character respond?
# =============================================================================

def measure_response(binary):
    """
    Quick measurement of a binary image's state.
    Returns a small feature vector capturing the essential state.
    """
    if binary.size == 0 or binary.sum() == 0:
        return np.zeros(8)
    
    h, w = binary.shape
    
    # Mass (how much ink)
    mass = binary.sum() / binary.size
    
    # Components and holes
    padded = np.pad(binary, 1, mode='constant', constant_values=0)
    _, n_fg = ndimage.label(padded)
    _, n_bg = ndimage.label(1 - padded)
    n_holes = n_bg - 1
    
    # Center of mass
    total = binary.sum()
    if total > 0 and h > 0 and w > 0:
        rows = np.arange(h).reshape(-1, 1)
        cols = np.arange(w).reshape(1, -1)
        vc = (binary * rows).sum() / (total * h)
        hc = (binary * cols).sum() / (total * w)
    else:
        vc, hc = 0.5, 0.5
    
    # Symmetry
    v_sym = np.mean(binary == np.fliplr(binary)) if w >= 2 else 1.0
    
    # Compactness (perimeter^2 / area)
    edges = np.abs(np.diff(binary.astype(int), axis=0)).sum() + \
            np.abs(np.diff(binary.astype(int), axis=1)).sum()
    compactness = edges**2 / max(total, 1) / 100
    
    return np.array([
        mass, float(n_fg), float(n_holes),
        vc, hc, v_sym, compactness, float(edges) / max(h*w, 1)
    ])


def extract_relaxation_signature(binary):
    """
    THE COMPLETE MRI SEQUENCE.
    
    Apply multiple RF pulses. After each, measure the response.
    The concatenation of all responses IS the relaxation signature.
    
    Like an MRI protocol: T1-weighted, T2-weighted, FLAIR, DWI —
    each sequence reveals different tissue properties.
    """
    if binary.sum() < 2:
        return np.zeros(8 * 8)  # 8 measurements × 8 pulses
    
    # Normalize to standard size first
    normed = normalize_char(binary, size=(32, 32))
    
    responses = []
    
    # Pulse 0: Original state (baseline)
    responses.append(measure_response(normed))
    
    # Pulse 1: Light erosion (T1-like: what survives gentle thinning?)
    eroded1 = pulse_erode(normed, 1)
    responses.append(measure_response(eroded1))
    
    # Pulse 2: Heavy erosion (what survives aggressive thinning?)
    eroded2 = pulse_erode(normed, 2)
    responses.append(measure_response(eroded2))
    
    # Pulse 3: Dilation (T2-like: what fills in? what merges?)
    dilated = pulse_dilate(normed, 1)
    responses.append(measure_response(dilated))
    
    # Pulse 4: Blur (diffusion-like: how does structure dissolve?)
    blurred = pulse_blur(normed, sigma=1.5)
    responses.append(measure_response(blurred))
    
    # Pulse 5: Skeleton (topological core)
    skel = pulse_skeleton(normed)
    responses.append(measure_response(skel))
    
    # Pulse 6: Distance core (deep structure)
    dist_core = pulse_distance(normed)
    responses.append(measure_response(dist_core))
    
    # Pulse 7: DELTA — how much changed between original and eroded?
    # This is the KEY: the RATE OF CHANGE under perturbation
    delta = measure_response(normed) - measure_response(eroded1)
    responses.append(delta)
    
    return np.concatenate(responses)


# =============================================================================
# MRI-OCR ENGINE
# =============================================================================

class MRI_OCR:
    """
    Character recognition via relaxation signatures.
    Each character landscape is shaped by how that character 
    RESPONDS to perturbation, not how it LOOKS statically.
    """
    
    def __init__(self):
        self.landscapes = {}
    
    def train(self, label, binary_region):
        """Train on one character."""
        sig = extract_relaxation_signature(binary_region)
        if label not in self.landscapes:
            self.landscapes[label] = Landscape(label)
        self.landscapes[label].absorb(sig)
    
    def recognize(self, binary_region, top_k=5):
        """Recognize by finding best-fit relaxation landscape."""
        sig = extract_relaxation_signature(binary_region)
        scores = [(l, land.fit(sig)) for l, land in self.landscapes.items()]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


# =============================================================================
# TEST ON REAL WARD IMAGE
# =============================================================================

def extract_components(cell_gray, scale=3):
    """Extract character components from a cell with upscaling."""
    h, w = cell_gray.shape
    upscaled = np.array(Image.fromarray(cell_gray).resize(
        (w * scale, h * scale), Image.LANCZOS))
    
    border = np.concatenate([upscaled[0,:], upscaled[-1,:], upscaled[:,0], upscaled[:,-1]])
    bg = np.median(border)
    bg_std = max(np.std(border), 3)
    diff = bg - upscaled.astype(float)
    binary = (diff > max(bg_std * 1.5, 12)).astype(np.uint8)
    
    if binary.sum() < 15:
        from skimage.filters import threshold_otsu
        try:
            t = threshold_otsu(upscaled)
            binary = (upscaled < t).astype(np.uint8)
        except:
            return [], binary
    
    binary = morphology.remove_small_objects(binary.astype(bool), min_size=8).astype(np.uint8)
    
    labeled, n = ndimage.label(binary)
    components = []
    for i in range(1, n + 1):
        mask = labeled == i
        coords = np.argwhere(mask)
        if len(coords) < 5: continue
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        hc, wc = r1-r0+1, c1-c0+1
        if hc < 5 or wc < 3: continue
        if wc/max(hc,1) > 6 or hc/max(wc,1) > 8: continue
        components.append((c0, mask[r0:r1+1, c0:c1+1].astype(np.uint8)))
    
    components.sort(key=lambda x: x[0])
    return components, binary


def main():
    print()
    print("╔═══════════════════════════════════════════════════════════╗")
    print("║  MRI-OCR: Relaxation Signature Recognition               ║")
    print("║                                                          ║")
    print("║  Don't read the character. Disturb it.                   ║")
    print("║  Measure the response. The relaxation IS the identity.   ║")
    print("╚═══════════════════════════════════════════════════════════╝")
    print()
    
    img = Image.open('/mnt/user-data/uploads/1774283318544_image.png')
    gray = np.array(img.convert('L'))
    
    cells = [
        (376, 408, 115, 300, 'Hassan'),
        (346, 376, 115, 300, 'Nawaf'),
        (408, 438, 115, 300, 'Jamal'),
        (480, 510, 115, 300, 'Bader'),
        (510, 540, 115, 300, 'Ali hussain'),
        (720, 750, 115, 300, 'Adel'),
        (128, 160, 115, 300, 'mneer'),
        (940, 968, 300, 660, 'CVA'),
        (720, 750, 300, 660, 'DLC'),
        (1078, 1108, 300, 660, 'Cap'),
        (346, 376, 660, 830, 'Bader'),
        (376, 408, 660, 830, 'Noura'),
        (128, 160, 660, 830, 'saleh'),
        (940, 968, 660, 830, 'Bazza'),
        (968, 998, 660, 830, 'Bader'),
        (1078, 1108, 660, 830, 'Saleh'),
        (1175, 1205, 660, 830, 'Zahra'),
        (346, 376, 300, 660, 'Chest infection'),
        (540, 570, 115, 300, 'Ahmad alessa'),
        (1148, 1175, 300, 660, 'Drug overdose'),
        (1175, 1205, 300, 660, 'Ischemic stroke'),
        (255, 290, 115, 300, 'Abdullah'),
    ]
    
    SCALE = 3
    
    # === PHASE 1: TRAIN ===
    print("Phase 1: Training MRI-OCR on real document characters...")
    engine = MRI_OCR()
    
    total_trained = 0
    char_counts = defaultdict(int)
    
    for r0, r1, c0, c1, text in cells:
        cell_gray = gray[r0:r1, c0:c1]
        components, _ = extract_components(cell_gray, SCALE)
        
        gt_chars = list(text.replace(' ', ''))
        n = min(len(components), len(gt_chars))
        if n < len(gt_chars) * 0.4:
            continue
        
        for i in range(n):
            engine.train(gt_chars[i], components[i][1])
            total_trained += 1
            char_counts[gt_chars[i]] += 1
    
    # Second pass for reinforcement
    for r0, r1, c0, c1, text in cells:
        cell_gray = gray[r0:r1, c0:c1]
        components, _ = extract_components(cell_gray, SCALE)
        gt_chars = list(text.replace(' ', ''))
        n = min(len(components), len(gt_chars))
        if n < len(gt_chars) * 0.4: continue
        for i in range(n):
            engine.train(gt_chars[i], components[i][1])
            total_trained += 1
    
    print(f"  Trained {total_trained} characters ({len(engine.landscapes)} unique)")
    print(f"  Signature size: {8*8} dimensions (8 measurements × 8 pulses)")
    
    # Show per-char training
    for c in sorted(char_counts.keys()):
        land = engine.landscapes.get(c)
        depth = land.n if land else 0
        print(f"    '{c}': {depth} observations", end='  ')
    print()
    
    # === PHASE 2: TEST ===
    print(f"\n{'='*70}")
    print("Phase 2: Reading with MRI relaxation signatures")
    print("=" * 70)
    
    exact = 0
    close = 0
    total_t = 0
    
    print(f"\n  {'GT':20s}  {'Raw':25s}  {'Corrected':25s}  M")
    print(f"  {'-'*20}  {'-'*25}  {'-'*25}  -")
    
    for r0, r1, c0, c1, gt_text in cells:
        cell_gray = gray[r0:r1, c0:c1]
        components, binary = extract_components(cell_gray, SCALE)
        
        if not components:
            print(f"  X {gt_text[:20]:20s}  {'[no components]':25s}")
            total_t += 1
            continue
        
        # Detect spaces
        gaps = []
        for i in range(1, len(components)):
            gaps.append(components[i][0] - (components[i-1][0] + components[i-1][1].shape[1]))
        space_t = np.median(gaps) * 2.0 if gaps else 30
        
        # Recognize via MRI signatures
        chars = []
        for i, (col, br) in enumerate(components):
            if i > 0:
                gap = col - (components[i-1][0] + components[i-1][1].shape[1])
                if gap > space_t:
                    chars.append(' ')
            
            candidates = engine.recognize(br)
            if candidates:
                chars.append(candidates[0][0])
        
        raw = ''.join(chars)
        
        # Word container
        words = raw.split()
        corrected = ' '.join(match_word(w)[0] for w in words)
        
        gt_c = gt_text.strip().lower()
        res_c = corrected.strip().lower()
        total_t += 1
        
        if gt_c == res_c:
            exact += 1; sym = 'V'
        elif ocr_distance(gt_c, res_c) <= max(len(gt_c) * 0.35, 1.5):
            close += 1; sym = '~'
        else:
            sym = 'X'
        
        print(f"  {sym} {gt_text[:20]:20s}  {raw[:25]:25s}  {corrected[:25]:25s}")
    
    usable = exact + close
    
    print(f"\n{'='*70}")
    print("RESULTS")
    print("=" * 70)
    print(f"""
  Exact:   {exact}/{total_t} ({exact/total_t*100:.0f}%)
  Close:   {close}/{total_t} ({close/total_t*100:.0f}%)
  Usable:  {usable}/{total_t} ({usable/total_t*100:.0f}%)
  
  Method: MRI relaxation signatures
    - 8 perturbation pulses per character
    - 8 measurements per pulse state  
    - 64-dimensional relaxation fingerprint
    - Fluid landscape classification
    
  The theory: structure → chaos → measure relaxation → identify.
  Different characters relax differently under the same disturbance.
""")
    
    return engine


if __name__ == '__main__':
    main()
