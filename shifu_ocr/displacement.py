"""
Medium Displacement OCR (MD-OCR)
================================
A novel approach to character recognition based on the principle:
"Don't model the ink — model the medium. Detect displacement."

Instead of learning what characters look like (discriminative/template approach),
this system:
1. Models the background (the medium/paper) as a smooth, predictable field
2. Detects perturbations — where the medium is displaced
3. Extracts a "Medium Displacement Signature" (MDS) for each perturbation
4. Classifies characters by their proportional effect on the medium

The MDS captures topological and proportional invariants:
- Euler number (connected components minus holes)
- Number of counter-spaces (enclosed background regions)  
- Proportional displacement in a normalized spatial grid
- Boundary contact pattern (how background flows into/out of bounding box)
- Displacement ratio (how much of the local medium is perturbed)

These features are invariant across fonts, sizes, and styles because they describe
what the character DOES to the medium, not what the character LOOKS like.

Author: Bader & Claude — March 2026
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage
from skimage import morphology, measure, filters
try:
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
except ImportError:
    plt = None
    gridspec = None
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# PHASE 1: BACKGROUND MODELING — "Know the medium"
# =============================================================================

def estimate_background(image_array, kernel_size=25):
    """
    Estimate what the background (paper/medium) looks like WITHOUT any ink.
    
    The idea: the background is smooth and slowly varying. Ink is abrupt and 
    high-frequency. A large morphological closing "fills in" the ink regions 
    with the surrounding background intensity, giving us the expected medium.
    
    This is the foundation: we learn the MEDIUM, not the signal.
    """
    # Morphological closing with a large disk fills in dark (ink) regions
    # with surrounding light (paper) values
    selem = morphology.disk(kernel_size)
    background = morphology.closing(image_array, selem)
    
    # Additional smoothing to ensure the background model is truly smooth
    background = filters.gaussian(background, sigma=kernel_size/2)
    
    return background


def compute_displacement_field(image_array, background):
    """
    The displacement field: where does reality deviate from expectation?
    
    displacement = expected_medium - observed_reality
    
    Positive values = something DISPLACED the medium here (ink)
    Near-zero values = medium is undisturbed (paper)
    
    This is the prediction error — the surprise — the perturbation.
    """
    displacement = background.astype(float) - image_array.astype(float)
    
    # Normalize to [0, 1]
    d_min, d_max = displacement.min(), displacement.max()
    if d_max > d_min:
        displacement = (displacement - d_min) / (d_max - d_min)
    else:
        displacement = np.zeros_like(displacement)

    return displacement


def detect_perturbations(displacement_field, threshold=0.3):
    """
    Threshold the displacement field to find perturbed regions.
    
    We don't need a complex binarization scheme — we're not asking 
    "is this ink or paper?" We're asking "is the medium significantly 
    displaced here?" A simple threshold on the displacement magnitude suffices
    because we've already normalized against the local background.
    """
    binary = displacement_field > threshold
    
    # Clean up: remove tiny perturbations (noise) via morphological opening
    binary = morphology.remove_small_objects(binary, min_size=10)
    
    return binary.astype(np.uint8)


# =============================================================================
# PHASE 2: MEDIUM DISPLACEMENT SIGNATURE — "Measure the effect"
# =============================================================================

def compute_euler_number(binary_region):
    """
    Euler number = connected_components - holes
    
    This is a TOPOLOGICAL invariant. No matter how you draw "A", 
    it always has Euler number 0 (1 component - 1 hole).
    "B" always has -1 (1 component - 2 holes).
    "C" always has 1 (1 component - 0 holes).
    
    This doesn't depend on font, size, or style — it depends on 
    what the character DOES to the medium's topology.
    """
    # Pad to ensure background connectivity
    padded = np.pad(binary_region, 1, mode='constant', constant_values=0)
    
    # Label connected foreground components
    labeled_fg, num_fg = ndimage.label(padded)
    
    # Label connected background components (holes + exterior)
    bg = 1 - padded
    labeled_bg, num_bg = ndimage.label(bg)
    
    # Number of holes = background components - 1 (subtract exterior)
    num_holes = num_bg - 1
    
    return num_fg, num_holes, num_fg - num_holes


def compute_regional_displacement(binary_region, grid_size=3):
    """
    Divide the bounding box into a grid and measure PROPORTIONAL displacement
    in each cell.
    
    This captures the spatial distribution of the perturbation.
    "T" displaces heavily at the top (crossbar) and center-bottom (stem).
    "O" displaces the periphery but not the center.
    "L" displaces the left and bottom.
    
    These proportions are INVARIANT across fonts because they reflect 
    the character's structural identity, not its appearance.
    """
    h, w = binary_region.shape
    if h == 0 or w == 0:
        return np.zeros((grid_size, grid_size))
    
    cell_h = max(1, h // grid_size)
    cell_w = max(1, w // grid_size)
    
    grid = np.zeros((grid_size, grid_size))
    
    for i in range(grid_size):
        for j in range(grid_size):
            r_start = i * cell_h
            r_end = min((i + 1) * cell_h, h)
            c_start = j * cell_w
            c_end = min((j + 1) * cell_w, w)
            
            cell = binary_region[r_start:r_end, c_start:c_end]
            if cell.size > 0:
                grid[i, j] = np.mean(cell)
    
    # Normalize: proportional displacement (sums to 1)
    total = grid.sum()
    if total > 0:
        grid = grid / total
    
    return grid


def compute_boundary_contact(binary_region):
    """
    How does the background (undisplaced medium) connect to the exterior 
    of the character's bounding box?
    
    We check each edge (top, bottom, left, right) for where background 
    "flows into" the character space. This is the character's INTERFACE 
    with the surrounding medium.
    
    "C" has a large opening on the right — background flows in from the right.
    "O" has no openings — background doesn't flow in at all (from the character's interior perspective).
    "U" has an opening at the top.
    
    This is invariant because the opening pattern IS the character's identity.
    """
    h, w = binary_region.shape
    if h < 2 or w < 2:
        return np.zeros(4)
    
    # Proportion of each edge that is background (not displaced)
    top_contact = 1.0 - np.mean(binary_region[0, :])
    bottom_contact = 1.0 - np.mean(binary_region[-1, :])
    left_contact = 1.0 - np.mean(binary_region[:, 0])
    right_contact = 1.0 - np.mean(binary_region[:, -1])
    
    return np.array([top_contact, right_contact, bottom_contact, left_contact])


def compute_displacement_ratio(binary_region):
    """
    What fraction of the bounding box is displaced?
    
    "I" displaces very little of its bounding box.
    "W" displaces a lot.
    "O" displaces a moderate amount.
    
    This ratio is proportionally stable across fonts.
    """
    if binary_region.size == 0:
        return 0.0
    return np.mean(binary_region)


def compute_aspect_ratio(binary_region):
    """
    The aspect ratio of the perturbation envelope.
    
    "I" is tall and narrow.
    "W" is wide and short.
    "O" is roughly square.
    """
    h, w = binary_region.shape
    if h == 0:
        return 0.0
    return w / h


def compute_vertical_symmetry(binary_region):
    """
    How symmetric is the displacement pattern about the vertical axis?
    
    "A", "T", "O", "V", "W", "M" are highly symmetric.
    "J", "P", "F" are asymmetric.
    
    This is structural — it reflects the character's geometry, not its rendering.
    """
    h, w = binary_region.shape
    if w < 2:
        return 1.0
    
    # Flip horizontally and compare
    flipped = np.fliplr(binary_region)
    
    # Overlap ratio
    agreement = np.mean(binary_region == flipped)
    return agreement


def compute_horizontal_symmetry(binary_region):
    """
    Symmetry about the horizontal axis.
    
    "B", "D", "O", "X", "H" are fairly symmetric.
    "P", "b", "L" are asymmetric.
    """
    h, w = binary_region.shape
    if h < 2:
        return 1.0
    
    flipped = np.flipud(binary_region)
    agreement = np.mean(binary_region == flipped)
    return agreement


def compute_vertical_center_of_mass(binary_region):
    """
    Where is the "weight" of the displacement concentrated vertically?
    
    Characters with top-heavy displacement (T, P, F) differ from 
    bottom-heavy (L, J) and centered (O, H, X).
    Normalized to [0, 1] where 0 = top, 1 = bottom.
    """
    h, w = binary_region.shape
    if h == 0 or binary_region.sum() == 0:
        return 0.5
    
    rows = np.arange(h).reshape(-1, 1)
    row_weights = binary_region * rows
    return row_weights.sum() / (binary_region.sum() * h)


def compute_horizontal_center_of_mass(binary_region):
    """Horizontal center of mass. 0 = left, 1 = right."""
    h, w = binary_region.shape
    if w == 0 or binary_region.sum() == 0:
        return 0.5
    
    cols = np.arange(w).reshape(1, -1)
    col_weights = binary_region * cols
    return col_weights.sum() / (binary_region.sum() * w)


def extract_medium_displacement_signature(binary_region):
    """
    The complete Medium Displacement Signature (MDS).
    
    This is the character's fingerprint — not what it LOOKS like,
    but what it DOES to the medium. The proportional effect.
    
    A strong prior: if you know the medium, the displacement signature 
    tells you the character. No need for a million training examples.
    """
    num_components, num_holes, euler = compute_euler_number(binary_region)
    regional = compute_regional_displacement(binary_region, grid_size=4)
    boundary = compute_boundary_contact(binary_region)
    displacement_ratio = compute_displacement_ratio(binary_region)
    aspect = compute_aspect_ratio(binary_region)
    v_sym = compute_vertical_symmetry(binary_region)
    h_sym = compute_horizontal_symmetry(binary_region)
    v_com = compute_vertical_center_of_mass(binary_region)
    h_com = compute_horizontal_center_of_mass(binary_region)
    
    # Build the signature vector
    signature = {
        'euler_number': euler,
        'num_holes': num_holes,
        'num_components': num_components,
        'regional_displacement': regional.flatten(),
        'boundary_contact': boundary,
        'displacement_ratio': displacement_ratio,
        'aspect_ratio': aspect,
        'vertical_symmetry': v_sym,
        'horizontal_symmetry': h_sym,
        'vertical_center_of_mass': v_com,
        'horizontal_center_of_mass': h_com,
    }
    
    # Also build a flat feature vector for distance computation
    feature_vector = np.concatenate([
        [euler],
        [num_holes],
        [num_components],
        regional.flatten(),
        boundary,
        [displacement_ratio],
        [aspect],
        [v_sym],
        [h_sym],
        [v_com],
        [h_com],
    ])
    
    signature['feature_vector'] = feature_vector
    
    return signature


# =============================================================================
# PHASE 3: CLASSIFICATION — "Predict from few examples"
# =============================================================================

def signature_distance(sig1, sig2, weights=None):
    """
    Distance between two MDS vectors.
    
    Topology features (Euler number, holes) are weighted heavily because 
    they're the most invariant. Proportional features get moderate weight.
    
    This is where the "strong prior" lives — we KNOW that topology matters 
    more than exact proportions, so we encode that knowledge in the weights.
    """
    v1 = sig1['feature_vector']
    v2 = sig2['feature_vector']
    
    if weights is None:
        # Default weights: topology > proportion > geometry
        # [euler, holes, components, 16x regional, 4x boundary, ratio, aspect, vsym, hsym, vcom, hcom]
        weights = np.concatenate([
            [5.0],        # euler - TOPOLOGICAL (most invariant)
            [5.0],        # holes - TOPOLOGICAL
            [3.0],        # components - TOPOLOGICAL
            [2.0] * 16,   # regional displacement - PROPORTIONAL
            [2.5] * 4,    # boundary contact - STRUCTURAL
            [2.0],        # displacement ratio - PROPORTIONAL
            [1.5],        # aspect ratio - GEOMETRIC (least invariant)
            [1.5],        # vertical symmetry
            [1.5],        # horizontal symmetry
            [1.5],        # vertical center of mass
            [1.5],        # horizontal center of mass
        ])
    
    diff = (v1 - v2) * weights
    return np.sqrt(np.sum(diff ** 2))


def classify_character(unknown_sig, known_signatures, top_k=1):
    """
    Classify by nearest MDS in the signature space.
    
    This is the "cheap radar" — we don't need millions of examples.
    We need a few examples per character, compute their MDS, and then 
    classify new characters by which known MDS they're closest to.
    
    The strong prior (topological + proportional features) means we need 
    very few examples because the feature space is already highly structured.
    """
    distances = []
    for label, sigs in known_signatures.items():
        for sig in sigs:
            d = signature_distance(unknown_sig, sig)
            distances.append((d, label))
    
    distances.sort(key=lambda x: x[0])
    
    if top_k == 1:
        return distances[0][1], distances[0][0]
    
    # Majority vote from top_k
    top = distances[:top_k]
    votes = defaultdict(int)
    for d, label in top:
        votes[label] += 1
    
    winner = max(votes, key=votes.get)
    best_dist = min(d for d, l in top if l == winner)
    return winner, best_dist


# =============================================================================
# IMAGE GENERATION — Test harness
# =============================================================================

def render_character(char, font_path, font_size=80, image_size=(100, 100)):
    """Render a single character as a grayscale image."""
    img = Image.new('L', image_size, color=255)
    draw = ImageDraw.Draw(img)
    
    try:
        font = ImageFont.truetype(font_path, font_size)
    except Exception:
        font = ImageFont.load_default()
    
    # Get text bounding box for centering
    bbox = draw.textbbox((0, 0), char, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    
    x = (image_size[0] - text_w) // 2 - bbox[0]
    y = (image_size[1] - text_h) // 2 - bbox[1]
    
    draw.text((x, y), char, fill=0, font=font)
    
    return np.array(img)


def get_character_region(binary_image):
    """Extract the tight bounding box around the character."""
    coords = np.argwhere(binary_image > 0)
    if len(coords) == 0:
        return binary_image
    
    r_min, c_min = coords.min(axis=0)
    r_max, c_max = coords.max(axis=0)
    
    # Add small padding
    pad = 2
    r_min = max(0, r_min - pad)
    c_min = max(0, c_min - pad)
    r_max = min(binary_image.shape[0] - 1, r_max + pad)
    c_max = min(binary_image.shape[1] - 1, c_max + pad)
    
    return binary_image[r_min:r_max+1, c_min:c_max+1]


def normalize_region(region, target_size=(64, 64)):
    """Normalize region to a standard size for fair comparison."""
    from PIL import Image as PILImage
    img = PILImage.fromarray((region * 255).astype(np.uint8))
    img = img.resize(target_size, PILImage.NEAREST)
    return (np.array(img) > 127).astype(np.uint8)


def process_character_image(char_image):
    """
    Full pipeline: image -> background model -> displacement -> MDS
    """
    # Step 1: Model the medium
    background = estimate_background(char_image, kernel_size=15)
    
    # Step 2: Compute displacement field
    displacement = compute_displacement_field(char_image, background)
    
    # Step 3: Detect perturbations
    binary = detect_perturbations(displacement, threshold=0.25)
    
    # Step 4: Extract character region and normalize
    region = get_character_region(binary)
    region_norm = normalize_region(region)
    
    # Step 5: Compute MDS
    signature = extract_medium_displacement_signature(region_norm)
    
    return {
        'original': char_image,
        'background': background,
        'displacement': displacement,
        'binary': binary,
        'region': region_norm,
        'signature': signature,
    }


# =============================================================================
# EXPERIMENT 1: Signature invariance across fonts
# =============================================================================

def run_invariance_experiment():
    """
    CORE TEST: Same character across different fonts should produce 
    similar MDS. Different characters should produce different MDS.
    
    This tests the fundamental claim: the proportional effect on the 
    medium is invariant — it's the character's identity, not appearance.
    """
    
    fonts = [
        ('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 'DejaVu Sans'),
        ('/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf', 'DejaVu Serif'),
        ('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 'DejaVu Bold'),
        ('/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf', 'Serif Bold'),
        ('/usr/share/fonts/truetype/freefont/FreeMono.ttf', 'FreeMono'),
        ('/usr/share/fonts/truetype/google-fonts/Poppins-Bold.ttf', 'Poppins'),
    ]
    
    test_chars = ['A', 'B', 'C', 'D', 'O', 'T', 'I', 'L', 'H', 'P']
    
    print("=" * 70)
    print("EXPERIMENT 1: Medium Displacement Signature Invariance")
    print("=" * 70)
    print()
    print("Testing: Do same characters produce similar MDS across fonts?")
    print("         Do different characters produce different MDS?")
    print()
    
    # Process all characters in all fonts
    all_results = {}
    for char in test_chars:
        all_results[char] = []
        for font_path, font_name in fonts:
            img = render_character(char, font_path, font_size=70, image_size=(100, 100))
            result = process_character_image(img)
            result['font_name'] = font_name
            result['char'] = char
            all_results[char].append(result)
    
    # Compute intra-class distances (same char, different fonts)
    print("INTRA-CLASS DISTANCES (same character, different fonts):")
    print("-" * 55)
    intra_distances = {}
    for char in test_chars:
        sigs = [r['signature'] for r in all_results[char]]
        dists = []
        for i in range(len(sigs)):
            for j in range(i+1, len(sigs)):
                d = signature_distance(sigs[i], sigs[j])
                dists.append(d)
        mean_d = np.mean(dists) if dists else 0
        std_d = np.std(dists) if dists else 0
        intra_distances[char] = (mean_d, std_d)
        print(f"  '{char}': mean distance = {mean_d:.3f} ± {std_d:.3f}")
    
    print()
    
    # Compute inter-class distances (different chars)
    print("INTER-CLASS DISTANCES (different characters):")
    print("-" * 55)
    inter_distances = []
    for i, c1 in enumerate(test_chars):
        for c2 in test_chars[i+1:]:
            sig1 = all_results[c1][0]['signature']  # Use first font
            sig2 = all_results[c2][0]['signature']
            d = signature_distance(sig1, sig2)
            inter_distances.append(d)
            if d < 3.0:  # Flag potentially confusable pairs
                print(f"  '{c1}' vs '{c2}': distance = {d:.3f}  ← close pair!")
    
    mean_inter = np.mean(inter_distances)
    mean_intra = np.mean([v[0] for v in intra_distances.values()])
    
    print()
    print(f"  Average inter-class distance: {mean_inter:.3f}")
    print(f"  Average intra-class distance: {mean_intra:.3f}")
    print(f"  Separation ratio (inter/intra): {mean_inter/mean_intra:.2f}x")
    print()
    
    if mean_inter / mean_intra > 2.0:
        print("  ✓ STRONG SEPARATION — The medium displacement signature")
        print("    is more similar within characters than between characters.")
        print("    The proportional effect IS the invariant identity.")
    else:
        print("  ~ MODERATE SEPARATION — Some overlap between classes.")
    
    return all_results, intra_distances


# =============================================================================
# EXPERIMENT 2: Few-shot classification
# =============================================================================

def run_classification_experiment():
    """
    THE CHEAP RADAR TEST: Train on ONE font, test on all others.
    
    If the MDS captures true invariants, we should be able to classify 
    characters across fonts with just a single example per character.
    This is the "strong prior, little data" principle in action.
    """
    
    train_font = ('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 'DejaVu Sans')
    
    test_fonts = [
        ('/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf', 'DejaVu Serif'),
        ('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 'DejaVu Bold'),
        ('/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf', 'Serif Bold'),
        ('/usr/share/fonts/truetype/freefont/FreeMono.ttf', 'FreeMono'),
        ('/usr/share/fonts/truetype/google-fonts/Poppins-Bold.ttf', 'Poppins'),
    ]
    
    test_chars = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
    
    print()
    print("=" * 70)
    print("EXPERIMENT 2: One-Shot Classification")  
    print("=" * 70)
    print()
    print(f"Training: 1 example per character ({train_font[1]})")
    print(f"Testing:  {len(test_fonts)} unseen fonts × {len(test_chars)} characters")
    print()
    
    # Build signature database from ONE font (one example per character)
    known_signatures = {}
    for char in test_chars:
        img = render_character(char, train_font[0], font_size=70, image_size=(100, 100))
        result = process_character_image(img)
        known_signatures[char] = [result['signature']]
    
    # Test on all other fonts
    correct = 0
    total = 0
    errors = []
    
    font_results = {}
    for font_path, font_name in test_fonts:
        font_correct = 0
        font_total = 0
        for char in test_chars:
            img = render_character(char, font_path, font_size=70, image_size=(100, 100))
            result = process_character_image(img)
            
            predicted, dist = classify_character(result['signature'], known_signatures)
            
            if predicted == char:
                correct += 1
                font_correct += 1
            else:
                errors.append((char, predicted, font_name, dist))
            
            total += 1
            font_total += 1
        
        accuracy = font_correct / font_total * 100
        font_results[font_name] = accuracy
        print(f"  {font_name:20s}: {accuracy:5.1f}% ({font_correct}/{font_total})")
    
    overall = correct / total * 100
    print(f"\n  {'OVERALL':20s}: {overall:5.1f}% ({correct}/{total})")
    print()
    
    if errors:
        print(f"Misclassifications ({len(errors)}):")
        print("-" * 55)
        for true_char, pred_char, font_name, dist in errors[:15]:
            print(f"  '{true_char}' → '{pred_char}'  ({font_name}, dist={dist:.2f})")
    
    return overall, font_results, errors


# =============================================================================
# EXPERIMENT 3: Topological signature analysis
# =============================================================================

def run_topology_experiment():
    """
    Show that TOPOLOGY alone carries significant discriminative power.
    
    Characters can be grouped by their topological displacement signature:
    - 0 holes: C, E, F, G, I, J, K, L, M, N, S, T, U, V, W, X, Y, Z
    - 1 hole:  A, D, O, P, Q, R
    - 2 holes: B
    
    Within each topological class, proportional features discriminate further.
    """
    
    font_path = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
    test_chars = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
    
    print()
    print("=" * 70)
    print("EXPERIMENT 3: Topological Classification of the Alphabet")
    print("=" * 70)
    print()
    print("Grouping characters by their TOPOLOGICAL effect on the medium:")
    print("(Euler number = components - holes)")
    print()
    
    topology_groups = defaultdict(list)
    char_topology = {}
    
    for char in test_chars:
        img = render_character(char, font_path, font_size=70, image_size=(100, 100))
        result = process_character_image(img)
        sig = result['signature']
        
        euler = sig['euler_number']
        holes = sig['num_holes']
        topology_groups[holes].append(char)
        char_topology[char] = {
            'euler': euler,
            'holes': holes,
            'displacement_ratio': sig['displacement_ratio'],
            'aspect_ratio': sig['aspect_ratio'],
            'v_sym': sig['vertical_symmetry'],
        }
    
    for n_holes in sorted(topology_groups.keys()):
        chars = topology_groups[n_holes]
        print(f"  {n_holes} hole(s): {', '.join(chars)}")
    
    print()
    print("Character displacement profiles:")
    print("-" * 65)
    print(f"  {'Char':>4} | {'Euler':>5} | {'Holes':>5} | {'Displ%':>6} | {'Aspect':>6} | {'V-Sym':>5}")
    print(f"  {'-'*4:>4} | {'-'*5:>5} | {'-'*5:>5} | {'-'*6:>6} | {'-'*6:>6} | {'-'*5:>5}")
    
    for char in test_chars:
        t = char_topology[char]
        print(f"  {char:>4} | {t['euler']:>5} | {t['holes']:>5} | {t['displacement_ratio']:>5.1%} | {t['aspect_ratio']:>6.2f} | {t['v_sym']:>5.2f}")
    
    return topology_groups, char_topology


# =============================================================================
# VISUALIZATION
# =============================================================================

def create_pipeline_visualization(all_results):
    """
    Visualize the full MD-OCR pipeline for selected characters across fonts.
    """
    
    # Select a few characters for detailed visualization
    show_chars = ['A', 'B', 'O', 'T', 'I']
    show_chars = [c for c in show_chars if c in all_results]
    
    n_chars = len(show_chars)
    n_fonts = min(4, len(all_results[show_chars[0]]))
    
    fig = plt.figure(figsize=(20, n_chars * 4 + 2))
    fig.suptitle('Medium Displacement OCR — Pipeline Visualization\n'
                 '"Don\'t model the ink. Model the medium. Detect displacement."',
                 fontsize=14, fontweight='bold', y=0.98)
    
    gs = gridspec.GridSpec(n_chars, n_fonts * 4 + 1, hspace=0.4, wspace=0.3,
                           top=0.93, bottom=0.02)
    
    col_labels = ['Original', 'Background\n(Medium Model)', 'Displacement\nField', 'Perturbation\nDetected']
    
    for i, char in enumerate(show_chars):
        for j in range(n_fonts):
            result = all_results[char][j]
            
            for k, (key, cmap, title) in enumerate([
                ('original', 'gray', col_labels[0]),
                ('background', 'gray', col_labels[1]),
                ('displacement', 'hot', col_labels[2]),
                ('binary', 'gray_r', col_labels[3]),
            ]):
                ax = fig.add_subplot(gs[i, j * 4 + k])
                
                data = result[key]
                if key == 'background':
                    data = (data * 255).astype(np.uint8) if data.max() <= 1 else data
                
                ax.imshow(data, cmap=cmap, vmin=0)
                ax.axis('off')
                
                if i == 0 and j == 0:
                    ax.set_title(title, fontsize=8, fontweight='bold')
                
                if k == 0:
                    font_name = result.get('font_name', f'Font {j}')
                    ax.set_ylabel(f"'{char}'\n{font_name}", fontsize=8, rotation=0, 
                                 labelpad=50, va='center')
    
    return fig


def create_signature_comparison_plot(all_results):
    """
    Visualize MDS feature vectors showing intra-class similarity 
    and inter-class difference.
    """
    
    show_chars = ['A', 'B', 'O', 'T', 'I', 'H', 'C', 'L']
    show_chars = [c for c in show_chars if c in all_results]
    
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle('Medium Displacement Signatures — Same Character, Different Fonts\n'
                 '"The proportional effect is the invariant"',
                 fontsize=14, fontweight='bold')
    
    for idx, char in enumerate(show_chars):
        ax = axes[idx // 4][idx % 4]
        
        results = all_results[char]
        for r in results:
            sig = r['signature']
            regional = sig['regional_displacement']
            font_name = r.get('font_name', 'Unknown')
            ax.plot(regional, label=font_name, alpha=0.7, linewidth=2)
        
        ax.set_title(f"'{char}' — Regional Displacement", fontweight='bold')
        ax.set_xlabel('Grid cell')
        ax.set_ylabel('Proportional displacement')
        ax.legend(fontsize=6)
        ax.set_ylim(0, 0.25)
    
    plt.tight_layout()
    return fig


def create_topology_visualization(char_topology):
    """
    Scatter plot showing how topology + proportion separates characters.
    """
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    fig.suptitle('Character Identity from Medium Displacement\n'
                 'Topology (holes) × Proportional displacement × Symmetry',
                 fontsize=14, fontweight='bold')
    
    colors = {0: '#2ecc71', 1: '#3498db', 2: '#e74c3c'}
    
    for char, t in char_topology.items():
        holes = t['holes']
        color = colors.get(holes, '#95a5a6')
        
        ax.scatter(t['displacement_ratio'], t['v_sym'], 
                   c=color, s=200, alpha=0.8, edgecolors='black', linewidth=1)
        ax.annotate(char, (t['displacement_ratio'], t['v_sym']),
                   fontsize=12, fontweight='bold',
                   ha='center', va='center')
    
    # Legend
    for n_holes, color in colors.items():
        ax.scatter([], [], c=color, s=150, label=f'{n_holes} holes', edgecolors='black')
    
    ax.set_xlabel('Displacement Ratio (how much medium is perturbed)', fontsize=12)
    ax.set_ylabel('Vertical Symmetry', fontsize=12)
    ax.legend(fontsize=11, title='Topological class')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


# =============================================================================
# MAIN
# =============================================================================

def main():
    print()
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║         MEDIUM DISPLACEMENT OCR (MD-OCR) — Prototype           ║")
    print("║                                                                ║")
    print("║  'Don't model the ink. Model the medium. Detect displacement.  ║")
    print("║   Calculate the effect, and you will know the letter.'         ║")
    print("╚══════════════════════════════════════════════════════════════════╝")
    print()
    
    # Run experiments
    all_results, intra_distances = run_invariance_experiment()
    overall_acc, font_results, errors = run_classification_experiment()
    topology_groups, char_topology = run_topology_experiment()
    
    # Generate visualizations
    print()
    print("Generating visualizations...")
    
    fig1 = create_pipeline_visualization(all_results)
    fig1.savefig('/home/claude/pipeline_visualization.png', dpi=150, bbox_inches='tight')
    plt.close(fig1)
    
    fig2 = create_signature_comparison_plot(all_results)
    fig2.savefig('/home/claude/signature_comparison.png', dpi=150, bbox_inches='tight')
    plt.close(fig2)
    
    fig3 = create_topology_visualization(char_topology)
    fig3.savefig('/home/claude/topology_map.png', dpi=150, bbox_inches='tight')
    plt.close(fig3)
    
    # Summary
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()
    print(f"  One-shot classification accuracy: {overall_acc:.1f}%")
    print(f"  Training data used: 26 images (one per character, one font)")
    print(f"  No neural network. No GPU. No pretrained weights.")
    print(f"  Just: model the medium → detect displacement → measure the effect.")
    print()
    print("  The strong prior (topological + proportional invariants)")
    print("  substitutes for the large dataset. Understanding > data.")
    print()
    
    return overall_acc


if __name__ == '__main__':
    main()
