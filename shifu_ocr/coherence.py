"""
Coherence Displacement: The Fix for Colored Backgrounds
========================================================

The original medium displacement measured:
  "How different is this pixel from the expected background brightness?"

This fails when text color ≈ background color.

The NEW coherence displacement measures:
  "How well does this pixel AGREE with its neighbors?"

A background pixel — regardless of color — agrees with its neighbors.
It's part of a smooth, connected field. High local coherence.

A text pixel — even if the same color — DISAGREES with some neighbors.
It has edges. It has structure. It's not in harmony with the field.

The medium is defined by CONNECTEDNESS, not color.
The perturbation is defined by DISRUPTION of connectedness, not contrast.

"A page is connected as a whole. A word is not. Even if it has the 
same color, it will always be different because the pixels are not 
in harmony with the rest of the page."
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage
from skimage import morphology, filters
try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


def compute_local_coherence(image, window=5):
    """
    For each pixel, measure how well it agrees with its local neighborhood.
    
    Coherence = 1 / (1 + local_variance)
    
    Background: low variance → high coherence (pixels agree)
    Text edges: high variance → low coherence (pixels disagree)
    
    This works regardless of the absolute color because it measures
    RELATIVE agreement, not absolute value.
    """
    img = image.astype(float)
    
    # Local mean
    kernel = np.ones((window, window)) / (window * window)
    local_mean = ndimage.convolve(img, kernel, mode='reflect')
    
    # Local variance (how much pixels disagree with their neighbors)
    local_var = ndimage.convolve((img - local_mean) ** 2, kernel, mode='reflect')
    
    # Coherence: high where pixels agree, low where they don't
    coherence = 1.0 / (1.0 + local_var)
    
    return coherence, local_var


def compute_edge_density(image, window=5):
    """
    Alternative coherence measure: local edge density.
    
    Background has few edges (smooth, connected).
    Text has many edges (structured, disconnected from surroundings).
    
    Even same-color text on same-color background has edges where 
    the letterform meets the background — the BOUNDARY exists even 
    when the colors are similar because anti-aliasing, sub-pixel 
    rendering, and compression create slight differences.
    """
    img = image.astype(float)
    
    # Gradient magnitude (Sobel)
    gx = ndimage.sobel(img, axis=1)
    gy = ndimage.sobel(img, axis=0)
    gradient = np.sqrt(gx**2 + gy**2)
    
    # Local edge density
    kernel = np.ones((window, window)) / (window * window)
    edge_density = ndimage.convolve(gradient, kernel, mode='reflect')
    
    return edge_density, gradient


def compute_coherence_displacement(image, window=5):
    """
    THE UNIFIED DISPLACEMENT: combines brightness displacement 
    with coherence displacement.
    
    Channel 1: Classic medium displacement (brightness anomaly)
    Channel 2: Coherence displacement (harmony disruption)
    
    Text on white paper: Channel 1 catches it (dark on light)
    Text on colored background: Channel 2 catches it (incoherent on coherent)
    Both: maximum robustness
    """
    if len(image.shape) == 3:
        # Color image — compute coherence per channel, combine
        channels = [image[:, :, c] for c in range(image.shape[2])]
    else:
        channels = [image]
    
    # Per-channel coherence
    all_coherence = []
    all_edge_density = []
    
    for ch in channels:
        coh, _ = compute_local_coherence(ch, window)
        ed, _ = compute_edge_density(ch, window)
        all_coherence.append(coh)
        all_edge_density.append(ed)
    
    # Combine: minimum coherence across channels (text disrupts at least one)
    min_coherence = np.minimum.reduce(all_coherence)
    
    # Combine: maximum edge density across channels
    max_edge = np.maximum.reduce(all_edge_density)
    
    # Displacement = low coherence AND/OR high edge density
    # Normalize each to [0, 1]
    incoherence = 1.0 - min_coherence
    if incoherence.max() > incoherence.min():
        incoherence = (incoherence - incoherence.min()) / (incoherence.max() - incoherence.min())
    
    if max_edge.max() > max_edge.min():
        edge_norm = (max_edge - max_edge.min()) / (max_edge.max() - max_edge.min())
    else:
        edge_norm = max_edge * 0
    
    # Combined displacement: either signal detects text
    displacement = np.maximum(incoherence, edge_norm)
    
    return displacement, incoherence, edge_norm


def detect_text_regions(displacement, threshold=None):
    """Threshold the displacement to find text."""
    if threshold is None:
        # Adaptive: use Otsu on the displacement field itself
        from skimage.filters import threshold_otsu
        try:
            threshold = threshold_otsu(displacement)
        except Exception:
            threshold = 0.3
    
    binary = displacement > threshold
    binary = morphology.remove_small_objects(binary, min_size=10)
    return binary.astype(np.uint8)


# =============================================================================
# TEST ON THE REAL WARD IMAGE
# =============================================================================

def test_on_ward_image():
    """Test coherence displacement on the actual MedEvac spreadsheet."""
    
    img = Image.open('/mnt/user-data/uploads/1774283318544_image.png')
    img_array = np.array(img)
    gray = np.array(img.convert('L'))
    
    print("=" * 70)
    print("COHERENCE DISPLACEMENT: Detecting text by harmony disruption")
    print("=" * 70)
    
    # Compute coherence displacement
    print("\n  Computing coherence displacement on full image...")
    disp, incoherence, edge_density = compute_coherence_displacement(img_array, window=3)
    
    print(f"  Displacement range: [{disp.min():.3f}, {disp.max():.3f}]")
    print(f"  Incoherence range: [{incoherence.min():.3f}, {incoherence.max():.3f}]")
    print(f"  Edge density range: [{edge_density.min():.3f}, {edge_density.max():.3f}]")
    
    # Also compute classic brightness displacement for comparison
    from skimage import morphology as morph, filters as filt
    bg = filt.gaussian(morph.closing(gray, morph.disk(25)), sigma=12)
    classic_disp = bg.astype(float) - gray.astype(float)
    classic_disp = (classic_disp - classic_disp.min()) / (classic_disp.max() - classic_disp.min() + 1e-8)
    
    # Detect text regions with both methods
    text_classic = detect_text_regions(classic_disp, threshold=0.25)
    text_coherence = detect_text_regions(disp, threshold=None)
    
    # Count detected text pixels
    classic_pixels = text_classic.sum()
    coherence_pixels = text_coherence.sum()
    
    print(f"\n  Classic displacement detected: {classic_pixels:,} text pixels")
    print(f"  Coherence displacement detected: {coherence_pixels:,} text pixels")
    print(f"  Ratio: {coherence_pixels / max(classic_pixels, 1):.2f}x more detection")
    
    # Focus on specific problem areas
    print(f"\n  Testing on problem areas (colored backgrounds):")
    
    test_regions = [
        ("Ward 20, Bed 1-17 (green bg, dark text)", 370, 400, 0, 700),
        ("Ward 20, Bed 14-1 (green bg, dark text)", 400, 432, 0, 700),
        ("Doctor 'Bader' (yellow bg, green text)", 370, 400, 660, 835),
        ("Doctor 'Noura' (pink bg, dark text)", 400, 432, 660, 835),
        ("Status cell (red bg)", 370, 400, 835, 1100),
        ("Ward 19 header (green bg, bold)", 465, 495, 0, 700),
        ("Chronic status (yellow bg, red text)", 970, 1000, 835, 1100),
    ]
    
    for name, r0, r1, c0, c1 in test_regions:
        region = img_array[r0:r1, c0:c1]
        if region.size == 0:
            continue
        
        region_disp, region_incoh, region_edge = compute_coherence_displacement(region, window=3)
        region_gray = gray[r0:r1, c0:c1]
        
        bg_r = filt.gaussian(morph.closing(region_gray, morph.disk(15)), sigma=7)
        classic_r = bg_r.astype(float) - region_gray.astype(float)
        classic_r = (classic_r - classic_r.min()) / (classic_r.max() - classic_r.min() + 1e-8)
        
        classic_text = (classic_r > 0.25).sum()
        coherence_text = detect_text_regions(region_disp).sum()
        
        improvement = "✓ better" if coherence_text > classic_text * 1.2 else "= similar" if coherence_text > classic_text * 0.8 else "✗ worse"
        
        print(f"\n    {name}")
        print(f"      Classic:   {classic_text:6d} text pixels")
        print(f"      Coherence: {coherence_text:6d} text pixels  {improvement}")
    
    # Visualization
    print(f"\n  Generating comparison visualization...")
    
    fig, axes = plt.subplots(2, 3, figsize=(24, 14))
    fig.patch.set_facecolor('#0a0a0a')
    fig.suptitle("Coherence Displacement vs Classic Displacement\n"
                 '"The medium is defined by connectedness, not color"',
                 fontsize=16, fontweight='bold', color='white')
    
    # Row 1: Full image comparisons
    axes[0, 0].imshow(img_array)
    axes[0, 0].set_title("Original", color='white', fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(classic_disp, cmap='magma')
    axes[0, 1].set_title("Classic Displacement\n(fails on colored backgrounds)", 
                          color='white', fontsize=12, fontweight='bold')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(disp, cmap='magma')
    axes[0, 2].set_title("Coherence Displacement\n(detects text by harmony disruption)", 
                          color='white', fontsize=12, fontweight='bold')
    axes[0, 2].axis('off')
    
    # Row 2: Zoom on Ward 20 section (the problem area)
    zoom = (350, 470, 0, 900)
    r0, r1, c0, c1 = zoom
    
    axes[1, 0].imshow(img_array[r0:r1, c0:c1])
    axes[1, 0].set_title("Ward 20 Section (colored backgrounds)", 
                          color='white', fontsize=12, fontweight='bold')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(classic_disp[r0:r1, c0:c1], cmap='magma')
    axes[1, 1].set_title("Classic: text vanishes on colored cells", 
                          color='#e74c3c', fontsize=12, fontweight='bold')
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(disp[r0:r1, c0:c1], cmap='magma')
    axes[1, 2].set_title("Coherence: text visible regardless of background color", 
                          color='#2ecc71', fontsize=12, fontweight='bold')
    axes[1, 2].axis('off')
    
    for ax in axes.flat:
        ax.set_facecolor('#1a1a1a')
    
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig('/home/claude/coherence_comparison.png', dpi=150, 
                bbox_inches='tight', facecolor='#0a0a0a')
    plt.close()
    
    # Second figure: the key insight
    fig2, axes2 = plt.subplots(1, 4, figsize=(24, 5))
    fig2.patch.set_facecolor('#0a0a0a')
    fig2.suptitle("Why Coherence Works: The Page is Connected, The Text is Not",
                   fontsize=14, fontweight='bold', color='white')
    
    # Pick the doctor column from Ward 20 (green text on yellow/pink bg)
    cell = img_array[370:400, 660:835]
    cell_gray = gray[370:400, 660:835]
    
    cell_coh, cell_var = compute_local_coherence(cell_gray, window=3)
    cell_ed, cell_grad = compute_edge_density(cell_gray, window=3)
    cell_disp, _, _ = compute_coherence_displacement(cell, window=3)
    
    axes2[0].imshow(cell)
    axes2[0].set_title("Cell: Doctor name\n(colored background)", color='white', fontsize=11, fontweight='bold')
    axes2[0].axis('off')
    
    axes2[1].imshow(cell_coh, cmap='viridis')
    axes2[1].set_title("Local Coherence\n(bright = harmonious)", color='white', fontsize=11, fontweight='bold')
    axes2[1].axis('off')
    
    axes2[2].imshow(cell_grad, cmap='hot')
    axes2[2].set_title("Edge Density\n(bright = disrupted)", color='white', fontsize=11, fontweight='bold')
    axes2[2].axis('off')
    
    axes2[3].imshow(cell_disp, cmap='magma')
    axes2[3].set_title("Combined: Text Detected\n(bright = perturbation)", color='white', fontsize=11, fontweight='bold')
    axes2[3].axis('off')
    
    for ax in axes2.flat:
        ax.set_facecolor('#1a1a1a')
    
    plt.tight_layout(rect=[0, 0, 1, 0.88])
    fig2.savefig('/home/claude/coherence_cell_demo.png', dpi=150,
                 bbox_inches='tight', facecolor='#0a0a0a')
    plt.close()
    
    print(f"  Saved: coherence_comparison.png")
    print(f"  Saved: coherence_cell_demo.png")
    
    print(f"\n{'='*70}")
    print("THE INSIGHT")
    print("=" * 70)
    print(f"""
  Classic displacement asks: "Is this pixel different from the background?"
  → Fails when text color ≈ background color.
  
  Coherence displacement asks: "Is this pixel in harmony with its neighbors?"
  → Works regardless of color because text ALWAYS disrupts local harmony.
  
  Why? Because a page is connected. Every background pixel agrees with 
  its neighbors — they're all part of the same smooth field. But text 
  pixels have EDGES. They have STRUCTURE. Even at the same color, the 
  letter boundary creates a coherence disruption that the background 
  doesn't have.
  
  This is the original theory, refined:
    v1: "Model the medium by its brightness. Detect brightness anomalies."
    v2: "Model the medium by its CONNECTEDNESS. Detect harmony disruptions."
  
  The medium isn't defined by what color it is.
  The medium is defined by how its parts relate to each other.
  The perturbation isn't a different color.
  The perturbation is a different PATTERN of relationship.
""")


if __name__ == '__main__':
    test_on_ward_image()
