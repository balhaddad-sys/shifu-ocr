"""
Shifu OCR — Medium Displacement Training

The principle: characters are not abstract shapes. They are perturbations
in a physical medium. Ink displaces paper. A scanner reads that displacement
through noise, texture, uneven illumination, and bleed.

This trainer renders characters INTO simulated page media:
  - Paper fiber texture (Perlin-like noise)
  - Ink bleed / spread (morphological dilation + gaussian blur)
  - Pressure variation (stroke weight randomization)
  - Scanner illumination gradient (uneven background)
  - Scanner noise (salt/pepper + gaussian)
  - Aging / fading (reduced contrast)

The landscapes learn the DISPLACEMENT SIGNATURE of each character
across hundreds of medium conditions — not the character shape itself.
"""

import os
import sys
import glob
import json
import time
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from shifu_ocr.engine import ShifuOCR

CHARS = list('abcdefghijklmnopqrstuvwxyz0123456789')
UPPER_CHARS = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')


# ═══════════════════════════════════════════════════════════════════
# MEDIUM SIMULATION — how ink meets paper meets scanner
# ═══════════════════════════════════════════════════════════════════

def make_paper_texture(size, rng, intensity=30):
    """Simulate paper fiber texture — low-frequency noise."""
    # Multi-scale noise simulates fiber structure
    h, w = size
    texture = np.ones((h, w), dtype=float) * 255

    for scale in [2, 4, 8, 16]:
        small = rng.random((max(h // scale, 1), max(w // scale, 1))) * intensity / scale
        zoomed = ndimage.zoom(small, (h / max(h // scale, 1), w / max(w // scale, 1)), order=1)
        zoomed = zoomed[:h, :w]
        texture -= zoomed

    return np.clip(texture, 0, 255).astype(np.uint8)


def simulate_ink_bleed(binary_char, rng, bleed_amount=1.0):
    """Simulate ink spreading into paper fibers."""
    if bleed_amount <= 0:
        return binary_char

    # Slight gaussian blur simulates ink diffusion
    sigma = bleed_amount * (0.5 + rng.random() * 0.5)
    blurred = ndimage.gaussian_filter(binary_char.astype(float), sigma=sigma)

    # Random threshold to create irregular edges
    thresh = 0.3 + rng.random() * 0.3
    return (blurred > thresh).astype(np.uint8)


def simulate_pressure_variation(char_img, rng):
    """Simulate uneven pen/print pressure across the character."""
    h, w = char_img.shape
    # Gradient across the character — left-right or top-bottom bias
    direction = rng.choice(['lr', 'tb', 'diag'])
    if direction == 'lr':
        grad = np.linspace(0.7 + rng.random() * 0.3, 1.0, w).reshape(1, -1)
    elif direction == 'tb':
        grad = np.linspace(0.7 + rng.random() * 0.3, 1.0, h).reshape(-1, 1)
    else:
        x = np.linspace(0, 1, w).reshape(1, -1)
        y = np.linspace(0, 1, h).reshape(-1, 1)
        grad = 0.7 + 0.3 * (x + y) / 2

    # Apply to ink regions only
    result = char_img.astype(float) * grad
    return np.clip(result, 0, 1).astype(float)


def simulate_scanner_illumination(size, rng):
    """Simulate uneven scanner lamp illumination."""
    h, w = size
    # Large-scale gradient — brighter center, darker edges (or vice versa)
    cx, cy = w * (0.3 + rng.random() * 0.4), h * (0.3 + rng.random() * 0.4)
    y, x = np.mgrid[0:h, 0:w].astype(float)
    dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    max_dist = np.sqrt(w ** 2 + h ** 2) / 2
    # 0.85-1.0 range — subtle illumination gradient
    illumination = 0.85 + 0.15 * (1 - dist / max_dist)
    return illumination


def simulate_scanner_noise(img, rng, noise_level=8):
    """Add scanner CCD noise — gaussian + salt/pepper."""
    result = img.astype(float)
    # Gaussian noise
    result += rng.normal(0, noise_level, img.shape)
    # Salt and pepper (stuck pixels)
    sp = rng.random(img.shape)
    result[sp < 0.003] = 0
    result[sp > 0.997] = 255
    return np.clip(result, 0, 255).astype(np.uint8)


def simulate_aging(img, rng, factor=None):
    """Simulate document aging — reduced contrast, yellowing."""
    if factor is None:
        factor = 0.6 + rng.random() * 0.35  # 0.6-0.95 contrast
    result = img.astype(float)
    mean = result.mean()
    result = mean + (result - mean) * factor
    return np.clip(result, 0, 255).astype(np.uint8)


def render_char_on_medium(char, font, rng, img_size=(100, 100)):
    """
    Render a character onto a simulated physical medium.
    Returns a grayscale image that looks like it came from a scanner.
    """
    h, w = img_size

    # 1. Start with paper texture (not flat white)
    paper = make_paper_texture(img_size, rng,
                               intensity=10 + rng.randint(0, 30))

    # 2. Render character as clean binary
    clean = Image.new('L', (w, h), color=255)
    draw = ImageDraw.Draw(clean)
    try:
        bbox = draw.textbbox((0, 0), char, font=font)
        cx = (w - (bbox[2] - bbox[0])) // 2 - bbox[0]
        cy = (h - (bbox[3] - bbox[1])) // 2 - bbox[1]
        draw.text((cx, cy), char, fill=0, font=font)
    except Exception:
        return None

    char_binary = (np.array(clean) < 128).astype(np.uint8)
    if char_binary.sum() < 10:
        return None

    # 3. Apply ink bleed
    bleed = rng.choice([0, 0.3, 0.6, 1.0, 1.5])
    char_bled = simulate_ink_bleed(char_binary, rng, bleed)

    # 4. Apply pressure variation
    char_pressure = simulate_pressure_variation(char_bled, rng)

    # 5. Composite: ink on paper
    # Ink darkness varies
    ink_darkness = rng.randint(0, 60)  # 0=black ink, 60=grey ink
    composite = paper.astype(float)
    ink_mask = char_pressure > 0.3
    composite[ink_mask] = ink_darkness + (1 - char_pressure[ink_mask]) * 40

    # 6. Scanner illumination gradient
    illumination = simulate_scanner_illumination(img_size, rng)
    composite = composite * illumination

    # 7. Scanner noise
    noise_level = 3 + rng.randint(0, 12)
    composite = simulate_scanner_noise(composite.astype(np.uint8), rng, noise_level)

    # 8. Aging (random chance)
    if rng.random() < 0.3:
        composite = simulate_aging(composite, rng)

    # 9. Slight rotation (scanner misalignment)
    if rng.random() < 0.2:
        angle = rng.uniform(-2, 2)
        composite = ndimage.rotate(composite, angle, reshape=False, mode='nearest', cval=255)

    return composite


# ═══════════════════════════════════════════════════════════════════
# TRAINING
# ═══════════════════════════════════════════════════════════════════

def find_fonts():
    font_dirs = [
        'C:/Windows/Fonts',
        os.path.expanduser('~/AppData/Local/Microsoft/Windows/Fonts'),
    ]
    font_files = []
    for d in font_dirs:
        if os.path.exists(d):
            font_files.extend(glob.glob(os.path.join(d, '*.ttf')))
            font_files.extend(glob.glob(os.path.join(d, '*.TTF')))
    seen = set()
    unique = []
    for f in sorted(font_files):
        base = os.path.basename(f).lower()
        if base not in seen:
            seen.add(base)
            unique.append(f)
    return unique


def check_font(font_path, size=36):
    try:
        font = ImageFont.truetype(font_path, size)
        img = Image.new('L', (80, 80), 255)
        draw = ImageDraw.Draw(img)
        draw.text((10, 10), 'A', fill=0, font=font)
        return (np.array(img) < 128).sum() > 20
    except Exception:
        return False


def train():
    model_path = os.path.join(os.path.dirname(__file__), 'trained_model.json')

    # Start fresh — medium displacement training replaces font-only training
    ocr = ShifuOCR()
    print('Starting medium displacement training')

    # Find usable fonts
    print('Scanning fonts...')
    all_fonts = find_fonts()
    usable = [f for f in all_fonts if check_font(f)]
    print(f'{len(usable)} usable fonts')

    # Use up to 100 fonts (quality > quantity now — medium variation matters more)
    max_fonts = min(len(usable), 100)
    if len(usable) > max_fonts:
        step = len(usable) // max_fonts
        usable = usable[::step][:max_fonts]
    print(f'Training on {len(usable)} fonts')

    SIZES = [24, 32, 40, 56, 72]
    # Each font+size gets multiple medium conditions
    MEDIUM_VARIATIONS = 6

    all_chars = CHARS + UPPER_CHARS
    total = 0
    start = time.time()

    for fi, font_path in enumerate(usable):
        font_name = os.path.basename(font_path)
        font_count = 0

        for size in SIZES:
            try:
                font = ImageFont.truetype(font_path, size)
            except Exception:
                continue

            for char in all_chars:
                label = char.lower()

                for variation in range(MEDIUM_VARIATIONS):
                    rng = np.random.RandomState(total + variation * 7919)
                    img = render_char_on_medium(char, font, rng, img_size=(100, 100))
                    if img is None:
                        continue

                    try:
                        ocr.train_character(label, img)
                        total += 1
                        font_count += 1
                    except Exception:
                        continue

        elapsed = time.time() - start
        rate = total / max(elapsed, 1)
        print(f'  [{fi+1}/{len(usable)}] {font_name}: {font_count} '
              f'(total: {total}, {rate:.0f}/sec, '
              f'{elapsed:.0f}s elapsed)')

    elapsed = time.time() - start
    stats = ocr.get_stats()
    print(f'\nTraining complete: {total} displacement observations')
    print(f'Time: {elapsed:.1f}s ({elapsed/60:.1f} min)')
    print(f'Landscapes: {stats["characters"]}')
    print(f'Depth: min={stats["min_depth"]}, max={stats["max_depth"]}, avg={stats["avg_depth"]:.0f}')

    # Save
    print(f'Saving to {model_path}...')
    ocr.save(model_path)
    size_mb = os.path.getsize(model_path) / 1024 / 1024
    print(f'Model size: {size_mb:.1f} MB')

    # Quick accuracy test
    print('\nQuick accuracy test...')
    test_rng = np.random.RandomState(42)
    correct = 0
    tested = 0
    test_font_path = usable[0]
    test_font = ImageFont.truetype(test_font_path, 40)

    for char in 'abcdefghijklmnopqrstuvwxyz0123456789':
        # Test with a NEW medium condition (unseen during training)
        img = render_char_on_medium(char, test_font, test_rng, img_size=(100, 100))
        if img is None:
            continue
        pred = ocr.predict_character(img)
        tested += 1
        if pred['predicted'] == char:
            correct += 1
        else:
            print(f'  {char} -> {pred["predicted"]} (conf: {pred["confidence"]:.3f})')

    print(f'Accuracy: {correct}/{tested} ({100*correct/max(tested,1):.0f}%)')
    print('Done.')


if __name__ == '__main__':
    train()
