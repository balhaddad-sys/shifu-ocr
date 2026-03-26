"""
Extensive ShifuOCR Training
Trains character landscapes on ALL system fonts, multiple sizes, with augmentations.
This makes the engine robust to different typefaces, sizes, and minor distortions.
"""

import os
import sys
import glob
import json
import time
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageFilter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from shifu_ocr.engine import ShifuOCR, extract_features, image_to_binary, normalize_region, extract_region

# Characters to train
CHARS = list('abcdefghijklmnopqrstuvwxyz0123456789')
# Add uppercase separately — they map to same landscapes as lowercase
UPPER_CHARS = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')

# Font sizes to train at (covers typical OCR scenarios)
FONT_SIZES = [20, 28, 36, 48, 64, 80]

# Augmentation: slight variations to simulate real-world conditions
def augment_image(img_array, seed=0):
    """Apply minor augmentations to simulate OCR conditions."""
    rng = np.random.RandomState(seed)
    augmented = []

    # Original
    augmented.append(img_array.copy())

    # Slight Gaussian blur (simulates low-res scan)
    img = Image.fromarray(img_array)
    blurred = np.array(img.filter(ImageFilter.GaussianBlur(radius=0.8)))
    augmented.append(blurred)

    # Salt and pepper noise (simulates scan artifacts)
    noisy = img_array.copy().astype(float)
    noise = rng.random(noisy.shape)
    noisy[noise < 0.02] = 0
    noisy[noise > 0.98] = 255
    augmented.append(noisy.astype(np.uint8))

    # Slight erosion (thinner strokes)
    from scipy import ndimage
    binary = (img_array < 128).astype(np.uint8)
    eroded = ndimage.binary_erosion(binary, iterations=1).astype(np.uint8)
    eroded_img = np.where(eroded, 0, 255).astype(np.uint8)
    augmented.append(eroded_img)

    # Slight dilation (thicker strokes)
    dilated = ndimage.binary_dilation(binary, iterations=1).astype(np.uint8)
    dilated_img = np.where(dilated, 0, 255).astype(np.uint8)
    augmented.append(dilated_img)

    return augmented


def render_char(char, font, img_size=(100, 100)):
    """Render a single character as a grayscale image."""
    img = Image.new('L', img_size, color=255)
    draw = ImageDraw.Draw(img)
    try:
        bbox = draw.textbbox((0, 0), char, font=font)
        x = (img_size[0] - (bbox[2] - bbox[0])) // 2 - bbox[0]
        y = (img_size[1] - (bbox[3] - bbox[1])) // 2 - bbox[1]
        draw.text((x, y), char, fill=0, font=font)
    except:
        return None
    arr = np.array(img)
    # Check if anything was actually rendered
    if arr.min() > 200:
        return None
    return arr


def find_fonts():
    """Find all usable TrueType fonts on the system."""
    font_dirs = [
        'C:/Windows/Fonts',
        os.path.expanduser('~/AppData/Local/Microsoft/Windows/Fonts'),
    ]
    font_files = []
    for d in font_dirs:
        if os.path.exists(d):
            font_files.extend(glob.glob(os.path.join(d, '*.ttf')))
            font_files.extend(glob.glob(os.path.join(d, '*.TTF')))
    # Deduplicate by basename
    seen = set()
    unique = []
    for f in sorted(font_files):
        base = os.path.basename(f).lower()
        if base not in seen:
            seen.add(base)
            unique.append(f)
    return unique


def test_font(font_path, size=36):
    """Check if a font can render Latin characters."""
    try:
        font = ImageFont.truetype(font_path, size)
        img = render_char('A', font)
        if img is None:
            return False
        # Check that it looks like a character (has enough ink)
        ink = (img < 128).sum()
        return ink > 20
    except:
        return False


def train():
    model_path = os.path.join(os.path.dirname(__file__), 'trained_model.json')

    # Start fresh or load existing
    if os.path.exists(model_path):
        ocr = ShifuOCR.load(model_path)
        print(f'Loaded existing model: {ocr.get_stats()["characters"]} characters')
    else:
        ocr = ShifuOCR()
        print('Starting fresh model')

    # Find all fonts
    print('Scanning system fonts...')
    all_fonts = find_fonts()
    print(f'Found {len(all_fonts)} font files')

    # Test which fonts render Latin characters
    print('Testing font compatibility...')
    usable_fonts = []
    for f in all_fonts:
        if test_font(f):
            usable_fonts.append(f)
    print(f'{len(usable_fonts)} fonts can render Latin characters')

    # Cap at reasonable number to avoid extremely long training
    max_fonts = min(len(usable_fonts), 200)
    if len(usable_fonts) > max_fonts:
        # Take a spread: every Nth font
        step = len(usable_fonts) // max_fonts
        usable_fonts = usable_fonts[::step][:max_fonts]
    print(f'Training on {len(usable_fonts)} fonts')

    total_trained = 0
    total_augmented = 0
    start = time.time()
    all_chars = CHARS + UPPER_CHARS

    for fi, font_path in enumerate(usable_fonts):
        font_name = os.path.basename(font_path)
        font_chars = 0

        for size in FONT_SIZES:
            try:
                font = ImageFont.truetype(font_path, size)
            except:
                continue

            for char in all_chars:
                img = render_char(char, font)
                if img is None:
                    continue

                # Map uppercase to lowercase label
                label = char.lower()

                # Train on original
                try:
                    ocr.train_character(label, img)
                    total_trained += 1
                    font_chars += 1
                except:
                    continue

                # Train on augmentations (only for a subset of sizes to save time)
                if size in [36, 64]:
                    for aug_img in augment_image(img, seed=total_trained)[1:]:
                        try:
                            ocr.train_character(label, aug_img)
                            total_augmented += 1
                        except:
                            continue

        elapsed = time.time() - start
        rate = total_trained / max(elapsed, 1)
        print(f'  [{fi+1}/{len(usable_fonts)}] {font_name}: {font_chars} chars '
              f'(total: {total_trained} + {total_augmented} aug, {rate:.0f}/sec)')

    # Save
    elapsed = time.time() - start
    print(f'\nTraining complete: {total_trained} + {total_augmented} augmented = {total_trained + total_augmented} total')
    print(f'Time: {elapsed:.1f}s')

    stats = ocr.get_stats()
    print(f'Landscapes: {stats["characters"]}')
    print(f'Min depth: {stats["min_depth"]}, Max depth: {stats["max_depth"]}, Avg: {stats["avg_depth"]:.0f}')

    print(f'Saving to {model_path}...')
    ocr.save(model_path)
    size_mb = os.path.getsize(model_path) / 1024 / 1024
    print(f'Model size: {size_mb:.1f} MB')
    print('Done.')


if __name__ == '__main__':
    train()
