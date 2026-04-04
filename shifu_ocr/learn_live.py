#!/usr/bin/env python3
"""
Shifu OCR — Real-time Learn Live Interface
Triggered in the background by the Node.js API upon UI correction.
Compares predicted vs truth, identifies confused characters, generates
training images, and updates the trained_model.json landscapes.
"""
import argparse
import sys
import os
import time
import glob
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

# ── Path setup: match pipeline_worker.py so shifu_ocr package resolves ──
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from shifu_ocr.engine import ShifuOCR

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'trained_model.json')
ALLOWED = set('abcdefghijklmnopqrstuvwxyz0123456789')
QUICK_SIZES = [28, 32, 40, 48]


def find_system_fonts(max_fonts=10):
    """Find usable TTF fonts on the system."""
    font_dirs = [
        'C:/Windows/Fonts',
        os.path.expanduser('~/AppData/Local/Microsoft/Windows/Fonts'),
    ]
    found = []
    for d in font_dirs:
        if not os.path.exists(d):
            continue
        for pattern in ['*.ttf', '*.TTF']:
            found.extend(glob.glob(os.path.join(d, pattern)))

    # Deduplicate and filter to usable fonts
    seen = set()
    usable = []
    for f in sorted(found):
        base = os.path.basename(f).lower()
        if base in seen:
            continue
        seen.add(base)
        try:
            font = ImageFont.truetype(f, 36)
            # Quick sanity check: can it render a character?
            img = Image.new('L', (60, 60), 255)
            draw = ImageDraw.Draw(img)
            draw.text((5, 5), 'A', fill=0, font=font)
            if (np.array(img) < 128).sum() > 10:
                usable.append(f)
        except Exception:
            continue
        if len(usable) >= max_fonts:
            break

    return usable


def identify_confusions(predicted, truth):
    """Compare predicted vs truth and return the set of truth characters
    that were misrecognized."""
    pred = predicted.strip().lower().replace(' ', '')
    truth = truth.strip().lower().replace(' ', '')

    if not pred or not truth:
        return set()

    confused = set()

    # Character-level alignment (simple positional)
    for i in range(min(len(pred), len(truth))):
        if pred[i] != truth[i] and truth[i] in ALLOWED:
            confused.add(truth[i])

    # Trailing characters the engine missed entirely
    if len(truth) > len(pred):
        for c in truth[len(pred):]:
            if c in ALLOWED:
                confused.add(c)

    return confused


def render_and_train(ocr, char, fonts, sizes):
    """Render a character at multiple fonts/sizes and train the engine.
    Returns the number of successful training events."""
    count = 0
    for fp in fonts:
        for size in sizes:
            try:
                font = ImageFont.truetype(fp, size)
            except Exception:
                continue

            # 2 random alignment offsets per font/size combo
            for _ in range(2):
                img = Image.new('L', (100, 100), color=255)
                draw = ImageDraw.Draw(img)
                try:
                    bbox = draw.textbbox((0, 0), char, font=font)
                    cx = (100 - (bbox[2] - bbox[0])) // 2 - bbox[0]
                    cy = (100 - (bbox[3] - bbox[1])) // 2 - bbox[1]
                    cx += np.random.randint(-2, 3)
                    cy += np.random.randint(-2, 3)
                    draw.text((cx, cy), char, fill=0, font=font)
                except Exception:
                    continue

                arr = np.array(img)
                if (arr < 128).sum() < 10:
                    continue

                try:
                    ocr.train_character(char, arr)
                    count += 1
                except Exception as e:
                    print(f"[LIVE] train_character failed for '{char}': {e}",
                          file=sys.stderr)
    return count


def main():
    parser = argparse.ArgumentParser(
        description="Shifu real-time landscape calibration")
    parser.add_argument('--predicted', required=True,
                        help="What the engine predicted")
    parser.add_argument('--truth', required=True,
                        help="What the human corrected it to")
    args = parser.parse_args()

    # 1. Identify confused characters
    confused = identify_confusions(args.predicted, args.truth)
    if not confused:
        sys.exit(0)

    print(f"[LIVE] Confused characters: {sorted(confused)}")

    # 2. Find fonts
    fonts = find_system_fonts(max_fonts=8)
    if not fonts:
        print("[LIVE] No usable TTF fonts found.", file=sys.stderr)
        sys.exit(1)
    print(f"[LIVE] Using {len(fonts)} fonts")

    # 3. Load model
    t0 = time.time()
    if os.path.exists(MODEL_PATH):
        ocr = ShifuOCR.load(MODEL_PATH)
    else:
        print(f"[LIVE] Model not found at {MODEL_PATH}", file=sys.stderr)
        sys.exit(1)

    # 4. Train on each confused character
    total_trained = 0
    for char in sorted(confused):
        n = render_and_train(ocr, char, fonts, QUICK_SIZES)
        total_trained += n
        if n > 0:
            print(f"[LIVE]   '{char}': {n} training samples")

    # 5. Save ONLY if we actually trained something
    if total_trained > 0:
        ocr.save(MODEL_PATH)
        elapsed = time.time() - t0
        print(f"[LIVE] Calibrated {total_trained} landscapes in {elapsed:.2f}s")
    else:
        print("[LIVE] No training samples generated — model unchanged.")


if __name__ == '__main__':
    main()
