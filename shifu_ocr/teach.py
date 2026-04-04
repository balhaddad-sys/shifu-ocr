#!/usr/bin/env python3
"""
Shifu OCR — Comprehensive Teaching Pipeline
=============================================

Teaches the topological engine to recognise every printable character
by dynamically discovering system fonts and iterating through:

  Phase 1  Foundation    — Train every char × every font × multiple sizes
  Phase 2  Augmentation  — Jitter, noise, blur, scale variation
  Phase 3  Validation    — Render words, read them back, measure accuracy
  Phase 4  Confusion     — Identify misreads, retrain specifically on those
  Phase 5  Repeat        — Loop Phase 3-4 until accuracy plateaus or target hit

Zero hardcoding: fonts are discovered, test words are generated from
character bigrams, and confusion pairs emerge organically from the engine's
own mistakes.

Usage:
    python shifu_ocr/teach.py                        # Full training
    python shifu_ocr/teach.py --rounds 5             # Limit rounds
    python shifu_ocr/teach.py --target 0.60          # Stop at 60% accuracy
    python shifu_ocr/teach.py --chars-only            # Phase 1+2 only (no validation loop)
    python shifu_ocr/teach.py --resume                # Continue from existing model
"""

import argparse
import glob
import os
import random
import string
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

# ── Path setup ──
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from shifu_ocr.engine import ShifuOCR

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          'trained_model.json')

# Full character set — discovered, not hardcoded
LOWERCASE = list(string.ascii_lowercase)        # a-z
UPPERCASE = list(string.ascii_uppercase)        # A-Z
DIGITS    = list(string.digits)                 # 0-9
ALL_CHARS = LOWERCASE + UPPERCASE + DIGITS


# =============================================================================
# PHASE 0 — Font Discovery
# =============================================================================

def discover_fonts(max_fonts=0):
    """Scan the system for usable TrueType fonts. Returns list of paths.
    max_fonts=0 means no limit."""
    search_dirs = [
        'C:/Windows/Fonts',
        os.path.expanduser('~/AppData/Local/Microsoft/Windows/Fonts'),
        '/usr/share/fonts',
        '/usr/local/share/fonts',
        os.path.expanduser('~/.fonts'),
        '/System/Library/Fonts',
        '/Library/Fonts',
    ]

    candidates = []
    for d in search_dirs:
        if os.path.isdir(d):
            candidates.extend(glob.glob(os.path.join(d, '**', '*.ttf'),
                                        recursive=True))
            candidates.extend(glob.glob(os.path.join(d, '**', '*.TTF'),
                                        recursive=True))
            candidates.extend(glob.glob(os.path.join(d, '**', '*.otf'),
                                        recursive=True))

    # Deduplicate by basename
    seen = set()
    unique = []
    for fp in sorted(candidates):
        base = os.path.basename(fp).lower()
        if base in seen:
            continue
        seen.add(base)
        unique.append(fp)

    # Filter to fonts that actually render cleanly
    usable = []
    for fp in unique:
        try:
            font = ImageFont.truetype(fp, 36)
            img = Image.new('L', (80, 80), 255)
            draw = ImageDraw.Draw(img)
            draw.text((10, 10), 'Ag5', fill=0, font=font)
            arr = np.array(img)
            if (arr < 128).sum() > 30:
                usable.append(fp)
        except Exception:
            continue
        if max_fonts and len(usable) >= max_fonts:
            break

    return usable


# =============================================================================
# PHASE 1 — Foundation Training
# =============================================================================

def render_char(char, font, size, img_size=(100, 100), offset=(0, 0)):
    """Render a single character centered in an image."""
    img = Image.new('L', img_size, color=255)
    draw = ImageDraw.Draw(img)
    try:
        pil_font = ImageFont.truetype(font, size)
        bbox = draw.textbbox((0, 0), char, font=pil_font)
        cx = (img_size[0] - (bbox[2] - bbox[0])) // 2 - bbox[0] + offset[0]
        cy = (img_size[1] - (bbox[3] - bbox[1])) // 2 - bbox[1] + offset[1]
        draw.text((cx, cy), char, fill=0, font=pil_font)
    except Exception:
        return None
    arr = np.array(img)
    if (arr < 128).sum() < 8:
        return None
    return arr


def train_foundation(ocr, fonts, chars, sizes=None):
    """Phase 1: Train every character × every font × multiple sizes."""
    if sizes is None:
        sizes = [24, 32, 40, 48, 60, 72]

    total = len(chars) * len(fonts) * len(sizes)
    trained = 0
    t0 = time.time()

    print(f"\n{'='*60}")
    print(f"  PHASE 1 — Foundation Training")
    print(f"  {len(chars)} chars × {len(fonts)} fonts × {len(sizes)} sizes "
          f"= {total} max samples")
    print(f"{'='*60}")

    for ci, char in enumerate(chars):
        char_count = 0
        for font in fonts:
            for size in sizes:
                arr = render_char(char, font, size)
                if arr is not None:
                    try:
                        ocr.train_character(char, arr)
                        trained += 1
                        char_count += 1
                    except Exception:
                        pass

        if (ci + 1) % 10 == 0 or ci == len(chars) - 1:
            elapsed = time.time() - t0
            pct = (ci + 1) / len(chars) * 100
            print(f"  [{pct:5.1f}%] '{char}' trained with {char_count} samples "
                  f"({trained} total, {elapsed:.0f}s)")

    elapsed = time.time() - t0
    print(f"\n  Foundation complete: {trained} samples in {elapsed:.1f}s")
    return trained


# =============================================================================
# PHASE 2 — Augmented Training
# =============================================================================

def augment_image(arr):
    """Apply random augmentations to a character image."""
    img = Image.fromarray(arr)
    augmentations = []

    # Random slight blur
    if random.random() < 0.4:
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.5)))
        augmentations.append('blur')

    # Random noise
    if random.random() < 0.3:
        noise = np.random.normal(0, random.randint(5, 15), arr.shape)
        noised = np.clip(np.array(img).astype(float) + noise, 0, 255).astype(np.uint8)
        img = Image.fromarray(noised)
        augmentations.append('noise')

    # Slight erosion or dilation effect via threshold shift
    if random.random() < 0.3:
        a = np.array(img)
        shift = random.randint(-20, 20)
        a = np.clip(a.astype(int) + shift, 0, 255).astype(np.uint8)
        img = Image.fromarray(a)
        augmentations.append('threshold')

    return np.array(img)


def train_augmented(ocr, fonts, chars, rounds=2):
    """Phase 2: Train with augmented versions for robustness."""
    sizes = [28, 36, 44, 56]
    offsets = [(0, 0), (-3, -2), (2, 3), (-1, 4), (3, -3)]
    trained = 0
    t0 = time.time()

    # Use a subset of fonts for augmentation (speed)
    aug_fonts = fonts[:min(len(fonts), 8)]

    print(f"\n{'='*60}")
    print(f"  PHASE 2 — Augmented Training")
    print(f"  {len(chars)} chars × {len(aug_fonts)} fonts × "
          f"{len(sizes)} sizes × {rounds} rounds")
    print(f"{'='*60}")

    for r in range(rounds):
        round_count = 0
        for char in chars:
            for font in aug_fonts:
                size = random.choice(sizes)
                offset = random.choice(offsets)
                arr = render_char(char, font, size, offset=offset)
                if arr is None:
                    continue

                # Train original + augmented
                try:
                    ocr.train_character(char, arr)
                    round_count += 1

                    aug = augment_image(arr)
                    if (aug < 128).sum() >= 8:
                        ocr.train_character(char, aug)
                        round_count += 1
                except Exception:
                    pass

        trained += round_count
        elapsed = time.time() - t0
        print(f"  Round {r+1}/{rounds}: {round_count} samples ({elapsed:.0f}s)")

    print(f"\n  Augmentation complete: {trained} total samples")
    return trained


# =============================================================================
# PHASE 3 — Validation
# =============================================================================

def generate_test_words(n=100):
    """Generate test words dynamically from character combinations.
    No hardcoded test data — words are built from random slices of the
    character set to cover the full recognition space."""
    words = []

    # Common English bigrams as building blocks (statistically derived)
    bigrams = [
        'th', 'he', 'in', 'er', 'an', 'on', 'at', 'en', 'nd', 'ti',
        'es', 'or', 'te', 'of', 'ed', 'is', 'it', 'al', 'ar', 'st',
        'to', 'nt', 'ng', 'se', 'ha', 'as', 're', 'ou', 'le', 'no',
        'de', 'io', 'co', 'me', 'ma', 'li', 'ne', 'om', 've', 'ea',
    ]

    # Build words from bigram sequences (length 3-8)
    for _ in range(n // 2):
        length = random.randint(1, 3)
        word = ''.join(random.choices(bigrams, k=length))
        words.append(word[:random.randint(3, 8)])

    # Pure random lowercase words
    for _ in range(n // 4):
        length = random.randint(3, 7)
        words.append(''.join(random.choices(LOWERCASE, k=length)))

    # Digit sequences
    for _ in range(n // 8):
        length = random.randint(2, 5)
        words.append(''.join(random.choices(DIGITS, k=length)))

    # Mixed alphanumeric
    for _ in range(n // 8):
        length = random.randint(3, 6)
        pool = LOWERCASE + DIGITS
        words.append(''.join(random.choices(pool, k=length)))

    return words


def validate(ocr, fonts, n_words=80):
    """Phase 3: Render test words, read them back, measure accuracy.
    Returns (accuracy, confusions_dict)."""
    words = generate_test_words(n_words)
    # Use a few diverse fonts for validation
    val_fonts = fonts[:min(len(fonts), 6)]

    total_chars = 0
    correct_chars = 0
    confusions = defaultdict(lambda: defaultdict(int))

    t0 = time.time()

    for word in words:
        font_path = random.choice(val_fonts)
        size = random.choice([32, 40, 48])

        # Render the word as a line image
        width = max(len(word) * size, 200)
        img = Image.new('L', (width, size + 40), color=255)
        draw = ImageDraw.Draw(img)

        try:
            font = ImageFont.truetype(font_path, size)
            draw.text((10, 10), word, fill=0, font=font)
        except Exception:
            continue

        arr = np.array(img)
        if (arr < 128).sum() < 10:
            continue

        # Read it back
        result = ocr.read_line(arr)
        predicted = result['text'].lower().replace(' ', '')
        truth = word.lower().replace(' ', '')

        # Character-level comparison
        for i in range(min(len(predicted), len(truth))):
            total_chars += 1
            if predicted[i] == truth[i]:
                correct_chars += 1
            else:
                confusions[truth[i]][predicted[i]] += 1

        # Count missed/extra characters
        if len(truth) > len(predicted):
            total_chars += len(truth) - len(predicted)

    accuracy = correct_chars / max(total_chars, 1)
    elapsed = time.time() - t0

    # Sort confusions by frequency
    confusion_pairs = []
    for true_char, preds in confusions.items():
        for pred_char, count in preds.items():
            confusion_pairs.append((true_char, pred_char, count))
    confusion_pairs.sort(key=lambda x: x[2], reverse=True)

    print(f"\n  Validation: {correct_chars}/{total_chars} chars correct "
          f"= {accuracy*100:.1f}% ({elapsed:.0f}s)")
    if confusion_pairs:
        print(f"  Top confusions:")
        for true_c, pred_c, count in confusion_pairs[:10]:
            print(f"    '{true_c}' read as '{pred_c}' — {count} times")

    return accuracy, confusions


# =============================================================================
# PHASE 4 — Confusion-Targeted Retraining
# =============================================================================

def retrain_confusions(ocr, confusions, fonts, extra_per_char=30):
    """Phase 4: Retrain specifically on the characters the engine confuses."""
    # Identify most-confused truth characters
    char_errors = defaultdict(int)
    for true_char, preds in confusions.items():
        for _, count in preds.items():
            char_errors[true_char] += count

    # Sort by error count, focus on worst offenders
    worst = sorted(char_errors.items(), key=lambda x: x[1], reverse=True)[:20]
    if not worst:
        print("  No confusions to retrain.")
        return 0

    target_chars = [c for c, _ in worst]
    print(f"\n  Retraining {len(target_chars)} confused characters: "
          f"{target_chars}")

    # Use diverse fonts and sizes
    retrain_fonts = fonts[:min(len(fonts), 12)]
    sizes = [24, 30, 36, 42, 48, 56, 64]
    offsets = [(0, 0), (-2, -1), (1, 2), (-1, 3), (2, -2)]

    trained = 0
    for char in target_chars:
        for font in retrain_fonts:
            attempts = extra_per_char // max(len(retrain_fonts), 1) + 1
            for _ in range(attempts):
                size = random.choice(sizes)
                offset = random.choice(offsets)
                arr = render_char(char, font, size, offset=offset)
                if arr is None:
                    continue
                try:
                    ocr.train_character(char, arr)
                    trained += 1
                except Exception:
                    pass

                # Also train an augmented version
                if random.random() < 0.5:
                    aug = augment_image(arr)
                    if (aug < 128).sum() >= 8:
                        try:
                            ocr.train_character(char, aug)
                            trained += 1
                        except Exception:
                            pass

    print(f"  Confusion retraining: {trained} additional samples")
    return trained


# =============================================================================
# MAIN TRAINING LOOP
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Shifu OCR — Comprehensive Teaching Pipeline')
    parser.add_argument('--rounds', type=int, default=15,
                        help='Max validation/retrain rounds (default: 15)')
    parser.add_argument('--target', type=float, default=0.55,
                        help='Target character accuracy to stop at (default: 0.55)')
    parser.add_argument('--max-fonts', type=int, default=0,
                        help='Limit number of fonts (0 = use all)')
    parser.add_argument('--chars-only', action='store_true',
                        help='Only run Phase 1 + 2 (no validation loop)')
    parser.add_argument('--resume', action='store_true',
                        help='Continue training from existing model')
    parser.add_argument('--lowercase-only', action='store_true',
                        help='Train only lowercase + digits (faster)')
    args = parser.parse_args()

    print(f"\n{'#'*60}")
    print(f"  SHIFU TEACH — Comprehensive Character Training")
    print(f"{'#'*60}")

    # ── Discover fonts ──
    print("\nDiscovering system fonts...")
    fonts = discover_fonts(max_fonts=args.max_fonts)
    print(f"Found {len(fonts)} usable fonts")
    if not fonts:
        print("ERROR: No TrueType fonts found on this system.")
        sys.exit(1)

    # Show a sample
    for f in fonts[:5]:
        print(f"  • {os.path.basename(f)}")
    if len(fonts) > 5:
        print(f"  ... and {len(fonts) - 5} more")

    # ── Choose character set ──
    if args.lowercase_only:
        chars = LOWERCASE + DIGITS
        print(f"\nCharacter set: {len(chars)} (lowercase + digits)")
    else:
        chars = ALL_CHARS
        print(f"\nCharacter set: {len(chars)} (a-z, A-Z, 0-9)")

    # ── Load or create model ──
    if args.resume and os.path.exists(MODEL_PATH):
        print(f"\nResuming from existing model: {MODEL_PATH}")
        ocr = ShifuOCR.load(MODEL_PATH)
        stats = ocr.get_stats()
        print(f"  Loaded: {stats['characters']} landscapes, "
              f"avg depth {stats['avg_depth']:.0f}")
    else:
        print("\nStarting fresh model")
        ocr = ShifuOCR()

    overall_t0 = time.time()

    # ── Phase 1: Foundation ──
    foundation_count = train_foundation(ocr, fonts, chars)

    # Save checkpoint
    ocr.save(MODEL_PATH)
    print(f"  ✓ Checkpoint saved ({MODEL_PATH})")

    # ── Phase 2: Augmentation ──
    aug_count = train_augmented(ocr, fonts, chars, rounds=2)

    # Save checkpoint
    ocr.save(MODEL_PATH)
    print(f"  ✓ Checkpoint saved")

    if args.chars_only:
        stats = ocr.get_stats()
        elapsed = time.time() - overall_t0
        print(f"\n{'='*60}")
        print(f"  TRAINING COMPLETE (chars-only mode)")
        print(f"  {foundation_count + aug_count} total training samples")
        print(f"  {stats['characters']} landscapes, "
              f"avg depth {stats['avg_depth']:.0f}")
        print(f"  Time: {elapsed:.0f}s")
        print(f"{'='*60}")
        return

    # ── Phase 3+4 Loop: Validate → Retrain ──
    print(f"\n{'='*60}")
    print(f"  PHASE 3+4 — Validation & Confusion Retraining Loop")
    print(f"  Max rounds: {args.rounds}, target accuracy: {args.target*100:.0f}%")
    print(f"{'='*60}")

    best_accuracy = 0
    history = []

    for round_num in range(1, args.rounds + 1):
        print(f"\n--- Round {round_num}/{args.rounds} ---")

        # Validate
        accuracy, confusions = validate(ocr, fonts, n_words=100)
        history.append(accuracy)

        # Check if target reached
        if accuracy >= args.target:
            print(f"\n  ★ TARGET REACHED: {accuracy*100:.1f}% >= "
                  f"{args.target*100:.0f}%")
            ocr.save(MODEL_PATH)
            break

        # Save if new best
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            ocr.save(MODEL_PATH)
            print(f"  ✓ New best: {accuracy*100:.1f}% — saved")

        # Check for plateau (no improvement in last 3 rounds)
        if len(history) >= 4:
            recent = history[-3:]
            if max(recent) - min(recent) < 0.005:
                print(f"\n  ⊘ PLATEAU: accuracy stable at ~{accuracy*100:.1f}%")
                break

        # Retrain on confusions
        retrained = retrain_confusions(ocr, confusions, fonts)
        if retrained == 0:
            print("  No confusions to fix — stopping.")
            break

    # ── Final save + summary ──
    ocr.save(MODEL_PATH)
    stats = ocr.get_stats()
    elapsed = time.time() - overall_t0

    print(f"\n{'#'*60}")
    print(f"  TEACHING COMPLETE")
    print(f"{'#'*60}")
    print(f"  Model:      {MODEL_PATH}")
    print(f"  Landscapes: {stats['characters']}")
    print(f"  Avg depth:  {stats['avg_depth']:.0f} observations per landscape")
    print(f"  Rounds:     {len(history)}")
    if history:
        print(f"  Accuracy:   {' → '.join(f'{a*100:.1f}%' for a in history)}")
        print(f"  Best:       {best_accuracy*100:.1f}%")
    print(f"  Time:       {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"{'#'*60}\n")


if __name__ == '__main__':
    main()
