"""
Shifu OCR — Train from REAL images

Uses the 50,000 labeled medical text images from med_ocr/training_data.
Each image contains a line of text with its label.

The engine segments each image into characters using vertical projection,
then aligns the segments to the label characters, and trains each landscape
on the ACTUAL pixel displacement — real ink, real medium, real noise.

This is Shifu's principle: learn from the medium itself, not from theory.
"""

import os
import sys
import time
import json
import numpy as np
from pathlib import Path
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from shifu_ocr.engine import ShifuOCR


TRAINING_DIR = 'c:/Users/balha/OneDrive/Desktop/Application/MedTriage/med_ocr/training_data'
ALLOWED_CHARS = set('abcdefghijklmnopqrstuvwxyz0123456789')


def load_training_list(path):
    """Load image paths and labels from train_list.txt"""
    entries = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or '\t' not in line:
                continue
            img_path, label = line.split('\t', 1)
            full_path = os.path.join(os.path.dirname(path), img_path)
            if os.path.exists(full_path):
                entries.append((full_path, label))
    return entries


def align_segments_to_label(segments, label):
    """
    Align character segments to label characters.

    The label is a string like "Metronidazole, Omeprazole".
    The segments are character bounding boxes from vertical projection.

    We strip non-alphanumeric characters from the label and try to
    match segments to remaining characters by position.
    """
    # Extract just the characters we can train on
    clean_chars = []
    for ch in label:
        if ch.lower() in ALLOWED_CHARS:
            clean_chars.append(ch.lower())
        elif ch == ' ':
            clean_chars.append(' ')  # space marker

    # Remove spaces — segments don't include spaces
    char_labels = [c for c in clean_chars if c != ' ']

    if len(char_labels) == 0:
        return []

    # Group segments into words by detecting gaps
    if len(segments) < 2:
        return list(zip(segments, char_labels[:len(segments)]))

    gaps = []
    for i in range(1, len(segments)):
        prev_end = segments[i - 1]['bbox'][3]
        curr_start = segments[i]['bbox'][1]
        gaps.append(curr_start - prev_end)

    # If segment count matches label character count, direct alignment
    if len(segments) == len(char_labels):
        return list(zip(segments, char_labels))

    # If close (within 20%), align from the start
    ratio = len(segments) / max(len(char_labels), 1)
    if 0.8 <= ratio <= 1.2:
        pairs = []
        n = min(len(segments), len(char_labels))
        for i in range(n):
            pairs.append((segments[i], char_labels[i]))
        return pairs

    # Too different — skip this image
    return []


def train():
    model_path = os.path.join(os.path.dirname(__file__), 'trained_model.json')

    # Resume if model exists, otherwise start fresh
    if os.path.exists(model_path):
        ocr = ShifuOCR.load(model_path)
        print(f'Resuming from existing model: {ocr.get_stats()["characters"]} landscapes')
    else:
        ocr = ShifuOCR()
        print('Starting training from real images')

    # Load training list
    train_list = os.path.join(TRAINING_DIR, 'train_list.txt')
    entries = load_training_list(train_list)
    print(f'Loaded {len(entries)} labeled images')

    # Also load validation if available
    val_list = os.path.join(TRAINING_DIR, 'val_list.txt')
    if os.path.exists(val_list):
        val_entries = load_training_list(val_list)
        print(f'Loaded {len(val_entries)} validation images')
    else:
        val_entries = []

    # Train on all entries
    total_chars = 0
    total_aligned = 0
    total_skipped = 0
    start = time.time()

    import gc

    for ei, (img_path, label) in enumerate(entries):
        try:
            img = Image.open(img_path)
            if img.mode != 'L':
                img = img.convert('L')
            arr = np.array(img)
            img.close()  # Free image memory immediately

            segments = ocr.segment_characters(arr, min_char_width=2)
            if not segments:
                total_skipped += 1
                continue

            pairs = align_segments_to_label(segments, label)
            if not pairs:
                total_skipped += 1
                continue

            for seg, char_label in pairs:
                try:
                    ocr.train_character(char_label, seg['image'])
                    total_chars += 1
                    total_aligned += 1
                except:
                    pass

            # Free segment images
            del segments, pairs, arr

        except Exception as e:
            total_skipped += 1
            continue

        if (ei + 1) % 1000 == 0:
            elapsed = time.time() - start
            rate = total_chars / max(elapsed, 1)
            print(f'  [{ei+1}/{len(entries)}] chars: {total_chars}, '
                  f'aligned: {total_aligned}, skipped: {total_skipped}, '
                  f'{rate:.0f} chars/sec, {elapsed:.0f}s', flush=True)

        # Checkpoint + GC every 5k images
        if (ei + 1) % 5000 == 0:
            ocr.save(model_path)
            gc.collect()
            print(f'  ** Checkpoint saved at {ei+1} images **', flush=True)

    elapsed = time.time() - start
    stats = ocr.get_stats()
    print(f'\nTraining complete:')
    print(f'  Images processed: {len(entries) - total_skipped}/{len(entries)}')
    print(f'  Characters trained: {total_chars}')
    print(f'  Skipped images: {total_skipped}')
    print(f'  Time: {elapsed:.1f}s ({elapsed/60:.1f} min)')
    print(f'  Landscapes: {stats["characters"]}')
    print(f'  Depth: min={stats["min_depth"]}, max={stats["max_depth"]}, avg={stats["avg_depth"]:.0f}')

    # Save
    print(f'\nSaving to {model_path}...')
    ocr.save(model_path)
    size_mb = os.path.getsize(model_path) / 1024 / 1024
    print(f'Model size: {size_mb:.1f} MB')

    # Quick test on validation data
    if val_entries:
        print('\nValidation test...')
        correct = 0
        tested = 0
        for img_path, label in val_entries[:200]:
            try:
                img = Image.open(img_path).convert('L')
                result = ocr.read_line(np.array(img))
                # Compare character by character
                pred = result['text'].lower().replace(' ', '')
                truth = ''.join(c.lower() for c in label if c.lower() in ALLOWED_CHARS)
                for i in range(min(len(pred), len(truth))):
                    tested += 1
                    if pred[i] == truth[i]:
                        correct += 1
            except:
                pass
        if tested > 0:
            print(f'  Character accuracy: {correct}/{tested} ({100*correct/tested:.1f}%)')

    print('Done.')


if __name__ == '__main__':
    train()
