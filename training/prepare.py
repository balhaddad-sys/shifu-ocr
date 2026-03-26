#!/usr/bin/env python3
"""
Prepare PaddleOCR training data from seed bank.
Converts seed bank JSON records into PaddleOCR recognition training format.

PaddleOCR expects:
  - Images: cropped text line images (32px height)
  - Labels: text file with "image_path\tground_truth_text" per line

Since we don't have real images yet, we generate synthetic text-line images
using PIL/Pillow with random fonts, noise, and distortion.

Usage:
    python prepare_training.py [--count 5000]
"""
import os
import sys
import json
import random
from pathlib import Path

try:
    from PIL import Image, ImageDraw, ImageFont, ImageFilter
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    print("WARNING: Pillow not installed. Run: pip install Pillow")

SEED_BANK = os.path.join(os.path.dirname(__file__), 'seed_bank')
TRAIN_DIR = os.path.join(os.path.dirname(__file__), 'training_data')
IMAGES_DIR = os.path.join(TRAIN_DIR, 'images')
os.makedirs(IMAGES_DIR, exist_ok=True)


def generate_text_image(text, idx, height=48):
    """Generate a synthetic text-line image for OCR training."""
    if not HAS_PIL:
        return None

    # Random styling to simulate real-world variation
    font_size = random.randint(18, 32)
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except (OSError, IOError):
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
        except (OSError, IOError):
            font = ImageFont.load_default()

    # Calculate text dimensions
    dummy = Image.new('RGB', (1, 1))
    draw = ImageDraw.Draw(dummy)
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0] + 20
    text_h = bbox[3] - bbox[1] + 10

    # Create image with slight background variation
    bg_val = random.randint(230, 255)
    img = Image.new('RGB', (max(text_w, 50), max(text_h, height)), (bg_val, bg_val, bg_val))
    draw = ImageDraw.Draw(img)

    # Text color variation (dark gray to black)
    text_color = random.randint(0, 40)
    draw.text((10, 5), text, fill=(text_color, text_color, text_color), font=font)

    # Add slight noise
    if random.random() < 0.3:
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.3, 0.8)))

    # Resize to standard height
    w_ratio = height / img.height
    new_w = max(int(img.width * w_ratio), 20)
    img = img.resize((new_w, height), Image.LANCZOS)

    # Save
    filename = f'train_{idx:06d}.png'
    filepath = os.path.join(IMAGES_DIR, filename)
    img.save(filepath)
    return filename


def extract_training_texts(max_count=5000):
    """Extract text lines from seed bank for training."""
    texts = []

    # 1. From seed bank records
    seed_files = sorted(f for f in os.listdir(SEED_BANK) if f.startswith('seed_') and f.endswith('.json'))
    for sf in seed_files[:max_count]:
        try:
            with open(os.path.join(SEED_BANK, sf), 'r', encoding='utf-8') as f:
                seed = json.load(f)
            output = seed.get('structured_output', {})
            if isinstance(output, dict):
                # Extract individual field values as training lines
                for field in ['fullName', 'bed', 'dx', 'meds', 'ward', 'assignedDoctor', 'allergies']:
                    val = output.get(field)
                    if val and len(str(val)) > 1:
                        texts.append(str(val))
                # Also add the age/gender combo
                age = output.get('age')
                gender = output.get('gender')
                if age and gender:
                    texts.append(f'{age}/{gender}')
        except Exception:
            continue

    # 2. From synthetic patients (if available)
    patients_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'seed_raw', 'synthetic_patients.json')
    if os.path.exists(patients_path):
        try:
            with open(patients_path, 'r', encoding='utf-8') as f:
                patients = json.load(f)
            for p in patients[:max_count]:
                texts.append(p.get('fullName', ''))
                texts.append(p.get('bed', ''))
                texts.append(f'{p.get("age", "")}/{p.get("gender", "")}')
                texts.append(p.get('dx', ''))
                if p.get('meds'):
                    texts.append(p['meds'])
        except Exception:
            pass

    # Deduplicate and filter
    texts = list(set(t.strip() for t in texts if t and len(t.strip()) > 1))
    random.shuffle(texts)
    return texts[:max_count]


def main():
    max_count = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 5000
    if '--count' in sys.argv:
        idx = sys.argv.index('--count')
        if idx + 1 < len(sys.argv):
            max_count = int(sys.argv[idx + 1])

    print(f'Extracting up to {max_count} training texts...')
    texts = extract_training_texts(max_count)
    print(f'Extracted {len(texts)} unique text lines')

    # Generate images and labels
    train_labels = []
    val_labels = []
    split_point = int(len(texts) * 0.9)  # 90% train, 10% val

    if HAS_PIL:
        print(f'Generating {len(texts)} synthetic text images...')
        for i, text in enumerate(texts):
            filename = generate_text_image(text, i)
            if filename:
                label_line = f'images/{filename}\t{text}'
                if i < split_point:
                    train_labels.append(label_line)
                else:
                    val_labels.append(label_line)

            if (i + 1) % 1000 == 0:
                print(f'  {i + 1}/{len(texts)} images generated')
    else:
        print('Skipping image generation (no Pillow). Creating text-only labels...')
        for i, text in enumerate(texts):
            label_line = f'synthetic_{i:06d}\t{text}'
            if i < split_point:
                train_labels.append(label_line)
            else:
                val_labels.append(label_line)

    # Write label files
    train_path = os.path.join(TRAIN_DIR, 'train_list.txt')
    val_path = os.path.join(TRAIN_DIR, 'val_list.txt')

    with open(train_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(train_labels) + '\n')
    with open(val_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(val_labels) + '\n')

    # Extract and write custom dictionary
    all_chars = set()
    for text in texts:
        all_chars.update(text)
    all_chars = sorted(all_chars)

    dict_path = os.path.join(TRAIN_DIR, 'med_dict.txt')
    with open(dict_path, 'w', encoding='utf-8') as f:
        for char in all_chars:
            f.write(char + '\n')

    print(f'\n=== TRAINING DATA READY ===')
    print(f'  Training samples: {len(train_labels)}')
    print(f'  Validation samples: {len(val_labels)}')
    print(f'  Dictionary size: {len(all_chars)} characters')
    print(f'  Label files: {train_path}, {val_path}')
    print(f'  Dictionary: {dict_path}')
    if HAS_PIL:
        print(f'  Images: {IMAGES_DIR}/')
    print(f'\nReady for PaddleOCR fine-tuning.')


if __name__ == '__main__':
    main()
