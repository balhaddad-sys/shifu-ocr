"""
Fine-tune ShifuOCR with real character crops extracted from a preprocessed screenshot.

The saved model has 121-dim features but the current engine extracts more dimensions
(v2 features + perturbation signatures were added after original training).
So we first retrain the base model from fonts with the current feature extractor,
then fine-tune with real document crops.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import string
from PIL import Image, ImageDraw, ImageFont
from collections import Counter

from shifu_ocr.engine import ShifuOCR

# =========================================================================
# Step 0: Retrain base model with current feature extractor (dimension fix)
# =========================================================================
print("=== Step 0: Retraining base model with current feature dimensions ===")
ocr = ShifuOCR()

# Use same fonts and sizes as the extensive trainer
FONTS = []
import glob
for pattern in [r'C:\Windows\Fonts\*.ttf', r'C:\Windows\Fonts\*.TTF']:
    FONTS.extend(glob.glob(pattern))

# Filter to common readable fonts (avoid symbol/wingdings fonts)
GOOD_FONTS = []
for fp in FONTS:
    fname = os.path.basename(fp).lower()
    # Skip known symbol/icon fonts
    skip = any(x in fname for x in ['wingding', 'webding', 'symbol', 'marlett',
                                      'emoji', 'segmdl', 'holomdl', 'gadugi'])
    if not skip:
        GOOD_FONTS.append(fp)

# Limit to a reasonable set for speed (use diverse fonts)
PRIORITY_FONTS = []
for name in ['arial', 'calibri', 'verdana', 'times', 'cour', 'consola',
             'tahoma', 'trebuc', 'georgia', 'impact', 'comic', 'lucida',
             'segoeui', 'cambria', 'palat', 'book', 'century', 'garamond']:
    for fp in GOOD_FONTS:
        if name in os.path.basename(fp).lower() and fp not in PRIORITY_FONTS:
            PRIORITY_FONTS.append(fp)

# Use priority fonts + first N other fonts
OTHER_FONTS = [f for f in GOOD_FONTS if f not in PRIORITY_FONTS][:10]
TRAIN_FONTS = PRIORITY_FONTS + OTHER_FONTS
print(f"Training with {len(TRAIN_FONTS)} fonts")

CHARS = string.ascii_letters + string.digits
SIZES = [24, 36, 48, 64, 80]

n_trained = 0
for font_path in TRAIN_FONTS:
    try:
        # Quick validation that the font can render ASCII
        test_font = ImageFont.truetype(font_path, 36)
        test_font.getbbox('A')
    except:
        continue

    for ch in CHARS:
        for sz in SIZES:
            try:
                font = ImageFont.truetype(font_path, sz)
                bbox = font.getbbox(ch)
                if bbox is None:
                    continue
                cw = bbox[2] - bbox[0] + 10
                ht = bbox[3] - bbox[1] + 10
                s = max(cw, ht, 30)
                x = (s - (bbox[2] - bbox[0])) // 2 - bbox[0]
                y = (s - (bbox[3] - bbox[1])) // 2 - bbox[1]
                # Render at 2x and downscale for antialiasing
                img = Image.new('L', (s * 2, s * 2), 255)
                ImageDraw.Draw(img).text((x * 2, y * 2), ch, fill=0,
                                          font=ImageFont.truetype(font_path, sz * 2))
                img = img.resize((s, s), Image.LANCZOS)
                ocr.train_character(ch, np.array(img))
                n_trained += 1
            except:
                continue

print(f"Base training complete: {n_trained} samples, {len(ocr.landscapes)} landscapes")

# Reset global variance cache
ocr._global_var = None

# =========================================================================
# Step 1: Load preprocessed image
# =========================================================================
print("\n=== Step 1: Loading preprocessed screenshot ===")
from shifu_ocr.pipeline_worker import read_image
img = read_image(r'C:\Users\balha\OneDrive\Pictures\Screenshots\Screenshot 2026-03-26 165249.png')
print(f"Image loaded: shape={img.shape}, dtype={img.dtype}")

# =========================================================================
# Step 2: Segment lines and characters, get predictions
# =========================================================================
print("\n=== Step 2: Segmenting and predicting ===")
lines = ocr.segment_lines(img)
print(f"Found {len(lines)} lines")

all_crops = []
for line_seg in lines:
    chars = ocr.segment_characters(line_seg['image'])
    for ch_seg in chars:
        pred = ocr.predict_character(ch_seg['image'])
        all_crops.append({
            'image': ch_seg['image'],
            'predicted': pred['predicted'],
            'confidence': pred['confidence'],
        })

print(f"Total character crops: {len(all_crops)}")

# =========================================================================
# Step 3: Filter high-confidence and train
# =========================================================================
high_conf = [c for c in all_crops if c['confidence'] > 0.5 and c['predicted'].isalnum()]
print(f"High-confidence (>0.5) alphanumeric crops: {len(high_conf)}")

label_counts = Counter(c['predicted'] for c in high_conf)
print(f"Unique labels: {len(label_counts)}")
print(f"Top 20 labels: {label_counts.most_common(20)}")

# Confidence distribution
confs = [c['confidence'] for c in high_conf]
if confs:
    print(f"Confidence range: {min(confs):.3f} - {max(confs):.3f}, mean: {np.mean(confs):.3f}")

samples_added = 0
for crop in high_conf:
    ocr.train_character(crop['predicted'], crop['image'])
    samples_added += 1

# Reset global variance cache
ocr._global_var = None

print(f"\nReal document samples added: {samples_added}")

# =========================================================================
# Step 4: Save the model
# =========================================================================
ocr.save('shifu_ocr/trained_model.json')
print("Model saved to shifu_ocr/trained_model.json")

# =========================================================================
# Step 5: Test accuracy
# =========================================================================
print("\n=== Step 5: Testing accuracy ===")
correct = 0
t = 0
for tfp in [r'C:\Windows\Fonts\arial.ttf', r'C:\Windows\Fonts\calibri.ttf', r'C:\Windows\Fonts\verdana.ttf']:
    for ch in string.ascii_letters + string.digits:
        for sz in [36, 60]:
            f = ImageFont.truetype(tfp, sz)
            bbox = f.getbbox(ch)
            cw = bbox[2] - bbox[0] + 10
            ht = bbox[3] - bbox[1] + 10
            s = max(cw, ht, 30)
            x = (s - (bbox[2] - bbox[0])) // 2 - bbox[0]
            y = (s - (bbox[3] - bbox[1])) // 2 - bbox[1]
            img_test = Image.new('L', (s * 2, s * 2), 255)
            ImageDraw.Draw(img_test).text((x * 2, y * 2), ch, fill=0,
                                           font=ImageFont.truetype(tfp, sz * 2))
            img_test = img_test.resize((s, s), Image.LANCZOS)
            pred = ocr.predict_character(np.array(img_test))
            t += 1
            if pred['predicted'] == ch:
                correct += 1

print(f'Accuracy: {correct}/{t} = {correct/t*100:.1f}%')
