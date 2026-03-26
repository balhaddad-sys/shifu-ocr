"""
Shifu OCR — Learn from Confusion

The principle: confusion is signal, not noise.
When the engine reads 'r' as 'L', that tells us the landscapes for r and L
overlap in feature space. The fix is not more data — it's targeted data
that sharpens the boundary between confused characters.

This script:
1. Tests the model → collects every confusion (predicted vs actual)
2. Generates targeted training images for confused pairs
3. Retrains with extra weight on confused characters
4. Tests again → measures improvement
5. Repeats until accuracy plateaus or target reached

Each round, the engine gets better at exactly what it was worst at.
Structure from chaos. Chaos from structure.
"""

import os, sys, glob, time, json
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from collections import defaultdict

sys.path.insert(0, str(os.path.join(os.path.dirname(__file__), '..')))
from shifu_ocr.engine import ShifuOCR

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'trained_model.json')
TRAINING_DIR = 'c:/Users/balha/OneDrive/Desktop/Application/MedTriage/med_ocr/training_data'

TESTS = [
    'stroke patient admitted',
    'Levetiracetam 500mg daily',
    'Dr Hisham Saleh',
    'bilateral weakness noted',
    'chest infection pneumonia',
    'seizure epilepsy status',
    'potassium 4.5 sodium 139',
    'aspirin clopidogrel warfarin',
    'Abdullah Mohammed Hassan',
    'pneumonia respiratory failure',
    'metoprolol bisoprolol atenolol',
    'ischemic hemorrhagic stroke',
    'creatinine 89 urea 5.2',
    'discharge home stable',
    'insulin glargine 20 units',
    'blood pressure heart rate',
    'atrial fibrillation rhythm',
    'Noura Bazzah ward 3',
    'paracetamol ibuprofen codeine',
    'meningitis encephalitis',
]

ALLOWED = set('abcdefghijklmnopqrstuvwxyz0123456789')


def find_fonts(max_fonts=50):
    """Find a diverse set of fonts for testing and retraining."""
    font_dirs = ['C:/Windows/Fonts', os.path.expanduser('~/AppData/Local/Microsoft/Windows/Fonts')]
    all_fonts = []
    for d in font_dirs:
        if os.path.exists(d):
            all_fonts.extend(glob.glob(os.path.join(d, '*.ttf')))
            all_fonts.extend(glob.glob(os.path.join(d, '*.TTF')))
    seen = set()
    usable = []
    for f in sorted(all_fonts):
        base = os.path.basename(f).lower()
        if base in seen: continue
        seen.add(base)
        try:
            font = ImageFont.truetype(f, 36)
            img = Image.new('L', (100, 50), 255)
            draw = ImageDraw.Draw(img)
            draw.text((5, 5), 'Abc', fill=0, font=font)
            if (np.array(img) < 128).sum() > 20:
                usable.append(f)
        except:
            pass
    if len(usable) > max_fonts:
        step = len(usable) // max_fonts
        usable = usable[::step][:max_fonts]
    return usable


def test_model(ocr, fonts, tests):
    """Test the model and collect confusions."""
    confusions = defaultdict(lambda: defaultdict(int))  # confusions[true_char][predicted_char] = count
    total_correct = 0
    total_chars = 0

    for fp in fonts:
        try:
            font = ImageFont.truetype(fp, 36)
        except:
            continue
        for text in tests:
            img = Image.new('L', (700, 60), color=255)
            draw = ImageDraw.Draw(img)
            draw.text((10, 10), text, fill=0, font=font)

            result = ocr.read_line(np.array(img))
            pred = result['text'].lower()
            truth = text.lower()

            # Align by removing spaces
            p = pred.replace(' ', '')
            t = truth.replace(' ', '')
            for i in range(min(len(p), len(t))):
                total_chars += 1
                if p[i] == t[i]:
                    total_correct += 1
                else:
                    if t[i] in ALLOWED:
                        confusions[t[i]][p[i]] += 1

    accuracy = total_correct / max(total_chars, 1)
    return accuracy, confusions, total_chars


def retrain_confused(ocr, confusions, fonts, rounds_per_char=20):
    """Retrain on confused characters with extra weight."""
    # Sort confusions by frequency
    confused_pairs = []
    for true_char, preds in confusions.items():
        for pred_char, count in preds.items():
            confused_pairs.append((true_char, pred_char, count))
    confused_pairs.sort(key=lambda x: x[2], reverse=True)

    if not confused_pairs:
        print('  No confusions to train on')
        return 0

    # Focus on top confused characters
    top_confused = set()
    for true_char, pred_char, count in confused_pairs[:30]:
        top_confused.add(true_char)

    print(f'  Retraining {len(top_confused)} confused characters: {sorted(top_confused)}')
    print(f'  Top confusions: {[(t,p,c) for t,p,c in confused_pairs[:10]]}')

    trained = 0
    sizes = [24, 32, 40, 48, 64]

    for char in top_confused:
        for fp in fonts:
            try:
                font = ImageFont.truetype(fp, np.random.choice(sizes))
            except:
                continue
            for _ in range(rounds_per_char):
                img = Image.new('L', (100, 100), color=255)
                draw = ImageDraw.Draw(img)
                try:
                    bbox = draw.textbbox((0, 0), char, font=font)
                    x = (100 - (bbox[2] - bbox[0])) // 2 - bbox[0]
                    y = (100 - (bbox[3] - bbox[1])) // 2 - bbox[1]
                    draw.text((x, y), char, fill=0, font=font)
                except:
                    continue
                arr = np.array(img)
                if (arr < 128).sum() < 10:
                    continue
                # Also train uppercase mapping
                label = char.lower()
                try:
                    ocr.train_character(label, arr)
                    trained += 1
                except:
                    pass

    return trained


def main():
    max_rounds = 20
    target_accuracy = 0.50  # Stop when we hit 50% char accuracy

    print('Loading model...')
    ocr = ShifuOCR.load(MODEL_PATH)
    stats = ocr.get_stats()
    print(f'Model: {stats["characters"]} landscapes, avg depth {stats["avg_depth"]:.0f}')

    fonts = find_fonts(50)
    print(f'Using {len(fonts)} fonts for testing')

    best_accuracy = 0
    history = []

    for round_num in range(1, max_rounds + 1):
        print(f'\n=== ROUND {round_num}/{max_rounds} ===', flush=True)

        # Test
        t0 = time.time()
        accuracy, confusions, total_chars = test_model(ocr, fonts, TESTS)
        test_time = time.time() - t0
        print(f'  Accuracy: {100*accuracy:.1f}% ({total_chars} chars, {test_time:.0f}s)', flush=True)

        history.append(accuracy)

        if accuracy >= target_accuracy:
            print(f'\n  TARGET REACHED: {100*accuracy:.1f}% >= {100*target_accuracy:.0f}%')
            break

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            print(f'  NEW BEST: {100*best_accuracy:.1f}%')
            # Save checkpoint
            ocr.save(MODEL_PATH)
            print(f'  Saved checkpoint')

        # Check for plateau (no improvement in 3 rounds)
        if len(history) >= 4 and max(history[-3:]) - min(history[-3:]) < 0.005:
            print(f'\n  PLATEAU: accuracy stable at ~{100*accuracy:.1f}%')
            break

        # Retrain on confusions
        t0 = time.time()
        trained = retrain_confused(ocr, confusions, fonts)
        train_time = time.time() - t0
        print(f'  Retrained: {trained} observations ({train_time:.0f}s)', flush=True)

    # Final save
    ocr.save(MODEL_PATH)
    stats = ocr.get_stats()

    print(f'\n========================================')
    print(f'  CONFUSION LEARNING COMPLETE')
    print(f'========================================')
    print(f'  Rounds: {len(history)}')
    print(f'  Accuracy: {" -> ".join(f"{100*a:.1f}%" for a in history)}')
    print(f'  Improvement: {100*(history[-1]-history[0]):.1f}%')
    print(f'  Landscapes: {stats["characters"]}, avg depth: {stats["avg_depth"]:.0f}')
    print(f'========================================')


if __name__ == '__main__':
    main()
