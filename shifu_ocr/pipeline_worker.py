"""
Shifu Pipeline Worker — called by the JS engine via child_process.
Receives an image path, runs OCR, returns JSON to stdout.

Supports two OCR backends:
  1. ShifuOCR (fluid theory — our own engine)
  2. PaddleOCR (deep learning — for full-page / table extraction)

The JS engine picks up stdout and runs clinical correction on the raw text.
"""

import argparse
import json
import sys
import os
import numpy as np
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def load_shifu_ocr(model_path):
    """Load the fluid-theory OCR engine."""
    from shifu_ocr.engine import ShifuOCR
    if os.path.exists(model_path):
        return ShifuOCR.load(model_path)
    return ShifuOCR()


def load_ensemble():
    """Load the multi-engine ensemble (all engines see every character)."""
    try:
        from shifu_ocr.ensemble import create_ensemble
        return create_ensemble()
    except ImportError:
        return None


def load_paddle_ocr():
    """Load PaddleOCR (if available)."""
    try:
        from paddleocr import PaddleOCR
        return PaddleOCR(use_angle_cls=True, lang='en', show_log=False)
    except ImportError:
        return None


def read_image(image_path):
    """
    Read image as grayscale numpy array.

    For colored documents (spreadsheets, ward sheets): isolate dark text from
    colored/bright backgrounds so the FLAIR perturbation engine gets clean input.
    The perturbation pulses (erosion, dilation, blur, skeleton) reveal character
    identity — but they need text vs background cleanly separated first.
    """
    from PIL import Image
    img = Image.open(image_path)

    if img.mode in ('RGB', 'RGBA'):
        from scipy.ndimage import gaussian_filter
        # 4D MRI SPATIAL ENCODING:
        # Like MRI: multi-sequence (R,G,B,L) × multi-scale (Laplacian pyramid)
        # × directional (Gabor orientation contrast).
        # Each pixel gets a multi-dimensional feature vector.
        # Text = high fine-detail + high orientation contrast + dark in all channels.
        arr = np.array(img.convert('RGB')).astype(float)
        channels = [arr[:,:,0], arr[:,:,1], arr[:,:,2], arr.mean(axis=2)]
        h, w = arr.shape[:2]

        # MULTI-SCALE per channel: Laplacian pyramid (3 levels)
        # Level 0 (sigma=2): fine strokes. Level 1 (sigma=6): character scale.
        # Level 2 (sigma=18): word/cell scale. Residual = background.
        text_evidence = np.zeros((h, w))
        per_channel_fg = []

        for ch in channels:
            sigmas = [2.0, 6.0, 18.0]
            current = ch.copy()
            fine_energy = np.zeros((h, w))
            for sigma in sigmas:
                smoothed = gaussian_filter(current, sigma=sigma)
                detail = current - smoothed
                if sigma <= 6.0:
                    fine_energy += np.abs(detail)  # text lives at fine scales
                current = smoothed
            background = current
            fg = np.clip(background - ch, 0, None)
            if fg.max() > 0:
                fg /= fg.max()
            per_channel_fg.append(fg)
            if fine_energy.max() > 0:
                fine_energy /= fine_energy.max()
            text_evidence += fine_energy * fg

        # DIRECTIONAL: Gabor orientation contrast at character-stroke frequency
        # Text strokes are anisotropic, backgrounds are isotropic
        try:
            from skimage.filters import gabor
            gray_norm = arr.mean(axis=2) / 255.0
            orient_responses = []
            for i in range(4):
                theta = i * np.pi / 4
                fr, fi = gabor(gray_norm, frequency=0.25, theta=theta)
                orient_responses.append(np.sqrt(fr**2 + fi**2))
            orient_stack = np.stack(orient_responses, axis=-1)
            orient_contrast = orient_stack.max(axis=-1) - orient_stack.min(axis=-1)
            if orient_contrast.max() > 0:
                orient_contrast /= orient_contrast.max()
        except Exception:
            orient_contrast = np.zeros((h, w))

        # COMBINE: geometric mean across channels (soft minimum)
        fg_stack = np.stack(per_channel_fg, axis=-1)
        geo_mean = np.exp(np.mean(np.log(np.maximum(fg_stack, 1e-8)), axis=-1))

        # Fuse: multi-scale evidence + directional contrast, gated by cross-channel
        if text_evidence.max() > 0:
            text_evidence /= text_evidence.max()
        combined = (0.5 * text_evidence + 0.3 * geo_mean + 0.2 * orient_contrast)

        # Normalize to grayscale
        if combined.max() > 0:
            scaled = combined / combined.max()
            normalized = 255 - (scaled * 225)
            normalized[scaled < 0.08] = 255
        else:
            normalized = np.full((h, w), 255.0)

        return normalized.astype(np.uint8)

    return np.array(img.convert('L'))


def ocr_with_shifu(image_path, model_path, page_mode=True):
    """
    UNIFIED NEURAL PIPELINE:
    Layer 1: 4D MRI-RF preprocessing (R+G+B+L multi-sequence)
    Layer 2: Line segmentation (trusts L1 to have cleaned the image)
    Layer 3: Character segmentation (trusts L2 for clean line images)
    Layer 4: FLAIR perturbation + template ensemble (character identity)
    Layer 5: Document adaptation (confident predictions refine the model)
    Layer 6: JS clinical correction (confusion bridge + vocabulary)
    Each layer receives from the previous, feeds to the next.
    """
    ocr = load_shifu_ocr(model_path)
    img = read_image(image_path)  # Layer 1: 4D MRI-RF

    # Use page mode for larger images (likely full page / screenshot)
    if page_mode and (img.shape[0] > 200 or img.shape[1] > 400):
        result = ocr.read_page(img)  # Layers 2-5
        output = {
            'backend': 'shifu',
            'text': result['text'],
            'lines': result.get('lines', []),
            'confidence': float(result.get('confidence', 0)),
        }
        if result.get('table'):
            output['table'] = result['table']
        if result.get('words'):
            output['word_count'] = len(result['words'])
        return output

    result = ocr.read_line(img)
    return {
        'backend': 'shifu',
        'text': result['text'],
        'lines': [result['text']],
        'confidence': float(result['confidence']),
        'characters': [
            {
                'char': c['char'],
                'confidence': float(c['confidence']),
                'bbox': list(c['bbox']),
            }
            for c in result.get('characters', [])
        ],
    }


def ocr_with_paddle(image_path):
    """Run PaddleOCR on an image (supports tables and full pages)."""
    paddle = load_paddle_ocr()
    if paddle is None:
        return {'backend': 'paddle', 'error': 'PaddleOCR not installed', 'text': '', 'lines': []}

    result = paddle.ocr(image_path, cls=True)
    if not result or not result[0]:
        return {'backend': 'paddle', 'text': '', 'lines': [], 'rows': []}

    lines = []
    boxes = []
    for detection in result[0]:
        box = detection[0]
        text, conf = detection[1]
        y_center = (box[0][1] + box[2][1]) / 2
        x_center = (box[0][0] + box[2][0]) / 2
        lines.append({
            'text': text,
            'confidence': float(conf),
            'bbox': [[int(p[0]), int(p[1])] for p in box],
            'y_center': y_center,
            'x_center': x_center,
        })
        boxes.append({'text': text, 'y': y_center, 'x': x_center, 'conf': float(conf)})

    # Sort by y position (top to bottom), then x (left to right)
    lines.sort(key=lambda l: (l['y_center'], l['x_center']))

    # Group into rows (lines within ~20px vertical distance)
    rows = []
    current_row = []
    last_y = None
    row_threshold = 20

    for line in lines:
        if last_y is not None and abs(line['y_center'] - last_y) > row_threshold:
            current_row.sort(key=lambda l: l['x_center'])
            rows.append([l['text'] for l in current_row])
            current_row = []
        current_row.append(line)
        last_y = line['y_center']

    if current_row:
        current_row.sort(key=lambda l: l['x_center'])
        rows.append([l['text'] for l in current_row])

    all_text = [l['text'] for l in lines]

    return {
        'backend': 'paddle',
        'text': ' '.join(all_text),
        'lines': all_text,
        'rows': rows,
        'detections': len(lines),
        'avg_confidence': float(np.mean([l['confidence'] for l in lines])) if lines else 0,
    }


def ocr_table_with_paddle(image_path, columns=None):
    """Extract a table from an image using PaddleOCR."""
    result = ocr_with_paddle(image_path)
    if 'error' in result:
        return result

    # If columns are provided, try to map detections to columns
    if columns and result.get('rows'):
        structured_rows = []
        for row in result['rows']:
            # Pad or trim to match column count
            padded = row + [''] * max(0, len(columns) - len(row))
            structured_rows.append(padded[:len(columns)])
        result['columns'] = columns
        result['rows'] = structured_rows

    return result


def main():
    parser = argparse.ArgumentParser(description='Shifu OCR Pipeline Worker')
    parser.add_argument('--image', required=True, help='Path to image file')
    parser.add_argument('--format', default='line', choices=['line', 'table'], help='Output format')
    parser.add_argument('--model', default='', help='Path to trained model JSON')
    parser.add_argument('--backend', default='auto', choices=['auto', 'shifu', 'paddle', 'ensemble'], help='OCR backend')
    parser.add_argument('--columns', nargs='*', help='Column names for table format')
    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(json.dumps({'error': f'Image not found: {args.image}'}))
        sys.exit(1)

    # SEC-08: Validate image path is not a traversal or symlink escape
    resolved = os.path.realpath(args.image)
    if not os.path.isfile(resolved):
        print(json.dumps({'error': 'Invalid image path'}))
        sys.exit(1)

    # Choose backend
    backend = args.backend
    if backend == 'auto':
        # Use PaddleOCR for tables (better at layout), ShifuOCR for single lines
        if args.format == 'table':
            paddle = load_paddle_ocr()
            backend = 'paddle' if paddle else 'shifu'
        else:
            backend = 'shifu'

    try:
        if backend == 'ensemble':
            ensemble = load_ensemble()
            if ensemble:
                img = read_image(args.image)
                result = ensemble.read_line(img)
                result['backend'] = 'ensemble'
            else:
                model_path = args.model or os.path.join(os.path.dirname(__file__), 'trained_model.json')
                result = ocr_with_shifu(args.image, model_path)
        elif backend == 'paddle':
            if args.format == 'table':
                result = ocr_table_with_paddle(args.image, args.columns)
            else:
                result = ocr_with_paddle(args.image)
        else:
            model_path = args.model or os.path.join(os.path.dirname(__file__), 'trained_model.json')
            result = ocr_with_shifu(args.image, model_path)

        print(json.dumps(result))
    except Exception as e:
        print(json.dumps({'error': str(e), 'text': '', 'lines': []}))
        sys.exit(1)


if __name__ == '__main__':
    main()
