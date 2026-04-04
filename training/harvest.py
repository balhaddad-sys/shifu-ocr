#!/usr/bin/env python3
"""
MedTriage Seed Harvester — Saves OCR results as training pairs.
Every successful scan becomes seed data for future model training.

Usage:
    python harvest.py --input shielded_text.txt --output structured.json
    python harvest.py --scan raw_scans/ward_sheet.jpg
"""
import os
import json
import hashlib
from datetime import datetime

SEED_BANK = os.path.join(os.path.dirname(__file__), 'seed_bank')
os.makedirs(SEED_BANK, exist_ok=True)


def save_seed(input_text, structured_output, metadata=None):
    """Save a training pair to the seed bank."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    content_hash = hashlib.md5(input_text.encode()).hexdigest()[:8]

    seed = {
        'id': f'seed_{timestamp}_{content_hash}',
        'timestamp': datetime.now().isoformat(),
        'input_text': input_text,
        'structured_output': structured_output,
        'metadata': metadata or {},
        'version': 'medtriage-ocr-v4',
    }

    filename = f'seed_{timestamp}_{content_hash}.json'
    filepath = os.path.join(SEED_BANK, filename)

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(seed, f, ensure_ascii=False, indent=2)

    print(f'Seed saved: {filename}')
    return filepath


def count_seeds():
    """Count total seeds in the bank."""
    files = [f for f in os.listdir(SEED_BANK) if f.endswith('.json')]
    return len(files)


def export_training_data(output_path=None):
    """Export all seeds as a single JSONL file for model training."""
    output_path = output_path or os.path.join(SEED_BANK, 'training_export.jsonl')
    files = sorted(f for f in os.listdir(SEED_BANK) if f.startswith('seed_') and f.endswith('.json'))

    count = 0
    with open(output_path, 'w', encoding='utf-8') as out:
        for filename in files:
            filepath = os.path.join(SEED_BANK, filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                seed = json.load(f)
            # JSONL format: one JSON object per line
            training_record = {
                'input': seed['input_text'],
                'output': json.dumps(seed['structured_output']) if isinstance(seed['structured_output'], dict) else seed['structured_output'],
            }
            out.write(json.dumps(training_record, ensure_ascii=False) + '\n')
            count += 1

    print(f'Exported {count} training records to {output_path}')
    return output_path


if __name__ == '__main__':
    import sys

    if '--count' in sys.argv:
        print(f'Seed bank contains {count_seeds()} records')
    elif '--export' in sys.argv:
        export_training_data()
    elif '--input' in sys.argv and '--output' in sys.argv:
        input_idx = sys.argv.index('--input') + 1
        output_idx = sys.argv.index('--output') + 1
        with open(sys.argv[input_idx], 'r', encoding='utf-8') as f:
            input_text = f.read()
        with open(sys.argv[output_idx], 'r', encoding='utf-8') as f:
            output_data = json.load(f)
        save_seed(input_text, output_data)
    else:
        print('Usage:')
        print('  python harvest.py --count')
        print('  python harvest.py --export')
        print('  python harvest.py --input text.txt --output result.json')
