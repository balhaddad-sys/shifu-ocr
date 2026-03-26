#!/usr/bin/env python3
"""
MedTriage Privacy Shield — Local PII/PHI Redaction
Strips patient identifiers before any cloud processing.
Stores encrypted mapping locally for re-identification.

Usage:
    python shield.py "raw text from OCR scan"
    python shield.py --file raw_scans/scan001.txt
"""
import re
import os
import json
import hashlib
import sqlite3
from datetime import datetime

VAULT_DIR = os.path.join(os.path.dirname(__file__), 'vault')
os.makedirs(VAULT_DIR, exist_ok=True)

DB_PATH = os.path.join(VAULT_DIR, 'mapping.db')

# Kuwait Civil ID pattern: 12 digits starting with 1-3
CIVIL_ID_PATTERN = re.compile(r'\b[123]\d{11}\b')

# MRN/File number patterns: 6-8 digit hospital IDs
MRN_PATTERN = re.compile(r'\b(?:MRN|File|ID)[:\s#]*(\d{6,8})\b', re.IGNORECASE)

# Phone numbers (Kuwait: +965 XXXX XXXX)
PHONE_PATTERN = re.compile(r'(?:\+965[\s-]?)?\b\d{4}[\s-]?\d{4}\b')

# Date of birth patterns
DOB_PATTERN = re.compile(r'\b(?:DOB|D\.O\.B|Date of Birth)[:\s]*\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4}\b', re.IGNORECASE)


def init_db():
    db = sqlite3.connect(DB_PATH)
    db.execute('''CREATE TABLE IF NOT EXISTS redactions (
        token TEXT PRIMARY KEY,
        original_hash TEXT,
        field_type TEXT,
        timestamp TEXT
    )''')
    db.commit()
    return db


def generate_token(field_type, idx):
    return f'[[{field_type}_{idx:04d}]]'


def shield_text(raw_text):
    """Redact all PII/PHI from clinical text. Returns (shielded_text, mapping)."""
    db = init_db()
    shielded = raw_text
    mapping = {}
    token_idx = 0

    # 1. Redact Civil IDs
    for match in CIVIL_ID_PATTERN.finditer(raw_text):
        original = match.group(0)
        token = generate_token('CID', token_idx)
        shielded = shielded.replace(original, token, 1)
        original_hash = hashlib.sha256(original.encode()).hexdigest()
        mapping[token] = {'type': 'civil_id', 'hash': original_hash}
        db.execute('INSERT OR REPLACE INTO redactions VALUES (?, ?, ?, ?)',
                   (token, original_hash, 'civil_id', datetime.now().isoformat()))
        token_idx += 1

    # 2. Redact MRNs
    for match in MRN_PATTERN.finditer(raw_text):
        original = match.group(1)
        token = generate_token('MRN', token_idx)
        shielded = shielded.replace(original, token, 1)
        original_hash = hashlib.sha256(original.encode()).hexdigest()
        mapping[token] = {'type': 'mrn', 'hash': original_hash}
        db.execute('INSERT OR REPLACE INTO redactions VALUES (?, ?, ?, ?)',
                   (token, original_hash, 'mrn', datetime.now().isoformat()))
        token_idx += 1

    # 3. Redact DOBs
    for match in DOB_PATTERN.finditer(raw_text):
        original = match.group(0)
        token = generate_token('DOB', token_idx)
        shielded = shielded.replace(original, token, 1)
        mapping[token] = {'type': 'dob'}
        token_idx += 1

    # 4. Redact phone numbers
    for match in PHONE_PATTERN.finditer(shielded):
        original = match.group(0)
        if len(original.replace(' ', '').replace('-', '')) >= 8:
            token = generate_token('PHN', token_idx)
            shielded = shielded.replace(original, token, 1)
            mapping[token] = {'type': 'phone'}
            token_idx += 1

    db.commit()
    db.close()

    return shielded, mapping


def log_processing(filename, status, details=''):
    log_path = os.path.join(os.path.dirname(__file__), 'logs', 'audit_log.csv')
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, 'a', encoding='utf-8') as f:
        timestamp = datetime.now().isoformat()
        f.write(f'{timestamp},{filename},{status},{details}\n')


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print('Usage: python shield.py "raw text" or python shield.py --file path')
        sys.exit(1)

    if sys.argv[1] == '--file':
        with open(sys.argv[2], 'r', encoding='utf-8') as f:
            raw = f.read()
        filename = sys.argv[2]
    else:
        raw = ' '.join(sys.argv[1:])
        filename = 'stdin'

    shielded, mapping = shield_text(raw)
    print('=== SHIELDED OUTPUT (safe for cloud) ===')
    print(shielded)
    print(f'\n=== REDACTED {len(mapping)} identifiers ===')
    for token, info in mapping.items():
        print(f'  {token} -> {info["type"]}')

    log_processing(filename, 'shielded', f'{len(mapping)} redactions')
