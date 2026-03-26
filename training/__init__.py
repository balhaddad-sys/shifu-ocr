"""
Shifu-OCR Training Pipeline
============================

End-to-end training data generation and model fine-tuning.

Modules:
  - shield.py       : PII/PHI redaction before cloud processing
  - harvest.py      : Seed harvesting from successful OCR scans
  - bulk_seed.py    : Synthetic ward sheet data generation (500+ records)
  - prepare.py      : Training image synthesis + label generation
  - finetune.py     : PaddleOCR recognition model fine-tuning
"""
