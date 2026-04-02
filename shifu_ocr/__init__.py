"""
Shifu-OCR: Fluid Theory Optical Character Recognition
======================================================

A novel OCR engine built on medium displacement theory.
No neural network. No GPU. No cloud.

Core principles:
  1. Model the medium, detect displacement
  2. Characters are fluid landscapes shaped by experience
  3. Structure and content co-define each other
  4. Perturbation response reveals identity (MRI pulses)
  5. Clinical context constrains interpretation

Engines:
  - engine.py          : Core OCR via medium displacement topology
  - fluid.py           : Fluid theory landscapes (no rules, probability-shaped)
  - coherence.py       : Coherence displacement (colored backgrounds)
  - displacement.py    : Formal medium displacement theory
  - perturbation.py    : MRI-style relaxation signature recognition
  - codefining.py      : V2 bidirectional constraint solving
  - theory_revision.py : Explainable error diagnosis with theory revision
  - photoreceptor.py   : Smart per-cell adaptive binarization
  - complete.py        : Full integrated pipeline

Author: Bader & Claude — March 2026
"""

from .engine import ShifuOCR, Landscape
from .clinical import ClinicalPostProcessor
from .ensemble import ShifuEnsemble, create_ensemble, train_ensemble
from .mind import ShifuMind

__version__ = "3.0.0"
__all__ = [
    "ShifuOCR",
    "Landscape",
    "ClinicalPostProcessor",
    "ShifuEnsemble",
    "create_ensemble",
    "train_ensemble",
    "ShifuMind",
]
