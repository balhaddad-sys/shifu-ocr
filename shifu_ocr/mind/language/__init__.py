"""
shifu_ocr.mind.language — Full Language Acquisition

The baby learns language from scratch:
  Morphology → Syntax → Semantics → Pragmatics

Every module learns from data. Nothing hardcoded.
"""

from .morphology import Morphology
from .syntax import Syntax
from .semantics import Semantics
from .curriculum import Curriculum

__all__ = ['Morphology', 'Syntax', 'Semantics', 'Curriculum']
