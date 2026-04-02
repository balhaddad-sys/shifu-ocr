"""
Unified Landscape — the universal learning primitive.

Welford's online algorithm: O(1) per observation, O(d) memory.
Features that are consistent across observations become NARROW (discriminative).
Features that vary become WIDE (ignored).
No hand-tuned weights. The data shapes the landscape.

This is the canonical implementation. Both OCR feature vectors and
cognitive word vectors use the same absorb/fit contract.
"""

from __future__ import annotations
import math
from collections import defaultdict
from typing import Optional, Dict, Any

try:
    import numpy as np
except ImportError:
    np = None


class Landscape:
    """
    A probability terrain for one concept/character/entity.

    absorb(fv) — learn from a new observation
    fit(fv) — how well does this observation match?

    The landscape converges: mean → centroid, variance → minimum spread.
    Variance floor shrinks with experience but stays positive
    (pink cats can exist).
    """

    __slots__ = (
        'label', 'mean', 'variance', '_m2', 'n', 'expected_dim',
        'n_correct', 'n_errors', 'confused_with',
    )

    def __init__(self, label: str, expected_dim: Optional[int] = None):
        self.label = label
        self.mean = None
        self.variance = None
        self._m2 = None
        self.n = 0
        self.expected_dim = expected_dim
        self.n_correct = 0
        self.n_errors = 0
        self.confused_with: Dict[str, int] = defaultdict(int)

    def absorb(self, fv) -> None:
        """
        Absorb a new observation into the landscape.
        Uses Welford's online algorithm for numerically stable
        running mean and variance.
        """
        if np is None:
            raise RuntimeError("numpy required for Landscape.absorb")

        fv = np.asarray(fv, dtype=float)

        if self.expected_dim is None:
            self.expected_dim = len(fv)
        elif len(fv) != self.expected_dim:
            raise ValueError(
                f"Landscape '{self.label}': expected {self.expected_dim}-dim, "
                f"got {len(fv)}-dim."
            )

        self.n += 1
        if self.n == 1:
            self.mean = fv.copy()
            self._m2 = np.zeros_like(fv)
            self.variance = np.ones_like(fv) * 2.0
        else:
            delta = fv - self.mean
            self.mean += delta / self.n
            delta2 = fv - self.mean
            self._m2 += delta * delta2
            raw_var = self._m2 / self.n
            # Variance floor shrinks with experience but stays positive
            self.variance = np.maximum(raw_var, 0.1 / math.sqrt(self.n))

    def fit(self, fv, global_var=None) -> float:
        """
        Score how well an observation fits this landscape.
        Returns log-likelihood-like score (higher = better match).

        Features where the landscape is NARROW (consistent) dominate.
        Features where it is WIDE (variable) are effectively ignored.
        """
        if self.mean is None:
            return -float('inf')

        if np is None:
            raise RuntimeError("numpy required for Landscape.fit")

        fv = np.asarray(fv, dtype=float)
        diff = fv - self.mean
        var = (
            np.minimum(self.variance, global_var)
            if global_var is not None
            else self.variance
        )
        precision = 1.0 / (var + 1e-8)
        score = -0.5 * np.sum(diff ** 2 * precision)
        # Confidence bonus: more observations → more trustworthy
        return score + math.log(self.n + 1) * 0.5

    def to_dict(self) -> dict:
        return {
            'label': self.label,
            'n': self.n,
            'expected_dim': self.expected_dim,
            'mean': self.mean.tolist() if self.mean is not None else None,
            'variance': self.variance.tolist() if self.variance is not None else None,
            'n_correct': self.n_correct,
            'n_errors': self.n_errors,
            'confused_with': dict(self.confused_with),
        }

    @classmethod
    def from_dict(cls, d: dict) -> Landscape:
        ls = cls(d['label'], expected_dim=d.get('expected_dim'))
        ls.n = d['n']
        if d.get('mean') is not None and np is not None:
            ls.mean = np.array(d['mean'])
            ls.expected_dim = len(ls.mean)
        if d.get('variance') is not None and np is not None:
            ls.variance = np.array(d['variance'])
            if ls.n > 0:
                ls._m2 = ls.variance * ls.n
        ls.n_correct = d.get('n_correct', 0)
        ls.n_errors = d.get('n_errors', 0)
        ls.confused_with = defaultdict(int, d.get('confused_with', {}))
        return ls
