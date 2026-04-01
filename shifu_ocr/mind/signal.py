"""
Signal — reward system based on prediction error.

The dopamine equivalent. Fires on SURPRISE, not on expected reward.
Drives self-tuning: when predictions are consistently wrong,
the system recalibrates.

Uses TD-like learning: predict expected quality for a situation,
observe actual quality, update via prediction error.
"""

from __future__ import annotations
import math
from typing import Dict, List, Optional, Tuple, Any


class Signal:
    """
    Prediction error signal for self-tuning.

    predict(state) → expected reward
    observe(state, actual) → prediction error
    surprise(state, actual) → absolute error magnitude

    Policies accumulate per-state statistics. States that consistently
    produce bad outcomes get their expectations recalibrated.
    """

    def __init__(
        self,
        learning_rate: float = 0.1,
        discount: float = 0.95,
        recalibrate_threshold: int = 5,
        recalibrate_quality: float = 0.2,
    ):
        self._lr = learning_rate
        self._discount = discount
        self._recal_threshold = recalibrate_threshold
        self._recal_quality = recalibrate_quality

        # State predictions: state_key -> expected quality
        self._predictions: Dict[str, float] = {}
        # State statistics: state_key -> {count, total_quality, avg_quality}
        self._policies: Dict[str, dict] = {}
        # History of recent observations
        self._history: List[dict] = []
        # Global running statistics
        self._total_observations = 0
        self._total_quality = 0.0

    def predict(self, state: str) -> float:
        """Expected quality for this state. 0.5 if never seen."""
        return self._predictions.get(state, 0.5)

    def observe(self, state: str, actual_quality: float) -> dict:
        """
        Observe actual outcome. Compute prediction error and update.

        Returns: {
            'state': str,
            'predicted': float,
            'actual': float,
            'error': float (-1 to +1, positive = better than expected),
        }
        """
        predicted = self.predict(state)
        error = actual_quality - predicted

        # TD update
        self._predictions[state] = predicted + self._lr * error

        # Update policy statistics
        if state not in self._policies:
            self._policies[state] = {
                'count': 0, 'total_quality': 0.0, 'avg_quality': 0.5,
            }
        pol = self._policies[state]
        pol['count'] += 1
        pol['total_quality'] += actual_quality
        pol['avg_quality'] = pol['total_quality'] / pol['count']

        # Recalibrate failed policies
        if (pol['count'] >= self._recal_threshold
                and pol['avg_quality'] < self._recal_quality):
            pol['avg_quality'] = 0.0
            pol['count'] = max(1, pol['count'] // 2)
            pol['total_quality'] = pol['avg_quality'] * pol['count']
            self._predictions[state] = 0.3  # Reset expectation

        # Global tracking
        self._total_observations += 1
        self._total_quality += actual_quality

        result = {
            'state': state,
            'predicted': predicted,
            'actual': actual_quality,
            'error': error,
        }
        self._history.append(result)
        if len(self._history) > 500:
            self._history = self._history[-250:]
        return result

    def surprise(self, state: str, actual_quality: float) -> float:
        """Absolute prediction error magnitude."""
        return abs(actual_quality - self.predict(state))

    def recent_trend(self, n: int = 5) -> float:
        """Average quality of last N observations."""
        if not self._history:
            return 0.5
        recent = self._history[-n:]
        return sum(h['actual'] for h in recent) / len(recent)

    def global_average(self) -> float:
        if self._total_observations == 0:
            return 0.5
        return self._total_quality / self._total_observations

    def to_dict(self) -> dict:
        return {
            'lr': self._lr,
            'discount': self._discount,
            'predictions': self._predictions,
            'policies': self._policies,
            'history': self._history[-100:],
            'total_observations': self._total_observations,
            'total_quality': self._total_quality,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Signal:
        s = cls(
            learning_rate=d.get('lr', 0.1),
            discount=d.get('discount', 0.95),
        )
        s._predictions = d.get('predictions', {})
        s._policies = d.get('policies', {})
        s._history = d.get('history', [])
        s._total_observations = d.get('total_observations', 0)
        s._total_quality = d.get('total_quality', 0.0)
        return s
