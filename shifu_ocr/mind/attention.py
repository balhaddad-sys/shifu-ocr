"""
Attention — relevance-gated activation.

The brain doesn't activate everything equally. Attention gates
which activations pass through based on:

1. SALIENCE: how surprising is this activation? (prediction error)
2. RELEVANCE: does this connect to the current goal?
3. INHIBITION OF RETURN: don't re-attend what was just attended

No hardcoded weights. The gates learn from the landscape.

Like the thalamic reticular nucleus: a thin shell of inhibitory
neurons that gates ALL information flowing to the cortex.
Only what passes the gate reaches conscious processing.
"""

from __future__ import annotations
import math
from typing import Dict, List, Set, Optional, Tuple


class Attention:
    """
    Thalamic gate: filters activations by salience and relevance.

    attend(activations, goal_context, history) → gated activations
    """

    def __init__(self):
        # Inhibition of return: recently attended words get dampened
        self._attended_history: List[Set[str]] = []
        self._max_history = 5
        # Salience memory: expected activation per word
        self._expected: Dict[str, float] = {}
        self._expected_lr = 0.1

    def attend(
        self,
        activations: Dict[str, float],
        goal_words: Optional[List[str]] = None,
        co_graph: Optional[Dict[str, Dict[str, float]]] = None,
    ) -> Dict[str, float]:
        """
        Gate activations by salience and relevance.

        activations: {word: energy} from the field
        goal_words: what we're trying to find (boosts relevant)
        co_graph: for computing relevance to goal

        Returns: gated activations — same keys, modified values
        """
        if not activations:
            return {}

        # ═══ SALIENCE: surprise amplifies, expectation dampens ═══
        # Words that activate MORE than expected are salient.
        # Words that activate AS expected are boring.
        salience: Dict[str, float] = {}
        for w, energy in activations.items():
            expected = self._expected.get(w, 0.0)
            surprise = energy - expected
            # Positive surprise → amplify. Negative → dampen.
            salience[w] = energy * (1.0 + max(0, surprise))
            # Update expectation
            self._expected[w] = expected + self._expected_lr * (energy - expected)

        # ═══ RELEVANCE: boost words connected to goal ═══
        if goal_words and co_graph:
            goal_set = set(goal_words)
            for w in salience:
                # How connected is this word to the goal words?
                if w in goal_set:
                    salience[w] *= 2.0  # Direct match
                    continue
                neighbors = co_graph.get(w, {})
                goal_overlap = sum(1 for g in goal_words if g in neighbors)
                if goal_overlap > 0:
                    salience[w] *= (1.0 + goal_overlap * 0.3)

        # ═══ INHIBITION OF RETURN: dampen recently attended ═══
        for i, prev_attended in enumerate(reversed(self._attended_history)):
            decay = 0.5 ** (i + 1)  # Most recent → strongest inhibition
            for w in prev_attended:
                if w in salience:
                    salience[w] *= (1.0 - decay * 0.5)

        # Record what we attended this cycle
        top_attended = set(
            w for w, _ in sorted(salience.items(), key=lambda x: -x[1])[:10]
        )
        self._attended_history.append(top_attended)
        if len(self._attended_history) > self._max_history:
            self._attended_history.pop(0)

        return salience

    def reset(self) -> None:
        """Reset attention state between conversations."""
        self._attended_history.clear()

    def to_dict(self) -> dict:
        return {
            'expected': dict(list(self._expected.items())[:500]),
        }

    @classmethod
    def from_dict(cls, d: dict) -> Attention:
        a = cls()
        a._expected = d.get('expected', {})
        return a
