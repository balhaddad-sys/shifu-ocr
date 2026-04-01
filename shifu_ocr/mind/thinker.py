"""
Thinker — deliberative reasoning loop.

Multi-step reasoning: activate, evaluate, use prediction error
to decide whether to continue or backtrack.

Takes callable functions as arguments — fully decoupled from
specific module implementations. The thinker doesn't know WHERE
knowledge comes from, only how to reason WITH it.
"""

from __future__ import annotations
from collections import deque
from typing import Dict, List, Optional, Callable, Any, Tuple


class WorkingMemory:
    """FIFO buffer with configurable capacity."""

    def __init__(self, capacity: int = 7):
        self._capacity = capacity
        self._items: deque = deque(maxlen=capacity)
        self.focus: List[str] = []
        self.situation: str = 'general'
        self.goal: str = 'describe'
        self.retrieved: List[dict] = []
        self.trace: List[str] = []

    def push(self, item: Any) -> None:
        self._items.append(item)

    def contents(self) -> List[Any]:
        return list(self._items)

    def clear(self) -> None:
        self._items.clear()
        self.focus = []
        self.retrieved = []
        self.trace = []


class Thinker:
    """
    Deliberative reasoning loop with prediction error guidance.

    deliberate(query, activate_fn, score_fn, signal_fn) -> result
    counterfactual(tokens, position, alternatives, score_fn) -> ranked alternatives

    The thinker is FUNCTION-AGNOSTIC: it takes callables that provide
    activation, scoring, and signaling. This decouples reasoning from
    any specific knowledge source.
    """

    def __init__(self, max_steps: int = 10):
        self._max_steps = max_steps
        self._history: List[dict] = []

    def deliberate(
        self,
        query_tokens: List[str],
        activate_fn: Callable[[str], Dict[str, float]],
        score_fn: Callable[[List[str]], dict],
        signal_fn: Optional[Callable[[str, float], dict]] = None,
        episodic_context: Optional[dict] = None,
    ) -> dict:
        """
        Multi-step reasoning about a query.

        activate_fn(word) -> {word: energy} — activate a concept
        score_fn(tokens) -> {coherence, scores, field} — score a sequence
        signal_fn(state, quality) -> {error, ...} — prediction error

        Returns: {
            'focus': List[str],
            'retrieved': List[dict],
            'coherence': float,
            'steps': int,
            'converged': bool,
            'trace': List[str],
        }
        """
        wm = WorkingMemory()
        wm.focus = list(query_tokens)

        # Phase 1: ATTEND — activate all query words
        all_activated: Dict[str, float] = {}
        for word in query_tokens:
            field = activate_fn(word)
            for w, e in field.items():
                all_activated[w] = all_activated.get(w, 0.0) + e

        # Rank by activation energy
        ranked = sorted(all_activated.items(), key=lambda x: -x[1])
        wm.retrieved = [
            {'word': w, 'energy': e}
            for w, e in ranked[:15]
            if w not in set(query_tokens)
        ]

        # Phase 2: DELIBERATE — iterative refinement
        converged = False
        steps = 0
        prev_coherence = 0.0

        for step in range(self._max_steps):
            steps += 1

            # Score current state
            candidate_tokens = wm.focus + [r['word'] for r in wm.retrieved[:5]]
            result = score_fn(candidate_tokens)
            coherence = result.get('coherence', 0.0)

            wm.trace.append(f"step {step}: coherence={coherence:.3f}")

            # Signal: prediction error
            if signal_fn:
                state = f"deliberate:{wm.situation}"
                signal_fn(state, coherence)

            # Check convergence
            if abs(coherence - prev_coherence) < 0.01 and step > 0:
                converged = True
                break

            # If coherence is improving, continue
            if coherence > prev_coherence + 0.05:
                prev_coherence = coherence
                # Refine focus: add strongly-connected retrieved words
                for r in wm.retrieved[:3]:
                    if r['word'] not in wm.focus and r['energy'] > 0.3:
                        sub_field = activate_fn(r['word'])
                        # Check: does this word connect back to focus?
                        connects_back = any(
                            f in sub_field for f in wm.focus
                        )
                        if connects_back:
                            wm.focus.append(r['word'])
                            wm.trace.append(f"  +focus: {r['word']}")
                continue

            prev_coherence = coherence

            # Not improving — try broadening retrieval
            if wm.retrieved and step < self._max_steps - 1:
                # Activate from the best retrieved word
                best = wm.retrieved[0]['word']
                new_field = activate_fn(best)
                new_retrieved = [
                    {'word': w, 'energy': e}
                    for w, e in sorted(new_field.items(), key=lambda x: -x[1])[:10]
                    if w not in set(wm.focus) and w not in {r['word'] for r in wm.retrieved}
                ]
                wm.retrieved.extend(new_retrieved)
                wm.retrieved.sort(key=lambda x: -x['energy'])
                wm.retrieved = wm.retrieved[:15]

        result = {
            'focus': wm.focus,
            'retrieved': wm.retrieved,
            'coherence': prev_coherence,
            'steps': steps,
            'converged': converged,
            'trace': wm.trace,
        }
        self._history.append(result)
        if len(self._history) > 100:
            self._history = self._history[-50:]
        return result

    def counterfactual(
        self,
        base_tokens: List[str],
        position: int,
        alternatives: List[str],
        score_fn: Callable[[List[str]], dict],
    ) -> List[dict]:
        """
        What-if reasoning: score alternatives at a position.

        base_tokens: the original sequence
        position: which position to substitute
        alternatives: candidate words for that position
        score_fn: scoring function

        Returns: [{'word', 'coherence', 'delta'}, ...] sorted by coherence
        """
        if position < 0 or position >= len(base_tokens):
            return []

        # Score baseline
        baseline = score_fn(base_tokens)
        base_coherence = baseline.get('coherence', 0.0)

        results = []
        for alt in alternatives:
            modified = list(base_tokens)
            modified[position] = alt
            result = score_fn(modified)
            coherence = result.get('coherence', 0.0)
            results.append({
                'word': alt,
                'coherence': coherence,
                'delta': coherence - base_coherence,
            })

        results.sort(key=lambda x: -x['coherence'])
        return results

    def to_dict(self) -> dict:
        return {
            'max_steps': self._max_steps,
            'history_count': len(self._history),
        }

    @classmethod
    def from_dict(cls, d: dict) -> Thinker:
        return cls(max_steps=d.get('max_steps', 10))
