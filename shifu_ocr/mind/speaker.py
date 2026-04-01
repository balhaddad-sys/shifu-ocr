"""
Speaker — language generation from connections.

Grammar is learned from observed sentence frames, not coded.
Generation follows graph connections with learned transition
probabilities and bigram context.

The Wernicke equivalent — but without hardcoded grammar rules.
"""

from __future__ import annotations
import math
from collections import deque
from typing import Dict, List, Set, Optional, Tuple, Any


class Speaker:
    """
    Language generation engine.

    learn_frame(tokens) — record sentence structure
    generate(seeds, graphs) — produce token sequence from graph
    find_path(src, tgt, graph) — conceptual path via BFS
    describe(word, graphs) — natural-language description from connections
    """

    def __init__(self, max_repetition_window: int = 10):
        self._frames: Dict[str, int] = {}  # frame pattern -> frequency
        self._bigrams: Dict[str, Dict[str, float]] = {}  # word -> {next: count}
        self._starters: Dict[str, int] = {}  # first-word patterns
        self._max_rep_window = max_repetition_window

    # ═══ LEARNING ═══

    def learn_frame(self, tokens: List[str]) -> None:
        """
        Learn sentence structure from a token sequence.
        Records bigram transitions and starter patterns.
        """
        if len(tokens) < 2:
            return

        # Record starter
        self._starters[tokens[0]] = self._starters.get(tokens[0], 0) + 1

        # Record bigrams
        for i in range(len(tokens) - 1):
            src = tokens[i]
            tgt = tokens[i + 1]
            if src not in self._bigrams:
                self._bigrams[src] = {}
            self._bigrams[src][tgt] = self._bigrams[src].get(tgt, 0) + 1

    # ═══ GENERATION ═══

    def generate(
        self,
        seed_words: List[str],
        co_graph: Dict[str, Dict[str, float]],
        nx_graph: Optional[Dict[str, Dict[str, float]]] = None,
        max_length: int = 20,
    ) -> List[str]:
        """
        Generate a token sequence starting from seed words.
        Follows graph connections with bigram transition probabilities.
        """
        if not seed_words:
            return []

        result = list(seed_words)
        used = set(seed_words)
        current = seed_words[-1]

        for _ in range(max_length - len(seed_words)):
            candidates: Dict[str, float] = {}

            # Source 1: learned bigram transitions
            bg = self._bigrams.get(current, {})
            for w, count in bg.items():
                if w not in used or len(w) > 3:
                    candidates[w] = candidates.get(w, 0) + count

            # Source 2: next-word graph
            if nx_graph:
                nx = nx_graph.get(current, {})
                for w, weight in nx.items():
                    if w not in used or len(w) > 3:
                        candidates[w] = candidates.get(w, 0) + weight * 0.5

            # Source 3: co-occurrence graph (weaker signal)
            co = co_graph.get(current, {})
            for w, weight in sorted(co.items(), key=lambda x: -x[1])[:10]:
                if w not in used:
                    candidates[w] = candidates.get(w, 0) + weight * 0.3

            if not candidates:
                break

            # Pick highest-scored candidate
            best = max(candidates, key=candidates.get)
            result.append(best)
            used.add(best)
            current = best

            # Prevent infinite loops
            if len(used) > self._max_rep_window * 2:
                break

        return result

    # ═══ PATH FINDING ═══

    def find_path(
        self,
        source: str,
        target: str,
        co_graph: Dict[str, Dict[str, float]],
        max_hops: int = 5,
    ) -> Optional[List[str]]:
        """
        Find shortest conceptual path from source to target
        via BFS on the co-occurrence graph.
        """
        if source == target:
            return [source]

        visited = {source}
        queue: deque = deque([(source, [source])])

        while queue:
            current, path = queue.popleft()
            if len(path) > max_hops:
                continue

            neighbors = co_graph.get(current, {})
            # Sort by weight descending, limit branching
            sorted_n = sorted(neighbors.items(), key=lambda x: -x[1])[:20]

            for neighbor, _ in sorted_n:
                if neighbor == target:
                    return path + [neighbor]
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))

        return None

    # ═══ DESCRIPTION ═══

    def describe(
        self,
        word: str,
        co_graph: Dict[str, Dict[str, float]],
        nx_graph: Optional[Dict[str, Dict[str, float]]] = None,
        layer_connections: Optional[Dict[str, Dict[str, float]]] = None,
    ) -> str:
        """
        Generate a natural-language description from connections.
        Uses layer connections if available, falls back to co-occurrence.
        """
        parts = []

        if layer_connections:
            for layer_name, neighbors in layer_connections.items():
                if not neighbors:
                    continue
                top = sorted(neighbors.items(), key=lambda x: -x[1])[:3]
                words = [w for w, _ in top]
                if words:
                    parts.append(f"{layer_name}: {', '.join(words)}")

        if not parts:
            # Fall back to co-occurrence neighbors
            neighbors = co_graph.get(word, {})
            top = sorted(neighbors.items(), key=lambda x: -x[1])[:6]
            if top:
                words = [w for w, _ in top]
                parts.append(f"{word} connects to {', '.join(words)}")
            else:
                return f"No connections found for {word}."

        return '. '.join(parts) + '.'

    # ═══ SERIALIZATION ═══

    def to_dict(self) -> dict:
        return {
            'frames': self._frames,
            'bigrams': self._bigrams,
            'starters': self._starters,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Speaker:
        s = cls()
        s._frames = d.get('frames', {})
        s._bigrams = d.get('bigrams', {})
        s._starters = d.get('starters', {})
        return s
