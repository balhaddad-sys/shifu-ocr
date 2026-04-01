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
        self._glue: Dict[str, Dict[str, Dict[str, float]]] = {}  # w1 -> w2 -> {connector: count}
        self._max_rep_window = max_repetition_window

    # ═══ LEARNING ═══

    def learn_frame(self, tokens: List[str]) -> None:
        """
        Learn sentence structure from a token sequence.
        Records bigram transitions, starter patterns, and grammar glue
        (the function words that connect content words).
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

        # Learn glue: function words between content words
        # "stroke IS CAUSED BY occlusion" → glue(stroke, occlusion) = "is caused by"
        content_positions = []
        for i, t in enumerate(tokens):
            if len(t) > 3:  # rough content word heuristic
                content_positions.append(i)
        for idx in range(len(content_positions) - 1):
            ci = content_positions[idx]
            cj = content_positions[idx + 1]
            w1 = tokens[ci]
            w2 = tokens[cj]
            # Extract function words between them
            glue_words = tokens[ci + 1:cj]
            if glue_words:
                glue_str = ' '.join(glue_words)
                if w1 not in self._glue:
                    self._glue[w1] = {}
                if w2 not in self._glue[w1]:
                    self._glue[w1][w2] = {}
                self._glue[w1][w2][glue_str] = self._glue[w1][w2].get(glue_str, 0) + 1

    # ═══ GENERATION ═══

    def _get_glue(self, w1: str, w2: str) -> str:
        """Get the most common function words connecting w1 to w2."""
        glue_map = self._glue.get(w1, {}).get(w2, {})
        if not glue_map:
            # Try reverse
            glue_map = self._glue.get(w2, {}).get(w1, {})
        if not glue_map:
            return ''
        return max(glue_map, key=glue_map.get)

    def generate(
        self,
        seed_words: List[str],
        co_graph: Dict[str, Dict[str, float]],
        nx_graph: Optional[Dict[str, Dict[str, float]]] = None,
        max_length: int = 20,
    ) -> List[str]:
        """
        Generate a token sequence from seed words.
        Uses learned glue to insert function words between content words,
        producing real prose rather than word chains.
        """
        if not seed_words:
            return []

        result = list(seed_words)
        used_content = set(seed_words)
        current = seed_words[-1]

        content_count = len(seed_words)
        for _ in range(max_length * 2):  # extra iterations since glue adds words
            if len(result) >= max_length:
                break

            candidates: Dict[str, float] = {}

            # Source 1: co-occurrence graph (strongest signal for content)
            co = co_graph.get(current, {})
            for w, weight in sorted(co.items(), key=lambda x: -x[1])[:15]:
                if w not in used_content and len(w) > 2:
                    candidates[w] = candidates.get(w, 0) + weight

            # Source 2: next-word graph
            if nx_graph:
                nx = nx_graph.get(current, {})
                for w, weight in nx.items():
                    if w not in used_content and len(w) > 2:
                        candidates[w] = candidates.get(w, 0) + weight * 0.5

            if not candidates:
                break

            # Pick best content word
            best = max(candidates, key=candidates.get)

            # Insert glue between current and best
            glue = self._get_glue(current, best)
            if glue:
                for gw in glue.split():
                    result.append(gw)
                    if len(result) >= max_length:
                        break

            if len(result) >= max_length:
                break

            result.append(best)
            used_content.add(best)
            current = best
            content_count += 1

            if content_count > self._max_rep_window:
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
            'glue': self._glue,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Speaker:
        s = cls()
        s._frames = d.get('frames', {})
        s._bigrams = d.get('bigrams', {})
        s._starters = d.get('starters', {})
        s._glue = d.get('glue', {})
        return s
