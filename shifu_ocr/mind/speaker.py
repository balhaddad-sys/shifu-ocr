"""
Speaker — packet-aware language generation.

specialize → relay → re-specialize applied to PRODUCTION:

1. CONTENT SKELETON: activate seed, collect content words (PATH 3-4)
   from co-graph. These are the MEANING carriers.

2. STRUCTURAL GLUE: for each pair of content words, find the
   structural words (PATH 1-2) that connect them in the nx-graph.
   These are the GRAMMAR carriers.

3. ASSEMBLE: interleave content + glue into a sentence.
   Content words carry meaning. Glue words carry structure.
   The sentence is the INTERFERENCE PATTERN of both.

Not one-word-at-a-time bigram chains.
Content skeleton first. Glue second. Like the brain:
Broca assembles meaning from Wernicke's content.
"""

from __future__ import annotations
from typing import Dict, List, Set, Optional, Tuple, Any
from collections import deque


class Speaker:

    def __init__(self, max_repetition_window: int = 10):
        self._bigrams: Dict[str, Dict[str, float]] = {}
        self._starters: Dict[str, int] = {}
        self._glue: Dict[str, Dict[str, Dict[str, float]]] = {}
        self._max_rep_window = max_repetition_window

    # ═══ LEARNING ═══

    def learn_frame(self, tokens: List[str]) -> None:
        """Learn sentence structure from a token sequence."""
        if len(tokens) < 2:
            return
        self._starters[tokens[0]] = self._starters.get(tokens[0], 0) + 1
        for i in range(len(tokens) - 1):
            src, tgt = tokens[i], tokens[i + 1]
            if src not in self._bigrams:
                self._bigrams[src] = {}
            self._bigrams[src][tgt] = self._bigrams[src].get(tgt, 0) + 1

        # Learn glue between content words
        content_positions = []
        for i, t in enumerate(tokens):
            if len(t) > 3:
                content_positions.append(i)
        for idx in range(len(content_positions) - 1):
            ci, cj = content_positions[idx], content_positions[idx + 1]
            w1, w2 = tokens[ci], tokens[cj]
            glue_words = tokens[ci + 1:cj]
            if glue_words:
                glue_str = ' '.join(glue_words)
                if w1 not in self._glue:
                    self._glue[w1] = {}
                if w2 not in self._glue[w1]:
                    self._glue[w1][w2] = {}
                self._glue[w1][w2][glue_str] = self._glue[w1][w2].get(glue_str, 0) + 1

    def _get_glue(self, w1: str, w2: str) -> str:
        """Get the most common function words connecting w1 to w2."""
        glue_map = self._glue.get(w1, {}).get(w2, {})
        if not glue_map:
            glue_map = self._glue.get(w2, {}).get(w1, {})
        if not glue_map:
            return ''
        return max(glue_map, key=glue_map.get)

    # ═══ PACKET-AWARE GENERATION ═══

    def generate(
        self,
        seed_words: List[str],
        co_graph: Dict[str, Dict[str, float]],
        nx_graph: Optional[Dict[str, Dict[str, float]]] = None,
        max_length: int = 20,
        golgi=None,
    ) -> List[str]:
        """
        specialize → relay → re-specialize for PRODUCTION.

        1. Build CONTENT SKELETON from co-graph (PATH 3-4 words)
        2. Insert STRUCTURAL GLUE between content words (PATH 1-2)
        3. Assemble into sentence
        """
        if not seed_words:
            return []

        seed = seed_words[0]

        # STEP 1: Content skeleton — collect PATH 3-4 words from co-graph
        skeleton = [seed]
        used = {seed}
        current = seed

        for _ in range(max_length // 2):  # Content words are ~half the sentence
            co = co_graph.get(current, {})
            # Pick best content neighbor (len > 5 = likely content/specialist)
            candidates = sorted(
                [(w, wt) for w, wt in co.items()
                 if w not in used and len(w) > 4],
                key=lambda x: -x[1],
            )
            if not candidates:
                # Fall back to any neighbor
                candidates = sorted(
                    [(w, wt) for w, wt in co.items()
                     if w not in used and len(w) > 2],
                    key=lambda x: -x[1],
                )
            if not candidates:
                break
            best = candidates[0][0]
            skeleton.append(best)
            used.add(best)
            current = best

        if len(skeleton) < 2:
            return skeleton

        # STEP 2: Insert glue between content words
        result = []
        for i in range(len(skeleton)):
            if i > 0:
                # Find glue between skeleton[i-1] and skeleton[i]
                glue = self._get_glue(skeleton[i - 1], skeleton[i])
                if glue:
                    for gw in glue.split():
                        result.append(gw)
                        if len(result) >= max_length:
                            break
                elif nx_graph:
                    # No learned glue — try nx-graph for a bridge word
                    nx = nx_graph.get(skeleton[i - 1], {})
                    # Find a short word that leads to the next content word
                    for bridge, bw in sorted(nx.items(), key=lambda x: -x[1])[:5]:
                        if len(bridge) <= 4:  # Structural
                            bridge_nx = nx_graph.get(bridge, {})
                            if skeleton[i] in bridge_nx:
                                result.append(bridge)
                                break

            if len(result) >= max_length:
                break
            result.append(skeleton[i])

        return result[:max_length]

    # ═══ PATH FINDING ═══

    def find_path(
        self,
        source: str,
        target: str,
        co_graph: Dict[str, Dict[str, float]],
        max_hops: int = 5,
    ) -> Optional[List[str]]:
        """Shortest conceptual path via BFS."""
        if source == target:
            return [source]
        visited = {source}
        queue: deque = deque([(source, [source])])
        while queue:
            current, path = queue.popleft()
            if len(path) > max_hops:
                continue
            neighbors = co_graph.get(current, {})
            for neighbor, _ in sorted(neighbors.items(), key=lambda x: -x[1])[:20]:
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
        """Generate description from connections."""
        parts = []
        if layer_connections:
            for layer_name, neighbors in layer_connections.items():
                if not neighbors:
                    continue
                top = sorted(neighbors.items(), key=lambda x: -x[1])[:3]
                words = [w for w, _ in top if len(w) > 2]
                if words:
                    parts.append(f"{layer_name}: {', '.join(words)}")
        if not parts:
            neighbors = co_graph.get(word, {})
            top = sorted(neighbors.items(), key=lambda x: -x[1])[:6]
            if top:
                words = [w for w, _ in top if len(w) > 2]
                parts.append(f"{word} connects to {', '.join(words)}")
            else:
                return f"No connections found for {word}."
        return '. '.join(parts) + '.'

    # ═══ SERIALIZATION ═══

    def to_dict(self) -> dict:
        return {
            'bigrams': self._bigrams,
            'starters': self._starters,
            'glue': self._glue,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Speaker:
        s = cls()
        s._bigrams = d.get('bigrams', {})
        s._starters = d.get('starters', {})
        s._glue = d.get('glue', {})
        return s
