"""
Speaker — pure reactive plasticity.

No prediction. No planning. No caching. No motor programs.

Stimulus → field activates → output IS the activation.

The neural field already knows the answer. The transitions
already know the grammar. The speaker just reads what's there.

When connections change (learning), the output changes.
That's plasticity. The speaker doesn't prepare for anything.
It reacts to what IS, right now, in this moment.

Change when needed. Don't predict.
"""

from __future__ import annotations
from typing import Dict, List, Set, Optional, Tuple, Any
from collections import deque


class Speaker:

    def __init__(self):
        self._transitions: Dict[str, Dict[str, float]] = {}
        self._starters: Dict[str, int] = {}

    # ═══ LEARNING — plasticity, not memorization ═══

    def learn_frame(self, tokens: List[str]) -> None:
        """Strengthen transitions from real sentences."""
        if len(tokens) < 2:
            return
        self._starters[tokens[0]] = self._starters.get(tokens[0], 0) + 1
        for i in range(len(tokens) - 1):
            src, tgt = tokens[i], tokens[i + 1]
            if src not in self._transitions:
                self._transitions[src] = {}
            self._transitions[src][tgt] = self._transitions[src].get(tgt, 0) + 1

    def reinforce(self, tokens: List[str], strength: float = 1.0) -> int:
        """Strengthen transitions in a good sentence."""
        reinforced = 0
        for i in range(len(tokens) - 1):
            src, tgt = tokens[i], tokens[i + 1]
            if src not in self._transitions:
                self._transitions[src] = {}
            self._transitions[src][tgt] = self._transitions[src].get(tgt, 0) + strength
            reinforced += 1
        return reinforced

    # ═══ GENERATE — react, don't predict ═══

    def generate(
        self,
        seed_words: List[str],
        co_graph: Dict[str, Dict[str, float]],
        nx_graph: Optional[Dict[str, Dict[str, float]]] = None,
        max_length: int = 20,
        golgi=None,
        neural_field=None,
        goal: Optional[str] = None,
        edges_by_goal_fn=None,
    ) -> List[str]:
        """
        Stimulus → field activates → read the activation → done.

        No trajectory planning. No motor programs. No caching.
        The field IS the response. Transitions order it.

        If goal provided, bias which edges are read
        (not filter — bias. The field still determines everything).
        """
        if not seed_words:
            return []

        seed = seed_words[0]

        # ═══ ACTIVATE — touch the web, feel what vibrates ═══
        field: Dict[str, float] = {}
        if neural_field and neural_field.neurons and seed in neural_field.neurons:
            field = dict(neural_field.activate(seed, energy=1.0))
        elif co_graph.get(seed):
            field = {seed: 1.0}
            for w, wt in co_graph[seed].items():
                field[w] = wt

        if not field:
            return [seed]

        # ═══ GOAL BIAS — soft, not hard ═══
        if goal and edges_by_goal_fn:
            goal_edges = dict(edges_by_goal_fn(seed, goal)[:20])
            for w in field:
                if w in goal_edges:
                    field[w] *= 2.0  # Boost goal-relevant words

        # ═══ READ — the activation IS the response ═══
        # Order by: transition flow from previous word × activation energy.
        # This gives natural word order, not random activation dump.
        result = [seed]
        used = {seed}
        current = seed

        for _ in range(max_length - 1):
            best_word = None
            best_score = -1

            trans = self._transitions.get(current, {})

            # Score every activated word by: energy × transition flow
            for word, energy in field.items():
                if word in used or energy < 0.01:
                    continue
                flow = trans.get(word, 0)
                # Energy = what the field says matters
                # Flow = what grammar says follows
                # Words with BOTH are the natural response
                score = energy * (1.0 + flow)
                if score > best_score:
                    best_score = score
                    best_word = word

            # If no activated word flows from current, check transitions only
            if best_word is None:
                for word, flow in trans.items():
                    if word in used:
                        continue
                    energy = field.get(word, 0)
                    if flow > 0:
                        score = flow * (1.0 + energy)
                        if score > best_score:
                            best_score = score
                            best_word = word

            if best_word is None:
                break

            result.append(best_word)
            used.add(best_word)
            current = best_word

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
            'transitions': self._transitions,
            'starters': self._starters,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Speaker:
        s = cls()
        s._transitions = d.get('transitions', d.get('bigrams', {}))
        s._starters = d.get('starters', {})
        return s
