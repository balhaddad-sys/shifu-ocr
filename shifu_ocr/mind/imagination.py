"""
Imagination — probabilistic gap-filling and analogy detection.

When two concepts are NOT directly connected, imagination asks:
COULD they be? HOW likely? THROUGH what?

Possibility: domain compatibility check
Probability: shared neighbors, spoke alignment, path distance
Projection: find the path, find the analogy

Dopamine-modulated: positive error → bolder exploration,
negative error → more cautious.
"""

from __future__ import annotations
from typing import Dict, List, Set, Optional, Tuple, Any


class Imagination:
    """
    Counterfactual reasoning and gap-filling.

    imagine(a, b) → could A connect to B? Through what?
    explore(concept) → what 2-hop concepts could be interesting?
    find_analogy(a, b) → "A is to B as C is to D"
    """

    def __init__(self):
        self.dopamine_error: float = 0.0  # Set by thinker feedback
        self._imagined_cache: Dict[str, dict] = {}

    def _possibility(self, a: str, b: str,
                     co_graph: Dict[str, Dict[str, float]],
                     word_domains: Dict[str, List[str]]) -> dict:
        """Can A connect to B at all? Domain compatibility check."""
        a_doms = set(word_domains.get(a, []))
        b_doms = set(word_domains.get(b, []))

        # Same domain → possible
        if a_doms & b_doms:
            return {'can': True, 'reason': 'same_domain', 'penalty': 0.0}
        # Both known but different domains → possible with penalty
        if a_doms and b_doms:
            return {'can': True, 'reason': 'cross_domain', 'penalty': 0.3}
        # At least one unknown → possible (benefit of doubt)
        return {'can': True, 'reason': 'unknown_domain', 'penalty': 0.1}

    def _probability(self, a: str, b: str,
                     co_graph: Dict[str, Dict[str, float]],
                     cortex_activate) -> dict:
        """How likely is a connection between A and B?"""
        evidence = []
        score = 0.0

        # 1. Shared neighbors
        a_neighbors = set(co_graph.get(a, {}).keys())
        b_neighbors = set(co_graph.get(b, {}).keys())
        shared = a_neighbors & b_neighbors
        if shared:
            contribution = min(len(shared) / max(len(a_neighbors | b_neighbors), 1), 1.0) * 0.4
            score += contribution
            evidence.append({
                'type': 'shared_neighbors', 'count': len(shared),
                'contribution': round(contribution, 3),
            })

        # 2. Spoke alignment — do A and B activate similar patterns?
        a_field = cortex_activate(a)
        b_field = cortex_activate(b)
        if a_field and b_field:
            a_words = set(list(a_field.keys())[:20])
            b_words = set(list(b_field.keys())[:20])
            overlap = len(a_words & b_words)
            if overlap > 0:
                contribution = min(overlap / 10, 1.0) * 0.3
                score += contribution
                evidence.append({
                    'type': 'spoke_alignment', 'overlap': overlap,
                    'contribution': round(contribution, 3),
                })

        # 3. Path exists (2-hop)?
        for mid in a_neighbors:
            if mid in b_neighbors and mid != a and mid != b:
                contribution = 0.2
                score += contribution
                evidence.append({
                    'type': 'path_exists', 'via': mid,
                    'contribution': round(contribution, 3),
                })
                break  # One bridge is enough evidence

        return {'score': min(score, 1.0), 'evidence': evidence}

    def _project(self, a: str, b: str,
                 co_graph: Dict[str, Dict[str, float]]) -> dict:
        """Find the path and analogy between A and B."""
        # Find bridge words
        a_neighbors = co_graph.get(a, {})
        b_neighbors = set(co_graph.get(b, {}).keys())
        bridges = []
        for mid, w in sorted(a_neighbors.items(), key=lambda x: -x[1]):
            if mid in b_neighbors and mid != a and mid != b:
                bridges.append(mid)
                if len(bridges) >= 3:
                    break

        path = [a] + bridges[:1] + [b] if bridges else [a, b]
        via = bridges[0] if bridges else None

        return {'path': path, 'via': via}

    def imagine(self, a: str, b: str,
                co_graph: Dict[str, Dict[str, float]],
                cortex_activate,
                word_domains: Dict[str, List[str]] = None) -> dict:
        """
        Imagine a connection between A and B.

        Returns: {
            'type': 'known' | 'impossible' | 'imagined',
            'probability': float,
            'evidence': List,
            'path': List[str],
            'via': str or None,
            'imagined': bool,
        }
        """
        cache_key = f"{a}:{b}"
        if cache_key in self._imagined_cache:
            return self._imagined_cache[cache_key]

        # Already connected?
        a_neighbors = co_graph.get(a, {})
        if b in a_neighbors:
            result = {
                'type': 'known', 'probability': 1.0,
                'evidence': [{'type': 'direct', 'weight': a_neighbors[b]}],
                'path': [a, b], 'via': None, 'imagined': False,
            }
            self._imagined_cache[cache_key] = result
            return result

        # Check possibility
        wd = word_domains or {}
        poss = self._possibility(a, b, co_graph, wd)
        if not poss['can']:
            result = {
                'type': 'impossible', 'probability': 0.0,
                'evidence': [], 'reason': poss['reason'],
                'path': [], 'via': None, 'imagined': False,
            }
            self._imagined_cache[cache_key] = result
            return result

        # Estimate probability
        prob = self._probability(a, b, co_graph, cortex_activate)
        prob['score'] = max(0, prob['score'] - poss['penalty'])

        # Project path
        proj = self._project(a, b, co_graph)

        result = {
            'type': 'imagined', 'probability': prob['score'],
            'evidence': prob['evidence'], 'path': proj['path'],
            'via': proj['via'], 'imagined': True,
        }
        self._imagined_cache[cache_key] = result
        return result

    def explore(self, concept: str,
                co_graph: Dict[str, Dict[str, float]],
                cortex_activate,
                word_domains: Dict[str, List[str]] = None,
                threshold: float = 0.2) -> List[dict]:
        """
        Explore: find 2-hop concepts not yet connected that COULD be.
        Dopamine-modulated: high error → lower threshold (bolder).
        """
        # Modulate threshold by dopamine
        adjusted = threshold - self.dopamine_error * 0.15
        adjusted = max(0.05, min(0.5, adjusted))

        neighbors = co_graph.get(concept, {})
        # Collect 2-hop candidates
        two_hop: Dict[str, float] = {}
        for mid, w1 in list(neighbors.items())[:20]:
            for hop2, w2 in list(co_graph.get(mid, {}).items())[:10]:
                if hop2 == concept or hop2 in neighbors:
                    continue
                two_hop[hop2] = two_hop.get(hop2, 0) + w1 * w2

        # Score each by probability
        results = []
        for candidate, energy in sorted(two_hop.items(), key=lambda x: -x[1])[:15]:
            prob = self._probability(concept, candidate, co_graph, cortex_activate)
            if prob['score'] >= adjusted:
                results.append({
                    'word': candidate, 'probability': prob['score'],
                    'evidence': prob['evidence'], 'energy': energy,
                })

        results.sort(key=lambda x: -x['probability'])
        return results[:8]

    def find_analogy(self, a: str, b: str,
                     co_graph: Dict[str, Dict[str, float]],
                     cortex_activate) -> Optional[dict]:
        """
        "A is to B as C is to D" — find analogous pair.

        Looks for C where C→D has similar spoke pattern to A→B.
        """
        a_field = cortex_activate(a)
        b_field = cortex_activate(b)
        if not a_field or not b_field:
            return None

        # The "relationship vector" = what B activates that A doesn't
        rel_words = set()
        for w in list(b_field.keys())[:15]:
            if w not in a_field or a_field.get(w, 0) < b_field[w] * 0.3:
                rel_words.add(w)

        if not rel_words:
            return None

        # Find C: a concept whose neighbors overlap with A's neighbors
        a_neighbors = set(co_graph.get(a, {}).keys())
        best_c = None
        best_score = 0
        for c in list(co_graph.keys())[:100]:
            if c == a or c == b:
                continue
            c_neighbors = set(co_graph.get(c, {}).keys())
            overlap = len(a_neighbors & c_neighbors)
            if overlap > best_score and overlap >= 2:
                best_score = overlap
                best_c = c

        if not best_c:
            return None

        # Find D: from C's neighbors, which one relates to C the way B relates to A?
        c_field = cortex_activate(best_c)
        best_d = None
        best_d_score = 0
        for d_candidate in list(co_graph.get(best_c, {}).keys())[:15]:
            if d_candidate == best_c or d_candidate == a or d_candidate == b:
                continue
            d_field = cortex_activate(d_candidate)
            # How much of the "relationship vector" does D share?
            match = sum(1 for w in rel_words if w in d_field)
            if match > best_d_score:
                best_d_score = match
                best_d = d_candidate

        if not best_d or best_d_score < 1:
            return None

        return {
            'a': a, 'b': b, 'c': best_c, 'd': best_d,
            'statement': f"{a} is to {b} as {best_c} is to {best_d}",
            'strength': best_d_score / max(len(rel_words), 1),
        }

    def invalidate_cache(self):
        self._imagined_cache.clear()

    def to_dict(self) -> dict:
        return {'dopamine_error': self.dopamine_error}

    @classmethod
    def from_dict(cls, d: dict) -> Imagination:
        im = cls()
        im.dopamine_error = d.get('dopamine_error', 0.0)
        return im
