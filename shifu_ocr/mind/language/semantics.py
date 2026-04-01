"""
Semantics — meaning relationships beyond co-occurrence.

The baby discovers:
  - SYNONYMY: words that appear in the SAME contexts → similar meaning
  - ANTONYMY: words that appear in OPPOSITE contexts
  - HYPERNYMY: "X is a Y" → Y is more general than X
  - MERONYMY: "X of Y" → X is part of Y
  - CAUSALITY: "X causes Y" → directional relationship

All derived from the co-graph and nx-graph. No WordNet. No dictionaries.
Meaning emerges from usage patterns.
"""

from __future__ import annotations
import math
from typing import Dict, List, Set, Optional, Tuple
from collections import defaultdict


class Semantics:
    """
    Discover meaning relationships from usage patterns.

    feed(co_graph, nx_graph) — analyze graphs for semantic relations
    synonyms(word) — words used in similar contexts
    hypernyms(word) — more general terms ("stroke is a disease" → disease)
    causes(word) — what this word causes
    parts(word) — parts of this concept
    """

    def __init__(self):
        self._synonyms: Dict[str, List[Tuple[str, float]]] = {}
        self._hypernyms: Dict[str, List[str]] = {}
        self._hyponyms: Dict[str, List[str]] = {}
        self._causes: Dict[str, List[str]] = {}
        self._caused_by: Dict[str, List[str]] = {}
        self._parts: Dict[str, List[str]] = {}
        self._whole_of: Dict[str, List[str]] = {}

    def feed(self, co_graph: Dict[str, Dict[str, float]],
             nx_graph: Dict[str, Dict[str, float]],
             word_freqs: Dict[str, int]) -> dict:
        """
        Analyze graphs for semantic relationships.
        """
        words = [w for w in word_freqs if len(w) > 3 and word_freqs[w] >= 2]

        # ═══ SYNONYMY: words with similar co-occurrence neighborhoods ═══
        # cos(neighbors(A), neighbors(B)) → similarity
        syn_count = 0
        for i, w1 in enumerate(words[:200]):
            n1 = co_graph.get(w1, {})
            if len(n1) < 3:
                continue
            n1_set = set(list(n1.keys())[:50])
            candidates = []
            for w2 in words[:200]:
                if w2 == w1:
                    continue
                n2 = co_graph.get(w2, {})
                if len(n2) < 3:
                    continue
                n2_set = set(list(n2.keys())[:50])
                # Jaccard similarity of neighborhoods
                inter = len(n1_set & n2_set)
                union = len(n1_set | n2_set)
                if union > 0 and inter >= 2:
                    sim = inter / union
                    if sim > 0.15:
                        candidates.append((w2, round(sim, 3)))
            if candidates:
                candidates.sort(key=lambda x: -x[1])
                self._synonyms[w1] = candidates[:5]
                syn_count += 1

        # ═══ HYPERNYMY: "X is a Y" patterns in nx_graph ═══
        # X → is → a → Y in the bigram chain
        hyper_count = 0
        is_followers = nx_graph.get('is', {})
        a_followers = nx_graph.get('a', {})
        for word in words:
            nx = nx_graph.get(word, {})
            if 'is' not in nx:
                continue
            # Word precedes "is" — look for what "is" leads to
            hypers = []
            for follower, freq in is_followers.items():
                if follower == word or len(follower) <= 2:
                    continue
                # Also check "a" followers for "is a X" pattern
                if follower == 'a':
                    for cat, cf in a_followers.items():
                        if len(cat) > 3 and cat != word:
                            hypers.append(cat)
                elif len(follower) > 3:
                    hypers.append(follower)
            if hypers:
                self._hypernyms[word] = list(set(hypers))[:5]
                for h in hypers[:5]:
                    if h not in self._hyponyms:
                        self._hyponyms[h] = []
                    self._hyponyms[h].append(word)
                hyper_count += 1

        # ═══ CAUSALITY: "X causes/leads Y" patterns ═══
        causal_count = 0
        causal_words = ['causes', 'leads', 'results', 'produces']
        for cw in causal_words:
            cw_preceders = {}
            for word in words:
                nx = nx_graph.get(word, {})
                if cw in nx:
                    cw_preceders[word] = nx[cw]
            cw_followers = nx_graph.get(cw, {})
            for cause_word, freq in cw_preceders.items():
                effects = [w for w, f in cw_followers.items() if len(w) > 3 and w != cause_word]
                if effects:
                    if cause_word not in self._causes:
                        self._causes[cause_word] = []
                    self._causes[cause_word].extend(effects[:3])
                    for e in effects[:3]:
                        if e not in self._caused_by:
                            self._caused_by[e] = []
                        self._caused_by[e].append(cause_word)
                    causal_count += 1

        # ═══ MERONYMY: "X of Y" patterns ═══
        mero_count = 0
        of_followers = nx_graph.get('of', {})
        for word in words:
            nx = nx_graph.get(word, {})
            if 'of' in nx:
                # word → of → Y → word is part of Y
                wholes = [w for w, f in of_followers.items() if len(w) > 3 and w != word]
                if wholes:
                    self._parts[word] = wholes[:3]
                    for wh in wholes[:3]:
                        if wh not in self._whole_of:
                            self._whole_of[wh] = []
                        self._whole_of[wh].append(word)
                    mero_count += 1

        return {
            'synonyms': syn_count,
            'hypernyms': hyper_count,
            'causal': causal_count,
            'meronymy': mero_count,
        }

    def synonyms(self, word: str) -> List[Tuple[str, float]]:
        return self._synonyms.get(word.lower(), [])

    def hypernyms(self, word: str) -> List[str]:
        """More general terms: stroke → disease."""
        return self._hypernyms.get(word.lower(), [])

    def hyponyms(self, word: str) -> List[str]:
        """More specific terms: disease → stroke, epilepsy."""
        return self._hyponyms.get(word.lower(), [])

    def causes(self, word: str) -> List[str]:
        """What does this cause?"""
        return self._causes.get(word.lower(), [])

    def caused_by(self, word: str) -> List[str]:
        """What causes this?"""
        return self._caused_by.get(word.lower(), [])

    def parts(self, word: str) -> List[str]:
        """Parts of this concept."""
        return self._whole_of.get(word.lower(), [])

    def whole_of(self, word: str) -> List[str]:
        """What is this a part of?"""
        return self._parts.get(word.lower(), [])

    def explain(self, word: str) -> str:
        """Full semantic profile of a word."""
        w = word.lower()
        parts = []
        hyp = self.hypernyms(w)
        if hyp:
            parts.append(f"{word} is a type of {', '.join(hyp[:3])}")
        syn = self.synonyms(w)
        if syn:
            parts.append(f"similar to {', '.join(s for s, _ in syn[:3])}")
        cau = self.causes(w)
        if cau:
            parts.append(f"causes {', '.join(cau[:3])}")
        cb = self.caused_by(w)
        if cb:
            parts.append(f"caused by {', '.join(cb[:3])}")
        pts = self.parts(w)
        if pts:
            parts.append(f"has parts: {', '.join(pts[:3])}")
        if not parts:
            return f"No semantic relationships discovered for {word} yet."
        return '. '.join(parts) + '.'

    def stats(self) -> dict:
        return {
            'synonyms': len(self._synonyms),
            'hypernyms': len(self._hypernyms),
            'hyponyms': len(self._hyponyms),
            'causal': len(self._causes),
            'meronymy': len(self._parts),
        }

    def to_dict(self) -> dict:
        return {
            'synonyms': {k: v for k, v in list(self._synonyms.items())[:200]},
            'hypernyms': dict(list(self._hypernyms.items())[:200]),
            'hyponyms': dict(list(self._hyponyms.items())[:200]),
            'causes': dict(list(self._causes.items())[:200]),
            'caused_by': dict(list(self._caused_by.items())[:200]),
            'parts': dict(list(self._parts.items())[:100]),
            'whole_of': dict(list(self._whole_of.items())[:100]),
        }

    @classmethod
    def from_dict(cls, d: dict) -> Semantics:
        s = cls()
        s._synonyms = {k: [(w, sc) for w, sc in v] for k, v in d.get('synonyms', {}).items()}
        s._hypernyms = d.get('hypernyms', {})
        s._hyponyms = d.get('hyponyms', {})
        s._causes = d.get('causes', {})
        s._caused_by = d.get('caused_by', {})
        s._parts = d.get('parts', {})
        s._whole_of = d.get('whole_of', {})
        return s
