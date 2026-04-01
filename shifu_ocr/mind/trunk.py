"""
Trunk — domain emergence and routing.

Domains are NOT predefined. They form from temporal clustering of
words that co-occur more with each other than with the outside.
Seed domains can be INJECTED as hints but are never required.

The trunk routes queries to the right domain, detects cross-domain
bridges, and protects weak truths from being pruned.
"""

from __future__ import annotations
import math
from typing import Dict, List, Set, Optional, Tuple, Any

from ._types import Domain


class Trunk:
    """
    Domain emergence, routing, and structural analysis.

    Domains emerge from:
    1. Injected seeds (optional)
    2. Temporal clustering of co-occurring words
    3. Merging of overlapping clusters

    Words that appear in 2+ domains become "trunk words" —
    cross-domain bridges.
    """

    def __init__(
        self,
        seed_domains: Optional[Dict[str, List[str]]] = None,
        min_domain_size: int = 5,
        coherence_threshold: float = 0.3,
        merge_overlap: float = 0.25,
    ):
        self._min_size = min_domain_size
        self._coherence_threshold = coherence_threshold
        self._merge_overlap = merge_overlap

        self.domains: Dict[str, Domain] = {}
        self.word_domain: Dict[str, List[str]] = {}  # word -> domain names
        self.trunk_words: Set[str] = set()  # words in 2+ domains
        self._cluster_seq = 0
        self._observation_count = 0

        # Build seed index for fast lookup
        self._seed_index: Dict[str, List[str]] = {}
        if seed_domains:
            for dom_name, words in seed_domains.items():
                domain = Domain(
                    name=dom_name,
                    words=set(words),
                    seed_words=set(words),
                    strength=len(words),
                    _taught=True,
                )
                self.domains[dom_name] = domain
                for w in words:
                    if w not in self._seed_index:
                        self._seed_index[w] = []
                    self._seed_index[w].append(dom_name)
                    if w not in self.word_domain:
                        self.word_domain[w] = []
                    if dom_name not in self.word_domain[w]:
                        self.word_domain[w].append(dom_name)

    # ═══ DOMAIN DETECTION ═══

    def _detect_domain(self, content_words: List[str],
                       co_graph: Dict[str, Dict[str, float]]) -> str:
        """
        Determine which domain a set of words belongs to.
        Returns domain name (existing or newly created).
        """
        if not content_words:
            return '_misc'

        # Phase 1: Check seed index
        domain_scores: Dict[str, float] = {}
        for w in content_words:
            direct = self._seed_index.get(w)
            if direct:
                for dom in direct:
                    domain_scores[dom] = domain_scores.get(dom, 0) + 1
                continue
            # Fuzzy morphological matching against seeds
            for dom_name, domain in self.domains.items():
                if not domain.seed_words:
                    continue
                for seed in domain.seed_words:
                    if (len(w) >= 4 and seed.startswith(w[:-1])) or \
                       (len(seed) >= 4 and w.startswith(seed[:-1])):
                        domain_scores[dom_name] = domain_scores.get(dom_name, 0) + 0.7
                        break

        # Strong seed match
        best_seed = max(domain_scores, key=domain_scores.get) if domain_scores else None
        if best_seed and domain_scores[best_seed] >= 2:
            return best_seed

        # Phase 2: Check existing domains by word overlap
        best_existing = None
        best_overlap = 0
        best_ratio = 0.0
        for dom_name, domain in self.domains.items():
            if domain.size() < self._min_size:
                continue
            overlap = sum(1 for w in content_words if w in domain.words)
            if overlap == 0:
                continue
            ratio = overlap / len(content_words)
            if ratio > best_ratio:
                best_ratio = ratio
                best_existing = dom_name
                best_overlap = overlap

        if best_existing and best_overlap >= 2 and best_ratio >= 0.2:
            return best_existing

        # Phase 3: Check co-occurrence coherence for new domain creation
        # Find if these words form a coherent cluster
        coherent = self._find_coherent_cluster(content_words, co_graph)
        if coherent and len(coherent) >= self._min_size:
            # Create new domain from cluster
            dom_name = f'_c{self._cluster_seq}'
            self._cluster_seq += 1
            self.domains[dom_name] = Domain(
                name=dom_name, words=coherent,
                strength=len(coherent),
            )
            return dom_name

        # Phase 4: Assign to current catch-all cluster
        current = f'_c{self._cluster_seq}'
        if current not in self.domains:
            self.domains[current] = Domain(name=current)
        if self.domains[current].size() >= 2000:
            self._cluster_seq += 1
            current = f'_c{self._cluster_seq}'
            self.domains[current] = Domain(name=current)
        return current

    def _find_coherent_cluster(self, words: List[str],
                               co_graph: Dict[str, Dict[str, float]]) -> Set[str]:
        """
        Find the largest subset of words that are mutually connected
        in the co-occurrence graph.
        """
        if len(words) < 2:
            return set()
        # Score each word by how many other words in the list it co-occurs with
        scores: Dict[str, int] = {}
        for w in words:
            neighbors = co_graph.get(w, {})
            count = sum(1 for other in words if other != w and other in neighbors)
            scores[w] = count
        # Keep words with at least 1 connection to other list members
        coherent = {w for w, s in scores.items() if s >= 1}
        return coherent

    # ═══ OBSERVATION ═══

    def observe(self, tokens: List[str],
                co_graph: Dict[str, Dict[str, float]]) -> str:
        """
        Observe a set of tokens. Detect domain, update word-domain mapping.
        Returns the detected domain name.
        """
        content = [w for w in tokens if len(w) > 2]
        if not content:
            return '_misc'

        self._observation_count += 1
        domain_name = self._detect_domain(content, co_graph)

        # Ensure domain exists
        if domain_name not in self.domains:
            self.domains[domain_name] = Domain(name=domain_name)
        domain = self.domains[domain_name]

        # Update domain membership
        for w in content:
            domain.absorb(w)
            if w not in self.word_domain:
                self.word_domain[w] = []
            if domain_name not in self.word_domain[w]:
                self.word_domain[w].append(domain_name)

        return domain_name

    # ═══ ROUTING ═══

    def route(self, tokens: List[str],
              co_graph: Optional[Dict[str, Dict[str, float]]] = None) -> dict:
        """
        Route a query to domain(s).
        Returns: {'domain', 'trunk', 'scores', 'all'}
        """
        scores: Dict[str, int] = {}
        for w in tokens:
            w = w.lower()
            doms = self.word_domain.get(w, [])
            for d in doms:
                scores[d] = scores.get(d, 0) + 1

        ranked = sorted(scores.items(), key=lambda x: -x[1])
        if not ranked:
            return {'domain': None, 'trunk': True, 'scores': {}, 'all': []}

        top_domain = ranked[0][0]
        top_score = ranked[0][1]
        is_trunk = (len(ranked) >= 2 and ranked[1][1] >= top_score * 0.7)

        return {
            'domain': top_domain,
            'trunk': is_trunk,
            'scores': dict(scores),
            'all': [
                {'domain': d, 'score': s, 'size': self.domains[d].size()}
                for d, s in ranked if d in self.domains
            ],
        }

    # ═══ STRUCTURAL ANALYSIS ═══

    def finalize(self) -> dict:
        """
        Merge similar domains, build trunk words, detect bridges.
        Call after a batch of observations.
        """
        self._merge_domains()

        # Build trunk words
        self.trunk_words = set()
        for w, doms in self.word_domain.items():
            if len(doms) >= 2:
                self.trunk_words.add(w)

        return self.stats()

    def _merge_domains(self) -> None:
        """Merge domains with high word overlap."""
        names = list(self.domains.keys())
        merged = set()
        for i in range(len(names)):
            if names[i] in merged:
                continue
            a = self.domains.get(names[i])
            if not a:
                continue
            for j in range(i + 1, len(names)):
                if names[j] in merged:
                    continue
                b = self.domains.get(names[j])
                if not b or not b.words or not a.words:
                    continue
                # Don't merge taught domains
                if a._taught or b._taught:
                    continue
                # Compute overlap
                smaller = a if a.size() < b.size() else b
                larger = b if a.size() < b.size() else a
                smaller_name = names[i] if a.size() < b.size() else names[j]
                larger_name = names[j] if a.size() < b.size() else names[i]
                overlap = len(smaller.words & larger.words)
                ratio = overlap / max(smaller.size(), 1)
                if ratio >= self._merge_overlap:
                    # Merge smaller into larger
                    for w in smaller.words:
                        larger.absorb(w)
                        wd = self.word_domain.get(w, [])
                        if smaller_name in wd:
                            idx = wd.index(smaller_name)
                            wd[idx] = larger_name
                    del self.domains[smaller_name]
                    merged.add(smaller_name)

    def detect_bridges(self, co_graph: Dict[str, Dict[str, float]]) -> List[dict]:
        """Detect high-breadth cross-domain words."""
        V = max(len(co_graph), 1)
        bridges = []
        for w, doms in self.word_domain.items():
            if len(doms) < 2:
                continue
            breadth = len(co_graph.get(w, {}))
            ratio = breadth / V
            if ratio > 0.08:
                bridges.append({
                    'word': w, 'breadth': breadth,
                    'domains': len(doms), 'ratio': round(ratio, 4),
                })
        bridges.sort(key=lambda x: -x['ratio'])
        return bridges[:20]

    # ═══ STATS ═══

    def stats(self) -> dict:
        return {
            'domains': len(self.domains),
            'trunk_words': len(self.trunk_words),
            'observations': self._observation_count,
            'domain_sizes': [
                {'name': n, 'size': d.size(), 'taught': d._taught}
                for n, d in self.domains.items()
            ],
        }

    # ═══ SERIALIZATION ═══

    def to_dict(self) -> dict:
        return {
            'domains': {n: d.to_dict() for n, d in self.domains.items()},
            'word_domain': self.word_domain,
            'trunk_words': sorted(self.trunk_words),
            'cluster_seq': self._cluster_seq,
            'observation_count': self._observation_count,
            'seed_index': self._seed_index,
            'params': {
                'min_domain_size': self._min_size,
                'coherence_threshold': self._coherence_threshold,
                'merge_overlap': self._merge_overlap,
            },
        }

    @classmethod
    def from_dict(cls, d: dict) -> Trunk:
        params = d.get('params', {})
        t = cls(
            min_domain_size=params.get('min_domain_size', 5),
            coherence_threshold=params.get('coherence_threshold', 0.3),
            merge_overlap=params.get('merge_overlap', 0.25),
        )
        for name, dom_d in d.get('domains', {}).items():
            t.domains[name] = Domain.from_dict(dom_d)
        t.word_domain = d.get('word_domain', {})
        t.trunk_words = set(d.get('trunk_words', []))
        t._cluster_seq = d.get('cluster_seq', 0)
        t._observation_count = d.get('observation_count', 0)
        t._seed_index = d.get('seed_index', {})
        return t
