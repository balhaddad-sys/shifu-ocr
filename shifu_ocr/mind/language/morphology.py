"""
Morphology — learn word structure from the corpus.

The baby discovers:
  - ROOTS: "neur" appears in neuron, neurology, neurological, neuropathy
  - PREFIXES: "hyper" in hypertension, hyperglycemia, hyperthyroidism
  - SUFFIXES: "-itis" in arthritis, hepatitis, meningitis, encephalitis
  - COMPOUNDS: "blood-brain" always appears together

All discovered from frequency patterns in the vocabulary.
No hardcoded affix lists. The corpus teaches morphology.
"""

from __future__ import annotations
from typing import Dict, List, Set, Optional, Tuple
from collections import defaultdict


class Morphology:
    """
    Learn word structure from corpus vocabulary.

    feed_vocabulary(word_freqs) — analyze all known words
    decompose(word) — break into root + affixes
    family(root) — all words sharing a root
    """

    def __init__(self, min_root_len: int = 3, min_family_size: int = 3):
        self._min_root = min_root_len
        self._min_family = min_family_size

        # Discovered structures
        self.roots: Dict[str, Set[str]] = {}      # root → {words containing it}
        self.prefixes: Dict[str, Set[str]] = {}    # prefix → {words starting with it}
        self.suffixes: Dict[str, Set[str]] = {}    # suffix → {words ending with it}
        self.compounds: Dict[str, int] = {}        # "word1 word2" → co-occurrence count

        self._analyzed = False

    def feed_vocabulary(self, word_freqs: Dict[str, int]) -> dict:
        """
        Analyze the entire vocabulary for morphological patterns.
        Discovers roots, prefixes, suffixes from frequency.
        """
        words = [w for w in word_freqs if len(w) >= 4]
        if len(words) < 20:
            return {'roots': 0, 'prefixes': 0, 'suffixes': 0}

        # Phase 1: Find shared substrings (potential roots)
        # For each pair of words, find their longest common substring
        # Substrings that appear in 3+ words are candidate roots
        substring_words: Dict[str, Set[str]] = defaultdict(set)

        for word in words:
            # Extract all substrings of length >= min_root
            for start in range(len(word)):
                for end in range(start + self._min_root, min(len(word) + 1, start + 12)):
                    sub = word[start:end]
                    substring_words[sub].add(word)

        # Roots: substrings appearing in min_family_size+ words
        self.roots = {}
        for sub, members in substring_words.items():
            if len(members) >= self._min_family and len(sub) >= self._min_root:
                # Filter: root shouldn't be a whole common word itself
                # unless it's genuinely a root (e.g., "nerve" in "nervous", "nervously")
                if len(sub) < max(len(w) for w in members):
                    self.roots[sub] = members

        # Remove roots that are substrings of other roots
        # Keep the LONGEST root for each word family
        to_remove = set()
        root_list = sorted(self.roots.keys(), key=len, reverse=True)
        for i, r1 in enumerate(root_list):
            for r2 in root_list[i + 1:]:
                if r2 in r1 and self.roots[r2] <= self.roots[r1]:
                    to_remove.add(r2)
        for r in to_remove:
            self.roots.pop(r, None)

        # Phase 2: Discover prefixes (shared word beginnings)
        self.prefixes = {}
        prefix_words: Dict[str, Set[str]] = defaultdict(set)
        for word in words:
            for plen in range(2, min(len(word) - 1, 8)):
                prefix_words[word[:plen]].add(word)
        for prefix, members in prefix_words.items():
            if len(members) >= self._min_family and len(prefix) >= 2:
                self.prefixes[prefix] = members

        # Phase 3: Discover suffixes (shared word endings)
        self.suffixes = {}
        suffix_words: Dict[str, Set[str]] = defaultdict(set)
        for word in words:
            for slen in range(2, min(len(word) - 1, 8)):
                suffix_words[word[-slen:]].add(word)
        for suffix, members in suffix_words.items():
            if len(members) >= self._min_family and len(suffix) >= 2:
                self.suffixes[suffix] = members

        self._analyzed = True
        return {
            'roots': len(self.roots),
            'prefixes': len(self.prefixes),
            'suffixes': len(self.suffixes),
        }

    def feed_bigrams(self, nx_graph: Dict[str, Dict[str, float]]) -> int:
        """Discover compounds from bigram frequency."""
        count = 0
        for w1, nexts in nx_graph.items():
            for w2, freq in nexts.items():
                if freq >= 3 and len(w1) > 2 and len(w2) > 2:
                    compound = f"{w1} {w2}"
                    self.compounds[compound] = int(freq)
                    count += 1
        return count

    def decompose(self, word: str) -> dict:
        """Break a word into its discovered components."""
        word = word.lower()
        result = {'word': word, 'root': None, 'prefix': None, 'suffix': None, 'family': []}

        # Find root
        best_root = None
        best_family_size = 0
        for root, members in self.roots.items():
            if root in word and len(members) > best_family_size:
                best_root = root
                best_family_size = len(members)
        if best_root:
            result['root'] = best_root
            result['family'] = sorted(self.roots[best_root])[:10]
            # Prefix = what comes before the root
            idx = word.index(best_root)
            if idx > 0:
                result['prefix'] = word[:idx]
            # Suffix = what comes after the root
            end = idx + len(best_root)
            if end < len(word):
                result['suffix'] = word[end:]

        return result

    def family(self, root: str) -> List[str]:
        """All words sharing this root."""
        return sorted(self.roots.get(root, set()))

    def similar_words(self, word: str) -> List[str]:
        """Words that share morphological structure with this word."""
        d = self.decompose(word)
        if d['root']:
            return [w for w in d['family'] if w != word][:10]
        return []

    def stats(self) -> dict:
        return {
            'roots': len(self.roots),
            'prefixes': len(self.prefixes),
            'suffixes': len(self.suffixes),
            'compounds': len(self.compounds),
            'top_roots': sorted(
                [(r, len(m)) for r, m in self.roots.items()],
                key=lambda x: -x[1],
            )[:10],
            'top_prefixes': sorted(
                [(p, len(m)) for p, m in self.prefixes.items()],
                key=lambda x: -x[1],
            )[:10],
            'top_suffixes': sorted(
                [(s, len(m)) for s, m in self.suffixes.items()],
                key=lambda x: -x[1],
            )[:10],
        }

    def to_dict(self) -> dict:
        return {
            'roots': {k: sorted(v) for k, v in list(self.roots.items())[:200]},
            'prefixes': {k: sorted(v) for k, v in list(self.prefixes.items())[:100]},
            'suffixes': {k: sorted(v) for k, v in list(self.suffixes.items())[:100]},
            'compounds': dict(sorted(self.compounds.items(), key=lambda x: -x[1])[:200]),
        }

    @classmethod
    def from_dict(cls, d: dict) -> Morphology:
        m = cls()
        m.roots = {k: set(v) for k, v in d.get('roots', {}).items()}
        m.prefixes = {k: set(v) for k, v in d.get('prefixes', {}).items()}
        m.suffixes = {k: set(v) for k, v in d.get('suffixes', {}).items()}
        m.compounds = d.get('compounds', {})
        m._analyzed = bool(m.roots)
        return m
