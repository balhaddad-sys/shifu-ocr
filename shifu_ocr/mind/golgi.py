"""
Golgi Apparatus — tag, sort, route.

Every word gets tagged BEFORE processing. The tag determines
which pathway it enters. Like the Golgi tags proteins with
mannose-6-phosphate for lysosomes or glycosylation for membranes.

TAGS:
  - birth_epoch: WHEN this word was first seen
  - cohort: WHICH words were born in the same epoch (spatiotemporal neighbors)
  - pathway: WHICH processing depth (fast/medium/deep)
  - destination: WHICH cortical area receives it

PATHWAYS by word length (proxy for specialization):
  PATH 1 (len 2-3): structural words. Fast highway.
    "the", "is", "in", "of", "to", "by", "an"
    Processing: co-graph count only. No spokes. No identity.
    These words are GLUE — they connect, they don't mean.

  PATH 2 (len 4-5): common words. Medium lane.
    "with", "from", "that", "this", "have", "been"
    Processing: co-graph + nx-graph. Basic connections.

  PATH 3 (len 6-8): content words. Standard processing.
    "stroke", "brain", "causes", "disease", "artery"
    Processing: full co-graph + identity extraction + spoke routing.

  PATH 4 (len 9+): specialized terms. Deep processing.
    "thrombolysis", "hemorrhagic", "neurotransmitter"
    Processing: everything in Path 3 + myelination candidate +
    morphological analysis + cross-domain bridging.
    These words are RARE so the processor has MORE TIME.

After tagging: STDP (spike-timing-dependent plasticity).
Words born in nearby epochs have temporal affinity.
The tag carries the epoch so STDP can strengthen
connections between temporal neighbors.
"""

from __future__ import annotations
from typing import Dict, List, Set, Optional, Tuple
from dataclasses import dataclass


@dataclass
class WordTag:
    """The tag attached to each word by the Golgi."""
    word: str
    length: int
    pathway: int          # 1=fast, 2=medium, 3=standard, 4=deep
    birth_epoch: int      # WHEN first seen
    cohort: List[str]     # WHICH words were born nearby (spatiotemporal)
    frequency: int        # How often seen
    destination: str      # Which cortical area ('structural', 'connector', 'content', 'specialist')


class Golgi:
    """
    Tag, sort, route. Every word gets labeled before processing.

    tag(word, epoch, cohort_words) → WordTag
    route(tags) → {pathway: [tags]} — sorted by pathway
    """

    def __init__(self):
        # Tag registry: word → WordTag
        self._tags: Dict[str, WordTag] = {}
        # Cohort index: epoch → [words born in that epoch]
        self._cohorts: Dict[int, List[str]] = {}

    def tag(self, word: str, birth_epoch: int, frequency: int = 1) -> WordTag:
        """
        Tag a word. If already tagged, update frequency.
        The tag determines its processing pathway.
        """
        existing = self._tags.get(word)
        if existing:
            existing.frequency = max(existing.frequency, frequency)
            return existing

        length = len(word)

        # PATHWAY by length — specialization, not generalization
        if length <= 3:
            pathway = 1
            destination = 'structural'
        elif length <= 5:
            pathway = 2
            destination = 'connector'
        elif length <= 8:
            pathway = 3
            destination = 'content'
        else:
            pathway = 4
            destination = 'specialist'

        # Cohort: words born in nearby epochs (+/- 5)
        cohort = []
        for ep in range(max(0, birth_epoch - 5), birth_epoch + 6):
            cohort.extend(self._cohorts.get(ep, []))
        # Remove self
        cohort = [w for w in cohort if w != word][:20]

        t = WordTag(
            word=word, length=length, pathway=pathway,
            birth_epoch=birth_epoch, cohort=cohort,
            frequency=frequency, destination=destination,
        )
        self._tags[word] = t

        # Register in cohort index
        if birth_epoch not in self._cohorts:
            self._cohorts[birth_epoch] = []
        self._cohorts[birth_epoch].append(word)

        return t

    def tag_sentence(self, tokens: List[str], epoch: int,
                     word_freqs: Dict[str, int]) -> List[WordTag]:
        """Tag all words in a sentence. Returns sorted by pathway."""
        tags = []
        for w in tokens:
            freq = word_freqs.get(w, 1)
            t = self.tag(w, epoch, freq)
            tags.append(t)
        return tags

    def route(self, tags: List[WordTag]) -> Dict[int, List[WordTag]]:
        """Sort tags by pathway. Each pathway gets its own batch."""
        routes: Dict[int, List[WordTag]] = {1: [], 2: [], 3: [], 4: []}
        for t in tags:
            routes[t.pathway].append(t)
        return routes

    def get_cohort(self, word: str) -> List[str]:
        """Who was born near this word? Spatiotemporal neighbors."""
        t = self._tags.get(word)
        if not t:
            return []
        return t.cohort

    def get_pathway(self, word: str) -> int:
        """Which pathway does this word use?"""
        t = self._tags.get(word)
        return t.pathway if t else 0

    def stdp_affinity(self, word_a: str, word_b: str) -> float:
        """
        Spike-Timing-Dependent Plasticity.
        How temporally close were these words born?
        Close births → high affinity → stronger connection.
        Distant births → low affinity → weaker connection.
        """
        ta = self._tags.get(word_a)
        tb = self._tags.get(word_b)
        if not ta or not tb:
            return 0.5  # Unknown → neutral
        # Temporal distance
        dt = abs(ta.birth_epoch - tb.birth_epoch)
        if dt == 0:
            return 1.0  # Born together — maximum affinity
        # Exponential decay: close → high, far → low
        return 1.0 / (1.0 + dt * 0.1)

    def stats(self) -> dict:
        pathways = {1: 0, 2: 0, 3: 0, 4: 0}
        for t in self._tags.values():
            pathways[t.pathway] += 1
        return {
            'tagged': len(self._tags),
            'cohorts': len(self._cohorts),
            'pathway_1_structural': pathways[1],
            'pathway_2_connector': pathways[2],
            'pathway_3_content': pathways[3],
            'pathway_4_specialist': pathways[4],
        }

    def to_dict(self) -> dict:
        tags = {}
        for w, t in list(self._tags.items())[:5000]:
            tags[w] = {
                'p': t.pathway, 'b': t.birth_epoch,
                'f': t.frequency, 'd': t.destination,
            }
        return {'tags': tags}

    @classmethod
    def from_dict(cls, d: dict) -> Golgi:
        g = cls()
        for w, td in d.get('tags', {}).items():
            t = WordTag(
                word=w, length=len(w), pathway=td['p'],
                birth_epoch=td['b'], cohort=[],
                frequency=td['f'], destination=td['d'],
            )
            g._tags[w] = t
            ep = td['b']
            if ep not in g._cohorts:
                g._cohorts[ep] = []
            g._cohorts[ep].append(w)
        return g
