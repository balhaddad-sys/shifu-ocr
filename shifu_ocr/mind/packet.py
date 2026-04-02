"""
Labeled-Line Packet — the signal that carries its own identity.

Like somatosensory afferents: the signal doesn't just carry data.
It carries WHERE it came from, WHAT modality detected it,
WHEN it was born, HOW confident the detector was.

The packet is never stripped of its tags. From the moment
a receptor fires to the moment cortex processes the signal,
the provenance travels WITH the data.

specialize → relay → re-specialize

Each packet carries:
  word:       the actual data
  origin:     which receptor/pathway created it (1=structural, 2=connector, 3=content, 4=specialist)
  modality:   what KIND of processing it received
  birth:      when it was first seen (epoch)
  cohort:     what was born alongside it (spatiotemporal neighbors)
  confidence: how strong the signal is (weight)
  position:   where in the sentence it appeared
  lineage:    list of every processing stage it passed through
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


@dataclass
class Packet:
    """One labeled-line signal. Carries data + full provenance."""

    word: str
    origin: int = 0              # Golgi pathway: 1=structural 2=connector 3=content 4=specialist
    modality: str = 'raw'        # What processing: 'raw', 'co_graph', 'identity', 'spoke', 'neural'
    birth: int = 0               # Epoch when first seen
    cohort: List[str] = field(default_factory=list)  # Spatiotemporal neighbors
    confidence: float = 1.0      # Signal strength
    position: int = 0            # Position in sentence
    lineage: List[str] = field(default_factory=list)  # Processing history
    destination: str = 'general' # Where it's headed

    def relay(self, stage: str, confidence_delta: float = 0.0) -> 'Packet':
        """
        Pass through a relay station. The packet gains a lineage entry
        and its confidence may change. But its IDENTITY is preserved.
        """
        self.lineage.append(stage)
        self.confidence = max(0.0, min(1.0, self.confidence + confidence_delta))
        return self

    def retag(self, destination: str) -> 'Packet':
        """Re-specialize: change destination for downstream processing."""
        self.destination = destination
        return self

    def is_structural(self) -> bool:
        return self.origin <= 2

    def is_content(self) -> bool:
        return self.origin >= 3


class PacketStream:
    """
    A stream of labeled-line packets flowing through the system.

    specialize → relay → re-specialize

    The stream preserves packet identity at every stage.
    Downstream consumers can filter by origin, modality,
    confidence, or destination — without losing provenance.
    """

    def __init__(self):
        self.packets: List[Packet] = []

    def add(self, packet: Packet) -> None:
        self.packets.append(packet)

    def from_tokens(self, tokens: List[str], epoch: int,
                    golgi=None, word_freqs: Optional[Dict[str, int]] = None) -> 'PacketStream':
        """
        SPECIALIZE: convert raw tokens into labeled packets.
        Each token gets tagged by the Golgi with its pathway.
        """
        for i, word in enumerate(tokens):
            origin = 0
            cohort = []
            if golgi:
                tag = golgi.tag(word, epoch, (word_freqs or {}).get(word, 1))
                origin = tag.pathway
                cohort = tag.cohort[:5]

            p = Packet(
                word=word,
                origin=origin,
                modality='raw',
                birth=epoch,
                cohort=cohort,
                confidence=1.0,
                position=i,
                lineage=['tokenize'],
                destination='general',
            )
            self.packets.append(p)
        return self

    # ═══ RELAY: transform without losing identity ═══

    def relay_co_graph(self, co_graph: Dict[str, Dict[str, float]]) -> 'PacketStream':
        """
        Relay through co-graph. Each packet gains co-occurrence context
        but keeps its origin tag.
        """
        for p in self.packets:
            co = co_graph.get(p.word, {})
            if co:
                # Confidence boosted by co-graph connectivity
                connectivity = min(len(co) / 50.0, 1.0)
                p.relay('co_graph', confidence_delta=connectivity * 0.2)
                p.modality = 'co_graph'
        return self

    def relay_identity(self, nx_graph: Dict[str, Dict[str, float]]) -> 'PacketStream':
        """
        Relay through identity detection. Only specialist packets (PATH 4)
        get identity extraction. Others pass through unchanged.
        """
        is_followers = nx_graph.get('is', {})
        for p in self.packets:
            if p.origin >= 4:  # Specialist words get identity check
                nx = nx_graph.get(p.word, {})
                if 'is' in nx:
                    # This word precedes "is" → it's a subject of identity
                    p.relay('identity', confidence_delta=0.1)
                    p.retag('identity')
                else:
                    p.relay('identity_skip')
            else:
                p.relay('identity_bypass')  # Structural words skip identity
        return self

    def relay_spokes(self, co_graph: Dict[str, Dict[str, float]]) -> 'PacketStream':
        """
        Relay through spoke detection. Content and specialist packets
        get routed to their semantic spoke based on co-graph neighborhood.
        """
        spoke_signals = {
            'mechanism': {'causes', 'caused', 'leads', 'results', 'produces', 'involves'},
            'function': {'treatment', 'therapy', 'treats', 'used', 'prevents', 'reduces'},
            'appearance': {'presents', 'shows', 'appears', 'seen', 'reveals'},
            'relation': {'associated', 'related', 'risk', 'factor', 'linked'},
        }
        for p in self.packets:
            if p.origin < 3:  # Structural/connector skip spoke routing
                p.relay('spoke_bypass')
                continue
            # Check which spoke this word's neighbors signal
            co = co_graph.get(p.word, {})
            best_spoke = 'general'
            best_score = 0
            for spoke, signals in spoke_signals.items():
                score = sum(co.get(s, 0) for s in signals)
                if score > best_score:
                    best_score = score
                    best_spoke = spoke
            if best_score > 0:
                p.retag(best_spoke)
                p.relay(f'spoke:{best_spoke}', confidence_delta=0.1)
            else:
                p.relay('spoke_none')
        return self

    # ═══ RE-SPECIALIZE: downstream consumers unpack by task ═══

    def for_task(self, task: str) -> List[Packet]:
        """
        Re-specialize: filter packets for a specific downstream task.
        The task determines which packets are relevant.
        """
        if task == 'identity':
            return [p for p in self.packets if p.destination == 'identity' or p.origin >= 4]
        elif task == 'mechanism':
            return [p for p in self.packets if p.destination == 'mechanism' or p.origin >= 3]
        elif task == 'function':
            return [p for p in self.packets if p.destination == 'function' or p.origin >= 3]
        elif task == 'appearance':
            return [p for p in self.packets if p.destination == 'appearance' or p.origin >= 3]
        elif task == 'relation':
            return [p for p in self.packets if p.destination == 'relation' or p.origin >= 3]
        elif task == 'structural':
            return [p for p in self.packets if p.origin <= 2]
        elif task == 'content':
            return [p for p in self.packets if p.origin >= 3]
        elif task == 'all':
            return list(self.packets)
        return [p for p in self.packets if p.destination == task]

    def by_confidence(self, min_conf: float = 0.0) -> List[Packet]:
        """Filter by signal strength."""
        return sorted(
            [p for p in self.packets if p.confidence >= min_conf],
            key=lambda p: -p.confidence,
        )

    def by_origin(self, pathway: int) -> List[Packet]:
        """Filter by Golgi pathway."""
        return [p for p in self.packets if p.origin == pathway]

    def content_words(self) -> List[str]:
        """Extract content words (PATH 3+) in order."""
        return [p.word for p in self.packets if p.origin >= 3]

    def structural_words(self) -> List[str]:
        """Extract structural words (PATH 1-2) in order."""
        return [p.word for p in self.packets if p.origin <= 2]

    def to_dict(self) -> List[dict]:
        return [{
            'word': p.word, 'origin': p.origin, 'modality': p.modality,
            'birth': p.birth, 'confidence': round(p.confidence, 3),
            'position': p.position, 'destination': p.destination,
            'lineage': p.lineage,
        } for p in self.packets]

    def summary(self) -> dict:
        origins = {1: 0, 2: 0, 3: 0, 4: 0}
        dests = {}
        for p in self.packets:
            origins[p.origin] = origins.get(p.origin, 0) + 1
            dests[p.destination] = dests.get(p.destination, 0) + 1
        return {
            'total': len(self.packets),
            'structural': origins.get(1, 0) + origins.get(2, 0),
            'content': origins.get(3, 0),
            'specialist': origins.get(4, 0),
            'destinations': dests,
            'avg_confidence': round(
                sum(p.confidence for p in self.packets) / max(len(self.packets), 1), 3
            ),
        }
