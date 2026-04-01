"""
Core data structures for the Shifu cognitive architecture.

Every structure is serializable, emergent, and carries its own temporal history.
Nothing here prescribes what layers exist, what domains form, or what words matter.
The structures hold whatever the system discovers.
"""

from __future__ import annotations
import math
import re
from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Any

# Shared token regex — used by cortex and gate
TOKEN_RE = re.compile(r'[a-z][a-z0-9-]*')


def tokenize(text: str) -> List[str]:
    """Extract lowercase alphabetic tokens."""
    return TOKEN_RE.findall(text.lower())


# ═══════════════════════════════════════════════════════════════
#  SYNAPSE — a single weighted connection between two nodes
#
#  Born at an epoch. Strengthened by co-activation.
#  Decays when unused. Myelinated when reliable.
#  Myelinated synapses resist decay — the system's long-term memory.
# ═══════════════════════════════════════════════════════════════

@dataclass
class Synapse:
    source: str
    target: str
    weight: float = 0.0
    birth_epoch: int = 0
    last_active: int = 0
    activation_count: int = 0
    _myelinated: bool = False

    @property
    def myelinated(self) -> bool:
        return self._myelinated

    def strengthen(self, amount: float, epoch: int) -> float:
        """Add energy to this connection. Returns new weight."""
        self.weight += amount
        self.last_active = epoch
        self.activation_count += 1
        return self.weight

    def decay(self, factor: float, myelinated_factor: Optional[float] = None) -> float:
        """
        Apply temporal decay. Myelinated synapses use a gentler factor.
        Returns new weight.
        """
        if self._myelinated:
            f = myelinated_factor if myelinated_factor is not None else math.sqrt(factor)
        else:
            f = factor
        self.weight *= f
        return self.weight

    def myelinate(self) -> None:
        """Mark this synapse as long-term. It will resist decay."""
        self._myelinated = True

    def age(self, current_epoch: int) -> int:
        """How many epochs since birth."""
        return max(0, current_epoch - self.birth_epoch)

    def dormancy(self, current_epoch: int) -> int:
        """How many epochs since last activation."""
        return max(0, current_epoch - self.last_active)

    def to_dict(self) -> dict:
        return {
            's': self.source, 't': self.target, 'w': self.weight,
            'b': self.birth_epoch, 'la': self.last_active,
            'ac': self.activation_count, 'm': self._myelinated,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Synapse:
        return cls(
            source=d['s'], target=d['t'], weight=d.get('w', 0.0),
            birth_epoch=d.get('b', 0), last_active=d.get('la', 0),
            activation_count=d.get('ac', 0), _myelinated=d.get('m', False),
        )


# ═══════════════════════════════════════════════════════════════
#  ASSEMBLY — an emergent cluster of co-occurring words
#
#  Not declared. Discovered from repeated co-activation.
#  Strengthened by reuse. Pruned by neglect.
#  The system's concept-level grouping.
# ═══════════════════════════════════════════════════════════════

@dataclass
class Assembly:
    id: str
    words: Set[str] = field(default_factory=set)
    strength: float = 1.0
    birth_epoch: int = 0
    last_active: int = 0
    activation_count: int = 0
    max_size: int = 25

    def add(self, word: str, epoch: int) -> bool:
        """Add a word if assembly hasn't reached max size. Returns success."""
        if len(self.words) >= self.max_size:
            return False
        self.words.add(word)
        self.last_active = epoch
        return True

    def reinforce(self, epoch: int) -> None:
        """Strengthen this assembly on reactivation."""
        self.strength += 1
        self.last_active = epoch
        self.activation_count += 1

    def overlap(self, other: Assembly) -> float:
        """Jaccard overlap with another assembly."""
        if not self.words or not other.words:
            return 0.0
        intersection = len(self.words & other.words)
        union = len(self.words | other.words)
        return intersection / union if union > 0 else 0.0

    def overlap_with_set(self, word_set: Set[str]) -> float:
        """Fraction of this assembly's words present in the given set."""
        if not self.words:
            return 0.0
        return len(self.words & word_set) / len(self.words)

    def dormancy(self, current_epoch: int) -> int:
        return max(0, current_epoch - self.last_active)

    def to_dict(self) -> dict:
        return {
            'id': self.id, 'words': sorted(self.words),
            'strength': self.strength, 'b': self.birth_epoch,
            'la': self.last_active, 'ac': self.activation_count,
            'max': self.max_size,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Assembly:
        return cls(
            id=d['id'], words=set(d.get('words', [])),
            strength=d.get('strength', 1.0), birth_epoch=d.get('b', 0),
            last_active=d.get('la', 0), activation_count=d.get('ac', 0),
            max_size=d.get('max', 25),
        )


# ═══════════════════════════════════════════════════════════════
#  DOMAIN — an emergent knowledge region
#
#  Not predefined. Discovered from temporal clustering.
#  Seeds are optional — injected hints, not requirements.
#  Affinity is computed from the co-occurrence graph, not from
#  membership in a word list.
# ═══════════════════════════════════════════════════════════════

@dataclass
class Domain:
    name: str
    words: Set[str] = field(default_factory=set)
    seed_words: Set[str] = field(default_factory=set)
    strength: float = 0.0
    birth_epoch: int = 0
    coherence: float = 0.0
    _taught: bool = False

    def affinity(self, word: str, co_graph: Dict[str, Dict[str, float]]) -> float:
        """
        How strongly does this word belong to this domain?
        Computed from co-occurrence overlap, not list membership.
        """
        if not self.words:
            return 0.0
        neighbors = co_graph.get(word, {})
        if not neighbors:
            # Fall back to set membership if no graph data
            return 1.0 if word in self.words else 0.0
        # Count how many of this word's neighbors are domain members
        overlap = sum(1 for n in neighbors if n in self.words)
        # Normalize by neighbor count
        return overlap / max(len(neighbors), 1)

    def absorb(self, word: str) -> None:
        """Add a word to this domain."""
        self.words.add(word)

    def size(self) -> int:
        return len(self.words)

    def to_dict(self) -> dict:
        return {
            'name': self.name, 'words': sorted(self.words),
            'seeds': sorted(self.seed_words), 'strength': self.strength,
            'b': self.birth_epoch, 'coherence': self.coherence,
            'taught': self._taught,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Domain:
        return cls(
            name=d['name'], words=set(d.get('words', [])),
            seed_words=set(d.get('seeds', [])), strength=d.get('strength', 0.0),
            birth_epoch=d.get('b', 0), coherence=d.get('coherence', 0.0),
            _taught=d.get('taught', False),
        )
