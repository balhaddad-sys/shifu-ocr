"""
Cortex — multi-layer semantic memory.

The cortex is a weighted directed graph organized into named layers.
Layers are NOT predefined. They emerge on first use via ensure_layer().
You can have 5 layers or 50 — the architecture doesn't care.

Connections form from co-occurrence within a sliding window.
Strength decays over time. Frequently-used connections myelinate
and resist decay. Pruning removes weak connections.

Plasticity starts high (critical period) and decreases as the
network matures — early experience shapes the landscape more
than late experience, but learning never stops.
"""

from __future__ import annotations
import math
import re
from typing import Dict, List, Set, Optional, Tuple, Any

from ._types import Synapse, Assembly


# ═══════════════════════════════════════════════════════════════
#  LAYER — a named semantic plane
# ═══════════════════════════════════════════════════════════════

class Layer:
    """
    One semantic dimension of the cortex.
    Holds synapses between words within this semantic context.
    """

    __slots__ = ('name', 'birth_epoch', '_connections', '_node_set')

    def __init__(self, name: str, birth_epoch: int = 0):
        self.name = name
        self.birth_epoch = birth_epoch
        # source -> target -> Synapse
        self._connections: Dict[str, Dict[str, Synapse]] = {}
        self._node_set: Set[str] = set()

    @property
    def node_count(self) -> int:
        return len(self._node_set)

    @property
    def synapse_count(self) -> int:
        count = 0
        for targets in self._connections.values():
            count += len(targets)
        return count

    def connect(self, source: str, target: str, weight: float, epoch: int) -> Synapse:
        """Create or strengthen a connection. Returns the synapse."""
        self._node_set.add(source)
        self._node_set.add(target)
        if source not in self._connections:
            self._connections[source] = {}
        existing = self._connections[source].get(target)
        if existing is not None:
            existing.strengthen(weight, epoch)
            return existing
        syn = Synapse(
            source=source, target=target, weight=weight,
            birth_epoch=epoch, last_active=epoch, activation_count=1,
        )
        self._connections[source][target] = syn
        return syn

    def get_neighbors(self, node: str, min_weight: float = 0.0) -> Dict[str, float]:
        """All outgoing connections from this node above min_weight."""
        targets = self._connections.get(node, {})
        return {
            t: s.weight for t, s in targets.items()
            if s.weight >= min_weight
        }

    def get_synapse(self, source: str, target: str) -> Optional[Synapse]:
        targets = self._connections.get(source, {})
        return targets.get(target)

    def has_node(self, node: str) -> bool:
        return node in self._node_set

    def prune(self, epoch: int, decay_factor: float, myelinated_factor: float,
              min_weight: float) -> int:
        """
        Decay all connections. Remove those below min_weight.
        Myelinated synapses use the gentler factor.
        Returns count of pruned synapses.
        """
        pruned = 0
        empty_sources = []
        for source, targets in self._connections.items():
            dead = []
            for target, syn in targets.items():
                syn.decay(decay_factor, myelinated_factor)
                if syn.weight < min_weight:
                    dead.append(target)
            for t in dead:
                del targets[t]
                pruned += 1
            if not targets:
                empty_sources.append(source)
        for s in empty_sources:
            del self._connections[s]
        return pruned

    def myelinate(self, min_activations: int, epoch: int) -> int:
        """
        Myelinate synapses that have been activated enough times.
        Returns count of newly myelinated synapses.
        """
        count = 0
        for targets in self._connections.values():
            for syn in targets.values():
                if not syn.myelinated and syn.activation_count >= min_activations:
                    syn.myelinate()
                    count += 1
        return count

    def to_dict(self) -> dict:
        conns = {}
        for source, targets in self._connections.items():
            conns[source] = {t: s.to_dict() for t, s in targets.items()}
        return {
            'name': self.name, 'birth': self.birth_epoch,
            'connections': conns,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Layer:
        layer = cls(d['name'], birth_epoch=d.get('birth', 0))
        for source, targets in d.get('connections', {}).items():
            layer._node_set.add(source)
            layer._connections[source] = {}
            for target, syn_d in targets.items():
                layer._node_set.add(target)
                layer._connections[source][target] = Synapse.from_dict(syn_d)
        return layer


# ═══════════════════════════════════════════════════════════════
#  CORTEX — the full multi-layer semantic memory
# ═══════════════════════════════════════════════════════════════

_TOKEN_RE = re.compile(r'[a-z][a-z0-9-]*')


def _tokenize(text: str) -> List[str]:
    """Extract lowercase alphabetic tokens. No hardcoded stop words."""
    return _TOKEN_RE.findall(text.lower())


class Cortex:
    """
    Multi-layer weighted directed graph with temporal dynamics.

    Layers are created on demand — pass initial_layers to seed,
    or let them emerge as you feed data.
    """

    def __init__(
        self,
        initial_layers: Optional[List[str]] = None,
        prune_interval: int = 100,
        decay_factor: float = 0.97,
        myelinated_decay: float = 0.995,
        myelination_threshold: int = 10,
        critical_period: int = 500,
        min_plasticity: float = 0.1,
        min_weight: float = 0.01,
        max_assembly_size: int = 25,
    ):
        self._layers: Dict[str, Layer] = {}
        self._prune_interval = prune_interval
        self._decay_factor = decay_factor
        self._myelinated_decay = myelinated_decay
        self._myelination_threshold = myelination_threshold
        self._critical_period = critical_period
        self._min_plasticity = min_plasticity
        self._min_weight = min_weight
        self._max_assembly_size = max_assembly_size

        # Global state
        self._epoch = 0
        self._feed_count = 0
        self._last_prune = 0
        self._prune_count = 0
        self._myel_count = 0

        # Word tracking
        self.word_freq: Dict[str, int] = {}
        self.total_words: int = 0

        # Breadth: word -> set of all connected words (any layer)
        self.breadth: Dict[str, Set[str]] = {}

        # Assemblies
        self._assemblies: Dict[str, Assembly] = {}
        self._assembly_seq: int = 0
        self._word_assemblies: Dict[str, List[str]] = {}  # word -> assembly ids

        # Temporal tracking
        self._connection_birth: Dict[str, int] = {}  # word -> epoch first seen

        # Activation cache — invalidated on feed/prune
        self._activation_cache: Dict[str, Dict[str, float]] = {}
        self._confidence_cache: Dict[str, dict] = {}
        self._cache_epoch: int = 0

        # Seed layers
        if initial_layers:
            for name in initial_layers:
                self.ensure_layer(name)
        # Always have a general layer
        self.ensure_layer('_general')

    # ═══ LAYER MANAGEMENT ═══

    def ensure_layer(self, name: str) -> Layer:
        """Get or create a layer. Layers emerge on demand."""
        if name not in self._layers:
            self._layers[name] = Layer(name, birth_epoch=self._epoch)
        return self._layers[name]

    @property
    def layer_names(self) -> List[str]:
        return list(self._layers.keys())

    def get_layer(self, name: str) -> Optional[Layer]:
        return self._layers.get(name)

    # ═══ PLASTICITY ═══

    @property
    def plasticity(self) -> float:
        """
        Learning rate modifier. Starts at 1.0, decays after critical period.
        Early experience shapes the landscape more.
        """
        if self._feed_count < self._critical_period:
            return 1.0
        elapsed = self._feed_count - self._critical_period
        return self._min_plasticity + (1.0 - self._min_plasticity) * math.exp(
            -elapsed / 2000.0
        )

    def _window_for_layer(self, layer_name: str) -> int:
        """Context window size, adaptive to plasticity."""
        p = self.plasticity
        # Different layers naturally have different context reach
        # But this is a DEFAULT — callers can override
        base = 4
        return round(base + p * 2)

    # ═══ FEEDING ═══

    def feed(self, tokens: List[str], layer: str = '_general',
             window: Optional[int] = None, classifier=None) -> int:
        """
        Build connections from a sequence of tokens.

        tokens: list of words (pre-tokenized)
        layer: which semantic layer to connect in
        window: context window size (None = adaptive)
        classifier: optional callable(word) -> layer_name for routing

        Returns: number of connections created/strengthened
        """
        self._epoch += 1

        # Filter content words using emergent stop-word detection
        content = []
        for w in tokens:
            if len(w) < 2:
                continue
            is_new = w not in self.word_freq
            self.word_freq[w] = self.word_freq.get(w, 0) + 1
            if is_new:
                self._connection_birth[w] = self._epoch
            self.total_words += 1
            content.append(w)

        if len(content) < 2:
            return 0

        self._feed_count += 1
        plast = self.plasticity
        win = window if window is not None else self._window_for_layer(layer)
        connections_made = 0

        # Cap per-feed to prevent quadratic blowup on long inputs
        feed_content = content[:20] if len(content) > 20 else content

        for i, src in enumerate(feed_content):
            if src not in self.breadth:
                self.breadth[src] = set()

            for j, tgt in enumerate(feed_content):
                if i == j:
                    continue
                dist = abs(i - j)
                if dist > win:
                    continue

                # Determine target layer
                target_layer = layer
                if classifier is not None:
                    classified = classifier(tgt)
                    if classified:
                        target_layer = classified

                # Distance-weighted strength
                wt = (1.0 / dist) * plast

                # Connect in the appropriate layer
                ly = self.ensure_layer(target_layer)
                ly.connect(src, tgt, wt, self._epoch)
                connections_made += 1

                # Update breadth
                self.breadth[src].add(tgt)
                if tgt not in self.breadth:
                    self.breadth[tgt] = set()
                self.breadth[tgt].add(src)

        # Form assemblies
        if 2 <= len(content) <= self._max_assembly_size:
            self._form_assembly(content)

        # Invalidate caches
        self._invalidate_cache()

        # Maybe prune
        if self._feed_count - self._last_prune >= self._prune_interval:
            self._prune()

        return connections_made

    def feed_text(self, text: str, layer: str = '_general',
                  classifier=None) -> int:
        """Convenience: tokenize then feed."""
        tokens = _tokenize(text)
        return self.feed(tokens, layer=layer, classifier=classifier)

    # ═══ ASSEMBLIES ═══

    def _form_assembly(self, content: List[str]) -> Optional[str]:
        """Form or reinforce an assembly from co-occurring words."""
        content_set = set(content)

        # Reinforce existing assemblies
        reinforced = set()
        for w in content:
            ids = self._word_assemblies.get(w, [])
            for aid in ids:
                if aid in reinforced:
                    continue
                asm = self._assemblies.get(aid)
                if asm is None:
                    continue
                asm.reinforce(self._epoch)
                reinforced.add(aid)
                # Grow assembly with new co-occurring words
                if asm.words and len(asm.words) < self._max_assembly_size:
                    for nw in content:
                        if nw not in asm.words:
                            asm.add(nw, self._epoch)
                            if nw not in self._word_assemblies:
                                self._word_assemblies[nw] = []
                            if aid not in self._word_assemblies[nw]:
                                self._word_assemblies[nw].append(aid)

        # Create new assembly
        if len(content_set) >= 2:
            aid = f'a{self._assembly_seq}'
            self._assembly_seq += 1
            asm = Assembly(
                id=aid, words=content_set, strength=1,
                birth_epoch=self._epoch, last_active=self._epoch,
                max_size=self._max_assembly_size,
            )
            self._assemblies[aid] = asm
            for w in content_set:
                if w not in self._word_assemblies:
                    self._word_assemblies[w] = []
                self._word_assemblies[w].append(aid)
            return aid
        return None

    def _invalidate_cache(self) -> None:
        """Clear activation/confidence caches after state change."""
        self._activation_cache.clear()
        self._confidence_cache.clear()
        self._cache_epoch = self._epoch

    # ═══ ACTIVATION ═══

    def activate(self, word: str, layer: Optional[str] = None,
                 min_weight: float = 0.0) -> Dict[str, float]:
        """
        Activate a word and return its neighborhood with weights.
        If layer is None, aggregates across all layers. Cached.
        """
        word = word.lower()
        if layer is not None:
            ly = self._layers.get(layer)
            if ly is None:
                return {}
            return ly.get_neighbors(word, min_weight)

        # Check cache
        cache_key = word
        if cache_key in self._activation_cache:
            return self._activation_cache[cache_key]

        # Cross-layer aggregation
        combined: Dict[str, float] = {}
        for ly in self._layers.values():
            for tgt, w in ly.get_neighbors(word, min_weight).items():
                combined[tgt] = combined.get(tgt, 0.0) + w

        self._activation_cache[cache_key] = combined
        return combined

    def cross_layer_activation(self, word: str) -> Dict[str, Dict[str, float]]:
        """Per-layer activation map for a word."""
        word = word.lower()
        result = {}
        for name, ly in self._layers.items():
            neighbors = ly.get_neighbors(word)
            if neighbors:
                result[name] = neighbors
        return result

    # ═══ CONFIDENCE ═══

    def confidence(self, word: str) -> dict:
        """
        How well does the cortex know this word?
        Returns score (0-100), state, and diagnostic info. Cached.
        """
        word = word.lower()
        if word in self._confidence_cache:
            return self._confidence_cache[word]
        freq = self.word_freq.get(word, 0)
        if freq == 0:
            return {
                'score': 0, 'state': 'unknown',
                'layers': 0, 'assemblies': 0, 'myelinated': 0,
            }

        b = len(self.breadth.get(word, set()))
        ac = len(self._word_assemblies.get(word, []))

        # Count layers with connections
        lc = 0
        for name, ly in self._layers.items():
            if name == '_general':
                continue
            if ly.get_neighbors(word):
                lc += 1

        # Count myelinated outgoing connections
        ms = 0
        for ly in self._layers.values():
            targets = ly._connections.get(word, {})
            for syn in targets.values():
                if syn.myelinated:
                    ms += 1

        typed_layer_count = max(len(self._layers) - 1, 1)  # exclude _general
        score = round(
            min(freq, 30) / 30 * 25 +
            min(b, 40) / 40 * 20 +
            min(ac, 5) / 5 * 15 +
            lc / typed_layer_count * 20 +
            min(ms, 10) / 10 * 20
        )
        score = min(score, 100)

        if score >= 60:
            state = 'know'
        elif score >= 25:
            state = 'learning'
        else:
            state = 'glimpsed'

        result = {
            'score': score, 'state': state,
            'layers': lc, 'assemblies': ac, 'myelinated': ms,
        }
        self._confidence_cache[word] = result
        return result

    # ═══ IDF ═══

    def idf(self, word: str) -> float:
        """Inverse document frequency analog. Rare words score higher."""
        b = len(self.breadth.get(word, set())) or 1
        V = max(len(self.word_freq), 1)
        return math.log(1 + V / b)

    # ═══ PRUNING ═══

    def _prune(self) -> int:
        """Decay connections and remove weak ones. Prune stale assemblies."""
        self._last_prune = self._feed_count
        self._prune_count += 1
        total_pruned = 0

        for ly in self._layers.values():
            total_pruned += ly.prune(
                self._epoch, self._decay_factor,
                self._myelinated_decay, self._min_weight,
            )
            self._myel_count += ly.myelinate(
                self._myelination_threshold, self._epoch,
            )

        # Prune stale assemblies
        stale_ids = []
        for aid, asm in self._assemblies.items():
            if (asm.dormancy(self._epoch) > self._prune_interval * 3
                    and asm.strength < 3):
                stale_ids.append(aid)
        for aid in stale_ids:
            asm = self._assemblies.pop(aid)
            for w in asm.words:
                wl = self._word_assemblies.get(w, [])
                if aid in wl:
                    wl.remove(aid)

        return total_pruned

    # ═══ HARMONIZE — cross-layer resonance ═══

    def harmonize(self, words_a: List[str], words_b: List[str],
                  weight: float = 0.15) -> None:
        """
        Connect two word groups across layers via _general.
        Expands each group through breadth neighbors, then cross-links.
        """
        expand_a = set(words_a)
        expand_b = set(words_b)
        for w in words_a:
            if w in self.breadth:
                for n in list(self.breadth[w])[:10]:
                    expand_a.add(n)
        for w in words_b:
            if w in self.breadth:
                for n in list(self.breadth[w])[:10]:
                    expand_b.add(n)

        gen = self.ensure_layer('_general')
        for a in expand_a:
            if len(a) <= 2:
                continue
            for b in expand_b:
                if len(b) <= 2 or a == b:
                    continue
                gen.connect(a, b, weight, self._epoch)
                gen.connect(b, a, weight, self._epoch)
                if a not in self.breadth:
                    self.breadth[a] = set()
                if b not in self.breadth:
                    self.breadth[b] = set()
                self.breadth[a].add(b)
                self.breadth[b].add(a)

    # ═══ TEMPORAL QUERIES ═══

    def birth_of(self, word: str) -> int:
        return self._connection_birth.get(word, 0)

    def learned_before(self, a: str, b: str) -> bool:
        return self.birth_of(a) < self.birth_of(b)

    # ═══ TICK (temporal heartbeat) ═══

    def tick(self, used_words: Optional[List[str]] = None,
             confidence_signal: float = 0.5) -> None:
        """
        Temporal heartbeat. Used connections get nurtured.
        Everything else decays gently.
        """
        base_decay = 0.995
        myel_decay = 0.9995
        nurture = 0.3
        floor = 0.01

        # Modulate decay by confidence
        decay_mod = min(base_decay + confidence_signal * 0.003, 0.999)
        used = set(used_words) if used_words else set()

        for ly in self._layers.values():
            for source, targets in ly._connections.items():
                src_used = source in used
                for target, syn in targets.items():
                    tgt_used = target in used
                    if src_used or tgt_used:
                        # Nurture: sigmoid growth
                        syn.weight += nurture * (1 - syn.weight / (syn.weight + 5))
                        syn.last_active = self._epoch
                    else:
                        rate = myel_decay if syn.myelinated else decay_mod
                        syn.weight = max(floor, syn.weight * rate)

        # Gentle frequency decay for unused words
        for w in self.word_freq:
            if w not in used:
                self.word_freq[w] *= 0.9995

    # ═══ STATS ═══

    def stats(self) -> dict:
        layer_stats = {}
        for name, ly in self._layers.items():
            layer_stats[name] = {
                'nodes': ly.node_count,
                'synapses': ly.synapse_count,
            }
        return {
            'epoch': self._epoch,
            'feed_count': self._feed_count,
            'vocabulary': len(self.word_freq),
            'total_words': self.total_words,
            'assemblies': len(self._assemblies),
            'layers': layer_stats,
            'plasticity': round(self.plasticity, 4),
            'prune_count': self._prune_count,
            'myelinated': self._myel_count,
        }

    # ═══ SERIALIZATION ═══

    def to_dict(self) -> dict:
        layers_d = {n: ly.to_dict() for n, ly in self._layers.items()}
        asms_d = {k: v.to_dict() for k, v in self._assemblies.items()}
        breadth_d = {k: sorted(v) for k, v in self.breadth.items()}
        return {
            'layers': layers_d,
            'assemblies': asms_d,
            'assembly_seq': self._assembly_seq,
            'word_assemblies': self._word_assemblies,
            'word_freq': self.word_freq,
            'total_words': self.total_words,
            'breadth': breadth_d,
            'epoch': self._epoch,
            'feed_count': self._feed_count,
            'last_prune': self._last_prune,
            'prune_count': self._prune_count,
            'myel_count': self._myel_count,
            'connection_birth': self._connection_birth,
            'params': {
                'prune_interval': self._prune_interval,
                'decay_factor': self._decay_factor,
                'myelinated_decay': self._myelinated_decay,
                'myelination_threshold': self._myelination_threshold,
                'critical_period': self._critical_period,
                'min_plasticity': self._min_plasticity,
                'min_weight': self._min_weight,
                'max_assembly_size': self._max_assembly_size,
            },
        }

    @classmethod
    def from_dict(cls, d: dict) -> Cortex:
        params = d.get('params', {})
        cx = cls(
            prune_interval=params.get('prune_interval', 100),
            decay_factor=params.get('decay_factor', 0.97),
            myelinated_decay=params.get('myelinated_decay', 0.995),
            myelination_threshold=params.get('myelination_threshold', 10),
            critical_period=params.get('critical_period', 500),
            min_plasticity=params.get('min_plasticity', 0.1),
            min_weight=params.get('min_weight', 0.01),
            max_assembly_size=params.get('max_assembly_size', 25),
        )
        # Restore layers
        for name, ly_d in d.get('layers', {}).items():
            cx._layers[name] = Layer.from_dict(ly_d)
        # Restore assemblies
        for aid, asm_d in d.get('assemblies', {}).items():
            cx._assemblies[aid] = Assembly.from_dict(asm_d)
        cx._assembly_seq = d.get('assembly_seq', 0)
        cx._word_assemblies = d.get('word_assemblies', {})
        cx.word_freq = d.get('word_freq', {})
        cx.total_words = d.get('total_words', 0)
        # Restore breadth
        for k, v in d.get('breadth', {}).items():
            cx.breadth[k] = set(v)
        cx._epoch = d.get('epoch', 0)
        cx._feed_count = d.get('feed_count', 0)
        cx._last_prune = d.get('last_prune', 0)
        cx._prune_count = d.get('prune_count', 0)
        cx._myel_count = d.get('myel_count', 0)
        cx._connection_birth = d.get('connection_birth', {})
        return cx
