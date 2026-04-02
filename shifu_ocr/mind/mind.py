"""
ShifuMind — the orchestrator.

Wires all cognitive subsystems together. Maintains shared graph state.
Implements the full feed → think → respond cycle.

Provides the OCR bridge: predict_candidates() re-ranks OCR word
candidates using cognitive context from the field.

Everything injectable. No hardcoded knowledge.
"""

from __future__ import annotations
import re
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Set, Optional, Tuple, Any

from .cortex import Cortex, _tokenize
from .field import Field
from .gate import Gate
from .signal import Signal
from .trunk import Trunk
from .memory import Memory
from .speaker import Speaker
from .thinker import Thinker
from .imagination import Imagination
from .attention import Attention
from .conviction import Conviction
from .neuron import NeuralField
from .language import Morphology, Syntax, Semantics, Curriculum
from .neuroglia import Neuroglia
from .trn import TRN


class ShifuMind:
    """
    Unified cognitive architecture.

    Feed text → builds knowledge graphs
    Activate/score → queries knowledge via wave propagation
    Predict candidates → bridges OCR perception to cognition
    Deliberate → multi-step reasoning

    All subsystems communicate through shared graph state.
    """

    def __init__(
        self,
        initial_layers: Optional[List[str]] = None,
        seed_domains: Optional[Dict[str, List[str]]] = None,
        cortex_config: Optional[dict] = None,
        field_config: Optional[dict] = None,
        gate_config: Optional[dict] = None,
        memory_capacity: int = 1000,
        thinker_max_steps: int = 10,
    ):
        # ═══ SUBSYSTEMS ═══
        cx_params = cortex_config or {}
        if initial_layers:
            cx_params['initial_layers'] = initial_layers
        self.cortex = Cortex(**cx_params)

        self.field = Field(**(field_config or {}))
        self.gate = Gate(**(gate_config or {}))
        self.signal = Signal()
        self.trunk = Trunk(seed_domains=seed_domains)
        self.memory = Memory(capacity=memory_capacity)
        self.speaker = Speaker()
        self.thinker = Thinker(max_steps=thinker_max_steps)
        self.imagination = Imagination()
        self.attention = Attention()
        self.conviction = Conviction()
        self.neural_field = NeuralField()

        # ═══ NEUROGLIA — the other half of the brain ═══
        # Not just blood flow. Regulation, generation, healing, cooling.
        self.neuroglia = Neuroglia()
        # Each spoke gets its own astrocyte with DIFFERENT thermal tolerance
        # Vertical regulation: identity is fundamental (high tolerance, stable)
        # Higher spokes are more volatile (lower tolerance, quicker to throttle)
        #
        # LAYER REGULATORY HIERARCHY (vertical):
        #   identity    → tolerance=15 (most stable, foundational)
        #   appearance  → tolerance=12 (observable, moderate)
        #   function    → tolerance=10 (active, standard)
        #   mechanism   → tolerance=8  (complex, needs care)
        #   relation    → tolerance=6  (highest-level, most volatile)
        #
        # Vertical rule: lower layers must stabilize before upper layers
        # can form connections. You must know WHAT something IS before
        # you can learn HOW it works.
        self._layer_tolerances = {
            'identity': 15, 'appearance': 12, 'function': 10,
            'mechanism': 8, 'relation': 6, '_general': 20,
        }
        for layer_name, tolerance in self._layer_tolerances.items():
            self.neuroglia.ensure_astrocyte(layer_name, tolerance)

        # ═══ TRN — thalamic reticular nucleus ═══
        # The attentional spotlight. Each spoke has its own TCR channel.
        # TRN sits between them and gates which channels relay.
        # Cortex → TCR (excitatory) + Cortex → TRN (excitatory)
        # TRN → TCR (inhibitory) = feedforward inhibition
        # Non-attended channels suppressed = sensory noise reduced
        self.trn = TRN()
        for layer_name in self._layer_tolerances:
            if layer_name != '_general':
                self.trn.ensure_channel(layer_name)

        # Language acquisition modules
        self.language = type('Language', (), {
            'morphology': Morphology(),
            'syntax': Syntax(),
            'semantics': Semantics(),
            'curriculum': Curriculum(),
        })()

        # ═══ SHARED GRAPH STATE ═══
        self._co_graph: Dict[str, Dict[str, float]] = {}   # weight layer
        self._co_tags: Dict[str, Dict[str, Set[str]]] = {} # semantic relation layer
        self._nx_graph: Dict[str, Dict[str, float]] = {}

        # ═══ TRACKING ═══
        self._feed_count = 0
        self._epoch = 0

    # ═══════════════════════════════════════════════════════════
    #  EDGE TAGGING — the node stores the concept, the edge
    #  stores the relationship, the goal shapes the path
    # ═══════════════════════════════════════════════════════════

    def add_edge_tag(self, source: str, target: str, tag: str) -> None:
        """Tag an edge with a relationship type. Additive — edges can have multiple tags."""
        # Ensure co-graph edge exists (tags without edges are orphaned)
        if source not in self._co_graph:
            self._co_graph[source] = {}
        if target not in self._co_graph[source]:
            self._co_graph[source][target] = 1.0
        # Add tag
        if source not in self._co_tags:
            self._co_tags[source] = {}
        if target not in self._co_tags[source]:
            self._co_tags[source][target] = set()
        self._co_tags[source][target].add(tag)

    def get_edge_tags(self, source: str, target: str) -> Set[str]:
        """What relationship types does this edge carry?"""
        return self._co_tags.get(source, {}).get(target, set())

    def edges_by_tag(self, word: str, tag: str) -> Dict[str, float]:
        """Get all edges from a word filtered by relationship tag. Returns {target: weight}."""
        co = self._co_graph.get(word, {})
        tags = self._co_tags.get(word, {})
        return {target: weight for target, weight in co.items()
                if tag in tags.get(target, set())}

    def edges_by_goal(self, word: str, goal: str) -> List[Tuple[str, float]]:
        """
        Goal-biased edge selection. Not hard-filtered — soft-scored.
        Primary tag gets bonus. Supporting tags get smaller bonus.
        Untagged edges still included but scored lower.

        Goal shapes the path, but doesn't imprison it.
        """
        # Goal → primary tag + supporting tags
        goal_map = {
            'identity': ('id', {'rel', 'mech'}),
            'mechanism': ('mech', {'id', 'rel'}),
            'function': ('fn', {'mech', 'rel'}),
            'appearance': ('app', {'id', 'rel'}),
            'relation': ('rel', {'id', 'mech'}),
        }
        primary, supports = goal_map.get(goal, ('rel', {'id', 'mech', 'fn', 'app'}))

        co = self._co_graph.get(word, {})
        tags = self._co_tags.get(word, {})
        scored = []
        for target, weight in co.items():
            edge_tags = tags.get(target, set())
            score = weight
            if primary in edge_tags:
                score *= 3.0     # Strong goal match
            elif edge_tags & supports:
                score *= 1.5     # Supporting relation
            # Untagged edges keep base weight — not rejected
            scored.append((target, score))
        scored.sort(key=lambda x: -x[1])
        return scored

    def semantic_coverage(self, word: str) -> Dict[str, Any]:
        """
        How rich is this word's semantic knowledge?
        Replaces layer-membership checking.

        Returns: {
            'tags': set of all relationship tags on this word's edges,
            'tag_counts': {tag: count of edges with this tag},
            'total_edges': int,
            'coverage': float (0-1, how many tag types are represented),
            'strength': float (average edge weight),
        }
        """
        co = self._co_graph.get(word, {})
        tags = self._co_tags.get(word, {})
        all_tags: Set[str] = set()
        tag_counts: Dict[str, int] = {}
        for target in co:
            edge_tags = tags.get(target, set())
            all_tags.update(edge_tags)
            for t in edge_tags:
                tag_counts[t] = tag_counts.get(t, 0) + 1
        total = len(co)
        possible_tags = 5  # id, app, fn, mech, rel
        return {
            'tags': all_tags,
            'tag_counts': tag_counts,
            'total_edges': total,
            'coverage': len(all_tags) / possible_tags if possible_tags else 0,
            'strength': sum(co.values()) / max(total, 1),
        }

    # ═══════════════════════════════════════════════════════════
    #  SELF — the current state IS the prediction
    # ═══════════════════════════════════════════════════════════

    def self_predict(self, word: str) -> Dict[str, float]:
        """
        What does the self expect to see near this word?
        The current co-graph IS the prediction model.
        Returns normalized probabilities.
        """
        co = self._co_graph.get(word, {})
        total = sum(co.values())
        if total == 0:
            return {}
        return {w: v / total for w, v in co.items()}

    def self_surprise(self, tokens: List[str]) -> float:
        """
        How surprising is this sentence to the self?

        Uses TWO levels of prediction:
        1. Word familiarity (word_freq) — have I heard these words before?
        2. Co-occurrence (co_graph) — have I seen these words TOGETHER?

        Both contribute. A sentence with all known words but novel
        combinations is still surprising. A sentence with unknown
        words is maximally surprising.
        """
        if len(tokens) < 2:
            return 1.0

        # Level 1: word familiarity surprise
        total_freq = sum(self.cortex.word_freq.values()) or 1
        word_surprises = []
        for w in tokens:
            if len(w) <= 2:
                continue
            freq = self.cortex.word_freq.get(w, 0)
            # Rare word = high surprise. Common word = low surprise.
            familiarity = min(freq / max(total_freq * 0.01, 1), 1.0)
            word_surprises.append(1.0 - familiarity)

        # Level 2: co-occurrence surprise (only if co-graph exists)
        pair_surprises = []
        for i, word in enumerate(tokens):
            if len(word) <= 2:
                continue
            predicted = self.self_predict(word)
            if not predicted:
                continue
            others = [t for j, t in enumerate(tokens) if j != i and len(t) > 2]
            if not others:
                continue
            expected = sum(predicted.get(o, 0) for o in others)
            pair_surprises.append(1.0 - min(expected, 1.0))

        # Combine: word-level + pair-level
        all_surprises = word_surprises + pair_surprises
        return sum(all_surprises) / len(all_surprises) if all_surprises else 1.0

    def self_precision(self, word: str) -> float:
        """
        How confident is the self about this word?
        High freq + many edges = high precision = strong predictions.
        Low freq + few edges = low precision = weak predictions.

        High precision → suppress expected input more.
        Low precision → allow everything through (critical period).
        """
        freq = self.cortex.word_freq.get(word, 0)
        edges = len(self._co_graph.get(word, {}))
        # Precision grows with experience but saturates
        raw = (freq * 0.1 + edges * 0.2)
        return min(raw / (raw + 5.0), 0.9)  # Sigmoid-like, max 0.9

    # ═══════════════════════════════════════════════════════════
    #  FEEDING — absorb information
    # ═══════════════════════════════════════════════════════════

    def feed(self, text: str, layer: str = '_general',
             classifier=None) -> dict:
        """
        EPISODIC-FIRST single feed.

        The brain remembers episodes first, semantics later.
        Same principle as feed_batch but for one sentence.

        1. Gate filters
        2. Store episodic memory
        3. Update word_freq (familiarity)
        4. Birth neurons
        5. NOTHING ELSE — semantics emerge during replay
        """
        self._epoch += 1

        # 1. Gate
        filtered = self.gate.filter(
            text,
            word_freqs=self.cortex.word_freq,
            known_vocab=set(self.cortex.word_freq.keys()) if self.cortex.word_freq else None,
        )
        if not filtered['accepted']:
            return {
                'accepted': False, 'tokens_absorbed': 0,
                'domain': None, 'quality': filtered['quality'],
            }

        tokens = filtered['tokens']
        content = filtered['content_tokens']

        # Thymic selection (after critical period)
        if self._feed_count > 200:
            known_count = sum(1 for w in content if w in self.cortex.word_freq and self.cortex.word_freq[w] > 1)
            if known_count == 0 and len(content) > 3:
                return {
                    'accepted': False, 'tokens_absorbed': 0,
                    'domain': None, 'quality': filtered['quality'],
                }

        # 2. Familiarity — word frequency
        for w in content:
            self.cortex.word_freq[w] = self.cortex.word_freq.get(w, 0) + 1
            self.cortex.total_words += 1
            if w not in self.cortex._connection_birth:
                self.cortex._connection_birth[w] = self.cortex._epoch
            if w not in self.cortex.breadth:
                self.cortex.breadth[w] = set()
        self.cortex._feed_count += 1
        self.cortex._epoch += 1

        # 3. Episodic memory
        shift = self.memory.detect_topic_shift(content)
        significance = shift * filtered['quality']
        self.memory.record(
            epoch=self._epoch, tokens=content,
            significance=significance,
            context={'all_tokens': tokens[:30], 'quality': filtered['quality']},
            timestamp=time.time(),
            raw_text=text,
            force=True,
        )

        # 4. Birth neurons (no connections yet — those come during replay)
        for w in content:
            if len(w) > 3:
                self.neural_field.ensure_neuron(w)

        self._feed_count += 1

        return {
            'accepted': True,
            'tokens_absorbed': len(content),
            'domain': None,
            'quality': filtered['quality'],
        }

    def feed_batch(self, texts: List[str], layer: str = '_general',
                   classifier=None, cycles: int = 1) -> dict:
        """
        DEVELOPMENTAL FEEDING — the baby eats what it can digest.

        Don't feed an elephant to a baby.
        Milk first. Puree later. Steak when teeth grow.

        The amount the baby can digest depends on its current size:
          vocab < 50:   MILK — 1 sentence at a time, immediate replay
          vocab < 500:  PUREE — 10 sentences, then replay batch
          vocab < 5000: SOLID — 100 sentences, then replay
          vocab > 5000: ADULT — full batch, replay later

        Each sentence is stored as an episode (episodic-first).
        Then immediately replayed (digestion) — proportional to
        what was just eaten. The brain doesn't store the whole
        elephant in one place. Visual goes to visual cortex,
        semantic to temporal, episodic to hippocampus.
        """
        vocab = len(self.cortex.word_freq)

        # How much can the baby digest per bite?
        if vocab < 50:
            bite_size = 5       # Milk
        elif vocab < 500:
            bite_size = 20      # Puree
        elif vocab < 5000:
            bite_size = 100     # Solid food
        else:
            bite_size = 500     # Adult

        stops = self.gate.stop_words(self.cortex.word_freq)
        accepted = 0
        rejected = 0
        neurons_born = 0
        replayed = 0
        seen_content = set()

        # Feed in BITES — not the whole elephant
        for bite_start in range(0, len(texts), bite_size):
            bite = texts[bite_start:bite_start + bite_size]

            for text in bite:
                tokens = self.gate.tokenize(text)
                if len(tokens) < 2:
                    rejected += 1
                    continue

                content = [t for t in tokens if t not in stops and len(t) > 2]
                if len(content) < 2:
                    rejected += 1
                    continue

                content_key = tuple(sorted(content[:8]))
                if content_key in seen_content:
                    accepted += 1
                    continue
                seen_content.add(content_key)

                self._epoch += 1
                self.cortex._epoch += 1

                # Familiarity
                for w in content:
                    self.cortex.word_freq[w] = self.cortex.word_freq.get(w, 0) + 1
                    self.cortex.total_words += 1
                    if w not in self.cortex._connection_birth:
                        self.cortex._connection_birth[w] = self.cortex._epoch
                    if w not in self.cortex.breadth:
                        self.cortex.breadth[w] = set()
                self.cortex._feed_count += 1

                # Episodic memory — significance = surprise
                # The self predicts what it expects. Surprising input
                # gets higher significance → survives longer in memory.
                # Expected input gets lower significance → evicted sooner.
                surprise = self.self_surprise(content)
                self.memory.record(
                    epoch=self._epoch, tokens=content,
                    significance=max(0.1, surprise),
                    context={'all_tokens': tokens[:30], 'surprise': round(surprise, 3)},
                    timestamp=time.time(), raw_text=text, force=True,
                )

                # Birth neurons
                for w in content:
                    if len(w) > 3:
                        self.neural_field.ensure_neuron(w)
                        neurons_born += 1

                accepted += 1
                self._feed_count += 1

            # ═══ DIGEST after each bite ═══
            # Immediate replay of what was just eaten.
            # Like digestion — break down the food NOW, don't wait.
            # The brain stores parts in different regions during replay:
            #   CP1: Golgi tags → which pathway
            #   CP2: co-graph, nx-graph → local preprocessing
            #   CP3: TRN gating → what deserves attention
            #   CP4: neural field wiring → structural connections
            #   CP5: identity, spokes → higher integration
            unreplayed = self.memory.unreplayed(k=bite_size)
            if unreplayed:
                r = self.replay_batch(unreplayed, max_episodes=bite_size)
                replayed += r.get('replayed', 0)

        self.gate.adapt_thresholds()

        return {
            'total': len(texts),
            'accepted': accepted,
            'rejected': rejected,
            'neurons_born': neurons_born,
            'episodes_stored': accepted,
            'replayed': replayed,
            'bite_size': bite_size,
            'stage': 'milk' if vocab < 50 else 'puree' if vocab < 500 else 'solid' if vocab < 5000 else 'adult',
        }

    # ═══════════════════════════════════════════════════════════
    #  RELAY CHECKPOINTS — signal earns passage through stages
    # ═══════════════════════════════════════════════════════════
    #
    # Not everything goes to cortex raw. Each stream earns passage
    # through checkpoints until the receiving system is ready.
    #
    #   CP1: Specialist receptors — Golgi tags, pathway classification
    #   CP2: Local preprocessing — co-graph, nx-graph, breadth (provenance preserved)
    #   CP3: Relay/gating — TRN decides what passes, neuroglia regulates heat
    #   CP4: Task-ready handoff — neural field wiring, speaker frames
    #   CP5: Higher integration — identity, spokes, syntax (only when lower layers stable)
    #
    # Each checkpoint can HOLD data in staging buffers until the
    # receiving checkpoint is ready. Like thalamic relay nuclei.

    @staticmethod
    def _word_pathway(word: str) -> int:
        """Classify word by length into processing pathway.
        1=structural (len<=3), 2=connector (len<=5), 3=content (len<=8), 4=specialist (len>=9)."""
        n = len(word)
        if n <= 3:
            return 1
        elif n <= 5:
            return 2
        elif n <= 8:
            return 3
        return 4

    def _cp1_specialize(self, episode) -> dict:
        """
        CHECKPOINT 1: Specialist receptors.

        Each word gets classified by length into a processing pathway.
        Short words (structural) get shallow processing.
        Long words (specialist) get deep processing.
        """
        content = episode.tokens
        # Classify each token by pathway
        classified = [(w, self._word_pathway(w)) for w in content]
        return {
            'classified': classified,
            'content': content,
            'raw_text': episode.raw_text,
            'all_tokens': episode.context.get('all_tokens', content),
            'replay_num': episode.replayed,
        }

    def _cp2_preprocess(self, signal: dict) -> dict:
        """
        CHECKPOINT 2: Local preprocessing.

        Build co-graph (pathway-isolated), nx-graph, breadth.
        No cross-contamination: content <-> content, structural <-> structural.
        Uses predictive coding: surprising co-occurrences get stronger updates.
        """
        classified = signal['classified']
        content = signal['content']
        all_tokens = signal['all_tokens']
        connections = 0

        # Co-graph with pathway isolation + PREDICTIVE CODING
        caps = {1: 30, 2: 50, 3: 100, 4: 100}
        for w, origin in classified:
            cap = caps.get(origin, 100)
            if w not in self._co_graph:
                self._co_graph[w] = {}
            co_w = self._co_graph[w]
            predicted = self.self_predict(w)
            precision = self.self_precision(w)
            for ob, ob_origin in classified:
                if ob == w:
                    continue
                # Pathway isolation: content only with content, structural only with structural
                if origin >= 3 and ob_origin < 3:
                    continue
                if origin <= 2 and ob_origin > 2:
                    continue
                expected = predicted.get(ob, 0)
                surprise = 1.0 - expected
                suppression = precision * expected
                update = max(0.1, 1.0 + surprise * 2.0 - suppression * 3.0)
                if ob in co_w:
                    co_w[ob] += update
                elif len(co_w) < cap:
                    co_w[ob] = update
            connections += 1

        # Breadth for content words (pathway >= 3)
        content_items = [(w, o) for w, o in classified if o >= 3]
        for w, _ in content_items:
            b = self.cortex.breadth.get(w)
            if b is not None and len(b) < 50:
                for ob, ob_o in content_items[:10]:
                    if ob != w and ob_o >= 3:
                        b.add(ob)

        # Nx-graph (grammar transitions -- ALL tokens)
        for i in range(len(all_tokens) - 1):
            w = all_tokens[i]
            if w not in self._nx_graph:
                self._nx_graph[w] = {}
            nx_w = self._nx_graph[w]
            nxt = all_tokens[i + 1]
            if nxt in nx_w:
                nx_w[nxt] += 1
            elif len(nx_w) < 50:
                nx_w[nxt] = 1

        signal['connections'] = connections
        return signal

    def _cp3_gate(self, signal: dict) -> dict:
        """
        CHECKPOINT 3: Relay/gating layer.

        TRN decides what deserves passage. Neuroglia monitors heat.
        Non-attended channels are suppressed.
        This is where attention meets regulation.
        """
        connections = signal.get('connections', 0)

        # TRN: broad attention during replay (all channels partially open)
        self.trn.broad_attention()

        # Neuroglia: report activity, check thermal state
        self.neuroglia.observe_region('_general', connections)

        # Horizontal regulation: lateral inhibition in co-graph
        classified = signal['classified']
        for w, origin in classified:
            if origin >= 3:
                co_w = self._co_graph.get(w, {})
                if len(co_w) > 30:
                    sorted_conns = sorted(co_w.items(), key=lambda x: -x[1])
                    threshold = sorted_conns[29][1] if len(sorted_conns) > 29 else 0
                    for neighbor, weight in list(co_w.items()):
                        if weight < threshold:
                            co_w[neighbor] = weight * 0.8

        # Gate decision: should this signal pass to CP4?
        signal['gated'] = not self.neuroglia.is_overheated()
        return signal

    def _cp4_handoff(self, signal: dict) -> dict:
        """
        CHECKPOINT 4: Task-ready cortical handoff.

        Wire neural field connections. Teach the speaker.
        The signal is now in a format the cortex can use.
        """
        if not signal.get('gated', True):
            return signal  # Blocked at CP3 -- hold for next cycle

        all_tokens = signal['all_tokens']
        classified = signal['classified']

        # Neural field: sequential connections (tongue learns order)
        _connect = getattr(self.neural_field, 'connect', None)
        for i in range(len(all_tokens) - 1):
            w, nxt = all_tokens[i], all_tokens[i + 1]
            if len(w) < 2 or len(nxt) < 2:
                continue
            self.neural_field.ensure_neuron(w)
            self.neural_field.ensure_neuron(nxt)
            if _connect:
                _connect(w, nxt, 0.4)
            else:
                self.neural_field.neurons[w].add_connection(nxt, 0.4)

        # Neural field: content co-occurrence (pathway >= 2, max 10)
        content_words = [w for w, o in classified if o >= 2][:10]
        for i, w in enumerate(content_words):
            self.neural_field.ensure_neuron(w)
            for j in range(i + 1, min(i + 5, len(content_words))):
                other = content_words[j]
                if other != w:
                    weight = 0.3 if len(w) > 3 else 0.15
                    if _connect:
                        _connect(w, other, weight)
                    else:
                        self.neural_field.neurons[w].add_connection(other, weight)

        # Speaker learns transitions (tongue)
        self.speaker.learn_frame(all_tokens)

        return signal

    def _cp5_integrate(self, signal: dict) -> dict:
        """
        CHECKPOINT 5: Higher integration.

        Combine streams only after they have become usable.
        Identity extraction, spoke routing, syntax learning.
        Each layer must stabilize before the next activates.

        Data that can't be integrated yet is STAGED — held in
        buffers until the receiving layer is ready.
        """
        if not signal.get('gated', True):
            return signal

        content = signal['content']
        raw_text = signal['raw_text']
        all_tokens = signal['all_tokens']
        replay_num = signal['replay_num']

        if not hasattr(self, '_layer_staging'):
            self._layer_staging: Dict[str, List[dict]] = {}

        # ═══ IDENTITY (needs replay >= 1, general layer settled) ═══
        id_depth = self.neuroglia.processing_depth('identity')
        if replay_num >= 1 and raw_text:
            if id_depth > 0.3:
                self._extract_identity(raw_text, content)
                self.neuroglia.observe_region('identity', len(content))
            else:
                if 'identity' not in self._layer_staging:
                    self._layer_staging['identity'] = []
                self._layer_staging['identity'].append({
                    'raw_text': raw_text, 'content': content,
                })

        # Drain identity staging
        if id_depth > 0.3 and self._layer_staging.get('identity'):
            staged = self._layer_staging.pop('identity')
            for item in staged[:5]:
                self._extract_identity(item['raw_text'], item['content'])
            self.neuroglia.observe_region('identity', len(staged))

        # ═══ SYNTAX (needs replay >= 2, identity stable) ═══
        if replay_num >= 2:
            id_astro = self.neuroglia.astrocytes.get('identity')
            id_stable = not id_astro or id_astro.heat < id_astro.tolerance
            if id_stable:
                self.language.syntax.feed(all_tokens)
            else:
                if 'syntax' not in self._layer_staging:
                    self._layer_staging['syntax'] = []
                self._layer_staging['syntax'].append({'tokens': all_tokens})

        # Drain syntax staging
        if self._layer_staging.get('syntax'):
            id_astro = self.neuroglia.astrocytes.get('identity')
            if not id_astro or id_astro.heat < id_astro.tolerance:
                staged = self._layer_staging.pop('syntax')
                for item in staged[:5]:
                    self.language.syntax.feed(item['tokens'])

        return signal

    def replay_episode(self, episode) -> dict:
        """
        Replay = run the signal through all 5 checkpoints.

        CP1: specialize → CP2: preprocess → CP3: gate → CP4: handoff → CP5: integrate

        Not everything goes to cortex raw. Each stream earns passage
        through checkpoints until the receiving system is ready for it.
        """
        if len(episode.tokens) < 2:
            episode.replayed += 1
            return {'extracted': 0}

        # Run the relay chain
        signal = self._cp1_specialize(episode)
        signal = self._cp2_preprocess(signal)
        signal = self._cp3_gate(signal)
        signal = self._cp4_handoff(signal)
        signal = self._cp5_integrate(signal)

        episode.replayed += 1

        return {
            'extracted': signal.get('connections', 0),
            'replay_depth': episode.replayed,
            'tokens': len(episode.tokens),
            'gated': signal.get('gated', True),
        }

    def replay_batch(self, episodes: list, max_episodes: int = 20) -> dict:
        """
        Replay multiple episodes in one pass.
        Called during delta state for bulk semantic extraction.

        NEUROGLIA REGULATED: if the brain overheats, stop replaying.
        Astrocytes monitor thermal load per region. If any region
        overheats, we pause and let it cool before continuing.
        """
        total_extracted = 0
        replayed = 0

        for ep in episodes[:max_episodes]:
            # ═══ THERMAL CHECK — astrocytes cool the brain ═══
            if self.neuroglia.is_overheated():
                break  # Let the brain cool down

            r = self.replay_episode(ep)
            total_extracted += r['extracted']
            replayed += 1

        # ═══ NEUROGLIA CYCLE — regulation, generation, healing ═══
        glia_result = self.neuroglia.cycle(self.neural_field, self._co_graph)

        # Post-replay: update caches
        if replayed > 0:
            self.field.update_medians(self.cortex.word_freq, self._co_graph)
            self.field.invalidate_cache()
            self.cortex._invalidate_cache()

        return {
            'replayed': replayed,
            'extracted': total_extracted,
            'glia': {
                'heat': glia_result.get('global_heat', 0),
                'throttle': glia_result.get('global_throttle', 0),
                'patrol': glia_result.get('patrol'),
                'neurogenesis': glia_result.get('neurogenesis'),
            },
        }

    # ═══ IDENTITY IS THE BRIDGE ═══
    #
    # Identity is NEVER a destination for property connections.
    # Identity only holds "X is a Y" declarations — the name tag.
    #
    # Spoke routing is LEARNED from the co-occurrence graph:
    # - Words that co-occur with MANY other content words → likely function words (go to _general)
    # - Words whose neighbors cluster in a single spoke → route to that spoke
    # - The routing is discovered, not prescribed

    def _extract_identity(self, raw_text: str, content: List[str]) -> None:
        """
        Extract copula patterns ("X is a Y").
        DUAL-WRITE: writes to both identity layer (backward compat) AND _co_tags.
        """
        if not content:
            return
        text_lower = raw_text.lower()
        identity_layer = self.cortex.ensure_layer('identity')
        subject = content[0]

        import re
        match = re.search(
            r'(?:is a|is an|is the|are|defined as|refers to|known as)\s+(\w{3,})',
            text_lower,
        )
        if not match:
            match2 = re.search(r'\b(\w{3,})\s+(?:\w+ )?is\s+(\w{4,})', text_lower)
            if match2:
                a = match2.group(1)
                b = match2.group(2)
                if a == subject and b != subject and len(b) > 2:
                    identity_layer.connect(a, b, 2.0, self.cortex._epoch)
                    identity_layer.connect(b, a, 1.0, self.cortex._epoch)
                    # Tag the edge
                    self.add_edge_tag(a, b, 'id')
                    self.add_edge_tag(b, a, 'id')
                    return

        if match:
            b = match.group(1)
            if b != subject and len(b) > 2:
                identity_layer.connect(subject, b, 2.0, self.cortex._epoch)
                identity_layer.connect(b, subject, 1.0, self.cortex._epoch)
                # Tag the edges
                self.add_edge_tag(subject, b, 'id')
                self.add_edge_tag(b, subject, 'id')
                if len(content) > 1 and content[1] != subject and content[1] != b:
                    identity_layer.connect(content[1], b, 1.0, self.cortex._epoch)
                    self.add_edge_tag(content[1], b, 'id')

    # ═══════════════════════════════════════════════════════════
    #  QUERYING — activate and score
    # ═══════════════════════════════════════════════════════════

    def activate(self, word: str) -> Dict[str, float]:
        """
        Activate a word. Each neuron fires independently.
        The wave propagates through axon connections.
        No central controller — just neurons talking to neurons.
        """
        # Use neural field if it has neurons, fall back to old field
        if self.neural_field.neurons:
            return self.neural_field.activate(word)
        return self.field.activate(
            word, self._co_graph, self._nx_graph,
            word_freqs=self.cortex.word_freq,
        )

    def score(self, tokens: List[str]) -> dict:
        """Score coherence through neural propagation."""
        if self.neural_field.neurons:
            return self.neural_field.score_sequence(tokens)
        return self.field.score_sequence(
            tokens, self._co_graph, self._nx_graph,
            word_freqs=self.cortex.word_freq,
        )

    def score_text(self, text: str) -> dict:
        tokens = _tokenize(text)
        return self.score(tokens)

    # ═══════════════════════════════════════════════════════════
    #  OCR BRIDGE — perception meets cognition
    # ═══════════════════════════════════════════════════════════

    def predict_candidates(
        self,
        candidates: List[Tuple[str, float]],
        context_words: Optional[List[str]] = None,
        ocr_weight: float = 0.4,
        field_weight: float = 0.6,
    ) -> List[dict]:
        """
        Re-rank OCR word candidates using cognitive context.

        This is THE BRIDGE between perception (OCR engines) and
        cognition (field activation). The OCR engines propose
        candidates with confidence scores. The mind evaluates
        which candidate fits the semantic context.

        candidates: [(word, ocr_confidence), ...]
        context_words: surrounding words for context field
        """
        ctx = context_words or []
        return self.field.score_ocr_candidates(
            candidates, ctx,
            self._co_graph, self._nx_graph,
            word_freqs=self.cortex.word_freq,
            ocr_weight=ocr_weight,
            field_weight=field_weight,
        )

    # ═══════════════════════════════════════════════════════════
    #  DELIBERATION — multi-step reasoning
    # ═══════════════════════════════════════════════════════════

    # ═══ INTERNAL COMPASS — the baby chooses ═══

    def compass(self) -> dict:
        """
        The baby looks inward and decides what it needs.
        Not forced. Not scheduled. It reads its own state
        and says what it wants to do next.

        Returns: {
            'state': what the baby is feeling,
            'desire': what it wants to do,
            'reason': why,
            'action': the command to execute (or None to rest)
        }
        """
        vocab = len(self.cortex.word_freq)
        trend = self.signal.recent_trend(5)
        myel = self.cortex._myel_count

        # The baby reads its own landscape and decides.

        # How many typed layers have content? (not just _general)
        typed_layers = sum(
            1 for n in self.cortex.layer_names
            if n != '_general' and self.cortex.get_layer(n) and self.cortex.get_layer(n).synapse_count > 0
        )

        # Newborn — barely any words
        if vocab < 30:
            return {
                'state': 'newborn',
                'desire': 'I need to see the world. Feed me.',
                'reason': f'I only know {vocab} words.',
                'action': None,
            }

        # Absorbed but unorganized — has words but no typed layers OR no domains
        domains = len(self.trunk.domains)
        if vocab > 50 and (typed_layers == 0 or domains == 0):
            return {
                'state': 'absorbing',
                'desire': 'I have taken in a lot. Let me organize.',
                'reason': f'{vocab} words but no typed layers yet. I need to consolidate.',
                'action': 'consolidate',
            }

        # Organized but unpracticed — has layers but hasn't tried using them
        if typed_layers > 0 and self.signal._total_observations < 5:
            return {
                'state': 'ready',
                'desire': 'I have organized my knowledge. Let me try using it.',
                'reason': f'{typed_layers} typed layers ready. Time to practice.',
                'action': 'practice',
            }

        # ═══ FATIGUE CHECK — adenosine model ═══
        # If fatigued or temporarily satisfied, rest.
        # The soldier puts down the gun. Not surrender — just rest.
        if self.conviction.is_fatigued() or self.conviction.is_satisfied():
            # Try to wake if conditions improved
            self.conviction.wake(self._epoch)
            if self.conviction.is_satisfied():
                voice = self.conviction._satisfaction_voice or 'I need to rest.'
                return {
                    'state': 'satisfied',
                    'desire': voice,
                    'reason': f'Adenosine: {self.conviction._adenosine:.1f}/{self.conviction._adenosine_threshold}. Resting until the road is clearer.',
                    'action': None,
                }

        # Has conviction?
        purpose = self.conviction.discover_purpose(self)

        # Still learning? (surprise signal positive)
        if trend > 0.4:
            return {
                'state': 'curious',
                'desire': 'I am still discovering. Let me practice more.',
                'reason': f'My surprise is {trend:.2f} — there is learning here.',
                'action': 'practice',
            }

        # Plateau but has conviction (AND not fatigued)
        if purpose and not self.conviction.is_fatigued():
            return {
                'state': 'determined',
                'desire': f'The landscape is flat but I feel there is more. {purpose}.',
                'reason': f'Dopamine is low ({trend:.2f}) but conviction pulls me forward. Adenosine: {self.conviction._adenosine:.1f}',
                'action': 'practice',
            }

        # Content — natural rest, not forced
        return {
            'state': 'resting',
            'desire': 'I am content for now. Talk to me or feed me more.',
            'reason': f'{vocab} words, {myel} myelinated, {typed_layers} layers.',
            'action': None,
        }

    # ═══ GHRELIN — hunger receptors for specific knowledge ═══

    def hunger_receptors(self) -> dict:
        """
        Like mechanoreceptors in the stomach detecting which nutrients are low.
        Scans every known concept for THIN SPOKES.

        Returns: {
            'cries': [{'word': concept, 'missing': [spokes], 'need': description}],
            'satiated': [words with full spokes],
            'hungriest': the most urgent need,
        }
        """
        all_layers = [n for n in self.cortex.layer_names if n != '_general']
        if not all_layers:
            return {'cries': [], 'satiated': [], 'hungriest': None}

        cries = []
        satiated = []

        # Scan top words by frequency — the ones the baby uses most
        top_words = sorted(
            [(w, f) for w, f in self.cortex.word_freq.items() if len(w) > 4 and f > 2],
            key=lambda x: -x[1],
        )[:100]

        for word, freq in top_words:
            layers = self.cortex.cross_layer_activation(word)
            present = [n for n, conns in layers.items() if conns and n != '_general']
            missing = [n for n in all_layers if n not in present]

            if not missing:
                satiated.append(word)
            elif present:  # Has SOME spokes but not all — this is a real gap
                # What specifically does it need?
                need_descriptions = {
                    'identity': f'what {word} IS (definition)',
                    'appearance': f'how {word} PRESENTS (symptoms/signs)',
                    'function': f'what {word} is USED FOR (treatment/purpose)',
                    'mechanism': f'HOW {word} WORKS (pathway/cause)',
                    'relation': f'what {word} RELATES TO (associations/risk)',
                }
                needs = [need_descriptions.get(m, m) for m in missing]
                cries.append({
                    'word': word,
                    'have': present,
                    'missing': missing,
                    'need': needs[0] if needs else 'more information',
                    'urgency': len(missing) / max(len(all_layers), 1),
                })

        cries.sort(key=lambda x: -x['urgency'])
        hungriest = cries[0] if cries else None

        return {
            'cries': cries[:10],
            'satiated': satiated[:10],
            'hungriest': hungriest,
        }

    def cry(self) -> str:
        """
        The baby cries for what it needs.
        Not a generic "I'm hungry." A specific cry for specific food.
        Like ghrelin targeting specific nutrient deficiencies.
        """
        receptors = self.hunger_receptors()
        hungriest = receptors.get('hungriest')

        if not hungriest:
            if not self.cortex.word_freq:
                return "I am empty. Feed me anything. I need to see the world."
            return "I am content. My spokes are balanced."

        word = hungriest['word']
        need = hungriest['need']
        have = ', '.join(hungriest['have'])
        missing = ', '.join(hungriest['missing'])
        urgency = hungriest['urgency']

        if urgency > 0.6:
            # Desperate cry
            return f"I NEED to know {need}. I know {word} through {have} but {missing} is completely dark. Please feed me text about this."
        elif urgency > 0.3:
            # Moderate hunger
            return f"I want to know {need}. My understanding of {word} is missing {missing}."
        else:
            # Mild curiosity
            return f"I'm curious about {need}. {word} is mostly understood but {missing} would complete it."

    def introspect(self) -> str:
        """The baby speaks about itself — what it's doing and what it needs."""
        c = self.compass()
        cry_text = self.cry()
        voice = self.conviction._voice[-1] if self.conviction._voice else None
        parts = [c['desire']]
        if cry_text and 'content' not in cry_text:
            parts.append(cry_text)
        if voice:
            parts.append(voice)
        return ' '.join(parts)

    def autonomous_step(self) -> dict:
        """
        The baby does ONE thing on its own and reports what it did.
        Always runs heartbeat (blood supply never stops).
        Then does what the compass says.
        """
        # HEARTBEAT — always runs, like blood supply
        hb = self.heartbeat()
        myel_new = hb.get('myelinated_new', 0)

        c = self.compass()
        action = c.get('action')
        voice = c['desire']

        if action == 'consolidate':
            r = self.consolidate(focus_size=200)
            result_text = f"routed {r.get('routed',0)}, {r.get('identities',0)} id, {myel_new} myel"
            return {'did': 'consolidate', 'result': result_text, 'voice': voice, 'state': c['state']}
        elif action == 'practice':
            r = self.practice(rounds=5)  # 5 rounds = 5.0 weight = myelinate in one session
            result_text = f"{r.get('improved',0)} reinforced, {myel_new} myel"
            return {'did': 'practice', 'result': result_text, 'voice': r.get('voice', voice), 'state': c['state']}
        else:
            result_text = f"resting, {myel_new} myel"
            return {'did': 'rest', 'result': result_text, 'voice': voice, 'state': c['state']}

    def deliberate(self, query: str) -> dict:
        """
        specialize -> TRN gate -> relay -> re-specialize

        The TRN gates which spokes contribute to deliberation.
        "What causes stroke?" -> TRN attends to MECHANISM spoke,
        suppresses appearance/relation. Only mechanism information
        relays through. Noise from irrelevant modalities reduced.
        """
        tokens = _tokenize(query)

        # SPECIALIZE: classify each token by pathway
        classified = [(w, self._word_pathway(w)) for w in tokens]
        content = [w for w, o in classified if o >= 3]

        # ═══ TRN: ATTENTIONAL SPOTLIGHT ═══
        if content:
            spoke_scores: Dict[str, float] = {}
            for spoke_name in self.trn.channels:
                layer = self.cortex.get_layer(spoke_name)
                if not layer:
                    continue
                score = 0.0
                for w in content:
                    neighbors = layer.get_neighbors(w)
                    if neighbors:
                        score += sum(neighbors.values())
                if score > 0:
                    spoke_scores[spoke_name] = score

            if spoke_scores:
                best_spoke = max(spoke_scores, key=spoke_scores.get)
                focus_strength = min(spoke_scores[best_spoke] / 5.0, 0.8)
                self.trn.cortical_command(best_spoke, focus_strength)
            else:
                self.trn.broad_attention()
        else:
            self.trn.broad_attention()

        # TRN gate cycle
        gate_result = self.trn.gate_cycle()

        # RE-SPECIALIZE with TRN gating:
        # Score each word based on co-graph connectivity and TRN gating
        scored_words: List[Tuple[str, float]] = []
        for w, origin in classified:
            # Base confidence from co-graph connectivity
            co = self._co_graph.get(w, {})
            confidence = min(len(co) / 50.0, 1.0) * 0.2 + 0.8 if co else 0.5

            # TRN gating: boost/suppress based on spoke connections
            for spoke_name, gate in gate_result.items():
                layer = self.cortex.get_layer(spoke_name)
                if not layer:
                    continue
                if gate['passed'] and layer.get_neighbors(w):
                    confidence *= (1.0 + gate['strength'] * 0.5)
                elif gate.get('suppression', 0) > 0 and layer.get_neighbors(w):
                    confidence *= max(0.1, 1.0 - gate['suppression'] * 0.3)

            scored_words.append((w, confidence))

        # Sort by confidence, extract content words
        scored_words.sort(key=lambda x: -x[1])
        content = [w for w, _ in scored_words if self._word_pathway(w) >= 3]
        if not content:
            content = [w for w, _ in scored_words if len(w) > 1][:10]
        if not content:
            return {'focus': [], 'retrieved': [], 'coherence': 0.0,
                    'steps': 0, 'converged': False, 'trace': [],
                    'situation': 'general', 'goal': 'general',
                    'trn': self.trn.spotlight()}

        result = self.thinker.deliberate(
            query_tokens=content,
            activate_fn=self.activate,
            score_fn=self.score,
            signal_fn=lambda s, q: self.signal.observe(s, q),
            cross_layer_fn=self.cortex.cross_layer_activation,
            confidence_fn=self.cortex.confidence,
            imagination=self.imagination,
            co_graph=self._co_graph,
            episodic_context=self.memory.get_context(),
            memory_recall_fn=self.memory.recall,
            attention_fn=self.attention.attend,
        )

        # Include TRN state in result
        result['trn'] = self.trn.spotlight()

        # Consolidation
        for word in result.get('consolidated', []):
            if word and result['focus']:
                primary = result['focus'][0]
                general = self.cortex.ensure_layer('_general')
                general.connect(primary, word, 0.5, self.cortex._epoch)
                general.connect(word, primary, 0.3, self.cortex._epoch)

        return result

    # ═══ CONSOLIDATE — the discipline phase ═══
    #
    # After wild feeding (textbooks, PDFs), the mind has a massive
    # _general layer and co-graph but no typed spokes. Consolidation
    # is the MATURATION step: go through the strongest connections
    # and route them into identity/appearance/function/mechanism/relation
    # based on the copula patterns and sentence structure learned during feeding.
    #
    # Like thymic selection: wild proliferation → selection pressure →
    # only the connections that fit the architecture survive.

    def consolidate(self, focus_size: Optional[int] = None) -> dict:
        """
        FOCUSED CONSOLIDATION — one chapter at a time.

        Not 57K words at once. The first pass stays deliberately small so the
        baby can organize itself quickly on first contact, then later passes
        widen the aperture.

        Default focus size:
          - first consolidation: 200 words
          - later consolidations: 500 words

        Focus selected by: conviction target, recent queries, then
        highest-frequency content words.

        Phase 1: IDENTITY — extract from co-graph structure.
          "X is a Y" detected from nx_graph bigrams.
          Any word that frequently precedes "is" gets identity links
          to the words that frequently follow "is".

        Phase 2: MECHANISM — words co-occurring with "causes/leads/results"
          get routed to mechanism layer.

        Phase 3: FUNCTION — words co-occurring with "treatment/therapy/used"
          get routed to function layer.

        Phase 4: APPEARANCE — words co-occurring with "presents/shows/appears"
          get routed to appearance layer.

        Phase 5: RELATION — words co-occurring with "associated/related/risk"
          get routed to relation layer.

        No hardcoded word lists. The routing uses the CO-GRAPH ITSELF:
        which words actually co-occur with structural verbs in the data.

        Phase 6: Prune, myelinate, rebuild caches.
        """
        t0 = time.time()
        gen = self.cortex.ensure_layer('_general')
        routed = 0
        identities = 0

        # ═══ INCREMENTAL CONSOLIDATION ═══
        # First time: process everything. Subsequent: only NEW words.
        # Like organizing a room: first time is slow,
        # second time only the new stuff needs a place.
        last_epoch = getattr(self, '_last_consolidation_epoch', 0)
        current_epoch = self.cortex._epoch
        if focus_size is None:
            focus_size = 200 if last_epoch == 0 else 500

        # Which words are NEW since last consolidation?
        new_words = set()
        for word, birth in self.cortex._connection_birth.items():
            if birth > last_epoch:
                new_words.add(word)

        # If first consolidation or >50% new, process everything
        process_all = last_epoch == 0 or len(new_words) > len(self.cortex.word_freq) * 0.5

        # ═══ FOCUSED ATTENTION: what does the baby care about? ═══
        # Not top-by-frequency. Top-by-RELEVANCE-TO-CURRENT-FOCUS.
        # 1. Conviction target words (what it's reaching toward)
        # 2. Recent attention words (what was just activated)
        # 3. Then fill with frequency-ranked content words
        # Only focus_size words get processed. One chapter at a time.

        focus_words = set()

        # Priority 1: Conviction — what the baby is reaching toward
        purpose = self.conviction.discover_purpose(self)
        if purpose:
            goal_info = self.conviction._goals.get(purpose, {})
            core = goal_info.get('core', '')
            frontier = goal_info.get('frontier', '')
            if core:
                focus_words.add(core)
                # Add core's co-graph neighborhood
                for n in list(self._co_graph.get(core, {}).keys())[:50]:
                    if len(n) > 3:
                        focus_words.add(n)
            if frontier and len(frontier) > 3:
                focus_words.add(frontier)
                for n in list(self._co_graph.get(frontier, {}).keys())[:50]:
                    if len(n) > 3:
                        focus_words.add(n)

        # Priority 2: Recent attention (what was just queried)
        for ep in self.memory.episodes[-5:]:
            for t in ep.tokens:
                if len(t) > 3:
                    focus_words.add(t)

        # Priority 3: Fill with frequency-ranked content words
        freq = self.cortex.word_freq
        if len(focus_words) < focus_size:
            remaining = focus_size - len(focus_words)
            ranked = sorted(
                [(w, freq.get(w, 0)) for w in self._co_graph
                 if len(w) > 3 and w not in focus_words],
                key=lambda x: -x[1],
            )[:remaining]
            for w, _ in ranked:
                focus_words.add(w)

        if process_all:
            words_to_process = list(focus_words)[:focus_size]
        else:
            # Incremental: only new words, but prioritize focused ones
            focused_new = [w for w in new_words if w in focus_words and len(w) > 2]
            other_new = [w for w in new_words if w not in focus_words and len(w) > 2]
            words_to_process = focused_new + other_new[:100]  # New non-focused get light treatment

        # ═══ PHASE 0: BUILD CORTEX from co-graph ═══
        cortex_built = 0
        for word in words_to_process:
            neighbors = self._co_graph.get(word, {})
            top = sorted(neighbors.items(), key=lambda x: -x[1])[:20]
            for neighbor, weight in top:
                if len(neighbor) <= 2 or weight < 2:
                    continue
                gen.connect(word, neighbor, min(weight, 10.0), current_epoch)
                cortex_built += 1

        # ═══ PHASE 1: IDENTITY (only for new words) ═══
        id_layer = self.cortex.ensure_layer('identity')
        is_preceders = {}
        for word, nexts in self._nx_graph.items():
            if 'is' in nexts and len(word) > 3:
                if process_all or word in new_words:
                    is_preceders[word] = nexts['is']

        is_followers = self._nx_graph.get('is', {})

        for subject, subj_freq in sorted(is_preceders.items(), key=lambda x: -x[1])[:200]:
            for category, cat_freq in sorted(is_followers.items(), key=lambda x: -x[1])[:20]:
                if category == subject or len(category) <= 2:
                    continue
                if category in self._co_graph.get(subject, {}):
                    id_layer.connect(subject, category, min(subj_freq, cat_freq), current_epoch)
                    # Tag the edge
                    self.add_edge_tag(subject, category, 'id')
                    self.add_edge_tag(category, subject, 'id')
                    identities += 1

        # ═══ PHASES 2-5: Route by CO-OCCURRENCE (only new words) ═══
        # Instead of hardcoded lists, find structural words from nx_graph:
        # words that appear as next-word for MANY different preceding words
        # are likely structural (verbs/prepositions). The TARGETS of those
        # structural words get routed to the appropriate layer.

        # Mechanism: words that co-occur with the top "causes/produces" equivalents
        # (words that many different words precede → structural verbs)
        # Spoke routing: signal words → relationship tag
        # DUAL-WRITE: writes to both layer (backward compat) AND _co_tags
        spoke_routing = {
            'mechanism': (['causes', 'caused', 'leads', 'results', 'produces', 'involves', 'through', 'due'], 'mech'),
            'function': (['treatment', 'therapy', 'treats', 'used', 'prevents', 'reduces', 'management'], 'fn'),
            'appearance': (['presents', 'shows', 'appears', 'seen', 'reveals', 'visible', 'characterized'], 'app'),
            'relation': (['associated', 'related', 'risk', 'factor', 'linked', 'increases', 'compared'], 'rel'),
        }

        focus_set = set(words_to_process)
        for layer_name, (signal_words, tag) in spoke_routing.items():
            typed = self.cortex.ensure_layer(layer_name)
            for signal in signal_words:
                co_neighbors = self._co_graph.get(signal, {})
                for neighbor, weight in sorted(co_neighbors.items(), key=lambda x: -x[1])[:10]:
                    if len(neighbor) > 3 and weight > 1 and neighbor in focus_set:
                        typed.connect(neighbor, signal, weight, current_epoch)
                        # Tag the edge
                        self.add_edge_tag(neighbor, signal, tag)
                        self.add_edge_tag(signal, neighbor, tag)
                        routed += 1

        # ═══ PHASE 6: Prune, myelinate, rebuild ═══
        pruned = 0
        for layer in self.cortex._layers.values():
            pruned += layer.prune(
                self.cortex._epoch, self.cortex._decay_factor,
                self.cortex._myelinated_decay, 0.3,
            )
            layer.myelinate(self.cortex._myelination_threshold, self.cortex._epoch)

        self.cortex._invalidate_cache()
        self.field.invalidate_cache()
        self.field.update_medians(self.cortex.word_freq, self._co_graph)

        # Phase 6b: DOMAIN DISCOVERY — only for focused words
        observed = 0
        scan_words = focus_set if process_all else (new_words & focus_set)
        for word in scan_words:
            if len(word) <= 3:
                continue
            co = self._co_graph.get(word, {})
            top_neighbors = [n for n, _ in sorted(co.items(), key=lambda x: -x[1])[:5] if len(n) > 3]
            if top_neighbors:
                self.trunk.observe([word] + top_neighbors, self._co_graph)
                observed += 1
            if observed > 2000:
                break
        self.trunk.finalize()

        # Mark consolidation epoch — next time only process what's new
        self._last_consolidation_epoch = current_epoch

        # Phase 7: BUILD NEURAL FIELD — each word becomes a neuron
        # Feed only top 5000 words by frequency to neural field.
        # The neurons that matter most get built first.
        myel_pairs = set()
        for ly in self.cortex._layers.values():
            for src, targets in ly._connections.items():
                for tgt, syn in targets.items():
                    if syn.myelinated:
                        myel_pairs.add((src, tgt))
        cortex_conns = {}
        gen_layer = self.cortex.get_layer('_general')
        if gen_layer:
            for src, targets in gen_layer._connections.items():
                cortex_conns[src] = {t: s.weight for t, s in targets.items()}
        # Frequency-ranked co-graph subset
        neural_ranked = sorted(
            [(w, freq.get(w, 0)) for w in self._co_graph if len(w) > 2],
            key=lambda x: -x[1],
        )[:5000]
        neural_co = {w: self._co_graph[w] for w, _ in neural_ranked if w in self._co_graph}
        neural_built = self.neural_field.build_from_graphs(
            neural_co, cortex_conns, myel_pairs,
        )

        # Phase 8: Language acquisition — analyze morphology, syntax, semantics
        morph = self.language.morphology.feed_vocabulary(self.cortex.word_freq)
        self.language.morphology.feed_bigrams(self._nx_graph)
        syn = 0
        if self.cortex.word_freq:
            syn_result = self.language.semantics.feed(
                self._co_graph, self._nx_graph, self.cortex.word_freq,
            )
            syn = sum(syn_result.values())

        elapsed = round(time.time() - t0, 1)
        return {
            'routed': routed, 'identities': identities, 'pruned': pruned,
            'morphology': morph, 'semantics': syn,
            'neural_field': neural_built,
            'incremental': not process_all,
            'new_words_processed': len(new_words) if not process_all else len(words_to_process),
            'elapsed_seconds': elapsed,
        }

    # ═══════════════════════════════════════════════════════════
    #  LANGUAGE GENERATION
    # ═══════════════════════════════════════════════════════════

    def describe(self, word: str) -> str:
        """
        Describe a word using relationship-tagged edges.
        Primary source: _co_tags.
        Fallback: raw co-graph.
        """
        w = word.lower()
        conf = self.cortex.confidence(w)

        if conf['state'] == 'unknown':
            return f"I don't have experience with {word} yet."

        tag_labels = {
            'id': 'is',
            'app': 'appears',
            'fn': 'used for',
            'mech': 'works through',
            'rel': 'related to',
        }

        parts = []
        # Primary: tag-based edges
        for tag, verb in tag_labels.items():
            edges = self.edges_by_tag(w, tag)
            if edges:
                top = sorted(edges.items(), key=lambda x: -x[1])
                words = [k for k, _ in top if k != w and len(k) > 2][:3]
                if words:
                    parts.append(f"{verb} {', '.join(words)}")

        # Fallback: raw co-graph
        if not parts:
            co = self._co_graph.get(w, {})
            top = sorted(co.items(), key=lambda x: -x[1])
            words = [k for k, _ in top if k != w and len(k) > 2][:5]
            if words:
                parts.append(f"connects to {', '.join(words)}")

        # Add unmentioned co-graph neighbors
        mentioned = set()
        for p in parts:
            mentioned.update(p.split())
        co_neighbors = self._co_graph.get(w, {})
        co_top = [k for k, v in sorted(co_neighbors.items(), key=lambda x: -x[1])
                  if k != w and k not in mentioned and len(k) > 2][:3]
        if co_top:
            parts.append(f"associated with {', '.join(co_top)}")

        if not parts:
            return f"I know {word} but can't describe it well yet."

        state = conf['state']
        score = conf['score']
        result = f"{word} {'. '.join(parts)}."
        result = result[0].upper() + result[1:]
        return f"{result} [{score}% {state}]"

    def generate(self, seed_words: List[str], max_length: int = 20,
                 goal: Optional[str] = None) -> List[str]:
        """Generate a token sequence from seeds — through the web.
        Goal biases Broca's trajectory (identity, mechanism, function, etc.)."""
        return self.speaker.generate(
            seed_words, self._co_graph, self._nx_graph, max_length,
            neural_field=self.neural_field if self.neural_field.neurons else None,
            goal=goal,
            edges_by_goal_fn=self.edges_by_goal if goal else None,
        )

    # ═══════════════════════════════════════════════════════════
    #  CONFIDENCE & INTROSPECTION
    # ═══════════════════════════════════════════════════════════

    def confidence(self, word: str) -> dict:
        """How well does the mind know this word?"""
        return self.cortex.confidence(word.lower())

    def hungry(self) -> List[dict]:
        """What concepts have gaps in their knowledge?
        Only SPECIFIC words — high-breadth generic words don't need spokes.
        The landscape decides what's specific: low breadth = specific."""
        gaps = []
        V = max(len(self.cortex.word_freq), 1)
        for word, freq in sorted(
            self.cortex.word_freq.items(), key=lambda x: -x[1]
        )[:200]:
            if len(word) <= 3:
                continue
            # Skip high-breadth words — they're structural, not content
            breadth = len(self.cortex.breadth.get(word, set()))
            if breadth > V * 0.15:
                continue  # Connected to too much — it's a connector, not a concept
            conf = self.cortex.confidence(word)
            if conf['score'] < 10:
                continue
            layers = self.cortex.cross_layer_activation(word)
            present = [n for n, conns in layers.items() if conns]
            all_layers = [n for n in self.cortex.layer_names if n != '_general']
            missing = [n for n in all_layers if n not in present]
            if present and missing:
                gaps.append({
                    'word': word,
                    'have': present,
                    'need': missing,
                    'hunger': len(missing) / max(len(all_layers), 1),
                })
        gaps.sort(key=lambda x: -x['hunger'])
        return gaps[:10]

    # ═══════════════════════════════════════════════════════════
    #  STATS
    # ═══════════════════════════════════════════════════════════

    def stats(self) -> dict:
        cx = self.cortex.stats()
        tr = self.trunk.stats()
        return {
            'vocabulary': cx['vocabulary'],
            'total_words': cx['total_words'],
            'assemblies': cx['assemblies'],
            'myelinated': cx['myelinated'] + self.neural_field.stats().get('myelinated', 0),
            'plasticity': cx['plasticity'],
            'domains': tr['domains'],
            'trunk_words': tr['trunk_words'],
            'feed_count': max(self._feed_count, self.cortex._feed_count),
            'epoch': max(self._epoch, self.cortex._epoch),
            'neural_neurons': self.neural_field.stats()['neurons'] if self.neural_field.neurons else 0,
            'co_graph_size': sum(len(v) for v in self._co_graph.values()),
            'nx_graph_size': sum(len(v) for v in self._nx_graph.values()),
            'gate': self.gate.stats(),
            'signal_trend': round(self.signal.recent_trend(), 3),
            'neuroglia': self.neuroglia.stats(),
            'trn': self.trn.stats(),
        }

    # ═══════════════════════════════════════════════════════════
    #  SERIALIZATION
    # ═══════════════════════════════════════════════════════════

    def to_dict(self) -> dict:
        return {
            'cortex': self.cortex.to_dict(),
            'field': self.field.to_dict(),
            'gate': self.gate.to_dict(),
            'signal': self.signal.to_dict(),
            'trunk': self.trunk.to_dict(),
            'memory': self.memory.to_dict(),
            'speaker': self.speaker.to_dict(),
            'thinker': self.thinker.to_dict(),
            'imagination': self.imagination.to_dict(),
            'attention': self.attention.to_dict(),
            'conviction': self.conviction.to_dict(),
            'co_graph': self._co_graph,
            'co_tags': {src: {tgt: sorted(tags) for tgt, tags in targets.items()}
                        for src, targets in self._co_tags.items()},
            'nx_graph': self._nx_graph,
            'feed_count': self._feed_count,
            'epoch': self._epoch,
        }

    @classmethod
    def from_dict(cls, d: dict) -> ShifuMind:
        mind = cls.__new__(cls)
        mind.cortex = Cortex.from_dict(d.get('cortex', {}))
        mind.field = Field.from_dict(d.get('field', {}))
        mind.gate = Gate.from_dict(d.get('gate', {}))
        mind.signal = Signal.from_dict(d.get('signal', {}))
        mind.trunk = Trunk.from_dict(d.get('trunk', {}))
        mind.memory = Memory.from_dict(d.get('memory', {}))
        mind.speaker = Speaker.from_dict(d.get('speaker', {}))
        mind.thinker = Thinker.from_dict(d.get('thinker', {}))
        mind.imagination = Imagination.from_dict(d.get('imagination', {}))
        mind.attention = Attention.from_dict(d.get('attention', {}))
        mind.conviction = Conviction.from_dict(d.get('conviction', {}))
        mind._co_graph = d.get('co_graph', {})
        # Deserialize tags: lists → sets (default to empty if missing)
        raw_tags = d.get('co_tags', {})
        mind._co_tags = {src: {tgt: set(tags) for tgt, tags in targets.items()}
                         for src, targets in raw_tags.items()} if raw_tags else {}
        mind._nx_graph = d.get('nx_graph', {})
        mind._feed_count = d.get('feed_count', 0)
        mind._epoch = d.get('epoch', 0)
        return mind

    def study(self, rounds: int = 5, level: Optional[int] = None) -> dict:
        """
        Language curriculum: structured practice at adaptive level.
        Level 1: word decomposition. Level 2: phrases. Level 3: sentences.
        Level 4: paragraphs. Level 5: reasoning.
        Auto-levels up/down based on performance.
        """
        return self.language.curriculum.practice(self, rounds=rounds, level=level)

    def assess_language(self) -> dict:
        """Assess current language ability across all levels."""
        return self.language.curriculum.assess(self)

    # ═══ LEARNING LAB — endless self-practice ═══
    #
    # The baby practices by:
    # 1. Pick a concept it's learning (not yet "know")
    # 2. GENERATE a sentence about it from its connections
    # 3. SCORE the sentence for coherence
    # 4. If coherence is high → reinforce those connections (reward)
    # 5. If coherence is low → weaken those connections (punish)
    # 6. Record the dopamine signal (surprise = learning)
    # 7. Repeat
    #
    # Like a child talking to itself, testing what it knows,
    # and learning from whether its own sentences make sense.

    def practice(self, rounds: int = 10) -> dict:
        """
        Self-practice loop. Pick concepts, generate, score, reinforce/punish.
        Returns summary of what was practiced and what improved.
        """
        results = []
        improved = 0
        degraded = 0

        # FOCUSED PRACTICE: pick ONE concept and repeat ALL rounds on it.
        # Like studying one chapter until you know it.
        # Repetition strengthens connections until they myelinate.
        # Conviction target gets priority. Otherwise least-connected.
        candidates = []
        for word, freq in self.cortex.word_freq.items():
            if len(word) <= 4 or freq < 2:
                continue
            breadth = len(self.cortex.breadth.get(word, set()))
            if breadth < 30:
                candidates.append((word, breadth))
        candidates.sort(key=lambda x: x[1])

        # Pick ONE focus word — ROTATE through candidates.
        # Don't re-study the same chapter forever.
        # Track which words have been practiced. Move to the next.
        if not hasattr(self, '_practiced_words'):
            self._practiced_words = set()

        focus_word = None
        # Try conviction target first — but only if not already practiced
        purpose = self.conviction.discover_purpose(self)
        if purpose:
            goal_info = self.conviction._goals.get(purpose, {})
            fw = goal_info.get('frontier') or goal_info.get('core')
            if fw and len(fw) > 4 and fw not in self._practiced_words:
                focus_word = fw

        # Otherwise pick the next unpracticed candidate
        if not focus_word:
            for word, _ in candidates:
                if word not in self._practiced_words:
                    focus_word = word
                    break

        # If all practiced, reset and start over (deeper learning)
        if not focus_word:
            self._practiced_words.clear()
            if candidates:
                focus_word = candidates[0][0]

        # Mark as practiced after this session
        if focus_word:
            self._practiced_words.add(focus_word)
        if not focus_word:
            return {'rounds': 0, 'improved': 0, 'degraded': 0, 'practice': []}

        # ALL rounds on the SAME word — repetition builds strength
        for i in range(rounds):
            word = focus_word
            before_conf = len(self.cortex.breadth.get(word, set()))

            # Generate a sentence from this concept
            generated = self.generate([word], max_length=10)
            # If generation is too short, use co-graph neighbors directly
            if len(generated) < 3:
                co = self._co_graph.get(word, {})
                neighbors = sorted(co.items(), key=lambda x: -x[1])[:5]
                generated = [word] + [n for n, _ in neighbors]
            sentence = ' '.join(generated)

            # Score it
            score_result = self.score(generated)
            coherence = score_result.get('coherence', 0.0)

            # Dopamine signal: was this better or worse than expected?
            state_key = f"practice:{word}"
            signal = self.signal.observe(state_key, coherence)
            surprise = signal['error']

            # ═══ IN A DARK PLACE EVERY STEP IS THE RIGHT STEP ═══
            #
            # No punishment. Ever. A baby doesn't weaken its legs
            # when it falls. It gets up. The fall IS the learning.
            #
            # High coherence → strong reinforcement (confident step)
            # Low coherence → gentle reinforcement (exploring step)
            # ZERO coherence → still reinforcement (brave step into the dark)
            #
            # The only difference is HOW MUCH, not WHETHER.
            # Every connection the mind makes is a step. Every step
            # is the right step because it's a step FORWARD.
            # Reinforce in CO-GRAPH (where heartbeat reads for myelination)
            # NOT just cortex. The co-graph is the shared knowledge.
            if len(generated) >= 2:
                step_weight = max(1.0, coherence * 3.0)
                for j in range(len(generated) - 1):
                    a, b = generated[j], generated[j + 1]
                    if len(a) > 4 and len(b) > 4:
                        if a not in self._co_graph:
                            self._co_graph[a] = {}
                        if b in self._co_graph[a]:
                            self._co_graph[a][b] += step_weight
                        elif len(self._co_graph[a]) < 100:
                            self._co_graph[a][b] = step_weight
                        improved += 1

            # CREATIVE LEAP: connect focus word to its GENERATED neighbors
            # Not random. FOCUSED. Every leap builds toward the conviction goal.
            # Repetition on the SAME connections = accumulation = myelination.
            if len(generated) >= 2:
                for j in range(len(generated) - 1):
                    a, b = generated[j], generated[j + 1]
                    if len(a) > 4 and len(b) > 4:
                        # Bidirectional — both directions accumulate
                        for x, y in [(a, b), (b, a)]:
                            if x not in self._co_graph:
                                self._co_graph[x] = {}
                            self._co_graph[x][y] = self._co_graph[x].get(y, 0) + 1.0

            after_conf = self.cortex.confidence(word)['score']
            results.append({
                'word': word,
                'sentence': sentence,
                'coherence': round(coherence, 3),
                'surprise': round(surprise, 3),
                'before': before_conf,
                'after': after_conf['score'] if isinstance(after_conf, dict) else after_conf,
            })

        # ═══ CONVICTION — when dopamine says stop, conviction pushes through ═══
        trend = self.signal.recent_trend(5)
        conviction_result = None
        # Conviction fires when the mind is COMFORTABLE — not just when failing.
        # Comfort (stable high trend) means the mind stopped growing.
        # A mind that always gets 0.7 has plateaued. Conviction breaks the plateau.
        stable = abs(trend - self.signal.recent_trend(10)) < 0.05  # Flat line
        if stable and len(self.cortex.word_freq) > 20:
            # Dopamine is flat. Normal mind would rest.
            # But conviction discovers purpose and pushes through.
            purpose = self.conviction.discover_purpose(self)
            if purpose:
                conviction_result = self.conviction.push_through(purpose, trend, self)

        self.cortex._invalidate_cache()
        self.field.invalidate_cache()

        result = {
            'rounds': len(results),
            'improved': improved,
            'degraded': degraded,
            'practice': results,
        }
        if conviction_result:
            result['conviction'] = conviction_result
            result['voice'] = self.conviction._voice[-1] if self.conviction._voice else None
        return result

    # ═══ METABOLISM — the brain's heartbeat ═══

    def heartbeat(self) -> dict:
        """
        Blood supply: promote co-graph edges to cortex AND myelinate.
        No separate consolidation needed for myelination.
        Only CONTENT words (PATH 3+, len > 4).
        """
        myelinated_new = 0
        shortcuts_created = 0
        gen = self.cortex.ensure_layer('_general')

        # Phase 1: AUC MYELINATION
        # Y = frequency of exposure. X = time. Area under curve = myelination.
        # Every heartbeat adds current co-graph weight to cumulative AUC.
        # AUC >= 3.0 → myelinate. No hard threshold on instantaneous weight.
        # 100 weak exposures (0.1 each) myelinate just like 3 strong ones.
        if not hasattr(self, '_auc'):
            self._auc = {}
        if not hasattr(self, '_hb_cursor'):
            self._hb_cursor = 0  # Rotating cursor through co-graph

        # Scan a WINDOW of 200 words per heartbeat — not the entire co-graph.
        # Rotate through the co-graph over multiple beats.
        # Like a heartbeat pumping blood to one organ at a time.
        co_words = list(self._co_graph.keys())
        # Window scales with vocabulary — bigger brain, bigger window
        # Baby (< 100 words): 50. Adult (> 5000): 500.
        window_size = min(max(len(co_words) // 10, 50), 500)
        start = self._hb_cursor % max(len(co_words), 1)
        end = min(start + window_size, len(co_words))
        self._hb_cursor = end  # Next beat starts where this one stopped

        promoted = 0
        for idx in range(start, end):
            source = co_words[idx]
            if len(source) <= 4:
                continue
            neighbors = self._co_graph[source]
            for target, weight in sorted(neighbors.items(), key=lambda x: -x[1])[:10]:
                if len(target) <= 4:
                    continue
                key = (source, target)
                self._auc[key] = self._auc.get(key, 0) + weight * 0.5
                syn = gen.connect(source, target, min(weight, 10.0), self.cortex._epoch)
                # AUC threshold scales: high-weight connections myelinate faster
                # weight=10 → threshold=1.5. weight=2 → threshold=3.0.
                auc_threshold = max(1.5, 3.0 - weight * 0.15)
                if not syn.myelinated and self._auc[key] >= auc_threshold:
                    syn.myelinate()
                    myelinated_new += 1
                    self.cortex._myel_count += 1
                promoted += 1

        # Phase 2: Saltatory shortcuts
        if myelinated_new > 0:
            new_shortcuts = []
            for source, targets in gen._connections.items():
                if len(source) <= 4:
                    continue
                for target, syn in targets.items():
                    if not syn.myelinated or len(target) <= 4:
                        continue
                    ext = gen._connections.get(target, {})
                    for et, es in ext.items():
                        if es.myelinated and et != source and len(et) > 4:
                            if not gen.get_synapse(source, et):
                                new_shortcuts.append((source, et, min(syn.weight, es.weight) * 0.7))
                                shortcuts_created += 1
                                if shortcuts_created > 100:
                                    break
                    if shortcuts_created > 100:
                        break
                if shortcuts_created > 100:
                    break
            for src, tgt, wt in new_shortcuts:
                s = gen.connect(src, tgt, wt, self.cortex._epoch)
                s.myelinate()
                self.cortex._myel_count += 1

        # ═══ NEUROGLIA: OPC differentiation ═══
        # OPCs near active axons differentiate and myelinate
        opc_diff = self.neuroglia.opc.differentiate(self.neural_field)
        myelinated_new += opc_diff.get('myelinated', 0)

        # Report thermal load
        self.neuroglia.observe_region('_general', myelinated_new + shortcuts_created)

        self.cortex._invalidate_cache()
        self.field.invalidate_cache()
        return {'myelinated_new': myelinated_new, 'shortcuts': shortcuts_created}

    def save(self, path: str) -> None:
        """Save full state to JSON file."""
        import json
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f)

    @classmethod
    def load(cls, path: str) -> ShifuMind:
        """Load from JSON file."""
        import json
        with open(path, 'r') as f:
            return cls.from_dict(json.load(f))
