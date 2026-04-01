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
from typing import Dict, List, Set, Optional, Tuple, Any

from .cortex import Cortex, _tokenize
from .field import Field
from .gate import Gate
from .signal import Signal
from .trunk import Trunk
from .memory import Memory
from .speaker import Speaker
from .thinker import Thinker


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

        # ═══ SHARED GRAPH STATE ═══
        # These are the graphs that field.py operates on.
        # Built incrementally by feed().
        self._co_graph: Dict[str, Dict[str, float]] = {}   # co-occurrence
        self._nx_graph: Dict[str, Dict[str, float]] = {}   # next-word
        self._px_graph: Dict[str, Dict[str, float]] = {}   # prev-word
        self._res_graph: Dict[str, Dict[str, float]] = {}  # resonance
        self._snx_graph: Dict[str, Dict[str, float]] = {}  # skip-gram

        # ═══ TRACKING ═══
        self._feed_count = 0
        self._epoch = 0

    # ═══════════════════════════════════════════════════════════
    #  FEEDING — absorb information
    # ═══════════════════════════════════════════════════════════

    def feed(self, text: str, layer: str = '_general',
             classifier=None) -> dict:
        """
        Main entry point for learning.

        Sequence:
        1. Gate filters input
        2. Cortex absorbs connections
        3. Shared graphs update (co, nx, px, res, snx)
        4. Speaker learns bigram frames
        5. Trunk observes for domain emergence
        6. Memory records if topic shifted
        7. Field updates inhibition medians

        Returns: {
            'accepted': bool,
            'tokens_absorbed': int,
            'domain': str,
            'quality': float,
        }
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

        # 2. Cortex
        connections = self.cortex.feed(
            tokens, layer=layer, classifier=classifier,
        )

        # 3. Shared graphs
        self._update_shared_graphs(tokens)

        # 4. Speaker
        self.speaker.learn_frame(tokens)

        # 5. Trunk
        domain = self.trunk.observe(content, self._co_graph)

        # 6. Memory — record if topic shift detected
        shift = self.memory.detect_topic_shift(content)
        significance = shift * filtered['quality']
        self.memory.record(
            epoch=self._epoch, tokens=content,
            significance=significance,
            context={'domain': domain, 'quality': filtered['quality']},
            timestamp=time.time(),
        )

        # 7. Field medians
        if self._feed_count % 50 == 0:
            self.field.update_medians(self.cortex.word_freq, self._co_graph)
            self.gate.adapt_thresholds()

        self._feed_count += 1

        return {
            'accepted': True,
            'tokens_absorbed': connections,
            'domain': domain,
            'quality': filtered['quality'],
        }

    def feed_batch(self, texts: List[str], layer: str = '_general',
                   classifier=None) -> dict:
        """Feed multiple texts. Returns aggregate stats."""
        accepted = 0
        total_tokens = 0
        for text in texts:
            result = self.feed(text, layer=layer, classifier=classifier)
            if result['accepted']:
                accepted += 1
                total_tokens += result['tokens_absorbed']

        # Finalize trunk after batch
        if accepted > 10:
            self.trunk.finalize()

        return {
            'total': len(texts),
            'accepted': accepted,
            'tokens_absorbed': total_tokens,
        }

    def _update_shared_graphs(self, tokens: List[str]) -> None:
        """Update co-occurrence, next-word, prev-word, skip-gram, resonance."""
        n = len(tokens)
        if n < 2:
            return

        for i in range(n):
            w = tokens[i]

            # Co-occurrence (window=5)
            for j in range(max(0, i - 5), min(n, i + 6)):
                if i == j:
                    continue
                nb = tokens[j]
                if w not in self._co_graph:
                    self._co_graph[w] = {}
                self._co_graph[w][nb] = self._co_graph[w].get(nb, 0) + 1

            # Next-word
            if i < n - 1:
                nxt = tokens[i + 1]
                if w not in self._nx_graph:
                    self._nx_graph[w] = {}
                self._nx_graph[w][nxt] = self._nx_graph[w].get(nxt, 0) + 1

            # Prev-word
            if i > 0:
                prv = tokens[i - 1]
                if w not in self._px_graph:
                    self._px_graph[w] = {}
                self._px_graph[w][prv] = self._px_graph[w].get(prv, 0) + 1

            # Skip-gram (window=7, min_dist=2)
            for j in range(max(0, i - 7), min(n, i + 8)):
                dist = abs(i - j)
                if dist < 2:
                    continue
                skip = tokens[j]
                if w not in self._snx_graph:
                    self._snx_graph[w] = {}
                self._snx_graph[w][skip] = self._snx_graph[w].get(skip, 0) + 1

        # Resonance: words that share many co-occurrence neighbors
        # Only update periodically (expensive)
        if self._feed_count % 20 == 0:
            self._update_resonance(tokens)

    def _update_resonance(self, tokens: List[str]) -> None:
        """Build resonance links between words sharing neighbors."""
        unique = list(set(tokens))
        for i in range(len(unique)):
            a = unique[i]
            a_neighbors = set(self._co_graph.get(a, {}).keys())
            if len(a_neighbors) < 3:
                continue
            for j in range(i + 1, len(unique)):
                b = unique[j]
                b_neighbors = set(self._co_graph.get(b, {}).keys())
                if len(b_neighbors) < 3:
                    continue
                shared = len(a_neighbors & b_neighbors)
                if shared >= 2:
                    amount = shared * 0.1
                    if a not in self._res_graph:
                        self._res_graph[a] = {}
                    if b not in self._res_graph:
                        self._res_graph[b] = {}
                    self._res_graph[a][b] = self._res_graph[a].get(b, 0) + amount
                    self._res_graph[b][a] = self._res_graph[b].get(a, 0) + amount

    # ═══════════════════════════════════════════════════════════
    #  QUERYING — activate and score
    # ═══════════════════════════════════════════════════════════

    def activate(self, word: str) -> Dict[str, float]:
        """Activate a word through the unified field."""
        return self.field.activate(
            word, self._co_graph, self._nx_graph,
            self._res_graph, self._snx_graph,
            self.cortex.word_freq,
        )

    def score(self, tokens: List[str]) -> dict:
        """Score a token sequence for coherence."""
        return self.field.score_sequence(
            tokens, self._co_graph, self._nx_graph,
            self._res_graph, self._snx_graph,
            self.cortex.word_freq,
        )

    def score_text(self, text: str) -> dict:
        """Score a text string for coherence."""
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
            self._res_graph, self._snx_graph,
            self.cortex.word_freq,
            ocr_weight=ocr_weight,
            field_weight=field_weight,
        )

    # ═══════════════════════════════════════════════════════════
    #  DELIBERATION — multi-step reasoning
    # ═══════════════════════════════════════════════════════════

    def deliberate(self, query: str) -> dict:
        """
        Multi-step reasoning about a query.
        Uses the thinker with bound activate/score/signal functions.
        """
        tokens = _tokenize(query)
        content = [t for t in tokens if len(t) > 2]
        if not content:
            return {'focus': [], 'retrieved': [], 'coherence': 0.0,
                    'steps': 0, 'converged': False, 'trace': []}

        return self.thinker.deliberate(
            query_tokens=content,
            activate_fn=self.activate,
            score_fn=self.score,
            signal_fn=lambda s, q: self.signal.observe(s, q),
            episodic_context=self.memory.get_context(),
        )

    def counterfactual(self, text: str, position: int,
                       alternatives: List[str]) -> List[dict]:
        """What-if reasoning: score alternatives at a position."""
        tokens = _tokenize(text)
        return self.thinker.counterfactual(
            tokens, position, alternatives, self.score,
        )

    # ═══════════════════════════════════════════════════════════
    #  LANGUAGE GENERATION
    # ═══════════════════════════════════════════════════════════

    def describe(self, word: str) -> str:
        """Generate a description of a word from its connections."""
        layer_conns = self.cortex.cross_layer_activation(word.lower())
        return self.speaker.describe(
            word.lower(), self._co_graph,
            self._nx_graph, layer_conns or None,
        )

    def generate(self, seed_words: List[str], max_length: int = 20) -> List[str]:
        """Generate a token sequence from seeds."""
        return self.speaker.generate(
            seed_words, self._co_graph, self._nx_graph, max_length,
        )

    # ═══════════════════════════════════════════════════════════
    #  DOMAIN ROUTING
    # ═══════════════════════════════════════════════════════════

    def route(self, text: str) -> dict:
        """Route text to domain(s)."""
        tokens = _tokenize(text)
        return self.trunk.route(tokens, self._co_graph)

    # ═══════════════════════════════════════════════════════════
    #  CONFIDENCE & INTROSPECTION
    # ═══════════════════════════════════════════════════════════

    def confidence(self, word: str) -> dict:
        """How well does the mind know this word?"""
        return self.cortex.confidence(word.lower())

    def hungry(self) -> List[dict]:
        """What concepts have gaps in their knowledge?"""
        gaps = []
        for word, freq in sorted(
            self.cortex.word_freq.items(), key=lambda x: -x[1]
        )[:200]:
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
    #  TEMPORAL
    # ═══════════════════════════════════════════════════════════

    def tick(self, used_words: Optional[List[str]] = None) -> None:
        """Temporal heartbeat. Nurture used connections, decay unused."""
        confidence = self.signal.recent_trend()
        self.cortex.tick(used_words, confidence)

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
            'myelinated': cx['myelinated'],
            'plasticity': cx['plasticity'],
            'domains': tr['domains'],
            'trunk_words': tr['trunk_words'],
            'feed_count': self._feed_count,
            'epoch': self._epoch,
            'co_graph_size': sum(len(v) for v in self._co_graph.values()),
            'nx_graph_size': sum(len(v) for v in self._nx_graph.values()),
            'res_graph_size': sum(len(v) for v in self._res_graph.values()),
            'gate': self.gate.stats(),
            'signal_trend': round(self.signal.recent_trend(), 3),
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
            'co_graph': self._co_graph,
            'nx_graph': self._nx_graph,
            'px_graph': self._px_graph,
            'res_graph': self._res_graph,
            'snx_graph': self._snx_graph,
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
        mind._co_graph = d.get('co_graph', {})
        mind._nx_graph = d.get('nx_graph', {})
        mind._px_graph = d.get('px_graph', {})
        mind._res_graph = d.get('res_graph', {})
        mind._snx_graph = d.get('snx_graph', {})
        mind._feed_count = d.get('feed_count', 0)
        mind._epoch = d.get('epoch', 0)
        return mind

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
