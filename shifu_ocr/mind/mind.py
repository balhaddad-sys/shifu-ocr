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

        # 2. Cortex — feed CONTENT words with identity-as-bridge routing
        #    Identity is NEVER the destination for property connections.
        #    Identity only holds "is a" declarations.
        #    All other connections route to appearance/function/mechanism/relation.
        connections = self.cortex.feed(
            content, layer=layer,
            classifier=classifier or self._classify_word_bridge,
        )

        # 2b. Extract identity declarations from the FULL sentence
        #     "Stroke is a disease" → identity(stroke, disease)
        self._extract_identity(text, content)

        # 3. Shared graphs — content words for co/snx/res, ALL tokens for nx/px
        #    (bigram transitions need function words for grammar)
        self._update_shared_graphs(content, tokens)

        # 4. Speaker — learns from ALL tokens (needs grammar structure)
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

    # ═══ IDENTITY IS THE BRIDGE ═══
    #
    # Identity is NEVER a destination for property connections.
    # Identity only holds "X is a Y" declarations — the name tag.
    #
    # When routing connections to layers:
    # 1. Check if nearby words signal a specific layer
    # 2. If the signal is "identity" — DON'T USE IT, keep searching
    # 3. Only the _extract_identity method writes to the identity layer
    #
    # This makes identity the HUB through which all spokes connect.
    # "Stroke" has an identity ("is a disease").
    # Its properties (appearance, function, mechanism, relation)
    # radiate FROM that identity. Without the bridge,
    # properties float disconnected.

    # Layer signal words — words that indicate which spoke is active.
    # These are NOT the content words. They are the GLUE that tells
    # the classifier where to route.
    _SPOKE_SIGNALS = {
        'appearance': {
            'appears', 'shows', 'presents', 'displays', 'reveals',
            'bright', 'dark', 'large', 'small', 'visible', 'seen',
            'acute', 'chronic', 'progressive', 'sudden', 'bilateral',
        },
        'function': {
            'used', 'treats', 'manages', 'prevents', 'reduces',
            'treatment', 'therapy', 'management', 'intervention',
            'for', 'purpose', 'restores', 'dissolves', 'inhibits',
        },
        'mechanism': {
            'causes', 'produces', 'leads', 'results', 'through',
            'mechanism', 'pathway', 'process', 'involves',
            'by', 'via', 'because', 'due', 'from', 'affects',
        },
        'relation': {
            'associated', 'related', 'linked', 'connected',
            'compared', 'versus', 'with', 'combined', 'risk',
            'factor', 'increases', 'decreases',
        },
    }

    # Identity signals — ONLY used by _extract_identity, not by the classifier
    _IDENTITY_PATTERNS = [
        r'(?:is a|is an|is the|are|defined as|refers to|known as)\s+(\w{3,})',
        r'\b(\w{3,})\s+(?:\w+\s+)?is\s+(\w{4,})',
    ]

    def _classify_word_bridge(self, word: str) -> Optional[str]:
        """
        Route a word to a semantic spoke — but NEVER to identity.
        Identity is the bridge, not a destination.
        """
        # Signal words themselves go to _general
        for signals in self._SPOKE_SIGNALS.values():
            if word in signals:
                return None

        # Check co-occurrence: which spoke signals does this word appear near?
        co_neighbors = self._co_graph.get(word, {})
        if not co_neighbors:
            return None

        spoke_scores: Dict[str, float] = {}
        for spoke, signals in self._SPOKE_SIGNALS.items():
            score = sum(co_neighbors.get(s, 0) for s in signals)
            if score > 0:
                spoke_scores[spoke] = score

        if not spoke_scores:
            return None

        best = max(spoke_scores, key=spoke_scores.get)
        if spoke_scores[best] >= 1.0:
            return best
        return None

    def _extract_identity(self, raw_text: str, content: List[str]) -> None:
        """
        Extract "X is a Y" patterns and write ONLY those to the identity layer.
        This is the ONLY way identity connections form.
        """
        if not content:
            return
        text_lower = raw_text.lower()
        identity_layer = self.cortex.ensure_layer('identity')
        subject = content[0]

        # Pattern 1: "X is a Y", "X are Y", "X defined as Y"
        import re
        match = re.search(
            r'(?:is a|is an|is the|are|defined as|refers to|known as)\s+(\w{3,})',
            text_lower,
        )
        if not match:
            # Pattern 2: "X ... is Y" (bare copula)
            match2 = re.search(r'\b(\w{3,})\s+(?:\w+\s+)?is\s+(\w{4,})', text_lower)
            if match2:
                a = match2.group(1)
                b = match2.group(2)
                if a == subject and b != subject and len(b) > 2:
                    identity_layer.connect(a, b, 2.0, self.cortex._epoch)
                    identity_layer.connect(b, a, 1.0, self.cortex._epoch)
                    return

        if match:
            b = match.group(1)
            if b != subject and len(b) > 2:
                # Subject IS b (strong: weight 2)
                identity_layer.connect(subject, b, 2.0, self.cortex._epoch)
                # b IS-INSTANCE subject (weaker: weight 1)
                identity_layer.connect(b, subject, 1.0, self.cortex._epoch)
                # Second content word also gets identity link
                if len(content) > 1 and content[1] != subject and content[1] != b:
                    identity_layer.connect(content[1], b, 1.0, self.cortex._epoch)

    def _update_shared_graphs(self, content: List[str],
                              all_tokens: Optional[List[str]] = None) -> None:
        """
        Update shared graphs.
        content: filtered content words → co-occurrence, skip-gram, resonance
        all_tokens: full token list → next-word, prev-word (needs grammar)
        """
        # Co-occurrence, skip-gram from CONTENT words only
        n = len(content)
        if n >= 2:
            for i in range(n):
                w = content[i]
                # Co-occurrence (window=5)
                for j in range(max(0, i - 5), min(n, i + 6)):
                    if i == j:
                        continue
                    nb = content[j]
                    if w not in self._co_graph:
                        self._co_graph[w] = {}
                    self._co_graph[w][nb] = self._co_graph[w].get(nb, 0) + 1
                # Skip-gram (window=7, min_dist=2)
                for j in range(max(0, i - 7), min(n, i + 8)):
                    dist = abs(i - j)
                    if dist < 2:
                        continue
                    skip = content[j]
                    if w not in self._snx_graph:
                        self._snx_graph[w] = {}
                    self._snx_graph[w][skip] = self._snx_graph[w].get(skip, 0) + 1

        # Next-word and prev-word from ALL tokens (grammar needs function words)
        seq = all_tokens if all_tokens is not None else content
        for i in range(len(seq)):
            w = seq[i]
            if i < len(seq) - 1:
                nxt = seq[i + 1]
                if w not in self._nx_graph:
                    self._nx_graph[w] = {}
                self._nx_graph[w][nxt] = self._nx_graph[w].get(nxt, 0) + 1
            if i > 0:
                prv = seq[i - 1]
                if w not in self._px_graph:
                    self._px_graph[w] = {}
                self._px_graph[w][prv] = self._px_graph[w].get(prv, 0) + 1

        # Resonance from content words (periodic)
        if self._feed_count % 20 == 0 and n >= 2:
            self._update_resonance(content)

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
        """
        Generate a rich 5-spoke description from connections.
        Each semantic layer contributes its perspective.
        """
        w = word.lower()
        layer_conns = self.cortex.cross_layer_activation(w)
        conf = self.cortex.confidence(w)

        if conf['state'] == 'unknown':
            return f"I don't have experience with {word} yet."

        labels = {
            'identity': 'is',
            'appearance': 'appears',
            'function': 'used for',
            'mechanism': 'works through',
            'relation': 'related to',
        }

        parts = []
        for layer_name, verb in labels.items():
            conns = layer_conns.get(layer_name, {})
            if not conns:
                continue
            # Filter and sort
            items = sorted(conns.items(), key=lambda x: -x[1])
            words = [k for k, v in items if k != w and len(k) > 2][:3]
            if words:
                parts.append(f"{verb} {', '.join(words)}")

        # Fall back to _general if typed layers are empty
        if not parts:
            gen = layer_conns.get('_general', {})
            if gen:
                items = sorted(gen.items(), key=lambda x: -x[1])
                words = [k for k, v in items if k != w and len(k) > 2][:5]
                if words:
                    parts.append(f"connects to {', '.join(words)}")

        # Also add co-occurrence neighbors not already mentioned
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
