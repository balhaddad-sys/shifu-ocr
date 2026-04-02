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
from .golgi import Golgi
from .packet import Packet, PacketStream
from .language import Morphology, Syntax, Semantics, Curriculum


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
        self.golgi = Golgi()

        # Language acquisition modules
        self.language = type('Language', (), {
            'morphology': Morphology(),
            'syntax': Syntax(),
            'semantics': Semantics(),
            'curriculum': Curriculum(),
        })()

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

        # 1b. THYMIC SELECTION — positive and negative gates
        #     Positive: must connect to at least 1 known concept (bypass during critical period)
        #     Negative: must not contradict established strong knowledge
        if self._feed_count > 200:
            known_count = sum(1 for w in content if w in self.cortex.word_freq and self.cortex.word_freq[w] > 1)
            # Positive selection: must connect to something known
            if known_count == 0 and len(content) > 3:
                return {
                    'accepted': False, 'tokens_absorbed': 0,
                    'domain': None, 'quality': filtered['quality'],
                }

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
            self.field.invalidate_cache()
            self.gate.adapt_thresholds()

        self._feed_count += 1

        return {
            'accepted': True,
            'tokens_absorbed': connections,
            'domain': domain,
            'quality': filtered['quality'],
        }

    def feed_batch(self, texts: List[str], layer: str = '_general',
                   classifier=None, cycles: int = 1) -> dict:
        """
        CAPILLARY-STYLE batch feeding.

        Like capillaries: many parallel small vessels reduce resistance.
        - Suppress pruning during entire batch (one prune at end)
        - Pre-compute stop words ONCE (not per sentence)
        - Inline gate logic (avoid per-sentence function call overhead)
        - Skip trunk/memory/resonance during batch (expensive, run once at end)
        - Skip assembly formation for duplicate content
        """
        old_prune = self.cortex._prune_interval
        self.cortex._prune_interval = max(old_prune, len(texts) * cycles + 1)

        # Pre-compute stop words ONCE
        stops = self.gate.stop_words(self.cortex.word_freq)
        # SKIP classification during batch — put everything in _general.
        # Classification scans all layers per word (O(layers × neighbors)).
        # During bulk ingestion this is 82% of CPU time for zero benefit.
        # The cortex will route connections properly on single feeds later.
        cls = classifier or (lambda w: None)

        accepted = 0
        total_tokens = 0
        rejected = 0
        seen_content = set()

        # ═══ NEUROGLIA ARCHITECTURE ═══
        #
        # Astrocytes detect which neurons fire hardest and dilate
        # blood vessels feeding THOSE neurons. We do the same:
        #
        # PHASE 1: Triage — classify every sentence by novelty.
        #   - How many words are NEW (never seen)?
        #   - How many words are RARE (seen < 3 times)?
        #   - novelty = (new + rare) / total
        #
        # PHASE 2: Route by load.
        #   - novelty > 0.5 → DEEP feed (full cortex connections + identity + speaker)
        #   - novelty 0.2-0.5 → MEDIUM feed (co-graph + nx only)
        #   - novelty < 0.2 → LIGHT feed (word freq increment only, skip graphs)
        #
        # This is NOT skipping. Every sentence is processed.
        # But resources flow to where the LEARNING is.

        gen_layer = self.cortex.ensure_layer('_general')

        for cycle in range(cycles):
            for text in texts:
                tokens = self.gate.tokenize(text)
                if len(tokens) < 2:
                    if cycle == 0:
                        rejected += 1
                    continue

                content = [t for t in tokens if t not in stops and len(t) > 2]
                if len(content) < 2:
                    if cycle == 0:
                        rejected += 1
                    continue

                # Deduplicate
                content_key = tuple(sorted(content[:8]))
                is_dup = content_key in seen_content
                seen_content.add(content_key)
                if is_dup:
                    if cycle == 0:
                        accepted += 1
                    continue

                # ═══ ASTROCYTE: measure novelty ═══
                # Not just "have I seen this word?" but "do I UNDERSTAND it?"
                # A word with freq=100 but 0 connections is still novel.
                # Novelty = new words + poorly-connected words + new combinations.
                new_words = 0
                thin_words = 0
                new_pairs = 0
                for w in content:
                    freq = self.cortex.word_freq.get(w, 0)
                    if freq == 0:
                        new_words += 1
                    else:
                        # Thin: seen but poorly connected (< 5 co-graph neighbors)
                        co_count = len(self._co_graph.get(w, {}))
                        if co_count < 5:
                            thin_words += 1
                # New combinations: pairs of content words not yet in co-graph
                if len(content) >= 2:
                    for i in range(min(len(content) - 1, 5)):
                        a, b = content[i], content[i + 1]
                        if a not in self._co_graph or b not in self._co_graph.get(a, {}):
                            new_pairs += 1
                novelty = (new_words + thin_words * 0.3 + new_pairs * 0.2) / max(len(content), 1)

                # ═══ ALL LEVELS: word frequency (always) ═══
                for w in content:
                    self.cortex.word_freq[w] = self.cortex.word_freq.get(w, 0) + 1
                    self.cortex.total_words += 1
                    if w not in self.cortex._connection_birth:
                        self.cortex._connection_birth[w] = self.cortex._epoch
                    # Cap breadth at 50 per word to prevent memory explosion
                    if w not in self.cortex.breadth:
                        self.cortex.breadth[w] = set()
                    elif len(self.cortex.breadth[w]) >= 50:
                        pass  # Already full — don't grow
                self.cortex._feed_count += 1
                self.cortex._epoch += 1

                connections = 0

                # ═══ SPECIALIZE → RELAY → RE-SPECIALIZE ═══
                #
                # 1. SPECIALIZE: Golgi tags each word with its pathway
                # 2. RELAY: each pathway processes with preserved provenance
                # 3. RE-SPECIALIZE: downstream uses packets by task
                #
                # The packet carries: origin, modality, birth, cohort,
                # confidence, position, lineage. Never stripped.

                stream = PacketStream().from_tokens(
                    content, self.cortex._epoch,
                    golgi=self.golgi, word_freqs=self.cortex.word_freq,
                )

                # RELAY through co-graph — each pathway gets its own cap
                caps = {1: 50, 2: 80, 3: 100, 4: 100}
                for p in stream.packets:
                    w = p.word
                    cap = caps.get(p.origin, 100)
                    if w not in self._co_graph:
                        self._co_graph[w] = {}
                    co_w = self._co_graph[w]
                    for other_p in stream.packets:
                        if other_p.word != w:
                            ob = other_p.word
                            if ob in co_w:
                                co_w[ob] += 1
                            elif len(co_w) < cap:
                                co_w[ob] = 1
                    p.relay('co_graph')

                # RELAY: content + specialist get breadth
                for p in stream.packets:
                    if p.origin >= 3:
                        b = self.cortex.breadth.get(p.word)
                        if b is not None and len(b) < 50:
                            for other_p in stream.packets[:10]:
                                if other_p.word != p.word:
                                    b.add(other_p.word)
                        p.relay('breadth')
                        connections += 1

                # Nx-graph from ALL tokens (grammar structure)
                for i in range(len(tokens) - 1):
                    w = tokens[i]
                    if w not in self._nx_graph:
                        self._nx_graph[w] = {}
                    nx_w = self._nx_graph[w]
                    nxt = tokens[i + 1]
                    if nxt in nx_w:
                        nx_w[nxt] += 1
                    elif len(nx_w) < 50:
                        nx_w[nxt] = 1

                # RE-SPECIALIZE: only specialist packets trigger deep processing
                specialists = stream.by_origin(4)
                if specialists:
                    self._extract_identity(text, content)
                    self.speaker.learn_frame(tokens)
                    self.language.syntax.feed(tokens)
                    for p in specialists:
                        p.relay('deep_processing')

                if cycle == 0:
                    accepted += 1
                    total_tokens += connections

                self._feed_count += 1
                self._epoch += 1

        # Single prune at end
        self.cortex._prune_interval = old_prune
        self.cortex._prune()

        # Post-batch finalization (all the expensive stuff, ONCE)
        if accepted > 5:
            self.trunk.finalize()
        self.field.update_medians(self.cortex.word_freq, self._co_graph)
        self.field.invalidate_cache()
        self.gate.adapt_thresholds()

        return {
            'total': len(texts),
            'accepted': accepted,
            'rejected': rejected,
            'tokens_absorbed': total_tokens,
            'cycles': cycles,
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

    def _classify_word_bridge(self, word: str) -> Optional[str]:
        """
        Route a word to a semantic spoke — NEVER to identity.
        Identity is the bridge, not a destination.

        Classification is LEARNED from the graph:
        - Which typed layers already have the strongest connections
          for this word's neighbors? Route there.
        - No hardcoded signal word lists.
        """
        co_neighbors = self._co_graph.get(word, {})
        if not co_neighbors:
            return None

        # Score each non-identity layer by how many of THIS word's
        # co-occurrence neighbors already live in that layer
        spoke_scores: Dict[str, float] = {}
        for spoke_name in self.cortex.layer_names:
            if spoke_name in ('_general', 'identity'):
                continue
            layer = self.cortex.get_layer(spoke_name)
            if not layer:
                continue
            score = 0.0
            for neighbor in co_neighbors:
                neighbor_conns = layer.get_neighbors(neighbor)
                if neighbor_conns:
                    score += sum(neighbor_conns.values())
            if score > 0:
                spoke_scores[spoke_name] = score

        if not spoke_scores:
            return None

        best = max(spoke_scores, key=spoke_scores.get)
        if spoke_scores[best] >= 0.5:
            return best
        return None

    def _extract_identity(self, raw_text: str, content: List[str]) -> None:
        """
        Extract copula patterns ("X is a Y") and write to identity layer.
        This is the ONLY way identity connections form.

        The patterns detected are structural (copula verbs), not
        domain-specific word lists.
        """
        if not content:
            return
        text_lower = raw_text.lower()
        identity_layer = self.cortex.ensure_layer('identity')
        subject = content[0]

        # Copula detection: "X is a Y", "X are Y", "X defined as Y"
        # These are GRAMMATICAL structures, not domain words
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
                    return

        if match:
            b = match.group(1)
            if b != subject and len(b) > 2:
                identity_layer.connect(subject, b, 2.0, self.cortex._epoch)
                identity_layer.connect(b, subject, 1.0, self.cortex._epoch)
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
            self._res_graph, self._snx_graph,
            self.cortex.word_freq,
        )

    def score(self, tokens: List[str]) -> dict:
        """Score coherence through neural propagation."""
        if self.neural_field.neurons:
            return self.neural_field.score_sequence(tokens)
        return self.field.score_sequence(
            tokens, self._co_graph, self._nx_graph,
            self._res_graph, self._snx_graph,
            self.cortex.word_freq,
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
            self._res_graph, self._snx_graph,
            self.cortex.word_freq,
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

        # Plateau but has conviction
        if purpose:
            return {
                'state': 'determined',
                'desire': f'The landscape is flat but I feel there is more. {purpose}.',
                'reason': f'Dopamine is low ({trend:.2f}) but conviction pulls me forward.',
                'action': 'practice',
            }

        # Content
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
        Called by the UI on every poll. The baby decides what to do.
        No commands. No guidance. It reads its compass and acts.

        Returns: {'did': what it did, 'result': outcome, 'voice': what it says}
        """
        c = self.compass()
        action = c.get('action')
        voice = c['desire']

        if action == 'consolidate':
            r = self.consolidate(focus_size=200)  # Small focused chunk
            return {
                'did': 'consolidate',
                'result': f"routed {r.get('routed',0)}, {r.get('identities',0)} identities",
                'voice': voice,
                'state': c['state'],
            }
        elif action == 'practice':
            r = self.practice(rounds=2)  # Quick practice
            return {
                'did': 'practice',
                'result': f"{r.get('improved',0)} reinforced",
                'voice': r.get('voice', voice),
                'state': c['state'],
            }
        else:
            # Resting or waiting — just report state
            return {
                'did': 'rest',
                'result': 'observing',
                'voice': voice,
                'state': c['state'],
            }

    def deliberate(self, query: str) -> dict:
        """
        specialize → relay → re-specialize

        Query tokens become PACKETS with provenance.
        Golgi tags them. Co-graph relays them. Spokes re-specialize.
        The thinker receives content packets sorted by confidence.
        Structural packets are used for grammar, not for meaning.
        """
        tokens = _tokenize(query)

        # SPECIALIZE: tag every token
        stream = PacketStream().from_tokens(
            tokens, self.cortex._epoch,
            golgi=self.golgi, word_freqs=self.cortex.word_freq,
        )
        # RELAY: through co-graph (boosts connected words)
        stream.relay_co_graph(self._co_graph)
        # RELAY: through spokes (routes to semantic destinations)
        stream.relay_spokes(self._co_graph)

        # RE-SPECIALIZE: content packets become the focus
        # Sorted by confidence — words the system KNOWS get priority
        content_packets = stream.by_confidence(min_conf=0.0)
        content = [p.word for p in content_packets if p.is_content()]
        if not content:
            # Fallback: use all packets sorted by confidence
            content = [p.word for p in content_packets if len(p.word) > 1][:10]
        if not content:
            return {'focus': [], 'retrieved': [], 'coherence': 0.0,
                    'steps': 0, 'converged': False, 'trace': [],
                    'situation': 'general', 'goal': 'general'}

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

        # Consolidation: high-quality imagined links → cortex
        for word in result.get('consolidated', []):
            if word and result['focus']:
                primary = result['focus'][0]
                general = self.cortex.ensure_layer('_general')
                general.connect(primary, word, 0.5, self.cortex._epoch)
                general.connect(word, primary, 0.3, self.cortex._epoch)

        # Self-tune every 10 deliberations
        if len(self.thinker._history) % 10 == 0:
            self._self_tune()

        return result

    def _self_tune(self) -> None:
        """Adjust cortex parameters based on recent quality trend."""
        trend = self.signal.recent_trend(5)
        if trend < 0.3:
            self.cortex._decay_factor = min(self.cortex._decay_factor * 1.01, 0.999)
        elif trend > 0.6:
            self.cortex._decay_factor = max(self.cortex._decay_factor * 0.999, 0.95)

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

    def consolidate(self, focus_size: int = 500) -> dict:
        """
        FOCUSED CONSOLIDATION — one chapter at a time.

        Not 57K words at once. 500 words the baby ATTENDS to.
        Focus selected by: conviction target, recent queries,
        then highest-frequency content words.

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
                    identities += 1

        # ═══ PHASES 2-5: Route by CO-OCCURRENCE (only new words) ═══
        # Instead of hardcoded lists, find structural words from nx_graph:
        # words that appear as next-word for MANY different preceding words
        # are likely structural (verbs/prepositions). The TARGETS of those
        # structural words get routed to the appropriate layer.

        # Mechanism: words that co-occur with the top "causes/produces" equivalents
        # (words that many different words precede → structural verbs)
        spoke_routing = {
            'mechanism': ['causes', 'caused', 'leads', 'results', 'produces', 'involves', 'through', 'due'],
            'function': ['treatment', 'therapy', 'treats', 'used', 'prevents', 'reduces', 'management'],
            'appearance': ['presents', 'shows', 'appears', 'seen', 'reveals', 'visible', 'characterized'],
            'relation': ['associated', 'related', 'risk', 'factor', 'linked', 'increases', 'compared'],
        }

        # FOCUSED spoke routing: only route words IN the focus set.
        # Top 10 neighbors per signal word. No nested fan-out.
        # 4 layers × 8 signals × 10 neighbors = 320 connections. Not 32,000.
        focus_set = set(words_to_process)
        for layer_name, signal_words in spoke_routing.items():
            typed = self.cortex.ensure_layer(layer_name)
            for signal in signal_words:
                co_neighbors = self._co_graph.get(signal, {})
                for neighbor, weight in sorted(co_neighbors.items(), key=lambda x: -x[1])[:10]:
                    if len(neighbor) > 3 and weight > 1 and neighbor in focus_set:
                        typed.connect(neighbor, signal, weight, current_epoch)
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
            neural_co, cortex_conns, myel_pairs, self.golgi,
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
            'feed_count': max(self._feed_count, self.cortex._feed_count),
            'epoch': max(self._epoch, self.cortex._epoch),
            'neural_neurons': self.neural_field.stats()['neurons'] if self.neural_field.neurons else 0,
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
            'imagination': self.imagination.to_dict(),
            'attention': self.attention.to_dict(),
            'conviction': self.conviction.to_dict(),
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
        mind.imagination = Imagination.from_dict(d.get('imagination', {}))
        mind.attention = Attention.from_dict(d.get('attention', {}))
        mind.conviction = Conviction.from_dict(d.get('conviction', {}))
        mind._co_graph = d.get('co_graph', {})
        mind._nx_graph = d.get('nx_graph', {})
        mind._px_graph = d.get('px_graph', {})
        mind._res_graph = d.get('res_graph', {})
        mind._snx_graph = d.get('snx_graph', {})
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

        # Find concepts worth practicing — FAST: skip confidence() call
        # (it's expensive). Use word_freq + breadth as proxy.
        candidates = []
        for word, freq in self.cortex.word_freq.items():
            if len(word) <= 3 or freq < 2:
                continue
            # Proxy for "still learning": has frequency but few connections
            breadth = len(self.cortex.breadth.get(word, set()))
            if breadth < 30:  # Not yet well-connected
                candidates.append((word, breadth))
        # Sort by breadth ascending — practice least connected first
        candidates.sort(key=lambda x: x[1])

        for i in range(min(rounds, len(candidates))):
            word, before_conf = candidates[i]

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
            gen_layer = self.cortex.get_layer('_general')
            if gen_layer and len(generated) >= 2:
                # Weight by coherence — but never zero, never negative
                # High coherence: strong step (0.5)
                # Low coherence: gentle step (0.05)
                # The floor is NOT zero. Every step matters.
                step_weight = max(0.05, coherence * 0.5)
                for j in range(len(generated) - 1):
                    a, b = generated[j], generated[j + 1]
                    gen_layer.connect(a, b, step_weight, self.cortex._epoch)
                    improved += 1

            # Feed EVERY sentence back — not just "good" ones.
            # The baby learns from ALL its attempts.
            # A bad sentence teaches what doesn't connect.
            # That knowledge is just as valuable.
            self.cortex.feed(generated, layer='_general')

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
        Continuous maintenance. Like blood + glucose supply.
        Called periodically to keep the mind alive.

        1. MYELINATION: connections used in recent deliberation/practice
           get their activation count incremented. Those reaching
           threshold get myelinated.
        2. SALTATORY CONDUCTION: myelinated connections create SHORTCUTS
           in the co-graph — direct links that skip intermediate nodes.
        3. DECAY: unused connections weaken slightly. Myelinated resist.
        """
        myelinated_new = 0
        shortcuts_created = 0

        # Phase 1: Myelinate connections that appear in co-graph with high weight
        # (proxy for "used repeatedly" — co-graph weight = co-occurrence count)
        gen = self.cortex.get_layer('_general')
        if gen:
            for source, targets in gen._connections.items():
                for target, syn in targets.items():
                    if syn.myelinated:
                        continue
                    # Check co-graph: how often do these words actually co-occur?
                    co_weight = self._co_graph.get(source, {}).get(target, 0)
                    if co_weight >= 3:
                        # Co-occurring 3+ times = reliable connection. Myelinate.
                        syn.myelinate()
                        myelinated_new += 1
                        self.cortex._myel_count += 1

        # Phase 2: Saltatory conduction — myelinated paths create shortcuts
        # If A→B myelinated and B→C myelinated, create A→→C shortcut
        if gen:
            new_shortcuts = []
            for source, targets in gen._connections.items():
                for target, syn in targets.items():
                    if not syn.myelinated:
                        continue
                    # This connection is myelinated. Look for myelinated extensions.
                    ext_targets = gen._connections.get(target, {})
                    for ext_target, ext_syn in ext_targets.items():
                        if not ext_syn.myelinated or ext_target == source:
                            continue
                        # A→B→C both myelinated. Create A→→C shortcut.
                        existing = gen.get_synapse(source, ext_target)
                        if not existing:
                            new_shortcuts.append((source, ext_target, min(syn.weight, ext_syn.weight) * 0.7))
                            shortcuts_created += 1
                            if shortcuts_created > 100:
                                break
                    if shortcuts_created > 100:
                        break
                if shortcuts_created > 100:
                    break
            for src, tgt, wt in new_shortcuts:
                s = gen.connect(src, tgt, wt, self.cortex._epoch)
                s.myelinate()  # Shortcuts are born myelinated
                self.cortex._myel_count += 1

        # Phase 3: Gentle decay (metabolism cost)
        self.cortex.tick()

        self.cortex._invalidate_cache()
        self.field.invalidate_cache()

        return {
            'myelinated_new': myelinated_new,
            'shortcuts': shortcuts_created,
        }

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
