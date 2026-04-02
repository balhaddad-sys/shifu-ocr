"""
Unified Activation Field — the bridge between cognition and perception.

Wave propagation through co-occurrence, next-word, resonance, and
skip-gram graphs. Context gating. Iterative settling.

This is the cognitive equivalent of the ShifuEngine wave system
(core/engine.js), translated to operate on any graph structure.
It scores both text coherence and OCR candidate quality using
the same wave mechanics.

The field operates on PASSED-IN graphs — it owns no data.
This makes it a pure function of whatever knowledge the cortex provides.
"""

from __future__ import annotations
import math
from typing import Dict, List, Optional, Tuple, Set, Callable, Any


def _field_cosine(a: Dict[str, float], b: Dict[str, float]) -> float:
    """Cosine similarity between two activation fields."""
    dot = 0.0
    na = 0.0
    nb = 0.0
    for k, v in a.items():
        na += v * v
        if k in b:
            dot += v * b[k]
    for v in b.values():
        nb += v * v
    na = math.sqrt(na)
    nb = math.sqrt(nb)
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return dot / (na * nb)


class Field:
    """
    Unified activation field with wave propagation and settling.

    All parameters are injectable. The field operates on external graphs
    passed as arguments — it holds no knowledge state itself.
    """

    def __init__(
        self,
        wave_decay: float = 0.3,
        max_hop1: int = 15,
        max_hop2_nodes: int = 5,
        max_hop2_each: int = 8,
        co_weight: float = 0.30,
        nx_weight: float = 0.25,
        res_weight: float = 0.25,
        snx_weight: float = 0.20,
        inhibition_floor: float = 0.1,
        inhibition_ceiling: float = 1.0,
        settle_max_iter: int = 3,
        settle_epsilon: float = 0.01,
        settle_reactivate_top: int = 8,
        settle_iter_decay: float = 0.5,
        gate_dampen: float = 0.4,
        gate_boost: float = 1.5,
    ):
        self._wave_decay = wave_decay
        self._max_hop1 = max_hop1
        self._max_hop2_nodes = max_hop2_nodes
        self._max_hop2_each = max_hop2_each
        self._co_weight = co_weight
        self._nx_weight = nx_weight
        self._res_weight = res_weight
        self._snx_weight = snx_weight
        self._inh_floor = inhibition_floor
        self._inh_ceiling = inhibition_ceiling
        self._settle_max_iter = settle_max_iter
        self._settle_epsilon = settle_epsilon
        self._settle_top = settle_reactivate_top
        self._settle_decay = settle_iter_decay
        self._gate_dampen = gate_dampen
        self._gate_boost = gate_boost

        # Cached median values for inhibition
        self._median_freq: float = 1.0
        self._median_degree: float = 1.0

    # ═══ INHIBITION ═══

    def update_medians(self, word_freqs: Dict[str, float],
                       co_graph: Dict[str, Dict[str, float]]) -> None:
        """Recompute median frequency/degree for inhibition normalization."""
        freqs = sorted(word_freqs.values())
        if freqs:
            self._median_freq = max(freqs[len(freqs) // 2], 1.0)
        degrees = sorted(len(co_graph.get(w, {})) for w in word_freqs)
        if degrees:
            self._median_degree = max(degrees[len(degrees) // 2], 1.0)

    def _inhibition_weight(self, word: str, word_freqs: Dict[str, float],
                           co_graph: Dict[str, Dict[str, float]]) -> float:
        """
        Frequency + degree based inhibition.
        High-frequency, high-degree words are dampened to prevent dominance.
        """
        freq = word_freqs.get(word, 0)
        if freq == 0:
            return self._inh_ceiling
        freq_ratio = freq / max(self._median_freq, 1)
        freq_inh = 1.0 / math.log2(freq_ratio + 2)
        degree = len(co_graph.get(word, {}))
        deg_ratio = degree / max(self._median_degree, 1)
        deg_inh = 1.0 / math.log2(deg_ratio + 2)
        raw = freq_inh * 0.5 + deg_inh * 0.5
        return max(self._inh_floor, min(self._inh_ceiling, raw))

    # ═══ WAVE PROPAGATION ═══

    def _spread_hop1(self, word: str, graph: Dict[str, Dict[str, float]],
                     base_energy: float, limit: int) -> Dict[str, float]:
        """Spread energy to direct neighbors, top-N by weight."""
        neighbors = graph.get(word, {})
        if not neighbors:
            return {}
        # Sort by weight descending, take top-limit
        sorted_n = sorted(neighbors.items(), key=lambda x: -x[1])[:limit]
        if not sorted_n:
            return {}
        max_w = sorted_n[0][1]
        if max_w < 1e-9:
            return {}
        result = {}
        for tgt, w in sorted_n:
            result[tgt] = base_energy * (w / max_w)
        return result

    def activate(
        self,
        word: str,
        co_graph: Dict[str, Dict[str, float]],
        nx_graph: Optional[Dict[str, Dict[str, float]]] = None,
        res_graph: Optional[Dict[str, Dict[str, float]]] = None,
        snx_graph: Optional[Dict[str, Dict[str, float]]] = None,
        word_freqs: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        """
        Activate a word. Spreads energy through 4 graph types:
        co-occurrence, next-word, resonance, skip-gram.

        Returns: Dict[word -> energy]
        """
        field: Dict[str, float] = {}
        field[word] = 1.0
        wf = word_freqs or {}

        # Hop-1: spread through each graph type
        for graph, weight in [
            (co_graph, self._co_weight),
            (nx_graph, self._nx_weight),
            (res_graph, self._res_weight),
            (snx_graph, self._snx_weight),
        ]:
            if graph is None:
                continue
            spread = self._spread_hop1(word, graph, weight, self._max_hop1)
            for tgt, energy in spread.items():
                if tgt == word:
                    continue
                inh = self._inhibition_weight(tgt, wf, co_graph)
                field[tgt] = field.get(tgt, 0.0) + energy * inh

        # Hop-2: selective propagation through top co-occurrence neighbors
        hop1_nodes = sorted(field.items(), key=lambda x: -x[1])
        hop2_sources = [
            (w, e) for w, e in hop1_nodes[:self._max_hop2_nodes]
            if w != word
        ]
        for h1_word, h1_energy in hop2_sources:
            h2_spread = self._spread_hop1(
                h1_word, co_graph,
                h1_energy * self._wave_decay,
                self._max_hop2_each,
            )
            for tgt, energy in h2_spread.items():
                if tgt == word or tgt == h1_word:
                    continue
                inh = self._inhibition_weight(tgt, wf, co_graph)
                field[tgt] = field.get(tgt, 0.0) + energy * inh

        return field

    def activate_in_context(
        self,
        word: str,
        context_field: Dict[str, float],
        co_graph: Dict[str, Dict[str, float]],
        nx_graph: Optional[Dict[str, Dict[str, float]]] = None,
        res_graph: Optional[Dict[str, Dict[str, float]]] = None,
        snx_graph: Optional[Dict[str, Dict[str, float]]] = None,
        word_freqs: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        """
        Context-gated activation. Energy is boosted for words present
        in the context field and dampened otherwise.
        """
        raw = self.activate(word, co_graph, nx_graph, res_graph, snx_graph, word_freqs)
        gated: Dict[str, float] = {}
        for w, energy in raw.items():
            if w in context_field:
                gated[w] = energy * self._gate_boost
            else:
                gated[w] = energy * self._gate_dampen
        return gated

    # ═══ SETTLING ═══

    def settle(
        self,
        initial_field: Dict[str, float],
        co_graph: Dict[str, Dict[str, float]],
        nx_graph: Optional[Dict[str, Dict[str, float]]] = None,
        res_graph: Optional[Dict[str, Dict[str, float]]] = None,
        snx_graph: Optional[Dict[str, Dict[str, float]]] = None,
        word_freqs: Optional[Dict[str, float]] = None,
    ) -> dict:
        """
        Iteratively reactivate top-N nodes until the field converges.
        Returns: {'field': Dict, 'iterations': int, 'converged': bool}
        """
        field = dict(initial_field)
        converged = False
        iterations = 0

        for i in range(self._settle_max_iter):
            iterations += 1
            old_field = dict(field)

            # Reactivate top-N nodes
            top_nodes = sorted(field.items(), key=lambda x: -x[1])[:self._settle_top]
            decay_mult = self._settle_decay ** i

            for node, energy in top_nodes:
                sub = self.activate(
                    node, co_graph, nx_graph, res_graph, snx_graph, word_freqs,
                )
                for w, e in sub.items():
                    field[w] = field.get(w, 0.0) + e * decay_mult

            # Check convergence
            delta = _field_cosine(old_field, field)
            if delta > 1.0 - self._settle_epsilon:
                converged = True
                break

        return {'field': field, 'iterations': iterations, 'converged': converged}

    # ═══ SEQUENCE SCORING ═══

    def score_sequence(
        self,
        tokens: List[str],
        co_graph: Dict[str, Dict[str, float]],
        nx_graph: Optional[Dict[str, Dict[str, float]]] = None,
        res_graph: Optional[Dict[str, Dict[str, float]]] = None,
        snx_graph: Optional[Dict[str, Dict[str, float]]] = None,
        word_freqs: Optional[Dict[str, float]] = None,
    ) -> dict:
        """
        Score a sequence of tokens for coherence.
        Builds a rolling field, scoring each word's alignment with context.

        Returns: {
            'coherence': float (0-1),
            'scores': List[float] (per-word),
            'field': Dict (final activation state)
        }
        """
        if not tokens:
            return {'coherence': 0.0, 'scores': [], 'field': {}}

        cumulative_field: Dict[str, float] = {}
        per_word_scores: List[float] = []

        for i, token in enumerate(tokens):
            if i == 0:
                # First word: just activate, no context to score against
                word_field = self.activate(
                    token, co_graph, nx_graph, res_graph, snx_graph, word_freqs,
                )
                per_word_scores.append(1.0)
            else:
                # Score: how well does this word fit the accumulated field?
                score = cumulative_field.get(token, 0.0)
                per_word_scores.append(min(score, 1.0))
                # Activate in context of what came before
                word_field = self.activate_in_context(
                    token, cumulative_field,
                    co_graph, nx_graph, res_graph, snx_graph, word_freqs,
                )

            # Merge into cumulative field
            for w, e in word_field.items():
                cumulative_field[w] = cumulative_field.get(w, 0.0) + e

        coherence = sum(per_word_scores) / len(per_word_scores) if per_word_scores else 0.0
        return {
            'coherence': coherence,
            'scores': per_word_scores,
            'field': cumulative_field,
        }

    # ═══ OCR CANDIDATE SCORING — THE PERCEPTION-COGNITION BRIDGE ═══

    def score_ocr_candidates(
        self,
        candidates: List[Tuple[str, float]],
        context_words: List[str],
        co_graph: Dict[str, Dict[str, float]],
        nx_graph: Optional[Dict[str, Dict[str, float]]] = None,
        res_graph: Optional[Dict[str, Dict[str, float]]] = None,
        snx_graph: Optional[Dict[str, Dict[str, float]]] = None,
        word_freqs: Optional[Dict[str, float]] = None,
        ocr_weight: float = 0.4,
        field_weight: float = 0.6,
    ) -> List[dict]:
        """
        Score OCR word candidates using field alignment + OCR confidence.

        candidates: [(word, ocr_confidence), ...]
        context_words: surrounding words for context field
        ocr_weight: how much to trust the raw OCR score
        field_weight: how much to trust cognitive context

        Returns: [{'word', 'ocr_score', 'field_score', 'combined', 'rank'}, ...]
                 sorted by combined score descending
        """
        if not candidates:
            return []

        # Build context field from surrounding words
        context_field: Dict[str, float] = {}
        for cw in context_words:
            word_field = self.activate(
                cw, co_graph, nx_graph, res_graph, snx_graph, word_freqs,
            )
            for w, e in word_field.items():
                context_field[w] = context_field.get(w, 0.0) + e

        # Score each candidate
        results = []
        for word, ocr_conf in candidates:
            # Field alignment: is this word activated by context?
            field_score = context_field.get(word, 0.0)
            # Also check: does activating this word produce overlap with context?
            if field_score < 0.01 and word_freqs and word in word_freqs:
                cand_field = self.activate(
                    word, co_graph, nx_graph, res_graph, snx_graph, word_freqs,
                )
                field_score = _field_cosine(cand_field, context_field)

            # Normalize field score to 0-1 range
            field_norm = min(field_score, 1.0)

            combined = ocr_weight * ocr_conf + field_weight * field_norm
            results.append({
                'word': word,
                'ocr_score': ocr_conf,
                'field_score': field_norm,
                'combined': combined,
            })

        # Sort by combined score
        results.sort(key=lambda x: -x['combined'])
        for i, r in enumerate(results):
            r['rank'] = i + 1
        return results

    # ═══ SERIALIZATION ═══

    def to_dict(self) -> dict:
        return {
            'wave_decay': self._wave_decay,
            'max_hop1': self._max_hop1,
            'max_hop2_nodes': self._max_hop2_nodes,
            'max_hop2_each': self._max_hop2_each,
            'co_weight': self._co_weight,
            'nx_weight': self._nx_weight,
            'res_weight': self._res_weight,
            'snx_weight': self._snx_weight,
            'inhibition_floor': self._inh_floor,
            'inhibition_ceiling': self._inh_ceiling,
            'settle_max_iter': self._settle_max_iter,
            'settle_epsilon': self._settle_epsilon,
            'settle_reactivate_top': self._settle_top,
            'settle_iter_decay': self._settle_decay,
            'gate_dampen': self._gate_dampen,
            'gate_boost': self._gate_boost,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Field:
        return cls(
            wave_decay=d.get('wave_decay', 0.3),
            max_hop1=d.get('max_hop1', 15),
            max_hop2_nodes=d.get('max_hop2_nodes', 5),
            max_hop2_each=d.get('max_hop2_each', 8),
            co_weight=d.get('co_weight', 0.30),
            nx_weight=d.get('nx_weight', 0.25),
            res_weight=d.get('res_weight', 0.25),
            snx_weight=d.get('snx_weight', 0.20),
            inhibition_floor=d.get('inhibition_floor', 0.1),
            inhibition_ceiling=d.get('inhibition_ceiling', 1.0),
            settle_max_iter=d.get('settle_max_iter', 3),
            settle_epsilon=d.get('settle_epsilon', 0.01),
            settle_reactivate_top=d.get('settle_reactivate_top', d.get('settle_top', 8)),
            settle_iter_decay=d.get('settle_iter_decay', d.get('settle_decay', 0.5)),
            gate_dampen=d.get('gate_dampen', 0.4),
            gate_boost=d.get('gate_boost', 1.5),
        )
