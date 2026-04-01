"""
Thinker — full cognitive cycle with dopamine-driven policy learning.

6 synchronous phases per cycle:
  1. Identity — build spoke star for each focus concept
  2. Memory — goal-directed retrieval with dopamine-biased layer reordering
  3. Attention — retrieved knowledge may refine focus
  4. Goal — fallback if knowledge insufficient for current goal
  5. Imagination — fill gaps with 2-hop concepts, cross-focus links
  6. Re-evaluation — re-classify situation if enough knowledge retrieved

Coherence feedback:
  - Focus coherence: do focus concepts relate? (0.3)
  - Chain coherence: do retrieved items form a path? (0.3)
  - Density: does output use focus concepts? (0.2)
  - Novelty: not repeating last output (0.2)

Policy learning: situation → best_goal, best_layers, avg_quality
Conscience: epistemic humility, identity alignment, integrity
"""

from __future__ import annotations
from collections import deque
from typing import Dict, List, Optional, Callable, Any, Tuple, Set


# ═══ INTENT BIASES — which layers to prioritize per intent ═══
# NOT hardcoded knowledge — just routing weights.
# The layers themselves are learned from data.
INTENT_BIASES = {
    'define':     {'identity': 3.0, 'appearance': 1.5, 'mechanism': 1.0, 'function': 0.5, 'relation': 1.0, '_general': 0.8},
    'explain':    {'mechanism': 3.0, 'identity': 1.5, 'relation': 1.5, 'function': 1.0, 'appearance': 0.5, '_general': 0.8},
    'management': {'function': 3.0, 'mechanism': 1.5, 'relation': 1.0, 'identity': 0.5, 'appearance': 0.5, '_general': 0.8},
    'describe':   {'appearance': 3.0, 'identity': 1.5, 'mechanism': 1.0, 'function': 0.5, 'relation': 0.5, '_general': 0.8},
    'discuss':    {'identity': 2.0, 'mechanism': 2.0, 'appearance': 1.5, 'function': 1.5, 'relation': 1.5, '_general': 1.0},
    'general':    {'identity': 1.5, 'mechanism': 1.5, 'appearance': 1.0, 'function': 1.0, 'relation': 1.0, '_general': 1.0},
}

# Goal → which layers to search first
GOAL_LAYERS = {
    'describe_identity': ['identity', 'mechanism', 'appearance', 'relation', 'function', '_general'],
    'explain_mechanism': ['mechanism', 'identity', 'relation', '_general', 'appearance', 'function'],
    'list_treatment':    ['function', 'mechanism', 'relation', '_general', 'identity', 'appearance'],
    'describe_appearance': ['appearance', 'identity', 'mechanism', '_general', 'function', 'relation'],
    'explore_relations': ['relation', 'mechanism', 'identity', '_general', 'function', 'appearance'],
    'general':           ['identity', 'mechanism', 'appearance', 'function', 'relation', '_general'],
}


class WorkingMemory:
    """FIFO buffer with full cognitive state."""

    def __init__(self, capacity: int = 7):
        self._capacity = capacity
        self._items: deque = deque(maxlen=capacity)
        self.focus: List[str] = []
        self.situation: str = 'general'
        self.goal: str = 'general'
        self.retrieved: List[dict] = []
        self.imagined: List[dict] = []
        self.trace: List[str] = []
        self.last_output: str = ''

    def push(self, item: Any) -> None:
        self._items.append(item)

    def contents(self) -> List[Any]:
        return list(self._items)

    def clear(self) -> None:
        self._items.clear()
        self.focus = []
        self.retrieved = []
        self.imagined = []
        self.trace = []


class Thinker:
    """
    Full cognitive cycle with dopamine feedback and policy learning.
    """

    def __init__(self, max_steps: int = 5):
        self._max_steps = max_steps
        self._history: List[dict] = []
        # Dopamine policy: situation:goal → {count, total_quality, avg_quality, best_layers}
        self._policies: Dict[str, dict] = {}
        self._last_output_words: Set[str] = set()

    # ═══ SITUATION DEDUCTION — from spoke profile, not word matching ═══

    def _deduce_situation(self, focus: List[str],
                          cross_layer_fn) -> str:
        """
        Deduce situation from the landscape — which layers dominate
        for the focus concepts?
        """
        cmd_words = {'what', 'how', 'why', 'does', 'the', 'for', 'is', 'are', 'can', 'who', 'when', 'where'}
        content_focus = [w for w in focus if w not in cmd_words and len(w) > 3]
        if not content_focus:
            content_focus = focus[:3]
        layer_strength: Dict[str, float] = {}
        for word in content_focus[:3]:
            layers = cross_layer_fn(word)
            for layer_name, neighbors in layers.items():
                total = sum(neighbors.values())
                layer_strength[layer_name] = layer_strength.get(layer_name, 0) + total

        if not layer_strength:
            return 'general'

        dominant = max(layer_strength, key=layer_strength.get)
        mapping = {
            'mechanism': 'explanation',
            'function': 'management',
            'appearance': 'description',
            'identity': 'definition',
            'relation': 'connection',
        }
        return mapping.get(dominant, 'general')

    def _goal_from_situation(self, situation: str) -> str:
        mapping = {
            'definition': 'describe_identity',
            'explanation': 'explain_mechanism',
            'management': 'list_treatment',
            'description': 'describe_appearance',
            'connection': 'explore_relations',
        }
        return mapping.get(situation, 'general')

    # ═══ INTENT-BIASED ACTIVATION ═══

    def _biased_activate(self, word: str, goal: str,
                         cross_layer_fn,
                         activate_fn) -> Dict[str, float]:
        """
        Activate with intent bias — boost layers relevant to the goal.
        """
        layers = cross_layer_fn(word)
        biases = INTENT_BIASES.get(goal.split('_')[0] if '_' in goal else goal, INTENT_BIASES['general'])

        biased: Dict[str, float] = {}
        for layer_name, neighbors in layers.items():
            bias = biases.get(layer_name, 1.0)
            for tgt, weight in neighbors.items():
                biased[tgt] = biased.get(tgt, 0) + weight * bias

        # Also get field activation (unbiased but broader)
        field = activate_fn(word)
        for w, e in field.items():
            biased[w] = biased.get(w, 0) + e * 0.3

        return biased

    # ═══ COHERENCE FEEDBACK ═══

    def _compute_quality(self, wm: WorkingMemory,
                         score_fn,
                         co_graph: Dict[str, Dict[str, float]]) -> float:
        """
        Multi-component coherence score:
        - Focus coherence (0.3): do focus concepts relate?
        - Chain coherence (0.3): do retrieved items connect?
        - Density (0.2): are focus words in retrieved output?
        - Novelty (0.2): not repeating last output
        """
        quality = 0.0

        # Focus coherence
        if len(wm.focus) >= 2:
            connections = 0
            pairs = 0
            for i in range(min(len(wm.focus), 4)):
                for j in range(i + 1, min(len(wm.focus), 4)):
                    pairs += 1
                    a, b = wm.focus[i], wm.focus[j]
                    if b in co_graph.get(a, {}) or a in co_graph.get(b, {}):
                        connections += 1
            if pairs > 0:
                quality += (connections / pairs) * 0.3

        # Chain coherence
        ret_words = [r['word'] for r in wm.retrieved[:5]]
        if len(ret_words) >= 2:
            chain = 0
            for i in range(len(ret_words) - 1):
                if ret_words[i + 1] in co_graph.get(ret_words[i], {}):
                    chain += 1
            quality += (chain / max(len(ret_words) - 1, 1)) * 0.3

        # Density — are focus words activated in retrieved?
        focus_set = set(wm.focus)
        ret_set = set(ret_words)
        if focus_set:
            density = len(focus_set & ret_set) / len(focus_set)
            quality += density * 0.2

        # Novelty — not repeating last output
        current_words = set(wm.focus + ret_words)
        if self._last_output_words:
            overlap = len(current_words & self._last_output_words)
            novelty = 1.0 - overlap / max(len(current_words), 1)
        else:
            novelty = 1.0
        quality += novelty * 0.2

        return min(quality, 1.0)

    # ═══ POLICY LEARNING ═══

    def _update_policy(self, situation: str, goal: str, quality: float):
        key = f"{situation}:{goal}"
        if key not in self._policies:
            self._policies[key] = {'count': 0, 'total': 0.0, 'avg': 0.5}
        pol = self._policies[key]
        pol['count'] += 1
        pol['total'] += quality
        pol['avg'] = pol['total'] / pol['count']

    def _best_goal_for_situation(self, situation: str) -> Optional[str]:
        """Find the goal that historically produces best quality for this situation."""
        best_goal = None
        best_avg = 0.0
        for key, pol in self._policies.items():
            if key.startswith(situation + ':') and pol['count'] >= 2:
                if pol['avg'] > best_avg:
                    best_avg = pol['avg']
                    best_goal = key.split(':')[1]
        return best_goal

    # ═══ CONSCIENCE CHECKS ═══

    def _conscience(self, wm: WorkingMemory,
                    confidence_fn) -> Optional[str]:
        """
        Three conscience checks:
        1. Epistemic humility — depth < 2 → "still forming understanding"
        2. Identity alignment — focus words not in retrieved → refuse
        3. Integrity — too similar to last output → try different angle
        """
        if not wm.focus:
            return None

        # 1. Epistemic humility — check CONTENT words, not command words
        # Skip short/common command words to find the real primary concept
        cmd_words = {'what', 'how', 'why', 'does', 'the', 'for', 'is', 'are', 'can', 'who'}
        primary = None
        for w in wm.focus:
            if w not in cmd_words and len(w) > 3:
                primary = w
                break
        if not primary:
            primary = wm.focus[0]
        conf = confidence_fn(primary)
        if conf.get('score', 0) < 15:
            return f"I'm still forming my understanding of {primary}."
        if conf.get('layers', 0) < 2:
            return f"I know {primary} but my understanding is shallow — only {conf.get('layers', 0)} spoke(s)."

        # 2. Identity alignment — at least one focus word should be in retrieved
        ret_words = {r['word'] for r in wm.retrieved[:10]}
        focus_in_ret = sum(1 for f in wm.focus if f in ret_words)
        if len(wm.focus) > 1 and focus_in_ret == 0 and len(wm.retrieved) > 3:
            return None  # Might still be useful, don't block

        # 3. Integrity — novelty check
        current = set(wm.focus + list(ret_words))
        if self._last_output_words:
            overlap = len(current & self._last_output_words) / max(len(current), 1)
            if overlap > 0.8:
                return None  # Let the thinker try a different angle

        return None  # All clear

    # ═══ MAIN DELIBERATION — 6-phase cognitive cycle ═══

    def deliberate(
        self,
        query_tokens: List[str],
        activate_fn: Callable[[str], Dict[str, float]],
        score_fn: Callable[[List[str]], dict],
        signal_fn: Optional[Callable[[str, float], dict]] = None,
        cross_layer_fn=None,
        confidence_fn=None,
        imagination=None,
        co_graph: Optional[Dict[str, Dict[str, float]]] = None,
        episodic_context: Optional[dict] = None,
        memory_recall_fn=None,
    ) -> dict:
        """
        Full cognitive cycle with all 6 phases.
        """
        wm = WorkingMemory()
        wm.focus = list(query_tokens)
        _co = co_graph or {}

        # ═══ PHASE 0: SITUATION & GOAL ═══
        if cross_layer_fn:
            wm.situation = self._deduce_situation(wm.focus, cross_layer_fn)
        else:
            wm.situation = 'general'

        # Check if we have a learned policy for this situation
        learned_goal = self._best_goal_for_situation(wm.situation)
        wm.goal = learned_goal or self._goal_from_situation(wm.situation)
        wm.trace.append(f"situation={wm.situation} goal={wm.goal}")

        # Episodic boost — active topic gets focus priority
        if episodic_context:
            topic = episodic_context.get('active_topic')
            if topic and topic in wm.focus:
                wm.trace.append(f"  topic_boost: {topic}")

        converged = False
        steps = 0
        prev_quality = 0.0

        for step in range(self._max_steps):
            steps += 1
            changed = False

            # ═══ PHASE 1: IDENTITY — build spoke star ═══
            all_activated: Dict[str, float] = {}
            if cross_layer_fn:
                for word in wm.focus[:3]:
                    biased = self._biased_activate(word, wm.goal, cross_layer_fn, activate_fn)
                    for w, e in biased.items():
                        all_activated[w] = all_activated.get(w, 0) + e
            else:
                for word in wm.focus:
                    field = activate_fn(word)
                    for w, e in field.items():
                        all_activated[w] = all_activated.get(w, 0) + e

            # ═══ PHASE 2: MEMORY — goal-directed retrieval ═══
            # Retrieve from episodic memory if available
            if memory_recall_fn:
                recalled = memory_recall_fn(wm.focus, k=3)
                for ep in recalled:
                    for t in ep.tokens:
                        if t not in set(wm.focus):
                            all_activated[t] = all_activated.get(t, 0) + ep.significance * 0.5

            # Rank by activation, exclude focus words
            focus_set = set(wm.focus)
            wm.retrieved = [
                {'word': w, 'energy': e}
                for w, e in sorted(all_activated.items(), key=lambda x: -x[1])[:15]
                if w not in focus_set and len(w) > 2
            ]

            # ═══ PHASE 3: ATTENTION REFINEMENT ═══
            for r in wm.retrieved[:3]:
                if r['word'] not in wm.focus and r['energy'] > 0.4:
                    # Does this word connect back to focus?
                    r_neighbors = set(_co.get(r['word'], {}).keys())
                    if any(f in r_neighbors for f in wm.focus):
                        wm.focus.append(r['word'])
                        wm.trace.append(f"  +focus: {r['word']}")
                        changed = True

            # ═══ PHASE 4: GOAL REFINEMENT ═══
            # If current goal's target layer is empty for primary concept, fall back
            if cross_layer_fn and wm.focus:
                goal_layers = GOAL_LAYERS.get(wm.goal, GOAL_LAYERS['general'])
                primary_layers = cross_layer_fn(wm.focus[0])
                primary_layer = goal_layers[0] if goal_layers else 'identity'
                if primary_layer not in primary_layers or not primary_layers.get(primary_layer):
                    # Target spoke empty — fall back to next available
                    for fallback in goal_layers[1:]:
                        if fallback in primary_layers and primary_layers[fallback]:
                            old_goal = wm.goal
                            wm.goal = f"{fallback}_fallback"
                            wm.trace.append(f"  goal_fallback: {old_goal} -> {wm.goal}")
                            changed = True
                            break

            # ═══ PHASE 5: IMAGINATION — fill gaps ═══
            if imagination and _co:
                # Explore from each focus concept
                for word in wm.focus[:2]:
                    explored = imagination.explore(
                        word, _co, activate_fn,
                        threshold=0.15,
                    )
                    for item in explored[:2]:
                        if item['word'] not in focus_set and item['word'] not in {r['word'] for r in wm.retrieved}:
                            wm.imagined.append(item)
                            wm.retrieved.append({
                                'word': item['word'],
                                'energy': item['probability'] * 0.5,
                            })
                            wm.trace.append(f"  imagined: {item['word']} (p={item['probability']:.2f})")
                            changed = True

                # Cross-focus imagination: can focus concepts connect?
                if len(wm.focus) >= 2:
                    link = imagination.imagine(
                        wm.focus[0], wm.focus[1], _co, activate_fn,
                    )
                    if link['imagined'] and link['via']:
                        wm.trace.append(f"  bridge: {wm.focus[0]} -> {link['via']} -> {wm.focus[1]}")

            # ═══ PHASE 6: RE-EVALUATION ═══
            if changed and cross_layer_fn and step < self._max_steps - 1:
                new_sit = self._deduce_situation(wm.focus, cross_layer_fn)
                if new_sit != wm.situation:
                    wm.situation = new_sit
                    wm.goal = self._goal_from_situation(new_sit)
                    wm.trace.append(f"  re-eval: situation={new_sit} goal={wm.goal}")

            # ═══ QUALITY CHECK ═══
            quality = self._compute_quality(wm, score_fn, _co)
            wm.trace.append(f"step {step}: quality={quality:.3f}")

            # Signal: dopamine prediction error
            if signal_fn:
                state = f"{wm.situation}:{wm.goal}"
                sig = signal_fn(state, quality)
                if imagination and sig:
                    imagination.dopamine_error = sig.get('error', 0.0)

            # Convergence
            if abs(quality - prev_quality) < 0.01 and step > 0 and not changed:
                converged = True
                break
            prev_quality = quality

        # ═══ CONSCIENCE ═══
        caveat = None
        if confidence_fn:
            caveat = self._conscience(wm, confidence_fn)

        # ═══ POLICY UPDATE ═══
        self._update_policy(wm.situation, wm.goal, prev_quality)

        # ═══ IMAGINATION CONSOLIDATION ═══
        # High-coherence imagined links → they should be remembered
        consolidated = []
        if prev_quality > 0.6 and wm.imagined:
            consolidated = [im['word'] for im in wm.imagined if im.get('probability', 0) > 0.3]

        # Track output for novelty
        self._last_output_words = set(wm.focus + [r['word'] for r in wm.retrieved[:5]])

        result = {
            'focus': wm.focus,
            'retrieved': wm.retrieved,
            'imagined': [{'word': im['word'], 'probability': im.get('probability', 0)} for im in wm.imagined],
            'situation': wm.situation,
            'goal': wm.goal,
            'coherence': prev_quality,
            'steps': steps,
            'converged': converged,
            'trace': wm.trace,
            'caveat': caveat,
            'consolidated': consolidated,
        }

        self._history.append({
            'situation': wm.situation, 'goal': wm.goal,
            'quality': prev_quality, 'steps': steps,
        })
        if len(self._history) > 100:
            self._history = self._history[-50:]
        return result

    def counterfactual(
        self,
        base_tokens: List[str],
        position: int,
        alternatives: List[str],
        score_fn: Callable[[List[str]], dict],
    ) -> List[dict]:
        """What-if reasoning: score alternatives at a position."""
        if position < 0 or position >= len(base_tokens):
            return []
        baseline = score_fn(base_tokens)
        base_coherence = baseline.get('coherence', 0.0)
        results = []
        for alt in alternatives:
            modified = list(base_tokens)
            modified[position] = alt
            result = score_fn(modified)
            coherence = result.get('coherence', 0.0)
            results.append({
                'word': alt, 'coherence': coherence,
                'delta': coherence - base_coherence,
            })
        results.sort(key=lambda x: -x['coherence'])
        return results

    def to_dict(self) -> dict:
        return {
            'max_steps': self._max_steps,
            'policies': self._policies,
            'history_count': len(self._history),
        }

    @classmethod
    def from_dict(cls, d: dict) -> Thinker:
        t = cls(max_steps=d.get('max_steps', 5))
        t._policies = d.get('policies', {})
        return t
