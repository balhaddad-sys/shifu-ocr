"""
Curriculum — structured practice exercises for language learning.

The baby practices at its level:
  Level 1 — WORD: decompose, find family, discover roots
  Level 2 — PHRASE: generate 2-3 word phrases, score structure
  Level 3 — SENTENCE: generate full sentences, score coherence + grammar
  Level 4 — PARAGRAPH: chain sentences, maintain topic coherence
  Level 5 — REASONING: answer questions, explain mechanisms, compare

Each level has exercises. The baby levels up when it consistently
scores above threshold. It levels down when it fails.

Autoregulation: the curriculum adjusts difficulty based on performance.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Any


class Curriculum:
    """
    Structured language practice with adaptive difficulty.

    practice(mind, level, rounds) → results
    assess(mind) → current level
    """

    def __init__(self):
        self.level = 1
        self.history: List[dict] = []
        self._level_thresholds = {
            1: 0.3,   # Word level: just find roots
            2: 0.4,   # Phrase level: produce coherent pairs
            3: 0.5,   # Sentence level: full sentences
            4: 0.6,   # Paragraph level: chained coherence
            5: 0.7,   # Reasoning level: answer + explain
        }
        self._consecutive_passes = 0
        self._consecutive_fails = 0

    def assess(self, mind) -> dict:
        """Assess current language ability across all levels."""
        results = {}
        for lvl in range(1, 6):
            score = self._test_level(mind, lvl, rounds=3)
            results[lvl] = score
        # Find appropriate level
        for lvl in range(5, 0, -1):
            if results[lvl] >= self._level_thresholds[lvl]:
                self.level = lvl
                break
        return {'level': self.level, 'scores': results}

    def practice(self, mind, rounds: int = 5, level: Optional[int] = None) -> dict:
        """
        Practice at current (or specified) level.
        Returns exercise results with scores.
        """
        lvl = level or self.level
        exercises = []

        for _ in range(rounds):
            if lvl == 1:
                ex = self._exercise_word(mind)
            elif lvl == 2:
                ex = self._exercise_phrase(mind)
            elif lvl == 3:
                ex = self._exercise_sentence(mind)
            elif lvl == 4:
                ex = self._exercise_paragraph(mind)
            else:
                ex = self._exercise_reasoning(mind)
            exercises.append(ex)

        # Compute average score
        avg = sum(e['score'] for e in exercises) / len(exercises) if exercises else 0
        threshold = self._level_thresholds.get(lvl, 0.5)

        # Autoregulate: level up/down based on performance
        if avg >= threshold:
            self._consecutive_passes += 1
            self._consecutive_fails = 0
            if self._consecutive_passes >= 3 and lvl < 5:
                self.level = lvl + 1
                self._consecutive_passes = 0
        else:
            self._consecutive_fails += 1
            self._consecutive_passes = 0
            if self._consecutive_fails >= 2 and lvl > 1:
                self.level = lvl - 1
                self._consecutive_fails = 0

        result = {
            'level': lvl,
            'rounds': len(exercises),
            'avg_score': round(avg, 3),
            'threshold': threshold,
            'passed': avg >= threshold,
            'current_level': self.level,
            'exercises': exercises,
        }
        self.history.append({
            'level': lvl, 'score': round(avg, 3),
            'passed': avg >= threshold,
        })
        if len(self.history) > 100:
            self.history = self.history[-50:]
        return result

    def _test_level(self, mind, level: int, rounds: int = 3) -> float:
        """Quick test of a level — returns average score."""
        total = 0
        for _ in range(rounds):
            if level == 1:
                ex = self._exercise_word(mind)
            elif level == 2:
                ex = self._exercise_phrase(mind)
            elif level == 3:
                ex = self._exercise_sentence(mind)
            elif level == 4:
                ex = self._exercise_paragraph(mind)
            else:
                ex = self._exercise_reasoning(mind)
            total += ex['score']
        return total / rounds if rounds > 0 else 0

    # ═══ LEVEL 1: WORD ═══
    def _exercise_word(self, mind) -> dict:
        """Find morphological family members for a random concept."""
        # Pick a word the mind knows
        candidates = sorted(
            mind.cortex.word_freq.items(), key=lambda x: -x[1]
        )[:100]
        if not candidates:
            return {'type': 'word', 'score': 0, 'detail': 'no vocabulary'}

        word = candidates[len(self.history) % len(candidates)][0]
        if not hasattr(mind, 'language') or not mind.language:
            return {'type': 'word', 'word': word, 'score': 0.3, 'detail': 'no language module'}

        decomp = mind.language.morphology.decompose(word)
        family = decomp.get('family', [])
        score = min(len(family) / 5.0, 1.0)  # 5+ family members = perfect
        return {
            'type': 'word', 'word': word,
            'root': decomp.get('root'),
            'family': family[:5],
            'score': round(score, 3),
        }

    # ═══ LEVEL 2: PHRASE ═══
    def _exercise_phrase(self, mind) -> dict:
        """Generate a 2-3 word phrase and score it."""
        candidates = [w for w, f in mind.cortex.word_freq.items() if f > 2 and len(w) > 3]
        if not candidates:
            return {'type': 'phrase', 'score': 0, 'detail': 'no vocabulary'}

        seed = candidates[len(self.history) % len(candidates)]
        phrase = mind.generate([seed], max_length=3)
        score_r = mind.score(phrase)
        coherence = score_r.get('coherence', 0)
        return {
            'type': 'phrase', 'seed': seed,
            'phrase': ' '.join(phrase),
            'score': round(coherence, 3),
        }

    # ═══ LEVEL 3: SENTENCE ═══
    def _exercise_sentence(self, mind) -> dict:
        """Generate a full sentence, score coherence + structure."""
        candidates = [w for w, f in mind.cortex.word_freq.items() if f > 2 and len(w) > 4]
        if not candidates:
            return {'type': 'sentence', 'score': 0, 'detail': 'no vocabulary'}

        seed = candidates[len(self.history) % len(candidates)]
        sentence = mind.generate([seed], max_length=10)
        coherence = mind.score(sentence).get('coherence', 0)

        # Structure score (if syntax module available)
        structure = 0.5
        if hasattr(mind, 'language') and mind.language and hasattr(mind.language, 'syntax'):
            structure = mind.language.syntax.score_structure(sentence)

        combined = coherence * 0.6 + structure * 0.4
        return {
            'type': 'sentence', 'seed': seed,
            'sentence': ' '.join(sentence),
            'coherence': round(coherence, 3),
            'structure': round(structure, 3),
            'score': round(combined, 3),
        }

    # ═══ LEVEL 4: PARAGRAPH ═══
    def _exercise_paragraph(self, mind) -> dict:
        """Generate 3 connected sentences, score topic coherence across them."""
        candidates = [w for w, f in mind.cortex.word_freq.items() if f > 3 and len(w) > 4]
        if len(candidates) < 3:
            return {'type': 'paragraph', 'score': 0, 'detail': 'need more vocabulary'}

        # Generate 3 sentences from related seeds
        seed1 = candidates[len(self.history) % len(candidates)]
        s1 = mind.generate([seed1], max_length=8)

        # Second sentence: seed from first sentence's strongest neighbor
        co = mind._co_graph.get(seed1, {})
        seed2 = max(co, key=co.get) if co else seed1
        s2 = mind.generate([seed2], max_length=8)

        # Third: seed from second
        co2 = mind._co_graph.get(seed2, {})
        seed3 = max(co2, key=co2.get) if co2 else seed2
        s3 = mind.generate([seed3], max_length=8)

        # Score: coherence of combined + topic continuity
        all_tokens = s1 + s2 + s3
        overall = mind.score(all_tokens).get('coherence', 0)
        # Topic continuity: overlap between consecutive sentences
        s1_set = set(s1)
        s2_set = set(s2)
        s3_set = set(s3)
        cont_12 = len(s1_set & s2_set) / max(len(s1_set | s2_set), 1)
        cont_23 = len(s2_set & s3_set) / max(len(s2_set | s3_set), 1)
        continuity = (cont_12 + cont_23) / 2

        combined = overall * 0.5 + continuity * 0.5
        return {
            'type': 'paragraph',
            'sentences': [' '.join(s1), ' '.join(s2), ' '.join(s3)],
            'coherence': round(overall, 3),
            'continuity': round(continuity, 3),
            'score': round(combined, 3),
        }

    # ═══ LEVEL 5: REASONING ═══
    def _exercise_reasoning(self, mind) -> dict:
        """Ask a question and assess the quality of deliberation."""
        candidates = [w for w, f in mind.cortex.word_freq.items() if f > 3 and len(w) > 4]
        if not candidates:
            return {'type': 'reasoning', 'score': 0, 'detail': 'need more vocabulary'}

        word = candidates[len(self.history) % len(candidates)]
        queries = [
            f"what is {word}",
            f"what causes {word}",
            f"how does {word} work",
            f"describe {word}",
        ]
        query = queries[len(self.history) % len(queries)]

        result = mind.deliberate(query)
        # Score: coherence + retrieved count + convergence
        coherence = result.get('coherence', 0)
        retrieved = len(result.get('retrieved', []))
        converged = 1.0 if result.get('converged') else 0.5
        retrieved_score = min(retrieved / 5.0, 1.0)

        combined = coherence * 0.4 + retrieved_score * 0.4 + converged * 0.2
        return {
            'type': 'reasoning', 'query': query,
            'retrieved': [r['word'] for r in result.get('retrieved', [])[:5]],
            'coherence': round(coherence, 3),
            'retrieved_count': retrieved,
            'converged': result.get('converged', False),
            'score': round(combined, 3),
        }

    def stats(self) -> dict:
        return {
            'level': self.level,
            'history_count': len(self.history),
            'consecutive_passes': self._consecutive_passes,
            'consecutive_fails': self._consecutive_fails,
            'recent': self.history[-5:],
        }

    def to_dict(self) -> dict:
        return {
            'level': self.level,
            'history': self.history[-50:],
            'passes': self._consecutive_passes,
            'fails': self._consecutive_fails,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Curriculum:
        c = cls()
        c.level = d.get('level', 1)
        c.history = d.get('history', [])
        c._consecutive_passes = d.get('passes', 0)
        c._consecutive_fails = d.get('fails', 0)
        return c
