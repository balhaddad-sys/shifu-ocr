"""
Curriculum — real learning through practice, not just scoring.

The difference between memorization and learning:
  Memorization: store "stroke causes disability" as a fact
  Learning: extract "X causes Y" as a PATTERN that applies to new cases

The curriculum practices by:
  1. GENERATE a sentence from a concept (using Broca's trajectory)
  2. SCORE it against stored episodes (does it match reality?)
  3. EXTRACT PATTERNS from high-scoring sentences (what structure repeats?)
  4. REINFORCE the pattern, not just the specific instance
  5. TEST transfer: can the pattern produce a new sentence about a DIFFERENT concept?

Patterns are stored as templates with slots:
  "X is a Y" → template with subject and category slots
  "X causes Y" → template with cause and effect slots
  "X is treated with Y" → template with condition and treatment slots

These patterns ARE the compressed knowledge. 25K words → 500 patterns.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Any
import re


class Pattern:
    """A reusable sentence template extracted from repeated structures."""

    __slots__ = ('template', 'slots', 'examples', 'strength', 'uses')

    def __init__(self, template: str, slots: List[str]):
        self.template = template    # e.g. "X is a Y"
        self.slots = slots          # e.g. ['subject', 'category']
        self.examples: List[dict] = []  # filled instances
        self.strength = 0.0         # how well-established
        self.uses = 0               # how many times used in generation

    def fill(self, values: Dict[str, str]) -> str:
        """Fill template with values."""
        result = self.template
        for slot, value in values.items():
            result = result.replace(slot.upper(), value)
        return result

    def matches(self, tokens: List[str]) -> Optional[Dict[str, str]]:
        """Does this token sequence match the pattern? Returns filled slots."""
        # Simple: check if template words appear in order
        tmpl_words = self.template.lower().split()
        slot_values = {}
        t_idx = 0
        for tw in tmpl_words:
            if tw in ('x', 'y', 'z'):
                # Slot — grab the next content word
                while t_idx < len(tokens) and len(tokens[t_idx]) <= 3:
                    t_idx += 1  # Skip structural words
                if t_idx < len(tokens):
                    slot_values[tw] = tokens[t_idx]
                    t_idx += 1
            else:
                # Fixed word — must match
                while t_idx < len(tokens) and tokens[t_idx] != tw:
                    t_idx += 1
                if t_idx < len(tokens):
                    t_idx += 1
                else:
                    return None  # Pattern doesn't match
        return slot_values if slot_values else None

    def to_dict(self) -> dict:
        return {
            'template': self.template,
            'slots': self.slots,
            'examples': self.examples[-10:],
            'strength': round(self.strength, 3),
            'uses': self.uses,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Pattern:
        p = cls(d['template'], d.get('slots', []))
        p.examples = d.get('examples', [])
        p.strength = d.get('strength', 0.0)
        p.uses = d.get('uses', 0)
        return p


class Curriculum:
    """
    Real learning: pattern extraction + transfer testing.

    practice() does:
      1. Pick a concept
      2. Generate sentence (Broca's trajectory)
      3. Score against episodes (reality check)
      4. Extract pattern if structure repeats
      5. Reinforce pattern
      6. Test: apply pattern to a NEW concept
    """

    def __init__(self):
        self.level = 1
        self.history: List[dict] = []
        self._auc: Dict[int, float] = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
        self._level_ratios: Dict[int, float] = {
            1: 0.02, 2: 0.06, 3: 0.15, 4: 0.40, 5: 1.00,
        }
        self._practice_count = 0
        self.patterns: List[Pattern] = []

    def _threshold(self, level: int, mind) -> float:
        vocab = max(len(mind.cortex.word_freq), 10)
        return self._level_ratios.get(level, 1.0) * vocab

    def assess(self, mind) -> dict:
        thresholds = {lvl: round(self._threshold(lvl, mind), 1) for lvl in range(1, 6)}
        return {
            'level': self.level,
            'auc': {k: round(v, 2) for k, v in self._auc.items()},
            'thresholds': thresholds,
            'practice_count': self._practice_count,
            'patterns': len(self.patterns),
        }

    def practice(self, mind, rounds: int = 2, level: Optional[int] = None) -> dict:
        lvl = level or self.level
        exercises = []

        for _ in range(rounds):
            if lvl <= 2:
                ex = self._exercise_generate_and_extract(mind)
            elif lvl <= 4:
                ex = self._exercise_transfer(mind)
            else:
                ex = self._exercise_transfer(mind)
            exercises.append(ex)
            self._practice_count += 1

        total_score = 0
        for ex in exercises:
            score = ex.get('score', 0)
            total_score += score
            self._auc[lvl] = self._auc.get(lvl, 0) + score

        avg = total_score / len(exercises) if exercises else 0
        threshold = self._threshold(lvl, mind)
        current_auc = self._auc.get(lvl, 0)
        passed = current_auc >= threshold

        if passed and lvl < 5:
            self.level = lvl + 1

        result = {
            'level': lvl,
            'rounds': len(exercises),
            'avg_score': round(avg, 3),
            'auc': round(current_auc, 2),
            'auc_needed': round(threshold, 1),
            'passed': passed,
            'current_level': self.level,
            'patterns': len(self.patterns),
            'exercises': exercises,
        }
        self.history.append({
            'level': lvl, 'score': round(avg, 3),
            'auc': round(current_auc, 2),
        })
        if len(self.history) > 100:
            self.history = self.history[-50:]
        return result

    def _exercise_generate_and_extract(self, mind) -> dict:
        """
        Generate a sentence. Compare with episodes. Extract pattern.
        This is where memorization becomes learning.
        """
        candidates = [w for w, f in mind.cortex.word_freq.items()
                      if f >= 1 and len(w) > 3]
        if not candidates:
            return {'type': 'generate', 'score': 0, 'detail': 'no vocabulary'}

        seed = candidates[self._practice_count % len(candidates)]

        # Generate using Broca's trajectory
        sentence = mind.generate([seed], max_length=10)
        if len(sentence) < 3:
            return {'type': 'generate', 'seed': seed, 'sentence': ' '.join(sentence), 'score': 0.1}

        # Score against neural field
        coherence = mind.score(sentence).get('coherence', 0)

        # Compare with stored episodes — does this match reality?
        episodes = mind.memory.recall(sentence, k=3)
        episode_match = 0.0
        if episodes:
            for ep in episodes:
                overlap = len(set(sentence) & set(ep.tokens))
                episode_match += overlap / max(len(sentence), 1)
            episode_match /= len(episodes)

        # Extract pattern from THIS sentence + matching episodes
        pattern_extracted = self._try_extract_pattern(sentence, episodes, mind)

        # Score = coherence × episode_match × pattern_bonus
        pattern_bonus = 0.2 if pattern_extracted else 0.0
        score = coherence * 0.5 + episode_match * 0.3 + pattern_bonus

        # PROCEDURALIZE: good sentences become motor programs
        # The cerebellum learns from successful practice
        if score > 0.3 and len(sentence) >= 4:
            trajectory = getattr(mind.speaker, '_last_trajectory', None)
            mind.speaker.proceduralize(
                tokens=sentence, seed=seed, goal='',
                trajectory=trajectory, score=score,
            )
            # Also reinforce transitions
            mind.speaker.reinforce(sentence, strength=score)

        return {
            'type': 'generate',
            'seed': seed,
            'sentence': ' '.join(sentence),
            'coherence': round(coherence, 3),
            'episode_match': round(episode_match, 3),
            'pattern': pattern_extracted,
            'score': round(score, 3),
        }

    def _exercise_transfer(self, mind) -> dict:
        """
        Pick a known pattern. Apply it to a NEW concept.
        Score how well the new sentence works.
        This tests real understanding, not memorization.
        """
        if not self.patterns:
            return self._exercise_generate_and_extract(mind)

        # Pick the strongest pattern
        pattern = max(self.patterns, key=lambda p: p.strength)

        # Find a concept NOT in the pattern's examples
        used_concepts = set()
        for ex in pattern.examples:
            used_concepts.update(ex.values())

        candidates = [w for w, f in mind.cortex.word_freq.items()
                      if f >= 2 and len(w) > 4 and w not in used_concepts]
        if not candidates:
            return self._exercise_generate_and_extract(mind)

        new_concept = candidates[self._practice_count % len(candidates)]

        # Fill the pattern with the new concept
        filled = pattern.fill({'X': new_concept, 'Y': '', 'Z': ''}).strip()
        tokens = filled.lower().split()

        # Generate the rest using Broca's from the concept
        generated = mind.generate([new_concept], max_length=8,
                                   goal=pattern.slots[0] if pattern.slots else None)

        # Score: does this make sense?
        coherence = mind.score(generated).get('coherence', 0)

        # If coherent, the pattern transferred — strengthen it
        if coherence > 0.3:
            pattern.strength += 0.1
            pattern.uses += 1
            pattern.examples.append({'X': new_concept, 'sentence': ' '.join(generated)})
            if len(pattern.examples) > 20:
                pattern.examples = pattern.examples[-10:]

            # Reinforce transitions AND proceduralize
            mind.speaker.reinforce(generated, strength=coherence)
            trajectory = getattr(mind.speaker, '_last_trajectory', None)
            mind.speaker.proceduralize(
                tokens=generated, seed=new_concept,
                goal=pattern.slots[0] if pattern.slots else '',
                trajectory=trajectory, score=coherence,
            )

        return {
            'type': 'transfer',
            'pattern': pattern.template,
            'concept': new_concept,
            'generated': ' '.join(generated),
            'coherence': round(coherence, 3),
            'transferred': coherence > 0.3,
            'score': round(coherence, 3),
        }

    def _try_extract_pattern(self, sentence: List[str], episodes: list, mind) -> Optional[str]:
        """
        Look for repeating structure between the generated sentence and episodes.
        If found, extract as a reusable pattern.

        "stroke is a disease" + "meningitis is an infection" → "X is a Y"
        "stroke causes disability" + "hypertension causes damage" → "X causes Y"
        """
        if not episodes or len(sentence) < 3:
            return None

        # Find words shared between sentence and episodes (structural words)
        sentence_set = set(sentence)
        for ep in episodes:
            ep_set = set(ep.tokens)
            shared = sentence_set & ep_set
            unique_sent = sentence_set - ep_set
            unique_ep = ep_set - sentence_set

            # Pattern = ONLY short structural words (len <= 3: is, a, the, of, by, to)
            # stay in template. Everything else becomes a slot.
            structural_shared = [w for w in shared if len(w) <= 3]
            if len(structural_shared) >= 1 and unique_sent and unique_ep:
                template_words = []
                slot_count = 0
                slots = []
                for w in sentence:
                    if len(w) <= 3 and w in shared:
                        template_words.append(w)
                    elif slot_count == 0:
                        template_words.append('X')
                        slot_count += 1
                        slots.append('subject')
                    elif slot_count == 1 and template_words[-1] != 'Y':
                        template_words.append('Y')
                        slot_count += 1
                        slots.append('object')
                    # Content words after Y are dropped — pattern stays short

                template = ' '.join(template_words)
                # Pattern must have at least 1 slot and 1 structural word
                if slot_count < 1 or len(structural_shared) < 1:
                    continue

                # Check if this pattern already exists
                for existing in self.patterns:
                    if existing.template == template:
                        existing.strength += 0.1
                        existing.examples.append({
                            'X': list(unique_sent)[0] if unique_sent else '',
                            'Y': list(unique_ep)[0] if unique_ep else '',
                        })
                        return template

                # New pattern
                if len(template_words) >= 3 and slot_count >= 1:
                    p = Pattern(template, slots)
                    p.strength = 0.2
                    p.examples.append({
                        'X': list(unique_sent)[0] if unique_sent else '',
                        'Y': list(unique_ep)[0] if unique_ep else '',
                    })
                    self.patterns.append(p)
                    # Cap patterns to prevent bloat
                    if len(self.patterns) > 200:
                        self.patterns.sort(key=lambda p: -p.strength)
                        self.patterns = self.patterns[:100]
                    return template

        return None

    def stats(self) -> dict:
        return {
            'level': self.level,
            'auc': dict(self._auc),
            'practice_count': self._practice_count,
            'patterns': len(self.patterns),
            'top_patterns': [p.to_dict() for p in sorted(self.patterns, key=lambda p: -p.strength)[:5]],
            'recent': self.history[-5:],
        }

    def to_dict(self) -> dict:
        return {
            'level': self.level,
            'auc': self._auc,
            'practice_count': self._practice_count,
            'history': self.history[-50:],
            'patterns': [p.to_dict() for p in self.patterns],
        }

    @classmethod
    def from_dict(cls, d: dict) -> Curriculum:
        c = cls()
        c.level = d.get('level', 1)
        c._auc = d.get('auc', {1: 0, 2: 0, 3: 0, 4: 0, 5: 0})
        c._practice_count = d.get('practice_count', 0)
        c.history = d.get('history', [])
        c.patterns = [Pattern.from_dict(p) for p in d.get('patterns', [])]
        return c
