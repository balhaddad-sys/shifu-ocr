"""
Syntax — learn sentence structure from data.

The baby discovers:
  - FRAMES: "X is a Y that Z" → [NOUN, copula, article, NOUN, relative, VERB]
  - PATTERNS: subject-verb-object ordering
  - TRANSITIONS: after a noun, verbs are likely; after "the", nouns follow
  - CLAUSE structure: main clause + subordinate clause

All learned from observed sentences. No grammar rules hardcoded.
The corpus IS the grammar.
"""

from __future__ import annotations
from typing import Dict, List, Set, Optional, Tuple
from collections import defaultdict


class Syntax:
    """
    Learn sentence structure from observed data.

    feed(tokens) — learn from one sentence
    predict_next(context) — what word class comes next?
    score_structure(tokens) — how grammatical is this?
    """

    def __init__(self):
        # POS tagging by position and context (emergent, not hardcoded)
        # word → {position_profile: {first: count, middle: count, last: count}}
        self._position: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

        # Transition patterns: what follows what?
        # (word_class, word_class) → count
        # word_class = "short" (len<=3), "medium" (4-6), "long" (7+)
        # This is a proxy for POS without hardcoded tags
        self._transitions: Dict[Tuple[str, str], int] = defaultdict(int)

        # Frame learning: sentence skeleton patterns
        # Frame = sequence of word-class tokens, e.g., "S L S M L" (short long short medium long)
        self._frames: Dict[str, int] = defaultdict(int)

        # Opener patterns: what words start sentences?
        self._openers: Dict[str, int] = defaultdict(int)

        # Closer patterns: what words end sentences?
        self._closers: Dict[str, int] = defaultdict(int)

        # Clause markers: words that separate clauses
        # (discovered from position + frequency)
        self._clause_markers: Dict[str, float] = {}

        self._sentence_count = 0

    def _word_class(self, word: str) -> str:
        """Classify a word by length — proxy for POS without hardcoded rules."""
        n = len(word)
        if n <= 2:
            return 'S'   # Short: articles, prepositions (a, in, of, is)
        elif n <= 4:
            return 'M'   # Medium: verbs, adjectives (with, from, that, this)
        elif n <= 7:
            return 'L'   # Long: nouns, specific terms (stroke, brain, causes)
        else:
            return 'X'   # Extra long: technical terms (thrombolysis, cerebral)

    def feed(self, tokens: List[str]) -> None:
        """Learn syntax from one sentence."""
        if len(tokens) < 3:
            return
        self._sentence_count += 1

        # Position profiling
        for i, token in enumerate(tokens):
            if i == 0:
                self._position[token]['first'] += 1
                self._openers[token] += 1
            elif i == len(tokens) - 1:
                self._position[token]['last'] += 1
                self._closers[token] += 1
            else:
                self._position[token]['middle'] += 1

        # Transition patterns
        for i in range(len(tokens) - 1):
            cls_a = self._word_class(tokens[i])
            cls_b = self._word_class(tokens[i + 1])
            self._transitions[(cls_a, cls_b)] += 1

        # Frame extraction
        frame = ' '.join(self._word_class(t) for t in tokens)
        self._frames[frame] += 1

    def predict_next_class(self, current_class: str) -> Dict[str, float]:
        """What word class is likely after this class?"""
        total = 0
        counts: Dict[str, int] = defaultdict(int)
        for (cls_a, cls_b), count in self._transitions.items():
            if cls_a == current_class:
                counts[cls_b] += count
                total += count
        if total == 0:
            return {}
        return {cls: count / total for cls, count in counts.items()}

    def score_structure(self, tokens: List[str]) -> float:
        """
        How grammatical is this sentence?
        Score = average transition probability.
        High score = common structure. Low = unusual.
        """
        if len(tokens) < 2:
            return 0.0

        # Check opener
        opener_score = 1.0 if tokens[0] in self._openers else 0.5

        # Transition score
        transition_scores = []
        for i in range(len(tokens) - 1):
            cls_a = self._word_class(tokens[i])
            cls_b = self._word_class(tokens[i + 1])
            # How common is this transition?
            this_count = self._transitions.get((cls_a, cls_b), 0)
            total_from_a = sum(
                c for (a, _), c in self._transitions.items() if a == cls_a
            )
            if total_from_a > 0:
                transition_scores.append(this_count / total_from_a)
            else:
                transition_scores.append(0.0)

        avg_transition = sum(transition_scores) / len(transition_scores) if transition_scores else 0.0
        return opener_score * 0.2 + avg_transition * 0.8

    def detect_clause_markers(self, word_freqs: Dict[str, int]) -> Dict[str, float]:
        """
        Discover clause markers: words that appear frequently in
        the middle of sentences but rarely at start/end.
        """
        markers = {}
        for word, pos in self._position.items():
            total = pos.get('first', 0) + pos.get('middle', 0) + pos.get('last', 0)
            if total < 5:
                continue
            middle_ratio = pos.get('middle', 0) / total
            edge_ratio = (pos.get('first', 0) + pos.get('last', 0)) / total
            # Clause markers: mostly in middle, rarely at edges
            if middle_ratio > 0.7 and edge_ratio < 0.3 and len(word) <= 5:
                markers[word] = round(middle_ratio, 3)
        self._clause_markers = markers
        return markers

    def top_frames(self, n: int = 10) -> List[Tuple[str, int]]:
        """Most common sentence structures."""
        return sorted(self._frames.items(), key=lambda x: -x[1])[:n]

    def stats(self) -> dict:
        return {
            'sentences_analyzed': self._sentence_count,
            'unique_frames': len(self._frames),
            'transition_patterns': len(self._transitions),
            'openers': len(self._openers),
            'clause_markers': len(self._clause_markers),
            'top_frames': self.top_frames(5),
            'top_openers': sorted(self._openers.items(), key=lambda x: -x[1])[:5],
        }

    def to_dict(self) -> dict:
        transitions = {f"{a},{b}": c for (a, b), c in self._transitions.items()}
        return {
            'frames': dict(sorted(self._frames.items(), key=lambda x: -x[1])[:200]),
            'transitions': transitions,
            'openers': dict(sorted(self._openers.items(), key=lambda x: -x[1])[:100]),
            'closers': dict(sorted(self._closers.items(), key=lambda x: -x[1])[:100]),
            'clause_markers': self._clause_markers,
            'sentence_count': self._sentence_count,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Syntax:
        s = cls()
        s._frames = defaultdict(int, d.get('frames', {}))
        for key, count in d.get('transitions', {}).items():
            parts = key.split(',')
            if len(parts) == 2:
                s._transitions[(parts[0], parts[1])] = count
        s._openers = defaultdict(int, d.get('openers', {}))
        s._closers = defaultdict(int, d.get('closers', {}))
        s._clause_markers = d.get('clause_markers', {})
        s._sentence_count = d.get('sentence_count', 0)
        return s
