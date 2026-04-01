"""
Memory — episodic memory system.

The hippocampus equivalent. Records significant moments,
retrieves relevant episodes, detects topic shifts.

Capacity-bounded with significance-based eviction:
the least significant episode is evicted first, not the oldest.
"""

from __future__ import annotations
from typing import Dict, List, Set, Optional, Any
from dataclasses import dataclass, field


@dataclass
class Episode:
    """A single recorded moment."""
    epoch: int
    tokens: List[str]
    significance: float
    context: dict = field(default_factory=dict)
    timestamp: float = 0.0

    def to_dict(self) -> dict:
        return {
            'epoch': self.epoch, 'tokens': self.tokens,
            'significance': self.significance,
            'context': self.context, 'ts': self.timestamp,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Episode:
        return cls(
            epoch=d['epoch'], tokens=d['tokens'],
            significance=d.get('significance', 0.0),
            context=d.get('context', {}),
            timestamp=d.get('ts', 0.0),
        )


class Memory:
    """
    Episodic memory with significance-based eviction.

    Records episodes when significance exceeds threshold.
    Retrieves by token overlap weighted by significance.
    Detects topic shifts from sliding window of recent focus.
    """

    def __init__(
        self,
        capacity: int = 1000,
        significance_threshold: float = 0.3,
        topic_window: int = 5,
    ):
        self._capacity = capacity
        self._sig_threshold = significance_threshold
        self._topic_window = topic_window
        self.episodes: List[Episode] = []
        self._active_topic: Optional[str] = None
        self._topic_strength: int = 0
        self._recent_focus: List[List[str]] = []

    def record(self, epoch: int, tokens: List[str],
               significance: float, context: Optional[dict] = None,
               timestamp: float = 0.0) -> Optional[Episode]:
        """
        Record an episode if significant enough.
        Evicts least significant when at capacity.
        """
        if significance < self._sig_threshold:
            return None

        ep = Episode(
            epoch=epoch, tokens=list(tokens),
            significance=significance,
            context=context or {},
            timestamp=timestamp,
        )
        self.episodes.append(ep)

        # Update topic tracking
        self._update_topic(tokens)

        # Evict if over capacity
        if len(self.episodes) > self._capacity:
            # Remove least significant
            min_idx = 0
            min_sig = self.episodes[0].significance
            for i, e in enumerate(self.episodes):
                if e.significance < min_sig:
                    min_sig = e.significance
                    min_idx = i
            self.episodes.pop(min_idx)

        return ep

    def recall(self, query_tokens: List[str], k: int = 5) -> List[Episode]:
        """
        Retrieve episodes most relevant to query.
        Scored by token overlap weighted by significance.
        """
        if not self.episodes or not query_tokens:
            return []

        query_set = set(query_tokens)
        scored = []
        for ep in self.episodes:
            ep_set = set(ep.tokens)
            overlap = len(query_set & ep_set)
            if overlap == 0:
                continue
            score = overlap * ep.significance
            scored.append((score, ep))

        scored.sort(key=lambda x: -x[0])
        return [ep for _, ep in scored[:k]]

    def _update_topic(self, tokens: List[str]) -> None:
        """Track active topic from recent episodes."""
        self._recent_focus.append(tokens)
        if len(self._recent_focus) > self._topic_window:
            self._recent_focus = self._recent_focus[-self._topic_window:]

        if not tokens:
            return

        # Find most frequent word across recent focus
        freq: Dict[str, int] = {}
        for focus in self._recent_focus:
            for w in focus:
                freq[w] = freq.get(w, 0) + 1

        if freq:
            top = max(freq, key=freq.get)
            if top == self._active_topic:
                self._topic_strength += 1
            else:
                self._active_topic = top
                self._topic_strength = 1

    def detect_topic_shift(self, current_tokens: List[str]) -> float:
        """
        How much has the topic shifted?
        0.0 = no shift, 1.0 = complete shift.
        """
        if not self._recent_focus or not current_tokens:
            return 1.0

        current_set = set(current_tokens)
        recent_words: Set[str] = set()
        for focus in self._recent_focus:
            recent_words.update(focus)

        if not recent_words:
            return 1.0

        overlap = len(current_set & recent_words)
        total = len(current_set | recent_words)
        continuity = overlap / max(total, 1)
        return 1.0 - continuity

    def get_context(self) -> dict:
        """Current episodic context for the thinker."""
        return {
            'active_topic': self._active_topic,
            'topic_strength': self._topic_strength,
            'last_focus': self._recent_focus[-1] if self._recent_focus else [],
            'turn_count': len(self.episodes),
        }

    # ═══ SERIALIZATION ═══

    def to_dict(self) -> dict:
        return {
            'capacity': self._capacity,
            'sig_threshold': self._sig_threshold,
            'topic_window': self._topic_window,
            'episodes': [e.to_dict() for e in self.episodes[-self._capacity:]],
            'active_topic': self._active_topic,
            'topic_strength': self._topic_strength,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Memory:
        m = cls(
            capacity=d.get('capacity', 1000),
            significance_threshold=d.get('sig_threshold', 0.3),
            topic_window=d.get('topic_window', 5),
        )
        m.episodes = [Episode.from_dict(e) for e in d.get('episodes', [])]
        m._active_topic = d.get('active_topic')
        m._topic_strength = d.get('topic_strength', 0)
        return m
