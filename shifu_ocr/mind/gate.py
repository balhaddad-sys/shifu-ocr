"""
Gate — input filtering with emergent stop-word detection.

The S2 equivalent. Cleans, tokenizes, and decides whether input
is worth absorbing. No hardcoded word lists. Stop words emerge
from frequency. Quality thresholds adapt with corpus maturity.
"""

from __future__ import annotations
import re
import math
from collections import Counter
from typing import Dict, List, Set, Optional, Tuple

_TOKEN_RE = re.compile(r'[a-z][a-z0-9-]*')
_URL_RE = re.compile(r'https?://\S+')
_BRACKET_RE = re.compile(r'\[[^\]]{0,100}\]')
_PAREN_LONG_RE = re.compile(r'\([^)]{40,}\)')
_PAREN_NUM_RE = re.compile(r'\(\d[^)]*\)')
_NUMBER_RE = re.compile(r'\b\d{3,}\b')
_BULLET_RE = re.compile(r'^\s*[-\u2022*]\s')
_NUMBERED_RE = re.compile(r'^\s*\d+[\s.)]+')
_POSSESSIVE_RE = re.compile(r"['\u2019]s\b")


class Gate:
    """
    Input filtering with emergent stop-word detection.

    No hardcoded stop words. Words that appear in too many contexts
    (top fraction by frequency) are identified as function words
    and filtered from content processing.

    Quality thresholds adapt based on observed distribution.
    """

    def __init__(
        self,
        reject_threshold: float = 0.02,
        low_threshold: float = 0.05,
        stop_fraction: float = 0.005,
        min_content_words: int = 2,
        max_tokens: int = 50,
    ):
        self._reject_threshold = reject_threshold
        self._low_threshold = low_threshold
        self._stop_fraction = stop_fraction
        self._min_content = min_content_words
        self._max_tokens = max_tokens
        self._quality_history: List[float] = []
        self._total_filtered = 0
        self._total_accepted = 0

    # ═══ STOP WORD DETECTION ═══

    def stop_words(self, word_freqs: Dict[str, float],
                   top_fraction: Optional[float] = None) -> Set[str]:
        """
        Emergent stop words: the most frequent words in the corpus.
        These are function words — they connect content but carry
        little meaning on their own.

        No hardcoded list. The corpus tells us what's common.
        """
        frac = top_fraction if top_fraction is not None else self._stop_fraction
        if not word_freqs:
            return set()
        sorted_words = sorted(word_freqs.items(), key=lambda x: -x[1])
        n = max(1, int(len(sorted_words) * frac))
        # Also include very short words (length <= 2) as likely function words
        stops = set()
        for w, _ in sorted_words[:n]:
            stops.add(w)
        for w in word_freqs:
            if len(w) <= 2:
                stops.add(w)
        return stops

    # ═══ CLEANING ═══

    def clean(self, text: str, known_vocab: Optional[Set[str]] = None) -> str:
        """
        Strip noise, normalize to what the cortex can process.
        Optionally normalizes word forms to known vocabulary.
        """
        s = text
        s = _URL_RE.sub('', s)
        s = _BRACKET_RE.sub('', s)
        s = _PAREN_LONG_RE.sub('', s)
        s = _PAREN_NUM_RE.sub('', s)
        s = _NUMBER_RE.sub('', s)
        s = _BULLET_RE.sub('', s)
        s = _NUMBERED_RE.sub('', s)
        s = _POSSESSIVE_RE.sub('', s)

        # Normalize to known vocabulary if available
        if known_vocab:
            words = s.split()
            for i, word in enumerate(words):
                w = word.lower().strip('.,;:!?()[]{}"\'-')
                if len(w) <= 2:
                    continue
                if w in known_vocab:
                    continue
                # Try common stemming
                for suffix, replacement in [
                    ('ed', ''), ('ing', ''), ('tion', 'te'),
                    ('ly', ''), ('ies', 'y'), ('es', ''), ('s', ''),
                ]:
                    if w.endswith(suffix) and len(w) > len(suffix) + 2:
                        stem = w[:-len(suffix)] + replacement
                        if stem in known_vocab:
                            words[i] = word.replace(w, stem)
                            break
            s = ' '.join(words)

        return re.sub(r'\s+', ' ', s).strip()

    # ═══ TOKENIZATION ═══

    def tokenize(self, text: str) -> List[str]:
        """Extract lowercase alphabetic tokens. Filter noise emergently:
        - Words with no vowels are likely abbreviations (xvii, bnf, ct)
        - Words where one letter is >60% of the word are likely noise (xxxx, aaaa)
        No hardcoded lists — just letter distribution."""
        raw = _TOKEN_RE.findall(text.lower())
        vowels = set('aeiouy')
        result = []
        for w in raw:
            if len(w) <= 2:
                result.append(w)  # Short words pass (is, a, the)
                continue
            # Vowel ratio: real English words have >= 20% vowels
            # "stroke" = 2/6 = 33%. "xvii" = 1/4 = 25%. "bnf" = 0%.
            # "osahs" = 2/5 = 40%. Tighten to 25% to catch "xvii" class.
            vowel_count = sum(1 for c in w if c in vowels)
            if vowel_count / len(w) < 0.25:
                continue
            # No single letter should be >50% of the word
            counts = Counter(w)
            if counts.most_common(1)[0][1] / len(w) > 0.5:
                continue
            result.append(w)
        return result

    # ═══ FILTERING ═══

    def filter(self, text: str, word_freqs: Optional[Dict[str, float]] = None,
               known_vocab: Optional[Set[str]] = None) -> dict:
        """
        Assess input quality and decide whether to accept.

        Returns: {
            'accepted': bool,
            'cleaned': str,
            'tokens': List[str],
            'content_tokens': List[str],
            'quality': float,
            'reason': str
        }
        """
        cleaned = self.clean(text, known_vocab)
        tokens = self.tokenize(cleaned)

        if len(tokens) < self._min_content:
            self._total_filtered += 1
            return {
                'accepted': False, 'cleaned': cleaned,
                'tokens': tokens, 'content_tokens': [],
                'quality': 0.0, 'reason': 'too_short',
            }

        if len(tokens) > self._max_tokens:
            tokens = tokens[:self._max_tokens]
            cleaned = ' '.join(tokens)

        # Filter stop words
        stops = self.stop_words(word_freqs or {})
        content = [t for t in tokens if t not in stops and len(t) > 2]

        if len(content) < self._min_content:
            self._total_filtered += 1
            return {
                'accepted': False, 'cleaned': cleaned,
                'tokens': tokens, 'content_tokens': content,
                'quality': 0.0, 'reason': 'no_content',
            }

        # Quality assessment
        unique_ratio = len(set(content)) / max(len(content), 1)
        avg_len = sum(len(w) for w in content) / max(len(content), 1)
        quality = unique_ratio * 0.5 + min(avg_len / 8.0, 1.0) * 0.5

        # Track quality history for threshold adaptation
        self._quality_history.append(quality)
        if len(self._quality_history) > 1000:
            self._quality_history = self._quality_history[-500:]

        if quality < self._reject_threshold:
            self._total_filtered += 1
            return {
                'accepted': False, 'cleaned': cleaned,
                'tokens': tokens, 'content_tokens': content,
                'quality': quality, 'reason': 'low_quality',
            }

        self._total_accepted += 1
        return {
            'accepted': True, 'cleaned': cleaned,
            'tokens': tokens, 'content_tokens': content,
            'quality': quality, 'reason': 'accepted',
        }

    # ═══ THRESHOLD ADAPTATION ═══

    def adapt_thresholds(self) -> None:
        """
        Adjust thresholds based on observed quality distribution.
        Reject threshold → 5th percentile. Low threshold → 25th percentile.
        """
        if len(self._quality_history) < 50:
            return
        sorted_q = sorted(self._quality_history)
        n = len(sorted_q)
        self._reject_threshold = sorted_q[int(n * 0.05)]
        self._low_threshold = sorted_q[int(n * 0.25)]

    # ═══ STATS ═══

    def stats(self) -> dict:
        return {
            'accepted': self._total_accepted,
            'filtered': self._total_filtered,
            'reject_threshold': round(self._reject_threshold, 4),
            'low_threshold': round(self._low_threshold, 4),
        }

    def to_dict(self) -> dict:
        return {
            'reject_threshold': self._reject_threshold,
            'low_threshold': self._low_threshold,
            'stop_fraction': self._stop_fraction,
            'quality_history': self._quality_history[-200:],
            'total_filtered': self._total_filtered,
            'total_accepted': self._total_accepted,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Gate:
        g = cls(
            reject_threshold=d.get('reject_threshold', 0.02),
            low_threshold=d.get('low_threshold', 0.05),
            stop_fraction=d.get('stop_fraction', 0.005),
        )
        g._quality_history = d.get('quality_history', [])
        g._total_filtered = d.get('total_filtered', 0)
        g._total_accepted = d.get('total_accepted', 0)
        return g
