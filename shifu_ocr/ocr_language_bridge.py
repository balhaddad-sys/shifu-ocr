"""
Shifu-OCR: Language Engine Bridge
===================================
Mod 10: Connect OCR to Shifu semantic language engine.
Use scoreSentence coherence to validate and re-rank OCR word candidates.

The OCR reads characters. The language engine knows if the resulting
words make sense together. This bridge connects them.
"""

import numpy as np


class SemanticReranker:
    """Bridge between Shifu OCR and Shifu Language Engine."""

    def __init__(self, language_engine, coherence_threshold=0.3):
        self.engine = language_engine
        self.threshold = coherence_threshold

    def score_text(self, text):
        result = self.engine.scoreSentence(text)
        return {
            'text': text,
            'forward': result.get('coherence', 0),
            'corrected': result.get('correctedCoherence', 0),
            'settled': result.get('settledCoherence', 0),
            'needs_review': result.get('settledCoherence', 0) < self.threshold,
        }

    def rerank_word(self, words, position, candidates, min_improvement=0.02):
        if not candidates or position >= len(words):
            return None
        original_word = words[position]
        baseline_text = ' '.join(words)
        baseline = self.engine.scoreSentence(baseline_text)
        baseline_coh = baseline.get('correctedCoherence', 0)
        scores = []
        for candidate, dist in candidates:
            if candidate == original_word:
                scores.append({'word': candidate, 'distance': dist, 'coherence': baseline_coh, 'gain': 0})
                continue
            test_words = words[:position] + [candidate] + words[position + 1:]
            test_result = self.engine.scoreSentence(' '.join(test_words))
            test_coh = test_result.get('correctedCoherence', 0)
            scores.append({'word': candidate, 'distance': dist, 'coherence': test_coh, 'gain': test_coh - baseline_coh})
        scores.sort(key=lambda x: x['coherence'], reverse=True)
        best = scores[0]
        return {
            'original': original_word,
            'best': best['word'],
            'accepted': best['gain'] >= min_improvement or best['word'] == original_word,
            'coherence_gain': best['gain'],
            'scores': scores,
        }

    def rerank_line(self, ocr_result, word_candidates=None):
        text = ocr_result.get('text', '')
        words = text.split()
        if not words:
            return {'text': '', 'coherence': 0, 'corrections': []}
        original_score = self.score_text(text)
        if not word_candidates:
            return {'text': text, 'words': words, 'coherence': original_score,
                    'corrections': [], 'needs_review': original_score['needs_review']}
        corrections = []
        corrected_words = list(words)
        for pos, candidates in sorted(word_candidates.items()):
            if pos >= len(corrected_words):
                continue
            result = self.rerank_word(corrected_words, pos, candidates)
            if result and result['accepted'] and result['best'] != result['original']:
                corrected_words[pos] = result['best']
                corrections.append({'position': pos, 'original': result['original'],
                                    'corrected': result['best'], 'coherence_gain': result['coherence_gain']})
        corrected_text = ' '.join(corrected_words)
        final_score = self.score_text(corrected_text)
        return {'text': corrected_text, 'original_text': text, 'words': corrected_words,
                'coherence': final_score, 'original_coherence': original_score,
                'corrections': corrections, 'needs_review': final_score['needs_review'],
                'total_gain': final_score['settled'] - original_score['settled']}

    def validate_corrections(self, original_text, corrected_text):
        orig = self.score_text(original_text)
        corr = self.score_text(corrected_text)
        gain = corr['settled'] - orig['settled']
        if gain > 0.02:
            recommendation = 'accept'
        elif gain < -0.05:
            recommendation = 'reject'
        else:
            recommendation = 'review'
        return {'original': orig, 'corrected': corr, 'gain': gain, 'recommendation': recommendation}

    def batch_score_lines(self, lines):
        scored = []
        for i, line in enumerate(lines):
            score = self.score_text(line)
            score['line_index'] = i
            scored.append(score)
        scored.sort(key=lambda x: x['settled'])
        return scored
