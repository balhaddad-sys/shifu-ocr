#!/usr/bin/env python3
"""
CEREBRAL CORTEX — the mind that answers.

Low latency. Stable. Stays under 100MB.
Handles: deliberate, describe, generate, activate, score, confidence, connect,
         stats, compass, introspect, decompose, synonyms, semantic.
Reads shared state from disk. Reloads when epoch changes.
"""

import json, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shifu_ocr.mind import ShifuMind
from shifu_ocr.mind.nervous_system import import_graphs, import_cortex, read_epoch

mind = ShifuMind(initial_layers=['identity', 'appearance', 'function', 'mechanism', 'relation'])
_last_epoch = 0

def reload_if_needed():
    """Reload shared state if feed worker has new data."""
    global _last_epoch
    epoch = read_epoch()
    if epoch > _last_epoch:
        import_graphs(mind)
        import_cortex(mind)
        mind.cortex._invalidate_cache()
        mind.field.invalidate_cache()
        mind.field.update_medians(mind.cortex.word_freq, mind._co_graph)
        _last_epoch = epoch

# Initial load
import_graphs(mind)
import_cortex(mind)

sys.stdout.write(json.dumps({'ok': True, 'status': 'ready', 'role': 'query', 'vocabulary': len(mind.cortex.word_freq)}) + '\n')
sys.stdout.flush()

import signal as _signal
_signal.signal(_signal.SIGINT, lambda *a: None)

for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    try:
        cmd = json.loads(line)
        req_id = cmd.pop('_id', None)
        op = cmd.get('cmd', '')

        # Lazy reload — check if feed worker wrote new data
        reload_if_needed()

        if op == 'ping':
            result = {'ok': True, 'status': 'alive', 'role': 'query'}
        elif op == 'deliberate':
            result = {'ok': True, **mind.deliberate(cmd.get('query', ''))}
            result['retrieved'] = [{'word': x['word'], 'energy': round(x['energy'], 4)} for x in result.get('retrieved', [])]
        elif op == 'describe':
            result = {'ok': True, 'description': mind.describe(cmd.get('word', ''))}
        elif op == 'generate':
            tokens = mind.generate([cmd.get('word', '')], max_length=cmd.get('max_length', 15))
            result = {'ok': True, 'text': ' '.join(tokens)}
        elif op == 'activate':
            field = mind.activate(cmd.get('word', ''))
            top = sorted(field.items(), key=lambda x: -x[1])[:20]
            result = {'ok': True, 'field': [{'word': w, 'energy': round(e, 4)} for w, e in top]}
        elif op == 'score':
            r = mind.score_text(cmd.get('text', ''))
            result = {'ok': True, 'coherence': r['coherence'], 'scores': r['scores']}
        elif op == 'confidence':
            result = {'ok': True, **mind.confidence(cmd.get('word', ''))}
        elif op == 'connect':
            path = mind.speaker.find_path(cmd.get('from', ''), cmd.get('to', ''), mind._co_graph)
            result = {'ok': True, 'path': path, 'connected': path is not None}
        elif op == 'candidates':
            cands = cmd.get('candidates', [])
            ctx = cmd.get('context', [])
            result = {'ok': True, 'ranked': mind.predict_candidates([(c[0], c[1]) for c in cands], context_words=ctx)}
        elif op == 'compass':
            result = {'ok': True, **mind.compass()}
        elif op == 'introspect':
            result = {'ok': True, 'voice': mind.introspect()}
        elif op == 'hungry':
            result = {'ok': True, 'gaps': mind.hungry()}
        elif op == 'stats':
            result = {'ok': True, **mind.stats()}
        elif op == 'decompose':
            result = {'ok': True, **mind.language.morphology.decompose(cmd.get('word', ''))}
        elif op == 'synonyms':
            result = {'ok': True, 'synonyms': mind.language.semantics.synonyms(cmd.get('word', ''))}
        elif op == 'explain_semantic':
            result = {'ok': True, 'explanation': mind.language.semantics.explain(cmd.get('word', ''))}
        elif op == 'language_stats':
            result = {'ok': True, 'morphology': mind.language.morphology.stats(), 'syntax': mind.language.syntax.stats(), 'semantics': mind.language.semantics.stats(), 'curriculum': mind.language.curriculum.stats()}
        else:
            result = {'ok': False, 'error': f'query worker: unknown cmd {op}'}
    except Exception as e:
        req_id = None
        result = {'ok': False, 'error': str(e)}

    if req_id is not None:
        result['_id'] = req_id
    try:
        sys.stdout.write(json.dumps(result, ensure_ascii=False) + '\n')
        sys.stdout.flush()
    except (BrokenPipeError, OSError):
        break
