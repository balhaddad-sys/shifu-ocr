#!/usr/bin/env python3
"""
Shifu Mind Worker — stdin/stdout JSON-RPC subprocess.

No background threads. No locks. No GIL contention.
The baby learns when FED. It practices when ASKED.
Like a real baby: it learns from experience, not alone.
"""

import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shifu_ocr.mind import ShifuMind

SAVE_PATH = os.path.join('.state', 'mind_brain.json')
mind = None
feed_count = 0


def boot():
    global mind
    if os.path.exists(SAVE_PATH):
        try:
            mind = ShifuMind.load(SAVE_PATH)
            return
        except Exception:
            pass
    mind = ShifuMind(
        initial_layers=['identity', 'appearance', 'function', 'mechanism', 'relation'],
    )
    seed = [
        'Stroke is a disease caused by arterial occlusion in the brain',
        'Dopamine is a neurotransmitter affected in parkinsons disease',
        'Thrombolysis is a treatment that dissolves blood clots',
        'The cerebral artery supplies blood to the cortex',
        'Occlusion of the middle cerebral artery causes hemiplegia',
        'Parkinsons disease causes tremor rigidity and bradykinesia',
        'Levodopa crosses the blood brain barrier to treat symptoms',
        'Hemorrhagic stroke results from rupture of cerebral blood vessels',
        'Hypertension is a major risk factor for cerebral stroke',
        'Anticoagulant therapy prevents recurrent thromboembolism',
        'The basal ganglia regulate voluntary motor movements',
        'Atrial fibrillation increases the risk of embolic stroke',
        'Hemorrhagic stroke results from rupture of cerebral blood vessels',
    ]
    for t in seed:
        mind.feed(t)


def save():
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    mind.save(SAVE_PATH)


def handle(cmd):
    global feed_count
    op = cmd.get('cmd', '')

    if op == 'ping':
        return {'ok': True, 'status': 'alive'}
    elif op == 'feed':
        r = mind.feed(cmd.get('text', ''))
        feed_count += 1
        if feed_count % 50 == 0:
            try: save()
            except: pass
        return {'ok': True, **r}
    elif op == 'feed_batch':
        r = mind.feed_batch(cmd.get('texts', []), cycles=cmd.get('cycles', 1))
        return {'ok': True, **r}
    elif op == 'score':
        r = mind.score_text(cmd.get('text', ''))
        return {'ok': True, 'coherence': r['coherence'], 'scores': r['scores']}
    elif op == 'candidates':
        cands = cmd.get('candidates', [])
        ctx = cmd.get('context', [])
        tuples = [(c[0], c[1]) for c in cands]
        ranked = mind.predict_candidates(tuples, context_words=ctx)
        return {'ok': True, 'ranked': ranked}
    elif op == 'deliberate':
        r = mind.deliberate(cmd.get('query', ''))
        r['retrieved'] = [
            {'word': x['word'], 'energy': round(x['energy'], 4)}
            for x in r.get('retrieved', [])
        ]
        return {'ok': True, **r}
    elif op == 'describe':
        return {'ok': True, 'description': mind.describe(cmd.get('word', ''))}
    elif op == 'generate':
        tokens = mind.generate([cmd.get('word', '')], max_length=cmd.get('max_length', 15))
        return {'ok': True, 'text': ' '.join(tokens)}
    elif op == 'activate':
        field = mind.activate(cmd.get('word', ''))
        top = sorted(field.items(), key=lambda x: -x[1])[:20]
        return {'ok': True, 'field': [{'word': w, 'energy': round(e, 4)} for w, e in top]}
    elif op == 'connect':
        path = mind.speaker.find_path(cmd.get('from', ''), cmd.get('to', ''), mind._co_graph)
        return {'ok': True, 'path': path, 'connected': path is not None}
    elif op == 'stats':
        return {'ok': True, **mind.stats()}
    elif op == 'confidence':
        return {'ok': True, **mind.confidence(cmd.get('word', ''))}
    elif op == 'hungry':
        return {'ok': True, 'gaps': mind.hungry()}
    elif op == 'consolidate':
        return {'ok': True, **mind.consolidate(focus_size=cmd.get('focus_size'))}
    elif op == 'practice':
        return {'ok': True, **mind.practice(rounds=cmd.get('rounds', 10))}
    elif op == 'study':
        return {'ok': True, **mind.study(rounds=cmd.get('rounds', 5), level=cmd.get('level'))}
    elif op == 'assess':
        return {'ok': True, **mind.assess_language()}
    elif op == 'decompose':
        return {'ok': True, **mind.language.morphology.decompose(cmd.get('word', ''))}
    elif op == 'synonyms':
        return {'ok': True, 'synonyms': mind.language.semantics.synonyms(cmd.get('word', ''))}
    elif op == 'explain_semantic':
        return {'ok': True, 'explanation': mind.language.semantics.explain(cmd.get('word', ''))}
    elif op == 'language_stats':
        return {'ok': True,
                'morphology': mind.language.morphology.stats(),
                'syntax': mind.language.syntax.stats(),
                'semantics': mind.language.semantics.stats(),
                'curriculum': mind.language.curriculum.stats()}
    elif op == 'heartbeat':
        return {'ok': True, **mind.heartbeat()}
    elif op == 'compass':
        return {'ok': True, **mind.compass()}
    elif op == 'introspect':
        return {'ok': True, 'voice': mind.introspect()}
    elif op == 'save':
        save()
        return {'ok': True}
    else:
        return {'ok': False, 'error': f'unknown command: {op}'}


if __name__ == '__main__':
    boot()
    sys.stdout.write(json.dumps({'ok': True, 'status': 'ready', 'vocabulary': mind.stats()['vocabulary']}) + '\n')
    sys.stdout.flush()

    import signal as _signal
    _signal.signal(_signal.SIGINT, lambda *a: None)
    try:
        _signal.signal(_signal.SIGPIPE, lambda *a: None)
    except AttributeError:
        pass

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            cmd = json.loads(line)
            req_id = cmd.pop('_id', None)
            result = handle(cmd)
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
