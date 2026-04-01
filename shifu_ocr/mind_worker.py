#!/usr/bin/env python3
"""
Shifu Mind Worker — stdin/stdout JSON-RPC subprocess.

The JS server spawns this process and communicates via line-delimited JSON.
Each line in is a command, each line out is a response.

Commands:
  {"cmd":"feed","text":"..."}
  {"cmd":"feed_batch","texts":["...",...]}
  {"cmd":"score","text":"..."}
  {"cmd":"candidates","candidates":[["word",conf],...],"context":["..."]}
  {"cmd":"deliberate","query":"..."}
  {"cmd":"describe","word":"..."}
  {"cmd":"generate","word":"...","max_length":15}
  {"cmd":"activate","word":"..."}
  {"cmd":"connect","from":"...","to":"..."}
  {"cmd":"stats"}
  {"cmd":"confidence","word":"..."}
  {"cmd":"hungry"}
  {"cmd":"save"}
  {"cmd":"ping"}
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
    # Seed corpus
    seed = [
        'Stroke is caused by arterial occlusion in the brain',
        'The cerebral artery supplies blood to the cortex',
        'Occlusion of the middle cerebral artery causes hemiplegia',
        'Treatment involves thrombolysis with tissue plasminogen activator',
        'Brain imaging reveals the extent of ischemic damage',
        'The patient presents with sudden onset weakness and paralysis',
        'Dopamine pathways are affected in parkinsons disease',
        'Levodopa crosses the blood brain barrier to treat symptoms',
        'Stroke patients require immediate emergency treatment',
        'Cerebral blood flow is reduced in ischemic stroke',
        'Thrombolysis must be administered within hours of stroke onset',
        'Hemiplegia affects the contralateral side of the body',
        'Ischemic damage results from prolonged lack of blood supply',
        'Parkinsons disease causes tremor rigidity and bradykinesia',
        'Dopamine deficiency leads to motor dysfunction in parkinsons',
        'Plasminogen activator dissolves clots in occluded arteries',
        'Acute stroke management includes airway breathing circulation',
        'Thrombolytic therapy restores blood flow to ischemic regions',
        'Hypertension is a major risk factor for stroke',
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
        if feed_count % 10 == 0:
            save()
        return {'ok': True, **r}

    elif op == 'feed_batch':
        texts = cmd.get('texts', [])
        cycles = cmd.get('cycles', 1)
        r = mind.feed_batch(texts, cycles=cycles)
        # Do NOT save after batch — let the user call /api/mind/save explicitly.
        # Saving 129K sentences of state mid-request blocks the response
        # and can OOM the process during JSON serialization.
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
        d = mind.describe(cmd.get('word', ''))
        return {'ok': True, 'description': d}

    elif op == 'generate':
        word = cmd.get('word', '')
        ml = cmd.get('max_length', 15)
        tokens = mind.generate([word], max_length=ml)
        return {'ok': True, 'text': ' '.join(tokens)}

    elif op == 'activate':
        field = mind.activate(cmd.get('word', ''))
        top = sorted(field.items(), key=lambda x: -x[1])[:20]
        return {'ok': True, 'field': [{'word': w, 'energy': round(e, 4)} for w, e in top]}

    elif op == 'connect':
        src = cmd.get('from', '')
        tgt = cmd.get('to', '')
        path = mind.speaker.find_path(src, tgt, mind._co_graph)
        return {'ok': True, 'path': path, 'connected': path is not None}

    elif op == 'stats':
        return {'ok': True, **mind.stats()}

    elif op == 'confidence':
        return {'ok': True, **mind.confidence(cmd.get('word', ''))}

    elif op == 'hungry':
        return {'ok': True, 'gaps': mind.hungry()}

    elif op == 'consolidate':
        r = mind.consolidate()
        return {'ok': True, **r}

    elif op == 'practice':
        rounds = cmd.get('rounds', 10)
        r = mind.practice(rounds=rounds)
        return {'ok': True, **r}

    elif op == 'study':
        rounds = cmd.get('rounds', 5)
        level = cmd.get('level', None)
        r = mind.study(rounds=rounds, level=level)
        return {'ok': True, **r}

    elif op == 'assess':
        r = mind.assess_language()
        return {'ok': True, **r}

    elif op == 'decompose':
        word = cmd.get('word', '')
        r = mind.language.morphology.decompose(word)
        return {'ok': True, **r}

    elif op == 'synonyms':
        word = cmd.get('word', '')
        return {'ok': True, 'synonyms': mind.language.semantics.synonyms(word)}

    elif op == 'explain_semantic':
        word = cmd.get('word', '')
        return {'ok': True, 'explanation': mind.language.semantics.explain(word)}

    elif op == 'language_stats':
        return {'ok': True,
                'morphology': mind.language.morphology.stats(),
                'syntax': mind.language.syntax.stats(),
                'semantics': mind.language.semantics.stats(),
                'curriculum': mind.language.curriculum.stats()}

    elif op == 'save':
        save()
        return {'ok': True}

    else:
        return {'ok': False, 'error': f'unknown command: {op}'}


if __name__ == '__main__':
    boot()
    # Signal ready
    sys.stdout.write(json.dumps({'ok': True, 'status': 'ready', 'vocabulary': mind.stats()['vocabulary']}) + '\n')
    sys.stdout.flush()

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
        sys.stdout.write(json.dumps(result, ensure_ascii=False) + '\n')
        sys.stdout.flush()
