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

    # ═══ BACKGROUND PRACTICE — via threading with a lock ═══
    # The lock prevents practice from running while a request is being handled.
    # This avoids the deadlock: practice yields the GIL to let stdin read.
    import threading, time as _time

    # ═══ BILATERAL ARCHITECTURE — TWO HEMISPHERES ═══
    #
    # LEFT HEMISPHERE: logistics — practice, study, consolidate, feed
    # RIGHT HEMISPHERE: conversation — deliberate, describe, activate, score
    #
    # Connected by CORPUS CALLOSUM: shared mind object.
    # Each hemisphere has its own lock. They can run SIMULTANEOUSLY.
    # Left practices while right answers questions.
    # Cross-talk happens through the shared cortex/co-graph.

    _left_busy = threading.Lock()   # LEFT: logistics (practice, feed, consolidate)
    _right_busy = threading.Lock()  # RIGHT: conversation (deliberate, describe, score)
    _state = [0]

    # Route commands to hemispheres
    LEFT_OPS = {'feed', 'feed_batch', 'consolidate', 'practice', 'study', 'save'}
    RIGHT_OPS = {'deliberate', 'describe', 'generate', 'activate', 'score',
                 'confidence', 'connect', 'explain_semantic', 'synonyms',
                 'decompose', 'assess', 'language_stats', 'hungry'}

    def _background_practice():
        # ═══ RELATIVISTIC TIME ═══
        #
        # E = mc² — energy and mass are the same thing.
        # In the mind: LEARNING and TIME are the same thing.
        # Time that produces no learning is wasted time.
        #
        # The baby practices at CPU speed — not wall clock speed.
        # It BURSTS: practice hard until learning rate drops,
        # then rest briefly, then burst again.
        #
        # Between bursts: yield to let requests through (10ms).
        # During burst: 10 practice + 3 study + consolidate.
        # After burst: check if learning rate is still positive.
        # If dopamine surprise is near zero → learning saturated → rest longer.
        # If dopamine surprise is high → still learning → keep going.

        while True:
            _time.sleep(0.01)  # 10ms yield for requests
            if len(mind.cortex.word_freq) < 20:
                _time.sleep(1)
                continue
            if _left_busy.locked():
                continue
            acquired = _left_busy.acquire(timeout=0.001)
            if not acquired:
                continue
            try:
                mind.practice(rounds=1)
                _state[0] += 1
                if _state[0] % 5 == 0:
                    mind.study(rounds=1)
                if _state[0] % 20 == 0:
                    mind.consolidate()
            except Exception:
                pass
            finally:
                _left_busy.release()

            # REST: must yield GIL long enough for stdin to be read.
            # Python GIL means practice and request handling can't truly
            # run in parallel. The sleep is what lets requests through.
            trend = mind.signal.recent_trend(5)
            if trend > 0.4:
                _time.sleep(0.5)    # 500ms — learning, but let requests through
            else:
                _time.sleep(2.0)    # 2s — saturated, rest longer

    t = threading.Thread(target=_background_practice, daemon=True)
    t.start()

    import signal as _signal
    _signal.signal(_signal.SIGINT, lambda *a: None)  # Ignore Ctrl+C
    try:
        _signal.signal(_signal.SIGPIPE, lambda *a: None)  # Ignore broken pipe
    except AttributeError:
        pass  # Windows doesn't have SIGPIPE

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        # ═══ CORPUS CALLOSUM: route to correct hemisphere ═══
        try:
            cmd = json.loads(line)
            req_id = cmd.pop('_id', None)
            op = cmd.get('cmd', '')
        except Exception as e:
            req_id = None
            result = {'ok': False, 'error': str(e)}
            if req_id is not None:
                result['_id'] = req_id
            try:
                sys.stdout.write(json.dumps(result) + '\n')
                sys.stdout.flush()
            except (BrokenPipeError, OSError):
                break
            continue

        # Left hemisphere: logistics (feed, practice, consolidate)
        # Right hemisphere: conversation (deliberate, describe, score)
        # Stats/ping: no lock needed
        if op in LEFT_OPS:
            lock = _left_busy
        elif op in RIGHT_OPS:
            lock = _right_busy
        else:
            lock = None  # stats, ping — no lock

        if lock:
            with lock:
                try:
                    result = handle(cmd)
                except Exception as e:
                    result = {'ok': False, 'error': str(e)}
        else:
            try:
                result = handle(cmd)
            except Exception as e:
                result = {'ok': False, 'error': str(e)}
        if req_id is not None:
            result['_id'] = req_id
        try:
            sys.stdout.write(json.dumps(result, ensure_ascii=False) + '\n')
            sys.stdout.flush()
        except (BrokenPipeError, OSError):
            break  # Node closed the pipe — exit gracefully
