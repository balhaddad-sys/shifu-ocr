#!/usr/bin/env python3
"""
SHIFU BRAIN — one process, one mind, one heartbeat.

The heart beats regardless of sleep or thinking.
The processors shift between brain wave states:

  DELTA (idle): deep maintenance. Consolidate, myelinate, reinforce.
  THETA (practice): generate, score, strengthen. Light learning.
  ALPHA (ready): waiting for input. Calm but aware.
  BETA (active): answering questions, processing input.

The heartbeat runs in ALL states.
The brain state shifts based on what's happening.
Feedback loops are intact — everything in one memory space.
"""

import json
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shifu_ocr.mind import ShifuMind

SAVE_PATH = os.path.join('.state', 'shifu_brain.json')
mind = None
brain_state = 'alpha'
last_request = 0
heartbeat_count = 0

# ═══ BARORECEPTOR — senses learning velocity, not absolute level ═══
_myel_history = []       # Last N myelination counts (for slope detection)
_learning_velocity = 0.0  # Slope of myelination over time
_baroreceptor_response = 'normal'  # normal, compensate, urgent


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
    for t in [
        'Stroke is a disease caused by arterial occlusion in the brain',
        'Dopamine is a neurotransmitter affected in parkinsons disease',
        'Thrombolysis is a treatment that dissolves blood clots',
        'The cerebral artery supplies blood to the cortex',
        'Hypertension is a major risk factor for cerebral stroke',
    ]:
        mind.feed(t)


def save():
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    try:
        mind.save(SAVE_PATH)
    except Exception:
        pass


def baroreceptor():
    """
    Senses LEARNING VELOCITY, not absolute level.
    Like arterial baroreceptors sensing pressure CHANGE.

    Measures: myelination slope over last N heartbeats.
    If slope flattening → compensate (raise practice intensity).
    If slope zero for too long → urgent (force consolidation + replay).
    """
    global _learning_velocity, _baroreceptor_response

    current_myel = mind.cortex._myel_count
    _myel_history.append(current_myel)
    if len(_myel_history) > 20:
        _myel_history.pop(0)

    # Compute slope: how fast is myelination growing?
    if len(_myel_history) >= 3:
        recent = _myel_history[-3:]
        slope = (recent[-1] - recent[0]) / max(len(recent) - 1, 1)
        _learning_velocity = slope

        if slope > 1:
            _baroreceptor_response = 'normal'    # Growing — all good
        elif slope > 0:
            _baroreceptor_response = 'compensate' # Slowing — increase drive
        else:
            _baroreceptor_response = 'urgent'     # Stalled — force action
    else:
        _baroreceptor_response = 'normal'


def detect_brain_state():
    """
    Brain state from elapsed time + baroreceptor feedback.
    Baroreceptor can OVERRIDE the time-based state:
    if learning stalled (urgent), force theta even if recently active.
    """
    global brain_state
    elapsed = time.time() - last_request
    baroreceptor()

    # Time-based baseline
    if elapsed < 2:
        brain_state = 'beta'
    elif elapsed < 10:
        brain_state = 'alpha'
    elif elapsed < 30:
        brain_state = 'theta'
    else:
        brain_state = 'delta'

    # Baroreceptor override: stalled learning → force deeper state
    if _baroreceptor_response == 'urgent' and brain_state in ('alpha', 'beta'):
        brain_state = 'theta'  # Force practice even during "active" time
    elif _baroreceptor_response == 'compensate' and brain_state == 'alpha':
        brain_state = 'theta'  # Nudge from idle to practice


def heartbeat():
    """
    The SPIDER sits in the center and FEELS the web.
    Don't scan. Vibrate. The web does its own maintenance.

    Each beat: pick a random thread, pluck it, feel the response.
    Every neuron that vibrates strengthens its connections (Hebbian).
    Myelination happens through USAGE, not through scanning.
    """
    global heartbeat_count
    heartbeat_count += 1

    hb = {'myelinated_new': 0, 'fired': 0}

    if brain_state == 'beta':
        pass  # Freeze. Don't pluck the web during active thinking.

    elif brain_state == 'alpha':
        # Gentle pluck — one random vibration
        if mind.neural_field.neurons:
            hb = mind.neural_field.heartbeat()

    elif brain_state == 'theta':
        # Strong pluck — MANY vibrations (theta is active learning)
        if mind.neural_field.neurons:
            for _ in range(20):  # 20 beats per theta cycle
                r = mind.neural_field.heartbeat()
                hb['myelinated_new'] += r.get('myelinated_new', 0)
                hb['fired'] += r.get('fired', 0)

    elif brain_state == 'delta':
        # Full maintenance: vibrate + prune
        if mind.neural_field.neurons:
            for _ in range(5):
                r = mind.neural_field.heartbeat()
                hb['myelinated_new'] += r.get('myelinated_new', 0)
                hb['fired'] += r.get('fired', 0)
            hb['pruned'] = mind.neural_field.prune()

    return hb


def idle_cycle():
    """
    Brain state determines what happens between requests.
    Everything goes through the web — vibrations, not scans.
    """
    detect_brain_state()
    hb = heartbeat()

    result = {
        'state': brain_state,
        'baroreceptor': _baroreceptor_response,
        'velocity': round(_learning_velocity, 2),
    }

    if brain_state == 'delta':
        # Deep sleep: consolidate (builds neural field) + prune
        r = mind.consolidate(focus_size=100)
        result['did'] = f'delta: consolidate + prune({hb.get("pruned",0)}) + myel({hb.get("myelinated_new",0)})'

    elif brain_state == 'theta':
        # Practice through the web: pluck the conviction target hard
        purpose = mind.conviction.discover_purpose(mind)
        focus = None
        if purpose:
            info = mind.conviction._goals.get(purpose, {})
            focus = info.get('frontier') or info.get('core')
        if focus and mind.neural_field.neurons and focus in mind.neural_field.neurons:
            pr = mind.neural_field.practice(focus)
            result['did'] = f'theta: practice({focus}, fired={pr.get("fired",0)}) + myel({hb.get("myelinated_new",0)+pr.get("improved",0)})'
        else:
            result['did'] = f'theta: heartbeat(fired={hb.get("fired",0)}, myel={hb.get("myelinated_new",0)})'

    elif brain_state == 'alpha':
        result['did'] = f'alpha: heartbeat(fired={hb.get("fired",0)}, myel={hb.get("myelinated_new",0)})'

    else:
        result['did'] = 'beta: active'

    return result


def handle(cmd):
    global last_request
    last_request = time.time()
    op = cmd.get('cmd', '')

    if op == 'ping':
        return {'ok': True, 'status': 'alive', 'brain_state': brain_state}

    elif op == 'feed':
        r = mind.feed(cmd.get('text', ''))
        heartbeat()
        return {'ok': True, **r}

    elif op == 'feed_batch':
        r = mind.feed_batch(cmd.get('texts', []), cycles=cmd.get('cycles', 1))
        heartbeat()
        return {'ok': True, **r}

    elif op == 'deliberate':
        heartbeat()
        r = mind.deliberate(cmd.get('query', ''))
        r['retrieved'] = [{'word': x['word'], 'energy': round(x['energy'], 4)} for x in r.get('retrieved', [])]
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

    elif op == 'score':
        r = mind.score_text(cmd.get('text', ''))
        return {'ok': True, 'coherence': r['coherence'], 'scores': r['scores']}

    elif op == 'confidence':
        return {'ok': True, **mind.confidence(cmd.get('word', ''))}

    elif op == 'connect':
        path = mind.speaker.find_path(cmd.get('from', ''), cmd.get('to', ''), mind._co_graph)
        return {'ok': True, 'path': path, 'connected': path is not None}

    elif op == 'candidates':
        cands = cmd.get('candidates', [])
        ctx = cmd.get('context', [])
        return {'ok': True, 'ranked': mind.predict_candidates([(c[0], c[1]) for c in cands], context_words=ctx)}

    elif op == 'compass':
        return {'ok': True, **mind.compass()}

    elif op == 'introspect':
        return {'ok': True, 'voice': mind.introspect()}

    elif op == 'cry':
        return {'ok': True, 'cry': mind.cry()}

    elif op == 'hunger':
        return {'ok': True, **mind.hunger_receptors()}

    elif op == 'hungry':
        return {'ok': True, 'gaps': mind.hungry()}

    elif op == 'consolidate':
        r = mind.consolidate(focus_size=cmd.get('focus_size', 500))
        heartbeat()
        return {'ok': True, **r}

    elif op == 'practice':
        r = mind.practice(rounds=cmd.get('rounds', 10))
        heartbeat()
        return {'ok': True, **r}

    elif op == 'study':
        return {'ok': True, **mind.study(rounds=cmd.get('rounds', 5), level=cmd.get('level'))}

    elif op == 'heartbeat':
        return {'ok': True, **heartbeat()}

    elif op == 'autonomous_step':
        r = mind.autonomous_step()
        return {'ok': True, **r}

    elif op == 'idle':
        return {'ok': True, **idle_cycle()}

    elif op == 'assess':
        return {'ok': True, **mind.assess_language()}

    elif op == 'decompose':
        return {'ok': True, **mind.language.morphology.decompose(cmd.get('word', ''))}

    elif op == 'synonyms':
        return {'ok': True, 'synonyms': mind.language.semantics.synonyms(cmd.get('word', ''))}

    elif op == 'explain_semantic':
        return {'ok': True, 'explanation': mind.language.semantics.explain(cmd.get('word', ''))}

    elif op == 'language_stats':
        return {'ok': True, 'morphology': mind.language.morphology.stats(), 'syntax': mind.language.syntax.stats(), 'semantics': mind.language.semantics.stats(), 'curriculum': mind.language.curriculum.stats()}

    elif op == 'stats':
        return {'ok': True, **mind.stats(), 'brain_state': brain_state, 'heartbeat_count': heartbeat_count}

    elif op == 'save':
        save()
        return {'ok': True}

    else:
        return {'ok': False, 'error': f'unknown: {op}'}


if __name__ == '__main__':
    boot()
    sys.stdout.write(json.dumps({
        'ok': True, 'status': 'ready',
        'vocabulary': len(mind.cortex.word_freq),
    }) + '\n')
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
