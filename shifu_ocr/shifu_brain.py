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
    Brain state from elapsed time + baroreceptor + fatigue.

    Fatigue override: if conviction is fatigued/satisfied, force alpha (rest).
    The prefrontal cortex doesn't need to be involved for learned actions —
    the cerebellum handles those. Only process what's NEW.
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

    # Fatigue override: force rest when conviction is exhausted
    if mind.conviction.is_fatigued() or mind.conviction.is_satisfied():
        if brain_state in ('theta', 'delta'):
            brain_state = 'alpha'  # Downshift to rest — stop pushing

    # Baroreceptor override: stalled learning → force deeper state
    # BUT only if not fatigued (can't force a tired brain)
    elif _baroreceptor_response == 'urgent' and brain_state in ('alpha', 'beta'):
        brain_state = 'theta'
    elif _baroreceptor_response == 'compensate' and brain_state == 'alpha':
        brain_state = 'theta'


def heartbeat():
    """
    PROPORTIONAL heartbeat — effort scales with what needs learning.

    Myelinated connections = cerebellum. Automatic. No effort.
    Unmyelinated connections = prefrontal cortex. Active processing.

    "I know how to walk — the cerebellum takes the work now.
    Now I can multitask."

    Beats per cycle = proportional to unmyelinated ratio.
    A brain that's 90% myelinated needs only 1-2 beats.
    A brain that's 10% myelinated needs 10+ beats.
    """
    global heartbeat_count
    heartbeat_count += 1

    hb = {'myelinated_new': 0, 'fired': 0}

    if brain_state == 'beta':
        pass  # Active thinking — don't pluck the web.

    elif mind.neural_field.neurons:
        # Proportional effort based on density
        if not hasattr(heartbeat, '_density') or heartbeat_count % 50 == 0:
            nf_stats = mind.neural_field.stats()
            n_neurons = nf_stats.get('neurons', 1)
            total_c = nf_stats.get('connections', 0)
            heartbeat._density = total_c / max(n_neurons, 1)
        density = heartbeat._density

        effort = 1.0 / (1.0 + density / 5.0)
        if brain_state == 'alpha':
            beats = max(1, int(effort * 3))
        elif brain_state == 'theta':
            beats = max(1, int(effort * 5))
        else:
            beats = max(1, int(effort * 3))

        # Rust neural field heartbeat (fast vibration)
        for _ in range(beats):
            r = mind.neural_field.heartbeat()
            hb['myelinated_new'] += r.get('myelinated_new', 0)
            hb['fired'] += r.get('fired', 0)

        # AUC myelination runs in the diastolic flow (every 300th beat)
        # Not here — this function is called from handle() too

        if brain_state == 'delta' and heartbeat_count % 10 == 0:
            hb['pruned'] = mind.neural_field.prune()

    return hb


def idle_cycle():
    """
    LIGHTWEIGHT status report only. No heavy processing.
    All heavy work (replay, vibration, pruning) happens in the
    diastolic flow of the main loop — NOT here.
    This is called from /api/mind/idle — must return in < 10ms.
    """
    detect_brain_state()
    unreplayed = len(mind.memory.unreplayed(k=1))
    nf_count = len(mind.neural_field.neurons)

    return {
        'ok': True,
        'state': brain_state,
        'baroreceptor': _baroreceptor_response,
        'velocity': round(_learning_velocity, 2),
        'did': f'{brain_state}: hb={heartbeat_count}, neurons={nf_count}, unreplayed={unreplayed}',
        'voice': mind.conviction._voice[-1] if mind.conviction._voice else None,
    }


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

    elif op == 'replay':
        unreplayed = mind.memory.unreplayed(k=cmd.get('k', 20))
        if unreplayed:
            r = mind.replay_batch(unreplayed, max_episodes=cmd.get('k', 20))
        else:
            least = mind.memory.least_replayed(k=cmd.get('k', 10))
            r = mind.replay_batch(least, max_episodes=cmd.get('k', 10))
        heartbeat()
        return {'ok': True, **r}

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
        result = {'ok': True, **mind.stats(), 'brain_state': brain_state, 'heartbeat_count': heartbeat_count}
        if hasattr(handle, '_last_passive'):
            result['passive'] = handle._last_passive
        return result

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

    # ═══ SIMPLEST POSSIBLE MAIN LOOP ═══
    # No threads. No queues. No GIL games.
    # Just block on stdin.readline(). When a line arrives, process it.
    # Between lines, nothing happens. The brain is quiet.
    #
    # This is the only approach that's 100% reliable on Windows
    # with piped stdin from Node.js.
    #
    # The diastolic flow (heartbeat, replay, AUC) runs INSIDE
    # command handlers — not in the background. When Node sends
    # a command, we process it AND do one heartbeat. When Node
    # is idle, we're idle. Like a real brain that's asleep.

    while True:
        try:
            line = sys.stdin.readline()
        except Exception:
            break
        if not line:
            break  # True EOF — Node closed the pipe

        line = line.strip()
        if not line:
            continue

        try:
            cmd = json.loads(line)
            req_id = cmd.pop('_id', None)

            # Process the command
            result = handle(cmd)

            # ═══ PASSIVE LEARNING — tied to CPU heartbeat, not wall clock ═══
            # Every command processed = one heartbeat. Every Nth heartbeat,
            # do a learning step. The brain's clock is its own activity.
            heartbeat_count += 1
            vocab = len(mind.cortex.word_freq)
            if vocab > 10:
                try:
                    # FAST: Rust heartbeat — every beat
                    if bool(mind.neural_field.neurons):
                        mind.neural_field.heartbeat()

                    # SLOW: Python work — every 100th beat (~10 sec at 10 BPS)
                    # Rotate: AUC → replay → consolidate → study
                    slow_phase = (heartbeat_count // 100) % 4
                    if heartbeat_count % 100 == 0:
                        if slow_phase == 0:
                            hb = mind.heartbeat()
                            handle._last_passive = f'myel+{hb.get("myelinated_new",0)}'
                        elif slow_phase == 1:
                            unreplayed = mind.memory.unreplayed(k=2)
                            if unreplayed:
                                r = mind.replay_batch(unreplayed, max_episodes=2)
                                handle._last_passive = f'replay({r.get("replayed",0)})'
                        elif slow_phase == 2:
                            cs = 20 if vocab > 10000 else 30 if vocab > 1000 else 50
                            r = mind.consolidate(focus_size=cs)
                            handle._last_passive = f'consolidate(routed={r.get("routed",0)})'
                        elif slow_phase == 3 and vocab > 50:
                            pr = mind.language.curriculum.practice(mind, rounds=1)
                            for ex in pr.get('exercises', []):
                                text = ex.get('sentence') or ex.get('phrase', '')
                                if text and ex.get('score', 0) > 0.2:
                                    mind.speaker.reinforce(text.split(), strength=ex['score'])
                            handle._last_passive = f'study(L{pr.get("current_level",1)})'
                except Exception:
                    pass

                # Auto-save every 100 heartbeats — calibration persists
                if heartbeat_count % 100 == 0 and heartbeat_count > 0:
                    try:
                        save()
                    except Exception:
                        pass

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
