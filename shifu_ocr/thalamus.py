#!/usr/bin/env python3
"""
THALAMUS — the central router of the mind.

ONE nervous system. ONE process handles ALL commands.
But internally, the thalamus DISMANTLES every request:

  1. RECEIVE — the signal arrives (question, text, PDF)
  2. DISMANTLE — break into modality-specific pieces
  3. ROUTE — send each piece to the right cortical area
  4. PROCESS — each area processes its modality
  5. ASSEMBLE — collect results from all areas
  6. RESPOND — send unified response

The thalamus doesn't think. It ROUTES.
The cortical areas think. They're all in one process
but they process different ASPECTS of the same input.

Isolation comes from FAULT TOLERANCE:
if one modality fails, the others still contribute.
"""

import json, sys, os, traceback
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shifu_ocr.mind import ShifuMind
from shifu_ocr.mind.nervous_system import (
    export_graphs, export_cortex, write_epoch,
    import_graphs, import_cortex, read_epoch,
)

mind = ShifuMind(initial_layers=['identity', 'appearance', 'function', 'mechanism', 'relation'])
_last_epoch = 0

def reload_shared_state():
    """Reload what feed worker and maintenance worker wrote."""
    global _last_epoch
    epoch = read_epoch()
    if epoch > _last_epoch:
        import_graphs(mind)
        import_cortex(mind)  # This also imports neural field from disk if available
        mind.cortex._invalidate_cache()
        mind.field.invalidate_cache()
        _last_epoch = epoch

# Seed only if no shared state exists
import_graphs(mind)
import_cortex(mind)
if len(mind.cortex.word_freq) < 10:
    for t in [
        'Stroke is a disease caused by arterial occlusion in the brain',
        'Dopamine is a neurotransmitter affected in parkinsons disease',
        'Thrombolysis is a treatment that dissolves blood clots',
        'The cerebral artery supplies blood to the cortex',
        'Hypertension is a major risk factor for cerebral stroke',
    ]:
        mind.feed(t)


# ═══ CORTICAL AREAS — each processes one modality ═══

def area_absorb(cmd):
    """Wernicke's area — language input processing."""
    op = cmd.get('cmd', '')
    if op == 'feed':
        r = mind.feed(cmd.get('text', ''))
        export_graphs(mind)
        return {'ok': True, **r}
    elif op == 'feed_batch':
        r = mind.feed_batch(cmd.get('texts', []), cycles=cmd.get('cycles', 1))
        export_graphs(mind)
        return {'ok': True, **r}
    return None


def area_comprehend(cmd):
    """Association cortex — meaning, deliberation, reasoning."""
    op = cmd.get('cmd', '')
    if op == 'deliberate':
        r = mind.deliberate(cmd.get('query', ''))
        r['retrieved'] = [{'word': x['word'], 'energy': round(x['energy'], 4)} for x in r.get('retrieved', [])]
        return {'ok': True, **r}
    elif op == 'describe':
        return {'ok': True, 'description': mind.describe(cmd.get('word', ''))}
    elif op == 'activate':
        field = mind.activate(cmd.get('word', ''))
        top = sorted(field.items(), key=lambda x: -x[1])[:20]
        return {'ok': True, 'field': [{'word': w, 'energy': round(e, 4)} for w, e in top]}
    elif op == 'score':
        r = mind.score_text(cmd.get('text', ''))
        return {'ok': True, 'coherence': r['coherence'], 'scores': r['scores']}
    elif op == 'connect':
        path = mind.speaker.find_path(cmd.get('from', ''), cmd.get('to', ''), mind._co_graph)
        return {'ok': True, 'path': path, 'connected': path is not None}
    elif op == 'candidates':
        cands = cmd.get('candidates', [])
        ctx = cmd.get('context', [])
        return {'ok': True, 'ranked': mind.predict_candidates([(c[0], c[1]) for c in cands], context_words=ctx)}
    return None


def area_produce(cmd):
    """Broca's area — language output production."""
    op = cmd.get('cmd', '')
    if op == 'generate':
        tokens = mind.generate([cmd.get('word', '')], max_length=cmd.get('max_length', 15))
        return {'ok': True, 'text': ' '.join(tokens)}
    return None


def area_introspect(cmd):
    """Prefrontal cortex — self-awareness, compass, conviction."""
    op = cmd.get('cmd', '')
    if op == 'compass':
        return {'ok': True, **mind.compass()}
    elif op == 'introspect':
        return {'ok': True, 'voice': mind.introspect()}
    elif op == 'confidence':
        return {'ok': True, **mind.confidence(cmd.get('word', ''))}
    elif op == 'hungry':
        return {'ok': True, 'gaps': mind.hungry()}
    return None


def area_maintain(cmd):
    """Autonomic — consolidation, practice, heartbeat, myelination."""
    op = cmd.get('cmd', '')
    if op == 'consolidate':
        r = mind.consolidate()
        total_myel, total_short = 0, 0
        for _ in range(5):
            hb = mind.heartbeat()
            total_myel += hb.get('myelinated_new', 0)
            total_short += hb.get('shortcuts', 0)
        r['heartbeat_myel'] = total_myel
        r['heartbeat_shortcuts'] = total_short
        export_cortex(mind)
        return {'ok': True, **r}
    elif op == 'practice':
        r = mind.practice(rounds=cmd.get('rounds', 10))
        return {'ok': True, **r}
    elif op == 'study':
        r = mind.study(rounds=cmd.get('rounds', 5), level=cmd.get('level'))
        return {'ok': True, **r}
    elif op == 'heartbeat':
        r = mind.heartbeat()
        return {'ok': True, **r}
    elif op == 'assess':
        return {'ok': True, **mind.assess_language()}
    elif op == 'save':
        export_cortex(mind)
        export_graphs(mind)
        return {'ok': True}
    return None


def area_language(cmd):
    """Language-specific cortex — morphology, semantics, syntax."""
    op = cmd.get('cmd', '')
    if op == 'decompose':
        return {'ok': True, **mind.language.morphology.decompose(cmd.get('word', ''))}
    elif op == 'synonyms':
        return {'ok': True, 'synonyms': mind.language.semantics.synonyms(cmd.get('word', ''))}
    elif op == 'explain_semantic':
        return {'ok': True, 'explanation': mind.language.semantics.explain(cmd.get('word', ''))}
    elif op == 'language_stats':
        return {'ok': True, 'morphology': mind.language.morphology.stats(), 'syntax': mind.language.syntax.stats(), 'semantics': mind.language.semantics.stats(), 'curriculum': mind.language.curriculum.stats()}
    return None


def area_meta(cmd):
    """Stats, ping — meta operations."""
    op = cmd.get('cmd', '')
    if op == 'ping':
        return {'ok': True, 'status': 'alive'}
    elif op == 'stats':
        return {'ok': True, **mind.stats()}
    return None


# ═══ THE THALAMUS — dismantle, route, assemble ═══

# Cortical areas in processing order
AREAS = [area_meta, area_absorb, area_comprehend, area_produce,
         area_introspect, area_maintain, area_language]


def thalamus_route(cmd):
    """
    The thalamus receives a signal and routes it.
    First: reload shared state from disk (what feed/maintenance wrote).
    Then: route to cortical areas.
    """
    try:
        reload_shared_state()
    except Exception as e:
        pass  # Don't fail the request if reload fails

    errors = []
    for area in AREAS:
        try:
            result = area(cmd)
            if result is not None:
                return result
        except Exception as e:
            errors.append(f'{area.__name__}: {e}')
            continue
    return {'ok': False, 'error': f'no area handles {cmd.get("cmd","?")}: {"; ".join(errors)}'}


# ═══ MAIN LOOP ═══

if __name__ == '__main__':
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
            result = thalamus_route(cmd)
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
