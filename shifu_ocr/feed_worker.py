#!/usr/bin/env python3
"""
ENTERIC NERVOUS SYSTEM — the gut that absorbs.

High plasticity. Reckless. Can bloat to 500MB.
Only handles: feed, feed_batch.
Writes co_graph, nx_graph, vocab to shared state after every batch.
"""

import json, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shifu_ocr.mind import ShifuMind
from shifu_ocr.mind.nervous_system import export_graphs, write_epoch

mind = ShifuMind(initial_layers=['identity', 'appearance', 'function', 'mechanism', 'relation'])

# Seed
for t in [
    'Stroke is a disease caused by arterial occlusion in the brain',
    'Dopamine is a neurotransmitter affected in parkinsons disease',
    'Thrombolysis is a treatment that dissolves blood clots',
    'The cerebral artery supplies blood to the cortex',
    'Hypertension is a major risk factor for cerebral stroke',
]:
    mind.feed(t)

sys.stdout.write(json.dumps({'ok': True, 'status': 'ready', 'role': 'feed', 'vocabulary': len(mind.cortex.word_freq)}) + '\n')
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

        if op == 'ping':
            result = {'ok': True, 'status': 'alive', 'role': 'feed'}
        elif op == 'feed':
            r = mind.feed(cmd.get('text', ''))
            export_graphs(mind)
            result = {'ok': True, **r}
        elif op == 'feed_batch':
            r = mind.feed_batch(cmd.get('texts', []), cycles=cmd.get('cycles', 1))
            try:
                exported = export_graphs(mind)
            except Exception:
                exported = {'error': 'export failed, will retry'}
            result = {'ok': True, **r, 'exported': exported}
        elif op == 'stats':
            result = {'ok': True, **mind.stats()}
        else:
            result = {'ok': False, 'error': f'feed worker: unknown cmd {op}'}
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
