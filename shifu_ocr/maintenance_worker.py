#!/usr/bin/env python3
"""
AUTONOMIC NERVOUS SYSTEM — maintains the heartbeat.

Handles: consolidate, practice, study, heartbeat, assess, save.
Reads co_graph/vocab from shared state, processes, writes cortex back.
Has conviction — pushes through when dopamine is flat.
"""

import json, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shifu_ocr.mind import ShifuMind
from shifu_ocr.mind.nervous_system import import_graphs, import_cortex, export_cortex, export_graphs

mind = ShifuMind(initial_layers=['identity', 'appearance', 'function', 'mechanism', 'relation'])

# Load shared state
import_graphs(mind)
import_cortex(mind)

sys.stdout.write(json.dumps({'ok': True, 'status': 'ready', 'role': 'maintenance', 'vocabulary': len(mind.cortex.word_freq)}) + '\n')
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

        # Always reload latest graphs before maintenance
        if op in ('consolidate', 'practice', 'study', 'heartbeat', 'autonomous_step'):
            import_graphs(mind)

        if op == 'ping':
            result = {'ok': True, 'status': 'alive', 'role': 'maintenance'}
        elif op == 'consolidate':
            r = mind.consolidate(focus_size=cmd.get('focus_size'))
            # Run heartbeats for myelination
            total_myel = 0
            total_short = 0
            for _ in range(5):
                hb = mind.heartbeat()
                total_myel += hb.get('myelinated_new', 0)
                total_short += hb.get('shortcuts', 0)
            r['heartbeat_myel'] = total_myel
            r['heartbeat_shortcuts'] = total_short
            # Write back to shared state
            export_cortex(mind)
            result = {'ok': True, **r}
        elif op == 'practice':
            r = mind.practice(rounds=cmd.get('rounds', 10))
            export_cortex(mind)  # Only cortex, not graphs
            result = {'ok': True, **r}
        elif op == 'study':
            r = mind.study(rounds=cmd.get('rounds', 5), level=cmd.get('level'))
            result = {'ok': True, **r}
        elif op == 'heartbeat':
            r = mind.heartbeat()
            export_cortex(mind)
            result = {'ok': True, **r}
        elif op == 'autonomous_step':
            r = mind.autonomous_step()
            # Export BOTH cortex AND co-graph.
            # The maintenance worker imported the feed worker's co-graph at boot.
            # Practice adds new connections to it. Export the combined graph.
            mind.cortex._epoch += 1
            export_cortex(mind)
            export_graphs(mind)
            from shifu_ocr.mind.nervous_system import write_epoch
            write_epoch(mind.cortex._epoch)
            result = {'ok': True, **r}
        elif op == 'assess':
            r = mind.assess_language()
            result = {'ok': True, **r}
        elif op == 'save':
            export_cortex(mind)
            export_graphs(mind)
            result = {'ok': True}
        elif op == 'stats':
            result = {'ok': True, **mind.stats()}
        else:
            result = {'ok': False, 'error': f'maintenance worker: unknown cmd {op}'}
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
