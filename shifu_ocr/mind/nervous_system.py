"""
Nervous System — shared state layer between brain processes.

Three processes, one mind. Like the body's nervous systems:
  - Enteric (gut/feed): absorbs recklessly, high plasticity
  - Cerebral (query): traverses carefully, low latency
  - Autonomic (maintenance): maintains heartbeat, conviction

They share knowledge through the filesystem:
  .state/mind_co_graph.json  — co-occurrence graph (the raw knowledge)
  .state/mind_nx_graph.json  — next-word transitions
  .state/mind_vocab.json     — word frequencies
  .state/mind_cortex.json    — cortex layers + synapses (consolidated)
  .state/mind_signal.json    — dopamine/conviction state
  .state/mind_epoch.txt      — current epoch (atomic counter)

Write protocol:
  - Feed worker WRITES co_graph, nx_graph, vocab after every batch
  - Maintenance worker READS co/nx/vocab, WRITES cortex after consolidate
  - Query worker READS cortex, co_graph on demand (lazy reload)

No locks needed — each file has ONE writer.
Readers get eventually-consistent state.
"""

from __future__ import annotations
import json
import os
import time
from typing import Dict, Optional, Any


STATE_DIR = '.state'

# File paths
PATHS = {
    'co_graph': os.path.join(STATE_DIR, 'mind_co_graph.json'),
    'nx_graph': os.path.join(STATE_DIR, 'mind_nx_graph.json'),
    'vocab': os.path.join(STATE_DIR, 'mind_vocab.json'),
    'cortex': os.path.join(STATE_DIR, 'mind_cortex.json'),
    'signal': os.path.join(STATE_DIR, 'mind_signal.json'),
    'trunk': os.path.join(STATE_DIR, 'mind_trunk.json'),
    'neural': os.path.join(STATE_DIR, 'mind_neural.json'),
    'epoch': os.path.join(STATE_DIR, 'mind_epoch.txt'),
}


def ensure_dir():
    os.makedirs(STATE_DIR, exist_ok=True)


def write_json(key: str, data: Any) -> None:
    """Write state to file. Direct write — Windows can't do atomic rename
    when other processes have the file open."""
    ensure_dir()
    path = PATHS[key]
    # Try atomic first, fall back to direct write
    tmp = path + '.tmp'
    try:
        with open(tmp, 'w') as f:
            json.dump(data, f)
        os.replace(tmp, path)
    except OSError:
        # Windows file lock — write directly
        try:
            os.remove(tmp)
        except OSError:
            pass
        with open(path, 'w') as f:
            json.dump(data, f)


def read_json(key: str) -> Optional[Any]:
    """Read state. Returns None if not found."""
    path = PATHS[key]
    if not os.path.exists(path):
        return None
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return None


def read_epoch() -> int:
    path = PATHS['epoch']
    if not os.path.exists(path):
        return 0
    try:
        with open(path, 'r') as f:
            return int(f.read().strip())
    except (ValueError, IOError):
        return 0


def write_epoch(epoch: int) -> None:
    ensure_dir()
    path = PATHS['epoch']
    with open(path, 'w') as f:
        f.write(str(epoch))


def export_graphs(mind) -> dict:
    """Export feed worker state to shared files."""
    ensure_dir()
    # Co-graph: cap each word's neighbors for size
    co = {}
    for w, neighbors in mind._co_graph.items():
        top = dict(sorted(neighbors.items(), key=lambda x: -x[1])[:100])
        co[w] = top
    write_json('co_graph', co)

    # Nx-graph
    nx = {}
    for w, neighbors in mind._nx_graph.items():
        top = dict(sorted(neighbors.items(), key=lambda x: -x[1])[:50])
        nx[w] = top
    write_json('nx_graph', nx)

    # Vocab
    write_json('vocab', dict(mind.cortex.word_freq))

    # Epoch
    write_epoch(mind.cortex._epoch)

    return {
        'co_words': len(co),
        'nx_words': len(nx),
        'vocab': len(mind.cortex.word_freq),
    }


def import_graphs(mind) -> dict:
    """Import shared state into a mind (for query/maintenance workers)."""
    co = read_json('co_graph')
    if co:
        mind._co_graph = co
    nx = read_json('nx_graph')
    if nx:
        mind._nx_graph = nx
    vocab = read_json('vocab')
    if vocab:
        mind.cortex.word_freq = vocab
        mind.cortex.total_words = sum(vocab.values())
    epoch = read_epoch()
    if epoch > mind.cortex._epoch:
        mind.cortex._epoch = epoch

    return {
        'co_words': len(mind._co_graph),
        'nx_words': len(mind._nx_graph),
        'vocab': len(mind.cortex.word_freq),
    }


def export_cortex(mind) -> dict:
    """Export cortex + trunk + neural field (maintenance worker writes)."""
    ensure_dir()
    write_json('cortex', mind.cortex.to_dict())
    write_json('signal', mind.signal.to_dict())
    if mind.trunk.domains:
        write_json('trunk', mind.trunk.to_dict())
    if mind.neural_field.neurons:
        write_json('neural', mind.neural_field.to_dict())
    return {'layers': len(mind.cortex._layers), 'myel': mind.cortex._myel_count}


def import_cortex(mind) -> dict:
    """Import cortex + trunk + neural field."""
    from .cortex import Cortex
    cortex_d = read_json('cortex')
    if cortex_d:
        mind.cortex = Cortex.from_dict(cortex_d)
    signal_d = read_json('signal')
    if signal_d:
        from .signal import Signal
        mind.signal = Signal.from_dict(signal_d)
    trunk_d = read_json('trunk')
    if trunk_d:
        from .trunk import Trunk
        mind.trunk = Trunk.from_dict(trunk_d)
    neural_d = read_json('neural')
    if neural_d:
        from .neuron import NeuralField
        mind.neural_field = NeuralField.from_dict(neural_d)
    return {'layers': len(mind.cortex._layers), 'myel': mind.cortex._myel_count}
