"""
shifu_ocr.mind — Unified Cognitive Architecture

The mind that sees. Bridges cognitive language understanding
with perceptual character recognition.

    from shifu_ocr.mind import ShifuMind

    mind = ShifuMind()
    mind.feed("Stroke is caused by arterial occlusion")
    mind.score_text("stroke occlusion artery")
    mind.predict_candidates([('stroke', 0.9), ('strake', 0.7)],
                            context_words=['patient', 'cerebral'])
"""

from .mind import ShifuMind
from .cortex import Cortex, Layer
from .field import Field
from .gate import Gate
from .signal import Signal
from .trunk import Trunk
from .memory import Memory, Episode
from .speaker import Speaker
from .thinker import Thinker, WorkingMemory
from .imagination import Imagination
from .attention import Attention
from .neuron import Neuron, NeuralField
from .landscape import Landscape
from ._types import Synapse, Assembly, Domain
from ._protocols import Feedable, Queryable, Temporal, Serializable

__all__ = [
    'ShifuMind',
    'Cortex', 'Layer',
    'Field',
    'Gate',
    'Signal',
    'Trunk',
    'Memory', 'Episode',
    'Speaker',
    'Thinker', 'WorkingMemory', 'Imagination', 'Attention',
    'Landscape',
    'Synapse', 'Assembly', 'Domain',
    'Feedable', 'Queryable', 'Temporal', 'Serializable',
]
