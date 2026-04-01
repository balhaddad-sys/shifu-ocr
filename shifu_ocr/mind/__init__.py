"""
shifu_ocr.mind -- Unified Cognitive Architecture

The mind that sees. Bridges cognitive language understanding
with perceptual character recognition.

    from shifu_ocr.mind import ShifuMind

    mind = ShifuMind()
    mind.feed("Stroke is caused by arterial occlusion")
    mind.score_text("stroke occlusion artery")
    mind.predict_candidates([('stroke', 0.9), ('strake', 0.7)],
                            context_words=['patient', 'cerebral'])
"""

# Lazy imports: only ShifuMind is loaded eagerly since it's the primary API.
# Other modules are imported on first access to avoid pulling in numpy
# and all submodules when only ShifuMind is needed.

from .mind import ShifuMind
from ._types import Synapse, Assembly, Domain

__all__ = [
    'ShifuMind',
    'Cortex', 'Layer',
    'Field',
    'Gate',
    'Signal',
    'Trunk',
    'Memory', 'Episode',
    'Speaker',
    'Thinker', 'WorkingMemory',
    'SemanticLandscape',
    'Synapse', 'Assembly', 'Domain',
    'Feedable', 'Queryable', 'Temporal', 'Serializable',
]


def __getattr__(name):
    """Lazy import for non-core symbols."""
    _lazy = {
        'Cortex': '.cortex', 'Layer': '.cortex',
        'Field': '.field',
        'Gate': '.gate',
        'Signal': '.signal',
        'Trunk': '.trunk',
        'Memory': '.memory', 'Episode': '.memory',
        'Speaker': '.speaker',
        'Thinker': '.thinker', 'WorkingMemory': '.thinker',
        'SemanticLandscape': '.landscape',
        'Feedable': '._protocols', 'Queryable': '._protocols',
        'Temporal': '._protocols', 'Serializable': '._protocols',
    }
    if name in _lazy:
        import importlib
        module = importlib.import_module(_lazy[name], __package__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
