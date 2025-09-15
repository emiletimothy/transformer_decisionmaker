"""
Transformer Decision Maker - Core Modules

This package contains the core implementations of multiplicative weights algorithms
and transformer architectures for online decision making.
"""

from .multiplicative_weights import MultiplicativeWeights
from .additive_weights import AdditiveMultiplicativeWeights

# Optional torch-based transformers (requires PyTorch)
try:
    from .transformer_mw import MultiplicativeWeightsTransformer
    from .learned_mw_transformer import LearnedMWTransformer, ModelConfig, TrainingConfig
    _has_torch = True
except ImportError:
    _has_torch = False
    MultiplicativeWeightsTransformer = None
    LearnedMWTransformer = None
    ModelConfig = None
    TrainingConfig = None

__all__ = [
    'MultiplicativeWeights',
    'AdditiveMultiplicativeWeights'
]

if _has_torch:
    __all__.extend([
        'MultiplicativeWeightsTransformer',
        'LearnedMWTransformer',
        'ModelConfig',
        'TrainingConfig'
    ])
