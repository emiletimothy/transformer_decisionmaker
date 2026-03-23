"""
Multiplicative Weights source package.

Core implementations:
  - multiplicative_weights: Pure MW algorithm
  - learned_mw_transformer: Learned GPT-2 style transformer + tokenizer + trainers
"""

from src.multiplicative_weights import MultiplicativeWeights
from src.learned_mw_transformer import (
    LearnedMWTransformer,
    ContinuousCoTTransformer,
    ModelConfig,
    TrainingConfig,
    MWTrainer,
    ContinuousCoTTrainer,
    MWTokenizer,
    MWSequenceDataset,
    generate_mw_training_data,
    generate_sequence_with_cot,
    generate_sequence_with_continuous_cot,
    collate_fn,
)
