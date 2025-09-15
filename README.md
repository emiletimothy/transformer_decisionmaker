# Transformer Decision Maker

Your Transformer is (maybe not so secretly) an online decision maker! This repository implements and analyzes multiplicative weights algorithms through both classical and transformer-based approaches.

## Project Structure

```
transformer_decisionmaker/
├── src/                    # Core modules and implementations
│   ├── multiplicative_weights.py    # Classical MW algorithm
│   ├── additive_weights.py          # Additive variant implementation
│   ├── transformer_mw.py            # Transformer-based MW
│   ├── learned_mw_transformer.py    # Learned MW via gradient training
│   └── __init__.py                  # Package initialization
├── scripts/                # Executable scripts and demos
│   ├── examples.py                  # Basic MW examples
│   ├── numpy_transformer_demo.py    # Numpy transformer demo
│   ├── transformer_demo.py          # Full transformer demo
│   ├── train_learned_mw.py          # Train learned MW transformer
│   ├── analyze_learned_mw.py        # Analyze learned model
│   └── theoretical_analysis.py      # Theoretical validation
├── figures/                # Generated plots and visualizations
│   ├── numpy_transformer_comparison.png
│   ├── regret_bounds_analysis.png
│   └── attention_patterns_analysis.png
├── run_demos.py            # Main runner script for all demos
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## Multiplicative Weights Algorithm

This repository implements the **Multiplicative Weights Algorithm**, a fundamental online learning algorithm for decision making under uncertainty, and demonstrates how transformer architectures can realize the same algorithmic behavior.

### Key Features

- **Classical Implementation**: Standard multiplicative weights with theoretical guarantees
- **Transformer Realization**: Neural architecture that implements MW through attention
- **Learned MW**: GPT-2 style transformer trained to learn MW updates via gradient descent
- **Online Learning**: Adapts to changing environments without knowing the future
- **Regret Minimization**: Theoretical guarantees on performance vs best expert in hindsight
- **Multiple Applications**: Portfolio selection, expert aggregation, multi-armed bandits
- **Comprehensive Analysis**: Theoretical validation and empirical comparisons

### Quick Start

```python
from src.multiplicative_weights import MultiplicativeWeights

# Create algorithm with 4 experts
mw = MultiplicativeWeights(num_experts=4, learning_rate=0.1)

# In each round:
selected_expert = mw.select_expert(method='sample')
losses = [0.2, 0.1, 0.8, 0.3]  # Observe losses for all experts
mw.update_weights(losses)

# View current probabilities
print(mw.get_probabilities())
```

### Installation

```bash
pip install -r requirements.txt
```

### Running Examples

#### Quick Start - Run All Demos
```bash
python run_demos.py                 # Run all demonstrations
python run_demos.py --list          # List available demos
python run_demos.py examples        # Run specific demo
```

#### Individual Scripts
```bash
cd scripts
python examples.py                  # Basic MW examples
python numpy_transformer_demo.py    # Numpy-based transformer
python transformer_demo.py          # Full PyTorch transformer
python train_learned_mw.py          # Train learned MW transformer
python theoretical_analysis.py      # Regret bounds and validation
```

### What You'll See

- **Basic Examples**: Classical MW on various problems (portfolio, expert advice, bandits)
- **Transformer Demos**: How attention mechanisms implement multiplicative weight updates
- **Learned MW Training**: Multi-stage curriculum learning for MW algorithm acquisition
- **Performance Comparisons**: Side-by-side analysis of classical vs transformer approaches
- **Theoretical Validation**: Regret bound analysis and attention pattern visualization

## Algorithm Details

### Classical Multiplicative Weights

The multiplicative weights update rule:
```
w_i ← w_i × exp(-η × loss_i)
```

Where `η` is the learning rate and weights are normalized to sum to 1.

**Regret Bound**: O(√T log n) where T is number of rounds and n is number of experts.

### Transformer Realization

The transformer architecture implements MW through:
1. **Layer 1**: Expert advice loading, weight copying, label copying via specialized attention heads
2. **Layer 2**: Softmax aggregation and multiplicative weight updates through attention mechanisms
3. **Token Embeddings**: Structured representation of experts, weights, losses, and updates

This demonstrates that transformers can implement classical online learning algorithms while maintaining their theoretical properties.

### Learned Multiplicative Weights

The learned MW approach trains a GPT-2 style transformer to acquire MW behavior through gradient-based learning:

**Architecture**: 2-layer transformer with d_model=768, n_heads=8, following GPT-2 design principles

**Training Strategy**: Multi-stage curriculum learning similar to chain-of-thought training:
- Stage i: Learn to perform i-step MW reasoning sequences
- Mix previous stage data with probability 0.1
- Train for 25 epochs per stage, up to 12 stages total
- AdamW optimizer (β₁=0.9, β₂=0.95, weight_decay=10⁻²), lr=10⁻⁴

**Tokenization**: Discrete tokens for experts, weights, losses, predictions, and step markers

**Supervision**: Multi-task learning with weight prediction and binary decision targets

## Key Results

- **Performance Equivalence**: Transformer MW achieves ~99.8% performance ratio vs classical MW
- **Regret Guarantees**: Theoretical bounds are preserved through the neural architecture
- **Attention Patterns**: Visualizable correspondence between attention and MW operations
- **Scalability**: Works across different numbers of experts and time horizons
