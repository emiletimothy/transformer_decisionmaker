# transformer_decisionmaker
Your Transformer is (maybe not so secretly) an online decision maker

## Multiplicative Weights Algorithm

This repository implements the **Multiplicative Weights Algorithm**, a fundamental online learning algorithm for decision making under uncertainty. The algorithm maintains and updates weights over a set of experts/actions based on observed losses, giving higher weight to better-performing experts over time.

### Key Features

- **Online Learning**: Adapts to changing environments without knowing the future
- **Regret Minimization**: Theoretical guarantees on performance vs best expert in hindsight
- **Multiple Applications**: Portfolio selection, expert aggregation, multi-armed bandits
- **Visualization**: Built-in plotting for weight evolution and performance analysis

### Quick Start

```python
from multiplicative_weights import MultiplicativeWeights

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

### Examples

Run comprehensive examples:
```bash
python examples.py
```

This includes:
- Basic usage and weight updates
- Online portfolio selection
- Expert advice aggregation  
- Adversarial loss scenarios
- Multi-armed bandit problems

### Algorithm Details

The multiplicative weights update rule:
```
w_i ← w_i × exp(-η × loss_i)
```

Where `η` is the learning rate and weights are normalized to sum to 1.

**Regret Bound**: O(√T log n) where T is number of rounds and n is number of experts.
