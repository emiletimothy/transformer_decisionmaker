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
├── tabular_q_learning/     # Q-learning extension (see section below)
│   ├── scripts/                    # Pipeline + analysis scripts
│   ├── data/                       # Generated trajectory datasets
│   ├── checkpoints/                # Trained transformer weights
│   └── figures/                    # Probe / attention / regret plots
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

## Tabular Q-Learning

The `tabular_q_learning/` subdirectory extends the project from online prediction
(multiplicative weights) to **online off-policy control** in finite MDPs. It
contains a tabular Q-learning baseline, a hand-wired transformer construction
that exactly reproduces the tabular update, and a learned recurrent-context
transformer that acquires Q-learning behavior end-to-end via gradient descent.

### Setting

We consider finite MDPs with `|S|` states and `|A|` actions. Trajectories are
generated by an ε-greedy policy over a chain MDP (or randomized variants for
OOD evaluation). The reference algorithm is **ε-greedy tabular Q-learning**:

```
Q(s_t, a_t) ← Q(s_t, a_t) + α · (r_t + γ · max_a Q(s_{t+1}, a) − Q(s_t, a_t))
```

with all other `(s, a)` entries unchanged each step. The Q-table is initialized
to zero. This update is the per-step target the transformer is asked to track.

### Components

- **`tabular_q_learning.py`** — Reference tabular learner. Maintains a
  `(|S|, |A|)` Q-table, applies the off-policy TD update one transition at a
  time, and exposes a Q-history for lock-step comparison with the transformer.
  Also defines the chain MDP and ε-greedy trajectory generator used to produce
  training data.
- **`transformer_handwired_q_learning.py`** — Hand-constructed transformer
  whose attention heads, projections, and FFNs are weight-engineered to
  implement the tabular update **exactly**. Serves as a constructive existence
  proof that the architecture is expressive enough to realize Q-learning.
- **`learned_qlearning_transformer.py` + `2_model.py`** — Recurrent-context
  transformer (`COCONUTTransformer`) trained to produce the same Q-table
  evolution from trajectory tokens. Each step consumes discrete tokens for
  `(s_t, a_t, r_t, s_{t+1})` together with `n_actions` continuous **context
  tokens** that carry the running Q-state across steps. The hidden state at
  the `UPDATE` position is written back into context slot `a_t`, mirroring
  the table-write of tabular Q-learning.

### Pipeline

The numbered scripts in `tabular_q_learning/scripts/` form an end-to-end
pipeline:

```bash
cd tabular_q_learning/scripts
python 1_generate_data.py     # Sample MDPs + ε-greedy trajectories → data/
python 2_model.py             # Model + config (importable; prints summary)
python 3_train.py             # Train COCONUTTransformer on (s,a,r,s') sequences
python 4_evaluate.py          # Probes, attention, regret, α/γ recovery → figures/
```

Auxiliary tools:

- `compare_q_learning_transformer.py` — Side-by-side Q-table evolution between
  the tabular learner and the (hand-wired or learned) transformer on a shared
  trajectory.
- `show_tokenization.py`, `inspect_dataset.py` — Debugging utilities for the
  token layout and dataset contents.
- `diagnose_one_step.py` — Single-step forward-pass diagnostics.

### Evaluation (`4_evaluate.py`)

`4_evaluate.py` produces all analysis figures in `tabular_q_learning/figures/`:

1. **Action prediction (ID + OOD).** Per-step greedy-action agreement against
   the tabular learner on in-distribution and out-of-distribution MDPs.
   → `action_agreement.png`, `per_state_agreement.png`
2. **Context-token Q-probe.** Trains a linear probe `c_a → [Q(s_1,a), …, Q(s_{|S|},a)]`
   on each context token and reports R² and Frobenius error vs. the tabular
   target. A second **bias-free** probe is trained alongside, which maps a
   zero context exactly to zero Q-values (removing the cosmetic step-0 offset
   produced by the bias term).
   → `probe_scatter.png`, `probe_frobenius.png`,
     `probe_scatter_nobias.png`, `probe_frobenius_nobias.png`
3. **Attention heatmaps.** Verifies that `SELECT` attends to `EVAL` tokens and
   `UPDATE` attends to context + `QCURR/QNEXT`, matching the hand-wired roles.
   → `attention_heatmap.png`, `attention_full_heatmap_4s2a.png`
4. **Closed-loop regret.** Runs the transformer autonomously on fresh MDPs
   and compares cumulative reward against greedy and ε-greedy tabular
   Q-learners.
   → `regret.png`, `long_horizon.png`
5. **Effective α/γ recovery.** Fits `(α_eff, γ_eff)` per trajectory from the
   probe-decoded Q dynamics — recovers the training-time hyperparameters when
   the model has internalized the update rule.
   → `effective_alpha_gamma.png`
6. **Reward probe from context delta.** Linear probe on
   `Δc_{a_t} = update_hidden − context[a_t]` predicting `r_t`, testing whether
   the trained model preserves the hand-wired reward-injection pathway.
   → `reward_probe.png`

### Key Findings

- **Constructive expressivity.** The hand-wired transformer reproduces the
  tabular update bit-for-bit, confirming that the recurrent-context layout
  is sufficient to realize ε-greedy Q-learning.
- **Behavioral match.** The learned transformer attains high greedy-action
  agreement with the tabular learner ID and degrades gracefully OOD.
- **Decodable Q-state.** A linear probe on context tokens recovers the
  tabular Q-values with high R², so the model encodes Q-values approximately
  linearly in the context.
- **Recovered hyperparameters.** Per-trajectory fits of `(α_eff, γ_eff)` from
  the probe-decoded dynamics align with the training-time `(α, γ)`,
  evidence that the learned dynamics implement (an approximation of) the
  Q-learning update rather than a different value-tracking heuristic.
