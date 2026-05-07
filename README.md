# Transformer Decision Maker

Your Transformer is (maybe not so secretly) an online decision maker! This repository
studies how transformer architectures realize classical online-learning and
control algorithms — multiplicative weights for prediction with expert advice,
tabular Q-learning for online control, and prompted LLMs as black-box online
decision makers.

The repo is organized as three self-contained projects:

| Subdirectory             | Topic                                      | Approach |
|--------------------------|--------------------------------------------|----------|
| `multiplicative_weights/`| Prediction with expert advice (MW)         | Hand-wired transformer + learned (GPT-style) transformer |
| `tabular_q_learning/`    | Online off-policy control in finite MDPs   | Hand-wired transformer + learned recurrent-context transformer |
| `llm/`                   | Online prediction with expert advice via LLM prompting | DeepSeek V3 / Qwen3 evaluated as agents |

## Project Structure

```
transformer_decisionmaker/
├── multiplicative_weights/         # MW project
│   ├── scripts/                        # Training, eval, attention analysis
│   │   ├── multiplicative_weights.py        # Classical MW reference
│   │   ├── transformer_handwired_multiplicative_weights.py  # Hand-wired transformer
│   │   ├── learned_mw_transformer.py        # Learned MW model (GPT-style)
│   │   ├── gradient_train_transformer_for_mwu.py  # Curriculum training loop
│   │   ├── generate_dataset.py / generate_eval_dataset.py
│   │   ├── eval.py / evaluate_model.py / eval_long_sequences.py
│   │   ├── eval_robustness.py
│   │   ├── eval_attention*.py               # Attention-pattern visualizations
│   │   ├── run_aws_train.sh / run_train_and_eval.bash
│   │   └── show_tokenization.py
│   ├── data/                           # Generated MW datasets
│   └── figures/                        # Training logs, checkpoints, plots
│       ├── checkpoints/                    # Trained model weights
│       ├── attention-figures/              # Per-stage attention heatmaps
│       └── eval*/, eval_robustness*/       # Evaluation artifacts
│
├── tabular_q_learning/             # Q-learning project
│   ├── scripts/
│   │   ├── tabular_q_learning.py            # Reference tabular learner + chain MDP
│   │   ├── transformer_handwired_q_learning.py  # Weight-engineered transformer
│   │   ├── learned_qlearning_transformer.py     # Recurrent-context transformer
│   │   ├── 1_generate_data.py               # Trajectory generation
│   │   ├── 2_model.py                       # Model definition
│   │   ├── 3_train.py                       # Training entrypoint
│   │   ├── 4_evaluate.py                    # Evaluation + probes
│   │   ├── compare_q_learning_transformer.py
│   │   ├── diagnose_one_step.py / inspect_dataset.py
│   │   ├── run_experiment.sh / run_compare_q_learning.sh
│   │   └── show_tokenization.py
│   └── figures/                         # Probe / attention / regret plots
│
├── llm/                            # LLM-as-online-learner project
│   ├── core/
│   │   ├── interactive_api_run.py           # Main experiment runner (API + vLLM)
│   │   ├── mw_lib.py                        # MW algorithm, analysis, plotting
│   │   ├── generate_dataset.py              # Expert-prediction dataset generator
│   │   ├── plot_regret.py
│   │   ├── prompts/                         # Prompt templates (weather/online × note/no-note)
│   │   └── mw_dataset*.json                 # Stratified / flat / anti-signal regimes
│   ├── exp_results*/                    # Per-regime experiment outputs
│   ├── plots/, docs/, archive/
│   ├── plot_regret*.py / mw_lib.py      # Top-level convenience copies
│   └── run_*.sh                         # Batch runners (DeepSeek, Qwen, antisignal, …)
│
├── requirements.txt                # Python dependencies
├── LICENSE
└── README.md                       # This file
```

## Installation

```bash
pip install -r requirements.txt
```

The `llm/` subproject has additional optional dependencies (`openai`, `vllm`,
`openpyxl`); see `llm/core/README.md` for details.

---

## 1. Multiplicative Weights (`multiplicative_weights/`)

A study of **prediction with expert advice** through three lenses:

- **Classical MW.** `scripts/multiplicative_weights.py` is the reference
  implementation. Per round: weights are updated by `w_i ← w_i · exp(-η · ℓ_i)`
  and renormalized. Regret bound `O(√T log n)`.
- **Hand-wired transformer.** `scripts/transformer_handwired_multiplicative_weights.py`
  realizes the MW update with a 2-layer transformer whose attention heads are
  weight-engineered to do expert-advice loading, weight copying, label copying,
  and softmax aggregation — a constructive existence proof.
- **Learned transformer.** `scripts/learned_mw_transformer.py` defines a
  GPT-style decoder (d_model=768, 8 heads, 2 layers) trained end-to-end via a
  multi-stage curriculum
  (`scripts/gradient_train_transformer_for_mwu.py`):
  - Stage *i*: learn *i*-step MW reasoning sequences
  - Mix previous-stage data with probability 0.1
  - 25 epochs per stage, up to 12 stages
  - AdamW (β₁=0.9, β₂=0.95, wd=10⁻²), lr=10⁻⁴

Tokenization is discrete (experts, weights, losses, predictions, step markers).
The model is supervised on both weight-prediction and binary-decision targets.

### Quick start

```bash
cd multiplicative_weights/scripts

# Generate dataset
python generate_dataset.py

# Train the learned MW transformer (GPU recommended)
bash run_train_and_eval.bash

# Evaluate a checkpoint
python eval.py
python eval_long_sequences.py
python eval_robustness.py

# Visualize learned attention patterns
python eval_attention_heatmaps.py
python eval_attention_expert_focus.py
```

Outputs (logs, checkpoints, per-stage eval JSON, attention heatmaps) land in
`multiplicative_weights/figures/`.

---

## 2. Tabular Q-Learning (`tabular_q_learning/`)

Extends the project from online prediction to **online off-policy control** in
finite MDPs. We consider chain MDPs (with randomized variants for OOD eval)
and ε-greedy trajectories.

Reference algorithm — ε-greedy tabular Q-learning:
```
Q(s_t, a_t) ← Q(s_t, a_t) + α · (r_t + γ · max_a Q(s_{t+1}, a) − Q(s_t, a_t))
```

### Components

- **`tabular_q_learning.py`** — Reference learner, chain MDP, trajectory generator.
- **`transformer_handwired_q_learning.py`** — Hand-engineered transformer that
  reproduces the tabular update bit-for-bit.
- **`learned_qlearning_transformer.py` + `2_model.py`** — `COCONUTTransformer`,
  a recurrent-context architecture trained to track the same Q-table evolution
  from trajectory tokens.

### Pipeline

```bash
cd tabular_q_learning/scripts

python 1_generate_data.py        # Generate trajectory dataset
python 3_train.py                # Train COCONUTTransformer
python 4_evaluate.py             # Greedy-action match, linear probes, (α,γ) recovery
python compare_q_learning_transformer.py   # Side-by-side comparison
bash run_experiment.sh           # Full pipeline driver
```

Plots (probes, attention, regret) are written to `tabular_q_learning/figures/`.

### Headline results

- **Constructive equivalence.** The hand-wired transformer reproduces the
  tabular update exactly.
- **Behavioral match.** The learned transformer attains high greedy-action
  agreement with the tabular learner ID, degrading gracefully OOD.
- **Decodable Q-state.** A linear probe on context tokens recovers the tabular
  Q-values with high R²: Q-values are encoded approximately linearly in context.
- **Recovered hyperparameters.** Per-trajectory fits of `(α_eff, γ_eff)` from
  the probe-decoded dynamics align with the training-time `(α, γ)`, evidence
  that the model implements (an approximation of) the Q-learning rule rather
  than a different value-tracking heuristic.

---

## 3. LLM as Online Learner (`llm/`)

Treats a frozen LLM as a black-box online decision maker on the prediction-
with-expert-advice problem. Across four prompting conditions (weather vs.
online framing, with vs. without a running "note"), DeepSeek V3 and a local
Qwen3-14B are evaluated against MW and best-expert baselines on three
loss regimes (stratified, flat, anti-signal).

### Quick start

```bash
cd llm/core

# Generate datasets (or use provided JSON files)
python generate_dataset.py --all

# DeepSeek V3 (requires DEEPSEEK_API_KEY)
python interactive_api_run.py cases \
  --idx $(seq 0 29) \
  --model ds \
  --prompt interactive_weather_notehist \
  --dataset mw_dataset.json \
  --steps 100

# Plot regret curves
python plot_regret.py
```

For local Qwen3-14B inference, see `llm/core/README.md` for the `vllm serve`
command. Convenience batch scripts (`run_overnight.sh`, `run_qwen_all.sh`,
`run_antisignal.sh`, …) live at `llm/`.

---

## Headline results across projects

- **MW realization.** The hand-wired transformer is exact; the learned
  transformer reaches ~99.8% performance ratio versus classical MW.
- **Regret bounds.** `O(√T log n)` regret is preserved through the learned
  architecture.
- **Attention ↔ algorithm.** Attention patterns line up with MW / Q-learning
  primitives (expert loading, weight copying, softmax aggregation, TD update).
- **Q-learning realization.** Constructive equivalence + learned
  approximation, with linearly decodable Q-state and recoverable `(α, γ)`.
- **LLM agents.** Prompted LLMs interpolate between best-expert imitation and
  MW depending on framing and note availability, regime-dependent.

