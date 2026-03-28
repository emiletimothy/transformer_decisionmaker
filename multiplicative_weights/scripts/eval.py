#!/usr/bin/env python3
"""
Compare weight evolution: Exact MW algorithm vs Transformer.

Creates controlled expert scenarios and plots side-by-side weight trajectories.
Supports up to three columns: Exact MW, Discrete CoT Transformer, Continuous CoT Transformer.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Tuple, Optional

from multiplicative_weights import MultiplicativeWeights
from learned_mw_transformer import (
    LearnedMWTransformer, ModelConfig, MWTokenizer,
    generate_sequence_with_cot,
    ContinuousCoTTransformer, generate_sequence_with_continuous_cot
)


# ---------------------------------------------------------------------------
# Scenario generators
# ---------------------------------------------------------------------------

def make_fixed_best_scenario(n_experts: int = 4, n_steps: int = 10,
                             best_expert: int = 0, noise: float = 0.1) -> Dict:
    """One expert is consistently the best across all rounds."""
    expert_predictions = []
    losses = []
    true_labels = []

    for step in range(n_steps):
        true_label = np.random.randint(0, 2)
        true_labels.append(true_label)

        step_preds = []
        step_losses = []
        for e in range(n_experts):
            if e == best_expert:
                # Best expert is correct with high probability
                correct = np.random.random() < (0.9 - noise)
            else:
                correct = np.random.random() < (0.4 + noise * np.random.random())
            pred = true_label if correct else 1 - true_label
            step_preds.append(pred)
            step_losses.append(0.0 if pred == true_label else 1.0)

        expert_predictions.append(step_preds)
        losses.append(step_losses)

    return {
        'expert_predictions': expert_predictions,
        'losses': losses,
        'true_labels': true_labels,
        'n_steps': n_steps,
        'name': f'Fixed Best (expert {best_expert})',
    }


def make_alternating_best_scenario(n_experts: int = 4, n_steps: int = 12,
                                   period: int = 4) -> Dict:
    """The best expert alternates every `period` steps."""
    expert_predictions = []
    losses = []
    true_labels = []

    for step in range(n_steps):
        true_label = np.random.randint(0, 2)
        true_labels.append(true_label)

        best_expert = (step // period) % n_experts

        step_preds = []
        step_losses = []
        for e in range(n_experts):
            if e == best_expert:
                correct = np.random.random() < 0.85
            else:
                correct = np.random.random() < 0.35
            pred = true_label if correct else 1 - true_label
            step_preds.append(pred)
            step_losses.append(0.0 if pred == true_label else 1.0)

        expert_predictions.append(step_preds)
        losses.append(step_losses)

    return {
        'expert_predictions': expert_predictions,
        'losses': losses,
        'true_labels': true_labels,
        'n_steps': n_steps,
        'name': f'Alternating Best (period={period})',
    }


def make_shifting_best_scenario(n_experts: int = 4, n_steps: int = 12) -> Dict:
    """The best expert gradually shifts — each expert has a window of dominance."""
    expert_predictions = []
    losses = []
    true_labels = []

    window = max(1, n_steps // n_experts)

    for step in range(n_steps):
        true_label = np.random.randint(0, 2)
        true_labels.append(true_label)

        # Smooth transition: expert quality depends on distance from its peak
        step_preds = []
        step_losses = []
        for e in range(n_experts):
            peak = (e + 0.5) * window
            dist = abs(step - peak) / window
            quality = max(0.2, 0.9 - 0.5 * dist)
            correct = np.random.random() < quality
            pred = true_label if correct else 1 - true_label
            step_preds.append(pred)
            step_losses.append(0.0 if pred == true_label else 1.0)

        expert_predictions.append(step_preds)
        losses.append(step_losses)

    return {
        'expert_predictions': expert_predictions,
        'losses': losses,
        'true_labels': true_labels,
        'n_steps': n_steps,
        'name': 'Shifting Best (gradual)',
    }


# ---------------------------------------------------------------------------
# Run MW and Transformer on a scenario
# ---------------------------------------------------------------------------

def run_exact_mw(scenario: Dict, n_experts: int) -> List[np.ndarray]:
    """Run exact MW algorithm, return weight trajectory (including initial)."""
    n_steps = scenario['n_steps']
    learning_rate = np.sqrt(np.log(n_experts) / max(n_steps, 1))
    mw = MultiplicativeWeights(n_experts, learning_rate)
    weight_trajectory = [mw.get_probabilities()]

    for step_losses in scenario['losses']:
        mw.update_weights(np.array(step_losses))
        weight_trajectory.append(mw.get_probabilities())

    return weight_trajectory


def run_transformer_discrete_cot(model, scenario: Dict, tokenizer: MWTokenizer,
                                  device: torch.device) -> List[np.ndarray]:
    """Run transformer with discrete CoT inference, return weight trajectory."""
    result = generate_sequence_with_cot(model, scenario, tokenizer, device)

    trajectory = [np.ones(tokenizer.n_experts) / tokenizer.n_experts]
    for w in result['cot_weights']:
        w = np.array(w, dtype=np.float64)
        w_sum = w.sum()
        if w_sum > 0:
            w = w / w_sum
        else:
            w = np.ones_like(w) / len(w)
        trajectory.append(w)
    return trajectory


def run_transformer_continuous_cot(model, scenario: Dict, tokenizer: MWTokenizer,
                                    device: torch.device) -> List[np.ndarray]:
    """Run transformer with continuous CoT inference, return weight trajectory."""
    result = generate_sequence_with_continuous_cot(model, scenario, tokenizer, device)

    trajectory = [np.ones(tokenizer.n_experts) / tokenizer.n_experts]
    for w in result['cot_weights']:
        w = np.array(w, dtype=np.float64)
        w_sum = w.sum()
        if w_sum > 0:
            w = w / w_sum
        else:
            w = np.ones_like(w) / len(w)
        trajectory.append(w)
    return trajectory


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_trajectory(ax, trajectory, n_experts, colors, title, marker='o'):
    """Plot a single weight trajectory on an axis."""
    steps = list(range(len(trajectory)))
    for e in range(n_experts):
        ax.plot(steps, [w[e] for w in trajectory],
                marker=marker, markersize=4, color=colors[e],
                label=f'Expert {e}')
    ax.set_title(title, fontsize=10)
    ax.set_xlabel('Step')
    ax.set_ylabel('Weight')
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)


def plot_weight_comparison(scenarios: List[Dict],
                           mw_trajectories: List[List[np.ndarray]],
                           discrete_trajectories: Optional[List[List[np.ndarray]]],
                           continuous_trajectories: Optional[List[List[np.ndarray]]],
                           n_experts: int,
                           save_path: str = None):
    """Side-by-side weight evolution plots: MW | Discrete CoT | Continuous CoT."""
    n_scenarios = len(scenarios)
    # Determine number of columns
    col_labels = ['Exact MW']
    col_trajs = [mw_trajectories]
    col_markers = ['o']
    if discrete_trajectories is not None:
        col_labels.append('Discrete CoT')
        col_trajs.append(discrete_trajectories)
        col_markers.append('s')
    if continuous_trajectories is not None:
        col_labels.append('Continuous CoT')
        col_trajs.append(continuous_trajectories)
        col_markers.append('^')
    n_cols = len(col_labels)

    fig, axes = plt.subplots(n_scenarios, n_cols,
                             figsize=(6 * n_cols, 4.5 * n_scenarios),
                             squeeze=False)
    colors = plt.cm.tab10.colors[:n_experts]

    for row, scenario in enumerate(scenarios):
        for col in range(n_cols):
            traj = col_trajs[col][row]
            title = f'{col_labels[col]} — {scenario["name"]}'
            _plot_trajectory(axes[row, col], traj, n_experts, colors,
                             title, col_markers[col])

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved to {save_path}")
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def load_model(path: str, device: torch.device):
    """Load a model checkpoint, auto-detecting discrete vs continuous."""
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model_config = checkpoint['model_config']
    cot_mode = checkpoint.get('cot_mode', 'discrete')
    tokenizer = MWTokenizer(n_experts=checkpoint['tokenizer_config']['n_experts'])

    if cot_mode == 'continuous':
        model = ContinuousCoTTransformer(model_config).to(device)
    else:
        model = LearnedMWTransformer(model_config).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, tokenizer, cot_mode, model_config


def main():
    parser = argparse.ArgumentParser(
        description='Compare weight evolution: MW vs Transformer(s)')
    parser.add_argument('--discrete_model', type=str, default=None,
                        help='Path to discrete CoT model checkpoint')
    parser.add_argument('--continuous_model', type=str, default=None,
                        help='Path to continuous CoT model checkpoint')
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to a single model (auto-detects mode)')
    parser.add_argument('--n_experts', type=int, default=4)
    parser.add_argument('--n_steps', type=int, default=8,
                        help='Number of steps per scenario')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--save_dir', type=str, default='../figures')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    # --- Load models ---
    discrete_model = discrete_tok = None
    continuous_model = continuous_tok = None

    # --model_path: auto-detect and assign to the right slot
    if args.model_path:
        m, tok, mode, mc = load_model(args.model_path, device)
        if mode == 'continuous':
            continuous_model, continuous_tok = m, tok
        else:
            discrete_model, discrete_tok = m, tok
        n_experts = tok.n_experts
        print(f"Loaded {mode} model — {n_experts} experts, "
              f"{mc.n_layers}L, {mc.d_model}d")

    if args.discrete_model:
        m, tok, _, mc = load_model(args.discrete_model, device)
        discrete_model, discrete_tok = m, tok
        n_experts = tok.n_experts
        print(f"Loaded discrete model — {n_experts} experts, "
              f"{mc.n_layers}L, {mc.d_model}d")

    if args.continuous_model:
        m, tok, _, mc = load_model(args.continuous_model, device)
        continuous_model, continuous_tok = m, tok
        n_experts = tok.n_experts
        print(f"Loaded continuous model — {n_experts} experts, "
              f"{mc.n_layers}L, {mc.d_model}d, "
              f"{mc.n_thought_steps} thought steps")

    if discrete_model is None and continuous_model is None:
        print("ERROR: Provide at least one model via --model_path, "
              "--discrete_model, or --continuous_model")
        sys.exit(1)

    # Infer n_experts from whichever model was loaded
    if 'n_experts' not in dir():
        n_experts = args.n_experts

    # --- Build scenarios (shared random seed → same data for all methods) ---
    scenarios = [
        make_fixed_best_scenario(n_experts, args.n_steps, best_expert=0),
        make_alternating_best_scenario(n_experts, args.n_steps,
                                       period=max(2, args.n_steps // 3)),
        make_shifting_best_scenario(n_experts, args.n_steps),
    ]

    mw_trajectories = []
    discrete_trajectories = [] if discrete_model else None
    continuous_trajectories = [] if continuous_model else None

    for scenario in scenarios:
        print(f"\nScenario: {scenario['name']}")

        # Exact MW
        mw_traj = run_exact_mw(scenario, n_experts)
        mw_trajectories.append(mw_traj)
        print(f"  MW final weights:          {mw_traj[-1].round(4)}")

        # Discrete CoT
        if discrete_model is not None:
            d_traj = run_transformer_discrete_cot(
                discrete_model, scenario, discrete_tok, device)
            discrete_trajectories.append(d_traj)
            print(f"  Discrete CoT final weights: {d_traj[-1].round(4)}")
            mse = np.mean([np.mean((mw_traj[i] - d_traj[i]) ** 2)
                           for i in range(min(len(mw_traj), len(d_traj)))])
            print(f"  Discrete vs MW mean MSE:    {mse:.4f}")

        # Continuous CoT
        if continuous_model is not None:
            c_traj = run_transformer_continuous_cot(
                continuous_model, scenario, continuous_tok, device)
            continuous_trajectories.append(c_traj)
            print(f"  Continuous CoT final weights: {c_traj[-1].round(4)}")
            mse = np.mean([np.mean((mw_traj[i] - c_traj[i]) ** 2)
                           for i in range(min(len(mw_traj), len(c_traj)))])
            print(f"  Continuous vs MW mean MSE:    {mse:.4f}")

    # --- Plot ---
    save_path = os.path.join(args.save_dir, 'weight_evolution_comparison.png')
    plot_weight_comparison(scenarios, mw_trajectories,
                           discrete_trajectories, continuous_trajectories,
                           n_experts, save_path)

    print(f"\nDone. Plot saved to {save_path}")


if __name__ == '__main__':
    main()
