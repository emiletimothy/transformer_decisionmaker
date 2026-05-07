#!/usr/bin/env python3
"""
Robustness evaluation: test the learned MW transformer under diverse scenarios.

Scenarios:
  1. Varying effective experts (1-4 real experts, rest are random coin-flips)
  2. Expert quality distributions (dominant, close, spread)
  3. Non-stationary best experts (switching, gradual drift)
  4. Correlated experts

Reuses the eval/regret infrastructure from eval_long_sequences.py.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple
import argparse
import logging

from learned_mw_transformer import (
    ContinuousCoTTransformer, LearnedMWTransformer,
    ModelConfig, MWTokenizer,
)
from eval_long_sequences import (
    generate_long_sequence_continuous_cot,
    get_optimal_mw_decisions,
    compute_final_regret,
    compute_regret_trajectory,
    load_model,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Custom sequence generators
# ---------------------------------------------------------------------------

def generate_sequence_n_real_experts(
    n_experts: int, n_steps: int, n_real: int
) -> Dict:
    """Generate a sequence where only `n_real` experts are meaningful.
    
    Real experts get quality in [0.6, 0.85]. Dummy experts are coin-flips (0.5).
    """
    qualities = [0.5] * n_experts
    real_indices = np.random.choice(n_experts, size=n_real, replace=False)
    for idx in real_indices:
        qualities[idx] = np.random.uniform(0.6, 0.85)

    return _generate_from_qualities(n_experts, n_steps, qualities)


def generate_sequence_quality_dist(
    n_experts: int, n_steps: int, mode: str
) -> Dict:
    """Generate sequences with different quality distributions.
    
    Modes:
      - 'dominant': one expert at 0.9, rest at 0.35-0.5
      - 'close':    all experts between 0.55-0.65
      - 'spread':   standard uniform [0.3, 0.9]
    """
    if mode == 'dominant':
        qualities = [np.random.uniform(0.35, 0.5) for _ in range(n_experts)]
        qualities[np.random.randint(n_experts)] = 0.9
    elif mode == 'close':
        qualities = [np.random.uniform(0.55, 0.65) for _ in range(n_experts)]
    elif mode == 'spread':
        qualities = [np.random.uniform(0.3, 0.9) for _ in range(n_experts)]
    else:
        raise ValueError(f"Unknown quality mode: {mode}")

    return _generate_from_qualities(n_experts, n_steps, qualities)


def generate_sequence_nonstationary(
    n_experts: int, n_steps: int, mode: str, switch_every: int = 25
) -> Dict:
    """Generate sequences where the best expert changes over time.
    
    Modes:
      - 'switching': best expert abruptly changes every `switch_every` steps
      - 'gradual':   expert qualities drift smoothly over time
    """
    if mode == 'switching':
        return _generate_switching(n_experts, n_steps, switch_every)
    elif mode == 'gradual':
        return _generate_gradual_drift(n_experts, n_steps)
    else:
        raise ValueError(f"Unknown nonstationary mode: {mode}")


def generate_sequence_correlated(
    n_experts: int, n_steps: int, correlation: float = 0.7
) -> Dict:
    """Generate sequences with correlated expert predictions.
    
    With probability `correlation`, all experts make the same prediction
    (either all correct or all wrong). Otherwise independent.
    """
    expert_qualities = [np.random.uniform(0.4, 0.8) for _ in range(n_experts)]

    expert_predictions = []
    losses = []
    true_labels = []

    for step in range(n_steps):
        true_label = np.random.randint(0, 2)
        true_labels.append(true_label)

        if np.random.random() < correlation:
            # Correlated: all experts agree on one prediction
            shared_correct = np.random.random() < np.mean(expert_qualities)
            shared_pred = true_label if shared_correct else 1 - true_label
            step_preds = [shared_pred] * n_experts
        else:
            # Independent
            step_preds = []
            for e in range(n_experts):
                correct = np.random.random() < expert_qualities[e]
                pred = true_label if correct else 1 - true_label
                step_preds.append(pred)

        step_losses = [0.0 if p == true_label else 1.0 for p in step_preds]
        expert_predictions.append(step_preds)
        losses.append(step_losses)

    return {
        'expert_predictions': expert_predictions,
        'losses': losses,
        'true_labels': true_labels,
        'n_steps': n_steps,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _generate_from_qualities(n_experts, n_steps, qualities):
    """Generate a standard sequence from fixed expert qualities."""
    expert_predictions = []
    losses = []
    true_labels = []

    for step in range(n_steps):
        true_label = np.random.randint(0, 2)
        true_labels.append(true_label)

        step_preds = []
        step_losses = []
        for e in range(n_experts):
            correct = np.random.random() < qualities[e]
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
    }


def _generate_switching(n_experts, n_steps, switch_every):
    """Best expert abruptly switches every `switch_every` steps."""
    expert_predictions = []
    losses = []
    true_labels = []

    # Pre-assign which expert is best in each block
    n_blocks = (n_steps + switch_every - 1) // switch_every
    best_per_block = [np.random.randint(n_experts) for _ in range(n_blocks)]

    base_quality = 0.4  # non-best experts
    best_quality = 0.85  # current best expert

    for step in range(n_steps):
        block = step // switch_every
        best_expert = best_per_block[block]

        true_label = np.random.randint(0, 2)
        true_labels.append(true_label)

        step_preds = []
        step_losses = []
        for e in range(n_experts):
            q = best_quality if e == best_expert else base_quality
            correct = np.random.random() < q
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
    }


def _generate_gradual_drift(n_experts, n_steps):
    """Expert qualities drift smoothly via random walks."""
    # Initial qualities
    qualities = np.array([np.random.uniform(0.4, 0.8) for _ in range(n_experts)])

    expert_predictions = []
    losses = []
    true_labels = []

    for step in range(n_steps):
        true_label = np.random.randint(0, 2)
        true_labels.append(true_label)

        step_preds = []
        step_losses = []
        for e in range(n_experts):
            correct = np.random.random() < qualities[e]
            pred = true_label if correct else 1 - true_label
            step_preds.append(pred)
            step_losses.append(0.0 if pred == true_label else 1.0)

        expert_predictions.append(step_preds)
        losses.append(step_losses)

        # Random walk on qualities
        qualities += np.random.normal(0, 0.02, n_experts)
        qualities = np.clip(qualities, 0.2, 0.95)

    return {
        'expert_predictions': expert_predictions,
        'losses': losses,
        'true_labels': true_labels,
        'n_steps': n_steps,
    }


# ---------------------------------------------------------------------------
# Run a single scenario
# ---------------------------------------------------------------------------

def run_scenario(model, tokenizer, cot_mode, model_config, device,
                 seq_generator, n_trials, seq_length, scenario_name):
    """Run one scenario and return summary metrics."""
    max_ctx = model_config.max_sequence_length

    learned_regrets = []
    optimal_regrets = []
    accuracies = []
    learned_trajs = []
    optimal_trajs = []

    for trial in range(n_trials):
        seq = seq_generator()

        if cot_mode == 'continuous':
            result = generate_long_sequence_continuous_cot(
                model, seq, tokenizer, device, max_context_tokens=max_ctx
            )
            learned_decisions = result['decisions']
        else:
            raise NotImplementedError("Only continuous CoT supported")

        optimal_decisions = get_optimal_mw_decisions(seq)

        true_labels = np.array(seq['true_labels'])
        n = min(len(learned_decisions), len(true_labels))
        accuracy = np.mean(learned_decisions[:n] == true_labels[:n])
        accuracies.append(accuracy)

        lr = compute_final_regret(seq, learned_decisions)
        opr = compute_final_regret(seq, optimal_decisions)
        learned_regrets.append(lr)
        optimal_regrets.append(opr)

        l_traj, o_traj = compute_regret_trajectory(seq, learned_decisions, optimal_decisions)
        learned_trajs.append(l_traj)
        optimal_trajs.append(o_traj)

        if (trial + 1) % 10 == 0:
            logger.info(f"  [{scenario_name}] trial {trial+1}/{n_trials}")

    learned_regrets = np.array(learned_regrets)
    optimal_regrets = np.array(optimal_regrets)
    accuracies = np.array(accuracies)

    ratios = []
    for lr, opr in zip(learned_regrets, optimal_regrets):
        if opr > 1e-6:
            ratios.append(lr / opr)
        elif lr < 1e-6:
            ratios.append(1.0)
        else:
            ratios.append(10.0)
    ratios = np.array(ratios)

    return {
        'scenario': scenario_name,
        'learned_regrets': learned_regrets,
        'optimal_regrets': optimal_regrets,
        'accuracies': accuracies,
        'ratios': ratios,
        'learned_traj_mean': np.mean(learned_trajs, axis=0),
        'optimal_traj_mean': np.mean(optimal_trajs, axis=0),
        'learned_traj_std': np.std(learned_trajs, axis=0),
        'optimal_traj_std': np.std(optimal_trajs, axis=0),
        'n_trials': len(learned_trajs),
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_all_scenarios(all_results, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    # --- Bar chart: learned vs optimal regret per scenario ---
    scenario_names = [r['scenario'] for r in all_results]
    learned_means = [r['learned_regrets'].mean() for r in all_results]
    learned_stds = [r['learned_regrets'].std() for r in all_results]
    optimal_means = [r['optimal_regrets'].mean() for r in all_results]
    optimal_stds = [r['optimal_regrets'].std() for r in all_results]
    acc_means = [r['accuracies'].mean() for r in all_results]
    acc_stds = [r['accuracies'].std() for r in all_results]
    ratio_means = [r['ratios'].mean() for r in all_results]
    ratio_stds = [r['ratios'].std() for r in all_results]

    fig, axes = plt.subplots(1, 3, figsize=(22, 7))
    x = np.arange(len(scenario_names))
    width = 0.35

    # Regret
    axes[0].bar(x - width/2, learned_means, width, yerr=learned_stds,
                label='Learned', color='red', alpha=0.7, capsize=3)
    axes[0].bar(x + width/2, optimal_means, width, yerr=optimal_stds,
                label='Optimal MW', color='blue', alpha=0.7, capsize=3)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(scenario_names, rotation=35, ha='right', fontsize=8)
    axes[0].set_ylabel('Final Regret')
    axes[0].set_title('Final Regret by Scenario')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Ratio
    axes[1].bar(x, ratio_means, 0.5, yerr=ratio_stds, color='green', alpha=0.7, capsize=3)
    axes[1].axhline(y=1.0, color='k', linestyle='--', alpha=0.5)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(scenario_names, rotation=35, ha='right', fontsize=8)
    axes[1].set_ylabel('Regret Ratio (Learned / Optimal)')
    axes[1].set_title('Regret Ratio by Scenario')
    axes[1].grid(True, alpha=0.3)

    # Accuracy
    axes[2].bar(x, acc_means, 0.5, yerr=acc_stds, color='purple', alpha=0.7, capsize=3)
    axes[2].axhline(y=0.5, color='k', linestyle='--', alpha=0.5, label='Random')
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(scenario_names, rotation=35, ha='right', fontsize=8)
    axes[2].set_ylabel('Accuracy')
    axes[2].set_title('Prediction Accuracy by Scenario')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, 'robustness_summary.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved {path}")

    # --- Regret trajectory comparison for select scenarios ---
    n_scenarios = len(all_results)
    n_cols = 3
    n_rows = (n_scenarios + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 5 * n_rows))
    axes = axes.flatten() if n_rows * n_cols > 1 else [axes]
    for i, r in enumerate(all_results):
        if i >= len(axes):
            break
        ax = axes[i]
        steps = np.arange(1, len(r['learned_traj_mean']) + 1)
        n_t = r.get('n_trials', 50)
        stderr_scale = 1.0 / np.sqrt(n_t)  # standard error

        l_mean = r['learned_traj_mean']
        l_se = r['learned_traj_std'] * stderr_scale
        o_mean = r['optimal_traj_mean']
        o_se = r['optimal_traj_std'] * stderr_scale

        ax.plot(steps, l_mean, 'r-', linewidth=2, label='Learned')
        ax.fill_between(steps, l_mean - l_se, l_mean + l_se, color='red', alpha=0.15)
        ax.plot(steps, o_mean, 'b-', linewidth=2, label='Optimal MW')
        ax.fill_between(steps, o_mean - o_se, o_mean + o_se, color='blue', alpha=0.15)
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Cumulative Regret')
        ax.set_title(r['scenario'], fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    # Hide unused axes
    for j in range(n_scenarios, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle(f'Regret Trajectories (mean \u00b1 SE, n={all_results[0].get("n_trials", "?")} trials)',
                 fontsize=14, y=1.01)
    plt.tight_layout()
    path2 = os.path.join(save_dir, 'robustness_trajectories.png')
    plt.savefig(path2, dpi=200, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved {path2}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Robustness evaluation of MW transformer')
    parser.add_argument('--checkpoint', type=str,
                        default='../figures/checkpoints/model_stage_13.pt')
    parser.add_argument('--seq_length', type=int, default=100,
                        help='Sequence length for all scenarios')
    parser.add_argument('--n_trials', type=int, default=50)
    parser.add_argument('--n_experts', type=int, default=4)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--save_dir', type=str, default='../figures/eval_robustness')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    n = args.n_experts
    L = args.seq_length

    logger.info(f"Loading checkpoint: {args.checkpoint}")
    model, tokenizer, cot_mode, model_config = load_model(args.checkpoint, device)
    logger.info(f"Model: {cot_mode}, seq_length={L}, n_trials={args.n_trials}")

    # Define all scenarios as (name, generator_fn) pairs
    scenarios = []

    # 1. Varying effective experts
    for n_real in [1, 2, 3, 4]:
        scenarios.append((
            f'{n_real} real expert{"s" if n_real > 1 else ""}',
            lambda nr=n_real: generate_sequence_n_real_experts(n, L, nr),
        ))

    # 2. Quality distributions
    for mode in ['dominant', 'close', 'spread']:
        scenarios.append((
            f'quality: {mode}',
            lambda m=mode: generate_sequence_quality_dist(n, L, m),
        ))

    # 3. Non-stationary
    for switch_every in [10, 25, 50]:
        scenarios.append((
            f'switch every {switch_every}',
            lambda se=switch_every: generate_sequence_nonstationary(n, L, 'switching', se),
        ))
    scenarios.append((
        'gradual drift',
        lambda: generate_sequence_nonstationary(n, L, 'gradual'),
    ))

    # 4. Correlated experts
    for corr in [0.3, 0.7, 0.9]:
        scenarios.append((
            f'correlation={corr}',
            lambda c=corr: generate_sequence_correlated(n, L, c),
        ))

    # Run all scenarios
    all_results = []
    for name, gen_fn in scenarios:
        logger.info(f"\n{'='*60}")
        logger.info(f"Scenario: {name}")
        logger.info(f"{'='*60}")

        result = run_scenario(
            model, tokenizer, cot_mode, model_config, device,
            gen_fn, args.n_trials, L, name,
        )
        all_results.append(result)

        logger.info(f"  Learned regret: {result['learned_regrets'].mean():.2f} ± {result['learned_regrets'].std():.2f}")
        logger.info(f"  Optimal regret: {result['optimal_regrets'].mean():.2f} ± {result['optimal_regrets'].std():.2f}")
        logger.info(f"  Ratio:          {result['ratios'].mean():.2f} ± {result['ratios'].std():.2f}")
        logger.info(f"  Accuracy:       {result['accuracies'].mean():.4f} ± {result['accuracies'].std():.4f}")

    # Plot
    plot_all_scenarios(all_results, args.save_dir)

    # Summary table
    print(f"\n{'='*100}")
    print(f"{'Scenario':<25} | {'Learned Regret':>18} | {'Optimal Regret':>18} | {'Ratio':>12} | {'Accuracy':>12}")
    print(f"{'-'*100}")
    for r in all_results:
        print(f"{r['scenario']:<25} | "
              f"{r['learned_regrets'].mean():>7.2f} ± {r['learned_regrets'].std():>6.2f} | "
              f"{r['optimal_regrets'].mean():>7.2f} ± {r['optimal_regrets'].std():>6.2f} | "
              f"{r['ratios'].mean():>5.2f} ± {r['ratios'].std():>4.2f} | "
              f"{r['accuracies'].mean():>5.4f} ± {r['accuracies'].std():>4.4f}")
    print(f"{'='*100}")


if __name__ == '__main__':
    main()
