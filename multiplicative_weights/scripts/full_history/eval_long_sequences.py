#!/usr/bin/env python3
"""
Evaluate the learned MW transformer on long sequences (100, 1000 steps).

Uses a sliding window to handle sequences longer than the model's context window.
Compares learned model regret against optimal MW regret, averaged over many trials.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import argparse
import logging

from learned_mw_transformer import (
    ContinuousCoTTransformer, LearnedMWTransformer,
    ModelConfig, MWTokenizer,
    generate_sequence_with_continuous_cot,
)
from generate_dataset import generate_single_sequence
from multiplicative_weights import MultiplicativeWeights

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Sliding-window continuous-CoT generation for long sequences
# ---------------------------------------------------------------------------

def generate_long_sequence_continuous_cot(
    model: ContinuousCoTTransformer,
    sequence: Dict,
    tokenizer: MWTokenizer,
    device: torch.device,
    max_context_tokens: int = 1024,
) -> Dict:
    """
    Generate decisions for an arbitrarily long sequence using a sliding window.

    When the token history exceeds max_context_tokens, we keep only the most
    recent tokens that fit, always retaining the START token at position 0.
    """
    model.eval()

    token_ids = [tokenizer.START_TOKEN]
    cot_weights = []
    decisions = []

    for step in range(len(sequence['expert_predictions'])):
        # Step marker
        token_ids.append(tokenizer.STEP_TOKENS[step % 100])

        # Expert predictions
        for expert_idx, pred in enumerate(sequence['expert_predictions'][step]):
            token_ids.append(tokenizer.EXPERT_TOKENS[expert_idx])
            token_ids.append(
                tokenizer.PRED_1_TOKEN if pred == 1 else tokenizer.PRED_0_TOKEN
            )

        # --- sliding window: truncate if too long ---
        if len(token_ids) > max_context_tokens:
            # Keep start token + most recent tokens
            overflow = len(token_ids) - max_context_tokens
            token_ids = [tokenizer.START_TOKEN] + token_ids[overflow + 1:]

        # Continuous thought: get weights and decision
        context_tensor = torch.tensor(
            [token_ids], dtype=torch.long, device=device
        )
        with torch.no_grad():
            weights, pred_logit = model.think_and_predict(context_tensor)

        step_weights = weights[0].cpu().numpy()
        cot_weights.append(step_weights)
        decision = 1 if torch.sigmoid(pred_logit[0, 0]) > 0.5 else 0
        decisions.append(decision)

        # SEP token
        token_ids.append(tokenizer.SEP_TOKEN)

        # True label
        true_label = sequence['true_labels'][step]
        token_ids.append(
            tokenizer.PRED_1_TOKEN if true_label == 1 else tokenizer.PRED_0_TOKEN
        )

        # Losses
        for expert_idx, loss_val in enumerate(sequence['losses'][step]):
            token_ids.append(tokenizer.EXPERT_TOKENS[expert_idx])
            token_ids.append(tokenizer.discretize_loss(loss_val))

    return {
        'decisions': np.array(decisions),
        'cot_weights': cot_weights,
    }


# ---------------------------------------------------------------------------
# Regret computation (same logic as training script)
# ---------------------------------------------------------------------------

def get_optimal_mw_decisions(seq: Dict) -> np.ndarray:
    n_experts = len(seq['expert_predictions'][0])
    n_steps = seq['n_steps']
    learning_rate = np.sqrt(np.log(n_experts) / max(n_steps, 1))
    mw = MultiplicativeWeights(n_experts, learning_rate)

    decisions = []
    for step in range(len(seq['true_labels'])):
        weights = mw.get_probabilities()
        expert_preds = seq['expert_predictions'][step]
        weighted_prediction = np.sum(weights * expert_preds)
        decision = 1 if weighted_prediction > 0.5 else 0
        decisions.append(decision)
        if step < len(seq['losses']):
            mw.update_weights(np.array(seq['losses'][step]))
    return np.array(decisions)


def compute_regret_trajectory(seq, learned_decisions, optimal_decisions):
    """Return per-step cumulative regret for learned and optimal strategies."""
    losses = seq['losses']
    true_labels = seq['true_labels']
    n_steps = len(true_labels)

    expert_cumulative_losses = np.zeros(len(losses[0]))
    learned_cum = 0.0
    optimal_cum = 0.0

    learned_traj = []
    optimal_traj = []

    for step in range(n_steps):
        expert_cumulative_losses += losses[step]
        best_expert = np.min(expert_cumulative_losses)

        l_loss = 0.0 if (step < len(learned_decisions) and learned_decisions[step] == true_labels[step]) else 1.0
        learned_cum += l_loss

        o_loss = 0.0 if (step < len(optimal_decisions) and optimal_decisions[step] == true_labels[step]) else 1.0
        optimal_cum += o_loss

        learned_traj.append(max(0.0, learned_cum - best_expert))
        optimal_traj.append(max(0.0, optimal_cum - best_expert))

    return np.array(learned_traj), np.array(optimal_traj)


def compute_final_regret(seq, decisions):
    """Return scalar final regret for a set of decisions."""
    losses = seq['losses']
    true_labels = seq['true_labels']
    n_steps = len(true_labels)

    expert_cumulative_losses = np.zeros(len(losses[0]))
    cum_loss = 0.0

    for step in range(n_steps):
        expert_cumulative_losses += losses[step]
        loss = 0.0 if (step < len(decisions) and decisions[step] == true_labels[step]) else 1.0
        cum_loss += loss

    return max(0.0, cum_loss - np.min(expert_cumulative_losses))


# ---------------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------------

def load_model(path, device):
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


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_experiment(model, tokenizer, cot_mode, model_config, device,
                   seq_lengths, n_trials, n_experts):
    max_ctx = model_config.max_sequence_length

    results = {}  # length -> dict of metrics

    for length in seq_lengths:
        logger.info(f"\n{'='*60}")
        logger.info(f"Sequence length: {length}  |  trials: {n_trials}")
        logger.info(f"{'='*60}")

        learned_final_regrets = []
        optimal_final_regrets = []
        learned_accuracies = []
        all_learned_trajs = []
        all_optimal_trajs = []

        for trial in range(n_trials):
            seq = generate_single_sequence(n_experts, length)

            # Learned model decisions
            if cot_mode == 'continuous':
                result = generate_long_sequence_continuous_cot(
                    model, seq, tokenizer, device, max_context_tokens=max_ctx
                )
                learned_decisions = result['decisions']
            else:
                # For discrete model, would need similar sliding window
                # Not implemented here since checkpoint is continuous
                raise NotImplementedError("Only continuous CoT supported in this script")

            # Optimal MW decisions
            optimal_decisions = get_optimal_mw_decisions(seq)

            # Accuracy
            true_labels = np.array(seq['true_labels'])
            n = min(len(learned_decisions), len(true_labels))
            accuracy = np.mean(learned_decisions[:n] == true_labels[:n])
            learned_accuracies.append(accuracy)

            # Final regrets
            learned_regret = compute_final_regret(seq, learned_decisions)
            optimal_regret = compute_final_regret(seq, optimal_decisions)
            learned_final_regrets.append(learned_regret)
            optimal_final_regrets.append(optimal_regret)

            # Regret trajectories
            l_traj, o_traj = compute_regret_trajectory(seq, learned_decisions, optimal_decisions)
            all_learned_trajs.append(l_traj)
            all_optimal_trajs.append(o_traj)

            if (trial + 1) % 10 == 0:
                logger.info(f"  trial {trial+1}/{n_trials} done")

        learned_final_regrets = np.array(learned_final_regrets)
        optimal_final_regrets = np.array(optimal_final_regrets)
        learned_accuracies = np.array(learned_accuracies)

        # Stack trajectories (all same length for a given seq_length)
        learned_matrix = np.array(all_learned_trajs)  # [n_trials, length]
        optimal_matrix = np.array(all_optimal_trajs)

        # Regret ratios (avoid div by zero)
        ratios = []
        for lr, opr in zip(learned_final_regrets, optimal_final_regrets):
            if opr > 1e-6:
                ratios.append(lr / opr)
            elif lr < 1e-6:
                ratios.append(1.0)
            else:
                ratios.append(10.0)
        ratios = np.array(ratios)

        results[length] = {
            'learned_final_regrets': learned_final_regrets,
            'optimal_final_regrets': optimal_final_regrets,
            'learned_accuracies': learned_accuracies,
            'ratios': ratios,
            'learned_traj_mean': np.mean(learned_matrix, axis=0),
            'learned_traj_std': np.std(learned_matrix, axis=0),
            'optimal_traj_mean': np.mean(optimal_matrix, axis=0),
            'optimal_traj_std': np.std(optimal_matrix, axis=0),
        }

        logger.info(f"\n  Results for length={length}:")
        logger.info(f"    Learned final regret:  {learned_final_regrets.mean():.2f} ± {learned_final_regrets.std():.2f}")
        logger.info(f"    Optimal MW regret:     {optimal_final_regrets.mean():.2f} ± {optimal_final_regrets.std():.2f}")
        logger.info(f"    Regret ratio:          {ratios.mean():.2f} ± {ratios.std():.2f}")
        logger.info(f"    Accuracy:              {learned_accuracies.mean():.4f} ± {learned_accuracies.std():.4f}")

    return results


def plot_results(results, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    lengths = sorted(results.keys())

    # --- Figure 1: Regret trajectories per length ---
    fig, axes = plt.subplots(1, len(lengths), figsize=(8 * len(lengths), 6), squeeze=False)
    for i, length in enumerate(lengths):
        ax = axes[0, i]
        r = results[length]
        steps = np.arange(1, length + 1)

        ax.plot(steps, r['learned_traj_mean'], 'r-', linewidth=2, label='Learned MW (mean)')
        ax.fill_between(steps,
                         r['learned_traj_mean'] - r['learned_traj_std'],
                         r['learned_traj_mean'] + r['learned_traj_std'],
                         color='red', alpha=0.15, label='Learned ±1σ')
        ax.plot(steps, r['optimal_traj_mean'], 'b-', linewidth=2, label='Optimal MW (mean)')
        ax.fill_between(steps,
                         r['optimal_traj_mean'] - r['optimal_traj_std'],
                         r['optimal_traj_mean'] + r['optimal_traj_std'],
                         color='blue', alpha=0.15, label='Optimal ±1σ')
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Cumulative Regret')
        ax.set_title(f'Regret Growth (T={length})')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path1 = os.path.join(save_dir, 'long_seq_regret_trajectories.png')
    plt.savefig(path1, dpi=200, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved {path1}")

    # --- Figure 2: Summary bar chart ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    x = np.arange(len(lengths))
    width = 0.35

    # Final regret
    learned_means = [results[l]['learned_final_regrets'].mean() for l in lengths]
    learned_stds = [results[l]['learned_final_regrets'].std() for l in lengths]
    optimal_means = [results[l]['optimal_final_regrets'].mean() for l in lengths]
    optimal_stds = [results[l]['optimal_final_regrets'].std() for l in lengths]

    axes[0].bar(x - width/2, learned_means, width, yerr=learned_stds, label='Learned', color='red', alpha=0.7)
    axes[0].bar(x + width/2, optimal_means, width, yerr=optimal_stds, label='Optimal MW', color='blue', alpha=0.7)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([str(l) for l in lengths])
    axes[0].set_xlabel('Sequence Length')
    axes[0].set_ylabel('Final Regret')
    axes[0].set_title('Final Regret Comparison')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Regret ratio
    ratio_means = [results[l]['ratios'].mean() for l in lengths]
    ratio_stds = [results[l]['ratios'].std() for l in lengths]
    axes[1].bar(x, ratio_means, 0.5, yerr=ratio_stds, color='green', alpha=0.7)
    axes[1].axhline(y=1.0, color='k', linestyle='--', alpha=0.5)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([str(l) for l in lengths])
    axes[1].set_xlabel('Sequence Length')
    axes[1].set_ylabel('Regret Ratio (Learned / Optimal)')
    axes[1].set_title('Regret Ratio')
    axes[1].grid(True, alpha=0.3)

    # Accuracy
    acc_means = [results[l]['learned_accuracies'].mean() for l in lengths]
    acc_stds = [results[l]['learned_accuracies'].std() for l in lengths]
    axes[2].bar(x, acc_means, 0.5, yerr=acc_stds, color='purple', alpha=0.7)
    axes[2].axhline(y=0.5, color='k', linestyle='--', alpha=0.5, label='Random')
    axes[2].set_xticks(x)
    axes[2].set_xticklabels([str(l) for l in lengths])
    axes[2].set_xlabel('Sequence Length')
    axes[2].set_ylabel('Accuracy')
    axes[2].set_title('Prediction Accuracy')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    path2 = os.path.join(save_dir, 'long_seq_summary.png')
    plt.savefig(path2, dpi=200, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved {path2}")

    return path1, path2


def main():
    parser = argparse.ArgumentParser(description='Evaluate MW transformer on long sequences')
    parser.add_argument('--checkpoint', type=str,
                        default='../figures/checkpoints/model_stage_10.pt')
    parser.add_argument('--seq_lengths', type=int, nargs='+', default=[50, 100, 500, 1000])
    parser.add_argument('--n_trials', type=int, default=50)
    parser.add_argument('--n_experts', type=int, default=4)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--save_dir', type=str, default='../figures/eval')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    logger.info(f"Loading checkpoint: {args.checkpoint}")
    model, tokenizer, cot_mode, model_config = load_model(args.checkpoint, device)
    logger.info(f"Model: {cot_mode}, d_model={model_config.d_model}, "
                f"n_layers={model_config.n_layers}, max_seq_len={model_config.max_sequence_length}, "
                f"n_thought_steps={model_config.n_thought_steps}")

    results = run_experiment(
        model, tokenizer, cot_mode, model_config, device,
        seq_lengths=args.seq_lengths,
        n_trials=args.n_trials,
        n_experts=args.n_experts,
    )

    plot_results(results, args.save_dir)

    # Print final summary table
    print("\n" + "=" * 80)
    print(f"{'Length':>8} | {'Learned Regret':>18} | {'Optimal Regret':>18} | {'Ratio':>12} | {'Accuracy':>12}")
    print("-" * 80)
    for length in sorted(results.keys()):
        r = results[length]
        print(f"{length:>8} | "
              f"{r['learned_final_regrets'].mean():>7.2f} ± {r['learned_final_regrets'].std():>6.2f} | "
              f"{r['optimal_final_regrets'].mean():>7.2f} ± {r['optimal_final_regrets'].std():>6.2f} | "
              f"{r['ratios'].mean():>5.2f} ± {r['ratios'].std():>4.2f} | "
              f"{r['learned_accuracies'].mean():>5.4f} ± {r['learned_accuracies'].std():>4.4f}")
    print("=" * 80)


if __name__ == '__main__':
    main()
