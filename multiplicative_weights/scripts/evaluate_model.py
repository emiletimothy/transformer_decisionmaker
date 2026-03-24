#!/usr/bin/env python3
"""
Rigorous evaluation of the trained MW Transformer.

Generates:
  1. Main results table (per-scenario metrics)
  2. OOD breakdown table (by sequence length / n_experts)
  3. Regret curves per scenario (learned vs optimal MW)
  4. Weight trajectory heatmaps (predicted vs ground-truth)
  5. OOD generalization bar chart (regret ratio vs sequence length)

Usage:
    python3 scripts/evaluate_model.py --model_path figures/learned_mw_transformer.pt
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from typing import Dict, List, Tuple
from tabulate import tabulate

from multiplicative_weights import MultiplicativeWeights
from learned_mw_transformer import (
    LearnedMWTransformer, ContinuousCoTTransformer, ModelConfig, MWTokenizer,
    generate_sequence_with_continuous_cot, compute_mw_weights
)

# ── Helpers ──────────────────────────────────────────────────────────────

def generate_scenario(name: str, n_steps: int, n_experts: int,
                      learning_rate: float, n_sequences: int = 50,
                      seed: int = None) -> List[Dict]:
    """Generate test sequences for a named scenario."""
    if seed is not None:
        np.random.seed(seed)

    sequences = []
    for _ in range(n_sequences):
        if name == "in_distribution":
            expert_qualities = [np.random.uniform(0.3, 0.9) for _ in range(n_experts)]
        elif name == "one_dominant":
            expert_qualities = [np.random.uniform(0.3, 0.5) for _ in range(n_experts)]
            expert_qualities[0] = np.random.uniform(0.90, 0.98)
        elif name == "all_mediocre":
            expert_qualities = [np.random.uniform(0.45, 0.55) for _ in range(n_experts)]
        elif name == "two_good_two_bad":
            expert_qualities = []
            for i in range(n_experts):
                if i < n_experts // 2:
                    expert_qualities.append(np.random.uniform(0.80, 0.95))
                else:
                    expert_qualities.append(np.random.uniform(0.20, 0.35))
        elif name == "adversarial":
            # Experts alternate being good/bad across steps — qualities flip
            expert_qualities = [np.random.uniform(0.3, 0.9) for _ in range(n_experts)]
        else:
            expert_qualities = [np.random.uniform(0.3, 0.9) for _ in range(n_experts)]

        expert_predictions = []
        losses = []
        true_labels = []

        for step in range(n_steps):
            true_label = np.random.randint(0, 2)
            true_labels.append(true_label)

            step_preds = []
            step_losses = []
            for e in range(n_experts):
                if name == "adversarial":
                    # Flip quality every other step for each expert
                    q = expert_qualities[e] if (step + e) % 2 == 0 else 1.0 - expert_qualities[e]
                else:
                    q = expert_qualities[e]
                correct = np.random.random() < q
                pred = true_label if correct else 1 - true_label
                step_preds.append(pred)
                step_losses.append(0.0 if pred == true_label else 1.0)

            expert_predictions.append(step_preds)
            losses.append(step_losses)

        sequences.append({
            'expert_predictions': expert_predictions,
            'losses': losses,
            'true_labels': true_labels,
            'n_steps': n_steps,
            'learning_rate': learning_rate,
            'expert_qualities': expert_qualities,
        })

    return sequences


def run_inference(model, tokenizer, sequence, device) -> Dict:
    """Run inference on a single sequence.

    Dispatches to continuous-CoT generation for ContinuousCoTTransformer,
    or teacher-forced forward pass for LearnedMWTransformer.
    """
    model.eval()

    if isinstance(model, ContinuousCoTTransformer):
        return _run_continuous_cot_inference(model, tokenizer, sequence, device)
    else:
        return _run_teacher_forced_inference(model, tokenizer, sequence, device)


def _run_continuous_cot_inference(model, tokenizer, sequence, device) -> Dict:
    """Run continuous-CoT inference using think_and_predict at each step."""
    with torch.no_grad():
        cot_result = generate_sequence_with_continuous_cot(
            model, sequence, tokenizer, device
        )

    pred_decisions = cot_result['decisions']
    gt_labels = np.array(sequence['true_labels'])
    n = min(len(pred_decisions), len(gt_labels))
    if n == 0:
        return None

    # Compute GT weights on-the-fly for evaluation
    cot_weights = np.array(cot_result['cot_weights'])  # [n_steps, n_experts]
    gt_weights_all = compute_mw_weights(sequence, len(sequence['losses'][0]))
    gt_weights_seq = np.array(gt_weights_all[1:n+1])  # skip initial uniform

    return {
        'pred_weights': cot_weights[:n],
        'gt_weights': gt_weights_seq[:n],
        'pred_decisions': pred_decisions[:n].astype(int),
        'gt_labels': gt_labels[:n].astype(int),
    }


def _run_teacher_forced_inference(model, tokenizer, sequence, device) -> Dict:
    """Run teacher-forced inference on a single sequence."""
    tokens = tokenizer.encode_sequence(sequence)
    input_ids = torch.tensor([tokens['input_ids']], device=device)

    with torch.no_grad():
        outputs = model(input_ids)

    weight_logits = outputs['weight_logits'][0]
    pred_logits = outputs['prediction_logits'][0].squeeze(-1)
    target_mask = torch.tensor(tokens['target_mask'], device=device)

    min_len = min(len(target_mask), weight_logits.shape[0], pred_logits.shape[0])
    target_mask = target_mask[:min_len]
    weight_logits = weight_logits[:min_len]
    pred_logits = pred_logits[:min_len]

    if not target_mask.any():
        return None

    pred_weights = torch.softmax(weight_logits[target_mask], dim=-1).cpu().numpy()
    pred_decisions = (torch.sigmoid(pred_logits[target_mask]) > 0.5).cpu().numpy().astype(int)

    prediction_targets = np.array(tokens['prediction_targets'])[:min_len]
    gt_labels = (prediction_targets[target_mask.cpu().numpy()] > 0.5).astype(int)

    # Compute GT weights on-the-fly for evaluation
    n_experts = len(sequence['losses'][0])
    gt_weights_all = compute_mw_weights(sequence, n_experts)
    # GT weights at decision points = weights after each step update (indices 1..n_steps)
    n_decisions = int(target_mask.sum().item())
    gt_weights = np.array(gt_weights_all[1:n_decisions+1])

    return {
        'pred_weights': pred_weights,
        'gt_weights': gt_weights,
        'pred_decisions': pred_decisions,
        'gt_labels': gt_labels,
    }


def compute_metrics(results: List[Dict]) -> Dict:
    """Aggregate per-sequence results into summary metrics."""
    weight_mses, accs, seq_accs = [], [], []
    learned_regrets, optimal_regrets, regret_ratios = [], [], []

    for r in results:
        if r is None:
            continue
        # Weight MSE
        valid = r['gt_weights'].sum(axis=-1) > 0
        if valid.any():
            mse = np.mean((r['pred_weights'][valid] - r['gt_weights'][valid]) ** 2)
            weight_mses.append(mse)

        # Decision accuracy
        n = min(len(r['pred_decisions']), len(r['gt_labels']))
        if n > 0:
            acc = np.mean(r['pred_decisions'][:n] == r['gt_labels'][:n])
            accs.append(acc)
            seq_accs.append(float(np.all(r['pred_decisions'][:n] == r['gt_labels'][:n])))

    def safe_stats(arr):
        if len(arr) == 0:
            return {'mean': 0.0, 'std': 0.0, 'n': 0}
        return {'mean': float(np.mean(arr)), 'std': float(np.std(arr)), 'n': len(arr)}

    return {
        'weight_mse': safe_stats(weight_mses),
        'decision_accuracy': safe_stats(accs),
        'sequence_accuracy': safe_stats(seq_accs),
    }


def compute_regret(sequence, pred_decisions) -> Tuple[List[float], List[float]]:
    """Compute cumulative regret trajectories for learned and optimal MW."""
    losses_list = sequence['losses']
    true_labels = sequence['true_labels']
    n_steps = len(true_labels)
    n_experts = len(losses_list[0])

    # Optimal MW decisions
    mw = MultiplicativeWeights(n_experts, sequence['learning_rate'])
    optimal_decisions = []
    for step in range(n_steps):
        weights = mw.get_probabilities()
        preds = sequence['expert_predictions'][step]
        weighted_pred = np.dot(weights, preds)
        optimal_decisions.append(1 if weighted_pred > 0.5 else 0)
        mw.update_weights(np.array(losses_list[step]))

    expert_cum = np.zeros(n_experts)
    learned_cum, optimal_cum = 0.0, 0.0
    learned_traj, optimal_traj = [], []

    for step in range(n_steps):
        expert_cum += np.array(losses_list[step])
        best_expert = np.min(expert_cum)

        if step < len(pred_decisions):
            learned_cum += 0.0 if pred_decisions[step] == true_labels[step] else 1.0
        else:
            learned_cum += 0.5

        if step < len(optimal_decisions):
            optimal_cum += 0.0 if optimal_decisions[step] == true_labels[step] else 1.0
        else:
            optimal_cum += 0.5

        learned_traj.append(max(0.0, learned_cum - best_expert))
        optimal_traj.append(max(0.0, optimal_cum - best_expert))

    return learned_traj, optimal_traj


# ── Figures ──────────────────────────────────────────────────────────────

def plot_regret_curves(all_scenario_data: Dict, save_path: str):
    """Fig 1: Regret curves per scenario — learned vs optimal MW."""
    scenarios = list(all_scenario_data.keys())
    n = len(scenarios)
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows), squeeze=False)

    for idx, scenario in enumerate(scenarios):
        ax = axes[idx // cols][idx % cols]
        data = all_scenario_data[scenario]

        # Collect trajectories
        all_learned, all_optimal = [], []
        for seq, res in zip(data['sequences'], data['results']):
            if res is None:
                continue
            lt, ot = compute_regret(seq, res['pred_decisions'])
            all_learned.append(lt)
            all_optimal.append(ot)

        if not all_learned:
            ax.set_title(scenario.replace('_', ' ').title())
            continue

        max_len = max(len(t) for t in all_learned)
        learned_mat = np.full((len(all_learned), max_len), np.nan)
        optimal_mat = np.full((len(all_optimal), max_len), np.nan)
        for i, t in enumerate(all_learned):
            learned_mat[i, :len(t)] = t
        for i, t in enumerate(all_optimal):
            optimal_mat[i, :len(t)] = t

        steps = np.arange(1, max_len + 1)
        l_mean = np.nanmean(learned_mat, axis=0)
        l_std = np.nanstd(learned_mat, axis=0)
        o_mean = np.nanmean(optimal_mat, axis=0)
        o_std = np.nanstd(optimal_mat, axis=0)

        ax.plot(steps, l_mean, 'r-', linewidth=2, label='Learned MW')
        ax.fill_between(steps, l_mean - l_std, l_mean + l_std, color='red', alpha=0.15)
        ax.plot(steps, o_mean, 'b--', linewidth=2, label='Optimal MW')
        ax.fill_between(steps, o_mean - o_std, o_mean + o_std, color='blue', alpha=0.15)

        ax.set_xlabel('Time Step')
        ax.set_ylabel('Cumulative Regret')
        ax.set_title(scenario.replace('_', ' ').title())
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for idx in range(n, rows * cols):
        axes[idx // cols][idx % cols].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved regret curves → {save_path}")


def plot_weight_heatmaps(all_scenario_data: Dict, save_path: str):
    """Fig 2: Weight trajectory heatmaps — predicted vs ground-truth."""
    # Pick one representative sequence per scenario
    scenarios = list(all_scenario_data.keys())
    n = len(scenarios)
    fig, axes = plt.subplots(n, 2, figsize=(12, 3 * n), squeeze=False)

    for idx, scenario in enumerate(scenarios):
        data = all_scenario_data[scenario]
        # Find first non-None result
        res, seq = None, None
        for s, r in zip(data['sequences'], data['results']):
            if r is not None and r['gt_weights'].sum() > 0:
                res, seq = r, s
                break
        if res is None:
            axes[idx][0].set_title(f'{scenario} (no data)')
            axes[idx][1].set_title(f'{scenario} (no data)')
            continue

        gt_w = res['gt_weights']  # [steps, n_experts]
        pred_w = res['pred_weights']

        vmin = 0
        vmax = max(gt_w.max(), pred_w.max())

        im0 = axes[idx][0].imshow(gt_w.T, aspect='auto', cmap='YlOrRd', vmin=vmin, vmax=vmax)
        axes[idx][0].set_title(f'{scenario.replace("_", " ").title()} — Ground Truth')
        axes[idx][0].set_ylabel('Expert')
        axes[idx][0].set_xlabel('Step')

        im1 = axes[idx][1].imshow(pred_w.T, aspect='auto', cmap='YlOrRd', vmin=vmin, vmax=vmax)
        axes[idx][1].set_title(f'{scenario.replace("_", " ").title()} — Predicted')
        axes[idx][1].set_ylabel('Expert')
        axes[idx][1].set_xlabel('Step')

        fig.colorbar(im1, ax=axes[idx][1], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved weight heatmaps → {save_path}")


def plot_ood_bar_chart(ood_metrics: Dict, save_path: str):
    """Fig 3: OOD generalization — regret ratio & accuracy vs sequence length."""
    labels = list(ood_metrics.keys())
    acc_means = [ood_metrics[k]['decision_accuracy']['mean'] for k in labels]
    acc_stds = [ood_metrics[k]['decision_accuracy']['std'] for k in labels]
    wmse_means = [ood_metrics[k]['weight_mse']['mean'] for k in labels]
    wmse_stds = [ood_metrics[k]['weight_mse']['std'] for k in labels]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    x = np.arange(len(labels))
    width = 0.5

    bars1 = ax1.bar(x, acc_means, width, yerr=acc_stds, color='steelblue', alpha=0.8, capsize=4)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=30, ha='right')
    ax1.set_ylabel('Decision Accuracy')
    ax1.set_title('Decision Accuracy by Condition')
    ax1.set_ylim(0, 1.05)
    ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Random')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')

    bars2 = ax2.bar(x, wmse_means, width, yerr=wmse_stds, color='coral', alpha=0.8, capsize=4)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=30, ha='right')
    ax2.set_ylabel('Weight MSE')
    ax2.set_title('Weight MSE by Condition')
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved OOD bar chart → {save_path}")


def print_table(title: str, rows: List, headers: List[str]):
    """Print a formatted table."""
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")
    print(tabulate(rows, headers=headers, tablefmt='grid', floatfmt='.4f'))


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Evaluate trained MW Transformer')
    parser.add_argument('--model_path', type=str,
                        default='figures/learned_mw_transformer.pt')
    parser.add_argument('--output_dir', type=str, default='figures/eval')
    parser.add_argument('--n_sequences', type=int, default=50,
                        help='Sequences per scenario')
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--device', type=str, default='auto')
    args = parser.parse_args()

    # ── Device ──
    if args.device == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    # ── Load model ──
    checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
    model_config = checkpoint['model_config']
    tok_cfg = checkpoint['tokenizer_config']
    cot_mode = checkpoint.get('cot_mode', 'discrete')

    tokenizer = MWTokenizer(n_experts=tok_cfg['n_experts'])
    model_config.vocab_size = tok_cfg['vocab_size']

    if cot_mode == 'continuous':
        model = ContinuousCoTTransformer(model_config).to(device)
    else:
        model = LearnedMWTransformer(model_config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"Loaded model ({cot_mode} CoT): {sum(p.numel() for p in model.parameters())} params")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    n_experts = tok_cfg['n_experts']
    N = args.n_sequences

    # ══════════════════════════════════════════════════════════════════════
    # 1. Define test scenarios
    # ══════════════════════════════════════════════════════════════════════
    structural_scenarios = {
        'in_distribution':  dict(n_steps=7, n_experts=n_experts, learning_rate=0.25),
        'one_dominant':     dict(n_steps=7, n_experts=n_experts, learning_rate=0.25),
        'all_mediocre':     dict(n_steps=7, n_experts=n_experts, learning_rate=0.25),
        'two_good_two_bad': dict(n_steps=7, n_experts=n_experts, learning_rate=0.25),
        'adversarial':      dict(n_steps=7, n_experts=n_experts, learning_rate=0.25),
    }

    ood_seq_lengths = {
        'steps=5':  dict(name='in_distribution', n_steps=5,  n_experts=n_experts, learning_rate=0.25),
        'steps=8':  dict(name='in_distribution', n_steps=8,  n_experts=n_experts, learning_rate=0.25),
        'steps=10': dict(name='in_distribution', n_steps=10, n_experts=n_experts, learning_rate=0.25),
        'steps=15': dict(name='in_distribution', n_steps=15, n_experts=n_experts, learning_rate=0.25),
        'steps=20': dict(name='in_distribution', n_steps=20, n_experts=n_experts, learning_rate=0.25),
        'steps=30': dict(name='in_distribution', n_steps=30, n_experts=n_experts, learning_rate=0.25),
    }

    ood_lr = {
        'lr=0.01':  dict(name='in_distribution', n_steps=7, n_experts=n_experts, learning_rate=0.01),
        'lr=0.1':   dict(name='in_distribution', n_steps=7, n_experts=n_experts, learning_rate=0.10),
        'lr=0.25':  dict(name='in_distribution', n_steps=7, n_experts=n_experts, learning_rate=0.25),
        'lr=0.5':   dict(name='in_distribution', n_steps=7, n_experts=n_experts, learning_rate=0.50),
        'lr=0.9':   dict(name='in_distribution', n_steps=7, n_experts=n_experts, learning_rate=0.90),
    }

    # ══════════════════════════════════════════════════════════════════════
    # 2. Run inference on structural scenarios
    # ══════════════════════════════════════════════════════════════════════
    print("\n── Structural Scenarios ──")
    all_scenario_data = {}
    structural_metrics = {}

    for scenario, params in structural_scenarios.items():
        print(f"  Running {scenario} …")
        seqs = generate_scenario(scenario, seed=args.seed, n_sequences=N, **params)
        results = [run_inference(model, tokenizer, s, device) for s in seqs]
        all_scenario_data[scenario] = {'sequences': seqs, 'results': results}
        structural_metrics[scenario] = compute_metrics(results)

    # ══════════════════════════════════════════════════════════════════════
    # 3. Run OOD evaluations
    # ══════════════════════════════════════════════════════════════════════
    print("\n── OOD: Sequence Lengths ──")
    ood_len_metrics = {}
    ood_len_data = {}
    for label, params in ood_seq_lengths.items():
        name = params.pop('name')
        print(f"  Running {label} …")
        seqs = generate_scenario(name, seed=args.seed, n_sequences=N, **params)
        results = [run_inference(model, tokenizer, s, device) for s in seqs]
        ood_len_metrics[label] = compute_metrics(results)
        ood_len_data[label] = {'sequences': seqs, 'results': results}
        params['name'] = name  # restore

    print("\n── OOD: Learning Rates ──")
    ood_lr_metrics = {}
    for label, params in ood_lr.items():
        name = params.pop('name')
        print(f"  Running {label} …")
        seqs = generate_scenario(name, seed=args.seed, n_sequences=N, **params)
        results = [run_inference(model, tokenizer, s, device) for s in seqs]
        ood_lr_metrics[label] = compute_metrics(results)
        params['name'] = name

    # ══════════════════════════════════════════════════════════════════════
    # 4. Tables
    # ══════════════════════════════════════════════════════════════════════
    headers = ['Scenario', 'Decision Acc', 'Seq Acc', 'Weight MSE']

    # Table 1: Structural scenarios
    rows = []
    for sc, m in structural_metrics.items():
        rows.append([
            sc.replace('_', ' ').title(),
            f"{m['decision_accuracy']['mean']:.4f} ± {m['decision_accuracy']['std']:.4f}",
            f"{m['sequence_accuracy']['mean']:.4f} ± {m['sequence_accuracy']['std']:.4f}",
            f"{m['weight_mse']['mean']:.4f} ± {m['weight_mse']['std']:.4f}",
        ])
    print_table("Table 1: Structural Scenario Results", rows, headers)

    # Table 2: OOD by sequence length
    rows = []
    for label, m in ood_len_metrics.items():
        rows.append([
            label,
            f"{m['decision_accuracy']['mean']:.4f} ± {m['decision_accuracy']['std']:.4f}",
            f"{m['sequence_accuracy']['mean']:.4f} ± {m['sequence_accuracy']['std']:.4f}",
            f"{m['weight_mse']['mean']:.4f} ± {m['weight_mse']['std']:.4f}",
        ])
    print_table("Table 2: OOD — Sequence Length", rows, headers)

    # Table 3: OOD by learning rate
    rows = []
    for label, m in ood_lr_metrics.items():
        rows.append([
            label,
            f"{m['decision_accuracy']['mean']:.4f} ± {m['decision_accuracy']['std']:.4f}",
            f"{m['sequence_accuracy']['mean']:.4f} ± {m['sequence_accuracy']['std']:.4f}",
            f"{m['weight_mse']['mean']:.4f} ± {m['weight_mse']['std']:.4f}",
        ])
    print_table("Table 3: OOD — Learning Rate", rows, headers)

    # ══════════════════════════════════════════════════════════════════════
    # 5. Figures
    # ══════════════════════════════════════════════════════════════════════
    print("\n── Generating Figures ──")

    plot_regret_curves(all_scenario_data,
                       str(output_dir / 'regret_curves_by_scenario.png'))

    plot_weight_heatmaps(all_scenario_data,
                         str(output_dir / 'weight_heatmaps.png'))

    # Combine OOD length + structural for the bar chart
    combined_metrics = {}
    combined_metrics.update(structural_metrics)
    combined_metrics.update(ood_len_metrics)
    plot_ood_bar_chart(combined_metrics,
                       str(output_dir / 'ood_bar_chart.png'))

    print(f"\n✅ Evaluation complete. Results saved to {output_dir}/")


if __name__ == '__main__':
    main()
