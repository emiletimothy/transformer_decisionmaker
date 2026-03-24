#!/usr/bin/env python3
"""
Generate eval datasets for all test scenarios and save to data/eval/.

Scenarios:
  - Structural: in_distribution, one_dominant, all_mediocre, two_good_two_bad, adversarial
  - OOD sequence lengths: 5, 10, 15, 20, 30, 50
  - OOD learning rates: 0.01, 0.1, 0.25, 0.5, 0.9
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import json
import argparse


def generate_scenario(name, n_steps, n_experts, learning_rate, n_sequences, seed=None):
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


def main():
    parser = argparse.ArgumentParser(description='Generate eval datasets')
    parser.add_argument('--n_sequences', type=int, default=50, help='Sequences per scenario')
    parser.add_argument('--n_experts', type=int, default=4)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--output_dir', type=str, default='../data/eval')
    args = parser.parse_args()

    out_dir = os.path.join(os.path.dirname(__file__), args.output_dir)
    os.makedirs(out_dir, exist_ok=True)

    N = args.n_sequences
    ne = args.n_experts

    # ── Structural scenarios ──
    structural = {
        'in_distribution':  dict(n_steps=20, learning_rate=0.25),
        'one_dominant':     dict(n_steps=20, learning_rate=0.25),
        'all_mediocre':     dict(n_steps=20, learning_rate=0.25),
        'two_good_two_bad': dict(n_steps=20, learning_rate=0.25),
        'adversarial':      dict(n_steps=20, learning_rate=0.25),
    }

    structural_data = {}
    for name, params in structural.items():
        seqs = generate_scenario(name, n_experts=ne, n_sequences=N, seed=args.seed, **params)
        structural_data[name] = seqs
        print(f"  structural/{name}: {len(seqs)} sequences, {params['n_steps']} steps")

    # ── OOD: sequence lengths ──
    ood_lengths = {
        'steps_5':  5,
        'steps_10': 10,
        'steps_15': 15,
        'steps_20': 20,
        'steps_30': 30,
        'steps_50': 50,
    }

    ood_len_data = {}
    for label, n_steps in ood_lengths.items():
        seqs = generate_scenario('in_distribution', n_steps=n_steps, n_experts=ne,
                                 learning_rate=0.25, n_sequences=N, seed=args.seed)
        ood_len_data[label] = seqs
        print(f"  ood_lengths/{label}: {len(seqs)} sequences")

    # ── OOD: learning rates ──
    ood_lrs = {
        'lr_0.01': 0.01,
        'lr_0.1':  0.10,
        'lr_0.25': 0.25,
        'lr_0.5':  0.50,
        'lr_0.9':  0.90,
    }

    ood_lr_data = {}
    for label, lr in ood_lrs.items():
        seqs = generate_scenario('in_distribution', n_steps=20, n_experts=ne,
                                 learning_rate=lr, n_sequences=N, seed=args.seed)
        ood_lr_data[label] = seqs
        print(f"  ood_lr/{label}: {len(seqs)} sequences")

    # ── Save ──
    dataset = {
        'config': {
            'n_sequences': N,
            'n_experts': ne,
            'seed': args.seed,
        },
        'structural': structural_data,
        'ood_lengths': ood_len_data,
        'ood_lr': ood_lr_data,
    }

    out_path = os.path.join(out_dir, 'eval_dataset.json')
    with open(out_path, 'w') as f:
        json.dump(dataset, f, indent=2)

    size_mb = os.path.getsize(out_path) / (1024 * 1024)
    print(f"\nEval dataset saved to {out_path}  ({size_mb:.1f} MB)")


if __name__ == '__main__':
    main()
