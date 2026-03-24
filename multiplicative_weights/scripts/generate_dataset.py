#!/usr/bin/env python3
"""
Generate MW training dataset and save to a JSON file.

Produces sequences of (expert_predictions, losses, weights, true_labels)
by running the MultiplicativeWeights algorithm on random online-learning problems.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import json
import argparse
from multiplicative_weights import MultiplicativeWeights


def generate_single_sequence(n_experts: int, n_steps: int, learning_rate: float):
    """Generate one MW training sequence.
    
    No MW weights are stored — the model must learn to track them implicitly.
    Weights can be recomputed from losses for evaluation via compute_mw_weights().
    """
    # Expert qualities are fixed for the entire sequence
    expert_qualities = [np.random.uniform(0.3, 0.9) for _ in range(n_experts)]

    expert_predictions = []
    losses = []
    true_labels = []

    for step in range(n_steps):
        # Random true label
        true_label = np.random.randint(0, 2)
        true_labels.append(true_label)

        # Each expert predicts correctly with its fixed quality probability
        step_preds = []
        step_losses = []
        for e in range(n_experts):
            correct = np.random.random() < expert_qualities[e]
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
        'learning_rate': learning_rate,
    }


def generate_mw_training_data(n_sequences: int, max_steps: int, n_experts: int):
    """Generate a batch of MW training sequences with varying lengths and learning rates."""
    sequences = []
    for _ in range(n_sequences):
        n_steps = np.random.randint(3, max_steps + 1)
        learning_rate = float(np.random.uniform(0.05, 0.5))
        seq = generate_single_sequence(n_experts, n_steps, learning_rate)
        sequences.append(seq)
    return sequences


def main():
    parser = argparse.ArgumentParser(description='Generate MW training dataset')
    parser.add_argument('--n_train', type=int, default=3000, help='Number of training sequences')
    parser.add_argument('--n_val', type=int, default=500, help='Number of validation sequences')
    parser.add_argument('--n_test', type=int, default=200, help='Number of test sequences')
    parser.add_argument('--max_steps', type=int, default=10, help='Maximum sequence length')
    parser.add_argument('--n_experts', type=int, default=4, help='Number of experts')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--output', type=str, default='../data/mw_dataset.json',
                        help='Output file path')
    args = parser.parse_args()

    np.random.seed(args.seed)

    print(f"Generating {args.n_train} train, {args.n_val} val, {args.n_test} test sequences "
          f"(max_steps={args.max_steps}, n_experts={args.n_experts}) ...")

    train = generate_mw_training_data(args.n_train, args.max_steps, args.n_experts)
    val   = generate_mw_training_data(args.n_val,   args.max_steps, args.n_experts)
    test  = generate_mw_training_data(args.n_test,  args.max_steps, args.n_experts)

    dataset = {
        'config': {
            'n_train': args.n_train,
            'n_val': args.n_val,
            'n_test': args.n_test,
            'max_steps': args.max_steps,
            'n_experts': args.n_experts,
            'seed': args.seed,
        },
        'train': train,
        'val': val,
        'test': test,
    }

    out_path = os.path.join(os.path.dirname(__file__), args.output)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    with open(out_path, 'w') as f:
        json.dump(dataset, f, indent=2)

    size_mb = os.path.getsize(out_path) / (1024 * 1024)
    print(f"Dataset saved to {out_path}  ({size_mb:.1f} MB)")
    print(f"  train: {len(train)} sequences")
    print(f"  val:   {len(val)} sequences")
    print(f"  test:  {len(test)} sequences")

    # Print a sample sequence for sanity check
    sample = train[0]
    print(f"\nSample sequence (n_steps={sample['n_steps']}, lr={sample['learning_rate']:.3f}):")
    print(f"  expert_predictions[0]: {sample['expert_predictions'][0]}")
    print(f"  losses[0]:             {sample['losses'][0]}")
    print(f"  true_labels:           {sample['true_labels']}")


if __name__ == '__main__':
    main()
