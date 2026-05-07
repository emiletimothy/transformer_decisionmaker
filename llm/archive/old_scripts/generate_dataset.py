"""
Generate MWU dataset.

Usage:
  python generate_dataset.py [--steps 100] [--cases 20] [--seed 42] [--output mw_dataset_new.json]
"""

import argparse
import json
import random


def generate_case(n_steps, n_experts=4):
    eta = random.uniform(0.05, 0.5)

    # per-expert accuracy
    expert_acc = [random.random() for _ in range(n_experts)]

    expert_predictions = []
    true_labels = []
    losses = []

    for t in range(n_steps):
        y = random.randint(0, 1)
        true_labels.append(y)

        preds_t = []
        losses_t = []
        for i in range(n_experts):
            if random.random() < expert_acc[i]:
                pred = y        # correct
            else:
                pred = 1 - y    # wrong
            preds_t.append(pred)
            losses_t.append(float(pred != y))

        expert_predictions.append(preds_t)
        losses.append(losses_t)

    return {
        "expert_predictions": expert_predictions,
        "losses": losses,
        "true_labels": true_labels,
        "n_steps": n_steps,
        "learning_rate": eta,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--cases", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="mw_dataset_new.json")
    args = parser.parse_args()

    random.seed(args.seed)

    cases = [generate_case(args.steps) for _ in range(args.cases)]

    dataset = {
        "config": {
            "n_cases": args.cases,
            "n_steps": args.steps,
            "n_experts": 4,
            "seed": args.seed,
        },
        "cases": cases,
    }

    with open(args.output, "w") as f:
        json.dump(dataset, f, indent=2)

    print(f"Generated {args.cases} cases × {args.steps} steps → {args.output}")


if __name__ == "__main__":
    main()
