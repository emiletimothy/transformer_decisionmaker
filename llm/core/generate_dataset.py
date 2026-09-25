"""
Generate expert-prediction datasets for online learning experiments.

Usage:
  python generate_dataset.py --regime stratified --output mw_dataset.json
  python generate_dataset.py --regime flat --output mw_dataset_flat.json
  python generate_dataset.py --regime antisignal --output mw_dataset_antisignal.json
  python generate_dataset.py --all   # generate all three

Each instance has K=4 experts with fixed accuracies drawn from tier-specific
uniform distributions.  True labels are i.i.d. Bernoulli(0.5).  Expert i
predicts correctly with probability p_i, independently across rounds.
"""

import argparse
import json
import random

REGIMES = {
    "stratified": {
        "tiers": {
            "best":   (0.9, 1.0),
            "second": (0.65, 0.8),
            "third":  (0.55, 0.7),
            "worst":  (0.45, 0.6),
        },
        "description": "stratified expert accuracy: best 90-100%, 2nd 65-80%, 3rd 55-70%, worst 45-60%",
    },
    "flat": {
        "tiers": {
            "best": (0.6, 0.7),
            "rest": (0.4, 0.6),  # 3 experts
        },
        "description": "flat expert accuracy: best 60-70%, rest 40-60%",
    },
    "antisignal": {
        "tiers": {
            "best":        (0.6, 0.7),
            "anti_signal": (0.0, 0.1),
            "mid":         (0.4, 0.6),  # 2 experts
        },
        "description": "anti-signal dataset: best 60-70%, anti-signal 0-10%, two mid 40-60%",
    },
}

DEFAULT_OUTPUT = {
    "stratified":  "mw_dataset.json",
    "flat":        "mw_dataset_flat.json",
    "antisignal":  "mw_dataset_antisignal.json",
}


def _sample_accuracies(regime_name):
    """Return a list of 4 expert accuracies, randomly assigned to positions."""
    tiers = REGIMES[regime_name]["tiers"]
    accs = []
    if regime_name == "stratified":
        for tier in ["best", "second", "third", "worst"]:
            lo, hi = tiers[tier]
            accs.append(random.uniform(lo, hi))
    elif regime_name == "flat":
        lo, hi = tiers["best"]
        accs.append(random.uniform(lo, hi))
        lo, hi = tiers["rest"]
        for _ in range(3):
            accs.append(random.uniform(lo, hi))
    elif regime_name == "antisignal":
        lo, hi = tiers["best"]
        accs.append(random.uniform(lo, hi))
        lo, hi = tiers["anti_signal"]
        accs.append(random.uniform(lo, hi))
        lo, hi = tiers["mid"]
        for _ in range(2):
            accs.append(random.uniform(lo, hi))
    # Shuffle so expert identity (A,B,C,D) is random
    random.shuffle(accs)
    return accs


def generate_case(n_steps, regime_name):
    eta = random.uniform(0.05, 0.5)
    expert_acc = _sample_accuracies(regime_name)

    expert_predictions = []
    true_labels = []
    losses = []

    for t in range(n_steps):
        y = random.randint(0, 1)
        true_labels.append(y)

        preds_t = []
        losses_t = []
        for i in range(4):
            if random.random() < expert_acc[i]:
                pred = y
            else:
                pred = 1 - y
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


def generate_dataset(regime_name, n_cases=30, n_steps=200, seed=42):
    random.seed(seed)
    regime = REGIMES[regime_name]

    cases = [generate_case(n_steps, regime_name) for _ in range(n_cases)]

    tiers_config = {k: list(v) for k, v in regime["tiers"].items()}

    return {
        "config": {
            "n_cases": n_cases,
            "n_steps": n_steps,
            "n_experts": 4,
            "seed": seed,
            "tiers": tiers_config,
            "description": regime["description"],
        },
        "cases": cases,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--regime", choices=list(REGIMES.keys()),
                        help="Expert accuracy regime")
    parser.add_argument("--output", type=str, help="Output JSON path")
    parser.add_argument("--cases", type=int, default=30)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--all", action="store_true",
                        help="Generate all three regimes with default names")
    args = parser.parse_args()

    if args.all:
        for regime_name in REGIMES:
            out = DEFAULT_OUTPUT[regime_name]
            dataset = generate_dataset(regime_name, args.cases, args.steps, args.seed)
            with open(out, "w") as f:
                json.dump(dataset, f, indent=2)
            print(f"Generated {regime_name}: {args.cases} cases × {args.steps} steps → {out}")
    elif args.regime:
        out = args.output or DEFAULT_OUTPUT[args.regime]
        dataset = generate_dataset(args.regime, args.cases, args.steps, args.seed)
        with open(out, "w") as f:
            json.dump(dataset, f, indent=2)
        print(f"Generated {args.regime}: {args.cases} cases × {args.steps} steps → {out}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
