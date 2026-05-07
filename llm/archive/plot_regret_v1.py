"""
Plot cumulative regret comparison across models for a given prompt and step count.
Uses hard prediction regret (0-1 loss) for all prompts.

Usage:
  python plot_regret.py <prompt> <n_steps> [--save <path>]

Examples:
  python plot_regret.py online 100
  python plot_regret.py online 100 --save plots/online_100.png
  python plot_regret.py mw_name 200
"""

import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

EXP_DIR = Path("exp_results")

MODEL_STYLE = {
    "ds":            {"label": "DeepSeek V3",      "color": "#E8740C", "zorder": 3},
    "ds_think":      {"label": "DeepSeek R1",      "color": "#D62728", "zorder": 4},
    "gpt4o":         {"label": "GPT-4o",           "color": "#2CA02C", "zorder": 5},
    "gpt4o_mini":    {"label": "GPT-4o-mini",      "color": "#74C476", "zorder": 2},
    "gemini25_flash": {"label": "Gemini 2.5 Flash", "color": "#7A3E9D", "zorder": 4},
    "gemini3_flash": {"label": "Gemini 3 Flash",   "color": "#C17CEB", "zorder": 4},
}
MW_STYLE = {"label": "True MW (optimal)", "color": "#4C78A8", "zorder": 10}


def _compute_hard_regret(predictions, true_labels, expert_predictions):
    """Compute cumulative hard 0-1 regret from predictions.
    regret[t] = model_cum_loss[t] - best_expert_cum_loss[t]
    """
    T = len(predictions)
    n_experts = len(expert_predictions[0])
    model_cum = 0
    expert_cum = [0] * n_experts
    regret = []
    for t in range(T):
        model_cum += int(predictions[t] != true_labels[t])
        for i in range(n_experts):
            expert_cum[i] += int(expert_predictions[t][i] != true_labels[t])
        regret.append(model_cum - min(expert_cum))
    return regret


def _extract_predictions(r):
    """Extract model predictions from result.json, handling both online and MW modes."""
    # online mode: predictions stored directly
    preds = r.get("response", {}).get("predictions")
    if preds is not None:
        return preds
    # MW mode: algorithm_predictions in response
    preds = r.get("response", {}).get("algorithm_predictions")
    if preds is not None:
        return preds
    return None


def _extract_mw_predictions(case_input):
    """Compute true MW hard predictions from input data."""
    from mw_lib import get_ground_truth_outputs
    gt = get_ground_truth_outputs(case_input)
    return gt["algorithm_predictions"]


def load_regret_curves(prompt, n_steps):
    """Load hard prediction regret curves, grouped by model."""
    prompt_dir = EXP_DIR / prompt
    if not prompt_dir.exists():
        print(f"No data for prompt '{prompt}'")
        return {}

    data = {}
    for model_dir in sorted(prompt_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        model_key = model_dir.name
        model_regrets = []
        true_regrets = []

        for case_dir in sorted(model_dir.iterdir()):
            if not case_dir.is_dir():
                continue
            for run_dir in sorted(case_dir.iterdir()):
                if not run_dir.is_dir():
                    continue
                result_f = run_dir / "result.json"
                if not result_f.exists():
                    continue
                try:
                    r = json.load(open(result_f))
                except:
                    continue
                meta = r.get("meta", {})
                if meta.get("n_steps") != n_steps:
                    continue

                inp = r.get("input", {})
                true_labels = inp.get("true_labels", [])
                expert_preds = inp.get("expert_predictions", [])
                if not true_labels or not expert_preds:
                    continue

                # model predictions
                preds = _extract_predictions(r)
                if preds is None or len(preds) != n_steps:
                    continue

                model_regret = _compute_hard_regret(preds, true_labels, expert_preds)
                model_regrets.append(model_regret)

                # true MW predictions
                case_data = {
                    "expert_predictions": expert_preds,
                    "true_labels": true_labels,
                    "losses": inp.get("losses", []),
                    "n_steps": n_steps,
                    "learning_rate": inp.get("eta", 0.1),
                }
                try:
                    mw_preds = _extract_mw_predictions(case_data)
                    true_regret = _compute_hard_regret(mw_preds, true_labels, expert_preds)
                    true_regrets.append(true_regret)
                except:
                    pass

                break  # only first run per case

        if model_regrets:
            data[model_key] = {
                "model_regret": model_regrets,
                "true_regret": true_regrets,
            }

    return data


def plot_regret_comparison(prompt, n_steps, save_path=None):
    data = load_regret_curves(prompt, n_steps)
    if not data:
        print(f"No data found for prompt='{prompt}', n_steps={n_steps}")
        return

    steps = np.arange(1, n_steps + 1)

    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot true MW first (from any model's data, they should be the same)
    for mk, d in data.items():
        if d["true_regret"]:
            true_arr = np.array(d["true_regret"])
            mean = true_arr.mean(axis=0)
            std = true_arr.std(axis=0)
            ax.plot(steps, mean, linewidth=2.5, color=MW_STYLE["color"],
                    label=MW_STYLE["label"], zorder=MW_STYLE["zorder"])
            ax.fill_between(steps, mean - std, mean + std,
                            alpha=0.15, color=MW_STYLE["color"], zorder=1)
            break

    # Plot each model
    for mk in sorted(data.keys()):
        d = data[mk]
        style = MODEL_STYLE.get(mk, {"label": mk, "color": "#888888", "zorder": 2})
        arr = np.array(d["model_regret"])
        # filter outliers
        final = arr[:, -1]
        med = np.median(final)
        keep = np.abs(final) <= max(10 * abs(med), 1000)
        if keep.sum() < arr.shape[0]:
            print(f"  {mk}: dropped {arr.shape[0] - keep.sum()} outlier case(s)")
            arr = arr[keep]
        n_cases = arr.shape[0]
        if n_cases == 0:
            continue
        mean = arr.mean(axis=0)
        std = arr.std(axis=0)

        label = f"{style['label']} (n={n_cases})"
        ax.plot(steps, mean, linewidth=2.5, color=style["color"],
                label=label, zorder=style["zorder"])
        ax.fill_between(steps, mean - std, mean + std,
                        alpha=0.15, color=style["color"], zorder=1)

    ax.set_xlabel("Step", fontsize=13)
    ax.set_ylabel("Cumulative Regret (hard 0-1)", fontsize=13)
    ax.set_title(f"Cumulative Regret — prompt: {prompt}, T={n_steps}", fontsize=14)
    ax.legend(fontsize=11, loc="upper left")
    ax.grid(alpha=0.25)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {save_path}")
    plt.show()
    plt.close()


if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) < 2:
        print(__doc__)
        sys.exit(1)
    prompt = args[0]
    n_steps = int(args[1])
    save_path = args[args.index("--save") + 1] if "--save" in args else None
    plot_regret_comparison(prompt, n_steps, save_path)
