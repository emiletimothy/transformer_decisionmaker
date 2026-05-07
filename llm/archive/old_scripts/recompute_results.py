"""
Recompute all result.json analysis using hard predictions, regenerate plots.
Does NOT touch response/conversation/steps data.
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # no display
import matplotlib.pyplot as plt
from pathlib import Path
from mw_lib import get_ground_truth_outputs, run_multiplicative_weights

EXP_DIR = Path("exp_results")


def compute_hard_regret(preds, true_labels, expert_preds):
    T = len(preds)
    n_exp = len(expert_preds[0])
    model_cum, expert_cum = 0, [0] * n_exp
    model_cum_loss, best_expert_cum, regret = [], [], []
    for t in range(T):
        model_cum += int(preds[t] != true_labels[t])
        for i in range(n_exp):
            expert_cum[i] += int(expert_preds[t][i] != true_labels[t])
        model_cum_loss.append(model_cum)
        best_expert_cum.append(min(expert_cum))
        regret.append(model_cum - min(expert_cum))
    return {
        "model_cum_loss": model_cum_loss,
        "best_expert_cum": best_expert_cum,
        "regret_curve": regret,
    }


def plot_hard_regret(model_regret, mw_regret, T, save_path):
    steps = np.arange(1, T + 1)
    fig, ax1 = plt.subplots(figsize=(10, 5))
    l1 = ax1.plot(steps, mw_regret["regret_curve"], lw=2.5, color="#4C78A8",
                  label="True MW cumulative regret")
    l2 = ax1.plot(steps, model_regret["regret_curve"], lw=2.5, color="#C17CEB",
                  linestyle="--", label="Model cumulative regret")
    ax1.set_xlabel("Step"); ax1.set_ylabel("Cumulative Regret"); ax1.grid(alpha=0.25)

    ax2 = ax1.twinx()
    model_avg = [r / t for r, t in zip(model_regret["regret_curve"], steps)]
    mw_avg = [r / t for r, t in zip(mw_regret["regret_curve"], steps)]
    l3 = ax2.plot(steps, mw_avg, lw=2.5, color="#4C78A8", alpha=0.5,
                  linestyle=":", label="True MW avg regret")
    l4 = ax2.plot(steps, model_avg, lw=2.5, color="#C17CEB", alpha=0.5,
                  linestyle=":", label="Model avg regret")
    ax2.set_ylabel("Average Regret")

    lines = l1 + l2 + l3 + l4
    ax1.legend(lines, [l.get_label() for l in lines], loc="upper right")
    plt.title("Regret (hard 0-1 prediction)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_hard_loss(model_regret, mw_regret, T, save_path):
    steps = np.arange(1, T + 1)
    plt.figure(figsize=(10, 5))
    plt.plot(steps, mw_regret["model_cum_loss"], lw=2.5, color="#4C78A8",
             label="True MW cumulative loss")
    plt.plot(steps, model_regret["model_cum_loss"], lw=2.5, color="#7A3E9D",
             marker="s", markersize=3, label="Model cumulative loss")
    plt.plot(steps, model_regret["best_expert_cum"], lw=2.0, color="#2CA02C",
             linestyle=":", label="Best expert cumulative loss")
    plt.xlabel("Step"); plt.ylabel("Cumulative Loss")
    plt.title("Cumulative 0-1 Loss")
    plt.grid(alpha=0.25); plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_weights(true_weights, model_weights, T, n_experts, save_path):
    steps = list(range(T + 1))
    true_colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd"]
    model_colors = ["#6baed6", "#fc8d59", "#74c476", "#c5b0d5"]

    model_T = min(len(model_weights), T + 1)
    plt.figure(figsize=(10, 6))
    for i in range(n_experts):
        plt.plot(steps, [true_weights[t][i] for t in range(T + 1)],
                 lw=2.5, color=true_colors[i % 4], linestyle="-",
                 label=f"Expert {i} true")
    for i in range(n_experts):
        plt.plot(steps[:model_T], [model_weights[t][i] for t in range(model_T)],
                 lw=2.0, color=model_colors[i % 4], linestyle="--",
                 label=f"Expert {i} model")
    plt.xlabel("Step"); plt.ylabel("Weight")
    plt.title("Weight Comparison: True MW vs Model")
    plt.grid(alpha=0.25); plt.legend(ncol=2)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def process_run(run_dir):
    result_f = run_dir / "result.json"
    r = json.load(open(result_f))
    meta = r.get("meta", {})
    inp = r.get("input", {})
    T = meta.get("n_steps", 0)
    if T == 0:
        return False

    true_labels = inp.get("true_labels", [])[:T]
    expert_preds = inp.get("expert_predictions", [])[:T]
    if not true_labels or not expert_preds:
        return False

    # Get model predictions
    preds = r.get("response", {}).get("predictions")
    if preds is None:
        preds = r.get("response", {}).get("algorithm_predictions")
    if preds is None or len(preds) < T:
        return False
    preds = preds[:T]

    # Build case for MW computation
    case = {
        "expert_predictions": expert_preds,
        "true_labels": true_labels,
        "losses": inp.get("losses", [])[:T],
        "n_steps": T,
        "learning_rate": inp.get("eta", 0.1),
    }

    # True MW predictions
    gt = get_ground_truth_outputs(case)
    mw_preds = gt["algorithm_predictions"]

    # Hard regret
    model_regret = compute_hard_regret(preds, true_labels, expert_preds)
    mw_regret = compute_hard_regret(mw_preds, true_labels, expert_preds)

    # Update analysis in result.json
    n_exp = len(expert_preds[0])
    model_loss = sum(int(preds[t] != true_labels[t]) for t in range(T))
    accuracy = 1 - model_loss / T

    r["hard_analysis"] = {
        "accuracy": accuracy,
        "model_loss": model_loss,
        "best_expert_loss": model_regret["best_expert_cum"][-1],
        "final_regret": model_regret["regret_curve"][-1],
        "mw_accuracy": 1 - mw_regret["model_cum_loss"][-1] / T,
        "mw_final_regret": mw_regret["regret_curve"][-1],
        "regret_curve": model_regret["regret_curve"],
        "mw_regret_curve": mw_regret["regret_curve"],
    }

    # Save updated result.json
    result_f.write_text(json.dumps(r, indent=2))

    # Regenerate plots
    plot_hard_regret(model_regret, mw_regret, T, run_dir / "plot_regret.png")
    plot_hard_loss(model_regret, mw_regret, T, run_dir / "plot_loss.png")

    # Weight plot for mw_name/explicit_update
    prompt = meta.get("prompt", "").removeprefix("interactive_")
    if prompt in ("mw_name", "explicit_update"):
        model_weights = r.get("response", {}).get("weights_sequence", [])
        if model_weights and len(model_weights) > 0:
            true_weights = run_multiplicative_weights(case)
            # pad model weights if needed
            n_experts = len(expert_preds[0])
            uniform = [1.0 / n_experts] * n_experts
            model_weights = [w if len(w) == n_experts else uniform for w in model_weights]
            plot_weights(true_weights, model_weights, T, n_experts,
                        run_dir / "plot_weights.png")

    return True


# Main
processed = 0
errors = 0
for root, subdirs, files in os.walk(EXP_DIR):
    if "result.json" not in files:
        continue
    run_dir = Path(root)
    try:
        if process_run(run_dir):
            processed += 1
            if processed % 10 == 0:
                print(f"  processed {processed}...", flush=True)
    except Exception as e:
        print(f"  ERROR: {run_dir}: {e}")
        errors += 1

print(f"\nDone: {processed} processed, {errors} errors")
