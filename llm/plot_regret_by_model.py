"""
Plot cumulative regret comparison across prompts for a given model and step count.
Uses hard prediction regret (0-1 loss) for all prompts.

Usage:
  python plot_regret_by_model.py <model> <n_steps> [--save <path>]

Examples:
  python plot_regret_by_model.py ds 100
  python plot_regret_by_model.py gpt4o 100 --save plots/model_gpt4o_100.png
"""

import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

EXP_DIR = Path("exp_results")

PROMPT_STYLE = {
    # v1 prompts
    "online":                      {"label": "online",                  "color": "#E8740C"},
    "weather":                     {"label": "weather",                 "color": "#2CA02C"},
    "mw_name":                     {"label": "mw_name",                 "color": "#D62728"},
    "explicit_update":             {"label": "explicit_update",         "color": "#7A3E9D"},
    # v2 prompts — weather (greens/blues)
    "weather_v2":                  {"label": "weather +hint +note",     "color": "#1B9E77"},
    "weather_no_hint_v2":          {"label": "weather -hint +note",     "color": "#66A61E"},
    "weather_nonote_v2":           {"label": "weather +hint -note",     "color": "#7570B3"},
    "weather_nonote_nohint_v2":    {"label": "weather -hint -note",     "color": "#A6CEE3"},
    # v2 prompts — online (reds/oranges)
    "online_v2":                   {"label": "online +hint +note",      "color": "#D95F02"},
    "online_nohint_v2":            {"label": "online -hint +note",      "color": "#E7298A"},
    "online_nonote_v2":            {"label": "online +hint -note",      "color": "#E6AB02"},
    "online_nonote_nohint_v2":     {"label": "online -hint -note",      "color": "#A6761D"},
    # v1 ablation prompts
    "weather_no_hint":             {"label": "weather -hint",           "color": "#66C2A5"},
    "weather_nonote":              {"label": "weather nonote",          "color": "#FC8D62"},
    "weather_nonote_nohint":       {"label": "weather nonote -hint",    "color": "#8DA0CB"},
    "online_nohint":               {"label": "online -hint",            "color": "#E78AC3"},
    "online_nonote":               {"label": "online nonote",           "color": "#A6D854"},
    "online_nonote_nohint":        {"label": "online nonote -hint",     "color": "#FFD92F"},
}
MW_STYLE = {"label": "True MW (optimal)", "color": "#4C78A8", "zorder": 10}

MODEL_DISPLAY = {
    "ds": "DeepSeek V3",
    "ds_think": "DeepSeek R1",
    "gpt4o": "GPT-4o",
    "gpt4o_mini": "GPT-4o-mini",
    "gemini25_flash": "Gemini 2.5 Flash",
    "gemini3_flash": "Gemini 3 Flash",
}


def _compute_hard_regret(predictions, true_labels, expert_predictions):
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
    preds = r.get("response", {}).get("predictions")
    if preds is not None:
        return preds
    preds = r.get("response", {}).get("algorithm_predictions")
    if preds is not None:
        return preds
    return None


def _extract_mw_predictions(case_data):
    from mw_lib import get_ground_truth_outputs
    gt = get_ground_truth_outputs(case_data)
    return gt["algorithm_predictions"]


def load_regret_by_prompt(model_key, n_steps):
    data = {}
    for prompt_dir in sorted(EXP_DIR.iterdir()):
        if not prompt_dir.is_dir():
            continue
        prompt_short = prompt_dir.name
        model_dir = prompt_dir / model_key
        if not model_dir.is_dir():
            continue

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

                preds = _extract_predictions(r)
                if preds is None or len(preds) != n_steps:
                    continue

                model_regret = _compute_hard_regret(preds, true_labels, expert_preds)
                model_regrets.append(model_regret)

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

                break

        if model_regrets:
            data[prompt_short] = {
                "model_regret": model_regrets,
                "true_regret": true_regrets,
            }
    return data


def plot_model_comparison(model_key, n_steps, save_path=None):
    data = load_regret_by_prompt(model_key, n_steps)
    if not data:
        print(f"No data found for model='{model_key}', n_steps={n_steps}")
        return

    steps = np.arange(1, n_steps + 1)
    display_name = MODEL_DISPLAY.get(model_key, model_key)

    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot true MW
    for pn, d in data.items():
        if d["true_regret"]:
            true_arr = np.array(d["true_regret"])
            mean = true_arr.mean(axis=0)
            std = true_arr.std(axis=0)
            ax.plot(steps, mean, linewidth=2.5, color=MW_STYLE["color"],
                    label=MW_STYLE["label"], zorder=MW_STYLE["zorder"])
            ax.fill_between(steps, mean - std, mean + std,
                            alpha=0.15, color=MW_STYLE["color"], zorder=1)
            break

    # Plot each prompt
    for pn in sorted(data.keys()):
        d = data[pn]
        style = PROMPT_STYLE.get(pn, {"label": pn, "color": "#888888"})
        arr = np.array(d["model_regret"])
        # filter outliers
        final = arr[:, -1]
        med = np.median(final)
        keep = np.abs(final) <= max(10 * abs(med), 1000)
        if keep.sum() < arr.shape[0]:
            print(f"  {pn}: dropped {arr.shape[0] - keep.sum()} outlier case(s)")
            arr = arr[keep]
        n_cases = arr.shape[0]
        if n_cases == 0:
            continue
        mean = arr.mean(axis=0)
        std = arr.std(axis=0)

        label = f"{style['label']} (n={n_cases})"
        ax.plot(steps, mean, linewidth=2.5, color=style["color"],
                label=label, zorder=3)
        ax.fill_between(steps, mean - std, mean + std,
                        alpha=0.15, color=style["color"], zorder=1)

    ax.set_xlabel("Step", fontsize=13)
    ax.set_ylabel("Cumulative Regret (hard 0-1)", fontsize=13)
    ax.set_title(f"Cumulative Regret — {display_name}, T={n_steps}", fontsize=14)
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
    model = args[0]
    n_steps = int(args[1])
    save_path = args[args.index("--save") + 1] if "--save" in args else None
    plot_model_comparison(model, n_steps, save_path)
