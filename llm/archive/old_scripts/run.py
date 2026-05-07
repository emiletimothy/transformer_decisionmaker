"""
Usage:
  python run.py generate <split> <idx> [--mode manual|api]
  python run.py evaluate <split> <idx> [--mode manual|api] [--plot-weights]

Mode controls which results folder is used:
  --mode manual   results/manual/   (default for generate/evaluate)
  --mode api      results/api/

Examples:
  python run.py generate train 0
  python run.py evaluate train 0
  python run.py evaluate train 0 --mode api
"""

import sys
import json
from pathlib import Path
from mw_lib import (
    load_data, get_case, build_prompt, parse_model_response,
    analyze_model_output,
    plot_regret_comparison, plot_weight_comparison, plot_cumulative_loss,
)

DATASET_PATH = "mw_dataset.json"
RESULTS_DIR  = Path("results")


def case_dir(split, idx, mode, model_name=None):
    if mode == "api" and model_name:
        # results/api/train_074/ds/
        return RESULTS_DIR / "api" / f"{split}_{idx:03d}" / model_name
    elif mode == "manual":
        return RESULTS_DIR / "manual" / f"{split}_{idx:03d}"
    else:
        return RESULTS_DIR / mode / f"{split}_{idx:03d}"


def next_run_dir(d, split, idx, model_name=None):
    """Return the next run folder path."""
    if model_name:
        # train_074_ds_run001
        prefix = f"{split}_{idx:03d}_{model_name}_run"
        existing = sorted(d.glob(f"{prefix}*/"))
        if not existing:
            return d / f"{prefix}001"
        last = int(existing[-1].name.replace(prefix, ""))
        return d / f"{prefix}{last + 1:03d}"
    else:
        # legacy: run_074_001
        existing = sorted(d.glob(f"run_{idx:03d}_*/"))
        if not existing:
            return d / f"run_{idx:03d}_001"
        last = int(existing[-1].name.split("_")[2])
        return d / f"run_{idx:03d}_{last + 1:03d}"


# ── generate ──────────────────────────────────────────────────────────────────

def cmd_generate(split, idx, mode, model_name=None):
    data = load_data(DATASET_PATH)
    case = get_case(data, split, idx)

    d = case_dir(split, idx, mode, model_name)
    d.mkdir(parents=True, exist_ok=True)

    (d / "case.json").write_text(json.dumps(case, indent=2))

    prompt = build_prompt(case)
    (d / "prompt.txt").write_text(prompt)

    response_path = d / "response.json"
    if not response_path.exists():
        response_path.write_text("{}\n")

    print(f"Case saved to: {d}/")
    print(f"  prompt.txt    <- copy this to the LLM")
    print(f"  response.json <- paste LLM output here, then run evaluate")
    print()
    print("=" * 60)
    print(prompt)
    print("=" * 60)


# ── evaluate ──────────────────────────────────────────────────────────────────

def cmd_evaluate(split, idx, mode, model_output=None, model_name=None, plot_weights=False):
    data = load_data(DATASET_PATH)
    case = get_case(data, split, idx)

    d = case_dir(split, idx, mode, model_name)

    if model_output is None:
        # manual mode: read from response.json
        response_path = d / "response.json"
        if not response_path.exists():
            print(f"ERROR: {response_path} not found.")
            print(f"Run 'python run.py generate {split} {idx}' first.")
            sys.exit(1)

        raw = response_path.read_text().strip()
        if not raw or raw == "{}":
            print("ERROR: response.json is empty. Paste the model's JSON output into it first.")
            sys.exit(1)

        try:
            model_output = parse_model_response(raw)
        except Exception as e:
            print(f"ERROR: Could not parse response.json as JSON.\n  {e}")
            sys.exit(1)

    d.mkdir(parents=True, exist_ok=True)

    # create numbered run folder
    run_dir = next_run_dir(d, split, idx, model_name)
    run_dir.mkdir()

    # archive the response
    (run_dir / "response.json").write_text(json.dumps(model_output, indent=2))

    # correctness analysis
    summary = analyze_model_output(case, model_output)

    # ground truth MW outputs
    from mw_lib import get_ground_truth_outputs
    ground_truth = get_ground_truth_outputs(case)

    # save result (with case input + model response + ground truth bundled in)
    result = {
        "meta": {
            "split":   split,
            "idx":     idx,
            "run":     run_dir.name,
            "mode":    mode,
            "n_steps": case["n_steps"],
        },
        "analysis": summary,
        "input": {
            "n":                  len(case["expert_predictions"][0]),
            "T":                  case["n_steps"],
            "eta":                case["learning_rate"],
            "expert_predictions": case["expert_predictions"],
            "true_labels":        case["true_labels"],
            "losses":             case["losses"],
        },
        "ground_truth": ground_truth,
        "response": model_output,
    }
    result_path = run_dir / "result.json"
    result_path.write_text(json.dumps(result, indent=2))

    # generate report notebook
    nb_path = run_dir / "report.ipynb"
    _write_report_notebook(nb_path, split, idx, run_dir)

    # print summary
    print(f"Run saved to: {run_dir}/")
    print()
    print("=" * 60)
    print("ANALYSIS")
    print("=" * 60)
    w = summary["weights"]
    p = summary["mixture_probabilities"]
    y = summary["algorithm_predictions"]

    print(f"  weights.check_norm    : {w['check_norm']}")
    print(f"  weights.mean_ell1     : {w['mean_ell1']:.4f}  std={w['std_ell1']:.4f}")
    print()
    print(f"  prob.check_compute    : {p['check_compute']}")
    print(f"  prob.mean_abs_error   : {p['mean_abs_error']:.4f}  std={p['std_abs_error']:.4f}")
    print()
    print(f"  pred.check_pred       : {y['check_pred']}")
    print(f"  pred.accuracy         : {y['accuracy']:.3f}")

    print()
    print(f"report.ipynb saved to: {nb_path}")
    print("Open it in JupyterLab and Run All to see the plots.")

    # save and show plots
    plot_regret_comparison(case, model_output, save_path=run_dir / "plot_regret.png")
    if plot_weights:
        plot_weight_comparison(case, model_output, save_path=run_dir / "plot_weights.png")
    plot_cumulative_loss(summary, case=case, model_output=model_output, save_path=run_dir / "plot_loss.png")
    plots_list = "plot_regret.png, plot_loss.png" + (", plot_weights.png" if plot_weights else "")
    print(f"Plots saved: {plots_list}")

    return run_dir, result


# ── report notebook ───────────────────────────────────────────────────────────

def _write_report_notebook(nb_path, split, idx, run_dir):
    import nbformat

    abs_lib     = str(Path(__file__).resolve().parent)
    abs_dataset = str(Path(DATASET_PATH).resolve())
    abs_run     = str(run_dir.resolve())

    cells = []
    def code(src): cells.append(nbformat.v4.new_code_cell(src))
    def md(src):   cells.append(nbformat.v4.new_markdown_cell(src))

    md(f"# Report: {split}_{idx:03d} / {run_dir.name}")

    code(
        f"import sys, json\n"
        f"sys.path.insert(0, {abs_lib!r})\n"
        f"from mw_lib import (\n"
        f"    load_data, get_case,\n"
        f"    check_weight_normalization, analyze_model_output,\n"
        f"    plot_regret_comparison, plot_weight_comparison,\n"
        f")\n"
        f"%matplotlib inline\n"
        f"\n"
        f"result       = json.load(open({abs_run + '/result.json'!r}))\n"
        f"model_output = result['response']\n"
        f"inp          = result['input']\n"
        f"\n"
        f"# reconstruct case for plot functions\n"
        f"data = load_data({abs_dataset!r})\n"
        f"case = get_case(data, {split!r}, {idx})\n"
    )

    md("## Case Info")
    code(
        f"print(f\"split      : {{result['meta']['split']}}\")\n"
        f"print(f\"case index : {{result['meta']['idx']}}\")\n"
        f"print(f\"run        : {{result['meta']['run']}}\")\n"
        f"print(f\"mode       : {{result['meta'].get('mode', 'manual')}}\")\n"
        f"print(f\"n_steps    : {{inp['T']}}\")\n"
        f"print(f\"n_experts  : {{inp['n']}}\")\n"
        f"print(f\"eta        : {{inp['eta']:.6f}}\")\n"
    )

    md("## Weight Normalization")
    code(
        "norm = check_weight_normalization(model_output)\n"
        "print('All normalized :', norm['all_normalized'])\n"
        "print('Max deviation  :', f\"{norm['max_deviation']:.2e}\")\n"
        "if norm['bad_steps']:\n"
        "    print('Bad steps      :', norm['bad_steps'][:10])\n"
    )

    md("## Correctness Summary")
    code(
        "w = result['analysis']['weights']\n"
        "p = result['analysis']['mixture_probabilities']\n"
        "y = result['analysis']['algorithm_predictions']\n"
        "\n"
        "print(f\"weights.check_norm    : {w['check_norm']}\")\n"
        "print(f\"weights.mean_ell1     : {w['mean_ell1']:.4f}  std={w['std_ell1']:.4f}\")\n"
        "print()\n"
        "print(f\"prob.check_compute    : {p['check_compute']}\")\n"
        "print(f\"prob.mean_abs_error   : {p['mean_abs_error']:.4f}  std={p['std_abs_error']:.4f}\")\n"
        "print()\n"
        "print(f\"pred.check_pred       : {y['check_pred']}\")\n"
        "print(f\"pred.accuracy         : {y['accuracy']:.3f}\")\n"
    )

    md("## Regret Comparison")
    code("plot_regret_comparison(case, model_output)")

    md("## Weight Comparison")
    code("plot_weight_comparison(case, model_output)")

    nb = nbformat.v4.new_notebook(cells=cells)
    nb.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    nb_path.write_text(nbformat.writes(nb))


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    args = sys.argv[1:]
    if len(args) < 3:
        print(__doc__)
        sys.exit(1)

    cmd   = args[0]
    split = args[1]
    idx   = int(args[2])
    mode  = "manual"
    if "--mode" in args:
        mode = args[args.index("--mode") + 1]
    plot_weights = "--plot-weights" in args

    if cmd == "generate":
        cmd_generate(split, idx, mode)
    elif cmd == "evaluate":
        cmd_evaluate(split, idx, mode, plot_weights=plot_weights)
    else:
        print(f"Unknown command: {cmd}")
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
