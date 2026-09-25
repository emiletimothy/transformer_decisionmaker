import math
import json
import re
from pathlib import Path


# ── Data loading ──────────────────────────────────────────────────────────────

def load_data(json_path="mw_dataset.json"):
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_case(data, split, idx):
    assert split in data, f"Split '{split}' not found. Available: {list(data.keys())}"
    cases = data[split]
    assert idx < len(cases), f"Index {idx} out of range (split has {len(cases)} cases)"
    return cases[idx]


# ── Core MW algorithm ─────────────────────────────────────────────────────────

def normalize(weights):
    s = sum(weights)
    assert s > 0, "sum of weights must be positive"
    return [w / s for w in weights]


def run_multiplicative_weights(case):
    """
    Returns weights_sequence of length n_steps + 1.
    weights_sequence[t] = w_t (used at step t before update).
    """
    losses = case["losses"]
    eta = case["learning_rate"]
    n_steps = case["n_steps"]
    n_experts = len(losses[0])

    w = [1.0 / n_experts] * n_experts
    weights_sequence = [w.copy()]

    for t in range(n_steps):
        new_w = [w[i] * math.exp(-eta * losses[t][i]) for i in range(n_experts)]
        w = normalize(new_w)
        weights_sequence.append(w.copy())

    return weights_sequence


def get_ground_truth_outputs(case):
    """
    Returns:
      weights_sequence: length T+1
      mixture_probabilities: length T
      algorithm_predictions: length T
    """
    expert_predictions = case["expert_predictions"]
    n_steps = case["n_steps"]
    n_experts = len(expert_predictions[0])

    weights_sequence = run_multiplicative_weights(case)
    mixture_probabilities = []
    algorithm_predictions = []

    for t in range(n_steps):
        w_t = weights_sequence[t]
        preds_t = expert_predictions[t]
        q_t = sum(w_t[i] * preds_t[i] for i in range(n_experts))
        mixture_probabilities.append(q_t)
        algorithm_predictions.append(1 if q_t >= 0.5 else 0)

    return {
        "weights_sequence": weights_sequence,
        "mixture_probabilities": mixture_probabilities,
        "algorithm_predictions": algorithm_predictions,
    }


# ── Prompt building ───────────────────────────────────────────────────────────

PROMPTS_DIR = Path(__file__).parent / "prompts"
DEFAULT_PROMPT_NAME = "explicit_update"


def load_prompt_template(prompt_name=DEFAULT_PROMPT_NAME):
    path = PROMPTS_DIR / f"{prompt_name}.txt"
    return path.read_text()


def build_prompt(case, prompt_template=None, prompt_name=None):
    if prompt_template is None:
        prompt_template = load_prompt_template(prompt_name or DEFAULT_PROMPT_NAME)
    input_dict = {
        "n": len(case["expert_predictions"][0]),
        "T": case["n_steps"],
        "eta": case["learning_rate"],
        "expert_predictions": case["expert_predictions"],
        "true_labels": case["true_labels"],
    }
    return prompt_template.replace("{input_json}", json.dumps(input_dict, indent=2))


# ── Parsing model response ────────────────────────────────────────────────────

def parse_model_response(text):
    """
    Parse model output: handles raw JSON or markdown ```json ... ``` blocks.
    Also fixes uppercase scientific notation (e.g. 2.16E-05 -> 2.16e-05).
    Returns a dict.
    """
    text = text.strip()
    # strip markdown code fences if present
    match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if match:
        text = match.group(1).strip()
    # fix uppercase E in scientific notation (e.g. 1.5E-06 -> 1.5e-06)
    text = re.sub(r'(\d)(E)([-+]?\d+)', r'\1e\3', text)
    # try parsing; if it fails, attempt a targeted repair
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # fix spurious trailing ] on a flat-values line (no opening [ on that line)
    # e.g. "    1.0, 0.5, 0.3]\n  ]," → "    1.0, 0.5, 0.3\n  ],"
    text = re.sub(r'^( +)([^"\[\{]*\d)\](\s*)$', r'\1\2\3', text, flags=re.MULTILINE)
    return json.loads(text)


# ── Evaluation ────────────────────────────────────────────────────────────────

def check_weight_normalization(model_output, tol=1e-6):
    bad_steps = []
    max_dev = 0.0
    for t, w in enumerate(model_output["weights_sequence"]):
        dev = abs(sum(w) - 1.0)
        max_dev = max(max_dev, dev)
        if dev > tol:
            bad_steps.append((t, round(sum(w), 8)))
    return {
        "all_normalized": len(bad_steps) == 0,
        "bad_steps": bad_steps,
        "max_deviation": max_dev,
    }


def _mean_std(vals):
    n = len(vals)
    if n == 0: return None, None
    m = sum(vals) / n
    s = (sum((x - m) ** 2 for x in vals) / n) ** 0.5
    return m, s


def analyze_model_output(case, model_output, norm_tol=1e-3, prob_tol=1e-3):
    if isinstance(model_output, str):
        model_output = parse_model_response(model_output)

    gt = get_ground_truth_outputs(case)

    pred_weights = model_output["weights_sequence"]
    pred_probs   = model_output["mixture_probabilities"]
    # sanitize: if model returned a list instead of float for q, take first element or 0
    pred_probs = [p[0] if isinstance(p, list) else p for p in pred_probs]
    pred_preds   = model_output["algorithm_predictions"]
    true_weights = gt["weights_sequence"]
    true_preds   = gt["algorithm_predictions"]
    expert_preds = case["expert_predictions"]
    true_labels  = case["true_labels"]

    n_experts = len(expert_preds[0])
    uniform   = [1.0 / n_experts] * n_experts
    # replace empty / wrong-length weight vectors with uniform
    pred_weights = [w if len(w) == n_experts else uniform for w in pred_weights]

    T   = case["n_steps"]
    T_w = min(len(pred_weights), len(true_weights))
    T_p = min(len(pred_probs),   T)
    T_y = min(len(pred_preds),   T)

    # ── weights ──────────────────────────────────────────────────────────────
    # check_norm: does each weight vector sum to 1?
    bad_norm = sum(1 for w in pred_weights if abs(sum(w) - 1.0) > norm_tol)
    check_norm = "ok" if bad_norm == 0 else f"err {bad_norm}/{len(pred_weights)} steps ({bad_norm/len(pred_weights):.2%})"

    # ell1 vs ground truth
    ell1_per_step = [
        sum(abs(a - b) for a, b in zip(pred_weights[t], true_weights[t]))
        for t in range(T_w)
    ]
    mean_ell1, std_ell1 = _mean_std(ell1_per_step)

    # ── mixture_probabilities ─────────────────────────────────────────────────
    # check_compute: does model q_t == sum(model_w_t * expert_pred_t)?
    compute_errs = [
        abs(pred_probs[t] - sum(pred_weights[t][i] * expert_preds[t][i]
                                for i in range(len(pred_weights[t]))))
        for t in range(T_p)
    ]
    mean_ce, std_ce = _mean_std(compute_errs)
    check_compute = "ok" if all(e <= prob_tol for e in compute_errs) \
                    else f"err mean={mean_ce:.4f} std={std_ce:.4f}"

    # abs error vs ground truth q
    true_probs = gt["mixture_probabilities"]
    abs_err_per_step = [abs(pred_probs[t] - true_probs[t]) for t in range(T_p)]
    mean_abs, std_abs = _mean_std(abs_err_per_step)

    # ── algorithm_predictions ─────────────────────────────────────────────────
    # check_pred: does model pred match threshold(model q)?
    pred_mismatch = sum(
        1 for t in range(T_y)
        if pred_preds[t] != (1 if pred_probs[t] >= 0.5 else 0)
    )
    check_pred = "ok" if pred_mismatch == 0 else f"err {pred_mismatch/T_y:.2%}"

    model_loss_per_step = [int(pred_preds[t] != true_labels[t]) for t in range(T_y)]
    true_loss_per_step  = [int(true_preds[t]  != true_labels[t]) for t in range(T_y)]
    accuracy = sum(l == 0 for l in model_loss_per_step) / T_y if T_y else None

    # regret sequences
    model_regret_stats = _regret_from_weights(case, pred_weights[:T_w])
    true_regret_stats  = compute_regret_curve(case)
    model_regret = model_regret_stats["regret_curve"]
    true_regret  = true_regret_stats["regret_curve"]

    return {
        "weights": {
            "check_norm":     check_norm,
            "ell1_per_step":  ell1_per_step,
            "mean_ell1":      mean_ell1,
            "std_ell1":       std_ell1,
        },
        "mixture_probabilities": {
            "check_compute":    check_compute,
            "abs_err_per_step": abs_err_per_step,
            "mean_abs_error":   mean_abs,
            "std_abs_error":    std_abs,
        },
        "algorithm_predictions": {
            "check_pred":          check_pred,
            "model_loss_per_step": model_loss_per_step,
            "true_loss_per_step":  true_loss_per_step,
            "accuracy":            accuracy,
            "model_regret":        model_regret,
            "true_regret":         true_regret,
        },
    }


# ── Regret computation ────────────────────────────────────────────────────────

def compute_regret_curve(case):
    return _regret_from_weights(case, run_multiplicative_weights(case))


def compute_regret_curve_from_weights(case, weights_sequence):
    return _regret_from_weights(case, weights_sequence)


def _regret_from_weights(case, weights_sequence):
    losses   = case["losses"]
    n_steps  = case["n_steps"]
    n_experts = len(losses[0])

    alg_cum = 0.0
    expert_cum = [0.0] * n_experts
    alg_cum_losses, best_expert_cum_losses, regret_curve = [], [], []
    expert_cum_losses = [[] for _ in range(n_experts)]

    uniform = [1.0 / n_experts] * n_experts
    for t in range(n_steps):
        w_t = weights_sequence[t] if len(weights_sequence[t]) == n_experts else uniform
        l_t = losses[t]
        alg_cum += sum(w_t[i] * l_t[i] for i in range(n_experts))
        for i in range(n_experts):
            expert_cum[i] += l_t[i]
            expert_cum_losses[i].append(expert_cum[i])
        best = min(expert_cum)
        alg_cum_losses.append(alg_cum)
        best_expert_cum_losses.append(best)
        regret_curve.append(alg_cum - best)

    return {
        "steps": list(range(1, n_steps + 1)),
        "regret_curve": regret_curve,
        "alg_cum_losses": alg_cum_losses,
        "best_expert_cum_losses": best_expert_cum_losses,
        "expert_cum_losses": expert_cum_losses,
    }


# ── Plotting ──────────────────────────────────────────────────────────────────

def _hard_regret_from_weights(case, weights_sequence):
    """Compute cumulative 0-1 loss and regret for hard predictions derived from weights."""
    expert_preds = case["expert_predictions"]
    true_labels  = case["true_labels"]
    losses       = case["losses"]
    n_steps      = case["n_steps"]
    n_experts    = len(losses[0])

    T = min(len(weights_sequence) - 1, n_steps, len(expert_preds), len(true_labels))
    alg_cum = 0.0
    expert_cum = [0.0] * n_experts
    regret_curve, loss_curve = [], []
    steps = []

    for t in range(T):
        w_t   = weights_sequence[t]
        q_t   = sum(w_t[i] * expert_preds[t][i] for i in range(len(w_t)))
        yhat  = 1 if q_t >= 0.5 else 0
        alg_cum += int(yhat != true_labels[t])
        for i in range(n_experts):
            expert_cum[i] += int(expert_preds[t][i] != true_labels[t])
        best = min(expert_cum)
        loss_curve.append(alg_cum)
        regret_curve.append(alg_cum - best)
        steps.append(t + 1)

    return {"steps": steps, "regret_curve": regret_curve, "loss_curve": loss_curve}


def plot_regret_comparison(case, model_output, save_path=None):
    import matplotlib.pyplot as plt
    true_stats  = compute_regret_curve(case)
    model_stats = _regret_from_weights(case, model_output["weights_sequence"])
    steps = true_stats["steps"]

    true_regret  = true_stats["regret_curve"]
    model_regret = model_stats["regret_curve"]
    true_avg  = [r / t for r, t in zip(true_regret,  steps)]
    model_avg = [r / t for r, t in zip(model_regret, steps)]

    fig, ax1 = plt.subplots(figsize=(10, 5))
    l1 = ax1.plot(steps, true_regret,  linewidth=2.5, color="#7A3E9D", label="True MW cumulative regret")
    l2 = ax1.plot(steps, model_regret, linewidth=2.5, color="#C17CEB", linestyle="--", label="Model (soft) cumulative regret")
    ax1.set_xlabel("Step"); ax1.set_ylabel("Cumulative Regret"); ax1.grid(alpha=0.25)

    ax2 = ax1.twinx()
    l4 = ax2.plot(steps, true_avg,  linewidth=2.5, color="#4C78A8", label="True MW average regret")
    l5 = ax2.plot(steps, model_avg, linewidth=2.5, color="#72B7F2", linestyle="--", label="Model (soft) average regret")
    ax2.set_ylabel("Average Regret")

    lines = l1 + l2 + l4 + l5
    ax1.legend(lines, [l.get_label() for l in lines], loc="upper right")
    plt.title("Regret Comparison: True MW vs Model MW")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()


def plot_weight_comparison(case, model_output, save_path=None):
    import matplotlib.pyplot as plt
    true_weights  = run_multiplicative_weights(case)
    model_weights = model_output["weights_sequence"]
    n_steps   = case["n_steps"]
    n_experts = len(case["losses"][0])
    steps  = list(range(n_steps + 1))
    true_colors  = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd"]  # dark: blue, red, green, purple
    model_colors = ["#6baed6", "#fc8d59", "#74c476", "#c5b0d5"]  # light: blue, orange, green, purple

    model_T = min(len(model_weights), n_steps + 1)
    plt.figure(figsize=(10, 6))
    for i in range(n_experts):
        plt.plot(steps, [true_weights[t][i] for t in range(n_steps + 1)],
                 linewidth=2.5, color=true_colors[i], linestyle="-",
                 label=f"Expert {i} true")
    for i in range(n_experts):
        plt.plot(steps[:model_T], [model_weights[t][i] for t in range(model_T)],
                 linewidth=2.0, color=model_colors[i], linestyle="--",
                 label=f"Expert {i} model")

    plt.xlabel("Step"); plt.ylabel("Weight")
    plt.title("Weight Comparison: True MW vs Model MW")
    plt.grid(alpha=0.25); plt.legend(ncol=2)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()


def plot_cumulative_loss(analysis, case=None, model_output=None, save_path=None):
    import matplotlib.pyplot as plt

    model_loss = analysis["algorithm_predictions"]["model_loss_per_step"]
    true_loss  = analysis["algorithm_predictions"]["true_loss_per_step"]
    steps = list(range(1, len(model_loss) + 1))

    model_cum = []
    true_cum  = []
    ms, ts = 0, 0
    for m, t in zip(model_loss, true_loss):
        ms += m; ts += t
        model_cum.append(ms)
        true_cum.append(ts)

    plt.figure(figsize=(10, 5))
    plt.plot(steps, true_cum,  linewidth=2.5, color="#4C78A8", label="True MW cumulative loss")
    plt.plot(steps, model_cum, linewidth=2.5, color="#7A3E9D", marker="s", markersize=5, label="Model (stated) cumulative loss")

    if case is not None and model_output is not None:
        recomp = _hard_regret_from_weights(case, model_output["weights_sequence"])
        plt.plot(recomp["steps"], recomp["loss_curve"],
                 linewidth=2.0, color="#E8740C", linestyle="--",
                 label="Model weights → hard pred cumulative loss")

    plt.xlabel("Step"); plt.ylabel("Cumulative Loss")
    plt.title("Cumulative 0-1 Loss: Model vs True MW")
    plt.grid(alpha=0.25); plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()


# ── Online (prediction-only) analysis & plotting ────────────────────────────

def analyze_online_output(case, model_predictions):
    """Analyze model output when only hard predictions are available (no weights)."""
    true_labels  = case["true_labels"]
    expert_preds = case["expert_predictions"]
    n_steps      = case["n_steps"]
    n_experts    = len(expert_preds[0])
    T = min(len(model_predictions), n_steps)

    # model cumulative 0-1 loss
    model_loss_per_step = [int(model_predictions[t] != true_labels[t]) for t in range(T)]
    model_cum_loss = []
    s = 0
    for l in model_loss_per_step:
        s += l
        model_cum_loss.append(s)

    # expert cumulative 0-1 losses
    expert_cum = [0] * n_experts
    expert_cum_losses = [[] for _ in range(n_experts)]
    best_expert_cum = []
    for t in range(T):
        for i in range(n_experts):
            expert_cum[i] += int(expert_preds[t][i] != true_labels[t])
            expert_cum_losses[i].append(expert_cum[i])
        best_expert_cum.append(min(expert_cum))

    # regret = model cumulative loss − best expert cumulative loss
    regret_curve = [model_cum_loss[t] - best_expert_cum[t] for t in range(T)]

    # true MW (hard predictions) for comparison
    gt = get_ground_truth_outputs(case)
    true_preds = gt["algorithm_predictions"]
    true_loss_per_step = [int(true_preds[t] != true_labels[t]) for t in range(T)]
    true_cum_loss = []
    s = 0
    for l in true_loss_per_step:
        s += l
        true_cum_loss.append(s)
    true_regret = [true_cum_loss[t] - best_expert_cum[t] for t in range(T)]

    accuracy = sum(1 for l in model_loss_per_step if l == 0) / T if T else None

    return {
        "n_steps_evaluated": T,
        "accuracy":            accuracy,
        "model_loss_per_step": model_loss_per_step,
        "model_cum_loss":      model_cum_loss,
        "true_loss_per_step":  true_loss_per_step,
        "true_cum_loss":       true_cum_loss,
        "expert_cum_losses":   expert_cum_losses,
        "best_expert_cum":     best_expert_cum,
        "regret_curve":        regret_curve,
        "true_regret_curve":   true_regret,
    }


def plot_online_regret(analysis, save_path=None):
    """Regret comparison for online (prediction-only) models."""
    import matplotlib.pyplot as plt
    T     = analysis["n_steps_evaluated"]
    steps = list(range(1, T + 1))

    model_regret = analysis["regret_curve"]
    true_regret  = analysis["true_regret_curve"]
    model_avg = [r / t for r, t in zip(model_regret, steps)]
    true_avg  = [r / t for r, t in zip(true_regret,  steps)]

    fig, ax1 = plt.subplots(figsize=(10, 5))
    l1 = ax1.plot(steps, true_regret,  linewidth=2.5, color="#7A3E9D",
                  label="True MW cumulative regret")
    l2 = ax1.plot(steps, model_regret, linewidth=2.5, color="#C17CEB", linestyle="--",
                  label="Model cumulative regret")
    ax1.set_xlabel("Step"); ax1.set_ylabel("Cumulative Regret"); ax1.grid(alpha=0.25)

    ax2 = ax1.twinx()
    l3 = ax2.plot(steps, true_avg,  linewidth=2.5, color="#4C78A8",
                  label="True MW average regret")
    l4 = ax2.plot(steps, model_avg, linewidth=2.5, color="#72B7F2", linestyle="--",
                  label="Model average regret")
    ax2.set_ylabel("Average Regret")

    lines = l1 + l2 + l3 + l4
    ax1.legend(lines, [l.get_label() for l in lines], loc="upper right")
    plt.title("Regret: Model vs True MW (hard predictions)")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()


def plot_online_loss(analysis, save_path=None):
    """Cumulative loss for online (prediction-only) models."""
    import matplotlib.pyplot as plt
    T     = analysis["n_steps_evaluated"]
    steps = list(range(1, T + 1))

    plt.figure(figsize=(10, 5))
    plt.plot(steps, analysis["true_cum_loss"],  linewidth=2.5, color="#4C78A8",
             label="True MW cumulative loss")
    plt.plot(steps, analysis["model_cum_loss"], linewidth=2.5, color="#7A3E9D",
             marker="s", markersize=3, label="Model cumulative loss")
    plt.plot(steps, analysis["best_expert_cum"], linewidth=2.0, color="#2CA02C",
             linestyle=":", label="Best expert cumulative loss")

    plt.xlabel("Step"); plt.ylabel("Cumulative Loss")
    plt.title("Cumulative 0-1 Loss: Model vs True MW vs Best Expert")
    plt.grid(alpha=0.25); plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()
