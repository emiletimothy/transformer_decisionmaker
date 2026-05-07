"""
Plot cumulative regret curves for LLM experiments with baselines.

Usage:
  python plot_regret.py --exp_dir exp_results --dataset mw_dataset.json --model ds --save plot.png
  python plot_regret.py --exp_dir exp_results --dataset mw_dataset.json --model ds qwen3_14b_nothink --save plot.png

Reads results from exp_dir/{prompt}/{model_key}/cases_XXX/run001/result.json
and computes baselines (MW, FTL, FPW, MajVote, Random) from the dataset.
"""

import argparse
import json
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path


# ── helpers ──────────────────────────────────────────────────────────────────

def compute_hard_regret(preds, true_labels, expert_preds):
    n_exp = len(expert_preds[0])
    model_cum, expert_cum = 0, [0] * n_exp
    regret = []
    for t in range(len(preds)):
        model_cum += int(preds[t] != true_labels[t])
        for i in range(n_exp):
            expert_cum[i] += int(expert_preds[t][i] != true_labels[t])
        regret.append(model_cum - min(expert_cum))
    return regret


def normalize(w):
    s = sum(w)
    return [x / s for x in w] if s > 0 else [1.0 / len(w)] * len(w)


def mw_predictions(case, T):
    ep = case['expert_predictions'][:T]
    tl = case['true_labels'][:T]
    losses = case['losses'][:T]
    eta = case['learning_rate']
    n_exp = len(ep[0])
    w = [1.0 / n_exp] * n_exp
    preds = []
    for t in range(T):
        q = sum(w[i] * ep[t][i] for i in range(n_exp))
        preds.append(1 if q >= 0.5 else 0)
        w = normalize([w[i] * math.exp(-eta * losses[t][i]) for i in range(n_exp)])
    return preds


def ftl_predictions(case, T):
    ep = case['expert_predictions'][:T]
    tl = case['true_labels'][:T]
    n_exp = len(ep[0])
    cum_correct = [0] * n_exp
    preds = []
    for t in range(T):
        leader = max(range(n_exp), key=lambda i: cum_correct[i]) if t > 0 else 0
        preds.append(ep[t][leader])
        for i in range(n_exp):
            cum_correct[i] += int(ep[t][i] == tl[t])
    return preds


def fpw_predictions(case, T):
    ep = case['expert_predictions'][:T]
    tl = case['true_labels'][:T]
    n_exp = len(ep[0])
    correct_mask = [True] * n_exp
    preds = []
    for t in range(T):
        trusted = [i for i in range(n_exp) if correct_mask[i]] or list(range(n_exp))
        preds.append(1 if sum(ep[t][i] for i in trusted) / len(trusted) >= 0.5 else 0)
        correct_mask = [ep[t][i] == tl[t] for i in range(n_exp)]
    return preds


def majority_predictions(case, T):
    ep = case['expert_predictions'][:T]
    n_exp = len(ep[0])
    return [1 if sum(ep[t]) / n_exp >= 0.5 else 0 for t in range(T)]


def random_predictions(T, seed):
    rng = np.random.RandomState(seed)
    return [int(rng.randint(0, 2)) for _ in range(T)]


# ── baseline computation ────────────────────────────────────────────────────

def compute_baselines(cases, N, T):
    results = {}
    for name, pred_fn in [
        ('MW Optimal',              lambda c, idx: mw_predictions(c, T)),
        ('Follow the Leader',       lambda c, idx: ftl_predictions(c, T)),
        ('Follow Previous Winners', lambda c, idx: fpw_predictions(c, T)),
        ('Majority Vote',           lambda c, idx: majority_predictions(c, T)),
        ('Random Guessing',         lambda c, idx: random_predictions(T, idx + 100)),
    ]:
        curves = []
        for idx in range(N):
            c = cases[idx]
            preds = pred_fn(c, idx)
            curves.append(compute_hard_regret(preds, c['true_labels'][:T], c['expert_predictions'][:T]))
        results[name] = np.array(curves)
    return results


BASELINE_STYLE = {
    'MW Optimal':              ('#4C78A8', 3.5, 10),
    'Random Guessing':         ('#AAAAAA', 2.5, 1),
    'Majority Vote':           ('#8B4513', 3.0, 2),
    'Follow the Leader':       ('#17BECF', 3.0, 2),
    'Follow Previous Winners': ('#9467BD', 3.0, 2),
}

LLM_STYLE = {
    # (prompt_short, model_key) -> (label, color, linestyle, linewidth)
    # DS (v2-free names)
    ('weather_notehist', 'ds'):    ('DS weather +note', '#2CA02C', '-', 3.5),
    ('weather_nonote', 'ds'):      ('DS weather -note', '#90EE90', '-', 3.5),
    ('online_notehist', 'ds'):     ('DS online +note',  '#D62728', '-', 3.5),
    ('online_nonote', 'ds'):       ('DS online -note',  '#FF9999', '-', 3.5),
    # Qwen (v2-free names)
    ('weather_notehist', 'qwen3_14b_nothink'):   ('Qwen weather +note', '#2CA02C', ':', 3.5),
    ('weather_nonote', 'qwen3_14b_nothink'):     ('Qwen weather -note', '#90EE90', ':', 3.5),
    ('online_notehist', 'qwen3_14b_nothink'):    ('Qwen online +note',  '#D62728', ':', 3.5),
    ('online_nonote', 'qwen3_14b_nothink'):      ('Qwen online -note',  '#FF9999', ':', 3.5),
    # Legacy v2 names (backward compat with existing results)
    ('weather_v2_notehist', 'ds'):    ('DS weather +note', '#2CA02C', '-', 3.5),
    ('weather_nonote_v2', 'ds'):      ('DS weather -note', '#90EE90', '-', 3.5),
    ('online_v2_notehist', 'ds'):     ('DS online +note',  '#D62728', '-', 3.5),
    ('online_nonote_v2', 'ds'):       ('DS online -note',  '#FF9999', '-', 3.5),
    ('weather_v2_notehist', 'qwen3_14b_nothink'):   ('Qwen weather +note', '#2CA02C', ':', 3.5),
    ('weather_nonote_v2', 'qwen3_14b_nothink'):     ('Qwen weather -note', '#90EE90', ':', 3.5),
    ('online_v2_notehist', 'qwen3_14b_nothink'):    ('Qwen online +note',  '#D62728', ':', 3.5),
    ('online_nonote_v2', 'qwen3_14b_nothink'):      ('Qwen online -note',  '#FF9999', ':', 3.5),
}


def load_llm_curves(exp_dir, prompt, model_key, cases, N, T):
    curves = []
    for idx in range(N):
        rf = Path(exp_dir) / prompt / model_key / f'cases_{idx:03d}' / 'run001' / 'result.json'
        if not rf.exists():
            continue
        r = json.loads(rf.read_text())
        preds = r.get('response', {}).get('predictions', [])
        if len(preds) < T:
            continue
        curves.append(compute_hard_regret(
            preds[:T], cases[idx]['true_labels'][:T], cases[idx]['expert_predictions'][:T]))
    return curves


def plot(exp_dirs, dataset_file, model_keys, save_path=None, T=100, N=30):
    plt.rcParams.update({
        'font.size': 22, 'axes.labelsize': 24,
        'legend.fontsize': 18, 'xtick.labelsize': 18, 'ytick.labelsize': 18,
    })

    data = json.loads(Path(dataset_file).read_text())
    cases = data['cases']
    steps = np.arange(1, T + 1)

    fig, ax = plt.subplots(figsize=(10, 13))

    # Baselines
    baselines = compute_baselines(cases, N, T)
    for bname in ['MW Optimal', 'Random Guessing', 'Majority Vote',
                   'Follow the Leader', 'Follow Previous Winners']:
        arr = baselines[bname]
        color, lw, zorder = BASELINE_STYLE[bname]
        mean = arr.mean(axis=0)
        ax.plot(steps, mean, lw=lw, color=color, linestyle='--', label=bname, zorder=zorder)
        ax.fill_between(steps, mean - arr.std(axis=0), mean + arr.std(axis=0),
                        alpha=0.08, color=color)

    # LLM curves
    if isinstance(exp_dirs, str):
        exp_dirs = [exp_dirs]
    for exp_dir in exp_dirs:
        for prompt_dir in sorted(Path(exp_dir).iterdir()):
            if not prompt_dir.is_dir():
                continue
            prompt = prompt_dir.name
            for mk in model_keys:
                mk_dir = prompt_dir / mk
                if not mk_dir.is_dir():
                    continue
                key = (prompt, mk)
                if key in LLM_STYLE:
                    label, color, ls, lw = LLM_STYLE[key]
                else:
                    label, color, ls, lw = f'{mk} {prompt}', '#888888', '-', 2.0

                curves = load_llm_curves(exp_dir, prompt, mk, cases, N, T)
                if not curves:
                    continue
                arr = np.array(curves)
                mean = arr.mean(axis=0)
                ax.plot(steps, mean, lw=lw, color=color, linestyle=ls,
                        label=f'{label} (n={len(curves)})', zorder=5)
                ax.fill_between(steps, mean - arr.std(axis=0), mean + arr.std(axis=0),
                                alpha=0.06, color=color)

    ax.set_xlabel('Step')
    ax.set_ylabel('Regret')
    ax.legend(loc='upper left')
    ax.grid(alpha=0.25)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'Saved: {save_path}')
    else:
        plt.show()
    plt.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--exp_dir', nargs='+', required=True,
                        help='Result directory(ies)')
    parser.add_argument('--dataset', required=True, help='Dataset JSON file')
    parser.add_argument('--model', nargs='+', default=['ds'],
                        help='Model key(s) to plot')
    parser.add_argument('--save', type=str, help='Save path for plot')
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--cases', type=int, default=30)
    args = parser.parse_args()

    plot(args.exp_dir, args.dataset, args.model,
         save_path=args.save, T=args.steps, N=args.cases)


if __name__ == '__main__':
    main()
