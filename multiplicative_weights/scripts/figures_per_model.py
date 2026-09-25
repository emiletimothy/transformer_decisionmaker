#!/usr/bin/env python3
"""
figures_per_model.py — the old MWU figures, recreated for the recurrent models.

The submitted paper's MWU figures were made for the full-history ContinuousCoTTransformer.
This script redraws each of them for the recurrent latent-token models, keeping the old
plot designs where the architecture allows. Output goes to --out_dir
(default ../figures/continuous_residual):

  evaluation/long_sequences/   long_seq_regret_trajectories.png, long_seq_summary.png
  evaluation/robustness/       robustness_summary.png, robustness_trajectories.png
  evaluation/scenarios/        regret_curves_by_scenario.png, weight_heatmaps.png (weights
                               decoded from the latent), ood_bar_chart.png
  attention/                   fig1..3, fig6..8 (recurrent analogues), heatmaps/heatmap_L*_mwstep*
  overview/                    learned_mw_training_results.png, learned_mw_regret_trajectories.png,
                               learned_mw_attention_patterns.png

Models: continuous = residual-latent v5 (seed 43; seed 42 reported alongside in the
long-sequence summary), discrete = the final discrete model (seed 42), raw history = the original
full-history model (learned_mw_transformer.pt), MW = weighted majority with
eta = sqrt(ln n / T), as in the old figures. Regret is NOT clipped at 0 (the old
scripts clipped per step); negative regret means beating the best expert.

  python3 figures_per_model.py [--out_dir ../figures/continuous_residual] [--n_trials 50]
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import paths  # noqa: E402
from train import (generate_single_sequence, encode_rounds, SEP_POS, UPD_POS,  # noqa: E402
                          ROUND_LEN)
from eval_matched_memory import (load_recurrent, load_full_history, full_history_decisions,  # noqa: E402
                               recurrent_decisions, mw_decisions, bayes_decisions, regret)

CKPT = {
    "continuous": str(paths.checkpoint("continuous_residual", 43)),
    "continuous_s42": str(paths.checkpoint("continuous_residual", 42)),
    "discrete": str(paths.checkpoint("discrete", 42)),
}
FULL_HIST = str(paths.FULL_HISTORY)
N_EXP = 4

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#9a9994", "#e4e3df"
STYLE = {"continuous": dict(color=BLUE, ls="-", label="continuous latent"),
         "discrete": dict(color=ORANGE, ls="-", label="discrete token"),
         "full_hist": dict(color=AQUA, ls="-", label="raw history"),
         "mw": dict(color=INK2, ls="--", label="MW"),
         "bayes": dict(color=INK, ls="-.", label="Bayes")}
plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9, "legend.fontsize": 8,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.edgecolor": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "axes.axisbelow": True, "grid.color": GRID, "grid.linewidth": 0.5,
    "lines.linewidth": 1.6, "pdf.fonttype": 42,
})
TOKENS = (["M"] + [x for e in range(1, 5) for x in (f"E{e}", f"p{e}")] + ["SEP", "y"]
          + [x for e in range(1, 5) for x in (f"E{e}", f"ℓ{e}")] + ["UPD"])
TYPE_OF = (["latent"] + ["expert id", "prediction"] * 4 + ["SEP", "label"]
           + ["expert id", "loss"] * 4 + ["UPD"])
TYPE_COLORS = {"latent": BLUE, "expert id": MUTED, "prediction": "#eda100", "SEP": INK,
               "label": "#e87ba4", "loss": "#4a3aa7", "UPD": AQUA}


def save(fig, out_dir, rel):
    path = os.path.join(out_dir, rel)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("saved", rel, flush=True)


# ---------------------------------------------------------------------------
# Sequence generators (same distributions as the old scripts)
# ---------------------------------------------------------------------------

def seq_from_q(T, q, rng):
    return generate_single_sequence(N_EXP, T, rng, q=q)


def seq_correlated(T, corr, rng):
    q = rng.uniform(0.4, 0.8, N_EXP)
    y = rng.integers(0, 2, T)
    preds = np.zeros((T, N_EXP), dtype=int)
    for t in range(T):
        if rng.random() < corr:
            ok = rng.random() < q.mean()
            preds[t] = y[t] if ok else 1 - y[t]
        else:
            ok = rng.random(N_EXP) < q
            preds[t] = np.where(ok, y[t], 1 - y[t])
    return pack(preds, y, q)


def seq_switching(T, every, rng):
    best = rng.integers(0, N_EXP, (T + every - 1) // every)
    y = rng.integers(0, 2, T)
    preds = np.zeros((T, N_EXP), dtype=int)
    for t in range(T):
        q = np.full(N_EXP, 0.4)
        q[best[t // every]] = 0.85
        ok = rng.random(N_EXP) < q
        preds[t] = np.where(ok, y[t], 1 - y[t])
    return pack(preds, y, None)


def seq_drift(T, rng):
    q = rng.uniform(0.4, 0.8, N_EXP)
    y = rng.integers(0, 2, T)
    preds = np.zeros((T, N_EXP), dtype=int)
    for t in range(T):
        ok = rng.random(N_EXP) < q
        preds[t] = np.where(ok, y[t], 1 - y[t])
        q = np.clip(q + rng.normal(0, 0.02, N_EXP), 0.2, 0.95)
    return pack(preds, y, None)


def seq_adversarial(T, rng):
    q = rng.uniform(0.3, 0.9, N_EXP)
    y = rng.integers(0, 2, T)
    preds = np.zeros((T, N_EXP), dtype=int)
    for t in range(T):
        qt = np.array([q[e] if (t + e) % 2 == 0 else 1 - q[e] for e in range(N_EXP)])
        ok = rng.random(N_EXP) < qt
        preds[t] = np.where(ok, y[t], 1 - y[t])
    return pack(preds, y, q)


def pack(preds, y, q):
    losses = (preds != y[:, None]).astype(float)
    return {"expert_predictions": preds.tolist(), "losses": losses.tolist(),
            "true_labels": y.tolist(), "n_steps": len(y),
            "qualities": None if q is None else list(q)}


def q_n_real(n_real, rng):
    q = np.full(N_EXP, 0.5)
    q[rng.choice(N_EXP, n_real, replace=False)] = rng.uniform(0.6, 0.85, n_real)
    return q


def q_mode(mode, rng):
    if mode == "dominant":
        q = rng.uniform(0.35, 0.5, N_EXP)
        q[rng.integers(N_EXP)] = 0.9
    elif mode == "close":
        q = rng.uniform(0.55, 0.65, N_EXP)
    else:
        q = rng.uniform(0.3, 0.9, N_EXP)
    return q


def q_scenario(name, rng):     # evaluate_model.py's structural scenarios
    if name == "one_dominant":
        q = rng.uniform(0.3, 0.5, N_EXP)
        q[0] = rng.uniform(0.90, 0.98)
    elif name == "all_mediocre":
        q = rng.uniform(0.45, 0.55, N_EXP)
    elif name == "two_good_two_bad":
        q = np.r_[rng.uniform(0.80, 0.95, 2), rng.uniform(0.20, 0.35, 2)]
    else:
        q = rng.uniform(0.3, 0.9, N_EXP)
    return q


# ---------------------------------------------------------------------------
# Running models
# ---------------------------------------------------------------------------

class Runner:
    def __init__(self, device):
        self.device = device
        self.models = {k: load_recurrent(p, device)[0] for k, p in CKPT.items()}
        self.fh = load_full_history(FULL_HIST, device)

    def decisions(self, name, seqs):
        if name in self.models:
            return recurrent_decisions(self.models[name], seqs, self.device)
        if name == "full_hist":
            return full_history_decisions(self.fh[0], self.fh[1], seqs, self.device)
        if name == "mw":
            return np.stack([mw_decisions(s) for s in seqs])
        if name == "bayes":
            P = [np.array(s["expert_predictions"]) for s in seqs]
            L = [np.array(s["losses"]) for s in seqs]
            return np.stack([bayes_decisions(p, l) for p, l in zip(P, L)])
        raise KeyError(name)

    def evaluate(self, names, seqs):
        """name -> dict(traj [n, T] unclipped regret, final [n], acc [n])."""
        y = np.array([s["true_labels"] for s in seqs])
        out = {}
        for n in names:
            d = self.decisions(n, seqs)
            traj = np.stack([regret(s, d[i]) for i, s in enumerate(seqs)])
            out[n] = {"traj": traj, "final": traj[:, -1], "acc": (d == y).mean(1), "dec": d}
        return out

    @torch.no_grad()
    def latents_attn(self, name, seqs, want_attn=True):
        """Latent after each round [n, T, d], logits [n, T], attention [n, T, L, H, 20, 20]."""
        m = self.models[name]
        ids = torch.from_numpy(np.stack([encode_rounds(s, m.tokenizer, m.UPD_TOKEN)
                                         for s in seqs])).to(self.device)
        B, T, _ = ids.shape
        M = m.initial_latent(B)
        Ms, Zs, As = [], [], []
        for t in range(T):
            z, M, _, attn = m.step(M, ids[:, t], return_attention=want_attn)
            Ms.append(M.cpu().numpy())
            Zs.append(z.cpu().numpy())
            if want_attn:
                As.append(torch.stack(attn, 1).cpu().numpy())     # [B, L, H, 20, 20]
        A = np.stack(As, 1) if want_attn else None
        return np.stack(Ms, 1), np.stack(Zs, 1), A


def ridge(X, Y, lam=1e-2):
    mx, my = X.mean(0), Y.mean(0)
    A = X - mx
    W = np.linalg.solve(A.T @ A + lam * len(A) * np.eye(A.shape[1]), A.T @ (Y - my))
    return lambda Xq: (Xq - mx) @ W + my


def mw_weights(L, eta):
    """MW weights before each round from cumulative losses AFTER each round [.., T, n]."""
    prev = np.concatenate([np.zeros_like(L[..., :1, :]), L[..., :-1, :]], -2)
    w = np.exp(-eta * (prev - prev.min(-1, keepdims=True)))
    return w / w.sum(-1, keepdims=True)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def fig_long_seq(run, out, n_trials, rng, lengths=(50, 100, 500, 1000)):
    names = ["continuous", "continuous_s42", "discrete", "full_hist", "mw", "bayes"]
    res = {}
    for T in lengths:
        t0 = time.time()
        seqs = [generate_single_sequence(N_EXP, T, rng) for _ in range(n_trials)]
        res[T] = run.evaluate(names, seqs)
        print(f"  long_seq T={T}: " + ", ".join(
            f"{n}={res[T][n]['final'].mean():.2f}" for n in names) + f" [{time.time() - t0:.0f}s]",
            flush=True)

    fig, axes = plt.subplots(1, len(lengths), figsize=(3.4 * len(lengths), 2.9),
                             constrained_layout=True)
    for ax, T in zip(axes, lengths):
        t = np.arange(1, T + 1)
        for n in ["discrete", "full_hist", "continuous", "mw", "bayes"]:
            m, s = res[T][n]["traj"].mean(0), res[T][n]["traj"].std(0)
            ax.plot(t, m, color=STYLE[n]["color"], ls=STYLE[n]["ls"], label=STYLE[n]["label"])
            if n in ("continuous", "mw"):
                ax.fill_between(t, m - s, m + s, color=STYLE[n]["color"], alpha=0.12, lw=0)
        ax.axhline(0, color=INK2, lw=0.6)
        ax.set_title(f"Regret growth (T={T})", loc="left")
        ax.set_xlabel("time step")
    axes[0].set_ylabel("cumulative regret")
    axes[0].legend(frameon=False, loc="upper left")
    save(fig, out, "evaluation/long_sequences/long_seq_regret_trajectories.png")

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.1), constrained_layout=True)
    x = np.arange(len(lengths))
    bars = ["continuous", "continuous_s42", "discrete", "full_hist", "mw"]
    labels = {"continuous_s42": "continuous (seed 42)"}
    w = 0.16
    for j, n in enumerate(bars):
        c = STYLE.get(n, STYLE["continuous"])["color"]
        alpha = 0.55 if n == "continuous_s42" else 1.0
        m = [res[T][n]["final"].mean() for T in lengths]
        se = [res[T][n]["final"].std() / np.sqrt(n_trials) for T in lengths]
        axes[0].bar(x + (j - 2) * w, m, w, yerr=se, color=c, alpha=alpha, capsize=2,
                    label=labels.get(n, STYLE.get(n, {}).get("label", n)))
    axes[0].axhline(0, color=INK2, lw=0.6)
    axes[0].set_title("Final regret (mean ± SEM)", loc="left")
    axes[0].legend(frameon=False, fontsize=7)
    for j, n in enumerate(["continuous", "discrete", "full_hist"]):
        diff = [res[T][n]["final"].mean() - res[T]["mw"]["final"].mean() for T in lengths]
        axes[1].bar(x + (j - 1) * 0.25, diff, 0.25, color=STYLE[n]["color"])
    axes[1].axhline(0, color=INK2, lw=0.8, ls="--")
    axes[1].set_title("Regret minus MW regret", loc="left")
    for j, n in enumerate(["continuous", "discrete", "full_hist", "mw"]):
        m = [res[T][n]["acc"].mean() for T in lengths]
        se = [res[T][n]["acc"].std() / np.sqrt(n_trials) for T in lengths]
        axes[2].bar(x + (j - 1.5) * 0.2, m, 0.2, yerr=se, color=STYLE[n]["color"], capsize=2)
    axes[2].axhline(0.5, color=INK2, lw=0.8, ls="--")
    axes[2].set_ylim(0.4, 0.9)
    axes[2].set_title("Prediction accuracy", loc="left")
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels([str(T) for T in lengths])
        ax.set_xlabel("sequence length T")
    save(fig, out, "evaluation/long_sequences/long_seq_summary.png")
    return res


def fig_robustness(run, out, n_trials, rng, T=100):
    scen = []
    for k in (1, 2, 3, 4):
        scen.append((f"{k} real expert{'s' if k > 1 else ''}", lambda k=k: seq_from_q(T, q_n_real(k, rng), rng)))
    for mode in ("dominant", "close", "spread"):
        scen.append((f"quality: {mode}", lambda m=mode: seq_from_q(T, q_mode(m, rng), rng)))
    for every in (10, 25, 50):
        scen.append((f"switch every {every}", lambda e=every: seq_switching(T, e, rng)))
    scen.append(("gradual drift", lambda: seq_drift(T, rng)))
    for c in (0.3, 0.7, 0.9):
        scen.append((f"correlation={c}", lambda c=c: seq_correlated(T, c, rng)))
    names = ["continuous", "discrete", "full_hist", "mw"]
    res = []
    for label, gen in scen:
        seqs = [gen() for _ in range(n_trials)]
        res.append((label, run.evaluate(names, seqs)))
    print("  robustness: " + "; ".join(
        f"{l}: cont={r['continuous']['final'].mean():.1f} mw={r['mw']['final'].mean():.1f}"
        for l, r in res), flush=True)

    fig, axes = plt.subplots(2, 1, figsize=(10, 6.2), constrained_layout=True, sharex=True)
    x = np.arange(len(res))
    for j, n in enumerate(names):
        m = [r[n]["final"].mean() for _, r in res]
        se = [r[n]["final"].std() / np.sqrt(n_trials) for _, r in res]
        axes[0].bar(x + (j - 1.5) * 0.2, m, 0.2, yerr=se, color=STYLE[n]["color"], capsize=1.5,
                    label=STYLE[n]["label"])
        a = [r[n]["acc"].mean() for _, r in res]
        ase = [r[n]["acc"].std() / np.sqrt(n_trials) for _, r in res]
        axes[1].bar(x + (j - 1.5) * 0.2, a, 0.2, yerr=ase, color=STYLE[n]["color"], capsize=1.5)
    axes[0].axhline(0, color=INK2, lw=0.6)
    axes[0].set_ylabel(f"final regret (T={T})")
    axes[0].set_title("Final regret by scenario (mean ± SEM)", loc="left")
    axes[0].legend(frameon=False, ncol=4, loc="upper left")
    axes[1].axhline(0.5, color=INK2, lw=0.8, ls="--")
    axes[1].set_ylim(0.4, 1.0)
    axes[1].set_ylabel("accuracy")
    axes[1].set_title("Prediction accuracy by scenario", loc="left")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([l for l, _ in res], rotation=35, ha="right")
    save(fig, out, "evaluation/robustness/robustness_summary.png")

    fig, axes = plt.subplots(2, 3, figsize=(11, 6), constrained_layout=True)
    for ax, (label, r) in zip(axes.flat, res[:6]):
        t = np.arange(1, T + 1)
        for n in names:
            ax.plot(t, r[n]["traj"].mean(0), color=STYLE[n]["color"], ls=STYLE[n]["ls"],
                    label=STYLE[n]["label"])
        ax.axhline(0, color=INK2, lw=0.6)
        ax.set_title(label, loc="left")
        ax.set_xlabel("time step")
    axes[0, 0].set_ylabel("cumulative regret")
    axes[1, 0].set_ylabel("cumulative regret")
    axes[0, 0].legend(frameon=False)
    save(fig, out, "evaluation/robustness/robustness_trajectories.png")
    return res


def fit_weight_probe(run, rng, T=200, n=300):
    seqs = [generate_single_sequence(N_EXP, T, rng) for _ in range(n)]
    M, _, _ = run.latents_attn("continuous", seqs, want_attn=False)
    L = np.cumsum(np.array([s["losses"] for s in seqs]), 1)
    S = L - L.mean(-1, keepdims=True)
    return ridge(M[:, 4:].reshape(-1, M.shape[-1]), S[:, 4:].reshape(-1, N_EXP))


def decoded_weights(run, probe, seqs, eta):
    """True MW weights and weights decoded from the latent, before each round [n, T, 4]."""
    M, _, _ = run.latents_attn("continuous", seqs, want_attn=False)
    L = np.cumsum(np.array([s["losses"] for s in seqs]), 1)
    Lhat = probe(M.reshape(-1, M.shape[-1])).reshape(L.shape)
    return mw_weights(L, eta), mw_weights(Lhat, eta)


def fig_eval(run, out, n_trials, rng, probe, T=100):
    scen = ["in_distribution", "one_dominant", "all_mediocre", "two_good_two_bad", "adversarial"]
    names = ["continuous", "discrete", "mw"]
    data = {}
    for s in scen:
        seqs = [(seq_adversarial(T, rng) if s == "adversarial" else seq_from_q(T, q_scenario(s, rng), rng))
                for _ in range(n_trials)]
        data[s] = (seqs, run.evaluate(names, seqs))

    fig, axes = plt.subplots(2, 3, figsize=(11, 6), constrained_layout=True)
    for ax, s in zip(axes.flat, scen):
        r = data[s][1]
        t = np.arange(1, T + 1)
        for n in names:
            m, sd = r[n]["traj"].mean(0), r[n]["traj"].std(0)
            ax.plot(t, m, color=STYLE[n]["color"], ls=STYLE[n]["ls"], label=STYLE[n]["label"])
            ax.fill_between(t, m - sd, m + sd, color=STYLE[n]["color"], alpha=0.1, lw=0)
        ax.axhline(0, color=INK2, lw=0.6)
        ax.set_title(s.replace("_", " ").title(), loc="left")
        ax.set_xlabel("time step")
        ax.set_ylabel("cumulative regret")
    axes[0, 0].legend(frameon=False)
    axes.flat[-1].set_visible(False)
    save(fig, out, "evaluation/scenarios/regret_curves_by_scenario.png")

    eta = np.sqrt(np.log(N_EXP) / T)
    fig, axes = plt.subplots(len(scen), 2, figsize=(9, 1.6 * len(scen)), constrained_layout=True)
    for i, s in enumerate(scen):
        gt, pr = decoded_weights(run, probe, data[s][0][:1], eta)
        vmax = max(gt.max(), pr.max())
        for j, (w, tag) in enumerate(((gt[0], "MW weights"), (pr[0], "decoded from latent"))):
            im = axes[i, j].imshow(w.T, aspect="auto", cmap="Blues", vmin=0, vmax=vmax,
                                   interpolation="nearest")
            axes[i, j].set_title(f"{s.replace('_', ' ').title()}: {tag}", loc="left", fontsize=9)
            axes[i, j].set_yticks(range(N_EXP))
            axes[i, j].set_yticklabels([f"E{e + 1}" for e in range(N_EXP)])
            axes[i, j].grid(False)
        fig.colorbar(im, ax=axes[i, 1], fraction=0.04, pad=0.02)
    axes[-1, 0].set_xlabel("round")
    axes[-1, 1].set_xlabel("round")
    save(fig, out, "evaluation/scenarios/weight_heatmaps.png")

    lengths = (10, 20, 50, 100, 200, 500, 1000)
    acc = {n: [] for n in names}
    mse = []
    for Tl in lengths:
        seqs = [generate_single_sequence(N_EXP, Tl, rng) for _ in range(max(20, n_trials // 2))]
        r = run.evaluate(names, seqs)
        for n in names:
            acc[n].append((r[n]["acc"].mean(), r[n]["acc"].std() / np.sqrt(len(seqs))))
        gt, pr = decoded_weights(run, probe, seqs, np.sqrt(np.log(N_EXP) / Tl))
        e = ((gt - pr) ** 2).mean((1, 2))
        mse.append((e.mean(), e.std() / np.sqrt(len(e))))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3), constrained_layout=True)
    x = np.arange(len(lengths))
    for j, n in enumerate(names):
        a1.bar(x + (j - 1) * 0.27, [v[0] for v in acc[n]], 0.27, yerr=[v[1] for v in acc[n]],
               color=STYLE[n]["color"], capsize=1.5, label=STYLE[n]["label"])
    a1.axhline(0.5, color=INK2, lw=0.8, ls="--")
    a1.set_ylim(0.4, 0.9)
    a1.set_title("Decision accuracy by sequence length", loc="left")
    a1.legend(frameon=False, ncol=3, loc="upper left")
    a2.bar(x, [v[0] for v in mse], 0.55, yerr=[v[1] for v in mse], color=BLUE, capsize=2)
    a2.set_title("MW-weight MSE, decoded from the latent", loc="left")
    a2.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    for ax in (a1, a2):
        ax.set_xticks(x)
        ax.set_xticklabels([str(v) for v in lengths])
        ax.set_xlabel("sequence length T (trained up to 165)")
    save(fig, out, "evaluation/scenarios/ood_bar_chart.png")
    print("  ood: acc cont " + ", ".join(f"{T}:{a[0]:.3f}" for T, a in zip(lengths, acc["continuous"]))
          + " | weight MSE " + ", ".join(f"{T}:{m[0]:.1e}" for T, m in zip(lengths, mse)), flush=True)


def label_ticks(ax, fontsize=6.5):
    ax.set_xticks(range(ROUND_LEN))
    ax.set_yticks(range(ROUND_LEN))
    ax.set_xticklabels(TOKENS, rotation=90, fontsize=fontsize)
    ax.set_yticklabels(TOKENS, fontsize=fontsize)
    for tk, ty in zip(ax.get_xticklabels(), TYPE_OF):
        tk.set_color(TYPE_COLORS[ty])
    for tk, ty in zip(ax.get_yticklabels(), TYPE_OF):
        tk.set_color(TYPE_COLORS[ty])
    ax.grid(False)


def fig_attention(run, out, rng, T=100, n=100):
    seqs = [generate_single_sequence(N_EXP, T, rng) for _ in range(n)]
    M, Z, A = run.latents_attn("continuous", seqs)          # A [n, T, L, H, 20, 20]
    nL, nH = A.shape[2], A.shape[3]
    L = np.cumsum(np.array([s["losses"] for s in seqs]), 1)
    prevL = np.concatenate([np.zeros_like(L[:, :1]), L[:, :-1]], 1)

    # fig1: per-head heatmaps at one round of one sequence
    t_show = 49
    fig, axes = plt.subplots(nL, nH, figsize=(3.1 * nH, 3.1 * nL), constrained_layout=True)
    for l in range(nL):
        for h in range(nH):
            ax = axes[l, h]
            ax.imshow(A[0, t_show, l, h], cmap="hot", vmin=0, vmax=1, interpolation="nearest")
            ax.set_title(f"Layer {l}, Head {h}", fontsize=9)
            label_ticks(ax, 5.5)
    fig.suptitle(f"Attention heatmaps, continuous latent (round {t_show + 1})", x=0.01, ha="left")
    save(fig, out, "attention/fig1_per_head_heatmaps.png")

    # heatmaps/: per layer, all heads, several rounds (same file names as before)
    for t in (2, 5, 10, 25, 40, 49):
        for l in range(nL):
            fig, axes = plt.subplots(1, nH, figsize=(3.2 * nH, 3.4), constrained_layout=True)
            for h in range(nH):
                axes[h].imshow(A[0, t, l, h], cmap="hot", vmin=0, vmax=1, interpolation="nearest")
                axes[h].set_title(f"Head {h}", fontsize=9)
                label_ticks(axes[h], 5.5)
            axes[0].set_ylabel("query")
            fig.suptitle(f"Layer {l}, round {t} (20 tokens; M = carried latent)", x=0.01, ha="left")
            save(fig, out, f"attention/heatmaps/heatmap_L{l}_mwstep{t}.png")

    # fig2: attention mass by token type from the decision (SEP) and update (UPD) queries
    types = ["latent", "expert id", "prediction", "SEP", "label", "loss", "UPD"]
    fig, axes = plt.subplots(2, nL, figsize=(5 * nL, 5.4), constrained_layout=True)
    for qi, (qpos, qname) in enumerate(((SEP_POS, "decision (SEP)"), (UPD_POS, "update (UPD)"))):
        for l in range(nL):
            ax = axes[qi, l]
            row = A[:, :, l, :, qpos, :].mean((0, 1))        # [H, 20]
            x = np.arange(len(types))
            for h in range(nH):
                mass = [row[h, [i for i, ty in enumerate(TYPE_OF) if ty == tn]].sum() for tn in types]
                ax.bar(x + (h - 1.5) * 0.2, mass, 0.2, label=f"head {h}",
                       color=["#86b6ef", "#3987e5", "#256abf", "#0d366b"][h])
            ax.set_xticks(x)
            ax.set_xticklabels(types, rotation=30, ha="right")
            ax.set_ylim(0, 1)
            ax.set_title(f"Layer {l}: attention from the {qname} token", loc="left")
            ax.set_ylabel("attention mass")
    axes[0, 0].legend(frameon=False, ncol=2)
    save(fig, out, "attention/fig2_attention_by_token_type.png")

    # fig3: attention from SEP to the best vs worst expert's tokens over rounds
    best, worst = prevL.argmin(-1), prevL.argmax(-1)     # [n, T]
    fig, axes = plt.subplots(1, nL, figsize=(5 * nL, 3.2), constrained_layout=True)
    t = np.arange(1, T + 1)
    for l in range(nL):
        att = A[:, :, l, :, SEP_POS, :]                   # [n, T, H, 20]
        per_exp = np.stack([att[..., 1 + 2 * e] + att[..., 2 + 2 * e] for e in range(N_EXP)], -1)
        idx = np.arange(n)[:, None]
        tt = np.arange(T)[None, :]
        b = per_exp[idx, tt, :, best]                     # [n, T, H]
        w = per_exp[idx, tt, :, worst]
        ax = axes[l]
        for h in range(nH):
            ax.plot(t, b[..., h].mean(0), color=AQUA, lw=0.6, alpha=0.35)
            ax.plot(t, w[..., h].mean(0), color=ORANGE, lw=0.6, alpha=0.35)
        ax.plot(t, b.mean((0, 2)), color=AQUA, lw=2, label="best expert so far")
        ax.plot(t, w.mean((0, 2)), color=ORANGE, lw=2, label="worst expert so far")
        ax.set_title(f"Layer {l}: attention from SEP to expert tokens", loc="left")
        ax.set_xlabel("round")
        ax.set_ylabel("attention mass")
    axes[0].legend(frameon=False)
    save(fig, out, "attention/fig3_attention_best_vs_worst_expert.png")

    # fig6 (analogue of 'attention over time'): mass on the carried latent vs predictions
    fig, axes = plt.subplots(1, nL, figsize=(5 * nL, 3.2), constrained_layout=True)
    for l in range(nL):
        att = A[:, :, l, :, :, :].mean((0, 2))           # [T, 20, 20]
        for qpos, qn, ls in ((SEP_POS, "SEP", "-"), (UPD_POS, "UPD", "--")):
            axes[l].plot(t, att[:, qpos, 0], color=BLUE, ls=ls, label=f"{qn} → latent M")
            axes[l].plot(t, att[:, qpos, [2, 4, 6, 8]].sum(-1), color="#eda100", ls=ls,
                         label=f"{qn} → predictions")
        axes[l].set_ylim(0, 1)
        axes[l].set_title(f"Layer {l}: attention over rounds", loc="left")
        axes[l].set_xlabel("round")
        axes[l].set_ylabel("attention mass")
    axes[0].legend(frameon=False, ncol=2, fontsize=7)
    save(fig, out, "attention/fig6_attention_over_time.png")

    # fig7: MW weight trajectories, true vs decoded from the latent
    probe = ridge(M[:50, 4:].reshape(-1, M.shape[-1]),
                  (L - L.mean(-1, keepdims=True))[:50, 4:].reshape(-1, N_EXP))
    eta = np.sqrt(np.log(N_EXP) / T)
    k = 60                                                # held-out sequence
    Lhat = probe(M[k]).reshape(T, N_EXP)
    gt, pr = mw_weights(L[k], eta), mw_weights(Lhat, eta)
    fig, ax = plt.subplots(figsize=(6, 3.2), constrained_layout=True)
    cols = [BLUE, ORANGE, AQUA, "#e87ba4"]
    for e in range(N_EXP):
        ax.plot(t, gt[:, e], color=cols[e], lw=1.8, label=f"E{e + 1} MW")
        ax.plot(t, pr[:, e], color=cols[e], lw=1.2, ls="--")
    ax.plot([], [], color=INK2, ls="--", label="decoded from latent")
    ax.set_xlabel("round")
    ax.set_ylabel("expert weight")
    ax.set_title("MW weights vs. weights decoded from the latent (held-out sequence)", loc="left")
    ax.legend(frameon=False, ncol=3, fontsize=7)
    save(fig, out, "attention/fig7_weight_trajectories.png")

    # fig8 (analogue of CoT hidden PCA): PCA of the carried latent
    X = M[:, 4:].reshape(-1, M.shape[-1])
    Xc = X - X.mean(0)
    _, sv, Vt = np.linalg.svd(Xc[::7], full_matrices=False)
    P = (M - X.mean(0)) @ Vt[:2].T                        # [n, T, 2]
    var = sv[:2] ** 2 / (sv ** 2).sum()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.5, 3.8), constrained_layout=True)
    bst = L.argmin(-1)
    for e in range(N_EXP):
        sel = bst[:, 10:] == e
        a1.scatter(P[:, 10:, 0][sel], P[:, 10:, 1][sel], s=2, color=cols[e], alpha=0.35,
                   label=f"best so far: E{e + 1}")
    a1.set_title("Latent PCA, coloured by current best expert", loc="left")
    a1.legend(frameon=False, markerscale=5, fontsize=7)
    for i in range(6):
        sc = a2.scatter(P[i, :, 0], P[i, :, 1], c=t, cmap="Blues", s=5)
        a2.plot(P[i, :, 0], P[i, :, 1], color=MUTED, lw=0.4)
    fig.colorbar(sc, ax=a2, label="round")
    a2.set_title("Latent trajectories of 6 sequences", loc="left")
    for ax in (a1, a2):
        ax.set_xlabel(f"PC1 ({100 * var[0]:.0f}% var.)")
        ax.set_ylabel(f"PC2 ({100 * var[1]:.0f}% var.)")
    save(fig, out, "attention/fig8_latent_pca.png")

    # learned_mw_attention_patterns: head-averaged attention + token-type profile by query
    fig, axes = plt.subplots(nL, 2, figsize=(10, 4.2 * nL), constrained_layout=True)
    for l in range(nL):
        mat = A[:, :, l].mean((0, 1, 2))                 # [20, 20]
        im = axes[l, 0].imshow(mat, cmap="Blues", vmin=0, interpolation="nearest")
        axes[l, 0].set_title(f"Layer {l}: attention (mean over heads, rounds)", loc="left")
        label_ticks(axes[l, 0], 6)
        fig.colorbar(im, ax=axes[l, 0], fraction=0.045)
        for tn in ["latent", "expert id", "prediction", "label", "loss"]:
            cols_ = [i for i, ty in enumerate(TYPE_OF) if ty == tn]
            axes[l, 1].plot(range(ROUND_LEN), mat[:, cols_].sum(1), marker="o", ms=3,
                            color=TYPE_COLORS[tn], label=tn)
        axes[l, 1].set_xticks(range(ROUND_LEN))
        axes[l, 1].set_xticklabels(TOKENS, rotation=90, fontsize=6.5)
        axes[l, 1].set_ylim(0, 1.05)
        axes[l, 1].set_title(f"Layer {l}: attention to token types, by query", loc="left")
        axes[l, 1].set_ylabel("attention mass")
    axes[0, 1].legend(frameon=False, fontsize=7)
    save(fig, out, "overview/learned_mw_attention_patterns.png")


def fig_training(run, out, rng, res_long):
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.4), constrained_layout=True)
    for ax, name, path in ((axes[0, 0], "continuous", CKPT["continuous"]),
                           (axes[0, 1], "discrete", CKPT["discrete"])):
        with open(os.path.join(os.path.dirname(path), "history.json")) as f:
            h = json.load(f)
        stages = sorted({r["stage"] for r in h})
        cmap = plt.get_cmap("Blues" if name == "continuous" else "Oranges")
        for i, s in enumerate(stages):
            rows = [r for r in h if r["stage"] == s]
            ax.plot(range(len(rows)), [r["train_loss"] for r in rows],
                    color=cmap(0.3 + 0.7 * i / max(len(stages) - 1, 1)), lw=1.2,
                    label=f"T={rows[0]['T']}" if i in (0, len(stages) // 2, len(stages) - 1) else None)
        ax.axhline(0.615, color=INK2, ls="--", lw=0.8)
        ax.text(0, 0.617, "memoryless optimum", fontsize=7, color=INK2, va="bottom")
        ax.set_title(f"{STYLE[name]['label']}: training loss by stage", loc="left")
        ax.set_xlabel("epoch within stage")
        ax.set_ylabel("train BCE")
        ax.legend(frameon=False, fontsize=7, title="stage", title_fontsize=7)
    T = 100
    names = ["continuous", "discrete", "full_hist", "mw"]
    r = res_long[T]
    x = np.arange(len(names))
    axes[1, 0].bar(x, [r[n]["acc"].mean() for n in names],
                   yerr=[r[n]["acc"].std() / np.sqrt(len(r[n]["acc"])) for n in names],
                   color=[STYLE[n]["color"] for n in names], capsize=2)
    axes[1, 0].set_ylim(0.5, 0.85)
    axes[1, 0].set_title(f"Final prediction accuracy (T={T})", loc="left")
    axes[1, 1].bar(x, [r[n]["final"].mean() for n in names],
                   yerr=[r[n]["final"].std() / np.sqrt(len(r[n]["final"])) for n in names],
                   color=[STYLE[n]["color"] for n in names], capsize=2)
    axes[1, 1].axhline(0, color=INK2, lw=0.6)
    axes[1, 1].set_title(f"Final regret (T={T})", loc="left")
    for ax in axes[1]:
        ax.set_xticks(x)
        ax.set_xticklabels([STYLE[n]["label"] for n in names], rotation=15)
    save(fig, out, "overview/learned_mw_training_results.png")


def fig_regret_traj(run, out, rng, res_long):
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.4), constrained_layout=True)
    for ax, T in ((axes[0, 0], 20), (axes[0, 1], 200)):
        seqs = [generate_single_sequence(N_EXP, T, rng) for _ in range(3)]
        r = run.evaluate(["continuous", "mw"], seqs)
        t = np.arange(1, T + 1)
        for i in range(3):
            ax.plot(t, r["continuous"]["traj"][i], color=BLUE, alpha=0.8,
                    label="continuous latent" if i == 0 else None)
            ax.plot(t, r["mw"]["traj"][i], color=INK2, ls="--", alpha=0.8,
                    label="MW" if i == 0 else None)
        ax.set_title(f"Regret growth: 3 sequences, T={T}", loc="left")
        ax.set_xlabel("time step")
        ax.set_ylabel("cumulative regret")
        ax.legend(frameon=False)
    T = 1000
    r = res_long[T]
    t = np.arange(1, T + 1)
    ax = axes[1, 0]
    for n in ["continuous", "mw", "discrete"]:
        m, s = r[n]["traj"].mean(0), r[n]["traj"].std(0)
        ax.plot(t, m, color=STYLE[n]["color"], ls=STYLE[n]["ls"], label=STYLE[n]["label"])
        ax.fill_between(t, m - s, m + s, color=STYLE[n]["color"], alpha=0.12, lw=0)
    ax.axhline(0, color=INK2, lw=0.6)
    ax.set_title("Average regret growth (±1σ), T=1000", loc="left")
    ax.set_xlabel("time step")
    ax.set_ylabel("cumulative regret")
    ax.legend(frameon=False)
    ax = axes[1, 1]
    d = r["continuous"]["traj"] - r["mw"]["traj"]
    ax.plot(t, d.mean(0), color=BLUE)
    ax.fill_between(t, d.mean(0) - d.std(0) / np.sqrt(len(d)), d.mean(0) + d.std(0) / np.sqrt(len(d)),
                    color=BLUE, alpha=0.15, lw=0)
    ax.axhline(0, color=INK2, lw=0.8, ls="--")
    ax.set_title("Regret of continuous latent minus MW (±SEM)", loc="left")
    ax.set_xlabel("time step")
    ax.set_ylabel("regret difference")
    save(fig, out, "overview/learned_mw_regret_trajectories.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default=str(paths.FIGURES / "continuous_residual"))
    ap.add_argument("--n_trials", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--only", nargs="*", default=None,
                    help="subset of: long_seq robustness eval attention training regret_traj")
    args = ap.parse_args()
    want = lambda k: args.only is None or k in args.only
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    run = Runner(device)
    t0 = time.time()
    res_long = None
    if want("long_seq") or want("training") or want("regret_traj"):
        res_long = fig_long_seq(run, args.out_dir, args.n_trials, rng)
    if want("robustness"):
        fig_robustness(run, args.out_dir, args.n_trials, rng)
    if want("eval"):
        fig_eval(run, args.out_dir, args.n_trials, rng, fit_weight_probe(run, rng))
    if want("attention"):
        fig_attention(run, args.out_dir, rng)
    if want("training"):
        fig_training(run, args.out_dir, rng, res_long)
    if want("regret_traj"):
        fig_regret_traj(run, args.out_dir, rng, res_long)
    print(f"done in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
