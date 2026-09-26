#!/usr/bin/env python3
"""
Verification figure for the v2 handwired exponential-weights transformer
(handwired_mwu.py: flag-free, causal softmax heads, one ReLU MLP, trained round
layout, residual latent write). Same design as the appendix comparison figures: exact MW weights,
the weights decoded from the construction's latent token, and their difference, per round.

  python3 handwired_figure.py   -> ../figures/handwired/mwu_handwired_4_experts.{png,pdf}
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import paths  # noqa: E402
from handwired_mwu import HandwiredMWUv2  # noqa: E402

COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]   # reference categorical slots 1-4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", type=int, default=500)
    ap.add_argument("--eta", type=float, default=0.1)
    ap.add_argument("--beta", type=float, default=1e3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_dir", default=str(paths.FIGURES / "handwired"))
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # the paper's data model: expert qualities U[0.3, 0.9], uniform binary labels
    rng = np.random.default_rng(args.seed)
    q = rng.uniform(0.3, 0.9, 4)
    y = rng.integers(0, 2, args.T)
    preds = np.where(rng.random((args.T, 4)) < q, y[:, None], 1 - y[:, None])
    out = HandwiredMWUv2(4, args.eta, args.beta).run(preds.tolist(), y.tolist())
    W_tf = np.stack([o[2] for o in out])
    W_ex = np.stack([o[3] for o in out])
    werr = np.abs(W_tf - W_ex).max()
    perr = max(abs(o[0] - o[1]) for o in out)
    print(f"v2 construction (beta={args.beta:g}, eta={args.eta}, T={args.T}): "
          f"max |w_tf - w_MW| = {werr:.2e}, max |p_tf - p_MW| = {perr:.2e}")

    t = np.arange(1, args.T + 1)
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2), constrained_layout=True)
    for i in range(4):
        axes[0].plot(t, W_ex[:, i], color=COLORS[i], lw=1.4, label=f"expert {i + 1} (q={q[i]:.2f})")
        axes[1].plot(t, W_tf[:, i], color=COLORS[i], lw=1.4)
        axes[2].plot(t, W_tf[:, i] - W_ex[:, i], color=COLORS[i], lw=1.2)
    axes[0].set_title("Multiplicative weights (exact)", loc="left", fontsize=10)
    axes[1].set_title("Transformer construction (residual write)", loc="left", fontsize=10)
    axes[2].set_title(f"Difference (max |Δw| = {werr:.1e})", loc="left", fontsize=10)
    for ax in axes[:2]:
        ax.set_ylabel("expert weight")
        ax.set_ylim(-0.02, 1.02)
    axes[2].set_ylabel("weight difference")
    for ax in axes:
        ax.set_xlabel("round")
        ax.grid(True, color="#e4e3df", lw=0.5)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, fontsize=8)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(args.out_dir, f"mwu_handwired_4_experts.{ext}"), dpi=300)
    print("saved", os.path.join(args.out_dir, "mwu_handwired_4_experts.png"))


if __name__ == "__main__":
    main()
