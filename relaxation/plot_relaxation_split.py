"""
Split version of plot_relaxation.py: one figure per sweep, (a)-(d), each with the
Weighted Majority panel on the left and the tabular Q-learning panel on the right.
Same data, colours, titles, axis labels and legend text as relaxation_sweeps.pdf;
only the layout changes (legend below the panels, larger type).

  python3 plot_relaxation_split.py              ->  relaxation_{a,b,c,d}.{pdf,png}
  python3 plot_relaxation_split.py --residual   ->  relaxation_residual_{a,b,c,d}.{pdf,png}
                                                   (residual-write constructions)
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import plot_relaxation as P

HERE = P.HERE
plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9, "legend.fontsize": 8,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "lines.linewidth": 1.6, "lines.markersize": 4,
})

TITLES = {"a": "(a) attention hardness", "b": "(b) residual-stream noise",
          "c": "(c) random embeddings", "d": "(d) overlapping buffers"}
NAMES = {"mwu": "Weighted Majority", "q": r"Tabular $Q$-learning"}


SUFFIX = ""


def load(c, T):
    df = pd.read_csv(os.path.join(HERE, "results", f"{c}_relaxation{SUFFIX}.csv"))
    return df[df["T"] == T]


def panel(part, ax, c, df):
    col, ylabel = P.ERR[c]
    if part == "a":
        d = df[df.sweep == "beta_route"].sort_values("beta")
        P.line(ax, d.beta, d, col, P.BLUE, "routing / mask heads")
        if c == "q":
            d = df[df.sweep == "beta_max"].sort_values("beta_max")
            P.line(ax, d.beta_max, d, col, P.ORANGE, "max head", marker="s")
        xlabel = r"inverse temperature $\beta$"
    elif part == "b":
        d = df[(df.sweep == "beta_x_noise") & (df.sigma > 0)]
        for i, b in enumerate(P.BETAS_PLOT):
            db = d[d.beta == b].sort_values("sigma")
            e = np.log10(b)
            lab = rf"$\beta=10^{{{int(e)}}}$" if e == int(e) else rf"$\beta={b:g}$"
            P.line(ax, db.sigma, db, col, P.BETA_RAMP[i], lab)
        xlabel = r"noise std $\sigma$ per layer"
    elif part == "c":
        d = df[(df.sweep == "beta_x_dim") & (df.beta == 50)].sort_values("dim")
        P.line(ax, d.dim, d, col, P.ORANGE, r"tied read-out $\Phi^\top$, $\sigma=0$", marker="s")
        d = df[(df.sweep == "dual_x_dim") & (df.sigma > 0)]
        for i, sg in enumerate(sorted(d.sigma.unique())):
            ds = d[d.sigma == sg].sort_values("dim")
            P.line(ax, ds.dim, ds, col, P.SIG_RAMP[i],
                   rf"dual read-out $\Phi^+$, $\sigma=10^{{{int(np.log10(sg))}}}$")
        xlabel = r"embedding dim $d$"
    else:
        # tied read-out (compounds through the recurrence, as in (c)) vs the same
        # overlapping buffers read out through the dual basis
        d = df[(df.sweep == "overlap") & (df.rho > 0)].sort_values("rho")
        P.line(ax, d.rho, d, col, P.ORANGE, r"tied read-out $\Phi^\top$, $\sigma=0$", marker="s")
        d = df[(df.sweep == "overlap_dual") & (df.rho > 0) & (df.sigma > 0)]
        for i, sg in enumerate(sorted(d.sigma.unique())):
            ds = d[d.sigma == sg].sort_values("rho")
            P.line(ax, ds.rho, ds, col, P.SIG_RAMP[i],
                   rf"dual read-out $\Phi^+$, $\sigma=10^{{{int(np.log10(sg))}}}$")
        xlabel = r"buffer overlap $\rho$"
    P.decorate(ax, xlabel, f"{NAMES[c]}\n{ylabel}")


def main(T=500):
    data = {c: load(c, T) for c in ("mwu", "q")}
    for part in "abcd":
        fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.6), constrained_layout=True)
        for ax, c in zip(axes, ("mwu", "q")):
            panel(part, ax, c, data[c])
        # one legend for the figure, below the panels, from whichever panel has
        # the most labelled lines (the Q panel for (a))
        handles, labels = max((ax.get_legend_handles_labels() for ax in axes),
                              key=lambda hl: len(hl[1]))
        if labels:
            fig.legend(handles, labels, loc="outside lower center", frameon=False,
                       ncol=2 if part in "cd" else min(len(labels), 5),
                       handlelength=1.6, columnspacing=1.2)
        fig.suptitle(TITLES[part], x=0.02, ha="left", color=P.INK, fontsize=10)
        for ext, kw in (("pdf", {}), ("png", {"dpi": 220})):
            fig.savefig(os.path.join(HERE, "figures", f"relaxation{SUFFIX}_{part}.{ext}"), **kw)
        plt.close(fig)
    print(f"saved relaxation{SUFFIX}_{{a,b,c,d}}.{{pdf,png}}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--residual", action="store_true")
    a = ap.parse_args()
    if a.residual:
        SUFFIX = "_residual"
    main()
