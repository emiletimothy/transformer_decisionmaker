"""
Plot the relaxation sweeps (reads {mwu,q}_relaxation.csv written by run_sweeps.py).

Figure: 2 rows (Weighted Majority, tabular Q-learning) x 4 columns
  (a) attention inverse temperature beta of the hard heads
  (b) residual-stream noise sigma, one line per beta
  (c) non-orthogonal token embeddings (random unit vectors in R^d): tied read-out
      (Phi^T, compounds through the recurrence) vs dual-basis read-out with noise sigma
  (d) overlapping buffer subspaces (max cross-block |cos| = rho)
y = deviation from the exact algorithm, max over t <= T, mean +/- SEM over 20 instances.
Points where any instance diverged (inf/NaN) are drawn as x at the top of the axis.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))

INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE = "#2a78d6", "#eb6834"                   # categorical slots 1, 2
# ordinal blue ramps (validated with --ordinal)
BETA_RAMP = ["#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"]
BETAS_PLOT = [10, 50, 100, 1e3, 1e6]
SIG_RAMP = ["#86b6ef", "#256abf", "#0d366b"]
FLOOR, CEIL = 1e-9, 1e3
THRESH = 1e-2

plt.rcParams.update({
    "font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5, "legend.fontsize": 6.2,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.5, "lines.linewidth": 1.4, "lines.markersize": 3.2,
    "pdf.fonttype": 42,
})

ERR = {"mwu": ("max_pred_err", r"max$_t\,|\hat p_t - p_t^{\mathrm{WM}}|$"),
       "q": ("max_err", r"max$_t\,\|\hat Q_t - Q_t\|_\infty$")}


def line(ax, x, df, col, color, label, marker="o"):
    x = np.asarray(x, float)
    m = df[col + "_mean"].to_numpy(float)
    s = df[col + "_sem"].fillna(0).to_numpy(float)
    div = (df["diverged_mean"].to_numpy(float) > 0) | ~np.isfinite(m) | (m > CEIL)
    ok = ~div
    mm = np.clip(m, FLOOR, CEIL)
    ax.plot(x[ok], mm[ok], color=color, marker=marker, label=label, zorder=3)
    ax.fill_between(x[ok], np.clip(m[ok] - s[ok], FLOOR, None), np.clip(m[ok] + s[ok], None, CEIL),
                    color=color, alpha=0.15, linewidth=0, zorder=2)
    if div.any():
        ax.plot(x[div], np.full(div.sum(), CEIL * 0.6), ls="none", marker="x", color=color,
                markersize=4.5, mew=1.2, zorder=4, clip_on=False)
        # connect last finite point to the first diverged marker
        if ok.any():
            i = np.where(ok)[0][-1]
            j = np.where(div)[0]
            j = j[j > i]
            if len(j):
                ax.plot([x[i], x[j[0]]], [mm[i], CEIL * 0.6], color=color, ls=":", lw=1, zorder=2)


def decorate(ax, xlabel, ylabel=None):
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(FLOOR, CEIL)
    ax.set_yticks([1e-9, 1e-6, 1e-3, 1, 1e3])
    ax.axhline(THRESH, color=INK2, lw=0.7, ls=(0, (3, 3)), zorder=1)
    ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)


def main(T=500):
    fig, axes = plt.subplots(2, 4, figsize=(7.2, 3.9), constrained_layout=True)
    for r, c in enumerate(["mwu", "q"]):
        df = pd.read_csv(os.path.join(HERE, "results", f"{c}_relaxation.csv"))
        df = df[df["T"] == T]
        col, ylabel = ERR[c]
        name = "Weighted Majority" if c == "mwu" else r"Tabular $Q$-learning"

        # (a) beta
        ax = axes[r, 0]
        d = df[df.sweep == "beta_route"].sort_values("beta")
        line(ax, d.beta, d, col, BLUE, "routing / mask heads")
        if c == "q":
            d = df[df.sweep == "beta_max"].sort_values("beta_max")
            line(ax, d.beta_max, d, col, ORANGE, "max head", marker="s")
            ax.legend(frameon=False, loc="upper right")
        decorate(ax, r"inverse temperature $\beta$", f"{name}\n{ylabel}")

        # (b) noise, one line per beta
        ax = axes[r, 1]
        d = df[(df.sweep == "beta_x_noise") & (df.sigma > 0)]
        for i, b in enumerate(BETAS_PLOT):
            db = d[d.beta == b].sort_values("sigma")
            e = np.log10(b)
            lab = rf"$\beta=10^{{{int(e)}}}$" if e == int(e) else rf"$\beta={b:g}$"
            line(ax, db.sigma, db, col, BETA_RAMP[i], lab)
        decorate(ax, r"noise std $\sigma$ per layer")
        if r == 1:
            ax.legend(frameon=False, loc="lower right", ncol=2, handlelength=1.2,
                      columnspacing=0.8)

        # (c) embeddings: tied vs dual read-out
        ax = axes[r, 2]
        d = df[(df.sweep == "beta_x_dim") & (df.beta == 50)].sort_values("dim")
        line(ax, d.dim, d, col, ORANGE, r"tied read-out $\Phi^\top$, $\sigma=0$", marker="s")
        d = df[(df.sweep == "dual_x_dim") & (df.sigma > 0)]
        for i, sg in enumerate(sorted(d.sigma.unique())):
            ds = d[d.sigma == sg].sort_values("dim")
            line(ax, ds.dim, ds, col, SIG_RAMP[i],
                 rf"dual read-out $\Phi^+$, $\sigma=10^{{{int(np.log10(sg))}}}$")
        decorate(ax, r"embedding dim $d$")
        if r == 1:
            ax.legend(frameon=False, loc="lower left", handlelength=1.2)

        # (d) overlap
        ax = axes[r, 3]
        d = df[(df.sweep == "overlap") & (df.rho > 0)].sort_values("rho")
        line(ax, d.rho, d, col, ORANGE, None, marker="s")
        decorate(ax, r"buffer overlap $\rho$")

    for j, t in enumerate(["(a) attention hardness", "(b) residual-stream noise",
                           "(c) random embeddings", "(d) overlapping buffers"]):
        axes[0, j].set_title(t, loc="left", color=INK)
    fig.savefig(os.path.join(HERE, "figures", "relaxation_sweeps.pdf"))
    fig.savefig(os.path.join(HERE, "figures", "relaxation_sweeps.png"), dpi=220)
    print("saved relaxation_sweeps.{pdf,png}")


if __name__ == "__main__":
    main()
