#!/usr/bin/env python3
"""
Main-text figures and the matched-control table for the experiments section.

  fig_mwu_main.pdf      regret vs round (T = 1000) and MW weights decoded from the latent
  fig_mwu_leak.pdf      latent retention rho vs regret at T = 1000, one point per trained model
  fig_mwu_steering.pdf  causal steering of the latent along probe directions
  fig_q_clean.pdf       clean-step margins: follows Q-learning minus follows the simpler rule
  fig_q_main.pdf        reward when the model acts, Q-table probe, agreement with Q-learning

Every input path is in CONFIG. Where a final result is not on disk yet, the
script falls back to the best available one and prints DRAFT; re-run it once
the jobs finish.

  python3 paper/make_main_figures.py
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MF = os.path.join(ROOT, "multiplicative_weights/figures")
QL = os.path.join(ROOT, "tabular_q_learning/figures")
OUT_FIG = os.path.join(ROOT, "paper/figures")
EARLY = f"{MF}/comparison/earlier_runs"

CONFIG = {
    # MWU: regret curves / table: continuous residual vs discrete (2 seeds each) + raw history
    "mwu_eval": [f"{MF}/comparison/matched_memory"],
    # per-model regret at T=1000 for the leak panel: label -> (eval dir, group, seed index)
    "mwu_runs": {
        "cont_v1_s42": (f"{EARLY}/eval_v1", "rec_cont", 0), "cont_v1_s43": (f"{EARLY}/eval_v1", "rec_cont", 1),
        "cont_v3_s42": (f"{EARLY}/eval_v3", "rec_cont", 0), "cont_v3_s43": (f"{EARLY}/eval_v3", "rec_cont", 1),
        "cont_v2_s42": (f"{EARLY}/eval_v2", "rec_cont", 0),
        "cont_v5plain_s42": (f"{MF}/continuous_overwrite/evaluation", "rec_cont", 0),
        "cont_v5plain_s43": (f"{MF}/continuous_overwrite/evaluation", "rec_cont", 1),
        "cont_v5res_s42": (f"{MF}/comparison/matched_memory", "rec_cont", 0),
        "cont_v5res_s43": (f"{MF}/comparison/matched_memory", "rec_cont", 1),
        "disc_v1_s42": (f"{EARLY}/eval_v1", "rec_disc", 0), "disc_v1_s43": (f"{EARLY}/eval_v1", "rec_disc", 1),
        "disc_v3_s42": (f"{EARLY}/eval_v3", "rec_disc", 0), "disc_v3_s43": (f"{EARLY}/eval_v3", "rec_disc", 1),
        "disc_v2_s42": (f"{EARLY}/eval_v2", "rec_disc", 0),
        "disc_v5_s42": (f"{MF}/comparison/matched_memory", "rec_disc", 0),
        "disc_v5_s43": (f"{MF}/comparison/matched_memory", "rec_disc", 1),
    },
    # mechanistic results (rho, probes, steering); later files override earlier
    "mwu_mech": [f"{MF}/comparison/mechanism/{f}" for f in
                 ("mech_existing.json", "mech_steer.json", "mech_all.json", "mech_disc_v5.json")],
    # MW weights vs weights decoded from the latents, one held-out sequence
    # (figures_per_model.py --only attention)
    # theorem-matched data (one hidden expert, labels flipped w.p. 0.2); seed 43 learns WMA
    "mwu_theorem": (f"{MF}/continuous_residual/theorem_matched/results.json", "seed43"),
    "mwu_weights": f"{MF}/continuous_residual/attention/weight_trajectories.npz",
    "mwu_steer_model": ["cont_v5res_s43", "cont_v2", "cont_v2_s42"],
    # Q-learning: clean-step disagreement (csv, continuous checkpoint key, discrete checkpoint key)
    "q_disagree": [((f"{QL}/comparison/clean_steps/clean_steps.csv", "qlv3-residual", "qlv3-discrete"),)],
    # Q-learning: closed-loop suite (continuous dir, discrete dir), model's own a*
    "q_closed": [(f"{QL}/continuous_residual/closed_loop", f"{QL}/discrete/closed_loop")],
    # the same long-horizon runs with an eps-greedy behaviour policy on the model's own argmax
    "q_closed_eps": (f"{QL}/continuous_residual/closed_loop_exploration",
                     f"{QL}/discrete/closed_loop_exploration"),
    "q_eps": 0.2,                       # the tabular eps-greedy baseline's epsilon
    # behavioural (alpha, gamma) fit on clean steps
    "q_alpha_gamma": f"{QL}/comparison/behavioural_fit/alpha_gamma.json",
    # Q-table probe: (csv, continuous checkpoint key, discrete checkpoint key)
    "q_probe": [(f"{QL}/comparison/q_probe/q_probe.csv", "qlv3-residual", "qlv3-discrete")],
    # main Q figure: held-out probe predictions (eval_q_probe.py) and teacher-forced agreement
    # (4_evaluate.py combined_row_data.npz) of the continuous and discrete models
    "q_probe_pred": (f"{QL}/comparison/q_probe/q_probe_pred.npz",
                     "coconut_transformer_qlv3-residual", "coconut_transformer_qlv3-discrete"),
    "q_row": (f"{QL}/continuous_residual/evaluation/combined_row_data.npz",
              f"{QL}/discrete/evaluation/combined_row_data.npz"),
}

# palette: categorical slots 1-3 of the reference palette (validated: all checks
# pass; aqua < 3:1 on white, so its line is direct-labelled), references in ink tones
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#9a9994", "#e4e3df"

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9, "legend.fontsize": 8,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "axes.axisbelow": True,
    "grid.color": GRID, "grid.linewidth": 0.5, "lines.linewidth": 1.4, "lines.markersize": 4,
    "pdf.fonttype": 42,
})
DRAFT = []
LABEL_FS = 8          # direct labels and value labels


def new_ax(w=3.3, h=2.4):
    fig, ax = plt.subplots(figsize=(w, h), constrained_layout=True)
    return fig, ax


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT_FIG, f"{name}.{ext}"), dpi=300)
    plt.close(fig)


def first_path(p):
    while not isinstance(p, str):
        p = p[0]
    return p


def first_existing(paths, what):
    for i, p in enumerate(paths):
        if os.path.exists(first_path(p)):
            if i > 0:
                DRAFT.append(f"{what}: using fallback {first_path(p)}")
            return p
    raise FileNotFoundError(f"{what}: none of {paths}")


# ---------------------------------------------------------------------------
# MWU
# ---------------------------------------------------------------------------

def load_mech():
    mech = {}
    for p in CONFIG["mwu_mech"]:
        if os.path.exists(p):
            with open(p) as f:
                for k, v in json.load(f).items():
                    mech.setdefault(k, {}).update(v)
    if not os.path.exists(CONFIG["mwu_mech"][-1]):
        DRAFT.append("MWU mechanistic: mech_all.json missing, using existing subsets")
    return mech


def run_regret(label):
    d, grp, i = CONFIG["mwu_runs"][label]
    p = os.path.join(d, "results.json")
    if not os.path.exists(p):
        return None
    with open(p) as f:
        h = json.load(f).get("horizon", {})
    if "1000" not in h:
        return None                     # eval still running
    seeds = h["1000"][grp]["regret_seed_means"]
    return seeds[i] if i < len(seeds) else None


def fig_mwu():
    # an eval writes results.json incrementally; it is complete once it has trajectories
    for i, ev in enumerate(CONFIG["mwu_eval"]):
        p = os.path.join(ev, "results.json")
        if os.path.exists(p):
            with open(p) as f:
                res = json.load(f)
            if "trajectories" in res and "regimes" in res:
                if i > 0:
                    DRAFT.append(f"MWU eval: using fallback {ev}")
                break
    else:
        raise FileNotFoundError("no complete MWU eval")
    mech = load_mech()
    fig_mwu_main(res["trajectories"])

    # retention rho vs regret at T=1000. rho comes from regressing the probe-decoded
    # state on its previous value, so it is only meaningful where the state is
    # decodable: models with MW-state probe R^2 < MIN_PROBE_R2 (every discrete model,
    # and two continuous runs) are left out.
    MIN_PROBE_R2 = 0.1
    small = {"font.size": 6.5, "axes.titlesize": 7, "axes.labelsize": 6.5, "legend.fontsize": 6,
             "xtick.labelsize": 6, "ytick.labelsize": 6}
    with plt.rc_context(small):
        fig, ax = new_ax(2.6, 2.0)
        skipped = []
        for label in CONFIG["mwu_runs"]:
            m = mech.get(label) or mech.get(label.replace("_s42", ""))
            r = run_regret(label)
            if m is None or r is None or "update_rule" not in m:
                continue
            rho = m["update_rule"].get("6-95", {}).get("rho")
            if rho is None:
                continue
            if m["probes"]["mw_state"]["linear"]["31-95"] < MIN_PROBE_R2:
                skipped.append(label)
                continue
            res_w = "v5res" in label        # the residual write of Algorithm 1
            ax.scatter(rho, r, s=18, color=BLUE if res_w else MUTED, edgecolor="white", linewidth=0.5,
                       zorder=3)
        if skipped:
            print(f"MWU leak panel: left out (state probe R^2 < {MIN_PROBE_R2}): {', '.join(skipped)}")
        mw = res["horizon"]["1000"]["mw"]["regret_mean"]
        ax.axhline(mw, color=INK2, ls=(0, (1.5, 1.5)), lw=0.9)
        ax.text(0.927, mw, "MW", va="bottom", ha="left", fontsize=6, color=INK2)
        from matplotlib.lines import Line2D
        ax.legend(handles=[Line2D([], [], ls="", marker="o", ms=3.5, color=BLUE, label="residual write"),
                           Line2D([], [], ls="", marker="o", ms=3.5, color=MUTED, label="other latents")],
                  frameon=False, loc="upper right", handletextpad=0.1, borderaxespad=0.2)
        ax.set_xlim(0.925, 1.005)
        ax.set_xlabel(r"retention $\rho$")
        ax.set_ylabel("regret at $T=1000$")
        # memory length of a latent that decays by rho per round
        top = ax.secondary_xaxis("top")
        ticks = [0.93, 0.95, 0.97, 0.99, 1.0]
        top.set_xticks(ticks, ["14", "20", "33", "100", r"$\infty$"])
        top.set_xlabel(r"memory $1/(1-\rho)$ (rounds)")
        save(fig, "fig_mwu_leak")

    # steering
    fig, ax = new_ax(3.6, 2.4)
    name = next((n for n in CONFIG["mwu_steer_model"]
                 if n in mech and any(k.startswith("mw_state") for k in
                                      mech[n].get("steering", {}).get("t=80", {}))), None)
    if name is None:
        DRAFT.append("MWU steering: no model with the new steering format")
    else:
        if name != CONFIG["mwu_steer_model"][0]:
            DRAFT.append(f"MWU steering: using {name}")
        st = mech[name]["steering"]["t=80"]
        for key, lab, c, ls in [("mw_state", "more loss for expert $i$", INK, "-"),
                                ("bayes", "higher log-odds for $i$", INK2, (0, (4, 1.5))),
                                ("random", "random direction", MUTED, (0, (1.5, 1.5)))]:
            xs = sorted(float(k.split()[1]) for k in st if k.startswith(key + " "))
            ys = [100 * st[f"{key} {x:g}"] for x in xs]
            ax.plot(xs, ys, color=c, ls=ls, marker="o", markersize=3.5, lw=1.3)
            ax.text(xs[-1] + 0.08, ys[-1], lab, color=INK if c == MUTED else c,
                    fontsize=LABEL_FS, va="center")
        ax.set_xlim(min(xs) - 0.05, max(xs) + 1.25)
        ax.set_xticks([-1, -0.5, 0, 0.5, 1])
    ax.set_xlabel(r"latent shift (fraction of $\|\mathbf{z}_t\|$)")
    ax.set_ylabel("follows expert $i$ (%)")
    ax.set_title("Steering the latent", loc="left")
    save(fig, "fig_mwu_steering")
    return res, mech


# ---------------------------------------------------------------------------
# Q-learning
# ---------------------------------------------------------------------------

def load_disagree():
    spec = first_existing(CONFIG["q_disagree"], "Q clean-step")
    frames = []
    if len(spec) == 1 and len(spec[0]) == 3:  # one file holding both models
        path, ck, dk = spec[0]
        df = pd.read_csv(path)
        if not (df.checkpoint.str.contains(dk).any() and df.checkpoint.str.contains(ck).any()):
            DRAFT.append(f"Q clean-step: {path} lacks {dk}; using fallback")
            spec = CONFIG["q_disagree"][1]
        else:
            for key, model in ((ck, "continuous"), (dk, "discrete")):
                d = df[df.checkpoint.str.contains(key)].copy()
                d["model"] = model
                frames.append(d)
            return pd.concat(frames)
    if isinstance(spec[0], str):              # one matched file
        df = pd.read_csv(spec[0])
        df["model"] = np.where(df.checkpoint.str.contains("discrete"), "discrete", "continuous")
        frames.append(df)
    else:
        for path, model in spec:
            df = pd.read_csv(path)
            df = df[df.checkpoint.str.contains(model)]
            df["model"] = model
            frames.append(df)
    return pd.concat(frames)


def clean_margin(df, model, group, rule):
    r = df[(df.model == model) & (df.group == group) & (df.rule == rule)].iloc[0]
    ft, fr, n = r.clean_follow_teacher, r.clean_follow_rule, r.clean_n
    # SEM of the difference of two proportions from the same multinomial
    sem = np.sqrt((ft + fr - (ft - fr) ** 2) / n)
    return 100 * (ft - fr), 100 * sem, 100 * ft, 100 * fr


def load_closed():
    c_dir, d_dir = next(((c, d) for c, d in CONFIG["q_closed"]
                         if os.path.exists(os.path.join(c, "long_horizon_summary.csv"))
                         and os.path.exists(os.path.join(d, "long_horizon_summary.csv"))),
                        CONFIG["q_closed"][-1])
    if (c_dir, d_dir) != CONFIG["q_closed"][0]:
        DRAFT.append(f"Q closed-loop: using fallback {c_dir}")
    out = {}
    for model, d in (("continuous", c_dir), ("discrete", d_dir)):
        df = pd.read_csv(os.path.join(d, "long_horizon_summary.csv"))
        out[model] = df[df.condition == "contrast"].set_index("agent")
    for model, d in zip(("continuous", "discrete"), CONFIG["q_closed_eps"]):
        df = pd.read_csv(os.path.join(d, "long_horizon_summary.csv"))
        df = df[df.condition == "contrast"].set_index("agent")
        out[model].loc["tf_eps"] = df.loc[f"tf_self_eps{CONFIG['q_eps']:g}"]
    return out


def fig_mwu_main(tr):
    """Two panels, continuous vs discrete transformer: regret against the best expert, and MW's
    weights vs the weights computed from the cumulative losses a linear probe decodes from the
    latent of each transformer (dashed: discrete, whose latent decodes nothing, R^2 = -0.00)."""
    small = {"font.size": 6.5, "axes.titlesize": 7, "axes.labelsize": 6.5, "legend.fontsize": 6,
             "xtick.labelsize": 6, "ytick.labelsize": 6}
    with plt.rc_context(small):
        fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.1), constrained_layout=True)
        fig.get_layout_engine().set(w_pad=0.1, wspace=0.08)
        # (left) cumulative regret
        ax = axes[0]
        t = np.arange(1, len(tr["mw"]) + 1)
        for k, lab, c, ls in (("rec_cont", "continuous", BLUE, "-"),
                              ("rec_disc", "discrete", ORANGE, "-"),
                              ("majority", "majority vote", MUTED, (0, (4, 1.5))),
                              ("mw", "MW", INK2, (0, (1.5, 1.5))),
                              ("bayes", "Bayes", INK, (0, (5, 1.5, 1, 1.5)))):
            ax.plot(t, np.asarray(tr[k]), color=c, ls=ls, lw=1.3 if c in (BLUE, ORANGE) else 1.0,
                    label=lab, zorder=3 if c == MUTED else 2)
        ax.axhline(0, color=GRID, lw=0.8, zorder=0)
        ax.axvline(200, color=INK2, lw=0.7, ls=":", zorder=1)   # longest training sequence
        ax.text(212, 0.52, "training horizon", transform=ax.get_xaxis_transform(), fontsize=5.5,
                color=INK2, va="bottom", ha="left", rotation=90)
        ax.set_xlim(0, t[-1])
        ax.set_xlabel("round $t$")
        ax.set_ylabel("cumulative regret")
        ax.legend(frameon=False, loc="lower right", bbox_to_anchor=(1, 0.18), handlelength=1.8)
        ax.set_title("Regret against the best expert", loc="left")
        # (right) MW weights and the weights decoded from the latent
        ax = axes[1]
        z = np.load(CONFIG["mwu_weights"])
        t = np.arange(1, len(z["mw"]) + 1)
        cols = [BLUE, ORANGE, AQUA, "#e87ba4"]
        for e, c in enumerate(cols):
            ax.plot(t, z["mw"][:, e], color=c, lw=2.2, alpha=0.45)
            ax.plot(t, z["continuous"][:, e], color=c, lw=1.0)
            ax.plot(t, z["discrete"][:, e], color=c, lw=0.8, ls=(0, (3, 2)))
        from matplotlib.lines import Line2D
        style = ax.legend(handles=[Line2D([], [], color=INK2, lw=2.2, alpha=0.45, label="MW"),
                                   Line2D([], [], color=INK2, lw=1.0, label="continuous"),
                                   Line2D([], [], color=INK2, lw=0.8, ls=(0, (3, 2)), label="discrete")],
                          frameon=False, loc="upper left", handlelength=1.5, borderaxespad=0.2)
        ax.add_artist(style)
        ax.legend(handles=[Line2D([], [], color=c, lw=1.2, label=f"expert {e + 1}")
                           for e, c in enumerate(cols)],
                  frameon=False, loc="upper right", ncol=2, handlelength=1.2, columnspacing=0.8,
                  borderaxespad=0.2)
        ax.set_xlim(0, t[-1])
        ax.set_ylim(0, 0.72)                                # headroom for the legend
        ax.set_xlabel("round $t$")
        ax.set_ylabel("expert weight")
        ax.set_title("MW weights decoded from the latent", loc="left")
        save(fig, "fig_mwu_main")


def fig_q_main():
    """Three panels as in the earlier figure, continuous vs discrete transformer: closed-loop
    cumulative reward (eps-greedy), Q-table probe, teacher-forced agreement over the episode."""
    small = {"font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7, "legend.fontsize": 6.5,
             "xtick.labelsize": 6.5, "ytick.labelsize": 6.5}
    with plt.rc_context(small):
        _fig_q_main()


def _fig_q_main():
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.1), constrained_layout=True)
    fig.get_layout_engine().set(w_pad=0.06, wspace=0.08)
    eps = CONFIG["q_eps"]
    # (left) 1000-step contrast MDPs, model's own a*, eps-greedy behaviour policy
    ax = axes[0]
    zc, zd = (np.load(os.path.join(d, "long_horizon_rewards.npz")) for d in CONFIG["q_closed_eps"])
    key = lambda a: f"contrast__{a}"
    series = [(zc[key("epsgreedy")], "tabular $Q$", MUTED, "--"),
              (zc[key(f"tf_self_eps{eps:g}")], "continuous", BLUE, "-"),
              (zd[key(f"tf_self_eps{eps:g}")], "discrete", ORANGE, "-"),
              (zc[key("random")], "random", "#c9c8c3", "-")]
    t = np.arange(1, zc[key("optimal")].shape[1] + 1)
    for r, lab, col, ls in series:
        c = np.cumsum(r, axis=1)
        m, s = c.mean(0), c.std(0) / np.sqrt(len(c))
        ax.plot(t, m, color=col, ls=ls, label=lab)
        ax.fill_between(t, m - 1.96 * s, m + 1.96 * s, color=col, alpha=0.15, lw=0)
    T_TRAIN = 200                                           # longest training episode
    ax.axvline(T_TRAIN, color=INK2, lw=0.7, ls=":", zorder=1)
    ax.text(T_TRAIN + 12, 0.3, "training horizon", transform=ax.get_xaxis_transform(),
            fontsize=6, color=INK2, va="bottom", ha="left", rotation=90)
    ax.set_xlabel("step $t$")
    ax.set_ylabel("cumulative reward")
    ax.set_xlim(0, t[-1])
    ax.set_ylim(0, ax.get_ylim()[1] * 1.12)                 # headroom for the legend
    ax.legend(frameon=False, loc="upper left", ncol=2, handlelength=1.6, columnspacing=1.0,
              borderaxespad=0.2)
    ax.set_title(f"Long-horizon rewards ($\\epsilon={eps:g}$)", loc="left")
    # (middle) agreement with tabular Q-learners of every discount (alpha = 0.2)
    ax = axes[1]
    with open(CONFIG["q_alpha_gamma"]) as f:
        d = json.load(f)
    ia, g = d["alphas"].index(0.2), np.array(d["gammas"])
    ax.axvline(0.9, color=INK2, lw=0.7, ls=":", zorder=1)
    ax.text(0.885, 0.33, "teacher's $\\gamma$", rotation=90, fontsize=6, color=INK2,
            ha="right", va="bottom", transform=ax.get_xaxis_transform())
    for key, lab, col in (("qlv3-residual", "continuous", BLUE), ("qlv3-discrete", "discrete", ORANGE)):
        ax.plot(g, np.array(d["models"][key]["all"])[ia], color=col, marker="o", ms=2.5, label=lab)
    ax.set_xlim(-0.03, 1.0)
    ax.set_ylim(0.4, 1.0)
    ax.set_xlabel("discount $\\gamma$ of the tabular learner")
    ax.set_ylabel("agreement")
    ax.legend(frameon=False, loc="center left", borderaxespad=0.2)
    ax.set_title("Agreement vs. discount", loc="left")
    # (right) teacher-forced agreement with tabular Q-learning's greedy action
    ax = axes[2]
    W = 5                                                   # steps per window
    for path, lab, col in zip(CONFIG["q_row"], ("continuous", "discrete"), (BLUE, ORANGE)):
        z = np.load(path)
        a = np.concatenate([z[f"agree_{i}"] for i in range(len(z["dist_labels"]))])  # [MDPs, T]
        a = a[:, :a.shape[1] // W * W].reshape(len(a), -1, W).mean(-1)            # [MDPs, T/W]
        m, s = a.mean(0), a.std(0) / np.sqrt(len(a))
        steps = W * np.arange(len(m)) + (W + 1) / 2
        ax.plot(steps, m, color=col, marker="o", ms=2.5, label=lab)
        ax.fill_between(steps, m - 1.96 * s, m + 1.96 * s, color=col, alpha=0.15, lw=0)
    ax.set_xlabel("step $t$ (teacher forced)")
    ax.set_ylabel("agreement")
    ax.set_ylim(0, 1.02)
    ax.set_xlim(0, W * len(m) + 2)
    ax.set_xticks(range(0, W * len(m) + 1, 10))
    ax.legend(frameon=False, loc="lower right")
    ax.set_title("Agreement over an episode", loc="left")
    save(fig, "fig_q_main")


def fig_q_probe():
    """Held-out linear probe from each context token c_a to the teacher's Q_t(., a) (eval_q_probe.py)."""
    small = {"font.size": 6.5, "axes.titlesize": 7, "axes.labelsize": 6.5, "legend.fontsize": 6,
             "xtick.labelsize": 6, "ytick.labelsize": 6}
    with plt.rc_context(small):
        fig, ax = new_ax(2.7, 2.3)
        path, ck, dk = CONFIG["q_probe_pred"]
        zp = np.load(path)
        rc, rd = q_probe()
        rng = np.random.default_rng(0)
        for k, lab, col, r2 in ((dk, "discrete", ORANGE, rd), (ck, "continuous", BLUE, rc)):
            qt, qp = zp[f"{k}__true"], zp[f"{k}__pred"]
            i = rng.choice(len(qt), size=min(4000, len(qt)), replace=False)
            ax.scatter(qt[i], qp[i], s=2, alpha=0.25, color=col, lw=0, rasterized=True,
                       label=f"{lab} ($R^2={round(r2, 2) + 0.0:.2f}$)")
        lo, hi = np.percentile(zp[f"{ck}__true"], [0.5, 99.5])
        ax.plot([lo, hi], [lo, hi], color=INK2, lw=0.8, ls="--")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo - 0.1 * (hi - lo), hi + 0.1 * (hi - lo))
        ax.set_xlabel("teacher $Q(s, a)$")
        ax.set_ylabel("decoded from context token $\\mathbf{c}_a$")
        h, l = ax.get_legend_handles_labels()
        leg = ax.legend(h[::-1], l[::-1], loc="upper left", markerscale=4, handletextpad=0.2,
                        borderaxespad=0.2, frameon=True, facecolor="white", edgecolor="none", framealpha=0.9)
        for lh in leg.legend_handles:
            lh.set_alpha(1)
        ax.set_title("$Q$-values decoded from memory", loc="left")
        save(fig, "fig_q_probe")


def fig_q():
    dis = load_disagree()
    closed = load_closed()
    # clean-step margins
    fig, ax = new_ax(4.6, 2.6)
    comps = [("ALL", "q_g0", "myopic\n($\\gamma=0$)"),
             ("task=goal", "q_g0", "myopic, goal-\nstate MDPs"),
             ("ALL", "q_g0.5", "$\\gamma=0.5$"),
             ("ALL", "most_tried", "reward-blind\n(most tried)")]
    w = 0.36
    for j, (model, c) in enumerate((("continuous", BLUE), ("discrete", ORANGE))):
        for i, (grp, rule, _) in enumerate(comps):
            m, s, ft, fr = clean_margin(dis, model, grp, rule)
            x = i + (j - 0.5) * w
            ax.bar(x, m, width=w - 0.04, color=c, yerr=1.96 * s, capsize=1.5,
                   error_kw={"elinewidth": 0.7, "ecolor": INK2}, zorder=3,
                   label=model if i == 0 else None)
            ax.text(x, m + (1.96 * s + 1.5 if m >= 0 else -1.96 * s - 1.5), f"{m:+.0f}",
                    ha="center", va="bottom" if m >= 0 else "top", fontsize=LABEL_FS, color=INK)
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_xticks(range(len(comps)))
    ax.set_xticklabels([c[2] for c in comps])
    ax.set_ylabel("follows $Q$-learning $-$\nfollows rule (pts)")
    # legend over the first group (its bar is the lowest positive one), so it covers no label
    ax.legend(frameon=False, loc="upper left", ncol=1)
    ax.set_title("Clean steps: $Q$-learning ($\\gamma=0.9$) vs. a simpler rule", loc="left")
    lo, hi = ax.get_ylim()
    ax.set_ylim(min(lo, -5) - 4, hi * 1.3)   # room for the value labels below / above the bars
    save(fig, "fig_q_clean")

    fig_q_main()
    fig_q_probe()
    return dis, closed


# ---------------------------------------------------------------------------
# Table
# ---------------------------------------------------------------------------

def q_probe():
    for path, c_key, d_key in CONFIG["q_probe"]:
        if os.path.exists(path):
            df = pd.read_csv(path)
            if df.checkpoint.str.contains(d_key).any():
                break
    else:
        raise FileNotFoundError("Q probe")
    if path != CONFIG["q_probe"][0][0]:
        DRAFT.append(f"Q probe: using fallback {path}")
    df = df[df.target == "q_g0.9"]
    get = lambda k: float(df[df.checkpoint.str.contains(k)].r2.iloc[0])
    return get(c_key), get(d_key)


def main():
    os.makedirs(OUT_FIG, exist_ok=True)
    res, mech = fig_mwu()
    dis, closed = fig_q()
    print("wrote paper/figures/fig_mwu_{main,leak,steering}, fig_q_{main,clean,probe} "
          "(.pdf/.png)")
    for d in DRAFT:
        print("DRAFT:", d)


if __name__ == "__main__":
    main()
