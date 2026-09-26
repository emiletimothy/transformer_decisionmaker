"""
Run all relaxation sweeps for both constructions and write CSVs.

    python run_sweeps.py [--n_inst 20] [--T 500] [--procs 16]

Sweeps (each point averaged over n_inst random instances, reported at T=100 and T):
  beta_route   : inverse temperature of the hard routing / masking heads
  beta_max     : (Q only) inverse temperature of the max-approximating policy head
  beta_x_noise : beta_route x residual-stream noise sigma (added after every layer)
  beta_x_dim   : beta_route x embedding dimension d (random unit token embeddings, so
                 non-orthogonal; weights written through the embeddings, W' = Phi W Phi^T)
  overlap      : buffer subspaces overlap (cross-block |cos| <= rho), beta_route = 50
  overlap_dual : the same overlap with dual-basis read-out, plus residual noise sigma
  dual_x_dim   : same random embeddings as beta_x_dim but with dual-basis read-out
                 (W' = Phi^{+T} W Phi^+), plus physical residual noise sigma; beta_route = 50.
                 Exact up to noise, which is amplified by (Phi^T Phi)^{-1}.

Default: the current constructions (relax_v2.py: HandwiredQv2 with two buffers, HandwiredMWUv2
with one buffer) -> results/{mwu,q}_relaxation{_raw}.csv.
--v1 [--residual] re-runs the earlier sweeps of the v1-style matrices (relax_q.py / relax_mwu.py)
-> results/earlier_runs/{mwu,q}_relaxation_v1{_residual}{_raw}.csv.
"""
import argparse
import os
from multiprocessing import Pool

import numpy as np
import pandas as pd

from relax_common import random_unit_gram, overlap_blocks_gram, block_diag
from relax_mwu import RelaxedMWU, expert_sequence
from relax_q import RelaxedQ, random_mdp_trajectory
from relax_v2 import RelaxedQv2, RelaxedMWUv2, full_gram, random_mdp_trajectory as traj_v2

HERE = os.path.dirname(os.path.abspath(__file__))

N_EXPERTS, ETA = 4, 0.1
RESIDUAL = False            # set by --residual (inherited by the worker processes)
V1 = False                  # set by --v1
SUFFIX = ""
RESULTS = os.path.join(HERE, "results")
NS, NA = 6, 3

BETAS_1D = [1, 2, 5, 10, 20, 50, 100, 1e3, 1e4, 1e6]
BETAS_2D = [10, 20, 50, 100, 1e3, 1e6]
SIGMAS = [0.0, 1e-5, 1e-4, 1e-3, 1e-2, 3e-2, 1e-1, 3e-1]
DIMS = [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]
RHOS = [0.0, 0.01, 0.03, 0.1, 0.2, 0.3, 0.5]
BETA_OVERLAP = 50
DUAL_DIMS = [16, 24, 32, 64, 128, 256, 512, 1024, 4096, 16384]
DUAL_SIGMAS = [0.0, 1e-5, 1e-4, 1e-3]


def grid(construction):
    g = [("exact", {})]
    g += [("beta_route", dict(beta=b)) for b in BETAS_1D]
    if construction == "q":
        g += [("beta_max", dict(beta_max=b)) for b in BETAS_1D]
    g += [("beta_x_noise", dict(beta=b, sigma=s)) for b in BETAS_2D for s in SIGMAS]
    g += [("beta_x_dim", dict(beta=b, dim=d)) for b in BETAS_2D for d in DIMS]
    g += [("overlap", dict(beta=BETA_OVERLAP, rho=r)) for r in RHOS]
    # same overlapping buffers, read out through the dual basis (as dual_x_dim)
    g += [("overlap_dual", dict(beta=BETA_OVERLAP, rho=r, sigma=s, dual=True))
          for s in DUAL_SIGMAS for r in RHOS]
    g += [("dual_x_dim", dict(beta=BETA_OVERLAP, dim=d, sigma=s, dual=True))
          for s in DUAL_SIGMAS for d in DUAL_DIMS]
    return g


def sweep_filter(g, names):
    return g if not names else [x for x in g if x[0] in names]


def make_gram(n_tok, n_blocks, extra, p, rng):
    """Gram matrix over [n_blocks x n_tok token blocks | extra unrelaxed dims]."""
    maxcos = 0.0
    if "dim" in p:
        Gu, maxcos = random_unit_gram(n_tok, int(p["dim"]), rng)
        Gb = block_diag(*([Gu] * n_blocks))
    elif "rho" in p:
        Gb, maxcos = overlap_blocks_gram(n_tok, n_blocks, p["rho"], rng)
    else:
        Gb = np.eye(n_tok * n_blocks)
    G = block_diag(Gb, np.eye(extra)) if extra else Gb
    return G, maxcos


def dual(G, p):
    """Dual-basis read-out: the computation sees G = I, noise covariance becomes G^{-1}."""
    if p.get("dual"):
        return np.eye(len(G)), np.linalg.inv(G)
    return G, None


def task_v2(construction, sweep, p, seed, T):
    if construction == "mwu":
        m = RelaxedMWUv2(N_EXPERTS, ETA)
        rng = np.random.default_rng(10_000 + seed)
        preds, labels = expert_sequence(N_EXPERTS, T, seed)
        run = lambda G, Gn: m.run(preds, labels, G, beta=p.get("beta", 1e3), sigma=p.get("sigma", 0.0),
                                  seed=seed, report_T=(100, T), Gnoise=Gn)
    else:
        m = RelaxedQv2(NS, NA)
        rng = np.random.default_rng(20_000 + seed)
        traj = traj_v2(NS, NA, T, seed)
        run = lambda G, Gn: m.run(traj, G, beta=p.get("beta", 1e3), beta_max=p.get("beta_max", 1e4),
                                  sigma=p.get("sigma", 0.0), seed=seed, report_T=(100, T), Gnoise=Gn)
    Gb, maxcos = make_gram(m.dTE, len(m.token_blocks), 0, p, rng)
    G, Gn = dual(full_gram(m.d, m.token_blocks, Gb), p)
    with np.errstate(all="ignore"):
        return run(G, Gn), maxcos


def task(args):
    construction, sweep, p, seed, T = args
    if not V1:
        res, maxcos = task_v2(construction, sweep, p, seed, T)
    elif construction == "mwu":
        m = RelaxedMWU(N_EXPERTS, ETA, residual=RESIDUAL)
        rng = np.random.default_rng(10_000 + seed)
        G, maxcos = make_gram(m.dTE, m.n_blocks, m.d - m.n_blocks * m.dTE, p, rng)
        G, Gn = dual(G, p)
        preds, labels = expert_sequence(N_EXPERTS, T, seed)
        with np.errstate(all="ignore"):
            res = m.run(preds, labels, G, beta=p.get("beta", 1e6), sigma=p.get("sigma", 0.0),
                        seed=seed, report_T=(100, T), Gnoise=Gn)
    else:
        m = RelaxedQ(NS, NA, residual=RESIDUAL)
        rng = np.random.default_rng(20_000 + seed)
        G, maxcos = make_gram(m.dTE, 3, 0, p, rng)
        G, Gn = dual(G, p)
        traj = random_mdp_trajectory(NS, NA, T, seed)
        with np.errstate(all="ignore"):
            res = m.run(traj, G, beta_route=p.get("beta"), beta_max=p.get("beta_max"),
                        sigma=p.get("sigma", 0.0), seed=seed, force_causal=True,
                        report_T=(100, T), Gnoise=Gn)
    base = dict(construction=construction, sweep=sweep, seed=seed, max_cos=maxcos,
                beta=p.get("beta", np.nan), beta_max=p.get("beta_max", np.nan),
                sigma=p.get("sigma", np.nan), dim=p.get("dim", np.nan), rho=p.get("rho", np.nan))
    return [dict(base, T=t, **r) for t, r in res.items()]


KEYS = ["construction", "sweep", "beta", "beta_max", "sigma", "dim", "rho", "T"]


def summarize(df, metrics):
    df = df.replace([np.inf, -np.inf], np.nan)
    # Diverged runs (NaN/inf) are counted as failures rather than dropped.
    df["diverged"] = df[metrics[0]].isna().astype(float)
    g = df.groupby(KEYS, dropna=False)
    out = g[metrics + ["max_cos", "diverged"]].mean().add_suffix("_mean")
    sem = g[metrics].std(ddof=1).div(np.sqrt(g.size()), axis=0).add_suffix("_sem")
    out = out.join(sem)
    out["n"] = g.size()
    return out.reset_index()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_inst", type=int, default=20)
    ap.add_argument("--T", type=int, default=500)
    ap.add_argument("--procs", type=int, default=16)
    ap.add_argument("--only", choices=["mwu", "q"], default=None)
    ap.add_argument("--v1", action="store_true",
                    help="the earlier sweeps of the v1-style matrices (results/earlier_runs/)")
    ap.add_argument("--residual", action="store_true",
                    help="with --v1: the residual-write v1 constructions")
    ap.add_argument("--sweeps", nargs="*", default=None,
                    help="subset of sweeps; results are merged into existing CSVs")
    args = ap.parse_args()
    global RESIDUAL, SUFFIX, V1, RESULTS
    V1, RESIDUAL = args.v1, args.residual
    if V1:
        SUFFIX = "_v1_residual" if RESIDUAL else "_v1"
        RESULTS = os.path.join(HERE, "results", "earlier_runs")
    print("constructions:", ("v1 " + ("residual" if RESIDUAL else "overwrite")) if V1 else
          "current (relax_v2.py)", flush=True)

    tasks = []
    for c in (["mwu", "q"] if args.only is None else [args.only]):
        tasks += [(c, s, p, i, args.T) for s, p in sweep_filter(grid(c), args.sweeps)
                  for i in range(args.n_inst)]
    # Q tasks are ~100x slower than MWU; schedule them first.
    tasks.sort(key=lambda t: t[0] != "q")
    print(f"{len(tasks)} tasks", flush=True)

    if args.procs == 1:
        rows = [r for t in tasks for r in task(t)]
    else:
        with Pool(args.procs) as pool:
            rows = [r for rs in pool.imap_unordered(task, tasks, chunksize=4) for r in rs]

    df = pd.DataFrame(rows)
    metrics = {"mwu": ["max_pred_err", "w_linf", "w_l1", "max_w_linf", "decision_agree",
                       "regret_diff"],
               "q": ["max_err", "final_err", "greedy_agree", "select_agree"]}
    for c in df.construction.unique():
        sub = df[df.construction == c]
        raw_path = os.path.join(RESULTS, f"{c}_relaxation{SUFFIX}_raw.csv")
        if args.sweeps and os.path.exists(raw_path):
            old = pd.read_csv(raw_path)
            sub = pd.concat([old[~old.sweep.isin(args.sweeps)], sub], ignore_index=True)
        sub.to_csv(raw_path, index=False)
        s = summarize(sub.copy(), metrics[c])
        s.to_csv(os.path.join(RESULTS, f"{c}_relaxation{SUFFIX}.csv"), index=False)
        print(f"wrote {c}: {len(s)} summary rows", flush=True)


if __name__ == "__main__":
    main()
