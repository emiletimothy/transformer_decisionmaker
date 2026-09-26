#!/usr/bin/env python3
"""
eval_theorem_matched.py — Does the model trained on the theorem's latent-expert
data compute MWU (eta = 1 on log-loss), which is the Bayes predictor there?

On fresh sequences (longer than training) it compares the model's predicted
P(y = 1) with
  mw1        MWU, eta = 1 on log-loss (= Bayes; the population-loss minimizer)
  mw_tuned   MWU, eta = sqrt(ln n / T) on log-loss (the paper's baseline tuning)
  uniform    equal-weight mixture of the forecasts (best memoryless predictor)
  ftl        forecast of the expert with the lowest cumulative log-loss so far
  oracle     forecast of the true expert (knows I*)
and reports, per round bucket: mean |p_model - p_mw1|, log-loss of every predictor,
cumulative log-loss regret against the best expert in hindsight, a linear probe of
the latent for MWU's log-weights, and the fitted update rule s_t = rho s_{t-1} + k l_t.

Adversarial tests (--adv_T rounds, same forecast process, labels NOT from a true
expert). For log-loss, MWU with eta = 1 has regret <= ln n against EVERY sequence, so
a model that has learned MWU should stay under ln 4 on all of them:
  vs_model    y_t chosen after the model's prediction to maximise the model's regret
              increment against a designated expert j: argmax_y -log q(y) + log p_j(y)
  vs_mwu      the same adversary run against MWU(eta=1) (labels fixed for all predictors)
  switching   the generating expert changes every --switch_every rounds
  unrelated   y_t ~ Bernoulli(0.5), independent of every forecast
  stochastic  the training distribution, for reference

  python3 eval_theorem_matched.py --ckpts a.pt b.pt --labels s42 s43 \
      --out_dir ../figures/continuous_residual/theorem_matched
"""
import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import paths  # noqa: E402
import train  # noqa: E402
from train import (sample_latent_expert, encode_latent_expert,  # noqa: E402
                          sample_noisy_expert, encode_binary)

# Data model under test, set in main() from the checkpoint. For 'noisy_expert' every
# computation below uses the experts' noisy forecasts f = 1-eps if p=1 else eps, under
# which the Bayes predictor is exactly the eta = 1 mixture ("mw1"), i.e. weighted
# majority with eta = log((1-eps)/eps) on the 0/1 predictions.
DATA_MODEL = 'latent_expert'


def sample(n, T, E, rng):
    """Forecasts f [n,T,E], labels y, true expert, and the raw model inputs."""
    if DATA_MODEL == 'noisy_expert':
        p, y, true = sample_noisy_expert(n, T, E, rng)
        eps = train.NOISY_EPS
        return np.where(p == 1, 1 - eps, eps), y, true, p
    f, y, true = sample_latent_expert(n, T, E, rng)
    return f, y, true, f


def encode(raw, y, tok, upd):
    if DATA_MODEL == 'noisy_expert':
        return encode_binary(raw, y, tok, upd)
    return encode_latent_expert(raw, y, tok, upd)
from eval_matched_memory import load_recurrent  # noqa: E402

BUCKETS = [(1, 10), (11, 50), (51, 200), (201, 500)]
BLUE, AQUA, INK, INK2, MUTED, GRID = "#2a78d6", "#1baf7a", "#0b0b0b", "#52514e", "#9a9994", "#e4e3df"


def centred(x):
    return x - x.mean(-1, keepdims=True)


def mixture(p, cum_ll, eta):
    w = np.exp(eta * (cum_ll - cum_ll.max(-1, keepdims=True)))
    w /= w.sum(-1, keepdims=True)
    return (w * p).sum(-1), w


def logloss(q, y):
    q = np.clip(q, 1e-6, 1 - 1e-6)
    return -(y * np.log(q) + (1 - y) * np.log(1 - q))


def ridge(X, Y, lam=1e-2):
    mx, my = X.mean(0), Y.mean(0)
    A = X - mx
    W = np.linalg.solve(A.T @ A + lam * len(A) * np.eye(A.shape[1]), A.T @ (Y - my))
    return lambda Z: (Z - mx) @ W + my


def r2(pred, Y):
    return float(1 - ((Y - pred) ** 2).sum() / ((Y - Y.mean(0)) ** 2).sum())


def regret_curve(q, p, y):
    """Cumulative log-loss regret vs the best expert in hindsight at every round [n, T]."""
    ell = -np.where(y[..., None] == 1, np.log(p), np.log(1 - p))
    return np.cumsum(logloss(q, y), 1) - np.cumsum(ell, 1).min(-1)


def baselines(p, y, T):
    n, _, E = p.shape
    ll = np.where(y[..., None] == 1, np.log(p), np.log(1 - p))
    cum = np.concatenate([np.zeros((n, 1, E)), np.cumsum(ll, 1)[:, :-1]], 1)
    return {'mw1': mixture(p, cum, 1.0)[0],
            'mw_tuned': mixture(p, cum, np.sqrt(np.log(E) / T))[0],
            'uniform': p.mean(-1),
            'ftl': np.take_along_axis(p, cum.argmax(-1)[..., None], -1)[..., 0]}


@torch.no_grad()
def model_probs(model, raw, y, device, bs=150):
    ids, _ = encode(raw, y, model.tokenizer, model.UPD_TOKEN)
    out = []
    for i in range(0, len(raw), bs):
        out.append(torch.sigmoid(model.rollout(ids[i:i + bs].to(device))).cpu().numpy())
    return np.concatenate(out)


def regret_adversary(q, pj):
    """Label maximising the learner's regret increment against expert j:
    argmax_y [-log q(y) + log p_j(y)] for y in {0, 1}."""
    q, pj = np.clip(q, 1e-6, 1 - 1e-6), np.clip(pj, 1e-6, 1 - 1e-6)
    gain1 = -np.log(q) + np.log(pj)
    gain0 = -np.log(1 - q) + np.log(1 - pj)
    return (gain1 > gain0).astype(np.int64)


@torch.no_grad()
def adaptive_vs_model(model, p, raw, j, device):
    """Regret-maximising labels chosen online against the model (see regret_adversary).
    The decision logit is read at SEP, which cannot see y_t, so it is computed first
    and the round is then re-run with the adversarial label to update the latent."""
    n, T, _ = p.shape
    y = np.zeros((n, T), dtype=np.int64)
    q = np.zeros((n, T))
    M = model.initial_latent(n)
    for t in range(T):
        ids0, _ = encode(raw[:, t:t + 1], y[:, t:t + 1], model.tokenizer, model.UPD_TOKEN)
        lg, _, _, _ = model.step(M, ids0[:, 0].to(device))
        q[:, t] = torch.sigmoid(lg).cpu().numpy()
        y[:, t] = regret_adversary(q[:, t], p[np.arange(n), t, j])
        ids1, _ = encode(raw[:, t:t + 1], y[:, t:t + 1], model.tokenizer, model.UPD_TOKEN)
        _, M, _, _ = model.step(M, ids1[:, 0].to(device))
    return y, q


def vs_mwu_labels(p, j):
    n, T, E = p.shape
    y = np.zeros((n, T), dtype=np.int64)
    cum = np.zeros((n, E))
    for t in range(T):
        w = np.exp(cum - cum.max(-1, keepdims=True))
        w /= w.sum(-1, keepdims=True)
        q = (w * p[:, t]).sum(-1)
        y[:, t] = regret_adversary(q, p[np.arange(n), t, j])
        cum += np.where(y[:, t, None] == 1, np.log(p[:, t]), np.log(1 - p[:, t]))
    return y


def adversarial_eval(models, args, device, out_dir):
    rng = np.random.default_rng(args.seed + 1)
    n, T, E = args.adv_n_seq, args.adv_T, 4
    bound = float(np.log(E))
    # forecasts f drive every label rule and baseline; raw is what the model reads
    f_st, y_st, _, r_st = sample(n, T, E, rng)
    f_sw, _, _, r_sw = sample(n, T, E, rng)
    blocks = rng.integers(E, size=(n, -(-T // args.switch_every)))
    true_t = np.repeat(blocks, args.switch_every, axis=1)[:, :T]
    y_sw = (rng.random((n, T)) < f_sw[np.arange(n)[:, None], np.arange(T), true_t]).astype(np.int64)
    f_un, _, _, r_un = sample(n, T, E, rng)
    y_un = rng.integers(2, size=(n, T))
    f_mw, _, _, r_mw = sample(n, T, E, rng)
    y_mw = vs_mwu_labels(f_mw, rng.integers(E, size=n))
    f_vm, _, _, r_vm = sample(n, T, E, rng)
    j_vm = rng.integers(E, size=n)
    fixed = {'stochastic': (f_st, r_st, y_st), 'switching': (f_sw, r_sw, y_sw),
             'unrelated': (f_un, r_un, y_un), 'vs_mwu': (f_mw, r_mw, y_mw)}

    out = {'bound_ln_n': bound, 'T': T, 'n_seq': n, 'switch_every': args.switch_every,
           'scenarios': {}}
    for scen in ['stochastic', 'switching', 'unrelated', 'vs_mwu', 'vs_model']:
        out['scenarios'][scen] = {}
        for label, model in models.items():
            if scen == 'vs_model':
                y, q = adaptive_vs_model(model, f_vm, r_vm, j_vm, device)
                p = f_vm
            else:
                p, raw, y = fixed[scen]
                q = model_probs(model, raw, y, device)
            preds = dict(baselines(p, y, T), model=q)
            row = {}
            for k, v in preds.items():
                rc = regret_curve(v, p, y)
                row[k] = {'final_regret': float(rc[:, -1].mean()),
                          'max_regret_over_t': float(rc.max(1).mean()),
                          # 0.01 tolerance: the adversary drives MWU(eta=1) to the (tight) bound
                          'frac_seq_above_bound': float((rc.max(1) > bound + 1e-2).mean())}
            out['scenarios'][scen][label] = row
            if scen != 'vs_model' and label != list(models)[0]:
                continue
            print(f"  [{scen:10s}] {label}: " + '  '.join(
                f"{k}={r['final_regret']:.2f} (max_t {r['max_regret_over_t']:.2f}, "
                f"{100 * r['frac_seq_above_bound']:.0f}% > ln4)" for k, r in row.items()), flush=True)

    # figure: final regret per scenario, model(s) vs MWU(eta=1) and tuned MW, ln n bound
    scen = list(out['scenarios'])
    names = [(l, 'model', BLUE, f'model ({l})') for l in models]
    names += [(list(models)[0], 'mw1', INK2, r'MWU, $\eta=1$'),
              (list(models)[0], 'mw_tuned', MUTED, r'MW, tuned $\eta$')]
    fig, ax = plt.subplots(figsize=(5.0, 2.5), constrained_layout=True)
    w = 0.8 / len(names)
    for j, (l, k, c, lab) in enumerate(names):
        vals = [out['scenarios'][sc][l][k]['max_regret_over_t'] for sc in scen]
        ax.bar(np.arange(len(scen)) + (j - (len(names) - 1) / 2) * w, vals, width=w * 0.92,
               color=c, alpha=1.0 if k == 'model' and j == 0 else 0.75, label=lab, zorder=3)
    ax.axhline(bound, color=INK, lw=1, ls=(0, (4, 2)))
    ax.text(len(scen) - 0.5, bound, r'  $\ln 4$ bound', va='bottom', ha='right', fontsize=8)
    ax.set_xticks(range(len(scen)))
    ax.set_xticklabels([s.replace('_', ' ') for s in scen])
    ax.set_ylabel('max$_t$ log-loss regret')
    ax.set_title(f'Adversarial sequences (T={T})', loc='left', fontsize=9.5)
    ax.legend(frameon=False, fontsize=7.5, ncol=2)
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(out_dir, f'adversarial_regret.{ext}'), dpi=300)
    plt.close(fig)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpts', nargs='+',
                    default=[str(paths.checkpoint('theorem_matched', s)) for s in paths.SEEDS])
    ap.add_argument('--labels', nargs='+', default=[f'seed{s}' for s in paths.SEEDS])
    ap.add_argument('--n_seq', type=int, default=600)
    ap.add_argument('--T', type=int, default=500)
    ap.add_argument('--T_train', type=int, default=200)
    ap.add_argument('--seed', type=int, default=2027)
    ap.add_argument('--out_dir', default=str(paths.FIGURES / 'continuous_residual' / 'theorem_matched'))
    ap.add_argument('--adv_T', type=int, default=500)
    ap.add_argument('--adv_n_seq', type=int, default=300)
    ap.add_argument('--switch_every', type=int, default=25)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # evaluate on the forecast spreads the (first) model was trained on
    ck_args = torch.load(args.ckpts[0], map_location='cpu', weights_only=False).get('args', {})
    global DATA_MODEL
    DATA_MODEL = ck_args.get('data_model', 'latent_expert')
    print(f'data model: {DATA_MODEL}')
    if DATA_MODEL == 'noisy_expert':
        print(f'label noise eps = {train.NOISY_EPS} -> weighted majority with eta = '
              f'{np.log((1 - train.NOISY_EPS) / train.NOISY_EPS):.3f} is Bayes')
    if ck_args.get('latent_sigmas'):
        train.LATENT_SIGMAS = tuple(ck_args['latent_sigmas'])
    print(f'forecast spreads sigma in {train.LATENT_SIGMAS}')
    args.sigmas = list(train.LATENT_SIGMAS)

    rng = np.random.default_rng(args.seed)
    p, y, true, raw = sample(args.n_seq, args.T, 4, rng)
    n, T, E = p.shape
    ll = np.where(y[..., None] == 1, np.log(p), np.log(1 - p))           # per-expert log-lik
    cum = np.concatenate([np.zeros((n, 1, E)), np.cumsum(ll, 1)[:, :-1]], 1)  # before round t
    preds = {'mw1': mixture(p, cum, 1.0)[0],
             'mw_tuned': mixture(p, cum, np.sqrt(np.log(E) / T))[0],
             'uniform': p.mean(-1),
             'ftl': np.take_along_axis(p, cum.argmax(-1)[..., None], -1)[..., 0],
             'oracle': p[np.arange(n), :, true]}
    best_hindsight = (-ll).sum(1).min(-1)                               # [n]

    results = {'args': vars(args), 'models': {}}
    curves = {}
    loaded = {}
    for path, label in zip(args.ckpts, args.labels):
        model, _ = load_recurrent(path, device)
        loaded[label] = model
        tok = model.tokenizer
        ids, _ = encode(raw, y, tok, model.UPD_TOKEN)
        Ms, Zs = [], []
        with torch.no_grad():
            for i in range(0, n, 150):
                z, M = model.rollout(ids[i:i + 150].to(device), return_latents=True)
                Zs.append(torch.sigmoid(z).cpu().numpy())
                Ms.append(M.cpu().numpy())
        q, M = np.concatenate(Zs), np.concatenate(Ms)
        allp = dict(preds, model=q)

        res = {'buckets': {}}
        for a, b in BUCKETS:
            sl = slice(a - 1, b)
            res['buckets'][f'{a}-{b}'] = {
                'abs_diff_vs_mw1': float(np.abs(q[:, sl] - preds['mw1'][:, sl]).mean()),
                'abs_diff_mw_tuned_vs_mw1': float(np.abs(preds['mw_tuned'][:, sl]
                                                        - preds['mw1'][:, sl]).mean()),
                'logloss': {k: float(logloss(v[:, sl], y[:, sl]).mean()) for k, v in allp.items()},
            }
        # cumulative log-loss regret vs best expert in hindsight (whole sequence)
        res['regret_T'] = {k: float((logloss(v, y).sum(1) - best_hindsight).mean())
                           for k, v in allp.items()}
        for Tc in (args.T_train, T):
            bh = (-ll[:, :Tc]).sum(1).min(-1)
            res[f'regret_{Tc}'] = {k: float((logloss(v[:, :Tc], y[:, :Tc]).sum(1) - bh).mean())
                                   for k, v in allp.items()}
        # probe the latent (after round t) for MWU's log-weights after round t
        after = centred(np.cumsum(ll, 1))                                # [n, T, E]
        ntr = n // 2
        fit = slice(4, args.T_train)
        f = ridge(M[:ntr, fit].reshape(-1, M.shape[-1]), after[:ntr, fit].reshape(-1, E))
        res['probe_r2'] = {}
        for a, b in BUCKETS:
            sl = slice(max(a, 5) - 1, b)
            res['probe_r2'][f'{a}-{b}'] = r2(f(M[ntr:, sl].reshape(-1, M.shape[-1])),
                                             after[ntr:, sl].reshape(-1, E))
        # update rule on the decoded state
        s_hat = f(M[ntr:].reshape(-1, M.shape[-1])).reshape(n - ntr, T, E)
        inc = centred(ll[ntr:])
        for name, sl in (('in_horizon', slice(5, args.T_train)), ('beyond', slice(args.T_train, T))):
            prev = s_hat[:, sl.start - 1:sl.stop - 1].reshape(-1)
            cur = s_hat[:, sl].reshape(-1)
            X = np.stack([prev, inc[:, sl].reshape(-1), np.ones_like(prev)], 1)
            coef, *_ = np.linalg.lstsq(X, cur, rcond=None)
            res[f'update_rule_{name}'] = {'rho': float(coef[0]), 'k': float(coef[1])}
        results['models'][label] = res
        curves[label] = {'absdiff': np.abs(q - preds['mw1']).mean(0)}
        curves[label]['q_sample'] = (q[:, 50:200].ravel(), preds['mw1'][:, 50:200].ravel())

        b = res['buckets']
        print(f'\n===== {label} ({path})')
        print('  |p_model - p_MWU(eta=1)| by rounds: ' +
              '  '.join(f"{k}: {v['abs_diff_vs_mw1']:.4f}" for k, v in b.items()))
        print('  (for scale, |p_MW(tuned eta) - p_MWU(eta=1)|: ' +
              '  '.join(f"{k}: {v['abs_diff_mw_tuned_vs_mw1']:.4f}" for k, v in b.items()) + ')')
        print('  log-loss by rounds:')
        for k in allp:
            print(f'    {k:9s} ' + '  '.join(f"{bk}: {bv['logloss'][k]:.4f}" for bk, bv in b.items()))
        for Tc in (args.T_train, T):
            print(f'  log-loss regret vs best expert at T={Tc}: ' +
                  '  '.join(f'{k}={v:.2f}' for k, v in res[f"regret_{Tc}"].items()))
        print('  probe R^2 latent -> MWU log-weights: ' +
              '  '.join(f'{k}: {v:.3f}' for k, v in res['probe_r2'].items()))
        print(f"  update rule: in-horizon rho={res['update_rule_in_horizon']['rho']:.3f} "
              f"k={res['update_rule_in_horizon']['k']:.3f} | beyond rho="
              f"{res['update_rule_beyond']['rho']:.3f} k={res['update_rule_beyond']['k']:.3f}")

    print('\nadversarial tests (log-loss regret vs best expert; MWU(eta=1) is <= ln 4 = '
          f'{np.log(4):.3f} on every sequence):')
    results['adversarial'] = adversarial_eval(loaded, args, device, args.out_dir)
    with open(os.path.join(args.out_dir, 'results.json'), 'w') as fjs:
        json.dump(results, fjs, indent=1)

    # figures: (1) distance to MWU over rounds, (2) model vs MWU scatter
    t = np.arange(1, T + 1)
    fig, ax = plt.subplots(figsize=(3.6, 2.5), constrained_layout=True)
    for i, (label, c) in enumerate(curves.items()):
        ax.plot(t, c['absdiff'], color=BLUE, lw=1.4, ls='-' if i == 0 else (0, (3, 1.5)),
                label=f'model ({label})')
    ax.plot(t, np.abs(preds['mw_tuned'] - preds['mw1']).mean(0), color=INK2, lw=1.1,
            ls=(0, (1.5, 1.5)), label=r'MW, tuned $\eta$')
    ax.plot(t, np.abs(preds['uniform'] - preds['mw1']).mean(0), color=MUTED, lw=1.1,
            ls=(0, (4, 2)), label='uniform mixture')
    ax.axvline(args.T_train, color=GRID, lw=1)
    ax.set_xscale('log')
    ax.set_xlabel('round $t$')
    ax.set_ylabel(r'$|\hat p_t - p_t^{\mathrm{MWU}(\eta=1)}|$')
    ax.set_title('Distance to MWU ($\\eta=1$)', loc='left', fontsize=9.5)
    ax.legend(frameon=False, fontsize=7.5)
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(args.out_dir, f'distance_to_mwu.{ext}'), dpi=300)
    plt.close(fig)

    label0 = args.labels[0]
    qs, ms = curves[label0]['q_sample']
    idx = np.random.default_rng(0).choice(len(qs), size=min(6000, len(qs)), replace=False)
    fig, ax = plt.subplots(figsize=(2.8, 2.6), constrained_layout=True)
    ax.scatter(ms[idx], qs[idx], s=2, alpha=0.3, color=BLUE, linewidths=0)
    ax.plot([0, 1], [0, 1], color=INK2, lw=0.8, ls=(0, (3, 2)))
    ax.set_xlabel(r'MWU ($\eta=1$) prediction')
    ax.set_ylabel('model prediction')
    ax.set_title(f'Rounds 51–200 ({label0})', loc='left', fontsize=9.5)
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(args.out_dir, f'model_vs_mwu_scatter.{ext}'), dpi=300)
    plt.close(fig)
    print(f"\nwrote {args.out_dir}")


if __name__ == '__main__':
    main()
