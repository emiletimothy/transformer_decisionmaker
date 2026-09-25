#!/usr/bin/env python3
"""
eval_drift.py — Does closed-loop performance decay past the training horizon?

Two checks on the 1000-step contrast-MDP closed loop (model's own a*, as in
eval_closed_loop.py):
  1. reward rate per window (1-100, 101-200, 201-500, 501-1000) for the model,
     greedy / eps-greedy tabular Q and random, from the saved
     long_horizon_rewards.npz of each reviewer suite;
  2. the norm of every value written into a context slot over the rollout,
     recorded by re-running the same rollout with a hook on contextualize().
If the model is fine inside the training horizon (<= 200 steps) and degrades after,
while its slot norms keep growing, the closed-loop weakness is drift of the
accumulated state rather than a wrong update.

  python3 eval_drift.py                     # both continuous models, from paths.py
  python3 eval_drift.py --runs label:closed_loop_dir:checkpoint ...
"""
import argparse
import importlib.util
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
import paths  # noqa: E402
_spec = importlib.util.spec_from_file_location('rv2', os.path.join(_HERE, 'eval_closed_loop.py'))
RV = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(RV)
E = RV.E

WINDOWS = [(1, 100), (101, 200), (201, 500), (501, 1000)]
AGENTS = ['tf_self', 'greedy', 'epsgreedy', 'random', 'optimal']


def window_rates(arr):
    return {f'{a}-{b}': float(arr[:, a - 1:b].mean()) for a, b in WINDOWS if b <= arr.shape[1]}


@torch.no_grad()
def slot_norms(ckpt, n_mdps, T, seed0, device):
    model, config = RV.load_model(ckpt, device)
    vocab = E.build_vocab(config.max_states, config.max_actions)
    ns, na = config.max_states, config.max_actions
    seeds = list(range(seed0, seed0 + n_mdps))
    phases = [[(*E.generate_contrast_mdp_seeded(ns, na, seed=sd), T)] for sd in seeds]
    norms = []
    orig = model.contextualize

    def hooked(upd, *a, **k):
        out = orig(upd, *a, **k)
        norms.append(out.norm(dim=-1).mean().item())
        return out
    model.contextualize = hooked
    RV.run_tf_batched(model, config, vocab, phases, ns, na, device, seeds, astar_mode='self')
    # run_tf_batched writes twice per step in 'self' mode only once to the context;
    # keep one entry per step
    norms = np.array(norms)
    if len(norms) >= 2 * T:
        norms = norms[1::2]
    return norms[:T]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', nargs='+',
                    default=[f"{m.split('_')[-1]}:{paths.FIGURES / m / 'closed_loop'}:{paths.checkpoint(m)}"
                             for m in ('continuous_residual', 'continuous_overwrite')],
                    help='label:closed_loop_dir:checkpoint (the closed-loop suite of eval_closed_loop.py)')
    ap.add_argument('--family', default='contrast')
    ap.add_argument('--n_mdps', type=int, default=20)
    ap.add_argument('--T', type=int, default=1000)
    ap.add_argument('--out', default=str(paths.FIGURES / 'comparison' / 'drift' / 'drift'), help='output prefix')
    args = ap.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    res = {}
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6), constrained_layout=True)
    colors = {'residual': '#2a78d6', 'overwrite': '#9a9994', 'discrete': '#eb6834'}
    for spec in args.runs:
        label, suite, ckpt = spec.split(':')
        z = np.load(os.path.join(suite, 'long_horizon_rewards.npz'))
        rates = {ag: window_rates(z[f'{args.family}__{ag}']) for ag in AGENTS
                 if f'{args.family}__{ag}' in z}
        norms = slot_norms(ckpt, args.n_mdps, args.T, 9999, device)
        res[label] = {'reward_rate': rates,
                      'slot_norm': {f'{a}-{b}': float(norms[a - 1:b].mean()) for a, b in WINDOWS}}
        print(f'\n=== {label}')
        for ag, r in rates.items():
            print(f'  {ag:10s} reward rate ' + '  '.join(f'{k}: {v:.3f}' for k, v in r.items()))
        print('  slot norm  ' + '  '.join(f'{k}: {v:.2f}' for k, v in res[label]['slot_norm'].items()))
        c = colors.get(label, '#0b0b0b')
        tf = z[f'{args.family}__tf_self']
        k = 50
        roll = np.convolve(tf.mean(0), np.ones(k) / k, mode='valid')
        axes[0].plot(np.arange(k, tf.shape[1] + 1), roll, color=c, lw=1.4, label=label)
        axes[1].plot(np.arange(1, len(norms) + 1), norms, color=c, lw=1.4, label=label)
    for ag, ls in (('greedy', (0, (1.5, 1.5))), ('random', (0, (4, 2)))):
        arr = z[f'{args.family}__{ag}']
        roll = np.convolve(arr.mean(0), np.ones(50) / 50, mode='valid')
        axes[0].plot(np.arange(50, arr.shape[1] + 1), roll, color='#52514e', lw=1, ls=ls,
                     label=f'{ag} Q' if ag == 'greedy' else ag)
    for ax in axes:
        ax.axvline(200, color='#e4e3df', lw=1)
        ax.set_xlabel('step')
    axes[0].set_ylabel('reward per step (50-step mean)')
    axes[0].set_title('Closed-loop reward (own $a^*$)', loc='left', fontsize=9.5)
    axes[1].set_ylabel('mean norm of written slot')
    axes[1].set_title('Context slot norm', loc='left', fontsize=9.5)
    axes[0].legend(frameon=False, fontsize=7.5)
    fig.savefig(args.out + '.png', dpi=250)
    with open(args.out + '.json', 'w') as f:
        json.dump(res, f, indent=1)
    print(f'\nwrote {args.out}.png / .json')


if __name__ == '__main__':
    main()
