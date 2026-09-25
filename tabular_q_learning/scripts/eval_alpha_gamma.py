#!/usr/bin/env python3
"""
eval_alpha_gamma.py — Behavioural fit of the model's effective (alpha, gamma).

4_evaluate.py's effective_alpha_gamma regresses probe-decoded Q changes on TD terms;
with a noisy probe and one-entry-per-step changes that fit has R^2 ~ 0, so its
alpha/gamma are not interpretable. This script instead replays tabular Q-learners
over an (alpha, gamma) grid on the same teacher-forced histories and scores how
often each learner's greedy action at s' matches the model's choice:
  - on all non-tie steps, and
  - on "clean" steps, where copying the last a* revealed for s' would not give that
    learner's label (so agreement cannot come from the a* input token).
The best-matching (alpha, gamma) is the model's effective update rule; the
teacher's own agreement surface is reported for reference (it peaks at the data's
alpha and gamma = 0.9 by construction).

  python3 eval_alpha_gamma.py --checkpoints a.pt b.pt \
      --data ../data/qlv3_dataset.pt --out ../figures/comparison/alphagamma_qlv3
"""
import argparse
import json
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import paths  # noqa: E402

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from eval_clean_steps import M, model_choices

ALPHAS = np.array([0.05, 0.1, 0.2, 0.3, 0.5, 0.7])
GAMMAS = np.array([0.0, 0.5, 0.7, 0.8, 0.9, 0.95])


def grid_labels(seq):
    """Greedy action at s' for every (alpha, gamma) learner, after each step's update.
    Returns labels [T, nA_grid, nG_grid] (-1 on ties) and last-a* labels [T]."""
    nS, nA = seq['n_states'], seq['n_actions']
    Q = np.zeros((len(ALPHAS), len(GAMMAS), nS, nA))
    al = ALPHAS[:, None]
    ga = GAMMAS[None, :]
    last_astar = np.full(nS, -1)
    labs, lasts = [], []
    for tr in seq['transitions']:
        s, a, r, s2 = tr['s'], tr['a'], tr['r'], tr['s_next']
        target = r + ga * Q[:, :, s2].max(-1)
        Q[:, :, s, a] = (1 - al) * Q[:, :, s, a] + al * target
        v = Q[:, :, s2]                                        # [nA_grid, nG_grid, nA]
        best = v.max(-1, keepdims=True)
        uniq = (v == best).sum(-1) == 1
        labs.append(np.where(uniq, v.argmax(-1), -1))
        lasts.append(last_astar[s2])
        last_astar[s2] = tr['a_star']
    return np.stack(labs), np.array(lasts)


def plot(results, out):
    surfaces = {n: np.array(v['clean']) for n, v in results['models'].items()}
    surfaces['teacher'] = np.array(results['teacher']['clean'])
    names = list(surfaces)
    fig, axes = plt.subplots(1, len(names), figsize=(2.9 * len(names) + 0.6, 2.6), constrained_layout=True)
    for ax, n in zip(np.atleast_1d(axes), names):
        z = surfaces[n]
        im = ax.imshow(z, cmap='Blues', aspect='auto', origin='lower', vmin=0, vmax=1)
        ia, ig = np.unravel_index(z.argmax(), z.shape)
        ax.scatter([ig], [ia], marker='*', s=90, color='#eb6834', zorder=3)
        ax.text(ig, ia + 0.42, f'{z.max():.2f}', ha='center', va='bottom', fontsize=7.5)
        ax.set_xticks(range(len(GAMMAS)), [f'{g:g}' for g in GAMMAS])
        ax.set_yticks(range(len(ALPHAS)), [f'{a:g}' for a in ALPHAS])
        ax.set_xlabel(r'$\gamma$')
        ax.set_ylabel(r'$\alpha$')
        ax.set_title(f'{n}: clean-step agreement', loc='left', fontsize=9)
    fig.colorbar(im, ax=np.atleast_1d(axes).tolist(), shrink=0.85, label='agreement')
    fig.savefig(out + '.png', dpi=250)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoints', nargs='+', default=[str(paths.checkpoint(m)) for m in paths.MODELS])
    ap.add_argument('--data', default=str(paths.DATASET))
    ap.add_argument('--split', default='val')
    ap.add_argument('--max_seqs', type=int, default=2000)
    ap.add_argument('--out', default=str(paths.FIGURES / 'comparison' / 'behavioural_fit' / 'alpha_gamma'),
                    help='output prefix (.json and .png)')
    args = ap.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    d = torch.load(args.data, map_location='cpu', weights_only=False)
    seqs = d[args.split][:args.max_seqs]
    print(f"data {os.path.basename(args.data)}:{args.split}, teacher alpha={d['config']['alpha']}, "
          f"gamma={d['config']['gamma']}")

    labels = [grid_labels(s) for s in seqs]
    teacher = [np.array([t['a_star'] for t in s['transitions']]) for s in seqs]
    results = {'alphas': ALPHAS.tolist(), 'gammas': GAMMAS.tolist(), 'models': {}}

    def surface(choices):
        hit = np.zeros((len(ALPHAS), len(GAMMAS)))
        tot = np.zeros_like(hit)
        chit, ctot = np.zeros_like(hit), np.zeros_like(hit)
        for (lab, last), ch in zip(labels, choices):
            valid = lab >= 0
            match = (lab == ch[:, None, None]) & valid
            hit += match.sum(0)
            tot += valid.sum(0)
            clean = valid & (lab != last[:, None, None])     # copying last a* would not give it
            chit += (match & clean).sum(0)
            ctot += clean.sum(0)
        return hit / tot, chit / np.maximum(ctot, 1)

    t_all, t_clean = surface(teacher)
    results['teacher'] = {'all': t_all.tolist(), 'clean': t_clean.tolist()}
    surfaces = {'teacher': (t_all, t_clean)}
    for ck in args.checkpoints:
        c = torch.load(ck, map_location='cpu', weights_only=False)
        model = M.COCONUTTransformer(M.COCONUTConfig.from_dict(c['config']))
        model.load_state_dict(c['model_state_dict'])
        model.to(device).eval()
        vocab = M.build_vocab(model.config.max_states, model.config.max_actions)
        choices = [None] * len(seqs)
        groups = {}
        for i, s in enumerate(seqs):
            groups.setdefault(s['n_actions'], []).append(i)
        for idx in groups.values():
            for b in range(0, len(idx), 256):
                part = idx[b:b + 256]
                preds = model_choices(model, [seqs[i] for i in part], vocab,
                                      model.config.max_actions, device)
                for i, p in zip(part, preds):
                    choices[i] = np.array(p)
        s_all, s_clean = surface(choices)
        name = os.path.basename(ck).replace('coconut_transformer_', '').replace('.pt', '')
        surfaces[name] = (s_all, s_clean)
        ia, ig = np.unravel_index(s_all.argmax(), s_all.shape)
        ca, cg = np.unravel_index(s_clean.argmax(), s_clean.shape)
        results['models'][name] = {
            'all': s_all.tolist(), 'clean': s_clean.tolist(),
            'best_all': {'alpha': float(ALPHAS[ia]), 'gamma': float(GAMMAS[ig]),
                         'agree': float(s_all[ia, ig])},
            'best_clean': {'alpha': float(ALPHAS[ca]), 'gamma': float(GAMMAS[cg]),
                           'agree': float(s_clean[ca, cg])}}
        print(f'\n=== {name}')
        print(f'  best fit, all steps:   alpha={ALPHAS[ia]:g} gamma={GAMMAS[ig]:g} '
              f'(agreement {s_all[ia, ig]:.3f})')
        print(f'  best fit, clean steps: alpha={ALPHAS[ca]:g} gamma={GAMMAS[cg]:g} '
              f'(agreement {s_clean[ca, cg]:.3f})')
        print('  clean-step agreement (rows alpha, cols gamma ' +
              ' '.join(f'{g:g}' for g in GAMMAS) + '):')
        for i, a in enumerate(ALPHAS):
            print(f'    alpha={a:<5g} ' + ' '.join(f'{v:.3f}' for v in s_clean[i]))

    with open(args.out + '.json', 'w') as f:
        json.dump(results, f, indent=1)

    names = [n for n in surfaces if n != 'teacher'] + ['teacher']
    fig, axes = plt.subplots(1, len(names), figsize=(2.9 * len(names) + 0.6, 2.6), constrained_layout=True)
    for ax, n in zip(np.atleast_1d(axes), names):
        z = surfaces[n][1]
        # one shared scale so panels are comparable (the teacher peaks at 1 by construction)
        im = ax.imshow(z, cmap='Blues', aspect='auto', origin='lower', vmin=0, vmax=1)
        ia, ig = np.unravel_index(z.argmax(), z.shape)
        ax.scatter([ig], [ia], marker='*', s=90, color='#eb6834', zorder=3)
        ax.set_xticks(range(len(GAMMAS)), [f'{g:g}' for g in GAMMAS])
        ax.set_yticks(range(len(ALPHAS)), [f'{a:g}' for a in ALPHAS])
        ax.set_xlabel(r'$\gamma$')
        ax.set_ylabel(r'$\alpha$')
        ax.set_title(f'{n}: clean-step agreement', loc='left', fontsize=9)
        best = z.max()
        ax.text(ig, ia + 0.42, f'{best:.2f}', ha='center', va='bottom', fontsize=7.5, color='#0b0b0b')
    fig.colorbar(im, ax=np.atleast_1d(axes).tolist(), shrink=0.85, label='agreement')
    fig.savefig(args.out + '.png', dpi=250)
    print(f'\nwrote {args.out}.json / .png')


if __name__ == '__main__':
    main()
