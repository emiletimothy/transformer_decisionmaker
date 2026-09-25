#!/usr/bin/env python3
"""
eval_memory_fit.py — How much history does the model's choice reflect?

Teacher-forced as in training. For every step, compares the model's greedy action
with the greedy action of tabular Q-learners (gamma 0.9) using different step sizes
alpha (small alpha = long memory; alpha = 1 = only the last visit counts) and with a
"highest last-seen reward" rule. The best-matching alpha is the model's effective
memory; the teacher's own agreement with each rule is printed for reference.

  python3 eval_memory_fit.py --checkpoints a.pt b.pt --data ../data/qlv3_dataset.pt
"""
import argparse
import csv
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import paths  # noqa: E402

import numpy as np
import torch

from eval_clean_steps import M, model_choices

ALPHAS = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]


def rule_labels(seq, gamma=0.9):
    nS, nA = seq['n_states'], seq['n_actions']
    Q = {a: np.zeros((nS, nA), np.float32) for a in ALPHAS}
    last = np.full((nS, nA), np.nan)
    out = []
    for tr in seq['transitions']:
        s, a, r, s2 = tr['s'], tr['a'], tr['r'], tr['s_next']
        row = {'teacher': tr['a_star']}
        for al, q in Q.items():
            q[s, a] = (1 - al) * q[s, a] + al * (r + gamma * q[s2].max())
            v = q[s2]
            row[f'Q alpha={al:g}'] = int(v.argmax()) if (v == v.max()).sum() == 1 else None
        last[s, a] = r
        v = last[s2]
        row['last reward'] = (int(np.nanargmax(v)) if np.isfinite(v).any()
                              and (v == np.nanmax(v)).sum() == 1 else None)
        out.append(row)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoints', nargs='+', default=[str(paths.checkpoint(m)) for m in paths.MODELS])
    ap.add_argument('--data', default=str(paths.DATASET))
    ap.add_argument('--split', default='val')
    ap.add_argument('--max_seqs', type=int, default=3000)
    ap.add_argument('--out', default=str(paths.FIGURES / 'comparison' / 'behavioural_fit' / 'memory_fit.csv'))
    args = ap.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    d = torch.load(args.data, map_location='cpu', weights_only=False)
    seqs = d[args.split][:args.max_seqs]
    print(f"data {os.path.basename(args.data)}:{args.split}, teacher alpha={d['config']['alpha']}")
    rows = []
    for ck in args.checkpoints:
        c = torch.load(ck, map_location='cpu', weights_only=False)
        model = M.COCONUTTransformer(M.COCONUTConfig.from_dict(c['config']))
        model.load_state_dict(c['model_state_dict'])
        model.to(device).eval()
        vocab = M.build_vocab(model.config.max_states, model.config.max_actions)
        groups = {}
        for s in seqs:
            groups.setdefault(s['n_actions'], []).append(s)
        agree, teach = {}, {}
        for grp in groups.values():
            for b in range(0, len(grp), 256):
                eps = grp[b:b + 256]
                preds = model_choices(model, eps, vocab, model.config.max_actions, device)
                for sq, pr in zip(eps, preds):
                    for row, p in zip(rule_labels(sq), pr):
                        for k, v in row.items():
                            if k == 'teacher' or v is None:
                                continue
                            agree.setdefault(k, []).append(p == v)
                            teach.setdefault(k, []).append(row['teacher'] == v)
        name = os.path.basename(ck)
        best = max((k for k in agree if k.startswith('Q')), key=lambda k: np.mean(agree[k]))
        print(f'\n=== {name}  (best-matching memory: {best})')
        for k in agree:
            print(f'  {k:14s} model {np.mean(agree[k]):.3f}   teacher {np.mean(teach[k]):.3f}   '
                  f'n={len(agree[k])}')
            rows.append({'checkpoint': name, 'rule': k, 'agree_model': np.mean(agree[k]),
                         'agree_teacher': np.mean(teach[k]), 'n': len(agree[k])})
    if args.out:
        with open(args.out, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f'\nwrote {args.out}')


if __name__ == '__main__':
    main()
