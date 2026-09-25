#!/usr/bin/env python3
"""
data_shortcuts.py — Can the training/eval labels be predicted WITHOUT
Q-learning? For every step of a dataset, compare the teacher's label
a* = argmax Q_{gamma=0.9}(s', .) with labels from simpler rules that see the same
history: a myopic learner (gamma = 0, same alpha), a reward-blind "most-tried
action at s'" rule, and greedy-on-last-reward. Reports each rule's agreement
with a* and the share of steps where a* DISAGREES with the rule — the only
steps on which the training loss rewards bootstrapped Q-learning over the rule.

  python3 data_shortcuts.py --data ../data/coconut_dataset.pt
"""
import argparse
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import paths  # noqa: E402

import numpy as np
import torch


def replay_labels(seq, rng, alpha=0.1):
    nS, nA = seq['n_states'], seq['n_actions']
    Q0 = np.zeros((nS, nA))
    tried = np.zeros((nS, nA))
    lastr = np.full((nS, nA), -1.0)
    out = []

    def pick(v):
        c = np.flatnonzero(v == v.max())
        return int(rng.choice(c))

    for t, tr in enumerate(seq['transitions']):
        s, a, r, s2 = tr['s'], tr['a'], tr['r'], tr['s_next']
        Q0[s, a] = (1 - alpha) * Q0[s, a] + alpha * r
        tried[s, a] += 1
        lastr[s, a] = r
        q = np.asarray(seq['q_snapshots'][t])[s2]
        out.append({
            'teacher': tr['a_star'],
            'teacher_tie': int((q == q.max()).sum() > 1),
            'myopic_gamma0': pick(Q0[s2]),
            'most_tried': pick(tried[s2]),
            'last_reward': pick(lastr[s2]),
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=str(paths.ORIGINAL_DATASET),
                    help='default: the submitted paper\'s dataset; the final data is paths.DATASET')
    ap.add_argument('--max_seqs', type=int, default=20000)
    args = ap.parse_args()
    d = torch.load(args.data, map_location='cpu', weights_only=False)
    alpha = float(d.get('config', {}).get('alpha', 0.1)) if isinstance(d, dict) else 0.1
    seqs = d['sequences'] if isinstance(d, dict) and 'sequences' in d else d
    if isinstance(seqs, dict):
        print('keys:', list(seqs.keys()))
        seqs = seqs.get('train', next(iter(seqs.values())))
    print(f'{len(seqs)} sequences; using {min(len(seqs), args.max_seqs)}; alpha={alpha}')
    rng = np.random.default_rng(0)
    rules = ['myopic_gamma0', 'most_tried', 'last_reward']
    groups = defaultdict(lambda: defaultdict(list))
    for seq in seqs[:args.max_seqs]:
        for row in replay_labels(seq, rng, alpha=alpha):
            if row['teacher_tie']:
                continue  # label is a random tie-break; uninformative
            for key in ['ALL', f"task={seq.get('task', 'orig')}",
                        f"reward={seq['reward_dist']}", f"trans={seq['trans_conc']}"]:
                for r in rules:
                    groups[key][r].append(row[r] == row['teacher'])
                groups[key]['all_three_agree'].append(
                    all(row[r] == row['teacher'] for r in rules[:2]))
    print('\nagreement of each rule with the teacher label (non-tie steps); '
          'disagreement share = 1 - agreement')
    print(f"{'group':22s} {'n':>8s} " + ' '.join(f'{r:>14s}' for r in rules) +
          f" {'myopic&tried':>13s}")
    for key in sorted(groups, key=lambda k: (k != 'ALL', k)):
        g = groups[key]
        print(f"{key:22s} {len(g[rules[0]]):8d} " +
              ' '.join(f'{np.mean(g[r]):14.3f}' for r in rules) +
              f" {np.mean(g['all_three_agree']):13.3f}")


if __name__ == '__main__':
    main()
