#!/usr/bin/env python3
"""
inspect_dataset.py — Inspect contents of coconut_dataset.pt

Usage:
    python inspect_dataset.py
    python inspect_dataset.py --path ../data/coconut_dataset.pt
    python inspect_dataset.py --split train --index 0 --full
    python inspect_dataset.py --stats
"""

import argparse
import os
from collections import Counter

import numpy as np
import torch


def print_header(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def show_config(config):
    print_header("CONFIG")
    for k, v in config.items():
        print(f"  {k:30s} {v}")


def show_split_overview(name, seqs):
    print_header(f"{name.upper()} SPLIT — {len(seqs):,} sequences")
    if not seqs:
        return
    s0 = seqs[0]
    print(f"  Keys per sequence: {list(s0.keys())}")
    print(f"  Example types:")
    for k, v in s0.items():
        if isinstance(v, list) and v and isinstance(v[0], np.ndarray):
            print(f"    {k:15s} list[np.ndarray] len={len(v)} elem_shape={v[0].shape} dtype={v[0].dtype}")
        elif isinstance(v, list):
            print(f"    {k:15s} list len={len(v)}  first={v[0] if v else None}")
        else:
            print(f"    {k:15s} {type(v).__name__}={v}")


def show_sequence(seqs, idx, full=False):
    print_header(f"SEQUENCE #{idx}")
    seq = seqs[idx]
    print(f"  n_steps   = {seq['n_steps']}")
    print(f"  n_states  = {seq['n_states']}")
    print(f"  n_actions = {seq['n_actions']}")
    print(f"  reward_dist = {seq.get('reward_dist')}")
    print(f"  trans_conc  = {seq.get('trans_conc')}")

    transitions = seq['transitions']
    n_show = len(transitions) if full else min(5, len(transitions))
    print(f"\n  Transitions (showing {n_show} of {len(transitions)}):")
    for t, tr in enumerate(transitions[:n_show]):
        print(f"    t={t:3d}  s={tr['s']}  a={tr['a']}  "
              f"r={tr['r']:.4f}  s_next={tr['s_next']}  a*={tr['a_star']}")

    q_snaps = seq['q_snapshots']
    if q_snaps:
        print(f"\n  Q-snapshots: {len(q_snaps)} arrays, shape={q_snaps[0].shape}")
        print(f"  First Q-table:\n{q_snaps[0]}")
        if full:
            print(f"  Final Q-table:\n{q_snaps[-1]}")


def show_stats(seqs, name):
    print_header(f"{name.upper()} STATISTICS")
    n_steps  = [s['n_steps']    for s in seqs]
    n_states = [s['n_states']   for s in seqs]
    n_acts   = [s['n_actions']  for s in seqs]
    rewards  = [tr['r'] for s in seqs for tr in s['transitions']]

    def stats(arr, label):
        a = np.array(arr)
        print(f"  {label:12s} min={a.min():.3f}  max={a.max():.3f}  "
              f"mean={a.mean():.3f}  std={a.std():.3f}")

    stats(n_steps,  'n_steps')
    stats(n_states, 'n_states')
    stats(n_acts,   'n_actions')
    stats(rewards,  'rewards')

    rd = Counter(s.get('reward_dist') for s in seqs)
    tc = Counter(s.get('trans_conc')  for s in seqs)
    print(f"\n  reward_dist counts: {dict(rd)}")
    print(f"  trans_conc  counts: {dict(tc)}")


def main():
    parser = argparse.ArgumentParser(description='Inspect coconut_dataset.pt')
    default_path = os.path.join(os.path.dirname(__file__),
                                '..', 'data', 'coconut_dataset.pt')
    parser.add_argument('--path',  type=str, default=default_path)
    parser.add_argument('--split', choices=['train', 'val'], default='train')
    parser.add_argument('--index', type=int, default=0,
                        help='Index of sequence to display in detail')
    parser.add_argument('--full',  action='store_true',
                        help='Show all transitions and final Q-table')
    parser.add_argument('--stats', action='store_true',
                        help='Print aggregate statistics over the split')
    args = parser.parse_args()

    print(f"Loading {args.path} ...")
    data = torch.load(args.path, weights_only=False)

    print(f"Top-level keys: {list(data.keys())}")
    show_config(data['config'])
    show_split_overview('train', data['train'])
    show_split_overview('val',   data['val'])

    seqs = data[args.split]
    if 0 <= args.index < len(seqs):
        show_sequence(seqs, args.index, full=args.full)

    if args.stats:
        show_stats(seqs, args.split)


if __name__ == '__main__':
    main()
