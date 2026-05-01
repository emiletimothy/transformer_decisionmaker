#!/usr/bin/env python3
"""
1_generate_data.py — Dataset Generator for Recurrent Context Q-Learning

Generates Q-learning trajectories as lists of independent transition tuples
per episode. The formatting into the exact token sequence happens in the
PyTorch Dataset/DataLoader collator (3_train.py).

MDPs are sampled with random sizes (n_states in [min_states, max_states],
n_actions in [min_actions, max_actions]) and diverse reward/transition
distributions to eliminate implicit biases in training.

Vocabulary layout (vocab_size = 2 + max_states + max_actions + 5):
    TOK_NULL   = 0
    TOK_START  = 1
    TOK_S      = [2, ..., 2 + max_states - 1]
    TOK_A      = [2 + max_states, ..., 2 + max_states + max_actions - 1]
    TOK_R      = 2 + max_states + max_actions
    TOK_EVAL   = TOK_R + 1
    TOK_SELECT = TOK_R + 2
    TOK_QCURR  = TOK_R + 3
    TOK_QNEXT  = TOK_R + 4
    TOK_UPDATE = TOK_R + 5

Per-step token sequence (formatted at collation time):
  Context tokens: c_1, c_2, ..., c_{|A|}  (continuous, prepended)
  Current step:   s_t, a_t, R, s_{t+1}
  Eval workspace: [s_{t+1}, a_c, EVAL] * |A|
  SELECT
  TD workspace:   QCURR, QNEXT, UPDATE

Output per episode:
    transitions  : List[dict]  — {s, a, r, s_next, a_star} per step
    n_steps      : int
    n_states     : int
    n_actions    : int
    q_snapshots  : List[np.ndarray]  — Q-table after each update
"""

import argparse
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Distribution registries for reward and transition diversity
# ---------------------------------------------------------------------------

REWARD_DISTRIBUTIONS: Dict[str, Optional[Tuple[float, float]]] = {
    'peaked':   (2.0, 2.0),
    'bimodal':  (0.5, 0.5),
    'uniform':  (1.0, 1.0),
    'sparse':   (0.2, 2.0),
    'dense':    (2.0, 0.2),
    'bernoulli': None,
}

TRANSITION_CONCENTRATIONS: Dict[str, float] = {
    'near_det': 0.1,
    'uniform':  1.0,
    'diffuse':  5.0,
}

_REWARD_DIST_NAMES  = list(REWARD_DISTRIBUTIONS.keys())
_TRANS_CONC_NAMES   = list(TRANSITION_CONCENTRATIONS.keys())


# ---------------------------------------------------------------------------
# Vocabulary builder
# ---------------------------------------------------------------------------

def build_vocab(max_states: int, max_actions: int) -> Dict[str, object]:
    TOK_NULL   = 0
    TOK_START  = 1
    TOK_S      = list(range(2, 2 + max_states))
    TOK_A      = list(range(2 + max_states, 2 + max_states + max_actions))
    TOK_R      = 2 + max_states + max_actions
    TOK_EVAL   = TOK_R + 1
    TOK_SELECT = TOK_R + 2
    TOK_QCURR  = TOK_R + 3
    TOK_QNEXT  = TOK_R + 4
    TOK_UPDATE = TOK_R + 5
    vocab_size = 2 + max_states + max_actions + 6
    return {
        'TOK_NULL':      TOK_NULL,
        'TOK_START':     TOK_START,
        'TOK_S':         TOK_S,
        'TOK_A':         TOK_A,
        'TOK_R':         TOK_R,
        'TOK_EVAL':      TOK_EVAL,
        'TOK_SELECT':    TOK_SELECT,
        'TOK_QCURR':     TOK_QCURR,
        'TOK_QNEXT':     TOK_QNEXT,
        'TOK_UPDATE':    TOK_UPDATE,
        'vocab_size':    vocab_size,
        'n_actions_max': max_actions,
    }


# ---------------------------------------------------------------------------
# MDP generation
# ---------------------------------------------------------------------------

def generate_random_mdp(
    n_states: int,
    n_actions: int,
    rng: np.random.Generator = None,
    reward_dist: str = 'peaked',
    trans_conc: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    if rng is None:
        rng = np.random.default_rng()

    P = rng.dirichlet(alpha=np.full(n_states, trans_conc), size=(n_states, n_actions))
    P = P.astype(np.float32)

    dist_params = REWARD_DISTRIBUTIONS[reward_dist]
    if dist_params is None:
        means = rng.uniform(0.1, 0.9, size=(n_states, n_actions))
        R = means.astype(np.float32)
    else:
        a_param, b_param = dist_params
        R = rng.beta(a_param, b_param, size=(n_states, n_actions)).astype(np.float32)
        R = np.clip(R, 0.0, 1.0)

    return P, R


# ---------------------------------------------------------------------------
# Episode generation
# ---------------------------------------------------------------------------

def generate_episode(
    P: np.ndarray,
    R: np.ndarray,
    n_states: int,
    n_actions: int,
    n_steps: int,
    alpha: float = 0.1,
    gamma: float = 0.9,
    epsilon: float = 0.2,
    q_init: float = 0.0,
    rng: np.random.Generator = None,
    stochastic_rewards: bool = False,
) -> Dict:
    if rng is None:
        rng = np.random.default_rng()

    Q = np.full((n_states, n_actions), q_init, dtype=np.float32)
    s = int(rng.integers(n_states))

    transitions  = []
    q_snapshots  = []

    for _ in range(n_steps):
        if rng.random() < epsilon:
            a = int(rng.integers(n_actions))
        else:
            best = float(np.max(Q[s]))
            ties = [ac for ac in range(n_actions) if Q[s, ac] == best]
            a = int(rng.choice(ties))

        if stochastic_rewards:
            r = float(rng.binomial(1, float(R[s, a])))
        else:
            r = float(R[s, a])
        s_next = int(rng.choice(n_states, p=P[s, a]))

        max_q_next = float(np.max(Q[s_next]))
        Q[s, a] = (1.0 - alpha) * Q[s, a] + alpha * (r + gamma * max_q_next)

        best_next = float(np.max(Q[s_next]))
        ties_next = [ac for ac in range(n_actions) if Q[s_next, ac] == best_next]
        a_star = int(rng.choice(ties_next))

        transitions.append({
            's': s, 'a': a, 'r': r, 's_next': s_next, 'a_star': a_star,
        })
        q_snapshots.append(Q.copy())
        s = s_next

    return {
        'transitions': transitions,
        'q_snapshots': q_snapshots,
    }


# ---------------------------------------------------------------------------
# Permutation invariance
# ---------------------------------------------------------------------------

def apply_permutations(
    episode: Dict,
    n_states: int,
    n_actions: int,
    rng: np.random.Generator,
) -> Dict:
    perm_s = rng.permutation(n_states)
    perm_a = rng.permutation(n_actions)
    inv_s = np.argsort(perm_s)
    inv_a = np.argsort(perm_a)

    new_transitions = []
    for t in episode['transitions']:
        new_transitions.append({
            's':      int(perm_s[t['s']]),
            'a':      int(perm_a[t['a']]),
            'r':      t['r'],
            's_next': int(perm_s[t['s_next']]),
            'a_star': int(perm_a[t['a_star']]),
        })

    new_q_snapshots = []
    for Q in episode['q_snapshots']:
        new_q_snapshots.append(Q[np.ix_(inv_s, inv_a)])

    return {
        'transitions': new_transitions,
        'q_snapshots': new_q_snapshots,
        'perm_s':      perm_s.tolist(),
        'perm_a':      perm_a.tolist(),
    }


# ---------------------------------------------------------------------------
# Dataset generation driver
# ---------------------------------------------------------------------------

def generate_dataset(
    n_sequences: int,
    max_states: int = 8,
    max_actions: int = 4,
    min_states: int = 2,
    min_actions: int = 2,
    min_steps: int = 10,
    max_steps: int = 50,
    alpha: float = 0.1,
    gamma: float = 0.9,
    seed: int = 42,
    stochastic_rewards: bool = False,
) -> List[Dict]:
    rng = np.random.default_rng(seed)
    vocab = build_vocab(max_states, max_actions)

    cells = [
        (rd, tc)
        for rd in _REWARD_DIST_NAMES
        for tc in _TRANS_CONC_NAMES
    ]
    n_cells = len(cells)
    base, remainder = divmod(n_sequences, n_cells)
    cell_assignments = []
    for idx, cell in enumerate(cells):
        count = base + (1 if idx < remainder else 0)
        cell_assignments.extend([cell] * count)
    shuffle_order = rng.permutation(len(cell_assignments))
    cell_assignments = [cell_assignments[i] for i in shuffle_order]

    sequences = []

    for i in range(n_sequences):
        n_steps  = int(rng.integers(min_steps, max_steps + 1))
        epsilon  = float(rng.uniform(0.0, 1.0))
        n_s      = int(rng.integers(min_states, max_states + 1))
        n_a      = int(rng.integers(min_actions, max_actions + 1))

        reward_dist_name, trans_conc_name = cell_assignments[i]
        trans_conc = TRANSITION_CONCENTRATIONS[trans_conc_name]

        P, R = generate_random_mdp(n_s, n_a, rng=rng,
                                   reward_dist=reward_dist_name,
                                   trans_conc=trans_conc)
        ep = generate_episode(
            P, R, n_s, n_a, n_steps,
            alpha=alpha, gamma=gamma, epsilon=epsilon, q_init=0.0,
            rng=rng, stochastic_rewards=stochastic_rewards,
        )
        ep = apply_permutations(ep, n_s, n_a, rng)

        seq = {
            'transitions':  ep['transitions'],
            'q_snapshots':  ep['q_snapshots'],
            'n_steps':      n_steps,
            'n_states':     n_s,
            'n_actions':    n_a,
            'reward_dist':  reward_dist_name,
            'trans_conc':   trans_conc_name,
        }
        sequences.append(seq)

        if (i + 1) % 5000 == 0:
            print(f"  generated {i + 1:,} / {n_sequences:,} sequences...")

    return sequences


# ---------------------------------------------------------------------------
# Sanity-check printer
# ---------------------------------------------------------------------------

def print_sample(seq: Dict, vocab: Dict) -> None:
    n = min(3, seq['n_steps'])
    print(f"\nSample episode (n_steps={seq['n_steps']}, "
          f"n_states={seq['n_states']}, n_actions={seq['n_actions']})")
    print(f"  First {n} transitions:")
    for t in range(n):
        tr = seq['transitions'][t]
        print(f"    t={t}: s={tr['s']} a={tr['a']} r={tr['r']:.4f} "
              f"s_next={tr['s_next']} a*={tr['a_star']}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Generate recurrent context Q-learning dataset'
    )
    parser.add_argument('--n_sequences',      type=int,   default=50_000)
    parser.add_argument('--max_states',       type=int,   default=8)
    parser.add_argument('--max_actions',      type=int,   default=4)
    parser.add_argument('--min_states',       type=int,   default=2)
    parser.add_argument('--min_actions',      type=int,   default=2)
    parser.add_argument('--n_states',         type=int,   default=None,
                        help='Deprecated: use --max_states')
    parser.add_argument('--n_actions',        type=int,   default=None,
                        help='Deprecated: use --max_actions')
    parser.add_argument('--min_steps',        type=int,   default=10)
    parser.add_argument('--max_steps',        type=int,   default=50)
    parser.add_argument('--alpha',            type=float, default=0.1)
    parser.add_argument('--gamma',            type=float, default=0.9)
    parser.add_argument('--seed',             type=int,   default=42)
    parser.add_argument('--stochastic_rewards', action='store_true',
                        help='Sample r_t ~ Bernoulli(R[s,a]) each step instead of using mean')
    parser.add_argument('--output',           type=str,
                        default=os.path.join(os.path.dirname(__file__),
                                             '..', 'data', 'coconut_dataset.pt'))
    args = parser.parse_args()

    if args.n_states is not None:
        args.max_states = args.n_states
    if args.n_actions is not None:
        args.max_actions = args.n_actions

    np.random.seed(args.seed)

    print(f"Generating {args.n_sequences:,} recurrent context Q-learning episodes")
    print(f"  states: [{args.min_states}, {args.max_states}]  "
          f"actions: [{args.min_actions}, {args.max_actions}]  "
          f"steps: [{args.min_steps}, {args.max_steps}]")
    print(f"  reward distributions: {_REWARD_DIST_NAMES}")
    print(f"  transition concentrations: {_TRANS_CONC_NAMES}")
    print(f"  stochastic_rewards: {args.stochastic_rewards}")

    vocab = build_vocab(args.max_states, args.max_actions)
    print(f"\nVocabulary (vocab_size={vocab['vocab_size']}):")
    print(f"  NULL={vocab['TOK_NULL']}  START={vocab['TOK_START']}")
    print(f"  S={vocab['TOK_S']}")
    print(f"  A={vocab['TOK_A']}")
    print(f"  R={vocab['TOK_R']}  EVAL={vocab['TOK_EVAL']}  SELECT={vocab['TOK_SELECT']}")
    print(f"  QCURR={vocab['TOK_QCURR']}  QNEXT={vocab['TOK_QNEXT']}  UPDATE={vocab['TOK_UPDATE']}")

    all_seqs = generate_dataset(
        n_sequences=args.n_sequences,
        max_states=args.max_states,
        max_actions=args.max_actions,
        min_states=args.min_states,
        min_actions=args.min_actions,
        min_steps=args.min_steps,
        max_steps=args.max_steps,
        alpha=args.alpha,
        gamma=args.gamma,
        seed=args.seed,
        stochastic_rewards=args.stochastic_rewards,
    )

    n_val   = args.n_sequences // 10
    n_train = args.n_sequences - n_val
    train_seqs = all_seqs[:n_train]
    val_seqs   = all_seqs[n_train:]

    step_counts = [s['n_steps'] for s in all_seqs]
    print(f"\nStep count stats:")
    print(f"  min={min(step_counts)}  max={max(step_counts)}  "
          f"mean={sum(step_counts)/len(step_counts):.1f}")

    print_sample(all_seqs[0], vocab)

    out_path = args.output
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    torch.save({
        'train':  train_seqs,
        'val':    val_seqs,
        'config': {
            'max_states':           args.max_states,
            'max_actions':          args.max_actions,
            'min_states':           args.min_states,
            'min_actions':          args.min_actions,
            'n_sequences':          args.n_sequences,
            'n_train':              n_train,
            'n_val':                n_val,
            'min_steps':            args.min_steps,
            'max_steps':            args.max_steps,
            'alpha':                args.alpha,
            'gamma':                args.gamma,
            'epsilon':              'uniform(0,1)',
            'q_init':               0.0,
            'seed':                 args.seed,
            'stochastic_rewards':   args.stochastic_rewards,
            'reward_distributions': _REWARD_DIST_NAMES,
            'transition_concentrations': _TRANS_CONC_NAMES,
            'vocab_size':           vocab['vocab_size'],
        },
    }, out_path)

    size_mb = os.path.getsize(out_path) / (1024 * 1024)
    print(f"\nDataset saved to {out_path}  ({size_mb:.1f} MB)")
    print(f"  train: {len(train_seqs):,} sequences")
    print(f"  val:   {len(val_seqs):,} sequences")


if __name__ == '__main__':
    main()
