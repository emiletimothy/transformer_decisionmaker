#!/usr/bin/env python3
"""
1_generate_data.py — Adversarial Environment & Data Generator

Generates 50,000 Q-learning trajectories formatted as strict COCONUT sequences
and saves them to data/coconut_dataset.pt.

Vocabulary layout (vocab_size = 2 + n_states + n_actions + 7):
    TOK_NULL   = 0
    TOK_START  = 1
    TOK_S      = [2, ..., 2 + n_states - 1]
    TOK_A      = [2 + n_states, ..., 2 + n_states + n_actions - 1]
    TOK_R      = 2 + n_states + n_actions        (float value stored separately)
    TOK_EVAL   = TOK_R + 1
    TOK_SELECT = TOK_R + 2
    TOK_QCURR  = TOK_R + 3
    TOK_QNEXT  = TOK_R + 4
    TOK_UPDATE = TOK_R + 5
    TOK_COT    = TOK_R + 6  (continuous thought placeholder; embedding overwritten at runtime)

Per-round COCONUT sequence (for n_actions=2, 15 tokens/round):
    s_t, a_t, TOK_R, s_{t+1},
    s_{t+1}, a_0, TOK_EVAL,
    s_{t+1}, a_1, TOK_EVAL,
    TOK_SELECT,
    a_{t+1}, TOK_QCURR, TOK_QNEXT, TOK_UPDATE, TOK_COT

Full episode:
    [TOK_NULL, TOK_START, round_0, round_1, ..., round_{T-1}]

Output per sequence:
    input_ids        : List[int]          — full token sequence
    reward_values    : List[float]        — actual r per step (not binned)
    reward_positions : List[int]          — positions of TOK_R in input_ids
    select_positions : List[int]          — positions of TOK_SELECT tokens
    select_targets   : List[int]          — true next action index for CE loss
    select_masks     : List[int]          — 1 if Q-values at s_{t+1} are differentiated (>0.01 spread), else 0
    update_positions : List[int]          — positions of TOK_UPDATE tokens
    update_targets   : List[List[float]]  — Q[s_t, :] (all actions) after update for MSE loss
    cot_positions    : List[int]          — positions of TOK_COT tokens (continuous thought placeholders)
    n_steps          : int
"""

import argparse
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Vocabulary builder
# ---------------------------------------------------------------------------

def build_vocab(n_states: int, n_actions: int) -> Dict[str, object]:
    """Build vocabulary token ID constants for the COCONUT sequence format."""
    TOK_NULL  = 0
    TOK_START = 1
    TOK_S     = list(range(2, 2 + n_states))
    TOK_A     = list(range(2 + n_states, 2 + n_states + n_actions))
    TOK_R     = 2 + n_states + n_actions
    TOK_EVAL  = TOK_R + 1
    TOK_SELECT = TOK_R + 2
    TOK_QCURR = TOK_R + 3
    TOK_QNEXT = TOK_R + 4
    TOK_UPDATE = TOK_R + 5
    TOK_COT    = TOK_R + 6
    vocab_size = 2 + n_states + n_actions + 7
    return {
        'TOK_NULL': TOK_NULL,
        'TOK_START': TOK_START,
        'TOK_S': TOK_S,
        'TOK_A': TOK_A,
        'TOK_R': TOK_R,
        'TOK_EVAL': TOK_EVAL,
        'TOK_SELECT': TOK_SELECT,
        'TOK_QCURR': TOK_QCURR,
        'TOK_QNEXT': TOK_QNEXT,
        'TOK_UPDATE': TOK_UPDATE,
        'TOK_COT': TOK_COT,
        'vocab_size': vocab_size,
    }


# ---------------------------------------------------------------------------
# MDP generation
# ---------------------------------------------------------------------------

def generate_random_mdp(
    n_states: int,
    n_actions: int,
    trap_prob: float = 0.2,
    rng: np.random.Generator = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate a random directed-graph MDP with trap states.

    Transitions P[s, a, s'] are sampled row-wise from Dirichlet(1) (i.e. uniform
    over simplices), giving a valid stochastic transition matrix.

    Rewards R[s, a] ~ Beta(2, 2) clamped to [0, 1], peaked near 0.5.

    Trap states (≈ trap_prob fraction of states) have:
        - deterministic self-loop: P[s, a, s] = 1 for all a
        - zero reward: R[s, a] = 0 for all a

    Returns
    -------
    P    : np.ndarray, shape (n_states, n_actions, n_states)
    R    : np.ndarray, shape (n_states, n_actions)
    trap : np.ndarray, shape (n_states,) bool — True for trap states
    """
    if rng is None:
        rng = np.random.default_rng()

    # Dirichlet(alpha=1) = uniform over simplex
    P = rng.dirichlet(alpha=np.ones(n_states), size=(n_states, n_actions))
    # P[s, a, :] sums to 1; shape (n_states, n_actions, n_states)

    R = rng.beta(2.0, 2.0, size=(n_states, n_actions)).astype(np.float32)
    R = np.clip(R, 0.0, 1.0)

    # Designate trap states
    trap = rng.random(n_states) < trap_prob
    for s in np.where(trap)[0]:
        # Self-loop for every action
        P[s, :, :] = 0.0
        P[s, :, s] = 1.0
        R[s, :] = 0.0

    P = P.astype(np.float32)
    return P, R, trap


# ---------------------------------------------------------------------------
# Episode generation
# ---------------------------------------------------------------------------

def generate_episode(
    P: np.ndarray,
    R: np.ndarray,
    n_states: int,
    n_actions: int,
    n_steps: int,
    use_random_walk: bool = False,
    alpha: float = 0.1,
    gamma: float = 0.9,
    epsilon: float = 0.2,
    rng: np.random.Generator = None,
) -> Dict:
    """Run one Q-learning (or random-walk) episode and return raw trajectory data.

    Parameters
    ----------
    P             : transition matrix (n_states, n_actions, n_states)
    R             : reward matrix (n_states, n_actions)
    n_states      : number of states
    n_actions     : number of actions
    n_steps       : episode length (number of Bellman updates)
    use_random_walk : if True, actions chosen uniformly at random (off-policy)
    alpha, gamma, epsilon : Q-learning hyperparameters (only alpha/gamma affect Q-table)
    rng           : numpy random Generator

    Returns
    -------
    dict with keys:
        states, actions, rewards, next_states : List[int/float], length n_steps
        next_actions : List[int], length n_steps  (action chosen at s_{t+1})
        q_snapshots  : List[np.ndarray]  Q-table AFTER each update, (n_states, n_actions)
    """
    if rng is None:
        rng = np.random.default_rng()

    Q = np.zeros((n_states, n_actions), dtype=np.float32)
    s = int(rng.integers(n_states))

    states       = []
    actions      = []
    rewards      = []
    next_states  = []
    next_actions = []
    q_snapshots  = []

    for _ in range(n_steps):
        # Behavior policy: epsilon-greedy or pure random
        if use_random_walk:
            a = int(rng.integers(n_actions))
        else:
            if rng.random() < epsilon:
                a = int(rng.integers(n_actions))
            else:
                best = float(np.max(Q[s]))
                ties = [ac for ac in range(n_actions) if Q[s, ac] == best]
                a = int(rng.choice(ties))

        # Environment step: deterministic reward, stochastic transition
        r = float(R[s, a])
        s_next = int(rng.choice(n_states, p=P[s, a]))

        # Q-learning update (off-policy: uses greedy max over s_next)
        max_q_next = float(np.max(Q[s_next]))
        Q[s, a] = (1.0 - alpha) * Q[s, a] + alpha * (r + gamma * max_q_next)

        # Next action chosen greedily from s_next (for the SELECT token target)
        best_next = float(np.max(Q[s_next]))
        ties_next = [ac for ac in range(n_actions) if Q[s_next, ac] == best_next]
        a_next = int(rng.choice(ties_next))

        states.append(s)
        actions.append(a)
        rewards.append(r)
        next_states.append(s_next)
        next_actions.append(a_next)
        q_snapshots.append(Q.copy())

        s = s_next

    return {
        'states':       states,
        'actions':      actions,
        'rewards':      rewards,
        'next_states':  next_states,
        'next_actions': next_actions,
        'q_snapshots':  q_snapshots,   # List of (n_states, n_actions) arrays
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
    """Randomly relabel state and action integers in-place.

    Shuffles the integer labels of states AND actions so the model cannot
    memorize specific state/action indices. Also permutes Q-table rows/columns
    to match the new labeling.

    Returns a new episode dict with permuted fields.
    """
    perm_s = rng.permutation(n_states)   # perm_s[old_state] = new_state
    perm_a = rng.permutation(n_actions)  # perm_a[old_action] = new_action

    # Inverse permutation for Q-table reordering
    inv_s = np.argsort(perm_s)
    inv_a = np.argsort(perm_a)

    new_states       = [int(perm_s[s])        for s in episode['states']]
    new_actions      = [int(perm_a[a])        for a in episode['actions']]
    new_next_states  = [int(perm_s[s])        for s in episode['next_states']]
    new_next_actions = [int(perm_a[a])        for a in episode['next_actions']]

    # Permute Q-table: Q_new[perm_s[s], perm_a[a]] = Q_old[s, a]
    new_q_snapshots = []
    for Q in episode['q_snapshots']:
        Q_new = Q[np.ix_(inv_s, inv_a)]   # reindex rows then cols
        new_q_snapshots.append(Q_new)

    return {
        'states':       new_states,
        'actions':      new_actions,
        'rewards':      episode['rewards'],
        'next_states':  new_next_states,
        'next_actions': new_next_actions,
        'q_snapshots':  new_q_snapshots,
        'perm_s':       perm_s.tolist(),
        'perm_a':       perm_a.tolist(),
    }


# ---------------------------------------------------------------------------
# COCONUT sequence builder
# ---------------------------------------------------------------------------

def build_coconut_sequence(
    episode: Dict,
    n_states: int,
    n_actions: int,
    vocab: Dict,
) -> Dict:
    """Convert a raw episode trajectory to a COCONUT token sequence.

    Per-round structure (PHASE I + PHASE II):
        PHASE I:   s_t, a_t, TOK_R, s_{t+1},
                   [s_{t+1}, a_c, TOK_EVAL] for c in 0..n_actions-1,
                   TOK_SELECT
        PHASE II:  a_{t+1}, TOK_QCURR, TOK_QNEXT, TOK_UPDATE

    round_len = 4 + 3*n_actions + 4  tokens

    The TOK_R position carries the reward as a *discrete token ID* but the
    actual float reward is stored in `reward_values` at the matching index.

    The SELECT token target is the greedy next action index (0-based integer).
    The UPDATE token target is the full Q-row Q[s_t, :] after the TD update.

    Returns
    -------
    dict with keys:
        input_ids        : List[int]
        reward_values    : List[float]
        reward_positions : List[int]
        select_positions : List[int]
        select_targets   : List[int]
        select_masks     : List[int]   — 1 if Q-spread at s_{t+1} > 0.01, else 0
        update_positions : List[int]
        update_targets   : List[List[float]]
        cot_positions    : List[int]
        n_steps          : int
    """
    TOK_S      = vocab['TOK_S']
    TOK_A      = vocab['TOK_A']
    TOK_R      = vocab['TOK_R']
    TOK_EVAL   = vocab['TOK_EVAL']
    TOK_SELECT = vocab['TOK_SELECT']
    TOK_QCURR  = vocab['TOK_QCURR']
    TOK_QNEXT  = vocab['TOK_QNEXT']
    TOK_UPDATE = vocab['TOK_UPDATE']
    TOK_COT    = vocab['TOK_COT']

    ids               = [vocab['TOK_NULL'], vocab['TOK_START']]
    reward_values     : List[float] = []
    reward_positions  : List[int]   = []
    select_positions  : List[int]   = []
    select_targets    : List[int]   = []
    select_masks      : List[int]   = []
    update_positions  : List[int]   = []
    update_targets    : List[List[float]] = []
    cot_positions     : List[int]   = []

    n_steps   = episode['n_steps'] if 'n_steps' in episode else len(episode['states'])
    states    = episode['states']
    actions   = episode['actions']
    rewards   = episode['rewards']
    nxt_s     = episode['next_states']
    nxt_a     = episode['next_actions']
    q_snaps   = episode['q_snapshots']

    for t in range(n_steps):
        s     = states[t]
        a     = actions[t]
        r     = rewards[t]
        s_p   = nxt_s[t]
        a_p   = nxt_a[t]
        Q_t   = q_snaps[t]   # Q-table AFTER step t update

        # ---- PHASE I ----
        # s_t
        ids.append(TOK_S[s])
        # a_t
        ids.append(TOK_A[a])
        # reward (token ID only; actual float stored separately)
        reward_positions.append(len(ids))
        reward_values.append(r)
        ids.append(TOK_R)
        # s_{t+1}
        ids.append(TOK_S[s_p])

        # Eval blocks: for each candidate action c at s_{t+1}
        for c in range(n_actions):
            ids.append(TOK_S[s_p])
            ids.append(TOK_A[c])
            ids.append(TOK_EVAL)

        # SELECT token — model predicts a_{t+1} here
        # select_mask: 1 if Q-values at s_{t+1} have spread > 0.01 (greedy action is meaningful)
        q_vals_sp = Q_t[s_p]
        select_mask_val = 1 if (float(q_vals_sp.max()) - float(q_vals_sp.min())) > 0.01 else 0
        select_masks.append(select_mask_val)
        select_positions.append(len(ids))
        select_targets.append(a_p)      # 0-based action index
        ids.append(TOK_SELECT)

        # ---- PHASE II ----
        # a_{t+1}
        ids.append(TOK_A[a_p])
        # QCURR
        ids.append(TOK_QCURR)
        # QNEXT
        ids.append(TOK_QNEXT)
        # UPDATE — model outputs full Q-row here
        update_positions.append(len(ids))
        update_targets.append(Q_t[s].tolist())   # Q[s_t, :] after TD update
        ids.append(TOK_UPDATE)

        # COT — continuous thought placeholder; embedding overwritten at train/eval time
        cot_positions.append(len(ids))
        ids.append(TOK_COT)

    return {
        'input_ids':        ids,
        'reward_values':    reward_values,
        'reward_positions': reward_positions,
        'select_positions': select_positions,
        'select_targets':   select_targets,
        'select_masks':     select_masks,
        'update_positions': update_positions,
        'update_targets':   update_targets,
        'cot_positions':    cot_positions,
        'n_steps':          n_steps,
    }


# ---------------------------------------------------------------------------
# Dataset generation driver
# ---------------------------------------------------------------------------

def generate_dataset(
    n_sequences: int,
    n_states: int,
    n_actions: int,
    min_steps: int = 10,
    max_steps: int = 50,
    alpha: float = 0.1,
    gamma: float = 0.9,
    epsilon: float = 0.2,
    random_walk_frac: float = 0.3,
    trap_prob: float = 0.2,
    seed: int = 42,
) -> List[Dict]:
    """Generate `n_sequences` COCONUT-formatted trajectories.

    70% use epsilon-greedy Q-learning, 30% use pure random walk.
    Each episode uses a freshly sampled random MDP with trap states.
    State and action labels are randomly permuted per episode.
    """
    rng = np.random.default_rng(seed)
    vocab = build_vocab(n_states, n_actions)
    sequences = []

    n_random = int(n_sequences * random_walk_frac)
    flags = ([True] * n_random + [False] * (n_sequences - n_random))
    rng.shuffle(flags)

    for i, use_rand in enumerate(flags):
        n_steps = int(rng.integers(min_steps, max_steps + 1))
        P, R, _ = generate_random_mdp(n_states, n_actions, trap_prob=trap_prob, rng=rng)
        ep = generate_episode(
            P, R, n_states, n_actions, n_steps,
            use_random_walk=use_rand,
            alpha=alpha, gamma=gamma, epsilon=epsilon,
            rng=rng,
        )
        ep['n_steps'] = n_steps
        ep = apply_permutations(ep, n_states, n_actions, rng)
        seq = build_coconut_sequence(ep, n_states, n_actions, vocab)
        sequences.append(seq)

        if (i + 1) % 5000 == 0:
            print(f"  generated {i + 1:,} / {n_sequences:,} sequences...")

    return sequences


# ---------------------------------------------------------------------------
# Sanity-check printer
# ---------------------------------------------------------------------------

def print_sample(seq: Dict, vocab: Dict, n_actions: int) -> None:
    """Pretty-print the first few tokens of a sample sequence."""
    tnames = {vocab['TOK_NULL']: 'NULL', vocab['TOK_START']: 'START',
              vocab['TOK_R']: 'R', vocab['TOK_EVAL']: 'EVAL',
              vocab['TOK_SELECT']: 'SELECT', vocab['TOK_QCURR']: 'QCURR',
              vocab['TOK_QNEXT']: 'QNEXT', vocab['TOK_UPDATE']: 'UPDATE',
              vocab['TOK_COT']: 'COT'}
    for i, s in enumerate(vocab['TOK_S']):
        tnames[s] = f'S{i}'
    for i, a in enumerate(vocab['TOK_A']):
        tnames[a] = f'A{i}'

    rp_set = set(seq['reward_positions'])
    sel_set = set(seq['select_positions'])
    upd_set = set(seq['update_positions'])
    r_idx = 0; sel_idx = 0; upd_idx = 0

    print(f"\nSample sequence  (n_steps={seq['n_steps']}, "
          f"seq_len={len(seq['input_ids'])})")
    print(f"  select_targets : {seq['select_targets'][:3]} ...")
    print(f"  update_targets[0] : {[f'{v:.4f}' for v in seq['update_targets'][0]]}")
    print()
    print(f"  {'pos':>4}  {'name':<10}  {'id':>4}  note")
    print(f"  {'-'*4}  {'-'*10}  {'-'*4}  {'-'*30}")
    for pos, tid in enumerate(seq['input_ids'][:50]):
        name = tnames.get(tid, f'?{tid}')
        note = ''
        if pos in rp_set:
            note = f'r={seq["reward_values"][r_idx]:.4f}'
            r_idx += 1
        elif pos in sel_set:
            note = f'target_action={seq["select_targets"][sel_idx]}'
            sel_idx += 1
        elif pos in upd_set:
            q = seq['update_targets'][upd_idx]
            note = 'Q=' + '[' + ', '.join(f'{v:.3f}' for v in q) + ']'
            upd_idx += 1
        print(f"  {pos:>4}  {name:<10}  {tid:>4}  {note}")
    if len(seq['input_ids']) > 50:
        print(f"  ... ({len(seq['input_ids']) - 50} more tokens)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Generate COCONUT Q-learning dataset'
    )
    parser.add_argument('--n_sequences', type=int, default=50_000)
    parser.add_argument('--n_states',    type=int, default=5)
    parser.add_argument('--n_actions',   type=int, default=2)
    parser.add_argument('--min_steps',   type=int, default=10)
    parser.add_argument('--max_steps',   type=int, default=50)
    parser.add_argument('--alpha',       type=float, default=0.1)
    parser.add_argument('--gamma',       type=float, default=0.9)
    parser.add_argument('--epsilon',     type=float, default=0.2)
    parser.add_argument('--trap_prob',   type=float, default=0.2)
    parser.add_argument('--random_walk_frac', type=float, default=0.3)
    parser.add_argument('--seed',        type=int, default=42)
    parser.add_argument('--output',      type=str,
                        default=os.path.join(os.path.dirname(__file__),
                                             '..', 'data', 'coconut_dataset.pt'))
    args = parser.parse_args()

    np.random.seed(args.seed)

    print(f"Generating {args.n_sequences:,} COCONUT sequences "
          f"({args.n_states} states, {args.n_actions} actions, "
          f"steps=[{args.min_steps},{args.max_steps}]) ...")
    print(f"  70% epsilon-greedy (ε={args.epsilon}), "
          f"30% random walk, trap_prob={args.trap_prob}")

    vocab = build_vocab(args.n_states, args.n_actions)
    print(f"\nVocabulary (vocab_size={vocab['vocab_size']}):")
    print(f"  NULL={vocab['TOK_NULL']}  START={vocab['TOK_START']}")
    print(f"  S={vocab['TOK_S']}")
    print(f"  A={vocab['TOK_A']}")
    print(f"  R={vocab['TOK_R']}  EVAL={vocab['TOK_EVAL']}  SELECT={vocab['TOK_SELECT']}")
    print(f"  QCURR={vocab['TOK_QCURR']}  QNEXT={vocab['TOK_QNEXT']}  UPDATE={vocab['TOK_UPDATE']}  COT={vocab['TOK_COT']}")

    round_len = 4 + 3 * args.n_actions + 4 + 1   # +1 for TOK_COT
    print(f"\nRound length: {round_len} tokens (includes TOK_COT placeholder)")
    print(f"Max sequence length: 2 + {round_len}*{args.max_steps} = "
          f"{2 + round_len * args.max_steps} tokens")

    all_seqs = generate_dataset(
        n_sequences=args.n_sequences,
        n_states=args.n_states,
        n_actions=args.n_actions,
        min_steps=args.min_steps,
        max_steps=args.max_steps,
        alpha=args.alpha,
        gamma=args.gamma,
        epsilon=args.epsilon,
        random_walk_frac=args.random_walk_frac,
        trap_prob=args.trap_prob,
        seed=args.seed,
    )

    # Train / val split: 90% / 10%
    n_val   = args.n_sequences // 10
    n_train = args.n_sequences - n_val
    train_seqs = all_seqs[:n_train]
    val_seqs   = all_seqs[n_train:]

    # Sequence length statistics
    seq_lens = [len(s['input_ids']) for s in all_seqs]
    print(f"\nSequence length stats:")
    print(f"  min={min(seq_lens)}  max={max(seq_lens)}  "
          f"mean={sum(seq_lens)/len(seq_lens):.1f}")

    # Print a sample
    print_sample(all_seqs[0], vocab, args.n_actions)

    # Save
    out_path = args.output
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    torch.save({
        'train':  train_seqs,
        'val':    val_seqs,
        'config': {
            'n_states':          args.n_states,
            'n_actions':         args.n_actions,
            'n_sequences':       args.n_sequences,
            'n_train':           n_train,
            'n_val':             n_val,
            'min_steps':         args.min_steps,
            'max_steps':         args.max_steps,
            'alpha':             args.alpha,
            'gamma':             args.gamma,
            'epsilon':           args.epsilon,
            'trap_prob':         args.trap_prob,
            'random_walk_frac':  args.random_walk_frac,
            'seed':              args.seed,
            'vocab_size':        vocab['vocab_size'],
            'round_len':         round_len,
        },
    }, out_path)

    size_mb = os.path.getsize(out_path) / (1024 * 1024)
    print(f"\nDataset saved to {out_path}  ({size_mb:.1f} MB)")
    print(f"  train: {len(train_seqs):,} sequences")
    print(f"  val:   {len(val_seqs):,} sequences")


if __name__ == '__main__':
    main()
