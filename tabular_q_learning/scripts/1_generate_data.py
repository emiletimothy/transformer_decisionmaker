#!/usr/bin/env python3
"""
1_generate_data.py — Adversarial Environment & Data Generator

Generates 50,000 Q-learning trajectories formatted as two-phase COCONUT sequences
and saves them to data/coconut_dataset.pt.

Vocabulary layout (vocab_size = 2 + n_states + n_actions + 9):
    TOK_NULL   = 0
    TOK_START  = 1
    TOK_S      = [2, ..., 2 + n_states - 1]
    TOK_A      = [2 + n_states, ..., 2 + n_states + n_actions - 1]
    TOK_R      = 2 + n_states + n_actions        (float value stored separately)
    TOK_EVAL   = TOK_R + 1
    TOK_SELECT = TOK_R + 2
    TOK_THINK  = TOK_R + 3  (legacy; never placed in new sequences)
    TOK_COT    = TOK_R + 4  (continuous thought placeholder; embedding overwritten at runtime)
    TOK_UPDATE = TOK_R + 5  (Phase 2 workspace end marker; no loss)
    TOK_QCURR  = TOK_R + 6  (scaffold; no loss)
    TOK_QNEXT  = TOK_R + 7  (scaffold; no loss)
    TOK_ANEXT  = TOK_R + 8  (a_{t+1} echo scaffold; no loss)

Per-round TWO-PHASE sequence (for n_actions=2, round_len=18 tokens discrete):

  Phase 1 — action selection (always supervised at TOK_SELECT):
    s_t, a_t, TOK_R, s_{t+1},
    s_{t+1}, a_0, TOK_EVAL,
    s_{t+1}, a_1, TOK_EVAL,
    TOK_SELECT

  Phase 2 — TD update workspace (scaffold + discrete thought block):
    TOK_ANEXT, TOK_QCURR, TOK_QNEXT, TOK_UPDATE,
    TOK_S[s_t], TOK_A[a_t], TOK_QBIN[0]  ← placeholder; real Q_bin computed at collation

Continuous rounds (emitted by 3_train.py collate) replace the 3-token thought block
with 1 TOK_COT, giving 16 tokens per round instead of 18.

Full episode:
    [TOK_NULL, TOK_START, round_0, round_1, ..., round_{T-1}]

Output per sequence:
    input_ids        : List[int]    — full token sequence (18 tokens/round, all discrete)
    reward_values    : List[float]  — actual r per step (not binned)
    reward_positions : List[int]    — positions of TOK_R in input_ids
    select_positions : List[int]    — positions of TOK_SELECT tokens
    select_targets   : List[int]    — greedy action at s_{t+1} after TD update (CE target)
    a_next_positions : List[int]    — positions of TOK_ANEXT tokens
    q_curr_positions : List[int]    — positions of TOK_QCURR tokens
    q_next_positions : List[int]    — positions of TOK_QNEXT tokens
    update_positions : List[int]    — positions of TOK_UPDATE tokens
    thought_positions: List[List[int]] — [pos0, pos1, pos2] per round (the 3-token thought block)
    q_values_for_cot : List[dict]   — {st, at, q_new} per step (for Q_bin at collation time)
    n_steps          : int
"""

import argparse
import os
from typing import Dict, List, Tuple

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Vocabulary builder
# ---------------------------------------------------------------------------

def build_vocab(n_states: int, n_actions: int) -> Dict[str, object]:
    """Build vocabulary token ID constants for the two-phase COCONUT sequence format."""
    TOK_NULL   = 0
    TOK_START  = 1
    TOK_S      = list(range(2, 2 + n_states))
    TOK_A      = list(range(2 + n_states, 2 + n_states + n_actions))
    TOK_R      = 2 + n_states + n_actions
    TOK_EVAL   = TOK_R + 1
    TOK_SELECT = TOK_R + 2
    TOK_THINK  = TOK_R + 3  # legacy; never placed in new sequences
    TOK_COT    = TOK_R + 4  # continuous thought placeholder
    TOK_UPDATE = TOK_R + 5  # Phase 2 workspace end marker
    TOK_QCURR  = TOK_R + 6  # scaffold (no loss)
    TOK_QNEXT  = TOK_R + 7  # scaffold (no loss)
    TOK_ANEXT  = TOK_R + 8  # a_{t+1} echo scaffold (no loss)
    vocab_size = 2 + n_states + n_actions + 9
    return {
        'TOK_NULL':   TOK_NULL,
        'TOK_START':  TOK_START,
        'TOK_S':      TOK_S,
        'TOK_A':      TOK_A,
        'TOK_R':      TOK_R,
        'TOK_EVAL':   TOK_EVAL,
        'TOK_SELECT': TOK_SELECT,
        'TOK_THINK':  TOK_THINK,
        'TOK_COT':    TOK_COT,
        'TOK_UPDATE': TOK_UPDATE,
        'TOK_QCURR':  TOK_QCURR,
        'TOK_QNEXT':  TOK_QNEXT,
        'TOK_ANEXT':  TOK_ANEXT,
        'vocab_size': vocab_size,
    }


# ---------------------------------------------------------------------------
# MDP generation
# ---------------------------------------------------------------------------

def generate_random_mdp(
    n_states: int,
    n_actions: int,
    rng: np.random.Generator = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a fully random MDP.

    Transitions P[s, a, s'] are sampled row-wise from Dirichlet(1) (i.e. uniform
    over simplices), giving a valid stochastic transition matrix.

    Rewards R[s, a] ~ Beta(2, 2) clamped to [0, 1], peaked near 0.5.

    Returns
    -------
    P : np.ndarray, shape (n_states, n_actions, n_states)
    R : np.ndarray, shape (n_states, n_actions)
    """
    if rng is None:
        rng = np.random.default_rng()

    P = rng.dirichlet(alpha=np.ones(n_states), size=(n_states, n_actions))
    R = rng.beta(2.0, 2.0, size=(n_states, n_actions)).astype(np.float32)
    R = np.clip(R, 0.0, 1.0)
    P = P.astype(np.float32)
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
) -> Dict:
    """Run one Q-learning episode and return raw trajectory data.

    Parameters
    ----------
    P             : transition matrix (n_states, n_actions, n_states)
    R             : reward matrix (n_states, n_actions)
    n_states      : number of states
    n_actions     : number of actions
    n_steps       : episode length (number of Bellman updates)
    alpha, gamma  : Q-learning hyperparameters
    epsilon       : exploration rate for ε-greedy behavior policy
    q_init        : initial value for all Q-table entries
    rng           : numpy random Generator

    Returns
    -------
    dict with keys:
        states, actions, rewards, next_states : List[int/float], length n_steps
        next_actions : List[int], length n_steps  (greedy action at s_{t+1} after TD update)
        q_snapshots  : List[np.ndarray]  Q-table AFTER each update, (n_states, n_actions)
        q_values_for_cot : List[dict]  {st, at, q_new} per step
    """
    if rng is None:
        rng = np.random.default_rng()

    Q = np.full((n_states, n_actions), q_init, dtype=np.float32)
    s = int(rng.integers(n_states))

    states           = []
    actions          = []
    rewards          = []
    next_states      = []
    next_actions     = []
    q_snapshots      = []
    q_before_updates = []
    q_values_for_cot = []

    for _ in range(n_steps):
        if rng.random() < epsilon:
            a = int(rng.integers(n_actions))
        else:
            best = float(np.max(Q[s]))
            ties = [ac for ac in range(n_actions) if Q[s, ac] == best]
            a = int(rng.choice(ties))

        r = float(R[s, a])
        s_next = int(rng.choice(n_states, p=P[s, a]))

        q_before = Q.copy()

        max_q_next = float(np.max(Q[s_next]))
        Q[s, a] = (1.0 - alpha) * Q[s, a] + alpha * (r + gamma * max_q_next)

        best_next = float(np.max(Q[s_next]))
        ties_next = [ac for ac in range(n_actions) if Q[s_next, ac] == best_next]
        a_next = int(rng.choice(ties_next))

        states.append(s)
        actions.append(a)
        rewards.append(r)
        next_states.append(s_next)
        next_actions.append(a_next)
        q_snapshots.append(Q.copy())
        q_before_updates.append(q_before)
        q_values_for_cot.append({'st': s, 'at': a, 'q_new': float(Q[s, a])})

        s = s_next

    return {
        'states':           states,
        'actions':          actions,
        'rewards':          rewards,
        'next_states':      next_states,
        'next_actions':     next_actions,
        'q_snapshots':      q_snapshots,
        'q_before_updates': q_before_updates,
        'q_values_for_cot': q_values_for_cot,
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
    """Randomly relabel state and action integers in-place."""
    perm_s = rng.permutation(n_states)
    perm_a = rng.permutation(n_actions)

    inv_s = np.argsort(perm_s)
    inv_a = np.argsort(perm_a)

    new_states       = [int(perm_s[s])        for s in episode['states']]
    new_actions      = [int(perm_a[a])        for a in episode['actions']]
    new_next_states  = [int(perm_s[s])        for s in episode['next_states']]
    new_next_actions = [int(perm_a[a])        for a in episode['next_actions']]

    new_q_snapshots = []
    for Q in episode['q_snapshots']:
        Q_new = Q[np.ix_(inv_s, inv_a)]
        new_q_snapshots.append(Q_new)

    new_q_before_updates = []
    for Q in episode.get('q_before_updates', []):
        new_q_before_updates.append(Q[np.ix_(inv_s, inv_a)])

    new_q_values_for_cot = []
    for entry in episode.get('q_values_for_cot', []):
        new_q_values_for_cot.append({
            'st':    int(perm_s[entry['st']]),
            'at':    int(perm_a[entry['at']]),
            'q_new': entry['q_new'],
        })

    return {
        'states':           new_states,
        'actions':          new_actions,
        'rewards':          episode['rewards'],
        'next_states':      new_next_states,
        'next_actions':     new_next_actions,
        'q_snapshots':      new_q_snapshots,
        'q_before_updates': new_q_before_updates,
        'q_values_for_cot': new_q_values_for_cot,
        'perm_s':           perm_s.tolist(),
        'perm_a':           perm_a.tolist(),
    }


# ---------------------------------------------------------------------------
# Two-phase COCONUT sequence builder
# ---------------------------------------------------------------------------

def build_coconut_sequence(
    episode: Dict,
    n_states: int,  # noqa: kept for API compat
    n_actions: int,
    vocab: Dict,
) -> Dict:
    """Convert a raw episode trajectory to a two-phase COCONUT token sequence.

    Always emits the discrete form (18 tokens/round for n_actions=2). The
    continuous form (16 tokens/round) is constructed at collation time by
    3_train.py's build_mixed_sequence, which replaces the 3-token thought block
    with 1 TOK_COT for rounds designated continuous by the curriculum mask.

    Per-round structure (discrete, n_actions=2):
      Phase 1 (11 tokens):
        s_t, a_t, TOK_R, s_{t+1},
        s_{t+1}, a_0, TOK_EVAL,
        s_{t+1}, a_1, TOK_EVAL,
        TOK_SELECT
      Phase 2 (7 tokens):
        TOK_ANEXT, TOK_QCURR, TOK_QNEXT, TOK_UPDATE,
        TOK_S[s_t], TOK_A[a_t], TOK_QBIN[0]   ← Q_bin placeholder

    round_len_discrete  = 4 + 3*n_actions + 1 + 4 + 3 = 12 + 3*n_actions
    round_len_continuous = 4 + 3*n_actions + 1 + 4 + 1 = 10 + 3*n_actions

    TOK_SELECT is the ONLY supervised Phase 1 position. Target = greedy a_{t+1}.
    TOK_ANEXT, TOK_QCURR, TOK_QNEXT, TOK_UPDATE are scaffold tokens (no loss).
    The 3-token thought block [s_t, a_t, Q_bin] is supervised in discrete rounds.

    Returns
    -------
    dict with keys:
        input_ids        : List[int]
        reward_values    : List[float]
        reward_positions : List[int]
        select_positions : List[int]
        select_targets   : List[int]
        a_next_positions : List[int]
        q_curr_positions : List[int]
        q_next_positions : List[int]
        update_positions : List[int]
        thought_positions: List[List[int]]  — [[p0,p1,p2], ...] per round
        q_values_for_cot : List[dict]       — {st, at, q_new} per round
        n_steps          : int
    """
    TOK_S      = vocab['TOK_S']
    TOK_A      = vocab['TOK_A']
    TOK_R      = vocab['TOK_R']
    TOK_EVAL   = vocab['TOK_EVAL']
    TOK_SELECT = vocab['TOK_SELECT']
    TOK_ANEXT  = vocab['TOK_ANEXT']
    TOK_QCURR  = vocab['TOK_QCURR']
    TOK_QNEXT  = vocab['TOK_QNEXT']
    TOK_UPDATE = vocab['TOK_UPDATE']

    ids               = [vocab['TOK_NULL'], vocab['TOK_START']]
    reward_values     : List[float] = []
    reward_positions  : List[int]   = []
    select_positions  : List[int]   = []
    select_targets    : List[int]   = []
    a_next_positions  : List[int]   = []
    q_curr_positions  : List[int]   = []
    q_next_positions  : List[int]   = []
    update_positions  : List[int]   = []
    thought_positions : List[List[int]] = []

    n_steps   = episode['n_steps'] if 'n_steps' in episode else len(episode['states'])
    states    = episode['states']
    actions   = episode['actions']
    rewards   = episode['rewards']
    nxt_s     = episode['next_states']
    nxt_a     = episode['next_actions']
    qvc       = episode.get('q_values_for_cot', [])

    for t in range(n_steps):
        s   = states[t]
        a   = actions[t]
        r   = rewards[t]
        s_p = nxt_s[t]
        a_p = nxt_a[t]

        # ---- Phase 1: action selection ----

        # s_t
        ids.append(TOK_S[s])
        # a_t
        ids.append(TOK_A[a])
        # TOK_R (reward; actual float stored in reward_values)
        reward_positions.append(len(ids))
        reward_values.append(r)
        ids.append(TOK_R)
        # s_{t+1}
        ids.append(TOK_S[s_p])

        # Eval blocks: [s_{t+1}, a_c, TOK_EVAL] for each candidate action c
        for c in range(n_actions):
            ids.append(TOK_S[s_p])
            ids.append(TOK_A[c])
            ids.append(TOK_EVAL)

        # TOK_SELECT — supervised; target = greedy action at s_{t+1}
        select_positions.append(len(ids))
        select_targets.append(a_p)
        ids.append(TOK_SELECT)

        # ---- Phase 2: TD workspace scaffold + thought block ----

        # TOK_ANEXT — a_{t+1} echo scaffold (no loss)
        a_next_positions.append(len(ids))
        ids.append(TOK_ANEXT)

        # TOK_QCURR — scaffold (no loss)
        q_curr_positions.append(len(ids))
        ids.append(TOK_QCURR)

        # TOK_QNEXT — scaffold (no loss)
        q_next_positions.append(len(ids))
        ids.append(TOK_QNEXT)

        # TOK_UPDATE — workspace end marker (no loss); COCONUT injection source
        update_positions.append(len(ids))
        ids.append(TOK_UPDATE)

        # 3-token discrete thought block: [s_t, a_t, Q_bin placeholder]
        # The actual Q_bin (from q_values_for_cot) is substituted at collation time.
        # We store TOK_S[s_t] and TOK_A[a_t] directly; Q_bin uses index 0 as placeholder.
        p0 = len(ids)
        ids.append(TOK_S[s])   # thought token 0: s_t

        p1 = len(ids)
        ids.append(TOK_A[a])   # thought token 1: a_t

        p2 = len(ids)
        # Q_bin placeholder = first Q-bin token; real bin computed at collation
        # We need to know the Q_bin token ID but 1_generate_data.py doesn't track n_q_bins.
        # Store a sentinel integer 0 here; build_mixed_sequence in 3_train.py replaces it.
        ids.append(0)          # thought token 2: Q_bin placeholder (token ID 0 = TOK_NULL)

        thought_positions.append([p0, p1, p2])

    return {
        'input_ids':         ids,
        'reward_values':     reward_values,
        'reward_positions':  reward_positions,
        'select_positions':  select_positions,
        'select_targets':    select_targets,
        'a_next_positions':  a_next_positions,
        'q_curr_positions':  q_curr_positions,
        'q_next_positions':  q_next_positions,
        'update_positions':  update_positions,
        'thought_positions': thought_positions,
        'q_values_for_cot':  qvc,
        'n_steps':           n_steps,
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
    seed: int = 42,
) -> List[Dict]:
    """Generate `n_sequences` two-phase COCONUT-formatted trajectories."""
    rng = np.random.default_rng(seed)
    vocab = build_vocab(n_states, n_actions)
    sequences = []

    for i in range(n_sequences):
        n_steps = int(rng.integers(min_steps, max_steps + 1))
        epsilon = float(rng.uniform(0.0, 1.0))

        P, R = generate_random_mdp(n_states, n_actions, rng=rng)
        ep = generate_episode(
            P, R, n_states, n_actions, n_steps,
            alpha=alpha, gamma=gamma, epsilon=epsilon, q_init=0.0,
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

def print_sample(seq: Dict, vocab: Dict, n_actions: int = 2) -> None:  # noqa: n_actions unused in body
    """Pretty-print the first few tokens of a sample sequence."""
    tnames = {
        vocab['TOK_NULL']:   'NULL',
        vocab['TOK_START']:  'START',
        vocab['TOK_R']:      'R',
        vocab['TOK_EVAL']:   'EVAL',
        vocab['TOK_SELECT']: 'SELECT',
        vocab['TOK_THINK']:  'THINK(legacy)',
        vocab['TOK_COT']:    'COT',
        vocab['TOK_UPDATE']: 'UPDATE',
        vocab['TOK_QCURR']:  'QCURR',
        vocab['TOK_QNEXT']:  'QNEXT',
        vocab['TOK_ANEXT']:  'ANEXT',
    }
    for i, s in enumerate(vocab['TOK_S']):
        tnames[s] = f'S{i}'
    for i, a in enumerate(vocab['TOK_A']):
        tnames[a] = f'A{i}'

    rp_set     = set(seq['reward_positions'])
    sel_set    = set(seq['select_positions'])
    upd_set    = set(seq['update_positions'])
    thought_set = set()
    for tp in seq['thought_positions']:
        thought_set.update(tp)

    r_idx = 0; sel_idx = 0

    print(f"\nSample sequence  (n_steps={seq['n_steps']}, "
          f"seq_len={len(seq['input_ids'])})")
    print(f"  select_targets : {seq['select_targets'][:3]} ...")
    print()
    print(f"  {'pos':>4}  {'name':<14}  {'id':>4}  note")
    print(f"  {'-'*4}  {'-'*14}  {'-'*4}  {'-'*35}")
    for pos, tid in enumerate(seq['input_ids'][:60]):
        name = tnames.get(tid, f'?{tid}')
        note = ''
        if pos in rp_set:
            note = f'r={seq["reward_values"][r_idx]:.4f}'
            r_idx += 1
        elif pos in sel_set:
            note = f'target_action={seq["select_targets"][sel_idx]}'
            sel_idx += 1
        elif pos in upd_set:
            note = '[COCONUT injection source]'
        elif pos in thought_set:
            note = '[thought block token]'
        print(f"  {pos:>4}  {name:<14}  {tid:>4}  {note}")
    if len(seq['input_ids']) > 60:
        print(f"  ... ({len(seq['input_ids']) - 60} more tokens)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Generate two-phase COCONUT Q-learning dataset'
    )
    parser.add_argument('--n_sequences', type=int, default=50_000)
    parser.add_argument('--n_states',    type=int, default=4)
    parser.add_argument('--n_actions',   type=int, default=2)
    parser.add_argument('--min_steps',   type=int, default=10)
    parser.add_argument('--max_steps',   type=int, default=50)
    parser.add_argument('--alpha',       type=float, default=0.1)
    parser.add_argument('--gamma',       type=float, default=0.9)
    parser.add_argument('--seed',        type=int, default=42)
    parser.add_argument('--output',      type=str,
                        default=os.path.join(os.path.dirname(__file__),
                                             '..', 'data', 'coconut_dataset.pt'))
    args = parser.parse_args()

    np.random.seed(args.seed)

    print(f"Generating {args.n_sequences:,} two-phase COCONUT sequences "
          f"({args.n_states} states, {args.n_actions} actions, "
          f"steps=[{args.min_steps},{args.max_steps}]) ...")
    print(f"  ε ~ Uniform(0,1) per trajectory, Q₀ = 0")

    vocab = build_vocab(args.n_states, args.n_actions)
    print(f"\nVocabulary (vocab_size={vocab['vocab_size']}):")
    print(f"  NULL={vocab['TOK_NULL']}  START={vocab['TOK_START']}")
    print(f"  S={vocab['TOK_S']}")
    print(f"  A={vocab['TOK_A']}")
    print(f"  R={vocab['TOK_R']}  EVAL={vocab['TOK_EVAL']}  SELECT={vocab['TOK_SELECT']}")
    print(f"  THINK(legacy)={vocab['TOK_THINK']}  COT={vocab['TOK_COT']}")
    print(f"  UPDATE={vocab['TOK_UPDATE']}  QCURR={vocab['TOK_QCURR']}")
    print(f"  QNEXT={vocab['TOK_QNEXT']}  ANEXT={vocab['TOK_ANEXT']}")

    round_len_discrete   = 4 + 3 * args.n_actions + 1 + 4 + 3
    round_len_continuous = 4 + 3 * args.n_actions + 1 + 4 + 1
    print(f"\nRound length (discrete):   {round_len_discrete} tokens")
    print(f"Round length (continuous): {round_len_continuous} tokens")
    print(f"Max sequence length (discrete form): 2 + {round_len_discrete}*{args.max_steps} = "
          f"{2 + round_len_discrete * args.max_steps} tokens")

    all_seqs = generate_dataset(
        n_sequences=args.n_sequences,
        n_states=args.n_states,
        n_actions=args.n_actions,
        min_steps=args.min_steps,
        max_steps=args.max_steps,
        alpha=args.alpha,
        gamma=args.gamma,
        seed=args.seed,
    )

    n_val   = args.n_sequences // 10
    n_train = args.n_sequences - n_val
    train_seqs = all_seqs[:n_train]
    val_seqs   = all_seqs[n_train:]

    seq_lens = [len(s['input_ids']) for s in all_seqs]
    print(f"\nSequence length stats (discrete form):")
    print(f"  min={min(seq_lens)}  max={max(seq_lens)}  "
          f"mean={sum(seq_lens)/len(seq_lens):.1f}")

    print_sample(all_seqs[0], vocab, args.n_actions)

    out_path = args.output
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    torch.save({
        'train':  train_seqs,
        'val':    val_seqs,
        'config': {
            'n_states':             args.n_states,
            'n_actions':            args.n_actions,
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
            'vocab_size':           vocab['vocab_size'],
            'round_len_discrete':   round_len_discrete,
            'round_len_continuous': round_len_continuous,
        },
    }, out_path)

    size_mb = os.path.getsize(out_path) / (1024 * 1024)
    print(f"\nDataset saved to {out_path}  ({size_mb:.1f} MB)")
    print(f"  train: {len(train_seqs):,} sequences")
    print(f"  val:   {len(val_seqs):,} sequences")


if __name__ == '__main__':
    main()
