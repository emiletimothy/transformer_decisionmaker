#!/usr/bin/env python3
"""Show what a single Q-learning dataset example looks like before and after tokenization."""
import json, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from scripts.learned_qlearning_transformer import QLearningTokenizer, compute_qlearning_qtable
import numpy as np

# Load dataset
dataset_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'qlearning_dataset.json')
if not os.path.exists(dataset_path):
    # Try current directory
    dataset_path = 'qlearning_dataset.json'

j = json.load(open(dataset_path))
seq = j['train'][0]
n_states = j['config']['n_states']
n_actions = j['config']['n_actions']

tok = QLearningTokenizer(n_states=n_states, n_actions=n_actions)

# Recompute Q-table trajectory for display
q_snapshots = compute_qlearning_qtable(seq, n_states=n_states, n_actions=n_actions)

print("=" * 70)
print("RAW SEQUENCE (first example from dataset)")
print("=" * 70)
print(f"n_steps:            {len(seq['states'])}")
print(f"alpha:              {seq['params']['alpha']:.4f}")
print(f"gamma:              {seq['params']['gamma']:.4f}")
print(f"epsilon:            {seq['params']['epsilon']:.4f}")
print()
for step in range(min(2, len(seq['states']))):
    print(f"--- Round {step} ---")
    print(f"  state:       s{seq['states'][step]}")
    print(f"  action:      a{seq['actions'][step]}")
    print(f"  reward:      {seq['rewards'][step]:.4f}")
    print(f"  next_state:  s{seq['next_states'][step]}")
    s, a = seq['states'][step], seq['actions'][step]
    q_after = q_snapshots[step]
    print(f"  Q(s{s},a{a}) after update: {q_after[s][a]:.4f}")
    print(f"  Full Q-table after:")
    for si in range(n_states):
        row = ", ".join(f"a{ai}={q_after[si][ai]:.4f}" for ai in range(n_actions))
        print(f"    s{si}: [{row}]")
print(f"... ({len(seq['states']) - 2} more steps)")

print()
print("=" * 70)
print("TOKEN VOCABULARY")
print("=" * 70)
print(f"  PAD=0  START=1  END=2  SEP=3")
print(f"  STATE tokens:   {tok.STATE_TOKENS}  (S0-S{n_states-1})")
print(f"  ACTION tokens:  {tok.ACTION_TOKENS}  (A0-A{n_actions-1})")
print(f"  REWARD tokens:  {tok.REWARD_TOKENS[0]}..{tok.REWARD_TOKENS[-1]}  (100 bins, 0.00-1.00)")
print(f"  QVALUE tokens:  {tok.QVALUE_TOKENS[0]}..{tok.QVALUE_TOKENS[-1]}  "
      f"(100 bins, {tok.q_min:.2f}-{tok.q_max:.2f})")
print(f"  ALPHA tokens:   {tok.ALPHA_TOKENS[0]}..{tok.ALPHA_TOKENS[-1]}  (100 bins, 0.00-1.00)")
print(f"  GAMMA tokens:   {tok.GAMMA_TOKENS[0]}..{tok.GAMMA_TOKENS[-1]}  (100 bins, 0.00-1.00)")
print(f"  STEP tokens:    {tok.STEP_TOKENS[0]}..{tok.STEP_TOKENS[-1]}")
print(f"  vocab_size:     {tok.vocab_size}")

print()
print("=" * 70)
print("TOKENIZED SEQUENCE (first 2 rounds)")
print("=" * 70)
print("Per-round token order:")
print("  STEP -> STATE(s) -> ACTION(a) -> REWARD(r) -> STATE(s') -> SEP [PREDICT Q(s,a)]")
print()


def token_name(t):
    if t == 0: return "PAD"
    if t == 1: return "START"
    if t == 2: return "END"
    if t == 3: return "SEP"
    if t in tok.STATE_TOKENS:
        idx = tok.STATE_TOKENS.index(t)
        return f"S{idx}"
    if t in tok.ACTION_TOKENS:
        idx = tok.ACTION_TOKENS.index(t)
        return f"A{idx}"
    if tok.REWARD_TOKENS[0] <= t <= tok.REWARD_TOKENS[-1]:
        val = (t - tok.REWARD_TOKENS[0]) / 99.0
        return f"R({val:.2f})"
    if tok.QVALUE_TOKENS[0] <= t <= tok.QVALUE_TOKENS[-1]:
        val = tok.decode_qvalue_token(t)
        return f"Q({val:.2f})"
    if tok.ALPHA_TOKENS[0] <= t <= tok.ALPHA_TOKENS[-1]:
        val = (t - tok.ALPHA_TOKENS[0]) / 99.0
        return f"ALPHA({val:.2f})"
    if tok.GAMMA_TOKENS[0] <= t <= tok.GAMMA_TOKENS[-1]:
        val = (t - tok.GAMMA_TOKENS[0]) / 99.0
        return f"GAMMA({val:.2f})"
    if t in tok.STEP_TOKENS:
        idx = tok.STEP_TOKENS.index(t)
        return f"STEP_{idx}"
    return f"?{t}"


encoded = tok.encode_sequence(seq)
ids = encoded['input_ids']
mask = encoded['target_mask']
q_tgt = encoded['q_target_values']

# Print tokens, stop after 2 rounds worth or END
round_count = 0
for i, tid in enumerate(ids):
    n = token_name(tid)
    marker = ""
    if mask[i]:
        marker = f"  <<<< MODEL PREDICTS Q-VALUE HERE (target={q_tgt[i]:.4f})"
    print(f"  [{i:3d}] {n:14s}  (id={tid:3d}){marker}")

    if n == "END":
        break
    if n.startswith("STEP"):
        round_count += 1
        if round_count > 2:
            print(f"  ... ({len(seq['states']) - 2} more steps, then END)")
            break
