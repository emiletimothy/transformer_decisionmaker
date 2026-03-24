#!/usr/bin/env python3
"""Show what a single dataset example looks like before and after tokenization."""
import json, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from learned_mw_transformer import MWTokenizer, compute_mw_weights
import numpy as np

j = json.load(open('data/mw_dataset.json'))
seq = j['train'][0]
tok = MWTokenizer(n_experts=4)

# Compute weights on-the-fly for display (not stored in training data)
weights_seq = compute_mw_weights(seq, n_experts=4)

print("=" * 70)
print("RAW SEQUENCE (first example from dataset)")
print("=" * 70)
print(f"n_steps:            {seq['n_steps']}")
print(f"learning_rate:      {seq['learning_rate']:.4f}")
print(f"true_labels:        {seq['true_labels']}")
print()
for step in range(min(2, seq['n_steps'])):
    print(f"--- Round {step} ---")
    print(f"  expert_predictions: {seq['expert_predictions'][step]}")
    print(f"  true_label:         {seq['true_labels'][step]}")
    print(f"  losses:             {seq['losses'][step]}")
    w = weights_seq[step]
    print(f"  weights (before):   [{', '.join(f'{x:.3f}' for x in w)}]")
    w2 = weights_seq[step + 1]
    print(f"  weights (after):    [{', '.join(f'{x:.3f}' for x in w2)}]")
print(f"... ({seq['n_steps'] - 2} more rounds)")

print()
print("=" * 70)
print("TOKEN VOCABULARY")
print("=" * 70)
print(f"  PAD=0  START=1  END=2  SEP=3")
print(f"  EXPERT tokens:  {tok.EXPERT_TOKENS}  (E0-E3)")
print(f"  WEIGHT tokens:  {tok.WEIGHT_TOKENS[0]}..{tok.WEIGHT_TOKENS[-1]}  (100 bins, 0.00-1.00)")
print(f"  LOSS tokens:    {tok.LOSS_TOKENS[0]}..{tok.LOSS_TOKENS[-1]}  (100 bins, 0.00-1.00)")
print(f"  PRED_0={tok.PRED_0_TOKEN}  PRED_1={tok.PRED_1_TOKEN}")
print(f"  STEP tokens:    {tok.STEP_TOKENS[0]}..{tok.STEP_TOKENS[-1]}")

print()
print("=" * 70)
print("TOKENIZED SEQUENCE (first 2 rounds)")
print("=" * 70)
print("Per-round token order:")
print("  STEP -> Expert preds -> SEP [PREDICT HERE] -> True label -> Losses")
print()

encoded = tok.encode_sequence(seq)
ids = encoded['input_ids']
mask = encoded['target_mask']
pred_targets = encoded['prediction_targets']


def token_name(t):
    if t == 0: return "PAD"
    if t == 1: return "START"
    if t == 2: return "END"
    if t == 3: return "SEP"
    if t in tok.EXPERT_TOKENS:
        idx = tok.EXPERT_TOKENS.index(t)
        return f"E{idx}"
    if tok.WEIGHT_TOKENS[0] <= t <= tok.WEIGHT_TOKENS[-1]:
        val = (t - tok.WEIGHT_TOKENS[0]) / 99.0
        return f"W({val:.2f})"
    if tok.LOSS_TOKENS[0] <= t <= tok.LOSS_TOKENS[-1]:
        val = (t - tok.LOSS_TOKENS[0]) / 99.0
        return f"L({val:.2f})"
    if t == tok.PRED_0_TOKEN: return "PRED_0"
    if t == tok.PRED_1_TOKEN: return "PRED_1"
    if t in tok.STEP_TOKENS:
        idx = tok.STEP_TOKENS.index(t)
        return f"STEP_{idx}"
    return f"?{t}"


# Print tokens, stop after 2 rounds worth or END
round_count = 0
for i, tid in enumerate(ids):
    n = token_name(tid)
    marker = ""
    if mask[i]:
        marker = f"  <<<< MODEL PREDICTS HERE (target={int(pred_targets[i])})"
    print(f"  [{i:3d}] {n:12s}  (id={tid:3d}){marker}")

    if n == "END":
        break
    if n.startswith("STEP"):
        round_count += 1
        if round_count > 2:
            print(f"  ... ({seq['n_steps'] - 2} more rounds, then END)")
            break
