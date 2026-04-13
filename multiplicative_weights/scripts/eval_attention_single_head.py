#!/usr/bin/env python3
"""Generate large, readable single-head attention heatmaps with full token labels."""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
import logging

from eval_attention import run_with_attention, classify_tokens, TOKEN_COLORS, load_model
from eval_attention_heatmaps import token_label, make_labels
from generate_dataset import generate_single_sequence

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def plot_single_head(step_attentions, tokenizer, save_dir, step_idx,
                     layer, head, zoom_last_n=None, label_suffix=''):
    ctx_ids, all_attention = step_attentions[step_idx]
    final_attn = all_attention[-1]
    attn = final_attn[layer][0].cpu().numpy()  # [heads, seq, seq]
    seq_len = attn.shape[1]

    n_thought = seq_len - len(ctx_ids)
    token_types = classify_tokens(ctx_ids, tokenizer, n_thought_steps=max(0, n_thought))

    if zoom_last_n and zoom_last_n < seq_len:
        start = seq_len - zoom_last_n
    else:
        start = 0
    end = seq_len
    n = end - start

    mat = attn[head, start:end, start:end]

    # Build full labels
    all_ids = list(ctx_ids) + [-1] * max(0, n_thought)
    all_types = token_types + ['THOUGHT'] * max(0, n_thought - max(0, len(token_types) - len(ctx_ids)))
    while len(all_ids) < seq_len:
        all_ids.append(-1)
    while len(all_types) < seq_len:
        all_types.append('THOUGHT')

    tick_labels = []
    tick_colors = []
    for i in range(start, end):
        ttype = all_types[i] if i < len(all_types) else 'THOUGHT'
        tid = all_ids[i] if i < len(all_ids) else -1
        tick_labels.append(token_label(tid, ttype, tokenizer))
        tick_colors.append(TOKEN_COLORS.get(ttype, '#000'))

    # Size based on tokens
    figsize = max(8, n * 0.25)
    fig, ax = plt.subplots(figsize=(figsize, figsize))

    im = ax.imshow(mat, cmap='hot', aspect='equal', interpolation='nearest',
                   vmin=0, vmax=min(0.4, mat.max() + 0.02))
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fontsize = max(5, min(9, 400 // n))
    ax.set_xticks(range(n))
    ax.set_xticklabels(tick_labels, fontsize=fontsize, rotation=90)
    ax.set_yticks(range(n))
    ax.set_yticklabels(tick_labels, fontsize=fontsize)
    for tick, color in zip(ax.get_xticklabels(), tick_colors):
        tick.set_color(color)
    for tick, color in zip(ax.get_yticklabels(), tick_colors):
        tick.set_color(color)

    ax.set_xlabel('Key', fontsize=12)
    ax.set_ylabel('Query', fontsize=12)

    actual_step = step_idx if step_idx >= 0 else len(step_attentions) + step_idx
    ax.set_title(f'Layer {layer}, Head {head} — MW step {actual_step} '
                 f'(tokens {start}-{end})', fontsize=13)

    plt.tight_layout()
    fname = f'single_L{layer}H{head}_step{actual_step}{label_suffix}.png'
    path = os.path.join(save_dir, fname)
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str,
                        default='../figures/checkpoints/model_stage_13.pt')
    parser.add_argument('--seq_length', type=int, default=50)
    parser.add_argument('--n_experts', type=int, default=4)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--save_dir', type=str,
                        default='../figures/attention-figures/heatmaps')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    os.makedirs(args.save_dir, exist_ok=True)

    model, tokenizer, cot_mode, model_config = load_model(args.checkpoint, device)
    seq = generate_single_sequence(args.n_experts, args.seq_length)
    logger.info(f"Sequence: {args.seq_length} steps, {args.n_experts} experts")

    decisions, step_attentions, _, _, _ = run_with_attention(
        model, seq, tokenizer, device, max_ctx=model_config.max_sequence_length
    )
    logger.info(f"Accuracy: {np.mean(np.array(decisions) == np.array(seq['true_labels'])):.4f}")

    # Full view at early step (step 5 = ~109 tokens)
    for layer in [0, 1]:
        for head in [0, 1, 2, 3]:
            plot_single_head(step_attentions, tokenizer, args.save_dir,
                             step_idx=5, layer=layer, head=head)

    # Zoomed view at late steps
    for step_idx in [25, -1]:
        for layer in [0, 1]:
            for head in [0, 1, 2, 3]:
                plot_single_head(step_attentions, tokenizer, args.save_dir,
                                 step_idx=step_idx, layer=layer, head=head,
                                 zoom_last_n=40, label_suffix='_zoom40')

    logger.info(f"\nAll single-head heatmaps saved to {args.save_dir}")


if __name__ == '__main__':
    main()
