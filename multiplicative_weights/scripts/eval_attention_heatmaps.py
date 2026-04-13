#!/usr/bin/env python3
"""
Generate detailed attention heatmaps at multiple points in a sequence,
zoomed to show readable token-level detail.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
import logging

from eval_attention import (
    run_with_attention, classify_tokens, TOKEN_COLORS, load_model,
)
from generate_dataset import generate_single_sequence
from learned_mw_transformer import MWTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


TOKEN_TYPE_SHORT = {
    'START': 'ST', 'STEP': '⌐', 'EXPERT': 'E', 'PRED': 'P',
    'SEP': '|', 'LABEL': 'L', 'LOSS': '$', 'THOUGHT': 'T',
}


def token_label(tid, ttype, tokenizer):
    """Return a human-readable label for a single token."""
    if ttype == 'START':
        return 'START'
    if ttype == 'SEP':
        return 'SEP'
    if ttype == 'THOUGHT':
        return 'THT'
    if ttype == 'EXPERT':
        idx = tokenizer.EXPERT_TOKENS.index(tid)
        return f'E{idx}'
    if ttype in ('PRED', 'LABEL'):
        val = '1' if tid == tokenizer.PRED_1_TOKEN else '0'
        prefix = 'P' if ttype == 'PRED' else 'LBL'
        return f'{prefix}{val}'
    if ttype == 'LOSS':
        idx = tid - tokenizer.LOSS_TOKENS[0]
        return f'L{idx}'
    if ttype == 'STEP':
        idx = tid - tokenizer.STEP_TOKENS[0]
        return f'S{idx}'
    return '?'


def make_labels(ctx_ids, token_types, tokenizer, seq_len):
    """Build tick labels and colors for a range of token positions."""
    labels = []
    colors = []
    for i in range(seq_len):
        if i < len(token_types) and i < len(ctx_ids):
            ttype = token_types[i]
            labels.append(token_label(ctx_ids[i], ttype, tokenizer))
            colors.append(TOKEN_COLORS.get(ttype, '#000'))
        else:
            labels.append('THT')
            colors.append(TOKEN_COLORS.get('THOUGHT', '#e377c2'))
    return labels, colors


def plot_zoomed_heatmap(step_attentions, tokenizer, save_dir, step_idx,
                        zoom_last_n=80, mw_step_label=None):
    """Plot attention heatmap zoomed to the last `zoom_last_n` tokens."""
    ctx_ids, all_attention = step_attentions[step_idx]
    final_attn = all_attention[-1]
    n_layers = len(final_attn)
    n_heads = final_attn[0].shape[1]

    n_thought = final_attn[0].shape[2] - len(ctx_ids)
    token_types = classify_tokens(ctx_ids, tokenizer, n_thought_steps=max(0, n_thought))

    for layer_idx in range(n_layers):
        attn = final_attn[layer_idx][0].cpu().numpy()  # [heads, seq, seq]
        seq_len = attn.shape[1]

        # Zoom range: last zoom_last_n tokens (or full if shorter)
        start = max(0, seq_len - zoom_last_n)
        end = seq_len

        fig, axes = plt.subplots(1, n_heads, figsize=(7 * n_heads, 7))
        if n_heads == 1:
            axes = [axes]

        for head_idx in range(n_heads):
            ax = axes[head_idx]
            # Show attention from zoomed query range to ALL keys, but zoom keys too
            zoomed = attn[head_idx, start:end, start:end]
            im = ax.imshow(zoomed, cmap='hot', aspect='auto', interpolation='nearest',
                           vmin=0, vmax=min(0.3, zoomed.max() + 0.01))
            ax.set_title(f'Head {head_idx}', fontsize=11)

            # Token type labels on axes
            n_ticks = end - start
            if n_ticks <= 100:
                # Build labels for zoomed range
                zoom_ids = ctx_ids[start:end] if start < len(ctx_ids) else []
                zoom_types = token_types[start:min(end, len(token_types))]
                # Pad with thought tokens if needed
                while len(zoom_ids) < n_ticks:
                    zoom_ids.append(-1)
                while len(zoom_types) < n_ticks:
                    zoom_types.append('THOUGHT')
                tick_labels, tick_colors = make_labels(
                    zoom_ids, zoom_types, tokenizer, n_ticks
                )

                fontsize = max(3, min(6, 350 // n_ticks))
                ax.set_xticks(range(n_ticks))
                ax.set_xticklabels(tick_labels, fontsize=fontsize, rotation=90)
                ax.set_yticks(range(n_ticks))
                ax.set_yticklabels(tick_labels, fontsize=fontsize)
                for tick, color in zip(ax.get_xticklabels(), tick_colors):
                    tick.set_color(color)
                for tick, color in zip(ax.get_yticklabels(), tick_colors):
                    tick.set_color(color)

            if head_idx == 0:
                ax.set_ylabel('Query')
            ax.set_xlabel('Key')

        label = mw_step_label or f'step {step_idx}'
        plt.suptitle(f'Layer {layer_idx} — MW {label} (tokens {start}-{end})',
                     fontsize=13)
        plt.tight_layout()
        fname = f'heatmap_L{layer_idx}_mwstep{step_idx if step_idx >= 0 else len(step_attentions)+step_idx}.png'
        path = os.path.join(save_dir, fname)
        plt.savefig(path, dpi=200, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved {path}")


def plot_full_early_heatmap(step_attentions, tokenizer, save_dir, step_idx,
                            mw_step_label=None):
    """Full (unzoomed) heatmap for an early step where context is small."""
    ctx_ids, all_attention = step_attentions[step_idx]
    final_attn = all_attention[-1]
    n_layers = len(final_attn)
    n_heads = final_attn[0].shape[1]

    n_thought = final_attn[0].shape[2] - len(ctx_ids)
    token_types = classify_tokens(ctx_ids, tokenizer, n_thought_steps=max(0, n_thought))

    for layer_idx in range(n_layers):
        attn = final_attn[layer_idx][0].cpu().numpy()
        seq_len = attn.shape[1]

        fig, axes = plt.subplots(1, n_heads, figsize=(5 * n_heads, 5))
        if n_heads == 1:
            axes = [axes]

        for head_idx in range(n_heads):
            ax = axes[head_idx]
            im = ax.imshow(attn[head_idx], cmap='hot', aspect='auto',
                           interpolation='nearest',
                           vmin=0, vmax=min(0.5, attn[head_idx].max() + 0.01))
            ax.set_title(f'Head {head_idx}', fontsize=11)

            # Detailed labels
            tick_labels, tick_colors = make_labels(
                ctx_ids, token_types, tokenizer, seq_len
            )

            fontsize = max(2, min(6, 350 // seq_len))
            ax.set_xticks(range(seq_len))
            ax.set_xticklabels(tick_labels, fontsize=fontsize, rotation=90)
            ax.set_yticks(range(seq_len))
            ax.set_yticklabels(tick_labels, fontsize=fontsize)
            for tick, color in zip(ax.get_xticklabels(), tick_colors):
                tick.set_color(color)
            for tick, color in zip(ax.get_yticklabels(), tick_colors):
                tick.set_color(color)

            if head_idx == 0:
                ax.set_ylabel('Query')
            ax.set_xlabel('Key')

        label = mw_step_label or f'step {step_idx}'
        plt.suptitle(f'Layer {layer_idx} — MW {label} (full, {seq_len} tokens)',
                     fontsize=13)
        plt.tight_layout()
        fname = f'heatmap_full_L{layer_idx}_mwstep{step_idx}.png'
        path = os.path.join(save_dir, fname)
        plt.savefig(path, dpi=200, bbox_inches='tight')
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
    accuracy = np.mean(np.array(decisions) == np.array(seq['true_labels']))
    logger.info(f"Accuracy: {accuracy:.4f}")

    # Full heatmaps at early steps (small enough to read)
    for step in [2, 5]:
        if step < len(step_attentions):
            logger.info(f"\nFull heatmap at MW step {step}...")
            plot_full_early_heatmap(step_attentions, tokenizer, args.save_dir,
                                    step, mw_step_label=f'step {step}/{args.seq_length}')

    # Zoomed heatmaps at various points (40 tokens = ~2 MW steps of detail)
    for step in [10, 25, 40, -1]:
        actual = step if step >= 0 else len(step_attentions) + step
        if actual < len(step_attentions):
            logger.info(f"\nZoomed heatmap at MW step {actual}...")
            plot_zoomed_heatmap(step_attentions, tokenizer, args.save_dir,
                                step, zoom_last_n=40,
                                mw_step_label=f'step {actual}/{args.seq_length}')

    logger.info(f"\nAll heatmaps saved to {args.save_dir}")


if __name__ == '__main__':
    main()
