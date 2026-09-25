#!/usr/bin/env python3
"""
Expert-focused attention heatmap: shows how much attention each head pays
to each expert's tokens at every MW step, with the best expert and true
label annotated.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import argparse
import logging

from eval_attention import (
    run_with_attention, classify_tokens, get_expert_token_mask,
    load_model,
)
from generate_dataset import generate_single_sequence

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def compute_per_expert_attention(step_attentions, tokenizer,
                                 include_pred=True, normalize=False):
    """
    For each MW step, compute attention from the final query position to
    each expert, per layer and head.

    Args:
        include_pred: if True, include the PRED token next to each EXPERT id
                      (default behaviour). If False, restrict to EXPERT id
                      tokens only.
        normalize: if True, normalize so the 4 experts sum to 1 (per head,
                   per layer, per step). This shows *relative* preference
                   among experts, removing dilution from other token types.

    Returns:
        expert_attn: [n_steps, n_layers, n_heads, n_experts]
    """
    n_experts = tokenizer.n_experts
    all_steps = []

    for step_idx, (ctx_ids, all_attention) in enumerate(step_attentions):
        final_pass = all_attention[-1]  # final CoT pass
        n_layers = len(final_pass)
        n_heads = final_pass[0].shape[1]

        n_thought = final_pass[0].shape[2] - len(ctx_ids)
        token_types = classify_tokens(ctx_ids, tokenizer,
                                      n_thought_steps=max(0, n_thought))

        # Build positions per expert
        expert_positions = {e: [] for e in range(n_experts)}
        for i, (ttype, tid) in enumerate(zip(token_types, ctx_ids)):
            if ttype == 'EXPERT' and tid in tokenizer.EXPERT_TOKENS:
                eidx = tokenizer.EXPERT_TOKENS.index(tid)
                expert_positions[eidx].append(i)
                if include_pred and i + 1 < len(ctx_ids):
                    expert_positions[eidx].append(i + 1)

        step_data = np.zeros((n_layers, n_heads, n_experts))
        for layer_idx in range(n_layers):
            attn = final_pass[layer_idx][0].cpu().numpy()  # [heads, seq, seq]
            last_row = attn[:, -1, :]  # [heads, seq]
            for expert_idx in range(n_experts):
                positions = expert_positions.get(expert_idx, [])
                if positions:
                    step_data[layer_idx, :, expert_idx] = last_row[:, positions].sum(axis=1)

        if normalize:
            # Normalize across experts so each row (layer, head) sums to 1
            totals = step_data.sum(axis=-1, keepdims=True)
            totals = np.where(totals < 1e-9, 1.0, totals)
            step_data = step_data / totals

        all_steps.append(step_data)

    return np.array(all_steps)  # [n_steps, n_layers, n_heads, n_experts]


def plot_expert_focus_heatmap(expert_attn, seq, save_dir, suffix='', title_extra=''):
    """
    Plot heatmaps: rows = MW steps, columns = experts.
    One subplot per (layer, head). Best expert starred, true label shown.
    """
    n_steps, n_layers, n_heads, n_experts = expert_attn.shape

    # Identify best expert
    expert_cum = np.zeros(n_experts)
    for losses in seq['losses']:
        expert_cum += np.array(losses)
    best_e = np.argmin(expert_cum)

    true_labels = seq['true_labels']

    fig, axes = plt.subplots(n_layers, n_heads, figsize=(4 * n_heads + 2, 0.25 * n_steps + 3))
    if n_layers == 1 and n_heads == 1:
        axes = np.array([[axes]])
    elif n_layers == 1:
        axes = axes[np.newaxis, :]
    elif n_heads == 1:
        axes = axes[:, np.newaxis]

    expert_labels = [f'E{i}' for i in range(n_experts)]
    expert_labels[best_e] += ' ★'

    for layer in range(n_layers):
        for head in range(n_heads):
            ax = axes[layer, head]
            data = expert_attn[:, layer, head, :]  # [n_steps, n_experts]
            im = ax.imshow(data, aspect='auto', cmap='YlOrRd',
                           interpolation='nearest', vmin=0,
                           vmax=max(0.05, data.max()))

            # Expert labels on x-axis
            ax.set_xticks(range(n_experts))
            ax.set_xticklabels(expert_labels, fontsize=9)
            # Color the best expert label
            for tick_idx, tick in enumerate(ax.get_xticklabels()):
                if tick_idx == best_e:
                    tick.set_color('green')
                    tick.set_fontweight('bold')

            # Y-axis: MW steps
            step_labels = []
            for s in range(n_steps):
                lbl = f'{s}'
                step_labels.append(lbl)

            if n_steps <= 60:
                ax.set_yticks(range(n_steps))
                ax.set_yticklabels(step_labels, fontsize=5)
            else:
                tick_positions = list(range(0, n_steps, max(1, n_steps // 20)))
                ax.set_yticks(tick_positions)
                ax.set_yticklabels([step_labels[i] for i in tick_positions], fontsize=6)

            ax.set_title(f'L{layer} H{head}', fontsize=11)
            if head == 0:
                ax.set_ylabel('MW Step')
            if layer == n_layers - 1:
                ax.set_xlabel('Expert')


    # Add a colorbar
    cbar_ax = fig.add_axes([0.93, 0.15, 0.015, 0.7])
    fig.colorbar(im, cax=cbar_ax, label='Attention to expert tokens')

    # Add a side strip showing true labels
    # Add annotation text for the best expert
    # Rank experts by cumulative loss (lower = better)
    ranking = np.argsort(expert_cum)
    rank_str = ' > '.join(f'E{e}({expert_cum[e]:.0f})' for e in ranking)
    fig.suptitle(
        f'Attention to Each Expert Over Time{title_extra}\n'
        f'Quality ranking (best→worst): {rank_str}',
        fontsize=12, y=1.0
    )
    plt.subplots_adjust(right=0.91, hspace=0.3, wspace=0.3)

    path = os.path.join(save_dir, f'fig10_expert_focus_heatmap{suffix}.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved {path}")


def plot_expert_attention_lines(expert_attn, seq, save_dir, suffix='', title_extra='', ylabel='Attention mass'):
    """
    Line plot version: per-expert attention over time, one subplot per head.
    Easier to read trends. Best expert in bold.
    """
    n_steps, n_layers, n_heads, n_experts = expert_attn.shape

    expert_cum = np.zeros(n_experts)
    for losses in seq['losses']:
        expert_cum += np.array(losses)
    best_e = np.argmin(expert_cum)
    worst_e = np.argmax(expert_cum)

    colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3']
    steps = np.arange(1, n_steps + 1)

    fig, axes = plt.subplots(n_layers, n_heads, figsize=(5 * n_heads, 4 * n_layers),
                             squeeze=False)

    for layer in range(n_layers):
        for head in range(n_heads):
            ax = axes[layer, head]
            for e in range(n_experts):
                lw = 2.5 if e == best_e else 1.2
                ls = '-' if e == best_e else '--'
                alpha = 1.0 if e == best_e else 0.6
                label = f'E{e}' + (' ★ best' if e == best_e else '')
                ax.plot(steps, expert_attn[:, layer, head, e],
                        color=colors[e], linewidth=lw, linestyle=ls,
                        alpha=alpha, label=label)
            ax.set_title(f'Layer {layer}, Head {head}', fontsize=11)
            ax.set_xlabel('MW Step')
            ax.set_ylabel(ylabel)
            ax.legend(fontsize=7, loc='upper left')
            ax.grid(True, alpha=0.3)

    plt.suptitle(f'Attention to Each Expert Over Time (best: E{best_e}){title_extra}', fontsize=13)
    plt.tight_layout()
    path = os.path.join(save_dir, f'fig10b_expert_attention_lines{suffix}.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved {path}")


def plot_expert_to_expert_matrix(step_attentions, tokenizer, seq, save_dir,
                                 step_idx=-1, normalize=True):
    """
    At a given step, compute a square (n_experts x n_experts) attention matrix
    where entry (i, j) is the average attention from positions of expert E_i
    (queries) to positions of expert E_j (keys), aggregated across all
    occurrences in the context.

    One subplot per (layer, head). Blue-black colormap.
    """
    n_experts = tokenizer.n_experts
    ctx_ids, all_attention = step_attentions[step_idx]
    final_pass = all_attention[-1]
    n_layers = len(final_pass)
    n_heads = final_pass[0].shape[1]

    n_thought = final_pass[0].shape[2] - len(ctx_ids)
    token_types = classify_tokens(ctx_ids, tokenizer,
                                  n_thought_steps=max(0, n_thought))

    # Positions of each expert ID token (no PRED)
    expert_positions = {e: [] for e in range(n_experts)}
    for i, (ttype, tid) in enumerate(zip(token_types, ctx_ids)):
        if ttype == 'EXPERT' and tid in tokenizer.EXPERT_TOKENS:
            eidx = tokenizer.EXPERT_TOKENS.index(tid)
            expert_positions[eidx].append(i)

    # Identify best expert
    expert_cum = np.zeros(n_experts)
    for losses in seq['losses']:
        expert_cum += np.array(losses)
    best_e = int(np.argmin(expert_cum))
    ranking = np.argsort(expert_cum)
    rank_str = ' > '.join(f'E{e}({expert_cum[e]:.0f})' for e in ranking)

    expert_labels = [f'E{i}' for i in range(n_experts)]
    expert_labels[best_e] += ' ★'

    # Build colormap: white -> black via deep blue
    from matplotlib.colors import LinearSegmentedColormap
    blue_black = LinearSegmentedColormap.from_list(
        'blue_black', ['#ffffff', '#a6c8ff', '#1f3b73', '#000000']
    )

    fig, axes = plt.subplots(n_layers, n_heads,
                             figsize=(2.5 * n_heads + 1, 2.5 * n_layers + 1.5),
                             squeeze=False)

    for layer in range(n_layers):
        for head in range(n_heads):
            ax = axes[layer, head]
            attn = final_pass[layer][0, head].cpu().numpy()  # [seq, seq]

            mat = np.zeros((n_experts, n_experts))
            for qi in range(n_experts):
                qpos = expert_positions[qi]
                if not qpos:
                    continue
                for kj in range(n_experts):
                    kpos = expert_positions[kj]
                    if not kpos:
                        continue
                    # Average attention from each qpos to all kpos, then mean over qpos
                    sub = attn[np.ix_(qpos, kpos)]  # [|qpos|, |kpos|]
                    # Sum across keys (total attention from each q-position to this expert),
                    # then mean across q-positions
                    mat[qi, kj] = sub.sum(axis=1).mean()

            if normalize:
                row_sum = mat.sum(axis=1, keepdims=True)
                row_sum = np.where(row_sum < 1e-9, 1.0, row_sum)
                mat = mat / row_sum

            im = ax.imshow(mat, cmap=blue_black, vmin=0,
                           vmax=max(1e-3, mat.max()),
                           aspect='equal', interpolation='nearest')

            ax.set_xticks(range(n_experts))
            ax.set_yticks(range(n_experts))
            ax.set_xticklabels(expert_labels, fontsize=8)
            ax.set_yticklabels(expert_labels, fontsize=8)
            for tick_idx, tick in enumerate(ax.get_xticklabels()):
                if tick_idx == best_e:
                    tick.set_color('green'); tick.set_fontweight('bold')
            for tick_idx, tick in enumerate(ax.get_yticklabels()):
                if tick_idx == best_e:
                    tick.set_color('green'); tick.set_fontweight('bold')

            # Annotate cell values
            for i in range(n_experts):
                for j in range(n_experts):
                    val = mat[i, j]
                    txt_color = 'white' if val > 0.5 * mat.max() else 'black'
                    ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                            fontsize=7, color=txt_color)

            ax.set_title(f'L{layer} H{head}', fontsize=10)
            if layer == n_layers - 1:
                ax.set_xlabel('Key (attended to)', fontsize=9)
            if head == 0:
                ax.set_ylabel('Query (attending from)', fontsize=9)

    actual_step = step_idx if step_idx >= 0 else len(step_attentions) + step_idx
    norm_str = ' (row-normalized)' if normalize else ' (raw)'
    fig.suptitle(
        f'Expert\u2192Expert Attention at MW step {actual_step}{norm_str}\n'
        f'Quality ranking: {rank_str}',
        fontsize=12
    )
    plt.tight_layout()
    suffix = '_norm' if normalize else '_raw'
    path = os.path.join(save_dir, f'fig11_expert_to_expert{suffix}.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved {path}")


def plot_category_attention_matrix(step_attentions, tokenizer, seq, save_dir,
                                   step_idx=-1, normalize=True):
    """
    Square attention matrix where rows/cols are token *categories*:
        E0, E1, E2, E3 (each expert separate), PRED, LABEL, LOSS, SEP, STEP, THOUGHT
    Cell (i, j) = mean attention from a token of category i to all tokens of
    category j (averaged across query positions, summed across key positions).
    Blue-black colormap.
    """
    n_experts = tokenizer.n_experts
    ctx_ids, all_attention = step_attentions[step_idx]
    final_pass = all_attention[-1]
    n_layers = len(final_pass)
    n_heads = final_pass[0].shape[1]

    seq_len_total = final_pass[0].shape[2]
    n_thought = seq_len_total - len(ctx_ids)
    token_types = classify_tokens(ctx_ids, tokenizer,
                                  n_thought_steps=max(0, n_thought))

    # Build category positions
    categories = [f'E{i}' for i in range(n_experts)] + \
                 ['PRED', 'LABEL', 'LOSS', 'SEP', 'STEP', 'THOUGHT']
    cat_positions = {c: [] for c in categories}

    for i, ttype in enumerate(token_types):
        if i >= seq_len_total:
            break
        if ttype == 'EXPERT' and i < len(ctx_ids):
            tid = ctx_ids[i]
            if tid in tokenizer.EXPERT_TOKENS:
                eidx = tokenizer.EXPERT_TOKENS.index(tid)
                cat_positions[f'E{eidx}'].append(i)
        elif ttype in cat_positions:
            cat_positions[ttype].append(i)

    # Drop empty categories
    active_cats = [c for c in categories if cat_positions[c]]

    # Best expert
    expert_cum = np.zeros(n_experts)
    for losses in seq['losses']:
        expert_cum += np.array(losses)
    best_e = int(np.argmin(expert_cum))
    ranking = np.argsort(expert_cum)
    rank_str = ' > '.join(f'E{e}({expert_cum[e]:.0f})' for e in ranking)

    cat_labels = list(active_cats)
    best_label = f'E{best_e}'

    # Blue-black colormap
    from matplotlib.colors import LinearSegmentedColormap
    blue_black = LinearSegmentedColormap.from_list(
        'blue_black', ['#ffffff', '#a6c8ff', '#1f3b73', '#000000']
    )

    n = len(active_cats)
    fig, axes = plt.subplots(n_layers, n_heads,
                             figsize=(3.2 * n_heads + 1, 3.2 * n_layers + 1.5),
                             squeeze=False)

    for layer in range(n_layers):
        for head in range(n_heads):
            ax = axes[layer, head]
            attn = final_pass[layer][0, head].cpu().numpy()  # [seq, seq]

            mat = np.zeros((n, n))
            for qi, qcat in enumerate(active_cats):
                qpos = cat_positions[qcat]
                for kj, kcat in enumerate(active_cats):
                    kpos = cat_positions[kcat]
                    sub = attn[np.ix_(qpos, kpos)]  # [|qpos|, |kpos|]
                    # For each query, sum attention across keys in this category;
                    # then average across queries
                    mat[qi, kj] = sub.sum(axis=1).mean()

            if normalize:
                row_sum = mat.sum(axis=1, keepdims=True)
                row_sum = np.where(row_sum < 1e-9, 1.0, row_sum)
                mat = mat / row_sum

            vmax = max(1e-3, mat.max())
            ax.imshow(mat, cmap=blue_black, vmin=0, vmax=vmax,
                      aspect='equal', interpolation='nearest')

            ax.set_xticks(range(n))
            ax.set_yticks(range(n))
            ax.set_xticklabels(cat_labels, fontsize=7, rotation=45, ha='right')
            ax.set_yticklabels(cat_labels, fontsize=7)
            for tick_idx, tick in enumerate(ax.get_xticklabels()):
                if cat_labels[tick_idx] == best_label:
                    tick.set_color('green'); tick.set_fontweight('bold')
            for tick_idx, tick in enumerate(ax.get_yticklabels()):
                if cat_labels[tick_idx] == best_label:
                    tick.set_color('green'); tick.set_fontweight('bold')

            # Annotate cells
            for i in range(n):
                for j in range(n):
                    val = mat[i, j]
                    if val < 1e-3:
                        continue
                    txt_color = 'white' if val > 0.5 * vmax else 'black'
                    ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                            fontsize=5.5, color=txt_color)

            ax.set_title(f'L{layer} H{head}', fontsize=10)
            if layer == n_layers - 1:
                ax.set_xlabel('Key (attended to)', fontsize=8)
            if head == 0:
                ax.set_ylabel('Query (attending from)', fontsize=8)

    actual_step = step_idx if step_idx >= 0 else len(step_attentions) + step_idx
    norm_str = ' (row-normalized)' if normalize else ' (raw)'
    fig.suptitle(
        f'Token-Category Attention Matrix at MW step {actual_step}{norm_str}\n'
        f'Quality ranking: {rank_str}',
        fontsize=12
    )
    plt.tight_layout()
    suffix = '_norm' if normalize else '_raw'
    path = os.path.join(save_dir, f'fig12_category_attention{suffix}.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved {path}")


def plot_thought_attention(step_attentions, tokenizer, seq, save_dir,
                           step_idx=-1, normalize=True):
    """
    Focused on the latent thought tokens. Each of the K thought tokens gets its
    OWN query row and key column, so we can see:
      - How much each thought attends to earlier thoughts (latent recurrence)
      - How each thought distributes attention over context categories
      - How much attention from later thoughts goes to earlier thoughts vs ctx

    Rows/cols: E0, E1, E2, E3, PRED, LABEL, LOSS, SEP, STEP, T0, T1, T2, ...
    """
    n_experts = tokenizer.n_experts
    ctx_ids, all_attention = step_attentions[step_idx]
    final_pass = all_attention[-1]
    n_layers = len(final_pass)
    n_heads = final_pass[0].shape[1]

    seq_len_total = final_pass[0].shape[2]
    n_thought = seq_len_total - len(ctx_ids)
    token_types = classify_tokens(ctx_ids, tokenizer,
                                  n_thought_steps=max(0, n_thought))

    # Categories for context tokens (NOT THOUGHT — thoughts are separate per-token)
    ctx_categories = [f'E{i}' for i in range(n_experts)] + \
                     ['PRED', 'LABEL', 'LOSS', 'SEP', 'STEP']
    cat_positions = {c: [] for c in ctx_categories}
    for i, ttype in enumerate(token_types):
        if i >= len(ctx_ids):
            break
        if ttype == 'EXPERT':
            tid = ctx_ids[i]
            if tid in tokenizer.EXPERT_TOKENS:
                eidx = tokenizer.EXPERT_TOKENS.index(tid)
                cat_positions[f'E{eidx}'].append(i)
        elif ttype in cat_positions:
            cat_positions[ttype].append(i)

    # Each thought is its own category at a single position
    thought_positions = list(range(len(ctx_ids), seq_len_total))
    thought_labels = [f'T{i}' for i in range(len(thought_positions))]

    # Build full label list and category->positions dict
    all_labels = ctx_categories + thought_labels
    all_positions = dict(cat_positions)
    for tlabel, tpos in zip(thought_labels, thought_positions):
        all_positions[tlabel] = [tpos]

    # Drop empty categories
    active = [c for c in all_labels if all_positions[c]]

    # Best expert
    expert_cum = np.zeros(n_experts)
    for losses in seq['losses']:
        expert_cum += np.array(losses)
    best_e = int(np.argmin(expert_cum))
    ranking = np.argsort(expert_cum)
    rank_str = ' > '.join(f'E{e}({expert_cum[e]:.0f})' for e in ranking)

    best_label = f'E{best_e}'

    # Blue-black colormap
    from matplotlib.colors import LinearSegmentedColormap
    blue_black = LinearSegmentedColormap.from_list(
        'blue_black', ['#ffffff', '#a6c8ff', '#1f3b73', '#000000']
    )

    n = len(active)
    fig, axes = plt.subplots(n_layers, n_heads,
                             figsize=(3.6 * n_heads + 1, 3.6 * n_layers + 1.5),
                             squeeze=False)

    for layer in range(n_layers):
        for head in range(n_heads):
            ax = axes[layer, head]
            attn = final_pass[layer][0, head].cpu().numpy()  # [seq, seq]

            mat = np.zeros((n, n))
            for qi, qcat in enumerate(active):
                qpos = all_positions[qcat]
                for kj, kcat in enumerate(active):
                    kpos = all_positions[kcat]
                    sub = attn[np.ix_(qpos, kpos)]
                    mat[qi, kj] = sub.sum(axis=1).mean()

            if normalize:
                row_sum = mat.sum(axis=1, keepdims=True)
                row_sum = np.where(row_sum < 1e-9, 1.0, row_sum)
                mat = mat / row_sum

            vmax = max(1e-3, mat.max())
            ax.imshow(mat, cmap=blue_black, vmin=0, vmax=vmax,
                      aspect='equal', interpolation='nearest')

            ax.set_xticks(range(n))
            ax.set_yticks(range(n))
            ax.set_xticklabels(active, fontsize=6.5, rotation=45, ha='right')
            ax.set_yticklabels(active, fontsize=6.5)
            for tick_idx, tick in enumerate(ax.get_xticklabels()):
                lbl = active[tick_idx]
                if lbl == best_label:
                    tick.set_color('green'); tick.set_fontweight('bold')
                elif lbl.startswith('T'):
                    tick.set_color('purple'); tick.set_fontweight('bold')
            for tick_idx, tick in enumerate(ax.get_yticklabels()):
                lbl = active[tick_idx]
                if lbl == best_label:
                    tick.set_color('green'); tick.set_fontweight('bold')
                elif lbl.startswith('T'):
                    tick.set_color('purple'); tick.set_fontweight('bold')

            for i in range(n):
                for j in range(n):
                    val = mat[i, j]
                    if val < 5e-3:
                        continue
                    txt_color = 'white' if val > 0.5 * vmax else 'black'
                    ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                            fontsize=5, color=txt_color)

            ax.set_title(f'L{layer} H{head}', fontsize=10)
            if layer == n_layers - 1:
                ax.set_xlabel('Key', fontsize=8)
            if head == 0:
                ax.set_ylabel('Query', fontsize=8)

    actual_step = step_idx if step_idx >= 0 else len(step_attentions) + step_idx
    norm_str = ' (row-normalized)' if normalize else ' (raw mass)'
    fig.suptitle(
        f'Per-Thought Attention at MW step {actual_step}{norm_str}\n'
        f'Quality ranking: {rank_str} | T0..T{len(thought_positions)-1} = latent thoughts (purple)',
        fontsize=11
    )
    plt.tight_layout()
    suffix = '_norm' if normalize else '_raw'
    path = os.path.join(save_dir, f'fig13_thought_attention{suffix}.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved {path}")


def plot_token_square_heatmap(step_attentions, tokenizer, seq, save_dir,
                              step_idx=-1, last_n=60, keep_thoughts=None,
                              cmap=None, cmap_name='blueblack', tag='',
                              drop_expert_ids=False):
    """
    Square token-by-token attention heatmap at the final MW step,
    zoomed to the last `last_n` positions. Each token gets a detailed label
    (E0, P_E0=expert 0's prediction, L_E0=expert 0's loss, LABEL, SEP, STEP, T0..).
    One subplot per (layer, head).

    Args:
        keep_thoughts: int or None. If int, only show the first `keep_thoughts`
            thought positions (e.g. keep_thoughts=1 shows just T0).
        cmap: matplotlib colormap. Defaults to a custom blue->black gradient.
        cmap_name: tag added to filenames to distinguish color schemes.
    """
    n_experts = tokenizer.n_experts
    ctx_ids, all_attention = step_attentions[step_idx]
    final_pass = all_attention[-1]
    n_layers = len(final_pass)
    n_heads = final_pass[0].shape[1]
    seq_len = final_pass[0].shape[2]
    n_thought = seq_len - len(ctx_ids)
    ttypes = classify_tokens(ctx_ids, tokenizer, n_thought_steps=max(0, n_thought))

    # Truncate to keep only the first `keep_thoughts` thought tokens
    if keep_thoughts is not None and keep_thoughts < n_thought:
        seq_len = len(ctx_ids) + keep_thoughts
        ttypes = ttypes[:seq_len]

    # Build detailed per-token labels
    def detailed_label(i):
        if i >= len(ctx_ids):
            return f'T{i - len(ctx_ids)}'
        t = ttypes[i]; tid = ctx_ids[i]
        if t == 'EXPERT':
            return f'E{tokenizer.EXPERT_TOKENS.index(tid)}'
        if t == 'PRED':
            # Look back to find what kind of PRED: after EXPERT (P_Ek) or after SEP (LABEL_OUT)
            for k in range(i - 1, -1, -1):
                if ttypes[k] == 'EXPERT':
                    return f'P{tokenizer.EXPERT_TOKENS.index(ctx_ids[k])}'
                if ttypes[k] == 'SEP':
                    return 'LBLO'
                if ttypes[k] in ('STEP', 'LOSS'):
                    break
            return 'P?'
        if t == 'LOSS':
            for k in range(i - 1, -1, -1):
                if ttypes[k] == 'EXPERT':
                    return f'L{tokenizer.EXPERT_TOKENS.index(ctx_ids[k])}'
                if ttypes[k] in ('SEP', 'STEP', 'PRED'):
                    break
            return 'L?'
        if t == 'STEP':
            return 'STP'
        if t == 'SEP':
            return 'SEP'
        if t == 'LABEL':
            return 'LBL'
        if t == 'START':
            return 'STR'
        return t[:3]

    labels_full = [detailed_label(i) for i in range(seq_len)]

    # Optionally drop EXPERT-id token positions (their attention is ~0 in L1)
    if drop_expert_ids:
        keep_idx = np.array([i for i in range(seq_len)
                             if not (i < len(ctx_ids) and ttypes[i] == 'EXPERT')],
                            dtype=int)
    else:
        keep_idx = np.arange(seq_len, dtype=int)

    labels = [labels_full[i] for i in keep_idx]

    # Restrict to last_n entries (in the filtered list)
    start = max(0, len(labels) - last_n)
    sub_labels = labels[start:]
    n_sub = len(sub_labels)
    sub_idx = keep_idx[start:]  # original positions corresponding to sub_labels

    # Best expert
    expert_cum = np.zeros(n_experts)
    for losses in seq['losses']:
        expert_cum += np.array(losses)
    best_e = int(np.argmin(expert_cum))
    ranking = np.argsort(expert_cum)
    rank_str = ' > '.join(f'E{e}({expert_cum[e]:.0f})' for e in ranking)

    if cmap is None:
        from matplotlib.colors import LinearSegmentedColormap
        cmap = LinearSegmentedColormap.from_list(
            'blue_black', ['#ffffff', '#a6c8ff', '#1f3b73', '#000000']
        )

    # Big figure so labels are readable: ~0.25 inch per cell
    cell = 0.22
    fig, axes = plt.subplots(n_layers, n_heads,
                             figsize=(cell * n_sub * n_heads + 2,
                                      cell * n_sub * n_layers + 2),
                             squeeze=False)

    for layer in range(n_layers):
        for head in range(n_heads):
            ax = axes[layer, head]
            attn = final_pass[layer][0, head].cpu().numpy()
            sub = attn[np.ix_(sub_idx, sub_idx)]
            # vmax = 99th percentile to keep dynamic range usable
            vmax = max(1e-3, np.quantile(sub, 0.995))
            ax.imshow(sub, cmap=cmap, vmin=0, vmax=vmax,
                      aspect='equal', interpolation='nearest')

            # Tick labels
            ax.set_xticks(range(n_sub))
            ax.set_yticks(range(n_sub))
            ax.set_xticklabels(sub_labels, fontsize=7, rotation=90)
            ax.set_yticklabels(sub_labels, fontsize=7)

            # Color tick labels
            best_pred = f'P{best_e}'; best_e_lbl = f'E{best_e}'
            for tick_idx, tick in enumerate(ax.get_xticklabels()):
                lbl = sub_labels[tick_idx]
                if lbl == best_e_lbl or lbl == best_pred:
                    tick.set_color('green'); tick.set_fontweight('bold')
                elif lbl.startswith('T'):
                    tick.set_color('purple'); tick.set_fontweight('bold')
            for tick_idx, tick in enumerate(ax.get_yticklabels()):
                lbl = sub_labels[tick_idx]
                if lbl == best_e_lbl or lbl == best_pred:
                    tick.set_color('green'); tick.set_fontweight('bold')
                elif lbl.startswith('T'):
                    tick.set_color('purple'); tick.set_fontweight('bold')

            ax.set_title(f'L{layer} H{head}', fontsize=11)
            if layer == n_layers - 1:
                ax.set_xlabel('Key', fontsize=9)
            if head == 0:
                ax.set_ylabel('Query', fontsize=9)

    actual_step = step_idx if step_idx >= 0 else len(step_attentions) + step_idx
    fig.suptitle(
        f'Token-level Attention Heatmap at MW step {actual_step} '
        f'(last {n_sub} positions)\n'
        f'Quality ranking: {rank_str}  |  '
        f'P{best_e}=best expert prediction (green)  |  T*=latent thoughts (purple)',
        fontsize=11
    )
    plt.tight_layout()
    fname_extra = f'_{cmap_name}{tag}'
    path = os.path.join(save_dir, f'fig14_token_square_last{n_sub}{fname_extra}.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved {path}")

    # Also save individual per-head figures with much larger labels
    best_pred = f'P{best_e}'; best_e_lbl = f'E{best_e}'
    for layer in range(n_layers):
        for head in range(n_heads):
            fig2, ax = plt.subplots(figsize=(max(10, 0.32 * n_sub),
                                             max(10, 0.32 * n_sub)))
            attn = final_pass[layer][0, head].cpu().numpy()
            sub = attn[np.ix_(sub_idx, sub_idx)]
            vmax = max(1e-3, np.quantile(sub, 0.995))
            ax.imshow(sub, cmap=cmap, vmin=0, vmax=vmax,
                      aspect='equal', interpolation='nearest')
            ax.set_xticks(range(n_sub)); ax.set_yticks(range(n_sub))
            ax.set_xticklabels(sub_labels, fontsize=10, rotation=90)
            ax.set_yticklabels(sub_labels, fontsize=10)
            for tick_idx, tick in enumerate(ax.get_xticklabels()):
                lbl = sub_labels[tick_idx]
                if lbl == best_e_lbl or lbl == best_pred:
                    tick.set_color('green'); tick.set_fontweight('bold')
                elif lbl.startswith('T'):
                    tick.set_color('purple'); tick.set_fontweight('bold')
            for tick_idx, tick in enumerate(ax.get_yticklabels()):
                lbl = sub_labels[tick_idx]
                if lbl == best_e_lbl or lbl == best_pred:
                    tick.set_color('green'); tick.set_fontweight('bold')
                elif lbl.startswith('T'):
                    tick.set_color('purple'); tick.set_fontweight('bold')
            ax.set_title(
                f'Token-level Attention L{layer} H{head} at MW step {actual_step}\n'
                f'Best expert: E{best_e} (green) | Thoughts: T* (purple)',
                fontsize=12
            )
            ax.set_xlabel('Key (attended to)', fontsize=11)
            ax.set_ylabel('Query (attending from)', fontsize=11)
            plt.tight_layout()
            sub_path = os.path.join(
                save_dir, f'fig14_token_square_last{n_sub}{fname_extra}_L{layer}H{head}.png'
            )
            plt.savefig(sub_path, dpi=180, bbox_inches='tight')
            plt.close()
    logger.info(f"Saved per-head heatmaps fig14_*_L*H*.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str,
                        default='../figures/checkpoints/model_stage_13.pt')
    parser.add_argument('--seq_length', type=int, default=50)
    parser.add_argument('--n_experts', type=int, default=4)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--save_dir', type=str,
                        default='../figures/attention-figures')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--qualities', type=str, default=None,
                        help='Comma-separated expert qualities, e.g. "0.15,0.9,0.2,0.18"')
    parser.add_argument('--tag', type=str, default='',
                        help='Suffix appended to output filenames')
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    os.makedirs(args.save_dir, exist_ok=True)

    model, tokenizer, cot_mode, model_config = load_model(args.checkpoint, device)
    seq = generate_single_sequence(args.n_experts, args.seq_length)
    logger.info(f"Sequence: {args.seq_length} steps, {args.n_experts} experts")

    expert_cum = np.zeros(args.n_experts)
    for losses in seq['losses']:
        expert_cum += np.array(losses)
    best_e = np.argmin(expert_cum)
    logger.info(f"Best expert: E{best_e} (cum loss {expert_cum[best_e]:.1f})")

    decisions, step_attentions, _, _, _ = run_with_attention(
        model, seq, tokenizer, device, max_ctx=model_config.max_sequence_length
    )
    accuracy = np.mean(np.array(decisions) == np.array(seq['true_labels']))
    logger.info(f"Accuracy: {accuracy:.4f}")

    # 1) Default: attention to expert ID + adjacent PRED token (raw mass)
    expert_attn = compute_per_expert_attention(step_attentions, tokenizer)
    logger.info(f"Expert attention shape: {expert_attn.shape}")
    plot_expert_focus_heatmap(expert_attn, seq, args.save_dir)
    plot_expert_attention_lines(expert_attn, seq, args.save_dir)

    # 2) Restricted to EXPERT id tokens only, normalized across the 4 experts
    #    -> shows relative preference among experts (cleaner signal)
    expert_attn_norm = compute_per_expert_attention(
        step_attentions, tokenizer, include_pred=False, normalize=True
    )
    plot_expert_focus_heatmap(
        expert_attn_norm, seq, args.save_dir,
        suffix='_normalized',
        title_extra=' — expert-ID tokens only, normalized'
    )
    plot_expert_attention_lines(
        expert_attn_norm, seq, args.save_dir,
        suffix='_normalized',
        title_extra=' — expert-ID tokens, normalized',
        ylabel='Relative attention (sums to 1)'
    )

    # 3) Restricted to EXPERT id tokens only, raw (un-normalized)
    expert_attn_idonly = compute_per_expert_attention(
        step_attentions, tokenizer, include_pred=False, normalize=False
    )
    plot_expert_focus_heatmap(
        expert_attn_idonly, seq, args.save_dir,
        suffix='_idonly',
        title_extra=' — expert-ID tokens only (raw)'
    )

    # 4) Square expert -> expert attention matrix at the final step
    plot_expert_to_expert_matrix(step_attentions, tokenizer, seq, args.save_dir,
                                 step_idx=-1, normalize=True)
    plot_expert_to_expert_matrix(step_attentions, tokenizer, seq, args.save_dir,
                                 step_idx=-1, normalize=False)

    # 5) Larger token-category attention matrix at the final step
    plot_category_attention_matrix(step_attentions, tokenizer, seq, args.save_dir,
                                   step_idx=-1, normalize=True)
    plot_category_attention_matrix(step_attentions, tokenizer, seq, args.save_dir,
                                   step_idx=-1, normalize=False)

    # 6) Per-thought attention: each latent thought as its own row/col
    plot_thought_attention(step_attentions, tokenizer, seq, args.save_dir,
                           step_idx=-1, normalize=True)
    plot_thought_attention(step_attentions, tokenizer, seq, args.save_dir,
                           step_idx=-1, normalize=False)

    # 7) Token-level square attention heatmap (zoom on last N positions)
    plot_token_square_heatmap(step_attentions, tokenizer, seq, args.save_dir,
                              step_idx=-1, last_n=60)

    logger.info("Done!")


if __name__ == '__main__':
    main()
