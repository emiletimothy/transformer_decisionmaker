#!/usr/bin/env python3
"""
Attention analysis for the learned MW transformer.

Generates 6 figures:
  1. Per-head attention heatmaps (single sequence, final CoT pass)
  2. Attention by token type per head
  3. Attention to best vs worst expert over sequence time
  4. Attention distance profile per head
  5. Attention evolution across CoT thought steps
  6. Attention entropy / distance over sequence time
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from typing import Dict, List, Tuple
import argparse
import logging

from learned_mw_transformer import (
    ContinuousCoTTransformer, ModelConfig, MWTokenizer,
)
from eval_long_sequences import load_model, get_optimal_mw_decisions
from generate_dataset import generate_single_sequence
from multiplicative_weights import MultiplicativeWeights
from sklearn.decomposition import PCA

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Token type classification
# ---------------------------------------------------------------------------

TOKEN_TYPES = ['START', 'STEP', 'EXPERT', 'PRED', 'SEP', 'LABEL', 'LOSS', 'THOUGHT']
TOKEN_COLORS = {
    'START': '#333333', 'STEP': '#1f77b4', 'EXPERT': '#ff7f0e',
    'PRED': '#2ca02c', 'SEP': '#999999', 'LABEL': '#d62728',
    'LOSS': '#9467bd', 'THOUGHT': '#e377c2',
}


def classify_tokens(token_ids: List[int], tokenizer: MWTokenizer,
                    n_thought_steps: int = 0) -> List[str]:
    """Classify each token position into a semantic type."""
    types = []
    # Track position within each MW step to distinguish PRED (expert pred)
    # from LABEL (true label after SEP).
    after_sep = False
    for i, tid in enumerate(token_ids):
        if tid == tokenizer.START_TOKEN:
            types.append('START')
            after_sep = False
        elif tid == tokenizer.SEP_TOKEN:
            types.append('SEP')
            after_sep = True
        elif tid in tokenizer.STEP_TOKENS:
            types.append('STEP')
            after_sep = False
        elif tid in tokenizer.EXPERT_TOKENS:
            types.append('EXPERT')
        elif tid in (tokenizer.PRED_0_TOKEN, tokenizer.PRED_1_TOKEN):
            if after_sep:
                types.append('LABEL')
                after_sep = False  # only the first PRED after SEP is the label
            else:
                types.append('PRED')
        elif tid in tokenizer.LOSS_TOKENS:
            types.append('LOSS')
        else:
            types.append('STEP')  # fallback
    # Append thought token types
    for _ in range(n_thought_steps):
        types.append('THOUGHT')
    return types


def get_expert_token_mask(token_types: List[str], token_ids: List[int],
                          tokenizer: MWTokenizer) -> Dict[int, List[int]]:
    """Return dict mapping expert_index -> list of token positions associated
    with that expert (expert ID tokens + their adjacent pred/loss tokens)."""
    n_experts = tokenizer.n_experts
    expert_positions = {e: [] for e in range(n_experts)}

    for i, (ttype, tid) in enumerate(zip(token_types, token_ids)):
        if ttype == 'EXPERT' and tid in tokenizer.EXPERT_TOKENS:
            eidx = tokenizer.EXPERT_TOKENS.index(tid)
            expert_positions[eidx].append(i)
            # Next token is the expert's pred or loss
            if i + 1 < len(token_ids):
                expert_positions[eidx].append(i + 1)
    return expert_positions


# ---------------------------------------------------------------------------
# Run model and collect attention
# ---------------------------------------------------------------------------

def run_with_attention(model, seq, tokenizer, device, max_ctx=1024):
    """Run the model on a sequence and collect attention from all CoT passes."""
    model.eval()
    token_ids = [tokenizer.START_TOKEN]

    # We'll collect attention at the decision point of each MW step
    step_attentions = []  # list of (token_ids_at_decision, all_attention)
    step_weights = []     # learned expert weights at each step
    step_logits = []      # raw prediction logits
    step_thought_hiddens = []  # [K, d_model] thought hidden states per step
    decisions = []

    for step in range(len(seq['expert_predictions'])):
        token_ids.append(tokenizer.STEP_TOKENS[step % 100])
        for eidx, pred in enumerate(seq['expert_predictions'][step]):
            token_ids.append(tokenizer.EXPERT_TOKENS[eidx])
            token_ids.append(tokenizer.PRED_1_TOKEN if pred == 1 else tokenizer.PRED_0_TOKEN)

        # Sliding window
        if len(token_ids) > max_ctx:
            ctx_ids = [token_ids[0]] + token_ids[len(token_ids) - max_ctx + 1:]
        else:
            ctx_ids = token_ids

        context_tensor = torch.tensor([ctx_ids], dtype=torch.long, device=device)
        with torch.no_grad():
            embeddings = model.token_embedding(context_tensor)
            weights_out, thought_hiddens, final_h, all_attention = model.think(
                embeddings, return_attention=True
            )
            pred_logit = model.prediction_head(final_h)

        decision = 1 if torch.sigmoid(pred_logit[0, 0]) > 0.5 else 0
        decisions.append(decision)
        step_attentions.append((list(ctx_ids), all_attention))
        step_weights.append(weights_out[0].cpu().numpy())  # [n_experts]
        step_logits.append(pred_logit[0, 0].cpu().item())
        step_thought_hiddens.append(thought_hiddens[0].cpu().numpy())  # [K, d_model]

        # Add feedback tokens
        token_ids.append(tokenizer.SEP_TOKEN)
        true_label = seq['true_labels'][step]
        token_ids.append(tokenizer.PRED_1_TOKEN if true_label == 1 else tokenizer.PRED_0_TOKEN)
        for eidx, loss_val in enumerate(seq['losses'][step]):
            token_ids.append(tokenizer.EXPERT_TOKENS[eidx])
            token_ids.append(tokenizer.discretize_loss(loss_val))

    return decisions, step_attentions, np.array(step_weights), np.array(step_logits), np.array(step_thought_hiddens)


# ---------------------------------------------------------------------------
# Figure 1: Per-head attention heatmaps
# ---------------------------------------------------------------------------

def plot_per_head_heatmaps(step_attentions, tokenizer, save_dir, step_idx=-1):
    """Plot attention heatmaps for all heads at a specific MW step.
    Uses the final CoT pass (thought_step=-1)."""
    ctx_ids, all_attention = step_attentions[step_idx]
    # all_attention: list of (K+1) passes, each is a list of n_layers tensors
    # Each tensor: [1, n_heads, seq_len, seq_len]
    # Use the final pass
    final_pass_attn = all_attention[-1]  # list of n_layers tensors
    n_layers = len(final_pass_attn)
    n_heads = final_pass_attn[0].shape[1]

    token_types = classify_tokens(ctx_ids, tokenizer,
                                  n_thought_steps=final_pass_attn[0].shape[2] - len(ctx_ids))

    fig, axes = plt.subplots(n_layers, n_heads, figsize=(5 * n_heads, 5 * n_layers))
    if n_layers == 1:
        axes = axes[np.newaxis, :]

    for layer_idx in range(n_layers):
        attn = final_pass_attn[layer_idx][0].cpu().numpy()  # [n_heads, seq, seq]
        seq_len = attn.shape[1]
        for head_idx in range(n_heads):
            ax = axes[layer_idx, head_idx]
            im = ax.imshow(attn[head_idx], cmap='hot', aspect='auto',
                           interpolation='nearest')
            ax.set_title(f'Layer {layer_idx}, Head {head_idx}', fontsize=10)
            if layer_idx == n_layers - 1:
                ax.set_xlabel('Key position')
            if head_idx == 0:
                ax.set_ylabel('Query position')

            # Color-code tick labels by token type
            if seq_len <= 80:
                tick_colors = [TOKEN_COLORS.get(t, '#000000')
                               for t in token_types[:seq_len]]
                ax.set_xticks(range(seq_len))
                ax.set_yticks(range(seq_len))
                ax.set_xticklabels(range(seq_len), fontsize=3, rotation=90)
                ax.set_yticklabels(range(seq_len), fontsize=3)
                for tick, color in zip(ax.get_xticklabels(), tick_colors):
                    tick.set_color(color)
                for tick, color in zip(ax.get_yticklabels(), tick_colors):
                    tick.set_color(color)

    plt.suptitle(f'Attention Heatmaps (MW step {step_idx}, final CoT pass)', fontsize=14)
    plt.tight_layout()
    path = os.path.join(save_dir, 'fig1_per_head_heatmaps.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved {path}")


# ---------------------------------------------------------------------------
# Figure 2: Attention by token type
# ---------------------------------------------------------------------------

def plot_attention_by_token_type(step_attentions, tokenizer, save_dir,
                                 n_steps_to_avg=5):
    """Bar chart showing how much attention each head pays to each token type.
    Averages over the last n_steps_to_avg MW steps."""
    # Collect from final CoT pass of the last few steps
    type_names = TOKEN_TYPES
    n_layers = None
    n_heads = None
    accum = None
    count = 0

    for sa_idx in range(max(0, len(step_attentions) - n_steps_to_avg),
                        len(step_attentions)):
        ctx_ids, all_attention = step_attentions[sa_idx]
        final_attn = all_attention[-1]
        if n_layers is None:
            n_layers = len(final_attn)
            n_heads = final_attn[0].shape[1]
            accum = np.zeros((n_layers, n_heads, len(type_names)))

        token_types = classify_tokens(ctx_ids, tokenizer,
                                      n_thought_steps=final_attn[0].shape[2] - len(ctx_ids))

        for layer_idx in range(n_layers):
            attn = final_attn[layer_idx][0].cpu().numpy()  # [heads, seq, seq]
            seq_len = attn.shape[1]
            # For the last query position (decision point), how much attention
            # goes to each token type?
            last_row = attn[:, -1, :]  # [heads, seq]

            for type_idx, tname in enumerate(type_names):
                mask = np.array([1.0 if (i < len(token_types) and token_types[i] == tname)
                                 else 0.0 for i in range(seq_len)])
                accum[layer_idx, :, type_idx] += (last_row * mask[np.newaxis, :]).sum(axis=1)
        count += 1

    accum /= max(count, 1)

    fig, axes = plt.subplots(1, n_layers, figsize=(8 * n_layers, 6))
    if n_layers == 1:
        axes = [axes]

    x = np.arange(len(type_names))
    width = 0.18
    colors = plt.cm.Set2(np.linspace(0, 1, n_heads))

    for layer_idx in range(n_layers):
        ax = axes[layer_idx]
        for head_idx in range(n_heads):
            offset = (head_idx - n_heads / 2 + 0.5) * width
            ax.bar(x + offset, accum[layer_idx, head_idx], width,
                   label=f'Head {head_idx}', color=colors[head_idx], alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(type_names, rotation=30, ha='right')
        ax.set_ylabel('Attention mass')
        ax.set_title(f'Layer {layer_idx}: Attention by Token Type (at decision)')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, 'fig2_attention_by_token_type.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved {path}")


# ---------------------------------------------------------------------------
# Figure 3: Attention to best vs worst expert
# ---------------------------------------------------------------------------

def plot_attention_best_vs_worst_expert(step_attentions, seq, tokenizer, save_dir):
    """Track how much attention flows to the best vs worst expert over time."""
    n_experts = len(seq['expert_predictions'][0])
    expert_cum_losses = np.zeros(n_experts)

    # For each step, compute attention to each expert's tokens
    n_layers = len(step_attentions[0][1][-1])
    n_heads = step_attentions[0][1][-1][0].shape[1]

    # Track attention to best/worst expert per step
    attn_to_best = np.zeros((len(step_attentions), n_layers, n_heads))
    attn_to_worst = np.zeros((len(step_attentions), n_layers, n_heads))

    for step_idx in range(len(step_attentions)):
        if step_idx > 0:
            expert_cum_losses += np.array(seq['losses'][step_idx - 1])

        best_expert = np.argmin(expert_cum_losses)
        worst_expert = np.argmax(expert_cum_losses)

        ctx_ids, all_attention = step_attentions[step_idx]
        final_attn = all_attention[-1]
        token_types = classify_tokens(ctx_ids, tokenizer,
                                      n_thought_steps=final_attn[0].shape[2] - len(ctx_ids))
        expert_positions = get_expert_token_mask(token_types, ctx_ids, tokenizer)

        for layer_idx in range(n_layers):
            attn = final_attn[layer_idx][0].cpu().numpy()
            seq_len = attn.shape[1]
            last_row = attn[:, -1, :]  # [heads, seq]

            for eidx, positions in expert_positions.items():
                pos_mask = np.zeros(seq_len)
                for p in positions:
                    if p < seq_len:
                        pos_mask[p] = 1.0
                expert_attn = (last_row * pos_mask[np.newaxis, :]).sum(axis=1)

                if eidx == best_expert:
                    attn_to_best[step_idx, layer_idx] = expert_attn
                if eidx == worst_expert:
                    attn_to_worst[step_idx, layer_idx] = expert_attn

    steps = np.arange(1, len(step_attentions) + 1)
    fig, axes = plt.subplots(1, n_layers, figsize=(8 * n_layers, 5))
    if n_layers == 1:
        axes = [axes]

    for layer_idx in range(n_layers):
        ax = axes[layer_idx]
        # Average across heads
        best_mean = attn_to_best[:, layer_idx].mean(axis=1)
        worst_mean = attn_to_worst[:, layer_idx].mean(axis=1)
        ax.plot(steps, best_mean, 'g-', linewidth=2, label='Best expert', alpha=0.8)
        ax.plot(steps, worst_mean, 'r-', linewidth=2, label='Worst expert', alpha=0.8)

        # Also show per-head as thin lines
        for h in range(n_heads):
            ax.plot(steps, attn_to_best[:, layer_idx, h], 'g-', alpha=0.15, linewidth=0.5)
            ax.plot(steps, attn_to_worst[:, layer_idx, h], 'r-', alpha=0.15, linewidth=0.5)

        ax.set_xlabel('MW Step')
        ax.set_ylabel('Attention to expert tokens')
        ax.set_title(f'Layer {layer_idx}: Attention to Best vs Worst Expert')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, 'fig3_attention_best_vs_worst_expert.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved {path}")


# ---------------------------------------------------------------------------
# Figure 4: Attention distance profile
# ---------------------------------------------------------------------------

def plot_attention_distance_profile(step_attentions, tokenizer, save_dir,
                                    n_steps_to_avg=5):
    """Average attention distance per head (how far back each head looks)."""
    n_layers = len(step_attentions[0][1][-1])
    n_heads = step_attentions[0][1][-1][0].shape[1]

    # Collect mean distance for each head
    mean_distances = np.zeros((n_layers, n_heads))
    count = 0

    for sa_idx in range(max(0, len(step_attentions) - n_steps_to_avg),
                        len(step_attentions)):
        ctx_ids, all_attention = step_attentions[sa_idx]
        final_attn = all_attention[-1]

        for layer_idx in range(n_layers):
            attn = final_attn[layer_idx][0].cpu().numpy()  # [heads, seq, seq]
            seq_len = attn.shape[1]
            # For each query position, compute weighted average distance to keys
            positions = np.arange(seq_len)
            for h in range(n_heads):
                # Focus on last query position (decision point)
                weights = attn[h, -1, :]  # attention from last position
                key_positions = np.arange(seq_len)
                # Distance = query_pos - key_pos
                distances = (seq_len - 1) - key_positions
                mean_dist = np.sum(weights * distances)
                mean_distances[layer_idx, h] += mean_dist
        count += 1

    mean_distances /= max(count, 1)

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(n_layers * n_heads)
    labels = [f'L{l}H{h}' for l in range(n_layers) for h in range(n_heads)]
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, n_layers * n_heads))

    bars = ax.bar(x, mean_distances.flatten(), color=colors, alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel('Mean attention distance (tokens back)')
    ax.set_title('Attention Distance Profile: How Far Back Each Head Looks')
    ax.grid(True, alpha=0.3)

    # Add value labels
    for bar, val in zip(bars, mean_distances.flatten()):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f'{val:.1f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    path = os.path.join(save_dir, 'fig4_attention_distance_profile.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved {path}")


# ---------------------------------------------------------------------------
# Figure 5: Attention evolution across CoT thought steps
# ---------------------------------------------------------------------------

def plot_attention_across_cot_steps(step_attentions, tokenizer, save_dir,
                                    step_idx=-1):
    """Show how attention patterns change across the K thought recurrence steps
    plus the final pass, for a single MW step."""
    ctx_ids, all_attention = step_attentions[step_idx]
    n_passes = len(all_attention)  # K+1
    n_layers = len(all_attention[0])

    token_types = classify_tokens(ctx_ids, tokenizer, n_thought_steps=0)
    type_names = TOKEN_TYPES

    # For each CoT pass, compute token-type attention from the last query position
    fig, axes = plt.subplots(n_layers, n_passes, figsize=(4 * n_passes, 5 * n_layers))
    if n_layers == 1:
        axes = axes[np.newaxis, :]
    if n_passes == 1:
        axes = axes[:, np.newaxis]

    for pass_idx in range(n_passes):
        pass_attn = all_attention[pass_idx]
        for layer_idx in range(n_layers):
            attn = pass_attn[layer_idx][0].cpu().numpy()  # [heads, seq, seq]
            seq_len = attn.shape[1]
            last_row = attn[:, -1, :]  # [heads, seq]

            # Build token type for this sequence length
            n_thought = seq_len - len(ctx_ids)
            types_here = classify_tokens(ctx_ids, tokenizer,
                                         n_thought_steps=max(0, n_thought))

            # Aggregate by type
            type_attn = np.zeros((attn.shape[0], len(type_names)))
            for ti, tname in enumerate(type_names):
                mask = np.array([1.0 if (i < len(types_here) and types_here[i] == tname)
                                 else 0.0 for i in range(seq_len)])
                type_attn[:, ti] = (last_row * mask[np.newaxis, :]).sum(axis=1)

            ax = axes[layer_idx, pass_idx]
            im = ax.imshow(type_attn, cmap='YlOrRd', aspect='auto',
                           interpolation='nearest')
            ax.set_yticks(range(type_attn.shape[0]))
            ax.set_yticklabels([f'Head {h}' for h in range(type_attn.shape[0])],
                               fontsize=8)
            ax.set_xticks(range(len(type_names)))
            ax.set_xticklabels(type_names, rotation=45, ha='right', fontsize=8)
            if pass_idx < n_passes - 1:
                ax.set_title(f'L{layer_idx} | Thought {pass_idx}', fontsize=9)
            else:
                ax.set_title(f'L{layer_idx} | Final pass', fontsize=9)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.suptitle('Attention by Token Type Across CoT Steps', fontsize=14)
    plt.tight_layout()
    path = os.path.join(save_dir, 'fig5_attention_across_cot_steps.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved {path}")


# ---------------------------------------------------------------------------
# Figure 6: Attention entropy/distance over sequence time
# ---------------------------------------------------------------------------

def plot_attention_over_time(step_attentions, tokenizer, save_dir):
    """Track attention entropy and mean distance at each MW step."""
    n_steps = len(step_attentions)
    n_layers = len(step_attentions[0][1][-1])
    n_heads = step_attentions[0][1][-1][0].shape[1]

    entropies = np.zeros((n_steps, n_layers, n_heads))
    distances = np.zeros((n_steps, n_layers, n_heads))

    for step_idx in range(n_steps):
        ctx_ids, all_attention = step_attentions[step_idx]
        final_attn = all_attention[-1]

        for layer_idx in range(n_layers):
            attn = final_attn[layer_idx][0].cpu().numpy()
            seq_len = attn.shape[1]

            for h in range(n_heads):
                weights = attn[h, -1, :]  # last query position
                # Entropy
                w_clipped = np.clip(weights, 1e-10, 1.0)
                entropies[step_idx, layer_idx, h] = -np.sum(w_clipped * np.log(w_clipped))
                # Mean distance
                key_positions = np.arange(seq_len)
                dist = (seq_len - 1) - key_positions
                distances[step_idx, layer_idx, h] = np.sum(weights * dist)

    steps = np.arange(1, n_steps + 1)

    fig, axes = plt.subplots(2, n_layers, figsize=(8 * n_layers, 10))
    if n_layers == 1:
        axes = axes[:, np.newaxis]

    colors = plt.cm.Set1(np.linspace(0, 0.8, n_heads))

    for layer_idx in range(n_layers):
        # Entropy
        ax = axes[0, layer_idx]
        for h in range(n_heads):
            ax.plot(steps, entropies[:, layer_idx, h],
                    color=colors[h], linewidth=1.5, label=f'Head {h}', alpha=0.8)
        ax.set_xlabel('MW Step')
        ax.set_ylabel('Attention Entropy')
        ax.set_title(f'Layer {layer_idx}: Entropy Over Time')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # Distance
        ax = axes[1, layer_idx]
        for h in range(n_heads):
            ax.plot(steps, distances[:, layer_idx, h],
                    color=colors[h], linewidth=1.5, label=f'Head {h}', alpha=0.8)
        ax.set_xlabel('MW Step')
        ax.set_ylabel('Mean Attention Distance')
        ax.set_title(f'Layer {layer_idx}: Attention Distance Over Time')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle('Attention Entropy & Distance Over Sequence Time', fontsize=14, y=1.01)
    plt.tight_layout()
    path = os.path.join(save_dir, 'fig6_attention_over_time.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved {path}")


# ---------------------------------------------------------------------------
# Figure 7: Learned weight trajectories vs optimal MW weights
# ---------------------------------------------------------------------------

def plot_weight_trajectories(step_weights, seq, tokenizer, save_dir):
    """Compare learned expert weights vs optimal MW weights over time."""
    n_experts = len(seq['expert_predictions'][0])
    n_steps = len(seq['true_labels'])

    # Compute optimal MW weights
    learning_rate = np.sqrt(np.log(n_experts) / max(n_steps, 1))
    mw = MultiplicativeWeights(n_experts, learning_rate)
    mw_weights_over_time = []

    for step in range(n_steps):
        mw_weights_over_time.append(mw.get_probabilities().copy())
        if step < len(seq['losses']):
            mw.update_weights(np.array(seq['losses'][step]))

    mw_weights_over_time = np.array(mw_weights_over_time)  # [n_steps, n_experts]

    # Find best expert for annotation
    expert_cum = np.zeros(n_experts)
    for losses in seq['losses']:
        expert_cum += np.array(losses)
    best_e = np.argmin(expert_cum)

    steps = np.arange(1, n_steps + 1)
    colors = plt.cm.Set1(np.linspace(0, 0.8, n_experts))

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Learned weights
    ax = axes[0]
    for e in range(n_experts):
        label = f'Expert {e}' + (' ★' if e == best_e else '')
        ax.plot(steps, step_weights[:, e], color=colors[e], linewidth=2,
                label=label, alpha=0.9)
    ax.set_xlabel('MW Step')
    ax.set_ylabel('Weight')
    ax.set_title('Learned Model Weights')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)

    # Optimal MW weights
    ax = axes[1]
    for e in range(n_experts):
        label = f'Expert {e}' + (' ★' if e == best_e else '')
        ax.plot(steps, mw_weights_over_time[:, e], color=colors[e], linewidth=2,
                label=label, alpha=0.9)
    ax.set_xlabel('MW Step')
    ax.set_ylabel('Weight')
    ax.set_title('Optimal MW Weights')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)

    plt.suptitle('Expert Weight Trajectories: Learned vs Optimal MW', fontsize=14)
    plt.tight_layout()
    path = os.path.join(save_dir, 'fig7_weight_trajectories.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved {path}")


# ---------------------------------------------------------------------------
# Figure 8: CoT hidden state PCA
# ---------------------------------------------------------------------------

def plot_cot_hidden_pca(step_thought_hiddens, seq, save_dir):
    """PCA of thought hidden states across CoT steps and MW steps.
    
    step_thought_hiddens: [n_steps, K, d_model]
    """
    n_steps, K, d_model = step_thought_hiddens.shape

    # Reshape to [n_steps * K, d_model] for PCA
    all_hiddens = step_thought_hiddens.reshape(-1, d_model)
    pca = PCA(n_components=2)
    projected = pca.fit_transform(all_hiddens)  # [n_steps * K, 2]
    projected = projected.reshape(n_steps, K, 2)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Plot 1: color by CoT thought step
    ax = axes[0]
    cot_colors = plt.cm.viridis(np.linspace(0.2, 0.9, K))
    for k in range(K):
        ax.scatter(projected[:, k, 0], projected[:, k, 1],
                   c=[cot_colors[k]], s=20, alpha=0.7, label=f'Thought {k}')
        # Draw arrows from thought k to k+1
        if k < K - 1:
            for s in range(0, n_steps, max(1, n_steps // 15)):
                ax.annotate('', xy=projected[s, k+1], xytext=projected[s, k],
                            arrowprops=dict(arrowstyle='->', color='gray',
                                            alpha=0.2, lw=0.5))
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
    ax.set_title('CoT Hidden States Colored by Thought Step')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Plot 2: color by MW time step (last thought only)
    ax = axes[1]
    scatter = ax.scatter(projected[:, -1, 0], projected[:, -1, 1],
                         c=np.arange(n_steps), cmap='coolwarm', s=30, alpha=0.8)
    # Connect sequential points
    ax.plot(projected[:, -1, 0], projected[:, -1, 1], 'k-', alpha=0.15, linewidth=0.5)
    plt.colorbar(scatter, ax=ax, label='MW Step')
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
    ax.set_title('Final Thought Hidden State Colored by MW Step')
    ax.grid(True, alpha=0.3)

    plt.suptitle('CoT Hidden State Space (PCA)', fontsize=14)
    plt.tight_layout()
    path = os.path.join(save_dir, 'fig8_cot_hidden_pca.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved {path}")


# ---------------------------------------------------------------------------
# Figure 9: Token embedding space
# ---------------------------------------------------------------------------

def plot_token_embedding_space(model, tokenizer, save_dir):
    """PCA/visualization of learned token embeddings colored by type."""
    with torch.no_grad():
        all_embeddings = model.token_embedding.weight.cpu().numpy()  # [vocab_size, d_model]

    # Collect token IDs and their types
    token_ids = []
    token_labels = []
    token_type_names = []

    # START
    token_ids.append(tokenizer.START_TOKEN)
    token_labels.append('START')
    token_type_names.append('START')

    # SEP
    token_ids.append(tokenizer.SEP_TOKEN)
    token_labels.append('SEP')
    token_type_names.append('SEP')

    # EXPERT tokens
    for i, t in enumerate(tokenizer.EXPERT_TOKENS):
        token_ids.append(t)
        token_labels.append(f'E{i}')
        token_type_names.append('EXPERT')

    # PRED tokens
    token_ids.append(tokenizer.PRED_0_TOKEN)
    token_labels.append('P0')
    token_type_names.append('PRED')
    token_ids.append(tokenizer.PRED_1_TOKEN)
    token_labels.append('P1')
    token_type_names.append('PRED')

    # STEP tokens (first 20)
    for i, t in enumerate(tokenizer.STEP_TOKENS[:20]):
        token_ids.append(t)
        token_labels.append(f'S{i}')
        token_type_names.append('STEP')

    # LOSS tokens (sample 10 evenly)
    loss_indices = np.linspace(0, len(tokenizer.LOSS_TOKENS) - 1, 10, dtype=int)
    for idx in loss_indices:
        t = tokenizer.LOSS_TOKENS[idx]
        token_ids.append(t)
        token_labels.append(f'L{idx}')
        token_type_names.append('LOSS')

    # Get embeddings for selected tokens
    selected_emb = all_embeddings[token_ids]  # [n_selected, d_model]

    pca = PCA(n_components=2)
    projected = pca.fit_transform(selected_emb)

    fig, ax = plt.subplots(figsize=(12, 9))
    type_set = list(set(token_type_names))
    type_color_map = {t: TOKEN_COLORS.get(t, '#000000') for t in type_set}

    for ttype in type_set:
        mask = [i for i, tn in enumerate(token_type_names) if tn == ttype]
        ax.scatter(projected[mask, 0], projected[mask, 1],
                   c=type_color_map[ttype], s=60, alpha=0.8, label=ttype,
                   edgecolors='white', linewidths=0.5)
        for i in mask:
            ax.annotate(token_labels[i], (projected[i, 0], projected[i, 1]),
                        fontsize=7, ha='center', va='bottom',
                        color=type_color_map[ttype], alpha=0.9)

    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
    ax.set_title('Learned Token Embedding Space (PCA)')
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, 'fig9_token_embedding_space.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Attention analysis for MW transformer')
    parser.add_argument('--checkpoint', type=str,
                        default='../figures/checkpoints/model_stage_13.pt')
    parser.add_argument('--seq_length', type=int, default=50,
                        help='Sequence length for the analysis')
    parser.add_argument('--n_experts', type=int, default=4)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--save_dir', type=str, default='../figures/attention-figures')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    os.makedirs(args.save_dir, exist_ok=True)

    logger.info(f"Loading checkpoint: {args.checkpoint}")
    model, tokenizer, cot_mode, model_config = load_model(args.checkpoint, device)
    logger.info(f"Model: {cot_mode}, n_thought_steps={model_config.n_thought_steps}")

    # Generate a single sequence for detailed analysis
    seq = generate_single_sequence(args.n_experts, args.seq_length)
    logger.info(f"Generated sequence: {args.seq_length} steps, {args.n_experts} experts")

    # Identify best/worst expert for reference
    expert_cum = np.zeros(args.n_experts)
    for losses in seq['losses']:
        expert_cum += np.array(losses)
    best_e = np.argmin(expert_cum)
    worst_e = np.argmax(expert_cum)
    logger.info(f"Best expert: {best_e} (cum loss {expert_cum[best_e]:.1f}), "
                f"Worst expert: {worst_e} (cum loss {expert_cum[worst_e]:.1f})")

    # Run model and collect attention
    logger.info("Running model with attention collection...")
    decisions, step_attentions, step_weights, step_logits, step_thought_hiddens = \
        run_with_attention(
            model, seq, tokenizer, device, max_ctx=model_config.max_sequence_length
        )

    accuracy = np.mean(np.array(decisions) == np.array(seq['true_labels']))
    logger.info(f"Accuracy on this sequence: {accuracy:.4f}")

    # Generate all 9 figures
    logger.info("\n--- Generating figures ---")

    logger.info("Figure 1: Per-head attention heatmaps...")
    plot_per_head_heatmaps(step_attentions, tokenizer, args.save_dir, step_idx=-1)

    logger.info("Figure 2: Attention by token type...")
    plot_attention_by_token_type(step_attentions, tokenizer, args.save_dir)

    logger.info("Figure 3: Best vs worst expert attention...")
    plot_attention_best_vs_worst_expert(step_attentions, seq, tokenizer, args.save_dir)

    logger.info("Figure 4: Attention distance profile...")
    plot_attention_distance_profile(step_attentions, tokenizer, args.save_dir)

    logger.info("Figure 5: Attention across CoT steps...")
    plot_attention_across_cot_steps(step_attentions, tokenizer, args.save_dir, step_idx=-1)

    logger.info("Figure 6: Attention over sequence time...")
    plot_attention_over_time(step_attentions, tokenizer, args.save_dir)

    logger.info("Figure 7: Weight trajectories (learned vs MW)...")
    plot_weight_trajectories(step_weights, seq, tokenizer, args.save_dir)

    logger.info("Figure 8: CoT hidden state PCA...")
    plot_cot_hidden_pca(step_thought_hiddens, seq, args.save_dir)

    logger.info("Figure 9: Token embedding space...")
    plot_token_embedding_space(model, tokenizer, args.save_dir)

    logger.info(f"\nAll 9 figures saved to {args.save_dir}")


if __name__ == '__main__':
    main()
