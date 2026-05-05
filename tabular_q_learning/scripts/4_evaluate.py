#!/usr/bin/env python3
"""
4_evaluate.py — Evaluation & Analysis for Recurrent Context Transformer

Part 1 — Action prediction (ID + OOD)
  Feeds trajectory step-by-step, passing context tokens explicitly.
  At each SELECT position, records argmax action vs. Q-learning greedy.

Part 2 — Context token Q-value probing
  Trains a linear probe on each context token c_a^(t) to decode the
  Q-value vector for action a across all states:
      W · c_a^(t) ≈ [Q(s_1, a), Q(s_2, a), ..., Q(s_{|S|}, a)]

Part 3 — Attention heatmaps
  Verifies SELECT attends to EVAL tokens, UPDATE attends to context + QCURR/QNEXT.

Part 4 — Regret comparison
  Runs the transformer autonomously (closed-loop) on fresh MDPs and compares
  cumulative reward against a greedy Q-learner (ε=0) and an ε-greedy Q-learner.

Part 5 — Effective α/γ recovery
  Fits (α_eff, γ_eff) per trajectory from context-probe Q-value dynamics.

Part 6 — Reward probe from context delta
  Trains a linear probe on the residual Δc_{a_t} = update_hidden - context[a_t]
  to predict the scalar reward r_t at each step. Tests whether the trained
  model preserves the handwired construction's reward-injection pathway
  (Layer 4's R/UPDATE head, which writes α·r into c_{a_t}'s buf₁).

Output files:
    figures/action_agreement.png
    figures/probe_scatter.png
    figures/probe_frobenius.png
    figures/attention_heatmap.png
    figures/attention_heatmap.csv
    figures/attention_full_heatmap_4s2a.png
    figures/training_curves.png
    figures/per_state_agreement.png
    figures/regret.png
    figures/effective_alpha_gamma.png
    figures/reward_probe.png
"""

import argparse
import csv
import importlib.util
import math
import os
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Import model and helpers
# ---------------------------------------------------------------------------
_script_dir = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "coconut_model",
    os.path.join(_script_dir, "2_model.py")
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
COCONUTConfig      = _mod.COCONUTConfig
COCONUTTransformer = _mod.COCONUTTransformer
build_vocab        = _mod.build_vocab

_spec3 = importlib.util.spec_from_file_location(
    "coconut_train",
    os.path.join(_script_dir, "3_train.py")
)
_mod3 = importlib.util.module_from_spec(_spec3)
_spec3.loader.exec_module(_mod3)
build_step_tokens = _mod3.build_step_tokens


# ---------------------------------------------------------------------------
# MDP + tabular Q-learning
# ---------------------------------------------------------------------------

def generate_eval_mdp(
    n_states: int,
    n_actions: int,
    seed: int = 9999,
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    P = rng.dirichlet(alpha=np.ones(n_states), size=(n_states, n_actions)).astype(np.float32)
    R = rng.beta(2.0, 2.0, size=(n_states, n_actions)).astype(np.float32)
    R = np.clip(R, 0.0, 1.0)
    return P, R


def generate_ood_mdp(
    n_states: int,
    n_actions: int,
    variant: str,
    seed: int = 9999,
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    if variant == 'deterministic':
        P = rng.dirichlet(np.full(n_states, 0.001), size=(n_states, n_actions)).astype(np.float32)
        R = rng.beta(2.0, 2.0, size=(n_states, n_actions)).astype(np.float32)
    elif variant == 'sparse_reward':
        P = rng.dirichlet(np.ones(n_states), size=(n_states, n_actions)).astype(np.float32)
        R = rng.beta(0.1, 2.0, size=(n_states, n_actions)).astype(np.float32)
    elif variant == 'dense_reward':
        P = rng.dirichlet(np.ones(n_states), size=(n_states, n_actions)).astype(np.float32)
        R = rng.beta(10.0, 10.0, size=(n_states, n_actions)).astype(np.float32)
    elif variant == 'adversarial':
        P = rng.dirichlet(np.full(n_states, 0.001), size=(n_states, n_actions)).astype(np.float32)
        R = rng.uniform(0.0, 1.0, size=(n_states, n_actions)).astype(np.float32)
    elif variant == 'high_variance':
        P = rng.dirichlet(np.full(n_states, 0.5), size=(n_states, n_actions)).astype(np.float32)
        R = rng.beta(0.1, 0.1, size=(n_states, n_actions)).astype(np.float32)
    else:
        raise ValueError(f"Unknown OOD variant: {variant!r}")
    R = np.clip(R, 0.0, 1.0)
    return P, R


OOD_VARIANTS: List[Tuple[str, str]] = [
    ('deterministic', 'Deterministic transitions (Dir 0.001), same rewards'),
    ('sparse_reward', 'Sparse rewards (Beta 0.1,2), same transitions'),
    ('dense_reward',  'Dense rewards (Beta 10,10), same transitions'),
    ('adversarial',   'Deterministic transitions + Uniform rewards'),
    ('high_variance', 'Moderate transitions (Dir 0.5) + extreme bimodal rewards (Beta 0.1,0.1)'),
]


def run_tabular_q_learning(
    P: np.ndarray,
    R: np.ndarray,
    n_states: int,
    n_actions: int,
    n_steps: int,
    alpha: float = 0.1,
    gamma: float = 0.9,
    epsilon: float = 0.2,
    seed: int = 9999,
) -> Tuple[List[Dict], np.ndarray]:
    rng = np.random.default_rng(seed + 1)
    Q = np.zeros((n_states, n_actions), dtype=np.float32)
    s = int(rng.integers(n_states))

    trajectory  = []
    q_snapshots = []

    for _ in range(n_steps):
        if rng.random() < epsilon:
            a = int(rng.integers(n_actions))
        else:
            best = float(np.max(Q[s]))
            ties = [ac for ac in range(n_actions) if Q[s, ac] == best]
            a = int(rng.choice(ties))

        r = float(R[s, a])
        s_next = int(rng.choice(n_states, p=P[s, a]))

        max_q_next = float(np.max(Q[s_next]))
        Q[s, a] = (1.0 - alpha) * Q[s, a] + alpha * (r + gamma * max_q_next)

        best_next = float(np.max(Q[s_next]))
        ties_next = [ac for ac in range(n_actions) if Q[s_next, ac] == best_next]
        a_next = int(rng.choice(ties_next))

        trajectory.append({
            's': s, 'a': a, 'r': r, 's_next': s_next,
            'a_star': a_next, 'a_next': a_next,
        })
        q_snapshots.append(Q.copy())
        s = s_next

    return trajectory, np.stack(q_snapshots, axis=0)


# ---------------------------------------------------------------------------
# Part 1: Action prediction with recurrent context
# ---------------------------------------------------------------------------

def run_action_inference(
    model: COCONUTTransformer,
    trajectory: List[Dict],
    vocab: Dict,
    n_actions: int,
    config: COCONUTConfig,
    device: torch.device,
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """Feed trajectory step-by-step with explicit context passing.

    Returns (predicted_actions, context_history) where context_history[t]
    is the context tensor (n_actions, d_model) after step t.
    """
    max_actions = config.max_actions
    model.eval()

    context = model.get_init_context(1, n_actions, device)
    predicted_actions = []
    context_history = []

    with torch.no_grad():
        for t, tr in enumerate(trajectory):
            token_list, r_off, s_off, u_off = build_step_tokens(tr, vocab, n_actions)
            token_ids = torch.tensor([token_list], dtype=torch.long, device=device)
            reward_val = torch.tensor([tr['r']], dtype=torch.float32, device=device)

            select_logits, update_hidden = model.forward_step(
                token_ids=token_ids,
                reward_value=reward_val,
                reward_offset=r_off,
                select_offset=s_off,
                update_offset=u_off,
                context=context,
            )

            if n_actions < max_actions:
                select_logits[:, n_actions:] = float('-inf')

            pred_a = int(select_logits[0].argmax().item())
            predicted_actions.append(pred_a)

            context_history.append(context[0].cpu().numpy().copy())

            a_t = tr['a']
            new_context = context.clone()
            new_context[0, a_t, :] = update_hidden[0]
            context = new_context

    context_history.append(context[0].cpu().numpy().copy())

    return np.array(predicted_actions, dtype=np.int32), context_history


# ---------------------------------------------------------------------------
# Part 2: Context token Q-value probing
# ---------------------------------------------------------------------------

class ContextQProbe(nn.Module):
    """Linear probe: context token c_a -> Q-values for action a across all states.

    Maps c_a^(t) in R^{d_model} to [Q(s_1, a), ..., Q(s_{|S|}, a)] in R^{|S|}.
    """
    def __init__(self, d_model: int, n_states: int):
        super().__init__()
        self.linear = nn.Linear(d_model, n_states)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.linear(h)


def collect_context_probe_data(
    model: COCONUTTransformer,
    n_trajectories: int,
    n_states: int,
    n_actions: int,
    vocab: Dict,
    config: COCONUTConfig,
    device: torch.device,
    n_steps: int = 50,
    seed_offset: int = 20000,
) -> Tuple[np.ndarray, np.ndarray, List[List[Dict]]]:
    """Collect context tokens and Q-value targets for probe training.

    Returns
    -------
    ctx_all : (n_trajectories * n_steps * n_actions, d_model)
        ctx_all[i*n_steps*n_actions + t*n_actions + a] = context for action a
        at step t of trajectory i (i.e. the context *before* step t's update).
    q_target_all : (n_trajectories * n_steps * n_actions, n_states)
        q_target_all[...] = Q[:, a] (tabular) before step t's update.
    trajectories : list of n_trajectories lists of transition dicts
    """
    model.eval()
    max_actions = config.max_actions
    d_model = config.d_model

    N = n_trajectories * n_steps * n_actions
    ctx_all = np.empty((N, d_model), dtype=np.float32)
    q_target_all = np.empty((N, n_states), dtype=np.float32)
    all_trajectories: List[List[Dict]] = []
    write_idx = 0

    with torch.no_grad():
        for i in range(n_trajectories):
            seed = seed_offset + i
            P, R = generate_eval_mdp(n_states, n_actions, seed=seed)
            trajectory, q_snapshots = run_tabular_q_learning(
                P, R, n_states, n_actions, n_steps=n_steps, seed=seed
            )

            context = model.get_init_context(1, n_actions, device)

            for t, tr in enumerate(trajectory):
                token_list, r_off, s_off, u_off = build_step_tokens(tr, vocab, n_actions)
                token_ids = torch.tensor([token_list], dtype=torch.long, device=device)
                reward_val = torch.tensor([tr['r']], dtype=torch.float32, device=device)

                _, update_hidden = model.forward_step(
                    token_ids=token_ids,
                    reward_value=reward_val,
                    reward_offset=r_off,
                    select_offset=s_off,
                    update_offset=u_off,
                    context=context,
                )

                ctx_t = context[0, :n_actions, :].detach().cpu().numpy()
                q_t = q_snapshots[t]

                ctx_all[write_idx:write_idx + n_actions] = ctx_t
                q_target_all[write_idx:write_idx + n_actions] = q_t.T
                write_idx += n_actions

                a_t = tr['a']
                new_context = context.clone()
                new_context[0, a_t, :] = update_hidden[0]
                context = new_context

                del token_ids, reward_val, update_hidden

            all_trajectories.append(trajectory)

    return ctx_all, q_target_all, all_trajectories


def train_probe(
    probe: ContextQProbe,
    ctx_all: np.ndarray,
    q_all: np.ndarray,
    device: torch.device,
    n_epochs: int = 10,
    batch_size: int = 256,
    lr: float = 1e-3,
) -> List[float]:
    probe.train()
    optimizer = torch.optim.Adam(probe.parameters(), lr=lr)
    N = ctx_all.shape[0]
    losses = []

    ctx_t = torch.tensor(ctx_all, dtype=torch.float32, device=device)
    q_t = torch.tensor(q_all, dtype=torch.float32, device=device)

    for _ in range(n_epochs):
        perm = torch.randperm(N, device=device)
        epoch_loss = 0.0
        n_batches = 0
        for i in range(0, N, batch_size):
            idx = perm[i:i + batch_size]
            pred = probe(ctx_t[idx])
            loss = F.mse_loss(pred, q_t[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        losses.append(epoch_loss / max(n_batches, 1))

    return losses


def evaluate_probe(
    probe: ContextQProbe,
    ctx_all: np.ndarray,
    q_all: np.ndarray,
    device: torch.device,
) -> Tuple[float, float, np.ndarray]:
    probe.eval()
    with torch.no_grad():
        ctx_t = torch.tensor(ctx_all, dtype=torch.float32, device=device)
        q_pred = probe(ctx_t).cpu().numpy()

    q_true_flat = q_all.reshape(-1)
    q_pred_flat = q_pred.reshape(-1)

    ss_res = np.sum((q_true_flat - q_pred_flat) ** 2)
    ss_tot = np.sum((q_true_flat - q_true_flat.mean()) ** 2) + 1e-12
    r2 = float(1.0 - ss_res / ss_tot)

    diff = q_pred - q_all
    frob = np.sqrt((diff ** 2).sum(axis=-1))
    frob_mean = float(frob.mean())

    return r2, frob_mean, q_pred


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def plot_action_agreement(
    aa_mean_id: np.ndarray,
    aa_std_id: np.ndarray,
    ood_results: List[Tuple[str, np.ndarray, np.ndarray]],
    save_path: str,
    n_mdps: int = 10,
    label_suffix: str = '',
) -> None:
    steps = np.arange(1, len(aa_mean_id) + 1)
    ood_colours = ['darkorange', 'crimson', 'forestgreen', 'purple', 'saddlebrown']
    ood_styles = ['--', '-.', ':', (0, (3, 1, 1, 1)), (0, (5, 1))]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(steps, aa_mean_id, color='steelblue', linewidth=2.5,
            label='In-distribution (ID)')
    ax.fill_between(steps, aa_mean_id - aa_std_id, aa_mean_id + aa_std_id,
                    alpha=0.2, color='steelblue')

    for i, (lbl, mean_ood, std_ood) in enumerate(ood_results):
        colour = ood_colours[i % len(ood_colours)]
        style = ood_styles[i % len(ood_styles)]
        ax.plot(steps, mean_ood, color=colour, linewidth=1.8,
                linestyle=style, label=f'OOD: {lbl}')
        ax.fill_between(steps, mean_ood - std_ood, mean_ood + std_ood,
                        alpha=0.10, color=colour)

    ax.axhline(y=1.0, color='gray', linestyle=':', linewidth=1, alpha=0.5,
               label='Perfect agreement')
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Greedy Action Agreement')
    ax.set_ylim(-0.05, 1.1)
    suffix = f' ({label_suffix})' if label_suffix else ''
    ax.set_title(
        f'Per-Step Action Agreement: ID vs OOD Variants{suffix}\n'
        f'(mean +/- std, {n_mdps} MDPs each variant)',
        fontsize=11,
    )
    ax.legend(fontsize=8, loc='lower right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_probe_scatter(
    q_true: np.ndarray,
    q_pred: np.ndarray,
    r2: float,
    save_path: str,
) -> None:
    q_t = q_true.reshape(-1)
    q_p = q_pred.reshape(-1)
    rng = np.random.default_rng(0)
    idx = rng.choice(len(q_t), size=min(10000, len(q_t)), replace=False)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(q_t[idx], q_p[idx], alpha=0.1, s=3, color='steelblue', rasterized=True)
    lo = min(q_t.min(), q_p.min())
    hi = max(q_t.max(), q_p.max())
    ax.plot([lo, hi], [lo, hi], 'r--', linewidth=1.5, label='y = x')
    ax.set_xlabel('Q tabular')
    ax.set_ylabel('Q probe (context token)')
    ax.set_title(f'Context Token Probe: Q-Values vs True\nR2 = {r2:.4f}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_probe_frobenius(
    q_pred_all: np.ndarray,
    q_true_all: np.ndarray,
    n_steps: int,
    n_traj: int,
    n_actions: int,
    save_path: str,
) -> None:
    n_per_step = n_actions
    q_pred_r = q_pred_all.reshape(n_traj, n_steps, n_per_step, -1)
    q_true_r = q_true_all.reshape(n_traj, n_steps, n_per_step, -1)

    diff = q_pred_r - q_true_r
    frob = np.sqrt((diff ** 2).sum(axis=(-2, -1)))
    frob_mean = frob.mean(axis=0)
    frob_std = frob.std(axis=0)

    zero_baseline = np.sqrt((q_true_r ** 2).sum(axis=(-2, -1))).mean(axis=0)

    steps = np.arange(1, n_steps + 1)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(steps, frob_mean, color='steelblue', linewidth=2, label='Context probe error')
    ax.fill_between(steps, frob_mean - frob_std, frob_mean + frob_std,
                    alpha=0.25, color='steelblue')
    ax.plot(steps, zero_baseline, color='tomato', linewidth=2, linestyle='--',
            label='Zero baseline')
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Frobenius error')
    ax.set_title(f'Context Token Q-Probe Error Over Time\n'
                 f'(mean +/- std, {n_traj} MDPs)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


# ---------------------------------------------------------------------------
# Attention heatmap
# ---------------------------------------------------------------------------

def get_token_labels(n_ctx: int, token_ids: List[int], vocab: Dict) -> List[str]:
    """Labels for the flat sequence layout: [BOS, ctx_0..ctx_{n_ctx-1}, rest_of_discrete].

    token_ids is the original discrete sequence whose first element is BOS.
    """
    inv = {}
    inv[vocab['TOK_NULL']]   = 'NULL'
    inv[vocab['TOK_START']]  = 'BOS'
    inv[vocab['TOK_R']]      = 'R'
    inv[vocab['TOK_SELECT']] = 'SEL'
    inv[vocab['TOK_UPDATE']] = 'UPD'
    inv[vocab['TOK_QCURR']]  = 'QCUR'
    inv[vocab['TOK_QNEXT']]  = 'QNXT'
    for i, tok in enumerate(vocab['TOK_S']):
        inv[tok] = f'S{i}'
    for i, tok in enumerate(vocab['TOK_A']):
        inv[tok] = f'A{i}'

    bos_label = inv.get(token_ids[0], f'?{token_ids[0]}')
    rest      = [inv.get(t, f'?{t}') for t in token_ids[1:]]
    return [bos_label] + [f'ctx_{i}' for i in range(n_ctx)] + rest


def plot_attention_heatmap(
    model: COCONUTTransformer,
    trajectory: List[Dict],
    vocab: Dict,
    n_actions: int,
    config: COCONUTConfig,
    device: torch.device,
    save_path: str,
) -> None:
    max_actions = config.max_actions
    model.eval()

    tr = trajectory[0]
    token_list, r_off, s_off, u_off = build_step_tokens(tr, vocab, n_actions)
    token_ids = torch.tensor([token_list], dtype=torch.long, device=device)
    reward_val = torch.tensor([tr['r']], dtype=torch.float32, device=device)

    context = model.get_init_context(1, n_actions, device)

    with torch.no_grad():
        select_logits, update_hidden, all_attn = model.forward_step(
            token_ids=token_ids,
            reward_value=reward_val,
            reward_offset=r_off,
            select_offset=s_off,
            update_offset=u_off,
            context=context,
            return_attention=True,
        )

    n_ctx = n_actions
    labels = get_token_labels(n_ctx, token_list, vocab)
    # Flat layout: [BOS, ctx_0..ctx_{n_ctx-1}, rest_of_discrete]; total length = n_ctx + len(token_list).
    T = n_ctx + len(token_list)

    sel_pos = n_ctx + s_off
    upd_pos = n_ctx + u_off

    L = len(all_attn)
    H = all_attn[0].shape[1]

    query_rows = [(sel_pos, 'SELECT'), (upd_pos, 'UPDATE')]
    n_queries = len(query_rows)

    bar_h = 0.6
    col_w = max(5.0, T * 0.12)
    fig, axes = plt.subplots(
        L, H,
        figsize=(H * col_w, L * bar_h * (n_queries + 1.5)),
        squeeze=False,
    )

    cmap = plt.cm.Blues

    for layer_idx in range(L):
        attn_np = all_attn[layer_idx][0].cpu().numpy()  # [H, T, T]
        for head_idx in range(H):
            ax = axes[layer_idx][head_idx]
            data = attn_np[head_idx]

            rows = np.stack([data[pos] for pos, _ in query_rows], axis=0)
            vmax = rows.max() + 1e-9

            ax.imshow(
                rows, aspect='auto', cmap=cmap,
                vmin=0.0, vmax=vmax, origin='upper', interpolation='nearest',
            )

            if layer_idx == L - 1:
                ax.set_xticks(range(T))
                ax.set_xticklabels(labels, rotation=90, fontsize=6)
            else:
                ax.set_xticks([])

            ax.set_yticks(range(n_queries))
            if head_idx == 0:
                ax.set_yticklabels([lbl for _, lbl in query_rows], fontsize=7)
            else:
                ax.set_yticklabels([])

            if layer_idx == 0:
                ax.set_title(f'Head {head_idx}', fontsize=8, pad=3)
            if head_idx == 0:
                ax.set_ylabel(f'L{layer_idx + 1}', fontsize=8)

    fig.suptitle(
        'Attention — query rows: SELECT & UPDATE  |  step 0 (initial context)',
        fontsize=10, y=1.02,
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")

    csv_path = os.path.splitext(save_path)[0] + '.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        header = ['layer', 'head', 'query_token'] + labels
        writer.writerow(header)
        for layer_idx in range(L):
            attn_np = all_attn[layer_idx][0].cpu().numpy()
            for head_idx in range(H):
                data = attn_np[head_idx]
                for pos, lbl in query_rows:
                    row = [layer_idx + 1, head_idx, lbl] + data[pos].tolist()
                    writer.writerow(row)
    print(f"  Saved: {csv_path}")


# ---------------------------------------------------------------------------
# Full causal-triangle attention heatmap (per layer × head, single sequence)
# ---------------------------------------------------------------------------

def plot_full_attention_heatmap(
    model: COCONUTTransformer,
    trajectory: List[Dict],
    vocab: Dict,
    n_actions_actual: int,
    config: COCONUTConfig,
    device: torch.device,
    save_path: str,
) -> None:
    """Plot the full T×T causal-attention matrix (triangle pattern), averaged
    over every transition step in the episode. One subplot per (layer, head).

    Each step has the same token-sequence length, so per-step attention
    matrices are directly averageable. Per-step token *identities* (S/A token
    ids) differ across steps, so axis labels reflect the structural slot
    rather than concrete values for any single step.
    """
    max_actions = config.max_actions
    model.eval()
    n_steps = len(trajectory)

    n_actions = n_actions_actual
    sample_tokens, _, _, _ = build_step_tokens(trajectory[0], vocab, n_actions)
    n_ctx = n_actions
    T = n_ctx + len(sample_tokens)

    accum: Optional[List[np.ndarray]] = None
    L: Optional[int] = None
    H: Optional[int] = None

    context = model.get_init_context(1, n_actions, device)

    with torch.no_grad():
        for t_idx, tr in enumerate(trajectory):
            token_list, r_off, s_off, u_off = build_step_tokens(
                tr, vocab, n_actions
            )
            token_ids = torch.tensor([token_list], dtype=torch.long, device=device)
            reward_val = torch.tensor([tr['r']], dtype=torch.float32, device=device)

            _, upd_h, all_attn = model.forward_step(
                token_ids=token_ids,
                reward_value=reward_val,
                reward_offset=r_off,
                select_offset=s_off,
                update_offset=u_off,
                context=context,
                return_attention=True,
            )

            if accum is None:
                L = len(all_attn)
                H = all_attn[0].shape[1]
                accum = [np.zeros((H, T, T), dtype=np.float64) for _ in range(L)]

            for layer_idx in range(L):
                accum[layer_idx] += all_attn[layer_idx][0].cpu().numpy()

            new_ctx = context.clone()
            new_ctx[0, tr['a'], :] = upd_h[0]
            context = new_ctx

    avg_attn = [a / max(n_steps, 1) for a in accum]
    # Average across heads: one [T, T] map per layer.
    per_layer = [a.mean(axis=0) for a in avg_attn]

    # Structural slot labels (BOS, then context, then per-step discrete tokens)
    slot_labels: List[str] = ['BOS']
    for i in range(n_ctx):
        slot_labels.append(f'ctx_{i}')
    A = n_actions
    slot_labels += ['QCUR', 's_t', 'a_t', 'R', 'QNXT']
    for c in range(A):
        slot_labels += ["s'", f'a{c}']
    slot_labels += ['SEL', 'a*', 'UPD']
    labels = slot_labels

    cell = max(0.18, 6.0 / max(T, 1))
    # One subplot per layer (heads averaged).
    fig, axes = plt.subplots(
        1, L,
        figsize=(L * (T * cell + 1.0), T * cell + 1.0),
        squeeze=False,
    )

    cmap = plt.cm.viridis

    for layer_idx in range(L):
        ax = axes[0][layer_idx]
        data = per_layer[layer_idx]
        ax.imshow(
            data, aspect='equal', cmap=cmap,
            vmin=0.0, vmax=max(float(data.max()), 1e-9),
            origin='upper', interpolation='nearest',
        )
        ax.set_xticks(range(T))
        ax.set_yticks(range(T))
        ax.set_xticklabels(labels, rotation=90, fontsize=6)
        if layer_idx == 0:
            ax.set_yticklabels(labels, fontsize=6)
        else:
            ax.set_yticklabels([])
        ax.set_title(f'Layer {layer_idx + 1} (heads averaged)', fontsize=10, pad=4)

    fig.suptitle(
        f'Causal attention (full T×T), averaged over {n_steps} steps and {H} heads — '
        f'n_actions={n_actions}  '
        f'(model max_states={config.max_states}, max_actions={max_actions})',
        fontsize=11, y=1.02,
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


# ---------------------------------------------------------------------------
# Per-distribution agreement heatmap
# ---------------------------------------------------------------------------

_BIN_PHASE_LABELS = ['Early', 'Mid', 'Late']

DIST_SHORT_LABELS: Dict[str, str] = {
    'deterministic': 'Deterministic trans.',
    'sparse_reward':  'Sparse rewards',
    'dense_reward':   'Dense rewards',
    'adversarial':    'Adversarial',
    'high_variance':  'High-variance',
}


def plot_per_distribution_agreement(
    dist_data: List[Tuple[str, np.ndarray]],
    n_steps: int,
    save_path: str,
    n_bins: int = 3,
) -> None:
    """Heatmap: rows = reward distribution type, columns = coarse timestep bins.

    Parameters
    ----------
    dist_data : list of (label, agreements_array)
        agreements_array has shape (n_mdps, n_steps) with per-step 0/1 agreement.
    """
    bin_edges = np.linspace(0, n_steps, n_bins + 1, dtype=int)
    phase_labels = _BIN_PHASE_LABELS[:n_bins] if n_bins <= len(_BIN_PHASE_LABELS) else [
        f'Bin {i+1}' for i in range(n_bins)
    ]
    bin_labels = [
        f'{phase}\nt={bin_edges[i]+1}–{bin_edges[i+1]}'
        for i, phase in enumerate(phase_labels)
    ]

    n_dists = len(dist_data)
    agree_matrix = np.full((n_dists, n_bins), np.nan)

    for row_i, (_, arr) in enumerate(dist_data):
        for b in range(n_bins):
            lo, hi = int(bin_edges[b]), int(bin_edges[b + 1])
            agree_matrix[row_i, b] = float(arr[:, lo:hi].mean())

    row_labels = [label for label, _ in dist_data]

    fig, ax = plt.subplots(figsize=(max(6, n_bins * 2.5), max(3, n_dists * 0.9)))
    im = ax.imshow(agree_matrix, aspect='auto', cmap='RdYlGn', vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, label='Greedy action agreement')
    ax.set_xticks(range(n_bins))
    ax.set_xticklabels(bin_labels, rotation=0, ha='center')
    ax.set_yticks(range(n_dists))
    ax.set_yticklabels(row_labels)
    ax.set_xlabel('Episode timestep bin')
    ax.set_ylabel('Reward distribution')
    ax.set_title('Action Agreement by Reward Distribution and Timestep Phase')
    for r in range(n_dists):
        for b in range(n_bins):
            val = agree_matrix[r, b]
            if not np.isnan(val):
                ax.text(b, r, f'{val:.2f}', ha='center', va='center',
                        fontsize=9, color='black' if 0.3 < val < 0.85 else 'white')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


# ---------------------------------------------------------------------------
# Part 4: Regret — autonomous transformer vs Q-learner baselines
# ---------------------------------------------------------------------------

def run_q_learner_autonomous(
    P: np.ndarray,
    R: np.ndarray,
    n_states: int,
    n_actions: int,
    n_steps: int,
    alpha: float,
    gamma: float,
    epsilon: float,
    rng: np.random.Generator,
) -> np.ndarray:
    Q = np.zeros((n_states, n_actions), dtype=np.float32)
    s = int(rng.integers(n_states))
    rewards = np.zeros(n_steps, dtype=np.float32)

    for t in range(n_steps):
        if rng.random() < epsilon:
            a = int(rng.integers(n_actions))
        else:
            best = float(np.max(Q[s]))
            ties = [ac for ac in range(n_actions) if Q[s, ac] == best]
            a = int(rng.choice(ties))

        r = float(R[s, a])
        s_next = int(rng.choice(n_states, p=P[s, a]))
        Q[s, a] = (1.0 - alpha) * Q[s, a] + alpha * (r + gamma * float(np.max(Q[s_next])))
        rewards[t] = r
        s = s_next

    return rewards


def run_transformer_autonomous(
    model: COCONUTTransformer,
    P: np.ndarray,
    R: np.ndarray,
    n_states: int,
    n_actions: int,
    n_steps: int,
    vocab: Dict,
    config: COCONUTConfig,
    device: torch.device,
    epsilon: float,
    rng: np.random.Generator,
) -> np.ndarray:
    max_actions = config.max_actions
    model.eval()
    context = model.get_init_context(1, n_actions, device)
    s = int(rng.integers(n_states))
    rewards = np.zeros(n_steps, dtype=np.float32)

    with torch.no_grad():
        for t in range(n_steps):
            if rng.random() < epsilon:
                a = int(rng.integers(n_actions))
            else:
                best = float(np.max(np.zeros(n_actions))) if t == 0 else None
                a = int(rng.integers(n_actions)) if t == 0 else pred_a

            r = float(R[s, a])
            s_next = int(rng.choice(n_states, p=P[s, a]))
            rewards[t] = r

            tr = {'s': s, 'a': a, 'r': r, 's_next': s_next, 'a_star': 0}
            token_list, r_off, s_off, u_off = build_step_tokens(tr, vocab, n_actions)
            token_ids = torch.tensor([token_list], dtype=torch.long, device=device)
            reward_val = torch.tensor([r], dtype=torch.float32, device=device)

            select_logits, update_hidden = model.forward_step(
                token_ids=token_ids,
                reward_value=reward_val,
                reward_offset=r_off,
                select_offset=s_off,
                update_offset=u_off,
                context=context,
            )

            if n_actions < max_actions:
                select_logits[:, n_actions:] = float('-inf')
            pred_a = int(select_logits[0].argmax().item())

            new_context = context.clone()
            new_context[0, a, :] = update_hidden[0]
            context = new_context
            s = s_next

    return rewards


def plot_regret(
    cumrew_transformer: np.ndarray,
    cumrew_greedy: np.ndarray,
    cumrew_epsgreedy: np.ndarray,
    n_mdps: int,
    save_path: str,
) -> None:
    n_steps = cumrew_transformer.shape[1]
    steps = np.arange(1, n_steps + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for label, data, color in [
        ('Transformer', cumrew_transformer, 'steelblue'),
        ('Greedy Q (ε=0)', cumrew_greedy, 'darkorange'),
        ('ε-greedy Q (ε=0.2)', cumrew_epsgreedy, 'forestgreen'),
    ]:
        mean = data.mean(axis=0)
        std = data.std(axis=0)
        ax1.plot(steps, mean, color=color, linewidth=2, label=label)
        ax1.fill_between(steps, mean - std, mean + std, alpha=0.15, color=color)

    ax1.set_xlabel('Timestep')
    ax1.set_ylabel('Cumulative Reward')
    ax1.set_title(f'Cumulative Reward (mean ± std, {n_mdps} MDPs)')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    regret_vs_greedy = cumrew_greedy - cumrew_transformer
    regret_vs_epsgreedy = cumrew_epsgreedy - cumrew_transformer

    for label, data, color in [
        ('vs Greedy Q (ε=0)', regret_vs_greedy, 'darkorange'),
        ('vs ε-greedy Q (ε=0.2)', regret_vs_epsgreedy, 'forestgreen'),
    ]:
        mean = data.mean(axis=0)
        std = data.std(axis=0)
        ax2.plot(steps, mean, color=color, linewidth=2, label=label)
        ax2.fill_between(steps, mean - std, mean + std, alpha=0.15, color=color)

    ax2.axhline(0, color='gray', linestyle=':', linewidth=1, alpha=0.5)
    ax2.set_xlabel('Timestep')
    ax2.set_ylabel('Regret (baseline − transformer)')
    ax2.set_title(f'Regret vs Q-Learning Baselines')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.suptitle('Autonomous Regret Comparison', fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


# ---------------------------------------------------------------------------
# Part 5: Effective alpha/gamma recovery
# ---------------------------------------------------------------------------

def estimate_effective_alpha_gamma(
    probe: ContextQProbe,
    ctx_all: np.ndarray,
    trajectories: List[List[Dict]],
    device: torch.device,
    n_steps: int,
    n_traj: int,
    n_actions: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit (alpha_eff, gamma_eff) per trajectory from context-probe Q dynamics.

    The Q-learning update for each visited transition (s_t, a_t, r_t, s'_t) is:
        ΔQ(s_t, a_t) = α * [r_t + γ * max_{a'} Q_t(s'_t, a') - Q_t(s_t, a_t)]

    Rearranging into a two-feature OLS:
        ΔQ = α * (r_t - Q_t(s_t, a_t))  +  (α*γ) * max_{a'} Q_t(s'_t, a')

    We fit [α, α*γ] then recover γ = (α*γ) / α.

    Notes
    -----
    - Only the visited (s_t, a_t) entry changes each step; fitting all entries
      would inject noise from the many zero-ΔQ entries.
    - Q_t is decoded from the context *before* step t's update.
    - max_{a'} Q_t(s'_t, a') is taken at the specific next state s'_t, not
      over the full Q matrix.
    - The reward r_t is included in the regression target as required by the
      TD update equation.
    """
    probe.eval()
    n_states = probe.linear.out_features

    with torch.no_grad():
        q_pred = probe(
            torch.tensor(ctx_all, dtype=torch.float32, device=device)
        ).cpu().numpy()

    # q_pred_r[i, t, a, s] = probe-decoded Q(s, a) before step t of trajectory i
    q_pred_r = q_pred.reshape(n_traj, n_steps, n_actions, n_states)

    alpha_list = []
    gamma_list = []
    r2_list    = []

    for i in range(n_traj):
        traj = trajectories[i]
        # Use steps 0..n_steps-2 so that Q_{t+1} is always available in the array
        T = min(len(traj) - 1, n_steps - 1)

        ys  = []   # ΔQ(s_t, a_t)
        x1s = []   # r_t - Q_t(s_t, a_t)          coefficient: α
        x2s = []   # max_{a'} Q_t(s'_t, a')        coefficient: α*γ

        for t in range(T):
            tr    = traj[t]
            s, a  = tr['s'], tr['a']
            r     = tr['r']
            s_nxt = tr['s_next']

            Q_sa_t    = float(q_pred_r[i, t,     a, s])
            Q_sa_tp1  = float(q_pred_r[i, t + 1, a, s])
            max_Q_nxt = float(q_pred_r[i, t, :, s_nxt].max())

            ys.append(Q_sa_tp1 - Q_sa_t)
            x1s.append(r - Q_sa_t)
            x2s.append(max_Q_nxt)

        y = np.array(ys,  dtype=np.float64)
        X = np.stack([x1s, x2s], axis=1).astype(np.float64)

        # Unconstrained OLS — we want to see the transformer's actual TD-like
        # dynamics, including cases where the best fit has negative α or γ
        # (which means the trajectory is not consistent with canonical
        # Q-learning). Quality is reported via R² downstream.
        betas, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        alpha_e = float(betas[0])
        # γ = (α·γ) / α; only well-defined when α is far from zero.
        gamma_e = float(betas[1] / alpha_e) if abs(alpha_e) > 1e-4 else float('nan')

        y_pred  = X @ betas
        ss_res  = float(np.sum((y - y_pred) ** 2))
        ss_tot  = float(np.sum((y - y.mean()) ** 2)) + 1e-12
        r2      = 1.0 - ss_res / ss_tot

        alpha_list.append(alpha_e)
        gamma_list.append(gamma_e)
        r2_list.append(r2)

    return np.array(alpha_list), np.array(gamma_list), np.array(r2_list)


def plot_effective_alpha_gamma(
    alpha_eff: np.ndarray,
    gamma_eff: np.ndarray,
    r2_vals: np.ndarray,
    expert_alpha: float,
    expert_gamma: float,
    save_path: str,
) -> None:
    # Unconstrained OLS — show the full distribution, no clipping. Out-of-range
    # fits (α<0, γ<0, γ>1) are real signal: they say the transformer's TD-like
    # dynamics on that trajectory are not consistent with canonical Q-learning.
    valid = np.isfinite(alpha_eff) & np.isfinite(gamma_eff)
    a = alpha_eff[valid]
    g = gamma_eff[valid]
    r = r2_vals[valid]

    n_total   = int(valid.sum())
    in_region = (a >= 0) & (a <= 1) & (g >= 0) & (g <= 1)
    n_in_box  = int(in_region.sum())
    frac_box  = n_in_box / max(n_total, 1)

    fig = plt.figure(figsize=(13, 5))
    gs = fig.add_gridspec(1, 3, width_ratios=[2, 1, 1], wspace=0.35)
    ax_scatter = fig.add_subplot(gs[0])
    ax_alpha = fig.add_subplot(gs[1])
    ax_gamma = fig.add_subplot(gs[2])

    sc = ax_scatter.scatter(a, g, c=r, cmap='plasma', alpha=0.7, s=30, vmin=0, vmax=1)
    plt.colorbar(sc, ax=ax_scatter, label='Regression R²')

    # Shade the canonical Q-learning region [0,1]×[0,1] so the eye can tell
    # apart "transformer-like-Q-learning" fits from the rest.
    ax_scatter.axvspan(0, 1, ymin=0, ymax=1, alpha=0.0)  # placeholder; use Rectangle for clarity
    from matplotlib.patches import Rectangle
    ax_scatter.add_patch(Rectangle(
        (0, 0), 1, 1, linewidth=1.2, edgecolor='black', facecolor='black',
        alpha=0.06, zorder=0, label=f'Valid Q-learning region [0,1]² ({frac_box:.0%} of fits)'
    ))
    ax_scatter.axhline(0, color='gray', linewidth=0.8, linestyle=':', alpha=0.7)
    ax_scatter.axvline(0, color='gray', linewidth=0.8, linestyle=':', alpha=0.7)

    ax_scatter.axvline(expert_alpha, color='red', linewidth=2, linestyle='--',
                       label=f'Expert α = {expert_alpha}')
    ax_scatter.axhline(expert_gamma, color='blue', linewidth=2, linestyle='--',
                       label=f'Expert γ = {expert_gamma}')
    ax_scatter.set_xlabel('α_eff')
    ax_scatter.set_ylabel('γ_eff')
    ax_scatter.set_title('Fitted (α_eff, γ_eff) per trajectory (unconstrained OLS)')
    ax_scatter.legend(fontsize=8, loc='best')
    ax_scatter.grid(True, alpha=0.3)

    ax_alpha.hist(a, bins=20, color='steelblue', edgecolor='white', alpha=0.8)
    ax_alpha.axvline(expert_alpha, color='red', linewidth=2, linestyle='--')
    ax_alpha.axvline(float(np.nanmedian(a)), color='navy', linewidth=1.5, linestyle=':',
                     label=f'Median={np.nanmedian(a):.3f}')
    ax_alpha.set_xlabel('α_eff')
    ax_alpha.set_title('α_eff distribution')
    ax_alpha.legend(fontsize=7)
    ax_alpha.grid(True, alpha=0.3)

    ax_gamma.hist(g, bins=20, color='darkorange', edgecolor='white', alpha=0.8)
    ax_gamma.axvline(expert_gamma, color='blue', linewidth=2, linestyle='--')
    ax_gamma.axvline(float(np.nanmedian(g)), color='darkred', linewidth=1.5, linestyle=':',
                     label=f'Median={np.nanmedian(g):.3f}')
    ax_gamma.set_xlabel('γ_eff')
    ax_gamma.set_title('γ_eff distribution')
    ax_gamma.legend(fontsize=7)
    ax_gamma.grid(True, alpha=0.3)

    fig.suptitle('Effective Q-Learning Hyperparameters Recovered from Context Probe', fontsize=12)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_training_curves(log_path: str, save_path: str) -> None:
    if not os.path.exists(log_path):
        print(f"  Warning: training log not found at {log_path}, skipping.")
        return
    data = np.load(log_path)
    steps = data['steps']
    train_ce = data['train_ce']
    train_acc = data['train_acc']
    val_ce = data['val_ce']
    val_acc = data['val_acc']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(steps, train_ce, color='steelblue', linewidth=1.5, label='Train CE')
    ax1.plot(steps, val_ce, color='darkorange', linewidth=1.5, label='Val CE')
    ax1.set_xlabel('Step')
    ax1.set_ylabel('CE Loss')
    ax1.set_title('CE Loss Over Training')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(steps, np.array(train_acc) * 100, color='steelblue', linewidth=1.5, label='Train Acc')
    ax2.plot(steps, np.array(val_acc) * 100, color='darkorange', linewidth=1.5, label='Val Acc')
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Accuracy Over Training')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.suptitle('Recurrent Context Q-Learning — Training Curves', fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


# ---------------------------------------------------------------------------
# Part 6: Reward probe from context delta
# ---------------------------------------------------------------------------

def collect_reward_probe_data(
    model: COCONUTTransformer,
    n_trajectories: int,
    n_states: int,
    n_actions: int,
    vocab: Dict,
    config: COCONUTConfig,
    device: torch.device,
    n_steps: int = 50,
    seed_offset: int = 40000,
) -> Tuple[np.ndarray, np.ndarray]:
    """Collect (Δc_{a_t}, r_t) pairs per step.

    Δc_{a_t} = update_hidden - context[a_t]   (residual written into the
    chosen action's slot at step t). One pair per (trajectory, step).
    """
    model.eval()
    d_model = config.d_model

    N = n_trajectories * n_steps
    delta_all = np.empty((N, d_model), dtype=np.float32)
    r_all = np.empty(N, dtype=np.float32)
    write_idx = 0

    with torch.no_grad():
        for i in range(n_trajectories):
            seed = seed_offset + i
            P, R = generate_eval_mdp(n_states, n_actions, seed=seed)
            trajectory, _ = run_tabular_q_learning(
                P, R, n_states, n_actions, n_steps=n_steps, seed=seed
            )

            context = model.get_init_context(1, n_actions, device)

            for t, tr in enumerate(trajectory):
                token_list, r_off, s_off, u_off = build_step_tokens(tr, vocab, n_actions)
                token_ids = torch.tensor([token_list], dtype=torch.long, device=device)
                reward_val = torch.tensor([tr['r']], dtype=torch.float32, device=device)

                _, update_hidden = model.forward_step(
                    token_ids=token_ids,
                    reward_value=reward_val,
                    reward_offset=r_off,
                    select_offset=s_off,
                    update_offset=u_off,
                    context=context,
                )

                a_t = tr['a']
                prev_slot = context[0, a_t, :].detach().cpu().numpy()
                new_slot = update_hidden[0].detach().cpu().numpy()
                delta_all[write_idx] = new_slot - prev_slot
                r_all[write_idx] = tr['r']
                write_idx += 1

                new_context = context.clone()
                new_context[0, a_t, :] = update_hidden[0]
                context = new_context

                del token_ids, reward_val, update_hidden

    return delta_all, r_all


class RewardProbe(nn.Module):
    """Linear probe: Δc in R^{d_model} -> scalar reward r."""
    def __init__(self, d_model: int):
        super().__init__()
        self.linear = nn.Linear(d_model, 1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.linear(h).squeeze(-1)


def train_reward_probe(
    probe: RewardProbe,
    delta_all: np.ndarray,
    r_all: np.ndarray,
    device: torch.device,
    n_epochs: int = 10,
    batch_size: int = 256,
    lr: float = 1e-3,
) -> List[float]:
    probe.train()
    optimizer = torch.optim.Adam(probe.parameters(), lr=lr)
    N = delta_all.shape[0]
    losses = []

    d_t = torch.tensor(delta_all, dtype=torch.float32, device=device)
    r_t = torch.tensor(r_all, dtype=torch.float32, device=device)

    for _ in range(n_epochs):
        perm = torch.randperm(N, device=device)
        epoch_loss = 0.0
        n_batches = 0
        for i in range(0, N, batch_size):
            idx = perm[i:i + batch_size]
            pred = probe(d_t[idx])
            loss = F.mse_loss(pred, r_t[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        losses.append(epoch_loss / max(n_batches, 1))

    return losses


def evaluate_reward_probe(
    probe: RewardProbe,
    delta_all: np.ndarray,
    r_all: np.ndarray,
    device: torch.device,
) -> Tuple[float, float, np.ndarray]:
    probe.eval()
    with torch.no_grad():
        d_t = torch.tensor(delta_all, dtype=torch.float32, device=device)
        r_pred = probe(d_t).cpu().numpy()

    ss_res = float(np.sum((r_all - r_pred) ** 2))
    ss_tot = float(np.sum((r_all - r_all.mean()) ** 2)) + 1e-12
    r2 = float(1.0 - ss_res / ss_tot)
    mae = float(np.mean(np.abs(r_all - r_pred)))
    return r2, mae, r_pred


def plot_reward_probe(
    r_true: np.ndarray,
    r_pred: np.ndarray,
    r2: float,
    mae: float,
    save_path: str,
) -> None:
    rng = np.random.default_rng(0)
    n = len(r_true)
    idx = rng.choice(n, size=min(10000, n), replace=False)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    ax1.scatter(r_true[idx], r_pred[idx], alpha=0.2, s=6,
                color='seagreen', rasterized=True)
    lo = float(min(r_true.min(), r_pred.min()))
    hi = float(max(r_true.max(), r_pred.max()))
    ax1.plot([lo, hi], [lo, hi], 'r--', linewidth=1.5, label='y = x')
    ax1.set_xlabel('True reward r_t')
    ax1.set_ylabel('Predicted reward (linear probe on Δc)')
    ax1.set_title(f'Reward Probe Scatter\nR² = {r2:.4f}  MAE = {mae:.4f}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    residuals = r_pred - r_true
    ax2.hist(residuals, bins=40, color='seagreen', edgecolor='white', alpha=0.85)
    ax2.axvline(0, color='red', linewidth=1.2, linestyle='--')
    ax2.set_xlabel('Residual (predicted − true)')
    ax2.set_ylabel('Count')
    ax2.set_title(f'Residual Distribution\nmean={residuals.mean():+.4f}  std={residuals.std():.4f}')
    ax2.grid(True, alpha=0.3)

    fig.suptitle('Reward Probe from Context Delta Δc_{a_t}', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def trace_step_log(
    model: COCONUTTransformer,
    q_probe: 'ContextQProbe',
    r_probe: 'RewardProbe',
    n_states: int,
    n_actions: int,
    vocab: Dict,
    config: COCONUTConfig,
    device: torch.device,
    seed: int,
    n_steps: int,
    alpha: float,
    gamma: float,
    epsilon: float,
    save_path: str,
) -> None:
    """Write a per-step text log comparing model context updates to the
    oracle tabular Q-learning update for a single MDP."""
    np.set_printoptions(precision=4, suppress=True, linewidth=140)

    P, R = generate_eval_mdp(n_states, n_actions, seed=seed)
    trajectory, q_snaps = run_tabular_q_learning(
        P, R, n_states, n_actions, n_steps=n_steps,
        alpha=alpha, gamma=gamma, epsilon=epsilon, seed=seed,
    )
    # q_snaps[t] is the Q table AFTER step t's oracle update.
    q_before_all = np.concatenate(
        [np.zeros((1, n_states, n_actions), dtype=np.float32), q_snaps[:-1]], axis=0,
    )

    lines: List[str] = []
    lines.append("=" * 90)
    lines.append(f"PER-STEP TRACE  |  seed={seed}  n_states={n_states}  "
                 f"n_actions={n_actions}  n_steps={n_steps}")
    lines.append(f"alpha={alpha}  gamma={gamma}  epsilon={epsilon}")
    lines.append("=" * 90)
    lines.append("")
    lines.append("MDP transition probabilities P[s, a, s']:")
    for s in range(n_states):
        for a in range(n_actions):
            lines.append(f"  P[s={s}, a={a}] = {P[s, a]}")
    lines.append("")
    lines.append("MDP reward matrix R[s, a]:")
    for s in range(n_states):
        lines.append(f"  R[s={s}] = {R[s]}")
    lines.append("")

    model.eval()
    context = model.get_init_context(1, n_actions, device)
    prev_delta_per_action: Dict[int, np.ndarray] = {}

    with torch.no_grad():
        for t, tr in enumerate(trajectory):
            s, a, r, s_next = tr['s'], tr['a'], tr['r'], tr['s_next']

            # ---- Oracle Q-learning update ----
            Q_before = q_before_all[t]
            Q_after = q_snaps[t]
            q_sa_before = float(Q_before[s, a])
            max_q_next = float(Q_before[s_next].max())
            td_target = r + gamma * max_q_next
            td_error = td_target - q_sa_before
            q_sa_after = float(Q_after[s, a])
            a_star_oracle = int(np.argmax(Q_after[s_next]))

            # ---- Model forward step ----
            token_list, r_off, s_off, u_off = build_step_tokens(tr, vocab, n_actions)
            token_ids = torch.tensor([token_list], dtype=torch.long, device=device)
            reward_val = torch.tensor([r], dtype=torch.float32, device=device)

            select_logits, update_hidden = model.forward_step(
                token_ids=token_ids,
                reward_value=reward_val,
                reward_offset=r_off,
                select_offset=s_off,
                update_offset=u_off,
                context=context,
            )
            if n_actions < config.max_actions:
                select_logits[:, n_actions:] = float('-inf')
            model_action = int(select_logits[0].argmax().item())

            ctx_before_a = context[0, a, :].detach().cpu().numpy()
            ctx_after_a = update_hidden[0].detach().cpu().numpy()
            delta = ctx_after_a - ctx_before_a
            delta_norm = float(np.linalg.norm(delta))
            ctx_before_norm = float(np.linalg.norm(ctx_before_a))
            ctx_after_norm = float(np.linalg.norm(ctx_after_a))

            cos_prev = None
            if a in prev_delta_per_action:
                pd = prev_delta_per_action[a]
                denom = float(np.linalg.norm(pd) * delta_norm)
                if denom > 1e-12:
                    cos_prev = float(np.dot(pd, delta) / denom)
            prev_delta_per_action[a] = delta.copy()

            # ---- Probes ----
            delta_t = torch.from_numpy(delta).unsqueeze(0).to(device)
            r_hat = float(r_probe(delta_t).item())

            ctx_before_t = torch.from_numpy(ctx_before_a).unsqueeze(0).to(device)
            ctx_after_t = torch.from_numpy(ctx_after_a).unsqueeze(0).to(device)
            q_pred_before = q_probe(ctx_before_t)[0].cpu().numpy()  # (n_states,)
            q_pred_after = q_probe(ctx_after_t)[0].cpu().numpy()
            q_oracle_col_before = Q_before[:, a]
            q_oracle_col_after = Q_after[:, a]

            # ---- Write step block ----
            lines.append(f"--- Step {t:3d} ---")
            lines.append(f"  Transition: s={s}  a={a}  r={r:+.4f}  s'={s_next}")
            lines.append(f"  ORACLE Q-learning:")
            lines.append(f"    Q[s={s},a={a}] before = {q_sa_before:+.4f}")
            lines.append(f"    max_a Q[s'={s_next},a] = {max_q_next:+.4f}")
            lines.append(f"    TD target = r + gamma * max = {td_target:+.4f}")
            lines.append(f"    TD error  = target - Q_before = {td_error:+.4f}")
            lines.append(f"    Q[s={s},a={a}] after  = {q_sa_after:+.4f}  "
                         f"(delta = {q_sa_after - q_sa_before:+.4f})")
            lines.append(f"    argmax_a Q[s'={s_next}] = a*={a_star_oracle}")
            lines.append(f"  MODEL action choice this step: a_hat={model_action}  "
                         f"(oracle a*={tr['a_star']})")
            lines.append(f"  MODEL context update (slot a={a}):")
            lines.append(f"    ||c[a]|| before = {ctx_before_norm:.4f}   "
                         f"after = {ctx_after_norm:.4f}   ||delta_c|| = {delta_norm:.4f}")
            if cos_prev is not None:
                lines.append(f"    cosine(delta_c_t, delta_c_prev for same a) = {cos_prev:+.4f}")
            lines.append(f"    reward probe on delta_c: r_hat = {r_hat:+.4f}   "
                         f"(true r = {r:+.4f}   |err| = {abs(r_hat - r):.4f})")
            lines.append(f"    Q-probe on c[a={a}]   BEFORE: pred Q[:,{a}] = {q_pred_before}")
            lines.append(f"                                    oracle Q[:,{a}] = {q_oracle_col_before}")
            lines.append(f"    Q-probe on c[a={a}]   AFTER : pred Q[:,{a}] = {q_pred_after}")
            lines.append(f"                                    oracle Q[:,{a}] = {q_oracle_col_after}")
            lines.append("")

            # advance context
            new_context = context.clone()
            new_context[0, a, :] = update_hidden[0]
            context = new_context

    lines.append("=" * 90)
    lines.append("End of trace.")
    lines.append("=" * 90)

    with open(save_path, 'w') as f:
        f.write("\n".join(lines))
    print(f"  Wrote step-trace log to {save_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate trained recurrent context Q-learning transformer')
    parser.add_argument('--checkpoint', type=str,
                        default=os.path.join(_script_dir, '..', 'checkpoints',
                                             'coconut_transformer.pt'))
    parser.add_argument('--figures_dir', type=str,
                        default=os.path.join(_script_dir, '..', 'figures'))
    parser.add_argument('--n_steps',       type=int,   default=50)
    parser.add_argument('--alpha',         type=float, default=0.1)
    parser.add_argument('--gamma',         type=float, default=0.9)
    parser.add_argument('--epsilon',       type=float, default=0.2)
    parser.add_argument('--eval_seed',     type=int,   default=9999)
    parser.add_argument('--n_eval_mdps',   type=int,   default=10)
    parser.add_argument('--n_probe_train', type=int,   default=1000)
    parser.add_argument('--n_probe_eval',  type=int,   default=100)
    parser.add_argument('--probe_epochs',  type=int,   default=10)
    args = parser.parse_args()

    os.makedirs(args.figures_dir, exist_ok=True)

    # ---- Load checkpoint ----
    print(f"Loading checkpoint from {args.checkpoint} ...")
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    config = COCONUTConfig.from_dict(ckpt['config'])
    model = COCONUTTransformer(config)
    model.load_state_dict(ckpt['model_state_dict'])

    print(f"  Loaded epoch={ckpt.get('epoch', '?')}, step={ckpt.get('step', '?')}, "
          f"val_ce={ckpt.get('val_ce_loss', ckpt.get('val_loss', '?')):.4f}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    vocab = build_vocab(config.max_states, config.max_actions)
    n_states = config.max_states
    n_actions = config.max_actions

    print(f"\nModel: {model.num_parameters():,} params  |  device: {device}")
    print(f"MDP:   n_states={n_states}, n_actions={n_actions}")
    print(f"FFNs:  {'enabled' if config.use_ffns else 'DISABLED'}")

    eval_seeds = list(range(args.eval_seed, args.eval_seed + args.n_eval_mdps))

    # -----------------------------------------------------------------------
    # Part 1a: In-distribution action prediction
    # -----------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print(f"Part 1a: In-distribution action prediction ({args.n_eval_mdps} MDPs)")
    print(f"{'=' * 60}")

    all_agreements_id = []
    traj_preds_id = []
    for seed_i, seed in enumerate(eval_seeds):
        print(f"  ID MDP {seed_i+1}/{args.n_eval_mdps} (seed={seed}) ...", end=' ', flush=True)
        P, R = generate_eval_mdp(n_states, n_actions, seed=seed)
        trajectory, q_snaps = run_tabular_q_learning(
            P, R, n_states, n_actions, n_steps=args.n_steps,
            alpha=args.alpha, gamma=args.gamma, epsilon=args.epsilon, seed=seed,
        )
        preds, _ = run_action_inference(model, trajectory, vocab, n_actions,
                                        config, device)
        targets = np.array([step['a_star'] for step in trajectory], dtype=np.int32)
        agree = (preds == targets).astype(np.float32)
        all_agreements_id.append(agree)
        traj_preds_id.append((trajectory, preds))
        print(f"agree={agree.mean():.2%}")

    aa_id_arr = np.stack(all_agreements_id, axis=0)
    aa_id_mean = aa_id_arr.mean(axis=0)
    aa_id_std = aa_id_arr.std(axis=0)
    print(f"\n  Mean ID agreement: "
          f"{aa_id_mean.mean():.2%} +/- {aa_id_arr.mean(axis=1).std():.4f}")

    # -----------------------------------------------------------------------
    # Part 1b: OOD variants
    # -----------------------------------------------------------------------
    ood_plot_results: List[Tuple[str, np.ndarray, np.ndarray]] = []
    ood_agreements: List[Tuple[str, np.ndarray]] = []

    for v_idx, (variant, variant_desc) in enumerate(OOD_VARIANTS):
        v_seeds = list(range(
            args.eval_seed + 50000 + v_idx * 10000,
            args.eval_seed + 50000 + v_idx * 10000 + args.n_eval_mdps,
        ))
        print(f"\n{'=' * 60}")
        print(f"Part 1b-{v_idx+1}: OOD '{variant}' — {variant_desc}")
        print(f"{'=' * 60}")

        all_agreements_v = []
        for seed_i, seed in enumerate(v_seeds):
            print(f"  {variant} {seed_i+1}/{args.n_eval_mdps} ...", end=' ', flush=True)
            P, R = generate_ood_mdp(n_states, n_actions, variant=variant, seed=seed)
            trajectory, q_snaps_v = run_tabular_q_learning(
                P, R, n_states, n_actions, n_steps=args.n_steps,
                alpha=args.alpha, gamma=args.gamma, epsilon=args.epsilon, seed=seed,
            )
            preds, _ = run_action_inference(model, trajectory, vocab, n_actions,
                                            config, device)
            targets = np.array([step['a_star'] for step in trajectory], dtype=np.int32)
            agree = (preds == targets).astype(np.float32)
            all_agreements_v.append(agree)
            print(f"agree={agree.mean():.2%}")

        aa_v_arr = np.stack(all_agreements_v, axis=0)
        aa_v_mean = aa_v_arr.mean(axis=0)
        aa_v_std = aa_v_arr.std(axis=0)
        gap = aa_id_mean.mean() - aa_v_mean.mean()
        print(f"\n  Mean {variant}: "
              f"{aa_v_mean.mean():.2%} +/- {aa_v_arr.mean(axis=1).std():.4f} (ID gap: {gap:+.4f})")
        ood_plot_results.append((variant_desc, aa_v_mean, aa_v_std))
        ood_agreements.append((variant, aa_v_arr))

    plot_action_agreement(
        aa_id_mean, aa_id_std, ood_plot_results,
        save_path=os.path.join(args.figures_dir, 'action_agreement.png'),
        n_mdps=args.n_eval_mdps, label_suffix='recurrent context',
    )

    # -----------------------------------------------------------------------
    # Part 2: Context token Q-value probing
    # -----------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("Part 2: Context token Q-value probing")
    print(f"{'=' * 60}")

    for p in model.parameters():
        p.requires_grad_(False)

    print(f"\nCollecting probe training data ({args.n_probe_train} trajectories) ...")
    ctx_tr, q_tr, _ = collect_context_probe_data(
        model, args.n_probe_train, n_states, n_actions, vocab,
        config, device, n_steps=args.n_steps, seed_offset=20000,
    )
    print(f"  ctx: {ctx_tr.shape}, q_target: {q_tr.shape}")

    print(f"\nCollecting probe eval data ({args.n_probe_eval} trajectories) ...")
    ctx_ev, q_ev, traj_ev = collect_context_probe_data(
        model, args.n_probe_eval, n_states, n_actions, vocab,
        config, device, n_steps=args.n_steps, seed_offset=30000,
    )

    print(f"\nTraining context Q-probe ...")
    probe = ContextQProbe(config.d_model, n_states).to(device)
    losses = train_probe(probe, ctx_tr, q_tr, device, n_epochs=args.probe_epochs)
    print(f"  MSE per epoch: {['%.4f' % l for l in losses]}")

    r2, frob_mean, q_pred = evaluate_probe(probe, ctx_ev, q_ev, device)
    print(f"  Eval: R2={r2:.4f}  Frobenius={frob_mean:.4f}")

    plot_probe_scatter(
        q_ev, q_pred, r2=r2,
        save_path=os.path.join(args.figures_dir, 'probe_scatter.png'),
    )
    plot_probe_frobenius(
        q_pred, q_ev, n_steps=args.n_steps, n_traj=args.n_probe_eval,
        n_actions=n_actions,
        save_path=os.path.join(args.figures_dir, 'probe_frobenius.png'),
    )

    # -----------------------------------------------------------------------
    # Part 3: Attention heatmap
    # -----------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("Attention heatmap (SELECT & UPDATE)")
    print(f"{'=' * 60}")
    _attn_seed = eval_seeds[0]
    P_attn, R_attn = generate_eval_mdp(n_states, n_actions, seed=_attn_seed)
    traj_attn, _ = run_tabular_q_learning(
        P_attn, R_attn, n_states, n_actions,
        n_steps=args.n_steps, alpha=args.alpha, gamma=args.gamma,
        epsilon=args.epsilon, seed=_attn_seed,
    )
    plot_attention_heatmap(
        model, traj_attn, vocab, n_actions, config, device,
        save_path=os.path.join(args.figures_dir, 'attention_heatmap.png'),
    )

    # ---- Full T×T causal attention on a fixed 4-state, 2-action sequence ----
    fixed_S, fixed_A = 4, 2
    if config.max_states >= fixed_S and config.max_actions >= fixed_A:
        print(f"\n  Full causal-attention heatmap on fixed "
              f"{fixed_S}-state / {fixed_A}-action MDP ...")
        P_fix, R_fix = generate_eval_mdp(fixed_S, fixed_A, seed=_attn_seed + 1)
        traj_fix, _ = run_tabular_q_learning(
            P_fix, R_fix, fixed_S, fixed_A,
            n_steps=args.n_steps, alpha=args.alpha, gamma=args.gamma,
            epsilon=args.epsilon, seed=_attn_seed + 1,
        )
        plot_full_attention_heatmap(
            model, traj_fix, vocab, fixed_A, config, device,
            save_path=os.path.join(args.figures_dir,
                                   'attention_full_heatmap_4s2a.png'),
        )
    else:
        print(f"\n  Skipping full-attention heatmap: model max_states="
              f"{config.max_states}, max_actions={config.max_actions} "
              f"< required {fixed_S}/{fixed_A}.")

    # -----------------------------------------------------------------------
    # Per-distribution agreement heatmap
    # -----------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("Per-distribution agreement heatmap")
    print(f"{'=' * 60}")
    dist_data: List[Tuple[str, np.ndarray]] = [('ID (Beta 2,2)', aa_id_arr)]
    for variant, aa_v in ood_agreements:
        dist_data.append((DIST_SHORT_LABELS.get(variant, variant), aa_v))
    plot_per_distribution_agreement(
        dist_data, args.n_steps,
        os.path.join(args.figures_dir, 'per_state_agreement.png'),
        n_bins=3,
    )

    # -----------------------------------------------------------------------
    # Part 4: Regret comparison (autonomous transformer vs Q-learners)
    # -----------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print(f"Part 4: Regret comparison ({args.n_eval_mdps} MDPs, autonomous)")
    print(f"{'=' * 60}")

    all_cumrew_transformer = []
    all_cumrew_greedy = []
    all_cumrew_epsgreedy = []

    for seed_i, seed in enumerate(eval_seeds):
        print(f"  Regret MDP {seed_i+1}/{args.n_eval_mdps} (seed={seed}) ...", flush=True)
        P, R = generate_eval_mdp(n_states, n_actions, seed=seed)

        rng_t = np.random.default_rng(seed + 100)
        rng_g = np.random.default_rng(seed + 100)
        rng_e = np.random.default_rng(seed + 100)

        rew_t = run_transformer_autonomous(
            model, P, R, n_states, n_actions, args.n_steps,
            vocab, config, device, epsilon=0.0, rng=rng_t,
        )
        rew_g = run_q_learner_autonomous(
            P, R, n_states, n_actions, args.n_steps,
            alpha=args.alpha, gamma=args.gamma, epsilon=0.0, rng=rng_g,
        )
        rew_e = run_q_learner_autonomous(
            P, R, n_states, n_actions, args.n_steps,
            alpha=args.alpha, gamma=args.gamma, epsilon=args.epsilon, rng=rng_e,
        )

        all_cumrew_transformer.append(np.cumsum(rew_t))
        all_cumrew_greedy.append(np.cumsum(rew_g))
        all_cumrew_epsgreedy.append(np.cumsum(rew_e))

    cumrew_t = np.stack(all_cumrew_transformer, axis=0)
    cumrew_g = np.stack(all_cumrew_greedy, axis=0)
    cumrew_e = np.stack(all_cumrew_epsgreedy, axis=0)

    print(f"\n  Final cumulative reward (mean over {args.n_eval_mdps} MDPs):")
    print(f"    Transformer:    {cumrew_t[:, -1].mean():.2f} ± {cumrew_t[:, -1].std():.2f}")
    print(f"    Greedy Q (ε=0): {cumrew_g[:, -1].mean():.2f} ± {cumrew_g[:, -1].std():.2f}")
    print(f"    ε-greedy Q:     {cumrew_e[:, -1].mean():.2f} ± {cumrew_e[:, -1].std():.2f}")

    plot_regret(
        cumrew_t, cumrew_g, cumrew_e,
        n_mdps=args.n_eval_mdps,
        save_path=os.path.join(args.figures_dir, 'regret.png'),
    )

    # -----------------------------------------------------------------------
    # Part 5: Effective alpha/gamma recovery
    # -----------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("Part 5: Effective alpha/gamma recovery")
    print(f"{'=' * 60}")

    alpha_eff, gamma_eff, r2_vals = estimate_effective_alpha_gamma(
        probe, ctx_ev, traj_ev, device,
        n_steps=args.n_steps, n_traj=args.n_probe_eval, n_actions=n_actions,
    )
    valid_mask = np.isfinite(alpha_eff) & np.isfinite(gamma_eff)
    print(f"  Valid fits: {valid_mask.sum()}/{len(valid_mask)}")
    print(f"  α_eff: median={np.nanmedian(alpha_eff):.4f}  mean={np.nanmean(alpha_eff):.4f}")
    print(f"  γ_eff: median={np.nanmedian(gamma_eff):.4f}  mean={np.nanmean(gamma_eff):.4f}")
    print(f"  R²:    median={np.nanmedian(r2_vals):.4f}  mean={np.nanmean(r2_vals):.4f}")

    plot_effective_alpha_gamma(
        alpha_eff, gamma_eff, r2_vals,
        expert_alpha=args.alpha, expert_gamma=args.gamma,
        save_path=os.path.join(args.figures_dir, 'effective_alpha_gamma.png'),
    )

    # -----------------------------------------------------------------------
    # Part 6: Reward probe from context delta
    # -----------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("Part 6: Reward probe from context delta")
    print(f"{'=' * 60}")

    print(f"\nCollecting reward-probe training data ({args.n_probe_train} trajectories) ...")
    delta_tr, r_tr = collect_reward_probe_data(
        model, args.n_probe_train, n_states, n_actions, vocab,
        config, device, n_steps=args.n_steps, seed_offset=40000,
    )
    print(f"  delta: {delta_tr.shape}, r: {r_tr.shape}")

    print(f"\nCollecting reward-probe eval data ({args.n_probe_eval} trajectories) ...")
    delta_ev, r_ev = collect_reward_probe_data(
        model, args.n_probe_eval, n_states, n_actions, vocab,
        config, device, n_steps=args.n_steps, seed_offset=50000,
    )

    print(f"\nTraining reward probe ...")
    r_probe = RewardProbe(config.d_model).to(device)
    r_losses = train_reward_probe(
        r_probe, delta_tr, r_tr, device, n_epochs=args.probe_epochs,
    )
    print(f"  MSE per epoch: {['%.4f' % l for l in r_losses]}")

    r2_r, mae_r, r_pred = evaluate_reward_probe(r_probe, delta_ev, r_ev, device)
    print(f"  Eval: R²={r2_r:.4f}  MAE={mae_r:.4f}")

    plot_reward_probe(
        r_ev, r_pred, r2=r2_r, mae=mae_r,
        save_path=os.path.join(args.figures_dir, 'reward_probe.png'),
    )

    # -----------------------------------------------------------------------
    # Part 6b: Per-step trace logs (1-2 single MDPs)
    # -----------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("Part 6b: Per-step trace logs")
    print(f"{'=' * 60}")
    trace_dir = os.path.join(args.figures_dir, 'step_traces')
    os.makedirs(trace_dir, exist_ok=True)
    trace_seeds = [args.eval_seed, args.eval_seed + 1]
    for tseed in trace_seeds:
        out_path = os.path.join(trace_dir, f'step_trace_seed{tseed}.txt')
        trace_step_log(
            model=model, q_probe=probe, r_probe=r_probe,
            n_states=n_states, n_actions=n_actions, vocab=vocab,
            config=config, device=device,
            seed=tseed, n_steps=args.n_steps,
            alpha=args.alpha, gamma=args.gamma, epsilon=args.epsilon,
            save_path=out_path,
        )

    # -----------------------------------------------------------------------
    # Training curves
    # -----------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("Training curves")
    print(f"{'=' * 60}")
    log_path = os.path.join(os.path.dirname(args.checkpoint), 'training_log.npz')
    plot_training_curves(log_path, os.path.join(args.figures_dir, 'training_curves.png'))

    print(f"\nDone. Figures saved to {args.figures_dir}/")


if __name__ == '__main__':
    main()
