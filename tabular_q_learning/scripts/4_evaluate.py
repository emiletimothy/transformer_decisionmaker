#!/usr/bin/env python3
"""
4_evaluate.py — Evaluation Script (Two-Phase Per-Round Layout)

Tests the trained COCONUTTransformer on complementary metrics.

Part 1 — Action prediction evaluation (in-distribution + OOD)
--------------------------------------------------------------
Runs action agreement evaluation on two MDP regimes (ID + 5 OOD variants).
For each regime, feeds trajectory step-by-step using forward_hao with the
checkpoint's stage_idx. At each SELECT position, records argmax action vs.
Q-learning greedy. Reports per-step agreement mean ± std over n_eval_mdps MDPs.

Part 2 — Q-value probing (emergence evidence)
-----------------------------------------------
Freeze the model and train three linear probes mapping hidden states to the
full Q-table:
    probe_select  : hidden at SELECT positions -> Q(n_states, n_actions)
    probe_update  : hidden at UPDATE positions -> Q(n_states, n_actions)
    probe_cot     : hidden at COT positions    -> Q(n_states, n_actions)
                    (only for continuous rounds; skipped at stage 0)

The UPDATE and COT probes are the primary emergence evidence: Q-values are
never directly supervised at those positions, yet a linear probe should
recover them if the model has internalized Q-value tracking.

Output files
------------
    figures/action_agreement.png      — per-step agreement mean ± std (ID vs OOD)
    figures/probe_scatter.png         — probed vs true Q-values scatter
    figures/probe_frobenius.png       — probe Frobenius error + zero baseline
    figures/attention_heatmap.png     — SELECT and UPDATE attention (two-row)
    figures/training_curves.png       — train/val CE + accuracy over training
    figures/effective_alpha_gamma.png — fitted (alpha_eff, gamma_eff) scatter
    figures/q_convergence.png         — model probe vs tabular convergence to Q*
    figures/per_state_agreement.png   — per-state action agreement heatmap
"""

import argparse
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
# Import COCONUTTransformer and helpers from 2_model.py
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
discretize_q_value = _mod.discretize_q_value

# HAO_STAGES needed for frac_continuous lookup
HAO_STAGES = [
    (5,  50, 0.00, "Stage 0"),
    (10, 50, 0.25, "Stage 1"),
    (15, 50, 0.50, "Stage 2"),
    (20, 50, 0.75, "Stage 3"),
    (25, 50, 1.00, "Stage 4"),
]


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

        trajectory.append({'s': s, 'a': a, 'r': r, 's_next': s_next, 'a_next': a_next})
        q_snapshots.append(Q.copy())
        s = s_next

    return trajectory, np.stack(q_snapshots, axis=0)


# ---------------------------------------------------------------------------
# Sequence builder — two-phase layout for eval
# ---------------------------------------------------------------------------

def _compute_continuous_round_mask(stage_idx: int, n_rounds: int) -> List[bool]:
    """Compute per-round continuous mask from stage_idx and total rounds."""
    frac   = HAO_STAGES[stage_idx][2]
    n_cont = int(math.floor(frac * n_rounds))
    return [True] * n_cont + [False] * (n_rounds - n_cont)


def trajectory_to_tensors(
    trajectory: List[Dict],
    vocab: Dict,
    n_actions: int,
    stage_idx: int,
    config: 'COCONUTConfig',
    device: torch.device,
    n_steps: Optional[int] = None,
    q_snapshots: Optional[np.ndarray] = None,
) -> Dict[str, torch.Tensor]:
    """Build two-phase COCONUT token sequence from a trajectory.

    Each round is either discrete (18 tokens for n_actions=2) or continuous
    (16 tokens), determined by the curriculum mask computed from stage_idx.

    Phase 1 (always): s_t, a_t, R, s_next, [s_next, a_c, EVAL]*n_actions, SELECT
    Phase 2 scaffold: ANEXT, QCURR, QNEXT, UPDATE
    Phase 2 thought:  [s_t, a_t, Q_bin]  (discrete) OR [COT]  (continuous)

    Returns dict with batch-dim=1 tensors for direct model input.
    """
    TOK_S      = vocab['TOK_S']
    TOK_A      = vocab['TOK_A']
    TOK_R      = vocab['TOK_R']
    TOK_EVAL   = vocab['TOK_EVAL']
    TOK_SELECT = vocab['TOK_SELECT']
    TOK_COT    = vocab['TOK_COT']
    TOK_UPDATE = vocab['TOK_UPDATE']
    TOK_QCURR  = vocab['TOK_QCURR']
    TOK_QNEXT  = vocab['TOK_QNEXT']
    TOK_ANEXT  = vocab['TOK_ANEXT']
    TOK_QBIN   = vocab['TOK_QBIN']
    TOK_NULL   = vocab['TOK_NULL']
    TOK_START  = vocab['TOK_START']

    if n_steps is None:
        n_steps = len(trajectory)

    n_q_bins  = config.n_q_bins
    q_bin_min = config.q_bin_min
    q_bin_max = config.q_bin_max

    crm = _compute_continuous_round_mask(stage_idx, n_steps)

    ids               = [TOK_NULL, TOK_START]
    reward_values     = []
    reward_positions  = []
    select_positions  = []
    update_positions  = []
    thought_positions = []   # [n_steps][3]

    for t in range(n_steps):
        step = trajectory[t]
        s    = step['s']
        a    = step['a']
        r    = step['r']
        s_p  = step['s_next']

        # Phase 1: action selection
        ids.append(TOK_S[s])
        ids.append(TOK_A[a])
        reward_positions.append(len(ids))
        reward_values.append(r)
        ids.append(TOK_R)
        ids.append(TOK_S[s_p])

        for c in range(n_actions):
            ids.append(TOK_S[s_p])
            ids.append(TOK_A[c])
            ids.append(TOK_EVAL)

        select_positions.append(len(ids))
        ids.append(TOK_SELECT)

        # Phase 2: scaffold tokens
        ids.append(TOK_ANEXT)
        ids.append(TOK_QCURR)
        ids.append(TOK_QNEXT)
        update_positions.append(len(ids))
        ids.append(TOK_UPDATE)

        # Phase 2: thought block
        if crm[t]:
            # Continuous: 1 COT token
            cot_pos = len(ids)
            ids.append(TOK_COT)
            thought_positions.append([cot_pos, -1, -1])
        else:
            # Discrete: [s_t, a_t, Q_bin]
            if q_snapshots is not None:
                q_val = float(q_snapshots[t, s, a])
                q_bin = discretize_q_value(q_val, n_q_bins, q_bin_min, q_bin_max)
            else:
                q_bin = 0
            p0 = len(ids); ids.append(TOK_S[s])
            p1 = len(ids); ids.append(TOK_A[a])
            p2 = len(ids); ids.append(TOK_QBIN[q_bin])
            thought_positions.append([p0, p1, p2])

    def to_tensor(lst, dtype):
        return torch.tensor([lst], dtype=dtype, device=device)

    return {
        'input_ids':            to_tensor(ids,              torch.long),
        'reward_values':        to_tensor(reward_values,    torch.float32),
        'reward_positions':     to_tensor(reward_positions, torch.long),
        'select_positions':     to_tensor(select_positions, torch.long),
        'update_positions':     to_tensor(update_positions, torch.long),
        'thought_positions':    torch.tensor([thought_positions], dtype=torch.long, device=device),
        'continuous_round_mask': torch.tensor([crm], dtype=torch.bool, device=device),
    }


# ---------------------------------------------------------------------------
# Part 1: Action prediction evaluation
# ---------------------------------------------------------------------------

def run_action_inference(
    model:       COCONUTTransformer,
    trajectory:  List[Dict],
    vocab:       Dict,
    n_actions:   int,
    stage_idx:   int,
    config:      'COCONUTConfig',
    device:      torch.device,
    q_snapshots: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Feed trajectory step-by-step and collect predicted actions.

    Returns predicted_actions : np.ndarray, shape (n_steps,)
    """
    TOK_S      = vocab['TOK_S']
    TOK_A      = vocab['TOK_A']
    TOK_R      = vocab['TOK_R']
    TOK_EVAL   = vocab['TOK_EVAL']
    TOK_SELECT = vocab['TOK_SELECT']
    TOK_COT    = vocab['TOK_COT']
    TOK_UPDATE = vocab['TOK_UPDATE']
    TOK_QCURR  = vocab['TOK_QCURR']
    TOK_QNEXT  = vocab['TOK_QNEXT']
    TOK_ANEXT  = vocab['TOK_ANEXT']
    TOK_QBIN   = vocab['TOK_QBIN']
    TOK_NULL   = vocab['TOK_NULL']
    TOK_START  = vocab['TOK_START']

    n_q_bins  = config.n_q_bins
    q_bin_min = config.q_bin_min
    q_bin_max = config.q_bin_max

    n_total = len(trajectory)
    crm     = _compute_continuous_round_mask(stage_idx, n_total)

    model.eval()
    predicted_actions = []

    ids               = [TOK_NULL, TOK_START]
    reward_values     = []
    reward_positions  = []
    select_positions  = []
    update_positions  = []
    thought_positions = []

    with torch.no_grad():
        for step_idx, traj in enumerate(trajectory):
            s   = traj['s']
            a   = traj['a']
            r   = traj['r']
            s_p = traj['s_next']

            # Phase 1
            ids.append(TOK_S[s])
            ids.append(TOK_A[a])
            reward_positions.append(len(ids))
            reward_values.append(r)
            ids.append(TOK_R)
            ids.append(TOK_S[s_p])

            for c in range(n_actions):
                ids.append(TOK_S[s_p])
                ids.append(TOK_A[c])
                ids.append(TOK_EVAL)

            select_positions.append(len(ids))
            ids.append(TOK_SELECT)

            # Phase 2 scaffold
            ids.append(TOK_ANEXT)
            ids.append(TOK_QCURR)
            ids.append(TOK_QNEXT)
            update_positions.append(len(ids))
            ids.append(TOK_UPDATE)

            # Phase 2 thought block
            if crm[step_idx]:
                cot_pos = len(ids)
                ids.append(TOK_COT)
                thought_positions.append([cot_pos, -1, -1])
            else:
                if q_snapshots is not None:
                    q_val = float(q_snapshots[step_idx, s, a])
                    q_bin = discretize_q_value(q_val, n_q_bins, q_bin_min, q_bin_max)
                else:
                    q_bin = 0
                p0 = len(ids); ids.append(TOK_S[s])
                p1 = len(ids); ids.append(TOK_A[a])
                p2 = len(ids); ids.append(TOK_QBIN[q_bin])
                thought_positions.append([p0, p1, p2])

            # Forward pass
            input_ids_t  = torch.tensor([ids],               dtype=torch.long,    device=device)
            rew_vals_t   = torch.tensor([reward_values],     dtype=torch.float32, device=device)
            rew_pos_t    = torch.tensor([reward_positions],  dtype=torch.long,    device=device)
            sel_pos_t    = torch.tensor([select_positions],  dtype=torch.long,    device=device)
            upd_pos_t    = torch.tensor([update_positions],  dtype=torch.long,    device=device)
            th_pos_t     = torch.tensor([thought_positions], dtype=torch.long,    device=device)
            crm_t        = torch.tensor([crm[:step_idx+1]],  dtype=torch.bool,    device=device)

            action_logits, _ = model.forward_hao(
                input_ids             = input_ids_t,
                reward_values         = rew_vals_t,
                reward_positions      = rew_pos_t,
                select_positions      = sel_pos_t,
                update_positions      = upd_pos_t,
                thought_positions     = th_pos_t,
                continuous_round_mask = crm_t,
            )

            pred_a = int(action_logits[0, -1].argmax().item())
            predicted_actions.append(pred_a)

    return np.array(predicted_actions, dtype=np.int32)


# ---------------------------------------------------------------------------
# Part 2: Q-value probing
# ---------------------------------------------------------------------------

class QProbe(nn.Module):
    """Linear probe: hidden state -> full Q-table (n_states × n_actions)."""

    def __init__(self, d_model: int, n_states: int, n_actions: int):
        super().__init__()
        self.linear   = nn.Linear(d_model, n_states * n_actions)
        self.n_states  = n_states
        self.n_actions = n_actions

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.linear(h).view(-1, self.n_states, self.n_actions)


def collect_probe_data(
    model:          COCONUTTransformer,
    n_trajectories: int,
    n_states:       int,
    n_actions:      int,
    vocab:          Dict,
    stage_idx:      int,
    config:         'COCONUTConfig',
    device:         torch.device,
    n_steps:        int = 50,
    seed_offset:    int = 20000,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Collect hidden states and Q-table targets for probe training/evaluation.

    For each trajectory, calls forward_hao ONCE on the full sequence with
    return_hidden=True, then extracts hidden states at SELECT, UPDATE, and COT
    positions.

    Returns
    -------
    h_select_all : (n_trajectories * n_steps, d_model)
    h_update_all : (n_trajectories * n_steps, d_model)
    h_cot_all    : (n_trajectories * n_steps, d_model) — zeros for discrete rounds
    q_true_all   : (n_trajectories * n_steps, n_states, n_actions)
    """
    model.eval()
    h_select_list = []
    h_update_list = []
    h_cot_list    = []
    q_true_list   = []

    with torch.no_grad():
        for i in range(n_trajectories):
            seed = seed_offset + i
            P, R = generate_eval_mdp(n_states, n_actions, seed=seed)
            trajectory, q_snapshots = run_tabular_q_learning(
                P, R, n_states, n_actions, n_steps=n_steps, seed=seed
            )

            tensors = trajectory_to_tensors(
                trajectory, vocab, n_actions, stage_idx, config, device,
                q_snapshots=q_snapshots,
            )

            result = model.forward_hao(
                input_ids             = tensors['input_ids'],
                reward_values         = tensors['reward_values'],
                reward_positions      = tensors['reward_positions'],
                select_positions      = tensors['select_positions'],
                update_positions      = tensors['update_positions'],
                thought_positions     = tensors['thought_positions'],
                continuous_round_mask = tensors['continuous_round_mask'],
                return_hidden         = True,
            )
            # result = (action_logits, thought_logits, h_final)
            h_final = result[2]   # [1, T, d_model]

            sel_pos  = tensors['select_positions'][0].cpu().numpy()   # (n_steps,)
            upd_pos  = tensors['update_positions'][0].cpu().numpy()   # (n_steps,)
            th_pos   = tensors['thought_positions'][0].cpu().numpy()  # (n_steps, 3)
            crm      = tensors['continuous_round_mask'][0].cpu().numpy()  # (n_steps,) bool
            h        = h_final[0].cpu().numpy()                       # (T, d_model)

            for t in range(n_steps):
                h_select_list.append(h[sel_pos[t]])
                h_update_list.append(h[upd_pos[t]])

                if crm[t]:
                    # Continuous round: COT position is thought_positions[t, 0]
                    cot_p = int(th_pos[t, 0])
                    h_cot_list.append(h[cot_p])
                else:
                    # Discrete round: no continuous thought; store zeros
                    h_cot_list.append(np.zeros(config.d_model, dtype=np.float32))

                q_true_list.append(q_snapshots[t])

    return (
        np.stack(h_select_list, axis=0),
        np.stack(h_update_list, axis=0),
        np.stack(h_cot_list,    axis=0),
        np.stack(q_true_list,   axis=0),
    )


def train_probe(
    probe:      QProbe,
    h_all:      np.ndarray,
    q_all:      np.ndarray,
    device:     torch.device,
    n_epochs:   int = 10,
    batch_size: int = 256,
    lr:         float = 1e-3,
) -> List[float]:
    probe.train()
    optimizer = torch.optim.Adam(probe.parameters(), lr=lr)
    N = h_all.shape[0]
    losses = []

    h_t = torch.tensor(h_all, dtype=torch.float32, device=device)
    q_t = torch.tensor(q_all, dtype=torch.float32, device=device)

    for _ in range(n_epochs):
        perm = torch.randperm(N, device=device)
        epoch_loss = 0.0
        n_batches  = 0
        for i in range(0, N, batch_size):
            idx  = perm[i:i + batch_size]
            pred = probe(h_t[idx])
            loss = F.mse_loss(pred, q_t[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches  += 1
        losses.append(epoch_loss / max(n_batches, 1))

    return losses


def evaluate_probe(
    probe:  QProbe,
    h_all:  np.ndarray,
    q_all:  np.ndarray,
    device: torch.device,
) -> Tuple[float, float, np.ndarray]:
    probe.eval()
    with torch.no_grad():
        h_t    = torch.tensor(h_all, dtype=torch.float32, device=device)
        q_pred = probe(h_t).cpu().numpy()

    q_true_flat = q_all.reshape(-1)
    q_pred_flat = q_pred.reshape(-1)

    ss_res = np.sum((q_true_flat - q_pred_flat) ** 2)
    ss_tot = np.sum((q_true_flat - q_true_flat.mean()) ** 2) + 1e-12
    r2     = float(1.0 - ss_res / ss_tot)

    diff      = q_pred - q_all
    frob      = np.sqrt((diff ** 2).sum(axis=(1, 2)))
    frob_mean = float(frob.mean())

    return r2, frob_mean, q_pred


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def plot_action_agreement(
    aa_mean_id:   np.ndarray,
    aa_std_id:    np.ndarray,
    ood_results:  List[Tuple[str, np.ndarray, np.ndarray]],
    save_path:    str,
    n_mdps:       int = 10,
    label_suffix: str = '',
) -> None:
    steps = np.arange(1, len(aa_mean_id) + 1)
    ood_colours = ['darkorange', 'crimson', 'forestgreen', 'purple', 'saddlebrown']
    ood_styles  = ['--', '-.', ':', (0,(3,1,1,1)), (0,(5,1))]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(steps, aa_mean_id, color='steelblue', linewidth=2.5,
            label='In-distribution (ID)  [Dir(1) + Beta(2,2)]')
    ax.fill_between(steps, aa_mean_id - aa_std_id, aa_mean_id + aa_std_id,
                    alpha=0.2, color='steelblue')

    for i, (lbl, mean_ood, std_ood) in enumerate(ood_results):
        colour = ood_colours[i % len(ood_colours)]
        style  = ood_styles[i % len(ood_styles)]
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
        f'(mean ± std, {n_mdps} MDPs each variant)',
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
    probe_name: str,
    r2: float,
    save_path: str,
    n_steps: int = 0,
    warmup_steps: int = 20,
) -> None:
    """Scatter plot of probed vs. true Q-values.

    When n_steps > 0, only trajectory steps t >= warmup_steps are included
    to avoid bias from zero-initialized Q-values early in each episode.
    Data is assumed to be laid out as (n_traj * n_steps, ...) with steps
    varying in the inner dimension (i.e., row i*n_steps + t belongs to
    trajectory i at timestep t).
    """
    if n_steps > 0 and warmup_steps > 0:
        n_total = q_true.shape[0]
        n_traj  = n_total // n_steps
        step_indices = np.tile(np.arange(n_steps), n_traj)
        keep = step_indices >= warmup_steps
        q_true = q_true[keep]
        q_pred = q_pred[keep]

        # Recompute R² on the filtered subset for the subtitle
        q_t_flat = q_true.reshape(-1)
        q_p_flat = q_pred.reshape(-1)
        ss_res = np.sum((q_t_flat - q_p_flat) ** 2)
        ss_tot = np.sum((q_t_flat - q_t_flat.mean()) ** 2) + 1e-12
        r2 = float(1.0 - ss_res / ss_tot)

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
    ax.set_ylabel('Q probe')
    warmup_note = f', t ≥ {warmup_steps}' if n_steps > 0 and warmup_steps > 0 else ''
    ax.set_title(f'Probe ({probe_name}) Q-Values vs True Q-Values{warmup_note}\nR² = {r2:.4f}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_probe_frobenius(
    q_pred_all:  np.ndarray,
    q_true_all:  np.ndarray,
    probe_name:  str,
    n_steps:     int,
    n_traj_eval: int,
    save_path:   str,
) -> None:
    q_pred_r = q_pred_all.reshape(n_traj_eval, n_steps, -1)
    q_true_r = q_true_all.reshape(n_traj_eval, n_steps, -1)

    diff       = q_pred_r - q_true_r
    frob_probe = np.sqrt((diff ** 2).sum(axis=-1))
    frob_mean  = frob_probe.mean(axis=0)
    frob_std   = frob_probe.std(axis=0)

    zero_baseline = np.sqrt((q_true_r ** 2).sum(axis=-1)).mean(axis=0)

    steps = np.arange(1, n_steps + 1)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(steps, frob_mean, color='steelblue', linewidth=2, label=f'Probe ({probe_name}) error')
    ax.fill_between(steps, frob_mean - frob_std, frob_mean + frob_std, alpha=0.25, color='steelblue')
    ax.plot(steps, zero_baseline, color='tomato', linewidth=2, linestyle='--',
            label='Zero baseline (always predict Q=0)')
    ax.set_xlabel('Timestep')
    ax.set_ylabel(r'$\|Q_{probe}(t) - Q_{tabular}(t)\|_F$')
    ax.set_title(f'Q-Value Probe Frobenius Error Over Time\n'
                 f'(probe: {probe_name}, mean ± std, {n_traj_eval} MDPs)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


# ---------------------------------------------------------------------------
# Attention heatmap — two-row layout (SELECT and UPDATE)
# ---------------------------------------------------------------------------

def get_token_labels(ids: List[int], vocab: Dict) -> List[str]:
    inv = {}
    inv[vocab['TOK_NULL']]   = 'NULL'
    inv[vocab['TOK_START']]  = 'START'
    inv[vocab['TOK_R']]      = 'R'
    inv[vocab['TOK_EVAL']]   = 'EVAL'
    inv[vocab['TOK_SELECT']] = 'SEL'
    inv[vocab['TOK_THINK']]  = 'THK(legacy)'
    inv[vocab['TOK_COT']]    = 'COT'
    inv[vocab['TOK_UPDATE']] = 'UPD'
    inv[vocab['TOK_QCURR']]  = 'QCUR'
    inv[vocab['TOK_QNEXT']]  = 'QNXT'
    inv[vocab['TOK_ANEXT']]  = 'ANXT'
    for i, tok in enumerate(vocab['TOK_S']):
        inv[tok] = f'S{i}'
    for i, tok in enumerate(vocab['TOK_A']):
        inv[tok] = f'A{i}'
    for i, tok in enumerate(vocab.get('TOK_QBIN', [])):
        inv[tok] = f'QB{i}'
    return [inv.get(t, f'?{t}') for t in ids]


def plot_attention_heatmap(
    model:       'COCONUTTransformer',
    trajectory:  List[Dict],
    vocab:       Dict,
    n_actions:   int,
    stage_idx:   int,
    config:      'COCONUTConfig',
    device:      torch.device,
    save_path:   str,
    q_snapshots: Optional[np.ndarray] = None,
) -> None:
    """Full T×T attention heatmap: one subplot per (layer, head).

    Layout: n_layers rows × n_heads columns, each cell showing the complete
    causal attention matrix for that layer/head (Query × Key).
    Special token positions are annotated on axes.
    Forces all-discrete mode (stage 0) for attention extraction.
    """
    model.eval()

    n_show      = min(3, len(trajectory))
    short_traj  = trajectory[:n_show]
    short_snaps = q_snapshots[:n_show] if q_snapshots is not None else None

    # Force all rounds discrete so the fast path returns attention weights
    vis_stage = 0
    tensors = trajectory_to_tensors(
        short_traj, vocab, n_actions, vis_stage, config, device,
        n_steps=n_show, q_snapshots=short_snaps,
    )

    with torch.no_grad():
        out = model.forward_hao(
            input_ids             = tensors['input_ids'],
            reward_values         = tensors['reward_values'],
            reward_positions      = tensors['reward_positions'],
            select_positions      = tensors['select_positions'],
            update_positions      = tensors['update_positions'],
            thought_positions     = tensors['thought_positions'],
            continuous_round_mask = tensors['continuous_round_mask'],
            return_attention      = True,
        )

    # out = (action_logits, thought_logits, all_attn) for all-discrete
    if not isinstance(out, tuple) or len(out) < 3:
        print("  Attention heatmap: no attention weights returned. Skipping.")
        return
    all_attn = out[-1]   # [L, B, H, T, T]
    if all_attn is None:
        print("  Attention heatmap: attention weights unavailable. Skipping.")
        return

    ids    = tensors['input_ids'][0].cpu().tolist()
    labels = get_token_labels(ids, vocab)
    T      = len(ids)

    # all_attn: [L, B, H, T, T] — take batch 0, convert to numpy
    attn_np = all_attn[:, 0, :, :, :].cpu().numpy()   # [L, H, T, T]
    L, H    = attn_np.shape[:2]

    sel_positions = tensors['select_positions'][0].cpu().tolist()
    upd_positions = tensors['update_positions'][0].cpu().tolist()

    cell_px  = max(4.0, T * 0.07)   # inches per subplot cell
    fig, axes = plt.subplots(
        L, H,
        figsize=(H * cell_px, L * cell_px),
        squeeze=False,
    )

    cmap = plt.cm.hot_r   # dark=low, bright=high — matches MWU style

    for layer_idx in range(L):
        for head_idx in range(H):
            ax   = axes[layer_idx][head_idx]
            data = attn_np[layer_idx, head_idx]   # [T, T]

            im = ax.imshow(data, aspect='equal', cmap=cmap,
                           vmin=0.0, vmax=data.max() + 1e-9,
                           origin='upper', interpolation='nearest')

            # Mark SELECT positions (red horizontal/vertical lines)
            for sp in sel_positions:
                if sp >= 0:
                    ax.axhline(sp, color='red',  linewidth=0.6, alpha=0.5)
                    ax.axvline(sp, color='red',  linewidth=0.6, alpha=0.5)
            # Mark UPDATE positions (blue)
            for up in upd_positions:
                if up >= 0:
                    ax.axhline(up, color='blue', linewidth=0.6, alpha=0.5)
                    ax.axvline(up, color='blue', linewidth=0.6, alpha=0.5)

            # Tick labels on outermost axes — every token
            tick_pos  = list(range(T))

            if layer_idx == L - 1:
                ax.set_xticks(tick_pos)
                ax.set_xticklabels(labels, rotation=90, fontsize=5)
            else:
                ax.set_xticks([])

            if head_idx == 0:
                ax.set_yticks(tick_pos)
                ax.set_yticklabels(labels, fontsize=5)
            else:
                ax.set_yticks([])

            # Column header (head index) on top row only
            if layer_idx == 0:
                ax.set_title(f'Head {head_idx}', fontsize=8, pad=3)

            # Row label (layer index) on leftmost column only
            if head_idx == 0:
                ax.set_ylabel(f'Layer {layer_idx + 1}', fontsize=8)

    fig.suptitle(
        f'Attention Heatmap — Layer {L}, Step {n_show}/{len(trajectory)} tokens\n'
        f'Red lines = SELECT positions  |  Blue lines = UPDATE positions',
        fontsize=10, y=1.01,
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


# ---------------------------------------------------------------------------
# Effective alpha / gamma recovery (unchanged from previous version)
# ---------------------------------------------------------------------------

def estimate_effective_alpha_gamma(
    probe:    'QProbe',
    h_eval:   np.ndarray,
    q_true:   np.ndarray,
    device:   torch.device,
    n_steps:  int,
    n_traj:   int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    probe.eval()
    with torch.no_grad():
        h_t    = torch.tensor(h_eval, dtype=torch.float32, device=device)
        q_pred = probe(h_t).cpu().numpy()

    Q      = q_pred.reshape(n_traj, n_steps, -1, q_pred.shape[-1])
    Q_true = q_true.reshape(n_traj, n_steps, -1, q_true.shape[-1])  # noqa: F841

    alpha_list = []
    gamma_list = []
    r2_list    = []

    for i in range(n_traj):
        T    = n_steps - 1
        dQ   = Q[i, 1:] - Q[i, :-1]
        Qcur = Q[i, :-1]
        maxQ_next = Q[i, 1:].max(axis=-1, keepdims=True)

        y  = dQ.reshape(T, -1).flatten()
        X1 = (-Qcur).reshape(T, -1).flatten()
        X2 = np.broadcast_to(maxQ_next, Q[i, 1:].shape).reshape(T, -1).flatten()

        A      = np.stack([X1, X2], axis=1)
        betas  = np.linalg.lstsq(A, y, rcond=None)[0]
        alpha_e = float(betas[0])
        gamma_e = float(betas[1] / alpha_e) if abs(alpha_e) > 1e-4 else float('nan')

        y_pred = A @ betas
        ss_res = float(np.sum((y - y_pred) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2)) + 1e-12

        alpha_list.append(alpha_e)
        gamma_list.append(gamma_e)
        r2_list.append(1.0 - ss_res / ss_tot)

    return np.array(alpha_list), np.array(gamma_list), np.array(r2_list)


def plot_effective_alpha_gamma(
    alpha_eff:    np.ndarray,
    gamma_eff:    np.ndarray,
    r2_vals:      np.ndarray,
    expert_alpha: float,
    expert_gamma: float,
    save_path:    str,
) -> None:
    valid = np.isfinite(alpha_eff) & np.isfinite(gamma_eff)
    a = np.clip(alpha_eff[valid], -0.5, 1.5)
    g = np.clip(gamma_eff[valid], -0.2, 1.5)
    r = r2_vals[valid]

    fig = plt.figure(figsize=(13, 5))
    gs  = fig.add_gridspec(1, 3, width_ratios=[2, 1, 1], wspace=0.35)
    ax_scatter = fig.add_subplot(gs[0])
    ax_alpha   = fig.add_subplot(gs[1])
    ax_gamma   = fig.add_subplot(gs[2])

    sc = ax_scatter.scatter(a, g, c=r, cmap='plasma', alpha=0.7, s=30, vmin=0, vmax=1)
    plt.colorbar(sc, ax=ax_scatter, label='Regression R²')
    ax_scatter.axvline(expert_alpha, color='red',  linewidth=2, linestyle='--',
                       label=f'Expert α = {expert_alpha}')
    ax_scatter.axhline(expert_gamma, color='blue', linewidth=2, linestyle='--',
                       label=f'Expert γ = {expert_gamma}')
    ax_scatter.set_xlabel('α_eff'); ax_scatter.set_ylabel('γ_eff')
    ax_scatter.set_title('Fitted (α_eff, γ_eff) per trajectory')
    ax_scatter.legend(fontsize=8); ax_scatter.grid(True, alpha=0.3)

    ax_alpha.hist(a, bins=20, color='steelblue', edgecolor='white', alpha=0.8)
    ax_alpha.axvline(expert_alpha, color='red', linewidth=2, linestyle='--')
    ax_alpha.axvline(float(np.nanmedian(a)), color='navy', linewidth=1.5, linestyle=':',
                     label=f'Median={np.nanmedian(a):.3f}')
    ax_alpha.set_xlabel('α_eff'); ax_alpha.set_title('α_eff distribution')
    ax_alpha.legend(fontsize=7); ax_alpha.grid(True, alpha=0.3)

    ax_gamma.hist(g, bins=20, color='darkorange', edgecolor='white', alpha=0.8)
    ax_gamma.axvline(expert_gamma, color='blue', linewidth=2, linestyle='--')
    ax_gamma.axvline(float(np.nanmedian(g)), color='darkred', linewidth=1.5, linestyle=':',
                     label=f'Median={np.nanmedian(g):.3f}')
    ax_gamma.set_xlabel('γ_eff'); ax_gamma.set_title('γ_eff distribution')
    ax_gamma.legend(fontsize=7); ax_gamma.grid(True, alpha=0.3)

    fig.suptitle('Effective Q-Learning Hyperparameters Recovered from Model Probe', fontsize=12)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


# ---------------------------------------------------------------------------
# Q-convergence tracking (unchanged logic)
# ---------------------------------------------------------------------------

def compute_q_star(
    P:       np.ndarray,
    R:       np.ndarray,
    gamma:   float = 0.9,
    n_iter:  int   = 500,
    alpha:   float = 0.1,
    epsilon: float = 0.2,
    seed:    int   = 0,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n_states, n_actions = R.shape
    Q = np.zeros((n_states, n_actions), dtype=np.float32)
    s = int(rng.integers(n_states))
    for _ in range(n_iter):
        a = int(rng.integers(n_actions)) if rng.random() < epsilon else int(np.argmax(Q[s]))
        r = float(R[s, a])
        s_next = int(rng.choice(n_states, p=P[s, a]))
        Q[s, a] += alpha * (r + gamma * float(np.max(Q[s_next])) - Q[s, a])
        s = s_next
    return Q


def plot_q_convergence(
    q_pred_all:  np.ndarray,
    q_true_all:  np.ndarray,
    q_star_list: List[np.ndarray],
    n_steps:     int,
    n_traj:      int,
    save_path:   str,
) -> None:
    Q_pred   = q_pred_all.reshape(n_traj, n_steps, -1)
    Q_true   = q_true_all.reshape(n_traj, n_steps, -1)
    frob_m   = np.zeros((n_traj, n_steps))
    frob_t   = np.zeros((n_traj, n_steps))

    for i in range(n_traj):
        qstar = q_star_list[i].flatten()
        for t in range(n_steps):
            frob_m[i, t] = np.linalg.norm(Q_pred[i, t] - qstar)
            frob_t[i, t] = np.linalg.norm(Q_true[i, t] - qstar)

    steps = np.arange(1, n_steps + 1)
    fig, ax = plt.subplots(figsize=(10, 5))
    mm, sm = frob_m.mean(0), frob_m.std(0)
    mt, st = frob_t.mean(0), frob_t.std(0)
    ax.plot(steps, mm, color='steelblue', linewidth=2, label='Model probe ||Q̂−Q*||_F')
    ax.fill_between(steps, mm - sm, mm + sm, alpha=0.25, color='steelblue')
    ax.plot(steps, mt, color='darkorange', linewidth=2, linestyle='--',
            label='Tabular Q-learning ||Q_tab−Q*||_F')
    ax.fill_between(steps, mt - st, mt + st, alpha=0.2, color='darkorange')
    ax.set_xlabel('Timestep'); ax.set_ylabel(r'$\|Q(t)-Q^*\|_F$')
    ax.set_title(f'Q-Value Convergence Toward Q*  (mean ± std, {n_traj} MDPs)')
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


# ---------------------------------------------------------------------------
# Per-state agreement heatmap (unchanged logic)
# ---------------------------------------------------------------------------

def plot_per_state_agreement(
    trajectory_data: List[Tuple[List[Dict], np.ndarray]],
    n_states:  int,
    n_steps:   int,
    save_path: str,
    n_bins:    int = 3,
) -> None:
    bin_edges   = np.linspace(0, n_steps, n_bins + 1, dtype=int)
    bin_labels  = [f't={bin_edges[i]+1}–{bin_edges[i+1]}' for i in range(n_bins)]
    agree_sum   = np.zeros((n_states, n_bins), dtype=np.float32)
    agree_count = np.zeros((n_states, n_bins), dtype=np.int32)

    for traj, preds in trajectory_data:
        targets = np.array([step['a_next'] for step in traj], dtype=np.int32)
        for t, step in enumerate(traj):
            s_next = step['s_next']
            b = min(int(np.searchsorted(bin_edges[1:], t, side='right')), n_bins - 1)
            agree_sum[s_next, b]   += float(preds[t] == targets[t])
            agree_count[s_next, b] += 1

    with np.errstate(invalid='ignore'):
        agree_rate = np.where(agree_count > 0,
                              agree_sum / agree_count.astype(np.float32), np.nan)

    fig, ax = plt.subplots(figsize=(max(6, n_bins * 2), max(4, n_states * 0.8)))
    im = ax.imshow(agree_rate, aspect='auto', cmap='RdYlGn', vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, label='Greedy action agreement')
    ax.set_xticks(range(n_bins)); ax.set_xticklabels(bin_labels, rotation=15, ha='right')
    ax.set_yticks(range(n_states))
    ax.set_yticklabels([f'Next-state {s}' for s in range(n_states)])
    ax.set_xlabel('Episode timestep bin'); ax.set_ylabel("Next-state (s')")
    ax.set_title('Per-State Action Agreement by Timestep Bin')
    for s in range(n_states):
        for b in range(n_bins):
            val = agree_rate[s, b]
            if not np.isnan(val):
                ax.text(b, s, f'{val:.2f}', ha='center', va='center',
                        fontsize=8, color='black' if 0.3 < val < 0.85 else 'white')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_training_curves(log_path: str, save_path: str) -> None:
    if not os.path.exists(log_path):
        print(f"  Warning: training log not found at {log_path}, skipping.")
        return
    data = np.load(log_path)
    steps = data['steps']; train_ce = data['train_ce']
    train_acc = data['train_acc']; val_ce = data['val_ce']; val_acc = data['val_acc']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(steps, train_ce, color='steelblue', linewidth=1.5, label='Train CE')
    ax1.plot(steps, val_ce,   color='darkorange', linewidth=1.5, label='Val CE')
    ax1.set_xlabel('Step'); ax1.set_ylabel('CE Loss'); ax1.set_title('CE Loss Over Training')
    ax1.legend(); ax1.grid(True, alpha=0.3)

    ax2.plot(steps, train_acc * 100, color='steelblue', linewidth=1.5, label='Train Acc')
    ax2.plot(steps, val_acc   * 100, color='darkorange', linewidth=1.5, label='Val Acc')
    ax2.set_xlabel('Step'); ax2.set_ylabel('Accuracy (%)'); ax2.set_title('Accuracy Over Training')
    ax2.legend(); ax2.grid(True, alpha=0.3)

    plt.suptitle('COCONUT Q-Learning — Training Curves', fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Evaluate trained two-phase COCONUT transformer')
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
    parser.add_argument('--no_coconut',    action='store_true',
                        help='Force stage 0 (all discrete) regardless of checkpoint stage')
    args = parser.parse_args()

    os.makedirs(args.figures_dir, exist_ok=True)

    # ---- Load checkpoint ----
    print(f"Loading checkpoint from {args.checkpoint} ...")
    ckpt   = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    config = COCONUTConfig.from_dict(ckpt['config'])
    model  = COCONUTTransformer(config)

    # Checkpoint key migration: old checkpoints use 'explain_head.*', new use 'thought_head.*'
    state = ckpt['model_state_dict']
    key_map = {}
    for k in list(state.keys()):
        if k.startswith('explain_head.') and 'thought_head.' + k[len('explain_head.'):] not in state:
            key_map[k] = 'thought_head.' + k[len('explain_head.'):]
    for old_k, new_k in key_map.items():
        state[new_k] = state.pop(old_k)
    model.load_state_dict(state)

    # Determine stage_idx
    stage_idx = ckpt.get('stage_idx', None)
    if stage_idx is None:
        old_nc = ckpt.get('n_continuous', 3)
        stage_idx = {0: 0, 1: 1, 2: 2, 3: 4}.get(old_nc, 4)

    use_coconut = ckpt.get('use_coconut', True)
    if args.no_coconut:
        stage_idx = 0

    frac_cont = HAO_STAGES[stage_idx][2]
    print(f"  Loaded epoch={ckpt.get('epoch','?')}, step={ckpt.get('step','?')}, "
          f"val_ce={ckpt.get('val_ce_loss', ckpt.get('val_loss','?')):.4f}")
    print(f"  stage_idx={stage_idx} (frac_continuous={frac_cont:.0%}), "
          f"use_coconut={use_coconut}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model  = model.to(device)
    model.eval()

    vocab     = build_vocab(config.n_states, config.n_actions, config.n_q_bins)
    n_states  = config.n_states
    n_actions = config.n_actions

    print(f"\nModel: {model.num_parameters():,} params  |  device: {device}")
    print(f"MDP:   n_states={n_states}, n_actions={n_actions}")

    eval_seeds = list(range(args.eval_seed, args.eval_seed + args.n_eval_mdps))
    label = f"COCONUT stage={stage_idx} ({frac_cont:.0%} cont)" if use_coconut else "No-COCONUT"

    # -----------------------------------------------------------------------
    # Part 1a: In-distribution action prediction
    # -----------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"Part 1a: In-distribution action prediction ({args.n_eval_mdps} MDPs)")
    print(f"{'='*60}")

    all_agreements_id = []
    for seed_i, seed in enumerate(eval_seeds):
        print(f"  ID MDP {seed_i+1}/{args.n_eval_mdps} (seed={seed}) ...", end=' ', flush=True)
        P, R = generate_eval_mdp(n_states, n_actions, seed=seed)
        trajectory, q_snaps = run_tabular_q_learning(
            P, R, n_states, n_actions, n_steps=args.n_steps,
            alpha=args.alpha, gamma=args.gamma, epsilon=args.epsilon, seed=seed,
        )
        preds   = run_action_inference(model, trajectory, vocab, n_actions,
                                       stage_idx, config, device, q_snapshots=q_snaps)
        targets = np.array([step['a_next'] for step in trajectory], dtype=np.int32)
        agree   = (preds == targets).astype(np.float32)
        all_agreements_id.append(agree)
        print(f"agree={agree.mean():.2%}")

    aa_id_arr  = np.stack(all_agreements_id, axis=0)
    aa_id_mean = aa_id_arr.mean(axis=0)
    aa_id_std  = aa_id_arr.std(axis=0)
    print(f"\n  Mean ID agreement ({label}): "
          f"{aa_id_mean.mean():.2%} ± {aa_id_arr.mean(axis=1).std():.4f}")

    # -----------------------------------------------------------------------
    # Part 1b: OOD variants
    # -----------------------------------------------------------------------
    ood_plot_results: List[Tuple[str, np.ndarray, np.ndarray]] = []

    for v_idx, (variant, variant_desc) in enumerate(OOD_VARIANTS):
        v_seeds = list(range(
            args.eval_seed + 50000 + v_idx * 10000,
            args.eval_seed + 50000 + v_idx * 10000 + args.n_eval_mdps,
        ))
        print(f"\n{'='*60}")
        print(f"Part 1b-{v_idx+1}: OOD '{variant}'  —  {variant_desc}")
        print(f"{'='*60}")

        all_agreements_v = []
        for seed_i, seed in enumerate(v_seeds):
            print(f"  {variant} {seed_i+1}/{args.n_eval_mdps} ...", end=' ', flush=True)
            P, R = generate_ood_mdp(n_states, n_actions, variant=variant, seed=seed)
            trajectory, q_snaps_v = run_tabular_q_learning(
                P, R, n_states, n_actions, n_steps=args.n_steps,
                alpha=args.alpha, gamma=args.gamma, epsilon=args.epsilon, seed=seed,
            )
            preds   = run_action_inference(model, trajectory, vocab, n_actions,
                                           stage_idx, config, device, q_snapshots=q_snaps_v)
            targets = np.array([step['a_next'] for step in trajectory], dtype=np.int32)
            agree   = (preds == targets).astype(np.float32)
            all_agreements_v.append(agree)
            print(f"agree={agree.mean():.2%}")

        aa_v_arr  = np.stack(all_agreements_v, axis=0)
        aa_v_mean = aa_v_arr.mean(axis=0)
        aa_v_std  = aa_v_arr.std(axis=0)
        gap = aa_id_mean.mean() - aa_v_mean.mean()
        print(f"\n  Mean {variant} ({label}): "
              f"{aa_v_mean.mean():.2%} ± {aa_v_arr.mean(axis=1).std():.4f}  (ID gap: {gap:+.4f})")
        ood_plot_results.append((variant_desc, aa_v_mean, aa_v_std))

    plot_action_agreement(
        aa_id_mean, aa_id_std, ood_plot_results,
        save_path=os.path.join(args.figures_dir, 'action_agreement.png'),
        n_mdps=args.n_eval_mdps, label_suffix=label,
    )

    # -----------------------------------------------------------------------
    # Part 2: Q-value probing
    # -----------------------------------------------------------------------
    probe_results = {}
    has_continuous = use_coconut and stage_idx > 0

    print(f"\n{'='*60}")
    print("Part 2: Q-value probing")
    print(f"{'='*60}")

    for p in model.parameters():
        p.requires_grad_(False)

    print(f"\nCollecting probe training data ({args.n_probe_train} trajectories) ...")
    h_sel_tr, h_upd_tr, h_cot_tr, q_tr = collect_probe_data(
        model, args.n_probe_train, n_states, n_actions, vocab,
        stage_idx, config, device, n_steps=args.n_steps, seed_offset=20000,
    )
    print(f"  h_select: {h_sel_tr.shape}, h_update: {h_upd_tr.shape}, "
          f"h_cot: {h_cot_tr.shape}, q_true: {q_tr.shape}")

    print(f"\nCollecting probe eval data ({args.n_probe_eval} trajectories) ...")
    h_sel_ev, h_upd_ev, h_cot_ev, q_ev = collect_probe_data(
        model, args.n_probe_eval, n_states, n_actions, vocab,
        stage_idx, config, device, n_steps=args.n_steps, seed_offset=30000,
    )

    probe_specs = [
        ('select', h_sel_tr, h_sel_ev),
        ('update', h_upd_tr, h_upd_ev),
    ]
    if has_continuous:
        probe_specs.append(('cot', h_cot_tr, h_cot_ev))

    for probe_name, h_train, h_eval in probe_specs:
        print(f"\nTraining probe_{probe_name} ...")
        probe = QProbe(config.d_model, n_states, n_actions).to(device)
        losses = train_probe(probe, h_train, q_tr, device, n_epochs=args.probe_epochs)
        print(f"  MSE per epoch: {['%.4f' % l for l in losses]}")
        r2, frob_mean, q_pred = evaluate_probe(probe, h_eval, q_ev, device)
        print(f"  Eval probe_{probe_name}: R²={r2:.4f}  Frobenius={frob_mean:.4f}")
        probe_results[probe_name] = {
            'r2': r2, 'frob': frob_mean, 'q_pred': q_pred,
            'probe_obj': probe, 'h_eval': h_eval,
        }

    if probe_results:
        # Primary emergence probes: update and cot (Q never supervised there)
        emergence_probes = [k for k in ('update', 'cot') if k in probe_results]
        primary = max(emergence_probes or list(probe_results.keys()),
                      key=lambda k: probe_results[k]['r2'])
        print(f"\n  Primary probe: probe_{primary}  "
              f"(R²={probe_results[primary]['r2']:.4f})")

        plot_probe_scatter(
            q_ev, probe_results[primary]['q_pred'],
            probe_name=primary, r2=probe_results[primary]['r2'],
            save_path=os.path.join(args.figures_dir, 'probe_scatter.png'),
            n_steps=args.n_steps, warmup_steps=20,
        )
        plot_probe_frobenius(
            probe_results[primary]['q_pred'], q_ev,
            probe_name=primary, n_steps=args.n_steps, n_traj_eval=args.n_probe_eval,
            save_path=os.path.join(args.figures_dir, 'probe_frobenius.png'),
        )
    else:
        print("  (No probes trained)")

    # -----------------------------------------------------------------------
    # Attention heatmap
    # -----------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("Attention heatmap (two-row: SELECT + UPDATE)")
    print(f"{'='*60}")
    _attn_seed  = eval_seeds[0]
    P_attn, R_attn = generate_eval_mdp(n_states, n_actions, seed=_attn_seed)
    traj_attn, q_snaps_attn = run_tabular_q_learning(
        P_attn, R_attn, n_states, n_actions,
        n_steps=args.n_steps, alpha=args.alpha, gamma=args.gamma,
        epsilon=args.epsilon, seed=_attn_seed,
    )
    plot_attention_heatmap(
        model, traj_attn, vocab, n_actions, stage_idx, config, device,
        save_path=os.path.join(args.figures_dir, 'attention_heatmap.png'),
        q_snapshots=q_snaps_attn,
    )

    # -----------------------------------------------------------------------
    # Per-state agreement heatmap
    # -----------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("Per-state agreement heatmap")
    print(f"{'='*60}")
    traj_preds_id = []
    for seed in eval_seeds:
        P_s, R_s = generate_eval_mdp(n_states, n_actions, seed=seed)
        traj_s, q_snaps_s = run_tabular_q_learning(
            P_s, R_s, n_states, n_actions,
            n_steps=args.n_steps, alpha=args.alpha, gamma=args.gamma,
            epsilon=args.epsilon, seed=seed,
        )
        preds_s = run_action_inference(model, traj_s, vocab, n_actions,
                                       stage_idx, config, device, q_snapshots=q_snaps_s)
        traj_preds_id.append((traj_s, preds_s))

    plot_per_state_agreement(
        traj_preds_id, n_states, args.n_steps,
        os.path.join(args.figures_dir, 'per_state_agreement.png'),
    )

    # -----------------------------------------------------------------------
    # Effective alpha/gamma + Q-convergence (require probe)
    # -----------------------------------------------------------------------
    if probe_results and has_continuous:
        prim_probe = probe_results[primary]['probe_obj']
        prim_h_ev  = probe_results[primary]['h_eval']

        print(f"\n{'='*60}")
        print("Effective alpha/gamma recovery")
        print(f"{'='*60}")
        alpha_eff, gamma_eff, r2_vals = estimate_effective_alpha_gamma(
            prim_probe, prim_h_ev, q_ev, device,
            n_steps=args.n_steps, n_traj=args.n_probe_eval,
        )
        print(f"  α_eff: median={np.nanmedian(alpha_eff):.4f}  (expert={args.alpha})")
        print(f"  γ_eff: median={np.nanmedian(gamma_eff):.4f}  (expert={args.gamma})")
        plot_effective_alpha_gamma(
            alpha_eff, gamma_eff, r2_vals,
            expert_alpha=args.alpha, expert_gamma=args.gamma,
            save_path=os.path.join(args.figures_dir, 'effective_alpha_gamma.png'),
        )

        print(f"\n{'='*60}")
        print("Q-convergence toward Q*")
        print(f"{'='*60}")
        q_star_list = []
        for i in range(args.n_probe_eval):
            seed_c = 30000 + i
            P_c, R_c = generate_eval_mdp(n_states, n_actions, seed=seed_c)
            q_star_list.append(compute_q_star(
                P_c, R_c, gamma=args.gamma, n_iter=500,
                alpha=args.alpha, epsilon=args.epsilon, seed=seed_c,
            ))
        plot_q_convergence(
            probe_results[primary]['q_pred'], q_ev, q_star_list,
            n_steps=args.n_steps, n_traj=args.n_probe_eval,
            save_path=os.path.join(args.figures_dir, 'q_convergence.png'),
        )
    else:
        print("\n  (Skipping alpha/gamma + Q-convergence — need continuous thought probe)")

    # -----------------------------------------------------------------------
    # Training curves
    # -----------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("Training curves")
    print(f"{'='*60}")
    log_path = os.path.join(os.path.dirname(args.checkpoint), 'training_log.npz')
    plot_training_curves(log_path, os.path.join(args.figures_dir, 'training_curves.png'))

    print(f"\nDone. Figures saved to {args.figures_dir}/")


if __name__ == '__main__':
    main()
