#!/usr/bin/env python3
"""
4_evaluate.py — Evaluation Script (Hao-Style Curriculum)

Tests the trained COCONUTTransformer on two complementary metrics:

Part 1 — Action prediction evaluation (in-distribution + OOD)
--------------------------------------------------------------
Runs action agreement evaluation on two MDP regimes:

  In-distribution (ID): same Dirichlet(1) transitions + Beta(2,2) rewards
    as training — no trap states.

  Out-of-distribution (OOD): altered reward and transition distributions
    not seen during training:
      • Transitions ~ Dirichlet(alpha=0.1) — sparse/peaked (nearly deterministic)
      • Rewards     ~ Beta(0.5, 0.5)       — bimodal (near 0 or 1, rarely 0.5)

For each regime, run a 50-step Q-learning trajectory and feed it to the
model step-by-step using forward_hao (teacher-forced: model sees the
reference agent's exact trajectory). At each SELECT position, record
model's argmax action vs Q-learning's greedy action. Report per-step
action agreement mean ± std over n_eval_mdps MDPs.

Part 2 — Q-value probing (emergence evidence)
-----------------------------------------------
Freeze the model and train two linear probes that map the COCONUT hidden
state at THINK and last-explain positions to the full Q-table:
    probe_think   : hidden_at_THINK        -> Q(n_states, n_actions)
    probe_explain : hidden_at_last_explain -> Q(n_states, n_actions)

High probe R² means the model has learned to represent Q-values internally
without ever being told to. Report both probes; use the better-R² probe as
primary result.

Training procedure: 1000 trajectories (one full forward_hao per trajectory,
NOT per step), extract hidden states at all rounds, train probe with MSE.

The Frobenius plot includes a zero baseline (||Q_tabular(t)||_F) to guard
against trivially low early errors when Q-values are near zero.

Output files
------------
    figures/action_agreement.png   — per-step agreement mean ± std (ID vs OOD)
    figures/probe_scatter.png      — probed vs true Q-values scatter
    figures/probe_frobenius.png    — probe Frobenius error + zero baseline
    figures/training_curves.png    — train/val CE + accuracy over training
"""

import argparse
import importlib.util
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


# ---------------------------------------------------------------------------
# MDP + tabular Q-learning (mirrors 1_generate_data.py logic)
# ---------------------------------------------------------------------------

def generate_eval_mdp(
    n_states: int,
    n_actions: int,
    seed: int = 9999,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a fresh MDP with a fixed seed, matching the training distribution exactly.

    Uses the same Dirichlet(1) transitions and Beta(2,2) rewards as generate_random_mdp
    in 1_generate_data.py — no trap states.
    """
    rng = np.random.default_rng(seed)
    P = rng.dirichlet(alpha=np.ones(n_states), size=(n_states, n_actions)).astype(np.float32)
    R = rng.beta(2.0, 2.0, size=(n_states, n_actions)).astype(np.float32)
    R = np.clip(R, 0.0, 1.0)
    return P, R


def generate_ood_mdp(
    n_states: int,
    n_actions: int,
    seed: int = 9999,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate an out-of-distribution MDP not seen during training.

    Uses distributions that differ structurally from training:
      - Transitions ~ Dirichlet(alpha=0.1): sparse/peaked, nearly deterministic
        (very different from Dirichlet(1) uniform-over-simplex used in training).
      - Rewards ~ Beta(0.5, 0.5): bimodal U-shape, rewards concentrate near 0 or 1
        (very different from Beta(2,2) peaked near 0.5 used in training).
    """
    rng = np.random.default_rng(seed)
    P = rng.dirichlet(alpha=np.full(n_states, 0.1), size=(n_states, n_actions)).astype(np.float32)
    R = rng.beta(0.5, 0.5, size=(n_states, n_actions)).astype(np.float32)
    R = np.clip(R, 0.0, 1.0)
    return P, R


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
    """Run tabular Q-learning; return trajectory data and Q-table snapshots.

    Returns
    -------
    trajectory  : list of step dicts (s, a, r, s_next, a_next)
                  a_next is the greedy action at s_next (NOT ε-greedy)
    q_snapshots : np.ndarray, shape (n_steps, n_states, n_actions)
    """
    rng = np.random.default_rng(seed + 1)
    Q = np.zeros((n_states, n_actions), dtype=np.float32)
    s = int(rng.integers(n_states))

    trajectory  = []
    q_snapshots = []

    for _ in range(n_steps):
        # Epsilon-greedy behavior
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

        # Greedy action at s_next (SELECT target — NOT ε-greedy)
        best_next = float(np.max(Q[s_next]))
        ties_next = [ac for ac in range(n_actions) if Q[s_next, ac] == best_next]
        a_next = int(rng.choice(ties_next))

        trajectory.append({'s': s, 'a': a, 'r': r, 's_next': s_next, 'a_next': a_next})
        q_snapshots.append(Q.copy())
        s = s_next

    return trajectory, np.stack(q_snapshots, axis=0)


# ---------------------------------------------------------------------------
# Sequence builder for eval — expanded 3-token explain format
# ---------------------------------------------------------------------------

def trajectory_to_tensors(
    trajectory: List[Dict],
    vocab: Dict,
    n_actions: int,
    n_continuous: int,
    config: 'COCONUTConfig',
    device: torch.device,
    n_steps: Optional[int] = None,
    q_snapshots: Optional[np.ndarray] = None,
) -> Dict[str, torch.Tensor]:
    """Build expanded COCONUT token sequence from a trajectory.

    Each round's single COT token is expanded into 3 explanation tokens:
        [<s_t>, <a_t>, <Q_bin>]
    where the last n_continuous are replaced with TOK_COT placeholders
    (for continuous thought injection by forward_hao).

    Parameters
    ----------
    q_snapshots : (n_steps, n_states, n_actions) — needed to compute Q-value
                  at (s_next, greedy_action) for the Q_bin token. If None,
                  Q-bin targets default to bin 0 (doesn't affect inference,
                  only matters for discrete explain tokens).

    Returns dict with batch dim=1 tensors for direct model input.
    """
    TOK_S      = vocab['TOK_S']
    TOK_A      = vocab['TOK_A']
    TOK_R      = vocab['TOK_R']
    TOK_EVAL   = vocab['TOK_EVAL']
    TOK_SELECT = vocab['TOK_SELECT']
    TOK_THINK  = vocab['TOK_THINK']
    TOK_COT    = vocab['TOK_COT']
    TOK_QBIN   = vocab['TOK_QBIN']
    TOK_NULL   = vocab['TOK_NULL']
    TOK_START  = vocab['TOK_START']

    if n_steps is None:
        n_steps = len(trajectory)

    n_q_bins  = config.n_q_bins
    q_bin_min = config.q_bin_min
    q_bin_max = config.q_bin_max
    n_discrete = 3 - n_continuous

    ids              = [TOK_NULL, TOK_START]
    reward_values    = []
    reward_positions = []
    select_positions = []
    think_positions  = []
    explain_positions = []  # list of [pos0, pos1, pos2]

    for t in range(n_steps):
        step = trajectory[t]
        s   = step['s']
        a   = step['a']
        r   = step['r']
        s_p = step['s_next']
        a_next = step['a_next']

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

        think_positions.append(len(ids))
        ids.append(TOK_THINK)

        # 3 explanation tokens in place of single COT
        # Compute Q-bin target for this round
        if q_snapshots is not None:
            # Q-value AFTER this step's TD update, at (s_t, a_t) — matches
            # q_values_for_cot in 1_generate_data.py and expand_sequence in 3_train.py.
            q_val = float(q_snapshots[t, s, a])
            q_bin = discretize_q_value(q_val, n_q_bins, q_bin_min, q_bin_max)
        else:
            q_bin = 0  # placeholder — doesn't matter for Stage 3 inference

        # Explain tokens are [s_t, a_t, Q_bin(s_t,a_t)] — matches 3_train.py expand_sequence.
        # NOT s_{t+1}/a_next: those are the *next* step and were never the training targets.
        discrete_tokens = [TOK_S[s], TOK_A[a], TOK_QBIN[q_bin]]

        exp_pos = []
        for j in range(3):
            pos = len(ids)
            exp_pos.append(pos)
            if j >= n_discrete:
                ids.append(TOK_COT)  # placeholder for continuous
            else:
                ids.append(discrete_tokens[j])
        explain_positions.append(exp_pos)

    def to_tensor(lst, dtype):
        return torch.tensor([lst], dtype=dtype, device=device)

    return {
        'input_ids':         to_tensor(ids,              torch.long),
        'reward_values':     to_tensor(reward_values,    torch.float32),
        'reward_positions':  to_tensor(reward_positions, torch.long),
        'select_positions':  to_tensor(select_positions, torch.long),
        'think_positions':   to_tensor(think_positions,  torch.long),
        'explain_positions': torch.tensor([explain_positions], dtype=torch.long, device=device),
        # explain_positions: [1, n_steps, 3]
    }


# ---------------------------------------------------------------------------
# Part 1: Action prediction evaluation
# ---------------------------------------------------------------------------

def run_action_inference(
    model: COCONUTTransformer,
    trajectory: List[Dict],
    vocab: Dict,
    n_actions: int,
    n_continuous: int,
    config: 'COCONUTConfig',
    device: torch.device,
    q_snapshots: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Feed trajectory to model step-by-step and collect predicted actions.

    At each step, builds the expanded COCONUT sequence accumulated so far,
    runs one forward_hao pass, and reads the action prediction at the latest
    SELECT position.

    Returns
    -------
    predicted_actions : np.ndarray, shape (n_steps,)  — argmax action at each step
    """
    TOK_S      = vocab['TOK_S']
    TOK_A      = vocab['TOK_A']
    TOK_R      = vocab['TOK_R']
    TOK_EVAL   = vocab['TOK_EVAL']
    TOK_SELECT = vocab['TOK_SELECT']
    TOK_THINK  = vocab['TOK_THINK']
    TOK_COT    = vocab['TOK_COT']
    TOK_QBIN   = vocab['TOK_QBIN']
    TOK_NULL   = vocab['TOK_NULL']
    TOK_START  = vocab['TOK_START']

    n_q_bins  = config.n_q_bins
    q_bin_min = config.q_bin_min
    q_bin_max = config.q_bin_max
    n_discrete = 3 - n_continuous

    model.eval()
    predicted_actions = []

    ids              = [TOK_NULL, TOK_START]
    reward_values    = []
    reward_positions = []
    select_positions = []
    think_positions  = []
    explain_positions = []  # list of [pos0, pos1, pos2]

    with torch.no_grad():
        for step_idx, traj in enumerate(trajectory):
            s   = traj['s']
            a   = traj['a']
            r   = traj['r']
            s_p = traj['s_next']
            a_next = traj['a_next']

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

            think_positions.append(len(ids))
            ids.append(TOK_THINK)

            # 3 explanation tokens: [s_t, a_t, Q_bin(s_t,a_t)] — matches training.
            # Use real Q-bin when q_snapshots is available (needed for the
            # no-COCONUT ablation where n_discrete=3 and all tokens go in the sequence).
            if q_snapshots is not None:
                q_val = float(q_snapshots[step_idx, s, a])
                q_bin = discretize_q_value(q_val, n_q_bins, q_bin_min, q_bin_max)
            else:
                q_bin = 0
            discrete_tokens = [TOK_S[s], TOK_A[a], TOK_QBIN[q_bin]]

            exp_pos = []
            for j in range(3):
                pos = len(ids)
                exp_pos.append(pos)
                if j >= n_discrete:
                    ids.append(TOK_COT)
                else:
                    ids.append(discrete_tokens[j])
            explain_positions.append(exp_pos)

            # Forward pass on full accumulated sequence
            input_ids_t = torch.tensor([ids],               dtype=torch.long,    device=device)
            rew_vals_t  = torch.tensor([reward_values],     dtype=torch.float32, device=device)
            rew_pos_t   = torch.tensor([reward_positions],  dtype=torch.long,    device=device)
            sel_pos_t   = torch.tensor([select_positions],  dtype=torch.long,    device=device)
            thk_pos_t   = torch.tensor([think_positions],   dtype=torch.long,    device=device)
            exp_pos_t   = torch.tensor([explain_positions], dtype=torch.long,    device=device)

            action_logits, _ = model.forward_hao(
                input_ids        = input_ids_t,
                reward_values    = rew_vals_t,
                reward_positions = rew_pos_t,
                select_positions = sel_pos_t,
                think_positions  = thk_pos_t,
                explain_positions = exp_pos_t,
                n_continuous     = n_continuous,
            )

            # action_logits : [1, n_sel_so_far, n_actions] — take the last (current step)
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
        self.linear = nn.Linear(d_model, n_states * n_actions)
        self.n_states  = n_states
        self.n_actions = n_actions

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """h : [N, d_model]  ->  [N, n_states, n_actions]"""
        return self.linear(h).view(-1, self.n_states, self.n_actions)


def collect_probe_data(
    model: COCONUTTransformer,
    n_trajectories: int,
    n_states: int,
    n_actions: int,
    vocab: Dict,
    n_continuous: int,
    config: 'COCONUTConfig',
    device: torch.device,
    n_steps: int = 50,
    seed_offset: int = 20000,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Collect hidden states and Q-table targets for probe training/evaluation.

    For each trajectory, calls forward_hao ONCE on the full sequence with
    return_hidden=True, then extracts hidden states at THINK positions and
    last explain positions (where the continuous thought lives in stages 1-3).

    Returns
    -------
    h_think_all   : (n_trajectories * n_steps, d_model)
    h_explain_all : (n_trajectories * n_steps, d_model)
    q_true_all    : (n_trajectories * n_steps, n_states, n_actions)
    """
    model.eval()
    h_think_list   = []
    h_explain_list = []
    q_true_list    = []

    with torch.no_grad():
        for i in range(n_trajectories):
            seed = seed_offset + i
            P, R = generate_eval_mdp(n_states, n_actions, seed=seed)
            trajectory, q_snapshots = run_tabular_q_learning(
                P, R, n_states, n_actions, n_steps=n_steps, seed=seed
            )

            # Build full-sequence tensors (1 forward call per trajectory)
            tensors = trajectory_to_tensors(
                trajectory, vocab, n_actions, n_continuous, config, device,
                q_snapshots=q_snapshots,
            )

            _, _, h_final = model.forward_hao(
                input_ids         = tensors['input_ids'],
                reward_values     = tensors['reward_values'],
                reward_positions  = tensors['reward_positions'],
                select_positions  = tensors['select_positions'],
                think_positions   = tensors['think_positions'],
                explain_positions = tensors['explain_positions'],
                n_continuous      = n_continuous,
                return_hidden     = True,
            )
            # h_final : [1, T, d_model]

            # Extract hidden states at THINK and last explain positions
            thk_pos = tensors['think_positions'][0].cpu().numpy()        # (n_steps,)
            exp_pos = tensors['explain_positions'][0, :, -1].cpu().numpy()  # (n_steps,) — last explain pos
            h = h_final[0].cpu().numpy()                                  # (T, d_model)

            for t in range(n_steps):
                h_think_list.append(h[thk_pos[t]])
                h_explain_list.append(h[exp_pos[t]])
                q_true_list.append(q_snapshots[t])

    return (
        np.stack(h_think_list,   axis=0),   # (N, d_model)
        np.stack(h_explain_list, axis=0),   # (N, d_model)
        np.stack(q_true_list,    axis=0),   # (N, n_states, n_actions)
    )


def train_probe(
    probe: QProbe,
    h_all: np.ndarray,     # (N, d_model)
    q_all: np.ndarray,     # (N, n_states, n_actions)
    device: torch.device,
    n_epochs: int = 10,
    batch_size: int = 256,
    lr: float = 1e-3,
) -> List[float]:
    """Train the linear probe with Adam + MSE loss. Returns per-epoch MSE."""
    probe.train()
    optimizer = torch.optim.Adam(probe.parameters(), lr=lr)
    N = h_all.shape[0]
    losses = []

    h_t = torch.tensor(h_all, dtype=torch.float32, device=device)
    q_t = torch.tensor(q_all, dtype=torch.float32, device=device)

    for epoch in range(n_epochs):
        perm = torch.randperm(N, device=device)
        epoch_loss = 0.0
        n_batches  = 0
        for i in range(0, N, batch_size):
            idx  = perm[i:i + batch_size]
            h_b  = h_t[idx]
            q_b  = q_t[idx]
            pred = probe(h_b)
            loss = F.mse_loss(pred, q_b)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches  += 1
        losses.append(epoch_loss / max(n_batches, 1))

    return losses


def evaluate_probe(
    probe: QProbe,
    h_all: np.ndarray,   # (N, d_model)
    q_all: np.ndarray,   # (N, n_states, n_actions)
    device: torch.device,
) -> Tuple[float, float, np.ndarray]:
    """Evaluate probe: Frobenius error per step and R².

    Returns
    -------
    r2       : float   — coefficient of determination (higher = better)
    frob_err : float   — mean Frobenius error per Q-table
    q_pred   : np.ndarray (N, n_states, n_actions) — probe predictions
    """
    probe.eval()
    with torch.no_grad():
        h_t    = torch.tensor(h_all, dtype=torch.float32, device=device)
        q_pred = probe(h_t).cpu().numpy()   # (N, n_states, n_actions)

    # Flatten for R² computation
    q_true_flat = q_all.reshape(-1)
    q_pred_flat = q_pred.reshape(-1)

    ss_res = np.sum((q_true_flat - q_pred_flat) ** 2)
    ss_tot = np.sum((q_true_flat - q_true_flat.mean()) ** 2) + 1e-12
    r2     = float(1.0 - ss_res / ss_tot)

    # Per-table Frobenius error
    diff  = q_pred - q_all
    frob  = np.sqrt((diff ** 2).sum(axis=(1, 2)))   # (N,)
    frob_mean = float(frob.mean())

    return r2, frob_mean, q_pred


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def plot_action_agreement(
    aa_mean_id:  np.ndarray,  # (n_steps,)
    aa_std_id:   np.ndarray,
    aa_mean_ood: np.ndarray,  # (n_steps,)
    aa_std_ood:  np.ndarray,
    save_path: str,
    n_mdps: int = 10,
    label_suffix: str = '',
) -> None:
    """Plot per-step action agreement for in-distribution and OOD MDPs."""
    steps = np.arange(1, len(aa_mean_id) + 1)
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(steps, aa_mean_id, color='steelblue', linewidth=2, label='In-distribution (ID)')
    ax.fill_between(steps, aa_mean_id - aa_std_id, aa_mean_id + aa_std_id,
                    alpha=0.25, color='steelblue')

    ax.plot(steps, aa_mean_ood, color='darkorange', linewidth=2,
            label='Out-of-distribution (OOD)', linestyle='--')
    ax.fill_between(steps, aa_mean_ood - aa_std_ood, aa_mean_ood + aa_std_ood,
                    alpha=0.2, color='darkorange')

    ax.axhline(y=1.0, color='gray', linestyle=':', alpha=0.5, label='Perfect agreement')
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Greedy Action Agreement')
    ax.set_ylim(-0.05, 1.1)
    suffix = f' ({label_suffix})' if label_suffix else ''
    ax.set_title(
        f'Per-Step Action Agreement: ID vs OOD{suffix}\n'
        f'(mean ± std, {n_mdps} MDPs each)'
    )
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_probe_scatter(
    q_true: np.ndarray,   # (N, n_states, n_actions)
    q_pred: np.ndarray,   # (N, n_states, n_actions)
    probe_name: str,
    r2: float,
    save_path: str,
) -> None:
    q_t = q_true.reshape(-1)
    q_p = q_pred.reshape(-1)
    # Subsample for plotting (avoid 50K points)
    idx = np.random.choice(len(q_t), size=min(10000, len(q_t)), replace=False)
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(q_t[idx], q_p[idx], alpha=0.1, s=3, color='steelblue', rasterized=True)
    lo = min(q_t.min(), q_p.min())
    hi = max(q_t.max(), q_p.max())
    ax.plot([lo, hi], [lo, hi], 'r--', linewidth=1.5, label='y = x')
    ax.set_xlabel('Q tabular')
    ax.set_ylabel('Q probe')
    ax.set_title(f'Probe ({probe_name}) Q-Values vs True Q-Values\n'
                 f'R² = {r2:.4f}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_probe_frobenius(
    q_pred_all:  np.ndarray,   # (n_traj * n_steps, n_states, n_actions)
    q_true_all:  np.ndarray,   # (n_traj * n_steps, n_states, n_actions)
    probe_name:  str,
    n_steps:     int,
    n_traj_eval: int,
    save_path:   str,
) -> None:
    """Plot probe Frobenius error over trajectory steps + zero baseline."""
    # Reshape to (n_traj_eval, n_steps, n_states, n_actions)
    q_pred_r = q_pred_all.reshape(n_traj_eval, n_steps, -1)
    q_true_r = q_true_all.reshape(n_traj_eval, n_steps, -1)

    # Probe error per step
    diff       = q_pred_r - q_true_r                             # (T, n_steps, n_s*n_a)
    frob_probe = np.sqrt((diff ** 2).sum(axis=-1))              # (T, n_steps)
    frob_mean  = frob_probe.mean(axis=0)                        # (n_steps,)
    frob_std   = frob_probe.std(axis=0)                         # (n_steps,)

    # Zero baseline: ||Q_tabular(t)||_F (predicting Q=0 everywhere)
    zero_baseline = np.sqrt((q_true_r ** 2).sum(axis=-1)).mean(axis=0)  # (n_steps,)

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


def plot_training_curves(log_path: str, save_path: str) -> None:
    """Load training log and plot train/val CE + accuracy."""
    if not os.path.exists(log_path):
        print(f"  Warning: training log not found at {log_path}, skipping training_curves.png")
        return

    data = np.load(log_path)
    steps     = data['steps']
    train_ce  = data['train_ce']
    train_acc = data['train_acc']
    val_ce    = data['val_ce']
    val_acc   = data['val_acc']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(steps, train_ce, color='steelblue', linewidth=1.5, label='Train CE')
    ax1.plot(steps, val_ce,   color='darkorange', linewidth=1.5, label='Val CE')
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Cross-Entropy Loss')
    ax1.set_title('CE Loss Over Training')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(steps, train_acc * 100, color='steelblue', linewidth=1.5, label='Train Acc')
    ax2.plot(steps, val_acc   * 100, color='darkorange', linewidth=1.5, label='Val Acc')
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Action Prediction Accuracy Over Training')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.suptitle('COCONUT Q-Learning — Training Curves', fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Evaluate trained COCONUT transformer')
    parser.add_argument('--checkpoint', type=str,
                        default=os.path.join(_script_dir, '..', 'checkpoints',
                                             'coconut_transformer.pt'))
    parser.add_argument('--figures_dir', type=str,
                        default=os.path.join(_script_dir, '..', 'figures'))
    parser.add_argument('--n_steps',     type=int,   default=50)
    parser.add_argument('--alpha',       type=float, default=0.1)
    parser.add_argument('--gamma',       type=float, default=0.9)
    parser.add_argument('--epsilon',     type=float, default=0.2)
    parser.add_argument('--eval_seed',   type=int,   default=9999,
                        help='Base seed; evaluation runs seeds eval_seed..eval_seed+9')
    parser.add_argument('--n_eval_mdps', type=int,   default=10)
    parser.add_argument('--n_probe_train', type=int, default=1000,
                        help='Trajectories for probe training')
    parser.add_argument('--n_probe_eval',  type=int, default=100,
                        help='Trajectories for probe evaluation')
    parser.add_argument('--probe_epochs',  type=int, default=10)
    args = parser.parse_args()

    os.makedirs(args.figures_dir, exist_ok=True)

    # ---- Load checkpoint ----
    print(f"Loading checkpoint from {args.checkpoint} ...")
    ckpt   = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    config = COCONUTConfig.from_dict(ckpt['config'])
    model  = COCONUTTransformer(config)
    model.load_state_dict(ckpt['model_state_dict'])

    # Determine n_continuous from checkpoint
    n_continuous = ckpt.get('n_continuous', 3)  # default 3 for old checkpoints
    use_coconut = ckpt.get('use_coconut', True)
    print(f"  Loaded (trained {ckpt.get('epoch', '?')} epochs, "
          f"step {ckpt.get('step', '?')}, "
          f"val_ce={ckpt.get('val_ce_loss', ckpt.get('val_loss', '?')):.4f})")
    print(f"  n_continuous = {n_continuous}, use_coconut = {use_coconut}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model  = model.to(device)
    model.eval()

    vocab     = build_vocab(config.n_states, config.n_actions, config.n_q_bins)
    n_states  = config.n_states
    n_actions = config.n_actions

    print(f"\nModel: {model.num_parameters():,} params  |  device: {device}")
    print(f"MDP:   n_states={n_states}, n_actions={n_actions}")

    eval_seeds = list(range(args.eval_seed, args.eval_seed + args.n_eval_mdps))
    # OOD seeds are offset so they don't overlap with ID seeds
    ood_seeds  = list(range(args.eval_seed + 50000, args.eval_seed + 50000 + args.n_eval_mdps))

    label = f"COCONUT (n_cont={n_continuous})" if use_coconut else "No-COCONUT"

    # -----------------------------------------------------------------------
    # Part 1a: Action prediction — in-distribution MDPs
    # -----------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"Part 1a: In-distribution action prediction on {args.n_eval_mdps} MDPs "
          f"(seeds {eval_seeds[0]}–{eval_seeds[-1]})")
    print(f"  Dirichlet(1) transitions, Beta(2,2) rewards  [training distribution]")
    print(f"{'='*60}")

    all_agreements_id = []
    for seed_i, seed in enumerate(eval_seeds):
        print(f"  ID MDP {seed_i+1}/{args.n_eval_mdps} (seed={seed}) ...", end=' ', flush=True)

        P, R = generate_eval_mdp(n_states, n_actions, seed=seed)
        trajectory, q_snapshots_eval = run_tabular_q_learning(
            P, R, n_states, n_actions,
            n_steps  = args.n_steps,
            alpha    = args.alpha,
            gamma    = args.gamma,
            epsilon  = args.epsilon,
            seed     = seed,
        )

        preds   = run_action_inference(model, trajectory, vocab, n_actions,
                                       n_continuous, config, device,
                                       q_snapshots=q_snapshots_eval)
        targets = np.array([step['a_next'] for step in trajectory], dtype=np.int32)
        agree   = (preds == targets).astype(np.float32)
        all_agreements_id.append(agree)
        print(f"agree={agree.mean():.2%}")

    aa_id_arr  = np.stack(all_agreements_id, axis=0)   # (n_mdps, n_steps)
    aa_id_mean = aa_id_arr.mean(axis=0)
    aa_id_std  = aa_id_arr.std(axis=0)
    print(f"\n  Mean ID action agreement ({label}): "
          f"{aa_id_mean.mean():.2%} ± {aa_id_arr.mean(axis=1).std():.4f}")

    # -----------------------------------------------------------------------
    # Part 1b: Action prediction — out-of-distribution MDPs
    # -----------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"Part 1b: Out-of-distribution action prediction on {args.n_eval_mdps} MDPs "
          f"(seeds {ood_seeds[0]}–{ood_seeds[-1]})")
    print(f"  Dirichlet(0.1) transitions (sparse/peaked), Beta(0.5,0.5) rewards (bimodal)")
    print(f"{'='*60}")

    all_agreements_ood = []
    for seed_i, seed in enumerate(ood_seeds):
        print(f"  OOD MDP {seed_i+1}/{args.n_eval_mdps} (seed={seed}) ...", end=' ', flush=True)

        P, R = generate_ood_mdp(n_states, n_actions, seed=seed)
        trajectory, q_snapshots_ood = run_tabular_q_learning(
            P, R, n_states, n_actions,
            n_steps  = args.n_steps,
            alpha    = args.alpha,
            gamma    = args.gamma,
            epsilon  = args.epsilon,
            seed     = seed,
        )

        preds   = run_action_inference(model, trajectory, vocab, n_actions,
                                       n_continuous, config, device,
                                       q_snapshots=q_snapshots_ood)
        targets = np.array([step['a_next'] for step in trajectory], dtype=np.int32)
        agree   = (preds == targets).astype(np.float32)
        all_agreements_ood.append(agree)
        print(f"agree={agree.mean():.2%}")

    aa_ood_arr  = np.stack(all_agreements_ood, axis=0)  # (n_mdps, n_steps)
    aa_ood_mean = aa_ood_arr.mean(axis=0)
    aa_ood_std  = aa_ood_arr.std(axis=0)
    print(f"\n  Mean OOD action agreement ({label}): "
          f"{aa_ood_mean.mean():.2%} ± {aa_ood_arr.mean(axis=1).std():.4f}")

    gap = aa_id_mean.mean() - aa_ood_mean.mean()
    print(f"  ID - OOD gap: {gap:+.4f}")

    aa_path = os.path.join(args.figures_dir, 'action_agreement.png')
    plot_action_agreement(
        aa_id_mean, aa_id_std,
        aa_ood_mean, aa_ood_std,
        save_path=aa_path,
        n_mdps=args.n_eval_mdps,
        label_suffix=label,
    )

    # Keep aa_arr pointing to ID results for probe collection below
    aa_arr = aa_id_arr

    # -----------------------------------------------------------------------
    # Part 2: Q-value probing (only makes sense for COCONUT model)
    # -----------------------------------------------------------------------
    if use_coconut and n_continuous > 0:
        print(f"\n{'='*60}")
        print("Part 2: Q-value probing")
        print(f"{'='*60}")

        # Freeze model
        for p in model.parameters():
            p.requires_grad_(False)

        # Collect probe training data (1000 trajectories, 1 forward pass each)
        print(f"\nCollecting probe training data ({args.n_probe_train} trajectories) ...")
        h_think_train, h_explain_train, q_true_train = collect_probe_data(
            model, args.n_probe_train, n_states, n_actions, vocab,
            n_continuous, config, device,
            n_steps=args.n_steps, seed_offset=20000,
        )
        print(f"  h_think_train: {h_think_train.shape},  q_true_train: {q_true_train.shape}")

        # Collect probe eval data (100 held-out trajectories)
        print(f"\nCollecting probe eval data ({args.n_probe_eval} trajectories) ...")
        h_think_eval, h_explain_eval, q_true_eval = collect_probe_data(
            model, args.n_probe_eval, n_states, n_actions, vocab,
            n_continuous, config, device,
            n_steps=args.n_steps, seed_offset=30000,
        )

        # Train and evaluate both probes
        results = {}
        for probe_name, h_train, h_eval in [
            ('think',   h_think_train,   h_think_eval),
            ('explain', h_explain_train, h_explain_eval),
        ]:
            print(f"\nTraining probe_{probe_name} ...")
            probe = QProbe(config.d_model, n_states, n_actions).to(device)
            losses = train_probe(probe, h_train, q_true_train, device,
                                 n_epochs=args.probe_epochs)
            print(f"  Training MSE per epoch: {['%.4f' % l for l in losses]}")

            r2, frob_mean, q_pred = evaluate_probe(probe, h_eval, q_true_eval, device)
            print(f"  Eval  probe_{probe_name}: R²={r2:.4f}  Frobenius={frob_mean:.4f}")
            results[probe_name] = {'r2': r2, 'frob': frob_mean, 'q_pred': q_pred}

        # Determine primary probe (higher R²)
        primary = 'think' if results['think']['r2'] >= results['explain']['r2'] else 'explain'
        print(f"\n  Primary probe: probe_{primary}  (R²={results[primary]['r2']:.4f})")
        if primary == 'think':
            print("  Interpretation: model computes Q-values at the THINK step; "
                  "COCONUT copies them forward via continuous thought injection.")
        else:
            print("  Interpretation: model refines Q-values during the final transformer pass "
                  "after COCONUT injection.")

        # Scatter plot (primary probe)
        scatter_path = os.path.join(args.figures_dir, 'probe_scatter.png')
        plot_probe_scatter(
            q_true_eval,
            results[primary]['q_pred'],
            probe_name=primary,
            r2=results[primary]['r2'],
            save_path=scatter_path,
        )

        # Frobenius plot (primary probe)
        frob_path = os.path.join(args.figures_dir, 'probe_frobenius.png')
        plot_probe_frobenius(
            results[primary]['q_pred'],
            q_true_eval,
            probe_name=primary,
            n_steps=args.n_steps,
            n_traj_eval=args.n_probe_eval,
            save_path=frob_path,
        )
    else:
        print("\n  (Skipping Q-value probing — not applicable without continuous thoughts)")

    # -----------------------------------------------------------------------
    # Training curves
    # -----------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("Training curves")
    print(f"{'='*60}")
    log_path  = os.path.join(os.path.dirname(args.checkpoint), 'training_log.npz')
    curves_path = os.path.join(args.figures_dir, 'training_curves.png')
    plot_training_curves(log_path, curves_path)

    print(f"\nDone. Figures saved to {args.figures_dir}/")


if __name__ == '__main__':
    main()
