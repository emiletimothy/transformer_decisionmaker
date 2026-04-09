#!/usr/bin/env python3
"""
4_evaluate.py — Evaluation Script

Verifies that the trained COCONUTTransformer implements Q-learning on unseen data
by comparing its <Update>-token Q-value predictions against a ground-truth tabular
Q-learning agent running on an identical trajectory.

Procedure
---------
1. Load trained checkpoint from checkpoints/coconut_transformer.pt
2. Generate a fresh random MDP (same logic as 1_generate_data.py) with a fixed seed
3. Run tabular Q-learning on a T=50 trajectory → ground-truth Q-table snapshots
4. Feed the same trajectory to the transformer step-by-step in COCONUT format:
       For each step t:
           - Append PHASE I + PHASE II tokens up to and including TOK_UPDATE
           - Run forward pass on the accumulated sequence
           - Extract q_value_preds at the latest <Update> position
5. Compute Frobenius norm ||Q_model(t) - Q_tabular(t)||_F at each timestep
6. Plot results and save to figures/

Output files
------------
    figures/frobenius_norm.png        — error over time
    figures/qtable_comparison.png     — Q-table heatmaps at t=5, t=15, t=T-1
"""

import importlib.util
import math
import os
import sys
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

# ---------------------------------------------------------------------------
# Import COCONUTTransformer and COCONUTConfig from 2_model.py
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


# ---------------------------------------------------------------------------
# MDP + tabular Q-learning (mirrors 1_generate_data.py logic)
# ---------------------------------------------------------------------------

def generate_eval_mdp(
    n_states: int,
    n_actions: int,
    trap_prob: float = 0.2,
    seed: int = 9999,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a fresh MDP with a fixed seed (different from training seed 42)."""
    rng = np.random.default_rng(seed)
    P = rng.dirichlet(alpha=np.ones(n_states), size=(n_states, n_actions)).astype(np.float32)
    R = rng.beta(2.0, 2.0, size=(n_states, n_actions)).astype(np.float32)
    R = np.clip(R, 0.0, 1.0)

    trap = rng.random(n_states) < trap_prob
    for s in np.where(trap)[0]:
        P[s, :, :] = 0.0
        P[s, :, s] = 1.0
        R[s, :] = 0.0

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
    trajectory : list of step dicts (s, a, r, s_next, a_next)
    q_snapshots: np.ndarray, shape (n_steps, n_states, n_actions)
    """
    rng = np.random.default_rng(seed + 1)
    Q = np.zeros((n_states, n_actions), dtype=np.float32)
    s = int(rng.integers(n_states))

    trajectory   = []
    q_snapshots  = []

    for _ in range(n_steps):
        # Epsilon-greedy
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
            's': s, 'a': a, 'r': r, 's_next': s_next, 'a_next': a_next
        })
        q_snapshots.append(Q.copy())
        s = s_next

    return trajectory, np.stack(q_snapshots, axis=0)


# ---------------------------------------------------------------------------
# Step-by-step transformer inference
# ---------------------------------------------------------------------------

def run_transformer_inference(
    model: COCONUTTransformer,
    trajectory: List[Dict],
    vocab: Dict,
    n_actions: int,
    device: torch.device,
) -> np.ndarray:
    """Feed the trajectory to the model one step at a time in COCONUT format.

    After appending each TOK_UPDATE + TOK_COT token pair, calls
    model.forward_coconut() on the full accumulated sequence.  The COCONUT
    forward pass injects the hidden state from each UPDATE position into the
    corresponding COT position before the final transformer pass, giving the
    model access to continuously-evolving Q-value representations.

    Returns
    -------
    q_preds : np.ndarray, shape (n_steps, n_actions)
              Q-row predictions Q[s_t, :] at each timestep t
    """
    TOK_NULL   = vocab['TOK_NULL']
    TOK_START  = vocab['TOK_START']
    TOK_S      = vocab['TOK_S']
    TOK_A      = vocab['TOK_A']
    TOK_R      = vocab['TOK_R']
    TOK_EVAL   = vocab['TOK_EVAL']
    TOK_SELECT = vocab['TOK_SELECT']
    TOK_QCURR  = vocab['TOK_QCURR']
    TOK_QNEXT  = vocab['TOK_QNEXT']
    TOK_UPDATE = vocab['TOK_UPDATE']
    TOK_COT    = vocab['TOK_COT']

    model.eval()
    q_preds = []

    # Accumulated sequence state
    ids              = [TOK_NULL, TOK_START]
    reward_values    : List[float] = []
    reward_positions : List[int]   = []
    update_positions : List[int]   = []
    cot_positions    : List[int]   = []

    with torch.no_grad():
        for step, traj in enumerate(trajectory):
            s     = traj['s']
            a     = traj['a']
            r     = traj['r']
            s_p   = traj['s_next']
            a_p   = traj['a_next']

            # ---- PHASE I ----
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

            ids.append(TOK_SELECT)

            # ---- PHASE II ----
            ids.append(TOK_A[a_p])
            ids.append(TOK_QCURR)
            ids.append(TOK_QNEXT)
            upd_pos = len(ids)   # position of this UPDATE token
            ids.append(TOK_UPDATE)
            update_positions.append(upd_pos)

            # ---- COT placeholder ----
            cot_pos = len(ids)   # position of the COT token
            ids.append(TOK_COT)
            cot_positions.append(cot_pos)

            # ---- COCONUT forward pass on full accumulated sequence ----
            input_ids_t = torch.tensor([ids],              dtype=torch.long,    device=device)
            rew_vals_t  = torch.tensor([reward_values],    dtype=torch.float32, device=device)
            rew_pos_t   = torch.tensor([reward_positions], dtype=torch.long,    device=device)
            sel_pos_t   = torch.tensor([[0]],              dtype=torch.long,    device=device)
            upd_pos_t   = torch.tensor([update_positions], dtype=torch.long,    device=device)
            cot_pos_t   = torch.tensor([cot_positions],   dtype=torch.long,    device=device)

            action_logits, q_value_preds_t = model.forward_coconut(
                input_ids        = input_ids_t,
                reward_values    = rew_vals_t,
                reward_positions = rew_pos_t,
                select_positions = sel_pos_t,
                update_positions = upd_pos_t,
                cot_positions    = cot_pos_t,
            )
            # q_value_preds_t : [1, n_upd_so_far, n_actions]
            # The last entry corresponds to the current step's UPDATE position
            q_row = q_value_preds_t[0, -1].cpu().numpy()   # [n_actions]
            q_preds.append(q_row)

    return np.stack(q_preds, axis=0)   # (n_steps, n_actions)


# ---------------------------------------------------------------------------
# Build full Q-table from single-row predictions
# ---------------------------------------------------------------------------

def build_qtable_from_row_preds(
    q_row_preds: np.ndarray,        # (n_steps, n_actions)  — Q[s_t, :] at each step
    trajectory: List[Dict],
    n_states: int,
    n_actions: int,
) -> np.ndarray:
    """Reconstruct a (n_steps, n_states, n_actions) Q-table history.

    At each timestep t, we know Q[s_t, :] from the model's Update prediction.
    We accumulate a running Q-table, overwriting the row for s_t at each step.
    Unvisited rows remain zero (matching tabular initialization).

    Returns
    -------
    qtable_history : (n_steps, n_states, n_actions)
    """
    Q = np.zeros((n_states, n_actions), dtype=np.float32)
    history = []
    for t, traj in enumerate(trajectory):
        s = traj['s']
        Q[s, :] = q_row_preds[t]
        history.append(Q.copy())
    return np.stack(history, axis=0)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def frobenius_errors(
    q_model: np.ndarray,    # (n_steps, n_states, n_actions)
    q_tabular: np.ndarray,  # (n_steps, n_states, n_actions)
) -> np.ndarray:
    """Compute ||Q_model(t) - Q_tabular(t)||_F for each t."""
    diff = q_model - q_tabular
    return np.sqrt((diff ** 2).sum(axis=(1, 2)))   # (n_steps,)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_frobenius_norm(
    errors: np.ndarray,
    save_path: str,
) -> None:
    """Plot Frobenius norm error over time."""
    steps = np.arange(1, len(errors) + 1)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(steps, errors, color='steelblue', linewidth=2, label='Frobenius error')
    ax.axhline(y=errors[0], color='gray', linestyle='--', alpha=0.6,
               label=f'Initial error ({errors[0]:.3f})')
    ax.set_xlabel('Timestep')
    ax.set_ylabel(r'$\|Q_{model}(t) - Q_{tabular}(t)\|_F$')
    ax.set_title('COCONUT Transformer vs Tabular Q-Learning\nFrobenius Norm Error Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_qtable_comparison(
    q_model: np.ndarray,     # (n_steps, n_states, n_actions)
    q_tabular: np.ndarray,   # (n_steps, n_states, n_actions)
    checkpoints: List[int],  # list of timestep indices to visualise
    save_path: str,
) -> None:
    """Plot side-by-side Q-table heatmaps at selected timesteps."""
    n_ckpt   = len(checkpoints)
    n_states = q_model.shape[1]

    fig, axes = plt.subplots(n_ckpt, 2, figsize=(8, 3 * n_ckpt))
    if n_ckpt == 1:
        axes = axes[np.newaxis, :]

    vmin = min(q_tabular.min(), q_model.min())
    vmax = max(q_tabular.max(), q_model.max())

    for row, t in enumerate(checkpoints):
        t_clamped = min(t, len(q_model) - 1)
        Qm = q_model[t_clamped]
        Qt = q_tabular[t_clamped]

        im0 = axes[row, 0].imshow(Qt, aspect='auto', vmin=vmin, vmax=vmax, cmap='viridis')
        axes[row, 0].set_title(f'Tabular  (t={t_clamped+1})')
        axes[row, 0].set_xlabel('Action')
        axes[row, 0].set_ylabel('State')
        plt.colorbar(im0, ax=axes[row, 0])

        im1 = axes[row, 1].imshow(Qm, aspect='auto', vmin=vmin, vmax=vmax, cmap='viridis')
        axes[row, 1].set_title(f'Transformer  (t={t_clamped+1})')
        axes[row, 1].set_xlabel('Action')
        axes[row, 1].set_ylabel('State')
        plt.colorbar(im1, ax=axes[row, 1])

    fig.suptitle('Q-Table Comparison: Tabular vs COCONUT Transformer', fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


# ---------------------------------------------------------------------------
# Additional metrics
# ---------------------------------------------------------------------------

def per_state_errors(
    q_model: np.ndarray,    # (n_steps, n_states, n_actions)
    q_tabular: np.ndarray,  # (n_steps, n_states, n_actions)
) -> np.ndarray:
    """Compute per-state L2 error ||Q_model(t,s,:) - Q_tabular(t,s,:)||_2 for each t,s.

    Returns : (n_steps, n_states)
    """
    diff = q_model - q_tabular   # (n_steps, n_states, n_actions)
    return np.linalg.norm(diff, axis=-1)  # (n_steps, n_states)


def relative_frobenius_errors(
    q_model: np.ndarray,    # (n_steps, n_states, n_actions)
    q_tabular: np.ndarray,  # (n_steps, n_states, n_actions)
) -> np.ndarray:
    """Compute relative Frobenius error ||Q_model(t) - Q_tabular(t)||_F / (||Q_tabular(t)||_F + 1e-8).

    Returns : (n_steps,)
    """
    T = q_model.shape[0]
    rel = np.zeros(T, dtype=np.float32)
    for t in range(T):
        num = np.linalg.norm(q_model[t] - q_tabular[t], 'fro')
        den = np.linalg.norm(q_tabular[t], 'fro') + 1e-8
        rel[t] = num / den
    return rel


def per_step_action_agreement(
    q_row_preds: np.ndarray,    # (n_steps, n_actions)
    q_tabular: np.ndarray,      # (n_steps, n_states, n_actions)
    trajectory: List[Dict],
) -> np.ndarray:
    """Compute per-step greedy action agreement between model and tabular Q-table.

    Returns : (n_steps,) float array with values 0.0 or 1.0
    """
    n_steps = len(trajectory)
    agree = np.zeros(n_steps, dtype=np.float32)
    for t, traj in enumerate(trajectory):
        model_a   = int(np.argmax(q_row_preds[t]))
        tabular_a = int(np.argmax(q_tabular[t, traj['s']]))
        agree[t]  = float(model_a == tabular_a)
    return agree


# ---------------------------------------------------------------------------
# Additional plots
# ---------------------------------------------------------------------------

def plot_per_state_error(
    per_state_err: np.ndarray,   # (n_steps, n_states)  or mean over MDPs
    save_path: str,
) -> None:
    """Plot per-state L2 error over time."""
    T, n_states = per_state_err.shape
    steps = np.arange(1, T + 1)

    fig, ax = plt.subplots(figsize=(9, 5))
    for s in range(n_states):
        ax.plot(steps, per_state_err[:, s], label=f'State {s}', linewidth=1.5)
    ax.set_xlabel('Timestep')
    ax.set_ylabel(r'$\|Q_{model}(t,s,:) - Q_{tabular}(t,s,:)\|_2$')
    ax.set_title('Per-State Q-Value Error Over Time')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_relative_error(
    rel_err_mean: np.ndarray,   # (n_steps,)
    rel_err_std:  np.ndarray,   # (n_steps,)
    save_path: str,
) -> None:
    """Plot relative Frobenius error over time."""
    steps = np.arange(1, len(rel_err_mean) + 1)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(steps, rel_err_mean, color='darkorange', linewidth=2, label='Relative Frobenius error')
    ax.fill_between(steps,
                    rel_err_mean - rel_err_std,
                    rel_err_mean + rel_err_std,
                    alpha=0.25, color='darkorange')
    ax.set_xlabel('Timestep')
    ax.set_ylabel(r'$\|Q_{model}(t) - Q_{tabular}(t)\|_F \;/\; (\|Q_{tabular}(t)\|_F + \epsilon)$')
    ax.set_title('Relative Frobenius Error Over Time (mean ± std, 10 MDPs)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_q_scatter(
    q_tabular_flat: np.ndarray,  # (N,)
    q_model_flat:   np.ndarray,  # (N,)
    save_path: str,
) -> None:
    """Scatter plot of model Q-values vs tabular Q-values with y=x reference line."""
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(q_tabular_flat, q_model_flat, alpha=0.05, s=2, color='steelblue', rasterized=True)
    lo = min(q_tabular_flat.min(), q_model_flat.min())
    hi = max(q_tabular_flat.max(), q_model_flat.max())
    ax.plot([lo, hi], [lo, hi], 'r--', linewidth=1.5, label='y = x')
    ax.set_xlabel('Q tabular')
    ax.set_ylabel('Q model')
    ax.set_title('Q-Value Scatter: Model vs Tabular (all MDPs, states, actions, timesteps)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_action_agreement(
    aa_mean: np.ndarray,  # (n_steps,)
    aa_std:  np.ndarray,  # (n_steps,)
    save_path: str,
) -> None:
    """Plot per-step action agreement mean ± std over time."""
    steps = np.arange(1, len(aa_mean) + 1)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(steps, aa_mean, color='green', linewidth=2, label='Action agreement')
    ax.fill_between(steps, aa_mean - aa_std, aa_mean + aa_std, alpha=0.25, color='green')
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Perfect agreement')
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Greedy Action Agreement')
    ax.set_ylim(-0.05, 1.1)
    ax.set_title('Per-Step Action Agreement: Model vs Tabular (mean ± std, 10 MDPs)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate trained COCONUT transformer')
    parser.add_argument('--checkpoint', type=str,
                        default=os.path.join(_script_dir, '..', 'checkpoints',
                                             'coconut_transformer.pt'))
    parser.add_argument('--figures_dir', type=str,
                        default=os.path.join(_script_dir, '..', 'figures'))
    parser.add_argument('--n_steps', type=int, default=50,
                        help='Number of steps in the evaluation trajectory')
    parser.add_argument('--alpha',   type=float, default=0.1)
    parser.add_argument('--gamma',   type=float, default=0.9)
    parser.add_argument('--epsilon', type=float, default=0.2)
    parser.add_argument('--eval_seed', type=int, default=9999,
                        help='Base seed; evaluation runs seeds eval_seed .. eval_seed+9')
    parser.add_argument('--n_eval_mdps', type=int, default=10,
                        help='Number of different random MDPs to evaluate on')
    args = parser.parse_args()

    os.makedirs(args.figures_dir, exist_ok=True)

    # ---- Load checkpoint ----
    print(f"Loading checkpoint from {args.checkpoint} ...")
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    config = COCONUTConfig.from_dict(ckpt['config'])
    model  = COCONUTTransformer(config)
    model.load_state_dict(ckpt['model_state_dict'])
    print(f"  Loaded (trained {ckpt.get('epoch', '?')} epochs, "
          f"step {ckpt.get('step', '?')}, val_loss={ckpt.get('val_loss', '?'):.4f})")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model  = model.to(device)
    model.eval()

    vocab     = build_vocab(config.n_states, config.n_actions)
    n_states  = config.n_states
    n_actions = config.n_actions

    print(f"\nModel: {model.num_parameters():,} params  |  device: {device}")
    print(f"MDP:   n_states={n_states}, n_actions={n_actions}")

    # ---- Multi-MDP evaluation loop ----
    eval_seeds = list(range(args.eval_seed, args.eval_seed + args.n_eval_mdps))
    print(f"\nEvaluating on {len(eval_seeds)} MDPs (seeds {eval_seeds[0]}–{eval_seeds[-1]}) ...")

    all_frob_errors    = []   # list of (n_steps,) arrays
    all_rel_errors     = []   # list of (n_steps,) arrays
    all_per_state_errs = []   # list of (n_steps, n_states) arrays
    all_action_agree   = []   # list of (n_steps,) arrays
    all_q_tab_flat     = []   # all Q_tabular values (for scatter)
    all_q_mod_flat     = []   # all Q_model values  (for scatter)

    # Keep first MDP's data for detailed comparison plots
    first_q_model   = None
    first_q_tabular = None
    first_traj      = None

    for seed_i, seed in enumerate(eval_seeds):
        print(f"  MDP {seed_i+1}/{len(eval_seeds)} (seed={seed}) ...", end=' ', flush=True)

        P, R = generate_eval_mdp(n_states, n_actions, seed=seed)
        trajectory, q_tabular = run_tabular_q_learning(
            P, R, n_states, n_actions,
            n_steps  = args.n_steps,
            alpha    = args.alpha,
            gamma    = args.gamma,
            epsilon  = args.epsilon,
            seed     = seed,
        )

        q_row_preds = run_transformer_inference(model, trajectory, vocab, n_actions, device)
        q_model_hist = build_qtable_from_row_preds(q_row_preds, trajectory, n_states, n_actions)

        frob_err  = frobenius_errors(q_model_hist, q_tabular)
        rel_err   = relative_frobenius_errors(q_model_hist, q_tabular)
        ps_err    = per_state_errors(q_model_hist, q_tabular)
        aa        = per_step_action_agreement(q_row_preds, q_tabular, trajectory)

        all_frob_errors.append(frob_err)
        all_rel_errors.append(rel_err)
        all_per_state_errs.append(ps_err)
        all_action_agree.append(aa)
        all_q_tab_flat.append(q_tabular.flatten())
        all_q_mod_flat.append(q_model_hist.flatten())

        if seed_i == 0:
            first_q_model   = q_model_hist
            first_q_tabular = q_tabular
            first_traj      = trajectory

        print(f"frob_final={frob_err[-1]:.4f}  agree={aa.mean():.2%}")

    # ---- Aggregate statistics ----
    frob_arr  = np.stack(all_frob_errors,    axis=0)  # (n_mdps, n_steps)
    rel_arr   = np.stack(all_rel_errors,     axis=0)  # (n_mdps, n_steps)
    ps_arr    = np.stack(all_per_state_errs, axis=0)  # (n_mdps, n_steps, n_states)
    aa_arr    = np.stack(all_action_agree,   axis=0)  # (n_mdps, n_steps)

    frob_mean = frob_arr.mean(axis=0);  frob_std = frob_arr.std(axis=0)
    rel_mean  = rel_arr.mean(axis=0);   rel_std  = rel_arr.std(axis=0)
    ps_mean   = ps_arr.mean(axis=0)                   # (n_steps, n_states)
    aa_mean   = aa_arr.mean(axis=0);    aa_std   = aa_arr.std(axis=0)

    q_tab_flat = np.concatenate(all_q_tab_flat)
    q_mod_flat = np.concatenate(all_q_mod_flat)

    print(f"\n--- Results across {len(eval_seeds)} MDPs ---")
    for label, t in [('Step  5', 4), (f'Step 15', 14), (f'Step {args.n_steps}', args.n_steps - 1)]:
        tc = min(t, frob_arr.shape[1] - 1)
        print(f"  {label}: Frobenius = {frob_mean[tc]:.4f} ± {frob_std[tc]:.4f}")
    print(f"  Mean Frobenius: {frob_mean.mean():.4f} ± {frob_arr.mean(axis=1).std():.4f}")
    print(f"  Mean action agreement: {aa_mean.mean():.2%} ± {aa_arr.mean(axis=1).std():.4f}")
    print(f"  Final relative error: {rel_mean[-1]:.4f} ± {rel_std[-1]:.4f}")

    # ---- Plots ----
    print("\nGenerating plots ...")

    # Frobenius norm (mean ± std over MDPs)
    frob_path = os.path.join(args.figures_dir, 'frobenius_norm.png')
    steps = np.arange(1, args.n_steps + 1)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(steps, frob_mean, color='steelblue', linewidth=2, label='Mean Frobenius error')
    ax.fill_between(steps, frob_mean - frob_std, frob_mean + frob_std, alpha=0.25, color='steelblue')
    ax.set_xlabel('Timestep')
    ax.set_ylabel(r'$\|Q_{model}(t) - Q_{tabular}(t)\|_F$')
    ax.set_title(f'COCONUT Transformer vs Tabular Q-Learning\nFrobenius Norm Error (mean ± std, {len(eval_seeds)} MDPs)')
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(frob_path, dpi=150, bbox_inches='tight'); plt.close()
    print(f"  Saved: {frob_path}")

    # Q-table comparison heatmaps (first MDP only)
    checkpoints = [4, 14, args.n_steps - 1]
    cmp_path = os.path.join(args.figures_dir, 'qtable_comparison.png')
    plot_qtable_comparison(first_q_model, first_q_tabular, checkpoints, cmp_path)

    # Per-state error (averaged over MDPs)
    ps_path = os.path.join(args.figures_dir, 'per_state_error.png')
    plot_per_state_error(ps_mean, ps_path)

    # Relative error (mean ± std over MDPs)
    rel_path = os.path.join(args.figures_dir, 'relative_error.png')
    plot_relative_error(rel_mean, rel_std, rel_path)

    # Q-value scatter (all MDPs combined)
    scatter_path = os.path.join(args.figures_dir, 'q_scatter.png')
    plot_q_scatter(q_tab_flat, q_mod_flat, scatter_path)

    # Per-step action agreement (mean ± std over MDPs)
    aa_path = os.path.join(args.figures_dir, 'action_agreement.png')
    plot_action_agreement(aa_mean, aa_std, aa_path)

    # Per-step agreement summary
    print(f"\n--- Per-step action agreement (first MDP) ---")
    for t in checkpoints:
        tc = min(t, aa_arr.shape[1] - 1)
        print(f"  Step {tc+1:3d}: {aa_mean[tc]:.2%} ± {aa_std[tc]:.4f}")
    total_agree = int(aa_arr[0].sum())
    print(f"  Overall (first MDP): {total_agree}/{args.n_steps} = {aa_arr[0].mean():.2%}")


if __name__ == '__main__':
    main()
