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

    After appending each TOK_UPDATE token, runs a forward pass and extracts
    the q_value_preds at the latest Update position.

    The model receives the full accumulated sequence (causal) at each step,
    so it has access to all prior context — exactly as intended by COCONUT.

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

    model.eval()
    q_preds = []

    # Accumulated sequence state
    ids              = [TOK_NULL, TOK_START]
    reward_values    : List[float] = []
    reward_positions : List[int]   = []

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

            # ---- Forward pass ----
            T = len(ids)
            input_ids_t = torch.tensor([ids], dtype=torch.long, device=device)    # [1, T]
            rew_vals_t  = torch.tensor([reward_values], dtype=torch.float32, device=device)  # [1, n_r]
            rew_pos_t   = torch.tensor([reward_positions], dtype=torch.long, device=device)  # [1, n_r]
            sel_pos_t   = torch.tensor([[0]], dtype=torch.long, device=device)     # dummy
            upd_pos_t   = torch.tensor([[upd_pos]], dtype=torch.long, device=device)

            action_logits, q_value_preds_t = model(
                input_ids        = input_ids_t,
                reward_values    = rew_vals_t,
                reward_positions = rew_pos_t,
                select_positions = sel_pos_t,
                update_positions = upd_pos_t,
            )
            # q_value_preds_t : [1, 1, n_actions]
            q_row = q_value_preds_t[0, 0].cpu().numpy()   # [n_actions]
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
                        help='Seed for the evaluation MDP (must differ from training seed)')
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

    # ---- Generate unseen eval MDP ----
    print(f"\nGenerating fresh eval MDP (seed={args.eval_seed}) ...")
    P, R = generate_eval_mdp(n_states, n_actions, seed=args.eval_seed)

    # ---- Tabular Q-learning (ground truth) ----
    print(f"Running tabular Q-learning for {args.n_steps} steps ...")
    trajectory, q_tabular = run_tabular_q_learning(
        P, R, n_states, n_actions,
        n_steps  = args.n_steps,
        alpha    = args.alpha,
        gamma    = args.gamma,
        epsilon  = args.epsilon,
        seed     = args.eval_seed,
    )
    # q_tabular : (n_steps, n_states, n_actions)

    # ---- Transformer inference ----
    print("Running transformer inference step-by-step ...")
    q_row_preds = run_transformer_inference(model, trajectory, vocab, n_actions, device)
    # q_row_preds : (n_steps, n_actions)

    # Reconstruct full Q-table history from row predictions
    q_model = build_qtable_from_row_preds(q_row_preds, trajectory, n_states, n_actions)
    # q_model : (n_steps, n_states, n_actions)

    # ---- Frobenius errors ----
    errors = frobenius_errors(q_model, q_tabular)   # (n_steps,)

    print("\n--- Frobenius Norm Error ||Q_model(t) - Q_tabular(t)||_F ---")
    checkpoints = [4, 14, args.n_steps - 1]
    for t in checkpoints:
        tc = min(t, len(errors) - 1)
        print(f"  Step {tc+1:3d}: {errors[tc]:.4f}")
    print(f"  Mean over all steps: {errors.mean():.4f}")
    print(f"  Final step {args.n_steps}: {errors[-1]:.4f}")

    # ---- Plots ----
    print("\nGenerating plots ...")
    frob_path = os.path.join(args.figures_dir, 'frobenius_norm.png')
    plot_frobenius_norm(errors, frob_path)

    cmp_path = os.path.join(args.figures_dir, 'qtable_comparison.png')
    plot_qtable_comparison(q_model, q_tabular, checkpoints, cmp_path)

    # ---- Action accuracy at SELECT positions ----
    # Count how often the greedy action from the transformer prediction
    # matches the greedy action from the tabular Q-table.
    model_greedy   = [int(np.argmax(q_row_preds[t]))    for t in range(args.n_steps)]
    tabular_greedy = [int(np.argmax(q_tabular[t, traj['s']])) for t, traj in enumerate(trajectory)]
    action_match = sum(m == t for m, t in zip(model_greedy, tabular_greedy))
    print(f"\nGreedy action agreement (model vs tabular): "
          f"{action_match}/{args.n_steps} = {action_match/args.n_steps:.2%}")


if __name__ == '__main__':
    main()
