"""
Compare Classical SARSA vs PyTorch Handwired Transformer SARSA (Figure 1 Analog).

Generates a 4-state / 2-action chain MDP with a dense position-based reward
signal so that all 8 Q(s,a) pairs converge to clearly distinct, positive
values within T=300 steps.  Both algorithms are fed the *identical* trajectory
and must agree exactly (Frobenius error ≈ 0).

Reward function (dense, position-based):
    r_t = (s_{t+1} + 1) / n_states
    → r ∈ {0.25, 0.50, 0.75, 1.00} for states 0–3.
This gives every (s,a) pair meaningful Q-value signal from the first visit.

Figure layout  (analog of Figure 1 in the paper):
    [0,0] Classical SARSA      — Q-value evolution for all 8 (s,a) pairs
    [0,1] PyTorch Transformer  — same 8 Q-value curves (must overlap exactly)
    [1,0] Overlay              — SARSA solid, Transformer dashed, top-4 pairs
    [1,1] Frobenius norm error — ||Q_TF − Q_SARSA||_F over time (should be ≈ 0)

Usage:
    python compare_sarsa_transformer.py [--T 300] [--seed 42] [--save_dir ../figures]
"""

import sys
import os
import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tabular_sarsa import TabularSARSA, make_chain_mdp
from transformer_handwired_sarsa import SARSATransformer, SARSATokenConfig


# ---------------------------------------------------------------------------
# Dense-reward trajectory generator
# ---------------------------------------------------------------------------

def generate_trajectory_dense(
    P: np.ndarray,
    n_states: int,
    n_actions: int,
    T: int = 300,
    alpha: float = 0.1,
    gamma: float = 0.9,
    epsilon: float = 0.2,
    seed: int = 42,
) -> List[Tuple[int, int, float, int, int]]:
    """
    On-policy epsilon-greedy trajectory with a dense position-based reward.

    Reward:  r_t = (s_{t+1} + 1) / n_states
    This ensures every transition carries positive reward so that all 8 Q-values
    (4 states × 2 actions) converge to meaningfully different non-zero values.

    Args:
        P:         Transition matrix (n_states, n_actions, n_states).
        n_states:  Number of states.
        n_actions: Number of actions.
        T:         Trajectory length.
        alpha:     Learning rate for the internal policy Q-table.
        gamma:     Discount factor.
        epsilon:   Exploration probability.
        seed:      RNG seed.

    Returns:
        List of (s, a, r, s_next, a_next) tuples of length T.
    """
    rng      = np.random.default_rng(seed)
    Q_policy = np.zeros((n_states, n_actions))

    def reward_fn(s_next: int) -> float:
        return float(s_next + 1) / n_states   # ∈ {0.25, 0.50, 0.75, 1.00}

    def epsilon_greedy(s: int) -> int:
        if rng.random() < epsilon:
            return int(rng.integers(n_actions))
        return int(np.argmax(Q_policy[s]))

    trajectory: List[Tuple[int, int, float, int, int]] = []
    s = int(rng.integers(n_states))
    a = epsilon_greedy(s)

    for _ in range(T):
        s_next = int(rng.choice(n_states, p=P[s, a]))
        r      = reward_fn(s_next)
        a_next = epsilon_greedy(s_next)

        trajectory.append((s, a, r, s_next, a_next))

        td = r + gamma * Q_policy[s_next, a_next] - Q_policy[s, a]
        Q_policy[s, a] += alpha * td

        s, a = s_next, a_next

    return trajectory


# ---------------------------------------------------------------------------
# Run both algorithms on the same trajectory
# ---------------------------------------------------------------------------

def run_comparison(
    trajectory: List[Tuple[int, int, float, int, int]],
    n_states: int,
    n_actions: int,
    alpha: float,
    gamma: float,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[float], List[float], List[float], List[float]]:
    """
    Feed the trajectory identically to TabularSARSA and SARSATransformer.

    Returns:
        sarsa_q_history:    Q-table snapshots, length T+1.
        tf_q_history:       Q-table snapshots, length T+1.
        sarsa_predictions:  Q(s_t, a_t) output at each step t, length T.
        tf_predictions:     Q(s_t, a_t) output at each step t, length T.
        sarsa_td_errors:    |TD error| at each step, length T.
        tf_td_errors:       |TD error| at each step, length T.
    """
    sarsa = TabularSARSA(n_states, n_actions, alpha=alpha, gamma=gamma)
    cfg   = SARSATokenConfig(n_states=n_states, n_actions=n_actions,
                              alpha=alpha, gamma=gamma)
    tf    = SARSATransformer(cfg)

    sarsa_predictions: List[float] = []
    tf_predictions:    List[float] = []
    sarsa_td_errors:   List[float] = []
    tf_td_errors:      List[float] = []

    for (s, a, r, s_next, a_next) in trajectory:
        # Record Q(s_t, a_t) *before* the update — the current prediction
        sarsa_predictions.append(float(sarsa.Q[s, a]))
        tf_q_before = float(tf.q_history[-1][s, a])
        tf_predictions.append(tf_q_before)

        # Compute TD errors before updating
        sarsa_td_errors.append(abs(
            r + gamma * sarsa.Q[s_next, a_next] - sarsa.Q[s, a]
        ))
        tf_q_next_before = float(tf.q_history[-1][s_next, a_next])
        tf_td_errors.append(abs(r + gamma * tf_q_next_before - tf_q_before))

        sarsa.step(s, a, r, s_next, a_next)
        tf.step(s, a, r, s_next, a_next)

    return (sarsa.get_q_history(), tf.get_q_history(),
            sarsa_predictions, tf_predictions,
            sarsa_td_errors, tf_td_errors)


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _plot_q_evolution(
    ax: plt.Axes,
    q_history: List[np.ndarray],
    n_states: int,
    n_actions: int,
    title: str,
    colors,
    linestyle: str = "-",
) -> None:
    """Plot Q(s,a) curves for all state-action pairs."""
    q_arr = np.array(q_history)   # (T+1, n_states, n_actions)
    steps = np.arange(len(q_history))
    idx   = 0
    for s in range(n_states):
        for a in range(n_actions):
            ax.plot(steps, q_arr[:, s, a],
                    color=colors[idx % len(colors)],
                    linestyle=linestyle,
                    linewidth=1.5,
                    label=f"Q(s{s},a{a})")
            idx += 1
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel("Step")
    ax.set_ylabel("Q-value")
    ax.legend(fontsize=7, ncol=2, loc="lower right")
    ax.grid(True, alpha=0.3)


def _plot_prediction_comparison(
    ax: plt.Axes,
    sarsa_predictions: List[float],
    tf_predictions: List[float],
) -> None:
    """
    Bottom-left: Q(s_t, a_t) at each step — analog of 'Prediction Comparison'
    in Figure 1.  The trace is noisy because (s_t, a_t) changes every round.
    """
    steps = np.arange(len(sarsa_predictions))
    ax.plot(steps, sarsa_predictions,
            color="steelblue", linewidth=1.0, alpha=0.85,
            label="Classical SARSA")
    ax.plot(steps, tf_predictions,
            color="darkorange", linewidth=1.0, alpha=0.85,
            linestyle="--", label="PyTorch Transformer")
    ax.set_title("Prediction Comparison", fontsize=11, fontweight="bold")
    ax.set_xlabel("Round")
    ax.set_ylabel("Q(s_t, a_t)")
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.3)


def _plot_cumulative_loss(
    ax: plt.Axes,
    sarsa_td_errors: List[float],
    tf_td_errors: List[float],
) -> None:
    """
    Bottom-right: cumulative |TD error| — analog of 'Loss Comparison' in Figure 1.
    Both curves should be indistinguishable, confirming the construction is exact.
    """
    sarsa_cum = np.cumsum(sarsa_td_errors)
    tf_cum    = np.cumsum(tf_td_errors)
    steps     = np.arange(1, len(sarsa_cum) + 1)
    ax.plot(steps, sarsa_cum,
            color="steelblue", linewidth=1.6,
            label="Classical SARSA (cumulative)")
    ax.plot(steps, tf_cum,
            color="darkorange", linewidth=1.6, linestyle="--",
            label="PyTorch Transformer (cumulative)")
    ax.set_title("Loss Comparison", fontsize=11, fontweight="bold")
    ax.set_xlabel("Round")
    ax.set_ylabel("Cumulative |TD Error|")
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.3)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare Classical SARSA vs PyTorch Transformer SARSA")
    parser.add_argument("--T",         type=int,   default=500)
    parser.add_argument("--n_states",  type=int,   default=4)
    parser.add_argument("--n_actions", type=int,   default=2)
    parser.add_argument("--alpha",     type=float, default=0.1)
    parser.add_argument("--gamma",     type=float, default=0.9)
    parser.add_argument("--epsilon",   type=float, default=0.3,
                        help="Epsilon-greedy exploration rate")
    parser.add_argument("--seed",      type=int,   default=42)
    parser.add_argument("--save_dir",  type=str,
                        default=os.path.join(
                            os.path.dirname(os.path.abspath(__file__)),
                            "..", "figures"))
    args = parser.parse_args()

    print(f"MDP : {args.n_states} states, {args.n_actions} actions")
    print(f"Reward : r_t = (s_next + 1) / {args.n_states}  "
          f"[dense, in {{0.25, 0.50, 0.75, 1.00}}]")
    print(f"SARSA : alpha={args.alpha}, gamma={args.gamma}")
    print(f"Trajectory : T={args.T}, epsilon={args.epsilon}, seed={args.seed}")

    P = make_chain_mdp(n_states=args.n_states, n_actions=args.n_actions)
    trajectory = generate_trajectory_dense(
        P,
        n_states=args.n_states,
        n_actions=args.n_actions,
        T=args.T,
        alpha=args.alpha,
        gamma=args.gamma,
        epsilon=args.epsilon,
        seed=args.seed,
    )

    print("Running Classical SARSA and PyTorch Transformer SARSA...")
    (sarsa_hist, tf_hist,
     sarsa_preds, tf_preds,
     sarsa_td, tf_td) = run_comparison(
        trajectory,
        n_states=args.n_states,
        n_actions=args.n_actions,
        alpha=args.alpha,
        gamma=args.gamma,
    )

    max_diff = float(np.max(np.abs(
        np.array(tf_hist, dtype=float) - np.array(sarsa_hist, dtype=float)
    )))
    print(f"Max |Q_TF - Q_SARSA| over all steps: {max_diff:.2e}")

    final_q = np.array(sarsa_hist[-1])
    print("Final Q-table (SARSA):")
    print(np.array2string(final_q, precision=4, suppress_small=True,
                          formatter={'float_kind': '{:.4f}'.format}))

    # ------------------------------------------------------------------
    # Figure (analog of Figure 1 in the paper)
    # ------------------------------------------------------------------
    colors = plt.cm.tab10.colors
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "Tabular SARSA vs PyTorch Handwired Transformer SARSA\n"
        f"4-state chain MDP  |  dense reward r=(s'+1)/{args.n_states}  |  "
        f"alpha={args.alpha},  gamma={args.gamma},  T={args.T}",
        fontsize=12,
    )

    _plot_q_evolution(
        axes[0, 0], sarsa_hist,
        args.n_states, args.n_actions,
        "Classical SARSA -- Q-value Evolution",
        colors,
    )
    _plot_q_evolution(
        axes[0, 1], tf_hist,
        args.n_states, args.n_actions,
        "PyTorch Transformer -- Q-value Evolution",
        colors,
    )
    _plot_prediction_comparison(axes[1, 0], sarsa_preds, tf_preds)
    _plot_cumulative_loss(axes[1, 1], sarsa_td, tf_td)

    plt.tight_layout()

    save_path = os.path.join(args.save_dir, "sarsa_transformer_comparison.png")
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Figure saved -> {save_path}")
    plt.close()


if __name__ == "__main__":
    main()
