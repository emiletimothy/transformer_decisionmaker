"""
Compare Classical Q-Learning vs PyTorch Handwired Transformer Q-Learning (Figure 3 Analog).

Generates an n-state / n-action chain MDP with a dense position-based reward
signal so that all Q(s,a) pairs converge to clearly distinct, positive values.
Both algorithms are fed the *identical* trajectory and must agree exactly
(Frobenius error ~ 0).

Reward function (dense, position-based):
    r_t = (s_{t+1} + 1) / n_states

Figure layout (analog of Figure 3 in the paper):
    [0,0] Classical Q-Learning    -- Q-value evolution for all (s,a) pairs
    [0,1] PyTorch Transformer     -- same Q-value curves (must overlap exactly)
    [1,0] Overlay                 -- Classical solid, Transformer dashed
    [1,1] Frobenius norm error    -- ||Q_TF - Q_Classical||_F over time (should be ~ 0)

Usage:
    python compare_q_learning_transformer.py [--T 500] [--n_states 4] [--n_actions 2] [--seed 42]
"""

import sys
import os
import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tabular_q_learning import TabularQLearning, make_chain_mdp, generate_trajectory
from transformer_handwired_q_learning import QLearningTransformer, QLearningConfig


# ---------------------------------------------------------------------------
# Run both algorithms on the same trajectory
# ---------------------------------------------------------------------------

def run_comparison(
    trajectory: List[Tuple[int, int, float, int]],
    n_states: int,
    n_actions: int,
    alpha: float,
    gamma: float,
) -> Tuple[List[np.ndarray], List[np.ndarray],
           List[float], List[float],
           List[float], List[float]]:
    """
    Feed the trajectory identically to TabularQLearning and QLearningTransformer.

    Returns:
        ql_q_history:    Q-table snapshots, length T+1.
        tf_q_history:    Q-table snapshots, length T+1.
        ql_predictions:  Q(s_t, a_t) before update at each step t, length T.
        tf_predictions:  Q(s_t, a_t) before update at each step t, length T.
        ql_td_errors:    |TD error| at each step, length T.
        tf_td_errors:    |TD error| at each step, length T.
    """
    ql = TabularQLearning(n_states, n_actions, alpha=alpha, gamma=gamma)
    cfg = QLearningConfig(n_states=n_states, n_actions=n_actions,
                          alpha=alpha, gamma=gamma)
    tf = QLearningTransformer(cfg)

    ql_predictions: List[float] = []
    tf_predictions: List[float] = []
    ql_td_errors: List[float] = []
    tf_td_errors: List[float] = []

    for (s, a, r, s_next) in trajectory:
        # Record Q(s_t, a_t) before update
        ql_predictions.append(float(ql.Q[s, a]))
        tf_q_before = float(tf.q_history[-1][s, a])
        tf_predictions.append(tf_q_before)

        # TD errors before updating
        ql_td_errors.append(abs(
            r + gamma * np.max(ql.Q[s_next]) - ql.Q[s, a]
        ))
        tf_q_table = tf.q_history[-1]
        tf_td_errors.append(abs(
            r + gamma * np.max(tf_q_table[s_next]) - tf_q_table[s, a]
        ))

        ql.step(s, a, r, s_next)
        tf.step(s, a, r, s_next)

    return (ql.get_q_history(), tf.get_q_history(),
            ql_predictions, tf_predictions,
            ql_td_errors, tf_td_errors)


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
    q_arr = np.array(q_history)
    steps = np.arange(len(q_history))
    idx = 0
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
    ql_predictions: List[float],
    tf_predictions: List[float],
) -> None:
    steps = np.arange(len(ql_predictions))
    ax.plot(steps, ql_predictions,
            color="steelblue", linewidth=1.0, alpha=0.85,
            label="Classical Q-Learning")
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
    ql_td_errors: List[float],
    tf_td_errors: List[float],
) -> None:
    ql_cum = np.cumsum(ql_td_errors)
    tf_cum = np.cumsum(tf_td_errors)
    steps = np.arange(1, len(ql_cum) + 1)
    ax.plot(steps, ql_cum,
            color="steelblue", linewidth=1.6,
            label="Classical Q-Learning (cumulative)")
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
        description="Compare Classical Q-Learning vs PyTorch Transformer Q-Learning")
    parser.add_argument("--T",         type=int,   default=500)
    parser.add_argument("--n_states",  type=int,   default=4)
    parser.add_argument("--n_actions", type=int,   default=2)
    parser.add_argument("--alpha",     type=float, default=0.1)
    parser.add_argument("--gamma",     type=float, default=0.9)
    parser.add_argument("--epsilon",   type=float, default=0.3,
                        help="Epsilon-greedy exploration rate for trajectory")
    parser.add_argument("--seed",      type=int,   default=42)
    parser.add_argument("--save_dir",  type=str,
                        default=os.path.join(
                            os.path.dirname(os.path.abspath(__file__)),
                            "..", "figures"))
    args = parser.parse_args()

    print(f"MDP : {args.n_states} states, {args.n_actions} actions")
    print(f"Reward : r_t = (s_next + 1) / {args.n_states}")
    print(f"Q-Learning : alpha={args.alpha}, gamma={args.gamma}")
    print(f"Trajectory : T={args.T}, epsilon={args.epsilon}, seed={args.seed}")

    P = make_chain_mdp(n_states=args.n_states, n_actions=args.n_actions)
    trajectory = generate_trajectory(
        P,
        n_states=args.n_states,
        n_actions=args.n_actions,
        T=args.T,
        alpha=args.alpha,
        gamma=args.gamma,
        epsilon=args.epsilon,
        seed=args.seed,
    )

    print("Running Classical Q-Learning and PyTorch Transformer Q-Learning...")
    (ql_hist, tf_hist,
     ql_preds, tf_preds,
     ql_td, tf_td) = run_comparison(
        trajectory,
        n_states=args.n_states,
        n_actions=args.n_actions,
        alpha=args.alpha,
        gamma=args.gamma,
    )

    max_diff = float(np.max(np.abs(
        np.array(tf_hist, dtype=float) - np.array(ql_hist, dtype=float)
    )))
    print(f"Max |Q_TF - Q_Classical| over all steps: {max_diff:.2e}")

    final_q = np.array(ql_hist[-1])
    print("Final Q-table (Classical Q-Learning):")
    print(np.array2string(final_q, precision=4, suppress_small=True,
                          formatter={'float_kind': '{:.4f}'.format}))

    # ------------------------------------------------------------------
    # Figure (analog of Figure 3 in the paper)
    # ------------------------------------------------------------------
    colors = plt.cm.tab10.colors
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "Tabular Q-Learning vs PyTorch Handwired Transformer Q-Learning\n"
        f"{args.n_states}-state chain MDP  |  dense reward r=(s'+1)/{args.n_states}  |  "
        f"alpha={args.alpha},  gamma={args.gamma},  T={args.T}",
        fontsize=12,
    )

    _plot_q_evolution(
        axes[0, 0], ql_hist,
        args.n_states, args.n_actions,
        "Classical Q-Learning -- Q-value Evolution",
        colors,
    )
    _plot_q_evolution(
        axes[0, 1], tf_hist,
        args.n_states, args.n_actions,
        "PyTorch Transformer -- Q-value Evolution",
        colors,
    )
    _plot_prediction_comparison(axes[1, 0], ql_preds, tf_preds)
    _plot_cumulative_loss(axes[1, 1], ql_td, tf_td)

    plt.tight_layout()

    save_path = os.path.join(args.save_dir, "q_learning_transformer_comparison.png")
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Figure saved -> {save_path}")
    plt.close()


if __name__ == "__main__":
    main()
