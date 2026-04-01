"""
Tabular SARSA: Online On-Policy TD Control.

Implements Algorithm 3 from the paper (Online Tabular SARSA) as a reference
classical algorithm.  The Q-table is initialized to zero and updated via the
one-step temporal-difference rule:

    Q(s_t, a_t) <- Q(s_t, a_t) + alpha * (r_t + gamma * Q(s_{t+1}, a_{t+1}) - Q(s_t, a_t))

All other (s, a) entries remain unchanged each step, exactly matching the
construction validated by the transformer in transformer_handwired_sarsa.py.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class TabularSARSA:
    """
    Tabular SARSA algorithm for finite MDPs.

    Maintains a Q-table of shape (n_states, n_actions) initialized to zero.
    Accepts a pre-generated trajectory one step at a time so that it can be
    run in lock-step with the numpy transformer construction for comparison.
    """

    def __init__(self, n_states: int, n_actions: int,
                 alpha: float = 0.1, gamma: float = 0.9) -> None:
        """
        Args:
            n_states:  Number of discrete states.
            n_actions: Number of discrete actions.
            alpha:     Learning rate in (0, 1].
            gamma:     Discount factor in (0, 1).
        """
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma

        self.Q: np.ndarray = np.zeros((n_states, n_actions))
        # q_history[0] = initial table; q_history[t] = table after t steps
        self.q_history: List[np.ndarray] = [self.Q.copy()]
        self.step_count: int = 0

    def step(self, s: int, a: int, r: float,
             s_next: int, a_next: int) -> float:
        """
        Apply the SARSA update for one transition.

        Args:
            s:      Current state index.
            a:      Current action index.
            r:      Observed scalar reward.
            s_next: Next state index.
            a_next: Next action chosen from s_next (on-policy).

        Returns:
            The updated Q(s, a) value.
        """
        td_target = r + self.gamma * self.Q[s_next, a_next]
        td_error = td_target - self.Q[s, a]
        self.Q[s, a] += self.alpha * td_error

        self.q_history.append(self.Q.copy())
        self.step_count += 1

        logger.debug(
            "Step %d: Q(%d,%d) <- %.4f  (td_error=%.4f)",
            self.step_count, s, a, self.Q[s, a], td_error,
        )
        return float(self.Q[s, a])

    def get_q_history(self) -> List[np.ndarray]:
        """
        Return the full Q-table history.

        Returns:
            List of Q-tables; index 0 is the initial all-zeros table,
            index t is the table after t calls to step().
        """
        return self.q_history

    def reset(self) -> None:
        """Reset the Q-table and history to initial state."""
        self.Q = np.zeros((self.n_states, self.n_actions))
        self.q_history = [self.Q.copy()]
        self.step_count = 0

    # ------------------------------------------------------------------
    # Convenience plotting (mirrors MultiplicativeWeights interface)
    # ------------------------------------------------------------------

    def plot_q_values(self, title: str = "Q-value Evolution",
                      save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot Q-value trajectories for every (s, a) pair over time.

        Args:
            title:     Figure title.
            save_path: If provided, save the figure to this path.

        Returns:
            The matplotlib Figure object.
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        q_arr = np.array(self.q_history)  # shape (T+1, n_states, n_actions)
        colors = plt.cm.tab10.colors

        color_idx = 0
        for s in range(self.n_states):
            for a in range(self.n_actions):
                ax.plot(q_arr[:, s, a],
                        label=f"Q(s{s}, a{a})",
                        color=colors[color_idx % len(colors)],
                        linewidth=1.5)
                color_idx += 1

        ax.set_xlabel("Step")
        ax.set_ylabel("Q-value")
        ax.set_title(title)
        ax.legend(fontsize=8, ncol=2)
        ax.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        return fig

# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

def make_chain_mdp(n_states: int = 4, n_actions: int = 2,
                   seed: int = 42) -> np.ndarray:
    """
    Build a simple stochastic chain MDP transition matrix.

    Action 1 tries to move the agent forward; action 0 tries to move it
    backward.  With probability 0.8 the intended move succeeds; with
    probability 0.2 the agent stays in place.

    Args:
        n_states:  Number of states (chain length).
        n_actions: Number of actions (must be 2).
        seed:      Unused (transition is deterministic in its definition).

    Returns:
        P: np.ndarray of shape (n_states, n_actions, n_states) where
           P[s, a, s'] = Pr(s' | s, a).
    """
    assert n_actions == 2, "Chain MDP requires exactly 2 actions."
    P = np.zeros((n_states, n_actions, n_states))

    for s in range(n_states):
        # Action 0: try to move backward
        s_back = max(s - 1, 0)
        P[s, 0, s_back] += 0.8
        P[s, 0, s] += 0.2

        # Action 1: try to move forward
        s_fwd = min(s + 1, n_states - 1)
        P[s, 1, s_fwd] += 0.8
        P[s, 1, s] += 0.2

    return P


def generate_trajectory(
    P: np.ndarray,
    n_states: int,
    n_actions: int,
    T: int = 200,
    alpha: float = 0.1,
    gamma: float = 0.9,
    epsilon: float = 0.1,
    seed: int = 42,
) -> List[Tuple[int, int, float, int, int]]:
    """
    Generate an on-policy epsilon-greedy SARSA trajectory.

    A separate, temporary Q-table is maintained purely for the epsilon-greedy
    policy; it is NOT the final SARSA learner.  The returned trajectory is
    fed identically to both TabularSARSA and NumpySARSATransformer.

    Args:
        P:         Transition matrix (n_states, n_actions, n_states).
        n_states:  Number of states.
        n_actions: Number of actions.
        T:         Number of interaction steps.
        alpha:     Learning rate used only for the trajectory policy.
        gamma:     Discount factor used only for the trajectory policy.
        epsilon:   Epsilon for epsilon-greedy exploration.
        seed:      Random seed for reproducibility.

    Returns:
        List of (s, a, r, s_next, a_next) tuples of length T.
    """
    rng = np.random.default_rng(seed)
    Q_policy = np.zeros((n_states, n_actions))

    def reward_fn(s_next: int, a: int) -> float:
        return 1.0 if (s_next == n_states - 1 and a == n_actions - 1) else 0.0

    def epsilon_greedy(s: int) -> int:
        if rng.random() < epsilon:
            return int(rng.integers(n_actions))
        return int(np.argmax(Q_policy[s]))

    trajectory = []
    s = int(rng.integers(n_states))
    a = epsilon_greedy(s)

    for _ in range(T):
        s_next = int(rng.choice(n_states, p=P[s, a]))
        r = reward_fn(s_next, a)
        a_next = epsilon_greedy(s_next)

        trajectory.append((s, a, r, s_next, a_next))

        # Update policy Q-table (not the evaluation learner)
        td = r + gamma * Q_policy[s_next, a_next] - Q_policy[s, a]
        Q_policy[s, a] += alpha * td

        s, a = s_next, a_next

    return trajectory


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)

    P = make_chain_mdp(n_states=4, n_actions=2)
    traj = generate_trajectory(P, n_states=4, n_actions=2, T=200)

    sarsa = TabularSARSA(n_states=4, n_actions=2, alpha=0.1, gamma=0.9)
    for (s, a, r, s_next, a_next) in traj:
        sarsa.step(s, a, r, s_next, a_next)

    print("Final Q-table:")
    print(sarsa.Q.round(4))
