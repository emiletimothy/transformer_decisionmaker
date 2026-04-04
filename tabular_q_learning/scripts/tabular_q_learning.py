"""
Tabular Q-Learning: Online Off-Policy TD Control (Algorithm 3 from the paper).

Implements the epsilon-greedy Q-learning algorithm where the Q-table is
initialized to zero and updated via the off-policy temporal difference rule:

    Q(s_t, a_t) <- Q(s_t, a_t) + alpha * (r_t + gamma * max_a Q(s_{t+1}, a) - Q(s_t, a_t))

All other (s, a) entries remain unchanged each step, exactly matching the
construction validated by the transformer in transformer_handwired_q_learning.py.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class TabularQLearning:
    """
    Tabular Q-Learning algorithm for finite MDPs.

    Maintains a Q-table of shape (n_states, n_actions) initialized to zero.
    Accepts a pre-generated trajectory one step at a time so that it can be
    run in lock-step with the transformer construction for comparison.
    """

    def __init__(self, n_states: int, n_actions: int,
                 alpha: float = 0.1, gamma: float = 0.9) -> None:
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma

        self.Q: np.ndarray = np.zeros((n_states, n_actions))
        self.q_history: List[np.ndarray] = [self.Q.copy()]
        self.step_count: int = 0

    def step(self, s: int, a: int, r: float, s_next: int) -> float:
        """
        Apply the Q-learning update for one transition.

        Args:
            s:      Current state index.
            a:      Current action index.
            r:      Observed scalar reward.
            s_next: Next state index.

        Returns:
            The updated Q(s, a) value.
        """
        td_target = r + self.gamma * np.max(self.Q[s_next])
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
        return self.q_history

    def reset(self) -> None:
        self.Q = np.zeros((self.n_states, self.n_actions))
        self.q_history = [self.Q.copy()]
        self.step_count = 0

    def plot_q_values(self, title: str = "Q-value Evolution",
                      save_path: Optional[str] = None) -> plt.Figure:
        fig, ax = plt.subplots(figsize=(10, 6))
        q_arr = np.array(self.q_history)
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
# MDP construction
# ---------------------------------------------------------------------------

def make_chain_mdp(n_states: int = 4, n_actions: int = 2) -> np.ndarray:
    """
    Build a stochastic chain MDP transition matrix.

    Action (n_actions-1) tries to move forward; action 0 tries to move backward.
    With probability 0.8 the intended move succeeds; with probability 0.2 the
    agent stays in place.

    Returns:
        P: np.ndarray of shape (n_states, n_actions, n_states) where
           P[s, a, s'] = Pr(s' | s, a).
    """
    P = np.zeros((n_states, n_actions, n_states))

    for s in range(n_states):
        for a in range(n_actions):
            # Map action to intended direction: a=0 backward, a=n_actions-1 forward
            # Linearly interpolate for actions in between
            direction = (2 * a / (n_actions - 1) - 1) if n_actions > 1 else 0
            if direction <= 0:
                s_intended = max(s - 1, 0)
            else:
                s_intended = min(s + 1, n_states - 1)
            P[s, a, s_intended] += 0.8
            P[s, a, s] += 0.2

    return P


def generate_trajectory(
    P: np.ndarray,
    n_states: int,
    n_actions: int,
    T: int = 500,
    alpha: float = 0.1,
    gamma: float = 0.9,
    epsilon: float = 0.3,
    seed: int = 42,
) -> List[Tuple[int, int, float, int]]:
    """
    Generate an epsilon-greedy Q-learning trajectory.

    A separate, temporary Q-table is maintained purely for the epsilon-greedy
    policy; it is NOT the final Q-learner.  The returned trajectory is
    fed identically to both TabularQLearning and the transformer.

    Returns:
        List of (s, a, r, s_next) tuples of length T.
    """
    rng = np.random.default_rng(seed)
    Q_policy = np.zeros((n_states, n_actions))

    def reward_fn(s_next: int) -> float:
        return float(s_next + 1) / n_states

    def epsilon_greedy(s: int) -> int:
        if rng.random() < epsilon:
            return int(rng.integers(n_actions))
        return int(np.argmax(Q_policy[s]))

    trajectory: List[Tuple[int, int, float, int]] = []
    s = int(rng.integers(n_states))

    for _ in range(T):
        a = epsilon_greedy(s)
        s_next = int(rng.choice(n_states, p=P[s, a]))
        r = reward_fn(s_next)

        trajectory.append((s, a, r, s_next))

        # Update policy Q-table (not the evaluation learner)
        td = r + gamma * np.max(Q_policy[s_next]) - Q_policy[s, a]
        Q_policy[s, a] += alpha * td

        s = s_next

    return trajectory


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    n_states, n_actions = 4, 2
    P = make_chain_mdp(n_states=n_states, n_actions=n_actions)
    traj = generate_trajectory(P, n_states=n_states, n_actions=n_actions, T=200)

    ql = TabularQLearning(n_states=n_states, n_actions=n_actions,
                          alpha=0.1, gamma=0.9)
    for (s, a, r, s_next) in traj:
        ql.step(s, a, r, s_next)

    print("Final Q-table:")
    print(ql.Q.round(4))
