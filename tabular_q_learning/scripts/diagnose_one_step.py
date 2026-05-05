"""
Diagnose ONE Q-learning transformer step against classical Q-learning.
"""
import numpy as np
import torch

from tabular_q_learning import TabularQLearning
from transformer_handwired_q_learning import ContinuousThoughtQTransformer, QLearningConfig
from compare_q_learning_transformer import build_paper_sequence


def main():
    n_states, n_actions = 4, 2
    alpha, gamma = 0.1, 0.9

    # Use a non-trivial existing Q-table so updates differ across actions
    rng = np.random.default_rng(0)
    Q = rng.uniform(0, 1, size=(n_states, n_actions)).astype(np.float64)

    s, a, r, s_next = 1, 0, 0.5, 2

    # Classical update
    ql = TabularQLearning(n_states, n_actions, alpha=alpha, gamma=gamma)
    ql.Q = Q.copy()
    ql.step(s, a, r, s_next)
    Q_new_classical = ql.Q.copy()

    # Transformer
    cfg = QLearningConfig(n_states=n_states, n_actions=n_actions,
                          alpha=alpha, gamma=gamma)
    tf = ContinuousThoughtQTransformer(cfg)
    tf.eval()

    seq = build_paper_sequence(torch.tensor(Q, dtype=torch.float32),
                               s, a, r, s_next, tf)
    with torch.no_grad():
        out = tf(seq, log=True)

    update_tok = out[0, -1]
    buf1 = update_tok[tf.d_TE:2*tf.d_TE]
    Q_new_tf = Q.copy()
    for s_idx in range(n_states):
        Q_new_tf[s_idx, a] = float(buf1[tf.S[s_idx]])

    print("\n===== RESULTS =====")
    print("Q before:\n", Q.round(4))
    print("Classical Q after:\n", Q_new_classical.round(4))
    print("Transformer Q after:\n", Q_new_tf.round(4))
    print("Diff (classical - transformer):\n",
          (Q_new_classical - Q_new_tf).round(4))

    # Specifically inspect a_star pick (should be argmax over s_next row of Q)
    print("\nargmax Q[s_next] =", int(np.argmax(Q[s_next])),
          "  expected a_star fed to seq:",
          int(np.argmax(Q[s_next])))


if __name__ == "__main__":
    main()
