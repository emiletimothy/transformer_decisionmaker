"""
Compare Classical Tabular Q-Learning vs Continuous Thought Transformer.
"""
import argparse
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import paths  # noqa: E402
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt

from tabular_q_learning import TabularQLearning, make_chain_mdp, generate_trajectory
from handwired_q_learning_v1 import ContinuousThoughtQTransformer, QLearningConfig

def build_paper_sequence(Q_table, s_t, a_t, r_t, s_next, tf):
    """Builds the length-(3|A| + 8) sequence exactly as defined in the input structure."""
    cfg = tf.cfg
    seq = []
    
    # --- Context Segment: rows of Q_t ---
    for a in range(cfg.n_actions):
        tok = torch.zeros(tf.d_model)
        tok[tf.A[a]] = 1.0; tok[tf.UPDATE] = 1.0 # id(c_a) = u_a + u_Update (context marker)
        for s in range(cfg.n_states):
            tok[tf.d_TE + tf.S[s]] = Q_table[s, a] # buf1(c_a) = sum Q u_s
        seq.append(tok)
        
    # --- Phase 1: Action Evaluation ---
    def id_tok(idx):
        t = torch.zeros(tf.d_model); t[idx] = 1.0; return t
        
    seq.append(id_tok(tf.QCURR))
    seq.append(id_tok(tf.S[s_t]))
    seq.append(id_tok(tf.A[a_t]))
    
    r_tok = id_tok(tf.R)
    r_tok[tf.d_TE + tf.R] = r_t # buf1(r_t) = r(s,a) u_r
    seq.append(r_tok)
    
    seq.append(id_tok(tf.QNEXT))
    
    # Interleave s_{t+1} and a_i for Phase 1 evaluations
    for a in range(cfg.n_actions):
        seq.append(id_tok(tf.S[s_next]))
        seq.append(id_tok(tf.A[a]))
        
    seq.append(id_tok(tf.SELECT))
    
    # --- Phase 2: Q-table update ---
    a_star = int(np.argmax(Q_table[s_next]))
    seq.append(id_tok(tf.A[a_star]))
    seq.append(id_tok(tf.UPDATE))
    
    return torch.stack(seq).unsqueeze(0)

def run_comparison_v2(trajectory, n_states, n_actions, alpha, gamma, beta, beta_max):
    """Standard-components, flag-free construction (handwired_q_learning):
    trained token layout, residual context write, model's own a*."""
    from handwired_q_learning import HandwiredQv2
    # one instance for every MDP size (max_states = 8, max_actions = 4, as the trained model)
    m = HandwiredQv2(alpha=alpha, gamma=gamma, beta=beta, beta_max=beta_max)
    ql = TabularQLearning(n_states, n_actions, alpha=alpha, gamma=gamma)
    Qs, _ = m.run([tuple(x) for x in trajectory], n_states, n_actions, astar_mode='self')
    ql_h, tf_h = [ql.Q.copy()], [np.zeros((n_states, n_actions))]
    for (s, a, r, s_next), (q_hat, _) in zip(trajectory, Qs):
        ql.step(s, a, r, s_next)
        ql_h.append(ql.Q.copy())
        tf_h.append(q_hat)
    return ql_h, tf_h


def run_comparison(trajectory, n_states, n_actions, alpha, gamma, residual=False):
    ql = TabularQLearning(n_states, n_actions, alpha=alpha, gamma=gamma)
    cfg = QLearningConfig(n_states=n_states, n_actions=n_actions, alpha=alpha, gamma=gamma,
                          residual=residual)
    
    tf_model = ContinuousThoughtQTransformer(cfg)
    tf_model.eval()
    
    tf_q_table = np.zeros((n_states, n_actions))
    ql_h, tf_h = [ql.Q.copy()], [tf_q_table.copy()]
    
    for (s, a, r, s_next) in trajectory:
        # 1. Classical Update
        ql.step(s, a, r, s_next)
        
        # 2. Transformer Update
        seq = build_paper_sequence(tf_q_table, s, a, r, s_next, tf_model)
        with torch.no_grad():
            out = tf_model(seq)
            
        update_tok = out[0, -1] # Last token is the Update token
        buf1 = update_tok[tf_model.d_TE : 2*tf_model.d_TE]
        
        # overwrite: buf1(Update) is the entire new column Q_{t+1}(., a_t);
        # residual: buf1(Update) is the increment, added to the carried column.
        for s_idx in range(n_states):
            if residual:
                tf_q_table[s_idx, a] += float(buf1[tf_model.S[s_idx]])
            else:
                tf_q_table[s_idx, a] = float(buf1[tf_model.S[s_idx]])
            
        ql_h.append(ql.Q.copy())
        tf_h.append(tf_q_table.copy())
        
    return ql_h, tf_h

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--T",         type=int,   default=200)
    parser.add_argument("--n_states",  type=int,   default=4)
    parser.add_argument("--n_actions", type=int,   default=2)
    parser.add_argument("--alpha",     type=float, default=0.1)
    parser.add_argument("--gamma",     type=float, default=0.9)
    parser.add_argument("--epsilon",   type=float, default=0.1)
    parser.add_argument("--seed",      type=int,   default=42)
    parser.add_argument("--save_dir",  type=str,   default=str(paths.FIGURES / "handwired"))
    parser.add_argument("--construction", choices=["v1", "v2"], default="v2",
                        help="v2: standard components only (handwired_q_learning)")
    parser.add_argument("--beta",      type=float, default=1e3, help="v2 routing inverse temperature")
    parser.add_argument("--beta_max",  type=float, default=1e4, help="v2 max-head inverse temperature")
    parser.add_argument("--residual",  action="store_true",
                        help="residual context write: c_a <- c_a + buf1(Update)")
    args = parser.parse_args()

    print(f"Running Exact Math Continuous Thought Q-Learning for T={args.T} steps...")
    
    P = make_chain_mdp(n_states=args.n_states, n_actions=args.n_actions)
    trajectory = generate_trajectory(
        P, n_states=args.n_states, n_actions=args.n_actions,
        T=args.T, alpha=args.alpha, gamma=args.gamma, epsilon=args.epsilon, seed=args.seed,
    )

    if args.construction == "v2":
        ql_h, tf_h = run_comparison_v2(trajectory, args.n_states, args.n_actions, args.alpha,
                                       args.gamma, args.beta, args.beta_max)
    else:
        ql_h, tf_h = run_comparison(trajectory, args.n_states, args.n_actions, args.alpha,
                                    args.gamma, residual=args.residual)
    
    max_err = max(np.linalg.norm(q1 - q2) for q1, q2 in zip(ql_h, tf_h))
    print(f"Max Frobenius error between Classical & Transformer over all steps: {max_err:.2e}")
    if max_err < 1e-5:
        print("✅ SUCCESS: The Mathematical Transformer perfectly matched Tabular Q-Learning!")
    
    if args.save_dir:
        Path(args.save_dir).mkdir(parents=True, exist_ok=True)
        suffix = "" if args.construction == "v2" else ("_v1_residual" if args.residual else "_v1")
        save_path = os.path.join(args.save_dir,
                                 f"q_learning_vs_transformer_{args.n_states}s_{args.n_actions}a{suffix}.png")
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        ql_arr, tf_arr = np.array(ql_h), np.array(tf_h)
        diff_arr = tf_arr - ql_arr
        steps = np.arange(len(ql_h))

        for s in range(args.n_states):
            for a in range(args.n_actions):
                axes[0].plot(steps, ql_arr[:, s, a], label=f"Q(s{s},a{a})")
                axes[1].plot(steps, tf_arr[:, s, a])
                axes[2].plot(steps, diff_arr[:, s, a], label=f"Q(s{s},a{a})")

        axes[0].set_title("Classical Tabular Q-Learning")
        axes[1].set_title("Continuous Thought (Mathematical) Transformer"
                          + (" — residual write" if args.residual or args.construction == "v2" else ""))
        axes[2].set_title("Difference (Transformer - Classical)")
        for ax in axes: ax.grid(True)
            
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        print(f"Figure saved to: {save_path}")