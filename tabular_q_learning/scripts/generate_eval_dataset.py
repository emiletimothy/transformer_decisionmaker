#!/usr/bin/env python3
"""
Generate eval datasets for all Q-learning test scenarios and save to data/eval/.

Scenarios:
  - Structural: in_distribution, sparse_reward, dense_reward, deterministic, stochastic_heavy
  - OOD sequence lengths: 5, 10, 20, 30, 50, 100
  - OOD alpha: 0.01, 0.05, 0.1, 0.3, 0.5, 0.9
  - OOD gamma: 0.5, 0.8, 0.9, 0.95, 0.99
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import json
import argparse


def generate_random_mdp(n_states, n_actions):
    """Generate random transition probabilities and rewards."""
    P = np.random.rand(n_states, n_actions, n_states)
    P = P / P.sum(axis=2, keepdims=True)
    R = np.random.rand(n_states, n_actions)
    return P, R


def generate_sparse_reward_mdp(n_states, n_actions):
    """Only the terminal state gives nonzero reward."""
    P = np.random.rand(n_states, n_actions, n_states)
    P = P / P.sum(axis=2, keepdims=True)
    R = np.zeros((n_states, n_actions))
    R[n_states - 1, :] = 1.0
    return P, R


def generate_dense_reward_mdp(n_states, n_actions):
    """All state-action pairs give meaningful reward."""
    P = np.random.rand(n_states, n_actions, n_states)
    P = P / P.sum(axis=2, keepdims=True)
    R = np.random.uniform(0.3, 1.0, (n_states, n_actions))
    return P, R


def generate_deterministic_mdp(n_states, n_actions):
    """Deterministic transitions (one-hot)."""
    P = np.zeros((n_states, n_actions, n_states))
    for s in range(n_states):
        for a in range(n_actions):
            s_next = np.random.randint(n_states)
            P[s, a, s_next] = 1.0
    R = np.random.rand(n_states, n_actions)
    return P, R


def generate_stochastic_heavy_mdp(n_states, n_actions):
    """Highly stochastic transitions (nearly uniform)."""
    P = np.ones((n_states, n_actions, n_states)) + np.random.rand(n_states, n_actions, n_states) * 0.1
    P = P / P.sum(axis=2, keepdims=True)
    R = np.random.rand(n_states, n_actions)
    return P, R


MDP_GENERATORS = {
    'in_distribution': generate_random_mdp,
    'sparse_reward': generate_sparse_reward_mdp,
    'dense_reward': generate_dense_reward_mdp,
    'deterministic': generate_deterministic_mdp,
    'stochastic_heavy': generate_stochastic_heavy_mdp,
}


def generate_single_sequence(P, R, n_states, n_actions, n_steps, alpha, gamma, epsilon):
    """Generate one Q-learning sequence on a given MDP."""
    Q = np.zeros((n_states, n_actions))
    s = np.random.randint(n_states)

    states = []
    actions = []
    rewards = []
    next_states = []
    q_values = []

    for _ in range(n_steps):
        # Epsilon-greedy action selection
        if np.random.rand() < epsilon:
            a = np.random.randint(n_actions)
        else:
            max_q = np.max(Q[s])
            best_actions = np.where(Q[s] == max_q)[0]
            a = np.random.choice(best_actions)

        # Environment step
        r = R[s, a]
        s_next = np.random.choice(n_states, p=P[s, a])

        # Q-learning update
        max_q_next = np.max(Q[s_next])
        Q[s, a] = (1 - alpha) * Q[s, a] + alpha * (r + gamma * max_q_next)

        states.append(int(s))
        actions.append(int(a))
        rewards.append(round(float(r), 4))
        next_states.append(int(s_next))
        q_values.append(np.round(Q, 4).tolist())

        s = s_next

    return {
        'states': states,
        'actions': actions,
        'rewards': rewards,
        'next_states': next_states,
        'q_values': q_values,
        'params': {
            'alpha': round(alpha, 4),
            'gamma': round(gamma, 4),
            'epsilon': round(epsilon, 4),
        }
    }


def generate_scenario(name, n_steps, n_states, n_actions, alpha, gamma, epsilon,
                      n_sequences, seed=None):
    """Generate test sequences for a named scenario."""
    if seed is not None:
        np.random.seed(seed)

    mdp_gen = MDP_GENERATORS.get(name, generate_random_mdp)
    sequences = []

    for _ in range(n_sequences):
        P, R = mdp_gen(n_states, n_actions)
        seq = generate_single_sequence(P, R, n_states, n_actions, n_steps,
                                       alpha, gamma, epsilon)
        seq['mdp'] = {
            'P': np.round(P, 4).tolist(),
            'R': np.round(R, 4).tolist(),
        }
        sequences.append(seq)

    return sequences


def main():
    parser = argparse.ArgumentParser(description='Generate Q-learning eval datasets')
    parser.add_argument('--n_sequences', type=int, default=50, help='Sequences per scenario')
    parser.add_argument('--n_states', type=int, default=4)
    parser.add_argument('--n_actions', type=int, default=2)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--output_dir', type=str, default='../data/eval')
    args = parser.parse_args()

    out_dir = os.path.join(os.path.dirname(__file__), args.output_dir)
    os.makedirs(out_dir, exist_ok=True)

    N = args.n_sequences
    ns = args.n_states
    na = args.n_actions

    # -- Structural scenarios --
    structural_params = {
        'in_distribution':  dict(n_steps=30, alpha=0.1, gamma=0.9, epsilon=0.3),
        'sparse_reward':    dict(n_steps=30, alpha=0.1, gamma=0.9, epsilon=0.3),
        'dense_reward':     dict(n_steps=30, alpha=0.1, gamma=0.9, epsilon=0.3),
        'deterministic':    dict(n_steps=30, alpha=0.1, gamma=0.9, epsilon=0.3),
        'stochastic_heavy': dict(n_steps=30, alpha=0.1, gamma=0.9, epsilon=0.3),
    }

    structural_data = {}
    for name, params in structural_params.items():
        seqs = generate_scenario(name, n_states=ns, n_actions=na,
                                 n_sequences=N, seed=args.seed, **params)
        structural_data[name] = seqs
        print(f"  structural/{name}: {len(seqs)} sequences, {params['n_steps']} steps")

    # -- OOD: sequence lengths --
    ood_lengths = {
        'steps_5':   5,
        'steps_10':  10,
        'steps_20':  20,
        'steps_30':  30,
        'steps_50':  50,
        'steps_100': 100,
    }

    ood_len_data = {}
    for label, n_steps in ood_lengths.items():
        seqs = generate_scenario('in_distribution', n_steps=n_steps,
                                 n_states=ns, n_actions=na,
                                 alpha=0.1, gamma=0.9, epsilon=0.3,
                                 n_sequences=N, seed=args.seed)
        ood_len_data[label] = seqs
        print(f"  ood_lengths/{label}: {len(seqs)} sequences")

    # -- OOD: alpha --
    ood_alphas = {
        'alpha_0.01': 0.01,
        'alpha_0.05': 0.05,
        'alpha_0.1':  0.10,
        'alpha_0.3':  0.30,
        'alpha_0.5':  0.50,
        'alpha_0.9':  0.90,
    }

    ood_alpha_data = {}
    for label, alpha in ood_alphas.items():
        seqs = generate_scenario('in_distribution', n_steps=30,
                                 n_states=ns, n_actions=na,
                                 alpha=alpha, gamma=0.9, epsilon=0.3,
                                 n_sequences=N, seed=args.seed)
        ood_alpha_data[label] = seqs
        print(f"  ood_alpha/{label}: {len(seqs)} sequences")

    # -- OOD: gamma --
    ood_gammas = {
        'gamma_0.5':  0.50,
        'gamma_0.8':  0.80,
        'gamma_0.9':  0.90,
        'gamma_0.95': 0.95,
        'gamma_0.99': 0.99,
    }

    ood_gamma_data = {}
    for label, gamma in ood_gammas.items():
        seqs = generate_scenario('in_distribution', n_steps=30,
                                 n_states=ns, n_actions=na,
                                 alpha=0.1, gamma=gamma, epsilon=0.3,
                                 n_sequences=N, seed=args.seed)
        ood_gamma_data[label] = seqs
        print(f"  ood_gamma/{label}: {len(seqs)} sequences")

    # -- Save --
    dataset = {
        'config': {
            'n_sequences': N,
            'n_states': ns,
            'n_actions': na,
            'seed': args.seed,
        },
        'structural': structural_data,
        'ood_lengths': ood_len_data,
        'ood_alpha': ood_alpha_data,
        'ood_gamma': ood_gamma_data,
    }

    out_path = os.path.join(out_dir, 'eval_dataset.json')
    with open(out_path, 'w') as f:
        json.dump(dataset, f)

    size_mb = os.path.getsize(out_path) / (1024 * 1024)
    print(f"\nEval dataset saved to {out_path}  ({size_mb:.1f} MB)")


if __name__ == '__main__':
    main()
