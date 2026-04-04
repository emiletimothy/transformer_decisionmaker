#!/usr/bin/env python3
"""
Generate Q-Learning training dataset and save to a JSON file.

Produces sequences of (states, actions, rewards, next_states, true_Q_tables)
by running epsilon-greedy Tabular Q-Learning on randomly generated MDPs.
"""

import os
import json
import argparse
import numpy as np


def generate_random_mdp(n_states: int, n_actions: int):
    """Generate random transition probabilities and rewards for an MDP."""
    # Transitions: P(s' | s, a) -> shape: (n_states, n_actions, n_states)
    P = np.random.rand(n_states, n_actions, n_states)
    P = P / P.sum(axis=2, keepdims=True)  # Normalize to probabilities
    
    # Rewards: R(s, a) in [0, 1]
    R = np.random.rand(n_states, n_actions)
    
    return P, R


def generate_single_sequence(n_states: int, n_actions: int, n_steps: int, 
                             alpha: float, gamma: float, epsilon: float):
    """Generate one Q-learning sequence on a random MDP."""
    P, R = generate_random_mdp(n_states, n_actions)
    
    # Initialize Q-table
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
            # Break ties randomly to avoid early bias
            max_q = np.max(Q[s])
            best_actions = np.where(Q[s] == max_q)[0]
            a = np.random.choice(best_actions)
            
        # Environment step
        r = R[s, a]
        s_next = np.random.choice(n_states, p=P[s, a])
        
        # Q-learning algorithm update
        max_q_next = np.max(Q[s_next])
        Q[s, a] = (1 - alpha) * Q[s, a] + alpha * (r + gamma * max_q_next)
        
        # Record trajectory
        states.append(int(s))
        actions.append(int(a))
        # Round reward to keep JSON size reasonable
        rewards.append(round(float(r), 4))
        next_states.append(int(s_next))
        
        # Save a snapshot of the Q-table to supervise/evaluate the model
        q_values.append(np.round(Q, 4).tolist())
        
        s = s_next
        
    return {
        'states': states,
        'actions': actions,
        'rewards': rewards,
        'next_states': next_states,
        'q_values': q_values,  # Shape: (n_steps, n_states, n_actions)
        # Store the MDP if you ever need to calculate optimal policies downstream
        'mdp': {
            'P': np.round(P, 4).tolist(),
            'R': np.round(R, 4).tolist()
        },
        'params': {
            'alpha': alpha,
            'gamma': gamma,
            'epsilon': epsilon
        }
    }


def generate_qlearning_data(n_sequences: int, max_steps: int, n_states: int, n_actions: int):
    """Generate a batch of Q-learning trajectories with randomized hyperparameters."""
    sequences = []
    for _ in range(n_sequences):
        # Allow varying sequence lengths 
        n_steps = np.random.randint(5, max_steps + 1)
        
        # Randomize hyperparameters to force the model to learn the general algorithm
        alpha = float(np.random.uniform(0.05, 0.5))
        gamma = float(np.random.uniform(0.8, 0.99))
        epsilon = float(np.random.uniform(0.1, 0.5))
        
        seq = generate_single_sequence(n_states, n_actions, n_steps, alpha, gamma, epsilon)
        sequences.append(seq)
    return sequences


def main():
    parser = argparse.ArgumentParser(description='Generate Q-learning training dataset')
    parser.add_argument('--n_train', type=int, default=80000, help='Number of training sequences')
    parser.add_argument('--n_val', type=int, default=10000, help='Number of validation sequences')
    parser.add_argument('--n_test', type=int, default=10000, help='Number of test sequences')
    parser.add_argument('--max_steps', type=int, default=50, help='Maximum sequence length')
    parser.add_argument('--n_states', type=int, default=4, help='Number of states in MDP')
    parser.add_argument('--n_actions', type=int, default=2, help='Number of actions in MDP')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--output', type=str, default='./qlearning_dataset.json',
                        help='Output file path')
    args = parser.parse_args()

    np.random.seed(args.seed)

    print(f"Generating {args.n_train} train, {args.n_val} val, {args.n_test} test sequences "
          f"(max_steps={args.max_steps}, states={args.n_states}, actions={args.n_actions}) ...")

    train = generate_qlearning_data(args.n_train, args.max_steps, args.n_states, args.n_actions)
    val   = generate_qlearning_data(args.n_val,   args.max_steps, args.n_states, args.n_actions)
    test  = generate_qlearning_data(args.n_test,  args.max_steps, args.n_states, args.n_actions)

    dataset = {
        'config': {
            'n_train': args.n_train,
            'n_val': args.n_val,
            'n_test': args.n_test,
            'max_steps': args.max_steps,
            'n_states': args.n_states,
            'n_actions': args.n_actions,
            'seed': args.seed,
        },
        'train': train,
        'val': val,
        'test': test,
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    with open(args.output, 'w') as f:
        json.dump(dataset, f)

    size_mb = os.path.getsize(args.output) / (1024 * 1024)
    print(f"Dataset saved to {args.output}  ({size_mb:.1f} MB)")


if __name__ == '__main__':
    main()