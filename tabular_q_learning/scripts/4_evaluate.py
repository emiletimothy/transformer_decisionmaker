#!/usr/bin/env python3
"""
4_evaluate.py — Evaluation & Analysis for Recurrent Context Transformer

Part 1 — Action prediction (ID + OOD)
  Feeds trajectory step-by-step, passing context tokens explicitly.
  At each SELECT position, records argmax action vs. Q-learning greedy.

Part 2 — Context token Q-value probing
  Trains a linear probe on each context token c_a^(t) to decode the
  Q-value vector for action a across all states:
      W · c_a^(t) ≈ [Q(s_1, a), Q(s_2, a), ..., Q(s_{|S|}, a)]

Part 3 — Attention heatmaps
  Verifies SELECT attends to EVAL tokens, UPDATE attends to context + QCURR/QNEXT.

Part 4 — Regret comparison
  Runs the transformer autonomously (closed-loop) on fresh MDPs and compares
  cumulative reward against a greedy Q-learner (ε=0) and an ε-greedy Q-learner.

Part 4d — Nonstationary MDPs
  Same closed-loop rollout, but the MDP is swapped mid-episode (default: 50
  steps, then 100 more) with no signal to the agent. Compares reward before
  and after the switch against an oracle that adapts instantly, an oracle
  frozen on the pre-switch policy, tabular Q-learning, and a random policy.

Part 4e — Reward interventions
  Closed-loop rollouts in which the reward the agent *observes* is corrupted
  (zeroed / constant / resampled from R / sign-flipped) while the reward it is
  *scored on* stays the true environmental reward. Agreement with the tabular
  policy cannot distinguish an online RL update from reward-independent
  state-action statistics; this can, because only an agent that uses reward
  feedback degrades when the feedback is broken.

Part 4f — State-action size sweep
  Action agreement and closed-loop return across the full trained range
  (|S| = 2..max_states, |A| = 2..max_actions), rather than at the largest MDP
  alone.

Part 5 — Effective α/γ recovery
  Fits (α_eff, γ_eff) per trajectory from context-probe Q-value dynamics.

Part 6 — Reward probe from context delta
  Trains a linear probe on the residual Δc_{a_t} = update_hidden - context[a_t]
  to predict the scalar reward r_t at each step. Tests whether the trained
  model preserves the handwired construction's reward-injection pathway
  (Layer 4's R/UPDATE head, which writes α·r into c_{a_t}'s buf₁).

Output files:
    figures/action_agreement.png
    figures/probe_scatter.png
    figures/probe_frobenius.png
    figures/probe_scatter_nobias.png
    figures/probe_frobenius_nobias.png
    figures/attention_heatmap.png
    figures/attention_heatmap.csv
    figures/attention_full_heatmap_4s2a.png
    figures/training_curves.png
    figures/per_state_agreement.png
    figures/regret.png
    figures/long_horizon.png
    figures/nonstationary.png
    figures/nonstationary_summary.csv
    figures/reward_intervention.png
    figures/reward_intervention_summary.csv
    figures/size_sweep.png
    figures/size_sweep_summary.csv
    figures/effective_alpha_gamma.png
    figures/reward_probe.png

`--parts` selects which of the above to run; it defaults to 'all'. The newer
evals can be regenerated on their own with
`--parts reward_intervention,size_sweep`, which skips the probe and attention
parts that dominate runtime and are unaffected by them.
"""

import argparse
import csv
import importlib.util
import math
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import paths  # noqa: E402
from typing import Callable, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.size':         13,
    'axes.titlesize':    15,
    'axes.labelsize':    14,
    'xtick.labelsize':   12,
    'ytick.labelsize':   12,
    'legend.fontsize':   12,
    'figure.titlesize':  16,
})
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Import model and helpers
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

_spec3 = importlib.util.spec_from_file_location(
    "coconut_train",
    os.path.join(_script_dir, "3_train.py")
)
_mod3 = importlib.util.module_from_spec(_spec3)
_spec3.loader.exec_module(_mod3)
build_step_tokens = _mod3.build_step_tokens


# ---------------------------------------------------------------------------
# MDP + tabular Q-learning
# ---------------------------------------------------------------------------

def generate_linear_chain_mdp(
    n_states: int,
    n_actions: int,
) -> Tuple[np.ndarray, np.ndarray, str]:
    """Easy-to-track MDP: deterministic linear chain with rewards increasing
    in state index. Action 0 advances (s -> s+1, wraps at terminal); all other
    actions stay put. Reward depends only on state, growing 0 -> 1.

    The optimal policy is to always choose action 0, so attention should
    consistently favor a0's slot.
    """
    P = np.zeros((n_states, n_actions, n_states), dtype=np.float32)
    for s in range(n_states):
        nxt = min(s + 1, n_states - 1)
        P[s, 0, nxt] = 1.0
        for a in range(1, n_actions):
            P[s, a, s] = 1.0
    state_rewards = np.linspace(0.0, 1.0, n_states, dtype=np.float32)
    R = np.broadcast_to(state_rewards[:, None], (n_states, n_actions)).copy()
    desc = (
        f'Linear {n_states}-state chain, {n_actions} actions: action 0 '
        f'advances s->s+1, others stay; rewards grow linearly 0->1 with '
        f'state index (a0 is optimal everywhere).'
    )
    return P, R, desc


def generate_eval_mdp(
    n_states: int,
    n_actions: int,
    seed: int = 9999,
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    P = rng.dirichlet(alpha=np.ones(n_states), size=(n_states, n_actions)).astype(np.float32)
    R = rng.beta(2.0, 2.0, size=(n_states, n_actions)).astype(np.float32)
    R = np.clip(R, 0.0, 1.0)
    return P, R


def generate_ood_mdp(
    n_states: int,
    n_actions: int,
    variant: str,
    seed: int = 9999,
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    if variant == 'deterministic':
        P = rng.dirichlet(np.full(n_states, 0.001), size=(n_states, n_actions)).astype(np.float32)
        R = rng.beta(2.0, 2.0, size=(n_states, n_actions)).astype(np.float32)
    elif variant == 'sparse_reward':
        P = rng.dirichlet(np.ones(n_states), size=(n_states, n_actions)).astype(np.float32)
        R = rng.beta(0.1, 2.0, size=(n_states, n_actions)).astype(np.float32)
    elif variant == 'dense_reward':
        P = rng.dirichlet(np.ones(n_states), size=(n_states, n_actions)).astype(np.float32)
        R = rng.beta(10.0, 10.0, size=(n_states, n_actions)).astype(np.float32)
    elif variant == 'adversarial':
        P = rng.dirichlet(np.full(n_states, 0.001), size=(n_states, n_actions)).astype(np.float32)
        R = rng.uniform(0.0, 1.0, size=(n_states, n_actions)).astype(np.float32)
    elif variant == 'high_variance':
        P = rng.dirichlet(np.full(n_states, 0.5), size=(n_states, n_actions)).astype(np.float32)
        R = rng.beta(0.1, 0.1, size=(n_states, n_actions)).astype(np.float32)
    else:
        raise ValueError(f"Unknown OOD variant: {variant!r}")
    R = np.clip(R, 0.0, 1.0)
    return P, R


def generate_reward_dist_mdp(
    n_states: int,
    n_actions: int,
    dist_name: str,
    seed: int = 9999,
) -> Tuple[np.ndarray, np.ndarray]:
    """MDPs with a fixed transition prior (Dirichlet 1.0) but varying reward
    distributions. Used to compare cumulative reward across reward shapes
    while holding dynamics structure constant.
    """
    rng = np.random.default_rng(seed)
    P = rng.dirichlet(np.ones(n_states), size=(n_states, n_actions)).astype(np.float32)
    if dist_name == 'beta_2_2':
        R = rng.beta(2.0, 2.0, size=(n_states, n_actions)).astype(np.float32)
    elif dist_name == 'sparse':
        R = rng.beta(0.1, 2.0, size=(n_states, n_actions)).astype(np.float32)
    elif dist_name == 'dense':
        R = rng.beta(10.0, 10.0, size=(n_states, n_actions)).astype(np.float32)
    elif dist_name == 'uniform':
        R = rng.uniform(0.0, 1.0, size=(n_states, n_actions)).astype(np.float32)
    elif dist_name == 'bimodal':
        R = rng.beta(0.1, 0.1, size=(n_states, n_actions)).astype(np.float32)
    elif dist_name == 'bernoulli':
        R = (rng.random(size=(n_states, n_actions)) < 0.5).astype(np.float32)
    else:
        raise ValueError(f"Unknown reward distribution: {dist_name!r}")
    R = np.clip(R, 0.0, 1.0)
    return P, R


REWARD_DISTRIBUTIONS: List[Tuple[str, str]] = [
    ('beta_2_2',  'Beta(2,2) — balanced'),
    ('sparse',    'Beta(0.1,2) — sparse'),
    ('dense',     'Beta(10,10) — dense'),
    ('uniform',   'Uniform(0,1)'),
    ('bimodal',   'Beta(0.1,0.1) — bimodal'),
    ('bernoulli', 'Bernoulli(0.5)'),
]


def generate_contrast_mdp(
    n_states: int,
    n_actions: int,
    rng: np.random.Generator,
    low: float = 0.05,
    high: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """High-contrast MDP: Dirichlet(1) transitions as usual, but exactly one
    action per state pays `high` and every other action pays `low`.

    The standard eval MDPs draw rewards from Beta(2,2), which leaves the mean
    per-state advantage gap around 0.25 — small enough that a uniform random
    policy already collects ~70% of the optimal return, so cumulative reward
    barely separates policies. Here the band is ~1.0 (optimal) vs ~0.3
    (random), which is what makes reward informative about behaviour.

    Takes an explicit `rng` so callers can reproduce a specific draw.
    """
    P = rng.dirichlet(alpha=np.ones(n_states),
                      size=(n_states, n_actions)).astype(np.float32)
    R = np.full((n_states, n_actions), low, dtype=np.float32)
    R[np.arange(n_states), rng.integers(n_actions, size=n_states)] = high
    return P, R


def generate_contrast_mdp_seeded(
    n_states: int,
    n_actions: int,
    seed: int = 9999,
) -> Tuple[np.ndarray, np.ndarray]:
    """`generate_contrast_mdp` behind the `(n_states, n_actions, seed=...)`
    signature every other MDP factory here uses, so it can be dropped into the
    `mdp_fn` hooks (long-horizon eval, reward interventions, size sweep)
    interchangeably with `generate_eval_mdp`.
    """
    return generate_contrast_mdp(n_states, n_actions,
                                 np.random.default_rng(seed))


# MDP families that the reward-intervention and long-horizon evals sweep over.
# The Beta(2,2) family is the one used throughout the paper; on it a uniform
# random policy already collects ~70% of the optimal return, so cumulative
# reward is nearly saturated and large behavioural changes barely move it. The
# contrast family widens the optimal/random band to ~1.0 vs ~0.3, which is what
# makes reward informative about behaviour. Results are reported on both.
MDP_FAMILIES: List[Tuple[str, str, object]] = [
    ('eval',     'Beta(2,2) rewards (paper default)', generate_eval_mdp),
    ('contrast', 'High-contrast rewards (one good action per state)',
     generate_contrast_mdp_seeded),
]


def generate_nonstationary_mdp(
    n_states: int,
    n_actions: int,
    variant: str,
    seed: int = 9999,
) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """Two-phase MDP: the agent runs in (P1, R1), then the world changes to
    (P2, R2) mid-episode without any signal that it happened.

    Phase 1 is always the standard in-distribution eval MDP for `seed`, so the
    pre-switch segment is directly comparable to the stationary evaluation.

    Variants differ in what the switch destroys:
      action_permute      — actions are relabeled (P and R permuted along the
                            action axis). The MDP is isomorphic to phase 1, so
                            the achievable reward band is identical before and
                            after; only the mapping from action index to
                            outcome changes. Any policy that keeps replaying
                            its phase-1 action choices is now wrong.
      reward_resample     — fresh reward matrix, same transitions.
      transition_resample — fresh transitions, same reward matrix.
      full_resample       — an entirely different MDP.
      contrast_permute    — like action_permute, but on a high-contrast MDP:
                            exactly one action per state pays ~1 and the rest
                            pay ~0. On the Beta(2,2) eval MDPs above, a random
                            policy already scores ~70% of optimal, so nothing
                            the agent does after the switch is visible in
                            reward; here the optimal/random band is ~1.0 vs
                            ~0.3, so re-adaptation actually shows up.
    """
    P1, R1 = generate_eval_mdp(n_states, n_actions, seed=seed)
    rng = np.random.default_rng(seed + 777)

    if variant == 'contrast_permute':
        # Phase 1 is *not* the standard eval MDP for this variant.
        P1, R1 = generate_contrast_mdp(n_states, n_actions, rng)
        perm = rng.permutation(n_actions)
        while n_actions > 1 and np.array_equal(perm, np.arange(n_actions)):
            perm = rng.permutation(n_actions)
        P2 = np.ascontiguousarray(P1[:, perm, :])
        R2 = np.ascontiguousarray(R1[:, perm])
    elif variant == 'action_permute':
        perm = rng.permutation(n_actions)
        # A no-op permutation would make the switch invisible.
        while n_actions > 1 and np.array_equal(perm, np.arange(n_actions)):
            perm = rng.permutation(n_actions)
        P2 = np.ascontiguousarray(P1[:, perm, :])
        R2 = np.ascontiguousarray(R1[:, perm])
    elif variant == 'reward_resample':
        P2 = P1.copy()
        R2 = rng.beta(2.0, 2.0, size=(n_states, n_actions)).astype(np.float32)
    elif variant == 'transition_resample':
        P2 = rng.dirichlet(alpha=np.ones(n_states),
                           size=(n_states, n_actions)).astype(np.float32)
        R2 = R1.copy()
    elif variant == 'full_resample':
        P2, R2 = generate_eval_mdp(n_states, n_actions, seed=seed + 31337)
    else:
        raise ValueError(f"Unknown nonstationary variant: {variant!r}")

    R2 = np.clip(R2, 0.0, 1.0).astype(np.float32)
    return (P1, R1), (P2, R2)


NONSTATIONARY_VARIANTS: List[Tuple[str, str]] = [
    ('action_permute',      'Actions relabeled (same task, new controls)'),
    ('reward_resample',     'New rewards, same transitions'),
    ('transition_resample', 'New transitions, same rewards'),
    ('full_resample',       'Entirely new MDP'),
    ('contrast_permute',    'High-contrast rewards, actions relabeled'),
]


OOD_VARIANTS: List[Tuple[str, str]] = [
    ('deterministic', 'Deterministic transitions (Dir 0.001), same rewards'),
    ('sparse_reward', 'Sparse rewards (Beta 0.1,2), same transitions'),
    ('dense_reward',  'Dense rewards (Beta 10,10), same transitions'),
    ('adversarial',   'Deterministic transitions + Uniform rewards'),
    ('high_variance', 'Moderate transitions (Dir 0.5) + extreme bimodal rewards (Beta 0.1,0.1)'),
]


# eps of the eps-greedy transformer rollouts drawn next to the greedy ones (set from --epsilon)
TF_EPS = 0.2


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
    rng = np.random.default_rng(seed + 1)
    Q = np.zeros((n_states, n_actions), dtype=np.float32)
    s = int(rng.integers(n_states))

    trajectory  = []
    q_snapshots = []

    for _ in range(n_steps):
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
            's': s, 'a': a, 'r': r, 's_next': s_next,
            'a_star': a_next, 'a_next': a_next,
        })
        q_snapshots.append(Q.copy())
        s = s_next

    return trajectory, np.stack(q_snapshots, axis=0)


# ---------------------------------------------------------------------------
# Part 1: Action prediction with recurrent context
# ---------------------------------------------------------------------------

def run_action_inference(
    model: COCONUTTransformer,
    trajectory: List[Dict],
    vocab: Dict,
    n_actions: int,
    config: COCONUTConfig,
    device: torch.device,
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """Feed trajectory step-by-step with explicit context passing.

    Returns (predicted_actions, context_history) where context_history[t]
    is the context tensor (n_actions, d_model) after step t.
    """
    max_actions = config.max_actions
    model.eval()

    context = model.get_init_context(1, n_actions, device)
    predicted_actions = []
    context_history = []

    with torch.no_grad():
        for t, tr in enumerate(trajectory):
            token_list, r_off, s_off, u_off = build_step_tokens(tr, vocab, n_actions)
            token_ids = torch.tensor([token_list], dtype=torch.long, device=device)
            reward_val = torch.tensor([tr['r']], dtype=torch.float32, device=device)

            select_logits, update_hidden = model.forward_step(
                token_ids=token_ids,
                reward_value=reward_val,
                reward_offset=r_off,
                select_offset=s_off,
                update_offset=u_off,
                context=context,
            )

            if n_actions < max_actions:
                select_logits[:, n_actions:] = float('-inf')

            pred_a = int(select_logits[0].argmax().item())
            predicted_actions.append(pred_a)

            context_history.append(context[0].cpu().numpy().copy())

            a_t = tr['a']
            new_context = context.clone()
            new_context[0, a_t, :] = model.contextualize(update_hidden)[0]
            context = new_context

    context_history.append(context[0].cpu().numpy().copy())

    return np.array(predicted_actions, dtype=np.int32), context_history


# ---------------------------------------------------------------------------
# Part 2: Context token Q-value probing
# ---------------------------------------------------------------------------

class ContextQProbe(nn.Module):
    """Linear probe: context token c_a -> Q-values for action a across all states.

    Maps c_a^(t) in R^{d_model} to [Q(s_1, a), ..., Q(s_{|S|}, a)] in R^{|S|}.
    """
    def __init__(self, d_model: int, n_states: int, bias: bool = True):
        super().__init__()
        self.linear = nn.Linear(d_model, n_states, bias=bias)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.linear(h)


def collect_context_probe_data(
    model: COCONUTTransformer,
    n_trajectories: int,
    n_states: int,
    n_actions: int,
    vocab: Dict,
    config: COCONUTConfig,
    device: torch.device,
    n_steps: int = 50,
    seed_offset: int = 20000,
) -> Tuple[np.ndarray, np.ndarray, List[List[Dict]]]:
    """Collect context tokens and Q-value targets for probe training.

    Returns
    -------
    ctx_all : (n_trajectories * n_steps * n_actions, d_model)
        ctx_all[i*n_steps*n_actions + t*n_actions + a] = context for action a
        at step t of trajectory i (i.e. the context *before* step t's update).
    q_target_all : (n_trajectories * n_steps * n_actions, n_states)
        q_target_all[...] = Q[:, a] (tabular) before step t's update.
    trajectories : list of n_trajectories lists of transition dicts
    """
    model.eval()
    max_actions = config.max_actions
    d_model = config.d_model

    N = n_trajectories * n_steps * n_actions
    ctx_all = np.empty((N, d_model), dtype=np.float32)
    q_target_all = np.empty((N, n_states), dtype=np.float32)
    all_trajectories: List[List[Dict]] = []
    write_idx = 0

    with torch.no_grad():
        for i in range(n_trajectories):
            seed = seed_offset + i
            P, R = generate_eval_mdp(n_states, n_actions, seed=seed)
            trajectory, q_snapshots = run_tabular_q_learning(
                P, R, n_states, n_actions, n_steps=n_steps, seed=seed
            )

            context = model.get_init_context(1, n_actions, device)

            for t, tr in enumerate(trajectory):
                token_list, r_off, s_off, u_off = build_step_tokens(tr, vocab, n_actions)
                token_ids = torch.tensor([token_list], dtype=torch.long, device=device)
                reward_val = torch.tensor([tr['r']], dtype=torch.float32, device=device)

                _, update_hidden = model.forward_step(
                    token_ids=token_ids,
                    reward_value=reward_val,
                    reward_offset=r_off,
                    select_offset=s_off,
                    update_offset=u_off,
                    context=context,
                )

                ctx_t = context[0, :n_actions, :].detach().cpu().numpy()
                q_t = q_snapshots[t]

                ctx_all[write_idx:write_idx + n_actions] = ctx_t
                q_target_all[write_idx:write_idx + n_actions] = q_t.T
                write_idx += n_actions

                a_t = tr['a']
                new_context = context.clone()
                new_context[0, a_t, :] = model.contextualize(update_hidden)[0]
                context = new_context

                del token_ids, reward_val, update_hidden

            all_trajectories.append(trajectory)

    return ctx_all, q_target_all, all_trajectories


def train_probe(
    probe: ContextQProbe,
    ctx_all: np.ndarray,
    q_all: np.ndarray,
    device: torch.device,
    n_epochs: int = 10,
    batch_size: int = 256,
    lr: float = 1e-3,
) -> List[float]:
    probe.train()
    optimizer = torch.optim.Adam(probe.parameters(), lr=lr)
    N = ctx_all.shape[0]
    losses = []

    ctx_t = torch.tensor(ctx_all, dtype=torch.float32, device=device)
    q_t = torch.tensor(q_all, dtype=torch.float32, device=device)

    for _ in range(n_epochs):
        perm = torch.randperm(N, device=device)
        epoch_loss = 0.0
        n_batches = 0
        for i in range(0, N, batch_size):
            idx = perm[i:i + batch_size]
            pred = probe(ctx_t[idx])
            loss = F.mse_loss(pred, q_t[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        losses.append(epoch_loss / max(n_batches, 1))

    return losses


def evaluate_probe(
    probe: ContextQProbe,
    ctx_all: np.ndarray,
    q_all: np.ndarray,
    device: torch.device,
) -> Tuple[float, float, np.ndarray]:
    probe.eval()
    with torch.no_grad():
        ctx_t = torch.tensor(ctx_all, dtype=torch.float32, device=device)
        q_pred = probe(ctx_t).cpu().numpy()

    q_true_flat = q_all.reshape(-1)
    q_pred_flat = q_pred.reshape(-1)

    ss_res = np.sum((q_true_flat - q_pred_flat) ** 2)
    ss_tot = np.sum((q_true_flat - q_true_flat.mean()) ** 2) + 1e-12
    r2 = float(1.0 - ss_res / ss_tot)

    diff = q_pred - q_all
    frob = np.sqrt((diff ** 2).sum(axis=-1))
    frob_mean = float(frob.mean())

    return r2, frob_mean, q_pred


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def plot_action_agreement(
    aa_mean_id: np.ndarray,
    aa_std_id: np.ndarray,
    ood_results: List[Tuple[str, np.ndarray, np.ndarray]],
    save_path: str,
    n_mdps: int = 10,
    label_suffix: str = '',
) -> None:
    steps = np.arange(1, len(aa_mean_id) + 1)
    ood_colours = ['darkorange', 'crimson', 'forestgreen', 'purple', 'saddlebrown']
    ood_styles = ['--', '-.', ':', (0, (3, 1, 1, 1)), (0, (5, 1))]

    fig, ax = plt.subplots(figsize=(7, 11))
    ax.plot(steps, aa_mean_id, color='steelblue', linewidth=2.5,
            label='In-distribution (ID)')
    ax.fill_between(steps, aa_mean_id - aa_std_id, aa_mean_id + aa_std_id,
                    alpha=0.2, color='steelblue')

    for i, (lbl, mean_ood, std_ood) in enumerate(ood_results):
        colour = ood_colours[i % len(ood_colours)]
        style = ood_styles[i % len(ood_styles)]
        ax.plot(steps, mean_ood, color=colour, linewidth=1.8,
                linestyle=style, label=f'OOD: {lbl}')
        ax.fill_between(steps, mean_ood - std_ood, mean_ood + std_ood,
                        alpha=0.10, color=colour)

    ax.axhline(y=1.0, color='gray', linestyle=':', linewidth=1, alpha=0.5,
               label='Perfect agreement')
    ax.set_xlabel('Timestep', fontsize=11)
    ax.set_ylabel('Greedy Action Agreement', fontsize=10)
    ax.tick_params(axis='x', labelsize=11)
    ax.tick_params(axis='y', labelsize=10)
    ax.set_ylim(-0.05, 1.1)
    suffix = f' ({label_suffix})' if label_suffix else ''
    ax.set_title(
        f'Per-Step Action Agreement: ID vs OOD Variants{suffix}\n'
        f'(mean +/- std, {n_mdps} MDPs each variant)',
        fontsize=12,
    )
    ax.legend(fontsize=9, loc='lower right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_probe_scatter(
    q_true: np.ndarray,
    q_pred: np.ndarray,
    r2: float,
    save_path: str,
) -> None:
    q_t = q_true.reshape(-1)
    q_p = q_pred.reshape(-1)
    rng = np.random.default_rng(0)
    idx = rng.choice(len(q_t), size=min(10000, len(q_t)), replace=False)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(q_t[idx], q_p[idx], alpha=0.35, s=10, color='steelblue',
               rasterized=True, edgecolors='none')
    lo = min(q_t.min(), q_p.min())
    hi = max(q_t.max(), q_p.max())
    ax.plot([lo, hi], [lo, hi], 'r--', linewidth=1.5, label='y = x')
    ax.set_xlim(q_t.min(), q_t.max())
    y_lo, y_hi = q_p.min(), q_p.max()
    y_pad = 0.02 * (y_hi - y_lo)
    ax.set_ylim(y_lo - y_pad, y_hi + y_pad)
    ax.set_xlabel('Q tabular', fontsize=14)
    ax.set_ylabel('Q probe (context token)', fontsize=14)
    ax.tick_params(axis='both', labelsize=13)
    ax.set_title(f'Context Token Probe: Q-Values vs True  (R$^2$ = {r2:.4f})',
                 fontsize=14)
    ax.legend(fontsize=12, loc='upper left')
    ax.grid(True, alpha=0.3)
    plt.subplots_adjust(left=0.10, right=0.98, top=0.94, bottom=0.10)
    plt.savefig(save_path, dpi=150, bbox_inches='tight', pad_inches=0.05)
    plt.close()
    print(f"  Saved: {save_path}")


def plot_probe_frobenius(
    q_pred_all: np.ndarray,
    q_true_all: np.ndarray,
    n_steps: int,
    n_traj: int,
    n_actions: int,
    save_path: str,
) -> None:
    n_per_step = n_actions
    q_pred_r = q_pred_all.reshape(n_traj, n_steps, n_per_step, -1)
    q_true_r = q_true_all.reshape(n_traj, n_steps, n_per_step, -1)

    diff = q_pred_r - q_true_r
    frob = np.sqrt((diff ** 2).sum(axis=(-2, -1)))
    frob_mean = frob.mean(axis=0)
    frob_std = frob.std(axis=0)

    zero_baseline = np.sqrt((q_true_r ** 2).sum(axis=(-2, -1))).mean(axis=0)

    steps = np.arange(1, n_steps + 1)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(steps, frob_mean, color='steelblue', linewidth=2, label='Context probe error')
    ax.fill_between(steps, frob_mean - frob_std, frob_mean + frob_std,
                    alpha=0.25, color='steelblue')
    ax.plot(steps, zero_baseline, color='tomato', linewidth=2, linestyle='--',
            label='Zero baseline')
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Frobenius error')
    ax.set_title(f'Context Token Q-Probe Error Over Time\n'
                 f'(mean +/- std, {n_traj} MDPs)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


# ---------------------------------------------------------------------------
# Attention heatmap
# ---------------------------------------------------------------------------

def get_token_labels(n_ctx: int, token_ids: List[int], vocab: Dict) -> List[str]:
    """Labels for the flat sequence layout: [BOS, ctx_0..ctx_{n_ctx-1}, rest_of_discrete].

    token_ids is the original discrete sequence whose first element is BOS.
    """
    inv = {}
    inv[vocab['TOK_NULL']]   = 'NULL'
    inv[vocab['TOK_START']]  = 'BOS'
    inv[vocab['TOK_R']]      = 'R'
    inv[vocab['TOK_SELECT']] = 'SEL'
    inv[vocab['TOK_UPDATE']] = 'UPD'
    inv[vocab['TOK_QCURR']]  = 'QCUR'
    inv[vocab['TOK_QNEXT']]  = 'QNXT'
    for i, tok in enumerate(vocab['TOK_S']):
        inv[tok] = f'S{i}'
    for i, tok in enumerate(vocab['TOK_A']):
        inv[tok] = f'A{i}'

    bos_label = inv.get(token_ids[0], f'?{token_ids[0]}')
    rest      = [inv.get(t, f'?{t}') for t in token_ids[1:]]
    return [bos_label] + [f'ctx_{i}' for i in range(n_ctx)] + rest


def plot_attention_heatmap(
    model: COCONUTTransformer,
    trajectory: List[Dict],
    vocab: Dict,
    n_actions: int,
    config: COCONUTConfig,
    device: torch.device,
    save_path: str,
) -> None:
    max_actions = config.max_actions
    model.eval()

    tr = trajectory[0]
    token_list, r_off, s_off, u_off = build_step_tokens(tr, vocab, n_actions)
    token_ids = torch.tensor([token_list], dtype=torch.long, device=device)
    reward_val = torch.tensor([tr['r']], dtype=torch.float32, device=device)

    context = model.get_init_context(1, n_actions, device)

    with torch.no_grad():
        select_logits, update_hidden, all_attn = model.forward_step(
            token_ids=token_ids,
            reward_value=reward_val,
            reward_offset=r_off,
            select_offset=s_off,
            update_offset=u_off,
            context=context,
            return_attention=True,
        )

    n_ctx = n_actions
    labels = get_token_labels(n_ctx, token_list, vocab)
    # Flat layout: [BOS, ctx_0..ctx_{n_ctx-1}, rest_of_discrete]; total length = n_ctx + len(token_list).
    T = n_ctx + len(token_list)

    sel_pos = n_ctx + s_off
    upd_pos = n_ctx + u_off

    L = len(all_attn)
    H = all_attn[0].shape[1]

    query_rows = [(sel_pos, 'SELECT'), (upd_pos, 'UPDATE')]
    n_queries = len(query_rows)

    bar_h = 0.6
    col_w = max(5.0, T * 0.12)
    fig, axes = plt.subplots(
        L, H,
        figsize=(H * col_w, L * bar_h * (n_queries + 1.5)),
        squeeze=False,
    )

    cmap = plt.cm.Blues

    for layer_idx in range(L):
        attn_np = all_attn[layer_idx][0].cpu().numpy()  # [H, T, T]
        for head_idx in range(H):
            ax = axes[layer_idx][head_idx]
            data = attn_np[head_idx]

            rows = np.stack([data[pos] for pos, _ in query_rows], axis=0)
            vmax = rows.max() + 1e-9

            ax.imshow(
                rows, aspect='auto', cmap=cmap,
                vmin=0.0, vmax=vmax, origin='upper', interpolation='nearest',
            )

            if layer_idx == L - 1:
                ax.set_xticks(range(T))
                ax.set_xticklabels(labels, rotation=90, fontsize=6)
            else:
                ax.set_xticks([])

            ax.set_yticks(range(n_queries))
            if head_idx == 0:
                ax.set_yticklabels([lbl for _, lbl in query_rows], fontsize=7)
            else:
                ax.set_yticklabels([])

            if layer_idx == 0:
                ax.set_title(f'Head {head_idx}', fontsize=8, pad=3)
            if head_idx == 0:
                ax.set_ylabel(f'L{layer_idx + 1}', fontsize=8)

    fig.suptitle(
        'Attention — query rows: SELECT & UPDATE  |  step 0 (initial context)',
        fontsize=10, y=1.02,
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")

    csv_path = os.path.splitext(save_path)[0] + '.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        header = ['layer', 'head', 'query_token'] + labels
        writer.writerow(header)
        for layer_idx in range(L):
            attn_np = all_attn[layer_idx][0].cpu().numpy()
            for head_idx in range(H):
                data = attn_np[head_idx]
                for pos, lbl in query_rows:
                    row = [layer_idx + 1, head_idx, lbl] + data[pos].tolist()
                    writer.writerow(row)
    print(f"  Saved: {csv_path}")


# ---------------------------------------------------------------------------
# Full causal-triangle attention heatmap (per layer × head, single sequence)
# ---------------------------------------------------------------------------

def plot_full_attention_heatmap(
    model: COCONUTTransformer,
    trajectory: List[Dict],
    vocab: Dict,
    n_actions_actual: int,
    config: COCONUTConfig,
    device: torch.device,
    save_path: str,
    mdp_description: Optional[str] = None,
) -> None:
    """Plot the full T×T causal-attention matrix (triangle pattern), averaged
    over every transition step in the episode. One subplot per (layer, head).

    Each step has the same token-sequence length, so per-step attention
    matrices are directly averageable. Per-step token *identities* (S/A token
    ids) differ across steps, so axis labels reflect the structural slot
    rather than concrete values for any single step.
    """
    max_actions = config.max_actions
    model.eval()
    n_steps = len(trajectory)

    n_actions = n_actions_actual
    sample_tokens, _, _, _ = build_step_tokens(trajectory[0], vocab, n_actions)
    n_ctx = n_actions
    T = n_ctx + len(sample_tokens)

    accum: Optional[List[np.ndarray]] = None
    L: Optional[int] = None
    H: Optional[int] = None

    context = model.get_init_context(1, n_actions, device)

    with torch.no_grad():
        for t_idx, tr in enumerate(trajectory):
            token_list, r_off, s_off, u_off = build_step_tokens(
                tr, vocab, n_actions
            )
            token_ids = torch.tensor([token_list], dtype=torch.long, device=device)
            reward_val = torch.tensor([tr['r']], dtype=torch.float32, device=device)

            _, upd_h, all_attn = model.forward_step(
                token_ids=token_ids,
                reward_value=reward_val,
                reward_offset=r_off,
                select_offset=s_off,
                update_offset=u_off,
                context=context,
                return_attention=True,
            )

            if accum is None:
                L = len(all_attn)
                H = all_attn[0].shape[1]
                accum = [np.zeros((H, T, T), dtype=np.float64) for _ in range(L)]

            for layer_idx in range(L):
                accum[layer_idx] += all_attn[layer_idx][0].cpu().numpy()

            new_ctx = context.clone()
            new_ctx[0, tr['a'], :] = model.contextualize(upd_h)[0]
            context = new_ctx

    avg_attn = [a / max(n_steps, 1) for a in accum]
    # Max across heads: one [T, T] map per layer (preserves specialist heads).
    per_layer = [a.max(axis=0) for a in avg_attn]

    # Structural slot labels (BOS, then context, then per-step discrete tokens)
    slot_labels: List[str] = ['BOS']
    for i in range(n_ctx):
        slot_labels.append(f'ctx_{i}')
    A = n_actions
    slot_labels += ['QCUR', 's_t', 'a_t', 'R', 'QNXT']
    for c in range(A):
        slot_labels += ["s'", f'a{c}']
    slot_labels += ['SEL', 'a*', 'UPD']
    labels = slot_labels

    cell = max(0.18, 6.0 / max(T, 1))
    # One subplot per layer (heads averaged).
    fig, axes = plt.subplots(
        1, L,
        figsize=(L * (T * cell + 1.0), T * cell + 1.0),
        squeeze=False,
    )

    cmap = plt.cm.viridis

    for layer_idx in range(L):
        ax = axes[0][layer_idx]
        data = per_layer[layer_idx]
        ax.imshow(
            data, aspect='equal', cmap=cmap,
            vmin=0.0, vmax=max(float(data.max()), 1e-9),
            origin='upper', interpolation='nearest',
        )
        ax.set_xticks(range(T))
        ax.set_yticks(range(T))
        ax.set_xticklabels(labels, rotation=90, fontsize=11)
        if layer_idx == 0:
            ax.set_yticklabels(labels, fontsize=11)
        else:
            ax.set_yticklabels([])
        ax.set_title(f'Layer {layer_idx + 1} (max over heads)', fontsize=14, pad=4)

    title_main = (
        f'Causal attention (full T×T), mean over {n_steps} steps, '
        f'max over {H} heads'
    )
    if mdp_description:
        fig.suptitle(
            f'{title_main}\nMDP: {mdp_description}',
            fontsize=13, y=1.02,
        )
    else:
        fig.suptitle(title_main, fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


# ---------------------------------------------------------------------------
# Per-distribution agreement heatmap
# ---------------------------------------------------------------------------

_BIN_PHASE_LABELS = ['Early', 'Mid', 'Late']

DIST_SHORT_LABELS: Dict[str, str] = {
    'deterministic': 'Deterministic trans.',
    'sparse_reward':  'Sparse rewards',
    'dense_reward':   'Dense rewards',
    'adversarial':    'Adversarial',
    'high_variance':  'High-variance',
}


def plot_per_distribution_agreement(
    dist_data: List[Tuple[str, np.ndarray]],
    n_steps: int,
    save_path: str,
    n_bins: int = 3,
) -> None:
    """Heatmap: rows = reward distribution type, columns = coarse timestep bins.

    Parameters
    ----------
    dist_data : list of (label, agreements_array)
        agreements_array has shape (n_mdps, n_steps) with per-step 0/1 agreement.
    """
    bin_edges = np.linspace(0, n_steps, n_bins + 1, dtype=int)
    use_phase = n_bins <= len(_BIN_PHASE_LABELS)
    if use_phase:
        bin_labels = [
            f'{_BIN_PHASE_LABELS[i]}\nt={bin_edges[i]+1}–{bin_edges[i+1]}'
            for i in range(n_bins)
        ]
    else:
        bin_labels = [
            f't={bin_edges[i]+1}–{bin_edges[i+1]}'
            for i in range(n_bins)
        ]

    n_dists = len(dist_data)
    agree_matrix = np.full((n_dists, n_bins), np.nan)

    for row_i, (_, arr) in enumerate(dist_data):
        for b in range(n_bins):
            lo, hi = int(bin_edges[b]), int(bin_edges[b + 1])
            agree_matrix[row_i, b] = float(arr[:, lo:hi].mean())

    row_labels = [label for label, _ in dist_data]

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(agree_matrix, aspect='auto', cmap='RdYlGn', vmin=0, vmax=1)
    ax.set_xticks(range(n_bins))
    rotation = 0 if use_phase else 45
    ha = 'center' if use_phase else 'right'
    ax.set_xticklabels(bin_labels, rotation=rotation, ha=ha, fontsize=11)
    ax.set_yticks(range(n_dists))
    ax.set_yticklabels(row_labels, fontsize=11)
    ax.set_xlabel('Episode timestep bin', fontsize=12)
    ax.set_ylabel('Reward distribution', fontsize=12)
    ax.tick_params(axis='both', labelsize=11)
    ax.set_title('Action Agreement by Reward Distribution\nand Timestep Phase',
                 fontsize=13)
    cell_fontsize = 12 if n_bins <= 8 else (10 if n_bins <= 16 else 9)
    for r in range(n_dists):
        for b in range(n_bins):
            val = agree_matrix[r, b]
            if not np.isnan(val):
                ax.text(b, r, f'{val:.2f}', ha='center', va='center',
                        fontsize=cell_fontsize,
                        color='black' if 0.3 < val < 0.85 else 'white')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_per_distribution_agreement_std(
    dist_data: List[Tuple[str, np.ndarray]],
    n_steps: int,
    save_path: str,
    n_bins: int = 3,
) -> None:
    """Heatmap variant of plot_per_distribution_agreement that annotates each
    cell with mean ± standard deviation.

    The mean is the fraction of agreeing (state, step) pairs in the bin. The
    standard deviation is taken across MDPs: for each MDP we compute its mean
    agreement over the bin's steps, and report the spread of those per-MDP
    means. This captures how much the agreement varies from environment to
    environment.

    Parameters
    ----------
    dist_data : list of (label, agreements_array)
        agreements_array has shape (n_mdps, n_steps) with per-step 0/1 agreement.
    """
    bin_edges = np.linspace(0, n_steps, n_bins + 1, dtype=int)
    use_phase = n_bins <= len(_BIN_PHASE_LABELS)
    if use_phase:
        bin_labels = [
            f'{_BIN_PHASE_LABELS[i]}\nt={bin_edges[i]+1}–{bin_edges[i+1]}'
            for i in range(n_bins)
        ]
    else:
        bin_labels = [
            f't={bin_edges[i]+1}–{bin_edges[i+1]}'
            for i in range(n_bins)
        ]

    n_dists = len(dist_data)
    agree_matrix = np.full((n_dists, n_bins), np.nan)
    std_matrix = np.full((n_dists, n_bins), np.nan)

    for row_i, (_, arr) in enumerate(dist_data):
        for b in range(n_bins):
            lo, hi = int(bin_edges[b]), int(bin_edges[b + 1])
            bin_slice = arr[:, lo:hi]
            agree_matrix[row_i, b] = float(bin_slice.mean())
            # Per-MDP mean over the bin's steps, then spread across MDPs.
            per_mdp_means = bin_slice.mean(axis=1)
            std_matrix[row_i, b] = float(per_mdp_means.std())

    row_labels = [label for label, _ in dist_data]

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(agree_matrix, aspect='auto', cmap='RdYlGn', vmin=0, vmax=1)
    ax.set_xticks(range(n_bins))
    rotation = 0 if use_phase else 45
    ha = 'center' if use_phase else 'right'
    ax.set_xticklabels(bin_labels, rotation=rotation, ha=ha, fontsize=11)
    ax.set_yticks(range(n_dists))
    ax.set_yticklabels(row_labels, fontsize=11)
    ax.set_xlabel('Episode timestep bin', fontsize=12)
    ax.set_ylabel('Reward distribution', fontsize=12)
    ax.tick_params(axis='both', labelsize=11)
    ax.set_title('Action Agreement by Reward Distribution\nand Timestep Phase',
                 fontsize=13)
    cell_fontsize = 11 if n_bins <= 8 else (9 if n_bins <= 16 else 8)
    for r in range(n_dists):
        for b in range(n_bins):
            val = agree_matrix[r, b]
            sd = std_matrix[r, b]
            if not np.isnan(val):
                ax.text(b, r, f'{val:.2f}\n±{sd:.2f}', ha='center',
                        va='center', fontsize=cell_fontsize,
                        color='black' if 0.3 < val < 0.85 else 'white')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


# ---------------------------------------------------------------------------
# Part 4: Regret — autonomous transformer vs Q-learner baselines
# ---------------------------------------------------------------------------

def run_q_learner_autonomous(
    P: np.ndarray,
    R: np.ndarray,
    n_states: int,
    n_actions: int,
    n_steps: int,
    alpha: float,
    gamma: float,
    epsilon: float,
    rng: np.random.Generator,
    reward_transform: Optional[Callable] = None,
) -> np.ndarray:
    """Online tabular Q-learning in closed loop.

    `reward_transform(r, s, a, R) -> r_obs` corrupts the reward the learner
    *observes* (and updates Q from) while leaving the reward it is *scored on*
    untouched — see `make_reward_transform`. `None` is the identity and leaves
    this function's behaviour bit-for-bit unchanged.
    """
    Q = np.zeros((n_states, n_actions), dtype=np.float32)
    s = int(rng.integers(n_states))
    rewards = np.zeros(n_steps, dtype=np.float32)

    for t in range(n_steps):
        if rng.random() < epsilon:
            a = int(rng.integers(n_actions))
        else:
            best = float(np.max(Q[s]))
            ties = [ac for ac in range(n_actions) if Q[s, ac] == best]
            a = int(rng.choice(ties))

        r = float(R[s, a])
        r_obs = r if reward_transform is None else float(reward_transform(r, s, a, R))
        s_next = int(rng.choice(n_states, p=P[s, a]))
        Q[s, a] = (1.0 - alpha) * Q[s, a] + alpha * (r_obs + gamma * float(np.max(Q[s_next])))
        rewards[t] = r
        s = s_next

    return rewards


def run_transformer_autonomous(
    model: COCONUTTransformer,
    P: np.ndarray,
    R: np.ndarray,
    n_states: int,
    n_actions: int,
    n_steps: int,
    vocab: Dict,
    config: COCONUTConfig,
    device: torch.device,
    epsilon: float,
    rng: np.random.Generator,
    reward_transform: Optional[Callable] = None,
) -> np.ndarray:
    """Transformer in closed loop: it picks the action, the MDP answers, and the
    resulting transition is fed back through the recurrent context.

    `reward_transform(r, s, a, R) -> r_obs` corrupts the reward the *model
    observes* while `rewards[t]` keeps recording the true environmental reward
    the policy actually earned — so a corrupted run is still scored on real
    reward. `None` is the identity and leaves behaviour bit-for-bit unchanged.
    Note the reward enters only through `reward_value`; `build_step_tokens`
    reads s/a/s_next/a_star and never `tr['r']`.
    """
    max_actions = config.max_actions
    model.eval()
    context = model.get_init_context(1, n_actions, device)
    s = int(rng.integers(n_states))
    rewards = np.zeros(n_steps, dtype=np.float32)

    with torch.no_grad():
        for t in range(n_steps):
            if rng.random() < epsilon:
                a = int(rng.integers(n_actions))
            else:
                best = float(np.max(np.zeros(n_actions))) if t == 0 else None
                a = int(rng.integers(n_actions)) if t == 0 else pred_a

            r = float(R[s, a])
            r_obs = r if reward_transform is None else float(reward_transform(r, s, a, R))
            s_next = int(rng.choice(n_states, p=P[s, a]))
            rewards[t] = r

            tr = {'s': s, 'a': a, 'r': r_obs, 's_next': s_next, 'a_star': 0}
            token_list, r_off, s_off, u_off = build_step_tokens(tr, vocab, n_actions)
            token_ids = torch.tensor([token_list], dtype=torch.long, device=device)
            reward_val = torch.tensor([r_obs], dtype=torch.float32, device=device)

            select_logits, update_hidden = model.forward_step(
                token_ids=token_ids,
                reward_value=reward_val,
                reward_offset=r_off,
                select_offset=s_off,
                update_offset=u_off,
                context=context,
            )

            if n_actions < max_actions:
                select_logits[:, n_actions:] = float('-inf')
            pred_a = int(select_logits[0].argmax().item())

            new_context = context.clone()
            new_context[0, a, :] = model.contextualize(update_hidden)[0]
            context = new_context
            s = s_next

    return rewards


def value_iteration(
    P: np.ndarray,
    R: np.ndarray,
    gamma: float,
    tol: float = 1e-8,
    max_iter: int = 10000,
) -> np.ndarray:
    n_states, n_actions = R.shape
    V = np.zeros(n_states, dtype=np.float64)
    for _ in range(max_iter):
        Q = R + gamma * (P @ V)
        V_new = Q.max(axis=1)
        if np.max(np.abs(V_new - V)) < tol:
            V = V_new
            break
        V = V_new
    Q = R + gamma * (P @ V)
    return Q.astype(np.float32)


def run_optimal_autonomous(
    P: np.ndarray,
    R: np.ndarray,
    n_states: int,
    n_actions: int,
    n_steps: int,
    gamma: float,
    rng: np.random.Generator,
) -> np.ndarray:
    Q_star = value_iteration(P, R, gamma)
    s = int(rng.integers(n_states))
    rewards = np.zeros(n_steps, dtype=np.float32)
    for t in range(n_steps):
        best = float(np.max(Q_star[s]))
        ties = [ac for ac in range(n_actions) if Q_star[s, ac] == best]
        a = int(rng.choice(ties))
        rewards[t] = float(R[s, a])
        s = int(rng.choice(n_states, p=P[s, a]))
    return rewards


# ---------------------------------------------------------------------------
# Nonstationary (mid-episode switch) rollouts
#
# Same closed-loop protocol as the *_autonomous runners above, except the
# environment is a schedule of phases: the agent is never told that the MDP
# changed, so post-switch reward measures in-context re-adaptation. Every
# agent keeps whatever internal state it had (transformer context, tabular Q)
# across the boundary.
# ---------------------------------------------------------------------------

def expand_phases(
    phases: List[Tuple[np.ndarray, np.ndarray, int]],
) -> Tuple[List[np.ndarray], List[np.ndarray], List[int]]:
    """[(P, R, n_steps), ...] -> per-timestep (P_t, R_t, phase_index) lists."""
    Ps: List[np.ndarray] = []
    Rs: List[np.ndarray] = []
    ids: List[int] = []
    for k, (P, R, n) in enumerate(phases):
        Ps.extend([P] * n)
        Rs.extend([R] * n)
        ids.extend([k] * n)
    return Ps, Rs, ids


def run_transformer_nonstationary(
    model: COCONUTTransformer,
    phases: List[Tuple[np.ndarray, np.ndarray, int]],
    n_states: int,
    n_actions: int,
    vocab: Dict,
    config: COCONUTConfig,
    device: torch.device,
    epsilon: float,
    rng: np.random.Generator,
) -> np.ndarray:
    Ps, Rs, _ = expand_phases(phases)
    n_steps = len(Ps)
    max_actions = config.max_actions
    model.eval()
    context = model.get_init_context(1, n_actions, device)
    s = int(rng.integers(n_states))
    rewards = np.zeros(n_steps, dtype=np.float32)
    pred_a = 0

    with torch.no_grad():
        for t in range(n_steps):
            P, R = Ps[t], Rs[t]
            # Same draw order as run_transformer_autonomous, so the phase-1
            # segment reproduces the stationary rollout exactly.
            if rng.random() < epsilon:
                a = int(rng.integers(n_actions))
            else:
                a = int(rng.integers(n_actions)) if t == 0 else pred_a

            r = float(R[s, a])
            s_next = int(rng.choice(n_states, p=P[s, a]))
            rewards[t] = r

            tr = {'s': s, 'a': a, 'r': r, 's_next': s_next, 'a_star': 0}
            token_list, r_off, s_off, u_off = build_step_tokens(tr, vocab, n_actions)
            token_ids = torch.tensor([token_list], dtype=torch.long, device=device)
            reward_val = torch.tensor([r], dtype=torch.float32, device=device)

            select_logits, update_hidden = model.forward_step(
                token_ids=token_ids,
                reward_value=reward_val,
                reward_offset=r_off,
                select_offset=s_off,
                update_offset=u_off,
                context=context,
            )

            if n_actions < max_actions:
                select_logits[:, n_actions:] = float('-inf')
            pred_a = int(select_logits[0].argmax().item())

            new_context = context.clone()
            new_context[0, a, :] = model.contextualize(update_hidden)[0]
            context = new_context
            s = s_next

    return rewards


def run_q_learner_nonstationary(
    phases: List[Tuple[np.ndarray, np.ndarray, int]],
    n_states: int,
    n_actions: int,
    alpha: float,
    gamma: float,
    epsilon: float,
    rng: np.random.Generator,
) -> np.ndarray:
    Ps, Rs, _ = expand_phases(phases)
    n_steps = len(Ps)
    Q = np.zeros((n_states, n_actions), dtype=np.float32)
    s = int(rng.integers(n_states))
    rewards = np.zeros(n_steps, dtype=np.float32)

    for t in range(n_steps):
        P, R = Ps[t], Rs[t]
        if rng.random() < epsilon:
            a = int(rng.integers(n_actions))
        else:
            best = float(np.max(Q[s]))
            ties = [ac for ac in range(n_actions) if Q[s, ac] == best]
            a = int(rng.choice(ties))

        r = float(R[s, a])
        s_next = int(rng.choice(n_states, p=P[s, a]))
        Q[s, a] = (1.0 - alpha) * Q[s, a] + alpha * (r + gamma * float(np.max(Q[s_next])))
        rewards[t] = r
        s = s_next

    return rewards


def run_optimal_nonstationary(
    phases: List[Tuple[np.ndarray, np.ndarray, int]],
    n_states: int,
    n_actions: int,
    gamma: float,
    rng: np.random.Generator,
    adapt: bool = True,
) -> np.ndarray:
    """Oracle reference. adapt=True recomputes Q* at every phase boundary (an
    agent that instantly knows the new MDP); adapt=False keeps the phase-1
    optimal policy forever (an agent that never notices the switch), which is
    the floor that any real adaptation must beat.
    """
    Ps, Rs, phase_ids = expand_phases(phases)
    n_steps = len(Ps)
    q_cache: Dict[int, np.ndarray] = {}
    s = int(rng.integers(n_states))
    rewards = np.zeros(n_steps, dtype=np.float32)

    for t in range(n_steps):
        P, R = Ps[t], Rs[t]
        key = phase_ids[t] if adapt else 0
        if key not in q_cache:
            src = t if adapt else 0
            q_cache[key] = value_iteration(Ps[src], Rs[src], gamma)
        Q_star = q_cache[key]

        best = float(np.max(Q_star[s]))
        ties = [ac for ac in range(n_actions) if Q_star[s, ac] == best]
        a = int(rng.choice(ties))
        rewards[t] = float(R[s, a])
        s = int(rng.choice(n_states, p=P[s, a]))

    return rewards


def run_random_nonstationary(
    phases: List[Tuple[np.ndarray, np.ndarray, int]],
    n_states: int,
    n_actions: int,
    rng: np.random.Generator,
) -> np.ndarray:
    Ps, Rs, _ = expand_phases(phases)
    n_steps = len(Ps)
    s = int(rng.integers(n_states))
    rewards = np.zeros(n_steps, dtype=np.float32)
    for t in range(n_steps):
        a = int(rng.integers(n_actions))
        rewards[t] = float(Rs[t][s, a])
        s = int(rng.choice(n_states, p=Ps[t][s, a]))
    return rewards


def collect_nonstationary(
    model: Optional[COCONUTTransformer],
    config: Optional[COCONUTConfig],
    vocab: Optional[Dict],
    n_states: int,
    n_actions: int,
    device: Optional[torch.device],
    variant: str,
    eval_seeds: List[int],
    switch_step: int,
    post_steps: int,
    alpha: float,
    gamma: float,
    epsilon: float,
    window: int,
) -> Dict[str, np.ndarray]:
    """Roll every agent through the two-phase MDP for each eval seed.

    Returns cum_X / rate_X arrays of shape [n_mdps, switch_step + post_steps]:
      t  = transformer (greedy)     o  = optimal, adapts at the switch
      e  = ε-greedy tabular Q       st = optimal, frozen on phase 1
      g  = greedy tabular Q         r  = uniform random
    Pass model=None to compute the baselines only.
    """
    rew: Dict[str, List[np.ndarray]] = {k: [] for k in ('t', 'te', 'o', 'st', 'e', 'g', 'r')}
    for seed in eval_seeds:
        (P1, R1), (P2, R2) = generate_nonstationary_mdp(
            n_states, n_actions, variant, seed=seed)
        phases = [(P1, R1, switch_step), (P2, R2, post_steps)]
        # Every agent gets its own generator seeded identically (same
        # convention as the *_autonomous evaluations): same start state and
        # the same random stream, so differences are policy differences.
        new_rng = lambda: np.random.default_rng(seed + 100)

        if model is not None:
            rew['t'].append(run_transformer_nonstationary(
                model, phases, n_states, n_actions, vocab, config, device,
                epsilon=0.0, rng=new_rng()))
            rew['te'].append(run_transformer_nonstationary(
                model, phases, n_states, n_actions, vocab, config, device,
                epsilon=epsilon, rng=new_rng()))
        rew['o'].append(run_optimal_nonstationary(
            phases, n_states, n_actions, gamma=gamma, rng=new_rng(), adapt=True))
        rew['st'].append(run_optimal_nonstationary(
            phases, n_states, n_actions, gamma=gamma, rng=new_rng(), adapt=False))
        rew['e'].append(run_q_learner_nonstationary(
            phases, n_states, n_actions, alpha=alpha, gamma=gamma,
            epsilon=epsilon, rng=new_rng()))
        rew['g'].append(run_q_learner_nonstationary(
            phases, n_states, n_actions, alpha=alpha, gamma=gamma,
            epsilon=0.0, rng=new_rng()))
        rew['r'].append(run_random_nonstationary(
            phases, n_states, n_actions, rng=new_rng()))

    out: Dict[str, np.ndarray] = {}
    for key, runs in rew.items():
        if not runs:
            continue
        arr = np.stack(runs, axis=0)
        out[f'cum_{key}'] = np.cumsum(arr, axis=1)
        out[f'rate_{key}'] = trailing_rate(arr, window)
    return out


def trailing_rate(rewards: np.ndarray, window: int) -> np.ndarray:
    """Trailing mean per-step reward over `window` steps, same length as
    `rewards` (the first `window-1` entries average what exists so far).
    Works on a [n_mdps, n_steps] array along the step axis.
    """
    cum = np.cumsum(rewards, axis=-1)
    n = rewards.shape[-1]
    idx = np.arange(n)
    lo = np.maximum(idx - window, -1)
    lagged = np.where(lo >= 0, cum[..., lo], 0.0)
    counts = idx - lo
    return (cum - lagged) / counts


def summarize_nonstationary(
    rate: np.ndarray,
    opt_rate: np.ndarray,
    switch_step: int,
    window: int,
) -> Dict[str, float]:
    """Reward rate (and % of the adapting oracle) in three windows: just before
    the switch, just after it, and at the end of the post-switch phase.
    """
    m = rate.mean(axis=0)
    o = opt_rate.mean(axis=0)
    pre_i = switch_step - 1
    post_i = min(switch_step + window - 1, len(m) - 1)
    end_i = len(m) - 1
    out = {}
    for tag, i in (('pre', pre_i), ('post', post_i), ('end', end_i)):
        out[f'{tag}_rate'] = float(m[i])
        out[f'{tag}_pct_opt'] = float(100.0 * m[i] / o[i]) if o[i] > 0 else float('nan')

    # Steps after the switch until the trailing rate returns to its pre-switch
    # level (on the mean curve); NaN if it never does.
    target = m[pre_i]
    recovered = np.nan
    for t in range(switch_step, len(m)):
        if m[t] >= target:
            recovered = float(t - switch_step)
            break
    out['steps_to_recover'] = recovered
    return out


def plot_nonstationary(
    results: List[Dict],
    switch_step: int,
    window: int,
    n_mdps: int,
    save_path: str,
    label: str = 'Transformer',
) -> None:
    """Per-variant columns; top row cumulative reward, bottom row trailing
    reward rate. The dashed vertical line is the mid-episode switch.
    """
    n = len(results)
    fig, axes = plt.subplots(2, n, figsize=(5.0 * n, 8.6), squeeze=False)

    series = [
        ('Optimal (adapts instantly)', 'cum_o',  'rate_o',  'black',       '--', 1.6),
        ('Optimal (never adapts)',     'cum_st', 'rate_st', '#9467bd',     ':',  1.6),
        (label,                        'cum_t',  'rate_t',  'steelblue',   '-',  2.2),
        (f'{label}, ε-greedy',          'cum_te', 'rate_te', 'steelblue',   '--', 2.0),
        ('ε-greedy Q (ε=0.2)',         'cum_e',  'rate_e',  'forestgreen', '-',  1.6),
        ('Greedy Q (ε=0)',             'cum_g',  'rate_g',  'darkorange',  '-',  1.6),
        ('Random policy',              'cum_r',  'rate_r',  'gray',        ':',  1.4),
    ]

    for j, res in enumerate(results):
        n_steps = res['cum_t'].shape[1]
        steps = np.arange(1, n_steps + 1)
        ax_c, ax_r = axes[0][j], axes[1][j]

        for name, ckey, rkey, color, ls, lw in series:
            if ckey not in res:
                continue
            cum = res[ckey]
            ax_c.plot(steps, cum.mean(0), color=color, linestyle=ls, linewidth=lw,
                      label=f'{name} ({cum[:, -1].mean():.1f})')
            if ls == '-':
                ax_c.fill_between(steps, cum.mean(0) - cum.std(0),
                                  cum.mean(0) + cum.std(0), color=color, alpha=0.12)
            rate = res[rkey]
            ax_r.plot(steps, rate.mean(0), color=color, linestyle=ls, linewidth=lw,
                      label=name)

        for ax in (ax_c, ax_r):
            ax.axvline(switch_step, color='red', linestyle='--', linewidth=1.5,
                       alpha=0.7)
            ax.grid(True, alpha=0.3)
            ax.set_xlabel('Timestep', fontsize=12)

        ax_c.set_title(f"{res['label']}", fontsize=12.5)
        ax_c.legend(fontsize=8.5, loc='upper left')
        ax_r.set_ylim(0.0, 1.0)
        if j == 0:
            ax_c.set_ylabel('Cumulative reward', fontsize=13)
            ax_r.set_ylabel(f'Reward rate ({window}-step trailing mean)', fontsize=12)
            ax_c.text(switch_step, ax_c.get_ylim()[1] * 0.02, ' switch',
                      color='red', fontsize=9, va='bottom')

    fig.suptitle(
        f'Nonstationary MDPs — the environment changes at t={switch_step} '
        f'with no signal to the agent\n'
        f'({switch_step} steps before, {results[0]["cum_t"].shape[1] - switch_step} '
        f'after; mean ± std over {n_mdps} MDPs)',
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {save_path}")


def run_nonstationary_eval(
    model: COCONUTTransformer,
    config: COCONUTConfig,
    vocab: Dict,
    n_states: int,
    n_actions: int,
    device: torch.device,
    args: argparse.Namespace,
    eval_seeds: List[int],
    run_label: str,
    figures_dir: str,
) -> List[Dict]:
    """Part 4d end to end: roll out every variant, print the summary table,
    write nonstationary.png + nonstationary_summary.csv into `figures_dir`.
    """
    switch = args.nonstationary_switch_step
    window = args.nonstationary_window
    total = switch + args.nonstationary_post_steps
    print(f"\n{'=' * 60}")
    print(f"Part 4d: Nonstationary MDPs "
          f"(switch at t={switch}, {args.nonstationary_post_steps} steps after, "
          f"{len(NONSTATIONARY_VARIANTS)} variants × {len(eval_seeds)} MDPs)")
    print(f"{'=' * 60}")

    ns_results: List[Dict] = []
    ns_rows: List[Dict] = []
    agent_names = [('t', run_label), ('te', f'{run_label}, eps-greedy'), ('o', 'optimal (adapts)'),
                   ('st', 'optimal (frozen)'), ('e', 'eps-greedy Q'),
                   ('g', 'greedy Q'), ('r', 'random')]
    for variant, vlabel in NONSTATIONARY_VARIANTS:
        print(f"\n  Variant: {variant} — {vlabel}", flush=True)
        res = collect_nonstationary(
            model, config, vocab, n_states, n_actions, device,
            variant=variant, eval_seeds=eval_seeds,
            switch_step=switch, post_steps=args.nonstationary_post_steps,
            alpha=args.alpha, gamma=args.gamma, epsilon=args.epsilon,
            window=window,
        )
        res['name'] = variant
        res['label'] = vlabel
        ns_results.append(res)

        print(f"    {'agent':<20}{'pre':>8}{'post':>8}{'end':>8}"
              f"{'pre%opt':>9}{'post%opt':>10}{'end%opt':>9}{'recov':>8}")
        for key, name in agent_names:
            s = summarize_nonstationary(res[f'rate_{key}'], res['rate_o'],
                                        switch_step=switch, window=window)
            rec = ('never' if np.isnan(s['steps_to_recover'])
                   else f"{s['steps_to_recover']:.0f}")
            print(f"    {name:<20}{s['pre_rate']:>8.3f}{s['post_rate']:>8.3f}"
                  f"{s['end_rate']:>8.3f}{s['pre_pct_opt']:>8.0f}%"
                  f"{s['post_pct_opt']:>9.0f}%{s['end_pct_opt']:>8.0f}%{rec:>8}")
            ns_rows.append({
                'variant': variant, 'agent': name,
                'final_cumreward': float(res[f'cum_{key}'][:, -1].mean()),
                'pre_rate': s['pre_rate'], 'post_rate': s['post_rate'],
                'end_rate': s['end_rate'],
                'pre_pct_optimal': s['pre_pct_opt'],
                'post_pct_optimal': s['post_pct_opt'],
                'end_pct_optimal': s['end_pct_opt'],
                'steps_to_recover': s['steps_to_recover'],
            })

    os.makedirs(figures_dir, exist_ok=True)
    plot_nonstationary(
        ns_results, switch_step=switch, window=window,
        n_mdps=len(eval_seeds), label=f'{run_label} transformer',
        save_path=os.path.join(figures_dir, 'nonstationary.png'),
    )
    csv_path = os.path.join(figures_dir, 'nonstationary_summary.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(ns_rows[0].keys()))
        writer.writeheader()
        writer.writerows(ns_rows)
    print(f"\n  Saved: {csv_path}   (horizon {total} steps)")
    return ns_results


# ---------------------------------------------------------------------------
# Part 4e: Reward interventions
#
# Agreement with the tabular policy cannot distinguish a model that runs an
# online RL update from one that has learned state/action statistics which
# happen to correlate with the teacher. These interventions separate the two
# causally: the reward the agent *observes* is corrupted while the reward it is
# *scored on* stays the true environmental reward, so any drop in return is
# attributable to the agent no longer being able to use reward feedback.
#
# `decorrelated` is the load-bearing condition. It resamples the observed
# reward from the same reward matrix, so the marginal reward distribution the
# agent sees is exactly unchanged and only its association with the transition
# just taken is destroyed. An agent that merely needs a plausible scalar in the
# reward slot is unaffected by it; an agent doing TD updates is not.
#
# `inverted` is the directional control: an agent running a value update on a
# flipped reward should actively pursue the worst actions and fall *below* the
# random-policy floor, which no amount of reward-independent policy prior can
# produce.
# ---------------------------------------------------------------------------

REWARD_INTERVENTIONS: List[Tuple[str, str]] = [
    ('intact',       'True reward (control)'),
    ('zeroed',       'Reward always 0'),
    ('constant',     'Reward always 0.5'),
    ('decorrelated', 'Reward resampled from R (marginal preserved, link broken)'),
    ('inverted',     'Reward flipped (1 - r)'),
]


def make_reward_transform(name: str, seed: int) -> Optional[Callable]:
    """Build the `reward_transform(r, s, a, R) -> r_obs` callable for a
    condition.

    The closure owns a private RNG rather than borrowing the rollout's, so
    corrupting the reward never perturbs the environment's transition draws:
    every condition sees the same MDP dynamics and differs only in what the
    agent is told about reward. `intact` returns None so that the control
    condition runs the exact code path of the uninstrumented evaluation.
    """
    if name == 'intact':
        return None
    if name == 'zeroed':
        return lambda r, s, a, R: 0.0
    if name == 'constant':
        return lambda r, s, a, R: 0.5
    if name == 'inverted':
        return lambda r, s, a, R: 1.0 - r
    if name == 'decorrelated':
        rng = np.random.default_rng(seed)
        return lambda r, s, a, R: float(
            R[rng.integers(R.shape[0]), rng.integers(R.shape[1])])
    raise ValueError(f"Unknown reward intervention: {name!r}")


def collect_reward_intervention(
    model: COCONUTTransformer,
    config: COCONUTConfig,
    vocab: Dict,
    n_states: int,
    n_actions: int,
    device: torch.device,
    mdp_fn: Callable,
    eval_seeds: List[int],
    n_steps: int,
    alpha: float,
    gamma: float,
    epsilon: float,
) -> Dict:
    """Roll out every intervention on one MDP family.

    Returns per-condition final returns for the transformer and both tabular
    Q-learners, plus the condition-independent optimal/random references (both
    ignore observed reward, so they are computed once).
    """
    out: Dict[str, Dict[str, List[float]]] = {
        name: {'transformer': [], 'transformer_eps': [], 'epsgreedy': [], 'greedy': []}
        for name, _ in REWARD_INTERVENTIONS
    }
    optimal, random_pol = [], []

    for seed in eval_seeds:
        P, R = mdp_fn(n_states, n_actions, seed=seed)

        optimal.append(float(run_optimal_autonomous(
            P, R, n_states, n_actions, n_steps, gamma=gamma,
            rng=np.random.default_rng(seed + 100)).sum()))
        random_pol.append(float(run_random_nonstationary(
            [(P, R, n_steps)], n_states, n_actions,
            rng=np.random.default_rng(seed + 100)).sum()))

        for name, _ in REWARD_INTERVENTIONS:
            # Same transform seed across agents/MDPs within a condition so the
            # corruption stream is shared, and a fresh rollout RNG seeded
            # exactly as in the uninstrumented eval.
            rt = lambda: make_reward_transform(name, seed + 7777)
            out[name]['transformer'].append(float(run_transformer_autonomous(
                model, P, R, n_states, n_actions, n_steps, vocab, config,
                device, epsilon=0.0, rng=np.random.default_rng(seed + 100),
                reward_transform=rt()).sum()))
            out[name]['transformer_eps'].append(float(run_transformer_autonomous(
                model, P, R, n_states, n_actions, n_steps, vocab, config,
                device, epsilon=epsilon, rng=np.random.default_rng(seed + 100),
                reward_transform=rt()).sum()))
            out[name]['epsgreedy'].append(float(run_q_learner_autonomous(
                P, R, n_states, n_actions, n_steps, alpha=alpha, gamma=gamma,
                epsilon=epsilon, rng=np.random.default_rng(seed + 100),
                reward_transform=rt()).sum()))
            out[name]['greedy'].append(float(run_q_learner_autonomous(
                P, R, n_states, n_actions, n_steps, alpha=alpha, gamma=gamma,
                epsilon=0.0, rng=np.random.default_rng(seed + 100),
                reward_transform=rt()).sum()))

    return {
        'conditions': {k: {a: np.array(v, dtype=np.float64)
                           for a, v in d.items()} for k, d in out.items()},
        'optimal': np.array(optimal, dtype=np.float64),
        'random':  np.array(random_pol, dtype=np.float64),
    }


def plot_reward_intervention(
    results: List[Dict],
    n_mdps: int,
    n_steps: int,
    label: str,
    save_path: str,
) -> None:
    """One panel per MDP family; grouped bars of final return as % of optimal.

    The random-policy floor is drawn as a reference line because on the
    Beta(2,2) family it sits near 70% of optimal — without it the bars there
    look like a large effect when they are not.
    """
    agents = [('transformer', f'{label} transformer', '#1f77b4'),
              ('transformer_eps', f'{label} transformer, ε-greedy', '#9ecae1'),
              ('epsgreedy',   'ε-greedy tabular Q',   '#2ca02c'),
              ('greedy',      'greedy tabular Q',     '#8c564b')]
    names = [n for n, _ in REWARD_INTERVENTIONS]
    xlabels = [d for _, d in REWARD_INTERVENTIONS]

    fig, axes = plt.subplots(1, len(results), figsize=(9 * len(results), 5.6),
                             squeeze=False)
    for ax, res in zip(axes[0], results):
        opt = res['data']['optimal'].mean()
        x = np.arange(len(names))
        width = 0.8 / len(agents)
        for k, (key, alabel, color) in enumerate(agents):
            vals = [100.0 * res['data']['conditions'][n][key].mean() / opt
                    for n in names]
            errs = [100.0 * res['data']['conditions'][n][key].std() / opt
                    for n in names]
            ax.bar(x + k * width - 0.4 + width / 2, vals, width, yerr=errs,
                   capsize=3, color=color, label=alabel, alpha=0.9)
        ax.axhline(100.0 * res['data']['random'].mean() / opt, color='black',
                   linestyle=':', linewidth=1.6, label='random policy')
        ax.axhline(100.0, color='black', linestyle='--', linewidth=1.4,
                   label='optimal policy')
        ax.set_xticks(x)
        ax.set_xticklabels([t.replace(' (', '\n(') for t in xlabels],
                           fontsize=9, rotation=20, ha='right')
        ax.set_ylabel('final return (% of optimal)')
        ax.set_title(f"{res['label']}\n({n_steps} steps, {n_mdps} MDPs)",
                     fontsize=12)
        ax.grid(True, axis='y', alpha=0.3)
        # Headroom above the optimal line so the legend never covers a bar.
        ax.set_ylim(0, 132)
        ax.legend(fontsize=8.5, loc='upper center', ncol=3, framealpha=0.95)
    fig.suptitle('Reward interventions — reward the agent OBSERVES is corrupted; '
                 'reward it is SCORED on is not', fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
    print(f"  Saved: {save_path}")


def run_reward_intervention_eval(
    model: COCONUTTransformer,
    config: COCONUTConfig,
    vocab: Dict,
    n_states: int,
    n_actions: int,
    device: torch.device,
    args: argparse.Namespace,
    eval_seeds: List[int],
    run_label: str,
    figures_dir: str,
) -> List[Dict]:
    """Part 4e end to end: sweep interventions × MDP families, print the table,
    write reward_intervention.png + reward_intervention_summary.csv.
    """
    n_steps = args.reward_intervention_steps
    print(f"\n{'=' * 60}")
    print(f"Part 4e: Reward interventions "
          f"({len(REWARD_INTERVENTIONS)} conditions × {len(MDP_FAMILIES)} "
          f"families × {len(eval_seeds)} MDPs, {n_steps} steps)")
    print(f"{'=' * 60}")

    results: List[Dict] = []
    rows: List[Dict] = []
    for fam, fam_label, mdp_fn in MDP_FAMILIES:
        print(f"\n  Family: {fam} — {fam_label}", flush=True)
        data = collect_reward_intervention(
            model, config, vocab, n_states, n_actions, device,
            mdp_fn=mdp_fn, eval_seeds=eval_seeds, n_steps=n_steps,
            alpha=args.alpha, gamma=args.gamma, epsilon=args.epsilon,
        )
        results.append({'name': fam, 'label': fam_label, 'data': data})

        opt = data['optimal'].mean()
        rnd = data['random'].mean()
        print(f"    optimal {opt:.1f}   random {rnd:.1f} "
              f"({100.0 * rnd / opt:.0f}% of optimal)")
        print(f"    {'condition':<16}{'transformer':>14}{'eps-greedy Q':>15}"
              f"{'greedy Q':>12}")
        for name, desc in REWARD_INTERVENTIONS:
            c = data['conditions'][name]
            print(f"    {name:<16}"
                  f"{100.0 * c['transformer'].mean() / opt:>13.0f}%"
                  f"{100.0 * c['epsgreedy'].mean() / opt:>14.0f}%"
                  f"{100.0 * c['greedy'].mean() / opt:>11.0f}%")
            for key, agent in (('transformer', f'{run_label} transformer'),
                               ('transformer_eps', f'{run_label} transformer, eps-greedy'),
                               ('epsgreedy', 'eps-greedy Q'),
                               ('greedy', 'greedy Q')):
                rows.append({
                    'family': fam, 'condition': name, 'description': desc,
                    'agent': agent,
                    'final_return': float(c[key].mean()),
                    'final_return_std': float(c[key].std()),
                    'pct_optimal': float(100.0 * c[key].mean() / opt),
                    'optimal': float(opt), 'random': float(rnd),
                    'random_pct_optimal': float(100.0 * rnd / opt),
                })

    os.makedirs(figures_dir, exist_ok=True)
    plot_reward_intervention(
        results, n_mdps=len(eval_seeds), n_steps=n_steps, label=run_label,
        save_path=os.path.join(figures_dir, 'reward_intervention.png'),
    )
    csv_path = os.path.join(figures_dir, 'reward_intervention_summary.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Saved: {csv_path}")
    return results


# ---------------------------------------------------------------------------
# Part 4f: State-action size sweep
#
# Every headline number in the paper is measured at the largest MDP the model
# was trained on (|S|=8, |A|=4), which says nothing about how behaviour varies
# with problem size. This sweeps the whole trained range.
#
# The vocabulary is deliberately NOT rebuilt per cell. Token ids are positional
# (`TOK_S = range(2, 2+max_states)`, `TOK_R = 2+max_states+max_actions`), so a
# vocab built at the smaller size would renumber every special token and the
# model would read garbage without raising. Smaller MDPs simply use a prefix of
# the full-size TOK_S/TOK_A lists, which is exactly what training did.
# ---------------------------------------------------------------------------

def collect_size_sweep(
    model: COCONUTTransformer,
    config: COCONUTConfig,
    vocab: Dict,
    device: torch.device,
    args: argparse.Namespace,
    eval_seeds: List[int],
    run_label: str,
) -> Dict:
    """Sweep the trained |S| x |A| grid; returns grids + per-cell CSV rows.

    Split out from `run_size_sweep_eval` so the continuous-vs-discrete
    comparison can collect both models through this exact code path.
    """
    states = list(range(2, config.max_states + 1))
    actions = list(range(2, config.max_actions + 1))
    n_steps = args.size_sweep_steps

    agree_grid = np.full((len(states), len(actions)), np.nan)
    pct_grid = np.full((len(states), len(actions)), np.nan)
    pct_eps_grid = np.full((len(states), len(actions)), np.nan)
    rnd_grid = np.full((len(states), len(actions)), np.nan)
    rows: List[Dict] = []

    for i, ns in enumerate(states):
        for j, na in enumerate(actions):
            agrees, rets, rets_eps, opts, rnds = [], [], [], [], []
            for seed in eval_seeds:
                # Teacher-forced agreement on the paper's Beta(2,2) family.
                P, R = generate_eval_mdp(ns, na, seed=seed)
                trajectory, _ = run_tabular_q_learning(
                    P, R, ns, na, n_steps=args.n_steps, alpha=args.alpha,
                    gamma=args.gamma, epsilon=args.epsilon, seed=seed)
                preds, _ = run_action_inference(model, trajectory, vocab, na,
                                                config, device)
                targets = np.array([st['a_star'] for st in trajectory],
                                   dtype=np.int32)
                agrees.append(float((preds == targets).mean()))

                # Closed-loop return on the contrast family, where reward has
                # enough dynamic range to separate policies.
                Pc, Rc = generate_contrast_mdp_seeded(ns, na, seed=seed)
                rets.append(float(run_transformer_autonomous(
                    model, Pc, Rc, ns, na, n_steps, vocab, config, device,
                    epsilon=0.0, rng=np.random.default_rng(seed + 100)).sum()))
                rets_eps.append(float(run_transformer_autonomous(
                    model, Pc, Rc, ns, na, n_steps, vocab, config, device,
                    epsilon=args.epsilon, rng=np.random.default_rng(seed + 100)).sum()))
                opts.append(float(run_optimal_autonomous(
                    Pc, Rc, ns, na, n_steps, gamma=args.gamma,
                    rng=np.random.default_rng(seed + 100)).sum()))
                rnds.append(float(run_random_nonstationary(
                    [(Pc, Rc, n_steps)], ns, na,
                    rng=np.random.default_rng(seed + 100)).sum()))

            opt_mean = float(np.mean(opts))
            agree_grid[i, j] = float(np.mean(agrees))
            pct_grid[i, j] = 100.0 * float(np.mean(rets)) / opt_mean
            pct_eps_grid[i, j] = 100.0 * float(np.mean(rets_eps)) / opt_mean
            rnd_grid[i, j] = 100.0 * float(np.mean(rnds)) / opt_mean
            print(f"  |S|={ns} |A|={na}: agreement {agree_grid[i, j]:.1%}  "
                  f"return {pct_grid[i, j]:.0f}% of optimal "
                  f"(random {rnd_grid[i, j]:.0f}%)", flush=True)
            rows.append({
                'n_states': ns, 'n_actions': na, 'agent': run_label,
                'agreement': agree_grid[i, j],
                'agreement_std': float(np.std(agrees)),
                'chance_agreement': 1.0 / na,
                'return_pct_optimal': pct_grid[i, j],
                'return_pct_optimal_eps': pct_eps_grid[i, j],
                'random_pct_optimal': rnd_grid[i, j],
                'final_return': float(np.mean(rets)),
                'optimal_return': opt_mean,
            })

    return {'states': states, 'actions': actions, 'agreement': agree_grid,
            'pct_optimal': pct_grid, 'pct_optimal_eps': pct_eps_grid, 'random_pct_optimal': rnd_grid,
            'rows': rows, 'n_steps': n_steps}


def run_size_sweep_eval(
    model: COCONUTTransformer,
    config: COCONUTConfig,
    vocab: Dict,
    device: torch.device,
    args: argparse.Namespace,
    eval_seeds: List[int],
    run_label: str,
    figures_dir: str,
) -> Dict:
    """Part 4f end to end: collect the grid, write size_sweep.png +
    size_sweep_summary.csv.
    """
    print(f"\n{'=' * 60}")
    print(f"Part 4f: State-action size sweep "
          f"(|S| in 2..{config.max_states} x |A| in 2..{config.max_actions}, "
          f"{len(eval_seeds)} MDPs/cell)")
    print(f"{'=' * 60}")

    res = collect_size_sweep(model, config, vocab, device, args, eval_seeds,
                             run_label)

    os.makedirs(figures_dir, exist_ok=True)
    plot_size_sweep(res['states'], res['actions'], res['agreement'],
                    res['pct_optimal'], res['random_pct_optimal'],
                    n_mdps=len(eval_seeds), n_steps=res['n_steps'],
                    label=run_label,
                    save_path=os.path.join(figures_dir, 'size_sweep.png'),
                    pct_eps_grid=res.get('pct_optimal_eps'))
    csv_path = os.path.join(figures_dir, 'size_sweep_summary.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(res['rows'][0].keys()))
        writer.writeheader()
        writer.writerows(res['rows'])
    print(f"  Saved: {csv_path}")
    return res


def plot_size_sweep(
    states: List[int],
    actions: List[int],
    agree_grid: np.ndarray,
    pct_grid: np.ndarray,
    rnd_grid: np.ndarray,
    n_mdps: int,
    n_steps: int,
    label: str,
    save_path: str,
    pct_eps_grid: Optional[np.ndarray] = None,
) -> None:
    """Left: agreement heatmap. Middle: closed-loop return as % of optimal.
    Right: the same return minus the random-policy floor for that cell, which
    is the only one of the three that is comparable across |A| (chance
    agreement and the random floor both move with the number of actions).
    """
    grids = [
        (agree_grid * 100.0, 'Action agreement (%)', 'viridis', None),
        (pct_grid, 'Closed-loop return (% of optimal)', 'magma', None),
        (pct_grid - rnd_grid, 'Return above random floor (pp)', 'coolwarm', 0.0),
    ]
    if pct_eps_grid is not None:
        grids.insert(2, (pct_eps_grid, f'Return, ε-greedy (ε={TF_EPS:g}) (% of opt.)', 'magma', None))
    fig, axes = plt.subplots(1, len(grids), figsize=(5.7 * len(grids), 4.8))
    for ax, (grid, title, cmap, center) in zip(axes, grids):
        vmax = np.nanmax(np.abs(grid)) if center is not None else None
        im = ax.imshow(grid, cmap=cmap, aspect='auto',
                       vmin=-vmax if center is not None else None,
                       vmax=vmax if center is not None else None)
        ax.set_xticks(range(len(actions)))
        ax.set_xticklabels(actions)
        ax.set_yticks(range(len(states)))
        ax.set_yticklabels(states)
        ax.set_xlabel('|A| (actions)')
        ax.set_ylabel('|S| (states)')
        ax.set_title(title, fontsize=12)
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                ax.text(j, i, f'{grid[i, j]:.0f}', ha='center', va='center',
                        fontsize=9, color='white'
                        if cmap != 'coolwarm' else 'black')
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(f'{label} transformer — behaviour across the trained '
                 f'state-action range '
                 f'({n_mdps} MDPs/cell; return over {n_steps} steps on '
                 f'contrast MDPs)', fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_long_horizon(
    cumrew_transformer: np.ndarray,
    cumrew_greedy: np.ndarray,
    cumrew_epsgreedy: np.ndarray,
    train_horizon: int,
    n_mdps: int,
    save_path: str,
    cumrew_transformer_eps: Optional[np.ndarray] = None,
) -> None:
    """Single panel: cumulative reward over a horizon longer than the
    training horizon, with a vertical marker at the training cutoff.
    """
    n_steps = cumrew_transformer.shape[1]
    steps = np.arange(1, n_steps + 1)

    fig, ax = plt.subplots(figsize=(9, 7))

    series = [
        ('Transformer', cumrew_transformer, 'steelblue'),
        ('Greedy Q (ε=0)', cumrew_greedy, 'darkorange'),
        ('ε-greedy Q (ε=0.2)', cumrew_epsgreedy, 'forestgreen'),
    ]
    if cumrew_transformer_eps is not None:
        series.insert(1, (f'Transformer, ε-greedy (ε={TF_EPS:g})', cumrew_transformer_eps,
                          'steelblue', '--'))
    for label, data, color, *ls in series:
        mean = data.mean(axis=0)
        std = data.std(axis=0)
        ax.plot(steps, mean, color=color, linewidth=2, label=label,
                linestyle=ls[0] if ls else '-')
        ax.fill_between(steps, mean - std, mean + std, alpha=0.15, color=color)

    ax.axvline(train_horizon, color='red', linestyle='--', linewidth=1.5,
               alpha=0.7, label=f'Training horizon (t={train_horizon})')
    ax.set_xlabel('Timestep', fontsize=13)
    ax.set_ylabel('Cumulative Reward', fontsize=13)
    ax.tick_params(axis='both', labelsize=12)
    ax.set_title(
        f'Cumulative Reward over {n_steps} steps\n'
        f'(mean ± std, {n_mdps} MDPs)',
        fontsize=13,
    )
    ax.legend(fontsize=11, loc='lower right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_regret(
    cumrew_transformer: np.ndarray,
    cumrew_greedy: np.ndarray,
    cumrew_epsgreedy: np.ndarray,
    n_mdps: int,
    save_path: str,
    cumrew_transformer_eps: Optional[np.ndarray] = None,
) -> None:
    """Single tall panel: cumulative reward for the three agents."""
    n_steps = cumrew_transformer.shape[1]
    steps = np.arange(1, n_steps + 1)

    fig, ax = plt.subplots(figsize=(6, 5))

    series = [
        ('Transformer', cumrew_transformer, 'steelblue'),
        ('Greedy Q (ε=0)', cumrew_greedy, 'darkorange'),
        ('ε-greedy Q (ε=0.2)', cumrew_epsgreedy, 'forestgreen'),
    ]

    if cumrew_transformer_eps is not None:
        series.insert(1, (f'Transformer, ε-greedy (ε={TF_EPS:g})', cumrew_transformer_eps,
                          'steelblue', '--'))
    for label, data, color, *ls in series:
        mean = data.mean(axis=0)
        std = data.std(axis=0)
        ax.plot(steps, mean, color=color, linewidth=2, label=label,
                linestyle=ls[0] if ls else '-')
        ax.fill_between(steps, mean - std, mean + std, alpha=0.15, color=color)

    ax.set_xlabel('Timestep', fontsize=13)
    ax.set_ylabel('Cumulative Reward', fontsize=13)
    ax.tick_params(axis='both', labelsize=12)
    ax.set_title(f'Cumulative Reward (mean ± std, {n_mdps} MDPs)',
                 fontsize=13)
    ax.legend(fontsize=11, loc='lower right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_reward_dist_grid(
    results: List[Dict],
    n_mdps: int,
    save_path: str,
) -> None:
    """Multi-panel figure: one subplot per reward distribution, each showing
    cumulative reward (mean ± std over MDPs) for transformer + benchmarks.

    `results` is a list of dicts with keys:
      'name', 'label', 'cumrew_t', 'cumrew_g', 'cumrew_e', 'cumrew_o'
    """
    n = len(results)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows),
                             squeeze=False)

    series_spec = [
        ('Optimal',           'cumrew_o', 'black'),
        ('Transformer',       'cumrew_t', 'steelblue'),
        ('Transformer, ε-greedy', 'cumrew_te', 'steelblue', '--'),
        ('Greedy Q (ε=0)',    'cumrew_g', 'darkorange'),
        ('ε-greedy Q',        'cumrew_e', 'forestgreen'),
    ]

    for idx, res in enumerate(results):
        ax = axes[idx // ncols][idx % ncols]
        n_steps = res['cumrew_t'].shape[1]
        steps = np.arange(1, n_steps + 1)
        for label, key, color, *ls in series_spec:
            if key not in res:
                continue
            data = res[key]
            mean = data.mean(axis=0)
            std = data.std(axis=0)
            ax.plot(steps, mean, color=color, linewidth=2, label=label,
                    linestyle=ls[0] if ls else '-')
            ax.fill_between(steps, mean - std, mean + std, alpha=0.15, color=color)
        ax.set_title(res['label'], fontsize=13)
        ax.set_xlabel('Timestep')
        ax.set_ylabel('Cumulative Reward')
        ax.grid(True, alpha=0.3)
        if idx == 0:
            ax.legend(fontsize=10, loc='upper left')

    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].axis('off')

    fig.suptitle(f'Cumulative Reward by Reward Distribution '
                 f'(mean ± std, {n_mdps} MDPs each)', fontsize=15)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_combined_row(
    cumrew_transformer: np.ndarray,
    cumrew_greedy: np.ndarray,
    cumrew_epsgreedy: np.ndarray,
    n_mdps: int,
    q_true: np.ndarray,
    q_pred: np.ndarray,
    r2: float,
    dist_data: List[Tuple[str, np.ndarray]],
    n_steps: int,
    save_path: str,
    n_bins: int = 3,
    cumrew_transformer_eps: Optional[np.ndarray] = None,
) -> None:
    """Single-row figure: regret (left), probe scatter (middle), agreement (right)."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # --- Left: regret ---
    ax = axes[0]
    n_steps_r = cumrew_transformer.shape[1]
    steps = np.arange(1, n_steps_r + 1)
    series = [
        ('Transformer', cumrew_transformer, 'steelblue'),
        ('Greedy Q (ε=0)', cumrew_greedy, 'darkorange'),
        ('ε-greedy Q (ε=0.2)', cumrew_epsgreedy, 'forestgreen'),
    ]
    if cumrew_transformer_eps is not None:
        series.insert(1, (f'Transformer, ε-greedy (ε={TF_EPS:g})', cumrew_transformer_eps,
                          'steelblue', '--'))
    for label, data, color, *ls in series:
        mean = data.mean(axis=0)
        std = data.std(axis=0)
        ax.plot(steps, mean, color=color, linewidth=2, label=label,
                linestyle=ls[0] if ls else '-')
        ax.fill_between(steps, mean - std, mean + std, alpha=0.15, color=color)
    ax.set_xlabel('Timestep', fontsize=12)
    ax.set_ylabel('Cumulative Reward', fontsize=12)
    ax.tick_params(axis='both', labelsize=11)
    ax.set_title(f'Cumulative Reward (mean ± std, {n_mdps} MDPs)', fontsize=12)
    ax.legend(fontsize=10, loc='lower right')
    ax.grid(True, alpha=0.3)

    # --- Middle: probe scatter ---
    ax = axes[1]
    q_t = q_true.reshape(-1)
    q_p = q_pred.reshape(-1)
    rng = np.random.default_rng(0)
    idx = rng.choice(len(q_t), size=min(10000, len(q_t)), replace=False)
    ax.scatter(q_t[idx], q_p[idx], alpha=0.35, s=10, color='steelblue',
               rasterized=True, edgecolors='none')
    lo = min(q_t.min(), q_p.min())
    hi = max(q_t.max(), q_p.max())
    ax.plot([lo, hi], [lo, hi], 'r--', linewidth=1.5, label='y = x')
    ax.set_xlim(q_t.min(), q_t.max())
    y_lo, y_hi = q_p.min(), q_p.max()
    y_pad = 0.02 * (y_hi - y_lo)
    ax.set_ylim(y_lo - y_pad, y_hi + y_pad)
    ax.set_xlabel('Q tabular', fontsize=12)
    ax.set_ylabel('Q probe (context token)', fontsize=12)
    ax.tick_params(axis='both', labelsize=11)
    ax.set_title(f'Context Token Probe: Q-Values vs True  (R$^2$ = {r2:.4f})',
                 fontsize=12)
    ax.legend(fontsize=10, loc='upper left')
    ax.grid(True, alpha=0.3)

    # --- Right: per-distribution agreement ---
    ax = axes[2]
    bin_edges = np.linspace(0, n_steps, n_bins + 1, dtype=int)
    use_phase = n_bins <= len(_BIN_PHASE_LABELS)
    if use_phase:
        bin_labels = [
            f'{_BIN_PHASE_LABELS[i]}\nt={bin_edges[i]+1}–{bin_edges[i+1]}'
            for i in range(n_bins)
        ]
    else:
        bin_labels = [
            f't={bin_edges[i]+1}–{bin_edges[i+1]}'
            for i in range(n_bins)
        ]
    n_dists = len(dist_data)
    agree_matrix = np.full((n_dists, n_bins), np.nan)
    for row_i, (_, arr) in enumerate(dist_data):
        for b in range(n_bins):
            blo, bhi = int(bin_edges[b]), int(bin_edges[b + 1])
            agree_matrix[row_i, b] = float(arr[:, blo:bhi].mean())
    row_labels = [label for label, _ in dist_data]
    ax.imshow(agree_matrix, aspect='auto', cmap='RdYlGn', vmin=0, vmax=1)
    ax.set_xticks(range(n_bins))
    rotation = 0 if use_phase else 45
    ha = 'center' if use_phase else 'right'
    ax.set_xticklabels(bin_labels, rotation=rotation, ha=ha, fontsize=10)
    ax.set_yticks(range(n_dists))
    ax.set_yticklabels(row_labels, fontsize=10)
    ax.set_xlabel('Episode timestep bin', fontsize=12)
    ax.set_ylabel('Reward distribution', fontsize=12)
    ax.tick_params(axis='both', labelsize=10)
    ax.set_title('Action Agreement by Reward Distribution\nand Timestep Phase',
                 fontsize=12)
    cell_fontsize = 11 if n_bins <= 8 else (9 if n_bins <= 16 else 8)
    for r in range(n_dists):
        for b in range(n_bins):
            val = agree_matrix[r, b]
            if not np.isnan(val):
                ax.text(b, r, f'{val:.2f}', ha='center', va='center',
                        fontsize=cell_fontsize,
                        color='black' if 0.3 < val < 0.85 else 'white')

    plt.tight_layout()
    # The right heatmap has long y-tick labels that eat into the gap between
    # the middle and right axes, making the inter-plot buffers look unequal.
    # Shift the middle axes left so the visual spacing matches.
    pos = axes[1].get_position()
    axes[1].set_position([pos.x0 - 0.03, pos.y0, pos.width, pos.height])
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


# ---------------------------------------------------------------------------
# Part 5: Effective alpha/gamma recovery
# ---------------------------------------------------------------------------

def estimate_effective_alpha_gamma(
    probe: ContextQProbe,
    ctx_all: np.ndarray,
    trajectories: List[List[Dict]],
    device: torch.device,
    n_steps: int,
    n_traj: int,
    n_actions: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit (alpha_eff, gamma_eff) per trajectory from context-probe Q dynamics.

    The Q-learning update for each visited transition (s_t, a_t, r_t, s'_t) is:
        ΔQ(s_t, a_t) = α * [r_t + γ * max_{a'} Q_t(s'_t, a') - Q_t(s_t, a_t)]

    Rearranging into a two-feature OLS:
        ΔQ = α * (r_t - Q_t(s_t, a_t))  +  (α*γ) * max_{a'} Q_t(s'_t, a')

    We fit [α, α*γ] then recover γ = (α*γ) / α.

    Notes
    -----
    - Only the visited (s_t, a_t) entry changes each step; fitting all entries
      would inject noise from the many zero-ΔQ entries.
    - Q_t is decoded from the context *before* step t's update.
    - max_{a'} Q_t(s'_t, a') is taken at the specific next state s'_t, not
      over the full Q matrix.
    - The reward r_t is included in the regression target as required by the
      TD update equation.
    """
    probe.eval()
    n_states = probe.linear.out_features

    with torch.no_grad():
        q_pred = probe(
            torch.tensor(ctx_all, dtype=torch.float32, device=device)
        ).cpu().numpy()

    # q_pred_r[i, t, a, s] = probe-decoded Q(s, a) before step t of trajectory i
    q_pred_r = q_pred.reshape(n_traj, n_steps, n_actions, n_states)

    alpha_list = []
    gamma_list = []
    r2_list    = []

    for i in range(n_traj):
        traj = trajectories[i]
        # Use steps 0..n_steps-2 so that Q_{t+1} is always available in the array
        T = min(len(traj) - 1, n_steps - 1)

        ys  = []   # ΔQ(s_t, a_t)
        x1s = []   # r_t - Q_t(s_t, a_t)          coefficient: α
        x2s = []   # max_{a'} Q_t(s'_t, a')        coefficient: α*γ

        for t in range(T):
            tr    = traj[t]
            s, a  = tr['s'], tr['a']
            r     = tr['r']
            s_nxt = tr['s_next']

            Q_sa_t    = float(q_pred_r[i, t,     a, s])
            Q_sa_tp1  = float(q_pred_r[i, t + 1, a, s])
            max_Q_nxt = float(q_pred_r[i, t, :, s_nxt].max())

            ys.append(Q_sa_tp1 - Q_sa_t)
            x1s.append(r - Q_sa_t)
            x2s.append(max_Q_nxt)

        y = np.array(ys,  dtype=np.float64)
        X = np.stack([x1s, x2s], axis=1).astype(np.float64)

        # Unconstrained OLS — we want to see the transformer's actual TD-like
        # dynamics, including cases where the best fit has negative α or γ
        # (which means the trajectory is not consistent with canonical
        # Q-learning). Quality is reported via R² downstream.
        betas, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        alpha_e = float(betas[0])
        # γ = (α·γ) / α; only well-defined when α is far from zero.
        gamma_e = float(betas[1] / alpha_e) if abs(alpha_e) > 1e-4 else float('nan')

        y_pred  = X @ betas
        ss_res  = float(np.sum((y - y_pred) ** 2))
        ss_tot  = float(np.sum((y - y.mean()) ** 2)) + 1e-12
        r2      = 1.0 - ss_res / ss_tot

        alpha_list.append(alpha_e)
        gamma_list.append(gamma_e)
        r2_list.append(r2)

    return np.array(alpha_list), np.array(gamma_list), np.array(r2_list)


def plot_effective_alpha_gamma(
    alpha_eff: np.ndarray,
    gamma_eff: np.ndarray,
    r2_vals: np.ndarray,
    expert_alpha: float,
    expert_gamma: float,
    save_path: str,
) -> None:
    # Unconstrained OLS — show the full distribution, no clipping. Out-of-range
    # fits (α<0, γ<0, γ>1) are real signal: they say the transformer's TD-like
    # dynamics on that trajectory are not consistent with canonical Q-learning.
    valid = np.isfinite(alpha_eff) & np.isfinite(gamma_eff)
    a = alpha_eff[valid]
    g = gamma_eff[valid]
    r = r2_vals[valid]

    n_total   = int(valid.sum())
    in_region = (a >= 0) & (a <= 1) & (g >= 0) & (g <= 1)
    n_in_box  = int(in_region.sum())
    frac_box  = n_in_box / max(n_total, 1)

    fig = plt.figure(figsize=(13, 5))
    gs = fig.add_gridspec(1, 3, width_ratios=[2, 1, 1], wspace=0.35)
    ax_scatter = fig.add_subplot(gs[0])
    ax_alpha = fig.add_subplot(gs[1])
    ax_gamma = fig.add_subplot(gs[2])

    sc = ax_scatter.scatter(a, g, c=r, cmap='plasma', alpha=0.7, s=30, vmin=0, vmax=1)
    plt.colorbar(sc, ax=ax_scatter, label='Regression R²')

    # Shade the canonical Q-learning region [0,1]×[0,1] so the eye can tell
    # apart "transformer-like-Q-learning" fits from the rest.
    ax_scatter.axvspan(0, 1, ymin=0, ymax=1, alpha=0.0)  # placeholder; use Rectangle for clarity
    from matplotlib.patches import Rectangle
    ax_scatter.add_patch(Rectangle(
        (0, 0), 1, 1, linewidth=1.2, edgecolor='black', facecolor='black',
        alpha=0.06, zorder=0, label=f'Valid Q-learning region [0,1]² ({frac_box:.0%} of fits)'
    ))
    ax_scatter.axhline(0, color='gray', linewidth=0.8, linestyle=':', alpha=0.7)
    ax_scatter.axvline(0, color='gray', linewidth=0.8, linestyle=':', alpha=0.7)

    ax_scatter.axvline(expert_alpha, color='red', linewidth=2, linestyle='--',
                       label=f'Expert α = {expert_alpha}')
    ax_scatter.axhline(expert_gamma, color='blue', linewidth=2, linestyle='--',
                       label=f'Expert γ = {expert_gamma}')
    ax_scatter.set_xlabel('α_eff')
    ax_scatter.set_ylabel('γ_eff')
    ax_scatter.set_title('Fitted (α_eff, γ_eff) per trajectory (unconstrained OLS)')
    ax_scatter.legend(fontsize=8, loc='best')
    ax_scatter.grid(True, alpha=0.3)

    ax_alpha.hist(a, bins=20, color='steelblue', edgecolor='white', alpha=0.8)
    ax_alpha.axvline(expert_alpha, color='red', linewidth=2, linestyle='--')
    ax_alpha.axvline(float(np.nanmedian(a)), color='navy', linewidth=1.5, linestyle=':',
                     label=f'Median={np.nanmedian(a):.3f}')
    ax_alpha.set_xlabel('α_eff')
    ax_alpha.set_title('α_eff distribution')
    ax_alpha.legend(fontsize=7)
    ax_alpha.grid(True, alpha=0.3)

    ax_gamma.hist(g, bins=20, color='darkorange', edgecolor='white', alpha=0.8)
    ax_gamma.axvline(expert_gamma, color='blue', linewidth=2, linestyle='--')
    ax_gamma.axvline(float(np.nanmedian(g)), color='darkred', linewidth=1.5, linestyle=':',
                     label=f'Median={np.nanmedian(g):.3f}')
    ax_gamma.set_xlabel('γ_eff')
    ax_gamma.set_title('γ_eff distribution')
    ax_gamma.legend(fontsize=7)
    ax_gamma.grid(True, alpha=0.3)

    fig.suptitle('Effective Q-Learning Hyperparameters Recovered from Context Probe', fontsize=12)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_training_curves(log_path: str, save_path: str) -> None:
    if not os.path.exists(log_path):
        print(f"  Warning: training log not found at {log_path}, skipping.")
        return
    data = np.load(log_path)
    steps = data['steps']
    train_ce = data['train_ce']
    train_acc = data['train_acc']
    val_ce = data['val_ce']
    val_acc = data['val_acc']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(steps, train_ce, color='steelblue', linewidth=1.5, label='Train CE')
    ax1.plot(steps, val_ce, color='darkorange', linewidth=1.5, label='Val CE')
    ax1.set_xlabel('Step')
    ax1.set_ylabel('CE Loss')
    ax1.set_title('CE Loss Over Training')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(steps, np.array(train_acc) * 100, color='steelblue', linewidth=1.5, label='Train Acc')
    ax2.plot(steps, np.array(val_acc) * 100, color='darkorange', linewidth=1.5, label='Val Acc')
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Accuracy Over Training')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.suptitle('Recurrent Context Q-Learning — Training Curves', fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


# ---------------------------------------------------------------------------
# Part 6: Reward probe from context delta
# ---------------------------------------------------------------------------

def collect_reward_probe_data(
    model: COCONUTTransformer,
    n_trajectories: int,
    n_states: int,
    n_actions: int,
    vocab: Dict,
    config: COCONUTConfig,
    device: torch.device,
    n_steps: int = 50,
    seed_offset: int = 40000,
) -> Tuple[np.ndarray, np.ndarray]:
    """Collect (Δc_{a_t}, r_t) pairs per step.

    Δc_{a_t} = update_hidden - context[a_t]   (residual written into the
    chosen action's slot at step t). One pair per (trajectory, step).
    """
    model.eval()
    d_model = config.d_model

    N = n_trajectories * n_steps
    delta_all = np.empty((N, d_model), dtype=np.float32)
    r_all = np.empty(N, dtype=np.float32)
    write_idx = 0

    with torch.no_grad():
        for i in range(n_trajectories):
            seed = seed_offset + i
            P, R = generate_eval_mdp(n_states, n_actions, seed=seed)
            trajectory, _ = run_tabular_q_learning(
                P, R, n_states, n_actions, n_steps=n_steps, seed=seed
            )

            context = model.get_init_context(1, n_actions, device)

            for t, tr in enumerate(trajectory):
                token_list, r_off, s_off, u_off = build_step_tokens(tr, vocab, n_actions)
                token_ids = torch.tensor([token_list], dtype=torch.long, device=device)
                reward_val = torch.tensor([tr['r']], dtype=torch.float32, device=device)

                _, update_hidden = model.forward_step(
                    token_ids=token_ids,
                    reward_value=reward_val,
                    reward_offset=r_off,
                    select_offset=s_off,
                    update_offset=u_off,
                    context=context,
                )

                a_t = tr['a']
                # Probe the residual actually written into the slot. In discrete
                # mode this is the token-embedding delta the channel transmits,
                # not the raw hidden state; in continuous mode contextualize is
                # the identity, so this matches the original behavior exactly.
                context_write = model.contextualize(update_hidden)
                prev_slot = context[0, a_t, :].detach().cpu().numpy()
                new_slot = context_write[0].detach().cpu().numpy()
                delta_all[write_idx] = new_slot - prev_slot
                r_all[write_idx] = tr['r']
                write_idx += 1

                new_context = context.clone()
                new_context[0, a_t, :] = context_write[0]
                context = new_context

                del token_ids, reward_val, update_hidden

    return delta_all, r_all


class RewardProbe(nn.Module):
    """Linear probe: Δc in R^{d_model} -> scalar reward r."""
    def __init__(self, d_model: int):
        super().__init__()
        self.linear = nn.Linear(d_model, 1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.linear(h).squeeze(-1)


def train_reward_probe(
    probe: RewardProbe,
    delta_all: np.ndarray,
    r_all: np.ndarray,
    device: torch.device,
    n_epochs: int = 10,
    batch_size: int = 256,
    lr: float = 1e-3,
) -> List[float]:
    probe.train()
    optimizer = torch.optim.Adam(probe.parameters(), lr=lr)
    N = delta_all.shape[0]
    losses = []

    d_t = torch.tensor(delta_all, dtype=torch.float32, device=device)
    r_t = torch.tensor(r_all, dtype=torch.float32, device=device)

    for _ in range(n_epochs):
        perm = torch.randperm(N, device=device)
        epoch_loss = 0.0
        n_batches = 0
        for i in range(0, N, batch_size):
            idx = perm[i:i + batch_size]
            pred = probe(d_t[idx])
            loss = F.mse_loss(pred, r_t[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        losses.append(epoch_loss / max(n_batches, 1))

    return losses


def evaluate_reward_probe(
    probe: RewardProbe,
    delta_all: np.ndarray,
    r_all: np.ndarray,
    device: torch.device,
) -> Tuple[float, float, np.ndarray]:
    probe.eval()
    with torch.no_grad():
        d_t = torch.tensor(delta_all, dtype=torch.float32, device=device)
        r_pred = probe(d_t).cpu().numpy()

    ss_res = float(np.sum((r_all - r_pred) ** 2))
    ss_tot = float(np.sum((r_all - r_all.mean()) ** 2)) + 1e-12
    r2 = float(1.0 - ss_res / ss_tot)
    mae = float(np.mean(np.abs(r_all - r_pred)))
    return r2, mae, r_pred


def plot_reward_probe(
    r_true: np.ndarray,
    r_pred: np.ndarray,
    r2: float,
    mae: float,
    save_path: str,
) -> None:
    rng = np.random.default_rng(0)
    n = len(r_true)
    idx = rng.choice(n, size=min(10000, n), replace=False)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    ax1.scatter(r_true[idx], r_pred[idx], alpha=0.2, s=6,
                color='seagreen', rasterized=True)
    lo = float(min(r_true.min(), r_pred.min()))
    hi = float(max(r_true.max(), r_pred.max()))
    ax1.plot([lo, hi], [lo, hi], 'r--', linewidth=1.5, label='y = x')
    ax1.set_xlabel('True reward r_t')
    ax1.set_ylabel('Predicted reward (linear probe on Δc)')
    ax1.set_title(f'Reward Probe Scatter\nR² = {r2:.4f}  MAE = {mae:.4f}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    residuals = r_pred - r_true
    ax2.hist(residuals, bins=40, color='seagreen', edgecolor='white', alpha=0.85)
    ax2.axvline(0, color='red', linewidth=1.2, linestyle='--')
    ax2.set_xlabel('Residual (predicted − true)')
    ax2.set_ylabel('Count')
    ax2.set_title(f'Residual Distribution\nmean={residuals.mean():+.4f}  std={residuals.std():.4f}')
    ax2.grid(True, alpha=0.3)

    fig.suptitle('Reward Probe from Context Delta Δc_{a_t}', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def trace_step_log(
    model: COCONUTTransformer,
    q_probe: 'ContextQProbe',
    r_probe: 'RewardProbe',
    n_states: int,
    n_actions: int,
    vocab: Dict,
    config: COCONUTConfig,
    device: torch.device,
    seed: int,
    n_steps: int,
    alpha: float,
    gamma: float,
    epsilon: float,
    save_path: str,
) -> None:
    """Write a per-step text log comparing model context updates to the
    oracle tabular Q-learning update for a single MDP."""
    np.set_printoptions(precision=4, suppress=True, linewidth=140)

    P, R = generate_eval_mdp(n_states, n_actions, seed=seed)
    trajectory, q_snaps = run_tabular_q_learning(
        P, R, n_states, n_actions, n_steps=n_steps,
        alpha=alpha, gamma=gamma, epsilon=epsilon, seed=seed,
    )
    # q_snaps[t] is the Q table AFTER step t's oracle update.
    q_before_all = np.concatenate(
        [np.zeros((1, n_states, n_actions), dtype=np.float32), q_snaps[:-1]], axis=0,
    )

    lines: List[str] = []
    lines.append("=" * 90)
    lines.append(f"PER-STEP TRACE  |  seed={seed}  n_states={n_states}  "
                 f"n_actions={n_actions}  n_steps={n_steps}")
    lines.append(f"alpha={alpha}  gamma={gamma}  epsilon={epsilon}")
    lines.append("=" * 90)
    lines.append("")
    lines.append("MDP transition probabilities P[s, a, s']:")
    for s in range(n_states):
        for a in range(n_actions):
            lines.append(f"  P[s={s}, a={a}] = {P[s, a]}")
    lines.append("")
    lines.append("MDP reward matrix R[s, a]:")
    for s in range(n_states):
        lines.append(f"  R[s={s}] = {R[s]}")
    lines.append("")

    model.eval()
    context = model.get_init_context(1, n_actions, device)
    prev_delta_per_action: Dict[int, np.ndarray] = {}

    with torch.no_grad():
        for t, tr in enumerate(trajectory):
            s, a, r, s_next = tr['s'], tr['a'], tr['r'], tr['s_next']

            # ---- Oracle Q-learning update ----
            Q_before = q_before_all[t]
            Q_after = q_snaps[t]
            q_sa_before = float(Q_before[s, a])
            max_q_next = float(Q_before[s_next].max())
            td_target = r + gamma * max_q_next
            td_error = td_target - q_sa_before
            q_sa_after = float(Q_after[s, a])
            a_star_oracle = int(np.argmax(Q_after[s_next]))

            # ---- Model forward step ----
            token_list, r_off, s_off, u_off = build_step_tokens(tr, vocab, n_actions)
            token_ids = torch.tensor([token_list], dtype=torch.long, device=device)
            reward_val = torch.tensor([r], dtype=torch.float32, device=device)

            select_logits, update_hidden = model.forward_step(
                token_ids=token_ids,
                reward_value=reward_val,
                reward_offset=r_off,
                select_offset=s_off,
                update_offset=u_off,
                context=context,
            )
            if n_actions < config.max_actions:
                select_logits[:, n_actions:] = float('-inf')
            model_action = int(select_logits[0].argmax().item())

            context_write = model.contextualize(update_hidden)
            ctx_before_a = context[0, a, :].detach().cpu().numpy()
            ctx_after_a = context_write[0].detach().cpu().numpy()
            delta = ctx_after_a - ctx_before_a
            delta_norm = float(np.linalg.norm(delta))
            ctx_before_norm = float(np.linalg.norm(ctx_before_a))
            ctx_after_norm = float(np.linalg.norm(ctx_after_a))

            cos_prev = None
            if a in prev_delta_per_action:
                pd = prev_delta_per_action[a]
                denom = float(np.linalg.norm(pd) * delta_norm)
                if denom > 1e-12:
                    cos_prev = float(np.dot(pd, delta) / denom)
            prev_delta_per_action[a] = delta.copy()

            # ---- Probes ----
            delta_t = torch.from_numpy(delta).unsqueeze(0).to(device)
            r_hat = float(r_probe(delta_t).item())

            ctx_before_t = torch.from_numpy(ctx_before_a).unsqueeze(0).to(device)
            ctx_after_t = torch.from_numpy(ctx_after_a).unsqueeze(0).to(device)
            q_pred_before = q_probe(ctx_before_t)[0].cpu().numpy()  # (n_states,)
            q_pred_after = q_probe(ctx_after_t)[0].cpu().numpy()
            q_oracle_col_before = Q_before[:, a]
            q_oracle_col_after = Q_after[:, a]

            # ---- Write step block ----
            lines.append(f"--- Step {t:3d} ---")
            lines.append(f"  Transition: s={s}  a={a}  r={r:+.4f}  s'={s_next}")
            lines.append(f"  ORACLE Q-learning:")
            lines.append(f"    Q[s={s},a={a}] before = {q_sa_before:+.4f}")
            lines.append(f"    max_a Q[s'={s_next},a] = {max_q_next:+.4f}")
            lines.append(f"    TD target = r + gamma * max = {td_target:+.4f}")
            lines.append(f"    TD error  = target - Q_before = {td_error:+.4f}")
            lines.append(f"    Q[s={s},a={a}] after  = {q_sa_after:+.4f}  "
                         f"(delta = {q_sa_after - q_sa_before:+.4f})")
            lines.append(f"    argmax_a Q[s'={s_next}] = a*={a_star_oracle}")
            lines.append(f"  MODEL action choice this step: a_hat={model_action}  "
                         f"(oracle a*={tr['a_star']})")
            lines.append(f"  MODEL context update (slot a={a}):")
            lines.append(f"    ||c[a]|| before = {ctx_before_norm:.4f}   "
                         f"after = {ctx_after_norm:.4f}   ||delta_c|| = {delta_norm:.4f}")
            if cos_prev is not None:
                lines.append(f"    cosine(delta_c_t, delta_c_prev for same a) = {cos_prev:+.4f}")
            lines.append(f"    reward probe on delta_c: r_hat = {r_hat:+.4f}   "
                         f"(true r = {r:+.4f}   |err| = {abs(r_hat - r):.4f})")
            lines.append(f"    Q-probe on c[a={a}]   BEFORE: pred Q[:,{a}] = {q_pred_before}")
            lines.append(f"                                    oracle Q[:,{a}] = {q_oracle_col_before}")
            lines.append(f"    Q-probe on c[a={a}]   AFTER : pred Q[:,{a}] = {q_pred_after}")
            lines.append(f"                                    oracle Q[:,{a}] = {q_oracle_col_after}")
            lines.append("")

            # advance context
            new_context = context.clone()
            new_context[0, a, :] = context_write[0]
            context = new_context

    lines.append("=" * 90)
    lines.append("End of trace.")
    lines.append("=" * 90)

    with open(save_path, 'w') as f:
        f.write("\n".join(lines))
    print(f"  Wrote step-trace log to {save_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate trained recurrent context Q-learning transformer')
    parser.add_argument('--model', choices=paths.MODELS, default=None,
                        help='final model to evaluate; sets --checkpoint and --figures_dir '
                             '(figures/<model>/evaluation)')
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--figures_dir', type=str, default=None)
    parser.add_argument('--label', type=str, default=None,
                        help="Human label for this run used in figure titles/"
                             "legends (defaults to the checkpoint's context_mode, "
                             "e.g. 'continuous' or 'discrete').")
    parser.add_argument('--n_steps',       type=int,   default=50)
    parser.add_argument('--long_horizon_steps', type=int, default=200,
                        help='Steps for the long-horizon evaluation that '
                             'tests behavior past the training horizon.')
    parser.add_argument('--nonstationary_switch_step', type=int, default=50,
                        help='Steps in the first phase before the MDP changes '
                             '(0 disables the nonstationary evaluation).')
    parser.add_argument('--nonstationary_post_steps', type=int, default=100,
                        help='Steps to keep running after the switch.')
    parser.add_argument('--nonstationary_window', type=int, default=10,
                        help='Trailing window for the reward-rate curves.')
    parser.add_argument('--reward_intervention_steps', type=int, default=200,
                        help='Closed-loop steps per reward-intervention '
                             'condition (0 disables Part 4e).')
    parser.add_argument('--size_sweep_steps', type=int, default=200,
                        help='Closed-loop steps per |S|x|A| cell in the size '
                             'sweep (0 disables Part 4f).')
    parser.add_argument('--parts', type=str, default='all',
                        help="Which parts to run: 'all' (default, full "
                             "pipeline) or a comma-separated subset of "
                             "'legacy' (Parts 1-6b: agreement, probes, "
                             "attention, regret, nonstationary, traces), "
                             "'reward_intervention' (4e), 'size_sweep' (4f). "
                             "Lets the newer evals be regenerated without "
                             "re-running the expensive probe/attention parts.")
    parser.add_argument('--alpha',         type=float, default=0.1)
    parser.add_argument('--gamma',         type=float, default=0.9)
    parser.add_argument('--epsilon',       type=float, default=0.2)
    parser.add_argument('--eval_seed',     type=int,   default=9999)
    parser.add_argument('--n_eval_mdps',   type=int,   default=10)
    parser.add_argument('--n_probe_train', type=int,   default=1000)
    parser.add_argument('--n_probe_eval',  type=int,   default=100)
    parser.add_argument('--probe_epochs',  type=int,   default=10)
    args = parser.parse_args()
    global TF_EPS
    TF_EPS = args.epsilon
    if args.model:
        args.checkpoint = args.checkpoint or str(paths.checkpoint(args.model))
        args.figures_dir = args.figures_dir or str(paths.FIGURES / args.model / 'evaluation')
        args.label = args.label or args.model
    if args.checkpoint is None or args.figures_dir is None:
        parser.error('give --model, or both --checkpoint and --figures_dir')

    os.makedirs(args.figures_dir, exist_ok=True)

    # ---- Load checkpoint ----
    print(f"Loading checkpoint from {args.checkpoint} ...")
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    config = COCONUTConfig.from_dict(ckpt['config'])
    model = COCONUTTransformer(config)
    model.load_state_dict(ckpt['model_state_dict'])

    print(f"  Loaded epoch={ckpt.get('epoch', '?')}, step={ckpt.get('step', '?')}, "
          f"val_ce={ckpt.get('val_ce_loss', ckpt.get('val_loss', '?')):.4f}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    vocab = build_vocab(config.max_states, config.max_actions)
    n_states = config.max_states
    n_actions = config.max_actions

    run_label = args.label if args.label else getattr(config, 'context_mode', 'continuous')

    print(f"\nModel: {model.num_parameters():,} params  |  device: {device}")
    print(f"MDP:   n_states={n_states}, n_actions={n_actions}")
    print(f"FFNs:  {'enabled' if config.use_ffns else 'DISABLED'}")
    print(f"Context mode: {getattr(config, 'context_mode', 'continuous')}  |  label: {run_label}")

    eval_seeds = list(range(args.eval_seed, args.eval_seed + args.n_eval_mdps))

    _parts = {p.strip() for p in args.parts.split(',') if p.strip()}
    unknown = _parts - {'all', 'legacy', 'reward_intervention', 'size_sweep'}
    if unknown:
        raise SystemExit(f"Unknown --parts value(s): {sorted(unknown)}")
    want = lambda part: ('all' in _parts) or (part in _parts)
    print(f"Parts: {args.parts}")

    # -----------------------------------------------------------------------
    # Part 4e / 4f: reviewer-response evals.
    #
    # These run first so they can be regenerated on their own with
    # `--parts reward_intervention,size_sweep`, which skips the probe and
    # attention parts below (those dominate runtime and are unchanged).
    # -----------------------------------------------------------------------
    if want('reward_intervention') and args.reward_intervention_steps > 0:
        run_reward_intervention_eval(model, config, vocab, n_states, n_actions,
                                     device, args, eval_seeds, run_label,
                                     args.figures_dir)
    if want('size_sweep') and args.size_sweep_steps > 0:
        run_size_sweep_eval(model, config, vocab, device, args, eval_seeds,
                            run_label, args.figures_dir)

    if not want('legacy'):
        print(f"\nDone. Figures saved to {args.figures_dir}/ "
              f"(legacy parts skipped via --parts).")
        return

    # -----------------------------------------------------------------------
    # Part 1a: In-distribution action prediction
    # -----------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print(f"Part 1a: In-distribution action prediction ({args.n_eval_mdps} MDPs)")
    print(f"{'=' * 60}")

    all_agreements_id = []
    traj_preds_id = []
    for seed_i, seed in enumerate(eval_seeds):
        print(f"  ID MDP {seed_i+1}/{args.n_eval_mdps} (seed={seed}) ...", end=' ', flush=True)
        P, R = generate_eval_mdp(n_states, n_actions, seed=seed)
        trajectory, q_snaps = run_tabular_q_learning(
            P, R, n_states, n_actions, n_steps=args.n_steps,
            alpha=args.alpha, gamma=args.gamma, epsilon=args.epsilon, seed=seed,
        )
        preds, _ = run_action_inference(model, trajectory, vocab, n_actions,
                                        config, device)
        targets = np.array([step['a_star'] for step in trajectory], dtype=np.int32)
        agree = (preds == targets).astype(np.float32)
        all_agreements_id.append(agree)
        traj_preds_id.append((trajectory, preds))
        print(f"agree={agree.mean():.2%}")

    aa_id_arr = np.stack(all_agreements_id, axis=0)
    aa_id_mean = aa_id_arr.mean(axis=0)
    aa_id_std = aa_id_arr.std(axis=0)
    print(f"\n  Mean ID agreement: "
          f"{aa_id_mean.mean():.2%} +/- {aa_id_arr.mean(axis=1).std():.4f}")

    # -----------------------------------------------------------------------
    # Part 1b: OOD variants
    # -----------------------------------------------------------------------
    ood_plot_results: List[Tuple[str, np.ndarray, np.ndarray]] = []
    ood_agreements: List[Tuple[str, np.ndarray]] = []

    for v_idx, (variant, variant_desc) in enumerate(OOD_VARIANTS):
        v_seeds = list(range(
            args.eval_seed + 50000 + v_idx * 10000,
            args.eval_seed + 50000 + v_idx * 10000 + args.n_eval_mdps,
        ))
        print(f"\n{'=' * 60}")
        print(f"Part 1b-{v_idx+1}: OOD '{variant}' — {variant_desc}")
        print(f"{'=' * 60}")

        all_agreements_v = []
        for seed_i, seed in enumerate(v_seeds):
            print(f"  {variant} {seed_i+1}/{args.n_eval_mdps} ...", end=' ', flush=True)
            P, R = generate_ood_mdp(n_states, n_actions, variant=variant, seed=seed)
            trajectory, q_snaps_v = run_tabular_q_learning(
                P, R, n_states, n_actions, n_steps=args.n_steps,
                alpha=args.alpha, gamma=args.gamma, epsilon=args.epsilon, seed=seed,
            )
            preds, _ = run_action_inference(model, trajectory, vocab, n_actions,
                                            config, device)
            targets = np.array([step['a_star'] for step in trajectory], dtype=np.int32)
            agree = (preds == targets).astype(np.float32)
            all_agreements_v.append(agree)
            print(f"agree={agree.mean():.2%}")

        aa_v_arr = np.stack(all_agreements_v, axis=0)
        aa_v_mean = aa_v_arr.mean(axis=0)
        aa_v_std = aa_v_arr.std(axis=0)
        gap = aa_id_mean.mean() - aa_v_mean.mean()
        print(f"\n  Mean {variant}: "
              f"{aa_v_mean.mean():.2%} +/- {aa_v_arr.mean(axis=1).std():.4f} (ID gap: {gap:+.4f})")
        ood_plot_results.append((variant_desc, aa_v_mean, aa_v_std))
        ood_agreements.append((variant, aa_v_arr))

    plot_action_agreement(
        aa_id_mean, aa_id_std, ood_plot_results,
        save_path=os.path.join(args.figures_dir, 'action_agreement.png'),
        n_mdps=args.n_eval_mdps, label_suffix=f'{run_label} context',
    )

    # -----------------------------------------------------------------------
    # Part 2: Context token Q-value probing
    # -----------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("Part 2: Context token Q-value probing")
    print(f"{'=' * 60}")

    for p in model.parameters():
        p.requires_grad_(False)

    print(f"\nCollecting probe training data ({args.n_probe_train} trajectories) ...")
    ctx_tr, q_tr, _ = collect_context_probe_data(
        model, args.n_probe_train, n_states, n_actions, vocab,
        config, device, n_steps=args.n_steps, seed_offset=20000,
    )
    print(f"  ctx: {ctx_tr.shape}, q_target: {q_tr.shape}")

    print(f"\nCollecting probe eval data ({args.n_probe_eval} trajectories) ...")
    ctx_ev, q_ev, traj_ev = collect_context_probe_data(
        model, args.n_probe_eval, n_states, n_actions, vocab,
        config, device, n_steps=args.n_steps, seed_offset=30000,
    )

    print(f"\nTraining context Q-probe ...")
    probe = ContextQProbe(config.d_model, n_states).to(device)
    losses = train_probe(probe, ctx_tr, q_tr, device, n_epochs=args.probe_epochs)
    print(f"  MSE per epoch: {['%.4f' % l for l in losses]}")

    r2, frob_mean, q_pred = evaluate_probe(probe, ctx_ev, q_ev, device)
    print(f"  Eval: R2={r2:.4f}  Frobenius={frob_mean:.4f}")

    plot_probe_scatter(
        q_ev, q_pred, r2=r2,
        save_path=os.path.join(args.figures_dir, 'probe_scatter.png'),
    )
    plot_probe_frobenius(
        q_pred, q_ev, n_steps=args.n_steps, n_traj=args.n_probe_eval,
        n_actions=n_actions,
        save_path=os.path.join(args.figures_dir, 'probe_frobenius.png'),
    )

    # Bias-free probe: maps zero context exactly to zero, removing the
    # cosmetic step-0 offset visible in the standard probe plots.
    print(f"\nTraining bias-free context Q-probe ...")
    probe_nb = ContextQProbe(config.d_model, n_states, bias=False).to(device)
    losses_nb = train_probe(probe_nb, ctx_tr, q_tr, device, n_epochs=args.probe_epochs)
    print(f"  MSE per epoch: {['%.4f' % l for l in losses_nb]}")

    r2_nb, frob_mean_nb, q_pred_nb = evaluate_probe(probe_nb, ctx_ev, q_ev, device)
    print(f"  Eval (no bias): R2={r2_nb:.4f}  Frobenius={frob_mean_nb:.4f}")

    plot_probe_scatter(
        q_ev, q_pred_nb, r2=r2_nb,
        save_path=os.path.join(args.figures_dir, 'probe_scatter_nobias.png'),
    )
    plot_probe_frobenius(
        q_pred_nb, q_ev, n_steps=args.n_steps, n_traj=args.n_probe_eval,
        n_actions=n_actions,
        save_path=os.path.join(args.figures_dir, 'probe_frobenius_nobias.png'),
    )

    # -----------------------------------------------------------------------
    # Part 3: Attention heatmap
    # -----------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("Attention heatmap (SELECT & UPDATE)")
    print(f"{'=' * 60}")
    _attn_seed = eval_seeds[0]
    P_attn, R_attn = generate_eval_mdp(n_states, n_actions, seed=_attn_seed)
    traj_attn, _ = run_tabular_q_learning(
        P_attn, R_attn, n_states, n_actions,
        n_steps=args.n_steps, alpha=args.alpha, gamma=args.gamma,
        epsilon=args.epsilon, seed=_attn_seed,
    )
    plot_attention_heatmap(
        model, traj_attn, vocab, n_actions, config, device,
        save_path=os.path.join(args.figures_dir, 'attention_heatmap.png'),
    )

    # ---- Full T×T causal attention on a fixed 4-state, 2-action sequence ----
    fixed_S, fixed_A = 4, 2
    if config.max_states >= fixed_S and config.max_actions >= fixed_A:
        print(f"\n  Full causal-attention heatmap on fixed "
              f"{fixed_S}-state / {fixed_A}-action MDP ...")
        P_fix, R_fix, mdp_desc = generate_linear_chain_mdp(fixed_S, fixed_A)
        traj_fix, _ = run_tabular_q_learning(
            P_fix, R_fix, fixed_S, fixed_A,
            n_steps=args.n_steps, alpha=args.alpha, gamma=args.gamma,
            epsilon=args.epsilon, seed=_attn_seed + 1,
        )
        plot_full_attention_heatmap(
            model, traj_fix, vocab, fixed_A, config, device,
            save_path=os.path.join(args.figures_dir,
                                   'attention_full_heatmap_4s2a.png'),
            mdp_description=mdp_desc,
        )
    else:
        print(f"\n  Skipping full-attention heatmap: model max_states="
              f"{config.max_states}, max_actions={config.max_actions} "
              f"< required {fixed_S}/{fixed_A}.")

    # -----------------------------------------------------------------------
    # Per-distribution agreement heatmap
    # -----------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("Per-distribution agreement heatmap")
    print(f"{'=' * 60}")
    dist_data: List[Tuple[str, np.ndarray]] = [('ID (Beta 2,2)', aa_id_arr)]
    for variant, aa_v in ood_agreements:
        dist_data.append((DIST_SHORT_LABELS.get(variant, variant), aa_v))
    plot_per_distribution_agreement(
        dist_data, args.n_steps,
        os.path.join(args.figures_dir, 'per_state_agreement.png'),
        n_bins=3,
    )
    plot_per_distribution_agreement_std(
        dist_data, args.n_steps,
        os.path.join(args.figures_dir, 'per_state_agreement_std.png'),
        n_bins=3,
    )

    # -----------------------------------------------------------------------
    # Part 4: Regret comparison (autonomous transformer vs Q-learners)
    # -----------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print(f"Part 4: Regret comparison ({args.n_eval_mdps} MDPs, autonomous)")
    print(f"{'=' * 60}")

    all_cumrew_transformer = []
    all_cumrew_transformer_eps = []
    all_cumrew_greedy = []
    all_cumrew_epsgreedy = []
    all_cumrew_optimal = []

    for seed_i, seed in enumerate(eval_seeds):
        print(f"  Regret MDP {seed_i+1}/{args.n_eval_mdps} (seed={seed}) ...", flush=True)
        P, R = generate_eval_mdp(n_states, n_actions, seed=seed)

        rng_t = np.random.default_rng(seed + 100)
        rng_g = np.random.default_rng(seed + 100)
        rng_e = np.random.default_rng(seed + 100)
        rng_o = np.random.default_rng(seed + 100)

        rew_t = run_transformer_autonomous(
            model, P, R, n_states, n_actions, args.n_steps,
            vocab, config, device, epsilon=0.0, rng=rng_t,
        )
        rew_te = run_transformer_autonomous(
            model, P, R, n_states, n_actions, args.n_steps,
            vocab, config, device, epsilon=args.epsilon, rng=np.random.default_rng(seed + 100),
        )
        rew_g = run_q_learner_autonomous(
            P, R, n_states, n_actions, args.n_steps,
            alpha=args.alpha, gamma=args.gamma, epsilon=0.0, rng=rng_g,
        )
        rew_e = run_q_learner_autonomous(
            P, R, n_states, n_actions, args.n_steps,
            alpha=args.alpha, gamma=args.gamma, epsilon=args.epsilon, rng=rng_e,
        )
        rew_o = run_optimal_autonomous(
            P, R, n_states, n_actions, args.n_steps,
            gamma=args.gamma, rng=rng_o,
        )

        all_cumrew_transformer.append(np.cumsum(rew_t))
        all_cumrew_transformer_eps.append(np.cumsum(rew_te))
        all_cumrew_greedy.append(np.cumsum(rew_g))
        all_cumrew_epsgreedy.append(np.cumsum(rew_e))
        all_cumrew_optimal.append(np.cumsum(rew_o))

    cumrew_t = np.stack(all_cumrew_transformer, axis=0)
    cumrew_te = np.stack(all_cumrew_transformer_eps, axis=0)
    cumrew_g = np.stack(all_cumrew_greedy, axis=0)
    cumrew_e = np.stack(all_cumrew_epsgreedy, axis=0)
    cumrew_o = np.stack(all_cumrew_optimal, axis=0)

    print(f"\n  Final cumulative reward (mean over {args.n_eval_mdps} MDPs):")
    print(f"    Optimal:        {cumrew_o[:, -1].mean():.2f} ± {cumrew_o[:, -1].std():.2f}")
    print(f"    Transformer:    {cumrew_t[:, -1].mean():.2f} ± {cumrew_t[:, -1].std():.2f}")
    print(f"    Transformer, ε: {cumrew_te[:, -1].mean():.2f} ± {cumrew_te[:, -1].std():.2f}")
    print(f"    Greedy Q (ε=0): {cumrew_g[:, -1].mean():.2f} ± {cumrew_g[:, -1].std():.2f}")
    print(f"    ε-greedy Q:     {cumrew_e[:, -1].mean():.2f} ± {cumrew_e[:, -1].std():.2f}")

    plot_regret(
        cumrew_t, cumrew_g, cumrew_e,
        n_mdps=args.n_eval_mdps,
        save_path=os.path.join(args.figures_dir, 'regret.png'),
        cumrew_transformer_eps=cumrew_te,
    )

    plot_combined_row(
        cumrew_t, cumrew_g, cumrew_e,
        n_mdps=args.n_eval_mdps,
        q_true=q_ev, q_pred=q_pred, r2=r2,
        dist_data=dist_data, n_steps=args.n_steps,
        save_path=os.path.join(args.figures_dir, 'combined_row.png'),
        n_bins=3, cumrew_transformer_eps=cumrew_te,
    )
    # raw data of the combined row, for paper/make_main_figures.py
    np.savez(os.path.join(args.figures_dir, 'combined_row_data.npz'),
             cumrew_tf=cumrew_t, cumrew_tf_eps=cumrew_te, cumrew_greedy=cumrew_g,
             cumrew_epsgreedy=cumrew_e, q_true=q_ev, q_pred=q_pred, r2=r2,
             dist_labels=np.array([l for l, _ in dist_data]),
             **{f'agree_{i}': a for i, (_, a) in enumerate(dist_data)})

    # -----------------------------------------------------------------------
    # Part 4b: Long-horizon evaluation (past the training horizon)
    # -----------------------------------------------------------------------
    if args.long_horizon_steps > args.n_steps:
        print(f"\n{'=' * 60}")
        print(f"Part 4b: Long-horizon eval "
              f"({args.long_horizon_steps} steps, train horizon={args.n_steps})")
        print(f"{'=' * 60}")
        long_t, long_te, long_g, long_e = [], [], [], []
        for seed_i, seed in enumerate(eval_seeds):
            print(f"  Long-horizon MDP {seed_i+1}/{args.n_eval_mdps} "
                  f"(seed={seed}) ...", flush=True)
            P, R = generate_eval_mdp(n_states, n_actions, seed=seed)
            rng_t = np.random.default_rng(seed + 100)
            rng_g = np.random.default_rng(seed + 100)
            rng_e = np.random.default_rng(seed + 100)
            rew_t = run_transformer_autonomous(
                model, P, R, n_states, n_actions, args.long_horizon_steps,
                vocab, config, device, epsilon=0.0, rng=rng_t,
            )
            long_te.append(np.cumsum(run_transformer_autonomous(
                model, P, R, n_states, n_actions, args.long_horizon_steps,
                vocab, config, device, epsilon=args.epsilon,
                rng=np.random.default_rng(seed + 100))))
            rew_g = run_q_learner_autonomous(
                P, R, n_states, n_actions, args.long_horizon_steps,
                alpha=args.alpha, gamma=args.gamma, epsilon=0.0,
                rng=rng_g,
            )
            rew_e = run_q_learner_autonomous(
                P, R, n_states, n_actions, args.long_horizon_steps,
                alpha=args.alpha, gamma=args.gamma, epsilon=args.epsilon,
                rng=rng_e,
            )
            long_t.append(np.cumsum(rew_t))
            long_g.append(np.cumsum(rew_g))
            long_e.append(np.cumsum(rew_e))

        plot_long_horizon(
            np.stack(long_t, axis=0),
            np.stack(long_g, axis=0),
            np.stack(long_e, axis=0),
            train_horizon=args.n_steps,
            n_mdps=args.n_eval_mdps,
            save_path=os.path.join(args.figures_dir, 'long_horizon.png'),
            cumrew_transformer_eps=np.stack(long_te, axis=0),
        )

    # -----------------------------------------------------------------------
    # Part 4c: Cumulative reward across reward distributions
    # -----------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print(f"Part 4c: Cumulative reward by reward distribution "
          f"({len(REWARD_DISTRIBUTIONS)} dists × {args.n_eval_mdps} MDPs)")
    print(f"{'=' * 60}")

    rdist_results: List[Dict] = []
    for dist_name, dist_label in REWARD_DISTRIBUTIONS:
        print(f"\n  Reward dist: {dist_name} — {dist_label}")
        cum_t, cum_te, cum_g, cum_e, cum_o = [], [], [], [], []
        for seed_i, seed in enumerate(eval_seeds):
            print(f"    MDP {seed_i+1}/{args.n_eval_mdps} (seed={seed}) ...",
                  flush=True)
            P, R = generate_reward_dist_mdp(n_states, n_actions, dist_name,
                                            seed=seed)
            rng_t = np.random.default_rng(seed + 100)
            rng_g = np.random.default_rng(seed + 100)
            rng_e = np.random.default_rng(seed + 100)
            rng_o = np.random.default_rng(seed + 100)

            rew_t = run_transformer_autonomous(
                model, P, R, n_states, n_actions, args.n_steps,
                vocab, config, device, epsilon=0.0, rng=rng_t,
            )
            cum_te.append(np.cumsum(run_transformer_autonomous(
                model, P, R, n_states, n_actions, args.n_steps,
                vocab, config, device, epsilon=args.epsilon,
                rng=np.random.default_rng(seed + 100))))
            rew_g = run_q_learner_autonomous(
                P, R, n_states, n_actions, args.n_steps,
                alpha=args.alpha, gamma=args.gamma, epsilon=0.0, rng=rng_g,
            )
            rew_e = run_q_learner_autonomous(
                P, R, n_states, n_actions, args.n_steps,
                alpha=args.alpha, gamma=args.gamma,
                epsilon=args.epsilon, rng=rng_e,
            )
            rew_o = run_optimal_autonomous(
                P, R, n_states, n_actions, args.n_steps,
                gamma=args.gamma, rng=rng_o,
            )
            cum_t.append(np.cumsum(rew_t))
            cum_g.append(np.cumsum(rew_g))
            cum_e.append(np.cumsum(rew_e))
            cum_o.append(np.cumsum(rew_o))

        ct = np.stack(cum_t, axis=0)
        cg = np.stack(cum_g, axis=0)
        ce = np.stack(cum_e, axis=0)
        co = np.stack(cum_o, axis=0)
        print(f"    Final cumulative reward (mean ± std):")
        print(f"      Optimal:     {co[:, -1].mean():.2f} ± {co[:, -1].std():.2f}")
        print(f"      Transformer: {ct[:, -1].mean():.2f} ± {ct[:, -1].std():.2f}")
        print(f"      Greedy Q:    {cg[:, -1].mean():.2f} ± {cg[:, -1].std():.2f}")
        print(f"      ε-greedy Q:  {ce[:, -1].mean():.2f} ± {ce[:, -1].std():.2f}")

        rdist_results.append({
            'name': dist_name, 'label': dist_label,
            'cumrew_t': ct, 'cumrew_te': np.stack(cum_te, axis=0), 'cumrew_g': cg,
            'cumrew_e': ce, 'cumrew_o': co,
        })

    plot_reward_dist_grid(
        rdist_results, n_mdps=args.n_eval_mdps,
        save_path=os.path.join(args.figures_dir, 'cumrew_by_reward_dist.png'),
    )

    # -----------------------------------------------------------------------
    # Part 4d: Nonstationary MDPs (the world changes mid-episode)
    # -----------------------------------------------------------------------
    if args.nonstationary_switch_step > 0 and args.nonstationary_post_steps > 0:
        run_nonstationary_eval(model, config, vocab, n_states, n_actions,
                               device, args, eval_seeds, run_label,
                               args.figures_dir)
    else:
        print("\nPart 4d: Nonstationary eval skipped "
              "(--nonstationary_switch_step/--nonstationary_post_steps = 0).")

    # -----------------------------------------------------------------------
    # Part 5: Effective alpha/gamma recovery
    # -----------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("Part 5: Effective alpha/gamma recovery")
    print(f"{'=' * 60}")

    alpha_eff, gamma_eff, r2_vals = estimate_effective_alpha_gamma(
        probe, ctx_ev, traj_ev, device,
        n_steps=args.n_steps, n_traj=args.n_probe_eval, n_actions=n_actions,
    )
    valid_mask = np.isfinite(alpha_eff) & np.isfinite(gamma_eff)
    print(f"  Valid fits: {valid_mask.sum()}/{len(valid_mask)}")
    print(f"  α_eff: median={np.nanmedian(alpha_eff):.4f}  mean={np.nanmean(alpha_eff):.4f}")
    print(f"  γ_eff: median={np.nanmedian(gamma_eff):.4f}  mean={np.nanmean(gamma_eff):.4f}")
    print(f"  R²:    median={np.nanmedian(r2_vals):.4f}  mean={np.nanmean(r2_vals):.4f}")

    plot_effective_alpha_gamma(
        alpha_eff, gamma_eff, r2_vals,
        expert_alpha=args.alpha, expert_gamma=args.gamma,
        save_path=os.path.join(args.figures_dir, 'effective_alpha_gamma.png'),
    )

    # -----------------------------------------------------------------------
    # Part 6: Reward probe from context delta
    # -----------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("Part 6: Reward probe from context delta")
    print(f"{'=' * 60}")

    print(f"\nCollecting reward-probe training data ({args.n_probe_train} trajectories) ...")
    delta_tr, r_tr = collect_reward_probe_data(
        model, args.n_probe_train, n_states, n_actions, vocab,
        config, device, n_steps=args.n_steps, seed_offset=40000,
    )
    print(f"  delta: {delta_tr.shape}, r: {r_tr.shape}")

    print(f"\nCollecting reward-probe eval data ({args.n_probe_eval} trajectories) ...")
    delta_ev, r_ev = collect_reward_probe_data(
        model, args.n_probe_eval, n_states, n_actions, vocab,
        config, device, n_steps=args.n_steps, seed_offset=50000,
    )

    print(f"\nTraining reward probe ...")
    r_probe = RewardProbe(config.d_model).to(device)
    r_losses = train_reward_probe(
        r_probe, delta_tr, r_tr, device, n_epochs=args.probe_epochs,
    )
    print(f"  MSE per epoch: {['%.4f' % l for l in r_losses]}")

    r2_r, mae_r, r_pred = evaluate_reward_probe(r_probe, delta_ev, r_ev, device)
    print(f"  Eval: R²={r2_r:.4f}  MAE={mae_r:.4f}")

    plot_reward_probe(
        r_ev, r_pred, r2=r2_r, mae=mae_r,
        save_path=os.path.join(args.figures_dir, 'reward_probe.png'),
    )

    # -----------------------------------------------------------------------
    # Part 6b: Per-step trace logs (1-2 single MDPs)
    # -----------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("Part 6b: Per-step trace logs")
    print(f"{'=' * 60}")
    trace_dir = os.path.join(args.figures_dir, 'step_traces')
    os.makedirs(trace_dir, exist_ok=True)
    trace_seeds = [args.eval_seed, args.eval_seed + 1]
    for tseed in trace_seeds:
        out_path = os.path.join(trace_dir, f'step_trace_seed{tseed}.txt')
        trace_step_log(
            model=model, q_probe=probe, r_probe=r_probe,
            n_states=n_states, n_actions=n_actions, vocab=vocab,
            config=config, device=device,
            seed=tseed, n_steps=args.n_steps,
            alpha=args.alpha, gamma=args.gamma, epsilon=args.epsilon,
            save_path=out_path,
        )

    # -----------------------------------------------------------------------
    # Training curves
    # -----------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("Training curves")
    print(f"{'=' * 60}")
    log_path = os.path.join(os.path.dirname(args.checkpoint), 'training_log.npz')
    plot_training_curves(log_path, os.path.join(args.figures_dir, 'training_curves.png'))

    print(f"\nDone. Figures saved to {args.figures_dir}/")


if __name__ == '__main__':
    main()
