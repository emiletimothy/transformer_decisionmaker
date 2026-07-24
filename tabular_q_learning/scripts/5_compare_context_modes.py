#!/usr/bin/env python3
"""
5_compare_context_modes.py — Continuous (latent) vs Discrete context, overlaid.

Loads two trained checkpoints of the SAME architecture — one with
context_mode='continuous' (the latent-thought model) and one with
context_mode='discrete' (context snapped to a vocabulary token each step) — and
reproduces the core quantitative metrics from 4_evaluate.py for BOTH models on
identical MDPs/seeds, overlaying them on shared axes so the performance gap is
directly visible.

All data-collection and inference routines are imported from 4_evaluate.py so
the two models are evaluated through exactly the same code path (the only
difference is each model's own `contextualize`).

Figures written to <figures_dir>/comparison/:
  action_agreement.png      — action-match vs step (both models)
  regret.png                — cumulative reward vs step (both models + baselines)
  long_horizon.png          — cumulative reward past the training horizon
  probe_scatter.png         — context-probe Q pred-vs-true (per-model panel)
  probe_frobenius.png       — context-probe Q Frobenius error vs step
  reward_probe.png          — reward decoded from context delta (per-model panel)
  effective_alpha_gamma.png — recovered (alpha_eff, gamma_eff) scatter
  training_curves.png       — val CE / accuracy for both training runs

Per-model qualitative artifacts (attention heatmaps, step traces) are not
overlaid — run 4_evaluate.py on each checkpoint for those.
"""

import argparse
import importlib.util
import os
from typing import Dict, List, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Import 4_evaluate.py as a module (its filename is not a valid identifier)
# ---------------------------------------------------------------------------
_here = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "coconut_eval", os.path.join(_here, "4_evaluate.py")
)
E = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(E)

COCONUTConfig      = E.COCONUTConfig
COCONUTTransformer = E.COCONUTTransformer
build_vocab        = E.build_vocab

# Consistent styling per context mode.
COLORS = {'continuous': '#1f77b4', 'discrete': '#d62728'}
FALLBACK_COLORS = ['#1f77b4', '#d62728', '#2ca02c', '#9467bd']


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_model(ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    config = COCONUTConfig.from_dict(ckpt['config'])
    model = COCONUTTransformer(config)
    model.load_state_dict(ckpt['model_state_dict'])
    model = model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    mode = getattr(config, 'context_mode', 'continuous')
    val_acc = ckpt.get('val_acc', None)
    print(f"  loaded {os.path.basename(ckpt_path)}  "
          f"context_mode={mode}  epoch={ckpt.get('epoch', '?')}  "
          f"val_acc={val_acc if val_acc is None else f'{val_acc:.3f}'}")
    return model, config, mode


# ---------------------------------------------------------------------------
# Metric 1: action agreement vs step (in-distribution)
# ---------------------------------------------------------------------------
def compute_action_agreement(model, config, vocab, n_states, n_actions,
                             eval_seeds, args, device) -> Tuple[np.ndarray, np.ndarray]:
    agreements = []
    for seed in eval_seeds:
        P, R = E.generate_eval_mdp(n_states, n_actions, seed=seed)
        trajectory, _ = E.run_tabular_q_learning(
            P, R, n_states, n_actions, n_steps=args.n_steps,
            alpha=args.alpha, gamma=args.gamma, epsilon=args.epsilon, seed=seed,
        )
        preds, _ = E.run_action_inference(model, trajectory, vocab, n_actions,
                                          config, device)
        targets = np.array([step['a_star'] for step in trajectory], dtype=np.int32)
        agreements.append((preds == targets).astype(np.float32))
    arr = np.stack(agreements, axis=0)
    return arr.mean(axis=0), arr.std(axis=0)


def plot_action_agreement(results, save_path, n_mdps):
    fig, ax = plt.subplots(figsize=(8, 5))
    for mode, (mean, std) in results.items():
        steps = np.arange(len(mean))
        c = COLORS.get(mode, 'gray')
        ax.plot(steps, mean, color=c, linewidth=2,
                label=f'{mode} context (final {mean[-1]:.0%})')
        ax.fill_between(steps, mean - std, mean + std, alpha=0.15, color=c)
    ax.axhline(1.0, color='gray', linestyle=':', linewidth=1, alpha=0.6)
    ax.set_xlabel('Q-learning step', fontsize=12)
    ax.set_ylabel('Action agreement with tabular Q*', fontsize=12)
    ax.set_title(f'Greedy-action agreement — continuous vs discrete context '
                 f'({n_mdps} MDPs)', fontsize=13)
    ax.set_ylim(0, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
    print(f"  wrote {save_path}")


# ---------------------------------------------------------------------------
# Metric 2 & 3: regret / long-horizon cumulative reward
# ---------------------------------------------------------------------------
def compute_cumrewards(models, vocab, n_states, n_actions, eval_seeds, n_steps,
                       args, device):
    """Returns dict mode->cumrew[n_mdps, n_steps], plus shared baselines."""
    per_model = {mode: [] for mode in models}
    greedy, epsgreedy, optimal = [], [], []
    for seed in eval_seeds:
        P, R = E.generate_eval_mdp(n_states, n_actions, seed=seed)
        for mode, (model, config) in models.items():
            rng = np.random.default_rng(seed + 100)
            rew = E.run_transformer_autonomous(
                model, P, R, n_states, n_actions, n_steps,
                vocab, config, device, epsilon=0.0, rng=rng,
            )
            per_model[mode].append(np.cumsum(rew))
        greedy.append(np.cumsum(E.run_q_learner_autonomous(
            P, R, n_states, n_actions, n_steps, alpha=args.alpha,
            gamma=args.gamma, epsilon=0.0,
            rng=np.random.default_rng(seed + 100))))
        epsgreedy.append(np.cumsum(E.run_q_learner_autonomous(
            P, R, n_states, n_actions, n_steps, alpha=args.alpha,
            gamma=args.gamma, epsilon=args.epsilon,
            rng=np.random.default_rng(seed + 100))))
        optimal.append(np.cumsum(E.run_optimal_autonomous(
            P, R, n_states, n_actions, n_steps, gamma=args.gamma,
            rng=np.random.default_rng(seed + 100))))
    cum = {mode: np.stack(v, axis=0) for mode, v in per_model.items()}
    return cum, np.stack(greedy), np.stack(epsgreedy), np.stack(optimal)


def plot_cumreward(cum, greedy, epsgreedy, optimal, save_path, title):
    fig, ax = plt.subplots(figsize=(8, 5))
    steps = np.arange(cum[next(iter(cum))].shape[1])
    # Shared baselines drawn once.
    if optimal is not None:
        ax.plot(steps, optimal.mean(0), color='black', linestyle='--',
                linewidth=1.6, label='Optimal policy')
    ax.plot(steps, epsgreedy.mean(0), color='gray', linewidth=1.4,
            label='ε-greedy tabular Q')
    ax.plot(steps, greedy.mean(0), color='darkgray', linestyle=':',
            linewidth=1.4, label='greedy tabular Q')
    for mode, c in cum.items():
        col = COLORS.get(mode, 'gray')
        m = c.mean(0)
        ax.plot(steps, m, color=col, linewidth=2.2,
                label=f'{mode} transformer (final {m[-1]:.1f})')
        ax.fill_between(steps, m - c.std(0), m + c.std(0), alpha=0.12, color=col)
    ax.set_xlabel('step', fontsize=12)
    ax.set_ylabel('cumulative reward', fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
    print(f"  wrote {save_path}")


# ---------------------------------------------------------------------------
# Metric 4 & 5: context Q-probe (scatter + Frobenius) per model
# ---------------------------------------------------------------------------
def fit_context_probe(model, config, n_states, n_actions, vocab, args, device):
    ctx_tr, q_tr, _ = E.collect_context_probe_data(
        model, args.n_probe_train, n_states, n_actions, vocab,
        config, device, n_steps=args.n_steps, seed_offset=20000)
    ctx_ev, q_ev, traj_ev = E.collect_context_probe_data(
        model, args.n_probe_eval, n_states, n_actions, vocab,
        config, device, n_steps=args.n_steps, seed_offset=30000)
    probe = E.ContextQProbe(config.d_model, n_states).to(device)
    E.train_probe(probe, ctx_tr, q_tr, device, n_epochs=args.probe_epochs)
    r2, frob, q_pred = E.evaluate_probe(probe, ctx_ev, q_ev, device)
    return dict(probe=probe, ctx_ev=ctx_ev, q_ev=q_ev, traj_ev=traj_ev,
                r2=r2, frob=frob, q_pred=q_pred)


def plot_probe_scatter(probe_results, save_path):
    modes = list(probe_results)
    fig, axes = plt.subplots(1, len(modes), figsize=(5.2 * len(modes), 5),
                             squeeze=False)
    for ax, mode in zip(axes[0], modes):
        pr = probe_results[mode]
        qt, qp = pr['q_ev'].reshape(-1), pr['q_pred'].reshape(-1)
        c = COLORS.get(mode, 'gray')
        ax.scatter(qt, qp, s=4, alpha=0.25, color=c)
        lim = [min(qt.min(), qp.min()), max(qt.max(), qp.max())]
        ax.plot(lim, lim, 'k--', linewidth=1, alpha=0.6)
        ax.set_xlabel('true Q', fontsize=12)
        ax.set_ylabel('probe-decoded Q', fontsize=12)
        ax.set_title(f'{mode} context   (R²={pr["r2"]:.3f})', fontsize=13)
        ax.grid(True, alpha=0.3)
    fig.suptitle('Linear decodability of Q from the carried context',
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
    print(f"  wrote {save_path}")


def plot_probe_frobenius(probe_results, n_steps, n_actions, save_path):
    fig, ax = plt.subplots(figsize=(8, 5))
    for mode, pr in probe_results.items():
        # q_pred/q_ev are flat [N, n_states]; N = n_traj * n_steps * n_actions.
        diff = pr['q_pred'] - pr['q_ev']
        frob = np.sqrt((diff ** 2).sum(axis=-1))  # [N]
        per_step = frob.reshape(-1, n_steps, n_actions).mean(axis=(0, 2))
        c = COLORS.get(mode, 'gray')
        ax.plot(np.arange(n_steps), per_step, color=c, linewidth=2,
                label=f'{mode} context (mean {frob.mean():.3f})')
    ax.set_xlabel('Q-learning step', fontsize=12)
    ax.set_ylabel('‖Q_pred − Q_true‖ (per action-row)', fontsize=12)
    ax.set_title('Context-probe Q reconstruction error vs step', fontsize=13)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
    print(f"  wrote {save_path}")


# ---------------------------------------------------------------------------
# Metric 6: reward probe from context delta (per model)
# ---------------------------------------------------------------------------
def fit_reward_probe(model, config, n_states, n_actions, vocab, args, device):
    d_tr, r_tr = E.collect_reward_probe_data(
        model, args.n_probe_train, n_states, n_actions, vocab,
        config, device, n_steps=args.n_steps, seed_offset=40000)
    d_ev, r_ev = E.collect_reward_probe_data(
        model, args.n_probe_eval, n_states, n_actions, vocab,
        config, device, n_steps=args.n_steps, seed_offset=50000)
    rprobe = E.RewardProbe(config.d_model).to(device)
    E.train_reward_probe(rprobe, d_tr, r_tr, device, n_epochs=args.probe_epochs)
    r2, mae, r_pred = E.evaluate_reward_probe(rprobe, d_ev, r_ev, device)
    return dict(r_ev=r_ev, r_pred=r_pred, r2=r2, mae=mae)


def plot_reward_probe(reward_results, save_path):
    modes = list(reward_results)
    fig, axes = plt.subplots(1, len(modes), figsize=(5.2 * len(modes), 5),
                             squeeze=False)
    for ax, mode in zip(axes[0], modes):
        rr = reward_results[mode]
        c = COLORS.get(mode, 'gray')
        ax.scatter(rr['r_ev'], rr['r_pred'], s=5, alpha=0.3, color=c)
        lim = [min(rr['r_ev'].min(), rr['r_pred'].min()),
               max(rr['r_ev'].max(), rr['r_pred'].max())]
        ax.plot(lim, lim, 'k--', linewidth=1, alpha=0.6)
        ax.set_xlabel('true reward', fontsize=12)
        ax.set_ylabel('reward decoded from Δcontext', fontsize=12)
        ax.set_title(f'{mode} context   (R²={rr["r2"]:.3f}, MAE={rr["mae"]:.3f})',
                     fontsize=12)
        ax.grid(True, alpha=0.3)
    fig.suptitle('Reward recoverable from the context update', fontsize=13)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
    print(f"  wrote {save_path}")


# ---------------------------------------------------------------------------
# Metric 7: effective alpha/gamma recovery (per model)
# ---------------------------------------------------------------------------
def plot_effective_alpha_gamma(probe_results, args, n_actions, device, save_path):
    fig, ax = plt.subplots(figsize=(7, 6))
    for mode, pr in probe_results.items():
        a_eff, g_eff, r2v = E.estimate_effective_alpha_gamma(
            pr['probe'], pr['ctx_ev'], pr['traj_ev'], device,
            n_steps=args.n_steps, n_traj=args.n_probe_eval, n_actions=n_actions)
        valid = np.isfinite(a_eff) & np.isfinite(g_eff)
        c = COLORS.get(mode, 'gray')
        ax.scatter(a_eff[valid], g_eff[valid], s=14, alpha=0.4, color=c,
                   label=f'{mode}: α̃={np.nanmedian(a_eff):.3f}, '
                         f'γ̃={np.nanmedian(g_eff):.3f}')
    ax.axvline(args.alpha, color='green', linestyle='--', linewidth=1.2,
               label=f'expert α={args.alpha}')
    ax.axhline(args.gamma, color='purple', linestyle='--', linewidth=1.2,
               label=f'expert γ={args.gamma}')
    ax.set_xlabel('recovered effective α', fontsize=12)
    ax.set_ylabel('recovered effective γ', fontsize=12)
    ax.set_title('Effective (α, γ) implied by context dynamics', fontsize=13)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
    print(f"  wrote {save_path}")


# ---------------------------------------------------------------------------
# Metric 8: training curves for both runs
# ---------------------------------------------------------------------------
def plot_training_curves(log_paths, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    any_data = False
    for mode, path in log_paths.items():
        if not path or not os.path.exists(path):
            print(f"    (training log for {mode} not found: {path})")
            continue
        any_data = True
        d = np.load(path)
        c = COLORS.get(mode, 'gray')
        axes[0].plot(d['steps'], d['val_ce'], color=c, linewidth=2,
                     label=f'{mode} (val)')
        axes[0].plot(d['steps'], d['train_ce'], color=c, linewidth=1,
                     alpha=0.4, linestyle=':')
        axes[1].plot(d['steps'], d['val_acc'] * 100, color=c, linewidth=2,
                     label=f'{mode} (val)')
        axes[1].plot(d['steps'], d['train_acc'] * 100, color=c, linewidth=1,
                     alpha=0.4, linestyle=':')
    axes[0].set_xlabel('step'); axes[0].set_ylabel('cross-entropy')
    axes[0].set_title('Training / validation CE'); axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=10)
    axes[1].set_xlabel('step'); axes[1].set_ylabel('accuracy (%)')
    axes[1].set_title('Training / validation accuracy'); axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=10)
    fig.suptitle('Optimization: continuous vs discrete context (solid=val, dotted=train)',
                 fontsize=13)
    fig.tight_layout()
    if any_data:
        fig.savefig(save_path, dpi=200)
        print(f"  wrote {save_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description='Overlay continuous vs discrete context models.')
    parser.add_argument('--continuous_ckpt', type=str, required=True)
    parser.add_argument('--discrete_ckpt',   type=str, required=True)
    parser.add_argument('--figures_dir', type=str,
                        default=os.path.join(_here, '..', 'figures', 'comparison'))
    parser.add_argument('--continuous_log', type=str, default=None,
                        help='training_log.npz for the continuous run (optional).')
    parser.add_argument('--discrete_log',   type=str, default=None,
                        help='training_log.npz for the discrete run (optional).')
    parser.add_argument('--n_steps',       type=int,   default=50)
    parser.add_argument('--long_horizon_steps', type=int, default=200)
    parser.add_argument('--alpha',         type=float, default=0.1)
    parser.add_argument('--gamma',         type=float, default=0.9)
    parser.add_argument('--epsilon',       type=float, default=0.2)
    parser.add_argument('--eval_seed',     type=int,   default=9999)
    parser.add_argument('--n_eval_mdps',   type=int,   default=10)
    parser.add_argument('--n_probe_train', type=int,   default=500)
    parser.add_argument('--n_probe_eval',  type=int,   default=100)
    parser.add_argument('--probe_epochs',  type=int,   default=10)
    args = parser.parse_args()

    os.makedirs(args.figures_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    print("\nLoading models:")
    cont_model, cont_cfg, cont_mode = load_model(args.continuous_ckpt, device)
    disc_model, disc_cfg, disc_mode = load_model(args.discrete_ckpt, device)
    if cont_mode != 'continuous':
        print(f"  WARNING: --continuous_ckpt has context_mode='{cont_mode}'")
    if disc_mode != 'discrete':
        print(f"  WARNING: --discrete_ckpt has context_mode='{disc_mode}'")

    # Both models must share the vocab / MDP dims (same architecture).
    assert cont_cfg.max_states == disc_cfg.max_states
    assert cont_cfg.max_actions == disc_cfg.max_actions
    n_states, n_actions = cont_cfg.max_states, cont_cfg.max_actions
    vocab = build_vocab(n_states, n_actions)
    eval_seeds = list(range(args.eval_seed, args.eval_seed + args.n_eval_mdps))

    models = {'continuous': (cont_model, cont_cfg),
              'discrete':   (disc_model, disc_cfg)}
    fig = lambda name: os.path.join(args.figures_dir, name)

    # ---- Metric 1: action agreement ----
    print("\n[1/8] Action agreement vs step ...")
    agree = {}
    for mode, (model, cfg) in models.items():
        agree[mode] = compute_action_agreement(
            model, cfg, vocab, n_states, n_actions, eval_seeds, args, device)
        print(f"    {mode}: mean agreement {agree[mode][0].mean():.2%}")
    plot_action_agreement(agree, fig('action_agreement.png'), args.n_eval_mdps)

    # ---- Metric 2: regret ----
    print("\n[2/8] Regret (autonomous cumulative reward) ...")
    cum, g, e, o = compute_cumrewards(
        models, vocab, n_states, n_actions, eval_seeds, args.n_steps, args, device)
    for mode, c in cum.items():
        print(f"    {mode}: final cumreward {c[:, -1].mean():.2f}")
    print(f"    optimal: {o[:, -1].mean():.2f}   ε-greedy Q: {e[:, -1].mean():.2f}")
    plot_cumreward(cum, g, e, o, fig('regret.png'),
                   f'Cumulative reward — continuous vs discrete ({args.n_eval_mdps} MDPs)')

    # ---- Metric 3: long horizon ----
    if args.long_horizon_steps > args.n_steps:
        print(f"\n[3/8] Long-horizon eval ({args.long_horizon_steps} steps) ...")
        cum_l, g_l, e_l, o_l = compute_cumrewards(
            models, vocab, n_states, n_actions, eval_seeds,
            args.long_horizon_steps, args, device)
        plot_cumreward(cum_l, g_l, e_l, o_l, fig('long_horizon.png'),
                       f'Long-horizon cumulative reward (train horizon={args.n_steps})')
    else:
        print("\n[3/8] Long-horizon skipped (long_horizon_steps <= n_steps).")

    # ---- Metrics 4,5,7: context Q-probe ----
    print("\n[4-5/8] Context Q-probe (per model) ...")
    probe_results = {}
    for mode, (model, cfg) in models.items():
        print(f"    fitting probe for {mode} ...")
        probe_results[mode] = fit_context_probe(
            model, cfg, n_states, n_actions, vocab, args, device)
        print(f"      {mode}: R²={probe_results[mode]['r2']:.3f}  "
              f"Frobenius={probe_results[mode]['frob']:.3f}")
    plot_probe_scatter(probe_results, fig('probe_scatter.png'))
    plot_probe_frobenius(probe_results, args.n_steps, n_actions,
                         fig('probe_frobenius.png'))

    print("\n[7/8] Effective alpha/gamma recovery ...")
    plot_effective_alpha_gamma(probe_results, args, n_actions, device,
                               fig('effective_alpha_gamma.png'))

    # ---- Metric 6: reward probe ----
    print("\n[6/8] Reward probe from context delta (per model) ...")
    reward_results = {}
    for mode, (model, cfg) in models.items():
        print(f"    fitting reward probe for {mode} ...")
        reward_results[mode] = fit_reward_probe(
            model, cfg, n_states, n_actions, vocab, args, device)
        print(f"      {mode}: R²={reward_results[mode]['r2']:.3f}  "
              f"MAE={reward_results[mode]['mae']:.3f}")
    plot_reward_probe(reward_results, fig('reward_probe.png'))

    # ---- Metric 8: training curves ----
    print("\n[8/8] Training curves ...")
    plot_training_curves(
        {'continuous': args.continuous_log, 'discrete': args.discrete_log},
        fig('training_curves.png'))

    # ---- Summary table ----
    print("\n" + "=" * 60)
    print(" SUMMARY  (continuous vs discrete context)")
    print("=" * 60)
    print(f"  {'metric':<28}{'continuous':>14}{'discrete':>14}")
    print(f"  {'final action agreement':<28}"
          f"{agree['continuous'][0].mean():>13.1%}{agree['discrete'][0].mean():>14.1%}")
    print(f"  {'final cumulative reward':<28}"
          f"{cum['continuous'][:, -1].mean():>14.2f}{cum['discrete'][:, -1].mean():>14.2f}")
    print(f"  {'context-probe Q R²':<28}"
          f"{probe_results['continuous']['r2']:>14.3f}{probe_results['discrete']['r2']:>14.3f}")
    print(f"  {'reward-probe R²':<28}"
          f"{reward_results['continuous']['r2']:>14.3f}{reward_results['discrete']['r2']:>14.3f}")
    print("=" * 60)
    print(f"\nDone. Comparison figures in {args.figures_dir}/")


if __name__ == '__main__':
    main()
