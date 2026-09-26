#!/usr/bin/env python3
"""
eval_closed_loop.py — per-MDP (per-seed) re-evaluation of a trained checkpoint.

Re-runs the closed-loop evals from 4_evaluate.py / 5_compare_context_modes.py
but saves every per-MDP number, so each row can be reported as mean ± SD / SEM:

  long_horizon         1000-step closed-loop return, `eval` + `contrast` families
  nonstationary        all five mid-episode-switch variants
  size_sweep           |S| = 2..8 x |A| = 2..4: teacher-forced agreement + contrast return
  reward_intervention  intact / zeroed / constant / decorrelated / inverted

Beyond return it records, per rollout,
  opt_frac   fraction of steps whose action is optimal under the value-iteration Q*
  gap_regret sum_t [V*(s_t) - Q*(s_t, a_t)]   (policy regret, independent of state luck)
and, teacher-forced, how often the model's SELECT prediction (and the tabular
teacher's target) equals the TRUE optimal action argmax Q*(s_{t+1}, .).

a* at UPDATE in closed loop
---------------------------
The legacy closed-loop runners in 4_evaluate.py feed `a_star = 0` into the step
tokens, i.e. the UPDATE token is always told that the greedy next action is a_0,
whereas in training a* is the teacher's argmax. `--astar_modes` evaluates both:
  legacy  a* = 0 (reproduces 4_evaluate.py exactly, same RNG draw order)
  self    a* = the model's own SELECT argmax (SELECT precedes a* causally, so the
          prediction is unaffected; a second pass recomputes the UPDATE hidden)
  self_eps<e>  as `self`, but the executed action is epsilon-greedy on the model's
          SELECT argmax: pi = (1 - e) onehot(argmax) + e / |A|, a fixed mixing of the
          greedy head with uniform samples (the eps-greedy baseline's behaviour policy;
          the training data were generated eps-greedy). a* stays the greedy argmax.

Rollouts are batched across MDPs (every MDP keeps its own RNG, drawn in exactly
the order the legacy single-MDP runners use), so this is fast on one GPU.

Run from tabular_q_learning/:
  python3 scripts/eval_closed_loop.py --checkpoint checkpoints/<ckpt>.pt \
      --label continuous --out_dir figures/continuous_residual/closed_loop --n_mdps 30
  python3 scripts/eval_closed_loop.py --model continuous_residual     # same, from paths.py
"""
import argparse
import csv
import importlib.util
import json
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import paths  # noqa: E402
import time
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch

_script_dir = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "coconut_eval", os.path.join(_script_dir, "4_evaluate.py"))
E = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(E)

AGENTS_BASE = ('optimal', 'epsgreedy', 'greedy', 'random')


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_model(ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    config = E.COCONUTConfig.from_dict(ckpt['config'])
    model = E.COCONUTTransformer(config)
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, config


def qstar_list(phases) -> List[np.ndarray]:
    return [E.value_iteration(P, R, GAMMA) for P, R, _ in phases]


def rollout_metrics(states, actions, phase_ids, qstars) -> Tuple[float, float]:
    """opt_frac and gap_regret for one rollout."""
    opt, gap = 0, 0.0
    for s, a, k in zip(states, actions, phase_ids):
        q = qstars[k][s]
        v = float(q.max())
        opt += int(q[a] >= v - 1e-6)
        gap += v - float(q[a])
    return opt / len(states), gap


# ---------------------------------------------------------------------------
# Batched transformer closed-loop rollout
# ---------------------------------------------------------------------------
@torch.no_grad()
def run_tf_batched(model, config, vocab, phases_per_mdp, n_states, n_actions,
                   device, seeds, astar_mode='legacy',
                   reward_transforms: Optional[List[Optional[Callable]]] = None):
    """Closed-loop rollout of B MDPs in lockstep (greedy unless astar_mode is self_eps<e>).

    phases_per_mdp[b] = [(P, R, n_steps), ...]; every MDP must have the same
    total length. Returns rewards/states/actions arrays [B, T].
    RNG protocol per MDP is identical to E.run_transformer_autonomous /
    E.run_transformer_nonstationary (rng = default_rng(seed + 100)).
    """
    explore = 0.0
    if astar_mode.startswith('self_eps'):
        explore, astar_mode = float(astar_mode[len('self_eps'):]), 'self'
    B = len(phases_per_mdp)
    expanded = [E.expand_phases(ph) for ph in phases_per_mdp]
    T = len(expanded[0][0])
    rngs = [np.random.default_rng(sd + 100) for sd in seeds]
    context = model.get_init_context(B, n_actions, device)
    s = np.array([int(r.integers(n_states)) for r in rngs])
    pred = np.zeros(B, dtype=np.int64)
    rewards = np.zeros((B, T), dtype=np.float32)
    S = np.zeros((B, T), dtype=np.int64)
    A = np.zeros((B, T), dtype=np.int64)
    bidx = torch.arange(B, device=device)

    for t in range(T):
        a = np.zeros(B, dtype=np.int64)
        r_obs = np.zeros(B, dtype=np.float32)
        s_next = np.zeros(B, dtype=np.int64)
        for b in range(B):
            rng = rngs[b]
            P, R = expanded[b][0][t], expanded[b][1][t]
            u = rng.random()  # epsilon draw (same order as the eps-greedy baseline)
            explore_now = t == 0 or u < explore
            a[b] = int(rng.integers(n_actions)) if explore_now else int(pred[b])
            r = float(R[s[b], a[b]])
            rt = reward_transforms[b] if reward_transforms else None
            r_obs[b] = r if rt is None else float(rt(r, s[b], a[b], R))
            s_next[b] = int(rng.choice(n_states, p=P[s[b], a[b]]))
            rewards[b, t] = r
        S[:, t], A[:, t] = s, a

        def step_tokens(astar):
            toks = []
            for b in range(B):
                tr = {'s': int(s[b]), 'a': int(a[b]), 'r': float(r_obs[b]),
                      's_next': int(s_next[b]), 'a_star': int(astar[b])}
                ids, r_off, s_off, u_off = E.build_step_tokens(tr, vocab, n_actions)
                toks.append(ids)
            return (torch.tensor(toks, dtype=torch.long, device=device),
                    r_off, s_off, u_off)

        ids, r_off, s_off, u_off = step_tokens(np.zeros(B, dtype=np.int64))
        rv = torch.tensor(r_obs, device=device)
        sel, upd = model.forward_step(ids, rv, r_off, s_off, u_off, context)
        if n_actions < config.max_actions:
            sel[:, n_actions:] = float('-inf')
        new_pred = sel.argmax(-1).cpu().numpy()
        if astar_mode == 'self':
            ids2, r_off, s_off, u_off = step_tokens(new_pred)
            _, upd = model.forward_step(ids2, rv, r_off, s_off, u_off, context)
        pred = new_pred
        new_ctx = context.clone()
        new_ctx[bidx, torch.as_tensor(a, device=device)] = model.contextualize(upd)
        context = new_ctx
        s = s_next

    return rewards, S, A


@torch.no_grad()
def teacher_forced_batched(model, config, vocab, trajectories, n_actions, device):
    """Batched E.run_action_inference: predictions [B, T] given teacher trajectories."""
    B, T = len(trajectories), len(trajectories[0])
    context = model.get_init_context(B, n_actions, device)
    preds = np.zeros((B, T), dtype=np.int64)
    bidx = torch.arange(B, device=device)
    for t in range(T):
        toks = []
        for b in range(B):
            ids, r_off, s_off, u_off = E.build_step_tokens(trajectories[b][t], vocab, n_actions)
            toks.append(ids)
        ids = torch.tensor(toks, dtype=torch.long, device=device)
        rv = torch.tensor([trajectories[b][t]['r'] for b in range(B)],
                          dtype=torch.float32, device=device)
        sel, upd = model.forward_step(ids, rv, r_off, s_off, u_off, context)
        if n_actions < config.max_actions:
            sel[:, n_actions:] = float('-inf')
        preds[:, t] = sel.argmax(-1).cpu().numpy()
        a_t = torch.tensor([trajectories[b][t]['a'] for b in range(B)], device=device)
        new_ctx = context.clone()
        new_ctx[bidx, a_t] = model.contextualize(upd)
        context = new_ctx
    return preds


# ---------------------------------------------------------------------------
# Baseline rollouts with (s, a) logging — same RNG draw order as 4_evaluate.py
# ---------------------------------------------------------------------------
def run_baseline(kind, phases, n_states, n_actions, seed, reward_transform=None):
    Ps, Rs, ids = E.expand_phases(phases)
    T = len(Ps)
    rng = np.random.default_rng(seed + 100)
    rewards = np.zeros(T, dtype=np.float32)
    S = np.zeros(T, dtype=np.int64)
    A = np.zeros(T, dtype=np.int64)
    s = int(rng.integers(n_states))
    if kind in ('epsgreedy', 'greedy'):
        eps = EPSILON if kind == 'epsgreedy' else 0.0
        Q = np.zeros((n_states, n_actions), dtype=np.float32)
        for t in range(T):
            P, R = Ps[t], Rs[t]
            if rng.random() < eps:
                a = int(rng.integers(n_actions))
            else:
                best = float(np.max(Q[s]))
                ties = [ac for ac in range(n_actions) if Q[s, ac] == best]
                a = int(rng.choice(ties))
            r = float(R[s, a])
            r_obs = r if reward_transform is None else float(reward_transform(r, s, a, R))
            s_next = int(rng.choice(n_states, p=P[s, a]))
            Q[s, a] = (1 - ALPHA) * Q[s, a] + ALPHA * (r_obs + GAMMA * float(np.max(Q[s_next])))
            rewards[t], S[t], A[t] = r, s, a
            s = s_next
    elif kind == 'optimal':  # adapts at phase boundaries
        cache = {}
        for t in range(T):
            P, R = Ps[t], Rs[t]
            if ids[t] not in cache:
                cache[ids[t]] = E.value_iteration(P, R, GAMMA)
            q = cache[ids[t]]
            best = float(np.max(q[s]))
            ties = [ac for ac in range(n_actions) if q[s, ac] == best]
            a = int(rng.choice(ties))
            rewards[t], S[t], A[t] = float(R[s, a]), s, a
            s = int(rng.choice(n_states, p=P[s, a]))
    elif kind == 'optimal_frozen':
        q = E.value_iteration(Ps[0], Rs[0], GAMMA)
        for t in range(T):
            P, R = Ps[t], Rs[t]
            best = float(np.max(q[s]))
            ties = [ac for ac in range(n_actions) if q[s, ac] == best]
            a = int(rng.choice(ties))
            rewards[t], S[t], A[t] = float(R[s, a]), s, a
            s = int(rng.choice(n_states, p=P[s, a]))
    elif kind == 'random':
        for t in range(T):
            a = int(rng.integers(n_actions))
            rewards[t], S[t], A[t] = float(Rs[t][s, a]), s, a
            s = int(rng.choice(n_states, p=Ps[t][s, a]))
    else:
        raise ValueError(kind)
    return rewards, S, A, np.array(ids)


def collect_rollouts(model, config, vocab, phases_per_mdp, n_states, n_actions,
                     device, seeds, astar_modes, baselines=AGENTS_BASE,
                     reward_transform_fn=None):
    """Run the transformer (each a* mode) + baselines. Returns
    {agent: {'rewards': [B,T], 'opt_frac': [B], 'gap_regret': [B]}}."""
    out = {}
    qstars = [qstar_list(ph) for ph in phases_per_mdp]
    phase_ids = np.array(E.expand_phases(phases_per_mdp[0])[2])
    for mode in astar_modes:
        rts = ([reward_transform_fn(sd) for sd in seeds]
               if reward_transform_fn else None)
        rew, S, A = run_tf_batched(model, config, vocab, phases_per_mdp,
                                   n_states, n_actions, device, seeds, mode, rts)
        m = [rollout_metrics(S[b], A[b], phase_ids, qstars[b]) for b in range(len(seeds))]
        out[f'tf_{mode}'] = {'rewards': rew, 'opt_frac': np.array([x[0] for x in m]),
                             'gap_regret': np.array([x[1] for x in m])}
    for kind in baselines:
        rews, ofs, gaps = [], [], []
        for b, sd in enumerate(seeds):
            rt = reward_transform_fn(sd) if reward_transform_fn else None
            rew, S, A, ids = run_baseline(kind, phases_per_mdp[b], n_states,
                                          n_actions, sd, rt)
            of, gp = rollout_metrics(S, A, ids, qstars[b])
            rews.append(rew); ofs.append(of); gaps.append(gp)
        out[kind] = {'rewards': np.stack(rews), 'opt_frac': np.array(ofs),
                     'gap_regret': np.array(gaps)}
    return out


def msd(x):
    x = np.asarray(x, dtype=np.float64)
    return float(x.mean()), float(x.std(ddof=1)) if len(x) > 1 else 0.0, \
        float(x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else 0.0


def write_csv(path, rows):
    if not rows:
        return
    keys = list(rows[0].keys())
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"  wrote {path}", flush=True)


def summary_rows(block, cond, res, ref_key='optimal', extra=None):
    """One row per agent: return / %opt / opt_frac / gap_regret, mean-SD-SEM."""
    rows = []
    opt_ret = res[ref_key]['rewards'].sum(1)
    for agent, d in res.items():
        ret = d['rewards'].sum(1)
        pct = 100.0 * ret / opt_ret
        row = {'block': block, 'condition': cond, 'agent': agent, 'n_mdps': len(ret)}
        for name, arr in (('return', ret), ('pct_opt', pct),
                          ('opt_frac', d['opt_frac']), ('gap_regret', d['gap_regret']),
                          ('return_regret', opt_ret - ret)):
            m, sd, se = msd(arr)
            row[f'{name}_mean'], row[f'{name}_sd'], row[f'{name}_sem'] = m, sd, se
        if extra:
            row.update(extra)
        rows.append(row)
    return rows


def per_mdp_rows(block, cond, res, seeds):
    rows = []
    for agent, d in res.items():
        ret = d['rewards'].sum(1)
        for b, sd in enumerate(seeds):
            rows.append({'block': block, 'condition': cond, 'agent': agent,
                         'seed': sd, 'return': float(ret[b]),
                         'opt_frac': float(d['opt_frac'][b]),
                         'gap_regret': float(d['gap_regret'][b])})
    return rows


# ---------------------------------------------------------------------------
# Blocks
# ---------------------------------------------------------------------------
def block_long_horizon(model, config, vocab, device, args, seeds, out_dir):
    ns, na = config.max_states, config.max_actions
    summ, per, arrays = [], [], {}
    for fam, _, fn in E.MDP_FAMILIES:
        t0 = time.time()
        phases = [[(*fn(ns, na, seed=sd), args.long_horizon_steps)] for sd in seeds]
        res = collect_rollouts(model, config, vocab, phases, ns, na, device, seeds,
                               args.astar_modes)
        summ += summary_rows('long_horizon', fam, res)
        per += per_mdp_rows('long_horizon', fam, res, seeds)
        for agent, d in res.items():
            arrays[f'{fam}__{agent}'] = d['rewards']
        print(f"  long_horizon/{fam}: {time.time() - t0:.0f}s  " +
              "  ".join(f"{k}={v['rewards'].sum(1).mean():.1f}" for k, v in res.items()),
              flush=True)
    np.savez_compressed(os.path.join(out_dir, 'long_horizon_rewards.npz'), **arrays)
    return summ, per


def block_nonstationary(model, config, vocab, device, args, seeds, out_dir):
    ns, na = config.max_states, config.max_actions
    sw, post, win = args.switch_step, args.post_steps, args.window
    summ, per, arrays = [], [], {}
    for variant, _ in E.NONSTATIONARY_VARIANTS:
        t0 = time.time()
        phases = []
        for sd in seeds:
            (P1, R1), (P2, R2) = E.generate_nonstationary_mdp(ns, na, variant, seed=sd)
            phases.append([(P1, R1, sw), (P2, R2, post)])
        res = collect_rollouts(model, config, vocab, phases, ns, na, device, seeds,
                               args.astar_modes,
                               baselines=AGENTS_BASE + ('optimal_frozen',))
        opt = res['optimal']['rewards']
        windows = {'pre': slice(sw - win, sw), 'post': slice(sw, sw + win),
                   'end': slice(sw + post - win, sw + post)}
        for agent, d in res.items():
            rew = d['rewards']
            row = {'block': 'nonstationary', 'condition': variant, 'agent': agent,
                   'n_mdps': len(seeds)}
            for name, arr in (('final_cumreward', rew.sum(1)),
                              ('opt_frac', d['opt_frac']),
                              ('gap_regret', d['gap_regret'])):
                m, s_, se = msd(arr)
                row[f'{name}_mean'], row[f'{name}_sd'], row[f'{name}_sem'] = m, s_, se
            for tag, sl in windows.items():
                rate = rew[:, sl].mean(1)
                pct = 100.0 * rate / np.maximum(opt[:, sl].mean(1), 1e-8)
                m, s_, se = msd(rate)
                row[f'{tag}_rate_mean'], row[f'{tag}_rate_sd'], row[f'{tag}_rate_sem'] = m, s_, se
                m, s_, se = msd(pct)
                row[f'{tag}_pct_opt_mean'], row[f'{tag}_pct_opt_sd'], row[f'{tag}_pct_opt_sem'] = m, s_, se
            summ.append(row)
            arrays[f'{variant}__{agent}'] = rew
        per += per_mdp_rows('nonstationary', variant, res, seeds)
        print(f"  nonstationary/{variant}: {time.time() - t0:.0f}s", flush=True)
    np.savez_compressed(os.path.join(out_dir, 'nonstationary_rewards.npz'), **arrays)
    return summ, per


def block_reward_intervention(model, config, vocab, device, args, seeds, out_dir):
    ns, na = config.max_states, config.max_actions
    summ, per = [], []
    for fam, _, fn in E.MDP_FAMILIES:
        phases = [[(*fn(ns, na, seed=sd), args.ri_steps)] for sd in seeds]
        for cond, _ in E.REWARD_INTERVENTIONS:
            t0 = time.time()
            rtf = (None if cond == 'intact'
                   else (lambda sd, c=cond: E.make_reward_transform(c, sd + 7777)))
            res = collect_rollouts(model, config, vocab, phases, ns, na, device,
                                   seeds, args.astar_modes, reward_transform_fn=rtf)
            summ += summary_rows('reward_intervention', f'{fam}/{cond}', res)
            per += per_mdp_rows('reward_intervention', f'{fam}/{cond}', res, seeds)
            print(f"  reward_intervention/{fam}/{cond}: {time.time() - t0:.0f}s  " +
                  "  ".join(f"{k}={v['rewards'].sum(1).mean():.1f}"
                            for k, v in res.items() if k.startswith('tf')), flush=True)
    return summ, per


def block_size_sweep(model, config, vocab, device, args, seeds, out_dir):
    summ, per = [], []
    for ns in range(2, config.max_states + 1):
        for na in range(2, config.max_actions + 1):
            t0 = time.time()
            # Teacher-forced agreement on Beta(2,2) + true-optimal-action rates.
            trajs, qst = [], []
            for sd in seeds:
                P, R = E.generate_eval_mdp(ns, na, seed=sd)
                tr, _ = E.run_tabular_q_learning(P, R, ns, na, n_steps=args.n_steps,
                                                 alpha=ALPHA, gamma=GAMMA,
                                                 epsilon=EPSILON, seed=sd)
                trajs.append(tr)
                qst.append(E.value_iteration(P, R, GAMMA))
            preds = teacher_forced_batched(model, config, vocab, trajs, na, device)
            agree, tf_opt, teach_opt = [], [], []
            for b in range(len(seeds)):
                tgt = np.array([st['a_star'] for st in trajs[b]])
                snext = np.array([st['s_next'] for st in trajs[b]])
                q = qst[b][snext]                      # [T, na]
                best = q.max(1, keepdims=True)
                is_opt = q >= best - 1e-6
                agree.append(float((preds[b] == tgt).mean()))
                tf_opt.append(float(is_opt[np.arange(len(tgt)), preds[b]].mean()))
                teach_opt.append(float(is_opt[np.arange(len(tgt)), tgt].mean()))
            # Closed-loop return on the contrast family.
            phases = [[(*E.generate_contrast_mdp_seeded(ns, na, seed=sd),
                        args.size_sweep_steps)] for sd in seeds]
            res = collect_rollouts(model, config, vocab, phases, ns, na, device,
                                   seeds, args.astar_modes,
                                   baselines=('optimal', 'random'))
            extra = {'n_states': ns, 'n_actions': na}
            for name, arr in (('agreement', agree), ('tf_pred_true_opt', tf_opt),
                              ('teacher_true_opt', teach_opt)):
                m, sd_, se = msd(arr)
                extra[f'{name}_mean'], extra[f'{name}_sd'], extra[f'{name}_sem'] = m, sd_, se
            summ += summary_rows('size_sweep', f'{ns}x{na}', res, extra=extra)
            rows = per_mdp_rows('size_sweep', f'{ns}x{na}', res, seeds)
            for b, sd in enumerate(seeds):
                rows.append({'block': 'size_sweep_teacher_forced',
                             'condition': f'{ns}x{na}', 'agent': 'tf', 'seed': sd,
                             'agreement': agree[b], 'tf_pred_true_opt': tf_opt[b],
                             'teacher_true_opt': teach_opt[b]})
            per += rows
            print(f"  size_sweep {ns}x{na}: {time.time() - t0:.0f}s  agree "
                  f"{np.mean(agree):.3f}±{np.std(agree, ddof=1):.3f}  "
                  f"pred-true-opt {np.mean(tf_opt):.3f}  teacher-true-opt "
                  f"{np.mean(teach_opt):.3f}", flush=True)
    return summ, per


BLOCKS = {
    'long_horizon': block_long_horizon,
    'nonstationary': block_nonstationary,
    'reward_intervention': block_reward_intervention,
    'size_sweep': block_size_sweep,
}


def main():
    global ALPHA, GAMMA, EPSILON
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model', choices=paths.MODELS, default=None,
                    help='final model; sets --checkpoint, --label and --out_dir '
                         '(figures/<model>/<--out_subdir>)')
    ap.add_argument('--out_subdir', default='closed_loop')
    ap.add_argument('--checkpoint', default=None)
    ap.add_argument('--label', default=None)
    ap.add_argument('--out_dir', default=None)
    ap.add_argument('--parts', default='long_horizon,nonstationary,reward_intervention,size_sweep')
    ap.add_argument('--astar_modes', default='legacy,self')
    ap.add_argument('--n_mdps', type=int, default=30)
    ap.add_argument('--eval_seed', type=int, default=9999)
    ap.add_argument('--n_steps', type=int, default=50)
    ap.add_argument('--long_horizon_steps', type=int, default=1000)
    ap.add_argument('--ri_steps', type=int, default=200)
    ap.add_argument('--size_sweep_steps', type=int, default=200)
    ap.add_argument('--switch_step', type=int, default=50)
    ap.add_argument('--post_steps', type=int, default=100)
    ap.add_argument('--window', type=int, default=10)
    ap.add_argument('--alpha', type=float, default=0.1)
    ap.add_argument('--gamma', type=float, default=0.9)
    ap.add_argument('--epsilon', type=float, default=0.2)
    args = ap.parse_args()
    if args.model:
        args.checkpoint = args.checkpoint or str(paths.checkpoint(args.model))
        args.out_dir = args.out_dir or str(paths.FIGURES / args.model / args.out_subdir)
        args.label = args.label or args.model
    if not (args.checkpoint and args.out_dir and args.label):
        ap.error('give --model, or --checkpoint, --label and --out_dir')
    args.astar_modes = [m for m in args.astar_modes.split(',') if m]
    ALPHA, GAMMA, EPSILON = args.alpha, args.gamma, args.epsilon

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, config = load_model(args.checkpoint, device)
    vocab = E.build_vocab(config.max_states, config.max_actions)
    seeds = list(range(args.eval_seed, args.eval_seed + args.n_mdps))
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"[{args.label}] {os.path.basename(args.checkpoint)}  mode="
          f"{config.context_mode}  device={device}  n_mdps={len(seeds)}  "
          f"astar_modes={args.astar_modes}", flush=True)
    with open(os.path.join(args.out_dir, 'run_args.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)

    for part in args.parts.split(','):
        t0 = time.time()
        print(f"\n=== {part} ===", flush=True)
        summ, per = BLOCKS[part](model, config, vocab, device, args, seeds, args.out_dir)
        for r in summ + per:
            r['model'] = args.label
        write_csv(os.path.join(args.out_dir, f'{part}_summary.csv'), summ)
        write_csv(os.path.join(args.out_dir, f'{part}_per_mdp.csv'), per)
        print(f"=== {part} done in {time.time() - t0:.0f}s ===", flush=True)


ALPHA, GAMMA, EPSILON = 0.1, 0.9, 0.2

if __name__ == '__main__':
    main()
