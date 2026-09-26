#!/usr/bin/env python3
"""
eval_q_probe.py — Which table does the context store?

Teacher-forced rollouts (as in training) on 8-state episodes. After every step,
the context slot c_a of each action a is probed (ridge, fit on half of the
episodes, R^2 on the other half) for the column T(., a) of three tables built
from the same history:

  q_g0.9      the teacher's Q-table (bootstrapped Q-learning, the data's alpha)
  q_g0        myopic incremental reward average (gamma = 0, same alpha)
  counts      reward-blind visit counts N(s, a)
  bootstrap_term  q_g0.9 - q_g0: the part of Q that only bootstrapping produces;
                  a myopic model has nothing here to decode

One probe is shared across actions (the slot layout is the same for every a).
The Q-table construction stores Q(., a) in slot a, so a Q-learning model
should decode q_g0.9 best.

  python3 eval_q_probe.py --checkpoints a.pt b.pt \
      --data ../data/bootstrap_dataset.pt --out ../figures/comparison/qprobe.csv
"""
import argparse
import csv
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import paths  # noqa: E402

import numpy as np
import torch

from eval_clean_steps import M, T

GAMMAS = (0.9, 0.0)


def tables(seq, alpha):
    """Per-step tables AFTER each update: {name: [T, S, A]}."""
    nS, nA = seq['n_states'], seq['n_actions']
    Q = {g: np.zeros((nS, nA), dtype=np.float32) for g in GAMMAS}
    N = np.zeros((nS, nA), dtype=np.float32)
    out = {f'q_g{g:g}': [] for g in GAMMAS}
    out['counts'] = []
    for tr in seq['transitions']:
        s, a, r, s2 = tr['s'], tr['a'], tr['r'], tr['s_next']
        for g in GAMMAS:
            Q[g][s, a] = (1 - alpha) * Q[g][s, a] + alpha * (r + g * Q[g][s2].max())
            out[f'q_g{g:g}'].append(Q[g].copy())
        N[s, a] += 1
        out['counts'].append(N.copy())
    out = {k: np.stack(v) for k, v in out.items()}
    # the part of the teacher's Q that only bootstrapping produces
    out['bootstrap_term'] = out['q_g0.9'] - out['q_g0']
    return out


@torch.no_grad()
def contexts(model, episodes, vocab, device):
    """Context after each step's write: [B, T, n_act, d]."""
    B = len(episodes)
    n_act = episodes[0]['n_actions']
    Tn = len(episodes[0]['transitions'])
    ctx = model.get_init_context(B, n_act, device)
    out = []
    for t in range(Tn):
        toks, rews, acts = [], [], []
        for ep in episodes:
            tr = ep['transitions'][t]
            tl, r_off, s_off, u_off = T.build_step_tokens(tr, vocab, n_act)
            toks.append(tl)
            rews.append(tr['r'])
            acts.append(tr['a'])
        _, upd = model.forward_step(
            token_ids=torch.tensor(toks, dtype=torch.long, device=device),
            reward_value=torch.tensor(rews, dtype=torch.float32, device=device),
            reward_offset=r_off, select_offset=s_off, update_offset=u_off, context=ctx)
        write = model.contextualize(upd)
        ctx = ctx.clone()
        ctx[torch.arange(B), torch.tensor(acts, device=device)] = write
        out.append(ctx.cpu().numpy())
    return np.stack(out, 1)


def ridge_r2(Xtr, Ytr, Xte, Yte, lam=1e-3, return_pred=False):
    mx, my = Xtr.mean(0), Ytr.mean(0)
    A = Xtr - mx
    W = np.linalg.solve(A.T @ A + lam * len(A) * np.eye(A.shape[1]), A.T @ (Ytr - my))
    pred = (Xte - mx) @ W + my
    r2 = float(1 - ((Yte - pred) ** 2).sum() / ((Yte - Yte.mean(0)) ** 2).sum())
    return (r2, pred) if return_pred else r2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoints', nargs='+', default=[str(paths.checkpoint(m)) for m in paths.MODELS])
    ap.add_argument('--data', default=str(paths.DATASET))
    ap.add_argument('--split', default='val')
    ap.add_argument('--n_states', type=int, default=8)
    ap.add_argument('--t_min', type=int, default=10, help='skip the first rounds (tables ~0)')
    ap.add_argument('--out', default=str(paths.FIGURES / 'comparison' / 'q_probe' / 'q_probe.csv'))
    ap.add_argument('--pred_out', default=str(paths.FIGURES / 'comparison' / 'q_probe' / 'q_probe_pred.npz'),
                    help='held-out (true, predicted) Q-values of the q_g0.9 probe, subsampled')
    args = ap.parse_args()
    preds = {}
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    d = torch.load(args.data, map_location='cpu', weights_only=False)
    alpha = float(d['config'].get('alpha', 0.1))
    seqs = [s for s in d[args.split] if s['n_states'] == args.n_states]
    rows = []
    for ck in args.checkpoints:
        c = torch.load(ck, map_location='cpu', weights_only=False)
        model = M.COCONUTTransformer(M.COCONUTConfig.from_dict(c['config']))
        model.load_state_dict(c['model_state_dict'])
        model.to(device).eval()
        vocab = M.build_vocab(model.config.max_states, model.config.max_actions)
        X, Y = [], {}
        # group by (n_actions, length) so a batch shares its shape
        groups = {}
        for s in seqs:
            groups.setdefault((s['n_actions'], len(s['transitions'])), []).append(s)
        ep_id = []
        for (na, Tn), grp in groups.items():
            if Tn <= args.t_min:
                continue
            for b in range(0, len(grp), 256):
                eps = grp[b:b + 256]
                C = contexts(model, eps, vocab, device)[:, args.t_min:]    # [B, T', A, d]
                for i, ep in enumerate(eps):
                    tb = tables(ep, alpha)
                    X.append(C[i].reshape(-1, C.shape[-1]))                 # rows = (t, a)
                    for k, v in tb.items():
                        # column T(., a) for each slot a: [T', A, S]
                        Y.setdefault(k, []).append(
                            v[args.t_min:].transpose(0, 2, 1).reshape(-1, args.n_states))
                    ep_id.append(np.full(C.shape[1] * C.shape[2], len(ep_id)))
        X = np.concatenate(X)
        ep_id = np.concatenate(ep_id)
        tr = ep_id % 2 == 0
        name = os.path.basename(ck)
        print(f'\n=== {name} on {os.path.basename(args.data)}:{args.split} '
              f'(alpha={alpha}, {args.n_states}-state episodes, rows={len(X)})')
        for k, v in Y.items():
            v = np.concatenate(v)
            r2, pred = ridge_r2(X[tr], v[tr], X[~tr], v[~tr], return_pred=True)
            if k == 'q_g0.9':
                idx = np.random.default_rng(0).choice(pred.size, size=min(20000, pred.size), replace=False)
                key = name.replace('.pt', '')
                preds[f'{key}__true'] = v[~tr].reshape(-1)[idx]
                preds[f'{key}__pred'] = pred.reshape(-1)[idx]
            print(f'  probe slot c_a -> {k:8s} column: R^2 = {r2:.3f}')
            rows.append({'checkpoint': name, 'target': k, 'r2': r2, 'alpha': alpha,
                         'n_rows': int(len(X))})
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f'\nwrote {args.out}')
    if args.pred_out:
        np.savez(args.pred_out, **preds)
        print(f'wrote {args.pred_out}')


if __name__ == '__main__':
    main()
