#!/usr/bin/env python3
"""
eval_mechanism.py — Is the recurrent latent executing something like MWU?

Four tests on a train.py checkpoint, all on fresh sequences from the
training distribution (qualities ~ U[0.3, 0.9]) rolled out past the training
horizon:

 1. Probes by round. Ridge and 1-hidden-layer MLP probes from the latent M_t to
      mw_state      centred cumulative loss L_t - mean_i L_t  (MWU log-weights / -eta)
      bayes_logodds log((right+1)/(wrong+1)) per expert  (BCE-optimal state)
      leaky_rho     centred discounted loss sum_s rho^(t-s) l_s, rho in a grid
    Probes are fit on rounds [t_min, T_train] and scored per round bucket, so
    we can see whether the state is kept or decays past the training horizon.
    The best rho gives the latent's effective memory length 1 / (1 - rho).
 2. Update rule. With the mw_state probe, decode s_t and fit
      s_t = rho * s_{t-1} + k * centred(l_t)
    Exact MWU has rho = 1 (no forgetting) and a constant step k.
 3. Behavioural fit. Grid over MWU learning rate eta and forgetting rho for the
    rule "predict sign(sum_i softmax(-eta L^rho)_i (2 p_i - 1))"; report the
    best fit's decision agreement with the model overall and on the rounds
    where memory matters (majority vote and the Bayes rule disagree).
 4. Causal steering. Move M_t along the minimum-norm direction that raises the
    decoded mw_state of one expert by delta (more loss => MWU trusts it less)
    and measure how often the next decision follows that expert, on rounds
    where it disagrees with the other experts' majority.

  python3 eval_mechanism.py --ckpts a.pt b.pt --labels cont_v3 disc_v3 \
      --out ../figures/comparison/mechanism/mech.json
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import paths  # noqa: E402
from train import generate_single_sequence, encode_rounds  # noqa: E402
from eval_matched_memory import load_recurrent  # noqa: E402

BUCKETS = [(1, 10), (11, 30), (31, 95), (96, 200), (201, 400)]
RHOS = [0.8, 0.9, 0.95, 0.98, 0.99, 0.995, 1.0]


def rollout(model, seqs, device, bs=200):
    """Latent after each round [n, T, d] and decision logits [n, T]."""
    Ms, Zs = [], []
    for i in range(0, len(seqs), bs):
        ids = torch.from_numpy(np.stack([encode_rounds(s, model.tokenizer, model.UPD_TOKEN)
                                         for s in seqs[i:i + bs]])).to(device)
        with torch.no_grad():
            z, M = model.rollout(ids, return_latents=True)
        Ms.append(M.cpu().numpy())
        Zs.append(z.cpu().numpy())
    return np.concatenate(Ms), np.concatenate(Zs)


def centred(x):
    return x - x.mean(-1, keepdims=True)


def leaky_cum(losses, rho):
    out = np.zeros_like(losses)
    acc = np.zeros(losses.shape[::2])
    for t in range(losses.shape[1]):
        acc = rho * acc + losses[:, t]
        out[:, t] = acc
    return out


class Ridge:
    def __init__(self, X, Y, lam=1e-2):
        self.mx, self.my = X.mean(0), Y.mean(0)
        A = X - self.mx
        self.W = np.linalg.solve(A.T @ A + lam * len(A) * np.eye(A.shape[1]), A.T @ (Y - self.my))

    def __call__(self, X):
        return (X - self.mx) @ self.W + self.my


def fit_mlp(X, Y, device, hidden=256, epochs=60, bs=4096, lr=2e-3):
    X = torch.tensor(X, dtype=torch.float32, device=device)
    Y = torch.tensor(Y, dtype=torch.float32, device=device)
    mx, sx, my, sy = X.mean(0), X.std(0) + 1e-6, Y.mean(0), Y.std(0) + 1e-6
    net = nn.Sequential(nn.Linear(X.shape[1], hidden), nn.GELU(),
                        nn.Linear(hidden, hidden), nn.GELU(),
                        nn.Linear(hidden, Y.shape[1])).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-5)
    Xn, Yn = (X - mx) / sx, (Y - my) / sy
    for _ in range(epochs):
        perm = torch.randperm(len(Xn), device=device)
        for i in range(0, len(Xn), bs):
            j = perm[i:i + bs]
            opt.zero_grad()
            ((net(Xn[j]) - Yn[j]) ** 2).mean().backward()
            opt.step()

    def predict(Xq):
        with torch.no_grad():
            Xq = torch.tensor(Xq, dtype=torch.float32, device=device)
            return (net((Xq - mx) / sx) * sy + my).cpu().numpy()
    return predict


def r2(pred, Y):
    ss_res = ((Y - pred) ** 2).sum(0)
    ss_tot = ((Y - Y.mean(0)) ** 2).sum(0)
    return float(np.mean(1 - ss_res / np.maximum(ss_tot, 1e-12)))


def probe_tests(M, targets, ntr, t_min, T_train, device):
    """Fit on train sequences, rounds [t_min, T_train]; score per bucket on held-out."""
    d = M.shape[-1]
    fit_sl = slice(t_min - 1, T_train)
    Xtr = M[:ntr, fit_sl].reshape(-1, d)
    out = {}
    for name, Y in targets.items():
        Ytr = Y[:ntr, fit_sl].reshape(-1, Y.shape[-1])
        probes = {'linear': Ridge(Xtr, Ytr), 'mlp': fit_mlp(Xtr, Ytr, device)}
        out[name] = {}
        for pname, f in probes.items():
            row = {}
            for a, b in BUCKETS:
                if b > M.shape[1]:
                    continue
                sl = slice(max(a, t_min) - 1, b)
                Xte = M[ntr:, sl].reshape(-1, d)
                Yte = Y[ntr:, sl].reshape(-1, Y.shape[-1])
                row[f'{a}-{b}'] = r2(f(Xte), Yte)
            out[name][pname] = row
    return out


def update_rule(M, L, ntr, t_min, T_train):
    """Decode s_t with a linear mw_state probe; fit s_t = rho s_{t-1} + k c(l_t)."""
    losses = np.diff(np.concatenate([np.zeros_like(L[:, :1]), L], 1), axis=1)
    S = centred(L)
    d = M.shape[-1]
    fit_sl = slice(t_min - 1, T_train)
    probe = Ridge(M[:ntr, fit_sl].reshape(-1, d), S[:ntr, fit_sl].reshape(-1, 4))
    out = {}
    for a, b in [(t_min + 1, T_train), (T_train + 1, M.shape[1])]:
        if a > M.shape[1]:
            continue
        s_hat = probe(M[ntr:].reshape(-1, d)).reshape(M[ntr:].shape[0], M.shape[1], 4)
        prev = s_hat[:, a - 2:b - 1].reshape(-1)
        cur = s_hat[:, a - 1:b].reshape(-1)
        inc = centred(losses[ntr:, a - 1:b]).reshape(-1)
        A = np.stack([prev, inc, np.ones_like(prev)], 1)
        coef, *_ = np.linalg.lstsq(A, cur, rcond=None)
        pred = A @ coef
        # the same regression on the TRUE MWU state (sanity: rho=1, k=1)
        tprev = S[ntr:, a - 2:b - 1].reshape(-1)
        tcur = S[ntr:, a - 1:b].reshape(-1)
        tcoef, *_ = np.linalg.lstsq(np.stack([tprev, inc, np.ones_like(tprev)], 1), tcur, rcond=None)
        out[f'{a}-{b}'] = {'rho': float(coef[0]), 'k': float(coef[1]),
                           'r2': float(1 - ((cur - pred) ** 2).sum() / ((cur - cur.mean()) ** 2).sum()),
                           'true_state_rho': float(tcoef[0]), 'true_state_k': float(tcoef[1])}
    return out


def behaviour_fit(Z, P, losses, y, T_train):
    dec = (Z > 0).astype(int)
    vote = 2 * P - 1                                          # [n, T, 4]
    cumL = np.concatenate([np.zeros_like(losses[:, :1]), np.cumsum(losses, 1)[:, :-1]], 1)
    right = np.arange(losses.shape[1])[None, :, None] - cumL
    bayes = ((np.log((right + 1) / (cumL + 1)) * vote).sum(-1) > 0).astype(int)
    majority = (P.mean(-1) > 0.5).astype(int)
    memory_steps = bayes != majority
    res = {'agree_majority': float((dec == majority).mean()),
           'agree_bayes': float((dec == bayes).mean()),
           'acc_model': float((dec == y).mean()),
           'frac_memory_steps': float(memory_steps.mean()),
           'memory_steps': {'agree_majority': float((dec == majority)[memory_steps].mean()),
                            'agree_bayes': float((dec == bayes)[memory_steps].mean())}}
    grid = []
    for rho in RHOS:
        prevL = np.concatenate([np.zeros_like(losses[:, :1]),
                                leaky_cum(losses, rho)[:, :-1]], 1)
        for eta in [0.03, 0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.2, 2.0]:
            w = np.exp(-eta * (prevL - prevL.min(-1, keepdims=True)))
            mw = ((w * vote).sum(-1) > 0).astype(int)
            grid.append({'rho': rho, 'eta': eta,
                         'agree': float((dec == mw).mean()),
                         'agree_memory_steps': float((dec == mw)[memory_steps].mean()),
                         'agree_in_horizon': float((dec == mw)[:, :T_train].mean()),
                         'agree_beyond_horizon': float((dec == mw)[:, T_train:].mean())})
    best = max(grid, key=lambda g: g['agree_memory_steps'])
    best_exact = max((g for g in grid if g['rho'] == 1.0), key=lambda g: g['agree_memory_steps'])
    res.update({'best_fit': best, 'best_exact_mwu': best_exact, 'grid': grid})
    return res


def steering(model, seqs, M, P, L, ntr, t_min, T_train, device,
             fracs=(-1, -0.5, -0.25, -0.1, 0, 0.1, 0.25, 0.5, 1), t_points=(30, 80)):
    """Shift M_t by frac * |M_t| along a probe direction for expert i; measure
    how often the next decision follows expert i (rounds where i disagrees with
    the other experts' majority). Directions:
      mw_state  'expert i has MORE loss'          MWU predicts follow-rate falls
      bayes     'expert i has HIGHER log-odds'    predicts follow-rate rises
      random    same-norm random direction       control
    """
    d = M.shape[-1]
    fit_sl = slice(t_min - 1, T_train)
    Xfit = M[:ntr, fit_sl].reshape(-1, d)
    right = np.arange(1, L.shape[1] + 1)[None, :, None] - L
    W_mw = Ridge(Xfit, centred(L)[:ntr, fit_sl].reshape(-1, 4)).W
    W_by = Ridge(Xfit, np.log((right + 1) / (L + 1))[:ntr, fit_sl].reshape(-1, 4)).W
    ids = torch.from_numpy(np.stack([encode_rounds(s, model.tokenizer, model.UPD_TOKEN)
                                     for s in seqs[ntr:]])).to(device)
    out = {}
    for t in t_points:
        Mt = torch.tensor(M[ntr:, t - 1], dtype=torch.float32, device=device)
        norm = float(np.linalg.norm(M[ntr:, t - 1], axis=1).mean())
        nxt = P[ntr:, t]                                        # predictions at round t+1
        res = {}
        for i in range(4):
            others = np.delete(nxt, i, axis=1).mean(1)
            mask = (np.abs(others - 0.5) > 0.1) & (nxt[:, i] != (others > 0.5))
            # mw_state is centred (columns sum to 0): achievable change is
            # 'expert i up by 1, the others down by 1/3 each'
            e = -np.full(4, 1 / 3)
            e[i] = 1.0
            dirs = {'mw_state': np.linalg.pinv(W_mw.T) @ e,
                    'bayes': np.linalg.pinv(W_by.T) @ np.eye(4)[i],
                    'random': np.random.default_rng(i).standard_normal(d)}
            for name, vec in dirs.items():
                vec = torch.tensor(vec / np.linalg.norm(vec) * norm, dtype=torch.float32,
                                   device=device)
                for f in fracs:
                    with torch.no_grad():
                        z, _, _, _ = model.step(Mt + f * vec, ids[:, t])
                    follow = ((z > 0).long().cpu().numpy() == nxt[:, i])[mask]
                    res.setdefault(f'{name} {f:g}', []).append(float(follow.mean()))
        out[f't={t}'] = {k: float(np.mean(v)) for k, v in res.items()}
    return out


def main():
    ap = argparse.ArgumentParser()
    finals = [('continuous_residual', 'cont_v5res'), ('discrete', 'disc_v5')]
    ap.add_argument('--ckpts', nargs='+',
                    default=[str(paths.checkpoint(m, s)) for m, _ in finals for s in paths.SEEDS])
    ap.add_argument('--labels', nargs='+',
                    default=[f'{l}_s{s}' for _, l in finals for s in paths.SEEDS])
    ap.add_argument('--n_seq', type=int, default=1200)
    ap.add_argument('--T', type=int, default=400)
    ap.add_argument('--T_train', type=int, default=95)
    ap.add_argument('--t_min', type=int, default=5)
    ap.add_argument('--seed', type=int, default=2026)
    ap.add_argument('--out', default=str(paths.FIGURES / 'comparison' / 'mechanism' / 'mech_final.json'))
    args = ap.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    rng = np.random.default_rng(args.seed)
    seqs = [generate_single_sequence(4, args.T, rng) for _ in range(args.n_seq)]
    P = np.array([s['expert_predictions'] for s in seqs])
    losses = np.array([s['losses'] for s in seqs])
    y = np.array([s['true_labels'] for s in seqs])
    L = np.cumsum(losses, 1)                                   # state AFTER round t
    right = np.arange(1, args.T + 1)[None, :, None] - L
    targets = {'mw_state': centred(L),
               'bayes_logodds': np.log((right + 1) / (L + 1))}
    for rho in RHOS[:-1]:
        targets[f'leaky_{rho}'] = centred(leaky_cum(losses, rho))
    ntr = args.n_seq // 2

    results = {}
    for path, label in zip(args.ckpts, args.labels):
        model, _ = load_recurrent(path, device)
        M, Z = rollout(model, seqs, device)
        r = {'probes': probe_tests(M, targets, ntr, args.t_min, args.T_train, device),
             'update_rule': update_rule(M, L, ntr, args.t_min, args.T_train),
             'behaviour': behaviour_fit(Z[ntr:], P[ntr:], losses[ntr:], y[ntr:], args.T_train),
             'steering': steering(model, seqs, M, P, L, ntr, args.t_min, args.T_train, device)}
        results[label] = r

        print(f'\n===== {label}  ({path})')
        print('  probe R^2 (held-out) by round bucket; fit on rounds '
              f'{args.t_min}-{args.T_train}')
        for name, pr in r['probes'].items():
            for pname, row in pr.items():
                print(f'    {name:14s} {pname:6s} ' +
                      '  '.join(f'{k}:{v:6.3f}' for k, v in row.items()))
        print('  update rule s_t = rho s_{t-1} + k c(l_t):')
        for k, v in r['update_rule'].items():
            print(f'    rounds {k:8s} rho={v["rho"]:.3f} k={v["k"]:.3f} R2={v["r2"]:.3f} '
                  f'(true MWU state: rho={v["true_state_rho"]:.3f} k={v["true_state_k"]:.3f})')
        b = r['behaviour']
        bf, be = b['best_fit'], b['best_exact_mwu']
        print(f'  behaviour: acc={b["acc_model"]:.3f}  agree majority={b["agree_majority"]:.3f} '
              f'bayes={b["agree_bayes"]:.3f}  | memory steps ({b["frac_memory_steps"]:.2f} of rounds): '
              f'majority={b["memory_steps"]["agree_majority"]:.3f} '
              f'bayes={b["memory_steps"]["agree_bayes"]:.3f}')
        print(f'    best leaky MWU: rho={bf["rho"]} eta={bf["eta"]} agree={bf["agree"]:.3f} '
              f'memory-steps={bf["agree_memory_steps"]:.3f} '
              f'(in/beyond horizon {bf["agree_in_horizon"]:.3f}/{bf["agree_beyond_horizon"]:.3f})')
        print(f'    best exact MWU (rho=1): eta={be["eta"]} agree={be["agree"]:.3f} '
              f'memory-steps={be["agree_memory_steps"]:.3f} '
              f'(in/beyond horizon {be["agree_in_horizon"]:.3f}/{be["agree_beyond_horizon"]:.3f})')
        print('  steering: follow-rate of expert i after shifting M_t by frac*|M_t| along:')
        for k, v in r['steering'].items():
            print(f'    {k}')
            for name in ('mw_state', 'bayes', 'random'):
                print(f'      {name:8s} ' + '  '.join(f'{dl.split()[1]}:{f:.3f}' for dl, f in v.items()
                                                    if dl.startswith(name + ' ')))

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(results, f, indent=1)
    print(f'\nwrote {args.out}')


if __name__ == '__main__':
    main()
