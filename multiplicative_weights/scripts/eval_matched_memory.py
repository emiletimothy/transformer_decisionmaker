#!/usr/bin/env python3
"""
eval_matched_memory.py — Matched-memory comparison for the experts problem.

Compares, on IDENTICAL test sequences:
  * recurrent transformer, continuous latent token   (train.py, continuous)
  * recurrent transformer, discrete latent token     (train.py, discrete)
  * full-history transformer (original paper model; re-reads raw tokens each
    round with a 1024-token sliding window, per-round continuous thoughts
    discarded) — learned_mw_transformer.pt
  * reference algorithms: weighted majority / MW (horizon-tuned eta, same as the
    paper), Bayes-optimal online learner (Beta(1,1) posterior per expert — the
    exact minimiser of the BCE training objective), follow-the-leader, majority
    vote, and a known-qualities oracle.

Mechanistic tests on the recurrent models:
  * linear probes: decode centred cumulative expert losses (= MW log-weights up
    to -eta), per-expert empirical error rate, and best-expert-so-far from M_t
  * latent swap: transplant M from sequence A into sequence B and measure how
    often decisions follow A's best expert; latent reset: replace M by M_0
  * attention: mass on the latent slot and per-head entropy at SEP / UPD rows

Outputs go to --out_dir: results.json, CSVs, and figures.
"""

import argparse
import csv
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import paths  # noqa: E402
from train import (RecurrentMWTransformer, RecurrentConfig, generate_single_sequence,  # noqa: E402
                          encode_rounds, SEP_POS, UPD_POS, ROUND_LEN)
from full_history_model import ContinuousCoTTransformer, MWTokenizer  # noqa: E402
from multiplicative_weights import MultiplicativeWeights  # noqa: E402

COLORS = {'rec_cont': '#2a78d6', 'rec_disc': '#eb6834', 'full_hist': '#1baf7a'}
LABELS = {'rec_cont': 'Recurrent, continuous latent (ours)',
          'rec_disc': 'Recurrent, discrete latent token',
          'full_hist': 'Full raw history (1024-token window)',
          'mw': 'Weighted majority (MW)', 'bayes': 'Bayes-optimal online',
          'ftl': 'Follow the leader', 'majority': 'Majority vote', 'oracle': 'Known-quality oracle'}
REF_STYLE = {'mw': dict(color='#222222', ls='--'), 'bayes': dict(color='#666666', ls='-.'),
             'ftl': dict(color='#999999', ls=':'), 'majority': dict(color='#bbbbbb', ls='-'),
             'oracle': dict(color='#444444', ls=(0, (1, 3)))}


# ---------------------------------------------------------------------------
# Reference algorithms (arrays: preds [T,n], losses [T,n], labels [T])
# ---------------------------------------------------------------------------

def mw_decisions(seq, eta=None):
    n = len(seq['expert_predictions'][0])
    T = len(seq['true_labels'])
    eta = np.sqrt(np.log(n) / max(T, 1)) if eta is None else eta
    mw = MultiplicativeWeights(n, eta)
    out = []
    for t in range(T):
        w = mw.get_probabilities()
        out.append(1 if np.sum(w * np.array(seq['expert_predictions'][t])) > 0.5 else 0)
        mw.update_weights(np.array(seq['losses'][t]))
    return np.array(out)


def logodds_vote(preds, logit_q):
    s = (logit_q[None] if logit_q.ndim == 1 else logit_q) * (2 * preds - 1)
    return (s.sum(-1) > 0).astype(int)


def bayes_decisions(preds, losses):
    wrong = np.vstack([np.zeros((1, preds.shape[1])), np.cumsum(losses, 0)[:-1]])
    t = np.arange(preds.shape[0])[:, None]
    right = t - wrong
    return logodds_vote(preds, np.log((right + 1) / (wrong + 1)))


def ftl_decisions(preds, losses):
    cum = np.vstack([np.zeros((1, preds.shape[1])), np.cumsum(losses, 0)[:-1]])
    best = cum.argmin(1)
    return preds[np.arange(len(best)), best]


def majority_decisions(preds):
    return (preds.mean(1) > 0.5).astype(int)  # ties -> 0


def oracle_decisions(preds, q):
    q = np.clip(np.array(q), 1e-3, 1 - 1e-3)
    return logodds_vote(preds, np.log(q / (1 - q)))


def regret(seq, dec):
    losses = np.array(seq['losses'])
    y = np.array(seq['true_labels'])
    cum = np.cumsum((dec != y).astype(float))
    best = np.cumsum(losses, 0).min(1)
    return cum - best  # unclipped trajectory


# ---------------------------------------------------------------------------
# Model loading / batched rollouts
# ---------------------------------------------------------------------------

def load_recurrent(path, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    m = RecurrentMWTransformer(RecurrentConfig(**ck['config'])).to(device)
    m.load_state_dict(ck['model_state_dict'])
    m.eval()
    return m, ck


def recurrent_decisions(model, seqs, device, bs=200):
    out = []
    for i in range(0, len(seqs), bs):
        ids = torch.from_numpy(np.stack([encode_rounds(s, model.tokenizer, model.UPD_TOKEN)
                                         for s in seqs[i:i + bs]])).to(device)
        with torch.no_grad():
            out.append((model.rollout(ids) > 0).long().cpu().numpy())
    return np.concatenate(out)


def load_full_history(path, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    m = ContinuousCoTTransformer(ck['model_config']).to(device)
    m.load_state_dict(ck['model_state_dict'])
    m.eval()
    return m, MWTokenizer(ck['tokenizer_config']['n_experts'])


def full_history_decisions(model, tok, seqs, device, bs=50):
    """Batched version of eval_long_sequences.generate_long_sequence_continuous_cot
    (same tokens, same sliding-window rule: START + most recent tokens)."""
    max_ctx = model.config.max_sequence_length
    T = len(seqs[0]['true_labels'])
    all_dec = []
    for i in range(0, len(seqs), bs):
        chunk = seqs[i:i + bs]
        toks = [[tok.START_TOKEN] for _ in chunk]
        dec = np.zeros((len(chunk), T), dtype=int)
        for t in range(T):
            for b, s in enumerate(chunk):
                toks[b].append(tok.STEP_TOKENS[t % 100])
                for e, p in enumerate(s['expert_predictions'][t]):
                    toks[b] += [tok.EXPERT_TOKENS[e], tok.PRED_1_TOKEN if p == 1 else tok.PRED_0_TOKEN]
                if len(toks[b]) > max_ctx:
                    overflow = len(toks[b]) - max_ctx
                    toks[b] = [tok.START_TOKEN] + toks[b][overflow + 1:]
            ctx = torch.tensor(toks, dtype=torch.long, device=device)
            with torch.no_grad():
                _, logit = model.think_and_predict(ctx)
            dec[:, t] = (logit[:, 0] > 0).long().cpu().numpy()
            for b, s in enumerate(chunk):
                y = s['true_labels'][t]
                toks[b] += [tok.SEP_TOKEN, tok.PRED_1_TOKEN if y == 1 else tok.PRED_0_TOKEN]
                for e, l in enumerate(s['losses'][t]):
                    toks[b] += [tok.EXPERT_TOKENS[e], tok.discretize_loss(l)]
        all_dec.append(dec)
    return np.concatenate(all_dec)


# ---------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------

# Expert-quality regimes from the LLM experiments (paper, Sec. "Data Generation and
# Expert Regimes"); expert identities are permuted after sampling. Note the
# anti-signal adversary (q < 0.1) is outside the training support q in [0.3, 0.9].
REGIMES = {
    'stratified': [(0.9, 1.0), (0.65, 0.8), (0.55, 0.7), (0.45, 0.6)],
    'flat': [(0.6, 0.7), (0.4, 0.6), (0.4, 0.6), (0.4, 0.6)],
    'anti_signal': [(0.6, 0.7), (0.0, 0.1), (0.4, 0.6), (0.4, 0.6)],
}


def regime_sampler(name):
    def sample(rng):
        q = np.array([rng.uniform(a, b) for a, b in REGIMES[name]])
        return rng.permutation(q)
    return sample


def horizon_eval(models, full_hist, lengths, n_trials, seed, device, sampler=None):
    res = {}
    trajs = {}
    for T in lengths:
        rng = np.random.default_rng(seed + T)
        seqs = [generate_single_sequence(4, T, rng, q=sampler(rng) if sampler else None)
                for _ in range(n_trials)]
        decs = {}
        t0 = time.time()
        for name, mlist in models.items():
            decs[name] = [recurrent_decisions(m, seqs, device) for m in mlist]
        if full_hist is not None:
            decs['full_hist'] = [full_history_decisions(full_hist[0], full_hist[1], seqs, device)]
        P = [np.array(s['expert_predictions']) for s in seqs]
        L = [np.array(s['losses']) for s in seqs]
        decs['mw'] = [np.stack([mw_decisions(s) for s in seqs])]
        decs['bayes'] = [np.stack([bayes_decisions(p, l) for p, l in zip(P, L)])]
        decs['ftl'] = [np.stack([ftl_decisions(p, l) for p, l in zip(P, L)])]
        decs['majority'] = [np.stack([majority_decisions(p) for p in P])]
        decs['oracle'] = [np.stack([oracle_decisions(p, s['qualities']) for p, s in zip(P, seqs)])]
        res[T] = {}
        for name, dlist in decs.items():
            per_seed_final, per_seed_acc, per_seed_traj = [], [], []
            for d in dlist:
                r = np.stack([regret(s, d[i]) for i, s in enumerate(seqs)])  # [n, T]
                acc = (d == np.array([s['true_labels'] for s in seqs])).mean(1)
                per_seed_final.append(r[:, -1])
                per_seed_acc.append(acc)
                per_seed_traj.append(r.mean(0))
            fin = np.stack(per_seed_final)  # [seeds, n]
            acc = np.stack(per_seed_acc)
            res[T][name] = {
                'regret_mean': float(fin.mean()),
                'regret_sem': float(fin.mean(0).std(ddof=1) / np.sqrt(fin.shape[1])),
                'regret_clipped_mean': float(np.maximum(fin, 0).mean()),
                'regret_seed_means': fin.mean(1).tolist(),
                'acc_mean': float(acc.mean()),
                'acc_sem': float(acc.mean(0).std(ddof=1) / np.sqrt(acc.shape[1])),
                'n_seeds': len(dlist), 'n_trials': n_trials,
            }
            if T == max(lengths):
                trajs[name] = np.stack(per_seed_traj).mean(0).tolist()
        print(f'[horizon] T={T} done in {time.time() - t0:.0f}s: ' +
              ', '.join(f"{k}={v['regret_mean']:.2f}" for k, v in res[T].items()), flush=True)
    return res, trajs


def ridge_r2(Xtr, Ytr, Xte, Yte, lam=1e-2):
    mx, my = Xtr.mean(0), Ytr.mean(0)
    A = Xtr - mx
    W = np.linalg.solve(A.T @ A + lam * len(A) * np.eye(A.shape[1]), A.T @ (Ytr - my))
    pred = (Xte - mx) @ W + my
    ss_res = ((Yte - pred) ** 2).sum(0)
    ss_tot = ((Yte - Yte.mean(0)) ** 2).sum(0)
    return float(np.mean(1 - ss_res / np.maximum(ss_tot, 1e-12)))


def logreg_acc(Xtr, ytr, Xte, yte, n_cls=4, iters=500):
    Xtr_t, Xte_t = torch.tensor(Xtr, dtype=torch.float32), torch.tensor(Xte, dtype=torch.float32)
    mu, sd = Xtr_t.mean(0), Xtr_t.std(0) + 1e-6
    Xtr_t, Xte_t = (Xtr_t - mu) / sd, (Xte_t - mu) / sd
    lin = torch.nn.Linear(Xtr.shape[1], n_cls)
    opt = torch.optim.Adam(lin.parameters(), lr=0.05, weight_decay=1e-4)
    yt = torch.tensor(ytr)
    for _ in range(iters):
        opt.zero_grad()
        F.cross_entropy(lin(Xtr_t), yt).backward()
        opt.step()
    with torch.no_grad():
        return float((lin(Xte_t).argmax(1).numpy() == yte).mean())


def probe_eval(model, device, n_seq=300, T=200, seed=777, t_min=10):
    rng = np.random.default_rng(seed)
    seqs = [generate_single_sequence(4, T, rng) for _ in range(n_seq)]
    ids = torch.from_numpy(np.stack([encode_rounds(s, model.tokenizer, model.UPD_TOKEN)
                                     for s in seqs])).to(device)
    with torch.no_grad():
        _, M = model.rollout(ids, return_latents=True)  # M[:, t] = latent after round t
    M = M.cpu().numpy()
    L = np.cumsum(np.array([s['losses'] for s in seqs]), 1)  # [n, T, 4]
    t = np.arange(1, T + 1)[None, :, None]
    targets = {
        'centered_cum_loss': L - L.mean(-1, keepdims=True),
        'empirical_error_rate': L / t,
        'true_quality': np.broadcast_to(
            np.array([s['qualities'] for s in seqs])[:, None, :], L.shape),
    }
    ntr = int(0.67 * n_seq)
    sl = slice(t_min - 1, None)
    Xtr = M[:ntr, sl].reshape(-1, M.shape[-1])
    Xte = M[ntr:, sl].reshape(-1, M.shape[-1])
    out = {}
    for k, Y in targets.items():
        Ytr, Yte = Y[:ntr, sl].reshape(-1, 4), Y[ntr:, sl].reshape(-1, 4)
        out[f'r2_{k}'] = ridge_r2(Xtr, Ytr, Xte, Yte)
    best = L.argmin(-1)
    ytr, yte = best[:ntr, sl].reshape(-1), best[ntr:, sl].reshape(-1)
    out['best_expert_acc'] = logreg_acc(Xtr, ytr, Xte, yte)
    out['best_expert_chance'] = float(np.bincount(yte, minlength=4).max() / len(yte))
    # Control: the same probes on a shuffled pairing of latents and targets.
    perm = np.random.default_rng(0).permutation(len(Xte))
    Yte = targets['centered_cum_loss'][ntr:, sl].reshape(-1, 4)
    out['r2_centered_cum_loss_shuffled'] = ridge_r2(
        Xtr, targets['centered_cum_loss'][:ntr, sl].reshape(-1, 4), Xte, Yte[perm])
    if model.cfg.context_mode == 'discrete':
        with torch.no_grad():
            tok_ids = []
            for i in range(0, n_seq, 100):
                b = ids[i:i + 100]
                Mb = model.initial_latent(b.shape[0])
                for tt in range(T):
                    _, Mb, h, _ = model.step(Mb, b[:, tt])
                    tok_ids.append(model.latent_token_ids(h).cpu().numpy())
        tok_ids = np.concatenate(tok_ids)
        cnt = np.bincount(tok_ids)
        p = cnt[cnt > 0] / cnt.sum()
        out['n_distinct_latent_tokens'] = int((cnt > 0).sum())
        out['latent_token_entropy_bits'] = float(-(p * np.log2(p)).sum())
    return out


def swap_eval(model, device, n_pairs=400, T_pre=50, T_post=20, seed=4242):
    """Transplant latent from sequence A into sequence B after T_pre rounds."""
    rng = np.random.default_rng(seed)
    A, B = [], []
    while len(A) < n_pairs:
        a = generate_single_sequence(4, T_pre + T_post, rng)
        b = generate_single_sequence(4, T_pre + T_post, rng)
        qa, qb = np.sort(a['qualities']), np.sort(b['qualities'])
        if (np.argmax(a['qualities']) != np.argmax(b['qualities']) and
                qa[-1] - qa[-2] > 0.15 and qb[-1] - qb[-2] > 0.15):
            A.append(a)
            B.append(b)
    enc = lambda S: torch.from_numpy(np.stack([encode_rounds(s, model.tokenizer, model.UPD_TOKEN)
                                               for s in S])).to(device)
    ia, ib = enc(A), enc(B)
    with torch.no_grad():
        _, Ma = model.rollout(ia[:, :T_pre], return_latents=True)
        _, Mb = model.rollout(ib[:, :T_pre], return_latents=True)
        post = ib[:, T_pre:]
        conds = {
            'control': model.rollout(post, M0=Mb[:, -1]),
            'swap': model.rollout(post, M0=Ma[:, -1]),
            'reset': model.rollout(post, M0=model.initial_latent(len(B))),
        }
    bestA = np.array([np.argmax(a['qualities']) for a in A])
    bestB = np.array([np.argmax(b['qualities']) for b in B])
    PB = np.array([b['expert_predictions'] for b in B])[:, T_pre:]  # [n, T_post, 4]
    yB = np.array([b['true_labels'] for b in B])[:, T_pre:]
    pa = PB[np.arange(n_pairs), :, bestA]
    pb = PB[np.arange(n_pairs), :, bestB]
    disagree = pa != pb
    out = {}
    for k, lg in conds.items():
        d = (lg > 0).long().cpu().numpy()
        follow_A = [(d[:, j] == pa[:, j])[disagree[:, j]].mean() for j in range(T_post)]
        out[k] = {'follow_A_by_round': [float(x) for x in follow_A],
                  'acc_post': float((d == yB).mean()),
                  'acc_first5': float((d[:, :5] == yB[:, :5]).mean())}
    # MW reference: same transplant on the exact algorithm (eta tuned to T_pre+T_post).
    eta = np.sqrt(np.log(4) / (T_pre + T_post))
    LA = np.array([a['losses'] for a in A])[:, :T_pre].sum(1)
    LB = np.array([b['losses'] for b in B])[:, :T_pre].sum(1)
    lossB_post = np.array([b['losses'] for b in B])[:, T_pre:]
    for k, L0 in [('mw_control', LB), ('mw_swap', LA)]:
        Lc = L0.copy()
        d = np.zeros((n_pairs, T_post), dtype=int)
        for j in range(T_post):
            w = np.exp(-eta * (Lc - Lc.min(1, keepdims=True)))
            w /= w.sum(1, keepdims=True)
            d[:, j] = ((w * PB[:, j]).sum(1) > 0.5).astype(int)
            Lc += lossB_post[:, j]
        follow_A = [(d[:, j] == pa[:, j])[disagree[:, j]].mean() for j in range(T_post)]
        out[k] = {'follow_A_by_round': [float(x) for x in follow_A],
                  'acc_post': float((d == yB).mean())}
    return out


def attention_eval(model, device, n_seq=100, T=100, seed=99):
    rng = np.random.default_rng(seed)
    seqs = [generate_single_sequence(4, T, rng) for _ in range(n_seq)]
    ids = torch.from_numpy(np.stack([encode_rounds(s, model.tokenizer, model.UPD_TOKEN)
                                     for s in seqs])).to(device)
    lat_mass = np.zeros((model.cfg.n_layers, model.cfg.n_heads))
    ent = {'SEP': np.zeros_like(lat_mass), 'UPD': np.zeros_like(lat_mass)}
    maxw = {'SEP': np.zeros_like(lat_mass), 'UPD': np.zeros_like(lat_mass)}
    with torch.no_grad():
        M = model.initial_latent(n_seq)
        for t in range(T):
            _, M, _, attn = model.step(M, ids[:, t], return_attention=True)
            for li, a in enumerate(attn):  # a: [B, H, 20, 20]
                lat_mass[li] += a[:, :, SEP_POS, 0].mean(0).cpu().numpy()
                for nm, pos in [('SEP', SEP_POS), ('UPD', UPD_POS)]:
                    row = a[:, :, pos, :pos + 1].clamp_min(1e-12)
                    h = -(row * row.log()).sum(-1) / np.log(pos + 1)
                    ent[nm][li] += h.mean(0).cpu().numpy()
                    maxw[nm][li] += row.max(-1).values.mean(0).cpu().numpy()
    return {'sep_attention_mass_on_latent': (lat_mass / T).tolist(),
            'normalized_entropy': {k: (v / T).tolist() for k, v in ent.items()},
            'max_attention_weight': {k: (v / T).tolist() for k, v in maxw.items()}}


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_all(res, trajs, swaps, probes, out_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.grid': True, 'grid.color': '#e6e6e6', 'grid.linewidth': 0.6})
    lengths = sorted(res.keys())
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))

    ax = axes[0]
    for name in list(COLORS) + list(REF_STYLE):
        if name not in res[lengths[0]]:
            continue
        m = np.array([res[T][name]['regret_mean'] for T in lengths])
        e = np.array([res[T][name]['regret_sem'] for T in lengths])
        kw = dict(color=COLORS[name], lw=2, marker='o', ms=4) if name in COLORS else \
            dict(lw=1.5, **REF_STYLE[name])
        ax.plot(lengths, m, label=LABELS[name], **kw)
        if name in COLORS:
            ax.fill_between(lengths, m - e, m + e, color=COLORS[name], alpha=0.15, lw=0)
    ax.set_xscale('log')
    ax.set_xticks(lengths)
    ax.set_xticklabels([str(T) for T in lengths])
    ax.minorticks_off()
    ax.set_xlabel('horizon T (rounds)')
    ax.set_ylabel('final regret vs best expert')
    ax.set_title('Regret vs horizon (mean ± SEM)', loc='left')

    ax = axes[1]
    Tmax = max(lengths)
    x = np.arange(1, Tmax + 1)
    for name in list(COLORS) + ['mw', 'bayes']:
        if name in trajs:
            kw = dict(color=COLORS[name], lw=2) if name in COLORS else dict(lw=1.5, **REF_STYLE[name])
            ax.plot(x, trajs[name], label=LABELS[name], **kw)
    ax.set_xlabel('round t')
    ax.set_ylabel('cumulative regret')
    ax.set_title(f'Regret trajectory, T={Tmax}', loc='left')

    ax = axes[2]
    for name, sw in swaps.items():
        j = np.arange(1, len(sw['swap']['follow_A_by_round']) + 1)
        ax.plot(j, sw['swap']['follow_A_by_round'], color=COLORS[name], lw=2, marker='o', ms=4,
                label=f'{LABELS[name]}: swap')
        ax.plot(j, sw['control']['follow_A_by_round'], color=COLORS[name], lw=1.2, ls=':',
                label=f'{LABELS[name]}: control')
    first = next(iter(swaps.values()))
    ax.plot(j, first['mw_swap']['follow_A_by_round'], lw=1.5, label='MW: swap', **REF_STYLE['mw'])
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel('rounds after transplant')
    ax.set_ylabel("P(decision = A's best expert)")
    ax.set_title('Latent transplant A → B', loc='left')

    handles, labels = [], []
    for a in axes:
        for h, l in zip(*a.get_legend_handles_labels()):
            if l not in labels:
                handles.append(h)
                labels.append(l)
    fig.tight_layout(rect=(0, 0.2, 1, 1))
    fig.legend(handles, labels, loc='lower center', ncol=4, frameon=False, fontsize=8,
               bbox_to_anchor=(0.5, 0.0))
    fig.savefig(os.path.join(out_dir, 'matched_memory_mwu.png'), dpi=200, bbox_inches='tight')
    fig.savefig(os.path.join(out_dir, 'matched_memory_mwu.pdf'), bbox_inches='tight')
    plt.close(fig)


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rec_cont', nargs='*',
                    default=[str(paths.checkpoint('continuous_residual', s)) for s in paths.SEEDS])
    ap.add_argument('--rec_disc', nargs='*',
                    default=[str(paths.checkpoint('discrete', s)) for s in paths.SEEDS])
    ap.add_argument('--full_hist', type=str, default=str(paths.FULL_HISTORY))
    ap.add_argument('--lengths', type=int, nargs='+', default=[10, 20, 50, 100, 200, 500, 1000])
    ap.add_argument('--n_trials', type=int, default=100)
    ap.add_argument('--full_hist_max_T', type=int, default=1000)
    ap.add_argument('--regime_T', type=int, default=100)
    ap.add_argument('--seed', type=int, default=2026)
    ap.add_argument('--out_dir', type=str, default=str(paths.FIGURES / 'comparison' / 'matched_memory'))
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device', device, flush=True)

    models = {}
    for key, ckpts in [('rec_cont', args.rec_cont), ('rec_disc', args.rec_disc)]:
        if ckpts:
            models[key] = [load_recurrent(p, device)[0] for p in ckpts]
    full_hist = load_full_history(args.full_hist, device) if args.full_hist else None

    results = {'args': vars(args)}
    t0 = time.time()
    # Mechanistic evals first (fast), then the horizon sweep.
    results['probe'], results['swap'], results['attention'] = {}, {}, {}
    for key, mlist in models.items():
        results['probe'][key] = [probe_eval(m, device) for m in mlist]
        results['swap'][key] = [swap_eval(m, device) for m in mlist]
        results['attention'][key] = [attention_eval(m, device) for m in mlist]
        print(f'[mech] {key}: probe={results["probe"][key]}', flush=True)
        print(f'[mech] {key}: swap acc/follow_A(first round) = ' +
              str([(s['control']['acc_post'], s['swap']['follow_A_by_round'][0],
                    s['control']['follow_A_by_round'][0], s['reset']['acc_first5'])
                   for s in results['swap'][key]]), flush=True)
    with open(os.path.join(args.out_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=1)

    lengths = args.lengths
    fh = full_hist
    res, trajs = {}, {}
    for T in lengths:
        r, tr = horizon_eval(models, fh if T <= args.full_hist_max_T else None,
                             [T], args.n_trials, args.seed, device)
        res.update(r)
        if T == max(lengths):
            trajs = tr
        results['horizon'] = {str(k): v for k, v in res.items()}
        with open(os.path.join(args.out_dir, 'results.json'), 'w') as f:
            json.dump(results, f, indent=1)
    # Out-of-distribution expert regimes (same as the LLM experiments), T=100.
    results['regimes'] = {}
    for i, name in enumerate(REGIMES):
        r, _ = horizon_eval(models, fh, [args.regime_T], args.n_trials, args.seed + 1000 * (i + 1),
                            device, sampler=regime_sampler(name))
        results['regimes'][name] = r[args.regime_T]
        print(f'[regime] {name}: ' + ', '.join(
            f"{k}={v['regret_mean']:.2f}/{v['acc_mean']:.3f}" for k, v in r[args.regime_T].items()),
            flush=True)
        with open(os.path.join(args.out_dir, 'results.json'), 'w') as f:
            json.dump(results, f, indent=1)
    with open(os.path.join(args.out_dir, 'regime_summary.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['regime', 'method', 'regret_mean', 'regret_sem', 'acc_mean', 'acc_sem',
                    'n_seeds', 'n_trials'])
        for name, rr in results['regimes'].items():
            for m, v in rr.items():
                w.writerow([name, m, f"{v['regret_mean']:.3f}", f"{v['regret_sem']:.3f}",
                            f"{v['acc_mean']:.4f}", f"{v['acc_sem']:.4f}", v['n_seeds'], v['n_trials']])
    results['trajectory_T'] = max(lengths)
    results['trajectories'] = trajs
    with open(os.path.join(args.out_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=1)

    with open(os.path.join(args.out_dir, 'horizon_summary.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['T', 'method', 'regret_mean', 'regret_sem', 'regret_clipped_mean',
                    'acc_mean', 'acc_sem', 'n_seeds', 'n_trials'])
        for T in lengths:
            for name, v in res[T].items():
                w.writerow([T, name, f"{v['regret_mean']:.3f}", f"{v['regret_sem']:.3f}",
                            f"{v['regret_clipped_mean']:.3f}", f"{v['acc_mean']:.4f}",
                            f"{v['acc_sem']:.4f}", v['n_seeds'], v['n_trials']])
    swaps = {k: v[0] for k, v in results['swap'].items()}
    plot_all(res, trajs, swaps, results['probe'], args.out_dir)
    print(f'done in {(time.time() - t0) / 60:.1f} min', flush=True)


if __name__ == '__main__':
    main()
