#!/usr/bin/env python3
"""
train.py — Recurrent latent-context transformer for expert prediction.

Unlike ContinuousCoTTransformer (which re-reads the full raw token history at
every step and discards its thought vectors), this model processes ONE round at
a time. A single latent context token M_{t-1} is the only channel carrying
information across rounds, matching the Weighted-Majority construction (log
weights stored in one latent token) and the tabular Q-learning model
(tabular_q_learning/scripts/2_model.py).

Token layout per round t (20 positions, in-round positional embeddings only):

  [M_{t-1},  E1 p1  E2 p2  E3 p3  E4 p4,  SEP,  y_t,  E1 l1  E2 l2  E3 l3  E4 l4,  UPD]
   pos 0     1 ............... 8          9     10    11 ............... 18       19

  - decision logit is read at SEP (causal mask => it has not seen y_t or losses)
  - M_t = h[UPD]  (continuous)  or  E[argmax(h[UPD] E^T)] (discrete, straight-
    through Gumbel-softmax in training, hard argmax at eval)

Architecture (blocks, width, heads, depth, dropout) is identical to the
full-history model in full_history_model.py: d_model=64, 4 heads, 2 layers.
No absolute time index is given, so the model can be rolled out for any horizon.
"""

import argparse
import json
import logging
import math
import os
import sys
import time
from dataclasses import dataclass, asdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import paths  # noqa: E402
from full_history_model import TransformerBlock, ModelConfig, MWTokenizer  # noqa: E402

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
logger = logging.getLogger(__name__)

ROUND_LEN = 20
SEP_POS = 9
UPD_POS = 19


@dataclass
class RecurrentConfig:
    n_experts: int = 4
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2
    dropout: float = 0.1
    context_mode: str = 'continuous'   # 'continuous' | 'discrete'
    gumbel_tau: float = 1.0
    residual_latent: bool = False      # continuous only: M_t = M_{t-1} + W h[UPD]


class RecurrentMWTransformer(nn.Module):
    def __init__(self, cfg: RecurrentConfig):
        super().__init__()
        self.cfg = cfg
        self.tokenizer = MWTokenizer(cfg.n_experts)
        self.UPD_TOKEN = self.tokenizer.vocab_size          # one extra token
        self.vocab_size = self.tokenizer.vocab_size + 1

        block_cfg = ModelConfig(d_model=cfg.d_model, n_heads=cfg.n_heads,
                                n_layers=cfg.n_layers, n_experts=cfg.n_experts,
                                vocab_size=self.vocab_size, dropout=cfg.dropout)
        self.tok_emb = nn.Embedding(self.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(ROUND_LEN, cfg.d_model)
        self.m0 = nn.Parameter(torch.zeros(cfg.d_model))
        self.blocks = nn.ModuleList([TransformerBlock(block_cfg) for _ in range(cfg.n_layers)])
        self.ln_final = nn.LayerNorm(cfg.d_model)
        self.prediction_head = nn.Linear(cfg.d_model, 1)
        self.dropout = nn.Dropout(cfg.dropout)
        if cfg.residual_latent:
            assert cfg.context_mode == 'continuous', 'residual latent is continuous-only'
            # learned increment written onto the carried latent; retaining the
            # state is then the default (the MW construction adds -eta*l to
            # log-weights stored in the latent)
            self.latent_delta = nn.Linear(cfg.d_model, cfg.d_model)
        self.register_buffer('mask', torch.tril(torch.ones(ROUND_LEN, ROUND_LEN))[None, None])
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0.0, 0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, 0.0, 0.02)

    def initial_latent(self, batch_size):
        return self.m0.unsqueeze(0).expand(batch_size, -1)

    def contextualize(self, h_upd, tau=None):
        """Map the UPD hidden state to the next latent (identity or token snap)."""
        if self.cfg.context_mode != 'discrete':
            return h_upd
        tau = max(float(tau if tau is not None else self.cfg.gumbel_tau), 1e-6)
        logits = h_upd @ self.tok_emb.weight.t()
        if self.training:
            onehot = F.gumbel_softmax(logits, tau=tau, hard=True, dim=-1)
        else:
            idx = logits.argmax(dim=-1)
            hard = F.one_hot(idx, logits.shape[-1]).to(logits.dtype)
            soft = F.softmax(logits / tau, dim=-1)
            onehot = hard + soft - soft.detach()
        return onehot @ self.tok_emb.weight

    def latent_token_ids(self, h_upd):
        """Discrete mode: the vocabulary id the latent was snapped to."""
        return (h_upd @ self.tok_emb.weight.t()).argmax(dim=-1)

    def step(self, M, round_ids, tau=None, return_attention=False):
        """One round. M: [B, d]; round_ids: [B, 19] token ids for positions 1..19.

        Returns (pred_logit [B], M_next [B, d], h_upd [B, d], attn list).
        """
        x = torch.cat([M.unsqueeze(1), self.tok_emb(round_ids)], dim=1)
        x = self.dropout(x + self.pos_emb.weight.unsqueeze(0))
        attn = []
        for blk in self.blocks:
            x, a = blk(x, self.mask)
            if return_attention:
                attn.append(a)
        h = self.ln_final(x)
        pred_logit = self.prediction_head(h[:, SEP_POS]).squeeze(-1)
        h_upd = h[:, UPD_POS]
        if self.cfg.residual_latent:
            return pred_logit, M + self.latent_delta(h_upd), h_upd, attn
        return pred_logit, self.contextualize(h_upd, tau), h_upd, attn

    def rollout(self, ids, tau=None, M0=None, return_latents=False):
        """ids: [B, T, 19]. Returns pred logits [B, T] (and latents [B, T, d])."""
        B, T, _ = ids.shape
        M = self.initial_latent(B) if M0 is None else M0
        logits, latents = [], []
        for t in range(T):
            lg, M, _, _ = self.step(M, ids[:, t], tau)
            logits.append(lg)
            if return_latents:
                latents.append(M)
        logits = torch.stack(logits, 1)
        if return_latents:
            return logits, torch.stack(latents, 1)
        return logits


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def generate_single_sequence(n_experts, n_steps, rng, q=None):
    """Same distribution as generate_dataset.generate_single_sequence
    (fixed expert qualities ~ U[0.3, 0.9], uniform random binary labels).
    Pass q to fix the expert qualities (e.g. for the LLM-section regimes)."""
    q = rng.uniform(0.3, 0.9, size=n_experts) if q is None else np.asarray(q, dtype=float)
    labels = rng.integers(0, 2, size=n_steps)
    correct = rng.random((n_steps, n_experts)) < q[None, :]
    preds = np.where(correct, labels[:, None], 1 - labels[:, None])
    losses = (~correct).astype(float)
    return {'expert_predictions': preds.tolist(), 'losses': losses.tolist(),
            'true_labels': labels.tolist(), 'n_steps': n_steps, 'qualities': q.tolist()}


def encode_rounds(seq, tok: MWTokenizer, upd_token):
    """Encode a sequence into [T, 19] token ids (positions 1..19 of each round)."""
    rows = []
    for t in range(len(seq['true_labels'])):
        r = []
        for e, p in enumerate(seq['expert_predictions'][t]):
            r += [tok.EXPERT_TOKENS[e], tok.PRED_1_TOKEN if p == 1 else tok.PRED_0_TOKEN]
        r.append(tok.SEP_TOKEN)
        r.append(tok.PRED_1_TOKEN if seq['true_labels'][t] == 1 else tok.PRED_0_TOKEN)
        for e, l in enumerate(seq['losses'][t]):
            r += [tok.EXPERT_TOKENS[e], tok.discretize_loss(l)]
        r.append(upd_token)
        rows.append(r)
    return np.array(rows, dtype=np.int64)


def make_split(n, T, n_experts, rng, tok, upd):
    seqs = [generate_single_sequence(n_experts, T, rng) for _ in range(n)]
    ids = np.stack([encode_rounds(s, tok, upd) for s in seqs])
    y = np.array([s['true_labels'] for s in seqs], dtype=np.float32)
    return torch.from_numpy(ids), torch.from_numpy(y)


def make_split_fast(n, T, n_experts, rng, tok, upd):
    """Vectorised make_split (same distribution and token layout), used to
    draw a fresh training set every epoch with --fresh_data."""
    q = rng.uniform(0.3, 0.9, size=(n, 1, n_experts))
    labels = rng.integers(0, 2, size=(n, T))
    correct = rng.random((n, T, n_experts)) < q
    preds = np.where(correct, labels[..., None], 1 - labels[..., None])
    pred_tok = np.where(preds == 1, tok.PRED_1_TOKEN, tok.PRED_0_TOKEN)
    loss_tok = np.where(correct, tok.discretize_loss(0.0), tok.discretize_loss(1.0))
    exp_tok = np.broadcast_to(np.array(tok.EXPERT_TOKENS), (n, T, n_experts))
    ids = np.concatenate([
        np.stack([exp_tok, pred_tok], -1).reshape(n, T, 2 * n_experts),
        np.full((n, T, 1), tok.SEP_TOKEN),
        np.where(labels == 1, tok.PRED_1_TOKEN, tok.PRED_0_TOKEN)[..., None],
        np.stack([exp_tok, loss_tok], -1).reshape(n, T, 2 * n_experts),
        np.full((n, T, 1), upd),
    ], -1).astype(np.int64)
    return torch.from_numpy(ids), torch.from_numpy(labels.astype(np.float32))


# The theorem's latent-expert data model (Appendix: population log-loss targets MWU).
# Each round every expert gives a probability forecast p_i = P(y = 1); one hidden
# expert I* per sequence generates the label, y ~ Bernoulli(p_{I*}). The Bayes
# predictor is then exactly MWU with eta = 1 on log-loss:
#   w_i ∝ prod_s p_i(y_s),  P(y_t = 1 | history) = sum_i w_i p_{i,t}.
# Forecasts lie on the tokenizer's 100-bin value grid (WEIGHT_TOKENS), so the model
# sees them exactly; per-expert log-loss -log p_i(y) goes in the loss slots
# (LOSS_TOKENS, scaled by LATENT_MAX_LOSS). The forecast spread sigma is drawn per
# sequence, so the model must follow MWU across learning speeds; it does not change
# the Bayes predictor.
LATENT_SIGMAS = (0.1, 0.2, 0.3)
LATENT_MAX_LOSS = float(-np.log(5 / 99))       # forecasts are clipped to [5/99, 94/99]


def sample_latent_expert(n, T, n_experts, rng):
    """Return forecasts p [n, T, E] (on the k/99 grid), labels y [n, T], true expert [n]."""
    sig = rng.choice(LATENT_SIGMAS, size=(n, 1, 1))
    base = rng.uniform(0.15, 0.85, size=(n, T, 1))
    k = np.clip(np.rint((base + sig * rng.standard_normal((n, T, n_experts))) * 99), 5, 94)
    p = k / 99.0
    true = rng.integers(n_experts, size=n)
    y = (rng.random((n, T)) < p[np.arange(n), :, true]).astype(np.int64)
    return p, y, true


def encode_latent_expert(p, y, tok, upd):
    n, T, E = p.shape
    k = np.rint(p * 99).astype(np.int64)
    prob_tok = np.asarray(tok.WEIGHT_TOKENS)[k]
    loss = -np.log(np.where(y[..., None] == 1, p, 1 - p))
    loss_bin = np.clip(loss / LATENT_MAX_LOSS * 99, 0, 99).astype(np.int64)
    loss_tok = np.asarray(tok.LOSS_TOKENS)[loss_bin]
    exp_tok = np.broadcast_to(np.array(tok.EXPERT_TOKENS), (n, T, E))
    ids = np.concatenate([
        np.stack([exp_tok, prob_tok], -1).reshape(n, T, 2 * E),
        np.full((n, T, 1), tok.SEP_TOKEN),
        np.where(y == 1, tok.PRED_1_TOKEN, tok.PRED_0_TOKEN)[..., None],
        np.stack([exp_tok, loss_tok], -1).reshape(n, T, 2 * E),
        np.full((n, T, 1), upd),
    ], -1).astype(np.int64)
    return torch.from_numpy(ids), torch.from_numpy(y.astype(np.float32))


def make_split_latent_expert(n, T, n_experts, rng, tok, upd):
    p, y, _ = sample_latent_expert(n, T, n_experts, rng)
    return encode_latent_expert(p, y, tok, upd)


# Binary version of the latent-expert model, in the paper's own format (0/1 predictions
# and 0/1 losses, as in the independent-experts data). Experts are noisy copies of a
# common bit (so they agree often and are hard to tell apart); one hidden expert I* per
# sequence is right up to label noise: y_t = p_{I*,t} flipped w.p. NOISY_EPS. The Bayes
# predictor is then weighted majority with eta = log((1-eps)/eps) on 0/1 losses (the
# construction of Theorem 3.1 with gamma = (1-eps)/eps), i.e. exponential weights with
# eta = 1 on the log-loss of the experts' noisy forecasts (1-eps if p=1 else eps).
NOISY_EPS = 0.2
NOISY_DELTA = (0.2, 0.5)


def sample_noisy_expert(n, T, n_experts, rng):
    """Return predictions p [n, T, E] in {0,1}, labels y [n, T], true expert [n]."""
    b = rng.integers(0, 2, size=(n, T, 1))
    delta = rng.uniform(*NOISY_DELTA, size=(n, 1, n_experts))
    p = np.where(rng.random((n, T, n_experts)) < delta, 1 - b, b)
    true = rng.integers(n_experts, size=n)
    flip = rng.random((n, T)) < NOISY_EPS
    pt = p[np.arange(n), :, true]
    y = np.where(flip, 1 - pt, pt).astype(np.int64)
    return p, y, true


def encode_binary(p, y, tok, upd):
    """Token layout of make_split_fast for given 0/1 predictions and labels."""
    n, T, E = p.shape
    correct = p == y[..., None]
    pred_tok = np.where(p == 1, tok.PRED_1_TOKEN, tok.PRED_0_TOKEN)
    loss_tok = np.where(correct, tok.discretize_loss(0.0), tok.discretize_loss(1.0))
    exp_tok = np.broadcast_to(np.array(tok.EXPERT_TOKENS), (n, T, E))
    ids = np.concatenate([
        np.stack([exp_tok, pred_tok], -1).reshape(n, T, 2 * E),
        np.full((n, T, 1), tok.SEP_TOKEN),
        np.where(y == 1, tok.PRED_1_TOKEN, tok.PRED_0_TOKEN)[..., None],
        np.stack([exp_tok, loss_tok], -1).reshape(n, T, 2 * E),
        np.full((n, T, 1), upd),
    ], -1).astype(np.int64)
    return torch.from_numpy(ids), torch.from_numpy(y.astype(np.float32))


def make_data(data_model, fresh, n, T, n_experts, rng, tok, upd):
    if data_model == 'noisy_expert':
        p, y, _ = sample_noisy_expert(n, T, n_experts, rng)
        return encode_binary(p, y, tok, upd)
    if data_model == 'latent_expert':
        return make_split_latent_expert(n, T, n_experts, rng, tok, upd)
    return (make_split_fast if fresh else make_split)(n, T, n_experts, rng, tok, upd)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def evaluate_loss(model, ids, y, T, device, bs=500):
    model.eval()
    tot, n, correct = 0.0, 0, 0
    with torch.no_grad():
        for i in range(0, ids.shape[0], bs):
            b_ids = ids[i:i + bs, :T].to(device)
            b_y = y[i:i + bs, :T].to(device)
            lg = model.rollout(b_ids)
            tot += F.binary_cross_entropy_with_logits(lg, b_y, reduction='sum').item()
            correct += ((lg > 0).float() == b_y).sum().item()
            n += b_y.numel()
    return tot / n, correct / n


def main():
    global LATENT_SIGMAS
    ap = argparse.ArgumentParser()
    ap.add_argument('--context_mode', choices=['continuous', 'discrete'], default='continuous')
    ap.add_argument('--n_train', type=int, default=3000)
    ap.add_argument('--n_val', type=int, default=500)
    ap.add_argument('--max_T', type=int, default=100)
    ap.add_argument('--stage_lengths', type=int, nargs='+',
                    default=[5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 65, 80, 95])
    ap.add_argument('--max_epochs_per_stage', type=int, default=30)
    ap.add_argument('--patience', type=int, default=6)
    ap.add_argument('--batch_size', type=int, default=32)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--weight_decay', type=float, default=1e-2)
    ap.add_argument('--tau_start', type=float, default=2.0)
    ap.add_argument('--tau_end', type=float, default=0.5)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--data_model', choices=['independent', 'latent_expert', 'noisy_expert'],
                    default='independent',
                    help="independent: each expert correct w.p. its quality (the paper's data); "
                         "latent_expert: the theorem's model, where Bayes = MWU with eta = 1")
    ap.add_argument('--latent_sigmas', type=float, nargs='+', default=list(LATENT_SIGMAS),
                    help='latent_expert data: per-sequence forecast spreads (larger = the true '
                         'expert is identifiable sooner, so memory is worth more)')
    ap.add_argument('--residual_latent', action='store_true',
                    help='continuous only: M_t = M_{t-1} + W h[UPD] instead of M_t = h[UPD]')
    ap.add_argument('--fresh_data', action='store_true',
                    help='draw a new n_train training set every epoch (unlimited data)')
    ap.add_argument('--save_dir', type=str, default=None,
                    help='default: checkpoints/<model>_seed<seed>, with <model> one of paths.MODELS')
    ap.add_argument('--keep_stage_checkpoints', action='store_true',
                    help='also save the best model of every curriculum stage (stage_<k>.pt)')
    ap.add_argument('--time_budget_min', type=float, default=150,
                    help='stop starting new stages after this many minutes')
    args = ap.parse_args()
    LATENT_SIGMAS = tuple(args.latent_sigmas)
    if args.save_dir is None:
        if args.data_model != 'independent':
            name = 'theorem_matched'
        elif args.context_mode == 'discrete':
            name = 'discrete'
        else:
            name = 'continuous_residual' if args.residual_latent else 'continuous_overwrite'
        args.save_dir = str(paths.CHECKPOINTS / f'{name}_seed{args.seed}')

    os.makedirs(args.save_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'device={device} args={vars(args)}')

    cfg = RecurrentConfig(context_mode=args.context_mode, residual_latent=args.residual_latent)
    model = RecurrentMWTransformer(cfg).to(device)
    logger.info(f'params={sum(p.numel() for p in model.parameters()):,}')
    tok = model.tokenizer
    tr_ids, tr_y = make_data(args.data_model, False, args.n_train, args.max_T, cfg.n_experts,
                             rng, tok, model.UPD_TOKEN)
    va_ids, va_y = make_data(args.data_model, False, args.n_val, args.max_T, cfg.n_experts,
                             rng, tok, model.UPD_TOKEN)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
                            betas=(0.9, 0.95))
    n_stages = len(args.stage_lengths)
    history = []
    t0 = time.time()
    best_path = os.path.join(args.save_dir, 'final.pt')

    for si, T in enumerate(args.stage_lengths):
        if (time.time() - t0) / 60 > args.time_budget_min:
            logger.info(f'time budget reached before stage {si + 1} (T={T}); stopping')
            break
        tau = args.tau_start * (args.tau_end / args.tau_start) ** (si / max(n_stages - 1, 1))
        steps_per_epoch = math.ceil(args.n_train / args.batch_size)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=args.max_epochs_per_stage * steps_per_epoch, eta_min=args.lr / 10)
        best_val, bad, best_state = float('inf'), 0, None
        for ep in range(args.max_epochs_per_stage):
            if args.fresh_data:
                tr_ids, tr_y = make_data(args.data_model, True, args.n_train, T, cfg.n_experts,
                                         rng, tok, model.UPD_TOKEN)
            model.train()
            perm = torch.randperm(args.n_train)
            ep_loss, nb = 0.0, 0
            for i in range(0, args.n_train, args.batch_size):
                idx = perm[i:i + args.batch_size]
                b_ids = tr_ids[idx, :T].to(device)
                b_y = tr_y[idx, :T].to(device)
                lg = model.rollout(b_ids, tau=tau)
                loss = F.binary_cross_entropy_with_logits(lg, b_y)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()
                ep_loss += loss.item()
                nb += 1
            val_loss, val_acc = evaluate_loss(model, va_ids, va_y, T, device)
            history.append({'stage': si + 1, 'T': T, 'epoch': ep, 'tau': tau,
                            'train_loss': ep_loss / nb, 'val_loss': val_loss, 'val_acc': val_acc,
                            'elapsed_min': (time.time() - t0) / 60})
            logger.info(f'stage {si + 1} T={T} ep {ep} tau={tau:.2f} train={ep_loss / nb:.4f} '
                        f'val={val_loss:.4f} acc={val_acc:.4f} [{(time.time() - t0) / 60:.1f} min]')
            if val_loss < best_val - 1e-4:
                best_val, bad = val_loss, 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
                if bad >= args.patience:
                    break
        model.load_state_dict(best_state)
        ckpt = {'model_state_dict': model.state_dict(), 'config': asdict(cfg),
                'stage': si + 1, 'T': T, 'history': history, 'args': vars(args)}
        if args.keep_stage_checkpoints:
            torch.save(ckpt, os.path.join(args.save_dir, f'stage_{si + 1}.pt'))
        torch.save(ckpt, best_path)
        with open(os.path.join(args.save_dir, 'history.json'), 'w') as f:
            json.dump(history, f, indent=1)

    logger.info(f'done in {(time.time() - t0) / 60:.1f} min; saved {best_path}')


if __name__ == '__main__':
    main()
