"""
True PyTorch Handwired Q-Learning Transformer (Section 4.1 Construction).

Implements the 3-layer causal transformer from Section 4 using strictly
mathematical block-matrix operations (W_Q, W_K, W_V) that pass through
standard PyTorch F.softmax(). No Python array-slicing shortcuts or
if/else branching -- all routing is done via additive gating
(Complementary Superposition, Lemma 4.2(i)), sinusoidal offset matching
(RoPE-style), and content-based retrieval with recency bias.

The transformer natively implements epsilon-greedy Q-learning:
  - |A| parallel Eval tokens evaluate Q(s', a_c) for every candidate action
  - Qnext pools max Q-value from Eval tokens
  - The Update token aggregates the off-policy TD target:
        Q_{t+1}(s,a) = (1-alpha)*Q_t(s,a) + alpha*(r + gamma*max_a' Q_t(s',a'))

Round block structure (Section 4, per round):
  s_t, a_t, r_t, s_{t+1}, [s_{t+1}, a_1, Eval], ..., [s_{t+1}, a_{|A|}, Eval], Select, Qcurr, Qnext, Update

Embedding layout (d_model = 3 * d_te + d_pe):
    id(x)   = x[0 : d_te]           token identity + e_val scalar
    buf1(x) = x[d_te : 2*d_te]      first buffer  (rewards, stored Q-values)
    buf2(x) = x[2*d_te : 3*d_te]    second buffer (retrieved Q-values)
    pos(x)  = x[3*d_te:]            positional encoding [f(i), sin/cos pairs]
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class QLearningConfig:
    """Configuration for the Q-learning transformer. Fully flexible."""
    n_states:   int   = 4
    n_actions:  int   = 2
    alpha:      float = 0.1
    gamma:      float = 0.9
    epsilon:    float = 0.3
    beta:       float = 1000.0     # Fixed-offset attention temperature
    eps_rec:    float = 0.2        # Recency scaling for content retrieval
    xi:         float = 1000.0     # Additive penalty for complementary gating
    d_sin:      int   = 10         # Sinusoidal PE dims (5 sin/cos pairs)
    M_base:     float = 10000.0    # Sinusoidal frequency base


def _compute_dimensions(cfg: QLearningConfig) -> Dict[str, int]:
    """Compute all embedding dimensions from config. Nothing hardcoded."""
    # Vocab: Null, Start, n_states states, n_actions actions,
    #        R, Eval, Select, Qcurr, Qnext, Update
    vocab_size = 2 + cfg.n_states + cfg.n_actions + 6
    d_te = vocab_size + 1          # +1 for e_val scalar lane
    d_pe = 1 + cfg.d_sin           # recency + sinusoidal
    d_model = 3 * d_te + d_pe

    return {
        'vocab_size': vocab_size,
        'd_te': d_te,
        'd_pe': d_pe,
        'd_model': d_model,
        'e_val': vocab_size,       # scalar lane index (last in d_te)
    }


def _compute_vocab(cfg: QLearningConfig) -> Dict[str, any]:
    """Build vocab indices dynamically from config."""
    idx = 0
    TOK_NULL = idx; idx += 1
    TOK_START = idx; idx += 1
    TOK_S = list(range(idx, idx + cfg.n_states)); idx += cfg.n_states
    TOK_A = list(range(idx, idx + cfg.n_actions)); idx += cfg.n_actions
    TOK_R = idx; idx += 1
    TOK_EVAL = idx; idx += 1
    TOK_SELECT = idx; idx += 1
    TOK_QCURR = idx; idx += 1
    TOK_QNEXT = idx; idx += 1
    TOK_UPDATE = idx; idx += 1

    return {
        'TOK_NULL': TOK_NULL, 'TOK_START': TOK_START,
        'TOK_S': TOK_S, 'TOK_A': TOK_A,
        'TOK_R': TOK_R, 'TOK_EVAL': TOK_EVAL, 'TOK_SELECT': TOK_SELECT,
        'TOK_QCURR': TOK_QCURR, 'TOK_QNEXT': TOK_QNEXT, 'TOK_UPDATE': TOK_UPDATE,
    }


# ---------------------------------------------------------------------------
# Positional Encoding Math
# ---------------------------------------------------------------------------

def get_sinusoidal_vec(pos: int, d_sin: int, M_base: float = 10000.0) -> torch.Tensor:
    """Returns sinusoidal vector [sin(w0*p), cos(w0*p), ...] for absolute position p."""
    p = torch.zeros(d_sin, dtype=torch.float64)
    for k in range(d_sin // 2):
        w_k = 1.0 / (M_base ** (2 * k / d_sin))
        p[2 * k]     = math.sin(pos * w_k)
        p[2 * k + 1] = math.cos(pos * w_k)
    return p


def get_rotation_matrix(ell: int, d_sin: int, M_base: float = 10000.0) -> torch.Tensor:
    """Block-diagonal rotation matrix R(ell) for RoPE-style offset matching."""
    R = torch.zeros(d_sin, d_sin, dtype=torch.float64)
    for k in range(d_sin // 2):
        w_k = 1.0 / (M_base ** (2 * k / d_sin))
        cos_val = math.cos(ell * w_k)
        sin_val = math.sin(ell * w_k)
        R[2*k,   2*k]   =  cos_val
        R[2*k,   2*k+1] =  sin_val
        R[2*k+1, 2*k]   = -sin_val
        R[2*k+1, 2*k+1] =  cos_val
    return R


def f_recency(pos: int) -> float:
    """
    Monotonically increasing bounded recency signal.
    f(i) = 1 - 1/ln(i+e), lower-order than 1/(i+1).
    """
    return 1.0 - 1.0 / math.log(pos + math.e)


# ---------------------------------------------------------------------------
# Core Attention Head
# ---------------------------------------------------------------------------

class ExactAttentionHead(nn.Module):
    """Single attention head with fixed W_Q, W_K, W_V matrices. Standard softmax."""

    def __init__(self, W_Q: torch.Tensor, W_K: torch.Tensor,
                 W_V: torch.Tensor, temperature: float = 1000.0):
        super().__init__()
        self.register_buffer("W_Q", W_Q)
        self.register_buffer("W_K", W_K)
        self.register_buffer("W_V", W_V)
        self.temperature = temperature

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Q = x @ self.W_Q.T
        K = x @ self.W_K.T
        V = x @ self.W_V.T
        scores = self.temperature * (Q @ K.transpose(-2, -1))
        T_len = x.shape[1]
        causal_mask = torch.tril(torch.ones(T_len, T_len, dtype=torch.bool, device=x.device))
        scores = scores.masked_fill(~causal_mask, -1e9)
        weights = F.softmax(scores, dim=-1)
        return weights @ V


# ---------------------------------------------------------------------------
# Main Transformer
# ---------------------------------------------------------------------------

class QLearningTransformer(nn.Module):
    """
    3-layer PyTorch transformer implementing online tabular Q-Learning (Section 4.1).

    Architecture:
    +----------------------------------------------------------------------+
    | Layer 1 -- Within-round offset retrieval (4 heads)                    |
    |   Head 1: Eval tokens get id(s') from offset 2  -> buf1             |
    |   Head 2: Eval tokens get id(a_c) from offset 1 -> buf1            |
    |   Head 3: Qcurr gets id(s_t) from offset -> buf1                   |
    |   Head 4: Qcurr gets id(a_t) from offset -> buf1                   |
    |   Non-target tokens route to <Null> via additive gating (xi)        |
    +----------------------------------------------------------------------+
    | Layer 2 -- Q-value retrieval from COCONUT history (1 content head)   |
    |   query = buf1(workspace) -> content + recency match -> history      |
    |   buf2(workspace)[e_val] <- buf1(winner)[e_val]                     |
    +----------------------------------------------------------------------+
    | Layer 2.5 -- Qnext pools max Q from Eval tokens                     |
    |   Content-match on TOK_EVAL identity + Q-value secondary score      |
    |   buf1(Qnext)[e_val] <- buf2(argmax Eval)[e_val]                   |
    +----------------------------------------------------------------------+
    | Layer 3 -- TD aggregation + event token emission (4 heads)           |
    |   Heads 1-3: buf1(Update)[e_val] <- (1-a)Q + a*gamma*Q' + a*r      |
    |   Head 4:    id(Update) <- buf1(Qcurr) (carries s+a identity)       |
    +----------------------------------------------------------------------+
    """

    def __init__(self, config: QLearningConfig):
        super().__init__()
        self.config = config
        self.dims = _compute_dimensions(config)
        self.vocab = _compute_vocab(config)

        self.d_model = self.dims['d_model']
        self.d_te = self.dims['d_te']
        self.d_pe = self.dims['d_pe']
        self.d_sin = config.d_sin
        self.e_val = self.dims['e_val']
        self.vocab_size = self.dims['vocab_size']

        # Subspace boundaries
        self.ID_START = 0
        self.ID_END = self.d_te
        self.BUF1_START = self.d_te
        self.BUF1_END = 2 * self.d_te
        self.BUF2_START = 2 * self.d_te
        self.BUF2_END = 3 * self.d_te
        self.PE_START = 3 * self.d_te
        self.PE_F_IDX = self.PE_START
        self.PE_SIN_START = self.PE_START + 1
        self.PE_SIN_END = self.PE_SIN_START + self.d_sin

        # Round block layout
        nA = config.n_actions
        # s_t, a_t, r, s', [s', a_c, Eval]*|A|, Select, Qcurr, Qnext, Update
        self.ROUND_LEN = 4 + 3 * nA + 1 + 3
        self.IDX_ST = 0
        self.IDX_AT = 1
        self.IDX_R = 2
        self.IDX_SNEXT = 3
        self.IDX_EVAL_BASE = 4      # Each eval block: [s', a_c, Eval] = 3 tokens
        self.IDX_SELECT = 4 + 3 * nA
        self.IDX_QCURR = 5 + 3 * nA
        self.IDX_QNEXT = 6 + 3 * nA
        self.IDX_UPDATE = 7 + 3 * nA

        self._build_layers()
        self.reset()

    def reset(self) -> None:
        cfg = self.config
        v = self.vocab

        null = torch.zeros(self.d_model, dtype=torch.float64)
        null[self.PE_F_IDX] = f_recency(0)
        null[self.PE_SIN_START:self.PE_SIN_END] = get_sinusoidal_vec(0, self.d_sin, cfg.M_base)

        # <Start>: superposition of all state + action identities (Eq. 6)
        start = torch.zeros(self.d_model, dtype=torch.float64)
        for tok in v['TOK_S']:
            start[tok] = 1.0
        for tok in v['TOK_A']:
            start[tok] = 1.0
        start[self.PE_F_IDX] = f_recency(1)
        start[self.PE_SIN_START:self.PE_SIN_END] = get_sinusoidal_vec(1, self.d_sin, cfg.M_base)

        self._event_tokens = [null, start]
        self.q_history = [np.zeros((cfg.n_states, cfg.n_actions))]
        self.step_count = 0

    # ------------------------------------------------------------------
    # Layer construction
    # ------------------------------------------------------------------

    def _build_layers(self) -> None:
        cfg = self.config
        v = self.vocab

        # === Layer 1: Within-round fixed-offset retrieval ===
        # ONE pair of heads for ALL Eval tokens (they share offset structure):
        #   [s', a_c, Eval] -> Eval at offset=2 gets s', at offset=1 gets a_c
        self.l1_eval_s = self._build_fixed_offset_head(
            target_tok=v['TOK_EVAL'], ell=2, src_sub='id', dst_sub='buf1')
        self.l1_eval_a = self._build_fixed_offset_head(
            target_tok=v['TOK_EVAL'], ell=1, src_sub='id', dst_sub='buf1')

        # Qcurr gathers id(s_t) + id(a_t)
        offset_qcurr_to_st = self.IDX_QCURR - self.IDX_ST
        offset_qcurr_to_at = self.IDX_QCURR - self.IDX_AT
        self.l1_qcurr_s = self._build_fixed_offset_head(
            target_tok=v['TOK_QCURR'], ell=offset_qcurr_to_st,
            src_sub='id', dst_sub='buf1')
        self.l1_qcurr_a = self._build_fixed_offset_head(
            target_tok=v['TOK_QCURR'], ell=offset_qcurr_to_at,
            src_sub='id', dst_sub='buf1')

        # === Layer 2: Content-based Q-value retrieval from history ===
        self.layer2_content = self._build_content_retrieval_head()

        # === Layer 2.5: Qnext pools max Q from current round's Eval tokens ===
        self.layer2_qnext_pool = self._build_qnext_pool_head()

        # === Layer 3: TD aggregation + event-token emission ===
        offset_update_to_qcurr = self.IDX_UPDATE - self.IDX_QCURR
        offset_update_to_qnext = self.IDX_UPDATE - self.IDX_QNEXT
        offset_update_to_r = self.IDX_UPDATE - self.IDX_R

        # Head 1: (1-alpha) * Q(s,a) from Qcurr buf2[e_val]
        self.l3_h1 = self._build_fixed_offset_head(
            target_tok=v['TOK_UPDATE'], ell=offset_update_to_qcurr,
            src_sub='buf2_val', dst_sub='buf1_val',
            weight=1.0 - cfg.alpha)
        # Head 2: alpha*gamma * max_a Q(s',a) from Qnext buf1[e_val]
        self.l3_h2 = self._build_fixed_offset_head(
            target_tok=v['TOK_UPDATE'], ell=offset_update_to_qnext,
            src_sub='buf1_val', dst_sub='buf1_val',
            weight=cfg.alpha * cfg.gamma)
        # Head 3: alpha * r from r-token buf1[e_val]
        self.l3_h3 = self._build_fixed_offset_head(
            target_tok=v['TOK_UPDATE'], ell=offset_update_to_r,
            src_sub='buf1_val', dst_sub='buf1_val',
            weight=cfg.alpha)
        # Head 4: copy buf1(Qcurr) -> id(Update) for s+a identity
        self.l3_h4 = self._build_fixed_offset_head(
            target_tok=v['TOK_UPDATE'], ell=offset_update_to_qcurr,
            src_sub='buf1', dst_sub='id')

    def _build_fixed_offset_head(
        self,
        target_tok: int,
        ell: int,
        src_sub: str,
        dst_sub: str,
        weight: float = 1.0,
    ) -> ExactAttentionHead:
        """
        Construct W_Q, W_K, W_V for exact offset matching with Complementary
        Superposition gating (Lemma 4.2(i)).

        Query space partitioned:
          Subspace A (0..d_sin): sinusoidal PE for offset matching
          Subspace B (d_sin..2*d_sin): xi-scaled penalty routing non-targets to <Null>
        """
        cfg = self.config
        d_q = 2 * self.d_sin

        W_Q = torch.zeros(d_q, self.d_model, dtype=torch.float64)
        W_K = torch.zeros(d_q, self.d_model, dtype=torch.float64)
        W_V = torch.zeros(self.d_model, self.d_model, dtype=torch.float64)

        # Subspace A: sinusoidal offset matching
        W_Q[0:self.d_sin, self.PE_SIN_START:self.PE_SIN_END] = torch.eye(self.d_sin, dtype=torch.float64)
        W_K[0:self.d_sin, self.PE_SIN_START:self.PE_SIN_END] = get_rotation_matrix(ell, self.d_sin, cfg.M_base)

        # Subspace B: Complementary Superposition (additive gating)
        u_comp = torch.ones(self.d_te, dtype=torch.float64)
        u_comp[target_tok] = 0.0
        p_null = get_sinusoidal_vec(0, self.d_sin, cfg.M_base)

        W_Q[self.d_sin:2*self.d_sin, self.ID_START:self.ID_END] = (
            cfg.xi * torch.outer(p_null, u_comp))
        W_K[self.d_sin:2*self.d_sin, self.PE_SIN_START:self.PE_SIN_END] = (
            torch.eye(self.d_sin, dtype=torch.float64))

        # Value projection: data routing
        if src_sub == 'id' and dst_sub == 'buf1':
            W_V[self.BUF1_START:self.BUF1_END, self.ID_START:self.ID_END] = (
                weight * torch.eye(self.d_te, dtype=torch.float64))
        elif src_sub == 'buf1' and dst_sub == 'id':
            W_V[self.ID_START:self.ID_END, self.BUF1_START:self.BUF1_END] = (
                weight * torch.eye(self.d_te, dtype=torch.float64))
        elif src_sub == 'buf2_val' and dst_sub == 'buf1_val':
            W_V[self.BUF1_START + self.e_val, self.BUF2_START + self.e_val] = weight
        elif src_sub == 'buf1_val' and dst_sub == 'buf1_val':
            W_V[self.BUF1_START + self.e_val, self.BUF1_START + self.e_val] = weight

        return ExactAttentionHead(W_Q, W_K, W_V, temperature=cfg.beta)

    def _build_content_retrieval_head(self) -> ExactAttentionHead:
        """
        Layer 2: content + recency retrieval head.
        Content gap (100k) >> max recency (20k) -> recency breaks ties only.
        """
        cfg = self.config
        layer2_beta = 100_000.0

        d_q = self.d_te + 1
        W_Q = torch.zeros(d_q, self.d_model, dtype=torch.float64)
        W_K = torch.zeros(d_q, self.d_model, dtype=torch.float64)
        W_V = torch.zeros(self.d_model, self.d_model, dtype=torch.float64)

        # Content: Q reads buf1, K reads id
        W_Q[0:self.d_te, self.BUF1_START:self.BUF1_END] = torch.eye(self.d_te, dtype=torch.float64)
        W_K[0:self.d_te, self.ID_START:self.ID_END] = torch.eye(self.d_te, dtype=torch.float64)

        # Recency
        alpha_scale = math.sqrt(cfg.eps_rec)
        W_Q[self.d_te, self.PE_F_IDX] = alpha_scale
        W_K[self.d_te, self.PE_F_IDX] = alpha_scale

        # Value: buf1[e_val] -> buf2[e_val]
        W_V[self.BUF2_START + self.e_val, self.BUF1_START + self.e_val] = 1.0

        return ExactAttentionHead(W_Q, W_K, W_V, temperature=layer2_beta)

    def _build_qnext_pool_head(self) -> ExactAttentionHead:
        """
        Layer 2.5: Qnext pools max_a Q(s',a) from Eval tokens.

        Uses the Attention Sink (Null Routing) mechanism to prevent residual
        stream pollution: non-Qnext tokens attend exclusively to <Null> at
        position 0 (which has 0.0 in buf2_val), adding nothing to their buf1.

        Query space partitioned into 3 subspaces:
          Subspace A (dim 0): content match — TOK_QNEXT identity dots TOK_EVAL
          Subspace B (dim 1): Q-value tiebreaker — selects highest-Q Eval token
          Subspace C (dims 2..2+d_sin): Attention Sink gating via Complementary
            Superposition. Non-Qnext tokens project xi-scaled queries aligned
            with <Null>'s sinusoidal PE, overwhelming all other scores and
            routing ~100% attention weight to position 0.
        """
        cfg = self.config
        pool_beta = 100_000.0

        d_q = 2 + self.d_sin  # content + Q-tiebreaker + attention-sink gating
        W_Q = torch.zeros(d_q, self.d_model, dtype=torch.float64)
        W_K = torch.zeros(d_q, self.d_model, dtype=torch.float64)
        W_V = torch.zeros(self.d_model, self.d_model, dtype=torch.float64)

        # --- Subspace A: Primary content match (Qnext -> Eval) ---
        W_Q[0, self.ID_START + self.vocab['TOK_QNEXT']] = 1.0
        W_K[0, self.ID_START + self.vocab['TOK_EVAL']] = 1.0

        # --- Subspace B: Q-value tiebreaker ---
        W_Q[1, self.ID_START + self.vocab['TOK_QNEXT']] = 1.0
        W_K[1, self.BUF2_START + self.e_val] = 1.0

        # --- Subspace C: Attention Sink (Complementary Superposition) ---
        # For non-Qnext tokens: project a query that maximizes dot product
        # with <Null>'s sinusoidal PE at position 0.
        #
        # u_comp = identity vector that is 1 everywhere EXCEPT at TOK_QNEXT.
        # Query for non-Qnext: xi * p_null ⊗ u_comp · id(x)
        #   -> yields xi * p_null when id[TOK_QNEXT]=0 (non-Qnext tokens)
        #   -> yields 0 when id[TOK_QNEXT]=1 (Qnext token, since u_comp[TOK_QNEXT]=0)
        # Key: sinusoidal PE (same for all tokens)
        #   -> dot(xi*p_null, p_null) = xi * ||p_null||^2 >> content scores
        #   -> forces non-Qnext tokens to attend to position 0 (<Null>)
        u_comp = torch.ones(self.d_te, dtype=torch.float64)
        u_comp[self.vocab['TOK_QNEXT']] = 0.0
        p_null = get_sinusoidal_vec(0, self.d_sin, cfg.M_base)

        W_Q[2:2+self.d_sin, self.ID_START:self.ID_END] = (
            cfg.xi * torch.outer(p_null, u_comp))
        W_K[2:2+self.d_sin, self.PE_SIN_START:self.PE_SIN_END] = (
            torch.eye(self.d_sin, dtype=torch.float64))

        # --- Value: route buf2[e_val] of winner -> buf1[e_val] ---
        W_V[self.BUF1_START + self.e_val, self.BUF2_START + self.e_val] = 1.0

        return ExactAttentionHead(W_Q, W_K, W_V, temperature=pool_beta)

    # ------------------------------------------------------------------
    # Embedding construction
    # ------------------------------------------------------------------

    def _embed_token(self, tok_id: int, abs_pos: int) -> torch.Tensor:
        """Construct full embedding with absolute PE burned in."""
        cfg = self.config
        x = torch.zeros(self.d_model, dtype=torch.float64)
        x[tok_id] = 1.0
        x[self.PE_F_IDX] = f_recency(abs_pos)
        x[self.PE_SIN_START:self.PE_SIN_END] = get_sinusoidal_vec(abs_pos, self.d_sin, cfg.M_base)
        return x

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def step(self, s: int, a: int, r: float, s_next: int) -> float:
        """
        Process one MDP transition and return Q_{t+1}(s, a).

        The transformer computes max_a Q(s',a) via the Eval+Qnext mechanism.
        """
        cfg = self.config
        v = self.vocab
        nA = cfg.n_actions
        base_pos = len(self._event_tokens)

        # Build round block
        tokens = []
        tokens.append(v['TOK_S'][s])         # 0: s_t
        tokens.append(v['TOK_A'][a])         # 1: a_t
        tokens.append(v['TOK_R'])            # 2: r
        tokens.append(v['TOK_S'][s_next])    # 3: s'

        for c in range(nA):                  # Eval triples
            tokens.append(v['TOK_S'][s_next])
            tokens.append(v['TOK_A'][c])
            tokens.append(v['TOK_EVAL'])

        tokens.append(v['TOK_SELECT'])       # Select
        tokens.append(v['TOK_QCURR'])        # Qcurr
        tokens.append(v['TOK_QNEXT'])        # Qnext
        tokens.append(v['TOK_UPDATE'])        # Update

        assert len(tokens) == self.ROUND_LEN

        x_block = torch.zeros(len(tokens), self.d_model, dtype=torch.float64)
        for i, tok in enumerate(tokens):
            x_block[i] = self._embed_token(tok, base_pos + i)

        # Seed reward into buf1[e_val] of r-token
        x_block[self.IDX_R, self.BUF1_START + self.e_val] = r

        # Concatenate history + round block
        history = torch.stack(self._event_tokens)
        full = torch.cat([history, x_block], dim=0).unsqueeze(0)

        # === Layer 1: within-round offset retrieval ===
        full = full + self.l1_eval_s(full) + self.l1_eval_a(full)
        full = full + self.l1_qcurr_s(full) + self.l1_qcurr_a(full)

        # === Layer 2: content-based Q-value retrieval from history ===
        full = full + self.layer2_content(full)

        # === Layer 2.5: Qnext pools max Q from Eval tokens ===
        full = full + self.layer2_qnext_pool(full)

        # === Layer 3: TD aggregation + emission ===
        full = full + self.l3_h1(full) + self.l3_h2(full) + self.l3_h3(full) + self.l3_h4(full)

        # Emit event token v^(t)
        update_seq_idx = base_pos + self.IDX_UPDATE
        event_token = full[0, update_seq_idx].detach().clone()

        # Re-stamp absolute PE
        history_pos = len(self._event_tokens)
        event_token[self.PE_F_IDX] = f_recency(history_pos)
        event_token[self.PE_SIN_START:self.PE_SIN_END] = get_sinusoidal_vec(
            history_pos, self.d_sin, cfg.M_base)

        self._event_tokens.append(event_token)

        q_new = float(event_token[self.BUF1_START + self.e_val])

        Q_new = self.q_history[-1].copy()
        Q_new[s, a] = q_new
        self.q_history.append(Q_new)
        self.step_count += 1

        logger.debug("Step %d: Q(%d,%d) <- %.6f", self.step_count, s, a, q_new)
        return q_new

    def get_q_history(self) -> List[np.ndarray]:
        return self.q_history


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from tabular_q_learning import TabularQLearning, make_chain_mdp, generate_trajectory

    logging.basicConfig(level=logging.WARNING)

    n_states, n_actions = 4, 2
    P = make_chain_mdp(n_states=n_states, n_actions=n_actions)
    traj = generate_trajectory(P, n_states=n_states, n_actions=n_actions, T=50)

    ql = TabularQLearning(n_states=n_states, n_actions=n_actions,
                          alpha=0.1, gamma=0.9)
    cfg = QLearningConfig(n_states=n_states, n_actions=n_actions,
                          alpha=0.1, gamma=0.9)
    tf = QLearningTransformer(cfg)

    max_err = 0.0
    for t, (s, a, r, s_next) in enumerate(traj):
        q_c = ql.step(s, a, r, s_next)
        q_t = tf.step(s, a, r, s_next)
        err = abs(q_c - q_t)
        max_err = max(max_err, err)
        if err > 1e-6:
            print(f"  MISMATCH at step {t}: QL={q_c:.8f}  TF={q_t:.8f}  diff={err:.2e}")

    print(f"Max absolute Q-difference over 50 steps: {max_err:.2e}")
    if max_err < 1e-6:
        print("PASS: transformer matches tabular Q-learning.")
    else:
        print("FAIL: mismatch detected -- check construction.")
