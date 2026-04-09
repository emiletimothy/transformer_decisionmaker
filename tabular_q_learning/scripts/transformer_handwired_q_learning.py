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


def f_recency(pos: int, n_actions: int) -> float:
    """
    Monotonically increasing bounded recency signal.
    f(i) = 1 - 1/ln(i+e), lower-order than 1/(i+1).
    """
    return 1.0 - 1.0 / math.log((pos/n_actions) + math.e)


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

        # Round block layout (Section 4, two phases)
        # Phase I:  s_t, a_t, r, s', [s', a_c, Eval]*|A|, Select
        # Phase II: a_{t+1}, Qcurr, Qnext, Update
        nA = config.n_actions
        self.PHASE1_LEN = 4 + 3 * nA + 1          # up to and including Select
        self.ROUND_LEN  = self.PHASE1_LEN + 4      # + a_{t+1}, Qcurr, Qnext, Update
        self.IDX_ST = 0
        self.IDX_AT = 1
        self.IDX_R = 2
        self.IDX_SNEXT = 3
        self.IDX_EVAL_BASE = 4      # Each eval block: [s', a_c, Eval] = 3 tokens
        self.IDX_SELECT  = 4 + 3 * nA
        self.IDX_ANEXT   = 5 + 3 * nA   # a_{t+1} decoded from Select
        self.IDX_QCURR   = 6 + 3 * nA
        self.IDX_QNEXT   = 7 + 3 * nA
        self.IDX_UPDATE  = 8 + 3 * nA

        self._build_layers()
        self.reset()

    def reset(self) -> None:
        cfg = self.config
        v = self.vocab

        null = torch.zeros(self.d_model, dtype=torch.float64)
        null[self.PE_F_IDX] = f_recency(0, cfg.n_actions)
        null[self.PE_SIN_START:self.PE_SIN_END] = get_sinusoidal_vec(0, self.d_sin, cfg.M_base)

        # <Start>: superposition of all state + action identities (Eq. 6)
        start = torch.zeros(self.d_model, dtype=torch.float64)
        for tok in v['TOK_S']:
            start[tok] = 1.0
        for tok in v['TOK_A']:
            start[tok] = 1.0
        start[self.PE_F_IDX] = f_recency(1, cfg.n_actions)
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

        # ------------------------------------------------------------------
        # Decoding matrix W_O ∈ R^{|A| × d_model}  (Section 4, Phase I)
        # Logit for action c = ⟨id(TF output at Select), ũ_{a_c}⟩
        # i.e. W_O[c, v['TOK_A'][c]] = 1  in the id subspace.
        # ------------------------------------------------------------------
        W_O = torch.zeros(cfg.n_actions, self.d_model, dtype=torch.float64)
        for c in range(cfg.n_actions):
            W_O[c, self.ID_START + v['TOK_A'][c]] = 1.0
        self.register_buffer("W_O", W_O)

        # ------------------------------------------------------------------
        # Layer 1: Within-round fixed-offset retrieval
        # All heads fire simultaneously on the residual stream; their outputs
        # are summed (multi-head attention residual update).
        #
        # Pass-1 heads (active when sequence ends at ⟨Select⟩):
        #   l1_eval_s / l1_eval_a : Eval tokens gather s' and a_c identities
        # Pass-2 heads (active when sequence includes Qcurr/Qnext/Update):
        #   l1_qcurr_s / l1_qcurr_a : Qcurr gathers s_t and a_t identities
        # Non-target tokens route to ⟨Null⟩ via Complementary Superposition.
        # ------------------------------------------------------------------
        self.l1_eval_s = self._build_fixed_offset_head(
            target_tok=v['TOK_EVAL'], ell=2, src_sub='id', dst_sub='buf1')
        self.l1_eval_a = self._build_fixed_offset_head(
            target_tok=v['TOK_EVAL'], ell=1, src_sub='id', dst_sub='buf1')

        offset_qcurr_to_st = self.IDX_QCURR - self.IDX_ST
        offset_qcurr_to_at = self.IDX_QCURR - self.IDX_AT
        self.l1_qcurr_s = self._build_fixed_offset_head(
            target_tok=v['TOK_QCURR'], ell=offset_qcurr_to_st,
            src_sub='id', dst_sub='buf1')
        self.l1_qcurr_a = self._build_fixed_offset_head(
            target_tok=v['TOK_QCURR'], ell=offset_qcurr_to_at,
            src_sub='id', dst_sub='buf1')

        # ------------------------------------------------------------------
        # Layer 2: Content-based Q-value retrieval from COCONUT history
        # ------------------------------------------------------------------
        self.layer2_content = self._build_content_retrieval_head()

        # ------------------------------------------------------------------
        # Layer 2.5: Qnext pools max_a Q(s',a) from current Eval tokens
        # ------------------------------------------------------------------
        self.layer2_qnext_pool = self._build_qnext_pool_head()

        # ------------------------------------------------------------------
        # Layer 3, Pass-1 head: Select gathers best Eval Q-value
        # Fired on both passes; Attention Sink ensures it adds nothing
        # to Qcurr/Qnext/Update tokens (they route to ⟨Null⟩).
        # ------------------------------------------------------------------
        self.l3_select = self._build_select_head()

        # ------------------------------------------------------------------
        # Layer 3, Pass-2 heads: TD aggregation at ⟨Update⟩
        # Offsets from Update (with a_{t+1} between Select and Qcurr):
        #   Update → Qcurr : IDX_UPDATE - IDX_QCURR = 2
        #   Update → Qnext : IDX_UPDATE - IDX_QNEXT = 1
        #   Update → r     : IDX_UPDATE - IDX_R
        # ------------------------------------------------------------------
        offset_update_to_qcurr = self.IDX_UPDATE - self.IDX_QCURR
        offset_update_to_qnext = self.IDX_UPDATE - self.IDX_QNEXT
        offset_update_to_r     = self.IDX_UPDATE - self.IDX_R

        self.l3_h1 = self._build_fixed_offset_head(
            target_tok=v['TOK_UPDATE'], ell=offset_update_to_qcurr,
            src_sub='buf2_val', dst_sub='buf1_val',
            weight=1.0 - cfg.alpha)
        self.l3_h2 = self._build_fixed_offset_head(
            target_tok=v['TOK_UPDATE'], ell=offset_update_to_qnext,
            src_sub='buf1_val', dst_sub='buf1_val',
            weight=cfg.alpha * cfg.gamma)
        self.l3_h3 = self._build_fixed_offset_head(
            target_tok=v['TOK_UPDATE'], ell=offset_update_to_r,
            src_sub='buf1_val', dst_sub='buf1_val',
            weight=cfg.alpha)
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

    def _build_select_head(self) -> ExactAttentionHead:
        """
        Layer 3 / Pass-1 action-selection head at ⟨Select⟩.

        Select attends over the current round's |A| Eval tokens using:
          Subspace A (dim 0): content match — TOK_SELECT identity dots TOK_EVAL
          Subspace B (dim 1): Q-value tiebreaker — selects highest-Q Eval token
            (Q-value lives in buf2[e_val] after Layer 2, but here we use it
             as a secondary score via buf2 of Eval tokens post-retrieval)
          Subspace C (dims 2..2+d_sin): Attention Sink gating (Complementary
            Superposition) — non-Select tokens attend to ⟨Null⟩ at position 0.

        Value projection copies the action identity (id subspace) of the winning
        Eval token into buf1 of Select, so that the action identity u˜_{a*} is
        available for ε-greedy decoding.
        """
        cfg = self.config
        select_beta = 100_000.0

        d_q = 2 + self.d_sin
        W_Q = torch.zeros(d_q, self.d_model, dtype=torch.float64)
        W_K = torch.zeros(d_q, self.d_model, dtype=torch.float64)
        W_V = torch.zeros(self.d_model, self.d_model, dtype=torch.float64)

        # --- Subspace A: Primary content match (Select -> Eval) ---
        W_Q[0, self.ID_START + self.vocab['TOK_SELECT']] = 1.0
        W_K[0, self.ID_START + self.vocab['TOK_EVAL']] = 1.0

        # --- Subspace B: Q-value tiebreaker (higher buf2[e_val] wins) ---
        W_Q[1, self.ID_START + self.vocab['TOK_SELECT']] = 1.0
        W_K[1, self.BUF2_START + self.e_val] = 1.0

        # --- Subspace C: Attention Sink for non-Select tokens ---
        u_comp = torch.ones(self.d_te, dtype=torch.float64)
        u_comp[self.vocab['TOK_SELECT']] = 0.0
        p_null = get_sinusoidal_vec(0, self.d_sin, cfg.M_base)

        W_Q[2:2+self.d_sin, self.ID_START:self.ID_END] = (
            cfg.xi * torch.outer(p_null, u_comp))
        W_K[2:2+self.d_sin, self.PE_SIN_START:self.PE_SIN_END] = (
            torch.eye(self.d_sin, dtype=torch.float64))

        # --- Value: copy id(Eval winner) -> buf1(Select) ---
        # This deposits u˜_{a*} (action identity) into Select's buf1
        W_V[self.BUF1_START:self.BUF1_END, self.ID_START:self.ID_END] = (
            torch.eye(self.d_te, dtype=torch.float64))

        return ExactAttentionHead(W_Q, W_K, W_V, temperature=select_beta)

    # ------------------------------------------------------------------
    # Embedding construction
    # ------------------------------------------------------------------

    def _embed_token(self, tok_id: int, abs_pos: int) -> torch.Tensor:
        """Construct full embedding with absolute PE burned in."""
        cfg = self.config
        x = torch.zeros(self.d_model, dtype=torch.float64)
        x[tok_id] = 1.0
        x[self.PE_F_IDX] = f_recency(abs_pos, cfg.n_actions)
        x[self.PE_SIN_START:self.PE_SIN_END] = get_sinusoidal_vec(abs_pos, self.d_sin, cfg.M_base)
        return x

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def _layer1(self, seq: torch.Tensor) -> torch.Tensor:
        """Layer 1: all offset-retrieval heads summed (multi-head residual update)."""
        return (seq
                + self.l1_eval_s(seq)  + self.l1_eval_a(seq)
                + self.l1_qcurr_s(seq) + self.l1_qcurr_a(seq))

    def _layer2(self, seq: torch.Tensor) -> torch.Tensor:
        """Layer 2: content-based Q-value retrieval from COCONUT history."""
        return seq + self.layer2_content(seq)

    def _layer2_5(self, seq: torch.Tensor) -> torch.Tensor:
        """Layer 2.5: Qnext pools max_a Q(s',a) from Eval tokens."""
        return seq + self.layer2_qnext_pool(seq)

    def _layer3(self, seq: torch.Tensor) -> torch.Tensor:
        """Layer 3: Select head (Pass 1) + TD aggregation heads (Pass 2), summed."""
        return (seq
                + self.l3_select(seq)
                + self.l3_h1(seq) + self.l3_h2(seq)
                + self.l3_h3(seq) + self.l3_h4(seq))

    def _forward(self, seq: torch.Tensor) -> torch.Tensor:
        """Full 3-layer causal transformer forward pass (residual stream)."""
        seq = self._layer1(seq)
        seq = self._layer2(seq)
        seq = self._layer2_5(seq)
        seq = self._layer3(seq)
        return seq

    def step(self, s: int, a: int, r: float, s_next: int) -> tuple:
        """
        Process one MDP transition via two autoregressive transformer passes.
        Returns (Q_{t+1}(s, a), a_next).

        ── Pass 1 (action selection, Section 4 Phase I) ──────────────────
        Input sequence ends at ⟨Select⟩:
            [history | s_t, a_t, r, s', (s', a_c, Eval)×|A|, Select]

        All 3 transformer layers run on this sequence. The output at the
        ⟨Select⟩ position is projected through the decoding matrix W_O to
        obtain action logits. After scaling by
            c = ln(|A|*(1-ε)/ε + 1)
        softmax sampling recovers the exact ε-greedy policy, yielding ⟨a_{t+1}⟩.

        ── Pass 2 (Q-value update + COCONUT emission, Section 4 Phase II) ─
        The real ⟨a_{t+1}⟩ token is appended, then [Qcurr, Qnext, Update]:
            [history | s_t, a_t, r, s', (s',a_c,Eval)×|A|, Select,
             a_{t+1}, Qcurr, Qnext, Update]

        All 3 transformer layers run again on this full sequence (causal
        attention, so Pass 1 residual states are visible to Pass 2 tokens).
        The output at ⟨Update⟩ is appended to the COCONUT history as the
        continuous event token v^(t), encoding Q_{t+1}(s_t, a_t).
        """
        cfg = self.config
        v   = self.vocab
        nA  = cfg.n_actions
        base_pos = len(self._event_tokens)

        # ── Shared prefix: s_t, a_t, r, s', Eval-blocks ──────────────────
        phase1_toks = []
        phase1_toks.append(v['TOK_S'][s])         # IDX_ST
        phase1_toks.append(v['TOK_A'][a])         # IDX_AT
        phase1_toks.append(v['TOK_R'])            # IDX_R
        phase1_toks.append(v['TOK_S'][s_next])    # IDX_SNEXT
        for c in range(nA):
            phase1_toks.append(v['TOK_S'][s_next])
            phase1_toks.append(v['TOK_A'][c])
            phase1_toks.append(v['TOK_EVAL'])
        phase1_toks.append(v['TOK_SELECT'])       # IDX_SELECT

        assert len(phase1_toks) == self.PHASE1_LEN

        x_p1 = torch.zeros(self.PHASE1_LEN, self.d_model, dtype=torch.float64)
        for i, tok in enumerate(phase1_toks):
            x_p1[i] = self._embed_token(tok, base_pos + i)
        x_p1[self.IDX_R, self.BUF1_START + self.e_val] = r

        history = torch.stack(self._event_tokens)   # shape [H, d_model]

        # ══════════════════════════════════════════════════════════════════
        # PASS 1: sequence = [history | phase1_block]  (ends at ⟨Select⟩)
        # ══════════════════════════════════════════════════════════════════
        seq1 = torch.cat([history, x_p1], dim=0).unsqueeze(0)  # [1, H+PHASE1_LEN, d]
        seq1 = self._forward(seq1)

        # Decode ⟨a_{t+1}⟩ from ⟨Select⟩ output via W_O (Section 4, Phase I)
        #   logit_c = c_scale * (W_O[c] · h_Select)
        #           = c_scale * ⟨ũ_{a_c}, id(h_Select)⟩
        select_out  = seq1[0, base_pos + self.IDX_SELECT]        # h_Select
        raw_logits  = self.W_O @ select_out                      # [nA]
        c_scale     = math.log(nA * (1.0 - cfg.epsilon) / cfg.epsilon + 1.0)
        action_probs = F.softmax(c_scale * raw_logits, dim=0)
        a_next = int(torch.multinomial(action_probs.float(), num_samples=1).item())

        # ══════════════════════════════════════════════════════════════════
        # PASS 2: append real ⟨a_{t+1}⟩ + [Qcurr, Qnext, Update]
        # ══════════════════════════════════════════════════════════════════
        phase2_toks = [
            v['TOK_A'][a_next],   # IDX_ANEXT — real decoded action
            v['TOK_QCURR'],       # IDX_QCURR
            v['TOK_QNEXT'],       # IDX_QNEXT
            v['TOK_UPDATE'],      # IDX_UPDATE
        ]
        x_p2 = torch.zeros(len(phase2_toks), self.d_model, dtype=torch.float64)
        for i, tok in enumerate(phase2_toks):
            x_p2[i] = self._embed_token(tok, base_pos + self.PHASE1_LEN + i)

        # Full sequence: re-embed from scratch so causal attention sees the
        # correct, unmodified input embeddings at all positions.
        # (Pass 1 residual states are NOT reused — each pass is a fresh
        # forward of the same fixed-weight transformer, as in autoregressive
        # decoding where the full context is reprocessed each step.)
        x_full = torch.cat([x_p1, x_p2], dim=0)
        seq2   = torch.cat([history, x_full], dim=0).unsqueeze(0)
        seq2   = self._forward(seq2)

        # Emit continuous event token v^(t) via COCONUT
        update_out  = seq2[0, base_pos + self.IDX_UPDATE].detach().clone()

        # Re-stamp PE to the event token's permanent history position
        history_pos = len(self._event_tokens)
        update_out[self.PE_F_IDX] = f_recency(history_pos, cfg.n_actions)
        update_out[self.PE_SIN_START:self.PE_SIN_END] = get_sinusoidal_vec(
            history_pos, self.d_sin, cfg.M_base)

        self._event_tokens.append(update_out)

        q_new = float(update_out[self.BUF1_START + self.e_val])
        Q_new = self.q_history[-1].copy()
        Q_new[s, a] = q_new
        self.q_history.append(Q_new)
        self.step_count += 1

        logger.debug("Step %d: Q(%d,%d) <- %.6f  a_next=%d",
                     self.step_count, s, a, q_new, a_next)
        return q_new, a_next

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
        q_t, a_next = tf.step(s, a, r, s_next)
        err = abs(q_c - q_t)
        max_err = max(max_err, err)
        if err > 1e-6:
            print(f"  MISMATCH at step {t}: QL={q_c:.8f}  TF={q_t:.8f}  diff={err:.2e}")

    print(f"Max absolute Q-difference over 50 steps: {max_err:.2e}")
    if max_err < 1e-6:
        print("PASS: transformer matches tabular Q-learning.")
    else:
        print("FAIL: mismatch detected -- check construction.")
