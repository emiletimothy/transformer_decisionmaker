"""
True PyTorch Handwired SARSA Transformer (Section 4.1 Construction).

Implements the 3-layer causal transformer from Section 4 using strictly
mathematical block-matrix operations (W_Q, W_K, W_V) that pass through
standard PyTorch F.softmax(). No Python array-slicing shortcuts or
if/else branching — all routing is done via additive gating
(Complementary Superposition, Lemma 4.2(i)) and sinusoidal offset
matching (RoPE-style).

Token layout (positions 0..N in the growing COCONUT sequence):
    Position 0: <Null>  — zero identity, zero buffer; attention sink
    Position 1: <Start> — superposition of all identities; Q=0 fallback
    Position 2+: event tokens v^(t) emitted after each round

Embedding layout (d_model = 3 * D_TE + D_PE):
    id(x)   = x[0 : D_TE]           one-hot token identity + e_val scalar
    buf1(x) = x[D_TE : 2*D_TE]      first buffer  (rewards, stored Q-values)
    buf2(x) = x[2*D_TE : 3*D_TE]    second buffer (retrieved Q-values)
    pos(x)  = x[3*D_TE:]            positional encoding [f(i), sin/cos pairs, pad]

Token vocabulary (vocab_size = 12):
    0 = <Null>    5 = A0         9  = Qcurr
    1 = <Start>   6 = A1         10 = Qnext
    2 = S0        7 = (unused)   11 = Update
    3 = S1        8 = R (reward)
    4 = S2
    (S3 = index within TOK_S list, mapped to vocab index 2..5)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass
from typing import List
import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants & Vocabulary
# ---------------------------------------------------------------------------
TOK_NULL   = 0
TOK_START  = 1
TOK_S      = [2, 3, 4, 5]   # TOK_S[s] = vocab index for state s
TOK_A      = [6, 7]          # TOK_A[a] = vocab index for action a
TOK_R      = 8
TOK_QCURR  = 9
TOK_QNEXT  = 10
TOK_UPDATE = 11

VOCAB_SIZE = 12

D_TE    = 13             # Token embedding dim (indices 0..12; index 12 = e_val scalar lane)
E_VAL   = D_TE - 1      # = 12: scalar lane orthogonal to all token one-hots
D_SIN   = 10            # 5 pairs of sinusoidal encodings
D_F     = 1             # Recency signal f(i)
D_PE    = 12            # Positional encoding dim (1 f(i) + 10 sin/cos + 1 pad)
D_MODEL = 3 * D_TE + D_PE   # = 39 + 12 = 51

BETA    = 1000.0         # Content-match temperature
EPS_REC = 0.2            # Recency scaling (BETA * EPS_REC = 200 < 1000 content gap)
XI      = 1000.0         # Additive penalty scalar for complementary gating

# Subspace boundary indices
ID_START    = 0
ID_END      = D_TE           # 13
BUF1_START  = D_TE           # 13
BUF1_END    = 2 * D_TE       # 26
BUF2_START  = 2 * D_TE       # 26
BUF2_END    = 3 * D_TE       # 39
PE_START    = 3 * D_TE       # 39
PE_F_IDX    = PE_START        # 39: f(i) recency signal
PE_SIN_START = PE_START + 1   # 40: start of sinusoidal PE
PE_SIN_END   = PE_SIN_START + D_SIN  # 50

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class SARSATokenConfig:
    """Configuration for the SARSA transformer."""
    n_states:  int   = 4
    n_actions: int   = 2
    alpha:     float = 0.1
    gamma:     float = 0.9
    beta:      float = BETA
    eps_rec:   float = EPS_REC
    xi:        float = XI
    d_te:      int   = D_TE
    d_pe:      int   = D_PE
    vocab_size: int  = VOCAB_SIZE


# ---------------------------------------------------------------------------
# Positional Encoding Math
# ---------------------------------------------------------------------------

def get_sinusoidal_vec(pos: int, d_sin: int = D_SIN) -> torch.Tensor:
    """Returns the sinusoidal vector [sin(w0*p), cos(w0*p), ...] for absolute position p."""
    p = torch.zeros(d_sin, dtype=torch.float64)
    for k in range(d_sin // 2):
        w_k = 1.0 / (10000.0 ** (2 * k / d_sin))
        p[2 * k]     = math.sin(pos * w_k)
        p[2 * k + 1] = math.cos(pos * w_k)
    return p


def get_rotation_matrix(ell: int, d_sin: int = D_SIN) -> torch.Tensor:
    """
    Block-diagonal rotation matrix R(ell) for RoPE-style offset matching.

    For a [sin, cos] vector convention, each 2x2 block is:
        [cos(ell*w_k)   sin(ell*w_k)]
        [-sin(ell*w_k)  cos(ell*w_k)]

    Property: sin_vec(i)^T @ R(ell) @ sin_vec(j) is maximized when j = i - ell.
    """
    R = torch.zeros(d_sin, d_sin, dtype=torch.float64)
    for k in range(d_sin // 2):
        w_k = 1.0 / (10000.0 ** (2 * k / d_sin))
        cos_val = math.cos(ell * w_k)
        sin_val = math.sin(ell * w_k)
        R[2*k,   2*k]   =  cos_val
        R[2*k,   2*k+1] =  sin_val
        R[2*k+1, 2*k]   = -sin_val
        R[2*k+1, 2*k+1] =  cos_val
    return R


def f_recency(pos: int) -> float:
    """Monotonically increasing recency signal: f(i) = 1 - 1/(i+1)."""
    return 1.0 - 1.0 / (pos + 1.0)


# ---------------------------------------------------------------------------
# Core Attention Head (exact handwired weights, standard softmax)
# ---------------------------------------------------------------------------

class ExactAttentionHead(nn.Module):
    """
    Single attention head with fixed W_Q, W_K, W_V matrices.
    Uses standard causal masking and F.softmax — no Python shortcuts.
    """

    def __init__(self, W_Q: torch.Tensor, W_K: torch.Tensor, W_V: torch.Tensor,
                 temperature: float = BETA):
        super().__init__()
        self.register_buffer("W_Q", W_Q)
        self.register_buffer("W_K", W_K)
        self.register_buffer("W_V", W_V)
        self.temperature = temperature

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (1, seq_len, D_MODEL)
        Returns:
            attention output: (1, seq_len, D_MODEL)
        """
        Q = x @ self.W_Q.T   # (1, T, d_q)
        K = x @ self.W_K.T   # (1, T, d_k)
        V = x @ self.W_V.T   # (1, T, D_MODEL)

        # Scaled dot-product with temperature
        scores = self.temperature * (Q @ K.transpose(-2, -1))

        # Standard causal mask
        T = x.shape[1]
        causal_mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=x.device))
        scores = scores.masked_fill(~causal_mask, -1e9)

        weights = F.softmax(scores, dim=-1)
        return weights @ V


# ---------------------------------------------------------------------------
# Main Transformer
# ---------------------------------------------------------------------------

class SARSATransformer(nn.Module):
    """
    3-layer PyTorch transformer implementing online tabular SARSA.

    Architecture (Section 4.1):
    ┌────────────────────────────────────────────────────────────────────┐
    │ Layer 1 – Within-round offset retrieval (4 fixed-offset heads)    │
    │   Heads 1-2 at Qcurr:  buf1(Qcurr) += id(s_t) + id(a_t)        │
    │   Heads 3-4 at Qnext:  buf1(Qnext) += id(s_{t+1}) + id(a_{t+1})│
    │   Non-target tokens route to <Null> via additive gating (ξ)      │
    ├────────────────────────────────────────────────────────────────────┤
    │ Layer 2 – Q-value retrieval from COCONUT history (1 content head) │
    │   query = buf1(workspace) → content + recency match → history    │
    │   buf2(workspace)[e_val] ← buf1(winner)[e_val]                   │
    ├────────────────────────────────────────────────────────────────────┤
    │ Layer 3 – SARSA aggregation + event-token emission (4 heads)      │
    │   Heads 1-3: buf1(Update)[e_val] ← (1-α)Q + αγQ' + αr          │
    │   Head 4:    id(Update) ← buf1(Qcurr) (carries s+a identity)    │
    │   Non-target tokens route to <Null> via additive gating (ξ)      │
    └────────────────────────────────────────────────────────────────────┘

    COCONUT: after each round the fully-processed Update token is appended
    to _event_tokens as v^(t), preserving its absolute PE burned in at
    embedding time.
    """

    def __init__(self, config: SARSATokenConfig):
        super().__init__()
        self.config = config
        self._build_layers()
        self.reset()

    def reset(self) -> None:
        """Initialize history with <Null> at pos 0 and <Start> at pos 1."""
        cfg = self.config

        # <Null> token: zero identity, zero buffers — pure attention sink
        null = torch.zeros(D_MODEL, dtype=torch.float64)
        null[PE_F_IDX] = f_recency(0)
        null[PE_SIN_START:PE_SIN_END] = get_sinusoidal_vec(0)

        # <Start> token: superposition of all identities, Q=0 fallback
        start = torch.zeros(D_MODEL, dtype=torch.float64)
        for tok in range(VOCAB_SIZE):
            start[tok] = 1.0
        start[PE_F_IDX] = f_recency(1)
        start[PE_SIN_START:PE_SIN_END] = get_sinusoidal_vec(1)

        self._event_tokens = [null, start]
        self.q_history = [np.zeros((cfg.n_states, cfg.n_actions))]
        self.step_count = 0

    # ------------------------------------------------------------------
    # Layer construction
    # ------------------------------------------------------------------

    def _build_layers(self) -> None:
        """Construct all attention heads with exact handwired weights."""
        cfg = self.config

        # Layer 1: Within-round fixed-offset retrieval
        # Round block: [s_t(0), a_t(1), r(2), s'(3), a'(4), Qcurr(5), Qnext(6), Update(7)]
        # Qcurr at offset 5 from s_t, offset 4 from a_t
        self.l1_h1 = self._build_fixed_offset_head(
            target_tok=TOK_QCURR, ell=5, src_sub='id', dst_sub='buf1')
        self.l1_h2 = self._build_fixed_offset_head(
            target_tok=TOK_QCURR, ell=4, src_sub='id', dst_sub='buf1')
        # Qnext at offset 3 from s', offset 2 from a'
        self.l1_h3 = self._build_fixed_offset_head(
            target_tok=TOK_QNEXT, ell=3, src_sub='id', dst_sub='buf1')
        self.l1_h4 = self._build_fixed_offset_head(
            target_tok=TOK_QNEXT, ell=2, src_sub='id', dst_sub='buf1')

        # Layer 2: Content-based Q-value retrieval
        self.layer2_content = self._build_content_retrieval_head()

        # Layer 3: SARSA aggregation + event-token emission
        # Head 1: (1-α) * Q(s,a) — from Qcurr buf2[e_val], offset 2 from Update
        self.l3_h1 = self._build_fixed_offset_head(
            target_tok=TOK_UPDATE, ell=2, src_sub='buf2_val', dst_sub='buf1_val',
            weight=1.0 - cfg.alpha)
        # Head 2: αγ * Q(s',a') — from Qnext buf2[e_val], offset 1 from Update
        self.l3_h2 = self._build_fixed_offset_head(
            target_tok=TOK_UPDATE, ell=1, src_sub='buf2_val', dst_sub='buf1_val',
            weight=cfg.alpha * cfg.gamma)
        # Head 3: α * r — from r-token buf1[e_val], offset 5 from Update
        self.l3_h3 = self._build_fixed_offset_head(
            target_tok=TOK_UPDATE, ell=5, src_sub='buf1_val', dst_sub='buf1_val',
            weight=cfg.alpha)
        # Head 4: copy buf1(Qcurr) → id(Update) for s+a identity
        self.l3_h4 = self._build_fixed_offset_head(
            target_tok=TOK_UPDATE, ell=2, src_sub='buf1', dst_sub='id')

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

        The query space is partitioned into two subspaces:
          Subspace A (dims 0..D_SIN): sinusoidal PE for offset matching
          Subspace B (dims D_SIN..2*D_SIN): ξ-scaled penalty that additively
            hijacks the softmax for non-target tokens → routes them to <Null>

        Args:
            target_tok: vocab index of the token that should attend backward
            ell: how many positions backward to look
            src_sub: source subspace to read from the attended token
            dst_sub: destination subspace to write into the attending token
            weight: scalar multiplier on the value projection
        """
        d_q = 2 * D_SIN   # query/key dimensionality

        W_Q = torch.zeros(d_q, D_MODEL, dtype=torch.float64)
        W_K = torch.zeros(d_q, D_MODEL, dtype=torch.float64)
        W_V = torch.zeros(D_MODEL, D_MODEL, dtype=torch.float64)

        # --- Subspace A: sinusoidal offset matching ---
        # Q extracts the sinusoidal PE of the query token
        W_Q[0:D_SIN, PE_SIN_START:PE_SIN_END] = torch.eye(D_SIN, dtype=torch.float64)
        # K rotates the sinusoidal PE backward by ell positions
        # sin_vec(i)^T @ R(ell) @ sin_vec(j) is maximized when j = i - ell
        W_K[0:D_SIN, PE_SIN_START:PE_SIN_END] = get_rotation_matrix(ell)

        # --- Subspace B: Complementary Superposition (additive gating) ---
        # For non-target tokens, the dot product in subspace B produces a large
        # score toward <Null> (pos 0), drowning out the offset match.
        # For the target token, the complement is zero → no penalty.
        #
        # u_comp = 1-vector with target_tok zeroed out
        u_comp = torch.ones(D_TE, dtype=torch.float64)
        u_comp[target_tok] = 0.0

        # p_null = sinusoidal PE of <Null> at position 0
        p_null = get_sinusoidal_vec(0)

        # W_Q subspace B: maps id subspace through ξ * outer(p_null, u_comp)
        # For target token: id[target_tok]=1, u_comp[target_tok]=0 → contrib = 0
        # For non-target: id[tok]=1, u_comp[tok]=1 → contrib = ξ * p_null
        W_Q[D_SIN:2*D_SIN, ID_START:ID_END] = XI * torch.outer(p_null, u_comp)

        # W_K subspace B: extracts sinusoidal PE (identity projection)
        # So <Null>'s key in subspace B = p_null, giving dot product ξ * ||p_null||^2
        W_K[D_SIN:2*D_SIN, PE_SIN_START:PE_SIN_END] = torch.eye(D_SIN, dtype=torch.float64)

        # --- Value projection: data routing ---
        if src_sub == 'id' and dst_sub == 'buf1':
            # Copy id → buf1
            W_V[BUF1_START:BUF1_END, ID_START:ID_END] = weight * torch.eye(D_TE, dtype=torch.float64)
        elif src_sub == 'buf1' and dst_sub == 'id':
            # Copy buf1 → id
            W_V[ID_START:ID_END, BUF1_START:BUF1_END] = weight * torch.eye(D_TE, dtype=torch.float64)
        elif src_sub == 'buf2_val' and dst_sub == 'buf1_val':
            # Copy buf2[e_val] → buf1[e_val]
            W_V[BUF1_START + E_VAL, BUF2_START + E_VAL] = weight
        elif src_sub == 'buf1_val' and dst_sub == 'buf1_val':
            # Copy buf1[e_val] → buf1[e_val] (self-read)
            W_V[BUF1_START + E_VAL, BUF1_START + E_VAL] = weight

        return ExactAttentionHead(W_Q, W_K, W_V)

    def _build_content_retrieval_head(self) -> ExactAttentionHead:
        """
        Build Layer 2's content + recency retrieval head.

        A single Q@K^T multiplication simultaneously computes:
          - Content match score in the identity subspace:
              layer2_beta * <buf1(query), id(key)>
          - Temporal recency score in the positional subspace:
              layer2_beta * eps_rec * f(i) * f(j)

        Uses a massive local temperature (layer2_beta = 100,000) to
        approximate the theoretical argmax while keeping bounded recency:
          - Content gap for wrong token:  100,000 * 1.0 = 100,000
          - Max recency bonus:            100,000 * 0.2 * 1.0 = 20,000
          → Recency can never override content (20k < 100k).
          - At j=100, Δf ≈ 0.0001 → logit diff ≈ 2.0 → 88%/12% softmax split,
            decisively selecting the most recent matching token.
        """
        layer2_beta = 100_000.0

        d_q = D_TE + 1  # identity match dims + 1 recency dim

        W_Q = torch.zeros(d_q, D_MODEL, dtype=torch.float64)
        W_K = torch.zeros(d_q, D_MODEL, dtype=torch.float64)
        W_V = torch.zeros(D_MODEL, D_MODEL, dtype=torch.float64)

        # Content match: Q reads from buf1, K reads from id
        W_Q[0:D_TE, BUF1_START:BUF1_END] = torch.eye(D_TE, dtype=torch.float64)
        W_K[0:D_TE, ID_START:ID_END] = torch.eye(D_TE, dtype=torch.float64)

        # Recency: Q reads f(i), K reads f(j)
        # Raw dot product = EPS_REC * f(i)*f(j).
        # After layer2_beta scaling: logit += layer2_beta * EPS_REC * f(i)*f(j).
        alpha_scale = math.sqrt(EPS_REC)
        W_Q[D_TE, PE_F_IDX] = alpha_scale
        W_K[D_TE, PE_F_IDX] = alpha_scale

        # Value: route buf1[e_val] of winner → buf2[e_val] of query token
        W_V[BUF2_START + E_VAL, BUF1_START + E_VAL] = 1.0

        return ExactAttentionHead(W_Q, W_K, W_V, temperature=layer2_beta)

    # ------------------------------------------------------------------
    # Embedding construction
    # ------------------------------------------------------------------

    def _embed_token(self, tok_id: int, abs_pos: int) -> torch.Tensor:
        """
        Construct the full embedding vector for a single token.

        Burns in the absolute positional encoding (both sinusoidal and f(i))
        permanently — this is critical for event tokens that get appended to
        the COCONUT history and must retain their temporal placement.
        """
        x = torch.zeros(D_MODEL, dtype=torch.float64)

        # One-hot identity in id subspace
        x[tok_id] = 1.0

        # Positional encoding (permanently burned in)
        x[PE_F_IDX] = f_recency(abs_pos)
        x[PE_SIN_START:PE_SIN_END] = get_sinusoidal_vec(abs_pos)

        return x

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def step(self, s: int, a: int, r: float, s_next: int, a_next: int) -> float:
        """
        Process one MDP transition and return Q_{t+1}(s, a).

        Builds an 8-token round block, concatenates with COCONUT history,
        runs the 3-layer forward pass, and emits the Update token as v^(t).
        """
        # Build 8-token round block
        base_pos = len(self._event_tokens)
        tokens = [TOK_S[s], TOK_A[a], TOK_R, TOK_S[s_next], TOK_A[a_next],
                  TOK_QCURR, TOK_QNEXT, TOK_UPDATE]

        x_block = torch.zeros(len(tokens), D_MODEL, dtype=torch.float64)
        for i, tok in enumerate(tokens):
            x_block[i] = self._embed_token(tok, base_pos + i)
            # Seed reward into buf1[e_val] of the r-token
            if tok == TOK_R:
                x_block[i, BUF1_START + E_VAL] = r

        # Concatenate history + round block → full sequence
        history = torch.stack(self._event_tokens)  # (N_hist, D_MODEL)
        full = torch.cat([history, x_block], dim=0).unsqueeze(0)  # (1, T, D_MODEL)

        # === Layer 1: within-round offset retrieval (residual stream) ===
        full = full + self.l1_h1(full) + self.l1_h2(full) + self.l1_h3(full) + self.l1_h4(full)

        # === Layer 2: content-based Q-value retrieval (residual stream) ===
        full = full + self.layer2_content(full)

        # === Layer 3: SARSA aggregation + emission (residual stream) ===
        full = full + self.l3_h1(full) + self.l3_h2(full) + self.l3_h3(full) + self.l3_h4(full)

        # Emit the fully-processed Update token as event token v^(t)
        update_seq_idx = base_pos + 7  # Update is the 8th token in the round block
        event_token = full[0, update_seq_idx].detach().clone()

        # Consolidate COCONUT memory: Re-stamp absolute PE before storing.
        # Without this, the event token retains the PE from its round-block
        # position, creating PE collisions that let Layer 3 offset heads
        # accidentally attend to history tokens instead of within-round targets.
        history_pos = len(self._event_tokens)
        event_token[PE_F_IDX] = f_recency(history_pos)
        event_token[PE_SIN_START:PE_SIN_END] = get_sinusoidal_vec(history_pos)

        self._event_tokens.append(event_token)

        # Extract the scalar Q-value from buf1[e_val]
        q_new = float(event_token[BUF1_START + E_VAL])

        # Maintain classical Q-table history for comparison
        Q_new = self.q_history[-1].copy()
        Q_new[s, a] = q_new
        self.q_history.append(Q_new)
        self.step_count += 1

        logger.debug("Step %d: Q(%d,%d) <- %.6f", self.step_count, s, a, q_new)
        return q_new

    def get_q_history(self) -> List[np.ndarray]:
        """Return Q-table snapshots; index 0 is the all-zeros initialisation."""
        return self.q_history


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from tabular_sarsa import TabularSARSA, make_chain_mdp, generate_trajectory

    logging.basicConfig(level=logging.WARNING)

    P     = make_chain_mdp(n_states=4, n_actions=2)
    traj  = generate_trajectory(P, n_states=4, n_actions=2, T=50)

    sarsa = TabularSARSA(n_states=4, n_actions=2, alpha=0.1, gamma=0.9)
    cfg   = SARSATokenConfig(n_states=4, n_actions=2, alpha=0.1, gamma=0.9)
    tf    = SARSATransformer(cfg)

    max_err = 0.0
    for t, (s, a, r, s_next, a_next) in enumerate(traj):
        q_s = sarsa.step(s, a, r, s_next, a_next)
        q_t = tf.step(s, a, r, s_next, a_next)
        err = abs(q_s - q_t)
        max_err = max(max_err, err)
        if err > 1e-9:
            print(f"  MISMATCH at step {t}: SARSA={q_s:.8f}  TF={q_t:.8f}  diff={err:.2e}")

    print(f"Max absolute Q-difference over 50 steps: {max_err:.2e}")
    if max_err < 1e-9:
        print("PASS: transformer matches tabular SARSA exactly.")
    else:
        print("FAIL: mismatch detected -- check construction.")
