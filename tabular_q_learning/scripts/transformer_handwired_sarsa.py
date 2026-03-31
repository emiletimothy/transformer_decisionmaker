"""
PyTorch Handwired SARSA Transformer (Section 4.1 Construction).

Implements the 3-layer causal transformer from Section 4 of the paper using
PyTorch nn.Module — analogous to transformer_handwired_multiplicative_weights.py
for Section 3.

The transformer maintains a growing causal sequence (COCONUT) of event tokens
v^(0), v^(1), ..., v^(t-1), which serve as content-addressable Q-value memory.
One MDP transition is processed per step(); the emitted event token v^(t) is
appended to the sequence after each round.

Embedding layout  (d_model = 3 * D_TE + D_PE,  D_TE = 12):
    id(x)   = x[0:12]      one-hot token identity + e_val scalar lane
    buf1(x) = x[12:24]     first buffer  (rewards, stored Q-values)
    buf2(x) = x[24:36]     second buffer (retrieved Q-values)
    pos(x)  = x[36:]       positional encoding (recency signal)

Token vocabulary  (vocab_size = 11):
    0 = Start
    1–4 = S0–S3  (states)
    5–6 = A0–A1  (actions)
    7 = r         (reward token)
    8 = Qcurr     (workspace)
    9 = Qnext     (workspace)
    10 = Update   (workspace / emitted event token)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TOK_START  = 0
TOK_S      = [1, 2, 3, 4]   # TOK_S[s] = token index for state s
TOK_A      = [5, 6]          # TOK_A[a] = token index for action a
TOK_R      = 7
TOK_QCURR  = 8
TOK_QNEXT  = 9
TOK_UPDATE = 10

VOCAB_SIZE = 11
D_TE       = 12          # token-embedding subspace dim  (|V| + 1 for e_val lane)
E_VAL      = D_TE - 1   # = 11: scalar lane orthogonal to all token one-hots
D_PE       = 12          # positional-encoding subspace dim
D_MODEL    = 3 * D_TE + D_PE   # = 48

BETA    = 100.0
EPS_REC = 0.01

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
    d_te:      int   = D_TE
    d_pe:      int   = D_PE
    vocab_size:int   = VOCAB_SIZE


# ---------------------------------------------------------------------------
# Attention module  (identical to the MWU version)
# ---------------------------------------------------------------------------

class MultiHeadAttention(nn.Module):
    """Standard scaled dot-product multi-head attention."""

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k     = d_model // n_heads

        self.query  = nn.Linear(d_model, d_model)
        self.key    = nn.Linear(d_model, d_model)
        self.value  = nn.Linear(d_model, d_model)
        self.output = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        Q = self.query(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        K = self.key(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        V = self.value(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        scores  = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_k)
        weights = F.softmax(scores, dim=-1)
        out     = torch.matmul(weights, V)
        out     = out.transpose(1, 2).contiguous().view(B, T, self.d_model)
        return self.output(out)


# ---------------------------------------------------------------------------
# Main transformer class
# ---------------------------------------------------------------------------

class SARSATransformer(nn.Module):
    """
    3-layer PyTorch transformer implementing online tabular SARSA.

    Architecture (Section 4.1):
    ┌─────────────────────────────────────────────────────────────────┐
    │ Layer 1 – Within-round info gathering  (4 fixed-offset heads)   │
    │   Heads 1-2 at Qcurr:  buf1(Qcurr) = id(s_t) + id(a_t)       │
    │   Heads 3-4 at Qnext:  buf1(Qnext) = id(s_{t+1})+id(a_{t+1}) │
    ├─────────────────────────────────────────────────────────────────┤
    │ Layer 2 – Q-value retrieval from history  (2 content heads)     │
    │   Head 1: buf2(Qcurr)[e_val] ← Qt(s_t, a_t)                   │
    │   Head 2: buf2(Qnext)[e_val] ← Qt(s_{t+1}, a_{t+1})           │
    ├─────────────────────────────────────────────────────────────────┤
    │ Layer 3 – SARSA aggregation + event-token emission  (4 heads)   │
    │   Heads 1-3: buf1(Update)[e_val] ← (1-α)Q+αγQ'+αr             │
    │   Head 4:    id(Update) ← u˜_Update + u˜_s + u˜_a             │
    └─────────────────────────────────────────────────────────────────┘

    COCONUT: after each round the rewritten Update token is appended to
    _event_tokens as v^(t), where it persists as retrievable Q-value memory.
    """

    def __init__(self, config: SARSATokenConfig):
        super().__init__()
        self.config = config
        d = 3 * config.d_te + config.d_pe   # d_model = 48

        # Embeddings — present for structural symmetry with the MWU transformer;
        # the exact handwired construction uses _build_embeddings instead.
        self.token_embedding    = nn.Embedding(config.vocab_size, d)
        self.position_embedding = nn.Embedding(1000, d)

        # Layer 1: 4 fixed-offset copy heads
        self.layer1_head1 = MultiHeadAttention(d, 1)   # Qcurr ← s_t
        self.layer1_head2 = MultiHeadAttention(d, 1)   # Qcurr ← a_t
        self.layer1_head3 = MultiHeadAttention(d, 1)   # Qnext ← s_{t+1}
        self.layer1_head4 = MultiHeadAttention(d, 1)   # Qnext ← a_{t+1}

        # Layer 2: 2 content-match retrieval heads
        self.layer2_head1 = MultiHeadAttention(d, 1)   # retrieve Qt(s_t, a_t)
        self.layer2_head2 = MultiHeadAttention(d, 1)   # retrieve Qt(s_{t+1}, a_{t+1})

        # Layer 3: 4 aggregation + emission heads
        self.layer3_head1 = MultiHeadAttention(d, 1)   # Qcurr buf2 → (1-α) term
        self.layer3_head2 = MultiHeadAttention(d, 1)   # Qnext buf2 → αγ term
        self.layer3_head3 = MultiHeadAttention(d, 1)   # r_token buf1 → α term
        self.layer3_head4 = MultiHeadAttention(d, 1)   # identity copy → Update id

        # Layer-normalisation modules (used during gradient-based training;
        # skipped in the exact handwired forward pass to preserve scalar precision)
        self.layer_norm1 = nn.LayerNorm(d)
        self.layer_norm2 = nn.LayerNorm(d)
        self.layer_norm3 = nn.LayerNorm(d)

        # Q-value readout
        self.q_value_head = nn.Linear(d, 1)

        # COCONUT growing sequence: Start token + emitted event tokens v^(t)
        self._event_tokens: List[torch.Tensor] = []
        self.q_history:     List[np.ndarray]   = []
        self.step_count:    int                = 0

        self._init_weights()
        self._reset_sequence()

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def _reset_sequence(self) -> None:
        """Place the Start token at position 0 (returns Q=0 for unvisited pairs)."""
        cfg = self.config
        d   = 3 * cfg.d_te + cfg.d_pe

        start = torch.zeros(d, dtype=torch.float64)
        for tok in range(cfg.vocab_size):
            start[tok] = 1.0   # id: superposition of all token one-hots (eq. 6)
        # buf1, buf2 remain zero → retrieval from Start returns Q = 0

        self._event_tokens = [start]
        self.q_history     = [np.zeros((cfg.n_states, cfg.n_actions))]
        self.step_count    = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def create_input_stream(
        self,
        s: int, a: int, r: float, s_next: int, a_next: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build the 8-token stream for one MDP transition.

        Round block layout:
            pos 0: s_t      pos 1: a_t      pos 2: r_t
            pos 3: s_{t+1}  pos 4: a_{t+1}
            pos 5: Qcurr    pos 6: Qnext    pos 7: Update

        Returns:
            input_ids:    (8,) token IDs
            position_ids: (8,) absolute positions in the full growing sequence
        """
        base = len(self._event_tokens)   # absolute offset of the first new token
        input_ids = torch.tensor([
            TOK_S[s],      TOK_A[a],      TOK_R,
            TOK_S[s_next], TOK_A[a_next],
            TOK_QCURR,     TOK_QNEXT,     TOK_UPDATE,
        ], dtype=torch.long)
        position_ids = torch.arange(base, base + 8, dtype=torch.long)
        return input_ids, position_ids

    def step(
        self, s: int, a: int, r: float, s_next: int, a_next: int
    ) -> float:
        """
        Process one MDP transition and return Q_{t+1}(s, a).

        Args:
            s, a:           Current state and action.
            r:              Observed reward.
            s_next, a_next: Next state and on-policy next action.

        Returns:
            Updated Q-value Q_{t+1}(s, a).
        """
        input_ids, position_ids = self.create_input_stream(s, a, r, s_next, a_next)
        result = self.forward(input_ids, position_ids, r)
        q_new  = result['q_value']

        Q_new = self.q_history[-1].copy()
        Q_new[s, a] = q_new
        self.q_history.append(Q_new)
        self.step_count += 1

        logger.debug("Step %d: Q(%d,%d) <- %.6f", self.step_count, s, a, q_new)
        return q_new

    def get_q_history(self) -> List[np.ndarray]:
        """Return Q-table snapshots; index 0 is the all-zeros initialisation."""
        return self.q_history

    def reset(self) -> None:
        """Reset to initial state (empty sequence except Start token)."""
        self._reset_sequence()

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(
        self,
        input_ids:    torch.Tensor,
        position_ids: torch.Tensor,
        r:            float,
    ) -> Dict:
        """
        3-layer forward pass for one round's 8-token block.

        Args:
            input_ids:    (8,) token IDs for the current round.
            position_ids: (8,) absolute positions in the growing sequence.
            r:            Reward scalar (seeded into buf1[E_VAL] of r-token).

        Returns:
            dict with 'q_value' (float) and 'hidden_state' (1, 8, d_model).
        """
        # Build exact paper embeddings for the 8-token block
        x = self._build_embeddings(input_ids, r)   # (1, 8, d_model)

        # Layer 1: copy s/a identities into buf1 of workspace tokens
        x = self._layer1_within_round(x, input_ids)

        # Layer 2: retrieve Q-values from the COCONUT history
        x = self._layer2_qvalue_retrieval(x, input_ids)

        # Layer 3: compute SARSA update; produce event token v^(t)
        x, q_new = self._layer3_aggregation(x, input_ids)

        # COCONUT: append emitted event token to the growing sequence
        upd_pos     = (input_ids == TOK_UPDATE).nonzero(as_tuple=True)[0][0].item()
        event_token = x[0, upd_pos].detach().clone()
        self._event_tokens.append(event_token)

        return {'q_value': q_new, 'hidden_state': x}

    # ------------------------------------------------------------------
    # Embedding construction
    # ------------------------------------------------------------------

    def _build_embeddings(
        self, input_ids: torch.Tensor, r: float
    ) -> torch.Tensor:
        """
        Construct exact paper embeddings for the 8-token round block.

        Each token i receives a one-hot in id[token_id] (index within [0:d_te]).
        The reward token additionally receives buf1[E_VAL] = r.

        Returns:
            x: (1, 8, d_model)
        """
        cfg  = self.config
        d_te = cfg.d_te
        d    = 3 * d_te + cfg.d_pe

        x = torch.zeros(1, input_ids.shape[0], d, dtype=torch.float64)
        for i, tok in enumerate(input_ids.tolist()):
            x[0, i, tok] = 1.0   # one-hot identity in id subspace [0:d_te]

        # Seed reward into buf1[E_VAL] of the r-token  (buf1 starts at d_te)
        r_idx = (input_ids == TOK_R).nonzero(as_tuple=True)[0][0].item()
        x[0, r_idx, d_te + E_VAL] = r
        return x

    # ------------------------------------------------------------------
    # Layer 1: within-round information gathering
    # ------------------------------------------------------------------

    def _layer1_within_round(
        self, x: torch.Tensor, input_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        Copy state/action identities into buf1 of Qcurr and Qnext.

        Heads 1-2 at Qcurr (offsets ℓ=5,4):
            buf1(Qcurr) = id(s_t) + id(a_t)              [eq. 10]
        Heads 3-4 at Qnext (offsets ℓ=3,2):
            buf1(Qnext) = id(s_{t+1}) + id(a_{t+1})      [eq. 11]

        Block layout: [s_t(0), a_t(1), r_t(2), s'(3), a'(4), Qcurr(5), Qnext(6), Update(7)]
        """
        cfg  = self.config
        d_te = cfg.d_te
        attn = torch.zeros_like(x)

        qcurr_idx = (input_ids == TOK_QCURR).nonzero(as_tuple=True)[0][0].item()
        qnext_idx = (input_ids == TOK_QNEXT).nonzero(as_tuple=True)[0][0].item()

        # Fixed offsets: Qcurr is at block index 5
        s_t_idx    = qcurr_idx - 5   # → 0
        a_t_idx    = qcurr_idx - 4   # → 1
        # Qnext is at block index 6
        s_next_idx = qnext_idx - 3   # → 3
        a_next_idx = qnext_idx - 2   # → 4

        # Heads 1-2: buf1(Qcurr) = id(s_t) + id(a_t)
        attn[0, qcurr_idx, d_te:2*d_te] = (
            x[0, s_t_idx, :d_te] + x[0, a_t_idx, :d_te]
        )

        # Heads 3-4: buf1(Qnext) = id(s_{t+1}) + id(a_{t+1})
        attn[0, qnext_idx, d_te:2*d_te] = (
            x[0, s_next_idx, :d_te] + x[0, a_next_idx, :d_te]
        )

        return x + attn

    # ------------------------------------------------------------------
    # Layer 2: Q-value retrieval from history
    # ------------------------------------------------------------------

    def _layer2_qvalue_retrieval(
        self, x: torch.Tensor, input_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        Retrieve most-recent Qt(s,a) values via content-match over history.

        For each workspace token (Qcurr, Qnext):
          query  = buf1(workspace)    [= u˜_s + u˜_a after Layer 1]
          winner = argmax_j  β·<query, id(j)>  +  eps_rec·f(j)
          write:  buf2(workspace)[E_VAL] ← buf1(winner)[E_VAL]

        Scoring:
          matching event token or Start → 2·β  (highest; fallback Q=0 from Start)
          all other tokens              → ≤ β
        Ref: Section 4.1, Layer 2 (eqs. 12-13).
        """
        cfg  = self.config
        d_te = cfg.d_te
        attn = torch.zeros_like(x)

        for tok_id in (TOK_QCURR, TOK_QNEXT):
            idx_t = (input_ids == tok_id).nonzero(as_tuple=True)[0]
            if len(idx_t) == 0:
                continue
            idx   = idx_t[0].item()
            query = x[0, idx, d_te:2*d_te]               # buf1 of workspace token
            q_val = self._retrieve_q_from_history(query)
            attn[0, idx, 2*d_te + E_VAL] = q_val          # → buf2[E_VAL]

        return x + attn

    def _retrieve_q_from_history(self, query: torch.Tensor) -> float:
        """
        Scan COCONUT history; return buf1[E_VAL] of the best-scoring token.

        score(j) = β · <query, id(j)>  +  eps_rec · (1 – 1/(j+1))

        Args:
            query: (d_te,) vector — buf1 of a workspace token (= u˜_s + u˜_a).

        Returns:
            Q-scalar from buf1[E_VAL] of the winning event token.
        """
        cfg        = self.config
        d_te       = cfg.d_te
        best_score = float('-inf')
        best_j     = 0
        q          = query.float()

        for j, tok_vec in enumerate(self._event_tokens):
            id_j    = tok_vec[:d_te].float()
            content = cfg.beta    * float(torch.dot(q, id_j))
            recency = cfg.eps_rec * (1.0 - 1.0 / (j + 1))
            score   = content + recency
            if score > best_score:
                best_score = score
                best_j     = j

        return float(self._event_tokens[best_j][d_te + E_VAL])

    # ------------------------------------------------------------------
    # Layer 3: SARSA aggregation and event-token emission
    # ------------------------------------------------------------------

    def _layer3_aggregation(
        self, x: torch.Tensor, input_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, float]:
        """
        Compute SARSA update at Update position; build event token v^(t).

        Head 1 (offset ℓ=2 → Qcurr):   reads buf2[E_VAL], weight (1-α)
        Head 2 (offset ℓ=1 → Qnext):   reads buf2[E_VAL], weight αγ
        Head 3 (offset ℓ=5 → r_token): reads buf1[E_VAL], weight α
        Head 4 (offset ℓ=2 → Qcurr):   copies buf1(Qcurr) → id(Update),
                                         then adds u˜_Update

        After this layer the Update token satisfies the invariant (eq. 8):
            id(v^(t))          = u˜_Update + u˜_s + u˜_a
            buf1(v^(t))[E_VAL] = Q_{t+1}(s, a)
        Ref: Section 4.1, Layer 3 (eqs. 14-15).

        Returns:
            x_out: tensor with Update rewritten as v^(t).
            q_new: Q_{t+1}(s, a) scalar.
        """
        cfg  = self.config
        d_te = cfg.d_te

        upd_idx   = (input_ids == TOK_UPDATE).nonzero(as_tuple=True)[0][0].item()
        qcurr_idx = upd_idx - 2    # Qcurr is 2 before Update
        qnext_idx = upd_idx - 1    # Qnext is 1 before Update
        r_idx     = upd_idx - 5    # r_t is 5 before Update

        # Read scalars from the relevant buffers
        q_curr = float(x[0, qcurr_idx, 2*d_te + E_VAL])   # buf2[E_VAL] of Qcurr
        q_next = float(x[0, qnext_idx, 2*d_te + E_VAL])   # buf2[E_VAL] of Qnext
        r_val  = float(x[0, r_idx,     d_te   + E_VAL])   # buf1[E_VAL] of r-token

        # SARSA TD update  (eq. 9)
        q_new = (
            (1.0 - cfg.alpha) * q_curr
            + cfg.alpha * cfg.gamma * q_next
            + cfg.alpha * r_val
        )

        # Write event token into the Update position (clone to preserve grad flow)
        x_out = x.clone()

        # buf1[E_VAL](Update) ← Q_{t+1}(s, a)       (eq. 14)
        x_out[0, upd_idx, d_te + E_VAL] = q_new

        # id(Update) ← u˜_Update + u˜_s + u˜_a      (eq. 15)
        sa_id  = x[0, qcurr_idx, d_te:2*d_te].clone()   # u˜_s + u˜_a from buf1(Qcurr)
        new_id = torch.zeros(d_te)
        new_id[TOK_UPDATE] = 1.0                          # u˜_Update one-hot
        new_id = new_id + sa_id                           # + u˜_s + u˜_a
        x_out[0, upd_idx, :d_te] = new_id

        logger.debug(
            "Layer3: q_curr=%.4f q_next=%.4f r=%.4f → q_new=%.6f",
            q_curr, q_next, r_val, q_new,
        )
        return x_out, q_new


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
