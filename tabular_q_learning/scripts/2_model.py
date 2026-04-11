#!/usr/bin/env python3
"""
2_model.py — COCONUT Q-Learning Transformer

Custom GPT-2 style Transformer with a single output head:
    action_head : Linear(d_model, n_actions)  at <Select> positions (CE loss only)

The model has no explicit Q-value head. Instead, it must discover Q-value
tracking internally through the COCONUT continuous thought mechanism. The
THINK token's hidden state is injected into the COT token's embedding before
the final transformer pass, allowing the model to propagate implicit Q-value
state across rounds.

Reward scalars are injected into the embedding of each TOK_R token via a learned
linear projection `reward_proj : R^1 -> R^d_model`, making the reward value
accessible to the transformer as a continuous signal.

Vocabulary layout (matches 1_generate_data.py):
    vocab_size = 2 + n_states + n_actions + 5

Architecture:
    - Token embedding    : nn.Embedding(vocab_size, d_model)
    - Position embedding : nn.Embedding(max_seq_len, d_model)
    - Reward projection  : nn.Linear(1, d_model, bias=False)
    - 4 x TransformerBlock (pre-norm, causal MHA + FFN with GELU)
    - Final LayerNorm
    - action_head : nn.Linear(d_model, n_actions)

Default capacity: d_model=256, n_heads=8, d_ff=1024, n_layers=4 (~3.5M params)
"""

import math
from dataclasses import dataclass, field, asdict
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class COCONUTConfig:
    n_states:    int   = 4
    n_actions:   int   = 2
    n_layers:    int   = 4
    n_heads:     int   = 8
    d_model:     int   = 256
    d_ff:        int   = 1024     # feed-forward hidden dim (4 * d_model)
    dropout:     float = 0.1
    max_seq_len: int   = 1024
    n_q_bins:    int   = 32       # number of Q-value discretization bins
    q_bin_min:   float = 0.0      # lower bound for Q-value binning
    q_bin_max:   float = 5.0      # upper bound (~1/(1-gamma) with headroom)

    @property
    def vocab_size(self) -> int:
        return 2 + self.n_states + self.n_actions + 5 + self.n_q_bins

    def to_dict(self) -> Dict:
        d = asdict(self)
        d['vocab_size'] = self.vocab_size
        return d

    @classmethod
    def from_dict(cls, d: Dict) -> 'COCONUTConfig':
        keys = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in keys})


# ---------------------------------------------------------------------------
# Vocabulary helpers (mirrors 1_generate_data.py)
# ---------------------------------------------------------------------------

def build_vocab(n_states: int, n_actions: int, n_q_bins: int = 32) -> Dict[str, object]:
    """Return token ID constants for the COCONUT vocabulary."""
    TOK_R = 2 + n_states + n_actions
    old_vocab = 2 + n_states + n_actions + 5
    return {
        'TOK_NULL':   0,
        'TOK_START':  1,
        'TOK_S':      list(range(2, 2 + n_states)),
        'TOK_A':      list(range(2 + n_states, 2 + n_states + n_actions)),
        'TOK_R':      TOK_R,
        'TOK_EVAL':   TOK_R + 1,
        'TOK_SELECT': TOK_R + 2,
        'TOK_THINK':  TOK_R + 3,
        'TOK_COT':    TOK_R + 4,
        'TOK_QBIN':   list(range(old_vocab, old_vocab + n_q_bins)),
        'vocab_size': old_vocab + n_q_bins,
    }


def discretize_q_value(q: float, n_bins: int, q_min: float, q_max: float) -> int:
    """Map a continuous Q-value to a bin index in [0, n_bins-1]."""
    frac = (q - q_min) / (q_max - q_min) if q_max > q_min else 0.0
    frac = max(0.0, min(1.0, frac))
    return min(int(frac * n_bins), n_bins - 1)


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class CausalMultiHeadAttention(nn.Module):
    """Multi-head self-attention with upper-triangular causal mask."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.n_heads = n_heads
        self.d_head  = d_model // n_heads
        self.scale   = math.sqrt(self.d_head)

        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : [B, T, d_model]
        Returns : [B, T, d_model]
        """
        B, T, C = x.shape
        qkv = self.qkv_proj(x)                             # [B, T, 3C]
        q, k, v = qkv.split(C, dim=-1)                     # each [B, T, C]

        # Reshape to [B, n_heads, T, d_head]
        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        # Scaled dot-product attention with causal mask
        attn = (q @ k.transpose(-2, -1)) / self.scale      # [B, H, T, T]
        causal_mask = torch.triu(
            torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1
        )
        attn = attn.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v)                                    # [B, H, T, d_head]
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.out_proj(out))

    def forward_with_kv_cache(
        self,
        x: torch.Tensor,
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Attention with KV cache for incremental decoding.

        Parameters
        ----------
        x      : [B, T_new, d_model] — new token embeddings only
        past_kv: (past_k, past_v) each [B, H, T_past, d_head], or None

        Returns
        -------
        out    : [B, T_new, d_model]
        new_kv : (k, v) each [B, H, T_past+T_new, d_head]
        """
        B, T_new, C = x.shape
        qkv = self.qkv_proj(x)                              # [B, T_new, 3C]
        q, k, v = qkv.split(C, dim=-1)

        q = q.view(B, T_new, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, T_new, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T_new, self.n_heads, self.d_head).transpose(1, 2)

        if past_kv is not None:
            k = torch.cat([past_kv[0], k], dim=2)
            v = torch.cat([past_kv[1], v], dim=2)

        T_total = k.shape[2]
        attn = (q @ k.transpose(-2, -1)) / self.scale       # [B, H, T_new, T_total]
        causal_mask = torch.triu(
            torch.ones(T_new, T_total, device=x.device, dtype=torch.bool),
            diagonal=T_total - T_new + 1,
        )
        attn = attn.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).contiguous().view(B, T_new, C)
        return self.resid_drop(self.out_proj(out)), (k, v)


class FeedForward(nn.Module):
    """Position-wise feed-forward network with GELU activation."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    """Pre-norm Transformer block: LayerNorm -> Attention -> residual,
                                   LayerNorm -> FFN -> residual."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn  = CausalMultiHeadAttention(d_model, n_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff    = FeedForward(d_model, d_ff, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x

    def forward_with_kv_cache(
        self,
        x: torch.Tensor,
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Pre-norm block with KV cache for incremental decoding."""
        attn_out, new_kv = self.attn.forward_with_kv_cache(self.norm1(x), past_kv)
        x = x + attn_out
        x = x + self.ff(self.norm2(x))
        return x, new_kv


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class COCONUTTransformer(nn.Module):
    """
    COCONUT Q-Learning Transformer.

    Architecture
    ------------
    1. Token embedding  + position embedding
    2. Reward scalar injection at TOK_R positions via reward_proj
    3. 4x TransformerBlock (pre-norm, causal)
    4. Final LayerNorm
    5. action_head : hidden[select_positions] -> [B, n_sel, n_actions] logits

    The model has a SINGLE output head (action_head). It has no explicit
    Q-value head — Q-value tracking must emerge implicitly in the COCONUT
    continuous thought state.

    Forward inputs
    --------------
    input_ids        : [B, T]       long
    reward_values    : [B, n_r]     float  (actual reward scalars, NOT token IDs)
    reward_positions : [B, n_r]     long   (positions of TOK_R in input_ids; -1 = pad)
    select_positions : [B, n_sel]   long   (positions of TOK_SELECT; -1 = pad)

    COCONUT additional inputs
    -------------------------
    think_positions  : [B, n_rounds] long  (positions of TOK_THINK; -1 = pad)
    cot_positions    : [B, n_rounds] long  (positions of TOK_COT; -1 = pad)

    Padding convention
    ------------------
    All position tensors use -1 as pad sentinel.  The model clamps negative
    positions to 0 before gathering (safe because padding is masked out in
    the loss), and returns zeros for padded slots.
    """

    def __init__(self, config: COCONUTConfig):
        super().__init__()
        self.config    = config
        self.d_model   = config.d_model
        self.n_actions = config.n_actions

        # Embeddings
        self.tok_emb  = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_emb  = nn.Embedding(config.max_seq_len, config.d_model)

        # Continuous reward injection: maps scalar r -> d_model-dim delta
        self.reward_proj = nn.Linear(1, config.d_model, bias=False)

        self.emb_drop = nn.Dropout(config.dropout)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(config.d_model, config.n_heads, config.d_ff, config.dropout)
            for _ in range(config.n_layers)
        ])

        self.final_norm = nn.LayerNorm(config.d_model)

        # Single output head — CE loss on action prediction at SELECT positions
        self.action_head = nn.Linear(config.d_model, config.n_actions)

        # Explanation head — CE loss on discrete explain tokens (Hao curriculum)
        self.explain_head = nn.Linear(config.d_model, config.vocab_size)

        # Precompute token ID offsets (mirrors build_vocab in 1_generate_data.py)
        self._tok_s_start = 2                    # TOK_S[i] = 2 + i
        self._tok_a_start = 2 + config.n_states  # TOK_A[i] = 2 + n_states + i

        # Teacher COT construction parameters (for forward_teacher_forced)
        # cot_update_bias: learnable "this is a Q-update COT" signal
        # eval_vector: direction in d_model space along which Q-values are encoded
        self.cot_update_bias = nn.Parameter(torch.empty(config.d_model))
        self.eval_vector     = nn.Parameter(torch.empty(config.d_model))

        self._init_weights()

        # Initialize teacher COT params after _init_weights (explicit init overrides)
        nn.init.normal_(self.cot_update_bias, mean=0.0, std=0.02)
        v = torch.randn(config.d_model)
        v = v / v.norm()
        self.eval_vector.data.copy_(v)

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    # ------------------------------------------------------------------
    # Core building blocks: embeddings and transformer passes
    # ------------------------------------------------------------------

    def _embed(
        self,
        input_ids: torch.Tensor,           # [B, T]
        reward_values: torch.Tensor,       # [B, n_r]
        reward_positions: torch.Tensor,    # [B, n_r]
    ) -> torch.Tensor:
        """Build embeddings with reward scalar injection."""
        B, T = input_ids.shape
        device = input_ids.device

        # Token + position embeddings
        positions = torch.arange(T, device=device).unsqueeze(0).expand(B, -1)  # [B, T]
        x = self.tok_emb(input_ids) + self.pos_emb(positions)                  # [B, T, d]

        # Inject reward scalars at TOK_R positions
        if reward_values is not None and reward_positions is not None:
            n_r = reward_values.shape[1]
            if n_r > 0:
                rv = reward_values.unsqueeze(-1).float()        # [B, n_r, 1]
                deltas = self.reward_proj(rv)                   # [B, n_r, d]

                safe_pos = reward_positions.clamp(min=0)        # [B, n_r]
                pos_idx = safe_pos.unsqueeze(-1).expand(-1, -1, self.d_model)

                valid = (reward_positions >= 0).unsqueeze(-1).expand_as(deltas)
                deltas = deltas * valid.float()

                x = x.scatter_add(1, pos_idx, deltas)

        return self.emb_drop(x)

    def _run_transformer(self, x: torch.Tensor) -> torch.Tensor:
        """Run all transformer blocks + final norm."""
        for block in self.blocks:
            x = block(x)
        return self.final_norm(x)

    def _run_blocks_only(self, x: torch.Tensor) -> torch.Tensor:
        """Run transformer blocks WITHOUT final_norm.

        Used during COCONUT prefix passes so that injected COT embeddings
        are raw block outputs — matching the scale of other positions'
        embeddings going into the final full-sequence pass.
        """
        for block in self.blocks:
            x = block(x)
        return x

    def _run_blocks_cached(
        self,
        x_new: torch.Tensor,
        past_kvs: Optional[list] = None,
    ) -> Tuple[torch.Tensor, list]:
        """Run blocks on new tokens only, reusing cached K/V.

        Parameters
        ----------
        x_new    : [B, T_new, d] — embeddings for new token(s)
        past_kvs : list of (past_k, past_v) per layer, or None for cold start

        Returns
        -------
        h_new    : [B, T_new, d] — block outputs (no final_norm)
        new_kvs  : list of (k, v) per layer (past + new concatenated)
        """
        new_kvs = []
        for i, block in enumerate(self.blocks):
            pkv = past_kvs[i] if past_kvs is not None else None
            x_new, kv = block.forward_with_kv_cache(x_new, pkv)
            new_kvs.append(kv)
        return x_new, new_kvs

    # ------------------------------------------------------------------
    # Gather helper
    # ------------------------------------------------------------------

    @staticmethod
    def _gather_at_positions(
        hidden: torch.Tensor,      # [B, T, d]
        positions: torch.Tensor,   # [B, n_pos]  — may contain -1 padding
    ) -> torch.Tensor:
        """Extract hidden states at the given positions.

        Negative (padding) positions are clamped to 0; callers must mask
        the resulting output before computing loss.

        Returns : [B, n_pos, d]
        """
        B, T, d = hidden.shape
        safe = positions.clamp(min=0)                      # [B, n_pos]
        idx  = safe.unsqueeze(-1).expand(-1, -1, d)        # [B, n_pos, d]
        return hidden.gather(1, idx)                        # [B, n_pos, d]

    # ------------------------------------------------------------------
    # Teacher COT embedding construction
    # ------------------------------------------------------------------

    def construct_teacher_cot_embedding(
        self,
        st_ids:   torch.Tensor,   # [B] long — permuted state indices (0-based)
        at_ids:   torch.Tensor,   # [B] long — permuted action indices (0-based)
        q_values: torch.Tensor,   # [B] float — Q_{t+1}(s_t, a_t)
    ) -> torch.Tensor:
        """Build teacher-prescribed COT embeddings for one round.

        Encodes the Q-value update as:
            id_part  = cot_update_bias + tok_emb(S[st]) + tok_emb(A[at])
            buf_part = q_value * eval_vector
            result   = id_part + buf_part   shape [B, d_model]

        This provides the model a structured signal of *what* Q-value was just
        updated for *which* (s, a) pair, giving Phase A training concrete targets
        to learn from before Phase B switches to model-generated COT.

        Returns
        -------
        teacher_emb : [B, d_model]
        """
        # Convert 0-based state/action indices to vocabulary token IDs
        tok_s_ids = st_ids + self._tok_s_start   # [B]
        tok_a_ids = at_ids + self._tok_a_start   # [B]

        # Identity part: learnable bias + state embedding + action embedding
        id_part = (
            self.cot_update_bias.unsqueeze(0)    # [1, d_model] broadcast to [B, d_model]
            + self.tok_emb(tok_s_ids)            # [B, d_model]
            + self.tok_emb(tok_a_ids)            # [B, d_model]
        )

        # Buffer part: Q-value scalar projected onto the eval_vector direction
        buf_part = q_values.unsqueeze(-1) * self.eval_vector.unsqueeze(0)  # [B, d_model]

        return id_part + buf_part   # [B, d_model]

    # ------------------------------------------------------------------
    # Standard forward (single pass — no COCONUT feedback)
    # Used as baseline / ablation (--no_coconut flag in training/eval)
    # ------------------------------------------------------------------

    def forward(
        self,
        input_ids:        torch.Tensor,            # [B, T]
        reward_values:    Optional[torch.Tensor],  # [B, n_r]
        reward_positions: Optional[torch.Tensor],  # [B, n_r]
        select_positions: torch.Tensor,            # [B, n_sel]
    ) -> torch.Tensor:
        """Single-pass forward. COT tokens use their static learned embedding.

        Returns
        -------
        action_logits : [B, n_sel, n_actions]
        """
        x = self._embed(input_ids, reward_values, reward_positions)
        h = self._run_transformer(x)   # [B, T, d_model]

        sel_hidden    = self._gather_at_positions(h, select_positions)  # [B, n_sel, d]
        action_logits = self.action_head(sel_hidden)                     # [B, n_sel, n_actions]
        return action_logits

    # ------------------------------------------------------------------
    # COCONUT forward (with continuous thought feedback) — corrected
    # ------------------------------------------------------------------

    def forward_coconut(
        self,
        input_ids:             torch.Tensor,            # [B, T]
        reward_values:         Optional[torch.Tensor],  # [B, n_r]
        reward_positions:      Optional[torch.Tensor],  # [B, n_r]
        select_positions:      torch.Tensor,            # [B, n_sel]
        think_positions:       torch.Tensor,            # [B, n_rounds]
        cot_positions:         torch.Tensor,            # [B, n_rounds]
        return_hidden:         bool = False,
        truncate_bptt_window:  int  = 5,
    ):
        """COCONUT forward pass with continuous thought feedback.

        For each round r, runs the transformer blocks (WITHOUT final_norm)
        on the prefix up to the THINK position, then injects the hidden state
        at THINK into the COT position. This corrects the distribution mismatch
        bug in the previous version (which applied final_norm during prefix passes,
        making injected embeddings out of distribution with other positions).

        After all rounds, a single full-sequence pass through blocks + final_norm
        produces the final hidden states used for action prediction.

        Parameters
        ----------
        return_hidden : if True, return (action_logits, h_final) so the caller
                        can extract hidden states at arbitrary positions
                        (needed for Q-value probing in evaluation).

        Returns
        -------
        action_logits : [B, n_sel, n_actions]
        h_final       : [B, T, d_model]  (only if return_hidden=True)
        """
        B, T = input_ids.shape
        n_rounds = think_positions.shape[1]

        # Step 1: Compute all embeddings (token + position + reward injection).
        # COT positions initially hold the learned TOK_COT token embedding;
        # we overwrite them sequentially in the loop below.
        x = self._embed(input_ids, reward_values, reward_positions).clone()  # [B, T, d]

        # Step 2: Sequential round-by-round COCONUT feedback.
        # For each round, run blocks (NOT final_norm) on the prefix up to THINK,
        # extract the hidden state at THINK, inject it into COT.
        for r in range(n_rounds):
            think_pos_r = think_positions[:, r]   # [B]
            cot_pos_r   = cot_positions[:, r]     # [B]
            valid = (cot_pos_r >= 0)              # [B] — False for padding rounds

            if not valid.any():
                break  # all remaining rounds are padding

            # Run transformer blocks (WITHOUT final_norm) on prefix up to
            # and including the THINK position. Causal mask is generated
            # dynamically from seq_len, so variable-length prefixes work correctly.
            max_prefix = int(think_pos_r[valid].max().item()) + 1
            h = self._run_blocks_only(x[:, :max_prefix, :].clone())  # [B, max_prefix, d]

            # Inject hidden state at THINK into COT position.
            # Truncated BPTT: only the last `truncate_bptt_window` rounds keep
            # live gradients; earlier rounds are detached to keep the graph shallow.
            # When n_rounds <= truncate_bptt_window (early curriculum), all rounds
            # have live gradients since the detach condition is never true.
            for b in range(B):
                if valid[b]:
                    t_pos = int(think_pos_r[b].item())
                    c_pos = int(cot_pos_r[b].item())
                    if r < n_rounds - truncate_bptt_window:
                        x[b, c_pos, :] = h[b, t_pos, :].detach()
                    else:
                        x[b, c_pos, :] = h[b, t_pos, :]

        # Step 3: Final full-sequence forward pass with COT embeddings injected.
        # Uses _run_transformer (blocks + final_norm) for the complete hidden states.
        h_final = self._run_transformer(x)  # [B, T, d]

        # Step 4: Extract action logits at SELECT positions.
        sel_hidden    = self._gather_at_positions(h_final, select_positions)  # [B, n_sel, d]
        action_logits = self.action_head(sel_hidden)                           # [B, n_sel, n_actions]

        if return_hidden:
            return action_logits, h_final
        return action_logits

    # ------------------------------------------------------------------
    # Teacher-forced forward (Phase A training — single pass, no recurrence)
    # ------------------------------------------------------------------

    def forward_teacher_forced(
        self,
        input_ids:              torch.Tensor,            # [B, T]
        reward_values:          Optional[torch.Tensor],  # [B, n_r]
        reward_positions:       Optional[torch.Tensor],  # [B, n_r]
        select_positions:       torch.Tensor,            # [B, n_sel]
        cot_positions:          torch.Tensor,            # [B, n_rounds]
        teacher_cot_embeddings: torch.Tensor,            # [B, n_rounds, d_model]
    ) -> torch.Tensor:
        """Teacher-forced COCONUT forward pass.

        Instead of running sequential prefix passes to produce COT embeddings,
        directly injects pre-computed teacher_cot_embeddings at each COT position.
        Then does a single full-sequence transformer pass.

        This is O(1) forward passes (vs O(n_rounds) for forward_coconut), giving
        ~5–8x speedup for Phase A training. Gradients flow through the injected
        teacher embeddings back into construct_teacher_cot_embedding's parameters
        (cot_update_bias, eval_vector, tok_emb for the s/a tokens).

        Parameters
        ----------
        teacher_cot_embeddings : [B, n_rounds, d_model]  — from construct_teacher_cot_embedding

        Returns
        -------
        action_logits : [B, n_sel, n_actions]
        """
        B = input_ids.shape[0]
        n_rounds = cot_positions.shape[1]

        # Build full embeddings (token + position + reward injection)
        x = self._embed(input_ids, reward_values, reward_positions).clone()  # [B, T, d]

        # Overwrite COT positions with teacher embeddings
        for r in range(n_rounds):
            cot_pos_r = cot_positions[:, r]   # [B]
            valid = (cot_pos_r >= 0)          # [B]
            for b in range(B):
                if valid[b]:
                    c_pos = int(cot_pos_r[b].item())
                    x[b, c_pos, :] = teacher_cot_embeddings[b, r, :]

        # Single full-sequence forward pass (blocks + final_norm)
        h = self._run_transformer(x)          # [B, T, d]

        # Extract action logits at SELECT positions
        sel_hidden    = self._gather_at_positions(h, select_positions)  # [B, n_sel, d]
        action_logits = self.action_head(sel_hidden)                     # [B, n_sel, n_actions]
        return action_logits

    # ------------------------------------------------------------------
    # Hao-style forward (mixed discrete/continuous explanation tokens)
    # ------------------------------------------------------------------

    def forward_hao(
        self,
        input_ids:             torch.Tensor,            # [B, T]
        reward_values:         Optional[torch.Tensor],  # [B, n_r]
        reward_positions:      Optional[torch.Tensor],  # [B, n_r]
        select_positions:      torch.Tensor,            # [B, n_sel]
        think_positions:       torch.Tensor,            # [B, n_rounds]
        explain_positions:     torch.Tensor,            # [B, n_rounds, 3]
        n_continuous:          int,                      # 0, 1, 2, or 3
        return_hidden:         bool = False,
        truncate_bptt_window:  int  = 5,
    ):
        """Hao-style forward with progressive discrete->continuous explanation.

        Each round has 3 explanation token positions. The first (3-n_continuous)
        are discrete tokens with standard embeddings; the last n_continuous are
        continuous thoughts produced via COCONUT hidden-state feedback.

        Uses KV caching for the COCONUT injection loop to avoid recomputing the
        full prefix for each continuous thought. After all injections, a final
        full-sequence pass through blocks+final_norm produces the hidden states
        used for action prediction and explanation logits.

        Parameters
        ----------
        explain_positions : [B, n_rounds, 3] — positions of the 3 explain tokens
        n_continuous      : how many of the 3 are continuous (0=fully discrete)
        return_hidden     : if True, also return h_final [B, T, d_model] for probing

        Returns
        -------
        action_logits  : [B, n_sel, n_actions]
        explain_logits : [B, n_rounds, 3, vocab_size]
        h_final        : [B, T, d_model]  (only if return_hidden=True)
        """
        B, T = input_ids.shape
        n_rounds = think_positions.shape[1]
        n_discrete = 3 - n_continuous

        # ---- Fast path: fully discrete (Stage 0) ----
        if n_continuous == 0:
            x = self._embed(input_ids, reward_values, reward_positions)
            h = self._run_transformer(x)                                # [B, T, d]

            sel_h = self._gather_at_positions(h, select_positions)      # [B, n_sel, d]
            action_logits = self.action_head(sel_h)                     # [B, n_sel, n_act]

            ep_flat = explain_positions.view(B, -1)                     # [B, n_rounds*3]
            exp_h = self._gather_at_positions(h, ep_flat)               # [B, n_rounds*3, d]
            exp_h = exp_h.view(B, n_rounds, 3, self.d_model)
            explain_logits = self.explain_head(exp_h)                   # [B, n_rounds, 3, V]

            if return_hidden:
                return action_logits, explain_logits, h
            return action_logits, explain_logits

        # ---- COCONUT path: n_continuous > 0 ----
        # Build embeddings. Continuous positions hold TOK_COT placeholders
        # which will be overwritten by hidden-state feedback below.
        x = self._embed(input_ids, reward_values, reward_positions).clone()  # [B, T, d]

        # KV-cached injection loop: incrementally process tokens and inject
        # continuous thoughts. The cache avoids recomputing the full prefix
        # each time we need a hidden state for injection.
        kv_cache = None    # list of (k, v) per layer
        cache_end = 0      # exclusive end of cached positions
        h_last = None      # [B, d] — block output at last cached position

        for r in range(n_rounds):
            think_pos_r = think_positions[:, r]       # [B]
            exp_pos_r   = explain_positions[:, r, :]  # [B, 3]
            valid = (think_pos_r >= 0)

            if not valid.any():
                break

            detach = (r < n_rounds - truncate_bptt_window)

            # --- Feed discrete tokens: everything from cache_end through
            #     THINK + any discrete explanation tokens ---
            if n_discrete > 0:
                feed_to = int(exp_pos_r[valid, n_discrete - 1].max().item()) + 1
            else:
                feed_to = int(think_pos_r[valid].max().item()) + 1

            if feed_to > cache_end:
                new_x = x[:, cache_end:feed_to, :]
                if detach:
                    new_x = new_x.detach()
                h_new, kv_cache = self._run_blocks_cached(new_x, kv_cache)
                h_last = h_new[:, -1, :]   # [B, d]
                cache_end = feed_to

            # --- Chain continuous thoughts ---
            for j in range(n_continuous):
                exp_idx = n_discrete + j
                # Inject source hidden into target continuous-thought position
                for b in range(B):
                    if valid[b]:
                        tgt = int(exp_pos_r[b, exp_idx].item())
                        if detach:
                            x[b, tgt, :] = h_last[b].detach()
                        else:
                            x[b, tgt, :] = h_last[b]

                # Feed the (now-injected) continuous thought token
                max_tgt = int(exp_pos_r[valid, exp_idx].max().item()) + 1
                ct_x = x[:, cache_end:max_tgt, :]
                if detach:
                    ct_x = ct_x.detach()
                h_ct, kv_cache = self._run_blocks_cached(ct_x, kv_cache)
                h_last = h_ct[:, -1, :]
                cache_end = max_tgt

        # --- Final full-sequence pass ---
        # All continuous thought embeddings have been injected into x.
        # Run full blocks+final_norm to get hidden states at all positions.
        # (Same pattern as forward_coconut's final step.)
        h_final = self._run_transformer(x)  # [B, T, d]

        # Extract action logits at SELECT positions
        sel_h = self._gather_at_positions(h_final, select_positions)
        action_logits = self.action_head(sel_h)

        # Extract explain logits at all 3 explanation positions per round
        ep_flat = explain_positions.view(B, -1)
        exp_h = self._gather_at_positions(h_final, ep_flat)
        exp_h = exp_h.view(B, n_rounds, 3, self.d_model)
        explain_logits = self.explain_head(exp_h)

        if return_hidden:
            return action_logits, explain_logits, h_final
        return action_logits, explain_logits

    # ------------------------------------------------------------------
    # Parameter count
    # ------------------------------------------------------------------

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------

def print_model_summary(config: COCONUTConfig) -> None:
    """Instantiate model, print architecture and parameter counts."""
    model = COCONUTTransformer(config)
    total = model.num_parameters()

    print("=" * 60)
    print("COCONUTTransformer — Architecture Summary")
    print("=" * 60)
    print(f"  n_states    : {config.n_states}")
    print(f"  n_actions   : {config.n_actions}")
    print(f"  vocab_size  : {config.vocab_size}")
    print(f"  n_layers    : {config.n_layers}")
    print(f"  n_heads     : {config.n_heads}")
    print(f"  d_model     : {config.d_model}")
    print(f"  d_ff        : {config.d_ff}")
    print(f"  max_seq_len : {config.max_seq_len}")
    print(f"  dropout     : {config.dropout}")
    print()
    print("  Modules:")
    for name, mod in model.named_children():
        n_params = sum(p.numel() for p in mod.parameters() if p.requires_grad)
        print(f"    {name:<20}  {n_params:>10,} params")
    print()
    print(f"  Total trainable parameters: {total:,}")
    print("=" * 60)

    # Quick forward-pass smoke test
    B, T, n_r, n_sel, n_rounds = 2, 64, 5, 5, 5
    dummy_ids    = torch.randint(0, config.vocab_size, (B, T))
    dummy_rv     = torch.rand(B, n_r)
    dummy_rp     = torch.randint(0, T - 2, (B, n_r))
    dummy_sel    = torch.randint(0, T, (B, n_sel))
    # think_positions must be strictly before cot_positions
    dummy_think  = torch.sort(torch.randint(1, T - 1, (B, n_rounds)), dim=1).values
    dummy_cot    = dummy_think + 1
    # clamp to valid range
    dummy_cot    = dummy_cot.clamp(max=T - 1)

    model.eval()
    with torch.no_grad():
        # Test standard forward
        al = model(dummy_ids, dummy_rv, dummy_rp, dummy_sel)
        # Test COCONUT forward with return_hidden
        al_coc, h_fin = model.forward_coconut(
            dummy_ids, dummy_rv, dummy_rp, dummy_sel,
            dummy_think, dummy_cot, return_hidden=True
        )

    print(f"  Smoke test — standard forward:")
    print(f"    input_ids     : {list(dummy_ids.shape)}")
    print(f"    action_logits : {list(al.shape)}   (expect [B={B}, n_sel={n_sel}, n_actions={config.n_actions}])")
    print(f"  Smoke test — forward_coconut (return_hidden=True):")
    print(f"    action_logits : {list(al_coc.shape)}")
    print(f"    h_final       : {list(h_fin.shape)}  (expect [B={B}, T={T}, d_model={config.d_model}])")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Main (prints model summary when run directly)
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    cfg = COCONUTConfig(
        n_states=4, n_actions=2,
        n_layers=4, n_heads=8, d_model=256, d_ff=1024,
        dropout=0.1, max_seq_len=1024,
    )
    print_model_summary(cfg)
