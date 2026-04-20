#!/usr/bin/env python3
"""
2_model.py — COCONUT Q-Learning Transformer (Two-Phase Per-Round Layout)

Custom GPT-2 style Transformer with two output heads:
    action_head : Linear(d_model, n_actions)   at <Select> positions (CE loss always on)
    thought_head: Linear(d_model, vocab_size)  at discrete thought positions (CE loss on
                  rounds where continuous_round_mask == False)

Vocabulary layout (matches 1_generate_data.py):
    vocab_size = 2 + n_states + n_actions + 9 + n_q_bins

Architecture:
    - Token embedding    : nn.Embedding(vocab_size, d_model)
    - Position embedding : nn.Embedding(max_seq_len, d_model)
    - Reward projection  : nn.Linear(1, d_model, bias=False)
    - 4 x TransformerBlock (pre-norm, causal MHA + FFN with GELU)
    - Final LayerNorm
    - action_head  : nn.Linear(d_model, n_actions)
    - thought_head : nn.Linear(d_model, vocab_size)

COCONUT mechanism (forward_hao):
    For continuous rounds, the hidden state at TOK_UPDATE is extracted via
    _run_blocks_only and injected into the single TOK_COT position before the
    final full-sequence pass. Discrete rounds proceed without injection.

Default capacity: d_model=256, n_heads=8, d_ff=1024, n_layers=4 (~3.5M params)
"""

import math
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple

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
    d_ff:        int   = 1024
    dropout:     float = 0.1
    max_seq_len: int   = 1024
    n_q_bins:    int   = 32
    q_bin_min:   float = 0.0
    q_bin_max:   float = 5.0

    @property
    def vocab_size(self) -> int:
        return 2 + self.n_states + self.n_actions + 9 + self.n_q_bins

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
    """Return token ID constants for the two-phase COCONUT vocabulary."""
    TOK_R    = 2 + n_states + n_actions
    old_vocab = 2 + n_states + n_actions + 9  # 9 special tokens: R,EVAL,SEL,THINK,COT,UPD,QCURR,QNEXT,ANEXT
    return {
        'TOK_NULL':   0,
        'TOK_START':  1,
        'TOK_S':      list(range(2, 2 + n_states)),
        'TOK_A':      list(range(2 + n_states, 2 + n_states + n_actions)),
        'TOK_R':      TOK_R,
        'TOK_EVAL':   TOK_R + 1,
        'TOK_SELECT': TOK_R + 2,
        'TOK_THINK':  TOK_R + 3,   # legacy; never placed in sequences
        'TOK_COT':    TOK_R + 4,   # continuous thought placeholder
        'TOK_UPDATE': TOK_R + 5,   # Phase 2 end marker; COCONUT injection source
        'TOK_QCURR':  TOK_R + 6,   # scaffold (no loss)
        'TOK_QNEXT':  TOK_R + 7,   # scaffold (no loss)
        'TOK_ANEXT':  TOK_R + 8,   # a_{t+1} echo scaffold (no loss)
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
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head  = d_model // n_heads
        self.scale   = math.sqrt(self.d_head)

        self.qkv_proj  = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj  = nn.Linear(d_model, d_model, bias=False)
        self.attn_drop  = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, return_attn: bool = False):
        B, T, C = x.shape
        qkv = self.qkv_proj(x)
        q, k, v = qkv.split(C, dim=-1)

        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) / self.scale
        causal_mask = torch.triu(
            torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1
        )
        attn = attn.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        attn_weights = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn_weights)

        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, C)
        out = self.resid_drop(self.out_proj(out))
        if return_attn:
            return out, attn_weights
        return out

    def forward_with_kv_cache(
        self,
        x: torch.Tensor,
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        B, T_new, C = x.shape
        qkv = self.qkv_proj(x)
        q, k, v = qkv.split(C, dim=-1)

        q = q.view(B, T_new, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, T_new, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T_new, self.n_heads, self.d_head).transpose(1, 2)

        if past_kv is not None:
            k = torch.cat([past_kv[0], k], dim=2)
            v = torch.cat([past_kv[1], v], dim=2)

        T_total = k.shape[2]
        attn = (q @ k.transpose(-2, -1)) / self.scale
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
    """Pre-norm Transformer block."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn  = CausalMultiHeadAttention(d_model, n_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff    = FeedForward(d_model, d_ff, dropout)

    def forward(self, x: torch.Tensor, return_attn: bool = False):
        if return_attn:
            attn_out, attn_weights = self.attn(self.norm1(x), return_attn=True)
            x = x + attn_out
            x = x + self.ff(self.norm2(x))
            return x, attn_weights
        x = x + self.attn(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x

    def forward_with_kv_cache(
        self,
        x: torch.Tensor,
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        attn_out, new_kv = self.attn.forward_with_kv_cache(self.norm1(x), past_kv)
        x = x + attn_out
        x = x + self.ff(self.norm2(x))
        return x, new_kv


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class COCONUTTransformer(nn.Module):
    """
    COCONUT Q-Learning Transformer (two-phase per-round layout).

    Forward inputs (for forward_hao)
    ---------------------------------
    input_ids             : [B, T]         long
    reward_values         : [B, n_r]       float
    reward_positions      : [B, n_r]       long   (-1 = pad)
    select_positions      : [B, n_rounds]  long   (-1 = pad)
    update_positions      : [B, n_rounds]  long   (-1 = pad)
    thought_positions     : [B, n_rounds, 3] long (-1 = pad / continuous slot unused)
    continuous_round_mask : [B, n_rounds]  bool   (True = continuous thought round)
    """

    def __init__(self, config: COCONUTConfig):
        super().__init__()
        self.config    = config
        self.d_model   = config.d_model
        self.n_actions = config.n_actions

        self.tok_emb     = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_emb     = nn.Embedding(config.max_seq_len, config.d_model)
        self.reward_proj = nn.Linear(1, config.d_model, bias=False)
        self.emb_drop    = nn.Dropout(config.dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(config.d_model, config.n_heads, config.d_ff, config.dropout)
            for _ in range(config.n_layers)
        ])
        self.final_norm = nn.LayerNorm(config.d_model)

        self.action_head = nn.Linear(config.d_model, config.n_actions)
        self.thought_head = nn.Linear(config.d_model, config.vocab_size)

        # Legacy parameters kept for any callers of forward_teacher_forced
        self._tok_s_start = 2
        self._tok_a_start = 2 + config.n_states
        self.cot_update_bias = nn.Parameter(torch.empty(config.d_model))
        self.eval_vector     = nn.Parameter(torch.empty(config.d_model))

        self._init_weights()
        nn.init.normal_(self.cot_update_bias, mean=0.0, std=0.02)
        v = torch.randn(config.d_model)
        self.eval_vector.data.copy_(v / v.norm())

    @property
    def explain_head(self):
        """Backward-compat alias for thought_head (checkpoint key migration)."""
        return self.thought_head

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
    # Core building blocks
    # ------------------------------------------------------------------

    def _embed(
        self,
        input_ids:        torch.Tensor,
        reward_values:    torch.Tensor,
        reward_positions: torch.Tensor,
    ) -> torch.Tensor:
        B, T = input_ids.shape
        device = input_ids.device
        positions = torch.arange(T, device=device).unsqueeze(0).expand(B, -1)
        x = self.tok_emb(input_ids) + self.pos_emb(positions)

        if reward_values is not None and reward_positions is not None:
            n_r = reward_values.shape[1]
            if n_r > 0:
                rv = reward_values.unsqueeze(-1).float()
                deltas = self.reward_proj(rv)
                safe_pos = reward_positions.clamp(min=0)
                pos_idx = safe_pos.unsqueeze(-1).expand(-1, -1, self.d_model)
                valid = (reward_positions >= 0).unsqueeze(-1).expand_as(deltas)
                deltas = deltas * valid.float()
                x = x.scatter_add(1, pos_idx, deltas)

        return self.emb_drop(x)

    def _run_transformer(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return self.final_norm(x)

    def _run_transformer_with_attn(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        all_attn = []
        for block in self.blocks:
            x, attn_w = block(x, return_attn=True)
            all_attn.append(attn_w)
        h = self.final_norm(x)
        return h, torch.stack(all_attn, dim=0)

    def _run_blocks_only(self, x: torch.Tensor) -> torch.Tensor:
        """Run blocks WITHOUT final_norm (used for COCONUT prefix passes)."""
        for block in self.blocks:
            x = block(x)
        return x

    def _run_blocks_cached(
        self,
        x_new: torch.Tensor,
        past_kvs: Optional[list] = None,
    ) -> Tuple[torch.Tensor, list]:
        new_kvs = []
        for i, block in enumerate(self.blocks):
            pkv = past_kvs[i] if past_kvs is not None else None
            x_new, kv = block.forward_with_kv_cache(x_new, pkv)
            new_kvs.append(kv)
        return x_new, new_kvs

    @staticmethod
    def _gather_at_positions(
        hidden: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        """Extract hidden states at given positions. Pads clamp to 0."""
        B, T, d = hidden.shape
        safe = positions.clamp(min=0)
        idx  = safe.unsqueeze(-1).expand(-1, -1, d)
        return hidden.gather(1, idx)

    # ------------------------------------------------------------------
    # Standard forward (no COCONUT feedback) — baseline / ablation
    # ------------------------------------------------------------------

    def forward(
        self,
        input_ids:        torch.Tensor,
        reward_values:    Optional[torch.Tensor],
        reward_positions: Optional[torch.Tensor],
        select_positions: torch.Tensor,
    ) -> torch.Tensor:
        x = self._embed(input_ids, reward_values, reward_positions)
        h = self._run_transformer(x)
        sel_hidden    = self._gather_at_positions(h, select_positions)
        action_logits = self.action_head(sel_hidden)
        return action_logits

    # ------------------------------------------------------------------
    # forward_hao — two-phase per-round layout with round-level curriculum
    # ------------------------------------------------------------------

    def forward_hao(
        self,
        input_ids:             torch.Tensor,            # [B, T]
        reward_values:         Optional[torch.Tensor],  # [B, n_r]
        reward_positions:      Optional[torch.Tensor],  # [B, n_r]
        select_positions:      torch.Tensor,            # [B, n_rounds]
        update_positions:      torch.Tensor,            # [B, n_rounds]
        thought_positions:     torch.Tensor,            # [B, n_rounds, 3]
        continuous_round_mask: torch.Tensor,            # [B, n_rounds] bool
        return_hidden:         bool = False,
        return_attention:      bool = False,
        truncate_bptt_window:  int  = 5,
    ):
        """Two-phase COCONUT forward with round-level continuous/discrete curriculum.

        For discrete rounds: tokens are embedded normally; thought_logits extracted
        from the 3 thought positions after the final transformer pass.

        For continuous rounds: the hidden state at TOK_UPDATE (Phase 2 workspace
        end) is extracted via _run_blocks_only on the prefix, then injected into
        the single TOK_COT slot (thought_positions[:, r, 0]). Positions 1 and 2
        of thought_positions are -1 for continuous rounds.

        Parameters
        ----------
        continuous_round_mask : [B, n_rounds] bool
            True  = this round uses 1 continuous COT token (TOK_COT injection)
            False = this round uses 3 discrete thought tokens (supervised)

        Returns
        -------
        action_logits  : [B, n_rounds, n_actions]
        thought_logits : [B, n_rounds, 3, vocab_size]  — zero-filled for continuous rounds
        h_final        : [B, T, d_model]               (only if return_hidden=True)
        all_attn       : [L, B, H, T, T]               (only if return_attention=True and all discrete)
        """
        B, T = input_ids.shape
        n_rounds = select_positions.shape[1]

        # ---- Fast path: all rounds discrete ----
        if not continuous_round_mask.any():
            x = self._embed(input_ids, reward_values, reward_positions)
            if return_attention:
                h, all_attn = self._run_transformer_with_attn(x)
            else:
                h = self._run_transformer(x)
                all_attn = None

            sel_h = self._gather_at_positions(h, select_positions)
            action_logits = self.action_head(sel_h)

            tp_flat = thought_positions.view(B, -1)         # [B, n_rounds*3]
            # clamp -1 positions to 0; loss mask handles them
            tp_safe = tp_flat.clamp(min=0)
            th_h = self._gather_at_positions(h, tp_safe)    # [B, n_rounds*3, d]
            th_h = th_h.view(B, n_rounds, 3, self.d_model)
            thought_logits = self.thought_head(th_h)        # [B, n_rounds, 3, V]

            outs = (action_logits, thought_logits)
            if return_hidden and return_attention:
                return outs + (h, all_attn)
            if return_hidden:
                return outs + (h,)
            if return_attention:
                return outs + (all_attn,)
            return outs

        # ---- COCONUT path: some rounds are continuous ----
        x = self._embed(input_ids, reward_values, reward_positions).clone()

        kv_cache = None
        cache_end = 0

        for r in range(n_rounds):
            upd_pos_r  = update_positions[:, r]        # [B]
            cot_pos_r  = thought_positions[:, r, 0]    # [B] — COT slot for continuous rounds
            is_cont_r  = continuous_round_mask[:, r]   # [B] bool
            valid      = (upd_pos_r >= 0)

            if not valid.any():
                break

            # Only process continuous rounds in this loop
            cont_valid = valid & is_cont_r
            if not cont_valid.any():
                # All valid rounds this iteration are discrete — advance cache to
                # the last update position so we stay in sync.
                max_upd = int(upd_pos_r[valid].max().item()) + 1
                if max_upd > cache_end:
                    detach = (r < n_rounds - truncate_bptt_window)
                    new_x = x[:, cache_end:max_upd, :]
                    if detach:
                        new_x = new_x.detach()
                    _, kv_cache = self._run_blocks_cached(new_x, kv_cache)
                    cache_end = max_upd
                continue

            detach = (r < n_rounds - truncate_bptt_window)

            # Feed tokens up to (and including) the update position
            max_upd = int(upd_pos_r[valid].max().item()) + 1
            if max_upd > cache_end:
                new_x = x[:, cache_end:max_upd, :]
                if detach:
                    new_x = new_x.detach()
                h_new, kv_cache = self._run_blocks_cached(new_x, kv_cache)
                cache_end = max_upd
                h_upd = h_new[:, -1, :]  # hidden at update position [B, d]
            else:
                # Already cached up to here — run on the single update token
                upd_x = x[:, max_upd - 1:max_upd, :]
                if detach:
                    upd_x = upd_x.detach()
                h_new, kv_cache = self._run_blocks_cached(upd_x, kv_cache)
                h_upd = h_new[:, -1, :]

            # Inject hidden state at TOK_UPDATE into the COT position
            cont_bs  = cont_valid.nonzero(as_tuple=True)[0]  # [n_cont]
            tgt_pos  = cot_pos_r[cont_bs].long()              # [n_cont]
            src_vals = h_upd[cont_bs]                         # [n_cont, d]
            if detach:
                src_vals = src_vals.detach()
            x = torch.index_put(x, (cont_bs, tgt_pos), src_vals)

            # Feed the injected COT token through the cache
            max_cot = int(cot_pos_r[cont_valid].max().item()) + 1
            if max_cot > cache_end:
                ct_x = x[:, cache_end:max_cot, :]
                if detach:
                    ct_x = ct_x.detach()
                _, kv_cache = self._run_blocks_cached(ct_x, kv_cache)
                cache_end = max_cot

        # Final full-sequence pass
        h_final = self._run_transformer(x)

        # Action logits at SELECT positions
        sel_h = self._gather_at_positions(h_final, select_positions)
        action_logits = self.action_head(sel_h)

        # Thought logits: gather at all thought positions
        # For continuous rounds, thought_positions[:,r,1] and [:,r,2] are -1;
        # they get clamped to 0 here — loss masking in compute_hao_loss handles exclusion.
        tp_flat = thought_positions.view(B, -1).clamp(min=0)
        th_h = self._gather_at_positions(h_final, tp_flat)
        th_h = th_h.view(B, n_rounds, 3, self.d_model)
        thought_logits = self.thought_head(th_h)

        # Zero out thought logits for continuous rounds (no loss there)
        cont_mask_exp = continuous_round_mask.unsqueeze(-1).unsqueeze(-1)  # [B, R, 1, 1]
        thought_logits = thought_logits * (~cont_mask_exp).float()

        outs = (action_logits, thought_logits)
        if return_hidden:
            outs = outs + (h_final,)
        return outs

    # ------------------------------------------------------------------
    # COCONUT forward (legacy; kept for ablation / reference)
    # ------------------------------------------------------------------

    def forward_coconut(
        self,
        input_ids:             torch.Tensor,
        reward_values:         Optional[torch.Tensor],
        reward_positions:      Optional[torch.Tensor],
        select_positions:      torch.Tensor,
        think_positions:       torch.Tensor,
        cot_positions:         torch.Tensor,
        return_hidden:         bool = False,
        truncate_bptt_window:  int  = 5,
    ):
        """Legacy COCONUT forward for backward compatibility (old single-COT layout)."""
        B, T = input_ids.shape
        n_rounds = think_positions.shape[1]
        x = self._embed(input_ids, reward_values, reward_positions).clone()

        for r in range(n_rounds):
            think_pos_r = think_positions[:, r]
            cot_pos_r   = cot_positions[:, r]
            valid = (cot_pos_r >= 0)
            if not valid.any():
                break
            max_prefix = int(think_pos_r[valid].max().item()) + 1
            h = self._run_blocks_only(x[:, :max_prefix, :].clone())
            for b in range(B):
                if valid[b]:
                    t_pos = int(think_pos_r[b].item())
                    c_pos = int(cot_pos_r[b].item())
                    if r < n_rounds - truncate_bptt_window:
                        x[b, c_pos, :] = h[b, t_pos, :].detach()
                    else:
                        x[b, c_pos, :] = h[b, t_pos, :]

        h_final = self._run_transformer(x)
        sel_hidden    = self._gather_at_positions(h_final, select_positions)
        action_logits = self.action_head(sel_hidden)

        if return_hidden:
            return action_logits, h_final
        return action_logits

    # ------------------------------------------------------------------
    # Teacher-forced forward (deprecated; left for reference)
    # ------------------------------------------------------------------

    def construct_teacher_cot_embedding(
        self,
        st_ids:   torch.Tensor,
        at_ids:   torch.Tensor,
        q_values: torch.Tensor,
    ) -> torch.Tensor:
        import warnings
        warnings.warn(
            "construct_teacher_cot_embedding is deprecated in the two-phase layout.",
            DeprecationWarning, stacklevel=2,
        )
        tok_s_ids = st_ids + self._tok_s_start
        tok_a_ids = at_ids + self._tok_a_start
        id_part = (
            self.cot_update_bias.unsqueeze(0)
            + self.tok_emb(tok_s_ids)
            + self.tok_emb(tok_a_ids)
        )
        buf_part = q_values.unsqueeze(-1) * self.eval_vector.unsqueeze(0)
        return id_part + buf_part

    def forward_teacher_forced(self, *args, **kwargs):
        import warnings
        warnings.warn(
            "forward_teacher_forced is deprecated in the two-phase layout.",
            DeprecationWarning, stacklevel=2,
        )
        raise NotImplementedError(
            "forward_teacher_forced is not supported in the two-phase layout. "
            "Use forward_hao instead."
        )

    # ------------------------------------------------------------------
    # Parameter count
    # ------------------------------------------------------------------

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------

def print_model_summary(config: COCONUTConfig) -> None:
    model = COCONUTTransformer(config)
    total = model.num_parameters()

    print("=" * 60)
    print("COCONUTTransformer — Two-Phase Layout Summary")
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

    # Smoke test: two-phase forward_hao (all discrete)
    B, n_rounds = 2, 3
    # 18 tokens/round (discrete) + 2 prefix = 56 tokens for n_actions=2
    round_len_disc = 4 + 3 * config.n_actions + 1 + 4 + 3
    T = 2 + n_rounds * round_len_disc

    dummy_ids  = torch.randint(0, config.vocab_size, (B, T))
    dummy_rv   = torch.rand(B, n_rounds)
    dummy_rp   = torch.randint(0, T - 2, (B, n_rounds))
    dummy_sel  = torch.randint(0, T, (B, n_rounds))
    dummy_upd  = (dummy_sel + 2).clamp(max=T - 1)
    dummy_th   = torch.stack([
        dummy_upd + 1,
        dummy_upd + 2,
        dummy_upd + 3,
    ], dim=-1).clamp(max=T - 1)                              # [B, n_rounds, 3]
    dummy_mask = torch.zeros(B, n_rounds, dtype=torch.bool)  # all discrete

    model.eval()
    with torch.no_grad():
        al, tl = model.forward_hao(
            input_ids             = dummy_ids,
            reward_values         = dummy_rv,
            reward_positions      = dummy_rp,
            select_positions      = dummy_sel,
            update_positions      = dummy_upd,
            thought_positions     = dummy_th,
            continuous_round_mask = dummy_mask,
        )

    print(f"  Smoke test — forward_hao (all discrete):")
    print(f"    input_ids     : {list(dummy_ids.shape)}")
    print(f"    action_logits : {list(al.shape)}  (expect [{B}, {n_rounds}, {config.n_actions}])")
    print(f"    thought_logits: {list(tl.shape)}  (expect [{B}, {n_rounds}, 3, {config.vocab_size}])")

    # Smoke test: mixed curriculum (first round continuous)
    dummy_mask_mixed = torch.zeros(B, n_rounds, dtype=torch.bool)
    dummy_mask_mixed[:, 0] = True
    dummy_th_mixed = dummy_th.clone()
    dummy_th_mixed[:, 0, 1] = -1   # continuous round: only slot 0 valid
    dummy_th_mixed[:, 0, 2] = -1

    with torch.no_grad():
        al2, tl2 = model.forward_hao(
            input_ids             = dummy_ids,
            reward_values         = dummy_rv,
            reward_positions      = dummy_rp,
            select_positions      = dummy_sel,
            update_positions      = dummy_upd,
            thought_positions     = dummy_th_mixed,
            continuous_round_mask = dummy_mask_mixed,
        )
    print(f"  Smoke test — forward_hao (mixed: round 0 continuous):")
    print(f"    action_logits : {list(al2.shape)}")
    print(f"    thought_logits: {list(tl2.shape)}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    cfg = COCONUTConfig(
        n_states=4, n_actions=2,
        n_layers=4, n_heads=8, d_model=256, d_ff=1024,
        dropout=0.1, max_seq_len=1024,
    )
    print_model_summary(cfg)
