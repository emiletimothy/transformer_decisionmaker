#!/usr/bin/env python3
"""
2_model.py — Recurrent Context Q-Learning Transformer

Transformer that processes one transition at a time with |A| continuous
context tokens carrying forward all historical knowledge. The network
outputs action logits at the SELECT position and a new context vector
at the UPDATE position.

Token sequence per step t (prepended context is continuous, not embedded):
  [c_1^(t), ..., c_{|A|}^(t), s_t, a_t, R, s_{t+1},
   s_{t+1} a_1 EVAL, ..., s_{t+1} a_{|A|} EVAL,
   SELECT, QCURR, QNEXT, UPDATE]

Attention constraint: tokens at step t attend only to tokens within step t
and the |A| context tokens. No cross-step attention (enforced by processing
one step at a time).

Recurrence: hidden state at UPDATE replaces c_{a_t} for the next step.

Architecture: Pre-norm Transformer with optional FFN toggle (use_ffns=False
disables MLP layers per the theoretical construction).
"""

import math
from dataclasses import dataclass, asdict
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class COCONUTConfig:
    max_states:  int   = 8
    max_actions: int   = 4
    n_layers:    int   = 4
    n_heads:     int   = 8
    d_model:     int   = 256
    d_ff:        int   = 1024
    dropout:     float = 0.1
    max_seq_len: int   = 128
    use_ffns:    bool  = True

    @property
    def vocab_size(self) -> int:
        return 2 + self.max_states + self.max_actions + 6

    def to_dict(self) -> Dict:
        d = asdict(self)
        d['vocab_size'] = self.vocab_size
        return d

    @classmethod
    def from_dict(cls, d: Dict) -> 'COCONUTConfig':
        keys = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in keys})


# ---------------------------------------------------------------------------
# Vocabulary helpers
# ---------------------------------------------------------------------------

def build_vocab(max_states: int, max_actions: int) -> Dict[str, object]:
    TOK_R = 2 + max_states + max_actions
    return {
        'TOK_NULL':      0,
        'TOK_START':     1,
        'TOK_S':         list(range(2, 2 + max_states)),
        'TOK_A':         list(range(2 + max_states, 2 + max_states + max_actions)),
        'TOK_R':         TOK_R,
        'TOK_EVAL':      TOK_R + 1,
        'TOK_SELECT':    TOK_R + 2,
        'TOK_QCURR':     TOK_R + 3,
        'TOK_QNEXT':     TOK_R + 4,
        'TOK_UPDATE':    TOK_R + 5,
        'vocab_size':    2 + max_states + max_actions + 6,
        'n_actions_max': max_actions,
    }


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class CausalMultiHeadAttention(nn.Module):
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

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        return_attn: bool = False,
    ):
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

        if attn_mask is not None:
            attn = attn.masked_fill(attn_mask.unsqueeze(1), float('-inf'))

        attn_weights = F.softmax(attn, dim=-1)
        attn_out = self.attn_drop(attn_weights)

        out = (attn_out @ v).transpose(1, 2).contiguous().view(B, T, C)
        out = self.resid_drop(self.out_proj(out))
        if return_attn:
            return out, attn_weights
        return out


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
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        use_ffn: bool = True,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn  = CausalMultiHeadAttention(d_model, n_heads, dropout)
        self.use_ffn = use_ffn
        if use_ffn:
            self.norm2 = nn.LayerNorm(d_model)
            self.ff    = FeedForward(d_model, d_ff, dropout)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        return_attn: bool = False,
    ):
        if return_attn:
            attn_out, attn_weights = self.attn(
                self.norm1(x), attn_mask=attn_mask, return_attn=True
            )
            x = x + attn_out
            if self.use_ffn:
                x = x + self.ff(self.norm2(x))
            return x, attn_weights
        x = x + self.attn(self.norm1(x), attn_mask=attn_mask)
        if self.use_ffn:
            x = x + self.ff(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class COCONUTTransformer(nn.Module):
    """
    Recurrent Context Q-Learning Transformer.

    forward_step processes a single transition with continuous context tokens.
    """

    def __init__(self, config: COCONUTConfig):
        super().__init__()
        self.config    = config
        self.d_model   = config.d_model
        self.n_actions = config.max_actions

        self.tok_emb     = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_emb     = nn.Embedding(config.max_seq_len, config.d_model)
        self.reward_proj = nn.Linear(1, config.d_model, bias=False)
        self.emb_drop    = nn.Dropout(config.dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(
                config.d_model, config.n_heads, config.d_ff,
                config.dropout, use_ffn=config.use_ffns,
            )
            for _ in range(config.n_layers)
        ])
        self.final_norm = nn.LayerNorm(config.d_model)

        self.action_head = nn.Linear(config.d_model, config.max_actions)

        self._init_weights()

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

    def get_init_context(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Return fixed zero context tokens: (batch, n_actions, d_model)."""
        return torch.zeros(batch_size, self.n_actions, self.d_model, device=device)

    def _embed_step_tokens(
        self,
        token_ids: torch.Tensor,
        reward_value: torch.Tensor,
        reward_offset: int,
        context_len: int,
    ) -> torch.Tensor:
        """Embed discrete tokens for one step with positional encoding.

        Positions are offset by context_len so context tokens occupy
        positions 0..context_len-1 and discrete tokens start at context_len.
        """
        B, T = token_ids.shape
        device = token_ids.device
        positions = torch.arange(context_len, context_len + T, device=device)
        positions = positions.unsqueeze(0).expand(B, -1)
        x = self.tok_emb(token_ids) + self.pos_emb(positions)

        if reward_value is not None:
            rv = reward_value.unsqueeze(-1).float()  # [B, 1]
            delta = self.reward_proj(rv)             # [B, d_model]
            x[:, reward_offset, :] = x[:, reward_offset, :] + delta

        return x

    def forward_step(
        self,
        token_ids: torch.Tensor,
        reward_value: torch.Tensor,
        reward_offset: int,
        select_offset: int,
        update_offset: int,
        context: torch.Tensor,
        return_attention: bool = False,
    ):
        """Process one transition step with context tokens.

        Parameters
        ----------
        token_ids     : [B, T_disc] discrete token ids for this step
        reward_value  : [B] reward float for this step
        reward_offset : int, position of TOK_R within token_ids
        select_offset : int, position of TOK_SELECT within token_ids
        update_offset : int, position of TOK_UPDATE within token_ids
        context       : [B, n_actions, d_model] continuous context tokens

        Returns
        -------
        select_logits : [B, n_actions]  — action logits at SELECT
        update_hidden : [B, d_model]    — hidden state at UPDATE (new context)
        attn_weights  : optional, list of [B, H, T_full, T_full] per layer
        """
        B = token_ids.shape[0]
        n_ctx = context.shape[1]

        x_disc = self._embed_step_tokens(
            token_ids, reward_value, reward_offset, context_len=n_ctx
        )

        ctx_pos = torch.arange(n_ctx, device=context.device).unsqueeze(0).expand(B, -1)
        context_with_pos = context + self.pos_emb(ctx_pos)

        x = torch.cat([context_with_pos, x_disc], dim=1)  # [B, n_ctx + T_disc, d]
        x = self.emb_drop(x)

        all_attn = []
        for block in self.blocks:
            if return_attention:
                x, attn_w = block(x, return_attn=True)
                all_attn.append(attn_w)
            else:
                x = block(x)

        h = self.final_norm(x)

        sel_idx = n_ctx + select_offset
        upd_idx = n_ctx + update_offset
        select_hidden = h[:, sel_idx, :]
        update_hidden = h[:, upd_idx, :]

        select_logits = self.action_head(select_hidden)

        if return_attention:
            return select_logits, update_hidden, all_attn
        return select_logits, update_hidden

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------

def print_model_summary(config: COCONUTConfig) -> None:
    model = COCONUTTransformer(config)
    total = model.num_parameters()

    print("=" * 60)
    print("COCONUTTransformer — Recurrent Context Summary")
    print("=" * 60)
    print(f"  max_states  : {config.max_states}")
    print(f"  max_actions : {config.max_actions}")
    print(f"  vocab_size  : {config.vocab_size}")
    print(f"  n_layers    : {config.n_layers}")
    print(f"  n_heads     : {config.n_heads}")
    print(f"  d_model     : {config.d_model}")
    print(f"  d_ff        : {config.d_ff}")
    print(f"  use_ffns    : {config.use_ffns}")
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

    B = 2
    n_actions = config.max_actions
    n_disc = 4 + 3 * n_actions + 3  # s,a,R,s' + evals + SELECT,QCURR,QNEXT,UPDATE
    dummy_ids = torch.randint(0, config.vocab_size, (B, n_disc))
    dummy_rv  = torch.rand(B)
    dummy_ctx = torch.randn(B, n_actions, config.d_model)

    model.eval()
    with torch.no_grad():
        sel_logits, upd_hidden = model.forward_step(
            token_ids=dummy_ids,
            reward_value=dummy_rv,
            reward_offset=2,
            select_offset=4 + 3 * n_actions,
            update_offset=n_disc - 1,
            context=dummy_ctx,
        )

    print(f"  Smoke test — forward_step:")
    print(f"    token_ids     : {list(dummy_ids.shape)}")
    print(f"    select_logits : {list(sel_logits.shape)}  (expect [{B}, {n_actions}])")
    print(f"    update_hidden : {list(upd_hidden.shape)}  (expect [{B}, {config.d_model}])")
    print("=" * 60)


if __name__ == '__main__':
    cfg = COCONUTConfig(
        max_states=4, max_actions=2,
        n_layers=4, n_heads=8, d_model=256, d_ff=1024,
        dropout=0.1, max_seq_len=128, use_ffns=True,
    )
    print_model_summary(cfg)

    print("\nSmoke test with use_ffns=False:")
    cfg_no_ffn = COCONUTConfig(
        max_states=4, max_actions=2,
        n_layers=4, n_heads=8, d_model=256, d_ff=1024,
        dropout=0.1, max_seq_len=128, use_ffns=False,
    )
    print_model_summary(cfg_no_ffn)
