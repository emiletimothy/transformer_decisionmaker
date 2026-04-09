#!/usr/bin/env python3
"""
2_model.py — Hybrid Continuous CoT Transformer

Custom GPT-2 style Transformer with two output heads:
    action_head   : Linear(d_model, n_actions)  at <Select> positions (CE loss)
    q_value_head  : Linear(d_model, n_actions)  at <Update> positions (MSE loss)

The <Update> token's hidden state is projected to a continuous vector of size
n_actions, representing the entire Q-row Q[s_t, :] as a superposition vector —
NO discretization.

Reward scalars are injected into the embedding of each TOK_R token via a learned
linear projection `reward_proj : R^1 -> R^d_model`, making the reward value
accessible to the transformer as a continuous signal.

Vocabulary layout (matches 1_generate_data.py):
    vocab_size = 2 + n_states + n_actions + 6

Architecture:
    - Token embedding    : nn.Embedding(vocab_size, d_model)
    - Position embedding : nn.Embedding(max_seq_len, d_model)
    - Reward projection  : nn.Linear(1, d_model, bias=False)
    - 4 x TransformerBlock (pre-norm, causal MHA + FFN with GELU)
    - Final LayerNorm
    - action_head   : nn.Linear(d_model, n_actions)
    - q_value_head  : nn.Linear(d_model, n_actions)
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
    n_states:    int   = 5
    n_actions:   int   = 2
    n_layers:    int   = 4
    n_heads:     int   = 4
    d_model:     int   = 128
    d_ff:        int   = 512      # feed-forward hidden dim (4 * d_model)
    dropout:     float = 0.1
    max_seq_len: int   = 1024

    @property
    def vocab_size(self) -> int:
        return 2 + self.n_states + self.n_actions + 7

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

def build_vocab(n_states: int, n_actions: int) -> Dict[str, object]:
    """Return token ID constants for the COCONUT vocabulary."""
    TOK_R = 2 + n_states + n_actions
    return {
        'TOK_NULL':   0,
        'TOK_START':  1,
        'TOK_S':      list(range(2, 2 + n_states)),
        'TOK_A':      list(range(2 + n_states, 2 + n_states + n_actions)),
        'TOK_R':      TOK_R,
        'TOK_EVAL':   TOK_R + 1,
        'TOK_SELECT': TOK_R + 2,
        'TOK_QCURR':  TOK_R + 3,
        'TOK_QNEXT':  TOK_R + 4,
        'TOK_UPDATE': TOK_R + 5,
        'TOK_COT':    TOK_R + 6,
        'vocab_size': 2 + n_states + n_actions + 7,
    }


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
    5. action_head  : hidden[select_positions]  -> [B, n_sel, n_actions] logits
    6. q_value_head : hidden[update_positions]  -> [B, n_upd, n_actions] Q-values

    Forward inputs
    --------------
    input_ids        : [B, T]       long
    reward_values    : [B, n_r]     float  (actual reward scalars, NOT token IDs)
    reward_positions : [B, n_r]     long   (positions of TOK_R in input_ids; -1 = pad)
    select_positions : [B, n_sel]   long   (positions of TOK_SELECT; -1 = pad)
    update_positions : [B, n_upd]   long   (positions of TOK_UPDATE; -1 = pad)

    Forward outputs
    ---------------
    action_logits : [B, n_sel, n_actions]   (raw logits for CE loss)
    q_value_preds : [B, n_upd, n_actions]   (continuous Q-row predictions for MSE loss)

    Padding convention
    ------------------
    All position tensors use -1 as pad sentinel.  The model clamps negative
    positions to 0 before gathering (safe because padding is masked out in
    the loss), and returns zeros for padded slots.
    """

    def __init__(self, config: COCONUTConfig):
        super().__init__()
        self.config   = config
        self.d_model  = config.d_model
        self.n_actions = config.n_actions

        vocab = build_vocab(config.n_states, config.n_actions)
        self.tok_r_id = vocab['TOK_R']

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

        # Dual output heads
        self.action_head   = nn.Linear(config.d_model, config.n_actions)
        self.q_value_head  = nn.Linear(config.d_model, config.n_actions)

        self._init_weights()

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
    # Core building pass: embeddings -> transformer blocks
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
        # reward_values : [B, n_r],  reward_positions : [B, n_r]
        if reward_values is not None and reward_positions is not None:
            n_r = reward_values.shape[1]
            if n_r > 0:
                # Project scalar rewards -> d_model delta vectors
                rv = reward_values.unsqueeze(-1).float()        # [B, n_r, 1]
                deltas = self.reward_proj(rv)                   # [B, n_r, d]

                # Clamp padding positions to 0 (will be masked in loss anyway)
                safe_pos = reward_positions.clamp(min=0)        # [B, n_r]

                # Expand positions for scatter: [B, n_r, d]
                pos_idx = safe_pos.unsqueeze(-1).expand(-1, -1, self.d_model)

                # Create padding mask: True where position is valid (>= 0)
                valid = (reward_positions >= 0).unsqueeze(-1).expand_as(deltas)
                deltas = deltas * valid.float()

                # Add delta to the corresponding positions
                x = x.scatter_add(1, pos_idx, deltas)

        return self.emb_drop(x)

    def _run_transformer(self, x: torch.Tensor) -> torch.Tensor:
        """Run all transformer blocks + final norm."""
        for block in self.blocks:
            x = block(x)
        return self.final_norm(x)

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
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        input_ids:        torch.Tensor,            # [B, T]
        reward_values:    Optional[torch.Tensor],  # [B, n_r]
        reward_positions: Optional[torch.Tensor],  # [B, n_r]
        select_positions: torch.Tensor,            # [B, n_sel]
        update_positions: torch.Tensor,            # [B, n_upd]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        action_logits : [B, n_sel, n_actions]
        q_value_preds : [B, n_upd, n_actions]
        """
        x = self._embed(input_ids, reward_values, reward_positions)
        h = self._run_transformer(x)   # [B, T, d_model]

        # --- Action logits at <Select> positions ---
        sel_hidden    = self._gather_at_positions(h, select_positions)  # [B, n_sel, d]
        action_logits = self.action_head(sel_hidden)                     # [B, n_sel, n_actions]

        # --- Q-value predictions at <Update> positions ---
        upd_hidden    = self._gather_at_positions(h, update_positions)  # [B, n_upd, d]
        q_value_preds = self.q_value_head(upd_hidden)                   # [B, n_upd, n_actions]

        return action_logits, q_value_preds

    # ------------------------------------------------------------------
    # COCONUT forward (with continuous thought feedback)
    # ------------------------------------------------------------------

    def forward_coconut(
        self,
        input_ids:        torch.Tensor,            # [B, T]
        reward_values:    Optional[torch.Tensor],  # [B, n_r]
        reward_positions: Optional[torch.Tensor],  # [B, n_r]
        select_positions: torch.Tensor,            # [B, n_sel]
        update_positions: torch.Tensor,            # [B, n_upd]
        cot_positions:    torch.Tensor,            # [B, n_cot]  (-1 = pad)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """COCONUT forward pass with continuous thought feedback.

        After each round's TOK_UPDATE, the transformer hidden state at that
        position is injected (detached) as the embedding for the following
        TOK_COT token before the next round is processed.

        The sequential loop ensures causality: round r's CoT embedding is
        derived from the hidden state of round r's Update token, which the
        model for round r+1 can then attend to via the standard causal mask.

        Returns
        -------
        action_logits : [B, n_sel, n_actions]
        q_value_preds : [B, n_upd, n_actions]
        """
        B, T = input_ids.shape
        S = cot_positions.shape[1]

        # Step 1: Compute all token + position + reward embeddings.
        # CoT positions initially hold the learned TOK_COT token embedding.
        # We will overwrite them with transformer hidden states in the loop.
        x = self._embed(input_ids, reward_values, reward_positions).clone()  # [B, T, d]

        # Step 2: Sequential round loop — fill CoT embeddings causally.
        for r in range(S):
            upd_pos = update_positions[:, r]   # [B]
            cot_pos = cot_positions[:, r]       # [B]
            valid   = (cot_pos >= 0)            # [B] bool — False for padding rounds

            if not valid.any():
                break  # all remaining rounds are padding

            # Run transformer on prefix up to (including) the furthest update pos.
            # Causal mask is generated dynamically in CausalMultiHeadAttention
            # from seq_len, so variable-length prefixes work correctly.
            max_prefix = int(upd_pos[valid].max().item()) + 1
            h_prefix = self._run_transformer(x[:, :max_prefix, :])  # [B, max_prefix, d]

            # Inject hidden state at each sample's update_pos into its cot_pos.
            # detach() stops gradients from flowing through the recurrence, which
            # would otherwise create an O(R)-deep computational graph.
            for b in range(B):
                if not valid[b]:
                    continue
                u = int(upd_pos[b].item())
                c = int(cot_pos[b].item())
                x[b, c, :] = h_prefix[b, u, :].detach()

        # Step 3: Final transformer pass on full sequence with CoT embeddings injected.
        h = self._run_transformer(x)  # [B, T, d]

        # Step 4: Extract outputs at select/update positions (same as forward()).
        sel_hidden    = self._gather_at_positions(h, select_positions)  # [B, n_sel, d]
        action_logits = self.action_head(sel_hidden)                     # [B, n_sel, n_actions]

        upd_hidden    = self._gather_at_positions(h, update_positions)  # [B, n_upd, d]
        q_value_preds = self.q_value_head(upd_hidden)                   # [B, n_upd, n_actions]

        return action_logits, q_value_preds

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
    B, T, n_r, n_sel, n_upd = 2, 64, 5, 5, 5
    dummy_ids   = torch.randint(0, config.vocab_size, (B, T))
    dummy_rv    = torch.rand(B, n_r)
    dummy_rp    = torch.randint(0, T, (B, n_r))
    dummy_sel   = torch.randint(0, T, (B, n_sel))
    dummy_upd   = torch.randint(0, T, (B, n_upd))

    model.eval()
    with torch.no_grad():
        al, qv = model(dummy_ids, dummy_rv, dummy_rp, dummy_sel, dummy_upd)
    print(f"  Smoke test forward pass:")
    print(f"    input_ids     : {list(dummy_ids.shape)}")
    print(f"    action_logits : {list(al.shape)}   (expect [B={B}, n_sel={n_sel}, n_actions={config.n_actions}])")
    print(f"    q_value_preds : {list(qv.shape)}   (expect [B={B}, n_upd={n_upd}, n_actions={config.n_actions}])")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Main (prints model summary when run directly)
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    cfg = COCONUTConfig(
        n_states=5, n_actions=2,
        n_layers=4, n_heads=4, d_model=128, d_ff=512,
        dropout=0.1, max_seq_len=1024,
    )
    print_model_summary(cfg)
