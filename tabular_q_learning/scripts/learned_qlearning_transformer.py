"""
Learned Q-Learning Transformer

GPT-2 style transformer that learns to reproduce the tabular Q-learning
algorithm through gradient-based training on Q-learning execution traces.

Includes:
  - LearnedQLearningTransformer: base model with discrete CoT support
  - ContinuousCoTTransformer: Coconut-style continuous hidden-state reasoning
  - QLearningTokenizer: tokenizes Q-learning traces into token sequences
  - QLearningSequenceDataset / collate_fn: PyTorch dataset utilities
  - QLearningTrainer / ContinuousCoTQLearningTrainer: training loops
  - compute_qlearning_qtable: recompute Q-table trajectory for evaluation
  - generate_sequence_with_cot / generate_sequence_with_continuous_cot: inference
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from torch.utils.data import Dataset, DataLoader
import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configs
# ---------------------------------------------------------------------------

@dataclass
class QLModelConfig:
    """Configuration for the learned Q-learning transformer."""
    d_model: int = 768
    n_heads: int = 8
    n_layers: int = 2
    n_states: int = 4
    n_actions: int = 2
    max_sequence_length: int = 512
    vocab_size: int = 1000  # Will be set based on tokenization
    dropout: float = 0.1
    n_thought_steps: int = 4  # Number of continuous thought recurrence steps (0 = disabled)
    n_qvalue_bins: int = 100
    q_value_range: Tuple[float, float] = (0.0, 5.0)

@dataclass
class QLTrainingConfig:
    """Configuration for training the learned Q-learning transformer."""
    learning_rate: float = 1e-4
    weight_decay: float = 1e-2
    batch_size: int = 32
    max_epochs_per_stage: int = 10
    early_stopping_patience: int = 3
    max_stages: int = 12  # Up to 12-step reasoning
    stage_mixing_prob: float = 0.1
    eval_interval: int = 50
    cot_loss_weight: float = 1.0

# ---------------------------------------------------------------------------
# Transformer building blocks
# ---------------------------------------------------------------------------

class MultiHeadAttention(nn.Module):
    """Multi-head self-attention with causal masking."""

    def __init__(self, config: QLModelConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.d_model = config.d_model
        self.head_dim = config.d_model // config.n_heads

        self.q_proj = nn.Linear(config.d_model, config.d_model)
        self.k_proj = nn.Linear(config.d_model, config.d_model)
        self.v_proj = nn.Linear(config.d_model, config.d_model)
        self.out_proj = nn.Linear(config.d_model, config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape

        q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

        return self.out_proj(attn_output), attn_weights


class TransformerBlock(nn.Module):
    """Transformer block with pre-norm architecture."""

    def __init__(self, config: QLModelConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_model)
        self.attn = MultiHeadAttention(config)
        self.ln2 = nn.LayerNorm(config.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(config.d_model, 4 * config.d_model),
            nn.GELU(),
            nn.Linear(4 * config.d_model, config.d_model),
            nn.Dropout(config.dropout),
        )

    def forward(self, x, mask=None):
        attn_out, attn_weights = self.attn(self.ln1(x), mask)
        x = x + attn_out
        x = x + self.mlp(self.ln2(x))
        return x, attn_weights

# ---------------------------------------------------------------------------
# LearnedQLearningTransformer (base model + discrete CoT)
# ---------------------------------------------------------------------------

class LearnedQLearningTransformer(nn.Module):
    """GPT-2 style transformer that learns Q-learning updates."""

    def __init__(self, config: QLModelConfig):
        super().__init__()
        self.config = config

        # Token embeddings
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.max_sequence_length, config.d_model)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layers)
        ])

        self.ln_final = nn.LayerNorm(config.d_model)

        # Output heads
        self.qvalue_head = nn.Linear(config.d_model, config.n_qvalue_bins)  # Predict Q-value bin
        self.regression_head = nn.Linear(config.d_model, 1)  # Optional regression Q-value
        self.lm_head = nn.Linear(config.d_model, config.vocab_size)  # CoT next-token prediction

        self.dropout = nn.Dropout(config.dropout)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize weights following GPT-2 style."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(self, input_ids, position_ids=None, return_attention=False):
        batch_size, seq_len = input_ids.shape

        if position_ids is None:
            position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)

        # Embeddings
        token_emb = self.token_embedding(input_ids)
        pos_emb = self.position_embedding(position_ids)
        x = self.dropout(token_emb + pos_emb)

        # Causal mask
        mask = torch.tril(torch.ones(seq_len, seq_len, device=input_ids.device)).unsqueeze(0).unsqueeze(0)

        # Transformer blocks
        attention_weights = []
        for block in self.blocks:
            x, attn = block(x, mask)
            if return_attention:
                attention_weights.append(attn)

        x = self.ln_final(x)

        # Output predictions
        qvalue_logits = self.qvalue_head(x)       # [batch, seq_len, n_qvalue_bins]
        regression_out = self.regression_head(x)   # [batch, seq_len, 1]
        lm_logits = self.lm_head(x)               # [batch, seq_len, vocab_size]

        outputs = {
            'qvalue_logits': qvalue_logits,
            'regression_out': regression_out,
            'lm_logits': lm_logits,
        }

        if return_attention:
            outputs['attention_weights'] = attention_weights

        return outputs

    @torch.no_grad()
    def generate_cot_qvalues(self, context_ids, tokenizer):
        """
        Autoregressively generate Q-value tokens as chain-of-thought reasoning.

        Given context (transition tokens for a step), generates a QVALUE token
        representing the updated Q(s,a).

        Args:
            context_ids: [batch, seq_len] tensor of context tokens
            tokenizer: QLearningTokenizer instance

        Returns:
            Extended sequence with generated Q-value token appended
        """
        device = context_ids.device
        generated = context_ids.clone()

        qvalue_start = tokenizer.QVALUE_TOKENS[0]
        qvalue_end = tokenizer.QVALUE_TOKENS[-1] + 1

        # Forward pass to predict the Q-value token
        outputs = self.forward(generated)
        next_logits = outputs['lm_logits'][:, -1, :]  # [batch, vocab_size]

        # Restrict to Q-value tokens only
        mask = torch.full_like(next_logits, float('-inf'))
        mask[:, qvalue_start:qvalue_end] = 0.0
        next_logits = next_logits + mask

        # Greedy selection
        next_token = next_logits.argmax(dim=-1, keepdim=True)  # [batch, 1]
        generated = torch.cat([generated, next_token], dim=1)

        return generated


def generate_sequence_with_cot(model, sequence, tokenizer, device):
    """
    Autoregressively generate a full Q-learning sequence using discrete CoT.

    At each step the model sees the transition (s, a, r, s') as context,
    then predicts the updated Q(s,a) value.

    Args:
        model: LearnedQLearningTransformer instance
        sequence: Dict with states, actions, rewards, next_states, params
        tokenizer: QLearningTokenizer instance
        device: torch device

    Returns:
        Dict with q_predictions, generated_ids
    """
    model.eval()

    alpha = sequence['params']['alpha']
    gamma = sequence['params']['gamma']
    n_steps = len(sequence['states'])

    token_ids = [tokenizer.START_TOKEN]
    token_ids.append(tokenizer.discretize_alpha(alpha))
    token_ids.append(tokenizer.discretize_gamma(gamma))

    q_predictions = []

    for step in range(n_steps):
        # Step marker
        token_ids.append(tokenizer.STEP_TOKENS[step % 100])

        # Transition tokens
        token_ids.append(tokenizer.STATE_TOKENS[sequence['states'][step]])
        token_ids.append(tokenizer.ACTION_TOKENS[sequence['actions'][step]])
        token_ids.append(tokenizer.discretize_reward(sequence['rewards'][step]))
        token_ids.append(tokenizer.STATE_TOKENS[sequence['next_states'][step]])

        # Predict Q(s,a) at SEP position (BEFORE appending SEP)
        context_tensor = torch.tensor(
            [token_ids], dtype=torch.long, device=device
        )
        with torch.no_grad():
            outputs = model(context_tensor)
            qvalue_logits = outputs['qvalue_logits'][0, -1, :]  # [n_qvalue_bins]
            predicted_bin = qvalue_logits.argmax().item()
            predicted_q = tokenizer.decode_qvalue_token(
                tokenizer.QVALUE_TOKENS[predicted_bin])
        q_predictions.append(predicted_q)

        # SEP token (marks the prediction point)
        token_ids.append(tokenizer.SEP_TOKEN)

    return {
        'q_predictions': np.array(q_predictions),
        'generated_ids': token_ids,
    }


# ---------------------------------------------------------------------------
# ContinuousCoTTransformer (Coconut-style hidden-state recurrence)
# ---------------------------------------------------------------------------

class ContinuousCoTTransformer(nn.Module):
    """
    Transformer with Coconut-style continuous chain-of-thought reasoning.

    Instead of generating discrete Q-value tokens autoregressively, this model
    performs K recurrence steps in continuous hidden-state space:
      1. Embed the context tokens and run through the transformer.
      2. Take the hidden state at the last position.
      3. Project it back to embedding space via thought_proj.
      4. Append it to the embedding sequence as a new "thought" position.
      5. Repeat steps 1-4 for K thought steps.
      6. Read off Q-value predictions from qvalue_head on the final hidden state.

    This avoids the information bottleneck of discretizing continuous Q-values
    into a finite token vocabulary.
    """

    def __init__(self, config: QLModelConfig):
        super().__init__()
        self.config = config
        self.n_thought_steps = config.n_thought_steps

        # Token embeddings (extra positions for thought steps)
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(
            config.max_sequence_length + config.n_thought_steps + 1,
            config.d_model
        )

        # Transformer blocks (shared across all passes including thought recurrence)
        self.blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layers)
        ])
        self.ln_final = nn.LayerNorm(config.d_model)

        # Output heads
        self.qvalue_head = nn.Linear(config.d_model, config.n_qvalue_bins)
        self.regression_head = nn.Linear(config.d_model, 1)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size)

        # Continuous thought projection: hidden state -> embedding space
        self.thought_proj = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, config.d_model),
        )

        self.dropout = nn.Dropout(config.dropout)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize weights following GPT-2 style."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def _run_blocks(self, x, return_attention=False):
        """Run transformer blocks on positioned embeddings."""
        seq_len = x.size(1)
        mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device))
        mask = mask.unsqueeze(0).unsqueeze(0)

        attention_weights = []
        for block in self.blocks:
            x, attn = block(x, mask)
            if return_attention:
                attention_weights.append(attn)
        x = self.ln_final(x)
        return x, attention_weights

    def forward(self, input_ids, position_ids=None, return_attention=False):
        """Standard forward pass (compatible with existing training code)."""
        batch_size, seq_len = input_ids.shape

        if position_ids is None:
            position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)

        x = self.dropout(
            self.token_embedding(input_ids) + self.position_embedding(position_ids)
        )
        h, attn = self._run_blocks(x, return_attention)

        outputs = {
            'qvalue_logits': self.qvalue_head(h),
            'regression_out': self.regression_head(h),
            'lm_logits': self.lm_head(h),
        }
        if return_attention:
            outputs['attention_weights'] = attn
        return outputs

    def think(self, context_embeddings, return_attention=False):
        """
        Coconut-style continuous thought recurrence.

        Given raw context embeddings [batch, ctx_len, d_model] (without position
        embeddings added yet), performs K recurrence steps:
          - Run transformer on current sequence.
          - Project last hidden state back to embedding space.
          - Append as a new position and repeat.

        Returns:
            qvalue_logits: [batch, n_qvalue_bins] Q-value bin predictions
            thought_hiddens: [batch, K, d_model] hidden states at each thought step
            final_h: [batch, d_model] hidden state from the final refinement pass
        """
        K = self.n_thought_steps
        device = context_embeddings.device
        current = context_embeddings  # [batch, growing_len, d_model]

        thought_hiddens = []
        all_attention = []

        for k in range(K):
            total_len = current.size(1)
            pos_ids = torch.arange(total_len, device=device).unsqueeze(0)
            positioned = current + self.position_embedding(pos_ids)

            h, attn = self._run_blocks(self.dropout(positioned), return_attention)
            if return_attention:
                all_attention.append(attn)
            last_h = h[:, -1:, :]  # [batch, 1, d_model]
            thought_hiddens.append(last_h.squeeze(1))

            # Project back to embedding space for next recurrence
            thought_embed = self.thought_proj(last_h)  # [batch, 1, d_model]
            current = torch.cat([current, thought_embed], dim=1)

        # Final forward with all thoughts appended
        total_len = current.size(1)
        pos_ids = torch.arange(total_len, device=device).unsqueeze(0)
        positioned = current + self.position_embedding(pos_ids)
        h, attn = self._run_blocks(self.dropout(positioned), return_attention)
        if return_attention:
            all_attention.append(attn)

        final_h = h[:, -1, :]  # [batch, d_model]
        qvalue_logits = self.qvalue_head(final_h)  # [batch, n_qvalue_bins]

        result = (qvalue_logits, torch.stack(thought_hiddens, dim=1), final_h)
        if return_attention:
            return result + (all_attention,)
        return result

    def think_and_predict(self, context_ids, return_attention=False):
        """
        Convenience method: embed context token ids, run thought recurrence,
        return Q-value predictions.

        Both Q-value logits and regression output use final_h from the last
        refinement pass, which has seen all K thought embeddings.
        """
        embeddings = self.token_embedding(context_ids)  # no position yet; think() adds them

        if return_attention:
            qvalue_logits, thought_hiddens, final_h, all_attention = self.think(
                embeddings, return_attention=True)
            regression_out = self.regression_head(final_h)  # [batch, 1]
            return qvalue_logits, regression_out, all_attention
        else:
            qvalue_logits, thought_hiddens, final_h = self.think(embeddings)
            regression_out = self.regression_head(final_h)  # [batch, 1]
            return qvalue_logits, regression_out


def generate_sequence_with_continuous_cot(model, sequence, tokenizer, device):
    """
    Generate a full Q-learning sequence using continuous chain-of-thought reasoning.

    At each step the model receives the transition as discrete context tokens,
    then performs K continuous thought recurrences in hidden space to produce
    Q-value predictions (no discretization bottleneck).

    Args:
        model: ContinuousCoTTransformer instance
        sequence: Dict with states, actions, rewards, next_states, params
        tokenizer: QLearningTokenizer instance
        device: torch device

    Returns:
        Dict with q_predictions (continuous)
    """
    model.eval()

    alpha = sequence['params']['alpha']
    gamma = sequence['params']['gamma']
    n_steps = len(sequence['states'])

    token_ids = [tokenizer.START_TOKEN]
    token_ids.append(tokenizer.discretize_alpha(alpha))
    token_ids.append(tokenizer.discretize_gamma(gamma))

    q_predictions = []

    for step in range(n_steps):
        # Step marker
        token_ids.append(tokenizer.STEP_TOKENS[step % 100])

        # Transition tokens
        token_ids.append(tokenizer.STATE_TOKENS[sequence['states'][step]])
        token_ids.append(tokenizer.ACTION_TOKENS[sequence['actions'][step]])
        token_ids.append(tokenizer.discretize_reward(sequence['rewards'][step]))
        token_ids.append(tokenizer.STATE_TOKENS[sequence['next_states'][step]])

        # Continuous thought: get Q-value prediction (BEFORE appending SEP)
        context_tensor = torch.tensor(
            [token_ids], dtype=torch.long, device=device
        )
        with torch.no_grad():
            qvalue_logits, regression_out = model.think_and_predict(context_tensor)

        # Use classification head (argmax over bins)
        predicted_bin = qvalue_logits[0].argmax().item()
        predicted_q = tokenizer.decode_qvalue_token(
            tokenizer.QVALUE_TOKENS[predicted_bin])
        q_predictions.append(predicted_q)

        # SEP token
        token_ids.append(tokenizer.SEP_TOKEN)

    return {
        'q_predictions': np.array(q_predictions),
    }

# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

class QLearningTokenizer:
    """Tokenizer for Q-learning execution trace sequences.

    Vocabulary layout:
        PAD=0, START=1, END=2, SEP=3
        STATE_TOKENS:  [4, ..., 4+n_states-1]
        ACTION_TOKENS: [4+n_states, ..., 4+n_states+n_actions-1]
        REWARD_TOKENS: 100 bins for r in [0, 1]
        QVALUE_TOKENS: 100 bins for Q in [q_min, q_max]
        ALPHA_TOKENS:  100 bins for alpha in [0, 1]
        GAMMA_TOKENS:  100 bins for gamma in [0, 1]
        STEP_TOKENS:   100 step markers

    Per-step token sequence:
        START -> ALPHA(a) -> GAMMA(g) ->
          STEP_0 -> STATE(s) -> ACTION(a) -> REWARD(r) -> STATE(s') -> SEP ->
          STEP_1 -> ... -> SEP ->
          ...
        END
    """

    def __init__(self, n_states: int = 4, n_actions: int = 2,
                 n_qvalue_bins: int = 100,
                 q_value_range: Tuple[float, float] = (0.0, 5.0)):
        self.n_states = n_states
        self.n_actions = n_actions
        self.n_qvalue_bins = n_qvalue_bins
        self.q_min, self.q_max = q_value_range

        self.PAD_TOKEN = 0
        self.START_TOKEN = 1
        self.END_TOKEN = 2
        self.SEP_TOKEN = 3

        base = 4
        self.STATE_TOKENS = list(range(base, base + n_states))
        base += n_states
        self.ACTION_TOKENS = list(range(base, base + n_actions))
        base += n_actions
        self.REWARD_TOKENS = list(range(base, base + 100))
        base += 100
        self.QVALUE_TOKENS = list(range(base, base + n_qvalue_bins))
        base += n_qvalue_bins
        self.ALPHA_TOKENS = list(range(base, base + 100))
        base += 100
        self.GAMMA_TOKENS = list(range(base, base + 100))
        base += 100
        self.STEP_TOKENS = list(range(base, base + 100))
        base += 100
        self.vocab_size = base

    def discretize_reward(self, reward: float) -> int:
        """Convert a reward value [0, 1] to a reward token."""
        bin_idx = int(np.clip(reward * 99, 0, 99))
        return self.REWARD_TOKENS[bin_idx]

    def discretize_qvalue(self, q: float) -> int:
        """Convert a Q-value to a Q-value token."""
        # Map [q_min, q_max] -> [0, n_bins-1]
        normalized = (q - self.q_min) / (self.q_max - self.q_min)
        bin_idx = int(np.clip(normalized * (self.n_qvalue_bins - 1), 0, self.n_qvalue_bins - 1))
        return self.QVALUE_TOKENS[bin_idx]

    def decode_qvalue_token(self, token: int) -> float:
        """Convert a Q-value token back to a float."""
        if token < self.QVALUE_TOKENS[0] or token > self.QVALUE_TOKENS[-1]:
            return 0.0
        bin_idx = token - self.QVALUE_TOKENS[0]
        normalized = bin_idx / (self.n_qvalue_bins - 1)
        return self.q_min + normalized * (self.q_max - self.q_min)

    def qvalue_to_bin_index(self, q: float) -> int:
        """Convert a Q-value to a bin index (0-based, for cross-entropy targets)."""
        normalized = (q - self.q_min) / (self.q_max - self.q_min)
        return int(np.clip(normalized * (self.n_qvalue_bins - 1), 0, self.n_qvalue_bins - 1))

    def discretize_alpha(self, alpha: float) -> int:
        """Convert alpha [0, 1] to an alpha token."""
        bin_idx = int(np.clip(alpha * 99, 0, 99))
        return self.ALPHA_TOKENS[bin_idx]

    def discretize_gamma(self, gamma: float) -> int:
        """Convert gamma [0, 1] to a gamma token."""
        bin_idx = int(np.clip(gamma * 99, 0, 99))
        return self.GAMMA_TOKENS[bin_idx]

    def encode_sequence(self, sequence: Dict) -> Dict:
        """
        Encode a full Q-learning execution trace into token ids with targets and masks.

        Args:
            sequence: Dict with keys:
                states, actions, rewards, next_states: lists of length n_steps
                q_values: list of Q-table snapshots (n_steps x n_states x n_actions)
                params: dict with alpha, gamma, epsilon

        Returns dict with:
            input_ids: list of token IDs
            q_targets: array of target Q-value bin indices at each position
            q_target_values: array of target Q-values (float) at each position
            target_mask: list of bools (True at SEP positions)
            is_cot_token: list of bools (for future CoT support)
        """
        n_steps = len(sequence['states'])
        alpha = sequence['params']['alpha']
        gamma = sequence['params']['gamma']

        tokens = [self.START_TOKEN]
        q_targets = [0]          # bin index for cross-entropy
        q_target_values = [0.0]  # float for regression
        target_mask = [False]
        is_cot_token = [False]

        def _append(tok, target_bin=0, target_val=0.0, is_target=False, is_cot=False):
            tokens.append(tok)
            q_targets.append(target_bin)
            q_target_values.append(target_val)
            target_mask.append(is_target)
            is_cot_token.append(is_cot)

        # Prefix: alpha and gamma
        _append(self.discretize_alpha(alpha))
        _append(self.discretize_gamma(gamma))

        for step in range(n_steps):
            s = sequence['states'][step]
            a = sequence['actions'][step]
            r = sequence['rewards'][step]
            s_next = sequence['next_states'][step]

            # The Q-table after this step's update
            q_table_after = sequence['q_values'][step]
            q_updated = q_table_after[s][a]
            q_bin = self.qvalue_to_bin_index(q_updated)

            # Step token
            _append(self.STEP_TOKENS[step % 100])
            # State
            _append(self.STATE_TOKENS[s])
            # Action
            _append(self.ACTION_TOKENS[a])
            # Reward
            _append(self.discretize_reward(r))
            # Next state
            _append(self.STATE_TOKENS[s_next])
            # SEP: model predicts Q(s, a) here
            _append(self.SEP_TOKEN, target_bin=q_bin, target_val=q_updated, is_target=True)

        _append(self.END_TOKEN)

        return {
            'input_ids': tokens,
            'q_targets': np.array(q_targets, dtype=np.int64),
            'q_target_values': np.array(q_target_values, dtype=np.float32),
            'target_mask': target_mask,
            'is_cot_token': is_cot_token,
        }

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class QLearningSequenceDataset(Dataset):
    """Dataset for Q-learning sequences."""

    def __init__(self, sequences: List[Dict], tokenizer: QLearningTokenizer):
        self.sequences = sequences
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        tokens = self.tokenizer.encode_sequence(sequence)
        return {
            'input_ids': torch.tensor(tokens['input_ids'], dtype=torch.long),
            'q_targets': torch.tensor(tokens['q_targets'], dtype=torch.long),
            'q_target_values': torch.tensor(tokens['q_target_values'], dtype=torch.float32),
            'target_mask': torch.tensor(tokens['target_mask'], dtype=torch.bool),
            'cot_mask': torch.tensor(tokens['is_cot_token'], dtype=torch.bool),
            'sequence_length': len(tokens['input_ids']),
        }


def collate_fn(batch):
    """Custom collate function for variable length sequences."""
    max_len = max(item['sequence_length'] for item in batch)

    input_ids = []
    q_targets = []
    q_target_values = []
    target_mask = []
    cot_mask = []

    for item in batch:
        seq_len = item['sequence_length']
        pad_len = max_len - seq_len

        padded_input = torch.cat([
            item['input_ids'],
            torch.zeros(pad_len, dtype=torch.long)
        ])
        input_ids.append(padded_input)

        padded_q_targets = torch.cat([
            item['q_targets'],
            torch.zeros(pad_len, dtype=torch.long)
        ])
        q_targets.append(padded_q_targets)

        padded_q_target_values = torch.cat([
            item['q_target_values'],
            torch.zeros(pad_len, dtype=torch.float32)
        ])
        q_target_values.append(padded_q_target_values)

        padded_mask = torch.cat([
            item['target_mask'],
            torch.zeros(pad_len, dtype=torch.bool)
        ])
        target_mask.append(padded_mask)

        padded_cot_mask = torch.cat([
            item['cot_mask'],
            torch.zeros(pad_len, dtype=torch.bool)
        ])
        cot_mask.append(padded_cot_mask)

    return {
        'input_ids': torch.stack(input_ids),
        'q_targets': torch.stack(q_targets),
        'q_target_values': torch.stack(q_target_values),
        'target_mask': torch.stack(target_mask),
        'cot_mask': torch.stack(cot_mask),
    }

# ---------------------------------------------------------------------------
# Training data utilities
# ---------------------------------------------------------------------------

def compute_qlearning_qtable(sequence: Dict, n_states: int = 4,
                             n_actions: int = 2) -> List[np.ndarray]:
    """
    Recompute Q-table trajectory from a sequence (for evaluation only).

    Returns list of Q-table snapshots: [after_step_0, after_step_1, ...].
    Length = n_steps.
    """
    alpha = sequence['params']['alpha']
    gamma = sequence['params']['gamma']

    Q = np.zeros((n_states, n_actions))
    snapshots = []

    for step in range(len(sequence['states'])):
        s = sequence['states'][step]
        a = sequence['actions'][step]
        r = sequence['rewards'][step]
        s_next = sequence['next_states'][step]

        max_q_next = np.max(Q[s_next])
        Q[s, a] = (1 - alpha) * Q[s, a] + alpha * (r + gamma * max_q_next)
        snapshots.append(Q.copy())

    return snapshots

# ---------------------------------------------------------------------------
# Trainers
# ---------------------------------------------------------------------------

class QLearningTrainer:
    """Trainer for the learned Q-learning transformer."""

    def __init__(self, model: LearnedQLearningTransformer,
                 train_config: QLTrainingConfig,
                 model_config: QLModelConfig):
        self.model = model
        self.train_config = train_config
        self.model_config = model_config

        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=train_config.learning_rate,
            weight_decay=train_config.weight_decay,
            betas=(0.9, 0.95)
        )

        # Loss: cross-entropy over Q-value bins at masked positions
        self.ce_loss_fn = nn.CrossEntropyLoss()
        self.device = next(model.parameters()).device

        self.step = 0
        self.training_history = []

    def train_stage(self, stage: int, train_loader: DataLoader,
                    val_loader: DataLoader = None) -> Dict:
        """Train for one stage of the multi-stage curriculum."""
        logger.info(f"Training stage {stage}")

        stage_losses = []

        for epoch in range(self.train_config.max_epochs_per_stage):
            epoch_loss = 0.0
            n_batches = 0

            self.model.train()
            for batch in train_loader:
                loss = self._train_batch(batch, stage)
                epoch_loss += loss
                n_batches += 1

                if self.step % self.train_config.eval_interval == 0 and val_loader:
                    val_metrics = self._evaluate(val_loader)
                    logger.info(f"Step {self.step}: Val loss = {val_metrics['loss']:.4f}")

                self.step += 1

            avg_loss = epoch_loss / max(n_batches, 1)
            stage_losses.append(avg_loss)
            logger.info(f"Stage {stage}, Epoch {epoch}: Loss = {avg_loss:.4f}")

        return {'losses': stage_losses}

    def _train_batch(self, batch: Dict, stage: int) -> float:
        """Train on a single batch."""
        self.optimizer.zero_grad()

        input_ids = batch['input_ids'].to(self.device)
        q_targets = batch['q_targets'].to(self.device)
        target_mask = batch['target_mask'].to(self.device)

        outputs = self.model(input_ids)
        batch_size = target_mask.shape[0]

        # Cross-entropy loss on Q-value bin predictions at masked positions
        losses = []
        for i in range(batch_size):
            sample_mask = target_mask[i]
            if sample_mask.any():
                logits = outputs['qvalue_logits'][i]  # [seq_len, n_qvalue_bins]
                targets = q_targets[i]                 # [seq_len]
                ml = min(len(sample_mask), logits.shape[0], len(targets))
                sample_mask = sample_mask[:ml]
                logits = logits[:ml]
                targets = targets[:ml]
                if sample_mask.any():
                    masked_logits = logits[sample_mask]   # [n_masked, n_qvalue_bins]
                    masked_targets = targets[sample_mask]  # [n_masked]
                    if len(masked_logits) > 0:
                        losses.append(self.ce_loss_fn(masked_logits, masked_targets))

        if losses:
            total_loss = torch.stack(losses).mean()
        else:
            return 0.0

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        return total_loss.item()

    def _evaluate(self, val_loader: DataLoader) -> Dict:
        """Evaluate on validation data."""
        self.model.eval()
        total_loss = 0.0
        n_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(self.device)
                q_targets = batch['q_targets'].to(self.device)
                target_mask = batch['target_mask'].to(self.device)

                outputs = self.model(input_ids)
                batch_size = target_mask.shape[0]

                losses = []
                for i in range(batch_size):
                    sm = target_mask[i]
                    if sm.any():
                        logits = outputs['qvalue_logits'][i]
                        targets = q_targets[i]
                        ml = min(len(sm), logits.shape[0], len(targets))
                        sm = sm[:ml]; logits = logits[:ml]; targets = targets[:ml]
                        if sm.any():
                            ml_logits = logits[sm]; ml_targets = targets[sm]
                            if len(ml_logits) > 0:
                                losses.append(self.ce_loss_fn(ml_logits, ml_targets))

                if losses:
                    batch_loss = torch.stack(losses).mean().item()
                else:
                    batch_loss = 0.0

                total_loss += batch_loss
                n_batches += 1

        return {'loss': total_loss / n_batches if n_batches > 0 else 0.0}


class ContinuousCoTQLearningTrainer:
    """
    Trainer for ContinuousCoTTransformer with Coconut-style multi-stage curriculum.

    At each training stage i, the model uses min(i, K_max) continuous thought
    recurrences before each prediction. This teaches the model to progressively
    use more "thinking" steps, learning to maintain and update Q-values
    in hidden space rather than through explicit tokens.

    At each decision point (SEP position), the trainer:
      1. Extracts context tokens up to that point.
      2. Runs think_and_predict with the current stage's thought depth.
      3. Computes cross-entropy loss on the Q-value bin prediction.
    """

    def __init__(self, model: ContinuousCoTTransformer,
                 train_config: QLTrainingConfig,
                 model_config: QLModelConfig):
        self.model = model
        self.train_config = train_config
        self.model_config = model_config

        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=train_config.learning_rate,
            weight_decay=train_config.weight_decay,
            betas=(0.9, 0.95)
        )

        self.ce_loss_fn = nn.CrossEntropyLoss()
        self.device = next(model.parameters()).device

        self.step = 0
        self.training_history = []

    def train_stage(self, stage: int, train_loader: DataLoader,
                    val_loader: DataLoader = None) -> Dict:
        """Train for one stage of the Coconut-style curriculum.

        Sets thought depth = min(stage, K_max) for this stage.
        Uses early stopping if val loss doesn't improve for `patience` epochs.
        """
        n_thoughts = min(stage, self.model.config.n_thought_steps)
        self.model.n_thought_steps = n_thoughts
        logger.info(f"Training stage {stage} with {n_thoughts} thought steps")

        stage_losses = []
        best_val_loss = float('inf')
        patience_counter = 0
        patience = self.train_config.early_stopping_patience

        for epoch in range(self.train_config.max_epochs_per_stage):
            epoch_loss = 0.0
            n_batches = 0

            self.model.train()
            for batch in train_loader:
                loss = self._train_batch(batch, stage)
                epoch_loss += loss
                n_batches += 1

                # Step-level wandb logging
                try:
                    import wandb
                    if wandb.run is not None:
                        wandb.log({'train/step_loss': loss, 'global_step': self.step})
                except (ImportError, Exception):
                    pass

                if self.step % self.train_config.eval_interval == 0 and val_loader:
                    val_metrics = self._evaluate(val_loader)
                    logger.info(f"Step {self.step}: Val loss = {val_metrics['loss']:.4f}")
                    try:
                        import wandb
                        if wandb.run is not None:
                            wandb.log({'val/step_loss': val_metrics['loss'], 'global_step': self.step})
                    except (ImportError, Exception):
                        pass

                self.step += 1

            avg_loss = epoch_loss / max(n_batches, 1)
            stage_losses.append(avg_loss)
            logger.info(f"Stage {stage}, Epoch {epoch}: Loss = {avg_loss:.4f}")
            try:
                import wandb
                if wandb.run is not None:
                    wandb.log({'train/epoch_loss': avg_loss, 'stage': stage,
                               'epoch': epoch, 'global_step': self.step})
            except (ImportError, Exception):
                pass

            # Early stopping check
            if val_loader:
                val_metrics = self._evaluate(val_loader)
                val_loss = val_metrics['loss']
                if val_loss < best_val_loss - 1e-4:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch} "
                                f"(no improvement for {patience} epochs)")
                    break

        return {'losses': stage_losses}

    def _train_batch(self, batch: Dict, stage: int) -> float:
        """Train on a single batch using think_and_predict at each decision point."""
        self.optimizer.zero_grad()

        input_ids = batch['input_ids'].to(self.device)
        q_targets = batch['q_targets'].to(self.device)
        target_mask = batch['target_mask'].to(self.device)

        batch_size = input_ids.shape[0]
        total_loss = 0.0
        n_decisions = 0

        for i in range(batch_size):
            seq_ids = input_ids[i]
            seq_mask = target_mask[i]
            seq_targets = q_targets[i]

            # Find decision positions (where target_mask is True)
            decision_pos = seq_mask.nonzero(as_tuple=True)[0]

            for pos in decision_pos:
                # Context: all tokens before the decision position (SEP)
                context = seq_ids[:pos].unsqueeze(0)  # [1, ctx_len]

                if context.shape[1] < 2:
                    continue

                # Run thought recurrence and get Q-value prediction
                qvalue_logits, _ = self.model.think_and_predict(context)

                # Cross-entropy loss on the Q-value bin
                target = seq_targets[pos].unsqueeze(0)  # [1]
                loss = self.ce_loss_fn(qvalue_logits, target)
                total_loss = total_loss + loss
                n_decisions += 1

        if n_decisions > 0:
            avg_loss = total_loss / n_decisions
            avg_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            return avg_loss.item()
        return 0.0

    def _evaluate(self, val_loader: DataLoader) -> Dict:
        """Evaluate on validation data using think_and_predict."""
        self.model.eval()
        total_loss = 0.0
        n_decisions = 0

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(self.device)
                q_targets = batch['q_targets'].to(self.device)
                target_mask = batch['target_mask'].to(self.device)

                batch_size = input_ids.shape[0]

                for i in range(batch_size):
                    seq_ids = input_ids[i]
                    seq_mask = target_mask[i]
                    seq_targets = q_targets[i]

                    decision_pos = seq_mask.nonzero(as_tuple=True)[0]

                    for pos in decision_pos:
                        context = seq_ids[:pos].unsqueeze(0)
                        if context.shape[1] < 2:
                            continue

                        qvalue_logits, _ = self.model.think_and_predict(context)
                        target = seq_targets[pos].unsqueeze(0)
                        loss = self.ce_loss_fn(qvalue_logits, target)
                        total_loss += loss.item()
                        n_decisions += 1

        return {'loss': total_loss / n_decisions if n_decisions > 0 else 0.0}
