"""
Learned Multiplicative Weights Transformer

GPT-2 style transformer that learns to reproduce the multiplicative weights
algorithm through gradient-based training on MW execution traces.

Includes:
  - LearnedMWTransformer: base model with discrete CoT support
  - ContinuousCoTTransformer: Coconut-style continuous hidden-state reasoning
  - MWTokenizer: tokenizes MW traces into token sequences
  - MWSequenceDataset / collate_fn: PyTorch dataset utilities
  - MWTrainer / ContinuousCoTTrainer: training loops
  - generate_mw_training_data: synthetic data generation
  - generate_sequence_with_cot / generate_sequence_with_continuous_cot: inference
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from torch.utils.data import Dataset, DataLoader
import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configs
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    """Configuration for the learned MW transformer."""
    d_model: int = 768
    n_heads: int = 8
    n_layers: int = 2
    n_experts: int = 4
    max_sequence_length: int = 512
    vocab_size: int = 1000  # Will be set based on tokenization
    dropout: float = 0.1
    n_thought_steps: int = 4  # Number of continuous thought recurrence steps (0 = disabled)

@dataclass
class TrainingConfig:
    """Configuration for training the learned MW transformer."""
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
    
    def __init__(self, config: ModelConfig):
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
    
    def __init__(self, config: ModelConfig):
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
# LearnedMWTransformer (base model + discrete CoT)
# ---------------------------------------------------------------------------

class LearnedMWTransformer(nn.Module):
    """GPT-2 style transformer that learns multiplicative weights updates."""
    
    def __init__(self, config: ModelConfig):
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
        
        # Output heads for different prediction tasks
        self.weight_head = nn.Linear(config.d_model, config.n_experts)  # Predict next weights
        self.prediction_head = nn.Linear(config.d_model, 1)  # Predict binary decision
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
        
        # Causal mask for autoregressive generation
        mask = torch.tril(torch.ones(seq_len, seq_len, device=input_ids.device)).unsqueeze(0).unsqueeze(0)
        
        # Transformer blocks
        attention_weights = []
        for block in self.blocks:
            x, attn = block(x, mask)
            if return_attention:
                attention_weights.append(attn)
        
        x = self.ln_final(x)
        
        # Output predictions
        weight_logits = self.weight_head(x)  # [batch, seq_len, n_experts]
        prediction_logits = self.prediction_head(x)  # [batch, seq_len, 1]
        lm_logits = self.lm_head(x)  # [batch, seq_len, vocab_size]
        
        outputs = {
            'weight_logits': weight_logits,
            'prediction_logits': prediction_logits,
            'lm_logits': lm_logits,
        }
        
        if return_attention:
            outputs['attention_weights'] = attention_weights
            
        return outputs

    @torch.no_grad()
    def generate_cot_weights(self, context_ids, tokenizer):
        """
        Autoregressively generate weight tokens as chain-of-thought reasoning.
        
        Given context (expert predictions + losses for a step), generates
        EXPERT_k, WEIGHT_val token pairs for each expert.
        
        Args:
            context_ids: [batch, seq_len] tensor of context tokens
            tokenizer: MWTokenizer instance
            
        Returns:
            Extended sequence with generated weight tokens appended
        """
        device = context_ids.device
        generated = context_ids.clone()
        n_experts = self.config.n_experts
        
        weight_start = tokenizer.WEIGHT_TOKENS[0]
        weight_end = tokenizer.WEIGHT_TOKENS[-1] + 1
        
        for expert_idx in range(n_experts):
            # Append the expert token (structural/deterministic)
            expert_token = torch.full(
                (generated.size(0), 1),
                tokenizer.EXPERT_TOKENS[expert_idx],
                dtype=torch.long, device=device
            )
            generated = torch.cat([generated, expert_token], dim=1)
            
            # Forward pass to predict the weight value token
            outputs = self.forward(generated)
            next_logits = outputs['lm_logits'][:, -1, :]  # [batch, vocab_size]
            
            # Restrict to weight tokens only
            mask = torch.full_like(next_logits, float('-inf'))
            mask[:, weight_start:weight_end] = 0.0
            next_logits = next_logits + mask
            
            # Greedy selection
            next_token = next_logits.argmax(dim=-1, keepdim=True)  # [batch, 1]
            generated = torch.cat([generated, next_token], dim=1)
        
        return generated


def generate_sequence_with_cot(model, sequence, tokenizer, device):
    """
    Autoregressively generate a full MW sequence using discrete CoT.

    At each step the model sees expert predictions + losses as context,
    then generates EXPERT_k WEIGHT_val token pairs for each expert.
    The generated weights are decoded and used for decision making.

    Args:
        model: LearnedMWTransformer instance
        sequence: Dict with expert_predictions, losses, true_labels
        tokenizer: MWTokenizer instance
        device: torch device

    Returns:
        Dict with decisions, cot_weights, generated_ids
    """
    model.eval()
    n_experts = model.config.n_experts

    generated = torch.tensor([[tokenizer.START_TOKEN]], dtype=torch.long, device=device)
    cot_weights = []
    decisions = []

    for step in range(len(sequence['expert_predictions'])):
        # --- Step token ---
        step_token = torch.tensor(
            [[tokenizer.STEP_TOKENS[step % 100]]], dtype=torch.long, device=device
        )
        generated = torch.cat([generated, step_token], dim=1)

        # --- Expert predictions ---
        for expert_idx, pred in enumerate(sequence['expert_predictions'][step]):
            expert_token = torch.tensor(
                [[tokenizer.EXPERT_TOKENS[expert_idx]]], dtype=torch.long, device=device
            )
            pred_token = torch.tensor(
                [[tokenizer.PRED_1_TOKEN if pred == 1 else tokenizer.PRED_0_TOKEN]],
                dtype=torch.long, device=device
            )
            generated = torch.cat([generated, expert_token, pred_token], dim=1)

        # --- Decision (BEFORE seeing losses) ---
        with torch.no_grad():
            outputs = model(generated)
            pred_logit = outputs['prediction_logits'][0, -1, 0]
            decision = 1 if torch.sigmoid(pred_logit) > 0.5 else 0
        decisions.append(decision)

        # --- True label token (revealed after decision) ---
        true_label = sequence['true_labels'][step]
        label_token = tokenizer.PRED_1_TOKEN if true_label == 1 else tokenizer.PRED_0_TOKEN
        generated = torch.cat([
            generated,
            torch.tensor([[label_token]], dtype=torch.long, device=device)
        ], dim=1)

        # --- Losses (observed after decision, context for next round) ---
        for expert_idx, loss_val in enumerate(sequence['losses'][step]):
            expert_token = torch.tensor(
                [[tokenizer.EXPERT_TOKENS[expert_idx]]], dtype=torch.long, device=device
            )
            loss_token = torch.tensor(
                [[tokenizer.discretize_loss(loss_val)]], dtype=torch.long, device=device
            )
            generated = torch.cat([generated, expert_token, loss_token], dim=1)

    return {
        'decisions': np.array(decisions),
        'generated_ids': generated
    }

# ---------------------------------------------------------------------------
# ContinuousCoTTransformer (Coconut-style hidden-state recurrence)
# ---------------------------------------------------------------------------

class ContinuousCoTTransformer(nn.Module):
    """
    Transformer with Coconut-style continuous chain-of-thought reasoning.
    
    Instead of generating discrete weight tokens autoregressively, this model
    performs K recurrence steps in continuous hidden-state space:
      1. Embed the context tokens and run through the transformer.
      2. Take the hidden state at the last position.
      3. Project it back to embedding space via thought_proj.
      4. Append it to the embedding sequence as a new "thought" position.
      5. Repeat steps 1-4 for K thought steps.
      6. Read off weight predictions from weight_head on the final hidden state.
    
    This avoids the information bottleneck of discretizing continuous MW weights
    into a finite token vocabulary.
    """
    
    def __init__(self, config: ModelConfig):
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
        self.weight_head = nn.Linear(config.d_model, config.n_experts)
        self.prediction_head = nn.Linear(config.d_model, 1)
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
            'weight_logits': self.weight_head(h),
            'prediction_logits': self.prediction_head(h),
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
            thought_weights: [batch, n_experts] softmax weight predictions
            thought_hiddens: [batch, K, d_model] hidden states at each thought step
            final_h: [batch, d_model] hidden state from the final refinement pass
        """
        K = self.n_thought_steps
        device = context_embeddings.device
        current = context_embeddings  # [batch, growing_len, d_model]
        
        thought_hiddens = []
        all_attention = []  # attention from each recurrence + final pass
        
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
        weight_logits = self.weight_head(final_h)
        thought_weights = F.softmax(weight_logits, dim=-1)
        
        result = (thought_weights, torch.stack(thought_hiddens, dim=1), final_h)
        if return_attention:
            return result + (all_attention,)
        return result
    
    def think_and_predict(self, context_ids, return_attention=False):
        """
        Convenience method: embed context token ids, run thought recurrence,
        return weight predictions and decision logit.
        
        Both weights and prediction use final_h from the last refinement pass,
        which has seen all K thought embeddings.
        """
        embeddings = self.token_embedding(context_ids)  # no position yet; think() adds them
        
        if return_attention:
            weights, thought_hiddens, final_h, all_attention = self.think(embeddings, return_attention=True)
            pred_logit = self.prediction_head(final_h)  # [batch, 1]
            return weights, pred_logit, all_attention
        else:
            weights, thought_hiddens, final_h = self.think(embeddings)
            pred_logit = self.prediction_head(final_h)  # [batch, 1]
            return weights, pred_logit


def generate_sequence_with_continuous_cot(model, sequence, tokenizer, device):
    """
    Generate a full MW sequence using continuous chain-of-thought reasoning.
    
    At each step the model receives expert predictions and losses as discrete
    context tokens, then performs K continuous thought recurrences in hidden
    space to produce weight predictions (no discretization).
    
    Args:
        model: ContinuousCoTTransformer instance
        sequence: Dict with expert_predictions, losses, true_labels
        tokenizer: MWTokenizer instance
        device: torch device
    
    Returns:
        Dict with decisions, cot_weights (continuous)
    """
    model.eval()

    token_ids = [tokenizer.START_TOKEN]
    cot_weights = []
    decisions = []

    for step in range(len(sequence['expert_predictions'])):
        # Step marker
        token_ids.append(tokenizer.STEP_TOKENS[step % 100])

        # Expert predictions (context)
        for expert_idx, pred in enumerate(sequence['expert_predictions'][step]):
            token_ids.append(tokenizer.EXPERT_TOKENS[expert_idx])
            token_ids.append(
                tokenizer.PRED_1_TOKEN if pred == 1 else tokenizer.PRED_0_TOKEN
            )

        # Continuous thought: get weights and decision (BEFORE seeing losses)
        context_tensor = torch.tensor(
            [token_ids], dtype=torch.long, device=device
        )
        with torch.no_grad():
            weights, pred_logit = model.think_and_predict(context_tensor)

        step_weights = weights[0].cpu().numpy()
        cot_weights.append(step_weights)

        decision = 1 if torch.sigmoid(pred_logit[0, 0]) > 0.5 else 0
        decisions.append(decision)

        # SEP token (marks the decision point — must match training tokenization)
        token_ids.append(tokenizer.SEP_TOKEN)

        # True label token (revealed after decision)
        true_label = sequence['true_labels'][step]
        token_ids.append(
            tokenizer.PRED_1_TOKEN if true_label == 1 else tokenizer.PRED_0_TOKEN
        )

        # Losses (observed after decision, context for weight update)
        for expert_idx, loss_val in enumerate(sequence['losses'][step]):
            token_ids.append(tokenizer.EXPERT_TOKENS[expert_idx])
            token_ids.append(tokenizer.discretize_loss(loss_val))

    return {
        'decisions': np.array(decisions),
        'cot_weights': cot_weights,
    }

# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

class MWTokenizer:
    """Tokenizer for multiplicative weights sequences."""
    
    def __init__(self, n_experts: int = 4):
        self.n_experts = n_experts
        self.PAD_TOKEN = 0
        self.START_TOKEN = 1
        self.END_TOKEN = 2
        self.SEP_TOKEN = 3
        self.EXPERT_TOKENS = list(range(4, 4 + n_experts))
        self.WEIGHT_TOKENS = list(range(4 + n_experts, 4 + n_experts + 100))
        self.LOSS_TOKENS = list(range(104 + n_experts, 204 + n_experts))
        self.PRED_0_TOKEN = 204 + n_experts
        self.PRED_1_TOKEN = 205 + n_experts
        self.STEP_TOKENS = list(range(206 + n_experts, 306 + n_experts))
        self.vocab_size = 306 + n_experts
    
    def discretize_weight(self, weight: float) -> int:
        """Convert a weight value [0, 1] to a weight token."""
        bin_idx = int(np.clip(weight * 99, 0, 99))
        return self.WEIGHT_TOKENS[bin_idx]
    
    def decode_weight_token(self, token: int) -> float:
        """Convert a weight token back to a float in [0, 1]."""
        if token < self.WEIGHT_TOKENS[0] or token > self.WEIGHT_TOKENS[-1]:
            return 0.0
        return (token - self.WEIGHT_TOKENS[0]) / 99.0
    
    def discretize_loss(self, loss: float) -> int:
        """Convert a loss value [0, 1] to a loss token."""
        bin_idx = int(np.clip(loss * 99, 0, 99))
        return self.LOSS_TOKENS[bin_idx]
    
    def encode_sequence(self, sequence: Dict) -> Dict:
        """
        Encode a full MW execution trace into token ids with targets and masks.
        
        Returns dict with:
            input_ids, weight_targets, prediction_targets, target_mask, is_cot_token
        """
        n_experts = self.n_experts
        tokens = [self.START_TOKEN]
        weight_targets = [np.zeros(n_experts)]
        prediction_targets = [0.0]
        target_mask = [False]
        is_cot_token = [False]
        
        n_steps = sequence['n_steps']
        
        for step in range(n_steps):
            # Step token
            tokens.append(self.STEP_TOKENS[step % 100])
            weight_targets.append(np.zeros(n_experts))
            prediction_targets.append(0.0)
            target_mask.append(False)
            is_cot_token.append(False)
            
            # Expert predictions
            for expert_idx in range(n_experts):
                pred = sequence['expert_predictions'][step][expert_idx]
                tokens.append(self.EXPERT_TOKENS[expert_idx])
                weight_targets.append(np.zeros(n_experts))
                prediction_targets.append(0.0)
                target_mask.append(False)
                is_cot_token.append(False)
                
                tokens.append(self.PRED_1_TOKEN if pred == 1 else self.PRED_0_TOKEN)
                weight_targets.append(np.zeros(n_experts))
                prediction_targets.append(0.0)
                target_mask.append(False)
                is_cot_token.append(False)
            
            # Separator — decision target (model predicts BEFORE seeing losses)
            tokens.append(self.SEP_TOKEN)
            weight_targets.append(np.zeros(n_experts))
            true_label = sequence['true_labels'][step]
            prediction_targets.append(float(true_label))
            target_mask.append(True)
            is_cot_token.append(False)
            
            # True label (revealed after decision)
            tokens.append(self.PRED_1_TOKEN if true_label == 1 else self.PRED_0_TOKEN)
            weight_targets.append(np.zeros(n_experts))
            prediction_targets.append(0.0)
            target_mask.append(False)
            is_cot_token.append(False)
            
            # Losses (observed after decision, context for weight update)
            for expert_idx in range(n_experts):
                loss_val = sequence['losses'][step][expert_idx]
                tokens.append(self.EXPERT_TOKENS[expert_idx])
                weight_targets.append(np.zeros(n_experts))
                prediction_targets.append(0.0)
                target_mask.append(False)
                is_cot_token.append(False)
                
                tokens.append(self.discretize_loss(loss_val))
                weight_targets.append(np.zeros(n_experts))
                prediction_targets.append(0.0)
                target_mask.append(False)
                is_cot_token.append(False)
        
        tokens.append(self.END_TOKEN)
        weight_targets.append(np.zeros(n_experts))
        prediction_targets.append(0.0)
        target_mask.append(False)
        is_cot_token.append(False)
        
        return {
            'input_ids': tokens,
            'weight_targets': np.array(weight_targets),
            'prediction_targets': np.array(prediction_targets, dtype=np.float32),
            'target_mask': target_mask,
            'is_cot_token': is_cot_token,
        }

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class MWSequenceDataset(Dataset):
    """Dataset for multiplicative weights sequences."""
    
    def __init__(self, sequences: List[Dict], tokenizer):
        self.sequences = sequences
        self.tokenizer = tokenizer
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        
        # Tokenize the sequence
        tokens = self.tokenizer.encode_sequence(sequence)
        
        return {
            'input_ids': torch.tensor(tokens['input_ids'], dtype=torch.long),
            'weight_targets': torch.tensor(tokens['weight_targets'], dtype=torch.float32),
            'prediction_targets': torch.tensor(tokens['prediction_targets'], dtype=torch.float32),
            'target_mask': torch.tensor(tokens['target_mask'], dtype=torch.bool),
            'cot_mask': torch.tensor(tokens['is_cot_token'], dtype=torch.bool),
            'sequence_length': len(tokens['input_ids'])
        }


def collate_fn(batch):
    """Custom collate function for variable length sequences."""
    max_len = max(item['sequence_length'] for item in batch)
    
    input_ids = []
    weight_targets = []
    prediction_targets = []
    target_mask = []
    cot_mask = []
    
    for item in batch:
        seq_len = item['sequence_length']
        pad_len = max_len - seq_len
        
        # Pad input_ids
        padded_input = torch.cat([
            item['input_ids'],
            torch.zeros(pad_len, dtype=torch.long)
        ])
        input_ids.append(padded_input)
        
        # Ensure weight_targets has correct shape
        weight_targets_tensor = item['weight_targets']
        if len(weight_targets_tensor.shape) == 1:
            n_experts = 4
            weight_targets_tensor = weight_targets_tensor.unsqueeze(-1).expand(-1, n_experts)
        
        # Pad targets
        if pad_len > 0:
            pad_shape = (pad_len, weight_targets_tensor.shape[-1])
            padded_weight_targets = torch.cat([
                weight_targets_tensor,
                torch.zeros(pad_shape)
            ])
        else:
            padded_weight_targets = weight_targets_tensor
        weight_targets.append(padded_weight_targets)
        
        padded_pred_targets = torch.cat([
            item['prediction_targets'],
            torch.zeros(pad_len)
        ])
        prediction_targets.append(padded_pred_targets)
        
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
        'weight_targets': torch.stack(weight_targets),
        'prediction_targets': torch.stack(prediction_targets),
        'target_mask': torch.stack(target_mask),
        'cot_mask': torch.stack(cot_mask)
    }

# ---------------------------------------------------------------------------
# Training data generation
# ---------------------------------------------------------------------------

def generate_mw_training_data(n_sequences: int, max_steps: int = 10,
                               n_experts: int = 4) -> List[Dict]:
    """
    Generate synthetic MW execution traces for training.
    
    Each trace records expert predictions, losses, and true labels.
    No MW weights are stored — the model must learn to track them implicitly.
    """
    sequences = []
    
    for _ in range(n_sequences):
        n_steps = np.random.randint(3, max_steps + 1)
        
        # Expert qualities are fixed for the entire sequence
        expert_qualities = [np.random.uniform(0.3, 0.9) for _ in range(n_experts)]
        
        expert_predictions = []
        losses = []
        true_labels = []
        
        for step in range(n_steps):
            true_label = np.random.randint(0, 2)
            true_labels.append(true_label)
            
            step_preds = []
            step_losses = []
            for e in range(n_experts):
                correct = np.random.random() < expert_qualities[e]
                pred = true_label if correct else 1 - true_label
                step_preds.append(pred)
                step_losses.append(0.0 if pred == true_label else 1.0)
            
            expert_predictions.append(step_preds)
            losses.append(step_losses)
        
        sequences.append({
            'expert_predictions': expert_predictions,
            'losses': losses,
            'true_labels': true_labels,
            'n_steps': n_steps,
        })
    
    return sequences


def compute_mw_weights(sequence: Dict, n_experts: int = 4) -> List[List[float]]:
    """
    Recompute MW weight trajectory for a sequence (for evaluation only).
    
    Returns list of weight vectors: [initial_uniform, after_step_0, after_step_1, ...].
    Length = n_steps + 1.
    """
    from scripts.multiplicative_weights import MultiplicativeWeights
    
    n_steps = sequence['n_steps']
    lr = np.sqrt(np.log(n_experts) / max(n_steps, 1))
    mw = MultiplicativeWeights(n_experts, lr)
    weights = [mw.get_probabilities().tolist()]
    
    for step_losses in sequence['losses']:
        mw.update_weights(np.array(step_losses))
        weights.append(mw.get_probabilities().tolist())
    
    return weights

# ---------------------------------------------------------------------------
# Trainers
# ---------------------------------------------------------------------------

class MWTrainer:
    """Trainer for the learned multiplicative weights transformer."""
    
    def __init__(self, model: LearnedMWTransformer, 
                 train_config: TrainingConfig,
                 model_config: ModelConfig):
        self.model = model
        self.train_config = train_config
        self.model_config = model_config
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=train_config.learning_rate,
            weight_decay=train_config.weight_decay,
            betas=(0.9, 0.95)
        )
        
        # Loss function: BCE on decisions at masked positions
        self.prediction_loss_fn = nn.BCEWithLogitsLoss()
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
        prediction_targets = batch['prediction_targets'].to(self.device)
        target_mask = batch['target_mask'].to(self.device)
        
        outputs = self.model(input_ids)
        total_loss = 0.0
        batch_size = target_mask.shape[0]
        
        # BCE loss on predicted decisions at masked positions
        pred_losses = []
        for i in range(batch_size):
            sample_mask = target_mask[i]
            if sample_mask.any():
                pl = outputs['prediction_logits'][i].squeeze(-1)
                pt = prediction_targets[i]
                ml = min(len(sample_mask), len(pl), len(pt))
                sample_mask = sample_mask[:ml]
                pl = pl[:ml]
                pt = pt[:ml]
                if sample_mask.any():
                    masked_pl = pl[sample_mask]
                    masked_pt = pt[sample_mask]
                    if len(masked_pl) > 0:
                        pred_losses.append(self.prediction_loss_fn(masked_pl, masked_pt))
        if pred_losses:
            total_loss = torch.stack(pred_losses).mean()
        
        if isinstance(total_loss, (int, float)):
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
                prediction_targets = batch['prediction_targets'].to(self.device)
                target_mask = batch['target_mask'].to(self.device)
                
                outputs = self.model(batch['input_ids'].to(self.device))
                batch_loss = 0.0
                batch_size = target_mask.shape[0]
                
                pred_losses = []
                for i in range(batch_size):
                    sm = target_mask[i]
                    if sm.any():
                        pl = outputs['prediction_logits'][i].squeeze(-1)
                        pt = prediction_targets[i]
                        ml = min(len(sm), len(pl), len(pt))
                        sm = sm[:ml]; pl = pl[:ml]; pt = pt[:ml]
                        if sm.any():
                            mpl = pl[sm]; mpt = pt[sm]
                            if len(mpl) > 0:
                                pred_losses.append(self.prediction_loss_fn(mpl, mpt))
                if pred_losses:
                    batch_loss = torch.stack(pred_losses).mean()
                
                total_loss += batch_loss.item() if not isinstance(batch_loss, float) else batch_loss
                n_batches += 1
        
        return {'loss': total_loss / n_batches if n_batches > 0 else 0.0}


class ContinuousCoTTrainer:
    """
    Trainer for ContinuousCoTTransformer with Coconut-style multi-stage curriculum.
    
    At each training stage i, the model uses min(i, K_max) continuous thought
    recurrences before each decision. This teaches the model to progressively
    use more "thinking" steps, learning to maintain and update expert weights
    in hidden space rather than through explicit tokens.
    
    At each decision point (SEP position), the trainer:
      1. Extracts context tokens up to that point.
      2. Runs think_and_predict with the current stage's thought depth.
      3. Computes BCE loss on the decision prediction.
    """
    
    def __init__(self, model: ContinuousCoTTransformer,
                 train_config: TrainingConfig,
                 model_config: ModelConfig):
        self.model = model
        self.train_config = train_config
        self.model_config = model_config
        
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=train_config.learning_rate,
            weight_decay=train_config.weight_decay,
            betas=(0.9, 0.95)
        )
        
        self.prediction_loss_fn = nn.BCEWithLogitsLoss()
        self.device = next(model.parameters()).device
        self.scheduler = None
        
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
        
        # Cosine LR schedule per stage: decays to lr/10 over max_epochs
        T_max = self.train_config.max_epochs_per_stage * len(train_loader)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=max(T_max, 1),
            eta_min=self.train_config.learning_rate / 10
        )
        
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
                
                if self.scheduler is not None:
                    self.scheduler.step()
                
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
                    wandb.log({'train/epoch_loss': avg_loss, 'stage': stage, 'epoch': epoch, 'global_step': self.step})
            except (ImportError, Exception):
                pass
            
            # Early stopping check on training loss
            if val_loader:
                val_metrics = self._evaluate(val_loader)
                val_loss = val_metrics['loss']
                if val_loss < best_val_loss - 1e-4:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
                    break
        
        return {'losses': stage_losses}
    
    def _train_batch(self, batch: Dict, stage: int) -> float:
        """Train on a single batch using think_and_predict at each decision point.
        
        For each sequence, finds decision positions (target_mask == True),
        extracts context up to each position, runs thought recurrence,
        and computes BCE loss on the prediction.
        """
        self.optimizer.zero_grad()
        
        input_ids = batch['input_ids'].to(self.device)
        prediction_targets = batch['prediction_targets'].to(self.device)
        target_mask = batch['target_mask'].to(self.device)
        
        batch_size = input_ids.shape[0]
        total_loss = 0.0
        n_decisions = 0
        
        for i in range(batch_size):
            seq_ids = input_ids[i]
            seq_mask = target_mask[i]
            seq_targets = prediction_targets[i]
            
            # Find decision positions (where target_mask is True)
            decision_pos = seq_mask.nonzero(as_tuple=True)[0]
            
            for pos in decision_pos:
                # Context: all tokens before the decision position (SEP)
                context = seq_ids[:pos].unsqueeze(0)  # [1, ctx_len]
                
                if context.shape[1] < 2:
                    continue
                
                # Run thought recurrence and get prediction
                _, pred_logit = self.model.think_and_predict(context)
                
                # BCE loss on the decision
                target = seq_targets[pos]
                loss = self.prediction_loss_fn(pred_logit.squeeze(), target)
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
                prediction_targets = batch['prediction_targets'].to(self.device)
                target_mask = batch['target_mask'].to(self.device)
                
                batch_size = input_ids.shape[0]
                
                for i in range(batch_size):
                    seq_ids = input_ids[i]
                    seq_mask = target_mask[i]
                    seq_targets = prediction_targets[i]
                    
                    decision_pos = seq_mask.nonzero(as_tuple=True)[0]
                    
                    for pos in decision_pos:
                        context = seq_ids[:pos].unsqueeze(0)
                        if context.shape[1] < 2:
                            continue
                        
                        _, pred_logit = self.model.think_and_predict(context)
                        target = seq_targets[pos]
                        loss = self.prediction_loss_fn(pred_logit.squeeze(), target)
                        total_loss += loss.item()
                        n_decisions += 1
        
        return {'loss': total_loss / max(n_decisions, 1)}
