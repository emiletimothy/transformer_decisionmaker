"""
Learned Encoder Transformer

GPT-2 style transformer that learns to reproduce a simple neural network
encoder's behavior through gradient-based training on encoding traces.

Includes:
  - LearnedEncoderTransformer: base model with discrete CoT support
  - ContinuousCoTEncoderTransformer: Coconut-style continuous hidden-state reasoning
  - EncoderTokenizer: tokenizes encoding traces into token sequences
  - EncoderSequenceDataset / collate_fn: PyTorch dataset utilities
  - EncoderTrainer / ContinuousCoTEncoderTrainer: training loops
  - generate_encoder_training_data: synthetic trace generation
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
    """Configuration for the learned encoder transformer."""
    d_model: int = 256
    n_heads: int = 4
    n_layers: int = 2
    input_dim: int = 16       # encoder input dimensionality
    latent_dim: int = 4       # encoder latent dimensionality
    max_sequence_length: int = 1024
    vocab_size: int = 1000    # set based on tokenization
    dropout: float = 0.1
    n_thought_steps: int = 4  # continuous thought recurrence steps (0 = disabled)


@dataclass
class TrainingConfig:
    """Configuration for training the learned encoder transformer."""
    learning_rate: float = 3e-4
    weight_decay: float = 1e-2
    batch_size: int = 32
    max_epochs_per_stage: int = 30
    early_stopping_patience: int = 12
    max_stages: int = 10
    stage_mixing_prob: float = 0.1
    eval_interval: int = 50


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
# LearnedEncoderTransformer (base model + discrete CoT)
# ---------------------------------------------------------------------------

class LearnedEncoderTransformer(nn.Module):
    """GPT-2 style transformer that learns to reproduce encoder mappings."""

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

        # Output heads
        self.latent_head = nn.Linear(config.d_model, config.latent_dim)   # predict latent code
        self.lm_head = nn.Linear(config.d_model, config.vocab_size)       # next-token prediction

        self.dropout = nn.Dropout(config.dropout)
        self.apply(self._init_weights)

    def _init_weights(self, module):
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

        token_emb = self.token_embedding(input_ids)
        pos_emb = self.position_embedding(position_ids)
        x = self.dropout(token_emb + pos_emb)

        # Causal mask
        mask = torch.tril(torch.ones(seq_len, seq_len, device=input_ids.device)).unsqueeze(0).unsqueeze(0)

        attention_weights = []
        for block in self.blocks:
            x, attn = block(x, mask)
            if return_attention:
                attention_weights.append(attn)

        x = self.ln_final(x)

        latent_logits = self.latent_head(x)      # [batch, seq_len, latent_dim]
        lm_logits = self.lm_head(x)              # [batch, seq_len, vocab_size]

        outputs = {
            'latent_logits': latent_logits,
            'lm_logits': lm_logits,
        }
        if return_attention:
            outputs['attention_weights'] = attention_weights

        return outputs


# ---------------------------------------------------------------------------
# ContinuousCoTEncoderTransformer (Coconut-style hidden-state recurrence)
# ---------------------------------------------------------------------------

class ContinuousCoTEncoderTransformer(nn.Module):
    """
    Transformer with Coconut-style continuous chain-of-thought reasoning
    for learning encoder behavior.

    Instead of generating discrete latent tokens autoregressively, performs
    K recurrence steps in continuous hidden-state space before predicting
    the latent code.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.n_thought_steps = config.n_thought_steps

        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(
            config.max_sequence_length + config.n_thought_steps + 1,
            config.d_model
        )

        self.blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layers)
        ])
        self.ln_final = nn.LayerNorm(config.d_model)

        # Output heads
        self.latent_head = nn.Linear(config.d_model, config.latent_dim)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size)

        # Continuous thought projection
        self.thought_proj = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, config.d_model),
        )

        self.dropout = nn.Dropout(config.dropout)
        self.apply(self._init_weights)

    def _init_weights(self, module):
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
        """Standard forward pass (compatible with discrete training code)."""
        batch_size, seq_len = input_ids.shape

        if position_ids is None:
            position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)

        x = self.dropout(
            self.token_embedding(input_ids) + self.position_embedding(position_ids)
        )
        h, attn = self._run_blocks(x, return_attention)

        outputs = {
            'latent_logits': self.latent_head(h),
            'lm_logits': self.lm_head(h),
        }
        if return_attention:
            outputs['attention_weights'] = attn
        return outputs

    def think(self, context_embeddings, return_attention=False):
        """
        Coconut-style continuous thought recurrence.

        Given raw context embeddings [batch, ctx_len, d_model], performs K
        recurrence steps and returns latent predictions from the final state.
        """
        K = self.n_thought_steps
        device = context_embeddings.device
        current = context_embeddings

        thought_hiddens = []
        all_attention = []

        for k in range(K):
            total_len = current.size(1)
            pos_ids = torch.arange(total_len, device=device).unsqueeze(0)
            positioned = current + self.position_embedding(pos_ids)

            h, attn = self._run_blocks(self.dropout(positioned), return_attention)
            if return_attention:
                all_attention.append(attn)
            last_h = h[:, -1:, :]
            thought_hiddens.append(last_h.squeeze(1))

            thought_embed = self.thought_proj(last_h)
            current = torch.cat([current, thought_embed], dim=1)

        # Final forward with all thoughts appended
        total_len = current.size(1)
        pos_ids = torch.arange(total_len, device=device).unsqueeze(0)
        positioned = current + self.position_embedding(pos_ids)
        h, attn = self._run_blocks(self.dropout(positioned), return_attention)
        if return_attention:
            all_attention.append(attn)

        final_h = h[:, -1, :]
        latent_pred = self.latent_head(final_h)   # [batch, latent_dim]

        result = (latent_pred, torch.stack(thought_hiddens, dim=1), final_h)
        if return_attention:
            return result + (all_attention,)
        return result

    def think_and_predict(self, context_ids, return_attention=False):
        """
        Convenience: embed context token ids, run thought recurrence,
        return latent prediction.
        """
        embeddings = self.token_embedding(context_ids)

        if return_attention:
            latent_pred, thought_hiddens, final_h, all_attention = self.think(
                embeddings, return_attention=True
            )
            return latent_pred, all_attention
        else:
            latent_pred, thought_hiddens, final_h = self.think(embeddings)
            return latent_pred


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

N_BINS = 100  # discretization bins for continuous values

class EncoderTokenizer:
    """
    Tokenizer for encoder execution traces.

    Token layout:
      0          PAD
      1          START
      2          END
      3          SEP  (marks the boundary between input and target latent)
      4..4+S-1   STEP tokens (step index markers, S=100)
      next block INPUT_DIM tokens (one per input dimension marker)
      next block VALUE tokens (N_BINS bins for discretized floats)
      next block LATENT_DIM tokens (one per latent dimension marker)
      next block LATENT_VALUE tokens (N_BINS bins for discretized latent floats)
    """

    def __init__(self, input_dim: int = 16, latent_dim: int = 4):
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        self.PAD_TOKEN = 0
        self.START_TOKEN = 1
        self.END_TOKEN = 2
        self.SEP_TOKEN = 3

        offset = 4
        self.STEP_TOKENS = list(range(offset, offset + 100))
        offset += 100

        # Input dimension markers: DIM_0, DIM_1, ..., DIM_{input_dim-1}
        self.INPUT_DIM_TOKENS = list(range(offset, offset + input_dim))
        offset += input_dim

        # Value tokens for input features (discretized to N_BINS)
        self.INPUT_VALUE_TOKENS = list(range(offset, offset + N_BINS))
        offset += N_BINS

        # Latent dimension markers
        self.LATENT_DIM_TOKENS = list(range(offset, offset + latent_dim))
        offset += latent_dim

        # Value tokens for latent features
        self.LATENT_VALUE_TOKENS = list(range(offset, offset + N_BINS))
        offset += N_BINS

        self.vocab_size = offset

    def discretize_value(self, value: float, vmin: float = -3.0, vmax: float = 3.0) -> int:
        """Map a continuous value to a bin index in [0, N_BINS-1]."""
        normalized = (value - vmin) / (vmax - vmin)
        bin_idx = int(np.clip(normalized * (N_BINS - 1), 0, N_BINS - 1))
        return bin_idx

    def undiscretize_value(self, bin_idx: int, vmin: float = -3.0, vmax: float = 3.0) -> float:
        """Map a bin index back to a continuous value."""
        return vmin + (bin_idx / (N_BINS - 1)) * (vmax - vmin)

    def encode_input_token(self, dim_idx: int, value: float) -> List[int]:
        """Encode a single input dimension as [DIM_marker, VALUE_token]."""
        return [
            self.INPUT_DIM_TOKENS[dim_idx],
            self.INPUT_VALUE_TOKENS[self.discretize_value(value)],
        ]

    def encode_latent_token(self, dim_idx: int, value: float) -> List[int]:
        """Encode a single latent dimension as [LATENT_marker, VALUE_token]."""
        return [
            self.LATENT_DIM_TOKENS[dim_idx],
            self.LATENT_VALUE_TOKENS[self.discretize_value(value)],
        ]

    def decode_latent_tokens(self, tokens: List[int]) -> np.ndarray:
        """Decode latent value tokens back to a float vector."""
        latent = np.zeros(self.latent_dim, dtype=np.float32)
        lv_start = self.LATENT_VALUE_TOKENS[0]
        lv_end = self.LATENT_VALUE_TOKENS[-1]
        dim_idx = 0
        for t in tokens:
            if t in self.LATENT_DIM_TOKENS:
                dim_idx = t - self.LATENT_DIM_TOKENS[0]
            elif lv_start <= t <= lv_end:
                bin_idx = t - lv_start
                latent[dim_idx] = self.undiscretize_value(bin_idx)
        return latent

    def encode_sequence(self, sequence: Dict) -> Dict:
        """
        Encode a full encoding trace into token ids with targets and masks.

        Each step in the sequence:
          STEP_t | DIM_0 VAL ... DIM_{d-1} VAL | SEP | LAT_0 VAL ... LAT_{l-1} VAL

        Returns dict with:
            input_ids, latent_targets, target_mask
        """
        tokens = [self.START_TOKEN]
        latent_targets = [np.zeros(self.latent_dim)]
        target_mask = [False]

        n_steps = sequence['n_steps']

        for step in range(n_steps):
            # Step marker
            tokens.append(self.STEP_TOKENS[step % 100])
            latent_targets.append(np.zeros(self.latent_dim))
            target_mask.append(False)

            # Input vector: DIM_i VALUE pairs
            input_vec = sequence['inputs'][step]
            for d in range(self.input_dim):
                pair = self.encode_input_token(d, input_vec[d])
                for t in pair:
                    tokens.append(t)
                    latent_targets.append(np.zeros(self.latent_dim))
                    target_mask.append(False)

            # SEP — this is the decision point where model predicts the latent code
            tokens.append(self.SEP_TOKEN)
            target_latent = np.array(sequence['latents'][step], dtype=np.float32)
            latent_targets.append(target_latent)
            target_mask.append(True)

            # Ground truth latent tokens (teacher forcing context for next steps)
            for d in range(self.latent_dim):
                pair = self.encode_latent_token(d, target_latent[d])
                for t in pair:
                    tokens.append(t)
                    latent_targets.append(np.zeros(self.latent_dim))
                    target_mask.append(False)

        tokens.append(self.END_TOKEN)
        latent_targets.append(np.zeros(self.latent_dim))
        target_mask.append(False)

        return {
            'input_ids': tokens,
            'latent_targets': np.array(latent_targets, dtype=np.float32),
            'target_mask': target_mask,
        }


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class EncoderSequenceDataset(Dataset):
    """Dataset for encoder execution traces."""

    def __init__(self, sequences: List[Dict], tokenizer: EncoderTokenizer):
        self.sequences = sequences
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        tokens = self.tokenizer.encode_sequence(sequence)

        return {
            'input_ids': torch.tensor(tokens['input_ids'], dtype=torch.long),
            'latent_targets': torch.tensor(tokens['latent_targets'], dtype=torch.float32),
            'target_mask': torch.tensor(tokens['target_mask'], dtype=torch.bool),
            'sequence_length': len(tokens['input_ids']),
        }


def collate_fn(batch):
    """Custom collate function for variable length sequences."""
    max_len = max(item['sequence_length'] for item in batch)
    latent_dim = batch[0]['latent_targets'].shape[-1]

    input_ids = []
    latent_targets = []
    target_mask = []

    for item in batch:
        seq_len = item['sequence_length']
        pad_len = max_len - seq_len

        padded_input = torch.cat([
            item['input_ids'],
            torch.zeros(pad_len, dtype=torch.long)
        ])
        input_ids.append(padded_input)

        padded_latent = torch.cat([
            item['latent_targets'],
            torch.zeros(pad_len, latent_dim)
        ])
        latent_targets.append(padded_latent)

        padded_mask = torch.cat([
            item['target_mask'],
            torch.zeros(pad_len, dtype=torch.bool)
        ])
        target_mask.append(padded_mask)

    return {
        'input_ids': torch.stack(input_ids),
        'latent_targets': torch.stack(latent_targets),
        'target_mask': torch.stack(target_mask),
    }


# ---------------------------------------------------------------------------
# Training data generation
# ---------------------------------------------------------------------------

def generate_encoder_training_data(
    encoder_model,
    n_sequences: int,
    max_steps: int = 10,
    input_dim: int = 16,
    n_clusters: int = 5,
    cluster_std: float = 0.3,
    device: Optional[torch.device] = None,
) -> List[Dict]:
    """
    Generate encoding traces by running the ground truth encoder on random inputs.

    Each trace is a sequence of (input_vector, latent_code) pairs — analogous
    to MW execution traces with (expert_predictions, losses).
    """
    if device is None:
        device = next(encoder_model.parameters()).device

    encoder_model.eval()
    sequences = []

    for _ in range(n_sequences):
        n_steps = np.random.randint(3, max_steps + 1)

        # Generate random cluster centers for this sequence
        centers = np.random.randn(n_clusters, input_dim).astype(np.float32)

        inputs = []
        latents = []
        labels = []

        for step in range(n_steps):
            # Pick a random cluster and sample around it
            cluster_id = np.random.randint(0, n_clusters)
            x = centers[cluster_id] + cluster_std * np.random.randn(input_dim).astype(np.float32)
            inputs.append(x.tolist())
            labels.append(int(cluster_id))

            # Run encoder
            with torch.no_grad():
                x_tensor = torch.tensor(x, dtype=torch.float32, device=device).unsqueeze(0)
                z = encoder_model.encode(x_tensor)
            latents.append(z[0].cpu().numpy().tolist())

        sequences.append({
            'inputs': inputs,
            'latents': latents,
            'labels': labels,
            'n_steps': n_steps,
        })

    return sequences


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def generate_sequence_with_cot(model, sequence, tokenizer, device):
    """
    Autoregressively generate a full encoding sequence using discrete CoT.

    At each step the model sees input tokens, makes a latent prediction,
    then sees the ground truth latent tokens before the next step.
    """
    model.eval()

    generated = torch.tensor([[tokenizer.START_TOKEN]], dtype=torch.long, device=device)
    predicted_latents = []

    for step in range(len(sequence['inputs'])):
        # Step token
        step_token = torch.tensor(
            [[tokenizer.STEP_TOKENS[step % 100]]], dtype=torch.long, device=device
        )
        generated = torch.cat([generated, step_token], dim=1)

        # Input vector tokens
        input_vec = sequence['inputs'][step]
        for d in range(tokenizer.input_dim):
            pair = tokenizer.encode_input_token(d, input_vec[d])
            pair_tensor = torch.tensor([pair], dtype=torch.long, device=device)
            generated = torch.cat([generated, pair_tensor], dim=1)

        # Predict latent (BEFORE seeing ground truth)
        with torch.no_grad():
            outputs = model(generated)
            latent_pred = outputs['latent_logits'][0, -1, :]  # [latent_dim]
        predicted_latents.append(latent_pred.cpu().numpy())

        # SEP token
        sep = torch.tensor([[tokenizer.SEP_TOKEN]], dtype=torch.long, device=device)
        generated = torch.cat([generated, sep], dim=1)

        # Ground truth latent tokens (revealed after prediction, teacher forcing)
        gt_latent = sequence['latents'][step]
        for d in range(tokenizer.latent_dim):
            pair = tokenizer.encode_latent_token(d, gt_latent[d])
            pair_tensor = torch.tensor([pair], dtype=torch.long, device=device)
            generated = torch.cat([generated, pair_tensor], dim=1)

    return {
        'predicted_latents': np.array(predicted_latents),
        'generated_ids': generated,
    }


def generate_sequence_with_continuous_cot(model, sequence, tokenizer, device):
    """
    Generate a full encoding sequence using continuous chain-of-thought.

    At each step the model receives input tokens as context, then performs
    K continuous thought recurrences to produce latent predictions.
    """
    model.eval()

    token_ids = [tokenizer.START_TOKEN]
    predicted_latents = []

    for step in range(len(sequence['inputs'])):
        # Step marker
        token_ids.append(tokenizer.STEP_TOKENS[step % 100])

        # Input vector tokens
        input_vec = sequence['inputs'][step]
        for d in range(tokenizer.input_dim):
            pair = tokenizer.encode_input_token(d, input_vec[d])
            token_ids.extend(pair)

        # Continuous thought: predict latent code (BEFORE seeing ground truth)
        context_tensor = torch.tensor(
            [token_ids], dtype=torch.long, device=device
        )
        with torch.no_grad():
            latent_pred = model.think_and_predict(context_tensor)
        predicted_latents.append(latent_pred[0].cpu().numpy())

        # SEP token
        token_ids.append(tokenizer.SEP_TOKEN)

        # Ground truth latent tokens (teacher forcing context for next step)
        gt_latent = sequence['latents'][step]
        for d in range(tokenizer.latent_dim):
            pair = tokenizer.encode_latent_token(d, gt_latent[d])
            token_ids.extend(pair)

    return {
        'predicted_latents': np.array(predicted_latents),
    }


# ---------------------------------------------------------------------------
# Trainers
# ---------------------------------------------------------------------------

class EncoderTrainer:
    """Trainer for the learned encoder transformer (discrete mode)."""

    def __init__(self, model: LearnedEncoderTransformer,
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

        self.latent_loss_fn = nn.MSELoss()
        self.device = next(model.parameters()).device

        self.step = 0
        self.training_history = []

    def train_stage(self, stage: int, train_loader: DataLoader,
                    val_loader: DataLoader = None) -> Dict:
        """Train for one stage of the multi-stage curriculum."""
        logger.info(f"Training stage {stage}")

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

                if self.step % self.train_config.eval_interval == 0 and val_loader:
                    val_metrics = self._evaluate(val_loader)
                    logger.info(f"Step {self.step}: Val loss = {val_metrics['loss']:.4f}")

                self.step += 1

            avg_loss = epoch_loss / max(n_batches, 1)
            stage_losses.append(avg_loss)
            logger.info(f"Stage {stage}, Epoch {epoch}: Loss = {avg_loss:.6f}")

            if val_loader:
                val_metrics = self._evaluate(val_loader)
                val_loss = val_metrics['loss']
                if val_loss < best_val_loss - 1e-4:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break

        return {'losses': stage_losses}

    def _train_batch(self, batch: Dict, stage: int) -> float:
        self.optimizer.zero_grad()

        input_ids = batch['input_ids'].to(self.device)
        latent_targets = batch['latent_targets'].to(self.device)
        target_mask = batch['target_mask'].to(self.device)

        outputs = self.model(input_ids)
        batch_size = target_mask.shape[0]

        losses = []
        for i in range(batch_size):
            sample_mask = target_mask[i]
            if sample_mask.any():
                pred = outputs['latent_logits'][i]
                tgt = latent_targets[i]
                ml = min(len(sample_mask), pred.shape[0], tgt.shape[0])
                sample_mask = sample_mask[:ml]
                pred = pred[:ml]
                tgt = tgt[:ml]
                if sample_mask.any():
                    masked_pred = pred[sample_mask]
                    masked_tgt = tgt[sample_mask]
                    if len(masked_pred) > 0:
                        losses.append(self.latent_loss_fn(masked_pred, masked_tgt))

        if losses:
            total_loss = torch.stack(losses).mean()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            return total_loss.item()
        return 0.0

    def _evaluate(self, val_loader: DataLoader) -> Dict:
        self.model.eval()
        total_loss = 0.0
        n_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(self.device)
                latent_targets = batch['latent_targets'].to(self.device)
                target_mask = batch['target_mask'].to(self.device)

                outputs = self.model(input_ids)
                batch_size = target_mask.shape[0]

                losses = []
                for i in range(batch_size):
                    sm = target_mask[i]
                    if sm.any():
                        pred = outputs['latent_logits'][i]
                        tgt = latent_targets[i]
                        ml = min(len(sm), pred.shape[0], tgt.shape[0])
                        sm = sm[:ml]; pred = pred[:ml]; tgt = tgt[:ml]
                        if sm.any():
                            mp = pred[sm]; mt = tgt[sm]
                            if len(mp) > 0:
                                losses.append(self.latent_loss_fn(mp, mt))

                if losses:
                    batch_loss = torch.stack(losses).mean()
                    total_loss += batch_loss.item()
                n_batches += 1

        return {'loss': total_loss / max(n_batches, 1)}


class ContinuousCoTEncoderTrainer:
    """
    Trainer for ContinuousCoTEncoderTransformer with Coconut-style curriculum.

    At each training stage i, uses min(i, K_max) continuous thought recurrences
    before each latent prediction.
    """

    def __init__(self, model: ContinuousCoTEncoderTransformer,
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

        self.latent_loss_fn = nn.MSELoss()
        self.device = next(model.parameters()).device
        self.scheduler = None

        self.step = 0
        self.training_history = []

    def train_stage(self, stage: int, train_loader: DataLoader,
                    val_loader: DataLoader = None) -> Dict:
        n_thoughts = min(stage, self.model.config.n_thought_steps)
        self.model.n_thought_steps = n_thoughts
        logger.info(f"Training stage {stage} with {n_thoughts} thought steps")

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

                self.step += 1

            avg_loss = epoch_loss / max(n_batches, 1)
            stage_losses.append(avg_loss)
            logger.info(f"Stage {stage}, Epoch {epoch}: Loss = {avg_loss:.6f}")

            try:
                import wandb
                if wandb.run is not None:
                    wandb.log({'train/epoch_loss': avg_loss, 'stage': stage,
                               'epoch': epoch, 'global_step': self.step})
            except (ImportError, Exception):
                pass

            if val_loader:
                val_metrics = self._evaluate(val_loader)
                val_loss = val_metrics['loss']
                if val_loss < best_val_loss - 1e-4:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break

        return {'losses': stage_losses}

    def _train_batch(self, batch: Dict, stage: int) -> float:
        self.optimizer.zero_grad()

        input_ids = batch['input_ids'].to(self.device)
        latent_targets = batch['latent_targets'].to(self.device)
        target_mask = batch['target_mask'].to(self.device)

        batch_size = input_ids.shape[0]
        total_loss = 0.0
        n_decisions = 0

        for i in range(batch_size):
            seq_ids = input_ids[i]
            seq_mask = target_mask[i]
            seq_targets = latent_targets[i]

            decision_pos = seq_mask.nonzero(as_tuple=True)[0]

            for pos in decision_pos:
                context = seq_ids[:pos].unsqueeze(0)
                if context.shape[1] < 2:
                    continue

                latent_pred = self.model.think_and_predict(context)
                target = seq_targets[pos]
                loss = self.latent_loss_fn(latent_pred.squeeze(0), target)
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
        self.model.eval()
        total_loss = 0.0
        n_decisions = 0

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(self.device)
                latent_targets = batch['latent_targets'].to(self.device)
                target_mask = batch['target_mask'].to(self.device)

                batch_size = input_ids.shape[0]

                for i in range(batch_size):
                    seq_ids = input_ids[i]
                    seq_mask = target_mask[i]
                    seq_targets = latent_targets[i]

                    decision_pos = seq_mask.nonzero(as_tuple=True)[0]

                    for pos in decision_pos:
                        context = seq_ids[:pos].unsqueeze(0)
                        if context.shape[1] < 2:
                            continue

                        latent_pred = self.model.think_and_predict(context)
                        target = seq_targets[pos]
                        loss = self.latent_loss_fn(latent_pred.squeeze(0), target)
                        total_loss += loss.item()
                        n_decisions += 1

        return {'loss': total_loss / max(n_decisions, 1)}
