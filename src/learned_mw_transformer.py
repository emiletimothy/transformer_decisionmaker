"""
Learned Multiplicative Weights Transformer

This module implements a GPT-2 style transformer that learns to perform
multiplicative weights updates through gradient-based training, similar to
how transformers can learn algorithmic reasoning tasks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import logging
from torch.utils.data import Dataset, DataLoader
import math

logger = logging.getLogger(__name__)

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

@dataclass
class TrainingConfig:
    """Configuration for training the learned MW transformer."""
    learning_rate: float = 1e-4
    weight_decay: float = 1e-2
    batch_size: int = 32
    max_epochs_per_stage: int = 25
    max_stages: int = 12  # Up to 12-step reasoning
    stage_mixing_prob: float = 0.1
    warmup_steps: int = 1000
    eval_interval: int = 500

class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism."""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.d_head = config.d_model // config.n_heads
        
        self.q_proj = nn.Linear(config.d_model, config.d_model)
        self.k_proj = nn.Linear(config.d_model, config.d_model)
        self.v_proj = nn.Linear(config.d_model, config.d_model)
        self.out_proj = nn.Linear(config.d_model, config.d_model)
        
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, x, mask=None):
        batch_size, seq_len, d_model = x.shape
        
        # Project to Q, K, V
        q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
            
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        
        return self.out_proj(out), attn_weights

class TransformerBlock(nn.Module):
    """Single transformer block with attention and feed-forward."""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.attention = MultiHeadAttention(config)
        self.ln1 = nn.LayerNorm(config.d_model)
        self.ln2 = nn.LayerNorm(config.d_model)
        
        self.mlp = nn.Sequential(
            nn.Linear(config.d_model, 4 * config.d_model),
            nn.GELU(),
            nn.Linear(4 * config.d_model, config.d_model),
            nn.Dropout(config.dropout)
        )
        
    def forward(self, x, mask=None):
        # Self-attention with residual connection
        attn_out, attn_weights = self.attention(self.ln1(x), mask)
        x = x + attn_out
        
        # Feed-forward with residual connection
        x = x + self.mlp(self.ln2(x))
        
        return x, attn_weights

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
        
        outputs = {
            'weight_logits': weight_logits,
            'prediction_logits': prediction_logits,
        }
        
        if return_attention:
            outputs['attention_weights'] = attention_weights
            
        return outputs

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
            'sequence_length': len(tokens['input_ids'])
        }

class MWTokenizer:
    """Tokenizer for multiplicative weights sequences."""
    
    def __init__(self, n_experts: int = 4):
        self.n_experts = n_experts
        
        # Special tokens
        self.PAD_TOKEN = 0
        self.START_TOKEN = 1
        self.END_TOKEN = 2
        self.SEP_TOKEN = 3
        
        # Expert tokens
        self.EXPERT_TOKENS = list(range(4, 4 + n_experts))
        
        # Weight tokens (discretized weights)
        self.WEIGHT_TOKENS = list(range(4 + n_experts, 4 + n_experts + 100))
        
        # Loss tokens (discretized losses)
        self.LOSS_TOKENS = list(range(104 + n_experts, 204 + n_experts))
        
        # Prediction tokens
        self.PRED_0_TOKEN = 204 + n_experts
        self.PRED_1_TOKEN = 205 + n_experts
        
        # Step tokens
        self.STEP_TOKENS = list(range(206 + n_experts, 306 + n_experts))
        
        self.vocab_size = 306 + n_experts
        
    def discretize_weight(self, weight: float) -> int:
        """Convert continuous weight to discrete token."""
        # Map [0, 1] to weight tokens
        idx = int(weight * 99)
        idx = max(0, min(99, idx))
        return self.WEIGHT_TOKENS[idx]
    
    def discretize_loss(self, loss: float) -> int:
        """Convert continuous loss to discrete token."""
        # Map [0, 1] to loss tokens
        idx = int(loss * 99)
        idx = max(0, min(99, idx))
        return self.LOSS_TOKENS[idx]
    
    def encode_sequence(self, sequence: Dict) -> Dict:
        """Encode a MW sequence into tokens."""
        expert_predictions = sequence['expert_predictions']
        losses = sequence['losses']
        weights_sequence = sequence['weights_sequence']
        true_labels = sequence['true_labels']
        
        input_ids = [self.START_TOKEN]
        weight_targets = []
        prediction_targets = []
        target_mask = []
        
        for step in range(len(expert_predictions)):
            # Add step marker
            input_ids.append(self.STEP_TOKENS[step % 100])
            weight_targets.append([0.0] * self.n_experts)  # No target for step token
            prediction_targets.append(0.0)
            target_mask.append(False)
            
            # Add expert predictions
            for expert_idx, pred in enumerate(expert_predictions[step]):
                input_ids.append(self.EXPERT_TOKENS[expert_idx])
                input_ids.append(self.PRED_1_TOKEN if pred == 1 else self.PRED_0_TOKEN)
                
                weight_targets.extend([[0.0] * self.n_experts, [0.0] * self.n_experts])
                prediction_targets.extend([0.0, 0.0])
                target_mask.extend([False, False])
            
            # Add losses
            for expert_idx, loss in enumerate(losses[step]):
                input_ids.append(self.EXPERT_TOKENS[expert_idx])
                input_ids.append(self.discretize_loss(loss))
                
                weight_targets.extend([[0.0] * self.n_experts, [0.0] * self.n_experts])
                prediction_targets.extend([0.0, 0.0])
                target_mask.extend([False, False])
            
            # Add current weights (target for next step)
            if step < len(weights_sequence) - 1:
                next_weights = weights_sequence[step + 1]
                for expert_idx, weight in enumerate(next_weights):
                    input_ids.append(self.EXPERT_TOKENS[expert_idx])
                    input_ids.append(self.discretize_weight(weight))
                    
                    # Target: predict the weight value
                    target_weights = [0.0] * self.n_experts
                    target_weights[expert_idx] = weight
                    
                    weight_targets.extend([[0.0] * self.n_experts, target_weights])
                    prediction_targets.extend([0.0, 0.0])
                    target_mask.extend([False, True])  # Only predict weight values
            
            # Add true label and prediction target
            input_ids.append(self.PRED_1_TOKEN if true_labels[step] == 1 else self.PRED_0_TOKEN)
            weight_targets.append([0.0] * self.n_experts)
            prediction_targets.append(float(true_labels[step]))
            target_mask.append(True)  # Predict the decision
        
        input_ids.append(self.END_TOKEN)
        weight_targets.append([0.0] * self.n_experts)
        prediction_targets.append(0.0)
        target_mask.append(False)
        
        return {
            'input_ids': input_ids,
            'weight_targets': weight_targets,
            'prediction_targets': prediction_targets,
            'target_mask': target_mask
        }

def generate_mw_training_data(n_sequences: int = 1000, 
                             max_steps: int = 10,
                             n_experts: int = 4) -> List[Dict]:
    """Generate training sequences for multiplicative weights learning."""
    from .multiplicative_weights import MultiplicativeWeights
    
    sequences = []
    
    for _ in range(n_sequences):
        # Random sequence length
        n_steps = np.random.randint(3, max_steps + 1)
        
        # Initialize MW algorithm
        learning_rate = np.random.uniform(0.05, 0.5)
        mw = MultiplicativeWeights(n_experts, learning_rate)
        
        expert_predictions = []
        losses = []
        weights_sequence = [mw.get_probabilities().copy()]
        true_labels = []
        
        # Generate expert qualities (some experts are better than others)
        expert_qualities = np.random.uniform(0.3, 0.9, n_experts)
        
        for step in range(n_steps):
            # Generate true label
            true_label = np.random.randint(0, 2)
            true_labels.append(true_label)
            
            # Generate expert predictions based on their quality
            step_predictions = []
            step_losses = []
            
            for expert_idx in range(n_experts):
                # Expert makes prediction based on quality
                if np.random.random() < expert_qualities[expert_idx]:
                    prediction = true_label
                else:
                    prediction = 1 - true_label
                
                step_predictions.append(prediction)
                
                # Loss is 0 if correct, 1 if incorrect
                loss = 0.0 if prediction == true_label else 1.0
                step_losses.append(loss)
            
            expert_predictions.append(step_predictions)
            losses.append(step_losses)
            
            # Update MW weights
            mw.update_weights(np.array(step_losses))
            weights_sequence.append(mw.get_probabilities().copy())
        
        sequences.append({
            'expert_predictions': expert_predictions,
            'losses': losses,
            'weights_sequence': weights_sequence,
            'true_labels': true_labels,
            'n_steps': n_steps,
            'learning_rate': learning_rate
        })
    
    return sequences

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
        
        # Loss functions
        self.weight_loss_fn = nn.KLDivLoss(reduction='batchmean')
        self.prediction_loss_fn = nn.BCEWithLogitsLoss()
        
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
            
            avg_loss = epoch_loss / n_batches
            stage_losses.append(avg_loss)
            logger.info(f"Stage {stage}, Epoch {epoch}: Loss = {avg_loss:.4f}")
        
        return {'losses': stage_losses}
    
    def _train_batch(self, batch: Dict, stage: int) -> float:
        """Train on a single batch."""
        self.optimizer.zero_grad()
        
        input_ids = batch['input_ids']
        weight_targets = batch['weight_targets']
        prediction_targets = batch['prediction_targets']
        target_mask = batch['target_mask']
        
        # Forward pass
        outputs = self.model(input_ids)
        
        # Calculate losses
        total_loss = 0.0
        
        # Weight prediction loss (only for positions where we have targets)
        # Process sample by sample to avoid shape issues
        batch_size, seq_len = target_mask.shape
        weight_losses = []
        
        for i in range(batch_size):
            sample_mask = target_mask[i]
            if sample_mask.any():
                # Get tensors and ensure consistent lengths
                sample_weight_logits = outputs['weight_logits'][i]  # [seq_len, n_experts]
                sample_weight_targets = weight_targets[i]  # [seq_len, n_experts]
                
                # Find the minimum length to avoid indexing errors
                min_len = min(len(sample_mask), sample_weight_logits.shape[0], sample_weight_targets.shape[0])
                
                # Truncate all to the same length
                sample_mask = sample_mask[:min_len]
                sample_weight_logits = sample_weight_logits[:min_len]
                sample_weight_targets = sample_weight_targets[:min_len]
                
                # Apply mask
                if sample_mask.any():
                    masked_logits = sample_weight_logits[sample_mask]  # [n_targets, n_experts]
                    masked_targets = sample_weight_targets[sample_mask]  # [n_targets, n_experts]
                else:
                    continue  # Skip if no valid targets
                
                # Only process targets that have non-zero weights (valid targets)
                if len(masked_targets) > 0:
                    valid_targets = masked_targets.sum(dim=-1) > 0
                    if valid_targets.any():
                        valid_logits = masked_logits[valid_targets]
                        valid_targets_weights = masked_targets[valid_targets]
                        
                        # Convert to log probabilities for KL divergence
                        weight_log_probs = F.log_softmax(valid_logits, dim=-1)
                        weight_loss = self.weight_loss_fn(weight_log_probs, valid_targets_weights)
                        weight_losses.append(weight_loss)
        
        if weight_losses:
            total_loss += torch.stack(weight_losses).mean()
        
        # Prediction loss - handle shape mismatches
        pred_losses = []
        for i in range(batch_size):
            sample_mask = target_mask[i]
            if sample_mask.any():
                sample_pred_logits = outputs['prediction_logits'][i].squeeze(-1)  # [seq_len]
                sample_pred_targets = prediction_targets[i]  # [seq_len]
                
                # Find minimum length to avoid indexing errors
                min_len = min(len(sample_mask), len(sample_pred_logits), len(sample_pred_targets))
                
                # Truncate all to the same length
                sample_mask = sample_mask[:min_len]
                sample_pred_logits = sample_pred_logits[:min_len]
                sample_pred_targets = sample_pred_targets[:min_len]
                
                # Apply mask
                if sample_mask.any():
                    masked_pred_logits = sample_pred_logits[sample_mask]
                    masked_pred_targets = sample_pred_targets[sample_mask]
                else:
                    continue  # Skip if no valid targets
                
                if len(masked_pred_logits) > 0:
                    pred_loss = self.prediction_loss_fn(masked_pred_logits, masked_pred_targets)
                    pred_losses.append(pred_loss)
        
        if pred_losses:
            total_loss += torch.stack(pred_losses).mean()
        
        # Backward pass
        if total_loss > 0:
            total_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            
            self.optimizer.step()
        
        return total_loss.item()
    
    def _evaluate(self, val_loader: DataLoader) -> Dict:
        """Evaluate the model on validation data."""
        self.model.eval()
        total_loss = 0.0
        n_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids']
                weight_targets = batch['weight_targets']
                prediction_targets = batch['prediction_targets']
                target_mask = batch['target_mask']
                
                outputs = self.model(input_ids)
                
                # Calculate validation loss (same as training logic)
                batch_loss = 0.0
                batch_size, seq_len = target_mask.shape
                
                # Weight losses
                weight_losses = []
                for i in range(batch_size):
                    sample_mask = target_mask[i]
                    if sample_mask.any():
                        sample_weight_logits = outputs['weight_logits'][i]
                        sample_weight_targets = weight_targets[i]
                        
                        # Handle shape mismatches by truncating to minimum length
                        min_len = min(len(sample_mask), sample_weight_logits.shape[0], sample_weight_targets.shape[0])
                        sample_mask = sample_mask[:min_len]
                        sample_weight_logits = sample_weight_logits[:min_len]
                        sample_weight_targets = sample_weight_targets[:min_len]
                        
                        if sample_mask.any():
                            masked_logits = sample_weight_logits[sample_mask]
                            masked_targets = sample_weight_targets[sample_mask]
                        else:
                            continue
                        
                        if len(masked_targets) > 0:
                            valid_targets = masked_targets.sum(dim=-1) > 0
                            if valid_targets.any():
                                valid_logits = masked_logits[valid_targets]
                                valid_targets_weights = masked_targets[valid_targets]
                                weight_log_probs = F.log_softmax(valid_logits, dim=-1)
                                weight_loss = self.weight_loss_fn(weight_log_probs, valid_targets_weights)
                                weight_losses.append(weight_loss)
                
                if weight_losses:
                    batch_loss += torch.stack(weight_losses).mean()
                
                # Prediction losses
                pred_losses = []
                for i in range(batch_size):
                    sample_mask = target_mask[i]
                    if sample_mask.any():
                        sample_pred_logits = outputs['prediction_logits'][i].squeeze(-1)
                        sample_pred_targets = prediction_targets[i]
                        
                        # Handle shape mismatches by truncating to minimum length
                        min_len = min(len(sample_mask), len(sample_pred_logits), len(sample_pred_targets))
                        sample_mask = sample_mask[:min_len]
                        sample_pred_logits = sample_pred_logits[:min_len]
                        sample_pred_targets = sample_pred_targets[:min_len]
                        
                        if sample_mask.any():
                            masked_pred_logits = sample_pred_logits[sample_mask]
                            masked_pred_targets = sample_pred_targets[sample_mask]
                        else:
                            continue
                        
                        if len(masked_pred_logits) > 0:
                            pred_loss = self.prediction_loss_fn(masked_pred_logits, masked_pred_targets)
                            pred_losses.append(pred_loss)
                
                if pred_losses:
                    batch_loss += torch.stack(pred_losses).mean()
                
                total_loss += batch_loss.item()
                n_batches += 1
        
        return {'loss': total_loss / n_batches if n_batches > 0 else 0.0}

def collate_fn(batch):
    """Custom collate function for variable length sequences."""
    # Find max length
    max_len = max(item['sequence_length'] for item in batch)
    
    # Pad sequences
    input_ids = []
    weight_targets = []
    prediction_targets = []
    target_mask = []
    
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
            # If 1D, reshape to [seq_len, n_experts]
            n_experts = 4  # Default, should match model config
            weight_targets_tensor = weight_targets_tensor.unsqueeze(-1).expand(-1, n_experts)
        
        # Pad targets - ensure consistent dimensions
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
    
    return {
        'input_ids': torch.stack(input_ids),
        'weight_targets': torch.stack(weight_targets),
        'prediction_targets': torch.stack(prediction_targets),
        'target_mask': torch.stack(target_mask)
    }
