"""
Transformer implementation of Multiplicative Weights Algorithm

This module implements a 2-layer transformer that realizes multiplicative weights
through attention mechanisms, following the theoretical framework where:
- Layer 1: Loads expert advice, copies weights and labels
- Layer 2: Computes softmax aggregation and weight updates
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from dataclasses import dataclass


@dataclass
class TokenConfig:
    """Configuration for special tokens and embeddings."""
    vocab_size: int = 1000
    d_model: int = 128
    n_experts: int = 4
    
    # Special token IDs
    w_token: int = 0        # <w> state token
    p_query_token: int = 1  # <p?> prediction query
    w_query_token: int = 2  # <w?> weight query
    a_token: int = 3        # <A> aggregation token
    mwu_token: int = 4      # <MWU> update token
    y_token: int = 5        # <y> label token
    
    # Expert tokens start after special tokens
    expert_token_start: int = 10
    
    # Prediction tokens (0 and 1)
    pred_0_token: int = 100
    pred_1_token: int = 101


class MultiplicativeWeightsTransformer(nn.Module):
    """
    2-layer transformer implementing multiplicative weights algorithm.
    
    Architecture:
    - Layer 1: Expert advice loading, weight copying, label copying
    - Layer 2: Softmax aggregation and weight updates
    """
    
    def __init__(self, config: TokenConfig, learning_rate: float = 0.1, beta: float = 1.0):
        super().__init__()
        self.config = config
        self.learning_rate = learning_rate
        self.beta = beta  # Temperature parameter for softmax
        
        # Embeddings
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(1000, config.d_model)  # Max sequence length
        
        # Layer 1 attention heads
        self.layer1_head1 = MultiHeadAttention(config.d_model, 1)  # Expert advice loading
        self.layer1_head2 = MultiHeadAttention(config.d_model, 1)  # Weight copying
        self.layer1_head3 = MultiHeadAttention(config.d_model, 1)  # Label copying
        
        # Layer 2 attention heads  
        self.layer2_head1 = MultiHeadAttention(config.d_model, 1)  # Softmax aggregation
        self.layer2_head2 = MultiHeadAttention(config.d_model, 1)  # Weight updates
        
        # Layer normalization and feed-forward
        self.layer_norm1 = nn.LayerNorm(config.d_model)
        self.layer_norm2 = nn.LayerNorm(config.d_model) 
        self.layer_norm3 = nn.LayerNorm(config.d_model)
        self.layer_norm4 = nn.LayerNorm(config.d_model)
        
        # Output projections
        self.prediction_head = nn.Linear(config.d_model, 1)
        self.weight_head = nn.Linear(config.d_model, config.n_experts)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with small values."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)
    
    def create_input_stream(self, expert_predictions: List[int], weights: torch.Tensor, 
                          label: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Create input stream following the prescribed format:
        <w>W_t <e_1><p_1^t>...<e_n><p_n^t> <y_t> <p?><A><w?><MWU>
        
        Args:
            expert_predictions: List of predictions (0 or 1) for each expert
            weights: Current weight vector for experts
            label: Ground truth label (0 or 1), optional
            
        Returns:
            input_ids: Token sequence
            position_ids: Position indices
        """
        tokens = []
        positions = []
        pos = 0
        
        # State slot: <w> followed by weight representation
        tokens.append(self.config.w_token)
        positions.append(pos)
        pos += 1
        
        # Encode weights as expert tokens (simplified representation)
        for i in range(self.config.n_experts):
            weight_token = self.config.expert_token_start + i
            tokens.append(weight_token)
            positions.append(pos)
            pos += 1
        
        # Advice dictionary: <e_i><p_i> pairs
        for i, pred in enumerate(expert_predictions):
            expert_token = self.config.expert_token_start + i
            pred_token = self.config.pred_1_token if pred == 1 else self.config.pred_0_token
            
            tokens.extend([expert_token, pred_token])
            positions.extend([pos, pos + 1])
            pos += 2
        
        # Label (if provided)
        if label is not None:
            label_token = self.config.pred_1_token if label == 1 else self.config.pred_0_token
            tokens.append(label_token)
            positions.append(pos)
            pos += 1
        
        # Query tokens
        tokens.extend([
            self.config.p_query_token,  # <p?>
            self.config.a_token,        # <A>
            self.config.w_query_token,  # <w?>
            self.config.mwu_token       # <MWU>
        ])
        positions.extend([pos, pos + 1, pos + 2, pos + 3])
        
        return torch.tensor(tokens), torch.tensor(positions)
    
    def forward(self, input_ids: torch.Tensor, position_ids: torch.Tensor, 
               current_weights: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through 2-layer transformer.
        
        Args:
            input_ids: Input token sequence
            position_ids: Position indices
            current_weights: Current expert weights [n_experts]
            
        Returns:
            Dictionary with predictions and updated weights
        """
        batch_size, seq_len = input_ids.shape
        
        # Input embeddings
        token_embeds = self.token_embedding(input_ids)
        pos_embeds = self.position_embedding(position_ids)
        x = token_embeds + pos_embeds  # [batch_size, seq_len, d_model]
        
        # === Layer 1 ===
        
        # Head 1: Load expert predictions
        # Attention from expert tokens to their predictions
        expert_advice = self._layer1_head1_expert_loading(x, input_ids)
        
        # Head 2: Copy weights to prediction positions
        weight_buffer = self._layer1_head2_weight_copying(x, input_ids, current_weights)
        
        # Head 3: Copy labels to weight positions
        label_buffer = self._layer1_head3_label_copying(x, input_ids)
        
        # Combine Layer 1 outputs
        layer1_out = x + expert_advice + weight_buffer + label_buffer
        layer1_out = self.layer_norm1(layer1_out)
        
        # === Layer 2 ===
        
        # Head 1: Softmax aggregation for prediction
        prediction = self._layer2_head1_softmax_aggregation(layer1_out, input_ids, current_weights)
        
        # Head 2: Multiplicative weights update
        updated_weights = self._layer2_head2_weight_update(layer1_out, input_ids, current_weights)
        
        # Combine Layer 2 outputs
        layer2_out = layer1_out + prediction + updated_weights
        layer2_out = self.layer_norm2(layer2_out)
        
        # Extract outputs
        # Find positions of query tokens
        p_query_pos = (input_ids == self.config.p_query_token).nonzero(as_tuple=True)[1]
        w_query_pos = (input_ids == self.config.w_query_token).nonzero(as_tuple=True)[1]
        
        pred_output = None
        weight_output = None
        
        if len(p_query_pos) > 0:
            pred_logit = self.prediction_head(layer2_out[0, p_query_pos[0]])
            pred_output = torch.sigmoid(pred_logit)
        
        if len(w_query_pos) > 0:
            weight_logits = self.weight_head(layer2_out[0, w_query_pos[0]])
            weight_output = F.softmax(weight_logits, dim=-1)
        
        return {
            'prediction': pred_output,
            'weights': weight_output,
            'hidden_state': layer2_out
        }
    
    def _layer1_head1_expert_loading(self, x: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        """Layer 1 Head 1: Load prediction for each expert."""
        # Create attention mask: expert tokens attend to their predictions
        seq_len = x.shape[1]
        attn_output = torch.zeros_like(x)
        
        # Find expert-prediction pairs
        for i in range(self.config.n_experts):
            expert_token = self.config.expert_token_start + i
            expert_positions = (input_ids == expert_token).nonzero(as_tuple=True)[1]
            
            for pos in expert_positions:
                if pos + 1 < seq_len:  # Check if prediction follows
                    pred_pos = pos + 1
                    # Copy prediction embedding to expert position
                    attn_output[0, pos] = x[0, pred_pos] * 0.1  # Small update
        
        return attn_output
    
    def _layer1_head2_weight_copying(self, x: torch.Tensor, input_ids: torch.Tensor, 
                                   weights: torch.Tensor) -> torch.Tensor:
        """Layer 1 Head 2: Copy weights to prediction query position."""
        attn_output = torch.zeros_like(x)
        
        # Find <p?> position and copy weight information
        p_query_pos = (input_ids == self.config.p_query_token).nonzero(as_tuple=True)[1]
        if len(p_query_pos) > 0:
            pos = p_query_pos[0]
            # Embed current weights into the representation
            weight_embed = torch.zeros(self.config.d_model)
            for i in range(min(self.config.n_experts, self.config.d_model)):
                weight_embed[i] = weights[i]
            attn_output[0, pos] = weight_embed
        
        return attn_output
    
    def _layer1_head3_label_copying(self, x: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        """Layer 1 Head 3: Copy label to weight positions."""
        attn_output = torch.zeros_like(x)
        
        # Find label positions (pred_0_token or pred_1_token not part of expert advice)
        label_positions = []
        
        # Skip expert advice pairs to find standalone label
        expert_advice_end = 1 + self.config.n_experts + 2 * self.config.n_experts
        for i in range(expert_advice_end, x.shape[1]):
            if input_ids[0, i] in [self.config.pred_0_token, self.config.pred_1_token]:
                label_positions.append(i)
                break
        
        # Copy label information to <w> position
        w_pos = (input_ids == self.config.w_token).nonzero(as_tuple=True)[1]
        if len(label_positions) > 0 and len(w_pos) > 0:
            label_pos = label_positions[0]
            attn_output[0, w_pos[0]] = x[0, label_pos] * 0.1
        
        return attn_output
    
    def _layer2_head1_softmax_aggregation(self, x: torch.Tensor, input_ids: torch.Tensor,
                                        weights: torch.Tensor) -> torch.Tensor:
        """Layer 2 Head 1: Compute weighted prediction using softmax."""
        attn_output = torch.zeros_like(x)
        
        # Find <p?> position for prediction output
        p_query_pos = (input_ids == self.config.p_query_token).nonzero(as_tuple=True)[1]
        if len(p_query_pos) == 0:
            return attn_output
        
        pos = p_query_pos[0]
        
        # Compute weighted prediction: Σ w_i * p_i
        weighted_pred = 0.0
        total_weight = 0.0
        
        # Find expert predictions from the input stream
        for i in range(self.config.n_experts):
            expert_token = self.config.expert_token_start + i
            expert_positions = (input_ids == expert_token).nonzero(as_tuple=True)[1]
            
            # Look for expert-prediction pairs
            for exp_pos in expert_positions:
                if exp_pos + 1 < x.shape[1]:
                    pred_token = input_ids[0, exp_pos + 1]
                    if pred_token in [self.config.pred_0_token, self.config.pred_1_token]:
                        pred_value = 1.0 if pred_token == self.config.pred_1_token else 0.0
                        weight = weights[i] if i < len(weights) else 0.0
                        weighted_pred += weight * pred_value
                        total_weight += weight
                        break
        
        if total_weight > 0:
            prediction = weighted_pred / total_weight
            # Encode prediction in the hidden state
            pred_encoding = torch.zeros(self.config.d_model)
            pred_encoding[0] = prediction
            attn_output[0, pos] = pred_encoding
        
        return attn_output
    
    def _layer2_head2_weight_update(self, x: torch.Tensor, input_ids: torch.Tensor,
                                  weights: torch.Tensor) -> torch.Tensor:
        """Layer 2 Head 2: Compute multiplicative weights update."""
        attn_output = torch.zeros_like(x)
        
        # Find <MWU> position for weight update output
        mwu_pos = (input_ids == self.config.mwu_token).nonzero(as_tuple=True)[1]
        if len(mwu_pos) == 0:
            return attn_output
        
        pos = mwu_pos[0]
        
        # Find label
        label_value = None
        expert_advice_end = 1 + self.config.n_experts + 2 * self.config.n_experts
        for i in range(expert_advice_end, x.shape[1]):
            if input_ids[0, i] in [self.config.pred_0_token, self.config.pred_1_token]:
                label_value = 1.0 if input_ids[0, i] == self.config.pred_1_token else 0.0
                break
        
        if label_value is None:
            return attn_output
        
        # Compute weight updates: w_i = w_i + η * m_i where m_i = 2 * match - 1
        new_weights = torch.zeros(self.config.n_experts)
        
        for i in range(self.config.n_experts):
            expert_token = self.config.expert_token_start + i
            expert_positions = (input_ids == expert_token).nonzero(as_tuple=True)[1]
            
            # Find prediction for this expert
            pred_value = None
            for exp_pos in expert_positions:
                if exp_pos + 1 < x.shape[1]:
                    pred_token = input_ids[0, exp_pos + 1]
                    if pred_token in [self.config.pred_0_token, self.config.pred_1_token]:
                        pred_value = 1.0 if pred_token == self.config.pred_1_token else 0.0
                        break
            
            if pred_value is not None:
                # Compute stage score: m_i = 2 * (y == p_i) - 1 ∈ {-1, +1}
                match = 1.0 if pred_value == label_value else 0.0
                stage_score = 2.0 * match - 1.0
                
                # Update: λ_i = λ_i + η * m_i
                old_weight = weights[i] if i < len(weights) else 0.0
                new_weights[i] = old_weight + self.learning_rate * stage_score
        
        # Encode updated weights in hidden state
        weight_encoding = torch.zeros(self.config.d_model)
        for i in range(min(self.config.n_experts, self.config.d_model)):
            weight_encoding[i] = new_weights[i]
        attn_output[0, pos] = weight_encoding
        
        return attn_output


class MultiHeadAttention(nn.Module):
    """Simple multi-head attention implementation."""
    
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model) 
        self.value = nn.Linear(d_model, d_model)
        self.output = nn.Linear(d_model, d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = x.shape[:2]
        
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_k)
        attn_weights = F.softmax(scores, dim=-1)
        attended = torch.matmul(attn_weights, V)
        
        # Concatenate heads
        attended = attended.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )
        
        return self.output(attended)
