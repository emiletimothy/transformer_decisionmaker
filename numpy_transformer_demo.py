#!/usr/bin/env python3
"""
Simplified numpy-based transformer implementation for multiplicative weights.
This version avoids PyTorch dependency issues while demonstrating the core concepts.
"""

import numpy as np
import matplotlib.pyplot as plt
from additive_weights import AdditiveMultiplicativeWeights

class NumpyTransformerMW:
    """Simplified transformer that realizes multiplicative weights using numpy operations."""
    
    def __init__(self, n_experts=4, d_model=64, learning_rate=0.1):
        self.n_experts = n_experts
        self.d_model = d_model
        self.learning_rate = learning_rate
        
        # Initialize weights and embeddings
        self.token_embeddings = self._init_embeddings()
        self.position_embeddings = np.random.randn(10, d_model) * 0.1  # Max 10 positions
        
        # Attention weights for different operations
        self.W_advice = np.random.randn(d_model, d_model) * 0.1
        self.W_weights = np.random.randn(d_model, d_model) * 0.1 
        self.W_update = np.random.randn(d_model, d_model) * 0.1
        
        # Current weights (initialized uniformly)
        self.weights = np.ones(n_experts) / n_experts
        
    def _init_embeddings(self):
        """Initialize token embeddings for different token types."""
        embeddings = {}
        # Expert advice tokens
        for i in range(self.n_experts):
            embeddings[f'expert_{i}'] = np.random.randn(self.d_model) * 0.1
        
        # Special tokens
        embeddings['weight'] = np.random.randn(self.d_model) * 0.1
        embeddings['loss'] = np.random.randn(self.d_model) * 0.1
        embeddings['update'] = np.random.randn(self.d_model) * 0.1
        
        return embeddings
    
    def _softmax(self, x, temperature=1.0):
        """Compute softmax with temperature."""
        x = x / temperature
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)
    
    def _attention(self, query, key, value):
        """Simplified attention mechanism."""
        scores = np.dot(query, key.T)
        attention_weights = self._softmax(scores.flatten())
        
        if len(value.shape) == 1:
            return np.dot(attention_weights, value.reshape(1, -1)).flatten()
        else:
            return np.dot(attention_weights, value)
    
    def forward(self, expert_advice, current_weights, loss_feedback=None):
        """
        Forward pass simulating transformer layers.
        
        Args:
            expert_advice: Array of shape (n_experts,) with expert predictions
            current_weights: Array of shape (n_experts,) with current weights
            loss_feedback: Array of shape (n_experts,) with loss per expert (optional)
        """
        
        # Layer 1: Process expert advice and weights
        advice_embeds = []
        for i, advice in enumerate(expert_advice):
            embed = self.token_embeddings[f'expert_{i}'] + advice * 0.1
            advice_embeds.append(embed)
        advice_embeds = np.array(advice_embeds)
        
        weight_embed = self.token_embeddings['weight']
        
        # Attention to load expert advice
        advice_context = self._attention(weight_embed, advice_embeds, advice_embeds)
        
        # Layer 2: Weight updates if loss feedback provided
        if loss_feedback is not None:
            loss_embed = self.token_embeddings['loss']
            update_embed = self.token_embeddings['update']
            
            # Combine context with loss information
            combined_context = advice_context + loss_embed * np.mean(loss_feedback)
            
            # Compute weight updates (multiplicative weights style)
            weight_updates = np.array([
                np.dot(combined_context, self.token_embeddings[f'expert_{i}'])
                for i in range(self.n_experts)
            ])
            
            # Apply multiplicative updates
            new_weights = current_weights * np.exp(-self.learning_rate * loss_feedback)
            new_weights = new_weights / np.sum(new_weights)  # Normalize
            
            return new_weights, weight_updates
        
        # Just return current weights if no loss feedback
        return current_weights, None
    
    def predict(self, expert_advice):
        """Make prediction using current weights."""
        return np.dot(self.weights, expert_advice)
    
    def update(self, expert_advice, true_label):
        """Update weights based on expert performance."""
        prediction = self.predict(expert_advice)
        
        # Compute losses for each expert
        losses = np.abs(expert_advice - true_label)
        
        # Forward pass with loss feedback
        self.weights, _ = self.forward(expert_advice, self.weights, losses)
        
        return prediction, losses

def run_comparison_demo():
    """Run demo comparing numpy transformer with standard additive MW."""
    
    print("🔄 Running Numpy Transformer vs Additive MW Comparison...")
    
    # Parameters
    n_experts = 4
    n_rounds = 200
    
    # Initialize both algorithms
    numpy_transformer = NumpyTransformerMW(n_experts=n_experts, learning_rate=0.1)
    additive_mw = AdditiveMultiplicativeWeights(num_experts=n_experts, learning_rate=0.1, temperature=1.0)
    
    # Track results
    transformer_weights_history = []
    additive_weights_history = []
    transformer_predictions = []
    additive_predictions = []
    transformer_losses = []
    additive_losses = []
    
    # Generate sequence with shifting best expert
    np.random.seed(42)
    
    for t in range(n_rounds):
        # Generate expert advice (expert 2 is best for first half, expert 0 for second half)
        if t < n_rounds // 2:
            best_expert = 2
            expert_advice = np.random.normal([0.3, 0.4, 0.8, 0.2], 0.1)
        else:
            best_expert = 0
            expert_advice = np.random.normal([0.9, 0.3, 0.2, 0.4], 0.1)
        
        # True label is close to best expert's advice
        true_label = expert_advice[best_expert] + np.random.normal(0, 0.05)
        
        # Numpy transformer prediction and update
        transformer_pred, _ = numpy_transformer.update(expert_advice, true_label)
        transformer_loss = abs(transformer_pred - true_label)
        
        # Additive MW prediction and update
        additive_weights = additive_mw.get_probabilities()
        additive_pred = np.dot(additive_weights, expert_advice)
        expert_losses = np.abs(expert_advice - true_label)
        additive_mw.update_weights_from_losses(expert_losses)
        additive_loss = abs(additive_pred - true_label)
        
        # Store results
        transformer_weights_history.append(numpy_transformer.weights.copy())
        additive_weights_history.append(additive_mw.get_probabilities().copy())
        transformer_predictions.append(transformer_pred)
        additive_predictions.append(additive_pred)
        transformer_losses.append(transformer_loss)
        additive_losses.append(additive_loss)
        
        if t % 50 == 0:
            print(f"Round {t}:")
            print(f"  Transformer weights: {numpy_transformer.weights}")
            print(f"  Additive MW weights: {additive_mw.get_probabilities()}")
            print(f"  Best expert: {best_expert}")
    
    # Convert to arrays for plotting
    transformer_weights_history = np.array(transformer_weights_history)
    additive_weights_history = np.array(additive_weights_history)
    
    # Create comparison plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Weight evolution for transformer
    axes[0,0].set_title('Numpy Transformer Weight Evolution')
    for i in range(n_experts):
        axes[0,0].plot(transformer_weights_history[:, i], label=f'Expert {i}')
    axes[0,0].axvline(x=n_rounds//2, color='red', linestyle='--', alpha=0.7, label='Expert shift')
    axes[0,0].set_ylabel('Weight')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # Plot 2: Weight evolution for additive MW
    axes[0,1].set_title('Additive MW Weight Evolution')
    for i in range(n_experts):
        axes[0,1].plot(additive_weights_history[:, i], label=f'Expert {i}')
    axes[0,1].axvline(x=n_rounds//2, color='red', linestyle='--', alpha=0.7, label='Expert shift')
    axes[0,1].set_ylabel('Weight')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    
    # Plot 3: Prediction comparison
    axes[1,0].set_title('Prediction Comparison')
    axes[1,0].plot(transformer_predictions, label='Numpy Transformer', alpha=0.8)
    axes[1,0].plot(additive_predictions, label='Additive MW', alpha=0.8)
    axes[1,0].axvline(x=n_rounds//2, color='red', linestyle='--', alpha=0.7, label='Expert shift')
    axes[1,0].set_xlabel('Round')
    axes[1,0].set_ylabel('Prediction')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    
    # Plot 4: Loss comparison
    axes[1,1].set_title('Loss Comparison')
    axes[1,1].plot(np.cumsum(transformer_losses), label='Numpy Transformer (cumulative)', alpha=0.8)
    axes[1,1].plot(np.cumsum(additive_losses), label='Additive MW (cumulative)', alpha=0.8)
    axes[1,1].axvline(x=n_rounds//2, color='red', linestyle='--', alpha=0.7, label='Expert shift')
    axes[1,1].set_xlabel('Round')
    axes[1,1].set_ylabel('Cumulative Loss')
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('numpy_transformer_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print summary statistics
    transformer_final_loss = np.sum(transformer_losses)
    additive_final_loss = np.sum(additive_losses)
    
    print(f"\n📊 Final Results:")
    print(f"Numpy Transformer - Total Loss: {transformer_final_loss:.4f}")
    print(f"Additive MW - Total Loss: {additive_final_loss:.4f}")
    print(f"Transformer vs Additive MW Loss Ratio: {transformer_final_loss/additive_final_loss:.4f}")
    
    print(f"\nFinal Weights:")
    print(f"Numpy Transformer: {numpy_transformer.weights}")
    print(f"Additive MW: {additive_mw.get_probabilities()}")

if __name__ == "__main__":
    run_comparison_demo()
    print("\n✅ Numpy transformer demo completed successfully!")
