#!/usr/bin/env python3
"""
Quick Demo of Learned Multiplicative Weights Transformer

This script provides a lightweight demonstration of the learned MW system
without the full training pipeline, useful for testing and quick experiments.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
import matplotlib.pyplot as plt
import logging
from typing import Dict, List

from src.learned_mw_transformer import (
    LearnedMWTransformer, ModelConfig, MWTokenizer, generate_mw_training_data,
    MWSequenceDataset, MWTrainer, TrainingConfig, collate_fn
)
from src.multiplicative_weights import MultiplicativeWeights
from torch.utils.data import DataLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def quick_training_demo(n_experts: int = 4, max_steps: int = 5, 
                       n_sequences: int = 100, n_epochs: int = 10):
    """Run a quick training demo with minimal data."""
    
    print("🚀 Quick Learned MW Transformer Demo")
    print("=" * 50)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Generate small dataset
    print(f"Generating {n_sequences} training sequences...")
    sequences = generate_mw_training_data(n_sequences, max_steps, n_experts)
    
    # Create tokenizer and dataset
    tokenizer = MWTokenizer(n_experts)
    dataset = MWSequenceDataset(sequences, tokenizer)
    
    # Create data loader with batch size 1 to avoid shape issues
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)
    
    # Model configuration (smaller for demo)
    model_config = ModelConfig(
        d_model=256,  # Smaller model
        n_heads=4,
        n_layers=2,
        n_experts=n_experts,
        vocab_size=tokenizer.vocab_size,
        max_sequence_length=128
    )
    
    # Training configuration
    train_config = TrainingConfig(
        learning_rate=1e-3,  # Higher learning rate for quick demo
        weight_decay=1e-2,
        batch_size=1,  # Use batch size 1 to avoid shape issues
        max_epochs_per_stage=n_epochs,
        max_stages=3  # Just 3 stages for demo
    )
    
    # Create model and trainer
    model = LearnedMWTransformer(model_config).to(device)
    trainer = MWTrainer(model, train_config, model_config)
    
    print(f"Model has {sum(p.numel() for p in model.parameters())} parameters")
    
    # Quick training on stage 1 sequences
    print(f"\nTraining for {n_epochs} epochs...")
    stage_sequences = [seq for seq in sequences if seq['n_steps'] >= 3][:50]  # Small subset
    stage_dataset = MWSequenceDataset(stage_sequences, tokenizer)
    stage_loader = DataLoader(stage_dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)
    
    # Train
    losses = []
    for epoch in range(n_epochs):
        epoch_loss = 0.0
        n_batches = 0
        
        model.train()
        for batch in stage_loader:
            loss = trainer._train_batch(batch, stage=1)
            epoch_loss += loss
            n_batches += 1
        
        avg_loss = epoch_loss / n_batches if n_batches > 0 else 0.0
        losses.append(avg_loss)
        print(f"Epoch {epoch + 1}: Loss = {avg_loss:.4f}")
    
    # Test the trained model
    print("\nTesting trained model...")
    test_sequences = generate_mw_training_data(10, max_steps, n_experts)
    
    model.eval()
    with torch.no_grad():
        for i, seq in enumerate(test_sequences[:3]):
            print(f"\nTest Sequence {i + 1}:")
            
            # Ground truth MW
            mw_gt = MultiplicativeWeights(n_experts, seq['learning_rate'])
            gt_weights = [mw_gt.get_probabilities().copy()]
            
            print(f"  Initial weights: {gt_weights[0]}")
            
            # Run ground truth
            for step in range(len(seq['losses'])):
                mw_gt.update_weights(seq['losses'][step])
                gt_weights.append(mw_gt.get_probabilities().copy())
            
            # Get model prediction
            tokens = tokenizer.encode_sequence(seq)
            input_ids = torch.tensor([tokens['input_ids']], device=device)
            outputs = model(input_ids)
            
            # Extract weight predictions
            weight_logits = outputs['weight_logits'][0]
            target_mask = torch.tensor(tokens['target_mask'])
            
            if target_mask.any():
                learned_weights = torch.softmax(weight_logits[target_mask], dim=-1).cpu().numpy()
                
                print(f"  Ground truth final weights: {gt_weights[-1]}")
                if len(learned_weights) > 0:
                    print(f"  Learned final weights:      {learned_weights[-1]}")
                    
                    # Calculate similarity
                    correlation = np.corrcoef(learned_weights[-1], gt_weights[-1])[0, 1]
                    if not np.isnan(correlation):
                        print(f"  Weight correlation: {correlation:.4f}")
    
    # Plot training curve
    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    plt.plot(losses, 'b-', linewidth=2)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    
    # Show a sample attention pattern
    plt.subplot(1, 2, 2)
    if len(test_sequences) > 0:
        seq = test_sequences[0]
        tokens = tokenizer.encode_sequence(seq)
        input_ids = torch.tensor([tokens['input_ids']], device=device)
        
        with torch.no_grad():
            outputs = model(input_ids, return_attention=True)
            if 'attention_weights' in outputs and len(outputs['attention_weights']) > 0:
                # Show attention from last layer, first head
                attn = outputs['attention_weights'][-1][0, 0].cpu().numpy()
                plt.imshow(attn, cmap='Blues', aspect='auto')
                plt.title('Sample Attention Pattern\n(Last Layer, Head 1)')
                plt.xlabel('Key Position')
                plt.ylabel('Query Position')
                plt.colorbar()
    
    plt.tight_layout()
    plt.savefig('../figures/learned_mw_quick_demo.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\n✅ Quick demo completed!")
    print(f"📊 Final training loss: {losses[-1]:.4f}")
    print("📁 Results saved to ../figures/learned_mw_quick_demo.png")
    
    return model, tokenizer, losses

def compare_tokenization_demo():
    """Demonstrate the tokenization process."""
    print("\n🔤 Tokenization Demo")
    print("=" * 30)
    
    # Generate a simple sequence
    sequences = generate_mw_training_data(1, max_steps=3, n_experts=3)
    seq = sequences[0]
    
    print("Original sequence:")
    print(f"  Expert predictions: {seq['expert_predictions']}")
    print(f"  Losses: {seq['losses']}")
    print(f"  True labels: {seq['true_labels']}")
    print(f"  Weight sequence: {[w.round(3) for w in seq['weights_sequence']]}")
    
    # Tokenize
    tokenizer = MWTokenizer(3)
    tokens = tokenizer.encode_sequence(seq)
    
    print(f"\nTokenized sequence:")
    print(f"  Input IDs: {tokens['input_ids'][:20]}...")  # Show first 20 tokens
    print(f"  Total tokens: {len(tokens['input_ids'])}")
    print(f"  Target positions: {sum(tokens['target_mask'])}")
    
    # Show token meanings for first few tokens
    print(f"\nToken meanings (first 10):")
    for i, token_id in enumerate(tokens['input_ids'][:10]):
        if token_id == tokenizer.START_TOKEN:
            meaning = "START"
        elif token_id in tokenizer.EXPERT_TOKENS:
            expert_idx = tokenizer.EXPERT_TOKENS.index(token_id)
            meaning = f"EXPERT_{expert_idx}"
        elif token_id in tokenizer.STEP_TOKENS:
            step_idx = tokenizer.STEP_TOKENS.index(token_id)
            meaning = f"STEP_{step_idx}"
        elif token_id == tokenizer.PRED_0_TOKEN:
            meaning = "PRED_0"
        elif token_id == tokenizer.PRED_1_TOKEN:
            meaning = "PRED_1"
        elif token_id in tokenizer.LOSS_TOKENS:
            loss_idx = tokenizer.LOSS_TOKENS.index(token_id)
            meaning = f"LOSS_{loss_idx/99:.2f}"
        elif token_id in tokenizer.WEIGHT_TOKENS:
            weight_idx = tokenizer.WEIGHT_TOKENS.index(token_id)
            meaning = f"WEIGHT_{weight_idx/99:.2f}"
        else:
            meaning = "OTHER"
        
        print(f"    Token {i}: {token_id} -> {meaning}")

def main():
    """Run the complete quick demo."""
    print("🤖 Learned Multiplicative Weights - Quick Demo")
    print("=" * 60)
    
    # Tokenization demo
    compare_tokenization_demo()
    
    # Quick training demo
    try:
        model, tokenizer, losses = quick_training_demo(
            n_experts=4, 
            max_steps=4, 
            n_sequences=200,  # Small dataset for quick demo
            n_epochs=15
        )
        
        print("\n🎉 Demo completed successfully!")
        print("\nNext steps:")
        print("  • Run full training: python train_learned_mw.py")
        print("  • Analyze results: python analyze_learned_mw.py --model_path ../figures/learned_mw_transformer.pt")
        print("  • See all demos: python ../run_demos.py --list")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        print("This might be due to missing dependencies or insufficient memory.")
        print("Try running: pip install -r ../requirements.txt")

if __name__ == "__main__":
    main()
