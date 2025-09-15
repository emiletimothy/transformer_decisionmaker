#!/usr/bin/env python3
"""
Training Script for Learned Multiplicative Weights Transformer

This script implements a multi-stage training strategy similar to the approach
used for teaching transformers algorithmic reasoning tasks like all-pairs reachability.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
import logging
from pathlib import Path
import json
from typing import Dict, List
import argparse

from src.learned_mw_transformer import (
    LearnedMWTransformer, ModelConfig, TrainingConfig, MWTrainer,
    MWSequenceDataset, MWTokenizer, generate_mw_training_data, collate_fn
)
from src.multiplicative_weights import MultiplicativeWeights

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_datasets(n_train: int = 5000, n_val: int = 1000, 
                   max_steps: int = 10, n_experts: int = 4) -> tuple:
    """Create training and validation datasets."""
    logger.info(f"Generating {n_train} training sequences...")
    train_sequences = generate_mw_training_data(n_train, max_steps, n_experts)
    
    logger.info(f"Generating {n_val} validation sequences...")
    val_sequences = generate_mw_training_data(n_val, max_steps, n_experts)
    
    # Create tokenizer
    tokenizer = MWTokenizer(n_experts)
    
    # Create datasets
    train_dataset = MWSequenceDataset(train_sequences, tokenizer)
    val_dataset = MWSequenceDataset(val_sequences, tokenizer)
    
    return train_dataset, val_dataset, tokenizer

def create_stage_datasets(all_sequences: List[Dict], stage: int, 
                         tokenizer: MWTokenizer, mixing_prob: float = 0.1) -> List[Dict]:
    """Create dataset for a specific training stage."""
    stage_sequences = []
    
    for seq in all_sequences:
        # For stage i, we want sequences with at least i steps
        if seq['n_steps'] >= stage:
            # Truncate sequence to stage length for this stage
            truncated_seq = {
                'expert_predictions': seq['expert_predictions'][:stage],
                'losses': seq['losses'][:stage],
                'weights_sequence': seq['weights_sequence'][:stage+1],
                'true_labels': seq['true_labels'][:stage],
                'n_steps': stage,
                'learning_rate': seq['learning_rate']
            }
            stage_sequences.append(truncated_seq)
            
            # Mix in data from previous stages
            if stage > 1 and np.random.random() < mixing_prob:
                for prev_stage in range(1, stage):
                    if seq['n_steps'] >= prev_stage:
                        prev_seq = {
                            'expert_predictions': seq['expert_predictions'][:prev_stage],
                            'losses': seq['losses'][:prev_stage],
                            'weights_sequence': seq['weights_sequence'][:prev_stage+1],
                            'true_labels': seq['true_labels'][:prev_stage],
                            'n_steps': prev_stage,
                            'learning_rate': seq['learning_rate']
                        }
                        stage_sequences.append(prev_seq)
    
    return stage_sequences

def evaluate_learned_model(model: LearnedMWTransformer, tokenizer: MWTokenizer,
                          test_sequences: List[Dict], device: torch.device) -> Dict:
    """Evaluate the learned model against ground truth MW algorithm."""
    model.eval()
    
    results = {
        'weight_mse': [],
        'prediction_accuracy': [],
        'sequence_accuracy': []
    }
    
    with torch.no_grad():
        for seq in test_sequences[:100]:  # Test on subset
            # Get ground truth
            gt_weights = seq['weights_sequence']
            gt_labels = seq['true_labels']
            
            # Encode sequence for model
            tokens = tokenizer.encode_sequence(seq)
            input_ids = torch.tensor([tokens['input_ids']], device=device)
            
            # Get model predictions
            outputs = model(input_ids)
            weight_logits = outputs['weight_logits'][0]  # Remove batch dim
            pred_logits = outputs['prediction_logits'][0]
            
            # Extract predictions at decision points
            target_mask = torch.tensor(tokens['target_mask'])
            weight_targets = torch.tensor(tokens['weight_targets'])
            
            # Calculate metrics
            if target_mask.any():
                # Handle shape mismatches
                min_len = min(len(target_mask), weight_logits.shape[0], weight_targets.shape[0])
                target_mask = target_mask[:min_len]
                weight_logits = weight_logits[:min_len]
                weight_targets = weight_targets[:min_len]
                
                if target_mask.any():
                    # Weight prediction accuracy
                    pred_weights = torch.softmax(weight_logits[target_mask], dim=-1)
                    gt_weights_tensor = weight_targets[target_mask]
                else:
                    continue
                
                # Only compare where we have valid targets
                valid_targets = (gt_weights_tensor.sum(dim=-1) > 0)
                if valid_targets.any():
                    mse = torch.mean((pred_weights[valid_targets] - gt_weights_tensor[valid_targets]) ** 2)
                    results['weight_mse'].append(mse.item())
                
                # Prediction accuracy
                pred_logits = pred_logits[:min_len]  # Ensure same length
                prediction_targets = torch.tensor(tokens['prediction_targets'])[:min_len]
                
                pred_decisions = torch.sigmoid(pred_logits[target_mask]) > 0.5
                gt_decisions = prediction_targets[target_mask] > 0.5
                
                if len(pred_decisions) > 0:
                    accuracy = (pred_decisions == gt_decisions).float().mean()
                    results['prediction_accuracy'].append(accuracy.item())
                    
                    # Sequence-level accuracy
                    seq_correct = (pred_decisions == gt_decisions).all()
                    results['sequence_accuracy'].append(seq_correct.item())
    
    # Aggregate results
    final_results = {}
    for key, values in results.items():
        if values:
            final_results[key] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'count': len(values)
            }
        else:
            final_results[key] = {'mean': 0.0, 'std': 0.0, 'count': 0}
    
    return final_results

def analyze_attention_patterns(model: LearnedMWTransformer, tokenizer: MWTokenizer,
                              test_sequences: List[Dict], device: torch.device,
                              save_path: str = None):
    """Analyze attention patterns in the learned model."""
    model.eval()
    
    # Select a representative sequence
    seq = test_sequences[0]
    tokens = tokenizer.encode_sequence(seq)
    input_ids = torch.tensor([tokens['input_ids']], device=device)
    
    # Get attention weights
    with torch.no_grad():
        outputs = model(input_ids, return_attention=True)
        attention_weights = outputs['attention_weights']
    
    # Plot attention patterns
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Learned MW Transformer: Attention Patterns', fontsize=16)
    
    for layer_idx, layer_attn in enumerate(attention_weights):
        if layer_idx >= 2:
            break
            
        # Average over heads and batch
        attn_matrix = layer_attn[0].mean(dim=0).cpu().numpy()  # [seq_len, seq_len]
        
        # Plot full attention matrix
        ax = axes[layer_idx, 0]
        im = ax.imshow(attn_matrix, cmap='Blues', aspect='auto')
        ax.set_title(f'Layer {layer_idx + 1}: Full Attention Matrix')
        ax.set_xlabel('Key Position')
        ax.set_ylabel('Query Position')
        plt.colorbar(im, ax=ax)
        
        # Plot attention to specific token types
        ax = axes[layer_idx, 1]
        
        # Find positions of different token types
        expert_positions = []
        weight_positions = []
        loss_positions = []
        
        for i, token_id in enumerate(tokens['input_ids']):
            if token_id in tokenizer.EXPERT_TOKENS:
                expert_positions.append(i)
            elif token_id in tokenizer.WEIGHT_TOKENS:
                weight_positions.append(i)
            elif token_id in tokenizer.LOSS_TOKENS:
                loss_positions.append(i)
        
        # Plot attention to different token types over sequence
        if expert_positions:
            expert_attn = attn_matrix[:, expert_positions].mean(axis=1)
            ax.plot(expert_attn, label='Expert Tokens', alpha=0.7)
        
        if weight_positions:
            weight_attn = attn_matrix[:, weight_positions].mean(axis=1)
            ax.plot(weight_attn, label='Weight Tokens', alpha=0.7)
        
        if loss_positions:
            loss_attn = attn_matrix[:, loss_positions].mean(axis=1)
            ax.plot(loss_attn, label='Loss Tokens', alpha=0.7)
        
        ax.set_title(f'Layer {layer_idx + 1}: Attention to Token Types')
        ax.set_xlabel('Query Position')
        ax.set_ylabel('Average Attention')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Attention patterns saved to {save_path}")
    
    plt.show()
    
    return attention_weights

def main():
    parser = argparse.ArgumentParser(description='Train Learned MW Transformer')
    parser.add_argument('--n_experts', type=int, default=4, help='Number of experts')
    parser.add_argument('--max_steps', type=int, default=8, help='Maximum sequence length')
    parser.add_argument('--n_train', type=int, default=3000, help='Number of training sequences')
    parser.add_argument('--n_val', type=int, default=500, help='Number of validation sequences')
    parser.add_argument('--max_stages', type=int, default=6, help='Maximum training stages')
    parser.add_argument('--device', type=str, default='auto', help='Device to use (auto/cpu/cuda)')
    parser.add_argument('--save_dir', type=str, default='../figures', help='Directory to save results')
    
    args = parser.parse_args()
    
    # Set device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    logger.info(f"Using device: {device}")
    
    # Create datasets
    train_dataset, val_dataset, tokenizer = create_datasets(
        args.n_train, args.n_val, args.max_steps, args.n_experts
    )
    
    # Model configuration
    model_config = ModelConfig(
        d_model=768,
        n_heads=8,
        n_layers=2,
        n_experts=args.n_experts,
        vocab_size=tokenizer.vocab_size,
        max_sequence_length=512
    )
    
    # Training configuration
    train_config = TrainingConfig(
        learning_rate=1e-4,
        weight_decay=1e-2,
        batch_size=16,  # Smaller batch size for memory
        max_epochs_per_stage=25,
        max_stages=args.max_stages,
        stage_mixing_prob=0.1
    )
    
    # Create model
    model = LearnedMWTransformer(model_config).to(device)
    logger.info(f"Model has {sum(p.numel() for p in model.parameters())} parameters")
    
    # Create trainer
    trainer = MWTrainer(model, train_config, model_config)
    
    # Get all training sequences for stage-wise training
    all_train_sequences = train_dataset.sequences
    all_val_sequences = val_dataset.sequences
    
    # Multi-stage training
    training_history = []
    
    for stage in range(1, train_config.max_stages + 1):
        logger.info(f"\n{'='*50}")
        logger.info(f"TRAINING STAGE {stage}")
        logger.info(f"{'='*50}")
        
        # Create stage-specific datasets
        stage_train_sequences = create_stage_datasets(
            all_train_sequences, stage, tokenizer, train_config.stage_mixing_prob
        )
        stage_val_sequences = create_stage_datasets(
            all_val_sequences, stage, tokenizer, 0.0  # No mixing for validation
        )
        
        logger.info(f"Stage {stage}: {len(stage_train_sequences)} train, {len(stage_val_sequences)} val sequences")
        
        # Create data loaders
        stage_train_dataset = MWSequenceDataset(stage_train_sequences, tokenizer)
        stage_val_dataset = MWSequenceDataset(stage_val_sequences, tokenizer)
        
        train_loader = DataLoader(
            stage_train_dataset, 
            batch_size=train_config.batch_size,
            shuffle=True,
            collate_fn=collate_fn
        )
        
        val_loader = DataLoader(
            stage_val_dataset,
            batch_size=train_config.batch_size,
            shuffle=False,
            collate_fn=collate_fn
        )
        
        # Train this stage
        stage_results = trainer.train_stage(stage, train_loader, val_loader)
        training_history.append({
            'stage': stage,
            'results': stage_results
        })
        
        # Evaluate after each stage
        if stage % 2 == 0:  # Evaluate every 2 stages
            logger.info(f"Evaluating after stage {stage}...")
            eval_results = evaluate_learned_model(model, tokenizer, all_val_sequences[:100], device)
            logger.info(f"Evaluation results: {eval_results}")
    
    # Final evaluation
    logger.info("\n" + "="*50)
    logger.info("FINAL EVALUATION")
    logger.info("="*50)
    
    # Generate test sequences
    test_sequences = generate_mw_training_data(200, args.max_steps, args.n_experts)
    final_results = evaluate_learned_model(model, tokenizer, test_sequences, device)
    
    logger.info("Final Results:")
    for metric, values in final_results.items():
        logger.info(f"  {metric}: {values['mean']:.4f} ± {values['std']:.4f} (n={values['count']})")
    
    # Analyze attention patterns
    logger.info("\nAnalyzing attention patterns...")
    save_path = os.path.join(args.save_dir, 'learned_mw_attention_patterns.png')
    attention_weights = analyze_attention_patterns(model, tokenizer, test_sequences, device, save_path)
    
    # Save model and results
    save_dir = Path(args.save_dir)
    save_dir.mkdir(exist_ok=True)
    
    # Save model
    model_path = save_dir / 'learned_mw_transformer.pt'
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': model_config,
        'tokenizer_config': {
            'n_experts': tokenizer.n_experts,
            'vocab_size': tokenizer.vocab_size
        },
        'training_history': training_history,
        'final_results': final_results
    }, model_path)
    
    logger.info(f"Model saved to {model_path}")
    
    # Plot training curves
    plt.figure(figsize=(12, 8))
    
    # Plot stage losses
    plt.subplot(2, 2, 1)
    for i, stage_data in enumerate(training_history):
        stage = stage_data['stage']
        losses = stage_data['results']['losses']
        epochs = range(len(losses))
        plt.plot(epochs, losses, label=f'Stage {stage}', alpha=0.7)
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss by Stage')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot evaluation metrics
    metrics = ['weight_mse', 'prediction_accuracy', 'sequence_accuracy']
    for i, metric in enumerate(metrics):
        plt.subplot(2, 2, i + 2)
        if metric in final_results:
            values = [final_results[metric]['mean']]
            errors = [final_results[metric]['std']]
            plt.bar([metric], values, yerr=errors, alpha=0.7)
            plt.title(f'Final {metric.replace("_", " ").title()}')
            plt.ylabel('Score')
    
    plt.tight_layout()
    plt.savefig(save_dir / 'learned_mw_training_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    logger.info(f"\n🎉 Training completed! Results saved to {save_dir}")
    
    # Summary
    print("\n" + "="*60)
    print("🤖 LEARNED MULTIPLICATIVE WEIGHTS TRANSFORMER")
    print("="*60)
    print(f"✅ Model Architecture: {model_config.n_layers} layers, {model_config.d_model} dim, {model_config.n_heads} heads")
    print(f"✅ Training: {train_config.max_stages} stages, {args.n_train} sequences")
    print(f"✅ Final Performance:")
    for metric, values in final_results.items():
        print(f"   • {metric.replace('_', ' ').title()}: {values['mean']:.4f} ± {values['std']:.4f}")
    print(f"✅ Model saved to: {model_path}")
    print("="*60)

if __name__ == "__main__":
    main()
