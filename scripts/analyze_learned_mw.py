#!/usr/bin/env python3
"""
Analysis Script for Learned Multiplicative Weights Transformer

This script provides comprehensive analysis of the learned model including:
- Comparison with ground truth MW algorithm
- Attention pattern visualization
- Generalization to different scenarios
- Ablation studies
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import logging
from typing import Dict, List, Tuple

from src.learned_mw_transformer import (
    LearnedMWTransformer, ModelConfig, MWTokenizer, generate_mw_training_data
)
from src.multiplicative_weights import MultiplicativeWeights

logger = logging.getLogger(__name__)

def load_trained_model(model_path: str, device: torch.device) -> Tuple[LearnedMWTransformer, MWTokenizer, Dict]:
    """Load a trained model and its configuration."""
    checkpoint = torch.load(model_path, map_location=device)
    
    # Reconstruct model
    model_config = checkpoint['model_config']
    model = LearnedMWTransformer(model_config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Reconstruct tokenizer
    tokenizer_config = checkpoint['tokenizer_config']
    tokenizer = MWTokenizer(tokenizer_config['n_experts'])
    
    return model, tokenizer, checkpoint

def compare_with_ground_truth(model: LearnedMWTransformer, tokenizer: MWTokenizer,
                             test_sequences: List[Dict], device: torch.device) -> Dict:
    """Compare learned model predictions with ground truth MW algorithm."""
    model.eval()
    
    results = {
        'weight_correlations': [],
        'weight_mse': [],
        'prediction_agreements': [],
        'regret_ratios': []
    }
    
    with torch.no_grad():
        for seq in test_sequences:
            # Ground truth MW algorithm
            mw_gt = MultiplicativeWeights(len(seq['expert_predictions'][0]), seq['learning_rate'])
            gt_weights_sequence = [mw_gt.get_probabilities().copy()]
            gt_predictions = []
            
            # Run ground truth algorithm
            for step in range(len(seq['expert_predictions'])):
                # Make prediction using current weights
                expert_preds = seq['expert_predictions'][step]
                weighted_pred = sum(w * p for w, p in zip(mw_gt.get_probabilities(), expert_preds))
                gt_predictions.append(1 if weighted_pred > 0.5 else 0)
                
                # Update weights
                mw_gt.update_weights(seq['losses'][step])
                gt_weights_sequence.append(mw_gt.get_probabilities().copy())
            
            # Get learned model predictions
            tokens = tokenizer.encode_sequence(seq)
            input_ids = torch.tensor([tokens['input_ids']], device=device)
            outputs = model(input_ids)
            
            # Extract learned predictions
            weight_logits = outputs['weight_logits'][0]
            pred_logits = outputs['prediction_logits'][0]
            target_mask = torch.tensor(tokens['target_mask'])
            
            if target_mask.any():
                # Weight predictions
                learned_weights = torch.softmax(weight_logits[target_mask], dim=-1).cpu().numpy()
                
                # Compare weights at each step
                for i, (learned_w, gt_w) in enumerate(zip(learned_weights, gt_weights_sequence[1:])):
                    if len(learned_w) == len(gt_w):
                        correlation = np.corrcoef(learned_w, gt_w)[0, 1]
                        if not np.isnan(correlation):
                            results['weight_correlations'].append(correlation)
                        
                        mse = np.mean((learned_w - gt_w) ** 2)
                        results['weight_mse'].append(mse)
                
                # Prediction agreement
                learned_preds = (torch.sigmoid(pred_logits[target_mask]) > 0.5).cpu().numpy()
                if len(learned_preds) == len(gt_predictions):
                    agreement = np.mean(learned_preds == gt_predictions)
                    results['prediction_agreements'].append(agreement)
                
                # Regret comparison (simplified)
                learned_regret = np.sum([seq['losses'][i][np.argmax(learned_weights[i])] 
                                       for i in range(min(len(learned_weights), len(seq['losses'])))])
                gt_regret = np.sum([seq['losses'][i][np.argmax(gt_w)] 
                                  for i, gt_w in enumerate(gt_weights_sequence[1:len(seq['losses'])+1])])
                
                if gt_regret > 0:
                    regret_ratio = learned_regret / gt_regret
                    results['regret_ratios'].append(regret_ratio)
    
    # Aggregate results
    aggregated = {}
    for key, values in results.items():
        if values:
            aggregated[key] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'median': np.median(values),
                'count': len(values)
            }
    
    return aggregated

def analyze_generalization(model: LearnedMWTransformer, tokenizer: MWTokenizer,
                          device: torch.device) -> Dict:
    """Test model generalization to different scenarios."""
    model.eval()
    
    scenarios = {
        'longer_sequences': generate_mw_training_data(100, max_steps=15, n_experts=4),
        'more_experts': generate_mw_training_data(100, max_steps=8, n_experts=6),
        'different_learning_rates': [],
        'adversarial_losses': []
    }
    
    # Generate scenarios with different learning rates
    for lr in [0.01, 0.1, 0.5, 1.0]:
        seqs = generate_mw_training_data(25, max_steps=8, n_experts=4)
        for seq in seqs:
            seq['learning_rate'] = lr
        scenarios['different_learning_rates'].extend(seqs)
    
    # Generate adversarial scenarios
    for _ in range(100):
        seq = generate_mw_training_data(1, max_steps=8, n_experts=4)[0]
        # Make losses more adversarial (higher variance)
        for step_losses in seq['losses']:
            for i in range(len(step_losses)):
                step_losses[i] = np.random.choice([0.0, 1.0], p=[0.3, 0.7])
        scenarios['adversarial_losses'].append(seq)
    
    results = {}
    
    for scenario_name, test_seqs in scenarios.items():
        if scenario_name == 'more_experts' and tokenizer.n_experts != 6:
            # Skip if tokenizer doesn't support more experts
            continue
            
        logger.info(f"Testing generalization: {scenario_name}")
        scenario_results = compare_with_ground_truth(model, tokenizer, test_seqs[:50], device)
        results[scenario_name] = scenario_results
    
    return results

def visualize_attention_evolution(model: LearnedMWTransformer, tokenizer: MWTokenizer,
                                 test_sequence: Dict, device: torch.device,
                                 save_path: str = None):
    """Visualize how attention patterns evolve during a sequence."""
    model.eval()
    
    tokens = tokenizer.encode_sequence(test_sequence)
    input_ids = torch.tensor([tokens['input_ids']], device=device)
    
    with torch.no_grad():
        outputs = model(input_ids, return_attention=True)
        attention_weights = outputs['attention_weights']
    
    # Create visualization
    n_layers = len(attention_weights)
    fig, axes = plt.subplots(n_layers, 2, figsize=(16, 6 * n_layers))
    if n_layers == 1:
        axes = axes.reshape(1, -1)
    
    for layer_idx, layer_attn in enumerate(attention_weights):
        # Average over heads
        attn_matrix = layer_attn[0].mean(dim=0).cpu().numpy()
        
        # Full attention matrix
        ax1 = axes[layer_idx, 0]
        im1 = ax1.imshow(attn_matrix, cmap='Blues', aspect='auto')
        ax1.set_title(f'Layer {layer_idx + 1}: Full Attention Matrix')
        ax1.set_xlabel('Key Position')
        ax1.set_ylabel('Query Position')
        plt.colorbar(im1, ax=ax1)
        
        # Attention to key positions over time
        ax2 = axes[layer_idx, 1]
        
        # Find key token positions
        expert_positions = [i for i, t in enumerate(tokens['input_ids']) if t in tokenizer.EXPERT_TOKENS]
        weight_positions = [i for i, t in enumerate(tokens['input_ids']) if t in tokenizer.WEIGHT_TOKENS]
        loss_positions = [i for i, t in enumerate(tokens['input_ids']) if t in tokenizer.LOSS_TOKENS]
        
        seq_len = len(tokens['input_ids'])
        positions = range(seq_len)
        
        if expert_positions:
            expert_attn = [attn_matrix[i, expert_positions].mean() for i in positions]
            ax2.plot(positions, expert_attn, label='Expert Tokens', linewidth=2)
        
        if weight_positions:
            weight_attn = [attn_matrix[i, weight_positions].mean() for i in positions]
            ax2.plot(positions, weight_attn, label='Weight Tokens', linewidth=2)
        
        if loss_positions:
            loss_attn = [attn_matrix[i, loss_positions].mean() for i in positions]
            ax2.plot(positions, loss_attn, label='Loss Tokens', linewidth=2)
        
        ax2.set_title(f'Layer {layer_idx + 1}: Attention to Token Types')
        ax2.set_xlabel('Query Position')
        ax2.set_ylabel('Average Attention Weight')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Attention evolution saved to {save_path}")
    
    plt.show()

def create_performance_comparison_plot(comparison_results: Dict, generalization_results: Dict,
                                     save_path: str = None):
    """Create comprehensive performance comparison plots."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Learned MW Transformer: Performance Analysis', fontsize=16)
    
    # Main performance metrics
    metrics = ['weight_correlations', 'weight_mse', 'prediction_agreements']
    metric_names = ['Weight Correlation', 'Weight MSE', 'Prediction Agreement']
    
    for i, (metric, name) in enumerate(zip(metrics, metric_names)):
        ax = axes[0, i]
        if metric in comparison_results:
            data = comparison_results[metric]
            ax.bar(['Learned Model'], [data['mean']], yerr=[data['std']], 
                  alpha=0.7, capsize=5)
            ax.set_title(f'{name}\n(Mean ± Std)')
            ax.set_ylabel('Score')
            ax.grid(True, alpha=0.3)
    
    # Generalization results
    gen_metrics = ['weight_correlations', 'prediction_agreements', 'regret_ratios']
    gen_names = ['Weight Correlation', 'Prediction Agreement', 'Regret Ratio']
    
    for i, (metric, name) in enumerate(zip(gen_metrics, gen_names)):
        ax = axes[1, i]
        
        scenarios = []
        means = []
        stds = []
        
        for scenario_name, results in generalization_results.items():
            if metric in results:
                scenarios.append(scenario_name.replace('_', '\n'))
                means.append(results[metric]['mean'])
                stds.append(results[metric]['std'])
        
        if scenarios:
            bars = ax.bar(scenarios, means, yerr=stds, alpha=0.7, capsize=3)
            ax.set_title(f'Generalization: {name}')
            ax.set_ylabel('Score')
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3)
            
            # Color bars based on performance
            for bar, mean in zip(bars, means):
                if metric == 'regret_ratios':
                    # Lower is better for regret ratios
                    color = 'green' if mean < 1.2 else 'orange' if mean < 2.0 else 'red'
                else:
                    # Higher is better for correlations and agreements
                    color = 'green' if mean > 0.8 else 'orange' if mean > 0.6 else 'red'
                bar.set_color(color)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Performance comparison saved to {save_path}")
    
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Analyze Learned MW Transformer')
    parser.add_argument('--model_path', type=str, required=True, 
                       help='Path to trained model checkpoint')
    parser.add_argument('--n_test', type=int, default=200,
                       help='Number of test sequences')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (auto/cpu/cuda)')
    parser.add_argument('--save_dir', type=str, default='../figures',
                       help='Directory to save analysis results')
    
    args = parser.parse_args()
    
    # Set device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    logger.info(f"Using device: {device}")
    
    # Load trained model
    logger.info(f"Loading model from {args.model_path}")
    model, tokenizer, checkpoint = load_trained_model(args.model_path, device)
    
    # Generate test sequences
    logger.info(f"Generating {args.n_test} test sequences...")
    test_sequences = generate_mw_training_data(args.n_test, max_steps=10, 
                                             n_experts=tokenizer.n_experts)
    
    # Compare with ground truth
    logger.info("Comparing with ground truth MW algorithm...")
    comparison_results = compare_with_ground_truth(model, tokenizer, test_sequences, device)
    
    logger.info("Comparison Results:")
    for metric, values in comparison_results.items():
        logger.info(f"  {metric}: {values['mean']:.4f} ± {values['std']:.4f}")
    
    # Test generalization
    logger.info("Testing generalization...")
    generalization_results = analyze_generalization(model, tokenizer, device)
    
    logger.info("Generalization Results:")
    for scenario, results in generalization_results.items():
        logger.info(f"  {scenario}:")
        for metric, values in results.items():
            logger.info(f"    {metric}: {values['mean']:.4f} ± {values['std']:.4f}")
    
    # Visualize attention patterns
    logger.info("Analyzing attention patterns...")
    save_dir = Path(args.save_dir)
    save_dir.mkdir(exist_ok=True)
    
    # Select an interesting test sequence
    interesting_seq = max(test_sequences, key=lambda x: x['n_steps'])
    
    attention_save_path = save_dir / 'learned_mw_attention_evolution.png'
    visualize_attention_evolution(model, tokenizer, interesting_seq, device, 
                                 str(attention_save_path))
    
    # Create performance comparison plot
    comparison_save_path = save_dir / 'learned_mw_performance_comparison.png'
    create_performance_comparison_plot(comparison_results, generalization_results,
                                     str(comparison_save_path))
    
    # Save detailed results
    results_path = save_dir / 'learned_mw_analysis_results.json'
    import json
    
    # Convert numpy types to Python types for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, dict):
            return {key: convert_numpy(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(item) for item in obj]
        else:
            return obj
    
    analysis_results = {
        'comparison_results': convert_numpy(comparison_results),
        'generalization_results': convert_numpy(generalization_results),
        'model_config': convert_numpy(checkpoint['model_config'].__dict__),
        'training_history': convert_numpy(checkpoint.get('training_history', [])),
        'final_training_results': convert_numpy(checkpoint.get('final_results', {}))
    }
    
    with open(results_path, 'w') as f:
        json.dump(analysis_results, f, indent=2)
    
    logger.info(f"Detailed results saved to {results_path}")
    
    # Print summary
    print("\n" + "="*70)
    print("🔍 LEARNED MW TRANSFORMER ANALYSIS SUMMARY")
    print("="*70)
    print(f"📊 Test Sequences: {args.n_test}")
    print(f"🎯 Main Performance:")
    for metric, values in comparison_results.items():
        print(f"   • {metric.replace('_', ' ').title()}: {values['mean']:.4f} ± {values['std']:.4f}")
    
    print(f"\n🌐 Generalization Performance:")
    for scenario, results in generalization_results.items():
        print(f"   • {scenario.replace('_', ' ').title()}:")
        for metric, values in results.items():
            if metric == 'weight_correlations':
                print(f"     - Weight Correlation: {values['mean']:.4f}")
            elif metric == 'prediction_agreements':
                print(f"     - Prediction Agreement: {values['mean']:.4f}")
    
    print(f"\n📁 Results saved to: {save_dir}")
    print("="*70)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
