#!/usr/bin/env python3
"""
Training Script for Learned Encoder Transformer

Multi-stage curriculum training strategy: at stage i the transformer learns
to predict latent codes for sequences of i encoding steps.

Supports both discrete (token-level) and continuous (Coconut-style) CoT modes.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import matplotlib
matplotlib.use('Agg')
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import logging
from pathlib import Path
import json
from typing import Dict, List
import argparse

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

from simple_encoder import SimpleAutoencoder, EncoderConfig, train_autoencoder
from learned_encoder_transformer import (
    LearnedEncoderTransformer, ContinuousCoTEncoderTransformer,
    ModelConfig, TrainingConfig,
    EncoderTrainer, ContinuousCoTEncoderTrainer,
    EncoderTokenizer, EncoderSequenceDataset, collate_fn,
    generate_encoder_training_data,
    generate_sequence_with_cot, generate_sequence_with_continuous_cot,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def create_datasets(encoder_model, n_train, n_val, max_steps, input_dim,
                    latent_dim, n_clusters, cluster_std, device):
    """Create training and validation datasets from encoder traces."""
    logger.info(f"Generating {n_train} training sequences...")
    train_sequences = generate_encoder_training_data(
        encoder_model, n_train, max_steps, input_dim, n_clusters, cluster_std, device
    )

    logger.info(f"Generating {n_val} validation sequences...")
    val_sequences = generate_encoder_training_data(
        encoder_model, n_val, max_steps, input_dim, n_clusters, cluster_std, device
    )

    tokenizer = EncoderTokenizer(input_dim, latent_dim)
    train_dataset = EncoderSequenceDataset(train_sequences, tokenizer)
    val_dataset = EncoderSequenceDataset(val_sequences, tokenizer)

    return train_dataset, val_dataset, tokenizer


def create_stage_datasets(all_sequences, stage, tokenizer, mixing_prob=0.1):
    """Create dataset for a specific training stage."""
    stage_sequences = []

    for seq in all_sequences:
        if seq['n_steps'] >= stage:
            truncated_seq = {
                'inputs': seq['inputs'][:stage],
                'latents': seq['latents'][:stage],
                'labels': seq['labels'][:stage],
                'n_steps': stage,
            }
            stage_sequences.append(truncated_seq)

            if stage > 1 and np.random.random() < mixing_prob:
                for prev_stage in range(1, stage):
                    if seq['n_steps'] >= prev_stage:
                        prev_seq = {
                            'inputs': seq['inputs'][:prev_stage],
                            'latents': seq['latents'][:prev_stage],
                            'labels': seq['labels'][:prev_stage],
                            'n_steps': prev_stage,
                        }
                        stage_sequences.append(prev_seq)

    return stage_sequences


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_learned_model(model, tokenizer, test_sequences, device, mode='discrete'):
    """Evaluate the learned model against ground truth encoder outputs."""
    model.eval()

    results = {
        'mse': [],
        'cosine_sim': [],
        'per_dim_mse': [],
    }

    for seq in test_sequences[:100]:
        try:
            if mode == 'continuous':
                out = generate_sequence_with_continuous_cot(model, seq, tokenizer, device)
            else:
                out = generate_sequence_with_cot(model, seq, tokenizer, device)
        except Exception as e:
            logger.warning(f"Generation failed: {e}")
            continue

        pred_latents = out['predicted_latents']
        gt_latents = np.array(seq['latents'], dtype=np.float32)
        n_steps = min(len(pred_latents), len(gt_latents))

        if n_steps > 0:
            pred = pred_latents[:n_steps]
            gt = gt_latents[:n_steps]

            mse = np.mean((pred - gt) ** 2)
            results['mse'].append(mse)

            # Cosine similarity per step
            cos_sims = []
            for s in range(n_steps):
                p_norm = np.linalg.norm(pred[s])
                g_norm = np.linalg.norm(gt[s])
                if p_norm > 1e-8 and g_norm > 1e-8:
                    cos_sim = np.dot(pred[s], gt[s]) / (p_norm * g_norm)
                    cos_sims.append(cos_sim)
            if cos_sims:
                results['cosine_sim'].append(np.mean(cos_sims))

            # Per-dimension MSE
            per_dim = np.mean((pred - gt) ** 2, axis=0)
            results['per_dim_mse'].append(per_dim)

    final_results = {}
    for key, values in results.items():
        if values:
            if key == 'per_dim_mse':
                arr = np.array(values)
                final_results[key] = {
                    'mean': arr.mean(axis=0).tolist(),
                    'std': arr.std(axis=0).tolist(),
                    'count': len(values),
                }
            else:
                final_results[key] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'count': len(values),
                }
        else:
            final_results[key] = {'mean': 0.0, 'std': 0.0, 'count': 0}

    return final_results


def evaluate_downstream_classification(model, tokenizer, encoder_model,
                                       test_sequences, device, mode='discrete'):
    """
    Downstream eval: use predicted latents for nearest-centroid cluster assignment.
    Compare accuracy using transformer latents vs ground truth encoder latents.
    """
    model.eval()
    encoder_model.eval()

    gt_correct = 0
    pred_correct = 0
    total = 0

    for seq in test_sequences[:50]:
        try:
            if mode == 'continuous':
                out = generate_sequence_with_continuous_cot(model, seq, tokenizer, device)
            else:
                out = generate_sequence_with_cot(model, seq, tokenizer, device)
        except Exception:
            continue

        pred_latents = out['predicted_latents']
        gt_latents = np.array(seq['latents'], dtype=np.float32)
        true_labels = np.array(seq['labels'])
        n_steps = min(len(pred_latents), len(gt_latents), len(true_labels))

        if n_steps < 2:
            continue

        # Build centroids from first half, classify second half
        mid = max(n_steps // 2, 1)
        n_clusters = max(true_labels[:mid]) + 1

        # Ground truth centroids
        gt_centroids = np.zeros((n_clusters, gt_latents.shape[1]))
        gt_counts = np.zeros(n_clusters)
        pred_centroids = np.zeros((n_clusters, pred_latents.shape[1]))
        pred_counts = np.zeros(n_clusters)

        for s in range(mid):
            c = true_labels[s]
            if c < n_clusters:
                gt_centroids[c] += gt_latents[s]
                gt_counts[c] += 1
                pred_centroids[c] += pred_latents[s]
                pred_counts[c] += 1

        for c in range(n_clusters):
            if gt_counts[c] > 0:
                gt_centroids[c] /= gt_counts[c]
            if pred_counts[c] > 0:
                pred_centroids[c] /= pred_counts[c]

        # Classify second half
        for s in range(mid, n_steps):
            true_c = true_labels[s]
            if true_c >= n_clusters:
                continue

            # GT encoder assignment
            gt_dists = np.linalg.norm(gt_centroids - gt_latents[s], axis=1)
            gt_pred_c = np.argmin(gt_dists)
            if gt_pred_c == true_c:
                gt_correct += 1

            # Transformer assignment
            pred_dists = np.linalg.norm(pred_centroids - pred_latents[s], axis=1)
            pred_pred_c = np.argmin(pred_dists)
            if pred_pred_c == true_c:
                pred_correct += 1

            total += 1

    return {
        'gt_accuracy': gt_correct / max(total, 1),
        'pred_accuracy': pred_correct / max(total, 1),
        'total_samples': total,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_training_results(training_history, final_results, save_dir):
    """Plot training curves and evaluation metrics."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Learned Encoder Transformer: Training Results', fontsize=14)

    # Plot 1: Stage losses
    ax = axes[0, 0]
    for stage_data in training_history:
        stage = stage_data['stage']
        losses = stage_data['results']['losses']
        ax.plot(range(len(losses)), losses, label=f'Stage {stage}', alpha=0.7)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss (MSE)')
    ax.set_title('Training Loss by Stage')
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    # Plot 2: MSE metric
    ax = axes[0, 1]
    if 'mse' in final_results and final_results['mse']['count'] > 0:
        ax.bar(['MSE'], [final_results['mse']['mean']],
               yerr=[final_results['mse']['std']], alpha=0.7, capsize=5)
        ax.set_title(f"Latent MSE: {final_results['mse']['mean']:.4f}")
    ax.set_ylabel('MSE')
    ax.grid(True, alpha=0.3)

    # Plot 3: Cosine similarity
    ax = axes[1, 0]
    if 'cosine_sim' in final_results and final_results['cosine_sim']['count'] > 0:
        ax.bar(['Cosine Sim'], [final_results['cosine_sim']['mean']],
               yerr=[final_results['cosine_sim']['std']], alpha=0.7, color='green', capsize=5)
        ax.set_title(f"Cosine Similarity: {final_results['cosine_sim']['mean']:.4f}")
    ax.set_ylabel('Cosine Similarity')
    ax.grid(True, alpha=0.3)

    # Plot 4: Per-dimension MSE
    ax = axes[1, 1]
    if 'per_dim_mse' in final_results and final_results['per_dim_mse']['count'] > 0:
        means = final_results['per_dim_mse']['mean']
        stds = final_results['per_dim_mse']['std']
        dims = range(len(means))
        ax.bar(dims, means, yerr=stds, alpha=0.7, color='orange', capsize=3)
        ax.set_xlabel('Latent Dimension')
        ax.set_ylabel('MSE')
        ax.set_title('Per-Dimension MSE')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, 'training_results.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Training results plot saved to {path}")


def plot_latent_comparison(model, tokenizer, encoder_model, test_sequences,
                           device, save_dir, mode='discrete'):
    """Scatter plot of predicted vs ground truth latent codes."""
    model.eval()

    all_pred = []
    all_gt = []

    for seq in test_sequences[:30]:
        try:
            if mode == 'continuous':
                out = generate_sequence_with_continuous_cot(model, seq, tokenizer, device)
            else:
                out = generate_sequence_with_cot(model, seq, tokenizer, device)
        except Exception:
            continue

        pred_latents = out['predicted_latents']
        gt_latents = np.array(seq['latents'], dtype=np.float32)
        n_steps = min(len(pred_latents), len(gt_latents))

        if n_steps > 0:
            all_pred.append(pred_latents[:n_steps])
            all_gt.append(gt_latents[:n_steps])

    if not all_pred:
        return

    all_pred = np.concatenate(all_pred, axis=0)
    all_gt = np.concatenate(all_gt, axis=0)
    latent_dim = all_gt.shape[1]

    n_dims = min(latent_dim, 4)
    fig, axes = plt.subplots(1, n_dims, figsize=(4 * n_dims, 4))
    if n_dims == 1:
        axes = [axes]
    fig.suptitle('Predicted vs Ground Truth Latent Codes', fontsize=14)

    for d in range(n_dims):
        ax = axes[d]
        ax.scatter(all_gt[:, d], all_pred[:, d], alpha=0.3, s=10)
        lims = [min(all_gt[:, d].min(), all_pred[:, d].min()) - 0.5,
                max(all_gt[:, d].max(), all_pred[:, d].max()) + 0.5]
        ax.plot(lims, lims, 'r--', alpha=0.5)
        ax.set_xlabel(f'GT dim {d}')
        ax.set_ylabel(f'Pred dim {d}')
        ax.set_title(f'Dim {d}')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, 'latent_comparison.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Latent comparison plot saved to {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Train Learned Encoder Transformer')
    parser.add_argument('--input_dim', type=int, default=16)
    parser.add_argument('--latent_dim', type=int, default=4)
    parser.add_argument('--hidden_dim', type=int, default=32)
    parser.add_argument('--n_clusters', type=int, default=5)
    parser.add_argument('--cluster_std', type=float, default=0.3)
    parser.add_argument('--max_steps', type=int, default=10)
    parser.add_argument('--n_train', type=int, default=3000)
    parser.add_argument('--n_val', type=int, default=500)
    parser.add_argument('--max_stages', type=int, default=10)
    parser.add_argument('--ae_epochs', type=int, default=50)
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--cot_mode', type=str, default='continuous',
                        choices=['discrete', 'continuous'])
    parser.add_argument('--n_thought_steps', type=int, default=4)
    parser.add_argument('--save_dir', type=str, default='../figures')
    parser.add_argument('--wandb_project', type=str, default=None)
    parser.add_argument('--wandb_entity', type=str, default=None)
    parser.add_argument('--wandb_run_name', type=str, default=None)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Device
    if args.device == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
    else:
        device = torch.device(args.device)
    logger.info(f"Using device: {device}")

    # Initialize wandb
    use_wandb = HAS_WANDB and args.wandb_project is not None
    if use_wandb:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name,
            config=vars(args),
        )
        logger.info(f"wandb initialized: {wandb.run.url}")

    # --- Train ground truth encoder ---
    enc_config = EncoderConfig(
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        n_clusters=args.n_clusters,
        cluster_std=args.cluster_std,
    )

    logger.info("Training ground truth autoencoder...")
    autoencoder = train_autoencoder(
        enc_config, n_train=5000, n_epochs=args.ae_epochs,
        seed=args.seed, device=device,
    )
    logger.info("Ground truth autoencoder trained.")

    # --- Create datasets ---
    train_dataset, val_dataset, tokenizer = create_datasets(
        autoencoder, args.n_train, args.n_val, args.max_steps,
        args.input_dim, args.latent_dim, args.n_clusters, args.cluster_std, device,
    )

    # --- Model configuration ---
    model_config = ModelConfig(
        d_model=256,
        n_heads=4,
        n_layers=2,
        input_dim=args.input_dim,
        latent_dim=args.latent_dim,
        vocab_size=tokenizer.vocab_size,
        max_sequence_length=1024,
        n_thought_steps=args.n_thought_steps if args.cot_mode == 'continuous' else 0,
    )

    train_config = TrainingConfig(
        learning_rate=3e-4,
        weight_decay=1e-2,
        batch_size=32,
        max_epochs_per_stage=30,
        early_stopping_patience=12,
        max_stages=args.max_stages,
        stage_mixing_prob=0.1,
    )

    # --- Create model and trainer ---
    if args.cot_mode == 'continuous':
        model = ContinuousCoTEncoderTransformer(model_config).to(device)
        logger.info(f"ContinuousCoTEncoderTransformer with {args.n_thought_steps} thought steps")
        trainer = ContinuousCoTEncoderTrainer(model, train_config, model_config)
    else:
        model = LearnedEncoderTransformer(model_config).to(device)
        trainer = EncoderTrainer(model, train_config, model_config)

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model has {n_params} parameters")

    if use_wandb:
        wandb.config.update({
            'd_model': model_config.d_model,
            'n_heads': model_config.n_heads,
            'n_layers': model_config.n_layers,
            'n_params': n_params,
        })

    all_train_sequences = train_dataset.sequences
    all_val_sequences = val_dataset.sequences

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # --- Multi-stage training ---
    training_history = []
    stage_evaluations = {}

    for stage in range(1, train_config.max_stages + 1):
        logger.info(f"\n{'='*50}")
        logger.info(f"TRAINING STAGE {stage}")
        logger.info(f"{'='*50}")

        stage_train_seqs = create_stage_datasets(
            all_train_sequences, stage, tokenizer, train_config.stage_mixing_prob
        )
        stage_val_seqs = create_stage_datasets(
            all_val_sequences, stage, tokenizer, 0.0
        )

        logger.info(f"Stage {stage}: {len(stage_train_seqs)} train, {len(stage_val_seqs)} val sequences")

        stage_train_dataset = EncoderSequenceDataset(stage_train_seqs, tokenizer)
        stage_val_dataset = EncoderSequenceDataset(stage_val_seqs, tokenizer)

        train_loader = DataLoader(
            stage_train_dataset,
            batch_size=train_config.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
        )
        val_loader = DataLoader(
            stage_val_dataset,
            batch_size=train_config.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
        )

        stage_results = trainer.train_stage(stage, train_loader, val_loader)
        training_history.append({'stage': stage, 'results': stage_results})

        # Checkpoint
        checkpoint_dir = save_dir / 'checkpoints'
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = checkpoint_dir / f'model_stage_{stage}.pt'
        torch.save({
            'stage': stage,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': trainer.optimizer.state_dict(),
            'model_config': model_config,
            'cot_mode': args.cot_mode,
            'tokenizer_config': {
                'input_dim': tokenizer.input_dim,
                'latent_dim': tokenizer.latent_dim,
                'vocab_size': tokenizer.vocab_size,
            },
            'training_history': training_history,
            'step': trainer.step,
        }, ckpt_path)
        logger.info(f"Checkpoint saved: {ckpt_path}")

        # Stage evaluation
        logger.info(f"Evaluating after stage {stage}...")
        stage_eval_seqs = [
            {k: (v[:stage] if isinstance(v, list) else (stage if k == 'n_steps' else v))
             for k, v in seq.items()}
            for seq in all_val_sequences[:100] if seq['n_steps'] >= stage
        ]

        eval_results = evaluate_learned_model(
            model, tokenizer, stage_eval_seqs[:50], device, mode=args.cot_mode
        )
        stage_evaluations[f'stage_{stage}'] = eval_results

        logger.info(f"Stage {stage} Results:")
        for metric, values in eval_results.items():
            if isinstance(values.get('mean'), list):
                logger.info(f"  {metric}: {values['mean']}")
            else:
                logger.info(f"  {metric}: {values['mean']:.4f} +/- {values['std']:.4f}")

        if use_wandb:
            wandb_log = {'stage': stage}
            stage_losses = stage_results['losses']
            if stage_losses:
                wandb_log['train/epoch_loss_final'] = stage_losses[-1]
                wandb_log['train/epoch_loss_best'] = min(stage_losses)
            for metric, values in eval_results.items():
                if not isinstance(values.get('mean'), list):
                    wandb_log[f'eval/{metric}'] = values['mean']
            wandb.log(wandb_log)

    # --- Final evaluation ---
    logger.info("\n" + "=" * 50)
    logger.info("FINAL EVALUATION")
    logger.info("=" * 50)

    test_sequences = generate_encoder_training_data(
        autoencoder, 200, args.max_steps,
        args.input_dim, args.n_clusters, args.cluster_std, device,
    )

    final_results = evaluate_learned_model(
        model, tokenizer, test_sequences, device, mode=args.cot_mode
    )
    logger.info("Final Results:")
    for metric, values in final_results.items():
        if isinstance(values.get('mean'), list):
            logger.info(f"  {metric}: {values['mean']}")
        else:
            logger.info(f"  {metric}: {values['mean']:.4f} +/- {values['std']:.4f} (n={values['count']})")

    # Downstream classification
    logger.info("\nDownstream classification evaluation...")
    class_results = evaluate_downstream_classification(
        model, tokenizer, autoencoder, test_sequences, device, mode=args.cot_mode
    )
    logger.info(f"  GT encoder accuracy:    {class_results['gt_accuracy']:.4f}")
    logger.info(f"  Transformer accuracy:   {class_results['pred_accuracy']:.4f}")
    logger.info(f"  Total samples:          {class_results['total_samples']}")

    # --- Plots ---
    plot_training_results(training_history, final_results, str(save_dir))
    plot_latent_comparison(model, tokenizer, autoencoder, test_sequences,
                           device, str(save_dir), mode=args.cot_mode)

    # --- Save model ---
    model_path = save_dir / 'learned_encoder_transformer.pt'
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': model_config,
        'encoder_config': enc_config,
        'cot_mode': args.cot_mode,
        'tokenizer_config': {
            'input_dim': tokenizer.input_dim,
            'latent_dim': tokenizer.latent_dim,
            'vocab_size': tokenizer.vocab_size,
        },
        'training_history': training_history,
        'final_results': final_results,
        'classification_results': class_results,
    }, model_path)
    logger.info(f"Model saved to {model_path}")

    # Save autoencoder alongside
    ae_path = save_dir / 'ground_truth_encoder.pt'
    torch.save({
        'model_state_dict': autoencoder.state_dict(),
        'config': enc_config,
    }, ae_path)

    if use_wandb:
        final_wandb = {}
        for metric, values in final_results.items():
            if not isinstance(values.get('mean'), list):
                final_wandb[f'final/{metric}'] = values['mean']
        final_wandb['final/gt_classification_accuracy'] = class_results['gt_accuracy']
        final_wandb['final/pred_classification_accuracy'] = class_results['pred_accuracy']
        wandb.log(final_wandb)
        wandb.finish()

    # Summary
    print("\n" + "=" * 60)
    print("LEARNED ENCODER TRANSFORMER")
    print("=" * 60)
    print(f"  Model: {model_config.n_layers} layers, {model_config.d_model} dim, {model_config.n_heads} heads")
    print(f"  CoT mode: {args.cot_mode}")
    print(f"  Training: {train_config.max_stages} stages, {args.n_train} sequences")
    print(f"  Encoder: {args.input_dim}d -> {args.latent_dim}d latent")
    print(f"  Final Performance:")
    for metric, values in final_results.items():
        if isinstance(values.get('mean'), list):
            print(f"    {metric}: {values['mean']}")
        else:
            print(f"    {metric}: {values['mean']:.4f} +/- {values['std']:.4f}")
    print(f"  Downstream Classification:")
    print(f"    GT encoder:    {class_results['gt_accuracy']:.4f}")
    print(f"    Transformer:   {class_results['pred_accuracy']:.4f}")
    print(f"  Model saved to: {model_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
