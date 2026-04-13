#!/usr/bin/env python3
"""
Evaluation script for Learned Encoder Transformer.

Loads a trained transformer and ground truth encoder, runs comprehensive
evaluation, and generates visualization plots.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import matplotlib
matplotlib.use('Agg')
import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
import logging
from pathlib import Path

from simple_encoder import SimpleAutoencoder, EncoderConfig
from learned_encoder_transformer import (
    LearnedEncoderTransformer, ContinuousCoTEncoderTransformer,
    ModelConfig, EncoderTokenizer,
    generate_encoder_training_data,
    generate_sequence_with_cot, generate_sequence_with_continuous_cot,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_models(model_path, device):
    """Load trained transformer and ground truth encoder from checkpoint."""
    ckpt = torch.load(model_path, map_location=device, weights_only=False)

    model_config = ckpt['model_config']
    cot_mode = ckpt['cot_mode']
    tok_config = ckpt['tokenizer_config']

    tokenizer = EncoderTokenizer(tok_config['input_dim'], tok_config['latent_dim'])

    if cot_mode == 'continuous':
        model = ContinuousCoTEncoderTransformer(model_config).to(device)
    else:
        model = LearnedEncoderTransformer(model_config).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    # Load encoder
    enc_config = ckpt.get('encoder_config', None)
    autoencoder = None
    if enc_config is not None:
        autoencoder = SimpleAutoencoder(enc_config).to(device)
        # Try loading from same directory
        enc_path = Path(model_path).parent / 'ground_truth_encoder.pt'
        if enc_path.exists():
            enc_ckpt = torch.load(enc_path, map_location=device, weights_only=False)
            autoencoder.load_state_dict(enc_ckpt['model_state_dict'])
            autoencoder.eval()
            logger.info(f"Loaded ground truth encoder from {enc_path}")

    return model, tokenizer, autoencoder, model_config, cot_mode, enc_config


def compute_metrics(pred_latents, gt_latents):
    """Compute MSE, cosine similarity, and per-dimension MSE."""
    n = min(len(pred_latents), len(gt_latents))
    pred = pred_latents[:n]
    gt = gt_latents[:n]

    mse = np.mean((pred - gt) ** 2)

    cos_sims = []
    for s in range(n):
        pn = np.linalg.norm(pred[s])
        gn = np.linalg.norm(gt[s])
        if pn > 1e-8 and gn > 1e-8:
            cos_sims.append(np.dot(pred[s], gt[s]) / (pn * gn))
    cos_sim = np.mean(cos_sims) if cos_sims else 0.0

    per_dim_mse = np.mean((pred - gt) ** 2, axis=0)

    return {'mse': mse, 'cosine_sim': cos_sim, 'per_dim_mse': per_dim_mse}


def run_evaluation(model, tokenizer, autoencoder, test_sequences, device, cot_mode):
    """Full evaluation: latent prediction quality + downstream classification."""
    model.eval()

    all_mse = []
    all_cos = []
    all_per_dim = []
    all_pred = []
    all_gt = []
    all_labels = []

    for seq in test_sequences:
        try:
            if cot_mode == 'continuous':
                out = generate_sequence_with_continuous_cot(model, seq, tokenizer, device)
            else:
                out = generate_sequence_with_cot(model, seq, tokenizer, device)
        except Exception as e:
            logger.warning(f"Generation failed: {e}")
            continue

        pred = out['predicted_latents']
        gt = np.array(seq['latents'], dtype=np.float32)
        n = min(len(pred), len(gt))
        if n == 0:
            continue

        metrics = compute_metrics(pred, gt)
        all_mse.append(metrics['mse'])
        all_cos.append(metrics['cosine_sim'])
        all_per_dim.append(metrics['per_dim_mse'])
        all_pred.append(pred[:n])
        all_gt.append(gt[:n])
        all_labels.extend(seq['labels'][:n])

    results = {
        'mse_mean': float(np.mean(all_mse)),
        'mse_std': float(np.std(all_mse)),
        'cosine_sim_mean': float(np.mean(all_cos)),
        'cosine_sim_std': float(np.std(all_cos)),
        'per_dim_mse_mean': np.mean(all_per_dim, axis=0).tolist() if all_per_dim else [],
        'n_sequences': len(all_mse),
    }

    return results, all_pred, all_gt, all_labels


def plot_eval_results(results, all_pred, all_gt, all_labels, save_dir):
    """Generate comprehensive evaluation plots."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # --- 1. Predicted vs Ground Truth scatter ---
    pred_cat = np.concatenate(all_pred, axis=0)
    gt_cat = np.concatenate(all_gt, axis=0)
    latent_dim = gt_cat.shape[1]

    n_dims = min(latent_dim, 4)
    fig, axes = plt.subplots(1, n_dims, figsize=(4 * n_dims, 4))
    if n_dims == 1:
        axes = [axes]
    fig.suptitle('Predicted vs Ground Truth Latent Codes', fontsize=14)

    for d in range(n_dims):
        ax = axes[d]
        ax.scatter(gt_cat[:, d], pred_cat[:, d], alpha=0.2, s=8)
        lims = [min(gt_cat[:, d].min(), pred_cat[:, d].min()) - 0.5,
                max(gt_cat[:, d].max(), pred_cat[:, d].max()) + 0.5]
        ax.plot(lims, lims, 'r--', alpha=0.5, linewidth=2)
        corr = np.corrcoef(gt_cat[:, d], pred_cat[:, d])[0, 1]
        ax.set_xlabel(f'GT dim {d}')
        ax.set_ylabel(f'Pred dim {d}')
        ax.set_title(f'Dim {d} (r={corr:.3f})')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_dir / 'eval_scatter.png', dpi=300, bbox_inches='tight')
    plt.close()

    # --- 2. Latent space t-SNE/PCA ---
    from sklearn.decomposition import PCA

    if gt_cat.shape[1] > 2:
        pca = PCA(n_components=2)
        gt_2d = pca.fit_transform(gt_cat)
        pred_2d = pca.transform(pred_cat)
    else:
        gt_2d = gt_cat[:, :2]
        pred_2d = pred_cat[:, :2]

    labels_arr = np.array(all_labels[:len(gt_cat)])

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Latent Space (PCA)', fontsize=14)

    scatter = axes[0].scatter(gt_2d[:, 0], gt_2d[:, 1], c=labels_arr, cmap='tab10',
                               alpha=0.5, s=10)
    axes[0].set_title('Ground Truth Encoder')
    axes[0].set_xlabel('PC1')
    axes[0].set_ylabel('PC2')
    axes[0].grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=axes[0])

    scatter = axes[1].scatter(pred_2d[:, 0], pred_2d[:, 1], c=labels_arr, cmap='tab10',
                               alpha=0.5, s=10)
    axes[1].set_title('Transformer Predicted')
    axes[1].set_xlabel('PC1')
    axes[1].set_ylabel('PC2')
    axes[1].grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=axes[1])

    plt.tight_layout()
    plt.savefig(save_dir / 'eval_latent_space.png', dpi=300, bbox_inches='tight')
    plt.close()

    # --- 3. Error distribution ---
    errors = np.sqrt(np.sum((pred_cat - gt_cat) ** 2, axis=1))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle('Prediction Error Analysis', fontsize=14)

    axes[0].hist(errors, bins=50, alpha=0.7, edgecolor='black')
    axes[0].axvline(np.mean(errors), color='r', linestyle='--',
                     label=f'Mean: {np.mean(errors):.3f}')
    axes[0].set_xlabel('L2 Error')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Error Distribution')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    per_dim = results['per_dim_mse_mean']
    if per_dim:
        axes[1].bar(range(len(per_dim)), per_dim, alpha=0.7, color='orange')
        axes[1].set_xlabel('Latent Dimension')
        axes[1].set_ylabel('MSE')
        axes[1].set_title('Per-Dimension MSE')
        axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_dir / 'eval_error_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"Plots saved to {save_dir}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate Learned Encoder Transformer')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--n_test', type=int, default=200, help='Number of test sequences')
    parser.add_argument('--max_steps', type=int, default=10)
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--save_dir', type=str, default='../figures/eval')
    parser.add_argument('--seed', type=int, default=123)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

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

    # Load models
    model, tokenizer, autoencoder, model_config, cot_mode, enc_config = load_models(
        args.model_path, device
    )
    logger.info(f"Loaded model (cot_mode={cot_mode})")

    if autoencoder is None:
        logger.error("Ground truth encoder not found. Cannot generate test data.")
        return

    # Generate test sequences
    logger.info(f"Generating {args.n_test} test sequences...")
    test_sequences = generate_encoder_training_data(
        autoencoder, args.n_test, args.max_steps,
        enc_config.input_dim, enc_config.n_clusters, enc_config.cluster_std, device,
    )

    # Run evaluation
    logger.info("Running evaluation...")
    results, all_pred, all_gt, all_labels = run_evaluation(
        model, tokenizer, autoencoder, test_sequences, device, cot_mode
    )

    # Print results
    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    print(f"  MSE:            {results['mse_mean']:.4f} +/- {results['mse_std']:.4f}")
    print(f"  Cosine Sim:     {results['cosine_sim_mean']:.4f} +/- {results['cosine_sim_std']:.4f}")
    print(f"  Per-dim MSE:    {results['per_dim_mse_mean']}")
    print(f"  N sequences:    {results['n_sequences']}")
    print("=" * 50)

    # Generate plots
    if all_pred and all_gt:
        plot_eval_results(results, all_pred, all_gt, all_labels, args.save_dir)


if __name__ == '__main__':
    main()
