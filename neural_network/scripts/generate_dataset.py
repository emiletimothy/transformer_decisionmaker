#!/usr/bin/env python3
"""
Generate encoder training dataset and save to a JSON file.

Produces sequences of (input_vector, latent_code) pairs by running
the trained ground truth encoder on random clustered data.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import json
import argparse
import torch

from simple_encoder import (
    SimpleAutoencoder, EncoderConfig, train_autoencoder,
    generate_clustered_data, encode_batch,
)
from learned_encoder_transformer import generate_encoder_training_data


def main():
    parser = argparse.ArgumentParser(description='Generate encoder training dataset')
    parser.add_argument('--n_train', type=int, default=3000, help='Number of training sequences')
    parser.add_argument('--n_val', type=int, default=500, help='Number of validation sequences')
    parser.add_argument('--n_test', type=int, default=200, help='Number of test sequences')
    parser.add_argument('--max_steps', type=int, default=10, help='Maximum sequence length')
    parser.add_argument('--input_dim', type=int, default=16, help='Input dimensionality')
    parser.add_argument('--latent_dim', type=int, default=4, help='Latent dimensionality')
    parser.add_argument('--hidden_dim', type=int, default=32, help='Encoder hidden dim')
    parser.add_argument('--n_clusters', type=int, default=5, help='Number of clusters')
    parser.add_argument('--cluster_std', type=float, default=0.3, help='Cluster std dev')
    parser.add_argument('--ae_epochs', type=int, default=50, help='Autoencoder training epochs')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--output', type=str, default='../data/encoder_dataset.json',
                        help='Output file path')
    parser.add_argument('--save_encoder', type=str, default='../data/ground_truth_encoder.pt',
                        help='Path to save trained encoder')
    parser.add_argument('--device', type=str, default='auto', help='Device')
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

    print(f"Using device: {device}")

    # Configure and train the ground truth encoder
    enc_config = EncoderConfig(
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        n_clusters=args.n_clusters,
        cluster_std=args.cluster_std,
    )

    print("Training ground truth autoencoder...")
    autoencoder = train_autoencoder(
        enc_config,
        n_train=5000,
        n_epochs=args.ae_epochs,
        seed=args.seed,
        device=device,
    )
    print("Autoencoder trained.")

    # Save encoder
    enc_path = os.path.join(os.path.dirname(__file__), args.save_encoder)
    os.makedirs(os.path.dirname(enc_path), exist_ok=True)
    torch.save({
        'model_state_dict': autoencoder.state_dict(),
        'config': enc_config,
    }, enc_path)
    print(f"Encoder saved to {enc_path}")

    # Generate traces
    print(f"Generating {args.n_train} train, {args.n_val} val, {args.n_test} test sequences ...")
    train = generate_encoder_training_data(
        autoencoder, args.n_train, args.max_steps,
        args.input_dim, args.n_clusters, args.cluster_std, device
    )
    val = generate_encoder_training_data(
        autoencoder, args.n_val, args.max_steps,
        args.input_dim, args.n_clusters, args.cluster_std, device
    )
    test = generate_encoder_training_data(
        autoencoder, args.n_test, args.max_steps,
        args.input_dim, args.n_clusters, args.cluster_std, device
    )

    dataset = {
        'config': {
            'n_train': args.n_train,
            'n_val': args.n_val,
            'n_test': args.n_test,
            'max_steps': args.max_steps,
            'input_dim': args.input_dim,
            'latent_dim': args.latent_dim,
            'hidden_dim': args.hidden_dim,
            'n_clusters': args.n_clusters,
            'cluster_std': args.cluster_std,
            'seed': args.seed,
        },
        'train': train,
        'val': val,
        'test': test,
    }

    out_path = os.path.join(os.path.dirname(__file__), args.output)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    with open(out_path, 'w') as f:
        json.dump(dataset, f)

    size_mb = os.path.getsize(out_path) / (1024 * 1024)
    print(f"Dataset saved to {out_path}  ({size_mb:.1f} MB)")
    print(f"  train: {len(train)} sequences")
    print(f"  val:   {len(val)} sequences")
    print(f"  test:  {len(test)} sequences")

    # Sanity check
    sample = train[0]
    print(f"\nSample sequence (n_steps={sample['n_steps']}):")
    print(f"  inputs[0] (first 4 dims): {sample['inputs'][0][:4]}")
    print(f"  latents[0]:               {sample['latents'][0]}")
    print(f"  labels:                    {sample['labels']}")


if __name__ == '__main__':
    main()
