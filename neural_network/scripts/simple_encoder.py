"""
Simple Encoder — Ground Truth Neural Network

A small autoencoder (2-layer MLP encoder + decoder) trained on synthetic
clustered data. The encoder maps input_dim → latent_dim and serves as the
"oracle" that the transformer must learn to mimic.

Analogous to multiplicative_weights.py in the MW module.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional


@dataclass
class EncoderConfig:
    """Configuration for the simple autoencoder."""
    input_dim: int = 16
    hidden_dim: int = 32
    latent_dim: int = 4
    n_clusters: int = 5
    cluster_std: float = 0.3
    dropout: float = 0.1


class SimpleEncoder(nn.Module):
    """2-layer MLP encoder: input_dim → hidden_dim → latent_dim."""

    def __init__(self, config: EncoderConfig):
        super().__init__()
        self.config = config
        self.net = nn.Sequential(
            nn.Linear(config.input_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.latent_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to latent space. x: [batch, input_dim] → [batch, latent_dim]."""
        return self.net(x)


class SimpleDecoder(nn.Module):
    """2-layer MLP decoder: latent_dim → hidden_dim → input_dim."""

    def __init__(self, config: EncoderConfig):
        super().__init__()
        self.config = config
        self.net = nn.Sequential(
            nn.Linear(config.latent_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.input_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent to input space. z: [batch, latent_dim] → [batch, input_dim]."""
        return self.net(z)


class SimpleAutoencoder(nn.Module):
    """Autoencoder = encoder + decoder, trained with reconstruction loss."""

    def __init__(self, config: EncoderConfig):
        super().__init__()
        self.config = config
        self.encoder = SimpleEncoder(config)
        self.decoder = SimpleDecoder(config)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (reconstruction, latent_code)."""
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Convenience wrapper for encoding only."""
        return self.encoder(x)


# ---------------------------------------------------------------------------
# Synthetic data generation
# ---------------------------------------------------------------------------

def generate_clustered_data(
    n_samples: int,
    config: EncoderConfig,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic data: n_clusters Gaussian blobs in R^input_dim.

    Returns:
        data: [n_samples, input_dim] float32
        labels: [n_samples] int — cluster assignments
    """
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random.RandomState()

    input_dim = config.input_dim
    n_clusters = config.n_clusters
    cluster_std = config.cluster_std

    # Random cluster centers
    centers = rng.randn(n_clusters, input_dim).astype(np.float32)

    # Assign samples to clusters uniformly
    labels = rng.randint(0, n_clusters, size=n_samples)

    # Generate samples around centers
    data = np.zeros((n_samples, input_dim), dtype=np.float32)
    for i in range(n_samples):
        data[i] = centers[labels[i]] + cluster_std * rng.randn(input_dim).astype(np.float32)

    return data, labels


def train_autoencoder(
    config: EncoderConfig,
    n_train: int = 5000,
    n_epochs: int = 50,
    batch_size: int = 128,
    lr: float = 1e-3,
    seed: int = 42,
    device: Optional[torch.device] = None,
) -> SimpleAutoencoder:
    """
    Train the ground truth autoencoder on synthetic clustered data.

    Returns the trained autoencoder (in eval mode).
    """
    if device is None:
        device = torch.device('cpu')

    # Generate training data
    data, labels = generate_clustered_data(n_train, config, seed=seed)
    data_tensor = torch.tensor(data, dtype=torch.float32, device=device)

    # Create model
    model = SimpleAutoencoder(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Training loop
    model.train()
    n_batches = (n_train + batch_size - 1) // batch_size

    for epoch in range(n_epochs):
        perm = torch.randperm(n_train, device=device)
        epoch_loss = 0.0

        for b in range(n_batches):
            start = b * batch_size
            end = min(start + batch_size, n_train)
            idx = perm[start:end]
            x = data_tensor[idx]

            optimizer.zero_grad()
            x_hat, z = model(x)
            loss = F.mse_loss(x_hat, x)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            avg = epoch_loss / n_batches
            print(f"  Autoencoder epoch {epoch+1}/{n_epochs}: loss = {avg:.6f}")

    model.eval()
    return model


def encode_batch(
    encoder: SimpleAutoencoder,
    data: np.ndarray,
    device: Optional[torch.device] = None,
) -> np.ndarray:
    """Run the trained encoder on a numpy array and return latent codes."""
    if device is None:
        device = next(encoder.parameters()).device
    with torch.no_grad():
        x = torch.tensor(data, dtype=torch.float32, device=device)
        z = encoder.encode(x)
    return z.cpu().numpy()
