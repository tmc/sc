"""
Train TopK SAE on Sudoku reasoning activations.

Discovers interpretable reasoning states from the iteration dynamics
of the Sudoku statechart model.
"""

import os
import json
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

from .extract_activations import ActivationDataset


@dataclass
class ReasoningSAEConfig:
    """Configuration for reasoning state SAE."""
    # Input dimension (from board state + context + guards)
    board_dim: int = 10      # 10 states per cell
    context_dim: int = 128   # Hidden dim from model
    guard_dim: int = 9       # 9 guards per cell

    # SAE architecture
    expansion_factor: int = 16  # Overcomplete factor
    k_active: int = 16          # TopK active features
    l1_coefficient: float = 1e-3  # Sparsity penalty

    # Training
    batch_size: int = 256
    learning_rate: float = 1e-4
    epochs: int = 50

    # Dead feature handling
    dead_threshold: float = 1e-6
    resample_dead: bool = True
    resample_interval: int = 1000

    @property
    def input_dim(self) -> int:
        return self.board_dim + self.context_dim + self.guard_dim

    @property
    def latent_dim(self) -> int:
        return self.input_dim * self.expansion_factor


class TopKSAE(nn.Module):
    """
    TopK Sparse Autoencoder for reasoning state discovery.

    Guarantees exactly k features are active, making it easier
    to identify discrete reasoning states.
    """

    def __init__(self, config: ReasoningSAEConfig):
        super().__init__()
        self.config = config
        self.latent_dim = config.latent_dim

        # Encoder: project to overcomplete space
        self.encoder = nn.Linear(config.input_dim, self.latent_dim)

        # Decoder: reconstruct from sparse code
        self.decoder = nn.Linear(self.latent_dim, config.input_dim, bias=False)

    def encode(self, x: mx.array) -> Tuple[mx.array, mx.array]:
        """
        Encode input to sparse TopK representation.

        Args:
            x: Input tensor [batch, input_dim]

        Returns:
            acts: Sparse activations [batch, latent_dim]
            indices: Active feature indices [batch, k]
        """
        # Project to latent space
        pre_acts = self.encoder(x)  # [batch, latent_dim]

        # TopK selection
        k = self.config.k_active
        batch_size = x.shape[0]

        # Get top-k indices
        sorted_indices = mx.argsort(-pre_acts, axis=-1)
        top_indices = sorted_indices[:, :k]

        # Gather top-k values
        top_values = mx.take_along_axis(pre_acts, top_indices, axis=-1)
        top_values = mx.maximum(top_values, 0.0)  # ReLU

        # Create sparse output using scatter
        acts = mx.zeros((batch_size, self.latent_dim))

        # Scatter top_values into acts at top_indices positions
        # Using numpy for efficient scatter (MLX lacks native scatter)
        acts_np = np.zeros((batch_size, self.latent_dim), dtype=np.float32)
        top_indices_np = np.array(top_indices.tolist())
        top_values_np = np.array(top_values.tolist())

        for b in range(batch_size):
            acts_np[b, top_indices_np[b]] = top_values_np[b]

        acts = mx.array(acts_np)

        return acts, top_indices

    def decode(self, acts: mx.array) -> mx.array:
        """Decode sparse representation."""
        return self.decoder(acts)

    def __call__(self, x: mx.array) -> Tuple[mx.array, mx.array, mx.array]:
        """
        Full forward pass.

        Returns:
            reconstruction: Reconstructed input
            acts: Sparse activations
            indices: Active feature indices
        """
        acts, indices = self.encode(x)
        recon = self.decode(acts)
        return recon, acts, indices


class ReasoningSAE:
    """
    SAE for discovering reasoning states in Sudoku solving.

    Wraps TopKSAE with:
    - Feature tracking and statistics
    - Dead feature resampling
    - Clustering for state discovery
    """

    def __init__(self, config: ReasoningSAEConfig):
        self.config = config
        self.sae = TopKSAE(config)

        # Feature statistics
        self.feature_counts = np.zeros(config.latent_dim)
        self.feature_means = np.zeros(config.latent_dim)
        self.feature_iteration_counts = np.zeros((config.latent_dim, 18))  # H×L iterations

        # Training step counter
        self.step_count = 0

    def compute_loss(
        self,
        board_states: mx.array,  # [B, 10]
        contexts: mx.array,      # [B, context_dim]
        guards: mx.array,        # [B, 9]
        iteration_idx: Optional[mx.array] = None,  # [B] iteration index 0-17
    ) -> Tuple[mx.array, Dict]:
        """
        Compute SAE loss.

        Args:
            board_states: Soft cell states
            contexts: Board context vectors
            guards: Guard values
            iteration_idx: Which iteration each sample came from

        Returns:
            loss: Total loss
            metrics: Dict with loss components
        """
        # Concatenate inputs
        x = mx.concatenate([board_states, contexts, guards], axis=-1)

        # Forward pass
        recon, acts, indices = self.sae(x)

        # Reconstruction loss
        recon_loss = mx.mean((x - recon) ** 2)

        # Sparsity loss (L1)
        sparsity_loss = self.config.l1_coefficient * mx.mean(mx.abs(acts))

        loss = recon_loss + sparsity_loss

        # Update feature statistics
        self._update_statistics(acts, indices, iteration_idx)

        metrics = {
            'reconstruction_loss': float(recon_loss.item()),
            'sparsity_loss': float(sparsity_loss.item()),
            'total_loss': float(loss.item()),
            'active_features': float(mx.mean(mx.sum(acts > 0, axis=-1)).item()),
        }

        return loss, metrics

    def _update_statistics(
        self,
        acts: mx.array,
        indices: mx.array,
        iteration_idx: Optional[mx.array] = None,
    ):
        """Update feature activation statistics."""
        indices_np = np.array(indices.tolist())
        acts_np = np.array(acts.tolist())

        for b in range(indices_np.shape[0]):
            for idx in indices_np[b]:
                self.feature_counts[idx] += 1
                self.feature_means[idx] = (
                    self.feature_means[idx] * 0.99 +
                    acts_np[b, idx] * 0.01
                )

                if iteration_idx is not None:
                    it_idx = int(iteration_idx[b].item())
                    self.feature_iteration_counts[idx, it_idx] += 1

    def get_dead_features(self, threshold: float = None) -> List[int]:
        """Get indices of dead (never activated) features."""
        threshold = threshold or self.config.dead_threshold
        return list(np.where(self.feature_counts < threshold)[0])

    def resample_dead_features(self, alive_features: List[int]):
        """
        Resample dead feature weights from alive features.

        Uses the approach from the original SAE paper.
        """
        dead_features = self.get_dead_features()
        if not dead_features or not alive_features:
            return 0

        # Get weights
        enc_weights = dict(self.sae.encoder.parameters())['weight']
        dec_weights = dict(self.sae.decoder.parameters())['weight']

        enc_np = np.array(enc_weights.tolist())
        dec_np = np.array(dec_weights.tolist())

        # Resample from alive features with some noise
        resampled = 0
        for dead_idx in dead_features[:100]:  # Limit resampling per call
            alive_idx = np.random.choice(alive_features)

            # Copy weights with noise
            enc_np[dead_idx] = enc_np[alive_idx] * (1 + 0.1 * np.random.randn())
            dec_np[:, dead_idx] = dec_np[:, alive_idx] * (1 + 0.1 * np.random.randn())

            resampled += 1

        # Update weights
        self.sae.encoder.weight = mx.array(enc_np)
        self.sae.decoder.weight = mx.array(dec_np)

        return resampled

    def encode(
        self,
        board_states: mx.array,
        contexts: mx.array,
        guards: mx.array,
    ) -> Tuple[mx.array, mx.array]:
        """
        Encode inputs to sparse features.

        Returns:
            acts: Sparse activations
            indices: Active feature indices
        """
        x = mx.concatenate([board_states, contexts, guards], axis=-1)
        return self.sae.encode(x)


def train_reasoning_sae(
    activation_dataset: ActivationDataset,
    config: ReasoningSAEConfig,
    save_dir: str = "sae_checkpoints",
) -> ReasoningSAE:
    """
    Train SAE on extracted activations.

    Args:
        activation_dataset: Dataset of extracted activations
        config: SAE configuration
        save_dir: Directory to save checkpoints

    Returns:
        Trained ReasoningSAE
    """
    print("=" * 60)
    print("Training Reasoning SAE")
    print("=" * 60)
    print(f"\nConfig: {json.dumps(asdict(config), indent=2)}")

    os.makedirs(save_dir, exist_ok=True)

    # Get all activations
    print("\nLoading activations...")
    board_states, contexts, guards = activation_dataset.get_all_activations(
        flatten_to_cells=True
    )
    print(f"  Board states: {board_states.shape}")
    print(f"  Contexts: {contexts.shape}")
    print(f"  Guards: {guards.shape}")

    # Update config dimensions
    config.board_dim = board_states.shape[-1]
    config.context_dim = contexts.shape[-1]
    config.guard_dim = guards.shape[-1]

    # Create SAE
    print(f"\nCreating SAE (input_dim={config.input_dim}, latent_dim={config.latent_dim})...")
    sae = ReasoningSAE(config)

    # Create optimizer
    optimizer = optim.Adam(learning_rate=config.learning_rate)

    # Training loop
    n_samples = board_states.shape[0]
    n_batches = n_samples // config.batch_size
    print(f"\nTraining on {n_samples} samples ({n_batches} batches/epoch)...")

    history = []
    for epoch in range(config.epochs):
        # Shuffle
        perm = np.random.permutation(n_samples)
        epoch_loss = 0.0
        epoch_recon = 0.0

        for batch_idx in range(n_batches):
            start = batch_idx * config.batch_size
            end = start + config.batch_size
            batch_indices = perm[start:end]

            # Get batch
            batch_board = mx.array(board_states[batch_indices])
            batch_context = mx.array(contexts[batch_indices])
            batch_guards = mx.array(guards[batch_indices])

            # Compute loss and gradients
            def loss_fn(sae_module):
                x = mx.concatenate([batch_board, batch_context, batch_guards], axis=-1)
                recon, acts, _ = sae_module.sae(x)
                recon_loss = mx.mean((x - recon) ** 2)
                sparsity_loss = config.l1_coefficient * mx.mean(mx.abs(acts))
                return recon_loss + sparsity_loss, recon_loss

            (loss, recon_loss), grads = nn.value_and_grad(
                sae, loss_fn, has_aux=True
            )(sae)

            # Update
            optimizer.update(sae, grads)

            epoch_loss += float(loss.item())
            epoch_recon += float(recon_loss.item())
            sae.step_count += 1

            # Dead feature resampling
            if config.resample_dead and sae.step_count % config.resample_interval == 0:
                alive = list(np.where(sae.feature_counts > config.dead_threshold)[0])
                resampled = sae.resample_dead_features(alive)
                if resampled > 0:
                    print(f"    Resampled {resampled} dead features")

        # Epoch stats
        avg_loss = epoch_loss / n_batches
        avg_recon = epoch_recon / n_batches
        dead_count = len(sae.get_dead_features())
        alive_count = config.latent_dim - dead_count

        history.append({
            'epoch': epoch + 1,
            'loss': avg_loss,
            'recon_loss': avg_recon,
            'alive_features': alive_count,
            'dead_features': dead_count,
        })

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch + 1}/{config.epochs}: "
                  f"loss={avg_loss:.4f}, recon={avg_recon:.4f}, "
                  f"alive={alive_count}, dead={dead_count}")

    # Save model and history
    weights_path = os.path.join(save_dir, "sae_weights.npz")
    np.savez(
        weights_path,
        encoder_weight=np.array(dict(sae.sae.encoder.parameters())['weight'].tolist()),
        encoder_bias=np.array(dict(sae.sae.encoder.parameters())['bias'].tolist()),
        decoder_weight=np.array(dict(sae.sae.decoder.parameters())['weight'].tolist()),
        feature_counts=sae.feature_counts,
        feature_means=sae.feature_means,
        feature_iteration_counts=sae.feature_iteration_counts,
    )
    print(f"\nSaved weights to {weights_path}")

    history_path = os.path.join(save_dir, "training_history.json")
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"Saved history to {history_path}")

    config_path = os.path.join(save_dir, "config.json")
    with open(config_path, 'w') as f:
        json.dump(asdict(config), f, indent=2)
    print(f"Saved config to {config_path}")

    print("\n" + "=" * 60)
    print("SAE training complete")
    print("=" * 60)

    return sae


def load_reasoning_sae(save_dir: str) -> Tuple[ReasoningSAE, ReasoningSAEConfig]:
    """Load trained SAE from disk."""
    # Load config
    config_path = os.path.join(save_dir, "config.json")
    with open(config_path) as f:
        config_dict = json.load(f)
    config = ReasoningSAEConfig(**config_dict)

    # Create SAE
    sae = ReasoningSAE(config)

    # Load weights
    weights_path = os.path.join(save_dir, "sae_weights.npz")
    data = np.load(weights_path)

    sae.sae.encoder.weight = mx.array(data['encoder_weight'])
    sae.sae.encoder.bias = mx.array(data['encoder_bias'])
    sae.sae.decoder.weight = mx.array(data['decoder_weight'])
    sae.feature_counts = data['feature_counts']
    sae.feature_means = data['feature_means']
    sae.feature_iteration_counts = data['feature_iteration_counts']

    return sae, config


def test_sae():
    """Test SAE training."""
    print("=" * 60)
    print("Testing Reasoning SAE")
    print("=" * 60)

    # Create synthetic activations
    print("\n1. Creating synthetic activations...")
    n_samples = 1000
    board_states = np.random.rand(n_samples, 10).astype(np.float32)
    contexts = np.random.rand(n_samples, 64).astype(np.float32)
    guards = np.random.rand(n_samples, 9).astype(np.float32)

    # Create config
    config = ReasoningSAEConfig(
        board_dim=10,
        context_dim=64,
        guard_dim=9,
        expansion_factor=8,
        k_active=8,
        epochs=10,
        batch_size=64,
    )

    print(f"\n2. Config: input_dim={config.input_dim}, latent_dim={config.latent_dim}")

    # Create and train SAE
    sae = ReasoningSAE(config)

    print("\n3. Testing forward pass...")
    x = mx.array(np.concatenate([board_states[:32], contexts[:32], guards[:32]], axis=-1))
    recon, acts, indices = sae.sae(x)
    print(f"   Input shape: {x.shape}")
    print(f"   Reconstruction shape: {recon.shape}")
    print(f"   Activations shape: {acts.shape}")
    print(f"   Indices shape: {indices.shape}")

    print("\n4. Testing loss computation...")
    loss, metrics = sae.compute_loss(
        mx.array(board_states[:32]),
        mx.array(contexts[:32]),
        mx.array(guards[:32]),
    )
    print(f"   Loss: {loss.item():.4f}")
    print(f"   Metrics: {metrics}")

    print("\n5. Testing gradient flow...")

    def loss_fn(sae_module):
        x = mx.array(np.concatenate([board_states[:32], contexts[:32], guards[:32]], axis=-1))
        recon, acts, _ = sae_module.sae(x)
        return mx.mean((x - recon) ** 2)

    loss, grads = nn.value_and_grad(sae, loss_fn)(sae)
    has_grad = any(mx.any(g != 0).item() for _, g in nn.utils.tree_flatten(grads))
    print(f"   Gradient flow: {'OK' if has_grad else 'FAILED'}")

    print("\n" + "=" * 60)
    print("SAE test complete")
    print("=" * 60)


if __name__ == "__main__":
    test_sae()
