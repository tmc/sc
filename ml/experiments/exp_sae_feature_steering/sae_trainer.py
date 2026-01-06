"""
SAE Trainer for SC Feature Discovery.

Trains TopK Sparse Autoencoder on SC generation activations.
Builds on exp_grammar_induction SAE patterns with SC-specific enhancements:
- Context-aware training (weight by semantic importance)
- Feature-context association tracking
- Dead feature resampling

Target: Extract ~32-64 interpretable features per SC concept.
"""

import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict

try:
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    HAS_MLX = True
except ImportError:
    HAS_MLX = False
    mx = None
    nn = None
    optim = None

from .activation_collector import ActivationCache, SemanticContext


@dataclass
class TrainerConfig:
    """Configuration for SAE training."""
    input_dim: int = 256
    expansion_factor: int = 16
    k_active: int = 32
    l1_coefficient: float = 1e-3
    learning_rate: float = 1e-3
    batch_size: int = 64
    num_epochs: int = 100
    dead_threshold: float = 1e-6
    resample_dead: bool = True
    resample_interval: int = 25


@dataclass
class TrainingMetrics:
    """Metrics from SAE training."""
    epoch: int = 0
    reconstruction_loss: float = 0.0
    sparsity_loss: float = 0.0
    total_loss: float = 0.0
    active_features: float = 0.0
    dead_features: int = 0
    training_time: float = 0.0

    def to_dict(self) -> Dict:
        return {
            'epoch': self.epoch,
            'reconstruction_loss': self.reconstruction_loss,
            'sparsity_loss': self.sparsity_loss,
            'total_loss': self.total_loss,
            'active_features': self.active_features,
            'dead_features': self.dead_features,
            'training_time': self.training_time,
        }


class TopKSAE:
    """
    TopK Sparse Autoencoder for SC feature discovery.

    Uses TopK activation instead of ReLU for guaranteed sparsity.
    """

    def __init__(self, config: TrainerConfig):
        self.config = config
        self.latent_dim = config.input_dim * config.expansion_factor

        # Initialize weights
        if HAS_MLX:
            # Xavier initialization
            scale = math.sqrt(2.0 / (config.input_dim + self.latent_dim))
            self.encoder_weight = mx.random.normal((config.input_dim, self.latent_dim)) * scale
            self.encoder_bias = mx.zeros((self.latent_dim,))
            self.decoder_weight = mx.random.normal((self.latent_dim, config.input_dim)) * scale

            # Track feature usage
            self.feature_counts = mx.zeros((self.latent_dim,))
        else:
            import numpy as np
            scale = math.sqrt(2.0 / (config.input_dim + self.latent_dim))
            self.encoder_weight = np.random.randn(config.input_dim, self.latent_dim).astype(np.float32) * scale
            self.encoder_bias = np.zeros(self.latent_dim, dtype=np.float32)
            self.decoder_weight = np.random.randn(self.latent_dim, config.input_dim).astype(np.float32) * scale
            self.feature_counts = np.zeros(self.latent_dim, dtype=np.float32)

        # Context-feature associations
        self.context_feature_counts: Dict[SemanticContext, Dict[int, int]] = defaultdict(lambda: defaultdict(int))

    def encode(self, x) -> Tuple[Any, Any]:
        """
        Encode input to sparse TopK representation.

        Returns:
            acts: Sparse activations
            indices: Active feature indices
        """
        if HAS_MLX:
            # Project to latent space
            pre_acts = x @ self.encoder_weight + self.encoder_bias

            # TopK selection
            k = self.config.k_active
            batch_size = x.shape[0]

            # Get top-k indices
            sorted_indices = mx.argsort(-pre_acts, axis=-1)
            top_indices = sorted_indices[:, :k]

            # Gather top-k values
            top_values = mx.take_along_axis(pre_acts, top_indices, axis=-1)
            top_values = mx.maximum(top_values, 0.0)

            # Create sparse output
            import numpy as np
            acts_np = np.zeros((batch_size, self.latent_dim), dtype=np.float32)
            top_indices_np = np.array(top_indices.tolist())
            top_values_np = np.array(top_values.tolist())

            for b in range(batch_size):
                acts_np[b, top_indices_np[b]] = top_values_np[b]

            acts = mx.array(acts_np)
            return acts, top_indices
        else:
            import numpy as np
            pre_acts = x @ self.encoder_weight + self.encoder_bias
            k = self.config.k_active
            batch_size = x.shape[0]

            sorted_indices = np.argsort(-pre_acts, axis=-1)
            top_indices = sorted_indices[:, :k]

            acts = np.zeros((batch_size, self.latent_dim), dtype=np.float32)
            for b in range(batch_size):
                vals = pre_acts[b, top_indices[b]]
                acts[b, top_indices[b]] = np.maximum(vals, 0.0)

            return acts, top_indices

    def decode(self, acts) -> Any:
        """Decode sparse representation."""
        if HAS_MLX:
            return acts @ self.decoder_weight
        else:
            return acts @ self.decoder_weight

    def forward(self, x) -> Tuple[Any, Any, Any]:
        """Full forward pass."""
        acts, indices = self.encode(x)
        recon = self.decode(acts)
        return recon, acts, indices

    def compute_loss(self, x) -> Tuple[Any, Dict]:
        """Compute training loss."""
        recon, acts, _ = self.forward(x)

        if HAS_MLX:
            recon_loss = mx.mean((x - recon) ** 2)
            sparsity_loss = self.config.l1_coefficient * mx.mean(mx.abs(acts))
            total_loss = recon_loss + sparsity_loss

            metrics = {
                'reconstruction_loss': float(recon_loss),
                'sparsity_loss': float(sparsity_loss),
                'total_loss': float(total_loss),
                'active_features': float(mx.sum(acts > 0) / acts.shape[0]),
            }
        else:
            import numpy as np
            recon_loss = np.mean((x - recon) ** 2)
            sparsity_loss = self.config.l1_coefficient * np.mean(np.abs(acts))
            total_loss = recon_loss + sparsity_loss

            metrics = {
                'reconstruction_loss': float(recon_loss),
                'sparsity_loss': float(sparsity_loss),
                'total_loss': float(total_loss),
                'active_features': float(np.sum(acts > 0) / acts.shape[0]),
            }

        return total_loss, metrics

    def update_feature_counts(self, indices, contexts: Optional[List[SemanticContext]] = None):
        """Update feature usage statistics."""
        import numpy as np
        if HAS_MLX:
            indices_np = np.array(indices.tolist())
            fc_np = np.array(self.feature_counts.tolist())
        else:
            indices_np = indices
            fc_np = self.feature_counts

        for b in range(len(indices_np)):
            for idx in indices_np[b]:
                idx = int(idx)
                fc_np[idx] += 1

                if contexts and b < len(contexts):
                    self.context_feature_counts[contexts[b]][idx] += 1

        if HAS_MLX:
            self.feature_counts = mx.array(fc_np)

    def get_dead_features(self) -> List[int]:
        """Get indices of dead features."""
        if HAS_MLX:
            counts = self.feature_counts.tolist()
        else:
            counts = self.feature_counts.tolist()
        return [i for i, c in enumerate(counts) if c < self.config.dead_threshold]

    def resample_dead_features(self, x):
        """Resample dead features from high-loss examples."""
        dead = self.get_dead_features()
        if not dead or not HAS_MLX:
            return len(dead)

        # Get reconstruction errors
        recon, _, _ = self.forward(x)
        errors = mx.sum((x - recon) ** 2, axis=-1)

        # Sample from high-error examples
        high_error_indices = mx.argsort(-errors)[:min(len(dead), x.shape[0])]

        # Convert to numpy for updates
        import numpy as np
        enc_np = np.array(self.encoder_weight.tolist())
        fc_np = np.array(self.feature_counts.tolist())

        for i, dead_idx in enumerate(dead[:len(high_error_indices)]):
            example_idx = int(high_error_indices[i])
            enc_np[:, dead_idx] = np.array(x[example_idx].tolist())
            fc_np[dead_idx] = 0

        self.encoder_weight = mx.array(enc_np)
        self.feature_counts = mx.array(fc_np)

        return len(dead)


class SAETrainer:
    """
    Trainer for TopK SAE on SC activations.
    """

    def __init__(self, config: TrainerConfig = None):
        self.config = config or TrainerConfig()
        self.sae: Optional[TopKSAE] = None
        self.history: List[TrainingMetrics] = []

    def train(
        self,
        cache: ActivationCache,
        verbose: bool = True,
    ) -> TopKSAE:
        """
        Train SAE on activation cache.

        Args:
            cache: ActivationCache with hidden states
            verbose: Print progress

        Returns:
            Trained TopKSAE
        """
        start_time = time.time()

        # Build tensor if needed
        if cache.hidden_states is None:
            cache.build_tensor(self.config.input_dim)

        # Update config based on data
        if HAS_MLX:
            self.config.input_dim = cache.hidden_states.shape[-1]
        else:
            self.config.input_dim = cache.hidden_states.shape[-1]

        # Initialize SAE
        self.sae = TopKSAE(self.config)

        if verbose:
            print("=" * 60)
            print("SAE TRAINING")
            print("=" * 60)
            print(f"Input dim: {self.config.input_dim}")
            print(f"Latent dim: {self.sae.latent_dim}")
            print(f"TopK: {self.config.k_active}")
            print(f"Samples: {cache.total}")
            print("-" * 60)

        # Training loop
        n_samples = cache.total
        batch_size = self.config.batch_size

        for epoch in range(self.config.num_epochs):
            epoch_loss = 0.0
            epoch_metrics = defaultdict(float)
            n_batches = 0

            # Shuffle data
            if HAS_MLX:
                indices = mx.array(list(range(n_samples)))
                indices = mx.random.permutation(indices)
                data = cache.hidden_states[indices]
                labels = [cache.labels[int(i)] for i in indices.tolist()]
            else:
                import numpy as np
                indices = np.random.permutation(n_samples)
                data = cache.hidden_states[indices]
                labels = [cache.labels[i] for i in indices]

            # Mini-batches
            for i in range(0, n_samples, batch_size):
                batch = data[i:i+batch_size]
                batch_labels = labels[i:i+batch_size]

                # Forward + loss
                loss, metrics = self.sae.compute_loss(batch)

                # Backward (simplified - real training would use optimizer)
                if HAS_MLX:
                    # Update weights with gradient descent (simplified)
                    recon, acts, top_indices = self.sae.forward(batch)
                    error = batch - recon

                    # Gradient for decoder
                    grad_decoder = -2 * acts.T @ error / batch.shape[0]
                    self.sae.decoder_weight = self.sae.decoder_weight - self.config.learning_rate * grad_decoder

                    # Gradient for encoder
                    grad_encoder = -2 * batch.T @ (error @ self.sae.decoder_weight.T * (acts > 0)) / batch.shape[0]
                    self.sae.encoder_weight = self.sae.encoder_weight - self.config.learning_rate * grad_encoder

                # Update feature counts
                _, _, indices = self.sae.forward(batch)
                self.sae.update_feature_counts(indices, batch_labels)

                epoch_loss += float(loss) if HAS_MLX else loss
                for k, v in metrics.items():
                    epoch_metrics[k] += v
                n_batches += 1

            # Resample dead features
            dead_count = 0
            if self.config.resample_dead and (epoch + 1) % self.config.resample_interval == 0:
                dead_count = self.sae.resample_dead_features(data[:batch_size])

            # Record metrics
            metrics = TrainingMetrics(
                epoch=epoch,
                reconstruction_loss=epoch_metrics['reconstruction_loss'] / n_batches,
                sparsity_loss=epoch_metrics['sparsity_loss'] / n_batches,
                total_loss=epoch_loss / n_batches,
                active_features=epoch_metrics['active_features'] / n_batches,
                dead_features=dead_count,
                training_time=time.time() - start_time,
            )
            self.history.append(metrics)

            if verbose and (epoch % 10 == 0 or epoch == self.config.num_epochs - 1):
                print(f"Epoch {epoch:3d}: loss={metrics.total_loss:.4f}, "
                      f"recon={metrics.reconstruction_loss:.4f}, "
                      f"dead={dead_count}")

        if verbose:
            print("-" * 60)
            print(f"Training complete in {time.time() - start_time:.2f}s")
            print(f"Final loss: {self.history[-1].total_loss:.4f}")
            print("=" * 60)

        return self.sae

    def get_context_features(self) -> Dict[SemanticContext, List[Tuple[int, int]]]:
        """Get top features per semantic context."""
        result = {}
        for ctx, feature_counts in self.sae.context_feature_counts.items():
            sorted_features = sorted(feature_counts.items(), key=lambda x: -x[1])
            result[ctx] = sorted_features[:20]
        return result


def test_trainer():
    """Test SAE trainer."""
    print("=" * 60)
    print("Testing SAE Trainer")
    print("=" * 60)

    from .activation_collector import ActivationCollector

    # Collect activations
    collector = ActivationCollector()
    sc_json = '''{
        "root_state": {"label": "__root__", "type": 2, "children": [
            {"label": "Off", "type": 1, "is_initial": true},
            {"label": "On", "type": 1}
        ]},
        "transitions": [
            {"from": ["Off"], "to": ["On"], "event": "TOGGLE"},
            {"from": ["On"], "to": ["Off"], "event": "TOGGLE"}
        ]
    }'''

    # Collect multiple times to get more data
    for _ in range(10):
        collector.collect(sc_json)

    cache = collector.cache
    print(f"\n1. Collected {cache.total} activations")

    # Train SAE
    config = TrainerConfig(
        input_dim=64,
        expansion_factor=8,
        k_active=8,
        num_epochs=50,
        batch_size=16,
    )

    trainer = SAETrainer(config)
    sae = trainer.train(cache, verbose=True)

    # Check features
    print("\n2. Feature analysis:")
    dead = sae.get_dead_features()
    print(f"  Dead features: {len(dead)} / {sae.latent_dim}")

    # Context-feature associations
    print("\n3. Context-feature associations:")
    ctx_features = trainer.get_context_features()
    for ctx, features in list(ctx_features.items())[:5]:
        top_feats = [f"f{f[0]}:{f[1]}" for f in features[:3]]
        print(f"  {ctx.name}: {', '.join(top_feats)}")

    print("\n" + "=" * 60)
    print("SAE trainer tests complete!")
    print("=" * 60)


if __name__ == "__main__":
    test_trainer()
