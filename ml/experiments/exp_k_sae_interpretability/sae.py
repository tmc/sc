"""Sparse Auto-Encoders for Statechart Interpretability.

Discovers interpretable features in differentiable statecharts:
- State patterns (which soft configurations represent meaningful states)
- Transition patterns (which transitions form logical sequences)
- Guard semantics (what conditions do learned guards detect)
"""

import mlx.core as mx
import mlx.nn as nn
from typing import Dict, List, Tuple, Optional


class SparseAutoEncoder(nn.Module):
    """Basic Sparse Auto-Encoder with L1 sparsity penalty.

    Architecture:
        Encoder: x -> ReLU(Wx + b) -> sparse activations z
        Decoder: z -> W^T z -> reconstruction x'

    Loss: MSE(x, x') + λ * L1(z)
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        sparsity_weight: float = 0.01,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.sparsity_weight = sparsity_weight

        # Encoder
        self.encoder = nn.Linear(input_dim, latent_dim)

        # Decoder (tied weights option: use encoder.weight.T)
        self.decoder = nn.Linear(latent_dim, input_dim)

    def encode(self, x: mx.array) -> mx.array:
        """Encode to sparse latent representation."""
        return mx.maximum(self.encoder(x), 0)  # ReLU activation

    def decode(self, z: mx.array) -> mx.array:
        """Decode from latent representation."""
        return self.decoder(z)

    def __call__(self, x: mx.array) -> Tuple[mx.array, mx.array, mx.array]:
        """Forward pass returning reconstruction, latents, and loss.

        Args:
            x: Input tensor [B, input_dim]

        Returns:
            Tuple of (reconstruction, latents, loss)
        """
        z = self.encode(x)
        x_hat = self.decode(z)

        # Reconstruction loss
        recon_loss = mx.mean((x - x_hat) ** 2)

        # Sparsity loss (L1 on activations)
        sparsity_loss = mx.mean(mx.abs(z))

        total_loss = recon_loss + self.sparsity_weight * sparsity_loss

        return x_hat, z, total_loss


class TopKSparseAutoEncoder(nn.Module):
    """Sparse Auto-Encoder with top-k activation instead of L1.

    Only keeps the k largest activations, forcing exact sparsity.
    Better for interpretability as sparsity level is guaranteed.
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        k: int = 10,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.k = k

        self.encoder = nn.Linear(input_dim, latent_dim)
        self.decoder = nn.Linear(latent_dim, input_dim)

    def encode(self, x: mx.array) -> mx.array:
        """Encode with top-k sparsity."""
        z = mx.maximum(self.encoder(x), 0)  # ReLU

        # Top-k selection (per sample)
        if z.ndim == 1:
            z = z[None, :]  # Add batch dim

        # Get top-k values and create mask
        # Sort and get threshold
        sorted_z = mx.sort(z, axis=-1)
        threshold = sorted_z[:, -self.k : -self.k + 1]  # k-th largest value

        # Zero out below threshold
        mask = z >= threshold
        z_sparse = z * mask

        return z_sparse

    def decode(self, z: mx.array) -> mx.array:
        """Decode from sparse representation."""
        return self.decoder(z)

    def __call__(self, x: mx.array) -> Tuple[mx.array, mx.array, mx.array]:
        """Forward pass."""
        z = self.encode(x)
        x_hat = self.decode(z)

        # Only reconstruction loss (sparsity is guaranteed by top-k)
        loss = mx.mean((x - x_hat) ** 2)

        return x_hat, z, loss


class StatechartSAE(nn.Module):
    """SAE specialized for statechart interpretability.

    Three separate SAEs for:
    1. State configurations (soft config vectors)
    2. Transition selections (attention weights)
    3. Guard activations (sigmoid outputs)
    """

    def __init__(
        self,
        num_states: int,
        num_transitions: int,
        num_guards: int,
        latent_multiplier: float = 2.0,
        sparsity_k: int = 5,
    ):
        super().__init__()

        state_latent = int(num_states * latent_multiplier)
        trans_latent = int(num_transitions * latent_multiplier)
        guard_latent = int(num_guards * latent_multiplier)

        self.state_sae = TopKSparseAutoEncoder(num_states, state_latent, sparsity_k)
        self.transition_sae = TopKSparseAutoEncoder(
            num_transitions, trans_latent, sparsity_k
        )
        self.guard_sae = TopKSparseAutoEncoder(num_guards, guard_latent, sparsity_k)

    def analyze_state(
        self, config: mx.array
    ) -> Tuple[mx.array, mx.array, Dict[str, float]]:
        """Analyze a soft state configuration.

        Args:
            config: Soft configuration [B, num_states]

        Returns:
            Tuple of (reconstruction, features, metrics)
        """
        recon, features, loss = self.state_sae(config)

        # Compute interpretability metrics
        active_features = mx.sum(features > 0.1, axis=-1)
        feature_entropy = -mx.sum(
            mx.where(
                features > 1e-6,
                features * mx.log(features + 1e-6),
                mx.zeros_like(features),
            ),
            axis=-1,
        )

        metrics = {
            "reconstruction_loss": float(loss),
            "active_features": float(mx.mean(active_features)),
            "feature_entropy": float(mx.mean(feature_entropy)),
        }

        return recon, features, metrics

    def analyze_transitions(
        self, trans_weights: mx.array
    ) -> Tuple[mx.array, mx.array, Dict[str, float]]:
        """Analyze transition selection patterns."""
        recon, features, loss = self.transition_sae(trans_weights)

        active_features = mx.sum(features > 0.1, axis=-1)

        metrics = {
            "reconstruction_loss": float(loss),
            "active_features": float(mx.mean(active_features)),
        }

        return recon, features, metrics

    def analyze_guards(
        self, guard_activations: mx.array
    ) -> Tuple[mx.array, mx.array, Dict[str, float]]:
        """Analyze guard activation patterns."""
        recon, features, loss = self.guard_sae(guard_activations)

        active_features = mx.sum(features > 0.1, axis=-1)

        metrics = {
            "reconstruction_loss": float(loss),
            "active_features": float(mx.mean(active_features)),
        }

        return recon, features, metrics


class FeatureInterpreter:
    """Interpret SAE features by finding representative examples."""

    def __init__(self, sae: StatechartSAE, state_labels: List[str]):
        self.sae = sae
        self.state_labels = state_labels
        self.feature_examples: Dict[int, List[mx.array]] = {}

    def collect_examples(
        self, configs: mx.array, max_per_feature: int = 10
    ) -> Dict[int, List[int]]:
        """Collect examples that strongly activate each feature.

        Args:
            configs: Batch of configurations [N, num_states]
            max_per_feature: Max examples to store per feature

        Returns:
            Dict mapping feature_idx -> list of sample indices
        """
        _, features, _ = self.sae.state_sae(configs)

        feature_examples: Dict[int, List[int]] = {}
        num_features = features.shape[-1]

        for feat_idx in range(num_features):
            # Find samples with high activation for this feature
            activations = features[:, feat_idx]
            top_indices = mx.argsort(-activations)[:max_per_feature]
            feature_examples[feat_idx] = [int(i) for i in top_indices]

        return feature_examples

    def describe_feature(
        self, feat_idx: int, configs: mx.array, threshold: float = 0.5
    ) -> str:
        """Generate human-readable description of a feature.

        Analyzes which states are commonly active when feature fires.
        """
        _, features, _ = self.sae.state_sae(configs)

        # Find samples where this feature is active
        active_mask = features[:, feat_idx] > threshold
        if mx.sum(active_mask) == 0:
            return f"Feature {feat_idx}: No activations above threshold"

        # Average configuration when feature is active
        active_configs = configs[active_mask]
        avg_config = mx.mean(active_configs, axis=0)

        # Find top states
        top_state_indices = mx.argsort(-avg_config)[:3]
        top_states = [self.state_labels[int(i)] for i in top_state_indices]
        top_probs = [float(avg_config[int(i)]) for i in top_state_indices]

        desc = f"Feature {feat_idx}: Active when "
        desc += ", ".join(f"{s} ({p:.2f})" for s, p in zip(top_states, top_probs))
        return desc


def test_statechart_sae():
    """Test the SAE implementation."""
    print("=" * 60)
    print("Testing Statechart SAE for Interpretability")
    print("=" * 60)

    mx.random.seed(42)

    # Create SAE for a small statechart
    num_states = 8
    num_transitions = 12
    num_guards = 6

    sae = StatechartSAE(
        num_states=num_states,
        num_transitions=num_transitions,
        num_guards=num_guards,
        latent_multiplier=2.0,
        sparsity_k=3,
    )

    # Generate synthetic soft configurations
    batch_size = 32
    configs = mx.softmax(mx.random.normal((batch_size, num_states)), axis=-1)
    trans_weights = mx.softmax(mx.random.normal((batch_size, num_transitions)), axis=-1)
    guard_acts = mx.sigmoid(mx.random.normal((batch_size, num_guards)))

    print(f"\nInput shapes:")
    print(f"  Configurations: {configs.shape}")
    print(f"  Transition weights: {trans_weights.shape}")
    print(f"  Guard activations: {guard_acts.shape}")

    # Analyze
    _, state_features, state_metrics = sae.analyze_state(configs)
    _, trans_features, trans_metrics = sae.analyze_transitions(trans_weights)
    _, guard_features, guard_metrics = sae.analyze_guards(guard_acts)

    print(f"\nState analysis:")
    print(f"  Features shape: {state_features.shape}")
    print(f"  Metrics: {state_metrics}")

    print(f"\nTransition analysis:")
    print(f"  Features shape: {trans_features.shape}")
    print(f"  Metrics: {trans_metrics}")

    print(f"\nGuard analysis:")
    print(f"  Features shape: {guard_features.shape}")
    print(f"  Metrics: {guard_metrics}")

    # Test gradient flow
    print("\nTesting gradient flow...")

    def loss_fn(sae, configs, trans_weights, guard_acts):
        _, _, loss1 = sae.state_sae(configs)
        _, _, loss2 = sae.transition_sae(trans_weights)
        _, _, loss3 = sae.guard_sae(guard_acts)
        return loss1 + loss2 + loss3

    loss, grads = nn.value_and_grad(sae, loss_fn)(
        sae, configs, trans_weights, guard_acts
    )
    print(f"Total loss: {float(loss):.6f}")

    # Count gradient tensors
    grad_count = 0

    def count_grads(g):
        nonlocal grad_count
        if isinstance(g, dict):
            for v in g.values():
                count_grads(v)
        elif isinstance(g, mx.array):
            if mx.sum(mx.abs(g)) > 1e-10:
                grad_count += 1

    count_grads(grads)
    print(f"Gradient tensors with values: {grad_count}")

    # Test feature interpretation
    print("\nTesting feature interpretation...")
    state_labels = [f"State_{i}" for i in range(num_states)]
    interpreter = FeatureInterpreter(sae, state_labels)

    examples = interpreter.collect_examples(configs, max_per_feature=5)
    print(f"Collected examples for {len(examples)} features")

    # Describe first few features
    for feat_idx in range(min(3, len(examples))):
        desc = interpreter.describe_feature(feat_idx, configs, threshold=0.3)
        print(f"  {desc}")

    print("\n" + "=" * 60)
    print("Statechart SAE test completed!")
    print("=" * 60)


if __name__ == "__main__":
    test_statechart_sae()
