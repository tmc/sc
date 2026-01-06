"""
Cross-Model Steering Vector Transfer.

Adapts steering vectors from source model (Qwen-0.5B) to target model (Qwen-1.5B).
Handles dimension mismatches through various projection methods.

Key insight: Steering vectors encode semantic directions that may be
preserved across model sizes within the same family.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum, auto


class TransferMethod(Enum):
    """Methods for adapting vectors between model sizes."""
    ZERO_PAD = auto()      # Pad smaller vectors with zeros
    TRUNCATE = auto()      # Truncate larger vectors
    LINEAR_INTERP = auto() # Linear interpolation to target dim
    PCA_PROJECT = auto()   # Project via shared PCA subspace
    LAYER_MAP = auto()     # Map layers by relative position


@dataclass
class ModelConfig:
    """Configuration for a model architecture."""
    name: str
    hidden_dim: int
    num_layers: int
    num_heads: int
    head_dim: int

    @property
    def total_head_dim(self) -> int:
        return self.num_heads * self.head_dim


# Known Qwen model configurations
QWEN_CONFIGS = {
    "0.5B": ModelConfig(
        name="Qwen2.5-0.5B",
        hidden_dim=896,
        num_layers=24,
        num_heads=14,
        head_dim=64,
    ),
    "1.5B": ModelConfig(
        name="Qwen2.5-1.5B",
        hidden_dim=1536,
        num_layers=28,
        num_heads=12,
        head_dim=128,
    ),
    "3B": ModelConfig(
        name="Qwen2.5-3B",
        hidden_dim=2048,
        num_layers=36,
        num_heads=16,
        head_dim=128,
    ),
}


@dataclass
class SteeringVector:
    """A steering vector with metadata."""
    name: str
    vector: np.ndarray
    layer: int
    head: Optional[int] = None  # None for residual stream vectors
    source_model: str = "0.5B"
    alpha: float = 1.0  # Original optimal alpha

    @property
    def dim(self) -> int:
        return self.vector.shape[0]

    def normalize(self) -> 'SteeringVector':
        """Return L2-normalized version."""
        norm = np.linalg.norm(self.vector)
        if norm > 0:
            return SteeringVector(
                name=self.name,
                vector=self.vector / norm,
                layer=self.layer,
                head=self.head,
                source_model=self.source_model,
                alpha=self.alpha,
            )
        return self


@dataclass
class TransferResult:
    """Result of transferring a vector to target model."""
    source_vector: SteeringVector
    target_vector: np.ndarray
    target_layer: int
    target_head: Optional[int]
    method: TransferMethod
    scale_factor: float = 1.0  # Recommended alpha scaling


class VectorTransfer:
    """
    Handles transfer of steering vectors between model sizes.

    Supports multiple adaptation methods for dimension mismatches.
    """

    def __init__(
        self,
        source_model: str = "0.5B",
        target_model: str = "1.5B",
        method: TransferMethod = TransferMethod.LINEAR_INTERP,
    ):
        self.source_config = QWEN_CONFIGS[source_model]
        self.target_config = QWEN_CONFIGS[target_model]
        self.method = method

        # Compute dimension ratios
        self.dim_ratio = self.target_config.hidden_dim / self.source_config.hidden_dim
        self.layer_ratio = self.target_config.num_layers / self.source_config.num_layers
        self.head_ratio = self.target_config.num_heads / self.source_config.num_heads

    def map_layer(self, source_layer: int) -> int:
        """Map source layer to equivalent target layer."""
        # Use relative position mapping
        relative_pos = source_layer / self.source_config.num_layers
        target_layer = int(relative_pos * self.target_config.num_layers)
        return min(target_layer, self.target_config.num_layers - 1)

    def map_head(self, source_head: int) -> int:
        """Map source attention head to equivalent target head."""
        # Use relative position mapping
        relative_pos = source_head / self.source_config.num_heads
        target_head = int(relative_pos * self.target_config.num_heads)
        return min(target_head, self.target_config.num_heads - 1)

    def adapt_dimension(self, vector: np.ndarray) -> np.ndarray:
        """Adapt vector dimension to target model."""
        source_dim = vector.shape[0]
        target_dim = self.target_config.hidden_dim

        if source_dim == target_dim:
            return vector.copy()

        if self.method == TransferMethod.ZERO_PAD:
            if target_dim > source_dim:
                # Pad with zeros
                padded = np.zeros(target_dim)
                padded[:source_dim] = vector
                return padded
            else:
                # Truncate
                return vector[:target_dim].copy()

        elif self.method == TransferMethod.TRUNCATE:
            if target_dim < source_dim:
                return vector[:target_dim].copy()
            else:
                padded = np.zeros(target_dim)
                padded[:source_dim] = vector
                return padded

        elif self.method == TransferMethod.LINEAR_INTERP:
            # Linear interpolation to target dimension
            # This preserves relative positions in the vector
            indices = np.linspace(0, source_dim - 1, target_dim)
            return np.interp(np.arange(target_dim),
                           np.arange(source_dim) * (target_dim / source_dim),
                           vector)

        elif self.method == TransferMethod.PCA_PROJECT:
            # Project through shared subspace
            # Simplified: use SVD to find principal components
            # then reconstruct in target dimension
            min_dim = min(source_dim, target_dim)

            # Normalize and project to shared dimension
            norm = np.linalg.norm(vector)
            if norm > 0:
                unit_vec = vector / norm
            else:
                unit_vec = vector

            # Simple approach: take first min_dim components
            if target_dim > source_dim:
                result = np.zeros(target_dim)
                result[:source_dim] = unit_vec
            else:
                result = unit_vec[:target_dim]

            return result * norm  # Restore magnitude

        else:
            # Default: linear interpolation
            return np.interp(
                np.linspace(0, 1, target_dim),
                np.linspace(0, 1, source_dim),
                vector
            )

    def compute_scale_factor(self, source_vector: SteeringVector) -> float:
        """
        Compute recommended alpha scaling for transferred vector.

        Larger models may need different activation magnitudes.
        """
        # Heuristic: scale inversely with sqrt of dimension ratio
        # to preserve relative activation magnitude
        base_scale = 1.0 / np.sqrt(self.dim_ratio)

        # Also account for layer depth difference
        layer_scale = np.sqrt(self.layer_ratio)

        return base_scale * layer_scale

    def transfer(self, source_vector: SteeringVector) -> TransferResult:
        """
        Transfer a steering vector to the target model.

        Returns adapted vector with recommended scaling.
        """
        # Map layer and head
        target_layer = self.map_layer(source_vector.layer)
        target_head = None
        if source_vector.head is not None:
            target_head = self.map_head(source_vector.head)

        # Adapt dimension
        target_vec = self.adapt_dimension(source_vector.vector)

        # Normalize to preserve direction
        source_norm = np.linalg.norm(source_vector.vector)
        target_norm = np.linalg.norm(target_vec)
        if target_norm > 0:
            target_vec = target_vec * (source_norm / target_norm)

        # Compute recommended scaling
        scale_factor = self.compute_scale_factor(source_vector)

        return TransferResult(
            source_vector=source_vector,
            target_vector=target_vec,
            target_layer=target_layer,
            target_head=target_head,
            method=self.method,
            scale_factor=scale_factor,
        )

    def transfer_batch(
        self,
        vectors: List[SteeringVector]
    ) -> List[TransferResult]:
        """Transfer multiple vectors."""
        return [self.transfer(v) for v in vectors]


def create_simulated_steering_vectors() -> List[SteeringVector]:
    """
    Create simulated steering vectors for testing.

    In production, these would be loaded from actual trained vectors.
    """
    np.random.seed(42)
    config = QWEN_CONFIGS["0.5B"]

    vectors = []

    # HIERARCHY head (L23H1) - promotes hierarchical structure
    hierarchy_vec = np.random.randn(config.hidden_dim)
    # Make it sparse-ish (hierarchy concept)
    hierarchy_vec[np.abs(hierarchy_vec) < 0.5] = 0
    vectors.append(SteeringVector(
        name="HIERARCHY",
        vector=hierarchy_vec,
        layer=23,
        head=1,
        source_model="0.5B",
        alpha=0.8,
    ))

    # STRUCTURE heads
    for layer, head, name in [(11, 13, "STRUCTURE_L11H13"),
                               (11, 7, "STRUCTURE_L11H7"),
                               (9, 7, "STRUCTURE_L9H7")]:
        vec = np.random.randn(config.hidden_dim)
        # Structure vectors tend to be denser
        vectors.append(SteeringVector(
            name=name,
            vector=vec,
            layer=layer,
            head=head,
            source_model="0.5B",
            alpha=0.6,
        ))

    # SAE feature vectors (if we had them)
    for i, feature_name in enumerate(["STATE_FEATURE", "TRANS_FEATURE", "GUARD_FEATURE"]):
        vec = np.random.randn(config.hidden_dim)
        # SAE features are typically sparser
        vec[np.abs(vec) < 0.8] = 0
        vectors.append(SteeringVector(
            name=feature_name,
            vector=vec,
            layer=16,  # Middle layer
            head=None,  # Residual stream
            source_model="0.5B",
            alpha=1.0,
        ))

    return vectors


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return np.dot(a, b) / (norm_a * norm_b)


def test_vector_transfer():
    """Test vector transfer functionality."""
    print("=" * 60)
    print("Testing Cross-Model Steering Vector Transfer")
    print("=" * 60)

    # Create source vectors
    source_vectors = create_simulated_steering_vectors()
    print(f"\nSource vectors from {QWEN_CONFIGS['0.5B'].name}:")
    for v in source_vectors:
        print(f"  {v.name}: layer={v.layer}, head={v.head}, dim={v.dim}")

    # Test different transfer methods
    print("\n" + "-" * 60)
    print("Transfer Methods Comparison")
    print("-" * 60)

    for method in [TransferMethod.LINEAR_INTERP, TransferMethod.ZERO_PAD, TransferMethod.PCA_PROJECT]:
        print(f"\n{method.name}:")
        transfer = VectorTransfer(
            source_model="0.5B",
            target_model="1.5B",
            method=method,
        )

        for sv in source_vectors[:3]:  # Just first 3 for brevity
            result = transfer.transfer(sv)
            print(f"  {sv.name}:")
            print(f"    Source: layer={sv.layer}, head={sv.head}, dim={sv.dim}")
            print(f"    Target: layer={result.target_layer}, head={result.target_head}, "
                  f"dim={result.target_vector.shape[0]}")
            print(f"    Scale factor: {result.scale_factor:.3f}")

    # Test layer/head mapping
    print("\n" + "-" * 60)
    print("Layer and Head Mapping (0.5B -> 1.5B)")
    print("-" * 60)
    transfer = VectorTransfer("0.5B", "1.5B")

    print("\nLayer mapping:")
    for src_layer in [0, 6, 12, 18, 23]:
        tgt_layer = transfer.map_layer(src_layer)
        print(f"  L{src_layer} -> L{tgt_layer}")

    print("\nHead mapping:")
    for src_head in [0, 3, 7, 10, 13]:
        tgt_head = transfer.map_head(src_head)
        print(f"  H{src_head} -> H{tgt_head}")

    print("\n" + "=" * 60)
    print("Vector transfer tests complete!")
    print("=" * 60)


if __name__ == "__main__":
    test_vector_transfer()
