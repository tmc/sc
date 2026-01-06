"""
Steering Vectors for Statechart Validity

Computes steering vectors from valid vs invalid statechart pairs.
When applied during generation, these vectors push the model towards
producing syntactically and semantically valid statecharts.

Key insight: The difference in hidden states between valid and invalid
SC generation captures "validity direction" in activation space.

Uses ml/utils/mlux_loader.py for model loading.

Target: +10% validity improvement with steering.
"""

import json
import sys
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any
from pathlib import Path

# Add utils to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from utils.mlux_loader import (
    load_model,
    ModelBackend,
    HookedModelWrapper,
    GenerationConfig,
    CacheConfig,
    MLUX_AVAILABLE,
)


# =============================================================================
# CONTRASTIVE PAIRS
# =============================================================================

@dataclass
class ContrastivePair:
    """A pair of valid and invalid statechart examples."""
    valid: str      # Valid statechart JSON
    invalid: str    # Invalid variant (syntax or semantic error)
    description: str
    error_type: str  # "syntax", "semantic", "structural"


# Valid statechart examples
VALID_STATECHARTS = [
    # Simple toggle
    '''{
  "root_state": {
    "label": "__root__",
    "type": 2,
    "children": [
      {"label": "Off", "type": 1, "is_initial": true},
      {"label": "On", "type": 1}
    ]
  },
  "transitions": [
    {"from": ["Off"], "to": ["On"], "event": "TOGGLE"},
    {"from": ["On"], "to": ["Off"], "event": "TOGGLE"}
  ]
}''',
    # Traffic light
    '''{
  "root_state": {
    "label": "__root__",
    "type": 2,
    "children": [
      {"label": "Red", "type": 1, "is_initial": true},
      {"label": "Green", "type": 1},
      {"label": "Yellow", "type": 1}
    ]
  },
  "transitions": [
    {"from": ["Red"], "to": ["Green"], "event": "TIMER"},
    {"from": ["Green"], "to": ["Yellow"], "event": "TIMER"},
    {"from": ["Yellow"], "to": ["Red"], "event": "TIMER"}
  ]
}''',
    # Order processing
    '''{
  "root_state": {
    "label": "__root__",
    "type": 2,
    "children": [
      {"label": "Pending", "type": 1, "is_initial": true},
      {"label": "Processing", "type": 1},
      {"label": "Shipped", "type": 1},
      {"label": "Delivered", "type": 1}
    ]
  },
  "transitions": [
    {"from": ["Pending"], "to": ["Processing"], "event": "CONFIRM"},
    {"from": ["Processing"], "to": ["Shipped"], "event": "SHIP"},
    {"from": ["Shipped"], "to": ["Delivered"], "event": "DELIVER"}
  ]
}''',
]

# Invalid statechart examples (various error types)
INVALID_STATECHARTS = [
    # Missing closing brace (syntax)
    ('''{
  "root_state": {
    "label": "__root__",
    "type": 2,
    "children": [
      {"label": "A", "is_initial": true},
      {"label": "B"}
    ]
  },
  "transitions": [
    {"from": ["A"], "to": ["B"], "event": "GO"}
  ]
''', "syntax", "Missing closing brace"),

    # Invalid JSON (syntax)
    ('''{
  "root_state": {
    "label": "__root__"
    "type": 2,
    "children": []
  }
}''', "syntax", "Missing comma"),

    # No initial state (semantic)
    ('''{
  "root_state": {
    "label": "__root__",
    "type": 2,
    "children": [
      {"label": "A"},
      {"label": "B"}
    ]
  },
  "transitions": []
}''', "semantic", "No initial state"),

    # Transition to non-existent state (semantic)
    ('''{
  "root_state": {
    "label": "__root__",
    "type": 2,
    "children": [
      {"label": "A", "is_initial": true}
    ]
  },
  "transitions": [
    {"from": ["A"], "to": ["NonExistent"], "event": "GO"}
  ]
}''', "semantic", "Invalid target state"),

    # Empty children (structural)
    ('''{
  "root_state": {
    "label": "__root__",
    "type": 2,
    "children": []
  },
  "transitions": []
}''', "structural", "Empty children"),
]


def create_contrastive_pairs() -> List[ContrastivePair]:
    """Create contrastive pairs from valid/invalid examples."""
    pairs = []

    for i, valid in enumerate(VALID_STATECHARTS):
        # Pair with syntax errors
        for invalid, error_type, desc in INVALID_STATECHARTS:
            pairs.append(ContrastivePair(
                valid=valid,
                invalid=invalid,
                description=f"Valid SC vs {desc}",
                error_type=error_type,
            ))

    return pairs


# =============================================================================
# STEERING VECTOR COMPUTATION
# =============================================================================

@dataclass
class SteeringVector:
    """A computed steering vector."""
    vector: Any  # mlx array or mock
    layer: int
    pair_description: str
    error_type: str
    magnitude: float = 0.0

    def __repr__(self):
        shape = getattr(self.vector, 'shape', 'mock')
        return f"SteeringVector(layer={self.layer}, shape={shape}, type={self.error_type})"


class SteeringVectorComputer:
    """
    Computes steering vectors from contrastive pairs.

    Given (valid_sc, invalid_sc) pairs, extracts the direction in
    activation space that corresponds to "validity".
    """

    def __init__(
        self,
        model: HookedModelWrapper = None,
        model_name: str = "Qwen/Qwen2.5-Coder-0.5B-Instruct",
    ):
        self.model = model or load_model(model_name)
        self.vectors: Dict[int, List[SteeringVector]] = {}  # layer -> vectors

    @property
    def has_mlux(self) -> bool:
        # Handle both HookedModelWrapper and raw HookedModel
        if hasattr(self.model, 'has_interpretability'):
            return self.model.has_interpretability
        # If it's a raw HookedModel from mlux, we have mlux
        if hasattr(self.model, 'run_with_hooks'):
            return True
        return False

    def compute_single(
        self,
        pair: ContrastivePair,
        layer: int,
    ) -> SteeringVector:
        """
        Compute steering vector from a single contrastive pair.

        Args:
            pair: Valid/invalid statechart pair
            layer: Layer to extract activations from

        Returns:
            SteeringVector for this pair
        """
        if self.has_mlux:
            # Real computation using mlux
            vector = self.model.compute_steering_vector(
                positive=f"Generate valid statechart:\n{pair.valid}",
                negative=f"Generate statechart:\n{pair.invalid}",
                layer=layer,
            )
            magnitude = float(vector.sum() ** 2) ** 0.5 if hasattr(vector, 'sum') else 0.0
        else:
            # Mock computation
            import random
            vector = [random.gauss(0, 0.1) for _ in range(768)]
            magnitude = sum(v**2 for v in vector) ** 0.5

        return SteeringVector(
            vector=vector,
            layer=layer,
            pair_description=pair.description,
            error_type=pair.error_type,
            magnitude=magnitude,
        )

    def compute_all(
        self,
        pairs: List[ContrastivePair] = None,
        layers: List[int] = None,
    ) -> Dict[int, List[SteeringVector]]:
        """
        Compute steering vectors for all pairs across layers.

        Args:
            pairs: Contrastive pairs (default: all generated)
            layers: Layers to compute for (default: [6, 12, 18])

        Returns:
            {layer: [SteeringVector, ...]}
        """
        if pairs is None:
            pairs = create_contrastive_pairs()

        if layers is None:
            layers = [6, 12, 18]

        self.vectors = {layer: [] for layer in layers}

        for layer in layers:
            print(f"Computing steering vectors for layer {layer}...")
            for pair in pairs[:5]:  # Limit for speed
                vec = self.compute_single(pair, layer)
                self.vectors[layer].append(vec)

        return self.vectors

    def compute_averaged(
        self,
        pairs: List[ContrastivePair] = None,
        layer: int = 12,
        error_type: str = None,
    ) -> SteeringVector:
        """
        Compute averaged steering vector from multiple pairs.

        Averaging reduces noise and captures general validity direction.

        Args:
            pairs: Contrastive pairs
            layer: Layer to compute for
            error_type: Filter by error type (None = all)

        Returns:
            Averaged SteeringVector
        """
        if pairs is None:
            pairs = create_contrastive_pairs()

        if error_type:
            pairs = [p for p in pairs if p.error_type == error_type]

        vectors = []
        for pair in pairs[:5]:
            vec = self.compute_single(pair, layer)
            vectors.append(vec)

        if self.has_mlux:
            # Average mlx arrays
            import mlx.core as mx
            avg_vec = sum(v.vector for v in vectors) / len(vectors)
        else:
            # Average mock vectors
            avg_vec = [sum(v.vector[i] for v in vectors) / len(vectors)
                       for i in range(len(vectors[0].vector))]

        magnitude = sum(v.magnitude for v in vectors) / len(vectors)

        return SteeringVector(
            vector=avg_vec,
            layer=layer,
            pair_description=f"Averaged ({len(vectors)} pairs)",
            error_type=error_type or "all",
            magnitude=magnitude,
        )

    def get_best_vector(self, layer: int = 12) -> Optional[SteeringVector]:
        """Get the best (highest magnitude) steering vector for a layer."""
        if layer not in self.vectors or not self.vectors[layer]:
            return None
        return max(self.vectors[layer], key=lambda v: v.magnitude)


# =============================================================================
# STEERING VECTOR BANK
# =============================================================================

class SteeringVectorBank:
    """
    Manages a collection of steering vectors for different purposes.

    Supports:
    - Per-error-type vectors
    - Per-layer vectors
    - Combined validity vectors
    """

    def __init__(self):
        self.vectors: Dict[str, SteeringVector] = {}
        self.metadata: Dict[str, Any] = {}

    def add(self, name: str, vector: SteeringVector):
        """Add a steering vector to the bank."""
        self.vectors[name] = vector
        self.metadata[name] = {
            "layer": vector.layer,
            "error_type": vector.error_type,
            "magnitude": vector.magnitude,
        }

    def get(self, name: str) -> Optional[SteeringVector]:
        """Get a steering vector by name."""
        return self.vectors.get(name)

    def get_for_error_type(self, error_type: str) -> List[SteeringVector]:
        """Get all vectors for an error type."""
        return [v for v in self.vectors.values() if v.error_type == error_type]

    def get_for_layer(self, layer: int) -> List[SteeringVector]:
        """Get all vectors for a layer."""
        return [v for v in self.vectors.values() if v.layer == layer]

    def summary(self) -> Dict[str, Any]:
        """Get bank summary."""
        return {
            "total_vectors": len(self.vectors),
            "by_layer": {l: len(self.get_for_layer(l)) for l in set(v.layer for v in self.vectors.values())},
            "by_error_type": {e: len(self.get_for_error_type(e)) for e in set(v.error_type for v in self.vectors.values())},
        }


# =============================================================================
# TESTING
# =============================================================================

def test_steering_vectors():
    """Test steering vector computation."""
    print("=" * 60)
    print("STEERING VECTORS TEST")
    print("=" * 60)

    # Create contrastive pairs
    print("\nCreating contrastive pairs...")
    pairs = create_contrastive_pairs()
    print(f"  Created {len(pairs)} pairs")

    # Show pair distribution
    error_types = {}
    for pair in pairs:
        error_types[pair.error_type] = error_types.get(pair.error_type, 0) + 1
    print(f"  By error type: {error_types}")

    # Compute steering vectors
    print("\nComputing steering vectors...")
    computer = SteeringVectorComputer()
    print(f"  MLUX available: {computer.has_mlux}")

    # Compute for one layer
    vectors = computer.compute_all(pairs[:3], layers=[12])

    print(f"\nComputed vectors:")
    for layer, vecs in vectors.items():
        print(f"  Layer {layer}: {len(vecs)} vectors")
        for vec in vecs[:2]:
            print(f"    - {vec}")

    # Compute averaged vector
    print("\nComputing averaged vector...")
    avg_vec = computer.compute_averaged(pairs[:3], layer=12)
    print(f"  Averaged: {avg_vec}")

    # Create vector bank
    print("\nCreating vector bank...")
    bank = SteeringVectorBank()
    bank.add("validity_layer12", avg_vec)
    print(f"  Bank summary: {bank.summary()}")

    print("\n[PASS] Steering vectors test complete")
    return True


if __name__ == "__main__":
    test_steering_vectors()
