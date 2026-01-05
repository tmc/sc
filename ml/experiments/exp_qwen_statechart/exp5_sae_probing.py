"""
Experiment 5: SAE State Probing

Goal: Discover which model features correspond to syntax states.

Approach:
1. Collect activations during generation
2. Train SAE on activations to learn sparse features
3. Map which features activate for which syntax states
4. Use discovered states for interpretable sampling constraints
"""

from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from collections import defaultdict

try:
    import mlx.core as mx
    import mlx.nn as nn
    HAS_MLX = True
except ImportError:
    HAS_MLX = False
    mx = None
    nn = None

from .starlark_statechart import StarlarkStatechart, StarlarkSyntaxState


@dataclass
class SyntaxFeatureAssociation:
    """Association between SAE feature and syntax state."""
    feature_id: int
    state: StarlarkSyntaxState
    activation_count: int
    lift: float  # P(state|feature) / P(state)


class SyntaxFeatureMapper:
    """Maps SAE features to syntax states."""

    def __init__(self):
        self.feature_state_counts: Dict[Tuple[int, StarlarkSyntaxState], int] = defaultdict(int)
        self.feature_counts: Dict[int, int] = defaultdict(int)
        self.state_counts: Dict[StarlarkSyntaxState, int] = defaultdict(int)
        self.total_observations = 0

    def record(self, feature_id: int, state: StarlarkSyntaxState):
        """Record a (feature, state) co-occurrence."""
        self.feature_state_counts[(feature_id, state)] += 1
        self.feature_counts[feature_id] += 1
        self.state_counts[state] += 1
        self.total_observations += 1

    def compute_associations(self, min_count: int = 10, min_lift: float = 2.0) -> Dict[StarlarkSyntaxState, List[SyntaxFeatureAssociation]]:
        """Compute feature-state associations with lift metric."""
        associations: Dict[StarlarkSyntaxState, List[SyntaxFeatureAssociation]] = defaultdict(list)

        for (fid, state), count in self.feature_state_counts.items():
            if count < min_count:
                continue

            # Compute lift
            p_state_given_feature = count / max(1, self.feature_counts[fid])
            p_state = self.state_counts[state] / max(1, self.total_observations)

            if p_state > 0:
                lift = p_state_given_feature / p_state
            else:
                lift = 0.0

            if lift >= min_lift:
                associations[state].append(SyntaxFeatureAssociation(
                    feature_id=fid,
                    state=state,
                    activation_count=count,
                    lift=lift,
                ))

        # Sort by lift
        for state in associations:
            associations[state].sort(key=lambda a: a.lift, reverse=True)

        return associations


class SimpleSAE:
    """Simple Sparse Autoencoder for demonstration."""

    def __init__(self, hidden_dim: int = 256, expansion: int = 8):
        self.hidden_dim = hidden_dim
        self.latent_dim = hidden_dim * expansion
        self.sparsity_k = expansion  # Top-k sparsity

        if HAS_MLX:
            self.encoder = mx.random.normal((hidden_dim, self.latent_dim)) * 0.01
            self.decoder = mx.random.normal((self.latent_dim, hidden_dim)) * 0.01
        else:
            self.encoder = None
            self.decoder = None

    def encode(self, x) -> List[int]:
        """Encode and return active feature indices."""
        if not HAS_MLX or self.encoder is None:
            # Demo mode: return random features
            import random
            return random.sample(range(self.latent_dim), self.sparsity_k)

        latent = x @ self.encoder
        latent = mx.maximum(latent, 0)  # ReLU

        # Top-k sparsity
        top_k_indices = mx.argsort(latent)[-self.sparsity_k:]
        return [int(i) for i in top_k_indices]


def run_experiment_5(n_samples: int = 100, verbose: bool = True) -> Dict[str, Any]:
    """Discover syntax state features using SAE probing."""

    statechart = StarlarkStatechart()
    sae = SimpleSAE(hidden_dim=256, expansion=8)
    mapper = SyntaxFeatureMapper()

    # Example code samples
    samples = [
        "def foo():\n    return 1",
        "def bar(x):\n    if x:\n        return x\n    return None",
        "def baz():\n    for i in range(10):\n        pass",
        "load('sc', 'machine')\ndef m():\n    return machine()",
    ]

    # Process samples
    for code in samples * (n_samples // len(samples)):
        statechart.reset()
        tokens = statechart._tokenize(code)

        for token in tokens:
            state = statechart.state

            # Simulate getting activation (would be from model in practice)
            if HAS_MLX:
                activation = mx.random.normal((256,))
            else:
                activation = None

            # Get active features
            active_features = sae.encode(activation)

            # Record associations
            for fid in active_features:
                mapper.record(fid, state)

            # Transition
            statechart.transition(token)

    # Compute associations
    associations = mapper.compute_associations(min_count=5, min_lift=1.5)

    results = {
        'total_observations': mapper.total_observations,
        'unique_features': len(mapper.feature_counts),
        'states_with_associations': len(associations),
        'top_associations': {},
    }

    for state, assocs in associations.items():
        results['top_associations'][state.name] = [
            {'feature': a.feature_id, 'lift': a.lift, 'count': a.activation_count}
            for a in assocs[:3]
        ]

    if verbose:
        print("\n" + "=" * 60)
        print("EXPERIMENT 5 RESULTS: SAE State Probing")
        print("=" * 60)
        print(f"Total observations: {results['total_observations']}")
        print(f"Unique features: {results['unique_features']}")
        print(f"States with associations: {results['states_with_associations']}")
        print("\nTop associations per state:")
        for state, assocs in results['top_associations'].items():
            if assocs:
                print(f"  {state}: {assocs[0]}")

    return results


if __name__ == "__main__":
    print("=" * 60)
    print("EXPERIMENT 5: SAE State Probing")
    print("=" * 60)
    run_experiment_5(n_samples=50, verbose=True)
