"""
SAE State Extractor: SAE Features ARE States

Key insight: No clustering needed. Each SAE feature IS a state.
The active feature set IS the state configuration.

From exp_sae_statechart:
- TopK SAE guarantees exactly k features active
- Feature co-activation patterns = compound states
- Transitions = changes in active feature sets

This module extracts states directly from SAE encodings.
"""

import mlx.core as mx
import mlx.nn as nn
from dataclasses import dataclass, field
from typing import List, Dict, Set, Tuple, Optional, FrozenSet
from collections import defaultdict


@dataclass
class SAEConfig:
    """Configuration for SAE state extraction."""
    input_dim: int = 256
    expansion_factor: int = 16
    k_active: int = 32
    state_threshold: float = 0.1  # Minimum activation to count as "in state"


@dataclass
class FeatureState:
    """
    A single SAE feature as a state.

    Each feature has:
    - An index (its position in the SAE latent space)
    - An activation level (how "active" it is)
    - Semantic meaning (learned, not hardcoded)
    """
    index: int
    activation: float = 0.0
    label: Optional[str] = None  # Optional human-readable label

    # Statistics for this feature
    activation_count: int = 0
    total_activations: float = 0.0

    @property
    def mean_activation(self) -> float:
        if self.activation_count == 0:
            return 0.0
        return self.total_activations / self.activation_count

    def __hash__(self):
        return hash(self.index)

    def __eq__(self, other):
        if not isinstance(other, FeatureState):
            return False
        return self.index == other.index


@dataclass
class StateConfiguration:
    """
    A configuration of active states (SAE features).

    This IS the statechart configuration:
    - Which features are active = which states we're in
    - Activation levels = "how much" we're in each state
    """
    active_features: FrozenSet[int]
    activations: Dict[int, float] = field(default_factory=dict)

    # Source information
    source_hidden: Optional[mx.array] = None
    step: int = 0

    def __hash__(self):
        return hash(self.active_features)

    def __eq__(self, other):
        if not isinstance(other, StateConfiguration):
            return False
        return self.active_features == other.active_features

    def similarity(self, other: "StateConfiguration") -> float:
        """Jaccard similarity between configurations."""
        if not self.active_features and not other.active_features:
            return 1.0
        intersection = len(self.active_features & other.active_features)
        union = len(self.active_features | other.active_features)
        return intersection / union if union > 0 else 0.0

    def difference(self, other: "StateConfiguration") -> Tuple[Set[int], Set[int]]:
        """
        Compute difference between configurations.

        Returns:
            (entered, exited) - features that were entered/exited
        """
        entered = self.active_features - other.active_features
        exited = other.active_features - self.active_features
        return set(entered), set(exited)


class TopKSAE(nn.Module):
    """
    TopK Sparse Autoencoder for state extraction.

    Guarantees exactly k features active at each step.
    """

    def __init__(self, config: SAEConfig):
        super().__init__()
        self.config = config
        self.latent_dim = config.input_dim * config.expansion_factor

        self.encoder = nn.Linear(config.input_dim, self.latent_dim)
        self.decoder = nn.Linear(self.latent_dim, config.input_dim, bias=False)

        # Feature statistics
        self.feature_counts = mx.zeros((self.latent_dim,))
        self.feature_sums = mx.zeros((self.latent_dim,))

    def encode(self, x: mx.array) -> Tuple[mx.array, mx.array]:
        """
        Encode to sparse TopK representation.

        Returns:
            activations: (batch, latent_dim) sparse tensor
            indices: (batch, k) active feature indices
        """
        pre_acts = self.encoder(x)

        # TopK selection
        k = self.config.k_active
        sorted_indices = mx.argsort(-pre_acts, axis=-1)
        top_indices = sorted_indices[:, :k]

        # Build sparse activation tensor
        batch_size = x.shape[0]
        activations = mx.zeros((batch_size, self.latent_dim))

        for b in range(batch_size):
            for i in range(k):
                idx = int(top_indices[b, i])
                val = float(mx.maximum(pre_acts[b, idx], 0.0))
                # Direct assignment in loop (MLX doesn't support advanced indexing)
                row = list(activations[b])
                row[idx] = val
                activations = mx.concatenate([
                    activations[:b],
                    mx.array([row]),
                    activations[b+1:]
                ], axis=0) if b < batch_size - 1 or b > 0 else mx.array([row])

        return activations, top_indices

    def decode(self, activations: mx.array) -> mx.array:
        """Decode sparse representation."""
        return self.decoder(activations)

    def __call__(self, x: mx.array) -> Tuple[mx.array, mx.array, mx.array]:
        """Forward pass."""
        activations, indices = self.encode(x)
        reconstruction = self.decode(activations)
        return reconstruction, activations, indices


class SAEStateExtractor:
    """
    Extract states from hidden representations using SAE.

    Key principle: SAE features ARE states. No clustering.
    """

    def __init__(self, config: SAEConfig = None):
        self.config = config or SAEConfig()
        self.sae = TopKSAE(self.config)

        # State registry
        self.feature_states: Dict[int, FeatureState] = {}

        # Transition tracking
        self.transitions: Dict[Tuple[FrozenSet[int], FrozenSet[int]], int] = defaultdict(int)

        # Configuration history
        self.config_history: List[StateConfiguration] = []

    def extract_state(self, hidden: mx.array) -> StateConfiguration:
        """
        Extract state configuration from hidden representation.

        Args:
            hidden: (dim,) or (1, dim) hidden state from model

        Returns:
            StateConfiguration representing active states
        """
        if hidden.ndim == 1:
            hidden = hidden.reshape(1, -1)

        _, activations, indices = self.sae(hidden)

        # Build configuration
        active_set = set()
        activation_dict = {}

        k = self.config.k_active
        for i in range(k):
            idx = int(indices[0, i])
            act = float(activations[0, idx]) if activations[0, idx] > self.config.state_threshold else 0.0

            if act > 0:
                active_set.add(idx)
                activation_dict[idx] = act

                # Update feature state
                if idx not in self.feature_states:
                    self.feature_states[idx] = FeatureState(index=idx)
                self.feature_states[idx].activation_count += 1
                self.feature_states[idx].total_activations += act

        config = StateConfiguration(
            active_features=frozenset(active_set),
            activations=activation_dict,
            source_hidden=hidden,
            step=len(self.config_history),
        )

        # Track transition
        if self.config_history:
            prev = self.config_history[-1]
            trans_key = (prev.active_features, config.active_features)
            self.transitions[trans_key] += 1

        self.config_history.append(config)

        return config

    def extract_sequence(self, hidden_sequence: mx.array) -> List[StateConfiguration]:
        """
        Extract state configurations from a sequence of hidden states.

        Args:
            hidden_sequence: (seq_len, dim) hidden states

        Returns:
            List of StateConfiguration for each step
        """
        configs = []
        for t in range(hidden_sequence.shape[0]):
            config = self.extract_state(hidden_sequence[t])
            configs.append(config)
        return configs

    def get_feature_state(self, idx: int) -> Optional[FeatureState]:
        """Get feature state by index."""
        return self.feature_states.get(idx)

    def get_all_active_features(self) -> Set[int]:
        """Get set of all features that have ever been active."""
        return set(self.feature_states.keys())

    def get_transition_counts(self) -> Dict[Tuple[FrozenSet[int], FrozenSet[int]], int]:
        """Get transition counts between configurations."""
        return dict(self.transitions)

    def get_common_transitions(self, min_count: int = 2) -> List[Tuple[FrozenSet[int], FrozenSet[int], int]]:
        """Get transitions that occurred at least min_count times."""
        common = []
        for (src, tgt), count in self.transitions.items():
            if count >= min_count:
                common.append((src, tgt, count))
        return sorted(common, key=lambda x: -x[2])

    def get_feature_co_occurrences(self) -> Dict[Tuple[int, int], int]:
        """
        Get co-occurrence counts between features.

        Features that often co-occur might form a "superstate".
        """
        co_occur = defaultdict(int)
        for config in self.config_history:
            features = list(config.active_features)
            for i in range(len(features)):
                for j in range(i + 1, len(features)):
                    pair = (min(features[i], features[j]), max(features[i], features[j]))
                    co_occur[pair] += 1
        return dict(co_occur)

    def discover_superstates(self, min_co_occurrence: float = 0.8) -> List[Set[int]]:
        """
        Discover superstates from co-occurrence patterns.

        Features that co-occur >80% of the time likely form a superstate.
        """
        co_occur = self.get_feature_co_occurrences()

        # Normalize by individual feature counts
        superstates = []
        checked = set()

        for (f1, f2), count in co_occur.items():
            if f1 in checked or f2 in checked:
                continue

            count1 = self.feature_states.get(f1, FeatureState(f1)).activation_count
            count2 = self.feature_states.get(f2, FeatureState(f2)).activation_count

            if count1 == 0 or count2 == 0:
                continue

            # Check if co-occurrence rate is high enough
            rate = count / min(count1, count2)
            if rate >= min_co_occurrence:
                superstates.append({f1, f2})
                checked.add(f1)
                checked.add(f2)

        return superstates

    def reset(self):
        """Reset extraction state."""
        self.config_history = []
        self.transitions.clear()

    def get_statistics(self) -> Dict:
        """Get extraction statistics."""
        return {
            'total_features_seen': len(self.feature_states),
            'total_configurations': len(self.config_history),
            'unique_configurations': len(set(self.config_history)),
            'total_transitions': sum(self.transitions.values()),
            'unique_transitions': len(self.transitions),
        }


def demo():
    """Demonstrate SAE state extraction."""
    print("=" * 60)
    print("SAE STATE EXTRACTOR: Features ARE States")
    print("=" * 60)

    config = SAEConfig(input_dim=64, expansion_factor=8, k_active=8)
    extractor = SAEStateExtractor(config)

    # Generate synthetic hidden states
    mx.random.seed(42)
    hidden_sequence = mx.random.normal((20, 64))

    print(f"\nExtracting states from {hidden_sequence.shape[0]} hidden states...")

    configs = extractor.extract_sequence(hidden_sequence)

    print(f"\nExtracted {len(configs)} configurations")
    print(f"Statistics: {extractor.get_statistics()}")

    # Show some configurations
    print("\nSample configurations:")
    for i, config in enumerate(configs[:5]):
        print(f"  Step {i}: {len(config.active_features)} active features")
        print(f"    Features: {sorted(config.active_features)[:5]}...")

    # Show transitions
    common_trans = extractor.get_common_transitions(min_count=1)
    print(f"\nFound {len(common_trans)} transitions")

    # Show superstates
    superstates = extractor.discover_superstates(min_co_occurrence=0.5)
    print(f"\nDiscovered {len(superstates)} potential superstates")

    return extractor


if __name__ == "__main__":
    demo()
