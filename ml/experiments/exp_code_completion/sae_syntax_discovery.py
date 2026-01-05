"""
SAE Syntax Discovery

Uses Sparse Autoencoders to discover syntax-related features in LM activations.
Connects discovered SAE features to Python syntax states.

Key insight: SAE features often correspond to grammatical constructs!
We can learn the mapping from features -> syntax states.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional, Tuple
from enum import Enum
from collections import defaultdict
import math
import random

try:
    import mlx.core as mx
    import mlx.nn as nn
    HAS_MLX = True
except ImportError:
    HAS_MLX = False
    mx = None
    nn = None

try:
    from .syntax_statechart import PythonSyntaxState, PythonSyntaxStatechart
except ImportError:
    from syntax_statechart import PythonSyntaxState, PythonSyntaxStatechart


@dataclass
class SAEFeature:
    """A discovered SAE feature."""
    feature_id: int
    activation_threshold: float = 0.5
    associated_states: Set[PythonSyntaxState] = field(default_factory=set)
    associated_tokens: Set[str] = field(default_factory=set)
    description: str = ""
    activation_count: int = 0

    def activation_pattern(self) -> str:
        """Describe when this feature activates."""
        states = ', '.join(s.name for s in self.associated_states)
        tokens = ', '.join(list(self.associated_tokens)[:5])
        return f"Feature {self.feature_id}: states=[{states}], tokens=[{tokens}]"


class SparseAutoencoder(nn.Module if HAS_MLX else object):
    """
    Sparse Autoencoder for extracting interpretable features.

    Architecture:
    - Encoder: hidden_dim -> expansion * hidden_dim (sparse)
    - Decoder: expansion * hidden_dim -> hidden_dim

    The sparse middle layer learns interpretable features.
    """

    def __init__(self, hidden_dim: int, expansion: int = 8, sparsity: float = 0.05):
        if HAS_MLX:
            super().__init__()
            self.hidden_dim = hidden_dim
            self.feature_dim = hidden_dim * expansion
            self.sparsity = sparsity

            # Encoder: project to higher dimension
            self.encoder = nn.Linear(hidden_dim, self.feature_dim)

            # Decoder: project back
            self.decoder = nn.Linear(self.feature_dim, hidden_dim)
        else:
            self.hidden_dim = hidden_dim
            self.feature_dim = hidden_dim * expansion
            self.sparsity = sparsity

    def encode(self, x: 'mx.array') -> 'mx.array':
        """Encode to sparse features."""
        if not HAS_MLX:
            raise RuntimeError("MLX not available")

        # Project to feature space
        features = self.encoder(x)

        # Apply ReLU for sparsity
        features = mx.maximum(features, 0)

        return features

    def decode(self, features: 'mx.array') -> 'mx.array':
        """Decode from features."""
        if not HAS_MLX:
            raise RuntimeError("MLX not available")
        return self.decoder(features)

    def __call__(self, x: 'mx.array') -> Tuple['mx.array', 'mx.array']:
        """Forward pass. Returns (reconstruction, features)."""
        if not HAS_MLX:
            raise RuntimeError("MLX not available")

        features = self.encode(x)
        reconstruction = self.decode(features)
        return reconstruction, features

    def get_active_features(self, x: 'mx.array', threshold: float = 0.1) -> List[int]:
        """Get indices of active features for input."""
        if not HAS_MLX:
            return []

        features = self.encode(x)
        active = mx.where(features > threshold)[0]
        return [int(i) for i in active]


class SyntaxFeatureMapper:
    """
    Maps SAE features to Python syntax states.

    Learns the association between:
    - SAE feature activations
    - Syntax state when feature fires
    - Token that triggered the feature
    """

    def __init__(self, num_features: int):
        self.num_features = num_features
        self.features: Dict[int, SAEFeature] = {
            i: SAEFeature(feature_id=i) for i in range(num_features)
        }

        # Co-occurrence tracking
        self.feature_state_counts: Dict[Tuple[int, PythonSyntaxState], int] = defaultdict(int)
        self.feature_token_counts: Dict[Tuple[int, str], int] = defaultdict(int)
        self.state_counts: Dict[PythonSyntaxState, int] = defaultdict(int)
        self.token_counts: Dict[str, int] = defaultdict(int)

    def record_activation(
        self,
        feature_ids: List[int],
        syntax_state: PythonSyntaxState,
        token: str,
    ):
        """Record which features activated for a state/token pair."""
        self.state_counts[syntax_state] += 1
        self.token_counts[token] += 1

        for fid in feature_ids:
            if fid < self.num_features:
                self.features[fid].activation_count += 1
                self.feature_state_counts[(fid, syntax_state)] += 1
                self.feature_token_counts[(fid, token)] += 1

    def compute_associations(self, min_count: int = 10, min_lift: float = 2.0):
        """
        Compute feature-to-state and feature-to-token associations.

        Uses lift metric: P(state|feature) / P(state)
        High lift = feature is predictive of state
        """
        total = sum(self.state_counts.values())
        if total == 0:
            return

        for fid, feature in self.features.items():
            if feature.activation_count < min_count:
                continue

            # Find associated states
            for state in PythonSyntaxState:
                count = self.feature_state_counts.get((fid, state), 0)
                if count < min_count:
                    continue

                # Compute lift
                p_state_given_feature = count / feature.activation_count
                p_state = self.state_counts[state] / total
                lift = p_state_given_feature / max(p_state, 1e-10)

                if lift >= min_lift:
                    feature.associated_states.add(state)

            # Find associated tokens
            for token, token_count in self.token_counts.items():
                count = self.feature_token_counts.get((fid, token), 0)
                if count < min_count // 2:
                    continue

                p_token_given_feature = count / feature.activation_count
                p_token = token_count / total
                lift = p_token_given_feature / max(p_token, 1e-10)

                if lift >= min_lift:
                    feature.associated_tokens.add(token)

    def get_features_for_state(self, state: PythonSyntaxState) -> List[SAEFeature]:
        """Get features associated with a syntax state."""
        return [f for f in self.features.values() if state in f.associated_states]

    def get_state_from_features(self, feature_ids: List[int]) -> Optional[PythonSyntaxState]:
        """
        Predict syntax state from active features.

        Uses voting from associated states.
        """
        state_votes: Dict[PythonSyntaxState, float] = defaultdict(float)

        for fid in feature_ids:
            if fid in self.features:
                feature = self.features[fid]
                for state in feature.associated_states:
                    # Weight by feature activation count
                    state_votes[state] += math.log(1 + feature.activation_count)

        if not state_votes:
            return None

        return max(state_votes.keys(), key=lambda s: state_votes[s])

    def describe_features(self, min_associations: int = 1) -> str:
        """Generate human-readable description of discovered features."""
        lines = ["=== Discovered Syntax Features ===\n"]

        # Group by associated state
        by_state: Dict[PythonSyntaxState, List[SAEFeature]] = defaultdict(list)

        for feature in self.features.values():
            if len(feature.associated_states) >= min_associations:
                for state in feature.associated_states:
                    by_state[state].append(feature)

        for state in sorted(by_state.keys(), key=lambda s: s.name):
            lines.append(f"\n{state.name}:")
            for feature in sorted(by_state[state], key=lambda f: -f.activation_count)[:5]:
                tokens = ', '.join(list(feature.associated_tokens)[:3])
                lines.append(f"  Feature {feature.feature_id}: "
                             f"count={feature.activation_count}, tokens=[{tokens}]")

        return '\n'.join(lines)


class SAESyntaxDiscovery:
    """
    End-to-end SAE-based syntax feature discovery.

    Pipeline:
    1. Train SAE on LM activations
    2. Collect (activation, token, syntax_state) tuples
    3. Learn feature-to-state mapping
    4. Use features for syntax prediction
    """

    def __init__(self, hidden_dim: int, expansion: int = 8):
        self.hidden_dim = hidden_dim
        self.expansion = expansion
        self.feature_dim = hidden_dim * expansion

        if HAS_MLX:
            self.sae = SparseAutoencoder(hidden_dim, expansion)
        else:
            self.sae = None

        self.mapper = SyntaxFeatureMapper(self.feature_dim)
        self.statechart = PythonSyntaxStatechart()

        # Training data
        self.activations: List[List[float]] = []
        self.tokens: List[str] = []
        self.states: List[PythonSyntaxState] = []

    def collect_sample(
        self,
        activation: List[float],
        token: str,
    ):
        """
        Collect a training sample.

        Records the activation, token, and current syntax state.
        """
        self.activations.append(activation)
        self.tokens.append(token)
        self.states.append(self.statechart.state)

        # Update statechart for next token
        self.statechart.transition(token)

    def train_sae(self, num_epochs: int = 100, lr: float = 0.001) -> Dict[str, float]:
        """
        Train the SAE on collected activations.

        Returns training statistics.
        """
        if not HAS_MLX or not self.activations:
            return {'error': 'MLX not available or no data'}

        # Convert to MLX arrays
        X = mx.array(self.activations)

        # Training loop
        optimizer = nn.optimizers.Adam(learning_rate=lr)

        losses = []
        for epoch in range(num_epochs):
            # Forward pass
            reconstruction, features = self.sae(X)

            # Reconstruction loss
            recon_loss = mx.mean((X - reconstruction) ** 2)

            # Sparsity loss (L1 on features)
            sparsity_loss = mx.mean(mx.abs(features)) * 0.01

            loss = recon_loss + sparsity_loss
            losses.append(float(loss))

            # Backward pass
            grads = mx.grad(lambda: float(loss))(self.sae.parameters())

            # Update
            optimizer.update(self.sae, grads)

            if epoch % 20 == 0:
                print(f"Epoch {epoch}: loss={float(loss):.4f}, "
                      f"recon={float(recon_loss):.4f}, "
                      f"sparsity={float(sparsity_loss):.4f}")

        return {
            'final_loss': losses[-1] if losses else 0,
            'loss_history': losses,
        }

    def learn_feature_mapping(self, threshold: float = 0.1):
        """
        Learn mapping from SAE features to syntax states.

        Runs collected activations through trained SAE and
        records which features fire for each state/token.
        """
        if not HAS_MLX:
            # Simulate for testing
            self._learn_mapping_simulated()
            return

        for i, (activation, token, state) in enumerate(
            zip(self.activations, self.tokens, self.states)
        ):
            # Get active features
            x = mx.array([activation])
            features = self.sae.encode(x)
            active = mx.where(features[0] > threshold)[0]
            active_ids = [int(idx) for idx in active]

            # Record associations
            self.mapper.record_activation(active_ids, state, token)

        # Compute final associations
        self.mapper.compute_associations()

    def _learn_mapping_simulated(self):
        """Simulated feature learning without MLX."""
        # Create synthetic feature activations based on tokens
        token_to_features = {
            'def': [0, 1, 2],
            'class': [0, 1, 3],
            'if': [4, 5],
            'else': [4, 6],
            'for': [7, 8],
            'while': [7, 9],
            'return': [10, 11],
            '(': [12, 13],
            ')': [12, 14],
            ':': [15, 16],
            '=': [17, 18],
            'IDENTIFIER': [19, 20, 21],
        }

        for token, state in zip(self.tokens, self.states):
            # Get features for token
            features = token_to_features.get(token, [])
            if not features:
                # Use hash for other tokens
                features = [hash(token) % 50, (hash(token) // 50) % 50]

            self.mapper.record_activation(features, state, token)

        self.mapper.compute_associations()

    def predict_state(self, activation: List[float]) -> Optional[PythonSyntaxState]:
        """Predict syntax state from activation."""
        if not HAS_MLX:
            return None

        x = mx.array([activation])
        features = self.sae.encode(x)
        active = mx.where(features[0] > 0.1)[0]
        active_ids = [int(idx) for idx in active]

        return self.mapper.get_state_from_features(active_ids)

    def get_interpretable_features(self) -> List[SAEFeature]:
        """Get features with clear syntax associations."""
        return [
            f for f in self.mapper.features.values()
            if len(f.associated_states) > 0 and f.activation_count > 10
        ]

    def describe(self) -> str:
        """Describe discovered syntax features."""
        lines = [
            "=== SAE Syntax Discovery Results ===\n",
            f"Total samples: {len(self.activations)}",
            f"Feature dimension: {self.feature_dim}",
            "",
            self.mapper.describe_features(),
        ]
        return '\n'.join(lines)


# Demo
if __name__ == "__main__":
    print("=" * 60)
    print("SAE SYNTAX DISCOVERY DEMO")
    print("=" * 60)

    # Create discovery system
    discovery = SAESyntaxDiscovery(hidden_dim=64, expansion=4)

    # Simulate collecting samples
    code_tokens = [
        'def', 'foo', '(', 'x', ')', ':', '\n',
        'if', 'x', '>', '0', ':', '\n',
        'return', 'x', '\n',
        'return', 'None', '\n',
    ]

    print("\nCollecting samples from code tokens...")
    for token in code_tokens:
        # Simulate activation (random for demo)
        activation = [random.gauss(0, 1) for _ in range(64)]
        discovery.collect_sample(activation, token)

    print(f"Collected {len(discovery.activations)} samples")

    # Learn mapping (simulated)
    print("\nLearning feature-to-state mapping...")
    discovery.learn_feature_mapping()

    # Show results
    print("\n" + discovery.describe())

    # Show interpretable features
    features = discovery.get_interpretable_features()
    print(f"\nFound {len(features)} interpretable features")
