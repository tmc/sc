"""
SAE for Character Sequences.

Trains a Sparse Autoencoder on character sequence embeddings.
SAE features become the states for regex matching.

Key design decisions:
1. CHARACTER-LEVEL: Process one char at a time (like DFA)
2. POSITIONAL: Include position info (prefix length)
3. RECURRENT: Hidden state accumulates history
4. SPARSE: TopK activation for clean state boundaries
"""

import math
import random
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Set
from collections import defaultdict


@dataclass
class SAEConfig:
    """Configuration for character sequence SAE."""
    # Embedding dimensions
    char_dim: int = 16          # Character embedding dimension
    hidden_dim: int = 32        # RNN hidden state dimension

    # SAE dimensions
    n_features: int = 64        # Number of SAE features (potential states)
    k_active: int = 4           # How many features active at once

    # Training
    learning_rate: float = 0.01
    sparsity_weight: float = 0.1
    n_epochs: int = 100

    # Alphabet
    alphabet: str = "abcdefghijklmnopqrstuvwxyz0123456789"


class CharEncoder:
    """Encode characters as vectors."""

    def __init__(self, config: SAEConfig):
        self.config = config
        self.char_dim = config.char_dim
        self.alphabet = config.alphabet

        # Character to index
        self.char_to_idx = {c: i for i, c in enumerate(self.alphabet)}
        self.idx_to_char = {i: c for i, c in enumerate(self.alphabet)}

        # Random embeddings (would be learned in full implementation)
        random.seed(42)
        self.embeddings: Dict[str, List[float]] = {}
        for c in self.alphabet:
            self.embeddings[c] = [random.gauss(0, 1) for _ in range(self.char_dim)]

        # Special tokens
        self.embeddings['<START>'] = [0.0] * self.char_dim
        self.embeddings['<END>'] = [1.0] * self.char_dim
        self.embeddings['<UNK>'] = [0.5] * self.char_dim

    def encode(self, char: str) -> List[float]:
        """Encode a single character."""
        if char in self.embeddings:
            return self.embeddings[char].copy()
        return self.embeddings['<UNK>'].copy()

    def encode_sequence(self, s: str) -> List[List[float]]:
        """Encode a string as sequence of vectors."""
        return [self.encode(c) for c in s]


class PrefixEmbedder:
    """
    Embed string prefixes using a simple RNN-like accumulator.

    Each prefix gets a hidden state that captures:
    1. Characters seen so far
    2. Position in string
    3. Pattern information (repetitions, etc.)
    """

    def __init__(self, config: SAEConfig, char_encoder: CharEncoder):
        self.config = config
        self.char_encoder = char_encoder
        self.hidden_dim = config.hidden_dim
        self.char_dim = config.char_dim

        # Simple linear weights (would be learned in full implementation)
        # W_h: hidden_dim x hidden_dim, W_x: hidden_dim x char_dim
        random.seed(43)
        self.W_h = [[random.gauss(0, 0.1) for _ in range(self.hidden_dim)]
                    for _ in range(self.hidden_dim)]
        self.W_x = [[random.gauss(0, 0.1) for _ in range(self.char_dim)]
                    for _ in range(self.hidden_dim)]
        self.b_h = [0.0] * self.hidden_dim

    def _matmul(self, W: List[List[float]], x: List[float]) -> List[float]:
        """Simple matrix-vector multiplication."""
        return [sum(W[i][j] * x[j] for j in range(len(x))) for i in range(len(W))]

    def _add(self, a: List[float], b: List[float]) -> List[float]:
        """Vector addition."""
        return [a[i] + b[i] for i in range(len(a))]

    def _tanh(self, x: List[float]) -> List[float]:
        """Element-wise tanh."""
        return [math.tanh(v) for v in x]

    def embed_prefix(self, s: str) -> List[float]:
        """
        Embed a string prefix.

        Returns the hidden state after processing all characters.
        This hidden state represents "everything seen so far".
        """
        h = [0.0] * self.hidden_dim

        for char in s:
            x = self.char_encoder.encode(char)

            # Simple RNN update: h = tanh(W_h @ h + W_x @ x + b)
            h_contrib = self._matmul(self.W_h, h)
            x_contrib = self._matmul(self.W_x, x)
            h = self._tanh(self._add(self._add(h_contrib, x_contrib), self.b_h))

        return h

    def embed_all_prefixes(self, s: str) -> List[Tuple[str, List[float]]]:
        """
        Embed all prefixes of a string.

        Returns [(prefix, embedding), ...] for prefixes of increasing length.
        Includes empty prefix.
        """
        prefixes = []
        h = [0.0] * self.hidden_dim

        # Empty prefix
        prefixes.append(("", h.copy()))

        for i, char in enumerate(s):
            x = self.char_encoder.encode(char)
            h_contrib = self._matmul(self.W_h, h)
            x_contrib = self._matmul(self.W_x, x)
            h = self._tanh(self._add(self._add(h_contrib, x_contrib), self.b_h))

            prefix = s[:i+1]
            prefixes.append((prefix, h.copy()))

        return prefixes


class CharSequenceSAE:
    """
    Sparse Autoencoder for character sequence embeddings.

    Maps prefix embeddings to sparse feature activations.
    Each feature can be interpreted as a "state" in the DFA.

    Architecture:
        input (hidden_dim) → encoder → features (n_features)
                                           ↓ TopK
                                       sparse_features
                                           ↓
        output (hidden_dim) ← decoder ← sparse_features
    """

    def __init__(self, config: SAEConfig = None):
        self.config = config or SAEConfig()

        # Components
        self.char_encoder = CharEncoder(self.config)
        self.prefix_embedder = PrefixEmbedder(self.config, self.char_encoder)

        # SAE weights
        random.seed(44)
        hidden_dim = self.config.hidden_dim
        n_features = self.config.n_features

        # Encoder: hidden_dim → n_features
        self.W_enc = [[random.gauss(0, 0.1) for _ in range(hidden_dim)]
                      for _ in range(n_features)]
        self.b_enc = [0.0] * n_features

        # Decoder: n_features → hidden_dim
        self.W_dec = [[random.gauss(0, 0.1) for _ in range(n_features)]
                      for _ in range(hidden_dim)]
        self.b_dec = [0.0] * hidden_dim

        # Feature statistics
        self.feature_activations: Dict[int, List[str]] = defaultdict(list)
        self.feature_counts: Dict[int, int] = defaultdict(int)

    def _matmul(self, W: List[List[float]], x: List[float]) -> List[float]:
        """Matrix-vector multiplication."""
        return [sum(W[i][j] * x[j] for j in range(len(x))) for i in range(len(W))]

    def _add(self, a: List[float], b: List[float]) -> List[float]:
        """Vector addition."""
        return [a[i] + b[i] for i in range(len(a))]

    def _relu(self, x: List[float]) -> List[float]:
        """Element-wise ReLU."""
        return [max(0, v) for v in x]

    def _topk(self, x: List[float], k: int) -> List[float]:
        """Keep only top-k activations, zero the rest."""
        if k >= len(x):
            return x.copy()

        # Find k-th largest value
        sorted_vals = sorted(x, reverse=True)
        threshold = sorted_vals[k-1] if k > 0 else float('inf')

        # Zero out values below threshold
        return [v if v >= threshold and v > 0 else 0.0 for v in x]

    def encode(self, prefix_embedding: List[float]) -> List[float]:
        """
        Encode prefix embedding to sparse features.

        Returns sparse feature activations (only k_active non-zero).
        """
        # Linear transform + bias
        pre_activation = self._add(
            self._matmul(self.W_enc, prefix_embedding),
            self.b_enc
        )

        # ReLU
        activations = self._relu(pre_activation)

        # TopK sparsity
        sparse = self._topk(activations, self.config.k_active)

        return sparse

    def decode(self, features: List[float]) -> List[float]:
        """Decode features back to embedding space."""
        return self._add(
            self._matmul(self.W_dec, features),
            self.b_dec
        )

    def get_active_features(self, features: List[float]) -> List[int]:
        """Get indices of active (non-zero) features."""
        return [i for i, v in enumerate(features) if v > 0]

    def get_feature_activation(self, prefix: str) -> Tuple[List[float], List[int]]:
        """
        Get feature activations for a prefix.

        Returns:
            (feature_values, active_indices)
        """
        embedding = self.prefix_embedder.embed_prefix(prefix)
        features = self.encode(embedding)
        active = self.get_active_features(features)
        return features, active

    def train(
        self,
        strings: List[str],
        verbose: bool = False
    ) -> Dict[str, float]:
        """
        Train SAE on string prefixes.

        Collects all prefixes from all strings and trains to minimize
        reconstruction loss + sparsity penalty.

        Returns training statistics.
        """
        # Collect all prefix embeddings
        all_prefixes: List[Tuple[str, List[float]]] = []
        for s in strings:
            all_prefixes.extend(self.prefix_embedder.embed_all_prefixes(s))

        if verbose:
            print(f"Training SAE on {len(all_prefixes)} prefix embeddings")
            print(f"Features: {self.config.n_features}, K: {self.config.k_active}")

        # Training loop (simplified gradient descent)
        lr = self.config.learning_rate
        total_loss = 0.0

        for epoch in range(self.config.n_epochs):
            epoch_loss = 0.0
            random.shuffle(all_prefixes)

            for prefix_str, embedding in all_prefixes:
                # Forward pass
                features = self.encode(embedding)
                reconstruction = self.decode(features)

                # Reconstruction loss
                recon_loss = sum((embedding[i] - reconstruction[i])**2
                               for i in range(len(embedding)))

                # Sparsity loss (L1 on features)
                sparsity_loss = sum(abs(f) for f in features) * self.config.sparsity_weight

                loss = recon_loss + sparsity_loss
                epoch_loss += loss

                # Simplified gradient update
                # In practice, would use proper backprop
                for i in range(len(self.b_enc)):
                    if features[i] > 0:
                        # Reduce activation slightly to minimize sparsity
                        self.b_enc[i] -= lr * 0.01 * self.config.sparsity_weight

                # Track feature activations
                active = self.get_active_features(features)
                for idx in active:
                    self.feature_activations[idx].append(prefix_str)
                    self.feature_counts[idx] += 1

            avg_loss = epoch_loss / len(all_prefixes)
            total_loss = avg_loss

            if verbose and (epoch + 1) % 20 == 0:
                print(f"Epoch {epoch + 1}: loss = {avg_loss:.4f}")

        # Compute statistics
        active_features = len([f for f in self.feature_counts.values() if f > 0])

        return {
            'final_loss': total_loss,
            'n_prefixes': len(all_prefixes),
            'active_features': active_features,
            'feature_utilization': active_features / self.config.n_features
        }

    def get_state_for_prefix(self, prefix: str) -> Tuple[int, ...]:
        """
        Get the SAE state (active features) for a prefix.

        Returns a tuple of active feature indices, which serves as
        the discrete state representation.
        """
        _, active = self.get_feature_activation(prefix)
        return tuple(sorted(active))

    def trace_states(self, s: str) -> List[Tuple[str, Tuple[int, ...]]]:
        """
        Trace the SAE states through a string.

        Returns [(prefix, state), ...] for each prefix.
        """
        trace = []
        for i in range(len(s) + 1):
            prefix = s[:i]
            state = self.get_state_for_prefix(prefix)
            trace.append((prefix, state))
        return trace

    def find_accepting_features(
        self,
        positive: List[str],
        negative: List[str]
    ) -> Set[int]:
        """
        Find features that correlate with acceptance.

        A feature is "accepting" if it's more often active at the end
        of positive strings than negative strings.
        """
        positive_end_features: Dict[int, int] = defaultdict(int)
        negative_end_features: Dict[int, int] = defaultdict(int)

        for s in positive:
            _, active = self.get_feature_activation(s)
            for idx in active:
                positive_end_features[idx] += 1

        for s in negative:
            _, active = self.get_feature_activation(s)
            for idx in active:
                negative_end_features[idx] += 1

        # Features that appear more in positive ends
        accepting = set()
        for idx in positive_end_features:
            pos_ratio = positive_end_features[idx] / max(len(positive), 1)
            neg_ratio = negative_end_features.get(idx, 0) / max(len(negative), 1)

            if pos_ratio > neg_ratio + 0.2:  # Threshold for acceptance
                accepting.add(idx)

        return accepting


def train_char_sae(
    strings: List[str],
    config: SAEConfig = None,
    verbose: bool = False
) -> CharSequenceSAE:
    """
    Train a character sequence SAE on the given strings.

    Convenience function that creates and trains SAE.
    """
    config = config or SAEConfig()
    sae = CharSequenceSAE(config)
    stats = sae.train(strings, verbose=verbose)

    if verbose:
        print(f"Training complete: {stats}")

    return sae


def test_sae_regex():
    """Test the character sequence SAE."""
    print("=" * 60)
    print("Testing CharSequenceSAE")
    print("=" * 60)

    # Test strings
    positive = ["ab", "aab", "aaab", "aaaab"]
    negative = ["", "a", "b", "ba", "bb", "baa"]
    all_strings = positive + negative

    # Train SAE
    print("\n1. Training SAE on a+ followed by b:")
    config = SAEConfig(n_features=16, k_active=3, n_epochs=50)
    sae = train_char_sae(all_strings, config, verbose=True)

    # Trace states through examples
    print("\n2. State traces:")
    for s in positive[:2] + negative[:2]:
        trace = sae.trace_states(s)
        label = "+" if s in positive else "-"
        print(f"\n  [{label}] '{s}':")
        for prefix, state in trace:
            print(f"    '{prefix}' -> state {state}")

    # Find accepting features
    print("\n3. Accepting features:")
    accepting = sae.find_accepting_features(positive, negative)
    print(f"  Features correlated with acceptance: {accepting}")

    # Show feature activations
    print("\n4. Feature activation patterns:")
    for idx in sorted(sae.feature_counts.keys())[:5]:
        prefixes = sae.feature_activations[idx][:5]
        count = sae.feature_counts[idx]
        print(f"  Feature {idx} ({count} activations): {prefixes}")

    print("\n" + "=" * 60)
    print("CharSequenceSAE tests complete!")
    print("=" * 60)


if __name__ == "__main__":
    test_sae_regex()
