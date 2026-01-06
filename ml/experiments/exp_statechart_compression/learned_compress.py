"""
Learned Statechart Compression

Uses neural autoencoders to learn compressed representations of statecharts.

APPROACH:
1. Encode statechart structure as a fixed-size tensor
2. Train autoencoder to compress and reconstruct
3. Use latent space as compressed representation

ADVANTAGES:
- Learns domain-specific compression patterns
- Can capture semantic regularities
- Adaptive to statechart distribution

ARCHITECTURE:
- Encoder: Statechart -> Latent vector
- Decoder: Latent vector -> Statechart
- Loss: Reconstruction accuracy + size penalty
"""

import mlx.core as mx
import mlx.nn as nn
import mlx.utils
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Set
import random

from .bisimulation import Statechart, State, Transition, StateType


# =============================================================================
# Statechart Encoding
# =============================================================================

@dataclass
class StatechartEncoding:
    """
    Fixed-size tensor representation of a statechart.

    Enables neural network processing.
    """
    max_states: int = 32
    max_transitions: int = 64
    max_events: int = 16
    embedding_dim: int = 16

    def encode(self, sc: Statechart) -> Dict[str, mx.array]:
        """
        Encode statechart to tensor representation.

        Returns dict with:
        - state_features: [max_states, state_feat_dim]
        - transition_matrix: [max_states, max_states, trans_feat_dim]
        - mask: [max_states] which states are valid
        """
        n_states = min(len(sc.states), self.max_states)

        # State label to index mapping
        label_to_idx = {label: i for i, label in enumerate(sorted(sc.states.keys())[:n_states])}

        # State features: [is_initial, is_final, type_onehot(3), has_entry, has_exit]
        state_feat_dim = 7
        state_features_list = []

        # Build state features as a list then convert to array
        for i in range(self.max_states):
            if i < n_states:
                label = list(label_to_idx.keys())[i]
                state = sc.states[label]
                features = [
                    float(state.is_initial),
                    float(state.is_final),
                    float(state.state_type == StateType.BASIC),
                    float(state.state_type == StateType.OR),
                    float(state.state_type == StateType.AND),
                    float(state.entry_action is not None),
                    float(state.exit_action is not None),
                ]
            else:
                features = [0.0] * state_feat_dim
            state_features_list.append(features)

        state_features = mx.array(state_features_list)

        # Transition matrix: [src, tgt, features]
        # Features: [has_transition, event_hash, has_guard, has_action]
        trans_feat_dim = 4

        # Build transition matrix as a list
        trans_matrix_list = [[[0.0] * trans_feat_dim for _ in range(self.max_states)]
                             for _ in range(self.max_states)]

        for t in sc.transitions:
            if t.source in label_to_idx and t.target in label_to_idx:
                src_idx = label_to_idx[t.source]
                tgt_idx = label_to_idx[t.target]

                # Hash event to normalized value
                event_hash = hash(t.event) % 1000 / 1000.0

                trans_matrix_list[src_idx][tgt_idx] = [
                    1.0,  # has_transition
                    event_hash,
                    float(t.guard is not None),
                    float(t.action is not None),
                ]

        trans_matrix = mx.array(trans_matrix_list)

        # Mask for valid states
        mask_list = [1.0 if i < n_states else 0.0 for i in range(self.max_states)]
        mask = mx.array(mask_list)

        return {
            "state_features": state_features,
            "transition_matrix": trans_matrix,
            "mask": mask,
            "n_states": n_states,
            "label_to_idx": label_to_idx,
        }

    def decode(
        self,
        state_features: mx.array,
        trans_matrix: mx.array,
        mask: mx.array,
        threshold: float = 0.5,
    ) -> Statechart:
        """
        Decode tensor representation back to statechart.

        Uses thresholds to binarize continuous values.
        """
        sc = Statechart(name="decoded")

        # Decode states
        n_states = int(mx.sum(mask))
        for i in range(n_states):
            feat = state_features[i]

            # Decode type
            type_probs = feat[2:5]
            state_type = StateType(int(mx.argmax(type_probs)))

            state = State(
                label=f"s{i}",
                is_initial=float(feat[0]) > threshold,
                is_final=float(feat[1]) > threshold,
                state_type=state_type,
                entry_action="entry" if float(feat[5]) > threshold else None,
                exit_action="exit" if float(feat[6]) > threshold else None,
            )
            sc.add_state(state)

        # Decode transitions
        for src in range(n_states):
            for tgt in range(n_states):
                feat = trans_matrix[src, tgt]
                if float(feat[0]) > threshold:
                    sc.add_transition(Transition(
                        source=f"s{src}",
                        target=f"s{tgt}",
                        event=f"e{int(float(feat[1]) * 16)}",
                        guard="guard" if float(feat[2]) > threshold else None,
                        action="action" if float(feat[3]) > threshold else None,
                    ))

        return sc


# =============================================================================
# Autoencoder Architecture
# =============================================================================

class StatechartEncoder(nn.Module):
    """
    Encodes statechart tensors to latent representation.
    """

    def __init__(
        self,
        max_states: int = 32,
        state_feat_dim: int = 7,
        trans_feat_dim: int = 4,
        hidden_dim: int = 64,
        latent_dim: int = 32,
    ):
        super().__init__()
        self.max_states = max_states
        self.state_feat_dim = state_feat_dim
        self.trans_feat_dim = trans_feat_dim
        self.latent_dim = latent_dim

        # State encoder
        self.state_enc = nn.Sequential(
            nn.Linear(state_feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
        )

        # Transition encoder (flatten and project)
        trans_input_dim = max_states * max_states * trans_feat_dim
        self.trans_enc = nn.Sequential(
            nn.Linear(trans_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
        )

        # Combine and project to latent
        self.combine = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def __call__(
        self,
        state_features: mx.array,  # [B, max_states, state_feat_dim]
        trans_matrix: mx.array,    # [B, max_states, max_states, trans_feat_dim]
        mask: mx.array,            # [B, max_states]
    ) -> mx.array:
        """
        Encode to latent representation.

        Returns: [B, latent_dim]
        """
        B = state_features.shape[0]

        # Encode states (pooled)
        state_enc = self.state_enc(state_features)  # [B, max_states, hidden//2]
        # Masked mean pooling
        mask_expanded = mx.expand_dims(mask, -1)  # [B, max_states, 1]
        state_pooled = mx.sum(state_enc * mask_expanded, axis=1) / (mx.sum(mask, axis=1, keepdims=True) + 1e-6)

        # Encode transitions (flattened)
        trans_flat = trans_matrix.reshape(B, -1)  # [B, max_states^2 * trans_feat_dim]
        trans_enc = self.trans_enc(trans_flat)  # [B, hidden//2]

        # Combine
        combined = mx.concatenate([state_pooled, trans_enc], axis=-1)  # [B, hidden]
        latent = self.combine(combined)  # [B, latent_dim]

        return latent


class StatechartDecoder(nn.Module):
    """
    Decodes latent representation back to statechart tensors.
    """

    def __init__(
        self,
        max_states: int = 32,
        state_feat_dim: int = 7,
        trans_feat_dim: int = 4,
        hidden_dim: int = 64,
        latent_dim: int = 32,
    ):
        super().__init__()
        self.max_states = max_states
        self.state_feat_dim = state_feat_dim
        self.trans_feat_dim = trans_feat_dim

        # Expand latent to hidden
        self.expand = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Decode to state features
        self.state_dec = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, max_states * state_feat_dim),
        )

        # Decode to transitions
        trans_output_dim = max_states * max_states * trans_feat_dim
        self.trans_dec = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, trans_output_dim),
        )

    def __call__(self, latent: mx.array) -> Tuple[mx.array, mx.array]:
        """
        Decode from latent representation.

        Returns: (state_features, trans_matrix)
        """
        B = latent.shape[0]

        # Expand
        hidden = self.expand(latent)  # [B, hidden]

        # Split and decode
        hidden_state = hidden[:, :hidden.shape[1]//2]
        hidden_trans = hidden[:, hidden.shape[1]//2:]

        # State features
        state_flat = self.state_dec(hidden_state)  # [B, max_states * state_feat_dim]
        state_features = state_flat.reshape(B, self.max_states, self.state_feat_dim)
        state_features = mx.sigmoid(state_features)  # Normalize to [0, 1]

        # Transition matrix
        trans_flat = self.trans_dec(hidden_trans)  # [B, max_states^2 * trans_feat_dim]
        trans_matrix = trans_flat.reshape(B, self.max_states, self.max_states, self.trans_feat_dim)
        trans_matrix = mx.sigmoid(trans_matrix)  # Normalize to [0, 1]

        return state_features, trans_matrix


class StatechartAutoencoder(nn.Module):
    """
    Complete autoencoder for statechart compression.
    """

    def __init__(
        self,
        max_states: int = 32,
        state_feat_dim: int = 7,
        trans_feat_dim: int = 4,
        hidden_dim: int = 64,
        latent_dim: int = 32,
    ):
        super().__init__()
        self.encoder = StatechartEncoder(
            max_states, state_feat_dim, trans_feat_dim, hidden_dim, latent_dim
        )
        self.decoder = StatechartDecoder(
            max_states, state_feat_dim, trans_feat_dim, hidden_dim, latent_dim
        )
        self.latent_dim = latent_dim

    def __call__(
        self,
        state_features: mx.array,
        trans_matrix: mx.array,
        mask: mx.array,
    ) -> Tuple[mx.array, mx.array, mx.array]:
        """
        Full forward pass.

        Returns: (latent, recon_states, recon_trans)
        """
        latent = self.encoder(state_features, trans_matrix, mask)
        recon_states, recon_trans = self.decoder(latent)
        return latent, recon_states, recon_trans

    def compress(
        self,
        state_features: mx.array,
        trans_matrix: mx.array,
        mask: mx.array,
    ) -> mx.array:
        """Compress to latent representation."""
        return self.encoder(state_features, trans_matrix, mask)

    def decompress(self, latent: mx.array) -> Tuple[mx.array, mx.array]:
        """Decompress from latent representation."""
        return self.decoder(latent)


# =============================================================================
# Training
# =============================================================================

def reconstruction_loss(
    orig_states: mx.array,
    orig_trans: mx.array,
    recon_states: mx.array,
    recon_trans: mx.array,
    mask: mx.array,
) -> mx.array:
    """Compute reconstruction loss."""
    # State reconstruction (masked MSE)
    mask_expanded = mx.expand_dims(mask, -1)
    state_loss = mx.mean((orig_states - recon_states) ** 2 * mask_expanded)

    # Transition reconstruction (MSE)
    trans_loss = mx.mean((orig_trans - recon_trans) ** 2)

    return state_loss + trans_loss


def train_autoencoder(
    model: StatechartAutoencoder,
    statecharts: List[Statechart],
    n_epochs: int = 100,
    learning_rate: float = 0.01,
    verbose: bool = True,
) -> List[float]:
    """
    Train the autoencoder on a collection of statecharts.
    """
    encoding = StatechartEncoding(max_states=32)
    losses = []

    # Encode all statecharts
    encoded = [encoding.encode(sc) for sc in statecharts]

    # Stack into batches
    state_features = mx.stack([e["state_features"] for e in encoded])
    trans_matrices = mx.stack([e["transition_matrix"] for e in encoded])
    masks = mx.stack([e["mask"] for e in encoded])

    def loss_fn(params):
        model.update(params)
        latent, recon_states, recon_trans = model(state_features, trans_matrices, masks)
        return reconstruction_loss(state_features, trans_matrices, recon_states, recon_trans, masks)

    loss_and_grad = mx.value_and_grad(loss_fn)

    for epoch in range(n_epochs):
        params = model.parameters()
        loss, grads = loss_and_grad(params)
        losses.append(float(loss))

        # Update parameters
        def update_fn(p, g):
            return p - learning_rate * g
        new_params = mlx.utils.tree_map(update_fn, params, grads)
        model.update(new_params)

        if verbose and epoch % 20 == 0:
            print(f"  Epoch {epoch}: loss={float(loss):.4f}")

    return losses


# =============================================================================
# Compression Interface
# =============================================================================

class LearnedCompressor:
    """
    High-level interface for learned compression.
    """

    def __init__(self, latent_dim: int = 32, max_states: int = 32):
        self.latent_dim = latent_dim
        self.max_states = max_states
        self.model = StatechartAutoencoder(
            max_states=max_states,
            latent_dim=latent_dim,
        )
        self.encoding = StatechartEncoding(max_states=max_states)
        self._trained = False

    def train(
        self,
        statecharts: List[Statechart],
        n_epochs: int = 100,
        verbose: bool = True,
    ):
        """Train compressor on sample statecharts."""
        if verbose:
            print(f"Training on {len(statecharts)} statecharts...")
        train_autoencoder(self.model, statecharts, n_epochs, verbose=verbose)
        self._trained = True

    def compress(self, sc: Statechart) -> mx.array:
        """Compress statechart to latent vector."""
        encoded = self.encoding.encode(sc)
        state_features = mx.expand_dims(encoded["state_features"], 0)
        trans_matrix = mx.expand_dims(encoded["transition_matrix"], 0)
        mask = mx.expand_dims(encoded["mask"], 0)

        latent = self.model.compress(state_features, trans_matrix, mask)
        return latent.squeeze(0)

    def decompress(self, latent: mx.array) -> Statechart:
        """Decompress latent vector to statechart."""
        latent = mx.expand_dims(latent, 0)
        recon_states, recon_trans = self.model.decompress(latent)
        mask = mx.ones((self.max_states,))

        return self.encoding.decode(
            recon_states.squeeze(0),
            recon_trans.squeeze(0),
            mask,
        )

    def get_compression_ratio(self, sc: Statechart) -> float:
        """
        Compute compression ratio.

        Original size: states * state_features + transitions * trans_features
        Compressed size: latent_dim floats
        """
        n_states, n_trans = sc.size()

        # Estimate original size (bytes)
        orig_size = n_states * 7 * 4 + n_trans * 4 * 4  # 4 bytes per float

        # Compressed size
        comp_size = self.latent_dim * 4

        return comp_size / orig_size if orig_size > 0 else 1.0


# =============================================================================
# Demo
# =============================================================================

def create_sample_statecharts(n: int = 20) -> List[Statechart]:
    """Create sample statecharts for training."""
    statecharts = []

    for i in range(n):
        sc = Statechart(name=f"sample_{i}")
        n_states = random.randint(4, 12)

        for j in range(n_states):
            sc.add_state(State(
                f"s{j}",
                is_initial=(j == 0),
                is_final=(j == n_states - 1),
                state_type=random.choice(list(StateType)),
            ))

        # Add random transitions
        n_trans = random.randint(n_states, n_states * 2)
        for _ in range(n_trans):
            src = random.randint(0, n_states - 1)
            tgt = random.randint(0, n_states - 1)
            sc.add_transition(Transition(
                f"s{src}", f"s{tgt}",
                f"e{random.randint(0, 5)}",
                guard=f"x > {random.randint(0, 10)}" if random.random() < 0.3 else None,
                action=f"a{random.randint(0, 3)}" if random.random() < 0.3 else None,
            ))

        statecharts.append(sc)

    return statecharts


def demo():
    """Demonstrate learned compression."""
    print("=" * 60)
    print("Learned Statechart Compression")
    print("=" * 60)

    # Create training data
    print("\nCreating sample statecharts...")
    statecharts = create_sample_statecharts(n=20)

    # Train compressor
    compressor = LearnedCompressor(latent_dim=32)
    compressor.train(statecharts, n_epochs=50, verbose=True)

    # Test compression
    print("\nTesting compression:")
    for sc in statecharts[:3]:
        n_states, n_trans = sc.size()
        latent = compressor.compress(sc)
        ratio = compressor.get_compression_ratio(sc)

        print(f"  {sc.name}: {n_states} states, {n_trans} trans -> "
              f"latent[{len(latent)}], ratio={ratio:.2f}")

    # Decode and check
    print("\nReconstruction test:")
    test_sc = statecharts[0]
    latent = compressor.compress(test_sc)
    recon_sc = compressor.decompress(latent)

    print(f"  Original: {test_sc.size()}")
    print(f"  Reconstructed: {recon_sc.size()}")

    return compressor


if __name__ == "__main__":
    demo()
