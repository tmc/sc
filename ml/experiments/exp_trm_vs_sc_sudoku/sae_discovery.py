"""
SAE Discovery: Find emergent reasoning states in TRM.

Key insight: Instead of hand-coding Sudoku guards, discover what
the model actually learns. SAE features may correspond to:
- Row/Column/Box focus (which constraint is being checked?)
- Solving phase (easy cells → hard cells)
- Confidence level (certain vs uncertain predictions)
- Constraint satisfaction progress

Train TopK SAE on TRM activations across H-cycles, then map
features to semantic concepts.
"""

import sys
sys.path.insert(0, '/Volumes/tmc/go/src/github.com/tmc/sc/ml')

from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple, Optional
from collections import defaultdict
import json
import os

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim


@dataclass
class SAEConfig:
    """Configuration for Sudoku reasoning SAE."""
    input_dim: int = 128          # TRM hidden dimension
    expansion_factor: int = 8     # Overcomplete factor (smaller for interpretability)
    k_active: int = 16            # TopK active features
    l1_coefficient: float = 1e-4  # Sparsity penalty
    learning_rate: float = 1e-3
    batch_size: int = 256
    num_epochs: int = 50
    dead_threshold: float = 1e-6  # Dead feature detection


@dataclass
class SudokuFeature:
    """A discovered Sudoku reasoning feature."""
    feature_id: int
    activation_mean: float = 0.0
    activation_std: float = 0.0
    activation_count: int = 0

    # Semantic associations
    row_affinity: List[float] = field(default_factory=lambda: [0.0] * 9)
    col_affinity: List[float] = field(default_factory=lambda: [0.0] * 9)
    box_affinity: List[float] = field(default_factory=lambda: [0.0] * 9)

    # Iteration dynamics
    h_cycle_activation: List[float] = field(default_factory=lambda: [0.0] * 3)

    # Prediction association
    digit_affinity: List[float] = field(default_factory=lambda: [0.0] * 9)

    # Description
    description: str = ""

    @property
    def is_dead(self) -> bool:
        return self.activation_count == 0

    def primary_row(self) -> int:
        """Which row does this feature focus on?"""
        return max(range(9), key=lambda i: self.row_affinity[i])

    def primary_col(self) -> int:
        """Which column does this feature focus on?"""
        return max(range(9), key=lambda i: self.col_affinity[i])

    def primary_box(self) -> int:
        """Which box does this feature focus on?"""
        return max(range(9), key=lambda i: self.box_affinity[i])


class TopKSAE(nn.Module):
    """
    TopK Sparse Autoencoder for Sudoku reasoning discovery.

    TopK guarantees exactly k features are active, making it
    easier to identify discrete reasoning states.
    """

    def __init__(self, config: SAEConfig):
        super().__init__()
        self.config = config
        self.latent_dim = config.input_dim * config.expansion_factor

        # Encoder: project to overcomplete space
        self.encoder = nn.Linear(config.input_dim, self.latent_dim)

        # Decoder: reconstruct from sparse code
        self.decoder = nn.Linear(self.latent_dim, config.input_dim, bias=False)

        # Track feature activations for dead feature detection
        self.activation_counts = mx.zeros((self.latent_dim,))

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

        # Gather top-k values and apply ReLU
        top_values = mx.take_along_axis(pre_acts, top_indices, axis=-1)
        top_values = mx.maximum(top_values, 0.0)

        # Scatter back to full latent space
        # Create sparse output
        acts = mx.zeros((batch_size, self.latent_dim))

        # Build index arrays for scatter
        batch_idx = mx.repeat(mx.arange(batch_size)[:, None], k, axis=1)

        # Use a loop for MLX compatibility (no scatter_)
        # This is slower but works
        for b in range(batch_size):
            for i in range(k):
                idx = int(top_indices[b, i].item())
                val = float(top_values[b, i].item())
                # Create one-hot and add
                one_hot = mx.zeros((self.latent_dim,))
                # Use masked update
                mask = mx.arange(self.latent_dim) == idx
                one_hot = mx.where(mask, mx.array(val), one_hot)
                acts = acts.at[b].add(one_hot)

        return acts, top_indices

    def encode_fast(self, x: mx.array) -> Tuple[mx.array, mx.array]:
        """
        Fast encoding using masked operations.

        Avoids Python loops for better performance.
        """
        pre_acts = self.encoder(x)  # [batch, latent_dim]
        k = self.config.k_active

        # Get top-k indices
        sorted_indices = mx.argsort(-pre_acts, axis=-1)
        top_indices = sorted_indices[:, :k]  # [batch, k]

        # Create mask: 1 where feature is in top-k, 0 otherwise
        # Efficient: use broadcast comparison
        batch_size = x.shape[0]
        feature_indices = mx.arange(self.latent_dim)[None, None, :]  # [1, 1, latent_dim]
        top_expanded = top_indices[:, :, None]  # [batch, k, 1]

        # For each position, check if it's in top-k
        # Shape: [batch, k, latent_dim]
        matches = (feature_indices == top_expanded).astype(mx.float32)

        # Sum over k dimension to get mask [batch, latent_dim]
        mask = mx.sum(matches, axis=1)

        # Apply mask and ReLU
        acts = mx.maximum(pre_acts, 0.0) * mask

        return acts, top_indices

    def decode(self, acts: mx.array) -> mx.array:
        """Decode sparse representation back to input space."""
        return self.decoder(acts)

    def __call__(self, x: mx.array) -> Tuple[mx.array, mx.array, mx.array]:
        """
        Full forward pass.

        Returns:
            recon: Reconstructed input
            acts: Sparse latent activations
            top_indices: Active feature indices
        """
        acts, top_indices = self.encode_fast(x)
        recon = self.decode(acts)
        return recon, acts, top_indices

    def loss(self, x: mx.array) -> Tuple[mx.array, Dict]:
        """
        Compute SAE loss: reconstruction + L1 sparsity.

        Args:
            x: Input batch [batch, input_dim]

        Returns:
            total_loss: Combined loss
            metrics: Dict with reconstruction and sparsity losses
        """
        recon, acts, _ = self(x)

        # Reconstruction loss (MSE)
        recon_loss = mx.mean((x - recon) ** 2)

        # Sparsity loss (L1 on activations)
        # Already sparse due to TopK, but L1 encourages smaller magnitudes
        l1_loss = self.config.l1_coefficient * mx.mean(mx.abs(acts))

        total_loss = recon_loss + l1_loss

        metrics = {
            'recon_loss': float(recon_loss.item()),
            'l1_loss': float(l1_loss.item()),
            'total_loss': float(total_loss.item()),
            'mean_activation': float(mx.mean(acts).item()),
            'sparsity': float(mx.mean((acts > 0).astype(mx.float32)).item()),
        }

        return total_loss, metrics


def load_activations(path: str) -> mx.array:
    """Load activation data from numpy file."""
    data = mx.load(path)
    if isinstance(data, dict):
        return data[list(data.keys())[0]]
    return data


def train_sae(
    activations: mx.array,
    config: Optional[SAEConfig] = None,
) -> Tuple[TopKSAE, List[Dict]]:
    """
    Train TopK SAE on activation data.

    Args:
        activations: [N, T, 81, hidden_dim] or [N, hidden_dim]
        config: SAE configuration

    Returns:
        Trained SAE model and training history
    """
    config = config or SAEConfig()

    # Reshape activations to [N * T * 81, hidden_dim] if needed
    if len(activations.shape) == 4:
        N, T, C, D = activations.shape
        activations = activations.reshape(-1, D)
        print(f"Reshaped activations: {N} samples × {T} H-cycles × {C} cells = {activations.shape[0]} vectors")
    elif len(activations.shape) == 2:
        D = activations.shape[1]
    else:
        raise ValueError(f"Expected 2D or 4D activations, got shape {activations.shape}")

    # Verify dimensions match config
    if activations.shape[1] != config.input_dim:
        config.input_dim = activations.shape[1]
        print(f"Adjusted input_dim to {config.input_dim}")

    print(f"\nTraining TopK SAE:")
    print(f"  Input dim: {config.input_dim}")
    print(f"  Latent dim: {config.input_dim * config.expansion_factor}")
    print(f"  TopK: {config.k_active}")
    print(f"  Training samples: {activations.shape[0]}")

    # Create model
    sae = TopKSAE(config)
    optimizer = optim.Adam(learning_rate=config.learning_rate)

    # Training loop
    history = []
    n_samples = activations.shape[0]

    def loss_fn(model, batch):
        loss, _ = model.loss(batch)
        return loss

    loss_and_grad = nn.value_and_grad(sae, loss_fn)

    for epoch in range(config.num_epochs):
        # Shuffle
        perm = mx.random.permutation(n_samples)
        activations_shuf = activations[perm]

        epoch_metrics = defaultdict(float)
        n_batches = 0

        for i in range(0, n_samples, config.batch_size):
            batch = activations_shuf[i:i + config.batch_size]

            loss, grads = loss_and_grad(sae, batch)
            optimizer.update(sae, grads)
            mx.eval(sae.parameters())

            # Track metrics
            _, metrics = sae.loss(batch)
            for k, v in metrics.items():
                epoch_metrics[k] += v
            n_batches += 1

        # Average metrics
        for k in epoch_metrics:
            epoch_metrics[k] /= n_batches

        history.append(dict(epoch_metrics))

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}: recon={epoch_metrics['recon_loss']:.4f}, "
                  f"sparsity={epoch_metrics['sparsity']:.3f}")

    return sae, history


def analyze_features(
    sae: TopKSAE,
    activations: mx.array,
    metadata: Dict,
    n_samples: int = 1000,
) -> List[SudokuFeature]:
    """
    Analyze SAE features for Sudoku-specific semantic meanings.

    Maps features to:
    - Row/Column/Box focus
    - H-cycle dynamics (when does feature activate?)
    - Position patterns
    """
    config = sae.config
    latent_dim = config.input_dim * config.expansion_factor

    # Initialize features
    features = [SudokuFeature(feature_id=i) for i in range(latent_dim)]

    # Get activation shape info
    if len(activations.shape) == 4:
        N, T, C, D = activations.shape
    else:
        # Reshape from flat
        N = metadata.get('num_samples', activations.shape[0] // (3 * 81))
        T = metadata.get('H_cycles', 3)
        C = 81
        D = metadata.get('hidden_dim', activations.shape[-1])
        activations = activations.reshape(N, T, C, D)

    print(f"\nAnalyzing {latent_dim} features on {min(n_samples, N)} samples...")

    # Sample subset for analysis
    sample_idx = mx.random.permutation(N)[:n_samples]
    sample_acts = activations[sample_idx]  # [n, T, 81, D]

    # For each sample, get SAE activations
    for si in range(min(n_samples, N)):
        for ti in range(T):
            for ci in range(C):
                h = sample_acts[si, ti, ci:ci+1]  # [1, D]

                # Get SAE encoding
                _, acts, top_indices = sae(h)
                mx.eval(acts, top_indices)

                # Update feature statistics
                for ki in range(config.k_active):
                    fid = int(top_indices[0, ki].item())
                    val = float(acts[0, fid].item())

                    features[fid].activation_count += 1
                    features[fid].activation_mean += val

                    # Track row/col/box affinity
                    row = ci // 9
                    col = ci % 9
                    box = (row // 3) * 3 + (col // 3)

                    features[fid].row_affinity[row] += val
                    features[fid].col_affinity[col] += val
                    features[fid].box_affinity[box] += val

                    # Track H-cycle dynamics
                    features[fid].h_cycle_activation[ti] += val

        if (si + 1) % 100 == 0:
            print(f"  Analyzed {si + 1}/{min(n_samples, N)} samples")

    # Normalize feature statistics
    for f in features:
        if f.activation_count > 0:
            f.activation_mean /= f.activation_count

            total_row = sum(f.row_affinity)
            if total_row > 0:
                f.row_affinity = [x / total_row for x in f.row_affinity]

            total_col = sum(f.col_affinity)
            if total_col > 0:
                f.col_affinity = [x / total_col for x in f.col_affinity]

            total_box = sum(f.box_affinity)
            if total_box > 0:
                f.box_affinity = [x / total_box for x in f.box_affinity]

            total_h = sum(f.h_cycle_activation)
            if total_h > 0:
                f.h_cycle_activation = [x / total_h for x in f.h_cycle_activation]

    return features


def interpret_features(features: List[SudokuFeature]) -> Dict:
    """
    Interpret discovered features semantically.

    Categorize features into:
    - Row-focused (attends strongly to specific row)
    - Column-focused
    - Box-focused
    - Iteration-dependent (early vs late H-cycles)
    - General (no clear focus)
    """
    interpretation = {
        'row_focused': [],
        'col_focused': [],
        'box_focused': [],
        'early_cycle': [],  # Active mainly in H-cycle 1
        'late_cycle': [],   # Active mainly in H-cycle 3
        'general': [],
        'dead': [],
    }

    alive_features = [f for f in features if not f.is_dead]
    print(f"\nInterpreting {len(alive_features)} active features "
          f"({len(features) - len(alive_features)} dead)")

    for f in alive_features:
        # Check row focus (entropy)
        row_entropy = -sum(p * (log2(p) if p > 0 else 0) for p in f.row_affinity)
        col_entropy = -sum(p * (log2(p) if p > 0 else 0) for p in f.col_affinity)
        box_entropy = -sum(p * (log2(p) if p > 0 else 0) for p in f.box_affinity)

        # Max entropy for uniform = log2(9) ≈ 3.17
        max_entropy = 3.17
        threshold = 2.5  # Below this = focused

        if row_entropy < threshold:
            f.description = f"Row-focused (row {f.primary_row()})"
            interpretation['row_focused'].append(f.feature_id)
        elif col_entropy < threshold:
            f.description = f"Column-focused (col {f.primary_col()})"
            interpretation['col_focused'].append(f.feature_id)
        elif box_entropy < threshold:
            f.description = f"Box-focused (box {f.primary_box()})"
            interpretation['box_focused'].append(f.feature_id)

        # Check H-cycle focus
        if f.h_cycle_activation[0] > 0.5:
            f.description += " [early-cycle]"
            interpretation['early_cycle'].append(f.feature_id)
        elif f.h_cycle_activation[2] > 0.5:
            f.description += " [late-cycle]"
            interpretation['late_cycle'].append(f.feature_id)

        if not f.description:
            f.description = "General"
            interpretation['general'].append(f.feature_id)

    for f in features:
        if f.is_dead:
            interpretation['dead'].append(f.feature_id)

    return interpretation


def log2(x):
    """Log base 2."""
    import math
    return math.log(x) / math.log(2) if x > 0 else 0


def generate_statechart_from_features(
    features: List[SudokuFeature],
    interpretation: Dict,
) -> Dict:
    """
    Generate a statechart structure from discovered features.

    Maps feature categories to statechart states.
    """
    statechart = {
        'root_state': {
            'label': 'SudokuReasoning',
            'type': 'parallel',  # Orthogonal regions
            'children': []
        },
        'transitions': [],
    }

    # Region 1: Constraint Focus (which unit is being checked?)
    constraint_region = {
        'label': 'ConstraintFocus',
        'type': 'normal',
        'children': [
            {'label': 'RowFocus', 'type': 'basic'},
            {'label': 'ColFocus', 'type': 'basic'},
            {'label': 'BoxFocus', 'type': 'basic'},
            {'label': 'NeutralFocus', 'type': 'basic', 'is_initial': True},
        ]
    }
    statechart['root_state']['children'].append(constraint_region)

    # Region 2: Solving Phase (iteration progress)
    phase_region = {
        'label': 'SolvingPhase',
        'type': 'normal',
        'children': [
            {'label': 'EarlyCycle', 'type': 'basic', 'is_initial': True},
            {'label': 'MidCycle', 'type': 'basic'},
            {'label': 'LateCycle', 'type': 'basic'},
        ]
    }
    statechart['root_state']['children'].append(phase_region)

    # Add transitions based on feature dynamics
    statechart['transitions'] = [
        {'from': ['EarlyCycle'], 'to': ['MidCycle'], 'event': 'H_CYCLE'},
        {'from': ['MidCycle'], 'to': ['LateCycle'], 'event': 'H_CYCLE'},
        # Constraint focus transitions triggered by attention patterns
        {'from': ['NeutralFocus'], 'to': ['RowFocus'], 'event': 'ROW_CONSTRAINT'},
        {'from': ['NeutralFocus'], 'to': ['ColFocus'], 'event': 'COL_CONSTRAINT'},
        {'from': ['NeutralFocus'], 'to': ['BoxFocus'], 'event': 'BOX_CONSTRAINT'},
    ]

    # Feature mapping
    statechart['feature_mapping'] = {
        'RowFocus': interpretation.get('row_focused', []),
        'ColFocus': interpretation.get('col_focused', []),
        'BoxFocus': interpretation.get('box_focused', []),
        'EarlyCycle': interpretation.get('early_cycle', []),
        'LateCycle': interpretation.get('late_cycle', []),
    }

    return statechart


def run_sae_discovery():
    """Main function to run SAE discovery pipeline."""
    print("=" * 70)
    print("SAE DISCOVERY: FINDING REASONING STATES IN TRM")
    print("=" * 70)

    # Load activations
    data_dir = 'experiments/exp_trm_vs_sc_sudoku/activations'
    hidden_path = os.path.join(data_dir, 'hidden_states.npy')
    meta_path = os.path.join(data_dir, 'metadata.json')

    if not os.path.exists(hidden_path):
        print(f"Activation data not found at {hidden_path}")
        print("Run: python activation_extractor.py --mode=extract first")
        return

    print(f"\nLoading activations from {hidden_path}...")
    activations = load_activations(hidden_path)
    print(f"Loaded shape: {activations.shape}")

    with open(meta_path) as f:
        metadata = json.load(f)
    print(f"Metadata: {metadata}")

    # Configure SAE
    config = SAEConfig(
        input_dim=metadata['hidden_dim'],
        expansion_factor=8,
        k_active=16,
        l1_coefficient=1e-4,
        num_epochs=30,
        batch_size=256,
    )

    # Train SAE
    print("\n" + "-" * 70)
    sae, history = train_sae(activations, config)
    print("-" * 70)

    # Analyze features
    print("\n" + "-" * 70)
    features = analyze_features(sae, activations, metadata, n_samples=100)
    print("-" * 70)

    # Interpret features
    print("\n" + "-" * 70)
    interpretation = interpret_features(features)

    print("\nFeature Categories:")
    for category, fids in interpretation.items():
        print(f"  {category}: {len(fids)} features")

    # Print top features per category
    for category in ['row_focused', 'col_focused', 'box_focused']:
        if interpretation[category]:
            print(f"\n{category.upper()} features:")
            for fid in interpretation[category][:5]:
                f = features[fid]
                print(f"  Feature {fid}: {f.description}, "
                      f"count={f.activation_count}, mean={f.activation_mean:.3f}")

    # Generate statechart
    print("\n" + "-" * 70)
    print("GENERATING STATECHART FROM FEATURES")
    print("-" * 70)
    statechart = generate_statechart_from_features(features, interpretation)

    # Save results
    output_dir = 'experiments/exp_trm_vs_sc_sudoku/sae_results'
    os.makedirs(output_dir, exist_ok=True)

    # Save statechart
    sc_path = os.path.join(output_dir, 'discovered_statechart.json')
    with open(sc_path, 'w') as f:
        json.dump(statechart, f, indent=2)
    print(f"\nSaved statechart to {sc_path}")

    # Save interpretation
    interp_path = os.path.join(output_dir, 'feature_interpretation.json')
    with open(interp_path, 'w') as f:
        json.dump(interpretation, f, indent=2)
    print(f"Saved interpretation to {interp_path}")

    # Save training history
    hist_path = os.path.join(output_dir, 'training_history.json')
    with open(hist_path, 'w') as f:
        json.dump(history, f, indent=2)

    print("\n" + "=" * 70)
    print("SAE DISCOVERY COMPLETE")
    print("=" * 70)
    print(f"\nKey findings:")
    print(f"  Active features: {len([f for f in features if not f.is_dead])}/{len(features)}")
    print(f"  Row-focused: {len(interpretation['row_focused'])}")
    print(f"  Col-focused: {len(interpretation['col_focused'])}")
    print(f"  Box-focused: {len(interpretation['box_focused'])}")
    print(f"  Early-cycle: {len(interpretation['early_cycle'])}")
    print(f"  Late-cycle: {len(interpretation['late_cycle'])}")


if __name__ == "__main__":
    run_sae_discovery()
