"""
Activation Extractor for SC-TRM.

Extracts hidden states at each (H, L) iteration for SAE analysis.
This allows us to discover what the model learns internally.

Key insight: TRM iteratively refines its predictions through H×L cycles.
By extracting activations at each step, we can:
1. Train SAE to find sparse features
2. Map features to semantic concepts (row/col/box focus, confidence)
3. Discover emergent reasoning structure
"""

import sys
sys.path.insert(0, '/Volumes/tmc/go/src/github.com/tmc/sc/ml')

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import json
import os

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from experiments.exp_trm_vs_sc_sudoku.vanilla_trm import VanillaTRM, VanillaTRMConfig


@dataclass
class ActivationConfig:
    """Configuration for activation extraction."""
    # Base TRM config
    hidden_dim: int = 128
    num_heads: int = 4
    num_layers: int = 3
    ff_dim: int = 256
    num_cells: int = 81
    num_digits: int = 9
    H_cycles: int = 3
    L_cycles: int = 4
    dropout: float = 0.1
    use_halting: bool = True

    # Extraction options
    extract_per_H: bool = True  # Extract after each H-cycle
    extract_per_L: bool = False  # Extract after each L-cycle (more data, slower)
    extract_attention: bool = True  # Also extract attention patterns
    max_samples: int = 10000  # Maximum samples to extract


class ActivationExtractor:
    """
    Extracts activations from TRM at each iteration step.

    Provides hooks into the model to capture hidden states during solve().
    """

    def __init__(self, model: VanillaTRM, config: Optional[ActivationConfig] = None):
        self.model = model
        self.config = config or ActivationConfig()

        # Storage for extracted activations
        self.activations = []
        self.attention_patterns = []

    def extract_single_puzzle(
        self,
        puzzle: mx.array,
        H_cycles: Optional[int] = None,
        L_cycles: Optional[int] = None,
    ) -> Dict[str, mx.array]:
        """
        Extract activations for a single puzzle through all iterations.

        Args:
            puzzle: [1, 81] single puzzle

        Returns:
            Dict containing:
            - hidden_states: [H, 81, hidden_dim] (if extract_per_H)
            - attention: [H, num_heads, 81, 81] (if extract_attention)
            - final_logits: [81, 9]
            - final_predictions: [81]
        """
        H = H_cycles or self.model.config.H_cycles
        L = L_cycles or self.model.config.L_cycles

        hidden_states = []
        attention_weights = []

        # Initial encoding
        h = self.model.encode(puzzle)  # [1, 81, hidden_dim]

        for hi in range(H):
            # Compute H-level context
            pooled = mx.mean(h, axis=1)  # [1, hidden_dim]
            h_context = self.model.h_context_net(pooled)  # [1, hidden_dim]

            for li in range(L):
                # Store pre-refinement state
                if self.config.extract_per_L:
                    hidden_states.append(h.squeeze(0))  # [81, hidden_dim]

                # Refine
                h = self.model.refine(h, h_context)

            # Store post-H-cycle state
            if self.config.extract_per_H:
                hidden_states.append(h.squeeze(0))  # [81, hidden_dim]

        # Final prediction
        logits = self.model.predict(h)  # [1, 81, 9]
        predictions = mx.argmax(logits, axis=-1) + 1  # [1, 81]

        result = {
            'hidden_states': mx.stack(hidden_states, axis=0) if hidden_states else None,  # [T, 81, hidden_dim]
            'final_logits': logits.squeeze(0),  # [81, 9]
            'final_predictions': predictions.squeeze(0),  # [81]
        }

        return result

    def extract_batch(
        self,
        puzzles: mx.array,
        solutions: Optional[mx.array] = None,
        batch_size: int = 32,
    ) -> Dict[str, List]:
        """
        Extract activations for a batch of puzzles.

        Args:
            puzzles: [N, 81] puzzles
            solutions: [N, 81] optional solutions for accuracy tracking
            batch_size: Process puzzles in batches

        Returns:
            Dict with lists of extracted data
        """
        N = puzzles.shape[0]
        all_hidden = []
        all_logits = []
        all_predictions = []
        all_correct = []

        for i in range(0, min(N, self.config.max_samples), batch_size):
            batch_puzzles = puzzles[i:i+batch_size]

            # Process each puzzle individually for now
            # (could be optimized to batch)
            for j in range(batch_puzzles.shape[0]):
                puzzle = batch_puzzles[j:j+1]
                result = self.extract_single_puzzle(puzzle)

                if result['hidden_states'] is not None:
                    all_hidden.append(result['hidden_states'])
                all_logits.append(result['final_logits'])
                all_predictions.append(result['final_predictions'])

                if solutions is not None:
                    correct = (result['final_predictions'] == solutions[i+j]).astype(mx.float32)
                    all_correct.append(correct)

            if (i + batch_size) % 100 == 0:
                print(f"  Extracted {min(i + batch_size, N)}/{min(N, self.config.max_samples)}")

        return {
            'hidden_states': all_hidden,  # List of [T, 81, hidden_dim]
            'logits': all_logits,  # List of [81, 9]
            'predictions': all_predictions,  # List of [81]
            'correct': all_correct if solutions is not None else None,  # List of [81]
        }

    def save_activations(self, data: Dict, output_dir: str):
        """Save extracted activations to disk."""
        os.makedirs(output_dir, exist_ok=True)

        # Save hidden states (large, save as separate MLX arrays)
        if data.get('hidden_states'):
            hidden_path = os.path.join(output_dir, 'hidden_states.npy')
            # Stack all hidden states
            stacked_hidden = mx.stack([mx.array(h) for h in data['hidden_states']], axis=0)
            mx.eval(stacked_hidden)
            mx.save(hidden_path, stacked_hidden)
            print(f"  Saved hidden states: {stacked_hidden.shape}")

        # Save logits and predictions
        if data.get('logits'):
            logits_path = os.path.join(output_dir, 'logits.npy')
            stacked_logits = mx.stack([mx.array(l) for l in data['logits']], axis=0)
            mx.eval(stacked_logits)
            mx.save(logits_path, stacked_logits)
            print(f"  Saved logits: {stacked_logits.shape}")

        if data.get('predictions'):
            preds_path = os.path.join(output_dir, 'predictions.npy')
            stacked_preds = mx.stack([mx.array(p) for p in data['predictions']], axis=0)
            mx.eval(stacked_preds)
            mx.save(preds_path, stacked_preds)
            print(f"  Saved predictions: {stacked_preds.shape}")

        # Save metadata
        metadata = {
            'num_samples': len(data.get('hidden_states', [])) or len(data.get('logits', [])),
            'hidden_dim': self.model.config.hidden_dim,
            'H_cycles': self.model.config.H_cycles,
            'L_cycles': self.model.config.L_cycles,
            'extract_per_H': self.config.extract_per_H,
            'extract_per_L': self.config.extract_per_L,
        }
        meta_path = os.path.join(output_dir, 'metadata.json')
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"  Saved metadata to {meta_path}")


def train_model_for_extraction(train_p, train_s, config: ActivationConfig, epochs: int = 30):
    """Train a TRM model and return it for activation extraction."""
    from experiments.exp_trm_vs_sc_sudoku.vanilla_trm import VanillaTRM, VanillaTRMConfig

    print("Training TRM model for activation extraction...")

    trm_config = VanillaTRMConfig(
        hidden_dim=config.hidden_dim,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        ff_dim=config.ff_dim,
        num_cells=config.num_cells,
        num_digits=config.num_digits,
        H_cycles=config.H_cycles,
        L_cycles=config.L_cycles,
        dropout=config.dropout,
        use_halting=config.use_halting,
    )

    model = VanillaTRM(trm_config)
    optimizer = optim.AdamW(learning_rate=1e-4, weight_decay=0.01)

    def loss_fn(m, p, s):
        loss, _ = m.loss(p, s)
        return loss

    loss_and_grad = nn.value_and_grad(model, loss_fn)

    for epoch in range(epochs):
        perm = mx.random.permutation(train_p.shape[0])
        train_p_shuf = train_p[perm]
        train_s_shuf = train_s[perm]

        total_loss = 0.0
        n_batches = 0

        for i in range(0, train_p.shape[0], 32):
            p = train_p_shuf[i:i+32]
            s = train_s_shuf[i:i+32]
            loss, grads = loss_and_grad(model, p, s)
            optimizer.update(model, grads)
            mx.eval(model.parameters())
            total_loss += float(loss.item())
            n_batches += 1

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}: loss={total_loss/n_batches:.4f}")

    return model


def extract_and_analyze():
    """Main function to extract activations and perform basic analysis."""
    from experiments.exp_trm_vs_sc_sudoku.sudoku_data import generate_random_sudoku

    print("=" * 70)
    print("ACTIVATION EXTRACTION FOR SAE ANALYSIS")
    print("=" * 70)

    # Configuration
    config = ActivationConfig(
        hidden_dim=128,
        num_heads=4,
        num_layers=3,
        ff_dim=256,
        H_cycles=3,
        L_cycles=4,
        extract_per_H=True,
        extract_per_L=False,
        max_samples=1000,
    )

    # Generate data
    print("\nGenerating data...")
    train_p, train_s, test_p, test_s = generate_random_sudoku(1000, 200, seed=42)

    # Train model
    print("\nTraining model...")
    model = train_model_for_extraction(train_p, train_s, config, epochs=30)

    # Create extractor
    extractor = ActivationExtractor(model, config)

    # Extract activations from test set
    print("\nExtracting activations...")
    data = extractor.extract_batch(test_p, test_s, batch_size=32)

    # Basic analysis
    print("\n" + "=" * 70)
    print("ACTIVATION ANALYSIS")
    print("=" * 70)

    if data['hidden_states']:
        # Stack all hidden states
        all_hidden = mx.stack([mx.array(h) for h in data['hidden_states']], axis=0)
        mx.eval(all_hidden)

        print(f"\nHidden states shape: {all_hidden.shape}")
        # Shape: [N, T, 81, hidden_dim] where T = H_cycles

        # Compute statistics
        mean_activation = float(mx.mean(all_hidden).item())
        std_activation = float(mx.std(all_hidden).item())
        print(f"Mean activation: {mean_activation:.4f}")
        print(f"Std activation: {std_activation:.4f}")

        # Analyze evolution across iterations
        print("\nActivation evolution across H-cycles:")
        for t in range(all_hidden.shape[1]):
            t_mean = float(mx.mean(all_hidden[:, t, :, :]).item())
            t_std = float(mx.std(all_hidden[:, t, :, :]).item())
            print(f"  H-cycle {t+1}: mean={t_mean:.4f}, std={t_std:.4f}")

        # Analyze per-cell variance (which cells change most?)
        cell_variance = mx.var(all_hidden[:, -1, :, :], axis=(0, 2))  # [81]
        mx.eval(cell_variance)
        top_variance_cells = mx.argsort(cell_variance)[-10:][::-1].tolist()
        print(f"\nTop 10 highest-variance cells: {top_variance_cells}")

    # Accuracy analysis
    if data['correct']:
        all_correct = mx.stack([mx.array(c) for c in data['correct']], axis=0)
        mx.eval(all_correct)
        accuracy = float(mx.mean(all_correct).item())
        print(f"\nOverall cell accuracy: {accuracy:.1%}")

        # Per-position accuracy
        pos_accuracy = mx.mean(all_correct, axis=0)  # [81]
        mx.eval(pos_accuracy)
        hardest_cells = mx.argsort(pos_accuracy)[:10].tolist()
        easiest_cells = mx.argsort(pos_accuracy)[-10:][::-1].tolist()
        print(f"Hardest cells: {hardest_cells}")
        print(f"Easiest cells: {easiest_cells}")

    # Save activations
    output_dir = 'experiments/exp_trm_vs_sc_sudoku/activations'
    print(f"\nSaving activations to {output_dir}...")
    extractor.save_activations(data, output_dir)

    print("\n" + "=" * 70)
    print("Extraction complete!")
    print("=" * 70)


def prepare_for_sae():
    """Prepare activation data in format suitable for SAE training."""
    from experiments.exp_trm_vs_sc_sudoku.sudoku_data import generate_random_sudoku

    print("=" * 70)
    print("PREPARING DATA FOR SAE TRAINING")
    print("=" * 70)

    # Configuration
    config = ActivationConfig(
        hidden_dim=128,
        num_heads=4,
        num_layers=3,
        ff_dim=256,
        H_cycles=3,
        L_cycles=4,
        extract_per_H=True,
        max_samples=5000,  # More samples for SAE
    )

    # Generate more data
    print("\nGenerating data...")
    train_p, train_s, test_p, test_s = generate_random_sudoku(5000, 1000, seed=42)

    # Train model
    print("\nTraining model...")
    model = train_model_for_extraction(train_p, train_s, config, epochs=30)

    # Extract from training set (need lots of data for SAE)
    print("\nExtracting activations from training set...")
    extractor = ActivationExtractor(model, config)
    data = extractor.extract_batch(train_p[:config.max_samples], train_s[:config.max_samples], batch_size=32)

    # Flatten for SAE: [N * T * 81, hidden_dim]
    if data['hidden_states']:
        all_hidden = mx.stack([mx.array(h) for h in data['hidden_states']], axis=0)
        mx.eval(all_hidden)
        print(f"\nOriginal shape: {all_hidden.shape}")

        # Reshape: [N, T, 81, D] -> [N * T * 81, D]
        N, T, C, D = all_hidden.shape
        flattened = all_hidden.reshape(-1, D)
        print(f"Flattened shape: {flattened.shape} (ready for SAE)")

        # Save
        output_dir = 'experiments/exp_trm_vs_sc_sudoku/sae_data'
        os.makedirs(output_dir, exist_ok=True)

        mx.save(os.path.join(output_dir, 'activations_flat.npy'), flattened)
        print(f"\nSaved flattened activations to {output_dir}/activations_flat.npy")

        # Also save metadata for reconstruction
        metadata = {
            'original_shape': [N, T, C, D],
            'hidden_dim': D,
            'H_cycles': T,
            'num_cells': C,
            'num_samples': N,
        }
        with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)

    print("\n" + "=" * 70)
    print("SAE data preparation complete!")
    print("=" * 70)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['extract', 'prepare_sae', 'all'], default='extract')
    args = parser.parse_args()

    if args.mode in ['extract', 'all']:
        extract_and_analyze()

    if args.mode in ['prepare_sae', 'all']:
        prepare_for_sae()
