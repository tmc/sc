"""
Unified Benchmark Runner for SC-TRM Approaches.

Runs systematic experiments comparing 5 SC-TRM approaches across
multiple epoch counts with per-epoch metrics tracking.

Usage:
    python -m experiments.exp_trm_vs_sc_sudoku.benchmark_runner --epochs 30,50,100 --seeds 3
"""

import sys
sys.path.insert(0, '/Volumes/tmc/go/src/github.com/tmc/sc/ml')

import argparse
import json
import os
import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Callable

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim


@dataclass
class EpochMetrics:
    """Metrics for a single epoch."""
    epoch: int
    loss: float
    cell_accuracy: float
    exact_accuracy: float
    violation_rate: float


@dataclass
class ExperimentResult:
    """Complete result from one experiment run."""
    model_name: str
    epochs: int
    seed: int

    # Per-epoch history
    train_history: List[Dict]

    # Final metrics
    final_cell_accuracy: float
    final_exact_accuracy: float
    final_violation_rate: float

    # Speed metrics
    epochs_to_50_cell: Optional[int]
    epochs_to_70_cell: Optional[int]
    training_time_seconds: float


def count_violations(predictions: mx.array) -> float:
    """Count constraint violations per puzzle."""
    B = predictions.shape[0]
    board = predictions.reshape(B, 9, 9)
    mx.eval(board)

    total_violations = 0
    for b in range(B):
        # Row violations
        for row in range(9):
            row_vals = board[b, row, :].tolist()
            unique = len(set(row_vals))
            total_violations += 9 - unique

        # Column violations
        for col in range(9):
            col_vals = board[b, :, col].tolist()
            unique = len(set(col_vals))
            total_violations += 9 - unique

        # Box violations
        for box_r in range(3):
            for box_c in range(3):
                box_vals = board[b, box_r*3:(box_r+1)*3, box_c*3:(box_c+1)*3].reshape(-1).tolist()
                unique = len(set(box_vals))
                total_violations += 9 - unique

    return total_violations / B


def evaluate_model(model, test_p: mx.array, test_s: mx.array, **solve_kwargs) -> Dict:
    """Evaluate a model and return metrics.

    NOTE: Only counts empty cells (where puzzle == 0) to match TRM's accuracy.
    TRM uses IGNORE_LABEL_ID for given cells.
    """
    result = model.solve(test_p, **solve_kwargs)
    preds = result['predictions']
    mx.eval(preds)

    # CRITICAL: Only count empty cells (where puzzle == 0)
    # This matches TRM's accuracy calculation (IGNORE_LABEL_ID for given cells)
    empty_mask = (test_p == 0)  # [B, 81] - True for cells to predict
    correct = (preds == test_s) & empty_mask

    # Cell accuracy: correct empty cells / total empty cells
    total_empty = mx.sum(empty_mask).item()
    correct_empty = mx.sum(correct.astype(mx.float32)).item()
    cell_acc = float(correct_empty / max(total_empty, 1))

    # Exact accuracy: puzzle correct if ALL empty cells are correct
    empty_per_puzzle = mx.sum(empty_mask.astype(mx.float32), axis=1)  # [B]
    correct_per_puzzle = mx.sum(correct.astype(mx.float32), axis=1)  # [B]
    puzzle_correct = (correct_per_puzzle == empty_per_puzzle)
    exact_acc = float(mx.mean(puzzle_correct.astype(mx.float32)).item())

    violations = count_violations(preds)

    return {
        'cell_accuracy': cell_acc,
        'exact_accuracy': exact_acc,
        'violation_rate': violations,
    }


def train_model_with_history(
    model_type: str,
    train_p: mx.array,
    train_s: mx.array,
    test_p: mx.array,
    test_s: mx.array,
    epochs: int,
    seed: int,
    output_path: Optional[str] = None,  # Save incremental results here
    eval_every: int = 1,
) -> ExperimentResult:
    """Train a model and return full experiment result with history."""

    start_time = time.time()
    mx.random.seed(seed)

    # Import and create model based on type
    if model_type == 'vanilla':
        from experiments.exp_trm_vs_sc_sudoku.vanilla_trm import VanillaTRM, VanillaTRMConfig
        config = VanillaTRMConfig(
            hidden_dim=128, num_heads=4, num_layers=3, ff_dim=256,
            H_cycles=3, L_cycles=4,
        )
        model = VanillaTRM(config)
        solve_kwargs = {}
        loss_kwargs = {}

    elif model_type == 'ste':
        from experiments.exp_trm_vs_sc_sudoku.ste_guards import STETRM, STEConfig
        config = STEConfig(
            hidden_dim=128, num_heads=4, num_layers=3, ff_dim=256,
            H_cycles=3, L_cycles=4, soft_epsilon=0.1,
        )
        model = STETRM(config)
        solve_kwargs = {'use_guards': True}
        loss_kwargs = {'use_guards': True}

    elif model_type == 'confidence':
        from experiments.exp_trm_vs_sc_sudoku.confidence_guards import ConfidenceTRM, ConfidenceGuardConfig
        config = ConfidenceGuardConfig(
            hidden_dim=128, num_heads=4, num_layers=3, ff_dim=256,
            H_cycles=3, L_cycles=4,
        )
        model = ConfidenceTRM(config)
        solve_kwargs = {'use_guards': True}
        loss_kwargs = {'use_guards': True}

    elif model_type == 'attention_bias':
        from experiments.exp_trm_vs_sc_sudoku.attention_bias import AttentionBiasTRM, AttentionBiasConfig
        config = AttentionBiasConfig(
            hidden_dim=128, num_heads=4, num_layers=3, ff_dim=256,
            H_cycles=3, L_cycles=4,
            initial_bias_scale=0.5, final_bias_scale=3.0,
        )
        model = AttentionBiasTRM(config)
        solve_kwargs = {'bias_anneal': True}
        loss_kwargs = {'bias_anneal': True}

    elif model_type == 'hierarchical':
        from experiments.exp_trm_vs_sc_sudoku.hierarchical_sc_trm import HierarchicalSCTRM, HierarchicalConfig
        config = HierarchicalConfig(
            hidden_dim=128, num_heads=4, num_layers=3, ff_dim=256,
            H_cycles=3, L_cycles=4,
            unit_hidden_dim=64, board_hidden_dim=32,
        )
        model = HierarchicalSCTRM(config)
        solve_kwargs = {}
        loss_kwargs = {}

    elif model_type == 'faithful':
        from experiments.exp_trm_vs_sc_sudoku.faithful_trm import FaithfulTRM, FaithfulTRMConfig
        config = FaithfulTRMConfig(
            d_model=128, d_hidden=256, num_heads=4, num_layers=3,
            d_z_H=128, d_z_L=128,
            H_cycles=3, L_cycles=4,
            truncate_grads=True,
        )
        model = FaithfulTRM(config)
        solve_kwargs = {}
        loss_kwargs = {}

    elif model_type == 'learned':
        from experiments.exp_trm_vs_sc_sudoku.learned_sc_trm import LearnedSCTRM, LearnedSCConfig
        config = LearnedSCConfig(
            hidden_dim=128, num_heads=4, num_layers=3, ff_dim=256,
            H_cycles=3, L_cycles=4,
            num_constraint_heads=4, num_state_features=16,
        )
        model = LearnedSCTRM(config)
        solve_kwargs = {}
        loss_kwargs = {}

    elif model_type == 'nas':
        from experiments.exp_trm_vs_sc_sudoku.nas_sc_trm import NASSCTRM, NASSCConfig
        config = NASSCConfig(
            hidden_dim=128, num_heads=4, num_layers=3, ff_dim=256,
            H_cycles=3, L_cycles=4,
            num_constraint_ops=8, num_transition_ops=4,
        )
        model = NASSCTRM(config)
        solve_kwargs = {}
        loss_kwargs = {}

    # === NEW MODELS ===

    elif model_type == 'iterative':
        from experiments.exp_trm_vs_sc_sudoku.iterative_trm import IterativeTRM, IterativeConfig
        config = IterativeConfig(
            hidden_size=128, num_layers=2, T=3, n=6,
            use_attention=True, use_halting=False,
        )
        model = IterativeTRM(config)
        solve_kwargs = {}
        loss_kwargs = {}

    elif model_type == 'iterative_mlp':
        from experiments.exp_trm_vs_sc_sudoku.iterative_trm import IterativeTRM_MLP, IterativeConfig
        config = IterativeConfig(
            hidden_size=128, num_layers=2, T=3, n=6,
            use_attention=False, use_halting=False,
        )
        model = IterativeTRM_MLP(config)
        solve_kwargs = {}
        loss_kwargs = {}

    elif model_type == 'faithful_v2':
        from experiments.exp_trm_vs_sc_sudoku.faithful_trm_v2 import FaithfulTRMv2, FaithfulTRMv2Config
        config = FaithfulTRMv2Config(
            hidden_size=128, H_cycles=3, L_cycles=6, L_layers=2,
            use_mlp_t=True,  # MLP across sequence (better for Sudoku)
        )
        model = FaithfulTRMv2(config)
        solve_kwargs = {}
        loss_kwargs = {}

    elif model_type == 'faithful_v3':
        from experiments.exp_trm_vs_sc_sudoku.faithful_trm_v3 import FaithfulTRMv3, FaithfulTRMv3Config
        config = FaithfulTRMv3Config(
            hidden_size=128, H_cycles=3, L_cycles=6, L_layers=2,
            use_mlp_t=True,
        )
        model = FaithfulTRMv3(config)
        solve_kwargs = {}
        loss_kwargs = {'truncate_grad': True}  # Key fix: gradient truncation

    elif model_type == 'faithful_v3_attn':
        from experiments.exp_trm_vs_sc_sudoku.faithful_trm_v3 import FaithfulTRMv3, FaithfulTRMv3Config
        config = FaithfulTRMv3Config(
            hidden_size=128, H_cycles=3, L_cycles=6, L_layers=2,
            use_mlp_t=False,  # Attention version
        )
        model = FaithfulTRMv3(config)
        solve_kwargs = {}
        loss_kwargs = {'truncate_grad': True}

    elif model_type == 'faithful_v4':
        # v4: StableMax + proper truncated normal + puzzle emb support
        from experiments.exp_trm_vs_sc_sudoku.faithful_trm_v4 import FaithfulTRMv4, FaithfulTRMv4Config
        config = FaithfulTRMv4Config(
            hidden_size=128, H_cycles=3, L_cycles=6, L_layers=2,
            use_mlp_t=True,
            use_stablemax=True,  # Key: StableMax for numerical stability
        )
        model = FaithfulTRMv4(config)
        solve_kwargs = {}
        loss_kwargs = {'truncate_grad': True}

    elif model_type == 'faithful_v4_attn':
        from experiments.exp_trm_vs_sc_sudoku.faithful_trm_v4 import FaithfulTRMv4, FaithfulTRMv4Config
        config = FaithfulTRMv4Config(
            hidden_size=128, H_cycles=3, L_cycles=6, L_layers=2,
            use_mlp_t=False,  # Attention version
            use_stablemax=True,
        )
        model = FaithfulTRMv4(config)
        solve_kwargs = {}
        loss_kwargs = {'truncate_grad': True}

    elif model_type == 'faithful_v4_rope':
        # v4 with RoPE instead of learned position embeddings
        from experiments.exp_trm_vs_sc_sudoku.faithful_trm_v4 import FaithfulTRMv4, FaithfulTRMv4Config
        config = FaithfulTRMv4Config(
            hidden_size=128, H_cycles=3, L_cycles=6, L_layers=2,
            use_mlp_t=False,  # Need attention for RoPE
            use_rope=True,
            use_stablemax=True,
        )
        model = FaithfulTRMv4(config)
        solve_kwargs = {}
        loss_kwargs = {'truncate_grad': True}

    elif model_type == 'faithful_act':
        # ACT: Adaptive Computation Time with Q-learning halting
        from experiments.exp_trm_vs_sc_sudoku.faithful_trm_act import FaithfulTRMACT, ACTConfig
        config = ACTConfig(
            hidden_size=128, max_H_cycles=5, L_cycles=6, L_layers=2,
            use_mlp_t=True,
            halt_exploration_prob=0.1,
        )
        model = FaithfulTRMACT(config)
        solve_kwargs = {}
        loss_kwargs = {}  # ACT handles its own loss computation

    elif model_type == 'faithful_act_attn':
        from experiments.exp_trm_vs_sc_sudoku.faithful_trm_act import FaithfulTRMACT, ACTConfig
        config = ACTConfig(
            hidden_size=128, max_H_cycles=5, L_cycles=6, L_layers=2,
            use_mlp_t=False,  # Attention version
            halt_exploration_prob=0.1,
        )
        model = FaithfulTRMACT(config)
        solve_kwargs = {}
        loss_kwargs = {}

    elif model_type == 'faithful_v2_attn':
        from experiments.exp_trm_vs_sc_sudoku.faithful_trm_v2 import FaithfulTRMv2, FaithfulTRMv2Config
        config = FaithfulTRMv2Config(
            hidden_size=128, H_cycles=3, L_cycles=6, L_layers=2,
            use_mlp_t=False,  # Use attention
        )
        model = FaithfulTRMv2(config)
        solve_kwargs = {}
        loss_kwargs = {}

    elif model_type == 'hard_mask':
        from experiments.exp_trm_vs_sc_sudoku.hard_mask_trm import HardMaskTRM, HardMaskConfig
        config = HardMaskConfig(
            hidden_size=64, num_heads=4, num_layers=2,
            mask_type='sudoku', mask_strength=1.0,
        )
        model = HardMaskTRM(config)
        solve_kwargs = {}
        loss_kwargs = {}

    elif model_type == 'hard_mask_iter':
        from experiments.exp_trm_vs_sc_sudoku.hard_mask_trm import HardMaskIterativeTRM, HardMaskConfig
        config = HardMaskConfig(
            hidden_size=64, num_heads=4, num_layers=2,
            mask_type='sudoku', mask_strength=1.0,
        )
        model = HardMaskIterativeTRM(config, num_iterations=3)
        solve_kwargs = {}
        loss_kwargs = {}

    elif model_type == 'nas_pretrained':
        from experiments.exp_trm_vs_sc_sudoku.nas_pretrained import NASPretrainedTRM, NASPretrainedConfig
        config = NASPretrainedConfig(
            hidden_size=64, num_heads=4, num_layers=2,
            base_constraint_weight=2.0, search_constraint_weight=0.0,
        )
        model = NASPretrainedTRM(config)
        solve_kwargs = {}
        loss_kwargs = {}

    elif model_type == 'large':
        from experiments.exp_trm_vs_sc_sudoku.large_trm import LargeTRM, LargeTRMConfig
        config = LargeTRMConfig(
            hidden_size=256, num_heads=8, num_layers=4,
            use_attention_bias=True,
        )
        model = LargeTRM(config)
        solve_kwargs = {}
        loss_kwargs = {}

    elif model_type == 'xlarge':
        from experiments.exp_trm_vs_sc_sudoku.large_trm import XLargeTRM, XLargeTRMConfig
        config = XLargeTRMConfig(
            hidden_size=512, num_heads=16, num_layers=6,
            use_attention_bias=True,
        )
        model = XLargeTRM(config)
        solve_kwargs = {}
        loss_kwargs = {}

    elif model_type == 'sae_diff':
        from experiments.exp_trm_vs_sc_sudoku.sae_diff_sc_trm import SAEDiffSCTRM, SAEDiffSCConfig
        config = SAEDiffSCConfig(
            hidden_size=64, num_heads=4, num_layers=2,
            sae_latent_dim=32, sae_k=8,
            topology_temp=1.0, topology_sparsity=0.1,
        )
        model = SAEDiffSCTRM(config)
        solve_kwargs = {}
        loss_kwargs = {'include_sae_loss': True, 'include_sparsity_loss': True}

    elif model_type == 'sae_diff_sudoku':
        from experiments.exp_trm_vs_sc_sudoku.sae_diff_sc_trm import SAEDiffSCTRM_Sudoku, SAEDiffSCConfig
        config = SAEDiffSCConfig(
            hidden_size=64, num_heads=4, num_layers=2,
            sae_latent_dim=32, sae_k=8,
            topology_temp=1.0, topology_sparsity=0.1,
        )
        model = SAEDiffSCTRM_Sudoku(config)
        solve_kwargs = {}
        loss_kwargs = {'include_sae_loss': True, 'include_sparsity_loss': True}

    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Training setup
    opt = optim.AdamW(learning_rate=1e-4, weight_decay=0.01)

    def loss_fn(m, p, s):
        loss, _ = m.loss(p, s, **loss_kwargs)
        return loss

    grad_fn = nn.value_and_grad(model, loss_fn)

    # Training history
    history = []
    epochs_to_50 = None
    epochs_to_70 = None

    batch_size = 32

    for epoch in range(epochs):
        epoch_loss = 0.0
        num_batches = 0

        perm = mx.random.permutation(train_p.shape[0])
        for i in range(0, train_p.shape[0], batch_size):
            p = train_p[perm[i:i+batch_size]]
            s = train_s[perm[i:i+batch_size]]
            loss, grads = grad_fn(model, p, s)
            opt.update(model, grads)
            mx.eval(model.parameters())
            epoch_loss += float(loss.item())
            num_batches += 1

        avg_loss = epoch_loss / num_batches

        # Evaluate periodically
        if (epoch + 1) % eval_every == 0 or epoch == epochs - 1:
            metrics = evaluate_model(model, test_p, test_s, **solve_kwargs)

            epoch_metrics = {
                'epoch': epoch + 1,
                'loss': avg_loss,
                'cell_accuracy': metrics['cell_accuracy'],
                'exact_accuracy': metrics['exact_accuracy'],
                'violation_rate': metrics['violation_rate'],
            }
            history.append(epoch_metrics)

            # Track milestones
            if epochs_to_50 is None and metrics['cell_accuracy'] >= 0.50:
                epochs_to_50 = epoch + 1
            if epochs_to_70 is None and metrics['cell_accuracy'] >= 0.70:
                epochs_to_70 = epoch + 1

            print(f"  Epoch {epoch+1}/{epochs}: loss={avg_loss:.4f}, "
                  f"cell={metrics['cell_accuracy']:.1%}, exact={metrics['exact_accuracy']:.1%}")

            # Incremental save if output_path provided
            if output_path:
                interim_result = {
                    'model_name': model_type,
                    'epochs': epochs,
                    'current_epoch': epoch + 1,
                    'seed': seed,
                    'status': 'running',
                    'train_history': history,
                    'final_cell_accuracy': metrics['cell_accuracy'],
                    'final_exact_accuracy': metrics['exact_accuracy'],
                    'final_violation_rate': metrics['violation_rate'],
                    'epochs_to_50_cell': epochs_to_50,
                    'epochs_to_70_cell': epochs_to_70,
                    'training_time_seconds': time.time() - start_time,
                }
                with open(output_path, 'w') as f:
                    json.dump(interim_result, f, indent=2)

    # Final evaluation
    final_metrics = evaluate_model(model, test_p, test_s, **solve_kwargs)
    training_time = time.time() - start_time

    return ExperimentResult(
        model_name=model_type,
        epochs=epochs,
        seed=seed,
        train_history=history,
        final_cell_accuracy=final_metrics['cell_accuracy'],
        final_exact_accuracy=final_metrics['exact_accuracy'],
        final_violation_rate=final_metrics['violation_rate'],
        epochs_to_50_cell=epochs_to_50,
        epochs_to_70_cell=epochs_to_70,
        training_time_seconds=training_time,
    )


def run_full_benchmark(
    epoch_counts: List[int],
    seeds: List[int],
    output_dir: str,
    n_train: int = 1000,
    n_test: int = 200,
):
    """Run full benchmark across all configurations."""
    from experiments.exp_trm_vs_sc_sudoku.sudoku_data import generate_random_sudoku

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'figures'), exist_ok=True)

    # Generate data once with fixed seed
    print("Generating Sudoku data...")
    train_p, train_s, test_p, test_s = generate_random_sudoku(n_train, n_test, seed=42)

    model_types = ['vanilla', 'faithful', 'ste', 'confidence', 'attention_bias', 'hierarchical']
    all_results = []

    total_configs = len(model_types) * len(epoch_counts) * len(seeds)
    config_idx = 0

    for model_type in model_types:
        for epochs in epoch_counts:
            for seed in seeds:
                config_idx += 1
                print(f"\n{'='*60}")
                print(f"[{config_idx}/{total_configs}] {model_type} | epochs={epochs} | seed={seed}")
                print('='*60)

                try:
                    result = train_model_with_history(
                        model_type=model_type,
                        train_p=train_p,
                        train_s=train_s,
                        test_p=test_p,
                        test_s=test_s,
                        epochs=epochs,
                        seed=seed,
                    )
                    all_results.append(result)

                    # Save individual result
                    filename = f"results_{model_type}_e{epochs}_s{seed}.json"
                    filepath = os.path.join(output_dir, filename)
                    with open(filepath, 'w') as f:
                        json.dump(asdict(result), f, indent=2)
                    print(f"  Saved to {filepath}")

                except Exception as e:
                    print(f"  ERROR: {e}")
                    import traceback
                    traceback.print_exc()

    # Save summary
    summary = {
        'epoch_counts': epoch_counts,
        'seeds': seeds,
        'model_types': model_types,
        'n_train': n_train,
        'n_test': n_test,
        'results': [asdict(r) for r in all_results],
    }

    summary_path = os.path.join(output_dir, 'summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {summary_path}")

    # Print final summary table
    print_summary_table(all_results)

    return all_results


def print_summary_table(results: List[ExperimentResult]):
    """Print a summary table of results."""
    print("\n" + "="*80)
    print("BENCHMARK SUMMARY")
    print("="*80)

    # Group by model and epochs
    from collections import defaultdict
    grouped = defaultdict(list)
    for r in results:
        key = (r.model_name, r.epochs)
        grouped[key].append(r)

    print(f"\n{'Model':<18} {'Epochs':>8} {'Cell Acc':>12} {'Exact Acc':>12} {'Time (s)':>10}")
    print("-"*70)

    for (model, epochs), runs in sorted(grouped.items()):
        cell_accs = [r.final_cell_accuracy for r in runs]
        exact_accs = [r.final_exact_accuracy for r in runs]
        times = [r.training_time_seconds for r in runs]

        mean_cell = sum(cell_accs) / len(cell_accs)
        mean_exact = sum(exact_accs) / len(exact_accs)
        mean_time = sum(times) / len(times)

        std_cell = (sum((x - mean_cell)**2 for x in cell_accs) / len(cell_accs))**0.5

        print(f"{model:<18} {epochs:>8} {mean_cell:>10.1%}±{std_cell:.1%} "
              f"{mean_exact:>10.1%} {mean_time:>10.1f}")


def load_data():
    """Load Sudoku data for single-script execution."""
    from experiments.exp_trm_vs_sc_sudoku.sudoku_data import load_sudoku_dataset
    # Force use of random generation for reliability in scripts
    # or rely on dataset loading if configured
    return load_sudoku_dataset(num_train=1000, num_test=200)


def run_quick_benchmark(output_dir: str = 'experiments/exp_trm_vs_sc_sudoku/benchmark_results'):
    """Run a quick benchmark with minimal settings for testing."""
    return run_full_benchmark(
        epoch_counts=[30],
        seeds=[42],
        output_dir=output_dir,
        n_train=500,
        n_test=100,
    )


def main():
    parser = argparse.ArgumentParser(description='SC-TRM Benchmark Runner')
    parser.add_argument('--epochs', type=str, default='30,50,100',
                        help='Comma-separated list of epoch counts')
    parser.add_argument('--seeds', type=int, default=3,
                        help='Number of seeds to run')
    parser.add_argument('--output', type=str,
                        default='experiments/exp_trm_vs_sc_sudoku/benchmark_results',
                        help='Output directory')
    parser.add_argument('--n-train', type=int, default=1000,
                        help='Number of training samples')
    parser.add_argument('--n-test', type=int, default=200,
                        help='Number of test samples')
    parser.add_argument('--quick', action='store_true',
                        help='Run quick benchmark (30 epochs, 1 seed)')

    args = parser.parse_args()

    if args.quick:
        run_quick_benchmark(args.output)
    else:
        epoch_counts = [int(x) for x in args.epochs.split(',')]
        seeds = list(range(args.seeds))

        run_full_benchmark(
            epoch_counts=epoch_counts,
            seeds=seeds,
            output_dir=args.output,
            n_train=args.n_train,
            n_test=args.n_test,
        )


if __name__ == "__main__":
    main()
