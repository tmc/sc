"""
Evaluation script for Sudoku Statechart 9x9.

Compares with TRM baseline metrics:
- TRM achieves 87% exact accuracy on Sudoku-Extreme
- Target: match or exceed with interpretable structure

Usage:
    python -m ml.experiments.exp_trm_sudoku_9x9.evaluate --checkpoint best_model.safetensors
"""

import argparse
import json
import os
import time
from dataclasses import dataclass
from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from .sudoku_statechart_9x9 import SudokuStatechart9x9
from .data_loader import SudokuExtremeDataset
from .losses import stablemax_cross_entropy


@dataclass
class EvalResult:
    """Evaluation results."""
    exact_accuracy: float      # Fraction of boards completely correct
    cell_accuracy: float       # Fraction of cells correct
    violation_rate: float      # Fraction of boards with constraint violations
    avg_iterations: float      # Average iterations used (if adaptive)
    samples_per_second: float  # Throughput
    total_samples: int


def evaluate_checkpoint(
    checkpoint_path: str,
    hidden_dim: int = 128,
    H_cycles: int = 3,
    L_cycles: int = 6,
    batch_size: int = 64,
    max_samples: Optional[int] = None,
    split: str = "test",
) -> EvalResult:
    """
    Evaluate a trained checkpoint.

    Args:
        checkpoint_path: Path to saved model weights
        hidden_dim: Model hidden dimension
        H_cycles: Number of outer cycles
        L_cycles: Number of inner cycles
        batch_size: Evaluation batch size
        max_samples: Maximum samples to evaluate
        split: Dataset split ("train" or "test")

    Returns:
        EvalResult with metrics
    """
    print(f"Loading model from {checkpoint_path}...")

    # Create model
    model = SudokuStatechart9x9(
        hidden_dim=hidden_dim,
        H_cycles=H_cycles,
        L_cycles=L_cycles,
    )

    # Load weights
    if os.path.exists(checkpoint_path):
        weights = nn.utils.load(checkpoint_path)
        # Convert flat dict to nested structure
        model.load_weights(list(weights.items()))
        print("Loaded weights successfully")
    else:
        print(f"Warning: Checkpoint not found at {checkpoint_path}")

    # Load dataset
    print(f"Loading {split} dataset...")
    try:
        dataset = SudokuExtremeDataset(
            split=split,
            max_samples=max_samples,
        )
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return None

    # Evaluate
    print(f"Evaluating on {len(dataset)} samples...")

    total_exact = 0
    total_cell_correct = 0
    total_cells = 0
    total_violations = 0
    total_samples = 0
    start_time = time.time()

    for questions, answers in dataset.iter_batches(batch_size, shuffle=False):
        # Forward pass
        logits, _ = model.forward(questions)
        predictions = mx.argmax(logits, axis=-1)

        # Exact accuracy
        exact_correct = mx.sum(mx.all(predictions == answers, axis=-1).astype(mx.float32)).item()
        total_exact += exact_correct

        # Cell accuracy
        cell_correct = mx.sum((predictions == answers).astype(mx.float32)).item()
        total_cell_correct += cell_correct
        total_cells += predictions.size

        # Constraint violations
        valid, _ = model.is_valid(predictions)
        violations = mx.sum((~valid).astype(mx.float32)).item()
        total_violations += violations

        total_samples += predictions.shape[0]

    elapsed = time.time() - start_time

    return EvalResult(
        exact_accuracy=total_exact / max(total_samples, 1),
        cell_accuracy=total_cell_correct / max(total_cells, 1),
        violation_rate=total_violations / max(total_samples, 1),
        avg_iterations=H_cycles * L_cycles,
        samples_per_second=total_samples / elapsed,
        total_samples=total_samples,
    )


def print_comparison(result: EvalResult):
    """Print comparison with TRM baseline."""
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)

    print(f"\nStatechart-TRM Results:")
    print(f"  Exact Accuracy:    {result.exact_accuracy * 100:.2f}%")
    print(f"  Cell Accuracy:     {result.cell_accuracy * 100:.2f}%")
    print(f"  Violation Rate:    {result.violation_rate * 100:.2f}%")
    print(f"  Avg Iterations:    {result.avg_iterations:.1f}")
    print(f"  Throughput:        {result.samples_per_second:.1f} samples/sec")
    print(f"  Total Samples:     {result.total_samples}")

    print(f"\nTRM Baseline Comparison:")
    print(f"  TRM Exact Accuracy: 87.00%")
    print(f"  TRM Iterations:     18 (H=3 × L=6)")

    diff = (result.exact_accuracy - 0.87) * 100
    print(f"\n  Difference:         {diff:+.2f}% {'(better)' if diff > 0 else '(worse)' if diff < 0 else '(same)'}")

    print("\n" + "=" * 60)


def analyze_by_difficulty(
    checkpoint_path: str,
    hidden_dim: int = 128,
    H_cycles: int = 3,
    L_cycles: int = 6,
    batch_size: int = 64,
):
    """
    Analyze performance by puzzle difficulty.

    Groups puzzles by their difficulty rating and reports
    accuracy for each group.
    """
    print("\nAnalyzing performance by difficulty...")

    # Load model
    model = SudokuStatechart9x9(
        hidden_dim=hidden_dim,
        H_cycles=H_cycles,
        L_cycles=L_cycles,
    )

    if os.path.exists(checkpoint_path):
        weights = nn.utils.load(checkpoint_path)
        model.load_weights(list(weights.items()))

    # Load full dataset with ratings
    try:
        dataset = SudokuExtremeDataset(split="test", max_samples=5000)
    except FileNotFoundError:
        print("Dataset not available")
        return

    # Group by difficulty (assuming we have access to ratings)
    # For now, analyze overall distribution
    difficulty_buckets = {
        "easy": (0, 1000),
        "medium": (1000, 2000),
        "hard": (2000, 3000),
        "extreme": (3000, float("inf")),
    }

    print("\nDifficulty Analysis:")
    print("-" * 40)

    # Since we don't store ratings per sample in batches,
    # we'll analyze based on error patterns instead
    total_correct = 0
    total_samples = 0
    error_by_empty_cells = {}

    for questions, answers in dataset.iter_batches(batch_size, shuffle=False):
        logits, _ = model.forward(questions)
        predictions = mx.argmax(logits, axis=-1)

        # Count empty cells per puzzle as difficulty proxy
        for i in range(questions.shape[0]):
            empty_count = mx.sum(questions[i] == 0).item()
            is_correct = mx.all(predictions[i] == answers[i]).item()

            bucket = empty_count // 10 * 10  # Group by 10s
            if bucket not in error_by_empty_cells:
                error_by_empty_cells[bucket] = {"correct": 0, "total": 0}
            error_by_empty_cells[bucket]["total"] += 1
            error_by_empty_cells[bucket]["correct"] += int(is_correct)

            total_correct += int(is_correct)
            total_samples += 1

    print(f"\nAccuracy by empty cell count (difficulty proxy):")
    for bucket in sorted(error_by_empty_cells.keys()):
        stats = error_by_empty_cells[bucket]
        acc = stats["correct"] / max(stats["total"], 1)
        print(f"  {bucket}-{bucket+9} empty cells: {acc*100:.1f}% ({stats['total']} samples)")

    print(f"\nOverall: {total_correct/max(total_samples, 1)*100:.2f}%")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Sudoku Statechart 9x9")

    parser.add_argument("--checkpoint", type=str, default="checkpoints/exp_trm_sudoku_9x9/best_model.safetensors")
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--H_cycles", type=int, default=3)
    parser.add_argument("--L_cycles", type=int, default=6)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--analyze_difficulty", action="store_true")

    args = parser.parse_args()

    # Main evaluation
    result = evaluate_checkpoint(
        checkpoint_path=args.checkpoint,
        hidden_dim=args.hidden_dim,
        H_cycles=args.H_cycles,
        L_cycles=args.L_cycles,
        batch_size=args.batch_size,
        max_samples=args.max_samples,
        split=args.split,
    )

    if result:
        print_comparison(result)

        # Difficulty analysis
        if args.analyze_difficulty:
            analyze_by_difficulty(
                checkpoint_path=args.checkpoint,
                hidden_dim=args.hidden_dim,
                H_cycles=args.H_cycles,
                L_cycles=args.L_cycles,
                batch_size=args.batch_size,
            )


if __name__ == "__main__":
    main()
