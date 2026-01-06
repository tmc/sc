"""
Benchmark: Full interpretability benchmark for TRM-style Sudoku model.

Runs comprehensive analysis combining:
1. Iteration dynamics analysis
2. Activation tracking
3. Solution correlation
4. Generates interpretability report
"""

import json
import os
import sys
import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional

import mlx.core as mx
import numpy as np

sys.path.insert(0, '/Volumes/tmc/go/src/github.com/tmc/sc/ml')

from .iteration_analyzer import (
    IterationAnalyzer,
    analyze_iterations,
    summarize_trajectories,
)
from .activation_tracker import (
    ActivationTracker,
    track_activations,
    analyze_evolution,
)
from .solution_correlator import (
    SolutionCorrelator,
    correlate_solutions,
)


@dataclass
class BenchmarkConfig:
    """Configuration for interpretability benchmark."""
    # Data
    num_puzzles: int = 100
    batch_size: int = 20

    # Model
    H_cycles: int = 3
    L_cycles: int = 6

    # Analysis
    run_iteration_analysis: bool = True
    run_activation_tracking: bool = True
    run_correlation_analysis: bool = True

    # Output
    save_dir: str = "interpretability_results"
    save_trajectories: bool = False


@dataclass
class BenchmarkResult:
    """Results from interpretability benchmark."""
    config: BenchmarkConfig
    runtime_seconds: float

    # Overall metrics
    num_puzzles: int
    num_correct: int
    accuracy: float

    # Iteration analysis
    iteration_summary: Dict

    # Activation analysis
    activation_summary: Dict

    # Correlation analysis
    correlation_summary: Dict

    # Key findings
    findings: List[str]


def run_interpretability_benchmark(
    model,
    questions: mx.array,
    answers: mx.array,
    config: Optional[BenchmarkConfig] = None,
) -> BenchmarkResult:
    """
    Run full interpretability benchmark.

    Args:
        model: Trained SudokuStatechart9x9 model
        questions: [N, 81] input puzzles
        answers: [N, 81] target solutions
        config: Benchmark configuration

    Returns:
        BenchmarkResult with all analyses
    """
    config = config or BenchmarkConfig()

    print("=" * 60)
    print("TRM INTERPRETABILITY BENCHMARK")
    print("=" * 60)
    print(f"Puzzles: {questions.shape[0]}")
    print(f"H_cycles: {config.H_cycles}, L_cycles: {config.L_cycles}")

    start_time = time.time()
    findings = []

    # Limit puzzles if needed
    N = min(questions.shape[0], config.num_puzzles)
    questions = questions[:N]
    answers = answers[:N]

    # 1. Iteration Analysis
    iteration_summary = {}
    if config.run_iteration_analysis:
        print("\n--- Iteration Analysis ---")
        trajectories = analyze_iterations(
            model, questions, answers,
            H_cycles=config.H_cycles,
            L_cycles=config.L_cycles,
        )
        iteration_summary = summarize_trajectories(trajectories)

        print(f"Accuracy: {iteration_summary['accuracy']:.2%}")
        if iteration_summary.get('avg_convergence_iter'):
            print(f"Avg convergence: {iteration_summary['avg_convergence_iter']:.1f} iterations")

        # Key findings
        entropy_traj = iteration_summary.get('entropy_by_iteration', [])
        if entropy_traj:
            entropy_drop = entropy_traj[0] - entropy_traj[-1]
            findings.append(f"Entropy drops by {entropy_drop:.2f} over iterations")

        acc_traj = iteration_summary.get('accuracy_by_iteration', [])
        if acc_traj and len(acc_traj) > 6:
            early_acc = acc_traj[5]  # After first H-cycle
            findings.append(f"Accuracy at iteration 6: {early_acc:.2%}")

    # 2. Activation Tracking
    activation_summary = {}
    if config.run_activation_tracking:
        print("\n--- Activation Tracking ---")

        # Track a subset
        sample_size = min(10, N)
        sample_questions = questions[:sample_size]

        tracker = ActivationTracker(model)
        all_evolutions = []

        for i in range(sample_size):
            evolutions = tracker.track_forward(
                sample_questions[i],
                H_cycles=config.H_cycles,
                L_cycles=config.L_cycles,
            )
            all_evolutions.append(evolutions)

        # Aggregate analysis
        if all_evolutions:
            first_evo = all_evolutions[0]
            activation_summary = analyze_evolution(first_evo)

            # Key findings from activations
            for hook_name, stats in activation_summary.items():
                if 'final_sparsity' in stats:
                    findings.append(
                        f"{hook_name} final sparsity: {stats['final_sparsity']:.2%}"
                    )

            # Representation drift
            for hook_name in first_evo.keys():
                drift = tracker.compute_representation_drift(hook_name)
                if len(drift) > 0:
                    activation_summary[f"{hook_name}_drift"] = {
                        "mean": float(np.mean(drift)),
                        "max": float(np.max(drift)),
                    }
                    findings.append(
                        f"{hook_name} avg drift: {np.mean(drift):.3f}"
                    )

        print(f"Tracked {len(activation_summary)} activation types")

    # 3. Correlation Analysis
    correlation_summary = {}
    if config.run_correlation_analysis:
        print("\n--- Correlation Analysis ---")

        correlation_summary = correlate_solutions(
            model, questions, answers,
            H_cycles=config.H_cycles,
            L_cycles=config.L_cycles,
        )

        print(f"Correct: {correlation_summary['num_correct']}, "
              f"Incorrect: {correlation_summary['num_incorrect']}")

        # Key findings from correlations
        for corr in correlation_summary.get('top_correlations', [])[:3]:
            if abs(corr['correlation']) > 0.1:
                findings.append(
                    f"{corr['feature']} correlated with success (r={corr['correlation']:.2f})"
                )

        for feat in correlation_summary.get('differentiating_features', []):
            findings.append(feat['description'])

    runtime = time.time() - start_time

    # Compute overall accuracy
    num_correct = correlation_summary.get('num_correct', 0) if correlation_summary else 0

    result = BenchmarkResult(
        config=config,
        runtime_seconds=runtime,
        num_puzzles=N,
        num_correct=num_correct,
        accuracy=num_correct / N if N > 0 else 0,
        iteration_summary=iteration_summary,
        activation_summary=activation_summary,
        correlation_summary=correlation_summary,
        findings=findings,
    )

    # Print summary
    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)
    print(f"Runtime: {runtime:.1f}s")
    print(f"Accuracy: {result.accuracy:.2%}")
    print(f"\nKey Findings:")
    for finding in findings[:10]:
        print(f"  - {finding}")

    # Save results
    if config.save_dir:
        save_benchmark_results(result, config.save_dir)

    return result


def save_benchmark_results(result: BenchmarkResult, save_dir: str):
    """Save benchmark results to disk."""
    os.makedirs(save_dir, exist_ok=True)

    # Save JSON summary
    summary = {
        "runtime_seconds": result.runtime_seconds,
        "num_puzzles": result.num_puzzles,
        "num_correct": result.num_correct,
        "accuracy": result.accuracy,
        "findings": result.findings,
        "iteration_summary": result.iteration_summary,
        "correlation_summary": result.correlation_summary,
    }

    summary_path = os.path.join(save_dir, "benchmark_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved results to {summary_path}")

    # Save detailed report
    report = generate_report(result)
    report_path = os.path.join(save_dir, "interpretability_report.md")
    with open(report_path, 'w') as f:
        f.write(report)

    print(f"Saved report to {report_path}")


def generate_report(result: BenchmarkResult) -> str:
    """Generate markdown interpretability report."""
    lines = [
        "# TRM Interpretability Benchmark Report",
        "",
        "## Overview",
        f"- Puzzles analyzed: {result.num_puzzles}",
        f"- Accuracy: {result.accuracy:.2%}",
        f"- Runtime: {result.runtime_seconds:.1f}s",
        "",
        "## Key Findings",
    ]

    for finding in result.findings:
        lines.append(f"- {finding}")

    lines.extend([
        "",
        "## Iteration Dynamics",
    ])

    if result.iteration_summary:
        if 'accuracy_by_iteration' in result.iteration_summary:
            lines.append("\n### Accuracy Trajectory")
            acc_traj = result.iteration_summary['accuracy_by_iteration']
            for i, acc in enumerate(acc_traj):
                lines.append(f"- Iteration {i}: {acc:.2%}")

        if 'entropy_by_iteration' in result.iteration_summary:
            lines.append("\n### Entropy Trajectory")
            ent_traj = result.iteration_summary['entropy_by_iteration']
            lines.append(f"- Start: {ent_traj[0]:.3f}")
            lines.append(f"- End: {ent_traj[-1]:.3f}")
            lines.append(f"- Reduction: {ent_traj[0] - ent_traj[-1]:.3f}")

    lines.extend([
        "",
        "## Correlation Analysis",
    ])

    if result.correlation_summary:
        corrs = result.correlation_summary.get('top_correlations', [])
        if corrs:
            lines.append("\n### Top Correlations with Correctness")
            for c in corrs:
                lines.append(
                    f"- {c['feature']}: r={c['correlation']:.3f}, "
                    f"effect_size={c['effect_size']:.2f}"
                )

        lines.append("\n### Correct vs Incorrect Signatures")
        correct = result.correlation_summary.get('correct_signature', {})
        incorrect = result.correlation_summary.get('incorrect_signature', {})

        lines.append("\n| Metric | Correct | Incorrect |")
        lines.append("|--------|---------|-----------|")
        for key in correct.keys():
            c_val = correct.get(key, 'N/A')
            i_val = incorrect.get(key, 'N/A')
            if isinstance(c_val, float):
                c_val = f"{c_val:.3f}"
                i_val = f"{i_val:.3f}" if isinstance(i_val, float) else i_val
            lines.append(f"| {key} | {c_val} | {i_val} |")

    lines.extend([
        "",
        "## Conclusions",
        "",
        "Based on the analysis:",
    ])

    # Generate conclusions from findings
    if result.accuracy > 0.5:
        lines.append("- Model achieves reasonable accuracy")
    else:
        lines.append("- Model accuracy is low, needs training")

    if result.iteration_summary.get('avg_convergence_iter'):
        conv = result.iteration_summary['avg_convergence_iter']
        total = result.config.H_cycles * result.config.L_cycles
        if conv < total * 0.7:
            lines.append(f"- Solutions converge early (iteration {conv:.1f}/{total})")
        else:
            lines.append("- Solutions use most available iterations")

    return "\n".join(lines)


def test_benchmark():
    """Test the benchmark."""
    print("=" * 60)
    print("Testing Interpretability Benchmark")
    print("=" * 60)

    from ..exp_trm_sudoku_9x9.sudoku_statechart_9x9 import SudokuStatechart9x9

    # Create model
    model = SudokuStatechart9x9(
        hidden_dim=64,
        H_cycles=2,
        L_cycles=3,
    )

    # Create test data
    mx.random.seed(42)
    questions = mx.random.randint(0, 10, (20, 81))
    answers = mx.random.randint(1, 10, (20, 81))

    # Run benchmark
    config = BenchmarkConfig(
        num_puzzles=20,
        H_cycles=2,
        L_cycles=3,
        save_dir="/tmp/test_interpretability",
    )

    result = run_interpretability_benchmark(model, questions, answers, config)

    print("\n" + "=" * 60)
    print(f"Benchmark complete: {result.num_puzzles} puzzles, "
          f"{result.accuracy:.2%} accuracy")
    print("=" * 60)


if __name__ == "__main__":
    test_benchmark()
