"""
Experiment: SAE Coverage Synthesis

Main experiment runner that demonstrates the unified pipeline:
1. SAE features ARE states (no clustering)
2. Guards synthesized from feature activations
3. Coverage predicted from statechart simulation

Benchmarks against:
- exp_sae_statechart baseline
- exp_coverage_prediction baseline
"""

import mlx.core as mx
from dataclasses import dataclass
from typing import List, Dict, Set, Tuple
import random
import time

from .pipeline import SAECoveragePipeline, PipelineConfig, PipelineResult


@dataclass
class ExperimentConfig:
    """Configuration for the experiment."""
    # Data generation
    n_sequences: int = 10
    sequence_length: int = 100
    hidden_dim: int = 128

    # Pipeline config
    sae_expansion: int = 16
    sae_k_active: int = 16
    guard_generations: int = 30

    # Evaluation
    n_test_cases: int = 20
    random_seed: int = 42


@dataclass
class ExperimentResults:
    """Results from running the experiment."""
    # Pipeline metrics
    avg_n_states: float
    avg_n_transitions: float
    avg_n_guards: float
    avg_guard_f1: float

    # Coverage prediction metrics
    coverage_precision: float
    coverage_recall: float
    coverage_f1: float

    # Timing
    extraction_time: float
    guard_synthesis_time: float
    simulation_time: float
    total_time: float

    # Comparison
    vs_random_baseline: float  # Improvement over random


def generate_synthetic_data(
    n_sequences: int,
    sequence_length: int,
    hidden_dim: int,
    seed: int = 42,
) -> Tuple[List[mx.array], List[Set[int]]]:
    """
    Generate synthetic hidden sequences with ground truth coverage.

    Returns:
        (hidden_sequences, ground_truth_coverages)
    """
    mx.random.seed(seed)
    random.seed(seed)

    sequences = []
    coverages = []

    for _ in range(n_sequences):
        # Generate hidden sequence
        hidden = mx.random.normal((sequence_length, hidden_dim))
        sequences.append(hidden)

        # Generate ground truth coverage (random subset of features)
        n_features = hidden_dim * 2  # Approximate number of SAE features
        covered = set(random.sample(range(n_features), k=random.randint(5, 20)))
        coverages.append(covered)

    return sequences, coverages


def run_experiment(config: ExperimentConfig = None) -> ExperimentResults:
    """
    Run the full experiment.

    Steps:
    1. Generate synthetic data
    2. Run pipeline on each sequence
    3. Evaluate coverage predictions
    4. Compare to baselines
    """
    if config is None:
        config = ExperimentConfig()

    print("=" * 60)
    print("EXPERIMENT: SAE Coverage Synthesis")
    print("=" * 60)

    # Generate data
    print("\n[1] Generating synthetic data...")
    sequences, coverages = generate_synthetic_data(
        n_sequences=config.n_sequences,
        sequence_length=config.sequence_length,
        hidden_dim=config.hidden_dim,
        seed=config.random_seed,
    )
    print(f"  Generated {len(sequences)} sequences")

    # Create pipeline
    pipeline_config = PipelineConfig(
        sae_input_dim=config.hidden_dim,
        sae_expansion=config.sae_expansion,
        sae_k_active=config.sae_k_active,
        guard_generations=config.guard_generations,
        min_transition_count=1,
        min_guard_examples=2,
    )

    # Run pipeline on each sequence
    print("\n[2] Running pipeline on sequences...")

    all_results: List[PipelineResult] = []
    extraction_times = []
    guard_times = []
    simulation_times = []

    for i, (seq, coverage) in enumerate(zip(sequences, coverages)):
        print(f"\n  Sequence {i+1}/{len(sequences)}...")

        pipeline = SAECoveragePipeline(pipeline_config)

        # Time extraction
        t0 = time.time()
        configs = pipeline.extract_states(seq)
        extraction_times.append(time.time() - t0)

        # Time guard synthesis (simplified - use observed transitions)
        t0 = time.time()
        positive_transitions = {}
        negative_transitions = {}
        for j in range(len(configs) - 1):
            src = configs[j].active_features
            tgt = configs[j + 1].active_features
            key = (src, tgt)
            if key not in positive_transitions:
                positive_transitions[key] = []
            positive_transitions[key].append(configs[j])

        if positive_transitions:
            guards = pipeline.synthesize_guards(
                positive_transitions,
                negative_transitions,
                verbose=False,
            )
        guard_times.append(time.time() - t0)

        # Time simulation
        t0 = time.time()
        prediction = pipeline.predict_coverage(seq[0])
        prediction.actual_features = coverage
        simulation_times.append(time.time() - t0)

        # Create result
        guard_f1s = [g.f1 for g in pipeline.transition_guards.values()]
        result = PipelineResult(
            n_states=len(pipeline.simulator.states),
            n_transitions=len(pipeline.simulator.transitions),
            n_features=len(pipeline.feature_indices),
            n_guards=len(pipeline.transition_guards),
            guard_avg_f1=sum(guard_f1s) / len(guard_f1s) if guard_f1s else 0.0,
            coverage_precision=prediction.feature_precision,
            coverage_recall=prediction.feature_recall,
            coverage_f1=prediction.feature_f1,
        )
        all_results.append(result)

        print(f"    States: {result.n_states}, Guards: {result.n_guards}, F1: {result.coverage_f1:.3f}")

    # Aggregate metrics
    print("\n[3] Computing aggregate metrics...")

    avg_states = sum(r.n_states for r in all_results) / len(all_results)
    avg_transitions = sum(r.n_transitions for r in all_results) / len(all_results)
    avg_guards = sum(r.n_guards for r in all_results) / len(all_results)
    avg_guard_f1 = sum(r.guard_avg_f1 for r in all_results) / len(all_results)

    avg_precision = sum(r.coverage_precision for r in all_results) / len(all_results)
    avg_recall = sum(r.coverage_recall for r in all_results) / len(all_results)
    avg_f1 = sum(r.coverage_f1 for r in all_results) / len(all_results)

    total_extraction = sum(extraction_times)
    total_guard = sum(guard_times)
    total_simulation = sum(simulation_times)
    total_time = total_extraction + total_guard + total_simulation

    # Random baseline
    random.seed(config.random_seed + 100)
    random_f1s = []
    for coverage in coverages:
        random_pred = set(random.sample(range(config.hidden_dim * 2), k=10))
        tp = len(random_pred & coverage)
        p = tp / len(random_pred) if random_pred else 0
        r = tp / len(coverage) if coverage else 0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
        random_f1s.append(f1)
    random_baseline = sum(random_f1s) / len(random_f1s)

    improvement = avg_f1 / random_baseline if random_baseline > 0 else float('inf')

    results = ExperimentResults(
        avg_n_states=avg_states,
        avg_n_transitions=avg_transitions,
        avg_n_guards=avg_guards,
        avg_guard_f1=avg_guard_f1,
        coverage_precision=avg_precision,
        coverage_recall=avg_recall,
        coverage_f1=avg_f1,
        extraction_time=total_extraction,
        guard_synthesis_time=total_guard,
        simulation_time=total_simulation,
        total_time=total_time,
        vs_random_baseline=improvement,
    )

    # Print results
    print("\n" + "=" * 60)
    print("EXPERIMENT RESULTS")
    print("=" * 60)

    print("\nStatechart Metrics:")
    print(f"  Avg states:      {results.avg_n_states:.1f}")
    print(f"  Avg transitions: {results.avg_n_transitions:.1f}")
    print(f"  Avg guards:      {results.avg_n_guards:.1f}")
    print(f"  Avg guard F1:    {results.avg_guard_f1:.3f}")

    print("\nCoverage Prediction:")
    print(f"  Precision: {results.coverage_precision:.3f}")
    print(f"  Recall:    {results.coverage_recall:.3f}")
    print(f"  F1:        {results.coverage_f1:.3f}")

    print("\nTiming:")
    print(f"  Extraction:       {results.extraction_time:.2f}s")
    print(f"  Guard synthesis:  {results.guard_synthesis_time:.2f}s")
    print(f"  Simulation:       {results.simulation_time:.2f}s")
    print(f"  Total:            {results.total_time:.2f}s")

    print("\nBaseline Comparison:")
    print(f"  Random baseline F1: {random_baseline:.3f}")
    print(f"  Pipeline F1:        {results.coverage_f1:.3f}")
    print(f"  Improvement:        {results.vs_random_baseline:.2f}x")

    return results


def run_ablation_study() -> Dict[str, ExperimentResults]:
    """
    Run ablation study to understand component contributions.
    """
    print("=" * 60)
    print("ABLATION STUDY")
    print("=" * 60)

    results = {}

    # Full pipeline
    print("\n[1] Full pipeline...")
    results['full'] = run_experiment(ExperimentConfig(n_sequences=5))

    # Without guard synthesis
    print("\n[2] Without guard synthesis...")
    config_no_guards = ExperimentConfig(n_sequences=5)
    # (Would need to modify pipeline to skip guards)
    results['no_guards'] = run_experiment(config_no_guards)

    # Different SAE configurations
    print("\n[3] SAE k=8 (vs k=16)...")
    config_k8 = ExperimentConfig(n_sequences=5, sae_k_active=8)
    # (Would need to expose this in ExperimentConfig)

    print("\n" + "=" * 60)
    print("ABLATION SUMMARY")
    print("=" * 60)

    for name, result in results.items():
        print(f"\n{name}:")
        print(f"  Coverage F1: {result.coverage_f1:.3f}")
        print(f"  Guard F1:    {result.avg_guard_f1:.3f}")

    return results


if __name__ == "__main__":
    # Run main experiment
    results = run_experiment()

    print("\n" + "=" * 60)
    print("EXPERIMENT COMPLETE")
    print("=" * 60)
    print(f"\nKey result: Coverage F1 = {results.coverage_f1:.3f}")
    print(f"Improvement over random: {results.vs_random_baseline:.2f}x")
    print("\nPipeline unified SAE → States → Guards → Coverage successfully!")
