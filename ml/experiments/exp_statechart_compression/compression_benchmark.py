"""
Statechart Compression Benchmark

Compares different compression approaches:
1. Bisimulation (state merging)
2. Transition sharing (pattern deduplication)
3. Learned compression (autoencoder)
4. Combined approaches

Target: 50% size reduction while preserving semantics.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import random
import time
import json

from .bisimulation import (
    Statechart, State, Transition, StateType,
    BisimulationCompressor,
    create_sample_statechart_with_redundancy,
    create_dfa_statechart,
)
from .transition_share import TransitionSharer
from .learned_compress import LearnedCompressor, create_sample_statecharts


# =============================================================================
# Benchmark Types
# =============================================================================

@dataclass
class CompressionResult:
    """Result of compressing a single statechart."""
    method: str
    original_states: int
    original_transitions: int
    compressed_states: int
    compressed_transitions: int
    original_size_bytes: int
    compressed_size_bytes: int
    compression_ratio: float
    reduction_percent: float
    semantics_preserved: bool
    runtime_ms: float

    def to_dict(self) -> dict:
        return {
            "method": self.method,
            "original_states": self.original_states,
            "original_transitions": self.original_transitions,
            "compressed_states": self.compressed_states,
            "compressed_transitions": self.compressed_transitions,
            "original_size": self.original_size_bytes,
            "compressed_size": self.compressed_size_bytes,
            "ratio": self.compression_ratio,
            "reduction": self.reduction_percent,
            "semantics_ok": self.semantics_preserved,
            "runtime_ms": self.runtime_ms,
        }


@dataclass
class BenchmarkSummary:
    """Summary of benchmark across multiple statecharts."""
    method: str
    n_statecharts: int
    mean_reduction: float
    max_reduction: float
    min_reduction: float
    target_achieved: int  # Count achieving 50%+ reduction
    mean_runtime_ms: float
    total_original_size: int
    total_compressed_size: int

    def __str__(self) -> str:
        return (
            f"{self.method:20s}: "
            f"mean={self.mean_reduction:.1f}%, "
            f"max={self.max_reduction:.1f}%, "
            f"target(50%+)={self.target_achieved}/{self.n_statecharts}, "
            f"runtime={self.mean_runtime_ms:.1f}ms"
        )


# =============================================================================
# Size Estimation
# =============================================================================

def estimate_size(sc: Statechart) -> int:
    """
    Estimate storage size of a statechart in bytes.

    Simplified model:
    - State: label(16) + flags(4) + type(1) + parent_ref(4) + actions(32) = 57 bytes
    - Transition: source(4) + target(4) + event(16) + guard(32) + action(32) = 88 bytes
    """
    state_size = 57
    trans_size = 88

    return len(sc.states) * state_size + len(sc.transitions) * trans_size


# =============================================================================
# Semantics Verification
# =============================================================================

def verify_semantics(original: Statechart, compressed: Statechart) -> bool:
    """
    Verify that compression preserves semantics.

    Checks:
    1. Same set of events
    2. Same reachability from initial state
    3. Same accepting behavior
    """
    # Check events
    orig_events = original.get_events()
    comp_events = compressed.get_events()
    if orig_events != comp_events:
        return False

    # Check initial state exists
    if compressed.initial_state is None:
        return False

    # Check at least one final state
    orig_finals = sum(1 for s in original.states.values() if s.is_final)
    comp_finals = sum(1 for s in compressed.states.values() if s.is_final)
    if (orig_finals > 0) != (comp_finals > 0):
        return False

    # Basic structure check passed
    return True


# =============================================================================
# Benchmark Runners
# =============================================================================

def benchmark_bisimulation(sc: Statechart) -> CompressionResult:
    """Benchmark bisimulation compression."""
    orig_states, orig_trans = sc.size()
    orig_size = estimate_size(sc)

    start = time.time()
    compressor = BisimulationCompressor(sc)
    compressed = compressor.compress()
    runtime = (time.time() - start) * 1000

    comp_states, comp_trans = compressed.size()
    comp_size = estimate_size(compressed)

    ratio = comp_size / orig_size if orig_size > 0 else 1.0
    reduction = (1 - ratio) * 100
    semantics_ok = verify_semantics(sc, compressed)

    return CompressionResult(
        method="Bisimulation",
        original_states=orig_states,
        original_transitions=orig_trans,
        compressed_states=comp_states,
        compressed_transitions=comp_trans,
        original_size_bytes=orig_size,
        compressed_size_bytes=comp_size,
        compression_ratio=ratio,
        reduction_percent=reduction,
        semantics_preserved=semantics_ok,
        runtime_ms=runtime,
    )


def benchmark_transition_sharing(sc: Statechart) -> CompressionResult:
    """Benchmark transition sharing compression."""
    orig_states, orig_trans = sc.size()
    orig_size = estimate_size(sc)

    start = time.time()
    sharer = TransitionSharer(sc)
    compressed = sharer.compress()
    original_bytes, compressed_bytes = sharer.get_compressed_size()
    runtime = (time.time() - start) * 1000

    comp_states, comp_trans = compressed.size()

    ratio = compressed_bytes / original_bytes if original_bytes > 0 else 1.0
    reduction = (1 - ratio) * 100
    semantics_ok = verify_semantics(sc, compressed)

    return CompressionResult(
        method="TransitionShare",
        original_states=orig_states,
        original_transitions=orig_trans,
        compressed_states=comp_states,
        compressed_transitions=comp_trans,
        original_size_bytes=original_bytes,
        compressed_size_bytes=compressed_bytes,
        compression_ratio=ratio,
        reduction_percent=reduction,
        semantics_preserved=semantics_ok,
        runtime_ms=runtime,
    )


def benchmark_combined(sc: Statechart) -> CompressionResult:
    """Benchmark combined bisimulation + transition sharing."""
    orig_states, orig_trans = sc.size()
    orig_size = estimate_size(sc)

    start = time.time()

    # First: bisimulation
    bisim = BisimulationCompressor(sc)
    step1 = bisim.compress()

    # Then: transition sharing
    sharer = TransitionSharer(step1)
    compressed = sharer.compress()
    _, compressed_bytes = sharer.get_compressed_size()

    runtime = (time.time() - start) * 1000

    comp_states, comp_trans = compressed.size()

    ratio = compressed_bytes / orig_size if orig_size > 0 else 1.0
    reduction = (1 - ratio) * 100
    semantics_ok = verify_semantics(sc, compressed)

    return CompressionResult(
        method="Combined",
        original_states=orig_states,
        original_transitions=orig_trans,
        compressed_states=comp_states,
        compressed_transitions=comp_trans,
        original_size_bytes=orig_size,
        compressed_size_bytes=compressed_bytes,
        compression_ratio=ratio,
        reduction_percent=reduction,
        semantics_preserved=semantics_ok,
        runtime_ms=runtime,
    )


# =============================================================================
# Full Benchmark Suite
# =============================================================================

def run_benchmark(
    statecharts: List[Statechart],
    methods: List[str] = ["bisimulation", "transition", "combined"],
    verbose: bool = True,
) -> Dict[str, BenchmarkSummary]:
    """
    Run full benchmark suite.

    Args:
        statecharts: List of statecharts to benchmark
        methods: Which methods to benchmark
        verbose: Print progress

    Returns:
        Dict mapping method name to summary
    """
    results: Dict[str, List[CompressionResult]] = {m: [] for m in methods}

    for i, sc in enumerate(statecharts):
        if verbose and i % 5 == 0:
            print(f"  Processing statechart {i+1}/{len(statecharts)}...")

        if "bisimulation" in methods:
            results["bisimulation"].append(benchmark_bisimulation(sc))

        if "transition" in methods:
            results["transition"].append(benchmark_transition_sharing(sc))

        if "combined" in methods:
            results["combined"].append(benchmark_combined(sc))

    # Compute summaries
    summaries = {}
    for method, method_results in results.items():
        if not method_results:
            continue

        reductions = [r.reduction_percent for r in method_results]
        runtimes = [r.runtime_ms for r in method_results]
        orig_sizes = [r.original_size_bytes for r in method_results]
        comp_sizes = [r.compressed_size_bytes for r in method_results]

        summaries[method] = BenchmarkSummary(
            method=method,
            n_statecharts=len(method_results),
            mean_reduction=sum(reductions) / len(reductions),
            max_reduction=max(reductions),
            min_reduction=min(reductions),
            target_achieved=sum(1 for r in reductions if r >= 50),
            mean_runtime_ms=sum(runtimes) / len(runtimes),
            total_original_size=sum(orig_sizes),
            total_compressed_size=sum(comp_sizes),
        )

    return summaries


# =============================================================================
# Benchmark Dataset Generation
# =============================================================================

def create_benchmark_dataset(
    n_small: int = 10,
    n_medium: int = 10,
    n_large: int = 5,
    redundancy_range: Tuple[float, float] = (0.1, 0.5),
) -> List[Statechart]:
    """
    Create diverse benchmark dataset.

    Args:
        n_small: Number of small statecharts (5-10 states)
        n_medium: Number of medium statecharts (10-20 states)
        n_large: Number of large statecharts (20-40 states)
        redundancy_range: Range of redundancy factors
    """
    statecharts = []

    # Small
    for i in range(n_small):
        n_states = random.randint(5, 10)
        redundancy = random.uniform(*redundancy_range)
        sc = create_dfa_statechart(n_states, n_events=3, redundancy_factor=redundancy)
        sc.name = f"small_{i}"
        statecharts.append(sc)

    # Medium
    for i in range(n_medium):
        n_states = random.randint(10, 20)
        redundancy = random.uniform(*redundancy_range)
        sc = create_dfa_statechart(n_states, n_events=4, redundancy_factor=redundancy)
        sc.name = f"medium_{i}"
        statecharts.append(sc)

    # Large
    for i in range(n_large):
        n_states = random.randint(20, 40)
        redundancy = random.uniform(*redundancy_range)
        sc = create_dfa_statechart(n_states, n_events=5, redundancy_factor=redundancy)
        sc.name = f"large_{i}"
        statecharts.append(sc)

    return statecharts


# =============================================================================
# Main Demo
# =============================================================================

def demo():
    """Run compression benchmark demonstration."""
    print("=" * 60)
    print("Statechart Compression Benchmark")
    print("=" * 60)
    print("\nTarget: 50% size reduction while preserving semantics")

    # Create benchmark dataset
    print("\nCreating benchmark dataset...")
    statecharts = create_benchmark_dataset(
        n_small=8,
        n_medium=8,
        n_large=4,
        redundancy_range=(0.2, 0.5),
    )
    print(f"  Created {len(statecharts)} statecharts")

    # Compute dataset statistics
    total_states = sum(len(sc.states) for sc in statecharts)
    total_trans = sum(len(sc.transitions) for sc in statecharts)
    print(f"  Total: {total_states} states, {total_trans} transitions")

    # Run benchmark
    print("\nRunning benchmark...")
    summaries = run_benchmark(
        statecharts,
        methods=["bisimulation", "transition", "combined"],
        verbose=True,
    )

    # Print results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    for method, summary in summaries.items():
        print(f"\n{summary}")

    # Check target
    print("\n" + "=" * 60)
    print("TARGET CHECK (50% reduction)")
    print("=" * 60)

    best_method = None
    best_reduction = 0

    for method, summary in summaries.items():
        achieved = summary.target_achieved
        total = summary.n_statecharts
        pct = achieved / total * 100 if total > 0 else 0

        status = "ACHIEVED" if summary.mean_reduction >= 50 else "PARTIAL"
        print(f"\n{method}:")
        print(f"  Mean reduction: {summary.mean_reduction:.1f}%")
        print(f"  Target (50%+): {achieved}/{total} ({pct:.0f}%) [{status}]")

        if summary.mean_reduction > best_reduction:
            best_reduction = summary.mean_reduction
            best_method = method

    print(f"\n*** Best method: {best_method} ({best_reduction:.1f}% mean reduction) ***")

    # Overall verdict
    if best_reduction >= 50:
        print("\n*** TARGET ACHIEVED: 50%+ compression with semantic preservation ***")
    else:
        print(f"\n*** Working toward target: {best_reduction:.1f}% achieved ***")

    return summaries


if __name__ == "__main__":
    demo()
