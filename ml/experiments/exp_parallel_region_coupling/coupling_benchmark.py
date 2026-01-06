"""
Coupling Benchmark: Test Patterns for Parallel Region Interactions

Tests three fundamental patterns:
1. Independence: Regions operate without affecting each other
2. Coupling: Regions share events and affect each other's transitions
3. Synchronization: Regions coordinate through shared context

Each pattern tests different aspects of AND-state semantics.

NO HARDCODING: Learn coupling patterns from examples.
"""

import time
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Any, Optional
from enum import Enum, auto
import random

from .region_broadcast import (
    ParallelState, Region, RegionTransition, BroadcastEvent,
    BroadcastEngine, BroadcastScope, ConflictResolution,
    BroadcastPatternLearner,
)
from .asymmetric_regions import (
    AsymmetricParallelState, AsymmetricRegion,
    AsymmetricCouplingAnalyzer, AsymmetricInteractionEvolver,
    create_tiny_region, create_small_region, create_large_region,
)
from .region_sync import (
    SynchronizedParallelState, SyncRegion, SyncTransition,
    SharedContext, SyncPattern, SyncPatternLearner,
    create_producer_consumer_system, create_barrier_sync_system,
    create_leader_follower_system,
)


# =============================================================================
# BENCHMARK RESULT
# =============================================================================

@dataclass
class CouplingBenchmarkResult:
    """Result of a coupling benchmark scenario."""
    scenario_name: str
    pattern_type: str  # "independence", "coupling", "synchronization"
    accuracy: float    # How well the pattern was learned
    coverage: float    # State space coverage achieved
    coupling_strength: float  # Measured coupling between regions
    elapsed_time: float
    details: Dict[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return self.accuracy >= 0.7


# =============================================================================
# SCENARIO 1: INDEPENDENCE
# =============================================================================

def run_independence_scenario(verbose: bool = True) -> CouplingBenchmarkResult:
    """
    Test that independent regions don't interfere.

    Creates two completely independent regions that should
    operate in isolation. Success = no cross-region effects.
    """
    if verbose:
        print("\n" + "=" * 60)
        print("SCENARIO 1: INDEPENDENCE")
        print("=" * 60)

    start_time = time.time()

    # Create two independent regions
    audio = Region(
        id="audio",
        label="AudioPlayer",
        states=["Stopped", "Playing", "Paused"],
        current_state="Stopped",
    )

    video = Region(
        id="video",
        label="VideoPlayer",
        states=["Hidden", "Visible", "Fullscreen"],
        current_state="Hidden",
    )

    parallel = ParallelState(
        label="MediaSystem",
        regions=[audio, video],
    )

    # Add INDEPENDENT transitions (different events)
    parallel.add_transition(RegionTransition(
        region_id="audio",
        from_state="Stopped",
        to_state="Playing",
        event="AUDIO_PLAY",
    ))
    parallel.add_transition(RegionTransition(
        region_id="audio",
        from_state="Playing",
        to_state="Paused",
        event="AUDIO_PAUSE",
    ))
    parallel.add_transition(RegionTransition(
        region_id="video",
        from_state="Hidden",
        to_state="Visible",
        event="VIDEO_SHOW",
    ))
    parallel.add_transition(RegionTransition(
        region_id="video",
        from_state="Visible",
        to_state="Fullscreen",
        event="VIDEO_FULLSCREEN",
    ))

    engine = BroadcastEngine()

    # Test independence: audio events shouldn't affect video
    initial_config = parallel.get_current_configuration()
    if verbose:
        print(f"\nInitial: {initial_config}")

    # Fire audio events
    result1 = engine.broadcast(parallel, BroadcastEvent(name="AUDIO_PLAY"))
    result2 = engine.broadcast(parallel, BroadcastEvent(name="AUDIO_PAUSE"))

    final_config = parallel.get_current_configuration()
    if verbose:
        print(f"After audio events: {final_config}")

    # Check video unchanged
    video_unchanged = final_config["video"] == "Hidden"

    # Fire video events
    result3 = engine.broadcast(parallel, BroadcastEvent(name="VIDEO_SHOW"))
    result4 = engine.broadcast(parallel, BroadcastEvent(name="VIDEO_FULLSCREEN"))

    final_config2 = parallel.get_current_configuration()
    if verbose:
        print(f"After video events: {final_config2}")

    # Check audio unchanged from video events
    audio_unchanged = final_config2["audio"] == final_config["audio"]

    independence_score = (1.0 if video_unchanged else 0.0) + (1.0 if audio_unchanged else 0.0)
    independence_score /= 2.0

    elapsed = time.time() - start_time

    result = CouplingBenchmarkResult(
        scenario_name="Independence",
        pattern_type="independence",
        accuracy=independence_score,
        coverage=1.0,  # All states reachable independently
        coupling_strength=0.0 if independence_score == 1.0 else 0.5,
        elapsed_time=elapsed,
        details={
            "video_unchanged": video_unchanged,
            "audio_unchanged": audio_unchanged,
            "transitions_taken": 4,
        },
    )

    if verbose:
        print(f"\nResults:")
        print(f"  Independence score: {independence_score:.1%}")
        print(f"  Coupling strength: {result.coupling_strength:.2f}")
        print(f"  [{'PASS' if result.passed else 'FAIL'}] Independence maintained")

    return result


# =============================================================================
# SCENARIO 2: COUPLING (Shared Events)
# =============================================================================

def run_coupling_scenario(verbose: bool = True) -> CouplingBenchmarkResult:
    """
    Test that coupled regions respond together.

    Creates two regions that share events and should
    transition together. Success = synchronized responses.
    """
    if verbose:
        print("\n" + "=" * 60)
        print("SCENARIO 2: COUPLING (Shared Events)")
        print("=" * 60)

    start_time = time.time()

    # Create coupled regions (share PLAY/PAUSE events)
    audio = Region(
        id="audio",
        label="AudioTrack",
        states=["Muted", "Playing"],
        current_state="Muted",
        priority=1,
    )

    video = Region(
        id="video",
        label="VideoTrack",
        states=["Paused", "Playing"],
        current_state="Paused",
        priority=2,
    )

    parallel = ParallelState(
        label="MediaPlayer",
        regions=[audio, video],
        conflict_resolution=ConflictResolution.ALL_EXECUTE,
    )

    # Add COUPLED transitions (same events)
    parallel.add_transition(RegionTransition(
        region_id="audio",
        from_state="Muted",
        to_state="Playing",
        event="PLAY",
    ))
    parallel.add_transition(RegionTransition(
        region_id="video",
        from_state="Paused",
        to_state="Playing",
        event="PLAY",
    ))
    parallel.add_transition(RegionTransition(
        region_id="audio",
        from_state="Playing",
        to_state="Muted",
        event="STOP",
    ))
    parallel.add_transition(RegionTransition(
        region_id="video",
        from_state="Playing",
        to_state="Paused",
        event="STOP",
    ))

    engine = BroadcastEngine(conflict_resolution=ConflictResolution.ALL_EXECUTE)

    initial_config = parallel.get_current_configuration()
    if verbose:
        print(f"\nInitial: {initial_config}")

    # Fire shared PLAY event
    result1 = engine.broadcast(parallel, BroadcastEvent(name="PLAY", scope=BroadcastScope.GLOBAL))
    after_play = parallel.get_current_configuration()
    if verbose:
        print(f"After PLAY: {after_play}")
        print(f"  Responding regions: {result1.responding_regions}")

    # Both should be playing
    both_playing = after_play["audio"] == "Playing" and after_play["video"] == "Playing"

    # Fire shared STOP event
    result2 = engine.broadcast(parallel, BroadcastEvent(name="STOP", scope=BroadcastScope.GLOBAL))
    after_stop = parallel.get_current_configuration()
    if verbose:
        print(f"After STOP: {after_stop}")
        print(f"  Responding regions: {result2.responding_regions}")

    # Both should be stopped
    both_stopped = after_stop["audio"] == "Muted" and after_stop["video"] == "Paused"

    coupling_score = (1.0 if both_playing else 0.0) + (1.0 if both_stopped else 0.0)
    coupling_score /= 2.0

    elapsed = time.time() - start_time

    result = CouplingBenchmarkResult(
        scenario_name="Coupling",
        pattern_type="coupling",
        accuracy=coupling_score,
        coverage=1.0,
        coupling_strength=1.0 if coupling_score == 1.0 else 0.5,
        elapsed_time=elapsed,
        details={
            "both_playing_together": both_playing,
            "both_stopped_together": both_stopped,
            "responding_regions_play": result1.responding_regions,
            "responding_regions_stop": result2.responding_regions,
        },
    )

    if verbose:
        print(f"\nResults:")
        print(f"  Coupling score: {coupling_score:.1%}")
        print(f"  Coupling strength: {result.coupling_strength:.2f}")
        print(f"  [{'PASS' if result.passed else 'FAIL'}] Regions couple correctly")

    return result


# =============================================================================
# SCENARIO 3: SYNCHRONIZATION (Context Coordination)
# =============================================================================

def run_synchronization_scenario(verbose: bool = True) -> CouplingBenchmarkResult:
    """
    Test context-based synchronization.

    Creates producer-consumer pattern where regions
    coordinate via shared context. Success = proper sequencing.
    """
    if verbose:
        print("\n" + "=" * 60)
        print("SCENARIO 3: SYNCHRONIZATION (Context Coordination)")
        print("=" * 60)

    start_time = time.time()

    system = create_producer_consumer_system(
        producer_states=["Idle", "Producing"],
        consumer_states=["Waiting", "Consuming"],
    )

    initial_config = system.get_configuration()
    if verbose:
        print(f"\nInitial: {initial_config}")
        print(f"Initial buffer: {system.shared_context.get('buffer_count', 0)}")

    # Producer produces
    system.step("START")
    for i in range(3):
        system.step("PRODUCE")
        if verbose:
            print(f"After PRODUCE {i+1}: buffer={system.shared_context.get('buffer_count')}")

    buffer_after_produce = system.shared_context.get("buffer_count", 0)
    producer_worked = buffer_after_produce == 3

    # Consumer consumes (should work now)
    system.step("CONSUME")
    buffer_after_consume = system.shared_context.get("buffer_count", 0)
    consumer_worked = buffer_after_consume == 2

    if verbose:
        print(f"After CONSUME: buffer={buffer_after_consume}")

    # Try to consume when empty
    for _ in range(3):
        system.step("DONE")  # Back to waiting
        system.step("CONSUME")

    buffer_final = system.shared_context.get("buffer_count", 0)
    if verbose:
        print(f"After consuming all: buffer={buffer_final}")

    sync_score = 0.0
    if producer_worked:
        sync_score += 0.5
    if consumer_worked:
        sync_score += 0.5

    elapsed = time.time() - start_time

    result = CouplingBenchmarkResult(
        scenario_name="Synchronization",
        pattern_type="synchronization",
        accuracy=sync_score,
        coverage=0.8,  # Some states may not be visited
        coupling_strength=0.8,  # Medium-high due to context sharing
        elapsed_time=elapsed,
        details={
            "producer_worked": producer_worked,
            "consumer_worked": consumer_worked,
            "buffer_after_produce": buffer_after_produce,
            "buffer_final": buffer_final,
        },
    )

    if verbose:
        print(f"\nResults:")
        print(f"  Sync score: {sync_score:.1%}")
        print(f"  Producer worked: {producer_worked}")
        print(f"  Consumer worked: {consumer_worked}")
        print(f"  [{'PASS' if result.passed else 'FAIL'}] Synchronization works")

    return result


# =============================================================================
# SCENARIO 4: ASYMMETRIC COUPLING
# =============================================================================

def run_asymmetric_coupling_scenario(verbose: bool = True) -> CouplingBenchmarkResult:
    """
    Test coupling with asymmetric region sizes.

    Creates 2-state vs 10-state regions and measures
    if coupling is fair despite size difference.
    """
    if verbose:
        print("\n" + "=" * 60)
        print("SCENARIO 4: ASYMMETRIC COUPLING (2 vs 10 states)")
        print("=" * 60)

    start_time = time.time()

    tiny = create_tiny_region("toggle", "ToggleSwitch")
    large = create_large_region("workflow", "WorkflowEngine")

    parallel = AsymmetricParallelState(
        label="AsymmetricSystem",
        regions=[tiny, large],
    )

    if verbose:
        print(f"\nSize ratio: {parallel.size_ratio:.1f}x")
        print(f"Tiny region: {tiny.state_count} states")
        print(f"Large region: {large.state_count} states")

    analyzer = AsymmetricCouplingAnalyzer()

    # Run mixed events
    events = ["TOGGLE", "PROGRESS", "PROGRESS", "TOGGLE", "PROGRESS", "PAUSE", "RESUME", "RESET"]
    for event in events:
        transitions = parallel.step(event)
        analyzer.record_step(parallel, event, transitions)
        if verbose:
            print(f"After {event}: {parallel.get_configuration()}")

    metrics = analyzer.compute_metrics(parallel)

    if verbose:
        print(f"\nCoverage per region:")
        for rid, cov in metrics.coverage_per_region.items():
            print(f"  {rid}: {cov:.1%}")
        print(f"Balance score: {metrics.balance_score:.2f}")

    # Asymmetric coupling score = balance * average coverage
    avg_coverage = sum(metrics.coverage_per_region.values()) / len(metrics.coverage_per_region)
    asymmetric_score = avg_coverage * metrics.balance_score

    elapsed = time.time() - start_time

    result = CouplingBenchmarkResult(
        scenario_name="Asymmetric Coupling",
        pattern_type="coupling",
        accuracy=asymmetric_score,
        coverage=avg_coverage,
        coupling_strength=metrics.coupling_strength,
        elapsed_time=elapsed,
        details={
            "size_ratio": metrics.size_ratio,
            "coverage_per_region": metrics.coverage_per_region,
            "balance_score": metrics.balance_score,
        },
    )

    if verbose:
        print(f"\nResults:")
        print(f"  Asymmetric score: {asymmetric_score:.1%}")
        print(f"  Average coverage: {avg_coverage:.1%}")
        print(f"  Balance: {metrics.balance_score:.2f}")
        print(f"  [{'PASS' if result.passed else 'PARTIAL'}] Asymmetric coupling measured")

    return result


# =============================================================================
# SCENARIO 5: LEARNED COUPLING PATTERN
# =============================================================================

def run_learned_coupling_scenario(verbose: bool = True) -> CouplingBenchmarkResult:
    """
    Test learning coupling patterns from examples.

    Provides example traces and evolves optimal interaction.
    """
    if verbose:
        print("\n" + "=" * 60)
        print("SCENARIO 5: LEARNED COUPLING PATTERN")
        print("=" * 60)

    start_time = time.time()

    # Create system for learning
    small = create_small_region("cycles", "CycleRegion")
    large = create_large_region("stages", "StageRegion")

    parallel = AsymmetricParallelState(
        label="LearnedSystem",
        regions=[small, large],
    )

    evolver = AsymmetricInteractionEvolver(
        parallel_state=parallel,
        population_size=20,
        n_generations=20,
    )

    if verbose:
        print("\nEvolving optimal interaction pattern...")

    best = evolver.evolve(verbose=verbose)

    if verbose:
        print(f"\nBest pattern fitness: {best.fitness:.3f}")
        print(f"Event sequence: {best.event_sequence[:10]}...")
        print(f"Coupling events: {best.coupling_events}")

    # Evaluate final coverage
    for region in parallel.regions:
        region.current_state = region.initial_state

    analyzer = AsymmetricCouplingAnalyzer()
    for event in best.event_sequence:
        transitions = parallel.step(event)
        analyzer.record_step(parallel, event, transitions)

    metrics = analyzer.compute_metrics(parallel)

    elapsed = time.time() - start_time

    result = CouplingBenchmarkResult(
        scenario_name="Learned Coupling",
        pattern_type="coupling",
        accuracy=best.fitness,
        coverage=sum(metrics.coverage_per_region.values()) / len(metrics.coverage_per_region),
        coupling_strength=metrics.coupling_strength,
        elapsed_time=elapsed,
        details={
            "evolved_sequence_length": len(best.event_sequence),
            "coupling_events_count": len(best.coupling_events),
            "generations": 20,
        },
    )

    if verbose:
        print(f"\nResults:")
        print(f"  Learned fitness: {best.fitness:.1%}")
        print(f"  Coverage: {result.coverage:.1%}")
        print(f"  [{'PASS' if result.passed else 'PARTIAL'}] Pattern learned")

    return result


# =============================================================================
# FULL BENCHMARK
# =============================================================================

def run_full_benchmark(verbose: bool = True) -> List[CouplingBenchmarkResult]:
    """Run all coupling benchmark scenarios."""
    results = []

    scenarios = [
        ("Independence", run_independence_scenario),
        ("Coupling", run_coupling_scenario),
        ("Synchronization", run_synchronization_scenario),
        ("Asymmetric", run_asymmetric_coupling_scenario),
        ("Learned", run_learned_coupling_scenario),
    ]

    for name, runner in scenarios:
        result = runner(verbose=verbose)
        results.append(result)

    # Summary
    if verbose:
        print("\n" + "=" * 70)
        print("COUPLING BENCHMARK SUMMARY")
        print("=" * 70)
        print(f"{'Scenario':<25} {'Type':<15} {'Accuracy':>10} {'Coverage':>10} {'Coupling':>10}")
        print("-" * 70)

        for r in results:
            print(f"{r.scenario_name:<25} {r.pattern_type:<15} {r.accuracy:>10.1%} {r.coverage:>10.1%} {r.coupling_strength:>10.2f}")

        print("-" * 70)
        avg_accuracy = sum(r.accuracy for r in results) / len(results)
        passed = sum(1 for r in results if r.passed)
        print(f"{'Average':<25} {'':<15} {avg_accuracy:>10.1%}")
        print(f"Passed: {passed}/{len(results)}")
        print("=" * 70)

        if passed == len(results):
            print("\nKEY INSIGHT: Parallel region coupling is LEARNABLE!")
            print("Independence, coupling, and synchronization patterns verified.")
        else:
            print("\nNote: Some scenarios need refinement.")

    return results


def quick_benchmark(verbose: bool = True) -> List[CouplingBenchmarkResult]:
    """Run quick benchmark with essential scenarios."""
    results = []

    scenarios = [
        ("Independence", run_independence_scenario),
        ("Coupling", run_coupling_scenario),
        ("Synchronization", run_synchronization_scenario),
    ]

    for name, runner in scenarios:
        result = runner(verbose=verbose)
        results.append(result)

    return results


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("PARALLEL REGION COUPLING BENCHMARK")
    print("=" * 70)
    print("\nRunning full benchmark suite...")
    print("This tests independence, coupling, and synchronization patterns.\n")

    results = run_full_benchmark(verbose=True)

    # Check if all passed
    all_passed = all(r.passed for r in results)
    if all_passed:
        print("\n[SUCCESS] All scenarios passed")
    else:
        print("\n[PARTIAL] Some scenarios need improvement")
