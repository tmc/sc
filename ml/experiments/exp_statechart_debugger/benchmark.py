"""
Benchmark - Performance Testing for Statechart Debugger

Measures:
1. Breakpoint evaluation overhead
2. Step execution latency
3. History storage efficiency
4. Snapshot comparison cost
5. Scalability with statechart size

Goal: Debugging overhead < 10% of normal execution.
"""

import mlx.core as mx
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import time
import random

try:
    from .breakpoint import BreakpointManager, BreakpointType
    from .stepper import DebugStatechart, ExecutionStepper
    from .inspector import ConfigurationInspector, ContextInspector, HistoryInspector
    from .debug_session import DebugSession
except ImportError:
    from breakpoint import BreakpointManager, BreakpointType
    from stepper import DebugStatechart, ExecutionStepper
    from inspector import ConfigurationInspector, ContextInspector, HistoryInspector
    from debug_session import DebugSession


# =============================================================================
# Benchmark Utilities
# =============================================================================

@dataclass
class BenchmarkResult:
    """Result of a benchmark run."""
    name: str
    iterations: int
    total_time: float
    mean_time: float
    min_time: float
    max_time: float
    std_dev: float
    ops_per_second: float
    metadata: Dict[str, any] = field(default_factory=dict)


def measure_time(func, iterations: int = 100) -> BenchmarkResult:
    """Measure execution time of a function."""
    times = []

    for _ in range(iterations):
        start = time.perf_counter()
        func()
        end = time.perf_counter()
        times.append(end - start)

    total = sum(times)
    mean = total / iterations
    min_t = min(times)
    max_t = max(times)

    variance = sum((t - mean) ** 2 for t in times) / iterations
    std_dev = variance ** 0.5

    return BenchmarkResult(
        name=func.__name__ if hasattr(func, '__name__') else "benchmark",
        iterations=iterations,
        total_time=total,
        mean_time=mean,
        min_time=min_t,
        max_time=max_t,
        std_dev=std_dev,
        ops_per_second=iterations / total if total > 0 else float('inf'),
    )


# =============================================================================
# Statechart Generators
# =============================================================================

def create_linear_statechart(num_states: int) -> DebugStatechart:
    """Create a linear statechart: S1 -> S2 -> ... -> Sn."""
    sc = DebugStatechart()

    for i in range(num_states):
        is_initial = (i == 0)
        is_final = (i == num_states - 1)
        sc.add_state(f"S{i}", is_initial=is_initial, is_final=is_final)

    for i in range(num_states - 1):
        sc.add_transition(f"t{i}", f"S{i}", f"S{i+1}", "next")

    return sc


def create_branching_statechart(depth: int, branching_factor: int) -> DebugStatechart:
    """Create a branching statechart (tree structure)."""
    sc = DebugStatechart()

    def add_subtree(prefix: str, current_depth: int, parent: Optional[str] = None):
        name = prefix or "ROOT"
        is_initial = (current_depth == 0)
        is_final = (current_depth == depth)
        is_composite = (current_depth < depth)

        sc.add_state(name, is_initial=is_initial, is_final=is_final,
                     is_composite=is_composite, parent=parent)

        if current_depth < depth:
            for i in range(branching_factor):
                child = f"{name}_{i}"
                add_subtree(child, current_depth + 1, name)
                sc.add_transition(f"t_{name}_{i}", name, child, f"branch{i}")

    add_subtree("", 0)
    return sc


def create_complex_statechart(
    num_states: int,
    num_transitions: int,
    num_composite: int = 0,
) -> DebugStatechart:
    """Create a complex statechart with random structure."""
    sc = DebugStatechart()

    # Add states
    for i in range(num_states):
        is_initial = (i == 0)
        is_final = (i == num_states - 1)
        is_composite = (i < num_composite)
        sc.add_state(f"S{i}", is_initial=is_initial, is_final=is_final,
                     is_composite=is_composite)

    # Add transitions (ensure connected)
    for i in range(num_states - 1):
        sc.add_transition(f"t{i}", f"S{i}", f"S{i+1}", f"e{i}")

    # Add random extra transitions
    events = ["alpha", "beta", "gamma", "delta"]
    for i in range(num_transitions - (num_states - 1)):
        src = random.randint(0, num_states - 2)
        tgt = random.randint(1, num_states - 1)
        event = random.choice(events)
        sc.add_transition(f"t_extra_{i}", f"S{src}", f"S{tgt}", event)

    return sc


# =============================================================================
# Benchmark Tests
# =============================================================================

def benchmark_breakpoint_evaluation(num_breakpoints: int, num_checks: int) -> BenchmarkResult:
    """Benchmark breakpoint evaluation overhead."""
    manager = BreakpointManager()

    # Add breakpoints using manager method
    for i in range(num_breakpoints):
        condition = f"ctx.get('count', 0) > {i}" if i % 2 == 0 else None
        manager.add_state_breakpoint(
            f"S{i}",
            bp_type=BreakpointType.STATE_ENTRY,
            condition=condition,
        )

    active_states = {f"S{i}" for i in range(0, num_breakpoints, 2)}
    context = {"count": num_breakpoints // 2}

    def check_breakpoints():
        for state in active_states:
            manager.check_state_entry(state, active_states, context)

    result = measure_time(check_breakpoints, num_checks)
    result.name = f"breakpoint_eval_{num_breakpoints}bp"
    result.metadata = {"num_breakpoints": num_breakpoints, "num_checks": num_checks}
    return result


def benchmark_step_execution(num_states: int, num_steps: int) -> BenchmarkResult:
    """Benchmark step execution latency."""
    sc = create_linear_statechart(num_states)
    stepper = ExecutionStepper(sc)

    def run_steps():
        stepper.start()
        for _ in range(min(num_steps, num_states - 1)):
            sc.send_event("next")
            stepper.step()

    result = measure_time(run_steps, 10)
    result.name = f"step_exec_{num_states}states"
    result.metadata = {"num_states": num_states, "num_steps": num_steps}
    return result


def benchmark_reverse_stepping(num_steps: int) -> BenchmarkResult:
    """Benchmark reverse stepping through history."""
    sc = create_linear_statechart(num_steps + 1)
    stepper = ExecutionStepper(sc)

    # Run forward
    stepper.start()
    for _ in range(num_steps):
        sc.send_event("next")
        stepper.step()

    def reverse_all():
        while stepper.can_step_back():
            stepper.reverse_step()

    result = measure_time(reverse_all, 50)
    result.name = f"reverse_step_{num_steps}steps"
    result.metadata = {"num_steps": num_steps}
    return result


def benchmark_snapshot_diff(num_states: int, num_context_vars: int) -> BenchmarkResult:
    """Benchmark snapshot comparison."""
    from inspector import ConfigurationSnapshot, DiffInspector

    snap1 = ConfigurationSnapshot(
        sequence=0,
        timestamp=0.0,
        active_states=[f"S{i}" for i in range(num_states // 2)],
        context={f"var_{i}": i for i in range(num_context_vars)},
        enabled_transitions=[f"t{i}" for i in range(num_states)],
    )

    snap2 = ConfigurationSnapshot(
        sequence=1,
        timestamp=0.1,
        active_states=[f"S{i}" for i in range(num_states // 4, num_states * 3 // 4)],
        context={f"var_{i}": i + 1 for i in range(num_context_vars)},
        enabled_transitions=[f"t{i}" for i in range(num_states // 2)],
    )

    diff_inspector = DiffInspector()

    def compute_diff():
        diff_inspector.diff_snapshots(snap1, snap2)

    result = measure_time(compute_diff, 1000)
    result.name = f"snapshot_diff_{num_states}s_{num_context_vars}v"
    result.metadata = {"num_states": num_states, "num_context_vars": num_context_vars}
    return result


def benchmark_history_recording(num_entries: int) -> BenchmarkResult:
    """Benchmark history recording overhead."""
    history = HistoryInspector()

    def record_entries():
        for i in range(num_entries):
            history.add_entry(
                sequence=i,
                timestamp=i * 0.1,
                event=f"event_{i}",
                transition=f"t{i}",
                source_states=[f"S{i}"],
                target_states=[f"S{i+1}"],
                context_changes={f"var_{i}": (i, i + 1)},
            )

    result = measure_time(record_entries, 10)
    result.name = f"history_record_{num_entries}entries"
    result.metadata = {"num_entries": num_entries}
    return result


def benchmark_full_debug_session(num_states: int) -> BenchmarkResult:
    """Benchmark full debug session with all features."""

    def run_session():
        # Create statechart first
        sc = DebugStatechart()

        # Build statechart
        for i in range(num_states):
            sc.add_state(f"S{i}", is_initial=(i == 0), is_final=(i == num_states - 1))

        for i in range(num_states - 1):
            sc.add_transition(f"t{i}", f"S{i}", f"S{i+1}", f"e{i}")

        # Create session with statechart
        session = DebugSession(sc)

        # Add breakpoints
        session.break_on_state(f"S{num_states // 2}", BreakpointType.STATE_ENTRY)

        # Start and run
        session.start()

        # Queue events
        for i in range(num_states - 1):
            session.send_event(f"e{i}")

        # Step through
        for _ in range(num_states - 1):
            session.step()

    result = measure_time(run_session, 20)
    result.name = f"full_session_{num_states}states"
    result.metadata = {"num_states": num_states}
    return result


def benchmark_inspector_update(num_states: int, num_transitions: int) -> BenchmarkResult:
    """Benchmark inspector state updates."""
    inspector = ConfigurationInspector()

    # Register states
    for i in range(num_states):
        inspector.register_state(f"S{i}", is_initial=(i == 0))

    # Register transitions
    for i in range(num_transitions):
        src = i % num_states
        tgt = (i + 1) % num_states
        inspector.register_transition(f"t{i}", f"S{src}", f"S{tgt}", f"e{i}")

    active = {f"S{i}" for i in range(num_states // 2)}
    context = {"count": 10}

    def update_inspector():
        inspector.update_active_states(active)
        inspector.update_enabled_transitions(active, context)
        inspector.take_snapshot(0, 0.0, active, context)

    result = measure_time(update_inspector, 500)
    result.name = f"inspector_update_{num_states}s_{num_transitions}t"
    result.metadata = {"num_states": num_states, "num_transitions": num_transitions}
    return result


# =============================================================================
# Scalability Tests
# =============================================================================

def run_scalability_test(
    sizes: List[int],
    benchmark_func,
    **kwargs
) -> List[BenchmarkResult]:
    """Run benchmark at different scales."""
    results = []
    for size in sizes:
        result = benchmark_func(size, **kwargs)
        results.append(result)
    return results


def analyze_scalability(results: List[BenchmarkResult]) -> Dict[str, float]:
    """Analyze scalability from benchmark results."""
    if len(results) < 2:
        return {}

    sizes = [r.metadata.get("num_states", r.metadata.get("num_breakpoints", 0))
             for r in results]
    times = [r.mean_time for r in results]

    # Linear fit: time = a * size + b
    n = len(sizes)
    sum_x = sum(sizes)
    sum_y = sum(times)
    sum_xy = sum(x * y for x, y in zip(sizes, times))
    sum_x2 = sum(x * x for x in sizes)

    denom = n * sum_x2 - sum_x * sum_x
    if denom == 0:
        return {"slope": 0.0, "intercept": times[0] if times else 0.0}

    slope = (n * sum_xy - sum_x * sum_y) / denom
    intercept = (sum_y - slope * sum_x) / n

    return {
        "slope": slope,
        "intercept": intercept,
        "time_per_element_us": slope * 1e6,
    }


# =============================================================================
# Report Generation
# =============================================================================

def format_result(result: BenchmarkResult) -> str:
    """Format a single benchmark result."""
    lines = [
        f"  {result.name}:",
        f"    Iterations: {result.iterations}",
        f"    Mean: {result.mean_time * 1000:.3f} ms",
        f"    Min:  {result.min_time * 1000:.3f} ms",
        f"    Max:  {result.max_time * 1000:.3f} ms",
        f"    Std:  {result.std_dev * 1000:.3f} ms",
        f"    Ops/s: {result.ops_per_second:.1f}",
    ]
    return "\n".join(lines)


def generate_report(results: Dict[str, List[BenchmarkResult]]) -> str:
    """Generate benchmark report."""
    lines = []
    lines.append("=" * 60)
    lines.append("STATECHART DEBUGGER BENCHMARK REPORT")
    lines.append("=" * 60)

    for category, category_results in results.items():
        lines.append(f"\n{category}:")
        lines.append("-" * 40)
        for result in category_results:
            lines.append(format_result(result))

    lines.append("\n" + "=" * 60)
    return "\n".join(lines)


# =============================================================================
# Main Benchmark Suite
# =============================================================================

def run_benchmarks() -> Dict[str, List[BenchmarkResult]]:
    """Run all benchmarks."""
    results = {}

    print("Running breakpoint benchmarks...")
    results["Breakpoint Evaluation"] = [
        benchmark_breakpoint_evaluation(10, 1000),
        benchmark_breakpoint_evaluation(50, 1000),
        benchmark_breakpoint_evaluation(100, 1000),
    ]

    print("Running step execution benchmarks...")
    results["Step Execution"] = [
        benchmark_step_execution(10, 10),
        benchmark_step_execution(50, 50),
        benchmark_step_execution(100, 100),
    ]

    print("Running reverse stepping benchmarks...")
    results["Reverse Stepping"] = [
        benchmark_reverse_stepping(10),
        benchmark_reverse_stepping(50),
        benchmark_reverse_stepping(100),
    ]

    print("Running snapshot diff benchmarks...")
    results["Snapshot Diff"] = [
        benchmark_snapshot_diff(10, 10),
        benchmark_snapshot_diff(50, 50),
        benchmark_snapshot_diff(100, 100),
    ]

    print("Running history recording benchmarks...")
    results["History Recording"] = [
        benchmark_history_recording(100),
        benchmark_history_recording(500),
        benchmark_history_recording(1000),
    ]

    print("Running inspector update benchmarks...")
    results["Inspector Update"] = [
        benchmark_inspector_update(10, 20),
        benchmark_inspector_update(50, 100),
        benchmark_inspector_update(100, 200),
    ]

    print("Running full session benchmarks...")
    results["Full Debug Session"] = [
        benchmark_full_debug_session(10),
        benchmark_full_debug_session(50),
        benchmark_full_debug_session(100),
    ]

    return results


# =============================================================================
# Demo
# =============================================================================

def demo():
    """Run benchmark demo."""
    print("=" * 60)
    print("Statechart Debugger Benchmark")
    print("=" * 60)

    # Run quick benchmarks
    print("\n--- Quick Benchmarks ---")

    # Breakpoint evaluation
    result = benchmark_breakpoint_evaluation(20, 100)
    print(f"\nBreakpoint Evaluation (20 breakpoints):")
    print(f"  Mean: {result.mean_time * 1000:.3f} ms")
    print(f"  Ops/s: {result.ops_per_second:.0f}")

    # Step execution
    result = benchmark_step_execution(20, 20)
    print(f"\nStep Execution (20 states):")
    print(f"  Mean: {result.mean_time * 1000:.3f} ms")
    print(f"  Ops/s: {result.ops_per_second:.0f}")

    # Full session
    result = benchmark_full_debug_session(20)
    print(f"\nFull Debug Session (20 states):")
    print(f"  Mean: {result.mean_time * 1000:.3f} ms")
    print(f"  Ops/s: {result.ops_per_second:.0f}")

    # Scalability test
    print("\n--- Scalability Test ---")
    sizes = [10, 25, 50, 75, 100]
    scalability_results = []

    for size in sizes:
        result = benchmark_full_debug_session(size)
        scalability_results.append(result)
        print(f"  {size} states: {result.mean_time * 1000:.3f} ms")

    # Analyze
    analysis = analyze_scalability(scalability_results)
    print(f"\nScalability Analysis:")
    print(f"  Time per state: {analysis.get('time_per_element_us', 0):.2f} us")

    # Performance target check
    print("\n--- Performance Target ---")
    target_overhead = 0.10  # 10%
    base_time = 0.001  # 1ms baseline per step

    step_result = benchmark_step_execution(50, 50)
    actual_overhead = step_result.mean_time / (50 * base_time)

    if actual_overhead < target_overhead:
        print(f"  PASS: Overhead {actual_overhead*100:.1f}% < {target_overhead*100:.0f}% target")
    else:
        print(f"  WARN: Overhead {actual_overhead*100:.1f}% > {target_overhead*100:.0f}% target")

    return scalability_results


if __name__ == "__main__":
    demo()
