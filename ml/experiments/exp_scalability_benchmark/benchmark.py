"""
Benchmark: Comprehensive Scalability Testing

Tests all major operations at each scale level:
- 10 / 100 / 1K / 10K / 100K / 1M states

Operations tested:
1. Generation - Create statecharts at each scale
2. Traversal - BFS, DFS, reachability analysis
3. Mutation - Topology evolution mutations
4. Crossover - Combine two statecharts
5. Guard evaluation - Check guards on transitions
6. Serialization - Convert to/from dict
7. Configuration - Track active states
"""

from dataclasses import dataclass, field
from typing import List, Dict, Set, Tuple, Optional, Any, Callable
import time
import gc
import json
import random

from .scale_generator import (
    ScaleConfig,
    StatechartGenerator,
    GeneratedStatechart,
    GeneratedState,
    GeneratedTransition,
    Topology,
    SCALE_LEVELS,
    generate_at_scale,
)
from .profiler import (
    Profiler,
    ProfileResult,
    MemoryTracker,
    TimeTracker,
)
from .bottleneck_analyzer import (
    BottleneckAnalyzer,
    BottleneckReport,
    analyze_bottlenecks,
)


@dataclass
class BenchmarkResult:
    """Result from benchmarking an operation."""
    operation: str
    scale: int
    topology: str
    time_ms: float
    memory_mb: float
    throughput: float  # States per second
    success: bool = True
    error: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ScalabilityReport:
    """Complete scalability benchmark report."""
    results: List[BenchmarkResult] = field(default_factory=list)
    bottleneck_report: Optional[BottleneckReport] = None
    summary: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def get_results_for_operation(self, operation: str) -> List[BenchmarkResult]:
        """Get results for a specific operation."""
        return [r for r in self.results if r.operation == operation]

    def get_results_for_scale(self, scale: int) -> List[BenchmarkResult]:
        """Get results for a specific scale."""
        return [r for r in self.results if r.scale == scale]


class ScalabilityBenchmark:
    """
    Comprehensive scalability benchmark suite.

    Tests operations at all scale levels and identifies bottlenecks.
    """

    def __init__(
        self,
        scales: List[int] = None,
        topologies: List[Topology] = None,
        max_scale: int = 100_000,  # Default max for safety
    ):
        self.scales = scales or [s for s in SCALE_LEVELS if s <= max_scale]
        self.topologies = topologies or [Topology.BALANCED]
        self.results: List[BenchmarkResult] = []
        self.profile_results: List[ProfileResult] = []

    def _benchmark_operation(
        self,
        operation_name: str,
        func: Callable,
        scale: int,
        topology: Topology,
        **kwargs,
    ) -> BenchmarkResult:
        """Benchmark a single operation."""
        gc.collect()

        profiler = Profiler(operation_name)
        profiler.n_states = scale
        profiler.start()

        try:
            func(**kwargs)
            profile = profiler.stop()
            self.profile_results.append(profile)

            result = BenchmarkResult(
                operation=operation_name,
                scale=scale,
                topology=topology.value,
                time_ms=profile.time_ms,
                memory_mb=profile.memory_delta_mb,
                throughput=profile.throughput,
                success=True,
            )
        except Exception as e:
            profiler.stop()
            result = BenchmarkResult(
                operation=operation_name,
                scale=scale,
                topology=topology.value,
                time_ms=0,
                memory_mb=0,
                throughput=0,
                success=False,
                error=str(e),
            )

        self.results.append(result)
        return result

    def benchmark_generation(
        self,
        scale: int,
        topology: Topology,
    ) -> Tuple[BenchmarkResult, Optional[GeneratedStatechart]]:
        """Benchmark statechart generation."""
        sc = [None]  # Use list to capture result

        def generate():
            sc[0] = generate_at_scale(scale, topology)

        result = self._benchmark_operation(
            "generation",
            generate,
            scale,
            topology,
        )

        return result, sc[0]

    def benchmark_traversal(
        self,
        sc: GeneratedStatechart,
        scale: int,
        topology: Topology,
    ) -> BenchmarkResult:
        """Benchmark state traversal (BFS)."""

        def traverse():
            visited = set()
            queue = [sc.root_id]

            while queue:
                state_id = queue.pop(0)
                if state_id in visited:
                    continue
                visited.add(state_id)

                state = sc.states.get(state_id)
                if state:
                    queue.extend(state.children)

            return len(visited)

        return self._benchmark_operation(
            "traversal_bfs",
            traverse,
            scale,
            topology,
        )

    def benchmark_reachability(
        self,
        sc: GeneratedStatechart,
        scale: int,
        topology: Topology,
    ) -> BenchmarkResult:
        """Benchmark reachability analysis via transitions."""

        def analyze_reachability():
            # Build adjacency from transitions
            adj: Dict[int, Set[int]] = {}
            for trans in sc.transitions:
                if trans.source_id not in adj:
                    adj[trans.source_id] = set()
                adj[trans.source_id].add(trans.target_id)

            # BFS from root
            reachable = set()
            queue = [sc.root_id]

            while queue:
                state_id = queue.pop(0)
                if state_id in reachable:
                    continue
                reachable.add(state_id)

                for target in adj.get(state_id, []):
                    if target not in reachable:
                        queue.append(target)

            return len(reachable)

        return self._benchmark_operation(
            "reachability",
            analyze_reachability,
            scale,
            topology,
        )

    def benchmark_mutation(
        self,
        sc: GeneratedStatechart,
        scale: int,
        topology: Topology,
        n_mutations: int = 100,
    ) -> BenchmarkResult:
        """Benchmark topology mutations."""

        def mutate():
            mutations_done = 0
            leaf_states = sc.get_leaf_states()

            for _ in range(n_mutations):
                mutation_type = random.choice(["add_state", "add_transition", "remove_transition"])

                if mutation_type == "add_state":
                    # Add a new state
                    parent = random.choice(list(sc.states.values()))
                    new_id = max(sc.states.keys()) + 1
                    new_state = GeneratedState(
                        id=new_id,
                        name=f"s{new_id}",
                        parent_id=parent.id,
                        depth=parent.depth + 1,
                    )
                    sc.states[new_id] = new_state
                    parent.children.append(new_id)
                    mutations_done += 1

                elif mutation_type == "add_transition" and leaf_states:
                    src = random.choice(leaf_states)
                    tgt = random.choice(leaf_states)
                    new_trans = GeneratedTransition(
                        id=len(sc.transitions),
                        source_id=src.id,
                        target_id=tgt.id,
                    )
                    sc.transitions.append(new_trans)
                    mutations_done += 1

                elif mutation_type == "remove_transition" and sc.transitions:
                    idx = random.randint(0, len(sc.transitions) - 1)
                    sc.transitions.pop(idx)
                    mutations_done += 1

            return mutations_done

        return self._benchmark_operation(
            f"mutation_x{n_mutations}",
            mutate,
            scale,
            topology,
        )

    def benchmark_guard_evaluation(
        self,
        sc: GeneratedStatechart,
        scale: int,
        topology: Topology,
        n_evaluations: int = 1000,
    ) -> BenchmarkResult:
        """Benchmark guard expression evaluation."""

        def evaluate_guards():
            context = {"x": 50, "y": 25, "count": 10, "timer": 100}
            evaluations = 0

            for _ in range(n_evaluations):
                for trans in sc.transitions[:100]:  # Sample 100 transitions
                    if trans.guard:
                        # Simple guard evaluation simulation
                        guard = trans.guard
                        result = True

                        if "<" in guard:
                            parts = guard.split("<")
                            var = parts[0].strip()
                            val = int(parts[1].strip())
                            result = context.get(var, 0) < val
                        elif ">" in guard:
                            parts = guard.split(">")
                            var = parts[0].strip()
                            val = int(parts[1].strip())
                            result = context.get(var, 0) > val

                        evaluations += 1

            return evaluations

        return self._benchmark_operation(
            f"guard_eval_x{n_evaluations}",
            evaluate_guards,
            scale,
            topology,
        )

    def benchmark_serialization(
        self,
        sc: GeneratedStatechart,
        scale: int,
        topology: Topology,
    ) -> BenchmarkResult:
        """Benchmark serialization to/from dict."""

        def serialize_deserialize():
            # Serialize
            data = {
                "name": sc.name,
                "n_states": sc.n_states,
                "states": {
                    str(sid): {
                        "id": s.id,
                        "name": s.name,
                        "parent_id": s.parent_id,
                        "children": s.children,
                        "depth": s.depth,
                    }
                    for sid, s in sc.states.items()
                },
                "transitions": [
                    {"id": t.id, "source": t.source_id, "target": t.target_id, "guard": t.guard}
                    for t in sc.transitions
                ],
            }

            # Serialize to JSON string
            json_str = json.dumps(data)

            # Deserialize back
            parsed = json.loads(json_str)

            return len(json_str)

        return self._benchmark_operation(
            "serialization",
            serialize_deserialize,
            scale,
            topology,
        )

    def benchmark_configuration(
        self,
        sc: GeneratedStatechart,
        scale: int,
        topology: Topology,
        n_steps: int = 100,
    ) -> BenchmarkResult:
        """Benchmark configuration tracking (active state management)."""

        def simulate_configuration():
            # Start with initial states
            active = {sc.root_id}
            for s in sc.states.values():
                if s.is_initial:
                    active.add(s.id)

            steps = 0
            for _ in range(n_steps):
                # Find enabled transitions
                enabled = [
                    t for t in sc.transitions
                    if t.source_id in active
                ]

                if enabled:
                    # Take a random transition
                    trans = random.choice(enabled)

                    # Update active set
                    active.discard(trans.source_id)
                    active.add(trans.target_id)
                    steps += 1

            return steps

        return self._benchmark_operation(
            f"configuration_x{n_steps}",
            simulate_configuration,
            scale,
            topology,
        )

    def run_all(self, verbose: bool = True) -> ScalabilityReport:
        """Run all benchmarks at all scales."""
        if verbose:
            print("=" * 70)
            print("SCALABILITY BENCHMARK")
            print("=" * 70)
            print(f"Scales: {self.scales}")
            print(f"Topologies: {[t.value for t in self.topologies]}")

        for topology in self.topologies:
            if verbose:
                print(f"\n{'='*50}")
                print(f"Topology: {topology.value.upper()}")
                print("=" * 50)

            for scale in self.scales:
                if verbose:
                    print(f"\n--- Scale: {scale:,} states ---")

                gc.collect()

                # Generation
                gen_result, sc = self.benchmark_generation(scale, topology)
                if verbose:
                    print(f"  Generation: {gen_result.time_ms:.2f}ms, "
                          f"{gen_result.memory_mb:.2f}MB")

                if not gen_result.success or sc is None:
                    if verbose:
                        print(f"  FAILED: {gen_result.error}")
                    continue

                # Traversal
                trav_result = self.benchmark_traversal(sc, scale, topology)
                if verbose:
                    print(f"  Traversal: {trav_result.time_ms:.2f}ms")

                # Reachability
                reach_result = self.benchmark_reachability(sc, scale, topology)
                if verbose:
                    print(f"  Reachability: {reach_result.time_ms:.2f}ms")

                # Mutation
                mut_result = self.benchmark_mutation(sc, scale, topology)
                if verbose:
                    print(f"  Mutation (100x): {mut_result.time_ms:.2f}ms")

                # Guard evaluation
                guard_result = self.benchmark_guard_evaluation(sc, scale, topology)
                if verbose:
                    print(f"  Guard Eval (1000x): {guard_result.time_ms:.2f}ms")

                # Serialization
                ser_result = self.benchmark_serialization(sc, scale, topology)
                if verbose:
                    print(f"  Serialization: {ser_result.time_ms:.2f}ms")

                # Configuration
                conf_result = self.benchmark_configuration(sc, scale, topology)
                if verbose:
                    print(f"  Configuration (100x): {conf_result.time_ms:.2f}ms")

                # Memory cleanup
                del sc
                gc.collect()

        # Analyze bottlenecks
        bottleneck_report = analyze_bottlenecks(self.profile_results)

        # Build report
        report = ScalabilityReport(
            results=self.results,
            bottleneck_report=bottleneck_report,
        )

        # Summary
        report.summary = self._build_summary()

        if verbose:
            print("\n" + "=" * 70)
            print("BOTTLENECK ANALYSIS")
            print("=" * 70)
            print(bottleneck_report.summary)

            print("\n" + "=" * 70)
            print("SUMMARY")
            print("=" * 70)
            for key, value in report.summary.items():
                print(f"  {key}: {value}")

        return report

    def _build_summary(self) -> Dict[str, Any]:
        """Build summary statistics."""
        summary = {}

        # Group by operation
        operations = set(r.operation for r in self.results)
        for op in operations:
            op_results = [r for r in self.results if r.operation == op and r.success]
            if op_results:
                avg_time = sum(r.time_ms for r in op_results) / len(op_results)
                max_time = max(r.time_ms for r in op_results)
                summary[f"{op}_avg_ms"] = round(avg_time, 2)
                summary[f"{op}_max_ms"] = round(max_time, 2)

        # Overall stats
        successful = [r for r in self.results if r.success]
        summary["total_benchmarks"] = len(self.results)
        summary["successful"] = len(successful)
        summary["failed"] = len(self.results) - len(successful)
        summary["max_scale_tested"] = max(r.scale for r in successful) if successful else 0

        return summary


def run_benchmark(
    max_scale: int = 10_000,
    topologies: List[Topology] = None,
    verbose: bool = True,
) -> ScalabilityReport:
    """Run the full scalability benchmark."""
    benchmark = ScalabilityBenchmark(
        max_scale=max_scale,
        topologies=topologies or [Topology.BALANCED, Topology.FLAT],
    )
    return benchmark.run_all(verbose=verbose)


def demo():
    """Quick demo of benchmarking."""
    print("Running scalability benchmark demo...")
    print("(Limited to 1K states for demo)")

    report = run_benchmark(max_scale=1000, verbose=True)

    print(f"\n\nCompleted {report.summary['total_benchmarks']} benchmarks")
    return report


if __name__ == "__main__":
    demo()
