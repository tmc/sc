"""
Benchmark for SC Circuit Discovery

Evaluates circuit discovery on different model configurations
and statechart generation tasks.

METRICS:
- Circuit consistency: Do same components appear across runs?
- Specificity: Are circuits specific to validity aspects?
- Coverage: What % of validity is explained by circuits?
- Interpretability: Can we understand circuit function?
"""

import json
import time
from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Any, Tuple
from collections import defaultdict

from .validity_measurer import ValidityMeasurer, ValidityMetrics
from .ablation_runner import (
    AblationRunner,
    AblationStudy,
    ComponentSpec,
    AblationType,
    default_parser,
)
from .circuit_finder import (
    CircuitFinder,
    CircuitGraph,
    CircuitType,
    Circuit,
    summarize_circuits,
)


@dataclass
class BenchmarkResult:
    """Result for a single benchmark configuration."""
    name: str
    n_prompts: int
    n_components_ablated: int
    duration: float

    # Circuit discovery results
    n_critical_components: int
    circuits_found: Dict[str, int]  # circuit_type -> n_nodes

    # Validity metrics
    baseline_validity: float
    min_ablated_validity: float
    max_validity_drop: float

    # Circuit quality
    circuit_specificity: float  # Do circuits specialize?
    circuit_coverage: float  # How much validity explained?

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "n_prompts": self.n_prompts,
            "n_components_ablated": self.n_components_ablated,
            "duration": self.duration,
            "n_critical_components": self.n_critical_components,
            "circuits_found": self.circuits_found,
            "baseline_validity": self.baseline_validity,
            "max_validity_drop": self.max_validity_drop,
            "circuit_specificity": self.circuit_specificity,
            "circuit_coverage": self.circuit_coverage,
        }


@dataclass
class BenchmarkSummary:
    """Summary of all benchmark results."""
    results: List[BenchmarkResult] = field(default_factory=list)
    total_duration: float = 0.0

    @property
    def mean_specificity(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.circuit_specificity for r in self.results) / len(self.results)

    @property
    def mean_coverage(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.circuit_coverage for r in self.results) / len(self.results)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_benchmarks": len(self.results),
            "total_duration": self.total_duration,
            "mean_specificity": self.mean_specificity,
            "mean_coverage": self.mean_coverage,
            "results": [r.to_dict() for r in self.results],
        }


# =============================================================================
# Benchmark prompts
# =============================================================================

BENCHMARK_PROMPTS = {
    "simple": [
        "Generate a simple on/off switch statechart with two states.",
        "Create a traffic light with red, yellow, green states.",
        "Design a door state machine: open, closed, locked.",
    ],
    "complex": [
        "Generate a media player statechart with stopped, playing, paused states and forward/back controls.",
        "Create a login flow with logged_out, authenticating, logged_in, error states.",
        "Design an e-commerce order process: cart, checkout, payment, shipping, delivered.",
    ],
    "hierarchical": [
        "Generate a hierarchical statechart for a phone with active/idle states, where active has calling and messaging substates.",
        "Create a game state machine with menu, playing (with running/paused substates), and game_over.",
        "Design a multi-step form wizard with step1, step2, step3 states.",
    ],
}


# =============================================================================
# Benchmark runner
# =============================================================================

class CircuitBenchmark:
    """Benchmark for circuit discovery."""

    def __init__(
        self,
        runner: Optional[AblationRunner] = None,
        finder: Optional[CircuitFinder] = None,
        verbose: bool = True,
    ):
        self.runner = runner or AblationRunner(verbose=False)
        self.finder = finder or CircuitFinder(runner=self.runner, verbose=False)
        self.verbose = verbose

    def run_benchmark(
        self,
        name: str,
        prompts: List[str],
        layers: Optional[List[int]] = None,
        include_heads: bool = False,
    ) -> Tuple[BenchmarkResult, CircuitGraph]:
        """Run a single benchmark configuration."""
        start = time.time()

        # Enumerate components
        components = self.runner.enumerate_components(
            include_attention=True,
            include_mlp=True,
            include_heads=include_heads,
            layers=layers,
        )

        if self.verbose:
            print(f"\n{name}: {len(prompts)} prompts, {len(components)} components")

        # Run ablation study
        study = self.runner.run_study(
            prompts=prompts,
            parse_fn=default_parser,
            components=components,
        )

        # Find circuits
        graph = self.finder.find_circuits(study)

        # Compute metrics
        duration = time.time() - start

        # Baseline validity
        baseline_validity = 0.0
        min_validity = 1.0
        max_drop = 0.0

        for result in study.results:
            if baseline_validity == 0:
                baseline_validity = result.baseline_metrics.total_score
            ablated = result.ablated_metrics.total_score
            min_validity = min(min_validity, ablated)
            max_drop = max(max_drop, result.validity_drop)

        # Circuit metrics
        circuits_found = {
            ct.value: len(c.nodes) for ct, c in graph.circuits.items()
        }

        specificity = self._compute_specificity(graph)
        coverage = self._compute_coverage(graph, study)

        result = BenchmarkResult(
            name=name,
            n_prompts=len(prompts),
            n_components_ablated=len(components),
            duration=duration,
            n_critical_components=len(graph.all_nodes),
            circuits_found=circuits_found,
            baseline_validity=baseline_validity,
            min_ablated_validity=min_validity,
            max_validity_drop=max_drop,
            circuit_specificity=specificity,
            circuit_coverage=coverage,
        )

        return result, graph

    def _compute_specificity(self, graph: CircuitGraph) -> float:
        """
        Compute how specific circuits are.

        High specificity = components belong to few circuits.
        Low specificity = components are in many circuits.
        """
        if not graph.all_nodes:
            return 0.0

        specificities = []
        for node in graph.all_nodes.values():
            n_circuits = len(node.circuit_types)
            total_circuits = len(CircuitType)
            # Specificity = 1 - (n_circuits / total) for each node
            spec = 1.0 - (n_circuits / total_circuits) if total_circuits > 0 else 0
            specificities.append(spec)

        return sum(specificities) / len(specificities)

    def _compute_coverage(
        self,
        graph: CircuitGraph,
        study: AblationStudy,
    ) -> float:
        """
        Compute how much of validity is explained by circuits.

        Coverage = sum(importance of circuit nodes) / sum(all importance)
        """
        total_importance = sum(
            r.validity_drop for r in study.results if r.validity_drop > 0
        )

        if total_importance == 0:
            return 0.0

        circuit_importance = sum(
            n.importance for n in graph.all_nodes.values()
        )

        return min(1.0, circuit_importance / total_importance)

    def run_full_benchmark(self) -> BenchmarkSummary:
        """Run complete benchmark suite."""
        summary = BenchmarkSummary()
        start = time.time()

        # Different layer sampling strategies
        all_layers = list(range(self.runner.n_layers))
        layer_configs = {
            "all_layers": all_layers,
            "early": all_layers[:8],
            "middle": all_layers[8:16],
            "late": all_layers[16:],
            "sampled": all_layers[::3],  # Every 3rd layer
        }

        for prompt_type, prompts in BENCHMARK_PROMPTS.items():
            for layer_name, layers in layer_configs.items():
                name = f"{prompt_type}_{layer_name}"

                if self.verbose:
                    print(f"\nRunning: {name}")

                result, graph = self.run_benchmark(
                    name=name,
                    prompts=prompts,
                    layers=layers,
                )

                summary.results.append(result)

                if self.verbose:
                    print(f"  Circuits found: {result.circuits_found}")
                    print(f"  Specificity: {result.circuit_specificity:.2%}")
                    print(f"  Coverage: {result.circuit_coverage:.2%}")

        summary.total_duration = time.time() - start
        return summary


# =============================================================================
# Demo
# =============================================================================

def demo():
    """Run benchmark demonstration."""
    print("=" * 60)
    print("SC Circuit Discovery Benchmark")
    print("=" * 60)

    benchmark = CircuitBenchmark(verbose=True)

    # Run a subset for demo
    print("\nRunning single benchmark configuration...")

    result, graph = benchmark.run_benchmark(
        name="demo_simple",
        prompts=BENCHMARK_PROMPTS["simple"],
        layers=list(range(0, 24, 4)),  # Sample every 4th layer
    )

    print("\n" + "=" * 60)
    print("BENCHMARK RESULT")
    print("=" * 60)

    print(f"\nConfiguration: {result.name}")
    print(f"  Prompts: {result.n_prompts}")
    print(f"  Components ablated: {result.n_components_ablated}")
    print(f"  Duration: {result.duration:.2f}s")

    print(f"\nValidity:")
    print(f"  Baseline: {result.baseline_validity:.2%}")
    print(f"  Min after ablation: {result.min_ablated_validity:.2%}")
    print(f"  Max drop: {result.max_validity_drop:.2%}")

    print(f"\nCircuits found:")
    for ct, n in result.circuits_found.items():
        print(f"  {ct}: {n} nodes")

    print(f"\nCircuit quality:")
    print(f"  Specificity: {result.circuit_specificity:.2%}")
    print(f"  Coverage: {result.circuit_coverage:.2%}")

    # Print circuit summary
    print("\n" + summarize_circuits(graph))

    # Print Mermaid diagram
    print("\n" + "-" * 40)
    print("MERMAID DIAGRAM (for visualization):")
    print("-" * 40)
    print(graph.to_mermaid())

    return result, graph


if __name__ == "__main__":
    demo()
