"""
Bottleneck Analyzer: Identify Performance Hotspots

Analyzes profiling data to:
1. Identify time bottlenecks (which phases are slow?)
2. Identify memory bottlenecks (which phases allocate most?)
3. Determine scaling factors (O(n), O(n²), O(n log n)?)
4. Generate actionable recommendations
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
from enum import Enum, auto
import math
import statistics

from .profiler import ProfileResult, TimeSpan


class ScalingClass(Enum):
    """Asymptotic scaling classification."""
    CONSTANT = "O(1)"
    LOGARITHMIC = "O(log n)"
    LINEAR = "O(n)"
    LINEARITHMIC = "O(n log n)"
    QUADRATIC = "O(n²)"
    CUBIC = "O(n³)"
    EXPONENTIAL = "O(2^n)"
    UNKNOWN = "Unknown"


@dataclass
class Bottleneck:
    """Identified bottleneck."""
    name: str
    category: str  # "time", "memory", "scaling"
    severity: float  # 0-1, higher is worse
    description: str
    recommendation: str
    data: Dict[str, Any] = field(default_factory=dict)

    def __str__(self):
        return f"[{self.category.upper()}] {self.name}: {self.description}"


@dataclass
class ScalingAnalysis:
    """Analysis of how an operation scales."""
    operation: str
    scaling_class: ScalingClass
    scaling_factor: float  # Actual measured factor
    r_squared: float  # Fit quality (0-1)
    data_points: List[Tuple[int, float]] = field(default_factory=list)  # (n, time)

    def is_superlinear(self) -> bool:
        """Check if scaling is worse than linear."""
        return self.scaling_class in [
            ScalingClass.QUADRATIC,
            ScalingClass.CUBIC,
            ScalingClass.EXPONENTIAL,
        ]


@dataclass
class BottleneckReport:
    """Complete bottleneck analysis report."""
    operation: str
    bottlenecks: List[Bottleneck] = field(default_factory=list)
    scaling_analyses: List[ScalingAnalysis] = field(default_factory=list)
    summary: str = ""
    total_score: float = 0.0  # 0-1, lower is better

    def get_critical_bottlenecks(self) -> List[Bottleneck]:
        """Get bottlenecks with severity > 0.7."""
        return [b for b in self.bottlenecks if b.severity > 0.7]

    def get_by_category(self, category: str) -> List[Bottleneck]:
        """Get bottlenecks by category."""
        return [b for b in self.bottlenecks if b.category == category]


class BottleneckAnalyzer:
    """
    Analyze profiling results to identify bottlenecks.
    """

    def __init__(self):
        self.results_by_scale: Dict[int, List[ProfileResult]] = {}

    def add_result(self, result: ProfileResult):
        """Add a profiling result for analysis."""
        n = result.n_states
        if n not in self.results_by_scale:
            self.results_by_scale[n] = []
        self.results_by_scale[n].append(result)

    def add_results(self, results: List[ProfileResult]):
        """Add multiple results."""
        for r in results:
            self.add_result(r)

    def analyze(self) -> BottleneckReport:
        """Perform full bottleneck analysis."""
        if not self.results_by_scale:
            return BottleneckReport(operation="none", summary="No data to analyze")

        # Get operation name from first result
        first_results = next(iter(self.results_by_scale.values()))
        operation = first_results[0].operation if first_results else "unknown"

        report = BottleneckReport(operation=operation)

        # Analyze time bottlenecks
        time_bottlenecks = self._analyze_time_bottlenecks()
        report.bottlenecks.extend(time_bottlenecks)

        # Analyze memory bottlenecks
        memory_bottlenecks = self._analyze_memory_bottlenecks()
        report.bottlenecks.extend(memory_bottlenecks)

        # Analyze scaling
        scaling = self._analyze_scaling()
        report.scaling_analyses = scaling

        # Add scaling bottlenecks
        for sa in scaling:
            if sa.is_superlinear():
                report.bottlenecks.append(Bottleneck(
                    name=f"superlinear_scaling_{sa.operation}",
                    category="scaling",
                    severity=0.8 if sa.scaling_class == ScalingClass.QUADRATIC else 0.95,
                    description=f"{sa.operation} scales as {sa.scaling_class.value}",
                    recommendation=f"Optimize {sa.operation} algorithm to improve scaling",
                    data={"scaling_class": sa.scaling_class.value, "factor": sa.scaling_factor},
                ))

        # Calculate total score
        if report.bottlenecks:
            report.total_score = sum(b.severity for b in report.bottlenecks) / len(report.bottlenecks)

        # Generate summary
        report.summary = self._generate_summary(report)

        return report

    def _analyze_time_bottlenecks(self) -> List[Bottleneck]:
        """Identify time-related bottlenecks."""
        bottlenecks = []

        for n, results in self.results_by_scale.items():
            for result in results:
                # Check if any phase dominates
                if result.time_spans:
                    total_time = result.total_time
                    if total_time > 0:
                        for span in result.time_spans:
                            if span.label == "total":
                                continue
                            ratio = span.duration / total_time
                            if ratio > 0.5:  # Phase takes > 50% of time
                                bottlenecks.append(Bottleneck(
                                    name=f"time_hotspot_{span.label}",
                                    category="time",
                                    severity=min(0.9, ratio),
                                    description=f"Phase '{span.label}' takes {ratio*100:.0f}% of total time at n={n}",
                                    recommendation=f"Optimize '{span.label}' phase for better performance",
                                    data={"phase": span.label, "ratio": ratio, "n_states": n},
                                ))

                # Check for absolute slowness
                if result.total_time > 10.0:  # > 10 seconds
                    bottlenecks.append(Bottleneck(
                        name=f"slow_operation",
                        category="time",
                        severity=min(0.95, result.total_time / 60),  # Cap at 60s
                        description=f"Operation takes {result.total_time:.1f}s at n={n}",
                        recommendation="Consider algorithmic improvements or parallelization",
                        data={"time": result.total_time, "n_states": n},
                    ))

        return bottlenecks

    def _analyze_memory_bottlenecks(self) -> List[Bottleneck]:
        """Identify memory-related bottlenecks."""
        bottlenecks = []

        for n, results in self.results_by_scale.items():
            for result in results:
                # Check for high memory per state
                if result.memory_per_state > 1000:  # > 1KB per state
                    bottlenecks.append(Bottleneck(
                        name="high_memory_per_state",
                        category="memory",
                        severity=min(0.9, result.memory_per_state / 10000),
                        description=f"Using {result.memory_per_state:.0f} bytes per state at n={n}",
                        recommendation="Reduce per-state memory footprint with more compact representation",
                        data={"bytes_per_state": result.memory_per_state, "n_states": n},
                    ))

                # Check for high peak memory
                peak_mb = result.memory_peak_mb
                if peak_mb > 1000:  # > 1GB
                    bottlenecks.append(Bottleneck(
                        name="high_peak_memory",
                        category="memory",
                        severity=min(0.95, peak_mb / 10000),
                        description=f"Peak memory {peak_mb:.0f}MB at n={n}",
                        recommendation="Implement streaming/batched processing to reduce peak memory",
                        data={"peak_mb": peak_mb, "n_states": n},
                    ))

        return bottlenecks

    def _analyze_scaling(self) -> List[ScalingAnalysis]:
        """Analyze how operations scale with input size."""
        analyses = []

        # Need at least 3 data points
        if len(self.results_by_scale) < 3:
            return analyses

        # Collect (n, time) pairs
        scales = sorted(self.results_by_scale.keys())
        times = []
        for n in scales:
            results = self.results_by_scale[n]
            avg_time = statistics.mean(r.total_time for r in results)
            times.append(avg_time)

        data_points = list(zip(scales, times))

        # Classify scaling
        scaling_class, scaling_factor, r_squared = self._classify_scaling(scales, times)

        analysis = ScalingAnalysis(
            operation="total",
            scaling_class=scaling_class,
            scaling_factor=scaling_factor,
            r_squared=r_squared,
            data_points=data_points,
        )
        analyses.append(analysis)

        return analyses

    def _classify_scaling(
        self,
        ns: List[int],
        times: List[float],
    ) -> Tuple[ScalingClass, float, float]:
        """
        Classify the scaling behavior.

        Tests different complexity classes and picks best fit.
        """
        if len(ns) < 2 or len(times) < 2:
            return ScalingClass.UNKNOWN, 0.0, 0.0

        # Avoid division by zero
        if any(t <= 0 for t in times) or any(n <= 0 for n in ns):
            return ScalingClass.UNKNOWN, 0.0, 0.0

        # Test each scaling class
        candidates = []

        # O(1): time should be constant
        if len(set(round(t, 2) for t in times)) == 1:
            candidates.append((ScalingClass.CONSTANT, 1.0, 1.0))

        # O(n): time / n should be constant
        ratios_linear = [t / n for n, t in zip(ns, times)]
        r2_linear = self._r_squared_constant(ratios_linear)
        avg_factor_linear = statistics.mean(ratios_linear)
        candidates.append((ScalingClass.LINEAR, avg_factor_linear, r2_linear))

        # O(n log n): time / (n log n) should be constant
        ratios_nlogn = [t / (n * math.log(n)) for n, t in zip(ns, times) if n > 1]
        if ratios_nlogn:
            r2_nlogn = self._r_squared_constant(ratios_nlogn)
            avg_factor_nlogn = statistics.mean(ratios_nlogn)
            candidates.append((ScalingClass.LINEARITHMIC, avg_factor_nlogn, r2_nlogn))

        # O(n²): time / n² should be constant
        ratios_quad = [t / (n * n) for n, t in zip(ns, times)]
        r2_quad = self._r_squared_constant(ratios_quad)
        avg_factor_quad = statistics.mean(ratios_quad)
        candidates.append((ScalingClass.QUADRATIC, avg_factor_quad, r2_quad))

        # O(log n): time / log(n) should be constant
        ratios_log = [t / math.log(n) for n, t in zip(ns, times) if n > 1]
        if ratios_log:
            r2_log = self._r_squared_constant(ratios_log)
            avg_factor_log = statistics.mean(ratios_log)
            candidates.append((ScalingClass.LOGARITHMIC, avg_factor_log, r2_log))

        # Pick best fit (highest R²)
        best = max(candidates, key=lambda x: x[2])
        return best

    def _r_squared_constant(self, values: List[float]) -> float:
        """Calculate R² for how constant a list of values is."""
        if len(values) < 2:
            return 0.0

        mean = statistics.mean(values)
        if mean == 0:
            return 0.0

        ss_tot = sum((v - mean) ** 2 for v in values)
        ss_res = sum((v - mean) ** 2 for v in values)  # Residuals from mean

        if ss_tot == 0:
            return 1.0

        # For "constantness", we want low variance relative to mean
        cv = statistics.stdev(values) / mean if mean != 0 else 0
        return max(0, 1 - cv)

    def _generate_summary(self, report: BottleneckReport) -> str:
        """Generate human-readable summary."""
        lines = []

        lines.append(f"Bottleneck Analysis for: {report.operation}")
        lines.append(f"Total bottlenecks found: {len(report.bottlenecks)}")
        lines.append(f"Overall score: {report.total_score:.2f} (lower is better)")

        critical = report.get_critical_bottlenecks()
        if critical:
            lines.append(f"\nCRITICAL ISSUES ({len(critical)}):")
            for b in critical:
                lines.append(f"  - {b.name}: {b.description}")

        if report.scaling_analyses:
            lines.append(f"\nScaling Analysis:")
            for sa in report.scaling_analyses:
                status = "⚠️ CONCERN" if sa.is_superlinear() else "✓ OK"
                lines.append(f"  {sa.operation}: {sa.scaling_class.value} (R²={sa.r_squared:.2f}) {status}")

        return "\n".join(lines)


def analyze_bottlenecks(results: List[ProfileResult]) -> BottleneckReport:
    """Convenience function to analyze a list of results."""
    analyzer = BottleneckAnalyzer()
    analyzer.add_results(results)
    return analyzer.analyze()


def demo():
    """Demonstrate bottleneck analysis."""
    print("=" * 60)
    print("BOTTLENECK ANALYZER: Find Performance Hotspots")
    print("=" * 60)

    from .profiler import ProfileResult, TimeSpan

    # Create synthetic profiling results
    results = []

    for n in [100, 1000, 10000]:
        # Simulate O(n²) scaling
        time_total = (n / 1000) ** 2

        result = ProfileResult(
            operation="test_operation",
            n_states=n,
            total_time=time_total,
            time_spans=[
                TimeSpan(label="total", start_time=0, end_time=time_total),
                TimeSpan(label="phase_a", start_time=0, end_time=time_total * 0.7),
                TimeSpan(label="phase_b", start_time=0, end_time=time_total * 0.3),
            ],
            memory_before=0,
            memory_after=n * 500,  # 500 bytes per state
            memory_peak=n * 800,
        )
        results.append(result)

    # Analyze
    report = analyze_bottlenecks(results)

    print(f"\n{report.summary}")

    print("\n--- All Bottlenecks ---")
    for b in report.bottlenecks:
        print(f"  [{b.severity:.2f}] {b.category}: {b.name}")
        print(f"       {b.description}")
        print(f"       Recommendation: {b.recommendation}")

    return report


if __name__ == "__main__":
    demo()
