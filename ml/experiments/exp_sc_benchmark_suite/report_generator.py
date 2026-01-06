"""
Report Generator: Markdown Comparison Reports.

Generates comprehensive benchmark reports including:
1. Summary statistics
2. Method comparison tables
3. Complexity breakdown
4. Feature analysis
5. Performance metrics
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime
from statistics import mean, stdev

from .test_suite import Complexity, Feature, TestCase
from .method_runner import GenerationMethod, MethodResult


@dataclass
class MethodStats:
    """Statistics for a single method."""
    method: GenerationMethod
    total_tests: int = 0
    valid_count: int = 0
    validity_rate: float = 0.0

    avg_states: float = 0.0
    avg_transitions: float = 0.0
    avg_depth: float = 0.0

    avg_latency_ms: float = 0.0
    min_latency_ms: float = 0.0
    max_latency_ms: float = 0.0

    avg_tokens: float = 0.0
    avg_tokens_per_state: float = 0.0
    avg_tokens_per_transition: float = 0.0

    parallel_count: int = 0
    history_count: int = 0


@dataclass
class ComplexityStats:
    """Statistics by complexity level."""
    complexity: Complexity
    total_tests: int = 0
    by_method: Dict[GenerationMethod, MethodStats] = field(default_factory=dict)


class ReportGenerator:
    """
    Generate markdown benchmark reports.
    """

    def __init__(self):
        self.results: Dict[str, Dict[GenerationMethod, MethodResult]] = {}
        self.test_cases: Dict[str, TestCase] = {}
        self.method_stats: Dict[GenerationMethod, MethodStats] = {}
        self.complexity_stats: Dict[Complexity, ComplexityStats] = {}

    def add_result(
        self,
        test_case: TestCase,
        method: GenerationMethod,
        result: MethodResult
    ):
        """Add a benchmark result."""
        if test_case.id not in self.results:
            self.results[test_case.id] = {}
            self.test_cases[test_case.id] = test_case

        self.results[test_case.id][method] = result

    def compute_statistics(self):
        """Compute all statistics from results."""
        # Initialize method stats
        for method in GenerationMethod:
            self.method_stats[method] = MethodStats(method=method)

        # Initialize complexity stats
        for complexity in Complexity:
            self.complexity_stats[complexity] = ComplexityStats(complexity=complexity)
            for method in GenerationMethod:
                self.complexity_stats[complexity].by_method[method] = MethodStats(method=method)

        # Collect results by method
        method_results: Dict[GenerationMethod, List[MethodResult]] = {
            m: [] for m in GenerationMethod
        }

        for test_id, method_results_dict in self.results.items():
            test_case = self.test_cases[test_id]

            for method, result in method_results_dict.items():
                method_results[method].append(result)

                # Update complexity stats
                cs = self.complexity_stats[test_case.complexity]
                cs.total_tests += 1
                ms = cs.by_method[method]
                ms.total_tests += 1
                if result.is_valid:
                    ms.valid_count += 1

        # Compute method statistics
        for method, results in method_results.items():
            if not results:
                continue

            stats = self.method_stats[method]
            stats.total_tests = len(results)
            stats.valid_count = sum(1 for r in results if r.is_valid)
            stats.validity_rate = stats.valid_count / stats.total_tests

            valid_results = [r for r in results if r.is_valid]
            if valid_results:
                stats.avg_states = mean(r.state_count for r in valid_results)
                stats.avg_transitions = mean(r.transition_count for r in valid_results)
                stats.avg_depth = mean(r.max_depth for r in valid_results)
                stats.parallel_count = sum(1 for r in valid_results if r.has_parallel)
                stats.history_count = sum(1 for r in valid_results if r.has_history)

            latencies = [r.generation_time * 1000 for r in results]
            stats.avg_latency_ms = mean(latencies)
            stats.min_latency_ms = min(latencies)
            stats.max_latency_ms = max(latencies)

            stats.avg_tokens = mean(r.token_count for r in results)

            if valid_results:
                tps = [r.tokens_per_state for r in valid_results if r.tokens_per_state > 0]
                tpt = [r.tokens_per_transition for r in valid_results if r.tokens_per_transition > 0]
                if tps:
                    stats.avg_tokens_per_state = mean(tps)
                if tpt:
                    stats.avg_tokens_per_transition = mean(tpt)

        # Compute complexity stats validity rates
        for complexity, cs in self.complexity_stats.items():
            for method, ms in cs.by_method.items():
                if ms.total_tests > 0:
                    ms.validity_rate = ms.valid_count / ms.total_tests

    def generate_report(self, output_path: Optional[str] = None) -> str:
        """Generate full markdown report."""
        self.compute_statistics()

        lines = []

        # Header
        lines.append("# SC Benchmark Suite Report")
        lines.append("")
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")

        # Executive Summary
        lines.append("## Executive Summary")
        lines.append("")
        lines.append(self._generate_summary())
        lines.append("")

        # Method Comparison Table
        lines.append("## Method Comparison")
        lines.append("")
        lines.append(self._generate_comparison_table())
        lines.append("")

        # Validity by Complexity
        lines.append("## Validity by Complexity")
        lines.append("")
        lines.append(self._generate_complexity_table())
        lines.append("")

        # Performance Metrics
        lines.append("## Performance Metrics")
        lines.append("")
        lines.append(self._generate_performance_table())
        lines.append("")

        # Efficiency Metrics
        lines.append("## Efficiency Metrics")
        lines.append("")
        lines.append(self._generate_efficiency_table())
        lines.append("")

        # Feature Support
        lines.append("## Feature Support")
        lines.append("")
        lines.append(self._generate_feature_table())
        lines.append("")

        # Detailed Results
        lines.append("## Detailed Results")
        lines.append("")
        lines.append(self._generate_detailed_results())
        lines.append("")

        # Recommendations
        lines.append("## Recommendations")
        lines.append("")
        lines.append(self._generate_recommendations())

        report = "\n".join(lines)

        if output_path:
            with open(output_path, 'w') as f:
                f.write(report)

        return report

    def _generate_summary(self) -> str:
        """Generate executive summary."""
        total_tests = len(self.results)
        methods_tested = len(self.method_stats)

        # Find best method
        best_method = max(
            self.method_stats.values(),
            key=lambda s: s.validity_rate
        )

        lines = [
            f"- **Total Test Cases**: {total_tests}",
            f"- **Methods Compared**: {methods_tested}",
            f"- **Best Overall Method**: {best_method.method.name} ({best_method.validity_rate:.1%} validity)",
        ]

        # Add complexity breakdown
        lines.append("")
        lines.append("### Test Distribution")
        for complexity in Complexity:
            count = sum(
                1 for tc in self.test_cases.values()
                if tc.complexity == complexity
            )
            lines.append(f"- {complexity.name}: {count} tests")

        return "\n".join(lines)

    def _generate_comparison_table(self) -> str:
        """Generate method comparison table."""
        headers = ["Method", "Valid", "Total", "Validity %", "Avg States", "Avg Trans"]

        rows = []
        for method in GenerationMethod:
            stats = self.method_stats[method]
            rows.append([
                method.name,
                str(stats.valid_count),
                str(stats.total_tests),
                f"{stats.validity_rate:.1%}",
                f"{stats.avg_states:.1f}",
                f"{stats.avg_transitions:.1f}",
            ])

        return self._format_table(headers, rows)

    def _generate_complexity_table(self) -> str:
        """Generate validity by complexity table."""
        headers = ["Complexity"] + [m.name for m in GenerationMethod]

        rows = []
        for complexity in Complexity:
            row = [complexity.name]
            cs = self.complexity_stats[complexity]
            for method in GenerationMethod:
                ms = cs.by_method[method]
                if ms.total_tests > 0:
                    row.append(f"{ms.validity_rate:.1%}")
                else:
                    row.append("-")
            rows.append(row)

        return self._format_table(headers, rows)

    def _generate_performance_table(self) -> str:
        """Generate performance metrics table."""
        headers = ["Method", "Avg Latency (ms)", "Min", "Max", "Tokens/sec"]

        rows = []
        for method in GenerationMethod:
            stats = self.method_stats[method]
            tokens_per_sec = stats.avg_tokens / (stats.avg_latency_ms / 1000) if stats.avg_latency_ms > 0 else 0
            rows.append([
                method.name,
                f"{stats.avg_latency_ms:.2f}",
                f"{stats.min_latency_ms:.2f}",
                f"{stats.max_latency_ms:.2f}",
                f"{tokens_per_sec:.0f}",
            ])

        return self._format_table(headers, rows)

    def _generate_efficiency_table(self) -> str:
        """Generate efficiency metrics table."""
        headers = ["Method", "Avg Tokens", "Tokens/State", "Tokens/Transition"]

        rows = []
        for method in GenerationMethod:
            stats = self.method_stats[method]
            rows.append([
                method.name,
                f"{stats.avg_tokens:.0f}",
                f"{stats.avg_tokens_per_state:.1f}",
                f"{stats.avg_tokens_per_transition:.1f}",
            ])

        return self._format_table(headers, rows)

    def _generate_feature_table(self) -> str:
        """Generate feature support table."""
        headers = ["Method", "Parallel", "History", "Deep Nesting"]

        rows = []
        for method in GenerationMethod:
            stats = self.method_stats[method]
            rows.append([
                method.name,
                f"{stats.parallel_count}",
                f"{stats.history_count}",
                f"{stats.avg_depth:.1f} levels",
            ])

        return self._format_table(headers, rows)

    def _generate_detailed_results(self) -> str:
        """Generate detailed results section."""
        lines = []

        # Group by complexity
        for complexity in Complexity:
            lines.append(f"### {complexity.name}")
            lines.append("")

            tests = [
                (tid, tc) for tid, tc in self.test_cases.items()
                if tc.complexity == complexity
            ]

            if not tests:
                lines.append("No tests in this category.")
                continue

            # Show first 5 tests as examples
            for test_id, test_case in tests[:5]:
                lines.append(f"**{test_id}**: {test_case.prompt[:50]}...")

                results_for_test = self.results.get(test_id, {})
                for method in GenerationMethod:
                    result = results_for_test.get(method)
                    if result:
                        status = "VALID" if result.is_valid else "INVALID"
                        lines.append(
                            f"  - {method.name}: {status} "
                            f"(S:{result.state_count} T:{result.transition_count})"
                        )
                lines.append("")

            if len(tests) > 5:
                lines.append(f"*...and {len(tests) - 5} more tests*")
                lines.append("")

        return "\n".join(lines)

    def _generate_recommendations(self) -> str:
        """Generate recommendations based on results."""
        lines = []

        # Find best method for each complexity
        for complexity in Complexity:
            cs = self.complexity_stats[complexity]
            best = max(
                cs.by_method.items(),
                key=lambda x: x[1].validity_rate
            )
            lines.append(
                f"- **{complexity.name} tasks**: Use {best[0].name} "
                f"({best[1].validity_rate:.1%} validity)"
            )

        lines.append("")

        # Overall recommendations
        best_validity = max(self.method_stats.values(), key=lambda s: s.validity_rate)
        best_speed = min(self.method_stats.values(), key=lambda s: s.avg_latency_ms)
        best_efficiency = min(
            (s for s in self.method_stats.values() if s.avg_tokens_per_state > 0),
            key=lambda s: s.avg_tokens_per_state,
            default=None
        )

        lines.append("### Overall Recommendations")
        lines.append(f"- **Best Validity**: {best_validity.method.name}")
        lines.append(f"- **Best Speed**: {best_speed.method.name}")
        if best_efficiency:
            lines.append(f"- **Best Efficiency**: {best_efficiency.method.name}")

        return "\n".join(lines)

    def _format_table(self, headers: List[str], rows: List[List[str]]) -> str:
        """Format data as markdown table."""
        # Calculate column widths
        widths = [len(h) for h in headers]
        for row in rows:
            for i, cell in enumerate(row):
                widths[i] = max(widths[i], len(cell))

        # Format header
        header_row = "| " + " | ".join(h.ljust(widths[i]) for i, h in enumerate(headers)) + " |"
        separator = "|" + "|".join("-" * (w + 2) for w in widths) + "|"

        # Format rows
        data_rows = []
        for row in rows:
            data_row = "| " + " | ".join(
                cell.ljust(widths[i]) for i, cell in enumerate(row)
            ) + " |"
            data_rows.append(data_row)

        return "\n".join([header_row, separator] + data_rows)


def generate_markdown_report(
    results: Dict[str, Dict[GenerationMethod, MethodResult]],
    test_cases: Dict[str, TestCase],
    output_path: Optional[str] = None
) -> str:
    """Convenience function to generate report."""
    generator = ReportGenerator()

    for test_id, method_results in results.items():
        test_case = test_cases[test_id]
        for method, result in method_results.items():
            generator.add_result(test_case, method, result)

    return generator.generate_report(output_path)


def generate_comparison_table(
    method_stats: Dict[GenerationMethod, MethodStats]
) -> str:
    """Generate just the comparison table."""
    generator = ReportGenerator()
    generator.method_stats = method_stats
    return generator._generate_comparison_table()


def test_report_generator():
    """Test report generation."""
    print("=" * 60)
    print("Testing Report Generator")
    print("=" * 60)

    from .test_suite import get_test_suite
    from .method_runner import get_runner

    # Get a subset of tests
    suite = get_test_suite()
    tests = suite.tests[:10]  # First 10 tests

    # Run all methods
    generator = ReportGenerator()

    for test in tests:
        for method in GenerationMethod:
            runner = get_runner(method)
            result = runner.generate(test.prompt, test.min_states)
            generator.add_result(test, method, result)

    # Generate report
    report = generator.generate_report()

    print("\n--- REPORT PREVIEW (first 2000 chars) ---")
    print(report[:2000])
    print("...")

    print("\n--- STATISTICS ---")
    generator.compute_statistics()
    for method, stats in generator.method_stats.items():
        print(f"{method.name}: {stats.validity_rate:.1%} validity, "
              f"{stats.avg_latency_ms:.2f}ms latency")

    print("\n" + "=" * 60)
    print("Report generator tests complete!")
    print("=" * 60)


if __name__ == "__main__":
    test_report_generator()
