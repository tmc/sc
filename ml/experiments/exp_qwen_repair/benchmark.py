#!/usr/bin/env python3
"""
Benchmark: Measure statechart repair success rate.

Generates invalid statecharts with known errors, repairs them,
and measures success rate. Target: 90%+ repair success.
"""

import json
import random
import time
import copy
from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional, Any, Tuple
from enum import Enum, auto

from .error_analyzer import ErrorAnalyzer, ErrorType, ValidationResult
from .repair_strategies import RepairStrategies
from .sc_repairer import SCRepairer, RepairConfig, RepairSession


@dataclass
class BenchmarkResult:
    """Result of benchmark run."""
    total_cases: int = 0
    successful_repairs: int = 0
    failed_repairs: int = 0
    deterministic_fixes: int = 0
    llm_fixes: int = 0
    total_duration: float = 0.0
    cases: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        if self.total_cases == 0:
            return 0.0
        return self.successful_repairs / self.total_cases

    @property
    def avg_duration(self) -> float:
        if self.total_cases == 0:
            return 0.0
        return self.total_duration / self.total_cases


class InvalidChartGenerator:
    """
    Generates invalid statecharts with specific error types.

    Used to create test cases for benchmarking.
    """

    def __init__(self, seed: Optional[int] = None):
        if seed is not None:
            random.seed(seed)

    def generate_valid_base(self, num_states: int = 5) -> Dict[str, Any]:
        """Generate a valid base chart to corrupt."""
        states = [f"S{i}" for i in range(num_states)]

        children = [
            {"label": states[0], "is_initial": True},
        ] + [
            {"label": s} for s in states[1:]
        ]

        transitions = []
        for i in range(len(states) - 1):
            transitions.append({
                "from": [states[i]],
                "to": [states[i + 1]],
                "event": f"E{i}",
            })
        # Add loop back
        transitions.append({
            "from": [states[-1]],
            "to": [states[0]],
            "event": "RESET",
        })

        return {
            "root_state": {
                "label": "__root__",
                "type": 2,
                "children": children,
            },
            "transitions": transitions,
        }

    def corrupt_chart(
        self,
        chart: Dict[str, Any],
        error_types: Optional[List[ErrorType]] = None,
    ) -> Tuple[Dict[str, Any], List[ErrorType]]:
        """
        Introduce errors into a valid chart.

        Returns corrupted chart and list of introduced error types.
        """
        corrupted = copy.deepcopy(chart)
        introduced = []

        if error_types is None:
            # Random selection
            available = [
                ErrorType.DUPLICATE_STATE,
                ErrorType.INVALID_SOURCE,
                ErrorType.INVALID_TARGET,
                ErrorType.NO_INITIAL_STATE,
                ErrorType.MISSING_EVENT,
            ]
            error_types = [random.choice(available)]

        for error_type in error_types:
            if error_type == ErrorType.DUPLICATE_STATE:
                if self._introduce_duplicate(corrupted):
                    introduced.append(error_type)

            elif error_type == ErrorType.INVALID_SOURCE:
                if self._introduce_invalid_source(corrupted):
                    introduced.append(error_type)

            elif error_type == ErrorType.INVALID_TARGET:
                if self._introduce_invalid_target(corrupted):
                    introduced.append(error_type)

            elif error_type == ErrorType.NO_INITIAL_STATE:
                if self._remove_initial(corrupted):
                    introduced.append(error_type)

            elif error_type == ErrorType.MISSING_EVENT:
                if self._remove_event(corrupted):
                    introduced.append(error_type)

        return corrupted, introduced

    def _introduce_duplicate(self, chart: Dict[str, Any]) -> bool:
        """Add a duplicate state label."""
        children = chart.get("root_state", {}).get("children", [])
        if len(children) >= 2:
            # Copy first label to second
            children[1]["label"] = children[0]["label"]
            return True
        return False

    def _introduce_invalid_source(self, chart: Dict[str, Any]) -> bool:
        """Create typo in transition source."""
        transitions = chart.get("transitions", [])
        if transitions:
            trans = random.choice(transitions)
            if trans.get("from"):
                orig = trans["from"][0]
                # Introduce typo
                if len(orig) > 2:
                    typo = orig[:-1] + "x"
                else:
                    typo = orig + "x"
                trans["from"] = [typo]
                return True
        return False

    def _introduce_invalid_target(self, chart: Dict[str, Any]) -> bool:
        """Create typo in transition target."""
        transitions = chart.get("transitions", [])
        if transitions:
            trans = random.choice(transitions)
            if trans.get("to"):
                orig = trans["to"][0]
                # Introduce typo
                if len(orig) > 2:
                    typo = orig[:-1] + "y"
                else:
                    typo = orig + "y"
                trans["to"] = [typo]
                return True
        return False

    def _remove_initial(self, chart: Dict[str, Any]) -> bool:
        """Remove initial state marker."""
        children = chart.get("root_state", {}).get("children", [])
        for child in children:
            if child.get("is_initial"):
                del child["is_initial"]
                return True
        return False

    def _remove_event(self, chart: Dict[str, Any]) -> bool:
        """Remove event from transition."""
        transitions = chart.get("transitions", [])
        if transitions:
            trans = random.choice(transitions)
            if "event" in trans:
                del trans["event"]
                return True
        return False

    def generate_test_suite(
        self,
        num_cases: int = 20,
        errors_per_case: int = 1,
    ) -> List[Dict[str, Any]]:
        """Generate a suite of test cases."""
        test_cases = []

        error_types = [
            ErrorType.DUPLICATE_STATE,
            ErrorType.INVALID_SOURCE,
            ErrorType.INVALID_TARGET,
            ErrorType.NO_INITIAL_STATE,
            ErrorType.MISSING_EVENT,
        ]

        for i in range(num_cases):
            # Generate base chart with varying sizes
            num_states = random.randint(3, 8)
            base = self.generate_valid_base(num_states)

            # Select errors to introduce
            selected_errors = random.sample(
                error_types,
                min(errors_per_case, len(error_types))
            )

            # Corrupt
            corrupted, introduced = self.corrupt_chart(base, selected_errors)

            test_cases.append({
                "id": i + 1,
                "original": base,
                "corrupted": corrupted,
                "error_types": [e.name for e in introduced],
            })

        return test_cases


class RepairBenchmark:
    """
    Benchmarks statechart repair performance.

    Tests both deterministic and LLM-based repair.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-Coder-0.5B-Instruct",
        sc_path: str = "./sc",
    ):
        self.repairer = SCRepairer(model_name, sc_path)
        self.analyzer = ErrorAnalyzer(sc_path)
        self.generator = InvalidChartGenerator()

    def run_benchmark(
        self,
        num_cases: int = 30,
        errors_per_case: int = 1,
        config: Optional[RepairConfig] = None,
    ) -> BenchmarkResult:
        """
        Run full benchmark.

        Args:
            num_cases: Number of test cases
            errors_per_case: Number of errors per case
            config: Repair configuration

        Returns:
            BenchmarkResult with statistics
        """
        config = config or RepairConfig(verbose=False)
        result = BenchmarkResult()

        print(f"\nGenerating {num_cases} test cases...")
        test_cases = self.generator.generate_test_suite(num_cases, errors_per_case)

        print(f"Running benchmark...")
        start_time = time.time()

        for tc in test_cases:
            case_start = time.time()

            # Validate corrupted chart
            initial_result = self.analyzer.validate_chart(tc["corrupted"])

            # Repair
            session = self.repairer.repair(tc["corrupted"], config)

            case_duration = time.time() - case_start

            # Record result
            case_result = {
                "id": tc["id"],
                "error_types": tc["error_types"],
                "initial_errors": len(initial_result.errors),
                "final_valid": session.success,
                "iterations": len(session.attempts),
                "duration": case_duration,
            }

            if session.success:
                result.successful_repairs += 1
                # Check which method succeeded
                for attempt in session.attempts:
                    if attempt.success:
                        if attempt.method == "deterministic":
                            result.deterministic_fixes += 1
                        else:
                            result.llm_fixes += 1
                        break
            else:
                result.failed_repairs += 1

            result.total_cases += 1
            result.cases.append(case_result)

            # Progress
            status = "OK" if session.success else "FAIL"
            print(f"  [{result.total_cases}/{num_cases}] {status} - {tc['error_types']} ({case_duration:.2f}s)")

        result.total_duration = time.time() - start_time

        return result

    def run_by_error_type(
        self,
        cases_per_type: int = 10,
        config: Optional[RepairConfig] = None,
    ) -> Dict[str, float]:
        """
        Benchmark success rate by error type.

        Returns dict of error_type -> success_rate.
        """
        config = config or RepairConfig(verbose=False)
        results = {}

        error_types = [
            ErrorType.DUPLICATE_STATE,
            ErrorType.INVALID_SOURCE,
            ErrorType.INVALID_TARGET,
            ErrorType.NO_INITIAL_STATE,
            ErrorType.MISSING_EVENT,
        ]

        for error_type in error_types:
            print(f"\nTesting {error_type.name}...")
            successes = 0

            for i in range(cases_per_type):
                base = self.generator.generate_valid_base()
                corrupted, _ = self.generator.corrupt_chart(base, [error_type])

                session = self.repairer.repair(corrupted, config)
                if session.success:
                    successes += 1

                status = "OK" if session.success else "FAIL"
                print(f"  [{i+1}/{cases_per_type}] {status}")

            results[error_type.name] = successes / cases_per_type

        return results


def run_full_benchmark(num_cases: int = 30) -> BenchmarkResult:
    """Run full benchmark and print results."""
    print("=" * 60)
    print("STATECHART REPAIR BENCHMARK")
    print("=" * 60)

    benchmark = RepairBenchmark()
    config = RepairConfig(
        verbose=False,
        max_iterations=3,
        use_deterministic_first=True,
    )

    result = benchmark.run_benchmark(num_cases, config=config)

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    print(f"\nSuccess Rate: {result.successful_repairs}/{result.total_cases} "
          f"({result.success_rate:.1%})")
    print(f"  - Deterministic fixes: {result.deterministic_fixes}")
    print(f"  - LLM fixes: {result.llm_fixes}")
    print(f"  - Failures: {result.failed_repairs}")

    print(f"\nTiming:")
    print(f"  - Total duration: {result.total_duration:.1f}s")
    print(f"  - Average per case: {result.avg_duration:.2f}s")

    # By error type
    print(f"\nBy Error Type:")
    error_counts = {}
    error_successes = {}

    for case in result.cases:
        for et in case["error_types"]:
            error_counts[et] = error_counts.get(et, 0) + 1
            if case["final_valid"]:
                error_successes[et] = error_successes.get(et, 0) + 1

    for et in sorted(error_counts.keys()):
        success = error_successes.get(et, 0)
        total = error_counts[et]
        rate = success / total if total > 0 else 0
        print(f"  - {et}: {success}/{total} ({rate:.1%})")

    target_met = result.success_rate >= 0.90
    print(f"\n{'='*60}")
    print(f"TARGET (90%): {'ACHIEVED' if target_met else 'NOT MET'}")
    print(f"{'='*60}")

    return result


def demo():
    """Quick demo with fewer cases."""
    print("=" * 60)
    print("REPAIR BENCHMARK DEMO")
    print("=" * 60)

    benchmark = RepairBenchmark()
    config = RepairConfig(
        verbose=False,
        max_iterations=2,
        use_deterministic_first=True,
    )

    # Run small benchmark
    result = benchmark.run_benchmark(num_cases=10, config=config)

    print(f"\nSuccess Rate: {result.success_rate:.1%}")
    print(f"Deterministic: {result.deterministic_fixes}, LLM: {result.llm_fixes}")

    return result


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--full":
        run_full_benchmark(num_cases=50)
    else:
        demo()
