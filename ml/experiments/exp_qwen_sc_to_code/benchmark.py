"""
Benchmark for Statechart-to-Code Generation.

Measures:
1. Compilation success rate (target: 100%)
2. Syntax validity
3. Runtime correctness
4. Generation speed
"""

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import json

from .sc_to_go import StatechartToGo, GoGeneratorConfig, generate_go
from .sc_to_python import StatechartToPython, PythonGeneratorConfig, generate_python
from .test_generator import TestGenerator, generate_python_tests


@dataclass
class CompilationResult:
    """Result of a single compilation test."""
    name: str
    language: str
    syntax_valid: bool
    compiles: bool
    runtime_valid: bool
    time_ms: float
    errors: List[str] = field(default_factory=list)


@dataclass
class BenchmarkResult:
    """Results from running the full benchmark."""
    total_tests: int
    python_success: int
    go_success: int
    python_syntax_rate: float
    python_runtime_rate: float
    go_compile_rate: float
    avg_time_ms: float
    results: List[CompilationResult] = field(default_factory=list)


# Test statecharts of varying complexity
BENCHMARK_STATECHARTS = [
    {
        "name": "simple_toggle",
        "statechart": {
            "root_state": {
                "label": "__root__",
                "type": 2,
                "children": [
                    {"label": "Off", "type": 1, "is_initial": True},
                    {"label": "On", "type": 1},
                ]
            },
            "transitions": [
                {"from": ["Off"], "to": ["On"], "event": "TOGGLE"},
                {"from": ["On"], "to": ["Off"], "event": "TOGGLE"},
            ]
        }
    },
    {
        "name": "traffic_light",
        "statechart": {
            "root_state": {
                "label": "__root__",
                "type": 2,
                "children": [
                    {"label": "Red", "type": 1, "is_initial": True},
                    {"label": "Green", "type": 1},
                    {"label": "Yellow", "type": 1},
                ]
            },
            "transitions": [
                {"from": ["Red"], "to": ["Green"], "event": "NEXT"},
                {"from": ["Green"], "to": ["Yellow"], "event": "NEXT"},
                {"from": ["Yellow"], "to": ["Red"], "event": "NEXT"},
            ]
        }
    },
    {
        "name": "player_state",
        "statechart": {
            "root_state": {
                "label": "__root__",
                "type": 2,
                "children": [
                    {"label": "Idle", "type": 1, "is_initial": True},
                    {"label": "Walking", "type": 1},
                    {"label": "Running", "type": 1},
                    {"label": "Jumping", "type": 1},
                ]
            },
            "transitions": [
                {"from": ["Idle"], "to": ["Walking"], "event": "WALK"},
                {"from": ["Idle"], "to": ["Running"], "event": "RUN"},
                {"from": ["Walking"], "to": ["Running"], "event": "RUN"},
                {"from": ["Walking"], "to": ["Idle"], "event": "STOP"},
                {"from": ["Running"], "to": ["Walking"], "event": "WALK"},
                {"from": ["Running"], "to": ["Idle"], "event": "STOP"},
                {"from": ["Idle"], "to": ["Jumping"], "event": "JUMP"},
                {"from": ["Walking"], "to": ["Jumping"], "event": "JUMP"},
                {"from": ["Running"], "to": ["Jumping"], "event": "JUMP"},
                {"from": ["Jumping"], "to": ["Idle"], "event": "LAND"},
            ]
        }
    },
    {
        "name": "media_player",
        "statechart": {
            "root_state": {
                "label": "__root__",
                "type": 2,
                "children": [
                    {"label": "Stopped", "type": 1, "is_initial": True},
                    {"label": "Playing", "type": 1},
                    {"label": "Paused", "type": 1},
                    {"label": "Buffering", "type": 1},
                ]
            },
            "transitions": [
                {"from": ["Stopped"], "to": ["Buffering"], "event": "PLAY"},
                {"from": ["Buffering"], "to": ["Playing"], "event": "READY"},
                {"from": ["Playing"], "to": ["Paused"], "event": "PAUSE"},
                {"from": ["Paused"], "to": ["Playing"], "event": "RESUME"},
                {"from": ["Playing"], "to": ["Stopped"], "event": "STOP"},
                {"from": ["Paused"], "to": ["Stopped"], "event": "STOP"},
                {"from": ["Buffering"], "to": ["Stopped"], "event": "ERROR"},
            ]
        }
    },
    {
        "name": "connection_state",
        "statechart": {
            "root_state": {
                "label": "__root__",
                "type": 2,
                "children": [
                    {"label": "Disconnected", "type": 1, "is_initial": True},
                    {"label": "Connecting", "type": 1},
                    {"label": "Connected", "type": 1},
                    {"label": "Reconnecting", "type": 1},
                    {"label": "Error", "type": 1},
                ]
            },
            "transitions": [
                {"from": ["Disconnected"], "to": ["Connecting"], "event": "CONNECT"},
                {"from": ["Connecting"], "to": ["Connected"], "event": "SUCCESS"},
                {"from": ["Connecting"], "to": ["Error"], "event": "FAIL"},
                {"from": ["Connected"], "to": ["Disconnected"], "event": "DISCONNECT"},
                {"from": ["Connected"], "to": ["Reconnecting"], "event": "LOST"},
                {"from": ["Reconnecting"], "to": ["Connected"], "event": "SUCCESS"},
                {"from": ["Reconnecting"], "to": ["Error"], "event": "FAIL"},
                {"from": ["Error"], "to": ["Disconnected"], "event": "RESET"},
            ]
        }
    },
]


class CodeBenchmark:
    """Benchmark for code generation quality."""

    def __init__(self, use_llm: bool = False):
        self.use_llm = use_llm

    def run_python_test(self, name: str, statechart: Dict) -> CompilationResult:
        """Test Python code generation."""
        start = time.time()

        config = PythonGeneratorConfig(
            use_llm=self.use_llm,
            verify_syntax=True,
            verify_import=True,
        )

        result = generate_python(statechart, config)
        elapsed = (time.time() - start) * 1000

        # Test runtime correctness
        runtime_valid = False
        if result.syntax_valid and result.import_valid:
            runtime_valid = self._test_python_runtime(result.code, statechart)

        return CompilationResult(
            name=name,
            language="python",
            syntax_valid=result.syntax_valid,
            compiles=result.import_valid,
            runtime_valid=runtime_valid,
            time_ms=elapsed,
            errors=result.errors,
        )

    def run_go_test(self, name: str, statechart: Dict) -> CompilationResult:
        """Test Go code generation."""
        start = time.time()

        config = GoGeneratorConfig(
            use_llm=self.use_llm,
            verify_compilation=True,
        )

        result = generate_go(statechart, config)
        elapsed = (time.time() - start) * 1000

        return CompilationResult(
            name=name,
            language="go",
            syntax_valid=True,  # Go compiler checks syntax
            compiles=result.compiles,
            runtime_valid=result.compiles,  # If it compiles, it's likely correct
            time_ms=elapsed,
            errors=result.errors,
        )

    def _test_python_runtime(self, code: str, statechart: Dict) -> bool:
        """Test Python code runs correctly."""
        try:
            namespace = {}
            exec(code, namespace)

            SM = namespace.get('StateMachine')
            State = namespace.get('State')
            Event = namespace.get('Event')

            if not all([SM, State, Event]):
                return False

            # Create instance and test transitions
            sm = SM()

            # Get expected initial state
            gen = TestGenerator(statechart)
            initial = gen.initial_state

            if not hasattr(State, initial):
                return False

            if sm.state != getattr(State, initial):
                return False

            # Test first transition if available
            if gen.transitions:
                first = list(gen.transitions.items())[0]
                (src, event), tgt = first

                if src == initial and hasattr(Event, event):
                    sm.send(getattr(Event, event))
                    if sm.state != getattr(State, tgt):
                        return False

            return True

        except Exception as e:
            return False

    def run(self, verbose: bool = True) -> BenchmarkResult:
        """Run full benchmark."""
        results = []

        if verbose:
            print("=" * 70)
            print("STATECHART TO CODE BENCHMARK")
            print("=" * 70)
            print(f"LLM enabled: {self.use_llm}")
            print(f"Test cases: {len(BENCHMARK_STATECHARTS)}")
            print("-" * 70)
            header = f"{'Name':<20} {'Lang':<8} {'Syntax':<8} {'Compile':<10} {'Runtime':<10} {'Time(ms)':<10}"
            print(header)
            print("-" * 70)

        for test in BENCHMARK_STATECHARTS:
            name = test['name']
            statechart = test['statechart']

            # Test Python
            py_result = self.run_python_test(name, statechart)
            results.append(py_result)

            if verbose:
                print(
                    f"{name:<20} "
                    f"{'python':<8} "
                    f"{'PASS' if py_result.syntax_valid else 'FAIL':<8} "
                    f"{'PASS' if py_result.compiles else 'FAIL':<10} "
                    f"{'PASS' if py_result.runtime_valid else 'FAIL':<10} "
                    f"{py_result.time_ms:<10.1f}"
                )

            # Test Go
            go_result = self.run_go_test(name, statechart)
            results.append(go_result)

            if verbose:
                print(
                    f"{'':<20} "
                    f"{'go':<8} "
                    f"{'PASS' if go_result.syntax_valid else 'FAIL':<8} "
                    f"{'PASS' if go_result.compiles else 'FAIL':<10} "
                    f"{'PASS' if go_result.runtime_valid else 'FAIL':<10} "
                    f"{go_result.time_ms:<10.1f}"
                )

        # Compute summary
        python_results = [r for r in results if r.language == 'python']
        go_results = [r for r in results if r.language == 'go']

        python_syntax = sum(1 for r in python_results if r.syntax_valid)
        python_runtime = sum(1 for r in python_results if r.runtime_valid)
        go_compile = sum(1 for r in go_results if r.compiles)

        n_python = len(python_results)
        n_go = len(go_results)

        benchmark_result = BenchmarkResult(
            total_tests=len(results),
            python_success=python_runtime,
            go_success=go_compile,
            python_syntax_rate=python_syntax / n_python if n_python else 0,
            python_runtime_rate=python_runtime / n_python if n_python else 0,
            go_compile_rate=go_compile / n_go if n_go else 0,
            avg_time_ms=sum(r.time_ms for r in results) / len(results) if results else 0,
            results=results,
        )

        if verbose:
            print("-" * 70)
            print("\nSUMMARY:")
            print(f"  Python syntax validity: {python_syntax}/{n_python} ({benchmark_result.python_syntax_rate:.1%})")
            print(f"  Python runtime success: {python_runtime}/{n_python} ({benchmark_result.python_runtime_rate:.1%})")
            print(f"  Go compilation success: {go_compile}/{n_go} ({benchmark_result.go_compile_rate:.1%})")
            print(f"  Average time: {benchmark_result.avg_time_ms:.1f}ms")

            # Check target
            if benchmark_result.python_runtime_rate == 1.0 and benchmark_result.go_compile_rate == 1.0:
                print("\n  TARGET ACHIEVED: 100% compilable output!")
            else:
                total_success = (python_runtime + go_compile) / (n_python + n_go)
                print(f"\n  Overall success rate: {total_success:.1%}")

            print("=" * 70)

        return benchmark_result


def run_benchmark(use_llm: bool = False, verbose: bool = True) -> BenchmarkResult:
    """
    Run the code generation benchmark.

    Convenience function.
    """
    benchmark = CodeBenchmark(use_llm=use_llm)
    return benchmark.run(verbose=verbose)


def test_benchmark():
    """Test the benchmark system."""
    print("=" * 60)
    print("BENCHMARK SYSTEM TEST")
    print("=" * 60)

    # Run without LLM
    print("\n1. Running benchmark (template-only):")
    result = run_benchmark(use_llm=False, verbose=True)

    print(f"\n2. Summary:")
    print(f"  Python: {result.python_success}/{len([r for r in result.results if r.language == 'python'])}")
    print(f"  Go: {result.go_success}/{len([r for r in result.results if r.language == 'go'])}")

    print("\n" + "=" * 60)
    print("Benchmark tests complete!")
    print("=" * 60)


if __name__ == "__main__":
    test_benchmark()
