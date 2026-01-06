"""
Qwen Guard Synthesis Benchmark

Benchmarks for NL -> Guard expression generation:
1. Syntax validity (target: 95%+)
2. Semantic correctness (target: 80%+)
3. Variable coverage
4. Generation speed

Test categories:
- Simple comparisons
- Boolean combinations
- Game rules (Ko, castling, etc.)
- State machine transitions

NO HARDCODING: All patterns learned from few-shot examples.
"""

import time
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any

from .guard_generator import (
    QwenGuardGenerator, GeneratorConfig,
    GenerationRequest, GenerationResult, batch_generate,
)
from .nl_parser import NLParser, ParseResult
from .validator import GuardValidator, batch_validate, BatchValidationResult


# =============================================================================
# BENCHMARK RESULT
# =============================================================================

@dataclass
class BenchmarkResult:
    """Result of a benchmark run."""
    category: str
    total_tests: int
    syntax_valid: int
    semantic_correct: int
    syntax_rate: float
    semantic_rate: float
    avg_generation_time: float
    details: Dict[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return self.syntax_rate >= 0.95 and self.semantic_rate >= 0.80


# =============================================================================
# TEST CASES
# =============================================================================

# Simple comparisons
SIMPLE_COMPARISON_TESTS = [
    {
        "description": "counter is greater than 5",
        "variables": ["counter", "limit", "step"],
        "expected": "counter > 5",
        "test_context": {"counter": 10, "limit": 100, "step": 1},
        "expected_result": True,
    },
    {
        "description": "health is zero or less",
        "variables": ["health", "max_health", "damage"],
        "expected": "health <= 0",
        "test_context": {"health": 0, "max_health": 100, "damage": 10},
        "expected_result": True,
    },
    {
        "description": "level is at least 10",
        "variables": ["level", "experience", "rank"],
        "expected": "level >= 10",
        "test_context": {"level": 15, "experience": 5000, "rank": 2},
        "expected_result": True,
    },
    {
        "description": "score equals 100",
        "variables": ["score", "bonus", "penalty"],
        "expected": "score == 100",
        "test_context": {"score": 100, "bonus": 0, "penalty": 0},
        "expected_result": True,
    },
]

# Boolean expressions
BOOLEAN_TESTS = [
    {
        "description": "player is not moving",
        "variables": ["is_moving", "speed", "direction"],
        "expected": "not is_moving",
        "test_context": {"is_moving": False, "speed": 0, "direction": "north"},
        "expected_result": True,
    },
    {
        "description": "system is active and ready",
        "variables": ["is_active", "is_ready", "is_paused"],
        "expected": "is_active and is_ready",
        "test_context": {"is_active": True, "is_ready": True, "is_paused": False},
        "expected_result": True,
    },
    {
        "description": "either timeout or cancelled",
        "variables": ["timeout", "cancelled", "completed"],
        "expected": "timeout or cancelled",
        "test_context": {"timeout": True, "cancelled": False, "completed": False},
        "expected_result": True,
    },
    {
        "description": "task is complete",
        "variables": ["is_complete", "progress", "total"],
        "expected": "is_complete",
        "test_context": {"is_complete": True, "progress": 100, "total": 100},
        "expected_result": True,
    },
]

# Game rules
GAME_RULE_TESTS = [
    {
        "description": "move to same position as last capture (Ko rule)",
        "variables": ["move_x", "move_y", "last_capture_x", "last_capture_y", "would_capture"],
        "expected": "(move_x == last_capture_x) and (move_y == last_capture_y)",
        "test_context": {"move_x": 3, "move_y": 4, "last_capture_x": 3, "last_capture_y": 4, "would_capture": True},
        "expected_result": True,
    },
    {
        "description": "king has not moved and not in check",
        "variables": ["king_moved", "in_check", "can_castle"],
        "expected": "(not king_moved) and (not in_check)",
        "test_context": {"king_moved": False, "in_check": False, "can_castle": True},
        "expected_result": True,
    },
    {
        "description": "piece can capture enemy",
        "variables": ["can_capture", "enemy_present", "is_blocked"],
        "expected": "can_capture and enemy_present",
        "test_context": {"can_capture": True, "enemy_present": True, "is_blocked": False},
        "expected_result": True,
    },
]

# State machine transitions
STATE_MACHINE_TESTS = [
    {
        "description": "buffer is not empty",
        "variables": ["buffer_count", "max_size", "is_full"],
        "expected": "buffer_count > 0",
        "test_context": {"buffer_count": 5, "max_size": 100, "is_full": False},
        "expected_result": True,
    },
    {
        "description": "connection is established and authenticated",
        "variables": ["connected", "authenticated", "session_id"],
        "expected": "connected and authenticated",
        "test_context": {"connected": True, "authenticated": True, "session_id": "abc123"},
        "expected_result": True,
    },
    {
        "description": "retry count is less than max retries",
        "variables": ["retry_count", "max_retries", "last_error"],
        "expected": "retry_count < max_retries",
        "test_context": {"retry_count": 2, "max_retries": 5, "last_error": None},
        "expected_result": True,
    },
]


# =============================================================================
# BENCHMARK CLASS
# =============================================================================

class GuardSynthesisBenchmark:
    """
    Benchmark for NL -> Guard expression synthesis.

    Wraps the benchmark functions in a class-based interface
    consistent with other experiments.
    """

    # Targets
    SYNTAX_TARGET = 0.95   # 95% syntax validity
    SEMANTIC_TARGET = 0.80  # 80% semantic correctness

    def __init__(
        self,
        generator: QwenGuardGenerator = None,
        validator: GuardValidator = None,
        verbose: bool = True,
    ):
        self.generator = generator or QwenGuardGenerator(GeneratorConfig(verbose=False))
        self.validator = validator or GuardValidator(allow_unknown_variables=True)
        self.verbose = verbose
        self.results: List[BenchmarkResult] = []

    @property
    def categories(self) -> List[Tuple[str, List[Dict]]]:
        """Get all test categories."""
        return [
            ("Simple Comparisons", SIMPLE_COMPARISON_TESTS),
            ("Boolean Expressions", BOOLEAN_TESTS),
            ("Game Rules", GAME_RULE_TESTS),
            ("State Machine", STATE_MACHINE_TESTS),
        ]

    def run(self) -> List[BenchmarkResult]:
        """Run full benchmark across all categories."""
        self.results = run_full_benchmark(verbose=self.verbose)
        return self.results

    def run_quick(self) -> List[BenchmarkResult]:
        """Run quick benchmark with fewer tests."""
        self.results = quick_benchmark(verbose=self.verbose)
        return self.results

    def run_category(self, category_name: str) -> Optional[BenchmarkResult]:
        """Run benchmark for a specific category."""
        for name, tests in self.categories:
            if name.lower() == category_name.lower():
                result = run_category_benchmark(
                    name, tests, self.generator, self.validator,
                    verbose=self.verbose
                )
                self.results.append(result)
                return result
        return None

    @property
    def overall_syntax_rate(self) -> float:
        """Get overall syntax validity rate."""
        if not self.results:
            return 0.0
        total = sum(r.total_tests for r in self.results)
        valid = sum(r.syntax_valid for r in self.results)
        return valid / max(total, 1)

    @property
    def overall_semantic_rate(self) -> float:
        """Get overall semantic correctness rate."""
        if not self.results:
            return 0.0
        total = sum(r.total_tests for r in self.results)
        correct = sum(r.semantic_correct for r in self.results)
        return correct / max(total, 1)

    @property
    def passed(self) -> bool:
        """Check if benchmark meets targets."""
        return (
            self.overall_syntax_rate >= self.SYNTAX_TARGET and
            self.overall_semantic_rate >= self.SEMANTIC_TARGET
        )

    @property
    def syntax_passed(self) -> bool:
        """Check if syntax target is met."""
        return self.overall_syntax_rate >= self.SYNTAX_TARGET

    @property
    def semantic_passed(self) -> bool:
        """Check if semantic target is met."""
        return self.overall_semantic_rate >= self.SEMANTIC_TARGET

    def summary(self) -> Dict[str, Any]:
        """Get benchmark summary."""
        return {
            "total_categories": len(self.results),
            "total_tests": sum(r.total_tests for r in self.results),
            "syntax_valid": sum(r.syntax_valid for r in self.results),
            "semantic_correct": sum(r.semantic_correct for r in self.results),
            "syntax_rate": self.overall_syntax_rate,
            "semantic_rate": self.overall_semantic_rate,
            "syntax_target": self.SYNTAX_TARGET,
            "semantic_target": self.SEMANTIC_TARGET,
            "syntax_passed": self.syntax_passed,
            "semantic_passed": self.semantic_passed,
            "passed": self.passed,
        }

    def print_summary(self):
        """Print benchmark summary."""
        s = self.summary()
        print("\n" + "=" * 50)
        print("GUARD SYNTHESIS BENCHMARK SUMMARY")
        print("=" * 50)
        print(f"Categories: {s['total_categories']}")
        print(f"Total tests: {s['total_tests']}")
        print(f"Syntax valid: {s['syntax_valid']} ({s['syntax_rate']:.1%})")
        print(f"Semantic correct: {s['semantic_correct']} ({s['semantic_rate']:.1%})")
        print("-" * 50)
        print(f"Syntax target ({s['syntax_target']:.0%}): {'PASS' if s['syntax_passed'] else 'FAIL'}")
        print(f"Semantic target ({s['semantic_target']:.0%}): {'PASS' if s['semantic_passed'] else 'FAIL'}")
        print("=" * 50)


# =============================================================================
# BENCHMARK RUNNER FUNCTIONS
# =============================================================================

def run_category_benchmark(
    category: str,
    tests: List[Dict],
    generator: QwenGuardGenerator,
    validator: GuardValidator,
    verbose: bool = True,
) -> BenchmarkResult:
    """Run benchmark for a test category."""
    if verbose:
        print(f"\n{'='*60}")
        print(f"BENCHMARK: {category}")
        print(f"{'='*60}")

    total_time = 0.0
    syntax_valid = 0
    semantic_correct = 0

    for i, test in enumerate(tests):
        desc = test["description"]
        variables = test["variables"]
        expected = test.get("expected")
        context = test.get("test_context", {})
        expected_result = test.get("expected_result")

        if verbose:
            print(f"\n[{i+1}/{len(tests)}] {desc[:50]}")

        # Generate
        start = time.time()
        candidates = generator.generate(desc, variables, num_candidates=3)
        elapsed = time.time() - start
        total_time += elapsed

        if verbose:
            print(f"  Generated {len(candidates)} candidates in {elapsed:.3f}s")

        # Validate syntax
        valid_candidates = []
        for cand in candidates:
            is_valid, error = validator.validate(cand)
            if is_valid:
                valid_candidates.append(cand)

        if valid_candidates:
            syntax_valid += 1
            if verbose:
                print(f"  Valid: {valid_candidates[0][:40]}")

            # Check semantic correctness
            if expected_result is not None and context:
                best = valid_candidates[0]
                try:
                    safe_ctx = {"True": True, "False": False, "None": None}
                    safe_ctx.update(context)
                    actual = eval(best, {"__builtins__": {}}, safe_ctx)
                    if actual == expected_result:
                        semantic_correct += 1
                        if verbose:
                            print(f"  Semantic: CORRECT")
                    else:
                        if verbose:
                            print(f"  Semantic: WRONG (expected {expected_result}, got {actual})")
                except Exception as e:
                    if verbose:
                        print(f"  Semantic: ERROR ({e})")
        else:
            if verbose:
                print(f"  No valid candidates!")
                for cand in candidates[:2]:
                    _, error = validator.validate(cand)
                    print(f"    {cand[:30]}: {error}")

    result = BenchmarkResult(
        category=category,
        total_tests=len(tests),
        syntax_valid=syntax_valid,
        semantic_correct=semantic_correct,
        syntax_rate=syntax_valid / max(len(tests), 1),
        semantic_rate=semantic_correct / max(len(tests), 1),
        avg_generation_time=total_time / max(len(tests), 1),
    )

    if verbose:
        print(f"\n{'-'*40}")
        print(f"Results for {category}:")
        print(f"  Syntax valid: {result.syntax_valid}/{result.total_tests} ({result.syntax_rate:.1%})")
        print(f"  Semantic correct: {result.semantic_correct}/{result.total_tests} ({result.semantic_rate:.1%})")
        print(f"  Avg time: {result.avg_generation_time:.3f}s")

    return result


def run_full_benchmark(verbose: bool = True) -> List[BenchmarkResult]:
    """Run full benchmark across all categories."""
    if verbose:
        print("=" * 70)
        print("QWEN GUARD SYNTHESIS BENCHMARK")
        print("=" * 70)
        print("\nInitializing generator and validator...")

    config = GeneratorConfig(verbose=False)
    generator = QwenGuardGenerator(config)
    validator = GuardValidator(allow_unknown_variables=True)

    categories = [
        ("Simple Comparisons", SIMPLE_COMPARISON_TESTS),
        ("Boolean Expressions", BOOLEAN_TESTS),
        ("Game Rules", GAME_RULE_TESTS),
        ("State Machine", STATE_MACHINE_TESTS),
    ]

    results = []
    for category, tests in categories:
        result = run_category_benchmark(
            category, tests, generator, validator, verbose=verbose
        )
        results.append(result)

    # Summary
    if verbose:
        print("\n" + "=" * 70)
        print("BENCHMARK SUMMARY")
        print("=" * 70)
        print(f"{'Category':<25} {'Syntax':>12} {'Semantic':>12} {'Time':>10}")
        print("-" * 70)

        total_tests = 0
        total_syntax = 0
        total_semantic = 0

        for r in results:
            status = "PASS" if r.passed else "FAIL"
            print(f"{r.category:<25} {r.syntax_rate:>11.1%} {r.semantic_rate:>11.1%} {r.avg_generation_time:>9.3f}s")
            total_tests += r.total_tests
            total_syntax += r.syntax_valid
            total_semantic += r.semantic_correct

        print("-" * 70)
        overall_syntax = total_syntax / max(total_tests, 1)
        overall_semantic = total_semantic / max(total_tests, 1)
        print(f"{'OVERALL':<25} {overall_syntax:>11.1%} {overall_semantic:>11.1%}")
        print("=" * 70)

        # Check targets
        print("\nTarget Check:")
        print(f"  Syntax validity target (95%): {'PASS' if overall_syntax >= 0.95 else 'FAIL'} ({overall_syntax:.1%})")
        print(f"  Semantic correctness target (80%): {'PASS' if overall_semantic >= 0.80 else 'WORK NEEDED'} ({overall_semantic:.1%})")

        if overall_syntax >= 0.95 and overall_semantic >= 0.80:
            print("\nKEY INSIGHT: Qwen2.5-Coder can generate valid guard expressions from NL!")
        else:
            print("\nNote: Model may need fine-tuning or more few-shot examples.")

    return results


def quick_benchmark(verbose: bool = True) -> List[BenchmarkResult]:
    """Run quick benchmark with fewer tests."""
    if verbose:
        print("=" * 60)
        print("QUICK BENCHMARK")
        print("=" * 60)

    config = GeneratorConfig(verbose=False)
    generator = QwenGuardGenerator(config)
    validator = GuardValidator(allow_unknown_variables=True)

    # Use first 2 tests from each category
    categories = [
        ("Simple Comparisons", SIMPLE_COMPARISON_TESTS[:2]),
        ("Boolean Expressions", BOOLEAN_TESTS[:2]),
    ]

    results = []
    for category, tests in categories:
        result = run_category_benchmark(
            category, tests, generator, validator, verbose=verbose
        )
        results.append(result)

    return results


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("Running Qwen Guard Synthesis Benchmark...")
    print("(Using mock generation if model not available)")
    print()

    results = run_full_benchmark(verbose=True)

    # Overall pass/fail
    all_passed = all(r.passed for r in results)
    if all_passed:
        print("\n[SUCCESS] All benchmarks passed!")
    else:
        print("\n[PARTIAL] Some benchmarks need improvement")
