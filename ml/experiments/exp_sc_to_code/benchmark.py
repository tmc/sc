#!/usr/bin/env python3
"""
SC to Code Benchmark - Test LLM code generation from statecharts.

Tests 5 statechart types:
1. Toggle (2 states, 2 transitions)
2. Traffic light (3 states, cycle)
3. Door FSM (3 states, multiple events)
4. Login flow (4+ states, guards)
5. Game state (parallel regions if possible)

Target: 60%+ runnable, 50%+ behaviorally correct.
"""

import json
import time
from dataclasses import dataclass, field
from typing import Dict, List

from .code_generator import SCCodeGenerator, GenerationResult
from .code_validator import CodeValidator, ValidationResult


# 5 Test statechart types
TEST_STATECHARTS = [
    # 1. Toggle (2 states, 2 transitions)
    {
        "name": "Toggle",
        "class_name": "Toggle",
        "sc": {
            "root_state": {
                "label": "__root__",
                "type": 2,
                "children": [
                    {"label": "Off", "type": 1, "is_initial": True},
                    {"label": "On", "type": 1}
                ]
            },
            "transitions": [
                {"from": ["Off"], "to": ["On"], "event": "TOGGLE"},
                {"from": ["On"], "to": ["Off"], "event": "TOGGLE"}
            ]
        },
        "description": "Simple toggle machine"
    },

    # 2. Traffic light (3 states, cycle)
    {
        "name": "TrafficLight",
        "class_name": "TrafficLight",
        "sc": {
            "root_state": {
                "label": "__root__",
                "type": 2,
                "children": [
                    {"label": "Red", "type": 1, "is_initial": True},
                    {"label": "Green", "type": 1},
                    {"label": "Yellow", "type": 1}
                ]
            },
            "transitions": [
                {"from": ["Red"], "to": ["Green"], "event": "NEXT"},
                {"from": ["Green"], "to": ["Yellow"], "event": "NEXT"},
                {"from": ["Yellow"], "to": ["Red"], "event": "NEXT"}
            ]
        },
        "description": "Traffic light cycle"
    },

    # 3. Door FSM (3 states, multiple events)
    {
        "name": "Door",
        "class_name": "Door",
        "sc": {
            "root_state": {
                "label": "__root__",
                "type": 2,
                "children": [
                    {"label": "Closed", "type": 1, "is_initial": True},
                    {"label": "Open", "type": 1},
                    {"label": "Locked", "type": 1}
                ]
            },
            "transitions": [
                {"from": ["Closed"], "to": ["Open"], "event": "OPEN"},
                {"from": ["Open"], "to": ["Closed"], "event": "CLOSE"},
                {"from": ["Closed"], "to": ["Locked"], "event": "LOCK"},
                {"from": ["Locked"], "to": ["Closed"], "event": "UNLOCK"}
            ]
        },
        "description": "Door with lock"
    },

    # 4. Login flow (4+ states, guards)
    {
        "name": "LoginFlow",
        "class_name": "LoginFlow",
        "sc": {
            "root_state": {
                "label": "__root__",
                "type": 2,
                "children": [
                    {"label": "Idle", "type": 1, "is_initial": True},
                    {"label": "Authenticating", "type": 1},
                    {"label": "LoggedIn", "type": 1},
                    {"label": "Error", "type": 1}
                ]
            },
            "transitions": [
                {"from": ["Idle"], "to": ["Authenticating"], "event": "LOGIN"},
                {"from": ["Authenticating"], "to": ["LoggedIn"], "event": "SUCCESS"},
                {"from": ["Authenticating"], "to": ["Error"], "event": "FAIL"},
                {"from": ["Error"], "to": ["Idle"], "event": "RETRY"},
                {"from": ["LoggedIn"], "to": ["Idle"], "event": "LOGOUT"}
            ]
        },
        "description": "Login authentication flow"
    },

    # 5. Game state (player states)
    {
        "name": "GameState",
        "class_name": "GameState",
        "sc": {
            "root_state": {
                "label": "__root__",
                "type": 2,
                "children": [
                    {"label": "Idle", "type": 1, "is_initial": True},
                    {"label": "Walking", "type": 1},
                    {"label": "Running", "type": 1},
                    {"label": "Jumping", "type": 1},
                    {"label": "Attacking", "type": 1}
                ]
            },
            "transitions": [
                {"from": ["Idle"], "to": ["Walking"], "event": "WALK"},
                {"from": ["Idle"], "to": ["Running"], "event": "RUN"},
                {"from": ["Idle"], "to": ["Jumping"], "event": "JUMP"},
                {"from": ["Idle"], "to": ["Attacking"], "event": "ATTACK"},
                {"from": ["Walking"], "to": ["Idle"], "event": "STOP"},
                {"from": ["Walking"], "to": ["Running"], "event": "RUN"},
                {"from": ["Running"], "to": ["Walking"], "event": "WALK"},
                {"from": ["Running"], "to": ["Idle"], "event": "STOP"},
                {"from": ["Jumping"], "to": ["Idle"], "event": "LAND"},
                {"from": ["Attacking"], "to": ["Idle"], "event": "DONE"}
            ]
        },
        "description": "Game character states"
    },

    # === ADVANCED: Hierarchical and Parallel SCs ===
    # These test Harel semantics - may fail with naive flat model

    # 6. Hierarchical SC (composite states with default substates)
    {
        "name": "HierarchicalPower",
        "class_name": "HierarchicalPower",
        "sc": {
            "root_state": {
                "label": "__root__",
                "type": 2,
                "children": [
                    {"label": "Off", "type": 1, "is_initial": True},
                    {
                        "label": "On",
                        "type": 2,  # OR-state (composite)
                        "children": [
                            {"label": "Low", "type": 1, "is_initial": True},
                            {"label": "High", "type": 1}
                        ]
                    }
                ]
            },
            "transitions": [
                {"from": ["Off"], "to": ["On"], "event": "POWER"},
                {"from": ["On"], "to": ["Off"], "event": "POWER"},
                {"from": ["Low"], "to": ["High"], "event": "BOOST"},
                {"from": ["High"], "to": ["Low"], "event": "REDUCE"}
            ]
        },
        "description": "Hierarchical: On has Low/High substates",
        "advanced": True,
    },

    # 7. Parallel SC (AND-state with orthogonal regions)
    {
        "name": "ParallelPlayer",
        "class_name": "ParallelPlayer",
        "sc": {
            "root_state": {
                "label": "__root__",
                "type": 2,
                "children": [
                    {
                        "label": "Active",
                        "type": 3,  # AND-state (parallel)
                        "is_initial": True,
                        "children": [
                            {
                                "label": "Movement",
                                "type": 2,
                                "children": [
                                    {"label": "Standing", "type": 1, "is_initial": True},
                                    {"label": "Walking", "type": 1}
                                ]
                            },
                            {
                                "label": "Combat",
                                "type": 2,
                                "children": [
                                    {"label": "Idle", "type": 1, "is_initial": True},
                                    {"label": "Attacking", "type": 1}
                                ]
                            }
                        ]
                    }
                ]
            },
            "transitions": [
                {"from": ["Standing"], "to": ["Walking"], "event": "WALK"},
                {"from": ["Walking"], "to": ["Standing"], "event": "STOP"},
                {"from": ["Idle"], "to": ["Attacking"], "event": "ATTACK"},
                {"from": ["Attacking"], "to": ["Idle"], "event": "DONE"}
            ]
        },
        "description": "Parallel: Movement + Combat regions",
        "advanced": True,
    },
]

# Separate flat and advanced tests
FLAT_STATECHARTS = [sc for sc in TEST_STATECHARTS if not sc.get("advanced")]
ADVANCED_STATECHARTS = [sc for sc in TEST_STATECHARTS if sc.get("advanced")]


@dataclass
class BenchmarkResult:
    """Result for a single benchmark test."""
    name: str
    syntax_valid: bool
    runnable: bool
    behavior_correct: bool
    gen_time_s: float
    errors: List[str] = field(default_factory=list)
    generated_code: str = ""


@dataclass
class BenchmarkSummary:
    """Summary of all benchmark results."""
    total: int
    syntax_valid: int
    runnable: int
    correct: int
    syntax_rate: float
    runnable_rate: float
    correct_rate: float
    results: List[BenchmarkResult] = field(default_factory=list)


class SCCodeBenchmark:
    """Benchmark for SC to code generation."""

    def __init__(self, model_id: str = "mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit"):
        self.model_id = model_id
        self.generator = None
        self.validator = CodeValidator()

    def run(
        self,
        verbose: bool = True,
        include_advanced: bool = False,
    ) -> BenchmarkSummary:
        """
        Run the benchmark on test statecharts.

        Args:
            verbose: Print progress
            include_advanced: Include hierarchical/parallel SCs (may fail with flat model)
        """
        # Select test set
        if include_advanced:
            test_set = TEST_STATECHARTS
            mode = "ALL (flat + advanced)"
        else:
            test_set = FLAT_STATECHARTS
            mode = "FLAT only"

        if verbose:
            print("=" * 70)
            print("SC TO CODE BENCHMARK")
            print("=" * 70)
            print(f"Model: {self.model_id}")
            print(f"Mode: {mode}")
            print(f"Test cases: {len(test_set)}")
            print("-" * 70)

        # Load model once
        self.generator = SCCodeGenerator(model_id=self.model_id)
        self.generator.load_model()

        results = []

        for i, test in enumerate(test_set, 1):
            name = test["name"]
            class_name = test["class_name"]
            sc = test["sc"]
            desc = test["description"]

            if verbose:
                print(f"\n{i}. {name}: {desc}")

            # Generate code
            gen_result = self.generator.generate(sc, class_name, max_tokens=600)

            if verbose:
                print(f"   Generated in {gen_result.gen_time_s:.2f}s")

            # Validate code
            val_result = self.validator.validate(
                gen_result.generated_code, sc, class_name
            )

            result = BenchmarkResult(
                name=name,
                syntax_valid=val_result.syntax_valid,
                runnable=val_result.runnable,
                behavior_correct=val_result.behavior_correct,
                gen_time_s=gen_result.gen_time_s,
                errors=val_result.errors,
                generated_code=gen_result.generated_code,
            )
            results.append(result)

            if verbose:
                syn = "PASS" if result.syntax_valid else "FAIL"
                run = "PASS" if result.runnable else "FAIL"
                cor = "PASS" if result.behavior_correct else "FAIL"
                print(f"   Syntax: {syn} | Runnable: {run} | Correct: {cor}")
                if result.errors:
                    for err in result.errors[:2]:  # Show first 2 errors
                        print(f"   Error: {err[:60]}...")

        # Compute summary
        total = len(results)
        syntax_valid = sum(1 for r in results if r.syntax_valid)
        runnable = sum(1 for r in results if r.runnable)
        correct = sum(1 for r in results if r.behavior_correct)

        summary = BenchmarkSummary(
            total=total,
            syntax_valid=syntax_valid,
            runnable=runnable,
            correct=correct,
            syntax_rate=syntax_valid / total * 100 if total else 0,
            runnable_rate=runnable / total * 100 if total else 0,
            correct_rate=correct / total * 100 if total else 0,
            results=results,
        )

        if verbose:
            print("\n" + "=" * 70)
            print("SUMMARY")
            print("=" * 70)
            print(f"Syntax Valid:     {syntax_valid}/{total} ({summary.syntax_rate:.0f}%)")
            print(f"Runnable:         {runnable}/{total} ({summary.runnable_rate:.0f}%)")
            print(f"Behavior Correct: {correct}/{total} ({summary.correct_rate:.0f}%)")
            print()
            print("Results by SC type:")
            for r in results:
                syn = "syntax" if r.syntax_valid else "------"
                run = "run" if r.runnable else "---"
                cor = "correct" if r.behavior_correct else "-------"
                print(f"  - {r.name}: {syn}/{run}/{cor}")

            # Check targets
            target_runnable = 60
            target_correct = 50
            runnable_met = summary.runnable_rate >= target_runnable
            correct_met = summary.correct_rate >= target_correct

            print()
            print(f"Target runnable (60%): {'MET' if runnable_met else 'MISSED'} ({summary.runnable_rate:.0f}%)")
            print(f"Target correct (50%):  {'MET' if correct_met else 'MISSED'} ({summary.correct_rate:.0f}%)")

        return summary


def run_benchmark(
    model_id: str = "mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit",
    verbose: bool = True,
    include_advanced: bool = False,
) -> BenchmarkSummary:
    """
    Convenience function to run the benchmark.

    Args:
        model_id: MLX model to use
        verbose: Print progress
        include_advanced: Include hierarchical/parallel SCs
    """
    benchmark = SCCodeBenchmark(model_id=model_id)
    return benchmark.run(verbose=verbose, include_advanced=include_advanced)


def demo_single(test_index: int = 0) -> None:
    """Demo code generation for a single test case."""
    if test_index >= len(TEST_STATECHARTS):
        print(f"Invalid index. Max: {len(TEST_STATECHARTS) - 1}")
        return

    test = TEST_STATECHARTS[test_index]

    print("=" * 60)
    print(f"DEMO: {test['name']}")
    print("=" * 60)

    print(f"\n--- Statechart Definition ---")
    print(json.dumps(test["sc"], indent=2))

    generator = SCCodeGenerator()
    result = generator.generate(test["sc"], test["class_name"])

    print(f"\n--- Generated Code ({result.gen_time_s:.2f}s) ---")
    print(result.generated_code)

    validator = CodeValidator()
    val_result = validator.validate(result.generated_code, test["sc"], test["class_name"])

    print(f"\n--- Validation ---")
    print(f"Syntax valid: {val_result.syntax_valid}")
    print(f"Runnable: {val_result.runnable}")
    print(f"Behavior correct: {val_result.behavior_correct}")
    if val_result.errors:
        print(f"Errors: {val_result.errors}")
    if val_result.transition_tests:
        print("Transition tests:")
        for t in val_result.transition_tests:
            status = "PASS" if t["passed"] else "FAIL"
            print(f"  {status}: {t['test']}")


if __name__ == "__main__":
    run_benchmark()
