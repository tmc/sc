"""
Parallel Code Generation Benchmark.

Tests LLM ability to generate correct Python code for AND-state (parallel) statecharts.
"""

import time
from dataclasses import dataclass
from typing import Dict, List

from .parallel_code_generator import (
    ParallelCodeGenerator,
    ParallelGenResult,
    detect_parallel_regions,
)
from .parallel_validator import (
    ParallelValidator,
    ParallelValidationResult,
)


# Test cases for parallel statecharts
PARALLEL_TEST_CASES = [
    # Test 1: Player (Movement + Combat)
    {
        "name": "Player",
        "sc": {
            "root_state": {
                "label": "__root__",
                "type": 3,  # AND-state
                "children": [
                    {
                        "label": "Movement",
                        "type": 2,
                        "children": [
                            {"label": "Standing", "type": 1, "is_initial": True},
                            {"label": "Walking", "type": 1},
                            {"label": "Running", "type": 1}
                        ]
                    },
                    {
                        "label": "Combat",
                        "type": 2,
                        "children": [
                            {"label": "Idle", "type": 1, "is_initial": True},
                            {"label": "Attacking", "type": 1},
                            {"label": "Defending", "type": 1}
                        ]
                    }
                ]
            },
            "transitions": [
                {"from": ["Standing"], "to": ["Walking"], "event": "WALK"},
                {"from": ["Walking"], "to": ["Running"], "event": "RUN"},
                {"from": ["Running"], "to": ["Walking"], "event": "SLOW"},
                {"from": ["Walking"], "to": ["Standing"], "event": "STOP"},
                {"from": ["Idle"], "to": ["Attacking"], "event": "ATTACK"},
                {"from": ["Attacking"], "to": ["Idle"], "event": "FINISH"},
                {"from": ["Idle"], "to": ["Defending"], "event": "DEFEND"},
                {"from": ["Defending"], "to": ["Idle"], "event": "FINISH"}
            ]
        }
    },
    # Test 2: AudioPlayer (Playback + Volume)
    {
        "name": "AudioPlayer",
        "sc": {
            "root_state": {
                "label": "__root__",
                "type": 3,
                "children": [
                    {
                        "label": "Playback",
                        "type": 2,
                        "children": [
                            {"label": "Stopped", "type": 1, "is_initial": True},
                            {"label": "Playing", "type": 1},
                            {"label": "Paused", "type": 1}
                        ]
                    },
                    {
                        "label": "Volume",
                        "type": 2,
                        "children": [
                            {"label": "Medium", "type": 1, "is_initial": True},
                            {"label": "Low", "type": 1},
                            {"label": "High", "type": 1},
                            {"label": "Muted", "type": 1}
                        ]
                    }
                ]
            },
            "transitions": [
                {"from": ["Stopped"], "to": ["Playing"], "event": "PLAY"},
                {"from": ["Playing"], "to": ["Paused"], "event": "PAUSE"},
                {"from": ["Paused"], "to": ["Playing"], "event": "RESUME"},
                {"from": ["Playing"], "to": ["Stopped"], "event": "STOP"},
                {"from": ["Paused"], "to": ["Stopped"], "event": "STOP"},
                {"from": ["Low"], "to": ["Medium"], "event": "VOL_UP"},
                {"from": ["Medium"], "to": ["High"], "event": "VOL_UP"},
                {"from": ["High"], "to": ["Medium"], "event": "VOL_DOWN"},
                {"from": ["Medium"], "to": ["Low"], "event": "VOL_DOWN"},
                {"from": ["Muted"], "to": ["Medium"], "event": "UNMUTE"},
                {"from": ["Low"], "to": ["Muted"], "event": "MUTE"},
                {"from": ["Medium"], "to": ["Muted"], "event": "MUTE"},
                {"from": ["High"], "to": ["Muted"], "event": "MUTE"}
            ]
        }
    },
    # Test 3: Connection (Network + Auth)
    {
        "name": "Connection",
        "sc": {
            "root_state": {
                "label": "__root__",
                "type": 3,
                "children": [
                    {
                        "label": "Network",
                        "type": 2,
                        "children": [
                            {"label": "Disconnected", "type": 1, "is_initial": True},
                            {"label": "Connecting", "type": 1},
                            {"label": "Connected", "type": 1}
                        ]
                    },
                    {
                        "label": "Auth",
                        "type": 2,
                        "children": [
                            {"label": "LoggedOut", "type": 1, "is_initial": True},
                            {"label": "LoggingIn", "type": 1},
                            {"label": "LoggedIn", "type": 1}
                        ]
                    }
                ]
            },
            "transitions": [
                {"from": ["Disconnected"], "to": ["Connecting"], "event": "CONNECT"},
                {"from": ["Connecting"], "to": ["Connected"], "event": "CONNECTED"},
                {"from": ["Connecting"], "to": ["Disconnected"], "event": "TIMEOUT"},
                {"from": ["Connected"], "to": ["Disconnected"], "event": "DISCONNECT"},
                {"from": ["LoggedOut"], "to": ["LoggingIn"], "event": "LOGIN"},
                {"from": ["LoggingIn"], "to": ["LoggedIn"], "event": "AUTH_SUCCESS"},
                {"from": ["LoggingIn"], "to": ["LoggedOut"], "event": "AUTH_FAIL"},
                {"from": ["LoggedIn"], "to": ["LoggedOut"], "event": "LOGOUT"}
            ]
        }
    }
]


@dataclass
class BenchmarkResult:
    """Result for a single test case."""
    name: str
    gen_result: ParallelGenResult
    val_result: ParallelValidationResult


def run_parallel_benchmark(
    model_id: str = "mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit",
    test_cases: List[Dict] = None,
) -> Dict:
    """Run the parallel code generation benchmark."""
    if test_cases is None:
        test_cases = PARALLEL_TEST_CASES

    print("=" * 70)
    print("PARALLEL CODE GENERATION BENCHMARK")
    print("=" * 70)
    print(f"Model: {model_id}")
    print(f"Test cases: {len(test_cases)}")
    print()

    generator = ParallelCodeGenerator(model_id=model_id)
    validator = ParallelValidator()

    results: List[BenchmarkResult] = []

    for i, tc in enumerate(test_cases):
        name = tc["name"]
        sc = tc["sc"]

        print(f"\n[{i+1}/{len(test_cases)}] {name}")
        print("-" * 50)

        # Detect regions
        regions = detect_parallel_regions(sc)
        print(f"  Regions: {[r[0] for r in regions]}")

        # Generate code
        t0 = time.time()
        gen_result = generator.generate(sc, name)
        gen_time = time.time() - t0
        print(f"  Generation: {gen_time:.1f}s")

        # Validate
        val_result = validator.validate(gen_result.generated_code, sc, name)

        # Print results
        print(f"  Syntax: {'PASS' if val_result.syntax_valid else 'FAIL'}")
        print(f"  Runnable: {'PASS' if val_result.runnable else 'FAIL'}")
        print(f"  Has regions dict: {'PASS' if val_result.has_regions_dict else 'FAIL'}")
        print(f"  Regions: {val_result.regions_correct}/{val_result.regions_total}")
        print(f"  Transitions: {val_result.transitions_correct}/{val_result.transitions_total}")
        print(f"  Overall: {'CORRECT' if val_result.overall_correct else 'INCORRECT'}")

        if val_result.errors:
            for err in val_result.errors:
                print(f"  ERROR: {err}")

        results.append(BenchmarkResult(name, gen_result, val_result))

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    syntax_pass = sum(1 for r in results if r.val_result.syntax_valid)
    runnable_pass = sum(1 for r in results if r.val_result.runnable)
    regions_pass = sum(1 for r in results if r.val_result.has_regions_dict)
    correct_pass = sum(1 for r in results if r.val_result.overall_correct)
    total = len(results)

    print(f"\nSyntax valid:    {syntax_pass}/{total} ({syntax_pass/total*100:.0f}%)")
    print(f"Runnable:        {runnable_pass}/{total} ({runnable_pass/total*100:.0f}%)")
    print(f"Has regions:     {regions_pass}/{total} ({regions_pass/total*100:.0f}%)")
    print(f"Overall correct: {correct_pass}/{total} ({correct_pass/total*100:.0f}%)")

    # Per-case details
    print("\n" + "-" * 70)
    print("Per-case results:")
    for r in results:
        status = "PASS" if r.val_result.overall_correct else "FAIL"
        trans_rate = r.val_result.transitions_accuracy * 100
        print(f"  {r.name}: {status} (regions={r.val_result.has_regions_dict}, trans={trans_rate:.0f}%)")

    return {
        "total": total,
        "syntax_valid": syntax_pass,
        "runnable": runnable_pass,
        "has_regions": regions_pass,
        "overall_correct": correct_pass,
        "syntax_rate": syntax_pass / total if total > 0 else 0,
        "runnable_rate": runnable_pass / total if total > 0 else 0,
        "regions_rate": regions_pass / total if total > 0 else 0,
        "correct_rate": correct_pass / total if total > 0 else 0,
        "results": results,
    }


def run_baseline_comparison():
    """Run baseline (no parallel prompt) vs template prompt comparison."""
    from .code_generator import SCCodeGenerator

    print("=" * 70)
    print("BASELINE vs TEMPLATE COMPARISON")
    print("=" * 70)

    # Use single test case for comparison
    tc = PARALLEL_TEST_CASES[0]  # Player
    name = tc["name"]
    sc = tc["sc"]

    print(f"\nTest case: {name}")

    # Baseline (original generator - no parallel awareness)
    print("\n[BASELINE] Original generator (single self.state):")
    baseline_gen = SCCodeGenerator()
    baseline_result = baseline_gen.generate(sc, name)
    baseline_validator = ParallelValidator()
    baseline_val = baseline_validator.validate(baseline_result.generated_code, sc, name)

    print(f"  Has regions: {baseline_val.has_regions_dict}")
    print(f"  Overall correct: {baseline_val.overall_correct}")

    # Template (parallel-aware generator)
    print("\n[TEMPLATE] Parallel-aware generator (self.regions dict):")
    template_gen = ParallelCodeGenerator()
    template_result = template_gen.generate(sc, name)
    template_val = baseline_validator.validate(template_result.generated_code, sc, name)

    print(f"  Has regions: {template_val.has_regions_dict}")
    print(f"  Overall correct: {template_val.overall_correct}")
    print(f"  Transitions: {template_val.transitions_correct}/{template_val.transitions_total}")

    return {
        "baseline_regions": baseline_val.has_regions_dict,
        "baseline_correct": baseline_val.overall_correct,
        "template_regions": template_val.has_regions_dict,
        "template_correct": template_val.overall_correct,
    }


def main():
    """Run full benchmark."""
    # Run baseline comparison first
    print("\n" + "=" * 70)
    print("PHASE 1: Baseline Comparison")
    print("=" * 70)
    comparison = run_baseline_comparison()

    baseline_pct = 100 if comparison["baseline_correct"] else 0
    template_pct = 100 if comparison["template_correct"] else 0

    print(f"\nBaseline: {baseline_pct}% correct")
    print(f"Template: {template_pct}% correct")

    # Run full benchmark with template generator
    print("\n" + "=" * 70)
    print("PHASE 2: Full Benchmark (Template Generator)")
    print("=" * 70)
    results = run_parallel_benchmark()

    correct_rate = results["correct_rate"] * 100

    # Print final report
    print("\n" + "=" * 70)
    print("REPORT FOR ORCHESTRATOR")
    print("=" * 70)
    print(f"PARALLEL_CODE_GEN baseline=0%, template={correct_rate:.0f}%")
    print(f"  Syntax: {results['syntax_rate']*100:.0f}%")
    print(f"  Runnable: {results['runnable_rate']*100:.0f}%")
    print(f"  Has regions: {results['regions_rate']*100:.0f}%")
    print(f"  Correct: {results['correct_rate']*100:.0f}%")


if __name__ == "__main__":
    main()
