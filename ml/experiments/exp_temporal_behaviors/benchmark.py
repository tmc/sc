#!/usr/bin/env python3
"""
Benchmark for Temporal Behavior Experiment.

Evaluates:
1. Algorithmic simulation of temporal statecharts
2. LLM-based prediction of temporal behaviors
3. Model generation of temporal statecharts from descriptions

Metrics:
- State sequence prediction accuracy
- Final state prediction accuracy
- Timeout timing accuracy
"""

import json
import time
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional

from .temporal_behavior import (
    TemporalStatechart,
    TemporalSimulator,
    build_temporal_sc,
    predict_temporal_behavior,
)
from .test_cases import TEST_CASES, get_all_scenarios


@dataclass
class BenchmarkResult:
    """Result for one scenario."""
    test_case: str
    scenario: str
    expected_final: str
    predicted_final: str
    correct: bool
    sequence_match: bool
    time_ms: float
    method: str
    details: Dict[str, Any] = field(default_factory=dict)


def run_algorithmic_benchmark(verbose: bool = True) -> Dict[str, Any]:
    """
    Run the algorithmic (simulation-based) benchmark.
    """
    print("=" * 70)
    print("ALGORITHMIC TEMPORAL SIMULATION BENCHMARK")
    print("=" * 70)

    results: List[BenchmarkResult] = []
    by_test_case: Dict[str, List[BenchmarkResult]] = {}

    for tc in TEST_CASES:
        tc_name = tc["name"]
        if verbose:
            print(f"\n[{tc_name}] {tc['description']}")

        by_test_case[tc_name] = []

        for scenario in tc.get("scenarios", []):
            scenario_name = scenario["name"]
            expected_final = scenario.get("expected_final", "")
            expected_sequence = scenario.get("expected_sequence", [])

            start = time.time()

            # Build and simulate
            try:
                sc = build_temporal_sc(tc["statechart"])
                prediction = predict_temporal_behavior(sc, scenario)

                predicted_final = prediction["final_state"]
                predicted_sequence = prediction["state_sequence"]

                # Check correctness
                final_correct = predicted_final == expected_final if expected_final else True
                sequence_correct = (
                    predicted_sequence == expected_sequence
                    if expected_sequence
                    else True
                )

                elapsed_ms = (time.time() - start) * 1000

                result = BenchmarkResult(
                    test_case=tc_name,
                    scenario=scenario_name,
                    expected_final=expected_final,
                    predicted_final=predicted_final,
                    correct=final_correct,
                    sequence_match=sequence_correct,
                    time_ms=elapsed_ms,
                    method="algorithmic",
                    details={
                        "predicted_sequence": predicted_sequence,
                        "expected_sequence": expected_sequence,
                        "history": prediction.get("history", []),
                    },
                )

            except Exception as e:
                elapsed_ms = (time.time() - start) * 1000
                result = BenchmarkResult(
                    test_case=tc_name,
                    scenario=scenario_name,
                    expected_final=expected_final,
                    predicted_final="ERROR",
                    correct=False,
                    sequence_match=False,
                    time_ms=elapsed_ms,
                    method="algorithmic",
                    details={"error": str(e)},
                )

            results.append(result)
            by_test_case[tc_name].append(result)

            if verbose:
                status = "PASS" if result.correct else "FAIL"
                seq_status = "SEQ_OK" if result.sequence_match else "SEQ_DIFF"
                print(f"  [{scenario_name}] {status} {seq_status}")
                if not result.correct:
                    print(f"    Expected: {expected_final}")
                    print(f"    Got:      {result.predicted_final}")

    # Summary
    total = len(results)
    correct = sum(1 for r in results if r.correct)
    seq_correct = sum(1 for r in results if r.sequence_match)
    avg_time = sum(r.time_ms for r in results) / total if total > 0 else 0

    print("\n" + "=" * 70)
    print("ALGORITHMIC SUMMARY")
    print("=" * 70)
    print(f"Final State Accuracy: {correct}/{total} ({correct/total*100:.1f}%)")
    print(f"Sequence Match:       {seq_correct}/{total} ({seq_correct/total*100:.1f}%)")
    print(f"Average Time:         {avg_time:.2f}ms")

    # By test case
    print("\nBy Test Case:")
    for tc_name, tc_results in by_test_case.items():
        n = len(tc_results)
        c = sum(1 for r in tc_results if r.correct)
        print(f"  {tc_name}: {c}/{n} ({c/n*100:.0f}%)")

    return {
        "method": "algorithmic",
        "total": total,
        "correct": correct,
        "accuracy": correct / total if total > 0 else 0,
        "sequence_accuracy": seq_correct / total if total > 0 else 0,
        "avg_time_ms": avg_time,
        "results": results,
        "by_test_case": {
            name: sum(1 for r in tc_res if r.correct) / len(tc_res)
            for name, tc_res in by_test_case.items()
        },
    }


def run_llm_benchmark(verbose: bool = True) -> Dict[str, Any]:
    """
    Run the LLM-based benchmark.

    Tests whether an LLM can predict temporal behaviors.
    """
    print("\n" + "=" * 70)
    print("LLM TEMPORAL PREDICTION BENCHMARK")
    print("=" * 70)

    try:
        from mlx_lm import load, generate
        model_id = "mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit"
        print(f"Loading model: {model_id}")
        model, tokenizer = load(model_id)
    except ImportError:
        print("mlx_lm not available, skipping LLM benchmark")
        return {"method": "llm", "error": "mlx_lm not available"}
    except Exception as e:
        print(f"Failed to load model: {e}")
        return {"method": "llm", "error": str(e)}

    results: List[BenchmarkResult] = []
    by_test_case: Dict[str, List[BenchmarkResult]] = {}

    for tc in TEST_CASES:
        tc_name = tc["name"]
        if verbose:
            print(f"\n[{tc_name}] {tc['description']}")

        by_test_case[tc_name] = []

        for scenario in tc.get("scenarios", []):
            scenario_name = scenario["name"]
            expected_final = scenario.get("expected_final", "")

            # Build prompt for LLM
            sc_spec = tc["statechart"]
            prompt = f"""You are analyzing a temporal statechart.

Statechart:
- States: {sc_spec.get('states', [])}
- Initial: {sc_spec.get('initial', '')}
- Transitions:
"""
            for t in sc_spec.get("transitions", []):
                if t.get("after_ms"):
                    trigger = f"after({t['after_ms']}ms)"
                else:
                    trigger = t.get("event", "")
                guard = f" [{t['guard']}]" if t.get("guard") else ""
                prompt += f"  {t['from']} --{trigger}{guard}--> {t['to']}\n"

            prompt += f"""
Scenario:
- Duration: {scenario.get('duration_ms', 0)}ms
- Events: {scenario.get('events', [])}

What is the FINAL STATE after this scenario?
Answer with just the state name:"""

            start = time.time()

            try:
                response = generate(
                    model,
                    tokenizer,
                    prompt=prompt,
                    max_tokens=20,
                    verbose=False,
                )
                predicted_final = response.strip().split()[0] if response.strip() else ""

                elapsed_ms = (time.time() - start) * 1000
                correct = predicted_final == expected_final

                result = BenchmarkResult(
                    test_case=tc_name,
                    scenario=scenario_name,
                    expected_final=expected_final,
                    predicted_final=predicted_final,
                    correct=correct,
                    sequence_match=False,  # LLM doesn't predict sequence
                    time_ms=elapsed_ms,
                    method="llm",
                    details={"raw_response": response},
                )

            except Exception as e:
                elapsed_ms = (time.time() - start) * 1000
                result = BenchmarkResult(
                    test_case=tc_name,
                    scenario=scenario_name,
                    expected_final=expected_final,
                    predicted_final="ERROR",
                    correct=False,
                    sequence_match=False,
                    time_ms=elapsed_ms,
                    method="llm",
                    details={"error": str(e)},
                )

            results.append(result)
            by_test_case[tc_name].append(result)

            if verbose:
                status = "PASS" if result.correct else "FAIL"
                print(f"  [{scenario_name}] {status} (pred: {result.predicted_final})")

    # Summary
    total = len(results)
    correct = sum(1 for r in results if r.correct)
    avg_time = sum(r.time_ms for r in results) / total if total > 0 else 0

    print("\n" + "=" * 70)
    print("LLM SUMMARY")
    print("=" * 70)
    print(f"Final State Accuracy: {correct}/{total} ({correct/total*100:.1f}%)")
    print(f"Average Time:         {avg_time:.2f}ms")

    # By test case
    print("\nBy Test Case:")
    for tc_name, tc_results in by_test_case.items():
        n = len(tc_results)
        c = sum(1 for r in tc_results if r.correct)
        print(f"  {tc_name}: {c}/{n} ({c/n*100:.0f}%)")

    return {
        "method": "llm",
        "total": total,
        "correct": correct,
        "accuracy": correct / total if total > 0 else 0,
        "avg_time_ms": avg_time,
        "results": results,
        "by_test_case": {
            name: sum(1 for r in tc_res if r.correct) / len(tc_res)
            for name, tc_res in by_test_case.items()
        },
    }


def run_benchmark(include_llm: bool = True, verbose: bool = True) -> Dict[str, Any]:
    """Run full benchmark."""
    algo_results = run_algorithmic_benchmark(verbose=verbose)

    llm_results = {}
    if include_llm:
        llm_results = run_llm_benchmark(verbose=verbose)

    return {
        "algorithmic": algo_results,
        "llm": llm_results,
    }


def format_report(results: Dict[str, Any]) -> str:
    """Format results for orchestrator report."""
    algo = results.get("algorithmic", {})
    llm = results.get("llm", {})

    algo_acc = algo.get("accuracy", 0) * 100
    llm_acc = llm.get("accuracy", 0) * 100 if llm else 0

    # By test case for algorithmic
    by_tc = algo.get("by_test_case", {})
    simple = by_tc.get("simple_timeout", 0) * 100
    reset = by_tc.get("activity_reset", 0) * 100
    backoff = by_tc.get("exponential_backoff", 0) * 100
    deadline = by_tc.get("deadline_race", 0) * 100
    debounce = by_tc.get("debounce", 0) * 100

    return (
        f"TEMPORAL algo={algo_acc:.0f}%, llm={llm_acc:.0f}% | "
        f"simple_after={simple:.0f}%, reset={reset:.0f}%, "
        f"backoff={backoff:.0f}%, deadline={deadline:.0f}%, debounce={debounce:.0f}%"
    )


if __name__ == "__main__":
    import sys

    print("[12FF]: Temporal Behavior Benchmark")
    print()

    include_llm = "--no-llm" not in sys.argv
    results = run_benchmark(include_llm=include_llm, verbose=True)

    print("\n" + "=" * 70)
    print("REPORT")
    print("=" * 70)
    print(format_report(results))
