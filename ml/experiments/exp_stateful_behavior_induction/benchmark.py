"""
Stateful Behavior Induction Benchmark.

Tests LLM's ability to:
1. Induce transition rules from I/O sequences
2. Predict states for seen patterns
3. Generalize to novel inputs
"""

import time
from dataclasses import dataclass
from typing import Dict, List, Any, Tuple

from . import (
    ALL_MACHINES,
    StatefulMachine,
    IOStep,
    generate_trace,
    generate_test_sequences,
    evaluate_predictions,
)
from .behavior_inducer import BehaviorInducer, InductionResult
from .state_predictor import StatePredictor, BatchPredictor


@dataclass
class MachineResult:
    """Results for a single machine type."""
    name: str
    induction_result: InductionResult
    rule_quality: float  # How well rules match expected behavior
    pred_seen_acc: float  # Accuracy on training-like inputs
    pred_novel_acc: float  # Accuracy on novel inputs
    elapsed: float


def evaluate_rules(
    induced: InductionResult,
    machine: StatefulMachine,
) -> float:
    """Evaluate quality of induced rules."""
    if not induced.parse_success:
        return 0.0

    score = 0.0
    checks = 0

    # Check events match
    expected_events = set(machine.events)
    induced_events = set(induced.machine.events)
    if expected_events == induced_events:
        score += 1.0
    elif expected_events & induced_events:  # Partial match
        score += 0.5
    checks += 1

    # Check state type
    sample_state = machine.reset()
    if isinstance(sample_state, list) and induced.machine.state_type == "list":
        score += 1.0
    elif isinstance(sample_state, int) and induced.machine.state_type == "int":
        score += 1.0
    elif isinstance(sample_state, tuple) and induced.machine.state_type == "tuple":
        score += 1.0
    checks += 1

    # Check rules exist for each event
    rule_events = {r.event for r in induced.machine.rules}
    for event in machine.events:
        if event in rule_events:
            score += 1.0
        checks += 1

    return score / checks if checks > 0 else 0.0


def run_single_machine_benchmark(
    machine: StatefulMachine,
    inducer: BehaviorInducer,
    predictor: StatePredictor,
    fast_mode: bool = True,
) -> MachineResult:
    """Run benchmark for a single machine type."""
    print(f"\n--- {machine.name} ---")

    t0 = time.time()

    # Generate training traces (reduced for speed)
    if fast_mode:
        num_train, num_test, steps = 3, 2, 4
    else:
        num_train, num_test, steps = 6, 4, 5

    train_traces, test_traces = generate_test_sequences(
        machine,
        num_train=num_train,
        num_test=num_test,
        steps_per_seq=steps,
    )

    print(f"  Train sequences: {len(train_traces)}")
    print(f"  Test sequences: {len(test_traces)}")

    # Phase 1: Induce rules
    induction = inducer.induce(train_traces)

    print(f"  Induced state type: {induction.machine.state_type}")
    print(f"  Induced events: {induction.machine.events}")
    print(f"  Rules: {len(induction.machine.rules)}")
    for rule in induction.machine.rules:
        print(f"    {rule.event}: {rule.action[:50]}...")

    # Evaluate rule quality
    rule_quality = evaluate_rules(induction, machine)
    print(f"  Rule quality: {rule_quality:.0%}")

    # Phase 2: Spot-check predictions (faster than full trace evaluation)
    # Only test a few steps from each trace to speed up

    seen_predictions = []
    seen_actuals = []
    for trace in train_traces[:2]:  # Use first 2 train traces
        for step in trace[:2]:  # Only first 2 steps per trace
            result = predictor.predict(
                induction.machine,
                step.state_before,
                step.input_event,
                step.input_arg,
            )
            if result.parse_success:
                seen_predictions.append(result.predicted_state)
                seen_actuals.append(step.state_after)

    seen_metrics = evaluate_predictions(seen_predictions, seen_actuals)
    print(f"  Seen pattern accuracy: {seen_metrics['accuracy']:.0%} ({seen_metrics['correct']}/{seen_metrics['total']})")

    # Phase 3: Spot-check novel patterns
    novel_predictions = []
    novel_actuals = []
    for trace in test_traces[:2]:  # Only 2 test traces
        for step in trace[:2]:  # Only first 2 steps
            result = predictor.predict(
                induction.machine,
                step.state_before,
                step.input_event,
                step.input_arg,
            )
            if result.parse_success:
                novel_predictions.append(result.predicted_state)
                novel_actuals.append(step.state_after)

    novel_metrics = evaluate_predictions(novel_predictions, novel_actuals)
    print(f"  Novel pattern accuracy: {novel_metrics['accuracy']:.0%} ({novel_metrics['correct']}/{novel_metrics['total']})")

    elapsed = time.time() - t0
    print(f"  Time: {elapsed:.1f}s")

    return MachineResult(
        name=machine.name,
        induction_result=induction,
        rule_quality=rule_quality,
        pred_seen_acc=seen_metrics['accuracy'],
        pred_novel_acc=novel_metrics['accuracy'],
        elapsed=elapsed,
    )


def run_benchmark(
    model_name: str = "mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit",
    fast_mode: bool = True,
) -> Dict:
    """Run full benchmark."""
    print("=" * 70)
    print("STATEFUL BEHAVIOR INDUCTION BENCHMARK")
    print("=" * 70)
    print(f"Model: {model_name}")
    print(f"Mode: {'fast' if fast_mode else 'full'}")
    print(f"Machines: {[m.name for m in ALL_MACHINES]}")

    inducer = BehaviorInducer(model_name=model_name)
    predictor = StatePredictor(model_name=model_name)

    # Pre-load model once
    inducer.load_model()
    # Share model with predictor
    predictor.model = inducer.model
    predictor.tokenizer = inducer.tokenizer

    results: List[MachineResult] = []

    for machine in ALL_MACHINES:
        machine.reset()  # Ensure clean state
        result = run_single_machine_benchmark(machine, inducer, predictor, fast_mode)
        results.append(result)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    n = len(results)
    avg_rule = sum(r.rule_quality for r in results) / n
    avg_seen = sum(r.pred_seen_acc for r in results) / n
    avg_novel = sum(r.pred_novel_acc for r in results) / n
    avg_time = sum(r.elapsed for r in results) / n

    print(f"\nOverall Metrics:")
    print(f"  Rule induction: {avg_rule:.0%}")
    print(f"  Predict (seen): {avg_seen:.0%}")
    print(f"  Predict (novel): {avg_novel:.0%}")
    print(f"  Avg time: {avg_time:.1f}s")

    print(f"\nBy Machine:")
    by_machine = {}
    for r in results:
        overall = (r.rule_quality + r.pred_seen_acc + r.pred_novel_acc) / 3
        print(f"  {r.name}: rule={r.rule_quality:.0%} seen={r.pred_seen_acc:.0%} novel={r.pred_novel_acc:.0%} | overall={overall:.0%}")
        by_machine[r.name.lower()] = {
            "rule": r.rule_quality,
            "seen": r.pred_seen_acc,
            "novel": r.pred_novel_acc,
            "overall": overall,
        }

    return {
        "avg_rule": avg_rule,
        "avg_seen": avg_seen,
        "avg_novel": avg_novel,
        "by_machine": by_machine,
        "results": results,
    }


def main():
    """Run benchmark and report."""
    results = run_benchmark()

    # Extract metrics for report
    rule_acc = results["avg_rule"]
    seen_acc = results["avg_seen"]
    novel_acc = results["avg_novel"]

    by_machine = results["by_machine"]
    stack_acc = by_machine.get("stack", {}).get("overall", 0)
    queue_acc = by_machine.get("queue", {}).get("overall", 0)
    counter_acc = by_machine.get("counter", {}).get("overall", 0)
    accum_acc = by_machine.get("accumulator", {}).get("overall", 0)

    # Print report
    print("\n" + "=" * 70)
    print("REPORT FOR ORCHESTRATOR")
    print("=" * 70)

    report = f"[9D1B]: STATEFUL_INDUCTION rule_acc={rule_acc:.0%}, pred_seen={seen_acc:.0%}, pred_novel={novel_acc:.0%}, by_machine=[stack:{stack_acc:.0%}, queue:{queue_acc:.0%}, counter:{counter_acc:.0%}, accum:{accum_acc:.0%}]"
    print(report)

    # Send to orchestrator
    orchestrator_sid = "B90CCCD4"
    try:
        import subprocess
        subprocess.run(
            ["it2", "session", "send-text", orchestrator_sid, report],
            capture_output=True,
            timeout=5,
        )
        print(f"\nReport sent to orchestrator {orchestrator_sid}")
    except Exception as e:
        print(f"\nCould not send to orchestrator: {e}")

    return results


if __name__ == "__main__":
    main()
