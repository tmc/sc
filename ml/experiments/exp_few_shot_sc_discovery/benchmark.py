#!/usr/bin/env python3
"""
Few-Shot SC Discovery Benchmark

Tests statechart discovery from minimal examples and prediction accuracy.
"""

import json
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

sys.path.insert(0, '/Volumes/tmc/go/src/github.com/tmc/sc/ml')

from experiments.exp_few_shot_sc_discovery.sc_discoverer import (
    SCDiscoverer,
    DiscoveredSC,
    IOExample,
    TEST_CASES,
)
from experiments.exp_few_shot_sc_discovery.predictor import (
    SCPredictor,
    PredictionResult,
)


@dataclass
class TestCase:
    """Complete test case with examples and predictions."""
    name: str
    num_examples: int
    examples: List[IOExample]
    expected_states: Set[str]
    expected_events: Set[str]
    expected_pattern: str
    predictions: List[Tuple[str, str, str, bool]]  # (event, state, expected, is_observed)


@dataclass
class DiscoveryResult:
    """Results of SC discovery for a test case."""
    name: str
    num_examples: int
    states_correct: bool
    events_correct: bool
    transitions_correct: bool
    pattern_correct: bool
    discovered: DiscoveredSC


@dataclass
class BenchmarkResults:
    """Overall benchmark results."""
    discovery_results: List[DiscoveryResult]
    prediction_results: List[PredictionResult]

    @property
    def sc_accuracy(self) -> float:
        correct = sum(1 for r in self.discovery_results
                     if r.states_correct and r.events_correct)
        return correct / len(self.discovery_results) * 100 if self.discovery_results else 0

    @property
    def pred_observed_accuracy(self) -> float:
        observed = [r for r in self.prediction_results if r.is_observed]
        correct = sum(1 for r in observed if r.correct)
        return correct / len(observed) * 100 if observed else 0

    @property
    def pred_novel_accuracy(self) -> float:
        novel = [r for r in self.prediction_results if not r.is_observed]
        correct = sum(1 for r in novel if r.correct)
        return correct / len(novel) * 100 if novel else 0


# Complete test cases with predictions
BENCHMARK_CASES: List[TestCase] = [
    TestCase(
        name="toggle_2ex",
        num_examples=2,
        examples=[
            IOExample("ON", "dark", "lit"),
            IOExample("OFF", "lit", "dark"),
        ],
        expected_states={"dark", "lit"},
        expected_events={"ON", "OFF"},
        expected_pattern="toggle",
        predictions=[
            ("ON", "dark", "lit", True),    # observed
            ("OFF", "lit", "dark", True),   # observed
            ("ON", "lit", "lit", False),    # novel: already lit
            ("OFF", "dark", "dark", False), # novel: already dark
        ],
    ),
    TestCase(
        name="counter_3ex",
        num_examples=3,
        examples=[
            IOExample("INC", "zero", "one"),
            IOExample("INC", "one", "two"),
            IOExample("RESET", "two", "zero"),
        ],
        expected_states={"zero", "one", "two"},
        expected_events={"INC", "RESET"},
        expected_pattern="counter",
        predictions=[
            ("INC", "zero", "one", True),    # observed
            ("INC", "one", "two", True),     # observed
            ("RESET", "two", "zero", True),  # observed
            ("RESET", "one", "zero", False), # novel: reset from one
            ("RESET", "zero", "zero", False),# novel: reset from zero (no-op)
        ],
    ),
    TestCase(
        name="traffic_3ex",
        num_examples=3,
        examples=[
            IOExample("NEXT", "red", "green"),
            IOExample("NEXT", "green", "yellow"),
            IOExample("NEXT", "yellow", "red"),
        ],
        expected_states={"red", "green", "yellow"},
        expected_events={"NEXT"},
        expected_pattern="cycle",
        predictions=[
            ("NEXT", "red", "green", True),    # observed
            ("NEXT", "green", "yellow", True), # observed
            ("NEXT", "yellow", "red", True),   # observed
            # All observed, add novel event
            ("RESET", "yellow", "red", False), # novel: new event
        ],
    ),
    TestCase(
        name="lock_3ex",
        num_examples=3,
        examples=[
            IOExample("LOCK", "unlocked", "locked"),
            IOExample("UNLOCK", "locked", "unlocked"),
            IOExample("OPEN", "unlocked", "open"),
        ],
        expected_states={"unlocked", "locked", "open"},
        expected_events={"LOCK", "UNLOCK", "OPEN"},
        expected_pattern="lock",
        predictions=[
            ("LOCK", "unlocked", "locked", True),   # observed
            ("UNLOCK", "locked", "unlocked", True), # observed
            ("OPEN", "unlocked", "open", True),     # observed
            ("LOCK", "locked", "locked", False),    # novel: already locked
            ("OPEN", "locked", "locked", False),    # novel: can't open locked
        ],
    ),
    TestCase(
        name="player_5ex",
        num_examples=5,
        examples=[
            IOExample("PLAY", "stopped", "playing"),
            IOExample("PAUSE", "playing", "paused"),
            IOExample("PLAY", "paused", "playing"),
            IOExample("STOP", "playing", "stopped"),
            IOExample("STOP", "paused", "stopped"),
        ],
        expected_states={"stopped", "playing", "paused"},
        expected_events={"PLAY", "PAUSE", "STOP"},
        expected_pattern="fsm",
        predictions=[
            ("PLAY", "stopped", "playing", True),  # observed
            ("PAUSE", "playing", "paused", True),  # observed
            ("PLAY", "paused", "playing", True),   # observed
            ("STOP", "playing", "stopped", True),  # observed
            ("STOP", "paused", "stopped", True),   # observed
            ("PAUSE", "paused", "paused", False),  # novel: already paused
            ("PLAY", "playing", "playing", False), # novel: already playing
        ],
    ),
]


def run_benchmark(model=None, tokenizer=None) -> BenchmarkResults:
    """Run the full benchmark."""
    discoverer = SCDiscoverer(model, tokenizer)
    predictor = SCPredictor(model, tokenizer)

    discovery_results: List[DiscoveryResult] = []
    prediction_results: List[PredictionResult] = []

    print("\n" + "=" * 70)
    print("FEW-SHOT SC DISCOVERY BENCHMARK")
    print("=" * 70)

    for case in BENCHMARK_CASES:
        print(f"\n{'=' * 70}")
        print(f"Test: {case.name} ({case.num_examples} examples)")
        print("=" * 70)

        # Show examples
        print("\nTraining examples:")
        for ex in case.examples:
            print(f"  {ex}")

        # Phase 1: Discover SC
        print("\nPhase 1: Discovery...")
        discovered = discoverer.discover(case.examples)

        states_correct = discovered.states == case.expected_states
        events_correct = discovered.events == case.expected_events

        # Check transitions (at least the observed ones)
        trans_correct = True
        for ex in case.examples:
            key = (ex.current_state, ex.event)
            if key not in discovered.transitions:
                trans_correct = False
                break
            if discovered.transitions[key] != ex.next_state:
                trans_correct = False
                break

        pattern_correct = discovered.pattern.lower() == case.expected_pattern.lower()

        disc_result = DiscoveryResult(
            name=case.name,
            num_examples=case.num_examples,
            states_correct=states_correct,
            events_correct=events_correct,
            transitions_correct=trans_correct,
            pattern_correct=pattern_correct,
            discovered=discovered,
        )
        discovery_results.append(disc_result)

        # Print discovery results
        print(f"\nDiscovered:")
        print(f"  States: {sorted(discovered.states)} {'[OK]' if states_correct else '[FAIL]'}")
        print(f"  Events: {sorted(discovered.events)} {'[OK]' if events_correct else '[FAIL]'}")
        print(f"  Transitions: {len(discovered.transitions)} {'[OK]' if trans_correct else '[FAIL]'}")
        print(f"  Pattern: {discovered.pattern} {'[OK]' if pattern_correct else '[FAIL]'}")

        # Phase 2: Predictions
        print("\nPhase 2: Predictions...")
        for event, state, expected, is_observed in case.predictions:
            pred_result = predictor.predict(
                discovered,
                event,
                state,
                expected_state=expected,
                training_examples=case.examples,
            )
            prediction_results.append(pred_result)

            marker = "[OK]" if pred_result.correct else "[FAIL]"
            obs_marker = "OBS" if is_observed else "NOV"
            print(f"  {marker} [{obs_marker}] ({event}, {state}) → {pred_result.predicted_state} (expected: {expected})")

    return BenchmarkResults(
        discovery_results=discovery_results,
        prediction_results=prediction_results,
    )


def print_summary(results: BenchmarkResults) -> Dict[str, float]:
    """Print summary of results."""
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    # Discovery summary
    print("\n--- SC Discovery ---")
    for dr in results.discovery_results:
        status = "OK" if (dr.states_correct and dr.events_correct and dr.transitions_correct) else "PARTIAL"
        print(f"  {dr.name}: {status}")
        if not dr.states_correct:
            print(f"    States: FAIL")
        if not dr.events_correct:
            print(f"    Events: FAIL")
        if not dr.transitions_correct:
            print(f"    Transitions: FAIL")

    # Prediction summary by type
    observed = [r for r in results.prediction_results if r.is_observed]
    novel = [r for r in results.prediction_results if not r.is_observed]

    print("\n--- Predictions ---")
    print(f"  Observed (should be 100%): {sum(1 for r in observed if r.correct)}/{len(observed)} ({results.pred_observed_accuracy:.0f}%)")
    print(f"  Novel (generalization): {sum(1 for r in novel if r.correct)}/{len(novel)} ({results.pred_novel_accuracy:.0f}%)")

    # By number of examples
    print("\n--- By Number of Examples ---")
    by_num = {}
    for dr in results.discovery_results:
        n = dr.num_examples
        if n not in by_num:
            by_num[n] = {"discovery": [], "observed": [], "novel": []}
        by_num[n]["discovery"].append(dr.states_correct and dr.events_correct and dr.transitions_correct)

    for pr in results.prediction_results:
        # Find which case this belongs to
        for dr in results.discovery_results:
            if pr.current_state in dr.discovered.states:
                n = dr.num_examples
                if pr.is_observed:
                    by_num.get(n, {"observed": []})["observed"].append(pr.correct)
                else:
                    by_num.get(n, {"novel": []})["novel"].append(pr.correct)
                break

    for n in sorted(by_num.keys()):
        data = by_num[n]
        disc_acc = sum(data["discovery"]) / len(data["discovery"]) * 100 if data["discovery"] else 0
        obs_acc = sum(data["observed"]) / len(data["observed"]) * 100 if data["observed"] else 0
        nov_acc = sum(data["novel"]) / len(data["novel"]) * 100 if data["novel"] else 0
        print(f"  {n} examples: disc={disc_acc:.0f}%, obs={obs_acc:.0f}%, novel={nov_acc:.0f}%")

    # Overall metrics
    print("\n--- Overall ---")
    print(f"  SC Discovery: {results.sc_accuracy:.0f}%")
    print(f"  Pred (Observed): {results.pred_observed_accuracy:.0f}%")
    print(f"  Pred (Novel): {results.pred_novel_accuracy:.0f}%")

    # Build metrics dict
    metrics = {
        "sc_acc": results.sc_accuracy,
        "pred_observed": results.pred_observed_accuracy,
        "pred_novel": results.pred_novel_accuracy,
        "by_examples": {},
    }

    for n in sorted(by_num.keys()):
        data = by_num[n]
        disc_acc = sum(data["discovery"]) / len(data["discovery"]) * 100 if data["discovery"] else 0
        metrics["by_examples"][f"{n}ex"] = disc_acc

    return metrics


def save_results(results: BenchmarkResults, metrics: Dict):
    """Save results to JSON."""
    results_dir = "/Volumes/tmc/go/src/github.com/tmc/sc/ml/experiments/exp_few_shot_sc_discovery/results"
    os.makedirs(results_dir, exist_ok=True)

    results_path = os.path.join(results_dir, "BENCHMARK_RESULTS.json")

    # Convert to serializable format
    serializable = {
        "session": "DDB5",
        "model": "Qwen2.5-Coder-1.5B-Instruct-4bit",
        "metrics": metrics,
        "discovery": [
            {
                "name": dr.name,
                "num_examples": dr.num_examples,
                "states_correct": dr.states_correct,
                "events_correct": dr.events_correct,
                "transitions_correct": dr.transitions_correct,
                "pattern_correct": dr.pattern_correct,
                "discovered_states": list(dr.discovered.states),
                "discovered_pattern": dr.discovered.pattern,
            }
            for dr in results.discovery_results
        ],
        "predictions": [
            {
                "event": pr.event,
                "state": pr.current_state,
                "predicted": pr.predicted_state,
                "expected": pr.expected_state,
                "is_observed": pr.is_observed,
                "correct": pr.correct,
            }
            for pr in results.prediction_results
        ],
    }

    with open(results_path, "w") as f:
        json.dump(serializable, f, indent=2)

    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    print("[DDB5]: Few-Shot SC Discovery Benchmark")

    try:
        from mlx_lm import load
        print("Loading Qwen2.5-Coder-1.5B-Instruct-4bit...")
        model, tokenizer = load("mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit")
    except Exception as e:
        print(f"Model load failed: {e}")
        model, tokenizer = None, None
        print("Running with mock model")

    results = run_benchmark(model, tokenizer)
    metrics = print_summary(results)
    save_results(results, metrics)

    # Build report string
    by_ex_str = ", ".join(f"{k}:{v:.0f}%" for k, v in metrics.get("by_examples", {}).items())

    print(f"\n[DDB5]: FEW_SHOT_SC_DISCOVERY sc_acc={metrics['sc_acc']:.0f}%, pred_observed={metrics['pred_observed']:.0f}%, pred_novel={metrics['pred_novel']:.0f}%, by_examples=[{by_ex_str}]")
