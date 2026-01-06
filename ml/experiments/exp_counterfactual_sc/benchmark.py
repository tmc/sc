"""
Counterfactual Benchmark: Evaluate prediction accuracy and explanation quality.

Tests:
1. Prediction accuracy - does it correctly predict next state?
2. Explanation quality - are explanations clear and accurate?
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
import time

from .predictor import CounterfactualPredictor, PredictionConfig
from .explainer import TransitionExplainer, ExplainerConfig


@dataclass
class BenchmarkCase:
    """A single benchmark test case."""
    name: str
    statechart: Dict[str, Any]
    current_state: str
    event: str
    expected_next: str
    description: str


@dataclass
class BenchmarkResult:
    """Result of running benchmark."""
    total_cases: int
    correct_predictions: int
    prediction_accuracy: float
    explanation_scores: List[float]
    avg_explanation_quality: float
    details: List[Dict[str, Any]]
    inference_time: float


def get_benchmark_cases() -> List[BenchmarkCase]:
    """Get test cases for counterfactual prediction."""
    cases = []

    # === TRAFFIC LIGHT ===
    traffic_light = {
        "name": "Traffic Light",
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
            {"from": ["Red"], "to": ["Green"], "event": "TIMER"},
            {"from": ["Green"], "to": ["Yellow"], "event": "TIMER"},
            {"from": ["Yellow"], "to": ["Red"], "event": "TIMER"},
        ]
    }

    cases.append(BenchmarkCase(
        name="traffic_red_timer",
        statechart=traffic_light,
        current_state="Red",
        event="TIMER",
        expected_next="Green",
        description="Traffic light: Red -> Green on TIMER",
    ))

    cases.append(BenchmarkCase(
        name="traffic_green_timer",
        statechart=traffic_light,
        current_state="Green",
        event="TIMER",
        expected_next="Yellow",
        description="Traffic light: Green -> Yellow on TIMER",
    ))

    cases.append(BenchmarkCase(
        name="traffic_yellow_timer",
        statechart=traffic_light,
        current_state="Yellow",
        event="TIMER",
        expected_next="Red",
        description="Traffic light: Yellow -> Red on TIMER (cycle complete)",
    ))

    cases.append(BenchmarkCase(
        name="traffic_no_event",
        statechart=traffic_light,
        current_state="Red",
        event="UNKNOWN",
        expected_next="Red",
        description="Traffic light: No transition on unknown event",
    ))

    # === DOOR LOCK ===
    door_lock = {
        "name": "Door Lock",
        "root_state": {
            "label": "__root__",
            "type": 2,
            "children": [
                {"label": "Locked", "type": 1, "is_initial": True},
                {"label": "Unlocked", "type": 1},
            ]
        },
        "transitions": [
            {"from": ["Locked"], "to": ["Unlocked"], "event": "CORRECT_CODE"},
            {"from": ["Locked"], "to": ["Locked"], "event": "WRONG_CODE"},
            {"from": ["Unlocked"], "to": ["Locked"], "event": "LOCK"},
            {"from": ["Unlocked"], "to": ["Locked"], "event": "TIMEOUT"},
        ]
    }

    cases.append(BenchmarkCase(
        name="door_unlock",
        statechart=door_lock,
        current_state="Locked",
        event="CORRECT_CODE",
        expected_next="Unlocked",
        description="Door: Locked -> Unlocked on correct code",
    ))

    cases.append(BenchmarkCase(
        name="door_wrong_code",
        statechart=door_lock,
        current_state="Locked",
        event="WRONG_CODE",
        expected_next="Locked",
        description="Door: Stay locked on wrong code",
    ))

    cases.append(BenchmarkCase(
        name="door_relock",
        statechart=door_lock,
        current_state="Unlocked",
        event="LOCK",
        expected_next="Locked",
        description="Door: Unlocked -> Locked on LOCK",
    ))

    cases.append(BenchmarkCase(
        name="door_timeout",
        statechart=door_lock,
        current_state="Unlocked",
        event="TIMEOUT",
        expected_next="Locked",
        description="Door: Auto-lock on timeout",
    ))

    # === ELEVATOR ===
    elevator = {
        "name": "Elevator",
        "root_state": {
            "label": "__root__",
            "type": 2,
            "children": [
                {"label": "Idle", "type": 1, "is_initial": True},
                {"label": "MovingUp", "type": 1},
                {"label": "MovingDown", "type": 1},
                {"label": "DoorsOpen", "type": 1},
            ]
        },
        "transitions": [
            {"from": ["Idle"], "to": ["MovingUp"], "event": "CALL_UP"},
            {"from": ["Idle"], "to": ["MovingDown"], "event": "CALL_DOWN"},
            {"from": ["MovingUp"], "to": ["DoorsOpen"], "event": "ARRIVED"},
            {"from": ["MovingDown"], "to": ["DoorsOpen"], "event": "ARRIVED"},
            {"from": ["DoorsOpen"], "to": ["Idle"], "event": "DOORS_CLOSED"},
        ]
    }

    cases.append(BenchmarkCase(
        name="elevator_call_up",
        statechart=elevator,
        current_state="Idle",
        event="CALL_UP",
        expected_next="MovingUp",
        description="Elevator: Idle -> MovingUp on CALL_UP",
    ))

    cases.append(BenchmarkCase(
        name="elevator_arrived",
        statechart=elevator,
        current_state="MovingUp",
        event="ARRIVED",
        expected_next="DoorsOpen",
        description="Elevator: MovingUp -> DoorsOpen on ARRIVED",
    ))

    cases.append(BenchmarkCase(
        name="elevator_close_doors",
        statechart=elevator,
        current_state="DoorsOpen",
        event="DOORS_CLOSED",
        expected_next="Idle",
        description="Elevator: DoorsOpen -> Idle on DOORS_CLOSED",
    ))

    # === VENDING MACHINE ===
    vending = {
        "name": "Vending Machine",
        "root_state": {
            "label": "__root__",
            "type": 2,
            "children": [
                {"label": "Ready", "type": 1, "is_initial": True},
                {"label": "Selecting", "type": 1},
                {"label": "Dispensing", "type": 1},
            ]
        },
        "transitions": [
            {"from": ["Ready"], "to": ["Selecting"], "event": "COIN_INSERT"},
            {"from": ["Selecting"], "to": ["Dispensing"], "event": "SELECT_ITEM"},
            {"from": ["Selecting"], "to": ["Ready"], "event": "CANCEL"},
            {"from": ["Dispensing"], "to": ["Ready"], "event": "ITEM_DISPENSED"},
        ]
    }

    cases.append(BenchmarkCase(
        name="vending_insert_coin",
        statechart=vending,
        current_state="Ready",
        event="COIN_INSERT",
        expected_next="Selecting",
        description="Vending: Ready -> Selecting on coin insert",
    ))

    cases.append(BenchmarkCase(
        name="vending_select",
        statechart=vending,
        current_state="Selecting",
        event="SELECT_ITEM",
        expected_next="Dispensing",
        description="Vending: Selecting -> Dispensing on item select",
    ))

    cases.append(BenchmarkCase(
        name="vending_cancel",
        statechart=vending,
        current_state="Selecting",
        event="CANCEL",
        expected_next="Ready",
        description="Vending: Cancel returns to Ready",
    ))

    return cases


def evaluate_explanation_quality(
    explanation: str,
    expected_next: str,
    actual_next: str,
    event: str,
) -> float:
    """
    Evaluate explanation quality on 1-5 scale.

    Criteria:
    - Mentions the event (1 point)
    - Mentions the target state (1 point)
    - Is grammatically coherent (1 point)
    - Provides reasoning (1 point)
    - Is concise and clear (1 point)
    """
    score = 0.0

    # Check if event is mentioned
    if event.lower() in explanation.lower():
        score += 1.0

    # Check if target state is mentioned
    if expected_next.lower() in explanation.lower():
        score += 1.0

    # Check for basic coherence (has subject-verb structure)
    if len(explanation) > 10 and "." in explanation:
        score += 1.0

    # Check for reasoning words
    reasoning_words = ["because", "when", "if", "since", "therefore", "transitions", "goes"]
    if any(w in explanation.lower() for w in reasoning_words):
        score += 1.0

    # Check for conciseness (not too long, not too short)
    if 20 <= len(explanation) <= 300:
        score += 1.0

    return score


class CounterfactualBenchmark:
    """Benchmark runner for counterfactual prediction."""

    def __init__(
        self,
        pred_config: Optional[PredictionConfig] = None,
        expl_config: Optional[ExplainerConfig] = None,
    ):
        self.pred_config = pred_config or PredictionConfig()
        self.expl_config = expl_config or ExplainerConfig()

    def run(self) -> BenchmarkResult:
        """Run full benchmark."""
        predictor = CounterfactualPredictor(self.pred_config)
        explainer = TransitionExplainer(self.expl_config)

        cases = get_benchmark_cases()
        results = []
        correct = 0
        explanation_scores = []

        start_time = time.time()

        for case in cases:
            # Make prediction
            pred = predictor.predict(
                case.statechart,
                case.current_state,
                case.event,
            )

            # Check correctness
            is_correct = pred.predicted_next == case.expected_next
            if is_correct:
                correct += 1

            # Evaluate explanation
            expl_score = evaluate_explanation_quality(
                pred.explanation,
                case.expected_next,
                pred.predicted_next,
                case.event,
            )
            explanation_scores.append(expl_score)

            results.append({
                "name": case.name,
                "description": case.description,
                "current": case.current_state,
                "event": case.event,
                "expected": case.expected_next,
                "predicted": pred.predicted_next,
                "correct": is_correct,
                "confidence": pred.confidence,
                "explanation": pred.explanation,
                "explanation_score": expl_score,
            })

        inference_time = time.time() - start_time
        accuracy = correct / len(cases) if cases else 0.0
        avg_expl = sum(explanation_scores) / len(explanation_scores) if explanation_scores else 0.0

        return BenchmarkResult(
            total_cases=len(cases),
            correct_predictions=correct,
            prediction_accuracy=accuracy,
            explanation_scores=explanation_scores,
            avg_explanation_quality=avg_expl,
            details=results,
            inference_time=inference_time,
        )


def run_benchmark() -> BenchmarkResult:
    """Convenience function to run benchmark."""
    benchmark = CounterfactualBenchmark()
    return benchmark.run()


def demo():
    """Run benchmark demo."""
    print("=" * 60)
    print("COUNTERFACTUAL BENCHMARK: Prediction + Explanation")
    print("=" * 60)

    benchmark = CounterfactualBenchmark()
    result = benchmark.run()

    print(f"\n=== RESULTS ===")
    print(f"Total cases: {result.total_cases}")
    print(f"Correct predictions: {result.correct_predictions}")
    print(f"PREDICTION ACCURACY: {result.prediction_accuracy * 100:.1f}%")
    print(f"AVG EXPLANATION QUALITY: {result.avg_explanation_quality:.2f}/5")
    print(f"Inference time: {result.inference_time:.2f}s")

    print(f"\n=== CASE DETAILS ===")
    for detail in result.details:
        status = "✓" if detail["correct"] else "✗"
        print(f"  {status} {detail['name']}: {detail['current']} --({detail['event']})--> "
              f"{detail['predicted']} (expected: {detail['expected']})")
        print(f"      Explanation ({detail['explanation_score']:.1f}/5): {detail['explanation'][:80]}...")

    return result


if __name__ == "__main__":
    demo()
