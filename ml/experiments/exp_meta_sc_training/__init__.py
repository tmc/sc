"""
exp_meta_sc_training: Train model to generate SCs that constrain other models

This is a meta-learning experiment:
1. Model A generates a statechart definition
2. That SC is used to constrain Model B's generation
3. We evaluate SC quality by how well Model B performs

The reward signal for Model A comes from Model B's output quality.
This creates a curriculum for learning to generate good constraint SCs.
"""

import json
import sys
import numpy as np
from typing import List, Dict, Optional, Tuple

sys.path.insert(0, '/Volumes/tmc/go/src/github.com/tmc/sc/ml/grammars')

from dynamic_constrained_sampler import DynamicConstrainedSampler, SCDefinition, SCExecutor


# Target tasks for constrained generation
TARGET_TASKS = {
    "traffic_sequence": {
        "description": "Generate valid traffic light sequence",
        "expected_events": ["GO", "SLOW", "STOP"],
        "valid_patterns": [
            ["GO", "SLOW", "STOP"],
            ["GO", "SLOW", "STOP", "GO", "SLOW", "STOP"]
        ]
    },
    "door_access": {
        "description": "Generate valid door access sequence",
        "expected_events": ["OPEN", "CLOSE", "LOCK", "UNLOCK"],
        "valid_patterns": [
            ["OPEN", "CLOSE"],
            ["LOCK", "UNLOCK", "OPEN", "CLOSE"],
            ["OPEN", "CLOSE", "LOCK"]
        ]
    },
    "counter_ops": {
        "description": "Generate valid counter operations",
        "expected_events": ["INC", "DEC", "RESET"],
        "valid_patterns": [
            ["INC", "INC", "DEC"],
            ["INC", "DEC", "INC", "INC"],
            ["INC", "RESET"]
        ]
    }
}


def create_sc_for_task(task_name: str) -> dict:
    """Create a ground-truth SC for a task."""
    if task_name == "traffic_sequence":
        return {
            "name": "TrafficLight",
            "root_state": {
                "label": "__root__", "type": 2,
                "children": [
                    {"label": "Red", "type": 1, "is_initial": True},
                    {"label": "Green", "type": 1},
                    {"label": "Yellow", "type": 1}
                ]
            },
            "transitions": [
                {"from": ["Red"], "to": ["Green"], "event": "GO"},
                {"from": ["Green"], "to": ["Yellow"], "event": "SLOW"},
                {"from": ["Yellow"], "to": ["Red"], "event": "STOP"}
            ]
        }
    elif task_name == "door_access":
        return {
            "name": "Door",
            "root_state": {
                "label": "__root__", "type": 2,
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
        }
    elif task_name == "counter_ops":
        return {
            "name": "Counter",
            "root_state": {
                "label": "__root__", "type": 2,
                "children": [
                    {"label": "Zero", "type": 1, "is_initial": True},
                    {"label": "Positive", "type": 1}
                ]
            },
            "transitions": [
                {"from": ["Zero"], "to": ["Positive"], "event": "INC"},
                {"from": ["Positive"], "to": ["Positive"], "event": "INC"},
                {"from": ["Positive"], "to": ["Positive"], "event": "DEC"},
                {"from": ["Positive"], "to": ["Zero"], "event": "RESET"}
            ]
        }
    return None


def evaluate_sc_quality(sc_json: dict, task_name: str) -> Dict:
    """
    Evaluate how well an SC constrains generation for a task.

    Metrics:
    - structural_validity: Is the SC well-formed?
    - event_coverage: Does it include expected events?
    - trace_validity: Do generated traces match expected patterns?
    """
    task = TARGET_TASKS.get(task_name, {})
    metrics = {
        "structural_validity": 0.0,
        "event_coverage": 0.0,
        "trace_validity": 0.0,
        "overall": 0.0
    }

    # Check structural validity
    try:
        sc = SCDefinition.from_json(sc_json)
        if sc.states and sc.initial_states:
            metrics["structural_validity"] = 1.0
    except Exception:
        return metrics

    # Check event coverage
    expected_events = set(task.get("expected_events", []))
    if expected_events:
        coverage = len(expected_events & sc.events) / len(expected_events)
        metrics["event_coverage"] = coverage

    # Generate traces and check validity
    sampler = DynamicConstrainedSampler()
    sampler.load_constraint_sc(sc_json)

    valid_traces = 0
    total_traces = 10

    for _ in range(total_traces):
        trace_json = sampler.generate("", max_tokens=5)
        try:
            trace = json.loads(trace_json)
            # Check if trace matches any valid pattern
            for pattern in task.get("valid_patterns", []):
                if trace == pattern[:len(trace)]:
                    valid_traces += 1
                    break
        except json.JSONDecodeError:
            pass

    metrics["trace_validity"] = valid_traces / total_traces

    # Overall score
    metrics["overall"] = (
        metrics["structural_validity"] * 0.3 +
        metrics["event_coverage"] * 0.3 +
        metrics["trace_validity"] * 0.4
    )

    return metrics


def simulate_sc_generation(task_name: str, quality: str = "good") -> dict:
    """
    Simulate Model A generating an SC for a task.

    In real training, this would be actual LLM generation.
    Here we simulate different quality levels.
    """
    if quality == "good":
        return create_sc_for_task(task_name)
    elif quality == "partial":
        # Missing some transitions
        sc = create_sc_for_task(task_name)
        if sc and sc.get("transitions"):
            sc["transitions"] = sc["transitions"][:len(sc["transitions"])//2]
        return sc
    elif quality == "bad":
        # Wrong structure
        return {
            "name": "Bad",
            "root_state": {"label": "__root__", "type": 2, "children": []},
            "transitions": []
        }
    return None


def meta_training_loop(num_iterations: int = 10) -> List[Dict]:
    """
    Simulate meta-training loop:
    1. Generate SC (Model A)
    2. Evaluate with constrained generation (Model B)
    3. Update Model A based on reward
    """
    history = []

    for iteration in range(num_iterations):
        task_name = np.random.choice(list(TARGET_TASKS.keys()))

        # Simulate quality improving over iterations
        if iteration < 3:
            quality = "bad"
        elif iteration < 7:
            quality = "partial"
        else:
            quality = "good"

        # Generate SC
        generated_sc = simulate_sc_generation(task_name, quality)

        # Evaluate
        metrics = evaluate_sc_quality(generated_sc, task_name)

        history.append({
            "iteration": iteration,
            "task": task_name,
            "quality": quality,
            "metrics": metrics
        })

        print(f"Iteration {iteration}: task={task_name}, quality={quality}, overall={metrics['overall']:.2f}")

    return history


def benchmark():
    """Run meta-training benchmark."""
    print("Meta SC Training Experiment")
    print("=" * 50)
    print("\nThis simulates training a model to generate constraint SCs")
    print("that enable high-quality constrained generation.\n")

    history = meta_training_loop()

    # Analyze learning curve
    early_avg = np.mean([h["metrics"]["overall"] for h in history[:3]])
    mid_avg = np.mean([h["metrics"]["overall"] for h in history[3:7]])
    late_avg = np.mean([h["metrics"]["overall"] for h in history[7:]])

    print("\n" + "=" * 50)
    print("Learning Curve (simulated):")
    print(f"  Early (bad quality):    {early_avg:.2f}")
    print(f"  Middle (partial):       {mid_avg:.2f}")
    print(f"  Late (good):            {late_avg:.2f}")

    print("\nNote: Real training would use actual LLM generation")
    print("and gradient updates based on trace quality rewards.")

    return {
        "history": history,
        "learning_curve": {
            "early": early_avg,
            "mid": mid_avg,
            "late": late_avg
        }
    }


if __name__ == "__main__":
    benchmark()
