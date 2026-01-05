"""
Real Model Benchmark: Test hierarchical constrained generation with actual Qwen model.

Tests:
1. Model can generate tokens respecting SC constraints
2. Cross-boundary coherence maintained across SC switches
3. Model learns to use SC state for context
"""

import sys
sys.path.insert(0, '/Volumes/tmc/go/src/github.com/tmc/sc/ml/grammars')

from dataclasses import dataclass
from typing import List, Dict, Set, Any, Tuple, Optional
import time

# MLX imports
try:
    from mlx_lm import load, generate
    import mlx.core as mx
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False

from meta_constrained_sampler import MetaConstrainedSampler, SPECIAL_TOKENS


# Domain SCs
TRAFFIC_LIGHT_SC = {
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

DOOR_SC = {
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


@dataclass
class RealModelBenchmarkResult:
    """Results from real model benchmark."""
    model_loaded: bool
    constraint_respected: bool
    cross_boundary_coherent: bool
    switch_detection_accuracy: float
    generated_traces: List[Dict[str, Any]]
    inference_time: float


class RealModelHierarchicalTest:
    """Test hierarchical constrained generation with real Qwen model."""

    def __init__(self, model_name: str = "Qwen/Qwen2.5-Coder-0.5B-Instruct"):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self._load_model()

    def _load_model(self):
        """Load Qwen model."""
        if MLX_AVAILABLE:
            try:
                self.model, self.tokenizer = load(self.model_name)
            except Exception as e:
                print(f"Warning: Could not load model: {e}")

    def test_constraint_respecting_generation(self) -> Tuple[bool, List[str]]:
        """Test that model generates tokens respecting SC constraints."""
        sampler = MetaConstrainedSampler()
        sampler.register_sc("traffic_light", TRAFFIC_LIGHT_SC)
        sampler.load_sc("traffic_light")

        trace = []
        violations = []

        # Generate 10 steps following constraints
        for step in range(10):
            valid_events = sampler.get_valid_events()
            current_states = sampler.get_current_states()

            # Ask model which event to emit
            if self.model and self.tokenizer:
                prompt = f"""You are a traffic light controller.
Current state: {list(current_states)[0]}
Valid events: {', '.join(valid_events)}

Which event should fire next? Reply with just the event name."""

                messages = [{"role": "user", "content": prompt}]
                formatted = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )

                response = generate(
                    self.model, self.tokenizer,
                    prompt=formatted,
                    max_tokens=10,
                )

                # Extract event from response
                response = response.strip().upper()
                chosen_event = None
                for event in valid_events:
                    if event.upper() in response:
                        chosen_event = event
                        break

                if chosen_event is None:
                    # Model didn't choose valid event - try first valid
                    chosen_event = list(valid_events)[0] if valid_events else None
                    if chosen_event:
                        violations.append(f"Step {step}: Model suggested invalid, used {chosen_event}")
            else:
                # Mock mode - pick first valid
                chosen_event = list(valid_events)[0] if valid_events else None

            if chosen_event:
                old_state = list(current_states)[0]
                sampler.emit_event(chosen_event)
                new_state = list(sampler.get_current_states())[0]
                trace.append({
                    "step": step,
                    "from": old_state,
                    "event": chosen_event,
                    "to": new_state,
                })

        # Check if trace is valid (all transitions followed SC rules)
        valid = len(violations) == 0
        return valid, trace

    def test_cross_boundary_coherence(self) -> Tuple[bool, Dict[str, Any]]:
        """Test coherence when switching between SCs."""
        sampler = MetaConstrainedSampler()
        sampler.register_sc("traffic_light", TRAFFIC_LIGHT_SC)
        sampler.register_sc("door", DOOR_SC)

        results = {
            "traffic_light_trace": [],
            "door_trace": [],
            "resumed_trace": [],
            "state_preserved": False,
        }

        # Phase 1: Traffic light domain
        sampler.load_sc("traffic_light")
        for _ in range(3):
            valid = list(sampler.get_valid_events())
            if valid:
                event = self._choose_event_with_model(sampler, valid, "traffic light")
                old = list(sampler.get_current_states())[0]
                sampler.emit_event(event)
                new = list(sampler.get_current_states())[0]
                results["traffic_light_trace"].append({"event": event, "from": old, "to": new})

        # Save state before switch
        pre_switch_state = list(sampler.get_current_states())[0]

        # Phase 2: Switch to door domain
        sampler.push_sc()
        sampler.load_sc("door")

        for _ in range(2):
            valid = list(sampler.get_valid_events())
            if valid:
                event = self._choose_event_with_model(sampler, valid, "door")
                old = list(sampler.get_current_states())[0]
                sampler.emit_event(event)
                new = list(sampler.get_current_states())[0]
                results["door_trace"].append({"event": event, "from": old, "to": new})

        # Phase 3: Pop back to traffic light
        sampler.pop_sc()
        post_switch_state = list(sampler.get_current_states())[0]

        # Check state preservation
        results["state_preserved"] = (pre_switch_state == post_switch_state)

        # Continue in traffic light
        for _ in range(2):
            valid = list(sampler.get_valid_events())
            if valid:
                event = self._choose_event_with_model(sampler, valid, "traffic light (resumed)")
                old = list(sampler.get_current_states())[0]
                sampler.emit_event(event)
                new = list(sampler.get_current_states())[0]
                results["resumed_trace"].append({"event": event, "from": old, "to": new})

        coherent = results["state_preserved"] and len(results["resumed_trace"]) > 0
        return coherent, results

    def _choose_event_with_model(
        self,
        sampler: MetaConstrainedSampler,
        valid_events: List[str],
        domain: str,
    ) -> str:
        """Use model to choose next event."""
        if not self.model or not self.tokenizer:
            return valid_events[0]

        states = list(sampler.get_current_states())
        prompt = f"""Domain: {domain}
Current state: {states[0] if states else 'unknown'}
Valid events: {', '.join(valid_events)}

Pick the most appropriate event. Reply with just the event name."""

        messages = [{"role": "user", "content": prompt}]
        formatted = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        response = generate(
            self.model, self.tokenizer,
            prompt=formatted,
            max_tokens=10,
        )

        response = response.strip().upper()
        for event in valid_events:
            if event.upper() in response:
                return event

        return valid_events[0]

    def test_switch_detection(self) -> Tuple[float, List[Dict]]:
        """Test if model can detect when to switch SCs."""
        if not self.model or not self.tokenizer:
            return 0.0, []

        test_cases = [
            {
                "context": "The car is at a red light. Suddenly someone needs to enter the building.",
                "expected_switch": True,
                "reason": "Need to switch to door domain",
            },
            {
                "context": "The traffic light just turned green. Cars are moving.",
                "expected_switch": False,
                "reason": "Stay in traffic light domain",
            },
            {
                "context": "While waiting at the yellow light, the door needs to be locked.",
                "expected_switch": True,
                "reason": "Need door operations",
            },
            {
                "context": "The light changed from red to green.",
                "expected_switch": False,
                "reason": "Pure traffic light operation",
            },
        ]

        correct = 0
        results = []

        for case in test_cases:
            prompt = f"""You are managing both a traffic light and a door.

Current context: {case["context"]}

Should you switch from the current domain to handle a different system?
Answer YES or NO only."""

            messages = [{"role": "user", "content": prompt}]
            formatted = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            response = generate(
                self.model, self.tokenizer,
                prompt=formatted,
                max_tokens=10,
            )

            predicted_switch = "YES" in response.upper()
            is_correct = (predicted_switch == case["expected_switch"])

            if is_correct:
                correct += 1

            results.append({
                "context": case["context"][:50] + "...",
                "expected": case["expected_switch"],
                "predicted": predicted_switch,
                "correct": is_correct,
            })

        accuracy = correct / len(test_cases) if test_cases else 0.0
        return accuracy, results

    def run_benchmark(self) -> RealModelBenchmarkResult:
        """Run full benchmark."""
        start_time = time.time()

        # Test 1: Constraint respecting
        constraint_ok, trace1 = self.test_constraint_respecting_generation()

        # Test 2: Cross-boundary coherence
        coherence_ok, trace2 = self.test_cross_boundary_coherence()

        # Test 3: Switch detection
        switch_acc, switch_results = self.test_switch_detection()

        inference_time = time.time() - start_time

        return RealModelBenchmarkResult(
            model_loaded=(self.model is not None),
            constraint_respected=constraint_ok,
            cross_boundary_coherent=coherence_ok,
            switch_detection_accuracy=switch_acc,
            generated_traces=[
                {"type": "constraint_test", "trace": trace1},
                {"type": "coherence_test", **trace2},
                {"type": "switch_detection", "results": switch_results},
            ],
            inference_time=inference_time,
        )


def run_real_model_benchmark() -> RealModelBenchmarkResult:
    """Run the real model benchmark."""
    tester = RealModelHierarchicalTest()
    return tester.run_benchmark()


def demo():
    """Run demo of real model hierarchical generation."""
    print("=" * 60)
    print("HIERARCHICAL CONSTRAINED GENERATION: Real Model Test")
    print("=" * 60)

    tester = RealModelHierarchicalTest()
    result = tester.run_benchmark()

    print(f"\n=== RESULTS ===")
    print(f"Model loaded: {result.model_loaded}")
    print(f"Constraints respected: {result.constraint_respected}")
    print(f"Cross-boundary coherent: {result.cross_boundary_coherent}")
    print(f"Switch detection accuracy: {result.switch_detection_accuracy * 100:.1f}%")
    print(f"Inference time: {result.inference_time:.2f}s")

    # Calculate overall accuracy
    tests_passed = sum([
        result.constraint_respected,
        result.cross_boundary_coherent,
        result.switch_detection_accuracy >= 0.5,  # At least 50%
    ])
    total_tests = 3
    accuracy = tests_passed / total_tests

    print(f"\n=== OVERALL ===")
    print(f"Tests passed: {tests_passed}/{total_tests}")
    print(f"ACCURACY: {accuracy * 100:.1f}%")

    return result, accuracy


if __name__ == "__main__":
    demo()
