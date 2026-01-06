"""
State Predictor - Apply induced rules to predict next states.

Uses LLM to apply the induced rules to new inputs.
"""

import re
from typing import List, Any, Optional, Tuple
from dataclasses import dataclass

try:
    from mlx_lm import load, generate
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False

from . import IOStep, InducedMachine


@dataclass
class PredictionResult:
    """Result of state prediction."""
    predicted_state: Any
    raw_output: str
    parse_success: bool


class StatePredictor:
    """Predicts next state using induced rules."""

    def __init__(
        self,
        model_name: str = "mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit",
    ):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None

    def load_model(self):
        """Load the LLM model."""
        if self.model is None and MLX_AVAILABLE:
            print(f"Loading model: {self.model_name}")
            self.model, self.tokenizer = load(self.model_name)
            print("Model loaded.")

    def _build_prediction_prompt(
        self,
        machine: InducedMachine,
        current_state: Any,
        event: str,
        arg: Any = None,
    ) -> str:
        """Build prompt for state prediction."""
        # Format rules
        rules_text = ""
        for rule in machine.rules:
            rules_text += f"  {rule.event}: {rule.action}\n"

        arg_str = f" {arg}" if arg is not None else ""

        prompt = f"""Apply the transition rules to predict the next state.

STATE TYPE: {machine.state_type}
AVAILABLE EVENTS: {machine.events}

RULES:
{rules_text}
CURRENT STATE: {current_state}
EVENT: {event}{arg_str}

Apply the rule for {event} to compute the next state.

Think step by step:
1. Find the rule for event {event}
2. Apply it to current state {current_state}
3. Compute the result

NEXT STATE:"""

        return prompt

    def predict(
        self,
        machine: InducedMachine,
        current_state: Any,
        event: str,
        arg: Any = None,
        max_tokens: int = 150,
    ) -> PredictionResult:
        """Predict next state."""
        self.load_model()

        if not self.model or not self.tokenizer:
            return self._fallback_predict(machine, current_state, event, arg)

        prompt = self._build_prediction_prompt(machine, current_state, event, arg)

        # Format for chat
        messages = [{"role": "user", "content": prompt}]
        formatted = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Generate
        output = generate(
            self.model,
            self.tokenizer,
            prompt=formatted,
            max_tokens=max_tokens,
            verbose=False,
        )

        # Parse output
        predicted, parse_success = self._parse_output(output, machine.state_type)

        return PredictionResult(
            predicted_state=predicted,
            raw_output=output,
            parse_success=parse_success,
        )

    def _parse_output(
        self,
        output: str,
        state_type: str,
    ) -> Tuple[Any, bool]:
        """Parse LLM output to extract predicted state."""
        # Try to find list in output
        if state_type == "list":
            # Look for list patterns like [1, 2, 3] or []
            list_match = re.search(r'\[([^\]]*)\]', output)
            if list_match:
                list_content = list_match.group(1).strip()
                if not list_content:
                    return [], True
                # Parse list elements
                try:
                    elements = []
                    for elem in list_content.split(','):
                        elem = elem.strip().strip('"\'')
                        if elem.isdigit():
                            elements.append(int(elem))
                        elif elem.replace('-', '').isdigit():
                            elements.append(int(elem))
                        elif elem:
                            elements.append(elem)
                    return elements, True
                except Exception:
                    pass

        elif state_type == "int":
            # Look for integer in output
            int_match = re.search(r'(?:=|:|\s)(-?\d+)(?:\s|$|\.)', output)
            if int_match:
                return int(int_match.group(1)), True
            # Try to find any number
            any_int = re.search(r'(-?\d+)', output)
            if any_int:
                return int(any_int.group(1)), True

        elif state_type == "tuple":
            # Look for tuple pattern
            tuple_match = re.search(r'\(([^)]+)\)', output)
            if tuple_match:
                try:
                    # Simple parsing
                    content = tuple_match.group(1)
                    parts = content.split(',')
                    result = []
                    for p in parts:
                        p = p.strip().strip('"\'')
                        if p.isdigit():
                            result.append(int(p))
                        else:
                            result.append(p)
                    return tuple(result), True
                except Exception:
                    pass

        return None, False

    def _fallback_predict(
        self,
        machine: InducedMachine,
        current_state: Any,
        event: str,
        arg: Any = None,
    ) -> PredictionResult:
        """Fallback to heuristic prediction based on rules."""
        # Find matching rule
        for rule in machine.rules:
            if rule.event == event:
                # Try to simulate based on action description
                action = rule.action.lower()

                if machine.state_type == "list":
                    state = list(current_state) if current_state else []

                    if "append" in action or "add" in action or "+" in action:
                        if arg is not None:
                            state.append(arg)
                    elif "remove last" in action or "pop" in action or "[:-1]" in action:
                        if state:
                            state.pop()
                    elif "remove first" in action or "dequeue" in action:
                        if state:
                            state.pop(0)

                    return PredictionResult(state, "[fallback]", True)

                elif machine.state_type == "int":
                    state = current_state if isinstance(current_state, int) else 0

                    if "increment" in action or "+1" in action or "inc" in action:
                        state += 1
                    elif "decrement" in action or "-1" in action or "dec" in action:
                        state = max(0, state - 1)
                    elif "reset" in action or "= 0" in action:
                        state = 0
                    elif "add" in action and arg is not None:
                        state += arg
                    elif "mult" in action and arg is not None:
                        state *= arg

                    return PredictionResult(state, "[fallback]", True)

        # No matching rule
        return PredictionResult(current_state, "[fallback: no rule]", False)


class BatchPredictor:
    """Predicts sequences of states."""

    def __init__(self, predictor: StatePredictor):
        self.predictor = predictor

    def predict_sequence(
        self,
        machine: InducedMachine,
        initial_state: Any,
        steps: List[Tuple[str, Any]],
    ) -> List[PredictionResult]:
        """Predict a sequence of states."""
        results = []
        current = initial_state

        for event, arg in steps:
            result = self.predictor.predict(machine, current, event, arg)
            results.append(result)
            if result.parse_success and result.predicted_state is not None:
                current = result.predicted_state

        return results

    def predict_trace(
        self,
        machine: InducedMachine,
        trace: List[IOStep],
    ) -> List[Tuple[PredictionResult, Any]]:
        """Predict all steps in a trace, compare with actual."""
        results = []
        current = trace[0].state_before if trace else machine.initial_state

        for step in trace:
            result = self.predictor.predict(machine, current, step.input_event, step.input_arg)
            results.append((result, step.state_after))

            # Use predicted state for next step (to test error accumulation)
            if result.parse_success and result.predicted_state is not None:
                current = result.predicted_state
            else:
                # Use actual if prediction failed
                current = step.state_after

        return results


def demo():
    """Demonstrate state prediction."""
    from . import StackMachine, generate_trace
    from .behavior_inducer import BehaviorInducer

    print("=" * 60)
    print("State Predictor Demo")
    print("=" * 60)

    # Generate traces
    machine = StackMachine()
    traces = [generate_trace(machine, 5, seed=i) for i in range(3)]

    # Induce rules
    inducer = BehaviorInducer()
    induction = inducer.induce(traces)

    print(f"\nInduced rules:")
    for rule in induction.machine.rules:
        print(f"  {rule.event}: {rule.action}")

    # Test predictions
    predictor = StatePredictor()

    test_cases = [
        ([], "PUSH", 5),
        ([5], "PUSH", 3),
        ([5, 3], "POP", None),
    ]

    print(f"\nPredictions:")
    for state, event, arg in test_cases:
        result = predictor.predict(induction.machine, state, event, arg)
        arg_str = f" {arg}" if arg is not None else ""
        print(f"  ({event}{arg_str}, {state}) → {result.predicted_state}")


if __name__ == "__main__":
    demo()
