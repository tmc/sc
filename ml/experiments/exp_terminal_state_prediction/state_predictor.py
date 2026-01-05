"""
State Predictor - LLM-based prediction of terminal states.

Uses Qwen model to predict the final configuration after executing
an event sequence on a statechart.
"""

import json
import re
from typing import Set, List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

try:
    from mlx_lm import load, generate
    import mlx.core as mx
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False


@dataclass
class PredictionResult:
    """Result of a terminal state prediction."""
    predicted_states: Set[str]
    raw_output: str
    parse_success: bool
    confidence: float = 0.0


class TerminalStatePredictor:
    """Predicts final configuration using LLM."""

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
            self.model, self.tokenizer = load(self.model_name)

    def _format_sc_summary(self, sc_json: dict) -> str:
        """Format statechart as concise summary for prompt."""
        states = []
        transitions = []

        def collect_states(state, depth=0):
            label = state.get("label", "")
            if label and not label.startswith("__"):
                state_type = state.get("type", 1)
                type_str = {1: "basic", 2: "composite", 3: "parallel"}.get(state_type, "")
                initial = " (initial)" if state.get("is_initial") else ""
                states.append(f"{'  '*depth}{label}: {type_str}{initial}")
            for child in state.get("children", []):
                collect_states(child, depth + 1)

        collect_states(sc_json.get("root_state", {}))

        for t in sc_json.get("transitions", []):
            from_s = t.get("from", [])
            to_s = t.get("to", [])
            event = t.get("event", "?")
            transitions.append(f"{from_s} --[{event}]--> {to_s}")

        return f"States:\n" + "\n".join(states) + "\n\nTransitions:\n" + "\n".join(transitions)

    def _build_prompt(
        self,
        sc_json: dict,
        initial_state: str,
        events: List[str],
    ) -> str:
        """Build the prediction prompt with few-shot example."""
        # Format transitions as simple rules
        rules = []
        for t in sc_json.get("transitions", []):
            from_s = t.get("from", ["?"])[0]
            to_s = t.get("to", ["?"])[0]
            event = t.get("event", "?")
            rules.append(f"{from_s} + {event} -> {to_s}")

        rules_str = "\n".join(rules)

        # Check for composite/parallel states
        has_hierarchy = any(
            state.get("type") == 2 and state.get("children")
            for state in self._collect_all_states(sc_json)
        )
        has_parallel = any(
            state.get("type") == 3
            for state in self._collect_all_states(sc_json)
        )

        # Base example
        prompt = f"""Execute state transitions step by step.

Example:
Rules: X + GO -> Y, Y + GO -> Z
Start: X, Events: GO, GO
Step 1: X + GO -> Y (now at Y)
Step 2: Y + GO -> Z (now at Z)
Final: Z"""

        # Add hierarchy clarification if needed
        if has_hierarchy:
            prompt += """

Note: When entering a composite state, enter its initial substate."""

        # Add parallel clarification if needed
        if has_parallel:
            prompt += """

Note: When entering a parallel state, list all active substates separated by comma."""

        prompt += f"""

Now solve:
Rules:
{rules_str}

Start: {initial_state}
Events: {', '.join(events)}

Trace each step and give the final state(s):"""

        return prompt

    def _collect_all_states(self, sc_json: dict) -> List[dict]:
        """Collect all states from SC JSON."""
        states = []
        def collect(state):
            states.append(state)
            for child in state.get("children", []):
                collect(child)
        collect(sc_json.get("root_state", {}))
        return states

    def predict(
        self,
        sc_json: dict,
        initial_state: str,
        events: List[str],
        max_tokens: int = 200,
    ) -> PredictionResult:
        """Predict the final configuration after executing events."""
        self.load_model()

        if not self.model or not self.tokenizer:
            # Fallback: simple rule-based prediction
            return self._fallback_predict(sc_json, initial_state, events)

        prompt = self._build_prompt(sc_json, initial_state, events)

        # Format for chat model
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
        predicted_states, parse_success = self._parse_output(output, sc_json)

        return PredictionResult(
            predicted_states=predicted_states,
            raw_output=output,
            parse_success=parse_success,
        )

    def _parse_output(
        self,
        output: str,
        sc_json: dict,
    ) -> Tuple[Set[str], bool]:
        """Parse LLM output to extract predicted states."""
        # Get all valid state names
        valid_states = set()
        def collect_states(state):
            label = state.get("label", "")
            if label and not label.startswith("__"):
                valid_states.add(label)
            for child in state.get("children", []):
                collect_states(child)
        collect_states(sc_json.get("root_state", {}))

        # Clean output
        output = output.strip()
        found_states = set()

        # Method 1: Look for "Final:" line
        final_match = re.search(r'[Ff]inal[:\s]+(\w+)', output)
        if final_match:
            state = final_match.group(1)
            if state in valid_states:
                found_states.add(state)
                return found_states, True

        # Method 2: Look for last mentioned state in parentheses like "(now at X)"
        now_at_matches = re.findall(r'\(now at (\w+)\)', output)
        if now_at_matches:
            last_state = now_at_matches[-1]
            if last_state in valid_states:
                found_states.add(last_state)
                return found_states, True

        # Method 3: Look for last line containing a valid state
        lines = output.strip().split('\n')
        for line in reversed(lines):
            for state in valid_states:
                if re.search(rf'\b{re.escape(state)}\b', line):
                    found_states.add(state)
                    return found_states, True

        # Method 4: First valid state on first line (for short answers)
        first_line = lines[0] if lines else output
        for state in valid_states:
            if re.search(rf'\b{re.escape(state)}\b', first_line):
                found_states.add(state)
                return found_states, True

        # Fallback: any state mentioned
        for state in valid_states:
            if re.search(rf'\b{re.escape(state)}\b', output):
                found_states.add(state)

        parse_success = len(found_states) > 0
        return found_states, parse_success

    def _fallback_predict(
        self,
        sc_json: dict,
        initial_state: str,
        events: List[str],
    ) -> PredictionResult:
        """Fallback prediction using simple execution."""
        from . import TraceExecutor

        executor = TraceExecutor(sc_json)
        final = executor.execute_trace(events)

        return PredictionResult(
            predicted_states=final,
            raw_output="[fallback execution]",
            parse_success=True,
            confidence=1.0,
        )


class BatchPredictor:
    """Batch prediction for efficiency."""

    def __init__(
        self,
        model_name: str = "mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit",
    ):
        self.predictor = TerminalStatePredictor(model_name)

    def predict_batch(
        self,
        test_cases: List,  # List[TestCase]
    ) -> List[PredictionResult]:
        """Predict for multiple test cases."""
        self.predictor.load_model()
        results = []

        for case in test_cases:
            result = self.predictor.predict(
                case.sc_json,
                case.initial_state,
                case.events,
            )
            results.append(result)

        return results


def demo():
    """Demonstrate the state predictor."""
    from . import LINEAR_CHAIN_SC, CYCLE_SC, HIERARCHY_SC, PARALLEL_SC

    print("=" * 60)
    print("Terminal State Predictor Demo")
    print("=" * 60)

    predictor = TerminalStatePredictor()

    test_cases = [
        (LINEAR_CHAIN_SC, "A", ["NEXT", "NEXT"]),
        (CYCLE_SC, "S1", ["TICK", "TICK", "TICK", "TICK"]),
        (HIERARCHY_SC, "Idle", ["START", "PAUSE"]),
        (PARALLEL_SC, "Off", ["POWER", "RUN", "LIGHT"]),
    ]

    for sc, initial, events in test_cases:
        result = predictor.predict(sc, initial, events)
        print(f"\n{sc['name']}: {initial} + {events}")
        print(f"  Predicted: {result.predicted_states}")
        print(f"  Raw output: {result.raw_output[:100]}...")
        print(f"  Parse success: {result.parse_success}")


if __name__ == "__main__":
    demo()
