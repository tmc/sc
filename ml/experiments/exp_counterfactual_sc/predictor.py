"""
Counterfactual State Predictor: Predict next state given current state + event.

Uses Qwen2.5-Coder-0.5B-Instruct to understand statechart execution semantics
and predict state transitions.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional, Any, Tuple
import json
import copy

# MLX imports
try:
    from mlx_lm import load, generate
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False


@dataclass
class PredictionConfig:
    """Configuration for predictor."""
    model_name: str = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
    max_tokens: int = 256
    use_llm: bool = True


@dataclass
class Prediction:
    """A state transition prediction."""
    current_state: str
    event: str
    predicted_next: str
    confidence: float
    explanation: str
    reasoning: List[str] = field(default_factory=list)
    alternatives: List[Tuple[str, float]] = field(default_factory=list)


class CounterfactualPredictor:
    """
    Predict next state given statechart, current state, and event.

    Implements both rule-based (deterministic) and LLM-based prediction.
    """

    def __init__(self, config: Optional[PredictionConfig] = None):
        self.config = config or PredictionConfig()
        self.model = None
        self.tokenizer = None
        self._load_model()

    def _load_model(self):
        """Load Qwen model for LLM-based prediction."""
        if MLX_AVAILABLE and self.config.use_llm:
            try:
                self.model, self.tokenizer = load(self.config.model_name)
            except Exception as e:
                print(f"Warning: Could not load model: {e}")

    def predict(
        self,
        statechart: Dict[str, Any],
        current_state: str,
        event: str,
    ) -> Prediction:
        """
        Predict the next state after event fires.

        Args:
            statechart: Statechart definition
            current_state: Current active state label
            event: Event to fire

        Returns:
            Prediction with next state and explanation
        """
        # First, use deterministic analysis
        transitions = statechart.get("transitions", [])

        # Find matching transitions
        matching = []
        for t in transitions:
            sources = t.get("from", [])
            if current_state in sources and t.get("event") == event:
                targets = t.get("to", [])
                guard = t.get("guard", "")
                matching.append((targets, guard, t))

        # Determine prediction
        if not matching:
            # No transition - stay in current state
            return Prediction(
                current_state=current_state,
                event=event,
                predicted_next=current_state,
                confidence=1.0,
                explanation=f"No transition defined from '{current_state}' on event '{event}'. State remains unchanged.",
                reasoning=[
                    f"Searched transitions for source='{current_state}' and event='{event}'",
                    "No matching transitions found",
                    "By default, state machine stays in current state",
                ],
            )

        if len(matching) == 1:
            # Single unambiguous transition
            targets, guard, trans = matching[0]
            next_state = targets[0] if targets else current_state

            explanation = f"Transition from '{current_state}' to '{next_state}' on event '{event}'."
            reasoning = [
                f"Found transition: {current_state} --({event})--> {next_state}",
            ]

            if guard:
                explanation += f" Guard condition: {guard}."
                reasoning.append(f"Guard '{guard}' assumed to be true")

            return Prediction(
                current_state=current_state,
                event=event,
                predicted_next=next_state,
                confidence=0.95 if not guard else 0.7,
                explanation=explanation,
                reasoning=reasoning,
            )

        # Multiple transitions - need to resolve
        # Use LLM if available, otherwise pick first
        if self.model is not None:
            return self._llm_predict(statechart, current_state, event, matching)

        # Deterministic: pick first (by transition order)
        targets, guard, trans = matching[0]
        next_state = targets[0] if targets else current_state

        alternatives = [
            (t[0][0] if t[0] else current_state, 0.5)
            for t in matching[1:]
        ]

        return Prediction(
            current_state=current_state,
            event=event,
            predicted_next=next_state,
            confidence=0.5,
            explanation=f"Multiple transitions possible. Selected '{next_state}' (first defined).",
            reasoning=[
                f"Found {len(matching)} transitions from '{current_state}' on '{event}'",
                "Selected first by definition order (deterministic fallback)",
            ],
            alternatives=alternatives,
        )

    def _llm_predict(
        self,
        statechart: Dict[str, Any],
        current_state: str,
        event: str,
        matching: List,
    ) -> Prediction:
        """Use LLM to resolve ambiguous transitions."""
        # Build context
        options = []
        for i, (targets, guard, trans) in enumerate(matching):
            tgt = targets[0] if targets else "same"
            guard_str = f" [guard: {guard}]" if guard else ""
            options.append(f"{i+1}. -> {tgt}{guard_str}")

        prompt = f"""Given a state machine in state '{current_state}' receiving event '{event}':

Possible transitions:
{chr(10).join(options)}

Which transition fires? Respond with just the target state name."""

        try:
            messages = [{"role": "user", "content": prompt}]
            formatted = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            response = generate(
                self.model, self.tokenizer,
                prompt=formatted,
                max_tokens=50,
            )

            # Parse response for state name
            response = response.strip()
            for targets, guard, trans in matching:
                tgt = targets[0] if targets else current_state
                if tgt.lower() in response.lower():
                    return Prediction(
                        current_state=current_state,
                        event=event,
                        predicted_next=tgt,
                        confidence=0.7,
                        explanation=f"LLM selected transition to '{tgt}' from {len(matching)} options.",
                        reasoning=[
                            f"Multiple transitions available from '{current_state}'",
                            f"LLM analyzed guards and selected '{tgt}'",
                        ],
                    )
        except Exception as e:
            pass

        # Fallback to first
        targets, guard, trans = matching[0]
        next_state = targets[0] if targets else current_state
        return Prediction(
            current_state=current_state,
            event=event,
            predicted_next=next_state,
            confidence=0.5,
            explanation=f"Defaulted to first transition to '{next_state}'.",
            reasoning=["LLM prediction failed, using first transition"],
        )

    def predict_sequence(
        self,
        statechart: Dict[str, Any],
        initial_state: str,
        events: List[str],
    ) -> List[Prediction]:
        """Predict sequence of state transitions."""
        predictions = []
        current = initial_state

        for event in events:
            pred = self.predict(statechart, current, event)
            predictions.append(pred)
            current = pred.predicted_next

        return predictions

    def explain_why_not(
        self,
        statechart: Dict[str, Any],
        current_state: str,
        event: str,
        expected_state: str,
    ) -> str:
        """Explain why a transition to expected_state didn't happen."""
        pred = self.predict(statechart, current_state, event)

        if pred.predicted_next == expected_state:
            return f"Actually, the transition DOES go to '{expected_state}'."

        reasons = []
        transitions = statechart.get("transitions", [])

        # Check if expected transition exists
        expected_exists = False
        for t in transitions:
            if (current_state in t.get("from", []) and
                expected_state in t.get("to", []) and
                t.get("event") == event):
                expected_exists = True
                guard = t.get("guard", "")
                if guard:
                    reasons.append(f"Transition to '{expected_state}' exists but guard '{guard}' may be false")
                break

        if not expected_exists:
            # Check if any transition from current on event exists
            any_trans = any(
                current_state in t.get("from", []) and t.get("event") == event
                for t in transitions
            )
            if any_trans:
                reasons.append(f"No transition from '{current_state}' to '{expected_state}' on '{event}'")
                reasons.append(f"Transition goes to '{pred.predicted_next}' instead")
            else:
                reasons.append(f"No transition from '{current_state}' on event '{event}'")
                reasons.append(f"State remains '{current_state}'")

        return " ".join(reasons) if reasons else "Unknown reason."


def predict_next_state(
    statechart: Dict[str, Any],
    current_state: str,
    event: str,
) -> Prediction:
    """Convenience function for prediction."""
    predictor = CounterfactualPredictor()
    return predictor.predict(statechart, current_state, event)


def demo():
    """Demonstrate counterfactual prediction."""
    print("=" * 60)
    print("COUNTERFACTUAL PREDICTOR: Predict Next State")
    print("=" * 60)

    # Traffic light example
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

    predictor = CounterfactualPredictor()

    print("\n--- Traffic Light Predictions ---")

    # Test predictions
    tests = [
        ("Red", "TIMER"),
        ("Green", "TIMER"),
        ("Yellow", "TIMER"),
        ("Red", "UNKNOWN_EVENT"),
    ]

    for current, event in tests:
        pred = predictor.predict(traffic_light, current, event)
        print(f"\nCurrent: {current}, Event: {event}")
        print(f"  -> Predicted: {pred.predicted_next} (conf: {pred.confidence:.2f})")
        print(f"  Explanation: {pred.explanation}")

    # Test sequence prediction
    print("\n--- Sequence Prediction ---")
    sequence = predictor.predict_sequence(
        traffic_light,
        "Red",
        ["TIMER", "TIMER", "TIMER", "TIMER"]
    )
    states = ["Red"] + [p.predicted_next for p in sequence]
    print(f"Sequence: {' -> '.join(states)}")

    # Test counterfactual explanation
    print("\n--- Why Not? Explanation ---")
    why = predictor.explain_why_not(traffic_light, "Red", "TIMER", "Yellow")
    print(f"Why not Red->Yellow on TIMER? {why}")

    return predictor


if __name__ == "__main__":
    demo()
