"""
SC Predictor - Predict outputs using discovered statechart.

Uses the discovered SC to:
1. Predict outputs for observed (state, event) pairs
2. Predict outputs for unobserved pairs (generalization)
3. Handle ambiguous cases with hypothesis reasoning
"""

import json
import re
import time
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple

from .sc_discoverer import DiscoveredSC, IOExample


@dataclass
class PredictionResult:
    """Result of predicting output for an input."""
    event: str
    current_state: str
    predicted_state: Optional[str]
    expected_state: Optional[str]
    is_observed: bool  # Was this (state, event) in training examples?
    correct: bool
    confidence: float
    reasoning: str
    generation_time_ms: float


PREDICTION_PROMPT = '''STATECHART PREDICTION

Given a discovered statechart, predict the output for a new input.

DISCOVERED SC:
{sc_description}

KNOWN TRANSITIONS:
{transitions}

TASK: Predict output for ({event}, {current_state})

REASONING PROCESS:
1. Check if ({current_state}, {event}) is in known transitions
2. If yes, return the known result
3. If no, reason about what should happen:
   - Self-loop (stay in same state)?
   - Error/blocked (no transition defined)?
   - Infer from pattern (toggle returns to other state, counter increments, etc.)

Think step by step, then output:
PREDICTION: <next_state>
CONFIDENCE: <high/medium/low>

Response:'''


class SCPredictor:
    """Predicts outputs using discovered SC."""

    def __init__(self, model=None, tokenizer=None):
        self.model = model
        self.tokenizer = tokenizer

    def predict(
        self,
        discovered_sc: DiscoveredSC,
        event: str,
        current_state: str,
        expected_state: Optional[str] = None,
        training_examples: Optional[List[IOExample]] = None,
    ) -> PredictionResult:
        """Predict next state given event and current state."""
        # Check if this is an observed transition
        is_observed = (current_state, event) in discovered_sc.transitions

        start = time.time()

        if is_observed:
            # Use known transition
            predicted = discovered_sc.transitions[(current_state, event)]
            reasoning = f"Known transition: ({current_state}, {event}) → {predicted}"
            confidence = 1.0
        else:
            # Need to reason about unobserved case
            predicted, reasoning, confidence = self._predict_novel(
                discovered_sc, event, current_state
            )

        elapsed_ms = (time.time() - start) * 1000

        correct = predicted == expected_state if expected_state else False

        return PredictionResult(
            event=event,
            current_state=current_state,
            predicted_state=predicted,
            expected_state=expected_state,
            is_observed=is_observed,
            correct=correct,
            confidence=confidence,
            reasoning=reasoning,
            generation_time_ms=elapsed_ms,
        )

    def _predict_novel(
        self,
        sc: DiscoveredSC,
        event: str,
        current_state: str,
    ) -> Tuple[Optional[str], str, float]:
        """Predict for unobserved (state, event) pair."""
        # Build prompt for LLM reasoning
        prompt = self._build_prediction_prompt(sc, event, current_state)

        output = self._generate(prompt)

        # Extract prediction from output
        predicted = self._extract_prediction(output, sc, event, current_state)
        confidence = self._extract_confidence(output)

        return predicted, output, confidence

    def _build_prediction_prompt(
        self,
        sc: DiscoveredSC,
        event: str,
        current_state: str,
    ) -> str:
        """Build prompt for novel prediction."""
        sc_desc = f"States: {sorted(sc.states)}\nEvents: {sorted(sc.events)}\nPattern: {sc.pattern}"

        trans_lines = []
        for (state, evt), target in sc.transitions.items():
            trans_lines.append(f"  ({state}, {evt}) → {target}")
        transitions = "\n".join(trans_lines) if trans_lines else "  (none known)"

        return PREDICTION_PROMPT.format(
            sc_description=sc_desc,
            transitions=transitions,
            event=event,
            current_state=current_state,
        )

    def _generate(self, prompt: str) -> str:
        """Generate using LLM."""
        if self.model is None:
            return "PREDICTION: unknown\nCONFIDENCE: low"

        try:
            from mlx_lm import generate
            from mlx_lm.sample_utils import make_sampler

            sampler = make_sampler(temp=0.1)
            output = generate(
                self.model,
                self.tokenizer,
                prompt=prompt,
                max_tokens=200,
                sampler=sampler,
            )
            return output
        except Exception as e:
            return f"Error: {e}"

    def _extract_prediction(
        self,
        output: str,
        sc: DiscoveredSC,
        event: str,
        current_state: str,
    ) -> Optional[str]:
        """Extract predicted state from output."""
        # Look for PREDICTION: <state>
        match = re.search(r'PREDICTION:\s*(\w+)', output, re.IGNORECASE)
        if match:
            predicted = match.group(1).lower()
            # Check if it's a valid state
            states_lower = {s.lower(): s for s in sc.states}
            if predicted in states_lower:
                return states_lower[predicted]
            # Check for partial match
            for state in sc.states:
                if predicted in state.lower() or state.lower() in predicted:
                    return state

        # Fallback: look for any state name in output
        for state in sc.states:
            if state.lower() in output.lower():
                return state

        # Pattern-based fallback
        return self._pattern_fallback(sc, event, current_state)

    def _pattern_fallback(
        self,
        sc: DiscoveredSC,
        event: str,
        current_state: str,
    ) -> Optional[str]:
        """Fallback prediction based on pattern."""
        pattern = sc.pattern.lower()

        if pattern == "toggle":
            # Toggle to the other state
            other_states = sc.states - {current_state}
            if other_states:
                return list(other_states)[0]

        elif pattern == "cycle":
            # Find next in cycle
            # Try to infer order from transitions
            for (state, evt), target in sc.transitions.items():
                if state == current_state and evt == event:
                    return target
            # Self-loop for unhandled
            return current_state

        elif pattern == "counter":
            # INC increments, DEC decrements, RESET goes to initial
            if "RESET" in event.upper():
                return sc.initial_state
            # Self-loop for unhandled
            return current_state

        # Default: self-loop (stay in current state)
        return current_state

    def _extract_confidence(self, output: str) -> float:
        """Extract confidence from output."""
        output_lower = output.lower()
        if "high" in output_lower:
            return 0.9
        elif "medium" in output_lower:
            return 0.6
        elif "low" in output_lower:
            return 0.3
        return 0.5


# Prediction test cases
# Format: (event, current_state, expected_state, is_observed)
TOGGLE_PREDICTIONS = [
    ("ON", "dark", "lit", True),    # observed
    ("OFF", "lit", "dark", True),   # observed
    ("ON", "lit", "lit", False),    # novel: already lit, stay lit?
    ("OFF", "dark", "dark", False), # novel: already dark, stay dark?
]

COUNTER_PREDICTIONS = [
    ("INC", "zero", "one", True),   # observed
    ("INC", "one", "two", True),    # observed
    ("RESET", "two", "zero", True), # observed
    ("RESET", "one", "zero", False), # novel: reset from one
    ("INC", "two", "two", False),   # novel: overflow? stay at two?
]

TRAFFIC_PREDICTIONS = [
    ("NEXT", "red", "green", True),    # observed
    ("NEXT", "green", "yellow", True), # observed
    ("NEXT", "yellow", "red", True),   # observed
    # All transitions observed in this case
]

LOCK_PREDICTIONS = [
    ("LOCK", "unlocked", "locked", True),   # observed
    ("UNLOCK", "locked", "unlocked", True), # observed
    ("OPEN", "unlocked", "open", True),     # observed
    ("LOCK", "locked", "locked", False),    # novel: already locked
    ("OPEN", "locked", "locked", False),    # novel: can't open locked
    ("CLOSE", "open", "unlocked", False),   # novel: new event
]


if __name__ == "__main__":
    print("SC Predictor Test")
    print("=" * 60)
