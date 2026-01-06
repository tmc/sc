"""
State Type Predictor - Template-based classification.

Predicts state types (BASIC/OR/PARALLEL) using few-shot prompting
with clear distinguishing features.
"""

import json
import re
import time
from dataclasses import dataclass
from enum import Enum
from typing import Optional, List


class StateType(Enum):
    """State types in statechart formalism."""
    BASIC = 1      # Leaf state, no children
    OR = 2         # Exclusive children (XOR semantics)
    PARALLEL = 3   # Concurrent children (AND semantics)


@dataclass
class TypePredictionResult:
    """Result of state type prediction."""
    description: str
    expected: StateType
    predicted: Optional[StateType]
    correct: bool
    raw_output: str
    generation_time_ms: float
    error: Optional[str] = None


# Template examples with clear distinguishing features
TYPE_EXAMPLES = '''STATE TYPE CLASSIFICATION

There are exactly 3 state types:

TYPE 1 - BASIC (no children):
- A state with no substates
- Leaf in the hierarchy
- "Idle state" → BASIC
- "Final state" → BASIC
- "Error state with no recovery options" → BASIC

TYPE 2 - OR (mutually exclusive children):
- Children where only ONE can be active at a time
- XOR semantics
- "Power state with On or Off" → OR
- "TrafficLight with Red, Yellow, or Green" → OR
- "Mode selector: either Edit or View" → OR
- Keywords: "or", "either", "one of", "switch between"

TYPE 3 - PARALLEL (concurrent children):
- Children where ALL are active simultaneously
- AND semantics
- "Player with Movement AND Combat regions" → PARALLEL
- "Connection handling Network AND Auth" → PARALLEL
- "Game with Physics, Audio, Input running together" → PARALLEL
- Keywords: "and", "simultaneously", "concurrent", "independent", "parallel"

Examples:

Q: "A simple Waiting state"
A: BASIC (no substates mentioned)

Q: "Status that can be Active or Inactive"
A: OR (mutually exclusive: Active vs Inactive)

Q: "Authentication handling both OAuth and Session"
A: OR (Auth METHOD selection, not concurrent)

Q: "Player state managing Movement, Combat, and Inventory simultaneously"
A: PARALLEL (concurrent independent systems)

Q: "TrafficLight cycling through Red, Yellow, Green"
A: OR (only one color at a time)

Q: "Audio system with Volume control and Mute toggle"
A: PARALLEL (Volume and Mute are independent)

Q: "Game state that is either Playing or Paused"
A: OR (mutually exclusive states)

Q: "Connection with Network and Auth running in parallel"
A: PARALLEL (concurrent subsystems)
'''


class StateTypePredictor:
    """Predicts state types using template-based prompting."""

    def __init__(self, model=None, tokenizer=None):
        self.model = model
        self.tokenizer = tokenizer

    def predict(self, description: str, expected: StateType) -> TypePredictionResult:
        """Predict state type from description."""
        prompt = self._build_prompt(description)

        start = time.time()
        output = self._generate(prompt)
        elapsed_ms = (time.time() - start) * 1000

        predicted = self._extract_type(output)
        correct = predicted == expected if predicted else False

        return TypePredictionResult(
            description=description,
            expected=expected,
            predicted=predicted,
            correct=correct,
            raw_output=output,
            generation_time_ms=elapsed_ms,
            error=None if predicted else "Could not extract type",
        )

    def _build_prompt(self, description: str) -> str:
        """Build prompt with template examples."""
        return f'''{TYPE_EXAMPLES}

Now classify this state:
Q: "{description}"
A:'''

    def _generate(self, prompt: str) -> str:
        """Generate using LLM."""
        if self.model is None:
            return "BASIC"

        try:
            from mlx_lm import generate
            from mlx_lm.sample_utils import make_sampler

            sampler = make_sampler(temp=0.1)
            output = generate(
                self.model,
                self.tokenizer,
                prompt=prompt,
                max_tokens=50,
                sampler=sampler,
            )
            return output
        except Exception as e:
            return f"Error: {e}"

    def _extract_type(self, output: str) -> Optional[StateType]:
        """Extract state type from output."""
        output_upper = output.upper()

        # Check for each type
        if "PARALLEL" in output_upper:
            return StateType.PARALLEL
        elif "OR" in output_upper:
            # Make sure it's not just part of another word
            if re.search(r'\bOR\b', output_upper):
                return StateType.OR
        elif "BASIC" in output_upper:
            return StateType.BASIC

        # Try numbered format
        if "TYPE 3" in output_upper or "TYPE=3" in output_upper:
            return StateType.PARALLEL
        elif "TYPE 2" in output_upper or "TYPE=2" in output_upper:
            return StateType.OR
        elif "TYPE 1" in output_upper or "TYPE=1" in output_upper:
            return StateType.BASIC

        # Fallback: check for keywords in explanation
        if any(kw in output_upper for kw in ["CONCURRENT", "SIMULTANEOUS", "PARALLEL", "BOTH ACTIVE"]):
            return StateType.PARALLEL
        elif any(kw in output_upper for kw in ["EXCLUSIVE", "ONE AT A TIME", "EITHER", "XOR"]):
            return StateType.OR
        elif any(kw in output_upper for kw in ["LEAF", "NO CHILDREN", "NO SUBSTATES", "SIMPLE"]):
            return StateType.BASIC

        return None


# Test cases by type
BASIC_TESTS = [
    ("A simple Idle state", StateType.BASIC),
    ("Final state after completion", StateType.BASIC),
    ("Error state with no recovery", StateType.BASIC),
    ("Waiting state for input", StateType.BASIC),
    ("Start state before initialization", StateType.BASIC),
    ("Terminal failure state", StateType.BASIC),
    ("Ready state listening for events", StateType.BASIC),
]

OR_TESTS = [
    ("Power state with On or Off", StateType.OR),
    ("TrafficLight cycling Red, Yellow, Green", StateType.OR),
    ("Mode: either Edit or View", StateType.OR),
    ("Connection status: Connected, Disconnected, or Connecting", StateType.OR),
    ("Player health: Healthy, Injured, or Dead", StateType.OR),
    ("Door state: Open, Closed, or Locked", StateType.OR),
    ("Authentication: LoggedIn or LoggedOut", StateType.OR),
]

PARALLEL_TESTS = [
    ("Player with Movement and Combat running simultaneously", StateType.PARALLEL),
    ("Game handling Physics, Audio, and Input concurrently", StateType.PARALLEL),
    ("Connection managing Network and Auth in parallel", StateType.PARALLEL),
    ("Editor with Syntax and Autocomplete independent systems", StateType.PARALLEL),
    ("Robot with Navigation and Sensor processing together", StateType.PARALLEL),
    ("App running UI and Background tasks simultaneously", StateType.PARALLEL),
    ("Media player with Video and Audio streams concurrent", StateType.PARALLEL),
]

# Edge cases / ambiguous
EDGE_TESTS = [
    # Should be OR (method selection, not concurrent)
    ("Authentication supporting OAuth or Password login", StateType.OR),
    # Should be OR (phases are sequential)
    ("Build process: Compile, Link, or Test phase", StateType.OR),
    # Should be PARALLEL (independent monitoring)
    ("System monitor tracking CPU and Memory usage", StateType.PARALLEL),
    # Should be BASIC (no mention of substates)
    ("Suspended state awaiting resume signal", StateType.BASIC),
    # Should be PARALLEL (independent controls)
    ("Audio with Volume and Balance controls", StateType.PARALLEL),
    # Should be OR (one active view)
    ("Dashboard showing either Graph or Table view", StateType.OR),
]


if __name__ == "__main__":
    print("State Type Predictor Test")
    print("=" * 60)

    predictor = StateTypePredictor()

    print("\nPrompt for: 'Player with Movement and Combat'")
    print("-" * 40)
    prompt = predictor._build_prompt("Player with Movement and Combat running simultaneously")
    print(prompt[:500] + "...")
