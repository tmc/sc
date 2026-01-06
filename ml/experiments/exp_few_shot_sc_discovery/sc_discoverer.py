"""
Statechart Discoverer - Induce SC from minimal I/O examples.

Uses scratchpad reasoning to:
1. Extract states from inputs/outputs
2. Identify transitions from examples
3. Hypothesize about unobserved cases
"""

import json
import re
import time
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Set, Tuple


@dataclass
class IOExample:
    """Single I/O example: (event, current_state) -> next_state"""
    event: str
    current_state: str
    next_state: str

    def __str__(self):
        return f"({self.event}, {self.current_state}) → {self.next_state}"


@dataclass
class DiscoveredSC:
    """Statechart discovered from examples."""
    states: Set[str]
    events: Set[str]
    transitions: Dict[Tuple[str, str], str]  # (state, event) -> next_state
    initial_state: Optional[str]
    pattern: str  # e.g., "toggle", "counter", "cycle"
    confidence: float
    raw_reasoning: str


DISCOVERY_PROMPT = '''STATECHART DISCOVERY FROM EXAMPLES

Given I/O examples, discover the underlying statechart.

FORMAT: (EVENT, current_state) → next_state

SCRATCHPAD PROCESS:
Step 1 - Extract states from all inputs and outputs
Step 2 - Extract events from all inputs
Step 3 - Build transition table from examples
Step 4 - Identify pattern (toggle, counter, cycle, etc.)
Step 5 - Output discovered SC as JSON

EXAMPLE 1 - Toggle Switch:
Examples:
  (ON, dark) → lit
  (OFF, lit) → dark

Scratchpad:
Step 1 - States: {dark, lit}
Step 2 - Events: {ON, OFF}
Step 3 - Transitions:
  (dark, ON) → lit
  (lit, OFF) → dark
Step 4 - Pattern: 2-state toggle

SC:
{"states":["dark","lit"],"events":["ON","OFF"],"transitions":{"dark,ON":"lit","lit,OFF":"dark"},"initial":"dark","pattern":"toggle"}

EXAMPLE 2 - Counter:
Examples:
  (INC, zero) → one
  (INC, one) → two
  (RESET, two) → zero

Scratchpad:
Step 1 - States: {zero, one, two}
Step 2 - Events: {INC, RESET}
Step 3 - Transitions:
  (zero, INC) → one
  (one, INC) → two
  (two, RESET) → zero
Step 4 - Pattern: counter with reset

SC:
{"states":["zero","one","two"],"events":["INC","RESET"],"transitions":{"zero,INC":"one","one,INC":"two","two,RESET":"zero"},"initial":"zero","pattern":"counter"}

EXAMPLE 3 - Traffic Light:
Examples:
  (NEXT, red) → green
  (NEXT, green) → yellow
  (NEXT, yellow) → red

Scratchpad:
Step 1 - States: {red, green, yellow}
Step 2 - Events: {NEXT}
Step 3 - Transitions:
  (red, NEXT) → green
  (green, NEXT) → yellow
  (yellow, NEXT) → red
Step 4 - Pattern: cycle (3-state rotation)

SC:
{"states":["red","green","yellow"],"events":["NEXT"],"transitions":{"red,NEXT":"green","green,NEXT":"yellow","yellow,NEXT":"red"},"initial":"red","pattern":"cycle"}

Now discover the statechart from these examples:
'''


class SCDiscoverer:
    """Discovers statecharts from I/O examples using LLM reasoning."""

    def __init__(self, model=None, tokenizer=None):
        self.model = model
        self.tokenizer = tokenizer

    def discover(self, examples: List[IOExample]) -> DiscoveredSC:
        """Discover statechart from I/O examples."""
        prompt = self._build_prompt(examples)

        start = time.time()
        output = self._generate(prompt)
        elapsed_ms = (time.time() - start) * 1000

        # Parse the discovered SC
        discovered = self._parse_discovery(output, examples)
        discovered.raw_reasoning = output

        return discovered

    def _build_prompt(self, examples: List[IOExample]) -> str:
        """Build discovery prompt with examples."""
        examples_str = "\n".join(f"  {ex}" for ex in examples)
        return f'''{DISCOVERY_PROMPT}
Examples:
{examples_str}

Scratchpad:'''

    def _generate(self, prompt: str) -> str:
        """Generate using LLM."""
        if self.model is None:
            return "{}"

        try:
            from mlx_lm import generate
            from mlx_lm.sample_utils import make_sampler

            sampler = make_sampler(temp=0.1)
            output = generate(
                self.model,
                self.tokenizer,
                prompt=prompt,
                max_tokens=500,
                sampler=sampler,
            )
            return output
        except Exception as e:
            return f"Error: {e}"

    def _parse_discovery(self, output: str, examples: List[IOExample]) -> DiscoveredSC:
        """Parse discovered SC from LLM output."""
        # Try to extract JSON
        json_match = re.search(r'\{[^{}]*"states"[^{}]*\}', output, re.DOTALL)

        states = set()
        events = set()
        transitions = {}
        initial = None
        pattern = "unknown"

        if json_match:
            try:
                sc_json = json.loads(json_match.group())
                states = set(sc_json.get("states", []))
                events = set(sc_json.get("events", []))
                initial = sc_json.get("initial")
                pattern = sc_json.get("pattern", "unknown")

                # Parse transitions
                trans_dict = sc_json.get("transitions", {})
                for key, target in trans_dict.items():
                    parts = key.split(",")
                    if len(parts) == 2:
                        state, event = parts[0].strip(), parts[1].strip()
                        transitions[(state, event)] = target
            except json.JSONDecodeError:
                pass

        # Fallback: extract from examples directly
        if not states:
            for ex in examples:
                states.add(ex.current_state)
                states.add(ex.next_state)
                events.add(ex.event)
                transitions[(ex.current_state, ex.event)] = ex.next_state

        # Infer initial state (first observed or alphabetically first)
        if not initial and states:
            # Use first example's current state if likely initial
            if examples:
                initial = examples[0].current_state
            else:
                initial = sorted(states)[0]

        # Infer pattern
        if not pattern or pattern == "unknown":
            pattern = self._infer_pattern(states, events, transitions)

        # Calculate confidence based on how much was extracted
        confidence = 0.5
        if json_match:
            confidence = 0.9
        elif len(transitions) == len(examples):
            confidence = 0.7

        return DiscoveredSC(
            states=states,
            events=events,
            transitions=transitions,
            initial_state=initial,
            pattern=pattern,
            confidence=confidence,
            raw_reasoning=output,
        )

    def _infer_pattern(
        self,
        states: Set[str],
        events: Set[str],
        transitions: Dict[Tuple[str, str], str]
    ) -> str:
        """Infer pattern from discovered SC."""
        n_states = len(states)
        n_events = len(events)

        # 2 states with 2 events -> toggle
        if n_states == 2 and n_events == 2:
            return "toggle"

        # Single event cycling through states -> cycle
        if n_events == 1:
            return "cycle"

        # INC/DEC/RESET events -> counter
        event_names = {e.upper() for e in events}
        if event_names & {"INC", "DEC", "RESET", "INCREMENT", "DECREMENT"}:
            return "counter"

        # LOCK/UNLOCK events -> lock
        if event_names & {"LOCK", "UNLOCK"}:
            return "lock"

        # Multiple states with NEXT -> sequence
        if "NEXT" in event_names:
            return "sequence"

        return "fsm"


# Test cases for discovery
TEST_CASES = {
    "toggle_2ex": {
        "examples": [
            IOExample("ON", "dark", "lit"),
            IOExample("OFF", "lit", "dark"),
        ],
        "expected_states": {"dark", "lit"},
        "expected_events": {"ON", "OFF"},
        "expected_pattern": "toggle",
    },
    "counter_3ex": {
        "examples": [
            IOExample("INC", "zero", "one"),
            IOExample("INC", "one", "two"),
            IOExample("RESET", "two", "zero"),
        ],
        "expected_states": {"zero", "one", "two"},
        "expected_events": {"INC", "RESET"},
        "expected_pattern": "counter",
    },
    "traffic_3ex": {
        "examples": [
            IOExample("NEXT", "red", "green"),
            IOExample("NEXT", "green", "yellow"),
            IOExample("NEXT", "yellow", "red"),
        ],
        "expected_states": {"red", "green", "yellow"},
        "expected_events": {"NEXT"},
        "expected_pattern": "cycle",
    },
    "lock_3ex": {
        "examples": [
            IOExample("LOCK", "unlocked", "locked"),
            IOExample("UNLOCK", "locked", "unlocked"),
            IOExample("OPEN", "unlocked", "open"),
        ],
        "expected_states": {"unlocked", "locked", "open"},
        "expected_events": {"LOCK", "UNLOCK", "OPEN"},
        "expected_pattern": "lock",
    },
    "player_5ex": {
        "examples": [
            IOExample("PLAY", "stopped", "playing"),
            IOExample("PAUSE", "playing", "paused"),
            IOExample("PLAY", "paused", "playing"),
            IOExample("STOP", "playing", "stopped"),
            IOExample("STOP", "paused", "stopped"),
        ],
        "expected_states": {"stopped", "playing", "paused"},
        "expected_events": {"PLAY", "PAUSE", "STOP"},
        "expected_pattern": "fsm",
    },
}


if __name__ == "__main__":
    print("SC Discoverer Test")
    print("=" * 60)

    discoverer = SCDiscoverer()

    for name, case in TEST_CASES.items():
        print(f"\n{name}:")
        examples = case["examples"]
        for ex in examples:
            print(f"  {ex}")
