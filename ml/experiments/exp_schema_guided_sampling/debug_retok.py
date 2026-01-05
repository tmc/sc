#!/usr/bin/env python3
"""Debug why retokenization approach fails."""

import json
from pathlib import Path
import sys

sys.path.insert(0, '/Volumes/tmc/go/src/github.com/tmc/sc/ml')

from experiments.exp_schema_guided_sampling.approaches import RetokenizationApproach
from experiments.exp_schema_guided_sampling.statechart_sampler import (
    StatechartMachine,
    JSONTokenizer,
)


def debug_retok():
    # The issue: Retokenization only checks if appending a token keeps the
    # string valid *so far*. It doesn't check if the resulting state can
    # ever reach completion.
    #
    # Example: After generating many elements, any valid string content is
    # "allowed" but the model keeps generating more and more, never closing.

    sc_path = Path(__file__).parent / "json_parser_v2.statechart.json"
    with open(sc_path) as f:
        sc_def = json.load(f)

    # Simulate the failing output
    failing = '{"root_state": {"label": "__root__", "type": 2, "children": [{"label": "Idle"}, {"label": "Running"}, {"label": "Stopped"}, {"label": "Stopped"}, {"label": "Stopped"}, {"label": "Stopped"}'

    print(f"Failing output: {failing}")
    print(f"Length: {len(failing)}")

    # Check machine state
    machine = StatechartMachine(statechart=sc_def)
    tokenizer = JSONTokenizer()

    for char in failing:
        event = tokenizer.process_char(char)
        if event:
            machine.send_event(event)

    print(f"\nMachine state: {machine.current_state}")
    print(f"Depth: {machine.context.depth}")
    print(f"Stack: {machine.context.stack}")
    print(f"Element count: {machine.context.element_count}")
    print(f"Max elements: {machine.context.max_elements}")

    enabled = machine.get_enabled_events()
    print(f"Enabled events: {enabled}")

    # The problem: The model can still generate valid content (COMMA, etc)
    # but it never chooses to close because:
    # 1. Retok doesn't penalize tokens that delay closure
    # 2. The model prefers generating more content

    # What TokenSimulation does differently:
    # - It simulates the FULL effect of each token
    # - Multi-char tokens like `"},` get properly rejected
    # - The statechart's element_count limit triggers ForceClose

    print("\n--- Problem Analysis ---")
    print("Retokenization approach fails because:")
    print("1. It only validates that the string is parseable SO FAR")
    print("2. It doesn't track statechart extended state (element_count)")
    print("3. It doesn't know when to force close")
    print("")
    print("TokenSimulation succeeds because:")
    print("1. It simulates each token through the statechart")
    print("2. The statechart tracks element_count and triggers ForceClose")
    print("3. Multi-character tokens are fully validated")


if __name__ == "__main__":
    debug_retok()
