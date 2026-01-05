#!/usr/bin/env python3
"""Debug trace: Watch statechart state progression through a prompt."""

import json
from pathlib import Path
from experiments.exp_schema_guided_sampling.statechart_sampler import (
    StatechartMachine,
    JSONTokenizer,
)

def trace_prompt(prompt: str):
    """Trace statechart state changes through a prompt."""
    sc_path = Path(__file__).parent / "json_parser_v2.statechart.json"
    with open(sc_path) as f:
        sc_def = json.load(f)

    machine = StatechartMachine(statechart=sc_def)
    tokenizer = JSONTokenizer()

    print(f"Prompt: {prompt}")
    print(f"\nInitial: state={machine.current_state}, depth={machine.context.depth}, stack={machine.context.stack}")
    print("\nProcessing characters:")

    for i, char in enumerate(prompt):
        old_state = machine.current_state
        old_depth = machine.context.depth
        old_stack = machine.context.stack[:]

        # Use proper tokenizer
        event = tokenizer.process_char(char)

        if event:
            enabled = machine.get_enabled_events()
            success = machine.send_event(event)

            # Only print if state or depth changed, or event was significant
            if machine.current_state != old_state or machine.context.depth != old_depth:
                print(f"  [{i:2d}] '{char}' -> {event}")
                print(f"       enabled={enabled}")
                print(f"       {old_state} -> {machine.current_state}")
                print(f"       depth: {old_depth} -> {machine.context.depth}")
                print(f"       stack: {old_stack} -> {machine.context.stack}")
            elif event in ('LBRACE', 'RBRACE', 'LBRACKET', 'RBRACKET', 'NUMBER'):
                print(f"  [{i:2d}] '{char}' -> {event} (no transition, enabled={enabled})")

    print(f"\nFinal: state={machine.current_state}, depth={machine.context.depth}, stack={machine.context.stack}")

    # Show what events are enabled now
    print(f"Enabled events: {machine.get_enabled_events()}")

    return machine, tokenizer


if __name__ == "__main__":
    # Test with the prompts from the benchmark
    prompts = [
        '{"root_state": {"label": "__root__", "type": 2, "children": [{"label": "Idle"}, {"label":',
    ]

    for prompt in prompts:
        print("=" * 70)
        machine, tokenizer = trace_prompt(prompt)
        print("=" * 70)

        # Simulate generating a closing sequence
        print("\nSimulating generation: \"Active\"}]}}")
        gen_text = '"Active"}]}}'
        for char in gen_text:
            old_state = machine.current_state
            event = tokenizer.process_char(char)
            if event:
                enabled = machine.get_enabled_events()
                machine.send_event(event)
                print(f"  '{char}' -> {event}, state={old_state}->{machine.current_state}, depth={machine.context.depth}, stack={machine.context.stack}")

        print(f"\nAfter generation: state={machine.current_state}, depth={machine.context.depth}, stack={machine.context.stack}")
