#!/usr/bin/env python3
"""Debug a single generation step by step."""

import json
import sys
from pathlib import Path

sys.path.insert(0, '/Volumes/tmc/go/src/github.com/tmc/sc/ml')

from experiments.exp_schema_guided_sampling.statechart_sampler import (
    StatechartMachine,
    JSONTokenizer,
)

try:
    import mlx.core as mx
    import numpy as np
    from mlx_lm import load
    from mlx_lm.sample_utils import make_sampler
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False


def debug_generation():
    """Debug a single generation."""
    print("=" * 70)
    print("DEBUG: Step-by-step generation")
    print("=" * 70)

    if not MLX_AVAILABLE:
        print("MLX not available")
        return

    # Load model
    model, tokenizer = load("mlx-community/Qwen2.5-Coder-0.5B-Instruct-4bit")

    # Load statechart
    sc_path = Path(__file__).parent / "json_parser_v2.statechart.json"
    with open(sc_path) as f:
        sc_def = json.load(f)

    # Create machine and tokenizer
    machine = StatechartMachine(statechart=sc_def)
    json_tok = JSONTokenizer()

    # Test prompt
    prompt = '{"root_state": {"label": "__root__", "type": 2, "children": [{"label": "Idle"}, {"label":'

    print(f"\nPrompt: {prompt}")

    # Process prompt
    print("\n--- Processing Prompt ---")
    for char in prompt:
        event = json_tok.process_char(char)
        if event:
            old_state = machine.current_state
            machine.send_event(event)
            if machine.current_state != old_state:
                print(f"  '{char}' -> {event}: {old_state} -> {machine.current_state}, depth={machine.context.depth}")

    print(f"\nAfter prompt: state={machine.current_state}, depth={machine.context.depth}, stack={machine.context.stack}")
    print(f"Enabled events: {machine.get_enabled_events()}")

    # Generate
    print("\n--- Generation (first 20 tokens) ---")
    tokens = mx.array(tokenizer.encode(prompt))[None]
    base_sampler = make_sampler(temp=0.3)

    generated_tokens = []
    for i in range(20):
        logits = model(tokens)
        next_logits = logits[:, -1, :]

        # Get enabled events and mask
        enabled = machine.get_enabled_events()

        # Sample
        next_token = base_sampler(next_logits)
        token_id = next_token.item()

        # Check EOS
        if hasattr(tokenizer, 'eos_token_id') and token_id == tokenizer.eos_token_id:
            print(f"  [{i}] EOS")
            break

        token_str = tokenizer.decode([token_id])
        generated_tokens.append(token_id)

        # Process through statechart
        old_state = machine.current_state
        old_depth = machine.context.depth
        for char in token_str:
            event = json_tok.process_char(char)
            if event:
                machine.send_event(event)

        print(f"  [{i}] token={repr(token_str)}, state={old_state}->{machine.current_state}, depth={old_depth}->{machine.context.depth}, enabled={enabled}")

        if machine.is_complete():
            print(f"  [{i}] COMPLETE!")
            break

        if machine.should_force_close():
            print(f"  [{i}] FORCE CLOSE!")
            break

        tokens = mx.concatenate([tokens, next_token[:, None]], axis=1)

    # Final output
    output = tokenizer.decode(generated_tokens)
    full_text = prompt + output

    print(f"\n--- Final Output ---")
    print(f"Generated: {output}")
    print(f"Full text: {full_text}")
    print(f"Final state: {machine.current_state}, depth={machine.context.depth}, stack={machine.context.stack}")

    # Validate
    open_braces = full_text.count('{') - full_text.count('}')
    open_brackets = full_text.count('[') - full_text.count(']')
    print(f"\nBrace balance: {open_braces} unclosed braces, {open_brackets} unclosed brackets")

    try:
        json.loads(full_text)
        print("Valid JSON: YES")
    except json.JSONDecodeError as e:
        print(f"Valid JSON: NO - {e}")


if __name__ == "__main__":
    debug_generation()
