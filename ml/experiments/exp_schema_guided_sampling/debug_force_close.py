#!/usr/bin/env python3
"""Debug force close behavior."""

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


def check_balanced(text):
    """Check if braces/brackets are balanced."""
    stack = []
    pairs = {'}': '{', ']': '['}
    for c in text:
        if c in '{[':
            stack.append(c)
        elif c in '}]':
            if not stack or stack[-1] != pairs[c]:
                return False
            stack.pop()
    return len(stack) == 0


def debug_force_close():
    """Debug force close with base model."""
    print("=" * 70)
    print("DEBUG: Force close behavior")
    print("=" * 70)

    if not MLX_AVAILABLE:
        print("MLX not available")
        return

    # Load base model
    model, tokenizer = load("mlx-community/Qwen2.5-Coder-0.5B-4bit")

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
    for char in prompt:
        event = json_tok.process_char(char)
        if event:
            machine.send_event(event)

    print(f"\nAfter prompt: state={machine.current_state}, depth={machine.context.depth}")
    print(f"Stack: {machine.context.stack}")
    print(f"Element count: {machine.context.element_count}")
    print(f"Max elements: {machine.context.max_elements}")

    # Generate with force close
    print("\n--- Generation ---")
    tokens = mx.array(tokenizer.encode(prompt))[None]
    base_sampler = make_sampler(temp=0.3)

    generated_tokens = []
    for i in range(200):
        logits = model(tokens)
        next_logits = logits[:, -1, :]

        # Check force close FIRST
        if machine.should_force_close():
            print(f"\n[{i}] FORCE CLOSE TRIGGERED!")
            print(f"    State: {machine.current_state}, depth={machine.context.depth}")
            print(f"    Stack: {machine.context.stack}")
            print(f"    Element count: {machine.context.element_count}")

            # Calculate closing sequence
            current_output = tokenizer.decode(generated_tokens)
            full_text = prompt + current_output

            # Strip trailing comma if present (invalid before closing brace/bracket)
            # Work at string level, then re-encode to preserve other characters
            if current_output.rstrip().endswith(','):
                print("\n    Stripping trailing comma...")
                # Find and remove trailing comma from the string
                stripped = current_output.rstrip()
                if stripped.endswith(','):
                    stripped = stripped[:-1]  # Remove trailing comma
                    # Re-encode the stripped output
                    generated_tokens = list(tokenizer.encode(stripped))
                    current_output = stripped
                    full_text = prompt + current_output
                print(f"    After stripping: ...{current_output[-50:]}")

            print(f"\n    Full text (last 100 chars): ...{full_text[-100:]}")

            # Count unclosed
            open_braces = full_text.count('{') - full_text.count('}')
            open_brackets = full_text.count('[') - full_text.count(']')
            unclosed_string = full_text.count('"') % 2 == 1

            print(f"    Open braces: {open_braces}")
            print(f"    Open brackets: {open_brackets}")
            print(f"    Unclosed string: {unclosed_string}")

            # Build closing
            closing_seq = ''
            if unclosed_string:
                closing_seq += '"'

            remaining_braces = open_braces
            remaining_brackets = open_brackets

            for item in reversed(machine.context.stack):
                if item == "object" and remaining_braces > 0:
                    closing_seq += '}'
                    remaining_braces -= 1
                elif item == "array" and remaining_brackets > 0:
                    closing_seq += ']'
                    remaining_brackets -= 1

            closing_seq += '}' * remaining_braces + ']' * remaining_brackets

            print(f"    Closing sequence: '{closing_seq}'")

            # Add closing tokens
            if closing_seq:
                closing_tokens = tokenizer.encode(closing_seq)
                print(f"    Closing tokens: {closing_tokens}")
                print(f"    Closing decoded: '{tokenizer.decode(closing_tokens)}'")
                generated_tokens.extend(closing_tokens)

            break

        # Sample
        next_token = base_sampler(next_logits)
        token_id = next_token.item()

        if hasattr(tokenizer, 'eos_token_id') and token_id == tokenizer.eos_token_id:
            print(f"[{i}] EOS")
            break

        token_str = tokenizer.decode([token_id])
        generated_tokens.append(token_id)

        # Update machine
        old_state = machine.current_state
        old_elem_count = machine.context.element_count
        for char in token_str:
            event = json_tok.process_char(char)
            if event:
                machine.send_event(event)

        # Show element count changes
        if machine.context.element_count != old_elem_count:
            print(f"[{i}] elem_count: {old_elem_count} -> {machine.context.element_count}")

        tokens = mx.concatenate([tokens, next_token[:, None]], axis=1)

    # Final output
    output = tokenizer.decode(generated_tokens)
    full_text = prompt + output

    print(f"\n--- Final ---")
    print(f"Generated (last 200): ...{output[-200:]}")
    print(f"\nFull text (last 200): ...{full_text[-200:]}")

    balanced = check_balanced(full_text)
    print(f"\nBalanced: {balanced}")

    try:
        json.loads(full_text)
        print("Valid JSON: YES")
    except json.JSONDecodeError as e:
        print(f"Valid JSON: NO - {e}")


if __name__ == "__main__":
    debug_force_close()
