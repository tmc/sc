#!/usr/bin/env python3
"""Debug why balanced JSON fails validation."""

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


def debug_validation():
    """Debug validation issues."""
    print("=" * 70)
    print("DEBUG: Validation issues")
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

    # Event to token mapping (simplified)
    vocab = tokenizer.get_vocab()
    struct_chars = {
        '{': 'LBRACE', '}': 'RBRACE',
        '[': 'LBRACKET', ']': 'RBRACKET',
        ':': 'COLON', ',': 'COMMA', '"': 'QUOTE'
    }

    event_to_tokens = {e: set() for e in ['LBRACE', 'RBRACE', 'LBRACKET', 'RBRACKET', 'COLON', 'COMMA', 'QUOTE', 'STRING', 'NUMBER', 'BOOL', 'NULL']}

    for token_str, token_id in vocab.items():
        first_event = None
        for char in token_str:
            if char in struct_chars:
                first_event = struct_chars[char]
                break

        if first_event:
            event_to_tokens[first_event].add(token_id)
        else:
            stripped = token_str.strip()
            if stripped in ('true', 'false'):
                event_to_tokens['BOOL'].add(token_id)
            elif stripped == 'null':
                event_to_tokens['NULL'].add(token_id)
            else:
                try:
                    float(stripped)
                    event_to_tokens['NUMBER'].add(token_id)
                except ValueError:
                    if stripped.isalnum() or (stripped and stripped[0].isalnum()):
                        event_to_tokens['STRING'].add(token_id)

    machine = StatechartMachine(statechart=sc_def)
    json_tok = JSONTokenizer()

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

    # Generate with guidance
    print("\n--- Guided Generation ---")
    tokens_input = mx.array(tokenizer.encode(prompt))[None]
    base_sampler = make_sampler(temp=0.3)

    generated_tokens = []
    for i in range(100):
        logits = model(tokens_input)
        next_logits = logits[:, -1, :]

        # Get enabled events
        enabled_events = machine.get_enabled_events()

        # Proactive comma blocking
        ctx = machine.context
        if 'COMMA' in enabled_events:
            if ctx.top() == 'array' and ctx.element_count >= ctx.max_elements - 1:
                enabled_events = enabled_events - {'COMMA'}
                print(f"  [{i}] Blocked COMMA (elem_count={ctx.element_count})")

        # Build allowed token IDs
        allowed_ids = set()
        for event in enabled_events:
            allowed_ids.update(event_to_tokens.get(event, set()))

        # Apply mask
        if allowed_ids:
            vocab_size = next_logits.shape[-1]
            mask_np = np.full(vocab_size, -100.0)
            for tid in allowed_ids:
                if tid < vocab_size:
                    mask_np[tid] = 0.0
            mask_array = mx.array(mask_np)
            next_logits = next_logits + mask_array

        # Check force close
        if machine.should_force_close():
            print(f"\n  [{i}] FORCE CLOSE!")
            current_output = tokenizer.decode(generated_tokens)
            full_text = prompt + current_output

            open_braces = full_text.count('{') - full_text.count('}')
            open_brackets = full_text.count('[') - full_text.count(']')
            unclosed_string = full_text.count('"') % 2 == 1

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

            print(f"  Closing: '{closing_seq}'")
            if closing_seq:
                closing_tokens = tokenizer.encode(closing_seq)
                generated_tokens.extend(closing_tokens)
            break

        # Sample
        next_token = base_sampler(next_logits)
        token_id = next_token.item()

        if hasattr(tokenizer, 'eos_token_id') and token_id == tokenizer.eos_token_id:
            break

        token_str = tokenizer.decode([token_id])
        generated_tokens.append(token_id)

        # Update machine
        for char in token_str:
            event = json_tok.process_char(char)
            if event:
                machine.send_event(event)

        if machine.is_complete():
            print(f"  [{i}] Complete!")
            break

        tokens_input = mx.concatenate([tokens_input, next_token[:, None]], axis=1)

    # Final output
    output = tokenizer.decode(generated_tokens)
    full_text = prompt + output

    print(f"\n--- Final ---")
    print(f"Full text:\n{full_text}")

    balanced = check_balanced(full_text)
    print(f"\nBalanced: {balanced}")

    try:
        parsed = json.loads(full_text)
        print("Valid JSON: YES")
        print(f"Parsed: {json.dumps(parsed, indent=2)[:500]}")
    except json.JSONDecodeError as e:
        print(f"Valid JSON: NO - {e}")
        # Find the error location
        print(f"\nError at position {e.pos}:")
        start = max(0, e.pos - 30)
        end = min(len(full_text), e.pos + 30)
        print(f"  Context: ...{full_text[start:end]}...")
        print(f"  Position: {' ' * (e.pos - start)}^")


if __name__ == "__main__":
    debug_validation()
