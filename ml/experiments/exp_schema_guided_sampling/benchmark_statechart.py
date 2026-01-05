#!/usr/bin/env python3
"""
Benchmark: Compare statechart-guided vs unguided generation.

Uses the actual statechart executor (not Python state machine) to guide
token selection during JSON generation.
"""

import json
import sys
from pathlib import Path
from typing import Dict, Any, Tuple, List, Set

sys.path.insert(0, '/Volumes/tmc/go/src/github.com/tmc/sc/ml')

from experiments.exp_schema_guided_sampling.statechart_sampler import (
    StatechartMachine,
    JSONTokenizer,
    TOKEN_TO_EVENT,
)

try:
    import mlx.core as mx
    import numpy as np
    MLX_AVAILABLE = True
except ImportError:
    mx = None
    np = None
    MLX_AVAILABLE = False


class StatechartGuidedGenerator:
    """Generator using statechart executor for guided sampling."""

    def __init__(self, model: Any, tokenizer: Any, statechart_path: str = None):
        self.model = model
        self.tokenizer = tokenizer

        # Load statechart
        if statechart_path is None:
            statechart_path = Path(__file__).parent / "json_parser_v2.statechart.json"
        with open(statechart_path) as f:
            self.sc_def = json.load(f)

        # JSON tokenizer for proper parsing
        self.json_tokenizer = JSONTokenizer()

        # Build token -> event mapping
        self._build_token_event_map()

    def _build_token_event_map(self):
        """Map vocabulary tokens to statechart events."""
        vocab = self.tokenizer.get_vocab() if hasattr(self.tokenizer, 'get_vocab') else {}

        # Map each token to the set of ALL events it triggers
        self.token_to_all_events: Dict[int, Set[str]] = {}

        self.event_to_token_ids: Dict[str, Set[int]] = {
            'LBRACE': set(),
            'RBRACE': set(),
            'LBRACKET': set(),
            'RBRACKET': set(),
            'COLON': set(),
            'COMMA': set(),
            'QUOTE': set(),
            'STRING': set(),
            'NUMBER': set(),
            'BOOL': set(),
            'NULL': set(),
        }

        # Structural characters and their events
        struct_chars = {
            '{': 'LBRACE', '}': 'RBRACE',
            '[': 'LBRACKET', ']': 'RBRACKET',
            ':': 'COLON', ',': 'COMMA', '"': 'QUOTE'
        }

        for token_str, token_id in vocab.items():
            # Find ALL structural characters in token
            token_events = set()
            first_event = None
            for char in token_str:
                if char in struct_chars:
                    event = struct_chars[char]
                    token_events.add(event)
                    if first_event is None:
                        first_event = event

            if first_event:
                # Map to first event for basic lookup
                self.event_to_token_ids[first_event].add(token_id)
                # Store ALL events for constraint checking
                self.token_to_all_events[token_id] = token_events
            else:
                # Check for keywords
                stripped = token_str.strip()
                if stripped in ('true', 'false'):
                    self.event_to_token_ids['BOOL'].add(token_id)
                    self.token_to_all_events[token_id] = {'BOOL'}
                elif stripped == 'null':
                    self.event_to_token_ids['NULL'].add(token_id)
                    self.token_to_all_events[token_id] = {'NULL'}
                else:
                    # Check for numbers
                    try:
                        float(stripped)
                        self.event_to_token_ids['NUMBER'].add(token_id)
                        self.token_to_all_events[token_id] = {'NUMBER'}
                    except ValueError:
                        # String content (alphanumeric)
                        if stripped.isalnum() or (stripped and stripped[0].isalnum()):
                            self.event_to_token_ids['STRING'].add(token_id)
                            self.token_to_all_events[token_id] = {'STRING'}

    def generate(
        self,
        prompt: str,
        max_tokens: int = 200,
        temperature: float = 0.3,
        use_guidance: bool = True,
    ) -> Tuple[str, Dict[str, Any]]:
        """Generate with optional statechart guidance."""
        if not MLX_AVAILABLE:
            return "[MOCK]", {"mock": True}

        from mlx_lm.sample_utils import make_sampler

        # Create fresh machine and tokenizer for this generation
        machine = StatechartMachine(statechart=self.sc_def)
        self.json_tokenizer.reset()

        # Process prompt through machine to establish initial state
        self._process_prompt(prompt, machine)

        # Encode prompt
        tokens = mx.array(self.tokenizer.encode(prompt))[None]
        base_sampler = make_sampler(temp=temperature)

        generated_tokens = []
        metadata = {
            "use_guidance": use_guidance,
            "forced_closes": 0,
            "mask_applications": 0,
            "initial_state": machine.current_state,
        }

        for _ in range(max_tokens):
            # Forward pass
            logits = self.model(tokens)
            next_logits = logits[:, -1, :]

            if use_guidance:
                # Get enabled events from statechart
                enabled_events = machine.get_enabled_events()

                # Proactively exclude COMMA if it would lead to ForceClose next iteration
                # This prevents trailing commas that require post-hoc stripping
                ctx = machine.context
                if 'COMMA' in enabled_events:
                    if ctx.top() == 'array' and ctx.element_count >= ctx.max_elements - 1:
                        enabled_events = enabled_events - {'COMMA'}

                # Build allowed token IDs
                # A token is only allowed if ALL its events are enabled
                allowed_ids = set()
                for event in enabled_events:
                    for token_id in self.event_to_token_ids.get(event, set()):
                        # Check if ALL events in this token are enabled
                        token_events = self.token_to_all_events.get(token_id, set())
                        if token_events.issubset(enabled_events):
                            allowed_ids.add(token_id)

                # Apply mask if we have constraints
                if allowed_ids:
                    vocab_size = next_logits.shape[-1]
                    mask_np = np.full(vocab_size, -100.0)
                    for tid in allowed_ids:
                        if tid < vocab_size:
                            mask_np[tid] = 0.0
                    mask_array = mx.array(mask_np)
                    next_logits = next_logits + mask_array
                    metadata["mask_applications"] += 1

                # Check force close
                if machine.should_force_close():
                    # Count actual braces in full text to determine closing sequence
                    current_output = self.tokenizer.decode(generated_tokens)
                    full_text = prompt + current_output

                    # Strip trailing comma - needed because tokens like "},` contain
                    # both RBRACE and COMMA, and we can't block them without lookahead
                    stripped = current_output.rstrip()
                    if stripped.endswith(','):
                        stripped = stripped[:-1]
                        generated_tokens = list(self.tokenizer.encode(stripped))
                        current_output = stripped
                        full_text = prompt + current_output

                    # Count unclosed structures
                    open_braces = full_text.count('{') - full_text.count('}')
                    open_brackets = full_text.count('[') - full_text.count(']')
                    unclosed_string = full_text.count('"') % 2 == 1

                    # Build proper closing sequence based on actual counts
                    closing_seq = ''
                    if unclosed_string:
                        closing_seq += '"'

                    # Simple approach: just close all unclosed structures
                    # Use the stack to determine correct ordering
                    remaining_braces = open_braces
                    remaining_brackets = open_brackets

                    for item in reversed(machine.context.stack):
                        if item == "object" and remaining_braces > 0:
                            closing_seq += '}'
                            remaining_braces -= 1
                        elif item == "array" and remaining_brackets > 0:
                            closing_seq += ']'
                            remaining_brackets -= 1

                    # Close any remaining (in case stack is out of sync)
                    closing_seq += '}' * remaining_braces + ']' * remaining_brackets

                    if closing_seq:
                        closing_tokens = self.tokenizer.encode(closing_seq)
                        generated_tokens.extend(closing_tokens)
                    metadata["forced_closes"] += 1
                    metadata["closing_seq"] = closing_seq
                    metadata["open_braces_at_close"] = open_braces
                    metadata["open_brackets_at_close"] = open_brackets
                    break

            # Sample
            next_token = base_sampler(next_logits)
            token_id = next_token.item()

            # Check EOS
            if hasattr(self.tokenizer, 'eos_token_id') and token_id == self.tokenizer.eos_token_id:
                break

            generated_tokens.append(token_id)

            # Update machine state
            if use_guidance:
                token_str = self.tokenizer.decode([token_id])
                self._process_token_str(token_str, machine)

                if machine.is_complete():
                    break

            tokens = mx.concatenate([tokens, next_token[:, None]], axis=1)

        output = self.tokenizer.decode(generated_tokens)
        metadata["tokens_generated"] = len(generated_tokens)
        metadata["final_state"] = machine.current_state
        metadata["final_depth"] = machine.context.depth
        metadata["stack"] = machine.context.stack[:]

        return output, metadata

    def _process_prompt(self, prompt: str, machine: StatechartMachine):
        """Process prompt characters through machine using proper JSON tokenizer."""
        for char in prompt:
            event = self.json_tokenizer.process_char(char)
            if event:
                machine.send_event(event)

    def _process_token_str(self, token_str: str, machine: StatechartMachine):
        """Process generated token string through machine."""
        for char in token_str:
            event = self.json_tokenizer.process_char(char)
            if event:
                machine.send_event(event)


def validate_json(text: str) -> Tuple[bool, str]:
    """Validate JSON and return (success, error_message)."""
    try:
        json.loads(text)
        return True, ""
    except json.JSONDecodeError as e:
        return False, str(e)


def check_balanced(text: str) -> bool:
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


def run_benchmark():
    """Run the benchmark."""
    print("=" * 70)
    print("STATECHART-GUIDED SAMPLING BENCHMARK")
    print("=" * 70)

    if not MLX_AVAILABLE:
        print("MLX not available, running in mock mode")
        return

    from mlx_lm import load

    # Test prompts
    prompts = [
        '{"root_state": {"label": "__root__", "type": 2, "children": [{"label": "Idle"}, {"label":',
        '{"transitions": [{"from": ["Start"], "to": ["Running"], "event":',
        '{"root_state": {"label": "Main", "children": [{"label": "A", "type": 1}, {"label": "B",',
    ]

    # Test both base and instruct models
    models_to_test = [
        ("mlx-community/Qwen2.5-Coder-0.5B-Instruct-4bit", "Instruct"),
        ("mlx-community/Qwen2.5-Coder-0.5B-4bit", "Base"),
    ]

    results = []

    for model_path, model_type in models_to_test:
        print(f"\n{'=' * 70}")
        print(f"Testing: {model_type} ({model_path})")
        print("=" * 70)

        try:
            model, tokenizer = load(model_path)
            generator = StatechartGuidedGenerator(model, tokenizer)

            for mode in ["Unguided", "Guided"]:
                use_guidance = mode == "Guided"
                print(f"\n--- {mode} Mode ---")

                valid_json = 0
                balanced = 0
                total = len(prompts)

                for i, prompt in enumerate(prompts):
                    output, meta = generator.generate(
                        prompt,
                        max_tokens=150,
                        temperature=0.3,
                        use_guidance=use_guidance,
                    )

                    full_output = prompt + output
                    is_valid, err = validate_json(full_output)
                    is_balanced = check_balanced(full_output)

                    if is_valid:
                        valid_json += 1
                    if is_balanced:
                        balanced += 1

                    print(f"\n  Prompt {i+1}:")
                    print(f"    Output (first 100): {output[:100]}...")
                    print(f"    Valid JSON: {is_valid}")
                    print(f"    Balanced: {is_balanced}")
                    meta_str = f"state={meta.get('final_state')}, depth={meta.get('final_depth')}"
                    if meta.get('forced_closes'):
                        meta_str += f", closing='{meta.get('closing_seq')}', braces={meta.get('open_braces_at_close')}, brackets={meta.get('open_brackets_at_close')}"
                    print(f"    Meta: {meta_str}")

                result = {
                    "model": model_type,
                    "mode": mode,
                    "valid_json": valid_json,
                    "balanced": balanced,
                    "total": total,
                    "valid_pct": valid_json / total * 100,
                    "balanced_pct": balanced / total * 100,
                }
                results.append(result)

                print(f"\n  Summary: Valid={valid_json}/{total} ({result['valid_pct']:.0f}%), Balanced={balanced}/{total} ({result['balanced_pct']:.0f}%)")

            del model
            mx.metal.clear_cache()

        except Exception as e:
            print(f"Error with {model_type}: {e}")
            import traceback
            traceback.print_exc()

    # Final summary
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    print(f"{'Model':<10} {'Mode':<10} {'Valid JSON':<15} {'Balanced':<15}")
    print("-" * 50)
    for r in results:
        print(f"{r['model']:<10} {r['mode']:<10} {r['valid_pct']:>6.0f}%{'':<8} {r['balanced_pct']:>6.0f}%")

    # Key insight
    print("\n" + "=" * 70)
    print("KEY INSIGHT: Statechart-guided sampling constrains token selection")
    print("to only those that are valid given the current parser state.")
    print("=" * 70)


if __name__ == "__main__":
    run_benchmark()
