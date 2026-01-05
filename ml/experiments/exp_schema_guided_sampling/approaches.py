#!/usr/bin/env python3
"""
Exploration of different constrained decoding approaches.

Approaches:
1. Naive event mapping (current) - map tokens to first event, check if enabled
2. Pre-computed token masks - for each state, pre-compute valid tokens
3. Token simulation - simulate each candidate token through machine
4. Re-tokenization - work at string level, re-tokenize after each step
5. Jump-forward - when only one valid path, emit without sampling
"""

import json
import copy
from dataclasses import dataclass, field
from typing import Dict, List, Set, Any, Optional, Tuple
from pathlib import Path

from experiments.exp_schema_guided_sampling.statechart_sampler import (
    StatechartMachine,
    ParserContext,
    JSONTokenizer,
    CHAR_TO_EVENT,
)

try:
    import mlx.core as mx
    import numpy as np
    MLX_AVAILABLE = True
except ImportError:
    mx = None
    np = None
    MLX_AVAILABLE = False


# =============================================================================
# Approach 1: Naive Event Mapping (baseline - current implementation)
# =============================================================================

class NaiveApproach:
    """
    Current approach: map tokens to events by first structural character.

    Problem: Token "}," contains both RBRACE and COMMA. If we only check
    the first event (RBRACE), we may allow tokens that would violate the
    grammar when fully processed.
    """

    def __init__(self, tokenizer: Any, statechart: Dict):
        self.tokenizer = tokenizer
        self.statechart = statechart
        self.machine = StatechartMachine(statechart=statechart)
        self.json_tokenizer = JSONTokenizer()
        self._build_token_event_map()

    def _build_token_event_map(self):
        """Map tokens to their FIRST structural event."""
        vocab = self.tokenizer.get_vocab()

        self.event_to_tokens: Dict[str, Set[int]] = {
            'LBRACE': set(), 'RBRACE': set(), 'LBRACKET': set(),
            'RBRACKET': set(), 'COLON': set(), 'COMMA': set(),
            'QUOTE': set(), 'STRING': set(), 'NUMBER': set(),
            'BOOL': set(), 'NULL': set(),
        }

        struct_chars = {
            '{': 'LBRACE', '}': 'RBRACE', '[': 'LBRACKET',
            ']': 'RBRACKET', ':': 'COLON', ',': 'COMMA', '"': 'QUOTE'
        }

        for token_str, token_id in vocab.items():
            # Find FIRST structural character
            first_event = None
            for char in token_str:
                if char in struct_chars:
                    first_event = struct_chars[char]
                    break

            if first_event:
                self.event_to_tokens[first_event].add(token_id)
            else:
                stripped = token_str.strip()
                if stripped in ('true', 'false'):
                    self.event_to_tokens['BOOL'].add(token_id)
                elif stripped == 'null':
                    self.event_to_tokens['NULL'].add(token_id)
                else:
                    try:
                        float(stripped)
                        self.event_to_tokens['NUMBER'].add(token_id)
                    except ValueError:
                        if stripped.isalnum() or (stripped and stripped[0].isalnum()):
                            self.event_to_tokens['STRING'].add(token_id)

    def reset(self):
        self.machine.reset()
        self.json_tokenizer.reset()

    def get_valid_tokens(self) -> Set[int]:
        """Get valid token IDs based on enabled events."""
        enabled = self.machine.get_enabled_events()
        valid = set()
        for event in enabled:
            valid.update(self.event_to_tokens.get(event, set()))
        return valid

    def process_token(self, token_str: str):
        """Process token through machine."""
        for char in token_str:
            event = self.json_tokenizer.process_char(char)
            if event:
                self.machine.send_event(event)


# =============================================================================
# Approach 2: Pre-computed Token Masks (Outlines-style)
# =============================================================================

class PrecomputedMasksApproach:
    """
    Pre-compute valid tokens for each (state, context_hash) pair.

    For each state and relevant context configuration:
    1. For each token in vocabulary
    2. Simulate processing the token
    3. If all events in the token are valid transitions, mark token as valid

    Trade-off: High startup cost, O(1) runtime lookup
    """

    def __init__(self, tokenizer: Any, statechart: Dict, max_precompute_depth: int = 4):
        self.tokenizer = tokenizer
        self.statechart = statechart
        self.machine = StatechartMachine(statechart=statechart)
        self.json_tokenizer = JSONTokenizer()
        self.max_precompute_depth = max_precompute_depth

        # Pre-analyze vocabulary
        self._analyze_vocabulary()

        # Pre-compute masks (expensive!)
        self._precompute_masks()

    def _analyze_vocabulary(self):
        """Analyze each token to determine what events it generates."""
        vocab = self.tokenizer.get_vocab()

        # For each token, store the sequence of events it generates
        self.token_events: Dict[int, List[str]] = {}

        for token_str, token_id in vocab.items():
            events = []
            tokenizer = JSONTokenizer()
            for char in token_str:
                event = tokenizer.process_char(char)
                if event:
                    events.append(event)
            self.token_events[token_id] = events

    def _get_context_key(self, machine: StatechartMachine) -> Tuple:
        """Get hashable key for current context."""
        ctx = machine.context
        # Include relevant context in key
        stack_tuple = tuple(ctx.stack[:self.max_precompute_depth])
        return (machine.current_state, ctx.depth, stack_tuple)

    def _precompute_masks(self):
        """Pre-compute valid token masks for reachable states."""
        print("Pre-computing token masks...")

        self.masks: Dict[Tuple, Set[int]] = {}

        # BFS to find reachable states with context
        visited = set()
        queue = [(StatechartMachine(statechart=self.statechart),)]

        while queue:
            (machine,) = queue.pop(0)
            key = self._get_context_key(machine)

            if key in visited:
                continue
            visited.add(key)

            # Compute valid tokens for this state
            valid_tokens = set()

            for token_id, events in self.token_events.items():
                if not events:
                    # Token generates no events (whitespace, etc)
                    valid_tokens.add(token_id)
                    continue

                # Simulate processing this token
                test_machine = StatechartMachine(statechart=self.statechart)
                test_machine.current_state = machine.current_state
                test_machine.context = machine.context.copy()

                valid = True
                for event in events:
                    enabled = test_machine.get_enabled_events()
                    if event not in enabled:
                        valid = False
                        break
                    test_machine.send_event(event)

                if valid:
                    valid_tokens.add(token_id)

                    # Add resulting state to queue
                    if len(visited) < 10000:  # Limit exploration
                        queue.append((test_machine,))

            self.masks[key] = valid_tokens

        print(f"Pre-computed masks for {len(self.masks)} states")

    def reset(self):
        self.machine.reset()
        self.json_tokenizer.reset()

    def get_valid_tokens(self) -> Set[int]:
        """O(1) lookup of valid tokens."""
        key = self._get_context_key(self.machine)
        return self.masks.get(key, set())

    def process_token(self, token_str: str):
        """Process token through machine."""
        for char in token_str:
            event = self.json_tokenizer.process_char(char)
            if event:
                self.machine.send_event(event)


# =============================================================================
# Approach 3: Token Simulation (compute-per-token like llguidance)
# =============================================================================

class TokenSimulationApproach:
    """
    For each candidate token, simulate processing it through the machine.

    No pre-computation, compute validity on-the-fly.

    Trade-off: No startup cost, O(V) per token where V is vocabulary size
    Can be optimized by only checking top-k candidates.
    """

    def __init__(self, tokenizer: Any, statechart: Dict):
        self.tokenizer = tokenizer
        self.statechart = statechart
        self.machine = StatechartMachine(statechart=statechart)
        self.json_tokenizer = JSONTokenizer()
        self._analyze_vocabulary()

    def _analyze_vocabulary(self):
        """Pre-analyze token -> events mapping (cheap)."""
        vocab = self.tokenizer.get_vocab()
        self.token_events: Dict[int, List[str]] = {}
        self.vocab_size = len(vocab)

        for token_str, token_id in vocab.items():
            events = []
            tokenizer = JSONTokenizer()
            for char in token_str:
                event = tokenizer.process_char(char)
                if event:
                    events.append(event)
            self.token_events[token_id] = events

    def reset(self):
        self.machine.reset()
        self.json_tokenizer.reset()

    def is_token_valid(self, token_id: int) -> bool:
        """Check if a specific token is valid from current state."""
        events = self.token_events.get(token_id, [])

        if not events:
            return True  # No events = always valid (whitespace, etc)

        # Create test machine
        test_machine = StatechartMachine(statechart=self.statechart)
        test_machine.current_state = self.machine.current_state
        test_machine.context = self.machine.context.copy()

        for event in events:
            enabled = test_machine.get_enabled_events()
            if event not in enabled:
                return False
            test_machine.send_event(event)

        return True

    def get_valid_tokens(self, top_k_ids: Optional[List[int]] = None) -> Set[int]:
        """
        Get valid tokens.

        If top_k_ids provided, only check those (optimization).
        Otherwise check all tokens (slow but complete).
        """
        candidates = top_k_ids if top_k_ids else list(self.token_events.keys())

        valid = set()
        for token_id in candidates:
            if self.is_token_valid(token_id):
                valid.add(token_id)

        return valid

    def get_valid_tokens_with_topk_optimization(
        self,
        logits: "mx.array",
        k: int = 100
    ) -> Set[int]:
        """
        Optimization: only simulate top-k tokens by logit score.

        This is the llguidance approach - compute on-the-fly but only
        for tokens that have a chance of being selected.
        """
        if not MLX_AVAILABLE:
            return self.get_valid_tokens()

        # Get top-k token IDs by logit
        top_k_indices = mx.argpartition(logits, -k)[-k:]
        top_k_ids = [int(i) for i in top_k_indices.tolist()]

        return self.get_valid_tokens(top_k_ids)

    def process_token(self, token_str: str):
        """Process token through machine."""
        for char in token_str:
            event = self.json_tokenizer.process_char(char)
            if event:
                self.machine.send_event(event)


# =============================================================================
# Approach 4: Re-tokenization (SGLang-style)
# =============================================================================

class RetokenizationApproach:
    """
    Work at string level, re-tokenize after each generation step.

    Instead of tracking token-by-token:
    1. Maintain current generated string
    2. For each candidate, append to string
    3. Validate resulting string against grammar
    4. Re-tokenize to get new token sequence

    Trade-off: Handles tokenization boundaries perfectly but slower

    NOTE: This approach now uses full statechart simulation to track
    extended state (element_count, depth limits, ForceClose).
    """

    def __init__(self, tokenizer: Any, statechart: Dict):
        self.tokenizer = tokenizer
        self.statechart = statechart
        self.generated_string = ""
        self.machine = StatechartMachine(statechart=statechart)
        self.json_tokenizer = JSONTokenizer()
        self._analyze_vocabulary()

    def _analyze_vocabulary(self):
        """Build vocabulary lookup."""
        self.vocab = self.tokenizer.get_vocab()
        self.id_to_str = {v: k for k, v in self.vocab.items()}

    def reset(self):
        self.generated_string = ""
        self.machine.reset()
        self.json_tokenizer.reset()

    def _validate_token(self, token_str: str) -> bool:
        """
        Validate if token can be processed from current state.

        Simulates processing each character through the statechart.
        Returns True if all characters produce valid transitions.
        """
        # Create test machine from current state
        test_machine = StatechartMachine(statechart=self.statechart)
        test_machine.current_state = self.machine.current_state
        test_machine.context = self.machine.context.copy()
        test_tokenizer = JSONTokenizer()
        test_tokenizer.in_string = self.json_tokenizer.in_string
        test_tokenizer.current_token = self.json_tokenizer.current_token
        test_tokenizer.escape_next = self.json_tokenizer.escape_next

        for char in token_str:
            event = test_tokenizer.process_char(char)
            if event:
                enabled = test_machine.get_enabled_events()
                if event not in enabled:
                    return False
                test_machine.send_event(event)

        return True

    def is_token_valid(self, token_id: int) -> bool:
        """Check if appending this token would be valid."""
        token_str = self.id_to_str.get(token_id, "")
        return self._validate_token(token_str)

    def get_valid_tokens(self, top_k_ids: Optional[List[int]] = None) -> Set[int]:
        """Get valid tokens by testing each candidate."""
        candidates = top_k_ids if top_k_ids else list(self.vocab.values())

        valid = set()
        for token_id in candidates:
            if self.is_token_valid(token_id):
                valid.add(token_id)

        return valid

    def process_token(self, token_str: str):
        """Append token and update machine state."""
        self.generated_string += token_str
        for char in token_str:
            event = self.json_tokenizer.process_char(char)
            if event:
                self.machine.send_event(event)

    def get_retokenized(self) -> List[int]:
        """Get token IDs for current generated string."""
        return self.tokenizer.encode(self.generated_string)


# =============================================================================
# Approach 5: Jump-Forward Decoding
# =============================================================================

class JumpForwardApproach:
    """
    When there's only one valid next token, emit it without sampling.

    Identifies "singular transition paths" where the grammar forces
    a specific token sequence, and emits them all at once.
    """

    def __init__(self, tokenizer: Any, statechart: Dict):
        self.tokenizer = tokenizer
        self.statechart = statechart
        self.machine = StatechartMachine(statechart=statechart)
        self.json_tokenizer = JSONTokenizer()

        # Use token simulation for validity checking
        self.sim = TokenSimulationApproach(tokenizer, statechart)

    def reset(self):
        self.machine.reset()
        self.json_tokenizer.reset()
        self.sim.reset()

    def get_forced_tokens(self) -> List[int]:
        """
        Get sequence of tokens that are forced by grammar.

        Returns list of token IDs that must be emitted next.
        Empty list if multiple valid choices exist.
        """
        forced = []

        # Create test machine for lookahead
        test_machine = StatechartMachine(statechart=self.statechart)
        test_machine.current_state = self.machine.current_state
        test_machine.context = self.machine.context.copy()

        # Create test sim
        test_sim = TokenSimulationApproach(self.tokenizer, self.statechart)
        test_sim.machine = test_machine

        while True:
            valid = test_sim.get_valid_tokens()

            if len(valid) == 0:
                # No valid tokens - stop
                break
            elif len(valid) == 1:
                # Exactly one valid token - force it
                token_id = list(valid)[0]
                forced.append(token_id)

                # Update test machine
                token_str = self.sim.tokenizer.decode([token_id])
                for char in token_str:
                    event = JSONTokenizer().process_char(char)
                    if event:
                        test_machine.send_event(event)

                # Limit forced sequence length
                if len(forced) >= 10:
                    break
            else:
                # Multiple valid tokens - stop forcing
                break

        return forced

    def get_valid_tokens(self, top_k_ids: Optional[List[int]] = None) -> Set[int]:
        """Get valid tokens (delegates to simulation approach)."""
        self.sim.machine = self.machine
        return self.sim.get_valid_tokens(top_k_ids)

    def process_token(self, token_str: str):
        """Process token through machine."""
        for char in token_str:
            event = self.json_tokenizer.process_char(char)
            if event:
                self.machine.send_event(event)
        self.sim.machine = self.machine


# =============================================================================
# Benchmark
# =============================================================================

def benchmark_approaches():
    """Benchmark different approaches."""
    print("=" * 70)
    print("CONSTRAINED DECODING APPROACHES COMPARISON")
    print("=" * 70)

    if not MLX_AVAILABLE:
        print("MLX not available")
        return

    from mlx_lm import load
    from mlx_lm.sample_utils import make_sampler
    import time

    # Load model
    print("Loading model...")
    model, tokenizer = load("mlx-community/Qwen2.5-Coder-0.5B-Instruct-4bit")
    print("Model loaded.")

    # Load statechart
    sc_path = Path(__file__).parent / "json_parser_v2.statechart.json"
    with open(sc_path) as f:
        sc_def = json.load(f)

    # Test prompt
    prompt = '{"root_state": {"label": "__root__", "type": 2, "children": [{"label": "Idle"}, {"label":'

    # Top-k for simulation approaches (only check top candidates)
    TOP_K = 500

    approaches = [
        ("Naive", NaiveApproach, False),
        ("TokenSim-TopK", TokenSimulationApproach, True),  # Use top-k optimization
        ("Retok-TopK", RetokenizationApproach, True),  # Use top-k optimization
        ("JumpForward", JumpForwardApproach, True),
    ]

    results = []

    for name, ApproachClass, use_topk in approaches:
        print(f"\n{'=' * 70}")
        print(f"Approach: {name}")
        print("=" * 70)

        # Initialize
        t0 = time.time()
        approach = ApproachClass(tokenizer, sc_def)
        init_time = time.time() - t0
        print(f"Init time: {init_time:.3f}s")

        # Process prompt
        approach.reset()
        if hasattr(approach, 'machine'):
            json_tok = JSONTokenizer()
            for char in prompt:
                event = json_tok.process_char(char)
                if event:
                    approach.machine.send_event(event)
        if hasattr(approach, 'generated_string'):
            approach.generated_string = prompt

        # Generate
        tokens = mx.array(tokenizer.encode(prompt))[None]
        sampler = make_sampler(temp=0.3)
        generated = []

        t0 = time.time()
        mask_times = []

        max_tokens = 100
        for i in range(max_tokens):
            # Forward
            logits = model(tokens)
            next_logits = logits[:, -1, :]

            # Get mask
            mt0 = time.time()

            if name == "JumpForward":
                # Check for forced tokens first
                forced = approach.get_forced_tokens()
                if forced:
                    for tid in forced:
                        generated.append(tid)
                        token_str = tokenizer.decode([tid])
                        approach.process_token(token_str)
                    tokens = mx.array(tokenizer.encode(prompt + tokenizer.decode(generated)))[None]
                    mask_times.append(time.time() - mt0)
                    continue

            # Get top-k candidates for approaches that need it
            if use_topk:
                top_k_indices = mx.argpartition(next_logits.squeeze(), -TOP_K)[-TOP_K:]
                top_k_ids = [int(i) for i in top_k_indices.tolist()]
                valid = approach.get_valid_tokens(top_k_ids)
            else:
                valid = approach.get_valid_tokens()

            mask_times.append(time.time() - mt0)

            # Apply mask
            if valid:
                vocab_size = next_logits.shape[-1]
                mask_np = np.full(vocab_size, -100.0)
                for tid in valid:
                    if tid < vocab_size:
                        mask_np[tid] = 0.0
                next_logits = next_logits + mx.array(mask_np)

            # Sample
            next_token = sampler(next_logits)
            token_id = next_token.item()

            if hasattr(tokenizer, 'eos_token_id') and token_id == tokenizer.eos_token_id:
                break

            generated.append(token_id)
            token_str = tokenizer.decode([token_id])
            approach.process_token(token_str)

            # Check completion
            if hasattr(approach, 'machine'):
                if approach.machine.is_complete():
                    print(f"  [{i}] Complete!")
                    break
                if approach.machine.should_force_close():
                    print(f"  [{i}] Force close triggered")
                    # Strip trailing comma if present (before closing)
                    current = tokenizer.decode(generated)
                    if current.rstrip().endswith(','):
                        current = current.rstrip().rstrip(',')
                        generated = list(tokenizer.encode(current))
                        print(f"  Stripped trailing comma")
                    # Add closing tokens based on stack
                    closing = ""
                    for item in reversed(approach.machine.context.stack):
                        if item == "object":
                            closing += "}"
                        elif item == "array":
                            closing += "]"
                    if closing:
                        closing_tokens = tokenizer.encode(closing)
                        generated.extend(closing_tokens)
                        print(f"  Added closing: {closing}")
                    break

            tokens = mx.concatenate([tokens, next_token[:, None]], axis=1)

            # Progress
            if (i + 1) % 20 == 0:
                print(f"  [{i+1}/{max_tokens}] tokens generated")

        gen_time = time.time() - t0
        avg_mask_time = sum(mask_times) / len(mask_times) if mask_times else 0

        # Results
        output = tokenizer.decode(generated)
        full = prompt + output

        try:
            json.loads(full)
            valid_json = True
        except json.JSONDecodeError as e:
            valid_json = False
            print(f"JSON error: {e}")

        balanced = check_balanced(full)

        print(f"Generated ({len(generated)} tokens): {output[:100]}...")
        print(f"Valid JSON: {valid_json}")
        print(f"Balanced: {balanced}")
        print(f"Gen time: {gen_time:.3f}s")
        print(f"Avg mask time: {avg_mask_time*1000:.2f}ms")

        results.append({
            "approach": name,
            "init_time": init_time,
            "gen_time": gen_time,
            "avg_mask_time": avg_mask_time,
            "valid_json": valid_json,
            "balanced": balanced,
            "tokens": len(generated),
        })

        # Clean up
        del approach

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Approach':<20} {'Init':<10} {'Gen':<10} {'Mask':<12} {'Valid':<8} {'Balanced':<8}")
    print("-" * 70)
    for r in results:
        print(f"{r['approach']:<20} {r['init_time']:.3f}s{'':<4} {r['gen_time']:.3f}s{'':<4} {r['avg_mask_time']*1000:.2f}ms{'':<5} {str(r['valid_json']):<8} {str(r['balanced']):<8}")


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


if __name__ == "__main__":
    benchmark_approaches()
