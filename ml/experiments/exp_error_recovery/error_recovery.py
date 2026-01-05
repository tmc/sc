"""
Error Recovery Testing for Constrained Samplers

Tests FSM recovery under adversarial conditions:
1. Inject invalid tokens mid-generation
2. Measure recovery to valid state
3. Compare recovery strategies: backtrack, skip, force-close

Targets: Retok-TopK and TokenSimulation approaches
"""

import json
import copy
import time
from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional, Tuple, Any
from enum import Enum
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    import mlx.core as mx
    import numpy as np
    from mlx_lm import load
    from mlx_lm.sample_utils import make_sampler
    MLX_AVAILABLE = True
except ImportError:
    mx = None
    np = None
    MLX_AVAILABLE = False


# =============================================================================
# RECOVERY STRATEGIES
# =============================================================================

class RecoveryStrategy(str, Enum):
    NONE = "none"           # No recovery - fail on invalid
    BACKTRACK = "backtrack"  # Remove invalid token, retry
    SKIP = "skip"           # Skip invalid token, continue
    FORCE_CLOSE = "force_close"  # Force close brackets/braces


@dataclass
class RecoveryResult:
    """Result of a single recovery attempt."""
    strategy: RecoveryStrategy
    error_position: int
    error_token: str
    recovered: bool
    tokens_removed: int
    final_valid: bool
    recovery_time_ms: float


@dataclass
class ErrorInjectionResult:
    """Result of error injection test."""
    total_injections: int
    recoveries_by_strategy: Dict[RecoveryStrategy, List[RecoveryResult]]
    recovery_rates: Dict[RecoveryStrategy, float]
    avg_tokens_removed: Dict[RecoveryStrategy, float]
    final_validity_rates: Dict[RecoveryStrategy, float]


# =============================================================================
# FSM STATE TRACKER
# =============================================================================

@dataclass
class FSMState:
    """Captured FSM state for backtracking."""
    current_state: str
    stack: List[str]
    depth: int
    element_count: int
    in_string: bool
    generated_tokens: List[int]
    generated_string: str


class StatechartMachine:
    """Simplified statechart machine for JSON parsing."""

    def __init__(self, max_elements: int = 20, max_depth: int = 10):
        self.max_elements = max_elements
        self.max_depth = max_depth
        self.reset()

    def reset(self):
        self.current_state = "START"
        self.stack = []
        self.depth = 0
        self.element_count = 0
        self.in_string = False

    def copy_state(self) -> FSMState:
        """Capture current state for backtracking."""
        return FSMState(
            current_state=self.current_state,
            stack=self.stack.copy(),
            depth=self.depth,
            element_count=self.element_count,
            in_string=self.in_string,
            generated_tokens=[],
            generated_string="",
        )

    def restore_state(self, state: FSMState):
        """Restore from captured state."""
        self.current_state = state.current_state
        self.stack = state.stack.copy()
        self.depth = state.depth
        self.element_count = state.element_count
        self.in_string = state.in_string

    def get_enabled_events(self) -> Set[str]:
        """Get currently valid events."""
        if self.in_string:
            return {"STRING_CHAR", "QUOTE"}  # Can continue or close string

        if self.current_state == "START":
            return {"LBRACE", "LBRACKET"}

        if not self.stack:
            return set()  # Complete

        top = self.stack[-1]

        if top == "object_key":
            return {"QUOTE"}  # Must start key
        elif top == "object_colon":
            return {"COLON"}
        elif top == "object_value":
            return {"QUOTE", "LBRACE", "LBRACKET", "NUMBER", "BOOL", "NULL"}
        elif top == "object_next":
            return {"COMMA", "RBRACE"}
        elif top == "array_value":
            return {"QUOTE", "LBRACE", "LBRACKET", "NUMBER", "BOOL", "NULL", "RBRACKET"}
        elif top == "array_next":
            return {"COMMA", "RBRACKET"}

        return set()

    def send_event(self, event: str) -> bool:
        """Process event, return True if valid transition."""
        enabled = self.get_enabled_events()

        if event not in enabled:
            return False

        # Handle string state
        if event == "QUOTE":
            self.in_string = not self.in_string
            if not self.in_string:
                # String closed - update stack
                if self.stack and self.stack[-1] == "object_key":
                    self.stack[-1] = "object_colon"
                elif self.stack and self.stack[-1] == "object_value":
                    self.stack[-1] = "object_next"
                elif self.stack and self.stack[-1] == "array_value":
                    self.stack[-1] = "array_next"
            return True

        if self.in_string and event == "STRING_CHAR":
            return True

        # Handle structural events
        if event == "LBRACE":
            self.depth += 1
            self.stack.append("object_key")
            self.current_state = "OBJECT"
            return True

        if event == "RBRACE":
            if self.stack and self.stack[-1] in ("object_key", "object_next"):
                self.stack.pop()
                self.depth -= 1
                self.element_count += 1
                if self.stack:
                    if self.stack[-1] == "object_value":
                        self.stack[-1] = "object_next"
                    elif self.stack[-1] == "array_value":
                        self.stack[-1] = "array_next"
                return True
            return False

        if event == "LBRACKET":
            self.depth += 1
            self.stack.append("array_value")
            self.current_state = "ARRAY"
            return True

        if event == "RBRACKET":
            if self.stack and self.stack[-1] in ("array_value", "array_next"):
                self.stack.pop()
                self.depth -= 1
                self.element_count += 1
                if self.stack:
                    if self.stack[-1] == "object_value":
                        self.stack[-1] = "object_next"
                    elif self.stack[-1] == "array_value":
                        self.stack[-1] = "array_next"
                return True
            return False

        if event == "COLON":
            if self.stack and self.stack[-1] == "object_colon":
                self.stack[-1] = "object_value"
                return True
            return False

        if event == "COMMA":
            if self.stack and self.stack[-1] == "object_next":
                self.stack[-1] = "object_key"
                return True
            if self.stack and self.stack[-1] == "array_next":
                self.stack[-1] = "array_value"
                return True
            return False

        if event in ("NUMBER", "BOOL", "NULL"):
            if self.stack and self.stack[-1] == "object_value":
                self.stack[-1] = "object_next"
                return True
            if self.stack and self.stack[-1] == "array_value":
                self.stack[-1] = "array_next"
                return True
            return False

        return False

    def is_complete(self) -> bool:
        return len(self.stack) == 0 and self.element_count > 0

    def should_force_close(self) -> bool:
        return self.element_count >= self.max_elements or self.depth >= self.max_depth


# =============================================================================
# ERROR INJECTOR
# =============================================================================

class ErrorInjector:
    """Injects invalid tokens to test recovery."""

    # Invalid tokens for different contexts
    INVALID_TOKENS = {
        "object_key": ["]", "}", ":", ",", "123", "true"],
        "object_value": [":", ",", "}", "]"],
        "array_value": [":", "}"],
        "general": ["<<<", ">>>", "@@@", "###"],
    }

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self._build_invalid_token_map()

    def _build_invalid_token_map(self):
        """Map invalid strings to token IDs."""
        self.invalid_token_ids = {}
        vocab = self.tokenizer.get_vocab() if hasattr(self.tokenizer, 'get_vocab') else {}

        for context, invalids in self.INVALID_TOKENS.items():
            self.invalid_token_ids[context] = []
            for invalid_str in invalids:
                # Find tokens containing this invalid string
                for token_str, token_id in vocab.items():
                    if invalid_str in token_str:
                        self.invalid_token_ids[context].append(token_id)
                        break

    def get_invalid_token(self, context: str = "general") -> Optional[int]:
        """Get an invalid token ID for the given context."""
        ids = self.invalid_token_ids.get(context, [])
        if ids:
            return ids[0]
        # Fallback: pick a random high ID that's likely invalid
        return 50000


# =============================================================================
# RECOVERY HANDLER
# =============================================================================

class RecoveryHandler:
    """Handles different recovery strategies."""

    def __init__(self, machine: StatechartMachine, tokenizer):
        self.machine = machine
        self.tokenizer = tokenizer
        self.checkpoints: List[Tuple[FSMState, List[int]]] = []

    def checkpoint(self, generated_tokens: List[int]):
        """Save checkpoint for potential backtrack."""
        state = self.machine.copy_state()
        state.generated_tokens = generated_tokens.copy()
        self.checkpoints.append((state, generated_tokens.copy()))

        # Keep only last 5 checkpoints
        if len(self.checkpoints) > 5:
            self.checkpoints.pop(0)

    def recover_backtrack(self, generated_tokens: List[int]) -> Tuple[bool, List[int], int]:
        """
        Backtrack recovery: remove tokens until valid state.
        Returns: (success, new_tokens, tokens_removed)
        """
        if not self.checkpoints:
            return False, generated_tokens, 0

        # Try each checkpoint from most recent
        for state, checkpoint_tokens in reversed(self.checkpoints):
            self.machine.restore_state(state)
            tokens_removed = len(generated_tokens) - len(checkpoint_tokens)

            # Verify state is valid
            if self.machine.get_enabled_events():
                return True, checkpoint_tokens, tokens_removed

        return False, generated_tokens, 0

    def recover_skip(self, invalid_token: int, generated_tokens: List[int]) -> Tuple[bool, List[int]]:
        """
        Skip recovery: ignore the invalid token, continue.
        Returns: (success, tokens_without_invalid)
        """
        # Simply don't add the invalid token
        # Check if current state can still proceed
        enabled = self.machine.get_enabled_events()
        return bool(enabled), generated_tokens

    def recover_force_close(self, generated_tokens: List[int]) -> Tuple[bool, List[int]]:
        """
        Force-close recovery: add closing brackets/braces.
        Returns: (success, tokens_with_closers)
        """
        new_tokens = generated_tokens.copy()

        # Close all open brackets/braces
        closing = ""
        for item in reversed(self.machine.stack):
            if "object" in item:
                closing += "}"
            elif "array" in item:
                closing += "]"

        if closing:
            closing_ids = self.tokenizer.encode(closing)
            new_tokens.extend(closing_ids)

            # Update machine state
            for char in closing:
                event = "RBRACE" if char == "}" else "RBRACKET"
                self.machine.send_event(event)

            return True, new_tokens

        return False, generated_tokens


# =============================================================================
# ERROR RECOVERY TESTER
# =============================================================================

class ErrorRecoveryTester:
    """
    Tests error recovery for constrained samplers.
    """

    def __init__(self, model_name: str = "mlx-community/Qwen2.5-Coder-0.5B-Instruct-4bit"):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None

    def load_model(self):
        if self.model is not None:
            return

        print(f"Loading {self.model_name}...")
        self.model, self.tokenizer = load(self.model_name)
        print("Model loaded.")

    def run_error_injection_test(
        self,
        num_injections: int = 10,
        injection_positions: List[int] = None,
        verbose: bool = True,
    ) -> ErrorInjectionResult:
        """
        Run error injection tests with all recovery strategies.
        """
        self.load_model()

        if injection_positions is None:
            injection_positions = [5, 10, 15, 20, 25]  # Default positions

        results_by_strategy = {s: [] for s in RecoveryStrategy}

        for i in range(num_injections):
            pos = injection_positions[i % len(injection_positions)]

            if verbose:
                print(f"\n[{i+1}/{num_injections}] Injection at position {pos}")

            for strategy in [RecoveryStrategy.BACKTRACK, RecoveryStrategy.SKIP, RecoveryStrategy.FORCE_CLOSE]:
                result = self._test_single_injection(pos, strategy, verbose)
                results_by_strategy[strategy].append(result)

        # Compute statistics
        recovery_rates = {}
        avg_tokens_removed = {}
        final_validity_rates = {}

        for strategy, results in results_by_strategy.items():
            if strategy == RecoveryStrategy.NONE:
                continue

            if results:
                recovery_rates[strategy] = sum(1 for r in results if r.recovered) / len(results)
                avg_tokens_removed[strategy] = sum(r.tokens_removed for r in results) / len(results)
                final_validity_rates[strategy] = sum(1 for r in results if r.final_valid) / len(results)
            else:
                recovery_rates[strategy] = 0.0
                avg_tokens_removed[strategy] = 0.0
                final_validity_rates[strategy] = 0.0

        return ErrorInjectionResult(
            total_injections=num_injections,
            recoveries_by_strategy=results_by_strategy,
            recovery_rates=recovery_rates,
            avg_tokens_removed=avg_tokens_removed,
            final_validity_rates=final_validity_rates,
        )

    def _test_single_injection(
        self,
        injection_position: int,
        strategy: RecoveryStrategy,
        verbose: bool,
    ) -> RecoveryResult:
        """Test a single error injection with given recovery strategy."""

        machine = StatechartMachine()
        injector = ErrorInjector(self.tokenizer)
        handler = RecoveryHandler(machine, self.tokenizer)

        # Generate valid JSON prefix
        prompt = '{"root_state": {"label": "'

        sampler = make_sampler(temp=0.3)
        tokens = mx.array(self.tokenizer.encode(prompt))[None]
        generated = []

        # Process prompt through machine
        for char in prompt:
            event = self._char_to_event(char)
            if event:
                machine.send_event(event)

        # Generate tokens with error injection
        error_injected = False
        error_token = ""
        recovery_start = 0
        recovered = False
        tokens_removed = 0

        for step in range(40):
            # Checkpoint before each token
            handler.checkpoint(generated)

            # Inject error at specified position
            if step == injection_position and not error_injected:
                invalid_id = injector.get_invalid_token("general")
                error_token = self.tokenizer.decode([invalid_id])
                error_injected = True
                recovery_start = time.time()

                if verbose:
                    print(f"    Injecting invalid token: {repr(error_token)}")

                # Attempt recovery based on strategy
                if strategy == RecoveryStrategy.BACKTRACK:
                    recovered, generated, tokens_removed = handler.recover_backtrack(generated)
                elif strategy == RecoveryStrategy.SKIP:
                    recovered, generated = handler.recover_skip(invalid_id, generated)
                elif strategy == RecoveryStrategy.FORCE_CLOSE:
                    recovered, generated = handler.recover_force_close(generated)

                if recovered and verbose:
                    print(f"    {strategy.value} recovered (removed {tokens_removed} tokens)")

                if recovered and strategy == RecoveryStrategy.FORCE_CLOSE:
                    break  # Force close ends generation

                continue

            # Normal generation step
            logits = self.model(tokens)
            next_logits = logits[:, -1, :]

            # Get valid tokens
            enabled = machine.get_enabled_events()
            if not enabled:
                break

            # Sample
            next_token = sampler(next_logits)
            token_id = next_token.item()
            token_str = self.tokenizer.decode([token_id])

            # Process token through machine
            valid = True
            for char in token_str:
                event = self._char_to_event(char)
                if event:
                    if not machine.send_event(event):
                        valid = False
                        break

            if valid:
                generated.append(token_id)
                tokens = mx.concatenate([tokens, next_token[:, None]], axis=1)

            if machine.is_complete():
                break

        # Check final validity
        output = prompt + self.tokenizer.decode(generated)
        try:
            json.loads(output)
            final_valid = True
        except:
            final_valid = False

        recovery_time = (time.time() - recovery_start) * 1000 if error_injected else 0

        return RecoveryResult(
            strategy=strategy,
            error_position=injection_position,
            error_token=error_token,
            recovered=recovered,
            tokens_removed=tokens_removed,
            final_valid=final_valid,
            recovery_time_ms=recovery_time,
        )

    def _char_to_event(self, char: str) -> Optional[str]:
        """Map character to FSM event."""
        mapping = {
            '{': 'LBRACE', '}': 'RBRACE',
            '[': 'LBRACKET', ']': 'RBRACKET',
            ':': 'COLON', ',': 'COMMA',
            '"': 'QUOTE',
        }
        return mapping.get(char)


# =============================================================================
# MAIN
# =============================================================================

def run_error_recovery_benchmark(num_tests: int = 10, verbose: bool = True):
    """Run full error recovery benchmark."""

    if not MLX_AVAILABLE:
        print("MLX not available")
        return None

    print("=" * 70)
    print("ERROR RECOVERY BENCHMARK")
    print("=" * 70)

    tester = ErrorRecoveryTester()
    result = tester.run_error_injection_test(
        num_injections=num_tests,
        injection_positions=[5, 10, 15, 20, 25],
        verbose=verbose,
    )

    # Print summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    print(f"\nTotal injections: {result.total_injections}")
    print(f"\n{'Strategy':<15} {'Recovery Rate':<15} {'Avg Removed':<15} {'Final Valid':<15}")
    print("-" * 60)

    for strategy in [RecoveryStrategy.BACKTRACK, RecoveryStrategy.SKIP, RecoveryStrategy.FORCE_CLOSE]:
        rate = result.recovery_rates.get(strategy, 0)
        removed = result.avg_tokens_removed.get(strategy, 0)
        valid = result.final_validity_rates.get(strategy, 0)
        print(f"{strategy.value:<15} {rate:>13.1%} {removed:>14.1f} {valid:>14.1%}")

    # Determine best strategy
    best_strategy = max(
        result.final_validity_rates.keys(),
        key=lambda s: result.final_validity_rates[s]
    )
    best_rate = result.final_validity_rates[best_strategy]

    print(f"\nBest strategy: {best_strategy.value} ({best_rate:.1%} final validity)")
    print("=" * 70)

    return result, best_strategy, best_rate


if __name__ == "__main__":
    import sys

    num_tests = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    run_error_recovery_benchmark(num_tests=num_tests, verbose=True)
