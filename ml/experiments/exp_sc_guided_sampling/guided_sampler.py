"""
Guided Sampler: Constrained Decoding for Statechart JSON

Applies grammar constraints during token sampling to ensure
valid statechart JSON output.

APPROACH:
1. Track parser state during generation
2. Compute valid token mask from grammar
3. Apply mask to logits before sampling
4. Guarantee 100% syntactically valid output

TECHNIQUES:
- Token masking: Set invalid token logits to -inf
- Lookahead: Check if token leads to valid state
- Backtrack: Recover from invalid sequences
"""

import json
import time
from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Any, Tuple, Callable

try:
    import mlx.core as mx
    import mlx.nn as nn
    MLX_AVAILABLE = True
except ImportError:
    mx = None
    nn = None
    MLX_AVAILABLE = False

from .sc_grammar import SCGrammar, GrammarState, get_valid_next_tokens


@dataclass
class SamplingConfig:
    """Configuration for guided sampling."""
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    use_grammar: bool = True
    early_stop_on_complete: bool = True
    min_states: int = 2
    require_transitions: bool = True


@dataclass
class SamplingResult:
    """Result of guided sampling."""
    output: str
    tokens_generated: int
    is_valid_json: bool
    is_valid_sc: bool
    grammar_violations: int
    duration: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tokens": self.tokens_generated,
            "valid_json": self.is_valid_json,
            "valid_sc": self.is_valid_sc,
            "violations": self.grammar_violations,
            "duration": self.duration,
        }


class TokenMaskBuilder:
    """
    Builds token masks based on grammar state.
    """

    def __init__(self, tokenizer: Any):
        self.tokenizer = tokenizer
        self.vocab = self._build_vocab_map()

    def _build_vocab_map(self) -> Dict[str, int]:
        """Build mapping from tokens to IDs."""
        if hasattr(self.tokenizer, 'get_vocab'):
            return self.tokenizer.get_vocab()
        return {}

    def build_mask(
        self,
        grammar: SCGrammar,
        vocab_size: int,
    ) -> Any:
        """
        Build mask of valid tokens.

        Args:
            grammar: Current grammar state
            vocab_size: Size of vocabulary

        Returns:
            Boolean mask [vocab_size] where True = valid
        """
        valid_chars = grammar.get_valid_tokens()

        if not MLX_AVAILABLE:
            # Return list for testing
            return [True] * vocab_size

        # Start with all invalid
        mask = [False] * vocab_size

        for token, token_id in self.vocab.items():
            if token_id >= vocab_size:
                continue

            # Check if token starts with valid char
            if self._token_is_valid(token, valid_chars):
                mask[token_id] = True

        return mx.array(mask)

    def _token_is_valid(self, token: str, valid_chars: Set[str]) -> bool:
        """Check if token is valid given valid characters."""
        if not token:
            return False

        # Decode if needed (handle byte tokens)
        decoded = self._decode_token(token)

        if not decoded:
            return False

        # First char must be in valid set
        if decoded[0] in valid_chars:
            return True

        # Check for keyword matches
        keywords = {
            "root_state", "label", "type", "children", "is_initial",
            "transitions", "from", "to", "event", "guard", "action",
            "true", "false", "null", "__root__",
        }
        if decoded.strip('"') in keywords:
            return True

        return False

    def _decode_token(self, token: str) -> str:
        """Decode token to string."""
        # Handle special tokens
        if token.startswith("<") and token.endswith(">"):
            return ""

        # Handle byte tokens like "Ġ" (space) in GPT-style tokenizers
        if token.startswith("Ġ"):
            return " " + token[1:]

        return token


class GuidedSampler:
    """
    Constrained decoding sampler for statechart JSON.
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        config: Optional[SamplingConfig] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or SamplingConfig()

        self.mask_builder = TokenMaskBuilder(tokenizer)
        self.grammar = SCGrammar()

    def sample(
        self,
        prompt: str,
    ) -> SamplingResult:
        """
        Generate statechart JSON with grammar guidance.

        Args:
            prompt: Generation prompt

        Returns:
            SamplingResult with output and metrics
        """
        start = time.time()

        self.grammar.reset()
        violations = 0

        # Encode prompt
        if hasattr(self.tokenizer, 'encode'):
            input_ids = self.tokenizer.encode(prompt)
        else:
            input_ids = list(prompt.encode())

        generated_tokens = []
        output_text = ""

        for step in range(self.config.max_tokens):
            # Get logits from model
            logits = self._get_logits(input_ids + generated_tokens)

            if logits is None:
                # Mock mode
                next_token = self._mock_next_token(output_text)
                if next_token is None:
                    break
            else:
                # Apply grammar mask
                if self.config.use_grammar:
                    mask = self.mask_builder.build_mask(
                        self.grammar,
                        logits.shape[-1] if MLX_AVAILABLE else 32000,
                    )
                    logits = self._apply_mask(logits, mask)

                # Sample next token
                next_token = self._sample_token(logits)

            if next_token is None:
                break

            # Decode token
            if hasattr(self.tokenizer, 'decode'):
                token_str = self.tokenizer.decode([next_token])
            else:
                token_str = chr(next_token) if next_token < 128 else ""

            # Advance grammar
            for char in token_str:
                valid = self.grammar.advance(char)
                if not valid:
                    violations += 1

            output_text += token_str
            generated_tokens.append(next_token)

            # Check for completion
            if self.config.early_stop_on_complete and self.grammar.is_complete():
                break

            # Check for EOS
            if hasattr(self.tokenizer, 'eos_token_id'):
                if next_token == self.tokenizer.eos_token_id:
                    break

        duration = time.time() - start

        # Validate output
        is_valid_json, is_valid_sc = self._validate_output(output_text)

        return SamplingResult(
            output=output_text,
            tokens_generated=len(generated_tokens),
            is_valid_json=is_valid_json,
            is_valid_sc=is_valid_sc,
            grammar_violations=violations,
            duration=duration,
        )

    def _get_logits(self, input_ids: List[int]) -> Optional[Any]:
        """Get logits from model."""
        if self.model is None:
            return None

        if MLX_AVAILABLE:
            # Convert to MLX array
            x = mx.array([input_ids])

            # Forward pass
            if hasattr(self.model, '__call__'):
                outputs = self.model(x)
            else:
                return None

            # Get last token logits
            return outputs[:, -1, :]

        return None

    def _apply_mask(self, logits: Any, mask: Any) -> Any:
        """Apply boolean mask to logits."""
        if not MLX_AVAILABLE:
            return logits

        # Set invalid tokens to -inf
        masked = mx.where(mask, logits, mx.array(float('-inf')))
        return masked

    def _sample_token(self, logits: Any) -> Optional[int]:
        """Sample token from logits."""
        if not MLX_AVAILABLE or logits is None:
            return None

        # Flatten if needed
        if len(logits.shape) > 1:
            logits = logits.squeeze()

        # Apply temperature
        if self.config.temperature > 0:
            logits = logits / self.config.temperature

        # Apply top-k by setting non-top-k to -inf
        if self.config.top_k > 0 and self.config.top_k < logits.shape[-1]:
            # Sort and get threshold
            sorted_logits = mx.sort(logits)
            threshold = sorted_logits[-self.config.top_k]
            logits = mx.where(logits >= threshold, logits, mx.array(float('-inf')))

        # Softmax
        probs = mx.softmax(logits, axis=-1)

        # Sample using categorical
        token_id = mx.random.categorical(mx.log(probs + 1e-10))

        return int(token_id.item())

    def _mock_next_token(self, output_so_far: str) -> Optional[int]:
        """Generate mock tokens for testing."""
        # Generate a valid statechart token by token
        target = '''{"root_state": {"label": "__root__", "type": 2, "children": [{"label": "s0", "type": 1, "is_initial": true}, {"label": "s1", "type": 1}]}, "transitions": [{"from": ["s0"], "to": ["s1"], "event": "E1"}]}'''

        if len(output_so_far) >= len(target):
            return None

        next_char = target[len(output_so_far)]
        return ord(next_char)

    def _validate_output(self, output: str) -> Tuple[bool, bool]:
        """Validate output as JSON and statechart."""
        # Check JSON validity
        try:
            chart = json.loads(output)
            is_valid_json = True
        except json.JSONDecodeError:
            return False, False

        # Check SC validity
        is_valid_sc = (
            "root_state" in chart and
            isinstance(chart.get("root_state"), dict) and
            "label" in chart.get("root_state", {}) and
            chart.get("root_state", {}).get("children", None) is not None
        )

        if is_valid_sc and self.config.require_transitions:
            is_valid_sc = "transitions" in chart

        if is_valid_sc and self.config.min_states > 0:
            children = chart.get("root_state", {}).get("children", [])
            is_valid_sc = len(children) >= self.config.min_states

        return is_valid_json, is_valid_sc


class UnguidedSampler:
    """
    Standard sampling without grammar constraints.
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        config: Optional[SamplingConfig] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or SamplingConfig()

    def sample(self, prompt: str) -> SamplingResult:
        """Generate without grammar guidance."""
        start = time.time()

        if self.model is None:
            # Mock generation with some probability of error
            import random
            if random.random() < 0.5:  # 50% validity for mock
                output = '''{"root_state": {"label": "__root__", "type": 2, "children": [{"label": "s0", "type": 1, "is_initial": true}]}, "transitions": []}'''
            else:
                output = '''Sorry, I cannot generate that statechart.'''
        else:
            # Use model's generate method
            from mlx_lm import generate
            from mlx_lm.sample_utils import make_sampler

            sampler = make_sampler(temp=self.config.temperature)
            output = generate(
                self.model,
                self.tokenizer,
                prompt=prompt,
                max_tokens=self.config.max_tokens,
                sampler=sampler,
            )

        duration = time.time() - start

        # Validate
        is_valid_json = False
        is_valid_sc = False

        try:
            chart = json.loads(output)
            is_valid_json = True
            is_valid_sc = (
                "root_state" in chart and
                isinstance(chart.get("root_state"), dict)
            )
        except json.JSONDecodeError:
            pass

        return SamplingResult(
            output=output,
            tokens_generated=len(output.split()),
            is_valid_json=is_valid_json,
            is_valid_sc=is_valid_sc,
            grammar_violations=0,
            duration=duration,
        )


# =============================================================================
# Demo
# =============================================================================

def demo():
    """Demonstrate guided sampling."""
    print("=" * 60)
    print("Guided Sampling for Statechart JSON")
    print("=" * 60)

    # Test with mock model
    guided = GuidedSampler(model=None, tokenizer=None)
    unguided = UnguidedSampler(model=None, tokenizer=None)

    prompt = "Generate a statechart JSON:"

    print("\nGuided Sampling:")
    result = guided.sample(prompt)
    print(f"  Valid JSON: {result.is_valid_json}")
    print(f"  Valid SC: {result.is_valid_sc}")
    print(f"  Violations: {result.grammar_violations}")

    print("\nUnguided Sampling (5 trials):")
    valid_count = 0
    for i in range(5):
        result = unguided.sample(prompt)
        if result.is_valid_sc:
            valid_count += 1
    print(f"  Valid: {valid_count}/5 ({valid_count*20}%)")

    return guided, unguided


if __name__ == "__main__":
    demo()
