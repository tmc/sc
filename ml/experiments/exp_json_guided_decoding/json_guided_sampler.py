"""
JSON Guided Sampler - Constrained LLM Token Generation

Uses the JSON grammar FSM to mask invalid tokens during generation,
guaranteeing syntactically valid JSON output.
"""

import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Any, Callable

from .json_grammar import JSONParser, JSONState, get_valid_next_tokens

# Try to import MLX/numpy for actual sampling
try:
    import mlx.core as mx
    HAS_MLX = True
except ImportError:
    HAS_MLX = False

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


@dataclass
class SamplerConfig:
    """Configuration for JSON-guided sampling."""
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    mask_penalty: float = -1e10  # Penalty for invalid tokens
    allow_early_termination: bool = True
    verbose: bool = False


@dataclass
class TokenMapping:
    """Mapping between characters and tokenizer tokens."""
    char_to_tokens: Dict[str, List[int]] = field(default_factory=dict)
    token_to_chars: Dict[int, str] = field(default_factory=dict)

    @classmethod
    def from_tokenizer(cls, tokenizer) -> "TokenMapping":
        """Build mapping from a tokenizer's vocabulary."""
        mapping = cls()

        vocab = tokenizer.get_vocab() if hasattr(tokenizer, 'get_vocab') else {}

        for token_str, token_id in vocab.items():
            # Decode the token to get characters
            try:
                decoded = tokenizer.decode([token_id])
                mapping.token_to_chars[token_id] = decoded

                # Map each starting character to tokens
                if decoded:
                    first_char = decoded[0]
                    if first_char not in mapping.char_to_tokens:
                        mapping.char_to_tokens[first_char] = []
                    mapping.char_to_tokens[first_char].append(token_id)
            except Exception:
                continue

        return mapping


class JSONGuidedSampler:
    """
    Sampler that constrains LLM generation to valid JSON.

    Uses the JSON grammar FSM to compute valid next tokens at each step,
    then masks invalid tokens in the logits before sampling.
    """

    def __init__(
        self,
        tokenizer,
        config: Optional[SamplerConfig] = None,
    ):
        self.tokenizer = tokenizer
        self.config = config or SamplerConfig()
        self.parser = JSONParser()
        self.token_mapping = self._build_token_mapping()

    def _build_token_mapping(self) -> TokenMapping:
        """Build token mapping from tokenizer."""
        if self.tokenizer is None:
            return TokenMapping()
        return TokenMapping.from_tokenizer(self.tokenizer)

    def get_valid_token_ids(self, valid_chars: Set[str]) -> Set[int]:
        """Get token IDs that start with valid characters."""
        valid_ids = set()

        for char in valid_chars:
            if char in self.token_mapping.char_to_tokens:
                valid_ids.update(self.token_mapping.char_to_tokens[char])

        return valid_ids

    def compute_mask(self, valid_chars: Set[str], vocab_size: int) -> List[float]:
        """
        Compute logits mask for valid tokens.

        Args:
            valid_chars: Set of valid next characters
            vocab_size: Size of vocabulary

        Returns:
            List of mask values (0 for valid, penalty for invalid)
        """
        mask = [self.config.mask_penalty] * vocab_size

        valid_ids = self.get_valid_token_ids(valid_chars)
        for token_id in valid_ids:
            if token_id < vocab_size:
                mask[token_id] = 0.0

        return mask

    def apply_mask(self, logits, valid_chars: Set[str]):
        """
        Apply grammar mask to logits.

        Args:
            logits: Model output logits
            valid_chars: Valid next characters

        Returns:
            Masked logits
        """
        if HAS_MLX:
            vocab_size = logits.shape[-1]
            mask = mx.array(self.compute_mask(valid_chars, vocab_size))
            return logits + mask
        elif HAS_NUMPY:
            vocab_size = logits.shape[-1]
            mask = np.array(self.compute_mask(valid_chars, vocab_size))
            return logits + mask
        else:
            return logits  # No masking without array library

    def sample_token(self, logits, valid_chars: Set[str]) -> int:
        """
        Sample a token with grammar constraints.

        Args:
            logits: Model output logits
            valid_chars: Valid next characters

        Returns:
            Sampled token ID
        """
        masked_logits = self.apply_mask(logits, valid_chars)

        if HAS_MLX:
            # Apply temperature
            if self.config.temperature > 0:
                masked_logits = masked_logits / self.config.temperature

            # Apply top-k
            if self.config.top_k > 0:
                top_k_indices = mx.argpartition(-masked_logits, self.config.top_k)[:self.config.top_k]
                mask = mx.full(masked_logits.shape, self.config.mask_penalty)
                mask = mask.at[top_k_indices].set(0.0)
                masked_logits = masked_logits + mask

            # Sample
            probs = mx.softmax(masked_logits)
            token_id = mx.random.categorical(mx.log(probs)).item()
            return token_id
        else:
            # Fallback: argmax from valid tokens
            valid_ids = self.get_valid_token_ids(valid_chars)
            if valid_ids:
                return min(valid_ids)  # Deterministic fallback
            return 0

    def generate(
        self,
        model,
        prompt: str,
        callback: Optional[Callable[[str, int], None]] = None,
    ) -> str:
        """
        Generate JSON with grammar constraints.

        Args:
            model: Language model with forward method
            prompt: Input prompt
            callback: Optional callback(token_str, token_id) for each token

        Returns:
            Generated JSON string
        """
        self.parser.reset()
        generated = ""
        tokens_generated = 0

        # Encode prompt
        if self.tokenizer:
            input_ids = self.tokenizer.encode(prompt)
        else:
            input_ids = []

        while tokens_generated < self.config.max_tokens:
            # Get valid next tokens from grammar
            valid_chars = self.parser.get_valid_next_tokens()

            if not valid_chars:
                if self.parser.is_complete():
                    break  # Valid JSON complete
                else:
                    if self.config.verbose:
                        print(f"[WARN] No valid tokens at position {tokens_generated}")
                    break

            # Get model logits
            if model is not None and hasattr(model, 'forward'):
                # Real model
                if HAS_MLX:
                    logits = model.forward(mx.array([input_ids]))[:, -1, :]
                    logits = logits.squeeze()
                else:
                    # Mock: use grammar directly
                    token_id = self._mock_sample(valid_chars)
                    token_str = self._decode_token(token_id)
            else:
                # Mock generation
                token_id = self._mock_sample(valid_chars)
                token_str = self._decode_token(token_id)
                logits = None

            if logits is not None:
                # Sample with grammar constraints
                token_id = self.sample_token(logits, valid_chars)
                token_str = self._decode_token(token_id)

            # Feed token to parser
            for char in token_str:
                if not self.parser.feed(char):
                    if self.config.verbose:
                        print(f"[WARN] Parser rejected char '{char}'")
                    # Truncate at invalid char
                    break

            generated += token_str
            tokens_generated += 1

            if callback:
                callback(token_str, token_id)

            # Update input for next iteration
            if self.tokenizer:
                input_ids.append(token_id)

            # Check for early termination
            if self.config.allow_early_termination and self.parser.is_complete():
                break

        return generated

    def _mock_sample(self, valid_chars: Set[str]) -> int:
        """Mock sampling for testing without model."""
        # Priority order for JSON structure
        priority = ['{', '[', '"', ':', ',', '}', ']', '0', '1', 't', 'f', 'n']

        for char in priority:
            if char in valid_chars:
                # Return a pseudo token ID
                return ord(char)

        # Fallback: any valid char
        if valid_chars:
            return ord(next(iter(valid_chars)))
        return 0

    def _decode_token(self, token_id: int) -> str:
        """Decode token ID to string."""
        if self.tokenizer and token_id in self.token_mapping.token_to_chars:
            return self.token_mapping.token_to_chars[token_id]
        # Fallback: treat as char code
        if 32 <= token_id < 127:
            return chr(token_id)
        return ""


def sample_with_grammar(
    model,
    tokenizer,
    prompt: str,
    config: Optional[SamplerConfig] = None,
) -> Tuple[str, bool]:
    """
    Generate JSON with grammar-guided sampling.

    Args:
        model: Language model
        tokenizer: Tokenizer
        prompt: Input prompt
        config: Sampling configuration

    Returns:
        (generated_json, is_valid)
    """
    sampler = JSONGuidedSampler(tokenizer, config)
    output = sampler.generate(model, prompt)

    # Validate
    try:
        json.loads(output)
        return output, True
    except json.JSONDecodeError:
        return output, False


class MockModel:
    """Mock model for testing without actual LLM."""

    def __init__(self, template: str = "object"):
        self.template = template
        self.position = 0
        self._output = self._get_template_output()

    def _get_template_output(self) -> str:
        """Get template JSON based on type."""
        templates = {
            "object": '{"key": "value"}',
            "array": '[1, 2, 3]',
            "nested": '{"a": {"b": [1, 2]}}',
            "complex": '{"name": "test", "count": 42, "active": true, "data": null}',
        }
        return templates.get(self.template, '{}')

    def forward(self, input_ids):
        """Mock forward pass returning logits for next char."""
        if self.position >= len(self._output):
            return None

        next_char = self._output[self.position]
        self.position += 1

        # Create mock logits with high score for expected char
        if HAS_MLX:
            logits = mx.full((1, 1, 128), -100.0)
            logits = logits.at[0, 0, ord(next_char)].set(10.0)
            return logits
        elif HAS_NUMPY:
            logits = np.full((1, 1, 128), -100.0)
            logits[0, 0, ord(next_char)] = 10.0
            return logits
        return None
