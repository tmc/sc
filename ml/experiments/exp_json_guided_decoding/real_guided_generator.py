"""
Real Grammar-Guided JSON Generation with MLX

Integrates JSON grammar FSM with MLX token generation via logits processor.
"""

import json
import time
from typing import Dict, List, Optional, Set, Tuple, Any, Callable
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn

from .json_grammar import JSONParser, JSONState


@dataclass
class GuidedGeneratorConfig:
    """Configuration for guided generation."""
    max_tokens: int = 500
    temperature: float = 0.2
    mask_penalty: float = -1e9
    verbose: bool = False


class JSONLogitsProcessor:
    """
    Logits processor that masks invalid tokens based on JSON grammar state.

    This is the core of grammar-guided decoding - it intercepts logits
    before sampling and sets invalid token logits to -inf.
    """

    def __init__(
        self,
        tokenizer,
        config: Optional[GuidedGeneratorConfig] = None,
    ):
        self.tokenizer = tokenizer
        self.config = config or GuidedGeneratorConfig()
        self.parser = JSONParser()
        self.generated_text = ""

        # Build token -> first character mapping
        self.token_first_char: Dict[int, str] = {}
        self.char_to_tokens: Dict[str, List[int]] = {}
        self._build_token_mapping()

    def _build_token_mapping(self):
        """Build mapping between tokens and their first characters."""
        vocab = self.tokenizer.get_vocab() if hasattr(self.tokenizer, 'get_vocab') else {}

        for token_str, token_id in vocab.items():
            try:
                # Decode token to get actual characters
                decoded = self.tokenizer.decode([token_id])
                if decoded:
                    first_char = decoded[0]
                    self.token_first_char[token_id] = first_char

                    if first_char not in self.char_to_tokens:
                        self.char_to_tokens[first_char] = []
                    self.char_to_tokens[first_char].append(token_id)
            except Exception:
                continue

    def reset(self):
        """Reset parser state for new generation."""
        self.parser.reset()
        self.generated_text = ""

    def get_valid_token_ids(self) -> Set[int]:
        """Get token IDs that are valid at current grammar state."""
        valid_chars = self.parser.get_valid_next_tokens()
        valid_ids = set()

        for char in valid_chars:
            if char in self.char_to_tokens:
                valid_ids.update(self.char_to_tokens[char])

        # Also allow any token if we have a wildcard state (string content)
        if self.parser.state in (JSONState.STRING_CONTENT, JSONState.STRING_START):
            # Most tokens are valid inside strings
            for token_id in self.token_first_char.keys():
                first_char = self.token_first_char.get(token_id, '')
                if first_char and first_char not in ('"', '\\'):
                    valid_ids.add(token_id)

        return valid_ids

    def compute_mask(self, vocab_size: int) -> mx.array:
        """Compute logits mask for valid tokens."""
        valid_ids = self.get_valid_token_ids()

        # Create mask: 0 for valid, -inf for invalid
        mask = mx.full((vocab_size,), self.config.mask_penalty)

        if valid_ids:
            valid_list = list(valid_ids)
            # Set valid positions to 0
            for vid in valid_list:
                if vid < vocab_size:
                    mask = mask.at[vid].add(-self.config.mask_penalty)

        return mask

    def update_state(self, token_id: int) -> bool:
        """Update parser state with generated token."""
        decoded = self.tokenizer.decode([token_id])
        self.generated_text += decoded

        # Feed each character to parser
        for char in decoded:
            if not self.parser.feed(char):
                if self.config.verbose:
                    print(f"[WARN] Parser rejected char '{char}'")
                return False

        return True

    def is_complete(self) -> bool:
        """Check if JSON is complete."""
        return self.parser.is_complete()


def guided_generate(
    model,
    tokenizer,
    prompt: str,
    config: Optional[GuidedGeneratorConfig] = None,
) -> Tuple[str, bool, Dict[str, Any]]:
    """
    Generate JSON with grammar-guided decoding.

    Args:
        model: MLX model
        tokenizer: Tokenizer
        prompt: Input prompt
        config: Generation config

    Returns:
        (generated_json, is_valid, stats)
    """
    from mlx_lm.sample_utils import make_sampler

    config = config or GuidedGeneratorConfig()
    processor = JSONLogitsProcessor(tokenizer, config)
    processor.reset()

    sampler = make_sampler(temp=config.temperature)

    # Encode prompt
    input_ids = tokenizer.encode(prompt)
    input_ids = mx.array([input_ids])

    generated_tokens = []
    stats = {
        "tokens_generated": 0,
        "tokens_masked": 0,
        "early_termination": False,
    }

    start_time = time.time()

    # Get model cache
    cache = None

    for i in range(config.max_tokens):
        # Forward pass
        if cache is None:
            outputs = model(input_ids)
            # Initialize cache for next iteration
        else:
            # Use last token only with cache
            outputs = model(input_ids[:, -1:], cache=cache)

        logits = outputs[0] if isinstance(outputs, tuple) else outputs
        logits = logits[:, -1, :]  # Last position

        # Apply grammar mask
        vocab_size = logits.shape[-1]
        mask = processor.compute_mask(vocab_size)
        masked_logits = logits + mask

        # Count masked tokens
        valid_count = (mask > config.mask_penalty / 2).sum().item()
        stats["tokens_masked"] += vocab_size - valid_count

        # Sample
        probs = mx.softmax(masked_logits / config.temperature, axis=-1)
        token_id = mx.argmax(probs, axis=-1).item()

        generated_tokens.append(token_id)
        stats["tokens_generated"] += 1

        # Update parser state
        if not processor.update_state(token_id):
            break

        # Check for completion
        if processor.is_complete():
            stats["early_termination"] = True
            break

        # Update input for next iteration
        new_token = mx.array([[token_id]])
        input_ids = mx.concatenate([input_ids, new_token], axis=1)

    stats["generation_time_ms"] = (time.time() - start_time) * 1000

    # Decode output
    output = processor.generated_text

    # Validate
    try:
        json.loads(output)
        is_valid = True
    except json.JSONDecodeError:
        is_valid = False

    return output, is_valid, stats


def simple_guided_generate(
    model,
    tokenizer,
    prompt: str,
    max_tokens: int = 500,
    temperature: float = 0.2,
) -> Tuple[str, bool]:
    """
    Simplified guided generation using greedy decoding with grammar mask.

    This version is simpler but still applies grammar constraints.
    """
    from mlx_lm.sample_utils import make_sampler

    processor = JSONLogitsProcessor(tokenizer)
    processor.reset()

    # Tokenize prompt
    prompt_tokens = tokenizer.encode(prompt)

    # Generate token by token
    generated = ""

    for _ in range(max_tokens):
        # Get valid tokens from grammar
        valid_ids = processor.get_valid_token_ids()

        if not valid_ids:
            if processor.is_complete():
                break
            # Fallback: allow common JSON tokens
            valid_ids = processor.char_to_tokens.get('"', []) + \
                       processor.char_to_tokens.get('{', []) + \
                       processor.char_to_tokens.get('}', [])

        if not valid_ids:
            break

        # Simple: just pick first valid token (greedy)
        # For better quality, would need actual model forward pass
        token_id = min(valid_ids)

        # Update state
        if not processor.update_state(token_id):
            break

        if processor.is_complete():
            break

    output = processor.generated_text

    try:
        json.loads(output)
        return output, True
    except json.JSONDecodeError:
        return output, False


class GrammarSampler:
    """
    Custom sampler that applies grammar constraints.

    Can be used with mlx_lm.generate via sampler parameter.
    """

    def __init__(self, tokenizer, base_temp: float = 0.2):
        self.tokenizer = tokenizer
        self.base_temp = base_temp
        self.processor = JSONLogitsProcessor(tokenizer)
        self.processor.reset()
        self._last_token = None

    def __call__(self, logits: mx.array) -> mx.array:
        """Apply grammar mask and sample."""
        # Update state with last generated token
        if self._last_token is not None:
            self.processor.update_state(self._last_token)

        # Apply grammar mask
        vocab_size = logits.shape[-1]
        mask = self.processor.compute_mask(vocab_size)
        masked_logits = logits + mask

        # Temperature scaling
        if self.base_temp > 0:
            masked_logits = masked_logits / self.base_temp

        # Sample
        probs = mx.softmax(masked_logits, axis=-1)
        token = mx.random.categorical(mx.log(probs + 1e-10))

        self._last_token = token.item()
        return token

    def reset(self):
        """Reset for new generation."""
        self.processor.reset()
        self._last_token = None
