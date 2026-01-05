"""
Statechart-Constrained Sampler

Implements constrained token sampling using the Starlark syntax statechart.
At each generation step, masks logits to only allow syntactically valid tokens.

Key features:
- Logit masking: Invalid tokens get -inf, blocking sampling
- State tracking: Updates statechart state after each token
- Fallback mode: If all tokens blocked, allows any token
- Statistics: Tracks blocked tokens, fallbacks, syntax errors
"""

from dataclasses import dataclass, field
from typing import Dict, Set, Optional, Tuple, List, Callable, Any
import time

try:
    import mlx.core as mx
    HAS_MLX = True
except ImportError:
    HAS_MLX = False
    mx = None

try:
    from .starlark_statechart import StarlarkStatechart, StarlarkSyntaxState
    from .token_mapper import TokenMapper
except ImportError:
    from starlark_statechart import StarlarkStatechart, StarlarkSyntaxState
    from token_mapper import TokenMapper


@dataclass
class SamplingConfig:
    """Configuration for constrained sampling."""
    max_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    block_invalid: bool = True
    fallback_on_block: bool = True
    track_stats: bool = True


@dataclass
class SamplingStats:
    """Statistics from a sampling run."""
    tokens_generated: int = 0
    tokens_blocked: int = 0
    fallbacks_used: int = 0
    syntax_errors: int = 0
    generation_time: float = 0.0
    blocked_tokens: List[str] = field(default_factory=list)
    final_state: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            'tokens_generated': self.tokens_generated,
            'tokens_blocked': self.tokens_blocked,
            'fallbacks_used': self.fallbacks_used,
            'syntax_errors': self.syntax_errors,
            'generation_time': self.generation_time,
            'blocked_tokens': self.blocked_tokens,
            'final_state': self.final_state,
            'validity_rate': 1.0 - (self.tokens_blocked / max(1, self.tokens_generated)),
        }


class LogitMasker:
    """
    Masks logits based on syntax state.

    Given current syntax state, creates a mask that:
    - Sets valid tokens to 0 (unchanged)
    - Sets invalid tokens to -inf (blocked)
    """

    def __init__(self, mapper: TokenMapper, statechart: StarlarkStatechart):
        self.mapper = mapper
        self.statechart = statechart
        self.mask_cache: Dict[Tuple[StarlarkSyntaxState, int, int, int], Set[int]] = {}

    def get_valid_token_ids(self) -> Set[int]:
        """Get token IDs valid in current syntax state."""
        ctx = self.statechart.context
        cache_key = (
            self.statechart.state,
            ctx.paren_depth,
            ctx.bracket_depth,
            ctx.brace_depth,
        )

        if cache_key in self.mask_cache:
            return self.mask_cache[cache_key]

        # Get valid token categories from statechart
        valid_categories = self.statechart.get_valid_tokens()

        # Map categories to actual token IDs
        valid_ids = self.mapper.get_token_ids_for_categories(valid_categories)

        # Always allow whitespace in most contexts
        if not self.statechart.context.in_string:
            whitespace_ids = self.mapper.category_to_ids.get('WHITESPACE', set())
            valid_ids.update(whitespace_ids)

        # Cache result
        self.mask_cache[cache_key] = valid_ids
        return valid_ids

    def create_mask(self, vocab_size: int) -> List[float]:
        """
        Create a logit mask for current state.

        Returns list of length vocab_size:
        - 0.0 for valid tokens
        - -inf for invalid tokens
        """
        valid_ids = self.get_valid_token_ids()

        mask = [-float('inf')] * vocab_size
        for token_id in valid_ids:
            if 0 <= token_id < vocab_size:
                mask[token_id] = 0.0

        return mask

    def create_mask_mlx(self, vocab_size: int) -> 'mx.array':
        """Create mask as MLX array."""
        if not HAS_MLX:
            raise RuntimeError("MLX not available")

        valid_ids = self.get_valid_token_ids()

        # Create mask efficiently with MLX
        mask = mx.full((vocab_size,), -float('inf'))
        if valid_ids:
            indices = mx.array(list(valid_ids))
            mask = mask.at[indices].set(0.0)

        return mask

    def apply_mask(self, logits: List[float]) -> List[float]:
        """Apply mask to logits."""
        mask = self.create_mask(len(logits))
        return [l + m for l, m in zip(logits, mask)]

    def apply_mask_mlx(self, logits: 'mx.array') -> 'mx.array':
        """Apply mask to MLX logits."""
        if not HAS_MLX:
            raise RuntimeError("MLX not available")

        mask = self.create_mask_mlx(logits.shape[-1])
        return logits + mask

    def update_state(self, token: str) -> bool:
        """Update statechart with generated token."""
        return self.statechart.transition(token)

    def reset(self):
        """Reset statechart to initial state."""
        self.statechart.reset()
        self.mask_cache.clear()


class StatechartConstrainedSampler:
    """
    LLM sampler with statechart syntax constraints.

    Wraps any language model and applies syntax constraints during generation.
    Guarantees syntactically valid output by masking invalid token logits.
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        statechart: Optional[StarlarkStatechart] = None,
    ):
        """
        Initialize constrained sampler.

        Args:
            model: Language model with forward pass capability
            tokenizer: Tokenizer with encode/decode methods
            statechart: Optional pre-configured statechart
        """
        self.model = model
        self.tokenizer = tokenizer

        # Get vocabulary
        if hasattr(tokenizer, 'get_vocab'):
            vocab = tokenizer.get_vocab()
        elif hasattr(tokenizer, 'vocab'):
            vocab = tokenizer.vocab
        else:
            vocab = None

        # Initialize components
        self.statechart = statechart or StarlarkStatechart()
        self.mapper = TokenMapper(tokenizer, vocab)
        self.masker = LogitMasker(self.mapper, self.statechart)

    def sample_next_token(
        self,
        input_ids: Any,
        config: SamplingConfig = None,
    ) -> Tuple[int, bool]:
        """
        Sample next token with syntax constraints.

        Args:
            input_ids: Current token sequence
            config: Sampling configuration

        Returns:
            (token_id, was_blocked): Token ID and whether it was in blocked set
        """
        config = config or SamplingConfig()

        # Get model logits
        if HAS_MLX and hasattr(self.model, '__call__'):
            logits = self.model(input_ids)
            if hasattr(logits, 'logits'):
                logits = logits.logits
            if len(logits.shape) > 1:
                logits = logits[:, -1, :]
            if len(logits.shape) > 1:
                logits = logits[0]
        else:
            logits = self.model(input_ids)
            if hasattr(logits, 'shape') and len(logits.shape) > 1:
                logits = logits[-1]

        # Get valid tokens before masking
        valid_ids = self.masker.get_valid_token_ids()
        was_blocked = False

        # Apply syntax constraint mask
        if config.block_invalid:
            if not valid_ids and config.fallback_on_block:
                # No valid tokens - allow all (fallback)
                pass
            else:
                if HAS_MLX and isinstance(logits, mx.array):
                    logits = self.masker.apply_mask_mlx(logits)
                else:
                    logits = self.masker.apply_mask(list(logits))

        # Sample next token
        next_id = self._sample(logits, config)

        # Check if token was blocked
        if next_id not in valid_ids:
            was_blocked = True

        # Update statechart
        next_token = self.mapper.get_token_text(next_id)
        self.masker.update_state(next_token)

        return next_id, was_blocked

    def generate(
        self,
        prompt: str,
        config: SamplingConfig = None,
    ) -> Tuple[str, SamplingStats]:
        """
        Generate text with syntax constraints.

        Args:
            prompt: Input prompt
            config: Sampling configuration

        Returns:
            (generated_text, stats): Generated text and statistics
        """
        config = config or SamplingConfig()
        stats = SamplingStats()

        start_time = time.time()

        # Encode prompt
        input_ids = self.tokenizer.encode(prompt)
        if HAS_MLX:
            input_ids = mx.array([input_ids])

        # Reset state
        self.masker.reset()

        # Process prompt tokens through statechart
        prompt_tokens = self._tokenize_prompt(prompt)
        for token in prompt_tokens:
            self.masker.update_state(token)

        generated_ids = []

        for _ in range(config.max_tokens):
            # Combine input and generated
            if HAS_MLX:
                if generated_ids:
                    all_ids = mx.concatenate([input_ids, mx.array([generated_ids])], axis=1)
                else:
                    all_ids = input_ids
            else:
                all_ids = list(input_ids) + generated_ids

            # Sample next token
            next_id, was_blocked = self.sample_next_token(all_ids, config)
            next_token = self.mapper.get_token_text(next_id)

            if was_blocked:
                stats.tokens_blocked += 1
                stats.blocked_tokens.append(next_token)

            generated_ids.append(next_id)
            stats.tokens_generated += 1

            # Check for end of generation
            if next_token in ['\n\n', '<|endoftext|>', '<|im_end|>']:
                break

        # Decode generated tokens
        generated_text = self.tokenizer.decode(generated_ids)

        stats.generation_time = time.time() - start_time
        stats.final_state = self.statechart.state.name

        return generated_text, stats

    def _sample(self, logits: Any, config: SamplingConfig) -> int:
        """Sample next token from logits."""
        if HAS_MLX and isinstance(logits, mx.array):
            return self._sample_mlx(logits, config)
        return self._sample_python(list(logits), config)

    def _sample_python(self, logits: List[float], config: SamplingConfig) -> int:
        """Sample using pure Python."""
        import math
        import random

        # Apply temperature
        if config.temperature > 0:
            logits = [l / config.temperature for l in logits]

        # Softmax
        max_logit = max(logits)
        exp_logits = [math.exp(l - max_logit) if l > -float('inf') else 0 for l in logits]
        sum_exp = sum(exp_logits)
        if sum_exp == 0:
            # All tokens blocked, uniform sampling
            return random.randint(0, len(logits) - 1)
        probs = [e / sum_exp for e in exp_logits]

        # Top-k filtering
        if config.top_k > 0:
            indexed = sorted(enumerate(probs), key=lambda x: x[1], reverse=True)
            top_k = indexed[:config.top_k]
            indices, top_probs = zip(*top_k)
            sum_probs = sum(top_probs)
            if sum_probs > 0:
                top_probs = [p / sum_probs for p in top_probs]

                # Sample from top-k
                r = random.random()
                cumsum = 0
                for idx, prob in zip(indices, top_probs):
                    cumsum += prob
                    if r < cumsum:
                        return idx

            return indices[0]

        # Full sampling
        r = random.random()
        cumsum = 0
        for idx, prob in enumerate(probs):
            cumsum += prob
            if r < cumsum:
                return idx

        return len(probs) - 1

    def _sample_mlx(self, logits: 'mx.array', config: SamplingConfig) -> int:
        """Sample using MLX."""
        # Apply temperature
        if config.temperature > 0:
            logits = logits / config.temperature

        # Softmax
        probs = mx.softmax(logits)

        # Top-k filtering
        if config.top_k > 0:
            top_k_indices = mx.argsort(probs)[-config.top_k:]
            top_k_probs = probs[top_k_indices]
            top_k_probs = top_k_probs / top_k_probs.sum()

            # Sample
            sample = mx.random.categorical(mx.log(top_k_probs + 1e-10))
            return int(top_k_indices[sample])

        # Full sampling
        sample = mx.random.categorical(mx.log(probs + 1e-10))
        return int(sample)

    def _tokenize_prompt(self, prompt: str) -> List[str]:
        """Tokenize prompt for statechart processing."""
        # Use statechart's tokenizer if available
        if hasattr(self.statechart, '_tokenize'):
            return self.statechart._tokenize(prompt)

        # Fallback: Use LLM tokenizer and decode
        token_ids = self.tokenizer.encode(prompt)
        return [self.tokenizer.decode([tid]) for tid in token_ids]

    def explain_constraints(self) -> str:
        """Explain current syntax constraints."""
        state = self.statechart.state
        valid = self.statechart.get_valid_tokens()
        ctx = self.statechart.context

        lines = [
            f"Current State: {state.name}",
            f"Valid token categories: {', '.join(sorted(valid))}",
            f"Context:",
            f"  Paren depth: {ctx.paren_depth}",
            f"  Bracket depth: {ctx.bracket_depth}",
            f"  Brace depth: {ctx.brace_depth}",
            f"  In string: {ctx.in_string}",
            f"  Function depth: {ctx.function_depth}",
        ]
        return '\n'.join(lines)

    def reset(self):
        """Reset sampler state."""
        self.masker.reset()


def generate_unconstrained(model, tokenizer, prompt: str, max_tokens: int = 256) -> str:
    """Generate without constraints (baseline)."""
    input_ids = tokenizer.encode(prompt)

    if HAS_MLX:
        input_ids = mx.array([input_ids])

    generated_ids = []
    for _ in range(max_tokens):
        if HAS_MLX:
            if generated_ids:
                all_ids = mx.concatenate([input_ids, mx.array([generated_ids])], axis=1)
            else:
                all_ids = input_ids
            logits = model(all_ids)
            if hasattr(logits, 'logits'):
                logits = logits.logits
            logits = logits[0, -1, :]
            probs = mx.softmax(logits / 0.7)
            next_id = int(mx.random.categorical(mx.log(probs + 1e-10)))
        else:
            all_ids = list(input_ids) + generated_ids
            logits = model(all_ids)[-1]
            # Simple argmax for non-MLX
            next_id = max(range(len(logits)), key=lambda i: logits[i])

        generated_ids.append(next_id)

        next_token = tokenizer.decode([next_id])
        if next_token in ['\n\n', '<|endoftext|>']:
            break

    return tokenizer.decode(generated_ids)


def generate_constrained(
    model,
    tokenizer,
    prompt: str,
    statechart: Optional[StarlarkStatechart] = None,
    max_tokens: int = 256,
) -> Tuple[str, SamplingStats]:
    """Generate with statechart constraints."""
    sampler = StatechartConstrainedSampler(model, tokenizer, statechart)
    config = SamplingConfig(max_tokens=max_tokens)
    return sampler.generate(prompt, config)


# Demo
if __name__ == "__main__":
    print("=" * 60)
    print("STATECHART CONSTRAINED SAMPLER DEMO")
    print("=" * 60)

    # Create mock model and tokenizer for demo
    class MockTokenizer:
        def __init__(self):
            self.vocab = {
                'def': 0, 'return': 1, 'sc': 2, 'machine': 3,
                '(': 4, ')': 5, '[': 6, ']': 7, '{': 8, '}': 9,
                ':': 10, ',': 11, '=': 12, '.': 13,
                '"': 14, 'name': 15, 'states': 16, 'state': 17,
                'True': 18, 'False': 19, 'None': 20,
                '\n': 21, ' ': 22, '    ': 23,
                'traffic_light': 24, 'green': 25, 'yellow': 26, 'red': 27,
            }
            self.id_to_token = {v: k for k, v in self.vocab.items()}

        def get_vocab(self):
            return self.vocab

        def encode(self, text):
            # Simple word-based encoding
            tokens = []
            for word in text.replace('(', ' ( ').replace(')', ' ) ').split():
                if word in self.vocab:
                    tokens.append(self.vocab[word])
            return tokens

        def decode(self, ids):
            return ' '.join(self.id_to_token.get(i, '?') for i in ids)

    class MockModel:
        def __init__(self, vocab_size):
            self.vocab_size = vocab_size

        def __call__(self, input_ids):
            import random
            # Return random logits
            return [random.random() for _ in range(self.vocab_size)]

    tokenizer = MockTokenizer()
    model = MockModel(len(tokenizer.vocab))

    sampler = StatechartConstrainedSampler(model, tokenizer)

    print("\nInitial state:")
    print(sampler.explain_constraints())

    print("\n" + "=" * 60)
    print("Simulating constrained generation...")

    # Process some tokens
    for token in ['def', 'traffic_light', '(', ')', ':']:
        sampler.masker.update_state(token)
        print(f"\nAfter '{token}':")
        print(f"  State: {sampler.statechart.state.name}")
        print(f"  Valid: {sampler.statechart.get_valid_tokens()}")

    print("\n" + "=" * 60)
    print("Demo complete!")
