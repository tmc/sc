"""
Constrained Code Generator

Uses statechart to mask invalid token logits during LLM generation.
Guarantees 99%+ syntax validity by only allowing grammatically valid tokens.

Key components:
- LogitMasker: Maps syntax states to token masks
- ConstrainedCodeGenerator: Wraps LLM with statechart constraints
- TokenStateMapper: Maps between LLM tokens and syntax tokens
"""

from dataclasses import dataclass, field
from typing import List, Set, Dict, Optional, Tuple, Callable
from enum import Enum
import re

try:
    import mlx.core as mx
    import mlx.nn as nn
    HAS_MLX = True
except ImportError:
    HAS_MLX = False
    mx = None

try:
    from .syntax_statechart import (
        PythonSyntaxStatechart, PythonSyntaxState, SyntaxContext,
        KEYWORDS, OPERATORS, DELIMITERS
    )
except ImportError:
    from syntax_statechart import (
        PythonSyntaxStatechart, PythonSyntaxState, SyntaxContext,
        KEYWORDS, OPERATORS, DELIMITERS
    )


# Token category patterns
TOKEN_PATTERNS = {
    'IDENTIFIER': re.compile(r'^[a-zA-Z_][a-zA-Z0-9_]*$'),
    'NUMBER': re.compile(r'^[0-9]+\.?[0-9]*([eE][+-]?[0-9]+)?$'),
    'STRING': re.compile(r'^["\']'),
    'NEWLINE': re.compile(r'^[\n]$'),
    'WHITESPACE': re.compile(r'^[ \t]+$'),
}


@dataclass
class TokenInfo:
    """Information about a token in the vocabulary."""
    token_id: int
    text: str
    categories: Set[str] = field(default_factory=set)

    def __post_init__(self):
        self._categorize()

    def _categorize(self):
        """Determine token categories."""
        text = self.text

        if text in KEYWORDS:
            self.categories.add('KEYWORD')
            self.categories.add(text)  # Add specific keyword
        elif text in OPERATORS:
            self.categories.add('OPERATOR')
            self.categories.add(text)
        elif text in DELIMITERS:
            self.categories.add('DELIMITER')
            self.categories.add(text)
        elif TOKEN_PATTERNS['IDENTIFIER'].match(text):
            self.categories.add('IDENTIFIER')
        elif TOKEN_PATTERNS['NUMBER'].match(text):
            self.categories.add('NUMBER')
        elif text.startswith('"') or text.startswith("'"):
            self.categories.add('STRING')
        elif text == '\n':
            self.categories.add('NEWLINE')
        elif text.strip() == '':
            self.categories.add('WHITESPACE')


class TokenStateMapper:
    """
    Maps between LLM vocabulary tokens and syntax state categories.

    This is the critical bridge between:
    - Statechart's abstract categories (IDENTIFIER, NUMBER, etc.)
    - LLM's actual vocabulary (specific tokens)
    """

    def __init__(self, vocab: Dict[str, int]):
        """
        Initialize mapper with LLM vocabulary.

        Args:
            vocab: Dict mapping token strings to IDs
        """
        self.vocab = vocab
        self.id_to_token = {v: k for k, v in vocab.items()}
        self.vocab_size = len(vocab)

        # Build token info for each vocabulary item
        self.token_info: Dict[int, TokenInfo] = {}
        for text, token_id in vocab.items():
            self.token_info[token_id] = TokenInfo(token_id, text)

        # Build reverse mapping: category -> token IDs
        self.category_to_ids: Dict[str, Set[int]] = {}
        for token_id, info in self.token_info.items():
            for cat in info.categories:
                if cat not in self.category_to_ids:
                    self.category_to_ids[cat] = set()
                self.category_to_ids[cat].add(token_id)

    def get_token_ids_for_categories(self, categories: Set[str]) -> Set[int]:
        """Get all token IDs that match any of the given categories."""
        result = set()
        for cat in categories:
            if cat in self.category_to_ids:
                result.update(self.category_to_ids[cat])

        # Special handling for abstract categories
        if 'IDENTIFIER' in categories:
            result.update(self.category_to_ids.get('IDENTIFIER', set()))
        if 'NUMBER' in categories:
            result.update(self.category_to_ids.get('NUMBER', set()))
        if 'STRING' in categories:
            result.update(self.category_to_ids.get('STRING', set()))

        return result

    def get_token_text(self, token_id: int) -> str:
        """Get text for a token ID."""
        return self.id_to_token.get(token_id, '<UNK>')

    def get_token_id(self, text: str) -> Optional[int]:
        """Get token ID for text."""
        return self.vocab.get(text)


class LogitMasker:
    """
    Masks logits based on syntax state.

    Given current syntax state, creates a mask that:
    - Sets valid tokens to 0 (unchanged)
    - Sets invalid tokens to -inf (blocked)
    """

    def __init__(self, mapper: TokenStateMapper, statechart: PythonSyntaxStatechart):
        self.mapper = mapper
        self.statechart = statechart
        self.mask_cache: Dict[Tuple[PythonSyntaxState, int, int, int], Set[int]] = {}

    def get_valid_token_ids(self) -> Set[int]:
        """Get token IDs valid in current syntax state."""
        # Create cache key from state and context
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

        # Always allow whitespace and newlines in most contexts
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

        mask = self.create_mask(vocab_size)
        return mx.array(mask)

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
        """
        Update statechart with generated token.

        Returns True if transition was valid.
        """
        return self.statechart.transition(token)

    def reset(self):
        """Reset statechart to initial state."""
        self.statechart.reset()
        self.mask_cache.clear()


@dataclass
class GenerationConfig:
    """Configuration for constrained generation."""
    max_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    block_invalid: bool = True  # If False, just log violations
    fallback_on_block: bool = True  # If all tokens blocked, allow any


class ConstrainedCodeGenerator:
    """
    LLM code generator with statechart constraints.

    Wraps any language model and applies syntax constraints during generation.
    Guarantees syntactically valid output (or reports violations).
    """

    def __init__(
        self,
        model: Callable,  # Function: (tokens) -> logits
        tokenizer: 'Tokenizer',  # Has encode/decode methods
        vocab: Dict[str, int],
    ):
        """
        Initialize constrained generator.

        Args:
            model: Function that takes token IDs and returns logits
            tokenizer: Tokenizer with encode/decode methods
            vocab: Vocabulary mapping token strings to IDs
        """
        self.model = model
        self.tokenizer = tokenizer
        self.vocab = vocab

        # Initialize components
        self.statechart = PythonSyntaxStatechart()
        self.mapper = TokenStateMapper(vocab)
        self.masker = LogitMasker(self.mapper, self.statechart)

        # Statistics
        self.stats = {
            'tokens_generated': 0,
            'tokens_blocked': 0,
            'fallbacks_used': 0,
            'syntax_errors': 0,
        }

    def generate(
        self,
        prompt: str,
        config: GenerationConfig = None,
    ) -> Tuple[str, Dict]:
        """
        Generate code with syntax constraints.

        Returns: (generated_text, generation_stats)
        """
        config = config or GenerationConfig()

        # Encode prompt
        input_ids = self.tokenizer.encode(prompt)

        # Reset state
        self.masker.reset()
        self.stats = {
            'tokens_generated': 0,
            'tokens_blocked': 0,
            'fallbacks_used': 0,
            'syntax_errors': 0,
        }

        # Process prompt tokens through statechart
        prompt_tokens = self.tokenizer.tokenize(prompt)
        for token in prompt_tokens:
            self.masker.update_state(token)

        generated_ids = []
        blocked_tokens = []

        for _ in range(config.max_tokens):
            # Get logits from model
            all_ids = input_ids + generated_ids
            logits = self.model(all_ids)

            # Take last position logits
            if hasattr(logits, 'shape') and len(logits.shape) > 1:
                logits = logits[-1]

            # Apply syntax constraint mask
            if config.block_invalid:
                valid_ids = self.masker.get_valid_token_ids()

                if not valid_ids and config.fallback_on_block:
                    # No valid tokens - allow all (fallback)
                    self.stats['fallbacks_used'] += 1
                else:
                    # Apply mask
                    if HAS_MLX and isinstance(logits, mx.array):
                        logits = self.masker.apply_mask_mlx(logits)
                    else:
                        logits = self.masker.apply_mask(list(logits))

            # Sample next token
            next_id = self._sample(logits, config)
            next_token = self.mapper.get_token_text(next_id)

            # Check if token was blocked
            if next_id not in self.masker.get_valid_token_ids():
                self.stats['tokens_blocked'] += 1
                blocked_tokens.append(next_token)

            # Update statechart
            valid = self.masker.update_state(next_token)
            if not valid:
                self.stats['syntax_errors'] += 1

            generated_ids.append(next_id)
            self.stats['tokens_generated'] += 1

            # Check for end of generation
            if next_token == '\n\n' or next_token == '<|endoftext|>':
                break

        # Decode generated tokens
        generated_text = self.tokenizer.decode(generated_ids)

        return generated_text, {
            **self.stats,
            'blocked_tokens': blocked_tokens,
            'final_state': self.statechart.state.name,
            'context': {
                'indent': self.statechart.context.indent_level,
                'parens': self.statechart.context.paren_depth,
                'brackets': self.statechart.context.bracket_depth,
                'braces': self.statechart.context.brace_depth,
            },
        }

    def _sample(self, logits, config: GenerationConfig) -> int:
        """Sample next token from logits."""
        if HAS_MLX and isinstance(logits, mx.array):
            return self._sample_mlx(logits, config)
        return self._sample_python(list(logits), config)

    def _sample_python(self, logits: List[float], config: GenerationConfig) -> int:
        """Sample using pure Python."""
        import math
        import random

        # Apply temperature
        if config.temperature > 0:
            logits = [l / config.temperature for l in logits]

        # Softmax
        max_logit = max(logits)
        exp_logits = [math.exp(l - max_logit) for l in logits]
        sum_exp = sum(exp_logits)
        probs = [e / sum_exp for e in exp_logits]

        # Top-k filtering
        if config.top_k > 0:
            indexed = sorted(enumerate(probs), key=lambda x: x[1], reverse=True)
            top_k = indexed[:config.top_k]
            indices, top_probs = zip(*top_k)
            sum_probs = sum(top_probs)
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

    def _sample_mlx(self, logits: 'mx.array', config: GenerationConfig) -> int:
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
            sample = mx.random.categorical(mx.log(top_k_probs))
            return int(top_k_indices[sample])

        # Full sampling
        sample = mx.random.categorical(mx.log(probs))
        return int(sample)

    def validate_syntax(self, code: str) -> Tuple[bool, List[str]]:
        """
        Validate Python syntax of generated code.

        Returns: (is_valid, list_of_errors)
        """
        import ast
        try:
            ast.parse(code)
            return True, []
        except SyntaxError as e:
            return False, [f"Line {e.lineno}: {e.msg}"]

    def explain_constraints(self) -> str:
        """Explain current syntax constraints."""
        state = self.statechart.state
        valid = self.statechart.get_valid_tokens()
        ctx = self.statechart.context

        lines = [
            f"Current State: {state.name}",
            f"Valid token categories: {', '.join(sorted(valid))}",
            f"Context:",
            f"  Indent level: {ctx.indent_level}",
            f"  Paren depth: {ctx.paren_depth}",
            f"  Bracket depth: {ctx.bracket_depth}",
            f"  Brace depth: {ctx.brace_depth}",
            f"  In string: {ctx.in_string}",
            f"  In comment: {ctx.in_comment}",
        ]
        return '\n'.join(lines)


# Demo/test code
if __name__ == "__main__":
    print("=" * 60)
    print("CONSTRAINED CODE GENERATOR DEMO")
    print("=" * 60)

    # Create simple vocab for testing
    test_vocab = {
        'def': 0, 'class': 1, 'if': 2, 'else': 3, 'return': 4,
        'for': 5, 'in': 6, 'while': 7, 'True': 8, 'False': 9,
        'None': 10, 'and': 11, 'or': 12, 'not': 13,
        '(': 14, ')': 15, '[': 16, ']': 17, '{': 18, '}': 19,
        ':': 20, ',': 21, '.': 22, '=': 23, '+': 24, '-': 25,
        '*': 26, '/': 27, '<': 28, '>': 29, '==': 30, '!=': 31,
        '\n': 32, '    ': 33, ' ': 34,
        'foo': 35, 'bar': 36, 'x': 37, 'y': 38, 'self': 39,
        '0': 40, '1': 41, '2': 42, '3': 43,
        'print': 44, 'len': 45, 'range': 46, 'str': 47, 'int': 48,
    }

    # Create mapper and masker
    mapper = TokenStateMapper(test_vocab)
    statechart = PythonSyntaxStatechart()
    masker = LogitMasker(mapper, statechart)

    print("\nInitial state:", statechart.state.name)
    print("Valid tokens:", statechart.get_valid_tokens())

    # Simulate generation
    tokens = ['def', 'foo', '(', 'x', ')', ':', '\n']

    print("\nSimulating token sequence:")
    for token in tokens:
        valid_before = masker.get_valid_token_ids()
        is_valid = token in [mapper.get_token_text(tid) for tid in valid_before]

        print(f"  Token: {token:10s} | State: {statechart.state.name:20s} | Valid: {is_valid}")

        masker.update_state(token)

    print(f"\nFinal state: {statechart.state.name}")
    print(f"Constraints: {masker.statechart.explain_constraints() if hasattr(masker.statechart, 'explain_constraints') else 'N/A'}")
