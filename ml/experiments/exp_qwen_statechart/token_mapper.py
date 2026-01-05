"""
Token Mapper

Maps between LLM vocabulary tokens and Starlark syntax categories.
This is the critical bridge between:
- Statechart's abstract categories (IDENTIFIER, NUMBER, etc.)
- LLM's actual vocabulary (specific tokens from Qwen/other models)

Supports:
- Token categorization (keyword, operator, identifier, etc.)
- Category to token ID mapping
- Efficient caching for fast lookups
"""

from dataclasses import dataclass, field
from typing import Dict, Set, Optional, List
import re

try:
    from .starlark_statechart import KEYWORDS, OPERATORS, DELIMITERS, SC_IDENTIFIERS
except ImportError:
    from starlark_statechart import KEYWORDS, OPERATORS, DELIMITERS, SC_IDENTIFIERS


# Token category patterns
TOKEN_PATTERNS = {
    'IDENTIFIER': re.compile(r'^[a-zA-Z_][a-zA-Z0-9_]*$'),
    'NUMBER': re.compile(r'^[0-9]+\.?[0-9]*([eE][+-]?[0-9]+)?$'),
    'STRING': re.compile(r'^["\']'),
    'NEWLINE': re.compile(r'^[\n]$'),
    'WHITESPACE': re.compile(r'^[ \t]+$'),
    'COMMENT': re.compile(r'^#'),
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

        # Empty or whitespace-only
        if not text or text.isspace():
            self.categories.add('WHITESPACE')
            return

        # Special tokens
        if text.startswith('<|') and text.endswith('|>'):
            self.categories.add('SPECIAL')
            return

        # Keywords
        if text in KEYWORDS:
            self.categories.add('KEYWORD')
            self.categories.add(text)  # Add specific keyword
        # Operators
        elif text in OPERATORS:
            self.categories.add('OPERATOR')
            self.categories.add(text)
        # Delimiters
        elif text in DELIMITERS:
            self.categories.add('DELIMITER')
            self.categories.add(text)
        # SC DSL identifiers
        elif text in SC_IDENTIFIERS:
            self.categories.add('IDENTIFIER')
            self.categories.add('SC_IDENTIFIER')
            self.categories.add(text)
        # Check patterns
        elif TOKEN_PATTERNS['IDENTIFIER'].match(text):
            self.categories.add('IDENTIFIER')
        elif TOKEN_PATTERNS['NUMBER'].match(text):
            self.categories.add('NUMBER')
        elif text.startswith('"') or text.startswith("'"):
            self.categories.add('STRING')
        elif text == '\n':
            self.categories.add('NEWLINE')
        elif text.startswith('#'):
            self.categories.add('COMMENT')
        elif text.strip() == '':
            self.categories.add('WHITESPACE')


class TokenMapper:
    """
    Maps between LLM vocabulary tokens and syntax state categories.

    This is the critical bridge between:
    - Statechart's abstract categories (IDENTIFIER, NUMBER, etc.)
    - LLM's actual vocabulary (specific tokens)
    """

    CATEGORIES = {
        'KEYWORD': KEYWORDS,
        'OPERATOR': OPERATORS,
        'DELIMITER': DELIMITERS,
        'SC_IDENTIFIER': SC_IDENTIFIERS,
    }

    def __init__(self, tokenizer=None, vocab: Optional[Dict[str, int]] = None):
        """
        Initialize mapper with tokenizer or vocabulary.

        Args:
            tokenizer: Tokenizer with get_vocab() method (e.g., HuggingFace)
            vocab: Dict mapping token strings to IDs
        """
        if vocab is None and tokenizer is not None:
            vocab = tokenizer.get_vocab()
        elif vocab is None:
            vocab = self._create_default_vocab()

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

        # Cache for frequent lookups
        self._cache: Dict[frozenset, Set[int]] = {}

    def _create_default_vocab(self) -> Dict[str, int]:
        """Create a minimal default vocabulary for testing."""
        vocab = {}
        token_id = 0

        # Keywords
        for kw in KEYWORDS:
            vocab[kw] = token_id
            token_id += 1

        # Operators
        for op in OPERATORS:
            vocab[op] = token_id
            token_id += 1

        # Delimiters
        for delim in DELIMITERS:
            vocab[delim] = token_id
            token_id += 1

        # SC identifiers
        for ident in SC_IDENTIFIERS:
            if ident not in vocab:
                vocab[ident] = token_id
                token_id += 1

        # Common tokens
        for token in ['NEWLINE', '\n', ' ', '    ', 'True', 'False', 'None']:
            if token not in vocab:
                vocab[token] = token_id
                token_id += 1

        # Numbers
        for n in range(10):
            vocab[str(n)] = token_id
            token_id += 1

        # Sample identifiers
        for ident in ['x', 'y', 'z', 'foo', 'bar', 'result', 'value', 'name']:
            vocab[ident] = token_id
            token_id += 1

        return vocab

    def get_token_ids_for_categories(self, categories: Set[str]) -> Set[int]:
        """Get all token IDs that match any of the given categories."""
        # Check cache
        cache_key = frozenset(categories)
        if cache_key in self._cache:
            return self._cache[cache_key]

        result = set()
        for cat in categories:
            if cat in self.category_to_ids:
                result.update(self.category_to_ids[cat])

        # Cache result
        self._cache[cache_key] = result
        return result

    def categories_to_ids(self, categories: Set[str]) -> Set[int]:
        """Alias for get_token_ids_for_categories."""
        return self.get_token_ids_for_categories(categories)

    def get_token_text(self, token_id: int) -> str:
        """Get text for a token ID."""
        return self.id_to_token.get(token_id, '<UNK>')

    def get_token_id(self, text: str) -> Optional[int]:
        """Get token ID for text."""
        return self.vocab.get(text)

    def get_token_categories(self, token_id: int) -> Set[str]:
        """Get categories for a token ID."""
        info = self.token_info.get(token_id)
        return info.categories if info else set()

    def is_identifier(self, token_id: int) -> bool:
        """Check if token is an identifier."""
        return 'IDENTIFIER' in self.get_token_categories(token_id)

    def is_keyword(self, token_id: int) -> bool:
        """Check if token is a keyword."""
        return 'KEYWORD' in self.get_token_categories(token_id)

    def is_operator(self, token_id: int) -> bool:
        """Check if token is an operator."""
        return 'OPERATOR' in self.get_token_categories(token_id)

    def is_delimiter(self, token_id: int) -> bool:
        """Check if token is a delimiter."""
        return 'DELIMITER' in self.get_token_categories(token_id)

    def get_stats(self) -> Dict[str, int]:
        """Get vocabulary statistics."""
        stats = {
            'total_tokens': self.vocab_size,
            'categories': len(self.category_to_ids),
        }
        for cat, ids in self.category_to_ids.items():
            stats[f'cat_{cat}'] = len(ids)
        return stats

    def clear_cache(self):
        """Clear the category lookup cache."""
        self._cache.clear()


class QwenTokenMapper(TokenMapper):
    """
    Specialized token mapper for Qwen Coder models.

    Handles Qwen-specific tokenization patterns and special tokens.
    """

    QWEN_SPECIAL_TOKENS = {
        '<|endoftext|>',
        '<|im_start|>',
        '<|im_end|>',
        '<|fim_prefix|>',
        '<|fim_middle|>',
        '<|fim_suffix|>',
        '<|fim_pad|>',
        '<|repo_name|>',
        '<|file_sep|>',
    }

    def __init__(self, tokenizer=None, vocab: Optional[Dict[str, int]] = None):
        super().__init__(tokenizer, vocab)

        # Mark Qwen special tokens
        for token in self.QWEN_SPECIAL_TOKENS:
            if token in self.vocab:
                token_id = self.vocab[token]
                if token_id in self.token_info:
                    self.token_info[token_id].categories.add('QWEN_SPECIAL')

    def get_end_token_id(self) -> Optional[int]:
        """Get the end-of-text token ID."""
        return self.vocab.get('<|endoftext|>')

    def get_fim_tokens(self) -> Dict[str, Optional[int]]:
        """Get fill-in-the-middle token IDs."""
        return {
            'prefix': self.vocab.get('<|fim_prefix|>'),
            'middle': self.vocab.get('<|fim_middle|>'),
            'suffix': self.vocab.get('<|fim_suffix|>'),
        }


# Demo
if __name__ == "__main__":
    print("=" * 60)
    print("TOKEN MAPPER DEMO")
    print("=" * 60)

    mapper = TokenMapper()

    print(f"\nVocab size: {mapper.vocab_size}")
    print(f"\nCategories: {list(mapper.category_to_ids.keys())}")

    print("\nStats:")
    for key, value in mapper.get_stats().items():
        print(f"  {key}: {value}")

    print("\nSample category lookups:")
    for cat in ['KEYWORD', 'OPERATOR', 'IDENTIFIER']:
        ids = mapper.categories_to_ids({cat})
        tokens = [mapper.get_token_text(tid) for tid in list(ids)[:5]]
        print(f"  {cat}: {tokens}...")

    # Test token categorization
    print("\nToken categorization:")
    for token in ['def', '+', 'foo', '42', '"hello"', '(', 'sc']:
        if token in mapper.vocab:
            tid = mapper.vocab[token]
            cats = mapper.get_token_categories(tid)
            print(f"  '{token}' -> {cats}")
