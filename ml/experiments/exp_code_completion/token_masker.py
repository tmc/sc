"""
Token Masker for Statechart-Guided Code Generation

Maps syntax states to token validity masks for LLM logit masking.
Supports both Python and Go syntax constraints.

Key insight: Each syntax state defines a SET of valid next tokens.
We convert this to a binary mask over the LLM vocabulary.

Usage:
    masker = TokenMasker(vocab, language="python")
    valid_mask = masker.get_mask(current_state, context)
    masked_logits = logits + (1 - valid_mask) * -1e9
"""

import mlx.core as mx
import mlx.nn as nn
from dataclasses import dataclass, field
from typing import List, Set, Dict, Optional, Tuple, FrozenSet
from enum import Enum, auto
import re
from collections import defaultdict


class TokenCategory(Enum):
    """Token categories for mask computation."""
    # Literals
    IDENTIFIER = auto()
    INT_LITERAL = auto()
    FLOAT_LITERAL = auto()
    STRING_LITERAL = auto()
    BOOL_LITERAL = auto()
    NIL_LITERAL = auto()
    
    # Keywords - Python
    KW_DEF = auto()
    KW_CLASS = auto()
    KW_IF = auto()
    KW_ELIF = auto()
    KW_ELSE = auto()
    KW_FOR = auto()
    KW_WHILE = auto()
    KW_RETURN = auto()
    KW_IMPORT = auto()
    KW_FROM = auto()
    KW_AS = auto()
    KW_TRY = auto()
    KW_EXCEPT = auto()
    KW_FINALLY = auto()
    KW_WITH = auto()
    KW_ASYNC = auto()
    KW_AWAIT = auto()
    KW_YIELD = auto()
    KW_LAMBDA = auto()
    KW_PASS = auto()
    KW_BREAK = auto()
    KW_CONTINUE = auto()
    KW_RAISE = auto()
    KW_ASSERT = auto()
    KW_IN = auto()
    KW_IS = auto()
    KW_AND = auto()
    KW_OR = auto()
    KW_NOT = auto()
    KW_TRUE = auto()
    KW_FALSE = auto()
    KW_NONE = auto()
    
    # Keywords - Go
    KW_FUNC = auto()
    KW_PACKAGE = auto()
    KW_TYPE = auto()
    KW_STRUCT = auto()
    KW_INTERFACE = auto()
    KW_MAP = auto()
    KW_CHAN = auto()
    KW_GO = auto()
    KW_SELECT = auto()
    KW_CASE = auto()
    KW_DEFAULT = auto()
    KW_SWITCH = auto()
    KW_FALLTHROUGH = auto()
    KW_DEFER = auto()
    KW_VAR = auto()
    KW_CONST = auto()
    KW_RANGE = auto()
    
    # Operators
    OP_PLUS = auto()
    OP_MINUS = auto()
    OP_STAR = auto()
    OP_SLASH = auto()
    OP_PERCENT = auto()
    OP_AMPERSAND = auto()
    OP_PIPE = auto()
    OP_CARET = auto()
    OP_TILDE = auto()
    OP_LT = auto()
    OP_GT = auto()
    OP_LE = auto()
    OP_GE = auto()
    OP_EQ = auto()
    OP_NE = auto()
    OP_ASSIGN = auto()
    OP_COLON_ASSIGN = auto()
    OP_PLUS_ASSIGN = auto()
    OP_MINUS_ASSIGN = auto()
    OP_ARROW = auto()
    OP_CHAN_ARROW = auto()
    
    # Delimiters
    LPAREN = auto()
    RPAREN = auto()
    LBRACKET = auto()
    RBRACKET = auto()
    LBRACE = auto()
    RBRACE = auto()
    COMMA = auto()
    COLON = auto()
    SEMICOLON = auto()
    DOT = auto()
    ELLIPSIS = auto()
    
    # Whitespace
    NEWLINE = auto()
    INDENT = auto()
    DEDENT = auto()
    WHITESPACE = auto()
    
    # Special
    EOF = auto()
    COMMENT = auto()
    
    # Meta
    ANY = auto()


# Token text to category mapping
TOKEN_TO_CATEGORY: Dict[str, TokenCategory] = {
    # Python keywords
    'def': TokenCategory.KW_DEF,
    'class': TokenCategory.KW_CLASS,
    'if': TokenCategory.KW_IF,
    'elif': TokenCategory.KW_ELIF,
    'else': TokenCategory.KW_ELSE,
    'for': TokenCategory.KW_FOR,
    'while': TokenCategory.KW_WHILE,
    'return': TokenCategory.KW_RETURN,
    'import': TokenCategory.KW_IMPORT,
    'from': TokenCategory.KW_FROM,
    'as': TokenCategory.KW_AS,
    'try': TokenCategory.KW_TRY,
    'except': TokenCategory.KW_EXCEPT,
    'finally': TokenCategory.KW_FINALLY,
    'with': TokenCategory.KW_WITH,
    'async': TokenCategory.KW_ASYNC,
    'await': TokenCategory.KW_AWAIT,
    'yield': TokenCategory.KW_YIELD,
    'lambda': TokenCategory.KW_LAMBDA,
    'pass': TokenCategory.KW_PASS,
    'break': TokenCategory.KW_BREAK,
    'continue': TokenCategory.KW_CONTINUE,
    'raise': TokenCategory.KW_RAISE,
    'assert': TokenCategory.KW_ASSERT,
    'in': TokenCategory.KW_IN,
    'is': TokenCategory.KW_IS,
    'and': TokenCategory.KW_AND,
    'or': TokenCategory.KW_OR,
    'not': TokenCategory.KW_NOT,
    'True': TokenCategory.KW_TRUE,
    'False': TokenCategory.KW_FALSE,
    'None': TokenCategory.KW_NONE,
    
    # Go keywords
    'func': TokenCategory.KW_FUNC,
    'package': TokenCategory.KW_PACKAGE,
    'type': TokenCategory.KW_TYPE,
    'struct': TokenCategory.KW_STRUCT,
    'interface': TokenCategory.KW_INTERFACE,
    'map': TokenCategory.KW_MAP,
    'chan': TokenCategory.KW_CHAN,
    'go': TokenCategory.KW_GO,
    'select': TokenCategory.KW_SELECT,
    'case': TokenCategory.KW_CASE,
    'default': TokenCategory.KW_DEFAULT,
    'switch': TokenCategory.KW_SWITCH,
    'fallthrough': TokenCategory.KW_FALLTHROUGH,
    'defer': TokenCategory.KW_DEFER,
    'var': TokenCategory.KW_VAR,
    'const': TokenCategory.KW_CONST,
    'range': TokenCategory.KW_RANGE,
    
    # Operators
    '+': TokenCategory.OP_PLUS,
    '-': TokenCategory.OP_MINUS,
    '*': TokenCategory.OP_STAR,
    '/': TokenCategory.OP_SLASH,
    '%': TokenCategory.OP_PERCENT,
    '&': TokenCategory.OP_AMPERSAND,
    '|': TokenCategory.OP_PIPE,
    '^': TokenCategory.OP_CARET,
    '~': TokenCategory.OP_TILDE,
    '<': TokenCategory.OP_LT,
    '>': TokenCategory.OP_GT,
    '<=': TokenCategory.OP_LE,
    '>=': TokenCategory.OP_GE,
    '==': TokenCategory.OP_EQ,
    '!=': TokenCategory.OP_NE,
    '=': TokenCategory.OP_ASSIGN,
    ':=': TokenCategory.OP_COLON_ASSIGN,
    '+=': TokenCategory.OP_PLUS_ASSIGN,
    '-=': TokenCategory.OP_MINUS_ASSIGN,
    '->': TokenCategory.OP_ARROW,
    '<-': TokenCategory.OP_CHAN_ARROW,
    
    # Delimiters
    '(': TokenCategory.LPAREN,
    ')': TokenCategory.RPAREN,
    '[': TokenCategory.LBRACKET,
    ']': TokenCategory.RBRACKET,
    '{': TokenCategory.LBRACE,
    '}': TokenCategory.RBRACE,
    ',': TokenCategory.COMMA,
    ':': TokenCategory.COLON,
    ';': TokenCategory.SEMICOLON,
    '.': TokenCategory.DOT,
    '...': TokenCategory.ELLIPSIS,
    
    # Whitespace
    '\n': TokenCategory.NEWLINE,
}


@dataclass
class MaskContext:
    """Context for mask computation."""
    paren_depth: int = 0
    bracket_depth: int = 0
    brace_depth: int = 0
    indent_level: int = 0
    in_string: bool = False
    in_comment: bool = False
    line_start: bool = True
    last_token: str = ""


class TokenMasker:
    """
    Computes token validity masks for statechart-guided generation.
    
    Precomputes category masks and combines them based on current state.
    """
    
    def __init__(self, vocab: Dict[str, int], language: str = "python"):
        """
        Initialize masker with LLM vocabulary.
        
        Args:
            vocab: Dict mapping token strings to IDs
            language: "python" or "go"
        """
        self.vocab = vocab
        self.vocab_size = max(vocab.values()) + 1 if vocab else 0  # Max ID + 1, not count
        self.language = language
        self.id_to_token = {v: k for k, v in vocab.items()}
        
        # Categorize all tokens
        self.token_categories: Dict[int, Set[TokenCategory]] = {}
        self._categorize_vocab()
        
        # Precompute category masks
        self.category_masks: Dict[TokenCategory, mx.array] = {}
        self._build_category_masks()
        
        # State -> valid categories mapping
        self.state_categories = self._build_state_categories()
    
    def _categorize_vocab(self):
        """Categorize each token in vocabulary."""
        for token, token_id in self.vocab.items():
            categories = set()
            
            # Check direct mapping
            if token in TOKEN_TO_CATEGORY:
                categories.add(TOKEN_TO_CATEGORY[token])
            
            # Check patterns
            if re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', token):
                if token not in TOKEN_TO_CATEGORY:
                    categories.add(TokenCategory.IDENTIFIER)
            
            if re.match(r'^[0-9]+$', token):
                categories.add(TokenCategory.INT_LITERAL)
            
            if re.match(r'^[0-9]+\.[0-9]*$', token):
                categories.add(TokenCategory.FLOAT_LITERAL)
            
            if token.startswith('"') or token.startswith("'"):
                categories.add(TokenCategory.STRING_LITERAL)
            
            if token.startswith('#') or token.startswith('//'):
                categories.add(TokenCategory.COMMENT)
            
            if token.strip() == '' and '\n' not in token:
                categories.add(TokenCategory.WHITESPACE)
            
            self.token_categories[token_id] = categories
    
    def _build_category_masks(self):
        """Precompute binary masks for each category."""
        for category in TokenCategory:
            mask_list = [0.0] * self.vocab_size
            for token_id, categories in self.token_categories.items():
                if category in categories or category == TokenCategory.ANY:
                    mask_list[token_id] = 1.0
            self.category_masks[category] = mx.array(mask_list)
    
    def _build_state_categories(self) -> Dict[str, Set[TokenCategory]]:
        """Build state -> valid categories mapping."""
        if self.language == "python":
            return self._python_state_categories()
        else:
            return self._go_state_categories()
    
    def _python_state_categories(self) -> Dict[str, Set[TokenCategory]]:
        """Python syntax state categories."""
        return {
            "MODULE_START": {
                TokenCategory.KW_DEF, TokenCategory.KW_CLASS,
                TokenCategory.KW_IF, TokenCategory.KW_FOR, TokenCategory.KW_WHILE,
                TokenCategory.KW_TRY, TokenCategory.KW_WITH,
                TokenCategory.KW_IMPORT, TokenCategory.KW_FROM,
                TokenCategory.KW_ASYNC, TokenCategory.IDENTIFIER,
                TokenCategory.NEWLINE, TokenCategory.COMMENT,
            },
            "FUNC_KEYWORD": {
                TokenCategory.IDENTIFIER,
            },
            "FUNC_NAME": {
                TokenCategory.LPAREN,
            },
            "FUNC_PARAMS": {
                TokenCategory.IDENTIFIER, TokenCategory.COMMA, TokenCategory.COLON,
                TokenCategory.OP_ASSIGN, TokenCategory.OP_STAR, TokenCategory.RPAREN,
                TokenCategory.KW_NONE, TokenCategory.INT_LITERAL, TokenCategory.STRING_LITERAL,
            },
            "FUNC_BODY_START": {
                TokenCategory.NEWLINE,
            },
            "FUNC_BODY": {
                TokenCategory.KW_RETURN, TokenCategory.KW_YIELD,
                TokenCategory.KW_IF, TokenCategory.KW_FOR, TokenCategory.KW_WHILE,
                TokenCategory.KW_TRY, TokenCategory.KW_WITH, TokenCategory.KW_RAISE,
                TokenCategory.KW_ASSERT, TokenCategory.KW_PASS,
                TokenCategory.KW_BREAK, TokenCategory.KW_CONTINUE,
                TokenCategory.IDENTIFIER, TokenCategory.INDENT, TokenCategory.DEDENT,
                TokenCategory.NEWLINE,
            },
            "IF_KEYWORD": {
                TokenCategory.IDENTIFIER, TokenCategory.KW_NOT, TokenCategory.LPAREN,
                TokenCategory.INT_LITERAL, TokenCategory.KW_TRUE, TokenCategory.KW_FALSE,
            },
            "IF_CONDITION": {
                TokenCategory.IDENTIFIER, TokenCategory.INT_LITERAL,
                TokenCategory.OP_EQ, TokenCategory.OP_NE, TokenCategory.OP_LT, TokenCategory.OP_GT,
                TokenCategory.OP_LE, TokenCategory.OP_GE,
                TokenCategory.KW_AND, TokenCategory.KW_OR, TokenCategory.KW_NOT,
                TokenCategory.KW_IN, TokenCategory.KW_IS,
                TokenCategory.LPAREN, TokenCategory.RPAREN,
                TokenCategory.COLON,
            },
            "FOR_KEYWORD": {
                TokenCategory.IDENTIFIER,
            },
            "FOR_TARGET": {
                TokenCategory.IDENTIFIER, TokenCategory.COMMA, TokenCategory.KW_IN,
            },
            "FOR_ITER": {
                TokenCategory.IDENTIFIER, TokenCategory.LPAREN, TokenCategory.RPAREN,
                TokenCategory.COLON,
            },
            "RETURN_STMT": {
                TokenCategory.IDENTIFIER, TokenCategory.INT_LITERAL,
                TokenCategory.STRING_LITERAL, TokenCategory.KW_NONE,
                TokenCategory.KW_TRUE, TokenCategory.KW_FALSE,
                TokenCategory.LPAREN, TokenCategory.LBRACKET, TokenCategory.LBRACE,
                TokenCategory.NEWLINE,
            },
            "EXPR_START": {
                TokenCategory.IDENTIFIER, TokenCategory.INT_LITERAL,
                TokenCategory.FLOAT_LITERAL, TokenCategory.STRING_LITERAL,
                TokenCategory.KW_TRUE, TokenCategory.KW_FALSE, TokenCategory.KW_NONE,
                TokenCategory.LPAREN, TokenCategory.LBRACKET, TokenCategory.LBRACE,
                TokenCategory.OP_MINUS, TokenCategory.KW_NOT,
            },
            "EXPR_BINARY": {
                TokenCategory.OP_PLUS, TokenCategory.OP_MINUS,
                TokenCategory.OP_STAR, TokenCategory.OP_SLASH, TokenCategory.OP_PERCENT,
                TokenCategory.OP_EQ, TokenCategory.OP_NE,
                TokenCategory.OP_LT, TokenCategory.OP_GT, TokenCategory.OP_LE, TokenCategory.OP_GE,
                TokenCategory.KW_AND, TokenCategory.KW_OR,
                TokenCategory.LPAREN,  # Function call
                TokenCategory.LBRACKET,  # Index
                TokenCategory.DOT,  # Attribute
                TokenCategory.OP_ASSIGN,  # Assignment
                TokenCategory.NEWLINE, TokenCategory.COMMA,
                TokenCategory.RPAREN, TokenCategory.RBRACKET, TokenCategory.RBRACE,
            },
            "CALL_ARGS": {
                TokenCategory.IDENTIFIER, TokenCategory.INT_LITERAL,
                TokenCategory.STRING_LITERAL,
                TokenCategory.KW_TRUE, TokenCategory.KW_FALSE, TokenCategory.KW_NONE,
                TokenCategory.COMMA, TokenCategory.OP_ASSIGN,
                TokenCategory.OP_STAR, TokenCategory.RPAREN,
                TokenCategory.LPAREN, TokenCategory.LBRACKET, TokenCategory.LBRACE,
            },
            "IN_PARENS": {
                TokenCategory.IDENTIFIER, TokenCategory.INT_LITERAL,
                TokenCategory.STRING_LITERAL, TokenCategory.COMMA,
                TokenCategory.OP_PLUS, TokenCategory.OP_MINUS,
                TokenCategory.LPAREN, TokenCategory.RPAREN,
            },
            "IN_BRACKETS": {
                TokenCategory.IDENTIFIER, TokenCategory.INT_LITERAL,
                TokenCategory.STRING_LITERAL, TokenCategory.COMMA, TokenCategory.COLON,
                TokenCategory.RBRACKET,
            },
            "IN_BRACES": {
                TokenCategory.IDENTIFIER, TokenCategory.STRING_LITERAL,
                TokenCategory.COLON, TokenCategory.COMMA,
                TokenCategory.RBRACE,
            },
            "STATEMENT_START": {
                TokenCategory.KW_DEF, TokenCategory.KW_CLASS,
                TokenCategory.KW_IF, TokenCategory.KW_ELIF, TokenCategory.KW_ELSE,
                TokenCategory.KW_FOR, TokenCategory.KW_WHILE,
                TokenCategory.KW_TRY, TokenCategory.KW_EXCEPT, TokenCategory.KW_FINALLY,
                TokenCategory.KW_WITH, TokenCategory.KW_RETURN, TokenCategory.KW_YIELD,
                TokenCategory.KW_RAISE, TokenCategory.KW_ASSERT,
                TokenCategory.KW_PASS, TokenCategory.KW_BREAK, TokenCategory.KW_CONTINUE,
                TokenCategory.KW_IMPORT, TokenCategory.KW_FROM,
                TokenCategory.KW_ASYNC, TokenCategory.KW_AWAIT,
                TokenCategory.IDENTIFIER, TokenCategory.INDENT, TokenCategory.DEDENT,
                TokenCategory.NEWLINE,
            },
        }
    
    def _go_state_categories(self) -> Dict[str, Set[TokenCategory]]:
        """Go syntax state categories."""
        return {
            "MODULE_START": {
                TokenCategory.KW_PACKAGE,
            },
            "PACKAGE_DECL": {
                TokenCategory.IDENTIFIER,
            },
            "TOP_LEVEL": {
                TokenCategory.KW_IMPORT, TokenCategory.KW_FUNC, TokenCategory.KW_TYPE,
                TokenCategory.KW_VAR, TokenCategory.KW_CONST,
                TokenCategory.NEWLINE, TokenCategory.COMMENT,
            },
            "IMPORT_DECL": {
                TokenCategory.LPAREN, TokenCategory.STRING_LITERAL,
            },
            "IMPORT_GROUP": {
                TokenCategory.STRING_LITERAL, TokenCategory.RPAREN, TokenCategory.NEWLINE,
            },
            "FUNC_KEYWORD": {
                TokenCategory.IDENTIFIER, TokenCategory.LPAREN,  # Method receiver
            },
            "FUNC_NAME": {
                TokenCategory.LPAREN,
            },
            "FUNC_PARAMS": {
                TokenCategory.IDENTIFIER, TokenCategory.COMMA,
                TokenCategory.OP_STAR, TokenCategory.RPAREN,
            },
            "FUNC_RETURN": {
                TokenCategory.IDENTIFIER, TokenCategory.LPAREN, TokenCategory.LBRACE,
            },
            "FUNC_BODY": {
                TokenCategory.KW_VAR, TokenCategory.KW_CONST,
                TokenCategory.KW_IF, TokenCategory.KW_FOR, TokenCategory.KW_SWITCH,
                TokenCategory.KW_SELECT, TokenCategory.KW_RETURN,
                TokenCategory.KW_GO, TokenCategory.KW_DEFER,
                TokenCategory.IDENTIFIER, TokenCategory.RBRACE,
                TokenCategory.NEWLINE,
            },
            "IF_KEYWORD": {
                TokenCategory.IDENTIFIER, TokenCategory.KW_NOT,
            },
            "IF_CONDITION": {
                TokenCategory.IDENTIFIER, TokenCategory.INT_LITERAL,
                TokenCategory.OP_EQ, TokenCategory.OP_NE, TokenCategory.OP_LT, TokenCategory.OP_GT,
                TokenCategory.LBRACE,
            },
            "FOR_KEYWORD": {
                TokenCategory.IDENTIFIER, TokenCategory.LBRACE,  # for {}
            },
            "FOR_RANGE": {
                TokenCategory.IDENTIFIER, TokenCategory.KW_RANGE, TokenCategory.LBRACE,
            },
            "SWITCH_KEYWORD": {
                TokenCategory.IDENTIFIER, TokenCategory.LBRACE,
            },
            "CASE_CLAUSE": {
                TokenCategory.KW_CASE, TokenCategory.KW_DEFAULT, TokenCategory.RBRACE,
            },
            "RETURN_STMT": {
                TokenCategory.IDENTIFIER, TokenCategory.INT_LITERAL,
                TokenCategory.STRING_LITERAL, TokenCategory.KW_NONE,
                TokenCategory.NEWLINE,
            },
            "TYPE_DEF": {
                TokenCategory.IDENTIFIER,
            },
            "STRUCT_DEF": {
                TokenCategory.KW_STRUCT, TokenCategory.KW_INTERFACE,
            },
            "STRUCT_BODY": {
                TokenCategory.IDENTIFIER, TokenCategory.OP_STAR,
                TokenCategory.STRING_LITERAL,  # Tags
                TokenCategory.RBRACE, TokenCategory.NEWLINE,
            },
            "EXPR_START": {
                TokenCategory.IDENTIFIER, TokenCategory.INT_LITERAL,
                TokenCategory.FLOAT_LITERAL, TokenCategory.STRING_LITERAL,
                TokenCategory.LPAREN, TokenCategory.LBRACKET, TokenCategory.LBRACE,
                TokenCategory.OP_AMPERSAND, TokenCategory.OP_STAR,
            },
            "EXPR_BINARY": {
                TokenCategory.OP_PLUS, TokenCategory.OP_MINUS,
                TokenCategory.OP_STAR, TokenCategory.OP_SLASH, TokenCategory.OP_PERCENT,
                TokenCategory.OP_EQ, TokenCategory.OP_NE,
                TokenCategory.OP_LT, TokenCategory.OP_GT, TokenCategory.OP_LE, TokenCategory.OP_GE,
                TokenCategory.OP_ASSIGN, TokenCategory.OP_COLON_ASSIGN,
                TokenCategory.LPAREN, TokenCategory.LBRACKET, TokenCategory.DOT,
                TokenCategory.NEWLINE, TokenCategory.SEMICOLON,
            },
        }
    
    def get_mask(self, state: str, context: Optional[MaskContext] = None) -> mx.array:
        """
        Get token validity mask for current state.
        
        Args:
            state: Current syntax state name
            context: Optional context for additional constraints
            
        Returns:
            Binary mask of shape (vocab_size,) where 1 = valid
        """
        # Get categories valid in this state
        categories = self.state_categories.get(state, {TokenCategory.ANY})
        
        # Combine category masks
        mask = mx.zeros((self.vocab_size,))
        for cat in categories:
            if cat in self.category_masks:
                mask = mx.maximum(mask, self.category_masks[cat])
        
        # Apply context-based constraints
        if context:
            mask = self._apply_context_constraints(mask, context)
        
        return mask
    
    def _apply_context_constraints(self, mask: mx.array, ctx: MaskContext) -> mx.array:
        """Apply context-based mask adjustments."""
        # If in string, only allow string content
        if ctx.in_string:
            mask = self.category_masks[TokenCategory.STRING_LITERAL].copy()
        
        # If in comment, only allow comment content
        if ctx.in_comment:
            mask = self.category_masks[TokenCategory.COMMENT].copy()
            mask = mx.maximum(mask, self.category_masks[TokenCategory.NEWLINE])
        
        # Always allow closing brackets if open
        if ctx.paren_depth > 0:
            mask = mx.maximum(mask, self.category_masks[TokenCategory.RPAREN])
        if ctx.bracket_depth > 0:
            mask = mx.maximum(mask, self.category_masks[TokenCategory.RBRACKET])
        if ctx.brace_depth > 0:
            mask = mx.maximum(mask, self.category_masks[TokenCategory.RBRACE])
        
        return mask
    
    def mask_logits(self, logits: mx.array, state: str, 
                    context: Optional[MaskContext] = None,
                    mask_value: float = -1e9) -> mx.array:
        """
        Apply mask to logits, setting invalid tokens to -inf.
        
        Args:
            logits: Raw logits from LLM (vocab_size,) or (batch, vocab_size)
            state: Current syntax state
            context: Optional context
            mask_value: Value to add for invalid tokens (default -1e9)
            
        Returns:
            Masked logits
        """
        mask = self.get_mask(state, context)
        
        # Reshape mask for broadcasting if needed
        if logits.ndim == 2:
            mask = mask.reshape(1, -1)
        
        # Apply mask: valid tokens keep logits, invalid get mask_value
        return logits + (1.0 - mask) * mask_value
    
    def get_valid_tokens(self, state: str, context: Optional[MaskContext] = None) -> List[str]:
        """Get list of valid token strings for debugging."""
        mask = self.get_mask(state, context)
        mask_np = mask.tolist()
        return [self.id_to_token[i] for i, v in enumerate(mask_np) if v > 0.5]


class SAEGuidedMasker(TokenMasker):
    """
    Token masker guided by SAE-discovered syntax features.
    
    Uses SAE feature activations to infer syntax state,
    then applies appropriate token masks.
    """
    
    def __init__(self, vocab: Dict[str, int], sae_module, language: str = "python"):
        super().__init__(vocab, language)
        self.sae = sae_module
        
        # Feature -> state mapping (learned from analysis)
        self.feature_to_state: Dict[int, str] = {}
    
    def learn_feature_mapping(self, hidden_states: List[mx.array], 
                              states: List[str]):
        """
        Learn which SAE features correspond to which syntax states.
        
        Args:
            hidden_states: List of LM hidden states
            states: Corresponding syntax state labels
        """
        feature_state_counts: Dict[int, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        
        for hidden, state in zip(hidden_states, states):
            # Get SAE activations
            _, acts, indices = self.sae.forward(hidden.reshape(1, -1))
            
            for idx in indices[0].tolist():
                feature_state_counts[idx][state] += 1
        
        # Map features to most common state
        for feature_idx, state_counts in feature_state_counts.items():
            if state_counts:
                best_state = max(state_counts.keys(), key=lambda s: state_counts[s])
                total = sum(state_counts.values())
                if state_counts[best_state] / total > 0.6:  # 60% threshold
                    self.feature_to_state[feature_idx] = best_state
    
    def infer_state(self, hidden: mx.array) -> str:
        """Infer syntax state from hidden state via SAE."""
        _, acts, indices = self.sae.forward(hidden.reshape(1, -1))
        
        # Count votes from active features
        state_votes: Dict[str, float] = defaultdict(float)
        
        for i, idx in enumerate(indices[0].tolist()):
            if idx in self.feature_to_state:
                # Weight vote by activation strength
                weight = float(acts[0, idx])
                state_votes[self.feature_to_state[idx]] += weight
        
        if state_votes:
            return max(state_votes.keys(), key=lambda s: state_votes[s])
        else:
            return "STATEMENT_START"  # Default fallback
    
    def get_mask_from_hidden(self, hidden: mx.array, 
                              context: Optional[MaskContext] = None) -> mx.array:
        """Get mask directly from hidden state."""
        state = self.infer_state(hidden)
        return self.get_mask(state, context)


def demo():
    """Demonstrate token masking."""
    print("=" * 60)
    print("TOKEN MASKER DEMO")
    print("=" * 60)
    
    # Simple demo vocabulary
    vocab = {
        'def': 0, 'class': 1, 'if': 2, 'else': 3, 'for': 4, 'while': 5,
        'return': 6, 'import': 7, 'from': 8,
        '+': 10, '-': 11, '*': 12, '/': 13, '=': 14, '==': 15,
        '(': 20, ')': 21, '[': 22, ']': 23, '{': 24, '}': 25,
        ':': 26, ',': 27, '.': 28,
        '\n': 30, '    ': 31,
        'x': 40, 'y': 41, 'foo': 42, 'bar': 43, 'self': 44,
        '0': 50, '1': 51, '2': 52, '10': 53,
        '"hello"': 60, "'test'": 61,
        'True': 70, 'False': 71, 'None': 72,
        'in': 80, 'is': 81, 'and': 82, 'or': 83, 'not': 84,
    }
    
    print(f"\nVocabulary size: {len(vocab)}")
    
    # Create masker
    masker = TokenMasker(vocab, language="python")
    
    # Test different states
    test_states = [
        "MODULE_START",
        "FUNC_KEYWORD",
        "FUNC_PARAMS",
        "IF_CONDITION",
        "EXPR_BINARY",
    ]
    
    for state in test_states:
        valid = masker.get_valid_tokens(state)
        print(f"\n{state}:")
        print(f"  Valid tokens ({len(valid)}): {', '.join(valid[:10])}...")
    
    # Test masking
    print("\n--- Logit Masking Demo ---")
    logits = mx.random.normal((masker.vocab_size,)) * 2

    masked = masker.mask_logits(logits, "FUNC_KEYWORD")
    
    print(f"Before masking: max={float(mx.max(logits)):.2f}, min={float(mx.min(logits)):.2f}")
    print(f"After masking: max={float(mx.max(masked)):.2f}, min={float(mx.min(masked)):.2f}")
    
    # Show which tokens survived
    probs = mx.softmax(masked)
    top_idx = int(mx.argmax(probs))
    print(f"Top token after masking: '{masker.id_to_token[top_idx]}'")
    
    print("\n" + "=" * 60)
    print("Token masking enforces syntax validity!")
    print("=" * 60)


if __name__ == "__main__":
    demo()
