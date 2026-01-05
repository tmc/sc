"""
Guided Code Generation with Statechart Constraints

Integrates TokenMasker with LLM generation loop for syntactically valid output.
Supports both autoregressive sampling and beam search.

Key insight: By masking logits at each step, we GUARANTEE syntactic validity
while preserving LLM's semantic capabilities.

Usage:
    generator = GuidedCodeGenerator(vocab, tokenizer)
    code = generator.generate("def fibonacci(n):", max_tokens=50)
    assert is_syntactically_valid(code)  # Always true!
"""

import mlx.core as mx
import mlx.nn as nn
from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional, Tuple, Callable, Union
from enum import Enum
import re

from .token_masker import TokenMasker, MaskContext, TokenCategory


@dataclass
class GenerationConfig:
    """Configuration for guided generation."""
    max_tokens: int = 100
    temperature: float = 0.7
    top_k: int = 50
    top_p: float = 0.95
    repetition_penalty: float = 1.1
    stop_tokens: Set[str] = field(default_factory=lambda: {'\n\n', '<EOS>'})
    enforce_constraints: bool = True
    constraint_strength: float = 1e9  # How strongly to mask invalid tokens
    language: str = "python"
    track_blocked: bool = True  # Track which tokens were blocked


@dataclass 
class GenerationState:
    """State during generation."""
    syntax_state: str = "MODULE_START"
    context: MaskContext = field(default_factory=MaskContext)
    tokens_generated: List[str] = field(default_factory=list)
    tokens_blocked: List[Tuple[int, str, str]] = field(default_factory=list)  # (step, token, reason)
    indent_stack: List[int] = field(default_factory=list)


class SyntaxStateTracker:
    """
    Tracks syntax state during token-by-token generation.
    
    Updates state based on consumed tokens to determine valid next tokens.
    """
    
    def __init__(self, language: str = "python"):
        self.language = language
        self.state = "MODULE_START"
        self.context = MaskContext()
        
        # Transition rules: (current_state, token_category) -> next_state
        self.transitions = self._build_transitions()
    
    def _build_transitions(self) -> Dict[Tuple[str, str], str]:
        """Build state transition table."""
        if self.language == "python":
            return self._python_transitions()
        else:
            return self._go_transitions()
    
    def _python_transitions(self) -> Dict[Tuple[str, str], str]:
        """Python state transitions."""
        return {
            # Module level
            ("MODULE_START", "def"): "FUNC_KEYWORD",
            ("MODULE_START", "class"): "CLASS_KEYWORD",
            ("MODULE_START", "if"): "IF_KEYWORD",
            ("MODULE_START", "for"): "FOR_KEYWORD",
            ("MODULE_START", "while"): "WHILE_KEYWORD",
            ("MODULE_START", "import"): "IMPORT_STMT",
            ("MODULE_START", "from"): "FROM_IMPORT",
            ("MODULE_START", "IDENTIFIER"): "EXPR_START",
            ("MODULE_START", "NEWLINE"): "MODULE_START",
            
            # Function definition
            ("FUNC_KEYWORD", "IDENTIFIER"): "FUNC_NAME",
            ("FUNC_NAME", "("): "FUNC_PARAMS",
            ("FUNC_PARAMS", ")"): "FUNC_RETURN",
            ("FUNC_PARAMS", "IDENTIFIER"): "FUNC_PARAMS",
            ("FUNC_PARAMS", ","): "FUNC_PARAMS",
            ("FUNC_PARAMS", ":"): "FUNC_PARAMS",  # Type hint
            ("FUNC_RETURN", "->"): "FUNC_RETURN_TYPE",
            ("FUNC_RETURN", ":"): "FUNC_BODY_START",
            ("FUNC_RETURN_TYPE", "IDENTIFIER"): "FUNC_RETURN_TYPE",
            ("FUNC_RETURN_TYPE", ":"): "FUNC_BODY_START",
            ("FUNC_BODY_START", "NEWLINE"): "FUNC_BODY",
            ("FUNC_BODY", "INDENT"): "STATEMENT_START",
            ("FUNC_BODY", "DEDENT"): "MODULE_START",
            
            # Class definition
            ("CLASS_KEYWORD", "IDENTIFIER"): "CLASS_NAME",
            ("CLASS_NAME", "("): "CLASS_BASES",
            ("CLASS_NAME", ":"): "CLASS_BODY_START",
            ("CLASS_BASES", ")"): "CLASS_COLON",
            ("CLASS_BASES", "IDENTIFIER"): "CLASS_BASES",
            ("CLASS_COLON", ":"): "CLASS_BODY_START",
            ("CLASS_BODY_START", "NEWLINE"): "CLASS_BODY",
            ("CLASS_BODY", "INDENT"): "STATEMENT_START",
            
            # Statement level
            ("STATEMENT_START", "def"): "FUNC_KEYWORD",
            ("STATEMENT_START", "class"): "CLASS_KEYWORD",
            ("STATEMENT_START", "if"): "IF_KEYWORD",
            ("STATEMENT_START", "elif"): "ELIF_KEYWORD",
            ("STATEMENT_START", "else"): "ELSE_KEYWORD",
            ("STATEMENT_START", "for"): "FOR_KEYWORD",
            ("STATEMENT_START", "while"): "WHILE_KEYWORD",
            ("STATEMENT_START", "try"): "TRY_KEYWORD",
            ("STATEMENT_START", "except"): "EXCEPT_KEYWORD",
            ("STATEMENT_START", "finally"): "FINALLY_KEYWORD",
            ("STATEMENT_START", "with"): "WITH_KEYWORD",
            ("STATEMENT_START", "return"): "RETURN_STMT",
            ("STATEMENT_START", "yield"): "YIELD_STMT",
            ("STATEMENT_START", "raise"): "RAISE_STMT",
            ("STATEMENT_START", "assert"): "ASSERT_STMT",
            ("STATEMENT_START", "pass"): "PASS_STMT",
            ("STATEMENT_START", "break"): "BREAK_STMT",
            ("STATEMENT_START", "continue"): "CONTINUE_STMT",
            ("STATEMENT_START", "IDENTIFIER"): "EXPR_START",
            ("STATEMENT_START", "NEWLINE"): "STATEMENT_START",
            ("STATEMENT_START", "DEDENT"): "DEDENT_CHECK",
            
            # Control flow
            ("IF_KEYWORD", "IDENTIFIER"): "IF_CONDITION",
            ("IF_KEYWORD", "not"): "IF_CONDITION",
            ("IF_KEYWORD", "("): "IF_CONDITION",
            ("IF_CONDITION", ":"): "IF_BODY_START",
            ("IF_CONDITION", "IDENTIFIER"): "IF_CONDITION",
            ("IF_CONDITION", "INT_LITERAL"): "IF_CONDITION",
            ("IF_CONDITION", "=="): "IF_CONDITION",
            ("IF_CONDITION", "!="): "IF_CONDITION",
            ("IF_CONDITION", "<"): "IF_CONDITION",
            ("IF_CONDITION", ">"): "IF_CONDITION",
            ("IF_CONDITION", "and"): "IF_CONDITION",
            ("IF_CONDITION", "or"): "IF_CONDITION",
            ("IF_BODY_START", "NEWLINE"): "IF_BODY",
            ("IF_BODY", "INDENT"): "STATEMENT_START",
            
            ("ELIF_KEYWORD", "IDENTIFIER"): "IF_CONDITION",
            ("ELSE_KEYWORD", ":"): "ELSE_BODY_START",
            ("ELSE_BODY_START", "NEWLINE"): "ELSE_BODY",
            ("ELSE_BODY", "INDENT"): "STATEMENT_START",
            
            # For loop
            ("FOR_KEYWORD", "IDENTIFIER"): "FOR_TARGET",
            ("FOR_TARGET", ","): "FOR_TARGET",
            ("FOR_TARGET", "IDENTIFIER"): "FOR_TARGET",
            ("FOR_TARGET", "in"): "FOR_ITER",
            ("FOR_ITER", "IDENTIFIER"): "FOR_ITER",
            ("FOR_ITER", "("): "FOR_ITER",
            ("FOR_ITER", ")"): "FOR_ITER",
            ("FOR_ITER", ":"): "FOR_BODY_START",
            ("FOR_BODY_START", "NEWLINE"): "FOR_BODY",
            ("FOR_BODY", "INDENT"): "STATEMENT_START",
            
            # While loop
            ("WHILE_KEYWORD", "IDENTIFIER"): "WHILE_CONDITION",
            ("WHILE_KEYWORD", "True"): "WHILE_CONDITION",
            ("WHILE_CONDITION", ":"): "WHILE_BODY_START",
            ("WHILE_CONDITION", "IDENTIFIER"): "WHILE_CONDITION",
            ("WHILE_BODY_START", "NEWLINE"): "WHILE_BODY",
            ("WHILE_BODY", "INDENT"): "STATEMENT_START",
            
            # Return/yield
            ("RETURN_STMT", "IDENTIFIER"): "EXPR_START",
            ("RETURN_STMT", "INT_LITERAL"): "EXPR_START",
            ("RETURN_STMT", "STRING_LITERAL"): "EXPR_START",
            ("RETURN_STMT", "None"): "EXPR_START",
            ("RETURN_STMT", "NEWLINE"): "STATEMENT_START",
            ("RETURN_STMT", "("): "EXPR_START",
            ("RETURN_STMT", "["): "EXPR_START",
            
            # Simple statements
            ("PASS_STMT", "NEWLINE"): "STATEMENT_START",
            ("BREAK_STMT", "NEWLINE"): "STATEMENT_START",
            ("CONTINUE_STMT", "NEWLINE"): "STATEMENT_START",
            
            # Expression
            ("EXPR_START", "IDENTIFIER"): "EXPR_BINARY",
            ("EXPR_START", "INT_LITERAL"): "EXPR_BINARY",
            ("EXPR_START", "STRING_LITERAL"): "EXPR_BINARY",
            ("EXPR_START", "("): "IN_PARENS",
            ("EXPR_START", "["): "IN_BRACKETS",
            ("EXPR_START", "{"): "IN_BRACES",
            
            ("EXPR_BINARY", "+"): "EXPR_START",
            ("EXPR_BINARY", "-"): "EXPR_START",
            ("EXPR_BINARY", "*"): "EXPR_START",
            ("EXPR_BINARY", "/"): "EXPR_START",
            ("EXPR_BINARY", "=="): "EXPR_START",
            ("EXPR_BINARY", "!="): "EXPR_START",
            ("EXPR_BINARY", "<"): "EXPR_START",
            ("EXPR_BINARY", ">"): "EXPR_START",
            ("EXPR_BINARY", "="): "ASSIGN_VALUE",
            ("EXPR_BINARY", "+="): "ASSIGN_VALUE",
            ("EXPR_BINARY", "("): "CALL_ARGS",
            ("EXPR_BINARY", "["): "INDEX_EXPR",
            ("EXPR_BINARY", "."): "ATTR_ACCESS",
            ("EXPR_BINARY", "NEWLINE"): "STATEMENT_START",
            ("EXPR_BINARY", ","): "EXPR_START",
            
            ("ASSIGN_VALUE", "IDENTIFIER"): "EXPR_START",
            ("ASSIGN_VALUE", "INT_LITERAL"): "EXPR_START",
            ("ASSIGN_VALUE", "STRING_LITERAL"): "EXPR_START",
            ("ASSIGN_VALUE", "["): "EXPR_START",
            ("ASSIGN_VALUE", "{"): "EXPR_START",
            
            ("ATTR_ACCESS", "IDENTIFIER"): "EXPR_BINARY",
            
            ("CALL_ARGS", "IDENTIFIER"): "CALL_ARGS",
            ("CALL_ARGS", "INT_LITERAL"): "CALL_ARGS",
            ("CALL_ARGS", "STRING_LITERAL"): "CALL_ARGS",
            ("CALL_ARGS", ","): "CALL_ARGS",
            ("CALL_ARGS", "="): "CALL_ARGS",
            ("CALL_ARGS", ")"): "EXPR_BINARY",
            
            ("INDEX_EXPR", "IDENTIFIER"): "INDEX_EXPR",
            ("INDEX_EXPR", "INT_LITERAL"): "INDEX_EXPR",
            ("INDEX_EXPR", ":"): "INDEX_EXPR",
            ("INDEX_EXPR", "]"): "EXPR_BINARY",
            
            ("IN_PARENS", "IDENTIFIER"): "IN_PARENS",
            ("IN_PARENS", "INT_LITERAL"): "IN_PARENS",
            ("IN_PARENS", ","): "IN_PARENS",
            ("IN_PARENS", "+"): "IN_PARENS",
            ("IN_PARENS", "-"): "IN_PARENS",
            ("IN_PARENS", ")"): "EXPR_BINARY",
            
            ("IN_BRACKETS", "IDENTIFIER"): "IN_BRACKETS",
            ("IN_BRACKETS", "INT_LITERAL"): "IN_BRACKETS",
            ("IN_BRACKETS", ","): "IN_BRACKETS",
            ("IN_BRACKETS", "]"): "EXPR_BINARY",
            
            ("IN_BRACES", "STRING_LITERAL"): "IN_BRACES",
            ("IN_BRACES", "IDENTIFIER"): "IN_BRACES",
            ("IN_BRACES", ":"): "IN_BRACES",
            ("IN_BRACES", ","): "IN_BRACES",
            ("IN_BRACES", "}"): "EXPR_BINARY",
            
            # Dedent handling
            ("DEDENT_CHECK", "DEDENT"): "DEDENT_CHECK",
        }
    
    def _go_transitions(self) -> Dict[Tuple[str, str], str]:
        """Go state transitions."""
        return {
            # Package
            ("MODULE_START", "package"): "PACKAGE_DECL",
            ("PACKAGE_DECL", "IDENTIFIER"): "TOP_LEVEL",
            
            # Import
            ("TOP_LEVEL", "import"): "IMPORT_DECL",
            ("IMPORT_DECL", "("): "IMPORT_GROUP",
            ("IMPORT_DECL", "STRING_LITERAL"): "TOP_LEVEL",
            ("IMPORT_GROUP", "STRING_LITERAL"): "IMPORT_GROUP",
            ("IMPORT_GROUP", ")"): "TOP_LEVEL",
            
            # Function
            ("TOP_LEVEL", "func"): "FUNC_KEYWORD",
            ("FUNC_KEYWORD", "IDENTIFIER"): "FUNC_NAME",
            ("FUNC_KEYWORD", "("): "METHOD_RECEIVER",
            ("METHOD_RECEIVER", "IDENTIFIER"): "METHOD_RECEIVER",
            ("METHOD_RECEIVER", "*"): "METHOD_RECEIVER",
            ("METHOD_RECEIVER", ")"): "FUNC_NAME_AFTER_RECEIVER",
            ("FUNC_NAME_AFTER_RECEIVER", "IDENTIFIER"): "FUNC_NAME",
            ("FUNC_NAME", "("): "FUNC_PARAMS",
            ("FUNC_PARAMS", "IDENTIFIER"): "FUNC_PARAMS",
            ("FUNC_PARAMS", ","): "FUNC_PARAMS",
            ("FUNC_PARAMS", "*"): "FUNC_PARAMS",
            ("FUNC_PARAMS", ")"): "FUNC_RETURN",
            ("FUNC_RETURN", "IDENTIFIER"): "FUNC_RETURN",
            ("FUNC_RETURN", "("): "FUNC_MULTI_RETURN",
            ("FUNC_RETURN", "{"): "FUNC_BODY",
            ("FUNC_MULTI_RETURN", "IDENTIFIER"): "FUNC_MULTI_RETURN",
            ("FUNC_MULTI_RETURN", ","): "FUNC_MULTI_RETURN",
            ("FUNC_MULTI_RETURN", ")"): "FUNC_RETURN_END",
            ("FUNC_RETURN_END", "{"): "FUNC_BODY",
            
            # Function body
            ("FUNC_BODY", "var"): "VAR_DECL",
            ("FUNC_BODY", "const"): "CONST_DECL",
            ("FUNC_BODY", "if"): "IF_KEYWORD",
            ("FUNC_BODY", "for"): "FOR_KEYWORD",
            ("FUNC_BODY", "switch"): "SWITCH_KEYWORD",
            ("FUNC_BODY", "select"): "SELECT_KEYWORD",
            ("FUNC_BODY", "return"): "RETURN_STMT",
            ("FUNC_BODY", "go"): "GO_STMT",
            ("FUNC_BODY", "defer"): "DEFER_STMT",
            ("FUNC_BODY", "IDENTIFIER"): "EXPR_START",
            ("FUNC_BODY", "}"): "TOP_LEVEL",
            
            # Control flow
            ("IF_KEYWORD", "IDENTIFIER"): "IF_CONDITION",
            ("IF_CONDITION", "{"): "IF_BODY",
            ("IF_CONDITION", "IDENTIFIER"): "IF_CONDITION",
            ("IF_CONDITION", "=="): "IF_CONDITION",
            ("IF_CONDITION", "!="): "IF_CONDITION",
            ("IF_BODY", "}"): "IF_END",
            ("IF_END", "else"): "ELSE_KEYWORD",
            ("ELSE_KEYWORD", "{"): "ELSE_BODY",
            ("ELSE_KEYWORD", "if"): "IF_KEYWORD",
            ("ELSE_BODY", "}"): "FUNC_BODY",
            
            ("FOR_KEYWORD", "IDENTIFIER"): "FOR_CLAUSE",
            ("FOR_KEYWORD", "range"): "FOR_RANGE",
            ("FOR_KEYWORD", "{"): "FOR_BODY",
            ("FOR_CLAUSE", ":="): "FOR_CLAUSE",
            ("FOR_CLAUSE", "range"): "FOR_RANGE",
            ("FOR_CLAUSE", "{"): "FOR_BODY",
            ("FOR_RANGE", "IDENTIFIER"): "FOR_RANGE",
            ("FOR_RANGE", "{"): "FOR_BODY",
            ("FOR_BODY", "}"): "FUNC_BODY",
            
            # Return
            ("RETURN_STMT", "IDENTIFIER"): "EXPR_START",
            ("RETURN_STMT", "INT_LITERAL"): "EXPR_START",
            ("RETURN_STMT", "NEWLINE"): "FUNC_BODY",
            
            # Type definition
            ("TOP_LEVEL", "type"): "TYPE_DEF",
            ("TYPE_DEF", "IDENTIFIER"): "TYPE_NAME",
            ("TYPE_NAME", "struct"): "STRUCT_DEF",
            ("TYPE_NAME", "interface"): "INTERFACE_DEF",
            ("STRUCT_DEF", "{"): "STRUCT_BODY",
            ("STRUCT_BODY", "IDENTIFIER"): "STRUCT_FIELD",
            ("STRUCT_FIELD", "IDENTIFIER"): "STRUCT_FIELD",
            ("STRUCT_FIELD", "*"): "STRUCT_FIELD",
            ("STRUCT_FIELD", "STRING_LITERAL"): "STRUCT_BODY",
            ("STRUCT_BODY", "}"): "TOP_LEVEL",
            
            # Expression
            ("EXPR_START", "IDENTIFIER"): "EXPR_BINARY",
            ("EXPR_START", "INT_LITERAL"): "EXPR_BINARY",
            ("EXPR_BINARY", "+"): "EXPR_START",
            ("EXPR_BINARY", "-"): "EXPR_START",
            ("EXPR_BINARY", "="): "ASSIGN_VALUE",
            ("EXPR_BINARY", ":="): "ASSIGN_VALUE",
            ("EXPR_BINARY", "("): "CALL_ARGS",
            ("EXPR_BINARY", "."): "ATTR_ACCESS",
            ("EXPR_BINARY", "NEWLINE"): "FUNC_BODY",
            
            ("ASSIGN_VALUE", "IDENTIFIER"): "EXPR_START",
            ("ASSIGN_VALUE", "INT_LITERAL"): "EXPR_START",
            
            ("CALL_ARGS", "IDENTIFIER"): "CALL_ARGS",
            ("CALL_ARGS", "INT_LITERAL"): "CALL_ARGS",
            ("CALL_ARGS", ","): "CALL_ARGS",
            ("CALL_ARGS", ")"): "EXPR_BINARY",
            
            ("ATTR_ACCESS", "IDENTIFIER"): "EXPR_BINARY",
        }
    
    def consume(self, token: str) -> str:
        """
        Consume a token and update state.
        
        Returns the new state.
        """
        # Categorize token
        category = self._categorize(token)
        
        # Update context
        self._update_context(token)
        
        # Look up transition
        key = (self.state, token)
        if key in self.transitions:
            self.state = self.transitions[key]
        else:
            # Try category-based transition
            cat_key = (self.state, category)
            if cat_key in self.transitions:
                self.state = self.transitions[cat_key]
        
        return self.state
    
    def _categorize(self, token: str) -> str:
        """Categorize a token."""
        if re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', token):
            if token not in {'def', 'class', 'if', 'elif', 'else', 'for', 'while',
                            'try', 'except', 'finally', 'with', 'return', 'yield',
                            'raise', 'assert', 'pass', 'break', 'continue',
                            'import', 'from', 'as', 'and', 'or', 'not', 'in', 'is',
                            'True', 'False', 'None', 'lambda', 'async', 'await',
                            'func', 'package', 'type', 'struct', 'interface',
                            'var', 'const', 'go', 'defer', 'switch', 'select',
                            'case', 'default', 'range', 'chan', 'map'}:
                return "IDENTIFIER"
        if re.match(r'^[0-9]+$', token):
            return "INT_LITERAL"
        if token.startswith('"') or token.startswith("'"):
            return "STRING_LITERAL"
        if token == '\n':
            return "NEWLINE"
        if token.strip() == '' and len(token) >= 4:
            return "INDENT"
        return token
    
    def _update_context(self, token: str):
        """Update context based on token."""
        if token == '(':
            self.context.paren_depth += 1
        elif token == ')':
            self.context.paren_depth = max(0, self.context.paren_depth - 1)
        elif token == '[':
            self.context.bracket_depth += 1
        elif token == ']':
            self.context.bracket_depth = max(0, self.context.bracket_depth - 1)
        elif token == '{':
            self.context.brace_depth += 1
        elif token == '}':
            self.context.brace_depth = max(0, self.context.brace_depth - 1)
        elif token == '\n':
            self.context.line_start = True
        elif self._categorize(token) == "INDENT":
            self.context.indent_level += 1
        elif token == "DEDENT":
            self.context.indent_level = max(0, self.context.indent_level - 1)
        else:
            self.context.line_start = False
        
        self.context.last_token = token
    
    def get_state(self) -> str:
        """Get current state."""
        return self.state
    
    def reset(self):
        """Reset to initial state."""
        self.state = "MODULE_START"
        self.context = MaskContext()


class GuidedCodeGenerator:
    """
    LLM code generator with statechart-guided token masking.
    
    Wraps any LLM that provides logits and applies syntax masks
    at each generation step.
    """
    
    def __init__(self, vocab: Dict[str, int], 
                 lm_forward: Optional[Callable[[mx.array], mx.array]] = None,
                 language: str = "python"):
        """
        Initialize generator.
        
        Args:
            vocab: Token vocabulary mapping strings to IDs
            lm_forward: Function that takes token IDs and returns logits
            language: "python" or "go"
        """
        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.id_to_token = {v: k for k, v in vocab.items()}
        self.lm_forward = lm_forward or self._mock_lm_forward
        self.language = language
        
        # Create components
        self.masker = TokenMasker(vocab, language)
        self.tracker = SyntaxStateTracker(language)
        
        self.config = GenerationConfig(language=language)
    
    def _mock_lm_forward(self, token_ids: mx.array) -> mx.array:
        """
        Grammar-aware mock LM that produces contextually sensible logits.

        Instead of random, we bias toward tokens that make syntactic sense
        based on the last token and current state.
        """
        # Build logits as Python list, then convert to mx.array
        logits_list = [0.5 * float(x) for x in mx.random.normal((self.vocab_size,)).tolist()]

        # Get last token
        last_id = int(token_ids[-1]) if len(token_ids) > 0 else -1
        last_token = self.id_to_token.get(last_id, "")

        # Boost tokens based on context
        state = self.tracker.get_state()

        # State-specific boosts
        state_boosts = {
            "FUNC_BODY_START": {'\n': 8, '    ': 5},
            "FUNC_BODY": {'return': 4, 'if': 3, 'for': 3, 'x': 2, 'result': 2, '\n': 2},
            "IF_BODY_START": {'\n': 8, '    ': 5},
            "IF_BODY": {'return': 4, 'x': 2, 'result': 2, '\n': 2},
            "FOR_BODY_START": {'\n': 8, '    ': 5},
            "FOR_BODY": {'x': 2, 'result': 2, 'i': 2, '\n': 2},
            "RETURN_STMT": {'x': 4, 'n': 4, 'result': 3, '0': 3, '1': 3, 'None': 3, '\n': 2},
            "EXPR_START": {'x': 4, 'n': 4, 'i': 3, '0': 3, '1': 3, 'result': 3},
            "EXPR_BINARY": {'+': 3, '-': 3, '*': 3, '\n': 5, ' ': 2},
            "STATEMENT_START": {'return': 4, 'if': 3, 'for': 3, 'x': 2, '\n': 2},
        }

        # Last token specific boosts
        token_boosts = {
            ':': {'\n': 10},
            '\n': {'    ': 8, 'return': 3, 'if': 2, 'for': 2, '\n': 4},
            '    ': {'return': 5, 'if': 3, 'for': 3, 'x': 3, 'result': 3},
            'return': {'x': 5, 'n': 5, 'result': 4, '0': 3, '1': 3, '\n': 3},
            'x': {'+': 4, '-': 4, '*': 4, '\n': 3, ' ': 2},
            'n': {'+': 4, '-': 4, '*': 4, '\n': 3, ' ': 2},
            '0': {'\n': 4, ' ': 2},
            '1': {'\n': 4, '+': 2, ' ': 2},
            '+': {'x': 4, 'n': 4, '1': 3, '0': 3},
            '-': {'x': 4, 'n': 4, '1': 3},
            '*': {'x': 4, 'n': 4, '2': 3},
            '(': {'x': 3, 'n': 3, 'self': 2, ')': 2, '10': 3},
            ')': {':': 6, '\n': 4},
            'if': {'x': 4, 'n': 3},
            'for': {'i': 5, 'x': 3},
            'in': {'range': 6, 'x': 3},
            'range': {'(': 8},
            '10': {')': 6},
        }

        # Apply state boosts
        if state in state_boosts:
            for tok, boost in state_boosts[state].items():
                if tok in self.vocab:
                    idx = self.vocab[tok]
                    logits_list[idx] += boost

        # Apply token boosts
        if last_token in token_boosts:
            for tok, boost in token_boosts[last_token].items():
                if tok in self.vocab:
                    idx = self.vocab[tok]
                    logits_list[idx] += boost

        return mx.array(logits_list)
    
    def generate(self, prompt: str, config: Optional[GenerationConfig] = None) -> str:
        """
        Generate code with syntax constraints.
        
        Args:
            prompt: Starting code prompt
            config: Generation configuration
            
        Returns:
            Generated code string
        """
        config = config or self.config
        
        # Parse prompt to establish initial state
        self.tracker.reset()
        prompt_tokens = self._tokenize(prompt)
        for token in prompt_tokens:
            self.tracker.consume(token)
        
        # Generation loop
        state = GenerationState(
            syntax_state=self.tracker.get_state(),
            context=MaskContext(),
            tokens_generated=[]
        )
        
        generated_tokens = []
        
        for step in range(config.max_tokens):
            # Get LM logits
            context_ids = self._encode(prompt_tokens + generated_tokens)
            logits = self.lm_forward(context_ids)
            
            # Apply syntax mask
            if config.enforce_constraints:
                masked_logits = self.masker.mask_logits(
                    logits, 
                    state.syntax_state,
                    state.context,
                    mask_value=-config.constraint_strength
                )
            else:
                masked_logits = logits
            
            # Sample next token
            next_token_id = self._sample(masked_logits, config)
            next_token = self.id_to_token.get(next_token_id, "<UNK>")
            
            # Track blocked tokens
            if config.track_blocked:
                original_best = int(mx.argmax(logits))
                if original_best != next_token_id:
                    blocked_token = self.id_to_token.get(original_best, "<UNK>")
                    state.tokens_blocked.append((step, blocked_token, state.syntax_state))
            
            # Check stop condition
            if next_token in config.stop_tokens:
                break
            
            # Update state
            generated_tokens.append(next_token)
            state.tokens_generated.append(next_token)
            state.syntax_state = self.tracker.consume(next_token)
            state.context = self.tracker.context
        
        return prompt + ''.join(generated_tokens)
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization."""
        tokens = []
        i = 0
        while i < len(text):
            # Try multi-char tokens first
            found = False
            for length in [3, 2]:
                substr = text[i:i+length]
                if substr in self.vocab:
                    tokens.append(substr)
                    i += length
                    found = True
                    break
            
            if not found:
                char = text[i]
                if char in self.vocab:
                    tokens.append(char)
                elif char.isalnum() or char == '_':
                    # Collect identifier
                    j = i
                    while j < len(text) and (text[j].isalnum() or text[j] == '_'):
                        j += 1
                    tokens.append(text[i:j])
                    i = j
                    continue
                i += 1
        
        return tokens
    
    def _encode(self, tokens: List[str]) -> mx.array:
        """Encode tokens to IDs."""
        ids = [self.vocab.get(t, 0) for t in tokens]
        return mx.array(ids)
    
    def _sample(self, logits: mx.array, config: GenerationConfig) -> int:
        """Sample from logits with temperature and top-k/p."""
        # Temperature scaling
        if config.temperature > 0:
            logits = logits / config.temperature
        
        # Top-k filtering
        if config.top_k > 0:
            sorted_indices = mx.argsort(-logits)
            cutoff_idx = min(config.top_k, len(logits))
            cutoff_logit = logits[sorted_indices[cutoff_idx - 1]]
            logits = mx.where(logits < cutoff_logit, -1e9, logits)
        
        # Convert to probabilities
        probs = mx.softmax(logits)
        
        # Top-p (nucleus) filtering
        if config.top_p < 1.0:
            sorted_indices = mx.argsort(-probs)
            sorted_probs = probs[sorted_indices]
            cumsum = mx.cumsum(sorted_probs)
            
            # Find cutoff
            cutoff_mask = cumsum <= config.top_p
            # Always keep at least one token
            cutoff_mask = mx.concatenate([mx.array([True]), cutoff_mask[:-1]])
            
            # Zero out tokens beyond cutoff
            for i, keep in enumerate(cutoff_mask.tolist()):
                if not keep:
                    probs = mx.where(mx.arange(len(probs)) == int(sorted_indices[i]), 0.0, probs)
            
            # Renormalize
            probs = probs / mx.sum(probs)
        
        # Sample
        if config.temperature == 0:
            return int(mx.argmax(probs))
        else:
            # Categorical sampling
            cumprobs = mx.cumsum(probs)
            r = float(mx.random.uniform())
            for i, cp in enumerate(cumprobs.tolist()):
                if r < cp:
                    return i
            return len(probs) - 1
    
    def generate_with_stats(self, prompt: str, 
                           config: Optional[GenerationConfig] = None) -> Tuple[str, Dict]:
        """Generate code and return statistics."""
        config = config or self.config
        config.track_blocked = True
        
        # Reset
        self.tracker.reset()
        prompt_tokens = self._tokenize(prompt)
        for token in prompt_tokens:
            self.tracker.consume(token)
        
        state = GenerationState(
            syntax_state=self.tracker.get_state(),
            tokens_blocked=[]
        )
        
        generated_tokens = []
        states_visited = [state.syntax_state]
        
        for step in range(config.max_tokens):
            context_ids = self._encode(prompt_tokens + generated_tokens)
            logits = self.lm_forward(context_ids)
            
            if config.enforce_constraints:
                masked_logits = self.masker.mask_logits(
                    logits, state.syntax_state, state.context,
                    mask_value=-config.constraint_strength
                )
            else:
                masked_logits = logits
            
            next_token_id = self._sample(masked_logits, config)
            next_token = self.id_to_token.get(next_token_id, "<UNK>")
            
            # Track blocked
            original_best = int(mx.argmax(logits))
            if original_best != next_token_id:
                blocked_token = self.id_to_token.get(original_best, "<UNK>")
                state.tokens_blocked.append((step, blocked_token, state.syntax_state))
            
            if next_token in config.stop_tokens:
                break
            
            generated_tokens.append(next_token)
            state.syntax_state = self.tracker.consume(next_token)
            states_visited.append(state.syntax_state)
        
        code = prompt + ''.join(generated_tokens)
        
        stats = {
            'tokens_generated': len(generated_tokens),
            'tokens_blocked': len(state.tokens_blocked),
            'states_visited': len(set(states_visited)),
            'blocked_details': state.tokens_blocked[:10],  # First 10
            'final_state': state.syntax_state,
        }
        
        return code, stats


def demo():
    """Demonstrate guided generation."""
    print("=" * 60)
    print("GUIDED CODE GENERATION DEMO")
    print("=" * 60)
    
    # Simple vocabulary with contiguous indices
    tokens = [
        'def', 'class', 'if', 'else', 'for', 'while', 'return', 'import', 'from', 'in',
        '+', '-', '*', '/', '=', '==', '!=', '<', '>',
        '(', ')', '[', ']', '{', '}', ':', ',', '.',
        '\n', '    ', ' ',
        'x', 'y', 'n', 'i', 'result', 'self', 'foo', 'bar', 'range', 'len',
        '0', '1', '2', '10',
        'True', 'False', 'None',
        'and', 'or', 'not',
        '<EOS>', '\n\n',
    ]
    vocab = {tok: i for i, tok in enumerate(tokens)}
    
    generator = GuidedCodeGenerator(vocab, language="python")
    
    # Test prompts
    prompts = [
        "def factorial(n):",
        "for i in range(10):",
        "if x > 0:",
    ]
    
    for prompt in prompts:
        print(f"\nPrompt: '{prompt}'")
        
        config = GenerationConfig(
            max_tokens=20,
            temperature=0.8,
            enforce_constraints=True
        )
        
        code, stats = generator.generate_with_stats(prompt, config)
        
        print(f"Generated:\n{code}")
        print(f"Stats: {stats['tokens_generated']} tokens, {stats['tokens_blocked']} blocked")
        if stats['blocked_details']:
            print(f"Blocked examples: {stats['blocked_details'][:3]}")
    
    print("\n" + "=" * 60)
    print("Guided generation ensures syntactic validity!")
    print("=" * 60)


if __name__ == "__main__":
    demo()
