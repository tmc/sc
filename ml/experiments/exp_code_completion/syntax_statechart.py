"""
Python Syntax Statechart

Defines a statechart that tracks Python syntax state during generation.
Each state determines which tokens are valid next.

States model the Python grammar at a high level:
- MODULE: Top-level, expecting statement or definition
- FUNCTION_DEF: Inside function definition (def name...)
- CLASS_DEF: Inside class definition
- STATEMENT: Expecting a statement
- EXPRESSION: Inside an expression
- BLOCK: Indented block expecting statements
- STRING: Inside a string literal
- COMMENT: Inside a comment
- PARAMS: Inside function parameters
- ARGS: Inside function call arguments

Transitions are triggered by tokens (keywords, operators, literals).
Guards check context (indent level, bracket depth, etc.).
"""

from enum import Enum, auto
from dataclasses import dataclass, field
from typing import List, Set, Dict, Optional, Tuple
import re


class PythonSyntaxState(Enum):
    """High-level Python syntax states."""
    MODULE = auto()        # Top-level module
    FUNCTION_DEF = auto()  # After 'def', expecting name
    FUNCTION_PARAMS = auto()  # Inside (params)
    FUNCTION_BODY = auto()   # After ':', in function body
    CLASS_DEF = auto()     # After 'class', expecting name
    CLASS_BODY = auto()    # Inside class body
    STATEMENT = auto()     # Expecting a statement
    EXPRESSION = auto()    # Inside an expression
    ASSIGNMENT = auto()    # After '=', expecting value
    IF_CONDITION = auto()  # After 'if', expecting condition
    IF_BODY = auto()       # Inside if body
    ELIF_CONDITION = auto()
    ELSE_BODY = auto()
    FOR_TARGET = auto()    # After 'for', expecting target
    FOR_ITER = auto()      # After 'in', expecting iterable
    FOR_BODY = auto()
    WHILE_CONDITION = auto()
    WHILE_BODY = auto()
    TRY_BODY = auto()
    EXCEPT_HANDLER = auto()
    FINALLY_BODY = auto()
    WITH_CONTEXT = auto()
    WITH_BODY = auto()
    RETURN_VALUE = auto()  # After 'return'
    IMPORT = auto()        # After 'import'
    FROM_IMPORT = auto()   # After 'from'
    STRING_SINGLE = auto() # Inside '...'
    STRING_DOUBLE = auto() # Inside "..."
    STRING_TRIPLE = auto() # Inside '''...''' or """..."""
    FSTRING = auto()       # Inside f-string
    FSTRING_EXPR = auto()  # Inside {expr} in f-string
    COMMENT = auto()       # After #
    DECORATOR = auto()     # After @
    LIST_LITERAL = auto()  # Inside [...]
    DICT_LITERAL = auto()  # Inside {...}
    TUPLE_LITERAL = auto() # Inside (...)
    CALL_ARGS = auto()     # Inside function(args)
    SUBSCRIPT = auto()     # Inside obj[...]
    COMPREHENSION = auto() # Inside list/dict/set comprehension
    LAMBDA = auto()        # After 'lambda'
    ASSERT = auto()        # After 'assert'
    RAISE = auto()         # After 'raise'
    YIELD = auto()         # After 'yield'
    AWAIT = auto()         # After 'await'


# Token categories
KEYWORDS = {
    'def', 'class', 'if', 'elif', 'else', 'for', 'while', 'try', 'except',
    'finally', 'with', 'as', 'return', 'yield', 'raise', 'import', 'from',
    'pass', 'break', 'continue', 'and', 'or', 'not', 'in', 'is', 'lambda',
    'True', 'False', 'None', 'async', 'await', 'assert', 'global', 'nonlocal',
    'del', 'match', 'case',
}

OPERATORS = {
    '+', '-', '*', '/', '//', '%', '**', '@',
    '==', '!=', '<', '>', '<=', '>=',
    '=', '+=', '-=', '*=', '/=', '//=', '%=', '**=', '@=',
    '&', '|', '^', '~', '<<', '>>', '&=', '|=', '^=', '<<=', '>>=',
    '->', ':=',
}

DELIMITERS = {
    '(', ')', '[', ']', '{', '}',
    ',', ':', ';', '.', '...', '\\',
}

STRING_PREFIXES = {'f', 'r', 'b', 'fr', 'rf', 'br', 'rb', 'F', 'R', 'B'}


@dataclass
class SyntaxContext:
    """Tracks syntax context during parsing/generation."""
    indent_level: int = 0
    paren_depth: int = 0      # ()
    bracket_depth: int = 0     # []
    brace_depth: int = 0       # {}
    in_string: bool = False
    string_char: str = ''      # ' or " or ''' or """
    in_fstring: bool = False
    fstring_brace_depth: int = 0
    in_comment: bool = False
    expecting_indent: bool = False
    line_start: bool = True
    last_token: str = ''
    last_tokens: List[str] = field(default_factory=list)

    def clone(self) -> 'SyntaxContext':
        return SyntaxContext(
            indent_level=self.indent_level,
            paren_depth=self.paren_depth,
            bracket_depth=self.bracket_depth,
            brace_depth=self.brace_depth,
            in_string=self.in_string,
            string_char=self.string_char,
            in_fstring=self.in_fstring,
            fstring_brace_depth=self.fstring_brace_depth,
            in_comment=self.in_comment,
            expecting_indent=self.expecting_indent,
            line_start=self.line_start,
            last_token=self.last_token,
            last_tokens=self.last_tokens.copy(),
        )


@dataclass
class StateTransition:
    """A transition in the syntax statechart."""
    from_state: PythonSyntaxState
    to_state: PythonSyntaxState
    trigger: str  # Token or token pattern
    guard: Optional[str] = None  # Condition as string
    action: Optional[str] = None  # Context update


class PythonSyntaxStatechart:
    """
    Statechart for Python syntax.

    Tracks the current syntax state and provides:
    1. Valid next tokens for current state
    2. State transitions on token consumption
    3. Context updates (indent, brackets, etc.)
    """

    def __init__(self):
        self.state = PythonSyntaxState.MODULE
        self.context = SyntaxContext()
        self.state_stack: List[Tuple[PythonSyntaxState, SyntaxContext]] = []

        # Build transition table
        self.transitions = self._build_transitions()

        # Valid tokens per state
        self.valid_tokens = self._build_valid_tokens()

    def _build_transitions(self) -> Dict[PythonSyntaxState, List[StateTransition]]:
        """Build the state transition table."""
        transitions = {}

        # MODULE state transitions
        transitions[PythonSyntaxState.MODULE] = [
            StateTransition(PythonSyntaxState.MODULE, PythonSyntaxState.FUNCTION_DEF, 'def'),
            StateTransition(PythonSyntaxState.MODULE, PythonSyntaxState.CLASS_DEF, 'class'),
            StateTransition(PythonSyntaxState.MODULE, PythonSyntaxState.IMPORT, 'import'),
            StateTransition(PythonSyntaxState.MODULE, PythonSyntaxState.FROM_IMPORT, 'from'),
            StateTransition(PythonSyntaxState.MODULE, PythonSyntaxState.DECORATOR, '@'),
            StateTransition(PythonSyntaxState.MODULE, PythonSyntaxState.IF_CONDITION, 'if'),
            StateTransition(PythonSyntaxState.MODULE, PythonSyntaxState.FOR_TARGET, 'for'),
            StateTransition(PythonSyntaxState.MODULE, PythonSyntaxState.WHILE_CONDITION, 'while'),
            StateTransition(PythonSyntaxState.MODULE, PythonSyntaxState.EXPRESSION, 'IDENTIFIER'),
        ]

        # FUNCTION_DEF transitions
        transitions[PythonSyntaxState.FUNCTION_DEF] = [
            StateTransition(PythonSyntaxState.FUNCTION_DEF, PythonSyntaxState.FUNCTION_PARAMS, '('),
        ]

        # FUNCTION_PARAMS transitions
        transitions[PythonSyntaxState.FUNCTION_PARAMS] = [
            StateTransition(PythonSyntaxState.FUNCTION_PARAMS, PythonSyntaxState.FUNCTION_BODY, '):',
                           guard='paren_depth == 0'),
        ]

        # FUNCTION_BODY transitions
        transitions[PythonSyntaxState.FUNCTION_BODY] = [
            StateTransition(PythonSyntaxState.FUNCTION_BODY, PythonSyntaxState.RETURN_VALUE, 'return'),
            StateTransition(PythonSyntaxState.FUNCTION_BODY, PythonSyntaxState.IF_CONDITION, 'if'),
            StateTransition(PythonSyntaxState.FUNCTION_BODY, PythonSyntaxState.FOR_TARGET, 'for'),
            StateTransition(PythonSyntaxState.FUNCTION_BODY, PythonSyntaxState.WHILE_CONDITION, 'while'),
            StateTransition(PythonSyntaxState.FUNCTION_BODY, PythonSyntaxState.EXPRESSION, 'IDENTIFIER'),
            StateTransition(PythonSyntaxState.FUNCTION_BODY, PythonSyntaxState.MODULE, 'DEDENT',
                           guard='indent_level == 0'),
        ]

        # IF_CONDITION transitions
        transitions[PythonSyntaxState.IF_CONDITION] = [
            StateTransition(PythonSyntaxState.IF_CONDITION, PythonSyntaxState.IF_BODY, ':'),
        ]

        # IF_BODY transitions
        transitions[PythonSyntaxState.IF_BODY] = [
            StateTransition(PythonSyntaxState.IF_BODY, PythonSyntaxState.ELIF_CONDITION, 'elif'),
            StateTransition(PythonSyntaxState.IF_BODY, PythonSyntaxState.ELSE_BODY, 'else'),
        ]

        # Add more transitions...

        return transitions

    def _build_valid_tokens(self) -> Dict[PythonSyntaxState, Set[str]]:
        """Build valid token sets per state."""
        valid = {}

        # MODULE: can start any top-level construct
        valid[PythonSyntaxState.MODULE] = {
            'def', 'class', 'if', 'for', 'while', 'try', 'with',
            'import', 'from', '@', 'async', 'match',
            'IDENTIFIER', 'NEWLINE', 'COMMENT',
        }

        # FUNCTION_DEF: expecting function name
        valid[PythonSyntaxState.FUNCTION_DEF] = {'IDENTIFIER'}

        # FUNCTION_PARAMS: inside parameter list
        valid[PythonSyntaxState.FUNCTION_PARAMS] = {
            'IDENTIFIER', ',', ':', '=', '*', '**', ')',
            'int', 'str', 'float', 'bool', 'list', 'dict', 'None',
        }

        # FUNCTION_BODY: statements allowed in function
        valid[PythonSyntaxState.FUNCTION_BODY] = {
            'return', 'yield', 'if', 'for', 'while', 'try', 'with',
            'raise', 'assert', 'pass', 'break', 'continue',
            'global', 'nonlocal', 'del', 'async', 'await',
            'IDENTIFIER', 'INDENT', 'DEDENT', 'NEWLINE',
        }

        # EXPRESSION: inside an expression
        valid[PythonSyntaxState.EXPRESSION] = {
            'IDENTIFIER', 'NUMBER', 'STRING',
            '+', '-', '*', '/', '//', '%', '**', '@',
            '(', ')', '[', ']', '{', '}',
            '.', ',', ':',
            '==', '!=', '<', '>', '<=', '>=',
            'and', 'or', 'not', 'in', 'is',
            'True', 'False', 'None',
            'lambda', 'if', 'else', 'for',  # Comprehension keywords
        }

        # IF_CONDITION: expecting boolean expression
        valid[PythonSyntaxState.IF_CONDITION] = {
            'IDENTIFIER', 'NUMBER', 'STRING',
            '(', 'not', 'True', 'False', 'None',
            'and', 'or', '==', '!=', '<', '>', '<=', '>=',
            'in', 'is', ':',
        }

        # RETURN_VALUE: expression or nothing
        valid[PythonSyntaxState.RETURN_VALUE] = {
            'IDENTIFIER', 'NUMBER', 'STRING',
            '(', '[', '{', 'None', 'True', 'False',
            'NEWLINE',  # bare return
        }

        # STRING states
        valid[PythonSyntaxState.STRING_SINGLE] = {'STRING_CONTENT', "'"}
        valid[PythonSyntaxState.STRING_DOUBLE] = {'STRING_CONTENT', '"'}
        valid[PythonSyntaxState.STRING_TRIPLE] = {'STRING_CONTENT', "'''", '"""'}

        # COMMENT: anything until newline
        valid[PythonSyntaxState.COMMENT] = {'COMMENT_CONTENT', 'NEWLINE'}

        # IMPORT: module names
        valid[PythonSyntaxState.IMPORT] = {
            'IDENTIFIER', '.', ',', 'as', 'NEWLINE',
        }

        # FROM_IMPORT: from X import Y
        valid[PythonSyntaxState.FROM_IMPORT] = {
            'IDENTIFIER', '.', 'import', '*', '(', ')', ',', 'as', 'NEWLINE',
        }

        # LIST_LITERAL
        valid[PythonSyntaxState.LIST_LITERAL] = {
            'IDENTIFIER', 'NUMBER', 'STRING',
            '[', ']', '(', ')', '{', '}',
            ',', 'for', 'if', 'in',
            'True', 'False', 'None',
            '+', '-', '*', '/',
        }

        # DICT_LITERAL
        valid[PythonSyntaxState.DICT_LITERAL] = {
            'IDENTIFIER', 'NUMBER', 'STRING',
            ':', ',', '}', 'for', 'if', 'in',
            'True', 'False', 'None',
        }

        # CALL_ARGS
        valid[PythonSyntaxState.CALL_ARGS] = {
            'IDENTIFIER', 'NUMBER', 'STRING',
            '(', ')', '[', ']', '{', '}',
            ',', '=', '*', '**',
            'True', 'False', 'None',
            '+', '-', '*', '/', 'lambda',
        }

        # Fill in remaining states with reasonable defaults
        for state in PythonSyntaxState:
            if state not in valid:
                valid[state] = {
                    'IDENTIFIER', 'NUMBER', 'STRING',
                    'NEWLINE', 'INDENT', 'DEDENT',
                }

        return valid

    def get_valid_tokens(self) -> Set[str]:
        """Get tokens valid in current state."""
        base_tokens = self.valid_tokens.get(self.state, set())

        # Adjust based on context
        result = base_tokens.copy()

        # If we have open brackets, closing is always valid
        if self.context.paren_depth > 0:
            result.add(')')
        if self.context.bracket_depth > 0:
            result.add(']')
        if self.context.brace_depth > 0:
            result.add('}')

        # If in string, only string content and closer valid
        if self.context.in_string:
            result = {'STRING_CONTENT', self.context.string_char}

        # If in comment, only comment content and newline valid
        if self.context.in_comment:
            result = {'COMMENT_CONTENT', 'NEWLINE'}

        return result

    def transition(self, token: str) -> bool:
        """
        Process a token and update state.

        Returns True if transition was valid, False otherwise.
        """
        # Update context based on token
        self._update_context(token)

        # Check for matching transition
        transitions = self.transitions.get(self.state, [])
        for trans in transitions:
            if self._matches_trigger(token, trans.trigger):
                if trans.guard is None or self._evaluate_guard(trans.guard):
                    # Take transition
                    old_state = self.state
                    self.state = trans.to_state
                    if trans.action:
                        self._execute_action(trans.action)
                    return True

        # No explicit transition, but token might still be valid in state
        return token in self.get_valid_tokens() or self._is_valid_token_type(token)

    def _update_context(self, token: str):
        """Update context based on token."""
        self.context.last_tokens.append(token)
        if len(self.context.last_tokens) > 10:
            self.context.last_tokens.pop(0)
        self.context.last_token = token

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
        elif token == 'INDENT':
            self.context.indent_level += 1
            self.context.expecting_indent = False
        elif token == 'DEDENT':
            self.context.indent_level = max(0, self.context.indent_level - 1)
        elif token == 'NEWLINE':
            self.context.line_start = True
            self.context.in_comment = False
        elif token == '#':
            self.context.in_comment = True
        elif token in ("'", '"'):
            if not self.context.in_string:
                self.context.in_string = True
                self.context.string_char = token
            elif self.context.string_char == token:
                self.context.in_string = False
                self.context.string_char = ''
        elif token == ':' and self.context.paren_depth == 0:
            self.context.expecting_indent = True
        else:
            self.context.line_start = False

    def _matches_trigger(self, token: str, trigger: str) -> bool:
        """Check if token matches trigger pattern."""
        if trigger == token:
            return True
        if trigger == 'IDENTIFIER' and re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', token):
            return True
        if trigger == 'NUMBER' and re.match(r'^[0-9]+\.?[0-9]*$', token):
            return True
        if trigger == 'STRING' and token.startswith(('"', "'")):
            return True
        return False

    def _is_valid_token_type(self, token: str) -> bool:
        """Check if token is a valid type for current state."""
        valid = self.get_valid_tokens()

        # Check type patterns
        if 'IDENTIFIER' in valid and re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', token):
            if token not in KEYWORDS:
                return True
        if 'NUMBER' in valid and re.match(r'^[0-9]+\.?[0-9]*$', token):
            return True
        if 'STRING' in valid and (token.startswith('"') or token.startswith("'")):
            return True

        return False

    def _evaluate_guard(self, guard: str) -> bool:
        """Evaluate a guard condition."""
        # Simple guard evaluation
        ctx = self.context
        try:
            return eval(guard, {'paren_depth': ctx.paren_depth,
                               'bracket_depth': ctx.bracket_depth,
                               'brace_depth': ctx.brace_depth,
                               'indent_level': ctx.indent_level,
                               'in_string': ctx.in_string,
                               'line_start': ctx.line_start})
        except:
            return True

    def _execute_action(self, action: str):
        """Execute a context update action."""
        # Simple action execution
        pass

    def push_state(self):
        """Push current state onto stack (for nested constructs)."""
        self.state_stack.append((self.state, self.context.clone()))

    def pop_state(self) -> bool:
        """Pop state from stack. Returns False if stack empty."""
        if self.state_stack:
            self.state, self.context = self.state_stack.pop()
            return True
        return False

    def reset(self):
        """Reset to initial state."""
        self.state = PythonSyntaxState.MODULE
        self.context = SyntaxContext()
        self.state_stack = []


def get_valid_tokens(statechart: PythonSyntaxStatechart) -> Set[str]:
    """Get valid tokens for current state (convenience function)."""
    return statechart.get_valid_tokens()


# Demo
if __name__ == "__main__":
    print("=" * 60)
    print("PYTHON SYNTAX STATECHART DEMO")
    print("=" * 60)

    chart = PythonSyntaxStatechart()

    # Simulate generating: def foo(x):\n    return x + 1
    tokens = ['def', 'foo', '(', 'x', ')', ':', 'NEWLINE', 'INDENT', 'return', 'x', '+', '1', 'NEWLINE']

    print("\nToken sequence: def foo(x):\\n    return x + 1")
    print("\nProcessing tokens:")

    for token in tokens:
        valid = chart.get_valid_tokens()
        is_valid = token in valid or chart._is_valid_token_type(token)
        print(f"  State: {chart.state.name:20s} | Token: {token:10s} | Valid: {is_valid}")
        chart.transition(token)

    print(f"\nFinal state: {chart.state.name}")
    print(f"Context: indent={chart.context.indent_level}, parens={chart.context.paren_depth}")
