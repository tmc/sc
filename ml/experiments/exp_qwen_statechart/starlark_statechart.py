"""
Starlark Syntax Statechart

Defines a statechart that tracks Starlark syntax state during generation.
Starlark is a Python dialect used in Bazel, Buck2, and other build systems.

Key differences from Python:
- No classes (only functions and variables)
- No decorators
- No async/await, yield
- No try/except/finally
- No list/dict/set comprehensions with multiple for clauses
- load() statements only at module level
- Stricter function call patterns
- No indentation-based blocks (uses explicit delimiters)

States are simpler than Python's 75-state FSM:
- MODULE: Top-level, expecting statement or definition
- LOAD_STMT: Inside load("...", ...)
- FUNCTION_DEF: After 'def', expecting name
- FUNCTION_PARAMS: Inside (params)
- FUNCTION_BODY: After ':', in function body
- CALL_EXPR: Inside function(...)
- DICT_LITERAL: Inside {...}
- LIST_LITERAL: Inside [...]
- STRING: Inside a string literal
- EXPRESSION: General expression context
"""

from enum import Enum, auto
from dataclasses import dataclass, field
from typing import List, Set, Dict, Optional, Tuple
import re


class StarlarkSyntaxState(Enum):
    """High-level Starlark syntax states (simpler than Python)."""
    MODULE = auto()           # Top-level module
    LOAD_STMT = auto()        # Inside load("...")
    FUNCTION_DEF = auto()     # After 'def', expecting name
    FUNCTION_PARAMS = auto()  # Inside (params)
    FUNCTION_BODY = auto()    # After ':', in function body
    CALL_EXPR = auto()        # Inside function(...)
    CALL_ARGS = auto()        # Inside function call arguments
    DICT_LITERAL = auto()     # Inside {...}
    DICT_KEY = auto()         # Expecting dict key
    DICT_VALUE = auto()       # After '=', expecting value
    LIST_LITERAL = auto()     # Inside [...]
    STRING = auto()           # Inside "..." or '...'
    EXPRESSION = auto()       # General expression
    RETURN_STMT = auto()      # After 'return'
    IF_CONDITION = auto()     # After 'if', expecting condition
    IF_BODY = auto()          # Inside if body
    ELIF_CONDITION = auto()   # After 'elif'
    ELSE_BODY = auto()        # After 'else:'
    FOR_TARGET = auto()       # After 'for', expecting target
    FOR_ITER = auto()         # After 'in', expecting iterable
    FOR_BODY = auto()         # Inside for body
    COMPREHENSION = auto()    # Inside list/dict comprehension
    LAMBDA = auto()           # After 'lambda'
    SUBSCRIPT = auto()        # Inside obj[...]


# Starlark keywords (subset of Python)
KEYWORDS = {
    'def', 'if', 'elif', 'else', 'for', 'in', 'return', 'pass', 'break',
    'continue', 'and', 'or', 'not', 'True', 'False', 'None', 'lambda',
    'load',  # Starlark-specific
}

# Operators
OPERATORS = {
    '+', '-', '*', '/', '//', '%', '**',
    '==', '!=', '<', '>', '<=', '>=',
    '=', '+=', '-=', '*=', '/=', '//=', '%=',
    '&', '|', '^', '~', '<<', '>>',
}

# Delimiters
DELIMITERS = {
    '(', ')', '[', ']', '{', '}',
    ',', ':', '.', '\\',
}

# Common Starlark identifiers (sc DSL)
SC_IDENTIFIERS = {
    'sc', 'machine', 'state', 'transition', 'event', 'guard', 'action',
    'parallel', 'history', 'initial', 'final', 'entry', 'exit',
    'name', 'states', 'transitions', 'to', 'on', 'cond',
}


@dataclass
class StarlarkContext:
    """Tracks syntax context during parsing/generation."""
    paren_depth: int = 0       # ()
    bracket_depth: int = 0     # []
    brace_depth: int = 0       # {}
    in_string: bool = False
    string_char: str = ''      # ' or "
    in_triple_string: bool = False
    function_depth: int = 0    # Nested function calls
    last_token: str = ''
    last_tokens: List[str] = field(default_factory=list)
    at_line_start: bool = True

    def clone(self) -> 'StarlarkContext':
        return StarlarkContext(
            paren_depth=self.paren_depth,
            bracket_depth=self.bracket_depth,
            brace_depth=self.brace_depth,
            in_string=self.in_string,
            string_char=self.string_char,
            in_triple_string=self.in_triple_string,
            function_depth=self.function_depth,
            last_token=self.last_token,
            last_tokens=self.last_tokens.copy(),
            at_line_start=self.at_line_start,
        )

    @property
    def total_depth(self) -> int:
        """Total nesting depth."""
        return self.paren_depth + self.bracket_depth + self.brace_depth


@dataclass
class StateTransition:
    """A transition in the syntax statechart."""
    from_state: StarlarkSyntaxState
    to_state: StarlarkSyntaxState
    trigger: str  # Token or token pattern
    guard: Optional[str] = None  # Condition as string
    action: Optional[str] = None  # Context update


class StarlarkStatechart:
    """
    Statechart for Starlark syntax.

    Tracks the current syntax state and provides:
    1. Valid next tokens for current state
    2. State transitions on token consumption
    3. Context updates (brackets, etc.)

    Simpler than Python statechart - no classes, decorators, try/except.
    """

    def __init__(self):
        self.state = StarlarkSyntaxState.MODULE
        self.context = StarlarkContext()
        self.state_stack: List[Tuple[StarlarkSyntaxState, StarlarkContext]] = []

        # Build transition table
        self.transitions = self._build_transitions()

        # Valid tokens per state
        self.valid_tokens = self._build_valid_tokens()

    def _build_transitions(self) -> Dict[StarlarkSyntaxState, List[StateTransition]]:
        """Build the state transition table."""
        transitions = {}

        # MODULE state transitions
        transitions[StarlarkSyntaxState.MODULE] = [
            StateTransition(StarlarkSyntaxState.MODULE, StarlarkSyntaxState.LOAD_STMT, 'load'),
            StateTransition(StarlarkSyntaxState.MODULE, StarlarkSyntaxState.FUNCTION_DEF, 'def'),
            StateTransition(StarlarkSyntaxState.MODULE, StarlarkSyntaxState.IF_CONDITION, 'if'),
            StateTransition(StarlarkSyntaxState.MODULE, StarlarkSyntaxState.FOR_TARGET, 'for'),
            StateTransition(StarlarkSyntaxState.MODULE, StarlarkSyntaxState.EXPRESSION, 'IDENTIFIER'),
        ]

        # LOAD_STMT transitions
        transitions[StarlarkSyntaxState.LOAD_STMT] = [
            StateTransition(StarlarkSyntaxState.LOAD_STMT, StarlarkSyntaxState.MODULE, ')',
                           guard='paren_depth == 0'),
        ]

        # FUNCTION_DEF transitions
        transitions[StarlarkSyntaxState.FUNCTION_DEF] = [
            StateTransition(StarlarkSyntaxState.FUNCTION_DEF, StarlarkSyntaxState.FUNCTION_PARAMS, '('),
        ]

        # FUNCTION_PARAMS transitions
        transitions[StarlarkSyntaxState.FUNCTION_PARAMS] = [
            StateTransition(StarlarkSyntaxState.FUNCTION_PARAMS, StarlarkSyntaxState.FUNCTION_BODY, '):',
                           guard='paren_depth == 0'),
            StateTransition(StarlarkSyntaxState.FUNCTION_PARAMS, StarlarkSyntaxState.FUNCTION_BODY, ')',
                           guard='paren_depth == 0'),
        ]

        # FUNCTION_BODY transitions
        transitions[StarlarkSyntaxState.FUNCTION_BODY] = [
            StateTransition(StarlarkSyntaxState.FUNCTION_BODY, StarlarkSyntaxState.RETURN_STMT, 'return'),
            StateTransition(StarlarkSyntaxState.FUNCTION_BODY, StarlarkSyntaxState.IF_CONDITION, 'if'),
            StateTransition(StarlarkSyntaxState.FUNCTION_BODY, StarlarkSyntaxState.FOR_TARGET, 'for'),
            StateTransition(StarlarkSyntaxState.FUNCTION_BODY, StarlarkSyntaxState.EXPRESSION, 'IDENTIFIER'),
        ]

        # CALL_EXPR transitions
        transitions[StarlarkSyntaxState.CALL_EXPR] = [
            StateTransition(StarlarkSyntaxState.CALL_EXPR, StarlarkSyntaxState.CALL_ARGS, '('),
        ]

        # CALL_ARGS transitions
        transitions[StarlarkSyntaxState.CALL_ARGS] = [
            StateTransition(StarlarkSyntaxState.CALL_ARGS, StarlarkSyntaxState.DICT_LITERAL, '{'),
            StateTransition(StarlarkSyntaxState.CALL_ARGS, StarlarkSyntaxState.LIST_LITERAL, '['),
        ]

        # DICT_LITERAL transitions
        transitions[StarlarkSyntaxState.DICT_LITERAL] = [
            StateTransition(StarlarkSyntaxState.DICT_LITERAL, StarlarkSyntaxState.DICT_KEY, 'IDENTIFIER'),
            StateTransition(StarlarkSyntaxState.DICT_LITERAL, StarlarkSyntaxState.DICT_KEY, 'STRING'),
        ]

        # DICT_KEY transitions
        transitions[StarlarkSyntaxState.DICT_KEY] = [
            StateTransition(StarlarkSyntaxState.DICT_KEY, StarlarkSyntaxState.DICT_VALUE, ':'),
            StateTransition(StarlarkSyntaxState.DICT_KEY, StarlarkSyntaxState.DICT_VALUE, '='),
        ]

        # IF_CONDITION transitions
        transitions[StarlarkSyntaxState.IF_CONDITION] = [
            StateTransition(StarlarkSyntaxState.IF_CONDITION, StarlarkSyntaxState.IF_BODY, ':'),
        ]

        # IF_BODY transitions
        transitions[StarlarkSyntaxState.IF_BODY] = [
            StateTransition(StarlarkSyntaxState.IF_BODY, StarlarkSyntaxState.ELIF_CONDITION, 'elif'),
            StateTransition(StarlarkSyntaxState.IF_BODY, StarlarkSyntaxState.ELSE_BODY, 'else'),
        ]

        # FOR_TARGET transitions
        transitions[StarlarkSyntaxState.FOR_TARGET] = [
            StateTransition(StarlarkSyntaxState.FOR_TARGET, StarlarkSyntaxState.FOR_ITER, 'in'),
        ]

        # FOR_ITER transitions
        transitions[StarlarkSyntaxState.FOR_ITER] = [
            StateTransition(StarlarkSyntaxState.FOR_ITER, StarlarkSyntaxState.FOR_BODY, ':'),
        ]

        # RETURN_STMT transitions
        transitions[StarlarkSyntaxState.RETURN_STMT] = [
            StateTransition(StarlarkSyntaxState.RETURN_STMT, StarlarkSyntaxState.CALL_EXPR, 'IDENTIFIER'),
            StateTransition(StarlarkSyntaxState.RETURN_STMT, StarlarkSyntaxState.DICT_LITERAL, '{'),
            StateTransition(StarlarkSyntaxState.RETURN_STMT, StarlarkSyntaxState.LIST_LITERAL, '['),
        ]

        return transitions

    def _build_valid_tokens(self) -> Dict[StarlarkSyntaxState, Set[str]]:
        """Build valid token sets per state."""
        valid = {}

        # MODULE: can start any top-level construct
        valid[StarlarkSyntaxState.MODULE] = {
            'def', 'load', 'if', 'for',
            'IDENTIFIER', 'NEWLINE', 'COMMENT',
        }

        # LOAD_STMT: load("module", "symbol", ...)
        valid[StarlarkSyntaxState.LOAD_STMT] = {
            '(', ')', 'STRING', ',', 'IDENTIFIER',
        }

        # FUNCTION_DEF: expecting function name
        valid[StarlarkSyntaxState.FUNCTION_DEF] = {'IDENTIFIER'}

        # FUNCTION_PARAMS: inside parameter list
        valid[StarlarkSyntaxState.FUNCTION_PARAMS] = {
            'IDENTIFIER', ',', '=', '*', '**', ')', ':',
            'True', 'False', 'None', 'NUMBER', 'STRING',
        }

        # FUNCTION_BODY: statements allowed in function
        valid[StarlarkSyntaxState.FUNCTION_BODY] = {
            'return', 'if', 'for', 'pass', 'break', 'continue',
            'IDENTIFIER', 'NEWLINE', 'INDENT', 'DEDENT',
        }

        # CALL_EXPR: function call context
        valid[StarlarkSyntaxState.CALL_EXPR] = {
            '(', '.', 'IDENTIFIER',
        }

        # CALL_ARGS: inside function call arguments
        valid[StarlarkSyntaxState.CALL_ARGS] = {
            'IDENTIFIER', 'NUMBER', 'STRING',
            '(', ')', '[', ']', '{', '}',
            ',', '=', '*', '**',
            'True', 'False', 'None',
            '+', '-', '*', '/', 'lambda',
            '.', ':',
        }

        # DICT_LITERAL: inside {...}
        valid[StarlarkSyntaxState.DICT_LITERAL] = {
            'IDENTIFIER', 'STRING', 'NUMBER',
            ':', '=', ',', '}',
            'True', 'False', 'None',
            '[', '{', '(',
        }

        # DICT_KEY: expecting key
        valid[StarlarkSyntaxState.DICT_KEY] = {
            'IDENTIFIER', 'STRING', ':', '=',
        }

        # DICT_VALUE: expecting value
        valid[StarlarkSyntaxState.DICT_VALUE] = {
            'IDENTIFIER', 'STRING', 'NUMBER',
            'True', 'False', 'None',
            '[', '{', '(',
            ',', '}',
        }

        # LIST_LITERAL: inside [...]
        valid[StarlarkSyntaxState.LIST_LITERAL] = {
            'IDENTIFIER', 'NUMBER', 'STRING',
            '[', ']', '(', ')', '{', '}',
            ',', 'for', 'if', 'in',
            'True', 'False', 'None',
            '+', '-', '*', '/',
        }

        # STRING: inside string literal
        valid[StarlarkSyntaxState.STRING] = {
            'STRING_CONTENT', '"', "'",
        }

        # EXPRESSION: general expression context
        valid[StarlarkSyntaxState.EXPRESSION] = {
            'IDENTIFIER', 'NUMBER', 'STRING',
            '+', '-', '*', '/', '//', '%', '**',
            '(', ')', '[', ']', '{', '}',
            '.', ',', ':',
            '==', '!=', '<', '>', '<=', '>=',
            'and', 'or', 'not', 'in',
            'True', 'False', 'None',
            'lambda', 'if', 'else', 'for',
        }

        # RETURN_STMT: after return keyword
        valid[StarlarkSyntaxState.RETURN_STMT] = {
            'IDENTIFIER', 'NUMBER', 'STRING',
            '(', '[', '{', 'None', 'True', 'False',
            'NEWLINE',
        }

        # IF_CONDITION: expecting boolean expression
        valid[StarlarkSyntaxState.IF_CONDITION] = {
            'IDENTIFIER', 'NUMBER', 'STRING',
            '(', 'not', 'True', 'False', 'None',
            'and', 'or', '==', '!=', '<', '>', '<=', '>=',
            'in', ':',
        }

        # IF_BODY: inside if body
        valid[StarlarkSyntaxState.IF_BODY] = {
            'return', 'if', 'elif', 'else', 'for', 'pass',
            'break', 'continue',
            'IDENTIFIER', 'NEWLINE', 'INDENT', 'DEDENT',
        }

        # ELIF_CONDITION: after elif
        valid[StarlarkSyntaxState.ELIF_CONDITION] = valid[StarlarkSyntaxState.IF_CONDITION].copy()

        # ELSE_BODY: after else:
        valid[StarlarkSyntaxState.ELSE_BODY] = valid[StarlarkSyntaxState.IF_BODY].copy()

        # FOR_TARGET: expecting loop variable
        valid[StarlarkSyntaxState.FOR_TARGET] = {
            'IDENTIFIER', ',', 'in',
        }

        # FOR_ITER: expecting iterable
        valid[StarlarkSyntaxState.FOR_ITER] = {
            'IDENTIFIER', 'NUMBER', 'STRING',
            '(', '[', '{', ':',
            'range',
        }

        # FOR_BODY: inside for body
        valid[StarlarkSyntaxState.FOR_BODY] = {
            'return', 'if', 'for', 'pass', 'break', 'continue',
            'IDENTIFIER', 'NEWLINE', 'INDENT', 'DEDENT',
        }

        # COMPREHENSION: inside list/dict comprehension
        valid[StarlarkSyntaxState.COMPREHENSION] = {
            'IDENTIFIER', 'NUMBER', 'STRING',
            'for', 'in', 'if',
            '(', ')', '[', ']', '{', '}',
            ',', ':',
        }

        # LAMBDA: after lambda keyword
        valid[StarlarkSyntaxState.LAMBDA] = {
            'IDENTIFIER', ',', ':', 'EXPRESSION',
        }

        # SUBSCRIPT: inside obj[...]
        valid[StarlarkSyntaxState.SUBSCRIPT] = {
            'IDENTIFIER', 'NUMBER', 'STRING',
            ':', ']', '+', '-',
        }

        # Fill in remaining states with reasonable defaults
        for state in StarlarkSyntaxState:
            if state not in valid:
                valid[state] = {
                    'IDENTIFIER', 'NUMBER', 'STRING',
                    'NEWLINE',
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
            if self.context.in_triple_string:
                result.add('"""')
                result.add("'''")

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
            self.context.function_depth += 1
        elif token == ')':
            self.context.paren_depth = max(0, self.context.paren_depth - 1)
            self.context.function_depth = max(0, self.context.function_depth - 1)
        elif token == '[':
            self.context.bracket_depth += 1
        elif token == ']':
            self.context.bracket_depth = max(0, self.context.bracket_depth - 1)
        elif token == '{':
            self.context.brace_depth += 1
        elif token == '}':
            self.context.brace_depth = max(0, self.context.brace_depth - 1)
        elif token == 'NEWLINE' or token == '\n':
            self.context.at_line_start = True
        elif token in ('"""', "'''"):
            if not self.context.in_string:
                self.context.in_string = True
                self.context.in_triple_string = True
                self.context.string_char = token
            elif self.context.string_char == token:
                self.context.in_string = False
                self.context.in_triple_string = False
                self.context.string_char = ''
        elif token in ('"', "'"):
            if not self.context.in_string:
                self.context.in_string = True
                self.context.string_char = token
            elif self.context.string_char == token and not self.context.in_triple_string:
                self.context.in_string = False
                self.context.string_char = ''
        else:
            self.context.at_line_start = False

    def _matches_trigger(self, token: str, trigger: str) -> bool:
        """Check if token matches trigger pattern."""
        if trigger == token:
            return True
        if trigger == 'IDENTIFIER' and re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', token):
            if token not in KEYWORDS:
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
        ctx = self.context
        try:
            return eval(guard, {
                'paren_depth': ctx.paren_depth,
                'bracket_depth': ctx.bracket_depth,
                'brace_depth': ctx.brace_depth,
                'function_depth': ctx.function_depth,
                'in_string': ctx.in_string,
                'at_line_start': ctx.at_line_start,
            })
        except Exception:
            return True

    def _execute_action(self, action: str):
        """Execute a context update action."""
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
        self.state = StarlarkSyntaxState.MODULE
        self.context = StarlarkContext()
        self.state_stack = []

    def parse_to_traversal(self, code: str) -> List[Tuple[StarlarkSyntaxState, str]]:
        """
        Parse code into a list of (state, token) pairs.

        This represents the code as a statechart traversal, useful for
        structure-preserving mutations.
        """
        traversal = []
        tokens = self._tokenize(code)

        self.reset()
        for token in tokens:
            traversal.append((self.state, token))
            self.transition(token)

        return traversal

    def _tokenize(self, code: str) -> List[str]:
        """Simple tokenizer for Starlark code."""
        tokens = []
        i = 0
        while i < len(code):
            # Skip whitespace (but track newlines)
            if code[i] == '\n':
                tokens.append('NEWLINE')
                i += 1
                continue
            if code[i] in ' \t':
                i += 1
                continue

            # Triple-quoted strings
            if code[i:i+3] in ('"""', "'''"):
                quote = code[i:i+3]
                j = i + 3
                while j < len(code) and code[j:j+3] != quote:
                    j += 1
                tokens.append(code[i:j+3])
                i = j + 3
                continue

            # Single/double quoted strings
            if code[i] in '"\'':
                quote = code[i]
                j = i + 1
                while j < len(code) and code[j] != quote:
                    if code[j] == '\\':
                        j += 2
                    else:
                        j += 1
                tokens.append(code[i:j+1])
                i = j + 1
                continue

            # Comments
            if code[i] == '#':
                j = i
                while j < len(code) and code[j] != '\n':
                    j += 1
                tokens.append(code[i:j])
                i = j
                continue

            # Multi-character operators
            for op in ['==', '!=', '<=', '>=', '//', '**', '+=', '-=', '*=', '/=']:
                if code[i:i+2] == op:
                    tokens.append(op)
                    i += 2
                    break
            else:
                # Single-character tokens
                if code[i] in '()[]{}:,=+-*/<>.':
                    tokens.append(code[i])
                    i += 1
                # Identifiers and numbers
                elif code[i].isalpha() or code[i] == '_':
                    j = i
                    while j < len(code) and (code[j].isalnum() or code[j] == '_'):
                        j += 1
                    tokens.append(code[i:j])
                    i = j
                elif code[i].isdigit():
                    j = i
                    while j < len(code) and (code[j].isdigit() or code[j] == '.'):
                        j += 1
                    tokens.append(code[i:j])
                    i = j
                else:
                    i += 1

        return tokens


def get_valid_tokens(statechart: StarlarkStatechart) -> Set[str]:
    """Get valid tokens for current state (convenience function)."""
    return statechart.get_valid_tokens()


# Demo
if __name__ == "__main__":
    print("=" * 60)
    print("STARLARK SYNTAX STATECHART DEMO")
    print("=" * 60)

    chart = StarlarkStatechart()

    # Example Starlark code (statechart DSL)
    example_code = '''def traffic_light():
    return sc.machine(
        name = "traffic_light",
        states = [
            sc.state("green", initial=True),
            sc.state("yellow"),
            sc.state("red"),
        ],
    )'''

    print(f"\nExample code:\n{example_code}")
    print("\n" + "=" * 60)
    print("Token-by-token processing:")

    traversal = chart.parse_to_traversal(example_code)
    for state, token in traversal[:30]:  # First 30 tokens
        print(f"  State: {state.name:20s} | Token: {token}")

    print(f"\n... ({len(traversal)} tokens total)")
    print(f"Final state: {chart.state.name}")
    print(f"Context: parens={chart.context.paren_depth}, brackets={chart.context.bracket_depth}, braces={chart.context.brace_depth}")
