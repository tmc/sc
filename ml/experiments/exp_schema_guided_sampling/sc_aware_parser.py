#!/usr/bin/env python3
"""
Statechart-Aware JSON Parser

Extends the JSON parser to understand statechart semantics:
- Recognizes field names (guard, actions, event, from, to, etc.)
- Validates guard expressions syntax
- Validates action expressions syntax
- Enforces statechart schema constraints
"""

import json
import re
from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Any
from pathlib import Path


# =============================================================================
# Expression Grammar
# =============================================================================

# Valid variable names in guards/actions
VALID_VARIABLES = {
    'depth', 'stack', 'element_count', 'max_depth', 'max_elements',
    'string_length', 'max_string_length',
    # User-defined variables are also valid
}

# Valid operators in guards
GUARD_OPERATORS = {'==', '!=', '<', '>', '<=', '>=', '&&', '||'}

# Valid functions in guards
GUARD_FUNCTIONS = {'stack.top', 'stack.empty', 'stack.size'}

# Valid actions
ACTION_PATTERNS = [
    r'\w+\+\+',           # var++
    r'\w+--',             # var--
    r'\w+\s*=\s*.+',      # var = expr
    r'stack\.push\(.+\)', # stack.push(x)
    r'stack\.pop\(\)',    # stack.pop()
]


@dataclass
class ExpressionValidator:
    """Validates guard and action expressions."""

    known_variables: Set[str] = field(default_factory=lambda: VALID_VARIABLES.copy())

    def validate_guard(self, expr: str) -> tuple[bool, str]:
        """
        Validate a guard expression.

        Valid forms:
        - variable < value
        - variable == value
        - stack.top() == 'string'
        - expr && expr
        - expr || expr
        """
        expr = expr.strip()

        if not expr:
            return True, ""

        # Handle compound expressions
        if '&&' in expr:
            parts = expr.split('&&')
            for part in parts:
                valid, err = self.validate_guard(part.strip())
                if not valid:
                    return False, err
            return True, ""

        if '||' in expr:
            parts = expr.split('||')
            for part in parts:
                valid, err = self.validate_guard(part.strip())
                if not valid:
                    return False, err
            return True, ""

        # Check for comparison operators
        for op in ['==', '!=', '<=', '>=', '<', '>']:
            if op in expr:
                left, right = expr.split(op, 1)
                left = left.strip()
                right = right.strip()

                # Validate left side (should be variable or function call)
                if not self._is_valid_lvalue(left):
                    return False, f"Invalid left operand: {left}"

                # Right side can be variable, number, or string
                if not self._is_valid_rvalue(right):
                    return False, f"Invalid right operand: {right}"

                return True, ""

        return False, f"No valid operator found in: {expr}"

    def validate_action(self, expr: str) -> tuple[bool, str]:
        """
        Validate an action expression.

        Valid forms:
        - var++
        - var--
        - var = value
        - stack.push('value')
        - stack.pop()
        """
        expr = expr.strip()

        if not expr:
            return True, ""

        # Handle multiple actions separated by ;
        if ';' in expr:
            parts = expr.split(';')
            for part in parts:
                part = part.strip()
                if part:
                    valid, err = self.validate_action(part)
                    if not valid:
                        return False, err
            return True, ""

        # Check increment/decrement
        if expr.endswith('++') or expr.endswith('--'):
            var = expr[:-2].strip()
            if var.isidentifier():
                return True, ""
            return False, f"Invalid variable for increment: {var}"

        # Check assignment
        if '=' in expr and not any(op in expr for op in ['==', '!=', '<=', '>=']):
            parts = expr.split('=', 1)
            var = parts[0].strip()
            value = parts[1].strip()

            if not var.isidentifier():
                return False, f"Invalid assignment target: {var}"

            return True, ""

        # Check stack operations
        if expr.startswith('stack.push(') and expr.endswith(')'):
            return True, ""

        if expr == 'stack.pop()':
            return True, ""

        return False, f"Invalid action: {expr}"

    def _is_valid_lvalue(self, expr: str) -> bool:
        """Check if expression is a valid left-value."""
        expr = expr.strip()

        # Simple variable
        if expr.isidentifier():
            return True

        # Function call like stack.top()
        if '(' in expr and expr.endswith(')'):
            func_name = expr[:expr.index('(')]
            return func_name in {'stack.top', 'stack.empty', 'stack.size'}

        return False

    def _is_valid_rvalue(self, expr: str) -> bool:
        """Check if expression is a valid right-value."""
        expr = expr.strip()

        # String literal
        if (expr.startswith("'") and expr.endswith("'")) or \
           (expr.startswith('"') and expr.endswith('"')):
            return True

        # Number
        try:
            float(expr)
            return True
        except ValueError:
            pass

        # Variable
        if expr.isidentifier():
            return True

        return False


# =============================================================================
# Schema-Aware State Machine
# =============================================================================

@dataclass
class SchemaContext:
    """Extended state for schema-aware parsing."""
    # JSON structure tracking
    depth: int = 0
    stack: List[str] = field(default_factory=list)

    # Schema context tracking
    current_key: str = ""
    key_stack: List[str] = field(default_factory=list)
    in_transitions: bool = False
    in_states: bool = False

    # Expression tracking
    current_string: str = ""

    # Limits
    max_depth: int = 10
    max_elements: int = 20
    max_string_length: int = 100

    def push_context(self, container: str):
        self.stack.append(container)
        self.key_stack.append(self.current_key)
        self.depth += 1

    def pop_context(self):
        if self.stack:
            self.stack.pop()
        if self.key_stack:
            self.current_key = self.key_stack.pop()
        self.depth -= 1

    def top(self) -> Optional[str]:
        return self.stack[-1] if self.stack else None

    def copy(self) -> "SchemaContext":
        ctx = SchemaContext(
            depth=self.depth,
            current_key=self.current_key,
            in_transitions=self.in_transitions,
            in_states=self.in_states,
            current_string=self.current_string,
            max_depth=self.max_depth,
            max_elements=self.max_elements,
            max_string_length=self.max_string_length,
        )
        ctx.stack = self.stack.copy()
        ctx.key_stack = self.key_stack.copy()
        return ctx


class SchemaAwareParser:
    """
    JSON parser that understands statechart schema.

    Tracks:
    - Which field we're currently in (guard, actions, event, etc.)
    - Validates expressions when inside guard/action fields
    - Enforces valid field names for statechart schema
    """

    # Valid top-level keys
    VALID_ROOT_KEYS = {
        'name', 'description', 'variables', 'root_state', 'transitions',
        'initial_context', 'metadata'
    }

    # Valid transition keys
    VALID_TRANSITION_KEYS = {
        'from', 'to', 'event', 'guard', 'actions', 'description'
    }

    # Valid state keys
    VALID_STATE_KEYS = {
        'label', 'type', 'children', 'is_initial', 'is_final',
        'entry_actions', 'exit_actions', 'history'
    }

    def __init__(self):
        self.context = SchemaContext()
        self.expr_validator = ExpressionValidator()
        self.in_string = False
        self.escape_next = False
        self.errors: List[str] = []

    def reset(self):
        self.context = SchemaContext()
        self.in_string = False
        self.escape_next = False
        self.errors = []

    def get_valid_events(self) -> Set[str]:
        """Get valid next events based on current state and context."""
        ctx = self.context
        events = set()

        if self.in_string:
            # Inside string - allow STRING content or closing QUOTE
            events.add('STRING')
            events.add('QUOTE')

            # Check string length limit
            if len(ctx.current_string) >= ctx.max_string_length:
                events.discard('STRING')  # Force close string

            return events

        container = ctx.top()

        if container == 'object':
            # In object - can have key (QUOTE) or close (RBRACE)
            events.add('QUOTE')  # Start key
            events.add('RBRACE')  # Close object

        elif container == 'array':
            # In array - can have value or close
            events.add('LBRACE')  # Object element
            events.add('LBRACKET')  # Array element
            events.add('QUOTE')  # String element
            events.add('NUMBER')
            events.add('BOOL')
            events.add('NULL')
            events.add('RBRACKET')  # Close array

        elif container == 'key':
            # After key - expect colon
            events.add('COLON')

        elif container == 'value':
            # Expecting value
            events.add('LBRACE')
            events.add('LBRACKET')
            events.add('QUOTE')
            events.add('NUMBER')
            events.add('BOOL')
            events.add('NULL')

        elif container == 'after_value':
            # After value - comma or close
            parent = ctx.stack[-2] if len(ctx.stack) >= 2 else None
            if parent == 'object':
                events.add('COMMA')
                events.add('RBRACE')
            elif parent == 'array':
                events.add('COMMA')
                events.add('RBRACKET')

        elif container is None:
            # At root - expect object or array
            events.add('LBRACE')
            events.add('LBRACKET')

        # Apply depth limit
        if ctx.depth >= ctx.max_depth:
            events.discard('LBRACE')
            events.discard('LBRACKET')

        return events

    def process_event(self, event: str, content: str = "") -> bool:
        """
        Process an event and update state.

        Returns True if valid, False if error.
        """
        ctx = self.context

        if event == 'LBRACE':
            ctx.push_context('object')
            return True

        elif event == 'RBRACE':
            if ctx.top() == 'after_value':
                ctx.pop_context()  # Pop after_value
            ctx.pop_context()  # Pop object
            if ctx.top() == 'value':
                ctx.pop_context()  # Pop value
                ctx.push_context('after_value')
            return True

        elif event == 'LBRACKET':
            ctx.push_context('array')
            return True

        elif event == 'RBRACKET':
            if ctx.top() == 'after_value':
                ctx.pop_context()
            ctx.pop_context()  # Pop array
            if ctx.top() == 'value':
                ctx.pop_context()
                ctx.push_context('after_value')
            return True

        elif event == 'QUOTE':
            if self.in_string:
                # End of string
                self.in_string = False
                string_content = ctx.current_string
                ctx.current_string = ""

                # Validate based on context
                if ctx.top() == 'key':
                    # This was a key - track it
                    ctx.current_key = string_content
                    ctx.pop_context()  # Pop key state
                    ctx.push_context('after_key')

                    # Validate key name based on context
                    return self._validate_key(string_content)

                elif ctx.current_key in ('guard', 'expression'):
                    # This was a guard expression
                    valid, err = self.expr_validator.validate_guard(string_content)
                    if not valid:
                        self.errors.append(f"Invalid guard: {err}")
                        return False
                    ctx.pop_context()
                    ctx.push_context('after_value')

                elif ctx.current_key == 'actions':
                    # This was an action expression
                    valid, err = self.expr_validator.validate_action(string_content)
                    if not valid:
                        self.errors.append(f"Invalid action: {err}")
                        return False

                return True
            else:
                # Start of string
                self.in_string = True
                ctx.current_string = ""

                if ctx.top() == 'object':
                    ctx.push_context('key')
                elif ctx.top() in ('value', 'array'):
                    pass  # String value

                return True

        elif event == 'STRING':
            if self.in_string:
                ctx.current_string += content
                return True
            return False

        elif event == 'COLON':
            if ctx.top() == 'after_key':
                ctx.pop_context()
                ctx.push_context('value')
                return True
            return False

        elif event == 'COMMA':
            if ctx.top() == 'after_value':
                ctx.pop_context()
                return True
            return False

        elif event in ('NUMBER', 'BOOL', 'NULL'):
            if ctx.top() in ('value', 'array'):
                if ctx.top() == 'value':
                    ctx.pop_context()
                ctx.push_context('after_value')
                return True
            return False

        return False

    def _validate_key(self, key: str) -> bool:
        """Validate key name based on current context."""
        ctx = self.context

        # Track special contexts
        if key == 'transitions':
            ctx.in_transitions = True
        elif key == 'children' or key == 'root_state':
            ctx.in_states = True

        # Validate based on nesting level
        if ctx.depth == 1:
            # Root level keys
            if key not in self.VALID_ROOT_KEYS:
                self.errors.append(f"Unknown root key: {key}")
                # Allow it but warn

        elif ctx.in_transitions and ctx.depth == 3:
            # Inside a transition object
            if key not in self.VALID_TRANSITION_KEYS:
                self.errors.append(f"Unknown transition key: {key}")

        elif ctx.in_states:
            # Inside a state object
            if key not in self.VALID_STATE_KEYS and key not in self.VALID_ROOT_KEYS:
                # Could be nested state, allow flexibility
                pass

        return True  # Don't fail on unknown keys, just warn


# =============================================================================
# Demo
# =============================================================================

def demo():
    """Demo the schema-aware parser."""
    print("=" * 70)
    print("SCHEMA-AWARE STATECHART PARSER")
    print("=" * 70)

    # Test expression validation
    validator = ExpressionValidator()

    print("\n--- Guard Expression Validation ---")
    test_guards = [
        "depth < max_depth",
        "depth == 1 && stack.top() == 'object'",
        "element_count >= max_elements",
        "invalid syntax here",
        "stack.top() == 'array'",
    ]

    for guard in test_guards:
        valid, err = validator.validate_guard(guard)
        status = "✓" if valid else f"✗ ({err})"
        print(f"  {guard:<45} {status}")

    print("\n--- Action Expression Validation ---")
    test_actions = [
        "depth++",
        "depth--; stack.pop()",
        "stack.push('object')",
        "element_count = 0",
        "invalid action",
        "depth++; stack.push('array'); element_count = 0",
    ]

    for action in test_actions:
        valid, err = validator.validate_action(action)
        status = "✓" if valid else f"✗ ({err})"
        print(f"  {action:<45} {status}")

    print("\n--- Valid Events by Context ---")
    parser = SchemaAwareParser()

    # Simulate parsing
    contexts = [
        ("At root", None),
        ("In object", lambda: parser.context.push_context('object')),
        ("After key 'guard'", lambda: (
            parser.context.push_context('object'),
            setattr(parser.context, 'current_key', 'guard'),
            parser.context.push_context('value')
        )),
        ("In string (guard expr)", lambda: (
            parser.context.push_context('object'),
            setattr(parser.context, 'current_key', 'guard'),
            setattr(parser, 'in_string', True)
        )),
    ]

    for desc, setup in contexts:
        parser.reset()
        if setup:
            setup()
        events = parser.get_valid_events()
        print(f"  {desc:<30} → {events}")


if __name__ == "__main__":
    demo()
