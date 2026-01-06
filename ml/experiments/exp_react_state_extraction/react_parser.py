"""
React Parser - Parse React/JSX code to extract state patterns.

Extracts:
- useState hooks with initial values and setter calls
- useReducer hooks with action types and reducer logic
- useEffect dependencies and state updates
- Event handlers that modify state
"""

import re
from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional, Tuple
from enum import Enum


class HookType(Enum):
    USE_STATE = "useState"
    USE_REDUCER = "useReducer"
    USE_EFFECT = "useEffect"
    USE_CALLBACK = "useCallback"
    USE_MEMO = "useMemo"


@dataclass
class UseStateHook:
    """Represents a useState hook."""
    state_name: str
    setter_name: str
    initial_value: str
    value_type: Optional[str] = None
    line_number: int = 0


@dataclass
class UseReducerHook:
    """Represents a useReducer hook."""
    state_name: str
    dispatch_name: str
    reducer_name: str
    initial_state: str
    action_types: List[str] = field(default_factory=list)
    line_number: int = 0


@dataclass
class StateUpdate:
    """Represents a state update call."""
    setter_or_dispatch: str
    new_value: str
    condition: Optional[str] = None
    in_handler: Optional[str] = None
    line_number: int = 0


@dataclass
class EventHandler:
    """Represents an event handler function."""
    name: str
    event_type: Optional[str] = None
    state_updates: List[StateUpdate] = field(default_factory=list)
    line_number: int = 0


@dataclass
class ReactComponent:
    """Parsed React component."""
    name: str
    use_state_hooks: List[UseStateHook] = field(default_factory=list)
    use_reducer_hooks: List[UseReducerHook] = field(default_factory=list)
    event_handlers: List[EventHandler] = field(default_factory=list)
    state_updates: List[StateUpdate] = field(default_factory=list)
    effects: List[Dict] = field(default_factory=list)


class ReactParser:
    """
    Parse React component code to extract state management patterns.

    Supports both JavaScript and TypeScript React components.
    """

    # Regex patterns for React hooks
    USE_STATE_PATTERN = re.compile(
        r'const\s+\[(\w+),\s*(\w+)\]\s*=\s*useState\s*(?:<[^>]+>)?\s*\(([^)]*)\)',
        re.MULTILINE
    )

    USE_REDUCER_PATTERN = re.compile(
        r'const\s+\[(\w+),\s*(\w+)\]\s*=\s*useReducer\s*\((\w+),\s*([^)]+)\)',
        re.MULTILINE
    )

    # Pattern for reducer function
    REDUCER_PATTERN = re.compile(
        r'(?:const|function)\s+(\w+)\s*=?\s*(?:\([^)]*\)\s*=>|\([^)]*\))\s*\{([^}]+(?:\{[^}]*\}[^}]*)*)\}',
        re.MULTILINE | re.DOTALL
    )

    # Pattern for action types in switch cases
    ACTION_TYPE_PATTERN = re.compile(
        r'case\s+[\'"](\w+)[\'"]',
        re.MULTILINE
    )

    # Pattern for setState calls
    SET_STATE_PATTERN = re.compile(
        r'(\w+)\s*\(([^)]+)\)',
        re.MULTILINE
    )

    # Pattern for event handlers
    EVENT_HANDLER_PATTERN = re.compile(
        r'(?:const|function)\s+(\w+)\s*=?\s*(?:\([^)]*\)\s*=>|\([^)]*\))\s*\{([^}]+(?:\{[^}]*\}[^}]*)*)\}',
        re.MULTILINE | re.DOTALL
    )

    # Pattern for component definition (must start with capital letter for React components)
    COMPONENT_PATTERN = re.compile(
        r'(?:export\s+)?(?:default\s+)?(?:function|const)\s+([A-Z]\w+)\s*[=:]?\s*(?:\([^)]*\)\s*(?::\s*\w+)?\s*=>|\([^)]*\))\s*(?::\s*\w+)?\s*[{]',
        re.MULTILINE
    )

    def __init__(self):
        self.components: List[ReactComponent] = []

    def parse(self, code: str) -> List[ReactComponent]:
        """
        Parse React code and extract components with state patterns.

        Args:
            code: React/JSX source code

        Returns:
            List of parsed ReactComponent objects
        """
        self.components = []
        self._full_code = code  # Store full code for reducer lookup

        # Find component definitions
        component_matches = list(self.COMPONENT_PATTERN.finditer(code))

        if not component_matches:
            # Try to parse as single component
            component = self._parse_component("UnnamedComponent", code, code)
            if component.use_state_hooks or component.use_reducer_hooks:
                self.components.append(component)
        else:
            for match in component_matches:
                component_name = match.group(1)
                # Extract component body (simplified - find matching brace)
                start = match.end() - 1
                body = self._extract_block(code, start)

                component = self._parse_component(component_name, body, code)
                self.components.append(component)

        return self.components

    def _parse_component(self, name: str, code: str, full_code: str = None) -> ReactComponent:
        """Parse a single component's code."""
        component = ReactComponent(name=name)
        full_code = full_code or code

        # Extract useState hooks
        for match in self.USE_STATE_PATTERN.finditer(code):
            hook = UseStateHook(
                state_name=match.group(1),
                setter_name=match.group(2),
                initial_value=match.group(3).strip(),
                line_number=code[:match.start()].count('\n') + 1
            )
            # Infer type from initial value
            hook.value_type = self._infer_type(hook.initial_value)
            component.use_state_hooks.append(hook)

        # Extract useReducer hooks
        for match in self.USE_REDUCER_PATTERN.finditer(code):
            hook = UseReducerHook(
                state_name=match.group(1),
                dispatch_name=match.group(2),
                reducer_name=match.group(3),
                initial_state=match.group(4).strip(),
                line_number=code[:match.start()].count('\n') + 1
            )
            # Find action types in reducer (search full code for reducer definition)
            hook.action_types = self._find_action_types(full_code, hook.reducer_name)
            component.use_reducer_hooks.append(hook)

        # Extract state updates
        setter_names = {h.setter_name for h in component.use_state_hooks}
        dispatch_names = {h.dispatch_name for h in component.use_reducer_hooks}
        all_setters = setter_names | dispatch_names

        for setter in all_setters:
            pattern = re.compile(rf'{re.escape(setter)}\s*\(([^)]+)\)', re.MULTILINE)
            for match in pattern.finditer(code):
                update = StateUpdate(
                    setter_or_dispatch=setter,
                    new_value=match.group(1).strip(),
                    line_number=code[:match.start()].count('\n') + 1
                )
                component.state_updates.append(update)

        # Extract event handlers
        component.event_handlers = self._extract_handlers(code, all_setters)

        return component

    def _extract_block(self, code: str, start: int) -> str:
        """Extract a code block starting from an opening brace."""
        if start >= len(code) or code[start] != '{':
            return ""

        depth = 0
        end = start

        for i in range(start, len(code)):
            if code[i] == '{':
                depth += 1
            elif code[i] == '}':
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break

        return code[start:end]

    def _infer_type(self, value: str) -> str:
        """Infer type from initial value."""
        value = value.strip()

        if value in ('true', 'false'):
            return 'boolean'
        elif value.startswith('"') or value.startswith("'") or value.startswith('`'):
            return 'string'
        elif value.replace('.', '').replace('-', '').isdigit():
            return 'number'
        elif value.startswith('['):
            return 'array'
        elif value.startswith('{'):
            return 'object'
        elif value == 'null':
            return 'null'
        elif value == 'undefined':
            return 'undefined'
        else:
            return 'unknown'

    def _find_action_types(self, code: str, reducer_name: str) -> List[str]:
        """Find action types used in a reducer."""
        action_types = []

        # Find the reducer function using brace matching
        pattern = re.compile(
            rf'(?:const|function)\s+{re.escape(reducer_name)}\s*(?:=\s*)?(?:\([^)]*\)\s*(?:=>)?\s*)?\{{',
            re.MULTILINE
        )

        match = pattern.search(code)
        if match:
            # Find the matching closing brace
            start = match.end() - 1
            reducer_body = self._extract_block(code, start)

            # Extract action types from switch cases
            for action_match in self.ACTION_TYPE_PATTERN.finditer(reducer_body):
                action_types.append(action_match.group(1))

        return action_types

    def _extract_handlers(self, code: str, setters: Set[str]) -> List[EventHandler]:
        """Extract event handler functions that modify state."""
        handlers = []

        # Common handler name patterns
        handler_patterns = [
            r'handle\w+',
            r'on\w+',
            r'\w+Handler',
            r'\w+Click',
            r'\w+Change',
            r'\w+Submit',
        ]

        combined_pattern = '|'.join(handler_patterns)

        # Match arrow function handlers: const handleX = () => { ... }
        handler_regex = re.compile(
            rf'const\s+({combined_pattern})\s*=\s*\([^)]*\)\s*=>\s*\{{',
            re.MULTILINE | re.IGNORECASE
        )

        for match in handler_regex.finditer(code):
            handler_name = match.group(1)
            # Extract handler body using brace matching
            start = match.end() - 1
            handler_body = self._extract_block(code, start)

            handler = EventHandler(
                name=handler_name,
                line_number=code[:match.start()].count('\n') + 1
            )

            # Determine event type from name
            if 'Click' in handler_name or 'click' in handler_name:
                handler.event_type = 'click'
            elif 'Change' in handler_name or 'change' in handler_name:
                handler.event_type = 'change'
            elif 'Submit' in handler_name or 'submit' in handler_name:
                handler.event_type = 'submit'
            elif 'Focus' in handler_name or 'focus' in handler_name:
                handler.event_type = 'focus'
            elif 'Blur' in handler_name or 'blur' in handler_name:
                handler.event_type = 'blur'

            # Find state updates in handler
            for setter in setters:
                pattern = re.compile(rf'{re.escape(setter)}\s*\(([^)]+)\)')
                for update_match in pattern.finditer(handler_body):
                    update = StateUpdate(
                        setter_or_dispatch=setter,
                        new_value=update_match.group(1).strip(),
                        in_handler=handler_name
                    )
                    handler.state_updates.append(update)

            if handler.state_updates:
                handlers.append(handler)

        return handlers

    def get_state_summary(self) -> Dict:
        """Get a summary of extracted state patterns."""
        summary = {
            'components': len(self.components),
            'total_use_state': sum(len(c.use_state_hooks) for c in self.components),
            'total_use_reducer': sum(len(c.use_reducer_hooks) for c in self.components),
            'total_handlers': sum(len(c.event_handlers) for c in self.components),
            'total_updates': sum(len(c.state_updates) for c in self.components),
            'components_detail': []
        }

        for comp in self.components:
            detail = {
                'name': comp.name,
                'useState': [
                    {'state': h.state_name, 'setter': h.setter_name, 'type': h.value_type}
                    for h in comp.use_state_hooks
                ],
                'useReducer': [
                    {'state': h.state_name, 'actions': h.action_types}
                    for h in comp.use_reducer_hooks
                ],
                'handlers': [h.name for h in comp.event_handlers]
            }
            summary['components_detail'].append(detail)

        return summary


def demo():
    """Demonstrate React parser."""
    sample_code = '''
    import React, { useState, useReducer } from 'react';

    const initialState = { count: 0, loading: false };

    function counterReducer(state, action) {
        switch (action.type) {
            case 'INCREMENT':
                return { ...state, count: state.count + 1 };
            case 'DECREMENT':
                return { ...state, count: state.count - 1 };
            case 'SET_LOADING':
                return { ...state, loading: action.payload };
            default:
                return state;
        }
    }

    function Counter() {
        const [isOpen, setIsOpen] = useState(false);
        const [name, setName] = useState("");
        const [state, dispatch] = useReducer(counterReducer, initialState);

        const handleClick = () => {
            setIsOpen(!isOpen);
        };

        const handleIncrement = () => {
            dispatch({ type: 'INCREMENT' });
        };

        const handleNameChange = (e) => {
            setName(e.target.value);
        };

        return (
            <div>
                <button onClick={handleClick}>Toggle</button>
                <button onClick={handleIncrement}>+</button>
            </div>
        );
    }
    '''

    parser = ReactParser()
    components = parser.parse(sample_code)

    print("=" * 60)
    print("REACT PARSER DEMO")
    print("=" * 60)

    summary = parser.get_state_summary()
    print(f"\nComponents found: {summary['components']}")
    print(f"Total useState hooks: {summary['total_use_state']}")
    print(f"Total useReducer hooks: {summary['total_use_reducer']}")
    print(f"Total handlers: {summary['total_handlers']}")

    for detail in summary['components_detail']:
        print(f"\n{detail['name']}:")
        print(f"  useState: {detail['useState']}")
        print(f"  useReducer: {detail['useReducer']}")
        print(f"  handlers: {detail['handlers']}")


if __name__ == "__main__":
    demo()
