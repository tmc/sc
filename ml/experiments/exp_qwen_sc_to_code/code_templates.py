"""
Code Templates for Statechart Code Generation.

Provides structured templates that guide LLM code generation.
Templates ensure:
1. Correct structure (package, imports, types)
2. Required methods (transition, enter, exit)
3. Compilable output by design

The LLM fills in the template slots, reducing the chance of
structural errors that prevent compilation.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
import json


@dataclass
class TemplateConfig:
    """Configuration for code templates."""
    package_name: str = "statemachine"
    class_name: str = "StateMachine"
    include_comments: bool = True
    include_logging: bool = False
    generate_tests: bool = True


@dataclass
class GoTemplate:
    """Go code template for statecharts."""
    package_name: str
    struct_name: str
    states: List[str]
    events: List[str]
    transitions: List[Dict]
    initial_state: str

    def render_header(self) -> str:
        """Render package and imports."""
        return f'''package {self.package_name}

import (
\t"fmt"
\t"sync"
)
'''

    def render_state_type(self) -> str:
        """Render state type definition."""
        lines = ["// State represents the possible states"]
        lines.append("type State int")
        lines.append("")
        lines.append("const (")

        for i, state in enumerate(self.states):
            if i == 0:
                lines.append(f"\tState{state} State = iota")
            else:
                lines.append(f"\tState{state}")

        lines.append(")")
        lines.append("")

        # String method
        lines.append("func (s State) String() string {")
        lines.append("\tswitch s {")
        for state in self.states:
            lines.append(f'\tcase State{state}:')
            lines.append(f'\t\treturn "{state}"')
        lines.append("\tdefault:")
        lines.append('\t\treturn "Unknown"')
        lines.append("\t}")
        lines.append("}")

        return "\n".join(lines)

    def render_event_type(self) -> str:
        """Render event type definition."""
        lines = ["// Event represents the possible events"]
        lines.append("type Event int")
        lines.append("")
        lines.append("const (")

        for i, event in enumerate(self.events):
            if i == 0:
                lines.append(f"\tEvent{event} Event = iota")
            else:
                lines.append(f"\tEvent{event}")

        lines.append(")")
        lines.append("")

        # String method
        lines.append("func (e Event) String() string {")
        lines.append("\tswitch e {")
        for event in self.events:
            lines.append(f'\tcase Event{event}:')
            lines.append(f'\t\treturn "{event}"')
        lines.append("\tdefault:")
        lines.append('\t\treturn "Unknown"')
        lines.append("\t}")
        lines.append("}")

        return "\n".join(lines)

    def render_struct(self) -> str:
        """Render main state machine struct."""
        return f'''// {self.struct_name} implements a state machine
type {self.struct_name} struct {{
\tmu           sync.RWMutex
\tcurrentState State
\tonEnter      map[State]func()
\tonExit       map[State]func()
}}

// New{self.struct_name} creates a new state machine
func New{self.struct_name}() *{self.struct_name} {{
\treturn &{self.struct_name}{{
\t\tcurrentState: State{self.initial_state},
\t\tonEnter:      make(map[State]func()),
\t\tonExit:       make(map[State]func()),
\t}}
}}

// CurrentState returns the current state
func (sm *{self.struct_name}) CurrentState() State {{
\tsm.mu.RLock()
\tdefer sm.mu.RUnlock()
\treturn sm.currentState
}}
'''

    def render_transition_method(self) -> str:
        """Render the main transition method."""
        lines = ["// Send processes an event and transitions if valid"]
        lines.append(f"func (sm *{self.struct_name}) Send(event Event) error {{")
        lines.append("\tsm.mu.Lock()")
        lines.append("\tdefer sm.mu.Unlock()")
        lines.append("")
        lines.append("\tvar nextState State")
        lines.append("\tvar valid bool")
        lines.append("")
        lines.append("\tswitch sm.currentState {")

        # Group transitions by source state
        by_source: Dict[str, List[Dict]] = {}
        for t in self.transitions:
            src = t.get('from', t.get('source', ''))
            if src not in by_source:
                by_source[src] = []
            by_source[src].append(t)

        for state in self.states:
            lines.append(f"\tcase State{state}:")
            if state in by_source:
                lines.append("\t\tswitch event {")
                for t in by_source[state]:
                    event = t.get('event', '')
                    target = t.get('to', t.get('target', ''))
                    lines.append(f"\t\tcase Event{event}:")
                    lines.append(f"\t\t\tnextState = State{target}")
                    lines.append("\t\t\tvalid = true")
                lines.append("\t\t}")
            else:
                lines.append("\t\t// No transitions from this state")

        lines.append("\t}")
        lines.append("")
        lines.append("\tif !valid {")
        lines.append('\t\treturn fmt.Errorf("invalid transition from %s on %s", sm.currentState, event)')
        lines.append("\t}")
        lines.append("")
        lines.append("\t// Exit current state")
        lines.append("\tif fn, ok := sm.onExit[sm.currentState]; ok {")
        lines.append("\t\tfn()")
        lines.append("\t}")
        lines.append("")
        lines.append("\t// Enter new state")
        lines.append("\tsm.currentState = nextState")
        lines.append("\tif fn, ok := sm.onEnter[nextState]; ok {")
        lines.append("\t\tfn()")
        lines.append("\t}")
        lines.append("")
        lines.append("\treturn nil")
        lines.append("}")

        return "\n".join(lines)

    def render(self) -> str:
        """Render complete Go code."""
        parts = [
            self.render_header(),
            self.render_state_type(),
            "",
            self.render_event_type(),
            "",
            self.render_struct(),
            "",
            self.render_transition_method(),
        ]
        return "\n".join(parts)


@dataclass
class PythonTemplate:
    """Python code template for statecharts."""
    class_name: str
    states: List[str]
    events: List[str]
    transitions: List[Dict]
    initial_state: str

    def render_imports(self) -> str:
        """Render imports."""
        return '''from enum import Enum, auto
from typing import Optional, Callable, Dict
from dataclasses import dataclass, field
'''

    def render_state_enum(self) -> str:
        """Render state enum."""
        lines = ["class State(Enum):"]
        lines.append('    """Possible states."""')
        for state in self.states:
            lines.append(f"    {state} = auto()")
        return "\n".join(lines)

    def render_event_enum(self) -> str:
        """Render event enum."""
        lines = ["class Event(Enum):"]
        lines.append('    """Possible events."""')
        for event in self.events:
            lines.append(f"    {event} = auto()")
        return "\n".join(lines)

    def render_class(self) -> str:
        """Render main class."""
        lines = [f"class {self.class_name}:"]
        lines.append(f'    """State machine implementation."""')
        lines.append("")
        lines.append("    def __init__(self):")
        lines.append(f"        self._state = State.{self.initial_state}")
        lines.append("        self._on_enter: Dict[State, Callable] = {}")
        lines.append("        self._on_exit: Dict[State, Callable] = {}")
        lines.append("        self._transitions = self._build_transitions()")
        lines.append("")

        # Property
        lines.append("    @property")
        lines.append("    def state(self) -> State:")
        lines.append('        """Current state."""')
        lines.append("        return self._state")
        lines.append("")

        # Build transitions
        lines.append("    def _build_transitions(self) -> Dict:")
        lines.append('        """Build transition table."""')
        lines.append("        return {")

        for t in self.transitions:
            src = t.get('from', t.get('source', ''))
            event = t.get('event', '')
            target = t.get('to', t.get('target', ''))
            lines.append(f"            (State.{src}, Event.{event}): State.{target},")

        lines.append("        }")
        lines.append("")

        # Send method
        lines.append("    def send(self, event: Event) -> bool:")
        lines.append('        """Process an event, return True if transition occurred."""')
        lines.append("        key = (self._state, event)")
        lines.append("        if key not in self._transitions:")
        lines.append("            return False")
        lines.append("")
        lines.append("        # Exit current state")
        lines.append("        if self._state in self._on_exit:")
        lines.append("            self._on_exit[self._state]()")
        lines.append("")
        lines.append("        # Transition")
        lines.append("        self._state = self._transitions[key]")
        lines.append("")
        lines.append("        # Enter new state")
        lines.append("        if self._state in self._on_enter:")
        lines.append("            self._on_enter[self._state]()")
        lines.append("")
        lines.append("        return True")
        lines.append("")

        # Callbacks
        lines.append("    def on_enter(self, state: State, callback: Callable):")
        lines.append('        """Register enter callback."""')
        lines.append("        self._on_enter[state] = callback")
        lines.append("")
        lines.append("    def on_exit(self, state: State, callback: Callable):")
        lines.append('        """Register exit callback."""')
        lines.append("        self._on_exit[state] = callback")

        return "\n".join(lines)

    def render(self) -> str:
        """Render complete Python code."""
        parts = [
            self.render_imports(),
            "",
            self.render_state_enum(),
            "",
            self.render_event_enum(),
            "",
            self.render_class(),
        ]
        return "\n".join(parts)


def build_go_template(
    statechart_json: Dict,
    config: TemplateConfig = None
) -> GoTemplate:
    """Build Go template from statechart JSON."""
    config = config or TemplateConfig()

    # Extract states
    states = []
    root = statechart_json.get('root_state', {})
    _extract_states(root, states)

    # Extract events and transitions
    events = set()
    transitions = []
    for t in statechart_json.get('transitions', []):
        event = t.get('event', 'Unknown')
        events.add(event)
        transitions.append({
            'from': t.get('from', [''])[0] if isinstance(t.get('from'), list) else t.get('from', ''),
            'to': t.get('to', [''])[0] if isinstance(t.get('to'), list) else t.get('to', ''),
            'event': event,
        })

    # Find initial state
    initial = _find_initial(root)

    return GoTemplate(
        package_name=config.package_name,
        struct_name=config.class_name,
        states=states,
        events=sorted(events),
        transitions=transitions,
        initial_state=initial or (states[0] if states else 'Unknown'),
    )


def build_python_template(
    statechart_json: Dict,
    config: TemplateConfig = None
) -> PythonTemplate:
    """Build Python template from statechart JSON."""
    config = config or TemplateConfig()

    # Extract states
    states = []
    root = statechart_json.get('root_state', {})
    _extract_states(root, states)

    # Extract events and transitions
    events = set()
    transitions = []
    for t in statechart_json.get('transitions', []):
        event = t.get('event', 'Unknown')
        events.add(event)
        transitions.append({
            'from': t.get('from', [''])[0] if isinstance(t.get('from'), list) else t.get('from', ''),
            'to': t.get('to', [''])[0] if isinstance(t.get('to'), list) else t.get('to', ''),
            'event': event,
        })

    # Find initial state
    initial = _find_initial(root)

    return PythonTemplate(
        class_name=config.class_name,
        states=states,
        events=sorted(events),
        transitions=transitions,
        initial_state=initial or (states[0] if states else 'Unknown'),
    )


def _extract_states(node: Dict, states: List[str], prefix: str = ""):
    """Recursively extract state labels."""
    label = node.get('label', '')
    if label and label != '__root__':
        states.append(label)

    for child in node.get('children', []):
        _extract_states(child, states, prefix)


def _find_initial(node: Dict) -> Optional[str]:
    """Find initial state."""
    if node.get('is_initial'):
        return node.get('label', '')

    for child in node.get('children', []):
        result = _find_initial(child)
        if result:
            return result

    # Return first child if no explicit initial
    children = node.get('children', [])
    if children:
        return children[0].get('label', '')

    return None


def test_templates():
    """Test code templates."""
    print("=" * 60)
    print("Testing Code Templates")
    print("=" * 60)

    # Sample statechart
    statechart = {
        "root_state": {
            "label": "__root__",
            "type": 2,
            "children": [
                {"label": "Off", "type": 1, "is_initial": True},
                {"label": "On", "type": 1},
            ]
        },
        "transitions": [
            {"from": ["Off"], "to": ["On"], "event": "TURN_ON"},
            {"from": ["On"], "to": ["Off"], "event": "TURN_OFF"},
        ]
    }

    print("\n1. Go Template:")
    go_template = build_go_template(statechart)
    go_code = go_template.render()
    print(go_code[:500] + "...")

    print("\n2. Python Template:")
    py_template = build_python_template(statechart)
    py_code = py_template.render()
    print(py_code[:500] + "...")

    print("\n3. Verifying Python syntax:")
    import ast
    try:
        ast.parse(py_code)
        print("  Python syntax: VALID")
    except SyntaxError as e:
        print(f"  Python syntax: INVALID - {e}")

    print("\n" + "=" * 60)
    print("Template tests complete!")
    print("=" * 60)


if __name__ == "__main__":
    test_templates()
