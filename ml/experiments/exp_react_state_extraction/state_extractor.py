"""
State Extractor - Extract state machines from parsed React components.

Analyzes useState/useReducer patterns to identify:
- State variables and their possible values
- State transitions triggered by events
- Guards/conditions on transitions
"""

from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional, Tuple
from enum import Enum

from .react_parser import (
    ReactComponent, UseStateHook, UseReducerHook,
    StateUpdate, EventHandler
)


class StateType(Enum):
    BOOLEAN = "boolean"
    ENUM = "enum"
    NUMERIC = "numeric"
    STRING = "string"
    OBJECT = "object"
    ARRAY = "array"


@dataclass
class StateVariable:
    """A state variable extracted from React hooks."""
    name: str
    setter: str
    state_type: StateType
    possible_values: List[str] = field(default_factory=list)
    initial_value: str = ""
    is_reducer: bool = False


@dataclass
class StateTransition:
    """A state transition in the extracted machine."""
    from_state: str
    to_state: str
    event: str
    guard: Optional[str] = None
    action: Optional[str] = None
    source_line: int = 0


@dataclass
class ExtractedStateMachine:
    """A state machine extracted from a React component."""
    name: str
    variables: List[StateVariable] = field(default_factory=list)
    states: List[str] = field(default_factory=list)
    transitions: List[StateTransition] = field(default_factory=list)
    initial_state: str = ""
    events: List[str] = field(default_factory=list)


class StateExtractor:
    """
    Extract state machines from parsed React components.

    Analyzes state updates to infer state transitions and builds
    a formal state machine representation.
    """

    def __init__(self):
        self.machines: List[ExtractedStateMachine] = []

    def extract(self, component: ReactComponent) -> ExtractedStateMachine:
        """
        Extract a state machine from a React component.

        Args:
            component: Parsed ReactComponent

        Returns:
            ExtractedStateMachine representing the component's state
        """
        machine = ExtractedStateMachine(name=component.name)

        # Extract state variables from hooks
        for hook in component.use_state_hooks:
            var = self._extract_state_variable(hook)
            machine.variables.append(var)

        for hook in component.use_reducer_hooks:
            var = self._extract_reducer_variable(hook)
            machine.variables.append(var)

        # Build states from boolean variables (most common pattern)
        machine.states, machine.initial_state = self._build_states(machine.variables)

        # Extract transitions from state updates
        machine.transitions = self._extract_transitions(
            component, machine.variables, machine.states
        )

        # Collect unique events
        machine.events = list(set(t.event for t in machine.transitions))

        self.machines.append(machine)
        return machine

    def _extract_state_variable(self, hook: UseStateHook) -> StateVariable:
        """Extract a StateVariable from a useState hook."""
        state_type = self._infer_state_type(hook.value_type, hook.initial_value)
        possible_values = self._infer_possible_values(state_type, hook.initial_value)

        return StateVariable(
            name=hook.state_name,
            setter=hook.setter_name,
            state_type=state_type,
            possible_values=possible_values,
            initial_value=hook.initial_value,
            is_reducer=False
        )

    def _extract_reducer_variable(self, hook: UseReducerHook) -> StateVariable:
        """Extract a StateVariable from a useReducer hook."""
        # Reducers typically manage object state
        return StateVariable(
            name=hook.state_name,
            setter=hook.dispatch_name,
            state_type=StateType.OBJECT,
            possible_values=hook.action_types,  # Actions are "transitions"
            initial_value=hook.initial_state,
            is_reducer=True
        )

    def _infer_state_type(self, value_type: Optional[str], initial: str) -> StateType:
        """Infer the state type from the value type and initial value."""
        if value_type == 'boolean':
            return StateType.BOOLEAN
        elif value_type == 'number':
            return StateType.NUMERIC
        elif value_type == 'string':
            return StateType.STRING
        elif value_type == 'array':
            return StateType.ARRAY
        elif value_type == 'object':
            return StateType.OBJECT
        else:
            # Try to infer from initial value
            initial = initial.strip()
            if initial in ('true', 'false'):
                return StateType.BOOLEAN
            elif initial.replace('.', '').replace('-', '').isdigit():
                return StateType.NUMERIC
            elif initial.startswith('"') or initial.startswith("'"):
                return StateType.STRING
            elif initial.startswith('['):
                return StateType.ARRAY
            elif initial.startswith('{'):
                return StateType.OBJECT
            return StateType.STRING

    def _infer_possible_values(self, state_type: StateType, initial: str) -> List[str]:
        """Infer possible values for a state variable."""
        if state_type == StateType.BOOLEAN:
            return ['true', 'false']
        elif state_type == StateType.ENUM:
            # Would need more context to determine enum values
            return [initial]
        else:
            return [initial]

    def _build_states(
        self, variables: List[StateVariable]
    ) -> Tuple[List[str], str]:
        """
        Build state names from variables.

        For boolean variables, creates states like "isOpen_true", "isOpen_false".
        For reducers, uses action types as transitions between states.
        """
        states = []
        initial_state = ""

        # Create states for ALL boolean variables
        boolean_vars = [v for v in variables if v.state_type == StateType.BOOLEAN]

        for var in boolean_vars:
            states.extend([f"{var.name}_true", f"{var.name}_false"])
            if not initial_state:
                initial_state = f"{var.name}_{var.initial_value}"

        # If no boolean vars, create default states
        if not boolean_vars:
            states = ["idle", "active"]
            initial_state = "idle"

        # Add states from reducer action types
        for var in variables:
            if var.is_reducer:
                for action in var.possible_values:
                    state_name = f"after_{action.lower()}"
                    if state_name not in states:
                        states.append(state_name)

        return states, initial_state

    def _extract_transitions(
        self,
        component: ReactComponent,
        variables: List[StateVariable],
        states: List[str]
    ) -> List[StateTransition]:
        """Extract transitions from state updates and event handlers."""
        transitions = []

        # Map setters to variables
        setter_to_var = {v.setter: v for v in variables}

        # Extract from event handlers
        for handler in component.event_handlers:
            event_name = self._handler_to_event(handler.name)

            for update in handler.state_updates:
                var = setter_to_var.get(update.setter_or_dispatch)
                if not var:
                    continue

                if var.state_type == StateType.BOOLEAN:
                    # Boolean toggle pattern
                    trans = self._create_boolean_transition(
                        var, update, event_name
                    )
                    if trans:
                        transitions.append(trans)

                elif var.is_reducer:
                    # Reducer dispatch pattern
                    trans = self._create_reducer_transition(
                        var, update, event_name, states
                    )
                    if trans:
                        transitions.append(trans)

        # Also check direct state updates not in handlers
        for update in component.state_updates:
            if update.in_handler:
                continue  # Already processed via handlers

            var = setter_to_var.get(update.setter_or_dispatch)
            if var and var.state_type == StateType.BOOLEAN:
                trans = self._create_boolean_transition(
                    var, update, "DIRECT_UPDATE"
                )
                if trans:
                    transitions.append(trans)

        return transitions

    def _handler_to_event(self, handler_name: str) -> str:
        """Convert handler name to event name."""
        # handleClick -> CLICK
        # onSubmit -> SUBMIT
        name = handler_name
        for prefix in ['handle', 'on']:
            if name.lower().startswith(prefix):
                name = name[len(prefix):]
                break
        return name.upper()

    def _create_boolean_transition(
        self,
        var: StateVariable,
        update: StateUpdate,
        event: str
    ) -> Optional[StateTransition]:
        """Create a transition for a boolean state update."""
        new_value = update.new_value.strip()

        if new_value == 'true':
            return StateTransition(
                from_state=f"{var.name}_false",
                to_state=f"{var.name}_true",
                event=event,
                source_line=update.line_number
            )
        elif new_value == 'false':
            return StateTransition(
                from_state=f"{var.name}_true",
                to_state=f"{var.name}_false",
                event=event,
                source_line=update.line_number
            )
        elif '!' in new_value or 'toggle' in new_value.lower():
            # Toggle pattern: setIsOpen(!isOpen)
            # Creates two transitions
            return StateTransition(
                from_state=f"{var.name}_false",
                to_state=f"{var.name}_true",
                event=event,
                guard=f"{var.name} == false",
                source_line=update.line_number
            )

        return None

    def _create_reducer_transition(
        self,
        var: StateVariable,
        update: StateUpdate,
        event: str,
        states: List[str]
    ) -> Optional[StateTransition]:
        """Create a transition for a reducer dispatch."""
        # Parse dispatch({ type: 'ACTION' })
        import re
        action_match = re.search(r"type:\s*['\"](\w+)['\"]", update.new_value)

        if action_match:
            action = action_match.group(1)
            target_state = f"after_{action.lower()}"

            # Find source state (simplified: from idle or any state)
            from_state = "idle" if "idle" in states else states[0] if states else "unknown"

            return StateTransition(
                from_state=from_state,
                to_state=target_state,
                event=event,
                action=f"dispatch({action})",
                source_line=update.line_number
            )

        return None

    def get_summary(self) -> Dict:
        """Get a summary of all extracted machines."""
        return {
            'total_machines': len(self.machines),
            'machines': [
                {
                    'name': m.name,
                    'states': len(m.states),
                    'transitions': len(m.transitions),
                    'events': len(m.events),
                    'variables': [v.name for v in m.variables]
                }
                for m in self.machines
            ]
        }


def demo():
    """Demonstrate state extraction."""
    from .react_parser import ReactParser

    sample_code = '''
    function Counter() {
        const [isOpen, setIsOpen] = useState(false);
        const [count, setCount] = useState(0);
        const [state, dispatch] = useReducer(counterReducer, initialState);

        const handleClick = () => {
            setIsOpen(!isOpen);
        };

        const handleIncrement = () => {
            dispatch({ type: 'INCREMENT' });
        };

        const handleDecrement = () => {
            dispatch({ type: 'DECREMENT' });
        };

        const handleClose = () => {
            setIsOpen(false);
        };

        return <div />;
    }
    '''

    # Parse React code
    parser = ReactParser()
    components = parser.parse(sample_code)

    # Extract state machines
    extractor = StateExtractor()

    print("=" * 60)
    print("STATE EXTRACTOR DEMO")
    print("=" * 60)

    for component in components:
        machine = extractor.extract(component)

        print(f"\nComponent: {machine.name}")
        print(f"  Variables: {[v.name for v in machine.variables]}")
        print(f"  States: {machine.states}")
        print(f"  Initial: {machine.initial_state}")
        print(f"  Events: {machine.events}")
        print(f"  Transitions:")
        for t in machine.transitions:
            guard_str = f" [{t.guard}]" if t.guard else ""
            action_str = f" / {t.action}" if t.action else ""
            print(f"    {t.from_state} --{t.event}{guard_str}--> {t.to_state}{action_str}")


if __name__ == "__main__":
    demo()
