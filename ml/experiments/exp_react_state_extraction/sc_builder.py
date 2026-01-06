"""
SC Builder - Build statechart proto from extracted React state machines.

Converts ExtractedStateMachine objects into the sc proto format
compatible with the sc tooling ecosystem.
"""

import json
from dataclasses import asdict
from typing import List, Dict, Optional

from .state_extractor import ExtractedStateMachine, StateTransition


class SCBuilder:
    """
    Build statechart proto definitions from extracted state machines.

    Outputs JSON format compatible with sc proto schema.
    """

    # State types from proto
    STATE_TYPE_UNSPECIFIED = 0
    STATE_TYPE_BASIC = 1
    STATE_TYPE_NORMAL = 2  # OR-state (XOR decomposition)
    STATE_TYPE_PARALLEL = 3  # AND-state (orthogonal regions)

    def __init__(self):
        self.statecharts: List[Dict] = []

    def build(self, machine: ExtractedStateMachine) -> Dict:
        """
        Build a statechart proto from an extracted state machine.

        Args:
            machine: ExtractedStateMachine from StateExtractor

        Returns:
            Dictionary in sc proto format
        """
        # Build child states
        children = []
        for i, state in enumerate(machine.states):
            child = {
                "label": state,
                "type": self.STATE_TYPE_BASIC,
                "is_initial": state == machine.initial_state
            }
            children.append(child)

        # Build root state
        root_state = {
            "label": "__root__",
            "type": self.STATE_TYPE_NORMAL,
            "children": children
        }

        # Build transitions
        transitions = []
        for trans in machine.transitions:
            t = {
                "from": [trans.from_state],
                "to": [trans.to_state],
                "event": trans.event
            }

            # Add guard if present
            if trans.guard:
                t["guard"] = {
                    "expression": trans.guard
                }

            # Add action if present
            if trans.action:
                t["actions"] = [{"name": trans.action}]

            transitions.append(t)

        # Build events
        events = []
        for event in machine.events:
            events.append({
                "name": event,
                "description": f"Event from React handler"
            })

        # Assemble statechart
        statechart = {
            "root_state": root_state,
            "transitions": transitions,
            "events": events,
            "metadata": {
                "source": "react_component",
                "component_name": machine.name,
                "variables": [v.name for v in machine.variables]
            }
        }

        self.statecharts.append(statechart)
        return statechart

    def build_hierarchical(self, machine: ExtractedStateMachine) -> Dict:
        """
        Build a hierarchical statechart with nested states.

        Groups related states under composite parent states.
        """
        # Group states by prefix (e.g., isOpen_true, isOpen_false -> isOpen)
        state_groups: Dict[str, List[str]] = {}

        for state in machine.states:
            if '_' in state:
                prefix = state.rsplit('_', 1)[0]
            else:
                prefix = state

            if prefix not in state_groups:
                state_groups[prefix] = []
            state_groups[prefix].append(state)

        # Build nested structure
        children = []

        for group_name, group_states in state_groups.items():
            if len(group_states) > 1:
                # Create composite state
                group_children = []
                for state in group_states:
                    is_initial = state == machine.initial_state
                    group_children.append({
                        "label": state,
                        "type": self.STATE_TYPE_BASIC,
                        "is_initial": is_initial
                    })

                composite = {
                    "label": group_name,
                    "type": self.STATE_TYPE_NORMAL,
                    "children": group_children,
                    "is_initial": any(
                        s == machine.initial_state for s in group_states
                    )
                }
                children.append(composite)
            else:
                # Single state, add directly
                state = group_states[0]
                children.append({
                    "label": state,
                    "type": self.STATE_TYPE_BASIC,
                    "is_initial": state == machine.initial_state
                })

        root_state = {
            "label": "__root__",
            "type": self.STATE_TYPE_NORMAL,
            "children": children
        }

        # Build transitions (same as flat)
        transitions = self._build_transitions(machine)
        events = self._build_events(machine)

        statechart = {
            "root_state": root_state,
            "transitions": transitions,
            "events": events,
            "metadata": {
                "source": "react_component",
                "component_name": machine.name,
                "hierarchical": True
            }
        }

        self.statecharts.append(statechart)
        return statechart

    def build_parallel(self, machine: ExtractedStateMachine) -> Dict:
        """
        Build a parallel (orthogonal) statechart.

        Each independent state variable becomes an orthogonal region.
        """
        regions = []

        for var in machine.variables:
            if var.state_type.value == "boolean":
                # Create orthogonal region for boolean variable
                region = {
                    "label": f"{var.name}_region",
                    "type": self.STATE_TYPE_NORMAL,
                    "children": [
                        {
                            "label": f"{var.name}_true",
                            "type": self.STATE_TYPE_BASIC,
                            "is_initial": var.initial_value == "true"
                        },
                        {
                            "label": f"{var.name}_false",
                            "type": self.STATE_TYPE_BASIC,
                            "is_initial": var.initial_value == "false"
                        }
                    ]
                }
                regions.append(region)

            elif var.is_reducer and var.possible_values:
                # Create region for reducer states
                reducer_children = [
                    {
                        "label": "idle",
                        "type": self.STATE_TYPE_BASIC,
                        "is_initial": True
                    }
                ]
                for action in var.possible_values:
                    reducer_children.append({
                        "label": f"after_{action.lower()}",
                        "type": self.STATE_TYPE_BASIC,
                        "is_initial": False
                    })

                region = {
                    "label": f"{var.name}_region",
                    "type": self.STATE_TYPE_NORMAL,
                    "children": reducer_children
                }
                regions.append(region)

        # If multiple regions, make root parallel
        if len(regions) > 1:
            root_type = self.STATE_TYPE_PARALLEL
        else:
            root_type = self.STATE_TYPE_NORMAL
            if regions:
                regions = regions[0].get("children", [])

        root_state = {
            "label": "__root__",
            "type": root_type,
            "children": regions if len(regions) > 1 else regions
        }

        transitions = self._build_transitions(machine)
        events = self._build_events(machine)

        statechart = {
            "root_state": root_state,
            "transitions": transitions,
            "events": events,
            "metadata": {
                "source": "react_component",
                "component_name": machine.name,
                "parallel": len(regions) > 1
            }
        }

        self.statecharts.append(statechart)
        return statechart

    def _build_transitions(self, machine: ExtractedStateMachine) -> List[Dict]:
        """Build transitions list from machine."""
        transitions = []
        for trans in machine.transitions:
            t = {
                "from": [trans.from_state],
                "to": [trans.to_state],
                "event": trans.event
            }
            if trans.guard:
                t["guard"] = {"expression": trans.guard}
            if trans.action:
                t["actions"] = [{"name": trans.action}]
            transitions.append(t)
        return transitions

    def _build_events(self, machine: ExtractedStateMachine) -> List[Dict]:
        """Build events list from machine."""
        return [
            {"name": event, "description": "Event from React handler"}
            for event in machine.events
        ]

    def to_json(self, statechart: Dict, indent: int = 2) -> str:
        """Convert statechart to JSON string."""
        return json.dumps(statechart, indent=indent)

    def to_mermaid(self, statechart: Dict) -> str:
        """Generate Mermaid diagram from statechart."""
        lines = ["stateDiagram-v2", "  direction TB"]

        # Find initial state
        root = statechart.get("root_state", {})
        for child in root.get("children", []):
            if child.get("is_initial"):
                lines.append(f"  [*] --> {child['label']}")
                break

        # Add transitions
        for trans in statechart.get("transitions", []):
            from_state = trans["from"][0] if trans.get("from") else "?"
            to_state = trans["to"][0] if trans.get("to") else "?"
            event = trans.get("event", "")

            if event:
                lines.append(f"  {from_state} --> {to_state} : {event}")
            else:
                lines.append(f"  {from_state} --> {to_state}")

        return "\n".join(lines)

    def save(self, statechart: Dict, path: str) -> None:
        """Save statechart to JSON file."""
        with open(path, 'w') as f:
            json.dump(statechart, f, indent=2)

    def save_mermaid(self, statechart: Dict, path: str) -> None:
        """Save Mermaid diagram to file."""
        with open(path, 'w') as f:
            f.write(self.to_mermaid(statechart))


def demo():
    """Demonstrate SC builder."""
    from .react_parser import ReactParser
    from .state_extractor import StateExtractor

    sample_code = '''
    function TodoApp() {
        const [isModalOpen, setIsModalOpen] = useState(false);
        const [isLoading, setIsLoading] = useState(false);
        const [todos, dispatch] = useReducer(todoReducer, []);

        const handleOpenModal = () => {
            setIsModalOpen(true);
        };

        const handleCloseModal = () => {
            setIsModalOpen(false);
        };

        const handleAddTodo = () => {
            setIsLoading(true);
            dispatch({ type: 'ADD_TODO' });
        };

        const handleRemoveTodo = (id) => {
            dispatch({ type: 'REMOVE_TODO' });
        };

        return <div />;
    }
    '''

    # Parse -> Extract -> Build
    parser = ReactParser()
    components = parser.parse(sample_code)

    extractor = StateExtractor()
    builder = SCBuilder()

    print("=" * 60)
    print("SC BUILDER DEMO")
    print("=" * 60)

    for component in components:
        machine = extractor.extract(component)

        # Build flat statechart
        print(f"\n--- Flat Statechart for {machine.name} ---")
        sc_flat = builder.build(machine)
        print(builder.to_json(sc_flat))

        print(f"\n--- Mermaid Diagram ---")
        print(builder.to_mermaid(sc_flat))

        # Build parallel statechart
        print(f"\n--- Parallel Statechart ---")
        sc_parallel = builder.build_parallel(machine)
        print(f"Root type: {'PARALLEL' if sc_parallel['root_state']['type'] == 3 else 'NORMAL'}")
        print(f"Regions: {len(sc_parallel['root_state'].get('children', []))}")


if __name__ == "__main__":
    demo()
