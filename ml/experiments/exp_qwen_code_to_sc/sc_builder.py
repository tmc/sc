"""
Statechart Builder - Build proto-compatible statechart from extracted patterns.

Converts extracted FSM patterns to the statechart proto format:
- Statechart message
- State hierarchy
- Transitions with events/guards
- Actions

Output formats:
- JSON (proto3 JSON format)
- TextProto
- Python dict (for validation)
"""

import json
from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional, Any
from enum import Enum

from .pattern_extractor import ExtractedPattern, ExtractionResult


class StateType(Enum):
    """State types matching proto."""
    BASIC = 1
    NORMAL = 2
    PARALLEL = 3


@dataclass
class StateProto:
    """Proto-compatible state representation."""
    label: str
    type: StateType = StateType.BASIC
    children: List['StateProto'] = field(default_factory=list)
    is_initial: bool = False
    is_final: bool = False
    entry_actions: List[str] = field(default_factory=list)
    exit_actions: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        result = {
            'label': self.label,
            'type': self.type.value,
        }
        if self.children:
            result['children'] = [c.to_dict() for c in self.children]
        if self.is_initial:
            result['is_initial'] = True
        if self.is_final:
            result['is_final'] = True
        if self.entry_actions:
            result['entry_actions'] = [{'expression': a} for a in self.entry_actions]
        if self.exit_actions:
            result['exit_actions'] = [{'expression': a} for a in self.exit_actions]
        return result

    def to_textproto(self, indent: int = 0) -> str:
        prefix = "  " * indent
        lines = [f'{prefix}label: "{self.label}"']
        lines.append(f'{prefix}type: STATE_TYPE_{self.type.name}')
        if self.is_initial:
            lines.append(f'{prefix}is_initial: true')
        if self.is_final:
            lines.append(f'{prefix}is_final: true')
        for child in self.children:
            lines.append(f'{prefix}children {{')
            lines.append(child.to_textproto(indent + 1))
            lines.append(f'{prefix}}}')
        for action in self.entry_actions:
            lines.append(f'{prefix}entry_actions {{ expression: "{action}" }}')
        for action in self.exit_actions:
            lines.append(f'{prefix}exit_actions {{ expression: "{action}" }}')
        return '\n'.join(lines)


@dataclass
class TransitionProto:
    """Proto-compatible transition representation."""
    source: List[str]
    target: List[str]
    event: str = ""
    guard: str = ""
    actions: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        result = {
            'from': self.source,
            'to': self.target,
        }
        if self.event:
            result['event'] = self.event
        if self.guard:
            result['guard'] = {'expression': self.guard}
        if self.actions:
            result['actions'] = [{'expression': a} for a in self.actions]
        return result

    def to_textproto(self) -> str:
        lines = []
        for src in self.source:
            lines.append(f'from: "{src}"')
        for tgt in self.target:
            lines.append(f'to: "{tgt}"')
        if self.event:
            lines.append(f'event: "{self.event}"')
        if self.guard:
            lines.append(f'guard {{ expression: "{self.guard}" }}')
        for action in self.actions:
            lines.append(f'actions {{ expression: "{action}" }}')
        return '\n'.join(lines)


@dataclass
class StatechartProto:
    """Proto-compatible statechart representation."""
    root_state: StateProto
    transitions: List[TransitionProto] = field(default_factory=list)
    events: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            'root_state': self.root_state.to_dict(),
            'transitions': [t.to_dict() for t in self.transitions],
            'events': [{'label': e} for e in self.events],
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def to_textproto(self) -> str:
        lines = ['root_state {']
        lines.append(self.root_state.to_textproto(1))
        lines.append('}')

        for transition in self.transitions:
            lines.append('transitions {')
            lines.append('  ' + transition.to_textproto().replace('\n', '\n  '))
            lines.append('}')

        for event in self.events:
            lines.append(f'events {{ label: "{event}" }}')

        return '\n'.join(lines)


class StatechartBuilder:
    """Build statechart from extracted patterns."""

    def __init__(self):
        self.root_name = "__root__"

    def build(self, result: ExtractionResult) -> StatechartProto:
        """Build statechart from extraction result."""
        # Collect all states
        all_states = set()
        initial_state = None
        final_states = set()

        for pattern in result.patterns:
            all_states.update(pattern.states)
            if pattern.initial_state:
                initial_state = pattern.initial_state
            final_states.update(pattern.final_states)

        # Also from combined
        all_states.update(result.combined_states)

        # Build state hierarchy
        root = self._build_state_hierarchy(
            all_states, initial_state, final_states
        )

        # Build transitions
        transitions = self._build_transitions(result.combined_transitions)

        # Collect events
        events = set()
        for pattern in result.patterns:
            events.update(pattern.events)
        for t in result.combined_transitions:
            if t.get('event'):
                events.add(t['event'])

        return StatechartProto(
            root_state=root,
            transitions=transitions,
            events=list(events),
        )

    def _build_state_hierarchy(
        self,
        states: Set[str],
        initial_state: Optional[str],
        final_states: Set[str],
    ) -> StateProto:
        """Build state hierarchy (flat for now)."""
        # Create root as NORMAL (OR) state
        root = StateProto(
            label=self.root_name,
            type=StateType.NORMAL,
        )

        # Add all states as children
        for state_name in sorted(states):
            child = StateProto(
                label=state_name,
                type=StateType.BASIC,
                is_initial=(state_name == initial_state),
                is_final=(state_name in final_states),
            )
            root.children.append(child)

        return root

    def _build_transitions(
        self,
        transitions: List[Dict],
    ) -> List[TransitionProto]:
        """Build transition protos."""
        result = []

        for t in transitions:
            from_state = t.get('from', '*')
            to_state = t.get('to', '')

            if not to_state:
                continue

            # Handle wildcard source
            source = [from_state] if from_state != '*' else []

            proto = TransitionProto(
                source=source if source else [from_state],
                target=[to_state],
                event=t.get('event', ''),
                guard=t.get('guard', ''),
                actions=[t.get('action')] if t.get('action') else [],
            )
            result.append(proto)

        return result

    def build_from_pattern(self, pattern: ExtractedPattern) -> StatechartProto:
        """Build statechart from single pattern."""
        # Create root
        root = StateProto(
            label=self.root_name,
            type=StateType.NORMAL,
        )

        # Add states
        for state_name in pattern.states:
            child = StateProto(
                label=state_name,
                type=StateType.BASIC,
                is_initial=(state_name == pattern.initial_state),
                is_final=(state_name in pattern.final_states),
            )
            root.children.append(child)

        # Add transitions
        transitions = []
        for t in pattern.transitions:
            proto = TransitionProto(
                source=[t.get('from', '')],
                target=[t.get('to', '')],
                event=t.get('event', ''),
                guard=t.get('guard', ''),
            )
            transitions.append(proto)

        return StatechartProto(
            root_state=root,
            transitions=transitions,
            events=pattern.events,
        )


class StatechartValidator:
    """Validate built statecharts."""

    def validate(self, sc: StatechartProto) -> List[str]:
        """Validate statechart structure."""
        errors = []

        # Check root has children
        if not sc.root_state.children:
            errors.append("Root state has no children")

        # Check initial state exists
        has_initial = any(c.is_initial for c in sc.root_state.children)
        if not has_initial and sc.root_state.children:
            errors.append("No initial state marked")

        # Check transitions reference valid states
        state_labels = {c.label for c in sc.root_state.children}
        for t in sc.transitions:
            for src in t.source:
                if src and src not in state_labels and src != '*':
                    errors.append(f"Transition source '{src}' not in states")
            for tgt in t.target:
                if tgt and tgt not in state_labels:
                    errors.append(f"Transition target '{tgt}' not in states")

        return errors


class StatechartSerializer:
    """Serialize statecharts to various formats."""

    @staticmethod
    def to_json(sc: StatechartProto) -> str:
        """Serialize to JSON."""
        return sc.to_json()

    @staticmethod
    def to_textproto(sc: StatechartProto) -> str:
        """Serialize to textproto format."""
        return sc.to_textproto()

    @staticmethod
    def to_mermaid(sc: StatechartProto) -> str:
        """Serialize to Mermaid stateDiagram."""
        lines = ['stateDiagram-v2']

        # Add states
        for child in sc.root_state.children:
            if child.is_initial:
                lines.append(f'    [*] --> {child.label}')
            if child.is_final:
                lines.append(f'    {child.label} --> [*]')

        # Add transitions
        for t in sc.transitions:
            for src in t.source:
                for tgt in t.target:
                    label = t.event if t.event else ''
                    if label:
                        lines.append(f'    {src} --> {tgt}: {label}')
                    else:
                        lines.append(f'    {src} --> {tgt}')

        return '\n'.join(lines)


def demo():
    """Demonstrate statechart building."""
    print("=" * 60)
    print("STATECHART BUILDER")
    print("=" * 60)

    from .pattern_extractor import RuleBasedExtractor

    code = '''
def traffic_light():
    state = "red"
    while True:
        if state == "red":
            wait(30)
            state = "green"
        elif state == "green":
            wait(25)
            state = "yellow"
        elif state == "yellow":
            wait(5)
            state = "red"
'''

    # Extract patterns
    extractor = RuleBasedExtractor()
    result = extractor.extract(code)

    # Build statechart
    builder = StatechartBuilder()
    sc = builder.build(result)

    # Validate
    validator = StatechartValidator()
    errors = validator.validate(sc)

    print("\n--- JSON Output ---")
    print(sc.to_json())

    print("\n--- TextProto Output ---")
    print(sc.to_textproto())

    print("\n--- Mermaid Output ---")
    print(StatechartSerializer.to_mermaid(sc))

    print("\n--- Validation ---")
    if errors:
        for e in errors:
            print(f"  Error: {e}")
    else:
        print("  Valid statechart!")

    # Stats
    print("\n--- Stats ---")
    print(f"  States: {len(sc.root_state.children)}")
    print(f"  Transitions: {len(sc.transitions)}")
    print(f"  Events: {len(sc.events)}")

    return sc


if __name__ == "__main__":
    demo()
