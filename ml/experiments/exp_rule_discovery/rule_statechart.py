"""
Rule Statechart - Combines discovered states and guards into executable statechart.

This is the output of the rule discovery pipeline:
- States discovered from clustering
- Guards synthesized from examples
- Fully interpretable and executable
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Set
import json

try:
    from .state_discoverer import AbstractState
    from .guard_synthesizer import SymbolicGuard, Predicate, Comparator
except ImportError:
    from state_discoverer import AbstractState
    from guard_synthesizer import SymbolicGuard, Predicate, Comparator


@dataclass
class DiscoveredRule:
    """A discovered game rule."""
    name: str
    rule_type: str  # 'legality', 'win', 'transition', 'action'
    guard: SymbolicGuard
    action: Optional[Any] = None
    from_state: Optional[str] = None
    to_state: Optional[str] = None
    confidence: float = 0.0
    examples_seen: int = 0

    def evaluate(self, features: Dict[str, float]) -> bool:
        """Check if rule applies given features."""
        return self.guard.evaluate(features)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'type': self.rule_type,
            'guard': self.guard.to_dict(),
            'action': str(self.action) if self.action else None,
            'from_state': self.from_state,
            'to_state': self.to_state,
            'confidence': self.confidence,
            'examples': self.examples_seen,
        }

    def __str__(self) -> str:
        parts = [f"Rule: {self.name}"]
        if self.from_state and self.to_state:
            parts.append(f"  Transition: {self.from_state} -> {self.to_state}")
        if self.action:
            parts.append(f"  Action: {self.action}")
        parts.append(f"  Guard: {self.guard}")
        parts.append(f"  Confidence: {self.confidence:.2%}")
        return '\n'.join(parts)


@dataclass
class RuleStatechart:
    """
    A discovered game as an interpretable statechart.

    Contains:
    - Abstract states (game phases)
    - Transition rules with guards
    - Legality rules
    - Win/loss conditions
    """
    name: str = "DiscoveredGame"
    states: List[AbstractState] = field(default_factory=list)
    rules: List[DiscoveredRule] = field(default_factory=list)
    initial_state: Optional[str] = None

    # Indexed access
    _legality_rules: List[DiscoveredRule] = field(default_factory=list)
    _transition_rules: Dict[str, List[DiscoveredRule]] = field(default_factory=dict)
    _win_rules: List[DiscoveredRule] = field(default_factory=list)

    def add_state(self, state: AbstractState):
        """Add an abstract state."""
        self.states.append(state)
        if self.initial_state is None:
            self.initial_state = state.name

    def add_rule(self, rule: DiscoveredRule):
        """Add a discovered rule."""
        self.rules.append(rule)

        # Index by type
        if rule.rule_type == 'legality':
            self._legality_rules.append(rule)
        elif rule.rule_type == 'win':
            self._win_rules.append(rule)
        elif rule.rule_type == 'transition' and rule.from_state:
            if rule.from_state not in self._transition_rules:
                self._transition_rules[rule.from_state] = []
            self._transition_rules[rule.from_state].append(rule)

    def is_legal(self, state_features: Dict[str, float],
                 action_features: Dict[str, float] = None) -> bool:
        """
        Check if action is legal given state features.

        If no legality rules, assume all legal.
        """
        if not self._legality_rules:
            return True

        features = state_features.copy()
        if action_features:
            features.update(action_features)

        # All legality rules must pass
        for rule in self._legality_rules:
            if not rule.evaluate(features):
                return False
        return True

    def is_winning(self, features: Dict[str, float]) -> bool:
        """Check if current state is a winning position."""
        for rule in self._win_rules:
            if rule.evaluate(features):
                return True
        return False

    def get_next_state(self, current_state: str,
                       features: Dict[str, float]) -> Optional[str]:
        """Get next state based on transition rules."""
        if current_state not in self._transition_rules:
            return current_state

        for rule in self._transition_rules[current_state]:
            if rule.evaluate(features):
                return rule.to_state

        return current_state

    def classify_state(self, features: Dict[str, float]) -> Optional[AbstractState]:
        """Classify features into an abstract state."""
        if not self.states:
            return None

        # Find closest state by centroid distance
        best_state = None
        best_dist = float('inf')

        for state in self.states:
            dist = 0.0
            for key, centroid_val in state.centroid.items():
                feat_val = features.get(key, 0.0)
                dist += (feat_val - centroid_val) ** 2

            if dist < best_dist:
                best_dist = dist
                best_state = state

        return best_state

    def to_dict(self) -> Dict[str, Any]:
        """Export statechart as dictionary."""
        return {
            'name': self.name,
            'states': [
                {
                    'id': s.id,
                    'name': s.name,
                    'centroid': s.centroid,
                    'win_rate': s.win_rate,
                    'is_terminal': s.is_terminal,
                }
                for s in self.states
            ],
            'rules': [r.to_dict() for r in self.rules],
            'initial_state': self.initial_state,
        }

    def to_json(self, indent: int = 2) -> str:
        """Export as JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RuleStatechart':
        """Load statechart from dictionary."""
        sc = cls(name=data.get('name', 'Loaded'))
        sc.initial_state = data.get('initial_state')

        # Load states
        for state_data in data.get('states', []):
            state = AbstractState(
                id=state_data['id'],
                name=state_data['name'],
                centroid=state_data.get('centroid', {}),
                win_rate=state_data.get('win_rate', 0),
                is_terminal=state_data.get('is_terminal', False),
            )
            sc.add_state(state)

        # Load rules
        for rule_data in data.get('rules', []):
            guard_data = rule_data.get('guard', {})
            predicates = []
            for pred_data in guard_data.get('predicates', []):
                comp = Comparator(pred_data['op'])
                pred = Predicate(
                    feature=pred_data['feature'],
                    comparator=comp,
                    threshold=pred_data['threshold'],
                )
                predicates.append(pred)

            guard = SymbolicGuard(
                predicates=predicates,
                coverage=guard_data.get('coverage', 0),
                precision=guard_data.get('precision', 0),
            )

            rule = DiscoveredRule(
                name=rule_data['name'],
                rule_type=rule_data['type'],
                guard=guard,
                action=rule_data.get('action'),
                from_state=rule_data.get('from_state'),
                to_state=rule_data.get('to_state'),
                confidence=rule_data.get('confidence', 0),
                examples_seen=rule_data.get('examples', 0),
            )
            sc.add_rule(rule)

        return sc

    def describe(self) -> str:
        """Generate human-readable description of the statechart."""
        lines = [
            f"=== Rule Statechart: {self.name} ===\n",
            f"States: {len(self.states)}",
            f"Rules: {len(self.rules)}",
            f"Initial State: {self.initial_state}",
            "",
        ]

        # Describe states
        lines.append("--- States ---")
        for state in self.states:
            lines.append(f"  {state.name}:")
            lines.append(f"    Win Rate: {state.win_rate:.1%}")
            lines.append(f"    Terminal: {state.is_terminal}")
            if state.centroid:
                lines.append(f"    Key Features: {list(state.centroid.keys())[:3]}")

        # Describe rules by type
        lines.append("\n--- Legality Rules ---")
        for rule in self._legality_rules:
            lines.append(f"  {rule.guard}")

        lines.append("\n--- Win Conditions ---")
        for rule in self._win_rules:
            lines.append(f"  {rule.guard}")

        lines.append("\n--- Transition Rules ---")
        for from_state, rules in self._transition_rules.items():
            for rule in rules:
                lines.append(f"  {from_state} -> {rule.to_state}:")
                lines.append(f"    when {rule.guard}")

        return '\n'.join(lines)

    def to_scxml(self) -> str:
        """Export as SCXML format."""
        lines = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            f'<scxml name="{self.name}" version="1.0" xmlns="http://www.w3.org/2005/07/scxml">',
        ]

        for state in self.states:
            terminal_attr = ' final="true"' if state.is_terminal else ''
            initial_attr = ' initial="true"' if state.name == self.initial_state else ''
            lines.append(f'  <state id="{state.name}"{terminal_attr}{initial_attr}>')

            # Add transitions from this state
            if state.name in self._transition_rules:
                for rule in self._transition_rules[state.name]:
                    cond = str(rule.guard).replace('"', '&quot;')
                    lines.append(f'    <transition target="{rule.to_state}" cond="{cond}"/>')

            lines.append('  </state>')

        lines.append('</scxml>')
        return '\n'.join(lines)
