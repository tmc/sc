"""
Extended Statechart with Semantic Components.

Extends RegexStatechart with:
- Extended state variables (counters, flags, buffers)
- Semantic transitions (guards + actions)
- Entry/exit actions on states
- Accept conditions (beyond just being in accept state)

This enables learning patterns that require counting, context, or memory:
- a{3,5}: exactly 3-5 'a's (requires counter)
- (ab)+: repeat detection (can use counter or flag)
- Backreference-like: match same char twice (requires buffer)
"""

from dataclasses import dataclass, field
from typing import List, Set, Dict, Optional, Tuple, FrozenSet
from enum import Enum, auto
import copy

from .regex_statechart import StateType
from .semantic_components import (
    StateVar, VarType, ExtendedState,
    GuardExpr, CharGuardExpr, CounterGuardExpr, FlagGuardExpr, CompositeGuardExpr,
    ActionExpr, NoOpAction, IncrementAction, ResetAction, SetFlagAction,
    AppendCharAction, CompositeAction,
    SemanticTransition, ComponentFactory
)


@dataclass
class AcceptCondition:
    """
    Condition for accepting input beyond just being in accept state.

    Examples:
    - counter == 3: accept only if saw exactly 3 matches
    - buffer == input: accept if captured equals input (for backrefs)
    """
    var_name: Optional[str] = None
    operator: str = "any"  # 'any', '==', '>=', '<=', '<', '>'
    value: Optional[int] = None

    def evaluate(self, state: ExtendedState) -> bool:
        """Check if accept condition is met."""
        if self.operator == "any":
            return True

        if self.var_name is None:
            return True

        var_val = state.get(self.var_name, 0)

        ops = {
            '==': lambda a, b: a == b,
            '>=': lambda a, b: a >= b,
            '<=': lambda a, b: a <= b,
            '<': lambda a, b: a < b,
            '>': lambda a, b: a > b,
        }

        return ops.get(self.operator, lambda a, b: True)(var_val, self.value)

    def to_string(self) -> str:
        if self.operator == "any":
            return "true"
        return f"{self.var_name} {self.operator} {self.value}"


@dataclass
class StateActions:
    """Entry and exit actions for a state."""
    entry_actions: List[ActionExpr] = field(default_factory=list)
    exit_actions: List[ActionExpr] = field(default_factory=list)

    def execute_entry(self, char: str, state: ExtendedState) -> ExtendedState:
        """Execute entry actions."""
        for action in self.entry_actions:
            state = action.execute(char, state)
        return state

    def execute_exit(self, char: str, state: ExtendedState) -> ExtendedState:
        """Execute exit actions."""
        for action in self.exit_actions:
            state = action.execute(char, state)
        return state


@dataclass
class ExtendedStatechart:
    """
    Statechart with extended state and semantic components.

    Combines:
    - Basic state structure (labels, types)
    - Extended state variables
    - Semantic transitions (guard + action)
    - State-level entry/exit actions
    - Accept conditions
    """
    # Basic structure
    state_labels: List[str] = field(default_factory=list)
    state_types: List[StateType] = field(default_factory=list)
    initial_state: int = 0

    # Extended state
    variables: List[StateVar] = field(default_factory=list)

    # Semantic transitions
    transitions: List[SemanticTransition] = field(default_factory=list)

    # State actions
    state_actions: Dict[int, StateActions] = field(default_factory=dict)

    # Accept conditions (per accept state)
    accept_conditions: Dict[int, AcceptCondition] = field(default_factory=dict)

    # Fitness (for evolution)
    fitness: float = 0.0

    @property
    def n_states(self) -> int:
        return len(self.state_labels)

    @property
    def n_transitions(self) -> int:
        return len(self.transitions)

    @property
    def accept_states(self) -> List[int]:
        """Get indices of accepting states."""
        return [i for i, t in enumerate(self.state_types) if t == StateType.ACCEPT]

    def get_initial_extended_state(self) -> ExtendedState:
        """Create initial extended state with all variables."""
        variables = {}
        for var in self.variables:
            variables[var.name] = var.initial_value
        return ExtendedState(variables=variables)

    def get_transitions_from(self, state: int) -> List[SemanticTransition]:
        """Get all transitions from a state, sorted by priority."""
        trans = [t for t in self.transitions if t.source == state]
        return sorted(trans, key=lambda t: t.priority, reverse=True)

    def step(
        self,
        current_states: Set[Tuple[int, FrozenSet]],
        char: str
    ) -> Set[Tuple[int, FrozenSet]]:
        """
        Process one character with extended state.

        Each configuration is (state_index, frozen_extended_state).
        Returns new set of configurations.
        """
        next_configs = set()

        for state_idx, frozen_vars in current_states:
            # Reconstruct extended state
            ext_state = ExtendedState(variables=dict(frozen_vars))
            ext_state.current_char = char

            # Execute exit actions for current state
            if state_idx in self.state_actions:
                ext_state = self.state_actions[state_idx].execute_exit(char, ext_state)

            # Try each transition from this state
            for trans in self.get_transitions_from(state_idx):
                if trans.can_fire(char, ext_state):
                    # Fire transition
                    new_ext_state = trans.fire(char, ext_state)

                    # Execute entry actions for target state
                    if trans.target in self.state_actions:
                        new_ext_state = self.state_actions[trans.target].execute_entry(char, new_ext_state)

                    # Create frozen configuration
                    frozen = frozenset(new_ext_state.variables.items())
                    next_configs.add((trans.target, frozen))

        return next_configs

    def matches(self, string: str) -> bool:
        """
        Check if string matches this extended statechart.

        Returns True if after processing all characters, any configuration
        is in an accepting state AND satisfies the accept condition.
        """
        if not string:
            # Empty string handling
            init_state = self.get_initial_extended_state()
            frozen = frozenset(init_state.variables.items())

            if self.state_types[self.initial_state] == StateType.ACCEPT:
                # Check accept condition
                condition = self.accept_conditions.get(self.initial_state, AcceptCondition())
                return condition.evaluate(init_state)
            return False

        # Start with initial configuration
        init_ext_state = self.get_initial_extended_state()
        frozen = frozenset(init_ext_state.variables.items())
        current = {(self.initial_state, frozen)}

        # Process each character
        for char in string:
            current = self.step(current, char)
            if not current:
                return False  # Dead end

        # Check if any configuration is accepting
        for state_idx, frozen_vars in current:
            if state_idx in self.accept_states:
                # Check accept condition
                ext_state = ExtendedState(variables=dict(frozen_vars))
                condition = self.accept_conditions.get(state_idx, AcceptCondition())
                if condition.evaluate(ext_state):
                    return True

        return False

    def evaluate(self, strings: List[str]) -> List[bool]:
        """Evaluate multiple strings."""
        return [self.matches(s) for s in strings]

    def is_valid(self) -> bool:
        """Check if statechart is structurally valid."""
        if not self.state_labels:
            return False
        if len(self.state_labels) != len(self.state_types):
            return False
        if self.initial_state < 0 or self.initial_state >= self.n_states:
            return False
        if not self.accept_states:
            return False
        for t in self.transitions:
            if t.source < 0 or t.source >= self.n_states:
                return False
            if t.target < 0 or t.target >= self.n_states:
                return False
        return True

    def complexity(self) -> int:
        """Total complexity score."""
        score = self.n_states + len(self.variables)
        for t in self.transitions:
            score += t.complexity()
        return score

    def to_string(self) -> str:
        """Convert to human-readable string."""
        lines = ["ExtendedStatechart:"]

        # Variables
        if self.variables:
            lines.append(f"  Variables ({len(self.variables)}):")
            for var in self.variables:
                lines.append(f"    {var.name}: {var.var_type.name} = {var.initial_value}")

        # States
        lines.append(f"  States ({self.n_states}):")
        for i, (label, stype) in enumerate(zip(self.state_labels, self.state_types)):
            marker = ""
            if i == self.initial_state:
                marker += " [INITIAL]"
            if stype == StateType.ACCEPT:
                marker += " [ACCEPT]"
                if i in self.accept_conditions:
                    cond = self.accept_conditions[i]
                    if cond.operator != "any":
                        marker += f" when {cond.to_string()}"
            lines.append(f"    {i}: {label} ({stype.name}){marker}")

        # Transitions
        lines.append(f"  Transitions ({self.n_transitions}):")
        for t in self.transitions:
            lines.append(f"    {t.to_string(self.state_labels)}")

        return "\n".join(lines)

    def copy(self) -> "ExtendedStatechart":
        """Create a deep copy."""
        return copy.deepcopy(self)

    # =========================================================================
    # Factory Methods
    # =========================================================================

    @classmethod
    def create_empty(cls, n_states: int = 2) -> "ExtendedStatechart":
        """Create an empty extended statechart."""
        labels = ["START"] + [f"S{i}" for i in range(1, n_states - 1)] + ["ACCEPT"]
        types = [StateType.START] + [StateType.INTERMEDIATE] * (n_states - 2) + [StateType.ACCEPT]
        return cls(state_labels=labels, state_types=types, initial_state=0)

    @classmethod
    def create_counter_pattern(cls, char: str, min_count: int, max_count: int) -> "ExtendedStatechart":
        """
        Create a statechart matching char{min,max}.

        Example: a{2,4} matches "aa", "aaa", "aaaa"
        """
        sc = cls(
            state_labels=["START", "COUNTING", "ACCEPT"],
            state_types=[StateType.START, StateType.INTERMEDIATE, StateType.ACCEPT],
            variables=[StateVar(name="count", var_type=VarType.COUNTER, initial_value=0)],
            initial_state=0
        )

        # START -> COUNTING on first char
        sc.transitions.append(SemanticTransition(
            source=0,
            target=1,
            guard=CharGuardExpr(chars=frozenset([char])),
            action=IncrementAction(var_name="count")
        ))

        # COUNTING -> COUNTING while count < max
        sc.transitions.append(SemanticTransition(
            source=1,
            target=1,
            guard=CompositeGuardExpr(
                left=CharGuardExpr(chars=frozenset([char])),
                right=CounterGuardExpr(var_name="count", operator="<", value=max_count),
                operator="and"
            ),
            action=IncrementAction(var_name="count"),
            priority=1
        ))

        # COUNTING -> ACCEPT when count >= min (epsilon transition approximation)
        # We handle this via accept condition instead
        sc.transitions.append(SemanticTransition(
            source=1,
            target=2,
            guard=CompositeGuardExpr(
                left=CharGuardExpr(chars=frozenset([char])),
                right=CounterGuardExpr(var_name="count", operator=">=", value=min_count - 1),
                operator="and"
            ),
            action=IncrementAction(var_name="count"),
            priority=0
        ))

        # Accept condition: count must be in range
        sc.accept_conditions[2] = AcceptCondition(
            var_name="count",
            operator=">=",
            value=min_count
        )

        return sc

    @classmethod
    def create_repeat_pattern(cls, sequence: str) -> "ExtendedStatechart":
        """
        Create a statechart matching (sequence)+.

        Uses a counter to track complete repetitions.
        """
        n = len(sequence)
        labels = ["START"] + [f"S{i}" for i in range(1, n)] + ["ACCEPT"]
        types = [StateType.START] + [StateType.INTERMEDIATE] * (n - 1) + [StateType.ACCEPT]

        sc = cls(
            state_labels=labels,
            state_types=types,
            variables=[StateVar(name="reps", var_type=VarType.COUNTER, initial_value=0)],
            initial_state=0
        )

        # Build chain for sequence
        for i, char in enumerate(sequence):
            if i == n - 1:
                # Last char goes to ACCEPT
                sc.transitions.append(SemanticTransition(
                    source=i,
                    target=n,  # ACCEPT
                    guard=CharGuardExpr(chars=frozenset([char])),
                    action=IncrementAction(var_name="reps")
                ))
            else:
                sc.transitions.append(SemanticTransition(
                    source=i,
                    target=i + 1,
                    guard=CharGuardExpr(chars=frozenset([char])),
                    action=NoOpAction()
                ))

        # Loop back from ACCEPT to S1 for repetition
        sc.transitions.append(SemanticTransition(
            source=n,  # ACCEPT
            target=1,  # After first char
            guard=CharGuardExpr(chars=frozenset([sequence[0]])),
            action=NoOpAction()
        ))

        return sc


def test_extended_statechart():
    """Test extended statechart functionality."""
    print("=" * 60)
    print("EXTENDED STATECHART TESTS")
    print("=" * 60)

    # Test 1: Basic extended statechart
    print("\n1. Basic extended statechart (a+ with counter):")
    sc = ExtendedStatechart(
        state_labels=["START", "ACCEPT"],
        state_types=[StateType.START, StateType.ACCEPT],
        variables=[StateVar(name="count", var_type=VarType.COUNTER)],
        transitions=[
            SemanticTransition(
                source=0,
                target=1,
                guard=CharGuardExpr(chars=frozenset(['a'])),
                action=IncrementAction(var_name="count")
            ),
            SemanticTransition(
                source=1,
                target=1,
                guard=CharGuardExpr(chars=frozenset(['a'])),
                action=IncrementAction(var_name="count")
            ),
        ],
        initial_state=0
    )
    print(sc.to_string())

    test_cases = [("a", True), ("aa", True), ("aaa", True), ("", False), ("b", False)]
    for s, expected in test_cases:
        result = sc.matches(s)
        status = "PASS" if result == expected else "FAIL"
        print(f"   '{s}' -> {result} (expected {expected}) [{status}]")

    # Test 2: Counter pattern a{2,4}
    print("\n2. Counter pattern a{2,4}:")
    sc2 = ExtendedStatechart.create_counter_pattern('a', 2, 4)
    print(sc2.to_string())

    test_cases = [
        ("a", False), ("aa", True), ("aaa", True), ("aaaa", True),
        ("aaaaa", False), ("", False), ("b", False)
    ]
    for s, expected in test_cases:
        result = sc2.matches(s)
        status = "PASS" if result == expected else "FAIL"
        print(f"   '{s}' -> {result} (expected {expected}) [{status}]")

    # Test 3: Repeat pattern (ab)+
    print("\n3. Repeat pattern (ab)+:")
    sc3 = ExtendedStatechart.create_repeat_pattern("ab")
    print(sc3.to_string())

    test_cases = [
        ("ab", True), ("abab", True), ("ababab", True),
        ("a", False), ("aba", False), ("", False)
    ]
    for s, expected in test_cases:
        result = sc3.matches(s)
        status = "PASS" if result == expected else "FAIL"
        print(f"   '{s}' -> {result} (expected {expected}) [{status}]")

    # Test 4: Composite guards with actions
    print("\n4. Composite guards (match 'a' only when count < 3):")
    sc4 = ExtendedStatechart(
        state_labels=["LOOP", "ACCEPT"],
        state_types=[StateType.START, StateType.ACCEPT],
        variables=[StateVar(name="count", var_type=VarType.COUNTER)],
        transitions=[
            SemanticTransition(
                source=0,
                target=0,
                guard=CompositeGuardExpr(
                    left=CharGuardExpr(chars=frozenset(['a'])),
                    right=CounterGuardExpr(var_name="count", operator="<", value=3),
                    operator="and"
                ),
                action=IncrementAction(var_name="count"),
                priority=1
            ),
            SemanticTransition(
                source=0,
                target=1,
                guard=CompositeGuardExpr(
                    left=CharGuardExpr(chars=frozenset(['a'])),
                    right=CounterGuardExpr(var_name="count", operator="==", value=3),
                    operator="and"
                ),
                action=NoOpAction(),
                priority=0
            ),
        ],
        initial_state=0
    )
    # This accepts exactly "aaaa" (3 in loop + 1 to accept)

    test_cases = [("a", False), ("aa", False), ("aaa", False), ("aaaa", True), ("aaaaa", False)]
    for s, expected in test_cases:
        result = sc4.matches(s)
        status = "PASS" if result == expected else "FAIL"
        print(f"   '{s}' -> {result} (expected {expected}) [{status}]")

    print("\n" + "=" * 60)
    print("Extended statechart tests complete!")
    print("=" * 60)


if __name__ == "__main__":
    test_extended_statechart()
