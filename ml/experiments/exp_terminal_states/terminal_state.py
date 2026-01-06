"""
Terminal State - Final state handling for statecharts.

Proto defines State.is_final (line 179):
  "Terminal state: no outgoing transitions"

Final states have special semantics:
1. No outgoing transitions allowed
2. Reaching final in all parallel regions signals completion
3. Completion triggers parent's completion transition (τ event)

FORMAL SEMANTICS (Harel):
- Final state f: is_final(f) = true ⟹ ∄t ∈ δ: src(t) = f
- Completion event τ: generated when final state reached
- Parallel completion: PARALLEL state p completes when
  ∀r ∈ children(p): ∃f ∈ descendants(r): is_final(f) ∧ f ∈ σ
"""

from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional, Any, Tuple
from enum import Enum, auto


class StateType(Enum):
    """State types matching proto StateType enum."""
    BASIC = 1       # Leaf state
    NORMAL = 2      # OR-state (XOR children)
    PARALLEL = 3    # AND-state (concurrent children)


@dataclass
class State:
    """State representation with final semantics."""
    label: str
    state_type: StateType = StateType.BASIC
    children: List['State'] = field(default_factory=list)
    is_initial: bool = False
    is_final: bool = False
    parent: Optional['State'] = None

    def __post_init__(self):
        # Set parent references for children
        for child in self.children:
            child.parent = self

    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def is_root(self) -> bool:
        return self.parent is None

    def get_ancestors(self) -> List['State']:
        """Get all ancestors from parent to root."""
        ancestors = []
        current = self.parent
        while current is not None:
            ancestors.append(current)
            current = current.parent
        return ancestors

    def get_descendants(self) -> List['State']:
        """Get all descendant states."""
        descendants = []
        for child in self.children:
            descendants.append(child)
            descendants.extend(child.get_descendants())
        return descendants

    def get_final_descendants(self) -> List['State']:
        """Get all final states in descendants."""
        return [d for d in self.get_descendants() if d.is_final]

    def has_final_child(self) -> bool:
        """Check if any direct child is final."""
        return any(c.is_final for c in self.children)

    def __repr__(self) -> str:
        final_mark = "*" if self.is_final else ""
        initial_mark = "→" if self.is_initial else ""
        return f"{initial_mark}{self.label}{final_mark}"


@dataclass
class Transition:
    """Transition with completion event support."""
    source: List[str]  # Source state labels
    target: List[str]  # Target state labels
    event: str = ""    # Empty = completion transition (τ)
    guard: str = ""
    actions: List[str] = field(default_factory=list)

    def is_completion_transition(self) -> bool:
        """Check if this is a completion (τ) transition."""
        return self.event == "" or self.event == "τ"


@dataclass
class Configuration:
    """Active state configuration."""
    active_states: Set[str]

    def is_active(self, state_label: str) -> bool:
        return state_label in self.active_states

    def add(self, state_label: str):
        self.active_states.add(state_label)

    def remove(self, state_label: str):
        self.active_states.discard(state_label)


class FinalStateValidator:
    """Validate final state constraints."""

    def validate_no_outgoing(
        self,
        state: State,
        transitions: List[Transition],
    ) -> List[str]:
        """
        Validate: is_final(s) ⟹ ∄t: src(t) = s
        Final states must have no outgoing transitions.
        """
        errors = []

        if state.is_final:
            for t in transitions:
                if state.label in t.source:
                    errors.append(
                        f"Final state '{state.label}' has outgoing transition "
                        f"on event '{t.event or 'τ'}' to {t.target}"
                    )

        # Recursively check children
        for child in state.children:
            errors.extend(self.validate_no_outgoing(child, transitions))

        return errors

    def validate_parallel_finals(self, state: State) -> List[str]:
        """
        Validate parallel state final semantics.
        Each region of a PARALLEL state should have at least one final state.
        """
        errors = []

        if state.state_type == StateType.PARALLEL:
            for region in state.children:
                finals = region.get_final_descendants()
                if not finals and not region.is_final:
                    errors.append(
                        f"Parallel region '{region.label}' in '{state.label}' "
                        f"has no final states (may never complete)"
                    )

        # Recursively check children
        for child in state.children:
            errors.extend(self.validate_parallel_finals(child))

        return errors

    def validate_all(
        self,
        root: State,
        transitions: List[Transition],
    ) -> Tuple[bool, List[str]]:
        """Run all final state validations."""
        errors = []
        errors.extend(self.validate_no_outgoing(root, transitions))
        errors.extend(self.validate_parallel_finals(root))
        return len(errors) == 0, errors


class FinalStateSemantics:
    """Implement final state semantics."""

    def __init__(self, root: State, transitions: List[Transition]):
        self.root = root
        self.transitions = transitions
        self.state_map = self._build_state_map(root)

    def _build_state_map(self, state: State) -> Dict[str, State]:
        """Build label -> state mapping."""
        result = {state.label: state}
        for child in state.children:
            result.update(self._build_state_map(child))
        return result

    def get_state(self, label: str) -> Optional[State]:
        return self.state_map.get(label)

    def is_in_final_state(self, state: State, config: Configuration) -> bool:
        """
        Check if state is "completed" (in a final state).

        For BASIC final states: is_final and active
        For NORMAL states: active child is final
        For PARALLEL states: all regions have reached final
        """
        if state.is_final and config.is_active(state.label):
            return True

        if state.state_type == StateType.BASIC:
            return state.is_final and config.is_active(state.label)

        elif state.state_type == StateType.NORMAL:
            # OR-state: completed if active child is in final
            for child in state.children:
                if config.is_active(child.label):
                    return self.is_in_final_state(child, config)
            return False

        elif state.state_type == StateType.PARALLEL:
            # AND-state: completed if ALL regions are in final
            for region in state.children:
                if not self.is_in_final_state(region, config):
                    return False
            return True

        return False

    def get_completion_transitions(
        self,
        state: State,
        config: Configuration,
    ) -> List[Transition]:
        """
        Get enabled completion transitions for state.

        Completion transition enabled when:
        1. Transition has empty event (τ)
        2. Source state is completed (in final)
        """
        enabled = []

        if not self.is_in_final_state(state, config):
            return enabled

        for t in self.transitions:
            if t.is_completion_transition() and state.label in t.source:
                enabled.append(t)

        return enabled

    def propagate_completion(
        self,
        config: Configuration,
    ) -> List[Tuple[State, List[Transition]]]:
        """
        Find all states with enabled completion transitions.
        Propagates from leaves up to root.
        """
        completions = []

        # Check from bottom up
        def check_state(state: State):
            # First check children
            for child in state.children:
                check_state(child)

            # Then check this state
            if config.is_active(state.label):
                enabled = self.get_completion_transitions(state, config)
                if enabled:
                    completions.append((state, enabled))

        check_state(self.root)
        return completions


def build_parallel_statechart() -> Tuple[State, List[Transition]]:
    """
    Build example parallel statechart with final states.

    Structure:
        Root (PARALLEL)
        ├── Region1 (NORMAL)
        │   ├── R1_Active (initial)
        │   └── R1_Final (final)
        └── Region2 (NORMAL)
            ├── R2_Active (initial)
            └── R2_Final (final)

    Completion: Root completes when BOTH Region1 AND Region2 reach final.
    """
    # Build from leaves up
    r1_active = State(label="R1_Active", is_initial=True)
    r1_final = State(label="R1_Final", is_final=True)
    region1 = State(
        label="Region1",
        state_type=StateType.NORMAL,
        children=[r1_active, r1_final],
    )

    r2_active = State(label="R2_Active", is_initial=True)
    r2_final = State(label="R2_Final", is_final=True)
    region2 = State(
        label="Region2",
        state_type=StateType.NORMAL,
        children=[r2_active, r2_final],
    )

    root = State(
        label="Root",
        state_type=StateType.PARALLEL,
        children=[region1, region2],
    )

    # Transitions
    transitions = [
        Transition(source=["R1_Active"], target=["R1_Final"], event="done1"),
        Transition(source=["R2_Active"], target=["R2_Final"], event="done2"),
        # Completion transition from Root when all regions final
        Transition(source=["Root"], target=["Completed"], event=""),  # τ
    ]

    return root, transitions


def demo():
    """Demonstrate final state semantics."""
    print("=" * 60)
    print("TERMINAL STATE SEMANTICS")
    print("=" * 60)

    root, transitions = build_parallel_statechart()

    print("\nStatechart structure:")
    print(f"  {root.label} (PARALLEL)")
    for region in root.children:
        print(f"    {region.label} (NORMAL)")
        for child in region.children:
            print(f"      {child}")

    # Validate
    print("\n" + "-" * 60)
    print("VALIDATION")
    print("-" * 60)

    validator = FinalStateValidator()
    valid, errors = validator.validate_all(root, transitions)
    print(f"Valid: {valid}")
    for err in errors:
        print(f"  Error: {err}")

    # Test semantics
    print("\n" + "-" * 60)
    print("COMPLETION SEMANTICS")
    print("-" * 60)

    semantics = FinalStateSemantics(root, transitions)

    # Initial configuration: both regions in active state
    config1 = Configuration(active_states={
        "Root", "Region1", "R1_Active", "Region2", "R2_Active"
    })
    print(f"\nConfig 1: Both regions active")
    print(f"  Root completed: {semantics.is_in_final_state(root, config1)}")
    completions = semantics.propagate_completion(config1)
    print(f"  Completion transitions: {len(completions)}")

    # Region1 reaches final
    config2 = Configuration(active_states={
        "Root", "Region1", "R1_Final", "Region2", "R2_Active"
    })
    print(f"\nConfig 2: Region1 final, Region2 active")
    print(f"  Root completed: {semantics.is_in_final_state(root, config2)}")
    print(f"  Region1 completed: {semantics.is_in_final_state(semantics.get_state('Region1'), config2)}")
    print(f"  Region2 completed: {semantics.is_in_final_state(semantics.get_state('Region2'), config2)}")

    # Both regions reach final
    config3 = Configuration(active_states={
        "Root", "Region1", "R1_Final", "Region2", "R2_Final"
    })
    print(f"\nConfig 3: Both regions final")
    print(f"  Root completed: {semantics.is_in_final_state(root, config3)}")
    completions = semantics.propagate_completion(config3)
    print(f"  Completion transitions: {len(completions)}")
    for state, trans in completions:
        print(f"    {state.label}: {[t.target for t in trans]}")

    return semantics


if __name__ == "__main__":
    demo()
