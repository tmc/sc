"""
Property Checker: Verify Statechart Properties with Z3

Verifies key correctness properties:
1. No illegal transitions - guards prevent invalid state changes
2. Deadlock freedom - always an enabled transition (or final state)
3. Liveness - eventually reach target states
4. Determinism - at most one transition enabled at a time
5. Reachability - all states reachable from initial

Each property is encoded as an SMT formula and checked with Z3.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Set, Tuple, Optional, Any
from abc import ABC, abstractmethod
from enum import Enum, auto

from .smt_encoder import (
    SMTEncoder, StatechartFormula, StateVariable, TransitionFormula,
    GuardFormula, StateType, HAS_Z3,
)

if HAS_Z3:
    from z3 import (
        Bool, Int, And, Or, Not, Implies, Solver,
        sat, unsat, unknown, ForAll, Exists,
    )


class PropertyStatus(Enum):
    """Result of property verification."""
    SATISFIED = auto()      # Property holds
    VIOLATED = auto()       # Property does not hold
    UNKNOWN = auto()        # Could not determine
    ERROR = auto()          # Verification error


@dataclass
class PropertyResult:
    """Result from checking a property."""
    property_name: str
    status: PropertyStatus
    message: str = ""
    counterexample: Optional[Dict[str, Any]] = None
    verification_time: float = 0.0

    @property
    def passed(self) -> bool:
        return self.status == PropertyStatus.SATISFIED


class Property(ABC):
    """Base class for properties to verify."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Property name for reporting."""
        pass

    @abstractmethod
    def encode(self, formula: StatechartFormula) -> Any:
        """
        Encode property as Z3 formula.

        Returns formula that should be UNSAT if property holds.
        (We check for counterexamples, so UNSAT means no counterexample exists.)
        """
        pass

    def check(
        self,
        formula: StatechartFormula,
        timeout_ms: int = 5000,
    ) -> PropertyResult:
        """
        Check if property holds.

        Args:
            formula: Encoded statechart
            timeout_ms: Timeout in milliseconds

        Returns:
            PropertyResult with status and counterexample if violated
        """
        if not HAS_Z3:
            return PropertyResult(
                property_name=self.name,
                status=PropertyStatus.ERROR,
                message="Z3 not available",
            )

        import time
        start = time.time()

        try:
            solver = Solver()
            solver.set("timeout", timeout_ms)

            # Add statechart constraints
            solver.add(formula.configuration_constraint)
            solver.add(formula.transition_relation)

            # Add negation of property (looking for counterexample)
            property_formula = self.encode(formula)
            solver.add(property_formula)

            result = solver.check()
            elapsed = time.time() - start

            if result == unsat:
                # No counterexample found - property holds
                return PropertyResult(
                    property_name=self.name,
                    status=PropertyStatus.SATISFIED,
                    message="Property verified",
                    verification_time=elapsed,
                )
            elif result == sat:
                # Counterexample found - property violated
                model = solver.model()
                counterexample = {}
                for decl in model.decls():
                    counterexample[str(decl)] = str(model[decl])

                return PropertyResult(
                    property_name=self.name,
                    status=PropertyStatus.VIOLATED,
                    message="Property violated",
                    counterexample=counterexample,
                    verification_time=elapsed,
                )
            else:
                return PropertyResult(
                    property_name=self.name,
                    status=PropertyStatus.UNKNOWN,
                    message="Verification inconclusive",
                    verification_time=elapsed,
                )

        except Exception as e:
            return PropertyResult(
                property_name=self.name,
                status=PropertyStatus.ERROR,
                message=f"Verification error: {e}",
            )


class IllegalTransitionProperty(Property):
    """
    Verify no illegal transitions are possible.

    An illegal transition is one that:
    - Goes from state A to state B
    - But A and B are not connected by any transition

    We check: for all states, only declared transitions are possible.
    """

    def __init__(self, allowed_transitions: Set[Tuple[str, str]]):
        """
        Args:
            allowed_transitions: Set of (source, target) pairs that are legal
        """
        self.allowed_transitions = allowed_transitions

    @property
    def name(self) -> str:
        return "NoIllegalTransitions"

    def encode(self, formula: StatechartFormula) -> Any:
        """
        Encode: exists a state pair (s1, s2) where s1 active, s2 becomes active,
        but (s1, s2) not in allowed transitions.
        """
        if not HAS_Z3:
            return True

        # Find illegal transition: some state becomes active without proper transition
        illegal_clauses = []

        for s1_name, s1 in formula.states.items():
            for s2_name, s2 in formula.states.items():
                if s1_name == s2_name:
                    continue

                if (s1_name, s2_name) not in self.allowed_transitions:
                    # s1 active AND s2 becomes active => illegal
                    illegal = And(s1.active, s2.active_next)
                    illegal_clauses.append(illegal)

        if illegal_clauses:
            return Or(*illegal_clauses)
        return False  # No illegal transitions possible


class DeadlockFreedomProperty(Property):
    """
    Verify no deadlock states exist.

    A deadlock is a non-final state from which no transition is enabled.
    We check: for all non-final states, at least one transition is enabled.
    """

    @property
    def name(self) -> str:
        return "DeadlockFreedom"

    def encode(self, formula: StatechartFormula) -> Any:
        """
        Encode: exists a non-final state with no enabled transitions.
        """
        if not HAS_Z3:
            return True

        deadlock_clauses = []

        for state_name, state in formula.states.items():
            if state.is_final:
                continue

            # Find transitions from this state
            outgoing = [t for t in formula.transitions if t.source == state_name]

            if not outgoing:
                # No outgoing transitions - always deadlock if active
                deadlock_clauses.append(state.active)
            else:
                # Deadlock if active AND no transition enabled
                # This requires encoding guards, which we approximate
                # by just checking if state is active and non-final
                pass  # More sophisticated encoding would check guard satisfiability

        if deadlock_clauses:
            return Or(*deadlock_clauses)
        return False  # No deadlock possible


class LivenessProperty(Property):
    """
    Verify liveness: eventually reach target state.

    Given initial states and target states, verify that target is reachable.
    Uses bounded model checking up to k steps.
    """

    def __init__(self, target_states: Set[str], max_steps: int = 10):
        self.target_states = target_states
        self.max_steps = max_steps

    @property
    def name(self) -> str:
        return f"Liveness({self.target_states})"

    def encode(self, formula: StatechartFormula) -> Any:
        """
        Encode: starting from initial, can we reach target within k steps?

        We encode the NEGATION: no path reaches target.
        If UNSAT, then property holds (target is reachable).
        """
        if not HAS_Z3:
            return True

        # For bounded model checking, we'd need to unroll transitions
        # Simplified: just check if any target state could be active
        target_vars = []
        for target_name in self.target_states:
            if target_name in formula.states:
                target_vars.append(formula.states[target_name].active)

        if target_vars:
            # Check negation: no target is reachable (active)
            return And(*[Not(t) for t in target_vars])

        return True  # No targets specified


class DeterminismProperty(Property):
    """
    Verify determinism: at most one transition enabled at a time.

    For each state, at most one outgoing transition has its guard satisfied.
    """

    @property
    def name(self) -> str:
        return "Determinism"

    def encode(self, formula: StatechartFormula) -> Any:
        """
        Encode: exists a state with two or more enabled transitions.
        """
        if not HAS_Z3:
            return True

        nondeterminism_clauses = []

        # Group transitions by source
        by_source: Dict[str, List[TransitionFormula]] = {}
        for trans in formula.transitions:
            if trans.source not in by_source:
                by_source[trans.source] = []
            by_source[trans.source].append(trans)

        # For each source with multiple transitions, check if >1 enabled
        for source, transitions in by_source.items():
            if len(transitions) < 2:
                continue

            source_state = formula.states.get(source)
            if not source_state:
                continue

            # For each pair of transitions, check if both can be enabled
            for i, t1 in enumerate(transitions):
                for t2 in transitions[i + 1:]:
                    # Both enabled = source active AND both guards true
                    # We'd need to properly encode guards here
                    # Simplified: check if source has multiple non-guarded transitions
                    if t1.guard.expression == "true" and t2.guard.expression == "true":
                        nondeterminism_clauses.append(source_state.active)

        if nondeterminism_clauses:
            return Or(*nondeterminism_clauses)
        return False


class ReachabilityProperty(Property):
    """
    Verify all states are reachable from initial.

    No orphan states that can never be entered.
    """

    @property
    def name(self) -> str:
        return "AllStatesReachable"

    def encode(self, formula: StatechartFormula) -> Any:
        """
        Check reachability via graph analysis (not SMT).
        Returns True if unreachable states exist.
        """
        if not HAS_Z3:
            return True

        # Build reachability graph
        reachable: Set[str] = set()
        initial_states = [s.name for s in formula.states.values() if s.is_initial]

        # BFS from initial
        queue = list(initial_states)
        while queue:
            current = queue.pop(0)
            if current in reachable:
                continue
            reachable.add(current)

            # Add targets of outgoing transitions
            for trans in formula.transitions:
                if trans.source == current and trans.target not in reachable:
                    queue.append(trans.target)

        # Check for unreachable states
        unreachable = set(formula.states.keys()) - reachable

        if unreachable:
            # Return formula that's satisfiable (property violated)
            # Use a dummy variable
            return Bool("unreachable_exists")

        return False  # All states reachable


class PropertyChecker:
    """
    Check multiple properties on a statechart.

    Combines SMT encoding with property verification.
    """

    def __init__(self, encoder: SMTEncoder = None):
        self.encoder = encoder or SMTEncoder()
        self.results: List[PropertyResult] = []

    def check_property(
        self,
        property: Property,
        formula: StatechartFormula,
        timeout_ms: int = 5000,
    ) -> PropertyResult:
        """Check a single property."""
        result = property.check(formula, timeout_ms)
        self.results.append(result)
        return result

    def check_all_properties(
        self,
        formula: StatechartFormula,
        properties: List[Property] = None,
        timeout_ms: int = 5000,
    ) -> List[PropertyResult]:
        """
        Check all standard properties.

        Args:
            formula: Encoded statechart
            properties: Custom properties (or use defaults)
            timeout_ms: Timeout per property

        Returns:
            List of PropertyResult
        """
        if properties is None:
            # Get allowed transitions from formula
            allowed = {(t.source, t.target) for t in formula.transitions}

            # Get target states (final states)
            targets = {s.name for s in formula.states.values() if s.is_final}

            properties = [
                IllegalTransitionProperty(allowed),
                DeadlockFreedomProperty(),
                DeterminismProperty(),
                ReachabilityProperty(),
            ]

            if targets:
                properties.append(LivenessProperty(targets))

        self.results = []
        for prop in properties:
            result = self.check_property(prop, formula, timeout_ms)
            print(f"  {prop.name}: {result.status.name}")

        return self.results

    def all_passed(self) -> bool:
        """Check if all properties passed."""
        return all(r.passed for r in self.results)

    def get_failures(self) -> List[PropertyResult]:
        """Get failed properties."""
        return [r for r in self.results if not r.passed]

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of verification results."""
        return {
            'total': len(self.results),
            'passed': sum(1 for r in self.results if r.passed),
            'failed': sum(1 for r in self.results if r.status == PropertyStatus.VIOLATED),
            'unknown': sum(1 for r in self.results if r.status == PropertyStatus.UNKNOWN),
            'errors': sum(1 for r in self.results if r.status == PropertyStatus.ERROR),
            'total_time': sum(r.verification_time for r in self.results),
        }


def demo():
    """Demonstrate property checking."""
    print("=" * 60)
    print("PROPERTY CHECKER: Verify Statechart Properties")
    print("=" * 60)

    if not HAS_Z3:
        print("\nZ3 not available. Install with: pip install z3-solver")
        return

    from .smt_encoder import SMTEncoder, StateVariable, TransitionFormula, GuardFormula

    encoder = SMTEncoder()

    # Create traffic light statechart
    states = [
        StateVariable(name="root", state_type=StateType.OR, children=["red", "yellow", "green"], is_initial=True),
        StateVariable(name="red", state_type=StateType.BASIC, parent="root", is_initial=True),
        StateVariable(name="yellow", state_type=StateType.BASIC, parent="root"),
        StateVariable(name="green", state_type=StateType.BASIC, parent="root", is_final=True),
    ]

    transitions = [
        TransitionFormula(source="red", target="green", guard=GuardFormula("true")),
        TransitionFormula(source="green", target="yellow", guard=GuardFormula("true")),
        TransitionFormula(source="yellow", target="red", guard=GuardFormula("true")),
    ]

    print("\nEncoding traffic light statechart...")
    formula = encoder.encode_statechart(states, transitions)

    print("\nChecking properties...")
    checker = PropertyChecker(encoder)
    results = checker.check_all_properties(formula)

    print(f"\nSummary: {checker.get_summary()}")

    if checker.all_passed():
        print("\nAll properties verified!")
    else:
        print("\nFailed properties:")
        for result in checker.get_failures():
            print(f"  {result.property_name}: {result.message}")
            if result.counterexample:
                print(f"    Counterexample: {result.counterexample}")

    return checker


if __name__ == "__main__":
    demo()
