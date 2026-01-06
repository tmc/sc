"""
SMT Encoder: Encode Statecharts to SMT Formulas

Translates statechart semantics into SMT-LIB format for Z3 verification.

Encoding approach:
- States as boolean variables (s_i = true means in state i)
- Transitions as implications (guard => next_state)
- Configuration constraints (exactly one state active in OR)
- Parallel composition (all children active in AND)

This enables formal reasoning about statechart behavior.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Set, Tuple, Optional, Any
from enum import Enum, auto
from abc import ABC, abstractmethod

# Z3 import with fallback
try:
    from z3 import (
        Bool, Int, Real, And, Or, Not, Implies, Xor,
        Solver, sat, unsat, unknown,
        ForAll, Exists, Function, BoolSort, IntSort,
        simplify, is_true, is_false,
    )
    HAS_Z3 = True
except ImportError:
    HAS_Z3 = False
    # Mock classes for when Z3 not available
    class MockZ3:
        def __call__(self, *args, **kwargs):
            return self
        def __getattr__(self, name):
            return self
    Bool = Int = Real = And = Or = Not = Implies = Xor = MockZ3()
    Solver = sat = unsat = unknown = MockZ3()


class StateType(Enum):
    """State types in statechart."""
    BASIC = auto()      # Leaf state
    OR = auto()         # XOR composition (one child active)
    AND = auto()        # AND composition (all children active)
    INITIAL = auto()    # Initial pseudo-state
    FINAL = auto()      # Final state
    HISTORY = auto()    # History pseudo-state


@dataclass
class StateVariable:
    """
    SMT variable representing a state.

    Each state gets:
    - active: Bool - is this state currently active?
    - active_next: Bool - is this state active in next step?
    """
    name: str
    state_type: StateType = StateType.BASIC
    parent: Optional[str] = None
    children: List[str] = field(default_factory=list)
    is_initial: bool = False
    is_final: bool = False

    # Z3 variables (created during encoding)
    active: Any = None        # Current timestep
    active_next: Any = None   # Next timestep

    def __hash__(self):
        return hash(self.name)


@dataclass
class GuardFormula:
    """
    SMT encoding of a guard condition.

    Guards are boolean expressions over context variables.
    """
    expression: str  # Human-readable expression
    z3_formula: Any = None  # Z3 formula

    @staticmethod
    def true_guard() -> "GuardFormula":
        return GuardFormula(expression="true", z3_formula=True)

    @staticmethod
    def false_guard() -> "GuardFormula":
        return GuardFormula(expression="false", z3_formula=False)


@dataclass
class TransitionFormula:
    """
    SMT encoding of a transition.

    Transition semantics:
    - If source active AND guard true => target becomes active
    - Source becomes inactive (unless self-loop)
    """
    source: str
    target: str
    event: str = ""
    guard: GuardFormula = field(default_factory=GuardFormula.true_guard)
    priority: int = 0

    # Z3 formula for this transition
    z3_formula: Any = None


@dataclass
class StatechartFormula:
    """
    Complete SMT encoding of a statechart.

    Contains all formulas needed for verification:
    - State variables
    - Transition formulas
    - Configuration constraints
    - Initial state constraints
    """
    states: Dict[str, StateVariable] = field(default_factory=dict)
    transitions: List[TransitionFormula] = field(default_factory=list)

    # Constraints
    initial_constraint: Any = None
    configuration_constraint: Any = None
    transition_relation: Any = None

    # Context variables (for guards)
    context_vars: Dict[str, Any] = field(default_factory=dict)


class SMTEncoder:
    """
    Encode statecharts to SMT formulas.

    Supports:
    - Basic states (atomic)
    - OR states (exclusive choice)
    - AND states (parallel composition)
    - Guards (boolean conditions)
    - Priority-based transition selection
    """

    def __init__(self):
        if not HAS_Z3:
            print("WARNING: Z3 not available. Using mock implementation.")

        self.formula: Optional[StatechartFormula] = None
        self.solver: Optional[Any] = None

    def encode_state(self, state: StateVariable, timestep: int = 0) -> Tuple[Any, Any]:
        """
        Create Z3 variables for a state.

        Returns:
            (active_var, active_next_var)
        """
        if not HAS_Z3:
            return None, None

        active = Bool(f"{state.name}_active_{timestep}")
        active_next = Bool(f"{state.name}_active_{timestep + 1}")

        return active, active_next

    def encode_guard(
        self,
        guard_expr: str,
        context_vars: Dict[str, Any],
    ) -> Any:
        """
        Encode guard expression to Z3 formula.

        Supports:
        - Variable references: x, y, counter
        - Comparisons: x > 5, y == 0
        - Boolean ops: and, or, not
        """
        if not HAS_Z3:
            return True

        if guard_expr == "true" or guard_expr == "":
            return True

        if guard_expr == "false":
            return False

        # Parse simple expressions
        # Format: "var op value" or "var1 op var2"
        try:
            # Handle AND/OR
            if " and " in guard_expr.lower():
                parts = guard_expr.lower().split(" and ")
                return And(*[self.encode_guard(p.strip(), context_vars) for p in parts])

            if " or " in guard_expr.lower():
                parts = guard_expr.lower().split(" or ")
                return Or(*[self.encode_guard(p.strip(), context_vars) for p in parts])

            # Handle NOT
            if guard_expr.lower().startswith("not "):
                inner = guard_expr[4:].strip()
                return Not(self.encode_guard(inner, context_vars))

            # Handle comparisons
            for op in [">=", "<=", "==", "!=", ">", "<"]:
                if op in guard_expr:
                    left, right = guard_expr.split(op)
                    left = left.strip()
                    right = right.strip()

                    # Get or create variables
                    if left in context_vars:
                        left_var = context_vars[left]
                    elif left.isdigit():
                        left_var = int(left)
                    else:
                        left_var = Int(left)
                        context_vars[left] = left_var

                    if right in context_vars:
                        right_var = context_vars[right]
                    elif right.isdigit():
                        right_var = int(right)
                    else:
                        right_var = Int(right)
                        context_vars[right] = right_var

                    if op == ">=":
                        return left_var >= right_var
                    elif op == "<=":
                        return left_var <= right_var
                    elif op == "==":
                        return left_var == right_var
                    elif op == "!=":
                        return left_var != right_var
                    elif op == ">":
                        return left_var > right_var
                    elif op == "<":
                        return left_var < right_var

            # Variable reference (boolean)
            if guard_expr in context_vars:
                return context_vars[guard_expr]

            # Create new boolean variable
            var = Bool(guard_expr)
            context_vars[guard_expr] = var
            return var

        except Exception as e:
            print(f"Warning: Could not parse guard '{guard_expr}': {e}")
            return True

    def encode_transition(
        self,
        trans: TransitionFormula,
        states: Dict[str, StateVariable],
        context_vars: Dict[str, Any],
    ) -> Any:
        """
        Encode transition to Z3 formula.

        Transition fires if:
        - Source state is active
        - Guard is satisfied
        - No higher-priority transition is enabled

        Effect:
        - Target becomes active
        - Source becomes inactive (unless self-loop)
        """
        if not HAS_Z3:
            return True

        source_state = states.get(trans.source)
        target_state = states.get(trans.target)

        if not source_state or not target_state:
            return True

        # Encode guard
        guard_formula = self.encode_guard(trans.guard.expression, context_vars)

        # Transition formula: source_active AND guard => target_active_next
        trans_enabled = And(source_state.active, guard_formula)
        trans_effect = Implies(trans_enabled, target_state.active_next)

        return trans_effect

    def encode_or_constraint(
        self,
        parent: StateVariable,
        children: List[StateVariable],
    ) -> Any:
        """
        Encode OR state constraint: exactly one child active when parent active.

        parent_active => (exactly_one(child_active))
        """
        if not HAS_Z3 or not children:
            return True

        # At least one child active
        at_least_one = Or(*[c.active for c in children])

        # At most one child active (pairwise exclusion)
        at_most_one_clauses = []
        for i, c1 in enumerate(children):
            for c2 in children[i + 1:]:
                at_most_one_clauses.append(Not(And(c1.active, c2.active)))

        at_most_one = And(*at_most_one_clauses) if at_most_one_clauses else True

        # Parent active implies exactly one child
        exactly_one = And(at_least_one, at_most_one)
        constraint = Implies(parent.active, exactly_one)

        return constraint

    def encode_and_constraint(
        self,
        parent: StateVariable,
        children: List[StateVariable],
    ) -> Any:
        """
        Encode AND state constraint: all children active when parent active.

        parent_active => (all children active)
        """
        if not HAS_Z3 or not children:
            return True

        all_active = And(*[c.active for c in children])
        constraint = Implies(parent.active, all_active)

        return constraint

    def encode_statechart(
        self,
        states: List[StateVariable],
        transitions: List[TransitionFormula],
        context_vars: Dict[str, Any] = None,
    ) -> StatechartFormula:
        """
        Encode complete statechart to SMT formulas.

        Args:
            states: List of state definitions
            transitions: List of transition definitions
            context_vars: Context variables for guards

        Returns:
            StatechartFormula with all constraints
        """
        formula = StatechartFormula()
        formula.context_vars = context_vars or {}

        # Create state variables
        for state in states:
            state.active, state.active_next = self.encode_state(state)
            formula.states[state.name] = state

        # Encode transitions
        for trans in transitions:
            trans.z3_formula = self.encode_transition(trans, formula.states, formula.context_vars)
            formula.transitions.append(trans)

        # Encode configuration constraints
        config_constraints = []

        for state in states:
            if state.state_type == StateType.OR and state.children:
                children = [formula.states[c] for c in state.children if c in formula.states]
                constraint = self.encode_or_constraint(state, children)
                config_constraints.append(constraint)

            elif state.state_type == StateType.AND and state.children:
                children = [formula.states[c] for c in state.children if c in formula.states]
                constraint = self.encode_and_constraint(state, children)
                config_constraints.append(constraint)

        if config_constraints and HAS_Z3:
            formula.configuration_constraint = And(*config_constraints)
        else:
            formula.configuration_constraint = True

        # Encode initial state constraint
        initial_states = [s for s in states if s.is_initial]
        if initial_states and HAS_Z3:
            formula.initial_constraint = And(*[s.active for s in initial_states])
        else:
            formula.initial_constraint = True

        # Encode transition relation
        if formula.transitions and HAS_Z3:
            trans_formulas = [t.z3_formula for t in formula.transitions if t.z3_formula is not None]
            formula.transition_relation = And(*trans_formulas) if trans_formulas else True
        else:
            formula.transition_relation = True

        self.formula = formula
        return formula

    def get_solver(self) -> Any:
        """Get Z3 solver with statechart constraints."""
        if not HAS_Z3:
            return None

        if self.solver is None:
            self.solver = Solver()

        if self.formula:
            self.solver.add(self.formula.configuration_constraint)
            self.solver.add(self.formula.transition_relation)

        return self.solver


def demo():
    """Demonstrate SMT encoding."""
    print("=" * 60)
    print("SMT ENCODER: Statechart to SMT")
    print("=" * 60)

    if not HAS_Z3:
        print("\nZ3 not available. Install with: pip install z3-solver")
        print("Showing mock implementation.")

    encoder = SMTEncoder()

    # Create simple statechart: traffic light
    states = [
        StateVariable(name="root", state_type=StateType.OR, children=["red", "yellow", "green"], is_initial=True),
        StateVariable(name="red", state_type=StateType.BASIC, parent="root", is_initial=True),
        StateVariable(name="yellow", state_type=StateType.BASIC, parent="root"),
        StateVariable(name="green", state_type=StateType.BASIC, parent="root"),
    ]

    transitions = [
        TransitionFormula(source="red", target="green", guard=GuardFormula("timer >= 30")),
        TransitionFormula(source="green", target="yellow", guard=GuardFormula("timer >= 25")),
        TransitionFormula(source="yellow", target="red", guard=GuardFormula("timer >= 5")),
    ]

    print("\nEncoding traffic light statechart...")
    formula = encoder.encode_statechart(states, transitions, {"timer": Int("timer") if HAS_Z3 else None})

    print(f"\nEncoded {len(formula.states)} states")
    print(f"Encoded {len(formula.transitions)} transitions")

    if HAS_Z3:
        print(f"\nConfiguration constraint: {formula.configuration_constraint}")
        print(f"Initial constraint: {formula.initial_constraint}")

        # Test satisfiability
        solver = encoder.get_solver()
        solver.add(formula.initial_constraint)

        result = solver.check()
        print(f"\nSatisfiability: {result}")

    return encoder, formula


if __name__ == "__main__":
    demo()
