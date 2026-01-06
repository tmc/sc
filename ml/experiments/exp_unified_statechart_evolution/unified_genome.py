"""
Unified Statechart Genome

The complete evolvable representation of a statechart, combining:
- Topology (states, hierarchy)
- History types (NONE/SHALLOW/DEEP)
- Guards (conditions with specificity)
- Priorities (conflict resolution)
- Actions (side effects)
- Events (triggers)

This is the FLAGSHIP genome that unifies all statechart components.
"""

import random
import copy
from dataclasses import dataclass, field
from typing import List, Dict, Set, Tuple, Optional, Any
from enum import IntEnum, Enum, auto


# =============================================================================
# Core Enums
# =============================================================================

class StateType(IntEnum):
    """State decomposition type."""
    BASIC = 0   # Leaf state (no children)
    OR = 1      # Exclusive children (XOR semantics)
    AND = 2     # Parallel children (AND semantics)


class HistoryType(IntEnum):
    """History restoration strategy."""
    NONE = 0      # No history - enter via initial
    SHALLOW = 1   # H - restore direct child only
    DEEP = 2      # H* - restore entire nested configuration


class ActionType(Enum):
    """Types of actions on context."""
    SET = auto()        # Set variable to value
    INCREMENT = auto()  # Increment numeric variable
    DECREMENT = auto()  # Decrement numeric variable
    TOGGLE = auto()     # Toggle boolean
    APPEND = auto()     # Append to list
    CLEAR = auto()      # Clear/reset variable


class GuardOp(Enum):
    """Guard comparison operators."""
    TRUE = auto()       # Always true
    FALSE = auto()      # Always false
    EQ = auto()         # ==
    NE = auto()         # !=
    LT = auto()         # <
    LE = auto()         # <=
    GT = auto()         # >
    GE = auto()         # >=
    IN = auto()         # In set/list
    HAS = auto()        # Has property


# =============================================================================
# Guard (Evolvable)
# =============================================================================

@dataclass
class UnifiedGuard:
    """
    Evolvable guard condition.

    Combines:
    - Variable reference
    - Comparison operator
    - Value to compare
    - Computed specificity (for priority)
    """
    variable: str = "true"
    op: GuardOp = GuardOp.TRUE
    value: Any = None

    # Cached specificity (computed from op and variable)
    _specificity: Optional[float] = None

    def evaluate(self, context: Dict[str, Any]) -> bool:
        """Evaluate guard against context."""
        if self.op == GuardOp.TRUE:
            return True
        if self.op == GuardOp.FALSE:
            return False

        var_value = context.get(self.variable)
        if var_value is None:
            return False

        if self.op == GuardOp.EQ:
            return var_value == self.value
        elif self.op == GuardOp.NE:
            return var_value != self.value
        elif self.op == GuardOp.LT:
            return var_value < self.value
        elif self.op == GuardOp.LE:
            return var_value <= self.value
        elif self.op == GuardOp.GT:
            return var_value > self.value
        elif self.op == GuardOp.GE:
            return var_value >= self.value
        elif self.op == GuardOp.IN:
            # value should be a collection; if it's a scalar, wrap it
            if isinstance(self.value, (list, tuple, set)):
                return var_value in self.value
            elif self.value is not None:
                return var_value == self.value
            return False
        elif self.op == GuardOp.HAS:
            return self.value in var_value if hasattr(var_value, '__contains__') else False

        return True

    @property
    def specificity(self) -> float:
        """
        Compute guard specificity (0-1).
        Higher = more restrictive = higher priority in conflicts.
        """
        if self._specificity is not None:
            return self._specificity

        # Base specificity by operator
        op_specificity = {
            GuardOp.TRUE: 0.0,   # Least specific
            GuardOp.FALSE: 1.0,  # Most specific (useless)
            GuardOp.EQ: 0.9,     # Very specific
            GuardOp.NE: 0.3,     # Not very specific
            GuardOp.LT: 0.5,
            GuardOp.LE: 0.4,
            GuardOp.GT: 0.5,
            GuardOp.GE: 0.4,
            GuardOp.IN: 0.7,
            GuardOp.HAS: 0.6,
        }

        return op_specificity.get(self.op, 0.5)

    def copy(self) -> 'UnifiedGuard':
        return UnifiedGuard(
            variable=self.variable,
            op=self.op,
            value=copy.deepcopy(self.value)
        )

    def __repr__(self):
        if self.op == GuardOp.TRUE:
            return "true"
        if self.op == GuardOp.FALSE:
            return "false"
        op_str = {
            GuardOp.EQ: "==", GuardOp.NE: "!=",
            GuardOp.LT: "<", GuardOp.LE: "<=",
            GuardOp.GT: ">", GuardOp.GE: ">=",
            GuardOp.IN: "in", GuardOp.HAS: "has"
        }.get(self.op, "?")
        return f"{self.variable} {op_str} {self.value}"


# =============================================================================
# Action (Evolvable)
# =============================================================================

@dataclass
class UnifiedAction:
    """
    Evolvable action that modifies context.

    Actions are executed when transitions fire.
    """
    action_type: ActionType = ActionType.SET
    variable: str = "x"
    value: Any = None

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute action on context, return modified context."""
        result = context.copy()

        if self.action_type == ActionType.SET:
            result[self.variable] = self.value
        elif self.action_type == ActionType.INCREMENT:
            result[self.variable] = result.get(self.variable, 0) + (self.value or 1)
        elif self.action_type == ActionType.DECREMENT:
            result[self.variable] = result.get(self.variable, 0) - (self.value or 1)
        elif self.action_type == ActionType.TOGGLE:
            result[self.variable] = not result.get(self.variable, False)
        elif self.action_type == ActionType.APPEND:
            lst = result.get(self.variable, [])
            if isinstance(lst, list):
                result[self.variable] = lst + [self.value]
        elif self.action_type == ActionType.CLEAR:
            result[self.variable] = None

        return result

    def copy(self) -> 'UnifiedAction':
        return UnifiedAction(
            action_type=self.action_type,
            variable=self.variable,
            value=copy.deepcopy(self.value)
        )

    def __repr__(self):
        type_str = {
            ActionType.SET: "=",
            ActionType.INCREMENT: "+=",
            ActionType.DECREMENT: "-=",
            ActionType.TOGGLE: "toggle",
            ActionType.APPEND: "append",
            ActionType.CLEAR: "clear"
        }.get(self.action_type, "?")
        return f"{self.variable} {type_str} {self.value}"


# =============================================================================
# Transition (Evolvable)
# =============================================================================

@dataclass
class UnifiedTransition:
    """
    Complete evolvable transition with all components.

    Combines:
    - Source/target states
    - Event trigger
    - Guard condition
    - Priority for conflict resolution
    - Actions to execute
    """
    src: int                           # Source state index
    tgt: int                           # Target state index
    event: int                         # Event index
    guard: UnifiedGuard = field(default_factory=UnifiedGuard)
    priority: float = 0.0              # Explicit priority
    actions: List[UnifiedAction] = field(default_factory=list)

    def is_enabled(self, current_state: int, event: int, context: Dict[str, Any]) -> bool:
        """Check if transition is enabled."""
        if self.src != current_state:
            return False
        if self.event != event:
            return False
        return self.guard.evaluate(context)

    def effective_priority(self, use_specificity: bool = True) -> float:
        """
        Compute effective priority.

        Can combine explicit priority with guard specificity.
        """
        if use_specificity:
            return self.priority + self.guard.specificity * 0.5
        return self.priority

    def copy(self) -> 'UnifiedTransition':
        return UnifiedTransition(
            src=self.src,
            tgt=self.tgt,
            event=self.event,
            guard=self.guard.copy(),
            priority=self.priority,
            actions=[a.copy() for a in self.actions]
        )


# =============================================================================
# Unified Genome
# =============================================================================

@dataclass
class UnifiedGenome:
    """
    THE FLAGSHIP GENOME: Complete evolvable statechart.

    Combines ALL statechart components:
    - Topology: n_states, parent[], state_type[]
    - History: history_type[]
    - Transitions: UnifiedTransition[] with guards, priorities, actions
    - Events: event labels
    - Context variables: initial context

    This is what we evolve to discover optimal statechart configurations.
    """
    # === TOPOLOGY GENES ===
    n_states: int = 3
    parent: List[int] = field(default_factory=lambda: [-1, 0, 0])
    state_type: List[StateType] = field(default_factory=lambda: [StateType.OR, StateType.BASIC, StateType.BASIC])
    state_labels: List[str] = field(default_factory=list)  # Optional labels

    # === HISTORY GENES ===
    history_type: List[HistoryType] = field(default_factory=list)

    # === TRANSITION GENES ===
    transitions: List[UnifiedTransition] = field(default_factory=list)

    # === EVENT GENES ===
    n_events: int = 5
    event_labels: List[str] = field(default_factory=list)

    # === CONTEXT GENES ===
    initial_context: Dict[str, Any] = field(default_factory=dict)
    context_variables: List[str] = field(default_factory=list)

    # === CONFIGURATION ===
    initial_state: int = 1
    max_states: int = 20
    max_transitions: int = 50
    max_actions_per_transition: int = 3

    # === FITNESS ===
    fitness: float = 0.0
    accuracy: float = 0.0
    complexity: float = 0.0

    def __post_init__(self):
        """Initialize derived fields."""
        # Initialize history types if not set
        if not self.history_type:
            self.history_type = [HistoryType.NONE] * self.n_states

        # Initialize state labels if not set
        if not self.state_labels:
            self.state_labels = [f"S{i}" for i in range(self.n_states)]

        # Initialize event labels if not set
        if not self.event_labels:
            self.event_labels = [f"E{i}" for i in range(self.n_events)]

        # Validate sizes
        self._validate_sizes()

    def _validate_sizes(self):
        """Ensure all arrays are correctly sized."""
        while len(self.parent) < self.n_states:
            self.parent.append(0)
        while len(self.state_type) < self.n_states:
            self.state_type.append(StateType.BASIC)
        while len(self.history_type) < self.n_states:
            self.history_type.append(HistoryType.NONE)
        while len(self.state_labels) < self.n_states:
            self.state_labels.append(f"S{len(self.state_labels)}")

    # === TOPOLOGY QUERIES ===

    def get_children(self, state_idx: int) -> List[int]:
        """Get children of a state."""
        return [i for i in range(self.n_states) if self.parent[i] == state_idx]

    def get_leaves(self) -> List[int]:
        """Get leaf states."""
        return [i for i in range(self.n_states)
                if self.state_type[i] == StateType.BASIC or not self.get_children(i)]

    def get_depth(self, state_idx: int) -> int:
        """Get depth in hierarchy."""
        depth = 0
        curr = state_idx
        while self.parent[curr] != -1:
            curr = self.parent[curr]
            depth += 1
        return depth

    def get_max_depth(self) -> int:
        """Get maximum hierarchy depth."""
        return max(self.get_depth(i) for i in range(self.n_states))

    def is_composite(self, state_idx: int) -> bool:
        """Check if state has children."""
        return len(self.get_children(state_idx)) > 0

    def get_ancestors(self, state_idx: int) -> List[int]:
        """Get all ancestors of a state."""
        ancestors = []
        curr = self.parent[state_idx]
        while curr != -1:
            ancestors.append(curr)
            curr = self.parent[curr]
        return ancestors

    # === VALIDATION ===

    def is_valid(self) -> bool:
        """Check if genome is valid."""
        # Check for cycles
        for i in range(self.n_states):
            visited = set()
            curr = i
            while curr != -1:
                if curr in visited:
                    return False
                visited.add(curr)
                curr = self.parent[curr]

        # Check root
        if self.parent[0] != -1:
            return False

        # Check transitions
        for t in self.transitions:
            if t.src < 0 or t.src >= self.n_states:
                return False
            if t.tgt < 0 or t.tgt >= self.n_states:
                return False
            if t.event < 0 or t.event >= self.n_events:
                return False

        return True

    # === COMPLEXITY METRICS ===

    def compute_complexity(self) -> float:
        """Compute structural complexity (lower is simpler)."""
        # Weighted sum of components
        n_transitions = len(self.transitions)
        n_actions = sum(len(t.actions) for t in self.transitions)
        n_guards = sum(1 for t in self.transitions if t.guard.op != GuardOp.TRUE)
        n_history = sum(1 for h in self.history_type if h != HistoryType.NONE)
        n_parallel = sum(1 for s in self.state_type if s == StateType.AND)

        complexity = (
            self.n_states * 1.0 +
            n_transitions * 0.5 +
            n_actions * 0.3 +
            n_guards * 0.2 +
            n_history * 0.1 +
            n_parallel * 0.4
        )

        return complexity

    def count_components(self) -> Dict[str, int]:
        """Count all genome components."""
        return {
            'states': self.n_states,
            'transitions': len(self.transitions),
            'actions': sum(len(t.actions) for t in self.transitions),
            'guards': sum(1 for t in self.transitions if t.guard.op != GuardOp.TRUE),
            'history_shallow': sum(1 for h in self.history_type if h == HistoryType.SHALLOW),
            'history_deep': sum(1 for h in self.history_type if h == HistoryType.DEEP),
            'parallel_states': sum(1 for s in self.state_type if s == StateType.AND),
            'max_depth': self.get_max_depth(),
        }

    # === COPY ===

    def copy(self) -> 'UnifiedGenome':
        """Deep copy."""
        return UnifiedGenome(
            n_states=self.n_states,
            parent=self.parent.copy(),
            state_type=self.state_type.copy(),
            state_labels=self.state_labels.copy(),
            history_type=self.history_type.copy(),
            transitions=[t.copy() for t in self.transitions],
            n_events=self.n_events,
            event_labels=self.event_labels.copy(),
            initial_context=copy.deepcopy(self.initial_context),
            context_variables=self.context_variables.copy(),
            initial_state=self.initial_state,
            max_states=self.max_states,
            max_transitions=self.max_transitions,
            max_actions_per_transition=self.max_actions_per_transition,
            fitness=self.fitness,
            accuracy=self.accuracy,
            complexity=self.complexity
        )


# =============================================================================
# Genome Creation
# =============================================================================

def create_random_unified_genome(
    max_states: int = 10,
    n_events: int = 5,
    context_variables: List[str] = None,
    include_history: bool = True,
    include_actions: bool = True,
    include_guards: bool = True
) -> UnifiedGenome:
    """
    Create a random unified genome with all components.

    Args:
        max_states: Maximum number of states
        n_events: Number of events
        context_variables: Variables for guards/actions
        include_history: Whether to include history genes
        include_actions: Whether to include action genes
        include_guards: Whether to include guard genes
    """
    context_variables = context_variables or ['x', 'y', 'count', 'mode', 'flag']

    # Random state count
    n_states = random.randint(3, max_states)

    # Build tree hierarchy
    parent = [-1]
    for i in range(1, n_states):
        parent.append(random.randint(0, i - 1))

    # Assign state types
    state_type = []
    for i in range(n_states):
        children = [j for j in range(n_states) if parent[j] == i]
        if children:
            state_type.append(random.choice([StateType.OR, StateType.AND]))
        else:
            state_type.append(StateType.BASIC)

    # History types
    history_type = []
    for i in range(n_states):
        if include_history and state_type[i] != StateType.BASIC:
            r = random.random()
            if r < 0.6:
                history_type.append(HistoryType.NONE)
            elif r < 0.8:
                history_type.append(HistoryType.SHALLOW)
            else:
                history_type.append(HistoryType.DEEP)
        else:
            history_type.append(HistoryType.NONE)

    # Transitions
    leaves = [i for i in range(n_states) if state_type[i] == StateType.BASIC]
    transitions = []

    n_transitions = random.randint(len(leaves), len(leaves) * 3)
    for _ in range(n_transitions):
        src = random.choice(leaves)
        tgt = random.choice(leaves)
        event = random.randint(0, n_events - 1)

        # Guard
        if include_guards and random.random() < 0.4:
            var = random.choice(context_variables)
            op = random.choice([GuardOp.EQ, GuardOp.GT, GuardOp.LT, GuardOp.GE, GuardOp.LE])
            value = random.randint(0, 10)
            guard = UnifiedGuard(variable=var, op=op, value=value)
        else:
            guard = UnifiedGuard()

        # Priority
        priority = random.uniform(0, 1)

        # Actions
        actions = []
        if include_actions and random.random() < 0.3:
            n_actions = random.randint(1, 2)
            for _ in range(n_actions):
                var = random.choice(context_variables)
                action_type = random.choice([ActionType.SET, ActionType.INCREMENT, ActionType.TOGGLE])
                value = random.randint(0, 5) if action_type != ActionType.TOGGLE else None
                actions.append(UnifiedAction(action_type=action_type, variable=var, value=value))

        transitions.append(UnifiedTransition(
            src=src, tgt=tgt, event=event,
            guard=guard, priority=priority, actions=actions
        ))

    # Initial state
    initial_state = leaves[0] if leaves else 1

    # Initial context
    initial_context = {var: 0 for var in context_variables}

    return UnifiedGenome(
        n_states=n_states,
        parent=parent,
        state_type=state_type,
        history_type=history_type,
        transitions=transitions,
        n_events=n_events,
        initial_context=initial_context,
        context_variables=context_variables,
        initial_state=initial_state,
        max_states=max_states
    )


# =============================================================================
# Testing
# =============================================================================

def test_unified_genome():
    """Test unified genome creation and operations."""
    print("=" * 60)
    print("UNIFIED GENOME TEST")
    print("=" * 60)

    genome = create_random_unified_genome(
        max_states=8,
        n_events=5,
        include_history=True,
        include_actions=True,
        include_guards=True
    )

    print(f"States: {genome.n_states}")
    print(f"Transitions: {len(genome.transitions)}")
    print(f"Valid: {genome.is_valid()}")
    print(f"Max depth: {genome.get_max_depth()}")
    print(f"Complexity: {genome.compute_complexity():.2f}")

    counts = genome.count_components()
    print(f"\nComponents:")
    for k, v in counts.items():
        print(f"  {k}: {v}")

    print("\nSample transitions:")
    for t in genome.transitions[:3]:
        print(f"  {genome.state_labels[t.src]} -> {genome.state_labels[t.tgt]} "
              f"on E{t.event} [{t.guard}] (pri={t.priority:.2f})")
        for a in t.actions:
            print(f"    action: {a}")

    print("\nUnified genome test complete!")
    return genome


if __name__ == "__main__":
    test_unified_genome()
