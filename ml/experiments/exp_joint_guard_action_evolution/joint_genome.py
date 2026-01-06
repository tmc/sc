"""
Joint Guard-Action Genome

Co-evolve guards and actions together. The key insight:
- Guards CHECK conditions that actions CREATE
- Actions MODIFY variables that guards TEST

This creates a causal dependency that must be respected:
  action(t) sets X → guard(t+1) checks X

Joint evolution ensures coherence: guards and actions evolve together
to form consistent transition rules.
"""

import mlx.core as mx
from dataclasses import dataclass, field
from typing import List, Dict, Set, Tuple, Optional, Any, FrozenSet
from enum import Enum, auto
from abc import ABC, abstractmethod
import random
import copy


# =============================================================================
# Guard Expressions
# =============================================================================

class GuardOp(Enum):
    """Guard comparison operators."""
    EQ = auto()    # ==
    NE = auto()    # !=
    LT = auto()    # <
    LE = auto()    # <=
    GT = auto()    # >
    GE = auto()    # >=
    IN = auto()    # membership
    TRUE = auto()  # always true
    FALSE = auto() # always false


@dataclass
class GuardClause:
    """A single guard clause: var op value."""
    variable: str
    op: GuardOp
    value: Any

    def evaluate(self, context: Dict[str, Any]) -> bool:
        """Evaluate clause against context."""
        if self.op == GuardOp.TRUE:
            return True
        if self.op == GuardOp.FALSE:
            return False

        var_value = context.get(self.variable)
        if var_value is None:
            return False

        ops = {
            GuardOp.EQ: lambda a, b: a == b,
            GuardOp.NE: lambda a, b: a != b,
            GuardOp.LT: lambda a, b: a < b,
            GuardOp.LE: lambda a, b: a <= b,
            GuardOp.GT: lambda a, b: a > b,
            GuardOp.GE: lambda a, b: a >= b,
            GuardOp.IN: lambda a, b: a in b if isinstance(b, (list, tuple, set)) else a == b,
        }
        try:
            return ops[self.op](var_value, self.value)
        except (TypeError, KeyError):
            return False

    def variables_read(self) -> Set[str]:
        """Variables this clause reads."""
        if self.op in (GuardOp.TRUE, GuardOp.FALSE):
            return set()
        return {self.variable}

    def to_string(self) -> str:
        op_str = {
            GuardOp.EQ: "==", GuardOp.NE: "!=",
            GuardOp.LT: "<", GuardOp.LE: "<=",
            GuardOp.GT: ">", GuardOp.GE: ">=",
            GuardOp.IN: "in", GuardOp.TRUE: "TRUE", GuardOp.FALSE: "FALSE"
        }
        if self.op in (GuardOp.TRUE, GuardOp.FALSE):
            return op_str[self.op]
        return f"{self.variable} {op_str[self.op]} {self.value}"


class GuardCombinator(Enum):
    """How to combine clauses."""
    AND = auto()  # All must be true
    OR = auto()   # Any must be true


@dataclass
class Guard:
    """A guard with multiple clauses combined by AND/OR."""
    clauses: List[GuardClause]
    combinator: GuardCombinator = GuardCombinator.AND

    def evaluate(self, context: Dict[str, Any]) -> bool:
        """Evaluate full guard."""
        if not self.clauses:
            return True

        results = [c.evaluate(context) for c in self.clauses]
        if self.combinator == GuardCombinator.AND:
            return all(results)
        else:
            return any(results)

    def variables_read(self) -> Set[str]:
        """All variables this guard reads."""
        vars_read = set()
        for clause in self.clauses:
            vars_read |= clause.variables_read()
        return vars_read

    def to_string(self) -> str:
        if not self.clauses:
            return "TRUE"
        joiner = " AND " if self.combinator == GuardCombinator.AND else " OR "
        return joiner.join(c.to_string() for c in self.clauses)


# =============================================================================
# Action Effects
# =============================================================================

class EffectOp(Enum):
    """Types of effects on variables."""
    SET = auto()        # x := v
    INCREMENT = auto()  # x := x + v
    DECREMENT = auto()  # x := x - v
    TOGGLE = auto()     # x := !x
    MULTIPLY = auto()   # x := x * v
    APPEND = auto()     # x.append(v)
    CLEAR = auto()      # x := default


@dataclass
class ActionEffect:
    """A single effect on a variable."""
    variable: str
    effect_op: EffectOp
    value: Any = None

    def apply(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Apply effect to context."""
        ctx = dict(context)
        current = ctx.get(self.variable, 0)

        try:
            if self.effect_op == EffectOp.SET:
                ctx[self.variable] = self.value
            elif self.effect_op == EffectOp.INCREMENT:
                if isinstance(current, (int, float)):
                    ctx[self.variable] = current + (self.value if self.value is not None else 1)
            elif self.effect_op == EffectOp.DECREMENT:
                if isinstance(current, (int, float)):
                    ctx[self.variable] = current - (self.value if self.value is not None else 1)
            elif self.effect_op == EffectOp.TOGGLE:
                ctx[self.variable] = not current
            elif self.effect_op == EffectOp.MULTIPLY:
                if isinstance(current, (int, float)):
                    ctx[self.variable] = current * (self.value if self.value is not None else 1)
            elif self.effect_op == EffectOp.APPEND:
                if isinstance(current, list):
                    ctx[self.variable] = current + [self.value]
                else:
                    ctx[self.variable] = [self.value]
            elif self.effect_op == EffectOp.CLEAR:
                ctx[self.variable] = 0 if isinstance(current, (int, float)) else None
        except (TypeError, ValueError):
            pass  # Skip invalid operations

        return ctx

    def variables_written(self) -> Set[str]:
        """Variables this effect writes."""
        return {self.variable}

    def variables_read(self) -> Set[str]:
        """Variables this effect reads (for increment, etc.)."""
        if self.effect_op in (EffectOp.INCREMENT, EffectOp.DECREMENT,
                              EffectOp.TOGGLE, EffectOp.MULTIPLY, EffectOp.APPEND):
            return {self.variable}
        return set()

    def to_string(self) -> str:
        op_str = {
            EffectOp.SET: ":=", EffectOp.INCREMENT: "+=",
            EffectOp.DECREMENT: "-=", EffectOp.TOGGLE: "toggle",
            EffectOp.MULTIPLY: "*=", EffectOp.APPEND: "append",
            EffectOp.CLEAR: "clear"
        }
        if self.effect_op in (EffectOp.TOGGLE, EffectOp.CLEAR):
            return f"{op_str[self.effect_op]}({self.variable})"
        return f"{self.variable} {op_str[self.effect_op]} {self.value}"


@dataclass
class Action:
    """An action with multiple effects."""
    name: str
    effects: List[ActionEffect] = field(default_factory=list)

    def apply(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Apply all effects."""
        ctx = dict(context)
        for effect in self.effects:
            ctx = effect.apply(ctx)
        return ctx

    def variables_written(self) -> Set[str]:
        """All variables this action writes."""
        written = set()
        for effect in self.effects:
            written |= effect.variables_written()
        return written

    def variables_read(self) -> Set[str]:
        """All variables this action reads."""
        read = set()
        for effect in self.effects:
            read |= effect.variables_read()
        return read

    def to_string(self) -> str:
        if not self.effects:
            return f"{self.name}(NOOP)"
        effects_str = "; ".join(e.to_string() for e in self.effects)
        return f"{self.name}[{effects_str}]"


# =============================================================================
# Joint Guard-Action Pair
# =============================================================================

@dataclass
class GuardActionPair:
    """A transition's guard and action together.

    This is the fundamental unit of co-evolution:
    - guard determines WHEN this transition fires
    - action determines WHAT happens when it fires

    Coherence: guard should check variables that previous actions set,
    and action should set variables that future guards check.
    """
    guard: Guard
    action: Action
    source_state: int = 0
    target_state: int = 0
    event: int = 0
    priority: int = 0

    def is_enabled(self, context: Dict[str, Any]) -> bool:
        """Check if transition is enabled."""
        return self.guard.evaluate(context)

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute action if guard passes."""
        if self.is_enabled(context):
            return self.action.apply(context)
        return context

    def guard_reads(self) -> Set[str]:
        """Variables the guard reads."""
        return self.guard.variables_read()

    def action_writes(self) -> Set[str]:
        """Variables the action writes."""
        return self.action.variables_written()

    def causal_dependency_strength(self, other: 'GuardActionPair') -> float:
        """How strongly this pair depends on other.

        Dependency: other.action writes what this.guard reads.
        """
        other_writes = other.action_writes()
        this_reads = self.guard_reads()

        if not this_reads or not other_writes:
            return 0.0

        overlap = this_reads & other_writes
        return len(overlap) / len(this_reads)

    def to_string(self) -> str:
        return f"[{self.guard.to_string()}] -> {self.action.to_string()}"


# =============================================================================
# Joint Genome
# =============================================================================

@dataclass
class JointGenome:
    """Genome encoding guard-action pairs for all transitions.

    The genome is a collection of GuardActionPairs that together
    define the complete transition behavior of a statechart.
    """
    pairs: List[GuardActionPair]
    n_states: int
    n_events: int
    variables: Set[str] = field(default_factory=set)

    def get_transitions_for_state(self, state: int) -> List[GuardActionPair]:
        """Get all transitions from a state."""
        return [p for p in self.pairs if p.source_state == state]

    def get_transitions_for_event(self, event: int) -> List[GuardActionPair]:
        """Get all transitions triggered by event."""
        return [p for p in self.pairs if p.event == event]

    def all_variables_read(self) -> Set[str]:
        """All variables read by guards."""
        vars_read = set()
        for pair in self.pairs:
            vars_read |= pair.guard_reads()
        return vars_read

    def all_variables_written(self) -> Set[str]:
        """All variables written by actions."""
        vars_written = set()
        for pair in self.pairs:
            vars_written |= pair.action_writes()
        return vars_written

    def variable_coverage(self) -> float:
        """How many guard-read variables are action-written."""
        reads = self.all_variables_read()
        writes = self.all_variables_written()

        if not reads:
            return 1.0

        covered = reads & writes
        return len(covered) / len(reads)

    def copy(self) -> 'JointGenome':
        """Deep copy."""
        return JointGenome(
            pairs=[copy.deepcopy(p) for p in self.pairs],
            n_states=self.n_states,
            n_events=self.n_events,
            variables=set(self.variables)
        )


# =============================================================================
# Random Generation
# =============================================================================

def random_guard_clause(
    variables: List[str],
    possible_values: Dict[str, List[Any]]
) -> GuardClause:
    """Generate random guard clause."""
    if random.random() < 0.1:
        return GuardClause("", GuardOp.TRUE, None)

    var = random.choice(variables)
    op = random.choice([GuardOp.EQ, GuardOp.NE, GuardOp.LT, GuardOp.LE,
                        GuardOp.GT, GuardOp.GE])

    values = possible_values.get(var, [0, 1, 2, True, False])
    value = random.choice(values)

    return GuardClause(var, op, value)


def random_guard(
    variables: List[str],
    possible_values: Dict[str, List[Any]],
    max_clauses: int = 3
) -> Guard:
    """Generate random guard."""
    n_clauses = random.randint(1, max_clauses)
    clauses = [random_guard_clause(variables, possible_values)
               for _ in range(n_clauses)]
    combinator = random.choice([GuardCombinator.AND, GuardCombinator.OR])
    return Guard(clauses, combinator)


def random_action_effect(
    variables: List[str],
    possible_values: Dict[str, List[Any]]
) -> ActionEffect:
    """Generate random action effect."""
    var = random.choice(variables)
    op = random.choice([EffectOp.SET, EffectOp.INCREMENT, EffectOp.DECREMENT,
                        EffectOp.TOGGLE])

    values = possible_values.get(var, [0, 1, 2, True, False])
    value = random.choice(values)

    return ActionEffect(var, op, value)


def random_action(
    name: str,
    variables: List[str],
    possible_values: Dict[str, List[Any]],
    max_effects: int = 2
) -> Action:
    """Generate random action."""
    n_effects = random.randint(0, max_effects)
    effects = [random_action_effect(variables, possible_values)
               for _ in range(n_effects)]
    return Action(name, effects)


def random_guard_action_pair(
    variables: List[str],
    possible_values: Dict[str, List[Any]],
    n_states: int,
    n_events: int
) -> GuardActionPair:
    """Generate random guard-action pair."""
    guard = random_guard(variables, possible_values)
    action = random_action(f"A{random.randint(0, 99)}", variables, possible_values)

    return GuardActionPair(
        guard=guard,
        action=action,
        source_state=random.randint(0, n_states - 1),
        target_state=random.randint(0, n_states - 1),
        event=random.randint(0, n_events - 1),
        priority=random.randint(0, 3)
    )


def random_joint_genome(
    n_states: int,
    n_events: int,
    n_pairs: int,
    variables: List[str],
    possible_values: Dict[str, List[Any]]
) -> JointGenome:
    """Generate random joint genome."""
    pairs = [
        random_guard_action_pair(variables, possible_values, n_states, n_events)
        for _ in range(n_pairs)
    ]
    return JointGenome(
        pairs=pairs,
        n_states=n_states,
        n_events=n_events,
        variables=set(variables)
    )


# =============================================================================
# Demo
# =============================================================================

def demo():
    """Demonstrate joint genome."""
    print("=" * 60)
    print("JOINT GUARD-ACTION GENOME DEMO")
    print("=" * 60)

    variables = ["turn", "phase", "score", "has_moved"]
    possible_values = {
        "turn": [0, 1, 2],
        "phase": ["init", "play", "end"],
        "score": [0, 1, 2, 3, 4, 5],
        "has_moved": [True, False]
    }

    genome = random_joint_genome(
        n_states=4,
        n_events=3,
        n_pairs=6,
        variables=variables,
        possible_values=possible_values
    )

    print(f"\nGenome: {len(genome.pairs)} guard-action pairs")
    print(f"States: {genome.n_states}, Events: {genome.n_events}")
    print(f"Variables: {genome.variables}")

    print("\n--- Guard-Action Pairs ---")
    for i, pair in enumerate(genome.pairs):
        print(f"\n{i}: s{pair.source_state} --[e{pair.event}]--> s{pair.target_state}")
        print(f"   Guard:  {pair.guard.to_string()}")
        print(f"   Action: {pair.action.to_string()}")
        print(f"   Reads:  {pair.guard_reads()}")
        print(f"   Writes: {pair.action_writes()}")

    print(f"\n--- Variable Coverage ---")
    print(f"Guards read: {genome.all_variables_read()}")
    print(f"Actions write: {genome.all_variables_written()}")
    print(f"Coverage: {genome.variable_coverage():.1%}")

    # Test execution
    print("\n--- Execution Test ---")
    context = {"turn": 0, "phase": "init", "score": 0, "has_moved": False}
    print(f"Initial context: {context}")

    for pair in genome.pairs[:3]:
        enabled = pair.is_enabled(context)
        print(f"\n{pair.guard.to_string()}")
        print(f"  Enabled: {enabled}")
        if enabled:
            new_ctx = pair.execute(context)
            print(f"  Result: {new_ctx}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    demo()
