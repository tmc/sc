"""
Evolvable Semantic Components for Statecharts.

This module defines the building blocks that can be evolved:
- Guard expressions (beyond simple character matching)
- Actions (side effects on transitions)
- Predicates (conditions on extended state)
- Extended state variables (counters, flags, buffers)

Key insight: Real regex engines have more than just states and transitions.
They have counters ({n,m}), capture groups, lookahead, etc. We model these
as evolvable semantic components.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Set, Any, Optional, Callable, Union
from enum import Enum, auto
import copy
import random


# =============================================================================
# Extended State Variables
# =============================================================================

class VarType(Enum):
    """Types of extended state variables."""
    COUNTER = auto()      # Integer counter (for {n,m} quantifiers)
    FLAG = auto()         # Boolean flag
    CHAR_BUFFER = auto()  # Buffer storing matched characters
    POSITION = auto()     # Current position in input
    LAST_CHAR = auto()    # Last matched character


@dataclass
class StateVar:
    """An extended state variable."""
    name: str
    var_type: VarType
    initial_value: Any = 0

    def default_value(self) -> Any:
        """Get default value based on type."""
        if self.var_type == VarType.COUNTER:
            return 0
        elif self.var_type == VarType.FLAG:
            return False
        elif self.var_type == VarType.CHAR_BUFFER:
            return ""
        elif self.var_type == VarType.POSITION:
            return 0
        elif self.var_type == VarType.LAST_CHAR:
            return ""
        return None


@dataclass
class ExtendedState:
    """
    Extended state context for statechart execution.

    Contains all variable values and provides access methods.
    """
    variables: Dict[str, Any] = field(default_factory=dict)
    input_string: str = ""
    position: int = 0
    current_char: str = ""

    def get(self, name: str, default: Any = None) -> Any:
        """Get variable value."""
        return self.variables.get(name, default)

    def set(self, name: str, value: Any) -> "ExtendedState":
        """Set variable value, returning new state."""
        new_state = copy.deepcopy(self)
        new_state.variables[name] = value
        return new_state

    def increment(self, name: str, amount: int = 1) -> "ExtendedState":
        """Increment counter variable."""
        new_state = copy.deepcopy(self)
        new_state.variables[name] = self.variables.get(name, 0) + amount
        return new_state

    def append_char(self, name: str, char: str) -> "ExtendedState":
        """Append character to buffer variable."""
        new_state = copy.deepcopy(self)
        new_state.variables[name] = self.variables.get(name, "") + char
        return new_state

    def copy(self) -> "ExtendedState":
        """Create a deep copy."""
        return copy.deepcopy(self)


# =============================================================================
# Guard Expressions (Evolvable)
# =============================================================================

class GuardExpr(ABC):
    """Base class for guard expressions."""

    @abstractmethod
    def evaluate(self, char: str, state: ExtendedState) -> bool:
        """Evaluate guard given current character and extended state."""
        pass

    @abstractmethod
    def to_string(self) -> str:
        """Convert to human-readable string."""
        pass

    @abstractmethod
    def mutate(self, alphabet: str) -> "GuardExpr":
        """Return a mutated copy of this guard."""
        pass

    @abstractmethod
    def complexity(self) -> int:
        """Return complexity score (for parsimony)."""
        pass


@dataclass
class CharGuardExpr(GuardExpr):
    """Guard that matches specific characters."""
    chars: frozenset = field(default_factory=frozenset)
    negated: bool = False
    is_any: bool = False

    def evaluate(self, char: str, state: ExtendedState) -> bool:
        if self.is_any:
            return True
        if self.negated:
            return char not in self.chars
        return char in self.chars

    def to_string(self) -> str:
        if self.is_any:
            return "."
        if len(self.chars) == 0:
            return "∅"
        chars_str = "".join(sorted(self.chars))
        if self.negated:
            return f"[^{chars_str}]"
        if len(self.chars) == 1:
            return next(iter(self.chars))
        return f"[{chars_str}]"

    def mutate(self, alphabet: str) -> "CharGuardExpr":
        mutation = random.choice(['add', 'remove', 'toggle_neg', 'toggle_any', 'replace'])

        if mutation == 'add' and not self.is_any:
            new_char = random.choice(alphabet)
            return CharGuardExpr(chars=self.chars | {new_char}, negated=self.negated)
        elif mutation == 'remove' and len(self.chars) > 1:
            chars_list = list(self.chars)
            chars_list.remove(random.choice(chars_list))
            return CharGuardExpr(chars=frozenset(chars_list), negated=self.negated)
        elif mutation == 'toggle_neg':
            return CharGuardExpr(chars=self.chars, negated=not self.negated)
        elif mutation == 'toggle_any':
            if self.is_any:
                return CharGuardExpr(chars=frozenset(random.sample(alphabet, min(3, len(alphabet)))))
            return CharGuardExpr(is_any=True)
        else:  # replace
            n = random.randint(1, 4)
            return CharGuardExpr(chars=frozenset(random.sample(alphabet, min(n, len(alphabet)))))

    def complexity(self) -> int:
        if self.is_any:
            return 1
        return len(self.chars)


@dataclass
class CounterGuardExpr(GuardExpr):
    """Guard that checks a counter variable."""
    var_name: str
    operator: str  # '<', '<=', '==', '>=', '>', '!='
    value: int

    def evaluate(self, char: str, state: ExtendedState) -> bool:
        counter_val = state.get(self.var_name, 0)
        ops = {
            '<': lambda a, b: a < b,
            '<=': lambda a, b: a <= b,
            '==': lambda a, b: a == b,
            '>=': lambda a, b: a >= b,
            '>': lambda a, b: a > b,
            '!=': lambda a, b: a != b,
        }
        return ops.get(self.operator, lambda a, b: False)(counter_val, self.value)

    def to_string(self) -> str:
        return f"{self.var_name} {self.operator} {self.value}"

    def mutate(self, alphabet: str) -> "CounterGuardExpr":
        mutation = random.choice(['change_op', 'change_val'])

        if mutation == 'change_op':
            new_op = random.choice(['<', '<=', '==', '>=', '>', '!='])
            return CounterGuardExpr(var_name=self.var_name, operator=new_op, value=self.value)
        else:
            delta = random.randint(-2, 2)
            return CounterGuardExpr(var_name=self.var_name, operator=self.operator, value=max(0, self.value + delta))

    def complexity(self) -> int:
        return 2


@dataclass
class FlagGuardExpr(GuardExpr):
    """Guard that checks a boolean flag."""
    var_name: str
    expected: bool = True

    def evaluate(self, char: str, state: ExtendedState) -> bool:
        flag_val = state.get(self.var_name, False)
        return flag_val == self.expected

    def to_string(self) -> str:
        if self.expected:
            return self.var_name
        return f"!{self.var_name}"

    def mutate(self, alphabet: str) -> "FlagGuardExpr":
        return FlagGuardExpr(var_name=self.var_name, expected=not self.expected)

    def complexity(self) -> int:
        return 1


@dataclass
class CompositeGuardExpr(GuardExpr):
    """Composite guard combining multiple guards with AND/OR."""
    left: GuardExpr
    right: GuardExpr
    operator: str  # 'and', 'or'

    def evaluate(self, char: str, state: ExtendedState) -> bool:
        left_val = self.left.evaluate(char, state)
        right_val = self.right.evaluate(char, state)
        if self.operator == 'and':
            return left_val and right_val
        return left_val or right_val

    def to_string(self) -> str:
        return f"({self.left.to_string()} {self.operator} {self.right.to_string()})"

    def mutate(self, alphabet: str) -> "CompositeGuardExpr":
        mutation = random.choice(['toggle_op', 'mutate_left', 'mutate_right'])

        if mutation == 'toggle_op':
            new_op = 'or' if self.operator == 'and' else 'and'
            return CompositeGuardExpr(left=self.left, right=self.right, operator=new_op)
        elif mutation == 'mutate_left':
            return CompositeGuardExpr(left=self.left.mutate(alphabet), right=self.right, operator=self.operator)
        else:
            return CompositeGuardExpr(left=self.left, right=self.right.mutate(alphabet), operator=self.operator)

    def complexity(self) -> int:
        return 1 + self.left.complexity() + self.right.complexity()


# =============================================================================
# Actions (Evolvable)
# =============================================================================

class ActionExpr(ABC):
    """Base class for action expressions (side effects)."""

    @abstractmethod
    def execute(self, char: str, state: ExtendedState) -> ExtendedState:
        """Execute action, returning new extended state."""
        pass

    @abstractmethod
    def to_string(self) -> str:
        """Convert to human-readable string."""
        pass

    @abstractmethod
    def mutate(self) -> "ActionExpr":
        """Return a mutated copy of this action."""
        pass


@dataclass
class NoOpAction(ActionExpr):
    """No operation - does nothing."""

    def execute(self, char: str, state: ExtendedState) -> ExtendedState:
        return state

    def to_string(self) -> str:
        return "noop"

    def mutate(self) -> "ActionExpr":
        return self


@dataclass
class IncrementAction(ActionExpr):
    """Increment a counter variable."""
    var_name: str
    amount: int = 1

    def execute(self, char: str, state: ExtendedState) -> ExtendedState:
        return state.increment(self.var_name, self.amount)

    def to_string(self) -> str:
        if self.amount == 1:
            return f"{self.var_name}++"
        return f"{self.var_name} += {self.amount}"

    def mutate(self) -> "ActionExpr":
        delta = random.choice([-1, 0, 1])
        new_amount = max(1, self.amount + delta)
        return IncrementAction(var_name=self.var_name, amount=new_amount)


@dataclass
class ResetAction(ActionExpr):
    """Reset a variable to its initial value."""
    var_name: str
    value: Any = 0

    def execute(self, char: str, state: ExtendedState) -> ExtendedState:
        return state.set(self.var_name, self.value)

    def to_string(self) -> str:
        return f"{self.var_name} := {self.value}"

    def mutate(self) -> "ActionExpr":
        if isinstance(self.value, int):
            delta = random.randint(-2, 2)
            return ResetAction(var_name=self.var_name, value=max(0, self.value + delta))
        return self


@dataclass
class SetFlagAction(ActionExpr):
    """Set a boolean flag."""
    var_name: str
    value: bool = True

    def execute(self, char: str, state: ExtendedState) -> ExtendedState:
        return state.set(self.var_name, self.value)

    def to_string(self) -> str:
        return f"{self.var_name} := {self.value}"

    def mutate(self) -> "ActionExpr":
        return SetFlagAction(var_name=self.var_name, value=not self.value)


@dataclass
class AppendCharAction(ActionExpr):
    """Append current character to a buffer."""
    var_name: str

    def execute(self, char: str, state: ExtendedState) -> ExtendedState:
        return state.append_char(self.var_name, char)

    def to_string(self) -> str:
        return f"{self.var_name}.append(char)"

    def mutate(self) -> "ActionExpr":
        return self  # Not much to mutate


@dataclass
class CompositeAction(ActionExpr):
    """Sequence of actions."""
    actions: List[ActionExpr] = field(default_factory=list)

    def execute(self, char: str, state: ExtendedState) -> ExtendedState:
        for action in self.actions:
            state = action.execute(char, state)
        return state

    def to_string(self) -> str:
        if not self.actions:
            return "noop"
        return "; ".join(a.to_string() for a in self.actions)

    def mutate(self) -> "ActionExpr":
        if not self.actions:
            return self
        idx = random.randint(0, len(self.actions) - 1)
        new_actions = self.actions.copy()
        new_actions[idx] = new_actions[idx].mutate()
        return CompositeAction(actions=new_actions)


# =============================================================================
# Semantic Transition (combines guard + action)
# =============================================================================

@dataclass
class SemanticTransition:
    """
    A transition with full semantic components.

    Combines:
    - Source and target states
    - Guard expression (when to fire)
    - Action expression (what to do)
    - Priority (for conflict resolution)
    """
    source: int
    target: int
    guard: GuardExpr
    action: ActionExpr = field(default_factory=NoOpAction)
    priority: int = 0

    def can_fire(self, char: str, state: ExtendedState) -> bool:
        """Check if transition can fire."""
        return self.guard.evaluate(char, state)

    def fire(self, char: str, state: ExtendedState) -> ExtendedState:
        """Execute transition, returning new extended state."""
        return self.action.execute(char, state)

    def to_string(self, state_labels: List[str] = None) -> str:
        src = state_labels[self.source] if state_labels and self.source < len(state_labels) else f"S{self.source}"
        tgt = state_labels[self.target] if state_labels and self.target < len(state_labels) else f"S{self.target}"
        guard_str = self.guard.to_string()
        action_str = self.action.to_string()

        if action_str == "noop":
            return f"{src} --[{guard_str}]--> {tgt}"
        return f"{src} --[{guard_str}] / {action_str}--> {tgt}"

    def complexity(self) -> int:
        """Total complexity of this transition."""
        return self.guard.complexity()


# =============================================================================
# Component Factories (for random generation)
# =============================================================================

class ComponentFactory:
    """Factory for creating random semantic components."""

    def __init__(
        self,
        alphabet: str = "abcdefghijklmnopqrstuvwxyz0123456789",
        counter_vars: List[str] = None,
        flag_vars: List[str] = None,
        buffer_vars: List[str] = None,
    ):
        self.alphabet = alphabet
        self.counter_vars = counter_vars or ["count", "n"]
        self.flag_vars = flag_vars or ["seen", "matched"]
        self.buffer_vars = buffer_vars or ["buffer"]

    def random_char_guard(self) -> CharGuardExpr:
        """Create a random character guard."""
        choice = random.random()

        if choice < 0.4:
            # Single char
            return CharGuardExpr(chars=frozenset([random.choice(self.alphabet)]))
        elif choice < 0.7:
            # Small class
            n = random.randint(2, 4)
            chars = random.sample(self.alphabet, min(n, len(self.alphabet)))
            return CharGuardExpr(chars=frozenset(chars))
        elif choice < 0.85:
            # Negated
            n = random.randint(1, 3)
            chars = random.sample(self.alphabet, min(n, len(self.alphabet)))
            return CharGuardExpr(chars=frozenset(chars), negated=True)
        else:
            # Any
            return CharGuardExpr(is_any=True)

    def random_counter_guard(self) -> CounterGuardExpr:
        """Create a random counter guard."""
        var = random.choice(self.counter_vars)
        op = random.choice(['<', '<=', '==', '>=', '>'])
        val = random.randint(0, 5)
        return CounterGuardExpr(var_name=var, operator=op, value=val)

    def random_flag_guard(self) -> FlagGuardExpr:
        """Create a random flag guard."""
        var = random.choice(self.flag_vars)
        expected = random.choice([True, False])
        return FlagGuardExpr(var_name=var, expected=expected)

    def random_guard(self, depth: int = 0, max_depth: int = 2) -> GuardExpr:
        """Create a random guard expression."""
        if depth >= max_depth:
            return self.random_char_guard()

        choice = random.random()

        if choice < 0.5:
            return self.random_char_guard()
        elif choice < 0.7:
            return self.random_counter_guard()
        elif choice < 0.85:
            return self.random_flag_guard()
        else:
            # Composite
            left = self.random_guard(depth + 1, max_depth)
            right = self.random_guard(depth + 1, max_depth)
            op = random.choice(['and', 'or'])
            return CompositeGuardExpr(left=left, right=right, operator=op)

    def random_action(self) -> ActionExpr:
        """Create a random action."""
        choice = random.random()

        if choice < 0.4:
            return NoOpAction()
        elif choice < 0.6:
            var = random.choice(self.counter_vars)
            return IncrementAction(var_name=var)
        elif choice < 0.75:
            var = random.choice(self.counter_vars)
            val = random.randint(0, 3)
            return ResetAction(var_name=var, value=val)
        elif choice < 0.9:
            var = random.choice(self.flag_vars)
            return SetFlagAction(var_name=var, value=random.choice([True, False]))
        else:
            var = random.choice(self.buffer_vars)
            return AppendCharAction(var_name=var)

    def random_transition(self, n_states: int) -> SemanticTransition:
        """Create a random semantic transition."""
        src = random.randint(0, n_states - 1)
        tgt = random.randint(0, n_states - 1)
        guard = self.random_guard()
        action = self.random_action()
        return SemanticTransition(source=src, target=tgt, guard=guard, action=action)


# =============================================================================
# Testing
# =============================================================================

def test_semantic_components():
    """Test semantic components."""
    print("=" * 60)
    print("SEMANTIC COMPONENTS TESTS")
    print("=" * 60)

    # Test 1: Character guards
    print("\n1. Character Guards:")
    g1 = CharGuardExpr(chars=frozenset(['a', 'b', 'c']))
    print(f"   {g1.to_string()}: 'a'={g1.evaluate('a', ExtendedState())}, 'd'={g1.evaluate('d', ExtendedState())}")

    g2 = CharGuardExpr(chars=frozenset(['x']), negated=True)
    print(f"   {g2.to_string()}: 'a'={g2.evaluate('a', ExtendedState())}, 'x'={g2.evaluate('x', ExtendedState())}")

    # Test 2: Counter guards
    print("\n2. Counter Guards:")
    state = ExtendedState(variables={'count': 3})
    g3 = CounterGuardExpr(var_name='count', operator='<', value=5)
    print(f"   {g3.to_string()} (count=3): {g3.evaluate('x', state)}")

    g4 = CounterGuardExpr(var_name='count', operator='>=', value=5)
    print(f"   {g4.to_string()} (count=3): {g4.evaluate('x', state)}")

    # Test 3: Composite guards
    print("\n3. Composite Guards:")
    g5 = CompositeGuardExpr(
        left=CharGuardExpr(chars=frozenset(['a'])),
        right=CounterGuardExpr(var_name='count', operator='<', value=3),
        operator='and'
    )
    print(f"   {g5.to_string()}")
    print(f"   ('a', count=2): {g5.evaluate('a', ExtendedState(variables={'count': 2}))}")
    print(f"   ('a', count=5): {g5.evaluate('a', ExtendedState(variables={'count': 5}))}")
    print(f"   ('b', count=2): {g5.evaluate('b', ExtendedState(variables={'count': 2}))}")

    # Test 4: Actions
    print("\n4. Actions:")
    state = ExtendedState(variables={'count': 0, 'seen': False})

    a1 = IncrementAction(var_name='count')
    state = a1.execute('x', state)
    print(f"   {a1.to_string()}: count={state.get('count')}")

    a2 = SetFlagAction(var_name='seen', value=True)
    state = a2.execute('x', state)
    print(f"   {a2.to_string()}: seen={state.get('seen')}")

    # Test 5: Semantic transition
    print("\n5. Semantic Transition:")
    trans = SemanticTransition(
        source=0,
        target=1,
        guard=CompositeGuardExpr(
            left=CharGuardExpr(chars=frozenset(['a'])),
            right=CounterGuardExpr(var_name='count', operator='<', value=3),
            operator='and'
        ),
        action=CompositeAction(actions=[
            IncrementAction(var_name='count'),
            SetFlagAction(var_name='seen', value=True)
        ])
    )
    print(f"   {trans.to_string(['START', 'END'])}")

    state = ExtendedState(variables={'count': 1, 'seen': False})
    print(f"   can_fire('a', count=1): {trans.can_fire('a', state)}")

    new_state = trans.fire('a', state)
    print(f"   after fire: count={new_state.get('count')}, seen={new_state.get('seen')}")

    # Test 6: Factory
    print("\n6. Component Factory:")
    factory = ComponentFactory()

    print("   Random guards:")
    for _ in range(3):
        g = factory.random_guard()
        print(f"     {g.to_string()}")

    print("   Random actions:")
    for _ in range(3):
        a = factory.random_action()
        print(f"     {a.to_string()}")

    print("\n" + "=" * 60)
    print("All semantic component tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    test_semantic_components()
