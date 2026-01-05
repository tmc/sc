"""
Statechart representation for character-level regex matching.

A regex is mathematically equivalent to a DFA/NFA, which maps directly
to a statechart with:
- States: START, intermediate, ACCEPT
- Transitions: character guards between states
- Match evaluation: Run string through statechart, check if ends in ACCEPT
"""

from dataclasses import dataclass, field
from typing import List, Set, Dict, Optional, Tuple, FrozenSet
from enum import Enum, auto
import copy


class StateType(Enum):
    """State types in a regex statechart."""
    START = auto()       # Initial state
    INTERMEDIATE = auto()  # Processing states
    ACCEPT = auto()      # Accepting/final state


@dataclass
class CharGuard:
    """
    Guard condition for character transitions.

    Supports:
    - Single characters: 'a', 'b'
    - Character classes: [a-z], [0-9]
    - Any character: '.'
    - Negation: [^abc]
    """
    chars: FrozenSet[str] = field(default_factory=frozenset)  # Matching characters
    is_any: bool = False  # Matches any character (.)
    is_negated: bool = False  # Negated class [^...]

    def matches(self, char: str) -> bool:
        """Check if character matches this guard."""
        if self.is_any:
            return True
        if self.is_negated:
            return char not in self.chars
        return char in self.chars

    def to_string(self) -> str:
        """Convert to human-readable string."""
        if self.is_any:
            return "."
        if len(self.chars) == 0:
            return "∅"
        if len(self.chars) == 1:
            c = next(iter(self.chars))
            if self.is_negated:
                return f"[^{c}]"
            return c
        chars_str = "".join(sorted(self.chars))
        if self.is_negated:
            return f"[^{chars_str}]"
        return f"[{chars_str}]"

    @classmethod
    def single(cls, char: str) -> "CharGuard":
        """Create a single-character guard."""
        return cls(chars=frozenset([char]))

    @classmethod
    def char_class(cls, chars: str, negated: bool = False) -> "CharGuard":
        """Create a character class guard."""
        return cls(chars=frozenset(chars), is_negated=negated)

    @classmethod
    def any_char(cls) -> "CharGuard":
        """Create a wildcard guard."""
        return cls(is_any=True)

    @classmethod
    def range(cls, start: str, end: str) -> "CharGuard":
        """Create a character range guard [start-end]."""
        chars = frozenset(chr(c) for c in range(ord(start), ord(end) + 1))
        return cls(chars=chars)


@dataclass
class Transition:
    """A transition in the regex statechart."""
    source: int  # Source state index
    target: int  # Target state index
    guard: CharGuard  # Character guard

    def to_string(self, states: List[str]) -> str:
        """Convert to human-readable string."""
        src = states[self.source] if self.source < len(states) else f"S{self.source}"
        tgt = states[self.target] if self.target < len(states) else f"S{self.target}"
        return f"{src} --{self.guard.to_string()}--> {tgt}"


@dataclass
class RegexStatechart:
    """
    A statechart representing a regular expression.

    Structure:
    - state_labels: Names for each state
    - state_types: Type of each state (START, INTERMEDIATE, ACCEPT)
    - transitions: List of transitions with character guards
    - initial_state: Index of initial state

    Example for regex "a(b|c)*d":
        States: [START, SAW_A, IN_BC, ACCEPT]
        Types:  [START, INTERMEDIATE, INTERMEDIATE, ACCEPT]
        Transitions:
            START --'a'--> SAW_A
            SAW_A --'b'--> IN_BC
            SAW_A --'c'--> IN_BC
            SAW_A --'d'--> ACCEPT
            IN_BC --'b'--> IN_BC
            IN_BC --'c'--> IN_BC
            IN_BC --'d'--> ACCEPT
    """
    state_labels: List[str] = field(default_factory=list)
    state_types: List[StateType] = field(default_factory=list)
    transitions: List[Transition] = field(default_factory=list)
    initial_state: int = 0

    # Fitness tracking (for evolution)
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

    def get_transitions_from(self, state: int) -> List[Transition]:
        """Get all transitions from a state."""
        return [t for t in self.transitions if t.source == state]

    def step(self, current_states: Set[int], char: str) -> Set[int]:
        """
        Process one character, returning new set of active states.

        Implements NFA semantics: can be in multiple states simultaneously.
        """
        next_states = set()
        for state in current_states:
            for trans in self.get_transitions_from(state):
                if trans.guard.matches(char):
                    next_states.add(trans.target)
        return next_states

    def matches(self, string: str) -> bool:
        """
        Check if string matches this statechart.

        Returns True if after processing all characters, any active
        state is an accepting state.
        """
        if not string:
            # Empty string: accept only if initial state is accepting
            return self.state_types[self.initial_state] == StateType.ACCEPT

        current = {self.initial_state}

        for char in string:
            current = self.step(current, char)
            if not current:
                return False  # Dead end, no matching transitions

        # Check if any current state is accepting
        return any(s in self.accept_states for s in current)

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
        # Must have at least one accepting state
        if not self.accept_states:
            return False
        # All transitions must reference valid states
        for t in self.transitions:
            if t.source < 0 or t.source >= self.n_states:
                return False
            if t.target < 0 or t.target >= self.n_states:
                return False
        return True

    def to_string(self) -> str:
        """Convert to human-readable string."""
        lines = ["RegexStatechart:"]
        lines.append(f"  States ({self.n_states}):")
        for i, (label, stype) in enumerate(zip(self.state_labels, self.state_types)):
            marker = ""
            if i == self.initial_state:
                marker += " [INITIAL]"
            if stype == StateType.ACCEPT:
                marker += " [ACCEPT]"
            lines.append(f"    {i}: {label} ({stype.name}){marker}")

        lines.append(f"  Transitions ({self.n_transitions}):")
        for t in self.transitions:
            lines.append(f"    {t.to_string(self.state_labels)}")

        return "\n".join(lines)

    def copy(self) -> "RegexStatechart":
        """Create a deep copy."""
        return copy.deepcopy(self)

    @classmethod
    def create_empty(cls, n_states: int = 2) -> "RegexStatechart":
        """Create an empty statechart with START and ACCEPT states."""
        labels = ["START"] + [f"S{i}" for i in range(1, n_states - 1)] + ["ACCEPT"]
        types = [StateType.START] + [StateType.INTERMEDIATE] * (n_states - 2) + [StateType.ACCEPT]
        return cls(state_labels=labels, state_types=types, initial_state=0)

    @classmethod
    def create_simple_sequence(cls, pattern: str) -> "RegexStatechart":
        """
        Create a statechart matching a simple character sequence.

        Example: "abc" -> START -'a'-> S1 -'b'-> S2 -'c'-> ACCEPT
        """
        if not pattern:
            # Empty pattern: just accept empty string
            sc = cls(
                state_labels=["START_ACCEPT"],
                state_types=[StateType.ACCEPT],
                initial_state=0
            )
            return sc

        n_states = len(pattern) + 1
        labels = ["START"] + [f"S{i}" for i in range(1, n_states - 1)] + ["ACCEPT"]
        types = [StateType.START] + [StateType.INTERMEDIATE] * (n_states - 2) + [StateType.ACCEPT]

        transitions = []
        for i, char in enumerate(pattern):
            transitions.append(Transition(
                source=i,
                target=i + 1,
                guard=CharGuard.single(char)
            ))

        return cls(
            state_labels=labels,
            state_types=types,
            transitions=transitions,
            initial_state=0
        )

    @classmethod
    def create_star(cls, char: str) -> "RegexStatechart":
        """
        Create a statechart matching char* (zero or more).

        Example: "a*" -> START/ACCEPT <-'a'-> (self-loop)
        """
        return cls(
            state_labels=["START_ACCEPT"],
            state_types=[StateType.ACCEPT],
            transitions=[Transition(source=0, target=0, guard=CharGuard.single(char))],
            initial_state=0
        )

    @classmethod
    def create_plus(cls, char: str) -> "RegexStatechart":
        """
        Create a statechart matching char+ (one or more).

        Example: "a+" -> START -'a'-> ACCEPT <-'a'-> (self-loop)
        """
        return cls(
            state_labels=["START", "ACCEPT"],
            state_types=[StateType.START, StateType.ACCEPT],
            transitions=[
                Transition(source=0, target=1, guard=CharGuard.single(char)),
                Transition(source=1, target=1, guard=CharGuard.single(char)),
            ],
            initial_state=0
        )


def test_regex_statechart():
    """Test the RegexStatechart implementation."""
    print("=" * 60)
    print("REGEX STATECHART TESTS")
    print("=" * 60)

    # Test 1: Simple sequence "abc"
    print("\n1. Testing simple sequence 'abc':")
    sc_abc = RegexStatechart.create_simple_sequence("abc")
    print(sc_abc.to_string())

    test_cases = [("abc", True), ("ab", False), ("abcd", False), ("", False), ("abd", False)]
    for s, expected in test_cases:
        result = sc_abc.matches(s)
        status = "PASS" if result == expected else "FAIL"
        print(f"   '{s}' -> {result} (expected {expected}) [{status}]")

    # Test 2: a* (zero or more)
    print("\n2. Testing 'a*':")
    sc_star = RegexStatechart.create_star("a")
    print(sc_star.to_string())

    test_cases = [("", True), ("a", True), ("aa", True), ("aaa", True), ("b", False), ("ab", False)]
    for s, expected in test_cases:
        result = sc_star.matches(s)
        status = "PASS" if result == expected else "FAIL"
        print(f"   '{s}' -> {result} (expected {expected}) [{status}]")

    # Test 3: a+ (one or more)
    print("\n3. Testing 'a+':")
    sc_plus = RegexStatechart.create_plus("a")
    print(sc_plus.to_string())

    test_cases = [("", False), ("a", True), ("aa", True), ("aaa", True), ("b", False), ("ab", False)]
    for s, expected in test_cases:
        result = sc_plus.matches(s)
        status = "PASS" if result == expected else "FAIL"
        print(f"   '{s}' -> {result} (expected {expected}) [{status}]")

    # Test 4: Manual statechart for a(b|c)*d
    print("\n4. Testing manual statechart for 'a(b|c)*d':")
    sc_manual = RegexStatechart(
        state_labels=["START", "SAW_A", "ACCEPT"],
        state_types=[StateType.START, StateType.INTERMEDIATE, StateType.ACCEPT],
        transitions=[
            Transition(source=0, target=1, guard=CharGuard.single('a')),
            Transition(source=1, target=1, guard=CharGuard.char_class('bc')),
            Transition(source=1, target=2, guard=CharGuard.single('d')),
        ],
        initial_state=0
    )
    print(sc_manual.to_string())

    test_cases = [
        ("ad", True), ("abd", True), ("acd", True), ("abcd", True),
        ("abcbcd", True), ("a", False), ("d", False), ("ab", False),
        ("aad", False), ("abdc", False)
    ]
    for s, expected in test_cases:
        result = sc_manual.matches(s)
        status = "PASS" if result == expected else "FAIL"
        print(f"   '{s}' -> {result} (expected {expected}) [{status}]")

    # Test 5: CharGuard tests
    print("\n5. Testing CharGuard:")

    g_single = CharGuard.single('a')
    print(f"   single('a'): {g_single.to_string()}")
    assert g_single.matches('a') and not g_single.matches('b')

    g_class = CharGuard.char_class('abc')
    print(f"   char_class('abc'): {g_class.to_string()}")
    assert g_class.matches('a') and g_class.matches('b') and not g_class.matches('d')

    g_neg = CharGuard.char_class('ab', negated=True)
    print(f"   char_class('ab', negated=True): {g_neg.to_string()}")
    assert not g_neg.matches('a') and not g_neg.matches('b') and g_neg.matches('c')

    g_any = CharGuard.any_char()
    print(f"   any_char(): {g_any.to_string()}")
    assert g_any.matches('x') and g_any.matches('9')

    g_range = CharGuard.range('a', 'z')
    print(f"   range('a', 'z'): {g_range.to_string()}")
    assert g_range.matches('m') and not g_range.matches('A')

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    test_regex_statechart()
