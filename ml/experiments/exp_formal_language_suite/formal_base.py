"""
Formal Base: Abstract Classes for Formal Language Automata

Provides unified interface for:
- DFA (Deterministic Finite Automaton)
- NFA (Non-deterministic Finite Automaton)
- PDA (Pushdown Automaton)
- Statecharts (hierarchical state machines)

All share common concepts:
- States (configurations)
- Transitions (state changes)
- Alphabet (input symbols)
- Acceptance (language membership)
"""

from dataclasses import dataclass, field
from typing import (
    List, Dict, Set, Tuple, Optional, FrozenSet,
    Iterator, Generic, TypeVar, Any, Callable,
)
from abc import ABC, abstractmethod
from enum import Enum, auto
import re


# Type variables for generic automata
S = TypeVar('S')  # State type
I = TypeVar('I')  # Input symbol type


class AcceptResult(Enum):
    """Result of acceptance check."""
    ACCEPT = auto()
    REJECT = auto()
    UNDECIDED = auto()  # For incomplete automata


@dataclass(frozen=True)
class Symbol:
    """
    Input symbol for automata.

    Can be:
    - Character (single char)
    - Token (string)
    - Epsilon (empty transition)
    - Wildcard (any symbol)
    """
    value: str
    is_epsilon: bool = False
    is_wildcard: bool = False

    @staticmethod
    def epsilon() -> 'Symbol':
        return Symbol(value='', is_epsilon=True)

    @staticmethod
    def wildcard() -> 'Symbol':
        return Symbol(value='.', is_wildcard=True)

    @staticmethod
    def char(c: str) -> 'Symbol':
        return Symbol(value=c)

    def matches(self, other: str) -> bool:
        """Check if this symbol matches input."""
        if self.is_epsilon:
            return False  # Epsilon doesn't consume input
        if self.is_wildcard:
            return len(other) == 1
        return self.value == other

    def __str__(self) -> str:
        if self.is_epsilon:
            return 'ε'
        if self.is_wildcard:
            return '.'
        return self.value


@dataclass
class Alphabet:
    """
    Alphabet for an automaton.

    Defines the set of valid input symbols.
    """
    symbols: Set[Symbol] = field(default_factory=set)
    allow_wildcard: bool = False

    def add(self, symbol: Symbol):
        """Add a symbol to the alphabet."""
        self.symbols.add(symbol)

    def contains(self, symbol: Symbol) -> bool:
        """Check if symbol is in alphabet."""
        if self.allow_wildcard and symbol.is_wildcard:
            return True
        return symbol in self.symbols

    @staticmethod
    def ascii_lowercase() -> 'Alphabet':
        """Create alphabet of lowercase ASCII letters."""
        return Alphabet(
            symbols={Symbol.char(chr(i)) for i in range(ord('a'), ord('z') + 1)}
        )

    @staticmethod
    def ascii_digits() -> 'Alphabet':
        """Create alphabet of ASCII digits."""
        return Alphabet(
            symbols={Symbol.char(chr(i)) for i in range(ord('0'), ord('9') + 1)}
        )

    @staticmethod
    def from_string(s: str) -> 'Alphabet':
        """Create alphabet from characters in string."""
        return Alphabet(symbols={Symbol.char(c) for c in set(s)})


@dataclass
class State:
    """
    State in an automaton.

    Can be:
    - Basic (leaf state)
    - Composite (has substates - for hierarchical machines)
    - Initial (starting state)
    - Final/Accepting (accepting state)
    """
    id: int
    name: str
    is_initial: bool = False
    is_final: bool = False
    is_composite: bool = False
    parent_id: Optional[int] = None
    children: List[int] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        if not isinstance(other, State):
            return False
        return self.id == other.id


@dataclass
class Transition:
    """
    Transition between states.

    Components:
    - Source state(s)
    - Target state(s)
    - Input symbol (trigger)
    - Guard (optional condition)
    - Action (optional side effect)
    """
    id: int
    source_id: int
    target_id: int
    symbol: Symbol
    guard: str = ""  # Boolean expression
    action: str = ""  # Action to execute
    priority: int = 0

    def __hash__(self):
        return hash(self.id)

    def is_epsilon(self) -> bool:
        """Check if this is an epsilon transition."""
        return self.symbol.is_epsilon


@dataclass
class Configuration:
    """
    Current configuration of an automaton.

    For DFA: single state
    For NFA: set of states
    For PDA: state + stack
    For Statechart: set of active states (configuration)
    """
    active_states: FrozenSet[int]
    stack: Tuple[str, ...] = ()  # For PDA
    context: Dict[str, Any] = field(default_factory=dict)

    def is_accepting(self, automaton: 'Automaton') -> bool:
        """Check if any active state is accepting."""
        for state_id in self.active_states:
            state = automaton.get_state(state_id)
            if state and state.is_final:
                return True
        return False


class Automaton(ABC):
    """
    Abstract base class for all automata.

    Provides common interface for:
    - State management
    - Transition handling
    - Acceptance checking
    - Language operations
    """

    def __init__(self, name: str = ""):
        self.name = name
        self.states: Dict[int, State] = {}
        self.transitions: List[Transition] = []
        self.alphabet: Alphabet = Alphabet()
        self._state_counter = 0
        self._transition_counter = 0
        self._initial_state_id: Optional[int] = None

    def add_state(
        self,
        name: str = "",
        is_initial: bool = False,
        is_final: bool = False,
        **kwargs,
    ) -> State:
        """Add a state to the automaton."""
        state = State(
            id=self._state_counter,
            name=name or f"q{self._state_counter}",
            is_initial=is_initial,
            is_final=is_final,
            **kwargs,
        )
        self.states[state.id] = state
        self._state_counter += 1

        if is_initial:
            self._initial_state_id = state.id

        return state

    def add_transition(
        self,
        source_id: int,
        target_id: int,
        symbol: Symbol,
        **kwargs,
    ) -> Transition:
        """Add a transition to the automaton."""
        trans = Transition(
            id=self._transition_counter,
            source_id=source_id,
            target_id=target_id,
            symbol=symbol,
            **kwargs,
        )
        self.transitions.append(trans)
        self._transition_counter += 1

        # Add symbol to alphabet
        if not symbol.is_epsilon:
            self.alphabet.add(symbol)

        return trans

    def get_state(self, state_id: int) -> Optional[State]:
        """Get state by ID."""
        return self.states.get(state_id)

    def get_initial_state(self) -> Optional[State]:
        """Get initial state."""
        if self._initial_state_id is not None:
            return self.states.get(self._initial_state_id)
        # Find first initial state
        for state in self.states.values():
            if state.is_initial:
                return state
        return None

    def get_final_states(self) -> List[State]:
        """Get all final/accepting states."""
        return [s for s in self.states.values() if s.is_final]

    def get_transitions_from(self, state_id: int) -> List[Transition]:
        """Get all transitions from a state."""
        return [t for t in self.transitions if t.source_id == state_id]

    def get_transitions_on(self, state_id: int, symbol: Symbol) -> List[Transition]:
        """Get transitions from state on specific symbol."""
        return [
            t for t in self.transitions
            if t.source_id == state_id and (t.symbol == symbol or t.symbol.is_wildcard)
        ]

    @abstractmethod
    def accepts(self, input_string: str) -> AcceptResult:
        """Check if automaton accepts input string."""
        pass

    @abstractmethod
    def initial_configuration(self) -> Configuration:
        """Get initial configuration."""
        pass

    @abstractmethod
    def step(self, config: Configuration, symbol: Symbol) -> Set[Configuration]:
        """Compute next configurations on input symbol."""
        pass

    def run(self, input_string: str) -> Iterator[Configuration]:
        """Run automaton on input, yielding configurations."""
        config = self.initial_configuration()
        yield config

        for char in input_string:
            symbol = Symbol.char(char)
            next_configs = self.step(config, symbol)

            if not next_configs:
                return  # Stuck

            # For deterministic: single config. For non-deterministic: pick one
            config = next(iter(next_configs))
            yield config

    def to_statechart_dict(self) -> Dict:
        """Convert to statechart format."""
        return {
            'name': self.name,
            'states': [
                {
                    'id': s.id,
                    'name': s.name,
                    'is_initial': s.is_initial,
                    'is_final': s.is_final,
                }
                for s in self.states.values()
            ],
            'transitions': [
                {
                    'id': t.id,
                    'source': t.source_id,
                    'target': t.target_id,
                    'symbol': str(t.symbol),
                    'guard': t.guard,
                }
                for t in self.transitions
            ],
        }


class FormalLanguage(ABC):
    """
    Abstract base class for formal languages.

    A language is a set of strings accepted by some automaton.
    """

    @abstractmethod
    def contains(self, string: str) -> bool:
        """Check if string is in the language."""
        pass

    @abstractmethod
    def generate(self, max_length: int = 10) -> Iterator[str]:
        """Generate strings in the language."""
        pass

    @abstractmethod
    def to_automaton(self) -> Automaton:
        """Convert to automaton representation."""
        pass

    @abstractmethod
    def to_regex(self) -> str:
        """Convert to regex string (if possible)."""
        pass

    def intersection(self, other: 'FormalLanguage') -> 'FormalLanguage':
        """Compute intersection of two languages."""
        raise NotImplementedError("Intersection not implemented")

    def union(self, other: 'FormalLanguage') -> 'FormalLanguage':
        """Compute union of two languages."""
        raise NotImplementedError("Union not implemented")

    def complement(self) -> 'FormalLanguage':
        """Compute complement of language."""
        raise NotImplementedError("Complement not implemented")


class DFA(Automaton):
    """
    Deterministic Finite Automaton.

    Properties:
    - Exactly one transition per (state, symbol) pair
    - No epsilon transitions
    - Single initial state
    """

    def __init__(self, name: str = "DFA"):
        super().__init__(name)

    def accepts(self, input_string: str) -> AcceptResult:
        """Check if DFA accepts input string."""
        state = self.get_initial_state()
        if not state:
            return AcceptResult.REJECT

        for char in input_string:
            symbol = Symbol.char(char)
            transitions = self.get_transitions_on(state.id, symbol)

            if not transitions:
                return AcceptResult.REJECT

            # DFA: exactly one transition
            state = self.get_state(transitions[0].target_id)
            if not state:
                return AcceptResult.REJECT

        return AcceptResult.ACCEPT if state.is_final else AcceptResult.REJECT

    def initial_configuration(self) -> Configuration:
        """Get initial configuration (single state)."""
        initial = self.get_initial_state()
        if initial:
            return Configuration(active_states=frozenset([initial.id]))
        return Configuration(active_states=frozenset())

    def step(self, config: Configuration, symbol: Symbol) -> Set[Configuration]:
        """Step DFA (deterministic: single next config)."""
        if not config.active_states:
            return set()

        state_id = next(iter(config.active_states))
        transitions = self.get_transitions_on(state_id, symbol)

        if not transitions:
            return set()

        next_state_id = transitions[0].target_id
        return {Configuration(active_states=frozenset([next_state_id]))}

    def is_deterministic(self) -> bool:
        """Verify DFA is deterministic."""
        for state in self.states.values():
            seen_symbols = set()
            for trans in self.get_transitions_from(state.id):
                if trans.symbol.is_epsilon:
                    return False  # DFA can't have epsilon
                if trans.symbol in seen_symbols:
                    return False  # Duplicate transition
                seen_symbols.add(trans.symbol)
        return True


class NFA(Automaton):
    """
    Non-deterministic Finite Automaton.

    Properties:
    - Multiple transitions per (state, symbol) allowed
    - Epsilon transitions allowed
    - Multiple initial states possible
    """

    def __init__(self, name: str = "NFA"):
        super().__init__(name)

    def epsilon_closure(self, state_ids: Set[int]) -> Set[int]:
        """Compute epsilon closure of state set."""
        closure = set(state_ids)
        queue = list(state_ids)

        while queue:
            state_id = queue.pop(0)
            for trans in self.get_transitions_from(state_id):
                if trans.is_epsilon() and trans.target_id not in closure:
                    closure.add(trans.target_id)
                    queue.append(trans.target_id)

        return closure

    def accepts(self, input_string: str) -> AcceptResult:
        """Check if NFA accepts input string."""
        initial = self.get_initial_state()
        if not initial:
            return AcceptResult.REJECT

        # Start with epsilon closure of initial state
        current_states = self.epsilon_closure({initial.id})

        for char in input_string:
            symbol = Symbol.char(char)
            next_states = set()

            for state_id in current_states:
                for trans in self.get_transitions_on(state_id, symbol):
                    next_states.add(trans.target_id)

            # Apply epsilon closure to next states
            current_states = self.epsilon_closure(next_states)

            if not current_states:
                return AcceptResult.REJECT

        # Accept if any current state is final
        for state_id in current_states:
            state = self.get_state(state_id)
            if state and state.is_final:
                return AcceptResult.ACCEPT

        return AcceptResult.REJECT

    def initial_configuration(self) -> Configuration:
        """Get initial configuration (epsilon closure of initial)."""
        initial = self.get_initial_state()
        if initial:
            closure = self.epsilon_closure({initial.id})
            return Configuration(active_states=frozenset(closure))
        return Configuration(active_states=frozenset())

    def step(self, config: Configuration, symbol: Symbol) -> Set[Configuration]:
        """Step NFA (non-deterministic: multiple next configs)."""
        next_states = set()

        for state_id in config.active_states:
            for trans in self.get_transitions_on(state_id, symbol):
                next_states.add(trans.target_id)

        if not next_states:
            return set()

        # Apply epsilon closure
        closure = self.epsilon_closure(next_states)
        return {Configuration(active_states=frozenset(closure))}

    def to_dfa(self) -> DFA:
        """Convert NFA to equivalent DFA (subset construction)."""
        dfa = DFA(name=f"{self.name}_DFA")

        initial = self.get_initial_state()
        if not initial:
            return dfa

        # Initial DFA state is epsilon closure of NFA initial
        initial_closure = frozenset(self.epsilon_closure({initial.id}))

        # Map NFA state sets to DFA state IDs
        state_map: Dict[FrozenSet[int], int] = {}

        # Create initial DFA state
        is_final = any(self.get_state(s).is_final for s in initial_closure if self.get_state(s))
        dfa_initial = dfa.add_state(is_initial=True, is_final=is_final)
        state_map[initial_closure] = dfa_initial.id

        # BFS to construct DFA
        queue = [initial_closure]
        visited = {initial_closure}

        while queue:
            current_set = queue.pop(0)
            current_dfa_id = state_map[current_set]

            # For each symbol in alphabet
            for symbol in self.alphabet.symbols:
                next_nfa_states = set()

                for nfa_state_id in current_set:
                    for trans in self.get_transitions_on(nfa_state_id, symbol):
                        next_nfa_states.add(trans.target_id)

                if not next_nfa_states:
                    continue

                # Apply epsilon closure
                next_set = frozenset(self.epsilon_closure(next_nfa_states))

                # Create new DFA state if needed
                if next_set not in state_map:
                    is_final = any(self.get_state(s).is_final for s in next_set if self.get_state(s))
                    new_dfa_state = dfa.add_state(is_final=is_final)
                    state_map[next_set] = new_dfa_state.id

                    if next_set not in visited:
                        visited.add(next_set)
                        queue.append(next_set)

                # Add transition
                dfa.add_transition(current_dfa_id, state_map[next_set], symbol)

        return dfa


def demo():
    """Demonstrate formal automata base classes."""
    print("=" * 60)
    print("FORMAL BASE: Abstract Automaton Classes")
    print("=" * 60)

    # Create a simple DFA for strings ending in 'ab'
    print("\n--- DFA for strings ending in 'ab' ---")
    dfa = DFA(name="ends_with_ab")

    q0 = dfa.add_state("q0", is_initial=True)
    q1 = dfa.add_state("q1")
    q2 = dfa.add_state("q2", is_final=True)

    dfa.add_transition(q0.id, q0.id, Symbol.char('b'))
    dfa.add_transition(q0.id, q1.id, Symbol.char('a'))
    dfa.add_transition(q1.id, q1.id, Symbol.char('a'))
    dfa.add_transition(q1.id, q2.id, Symbol.char('b'))
    dfa.add_transition(q2.id, q0.id, Symbol.char('b'))
    dfa.add_transition(q2.id, q1.id, Symbol.char('a'))

    test_strings = ["ab", "aab", "bab", "abb", "aa", "bb", ""]
    print("Test results:")
    for s in test_strings:
        result = dfa.accepts(s)
        print(f"  '{s}': {result.name}")

    # Create NFA with epsilon transitions
    print("\n--- NFA with epsilon transitions ---")
    nfa = NFA(name="a*b*")

    n0 = nfa.add_state("n0", is_initial=True)
    n1 = nfa.add_state("n1")
    n2 = nfa.add_state("n2", is_final=True)

    nfa.add_transition(n0.id, n0.id, Symbol.char('a'))
    nfa.add_transition(n0.id, n1.id, Symbol.epsilon())
    nfa.add_transition(n1.id, n1.id, Symbol.char('b'))
    nfa.add_transition(n1.id, n2.id, Symbol.epsilon())

    test_strings = ["", "a", "b", "aa", "bb", "ab", "aab", "abb", "ba"]
    print("NFA test results:")
    for s in test_strings:
        result = nfa.accepts(s)
        print(f"  '{s}': {result.name}")

    # Convert NFA to DFA
    print("\n--- NFA to DFA conversion ---")
    converted_dfa = nfa.to_dfa()
    print(f"NFA states: {len(nfa.states)}")
    print(f"DFA states: {len(converted_dfa.states)}")
    print(f"DFA is deterministic: {converted_dfa.is_deterministic()}")

    return dfa, nfa


if __name__ == "__main__":
    demo()
