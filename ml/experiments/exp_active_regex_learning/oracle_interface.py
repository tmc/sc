"""
Oracle Interface for Active Regex Learning

Provides ground truth labels for strings against target patterns.
Simulates a human oracle that can label strings as matching or not.

The oracle knows the true regex/statechart and answers queries.
In practice, this would be a human labeler or a test suite.
"""

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Set, Tuple, Optional, Callable
import random
import string


# =============================================================================
# Oracle Interface
# =============================================================================

class Oracle(ABC):
    """Abstract base class for oracles that label strings."""

    @abstractmethod
    def query(self, s: str) -> bool:
        """Query whether string s matches the target pattern."""
        pass

    @abstractmethod
    def get_alphabet(self) -> Set[str]:
        """Get the alphabet of valid characters."""
        pass

    def batch_query(self, strings: List[str]) -> List[bool]:
        """Query multiple strings at once."""
        return [self.query(s) for s in strings]

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name of the pattern."""
        pass


# =============================================================================
# Regex Oracle
# =============================================================================

class RegexOracle(Oracle):
    """Oracle backed by a compiled regex pattern."""

    def __init__(self, pattern: str, alphabet: Optional[Set[str]] = None):
        """
        Args:
            pattern: Regex pattern string
            alphabet: Set of valid characters (default: lowercase letters)
        """
        self._pattern = pattern
        self._compiled = re.compile(f"^{pattern}$")
        self._alphabet = alphabet or set(string.ascii_lowercase)
        self._query_count = 0

    def query(self, s: str) -> bool:
        """Check if string fully matches the pattern."""
        self._query_count += 1
        return bool(self._compiled.match(s))

    def get_alphabet(self) -> Set[str]:
        return self._alphabet

    @property
    def name(self) -> str:
        return f"Regex({self._pattern})"

    @property
    def query_count(self) -> int:
        """Number of queries made to this oracle."""
        return self._query_count

    def reset_count(self):
        """Reset query counter."""
        self._query_count = 0


# =============================================================================
# Statechart Oracle
# =============================================================================

@dataclass
class SimpleState:
    """Simple state for DFA-based oracle."""
    name: str
    is_accepting: bool = False
    transitions: dict = field(default_factory=dict)  # char -> state_name


class StatechartOracle(Oracle):
    """Oracle backed by a DFA statechart."""

    def __init__(
        self,
        states: List[SimpleState],
        initial_state: str,
        alphabet: Set[str],
        name: str = "Statechart"
    ):
        self._states = {s.name: s for s in states}
        self._initial = initial_state
        self._alphabet = alphabet
        self._name = name
        self._query_count = 0

    def query(self, s: str) -> bool:
        """Simulate DFA execution on string."""
        self._query_count += 1
        current = self._initial

        for char in s:
            if char not in self._alphabet:
                return False
            state = self._states.get(current)
            if state is None:
                return False
            next_state = state.transitions.get(char)
            if next_state is None:
                return False
            current = next_state

        final_state = self._states.get(current)
        return final_state is not None and final_state.is_accepting

    def get_alphabet(self) -> Set[str]:
        return self._alphabet

    @property
    def name(self) -> str:
        return self._name

    @property
    def query_count(self) -> int:
        return self._query_count

    def reset_count(self):
        self._query_count = 0


# =============================================================================
# Predefined Oracles for Benchmarks
# =============================================================================

def create_email_prefix_oracle() -> RegexOracle:
    """
    Email username pattern: letters/digits, optionally dots between.
    Pattern: [a-z][a-z0-9]*(\.[a-z0-9]+)*
    """
    return RegexOracle(
        r"[a-z][a-z0-9]*(\.[a-z0-9]+)*",
        alphabet=set(string.ascii_lowercase + string.digits + ".")
    )


def create_identifier_oracle() -> RegexOracle:
    """
    Programming identifier: starts with letter, then letters/digits/underscore.
    Pattern: [a-zA-Z_][a-zA-Z0-9_]*
    """
    return RegexOracle(
        r"[a-zA-Z_][a-zA-Z0-9_]*",
        alphabet=set(string.ascii_letters + string.digits + "_")
    )


def create_binary_divisible_by_3_oracle() -> StatechartOracle:
    """
    Binary strings divisible by 3.
    Classic DFA example with 3 states for remainder.
    """
    states = [
        SimpleState("q0", is_accepting=True, transitions={"0": "q0", "1": "q1"}),
        SimpleState("q1", is_accepting=False, transitions={"0": "q2", "1": "q0"}),
        SimpleState("q2", is_accepting=False, transitions={"0": "q1", "1": "q2"}),
    ]
    return StatechartOracle(states, "q0", {"0", "1"}, "BinaryDiv3")


def create_balanced_parens_oracle(max_depth: int = 3) -> RegexOracle:
    """
    Balanced parentheses up to max_depth.
    Note: True balanced parens aren't regular, this is an approximation.
    """
    # Build nested pattern up to max_depth
    pattern = ""
    for _ in range(max_depth):
        pattern = f"(\\({pattern}\\))*"
    return RegexOracle(pattern, alphabet={"(", ")"})


def create_phone_number_oracle() -> RegexOracle:
    """
    Simple phone number: XXX-XXX-XXXX
    """
    return RegexOracle(
        r"[0-9]{3}-[0-9]{3}-[0-9]{4}",
        alphabet=set(string.digits + "-")
    )


def create_even_zeros_oracle() -> StatechartOracle:
    """
    Binary strings with even number of zeros.
    """
    states = [
        SimpleState("even", is_accepting=True, transitions={"0": "odd", "1": "even"}),
        SimpleState("odd", is_accepting=False, transitions={"0": "even", "1": "odd"}),
    ]
    return StatechartOracle(states, "even", {"0", "1"}, "EvenZeros")


def create_no_consecutive_oracle() -> RegexOracle:
    """
    Binary strings with no consecutive 1s.
    Pattern: (0|10)*1?
    """
    return RegexOracle(r"(0|10)*1?", alphabet={"0", "1"})


def create_ab_star_oracle() -> RegexOracle:
    """
    Simple (ab)* pattern - alternating a and b.
    """
    return RegexOracle(r"(ab)*", alphabet={"a", "b"})


def create_ends_with_ing_oracle() -> RegexOracle:
    """
    Words ending in 'ing'.
    """
    return RegexOracle(r"[a-z]*ing", alphabet=set(string.ascii_lowercase))


# =============================================================================
# Oracle Collection for Benchmarks
# =============================================================================

BENCHMARK_ORACLES = {
    "ab_star": create_ab_star_oracle,
    "even_zeros": create_even_zeros_oracle,
    "no_consecutive": create_no_consecutive_oracle,
    "binary_div3": create_binary_divisible_by_3_oracle,
    "identifier": create_identifier_oracle,
    "email_prefix": create_email_prefix_oracle,
    "ends_with_ing": create_ends_with_ing_oracle,
    "phone_number": create_phone_number_oracle,
}


def get_oracle(name: str) -> Oracle:
    """Get a benchmark oracle by name."""
    if name not in BENCHMARK_ORACLES:
        raise ValueError(f"Unknown oracle: {name}. Available: {list(BENCHMARK_ORACLES.keys())}")
    return BENCHMARK_ORACLES[name]()


def list_oracles() -> List[str]:
    """List available benchmark oracles."""
    return list(BENCHMARK_ORACLES.keys())


# =============================================================================
# String Generation Utilities
# =============================================================================

def generate_random_string(alphabet: Set[str], max_length: int = 10) -> str:
    """Generate a random string from alphabet."""
    length = random.randint(0, max_length)
    chars = list(alphabet)
    return "".join(random.choice(chars) for _ in range(length))


def generate_string_pool(
    alphabet: Set[str],
    n_strings: int = 1000,
    max_length: int = 10
) -> List[str]:
    """Generate a pool of random strings for querying."""
    return [generate_random_string(alphabet, max_length) for _ in range(n_strings)]


def generate_systematic_strings(
    alphabet: Set[str],
    max_length: int = 5
) -> List[str]:
    """Generate all strings up to max_length systematically."""
    chars = sorted(alphabet)
    strings = [""]  # Include empty string

    for length in range(1, max_length + 1):
        # Generate all strings of this length
        def gen(prefix: str, remaining: int):
            if remaining == 0:
                strings.append(prefix)
                return
            for c in chars:
                gen(prefix + c, remaining - 1)
        gen("", length)

    return strings


# =============================================================================
# Demo
# =============================================================================

if __name__ == "__main__":
    print("Oracle Interface Demo")
    print("=" * 50)

    # Test regex oracle
    oracle = create_ab_star_oracle()
    test_strings = ["", "ab", "abab", "a", "b", "aba", "aabb"]
    print(f"\nOracle: {oracle.name}")
    for s in test_strings:
        result = oracle.query(s)
        print(f"  '{s}' -> {result}")
    print(f"  Query count: {oracle.query_count}")

    # Test statechart oracle
    oracle2 = create_even_zeros_oracle()
    test_strings2 = ["", "1", "0", "00", "01", "10", "11", "000", "001"]
    print(f"\nOracle: {oracle2.name}")
    for s in test_strings2:
        result = oracle2.query(s)
        print(f"  '{s}' -> {result}")

    # List all oracles
    print(f"\nAvailable oracles: {list_oracles()}")
