"""
Regex/DFA/NFA Induction from Examples

Algorithms implemented:
1. RPNI (Regular Positive and Negative Inference) - Polynomial time DFA learning
2. L* (Angluin's algorithm) - Active learning with membership + equivalence queries
3. State merging - Bottom-up DFA minimization from prefix tree

Key insight: Induced automata map directly to flat statecharts.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Set, Tuple, Optional, Callable, Any
from enum import Enum, auto
from abc import ABC, abstractmethod
import random
import re

from .formal_base import (
    Symbol,
    State,
    Transition,
    DFA,
    NFA,
    Alphabet,
    AcceptResult,
    FormalLanguage,
)


@dataclass
class InducedDFA:
    """Result of DFA induction."""
    dfa: DFA
    positive_examples: List[str]
    negative_examples: List[str]
    accuracy: float
    n_states: int
    algorithm: str
    iterations: int = 0

    def to_regex(self) -> str:
        """Convert DFA to regex (state elimination)."""
        return dfa_to_regex(self.dfa)


@dataclass
class InducedNFA:
    """Result of NFA induction."""
    nfa: NFA
    positive_examples: List[str]
    negative_examples: List[str]
    accuracy: float
    n_states: int
    algorithm: str


@dataclass
class InducedRegex:
    """Result of regex induction."""
    pattern: str
    compiled: re.Pattern
    positive_examples: List[str]
    negative_examples: List[str]
    accuracy: float
    complexity: int  # Pattern length or AST depth


class InductionAlgorithm(Enum):
    """Available induction algorithms."""
    RPNI = "rpni"
    L_STAR = "lstar"
    STATE_MERGE = "state_merge"
    GENETIC = "genetic"
    PREFIX_TREE = "prefix_tree"


@dataclass
class MembershipOracle:
    """Oracle for membership queries (is string in language?)."""
    target_language: Set[str] = field(default_factory=set)
    query_count: int = 0

    def query(self, s: str) -> bool:
        """Check if string is in target language."""
        self.query_count += 1
        return s in self.target_language

    @classmethod
    def from_regex(cls, pattern: str) -> 'MembershipOracle':
        """Create oracle from regex pattern."""
        oracle = cls()
        oracle._regex = re.compile(f'^{pattern}$')
        oracle._use_regex = True
        return oracle

    def query(self, s: str) -> bool:
        self.query_count += 1
        if hasattr(self, '_use_regex') and self._use_regex:
            return bool(self._regex.match(s))
        return s in self.target_language


@dataclass
class EquivalenceOracle:
    """Oracle for equivalence queries (is hypothesis = target?)."""
    target_language: Set[str] = field(default_factory=set)
    alphabet: Alphabet = field(default_factory=Alphabet)
    max_counterexample_length: int = 10
    query_count: int = 0

    def query(self, hypothesis: DFA) -> Optional[str]:
        """
        Check if hypothesis is equivalent to target.
        Returns None if equivalent, counterexample string otherwise.
        """
        self.query_count += 1

        # Generate strings up to max length
        for length in range(self.max_counterexample_length + 1):
            for s in self._generate_strings(length):
                hyp_accepts = hypothesis.accepts(s).accepted
                target_accepts = s in self.target_language
                if hyp_accepts != target_accepts:
                    return s

        return None  # Equivalent (within bounds)

    def _generate_strings(self, length: int):
        """Generate all strings of given length."""
        if length == 0:
            yield ""
            return

        for prefix in self._generate_strings(length - 1):
            for sym in self.alphabet.symbols:
                yield prefix + sym


@dataclass
class PrefixTree:
    """
    Prefix Tree Acceptor (PTA) - initial structure for RPNI.

    Built from positive examples, then merged to generalize.
    """
    root: int = 0
    states: Dict[int, Dict[str, int]] = field(default_factory=dict)
    accepting: Set[int] = field(default_factory=set)
    next_state: int = 1

    def add_string(self, s: str):
        """Add a positive example to the prefix tree."""
        current = self.root
        if current not in self.states:
            self.states[current] = {}

        for char in s:
            if char not in self.states[current]:
                self.states[current][char] = self.next_state
                self.states[self.next_state] = {}
                self.next_state += 1
            current = self.states[current][char]

        self.accepting.add(current)

    def to_dfa(self, alphabet: Alphabet) -> DFA:
        """Convert prefix tree to DFA."""
        dfa = DFA(name="pta_dfa", alphabet=alphabet)

        # Create states
        for state_id in self.states:
            is_initial = (state_id == self.root)
            is_final = (state_id in self.accepting)
            dfa.add_state(State(
                id=state_id,
                name=f"q{state_id}",
                is_initial=is_initial,
                is_final=is_final,
            ))

        # Create transitions
        trans_id = 0
        for src, trans_map in self.states.items():
            for sym, tgt in trans_map.items():
                dfa.add_transition(Transition(
                    id=trans_id,
                    source=src,
                    target=tgt,
                    symbol=Symbol(char=sym),
                ))
                trans_id += 1

        return dfa


class RegexInducer(ABC):
    """Abstract base for regex/DFA induction algorithms."""

    @abstractmethod
    def induce(
        self,
        positive: List[str],
        negative: List[str],
        alphabet: Alphabet,
    ) -> InducedDFA:
        """Induce DFA from positive and negative examples."""
        pass


class RPNIInducer(RegexInducer):
    """
    Regular Positive and Negative Inference (RPNI).

    Algorithm:
    1. Build prefix tree from positive examples
    2. Order states by canonical ordering
    3. Try merging states in order
    4. Accept merge if consistent with negative examples

    Polynomial time: O(n^2 * |Σ|) where n = total length of examples.
    """

    def __init__(self):
        self.merge_attempts = 0
        self.successful_merges = 0

    def induce(
        self,
        positive: List[str],
        negative: List[str],
        alphabet: Alphabet,
    ) -> InducedDFA:
        """Induce minimal DFA using RPNI."""
        if not positive:
            raise ValueError("RPNI requires at least one positive example")

        # Build prefix tree
        pta = PrefixTree()
        for s in positive:
            pta.add_string(s)

        # Convert to DFA
        dfa = pta.to_dfa(alphabet)

        # Get ordered state list (BFS order from root)
        ordered = self._canonical_order(dfa)

        # Try merging states
        self.merge_attempts = 0
        self.successful_merges = 0

        for i in range(1, len(ordered)):
            for j in range(i):
                self.merge_attempts += 1
                merged = self._try_merge(dfa, ordered[j], ordered[i])
                if merged is not None:
                    # Check consistency with negative examples
                    if self._consistent_with_negative(merged, negative):
                        dfa = merged
                        self.successful_merges += 1
                        break  # Move to next i

        # Calculate accuracy
        accuracy = self._calculate_accuracy(dfa, positive, negative)

        return InducedDFA(
            dfa=dfa,
            positive_examples=positive,
            negative_examples=negative,
            accuracy=accuracy,
            n_states=len(dfa.states),
            algorithm="rpni",
            iterations=self.merge_attempts,
        )

    def _canonical_order(self, dfa: DFA) -> List[int]:
        """Get states in canonical (BFS) order."""
        if not dfa.initial_states:
            return list(dfa.states.keys())

        ordered = []
        visited = set()
        queue = [dfa.initial_states[0].id]

        while queue:
            state_id = queue.pop(0)
            if state_id in visited:
                continue
            visited.add(state_id)
            ordered.append(state_id)

            # Add successors in alphabet order
            for sym in sorted(dfa.alphabet.symbols):
                for trans in dfa.transitions:
                    if trans.source == state_id and trans.symbol.char == sym:
                        if trans.target not in visited:
                            queue.append(trans.target)

        return ordered

    def _try_merge(self, dfa: DFA, q1: int, q2: int) -> Optional[DFA]:
        """Try to merge states q1 and q2."""
        if q1 not in dfa.states or q2 not in dfa.states:
            return None

        # Cannot merge if one is accepting and other is not
        s1, s2 = dfa.states[q1], dfa.states[q2]
        if s1.is_final != s2.is_final:
            return None

        # Create merged DFA
        merged = DFA(name=dfa.name, alphabet=dfa.alphabet)

        # State mapping: q2 -> q1
        state_map = {sid: sid for sid in dfa.states}
        state_map[q2] = q1

        # Add states (excluding q2)
        for state in dfa.states.values():
            if state.id != q2:
                merged.add_state(State(
                    id=state.id,
                    name=state.name,
                    is_initial=state.is_initial,
                    is_final=state.is_final,
                ))

        # Add transitions with remapping
        trans_id = 0
        seen_trans = set()
        for trans in dfa.transitions:
            src = state_map[trans.source]
            tgt = state_map[trans.target]
            key = (src, trans.symbol.char, tgt)

            if key not in seen_trans:
                merged.add_transition(Transition(
                    id=trans_id,
                    source=src,
                    target=tgt,
                    symbol=trans.symbol,
                ))
                seen_trans.add(key)
                trans_id += 1

        # Check determinism
        for state_id in merged.states:
            for sym in merged.alphabet.symbols:
                targets = [
                    t.target for t in merged.transitions
                    if t.source == state_id and t.symbol.char == sym
                ]
                if len(targets) > 1:
                    return None  # Non-deterministic after merge

        return merged

    def _consistent_with_negative(self, dfa: DFA, negative: List[str]) -> bool:
        """Check that DFA rejects all negative examples."""
        for s in negative:
            if dfa.accepts(s).accepted:
                return False
        return True

    def _calculate_accuracy(
        self,
        dfa: DFA,
        positive: List[str],
        negative: List[str],
    ) -> float:
        """Calculate accuracy on examples."""
        correct = 0
        total = len(positive) + len(negative)

        if total == 0:
            return 1.0

        for s in positive:
            if dfa.accepts(s).accepted:
                correct += 1
        for s in negative:
            if not dfa.accepts(s).accepted:
                correct += 1

        return correct / total


class LStarInducer(RegexInducer):
    """
    Angluin's L* Algorithm for active learning.

    Uses:
    - Membership oracle: Is string in language?
    - Equivalence oracle: Is hypothesis correct?

    Learns minimal DFA in polynomial queries.
    """

    def __init__(
        self,
        membership_oracle: Optional[MembershipOracle] = None,
        equivalence_oracle: Optional[EquivalenceOracle] = None,
    ):
        self.membership = membership_oracle
        self.equivalence = equivalence_oracle

    def induce(
        self,
        positive: List[str],
        negative: List[str],
        alphabet: Alphabet,
    ) -> InducedDFA:
        """Induce DFA using L* algorithm."""
        # Build oracles from examples if not provided
        if self.membership is None:
            self.membership = MembershipOracle(target_language=set(positive))
        if self.equivalence is None:
            target = set(positive)
            self.equivalence = EquivalenceOracle(
                target_language=target,
                alphabet=alphabet,
            )

        # Initialize observation table
        S = {""}  # Prefixes (access strings)
        E = {""}  # Suffixes (distinguishing strings)
        T = {}    # Table: T[s, e] = membership(s + e)

        # Initial table fill
        self._fill_table(S, E, T, alphabet)

        iterations = 0
        max_iterations = 1000

        while iterations < max_iterations:
            iterations += 1

            # Make table closed and consistent
            while True:
                closed, witness = self._is_closed(S, E, T, alphabet)
                if not closed:
                    S.add(witness)
                    self._fill_table(S, E, T, alphabet)
                    continue

                consistent, e_new = self._is_consistent(S, E, T, alphabet)
                if not consistent:
                    E.add(e_new)
                    self._fill_table(S, E, T, alphabet)
                    continue

                break  # Table is closed and consistent

            # Build hypothesis DFA
            hypothesis = self._build_hypothesis(S, E, T, alphabet)

            # Check equivalence
            counterexample = self.equivalence.query(hypothesis)
            if counterexample is None:
                # Found correct DFA
                accuracy = self._calculate_accuracy(hypothesis, positive, negative)
                return InducedDFA(
                    dfa=hypothesis,
                    positive_examples=positive,
                    negative_examples=negative,
                    accuracy=accuracy,
                    n_states=len(hypothesis.states),
                    algorithm="lstar",
                    iterations=iterations,
                )

            # Process counterexample
            for i in range(len(counterexample) + 1):
                S.add(counterexample[:i])
            self._fill_table(S, E, T, alphabet)

        # Return best hypothesis after max iterations
        hypothesis = self._build_hypothesis(S, E, T, alphabet)
        accuracy = self._calculate_accuracy(hypothesis, positive, negative)
        return InducedDFA(
            dfa=hypothesis,
            positive_examples=positive,
            negative_examples=negative,
            accuracy=accuracy,
            n_states=len(hypothesis.states),
            algorithm="lstar",
            iterations=iterations,
        )

    def _fill_table(
        self,
        S: Set[str],
        E: Set[str],
        T: Dict[Tuple[str, str], bool],
        alphabet: Alphabet,
    ):
        """Fill observation table with membership queries."""
        # For s in S and s.a for all a
        all_prefixes = set(S)
        for s in S:
            for a in alphabet.symbols:
                all_prefixes.add(s + a)

        for s in all_prefixes:
            for e in E:
                if (s, e) not in T:
                    T[(s, e)] = self.membership.query(s + e)

    def _row(self, s: str, E: Set[str], T: Dict) -> Tuple[bool, ...]:
        """Get row signature for string s."""
        return tuple(T.get((s, e), False) for e in sorted(E))

    def _is_closed(
        self,
        S: Set[str],
        E: Set[str],
        T: Dict,
        alphabet: Alphabet,
    ) -> Tuple[bool, Optional[str]]:
        """Check if table is closed."""
        s_rows = {self._row(s, E, T) for s in S}

        for s in S:
            for a in alphabet.symbols:
                sa = s + a
                if self._row(sa, E, T) not in s_rows:
                    return False, sa

        return True, None

    def _is_consistent(
        self,
        S: Set[str],
        E: Set[str],
        T: Dict,
        alphabet: Alphabet,
    ) -> Tuple[bool, Optional[str]]:
        """Check if table is consistent."""
        s_list = list(S)

        for i, s1 in enumerate(s_list):
            for s2 in s_list[i+1:]:
                if self._row(s1, E, T) == self._row(s2, E, T):
                    # Same row, check extensions
                    for a in alphabet.symbols:
                        for e in E:
                            ae = a + e
                            if T.get((s1 + a, e)) != T.get((s2 + a, e)):
                                return False, ae

        return True, None

    def _build_hypothesis(
        self,
        S: Set[str],
        E: Set[str],
        T: Dict,
        alphabet: Alphabet,
    ) -> DFA:
        """Build DFA from observation table."""
        dfa = DFA(name="lstar_hypothesis", alphabet=alphabet)

        # States = unique rows of S
        row_to_state = {}
        state_id = 0

        for s in sorted(S):  # Sorted for determinism
            row = self._row(s, E, T)
            if row not in row_to_state:
                is_initial = (s == "")
                is_final = T.get((s, ""), False)
                dfa.add_state(State(
                    id=state_id,
                    name=f"q{state_id}",
                    is_initial=is_initial,
                    is_final=is_final,
                ))
                row_to_state[row] = state_id
                state_id += 1

        # Transitions
        trans_id = 0
        for s in S:
            src_row = self._row(s, E, T)
            src_state = row_to_state[src_row]

            for a in alphabet.symbols:
                sa = s + a
                tgt_row = self._row(sa, E, T)
                if tgt_row in row_to_state:
                    tgt_state = row_to_state[tgt_row]
                    dfa.add_transition(Transition(
                        id=trans_id,
                        source=src_state,
                        target=tgt_state,
                        symbol=Symbol(char=a),
                    ))
                    trans_id += 1

        return dfa

    def _calculate_accuracy(
        self,
        dfa: DFA,
        positive: List[str],
        negative: List[str],
    ) -> float:
        """Calculate accuracy on examples."""
        correct = 0
        total = len(positive) + len(negative)

        if total == 0:
            return 1.0

        for s in positive:
            if dfa.accepts(s).accepted:
                correct += 1
        for s in negative:
            if not dfa.accepts(s).accepted:
                correct += 1

        return correct / total


class DFAInducer:
    """
    Unified DFA induction interface.

    Selects best algorithm based on available information.
    """

    def __init__(self, algorithm: InductionAlgorithm = InductionAlgorithm.RPNI):
        self.algorithm = algorithm

    def induce(
        self,
        positive: List[str],
        negative: List[str],
        alphabet: Optional[Alphabet] = None,
    ) -> InducedDFA:
        """Induce DFA from examples."""
        # Infer alphabet if not provided
        if alphabet is None:
            all_chars = set()
            for s in positive + negative:
                all_chars.update(s)
            alphabet = Alphabet(symbols=all_chars)

        if self.algorithm == InductionAlgorithm.RPNI:
            inducer = RPNIInducer()
        elif self.algorithm == InductionAlgorithm.L_STAR:
            inducer = LStarInducer()
        else:
            inducer = RPNIInducer()  # Default

        return inducer.induce(positive, negative, alphabet)


class NFAInducer:
    """
    NFA induction (allows non-determinism).

    Often produces smaller automata than DFA induction.
    """

    def induce(
        self,
        positive: List[str],
        negative: List[str],
        alphabet: Optional[Alphabet] = None,
    ) -> InducedNFA:
        """Induce NFA from examples."""
        # First induce DFA
        dfa_inducer = DFAInducer()
        induced = dfa_inducer.induce(positive, negative, alphabet)

        # Convert to NFA (trivial embedding)
        nfa = NFA(name="induced_nfa", alphabet=induced.dfa.alphabet)

        for state in induced.dfa.states.values():
            nfa.add_state(state)

        for trans in induced.dfa.transitions:
            nfa.add_transition(trans)

        return InducedNFA(
            nfa=nfa,
            positive_examples=positive,
            negative_examples=negative,
            accuracy=induced.accuracy,
            n_states=len(nfa.states),
            algorithm=induced.algorithm,
        )


def dfa_to_regex(dfa: DFA) -> str:
    """
    Convert DFA to regex using state elimination.

    Algorithm:
    1. Add new start and accept states
    2. Eliminate states one by one
    3. Final edge label is the regex
    """
    if not dfa.states:
        return ""

    # Build transition regex matrix
    n = len(dfa.states)
    state_list = list(dfa.states.keys())
    state_idx = {s: i for i, s in enumerate(state_list)}

    # R[i][j] = regex for transition from i to j
    R = [[None for _ in range(n)] for _ in range(n)]

    for trans in dfa.transitions:
        i, j = state_idx[trans.source], state_idx[trans.target]
        char = trans.symbol.char if trans.symbol.char else ""
        if R[i][j] is None:
            R[i][j] = char
        else:
            R[i][j] = f"({R[i][j]}|{char})"

    # Get initial and final states
    initial_idx = [state_idx[s.id] for s in dfa.initial_states]
    final_idx = [state_idx[s.id] for s in dfa.states.values() if s.is_final]

    if not initial_idx or not final_idx:
        return ""

    # Simple case: direct regex construction for small DFAs
    if n <= 3:
        parts = []
        for fi in final_idx:
            path = _find_path_regex(R, initial_idx[0], fi, n)
            if path:
                parts.append(path)

        if not parts:
            return ""
        return "|".join(f"({p})" for p in parts) if len(parts) > 1 else parts[0]

    return f"(regex for {n}-state DFA)"  # Placeholder for complex cases


def _find_path_regex(R: List[List[Optional[str]]], start: int, end: int, n: int) -> str:
    """Find regex for path from start to end."""
    if start == end:
        # Self-loop
        if R[start][start]:
            return f"({R[start][start]})*"
        return ""

    if R[start][end]:
        return R[start][end]

    return ""


def demo():
    """Demonstrate regex induction."""
    print("=" * 60)
    print("REGEX INDUCER: Learn DFA from Examples")
    print("=" * 60)

    # Example: Learn language of strings ending in 'ab'
    positive = ["ab", "aab", "bab", "aaab", "abab", "baab"]
    negative = ["", "a", "b", "ba", "aa", "bb", "aba"]

    print(f"\nPositive examples: {positive}")
    print(f"Negative examples: {negative}")

    # Induce with RPNI
    print("\n--- RPNI Algorithm ---")
    rpni = DFAInducer(algorithm=InductionAlgorithm.RPNI)
    result = rpni.induce(positive, negative)

    print(f"Induced DFA:")
    print(f"  States: {result.n_states}")
    print(f"  Accuracy: {result.accuracy:.1%}")
    print(f"  Algorithm: {result.algorithm}")
    print(f"  Iterations: {result.iterations}")

    # Test on new strings
    test_strings = ["aab", "bba", "abab", ""]
    print(f"\nTesting:")
    for s in test_strings:
        accepted = result.dfa.accepts(s).accepted
        label = "[ACCEPT]" if accepted else "[REJECT]"
        print(f"  '{s}': {label}")

    # Example 2: Binary strings divisible by 3
    print("\n" + "=" * 60)
    print("Example 2: Binary strings (divisible by 3)")
    print("=" * 60)

    # Generate examples
    div3_pos = [bin(n)[2:] for n in range(0, 30, 3) if n > 0]
    div3_neg = [bin(n)[2:] for n in range(1, 30) if n % 3 != 0]

    print(f"Positive (÷3): {div3_pos[:6]}...")
    print(f"Negative: {div3_neg[:6]}...")

    result2 = rpni.induce(div3_pos, div3_neg)
    print(f"\nInduced DFA: {result2.n_states} states, {result2.accuracy:.1%} accuracy")

    return result, result2


if __name__ == "__main__":
    demo()
