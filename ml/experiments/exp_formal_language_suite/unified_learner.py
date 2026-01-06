"""
Unified Formal Language Learning Framework

Provides single interface for learning:
- Regular languages (DFA, NFA, Regex)
- Context-free languages (CFG, PDA)

Key insight: All formal languages map to statecharts:
- DFA = flat statechart
- NFA = statechart with epsilon-transitions
- CFG = hierarchical statechart with recursive substates
- PDA = statechart with stack context variable
"""

from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional, Any, Union
from enum import Enum, auto
import time

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
from .regex_inducer import (
    DFAInducer,
    NFAInducer,
    InducedDFA,
    InducedNFA,
    InductionAlgorithm,
)
from .grammar_inducer import (
    CFGInducer,
    InducedCFG,
    CFG,
    Production,
    Terminal,
    NonTerminal,
)


class LanguageType(Enum):
    """Type of formal language in Chomsky hierarchy."""
    REGULAR = "regular"           # Type 3: DFA/NFA/Regex
    CONTEXT_FREE = "context_free"  # Type 2: CFG/PDA
    CONTEXT_SENSITIVE = "context_sensitive"  # Type 1: LBA
    RECURSIVELY_ENUMERABLE = "recursively_enumerable"  # Type 0: TM


class LanguageComplexity(Enum):
    """Complexity classification for learned languages."""
    TRIVIAL = "trivial"      # Single string or empty
    FINITE = "finite"        # Finite set of strings
    REGULAR = "regular"      # Infinite but regular
    CONTEXT_FREE = "cf"      # Requires pushdown
    UNKNOWN = "unknown"


@dataclass
class LearningResult:
    """Unified result from language learning."""
    language_type: LanguageType
    complexity: LanguageComplexity

    # Learned representation (one of these)
    dfa: Optional[DFA] = None
    nfa: Optional[NFA] = None
    cfg: Optional[CFG] = None

    # Metrics
    accuracy: float = 0.0
    n_states: int = 0
    n_productions: int = 0
    learning_time_ms: float = 0.0

    # Metadata
    algorithm_used: str = ""
    positive_examples: List[str] = field(default_factory=list)
    negative_examples: List[str] = field(default_factory=list)

    def to_statechart(self) -> Dict[str, Any]:
        """Convert learned language to statechart format."""
        if self.dfa:
            return self._dfa_to_statechart(self.dfa)
        elif self.nfa:
            return self._nfa_to_statechart(self.nfa)
        elif self.cfg:
            return self._cfg_to_statechart(self.cfg)
        return {}

    def _dfa_to_statechart(self, dfa: DFA) -> Dict[str, Any]:
        """Convert DFA to flat statechart."""
        states = []
        for s in dfa.states.values():
            states.append({
                "label": s.name,
                "type": 1,  # BASIC
                "is_initial": s.is_initial,
                "is_final": s.is_final,
            })

        transitions = []
        for t in dfa.transitions:
            transitions.append({
                "from": [dfa.states[t.source].name],
                "to": [dfa.states[t.target].name],
                "event": t.symbol.char if t.symbol.char else "",
            })

        return {
            "name": dfa.name,
            "root_state": {
                "label": "__root__",
                "type": 2,  # NORMAL (OR-state)
                "children": states,
            },
            "transitions": transitions,
        }

    def _nfa_to_statechart(self, nfa: NFA) -> Dict[str, Any]:
        """Convert NFA to statechart with epsilon transitions."""
        sc = self._dfa_to_statechart(DFA(
            name=nfa.name,
            alphabet=nfa.alphabet,
        ))

        # Add epsilon transitions as internal events
        for t in nfa.transitions:
            if t.symbol.is_epsilon:
                sc["transitions"].append({
                    "from": [nfa.states[t.source].name],
                    "to": [nfa.states[t.target].name],
                    "event": "",  # Epsilon = no event (completion)
                })

        return sc

    def _cfg_to_statechart(self, cfg: CFG) -> Dict[str, Any]:
        """Convert CFG to hierarchical statechart."""
        # Each non-terminal becomes a composite state
        # Each production becomes a transition path

        def build_state(nt: NonTerminal, depth: int = 0) -> Dict:
            if depth > 5:  # Prevent infinite recursion
                return {"label": nt.name, "type": 1}

            children = []
            for prod in cfg.productions:
                if prod.lhs.name == nt.name:
                    # Create sequence of states for RHS
                    seq_states = []
                    for i, sym in enumerate(prod.rhs):
                        if isinstance(sym, Terminal):
                            seq_states.append({
                                "label": f"{nt.name}_{prod.id}_{i}",
                                "type": 1,
                                "metadata": {"terminal": sym.value},
                            })
                        elif isinstance(sym, NonTerminal):
                            seq_states.append(build_state(sym, depth + 1))

                    if seq_states:
                        children.append({
                            "label": f"{nt.name}_prod{prod.id}",
                            "type": 2,  # Sequence
                            "children": seq_states,
                        })

            return {
                "label": nt.name,
                "type": 2,  # NORMAL (OR-state for alternatives)
                "children": children if children else [{"label": f"{nt.name}_empty", "type": 1}],
            }

        root = build_state(cfg.start_symbol) if cfg.start_symbol else {"label": "S", "type": 1}

        return {
            "name": cfg.name,
            "root_state": root,
            "transitions": [],
        }

    def accepts(self, s: str) -> bool:
        """Check if learned language accepts string."""
        if self.dfa:
            return self.dfa.accepts(s).accepted
        elif self.nfa:
            return self.nfa.accepts(s).accepted
        elif self.cfg:
            # Simplified: check if string matches any generated string
            for _ in range(100):
                if self.cfg.generate(max_depth=len(s) + 5) == s:
                    return True
            return False
        return False


class UnifiedLearner:
    """
    Unified framework for learning formal languages.

    Automatically selects appropriate algorithm based on:
    1. Available examples (positive only vs positive+negative)
    2. Example complexity (finite vs infinite language hints)
    3. Structure detection (nested brackets suggest CFG)
    """

    def __init__(
        self,
        prefer_type: Optional[LanguageType] = None,
        algorithms: Optional[Dict[LanguageType, str]] = None,
    ):
        self.prefer_type = prefer_type
        self.algorithms = algorithms or {
            LanguageType.REGULAR: "rpni",
            LanguageType.CONTEXT_FREE: "sequitur",
        }

    def learn(
        self,
        positive: List[str],
        negative: Optional[List[str]] = None,
    ) -> LearningResult:
        """Learn language from examples."""
        negative = negative or []
        t0 = time.time()

        # Classify language complexity
        complexity = self._classify_complexity(positive, negative)

        # Choose language type
        if self.prefer_type:
            lang_type = self.prefer_type
        else:
            lang_type = self._infer_language_type(positive, complexity)

        # Learn based on type
        if lang_type == LanguageType.REGULAR:
            result = self._learn_regular(positive, negative)
        elif lang_type == LanguageType.CONTEXT_FREE:
            result = self._learn_context_free(positive)
        else:
            # Default to regular
            result = self._learn_regular(positive, negative)

        result.learning_time_ms = (time.time() - t0) * 1000
        result.positive_examples = positive
        result.negative_examples = negative

        return result

    def _classify_complexity(
        self,
        positive: List[str],
        negative: List[str],
    ) -> LanguageComplexity:
        """Classify language complexity from examples."""
        if not positive:
            return LanguageComplexity.TRIVIAL

        if len(positive) == 1 and not negative:
            return LanguageComplexity.TRIVIAL

        # Check for finite patterns
        unique_lengths = set(len(s) for s in positive)
        if len(unique_lengths) <= 3 and len(positive) < 20:
            return LanguageComplexity.FINITE

        # Check for nested structure (suggests CFG)
        if self._has_nested_structure(positive):
            return LanguageComplexity.CONTEXT_FREE

        return LanguageComplexity.REGULAR

    def _has_nested_structure(self, examples: List[str]) -> bool:
        """Check if examples suggest nested structure."""
        brackets = {"(": ")", "[": "]", "{": "}"}

        for ex in examples:
            stack = []
            has_nesting = False

            for c in ex:
                if c in brackets:
                    stack.append(c)
                    if len(stack) > 1:
                        has_nesting = True
                elif c in brackets.values():
                    if stack:
                        stack.pop()

            if has_nesting:
                return True

        return False

    def _infer_language_type(
        self,
        positive: List[str],
        complexity: LanguageComplexity,
    ) -> LanguageType:
        """Infer appropriate language type."""
        if complexity == LanguageComplexity.CONTEXT_FREE:
            return LanguageType.CONTEXT_FREE

        # Check for recursive patterns
        if self._has_recursive_patterns(positive):
            return LanguageType.CONTEXT_FREE

        return LanguageType.REGULAR

    def _has_recursive_patterns(self, examples: List[str]) -> bool:
        """Check for recursive patterns like anbn."""
        for ex in examples:
            # Check for balanced patterns
            if len(ex) >= 4:
                mid = len(ex) // 2
                first_half = ex[:mid]
                second_half = ex[mid:]

                # Check if halves are related
                if len(set(first_half)) == 1 and len(set(second_half)) == 1:
                    if first_half[0] != second_half[0]:
                        return True

        return False

    def _learn_regular(
        self,
        positive: List[str],
        negative: List[str],
    ) -> LearningResult:
        """Learn regular language (DFA)."""
        algorithm = self.algorithms.get(LanguageType.REGULAR, "rpni")

        if algorithm == "rpni":
            algo_enum = InductionAlgorithm.RPNI
        elif algorithm == "lstar":
            algo_enum = InductionAlgorithm.L_STAR
        else:
            algo_enum = InductionAlgorithm.RPNI

        inducer = DFAInducer(algorithm=algo_enum)
        induced = inducer.induce(positive, negative)

        return LearningResult(
            language_type=LanguageType.REGULAR,
            complexity=LanguageComplexity.REGULAR,
            dfa=induced.dfa,
            accuracy=induced.accuracy,
            n_states=induced.n_states,
            algorithm_used=induced.algorithm,
        )

    def _learn_context_free(self, positive: List[str]) -> LearningResult:
        """Learn context-free language (CFG)."""
        algorithm = self.algorithms.get(LanguageType.CONTEXT_FREE, "sequitur")

        inducer = CFGInducer(algorithm=algorithm)
        induced = inducer.induce(positive)

        return LearningResult(
            language_type=LanguageType.CONTEXT_FREE,
            complexity=LanguageComplexity.CONTEXT_FREE,
            cfg=induced.cfg,
            accuracy=induced.accuracy,
            n_productions=induced.n_productions,
            algorithm_used=induced.algorithm,
        )


def learn_from_examples(
    positive: List[str],
    negative: Optional[List[str]] = None,
    language_type: Optional[LanguageType] = None,
) -> LearningResult:
    """
    Convenience function to learn language from examples.

    Args:
        positive: Strings that should be accepted
        negative: Strings that should be rejected (optional)
        language_type: Force specific language type (optional)

    Returns:
        LearningResult with learned automaton/grammar
    """
    learner = UnifiedLearner(prefer_type=language_type)
    return learner.learn(positive, negative)


def demo():
    """Demonstrate unified learning."""
    print("=" * 60)
    print("UNIFIED LEARNER: Automatic Language Learning")
    print("=" * 60)

    # Example 1: Regular language (strings ending in 'ab')
    print("\n--- Example 1: Regular Language ---")
    positive1 = ["ab", "aab", "bab", "aaab", "abab"]
    negative1 = ["", "a", "b", "ba", "aa"]

    result1 = learn_from_examples(positive1, negative1)
    print(f"Detected type: {result1.language_type.value}")
    print(f"Complexity: {result1.complexity.value}")
    print(f"Algorithm: {result1.algorithm_used}")
    print(f"States: {result1.n_states}")
    print(f"Accuracy: {result1.accuracy:.1%}")
    print(f"Time: {result1.learning_time_ms:.1f}ms")

    # Test
    test1 = ["ab", "bba", "aab"]
    print(f"Test results:")
    for s in test1:
        print(f"  '{s}': {result1.accepts(s)}")

    # Example 2: Context-free (nested parens)
    print("\n--- Example 2: Context-Free Language ---")
    positive2 = ["()", "(())", "((()))", "(()())"]

    result2 = learn_from_examples(positive2)
    print(f"Detected type: {result2.language_type.value}")
    print(f"Complexity: {result2.complexity.value}")
    print(f"Algorithm: {result2.algorithm_used}")
    print(f"Productions: {result2.n_productions}")
    print(f"Time: {result2.learning_time_ms:.1f}ms")

    # Example 3: Force type
    print("\n--- Example 3: Forced Regular ---")
    positive3 = ["ab", "abab", "ababab"]

    result3 = learn_from_examples(positive3, language_type=LanguageType.REGULAR)
    print(f"Forced type: {result3.language_type.value}")
    print(f"States: {result3.n_states}")

    # Convert to statechart
    print("\n--- Statechart Conversion ---")
    sc = result1.to_statechart()
    print(f"Statechart name: {sc.get('name')}")
    print(f"Root type: {sc.get('root_state', {}).get('type')}")
    print(f"Transitions: {len(sc.get('transitions', []))}")

    return result1, result2, result3


if __name__ == "__main__":
    demo()
