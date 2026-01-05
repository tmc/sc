"""
End-to-end regex synthesis from examples.

Input: Positive (matching) and negative (non-matching) strings
Output: A minimal statechart that matches the behavior

This is the main entry point for regex learning from examples.
"""

import time
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict

from .regex_statechart import RegexStatechart, StateType, Transition, CharGuard
from .re2_oracle import RE2Oracle, RegexExamples
from .evolver import RegexEvolver, EvolutionConfig, EvolutionStats


@dataclass
class SynthesisResult:
    """Result of regex synthesis."""
    statechart: RegexStatechart
    f1_score: float
    accuracy: float
    n_states: int
    n_transitions: int
    evolution_stats: EvolutionStats
    elapsed_time: float

    # Comparison with original regex (if provided)
    original_pattern: Optional[str] = None
    match_rate: float = 0.0  # Agreement with original on test set

    def summary(self) -> str:
        """Generate summary string."""
        lines = [
            "Synthesis Result:",
            f"  F1 Score: {self.f1_score:.4f}",
            f"  Accuracy: {self.accuracy:.4f}",
            f"  States: {self.n_states}",
            f"  Transitions: {self.n_transitions}",
            f"  Time: {self.elapsed_time:.2f}s",
            f"  Generations: {self.evolution_stats.generations}",
        ]
        if self.original_pattern:
            lines.append(f"  Original: {self.original_pattern}")
            lines.append(f"  Match Rate: {self.match_rate:.4f}")
        return "\n".join(lines)


class RegexSynthesizer:
    """
    Synthesize regex statecharts from examples.

    Main workflow:
    1. Analyze examples to infer alphabet and structure hints
    2. Configure evolution based on complexity
    3. Run evolution to find matching statechart
    4. Optionally minimize the result
    """

    def __init__(self, config: EvolutionConfig = None):
        """Initialize synthesizer."""
        self.config = config or EvolutionConfig()
        self.evolver = RegexEvolver(self.config)

    def synthesize(
        self,
        positive: List[str],
        negative: List[str],
        original_pattern: str = None,
        verbose: bool = True
    ) -> SynthesisResult:
        """
        Synthesize a statechart from examples.

        Args:
            positive: Strings that should match
            negative: Strings that should not match
            original_pattern: Original regex (for comparison only)
            verbose: Print progress

        Returns:
            SynthesisResult with best statechart and metrics
        """
        start_time = time.time()

        if verbose:
            print("=" * 60)
            print("REGEX SYNTHESIS")
            print("=" * 60)
            print(f"Positive examples: {len(positive)}")
            print(f"Negative examples: {len(negative)}")
            if original_pattern:
                print(f"Original pattern: {original_pattern}")

        # Analyze examples
        alphabet = self._infer_alphabet(positive, negative)
        max_len = max(len(s) for s in positive + negative) if positive + negative else 0

        # Adjust config based on complexity
        adjusted_config = self._adjust_config(
            len(positive) + len(negative),
            len(alphabet),
            max_len
        )
        self.evolver = RegexEvolver(adjusted_config)

        if verbose:
            print(f"Inferred alphabet: {alphabet[:30]}{'...' if len(alphabet) > 30 else ''}")
            print(f"Max string length: {max_len}")
            print("-" * 60)

        # Run evolution
        best, stats = self.evolver.evolve(
            positive, negative, alphabet,
            callback=None
        )

        # Evaluate final result
        tp = fp = tn = fn = 0
        for s in positive:
            if best.matches(s):
                tp += 1
            else:
                fn += 1
        for s in negative:
            if best.matches(s):
                fp += 1
            else:
                tn += 1

        total = tp + fp + tn + fn
        accuracy = (tp + tn) / total if total > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        # Compare with original if provided
        match_rate = 0.0
        if original_pattern:
            oracle = RE2Oracle(original_pattern)
            test_strings = positive + negative
            matches = 0
            for s in test_strings:
                oracle_match = oracle.matches(s)
                synth_match = best.matches(s)
                if oracle_match == synth_match:
                    matches += 1
            match_rate = matches / len(test_strings) if test_strings else 0.0

        elapsed = time.time() - start_time

        result = SynthesisResult(
            statechart=best,
            f1_score=f1,
            accuracy=accuracy,
            n_states=best.n_states,
            n_transitions=best.n_transitions,
            evolution_stats=stats,
            elapsed_time=elapsed,
            original_pattern=original_pattern,
            match_rate=match_rate
        )

        if verbose:
            print("\n" + result.summary())
            print("=" * 60)

        return result

    def synthesize_from_pattern(
        self,
        pattern: str,
        n_examples: int = 100,
        verbose: bool = True
    ) -> SynthesisResult:
        """
        Synthesize a statechart from a regex pattern.

        Generates examples from the pattern, then synthesizes a statechart.
        Useful for testing if evolution can recover pattern structure.

        Args:
            pattern: RE2-compatible regex pattern
            n_examples: Number of examples to generate (split pos/neg)
            verbose: Print progress

        Returns:
            SynthesisResult
        """
        oracle = RE2Oracle(pattern)
        examples = oracle.generate_examples(
            n_positive=n_examples // 2,
            n_negative=n_examples // 2
        )

        return self.synthesize(
            examples.positive,
            examples.negative,
            original_pattern=pattern,
            verbose=verbose
        )

    def minimize(self, sc: RegexStatechart, positive: List[str], negative: List[str]) -> RegexStatechart:
        """
        Minimize a statechart while preserving behavior.

        Removes redundant states and transitions.
        """
        # Try removing each state and transition, keep if behavior unchanged
        minimized = sc.copy()

        # Try removing transitions
        i = 0
        while i < len(minimized.transitions):
            test = minimized.copy()
            test.transitions.pop(i)

            # Check if behavior preserved
            if self._behaviors_equal(test, minimized, positive, negative):
                minimized = test
            else:
                i += 1

        # Try removing states (from end, skip START and ACCEPT)
        for idx in range(minimized.n_states - 2, 0, -1):
            if minimized.state_types[idx] == StateType.ACCEPT:
                continue

            test = minimized.copy()
            # Check if state is used
            used = any(
                t.source == idx or t.target == idx
                for t in test.transitions
            )
            if not used:
                # Safe to remove
                test.state_labels.pop(idx)
                test.state_types.pop(idx)
                # Update transition indices
                new_trans = []
                for t in test.transitions:
                    src = t.source if t.source < idx else t.source - 1
                    tgt = t.target if t.target < idx else t.target - 1
                    new_trans.append(Transition(source=src, target=tgt, guard=t.guard))
                test.transitions = new_trans
                minimized = test

        return minimized

    def _behaviors_equal(
        self,
        sc1: RegexStatechart,
        sc2: RegexStatechart,
        positive: List[str],
        negative: List[str]
    ) -> bool:
        """Check if two statecharts have same behavior on examples."""
        if not sc1.is_valid():
            return False

        for s in positive + negative:
            if sc1.matches(s) != sc2.matches(s):
                return False
        return True

    def _infer_alphabet(self, positive: List[str], negative: List[str]) -> str:
        """Infer alphabet from examples."""
        chars = set()
        for s in positive + negative:
            chars.update(s)

        # Add some extras for evolution flexibility
        if any(c.islower() for c in chars):
            chars.update('abcxyz')
        if any(c.isdigit() for c in chars):
            chars.update('0123')

        return ''.join(sorted(chars))

    def _adjust_config(
        self,
        n_examples: int,
        alphabet_size: int,
        max_length: int
    ) -> EvolutionConfig:
        """Adjust evolution config based on problem complexity."""
        config = EvolutionConfig(
            population_size=self.config.population_size,
            n_generations=self.config.n_generations,
            max_states=self.config.max_states,
            verbose=self.config.verbose,
            log_every=self.config.log_every
        )

        # Increase effort for complex patterns
        if max_length > 10 or alphabet_size > 10:
            config.n_generations = min(200, config.n_generations * 2)
            config.population_size = min(100, config.population_size * 2)

        # Increase max states for longer strings
        if max_length > 5:
            config.max_states = min(20, max_length + 5)

        return config


def synthesize_regex(
    positive: List[str],
    negative: List[str],
    verbose: bool = True
) -> RegexStatechart:
    """
    Convenience function: synthesize a statechart from examples.

    Args:
        positive: Strings that should match
        negative: Strings that should not match
        verbose: Print progress

    Returns:
        Best statechart found
    """
    synthesizer = RegexSynthesizer()
    result = synthesizer.synthesize(positive, negative, verbose=verbose)
    return result.statechart


def test_synthesis():
    """Test regex synthesis."""
    print("=" * 60)
    print("REGEX SYNTHESIS TESTS")
    print("=" * 60)

    synthesizer = RegexSynthesizer(EvolutionConfig(
        population_size=30,
        n_generations=50,
        max_states=8,
        verbose=True,
        log_every=10
    ))

    # Test 1: Simple pattern a*
    print("\n1. Synthesizing from pattern 'a*':")
    result1 = synthesizer.synthesize_from_pattern("a*", n_examples=60)
    print(f"\nResulting statechart:")
    print(result1.statechart.to_string())

    # Test with examples
    test_cases = [("", True), ("a", True), ("aa", True), ("b", False)]
    print("\nTest cases:")
    for s, expected in test_cases:
        actual = result1.statechart.matches(s)
        status = "PASS" if actual == expected else "FAIL"
        print(f"  '{s}' -> {actual} (expected {expected}) [{status}]")

    # Test 2: Pattern ab+
    print("\n" + "-" * 60)
    print("2. Synthesizing from pattern 'ab+':")
    result2 = synthesizer.synthesize_from_pattern("ab+", n_examples=80)

    # Test with examples
    test_cases = [("ab", True), ("abb", True), ("a", False), ("b", False)]
    print("\nTest cases:")
    for s, expected in test_cases:
        actual = result2.statechart.matches(s)
        status = "PASS" if actual == expected else "FAIL"
        print(f"  '{s}' -> {actual} (expected {expected}) [{status}]")

    # Test 3: Direct synthesis from examples
    print("\n" + "-" * 60)
    print("3. Direct synthesis from examples:")
    positive = ["cat", "bat", "hat", "sat"]
    negative = ["dog", "ca", "at", "cats", "c", ""]
    print(f"  Positive: {positive}")
    print(f"  Negative: {negative}")

    result3 = synthesizer.synthesize(positive, negative)
    print(f"\nResulting statechart accepts pattern: ?at")
    print(result3.statechart.to_string())

    # Test
    for s in positive + negative:
        print(f"  '{s}' -> {result3.statechart.matches(s)}")

    print("\n" + "=" * 60)
    print("Synthesis tests complete!")
    print("=" * 60)


if __name__ == "__main__":
    test_synthesis()
