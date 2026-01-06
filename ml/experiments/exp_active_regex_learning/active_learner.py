"""
Active Learner for Regex/Statechart Synthesis

Main active learning loop that:
1. Maintains a hypothesis (current best statechart)
2. Uses query strategy to select informative strings
3. Queries oracle for labels
4. Updates hypothesis based on new examples
5. Repeats until convergence or budget exhausted

The goal is to minimize queries while achieving high accuracy.
"""

import mlx.core as mx
from dataclasses import dataclass, field
from typing import List, Set, Tuple, Optional, Dict, Any, Callable
import random
import time

from .oracle_interface import Oracle, generate_string_pool, generate_systematic_strings
from .query_strategy import (
    QueryStrategy,
    RandomStrategy,
    UncertaintySampling,
    HybridStrategy,
    StatechartHypothesis,
    get_strategy,
)


# =============================================================================
# Learning State
# =============================================================================

@dataclass
class LearningState:
    """Tracks the state of the active learning process."""
    # Labeled examples
    positive_examples: Set[str] = field(default_factory=set)
    negative_examples: Set[str] = field(default_factory=set)

    # Query history
    queries_made: List[str] = field(default_factory=list)
    query_results: List[bool] = field(default_factory=list)

    # Performance metrics over time
    accuracy_history: List[float] = field(default_factory=list)
    precision_history: List[float] = field(default_factory=list)
    recall_history: List[float] = field(default_factory=list)

    # Timing
    start_time: float = 0.0
    query_times: List[float] = field(default_factory=list)

    @property
    def n_queries(self) -> int:
        return len(self.queries_made)

    @property
    def n_positive(self) -> int:
        return len(self.positive_examples)

    @property
    def n_negative(self) -> int:
        return len(self.negative_examples)

    def add_labeled_example(self, s: str, is_positive: bool):
        """Add a newly labeled example."""
        self.queries_made.append(s)
        self.query_results.append(is_positive)
        if is_positive:
            self.positive_examples.add(s)
        else:
            self.negative_examples.add(s)

    def get_hypothesis(self) -> StatechartHypothesis:
        """Create hypothesis from current examples."""
        h = StatechartHypothesis()
        for s in self.positive_examples:
            h.add_example(s, True)
        for s in self.negative_examples:
            h.add_example(s, False)
        return h


# =============================================================================
# DFA Hypothesis Learner
# =============================================================================

class DFAHypothesis:
    """
    DFA-based hypothesis that learns from examples.

    Uses a simple state-merging approach to infer DFA structure.
    """

    def __init__(self, alphabet: Set[str]):
        self.alphabet = alphabet
        self.positive: Set[str] = set()
        self.negative: Set[str] = set()

        # Prefix tree for positive examples
        self._prefix_tree: Dict[str, Dict[str, str]] = {}
        self._accepting: Set[str] = set()

    def add_example(self, s: str, is_positive: bool):
        """Add a labeled example."""
        if is_positive:
            self.positive.add(s)
            self._add_to_prefix_tree(s)
        else:
            self.negative.add(s)

    def _add_to_prefix_tree(self, s: str):
        """Add string to prefix tree."""
        state = ""
        for i, c in enumerate(s):
            if state not in self._prefix_tree:
                self._prefix_tree[state] = {}
            next_state = s[:i+1]
            self._prefix_tree[state][c] = next_state
            state = next_state
        self._accepting.add(s)

    def predict(self, s: str) -> bool:
        """Predict if string matches."""
        # Check exact matches first
        if s in self.positive:
            return True
        if s in self.negative:
            return False

        # Check if string follows prefix tree pattern
        state = ""
        for c in s:
            if state not in self._prefix_tree:
                return False
            transitions = self._prefix_tree[state]
            if c not in transitions:
                # Check for wildcard pattern
                return self._predict_by_similarity(s)
            state = transitions[c]

        return state in self._accepting

    def _predict_by_similarity(self, s: str) -> bool:
        """Predict by similarity to positive examples."""
        if not self.positive:
            return False

        # Simple heuristic: similar length and character distribution
        avg_pos_len = sum(len(p) for p in self.positive) / len(self.positive)
        if abs(len(s) - avg_pos_len) > 3:
            return False

        # Check character overlap with positive examples
        s_chars = set(s)
        pos_chars = set()
        for p in self.positive:
            pos_chars.update(p)

        if s_chars and not s_chars.issubset(pos_chars | self.alphabet):
            return False

        return True

    def predict_probability(self, s: str) -> float:
        """Estimate match probability."""
        if s in self.positive:
            return 1.0
        if s in self.negative:
            return 0.0

        # Heuristic probability based on similarity
        if not self.positive:
            return 0.5

        # Length similarity
        avg_len = sum(len(p) for p in self.positive) / len(self.positive)
        len_diff = abs(len(s) - avg_len)
        len_score = max(0, 1 - len_diff / 5)

        # Prefix match score
        prefix_score = 0
        for p in self.positive:
            common = 0
            for i in range(min(len(s), len(p))):
                if s[i] == p[i]:
                    common += 1
                else:
                    break
            prefix_score = max(prefix_score, common / max(len(s), len(p), 1))

        return 0.3 * len_score + 0.7 * prefix_score


# =============================================================================
# Active Learner
# =============================================================================

class ActiveLearner:
    """
    Active learning system for regex/statechart synthesis.

    Iteratively queries oracle and updates hypothesis to
    minimize total queries while maximizing accuracy.
    """

    def __init__(
        self,
        oracle: Oracle,
        strategy: QueryStrategy,
        max_queries: int = 100,
        batch_size: int = 5,
        convergence_threshold: float = 0.99,
        patience: int = 10,
    ):
        """
        Args:
            oracle: Ground truth oracle to query
            strategy: Query selection strategy
            max_queries: Maximum queries before stopping
            batch_size: Queries per iteration
            convergence_threshold: Stop if accuracy exceeds this
            patience: Stop if no improvement for this many batches
        """
        self.oracle = oracle
        self.strategy = strategy
        self.max_queries = max_queries
        self.batch_size = batch_size
        self.convergence_threshold = convergence_threshold
        self.patience = patience

        self.state = LearningState()
        self.hypothesis = DFAHypothesis(oracle.get_alphabet())

        # Candidate pool for querying
        self._candidate_pool: List[str] = []
        self._queried: Set[str] = set()

    def initialize_candidate_pool(
        self,
        pool_size: int = 1000,
        max_length: int = 10,
        systematic: bool = False,
    ):
        """Initialize the pool of candidate strings."""
        alphabet = self.oracle.get_alphabet()

        if systematic and len(alphabet) <= 3:
            # For small alphabets, generate all strings up to max_length
            self._candidate_pool = generate_systematic_strings(alphabet, min(max_length, 6))
        else:
            # Random sampling
            self._candidate_pool = [
                self._random_string(alphabet, max_length)
                for _ in range(pool_size)
            ]

        # Add empty string and single chars
        self._candidate_pool.append("")
        for c in alphabet:
            if c not in self._candidate_pool:
                self._candidate_pool.append(c)

    def _random_string(self, alphabet: Set[str], max_length: int) -> str:
        """Generate random string from alphabet."""
        length = random.randint(0, max_length)
        chars = list(alphabet)
        return "".join(random.choice(chars) for _ in range(length))

    def _get_available_candidates(self) -> List[str]:
        """Get candidates not yet queried."""
        return [s for s in self._candidate_pool if s not in self._queried]

    def _query_batch(self, strings: List[str]) -> List[bool]:
        """Query oracle for a batch of strings."""
        results = []
        for s in strings:
            label = self.oracle.query(s)
            results.append(label)
            self.state.add_labeled_example(s, label)
            self.hypothesis.add_example(s, label)
            self._queried.add(s)
        return results

    def _evaluate_on_holdout(self, holdout: List[str]) -> Tuple[float, float, float]:
        """Evaluate current hypothesis on holdout set."""
        if not holdout:
            return 0.0, 0.0, 0.0

        tp, fp, tn, fn = 0, 0, 0, 0
        for s in holdout:
            pred = self.hypothesis.predict(s)
            actual = self.oracle.query(s)

            if pred and actual:
                tp += 1
            elif pred and not actual:
                fp += 1
            elif not pred and actual:
                fn += 1
            else:
                tn += 1

        accuracy = (tp + tn) / len(holdout) if holdout else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0

        return accuracy, precision, recall

    def learn(
        self,
        initial_examples: int = 5,
        holdout_size: int = 100,
        verbose: bool = True,
    ) -> LearningState:
        """
        Run active learning loop.

        Args:
            initial_examples: Random examples to start with
            holdout_size: Size of holdout set for evaluation
            verbose: Print progress

        Returns:
            Final learning state with metrics
        """
        self.state.start_time = time.time()

        # Initialize candidate pool if not done
        if not self._candidate_pool:
            self.initialize_candidate_pool()

        # Create holdout set for evaluation
        available = self._get_available_candidates()
        holdout = random.sample(available, min(holdout_size, len(available) // 2))
        for s in holdout:
            self._queried.add(s)  # Don't query holdout strings

        # Initial random queries to bootstrap
        available = self._get_available_candidates()
        initial = random.sample(available, min(initial_examples, len(available)))
        self._query_batch(initial)

        if verbose:
            print(f"Initial: {initial_examples} examples")

        # Active learning loop
        best_accuracy = 0.0
        no_improvement = 0

        while self.state.n_queries < self.max_queries:
            # Get available candidates
            available = self._get_available_candidates()
            if not available:
                if verbose:
                    print("Exhausted candidate pool")
                break

            # Select queries using strategy
            hypothesis_state = self.state.get_hypothesis()
            queries = self.strategy.select_queries(
                available,
                self.batch_size,
                hypothesis_state,
            )

            if not queries:
                break

            # Query oracle
            self._query_batch(queries)

            # Evaluate on holdout
            accuracy, precision, recall = self._evaluate_on_holdout(holdout)
            self.state.accuracy_history.append(accuracy)
            self.state.precision_history.append(precision)
            self.state.recall_history.append(recall)

            if verbose and self.state.n_queries % 10 == 0:
                print(f"  Queries: {self.state.n_queries}, "
                      f"Accuracy: {accuracy:.3f}, "
                      f"+:{self.state.n_positive} -:{self.state.n_negative}")

            # Check convergence
            if accuracy >= self.convergence_threshold:
                if verbose:
                    print(f"Converged at {self.state.n_queries} queries")
                break

            # Check patience
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                no_improvement = 0
            else:
                no_improvement += 1
                if no_improvement >= self.patience:
                    if verbose:
                        print(f"Stopped early (no improvement for {self.patience} batches)")
                    break

        return self.state

    def get_final_accuracy(self, test_size: int = 200) -> float:
        """Evaluate final accuracy on fresh test set."""
        alphabet = self.oracle.get_alphabet()
        test_strings = [self._random_string(alphabet, 10) for _ in range(test_size)]

        correct = 0
        for s in test_strings:
            pred = self.hypothesis.predict(s)
            actual = self.oracle.query(s)
            if pred == actual:
                correct += 1

        return correct / test_size


# =============================================================================
# Convenience Functions
# =============================================================================

def run_active_learning(
    oracle: Oracle,
    strategy_name: str = "uncertainty",
    max_queries: int = 100,
    batch_size: int = 5,
    verbose: bool = True,
) -> Tuple[LearningState, float]:
    """
    Run active learning with specified strategy.

    Returns:
        (learning_state, final_accuracy)
    """
    strategy = get_strategy(strategy_name)
    learner = ActiveLearner(
        oracle=oracle,
        strategy=strategy,
        max_queries=max_queries,
        batch_size=batch_size,
    )
    learner.initialize_candidate_pool()
    state = learner.learn(verbose=verbose)
    accuracy = learner.get_final_accuracy()
    return state, accuracy


def compare_strategies(
    oracle: Oracle,
    strategies: List[str],
    max_queries: int = 100,
    n_runs: int = 3,
    verbose: bool = True,
) -> Dict[str, Dict[str, float]]:
    """
    Compare multiple strategies on the same oracle.

    Returns:
        Dict mapping strategy name to metrics
    """
    results = {}

    for strat_name in strategies:
        if verbose:
            print(f"\n=== Strategy: {strat_name} ===")

        accuracies = []
        queries_to_90 = []

        for run in range(n_runs):
            strategy = get_strategy(strat_name)
            learner = ActiveLearner(
                oracle=oracle,
                strategy=strategy,
                max_queries=max_queries,
                batch_size=5,
            )
            learner.initialize_candidate_pool()
            state = learner.learn(verbose=False)
            accuracy = learner.get_final_accuracy()
            accuracies.append(accuracy)

            # Find queries needed for 90% accuracy
            for i, acc in enumerate(state.accuracy_history):
                if acc >= 0.9:
                    queries_to_90.append((i + 1) * 5)  # batch_size = 5
                    break
            else:
                queries_to_90.append(max_queries)

            if verbose:
                print(f"  Run {run + 1}: accuracy={accuracy:.3f}, "
                      f"queries={state.n_queries}")

        results[strat_name] = {
            "mean_accuracy": sum(accuracies) / len(accuracies),
            "mean_queries_to_90": sum(queries_to_90) / len(queries_to_90),
            "accuracies": accuracies,
        }

    return results


# =============================================================================
# Demo
# =============================================================================

if __name__ == "__main__":
    from .oracle_interface import create_ab_star_oracle, create_even_zeros_oracle

    print("Active Learner Demo")
    print("=" * 50)

    # Test on (ab)* pattern
    oracle = create_ab_star_oracle()
    print(f"\nOracle: {oracle.name}")

    # Run with uncertainty sampling
    state, accuracy = run_active_learning(
        oracle,
        strategy_name="uncertainty",
        max_queries=50,
        verbose=True,
    )

    print(f"\nFinal: {state.n_queries} queries, accuracy={accuracy:.3f}")
    print(f"Positive examples: {state.n_positive}")
    print(f"Negative examples: {state.n_negative}")
