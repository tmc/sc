"""
Query Strategies for Active Regex Learning

Implements various strategies for selecting which strings to query next.
The goal is to maximize information gain per query.

Strategies:
1. Random - baseline, query random strings
2. Uncertainty Sampling - query strings where model is most uncertain
3. Query-by-Committee - query strings where ensemble disagrees
4. Boundary Sampling - query strings near decision boundary
5. Expected Model Change - query strings that would most change model
"""

import mlx.core as mx
import mlx.nn as nn
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Set, Tuple, Optional, Dict, Any
import random
import math

from .oracle_interface import Oracle, generate_random_string


# =============================================================================
# Query Strategy Interface
# =============================================================================

class QueryStrategy(ABC):
    """Abstract base class for query selection strategies."""

    @abstractmethod
    def select_queries(
        self,
        candidate_pool: List[str],
        n_queries: int,
        model_state: Any,
    ) -> List[str]:
        """
        Select strings to query from candidate pool.

        Args:
            candidate_pool: Available strings to query
            n_queries: Number of queries to select
            model_state: Current learned model (for uncertainty estimation)

        Returns:
            List of selected strings to query
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Strategy name."""
        pass


# =============================================================================
# Random Strategy (Baseline)
# =============================================================================

class RandomStrategy(QueryStrategy):
    """Randomly select strings to query - baseline strategy."""

    def select_queries(
        self,
        candidate_pool: List[str],
        n_queries: int,
        model_state: Any,
    ) -> List[str]:
        n = min(n_queries, len(candidate_pool))
        return random.sample(candidate_pool, n)

    @property
    def name(self) -> str:
        return "Random"


# =============================================================================
# Uncertainty Sampling
# =============================================================================

class UncertaintySampling(QueryStrategy):
    """
    Query strings where the current model is most uncertain.

    Uses entropy of predicted probability as uncertainty measure:
    H(p) = -p*log(p) - (1-p)*log(1-p)

    High entropy = high uncertainty = good query candidate.
    """

    def __init__(self, temperature: float = 1.0):
        self.temperature = temperature

    def _compute_uncertainty(
        self,
        s: str,
        model_state: "StatechartHypothesis",
    ) -> float:
        """Compute uncertainty score for a string."""
        # Get model's predicted probability
        prob = model_state.predict_probability(s)

        # Compute entropy (uncertainty)
        if prob <= 0 or prob >= 1:
            return 0.0  # No uncertainty at extremes

        entropy = -prob * math.log(prob + 1e-10) - (1 - prob) * math.log(1 - prob + 1e-10)
        return entropy

    def select_queries(
        self,
        candidate_pool: List[str],
        n_queries: int,
        model_state: Any,
    ) -> List[str]:
        if model_state is None:
            # No model yet, fall back to random
            return random.sample(candidate_pool, min(n_queries, len(candidate_pool)))

        # Score all candidates by uncertainty
        scored = [(s, self._compute_uncertainty(s, model_state)) for s in candidate_pool]

        # Sort by uncertainty (highest first)
        scored.sort(key=lambda x: -x[1])

        # Return top-n
        return [s for s, _ in scored[:n_queries]]

    @property
    def name(self) -> str:
        return "UncertaintySampling"


# =============================================================================
# Boundary Sampling
# =============================================================================

class BoundarySampling(QueryStrategy):
    """
    Query strings near the decision boundary.

    For regex learning, the boundary is between accepting and rejecting.
    We look for strings where small changes flip the prediction.
    """

    def __init__(self, n_mutations: int = 3):
        self.n_mutations = n_mutations

    def _is_near_boundary(
        self,
        s: str,
        model_state: "StatechartHypothesis",
        alphabet: Set[str],
    ) -> float:
        """Check if string is near decision boundary via mutations."""
        base_pred = model_state.predict(s)
        flips = 0

        # Try mutations
        chars = list(alphabet)
        mutations_tried = 0

        for i in range(len(s)):
            for c in chars[:self.n_mutations]:
                if c != s[i]:
                    mutated = s[:i] + c + s[i+1:]
                    if model_state.predict(mutated) != base_pred:
                        flips += 1
                    mutations_tried += 1

        # Also try insertions and deletions
        for i in range(len(s) + 1):
            for c in chars[:2]:
                inserted = s[:i] + c + s[i:]
                if model_state.predict(inserted) != base_pred:
                    flips += 1
                mutations_tried += 1

        for i in range(len(s)):
            deleted = s[:i] + s[i+1:]
            if model_state.predict(deleted) != base_pred:
                flips += 1
            mutations_tried += 1

        # Score: fraction of mutations that flip prediction
        return flips / max(1, mutations_tried)

    def select_queries(
        self,
        candidate_pool: List[str],
        n_queries: int,
        model_state: Any,
    ) -> List[str]:
        if model_state is None:
            return random.sample(candidate_pool, min(n_queries, len(candidate_pool)))

        alphabet = model_state.alphabet if hasattr(model_state, 'alphabet') else set("ab01")

        # Score all candidates by boundary proximity
        scored = [(s, self._is_near_boundary(s, model_state, alphabet)) for s in candidate_pool]

        # Sort by boundary score (highest first)
        scored.sort(key=lambda x: -x[1])

        return [s for s, _ in scored[:n_queries]]

    @property
    def name(self) -> str:
        return "BoundarySampling"


# =============================================================================
# Query-by-Committee
# =============================================================================

class QueryByCommittee(QueryStrategy):
    """
    Query strings where an ensemble of models disagrees.

    Maintains a committee of hypotheses consistent with labeled examples.
    Queries strings with maximum disagreement (vote entropy).
    """

    def __init__(self, committee_size: int = 5):
        self.committee_size = committee_size

    def _compute_disagreement(
        self,
        s: str,
        committee: List["StatechartHypothesis"],
    ) -> float:
        """Compute disagreement among committee members."""
        votes = [h.predict(s) for h in committee]
        pos_votes = sum(votes)
        neg_votes = len(votes) - pos_votes

        # Vote entropy
        total = len(votes)
        if pos_votes == 0 or neg_votes == 0:
            return 0.0

        p_pos = pos_votes / total
        p_neg = neg_votes / total
        entropy = -p_pos * math.log(p_pos) - p_neg * math.log(p_neg)
        return entropy

    def select_queries(
        self,
        candidate_pool: List[str],
        n_queries: int,
        model_state: Any,
    ) -> List[str]:
        # model_state should be a committee (list of hypotheses)
        if model_state is None or not isinstance(model_state, list):
            return random.sample(candidate_pool, min(n_queries, len(candidate_pool)))

        committee = model_state

        # Score all candidates by disagreement
        scored = [(s, self._compute_disagreement(s, committee)) for s in candidate_pool]

        # Sort by disagreement (highest first)
        scored.sort(key=lambda x: -x[1])

        return [s for s, _ in scored[:n_queries]]

    @property
    def name(self) -> str:
        return "QueryByCommittee"


# =============================================================================
# Length-Stratified Sampling
# =============================================================================

class LengthStratifiedSampling(QueryStrategy):
    """
    Sample strings with diverse lengths.

    Ensures coverage across different string lengths to avoid
    bias toward short or long strings.
    """

    def __init__(self, uncertainty_weight: float = 0.5):
        self.uncertainty_weight = uncertainty_weight

    def select_queries(
        self,
        candidate_pool: List[str],
        n_queries: int,
        model_state: Any,
    ) -> List[str]:
        # Group by length
        by_length: Dict[int, List[str]] = {}
        for s in candidate_pool:
            length = len(s)
            if length not in by_length:
                by_length[length] = []
            by_length[length].append(s)

        # Distribute queries across lengths
        lengths = sorted(by_length.keys())
        queries_per_length = max(1, n_queries // len(lengths))

        selected = []
        for length in lengths:
            candidates = by_length[length]
            if model_state is not None and hasattr(model_state, 'predict_probability'):
                # Sort by uncertainty within length
                scored = [
                    (s, abs(0.5 - model_state.predict_probability(s)))
                    for s in candidates
                ]
                scored.sort(key=lambda x: x[1])  # Most uncertain first
                selected.extend([s for s, _ in scored[:queries_per_length]])
            else:
                selected.extend(random.sample(candidates, min(queries_per_length, len(candidates))))

            if len(selected) >= n_queries:
                break

        return selected[:n_queries]

    @property
    def name(self) -> str:
        return "LengthStratified"


# =============================================================================
# Diversity Sampling
# =============================================================================

class DiversitySampling(QueryStrategy):
    """
    Select diverse strings to maximize coverage.

    Uses edit distance to ensure selected strings are different from
    each other and from already-labeled examples.
    """

    def __init__(self, diversity_weight: float = 0.5):
        self.diversity_weight = diversity_weight

    def _edit_distance(self, s1: str, s2: str) -> int:
        """Compute Levenshtein edit distance."""
        m, n = len(s1), len(s2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if s1[i-1] == s2[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])

        return dp[m][n]

    def _min_distance_to_set(self, s: str, selected: List[str]) -> int:
        """Minimum edit distance to any string in set."""
        if not selected:
            return float('inf')
        return min(self._edit_distance(s, t) for t in selected)

    def select_queries(
        self,
        candidate_pool: List[str],
        n_queries: int,
        model_state: Any,
    ) -> List[str]:
        selected = []
        remaining = list(candidate_pool)

        # Greedy selection for diversity
        for _ in range(min(n_queries, len(candidate_pool))):
            if not remaining:
                break

            # Score by distance to already selected
            if selected:
                scored = [
                    (s, self._min_distance_to_set(s, selected))
                    for s in remaining
                ]
                scored.sort(key=lambda x: -x[1])  # Most distant first
                best = scored[0][0]
            else:
                best = random.choice(remaining)

            selected.append(best)
            remaining.remove(best)

        return selected

    @property
    def name(self) -> str:
        return "DiversitySampling"


# =============================================================================
# Hybrid Strategy
# =============================================================================

class HybridStrategy(QueryStrategy):
    """
    Combines uncertainty sampling with diversity.

    Balances exploitation (query uncertain strings) with
    exploration (query diverse strings).
    """

    def __init__(
        self,
        uncertainty_weight: float = 0.6,
        diversity_weight: float = 0.4,
    ):
        self.uncertainty_weight = uncertainty_weight
        self.diversity_weight = diversity_weight
        self._uncertainty = UncertaintySampling()
        self._diversity = DiversitySampling()

    def select_queries(
        self,
        candidate_pool: List[str],
        n_queries: int,
        model_state: Any,
    ) -> List[str]:
        # Get uncertainty-based candidates
        n_uncertainty = int(n_queries * self.uncertainty_weight)
        n_diversity = n_queries - n_uncertainty

        uncertainty_picks = self._uncertainty.select_queries(
            candidate_pool, n_uncertainty, model_state
        )

        # Get diversity-based candidates from remaining pool
        remaining = [s for s in candidate_pool if s not in uncertainty_picks]
        diversity_picks = self._diversity.select_queries(
            remaining, n_diversity, model_state
        )

        return uncertainty_picks + diversity_picks

    @property
    def name(self) -> str:
        return "Hybrid"


# =============================================================================
# Statechart Hypothesis (for uncertainty estimation)
# =============================================================================

@dataclass
class StatechartHypothesis:
    """
    A hypothesis about the target regex/statechart.

    Maintains positive and negative examples and can estimate
    probability that a new string matches.
    """
    positive_examples: Set[str] = field(default_factory=set)
    negative_examples: Set[str] = field(default_factory=set)
    alphabet: Set[str] = field(default_factory=lambda: set("ab"))

    # Simple n-gram features for prediction
    _ngram_pos_counts: Dict[str, int] = field(default_factory=dict)
    _ngram_neg_counts: Dict[str, int] = field(default_factory=dict)
    _n: int = 2  # n-gram size

    def add_example(self, s: str, is_positive: bool):
        """Add a labeled example."""
        if is_positive:
            self.positive_examples.add(s)
            self._update_ngrams(s, self._ngram_pos_counts)
        else:
            self.negative_examples.add(s)
            self._update_ngrams(s, self._ngram_neg_counts)

    def _update_ngrams(self, s: str, counts: Dict[str, int]):
        """Update n-gram counts for a string."""
        padded = "^" + s + "$"
        for i in range(len(padded) - self._n + 1):
            ngram = padded[i:i + self._n]
            counts[ngram] = counts.get(ngram, 0) + 1

    def _get_ngrams(self, s: str) -> List[str]:
        """Get n-grams from a string."""
        padded = "^" + s + "$"
        return [padded[i:i + self._n] for i in range(len(padded) - self._n + 1)]

    def predict_probability(self, s: str) -> float:
        """
        Estimate probability that string matches.

        Uses n-gram likelihood ratio with Laplace smoothing.
        """
        if s in self.positive_examples:
            return 1.0
        if s in self.negative_examples:
            return 0.0

        ngrams = self._get_ngrams(s)
        if not ngrams:
            return 0.5

        # Compute log-likelihood ratio
        total_pos = sum(self._ngram_pos_counts.values()) + 1
        total_neg = sum(self._ngram_neg_counts.values()) + 1

        log_ratio = 0.0
        for ng in ngrams:
            pos_count = self._ngram_pos_counts.get(ng, 0) + 1  # Laplace
            neg_count = self._ngram_neg_counts.get(ng, 0) + 1

            pos_prob = pos_count / total_pos
            neg_prob = neg_count / total_neg

            log_ratio += math.log(pos_prob / neg_prob + 1e-10)

        # Convert to probability via sigmoid
        prob = 1.0 / (1.0 + math.exp(-log_ratio / len(ngrams)))
        return prob

    def predict(self, s: str) -> bool:
        """Predict whether string matches."""
        return self.predict_probability(s) > 0.5


# =============================================================================
# Strategy Factory
# =============================================================================

STRATEGIES = {
    "random": RandomStrategy,
    "uncertainty": UncertaintySampling,
    "boundary": BoundarySampling,
    "committee": QueryByCommittee,
    "length_stratified": LengthStratifiedSampling,
    "diversity": DiversitySampling,
    "hybrid": HybridStrategy,
}


def get_strategy(name: str, **kwargs) -> QueryStrategy:
    """Get a query strategy by name."""
    if name not in STRATEGIES:
        raise ValueError(f"Unknown strategy: {name}. Available: {list(STRATEGIES.keys())}")
    return STRATEGIES[name](**kwargs)


def list_strategies() -> List[str]:
    """List available query strategies."""
    return list(STRATEGIES.keys())


# =============================================================================
# Demo
# =============================================================================

if __name__ == "__main__":
    print("Query Strategy Demo")
    print("=" * 50)

    # Create a hypothesis with some examples
    hypothesis = StatechartHypothesis(alphabet={"a", "b"})
    hypothesis.add_example("", True)
    hypothesis.add_example("ab", True)
    hypothesis.add_example("abab", True)
    hypothesis.add_example("a", False)
    hypothesis.add_example("b", False)
    hypothesis.add_example("aa", False)

    # Generate candidate pool
    candidates = [
        "", "a", "b", "ab", "ba", "aa", "bb",
        "aba", "abb", "bab", "abab", "abba", "baab"
    ]

    print(f"\nHypothesis: {len(hypothesis.positive_examples)} positive, "
          f"{len(hypothesis.negative_examples)} negative examples")

    # Test each strategy
    for name in ["random", "uncertainty", "diversity", "hybrid"]:
        strategy = get_strategy(name)
        selected = strategy.select_queries(candidates, 3, hypothesis)
        print(f"\n{strategy.name}: {selected}")

    # Show probabilities
    print("\nProbabilities:")
    for s in candidates:
        prob = hypothesis.predict_probability(s)
        pred = hypothesis.predict(s)
        print(f"  '{s}': prob={prob:.3f}, pred={pred}")
