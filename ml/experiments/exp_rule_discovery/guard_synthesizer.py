"""
Guard Synthesizer - Learns symbolic guards from transition examples.

Given examples of valid/invalid transitions, synthesizes interpretable
guard conditions that distinguish them.

Uses decision tree-like splitting to find symbolic predicates.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Callable, Set
from collections import defaultdict
from enum import Enum, auto

try:
    from .trace_collector import Transition, Outcome
except ImportError:
    from trace_collector import Transition, Outcome


class Comparator(Enum):
    """Comparison operators for guard conditions."""
    EQ = "=="
    NE = "!="
    LT = "<"
    LE = "<="
    GT = ">"
    GE = ">="


@dataclass
class Predicate:
    """A single predicate in a guard condition."""
    feature: str
    comparator: Comparator
    threshold: float

    def evaluate(self, features: Dict[str, float]) -> bool:
        """Evaluate predicate on feature dict."""
        value = features.get(self.feature, 0.0)
        if self.comparator == Comparator.EQ:
            return abs(value - self.threshold) < 0.001
        elif self.comparator == Comparator.NE:
            return abs(value - self.threshold) >= 0.001
        elif self.comparator == Comparator.LT:
            return value < self.threshold
        elif self.comparator == Comparator.LE:
            return value <= self.threshold
        elif self.comparator == Comparator.GT:
            return value > self.threshold
        elif self.comparator == Comparator.GE:
            return value >= self.threshold
        return False

    def __str__(self) -> str:
        return f"{self.feature} {self.comparator.value} {self.threshold}"


@dataclass
class SymbolicGuard:
    """
    A symbolic guard condition (conjunction of predicates).

    Represents: pred1 AND pred2 AND pred3 ...
    """
    predicates: List[Predicate] = field(default_factory=list)
    coverage: float = 0.0  # Fraction of positive examples covered
    precision: float = 0.0  # Fraction of covered examples that are positive
    description: str = ""

    def evaluate(self, features: Dict[str, float]) -> bool:
        """Evaluate guard on feature dict."""
        return all(p.evaluate(features) for p in self.predicates)

    def __str__(self) -> str:
        if not self.predicates:
            return "TRUE"
        return " AND ".join(str(p) for p in self.predicates)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'expression': str(self),
            'predicates': [
                {'feature': p.feature, 'op': p.comparator.value, 'threshold': p.threshold}
                for p in self.predicates
            ],
            'coverage': self.coverage,
            'precision': self.precision,
        }


class GuardSynthesizer:
    """
    Synthesizes symbolic guards from positive/negative examples.

    Approach:
    1. Find features that discriminate positive from negative
    2. Find threshold values that maximize separation
    3. Combine predicates into conjunctive guards
    4. Simplify and name the guards
    """

    def __init__(self, min_coverage: float = 0.5, min_precision: float = 0.8):
        """
        Args:
            min_coverage: Minimum fraction of positives a guard should cover
            min_precision: Minimum precision (true positives / all positives)
        """
        self.min_coverage = min_coverage
        self.min_precision = min_precision
        self.guards: List[SymbolicGuard] = []

    def _compute_feature_values(self, transitions: List[Transition],
                                feature: str) -> List[float]:
        """Get all values of a feature across transitions."""
        values = []
        for t in transitions:
            if feature in t.features:
                values.append(t.features[feature])
        return values

    def _find_best_split(self, positives: List[Transition],
                         negatives: List[Transition],
                         feature: str) -> Tuple[Optional[Predicate], float]:
        """
        Find best threshold split for a feature.

        Returns (predicate, information_gain) or (None, 0) if no good split.
        """
        # Get all values
        all_trans = positives + negatives
        values = set()
        for t in all_trans:
            if feature in t.features:
                values.add(t.features[feature])

        if len(values) < 2:
            return None, 0.0

        sorted_values = sorted(values)
        best_pred = None
        best_score = 0.0

        # Try threshold at each midpoint
        for i in range(len(sorted_values) - 1):
            threshold = (sorted_values[i] + sorted_values[i + 1]) / 2

            # Try both <= and >
            for comparator in [Comparator.LE, Comparator.GT]:
                pred = Predicate(feature, comparator, threshold)

                # Count how many positives/negatives pass this predicate
                pos_pass = sum(1 for t in positives
                               if pred.evaluate(t.features))
                neg_pass = sum(1 for t in negatives
                               if pred.evaluate(t.features))

                # Compute precision and recall
                total_pass = pos_pass + neg_pass
                if total_pass == 0:
                    continue

                precision = pos_pass / total_pass
                recall = pos_pass / max(1, len(positives))

                # Score combines precision and recall
                if precision >= self.min_precision:
                    score = precision * recall  # F1-like
                    if score > best_score:
                        best_score = score
                        best_pred = pred

        return best_pred, best_score

    def synthesize_guard(self, positives: List[Transition],
                         negatives: List[Transition],
                         max_predicates: int = 3) -> SymbolicGuard:
        """
        Synthesize a guard that accepts positives and rejects negatives.

        Uses greedy predicate addition.
        """
        # Get all features
        all_features = set()
        for t in positives + negatives:
            all_features.update(t.features.keys())

        guard = SymbolicGuard()
        remaining_positives = positives.copy()
        remaining_negatives = negatives.copy()

        for _ in range(max_predicates):
            if not remaining_negatives:
                break  # Perfect separation achieved

            # Find best predicate to add
            best_pred = None
            best_score = 0.0

            for feature in all_features:
                pred, score = self._find_best_split(
                    remaining_positives, remaining_negatives, feature
                )
                if pred and score > best_score:
                    best_score = score
                    best_pred = pred

            if best_pred is None:
                break  # No improving predicate found

            guard.predicates.append(best_pred)

            # Filter remaining examples
            remaining_positives = [t for t in remaining_positives
                                   if best_pred.evaluate(t.features)]
            remaining_negatives = [t for t in remaining_negatives
                                   if best_pred.evaluate(t.features)]

        # Compute final coverage and precision
        true_positives = sum(1 for t in positives if guard.evaluate(t.features))
        false_positives = sum(1 for t in negatives if guard.evaluate(t.features))

        guard.coverage = true_positives / max(1, len(positives))
        guard.precision = true_positives / max(1, true_positives + false_positives)

        return guard

    def synthesize_legality_guard(self, transitions: List[Transition]) -> SymbolicGuard:
        """
        Synthesize a guard for legal moves.

        Positives = valid moves, Negatives = invalid moves
        """
        positives = [t for t in transitions if t.outcome != Outcome.INVALID]
        negatives = [t for t in transitions if t.outcome == Outcome.INVALID]

        if not negatives:
            # All moves legal - return trivial guard
            return SymbolicGuard(description="All moves legal")

        guard = self.synthesize_guard(positives, negatives)
        guard.description = "Legality guard: move is valid when"
        return guard

    def synthesize_winning_guard(self, transitions: List[Transition]) -> SymbolicGuard:
        """
        Synthesize a guard that identifies winning conditions.

        Positives = winning moves, Negatives = non-winning moves
        """
        positives = [t for t in transitions if t.outcome == Outcome.WIN]
        negatives = [t for t in transitions if t.outcome != Outcome.WIN]

        if not positives:
            return SymbolicGuard(description="No winning patterns found")

        guard = self.synthesize_guard(positives, negatives)
        guard.description = "Win condition"
        return guard

    def synthesize_action_guards(self, transitions: List[Transition]
                                 ) -> Dict[Any, SymbolicGuard]:
        """
        Synthesize guards for each distinct action.

        For each action, find conditions under which it was chosen.
        """
        action_guards = {}

        # Group transitions by action
        by_action = defaultdict(list)
        for t in transitions:
            by_action[str(t.action)].append(t)

        for action, action_trans in by_action.items():
            # Positives = this action, Negatives = other actions
            positives = action_trans
            negatives = [t for t in transitions if str(t.action) != action]

            if len(positives) < 5:  # Skip rare actions
                continue

            guard = self.synthesize_guard(positives, negatives)
            guard.description = f"Guard for action {action}"
            action_guards[action] = guard

        return action_guards

    def synthesize_transition_guards(
            self,
            transitions: List[Transition],
            from_state_classifier: Callable[[Dict], str],
            to_state_classifier: Callable[[Dict], str],
    ) -> Dict[Tuple[str, str], SymbolicGuard]:
        """
        Synthesize guards for state-to-state transitions.

        Given classifiers that map states to abstract state names,
        finds guards for when transitions between states occur.
        """
        transition_guards = {}

        # Classify all transitions
        transition_types = defaultdict(list)
        for t in transitions:
            from_state = from_state_classifier(t.state)
            to_state = to_state_classifier(t.next_state)
            transition_types[(from_state, to_state)].append(t)

        for (from_s, to_s), trans_list in transition_types.items():
            if len(trans_list) < 3:
                continue

            # Positives = transitions from_s -> to_s
            # Negatives = transitions from from_s to anywhere else
            positives = trans_list
            negatives = [t for t in transitions
                         if from_state_classifier(t.state) == from_s
                         and to_state_classifier(t.next_state) != to_s]

            if not negatives:
                # All transitions from from_s go to to_s
                guard = SymbolicGuard(description=f"Always: {from_s} -> {to_s}")
            else:
                guard = self.synthesize_guard(positives, negatives)
                guard.description = f"Guard: {from_s} -> {to_s}"

            transition_guards[(from_s, to_s)] = guard

        return transition_guards

    def explain_guards(self) -> str:
        """Generate human-readable explanation of synthesized guards."""
        lines = ["=== Synthesized Guards ===\n"]

        for guard in self.guards:
            lines.append(f"Guard: {guard.description}")
            lines.append(f"  Expression: {guard}")
            lines.append(f"  Coverage: {guard.coverage*100:.1f}%")
            lines.append(f"  Precision: {guard.precision*100:.1f}%")
            lines.append("")

        return '\n'.join(lines)
