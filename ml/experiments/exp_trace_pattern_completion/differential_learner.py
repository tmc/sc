"""
Differential Learner: Multiple approaches for trace completion.

Compares different learning strategies:
1. Unigram (baseline): Single transition map
2. N-gram: Context-aware with N previous states
3. Probabilistic: Handles noisy patterns with probability
4. Stack-aware: Tracks push/pop operations
5. Mode-aware: Tracks conditional flags/modes
6. Combined: Ensemble of above approaches
"""

import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple, Optional, Any
import re


@dataclass
class LearnerResult:
    """Result from a learner."""
    predicted: List[str] = field(default_factory=list)
    confidence: float = 0.0
    method: str = ""
    debug_info: Dict[str, Any] = field(default_factory=dict)


class UnigramLearner:
    """Baseline: Simple transition map (what we had before)."""

    def __init__(self):
        self.transitions: Dict[str, Counter] = defaultdict(Counter)

    def learn(self, traces: List[List[str]]):
        """Learn from example traces."""
        self.transitions.clear()
        for trace in traces:
            for i in range(len(trace) - 1):
                self.transitions[trace[i]][trace[i + 1]] += 1

    def predict(self, partial: List[str], n: int = 1) -> LearnerResult:
        """Predict next n events."""
        result = LearnerResult(method="unigram")
        if not partial:
            return result

        current = partial[-1]
        predictions = []

        for _ in range(n):
            if current in self.transitions and self.transitions[current]:
                next_event = self.transitions[current].most_common(1)[0][0]
                predictions.append(next_event)
                current = next_event
            else:
                break

        result.predicted = predictions
        result.confidence = 0.8 if predictions else 0.0
        return result


class NGramLearner:
    """N-gram context: Looks at last N states."""

    def __init__(self, n: int = 2):
        self.n = n
        self.transitions: Dict[Tuple[str, ...], Counter] = defaultdict(Counter)
        self.fallback = UnigramLearner()

    def learn(self, traces: List[List[str]]):
        """Learn from example traces."""
        self.transitions.clear()
        self.fallback.learn(traces)

        for trace in traces:
            for i in range(len(trace) - 1):
                # Build context of size n
                start = max(0, i - self.n + 1)
                context = tuple(trace[start:i + 1])
                next_event = trace[i + 1]
                self.transitions[context][next_event] += 1

    def predict(self, partial: List[str], n: int = 1) -> LearnerResult:
        """Predict next n events using n-gram context."""
        result = LearnerResult(method=f"{self.n}-gram")
        if not partial:
            return result

        predictions = []
        current_trace = list(partial)

        for _ in range(n):
            # Try full context, then back off
            predicted = None
            for ctx_len in range(self.n, 0, -1):
                start = max(0, len(current_trace) - ctx_len)
                context = tuple(current_trace[start:])
                if context in self.transitions and self.transitions[context]:
                    predicted = self.transitions[context].most_common(1)[0][0]
                    result.debug_info["context_used"] = context
                    break

            if predicted is None:
                # Fall back to unigram
                fallback_result = self.fallback.predict(current_trace, 1)
                if fallback_result.predicted:
                    predicted = fallback_result.predicted[0]

            if predicted:
                predictions.append(predicted)
                current_trace.append(predicted)
            else:
                break

        result.predicted = predictions
        result.confidence = 0.9 if predictions else 0.0
        return result


class ProbabilisticLearner:
    """Handles noisy patterns with probability distributions."""

    def __init__(self, threshold: float = 0.6):
        self.threshold = threshold  # Minimum probability to predict
        self.transitions: Dict[str, Counter] = defaultdict(Counter)
        self.totals: Dict[str, int] = defaultdict(int)

    def learn(self, traces: List[List[str]]):
        """Learn transition probabilities."""
        self.transitions.clear()
        self.totals.clear()

        for trace in traces:
            for i in range(len(trace) - 1):
                src = trace[i]
                tgt = trace[i + 1]
                self.transitions[src][tgt] += 1
                self.totals[src] += 1

    def get_probability(self, src: str, tgt: str) -> float:
        """Get P(tgt | src)."""
        if src not in self.totals or self.totals[src] == 0:
            return 0.0
        return self.transitions[src][tgt] / self.totals[src]

    def predict(self, partial: List[str], n: int = 1) -> LearnerResult:
        """Predict using majority probability."""
        result = LearnerResult(method="probabilistic")
        if not partial:
            return result

        predictions = []
        current = partial[-1]

        for _ in range(n):
            if current in self.transitions and self.transitions[current]:
                best, count = self.transitions[current].most_common(1)[0]
                prob = count / self.totals[current]
                if prob >= self.threshold:
                    predictions.append(best)
                    result.debug_info[f"prob_{best}"] = prob
                    current = best
                else:
                    # Too uncertain
                    break
            else:
                break

        result.predicted = predictions
        result.confidence = min(
            self.get_probability(partial[-1], predictions[0]) if predictions else 0.0,
            1.0
        )
        return result


class StackAwareLearner:
    """Handles push/pop patterns with a simulated stack."""

    def __init__(self):
        self.push_pattern = re.compile(r"PUSH_(\w+)")
        self.pop_pattern = re.compile(r"POP_(\w+)")
        self.fallback = UnigramLearner()

    def learn(self, traces: List[List[str]]):
        """Learn from traces (also learn fallback)."""
        self.fallback.learn(traces)

    def predict(self, partial: List[str], n: int = 1) -> LearnerResult:
        """Predict using stack simulation."""
        result = LearnerResult(method="stack-aware")

        # Simulate stack from partial trace
        stack = []
        for event in partial:
            push_match = self.push_pattern.match(event)
            pop_match = self.pop_pattern.match(event)

            if push_match:
                stack.append(push_match.group(1))
            elif pop_match and stack:
                stack.pop()

        result.debug_info["stack"] = list(stack)

        # Predict: if stack non-empty, likely need to pop
        predictions = []
        for _ in range(n):
            if stack:
                # Predict POP for top of stack
                next_pop = f"POP_{stack[-1]}"
                predictions.append(next_pop)
                stack.pop()
            else:
                # Fall back to unigram
                fallback_result = self.fallback.predict(
                    partial + predictions if predictions else partial, 1
                )
                if fallback_result.predicted:
                    predictions.append(fallback_result.predicted[0])
                else:
                    break

        result.predicted = predictions
        result.confidence = 0.95 if predictions and result.debug_info["stack"] else 0.5
        return result


class ModeAwareLearner:
    """Handles conditional patterns with mode tracking."""

    def __init__(self):
        self.mode_patterns = ["MODE_", "SET_", "FLAG_"]
        self.mode_transitions: Dict[Tuple[str, str], Counter] = defaultdict(Counter)
        self.fallback = UnigramLearner()

    def learn(self, traces: List[List[str]]):
        """Learn mode-aware transitions."""
        self.mode_transitions.clear()
        self.fallback.learn(traces)

        for trace in traces:
            current_mode = "DEFAULT"
            for i in range(len(trace) - 1):
                event = trace[i]
                next_event = trace[i + 1]

                # Check for mode changes
                for pattern in self.mode_patterns:
                    if event.startswith(pattern):
                        current_mode = event
                        break

                # Record transition with mode context
                key = (current_mode, event)
                self.mode_transitions[key][next_event] += 1

    def predict(self, partial: List[str], n: int = 1) -> LearnerResult:
        """Predict using mode context."""
        result = LearnerResult(method="mode-aware")

        # Determine current mode
        current_mode = "DEFAULT"
        for event in partial:
            for pattern in self.mode_patterns:
                if event.startswith(pattern):
                    current_mode = event
                    break

        result.debug_info["mode"] = current_mode

        predictions = []
        current_trace = list(partial)

        for _ in range(n):
            if not current_trace:
                break

            last_event = current_trace[-1]
            key = (current_mode, last_event)

            if key in self.mode_transitions and self.mode_transitions[key]:
                next_event = self.mode_transitions[key].most_common(1)[0][0]
                predictions.append(next_event)
                current_trace.append(next_event)

                # Update mode if needed
                for pattern in self.mode_patterns:
                    if next_event.startswith(pattern):
                        current_mode = next_event
                        break
            else:
                # Fall back
                fallback_result = self.fallback.predict(current_trace, 1)
                if fallback_result.predicted:
                    predictions.append(fallback_result.predicted[0])
                    current_trace.append(fallback_result.predicted[0])
                else:
                    break

        result.predicted = predictions
        result.confidence = 0.9 if predictions else 0.0
        return result


class CounterAwareLearner:
    """Handles counter-based patterns."""

    def __init__(self):
        self.repeat_counts: Dict[str, List[int]] = defaultdict(list)
        self.fallback = UnigramLearner()

    def learn(self, traces: List[List[str]]):
        """Learn repeat patterns."""
        self.repeat_counts.clear()
        self.fallback.learn(traces)

        for trace in traces:
            current = None
            count = 0
            for event in trace:
                if event == current:
                    count += 1
                else:
                    if current is not None and count > 1:
                        self.repeat_counts[current].append(count)
                    current = event
                    count = 1

    def predict(self, partial: List[str], n: int = 1) -> LearnerResult:
        """Predict using counter awareness."""
        result = LearnerResult(method="counter-aware")

        # Count trailing repeats
        if not partial:
            return result

        last = partial[-1]
        repeat_count = 0
        for event in reversed(partial):
            if event == last:
                repeat_count += 1
            else:
                break

        result.debug_info["repeat_count"] = repeat_count
        result.debug_info["known_counts"] = self.repeat_counts.get(last, [])

        # Check if we're at a known repeat boundary
        predictions = []
        if last in self.repeat_counts:
            expected_counts = self.repeat_counts[last]
            if expected_counts:
                typical_count = max(set(expected_counts), key=expected_counts.count)
                result.debug_info["typical_count"] = typical_count

                if repeat_count < typical_count:
                    # Continue repeating
                    for _ in range(min(n, typical_count - repeat_count)):
                        predictions.append(last)
                else:
                    # Done repeating, use fallback for what comes next
                    fallback_result = self.fallback.predict(partial, n)
                    predictions = fallback_result.predicted

        if not predictions:
            fallback_result = self.fallback.predict(partial, n)
            predictions = fallback_result.predicted

        result.predicted = predictions
        result.confidence = 0.8 if predictions else 0.0
        return result


class EnsembleLearner:
    """Combines multiple learners with voting."""

    def __init__(self):
        self.learners = {
            "unigram": UnigramLearner(),
            "2gram": NGramLearner(n=2),
            "3gram": NGramLearner(n=3),
            "probabilistic": ProbabilisticLearner(),
            "stack": StackAwareLearner(),
            "mode": ModeAwareLearner(),
            "counter": CounterAwareLearner(),
        }

    def learn(self, traces: List[List[str]]):
        """Train all learners."""
        for learner in self.learners.values():
            learner.learn(traces)

    def predict(self, partial: List[str], n: int = 1) -> LearnerResult:
        """Predict using ensemble voting."""
        result = LearnerResult(method="ensemble")

        # Get predictions from all learners
        all_predictions: Dict[str, List[LearnerResult]] = {}
        for name, learner in self.learners.items():
            pred_result = learner.predict(partial, n)
            all_predictions[name] = pred_result

        # Vote on first prediction
        votes: Counter = Counter()
        for name, pred_result in all_predictions.items():
            if pred_result.predicted:
                first_pred = pred_result.predicted[0]
                # Weight by confidence
                votes[first_pred] += pred_result.confidence

        result.debug_info["votes"] = dict(votes)
        result.debug_info["learner_predictions"] = {
            name: r.predicted for name, r in all_predictions.items()
        }

        if not votes:
            return result

        # Use highest voted prediction, then continue from best learner
        best_first = votes.most_common(1)[0][0]

        # Find learner that predicted this
        best_learner = None
        best_conf = 0.0
        for name, pred_result in all_predictions.items():
            if pred_result.predicted and pred_result.predicted[0] == best_first:
                if pred_result.confidence > best_conf:
                    best_conf = pred_result.confidence
                    best_learner = name

        result.debug_info["winning_learner"] = best_learner

        # Use that learner's full prediction
        if best_learner:
            result.predicted = all_predictions[best_learner].predicted
        else:
            result.predicted = [best_first]

        result.confidence = best_conf
        return result


class DifferentialLearner:
    """
    Main class that selects best approach based on pattern analysis.
    """

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.ensemble = EnsembleLearner()
        self.individual_learners = {
            "unigram": UnigramLearner(),
            "2gram": NGramLearner(n=2),
            "3gram": NGramLearner(n=3),
            "stack": StackAwareLearner(),
            "mode": ModeAwareLearner(),
            "counter": CounterAwareLearner(),
        }

    def learn(self, traces: List[List[str]]):
        """Train all approaches."""
        self.ensemble.learn(traces)
        for learner in self.individual_learners.values():
            learner.learn(traces)

    def predict_all(
        self,
        partial: List[str],
        n: int = 1
    ) -> Dict[str, LearnerResult]:
        """Get predictions from all approaches."""
        results = {}
        results["ensemble"] = self.ensemble.predict(partial, n)
        for name, learner in self.individual_learners.items():
            results[name] = learner.predict(partial, n)
        return results

    def predict_best(self, partial: List[str], n: int = 1) -> LearnerResult:
        """Predict using ensemble (best combined approach)."""
        return self.ensemble.predict(partial, n)


if __name__ == "__main__":
    print("Differential Learner Test")
    print("=" * 60)

    # Test context-dependent case
    traces = [
        ["A", "B", "C", "B", "A", "B", "C", "B", "A"],
        ["A", "B", "C", "B", "A", "B", "C"],
    ]
    partial = ["A", "B", "C", "B"]

    print("\nContext-dependent test:")
    print(f"  Traces: {traces}")
    print(f"  Partial: {partial}")
    print(f"  Expected: A (after C,B → A)")

    learner = DifferentialLearner(verbose=True)
    learner.learn(traces)

    results = learner.predict_all(partial, n=1)
    for name, result in results.items():
        print(f"  {name}: {result.predicted} (conf={result.confidence:.2f})")

    # Test counter case
    print("\n" + "=" * 60)
    traces2 = [
        ["X", "X", "X", "Y", "X", "X", "X", "Y"],
        ["X", "X", "X", "Y", "X", "X", "X", "Y", "X", "X", "X", "Y"],
    ]
    partial2 = ["X", "X", "X", "Y", "X", "X"]

    print("\nCounter test (3x X then Y):")
    print(f"  Partial: {partial2}")
    print(f"  Expected: X (need one more before Y)")

    learner2 = DifferentialLearner()
    learner2.learn(traces2)
    results2 = learner2.predict_all(partial2, n=1)
    for name, result in results2.items():
        print(f"  {name}: {result.predicted}")
