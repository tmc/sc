"""
Benchmark: Clarity Evaluation for Statechart Explanations

Metrics:
1. Clarity Score (1-5): How understandable is the explanation?
2. Completeness: Does it cover all key elements?
3. Accuracy: Is the explanation factually correct?
4. Fluency: Is the language natural and readable?

Target: 4+/5 average clarity rating.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional, Any, Tuple, Callable
from enum import Enum, auto
import time
import re
import json

from .explainer import StatechartExplainer, ExplanationConfig, Explanation, ExplanationLevel
from .summary_generator import SummaryGenerator, StatechartSummary
from .detail_generator import DetailGenerator, TransitionExplanation, StateExplanation


class ClarityDimension(Enum):
    """Dimensions of clarity assessment."""
    UNDERSTANDABILITY = "understandability"
    COMPLETENESS = "completeness"
    ACCURACY = "accuracy"
    FLUENCY = "fluency"
    CONCISENESS = "conciseness"


@dataclass
class ClarityRating:
    """Rating for a single clarity dimension."""
    dimension: ClarityDimension
    score: float  # 1-5 scale
    rationale: str


@dataclass
class BenchmarkResult:
    """Result from benchmarking a single explanation."""
    statechart_name: str
    explanation_level: str
    time_ms: float

    # Overall scores
    overall_clarity: float  # 1-5
    meets_target: bool      # >= 4.0

    # Dimension scores
    dimension_scores: Dict[str, float]
    ratings: List[ClarityRating]

    # Metrics
    word_count: int
    sentence_count: int
    avg_sentence_length: float
    coverage_score: float   # % of elements explained

    def to_dict(self) -> Dict:
        return {
            "name": self.statechart_name,
            "level": self.explanation_level,
            "overall_clarity": self.overall_clarity,
            "meets_target": self.meets_target,
            "dimensions": self.dimension_scores,
            "word_count": self.word_count,
            "coverage": self.coverage_score,
            "time_ms": self.time_ms,
        }


@dataclass
class BenchmarkSuite:
    """Collection of benchmark results."""
    results: List[BenchmarkResult] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)

    def add_result(self, result: BenchmarkResult):
        self.results.append(result)

    @property
    def average_clarity(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.overall_clarity for r in self.results) / len(self.results)

    @property
    def pass_rate(self) -> float:
        if not self.results:
            return 0.0
        passing = sum(1 for r in self.results if r.meets_target)
        return passing / len(self.results)

    def summary(self) -> Dict[str, Any]:
        return {
            "total_benchmarks": len(self.results),
            "average_clarity": round(self.average_clarity, 2),
            "pass_rate": f"{self.pass_rate:.1%}",
            "target_met": self.average_clarity >= 4.0,
            "by_level": self._group_by_level(),
        }

    def _group_by_level(self) -> Dict[str, float]:
        by_level = {}
        for result in self.results:
            level = result.explanation_level
            if level not in by_level:
                by_level[level] = []
            by_level[level].append(result.overall_clarity)

        return {
            level: round(sum(scores) / len(scores), 2)
            for level, scores in by_level.items()
        }


class ClarityEvaluator:
    """
    Evaluate clarity of statechart explanations.

    Uses heuristics and optional LLM-based evaluation.
    """

    def __init__(self, use_llm: bool = False):
        self.use_llm = use_llm

    def evaluate(
        self,
        explanation: Explanation,
        statechart: Dict[str, Any],
    ) -> List[ClarityRating]:
        """Evaluate explanation clarity across all dimensions."""
        ratings = []

        # 1. Understandability
        ratings.append(self._evaluate_understandability(explanation))

        # 2. Completeness
        ratings.append(self._evaluate_completeness(explanation, statechart))

        # 3. Accuracy
        ratings.append(self._evaluate_accuracy(explanation, statechart))

        # 4. Fluency
        ratings.append(self._evaluate_fluency(explanation))

        # 5. Conciseness
        ratings.append(self._evaluate_conciseness(explanation, statechart))

        return ratings

    def _evaluate_understandability(self, exp: Explanation) -> ClarityRating:
        """Evaluate how understandable the explanation is."""
        score = 3.0
        reasons = []

        text = exp.full_text

        # Sentence length (shorter is clearer)
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        if sentences:
            avg_len = sum(len(s.split()) for s in sentences) / len(sentences)
            if avg_len <= 15:
                score += 1.0
                reasons.append("Good sentence length")
            elif avg_len <= 20:
                score += 0.5
                reasons.append("Acceptable sentence length")
            elif avg_len > 30:
                score -= 0.5
                reasons.append("Sentences too long")

        # Jargon check
        jargon = ["orthogonal", "hierarchical", "LCA", "configuration", "semantics"]
        jargon_count = sum(1 for j in jargon if j.lower() in text.lower())
        if jargon_count == 0:
            score += 0.5
            reasons.append("No jargon")
        elif jargon_count > 3:
            score -= 0.5
            reasons.append("Too much jargon")

        # Structure (has summary and details)
        if exp.summary and exp.state_descriptions:
            score += 0.5
            reasons.append("Well-structured")

        score = max(1.0, min(5.0, score))
        return ClarityRating(
            dimension=ClarityDimension.UNDERSTANDABILITY,
            score=score,
            rationale="; ".join(reasons) if reasons else "Average understandability",
        )

    def _evaluate_completeness(
        self,
        exp: Explanation,
        sc: Dict,
    ) -> ClarityRating:
        """Evaluate if all key elements are explained."""
        score = 3.0
        reasons = []

        # Count states and transitions
        n_states = len(exp.state_descriptions)
        n_trans = len(exp.transition_descriptions)

        # Expected counts
        expected_states = 0
        root = sc.get("root_state", {})

        def count_states(state):
            nonlocal expected_states
            if not state.get("label", "").startswith("__"):
                expected_states += 1
            for child in state.get("children", []):
                count_states(child)

        count_states(root)
        expected_trans = len(sc.get("transitions", []))

        # Coverage ratios
        state_coverage = n_states / expected_states if expected_states > 0 else 1.0
        trans_coverage = n_trans / expected_trans if expected_trans > 0 else 1.0

        avg_coverage = (state_coverage + trans_coverage) / 2

        if avg_coverage >= 0.9:
            score = 5.0
            reasons.append("Complete coverage")
        elif avg_coverage >= 0.7:
            score = 4.0
            reasons.append("Good coverage")
        elif avg_coverage >= 0.5:
            score = 3.0
            reasons.append("Partial coverage")
        else:
            score = 2.0
            reasons.append("Incomplete coverage")

        # Check for summary
        if exp.summary and len(exp.summary) > 20:
            score += 0.5
            reasons.append("Has summary")
        else:
            score -= 0.5
            reasons.append("Missing or weak summary")

        score = max(1.0, min(5.0, score))
        return ClarityRating(
            dimension=ClarityDimension.COMPLETENESS,
            score=score,
            rationale="; ".join(reasons),
        )

    def _evaluate_accuracy(
        self,
        exp: Explanation,
        sc: Dict,
    ) -> ClarityRating:
        """Evaluate factual accuracy of explanation."""
        score = 4.0  # Start optimistic
        reasons = []

        text = exp.full_text.lower()

        # Check that mentioned states exist
        states_in_sc = set()
        root = sc.get("root_state", {})

        def collect_states(state):
            label = state.get("label", "")
            if not label.startswith("__"):
                states_in_sc.add(label.lower())
            for child in state.get("children", []):
                collect_states(child)

        collect_states(root)

        # Check state mentions
        for state in exp.state_descriptions:
            if state.lower() not in states_in_sc:
                score -= 0.5
                reasons.append(f"Mentions non-existent state: {state}")

        # Check transition descriptions
        transitions_in_sc = sc.get("transitions", [])
        events_in_sc = set(t.get("event", "").lower() for t in transitions_in_sc)

        for desc in exp.transition_descriptions:
            desc_lower = desc.lower()
            # Check if event mentioned exists
            for event in events_in_sc:
                if event and event in desc_lower:
                    break
            else:
                # Check if it's a completion transition
                if "completion" not in desc_lower and "automatic" not in desc_lower:
                    score -= 0.25

        if not reasons:
            reasons.append("Accurate representation")

        score = max(1.0, min(5.0, score))
        return ClarityRating(
            dimension=ClarityDimension.ACCURACY,
            score=score,
            rationale="; ".join(reasons),
        )

    def _evaluate_fluency(self, exp: Explanation) -> ClarityRating:
        """Evaluate language fluency and readability."""
        score = 3.5
        reasons = []

        text = exp.full_text

        # Check for complete sentences
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]

        if sentences:
            # Capitalization check
            capitalized = sum(1 for s in sentences if s[0].isupper())
            cap_ratio = capitalized / len(sentences)
            if cap_ratio > 0.9:
                score += 0.5
                reasons.append("Proper capitalization")

            # No fragment sentences (> 3 words)
            complete = sum(1 for s in sentences if len(s.split()) >= 4)
            complete_ratio = complete / len(sentences)
            if complete_ratio > 0.8:
                score += 0.5
                reasons.append("Complete sentences")

        # Check for repetition
        words = text.lower().split()
        if len(words) > 10:
            word_freq = {}
            for w in words:
                word_freq[w] = word_freq.get(w, 0) + 1

            # Ignore common words
            common = {"the", "a", "an", "is", "are", "in", "to", "and", "of", "for"}
            high_freq = [w for w, c in word_freq.items() if c > 3 and w not in common]
            if len(high_freq) > 3:
                score -= 0.5
                reasons.append("Some repetition")

        # Natural language markers
        natural_markers = ["when", "while", "if", "then", "because", "after"]
        has_markers = any(m in text.lower() for m in natural_markers)
        if has_markers:
            score += 0.5
            reasons.append("Natural phrasing")

        score = max(1.0, min(5.0, score))
        return ClarityRating(
            dimension=ClarityDimension.FLUENCY,
            score=score,
            rationale="; ".join(reasons) if reasons else "Average fluency",
        )

    def _evaluate_conciseness(
        self,
        exp: Explanation,
        sc: Dict,
    ) -> ClarityRating:
        """Evaluate conciseness (not too verbose, not too terse)."""
        score = 3.5
        reasons = []

        # Count elements
        n_states = sum(1 for _ in self._iter_states(sc.get("root_state", {})))
        n_trans = len(sc.get("transitions", []))
        complexity = n_states + n_trans

        # Word count
        words = len(exp.full_text.split())

        # Target: ~20-40 words per element
        words_per_element = words / complexity if complexity > 0 else words

        if 20 <= words_per_element <= 40:
            score = 5.0
            reasons.append("Ideal length")
        elif 15 <= words_per_element <= 50:
            score = 4.0
            reasons.append("Good length")
        elif words_per_element < 10:
            score = 2.5
            reasons.append("Too terse")
        elif words_per_element > 60:
            score = 2.5
            reasons.append("Too verbose")
        else:
            score = 3.0

        score = max(1.0, min(5.0, score))
        return ClarityRating(
            dimension=ClarityDimension.CONCISENESS,
            score=score,
            rationale="; ".join(reasons) if reasons else "Acceptable length",
        )

    def _iter_states(self, state: Dict):
        """Iterate over states."""
        if not state.get("label", "").startswith("__"):
            yield state
        for child in state.get("children", []):
            yield from self._iter_states(child)


class ClarityBenchmark:
    """
    Comprehensive benchmark for explanation clarity.

    Target: 4+/5 average clarity rating.
    """

    def __init__(self):
        self.suite = BenchmarkSuite()
        self.evaluator = ClarityEvaluator()
        self.explainer = StatechartExplainer()

    def benchmark_statechart(
        self,
        statechart: Dict[str, Any],
        levels: List[ExplanationLevel] = None,
    ) -> List[BenchmarkResult]:
        """Benchmark explanations at different levels."""
        levels = levels or [ExplanationLevel.STANDARD]
        results = []

        for level in levels:
            t0 = time.time()

            # Configure and generate
            config = ExplanationConfig(level=level)
            explainer = StatechartExplainer(config)
            explanation = explainer.explain(statechart)

            time_ms = (time.time() - t0) * 1000

            # Evaluate clarity
            ratings = self.evaluator.evaluate(explanation, statechart)

            # Calculate scores
            dimension_scores = {r.dimension.value: r.score for r in ratings}
            overall = sum(r.score for r in ratings) / len(ratings) if ratings else 0.0

            # Text metrics
            text = explanation.full_text
            words = text.split()
            sentences = re.split(r'[.!?]+', text)
            sentences = [s.strip() for s in sentences if s.strip()]

            # Coverage
            n_states = sum(1 for _ in self.evaluator._iter_states(
                statechart.get("root_state", {})
            ))
            explained_states = len(explanation.state_descriptions)
            coverage = explained_states / n_states if n_states > 0 else 1.0

            result = BenchmarkResult(
                statechart_name=statechart.get("name", "Unknown"),
                explanation_level=level.value,
                time_ms=time_ms,
                overall_clarity=round(overall, 2),
                meets_target=overall >= 4.0,
                dimension_scores=dimension_scores,
                ratings=ratings,
                word_count=len(words),
                sentence_count=len(sentences),
                avg_sentence_length=len(words) / len(sentences) if sentences else 0,
                coverage_score=round(coverage, 2),
            )

            results.append(result)
            self.suite.add_result(result)

        return results


# Test statecharts for benchmarking
BENCHMARK_STATECHARTS = [
    {
        "name": "Traffic Light",
        "root_state": {
            "label": "__root__",
            "type": 2,
            "children": [
                {"label": "Red", "type": 1, "is_initial": True},
                {"label": "Green", "type": 1},
                {"label": "Yellow", "type": 1},
            ]
        },
        "transitions": [
            {"from": ["Red"], "to": ["Green"], "event": "TIMER"},
            {"from": ["Green"], "to": ["Yellow"], "event": "TIMER"},
            {"from": ["Yellow"], "to": ["Red"], "event": "TIMER"},
        ]
    },
    {
        "name": "User Authentication",
        "root_state": {
            "label": "__root__",
            "type": 2,
            "children": [
                {"label": "LoggedOut", "type": 1, "is_initial": True},
                {"label": "Authenticating", "type": 1},
                {"label": "LoggedIn", "type": 1},
                {"label": "Locked", "type": 1},
            ]
        },
        "transitions": [
            {"from": ["LoggedOut"], "to": ["Authenticating"], "event": "LOGIN_ATTEMPT"},
            {"from": ["Authenticating"], "to": ["LoggedIn"], "event": "SUCCESS"},
            {"from": ["Authenticating"], "to": ["LoggedOut"], "event": "FAILURE",
             "guard": "attempts < 3"},
            {"from": ["Authenticating"], "to": ["Locked"], "event": "FAILURE",
             "guard": "attempts >= 3"},
            {"from": ["LoggedIn"], "to": ["LoggedOut"], "event": "LOGOUT"},
            {"from": ["Locked"], "to": ["LoggedOut"], "event": "TIMEOUT"},
        ]
    },
    {
        "name": "Media Player",
        "root_state": {
            "label": "__root__",
            "type": 2,
            "children": [
                {"label": "Stopped", "type": 1, "is_initial": True},
                {"label": "Playing", "type": 1},
                {"label": "Paused", "type": 1},
            ]
        },
        "transitions": [
            {"from": ["Stopped"], "to": ["Playing"], "event": "PLAY"},
            {"from": ["Playing"], "to": ["Paused"], "event": "PAUSE"},
            {"from": ["Paused"], "to": ["Playing"], "event": "PLAY"},
            {"from": ["Playing"], "to": ["Stopped"], "event": "STOP"},
            {"from": ["Paused"], "to": ["Stopped"], "event": "STOP"},
        ]
    },
]


def run_benchmark(verbose: bool = True) -> BenchmarkSuite:
    """Run comprehensive clarity benchmark."""
    benchmark = ClarityBenchmark()

    if verbose:
        print("=" * 70)
        print("CLARITY BENCHMARK: Target 4+/5 Rating")
        print("=" * 70)

    levels = [ExplanationLevel.BRIEF, ExplanationLevel.STANDARD, ExplanationLevel.DETAILED]

    for sc in BENCHMARK_STATECHARTS:
        if verbose:
            print(f"\n--- {sc['name']} ---")

        results = benchmark.benchmark_statechart(sc, levels)

        if verbose:
            for r in results:
                status = "PASS" if r.meets_target else "FAIL"
                print(f"  [{status}] {r.explanation_level}: {r.overall_clarity}/5 "
                      f"({r.word_count} words, {r.time_ms:.1f}ms)")

                for dim, score in r.dimension_scores.items():
                    print(f"       {dim}: {score}")

    # Summary
    if verbose:
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        summary = benchmark.suite.summary()
        print(f"Total benchmarks: {summary['total_benchmarks']}")
        print(f"Average clarity: {summary['average_clarity']}/5")
        print(f"Pass rate (>=4): {summary['pass_rate']}")
        print(f"Target met: {'YES' if summary['target_met'] else 'NO'}")
        print(f"\nBy level:")
        for level, score in summary['by_level'].items():
            print(f"  {level}: {score}/5")

    return benchmark.suite


def demo():
    """Quick benchmark demo."""
    print("Running clarity benchmark...")
    return run_benchmark(verbose=True)


if __name__ == "__main__":
    demo()
