"""
Solution Correlator: Find patterns correlating with correct vs incorrect solutions.

Analyzes what distinguishes successful solving from failures:
1. Activation patterns at key iterations
2. Guard satisfaction trajectories
3. Confidence evolution
4. Attention-like patterns in cell interactions
"""

import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict

import mlx.core as mx
import numpy as np
from scipy import stats as scipy_stats

sys.path.insert(0, '/Volumes/tmc/go/src/github.com/tmc/sc/ml')

from .iteration_analyzer import IterationAnalyzer, IterationTrajectory


@dataclass
class CorrelationResult:
    """Result of correlation analysis."""
    feature_name: str
    correlation: float        # Pearson correlation with correctness
    p_value: float
    correct_mean: float       # Mean for correct solutions
    incorrect_mean: float     # Mean for incorrect solutions
    effect_size: float        # Cohen's d


@dataclass
class PatternSignature:
    """Signature pattern for correct/incorrect solutions."""
    category: str             # "correct" or "incorrect"
    num_samples: int

    # Trajectory statistics
    entropy_trajectory_mean: np.ndarray
    entropy_trajectory_std: np.ndarray
    confidence_trajectory_mean: np.ndarray
    confidence_trajectory_std: np.ndarray
    accuracy_trajectory_mean: np.ndarray
    changes_trajectory_mean: np.ndarray

    # Key metrics
    convergence_iteration_mean: Optional[float]
    final_entropy_mean: float
    final_confidence_mean: float
    guard_activation_mean: float


@dataclass
class DifferentiatingFeature:
    """A feature that differentiates correct from incorrect."""
    name: str
    iteration: int            # Iteration where difference is most pronounced
    correct_value: float
    incorrect_value: float
    ratio: float              # correct / incorrect (or vice versa)
    description: str


class SolutionCorrelator:
    """
    Analyzes patterns that correlate with solution correctness.

    Uses iteration trajectories to find:
    - Early indicators of success/failure
    - Critical iterations where outcomes diverge
    - Feature patterns predictive of correctness
    """

    def __init__(
        self,
        model,
        H_cycles: int = 3,
        L_cycles: int = 6,
    ):
        self.model = model
        self.H_cycles = H_cycles
        self.L_cycles = L_cycles
        self.analyzer = IterationAnalyzer(model, H_cycles, L_cycles)

        # Storage
        self.correct_trajectories: List[IterationTrajectory] = []
        self.incorrect_trajectories: List[IterationTrajectory] = []

    def add_trajectory(self, trajectory: IterationTrajectory):
        """Add a trajectory to the analysis."""
        if trajectory.is_correct:
            self.correct_trajectories.append(trajectory)
        else:
            self.incorrect_trajectories.append(trajectory)

    def analyze_puzzles(
        self,
        questions: mx.array,
        answers: mx.array,
    ):
        """Analyze a batch of puzzles and categorize by correctness."""
        trajectories = self.analyzer.analyze_batch(questions, answers)
        for t in trajectories:
            self.add_trajectory(t)

    def compute_signatures(self) -> Tuple[PatternSignature, PatternSignature]:
        """
        Compute pattern signatures for correct and incorrect solutions.

        Returns:
            (correct_signature, incorrect_signature)
        """
        correct_sig = self._compute_signature(self.correct_trajectories, "correct")
        incorrect_sig = self._compute_signature(self.incorrect_trajectories, "incorrect")
        return correct_sig, incorrect_sig

    def _compute_signature(
        self,
        trajectories: List[IterationTrajectory],
        category: str,
    ) -> PatternSignature:
        """Compute signature for a set of trajectories."""
        if not trajectories:
            return PatternSignature(
                category=category,
                num_samples=0,
                entropy_trajectory_mean=np.array([]),
                entropy_trajectory_std=np.array([]),
                confidence_trajectory_mean=np.array([]),
                confidence_trajectory_std=np.array([]),
                accuracy_trajectory_mean=np.array([]),
                changes_trajectory_mean=np.array([]),
                convergence_iteration_mean=None,
                final_entropy_mean=0,
                final_confidence_mean=0,
                guard_activation_mean=0,
            )

        # Collect trajectories
        entropy_trajs = np.array([t.get_entropy_trajectory() for t in trajectories])
        confidence_trajs = np.array([t.get_confidence_trajectory() for t in trajectories])
        accuracy_trajs = np.array([t.get_accuracy_trajectory() for t in trajectories])
        change_trajs = np.array([t.get_change_trajectory() for t in trajectories])

        # Convergence iterations
        convergences = [t.convergence_iteration for t in trajectories if t.convergence_iteration]

        # Guard activations
        guard_acts = []
        for t in trajectories:
            if t.snapshots:
                guard_acts.append(np.mean([s.guard_activations for s in t.snapshots]))

        return PatternSignature(
            category=category,
            num_samples=len(trajectories),
            entropy_trajectory_mean=np.mean(entropy_trajs, axis=0),
            entropy_trajectory_std=np.std(entropy_trajs, axis=0),
            confidence_trajectory_mean=np.mean(confidence_trajs, axis=0),
            confidence_trajectory_std=np.std(confidence_trajs, axis=0),
            accuracy_trajectory_mean=np.mean(accuracy_trajs, axis=0),
            changes_trajectory_mean=np.mean(change_trajs, axis=0),
            convergence_iteration_mean=np.mean(convergences) if convergences else None,
            final_entropy_mean=float(np.mean(entropy_trajs[:, -1])) if len(entropy_trajs) > 0 else 0,
            final_confidence_mean=float(np.mean(confidence_trajs[:, -1])) if len(confidence_trajs) > 0 else 0,
            guard_activation_mean=float(np.mean(guard_acts)) if guard_acts else 0,
        )

    def find_correlations(self) -> List[CorrelationResult]:
        """
        Find features correlated with solution correctness.

        Returns:
            List of CorrelationResult sorted by absolute correlation
        """
        results = []

        # Collect features from all trajectories
        all_trajectories = self.correct_trajectories + self.incorrect_trajectories
        if len(all_trajectories) < 5:
            return results

        # Binary outcome
        y = np.array([1 if t.is_correct else 0 for t in all_trajectories])

        # Feature: final entropy
        final_entropy = np.array([
            np.mean(t.snapshots[-1].board_entropy) if t.snapshots else 0
            for t in all_trajectories
        ])
        results.append(self._compute_correlation("final_entropy", final_entropy, y))

        # Feature: final confidence
        final_confidence = np.array([
            np.mean(t.snapshots[-1].board_confidence) if t.snapshots else 0
            for t in all_trajectories
        ])
        results.append(self._compute_correlation("final_confidence", final_confidence, y))

        # Feature: convergence speed
        convergence = np.array([
            t.convergence_iteration if t.convergence_iteration else self.H_cycles * self.L_cycles
            for t in all_trajectories
        ])
        results.append(self._compute_correlation("convergence_speed", -convergence, y))

        # Feature: guard activation mean
        guard_acts = np.array([
            np.mean([s.guard_activations for s in t.snapshots]) if t.snapshots else 0
            for t in all_trajectories
        ])
        results.append(self._compute_correlation("guard_activation", guard_acts, y))

        # Feature: accuracy at iteration 6 (after first H-cycle)
        if all(len(t.snapshots) > 6 for t in all_trajectories):
            acc_at_6 = np.array([
                t.snapshots[5].num_correct / 81.0
                for t in all_trajectories
            ])
            results.append(self._compute_correlation("accuracy_at_iter_6", acc_at_6, y))

        # Feature: entropy reduction rate
        entropy_reduction = np.array([
            (np.mean(t.snapshots[0].board_entropy) - np.mean(t.snapshots[-1].board_entropy))
            if t.snapshots else 0
            for t in all_trajectories
        ])
        results.append(self._compute_correlation("entropy_reduction", entropy_reduction, y))

        # Sort by absolute correlation
        results.sort(key=lambda r: abs(r.correlation), reverse=True)
        return results

    def _compute_correlation(
        self,
        name: str,
        feature: np.ndarray,
        outcome: np.ndarray,
    ) -> CorrelationResult:
        """Compute correlation between feature and binary outcome."""
        # Handle edge cases
        if len(feature) < 3 or np.std(feature) == 0:
            return CorrelationResult(
                feature_name=name,
                correlation=0.0,
                p_value=1.0,
                correct_mean=0.0,
                incorrect_mean=0.0,
                effect_size=0.0,
            )

        try:
            corr, p_val = scipy_stats.pearsonr(feature, outcome)
        except:
            corr, p_val = 0.0, 1.0

        # Compute means for each group
        correct_mask = outcome == 1
        incorrect_mask = outcome == 0

        correct_mean = float(np.mean(feature[correct_mask])) if np.any(correct_mask) else 0
        incorrect_mean = float(np.mean(feature[incorrect_mask])) if np.any(incorrect_mask) else 0

        # Cohen's d
        correct_std = np.std(feature[correct_mask]) if np.any(correct_mask) else 1
        incorrect_std = np.std(feature[incorrect_mask]) if np.any(incorrect_mask) else 1
        pooled_std = np.sqrt((correct_std**2 + incorrect_std**2) / 2)

        if pooled_std > 0:
            effect_size = (correct_mean - incorrect_mean) / pooled_std
        else:
            effect_size = 0.0

        return CorrelationResult(
            feature_name=name,
            correlation=float(corr),
            p_value=float(p_val),
            correct_mean=correct_mean,
            incorrect_mean=incorrect_mean,
            effect_size=float(effect_size),
        )

    def find_differentiating_features(self) -> List[DifferentiatingFeature]:
        """
        Find features that most differentiate correct from incorrect.

        Returns:
            List of DifferentiatingFeature
        """
        features = []
        correct_sig, incorrect_sig = self.compute_signatures()

        if correct_sig.num_samples == 0 or incorrect_sig.num_samples == 0:
            return features

        # Find iteration with largest accuracy gap
        acc_diff = correct_sig.accuracy_trajectory_mean - incorrect_sig.accuracy_trajectory_mean
        if len(acc_diff) > 0:
            best_iter = int(np.argmax(np.abs(acc_diff)))
            features.append(DifferentiatingFeature(
                name="accuracy_gap",
                iteration=best_iter,
                correct_value=float(correct_sig.accuracy_trajectory_mean[best_iter]),
                incorrect_value=float(incorrect_sig.accuracy_trajectory_mean[best_iter]),
                ratio=float(correct_sig.accuracy_trajectory_mean[best_iter] /
                           max(incorrect_sig.accuracy_trajectory_mean[best_iter], 0.01)),
                description=f"Accuracy gap largest at iteration {best_iter}",
            ))

        # Find iteration with largest confidence gap
        conf_diff = correct_sig.confidence_trajectory_mean - incorrect_sig.confidence_trajectory_mean
        if len(conf_diff) > 0:
            best_iter = int(np.argmax(np.abs(conf_diff)))
            features.append(DifferentiatingFeature(
                name="confidence_gap",
                iteration=best_iter,
                correct_value=float(correct_sig.confidence_trajectory_mean[best_iter]),
                incorrect_value=float(incorrect_sig.confidence_trajectory_mean[best_iter]),
                ratio=float(correct_sig.confidence_trajectory_mean[best_iter] /
                           max(incorrect_sig.confidence_trajectory_mean[best_iter], 0.01)),
                description=f"Confidence gap largest at iteration {best_iter}",
            ))

        # Convergence speed
        if correct_sig.convergence_iteration_mean and incorrect_sig.convergence_iteration_mean:
            features.append(DifferentiatingFeature(
                name="convergence_speed",
                iteration=-1,
                correct_value=correct_sig.convergence_iteration_mean,
                incorrect_value=incorrect_sig.convergence_iteration_mean,
                ratio=incorrect_sig.convergence_iteration_mean / max(correct_sig.convergence_iteration_mean, 0.1),
                description="Correct solutions converge faster",
            ))

        return features


def correlate_solutions(
    model,
    questions: mx.array,
    answers: mx.array,
    H_cycles: int = 3,
    L_cycles: int = 6,
) -> Dict:
    """
    Convenience function to run full correlation analysis.

    Returns:
        Dict with signatures, correlations, and differentiating features
    """
    correlator = SolutionCorrelator(model, H_cycles, L_cycles)
    correlator.analyze_puzzles(questions, answers)

    correct_sig, incorrect_sig = correlator.compute_signatures()
    correlations = correlator.find_correlations()
    diff_features = correlator.find_differentiating_features()

    return {
        "num_correct": correct_sig.num_samples,
        "num_incorrect": incorrect_sig.num_samples,
        "correct_signature": {
            "final_entropy": correct_sig.final_entropy_mean,
            "final_confidence": correct_sig.final_confidence_mean,
            "convergence": correct_sig.convergence_iteration_mean,
        },
        "incorrect_signature": {
            "final_entropy": incorrect_sig.final_entropy_mean,
            "final_confidence": incorrect_sig.final_confidence_mean,
            "convergence": incorrect_sig.convergence_iteration_mean,
        },
        "top_correlations": [
            {
                "feature": c.feature_name,
                "correlation": c.correlation,
                "p_value": c.p_value,
                "effect_size": c.effect_size,
            }
            for c in correlations[:5]
        ],
        "differentiating_features": [
            {
                "name": f.name,
                "iteration": f.iteration,
                "ratio": f.ratio,
                "description": f.description,
            }
            for f in diff_features
        ],
    }


def test_solution_correlator():
    """Test solution correlation analysis."""
    print("=" * 60)
    print("Testing Solution Correlator")
    print("=" * 60)

    from ..exp_trm_sudoku_9x9.sudoku_statechart_9x9 import SudokuStatechart9x9

    # Create model
    model = SudokuStatechart9x9(
        hidden_dim=64,
        H_cycles=2,
        L_cycles=3,
    )

    # Create correlator
    correlator = SolutionCorrelator(model, H_cycles=2, L_cycles=3)

    # Generate test puzzles
    mx.random.seed(42)
    questions = mx.random.randint(0, 10, (20, 81))
    answers = mx.random.randint(1, 10, (20, 81))

    print("\n1. Analyzing puzzles...")
    correlator.analyze_puzzles(questions, answers)

    print(f"   Correct: {len(correlator.correct_trajectories)}")
    print(f"   Incorrect: {len(correlator.incorrect_trajectories)}")

    print("\n2. Computing signatures...")
    correct_sig, incorrect_sig = correlator.compute_signatures()

    print(f"   Correct signature:")
    print(f"     Final entropy: {correct_sig.final_entropy_mean:.3f}")
    print(f"     Final confidence: {correct_sig.final_confidence_mean:.3f}")

    print(f"   Incorrect signature:")
    print(f"     Final entropy: {incorrect_sig.final_entropy_mean:.3f}")
    print(f"     Final confidence: {incorrect_sig.final_confidence_mean:.3f}")

    print("\n3. Finding correlations...")
    correlations = correlator.find_correlations()

    for c in correlations[:3]:
        print(f"   {c.feature_name}: r={c.correlation:.3f}, p={c.p_value:.3f}, d={c.effect_size:.3f}")

    print("\n4. Finding differentiating features...")
    diff_features = correlator.find_differentiating_features()

    for f in diff_features:
        print(f"   {f.name}: {f.description}")

    print("\n5. Full analysis...")
    results = correlate_solutions(model, questions, answers, H_cycles=2, L_cycles=3)
    print(f"   Results: {list(results.keys())}")

    print("\n" + "=" * 60)
    print("Solution correlator test complete")
    print("=" * 60)


if __name__ == "__main__":
    test_solution_correlator()
