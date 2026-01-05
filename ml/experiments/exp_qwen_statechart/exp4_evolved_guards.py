"""
Experiment 4: Evolved Guard Fine-Tuning

Goal: Use guard synthesis to identify and penalize error-inducing patterns.

Approach:
1. Collect failure examples during generation
2. Evolve guards that predict failures
3. Train with guard-weighted loss (penalize violations more)
"""

from typing import List, Dict, Any
from dataclasses import dataclass
import random


@dataclass
class FailureFeatures:
    """Features extracted from a program for failure prediction."""
    has_loop: bool
    has_recursion: bool
    max_depth: int
    unbalanced_brackets: bool
    line_count: int
    code: str


class FailurePredictor:
    """Predicts program failures using evolved guard patterns."""

    def __init__(self):
        self.failure_examples: List[FailureFeatures] = []
        self.success_examples: List[FailureFeatures] = []
        self.guard_expr: str = "True"  # Default: no filtering
        self.guard_f1: float = 0.0

    def extract_features(self, code: str) -> FailureFeatures:
        """Extract features from code."""
        return FailureFeatures(
            has_loop='for ' in code or 'while ' in code,
            has_recursion=False,  # Simplified
            max_depth=max((len(line) - len(line.lstrip())) // 4
                         for line in code.split('\n') if line.strip()),
            unbalanced_brackets=(
                code.count('(') != code.count(')') or
                code.count('[') != code.count(']') or
                code.count('{') != code.count('}')
            ),
            line_count=len(code.split('\n')),
            code=code,
        )

    def collect(self, code: str, success: bool):
        """Collect program execution outcome."""
        features = self.extract_features(code)
        if success:
            self.success_examples.append(features)
        else:
            self.failure_examples.append(features)

    def evolve_guard(self, n_generations: int = 50) -> str:
        """Evolve a guard expression that predicts failures."""
        if not self.failure_examples or not self.success_examples:
            return "True"

        # Simple guard patterns to try
        patterns = [
            "unbalanced_brackets",
            "max_depth > 5",
            "line_count > 20",
            "has_loop and max_depth > 3",
            "unbalanced_brackets or max_depth > 6",
        ]

        best_guard = "True"
        best_f1 = 0.0

        for pattern in patterns:
            tp, fp, fn = 0, 0, 0

            for f in self.failure_examples:
                try:
                    if eval(pattern, vars(f)):
                        tp += 1
                    else:
                        fn += 1
                except:
                    fn += 1

            for f in self.success_examples:
                try:
                    if eval(pattern, vars(f)):
                        fp += 1
                except:
                    pass

            precision = tp / max(1, tp + fp)
            recall = tp / max(1, tp + fn)
            f1 = 2 * precision * recall / max(0.001, precision + recall)

            if f1 > best_f1:
                best_f1 = f1
                best_guard = pattern

        self.guard_expr = best_guard
        self.guard_f1 = best_f1
        return best_guard

    def predict_failure(self, code: str) -> bool:
        """Predict if code will fail."""
        features = self.extract_features(code)
        try:
            return eval(self.guard_expr, vars(features))
        except:
            return False


def run_experiment_4(n_programs: int = 100, verbose: bool = True) -> Dict[str, Any]:
    """Evolve failure predictor and measure effectiveness."""

    predictor = FailurePredictor()

    # Generate random programs and simulate execution
    for _ in range(n_programs):
        code = generate_random_code()
        success = simulate_execution(code)
        predictor.collect(code, success)

    # Evolve guard
    guard = predictor.evolve_guard()

    results = {
        'total_programs': n_programs,
        'failures': len(predictor.failure_examples),
        'successes': len(predictor.success_examples),
        'evolved_guard': guard,
        'guard_f1': predictor.guard_f1,
    }

    if verbose:
        print("\n" + "=" * 60)
        print("EXPERIMENT 4 RESULTS: Evolved Guards")
        print("=" * 60)
        print(f"Programs: {results['total_programs']}")
        print(f"Failures: {results['failures']}, Successes: {results['successes']}")
        print(f"Evolved guard: {results['evolved_guard']}")
        print(f"Guard F1: {results['guard_f1']:.3f}")

    return results


def generate_random_code() -> str:
    """Generate random Starlark-like code."""
    templates = [
        "def foo():\n    return None",
        "def bar():\n    for i in range(10):\n        x = i",
        "def baz():\n    return [1, 2, 3]",
        "def broken(:\n    return",  # Invalid
        "def deep():\n    if True:\n        if True:\n            if True:\n                pass",
    ]
    return random.choice(templates)


def simulate_execution(code: str) -> bool:
    """Simulate program execution."""
    try:
        import ast
        ast.parse(code)
        return random.random() > 0.3  # 70% of valid code succeeds
    except:
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("EXPERIMENT 4: Evolved Guard Fine-Tuning")
    print("=" * 60)
    run_experiment_4(n_programs=50, verbose=True)
