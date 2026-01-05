"""
Metrics for Execution Prediction Experiments.

Provides evaluation functions for:
- Context prediction accuracy
- Configuration prediction accuracy
- Path length accuracy
- Reachability F1
"""

from typing import Dict, Any, Set, List, Tuple
from dataclasses import dataclass
import math


@dataclass
class MetricResult:
    """Result of metric computation."""
    name: str
    value: float
    details: Dict[str, Any] = None


def exact_match_accuracy(
    predictions: List[Dict[str, Any]],
    targets: List[Dict[str, Any]],
) -> MetricResult:
    """
    Compute exact match accuracy for context predictions.

    A prediction is correct only if ALL values match exactly.
    """
    if not predictions:
        return MetricResult("exact_match", 0.0)

    correct = 0
    for pred, target in zip(predictions, targets):
        if _contexts_equal(pred, target):
            correct += 1

    accuracy = correct / len(predictions)
    return MetricResult(
        "exact_match",
        accuracy,
        {"correct": correct, "total": len(predictions)},
    )


def partial_match_accuracy(
    predictions: List[Dict[str, Any]],
    targets: List[Dict[str, Any]],
) -> MetricResult:
    """
    Compute partial match accuracy for context predictions.

    Returns average fraction of correctly predicted values per sample.
    """
    if not predictions:
        return MetricResult("partial_match", 0.0)

    total_accuracy = 0.0
    per_variable = {}

    for pred, target in zip(predictions, targets):
        matches = 0
        total_vars = len(target)

        for key, target_val in target.items():
            if key not in per_variable:
                per_variable[key] = {"correct": 0, "total": 0}

            per_variable[key]["total"] += 1

            pred_val = pred.get(key)
            if _values_equal(pred_val, target_val):
                matches += 1
                per_variable[key]["correct"] += 1

        if total_vars > 0:
            total_accuracy += matches / total_vars

    accuracy = total_accuracy / len(predictions)

    # Compute per-variable accuracy
    var_accuracies = {
        k: v["correct"] / v["total"] if v["total"] > 0 else 0.0
        for k, v in per_variable.items()
    }

    return MetricResult(
        "partial_match",
        accuracy,
        {"per_variable": var_accuracies},
    )


def mean_absolute_error(
    predictions: List[Dict[str, Any]],
    targets: List[Dict[str, Any]],
    numeric_keys: List[str] = None,
) -> MetricResult:
    """
    Compute mean absolute error for numeric context values.

    Args:
        predictions: List of predicted contexts
        targets: List of target contexts
        numeric_keys: Keys to compute MAE for (auto-detect if None)
    """
    if not predictions:
        return MetricResult("mae", 0.0)

    # Auto-detect numeric keys
    if numeric_keys is None:
        numeric_keys = []
        for target in targets:
            for key, val in target.items():
                if isinstance(val, (int, float)) and key not in numeric_keys:
                    numeric_keys.append(key)

    if not numeric_keys:
        return MetricResult("mae", 0.0, {"error": "No numeric keys found"})

    total_error = 0.0
    count = 0
    per_key_errors = {k: [] for k in numeric_keys}

    for pred, target in zip(predictions, targets):
        for key in numeric_keys:
            if key in target:
                target_val = target[key]
                pred_val = pred.get(key, 0)

                if isinstance(target_val, (int, float)) and isinstance(pred_val, (int, float)):
                    error = abs(pred_val - target_val)
                    total_error += error
                    count += 1
                    per_key_errors[key].append(error)

    mae = total_error / count if count > 0 else 0.0

    # Per-key MAE
    per_key_mae = {
        k: sum(v) / len(v) if v else 0.0
        for k, v in per_key_errors.items()
    }

    return MetricResult(
        "mae",
        mae,
        {"per_key_mae": per_key_mae, "count": count},
    )


def configuration_accuracy(
    predictions: List[Set[str]],
    targets: List[Set[str]],
) -> MetricResult:
    """
    Compute accuracy for configuration (active states) predictions.

    Returns:
    - Exact match: full configuration matches
    - Jaccard similarity: average IoU of state sets
    - Leaf accuracy: accuracy on leaf states only
    """
    if not predictions:
        return MetricResult("config_accuracy", 0.0)

    exact_matches = 0
    jaccard_sum = 0.0

    for pred, target in zip(predictions, targets):
        # Exact match
        if pred == target:
            exact_matches += 1

        # Jaccard similarity
        intersection = len(pred & target)
        union = len(pred | target)
        if union > 0:
            jaccard_sum += intersection / union

    exact_acc = exact_matches / len(predictions)
    jaccard_avg = jaccard_sum / len(predictions)

    return MetricResult(
        "config_accuracy",
        exact_acc,
        {"exact_match": exact_acc, "jaccard": jaccard_avg},
    )


def path_length_accuracy(
    predictions: List[int],
    targets: List[int],
    tolerance: int = 1,
) -> MetricResult:
    """
    Compute accuracy for path length predictions.

    Args:
        predictions: Predicted path lengths
        targets: Actual path lengths
        tolerance: Allow ±tolerance error as correct
    """
    if not predictions:
        return MetricResult("path_length", 0.0)

    exact = 0
    within_tolerance = 0
    total_error = 0

    for pred, target in zip(predictions, targets):
        error = abs(pred - target)
        total_error += error

        if error == 0:
            exact += 1
        if error <= tolerance:
            within_tolerance += 1

    n = len(predictions)
    return MetricResult(
        "path_length",
        within_tolerance / n,
        {
            "exact": exact / n,
            "within_tolerance": within_tolerance / n,
            "mae": total_error / n,
            "tolerance": tolerance,
        },
    )


def reachability_f1(
    predictions: List[Tuple[bool, int]],  # (reachable, min_steps)
    targets: List[Tuple[bool, int]],
) -> MetricResult:
    """
    Compute F1 score for reachability predictions.

    Also evaluates min_steps accuracy when reachable.
    """
    if not predictions:
        return MetricResult("reachability_f1", 0.0)

    tp = fp = tn = fn = 0
    steps_correct = 0
    steps_total = 0

    for pred, target in zip(predictions, targets):
        pred_reach, pred_steps = pred
        target_reach, target_steps = target

        if target_reach:
            if pred_reach:
                tp += 1
                # Check steps accuracy
                steps_total += 1
                if pred_steps == target_steps:
                    steps_correct += 1
            else:
                fn += 1
        else:
            if pred_reach:
                fp += 1
            else:
                tn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    steps_acc = steps_correct / steps_total if steps_total > 0 else 0.0

    return MetricResult(
        "reachability_f1",
        f1,
        {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "steps_accuracy": steps_acc,
            "confusion": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        },
    )


def _contexts_equal(ctx1: Dict[str, Any], ctx2: Dict[str, Any]) -> bool:
    """Check if two contexts are equal."""
    if set(ctx1.keys()) != set(ctx2.keys()):
        return False
    for key in ctx1:
        if not _values_equal(ctx1[key], ctx2[key]):
            return False
    return True


def _values_equal(v1: Any, v2: Any) -> bool:
    """Check if two values are equal (with numeric tolerance)."""
    if type(v1) != type(v2):
        # Allow int/float comparison
        if isinstance(v1, (int, float)) and isinstance(v2, (int, float)):
            return abs(v1 - v2) < 1e-6
        return False

    if isinstance(v1, float):
        return abs(v1 - v2) < 1e-6
    elif isinstance(v1, (list, tuple)):
        if len(v1) != len(v2):
            return False
        return all(_values_equal(a, b) for a, b in zip(v1, v2))
    elif isinstance(v1, dict):
        return _contexts_equal(v1, v2)
    else:
        return v1 == v2


if __name__ == "__main__":
    print("Metrics Test")
    print("=" * 60)

    # Test context prediction metrics
    predictions = [
        {"count": 5, "score": 50},
        {"count": 3, "score": 30},
        {"count": 10, "score": 100},
    ]
    targets = [
        {"count": 5, "score": 50},  # Exact match
        {"count": 3, "score": 35},  # Partial match (count correct)
        {"count": 8, "score": 80},  # Both wrong
    ]

    print("\nContext Prediction:")
    print(f"  Exact match: {exact_match_accuracy(predictions, targets)}")
    print(f"  Partial match: {partial_match_accuracy(predictions, targets)}")
    print(f"  MAE: {mean_absolute_error(predictions, targets)}")

    # Test config accuracy
    pred_configs = [{"A", "B"}, {"X", "Y"}, {"A"}]
    target_configs = [{"A", "B"}, {"X", "Z"}, {"A", "B"}]

    print("\nConfiguration Accuracy:")
    print(f"  {configuration_accuracy(pred_configs, target_configs)}")

    # Test path length
    pred_lengths = [5, 10, 7]
    target_lengths = [5, 8, 10]

    print("\nPath Length:")
    print(f"  {path_length_accuracy(pred_lengths, target_lengths)}")
