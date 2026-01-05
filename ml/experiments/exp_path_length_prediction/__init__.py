"""
exp_path_length_prediction: Predict steps until terminal state.

Tests model's ability to predict path lengths for:
- Fixed path: Deterministic, same steps every time
- Variable path: Depends on guards and context values
- Unbounded: May not terminate (predict "infinite")

Uses REAL MLX inference (no mocking).
"""

from .length_predictor import (
    PathLengthExample,
    generate_fixed_path_examples,
    generate_variable_path_examples,
    generate_unbounded_examples,
    get_all_examples,
    compute_actual_path_length,
    create_path_length_prompt,
    parse_path_length_response,
    format_examples_for_few_shot,
)

from .benchmark import (
    run_benchmark,
    format_report,
    BenchmarkResult,
    PredictionResult,
)


def benchmark():
    """Run benchmark and return formatted report."""
    result = run_benchmark(samples_per_type=3)
    return format_report(result)


__all__ = [
    "run_benchmark",
    "format_report",
    "benchmark",
    "BenchmarkResult",
    "PredictionResult",
    "PathLengthExample",
    "generate_fixed_path_examples",
    "generate_variable_path_examples",
    "generate_unbounded_examples",
    "get_all_examples",
    "compute_actual_path_length",
    "create_path_length_prompt",
    "parse_path_length_response",
]
