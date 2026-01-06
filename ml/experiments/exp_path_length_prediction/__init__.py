"""
exp_path_length_prediction: Predict steps until terminal state.

Tests model's ability to predict path lengths for:
- Fixed path: Deterministic, same steps every time
- Variable path: Depends on guards and context values
- Unbounded: May not terminate (predict "infinite")

Approaches:
- Baseline: Direct prediction (12% baseline)
- Scratchpad: Explicit step-by-step counting
- BFS simulation: Teach model to simulate BFS
- ASCII visualization: Graph structure in prompt

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

from .scratchpad_predictor import (
    ScratchpadPredictor,
    PromptStyle,
    StatechartPath,
    create_scratchpad_prompt,
    create_scratchpad_few_shot,
    create_bfs_simulation_prompt,
    parse_path_length,
    get_test_cases,
)

from .path_visualizer import (
    PathVisualizer,
    create_ascii_graph,
    create_transition_list,
    create_distance_table,
    create_path_trace,
)

from .benchmark_scratchpad import (
    run_benchmark as run_scratchpad_benchmark,
    format_report as format_scratchpad_report,
    BenchmarkConfig as ScratchpadConfig,
)


def benchmark():
    """Run benchmark and return formatted report."""
    result = run_benchmark(samples_per_type=3)
    return format_report(result)


__all__ = [
    # Original
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
    # Scratchpad
    "ScratchpadPredictor",
    "PromptStyle",
    "StatechartPath",
    "create_scratchpad_prompt",
    "create_scratchpad_few_shot",
    "create_bfs_simulation_prompt",
    "parse_path_length",
    "get_test_cases",
    # Visualizer
    "PathVisualizer",
    "create_ascii_graph",
    "create_transition_list",
    "create_distance_table",
    "create_path_trace",
    # Scratchpad benchmark
    "run_scratchpad_benchmark",
    "format_scratchpad_report",
    "ScratchpadConfig",
]
