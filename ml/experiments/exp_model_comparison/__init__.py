"""
Model Comparison Experiment

Unified benchmark for comparing Qwen-Coder model sizes on SC generation:
- 0.5B: Qwen2.5-Coder-0.5B-Instruct
- 1.5B: Qwen2.5-Coder-1.5B-Instruct
- 3B: Qwen2.5-Coder-3B-Instruct
- 7B: Qwen2.5-Coder-7B-Instruct

Usage:
    python benchmark.py --all-models
    python benchmark.py --model 0.5B
    python benchmark.py --compare 0.5B 3B
"""

from .benchmark import (
    ModelBenchmark,
    ComparisonRunner,
    run_all_models,
    run_comparison,
    generate_report,
)

from .model_configs import (
    MODEL_CONFIGS,
    get_model_config,
    ModelConfig,
)

from .metrics import (
    SCMetrics,
    compute_metrics,
    aggregate_results,
)

__all__ = [
    # Benchmark
    "ModelBenchmark",
    "ComparisonRunner",
    "run_all_models",
    "run_comparison",
    "generate_report",
    # Configs
    "MODEL_CONFIGS",
    "get_model_config",
    "ModelConfig",
    # Metrics
    "SCMetrics",
    "compute_metrics",
    "aggregate_results",
]
