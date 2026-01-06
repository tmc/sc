"""
exp_model_scaling: Model Size Scaling Experiments

Tests if steering improvements scale with model size.

Models tested:
- Qwen2.5-Coder-0.5B-Instruct-4bit (baseline)
- Qwen2.5-Coder-1.5B-Instruct-4bit
- Qwen2.5-Coder-3B-Instruct-4bit (future)

Configurations:
- baseline: No steering
- hierarchy_boost: L23H1 x1.5
- structure_boost: L11H13, L11H7, L9H7 x1.5
- full_steering: Both x1.5

Hypothesis: Larger model = higher baseline + larger steering delta
"""

from .scaling_1_5b import (
    MODEL_NAME,
    STEERING_CONFIGS,
    TEST_PROMPTS,
    GenerationResult,
    ConfigMetrics,
    parse_statechart,
    validate_statechart,
    count_states,
    get_max_depth,
    has_nested_states,
    ScalingExperiment,
    run_scaling_test,
    demo,
)

__all__ = [
    "MODEL_NAME",
    "STEERING_CONFIGS",
    "TEST_PROMPTS",
    "GenerationResult",
    "ConfigMetrics",
    "parse_statechart",
    "validate_statechart",
    "count_states",
    "get_max_depth",
    "has_nested_states",
    "ScalingExperiment",
    "run_scaling_test",
    "demo",
]
