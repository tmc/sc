"""
exp_structure_head_steering: Use discovered heads for targeted steering.

Uses attention heads discovered in exp_mlux_sc_attention:
- Structure heads: L11H13, L11H7, L9H7 (attend to structural tokens)
- Hierarchy head: L23H1 (tracks parent-child relationships)

Key Components:
- head_amplifier: Amplify/suppress specific attention heads
- targeted_steering: Apply steering using discovered heads
- benchmark: Test if amplifying L23H1 improves hierarchy validity

Goal: Demonstrate that amplifying L23H1 during generation
increases the rate of valid hierarchical statechart outputs.
"""

from .head_amplifier import (
    HeadConfig,
    AmplificationConfig,
    HeadAmplifier,
    STRUCTURE_HEADS,
    HIERARCHY_HEADS,
    KEYWORD_HEADS,
)

from .targeted_steering import (
    SteeringVector,
    TargetedSteeringConfig,
    SteeringResult,
    TargetedSteering,
    CONFIGS,
)

from .benchmark import (
    BenchmarkMetrics,
    BenchmarkResult,
    TEST_PROMPTS,
    SteeringBenchmark,
    demo,
    run_full_benchmark,
)

__all__ = [
    # head_amplifier
    "HeadConfig",
    "AmplificationConfig",
    "HeadAmplifier",
    "STRUCTURE_HEADS",
    "HIERARCHY_HEADS",
    "KEYWORD_HEADS",
    # targeted_steering
    "SteeringVector",
    "TargetedSteeringConfig",
    "SteeringResult",
    "TargetedSteering",
    "CONFIGS",
    # benchmark
    "BenchmarkMetrics",
    "BenchmarkResult",
    "TEST_PROMPTS",
    "SteeringBenchmark",
    "demo",
    "run_full_benchmark",
]
