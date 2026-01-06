"""
exp_mlux_baseline: Demonstrate mlux integration patterns for statecharts.

Goal: Provide baseline patterns for using mlux (or mlx_lm fallback)
with statechart generation, including activation caching and steering.

Key Components:
- baseline_demo: HookedModelWrapper usage with activation caching
- attention_viz: Extract and visualize attention patterns
- benchmark: Performance comparison across backends

Features Demonstrated:
1. Unified model loading via HookedModelWrapper
2. Generation with activation caching (mlux only)
3. Attention pattern extraction and analysis
4. Steering vector computation and application
5. Graceful fallback when mlux unavailable

Backend Availability:
- MLUX: Full interpretability features
- MLX_LM: Basic generation only
- MOCK: Testing without GPU
"""

from .baseline_demo import (
    StatechartPrompt,
    GenerationResult,
    StatechartGenerator,
    demo_basic_generation,
    demo_with_cache,
    demo_steering,
    demo_fallback_behavior,
    run_all_demos as run_baseline_demos,
)

from .attention_viz import (
    AttentionStats,
    TokenAttention,
    AttentionAnalysis,
    AttentionAnalyzer,
    visualize_attention_heatmap,
    demo_attention_analysis,
    demo_token_relationships,
    run_all_demos as run_attention_demos,
)

from .benchmark import (
    BenchmarkResult,
    BenchmarkSuite,
    MLUXBenchmark,
    BENCHMARK_PROMPTS,
    demo as run_benchmark_demo,
    run_full_benchmark,
)

__all__ = [
    # baseline_demo
    "StatechartPrompt",
    "GenerationResult",
    "StatechartGenerator",
    "demo_basic_generation",
    "demo_with_cache",
    "demo_steering",
    "demo_fallback_behavior",
    "run_baseline_demos",
    # attention_viz
    "AttentionStats",
    "TokenAttention",
    "AttentionAnalysis",
    "AttentionAnalyzer",
    "visualize_attention_heatmap",
    "demo_attention_analysis",
    "demo_token_relationships",
    "run_attention_demos",
    # benchmark
    "BenchmarkResult",
    "BenchmarkSuite",
    "MLUXBenchmark",
    "BENCHMARK_PROMPTS",
    "run_benchmark_demo",
    "run_full_benchmark",
]
