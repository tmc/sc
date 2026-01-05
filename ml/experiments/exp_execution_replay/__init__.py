"""
Experiment: Execution Replay - Offline Learning from Traces

Learn statecharts from ExecutionTrace logs without online interaction.

Based on proto/statecharts/v1/execution.proto:
- ExecutionTrace: Complete execution history (σ₀, Γ₀, L*, σₙ, Γₙ)
- TransitionLogEntry: Single step (event, source→target config)
- Configuration: Set of active states

Approaches:
1. Direct Extraction: Count transitions, compute probabilities
2. State Inference: Cluster trace patterns to infer hidden states
3. Neural Fitting: Train network to predict transitions
4. Hybrid Learning: Mix online exploration with offline traces

Key insight: Offline learning is more sample-efficient for
statechart recovery because traces provide complete transition
information (source, event, target) without exploration overhead.

Components:
- TraceParser: Parse ExecutionTrace protos
- DirectExtractionLearner: Maximum likelihood from counts
- StateInferenceLearner: K-means clustering for hidden states
- NeuralOfflineLearner: Neural transition prediction
- OnlineOfflineComparison: Compare sample efficiency
- ExecutionReplayBenchmark: Comprehensive evaluation
"""

from .trace_parser import (
    # Proto-mirrored types
    Configuration,
    Event,
    TransitionRef,
    GuardEvaluation,
    ActionExecution,
    TransitionLogEntry,
    TraceMetadata,
    ExecutionTrace,
    # Utilities
    TraceParser,
    SyntheticTraceGenerator,
)

from .offline_learner import (
    # Learned representations
    LearnedTransition,
    LearnedState,
    LearnedStatechart,
    # Learners
    DirectExtractionLearner,
    StateInferenceLearner,
    GuardLearner,
    NeuralTransitionModel,
    NeuralOfflineLearner,
)

from .online_vs_offline import (
    # Environment
    StatechartEnvironment,
    # Learners
    OnlineLearner,
    HybridLearner,
    ReplayBuffer,
    # Comparison
    LearningCurvePoint,
    ComparisonResult,
    OnlineOfflineComparison,
)

from .benchmark import (
    BenchmarkConfig,
    MethodResult,
    BenchmarkResult,
    ExecutionReplayBenchmark,
    analyze_sample_efficiency,
)

__all__ = [
    # Trace parsing
    "Configuration",
    "Event",
    "TransitionRef",
    "GuardEvaluation",
    "ActionExecution",
    "TransitionLogEntry",
    "TraceMetadata",
    "ExecutionTrace",
    "TraceParser",
    "SyntheticTraceGenerator",
    # Offline learning
    "LearnedTransition",
    "LearnedState",
    "LearnedStatechart",
    "DirectExtractionLearner",
    "StateInferenceLearner",
    "GuardLearner",
    "NeuralTransitionModel",
    "NeuralOfflineLearner",
    # Online vs offline
    "StatechartEnvironment",
    "OnlineLearner",
    "HybridLearner",
    "ReplayBuffer",
    "LearningCurvePoint",
    "ComparisonResult",
    "OnlineOfflineComparison",
    # Benchmark
    "BenchmarkConfig",
    "MethodResult",
    "BenchmarkResult",
    "ExecutionReplayBenchmark",
    "analyze_sample_efficiency",
]
