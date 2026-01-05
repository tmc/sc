"""
exp_coverage_prediction: Statechart-Based Code Coverage Prediction

Train models to predict which lines will be covered given (program, input).
Compare statechart-based approaches vs neural baselines.

Hypothesis: Programs as statecharts outperform flat representations because
coverage prediction = "which states will be visited?" = statechart simulation.

Key Approaches:
1. Statechart Traversal - Convert CFG to statechart, simulate execution
2. Guard-Based Prediction - Learn branch conditions, evaluate on input
3. SAE Coverage Features - Discover features that predict coverage

Baselines:
- Sequence models (program + input → coverage)
- AST-based embeddings
- GNN on control flow graph

Usage:
    from ml.experiments.exp_coverage_prediction import (
        CoverageCollector,
        build_cfg,
        StatechartCoveragePredictor,
        CoverageBenchmark,
    )

    # Collect coverage data
    collector = CoverageCollector()
    data = collector.collect_from_source(source, 'func_name', [input1, input2])

    # Predict coverage with statechart
    predictor = StatechartCoveragePredictor(StatechartConfig())
    covered, debug = predictor.predict_coverage(source, input_value)

    # Run benchmark
    benchmark = CoverageBenchmark()
    results = benchmark.run_full_benchmark()
"""

from .coverage_collector import (
    CoverageCollector,
    CoverageTriple,
    SyntheticDataGenerator,
)

from .program_statechart import (
    ProgramStatechart,
    BasicBlock,
    CFGTransition,
    BlockType,
    build_cfg,
    build_cfg_for_function,
)

from .statechart_predictor import (
    StatechartCoveragePredictor,
    StatechartConfig,
    SymbolicExecutor,
    AbstractValue,
    DifferentiableStatechartPredictor,
)

from .baseline_models import (
    SequenceCoverageModel,
    ASTCoverageModel,
    GNNCoverageModel,
    CoverageModelConfig,
    SimpleTokenizer,
    coverage_loss,
)

from .benchmark import (
    CoverageBenchmark,
    BenchmarkResult,
    EvaluationMetrics,
    compute_metrics,
    aggregate_metrics,
)

from .dataset import (
    Dataset,
    DatasetExample,
    DatasetGenerator,
    ProgramCategory,
    ProgramTemplate,
    ProgramAnalyzer,
)

from .state_discovery import (
    StateDiscoverer,
    DiscoveredState,
    StateType,
    LineClusterDiscovery,
    VariablePatternDiscovery,
    ControlFlowPhaseDiscovery,
)

from .transition_learner import (
    TransitionLearner,
    LearnedTransition,
    LearnedGuard,
    LearnedAction,
    GuardSynthesizer,
)

from .statechart_evolver import (
    StatechartEvolver,
    EvolutionConfig,
    CoverageStatechartGenome,
    EvolvedState,
    EvolvedTransition,
    GenomeFactory,
    CoverageEvaluator,
    compare_evolved_vs_hardcoded,
)

from .discovery_benchmark import (
    DiscoveryBenchmark,
    DiscoveryResult,
    compute_state_coherence,
)

__all__ = [
    # Coverage collection
    'CoverageCollector',
    'CoverageTriple',
    'SyntheticDataGenerator',
    # Statechart conversion
    'ProgramStatechart',
    'BasicBlock',
    'CFGTransition',
    'BlockType',
    'build_cfg',
    'build_cfg_for_function',
    # Statechart prediction
    'StatechartCoveragePredictor',
    'StatechartConfig',
    'SymbolicExecutor',
    'AbstractValue',
    'DifferentiableStatechartPredictor',
    # Baseline models
    'SequenceCoverageModel',
    'ASTCoverageModel',
    'GNNCoverageModel',
    'CoverageModelConfig',
    'SimpleTokenizer',
    'coverage_loss',
    # Benchmark
    'CoverageBenchmark',
    'BenchmarkResult',
    'EvaluationMetrics',
    'compute_metrics',
    'aggregate_metrics',
    # Dataset
    'Dataset',
    'DatasetExample',
    'DatasetGenerator',
    'ProgramCategory',
    'ProgramTemplate',
    'ProgramAnalyzer',
    # State discovery
    'StateDiscoverer',
    'DiscoveredState',
    'StateType',
    'LineClusterDiscovery',
    'VariablePatternDiscovery',
    'ControlFlowPhaseDiscovery',
    # Transition learning
    'TransitionLearner',
    'LearnedTransition',
    'LearnedGuard',
    'LearnedAction',
    'GuardSynthesizer',
    # Statechart evolution
    'StatechartEvolver',
    'EvolutionConfig',
    'CoverageStatechartGenome',
    'EvolvedState',
    'EvolvedTransition',
    'GenomeFactory',
    'CoverageEvaluator',
    'compare_evolved_vs_hardcoded',
    # Discovery benchmark
    'DiscoveryBenchmark',
    'DiscoveryResult',
    'compute_state_coherence',
]
