"""
exp_mlux_sc_steering: Validity Steering Vectors for Statecharts

GOAL: Improve statechart generation validity using steering vectors.

Pipeline:
1. Compute steering vectors from valid vs invalid SC pairs
2. Test steering effect on generation validity rate
3. Find optimal layer and alpha for SC steering

Targets:
- +10% validity improvement with steering

Key Components:
- steering_vectors.py: Compute steering vectors from contrastive pairs
- validity_tester.py: Test steering effect on generation
- benchmark.py: Grid search for optimal layer/alpha

Uses ml/utils/mlux_loader.py for model loading with mlux interpretability.

Key Insight: The difference in hidden states between valid and invalid
statechart generation captures a "validity direction" in activation space.
"""

# Steering vectors
from .steering_vectors import (
    # Data classes
    ContrastivePair,
    SteeringVector,
    # Computer
    SteeringVectorComputer,
    # Bank
    SteeringVectorBank,
    # Functions
    create_contrastive_pairs,
    # Test data
    VALID_STATECHARTS,
    INVALID_STATECHARTS,
)

# Validity tester
from .validity_tester import (
    # Data classes
    GenerationResult,
    ValidityTestResult,
    # Tester
    ValidityTester,
    # Validation
    validate_statechart,
    # Prompts
    SC_GENERATION_PROMPTS,
    SC_GENERATION_TEMPLATE,
)

# Benchmark
from .benchmark import (
    # Config
    BenchmarkConfig,
    BenchmarkResult,
    # Benchmark class
    SteeringBenchmark,
    # Functions
    run_full_benchmark,
)

__all__ = [
    # Steering vectors
    'ContrastivePair',
    'SteeringVector',
    'SteeringVectorComputer',
    'SteeringVectorBank',
    'create_contrastive_pairs',
    'VALID_STATECHARTS',
    'INVALID_STATECHARTS',
    # Validity tester
    'GenerationResult',
    'ValidityTestResult',
    'ValidityTester',
    'validate_statechart',
    'SC_GENERATION_PROMPTS',
    'SC_GENERATION_TEMPLATE',
    # Benchmark
    'BenchmarkConfig',
    'BenchmarkResult',
    'SteeringBenchmark',
    'run_full_benchmark',
]
