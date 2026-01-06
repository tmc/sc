"""
exp_lca_neural: Neural Network LCA Prediction

Train neural networks to predict LCA (Lowest Common Ancestor) from
(source, target) state pairs in hierarchical statecharts.

Key insight: LCA computation is critical for inter-level transitions.
Learning to predict LCA enables faster transition resolution.

Approaches:
1. Supervised: Train on (source, target) -> LCA with cross-entropy
2. Evolutionary: Evolve lookup tables for LCA prediction
3. Hybrid: Pre-train supervised, fine-tune with evolution

Tests:
- In-distribution accuracy
- Generalization to unseen hierarchies
- Sample efficiency

Usage:
    from experiments.exp_lca_neural import (
        LCABenchmark,
        SupervisedLCATrainer,
        EvolutionaryLCAPredictor,
        HybridLCAPredictor,
    )

    # Quick benchmark
    benchmark = LCABenchmark()
    results = benchmark.run()

    # Train specific model
    trainer = SupervisedLCATrainer(LCAModelConfig())
    trainer.train(examples, hierarchies)
    prediction = trainer.predict(source_id, target_id)
"""

from .lca_dataset import (
    LCAExample,
    Hierarchy,
    HierarchyNode,
    StateType,
    HierarchyGenerator,
    LCADatasetGenerator,
)

from .neural_lca import (
    LCAModelConfig,
    StateEncoder,
    SupervisedLCATrainer,
    EvolutionaryLCAPredictor,
    HybridLCAPredictor,
)

from .lca_benchmark import (
    LCABenchmark,
    BenchmarkResult,
    RandomBaseline,
    RootBaseline,
)

__all__ = [
    # Dataset
    'LCAExample',
    'Hierarchy',
    'HierarchyNode',
    'StateType',
    'HierarchyGenerator',
    'LCADatasetGenerator',
    # Neural models
    'LCAModelConfig',
    'StateEncoder',
    'SupervisedLCATrainer',
    'EvolutionaryLCAPredictor',
    'HybridLCAPredictor',
    # Benchmark
    'LCABenchmark',
    'BenchmarkResult',
    'RandomBaseline',
    'RootBaseline',
]
