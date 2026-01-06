"""
exp_sae_feature_steering: SAE Feature Discovery and Steering for SC Generation

GOAL: Train Sparse Autoencoder on SC generation activations, extract
interpretable features for states/transitions/guards, and test feature-based steering.

Building on exp_grammar_induction patterns:
- TopK SAE for sparse feature extraction
- Feature-token associations
- Cluster-based state discovery

Key Insight:
    SC generation involves distinct semantic concepts:
    - STATE features: Fire when generating state definitions
    - TRANSITION features: Fire when generating transition specs
    - GUARD features: Fire when generating guard conditions
    - STRUCTURAL features: Fire for brackets, nesting, etc.

Architecture:
    SC Generator --> Activation Cache --> SAE Training
                                              |
                                              v
                                    Feature Discovery
                                              |
                                              v
                                    Steering Vectors

Usage:
    from ml.experiments.exp_sae_feature_steering import (
        ActivationCollector,
        SAETrainer,
        FeatureAnalyzer,
        SteeringTest,
    )

    # Collect activations
    collector = ActivationCollector()
    activations = collector.collect(prompts)

    # Train SAE
    trainer = SAETrainer()
    sae = trainer.train(activations)

    # Analyze features
    analyzer = FeatureAnalyzer(sae)
    features = analyzer.discover_sc_features()

    # Test steering
    tester = SteeringTest(sae, analyzer)
    results = tester.run()
"""

from .activation_collector import (
    ActivationCollector,
    ActivationCache,
    TokenActivation,
    CollectionConfig,
)

from .sae_trainer import (
    SAETrainer,
    TrainerConfig,
    TopKSAE,
    TrainingMetrics,
)

from .feature_analyzer import (
    FeatureAnalyzer,
    SCFeature,
    SCFeatureType,
    FeatureCluster,
    AnalysisResult,
)

from .steering_test import (
    SteeringTest,
    SteeringVector,
    SteeringConfig,
    SteeringResult,
    run_steering_benchmark,
)

__all__ = [
    # Activation collection
    'ActivationCollector',
    'ActivationCache',
    'TokenActivation',
    'CollectionConfig',
    # SAE training
    'SAETrainer',
    'TrainerConfig',
    'TopKSAE',
    'TrainingMetrics',
    # Feature analysis
    'FeatureAnalyzer',
    'SCFeature',
    'SCFeatureType',
    'FeatureCluster',
    'AnalysisResult',
    # Steering
    'SteeringTest',
    'SteeringVector',
    'SteeringConfig',
    'SteeringResult',
    'run_steering_benchmark',
]
