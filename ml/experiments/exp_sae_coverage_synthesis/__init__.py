"""
exp_sae_coverage_synthesis: Unified SAE→States→Guards→Coverage Pipeline

Key insight: SAE features ARE states. No clustering needed.

Pipeline:
1. SAE Encoding: Hidden states → Sparse features (TopK active = state config)
2. Guard Synthesis: Feature activation patterns → Boolean guards
3. Coverage Prediction: Simulate statechart → Predict which states visited

This unifies:
- exp_sae_statechart: SAE as state bottleneck
- exp_coverage_prediction: Coverage via statechart simulation
- exp_guard_synthesis: Learn guards from examples

Single end-to-end differentiable pipeline.
"""

from .sae_state_extractor import (
    SAEStateExtractor,
    FeatureState,
    StateConfiguration,
)
from .guard_synthesizer import (
    FeatureGuard,
    GuardSynthesizer,
    ActivationPattern,
)
from .coverage_simulator import (
    CoverageSimulator,
    SimulationResult,
    CoveragePrediction,
)
from .pipeline import (
    SAECoveragePipeline,
    PipelineConfig,
    PipelineResult,
)
from .experiment import run_experiment

__all__ = [
    'SAEStateExtractor',
    'FeatureState',
    'StateConfiguration',
    'FeatureGuard',
    'GuardSynthesizer',
    'ActivationPattern',
    'CoverageSimulator',
    'SimulationResult',
    'CoveragePrediction',
    'SAECoveragePipeline',
    'PipelineConfig',
    'PipelineResult',
    'run_experiment',
]
