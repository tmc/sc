"""
exp_counterfactual_sc: Predict Next State + Explain Why

GOAL: Given statechart + current state + event, predict next state
and generate natural language explanation of why.

Tests execution understanding - can the model predict state
transitions correctly based on statechart semantics?

Uses REAL Qwen2.5-Coder-0.5B-Instruct inference via mlx_lm.
"""

from .predictor import (
    CounterfactualPredictor,
    PredictionConfig,
    Prediction,
    predict_next_state,
)
from .explainer import (
    TransitionExplainer,
    ExplainerConfig,
    ExplanationStyle,
    TransitionExplanation,
    explain_transition,
)
from .benchmark import (
    CounterfactualBenchmark,
    BenchmarkResult,
    run_benchmark,
)

__all__ = [
    # Predictor
    'CounterfactualPredictor',
    'PredictionConfig',
    'Prediction',
    'predict_next_state',
    # Explainer
    'TransitionExplainer',
    'ExplainerConfig',
    'ExplanationStyle',
    'TransitionExplanation',
    'explain_transition',
    # Benchmark
    'CounterfactualBenchmark',
    'BenchmarkResult',
    'run_benchmark',
]
