"""
Context Prediction Experiment

Predict final context values after statechart execution.
Uses few-shot LLM prompting to predict context variable outcomes.

Complexity Levels:
- L1: Single increment (count++)
- L2: Arithmetic (score = count * 10)
- L3: Conditionals (if count > 5: bonus = 100)
- L4: Accumulation (sum over loop)
"""

from .context_predictor import (
    ContextPredictor,
    PredictionResult,
    predict_context,
)

from .benchmark import (
    ContextPredictionBenchmark,
    run_benchmark,
)

__all__ = [
    "ContextPredictor",
    "PredictionResult",
    "predict_context",
    "ContextPredictionBenchmark",
    "run_benchmark",
]
