"""
exp_3b_model_scaling: Verify 3B model improvements on semantic SC tasks.

Phase 6 experiment comparing model size impact on:
- SC Repair (error fixing)
- SC Debugging (bug identification)
- SC Completion (constrained generation)

Hypothesis: 3B model achieves 90%+ repair, 50%+ debugging.
"""

from .scaling_benchmark import run_full_scaling_benchmark, ModelSize

__all__ = ["run_full_scaling_benchmark", "ModelSize"]
