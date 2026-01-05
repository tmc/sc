"""
exp_grammar_internalization: Train model to internalize SC grammar.

Problem: Without grammar constraints = ~0% validity. With constraints = 100%.
Goal: Fine-tune model to generate valid SC without runtime constraints.

Approach:
1. Generate training data via constrained sampling (valid SC examples)
2. Fine-tune Qwen-1.5B with LoRA on (prompt, valid_sc) pairs
3. Test unconstrained generation before/after

Target: 80%+ unconstrained validity after training.
"""

from .data_generator import generate_training_data
from .validity_tester import test_validity
from .benchmark import run_benchmark

__all__ = ["generate_training_data", "test_validity", "run_benchmark"]
