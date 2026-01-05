"""
exp_sc_to_code: Generate Python code from Statechart definitions.

Uses Qwen-1.5B with few-shot examples to generate runnable Python state machines.

Target:
- 60%+ runnable code
- 50%+ behaviorally correct
"""

from .code_generator import SCCodeGenerator, generate_python_from_sc
from .code_validator import CodeValidator, validate_generated_code
from .benchmark import run_benchmark, SCCodeBenchmark

__all__ = [
    'SCCodeGenerator',
    'generate_python_from_sc',
    'CodeValidator',
    'validate_generated_code',
    'run_benchmark',
    'SCCodeBenchmark',
]
