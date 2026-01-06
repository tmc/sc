"""
exp_sc_to_code: Generate Python code from Statechart definitions.

Uses Qwen-1.5B with few-shot examples to generate runnable Python state machines.

Target:
- 60%+ runnable code
- 50%+ behaviorally correct
- 60%+ parallel region code correct (with template prompting)
"""

from .code_generator import SCCodeGenerator, generate_python_from_sc
from .code_validator import CodeValidator, validate_generated_code
from .benchmark import run_benchmark, SCCodeBenchmark
from .parallel_code_generator import (
    ParallelCodeGenerator,
    generate_parallel_code,
    detect_parallel_regions,
)
from .parallel_validator import (
    ParallelValidator,
    validate_parallel_code,
)
from .benchmark_parallel import (
    run_parallel_benchmark,
    PARALLEL_TEST_CASES,
)

__all__ = [
    # Flat SC generation
    'SCCodeGenerator',
    'generate_python_from_sc',
    'CodeValidator',
    'validate_generated_code',
    'run_benchmark',
    'SCCodeBenchmark',
    # Parallel SC generation
    'ParallelCodeGenerator',
    'generate_parallel_code',
    'detect_parallel_regions',
    'ParallelValidator',
    'validate_parallel_code',
    'run_parallel_benchmark',
    'PARALLEL_TEST_CASES',
]
