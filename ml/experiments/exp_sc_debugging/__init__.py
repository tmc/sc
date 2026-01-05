"""
exp_sc_debugging: Statechart Debugging with LLM Reasoning

Given a statechart and a failing execution trace, use LLM to identify
which transition is wrong or missing.

No activation steering - pure LLM reasoning task.

Usage:
    from ml.experiments.exp_sc_debugging import run_debug_benchmark

    results = run_debug_benchmark()
"""

from .test_cases import DebugTestCase, generate_test_cases
from .debugger import SCDebugger, DebugResult
from .benchmark import run_debug_benchmark, DebugBenchmarkResult

__all__ = [
    'DebugTestCase',
    'generate_test_cases',
    'SCDebugger',
    'DebugResult',
    'run_debug_benchmark',
    'DebugBenchmarkResult',
]
