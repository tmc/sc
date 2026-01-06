"""
exp_sc_equivalence: Behavioral Equivalence Checking

GOAL: Given two statecharts, determine if they are behaviorally
equivalent (accept the same traces).

Methods:
1. Trace comparison - enumerate and compare traces
2. Structural isomorphism - check if same structure
3. Bisimulation - formal equivalence relation

Uses trace enumeration and bisimulation checking.
"""

from .trace_generator import (
    TraceGenerator,
    Trace,
    TraceSet,
    generate_traces,
)
from .equivalence_checker import (
    EquivalenceChecker,
    EquivalenceConfig,
    EquivalenceMethod,
    EquivalenceResult,
    check_equivalence,
)
from .benchmark import (
    EquivalenceBenchmark,
    BenchmarkResult,
    run_benchmark,
)

__all__ = [
    # Trace Generator
    'TraceGenerator',
    'Trace',
    'TraceSet',
    'generate_traces',
    # Equivalence Checker
    'EquivalenceChecker',
    'EquivalenceConfig',
    'EquivalenceMethod',
    'EquivalenceResult',
    'check_equivalence',
    # Benchmark
    'EquivalenceBenchmark',
    'BenchmarkResult',
    'run_benchmark',
]
