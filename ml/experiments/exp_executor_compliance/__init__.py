"""
exp_executor_compliance: Validate SCExecutor against Harel statechart semantics.

Tests:
1. Hierarchy: Enter composite -> cascade to initial substate
2. Orthogonality: AND-states with all regions active
3. Actions: Entry/exit action execution order
4. History: H and H* restoration

This is a BLOCKING prerequisite for other experiments.
"""

from .benchmark import run_compliance_benchmark

__all__ = ['run_compliance_benchmark']
