"""
exp_soar_statechart: SOAR-Inspired Statechart Program Synthesis

Integrates SOAR (Self-improving Operators for Automated program Refinement)
concepts with differentiable statecharts for ARC-style reasoning tasks.

Key Ideas from SOAR:
1. Evolutionary refinement (REX algorithm) for iterative improvement
2. Hindsight relabeling - failed solutions as correct for synthetic tasks
3. Large synthetic solution archives for training
4. Learned refinement operators that improve with experience

Our Integration:
1. Statechart-Constrained Synthesis - Programs as state machine traversals
2. REX for Topology Evolution - Hybrid gradient + evolutionary refinement
3. Hindsight State Relabeling - Failed trajectories create new transition data
4. Structural Priors - Statechart constraints reduce search space

Hypothesis: Statechart structure provides inductive bias that improves
sample efficiency over unconstrained program synthesis (SOAR baseline).

Components:
- SOARStatechart: Statechart with REX-compatible mutation operators
- HindsightRelabeler: Convert failed executions to training data
- ARCStatechartSynthesizer: End-to-end synthesis for ARC tasks
- REXEvolver: Evolutionary refinement with gradient-based guard learning
"""

from .soar_statechart import (
    SOARStatechart,
    StatechartProgram,
    ARCStatechartSynthesizer,
)

from .rex_refinement import (
    REXEvolver,
    RefinementOperator,
    MutationArchive,
)

from .hindsight_relabel import (
    HindsightRelabeler,
    StateTrajectory,
    SyntheticTask,
)

from .arc_evaluation import (
    ARCEvaluator,
    ARCDataset,
    ARCTask,
    BenchmarkResult,
)

__all__ = [
    "SOARStatechart",
    "StatechartProgram",
    "ARCStatechartSynthesizer",
    "REXEvolver",
    "RefinementOperator",
    "MutationArchive",
    "HindsightRelabeler",
    "StateTrajectory",
    "SyntheticTask",
    "ARCEvaluator",
    "ARCDataset",
    "ARCTask",
    "BenchmarkResult",
]
