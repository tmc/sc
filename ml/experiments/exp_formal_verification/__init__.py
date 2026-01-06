"""
exp_formal_verification: SMT-Based Statechart Verification

Bridge between ML-evolved statecharts and formal methods.

Key Properties Verified:
1. No illegal transitions - guards prevent invalid state changes
2. Deadlock freedom - always exists an enabled transition (or final state)
3. Liveness - eventually reach target states
4. Determinism - at most one transition enabled at a time
5. Coverage - all states reachable from initial

Uses Z3 SMT solver for verification. Integrates with evolution
to reject invalid statecharts during fitness evaluation.

Key insight: Formal verification as fitness component ensures
evolved statecharts are not just accurate but CORRECT.
"""

from .smt_encoder import (
    SMTEncoder,
    StatechartFormula,
    StateVariable,
    TransitionFormula,
)
from .property_checker import (
    PropertyChecker,
    Property,
    PropertyResult,
    IllegalTransitionProperty,
    DeadlockFreedomProperty,
    LivenessProperty,
    DeterminismProperty,
    ReachabilityProperty,
)
from .verified_evolution import (
    VerifiedEvolver,
    VerificationFitness,
    VerifiedGenome,
)
from .experiment import run_experiment

__all__ = [
    'SMTEncoder',
    'StatechartFormula',
    'StateVariable',
    'TransitionFormula',
    'PropertyChecker',
    'Property',
    'PropertyResult',
    'IllegalTransitionProperty',
    'DeadlockFreedomProperty',
    'LivenessProperty',
    'DeterminismProperty',
    'ReachabilityProperty',
    'VerifiedEvolver',
    'VerificationFitness',
    'VerifiedGenome',
    'run_experiment',
]
