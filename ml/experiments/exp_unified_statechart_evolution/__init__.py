"""
Experiment: Unified Statechart Evolution - FLAGSHIP

THE ENDGAME: Co-evolve ALL statechart components in a single UnifiedGenome.

This experiment unifies:
1. TOPOLOGY: States, hierarchy, parent relationships (from exp_topology_evolution)
2. HISTORY: NONE/SHALLOW/DEEP per state (from exp_deep_history)
3. GUARDS: Conditions on transitions with specificity (from exp_transition_priorities)
4. PRIORITIES: Conflict resolution when multiple transitions enabled
5. ACTIONS: Side effects that modify context variables (from exp_action_side_effects)
6. EVENTS: Trigger labels for transitions

MULTI-OBJECTIVE OPTIMIZATION (NSGA-II):
- CORRECTNESS: Accuracy on test scenarios
- MINIMALITY: Fewer states, transitions, actions
- INTERPRETABILITY: Human-readable structure, clear naming

KEY INSIGHT:
All these components interact. A change in topology affects which guards
are needed. Adding history changes state reachability. Actions enable
guards. Priorities resolve conflicts. They must be co-evolved together.

NO HARDCODING:
The evolution discovers the optimal combination of all components
through multi-objective optimization. The Pareto front contains
diverse solutions trading off correctness, minimality, interpretability.

REFERENCE:
- semantics/v1/machine.go - Execution semantics
- proto/statecharts/v1/statecharts.proto - Full statechart model
"""

from .unified_genome import (
    UnifiedGenome,
    UnifiedTransition,
    UnifiedGuard,
    UnifiedAction,
    ActionType,
    HistoryType,
    StateType,
)

from .unified_evolver import (
    UnifiedEvolver,
    UnifiedEvolverConfig,
    UnifiedEvolutionResult,
)

from .unified_fitness import (
    UnifiedFitnessComponents,
    compute_unified_fitness,
    nsga2_select,
)

from .unified_executor import (
    UnifiedMachine,
    UnifiedContext,
)

__all__ = [
    # Genome
    'UnifiedGenome',
    'UnifiedTransition',
    'UnifiedGuard',
    'UnifiedAction',
    'ActionType',
    'HistoryType',
    'StateType',
    # Evolution
    'UnifiedEvolver',
    'UnifiedEvolverConfig',
    'UnifiedEvolutionResult',
    # Fitness
    'UnifiedFitnessComponents',
    'compute_unified_fitness',
    'nsga2_select',
    # Execution
    'UnifiedMachine',
    'UnifiedContext',
]
