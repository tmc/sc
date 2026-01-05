"""
Experiment: Deep vs Shallow History Evolution

GOAL: Learn when to use deep history (H*) vs shallow history (H)
through evolution - NO hardcoding.

KEY INSIGHT from Harel's statecharts:
- SHALLOW HISTORY (H): Restores only the DIRECT child of the composite state
- DEEP HISTORY (H*): Restores the ENTIRE nested configuration

Example:
  Composite
  ├── A (initial)
  │   ├── A1 (initial)
  │   └── A2
  └── B

  If we were in A.A2, then went to B, then back via history:
  - Shallow (H): Returns to A (initial = A.A1)
  - Deep (H*):   Returns to A.A2 (full restoration)

APPROACH:
1. Extend genome with HistoryType gene per state: NONE | SHALLOW | DEEP
2. Create environments where:
   - Nested state context matters → evolve DEEP history
   - Only direct child matters → evolve SHALLOW history
   - No history needed → evolve NONE
3. Fitness = accuracy on scenarios requiring correct history restoration

REFERENCE: semantics/v1/machine.go:resolveHistory
- Deep history returns all stored states
- Shallow history filters to direct children only

NO HARDCODING: The algorithm discovers when each type is beneficial.
"""

from .history_evolver import (
    HistoryType,
    HistoryGenome,
    HistoryEvolver,
    HistoryMachine,
    HistoryEvolverConfig,
    HistoryEvolutionResult,
    evolve_history_strategy,
    create_random_history_genome,
)

from .scenarios import (
    HistoryScenario,
    NestedNavigationScenario,
    TextEditorScenario,
    GamePauseScenario,
    SCENARIOS,
    create_scenario_dataset,
    create_mixed_scenario_generator,
)

from .benchmark import (
    compare_history_strategies,
    HistoryBenchmarkResult,
    BenchmarkSummary,
    quick_benchmark,
)

__all__ = [
    # Core types
    'HistoryType',
    'HistoryGenome',
    'HistoryMachine',
    # Evolution
    'HistoryEvolver',
    'HistoryEvolverConfig',
    'HistoryEvolutionResult',
    'evolve_history_strategy',
    'create_random_history_genome',
    # Scenarios
    'HistoryScenario',
    'NestedNavigationScenario',
    'TextEditorScenario',
    'GamePauseScenario',
    'SCENARIOS',
    'create_scenario_dataset',
    'create_mixed_scenario_generator',
    # Benchmark
    'compare_history_strategies',
    'HistoryBenchmarkResult',
    'BenchmarkSummary',
    'quick_benchmark',
]
