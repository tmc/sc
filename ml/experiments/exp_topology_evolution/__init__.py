# Experiment: Topology Evolution
# Evolve statechart STRUCTURE through evolutionary algorithms
# C1: signals, C2: guards, C3: topology (this experiment)

"""
Flexible & Scalable Topology Evolution

This experiment discovers statechart structure through evolution:
- Works with any game (TicTacToe, Connect4, etc.)
- Scales to larger state spaces (10+ → 100+ states)
- Co-evolves guards alongside topology
- Supports parallel/distributed evolution

Quick Start:
    from experiments.exp_topology_evolution import TopologyEvolver, TicTacToeEnv

    evolver = TopologyEvolver(TicTacToeEnv)
    best = evolver.evolve(n_generations=100)
    print(f"Accuracy: {best.accuracy:.1%}")

Modules:
    environments - Game environment abstraction (TicTacToe, Connect4)
    evolve       - Genome encoding and evolution operators
    parallel     - Parallel evaluation with caching
    guards       - Evolvable guard expressions
    fitness      - Multi-objective fitness (NSGA-II)
    framework    - Unified evolution interface
"""

from .environments import (
    GameEnvironment,
    GameConfig,
    TicTacToeEnv,
    Connect4Env,
    OthelloEnv,
    Go9x9Env,
    ENVIRONMENTS,
)

from .framework import (
    TopologyEvolver,
    EvolutionConfig,
    SelectionMethod,
    run_evolution,
)

from .evolve import (
    StatechartGenome,
    create_genome_for_game,
)

from .guards import (
    GuardExpr,
    create_random_guard,
    GuardLibrary,
)

from .fitness import (
    FitnessComponents,
    MultiObjectiveFitness,
)

__all__ = [
    # Environments
    "GameEnvironment",
    "GameConfig",
    "TicTacToeEnv",
    "Connect4Env",
    "OthelloEnv",
    "Go9x9Env",
    "ENVIRONMENTS",
    # Framework
    "TopologyEvolver",
    "EvolutionConfig",
    "SelectionMethod",
    "run_evolution",
    # Genome
    "StatechartGenome",
    "create_genome_for_game",
    # Guards
    "GuardExpr",
    "create_random_guard",
    "GuardLibrary",
    # Fitness
    "FitnessComponents",
    "MultiObjectiveFitness",
]
