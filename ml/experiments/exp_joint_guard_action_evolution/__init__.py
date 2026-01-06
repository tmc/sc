"""
exp_joint_guard_action_evolution

Co-evolve guards and actions together for coherent statecharts.
Guards depend on action effects, actions depend on guard conditions.

Key insight: Random guards and actions are incoherent. Joint evolution
with causal awareness produces statecharts where guards check variables
that actions set, creating meaningful causal chains.
"""

from .joint_genome import (
    JointGenome,
    GuardActionPair,
    Guard,
    GuardClause,
    GuardOp,
    GuardCombinator,
    Action,
    ActionEffect,
    EffectOp,
    random_joint_genome,
)

from .causal_graph import (
    CausalGraph,
    CausalEdge,
    CausalAnalysis,
    build_causal_graph,
    analyze_causal_graph,
    get_mutation_cluster,
    suggest_coherent_mutation,
)

from .coherence_fitness import (
    CoherenceFitness,
    Scenario,
    compute_coherence_fitness,
    execute_scenario,
)

from .joint_evolver import (
    JointEvolver,
    JointEvolverConfig,
    EvolutionResult,
    causal_aware_mutate,
    cluster_crossover,
)

from .benchmark import (
    BenchmarkResult,
    IndependentEvolver,
    run_benchmark,
    print_benchmark_results,
    random_baseline,
)

__all__ = [
    # Genome
    'JointGenome',
    'GuardActionPair',
    'Guard',
    'GuardClause',
    'GuardOp',
    'GuardCombinator',
    'Action',
    'ActionEffect',
    'EffectOp',
    'random_joint_genome',

    # Causal graph
    'CausalGraph',
    'CausalEdge',
    'CausalAnalysis',
    'build_causal_graph',
    'analyze_causal_graph',
    'get_mutation_cluster',
    'suggest_coherent_mutation',

    # Fitness
    'CoherenceFitness',
    'Scenario',
    'compute_coherence_fitness',
    'execute_scenario',

    # Evolver
    'JointEvolver',
    'JointEvolverConfig',
    'EvolutionResult',
    'causal_aware_mutate',
    'cluster_crossover',

    # Benchmark
    'BenchmarkResult',
    'IndependentEvolver',
    'run_benchmark',
    'print_benchmark_results',
    'random_baseline',
]
