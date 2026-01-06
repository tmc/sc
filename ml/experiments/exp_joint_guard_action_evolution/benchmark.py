"""
Benchmark: Joint vs Independent Guard-Action Evolution

Compares:
1. JOINT EVOLUTION: Co-evolve guards and actions with causal awareness
2. INDEPENDENT EVOLUTION: Evolve guards and actions separately
3. RANDOM BASELINE: Random guards and actions

Hypothesis: Joint evolution produces more coherent statecharts because
causal dependencies are maintained during evolution.
"""

import random
import time
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import copy

from .joint_genome import (
    JointGenome, GuardActionPair, Guard, Action,
    random_joint_genome
)
from .causal_graph import build_causal_graph, analyze_causal_graph
from .coherence_fitness import (
    CoherenceFitness, compute_coherence_fitness, Scenario
)
from .joint_evolver import (
    JointEvolver, JointEvolverConfig, EvolutionResult,
    mutate_guard, mutate_action
)


# =============================================================================
# Independent Evolution (Baseline)
# =============================================================================

class IndependentEvolver:
    """Evolves guards and actions independently (no causal awareness)."""

    def __init__(self, config: JointEvolverConfig):
        self.config = config
        self.population: List[JointGenome] = []

    def evolve_guards_only(self, genome: JointGenome) -> JointGenome:
        """Mutate only guards."""
        new_genome = genome.copy()
        for i in range(len(new_genome.pairs)):
            if random.random() < self.config.mutation_rate:
                new_genome.pairs[i].guard = mutate_guard(
                    new_genome.pairs[i].guard,
                    self.config.variables,
                    self.config.possible_values
                )
        return new_genome

    def evolve_actions_only(self, genome: JointGenome) -> JointGenome:
        """Mutate only actions."""
        new_genome = genome.copy()
        for i in range(len(new_genome.pairs)):
            if random.random() < self.config.mutation_rate:
                new_genome.pairs[i].action = mutate_action(
                    new_genome.pairs[i].action,
                    self.config.variables,
                    self.config.possible_values
                )
        return new_genome

    def run(
        self,
        scenarios: List[Scenario] = None,
        verbose: bool = False
    ) -> EvolutionResult:
        """Run independent evolution.

        Alternates between evolving guards and actions.
        """
        # Initialize
        self.population = [
            random_joint_genome(
                n_states=self.config.n_states,
                n_events=self.config.n_events,
                n_pairs=self.config.n_pairs,
                variables=self.config.variables,
                possible_values=self.config.possible_values
            )
            for _ in range(self.config.population_size)
        ]

        fitness_history = []
        generation_stats = []

        for gen in range(self.config.n_generations):
            # Evaluate
            fitness_list = [compute_coherence_fitness(g, scenarios)
                            for g in self.population]

            # Stats
            best_idx = max(range(len(fitness_list)),
                           key=lambda i: fitness_list[i].total_fitness())
            best_fitness = fitness_list[best_idx]
            avg_fitness = sum(f.total_fitness() for f in fitness_list) / len(fitness_list)

            fitness_history.append(best_fitness.total_fitness())
            generation_stats.append({
                'best': best_fitness.total_fitness(),
                'avg': avg_fitness,
            })

            if verbose and gen % 10 == 0:
                print(f"Gen {gen:3d} [IND]: best={best_fitness.total_fitness():.3f} "
                      f"avg={avg_fitness:.3f}")

            # Selection (tournament)
            new_population = []
            for _ in range(self.config.population_size):
                candidates = random.sample(range(len(self.population)), 3)
                winner = max(candidates, key=lambda i: fitness_list[i].total_fitness())
                new_population.append(self.population[winner].copy())

            # Alternate: guards one generation, actions the next
            if gen % 2 == 0:
                self.population = [self.evolve_guards_only(g) for g in new_population]
            else:
                self.population = [self.evolve_actions_only(g) for g in new_population]

        # Final evaluation
        fitness_list = [compute_coherence_fitness(g, scenarios)
                        for g in self.population]
        best_idx = max(range(len(fitness_list)),
                       key=lambda i: fitness_list[i].total_fitness())

        return EvolutionResult(
            best_genome=self.population[best_idx],
            best_fitness=fitness_list[best_idx],
            fitness_history=fitness_history,
            generation_stats=generation_stats
        )


# =============================================================================
# Random Baseline
# =============================================================================

def random_baseline(
    config: JointEvolverConfig,
    scenarios: List[Scenario] = None,
    n_samples: int = 100
) -> EvolutionResult:
    """Generate random genomes and return best."""
    best_genome = None
    best_fitness = None

    fitness_history = []

    for i in range(n_samples):
        genome = random_joint_genome(
            n_states=config.n_states,
            n_events=config.n_events,
            n_pairs=config.n_pairs,
            variables=config.variables,
            possible_values=config.possible_values
        )
        fitness = compute_coherence_fitness(genome, scenarios)

        if best_fitness is None or fitness.total_fitness() > best_fitness.total_fitness():
            best_genome = genome
            best_fitness = fitness

        fitness_history.append(fitness.total_fitness())

    return EvolutionResult(
        best_genome=best_genome,
        best_fitness=best_fitness,
        fitness_history=fitness_history,
        generation_stats=[]
    )


# =============================================================================
# Benchmark Suite
# =============================================================================

@dataclass
class BenchmarkResult:
    """Results from benchmark comparison."""
    joint_result: EvolutionResult
    independent_result: EvolutionResult
    random_result: EvolutionResult

    joint_time: float
    independent_time: float
    random_time: float

    def joint_vs_independent(self) -> float:
        """Improvement of joint over independent."""
        ind = self.independent_result.best_fitness.total_fitness()
        jnt = self.joint_result.best_fitness.total_fitness()
        if ind == 0:
            return float('inf') if jnt > 0 else 0
        return (jnt - ind) / ind

    def joint_vs_random(self) -> float:
        """Improvement of joint over random."""
        rnd = self.random_result.best_fitness.total_fitness()
        jnt = self.joint_result.best_fitness.total_fitness()
        if rnd == 0:
            return float('inf') if jnt > 0 else 0
        return (jnt - rnd) / rnd


def run_benchmark(
    config: JointEvolverConfig,
    scenarios: List[Scenario] = None,
    verbose: bool = True
) -> BenchmarkResult:
    """Run full benchmark comparison."""

    if verbose:
        print("=" * 60)
        print("BENCHMARK: Joint vs Independent vs Random")
        print("=" * 60)

    # Joint evolution
    if verbose:
        print("\n--- Joint Evolution ---")
    start = time.time()
    joint_evolver = JointEvolver(config)
    joint_result = joint_evolver.run(scenarios, verbose=verbose)
    joint_time = time.time() - start

    # Independent evolution
    if verbose:
        print("\n--- Independent Evolution ---")
    start = time.time()
    independent_evolver = IndependentEvolver(config)
    independent_result = independent_evolver.run(scenarios, verbose=verbose)
    independent_time = time.time() - start

    # Random baseline
    if verbose:
        print("\n--- Random Baseline ---")
    start = time.time()
    random_result = random_baseline(config, scenarios,
                                     n_samples=config.population_size * config.n_generations)
    random_time = time.time() - start

    return BenchmarkResult(
        joint_result=joint_result,
        independent_result=independent_result,
        random_result=random_result,
        joint_time=joint_time,
        independent_time=independent_time,
        random_time=random_time
    )


def print_benchmark_results(result: BenchmarkResult):
    """Print benchmark results."""
    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)

    print(f"\n{'Method':<15} {'Fitness':>10} {'Causal':>10} {'Behav':>10} {'Time':>10}")
    print("-" * 60)

    jf = result.joint_result.best_fitness
    print(f"{'Joint':<15} {jf.total_fitness():>10.3f} "
          f"{jf.causal_coherence:>10.3f} {jf.behavioral_coherence:>10.3f} "
          f"{result.joint_time:>10.2f}s")

    inf = result.independent_result.best_fitness
    print(f"{'Independent':<15} {inf.total_fitness():>10.3f} "
          f"{inf.causal_coherence:>10.3f} {inf.behavioral_coherence:>10.3f} "
          f"{result.independent_time:>10.2f}s")

    rf = result.random_result.best_fitness
    print(f"{'Random':<15} {rf.total_fitness():>10.3f} "
          f"{rf.causal_coherence:>10.3f} {rf.behavioral_coherence:>10.3f} "
          f"{result.random_time:>10.2f}s")

    print("\n--- Improvement ---")
    print(f"Joint vs Independent: {result.joint_vs_independent()*100:+.1f}%")
    print(f"Joint vs Random:      {result.joint_vs_random()*100:+.1f}%")

    print("\n--- Coherence Breakdown ---")
    print(f"\n{'Component':<20} {'Joint':>10} {'Independent':>10} {'Random':>10}")
    print("-" * 55)

    components = ['causal_coherence', 'temporal_coherence',
                  'behavioral_coherence', 'structural_coherence']

    for comp in components:
        jv = getattr(jf, comp)
        iv = getattr(inf, comp)
        rv = getattr(rf, comp)
        print(f"{comp:<20} {jv:>10.3f} {iv:>10.3f} {rv:>10.3f}")


# =============================================================================
# Demo
# =============================================================================

def demo():
    """Run benchmark demo."""
    print("=" * 60)
    print("JOINT GUARD-ACTION EVOLUTION BENCHMARK")
    print("=" * 60)

    config = JointEvolverConfig(
        population_size=20,
        n_generations=30,
        mutation_rate=0.3,
        crossover_rate=0.5,
        elite_size=2,
        causal_mutation_prob=0.7,
        cluster_crossover=True,
        variables=["turn", "phase", "score", "ready"],
        possible_values={
            "turn": [0, 1, 2],
            "phase": ["init", "play", "end"],
            "score": [0, 1, 2, 3],
            "ready": [True, False]
        },
        n_states=4,
        n_events=3,
        n_pairs=8
    )

    scenarios = [
        Scenario(
            initial_context={"turn": 0, "phase": "init", "score": 0, "ready": False},
            events=[0, 1, 2],
            expected_final_context={"ready": True}
        ),
        Scenario(
            initial_context={"turn": 1, "phase": "play", "score": 0, "ready": True},
            events=[1, 1],
            expected_final_context={"score": 2}
        ),
        Scenario(
            initial_context={"turn": 2, "phase": "play", "score": 3, "ready": True},
            events=[2, 0],
            expected_final_context={"phase": "end"}
        ),
    ]

    result = run_benchmark(config, scenarios, verbose=True)
    print_benchmark_results(result)

    print("\n" + "=" * 60)


if __name__ == "__main__":
    demo()
