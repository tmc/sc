"""
Deep vs Shallow History Benchmark

Compares learning performance across scenarios:
1. FIXED-DEEP: All states use deep history (hardcoded)
2. FIXED-SHALLOW: All states use shallow history (hardcoded)
3. FIXED-NONE: No history anywhere (hardcoded)
4. EVOLVED: Evolution discovers optimal history types

KEY QUESTION: Can evolution discover the right history strategy
for each scenario without hardcoding?

Metrics:
- Accuracy on scenario test cases
- Generations to reach target accuracy
- Final history type distribution
"""

import random
import time
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Type

from .history_evolver import (
    HistoryType, HistoryGenome, HistoryMachine,
    HistoryEvolver, HistoryEvolverConfig, HistoryEvolutionResult,
    create_random_history_genome
)
from .scenarios import (
    HistoryScenario, NestedNavigationScenario, TextEditorScenario,
    GamePauseScenario, SCENARIOS, create_scenario_dataset
)


# =============================================================================
# Benchmark Results
# =============================================================================

@dataclass
class HistoryBenchmarkResult:
    """Result of a single benchmark run."""
    scenario_name: str
    method: str  # "EVOLVED", "FIXED_DEEP", "FIXED_SHALLOW", "FIXED_NONE"
    final_accuracy: float
    generations: int  # -1 for fixed methods
    elapsed_time: float
    history_type_counts: Dict[str, int] = field(default_factory=dict)
    optimal_match: bool = False  # Whether learned matches optimal


@dataclass
class BenchmarkSummary:
    """Summary of all benchmark runs."""
    results: List[HistoryBenchmarkResult] = field(default_factory=list)

    def add(self, result: HistoryBenchmarkResult):
        self.results.append(result)

    def summary(self) -> str:
        """Generate summary table."""
        lines = ["=" * 70, "DEEP VS SHALLOW HISTORY BENCHMARK", "=" * 70]

        # Group by scenario
        scenarios = set(r.scenario_name for r in self.results)

        for scenario in sorted(scenarios):
            scenario_results = [r for r in self.results if r.scenario_name == scenario]
            lines.append(f"\n--- {scenario} ---")

            for r in scenario_results:
                counts = r.history_type_counts
                hist_str = f"H={counts.get('SHALLOW', 0)}, H*={counts.get('DEEP', 0)}, None={counts.get('NONE', 0)}"
                lines.append(
                    f"  {r.method:15s}: {r.final_accuracy:.1%} | {hist_str}"
                )

            # Find best method
            best = max(scenario_results, key=lambda r: r.final_accuracy)
            lines.append(f"  BEST: {best.method} ({best.final_accuracy:.1%})")

        lines.append("\n" + "=" * 70)
        return "\n".join(lines)


# =============================================================================
# Fixed Strategy Evaluation
# =============================================================================

def evaluate_fixed_strategy(
    scenario: HistoryScenario,
    history_type: HistoryType,
    n_cases: int = 100
) -> float:
    """
    Evaluate a fixed history strategy on a scenario.

    All composite states get the same history type.
    """
    structure = scenario.create_structure()
    cases = scenario.generate_cases(n_cases)

    # Create genome with fixed history type
    n_states = structure['n_states']
    genome = HistoryGenome(
        n_states=n_states,
        parent=structure['parent'],
        state_type=structure['state_type'],
        history_type=[
            history_type if structure['state_type'][i] != 0 else HistoryType.NONE
            for i in range(n_states)
        ],
        transitions=structure['transitions'],
        initial_state=structure['initial_state']
    )

    # Evaluate
    correct = 0
    for events, expected in cases:
        machine = HistoryMachine(genome)
        for event in events:
            machine.step(event)

        if machine.current_state == expected:
            correct += 1

    return correct / len(cases) if cases else 0.0


# =============================================================================
# Evolved Strategy Evaluation
# =============================================================================

def evolve_for_scenario(
    scenario: HistoryScenario,
    n_generations: int = 50,
    population_size: int = 30,
    verbose: bool = True
) -> HistoryEvolutionResult:
    """
    Evolve history strategy for a specific scenario.
    """
    structure = scenario.create_structure()
    cases = scenario.generate_cases(200)  # More cases for evolution

    # Create scenario-specific genome creator
    def create_genome():
        n_states = structure['n_states']
        history_types = []
        for i in range(n_states):
            if structure['state_type'][i] != 0:  # Composite
                history_types.append(random.choice(list(HistoryType)))
            else:
                history_types.append(HistoryType.NONE)

        return HistoryGenome(
            n_states=n_states,
            parent=structure['parent'].copy(),
            state_type=structure['state_type'].copy(),
            history_type=history_types,
            transitions=[t for t in structure['transitions']],
            initial_state=structure['initial_state']
        )

    # Scenario generator
    def scenario_generator():
        return cases

    config = HistoryEvolverConfig(
        population_size=population_size,
        n_generations=n_generations,
        n_states=structure['n_states'],
        verbose=verbose,
        log_every=10
    )

    # Custom evolver that uses our structure
    class ScenarioEvolver(HistoryEvolver):
        def evolve(self, target_accuracy=0.9, callback=None):
            config = self.config
            start_time = time.time()

            if config.verbose:
                print(f"\nEvolving history for: {scenario.name}")
                print("-" * 40)

            scenarios_data = self.scenario_generator()

            # Initialize with scenario structure
            population = [create_genome() for _ in range(config.population_size)]

            # Evaluate
            for genome in population:
                genome.fitness, genome.accuracy = self.evaluate_fitness(genome, scenarios_data)

            best_ever = max(population, key=lambda g: g.fitness)
            accuracy_history = []
            history_type_evolution = []
            generations_to_target = -1

            for gen in range(config.n_generations):
                population.sort(key=lambda g: g.fitness, reverse=True)

                if population[0].fitness > best_ever.fitness:
                    best_ever = population[0].copy()

                accuracy_history.append(best_ever.accuracy)
                history_type_evolution.append(best_ever.count_history_types())

                if generations_to_target < 0 and best_ever.accuracy >= target_accuracy:
                    generations_to_target = gen

                if config.verbose and (gen % config.log_every == 0 or gen == config.n_generations - 1):
                    counts = best_ever.count_history_types()
                    print(f"Gen {gen:3d} | Acc: {population[0].accuracy:.3f} | "
                          f"H={counts['SHALLOW']}, H*={counts['DEEP']}")

                # Elite
                new_pop = [g.copy() for g in population[:config.elite_size]]

                # Evolve
                while len(new_pop) < config.population_size:
                    p1 = random.choice(population[:config.population_size//2])
                    child = p1.copy()

                    # Mutate history types
                    for i in range(child.n_states):
                        if random.random() < config.mutation_rate:
                            if child.state_type[i] != 0:
                                child.history_type[i] = random.choice(list(HistoryType))

                    child.fitness, child.accuracy = self.evaluate_fitness(child, scenarios_data)
                    new_pop.append(child)

                population = new_pop

            elapsed = time.time() - start_time

            return HistoryEvolutionResult(
                best_genome=best_ever,
                final_accuracy=best_ever.accuracy,
                generations_to_target=generations_to_target,
                total_generations=config.n_generations,
                elapsed_time=elapsed,
                accuracy_history=accuracy_history,
                history_type_evolution=history_type_evolution
            )

    evolver = ScenarioEvolver(scenario_generator, config)
    return evolver.evolve()


# =============================================================================
# Full Benchmark
# =============================================================================

def compare_history_strategies(
    scenario_names: List[str] = None,
    n_cases: int = 100,
    n_generations: int = 50,
    verbose: bool = True
) -> BenchmarkSummary:
    """
    Compare evolved vs fixed history strategies across scenarios.

    Returns comprehensive benchmark summary.
    """
    scenario_names = scenario_names or list(SCENARIOS.keys())
    summary = BenchmarkSummary()

    if verbose:
        print("=" * 70)
        print("HISTORY STRATEGY BENCHMARK")
        print("=" * 70)
        print(f"Scenarios: {scenario_names}")
        print(f"Test cases: {n_cases}")
        print("-" * 70)

    for scenario_name in scenario_names:
        scenario = SCENARIOS[scenario_name]()

        if verbose:
            print(f"\n{'='*70}")
            print(f"SCENARIO: {scenario.name}")
            print(f"{'='*70}")
            print(f"Description: {scenario.description}")

        # =================================================================
        # Evaluate Fixed Strategies
        # =================================================================

        for ht, method in [(HistoryType.NONE, "FIXED_NONE"),
                           (HistoryType.SHALLOW, "FIXED_SHALLOW"),
                           (HistoryType.DEEP, "FIXED_DEEP")]:
            start = time.time()
            accuracy = evaluate_fixed_strategy(scenario, ht, n_cases)
            elapsed = time.time() - start

            # Count history types
            structure = scenario.create_structure()
            n_composite = sum(1 for st in structure['state_type'] if st != 0)
            counts = {
                'NONE': structure['n_states'] if ht == HistoryType.NONE else structure['n_states'] - n_composite,
                'SHALLOW': n_composite if ht == HistoryType.SHALLOW else 0,
                'DEEP': n_composite if ht == HistoryType.DEEP else 0,
            }

            result = HistoryBenchmarkResult(
                scenario_name=scenario_name,
                method=method,
                final_accuracy=accuracy,
                generations=-1,  # No evolution
                elapsed_time=elapsed,
                history_type_counts=counts
            )
            summary.add(result)

            if verbose:
                print(f"{method}: {accuracy:.1%}")

        # =================================================================
        # Evolve Strategy
        # =================================================================

        if verbose:
            print(f"\n--- Evolving optimal strategy ---")

        evo_result = evolve_for_scenario(
            scenario,
            n_generations=n_generations,
            verbose=verbose
        )

        # Check if matches optimal
        optimal = scenario.optimal_history_type()
        learned = evo_result.best_genome.history_type
        optimal_match = all(o == l for o, l in zip(optimal, learned))

        result = HistoryBenchmarkResult(
            scenario_name=scenario_name,
            method="EVOLVED",
            final_accuracy=evo_result.final_accuracy,
            generations=evo_result.generations_to_target,
            elapsed_time=evo_result.elapsed_time,
            history_type_counts=evo_result.best_genome.count_history_types(),
            optimal_match=optimal_match
        )
        summary.add(result)

        if verbose:
            print(f"\nEVOLVED: {evo_result.final_accuracy:.1%}")
            print(f"Matches optimal: {optimal_match}")
            print(f"Learned: {learned}")
            print(f"Optimal: {optimal}")

    if verbose:
        print("\n" + summary.summary())

    return summary


# =============================================================================
# Quick Benchmark
# =============================================================================

def quick_benchmark() -> BenchmarkSummary:
    """Quick benchmark for testing."""
    return compare_history_strategies(
        scenario_names=['nested_navigation', 'text_editor', 'game_pause'],
        n_cases=50,
        n_generations=30,
        verbose=True
    )


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--quick":
        summary = quick_benchmark()
    else:
        summary = compare_history_strategies(
            n_cases=100,
            n_generations=50,
            verbose=True
        )

    print("\nBenchmark complete!")
