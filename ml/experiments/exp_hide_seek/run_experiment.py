#!/usr/bin/env python3
"""
Hide-and-Seek Emergence Experiment

Evolves seeker and hider strategies through competitive self-play.
Like OpenAI's emergent tool use, but with EXTRACTABLE statecharts!

Key insight: The strategies that emerge are fully observable and interpretable.
"""

import json
import sys
from datetime import datetime
from pathlib import Path

# Handle imports for both module and direct execution
try:
    from .grid_world import GridWorld
    from .seeker_statechart import SeekerStatechart, SeekerGenome
    from .hider_statechart import HiderStatechart, HiderGenome
    from .evolution import HideSeekEvolution, EvolutionConfig, demonstrate_game
except ImportError:
    from grid_world import GridWorld
    from seeker_statechart import SeekerStatechart, SeekerGenome
    from hider_statechart import HiderStatechart, HiderGenome
    from evolution import HideSeekEvolution, EvolutionConfig, demonstrate_game


def run_hide_seek_experiment(
    generations: int = 30,
    population_size: int = 15,
    games_per_eval: int = 8,
    verbose: bool = True,
) -> dict:
    """
    Run the full hide-and-seek evolution experiment.

    Returns results including emerged strategies as extractable statecharts.
    """
    print("=" * 60)
    print("HIDE-AND-SEEK EMERGENCE EXPERIMENT")
    print("=" * 60)
    print(f"Generations: {generations}")
    print(f"Population Size: {population_size}")
    print(f"Games per Evaluation: {games_per_eval}")
    print()

    # Configure evolution
    config = EvolutionConfig(
        population_size=population_size,
        num_generations=generations,
        games_per_evaluation=games_per_eval,
        tournament_size=3,
        elite_count=2,
        mutation_rate=0.25,
        crossover_rate=0.6,
        arena_width=15,
        arena_height=15,
        num_walls=8,
        num_boxes=4,
        num_shelters=3,
        max_steps=150,
    )

    # Run evolution
    evolution = HideSeekEvolution(config)
    print("Starting co-evolution...")
    print("-" * 60)

    results = evolution.run_evolution(verbose=verbose)

    # Analyze emergence
    print("-" * 60)
    print("\nANALYZING EMERGENT STRATEGIES...")
    emergence = evolution.analyze_emergence()

    print("\n=== SEEKER POPULATION TRAITS ===")
    for trait, freq in emergence['seeker_traits'].items():
        bar = '█' * int(freq * 20)
        print(f"  {trait:25s} [{bar:20s}] {freq*100:.0f}%")

    print("\n=== HIDER POPULATION TRAITS ===")
    for trait, freq in emergence['hider_traits'].items():
        bar = '█' * int(freq * 20)
        print(f"  {trait:25s} [{bar:20s}] {freq*100:.0f}%")

    print("\n=== DOMINANT STRATEGIES ===")
    print(f"Seekers: {emergence['seeker_dominant_strategies']}")
    print(f"Hiders: {emergence['hider_dominant_strategies']}")

    print("\n=== EMERGENT BEHAVIORS ===")
    for behavior, observed in emergence['emergent_behaviors'].items():
        status = "✓ EMERGED" if observed else "✗ not observed"
        print(f"  {behavior}: {status}")

    # Add emergence analysis to results
    results['emergence_analysis'] = emergence

    # Show extracted strategies
    print("\n" + "=" * 60)
    print("EXTRACTED STATECHARTS (Interpretable Strategies)")
    print("=" * 60)

    print("\n--- Best Seeker Strategy ---")
    seeker_strat = results['emerged_strategies']['seeker']
    print(f"Modes: {seeker_strat['modes']}")
    print("Transitions:")
    for t in seeker_strat['transitions']:
        print(f"  {t['from']} -> {t['to']}: {t['guard']}")
    print("Mode Behaviors:")
    for mode, behavior in seeker_strat['mode_behaviors'].items():
        print(f"  {mode}: {behavior}")

    print("\n--- Best Hider Strategy ---")
    hider_strat = results['emerged_strategies']['hider']
    print(f"Modes: {hider_strat['modes']}")
    print("Transitions:")
    for t in hider_strat['transitions']:
        print(f"  {t['from']} -> {t['to']}: {t['guard']}")
    print("Mode Behaviors:")
    for mode, behavior in hider_strat['mode_behaviors'].items():
        print(f"  {mode}: {behavior}")

    # Run demonstration game with best strategies
    print("\n" + "=" * 60)
    print("DEMONSTRATION GAME (Best vs Best)")
    print("=" * 60)

    # Get best genomes
    best_seeker = max(evolution.seeker_population, key=lambda i: i.fitness)
    best_hider = max(evolution.hider_population, key=lambda i: i.fitness)

    demo = demonstrate_game(
        seeker_genome=best_seeker.genome,
        hider_genome=best_hider.genome,
        max_steps=50,
    )
    print(demo)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Final Seeker Win Rate: {results['final_stats']['seeker_win_rate']*100:.1f}%")
    print(f"Best Seeker Fitness: {results['final_stats']['seeker_best_fitness']:.2f}")
    print(f"Best Hider Fitness: {results['final_stats']['hider_best_fitness']:.2f}")

    # Key innovation highlight
    print("\n*** KEY INSIGHT ***")
    print("Unlike black-box neural networks, these strategies are:")
    print("  1. EXTRACTABLE - full statechart structure available")
    print("  2. INTERPRETABLE - human-readable guards and modes")
    print("  3. TRANSFERABLE - can be deployed without neural network")
    print("  4. EVOLVABLE - structure emerged from self-play")

    return results


def save_results(results: dict, filename: str = None):
    """Save results to JSON file."""
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"hide_seek_results_{timestamp}.json"

    # Make results JSON serializable
    def make_serializable(obj):
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_serializable(v) for v in obj]
        elif hasattr(obj, '__dict__'):
            return make_serializable(obj.__dict__)
        else:
            return obj

    with open(filename, 'w') as f:
        json.dump(make_serializable(results), f, indent=2)

    print(f"\nResults saved to: {filename}")
    return filename


if __name__ == '__main__':
    # Parse command line args
    generations = 30
    population = 15
    games = 8

    if len(sys.argv) > 1:
        generations = int(sys.argv[1])
    if len(sys.argv) > 2:
        population = int(sys.argv[2])
    if len(sys.argv) > 3:
        games = int(sys.argv[3])

    # Run experiment
    results = run_hide_seek_experiment(
        generations=generations,
        population_size=population,
        games_per_eval=games,
    )

    # Save results
    save_results(results)

    print("\n" + "=" * 60)
    print("EXPERIMENT COMPLETE")
    print("=" * 60)
