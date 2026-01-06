"""
Benchmark: Evaluate Belief-Conditioned Policies

Comprehensive benchmarking of:
1. Belief tracking accuracy
2. Policy performance vs baselines
3. Belief update efficiency
4. Evolution convergence

Compares:
- Random policy (baseline)
- Static policy (no belief updates)
- Belief-conditioned policy (this experiment)
- Evolved belief policy
"""

from dataclasses import dataclass, field
from typing import List, Dict, Set, Tuple, Optional, Any
import random
import time
import math

from .belief_state import (
    ParallelBeliefState,
    BeliefDimension,
    BeliefConfiguration,
    create_poker_belief_state,
)
from .belief_tracker import (
    BeliefTracker,
    Observation,
    create_poker_observation_models,
)
from .poker_beliefs import (
    PokerBeliefState,
    PokerAction,
    PokerObservation,
)
from .belief_policy import (
    PolicyGenome,
    ActionType,
    EvolvedBeliefPolicy,
    mutate_policy,
)


@dataclass
class BenchmarkResult:
    """Result from a benchmark run."""
    name: str
    metric: str
    value: float
    std_dev: float = 0.0
    n_samples: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PokerGame:
    """Simple poker game for benchmarking."""
    our_strength: float = 0.0
    opponent_strength: float = 0.0
    pot: float = 2.0  # Antes
    our_bet: float = 1.0
    opponent_bet: float = 1.0
    is_over: bool = False
    winner: Optional[int] = None  # 0 = us, 1 = opponent, None = tie

    def reset(self):
        """Reset for new hand."""
        self.our_strength = random.random()
        self.opponent_strength = random.random()
        self.pot = 2.0
        self.our_bet = 1.0
        self.opponent_bet = 1.0
        self.is_over = False
        self.winner = None


class RandomPolicy:
    """Baseline: random action selection."""

    def get_action(self, belief: PokerBeliefState, our_strength: float) -> ActionType:
        return random.choice(list(ActionType))


class StaticPolicy:
    """Baseline: static rules without belief updates."""

    def __init__(self, aggression: float = 0.5):
        self.aggression = aggression

    def get_action(self, belief: PokerBeliefState, our_strength: float) -> ActionType:
        # Simple threshold-based policy
        if our_strength > 0.7:
            return ActionType.AGGRESSIVE
        elif our_strength > 0.4:
            if random.random() < self.aggression:
                return ActionType.AGGRESSIVE
            return ActionType.PASSIVE
        else:
            if random.random() < 0.3:
                return ActionType.FOLD
            return ActionType.PASSIVE


class OpponentSimulator:
    """Simulate opponent behavior for benchmarking."""

    def __init__(self, style: str = "balanced"):
        self.style = style
        # Style affects behavior
        self.bluff_rate = {"passive": 0.1, "aggressive": 0.3, "bluffer": 0.5, "balanced": 0.2}[style]
        self.fold_rate = {"passive": 0.4, "aggressive": 0.1, "bluffer": 0.2, "balanced": 0.25}[style]

    def get_action(self, strength: float, facing_bet: bool = False) -> PokerAction:
        """Get opponent action based on their strength."""
        if facing_bet:
            if strength > 0.6:
                return random.choice([PokerAction.CALL, PokerAction.RAISE])
            elif strength > 0.3:
                if random.random() < self.fold_rate:
                    return PokerAction.FOLD
                return PokerAction.CALL
            else:
                if random.random() < self.bluff_rate:
                    return PokerAction.RAISE
                if random.random() < self.fold_rate:
                    return PokerAction.FOLD
                return PokerAction.CALL
        else:
            if strength > 0.7:
                return PokerAction.BET_BIG
            elif strength > 0.4:
                return random.choice([PokerAction.CHECK, PokerAction.BET_SMALL])
            else:
                if random.random() < self.bluff_rate:
                    return PokerAction.BET_BIG
                return PokerAction.CHECK


def evaluate_policy(
    policy: Any,
    opponent_style: str = "balanced",
    n_hands: int = 500,
) -> BenchmarkResult:
    """Evaluate a policy against an opponent."""
    opponent = OpponentSimulator(style=opponent_style)
    total_profit = 0.0
    profits = []

    for _ in range(n_hands):
        belief = PokerBeliefState()
        game = PokerGame()
        game.reset()

        # Get our action
        our_action = policy.get_action(belief, game.our_strength)

        if our_action == ActionType.FOLD:
            profit = -game.our_bet
            profits.append(profit)
            total_profit += profit
            continue

        # We act first
        if our_action in [ActionType.AGGRESSIVE, ActionType.BLUFF]:
            bet_size = 2
            game.pot += bet_size
            game.our_bet += bet_size

            # Opponent observes our bet
            opp_action = opponent.get_action(game.opponent_strength, facing_bet=True)
        else:
            opp_action = opponent.get_action(game.opponent_strength, facing_bet=False)

        # Update our belief
        belief.observe_action(opp_action)

        # Handle opponent action
        if opp_action == PokerAction.FOLD:
            profit = game.pot - game.our_bet
            profits.append(profit)
            total_profit += profit
            continue

        if opp_action in [PokerAction.BET_BIG, PokerAction.RAISE]:
            bet_size = 2
            game.pot += bet_size
            game.opponent_bet += bet_size

            # We might need to respond
            if game.our_bet < game.opponent_bet:
                # Decide to call or fold based on updated belief
                if policy.get_action(belief, game.our_strength) == ActionType.FOLD:
                    profit = -game.our_bet
                    profits.append(profit)
                    total_profit += profit
                    continue
                else:
                    # Call
                    call_amount = game.opponent_bet - game.our_bet
                    game.pot += call_amount
                    game.our_bet += call_amount

        # Showdown
        if game.our_strength > game.opponent_strength:
            profit = game.pot - game.our_bet
        elif game.opponent_strength > game.our_strength:
            profit = -game.our_bet
        else:
            profit = 0  # Tie

        profits.append(profit)
        total_profit += profit

    mean_profit = total_profit / n_hands
    std_dev = math.sqrt(sum((p - mean_profit) ** 2 for p in profits) / n_hands)

    return BenchmarkResult(
        name=type(policy).__name__,
        metric="profit_per_hand",
        value=mean_profit,
        std_dev=std_dev,
        n_samples=n_hands,
        metadata={"opponent_style": opponent_style},
    )


def benchmark_belief_accuracy(n_trials: int = 100) -> BenchmarkResult:
    """Benchmark belief tracking accuracy."""
    errors = []

    for _ in range(n_trials):
        # True opponent state
        true_strength = random.random()
        true_style = random.choice(["passive", "aggressive", "bluffer"])

        # Simulate opponent actions based on true state
        opponent = OpponentSimulator(style=true_style)
        belief = PokerBeliefState()

        # 10 actions
        for _ in range(10):
            action = opponent.get_action(true_strength, facing_bet=random.random() < 0.5)
            belief.observe_action(action)

        # Measure error
        predicted_strength = belief.expected_opponent_strength()
        strength_error = abs(predicted_strength - true_strength)

        # Strategy prediction accuracy
        predicted_style = belief.opponent_strategy.get_active_state()
        style_correct = any(
            style_part in predicted_style.lower()
            for style_part in true_style.split("_")
        )

        errors.append(strength_error)

    mean_error = sum(errors) / len(errors)
    std_dev = math.sqrt(sum((e - mean_error) ** 2 for e in errors) / len(errors))

    return BenchmarkResult(
        name="BeliefAccuracy",
        metric="mean_absolute_error",
        value=mean_error,
        std_dev=std_dev,
        n_samples=n_trials,
    )


def benchmark_update_speed(n_trials: int = 1000) -> BenchmarkResult:
    """Benchmark belief update speed."""
    times = []

    for _ in range(n_trials):
        belief = PokerBeliefState()

        t0 = time.perf_counter()
        for _ in range(10):
            action = random.choice(list(PokerAction))
            belief.observe_action(action)
        elapsed = time.perf_counter() - t0

        times.append(elapsed)

    mean_time = sum(times) / len(times)
    std_dev = math.sqrt(sum((t - mean_time) ** 2 for t in times) / len(times))

    return BenchmarkResult(
        name="UpdateSpeed",
        metric="seconds_per_10_updates",
        value=mean_time,
        std_dev=std_dev,
        n_samples=n_trials,
    )


def benchmark_evolution_convergence(
    n_runs: int = 5,
    n_generations: int = 30,
) -> BenchmarkResult:
    """Benchmark evolution convergence."""
    final_fitnesses = []

    for run in range(n_runs):
        evolver = EvolvedBeliefPolicy(
            population_size=20,
            n_generations=n_generations,
        )
        best = evolver.evolve(verbose=False)
        final_fitnesses.append(best.fitness)

    mean_fitness = sum(final_fitnesses) / len(final_fitnesses)
    std_dev = math.sqrt(sum((f - mean_fitness) ** 2 for f in final_fitnesses) / len(final_fitnesses))

    return BenchmarkResult(
        name="EvolutionConvergence",
        metric="final_fitness",
        value=mean_fitness,
        std_dev=std_dev,
        n_samples=n_runs,
        metadata={"n_generations": n_generations},
    )


def run_benchmark(verbose: bool = True) -> Dict[str, BenchmarkResult]:
    """Run all benchmarks."""
    if verbose:
        print("=" * 60)
        print("PARTIAL OBSERVABILITY BENCHMARK")
        print("=" * 60)

    results = {}

    # 1. Policy comparison
    if verbose:
        print("\n--- Policy Comparison ---")

    policies = {
        "Random": RandomPolicy(),
        "Static_Passive": StaticPolicy(aggression=0.3),
        "Static_Aggressive": StaticPolicy(aggression=0.7),
        "Belief_Default": PolicyGenome(),
    }

    for name, policy in policies.items():
        for opponent_style in ["passive", "aggressive", "bluffer"]:
            result = evaluate_policy(policy, opponent_style, n_hands=300)
            key = f"{name}_vs_{opponent_style}"
            results[key] = result
            if verbose:
                print(f"  {key}: {result.value:+.3f} +/- {result.std_dev:.3f}")

    # 2. Belief accuracy
    if verbose:
        print("\n--- Belief Accuracy ---")

    result = benchmark_belief_accuracy(n_trials=100)
    results["belief_accuracy"] = result
    if verbose:
        print(f"  Mean error: {result.value:.3f} +/- {result.std_dev:.3f}")

    # 3. Update speed
    if verbose:
        print("\n--- Update Speed ---")

    result = benchmark_update_speed(n_trials=500)
    results["update_speed"] = result
    if verbose:
        print(f"  Time per 10 updates: {result.value*1000:.2f} +/- {result.std_dev*1000:.2f} ms")

    # 4. Evolution convergence
    if verbose:
        print("\n--- Evolution Convergence ---")

    result = benchmark_evolution_convergence(n_runs=3, n_generations=20)
    results["evolution"] = result
    if verbose:
        print(f"  Final fitness: {result.value:+.3f} +/- {result.std_dev:.3f}")

    # 5. Evolved vs baselines
    if verbose:
        print("\n--- Evolved Policy vs Baselines ---")

    evolver = EvolvedBeliefPolicy(population_size=25, n_generations=30)
    evolved = evolver.evolve(verbose=False)

    for opponent_style in ["passive", "aggressive", "bluffer"]:
        result = evaluate_policy(evolved, opponent_style, n_hands=300)
        key = f"Evolved_vs_{opponent_style}"
        results[key] = result
        if verbose:
            print(f"  {key}: {result.value:+.3f} +/- {result.std_dev:.3f}")

    # Summary
    if verbose:
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)

        # Compare belief vs static
        belief_avg = sum(results[f"Belief_Default_vs_{s}"].value for s in ["passive", "aggressive", "bluffer"]) / 3
        static_avg = sum(results[f"Static_Passive_vs_{s}"].value for s in ["passive", "aggressive", "bluffer"]) / 3
        evolved_avg = sum(results[f"Evolved_vs_{s}"].value for s in ["passive", "aggressive", "bluffer"]) / 3

        print(f"\nAverage profit per hand:")
        print(f"  Static policy:   {static_avg:+.3f}")
        print(f"  Belief policy:   {belief_avg:+.3f}")
        print(f"  Evolved policy:  {evolved_avg:+.3f}")

        improvement = (evolved_avg - static_avg) / abs(static_avg) * 100 if static_avg != 0 else 0
        print(f"\nEvolved improvement over static: {improvement:+.1f}%")

    return results


def demo():
    """Quick demo of benchmarking."""
    print("Running quick benchmark...")
    results = run_benchmark(verbose=True)
    return results


if __name__ == "__main__":
    demo()
