"""
Experiment: Action Side Effects

Main experiment runner that demonstrates:
1. Actions mutate context variables
2. Guards check context variables
3. Action->guard dependencies are EVOLVED, not hardcoded

Key metrics:
- Dependency discovery accuracy (do we find the right relationships?)
- Prediction accuracy (can we predict guard changes from actions?)
- Causal strength estimation (do we correctly weight relationships?)
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Tuple
import random
from collections import defaultdict

from .action_evolver import ActionEvolver, ActionObservation, DependencyGenome
from .context_tracker import ContextTracker
from .causal_learner import CausalLearner, CausalGraph


@dataclass
class ExperimentConfig:
    """Configuration for the experiment."""
    n_observations: int = 200
    n_evolution_generations: int = 50
    population_size: int = 30
    n_test_cases: int = 50
    random_seed: int = 42


@dataclass
class ExperimentResults:
    """Results from running the experiment."""
    # Discovery metrics
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0

    # Prediction metrics
    prediction_accuracy: float = 0.0
    prediction_f1: float = 0.0

    # Learned relationships
    learned_dependencies: Dict[str, Dict[str, float]] = None

    # Causal graph
    causal_graph: CausalGraph = None

    @property
    def precision(self) -> float:
        if self.true_positives + self.false_positives == 0:
            return 0.0
        return self.true_positives / (self.true_positives + self.false_positives)

    @property
    def recall(self) -> float:
        if self.true_positives + self.false_negatives == 0:
            return 0.0
        return self.true_positives / (self.true_positives + self.false_negatives)

    @property
    def f1(self) -> float:
        if self.precision + self.recall == 0:
            return 0.0
        return 2 * self.precision * self.recall / (self.precision + self.recall)


# =============================================================================
# Test Domain: Game State Machine
# =============================================================================

class GameStateMachine:
    """
    A game state machine with actions and guards.

    The TRUE dependencies between actions and guards are defined here,
    but the experiment must DISCOVER them through evolution.
    """

    def __init__(self):
        # Define actions and their effects (GROUND TRUTH - experiment must discover)
        self.action_effects = {
            "start_game": {"game_started": True, "lives": 3, "score": 0},
            "collect_coin": {"score": "+10"},
            "take_damage": {"lives": "-1"},
            "get_powerup": {"has_powerup": True, "powerup_timer": 10},
            "use_powerup": {"has_powerup": False, "invincible": True},
            "tick_timer": {"powerup_timer": "-1"},
            "lose_invincibility": {"invincible": False},
            "reach_checkpoint": {"checkpoint_reached": True},
            "complete_level": {"level_complete": True, "score": "+100"},
        }

        # Define guards and their dependencies (GROUND TRUTH - experiment must discover)
        self.guard_dependencies = {
            "can_play": {"game_started"},
            "is_alive": {"lives"},
            "has_powerup": {"has_powerup"},
            "is_invincible": {"invincible"},
            "powerup_active": {"powerup_timer"},
            "high_score": {"score"},
            "at_checkpoint": {"checkpoint_reached"},
            "level_won": {"level_complete"},
        }

        # Initial state
        self.default_context = {
            "game_started": False,
            "lives": 0,
            "score": 0,
            "has_powerup": False,
            "powerup_timer": 0,
            "invincible": False,
            "checkpoint_reached": False,
            "level_complete": False,
        }

    def get_actions(self) -> List[str]:
        return list(self.action_effects.keys())

    def get_guards(self) -> List[str]:
        return list(self.guard_dependencies.keys())

    def get_variables(self) -> List[str]:
        return list(self.default_context.keys())

    def execute_action(self, action: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute an action and return modified context."""
        new_context = context.copy()

        if action not in self.action_effects:
            return new_context

        effects = self.action_effects[action]
        for var, value in effects.items():
            if isinstance(value, str) and value.startswith("+"):
                # Increment
                delta = int(value[1:])
                new_context[var] = new_context.get(var, 0) + delta
            elif isinstance(value, str) and value.startswith("-"):
                # Decrement
                delta = int(value[1:])
                new_context[var] = new_context.get(var, 0) - delta
            else:
                # Set
                new_context[var] = value

        return new_context

    def evaluate_guard(self, guard: str, context: Dict[str, Any]) -> bool:
        """Evaluate a guard in context."""
        if guard == "can_play":
            return context.get("game_started", False)
        elif guard == "is_alive":
            return context.get("lives", 0) > 0
        elif guard == "has_powerup":
            return context.get("has_powerup", False)
        elif guard == "is_invincible":
            return context.get("invincible", False)
        elif guard == "powerup_active":
            return context.get("powerup_timer", 0) > 0
        elif guard == "high_score":
            return context.get("score", 0) >= 100
        elif guard == "at_checkpoint":
            return context.get("checkpoint_reached", False)
        elif guard == "level_won":
            return context.get("level_complete", False)
        return False

    def get_ground_truth_dependencies(self) -> Dict[str, Set[str]]:
        """
        Get ground truth action->guard dependencies.

        An action affects a guard if:
        - Action modifies variable X
        - Guard depends on variable X
        """
        dependencies = {}
        for action, effects in self.action_effects.items():
            modified_vars = {k for k in effects.keys()}
            affected_guards = set()
            for guard, deps in self.guard_dependencies.items():
                if modified_vars & deps:  # Intersection
                    affected_guards.add(guard)
            dependencies[action] = affected_guards
        return dependencies


def generate_observations(
    game: GameStateMachine,
    n_observations: int,
) -> List[ActionObservation]:
    """Generate observations by simulating game play."""
    observations = []

    for _ in range(n_observations):
        # Random context
        context = game.default_context.copy()
        context["game_started"] = random.random() < 0.8
        context["lives"] = random.randint(0, 3)
        context["score"] = random.randint(0, 200)
        context["has_powerup"] = random.random() < 0.3
        context["powerup_timer"] = random.randint(0, 10) if context["has_powerup"] else 0
        context["invincible"] = random.random() < 0.2
        context["checkpoint_reached"] = random.random() < 0.4
        context["level_complete"] = random.random() < 0.1

        # Random action
        action = random.choice(game.get_actions())

        # Execute action
        context_before = context.copy()
        context_after = game.execute_action(action, context)

        # Evaluate guards before and after
        guards_before = {g: game.evaluate_guard(g, context_before) for g in game.get_guards()}
        guards_after = {g: game.evaluate_guard(g, context_after) for g in game.get_guards()}

        obs = ActionObservation(
            action_name=action,
            context_before=context_before,
            context_after=context_after,
            guards_before=guards_before,
            guards_after=guards_after,
        )
        observations.append(obs)

    return observations


def evaluate_learned_dependencies(
    learned: Dict[str, Dict[str, float]],
    ground_truth: Dict[str, Set[str]],
    threshold: float = 0.3,
) -> Tuple[int, int, int]:
    """
    Evaluate learned dependencies against ground truth.

    Returns (true_positives, false_positives, false_negatives)
    """
    tp, fp, fn = 0, 0, 0

    for action, true_guards in ground_truth.items():
        learned_guards = {g for g, p in learned.get(action, {}).items() if p >= threshold}

        tp += len(true_guards & learned_guards)
        fp += len(learned_guards - true_guards)
        fn += len(true_guards - learned_guards)

    return tp, fp, fn


def run_experiment(config: ExperimentConfig = None) -> ExperimentResults:
    """
    Run the full experiment.

    Steps:
    1. Create game domain with ground truth dependencies
    2. Generate observations through simulation
    3. Evolve action->guard dependencies
    4. Evaluate against ground truth
    """
    if config is None:
        config = ExperimentConfig()

    random.seed(config.random_seed)

    print("=" * 60)
    print("EXPERIMENT: Action Side Effects")
    print("=" * 60)

    # Create game domain
    game = GameStateMachine()
    ground_truth = game.get_ground_truth_dependencies()

    print("\nGround truth dependencies (UNKNOWN to evolver):")
    for action, guards in ground_truth.items():
        if guards:
            print(f"  {action} -> {', '.join(guards)}")

    # Generate observations
    print(f"\nGenerating {config.n_observations} observations...")
    observations = generate_observations(game, config.n_observations)

    # Create evolver
    evolver = ActionEvolver(
        known_actions=game.get_actions(),
        known_guards=game.get_guards(),
        known_variables=game.get_variables(),
        population_size=config.population_size,
    )

    # Collect observations
    for obs in observations:
        evolver.collect_observation(obs)

    # Evolve dependencies
    print(f"\nEvolving dependencies over {config.n_evolution_generations} generations...")
    best_genome = evolver.evolve(
        n_generations=config.n_evolution_generations,
        verbose=True,
    )

    # Get learned dependencies
    learned = evolver.get_learned_dependencies()

    # Evaluate
    tp, fp, fn = evaluate_learned_dependencies(learned, ground_truth)

    results = ExperimentResults(
        true_positives=tp,
        false_positives=fp,
        false_negatives=fn,
        prediction_accuracy=best_genome.prediction_accuracy,
        prediction_f1=best_genome.fitness,
        learned_dependencies=learned,
    )

    # Print results
    print("\n" + "=" * 60)
    print("EXPERIMENT RESULTS")
    print("=" * 60)

    print(f"\nDependency Discovery:")
    print(f"  True Positives:  {tp}")
    print(f"  False Positives: {fp}")
    print(f"  False Negatives: {fn}")
    print(f"  Precision:       {results.precision:.3f}")
    print(f"  Recall:          {results.recall:.3f}")
    print(f"  F1 Score:        {results.f1:.3f}")

    print(f"\nPrediction Metrics:")
    print(f"  Accuracy: {results.prediction_accuracy:.3f}")
    print(f"  F1:       {results.prediction_f1:.3f}")

    # Print learned dependencies
    evolver.print_learned_dependencies()

    # Compare with ground truth
    print("\n" + "=" * 60)
    print("COMPARISON WITH GROUND TRUTH")
    print("=" * 60)

    for action in game.get_actions():
        true_guards = ground_truth.get(action, set())
        learned_guards = {g for g, p in learned.get(action, {}).items() if p >= 0.3}

        correct = true_guards & learned_guards
        missed = true_guards - learned_guards
        extra = learned_guards - true_guards

        print(f"\n{action}:")
        if correct:
            print(f"  Correct: {', '.join(correct)}")
        if missed:
            print(f"  Missed:  {', '.join(missed)}")
        if extra:
            print(f"  Extra:   {', '.join(extra)}")
        if not correct and not missed and not extra:
            print(f"  (no dependencies)")

    # Run causal learning as well
    print("\n" + "=" * 60)
    print("CAUSAL LEARNING")
    print("=" * 60)

    causal_learner = CausalLearner(
        actions=game.get_actions(),
        guards=game.get_guards(),
        variables=game.get_variables(),
        action_executors={a: lambda c, a=a: game.execute_action(a, c) for a in game.get_actions()},
        guard_evaluators={g: lambda c, g=g: game.evaluate_guard(g, c) for g in game.get_guards()},
    )

    # Add observations to causal learner
    for obs in observations:
        causal_learner.observe(obs.action_name, obs.context_before, obs.context_after)

    # Learn causal graph
    causal_graph = causal_learner.learn_causal_graph()
    causal_graph.print_graph()

    results.causal_graph = causal_graph

    return results


def run_benchmark(n_trials: int = 5) -> Dict[str, float]:
    """
    Run multiple trials and compute aggregate metrics.
    """
    print("=" * 60)
    print("BENCHMARK: Multiple Trials")
    print("=" * 60)

    metrics = defaultdict(list)

    for trial in range(n_trials):
        print(f"\n--- Trial {trial + 1}/{n_trials} ---")
        config = ExperimentConfig(random_seed=42 + trial)
        results = run_experiment(config)

        metrics["precision"].append(results.precision)
        metrics["recall"].append(results.recall)
        metrics["f1"].append(results.f1)
        metrics["prediction_accuracy"].append(results.prediction_accuracy)

    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)

    summary = {}
    for metric, values in metrics.items():
        avg = sum(values) / len(values)
        summary[metric] = avg
        print(f"{metric}: {avg:.3f} (std={sum((v-avg)**2 for v in values)/len(values)**0.5:.3f})")

    return summary


if __name__ == "__main__":
    # Run single experiment
    results = run_experiment()

    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    print(f"Dependency Discovery F1: {results.f1:.3f}")
    print(f"Prediction Accuracy:     {results.prediction_accuracy:.3f}")
    print("\nKey insight: Dependencies were LEARNED through evolution, not hardcoded!")
