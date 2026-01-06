"""
Protocol Evolution for Multi-Agent Coordination

Evolves:
1. Role assignment (who becomes LEADER/FOLLOWER)
2. Communication protocols (when to signal, what signals mean)
3. Coordination patterns (formations, sequencing)

Uses co-evolution: agents evolve together, creating emergent conventions.
"""

import mlx.core as mx
from dataclasses import dataclass, field
from typing import List, Dict, Set, Tuple, Optional, Any, Callable
from enum import Enum, auto
from collections import defaultdict
import random
import copy
import math

from .multi_agent_env import MultiAgentEnv, Event, EventType, Task, TaskType
from .agent_statechart import (
    AgentStatechart, AgentTransition, BehaviorMode, AgentRole,
    MultiAgentStatechartController
)


# =============================================================================
# Protocol Genome
# =============================================================================

@dataclass
class SignalMeaning:
    """What a signal means (evolved)."""
    signal_type: EventType
    action: str  # "MOVE_TO_SENDER", "MOVE_TO_TARGET", "WAIT", "FOLLOW", etc.
    priority: float = 0.5


@dataclass
class RoleCondition:
    """Condition for taking a role (evolved)."""
    role: AgentRole
    condition: str  # "FIRST_TO_SIGNAL", "MOST_CENTRAL", "RANDOM", etc.
    threshold: float = 0.5


@dataclass
class ProtocolGenome:
    """
    Genome encoding communication protocol.

    Evolved parameters:
    - Signal meanings
    - Role assignment rules
    - Transition probabilities
    - Timing parameters
    """
    # Signal interpretations
    signal_meanings: Dict[EventType, SignalMeaning] = field(default_factory=dict)

    # Role assignment
    role_conditions: List[RoleCondition] = field(default_factory=list)

    # Transition probabilities (evolved)
    transition_probs: Dict[Tuple[BehaviorMode, BehaviorMode], float] = field(
        default_factory=dict
    )

    # Timing
    signal_cooldown: float = 0.5
    leader_claim_prob: float = 0.1
    follow_threshold: float = 0.3

    # Fitness
    fitness: float = 0.0

    def clone(self) -> "ProtocolGenome":
        return ProtocolGenome(
            signal_meanings=dict(self.signal_meanings),
            role_conditions=list(self.role_conditions),
            transition_probs=dict(self.transition_probs),
            signal_cooldown=self.signal_cooldown,
            leader_claim_prob=self.leader_claim_prob,
            follow_threshold=self.follow_threshold,
        )


# =============================================================================
# Evolved Agent Statechart
# =============================================================================

class EvolvedAgentStatechart(AgentStatechart):
    """Agent statechart with evolved protocol."""

    def __init__(self, agent_id: int, genome: ProtocolGenome):
        super().__init__(agent_id)
        self.genome = genome
        self._apply_genome()

    def _apply_genome(self):
        """Apply genome parameters to statechart."""
        # Update transition probabilities
        for trans in self.transitions:
            key = (trans.source, trans.target)
            if key in self.genome.transition_probs:
                trans.probability = self.genome.transition_probs[key]

        # Update timing
        self.leader_claim_prob = self.genome.leader_claim_prob

    def _explore_action(
        self, agent, context, events
    ) -> Tuple[Tuple[float, float], Optional[Event]]:
        """EXPLORING with evolved leader claim probability."""
        vx = agent.vx + random.gauss(0, 0.02)
        vy = agent.vy + random.gauss(0, 0.02)

        speed = math.sqrt(vx * vx + vy * vy) + 1e-6
        if speed > 0.1:
            vx = (vx / speed) * 0.1
            vy = (vy / speed) * 0.1

        event = None
        if self.signal_cooldown <= 0:
            if random.random() < self.genome.leader_claim_prob:
                if self.role == AgentRole.NONE:
                    self.role = AgentRole.LEADER
                    event = Event(
                        event_type=EventType.LEADER_CLAIM,
                        sender_id=agent.agent_id,
                    )
                    self.signal_cooldown = self.genome.signal_cooldown

        return (vx, vy), event

    def interpret_signal(self, event: Event, agent) -> Optional[str]:
        """Interpret signal using evolved meanings."""
        meaning = self.genome.signal_meanings.get(event.event_type)
        if meaning:
            return meaning.action
        return None


class EvolvedMultiAgentController(MultiAgentStatechartController):
    """Controller using evolved statecharts."""

    def __init__(self, n_agents: int, genome: ProtocolGenome):
        self.genome = genome
        self.statecharts = {
            i: EvolvedAgentStatechart(agent_id=i, genome=genome)
            for i in range(n_agents)
        }


# =============================================================================
# Protocol Evolver
# =============================================================================

class ProtocolEvolver:
    """
    Evolves communication protocols through co-evolution.

    All agents in a team share the same protocol genome.
    Teams compete on coordination tasks.
    """

    def __init__(
        self,
        n_agents: int = 4,
        population_size: int = 20,
        mutation_rate: float = 0.3,
    ):
        self.n_agents = n_agents
        self.population_size = population_size
        self.mutation_rate = mutation_rate

        # Population of protocols
        self.population: List[ProtocolGenome] = []

        # Environment
        self.env = MultiAgentEnv(n_agents=n_agents)

        # Statistics
        self.generation = 0
        self.best_fitness = 0.0
        self.fitness_history: List[float] = []

    def _random_genome(self) -> ProtocolGenome:
        """Create random protocol genome."""
        genome = ProtocolGenome()

        # Random signal meanings
        actions = ["MOVE_TO_SENDER", "MOVE_TO_TARGET", "WAIT", "FOLLOW", "EXPLORE"]
        for signal in [EventType.SIGNAL_A, EventType.SIGNAL_B,
                       EventType.SIGNAL_C, EventType.SIGNAL_D]:
            genome.signal_meanings[signal] = SignalMeaning(
                signal_type=signal,
                action=random.choice(actions),
                priority=random.random(),
            )

        # Random role conditions
        for role in [AgentRole.LEADER, AgentRole.FOLLOWER, AgentRole.SCOUT]:
            genome.role_conditions.append(RoleCondition(
                role=role,
                condition=random.choice(["FIRST_TO_SIGNAL", "MOST_CENTRAL", "RANDOM"]),
                threshold=random.random(),
            ))

        # Random transition probabilities
        modes = list(BehaviorMode)
        for m1 in modes:
            for m2 in modes:
                if m1 != m2 and random.random() < 0.3:
                    genome.transition_probs[(m1, m2)] = random.random()

        # Random timing
        genome.signal_cooldown = random.uniform(0.2, 1.0)
        genome.leader_claim_prob = random.uniform(0.05, 0.3)
        genome.follow_threshold = random.uniform(0.1, 0.5)

        return genome

    def _mutate(self, genome: ProtocolGenome) -> ProtocolGenome:
        """Mutate protocol genome."""
        g = genome.clone()

        # Mutate signal meanings
        if random.random() < self.mutation_rate:
            signals = list(g.signal_meanings.keys())
            if signals:
                signal = random.choice(signals)
                actions = ["MOVE_TO_SENDER", "MOVE_TO_TARGET", "WAIT", "FOLLOW", "EXPLORE"]
                g.signal_meanings[signal].action = random.choice(actions)

        # Mutate timing
        if random.random() < self.mutation_rate:
            g.signal_cooldown = max(0.1, g.signal_cooldown + random.gauss(0, 0.1))

        if random.random() < self.mutation_rate:
            g.leader_claim_prob = max(0.01, min(0.5,
                g.leader_claim_prob + random.gauss(0, 0.05)))

        if random.random() < self.mutation_rate:
            g.follow_threshold = max(0.05, min(0.8,
                g.follow_threshold + random.gauss(0, 0.1)))

        # Mutate transition probabilities
        if random.random() < self.mutation_rate:
            modes = list(BehaviorMode)
            m1, m2 = random.choice(modes), random.choice(modes)
            if m1 != m2:
                current = g.transition_probs.get((m1, m2), 0.5)
                g.transition_probs[(m1, m2)] = max(0, min(1,
                    current + random.gauss(0, 0.1)))

        return g

    def _crossover(
        self,
        parent1: ProtocolGenome,
        parent2: ProtocolGenome
    ) -> ProtocolGenome:
        """Crossover two protocol genomes."""
        child = parent1.clone()

        # Crossover signal meanings
        for signal in parent2.signal_meanings:
            if random.random() < 0.5:
                child.signal_meanings[signal] = copy.deepcopy(
                    parent2.signal_meanings[signal]
                )

        # Crossover timing (blend)
        alpha = random.random()
        child.signal_cooldown = (
            alpha * parent1.signal_cooldown +
            (1 - alpha) * parent2.signal_cooldown
        )
        child.leader_claim_prob = (
            alpha * parent1.leader_claim_prob +
            (1 - alpha) * parent2.leader_claim_prob
        )

        return child

    def _evaluate(
        self,
        genome: ProtocolGenome,
        tasks: List[Task],
        n_episodes: int = 3,
    ) -> float:
        """Evaluate protocol on coordination tasks."""
        total_reward = 0.0
        total_success = 0

        controller = EvolvedMultiAgentController(self.n_agents, genome)

        for task in tasks:
            for _ in range(n_episodes):
                obs = self.env.reset(task)
                controller.reset()
                episode_reward = 0.0

                for step in range(200):
                    actions = controller.step(obs, dt=self.env.dt)
                    obs, rewards, done, info = self.env.step(actions)
                    episode_reward += sum(rewards.values())

                    if done:
                        if info.get("task_complete", False):
                            total_success += 1
                            episode_reward += 50  # Bonus for completion
                        break

                total_reward += episode_reward

        n_total = len(tasks) * n_episodes
        return total_reward / n_total + (total_success / n_total) * 10

    def evolve(
        self,
        tasks: List[Task],
        n_generations: int = 50,
        verbose: bool = True,
    ) -> ProtocolGenome:
        """Evolve communication protocols."""
        # Initialize population
        if not self.population:
            self.population = [
                self._random_genome()
                for _ in range(self.population_size)
            ]

        best_genome = None

        for gen in range(n_generations):
            self.generation = gen

            # Evaluate population
            for genome in self.population:
                genome.fitness = self._evaluate(genome, tasks)

                if genome.fitness > self.best_fitness:
                    self.best_fitness = genome.fitness
                    best_genome = genome.clone()

            self.fitness_history.append(self.best_fitness)

            if verbose and gen % 5 == 0:
                avg_fitness = sum(g.fitness for g in self.population) / len(self.population)
                print(f"Gen {gen}: best={self.best_fitness:.2f}, avg={avg_fitness:.2f}")

            # Selection and reproduction
            self.population.sort(key=lambda g: -g.fitness)

            next_pop = [self.population[0].clone()]  # Elitism

            while len(next_pop) < self.population_size:
                if random.random() < 0.7:
                    # Mutation
                    parent = random.choice(self.population[:self.population_size // 2])
                    child = self._mutate(parent)
                else:
                    # Crossover
                    p1 = random.choice(self.population[:self.population_size // 2])
                    p2 = random.choice(self.population[:self.population_size // 2])
                    child = self._crossover(p1, p2)

                next_pop.append(child)

            self.population = next_pop

        return best_genome

    def analyze_protocol(self, genome: ProtocolGenome) -> Dict[str, Any]:
        """Analyze evolved protocol."""
        analysis = {
            "signal_meanings": {},
            "timing": {
                "signal_cooldown": genome.signal_cooldown,
                "leader_claim_prob": genome.leader_claim_prob,
                "follow_threshold": genome.follow_threshold,
            },
            "transition_probs": {},
        }

        for signal, meaning in genome.signal_meanings.items():
            analysis["signal_meanings"][signal.name] = meaning.action

        for (m1, m2), prob in genome.transition_probs.items():
            if prob > 0.1:
                analysis["transition_probs"][f"{m1.name}->{m2.name}"] = prob

        return analysis


# =============================================================================
# Demo
# =============================================================================

def demo():
    """Demonstrate protocol evolution."""
    print("=" * 60)
    print("Protocol Evolution for Multi-Agent Coordination")
    print("=" * 60)

    # Create tasks
    tasks = [
        Task(TaskType.GATHER, 0.5, 0.5, {"threshold": 0.15}),
        Task(TaskType.COVERAGE, 0.5, 0.5, {"min_distance": 0.25}),
    ]

    # Create evolver
    evolver = ProtocolEvolver(
        n_agents=4,
        population_size=15,
    )

    print(f"\nEvolving protocols for {len(tasks)} tasks...")
    best = evolver.evolve(tasks, n_generations=20, verbose=True)

    print(f"\n--- Best Protocol ---")
    analysis = evolver.analyze_protocol(best)

    print(f"Signal meanings:")
    for signal, action in analysis["signal_meanings"].items():
        print(f"  {signal}: {action}")

    print(f"\nTiming:")
    for param, value in analysis["timing"].items():
        print(f"  {param}: {value:.3f}")

    print(f"\nKey transitions:")
    for trans, prob in list(analysis["transition_probs"].items())[:5]:
        print(f"  {trans}: {prob:.2f}")

    print(f"\nBest fitness: {best.fitness:.2f}")

    return evolver, best


if __name__ == "__main__":
    demo()
