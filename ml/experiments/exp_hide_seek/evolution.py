"""
Hide-and-Seek Evolution System

Co-evolves seekers and hiders through competitive self-play.
Key: Both populations improve against each other, leading to emergent strategies.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any, Optional
import random
import json
from datetime import datetime

try:
    from .grid_world import GridWorld, Direction
    from .seeker_statechart import SeekerStatechart, SeekerGenome
    from .hider_statechart import HiderStatechart, HiderGenome
except ImportError:
    from grid_world import GridWorld, Direction
    from seeker_statechart import SeekerStatechart, SeekerGenome
    from hider_statechart import HiderStatechart, HiderGenome


@dataclass
class EvolutionConfig:
    """Configuration for the evolution process."""
    population_size: int = 20
    num_generations: int = 50
    games_per_evaluation: int = 10
    tournament_size: int = 3
    elite_count: int = 2
    mutation_rate: float = 0.2
    crossover_rate: float = 0.7

    # Arena settings
    arena_width: int = 15
    arena_height: int = 15
    num_walls: int = 8
    num_boxes: int = 4
    num_shelters: int = 3
    max_steps: int = 200


@dataclass
class Individual:
    """An individual in the population."""
    genome: Any  # SeekerGenome or HiderGenome
    fitness: float = 0.0
    games_played: int = 0
    wins: int = 0
    total_reward: float = 0.0


@dataclass
class EvolutionStats:
    """Statistics from evolution."""
    generation: int
    seeker_best_fitness: float
    seeker_avg_fitness: float
    hider_best_fitness: float
    hider_avg_fitness: float
    avg_game_length: float
    seeker_win_rate: float


class HideSeekEvolution:
    """
    Co-evolution system for hide-and-seek.

    Uses competitive co-evolution where:
    - Seekers evolve to find hiders better
    - Hiders evolve to avoid seekers better
    - Both populations drive each other's improvement
    """

    def __init__(self, config: EvolutionConfig = None):
        self.config = config or EvolutionConfig()
        self.seeker_population: List[Individual] = []
        self.hider_population: List[Individual] = []
        self.history: List[EvolutionStats] = []

    def initialize_populations(self):
        """Create initial random populations."""
        self.seeker_population = [
            Individual(genome=SeekerGenome())
            for _ in range(self.config.population_size)
        ]
        self.hider_population = [
            Individual(genome=HiderGenome())
            for _ in range(self.config.population_size)
        ]

        # Apply random mutations to create diversity
        for ind in self.seeker_population:
            for _ in range(3):  # Multiple mutations for variety
                ind.genome = ind.genome.mutate(rate=0.5)
        for ind in self.hider_population:
            for _ in range(3):
                ind.genome = ind.genome.mutate(rate=0.5)

    def run_game(self, seeker_genome: SeekerGenome,
                 hider_genome: HiderGenome) -> Tuple[float, float, int]:
        """
        Run a single game between seeker and hider.

        Returns: (seeker_reward, hider_reward, game_length)
        """
        # Create arena
        world = GridWorld.create_arena(
            width=self.config.arena_width,
            height=self.config.arena_height,
            num_walls=self.config.num_walls,
            num_boxes=self.config.num_boxes,
            num_shelters=self.config.num_shelters,
        )
        world.max_steps = self.config.max_steps
        world.spawn_agents(num_seekers=1, num_hiders=1)

        # Create statecharts
        seeker = SeekerStatechart(seeker_genome)
        hider = HiderStatechart(hider_genome)

        # Run game
        total_seeker_reward = 0.0
        total_hider_reward = 0.0
        seeker_memory = []
        hider_memory = []

        while True:
            # Get observations
            seeker_obs = world.get_observation(world.seekers[0], seeker_memory)
            hider_obs = world.get_observation(world.hiders[0], hider_memory)

            # Get actions from statecharts
            seeker_action = seeker.decide(seeker_obs)
            hider_action = hider.decide(hider_obs)

            # Update memories
            if seeker_obs.visible_agents:
                seeker_memory.append(
                    (seeker_obs.visible_agents[0].x, seeker_obs.visible_agents[0].y)
                )
            if hider_obs.visible_agents:
                hider_memory.append(
                    (hider_obs.visible_agents[0].x, hider_obs.visible_agents[0].y)
                )
            # Keep memory limited
            seeker_memory = seeker_memory[-10:]
            hider_memory = hider_memory[-10:]

            # Execute step
            game_over, seeker_r, hider_r = world.step(
                [seeker_action], [hider_action]
            )
            total_seeker_reward += seeker_r
            total_hider_reward += hider_r

            if game_over:
                break

        return total_seeker_reward, total_hider_reward, world.time_step

    def evaluate_population(self):
        """Evaluate all individuals through round-robin games."""
        # Reset fitness
        for ind in self.seeker_population:
            ind.fitness = 0.0
            ind.games_played = 0
            ind.wins = 0
            ind.total_reward = 0.0
        for ind in self.hider_population:
            ind.fitness = 0.0
            ind.games_played = 0
            ind.wins = 0
            ind.total_reward = 0.0

        total_game_length = 0
        total_games = 0
        seeker_wins = 0

        # Each seeker plays against random hiders
        for seeker in self.seeker_population:
            opponents = random.sample(
                self.hider_population,
                min(self.config.games_per_evaluation, len(self.hider_population))
            )
            for hider in opponents:
                seeker_r, hider_r, length = self.run_game(
                    seeker.genome, hider.genome
                )

                # Update seeker stats
                seeker.games_played += 1
                seeker.total_reward += seeker_r
                if seeker_r > hider_r:
                    seeker.wins += 1
                    seeker_wins += 1

                # Update hider stats
                hider.games_played += 1
                hider.total_reward += hider_r
                if hider_r > seeker_r:
                    hider.wins += 1

                total_game_length += length
                total_games += 1

        # Calculate fitness
        for ind in self.seeker_population:
            if ind.games_played > 0:
                ind.fitness = ind.total_reward / ind.games_played
        for ind in self.hider_population:
            if ind.games_played > 0:
                ind.fitness = ind.total_reward / ind.games_played

        return total_game_length / max(1, total_games), seeker_wins / max(1, total_games)

    def tournament_select(self, population: List[Individual]) -> Individual:
        """Select individual using tournament selection."""
        tournament = random.sample(population, self.config.tournament_size)
        return max(tournament, key=lambda ind: ind.fitness)

    def evolve_population(self, population: List[Individual],
                          genome_class) -> List[Individual]:
        """Evolve a population using genetic operators."""
        # Sort by fitness
        sorted_pop = sorted(population, key=lambda ind: ind.fitness, reverse=True)

        # Keep elites
        new_population = [
            Individual(genome=sorted_pop[i].genome)
            for i in range(min(self.config.elite_count, len(sorted_pop)))
        ]

        # Fill rest through selection and reproduction
        while len(new_population) < self.config.population_size:
            if random.random() < self.config.crossover_rate:
                # Crossover
                parent1 = self.tournament_select(population)
                parent2 = self.tournament_select(population)
                child_genome = genome_class.crossover(
                    parent1.genome, parent2.genome
                )
            else:
                # Clone
                parent = self.tournament_select(population)
                child_genome = parent.genome

            # Mutate
            child_genome = child_genome.mutate(rate=self.config.mutation_rate)
            new_population.append(Individual(genome=child_genome))

        return new_population

    def run_evolution(self, verbose: bool = True) -> Dict[str, Any]:
        """
        Run the full evolution process.

        Returns statistics and best strategies.
        """
        self.initialize_populations()

        for gen in range(self.config.num_generations):
            # Evaluate
            avg_length, seeker_win_rate = self.evaluate_population()

            # Collect stats
            seeker_fitnesses = [ind.fitness for ind in self.seeker_population]
            hider_fitnesses = [ind.fitness for ind in self.hider_population]

            stats = EvolutionStats(
                generation=gen + 1,
                seeker_best_fitness=max(seeker_fitnesses),
                seeker_avg_fitness=sum(seeker_fitnesses) / len(seeker_fitnesses),
                hider_best_fitness=max(hider_fitnesses),
                hider_avg_fitness=sum(hider_fitnesses) / len(hider_fitnesses),
                avg_game_length=avg_length,
                seeker_win_rate=seeker_win_rate,
            )
            self.history.append(stats)

            if verbose:
                print(f"Gen {gen + 1:3d} | "
                      f"Seeker: best={stats.seeker_best_fitness:+.2f} avg={stats.seeker_avg_fitness:+.2f} | "
                      f"Hider: best={stats.hider_best_fitness:+.2f} avg={stats.hider_avg_fitness:+.2f} | "
                      f"Win%={seeker_win_rate*100:.1f} Len={avg_length:.0f}")

            # Evolve
            self.seeker_population = self.evolve_population(
                self.seeker_population, SeekerGenome
            )
            self.hider_population = self.evolve_population(
                self.hider_population, HiderGenome
            )

        # Extract best strategies
        best_seeker = max(self.seeker_population, key=lambda ind: ind.fitness)
        best_hider = max(self.hider_population, key=lambda ind: ind.fitness)

        seeker_statechart = SeekerStatechart(best_seeker.genome)
        hider_statechart = HiderStatechart(best_hider.genome)

        return {
            'timestamp': datetime.now().isoformat(),
            'config': {
                'population_size': self.config.population_size,
                'num_generations': self.config.num_generations,
                'games_per_evaluation': self.config.games_per_evaluation,
            },
            'final_stats': {
                'seeker_best_fitness': best_seeker.fitness,
                'hider_best_fitness': best_hider.fitness,
                'seeker_win_rate': self.history[-1].seeker_win_rate if self.history else 0,
            },
            'history': [
                {
                    'generation': s.generation,
                    'seeker_best': s.seeker_best_fitness,
                    'seeker_avg': s.seeker_avg_fitness,
                    'hider_best': s.hider_best_fitness,
                    'hider_avg': s.hider_avg_fitness,
                    'seeker_win_rate': s.seeker_win_rate,
                    'avg_game_length': s.avg_game_length,
                }
                for s in self.history
            ],
            'emerged_strategies': {
                'seeker': seeker_statechart.extract_strategy(),
                'hider': hider_statechart.extract_strategy(),
            },
        }

    def analyze_emergence(self) -> Dict[str, Any]:
        """
        Analyze what strategies emerged from evolution.

        This is the KEY INSIGHT - we can see WHAT the agents learned!
        """
        if not self.seeker_population or not self.hider_population:
            return {'error': 'No populations to analyze'}

        # Analyze seeker population
        seeker_traits = {
            'search_spiral': 0,
            'search_wall_follow': 0,
            'chase_intercept': 0,
            'predict_use_shelter': 0,
            'predict_use_corners': 0,
            'predict_use_history': 0,
        }
        for ind in self.seeker_population:
            g = ind.genome
            seeker_traits['search_spiral'] += 1 if g.search_spiral else 0
            seeker_traits['search_wall_follow'] += 1 if g.search_wall_follow else 0
            seeker_traits['chase_intercept'] += 1 if g.chase_intercept else 0
            seeker_traits['predict_use_shelter'] += 1 if g.predict_use_shelter else 0
            seeker_traits['predict_use_corners'] += 1 if g.predict_use_corners else 0
            seeker_traits['predict_use_history'] += 1 if g.predict_use_history else 0

        # Normalize
        n = len(self.seeker_population)
        seeker_traits = {k: v/n for k, v in seeker_traits.items()}

        # Analyze hider population
        hider_traits = {
            'hide_prefer_shelter': 0,
            'hide_prefer_corners': 0,
            'evade_away': 0,
            'evade_perpendicular': 0,
            'evade_juke': 0,
            'distract_reverse': 0,
        }
        for ind in self.hider_population:
            g = ind.genome
            hider_traits['hide_prefer_shelter'] += 1 if g.hide_prefer_shelter else 0
            hider_traits['hide_prefer_corners'] += 1 if g.hide_prefer_corners else 0
            hider_traits['evade_away'] += 1 if g.evade_away else 0
            hider_traits['evade_perpendicular'] += 1 if g.evade_perpendicular else 0
            hider_traits['evade_juke'] += 1 if g.evade_juke else 0
            hider_traits['distract_reverse'] += 1 if g.distract_reverse else 0

        # Normalize
        m = len(self.hider_population)
        hider_traits = {k: v/m for k, v in hider_traits.items()}

        # Identify dominant strategies
        seeker_dominant = [k for k, v in seeker_traits.items() if v > 0.5]
        hider_dominant = [k for k, v in hider_traits.items() if v > 0.5]

        return {
            'seeker_traits': seeker_traits,
            'hider_traits': hider_traits,
            'seeker_dominant_strategies': seeker_dominant,
            'hider_dominant_strategies': hider_dominant,
            'emergent_behaviors': {
                'seekers_use_prediction': seeker_traits['predict_use_history'] > 0.3,
                'hiders_use_distraction': any(
                    ind.genome.distract_chance > 0.15 for ind in self.hider_population
                ),
                'arms_race_observed': (
                    seeker_traits['chase_intercept'] > 0.3 and
                    hider_traits['evade_juke'] > 0.3
                ),
            },
        }


def demonstrate_game(seeker_genome: SeekerGenome = None,
                     hider_genome: HiderGenome = None,
                     max_steps: int = 50) -> str:
    """
    Run a demonstration game with visual output.

    Useful for observing learned behaviors.
    """
    # Create arena
    world = GridWorld.create_arena(
        width=15, height=15,
        num_walls=8, num_boxes=4, num_shelters=3,
    )
    world.max_steps = max_steps
    world.spawn_agents(num_seekers=1, num_hiders=1)

    # Create statecharts
    seeker = SeekerStatechart(seeker_genome or SeekerGenome())
    hider = HiderStatechart(hider_genome or HiderGenome())

    frames = []
    frames.append(f"=== Initial State ===\n{world.render()}\n")

    seeker_memory = []
    hider_memory = []

    while world.time_step < max_steps:
        # Get observations
        seeker_obs = world.get_observation(world.seekers[0], seeker_memory)
        hider_obs = world.get_observation(world.hiders[0], hider_memory)

        # Get actions
        seeker_action = seeker.decide(seeker_obs)
        hider_action = hider.decide(hider_obs)

        # Update memories
        if seeker_obs.visible_agents:
            seeker_memory.append(
                (seeker_obs.visible_agents[0].x, seeker_obs.visible_agents[0].y)
            )
        seeker_memory = seeker_memory[-10:]

        # Execute step
        game_over, seeker_r, hider_r = world.step([seeker_action], [hider_action])

        # Record frame periodically
        if world.time_step % 10 == 0 or game_over:
            frame = f"=== Step {world.time_step} ===\n"
            frame += f"Seeker: {seeker.state.mode.name} -> {seeker_action.name}\n"
            frame += f"Hider: {hider.state.mode.name} -> {hider_action.name}\n"
            frame += world.render() + "\n"
            frames.append(frame)

        if game_over:
            if seeker_r > hider_r:
                frames.append("*** SEEKER WINS! ***")
            else:
                frames.append("*** HIDER SURVIVES! ***")
            break

    return '\n'.join(frames)
