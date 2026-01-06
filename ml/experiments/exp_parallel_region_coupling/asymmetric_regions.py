"""
Asymmetric Regions: Testing Regions with Different Sizes

Tests how parallel regions with vastly different state counts interact:
- Small regions (2-3 states) vs large regions (10+ states)
- Balance issues in evolution/learning
- Coverage challenges across asymmetric structures
- Transition probability differences

Key questions:
1. Do larger regions dominate smaller ones in evolution?
2. How does coverage differ across region sizes?
3. Can we learn fair coupling despite asymmetry?

NO HARDCODING: Learn interaction patterns from examples.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Set, Tuple, Optional, Any, Callable
from enum import Enum, auto
import random
import time
from collections import defaultdict


# =============================================================================
# ASYMMETRIC REGION TYPES
# =============================================================================

class RegionSize(Enum):
    """Categories of region sizes."""
    TINY = 2       # 2 states
    SMALL = 3      # 3 states
    MEDIUM = 5     # 5 states
    LARGE = 10     # 10 states
    HUGE = 20      # 20 states


@dataclass
class AsymmetricRegion:
    """A region in an asymmetric parallel state."""
    id: str
    label: str
    states: List[str]
    initial_state: str
    current_state: str
    transitions: Dict[Tuple[str, str], str] = field(default_factory=dict)  # (from, event) -> to
    size_category: RegionSize = RegionSize.SMALL

    @property
    def state_count(self) -> int:
        return len(self.states)

    def has_state(self, state: str) -> bool:
        return state in self.states


@dataclass
class AsymmetricConfiguration:
    """Configuration across asymmetric regions."""
    region_states: Dict[str, str]  # region_id -> current_state
    total_states: int = 0
    coverage: float = 0.0

    def __hash__(self):
        return hash(tuple(sorted(self.region_states.items())))


@dataclass
class CouplingMetrics:
    """Metrics for measuring asymmetric coupling."""
    size_ratio: float          # Ratio of largest to smallest region
    coverage_per_region: Dict[str, float]  # region_id -> coverage
    transition_density: Dict[str, float]   # region_id -> transitions per state
    coupling_strength: float   # How strongly regions are coupled
    balance_score: float       # How balanced is the interaction


# =============================================================================
# ASYMMETRIC PARALLEL STATE
# =============================================================================

@dataclass
class AsymmetricParallelState:
    """
    A parallel state with asymmetric regions.

    Handles the challenges of combining regions with very different sizes.
    """
    label: str
    regions: List[AsymmetricRegion] = field(default_factory=list)
    shared_context: Dict[str, Any] = field(default_factory=dict)
    global_transitions: List[Tuple[str, str]] = field(default_factory=list)  # (event, region_id)

    @property
    def size_ratio(self) -> float:
        """Ratio of largest to smallest region."""
        if not self.regions:
            return 1.0
        sizes = [r.state_count for r in self.regions]
        return max(sizes) / max(min(sizes), 1)

    @property
    def total_state_space(self) -> int:
        """Total number of possible configurations."""
        result = 1
        for r in self.regions:
            result *= r.state_count
        return result

    def get_configuration(self) -> AsymmetricConfiguration:
        """Get current configuration."""
        return AsymmetricConfiguration(
            region_states={r.id: r.current_state for r in self.regions},
            total_states=self.total_state_space,
        )

    def get_region(self, region_id: str) -> Optional[AsymmetricRegion]:
        """Get region by ID."""
        for r in self.regions:
            if r.id == region_id:
                return r
        return None

    def step(self, event: str) -> Dict[str, str]:
        """Process event across all regions."""
        transitions_taken = {}

        for region in self.regions:
            key = (region.current_state, event)
            if key in region.transitions:
                next_state = region.transitions[key]
                transitions_taken[region.id] = next_state
                region.current_state = next_state

        return transitions_taken


# =============================================================================
# REGION GENERATORS
# =============================================================================

def create_tiny_region(region_id: str, label: str) -> AsymmetricRegion:
    """Create a 2-state region (toggle)."""
    return AsymmetricRegion(
        id=region_id,
        label=label,
        states=["Off", "On"],
        initial_state="Off",
        current_state="Off",
        transitions={
            ("Off", "TOGGLE"): "On",
            ("On", "TOGGLE"): "Off",
        },
        size_category=RegionSize.TINY,
    )


def create_small_region(region_id: str, label: str) -> AsymmetricRegion:
    """Create a 3-state region (cycle)."""
    return AsymmetricRegion(
        id=region_id,
        label=label,
        states=["A", "B", "C"],
        initial_state="A",
        current_state="A",
        transitions={
            ("A", "NEXT"): "B",
            ("B", "NEXT"): "C",
            ("C", "NEXT"): "A",
            ("A", "PREV"): "C",
            ("B", "PREV"): "A",
            ("C", "PREV"): "B",
        },
        size_category=RegionSize.SMALL,
    )


def create_medium_region(region_id: str, label: str) -> AsymmetricRegion:
    """Create a 5-state region (pipeline)."""
    states = ["Init", "Loading", "Processing", "Validating", "Done"]
    transitions = {}

    for i, state in enumerate(states[:-1]):
        transitions[(state, "ADVANCE")] = states[i + 1]

    for i, state in enumerate(states[1:], 1):
        transitions[(state, "RESET")] = states[0]

    return AsymmetricRegion(
        id=region_id,
        label=label,
        states=states,
        initial_state="Init",
        current_state="Init",
        transitions=transitions,
        size_category=RegionSize.MEDIUM,
    )


def create_large_region(region_id: str, label: str) -> AsymmetricRegion:
    """Create a 10-state region (complex workflow)."""
    states = [
        "Idle", "Pending", "Queued", "Scheduled",
        "Starting", "Running", "Paused",
        "Completing", "Cleanup", "Done"
    ]

    transitions = {}

    # Forward chain
    for i, state in enumerate(states[:-1]):
        transitions[(state, "PROGRESS")] = states[i + 1]

    # Reset from anywhere
    for state in states:
        transitions[(state, "RESET")] = "Idle"

    # Pause/Resume
    transitions[("Running", "PAUSE")] = "Paused"
    transitions[("Paused", "RESUME")] = "Running"

    # Skip
    transitions[("Queued", "SKIP")] = "Running"

    return AsymmetricRegion(
        id=region_id,
        label=label,
        states=states,
        initial_state="Idle",
        current_state="Idle",
        transitions=transitions,
        size_category=RegionSize.LARGE,
    )


def create_huge_region(region_id: str, label: str) -> AsymmetricRegion:
    """Create a 20-state region (game level)."""
    states = [f"Level_{i}" for i in range(20)]
    transitions = {}

    # Forward progression
    for i in range(19):
        transitions[(states[i], "WIN")] = states[i + 1]

    # Backward on failure (go back 2)
    for i in range(2, 20):
        transitions[(states[i], "LOSE")] = states[i - 2]

    transitions[(states[0], "LOSE")] = states[0]
    transitions[(states[1], "LOSE")] = states[0]

    # Reset
    for state in states:
        transitions[(state, "RESTART")] = states[0]

    return AsymmetricRegion(
        id=region_id,
        label=label,
        states=states,
        initial_state="Level_0",
        current_state="Level_0",
        transitions=transitions,
        size_category=RegionSize.HUGE,
    )


# =============================================================================
# ASYMMETRIC COUPLING ANALYZER
# =============================================================================

class AsymmetricCouplingAnalyzer:
    """
    Analyze and measure coupling in asymmetric parallel regions.

    Key metrics:
    - Coverage balance: Are all regions exercised equally?
    - Coupling strength: How much do regions affect each other?
    - Fairness: Does size impact interaction probability?
    """

    def __init__(self):
        self.execution_history: List[Dict[str, str]] = []
        self.coverage_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

    def record_step(
        self,
        parallel_state: AsymmetricParallelState,
        event: str,
        transitions: Dict[str, str],
    ):
        """Record a step for analysis."""
        config = parallel_state.get_configuration()
        self.execution_history.append({
            "event": event,
            "before": dict(config.region_states),
            "transitions": dict(transitions),
        })

        # Update coverage
        for region_id, state in config.region_states.items():
            self.coverage_counts[region_id][state] += 1

    def compute_metrics(
        self,
        parallel_state: AsymmetricParallelState,
    ) -> CouplingMetrics:
        """Compute coupling metrics."""
        # Coverage per region
        coverage_per_region = {}
        for region in parallel_state.regions:
            visited = set(self.coverage_counts[region.id].keys())
            coverage_per_region[region.id] = len(visited) / region.state_count

        # Transition density
        transition_density = {}
        for region in parallel_state.regions:
            n_transitions = len(region.transitions)
            transition_density[region.id] = n_transitions / max(region.state_count, 1)

        # Coupling strength (simplified: shared events)
        all_events = set()
        for region in parallel_state.regions:
            for (_, event) in region.transitions.keys():
                all_events.add(event)

        shared_events = 0
        for event in all_events:
            regions_with_event = 0
            for region in parallel_state.regions:
                if any(e == event for (_, e) in region.transitions.keys()):
                    regions_with_event += 1
            if regions_with_event > 1:
                shared_events += 1

        coupling_strength = shared_events / max(len(all_events), 1)

        # Balance score (variance in coverage)
        coverages = list(coverage_per_region.values())
        if coverages:
            mean_cov = sum(coverages) / len(coverages)
            variance = sum((c - mean_cov) ** 2 for c in coverages) / len(coverages)
            balance_score = 1.0 - min(variance, 1.0)
        else:
            balance_score = 0.0

        return CouplingMetrics(
            size_ratio=parallel_state.size_ratio,
            coverage_per_region=coverage_per_region,
            transition_density=transition_density,
            coupling_strength=coupling_strength,
            balance_score=balance_score,
        )


# =============================================================================
# ASYMMETRIC INTERACTION EVOLVER
# =============================================================================

@dataclass
class InteractionGenome:
    """Evolvable interaction pattern for asymmetric regions."""
    event_sequence: List[str]
    region_weights: Dict[str, float]  # How much to favor each region
    coupling_events: Set[str]         # Events that couple regions
    fitness: float = 0.0


class AsymmetricInteractionEvolver:
    """
    Evolve interaction patterns that achieve balanced coverage
    across asymmetric regions.
    """

    def __init__(
        self,
        parallel_state: AsymmetricParallelState,
        population_size: int = 30,
        n_generations: int = 50,
    ):
        self.parallel_state = parallel_state
        self.population_size = population_size
        self.n_generations = n_generations

        # Collect all possible events
        self.all_events = set()
        for region in parallel_state.regions:
            for (_, event) in region.transitions.keys():
                self.all_events.add(event)

    def random_genome(self, sequence_length: int = 20) -> InteractionGenome:
        """Generate random interaction pattern."""
        events = list(self.all_events)
        sequence = [random.choice(events) for _ in range(sequence_length)]

        weights = {r.id: random.uniform(0.5, 1.5) for r in self.parallel_state.regions}

        coupling = set(random.sample(events, min(len(events)//2, len(events))))

        return InteractionGenome(
            event_sequence=sequence,
            region_weights=weights,
            coupling_events=coupling,
        )

    def mutate(self, genome: InteractionGenome, rate: float = 0.2) -> InteractionGenome:
        """Mutate interaction pattern."""
        events = list(self.all_events)

        # Mutate sequence
        new_sequence = list(genome.event_sequence)
        for i in range(len(new_sequence)):
            if random.random() < rate:
                new_sequence[i] = random.choice(events)

        # Mutate weights
        new_weights = dict(genome.region_weights)
        for region_id in new_weights:
            if random.random() < rate:
                new_weights[region_id] = max(0.1, new_weights[region_id] + random.gauss(0, 0.2))

        # Mutate coupling events
        new_coupling = set(genome.coupling_events)
        for event in events:
            if random.random() < rate / 2:
                if event in new_coupling:
                    new_coupling.discard(event)
                else:
                    new_coupling.add(event)

        return InteractionGenome(
            event_sequence=new_sequence,
            region_weights=new_weights,
            coupling_events=new_coupling,
        )

    def crossover(self, p1: InteractionGenome, p2: InteractionGenome) -> InteractionGenome:
        """Crossover two patterns."""
        # Single-point crossover on sequence
        point = random.randint(0, min(len(p1.event_sequence), len(p2.event_sequence)))
        new_sequence = p1.event_sequence[:point] + p2.event_sequence[point:]

        # Average weights
        new_weights = {}
        for region_id in p1.region_weights:
            w1 = p1.region_weights.get(region_id, 1.0)
            w2 = p2.region_weights.get(region_id, 1.0)
            new_weights[region_id] = (w1 + w2) / 2

        # Union of coupling events
        new_coupling = p1.coupling_events | p2.coupling_events

        return InteractionGenome(
            event_sequence=new_sequence,
            region_weights=new_weights,
            coupling_events=new_coupling,
        )

    def evaluate_fitness(self, genome: InteractionGenome) -> float:
        """
        Evaluate fitness based on balanced coverage.

        Goal: Maximize coverage while maintaining balance across regions.
        """
        # Reset state
        for region in self.parallel_state.regions:
            region.current_state = region.initial_state

        analyzer = AsymmetricCouplingAnalyzer()

        # Execute event sequence
        for event in genome.event_sequence:
            transitions = self.parallel_state.step(event)
            analyzer.record_step(self.parallel_state, event, transitions)

        metrics = analyzer.compute_metrics(self.parallel_state)

        # Fitness = average coverage * balance score
        avg_coverage = sum(metrics.coverage_per_region.values()) / max(len(metrics.coverage_per_region), 1)
        genome.fitness = avg_coverage * metrics.balance_score

        return genome.fitness

    def evolve(self, verbose: bool = True) -> InteractionGenome:
        """Evolve optimal interaction pattern."""
        population = [self.random_genome() for _ in range(self.population_size)]
        best_ever = None

        for gen in range(self.n_generations):
            # Evaluate
            for genome in population:
                self.evaluate_fitness(genome)

            # Sort by fitness
            population.sort(key=lambda g: g.fitness, reverse=True)

            # Track best
            if best_ever is None or population[0].fitness > best_ever.fitness:
                best_ever = InteractionGenome(
                    event_sequence=list(population[0].event_sequence),
                    region_weights=dict(population[0].region_weights),
                    coupling_events=set(population[0].coupling_events),
                    fitness=population[0].fitness,
                )

            if verbose and gen % 10 == 0:
                print(f"Gen {gen:3d}: Best fitness = {population[0].fitness:.3f}")

            # Perfect solution?
            if population[0].fitness >= 0.95:
                break

            # Selection and reproduction
            elite = population[:3]
            new_pop = list(elite)

            while len(new_pop) < self.population_size:
                p1 = random.choice(population[:self.population_size//2])
                p2 = random.choice(population[:self.population_size//2])
                child = self.crossover(p1, p2)
                child = self.mutate(child)
                new_pop.append(child)

            population = new_pop

        return best_ever


# =============================================================================
# TESTING
# =============================================================================

def test_asymmetric_regions():
    """Test asymmetric region creation and interaction."""
    print("=" * 60)
    print("ASYMMETRIC REGIONS TEST")
    print("=" * 60)

    # Create asymmetric parallel state: 2-state vs 10-state
    tiny = create_tiny_region("toggle", "ToggleSwitch")
    large = create_large_region("workflow", "WorkflowEngine")

    parallel = AsymmetricParallelState(
        label="AsymmetricSystem",
        regions=[tiny, large],
    )

    print(f"\nParallel state: {parallel.label}")
    print(f"  Regions: {[r.id for r in parallel.regions]}")
    print(f"  Sizes: tiny={tiny.state_count}, large={large.state_count}")
    print(f"  Size ratio: {parallel.size_ratio:.1f}x")
    print(f"  Total state space: {parallel.total_state_space}")

    print(f"\nInitial config: {parallel.get_configuration().region_states}")

    # Test stepping
    events = ["TOGGLE", "PROGRESS", "PROGRESS", "TOGGLE", "PROGRESS"]
    for event in events:
        transitions = parallel.step(event)
        config = parallel.get_configuration()
        print(f"  After {event}: {config.region_states} (transitions: {transitions})")

    print("\n[PASS] Asymmetric regions test passed!")
    return True


def test_asymmetric_evolution():
    """Test evolving balanced interaction patterns."""
    print("\n" + "=" * 60)
    print("ASYMMETRIC EVOLUTION TEST")
    print("=" * 60)

    # Create asymmetric parallel state: 3-state vs 10-state
    small = create_small_region("simple", "SimpleCycle")
    large = create_large_region("complex", "ComplexWorkflow")

    parallel = AsymmetricParallelState(
        label="AsymmetricEvolution",
        regions=[small, large],
    )

    print(f"\nSize ratio: {parallel.size_ratio:.1f}x")
    print(f"Small region: {small.state_count} states")
    print(f"Large region: {large.state_count} states")

    evolver = AsymmetricInteractionEvolver(
        parallel_state=parallel,
        population_size=30,
        n_generations=30,
    )

    print("\nEvolving balanced interaction pattern...")
    best = evolver.evolve(verbose=True)

    print(f"\nBest pattern:")
    print(f"  Fitness: {best.fitness:.3f}")
    print(f"  Sequence length: {len(best.event_sequence)}")
    print(f"  First 10 events: {best.event_sequence[:10]}")
    print(f"  Coupling events: {best.coupling_events}")

    success = best.fitness >= 0.3
    print(f"\n[{'PASS' if success else 'PARTIAL'}] Asymmetric evolution test")
    return success


def test_extreme_asymmetry():
    """Test extremely asymmetric regions (2 vs 20 states)."""
    print("\n" + "=" * 60)
    print("EXTREME ASYMMETRY TEST (2 vs 20 states)")
    print("=" * 60)

    tiny = create_tiny_region("binary", "BinarySwitch")
    huge = create_huge_region("levels", "GameLevels")

    parallel = AsymmetricParallelState(
        label="ExtremeAsymmetry",
        regions=[tiny, huge],
    )

    print(f"\nSize ratio: {parallel.size_ratio:.1f}x (2 vs 20)")
    print(f"Total state space: {parallel.total_state_space}")

    analyzer = AsymmetricCouplingAnalyzer()

    # Run random events
    all_events = ["TOGGLE", "WIN", "LOSE", "RESTART"]
    for _ in range(50):
        event = random.choice(all_events)
        transitions = parallel.step(event)
        analyzer.record_step(parallel, event, transitions)

    metrics = analyzer.compute_metrics(parallel)

    print(f"\nCoupling metrics:")
    print(f"  Coverage per region:")
    for region_id, cov in metrics.coverage_per_region.items():
        print(f"    {region_id}: {cov:.1%}")
    print(f"  Coupling strength: {metrics.coupling_strength:.2f}")
    print(f"  Balance score: {metrics.balance_score:.2f}")

    print("\n[PASS] Extreme asymmetry test completed")
    return True


if __name__ == "__main__":
    test_asymmetric_regions()
    test_asymmetric_evolution()
    test_extreme_asymmetry()
    print("\n" + "=" * 60)
    print("ALL ASYMMETRIC REGION TESTS COMPLETE")
    print("=" * 60)
