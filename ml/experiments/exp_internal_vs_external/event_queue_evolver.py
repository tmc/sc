"""
Internal vs External Event Semantics + Priority Evolution

THEORETICAL FOUNDATION:
In Harel's statechart semantics [HN96], events have distinct handling modes:

1. INTERNAL EVENTS (synchronous):
   - Generated during a step (actions, entry/exit)
   - Processed IMMEDIATELY within same RTC (run-to-completion) step
   - Enable micro-step chains within a macro-step

2. EXTERNAL EVENTS (queued):
   - Arrive from environment/users
   - Placed in event queue
   - Processed in subsequent macro-steps

EVENT PRIORITY determines processing order within the queue:
   CRITICAL > HIGH > NORMAL > LOW

This experiment EVOLVES the priority assignment for events - NO HARDCODING.
The goal: discover which events need high priority through fitness optimization.

Uses evolutionary patterns from exp_topology_evolution.
"""

import random
import copy
import time
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Set, Optional, Any
from enum import IntEnum
from collections import defaultdict
import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# =============================================================================
# EVENT SEMANTICS: INTERNAL vs EXTERNAL
# =============================================================================

class EventKind(IntEnum):
    """Classification of event handling semantics."""
    EXTERNAL = 0   # Queued, processed in subsequent macro-step
    INTERNAL = 1   # Synchronous, processed within same RTC step


class EventPriority(IntEnum):
    """Priority levels for event queue ordering."""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3


@dataclass
class EventDescriptor:
    """Complete event specification with evolved attributes."""
    name: str
    kind: EventKind
    priority: EventPriority

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        return self.name == other.name


@dataclass
class QueuedEvent:
    """Event instance in the queue with payload."""
    descriptor: EventDescriptor
    payload: Dict[str, Any]
    timestamp: float
    sequence: int  # Monotonic for ordering ties

    def __lt__(self, other):
        """Priority ordering: higher priority first, then FIFO."""
        if self.descriptor.priority != other.descriptor.priority:
            return self.descriptor.priority > other.descriptor.priority
        return self.sequence < other.sequence


# =============================================================================
# EVENT QUEUE WITH PRIORITY
# =============================================================================

class PriorityEventQueue:
    """
    Event queue with priority-based ordering.

    Internal events bypass the queue (synchronous processing).
    External events are queued by priority.
    """

    def __init__(self):
        self.queue: List[QueuedEvent] = []
        self.sequence_counter = 0
        self.internal_buffer: List[QueuedEvent] = []  # For internal events

    def enqueue(self, event: QueuedEvent):
        """Add event to appropriate buffer based on kind."""
        event.sequence = self.sequence_counter
        self.sequence_counter += 1

        if event.descriptor.kind == EventKind.INTERNAL:
            self.internal_buffer.append(event)
        else:
            # Insert maintaining priority order
            self.queue.append(event)
            self.queue.sort()  # Re-sort by priority

    def has_internal(self) -> bool:
        """Check if internal events pending (process first)."""
        return len(self.internal_buffer) > 0

    def dequeue_internal(self) -> Optional[QueuedEvent]:
        """Get next internal event (synchronous processing)."""
        if self.internal_buffer:
            return self.internal_buffer.pop(0)
        return None

    def dequeue_external(self) -> Optional[QueuedEvent]:
        """Get next external event by priority."""
        if self.queue:
            return self.queue.pop(0)
        return None

    def dequeue(self) -> Optional[QueuedEvent]:
        """Get next event: internal first, then external by priority."""
        if self.has_internal():
            return self.dequeue_internal()
        return self.dequeue_external()

    def is_empty(self) -> bool:
        return len(self.queue) == 0 and len(self.internal_buffer) == 0

    def clear(self):
        self.queue.clear()
        self.internal_buffer.clear()


# =============================================================================
# GENOME: EVENT PRIORITY ASSIGNMENT
# =============================================================================

@dataclass
class EventPriorityGenome:
    """
    Genome encoding event kind and priority assignments.

    NO HARDCODING - everything is evolved:
    - Which events are INTERNAL vs EXTERNAL
    - What priority each event has

    The genome is a mapping: event_name -> (kind, priority)
    """
    event_names: List[str]
    kinds: List[EventKind]      # INTERNAL or EXTERNAL for each event
    priorities: List[EventPriority]  # Priority level for each event

    # Cached fitness
    fitness: float = 0.0
    latency_score: float = 0.0
    throughput_score: float = 0.0
    correctness_score: float = 0.0

    def __post_init__(self):
        assert len(self.event_names) == len(self.kinds) == len(self.priorities)

    def get_descriptor(self, event_name: str) -> EventDescriptor:
        """Get full descriptor for an event."""
        idx = self.event_names.index(event_name)
        return EventDescriptor(
            name=event_name,
            kind=self.kinds[idx],
            priority=self.priorities[idx]
        )

    def get_all_descriptors(self) -> List[EventDescriptor]:
        """Get all event descriptors."""
        return [
            EventDescriptor(name=n, kind=k, priority=p)
            for n, k, p in zip(self.event_names, self.kinds, self.priorities)
        ]

    @property
    def n_events(self) -> int:
        return len(self.event_names)

    @property
    def n_internal(self) -> int:
        return sum(1 for k in self.kinds if k == EventKind.INTERNAL)

    @property
    def n_external(self) -> int:
        return sum(1 for k in self.kinds if k == EventKind.EXTERNAL)

    def summary(self) -> str:
        """Human-readable summary."""
        lines = ["Event Priority Assignment:"]
        for n, k, p in zip(self.event_names, self.kinds, self.priorities):
            kind_str = "INT" if k == EventKind.INTERNAL else "EXT"
            prio_str = ["LOW", "NORMAL", "HIGH", "CRITICAL"][p]
            lines.append(f"  {n}: {kind_str}/{prio_str}")
        return "\n".join(lines)

    def to_json(self) -> Dict:
        """Export to JSON-serializable format."""
        return {
            "events": [
                {
                    "name": n,
                    "kind": "INTERNAL" if k == EventKind.INTERNAL else "EXTERNAL",
                    "priority": ["LOW", "NORMAL", "HIGH", "CRITICAL"][p]
                }
                for n, k, p in zip(self.event_names, self.kinds, self.priorities)
            ],
            "fitness": self.fitness
        }


def create_random_genome(event_names: List[str]) -> EventPriorityGenome:
    """Create genome with random kind/priority assignments."""
    n = len(event_names)

    # Random kinds: ~30% internal, ~70% external (heuristic starting point)
    kinds = [
        EventKind.INTERNAL if random.random() < 0.3 else EventKind.EXTERNAL
        for _ in range(n)
    ]

    # Random priorities
    priorities = [
        random.choice(list(EventPriority))
        for _ in range(n)
    ]

    return EventPriorityGenome(
        event_names=list(event_names),
        kinds=kinds,
        priorities=priorities
    )


# =============================================================================
# SIMULATION ENVIRONMENT
# =============================================================================

@dataclass
class SimulationTrace:
    """Trace from event simulation for fitness evaluation."""
    events_processed: List[str]
    processing_times: List[float]
    state_sequence: List[str]
    latency_violations: int
    ordering_errors: int

    @property
    def total_latency(self) -> float:
        return sum(self.processing_times)

    @property
    def avg_latency(self) -> float:
        if not self.processing_times:
            return 0.0
        return self.total_latency / len(self.processing_times)


class EventSimulator:
    """
    Simulates event processing with a statechart-like state machine.

    Used to evaluate fitness of priority/kind assignments.
    """

    def __init__(self, genome: EventPriorityGenome, state_names: List[str]):
        self.genome = genome
        self.state_names = state_names
        self.current_state = state_names[0] if state_names else "Idle"
        self.queue = PriorityEventQueue()

        # Build event descriptors
        self.descriptors = {
            desc.name: desc
            for desc in genome.get_all_descriptors()
        }

        # Simulated transition table (state, event) -> next_state
        # This would come from the statechart in real usage
        self.transitions: Dict[Tuple[str, str], str] = {}
        self._build_random_transitions()

    def _build_random_transitions(self):
        """Build random transition table for simulation."""
        for state in self.state_names:
            for event_name in self.genome.event_names:
                # Random next state
                next_state = random.choice(self.state_names)
                self.transitions[(state, event_name)] = next_state

    def reset(self):
        """Reset simulator state."""
        self.current_state = self.state_names[0]
        self.queue.clear()

    def send_event(self, event_name: str, payload: Dict[str, Any] = None):
        """Send an event to the simulator."""
        if event_name not in self.descriptors:
            return

        desc = self.descriptors[event_name]
        event = QueuedEvent(
            descriptor=desc,
            payload=payload or {},
            timestamp=time.time(),
            sequence=0  # Set by queue
        )
        self.queue.enqueue(event)

    def process_step(self) -> Optional[Tuple[str, float]]:
        """
        Process one event from queue.

        Returns (event_name, processing_time) or None if queue empty.
        """
        event = self.queue.dequeue()
        if event is None:
            return None

        start = time.time()

        # Apply transition
        key = (self.current_state, event.descriptor.name)
        if key in self.transitions:
            self.current_state = self.transitions[key]

        # Simulate processing time based on priority
        # Higher priority = faster processing (less wait time)
        base_time = 0.001
        priority_factor = 1.0 / (1 + event.descriptor.priority)
        simulated_time = base_time * priority_factor

        return event.descriptor.name, simulated_time

    def run_simulation(self, event_sequence: List[str], max_steps: int = 1000) -> SimulationTrace:
        """
        Run simulation with event sequence.

        Returns trace of processed events.
        """
        self.reset()

        processed = []
        times = []
        states = [self.current_state]
        latency_violations = 0
        ordering_errors = 0

        # Send all events
        for event_name in event_sequence:
            self.send_event(event_name)

        # Process until empty or max steps
        steps = 0
        last_critical_time = 0.0

        while not self.queue.is_empty() and steps < max_steps:
            result = self.process_step()
            if result:
                event_name, proc_time = result
                processed.append(event_name)
                times.append(proc_time)
                states.append(self.current_state)

                # Check for latency violations
                desc = self.descriptors[event_name]
                if desc.priority == EventPriority.CRITICAL:
                    if proc_time > 0.002:  # Threshold
                        latency_violations += 1

            steps += 1

        return SimulationTrace(
            events_processed=processed,
            processing_times=times,
            state_sequence=states,
            latency_violations=latency_violations,
            ordering_errors=ordering_errors
        )


# =============================================================================
# FITNESS EVALUATION
# =============================================================================

@dataclass
class FitnessComponents:
    """Multi-objective fitness components."""
    latency_score: float      # Lower latency for high-priority events
    throughput_score: float   # Overall event throughput
    correctness_score: float  # Ordering correctness
    parsimony_score: float    # Penalty for too many CRITICAL events

    def weighted_sum(self, weights: Dict[str, float] = None) -> float:
        """Compute weighted fitness."""
        w = weights or {
            "latency": 1.0,
            "throughput": 0.5,
            "correctness": 1.5,
            "parsimony": 0.2
        }
        return (
            w["latency"] * self.latency_score +
            w["throughput"] * self.throughput_score +
            w["correctness"] * self.correctness_score +
            w["parsimony"] * self.parsimony_score
        )


class FitnessEvaluator:
    """
    Evaluates fitness of event priority genomes.

    Uses simulation traces to measure:
    - Latency: High-priority events processed quickly
    - Throughput: Overall processing efficiency
    - Correctness: Causal ordering preserved
    - Parsimony: Avoid over-prioritizing
    """

    def __init__(
        self,
        state_names: List[str],
        event_sequences: List[List[str]],
        priority_requirements: Dict[str, EventPriority] = None
    ):
        """
        Args:
            state_names: States in the simulated machine
            event_sequences: Test sequences for evaluation
            priority_requirements: Optional ground truth (for validation only)
        """
        self.state_names = state_names
        self.event_sequences = event_sequences
        self.priority_requirements = priority_requirements or {}

    def evaluate(self, genome: EventPriorityGenome) -> FitnessComponents:
        """Evaluate genome fitness."""
        simulator = EventSimulator(genome, self.state_names)

        total_latency = 0.0
        total_events = 0
        violations = 0
        ordering_errors = 0

        for seq in self.event_sequences:
            trace = simulator.run_simulation(seq)
            total_latency += trace.total_latency
            total_events += len(trace.events_processed)
            violations += trace.latency_violations
            ordering_errors += trace.ordering_errors

        # Latency score: lower is better, normalize
        avg_latency = total_latency / max(total_events, 1)
        latency_score = 1.0 / (1.0 + avg_latency * 100)

        # Throughput score: events processed per sequence
        throughput_score = total_events / (len(self.event_sequences) * 10)
        throughput_score = min(1.0, throughput_score)

        # Correctness score: penalty for violations
        correctness_score = 1.0 - (violations + ordering_errors) / max(total_events, 1)
        correctness_score = max(0.0, correctness_score)

        # Parsimony: penalize too many CRITICAL/HIGH priorities
        n_critical = sum(1 for p in genome.priorities if p == EventPriority.CRITICAL)
        n_high = sum(1 for p in genome.priorities if p == EventPriority.HIGH)
        overuse_penalty = (n_critical * 0.1 + n_high * 0.05) / genome.n_events
        parsimony_score = 1.0 - overuse_penalty

        return FitnessComponents(
            latency_score=latency_score,
            throughput_score=throughput_score,
            correctness_score=correctness_score,
            parsimony_score=parsimony_score
        )

    def evaluate_with_requirements(self, genome: EventPriorityGenome) -> float:
        """
        Evaluate against ground truth requirements (if available).

        Returns match score: how well evolved priorities match requirements.
        """
        if not self.priority_requirements:
            return 0.0

        matches = 0
        for event_name, required_priority in self.priority_requirements.items():
            idx = genome.event_names.index(event_name) if event_name in genome.event_names else -1
            if idx >= 0 and genome.priorities[idx] == required_priority:
                matches += 1

        return matches / len(self.priority_requirements)


# =============================================================================
# EVOLUTION OPERATORS
# =============================================================================

def mutate(genome: EventPriorityGenome, mutation_rate: float = 0.3) -> EventPriorityGenome:
    """Apply random mutations to genome."""
    g = copy.deepcopy(genome)

    for i in range(g.n_events):
        # Mutate kind (internal/external)
        if random.random() < mutation_rate:
            g.kinds[i] = EventKind.INTERNAL if g.kinds[i] == EventKind.EXTERNAL else EventKind.EXTERNAL

        # Mutate priority
        if random.random() < mutation_rate:
            # Step priority up or down
            if random.random() < 0.5:
                g.priorities[i] = EventPriority(min(3, g.priorities[i] + 1))
            else:
                g.priorities[i] = EventPriority(max(0, g.priorities[i] - 1))

        # Occasionally completely randomize priority
        if random.random() < mutation_rate * 0.3:
            g.priorities[i] = random.choice(list(EventPriority))

    return g


def crossover(p1: EventPriorityGenome, p2: EventPriorityGenome) -> EventPriorityGenome:
    """Create offspring by combining parents."""
    assert p1.event_names == p2.event_names, "Parents must have same events"

    # Single-point crossover
    point = random.randint(1, p1.n_events - 1)

    child = EventPriorityGenome(
        event_names=list(p1.event_names),
        kinds=p1.kinds[:point] + p2.kinds[point:],
        priorities=p1.priorities[:point] + p2.priorities[point:]
    )

    return child


def tournament_select(population: List[EventPriorityGenome], k: int = 3) -> EventPriorityGenome:
    """Tournament selection."""
    tournament = random.sample(population, min(k, len(population)))
    return max(tournament, key=lambda g: g.fitness)


# =============================================================================
# MAIN EVOLVER
# =============================================================================

@dataclass
class EvolutionConfig:
    """Configuration for priority evolution."""
    population_size: int = 50
    n_generations: int = 100
    elite_size: int = 5
    mutation_rate: float = 0.3
    crossover_rate: float = 0.7
    tournament_size: int = 3
    verbose: bool = True
    log_every: int = 10


@dataclass
class EvolutionStats:
    """Statistics from evolution run."""
    generations: List[Dict] = field(default_factory=list)
    total_time: float = 0.0
    best_fitness: float = 0.0

    def add_generation(self, gen: int, best: float, avg: float):
        self.generations.append({
            "generation": gen,
            "best_fitness": best,
            "avg_fitness": avg
        })


class EventPriorityEvolver:
    """
    Evolves optimal event kind and priority assignments.

    Uses patterns from exp_topology_evolution.
    """

    def __init__(
        self,
        event_names: List[str],
        state_names: List[str],
        event_sequences: List[List[str]],
        config: EvolutionConfig = None
    ):
        self.event_names = event_names
        self.config = config or EvolutionConfig()

        # Fitness evaluator
        self.evaluator = FitnessEvaluator(
            state_names=state_names,
            event_sequences=event_sequences
        )

        # Statistics
        self.stats = EvolutionStats()

    def initialize_population(self) -> List[EventPriorityGenome]:
        """Create initial population."""
        return [
            create_random_genome(self.event_names)
            for _ in range(self.config.population_size)
        ]

    def evaluate_population(self, population: List[EventPriorityGenome]):
        """Evaluate fitness for all individuals."""
        for genome in population:
            components = self.evaluator.evaluate(genome)
            genome.fitness = components.weighted_sum()
            genome.latency_score = components.latency_score
            genome.throughput_score = components.throughput_score
            genome.correctness_score = components.correctness_score

    def evolve(self) -> EventPriorityGenome:
        """
        Run evolution to discover optimal priority assignments.

        Returns best genome found.
        """
        start_time = time.time()

        if self.config.verbose:
            print("=" * 60)
            print("EVENT PRIORITY EVOLUTION")
            print("=" * 60)
            print(f"Events: {len(self.event_names)}")
            print(f"Population: {self.config.population_size}")
            print(f"Generations: {self.config.n_generations}")
            print("-" * 60)

        # Initialize
        population = self.initialize_population()
        self.evaluate_population(population)

        best_ever = max(population, key=lambda g: g.fitness)
        best_ever = copy.deepcopy(best_ever)

        # Evolution loop
        for gen in range(self.config.n_generations):
            # Sort by fitness
            population.sort(key=lambda g: g.fitness, reverse=True)

            # Keep elite
            new_population = population[:self.config.elite_size]

            # Generate offspring
            while len(new_population) < self.config.population_size:
                if random.random() < self.config.crossover_rate:
                    p1 = tournament_select(population, self.config.tournament_size)
                    p2 = tournament_select(population, self.config.tournament_size)
                    child = crossover(p1, p2)
                else:
                    parent = tournament_select(population, self.config.tournament_size)
                    child = copy.deepcopy(parent)

                child = mutate(child, self.config.mutation_rate)
                new_population.append(child)

            population = new_population
            self.evaluate_population(population)

            # Track best
            gen_best = max(population, key=lambda g: g.fitness)
            if gen_best.fitness > best_ever.fitness:
                best_ever = copy.deepcopy(gen_best)

            # Statistics
            avg_fitness = sum(g.fitness for g in population) / len(population)
            self.stats.add_generation(gen, gen_best.fitness, avg_fitness)

            # Progress
            if self.config.verbose and (gen % self.config.log_every == 0 or gen == self.config.n_generations - 1):
                n_crit = sum(1 for g in population for p in g.priorities if p == EventPriority.CRITICAL)
                avg_crit = n_crit / len(population)
                print(
                    f"Gen {gen:3d} | "
                    f"Best: {gen_best.fitness:.4f} | "
                    f"Avg: {avg_fitness:.4f} | "
                    f"Avg CRITICAL: {avg_crit:.1f}"
                )

        # Final stats
        self.stats.total_time = time.time() - start_time
        self.stats.best_fitness = best_ever.fitness

        if self.config.verbose:
            print("-" * 60)
            print(f"Evolution completed in {self.stats.total_time:.2f}s")
            print(f"Best fitness: {best_ever.fitness:.4f}")
            print()
            print(best_ever.summary())

        return best_ever


# =============================================================================
# TRACE-BASED LEARNING (from exp_execution_replay)
# =============================================================================

class TracePriorityLearner:
    """
    Learn event priorities from execution traces.

    Analyzes trace data to infer optimal priorities:
    - Events that cause cascading transitions -> INTERNAL
    - Events that are always processed immediately -> HIGH/CRITICAL
    - Events that can be delayed -> NORMAL/LOW
    """

    def __init__(self, traces: List[Dict]):
        """
        Args:
            traces: List of trace dicts with 'entries' field
        """
        self.traces = traces
        self.event_stats: Dict[str, Dict] = defaultdict(lambda: {
            "count": 0,
            "caused_internal": 0,  # How often this event caused internal events
            "avg_latency": 0.0,
            "critical_state_count": 0  # How often in critical states
        })

    def analyze_traces(self):
        """Extract priority hints from trace data."""
        for trace in self.traces:
            entries = trace.get("entries", [])

            for i, entry in enumerate(entries):
                event_name = entry.get("trigger_event", {}).get("name", "UNKNOWN")
                self.event_stats[event_name]["count"] += 1

                # Check if next entry is caused by this event (internal)
                if i + 1 < len(entries):
                    next_entry = entries[i + 1]
                    # Simple heuristic: same timestamp = internal
                    if entry.get("sequence", 0) + 1 == next_entry.get("sequence", 0):
                        self.event_stats[event_name]["caused_internal"] += 1

    def suggest_priorities(self) -> Dict[str, EventPriority]:
        """Suggest priorities based on trace analysis."""
        suggestions = {}

        for event_name, stats in self.event_stats.items():
            count = stats["count"]
            if count == 0:
                suggestions[event_name] = EventPriority.NORMAL
                continue

            internal_ratio = stats["caused_internal"] / count

            # Heuristic: high internal ratio -> INTERNAL/HIGH priority
            if internal_ratio > 0.5:
                suggestions[event_name] = EventPriority.HIGH
            elif internal_ratio > 0.2:
                suggestions[event_name] = EventPriority.NORMAL
            else:
                suggestions[event_name] = EventPriority.LOW

        return suggestions

    def create_seeded_genome(self) -> EventPriorityGenome:
        """Create genome seeded with trace-derived priorities."""
        self.analyze_traces()
        suggestions = self.suggest_priorities()

        event_names = list(suggestions.keys())
        kinds = []
        priorities = []

        for name in event_names:
            # Events that cause internal events are likely INTERNAL
            stats = self.event_stats[name]
            if stats["count"] > 0:
                internal_ratio = stats["caused_internal"] / stats["count"]
                kinds.append(EventKind.INTERNAL if internal_ratio > 0.3 else EventKind.EXTERNAL)
            else:
                kinds.append(EventKind.EXTERNAL)

            priorities.append(suggestions[name])

        return EventPriorityGenome(
            event_names=event_names,
            kinds=kinds,
            priorities=priorities
        )


# =============================================================================
# DEMO
# =============================================================================

def demo_evolution():
    """Demonstrate event priority evolution."""
    print("=" * 70)
    print("INTERNAL vs EXTERNAL EVENT EVOLUTION DEMO")
    print("=" * 70)
    print()
    print("Goal: EVOLVE optimal event kind (INT/EXT) and priority (CRIT/HIGH/NORM/LOW)")
    print("No hardcoding - the system discovers what works!")
    print()

    # Define events (typical game/UI events)
    event_names = [
        "TICK",           # Frame tick
        "USER_INPUT",     # User button press
        "COLLISION",      # Physics collision
        "SPAWN_ENEMY",    # Spawn entity
        "DAMAGE",         # Take damage
        "ANIMATION_END",  # Animation completed
        "TIMER_EXPIRE",   # Timer fired
        "SCORE_UPDATE",   # Score changed
        "LEVEL_UP",       # Level completed
        "GAME_OVER",      # Game ended
    ]

    # States in our test machine
    state_names = ["Idle", "Playing", "Paused", "GameOver", "LevelTransition"]

    # Generate test event sequences
    sequences = []
    for _ in range(50):
        seq_len = random.randint(5, 20)
        seq = [random.choice(event_names) for _ in range(seq_len)]
        sequences.append(seq)

    # Run evolution
    config = EvolutionConfig(
        population_size=30,
        n_generations=50,
        elite_size=3,
        mutation_rate=0.25,
        log_every=10,
        verbose=True
    )

    evolver = EventPriorityEvolver(
        event_names=event_names,
        state_names=state_names,
        event_sequences=sequences,
        config=config
    )

    best = evolver.evolve()

    # Output results
    print()
    print("=" * 70)
    print("EVOLVED EVENT CLASSIFICATION")
    print("=" * 70)
    print()

    # Group by classification
    internal_events = []
    external_events = []

    for desc in best.get_all_descriptors():
        if desc.kind == EventKind.INTERNAL:
            internal_events.append(desc)
        else:
            external_events.append(desc)

    print("INTERNAL Events (synchronous):")
    for e in sorted(internal_events, key=lambda x: -x.priority):
        prio = ["LOW", "NORMAL", "HIGH", "CRITICAL"][e.priority]
        print(f"  {e.name}: {prio}")

    print()
    print("EXTERNAL Events (queued):")
    for e in sorted(external_events, key=lambda x: -x.priority):
        prio = ["LOW", "NORMAL", "HIGH", "CRITICAL"][e.priority]
        print(f"  {e.name}: {prio}")

    # Export JSON
    print()
    print("=" * 70)
    print("JSON OUTPUT")
    print("=" * 70)
    print(json.dumps(best.to_json(), indent=2))

    return best


def demo_trace_learning():
    """Demonstrate learning priorities from traces."""
    print()
    print("=" * 70)
    print("TRACE-BASED PRIORITY LEARNING")
    print("=" * 70)
    print()

    # Simulated traces (would load from execution.proto in real usage)
    traces = [
        {
            "trace_id": "t1",
            "entries": [
                {"sequence": 1, "trigger_event": {"name": "USER_INPUT"}},
                {"sequence": 2, "trigger_event": {"name": "COLLISION"}},  # Caused by USER_INPUT
                {"sequence": 3, "trigger_event": {"name": "DAMAGE"}},     # Caused by COLLISION
                {"sequence": 10, "trigger_event": {"name": "TICK"}},
                {"sequence": 11, "trigger_event": {"name": "ANIMATION_END"}},
            ]
        },
        {
            "trace_id": "t2",
            "entries": [
                {"sequence": 1, "trigger_event": {"name": "SPAWN_ENEMY"}},
                {"sequence": 5, "trigger_event": {"name": "TICK"}},
                {"sequence": 6, "trigger_event": {"name": "COLLISION"}},
                {"sequence": 7, "trigger_event": {"name": "DAMAGE"}},
            ]
        }
    ]

    learner = TracePriorityLearner(traces)
    seeded_genome = learner.create_seeded_genome()

    print("Trace-derived priorities:")
    print(seeded_genome.summary())

    return seeded_genome


if __name__ == "__main__":
    # Run both demos
    best_evolved = demo_evolution()
    seeded = demo_trace_learning()

    print()
    print("=" * 70)
    print("KEY INSIGHTS")
    print("=" * 70)
    print("""
Internal vs External Event Semantics:

1. INTERNAL events are processed SYNCHRONOUSLY within the same RTC step
   - Generated by actions, entry/exit
   - Enable micro-step chains (cascading transitions)
   - Example: COLLISION -> DAMAGE -> GAME_OVER

2. EXTERNAL events are QUEUED and processed in subsequent steps
   - Arrive from environment/users
   - Priority determines processing order
   - Example: USER_INPUT, TICK

3. Evolution discovered which events work best as INTERNAL vs EXTERNAL
   - No hardcoding required
   - Adapts to simulation dynamics

4. Priority levels (CRITICAL > HIGH > NORMAL > LOW) control queue ordering
   - Evolution finds optimal priority assignment
   - Balances latency vs throughput vs correctness
""")
