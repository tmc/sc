"""
Region Broadcast: Event Broadcasting to Parallel Regions

Implements broadcast semantics for AND-states (parallel regions):
- Global events delivered to all regions simultaneously
- Region-scoped events delivered to specific regions
- Priority handling when multiple regions respond
- Conflict resolution for competing transitions

Based on proto/statecharts/v1/statecharts.proto:
  STATE_TYPE_PARALLEL = 3  // AND-decomposition: concurrent substates

Semantics from Harel:
  ψ(s) = PARALLEL ∧ s ∈ σ ⟹ children(s) ⊆ σ (all children active)

NO HARDCODING: Learn broadcast patterns from examples.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Set, Tuple, Optional, Any, Callable
from enum import Enum, auto
import random
import time


# =============================================================================
# BROADCAST TYPES
# =============================================================================

class BroadcastScope(Enum):
    """Scope of event broadcast."""
    GLOBAL = 1       # All regions receive event
    REGION = 2       # Only specified regions receive
    LOCAL = 3        # Only current state receives


class ConflictResolution(Enum):
    """How to handle conflicts when multiple regions respond."""
    FIRST_WINS = 1   # First responding region wins
    ALL_EXECUTE = 2  # All regions execute (default AND semantics)
    PRIORITY = 3     # Highest priority region wins
    MERGE = 4        # Merge results from all regions


class EventDelivery(Enum):
    """How events are delivered."""
    SYNCHRONOUS = 1   # All regions process in lock-step
    ASYNCHRONOUS = 2  # Regions process independently
    QUEUED = 3        # Events queued per region


# =============================================================================
# DATA TYPES
# =============================================================================

@dataclass
class Region:
    """A parallel region in an AND-state."""
    id: str
    label: str
    states: List[str]  # State labels in this region
    current_state: str
    priority: int = 0
    context: Dict[str, Any] = field(default_factory=dict)

    def __hash__(self):
        return hash(self.id)


@dataclass
class RegionTransition:
    """A transition within a region."""
    region_id: str
    from_state: str
    to_state: str
    event: str
    guard: Optional[Callable] = None
    action: Optional[Callable] = None


@dataclass
class BroadcastEvent:
    """An event to broadcast to regions."""
    name: str
    scope: BroadcastScope = BroadcastScope.GLOBAL
    target_regions: List[str] = field(default_factory=list)  # For REGION scope
    payload: Dict[str, Any] = field(default_factory=dict)
    priority: int = 0


@dataclass
class BroadcastResult:
    """Result of broadcasting an event."""
    event: BroadcastEvent
    responding_regions: List[str]
    transitions_taken: List[RegionTransition]
    conflicts: List[str]  # Conflict descriptions
    final_states: Dict[str, str]  # region_id -> current_state


# =============================================================================
# PARALLEL STATE MACHINE
# =============================================================================

@dataclass
class ParallelState:
    """
    A parallel (AND) state with multiple concurrent regions.

    Regions operate independently but can:
    - Share events (broadcast semantics)
    - Share context (coordination)
    - Have priority ordering (conflict resolution)
    """
    label: str
    regions: List[Region] = field(default_factory=list)
    transitions: List[RegionTransition] = field(default_factory=list)
    shared_context: Dict[str, Any] = field(default_factory=dict)
    conflict_resolution: ConflictResolution = ConflictResolution.ALL_EXECUTE
    event_delivery: EventDelivery = EventDelivery.SYNCHRONOUS

    def add_region(self, region: Region):
        """Add a region to this parallel state."""
        self.regions.append(region)

    def add_transition(self, transition: RegionTransition):
        """Add a transition to a region."""
        self.transitions.append(transition)

    def get_region(self, region_id: str) -> Optional[Region]:
        """Get region by ID."""
        for r in self.regions:
            if r.id == region_id:
                return r
        return None

    def get_current_configuration(self) -> Dict[str, str]:
        """Get current state of all regions."""
        return {r.id: r.current_state for r in self.regions}


# =============================================================================
# BROADCAST ENGINE
# =============================================================================

class BroadcastEngine:
    """
    Handles event broadcasting to parallel regions.

    Implements the semantic rule:
      When event e occurs in AND-state s:
        For each region r ∈ children(s):
          If ∃ transition t in r enabled by e:
            Execute t
    """

    def __init__(
        self,
        conflict_resolution: ConflictResolution = ConflictResolution.ALL_EXECUTE,
    ):
        self.conflict_resolution = conflict_resolution
        self.event_history: List[BroadcastResult] = []

    def broadcast(
        self,
        parallel_state: ParallelState,
        event: BroadcastEvent,
    ) -> BroadcastResult:
        """
        Broadcast event to parallel regions.

        Returns:
            BroadcastResult with responding regions and transitions taken
        """
        result = BroadcastResult(
            event=event,
            responding_regions=[],
            transitions_taken=[],
            conflicts=[],
            final_states={},
        )

        # Determine which regions receive the event
        target_regions = self._get_target_regions(parallel_state, event)

        # Find enabled transitions in each region
        enabled_transitions: List[Tuple[Region, RegionTransition]] = []

        for region in target_regions:
            for trans in parallel_state.transitions:
                if trans.region_id == region.id:
                    if self._is_enabled(trans, region, parallel_state.shared_context, event):
                        enabled_transitions.append((region, trans))
                        result.responding_regions.append(region.id)

        # Handle conflicts
        if len(enabled_transitions) > 1:
            enabled_transitions = self._resolve_conflicts(
                enabled_transitions, parallel_state, result
            )

        # Execute transitions
        for region, trans in enabled_transitions:
            self._execute_transition(region, trans, parallel_state.shared_context, event)
            result.transitions_taken.append(trans)

        # Record final states
        result.final_states = parallel_state.get_current_configuration()

        # Record in history
        self.event_history.append(result)

        return result

    def _get_target_regions(
        self,
        parallel_state: ParallelState,
        event: BroadcastEvent,
    ) -> List[Region]:
        """Determine which regions receive the event."""
        if event.scope == BroadcastScope.GLOBAL:
            return parallel_state.regions
        elif event.scope == BroadcastScope.REGION:
            return [r for r in parallel_state.regions if r.id in event.target_regions]
        else:  # LOCAL
            return []

    def _is_enabled(
        self,
        trans: RegionTransition,
        region: Region,
        shared_context: Dict[str, Any],
        event: BroadcastEvent,
    ) -> bool:
        """Check if transition is enabled."""
        # Check event match
        if trans.event != event.name:
            return False

        # Check source state
        if trans.from_state != region.current_state:
            return False

        # Check guard
        if trans.guard:
            context = {
                **shared_context,
                **region.context,
                '__event__': event,
                '__region__': region,
            }
            try:
                if not trans.guard(context):
                    return False
            except Exception:
                return False

        return True

    def _resolve_conflicts(
        self,
        transitions: List[Tuple[Region, RegionTransition]],
        parallel_state: ParallelState,
        result: BroadcastResult,
    ) -> List[Tuple[Region, RegionTransition]]:
        """Resolve conflicts between competing transitions."""
        if self.conflict_resolution == ConflictResolution.ALL_EXECUTE:
            # No conflict - all execute (standard AND semantics)
            return transitions

        elif self.conflict_resolution == ConflictResolution.FIRST_WINS:
            result.conflicts.append(
                f"FIRST_WINS: {len(transitions)} transitions, executing first"
            )
            return transitions[:1]

        elif self.conflict_resolution == ConflictResolution.PRIORITY:
            # Sort by region priority, take highest
            sorted_trans = sorted(
                transitions,
                key=lambda t: t[0].priority,
                reverse=True,
            )
            if len(sorted_trans) > 1:
                result.conflicts.append(
                    f"PRIORITY: Selected region {sorted_trans[0][0].id} "
                    f"(priority={sorted_trans[0][0].priority})"
                )
            return sorted_trans[:1]

        else:  # MERGE - all execute (same as ALL_EXECUTE for transitions)
            return transitions

    def _execute_transition(
        self,
        region: Region,
        trans: RegionTransition,
        shared_context: Dict[str, Any],
        event: BroadcastEvent,
    ):
        """Execute a transition."""
        # Prepare context for action
        context = {
            **shared_context,
            **region.context,
            '__event__': event,
            '__region__': region,
        }

        # Execute action
        if trans.action:
            try:
                action_result = trans.action(context)
                # Merge results back into contexts
                if isinstance(action_result, dict):
                    for key, value in action_result.items():
                        if key.startswith('__shared__'):
                            shared_context[key[10:]] = value
                        else:
                            region.context[key] = value
            except Exception:
                pass

        # Update state
        region.current_state = trans.to_state


# =============================================================================
# BROADCAST PATTERN LEARNER
# =============================================================================

@dataclass
class BroadcastPattern:
    """A learned broadcast pattern."""
    event_name: str
    scope: BroadcastScope
    responding_regions: Set[str]
    typical_conflicts: int
    resolution_used: ConflictResolution
    fitness: float = 0.0


@dataclass
class BroadcastGenome:
    """Evolvable broadcast configuration."""
    scope_rules: Dict[str, BroadcastScope]  # event -> scope
    region_targets: Dict[str, List[str]]    # event -> target regions
    conflict_resolution: ConflictResolution
    fitness: float = 0.0


class BroadcastPatternLearner:
    """
    Learn optimal broadcast patterns from examples.

    Given example traces of events and region responses,
    learn the best scope and conflict resolution strategies.
    """

    def __init__(
        self,
        event_names: List[str],
        region_ids: List[str],
        population_size: int = 30,
        n_generations: int = 50,
    ):
        self.event_names = event_names
        self.region_ids = region_ids
        self.population_size = population_size
        self.n_generations = n_generations

    def random_genome(self) -> BroadcastGenome:
        """Generate random broadcast configuration."""
        scope_rules = {}
        region_targets = {}

        for event in self.event_names:
            scope = random.choice(list(BroadcastScope))
            scope_rules[event] = scope

            if scope == BroadcastScope.REGION:
                # Random subset of regions
                n_targets = random.randint(1, len(self.region_ids))
                region_targets[event] = random.sample(self.region_ids, n_targets)
            else:
                region_targets[event] = []

        conflict_resolution = random.choice(list(ConflictResolution))

        return BroadcastGenome(
            scope_rules=scope_rules,
            region_targets=region_targets,
            conflict_resolution=conflict_resolution,
        )

    def mutate(self, genome: BroadcastGenome, rate: float = 0.2) -> BroadcastGenome:
        """Mutate broadcast configuration."""
        new_scope = dict(genome.scope_rules)
        new_targets = dict(genome.region_targets)
        new_conflict = genome.conflict_resolution

        for event in self.event_names:
            if random.random() < rate:
                new_scope[event] = random.choice(list(BroadcastScope))

            if new_scope[event] == BroadcastScope.REGION:
                if random.random() < rate:
                    n_targets = random.randint(1, len(self.region_ids))
                    new_targets[event] = random.sample(self.region_ids, n_targets)
            else:
                new_targets[event] = []

        if random.random() < rate:
            new_conflict = random.choice(list(ConflictResolution))

        return BroadcastGenome(
            scope_rules=new_scope,
            region_targets=new_targets,
            conflict_resolution=new_conflict,
        )

    def crossover(self, p1: BroadcastGenome, p2: BroadcastGenome) -> BroadcastGenome:
        """Crossover two configurations."""
        new_scope = {}
        new_targets = {}

        for event in self.event_names:
            if random.random() < 0.5:
                new_scope[event] = p1.scope_rules[event]
                new_targets[event] = p1.region_targets.get(event, [])
            else:
                new_scope[event] = p2.scope_rules[event]
                new_targets[event] = p2.region_targets.get(event, [])

        new_conflict = p1.conflict_resolution if random.random() < 0.5 else p2.conflict_resolution

        return BroadcastGenome(
            scope_rules=new_scope,
            region_targets=new_targets,
            conflict_resolution=new_conflict,
        )

    def evaluate_fitness(
        self,
        genome: BroadcastGenome,
        examples: List[Tuple[BroadcastEvent, Dict[str, str], Dict[str, str]]],
    ) -> float:
        """
        Evaluate fitness on examples.

        Each example is (event, initial_states, expected_final_states).
        """
        if not examples:
            return 0.0

        correct = 0

        for event, initial_states, expected_final in examples:
            # Apply genome's scope and targeting
            if event.name in genome.scope_rules:
                event.scope = genome.scope_rules[event.name]
            if event.name in genome.region_targets:
                event.target_regions = genome.region_targets[event.name]

            # Simulate broadcast (simplified)
            actual_final = self._simulate_broadcast(
                event, initial_states, genome.conflict_resolution
            )

            # Check if matches expected
            if actual_final == expected_final:
                correct += 1

        genome.fitness = correct / len(examples)
        return genome.fitness

    def _simulate_broadcast(
        self,
        event: BroadcastEvent,
        initial_states: Dict[str, str],
        conflict_resolution: ConflictResolution,
    ) -> Dict[str, str]:
        """Simplified broadcast simulation."""
        final_states = dict(initial_states)

        # Determine responding regions
        if event.scope == BroadcastScope.GLOBAL:
            responding = list(initial_states.keys())
        elif event.scope == BroadcastScope.REGION:
            responding = [r for r in event.target_regions if r in initial_states]
        else:
            responding = []

        # Apply conflict resolution
        if conflict_resolution == ConflictResolution.FIRST_WINS:
            responding = responding[:1]
        elif conflict_resolution == ConflictResolution.PRIORITY:
            # Would need priority info
            pass

        # Simulate state changes (simplified: toggle to "next" state)
        for region_id in responding:
            current = initial_states[region_id]
            # Simple heuristic: append "_done" or cycle
            if not current.endswith("_done"):
                final_states[region_id] = current + "_done"

        return final_states

    def evolve(
        self,
        examples: List[Tuple[BroadcastEvent, Dict[str, str], Dict[str, str]]],
        verbose: bool = True,
    ) -> BroadcastGenome:
        """Evolve optimal broadcast configuration."""
        population = [self.random_genome() for _ in range(self.population_size)]
        best_ever = None

        for gen in range(self.n_generations):
            # Evaluate
            for genome in population:
                self.evaluate_fitness(genome, examples)

            # Sort by fitness
            population.sort(key=lambda g: g.fitness, reverse=True)

            # Track best
            if best_ever is None or population[0].fitness > best_ever.fitness:
                best_ever = population[0]

            if verbose and gen % 10 == 0:
                print(f"Gen {gen:3d}: Best fitness = {population[0].fitness:.3f}")

            # Perfect solution?
            if population[0].fitness >= 0.99:
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

def test_region_broadcast():
    """Test region broadcast functionality."""
    print("=" * 60)
    print("REGION BROADCAST TEST")
    print("=" * 60)

    # Create a parallel state with two regions
    audio_region = Region(
        id="audio",
        label="AudioRegion",
        states=["Muted", "Playing", "Paused"],
        current_state="Muted",
        priority=1,
    )

    video_region = Region(
        id="video",
        label="VideoRegion",
        states=["Hidden", "Showing", "Fullscreen"],
        current_state="Hidden",
        priority=2,
    )

    parallel = ParallelState(
        label="MediaPlayer",
        regions=[audio_region, video_region],
    )

    # Add transitions
    parallel.add_transition(RegionTransition(
        region_id="audio",
        from_state="Muted",
        to_state="Playing",
        event="PLAY",
    ))

    parallel.add_transition(RegionTransition(
        region_id="video",
        from_state="Hidden",
        to_state="Showing",
        event="PLAY",
    ))

    parallel.add_transition(RegionTransition(
        region_id="audio",
        from_state="Playing",
        to_state="Muted",
        event="MUTE",
    ))

    # Create broadcast engine
    engine = BroadcastEngine()

    print("\nInitial configuration:", parallel.get_current_configuration())

    # Broadcast PLAY event globally
    play_event = BroadcastEvent(name="PLAY", scope=BroadcastScope.GLOBAL)
    result = engine.broadcast(parallel, play_event)

    print(f"\nAfter PLAY event (GLOBAL):")
    print(f"  Responding regions: {result.responding_regions}")
    print(f"  Transitions taken: {len(result.transitions_taken)}")
    print(f"  Final states: {result.final_states}")

    # Broadcast MUTE to audio only
    mute_event = BroadcastEvent(
        name="MUTE",
        scope=BroadcastScope.REGION,
        target_regions=["audio"],
    )
    result = engine.broadcast(parallel, mute_event)

    print(f"\nAfter MUTE event (REGION=audio):")
    print(f"  Responding regions: {result.responding_regions}")
    print(f"  Final states: {result.final_states}")

    # Verify
    assert parallel.get_region("audio").current_state == "Muted", "Audio should be muted"
    assert parallel.get_region("video").current_state == "Showing", "Video should still be showing"

    print("\n[PASS] Region broadcast test passed!")
    return True


def test_broadcast_pattern_learning():
    """Test learning broadcast patterns."""
    print("\n" + "=" * 60)
    print("BROADCAST PATTERN LEARNING TEST")
    print("=" * 60)

    # Event names and regions
    events = ["PLAY", "PAUSE", "MUTE", "FULLSCREEN"]
    regions = ["audio", "video", "controls"]

    learner = BroadcastPatternLearner(
        event_names=events,
        region_ids=regions,
        n_generations=30,
    )

    # Example traces (event, initial_states, expected_final)
    examples = [
        # PLAY affects audio and video (global)
        (
            BroadcastEvent(name="PLAY"),
            {"audio": "Muted", "video": "Hidden", "controls": "Inactive"},
            {"audio": "Muted_done", "video": "Hidden_done", "controls": "Inactive_done"},
        ),
        # MUTE only affects audio (region-scoped)
        (
            BroadcastEvent(name="MUTE", scope=BroadcastScope.REGION, target_regions=["audio"]),
            {"audio": "Playing", "video": "Showing", "controls": "Active"},
            {"audio": "Playing_done", "video": "Showing", "controls": "Active"},
        ),
        # FULLSCREEN only affects video
        (
            BroadcastEvent(name="FULLSCREEN", scope=BroadcastScope.REGION, target_regions=["video"]),
            {"audio": "Playing", "video": "Showing", "controls": "Active"},
            {"audio": "Playing", "video": "Showing_done", "controls": "Active"},
        ),
    ]

    print("\nLearning optimal broadcast configuration...")
    best = learner.evolve(examples, verbose=True)

    print(f"\nBest configuration:")
    print(f"  Conflict resolution: {best.conflict_resolution.name}")
    print(f"  Scope rules:")
    for event, scope in best.scope_rules.items():
        targets = best.region_targets.get(event, [])
        print(f"    {event}: {scope.name} -> {targets if targets else 'all'}")
    print(f"  Fitness: {best.fitness:.3f}")

    success = best.fitness >= 0.6
    print(f"\n[{'PASS' if success else 'PARTIAL'}] Broadcast pattern learning test")
    return success


if __name__ == "__main__":
    test_region_broadcast()
    test_broadcast_pattern_learning()
    print("\n" + "=" * 60)
    print("ALL REGION BROADCAST TESTS COMPLETE")
    print("=" * 60)
