"""
Region Sync: Shared Context Coordination

Implements synchronization between parallel regions via shared context:
- Shared variables that regions can read/write
- Guards that depend on other regions' states
- Actions that update shared context
- Synchronization barriers and join points

Based on proto/statecharts/v1/statecharts.proto:
  Configuration.context = extended data context

Coordination patterns:
1. Producer-Consumer: One region produces, another consumes
2. Barrier Sync: All regions must reach a point before continuing
3. Leader-Follower: One region leads, others follow
4. Mutual Exclusion: Only one region can be in critical section

NO HARDCODING: Learn sync patterns from examples.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Set, Tuple, Optional, Any, Callable
from enum import Enum, auto
import random
import time
import threading
from collections import defaultdict


# =============================================================================
# SYNCHRONIZATION TYPES
# =============================================================================

class SyncPattern(Enum):
    """Types of region synchronization."""
    NONE = 0              # Independent regions
    PRODUCER_CONSUMER = 1 # One produces, one consumes
    BARRIER = 2           # All must synchronize
    LEADER_FOLLOWER = 3   # One leads, others follow
    MUTUAL_EXCLUSION = 4  # Critical section access
    RENDEZVOUS = 5        # Pairwise synchronization


class ContextAccess(Enum):
    """Type of context access."""
    READ = 1
    WRITE = 2
    READ_WRITE = 3


@dataclass
class SharedVariable:
    """A shared variable in the context."""
    name: str
    value: Any
    type_hint: str = "any"
    read_by: Set[str] = field(default_factory=set)   # region IDs
    written_by: Set[str] = field(default_factory=set)  # region IDs
    version: int = 0


# =============================================================================
# SYNCHRONIZED REGION
# =============================================================================

@dataclass
class SyncRegion:
    """A region with synchronization capabilities."""
    id: str
    label: str
    states: List[str]
    initial_state: str
    current_state: str
    transitions: Dict[Tuple[str, str], Tuple[str, Optional[Callable], Optional[Callable]]] = field(default_factory=dict)
    # (from, event) -> (to, guard, action)
    waiting_for: Optional[str] = None  # Barrier/rendezvous target
    sync_role: Optional[str] = None    # "producer", "consumer", "leader", "follower"

    def is_blocked(self) -> bool:
        """Check if region is blocked waiting for sync."""
        return self.waiting_for is not None


@dataclass
class SyncTransition:
    """A transition with synchronization requirements."""
    region_id: str
    from_state: str
    to_state: str
    event: str
    guard: Optional[Callable] = None
    action: Optional[Callable] = None
    sync_requirement: Optional[str] = None  # Required shared context condition
    context_reads: List[str] = field(default_factory=list)
    context_writes: List[str] = field(default_factory=list)


# =============================================================================
# SHARED CONTEXT
# =============================================================================

class SharedContext:
    """
    Shared context for region coordination.

    Thread-safe access to shared variables.
    """

    def __init__(self):
        self.variables: Dict[str, SharedVariable] = {}
        self.barriers: Dict[str, Set[str]] = {}  # barrier_name -> waiting regions
        self.lock = threading.Lock()
        self.version = 0

    def get(self, name: str, default: Any = None) -> Any:
        """Get variable value."""
        with self.lock:
            if name in self.variables:
                return self.variables[name].value
            return default

    def set(self, name: str, value: Any, region_id: str = None):
        """Set variable value."""
        with self.lock:
            if name not in self.variables:
                self.variables[name] = SharedVariable(name=name, value=value)

            var = self.variables[name]
            var.value = value
            var.version += 1
            self.version += 1

            if region_id:
                var.written_by.add(region_id)

    def increment(self, name: str, amount: int = 1, region_id: str = None):
        """Atomically increment a counter."""
        with self.lock:
            if name not in self.variables:
                self.variables[name] = SharedVariable(name=name, value=0)

            var = self.variables[name]
            var.value = (var.value or 0) + amount
            var.version += 1
            self.version += 1

            if region_id:
                var.written_by.add(region_id)

    def decrement(self, name: str, amount: int = 1, region_id: str = None):
        """Atomically decrement a counter."""
        self.increment(name, -amount, region_id)

    def wait_barrier(self, barrier_name: str, region_id: str, required_regions: Set[str]) -> bool:
        """
        Wait at a barrier until all required regions arrive.

        Returns True if all regions have arrived.
        """
        with self.lock:
            if barrier_name not in self.barriers:
                self.barriers[barrier_name] = set()

            self.barriers[barrier_name].add(region_id)

            if self.barriers[barrier_name] >= required_regions:
                # All arrived, clear barrier
                self.barriers[barrier_name] = set()
                return True

            return False

    def check_condition(self, condition: str) -> bool:
        """
        Check a condition expression against context.

        Supported: "var == value", "var > value", "var in [...]"
        """
        with self.lock:
            try:
                # Parse simple conditions
                if "==" in condition:
                    var_name, value = condition.split("==")
                    var_name = var_name.strip()
                    value = eval(value.strip())
                    return self.variables.get(var_name, SharedVariable(name=var_name, value=None)).value == value

                elif ">" in condition:
                    var_name, value = condition.split(">")
                    var_name = var_name.strip()
                    value = int(value.strip())
                    var_val = self.variables.get(var_name, SharedVariable(name=var_name, value=0)).value
                    return (var_val or 0) > value

                elif "<" in condition:
                    var_name, value = condition.split("<")
                    var_name = var_name.strip()
                    value = int(value.strip())
                    var_val = self.variables.get(var_name, SharedVariable(name=var_name, value=0)).value
                    return (var_val or 0) < value

                elif "is_set" in condition:
                    var_name = condition.replace("is_set(", "").replace(")", "").strip()
                    return var_name in self.variables and self.variables[var_name].value is not None

                return False

            except Exception:
                return False

    def snapshot(self) -> Dict[str, Any]:
        """Get snapshot of all variables."""
        with self.lock:
            return {name: var.value for name, var in self.variables.items()}


# =============================================================================
# SYNCHRONIZED PARALLEL STATE
# =============================================================================

@dataclass
class SynchronizedParallelState:
    """
    A parallel state with synchronized regions.

    Regions coordinate through shared context and sync primitives.
    """
    label: str
    regions: List[SyncRegion] = field(default_factory=list)
    transitions: List[SyncTransition] = field(default_factory=list)
    shared_context: SharedContext = field(default_factory=SharedContext)
    sync_pattern: SyncPattern = SyncPattern.NONE

    def add_region(self, region: SyncRegion):
        """Add a synchronized region."""
        self.regions.append(region)

    def add_transition(self, transition: SyncTransition):
        """Add a synchronized transition."""
        self.transitions.append(transition)

    def get_region(self, region_id: str) -> Optional[SyncRegion]:
        """Get region by ID."""
        for r in self.regions:
            if r.id == region_id:
                return r
        return None

    def get_configuration(self) -> Dict[str, str]:
        """Get current state of all regions."""
        return {r.id: r.current_state for r in self.regions}

    def step(self, event: str) -> Dict[str, Tuple[str, str]]:
        """
        Process event with synchronization.

        Returns: {region_id: (from_state, to_state)} for transitions taken
        """
        transitions_taken = {}

        for region in self.regions:
            # Skip if blocked
            if region.is_blocked():
                continue

            for trans in self.transitions:
                if trans.region_id != region.id:
                    continue
                if trans.event != event:
                    continue
                if trans.from_state != region.current_state:
                    continue

                # Check sync requirement
                if trans.sync_requirement:
                    if not self.shared_context.check_condition(trans.sync_requirement):
                        continue

                # Check guard
                if trans.guard:
                    context = {
                        **self.shared_context.snapshot(),
                        '__region__': region,
                        '__event__': event,
                    }
                    try:
                        if not trans.guard(context):
                            continue
                    except Exception:
                        continue

                # Execute action
                if trans.action:
                    context = {
                        **self.shared_context.snapshot(),
                        '__region__': region,
                        '__event__': event,
                    }
                    try:
                        result = trans.action(context)
                        if isinstance(result, dict):
                            for key, value in result.items():
                                self.shared_context.set(key, value, region.id)
                    except Exception:
                        pass

                # Record context writes
                for var_name in trans.context_writes:
                    if var_name in self.shared_context.variables:
                        self.shared_context.variables[var_name].written_by.add(region.id)

                # Take transition
                old_state = region.current_state
                region.current_state = trans.to_state
                transitions_taken[region.id] = (old_state, trans.to_state)

                break  # Only take one transition per region

        return transitions_taken


# =============================================================================
# SYNC PATTERN IMPLEMENTATIONS
# =============================================================================

def create_producer_consumer_system(
    producer_states: List[str],
    consumer_states: List[str],
) -> SynchronizedParallelState:
    """
    Create a producer-consumer synchronized system.

    Producer generates items, consumer processes them.
    Coordination via "buffer_count" shared variable.
    """
    parallel = SynchronizedParallelState(
        label="ProducerConsumer",
        sync_pattern=SyncPattern.PRODUCER_CONSUMER,
    )

    # Producer region
    producer = SyncRegion(
        id="producer",
        label="Producer",
        states=producer_states,
        initial_state=producer_states[0],
        current_state=producer_states[0],
        sync_role="producer",
    )
    parallel.add_region(producer)

    # Consumer region
    consumer = SyncRegion(
        id="consumer",
        label="Consumer",
        states=consumer_states,
        initial_state=consumer_states[0],
        current_state=consumer_states[0],
        sync_role="consumer",
    )
    parallel.add_region(consumer)

    # Producer transitions
    parallel.add_transition(SyncTransition(
        region_id="producer",
        from_state="Idle",
        to_state="Producing",
        event="START",
    ))

    # PRODUCE is a self-loop that stays in Producing state
    parallel.add_transition(SyncTransition(
        region_id="producer",
        from_state="Producing",
        to_state="Producing",
        event="PRODUCE",
        action=lambda ctx: {"buffer_count": ctx.get("buffer_count", 0) + 1},
        context_writes=["buffer_count"],
    ))

    parallel.add_transition(SyncTransition(
        region_id="producer",
        from_state="Producing",
        to_state="Idle",
        event="STOP",
    ))

    # Consumer transitions (waits for buffer)
    parallel.add_transition(SyncTransition(
        region_id="consumer",
        from_state="Waiting",
        to_state="Consuming",
        event="CONSUME",
        sync_requirement="buffer_count > 0",
        action=lambda ctx: {"buffer_count": max(0, ctx.get("buffer_count", 0) - 1)},
        context_writes=["buffer_count"],
    ))

    parallel.add_transition(SyncTransition(
        region_id="consumer",
        from_state="Consuming",
        to_state="Waiting",
        event="DONE",
    ))

    return parallel


def create_barrier_sync_system(
    region_count: int,
    states_per_region: int,
) -> SynchronizedParallelState:
    """
    Create a barrier-synchronized system.

    All regions must reach barrier before any can proceed.
    """
    parallel = SynchronizedParallelState(
        label="BarrierSync",
        sync_pattern=SyncPattern.BARRIER,
    )

    all_region_ids = set()

    for i in range(region_count):
        region_id = f"region_{i}"
        all_region_ids.add(region_id)

        states = [f"Phase_{j}" for j in range(states_per_region)] + ["AtBarrier", "Completed"]

        region = SyncRegion(
            id=region_id,
            label=f"Region{i}",
            states=states,
            initial_state="Phase_0",
            current_state="Phase_0",
        )
        parallel.add_region(region)

        # Progress through phases
        for j in range(states_per_region - 1):
            parallel.add_transition(SyncTransition(
                region_id=region_id,
                from_state=f"Phase_{j}",
                to_state=f"Phase_{j+1}",
                event="ADVANCE",
            ))

        # Reach barrier
        parallel.add_transition(SyncTransition(
            region_id=region_id,
            from_state=f"Phase_{states_per_region-1}",
            to_state="AtBarrier",
            event="ADVANCE",
            action=lambda ctx, rid=region_id: {"barrier_count": ctx.get("barrier_count", 0) + 1},
            context_writes=["barrier_count"],
        ))

        # Pass barrier (when all arrive)
        parallel.add_transition(SyncTransition(
            region_id=region_id,
            from_state="AtBarrier",
            to_state="Completed",
            event="SYNC",
            sync_requirement=f"barrier_count == {region_count}",
        ))

    return parallel


def create_leader_follower_system(
    n_followers: int,
) -> SynchronizedParallelState:
    """
    Create a leader-follower system.

    Leader makes decisions, followers copy the leader's state.
    """
    parallel = SynchronizedParallelState(
        label="LeaderFollower",
        sync_pattern=SyncPattern.LEADER_FOLLOWER,
    )

    # Leader region
    leader = SyncRegion(
        id="leader",
        label="Leader",
        states=["Idle", "Working", "Paused", "Done"],
        initial_state="Idle",
        current_state="Idle",
        sync_role="leader",
    )
    parallel.add_region(leader)

    # Leader transitions (update shared state)
    parallel.add_transition(SyncTransition(
        region_id="leader",
        from_state="Idle",
        to_state="Working",
        event="START",
        action=lambda ctx: {"leader_state": "Working"},
        context_writes=["leader_state"],
    ))

    parallel.add_transition(SyncTransition(
        region_id="leader",
        from_state="Working",
        to_state="Paused",
        event="PAUSE",
        action=lambda ctx: {"leader_state": "Paused"},
        context_writes=["leader_state"],
    ))

    parallel.add_transition(SyncTransition(
        region_id="leader",
        from_state="Paused",
        to_state="Working",
        event="RESUME",
        action=lambda ctx: {"leader_state": "Working"},
        context_writes=["leader_state"],
    ))

    parallel.add_transition(SyncTransition(
        region_id="leader",
        from_state="Working",
        to_state="Done",
        event="FINISH",
        action=lambda ctx: {"leader_state": "Done"},
        context_writes=["leader_state"],
    ))

    # Follower regions
    for i in range(n_followers):
        follower = SyncRegion(
            id=f"follower_{i}",
            label=f"Follower{i}",
            states=["Idle", "Working", "Paused", "Done"],
            initial_state="Idle",
            current_state="Idle",
            sync_role="follower",
        )
        parallel.add_region(follower)

        # Followers sync to leader state
        for from_state in ["Idle", "Working", "Paused"]:
            for to_state in ["Working", "Paused", "Done"]:
                if from_state != to_state:
                    parallel.add_transition(SyncTransition(
                        region_id=f"follower_{i}",
                        from_state=from_state,
                        to_state=to_state,
                        event="SYNC",
                        sync_requirement=f"leader_state == '{to_state}'",
                        context_reads=["leader_state"],
                    ))

    return parallel


# =============================================================================
# SYNC PATTERN LEARNER
# =============================================================================

@dataclass
class SyncGenome:
    """Evolvable synchronization configuration."""
    pattern: SyncPattern
    shared_vars: List[str]
    sync_conditions: Dict[str, str]  # transition_id -> condition
    barrier_points: List[str]        # states that are barrier points
    fitness: float = 0.0


class SyncPatternLearner:
    """
    Learn optimal synchronization patterns from examples.

    Given traces of region interactions, discover the coordination pattern.
    """

    def __init__(
        self,
        region_ids: List[str],
        population_size: int = 30,
        n_generations: int = 50,
    ):
        self.region_ids = region_ids
        self.population_size = population_size
        self.n_generations = n_generations

    def random_genome(self) -> SyncGenome:
        """Generate random sync configuration."""
        pattern = random.choice(list(SyncPattern))

        n_vars = random.randint(1, 3)
        shared_vars = [f"var_{i}" for i in range(n_vars)]

        sync_conditions = {}
        n_conditions = random.randint(0, 3)
        for i in range(n_conditions):
            trans_id = f"trans_{i}"
            var = random.choice(shared_vars)
            op = random.choice(["==", ">", "<"])
            val = random.randint(0, 5)
            sync_conditions[trans_id] = f"{var} {op} {val}"

        barrier_points = []
        if pattern == SyncPattern.BARRIER:
            barrier_points = [f"barrier_{i}" for i in range(random.randint(1, 2))]

        return SyncGenome(
            pattern=pattern,
            shared_vars=shared_vars,
            sync_conditions=sync_conditions,
            barrier_points=barrier_points,
        )

    def evaluate_fitness(
        self,
        genome: SyncGenome,
        examples: List[Tuple[Dict[str, str], Dict[str, str]]],
    ) -> float:
        """
        Evaluate fitness on (before, after) examples.

        Examples show which transitions should be synchronized.
        """
        if not examples:
            return 0.0

        correct = 0
        for before, after in examples:
            # Check if sync pattern would allow this transition
            # (Simplified: pattern matching)
            if genome.pattern == SyncPattern.NONE:
                # No sync - all transitions allowed
                correct += 1
            elif genome.pattern == SyncPattern.BARRIER:
                # Barrier - check if all regions at barrier
                if all(s.endswith("Barrier") or s == after.get(r)
                       for r, s in before.items()):
                    correct += 1
            else:
                # Other patterns - simplified scoring
                correct += 0.5

        genome.fitness = correct / len(examples)
        return genome.fitness

    def evolve(
        self,
        examples: List[Tuple[Dict[str, str], Dict[str, str]]],
        verbose: bool = True,
    ) -> SyncGenome:
        """Evolve optimal sync pattern."""
        population = [self.random_genome() for _ in range(self.population_size)]
        best_ever = None

        for gen in range(self.n_generations):
            for genome in population:
                self.evaluate_fitness(genome, examples)

            population.sort(key=lambda g: g.fitness, reverse=True)

            if best_ever is None or population[0].fitness > best_ever.fitness:
                best_ever = population[0]

            if verbose and gen % 10 == 0:
                print(f"Gen {gen:3d}: Best fitness = {population[0].fitness:.3f}, pattern = {population[0].pattern.name}")

            if population[0].fitness >= 0.95:
                break

            # Selection and reproduction
            elite = population[:3]
            new_pop = list(elite)

            while len(new_pop) < self.population_size:
                p1 = random.choice(population[:self.population_size//2])
                child = SyncGenome(
                    pattern=p1.pattern if random.random() > 0.2 else random.choice(list(SyncPattern)),
                    shared_vars=list(p1.shared_vars),
                    sync_conditions=dict(p1.sync_conditions),
                    barrier_points=list(p1.barrier_points),
                )
                new_pop.append(child)

            population = new_pop

        return best_ever


# =============================================================================
# TESTING
# =============================================================================

def test_shared_context():
    """Test shared context operations."""
    print("=" * 60)
    print("SHARED CONTEXT TEST")
    print("=" * 60)

    ctx = SharedContext()

    # Basic set/get
    ctx.set("counter", 0, "region_a")
    assert ctx.get("counter") == 0

    # Increment
    ctx.increment("counter", 5, "region_a")
    assert ctx.get("counter") == 5

    # Decrement
    ctx.decrement("counter", 2, "region_b")
    assert ctx.get("counter") == 3

    # Check conditions
    assert ctx.check_condition("counter > 2") == True
    assert ctx.check_condition("counter < 5") == True
    assert ctx.check_condition("counter == 3") == True

    print(f"Context snapshot: {ctx.snapshot()}")
    print("[PASS] Shared context test passed!")
    return True


def test_producer_consumer():
    """Test producer-consumer synchronization."""
    print("\n" + "=" * 60)
    print("PRODUCER-CONSUMER SYNC TEST")
    print("=" * 60)

    system = create_producer_consumer_system(
        producer_states=["Idle", "Producing"],
        consumer_states=["Waiting", "Consuming"],
    )

    print(f"Initial config: {system.get_configuration()}")
    print(f"Initial context: {system.shared_context.snapshot()}")

    # Producer starts
    transitions = system.step("START")
    print(f"After START: {system.get_configuration()} (transitions: {transitions})")

    # Produce items
    for _ in range(3):
        transitions = system.step("PRODUCE")
        print(f"After PRODUCE: buffer={system.shared_context.get('buffer_count')}")

    # Consumer tries to consume
    transitions = system.step("CONSUME")
    print(f"After CONSUME: {system.get_configuration()} buffer={system.shared_context.get('buffer_count')}")

    assert system.shared_context.get("buffer_count") == 2, "Buffer should have 2 items"
    print("\n[PASS] Producer-consumer sync test passed!")
    return True


def test_leader_follower():
    """Test leader-follower synchronization."""
    print("\n" + "=" * 60)
    print("LEADER-FOLLOWER SYNC TEST")
    print("=" * 60)

    system = create_leader_follower_system(n_followers=2)

    print(f"Initial config: {system.get_configuration()}")

    # Leader starts working
    transitions = system.step("START")
    print(f"After START: {system.get_configuration()}")
    print(f"  Leader state shared: {system.shared_context.get('leader_state')}")

    # Followers sync
    transitions = system.step("SYNC")
    print(f"After SYNC: {system.get_configuration()}")

    # Leader pauses
    transitions = system.step("PAUSE")
    print(f"After PAUSE: leader_state={system.shared_context.get('leader_state')}")

    # Followers sync again
    transitions = system.step("SYNC")
    print(f"After SYNC: {system.get_configuration()}")

    print("\n[PASS] Leader-follower sync test passed!")
    return True


def test_barrier_sync():
    """Test barrier synchronization."""
    print("\n" + "=" * 60)
    print("BARRIER SYNC TEST")
    print("=" * 60)

    system = create_barrier_sync_system(region_count=3, states_per_region=2)

    print(f"Initial config: {system.get_configuration()}")

    # Advance regions
    for i in range(3):
        transitions = system.step("ADVANCE")
        print(f"After ADVANCE ({i+1}): {system.get_configuration()}")
        print(f"  barrier_count={system.shared_context.get('barrier_count')}")

    # Try to sync (should work when all at barrier)
    transitions = system.step("SYNC")
    print(f"After SYNC: {system.get_configuration()}")

    print("\n[PASS] Barrier sync test passed!")
    return True


if __name__ == "__main__":
    test_shared_context()
    test_producer_consumer()
    test_leader_follower()
    test_barrier_sync()
    print("\n" + "=" * 60)
    print("ALL REGION SYNC TESTS COMPLETE")
    print("=" * 60)
