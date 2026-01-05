"""
Temporal Guard Synthesis

Extends guard synthesis with TIME-BASED predicates for statechart transitions.
This enables learning guards that depend on temporal properties like:
- How long we've been in a state (timeouts)
- Time since a previous state was active
- Deadline constraints
- Rate limiting patterns

Key Research Contribution:
  TEMPORAL GUARDS ARE LEARNABLE - evolution can discover time-based
  conditions from timestamped traces, enabling automatic extraction
  of timeout logic, rate limiters, and deadline patterns.

Applications:
  - Session timeout discovery
  - Game clock constraints
  - Rate limiting policies
  - Deadline-driven transitions
  - Cooldown mechanics
"""

import mlx.core as mx
from dataclasses import dataclass, field
from typing import List, Dict, Set, Tuple, Optional, Callable, Any
from enum import Enum, auto
import random
import time
from abc import ABC, abstractmethod

# Import base classes from guard_synthesizer
import sys
sys.path.insert(0, '/Users/tmc/go/src/github.com/tmc/sc/ml/experiments')
from exp_guard_synthesis.guard_synthesizer import (
    Expr, ExprType, Const, Var, BinOp, UnaryOp, FuncCall,
    GuardGenome, GuardSynthesizer
)


# =============================================================================
# TEMPORAL EXPRESSION AST EXTENSIONS
# =============================================================================

@dataclass
class TemporalExpr(Expr):
    """Base class for temporal expressions."""
    pass


@dataclass
class After(TemporalExpr):
    """
    after(duration) - True if time since entering current state >= duration.

    Example: after(30) is true if we've been in current state for 30+ seconds.
    """
    duration: float  # seconds

    def __post_init__(self):
        self.expr_type = ExprType.BOOL

    def evaluate(self, context: Dict[str, Any]) -> bool:
        current_time = context.get('__timestamp__', time.time())
        entry_time = context.get('__state_entry_time__', current_time)
        elapsed = current_time - entry_time
        return elapsed >= self.duration

    def to_string(self) -> str:
        return f"after({self.duration}s)"


@dataclass
class Elapsed(TemporalExpr):
    """
    elapsed - Returns time (in seconds) since entering current state.

    Used in comparisons: elapsed > 30, elapsed < 5, etc.
    """
    def __post_init__(self):
        self.expr_type = ExprType.FLOAT

    def evaluate(self, context: Dict[str, Any]) -> float:
        current_time = context.get('__timestamp__', time.time())
        entry_time = context.get('__state_entry_time__', current_time)
        return current_time - entry_time

    def to_string(self) -> str:
        return "elapsed"


@dataclass
class Since(TemporalExpr):
    """
    since(state_name) - Time since leaving a specific state.

    Example: since('LOGIN') > 3600 (been over an hour since login state)
    """
    state_name: str

    def __post_init__(self):
        self.expr_type = ExprType.FLOAT

    def evaluate(self, context: Dict[str, Any]) -> float:
        current_time = context.get('__timestamp__', time.time())
        state_exit_times = context.get('__state_exit_times__', {})
        exit_time = state_exit_times.get(self.state_name, 0)
        if exit_time == 0:
            return float('inf')  # Never been in that state
        return current_time - exit_time

    def to_string(self) -> str:
        return f"since('{self.state_name}')"


@dataclass
class Within(TemporalExpr):
    """
    within(duration) - True if still within deadline since entering state.

    Example: within(5) is true if we've been in state < 5 seconds.
    Opposite of after().
    """
    duration: float  # seconds

    def __post_init__(self):
        self.expr_type = ExprType.BOOL

    def evaluate(self, context: Dict[str, Any]) -> bool:
        current_time = context.get('__timestamp__', time.time())
        entry_time = context.get('__state_entry_time__', current_time)
        elapsed = current_time - entry_time
        return elapsed < self.duration

    def to_string(self) -> str:
        return f"within({self.duration}s)"


@dataclass
class RateLimit(TemporalExpr):
    """
    rate_limit(count, window) - True if event count within window is under limit.

    Example: rate_limit(5, 60) is true if < 5 events in last 60 seconds.
    """
    count: int
    window: float  # seconds

    def __post_init__(self):
        self.expr_type = ExprType.BOOL

    def evaluate(self, context: Dict[str, Any]) -> bool:
        current_time = context.get('__timestamp__', time.time())
        event_times = context.get('__event_times__', [])

        # Count events within window
        cutoff = current_time - self.window
        recent_count = sum(1 for t in event_times if t > cutoff)

        return recent_count < self.count

    def to_string(self) -> str:
        return f"rate_limit({self.count}, {self.window}s)"


@dataclass
class Cooldown(TemporalExpr):
    """
    cooldown(duration) - True if enough time has passed since last transition.

    Example: cooldown(2) is true if 2+ seconds since last transition fired.
    """
    duration: float  # seconds

    def __post_init__(self):
        self.expr_type = ExprType.BOOL

    def evaluate(self, context: Dict[str, Any]) -> bool:
        current_time = context.get('__timestamp__', time.time())
        last_transition = context.get('__last_transition_time__', 0)

        if last_transition == 0:
            return True  # No previous transition

        return (current_time - last_transition) >= self.duration

    def to_string(self) -> str:
        return f"cooldown({self.duration}s)"


@dataclass
class TimeOfDay(TemporalExpr):
    """
    time_of_day() - Returns hour of day (0-23).

    Useful for: business hours checks, night mode, scheduled transitions.
    """
    def __post_init__(self):
        self.expr_type = ExprType.INT

    def evaluate(self, context: Dict[str, Any]) -> int:
        import datetime
        timestamp = context.get('__timestamp__', time.time())
        dt = datetime.datetime.fromtimestamp(timestamp)
        return dt.hour

    def to_string(self) -> str:
        return "time_of_day()"


# =============================================================================
# TEMPORAL GUARD GENOME
# =============================================================================

@dataclass
class TemporalGuardGenome(GuardGenome):
    """Guard genome with temporal expression support."""
    learned_durations: List[float] = field(default_factory=list)

    def get_temporal_summary(self) -> str:
        """Summarize temporal aspects of the guard."""
        summary = []

        def scan_temporal(expr):
            if isinstance(expr, After):
                summary.append(f"timeout: {expr.duration}s")
            elif isinstance(expr, Within):
                summary.append(f"deadline: {expr.duration}s")
            elif isinstance(expr, Since):
                summary.append(f"since_state: {expr.state_name}")
            elif isinstance(expr, RateLimit):
                summary.append(f"rate: {expr.count}/{expr.window}s")
            elif isinstance(expr, Cooldown):
                summary.append(f"cooldown: {expr.duration}s")
            elif isinstance(expr, BinOp):
                scan_temporal(expr.left)
                scan_temporal(expr.right)
            elif isinstance(expr, UnaryOp):
                scan_temporal(expr.operand)

        scan_temporal(self.expr)
        return ", ".join(summary) if summary else "no temporal constraints"


# =============================================================================
# TEMPORAL GUARD SYNTHESIZER
# =============================================================================

class TemporalGuardSynthesizer(GuardSynthesizer):
    """
    Synthesize guards with temporal predicates.

    Extends GuardSynthesizer with:
    - Temporal expression types (after, within, since, etc.)
    - Duration evolution (learning optimal timeouts)
    - Timestamped trace handling
    """

    DURATION_CANDIDATES = [
        0.5, 1, 2, 3, 5, 10, 15, 30, 60, 120, 300, 600, 1800, 3600
    ]  # Common timeout values in seconds

    def __init__(
        self,
        variables: List[str],
        constants: List[Any],
        functions: Dict[str, Callable] = None,
        max_depth: int = 4,
        state_names: List[str] = None,  # For since() expressions
    ):
        super().__init__(variables, constants, functions, max_depth)
        self.state_names = state_names or []

    def random_temporal_expr(self, depth: int = 0) -> Expr:
        """Generate a random temporal expression."""
        choice = random.random()

        if choice < 0.2:
            # after(duration)
            duration = random.choice(self.DURATION_CANDIDATES)
            return After(duration=duration, expr_type=ExprType.BOOL)

        elif choice < 0.4:
            # within(duration)
            duration = random.choice(self.DURATION_CANDIDATES)
            return Within(duration=duration, expr_type=ExprType.BOOL)

        elif choice < 0.55:
            # elapsed > N
            duration = random.choice(self.DURATION_CANDIDATES)
            return BinOp(
                op=random.choice(['>', '<', '>=', '<=']),
                left=Elapsed(expr_type=ExprType.FLOAT),
                right=Const(expr_type=ExprType.FLOAT, value=duration),
                expr_type=ExprType.BOOL,
            )

        elif choice < 0.7 and self.state_names:
            # since(state) > N
            state = random.choice(self.state_names)
            duration = random.choice(self.DURATION_CANDIDATES)
            return BinOp(
                op=random.choice(['>', '<', '>=', '<=']),
                left=Since(state_name=state, expr_type=ExprType.FLOAT),
                right=Const(expr_type=ExprType.FLOAT, value=duration),
                expr_type=ExprType.BOOL,
            )

        elif choice < 0.8:
            # rate_limit(count, window)
            count = random.choice([1, 3, 5, 10, 20, 50, 100])
            window = random.choice([1, 5, 10, 60, 300, 3600])
            return RateLimit(count=count, window=window, expr_type=ExprType.BOOL)

        elif choice < 0.9:
            # cooldown(duration)
            duration = random.choice(self.DURATION_CANDIDATES[:8])  # Shorter cooldowns
            return Cooldown(duration=duration, expr_type=ExprType.BOOL)

        else:
            # time_of_day comparison
            hour = random.randint(0, 23)
            return BinOp(
                op=random.choice(['==', '>=', '<=']),
                left=TimeOfDay(expr_type=ExprType.INT),
                right=Const(expr_type=ExprType.INT, value=hour),
                expr_type=ExprType.BOOL,
            )

    def random_expr(self, depth: int = 0, target_type: ExprType = ExprType.BOOL) -> Expr:
        """Generate random expression, including temporal ones."""
        if depth >= self.max_depth:
            return super().random_expr(depth, target_type)

        # 40% chance of temporal expression at top level
        if target_type == ExprType.BOOL and random.random() < 0.4:
            return self.random_temporal_expr(depth)

        # 30% chance of combining temporal with non-temporal
        if target_type == ExprType.BOOL and random.random() < 0.3:
            temporal = self.random_temporal_expr(depth)
            non_temporal = super().random_expr(depth + 1, ExprType.BOOL)
            op = random.choice(['and', 'or'])
            return BinOp(
                op=op,
                left=temporal,
                right=non_temporal,
                expr_type=ExprType.BOOL,
            )

        return super().random_expr(depth, target_type)

    def mutate_duration(self, duration: float) -> float:
        """Mutate a duration value."""
        if random.random() < 0.3:
            # Small adjustment
            factor = random.uniform(0.5, 2.0)
            return duration * factor
        elif random.random() < 0.5:
            # Pick from candidates
            return random.choice(self.DURATION_CANDIDATES)
        else:
            # Keep as is
            return duration

    def mutate(self, genome: GuardGenome, mutation_rate: float = 0.3) -> TemporalGuardGenome:
        """Mutate guard with special handling for temporal expressions."""

        def mutate_expr(expr: Expr, depth: int = 0) -> Expr:
            if random.random() < mutation_rate:
                # Replace with new random
                return self.random_expr(depth, expr.expr_type)

            if isinstance(expr, After):
                new_duration = self.mutate_duration(expr.duration)
                return After(duration=new_duration, expr_type=ExprType.BOOL)

            elif isinstance(expr, Within):
                new_duration = self.mutate_duration(expr.duration)
                return Within(duration=new_duration, expr_type=ExprType.BOOL)

            elif isinstance(expr, Since):
                if self.state_names and random.random() < 0.3:
                    new_state = random.choice(self.state_names)
                    return Since(state_name=new_state, expr_type=ExprType.FLOAT)
                return expr

            elif isinstance(expr, RateLimit):
                new_count = expr.count
                new_window = expr.window
                if random.random() < 0.5:
                    new_count = max(1, expr.count + random.randint(-2, 2))
                if random.random() < 0.5:
                    new_window = self.mutate_duration(expr.window)
                return RateLimit(count=new_count, window=new_window, expr_type=ExprType.BOOL)

            elif isinstance(expr, Cooldown):
                new_duration = self.mutate_duration(expr.duration)
                return Cooldown(duration=new_duration, expr_type=ExprType.BOOL)

            elif isinstance(expr, BinOp):
                return BinOp(
                    op=expr.op,
                    left=mutate_expr(expr.left, depth + 1),
                    right=mutate_expr(expr.right, depth + 1),
                    expr_type=expr.expr_type,
                )

            elif isinstance(expr, UnaryOp):
                return UnaryOp(
                    op=expr.op,
                    operand=mutate_expr(expr.operand, depth + 1),
                    expr_type=expr.expr_type,
                )

            elif isinstance(expr, Const):
                if isinstance(expr.value, (int, float)) and random.random() < 0.3:
                    new_val = expr.value * random.uniform(0.5, 2.0)
                    return Const(expr_type=expr.expr_type, value=new_val)
                return expr

            return expr

        new_expr = mutate_expr(genome.expr)
        return TemporalGuardGenome(expr=new_expr)

    def evolve(
        self,
        positive_examples: List[Dict],
        negative_examples: List[Dict],
        population_size: int = 50,
        n_generations: int = 100,
        verbose: bool = True,
    ) -> TemporalGuardGenome:
        """Evolve temporal guard expression."""

        # Initialize with TemporalGuardGenome
        population = [
            TemporalGuardGenome(expr=self.random_expr())
            for _ in range(population_size)
        ]

        best_ever = None

        for gen in range(n_generations):
            # Evaluate
            for genome in population:
                self.evaluate_fitness(genome, positive_examples, negative_examples)

            # Sort by fitness
            population.sort(key=lambda g: g.fitness, reverse=True)

            # Track best
            if best_ever is None or population[0].fitness > best_ever.fitness:
                best = population[0]
                best_ever = TemporalGuardGenome(
                    expr=best.expr,
                    fitness=best.fitness,
                    true_positives=best.true_positives,
                    false_positives=best.false_positives,
                    true_negatives=best.true_negatives,
                    false_negatives=best.false_negatives,
                )

            if verbose and gen % 10 == 0:
                best = population[0]
                temporal_info = best.get_temporal_summary() if hasattr(best, 'get_temporal_summary') else ""
                print(f"Gen {gen:3d}: F1={best.f1:.3f} | {temporal_info}")
                print(f"         {best.expr.to_string()[:70]}...")

            # Perfect solution?
            if population[0].f1 >= 0.99:
                break

            # Selection and reproduction
            elite = population[:5]
            new_pop = [TemporalGuardGenome(expr=g.expr) for g in elite]

            while len(new_pop) < population_size:
                tournament = random.sample(population[:population_size//2], 3)
                parent1 = max(tournament, key=lambda g: g.fitness)

                tournament = random.sample(population[:population_size//2], 3)
                parent2 = max(tournament, key=lambda g: g.fitness)

                if random.random() < 0.7:
                    child = self.crossover(parent1, parent2)
                    child = TemporalGuardGenome(expr=child.expr)
                else:
                    child = TemporalGuardGenome(expr=parent1.expr)

                child = self.mutate(child)
                new_pop.append(child)

            population = new_pop

        return best_ever


# =============================================================================
# TEST SCENARIOS
# =============================================================================

def generate_session_timeout_data(
    timeout_seconds: float = 30.0,
    n_samples: int = 100,
) -> Tuple[List[Dict], List[Dict]]:
    """
    Generate training data for session timeout detection.

    Positive: idle for >= timeout_seconds
    Negative: active or idle < timeout_seconds
    """
    positive = []
    negative = []

    base_time = time.time()

    for _ in range(n_samples):
        # Positive: session should timeout
        idle_time = random.uniform(timeout_seconds, timeout_seconds * 3)
        positive.append({
            '__timestamp__': base_time,
            '__state_entry_time__': base_time - idle_time,
            'user_active': False,
            'requests_pending': 0,
        })

        # Negative: session should stay active
        if random.random() < 0.5:
            # Recently active
            idle_time = random.uniform(0, timeout_seconds * 0.9)
            negative.append({
                '__timestamp__': base_time,
                '__state_entry_time__': base_time - idle_time,
                'user_active': random.choice([True, False]),
                'requests_pending': random.randint(0, 3),
            })
        else:
            # Active user
            negative.append({
                '__timestamp__': base_time,
                '__state_entry_time__': base_time - random.uniform(0, 60),
                'user_active': True,
                'requests_pending': random.randint(1, 5),
            })

    return positive, negative


def generate_rate_limit_data(
    max_requests: int = 5,
    window_seconds: float = 60.0,
    n_samples: int = 100,
) -> Tuple[List[Dict], List[Dict]]:
    """
    Generate training data for rate limiting.

    Positive: request count within window exceeds limit
    Negative: under limit
    """
    positive = []
    negative = []

    base_time = time.time()

    for _ in range(n_samples):
        # Positive: should be rate limited
        event_count = random.randint(max_requests, max_requests * 2)
        event_times = [
            base_time - random.uniform(0, window_seconds * 0.9)
            for _ in range(event_count)
        ]
        positive.append({
            '__timestamp__': base_time,
            '__event_times__': event_times,
            'request_type': random.choice(['api', 'web', 'rpc']),
        })

        # Negative: under rate limit
        event_count = random.randint(0, max_requests - 1)
        event_times = [
            base_time - random.uniform(0, window_seconds * 1.5)
            for _ in range(event_count)
        ]
        negative.append({
            '__timestamp__': base_time,
            '__event_times__': event_times,
            'request_type': random.choice(['api', 'web', 'rpc']),
        })

    return positive, negative


def generate_game_clock_data(
    move_time_limit: float = 30.0,
    n_samples: int = 100,
) -> Tuple[List[Dict], List[Dict]]:
    """
    Generate training data for game move timeouts.

    Positive: player took too long, should forfeit turn
    Negative: player moved in time
    """
    positive = []
    negative = []

    base_time = time.time()

    for _ in range(n_samples):
        # Positive: time exceeded
        think_time = random.uniform(move_time_limit, move_time_limit * 3)
        positive.append({
            '__timestamp__': base_time,
            '__state_entry_time__': base_time - think_time,
            'current_player': random.choice(['white', 'black']),
            'move_count': random.randint(1, 50),
            'game_phase': random.choice(['opening', 'middle', 'endgame']),
        })

        # Negative: moved in time
        think_time = random.uniform(0, move_time_limit * 0.9)
        negative.append({
            '__timestamp__': base_time,
            '__state_entry_time__': base_time - think_time,
            'current_player': random.choice(['white', 'black']),
            'move_count': random.randint(1, 50),
            'game_phase': random.choice(['opening', 'middle', 'endgame']),
        })

    return positive, negative


def generate_cooldown_data(
    cooldown_seconds: float = 5.0,
    n_samples: int = 100,
) -> Tuple[List[Dict], List[Dict]]:
    """
    Generate training data for ability cooldowns.

    Positive: cooldown complete, can use ability
    Negative: still on cooldown
    """
    positive = []
    negative = []

    base_time = time.time()

    for _ in range(n_samples):
        # Positive: cooldown complete
        last_use = base_time - random.uniform(cooldown_seconds, cooldown_seconds * 3)
        positive.append({
            '__timestamp__': base_time,
            '__last_transition_time__': last_use,
            'ability': random.choice(['fireball', 'heal', 'shield']),
            'mana': random.randint(50, 100),
        })

        # Negative: still on cooldown
        last_use = base_time - random.uniform(0, cooldown_seconds * 0.9)
        negative.append({
            '__timestamp__': base_time,
            '__last_transition_time__': last_use,
            'ability': random.choice(['fireball', 'heal', 'shield']),
            'mana': random.randint(0, 100),
        })

    return positive, negative


# =============================================================================
# DEMOS
# =============================================================================

def demo_session_timeout():
    """Learn session timeout guard from examples."""
    print("=" * 70)
    print("TEMPORAL GUARD SYNTHESIS: Session Timeout")
    print("=" * 70)

    TIMEOUT = 30.0
    positive, negative = generate_session_timeout_data(TIMEOUT, 50)

    print(f"\nTarget: Learn guard for {TIMEOUT}s timeout")
    print(f"Positive examples (should timeout): {len(positive)}")
    print(f"Negative examples (stay active): {len(negative)}")

    synth = TemporalGuardSynthesizer(
        variables=['user_active', 'requests_pending'],
        constants=[True, False, 0, 1, 5],
        max_depth=3,
    )

    print("\nEvolving temporal guard...")
    best = synth.evolve(positive, negative, population_size=40, n_generations=60, verbose=True)

    print(f"\n{'='*70}")
    print(f"LEARNED GUARD: {best.expr.to_string()}")
    print(f"Temporal Summary: {best.get_temporal_summary()}")
    print(f"F1={best.f1:.3f}, P={best.precision:.3f}, R={best.recall:.3f}")
    print(f"{'='*70}")

    return best


def demo_rate_limit():
    """Learn rate limiting guard from examples."""
    print("\n" + "=" * 70)
    print("TEMPORAL GUARD SYNTHESIS: Rate Limiting")
    print("=" * 70)

    MAX_REQ = 5
    WINDOW = 60.0
    positive, negative = generate_rate_limit_data(MAX_REQ, WINDOW, 50)

    print(f"\nTarget: Learn rate limit of {MAX_REQ} requests per {WINDOW}s")
    print(f"Positive examples (should block): {len(positive)}")
    print(f"Negative examples (allow): {len(negative)}")

    synth = TemporalGuardSynthesizer(
        variables=['request_type'],
        constants=['api', 'web', 'rpc'],
        max_depth=3,
    )

    print("\nEvolving temporal guard...")
    best = synth.evolve(positive, negative, population_size=40, n_generations=60, verbose=True)

    print(f"\n{'='*70}")
    print(f"LEARNED GUARD: {best.expr.to_string()}")
    print(f"Temporal Summary: {best.get_temporal_summary()}")
    print(f"F1={best.f1:.3f}, P={best.precision:.3f}, R={best.recall:.3f}")
    print(f"{'='*70}")

    return best


def demo_game_clock():
    """Learn game clock timeout guard."""
    print("\n" + "=" * 70)
    print("TEMPORAL GUARD SYNTHESIS: Game Clock")
    print("=" * 70)

    MOVE_TIME = 30.0
    positive, negative = generate_game_clock_data(MOVE_TIME, 50)

    print(f"\nTarget: Learn move timeout of {MOVE_TIME}s")

    synth = TemporalGuardSynthesizer(
        variables=['current_player', 'move_count', 'game_phase'],
        constants=['white', 'black', 'opening', 'middle', 'endgame', 0, 10, 30],
        max_depth=3,
    )

    print("\nEvolving temporal guard...")
    best = synth.evolve(positive, negative, population_size=40, n_generations=60, verbose=True)

    print(f"\n{'='*70}")
    print(f"LEARNED GUARD: {best.expr.to_string()}")
    print(f"Temporal Summary: {best.get_temporal_summary()}")
    print(f"F1={best.f1:.3f}, P={best.precision:.3f}, R={best.recall:.3f}")
    print(f"{'='*70}")

    return best


def demo_cooldown():
    """Learn ability cooldown guard."""
    print("\n" + "=" * 70)
    print("TEMPORAL GUARD SYNTHESIS: Ability Cooldown")
    print("=" * 70)

    COOLDOWN = 5.0
    positive, negative = generate_cooldown_data(COOLDOWN, 50)

    print(f"\nTarget: Learn cooldown of {COOLDOWN}s")

    synth = TemporalGuardSynthesizer(
        variables=['ability', 'mana'],
        constants=['fireball', 'heal', 'shield', 0, 50, 100],
        max_depth=3,
    )

    print("\nEvolving temporal guard...")
    best = synth.evolve(positive, negative, population_size=40, n_generations=60, verbose=True)

    print(f"\n{'='*70}")
    print(f"LEARNED GUARD: {best.expr.to_string()}")
    print(f"Temporal Summary: {best.get_temporal_summary()}")
    print(f"F1={best.f1:.3f}, P={best.precision:.3f}, R={best.recall:.3f}")
    print(f"{'='*70}")

    return best


def test_temporal_guards():
    """Run all temporal guard synthesis tests."""
    print("\n" + "=" * 70)
    print("TEMPORAL GUARD SYNTHESIS - TEST SUITE")
    print("=" * 70)

    results = {}

    # Test 1: Session Timeout
    results['session_timeout'] = demo_session_timeout()

    # Test 2: Rate Limiting
    results['rate_limit'] = demo_rate_limit()

    # Test 3: Game Clock
    results['game_clock'] = demo_game_clock()

    # Test 4: Cooldown
    results['cooldown'] = demo_cooldown()

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Scenario':<20} {'F1':>8} {'Temporal Constraint':<40}")
    print("-" * 70)
    for name, guard in results.items():
        temporal = guard.get_temporal_summary()[:40]
        print(f"{name:<20} {guard.f1:>8.3f} {temporal:<40}")

    avg_f1 = sum(g.f1 for g in results.values()) / len(results)
    print("-" * 70)
    print(f"{'Average':<20} {avg_f1:>8.3f}")
    print("=" * 70)

    print("\nKEY INSIGHT: Temporal guards are LEARNABLE from timestamped traces!")
    print("Evolution discovers timeouts, rate limits, and cooldowns automatically.")

    return results


if __name__ == "__main__":
    test_temporal_guards()
