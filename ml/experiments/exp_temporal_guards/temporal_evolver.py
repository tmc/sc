"""
Temporal Guard Evolver

Evolves time-based guard conditions from timestamped examples.
NO HARDCODING - discovers optimal durations, thresholds, and patterns.

Key capabilities:
1. Duration evolution: Learn timeout values from data
2. Threshold discovery: Find rate limit parameters
3. Pattern detection: Identify cooldown/debounce needs
4. Compound guards: Combine temporal with non-temporal conditions

Reference: exp_guard_synthesis for base evolution framework
"""

import random
import time
import copy
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any, Callable

from .temporal_guard import (
    TemporalExpr, TemporalGuard, ExprType,
    After, Within, Elapsed, Since, Timeout,
    RateLimit, Cooldown, Debounce, BusinessHours,
    TimeOfDay, DayOfWeek,
    BinOp, UnaryOp, Const, Var,
    DURATION_CANDIDATES, RATE_LIMIT_COUNTS, RATE_LIMIT_WINDOWS,
)


# =============================================================================
# TEMPORAL GENOME
# =============================================================================

@dataclass
class TemporalGenome:
    """
    Evolvable temporal guard expression.

    Contains both the expression tree and fitness metrics.
    """
    expr: TemporalExpr
    fitness: float = 0.0
    true_positives: int = 0
    false_positives: int = 0
    true_negatives: int = 0
    false_negatives: int = 0

    # Learned parameters
    learned_durations: List[float] = field(default_factory=list)

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

    @property
    def accuracy(self) -> float:
        total = self.true_positives + self.false_positives + self.true_negatives + self.false_negatives
        if total == 0:
            return 0.0
        return (self.true_positives + self.true_negatives) / total

    def copy(self) -> 'TemporalGenome':
        return TemporalGenome(
            expr=copy.deepcopy(self.expr),
            fitness=self.fitness,
            true_positives=self.true_positives,
            false_positives=self.false_positives,
            true_negatives=self.true_negatives,
            false_negatives=self.false_negatives,
            learned_durations=self.learned_durations.copy(),
        )

    def get_temporal_summary(self) -> str:
        """Extract summary of temporal constraints."""
        def summarize(expr):
            summaries = []
            if isinstance(expr, After):
                summaries.append(f"timeout:{expr.duration:.1f}s")
            elif isinstance(expr, Within):
                summaries.append(f"deadline:{expr.duration:.1f}s")
            elif isinstance(expr, Since):
                summaries.append(f"since:{expr.state_name}")
            elif isinstance(expr, Timeout):
                summaries.append(f"timeout({expr.state_name}):{expr.duration:.1f}s")
            elif isinstance(expr, RateLimit):
                summaries.append(f"rate:{expr.count}/{expr.window:.0f}s")
            elif isinstance(expr, Cooldown):
                summaries.append(f"cooldown:{expr.duration:.1f}s")
            elif isinstance(expr, Debounce):
                summaries.append(f"debounce:{expr.duration:.1f}s")
            elif isinstance(expr, BusinessHours):
                summaries.append(f"hours:{expr.start_hour}-{expr.end_hour}")
            elif isinstance(expr, BinOp):
                if expr.left:
                    summaries.extend(summarize(expr.left))
                if expr.right:
                    summaries.extend(summarize(expr.right))
            elif isinstance(expr, UnaryOp):
                if expr.operand:
                    summaries.extend(summarize(expr.operand))
            return summaries

        items = summarize(self.expr)
        return ", ".join(items) if items else "no temporal constraints"


# =============================================================================
# TEMPORAL EVOLVER
# =============================================================================

class TemporalEvolver:
    """
    Evolves temporal guard expressions from examples.

    NO HARDCODING: All durations and thresholds are discovered
    through evolution from positive/negative examples.
    """

    def __init__(
        self,
        variables: List[str] = None,
        constants: List[Any] = None,
        state_names: List[str] = None,
        max_depth: int = 4,
    ):
        self.variables = variables or []
        self.constants = constants or [True, False, 0, 1, 5, 10]
        self.state_names = state_names or []
        self.max_depth = max_depth

    # =========================================================================
    # RANDOM EXPRESSION GENERATION
    # =========================================================================

    def random_duration(self) -> float:
        """Generate a random duration from candidates."""
        return random.choice(DURATION_CANDIDATES)

    def random_rate_limit_params(self) -> Tuple[int, float]:
        """Generate random rate limit parameters."""
        count = random.choice(RATE_LIMIT_COUNTS)
        window = random.choice(RATE_LIMIT_WINDOWS)
        return count, window

    def random_temporal_expr(self, depth: int = 0) -> TemporalExpr:
        """Generate a random temporal expression."""
        if depth >= self.max_depth:
            # Terminal: simple temporal predicate
            return After(duration=self.random_duration())

        choice = random.random()

        if choice < 0.15:
            # after(duration)
            return After(duration=self.random_duration())

        elif choice < 0.30:
            # within(duration)
            return Within(duration=self.random_duration())

        elif choice < 0.40:
            # elapsed > N comparison
            return BinOp(
                op=random.choice(['>', '<', '>=', '<=']),
                left=Elapsed(),
                right=Const(value=self.random_duration(), expr_type=ExprType.FLOAT),
                expr_type=ExprType.BOOL,
            )

        elif choice < 0.50 and self.state_names:
            # since(state) comparison
            state = random.choice(self.state_names)
            return BinOp(
                op=random.choice(['>', '<', '>=', '<=']),
                left=Since(state_name=state),
                right=Const(value=self.random_duration(), expr_type=ExprType.FLOAT),
                expr_type=ExprType.BOOL,
            )

        elif choice < 0.60:
            # rate_limit(count, window)
            count, window = self.random_rate_limit_params()
            return RateLimit(count=count, window=window)

        elif choice < 0.70:
            # cooldown(duration)
            duration = random.choice(DURATION_CANDIDATES[:8])  # Shorter cooldowns
            return Cooldown(duration=duration)

        elif choice < 0.78:
            # debounce(duration)
            duration = random.choice(DURATION_CANDIDATES[:5])  # Very short
            return Debounce(duration=duration)

        elif choice < 0.85:
            # time_of_day comparison
            hour = random.randint(0, 23)
            return BinOp(
                op=random.choice(['==', '>=', '<=', '<', '>']),
                left=TimeOfDay(),
                right=Const(value=hour, expr_type=ExprType.INT),
                expr_type=ExprType.BOOL,
            )

        elif choice < 0.92:
            # business_hours
            start = random.randint(6, 12)
            end = random.randint(start + 4, 22)
            return BusinessHours(start_hour=start, end_hour=end)

        else:
            # Compound temporal: combine two temporal exprs
            op = random.choice(['and', 'or'])
            return BinOp(
                op=op,
                left=self.random_temporal_expr(depth + 1),
                right=self.random_temporal_expr(depth + 1),
                expr_type=ExprType.BOOL,
            )

    def random_base_expr(self, depth: int = 0) -> TemporalExpr:
        """Generate a random non-temporal expression."""
        if depth >= self.max_depth or not self.variables:
            return Const(value=True, expr_type=ExprType.BOOL)

        choice = random.random()

        if choice < 0.4 and self.variables:
            # Variable comparison
            var = random.choice(self.variables)
            const = random.choice(self.constants)
            return BinOp(
                op=random.choice(['==', '!=']),
                left=Var(name=var, expr_type=ExprType.BOOL),
                right=Const(value=const, expr_type=ExprType.BOOL),
                expr_type=ExprType.BOOL,
            )

        elif choice < 0.6:
            # Boolean combination
            return BinOp(
                op=random.choice(['and', 'or']),
                left=self.random_base_expr(depth + 1),
                right=self.random_base_expr(depth + 1),
                expr_type=ExprType.BOOL,
            )

        elif choice < 0.8:
            # Negation
            return UnaryOp(
                op='not',
                operand=self.random_base_expr(depth + 1),
                expr_type=ExprType.BOOL,
            )

        else:
            return Const(value=True, expr_type=ExprType.BOOL)

    def random_expr(self, depth: int = 0) -> TemporalExpr:
        """Generate random expression (temporal or mixed)."""
        choice = random.random()

        if choice < 0.5:
            # Pure temporal
            return self.random_temporal_expr(depth)

        elif choice < 0.8:
            # Temporal combined with base
            return BinOp(
                op=random.choice(['and', 'or']),
                left=self.random_temporal_expr(depth + 1),
                right=self.random_base_expr(depth + 1),
                expr_type=ExprType.BOOL,
            )

        else:
            # Pure base (for variety)
            return self.random_base_expr(depth)

    # =========================================================================
    # MUTATION
    # =========================================================================

    def mutate_duration(self, duration: float) -> float:
        """Mutate a duration value."""
        choice = random.random()

        if choice < 0.3:
            # Small adjustment (gradual refinement)
            factor = random.uniform(0.7, 1.4)
            return max(0.1, duration * factor)

        elif choice < 0.6:
            # Pick nearby candidate
            candidates = sorted(DURATION_CANDIDATES, key=lambda d: abs(d - duration))
            nearby = candidates[:5]
            return random.choice(nearby)

        elif choice < 0.8:
            # Random candidate
            return random.choice(DURATION_CANDIDATES)

        else:
            # Keep as is
            return duration

    def mutate_rate_limit(self, count: int, window: float) -> Tuple[int, float]:
        """Mutate rate limit parameters."""
        new_count = count
        new_window = window

        if random.random() < 0.5:
            # Mutate count
            new_count = max(1, count + random.randint(-2, 2))

        if random.random() < 0.5:
            # Mutate window
            new_window = self.mutate_duration(window)

        return new_count, new_window

    def mutate(self, genome: TemporalGenome, mutation_rate: float = 0.3) -> TemporalGenome:
        """Mutate a temporal genome."""

        def mutate_expr(expr: TemporalExpr, depth: int = 0) -> TemporalExpr:
            if random.random() < mutation_rate:
                # Replace entire subtree
                return self.random_expr(depth)

            if isinstance(expr, After):
                new_duration = self.mutate_duration(expr.duration)
                return After(duration=new_duration)

            elif isinstance(expr, Within):
                new_duration = self.mutate_duration(expr.duration)
                return Within(duration=new_duration)

            elif isinstance(expr, Cooldown):
                new_duration = self.mutate_duration(expr.duration)
                return Cooldown(duration=new_duration)

            elif isinstance(expr, Debounce):
                new_duration = self.mutate_duration(expr.duration)
                return Debounce(duration=new_duration)

            elif isinstance(expr, RateLimit):
                new_count, new_window = self.mutate_rate_limit(expr.count, expr.window)
                return RateLimit(count=new_count, window=new_window)

            elif isinstance(expr, Since):
                if self.state_names and random.random() < 0.3:
                    new_state = random.choice(self.state_names)
                    return Since(state_name=new_state)
                return expr

            elif isinstance(expr, Timeout):
                new_duration = self.mutate_duration(expr.duration)
                if self.state_names and random.random() < 0.3:
                    new_state = random.choice(self.state_names)
                    return Timeout(state_name=new_state, duration=new_duration)
                return Timeout(state_name=expr.state_name, duration=new_duration)

            elif isinstance(expr, BusinessHours):
                new_start = expr.start_hour
                new_end = expr.end_hour
                if random.random() < 0.5:
                    new_start = max(0, min(23, expr.start_hour + random.randint(-2, 2)))
                if random.random() < 0.5:
                    new_end = max(new_start + 1, min(24, expr.end_hour + random.randint(-2, 2)))
                return BusinessHours(start_hour=new_start, end_hour=new_end)

            elif isinstance(expr, BinOp):
                # Maybe mutate operator
                new_op = expr.op
                if random.random() < 0.2:
                    if expr.op in ['and', 'or']:
                        new_op = 'or' if expr.op == 'and' else 'and'
                    elif expr.op in ['<', '>']:
                        new_op = random.choice(['<', '>', '<=', '>='])
                    elif expr.op in ['<=', '>=']:
                        new_op = random.choice(['<', '>', '<=', '>='])

                return BinOp(
                    op=new_op,
                    left=mutate_expr(expr.left, depth + 1) if expr.left else None,
                    right=mutate_expr(expr.right, depth + 1) if expr.right else None,
                    expr_type=expr.expr_type,
                )

            elif isinstance(expr, UnaryOp):
                return UnaryOp(
                    op=expr.op,
                    operand=mutate_expr(expr.operand, depth + 1) if expr.operand else None,
                    expr_type=expr.expr_type,
                )

            elif isinstance(expr, Const):
                if isinstance(expr.value, (int, float)) and random.random() < 0.3:
                    if isinstance(expr.value, float):
                        new_val = self.mutate_duration(expr.value)
                    else:
                        new_val = max(0, expr.value + random.randint(-2, 2))
                    return Const(value=new_val, expr_type=expr.expr_type)
                return expr

            elif isinstance(expr, Var):
                if self.variables and random.random() < 0.3:
                    return Var(name=random.choice(self.variables), expr_type=expr.expr_type)
                return expr

            return expr

        new_expr = mutate_expr(genome.expr)
        return TemporalGenome(expr=new_expr)

    # =========================================================================
    # CROSSOVER
    # =========================================================================

    def crossover(self, parent1: TemporalGenome, parent2: TemporalGenome) -> TemporalGenome:
        """Crossover two temporal genomes."""

        def get_subtrees(expr: TemporalExpr, path: str = '') -> List[Tuple[TemporalExpr, str]]:
            """Get all subtrees with their paths."""
            if expr is None:
                return []
            result = [(expr, path)]
            if isinstance(expr, BinOp):
                if expr.left:
                    result.extend(get_subtrees(expr.left, path + 'L'))
                if expr.right:
                    result.extend(get_subtrees(expr.right, path + 'R'))
            elif isinstance(expr, UnaryOp):
                if expr.operand:
                    result.extend(get_subtrees(expr.operand, path + 'O'))
            return result

        def replace_at_path(expr: TemporalExpr, path: str, replacement: TemporalExpr) -> TemporalExpr:
            """Replace subtree at path."""
            if not path:
                return replacement
            if isinstance(expr, BinOp):
                if path[0] == 'L':
                    return BinOp(
                        op=expr.op,
                        left=replace_at_path(expr.left, path[1:], replacement),
                        right=expr.right,
                        expr_type=expr.expr_type,
                    )
                elif path[0] == 'R':
                    return BinOp(
                        op=expr.op,
                        left=expr.left,
                        right=replace_at_path(expr.right, path[1:], replacement),
                        expr_type=expr.expr_type,
                    )
            elif isinstance(expr, UnaryOp):
                if path[0] == 'O':
                    return UnaryOp(
                        op=expr.op,
                        operand=replace_at_path(expr.operand, path[1:], replacement),
                        expr_type=expr.expr_type,
                    )
            return expr

        # Get subtrees from both parents
        subtrees1 = get_subtrees(parent1.expr)
        subtrees2 = get_subtrees(parent2.expr)

        # Find compatible subtrees (same type)
        compatible = []
        for sub1, path1 in subtrees1:
            for sub2, path2 in subtrees2:
                if sub1.expr_type == sub2.expr_type:
                    compatible.append((path1, sub2))

        if compatible:
            path, replacement = random.choice(compatible)
            new_expr = replace_at_path(copy.deepcopy(parent1.expr), path, copy.deepcopy(replacement))
            return TemporalGenome(expr=new_expr)

        return parent1.copy()

    # =========================================================================
    # FITNESS EVALUATION
    # =========================================================================

    def evaluate_fitness(
        self,
        genome: TemporalGenome,
        positive_examples: List[Dict],
        negative_examples: List[Dict],
    ) -> float:
        """Evaluate genome on examples."""
        tp, fp, tn, fn = 0, 0, 0, 0

        for ctx in positive_examples:
            try:
                result = genome.expr.evaluate(ctx)
                if result:
                    tp += 1
                else:
                    fn += 1
            except Exception:
                fn += 1

        for ctx in negative_examples:
            try:
                result = genome.expr.evaluate(ctx)
                if result:
                    fp += 1
                else:
                    tn += 1
            except Exception:
                tn += 1

        genome.true_positives = tp
        genome.false_positives = fp
        genome.true_negatives = tn
        genome.false_negatives = fn
        genome.fitness = genome.f1

        return genome.fitness

    # =========================================================================
    # EVOLUTION
    # =========================================================================

    def evolve(
        self,
        positive_examples: List[Dict],
        negative_examples: List[Dict],
        population_size: int = 50,
        n_generations: int = 100,
        mutation_rate: float = 0.3,
        crossover_rate: float = 0.7,
        elite_size: int = 5,
        verbose: bool = True,
    ) -> TemporalGenome:
        """
        Evolve temporal guard from examples.

        Args:
            positive_examples: Contexts where guard should be True
            negative_examples: Contexts where guard should be False
            population_size: Number of genomes per generation
            n_generations: Number of generations
            mutation_rate: Probability of mutation
            crossover_rate: Probability of crossover
            elite_size: Number of best genomes to preserve
            verbose: Print progress

        Returns:
            Best evolved genome
        """
        # Initialize population
        population = [
            TemporalGenome(expr=self.random_expr())
            for _ in range(population_size)
        ]

        best_ever = None

        for gen in range(n_generations):
            # Evaluate fitness
            for genome in population:
                self.evaluate_fitness(genome, positive_examples, negative_examples)

            # Sort by fitness
            population.sort(key=lambda g: g.fitness, reverse=True)

            # Track best
            if best_ever is None or population[0].fitness > best_ever.fitness:
                best_ever = population[0].copy()

            # Progress
            if verbose and gen % 10 == 0:
                best = population[0]
                summary = best.get_temporal_summary()
                print(f"Gen {gen:3d}: F1={best.f1:.3f} | {summary[:50]}")

            # Perfect solution?
            if population[0].f1 >= 0.99:
                if verbose:
                    print(f"Perfect solution found at generation {gen}!")
                break

            # Selection and reproduction
            elite = [g.copy() for g in population[:elite_size]]
            new_pop = elite

            while len(new_pop) < population_size:
                # Tournament selection
                tournament = random.sample(population[:population_size // 2], 3)
                parent1 = max(tournament, key=lambda g: g.fitness)

                tournament = random.sample(population[:population_size // 2], 3)
                parent2 = max(tournament, key=lambda g: g.fitness)

                # Crossover
                if random.random() < crossover_rate:
                    child = self.crossover(parent1, parent2)
                else:
                    child = parent1.copy()

                # Mutation
                if random.random() < mutation_rate:
                    child = self.mutate(child, mutation_rate)

                new_pop.append(child)

            population = new_pop

        return best_ever


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def evolve_timeout(
    positive_examples: List[Dict],
    negative_examples: List[Dict],
    n_generations: int = 50,
    verbose: bool = True,
) -> TemporalGenome:
    """Evolve a timeout guard."""
    evolver = TemporalEvolver()
    return evolver.evolve(
        positive_examples,
        negative_examples,
        n_generations=n_generations,
        verbose=verbose,
    )


def evolve_rate_limit(
    positive_examples: List[Dict],
    negative_examples: List[Dict],
    n_generations: int = 50,
    verbose: bool = True,
) -> TemporalGenome:
    """Evolve a rate limiting guard."""
    evolver = TemporalEvolver()
    return evolver.evolve(
        positive_examples,
        negative_examples,
        n_generations=n_generations,
        verbose=verbose,
    )


# =============================================================================
# TESTING
# =============================================================================

def test_temporal_evolver():
    """Test temporal evolver."""
    print("=" * 60)
    print("TEMPORAL EVOLVER TEST")
    print("=" * 60)

    base_time = time.time()

    # Test: Learn 30-second timeout
    print("\nLearning 30-second timeout...")

    positive = []
    negative = []

    for _ in range(50):
        # Positive: should timeout (elapsed >= 30)
        elapsed = random.uniform(30, 90)
        positive.append({
            '__timestamp__': base_time,
            '__state_entry_time__': base_time - elapsed,
        })

        # Negative: should not timeout (elapsed < 30)
        elapsed = random.uniform(0, 29)
        negative.append({
            '__timestamp__': base_time,
            '__state_entry_time__': base_time - elapsed,
        })

    evolver = TemporalEvolver()
    best = evolver.evolve(
        positive,
        negative,
        population_size=30,
        n_generations=40,
        verbose=True,
    )

    print(f"\nLearned guard: {best.expr.to_string()}")
    print(f"Temporal summary: {best.get_temporal_summary()}")
    print(f"F1={best.f1:.3f}, Precision={best.precision:.3f}, Recall={best.recall:.3f}")

    print("\n" + "=" * 60)
    print("TEMPORAL EVOLVER TEST COMPLETE")
    print("=" * 60)

    return best


if __name__ == "__main__":
    test_temporal_evolver()
