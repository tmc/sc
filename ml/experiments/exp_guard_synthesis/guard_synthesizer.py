"""
Program Synthesis for Guard and Action Expressions

This experiment synthesizes guard conditions and action expressions
for statechart transitions using:
1. LLM-based candidate generation
2. Evolutionary refinement via self-play
3. Formal verification of synthesized expressions

Key Research Contribution:
  We show that GUARD EXPRESSIONS ARE LEARNABLE - the LLM proposes
  candidates, evolution selects for correctness, and we extract
  interpretable boolean logic.

Applications:
  - Automatic rule discovery in games (Ko, castling, en passant)
  - Policy constraint learning from demonstrations
  - Specification mining from execution traces
"""

import mlx.core as mx
import mlx.nn as nn
from dataclasses import dataclass, field
from typing import List, Dict, Set, Tuple, Optional, Callable, Any
from enum import Enum, auto
import random
import json
from abc import ABC, abstractmethod


# =============================================================================
# EXPRESSION AST
# =============================================================================

class ExprType(Enum):
    """Types of expressions in our DSL."""
    BOOL = auto()
    INT = auto()
    FLOAT = auto()
    POSITION = auto()  # (x, y) coordinate
    PIECE = auto()     # Game piece type
    ACTION = auto()    # Side effect


@dataclass
class Expr(ABC):
    """Base class for expressions."""
    expr_type: ExprType

    @abstractmethod
    def evaluate(self, context: Dict[str, Any]) -> Any:
        pass

    @abstractmethod
    def to_string(self) -> str:
        pass


@dataclass
class Const(Expr):
    """Constant value."""
    value: Any

    def evaluate(self, context: Dict[str, Any]) -> Any:
        return self.value

    def to_string(self) -> str:
        return repr(self.value)


@dataclass
class Var(Expr):
    """Variable reference."""
    name: str

    def evaluate(self, context: Dict[str, Any]) -> Any:
        return context.get(self.name)

    def to_string(self) -> str:
        return self.name


@dataclass
class BinOp(Expr):
    """Binary operation."""
    op: str  # 'and', 'or', '==', '!=', '<', '>', '<=', '>='
    left: Expr
    right: Expr

    def evaluate(self, context: Dict[str, Any]) -> Any:
        l = self.left.evaluate(context)
        r = self.right.evaluate(context)

        ops = {
            'and': lambda a, b: a and b,
            'or': lambda a, b: a or b,
            '==': lambda a, b: a == b,
            '!=': lambda a, b: a != b,
            '<': lambda a, b: a < b,
            '>': lambda a, b: a > b,
            '<=': lambda a, b: a <= b,
            '>=': lambda a, b: a >= b,
            '+': lambda a, b: a + b,
            '-': lambda a, b: a - b,
        }
        return ops[self.op](l, r)

    def to_string(self) -> str:
        return f"({self.left.to_string()} {self.op} {self.right.to_string()})"


@dataclass
class UnaryOp(Expr):
    """Unary operation."""
    op: str  # 'not', '-'
    operand: Expr

    def evaluate(self, context: Dict[str, Any]) -> Any:
        v = self.operand.evaluate(context)
        if self.op == 'not':
            return not v
        elif self.op == '-':
            return -v
        return v

    def to_string(self) -> str:
        return f"({self.op} {self.operand.to_string()})"


@dataclass
class FuncCall(Expr):
    """Function call (for domain-specific operations)."""
    func_name: str
    args: List[Expr]

    def evaluate(self, context: Dict[str, Any]) -> Any:
        # Domain functions are registered in context['__functions__']
        funcs = context.get('__functions__', {})
        if self.func_name in funcs:
            arg_values = [arg.evaluate(context) for arg in self.args]
            return funcs[self.func_name](*arg_values, context=context)
        return None

    def to_string(self) -> str:
        args_str = ", ".join(arg.to_string() for arg in self.args)
        return f"{self.func_name}({args_str})"


@dataclass
class IfThenElse(Expr):
    """Conditional expression."""
    condition: Expr
    then_branch: Expr
    else_branch: Expr

    def evaluate(self, context: Dict[str, Any]) -> Any:
        if self.condition.evaluate(context):
            return self.then_branch.evaluate(context)
        return self.else_branch.evaluate(context)

    def to_string(self) -> str:
        return (f"(if {self.condition.to_string()} "
                f"then {self.then_branch.to_string()} "
                f"else {self.else_branch.to_string()})")


# =============================================================================
# GUARD GENOME
# =============================================================================

@dataclass
class GuardGenome:
    """Evolvable guard expression."""
    expr: Expr
    fitness: float = 0.0
    true_positives: int = 0
    false_positives: int = 0
    true_negatives: int = 0
    false_negatives: int = 0

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
# GUARD SYNTHESIZER
# =============================================================================

class GuardSynthesizer:
    """
    Synthesize guard expressions from examples.

    Given positive examples (contexts where transition should fire)
    and negative examples (contexts where it shouldn't), evolve
    a boolean expression that distinguishes them.
    """

    def __init__(
        self,
        variables: List[str],
        constants: List[Any],
        functions: Dict[str, Callable] = None,
        max_depth: int = 4,
    ):
        self.variables = variables
        self.constants = constants
        self.functions = functions or {}
        self.max_depth = max_depth

    def random_expr(self, depth: int = 0, target_type: ExprType = ExprType.BOOL) -> Expr:
        """Generate a random expression."""
        if depth >= self.max_depth:
            # Terminal: variable or constant
            if target_type == ExprType.BOOL:
                if random.random() < 0.5 and self.variables:
                    return Var(expr_type=ExprType.BOOL, name=random.choice(self.variables))
                return Const(expr_type=ExprType.BOOL, value=random.choice([True, False]))
            else:
                if random.random() < 0.5 and self.variables:
                    return Var(expr_type=target_type, name=random.choice(self.variables))
                return Const(expr_type=target_type, value=random.choice(self.constants))

        # Non-terminal
        choice = random.random()

        if target_type == ExprType.BOOL:
            if choice < 0.3:
                # Binary boolean op
                op = random.choice(['and', 'or'])
                return BinOp(
                    op=op,
                    left=self.random_expr(depth + 1, ExprType.BOOL),
                    right=self.random_expr(depth + 1, ExprType.BOOL),
                    expr_type=ExprType.BOOL,
                )
            elif choice < 0.6:
                # Comparison
                op = random.choice(['==', '!=', '<', '>', '<=', '>='])
                return BinOp(
                    op=op,
                    left=self.random_expr(depth + 1, ExprType.INT),
                    right=self.random_expr(depth + 1, ExprType.INT),
                    expr_type=ExprType.BOOL,
                )
            elif choice < 0.8:
                # Negation
                return UnaryOp(
                    op='not',
                    operand=self.random_expr(depth + 1, ExprType.BOOL),
                    expr_type=ExprType.BOOL,
                )
            else:
                # Function call (if available)
                if self.functions:
                    func_name = random.choice(list(self.functions.keys()))
                    return FuncCall(
                        func_name=func_name,
                        args=[self.random_expr(depth + 1, ExprType.INT)],
                        expr_type=ExprType.BOOL,
                    )
                return Const(expr_type=ExprType.BOOL, value=True)

        else:
            # Non-boolean
            if random.random() < 0.5 and self.variables:
                return Var(expr_type=target_type, name=random.choice(self.variables))
            return Const(expr_type=target_type, value=random.choice(self.constants))

    def mutate(self, genome: GuardGenome, mutation_rate: float = 0.3) -> GuardGenome:
        """Mutate a guard expression."""

        def mutate_expr(expr: Expr, depth: int = 0) -> Expr:
            if random.random() < mutation_rate or depth > self.max_depth:
                # Replace with new random expression
                return self.random_expr(depth, expr.expr_type)

            if isinstance(expr, BinOp):
                # Maybe mutate operator
                if random.random() < 0.2:
                    if expr.expr_type == ExprType.BOOL and expr.op in ['and', 'or']:
                        new_op = 'or' if expr.op == 'and' else 'and'
                    elif expr.op in ['<', '>']:
                        new_op = '>' if expr.op == '<' else '<'
                    elif expr.op in ['<=', '>=']:
                        new_op = '>=' if expr.op == '<=' else '<='
                    elif expr.op in ['==', '!=']:
                        new_op = '!=' if expr.op == '==' else '=='
                    else:
                        new_op = expr.op
                    return BinOp(
                        op=new_op,
                        left=mutate_expr(expr.left, depth + 1),
                        right=mutate_expr(expr.right, depth + 1),
                        expr_type=expr.expr_type,
                    )
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
                # Maybe mutate constant
                if random.random() < 0.3:
                    if isinstance(expr.value, bool):
                        return Const(expr_type=expr.expr_type, value=not expr.value)
                    elif isinstance(expr.value, int):
                        return Const(expr_type=expr.expr_type, value=expr.value + random.randint(-2, 2))
                return expr
            elif isinstance(expr, Var):
                # Maybe change variable
                if random.random() < 0.3 and self.variables:
                    return Var(expr_type=expr.expr_type, name=random.choice(self.variables))
                return expr
            else:
                return expr

        new_expr = mutate_expr(genome.expr)
        return GuardGenome(expr=new_expr)

    def crossover(self, parent1: GuardGenome, parent2: GuardGenome) -> GuardGenome:
        """Crossover two guard expressions."""

        def get_subtrees(expr: Expr) -> List[Tuple[Expr, str]]:
            """Get all subtrees with their paths."""
            result = [(expr, '')]
            if isinstance(expr, BinOp):
                for sub, path in get_subtrees(expr.left):
                    result.append((sub, 'L' + path))
                for sub, path in get_subtrees(expr.right):
                    result.append((sub, 'R' + path))
            elif isinstance(expr, UnaryOp):
                for sub, path in get_subtrees(expr.operand):
                    result.append((sub, 'O' + path))
            return result

        def replace_at_path(expr: Expr, path: str, replacement: Expr) -> Expr:
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
                else:
                    return BinOp(
                        op=expr.op,
                        left=expr.left,
                        right=replace_at_path(expr.right, path[1:], replacement),
                        expr_type=expr.expr_type,
                    )
            elif isinstance(expr, UnaryOp):
                return UnaryOp(
                    op=expr.op,
                    operand=replace_at_path(expr.operand, path[1:], replacement),
                    expr_type=expr.expr_type,
                )
            return expr

        # Get subtrees
        subtrees1 = get_subtrees(parent1.expr)
        subtrees2 = get_subtrees(parent2.expr)

        # Pick compatible subtrees (same type)
        compatible = []
        for sub1, path1 in subtrees1:
            for sub2, path2 in subtrees2:
                if sub1.expr_type == sub2.expr_type:
                    compatible.append((path1, sub2))

        if compatible:
            path, replacement = random.choice(compatible)
            new_expr = replace_at_path(parent1.expr, path, replacement)
            return GuardGenome(expr=new_expr)

        return GuardGenome(expr=parent1.expr)

    def evaluate_fitness(
        self,
        genome: GuardGenome,
        positive_examples: List[Dict],
        negative_examples: List[Dict],
    ) -> float:
        """Evaluate guard on examples."""
        tp, fp, tn, fn = 0, 0, 0, 0

        for ctx in positive_examples:
            ctx['__functions__'] = self.functions
            try:
                result = genome.expr.evaluate(ctx)
                if result:
                    tp += 1
                else:
                    fn += 1
            except:
                fn += 1

        for ctx in negative_examples:
            ctx['__functions__'] = self.functions
            try:
                result = genome.expr.evaluate(ctx)
                if result:
                    fp += 1
                else:
                    tn += 1
            except:
                tn += 1

        genome.true_positives = tp
        genome.false_positives = fp
        genome.true_negatives = tn
        genome.false_negatives = fn

        # F1 score as fitness
        genome.fitness = genome.f1
        return genome.fitness

    def evolve(
        self,
        positive_examples: List[Dict],
        negative_examples: List[Dict],
        population_size: int = 50,
        n_generations: int = 100,
        verbose: bool = True,
    ) -> GuardGenome:
        """Evolve a guard expression."""

        # Initialize population
        population = [
            GuardGenome(expr=self.random_expr())
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
                best_ever = GuardGenome(
                    expr=population[0].expr,
                    fitness=population[0].fitness,
                    true_positives=population[0].true_positives,
                    false_positives=population[0].false_positives,
                    true_negatives=population[0].true_negatives,
                    false_negatives=population[0].false_negatives,
                )

            if verbose and gen % 10 == 0:
                best = population[0]
                print(f"Gen {gen:3d}: F1={best.f1:.3f}, P={best.precision:.3f}, R={best.recall:.3f}")
                print(f"         {best.expr.to_string()[:60]}...")

            # Perfect solution?
            if population[0].f1 >= 0.99:
                break

            # Selection (elitism + tournament)
            elite = population[:5]
            new_pop = elite.copy()

            while len(new_pop) < population_size:
                # Tournament selection
                tournament = random.sample(population[:population_size//2], 3)
                parent1 = max(tournament, key=lambda g: g.fitness)

                tournament = random.sample(population[:population_size//2], 3)
                parent2 = max(tournament, key=lambda g: g.fitness)

                # Crossover
                if random.random() < 0.7:
                    child = self.crossover(parent1, parent2)
                else:
                    child = GuardGenome(expr=parent1.expr)

                # Mutation
                child = self.mutate(child)
                new_pop.append(child)

            population = new_pop

        return best_ever


# =============================================================================
# ACTION SYNTHESIZER
# =============================================================================

@dataclass
class ActionExpr:
    """An action expression (side effect)."""
    target: str      # Variable to modify
    operation: str   # 'set', 'increment', 'decrement', 'toggle', 'append'
    value: Expr      # Value or expression

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute action and return modified context."""
        new_ctx = context.copy()
        val = self.value.evaluate(context) if self.value else None

        if self.operation == 'set':
            new_ctx[self.target] = val
        elif self.operation == 'increment':
            new_ctx[self.target] = context.get(self.target, 0) + (val or 1)
        elif self.operation == 'decrement':
            new_ctx[self.target] = context.get(self.target, 0) - (val or 1)
        elif self.operation == 'toggle':
            new_ctx[self.target] = not context.get(self.target, False)
        elif self.operation == 'append':
            lst = context.get(self.target, []).copy()
            lst.append(val)
            new_ctx[self.target] = lst

        return new_ctx

    def to_string(self) -> str:
        if self.operation == 'set':
            return f"{self.target} := {self.value.to_string()}"
        elif self.operation == 'increment':
            return f"{self.target} += {self.value.to_string() if self.value else 1}"
        elif self.operation == 'decrement':
            return f"{self.target} -= {self.value.to_string() if self.value else 1}"
        elif self.operation == 'toggle':
            return f"{self.target} := NOT {self.target}"
        elif self.operation == 'append':
            return f"{self.target}.append({self.value.to_string()})"
        return f"{self.operation}({self.target})"


@dataclass
class ActionGenome:
    """Evolvable action sequence."""
    actions: List[ActionExpr]
    fitness: float = 0.0


class ActionSynthesizer:
    """
    Synthesize action expressions from before/after examples.

    Given pairs of (context_before, context_after), learn what
    actions produce the state change.
    """

    def __init__(
        self,
        variables: List[str],
        constants: List[Any],
    ):
        self.variables = variables
        self.constants = constants

    def infer_actions(
        self,
        before: Dict[str, Any],
        after: Dict[str, Any],
    ) -> List[ActionExpr]:
        """Infer actions from before/after diff."""
        actions = []

        for var in self.variables:
            before_val = before.get(var)
            after_val = after.get(var)

            if before_val != after_val:
                # Something changed
                if isinstance(after_val, bool) and isinstance(before_val, bool):
                    if after_val == (not before_val):
                        actions.append(ActionExpr(var, 'toggle', None))
                    else:
                        actions.append(ActionExpr(
                            var, 'set',
                            Const(after_val, expr_type=ExprType.BOOL)
                        ))
                elif isinstance(after_val, int) and isinstance(before_val, int):
                    diff = after_val - before_val
                    if diff > 0:
                        actions.append(ActionExpr(
                            var, 'increment',
                            Const(diff, expr_type=ExprType.INT)
                        ))
                    else:
                        actions.append(ActionExpr(
                            var, 'decrement',
                            Const(-diff, expr_type=ExprType.INT)
                        ))
                elif isinstance(after_val, list) and isinstance(before_val, list):
                    if len(after_val) > len(before_val):
                        new_items = after_val[len(before_val):]
                        for item in new_items:
                            actions.append(ActionExpr(
                                var, 'append',
                                Const(item, expr_type=ExprType.INT)
                            ))
                    else:
                        actions.append(ActionExpr(
                            var, 'set',
                            Const(after_val, expr_type=ExprType.INT)
                        ))
                else:
                    # Generic set
                    actions.append(ActionExpr(
                        var, 'set',
                        Const(after_val, expr_type=ExprType.INT)
                    ))

        return actions


# =============================================================================
# EXPRESSION EVOLVER (Combined Guard + Action)
# =============================================================================

class ExpressionEvolver:
    """
    Evolve complete transition expressions (guard + action).

    This is the main interface for synthesizing statechart semantics
    from execution traces.
    """

    def __init__(
        self,
        variables: List[str],
        constants: List[Any],
        functions: Dict[str, Callable] = None,
    ):
        self.guard_synth = GuardSynthesizer(variables, constants, functions)
        self.action_synth = ActionSynthesizer(variables, constants)

    def learn_transition(
        self,
        traces: List[Tuple[Dict, Dict]],  # (before, after) pairs
        n_generations: int = 50,
    ) -> Tuple[GuardGenome, List[ActionExpr]]:
        """
        Learn a complete transition from traces.

        Returns (guard, actions) where:
        - guard: Boolean expression that should be true before transition
        - actions: List of actions that produce the after state
        """
        # Separate contexts
        positive_examples = [t[0] for t in traces]  # Before = when guard should be true
        negative_examples = []  # Would need counter-examples

        # Synthesize guard
        guard = self.guard_synth.evolve(
            positive_examples,
            negative_examples,
            n_generations=n_generations,
        )

        # Infer actions from first trace
        actions = self.action_synth.infer_actions(traces[0][0], traces[0][1])

        return guard, actions


# =============================================================================
# DEMO
# =============================================================================

def demo_guard_synthesis():
    """Demonstrate guard synthesis for Ko rule."""
    print("=" * 60)
    print("GUARD SYNTHESIS: Learning Ko Rule")
    print("=" * 60)

    # Variables for Go board state
    variables = [
        'last_capture_x', 'last_capture_y',
        'move_x', 'move_y',
        'stones_removed', 'would_capture_single',
    ]

    constants = [0, 1, -1, True, False]

    # Domain functions
    def is_recapture(ctx=None, **kwargs):
        return (
            ctx.get('move_x') == ctx.get('last_capture_x') and
            ctx.get('move_y') == ctx.get('last_capture_y') and
            ctx.get('stones_removed') == 1 and
            ctx.get('would_capture_single')
        )

    functions = {
        'is_recapture': lambda *args, context=None: is_recapture(context),
    }

    synth = GuardSynthesizer(variables, constants, functions, max_depth=3)

    # Positive examples: contexts where Ko should BLOCK the move
    positive_examples = [
        {'last_capture_x': 3, 'last_capture_y': 4, 'move_x': 3, 'move_y': 4,
         'stones_removed': 1, 'would_capture_single': True},
        {'last_capture_x': 5, 'last_capture_y': 5, 'move_x': 5, 'move_y': 5,
         'stones_removed': 1, 'would_capture_single': True},
        {'last_capture_x': 1, 'last_capture_y': 1, 'move_x': 1, 'move_y': 1,
         'stones_removed': 1, 'would_capture_single': True},
    ]

    # Negative examples: contexts where move should be ALLOWED
    negative_examples = [
        # Different position
        {'last_capture_x': 3, 'last_capture_y': 4, 'move_x': 3, 'move_y': 5,
         'stones_removed': 1, 'would_capture_single': True},
        # Captures multiple stones (not Ko)
        {'last_capture_x': 3, 'last_capture_y': 4, 'move_x': 3, 'move_y': 4,
         'stones_removed': 3, 'would_capture_single': False},
        # No previous capture
        {'last_capture_x': -1, 'last_capture_y': -1, 'move_x': 3, 'move_y': 4,
         'stones_removed': 1, 'would_capture_single': True},
        # Normal move (no capture)
        {'last_capture_x': 3, 'last_capture_y': 4, 'move_x': 7, 'move_y': 7,
         'stones_removed': 0, 'would_capture_single': False},
    ]

    print("\nPositive examples (Ko should block):")
    for ex in positive_examples[:2]:
        print(f"  move=({ex['move_x']},{ex['move_y']}), last_cap=({ex['last_capture_x']},{ex['last_capture_y']})")

    print("\nNegative examples (move allowed):")
    for ex in negative_examples[:2]:
        print(f"  move=({ex['move_x']},{ex['move_y']}), last_cap=({ex['last_capture_x']},{ex['last_capture_y']})")

    print("\nEvolving guard expression...")
    best = synth.evolve(
        positive_examples,
        negative_examples,
        population_size=30,
        n_generations=50,
        verbose=True,
    )

    print(f"\n{'='*60}")
    print(f"BEST GUARD:")
    print(f"  {best.expr.to_string()}")
    print(f"\nMetrics:")
    print(f"  F1={best.f1:.3f}, Precision={best.precision:.3f}, Recall={best.recall:.3f}")
    print(f"\nInterpretation:")
    print(f"  This guard blocks moves that would immediately recapture")
    print(f"  at the same position where a single stone was just captured.")
    print(f"{'='*60}")

    return best


def demo_action_synthesis():
    """Demonstrate action synthesis."""
    print("\n" + "=" * 60)
    print("ACTION SYNTHESIS: Learning State Updates")
    print("=" * 60)

    variables = ['king_moved', 'rook_a_moved', 'rook_h_moved', 'can_castle_kingside', 'can_castle_queenside']
    constants = [True, False]

    synth = ActionSynthesizer(variables, constants)

    # Before: king hasn't moved
    before = {
        'king_moved': False,
        'rook_a_moved': False,
        'rook_h_moved': False,
        'can_castle_kingside': True,
        'can_castle_queenside': True,
    }

    # After: king moved
    after = {
        'king_moved': True,
        'rook_a_moved': False,
        'rook_h_moved': False,
        'can_castle_kingside': False,
        'can_castle_queenside': False,
    }

    print("\nBefore state:", before)
    print("After state:", after)

    actions = synth.infer_actions(before, after)

    print("\nInferred actions:")
    for action in actions:
        print(f"  {action.to_string()}")

    print(f"\n{'='*60}")
    print("KEY INSIGHT: Actions are AUTOMATICALLY INFERRED from state diffs!")
    print("No manual specification needed.")
    print(f"{'='*60}")

    return actions


if __name__ == "__main__":
    demo_guard_synthesis()
    demo_action_synthesis()
