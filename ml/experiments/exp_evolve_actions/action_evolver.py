"""
Action Evolver

Evolves sequences of assignment statements to transform context.
"""

import random
import copy
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Union
from enum import Enum, auto

# =============================================================================
# AST NODES
# =============================================================================

class Op(Enum):
    ADD = "+"
    SUB = "-"
    MUL = "*"
    DIV = "//"  # Integer division
    MOD = "%"

@dataclass
class Expr:
    """Base class for expressions."""
    def eval(self, context: Dict[str, Any]) -> Any:
        raise NotImplementedError

@dataclass
class Const(Expr):
    value: int
    def eval(self, context: Dict[str, Any]) -> Any:
        return self.value
    def __repr__(self):
        return str(self.value)

@dataclass
class Var(Expr):
    name: str
    def eval(self, context: Dict[str, Any]) -> Any:
        return context.get(self.name, 0)
    def __repr__(self):
        return self.name

@dataclass
class BinOp(Expr):
    left: Expr
    op: Op
    right: Expr
    def eval(self, context: Dict[str, Any]) -> Any:
        l = self.left.eval(context)
        r = self.right.eval(context)
        if self.op == Op.ADD: return l + r
        if self.op == Op.SUB: return l - r
        if self.op == Op.MUL: return l * r
        if self.op == Op.DIV: return l // r if r != 0 else 0
        if self.op == Op.MOD: return l % r if r != 0 else 0
        return 0
    def __repr__(self):
        return f"({self.left} {self.op.value} {self.right})"

@dataclass
class Assignment:
    target: str
    expr: Expr
    def exec(self, context: Dict[str, Any]):
        try:
            val = self.expr.eval(context)
            context[self.target] = val
        except Exception:
            pass # Ignore errors during eval
    def __repr__(self):
        return f"{self.target} = {self.expr}"

# =============================================================================
# ACTION GENOME
# =============================================================================

@dataclass
class ActionGenome:
    statements: List[Assignment] = field(default_factory=list)
    fitness: float = 0.0

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the action sequence on a context copy."""
        ctx = context.copy()
        for stmt in self.statements:
            stmt.exec(ctx)
        return ctx

    def __repr__(self):
        return "; ".join([str(s) for s in self.statements])

    def copy(self) -> 'ActionGenome':
        return copy.deepcopy(self)

# =============================================================================
# EVOLVER
# =============================================================================

class ActionEvolver:
    def __init__(self, variables: List[str], constants: List[int] = None):
        self.variables = variables
        self.constants = constants or [0, 1, 2, 5, 10, -1]
        self.ops = list(Op)

    def create_random_expr(self, depth: int = 2) -> Expr:
        if depth == 0 or random.random() < 0.3:
            if random.random() < 0.5:
                return Const(random.choice(self.constants))
            else:
                return Var(random.choice(self.variables))
        else:
            return BinOp(
                left=self.create_random_expr(depth - 1),
                op=random.choice(self.ops),
                right=self.create_random_expr(depth - 1)
            )

    def create_random_assignment(self) -> Assignment:
        return Assignment(
            target=random.choice(self.variables),
            expr=self.create_random_expr()
        )

    def create_genome(self, max_stmts: int = 3) -> ActionGenome:
        n = random.randint(1, max_stmts)
        stmts = [self.create_random_assignment() for _ in range(n)]
        return ActionGenome(statements=stmts)

    def mutate(self, genome: ActionGenome) -> ActionGenome:
        mutant = genome.copy()
        if not mutant.statements:
            mutant.statements.append(self.create_random_assignment())
            return mutant
            
        choice = random.choice(['add', 'remove', 'swap', 'modify'])
        
        if choice == 'add':
            mutant.statements.insert(
                random.randint(0, len(mutant.statements)),
                self.create_random_assignment()
            )
        elif choice == 'remove' and len(mutant.statements) > 1:
            mutant.statements.pop(random.randint(0, len(mutant.statements)-1))
        elif choice == 'swap' and len(mutant.statements) > 1:
            i, j = random.sample(range(len(mutant.statements)), 2)
            mutant.statements[i], mutant.statements[j] = mutant.statements[j], mutant.statements[i]
        elif choice == 'modify':
            # Simplified: just replace a statement
            idx = random.randint(0, len(mutant.statements)-1)
            mutant.statements[idx] = self.create_random_assignment()
            
        return mutant

    def crossover(self, p1: ActionGenome, p2: ActionGenome) -> ActionGenome:
        # Single point crossover
        if not p1.statements or not p2.statements:
            return p1.copy()
            
        pt1 = random.randint(0, len(p1.statements))
        pt2 = random.randint(0, len(p2.statements))
        
        child_stmts = p1.statements[:pt1] + p2.statements[pt2:]
        if not child_stmts:
            child_stmts = [self.create_random_assignment()]
            
        return ActionGenome(statements=child_stmts)

    def evaluate(self, genome: ActionGenome, cases: List[Dict[str, Any]]) -> float:
        """
        cases: List of {'before': dict, 'after': dict}
        """
        correct = 0
        for case in cases:
            result = genome.execute(case['before'])
            # Check equality for all keys in 'after'
            matches = True
            for k, v in case['after'].items():
                if result.get(k) != v:
                    matches = False
                    break
            if matches:
                correct += 1
        
        accuracy = correct / len(cases)
        parsimony = 1.0 / (1 + len(genome.statements) + sum(self._count_nodes(s.expr) for s in genome.statements))
        
        genome.fitness = accuracy * 0.9 + parsimony * 0.1
        return genome.fitness

    def _count_nodes(self, expr: Expr) -> int:
        if isinstance(expr, BinOp):
            return 1 + self._count_nodes(expr.left) + self._count_nodes(expr.right)
        return 1
