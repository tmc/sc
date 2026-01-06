"""
Guard Evolver

Evolves boolean guard expressions.
"""

import random
import copy
from dataclasses import dataclass, field
from typing import List, Dict, Any, Union
from enum import Enum

class LogicOp(Enum):
    AND = "&&"
    OR = "||"

class CompOp(Enum):
    LT = "<"
    GT = ">"
    LE = "<="
    GE = ">="
    EQ = "=="
    NE = "!="

@dataclass
class Expr:
    def eval(self, ctx: Dict[str, Any]) -> Any:
        raise NotImplementedError

@dataclass
class Const(Expr):
    val: Any
    def eval(self, ctx: Dict[str, Any]) -> Any: return self.val
    def __repr__(self): return str(self.val)

@dataclass
class Var(Expr):
    name: str
    def eval(self, ctx: Dict[str, Any]) -> Any: return ctx.get(self.name, 0)
    def __repr__(self): return self.name

@dataclass
class Comparison(Expr):
    left: Expr
    op: CompOp
    right: Expr
    def eval(self, ctx: Dict[str, Any]) -> bool:
        l = self.left.eval(ctx)
        r = self.right.eval(ctx)
        if self.op == CompOp.LT: return l < r
        if self.op == CompOp.GT: return l > r
        if self.op == CompOp.LE: return l <= r
        if self.op == CompOp.GE: return l >= r
        if self.op == CompOp.EQ: return l == r
        if self.op == CompOp.NE: return l != r
        return False
    def __repr__(self): return f"({self.left} {self.op.value} {self.right})"

@dataclass
class Logic(Expr):
    left: Expr
    op: LogicOp
    right: Expr
    def eval(self, ctx: Dict[str, Any]) -> bool:
        l = self.left.eval(ctx)
        r = self.right.eval(ctx)
        if self.op == LogicOp.AND: return l and r
        if self.op == LogicOp.OR: return l or r
        return False
    def __repr__(self): return f"({self.left} {self.op.value} {self.right})"

@dataclass
class GuardGenome:
    root: Expr
    fitness: float = 0.0

    def evaluate(self, context: Dict[str, Any]) -> bool:
        try:
            return bool(self.root.eval(context))
        except:
            return False

    def copy(self) -> 'GuardGenome':
        return copy.deepcopy(self)

    def size(self) -> int:
        def _size(node):
            if isinstance(node, (Comparison, Logic)):
                return 1 + _size(node.left) + _size(node.right)
            return 1
        return _size(self.root)

class GuardEvolver:
    def __init__(self, variables: List[str]):
        self.variables = variables
        self.basic_constants = [0, 1, 5, 10, 100]

    def create_random_expr(self, depth: int) -> Expr:
        if depth <= 0 or random.random() < 0.3:
            # Terminal: Comparison
            v = Var(random.choice(self.variables))
            c = Const(random.choice(self.basic_constants))
            # Randomly flip order
            l, r = (v, c) if random.random() < 0.7 else (c, v)
            return Comparison(l, random.choice(list(CompOp)), r)
        else:
            return Logic(
                self.create_random_expr(depth-1),
                random.choice(list(LogicOp)),
                self.create_random_expr(depth-1)
            )

    def create_genome(self) -> GuardGenome:
        return GuardGenome(root=self.create_random_expr(2))

    def mut_node(self, node: Expr, depth: int) -> Expr:
        # 1. Replace with new random subtree
        if random.random() < 0.1:
            return self.create_random_expr(depth)
            
        # 2. Mutate specific types
        if isinstance(node, Comparison):
            if random.random() < 0.3: # Change Op
                node.op = random.choice(list(CompOp))
            elif random.random() < 0.3: # Change Const
                 # Simple hack: modify const if it's there
                 if isinstance(node.right, Const):
                     node.right.val += random.choice([-1, 1])
            return node
            
        if isinstance(node, Logic):
             node.left = self.mut_node(node.left, depth-1)
             node.right = self.mut_node(node.right, depth-1)
             if random.random() < 0.2:
                 node.op = random.choice(list(LogicOp))
             return node
             
        return node

    def mutate(self, genome: GuardGenome) -> GuardGenome:
        mutant = genome.copy()
        mutant.root = self.mut_node(mutant.root, 2)
        return mutant

    def crossover(self, p1: GuardGenome, p2: GuardGenome) -> GuardGenome:
        # Root swap (simple) or subtree swap? Let's do simple logic composition
        if random.random() < 0.5:
             # Make a new AND/OR joining them
             return GuardGenome(Logic(p1.root, random.choice(list(LogicOp)), p2.root))
        return p1.copy()

    def evaluate_fitness(self, genome: GuardGenome, traces: List[Dict]) -> float:
        correct = 0
        for t in traces:
            pred = genome.evaluate(t['context'])
            if pred == t['fired']:
                correct += 1
        
        acc = correct / len(traces)
        parsimony = 1.0 / (1 + genome.size())
        genome.fitness = acc * 0.9 + parsimony * 0.1
        return genome.fitness
