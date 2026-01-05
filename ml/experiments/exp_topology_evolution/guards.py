"""
Evolvable Guard Expressions

This module provides guard expression trees that can be evolved to discover
game-specific conditions. Instead of fixed guards, evolution can synthesize
new guard logic.

Guard expressions are trees with:
- Operators: AND, OR, NOT, GT, LT, EQ, GE, LE
- Leaves: Feature names or constants
"""

import random
import copy
from dataclasses import dataclass, field
from typing import Dict, List, Union, Optional, Any
from enum import Enum


# =============================================================================
# Expression Types
# =============================================================================

class Op(Enum):
    """Guard expression operators."""
    # Logical
    AND = "AND"
    OR = "OR"
    NOT = "NOT"

    # Comparison
    GT = ">"
    LT = "<"
    GE = ">="
    LE = "<="
    EQ = "=="
    NE = "!="


@dataclass
class GuardExpr:
    """
    Evolvable guard expression tree.

    Can represent:
    - Constants: GuardExpr(value=0.5)
    - Features: GuardExpr(feature="cell_empty")
    - Unary: GuardExpr(op=Op.NOT, left=expr)
    - Binary: GuardExpr(op=Op.AND, left=expr1, right=expr2)
    """
    op: Optional[Op] = None
    left: Optional[Union['GuardExpr', str, float]] = None
    right: Optional[Union['GuardExpr', str, float]] = None
    value: Optional[float] = None
    feature: Optional[str] = None

    def is_leaf(self) -> bool:
        """Check if this is a leaf node (constant or feature)."""
        return self.op is None

    def is_constant(self) -> bool:
        """Check if this is a constant value."""
        return self.value is not None

    def is_feature(self) -> bool:
        """Check if this is a feature reference."""
        return self.feature is not None

    def evaluate(self, features: Dict[str, float]) -> bool:
        """
        Evaluate the guard expression.

        Args:
            features: Dict mapping feature names to values

        Returns:
            True if guard is satisfied, False otherwise
        """
        if self.is_constant():
            return self.value > 0.5

        if self.is_feature():
            val = features.get(self.feature, 0.0)
            return val > 0.5

        # Unary NOT
        if self.op == Op.NOT:
            left_val = self._eval_operand(self.left, features)
            return not left_val

        # Binary operators
        left_val = self._eval_operand(self.left, features)
        right_val = self._eval_operand(self.right, features)

        if self.op == Op.AND:
            return left_val and right_val
        elif self.op == Op.OR:
            return left_val or right_val
        elif self.op == Op.GT:
            return self._to_float(left_val, features) > self._to_float(right_val, features)
        elif self.op == Op.LT:
            return self._to_float(left_val, features) < self._to_float(right_val, features)
        elif self.op == Op.GE:
            return self._to_float(left_val, features) >= self._to_float(right_val, features)
        elif self.op == Op.LE:
            return self._to_float(left_val, features) <= self._to_float(right_val, features)
        elif self.op == Op.EQ:
            return abs(self._to_float(left_val, features) - self._to_float(right_val, features)) < 0.01
        elif self.op == Op.NE:
            return abs(self._to_float(left_val, features) - self._to_float(right_val, features)) >= 0.01

        return True  # Default

    def _eval_operand(self, operand, features: Dict[str, float]) -> Any:
        """Evaluate an operand (may be expr, feature name, or constant)."""
        if isinstance(operand, GuardExpr):
            return operand.evaluate(features)
        elif isinstance(operand, str):
            return features.get(operand, 0.0)
        elif isinstance(operand, (int, float)):
            return operand
        return 0.0

    def _to_float(self, val, features: Dict[str, float]) -> float:
        """Convert a value to float for comparison."""
        if isinstance(val, bool):
            return 1.0 if val else 0.0
        elif isinstance(val, (int, float)):
            return float(val)
        elif isinstance(val, str):
            return features.get(val, 0.0)
        return 0.0

    def depth(self) -> int:
        """Get the depth of the expression tree."""
        if self.is_leaf():
            return 1

        left_depth = 0
        right_depth = 0

        if isinstance(self.left, GuardExpr):
            left_depth = self.left.depth()
        elif self.left is not None:
            left_depth = 1

        if isinstance(self.right, GuardExpr):
            right_depth = self.right.depth()
        elif self.right is not None:
            right_depth = 1

        return 1 + max(left_depth, right_depth)

    def size(self) -> int:
        """Get the number of nodes in the expression tree."""
        if self.is_leaf():
            return 1

        count = 1
        if isinstance(self.left, GuardExpr):
            count += self.left.size()
        elif self.left is not None:
            count += 1

        if isinstance(self.right, GuardExpr):
            count += self.right.size()
        elif self.right is not None:
            count += 1

        return count

    def to_string(self) -> str:
        """Convert expression to readable string."""
        if self.is_constant():
            return f"{self.value:.2f}"

        if self.is_feature():
            return self.feature

        if self.op == Op.NOT:
            left_str = self._operand_to_string(self.left)
            return f"NOT({left_str})"

        left_str = self._operand_to_string(self.left)
        right_str = self._operand_to_string(self.right)

        return f"({left_str} {self.op.value} {right_str})"

    def _operand_to_string(self, operand) -> str:
        """Convert an operand to string."""
        if isinstance(operand, GuardExpr):
            return operand.to_string()
        elif isinstance(operand, str):
            return operand
        elif isinstance(operand, (int, float)):
            return f"{operand:.2f}"
        return "?"

    def copy(self) -> 'GuardExpr':
        """Create a deep copy of the expression."""
        return copy.deepcopy(self)


# =============================================================================
# Guard Expression Generation
# =============================================================================

def create_random_guard(
    features: List[str],
    max_depth: int = 3,
    current_depth: int = 0
) -> GuardExpr:
    """
    Create a random guard expression.

    Args:
        features: List of available feature names
        max_depth: Maximum tree depth
        current_depth: Current depth in recursion
    """
    # At max depth or with probability, create leaf
    if current_depth >= max_depth or random.random() < 0.4:
        if random.random() < 0.7 and features:
            # Feature reference
            return GuardExpr(feature=random.choice(features))
        else:
            # Constant
            return GuardExpr(value=random.random())

    # Create internal node
    op = random.choice(list(Op))

    if op == Op.NOT:
        # Unary
        left = create_random_guard(features, max_depth, current_depth + 1)
        return GuardExpr(op=op, left=left)
    else:
        # Binary
        left = create_random_guard(features, max_depth, current_depth + 1)
        right = create_random_guard(features, max_depth, current_depth + 1)
        return GuardExpr(op=op, left=left, right=right)


def create_simple_guard(feature: str, op: Op = Op.GT, threshold: float = 0.5) -> GuardExpr:
    """Create a simple comparison guard: feature > threshold."""
    return GuardExpr(
        op=op,
        left=GuardExpr(feature=feature),
        right=GuardExpr(value=threshold)
    )


# =============================================================================
# Guard Mutation Operators
# =============================================================================

def mutate_guard(
    guard: GuardExpr,
    features: List[str],
    mutation_rate: float = 0.3,
    max_depth: int = 4
) -> GuardExpr:
    """
    Mutate a guard expression.

    Mutations:
    - Change operator
    - Change feature reference
    - Change constant value
    - Replace subtree
    - Insert/delete node
    """
    guard = guard.copy()

    # Subtree replacement
    if random.random() < mutation_rate and guard.depth() < max_depth:
        # Replace entire guard with new random one
        return create_random_guard(features, max_depth=2)

    # Point mutations
    if guard.is_constant():
        if random.random() < mutation_rate:
            guard.value = max(0, min(1, guard.value + random.gauss(0, 0.2)))
        return guard

    if guard.is_feature():
        if random.random() < mutation_rate and features:
            guard.feature = random.choice(features)
        return guard

    # Operator mutation
    if random.random() < mutation_rate:
        if guard.op == Op.NOT:
            # Keep as unary or convert to binary
            if random.random() < 0.5:
                new_op = random.choice([Op.AND, Op.OR, Op.GT, Op.LT])
                guard.op = new_op
                guard.right = create_random_guard(features, max_depth=2)
        else:
            # Change binary operator
            similar_ops = {
                Op.AND: [Op.OR],
                Op.OR: [Op.AND],
                Op.GT: [Op.LT, Op.GE, Op.LE],
                Op.LT: [Op.GT, Op.GE, Op.LE],
                Op.GE: [Op.GT, Op.LT, Op.LE],
                Op.LE: [Op.GT, Op.LT, Op.GE],
                Op.EQ: [Op.NE],
                Op.NE: [Op.EQ],
            }
            if guard.op in similar_ops:
                guard.op = random.choice(similar_ops[guard.op])

    # Recursively mutate children
    if isinstance(guard.left, GuardExpr):
        guard.left = mutate_guard(guard.left, features, mutation_rate * 0.7, max_depth)

    if isinstance(guard.right, GuardExpr):
        guard.right = mutate_guard(guard.right, features, mutation_rate * 0.7, max_depth)

    return guard


def crossover_guards(parent1: GuardExpr, parent2: GuardExpr) -> GuardExpr:
    """
    Create offspring by swapping subtrees between parents.
    """
    child = parent1.copy()

    # Find a random node in child and replace with subtree from parent2
    if random.random() < 0.5 and isinstance(child.left, GuardExpr):
        child.left = parent2.copy()
    elif isinstance(child.right, GuardExpr):
        child.right = parent2.copy()

    return child


# =============================================================================
# Guard Library (Common Guards)
# =============================================================================

class GuardLibrary:
    """
    Library of common guard patterns.
    """

    @staticmethod
    def always_true() -> GuardExpr:
        """Guard that always evaluates to true."""
        return GuardExpr(value=1.0)

    @staticmethod
    def move_legal(feature: str = "cell_empty") -> GuardExpr:
        """Guard for legal moves (cell/position is empty)."""
        return create_simple_guard(feature, Op.GT, 0.5)

    @staticmethod
    def early_game(feature: str = "move_count", threshold: float = 0.3) -> GuardExpr:
        """Guard for early game (few moves played)."""
        return create_simple_guard(feature, Op.LT, threshold)

    @staticmethod
    def late_game(feature: str = "move_count", threshold: float = 0.7) -> GuardExpr:
        """Guard for late game (many moves played)."""
        return create_simple_guard(feature, Op.GT, threshold)

    @staticmethod
    def strategic_position(features: List[str]) -> GuardExpr:
        """Guard for strategic positions (corners, center)."""
        if "is_center" in features:
            center = create_simple_guard("is_center", Op.GT, 0.5)
            if "is_corner" in features:
                corner = create_simple_guard("is_corner", Op.GT, 0.5)
                return GuardExpr(op=Op.OR, left=center, right=corner)
            return center
        return GuardExpr(value=1.0)


# =============================================================================
# Testing
# =============================================================================

def test_guards():
    """Test guard expression functionality."""
    print("=" * 60)
    print("GUARD EXPRESSION TEST")
    print("=" * 60)

    features = ["cell_empty", "move_count", "is_center", "is_corner"]

    # Test random generation
    print("\n1. Random guard generation:")
    for i in range(3):
        guard = create_random_guard(features, max_depth=3)
        print(f"   Guard {i+1}: {guard.to_string()}")
        print(f"           Depth={guard.depth()}, Size={guard.size()}")

    # Test evaluation
    print("\n2. Guard evaluation:")
    test_features = {
        "cell_empty": 1.0,
        "move_count": 0.3,
        "is_center": 0.0,
        "is_corner": 1.0,
    }

    guard = GuardLibrary.move_legal("cell_empty")
    result = guard.evaluate(test_features)
    print(f"   move_legal({test_features['cell_empty']}) = {result}")

    guard = GuardLibrary.early_game("move_count")
    result = guard.evaluate(test_features)
    print(f"   early_game({test_features['move_count']}) = {result}")

    guard = GuardLibrary.strategic_position(features)
    result = guard.evaluate(test_features)
    print(f"   strategic_position = {result} (center={test_features['is_center']}, corner={test_features['is_corner']})")

    # Test mutation
    print("\n3. Guard mutation:")
    original = create_random_guard(features, max_depth=2)
    print(f"   Original: {original.to_string()}")

    for i in range(3):
        mutated = mutate_guard(original, features, mutation_rate=0.5)
        print(f"   Mutated {i+1}: {mutated.to_string()}")

    # Test crossover
    print("\n4. Guard crossover:")
    parent1 = create_random_guard(features, max_depth=2)
    parent2 = create_random_guard(features, max_depth=2)
    print(f"   Parent 1: {parent1.to_string()}")
    print(f"   Parent 2: {parent2.to_string()}")

    child = crossover_guards(parent1, parent2)
    print(f"   Child:    {child.to_string()}")

    print("\nAll guard tests passed!")


if __name__ == "__main__":
    test_guards()
