"""
Statechart-Based Coverage Predictor

This is the key insight: Coverage prediction IS statechart simulation.

Given a program as a statechart (CFG) and an abstract input representation,
we simulate which states will be visited. The visited states map directly
to covered lines.

Approaches:
1. Symbolic Execution: Evaluate guards symbolically, track possible states
2. Learned Guards: Use neural networks to predict guard outcomes
3. Abstract Interpretation: Propagate abstract values through statechart

This module implements these approaches and compares them to baselines.
"""

import mlx.core as mx
import mlx.nn as nn
from dataclasses import dataclass, field
from typing import List, Dict, Set, Tuple, Optional, Any, Callable
import ast
import json

from .program_statechart import ProgramStatechart, BasicBlock, CFGTransition, BlockType, build_cfg


@dataclass
class AbstractValue:
    """
    Abstract value for abstract interpretation.

    Represents a set of possible concrete values.
    """
    is_top: bool = False          # Unknown value (could be anything)
    is_bottom: bool = False       # No possible value (unreachable)
    possible_types: Set[str] = field(default_factory=set)  # int, str, bool, list, etc.
    int_range: Optional[Tuple[int, int]] = None  # [min, max] for integers
    bool_value: Optional[bool] = None  # Definite boolean value
    is_truthy: Optional[bool] = None   # Definitely truthy/falsy
    list_length: Optional[Tuple[int, int]] = None  # Length range for lists

    @staticmethod
    def top() -> 'AbstractValue':
        """Unknown value."""
        return AbstractValue(is_top=True)

    @staticmethod
    def bottom() -> 'AbstractValue':
        """Unreachable."""
        return AbstractValue(is_bottom=True)

    @staticmethod
    def from_python(value: Any) -> 'AbstractValue':
        """Create abstract value from concrete Python value."""
        if isinstance(value, bool):
            return AbstractValue(
                possible_types={'bool'},
                bool_value=value,
                is_truthy=value,
            )
        elif isinstance(value, int):
            return AbstractValue(
                possible_types={'int'},
                int_range=(value, value),
                is_truthy=value != 0,
            )
        elif isinstance(value, str):
            return AbstractValue(
                possible_types={'str'},
                is_truthy=len(value) > 0,
            )
        elif isinstance(value, (list, tuple)):
            n = len(value)
            return AbstractValue(
                possible_types={'list'},
                list_length=(n, n),
                is_truthy=n > 0,
            )
        else:
            return AbstractValue.top()

    def could_be_truthy(self) -> bool:
        """Could this value be truthy?"""
        if self.is_bottom:
            return False
        if self.is_top:
            return True
        if self.bool_value is not None:
            return self.bool_value
        if self.is_truthy is not None:
            return self.is_truthy
        if self.int_range is not None:
            low, high = self.int_range
            return not (low == 0 and high == 0)
        return True

    def could_be_falsy(self) -> bool:
        """Could this value be falsy?"""
        if self.is_bottom:
            return False
        if self.is_top:
            return True
        if self.bool_value is not None:
            return not self.bool_value
        if self.is_truthy is not None:
            return not self.is_truthy
        if self.int_range is not None:
            low, high = self.int_range
            return low <= 0 <= high
        return True

    def join(self, other: 'AbstractValue') -> 'AbstractValue':
        """Join two abstract values (union)."""
        if self.is_bottom:
            return other
        if other.is_bottom:
            return self
        if self.is_top or other.is_top:
            return AbstractValue.top()

        # Combine information
        types = self.possible_types | other.possible_types

        # Join int ranges
        int_range = None
        if self.int_range and other.int_range:
            int_range = (
                min(self.int_range[0], other.int_range[0]),
                max(self.int_range[1], other.int_range[1]),
            )
        elif self.int_range:
            int_range = self.int_range
        elif other.int_range:
            int_range = other.int_range

        # Join booleans
        bool_value = None
        if self.bool_value == other.bool_value:
            bool_value = self.bool_value

        is_truthy = None
        if self.is_truthy == other.is_truthy:
            is_truthy = self.is_truthy

        return AbstractValue(
            possible_types=types,
            int_range=int_range,
            bool_value=bool_value,
            is_truthy=is_truthy,
        )


@dataclass
class StatechartConfig:
    """Configuration for statechart predictor."""
    max_iterations: int = 100     # Max simulation steps
    use_learned_guards: bool = True  # Use neural guard prediction
    embed_dim: int = 64           # Embedding dimension for learned guards


class SymbolicExecutor:
    """
    Symbolic execution on program statechart.

    Tracks possible states and evaluates guards symbolically
    to determine which paths are taken.
    """

    def __init__(self, statechart: ProgramStatechart):
        self.statechart = statechart
        self.current_states: Set[int] = {statechart.entry_block}
        self.visited_states: Set[int] = set()
        self.env: Dict[str, AbstractValue] = {}

    def set_input(self, var_name: str, value: AbstractValue):
        """Set an input variable's abstract value."""
        self.env[var_name] = value

    def evaluate_guard(self, guard_expr: str) -> Tuple[bool, bool]:
        """
        Evaluate a guard expression symbolically.

        Returns (can_be_true, can_be_false).
        """
        if not guard_expr:
            return (True, False)  # Always true if no guard

        try:
            tree = ast.parse(guard_expr, mode='eval')
            return self._eval_expr(tree.body)
        except:
            # If we can't parse, assume both branches possible
            return (True, True)

    def _eval_expr(self, node: ast.expr) -> Tuple[bool, bool]:
        """Evaluate AST expression symbolically."""
        if isinstance(node, ast.Compare):
            return self._eval_compare(node)
        elif isinstance(node, ast.BoolOp):
            return self._eval_boolop(node)
        elif isinstance(node, ast.UnaryOp):
            if isinstance(node.op, ast.Not):
                can_true, can_false = self._eval_expr(node.operand)
                return (can_false, can_true)  # Flip
        elif isinstance(node, ast.Name):
            if node.id in self.env:
                val = self.env[node.id]
                return (val.could_be_truthy(), val.could_be_falsy())
        elif isinstance(node, ast.Constant):
            if node.value:
                return (True, False)
            else:
                return (False, True)

        # Unknown - assume both possible
        return (True, True)

    def _eval_compare(self, node: ast.Compare) -> Tuple[bool, bool]:
        """Evaluate comparison expression."""
        # Simplified: just check if we know the variable values
        # In full implementation, would track ranges

        # Check for common patterns like x > 0, x == value
        if isinstance(node.left, ast.Name) and len(node.comparators) == 1:
            var_name = node.left.id
            if var_name in self.env:
                val = self.env[var_name]
                if val.int_range and isinstance(node.comparators[0], ast.Constant):
                    const = node.comparators[0].value
                    if isinstance(const, int) and isinstance(node.ops[0], ast.Gt):
                        low, high = val.int_range
                        can_true = high > const
                        can_false = low <= const
                        return (can_true, can_false)
                    elif isinstance(const, int) and isinstance(node.ops[0], ast.Lt):
                        low, high = val.int_range
                        can_true = low < const
                        can_false = high >= const
                        return (can_true, can_false)
                    elif isinstance(const, int) and isinstance(node.ops[0], ast.Eq):
                        low, high = val.int_range
                        can_true = low <= const <= high
                        can_false = not (low == high == const)
                        return (can_true, can_false)

        return (True, True)

    def _eval_boolop(self, node: ast.BoolOp) -> Tuple[bool, bool]:
        """Evaluate boolean operation."""
        results = [self._eval_expr(v) for v in node.values]

        if isinstance(node.op, ast.And):
            # All must be true for result to be true
            can_true = all(r[0] for r in results)
            # Any can be false for result to be false
            can_false = any(r[1] for r in results)
            return (can_true, can_false)
        elif isinstance(node.op, ast.Or):
            # Any can be true for result to be true
            can_true = any(r[0] for r in results)
            # All must be false for result to be false
            can_false = all(r[1] for r in results)
            return (can_true, can_false)

        return (True, True)

    def step(self) -> bool:
        """
        Take one simulation step.

        Returns True if any progress was made.
        """
        new_states = set()

        for state_id in self.current_states:
            self.visited_states.add(state_id)

            # Find outgoing transitions
            for trans in self.statechart.transitions:
                if trans.source_id != state_id:
                    continue

                # Evaluate guard
                if trans.guard:
                    can_true, can_false = self.evaluate_guard(trans.guard)
                    if trans.label == "True" and not can_true:
                        continue
                    if trans.label == "False" and not can_false:
                        continue

                new_states.add(trans.target_id)

        # Check for progress
        old_visited = len(self.visited_states)
        self.current_states = new_states
        self.visited_states |= new_states

        return len(self.visited_states) > old_visited

    def run(self, max_steps: int = 100) -> Set[int]:
        """Run simulation until fixed point or max steps."""
        for _ in range(max_steps):
            if not self.step():
                break
            if not self.current_states:
                break

        return self.visited_states

    def get_covered_lines(self) -> Set[int]:
        """Get source lines covered by visited states."""
        return self.statechart.get_covered_lines(self.visited_states)


class LearnedGuardPredictor(nn.Module):
    """
    Neural network to predict guard outcomes.

    Given:
    - Abstract input representation
    - Guard expression embedding

    Predicts:
    - Probability that guard is true
    """

    def __init__(self, config: StatechartConfig, vocab_size: int = 1000):
        super().__init__()
        self.config = config

        # Input encoder
        self.input_encoder = nn.Sequential(
            nn.Linear(config.embed_dim, config.embed_dim * 2),
            nn.ReLU(),
            nn.Linear(config.embed_dim * 2, config.embed_dim),
        )

        # Guard expression encoder
        self.guard_encoder = nn.Sequential(
            nn.Embedding(vocab_size, config.embed_dim),
        )

        # Predictor
        self.predictor = nn.Sequential(
            nn.Linear(config.embed_dim * 2, config.embed_dim),
            nn.ReLU(),
            nn.Linear(config.embed_dim, 1),
        )

    def __call__(
        self,
        input_embedding: mx.array,  # (batch, embed_dim) abstract input
        guard_tokens: mx.array,     # (batch, max_guard_len) guard expression
    ) -> mx.array:
        """Predict guard outcome probability."""
        # Encode input
        input_enc = self.input_encoder(input_embedding)  # (B, D)

        # Encode guard
        guard_emb = self.guard_encoder[0](guard_tokens)  # (B, L, D)
        guard_enc = mx.mean(guard_emb, axis=1)  # (B, D)

        # Combine and predict
        combined = mx.concatenate([input_enc, guard_enc], axis=-1)  # (B, 2D)
        logit = self.predictor(combined)  # (B, 1)

        return mx.sigmoid(logit.squeeze(-1))


class StatechartCoveragePredictor:
    """
    Main statechart-based coverage predictor.

    Combines:
    1. Program -> Statechart conversion
    2. Input -> Abstract value conversion
    3. Statechart simulation with symbolic/learned guards
    4. Visited states -> Covered lines mapping
    """

    def __init__(self, config: StatechartConfig):
        self.config = config
        self.learned_guard_predictor: Optional[LearnedGuardPredictor] = None

        if config.use_learned_guards:
            self.learned_guard_predictor = LearnedGuardPredictor(config)

    def predict_coverage(
        self,
        source: str,
        input_value: Any,
        input_var_name: str = "x",
    ) -> Tuple[Set[int], Dict]:
        """
        Predict coverage for a program with given input.

        Args:
            source: Python source code
            input_value: The input to the program
            input_var_name: Name of the input variable

        Returns:
            (covered_lines, debug_info)
        """
        # Build statechart
        statechart = build_cfg(source)

        # Convert input to abstract value
        abstract_input = AbstractValue.from_python(input_value)

        # Create executor
        executor = SymbolicExecutor(statechart)
        executor.set_input(input_var_name, abstract_input)

        # Run simulation
        visited_states = executor.run(self.config.max_iterations)

        # Get covered lines
        covered_lines = executor.get_covered_lines()

        debug_info = {
            'n_blocks': len(statechart.blocks),
            'n_transitions': len(statechart.transitions),
            'visited_blocks': len(visited_states),
            'visited_block_ids': sorted(visited_states),
        }

        return covered_lines, debug_info

    def predict_coverage_batch(
        self,
        sources: List[str],
        inputs: List[Any],
        input_var_name: str = "x",
    ) -> List[Set[int]]:
        """Predict coverage for a batch of (program, input) pairs."""
        results = []
        for source, inp in zip(sources, inputs):
            covered, _ = self.predict_coverage(source, inp, input_var_name)
            results.append(covered)
        return results


class DifferentiableStatechartPredictor(nn.Module):
    """
    Differentiable statechart coverage predictor.

    Uses soft state distributions instead of discrete states,
    allowing end-to-end training.

    State distribution: (batch, n_blocks) probability of being in each block
    Transitions: Soft attention-weighted message passing
    """

    def __init__(self, config: StatechartConfig, max_blocks: int = 50):
        super().__init__()
        self.config = config
        self.max_blocks = max_blocks

        # Block type embedding
        self.block_embed = nn.Embedding(10, config.embed_dim)  # 10 block types

        # Input encoder
        self.input_encoder = nn.Linear(config.embed_dim, config.embed_dim)

        # Transition scorer (predicts edge weights)
        self.transition_scorer = nn.Sequential(
            nn.Linear(config.embed_dim * 3, config.embed_dim),
            nn.ReLU(),
            nn.Linear(config.embed_dim, 1),
        )

        # Guard predictor
        self.guard_predictor = nn.Sequential(
            nn.Linear(config.embed_dim * 2, config.embed_dim),
            nn.ReLU(),
            nn.Linear(config.embed_dim, 1),
        )

        # Line classifier (from block distribution to line coverage)
        self.line_classifier = nn.Linear(max_blocks, 100)  # max 100 lines

    def __call__(
        self,
        block_types: mx.array,
        adjacency: mx.array,
        entry_mask: mx.array,
        input_embedding: mx.array,
        n_steps: int = 10,
    ) -> mx.array:
        """Make module callable."""
        return self.forward(block_types, adjacency, entry_mask, input_embedding, n_steps)

    def forward(
        self,
        block_types: mx.array,     # (batch, n_blocks) block type IDs
        adjacency: mx.array,       # (batch, n_blocks, n_blocks) CFG edges
        entry_mask: mx.array,      # (batch, n_blocks) 1 for entry blocks
        input_embedding: mx.array,  # (batch, embed_dim) input representation
        n_steps: int = 10,
    ) -> mx.array:
        """
        Predict line coverage distribution.

        Returns:
            (batch, max_lines) coverage probabilities
        """
        batch_size = block_types.shape[0]
        n_blocks = block_types.shape[1]

        # Embed blocks
        block_emb = self.block_embed(block_types)  # (B, N, D)

        # Encode input
        input_ctx = self.input_encoder(input_embedding)  # (B, D)
        input_ctx = input_ctx.reshape(batch_size, 1, -1)  # (B, 1, D)

        # Initialize state distribution at entry
        state_dist = entry_mask.astype(mx.float32)  # (B, N)
        state_dist = state_dist / (mx.sum(state_dist, axis=-1, keepdims=True) + 1e-8)

        # Accumulate visited states
        visited_dist = mx.array(state_dist)

        # Soft state propagation
        for _ in range(n_steps):
            # Score transitions from current distribution
            # For each edge (i, j), compute transition probability

            # Broadcast state distribution to edges
            src_dist = state_dist.reshape(batch_size, n_blocks, 1)  # (B, N, 1)
            src_emb = block_emb  # (B, N, D)

            # Target embeddings
            tgt_emb = block_emb  # (B, N, D)

            # Compute edge features
            # src_emb[i], tgt_emb[j], input_ctx
            edge_scores_list = []
            for i in range(n_blocks):
                for j in range(n_blocks):
                    if adjacency[0, i, j] > 0:  # Simplified: use first batch item
                        # Combine source, target, input
                        src = src_emb[:, i, :]  # (B, D)
                        tgt = tgt_emb[:, j, :]  # (B, D)
                        inp = input_ctx.squeeze(1)  # (B, D)
                        edge_feat = mx.concatenate([src, tgt, inp], axis=-1)  # (B, 3D)
                        score = self.transition_scorer(edge_feat)  # (B, 1)
                        edge_scores_list.append((i, j, score))

            # Soft message passing (simplified)
            new_dist = mx.zeros_like(state_dist)
            for i, j, score in edge_scores_list:
                # Add probability mass from i to j, weighted by transition score
                prob = mx.sigmoid(score.squeeze(-1))  # (B,)
                new_dist = new_dist.at[:, j].add(state_dist[:, i] * prob)

            # Normalize
            total = mx.sum(new_dist, axis=-1, keepdims=True) + 1e-8
            state_dist = new_dist / total

            # Accumulate visited
            visited_dist = mx.maximum(visited_dist, state_dist)

        # Convert block distribution to line coverage
        # This is learned - maps from block probs to line probs
        line_probs = mx.sigmoid(self.line_classifier(visited_dist))  # (B, max_lines)

        return line_probs


def demo():
    """Demonstrate statechart coverage prediction."""
    print("=" * 60)
    print("STATECHART COVERAGE PREDICTOR DEMO")
    print("=" * 60)

    config = StatechartConfig(max_iterations=50)
    predictor = StatechartCoveragePredictor(config)

    # Test program
    source = '''
def check_value(x):
    if x > 0:
        return "positive"
    elif x < 0:
        return "negative"
    else:
        return "zero"
'''

    print("\nSource code:")
    for i, line in enumerate(source.strip().split('\n'), 1):
        print(f"  {i:2d}: {line}")

    # Test with different inputs
    test_inputs = [5, -3, 0]

    print("\nPredicting coverage:")
    for inp in test_inputs:
        covered, debug = predictor.predict_coverage(source.strip(), inp, "x")
        print(f"\n  Input: x = {inp}")
        print(f"    Covered lines: {sorted(covered)}")
        print(f"    Visited blocks: {debug['visited_block_ids']}")

    # Test with list input
    print("\n" + "=" * 60)
    print("LIST INPUT EXAMPLE")
    print("=" * 60)

    list_source = '''
def sum_positive(items):
    total = 0
    for item in items:
        if item > 0:
            total += item
    return total
'''

    print("\nSource code:")
    for i, line in enumerate(list_source.strip().split('\n'), 1):
        print(f"  {i:2d}: {line}")

    list_inputs = [[], [1, 2, 3], [-1, -2], [1, -1, 2]]

    print("\nPredicting coverage:")
    for inp in list_inputs:
        covered, debug = predictor.predict_coverage(list_source.strip(), inp, "items")
        print(f"\n  Input: items = {inp}")
        print(f"    Covered lines: {sorted(covered)}")

    # Test differentiable predictor
    print("\n" + "=" * 60)
    print("DIFFERENTIABLE PREDICTOR")
    print("=" * 60)

    diff_config = StatechartConfig(embed_dim=32)
    diff_predictor = DifferentiableStatechartPredictor(diff_config, max_blocks=10)

    # Fake data
    block_types = mx.array([[1, 2, 3, 4, 5, 0, 0, 0, 0, 0]])
    adjacency = mx.array([[[1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                           [0, 0, 1, 0, 1, 0, 0, 0, 0, 0],
                           [0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]], dtype=mx.float32)
    entry_mask = mx.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=mx.float32)
    input_emb = mx.random.normal((1, 32))

    line_probs = diff_predictor(block_types, adjacency, entry_mask, input_emb)
    print(f"\nLine probabilities shape: {line_probs.shape}")
    print(f"First 10 line probabilities: {line_probs[0, :10].tolist()}")

    print("\n" + "=" * 60)
    print("Statechart coverage prediction complete!")
    print("=" * 60)


if __name__ == "__main__":
    demo()
