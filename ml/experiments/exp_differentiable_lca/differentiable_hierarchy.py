"""
Differentiable Hierarchy: Learnable Tree Structures.

Makes tree hierarchy structure itself learnable through gradients.

Traditional tree: Fixed parent pointers (discrete)
Learnable tree: Soft parent matrix with gradient flow

Key components:
1. SoftParentMatrix: Differentiable parent relationships
2. LearnableTree: Tree structure as learnable parameters
3. DifferentiableHierarchy: Complete differentiable hierarchy

This enables:
- Learning hierarchy from data
- End-to-end training with LCA loss
- Discovering optimal tree structure
"""

import math
import random
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Set
from collections import defaultdict


@dataclass
class HierarchyConfig:
    """Configuration for differentiable hierarchy."""
    n_nodes: int = 10
    max_depth: int = 10

    # Learning parameters
    learning_rate: float = 0.01
    temperature: float = 1.0

    # Structural constraints
    enforce_single_parent: bool = True  # Each node has one parent
    enforce_acyclic: bool = True        # No cycles

    # Regularization
    sparsity_weight: float = 0.1        # Encourage sparse parent matrix
    depth_weight: float = 0.01          # Prefer shallow trees

    # Numerical stability
    eps: float = 1e-8


class SoftParentMatrix:
    """
    Soft parent relationships as a learnable matrix.

    parent_matrix[i][j] = P(node j is parent of node i)

    Constraints:
    - Row sums to 1 (softmax): each node has one soft parent
    - Diagonal is 0: node is not its own parent (except root)
    - Acyclic: no cycles in the structure
    """

    def __init__(self, config: HierarchyConfig):
        self.config = config
        self.n_nodes = config.n_nodes

        # Learnable logits (pre-softmax)
        # logits[i][j] = unnormalized score for j being parent of i
        random.seed(44)
        self.logits = [[random.gauss(0, 0.1) for _ in range(self.n_nodes)]
                       for _ in range(self.n_nodes)]

        # Set diagonal to very negative (node can't be own parent)
        for i in range(self.n_nodes):
            self.logits[i][i] = -100.0

    def _softmax(self, x: List[float], temperature: float = None) -> List[float]:
        """Softmax with temperature."""
        temp = temperature or self.config.temperature

        max_x = max(x) if x else 0
        exp_x = [math.exp((v - max_x) / temp) for v in x]
        sum_exp = sum(exp_x) + self.config.eps

        return [e / sum_exp for e in exp_x]

    def get_matrix(self) -> List[List[float]]:
        """
        Get the soft parent matrix.

        Returns [n_nodes, n_nodes] where M[i][j] = P(j is parent of i)
        """
        matrix = []
        for i in range(self.n_nodes):
            row = self._softmax(self.logits[i])
            matrix.append(row)
        return matrix

    def get_hard_parents(self) -> Dict[int, int]:
        """
        Get hard parent assignments (argmax).

        Returns dict mapping child -> parent.
        """
        matrix = self.get_matrix()
        parents = {}

        for i in range(self.n_nodes):
            parent = max(range(self.n_nodes), key=lambda j: matrix[i][j])
            if parent != i:  # Exclude self-loops
                parents[i] = parent

        return parents

    def update(self, gradients: List[List[float]], lr: float = None):
        """
        Update logits with gradients.

        Simple gradient descent update.
        """
        lr = lr or self.config.learning_rate

        for i in range(self.n_nodes):
            for j in range(self.n_nodes):
                self.logits[i][j] -= lr * gradients[i][j]

            # Re-enforce diagonal constraint
            self.logits[i][i] = -100.0

    def set_from_tree(self, parent_map: Dict[int, int], root: int):
        """
        Initialize logits from a known tree structure.

        Sets high logit for true parents, low for others.
        """
        for i in range(self.n_nodes):
            for j in range(self.n_nodes):
                self.logits[i][j] = -10.0  # Default low

            if i == root:
                # Root is its own parent (or has no parent)
                self.logits[i][i] = 10.0
            elif i in parent_map:
                parent = parent_map[i]
                self.logits[i][parent] = 10.0

    def sparsity_loss(self) -> float:
        """
        Compute sparsity loss to encourage clear parent assignments.

        Lower when parent matrix is close to one-hot.
        """
        matrix = self.get_matrix()
        entropy = 0.0

        for i in range(self.n_nodes):
            for j in range(self.n_nodes):
                p = matrix[i][j]
                if p > self.config.eps:
                    entropy -= p * math.log(p + self.config.eps)

        return entropy * self.config.sparsity_weight


class LearnableTree:
    """
    A tree structure with learnable topology.

    Combines soft parent matrix with constraint enforcement.
    """

    def __init__(self, config: HierarchyConfig):
        self.config = config
        self.n_nodes = config.n_nodes

        # Soft parent relationships
        self.parent_matrix = SoftParentMatrix(config)

        # Learnable depth (for regularization)
        self.soft_depths = [float(i % config.max_depth) for i in range(self.n_nodes)]

        # Root node (fixed or learnable)
        self.root = 0
        self.root_logits = [0.0] * self.n_nodes
        self.root_logits[0] = 10.0  # Default root at node 0

    def _softmax(self, x: List[float]) -> List[float]:
        """Softmax."""
        max_x = max(x) if x else 0
        exp_x = [math.exp(v - max_x) for v in x]
        sum_exp = sum(exp_x) + self.config.eps
        return [e / sum_exp for e in exp_x]

    def get_root_probs(self) -> List[float]:
        """Get soft probability of each node being root."""
        return self._softmax(self.root_logits)

    def get_depth_estimates(self) -> List[float]:
        """
        Estimate depth of each node from soft parent matrix.

        Uses iterative propagation.
        """
        matrix = self.parent_matrix.get_matrix()
        root_probs = self.get_root_probs()

        # Initialize depths: root has depth 0
        depths = root_probs.copy()  # Soft "is root" = depth 0 probability

        # Iterate to propagate depths
        for _ in range(self.config.max_depth):
            new_depths = [0.0] * self.n_nodes

            for i in range(self.n_nodes):
                # Depth of i = 1 + depth of parent
                for j in range(self.n_nodes):
                    parent_prob = matrix[i][j]
                    parent_depth = depths[j]
                    new_depths[i] += parent_prob * (parent_depth + 1)

                # If i is root, depth is 0
                new_depths[i] = root_probs[i] * 0 + (1 - root_probs[i]) * new_depths[i]

            depths = new_depths

        return depths

    def to_hard_tree(self) -> Tuple[Dict[int, int], int]:
        """
        Convert to hard tree structure.

        Returns (parent_map, root).
        """
        # Get hard root
        root_probs = self.get_root_probs()
        root = max(range(self.n_nodes), key=lambda i: root_probs[i])

        # Get hard parents
        parents = self.parent_matrix.get_hard_parents()

        # Remove root's parent
        if root in parents:
            del parents[root]

        return parents, root

    def depth_regularization(self) -> float:
        """
        Compute depth regularization loss.

        Prefers shallower trees.
        """
        depths = self.get_depth_estimates()
        avg_depth = sum(depths) / len(depths)
        return avg_depth * self.config.depth_weight


class DifferentiableHierarchy:
    """
    Complete differentiable hierarchy with LCA support.

    Combines:
    - Learnable tree structure
    - Soft LCA computation
    - Gradient-based optimization
    """

    def __init__(self, config: HierarchyConfig = None):
        self.config = config or HierarchyConfig()
        self.n_nodes = self.config.n_nodes

        # Learnable tree
        self.tree = LearnableTree(self.config)

        # Node embeddings (optional, for attention-based LCA)
        random.seed(45)
        self.node_embeddings = [[random.gauss(0, 0.1) for _ in range(32)]
                               for _ in range(self.n_nodes)]

    def set_tree_structure(self, parent_map: Dict[int, int], root: int):
        """Initialize from known tree structure."""
        self.tree.parent_matrix.set_from_tree(parent_map, root)
        self.tree.root = root
        self.tree.root_logits = [-10.0] * self.n_nodes
        self.tree.root_logits[root] = 10.0

    def compute_soft_lca(
        self,
        node_a: int,
        node_b: int,
        return_debug: bool = False
    ) -> Tuple[List[float], Optional[Dict]]:
        """
        Compute soft LCA using differentiable operations.

        Returns soft probability distribution over nodes being LCA.
        """
        matrix = self.tree.parent_matrix.get_matrix()
        depths = self.tree.get_depth_estimates()

        # Compute soft ancestor membership via iterative propagation
        membership_a = [0.0] * self.n_nodes
        membership_b = [0.0] * self.n_nodes

        membership_a[node_a] = 1.0
        membership_b[node_b] = 1.0

        # Propagate membership up the tree
        for _ in range(self.config.max_depth):
            new_a = membership_a.copy()
            new_b = membership_b.copy()

            for child in range(self.n_nodes):
                if membership_a[child] > self.config.eps:
                    for parent in range(self.n_nodes):
                        prob = matrix[child][parent]
                        new_a[parent] = max(new_a[parent],
                                           membership_a[child] * prob)

                if membership_b[child] > self.config.eps:
                    for parent in range(self.n_nodes):
                        prob = matrix[child][parent]
                        new_b[parent] = max(new_b[parent],
                                           membership_b[child] * prob)

            membership_a = new_a
            membership_b = new_b

        # Soft intersection
        common = [min(membership_a[i], membership_b[i])
                 for i in range(self.n_nodes)]

        # Weight by depth (prefer deeper)
        weighted = [common[i] * (depths[i] + 1) for i in range(self.n_nodes)]

        # Softmax to get distribution
        max_w = max(weighted) if weighted else 0
        exp_w = [math.exp((w - max_w) / self.config.temperature) for w in weighted]
        sum_exp = sum(exp_w) + self.config.eps
        lca_probs = [e / sum_exp for e in exp_w]

        debug = None
        if return_debug:
            debug = {
                'membership_a': membership_a,
                'membership_b': membership_b,
                'common': common,
                'weighted': weighted,
                'depths': depths,
            }

        return lca_probs, debug

    def lca_loss(
        self,
        node_a: int,
        node_b: int,
        target_lca: int
    ) -> float:
        """
        Compute loss for LCA prediction.

        Cross-entropy between soft LCA and target.
        """
        lca_probs, _ = self.compute_soft_lca(node_a, node_b)

        # Cross-entropy loss
        target_prob = lca_probs[target_lca]
        loss = -math.log(target_prob + self.config.eps)

        return loss

    def total_loss(
        self,
        examples: List[Tuple[int, int, int]]  # (node_a, node_b, target_lca)
    ) -> float:
        """
        Compute total loss over examples with regularization.
        """
        # LCA loss
        lca_loss = sum(self.lca_loss(a, b, lca) for a, b, lca in examples)
        lca_loss /= len(examples) if examples else 1

        # Regularization
        sparsity_loss = self.tree.parent_matrix.sparsity_loss()
        depth_loss = self.tree.depth_regularization()

        return lca_loss + sparsity_loss + depth_loss

    def get_hard_lca(self, node_a: int, node_b: int) -> int:
        """Get hard LCA from current tree structure."""
        parents, root = self.tree.to_hard_tree()

        # Get ancestors of a
        ancestors_a = set()
        current = node_a
        while current in parents:
            ancestors_a.add(current)
            current = parents[current]
        ancestors_a.add(current)  # Add root

        # Find first ancestor of b in ancestors_a
        current = node_b
        while current not in ancestors_a and current in parents:
            current = parents[current]

        return current


def hierarchy_from_adjacency(adjacency: List[List[int]], root: int = 0) -> DifferentiableHierarchy:
    """
    Create hierarchy from adjacency list.

    adjacency[i] = list of children of node i
    """
    n_nodes = len(adjacency)
    config = HierarchyConfig(n_nodes=n_nodes)
    hierarchy = DifferentiableHierarchy(config)

    # Build parent map from adjacency
    parent_map = {}
    for parent, children in enumerate(adjacency):
        for child in children:
            parent_map[child] = parent

    hierarchy.set_tree_structure(parent_map, root)
    return hierarchy


def test_differentiable_hierarchy():
    """Test differentiable hierarchy."""
    print("=" * 60)
    print("Testing Differentiable Hierarchy")
    print("=" * 60)

    # Create tree:
    #       0 (root)
    #      / \
    #     1   2
    #    / \   \
    #   3   4   5

    n_nodes = 6
    parent_map = {1: 0, 2: 0, 3: 1, 4: 1, 5: 2}

    config = HierarchyConfig(n_nodes=n_nodes, temperature=0.5)
    hierarchy = DifferentiableHierarchy(config)
    hierarchy.set_tree_structure(parent_map, root=0)

    print("\n1. Tree structure:")
    print("       0 (root)")
    print("      / \\")
    print("     1   2")
    print("    / \\   \\")
    print("   3   4   5")

    print("\n2. Soft parent matrix (rows=children, cols=parents):")
    matrix = hierarchy.tree.parent_matrix.get_matrix()
    print("     ", end="")
    for j in range(n_nodes):
        print(f"  {j}   ", end="")
    print()
    for i in range(n_nodes):
        print(f"  {i}: ", end="")
        for j in range(n_nodes):
            print(f"{matrix[i][j]:.3f} ", end="")
        print()

    print("\n3. Depth estimates:")
    depths = hierarchy.tree.get_depth_estimates()
    for i in range(n_nodes):
        print(f"  Node {i}: depth={depths[i]:.3f}")

    print("\n4. Soft LCA tests:")
    test_pairs = [
        (3, 4, 1, "siblings -> parent"),
        (3, 5, 0, "cousins -> root"),
        (3, 1, 1, "child-parent -> parent"),
        (1, 2, 0, "siblings -> root"),
    ]

    for a, b, expected, desc in test_pairs:
        lca_probs, debug = hierarchy.compute_soft_lca(a, b, return_debug=True)
        soft_lca = max(range(n_nodes), key=lambda i: lca_probs[i])
        hard_lca = hierarchy.get_hard_lca(a, b)

        loss = hierarchy.lca_loss(a, b, expected)

        status = "PASS" if soft_lca == expected else "FAIL"
        print(f"\n  LCA({a}, {b}): soft={soft_lca}, hard={hard_lca}, "
              f"expected={expected} [{status}] ({desc})")
        print(f"    Probs: {[f'{p:.3f}' for p in lca_probs]}")
        print(f"    Loss: {loss:.4f}")

    print("\n5. Gradient flow test:")
    # Perturb the tree structure slightly
    original_logits = [row.copy() for row in hierarchy.tree.parent_matrix.logits]

    # Compute loss before perturbation
    examples = [(3, 4, 1), (3, 5, 0), (1, 2, 0)]
    loss_before = hierarchy.total_loss(examples)

    # Perturb
    hierarchy.tree.parent_matrix.logits[3][1] += 0.5
    hierarchy.tree.parent_matrix.logits[3][2] -= 0.5

    # Loss after
    loss_after = hierarchy.total_loss(examples)

    print(f"  Loss before perturbation: {loss_before:.4f}")
    print(f"  Loss after perturbation: {loss_after:.4f}")
    print(f"  Change: {loss_after - loss_before:+.4f}")

    # Restore
    hierarchy.tree.parent_matrix.logits = original_logits

    print("\n6. Creating hierarchy from adjacency:")
    adjacency = [[1, 2], [3, 4], [5], [], [], []]  # Same tree
    h2 = hierarchy_from_adjacency(adjacency, root=0)
    p2, r2 = h2.tree.to_hard_tree()
    print(f"  Root: {r2}")
    print(f"  Parents: {p2}")

    print("\n" + "=" * 60)
    print("Differentiable Hierarchy tests complete!")
    print("=" * 60)


if __name__ == "__main__":
    test_differentiable_hierarchy()
