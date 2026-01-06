"""
Soft LCA: Differentiable Lowest Common Ancestor.

Makes LCA computation differentiable by replacing discrete operations
with continuous approximations:

    Discrete                    Soft/Differentiable
    --------                    -------------------
    Set intersection    -->     Element-wise min of membership
    Argmin over depth   -->     Softmin (negative softmax)
    Hard parent lookup  -->     Soft parent attention

The key insight: LCA(a, b) = deepest common ancestor
    = argmax_{c in ancestors(a) ∩ ancestors(b)} depth(c)

Soft version:
    soft_LCA(a, b) = softmax(depth * membership_a * membership_b)

This allows gradients to flow through LCA for:
- Learning hierarchy structure
- End-to-end transition training
- Differentiable statechart execution
"""

import math
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Set
from collections import defaultdict


@dataclass
class SoftLCAConfig:
    """Configuration for soft LCA computation."""
    # Temperature for softmax operations
    temperature: float = 1.0

    # Depth weighting (higher = prefer deeper ancestors)
    depth_weight: float = 1.0

    # Numerical stability
    eps: float = 1e-8

    # Max depth for normalization
    max_depth: int = 20


@dataclass
class AncestorPath:
    """Path from a node to the root."""
    node_id: int
    ancestors: List[int]          # [parent, grandparent, ..., root]
    depths: List[int]             # [depth-1, depth-2, ..., 0]
    membership_scores: List[float]  # Soft membership (1.0 for hard)


class SoftLCA:
    """
    Differentiable LCA computation.

    Computes LCA using soft operations that maintain gradient flow.
    """

    def __init__(self, config: SoftLCAConfig = None):
        self.config = config or SoftLCAConfig()

        # Tree structure (set externally or learned)
        self.parent: Dict[int, int] = {}  # node -> parent
        self.depth: Dict[int, int] = {}   # node -> depth
        self.n_nodes: int = 0
        self.root: int = -1

    def set_tree(self, parent_map: Dict[int, int], root: int):
        """Set the tree structure."""
        self.parent = parent_map
        self.root = root
        self.n_nodes = len(parent_map) + 1  # +1 for root

        # Compute depths
        self.depth = {root: 0}
        for node in parent_map:
            self._compute_depth(node)

    def _compute_depth(self, node: int) -> int:
        """Compute depth of a node (memoized)."""
        if node in self.depth:
            return self.depth[node]

        if node == self.root:
            self.depth[node] = 0
            return 0

        parent = self.parent.get(node, self.root)
        self.depth[node] = self._compute_depth(parent) + 1
        return self.depth[node]

    def get_ancestor_path(self, node: int) -> AncestorPath:
        """
        Get the path from node to root.

        Returns ancestor list with depths and membership scores.
        """
        ancestors = []
        depths = []
        current = node

        while current != self.root:
            parent = self.parent.get(current, self.root)
            ancestors.append(parent)
            depths.append(self.depth.get(parent, 0))
            current = parent

        # Hard membership for now (will be soft in tree_attention)
        membership = [1.0] * len(ancestors)

        return AncestorPath(
            node_id=node,
            ancestors=ancestors,
            depths=depths,
            membership_scores=membership
        )

    def _softmax(self, x: List[float], temperature: float = None) -> List[float]:
        """Compute softmax with temperature."""
        temp = temperature or self.config.temperature

        # Numerical stability: subtract max
        max_x = max(x) if x else 0
        exp_x = [math.exp((v - max_x) / temp) for v in x]
        sum_exp = sum(exp_x) + self.config.eps

        return [e / sum_exp for e in exp_x]

    def _element_wise_min(
        self,
        a: List[float],
        b: List[float]
    ) -> List[float]:
        """
        Soft intersection via element-wise min.

        For hard membership (0/1), min gives intersection.
        For soft membership, min gives soft intersection.
        """
        result = []
        for i in range(min(len(a), len(b))):
            result.append(min(a[i], b[i]))
        return result

    def compute_hard(self, node_a: int, node_b: int) -> int:
        """
        Compute LCA using traditional discrete algorithm.

        Used as ground truth for comparison.
        """
        # Get ancestors of both nodes
        ancestors_a = set()
        current = node_a
        while current != self.root:
            ancestors_a.add(current)
            current = self.parent.get(current, self.root)
        ancestors_a.add(self.root)

        # Find first ancestor of b that's in ancestors_a
        current = node_b
        while current not in ancestors_a:
            current = self.parent.get(current, self.root)

        return current

    def compute_soft(
        self,
        node_a: int,
        node_b: int,
        return_weights: bool = False
    ) -> Tuple[List[float], Optional[Dict]]:
        """
        Compute soft LCA as weighted combination of ancestors.

        Returns:
            lca_weights: Soft probability over all nodes being LCA
            debug_info: Optional debug information
        """
        # Initialize weights for all nodes
        lca_weights = [0.0] * self.n_nodes

        # Get ancestor paths
        path_a = self.get_ancestor_path(node_a)
        path_b = self.get_ancestor_path(node_b)

        # Include the nodes themselves
        all_a = [node_a] + path_a.ancestors
        all_b = [node_b] + path_b.ancestors
        depths_a = [self.depth.get(node_a, 0)] + path_a.depths
        depths_b = [self.depth.get(node_b, 0)] + path_b.depths

        # Find common ancestors
        set_a = set(all_a)
        common = [n for n in all_b if n in set_a]

        if not common:
            # No common ancestor (shouldn't happen in valid tree)
            lca_weights[self.root] = 1.0
            return lca_weights, None

        # Score each common ancestor by depth (prefer deeper)
        scores = []
        for node in common:
            depth = self.depth.get(node, 0)
            score = depth * self.config.depth_weight
            scores.append(score)

        # Softmax to get weights
        weights = self._softmax(scores)

        # Assign weights to common ancestors
        for i, node in enumerate(common):
            if node < len(lca_weights):
                lca_weights[node] = weights[i]

        debug = None
        if return_weights:
            debug = {
                'common_ancestors': common,
                'scores': scores,
                'weights': weights,
                'path_a': all_a,
                'path_b': all_b,
            }

        return lca_weights, debug

    def compute(
        self,
        node_a: int,
        node_b: int
    ) -> List[float]:
        """
        Compute soft LCA weights.

        Main entry point for differentiable LCA.
        """
        weights, _ = self.compute_soft(node_a, node_b)
        return weights

    def get_soft_lca_node(self, weights: List[float]) -> int:
        """Get the most likely LCA node from soft weights."""
        if not weights:
            return self.root
        return max(range(len(weights)), key=lambda i: weights[i])


def soft_lca_forward(
    node_a: int,
    node_b: int,
    parent_matrix: List[List[float]],
    depth_vector: List[float],
    temperature: float = 1.0
) -> List[float]:
    """
    Forward pass for differentiable LCA.

    This version uses soft parent matrix for fully differentiable computation.

    Args:
        node_a, node_b: Query nodes
        parent_matrix: Soft parent relationships [n_nodes, n_nodes]
                      parent_matrix[i][j] = P(j is parent of i)
        depth_vector: Depth of each node [n_nodes]
        temperature: Softmax temperature

    Returns:
        lca_weights: Soft probability distribution over nodes
    """
    n_nodes = len(parent_matrix)
    eps = 1e-8

    # Compute soft ancestor membership via iterative parent traversal
    # membership_a[i] = P(node i is ancestor of node_a)
    membership_a = [0.0] * n_nodes
    membership_b = [0.0] * n_nodes

    # Start with full membership at the query nodes
    membership_a[node_a] = 1.0
    membership_b[node_b] = 1.0

    # Propagate membership up the tree (soft)
    max_depth = int(max(depth_vector)) + 1

    for _ in range(max_depth):
        new_a = membership_a.copy()
        new_b = membership_b.copy()

        for child in range(n_nodes):
            if membership_a[child] > eps:
                # Propagate to parents
                for parent in range(n_nodes):
                    prob = parent_matrix[child][parent]
                    new_a[parent] = max(new_a[parent], membership_a[child] * prob)

            if membership_b[child] > eps:
                for parent in range(n_nodes):
                    prob = parent_matrix[child][parent]
                    new_b[parent] = max(new_b[parent], membership_b[child] * prob)

        membership_a = new_a
        membership_b = new_b

    # Soft intersection: element-wise min
    common_membership = [min(membership_a[i], membership_b[i])
                        for i in range(n_nodes)]

    # Weight by depth (prefer deeper common ancestors)
    scores = [common_membership[i] * depth_vector[i]
              for i in range(n_nodes)]

    # Softmax over scores
    max_score = max(scores) if scores else 0
    exp_scores = [math.exp((s - max_score) / temperature) for s in scores]
    sum_exp = sum(exp_scores) + eps

    lca_weights = [e / sum_exp for e in exp_scores]

    return lca_weights


def compute_soft_intersection(
    membership_a: List[float],
    membership_b: List[float],
    mode: str = "min"
) -> List[float]:
    """
    Compute soft intersection of two membership vectors.

    Modes:
        "min": Element-wise min (standard fuzzy intersection)
        "product": Element-wise product (probabilistic)
        "harmonic": Harmonic mean (balanced)
    """
    n = min(len(membership_a), len(membership_b))

    if mode == "min":
        return [min(membership_a[i], membership_b[i]) for i in range(n)]
    elif mode == "product":
        return [membership_a[i] * membership_b[i] for i in range(n)]
    elif mode == "harmonic":
        result = []
        for i in range(n):
            a, b = membership_a[i], membership_b[i]
            if a + b > 0:
                result.append(2 * a * b / (a + b))
            else:
                result.append(0.0)
        return result
    else:
        raise ValueError(f"Unknown mode: {mode}")


def test_soft_lca():
    """Test the soft LCA computation."""
    print("=" * 60)
    print("Testing Soft LCA")
    print("=" * 60)

    # Create a simple tree:
    #       0 (root)
    #      / \
    #     1   2
    #    / \   \
    #   3   4   5

    parent_map = {
        1: 0,  # 1's parent is 0
        2: 0,  # 2's parent is 0
        3: 1,  # 3's parent is 1
        4: 1,  # 4's parent is 1
        5: 2,  # 5's parent is 2
    }

    soft_lca = SoftLCA(SoftLCAConfig(temperature=0.5))
    soft_lca.set_tree(parent_map, root=0)

    print("\n1. Tree structure:")
    print("       0 (root)")
    print("      / \\")
    print("     1   2")
    print("    / \\   \\")
    print("   3   4   5")

    print("\n2. Testing LCA computations:")

    test_cases = [
        (3, 4, 1, "siblings -> parent"),
        (3, 5, 0, "cousins -> root"),
        (3, 1, 1, "child-parent -> parent"),
        (1, 2, 0, "siblings -> root"),
        (3, 3, 3, "same node -> self"),
    ]

    for node_a, node_b, expected, description in test_cases:
        # Hard LCA
        hard_lca = soft_lca.compute_hard(node_a, node_b)

        # Soft LCA
        weights, debug = soft_lca.compute_soft(node_a, node_b, return_weights=True)
        soft_result = soft_lca.get_soft_lca_node(weights)

        status = "PASS" if hard_lca == expected and soft_result == expected else "FAIL"
        print(f"  LCA({node_a}, {node_b}): hard={hard_lca}, soft={soft_result}, "
              f"expected={expected} [{status}] ({description})")

        if debug:
            print(f"    Common ancestors: {debug['common_ancestors']}")
            print(f"    Weights: {[f'{w:.3f}' for w in debug['weights']]}")

    # Test with soft parent matrix
    print("\n3. Testing fully differentiable forward pass:")

    # Create soft parent matrix (hard for this test)
    n_nodes = 6
    parent_matrix = [[0.0] * n_nodes for _ in range(n_nodes)]
    for child, parent in parent_map.items():
        parent_matrix[child][parent] = 1.0
    # Root is its own parent
    parent_matrix[0][0] = 1.0

    depth_vector = [0, 1, 1, 2, 2, 2]

    lca_weights = soft_lca_forward(3, 4, parent_matrix, depth_vector, temperature=0.5)
    soft_result = max(range(len(lca_weights)), key=lambda i: lca_weights[i])

    print(f"  LCA(3, 4) via soft forward: {soft_result}")
    print(f"  Weight distribution: {[f'{w:.3f}' for w in lca_weights]}")

    # Test gradient-like perturbation
    print("\n4. Testing gradient sensitivity:")

    # Slightly perturb parent relationship
    parent_matrix_perturbed = [row.copy() for row in parent_matrix]
    parent_matrix_perturbed[3][1] = 0.9  # Reduce confidence in 3->1
    parent_matrix_perturbed[3][2] = 0.1  # Add small probability 3->2

    lca_weights_perturbed = soft_lca_forward(3, 4, parent_matrix_perturbed, depth_vector)
    soft_result_perturbed = max(range(len(lca_weights_perturbed)),
                                key=lambda i: lca_weights_perturbed[i])

    print(f"  Original LCA(3,4): {soft_result}")
    print(f"  Perturbed LCA(3,4): {soft_result_perturbed}")
    print(f"  Weight change: {[f'{lca_weights_perturbed[i] - lca_weights[i]:+.3f}' for i in range(n_nodes)]}")

    print("\n" + "=" * 60)
    print("Soft LCA tests complete!")
    print("=" * 60)


if __name__ == "__main__":
    test_soft_lca()
