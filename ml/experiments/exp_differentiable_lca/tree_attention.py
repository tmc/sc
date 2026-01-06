"""
Tree Attention: Soft Attention Mechanisms for Tree Structures.

Implements attention mechanisms that respect tree structure:
1. Ancestor Attention: Attend over ancestors of a node
2. Path Attention: Attend over path from node to root
3. Subtree Attention: Attend over descendants

These enable soft/differentiable tree traversal for LCA computation.

Key insight: Traditional tree traversal is discrete (follow parent pointers).
Tree attention makes this soft by computing attention weights over all
potential ancestors, weighted by structural plausibility.
"""

import math
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Set


@dataclass
class TreeAttentionConfig:
    """Configuration for tree attention mechanisms."""
    # Attention parameters
    n_heads: int = 4
    embed_dim: int = 64
    temperature: float = 1.0

    # Structural constraints
    enforce_tree: bool = True    # Enforce single parent constraint
    max_depth: int = 20

    # Numerical stability
    eps: float = 1e-8


class PathEncoder:
    """
    Encode paths in tree structure.

    Creates embeddings that capture:
    - Node identity
    - Position in path
    - Depth information
    """

    def __init__(self, config: TreeAttentionConfig, n_nodes: int):
        self.config = config
        self.n_nodes = n_nodes
        self.embed_dim = config.embed_dim

        # Initialize embeddings (would be learned in full implementation)
        import random
        random.seed(42)

        # Node embeddings
        self.node_embed = [[random.gauss(0, 0.1) for _ in range(self.embed_dim)]
                          for _ in range(n_nodes)]

        # Positional embeddings for path position
        self.pos_embed = [[random.gauss(0, 0.1) for _ in range(self.embed_dim)]
                         for _ in range(config.max_depth)]

        # Depth embeddings
        self.depth_embed = [[random.gauss(0, 0.1) for _ in range(self.embed_dim)]
                           for _ in range(config.max_depth)]

    def encode_node(self, node_id: int, path_pos: int = 0, depth: int = 0) -> List[float]:
        """
        Encode a node with positional and depth information.

        Returns embedding vector.
        """
        if node_id >= self.n_nodes:
            return [0.0] * self.embed_dim

        node_emb = self.node_embed[node_id]
        pos_emb = self.pos_embed[min(path_pos, self.config.max_depth - 1)]
        depth_emb = self.depth_embed[min(depth, self.config.max_depth - 1)]

        # Combine: sum of embeddings (could be learned combination)
        result = []
        for i in range(self.embed_dim):
            result.append(node_emb[i] + pos_emb[i] + depth_emb[i])

        return result

    def encode_path(self, path: List[int], depths: List[int]) -> List[List[float]]:
        """Encode a complete path."""
        embeddings = []
        for i, (node_id, depth) in enumerate(zip(path, depths)):
            emb = self.encode_node(node_id, path_pos=i, depth=depth)
            embeddings.append(emb)
        return embeddings


class AncestorAttention:
    """
    Attention mechanism for selecting ancestors.

    Given a query node, computes attention weights over all potential
    ancestors based on structural likelihood and learned relevance.
    """

    def __init__(self, config: TreeAttentionConfig, n_nodes: int):
        self.config = config
        self.n_nodes = n_nodes

        # Initialize attention parameters (would be learned)
        import random
        random.seed(43)

        embed_dim = config.embed_dim
        head_dim = embed_dim // config.n_heads

        # Query, Key, Value projections per head
        self.W_q = [[[random.gauss(0, 0.1) for _ in range(embed_dim)]
                    for _ in range(head_dim)]
                   for _ in range(config.n_heads)]

        self.W_k = [[[random.gauss(0, 0.1) for _ in range(embed_dim)]
                    for _ in range(head_dim)]
                   for _ in range(config.n_heads)]

        self.W_v = [[[random.gauss(0, 0.1) for _ in range(embed_dim)]
                    for _ in range(head_dim)]
                   for _ in range(config.n_heads)]

    def _matmul(self, W: List[List[float]], x: List[float]) -> List[float]:
        """Matrix-vector multiplication."""
        return [sum(W[i][j] * x[j] for j in range(len(x))) for i in range(len(W))]

    def _dot(self, a: List[float], b: List[float]) -> float:
        """Dot product."""
        return sum(a[i] * b[i] for i in range(min(len(a), len(b))))

    def _softmax(self, x: List[float], temperature: float = None) -> List[float]:
        """Softmax with temperature."""
        temp = temperature or self.config.temperature

        if not x:
            return []

        max_x = max(x)
        exp_x = [math.exp((v - max_x) / temp) for v in x]
        sum_exp = sum(exp_x) + self.config.eps

        return [e / sum_exp for e in exp_x]

    def compute_attention(
        self,
        query_embedding: List[float],
        key_embeddings: List[List[float]],
        value_embeddings: List[List[float]],
        mask: List[bool] = None
    ) -> Tuple[List[float], List[float]]:
        """
        Compute attention over candidate ancestors.

        Args:
            query_embedding: Query node embedding [embed_dim]
            key_embeddings: Candidate ancestor embeddings [n_candidates, embed_dim]
            value_embeddings: Value embeddings [n_candidates, embed_dim]
            mask: Optional mask for valid candidates [n_candidates]

        Returns:
            attended_value: Weighted sum of values [embed_dim]
            attention_weights: Attention distribution [n_candidates]
        """
        n_candidates = len(key_embeddings)

        if n_candidates == 0:
            return [0.0] * self.config.embed_dim, []

        # Multi-head attention (simplified: average over heads)
        all_weights = []

        for head in range(self.config.n_heads):
            # Project query
            q = self._matmul(self.W_q[head], query_embedding)

            # Project keys and compute scores
            scores = []
            for k_emb in key_embeddings:
                k = self._matmul(self.W_k[head], k_emb)
                score = self._dot(q, k) / math.sqrt(len(q))
                scores.append(score)

            # Apply mask
            if mask:
                for i, m in enumerate(mask):
                    if not m:
                        scores[i] = float('-inf')

            # Softmax
            weights = self._softmax(scores)
            all_weights.append(weights)

        # Average weights across heads
        avg_weights = []
        for i in range(n_candidates):
            avg = sum(all_weights[h][i] for h in range(self.config.n_heads))
            avg_weights.append(avg / self.config.n_heads)

        # Compute attended value
        attended = [0.0] * self.config.embed_dim
        for i, (v_emb, weight) in enumerate(zip(value_embeddings, avg_weights)):
            for j in range(min(len(attended), len(v_emb))):
                attended[j] += weight * v_emb[j]

        return attended, avg_weights


class TreeAttention:
    """
    Full tree attention mechanism.

    Combines path encoding and ancestor attention for soft tree traversal.
    """

    def __init__(self, config: TreeAttentionConfig = None, n_nodes: int = 10):
        self.config = config or TreeAttentionConfig()
        self.n_nodes = n_nodes

        # Components
        self.path_encoder = PathEncoder(self.config, n_nodes)
        self.ancestor_attention = AncestorAttention(self.config, n_nodes)

        # Tree structure (set externally)
        self.parent: Dict[int, int] = {}
        self.children: Dict[int, List[int]] = {}
        self.depth: Dict[int, int] = {}
        self.root: int = 0

    def set_tree(self, parent_map: Dict[int, int], root: int):
        """Set the tree structure."""
        self.parent = parent_map
        self.root = root

        # Build children map
        self.children = {i: [] for i in range(self.n_nodes)}
        for child, parent in parent_map.items():
            if parent < self.n_nodes:
                self.children[parent].append(child)

        # Compute depths
        self.depth = {root: 0}
        for node in parent_map:
            self._compute_depth(node)

    def _compute_depth(self, node: int) -> int:
        """Compute depth of a node."""
        if node in self.depth:
            return self.depth[node]

        if node == self.root:
            self.depth[node] = 0
            return 0

        parent = self.parent.get(node, self.root)
        self.depth[node] = self._compute_depth(parent) + 1
        return self.depth[node]

    def get_ancestors(self, node: int) -> List[int]:
        """Get list of ancestors from node to root."""
        ancestors = []
        current = node

        while current != self.root:
            parent = self.parent.get(current, self.root)
            ancestors.append(parent)
            current = parent

        return ancestors

    def soft_ancestor_weights(
        self,
        node: int,
        candidates: List[int] = None
    ) -> List[float]:
        """
        Compute soft weights over potential ancestors.

        Uses attention to weight candidates by structural plausibility.
        """
        if candidates is None:
            # All nodes are candidates
            candidates = list(range(self.n_nodes))

        # Get true ancestors for masking
        true_ancestors = set(self.get_ancestors(node))
        true_ancestors.add(node)  # Node is its own ancestor (trivially)

        # Encode query node
        query_depth = self.depth.get(node, 0)
        query_emb = self.path_encoder.encode_node(node, path_pos=0, depth=query_depth)

        # Encode candidates
        key_embs = []
        value_embs = []
        mask = []

        for cand in candidates:
            cand_depth = self.depth.get(cand, 0)
            emb = self.path_encoder.encode_node(cand, path_pos=0, depth=cand_depth)
            key_embs.append(emb)
            value_embs.append(emb)

            # Mask: only true ancestors (or relax for soft version)
            is_ancestor = cand in true_ancestors or cand_depth < query_depth
            mask.append(is_ancestor)

        # Compute attention
        _, weights = self.ancestor_attention.compute_attention(
            query_emb, key_embs, value_embs, mask
        )

        return weights

    def soft_lca_attention(
        self,
        node_a: int,
        node_b: int
    ) -> List[float]:
        """
        Compute soft LCA using tree attention.

        Returns attention weights over all nodes for LCA.
        """
        # Get ancestor weights for each node
        weights_a = self.soft_ancestor_weights(node_a)
        weights_b = self.soft_ancestor_weights(node_b)

        # Soft intersection (element-wise min)
        common_weights = [min(weights_a[i], weights_b[i])
                         for i in range(self.n_nodes)]

        # Weight by depth (prefer deeper common ancestors)
        depth_weighted = []
        for i in range(self.n_nodes):
            depth = self.depth.get(i, 0)
            depth_weighted.append(common_weights[i] * (depth + 1))

        # Normalize
        total = sum(depth_weighted) + self.config.eps
        normalized = [w / total for w in depth_weighted]

        return normalized


def soft_ancestor_weights(
    node: int,
    parent_map: Dict[int, int],
    n_nodes: int,
    temperature: float = 1.0
) -> List[float]:
    """
    Compute soft ancestor membership weights.

    Convenience function for simple cases.
    """
    config = TreeAttentionConfig(temperature=temperature)
    tree_attn = TreeAttention(config, n_nodes)

    # Find root
    nodes_with_parents = set(parent_map.keys())
    parents = set(parent_map.values())
    root_candidates = parents - nodes_with_parents
    root = list(root_candidates)[0] if root_candidates else 0

    tree_attn.set_tree(parent_map, root)
    return tree_attn.soft_ancestor_weights(node)


def test_tree_attention():
    """Test tree attention mechanisms."""
    print("=" * 60)
    print("Testing Tree Attention")
    print("=" * 60)

    # Create tree:
    #       0 (root)
    #      / \
    #     1   2
    #    / \   \
    #   3   4   5

    parent_map = {1: 0, 2: 0, 3: 1, 4: 1, 5: 2}
    n_nodes = 6

    config = TreeAttentionConfig(n_heads=2, embed_dim=32)
    tree_attn = TreeAttention(config, n_nodes)
    tree_attn.set_tree(parent_map, root=0)

    print("\n1. Tree structure:")
    print("       0 (root)")
    print("      / \\")
    print("     1   2")
    print("    / \\   \\")
    print("   3   4   5")

    print("\n2. Ancestor weights:")
    for node in range(n_nodes):
        true_ancestors = tree_attn.get_ancestors(node)
        weights = tree_attn.soft_ancestor_weights(node)

        print(f"\n  Node {node}:")
        print(f"    True ancestors: {true_ancestors}")
        print(f"    Soft weights: {[f'{w:.3f}' for w in weights]}")

        # Find max weight
        max_idx = max(range(len(weights)), key=lambda i: weights[i])
        print(f"    Most likely ancestor: {max_idx} (weight={weights[max_idx]:.3f})")

    print("\n3. Soft LCA via attention:")
    test_pairs = [(3, 4), (3, 5), (3, 1), (1, 2)]

    for a, b in test_pairs:
        lca_weights = tree_attn.soft_lca_attention(a, b)
        soft_lca = max(range(len(lca_weights)), key=lambda i: lca_weights[i])

        # Hard LCA for comparison
        ancestors_a = set(tree_attn.get_ancestors(a)) | {a}
        current = b
        hard_lca = b
        while current not in ancestors_a:
            current = parent_map.get(current, 0)
            hard_lca = current

        print(f"\n  LCA({a}, {b}):")
        print(f"    Hard LCA: {hard_lca}")
        print(f"    Soft LCA: {soft_lca}")
        print(f"    Weights: {[f'{w:.3f}' for w in lca_weights]}")

    print("\n" + "=" * 60)
    print("Tree Attention tests complete!")
    print("=" * 60)


if __name__ == "__main__":
    test_tree_attention()
