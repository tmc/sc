"""
exp_differentiable_lca: Differentiable LCA Computation

GOAL: Make LCA (Lowest Common Ancestor) computation differentiable for
end-to-end training of statechart hierarchies.

WHY THIS MATTERS:
LCA is critical for statechart semantics:
- Inter-level transitions require LCA to find exit/entry paths
- History state restoration needs LCA for scope
- Compound transitions chain through LCA

Traditional LCA is discrete (argmax over ancestors), blocking gradients.
Differentiable LCA enables learning hierarchy structure end-to-end.

KEY INSIGHT:
    Discrete LCA:  ancestors(a) ∩ ancestors(b) -> argmin(depth)
    Soft LCA:      attention(a_path, b_path) -> weighted ancestor

The soft version maintains gradient flow by replacing:
- Set intersection -> soft intersection (min of membership scores)
- Argmin over depth -> softmin over depth
- Hard ancestor -> soft ancestor attention

APPROACH:
1. SOFT TREE ATTENTION: Compute soft ancestor membership
   - Each node has soft membership in each ancestor
   - Attention mechanism over ancestor paths

2. DIFFERENTIABLE LCA: Gradient-friendly LCA computation
   - Soft path from node to root
   - Soft intersection of paths
   - Weighted combination of ancestors

3. TOPOLOGY LEARNING: Learn hierarchy structure
   - Gradient flows from LCA loss to topology
   - Can learn parent-child relationships

COMPARISON TO exp_lca_neural:
- exp_lca_neural: Predicts LCA via classification (discrete)
- exp_differentiable_lca: Computes LCA differentiably (continuous)

The key difference: we don't PREDICT LCA, we COMPUTE it differentiably.

Usage:
    from ml.experiments.exp_differentiable_lca import (
        SoftLCA,
        TreeAttention,
        DifferentiableHierarchy,
        LCABenchmark,
    )

    # Create differentiable hierarchy
    hierarchy = DifferentiableHierarchy(n_states=10)
    hierarchy.set_tree_structure(parent_indices)

    # Compute soft LCA
    soft_lca = SoftLCA(hierarchy)
    lca_scores = soft_lca.compute(state_a, state_b)  # Differentiable!

    # Use in training
    loss = mse(lca_scores, target_lca)
    loss.backward()  # Gradients flow through LCA!
"""

from .soft_lca import (
    SoftLCA,
    SoftLCAConfig,
    AncestorPath,
    soft_lca_forward,
    compute_soft_intersection,
)

from .tree_attention import (
    TreeAttention,
    TreeAttentionConfig,
    AncestorAttention,
    PathEncoder,
    soft_ancestor_weights,
)

from .differentiable_hierarchy import (
    DifferentiableHierarchy,
    HierarchyConfig,
    SoftParentMatrix,
    LearnableTree,
    hierarchy_from_adjacency,
)

from .benchmark import (
    DifferentiableLCABenchmark,
    BenchmarkResult,
    compare_soft_vs_discrete,
    run_gradient_flow_test,
)

__all__ = [
    # Soft LCA
    'SoftLCA',
    'SoftLCAConfig',
    'AncestorPath',
    'soft_lca_forward',
    'compute_soft_intersection',
    # Tree attention
    'TreeAttention',
    'TreeAttentionConfig',
    'AncestorAttention',
    'PathEncoder',
    'soft_ancestor_weights',
    # Differentiable hierarchy
    'DifferentiableHierarchy',
    'HierarchyConfig',
    'SoftParentMatrix',
    'LearnableTree',
    'hierarchy_from_adjacency',
    # Benchmark
    'DifferentiableLCABenchmark',
    'BenchmarkResult',
    'compare_soft_vs_discrete',
    'run_gradient_flow_test',
]
