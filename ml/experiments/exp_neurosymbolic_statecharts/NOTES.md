# Neuro-Symbolic Statecharts for Interactive Reasoning

Paper experiment code implementing the full architecture from
"Neuro-Symbolic Statecharts for Interactive Reasoning: Integrating
Nested Hulls and Recursive Language Models."

## Architecture

```
RLMStatechart
├── NeuroSymbolicStatechart (differentiable_sc.py)
│   ├── SoftAdjacency        — learnable topology A[i,j,e]
│   ├── NeuralPredicate      — continuous guards [0,1]
│   ├── TransitionAttention   — multi-head transition selection
│   ├── TypedMemory           — ephemeral/history/belief (typed_memory via hull)
│   │   └── HullKVCache      — O(log n) exact retrieval
│   └── HybridController     — mode-conditioned continuous control
├── HybridMemory (hybrid_memory.py)
│   ├── HullKVCache          — exact structural memory (O(log n))
│   ├── SSMMemory             — long-horizon temporal memory
│   │   └── SelectiveSSMBlock — input-dependent A, B discretization
│   ├── MemoryRouter          — learned hull-vs-SSM query routing
│   └── Combination modes     — CONCAT / ADD / GATE / ATTENTION
├── RecursiveDelegationManager (recursive_delegation.py)
│   ├── DelegationDecider    — learned delegate-or-handle decision
│   ├── ContextProjection    — gated projection to child
│   ├── ChildAgent           — independent sub-statechart
│   └── ResultAggregator     — attention/MoE over child results
├── ConstrainedDecoder (constrained_synthesis.py)
│   └── GrammarStatechart    — grammar as parser state machine
└── Training (training.py)
    ├── Trainer              — baseline AdamW + temperature annealing
    ├── RLMTrainer           — delegation loss extension
    ├── GRPOTrainer          — Group Relative Policy Optimization
    │   └── group-relative advantages, clipped surrogate, KL penalty
    └── SDPOTrainer          — Self-Distilled Policy Optimization
        ├── EMA teacher       — slow-moving parameter average
        └── Hindsight relabel — failed traces → synthetic positives
```

## Training Algorithms

| Algorithm | Loss | Critic? | Key Feature |
|-----------|------|---------|-------------|
| AdamW (Trainer) | CE + sparsity + entropy | N/A | Standard supervised |
| GRPO (GRPOTrainer) | Clipped surrogate + KL | No | Group-relative advantages |
| SDPO (SDPOTrainer) | GRPO + D_KL(θ∥θ_EMA) | No | EMA self-distillation + hindsight |

## Key Results

| Metric | Value | Paper Section |
|--------|-------|---------------|
| Constrained generation validity | 100% | §5 |
| Hull correctness (exact argmax) | 100% | §3 |
| Hull speedup over linear scan | O(log n) | §3 |
| Topology discovery F1 | varies | §5 |
| Hybrid memory combination modes | 4/4 pass | §3.12 |
| SSM temporal modeling | functional | §3.12 |

## Connections

- **exp_differentiable_statecharts**: Base Gumbel-Softmax implementation
- **exp_differentiable_lca**: Soft LCA for hierarchical operations
- **exp_mamba_sc_executor**: SSM-based statechart execution
- **exp_trm_vs_sc_sudoku**: TRM recursive reasoning model
- **mlx-go-computer**: HullKVCache original implementation

## Proto Proposal Coverage

Experiment code implements proto proposal sections:
- §3.4 NeuralMemoryConfig → HullKVCache, TypedMemory
- §3.5 EnsembleConfig → ResultAggregator (6 strategies)
- §3.7 HybridControlConfig → HybridController
- §3.8 RecursiveDelegationConfig → RLMStatechart
- §3.11 PolicyOptimizationConfig → GRPOTrainer, SDPOTrainer
- §3.12 HybridMemoryConfig → HybridMemory (Hull + SSM)
