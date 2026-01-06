# exp_differentiable_statecharts: Research Notes

## Core Insight

**Statecharts become learnable end-to-end when we relax discrete operations to continuous ones.**

| Hard Operation | Soft Relaxation |
|----------------|-----------------|
| `argmax(scores)` | `softmax(scores/τ)` |
| `guard ∈ {0,1}` | `σ(score) ∈ [0,1]` |
| `A[i,j] ∈ {0,1}` | `A[i,j] ∈ [0,1]` |
| State selection | Gumbel-softmax |

---

## Key Techniques

### 1. Gumbel-Softmax [Jang et al., 2017]

Enables gradient flow through categorical sampling:

```python
y = softmax((log(π) + g) / τ)  # g ~ Gumbel(0,1)
```

- τ → 0: approaches one-hot (discrete)
- τ → ∞: approaches uniform
- Annealing: start soft, end discrete

### 2. Differentiable Guards

Replace boolean predicates with soft scores:

```python
# Hard: x > threshold
# Soft: σ((x - threshold) / τ)
```

Supports:
- Threshold predicates
- Comparison predicates
- Learned neural predicates

### 3. Soft Adjacency Matrix

Topology as learnable parameters:

```
A[i,j,e] = P(transition from i to j on event e)
```

Regularization:
- L1 sparsity: encourage minimal edges
- Entropy: encourage decisive transitions

### 4. Attention-Based Transitions

Complex transition selection via attention:

```
Query:  current_state + event
Keys:   transition embeddings
Values: target state embeddings
Mask:   soft guard scores
```

---

## What Worked

1. **Temperature annealing** - Critical for convergence
   - Start: τ=1.0 (exploration)
   - End: τ=0.1 (exploitation)
   - Rate: 0.995 per epoch

2. **Sparsity regularization** - Prevents fully connected topology
   - L1 on edge probabilities
   - Coefficient: 0.1

3. **Combined state+context encoding** - Better than separate
   - Add state embedding + context encoding
   - Share representation for guard evaluation

4. **Guard modulation of topology** - More expressive than fixed edges
   - `transition_prob = topology[i,j] * guard_score`

## What Didn't Work (Initially)

1. **Hard Gumbel-softmax** - Gradient issues
   - Straight-through estimator unstable
   - Soft version more reliable

2. **Per-transition guards** - Too many parameters
   - n_states² × n_events guards
   - Shared guard network better

3. **No regularization** - Learns dense topology
   - Everything connected
   - No structure emerges

4. **Fixed temperature** - Poor convergence
   - Too high: never discrete
   - Too low: no exploration

---

## Connections to Other Experiments

### exp_topology_evolution
- **This**: Gradient-based topology search
- **That**: Evolutionary topology search
- **Synthesis**: Hybrid - evolution for structure, gradients for fine-tuning

### exp_transition_priorities
- **This**: Attention weights = soft priorities
- **That**: Explicit priority evolution
- **Synthesis**: Learn priority as attention temperature

### exp_guard_synthesis
- **This**: Differentiable guards (soft predicates)
- **That**: Symbolic guard evolution
- **Synthesis**: Distill learned guards to symbolic form

### exp_sae_statechart
- **This**: Learned state embeddings
- **That**: SAE-discovered states
- **Synthesis**: Use SAE states as initialization

### exp_learnable_policies
- **This**: Policy as differentiable statechart
- **That**: Policy as guard-based FSM
- **Synthesis**: Differentiable guard-based policies

### exp_temporal_guards
- **This**: Static guards
- **That**: Time-dependent guards
- **Synthesis**: Differentiable temporal predicates

---

## Future Directions

### 1. Hierarchical Differentiable Statecharts
- Soft parent-child relationships
- Learnable composite state structure
- Differentiable LCA computation

### 2. Differentiable History States
- Soft history storage/retrieval
- Learned history type (shallow/deep)
- Gradient through history restoration

### 3. Parallel Region Learning
- Soft AND-state decomposition
- Learn which states should be parallel
- Differentiable region synchronization

### 4. Symbolic Distillation
- Train differentiable model
- Extract discrete statechart
- Verify equivalence

### 5. Meta-Learning Topology
- Learn topology prior from multiple tasks
- Fast adaptation to new domains
- Transfer learned structure

### 6. Real-Time Constraints
- Differentiable deadline predicates
- Learn timing requirements
- Soft real-time verification

---

## Implementation Notes

### Memory Efficiency
- Soft adjacency: O(n_states² × n_events)
- For large statecharts, use sparse representation
- Prune edges below threshold during training

### Gradient Flow
- Check gradient norms during training
- Temperature too low → vanishing gradients
- Temperature too high → noisy gradients

### Initialization
- State embeddings: small random
- Edge logits: bias toward sparse (-2.0)
- Self-loops: positive bias (+1.0)

### Evaluation
- Extract discrete topology at threshold
- Compare to ground truth (if available)
- Measure edit distance to target

---

## Theoretical Foundation

### Statechart as Differentiable Program

Traditional statechart:
```
σ' = δ(σ, e, Γ)  # discrete transition function
```

Differentiable statechart:
```
p(σ') = ∫ p(σ'|σ,e,Γ) p(σ) dσ  # soft transition
```

With Gumbel-softmax:
```
σ' ~ Gumbel-Softmax(f(σ, e, Γ))
```

### Loss Function

```
L = L_ce + λ₁ L_sparsity + λ₂ L_entropy

L_ce = -Σ log p(σ*_t | σ_{t-1}, e_t)  # cross-entropy
L_sparsity = ||A||₁                    # edge sparsity
L_entropy = -Σ p log p                 # transition entropy
```

---

## References

- Jang et al., "Categorical Reparameterization with Gumbel-Softmax", ICLR 2017
- Maddison et al., "The Concrete Distribution", ICLR 2017
- Kool et al., "Stochastic Beams and Where to Find Them", ICML 2019
- exp_topology_evolution/framework.py (evolutionary baseline)

---

## Files

```
exp_differentiable_statecharts/
├── __init__.py                    # Package exports
├── differentiable_statechart.py   # Core implementation
└── NOTES.md                       # This file
```

## Status

- [x] Gumbel-softmax state selection
- [x] Differentiable guards
- [x] Soft adjacency topology
- [x] Attention-based transitions
- [x] Temperature annealing
- [x] Topology extraction
- [ ] Gradient-based optimization (manual SGD placeholder)
- [ ] Full training with MLX optimizers
- [ ] Benchmark vs evolutionary approach

---

Created: 2026-01-04
