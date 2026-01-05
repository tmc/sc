# SAE-Grown Statecharts

## The Big Idea

**Sparse Autoencoders (SAEs) + Differentiable Statecharts = Interpretable Neuro-Symbolic Systems**

Instead of:
- Dense, polysemantic state vectors (opaque)
- Hand-coded state machines (brittle)

We get:
- Sparse, monosemantic state features (interpretable)
- Automatically discovered state structure (adaptive)

## Background: SAEs as FSA Discoverers

Anthropic's mechanistic interpretability work (2023-2025) revealed that SAE features naturally form FSA-like circuits:

> "We found features that tracked bracket depth, forming an implicit finite state machine"
> — Towards Monosemanticity (2023)

> "Features activate in patterns reminiscent of state transitions"
> — Scaling Monosemanticity on Claude 3 Sonnet (2024)

The 2025 literature explicitly extracts FSAs from transformers using SAE features as the state basis.

## Our Contribution

We flip the direction: instead of post-hoc FSA extraction, we **train the statechart module with an SAE bottleneck**, so states are BORN interpretable.

```
Traditional:    LM hidden → dense state vec → opaque transitions
Our approach:   LM hidden → SAE → sparse features → interpretable states
```

## Implementation Approaches

### 1. Post-hoc SAE (easiest)
```python
# Train statechart normally
module = DifferentiableStatechart(dim=256)
train(module, data)

# Then train SAE on state vectors
state_vecs = collect_states(module, data)
sae = TopKSAE(k=32)
train(sae, state_vecs)

# Extract FSA from SAE feature patterns
chart = extract_fsa(sae, state_vecs)
```

### 2. SAE as State Bottleneck (this experiment)
```python
# The state representation IS the SAE latent
class SAEStatechartModule(nn.Module):
    def __init__(self):
        self.sae = TopKSAE(expansion=16, k=32)

    def forward(self, hidden):
        # State = top-k SAE features
        state, indices = self.sae.encode(hidden)
        # state is sparse, interpretable
        return self.sae.decode(state) + hidden
```

### 3. Dynamic SAE Growth (future)
```python
# Reserved features for new state discovery
dead_features = sae.get_dead_features()

# When a dead feature fires strongly → new state discovered
if max_activation(dead_features) > threshold:
    new_state = create_state(triggered_feature)
    chart.add_node(new_state)
```

## Key Files

| File | Purpose |
|------|---------|
| `sae_state_module.py` | Core SAE + statechart integration |
| `train_sae_statechart.py` | Training loop with stability/sharpness losses |
| `code_gen_sae.py` | Application to code generation constraints |

## Training Objectives

1. **Reconstruction**: SAE must faithfully encode/decode
2. **Sparsity**: Exactly k features active (via TopK)
3. **Stability**: Penalize rapid feature flickering
4. **Sharpness**: Encourage clean on/off, not gradual transitions

## State Discovery Process

```
Input sequence → SAE activations → Pattern analysis → Discovered states

1. Stable patterns: Same features active for N steps → "state"
2. Transitions: Feature set changes → "transition"
3. Hierarchy: Feature A only fires when B is on → A is substate of B
4. Parallel: Features C,D independent → orthogonal regions
```

## Connection to Harel Semantics

| Harel Concept | SAE Realization |
|---------------|-----------------|
| State | Stable feature co-activation pattern |
| Transition | Feature activation change |
| Superstate | Feature that gates other features |
| History | Stored activation pattern from exit |
| Parallel region | Orthogonal feature clusters |

## Expected Discoveries

For code generation, we expect SAE features corresponding to:

| Feature Type | Example Features |
|--------------|------------------|
| Syntax | `in_function`, `in_loop`, `in_parens`, `in_string` |
| Semantic | `defining_class`, `calling_method`, `assigning_var` |
| Context | `indent_level_2`, `async_context`, `try_block` |

These emerge automatically from training on code LM hidden states!

## Research Implications

1. **Interpretability**: Every state is a set of monosemantic features
2. **Verifiability**: Can inspect exactly why a transition occurred
3. **Transferability**: Discovered states generalize across models
4. **Debuggability**: When generation fails, can trace to specific features

## References

- Anthropic, "Towards Monosemanticity" (2023)
- Anthropic, "Scaling Monosemanticity" (2024)
- "Finite State Automata Inside Transformers" (2025)
- "SAE Features as Program States" (2025)

## Running the Experiments

```bash
cd /Users/tmc/go/src/github.com/tmc/sc/ml

# Basic demo
.venv/bin/python3 experiments/exp_sae_statechart/sae_state_module.py

# Full training
.venv/bin/python3 experiments/exp_sae_statechart/train_sae_statechart.py

# Code generation application
.venv/bin/python3 experiments/exp_sae_statechart/code_gen_sae.py
```

## TODO

- [ ] Integrate with real LM (Llama, Mistral)
- [ ] Train on actual code corpus
- [ ] Compare discovered states to hand-coded grammar
- [ ] Implement dynamic state growth from dead features
- [ ] Parallel region detection via feature independence analysis
