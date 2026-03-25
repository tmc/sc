# ML Memory Architectures & Training Semantics for Statecharts

**Status:** Draft
**Author:** tmc
**Date:** 2026-03-24
**Companion doc:** [`simulation-stochastic-proposal.md`](simulation-stochastic-proposal.md) — protobuf extension definitions

---

## 1. Overview

This document covers the ML-specific deep dives that inform the protobuf extensions defined in the companion proposal. It is organized into three sections:

1. **Neural memory architectures** — detailed analysis of each `NeuralMemoryType`, with emphasis on the Hull KV-Cache
2. **Empirical grounding** — evidence from sc-ml experiments, mlx-go-computer, ARC-AGI research, and mlx-go-rlm that validates design decisions
3. **Training integration** — how differentiable statechart execution connects to end-to-end ML training pipelines

---

## 2. Neural Memory Architecture Analysis

The `NeuralMemoryConfig` message (§3.6 of the companion proposal) supports seven memory architectures. Each addresses a different tradeoff in the statechart context:

| Type | Complexity | Exact? | Differentiable? | Best For |
|------|-----------|--------|----------------|----------|
| NTM | O(n) | Soft | Yes | Content-addressed recall |
| ATTENTION | O(K) | Soft | Yes | Recent history windowing |
| HTM | O(n) | Hard | No | Temporal pattern detection |
| LSTM_CONTEXT | O(1) | N/A | Yes | Compressed state tracking |
| TRANSFORMER | O(n²) | Soft | Yes | Full cross-attention |
| SSM | O(n log n) | Soft | Yes | Long-sequence temporal |
| HULL | O(log n) | Exact | Yes (via sparse SDPA) | Exact history + topology |

### 2.1 Neural Turing Machine (NTM)

**Architecture:** Content-based + location-based addressing over an external memory matrix M ∈ ℝ^(N×D) [Graves et al. 2014].

**Read:** w_t = softmax(K(k_t, M)) where K is cosine similarity. Read vector r_t = w_t^T · M.

**Write:** M_t = M_{t-1} · (1 - w_t · e_t^T) + w_t · a_t^T where e_t is erase vector, a_t is add vector.

**Statechart relevance:** General-purpose memory for storing and retrieving arbitrary context patterns. Useful when the agent needs to recall information by content (e.g., "what was the guard condition when I last visited state X?").

**Limitation:** O(n) per read/write. Soft addressing means retrieved values are interpolations, not exact recalls.

### 2.2 Soft Attention Over History

**Architecture:** Maintains a sliding window of the past K configurations. At each step, the model attends over the window:

```
memory(t) = Σᵢ αᵢ · σ(t-i)    where αᵢ = softmax(q · kᵢ / √d)
```

**Statechart relevance:** Efficient for tasks where recent history determines the next transition (e.g., debouncing, pattern detection over the last K states). The attention window K is a tunable parameter.

**Limitation:** Fixed window size. Cannot recall events beyond K steps. Soft attention smears across similar configurations.

### 2.3 Hierarchical Temporal Memory (HTM)

**Architecture:** Sparse Distributed Representations (SDR) where each state configuration is encoded as a sparse binary vector [Hawkins & Ahmad 2016]. Temporal patterns are learned via Hebbian-like connection updates.

**Statechart relevance:** Natural fit for detecting temporal sequences of state configurations. The sparsity ensures that overlapping patterns don't interfere. Useful for anomaly detection — "this sequence of states has never occurred before."

**Limitation:** Not natively differentiable. Requires custom backward pass or REINFORCE-style gradient estimation.

### 2.4 LSTM Context

**Architecture:** Standard LSTM cell where the hidden state serves as the statechart context:

```
h_t, c_t = LSTM(h_{t-1}, c_{t-1}, [config_t, event_t])
```

**Statechart relevance:** Simplest differentiable memory. O(1) per step. The LSTM cell learns to compress execution history into a fixed-size vector. Guards and actions can reference the LSTM hidden state via the `context_key` field.

**Limitation:** Fixed capacity. Information from early steps degrades over time (vanishing gradient). No explicit content-based addressing.

### 2.5 Transformer-XL Memory

**Architecture:** Segment-level recurrence with relative positional encoding [Dai et al. 2019]. Past segment hidden states are cached and attended to during the current segment, extending effective context length.

**Statechart relevance:** Full cross-attention over execution history without a fixed window. Relative positional encoding handles variable-length execution traces. The segment boundary naturally aligns with statechart step boundaries.

**Limitation:** O(n²) attention cost. Memory grows linearly with execution history.

### 2.6 State Space Models (S4/Mamba)

**Architecture:** Structured state spaces [Gu et al. 2022] parameterized as:

```
x'(t) = Ax(t) + Bu(t)
y(t) = Cx(t) + Du(t)
```

Discretized via zero-order hold. Mamba [Gu & Dao 2023] adds input-dependent selection for adaptive computation.

**Statechart relevance:** Efficient O(n log n) sequence modeling via FFT-based convolution. Handles long execution traces without quadratic attention cost. The continuous-time formulation naturally aligns with CTMC-annotated statecharts.

**Empirical validation:** sc-ml exp_s4_statechart and exp_mamba_sc_executor demonstrate SSMs outperform attention-based approaches for long statechart execution traces (>1000 steps).

### 2.7 Hull KV-Cache Memory

**Architecture:** Convex hull geometric addressing over a KV-cache [mlx-go-computer].

This is the most novel memory type and warrants detailed treatment.

#### 2.7.1 Geometric Encoding

Each cached key at position j is encoded as coordinates on a 2D upper convex hull:

```
point(j) = (2j, -j²)
```

The model generates query directions via standard W_Q weight matrices:

```
query(i) = (i, 1)     (produced by W_Q · h_src)
```

The dot product between query(i) and point(j) is:

```
dot(i, j) = 2ij - j² = -(j - i)²  + i²
```

This is strictly unimodal in j with maximum at j = i — meaning the query for position i exactly retrieves position i.

#### 2.7.2 O(log n) Supporting Point Query

Because the dot product is unimodal on the convex hull, a binary search over edge slopes finds the supporting point:

1. Maintain the upper convex hull of all cached (2j, -j²) points
2. For query direction (i, 1), find the hull edge where the slope transitions from > i to < i
3. Binary search over hull edges: O(log n)
4. The supporting point is the exact argmax — no approximation

This replaces O(n) linear scan with O(log n) geometric lookup while returning the exact result.

#### 2.7.3 Nested Hulls for Top-K

For k-sparse softmax (needed for probabilistic branching and Gumbel-Softmax):

1. **Outer hull:** Standard convex hull over all points
2. **Inner hulls:** Points deleted from the outer hull (due to convexity violations) are pushed to inner hull layers
3. **Query:** Find supporting point on each hull layer; maintain max-heap across layers
4. **Result:** Exact top-k retrieval in O(k + log n) time

This enables efficient probabilistic routing over large branching spaces without full O(n) softmax.

#### 2.7.4 Differentiable Training via Sparse SDPA

While hull lookups are deterministic at inference, training requires gradient flow:

1. Construct sparse attention mask M initialized to -1e9 (effectively -∞)
2. Unmask only the top-k indices retrieved from hull query: M[top_k_indices] = 0
3. Pass sparse mask into standard Scaled Dot-Product Attention:
   ```
   SDPA(Q, K, V, mask=M)  →  softmax((QK^T + M) / √d) · V
   ```
4. Because SDPA is a standard differentiable operation, the MLX/PyTorch autodiff framework traces gradients through the VJP back to W_Q, W_K, W_V

The hull acts as a *selection oracle* that chooses which entries the differentiable attention mechanism operates over. Gradients flow through the attention computation; the hull provides the sparse structure.

#### 2.7.5 Statechart Applications

**Exact history restoration.** H/H\* pseudostates require restoring precise hierarchical configurations. Nested state configurations can be mapped to geometric coordinates for instantaneous, deterministic context restoration in O(log n) rather than O(n) soft attention.

**Deterministic topology execution.** The hull's SupportingPoint query returns an exact argmax. Wiring this into the forward pass enables statechart topologies compiled into transformer weights to enforce strict, deterministic transition paths without external logit masking.

**Memory isolation.** Separate hull namespaces (parameterized via `HullMemoryConfig.num_namespaces`) maintain isolation between state tracking and data context, preventing interference between the statechart's topology state and its computation data.

**Probabilistic branching.** Nested hulls enable k-sparse Gumbel-Softmax over large topological branching spaces. A max-heap over candidates across all hull layers produces the precise top-k indices needed for sparse softmax masks.

#### 2.7.6 HullMemoryConfig Parameters

| Field | Type | Purpose |
|-------|------|---------|
| `num_hull_layers` | int32 | 1 = argmax only; k > 1 = nested hulls for top-k |
| `top_k` | int32 | Sparse softmax mask width during training |
| `num_namespaces` | int32 | Independent address spaces for isolation |
| `namespace_labels` | repeated string | Human-readable namespace names |

---

## 3. Empirical Grounding

### 3.1 sc-ml Experiments

948 Python/MD files across 60+ experiments. Key findings mapped to proto design:

| Experiment | Validates | Key Finding |
|-----------|-----------|-------------|
| exp_differentiable_statecharts / exp_differentiable_lca | §3.1 DifferentiableConfig | Gumbel-Softmax annealing works. LCA variant confirms multiplicative hierarchical discount. |
| exp_sae_feature_steering / exp_circuit_steering_fusion | §3.3 NeuralPredicateConfig | SAE-extracted features serve as continuous guards at inference time. |
| exp_grammar_constrained_ceiling / exp_sc_guided_decoding | Guard semantics | Guards can gate token emission during autoregressive generation. |
| exp_transfer_learning / exp_cross_model_steering_transfer | §3.4 EnsembleConfig | Independent member topologies + explicit channels outperform shared state spaces. |
| exp_s4_statechart / exp_mamba_sc_executor | §3.6 SSM memory | SSMs outperform attention for long traces (>1000 steps). |
| exp_c_rl_policy / exp_learnable_policies / exp_sc_lora_grpo | §3.7 MLProvenance | RL confidence scores are meaningful provenance signals. |

### 3.2 mlx-go-computer (Percepta)

A vanilla transformer (d_model=36, 18 heads, 7 layers) that executes programs via algebraic weight construction.

**Orthogonal Subspace Routing.** The 36-dim residual stream is partitioned: opcode bus (0-15), state tracking (16-19), ALU bus (20-27), memory routing (28-35). This is a concrete instance of HybridControlConfig — discrete modes (opcodes) govern continuous controllers (ALU operations).

**Gated FFN Conditional Dispatch.** relu(gate) \* val with +100/-50 thresholds implements NeuralPredicateConfig in pure weight space — guards compiled into FFN neurons rather than learned.

**HullKVCache.** Demonstrates O(log n) exact addressing via convex hull geometry. Standard attention is O(n); hull-backed max is O(log n) for monotonic state variables. See §2.7 for full analysis.

**Softmax vs Hardmax boundary.** The central engineering challenge. Standard softmax smears attention; exact addressing requires hardmax. This directly informs DifferentiableConfig — the `straight_through_estimator` flag is essential, not optional.

### 3.3 ARC-AGI Research

27 cloned repos from ARC Prize competition:

**Poetiq-ai (SOTA 82-86%).** Multi-model ensemble with dynamic instruction refinement. Validates EnsembleConfig — MIXTURE_OF_EXPERTS with per-task gating matches their approach.

**Samsung TinyRecursiveModels (7M params, 45% ARC-1).** Hierarchical multi-timescale reasoning: slow planning + fast computation. Validates HybridControlConfig — discrete behavioral modes at different timescales.

**Sapientinc HRM (27M params).** Brain-inspired hierarchical RNN. Near-perfect on Sudoku/mazes with 1K samples. Validates that NeuralMemoryConfig hierarchy matters more than capacity.

**Symbolica arcgentica.** REPL-agent approach using Claude. Validates EnsembleConfig communication channels — the agent loop is an ensemble of generate→test→refine members.

**sc-ml/arcagi3.** Statechart-based ARC solver using adapter/controller/planner/frame/memory architecture. Direct integration point for this proposal.

### 3.4 mlx-go-rlm (Reasoning LM with REPL)

Starlark REPL embedded in LLM decode loop.

**Hybrid discrete-continuous.** The REPL state machine (parse→execute→observe→reason) is a behavioral mode controller where each mode gates different neural generation strategies. Validates HybridControlConfig.

**Neural memory.** The REPL's execution trace serves as working memory that influences subsequent generation. Validates NeuralMemoryConfig ATTENTION type — soft attention over execution history.

---

## 4. Training Integration Patterns

### 4.1 End-to-End Differentiable Statechart Training

The combination of DifferentiableConfig (Gumbel-Softmax), NeuralPredicateConfig (soft guards), and AttentionTransitionConfig (soft topology) enables full end-to-end training:

```
Loss = L_task + β·H(π) + λ_sparse·||A||₁
         ↑          ↑              ↑
    task loss   entropy reg   adjacency sparsity
```

**Forward pass:**
1. Soft configuration π via Gumbel-Softmax (temperature τ)
2. Soft guard evaluation g̃ = σ(f_θ(context))
3. Transition selection: P(tᵢ) = g̃ᵢ · wᵢ · Aᵢⱼ / Z
4. Soft target: next_config = Σ P(tᵢ) · target_config(tᵢ)
5. Memory update: Γ' = Γ ∪ {memory_key: memory_read(t)}

**Backward pass:**
- Gumbel-Softmax: reparameterization gradient through π
- Straight-through estimator: hard argmax forward, soft gradient backward
- Neural predicates: standard backprop through sigmoid MLP
- Hull memory: sparse SDPA mask → VJP through W_Q, W_K, W_V
- Adjacency matrix: gradient through A with L1 sparsity penalty

### 4.2 Temperature Annealing Schedule

```
τ(t) = τ_max · (τ_min / τ_max)^(t / T)
```

Typical values from sc-ml experiments:
- τ_max = 5.0 (early exploration)
- τ_min = 0.1 (near-discrete)
- T = total training steps

### 4.3 RL Integration

For reinforcement learning over statechart topologies:

```
State:   s = (config, context, memory)
Action:  a = enabled_transition_index
Reward:  r = StateSimConfig.reward + TransitionSimConfig.reward
Discount: γ = Π { StateSimConfig.discount(p) | p ∈ ancestors(s) }
```

**Value function:**
```
V*(s) = reward(s) + γ(s) · max_a Σ_s' P(s'|s,a) · V*(s')
```

**Policy gradient:**
```
∇_θ J(θ) = E[Σ_t ∇_θ log π_θ(aₜ|sₜ) · (Rₜ - b(sₜ))]
```

where π_θ is parameterized by the TransitionSimConfig weights (optionally with temperature scaling).

### 4.4 POMDP Belief Updates

When ObservationModel is configured:

```
b'(s') = η · O(o|s') · Σ_s P(s'|s,a) · b(s)
```

where b is the belief state, O is the observation model, and η is a normalizing constant.

### 4.5 Ensemble Training

For MIXTURE_OF_EXPERTS combination:

```
P_ensemble(a|x) = Σᵢ gᵢ(x) · Pᵢ(a|x)
```

where gᵢ(x) = softmax(W_gate · x) is the gating network. The diversity bonus:

```
r' = r + λ_div · H(votes)
```

encourages member specialization.

---

## 5. Dynamic Topology Construction

Four methodologies for runtime topology construction, each corresponding to a `TopologyConstructionMethod` enum value in the companion proposal.

### 5.1 Gradient-Based (GRADIENT)

The topology is a learnable parameter optimized via gradient descent. Soft adjacency matrix A[i,j,e] in [0,1] parameterizes edge existence probabilities. Gumbel-Softmax annealing (tau: 5.0 -> 0.1) transitions from exploration to discrete crystallization. L1 sparsity regularization prevents dense, uninterpretable graphs.

**Experiment:** `exp_neurosymbolic_statecharts` demonstrates gradient-based topology discovery on cyclic and branching patterns. The TopologySearcher class extracts discrete statecharts from learned soft adjacency matrices via thresholding.

### 5.2 SAE Extraction (SAE_EXTRACTION)

Sparse Autoencoder features map to states via activation thresholding. Co-activation patterns induce hierarchy: if feature F3 only fires when F1 is active, F3 is a substate of F1. Guard synthesis monitors activation patterns and evolves boolean expressions from positive/negative examples.

**Experiment:** `exp_sae_statechart` and `exp_circuit_steering_fusion` demonstrate SAE-extracted features serving as continuous guards. The SAEExtractionConfig.hierarchy_threshold controls how aggressively co-activation is interpreted as superstate-substate relationships.

### 5.3 Evolutionary (EVOLUTIONARY)

NSGA-II treats the entire statechart as an evolvable genome. Structural mutations (add_state, reparent, toggle_type between OR/AND, add_transition) co-evolve with guards, priorities, and actions within a UnifiedGenome. Fitness balances correctness, minimality, and interpretability on a Pareto front.

**Experiment:** `exp_topology_evolution` demonstrates autonomous discovery of Harel's hierarchical and concurrent structures for game environments (93.3% legal move prediction for Tic-Tac-Toe).

### 5.4 Meta-LLM Induction (META_LLM)

LLMs emit special tokens that mutate the active statechart at runtime. SC:LOAD switches the governing grammar; SC:PUSH/SC:POP manage a statechart constraint stack. Offline trace-to-statechart induction reconstructs topology from (state, event, action, next_state) logs via CoT scratchpad reasoning.

**Experiment:** `exp_sc_guided_decoding` and `exp_grammar_constrained_ceiling` demonstrate constrained decoding via statechart grammar. `exp_neurosymbolic_statecharts/constrained_synthesis.py` achieves 100% syntactic validity by construction.

### 5.5 RLM Recursive Delegation

Recursive Language Model delegation enables agents to spawn sub-agents with projected context. The delegation decision is differentiable — the model learns when subtasks benefit from delegation. Gated context projection (child_ctx = sigma(gate) * proj(parent_ctx)) prevents context rot.

**Experiment:** `exp_neurosymbolic_statecharts/recursive_delegation.py` implements the full RLM pipeline with DelegationDecider, ContextProjection, ChildAgent, and ResultAggregator. Delegation parsimony is enforced via an RL cost penalty.

---

## 6. References

- [Graves 2014] A. Graves, G. Wayne, I. Danihelka, "Neural Turing Machines," 2014
- [Hawkins 2016] J. Hawkins and S. Ahmad, "Why Neurons Have Thousands of Synapses: A Theory of Sequence Memory in Neocortex," 2016
- [Dai 2019] Z. Dai et al., "Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context," ACL 2019
- [Gu 2022] A. Gu et al., "Efficiently Modeling Long Sequences with Structured State Spaces," ICLR 2022
- [Gu 2023] A. Gu and T. Dao, "Mamba: Linear-Time Sequence Modeling with Selective State Spaces," 2023
- [Jang 2017] E. Jang et al., "Categorical Reparameterization with Gumbel-Softmax," ICLR 2017
- [Maddison 2017] C. Maddison et al., "The Concrete Distribution," ICLR 2017
- [Bengio 2013] Y. Bengio et al., "Estimating or Propagating Gradients Through Stochastic Neurons," 2013
- [Sutton 1999] R. Sutton, D. Precup, S. Singh, "Between MDPs and Semi-MDPs," Artificial Intelligence, 1999
- [Kuncheva 2003] L. Kuncheva and C. Whitaker, "Measures of Diversity in Classifier Ensembles," Machine Learning, 2003
- [Henzinger 1996] T. Henzinger, "The Theory of Hybrid Automata," LICS 1996
- [Beurer-Kellner 2025] L. Beurer-Kellner et al., "Recursive Language Models," 2025
- [Deb 2002] K. Deb et al., "NSGA-II: A Fast Multiobjective Evolutionary Algorithm," IEEE TEC, 2002
- [Bricken 2023] T. Bricken et al., "Towards Monosemanticity," Anthropic, 2023
