# Proposal: Stochastic & ML Protobuf Extensions for Statecharts

**Status:** Draft
**Author:** tmc
**Date:** 2026-03-24
**Target protos:** `proto/extensions/v1/simulation.proto` (stochastic), new `proto/extensions/v1/ml.proto` (differentiable/ensemble/hybrid)
**Companion doc:** [`ml-memory-and-training.md`](ml-memory-and-training.md) — memory architecture analysis, empirical grounding, Hull KV-Cache semantics

---

## 1. Motivation

The core Harel statechart formalism SC = (S, ρ, ψ, δ, γ, λ, σ₀) is deterministic and synchronous. Real-world ML and agentic systems require stochastic semantics, differentiable execution, ensemble composition, hybrid control, and neural memory — all without modifying the core schema.

| Scope | Proto | Modifies core? |
|-------|-------|---------------|
| Stochastic simulation | `extensions/v1/simulation.proto` (existing) | No — additive fields |
| ML semantics | `extensions/v1/ml.proto` (new) | No — packed via `google.protobuf.Any` |
| RLM delegation | `extensions/v1/ml.proto` (new) | No — packed via `google.protobuf.Any` |
| Dynamic topology | `extensions/v1/ml.proto` (new) | No — packed via `google.protobuf.Any` |

All extensions are packed into `State.extensions` and `Transition.extensions` via `google.protobuf.Any`.

---

## 2. Stochastic Extensions to `simulation.proto`

### 2.1 DistributionType Enum

**Problem:** `Distribution.type` is a bare string — no compile-time safety, no documentation of parameter contracts.

```protobuf
enum DistributionType {
  DISTRIBUTION_TYPE_UNSPECIFIED = 0;

  // δ(x - c): Dirac delta at constant value c.
  // Params: {value: c}
  DISTRIBUTION_TYPE_CONSTANT = 1;

  // f(x) = 1/(b-a) for x ∈ [a,b].
  // Params: {min: a, max: b}
  DISTRIBUTION_TYPE_UNIFORM = 2;

  // f(x) = (1/σ√(2π)) exp(-(x-μ)²/2σ²).
  // Params: {mean: μ, stddev: σ}
  DISTRIBUTION_TYPE_NORMAL = 3;

  // f(x) = λ exp(-λx) for x ≥ 0.
  // Params: {rate: λ}   Mean = 1/λ
  DISTRIBUTION_TYPE_EXPONENTIAL = 4;

  // f(x) = (1/xσ√(2π)) exp(-(ln(x)-μ)²/2σ²).
  // Params: {mu: μ, sigma: σ}
  DISTRIBUTION_TYPE_LOGNORMAL = 5;

  // f(x) = (k/λ)(x/λ)^(k-1) exp(-(x/λ)^k).
  // Params: {shape: k, scale: λ}
  DISTRIBUTION_TYPE_WEIBULL = 6;

  // f(x) = x^(α-1)(1-x)^(β-1) / B(α,β) for x ∈ [0,1].
  // Params: {alpha: α, beta: β}
  DISTRIBUTION_TYPE_BETA = 7;

  // f(x) = (β^α / Γ(α)) x^(α-1) exp(-βx).
  // Params: {shape: α, rate: β}
  DISTRIBUTION_TYPE_GAMMA = 8;

  // P(X=k) = (λ^k e^(-λ)) / k!
  // Params: {rate: λ}
  DISTRIBUTION_TYPE_POISSON = 9;

  // Erlang(k, λ): sum of k iid Exp(λ).
  // Models k sequential phases each with rate λ.
  // Params: {shape: k, rate: λ}
  DISTRIBUTION_TYPE_ERLANG = 10;

  // Empirical distribution from observed samples.
  // CDF constructed via inverse-transform sampling.
  // Samples provided in Distribution.empirical_samples.
  DISTRIBUTION_TYPE_EMPIRICAL = 11;
}
```

### 2.2 Enhanced Distribution Message

```protobuf
message Distribution {
  string type = 1 [deprecated = true];       // Legacy string type
  DistributionType distribution_type = 6;    // Typed enum (new)
  map<string, double> params = 2;            // Named parameters per enum docs
  double min = 3;                            // Truncation lower bound
  double max = 4;                            // Truncation upper bound
  repeated double empirical_samples = 5;     // For EMPIRICAL type
}
```

Wire compatibility: Fields 1-4 unchanged; fields 5-6 are new.

### 2.3 Enhanced TransitionSimConfig

Existing fields 1-8 unchanged. Addition:

```protobuf
message TransitionSimConfig {
  // ... existing fields 1-8 ...

  // Softmax temperature for RL exploration-exploitation tradeoff.
  // P(tᵢ) ∝ exp(Q(s, tᵢ) / τ)
  // τ → 0: greedy (exploit), τ → ∞: uniform (explore).
  // Default 1.0. Only used when a learned policy overrides static weights.
  double temperature = 9;
}
```

Documentation enhancement (no structural change) for existing fields:
- **weight** (field 1): SPN categorical sampling — P(tᵢ | enabled) = wᵢ / Σⱼ wⱼ. Renormalized over enabled transitions when guards disable some (see §5.3).
- **rate** (field 7): CTMC exponential firing — τᵢ ~ Exp(λᵢ), race condition selects t\* = argmin τᵢ, equivalent to P(t\*=tᵢ) = λᵢ / Σⱼ λⱼ by memoryless property.
- **reward** (field 6): MDP immediate reward r(s, a, s') for value iteration / policy gradient.

### 2.4 Enhanced StateSimConfig

Existing fields 1-9 unchanged. Additions:

```protobuf
message StateSimConfig {
  // ... existing fields 1-9 ...

  // Discount factor γ ∈ [0,1] for future rewards.
  // V(s) = r(s) + γ · Σ P(s'|s,a) · V(s')
  // Default 1.0 (undiscounted). Set < 1.0 for infinite-horizon MDPs.
  //
  // HIERARCHICAL COMPOSITION:
  // γ_effective(s) = Π { γ(p) | p ∈ ancestors(s) ∪ {s} }
  // Follows the Options framework [Sutton, Precup & Singh 1999].
  double discount = 10;

  // Observation model for partially observable settings (POMDP).
  ObservationModel observation = 11;
}
```

### 2.5 ObservationModel (new message)

```protobuf
// ObservationModel defines emission probabilities for POMDP formulations.
//
// FORMAL DEFINITION:
// O: S × Z → [0,1] where Z is the observation space.
// For state s: Σ_z O(z|s) = 1.
//
// Enables belief-state planning (point-based value iteration, POMCP)
// over statechart hierarchies.
message ObservationModel {
  // Observation label → emission probability. Must sum to 1.0.
  map<string, double> emissions = 1;

  // Continuous observation noise (additive).
  Distribution noise = 2;
}
```

### 2.6 PayloadGenerator + PayloadGeneratorType

```protobuf
enum PayloadGeneratorType {
  PAYLOAD_GENERATOR_TYPE_UNSPECIFIED = 0;
  PAYLOAD_GENERATOR_TYPE_CONSTANT = 1;       // Fixed payload from params
  PAYLOAD_GENERATOR_TYPE_RANDOM_INT = 2;     // Uniform int in [min, max]
  PAYLOAD_GENERATOR_TYPE_RANDOM_DOUBLE = 3;  // Sample from Distribution
  PAYLOAD_GENERATOR_TYPE_RANDOM_STRING = 4;  // Random string of given length/charset
  PAYLOAD_GENERATOR_TYPE_RANDOM_JSON = 5;    // Random JSON matching schema
  PAYLOAD_GENERATOR_TYPE_ONE_OF = 6;         // Uniform selection from values array
  PAYLOAD_GENERATOR_TYPE_COMPOSITE = 7;      // Compose multiple generators
}

message PayloadGenerator {
  PayloadGeneratorType type = 1;
  google.protobuf.Struct params = 2;
  string json_schema = 3;  // Validation constraint for generated payloads
}
```

Migration for `RandomEvent`: Add `PayloadGenerator generator = 6` alongside deprecated `string payload_generator = 3`.

### 2.7 FailureType, FailureTrigger, ArrivalProcess Enums

```protobuf
// Formal fault taxonomy [Avizienis et al., IEEE TDSC 2004].
enum FailureType {
  FAILURE_TYPE_UNSPECIFIED = 0;
  FAILURE_TYPE_DROP_TRANSITION = 1;   // Omission: transition silently dropped
  FAILURE_TYPE_CORRUPT_CONTEXT = 2;   // Commission: context variables corrupted
  FAILURE_TYPE_EXCEPTION = 3;         // Exception raised, halts current step
  FAILURE_TYPE_CRASH = 4;             // Immediate machine termination
  FAILURE_TYPE_TIMEOUT = 5;           // Infinite blocking (timing fault)
  FAILURE_TYPE_SLOW = 6;              // Extreme delay injection
  FAILURE_TYPE_PARTIAL = 7;           // Partial completion (partial omission)
  FAILURE_TYPE_BYZANTINE = 8;         // Arbitrary incorrect behavior
}

enum FailureTrigger {
  FAILURE_TRIGGER_UNSPECIFIED = 0;
  FAILURE_TRIGGER_RANDOM = 1;         // Bernoulli(failure_rate) each execution
  FAILURE_TRIGGER_AFTER_N = 2;        // After N successful executions
  FAILURE_TRIGGER_CONDITION = 3;      // When CEL expression is true
  FAILURE_TRIGGER_SCHEDULED = 4;      // At specific simulation time
  FAILURE_TRIGGER_PERIODIC = 5;       // Every N steps
}

enum ArrivalProcess {
  ARRIVAL_PROCESS_UNSPECIFIED = 0;
  ARRIVAL_PROCESS_POISSON = 1;        // T ~ Exp(λ), memoryless
  ARRIVAL_PROCESS_UNIFORM = 2;        // T ~ Uniform(a, b)
  ARRIVAL_PROCESS_BURST = 3;          // Periodic bursts
  ARRIVAL_PROCESS_DETERMINISTIC = 4;  // Fixed interval = 1/rate
  ARRIVAL_PROCESS_MMPP = 5;           // Markov-modulated Poisson process
}
```

Migration for `FailureConfig`: Add `FailureType typed_failure = 17` and `FailureTrigger typed_trigger = 18`.
Migration for `RandomEventConfig`: Add `ArrivalProcess typed_arrival_process = 5`.

### 2.8 SimulationMode + StochasticSeedConfig (new messages)

```protobuf
enum SimulationMode {
  SIMULATION_MODE_UNSPECIFIED = 0;
  SIMULATION_MODE_DISCRETE = 1;     // weight-based, rate fields ignored
  SIMULATION_MODE_CONTINUOUS = 2;   // rate-based CTMC race condition
  SIMULATION_MODE_HYBRID = 3;       // per-transition; requires sync protocol
}

message StochasticSeedConfig {
  int64 global_seed = 1;                   // 0 = use current time
  RegionSeedStrategy region_strategy = 2;  // Default: INDEPENDENT
}

enum RegionSeedStrategy {
  REGION_SEED_STRATEGY_UNSPECIFIED = 0;
  REGION_SEED_STRATEGY_INDEPENDENT = 1;  // hash-derived per region
  REGION_SEED_STRATEGY_SHARED = 2;       // single global stream
}

enum DwellResumePolicy {
  DWELL_RESUME_POLICY_UNSPECIFIED = 0;
  DWELL_RESUME_POLICY_RESUME = 1;  // Resume on history re-entry (default)
  DWELL_RESUME_POLICY_RESET = 2;   // Fresh sample even on history re-entry
}
```

### 2.9 Summary of simulation.proto Changes

| Change | Wire-compatible? | Field # |
|--------|-----------------|---------|
| `DistributionType` enum | Yes (new type) | — |
| `Distribution.distribution_type` | Yes | 6 |
| `Distribution.empirical_samples` | Yes | 5 |
| `TransitionSimConfig.temperature` | Yes | 9 |
| `StateSimConfig.discount` | Yes | 10 |
| `StateSimConfig.observation` | Yes | 11 |
| `ObservationModel` message | Yes (new type) | — |
| `PayloadGenerator` message | Yes (new type) | — |
| `PayloadGeneratorType` enum | Yes (new type) | — |
| `RandomEvent.generator` | Yes | 6 |
| `FailureType` enum | Yes (new type) | — |
| `FailureTrigger` enum | Yes (new type) | — |
| `FailureConfig.typed_failure` | Yes | 17 |
| `FailureConfig.typed_trigger` | Yes | 18 |
| `ArrivalProcess` enum | Yes (new type) | — |
| `RandomEventConfig.typed_arrival_process` | Yes | 5 |
| `SimulationMode` enum | Yes (new type) | — |
| `StochasticSeedConfig` message | Yes (new type) | — |
| `RegionSeedStrategy` enum | Yes (new type) | — |
| `DwellResumePolicy` enum | Yes (new type) | — |

---

## 3. ML Semantic Extensions (`ml.proto`, new file)

Each message is packed into `State.extensions` or `Transition.extensions` via `google.protobuf.Any`.

### 3.1 DifferentiableConfig

```protobuf
// DifferentiableConfig enables continuous relaxation of discrete
// statechart execution for end-to-end neural network training.
//
// GUMBEL-SOFTMAX RELAXATION [Jang et al. 2017, Maddison et al. 2017]:
// Instead of hard configuration σ ∈ {0,1}^|S|, we maintain a soft
// configuration π ∈ [0,1]^|S| where:
//
//   πᵢ = exp((log(αᵢ) + gᵢ) / τ) / Σⱼ exp((log(αⱼ) + gⱼ) / τ)
//
// where αᵢ are unnormalized logits, gᵢ ~ Gumbel(0,1), and τ is
// temperature. As τ → 0, π approaches one-hot (discrete).
//
// ANNEALING SCHEDULE:
// τ follows an exponential decay during training:
//   τ(t) = τ_max · (τ_min / τ_max)^(t / T)
//
// STRAIGHT-THROUGH ESTIMATOR [Bengio et al. 2013]:
// Forward pass uses hard argmax; backward pass uses soft gradients.
message DifferentiableConfig {
  bool enabled = 1;
  double temperature = 2;
  double temperature_min = 3;
  double temperature_max = 4;
  double annealing_rate = 5;
  bool straight_through_estimator = 6;

  // Entropy regularization: L' = L - β·H(π)
  double entropy_coefficient = 7;
}
```

### 3.2 AttentionTransitionConfig

```protobuf
// AttentionTransitionConfig replaces fixed transition targets with
// learned query-key-value attention over candidate target states.
//
// ATTENTION SEMANTICS:
//   α_j = softmax(q · kⱼ / √d_k)
//   target = Σ αⱼ · vⱼ          (soft target, training)
//   target = argmax αⱼ           (hard target, inference)
//
// SOFT ADJACENCY MATRIX:
// A ∈ [0,1]^(|S|×|S|) — learnable transition topology.
message AttentionTransitionConfig {
  bool enabled = 1;
  int32 attention_dim = 2;
  int32 num_heads = 3;
  bool hard_attention_inference = 4;
  repeated double adjacency_matrix = 5;  // Flattened, row-major, |S|×|S|
  double sparsity_penalty = 6;           // L1 on adjacency
}
```

### 3.3 NeuralPredicateConfig

```protobuf
// NeuralPredicateConfig replaces hard boolean guards with
// differentiable neural predicates for gradient-based learning.
//
// CONTINUOUS GUARD SEMANTICS:
//   g̃: Context × Event → [0,1]  where g̃ = σ(f_θ(x))
//
// Training: soft (gradient flow). Inference: g̃ > threshold → enabled.
message NeuralPredicateConfig {
  bool enabled = 1;
  string model_id = 2;
  repeated string input_features = 3;
  double threshold = 4;
  repeated int32 hidden_dims = 5;
  string activation = 6;  // "sigmoid", "tanh", "relu"
}
```

### 3.4 EnsembleConfig

```protobuf
// EnsembleConfig defines multi-agent / multi-policy composition.
//
// E = (M₁, ..., Mₙ, C) where Mᵢ are member machines, C is combination.
//
// COMBINATION STRATEGIES:
// - MAJORITY: action = mode({aᵢ})
// - WEIGHTED: action = argmax_a Σ wᵢ · 1(aᵢ = a)
// - MIXTURE_OF_EXPERTS: P(a) = Σ gᵢ(x) · Pᵢ(a|x)
// - ATTENTION: output = Σ αᵢ · oᵢ
//
// Members maintain independent state spaces (see §5.6).
message EnsembleConfig {
  repeated EnsembleMember members = 1;
  CombinationStrategy strategy = 2;
  repeated CommunicationChannel channels = 3;
  DiversityConfig diversity = 4;
}

message EnsembleMember {
  string id = 1;
  string statechart_ref = 2;
  double weight = 3;
  string role = 4;  // "leader", "follower", "specialist", "generalist"
}

enum CombinationStrategy {
  COMBINATION_STRATEGY_UNSPECIFIED = 0;
  COMBINATION_STRATEGY_MAJORITY = 1;
  COMBINATION_STRATEGY_UNANIMOUS = 2;
  COMBINATION_STRATEGY_WEIGHTED = 3;
  COMBINATION_STRATEGY_MIXTURE_OF_EXPERTS = 4;
  COMBINATION_STRATEGY_ATTENTION = 5;
}

message CommunicationChannel {
  string id = 1;
  string type = 2;  // "event_bus", "config_share", "gradient_share", "action_share"
  repeated string participants = 3;
  int32 bandwidth = 4;  // messages per step, 0 = unlimited
}

message DiversityConfig {
  bool track_agreement = 1;
  bool track_entropy = 2;
  bool track_kl_divergence = 3;
  double diversity_bonus_coefficient = 4;  // r' = r + λ_div · diversity
}
```

### 3.5 HybridControlConfig

```protobuf
// HybridControlConfig extends a state to serve as a behavioral mode
// governing a continuous neural controller.
//
// HYBRID AUTOMATA SEMANTICS [Henzinger 1996]:
// H = (Q, X, Init, f, Inv, E, G, R) where:
// - Q: discrete modes (statechart states)
// - X: continuous variables (controller outputs)
// - f: Q → (X → Ẋ): flow function per mode (neural controller)
// - Inv: Q → 2^X: mode invariants
// - G: E → 2^X: guard conditions for transitions
// - R: E → (X → X): reset maps on transition
message HybridControlConfig {
  repeated ContinuousVariable variables = 1;
  string controller_id = 2;
  int32 output_dim = 3;
  double control_frequency = 4;  // Hz
  string invariant_expression = 5;
  map<string, double> reset_on_entry = 6;
}

message ContinuousVariable {
  string name = 1;
  int32 dimension = 2;
  double min_value = 3;
  double max_value = 4;
  string unit = 5;  // "m/s", "N", "rad"
}
```

### 3.6 NeuralMemoryConfig

See [`ml-memory-and-training.md`](ml-memory-and-training.md) for detailed architecture analysis, Hull KV-Cache formal semantics, and empirical grounding.

```protobuf
// NeuralMemoryConfig extends statechart context with differentiable
// memory mechanisms that replace or augment flat key-value stores.
//
// The memory output is concatenated with standard context Γ:
//   Γ' = Γ ∪ {memory_key: memory_read(t)}
message NeuralMemoryConfig {
  NeuralMemoryType type = 1;
  int32 capacity = 2;       // Slots for NTM/attention
  int32 memory_dim = 3;     // Vector dimension per slot
  int32 num_read_heads = 4;
  int32 num_write_heads = 5;
  string context_key = 6;   // Key in Γ where memory output is stored
  int32 attention_window = 7;
  bool persistent = 8;      // Survives machine resets
  HullMemoryConfig hull = 9;
}

message HullMemoryConfig {
  int32 num_hull_layers = 1;           // Nested hulls for top-k
  int32 top_k = 2;                     // Sparse softmax mask width
  int32 num_namespaces = 3;            // Independent hull address spaces
  repeated string namespace_labels = 4;
}

enum NeuralMemoryType {
  NEURAL_MEMORY_TYPE_UNSPECIFIED = 0;
  NEURAL_MEMORY_TYPE_NTM = 1;          // O(n) content-based addressing
  NEURAL_MEMORY_TYPE_ATTENTION = 2;    // O(K) soft attention over history
  NEURAL_MEMORY_TYPE_HTM = 3;          // Sparse Distributed Representations
  NEURAL_MEMORY_TYPE_LSTM_CONTEXT = 4; // O(1) per step
  NEURAL_MEMORY_TYPE_TRANSFORMER = 5;  // Transformer-XL recurrence
  NEURAL_MEMORY_TYPE_SSM = 6;          // S4/Mamba O(n log n)
  NEURAL_MEMORY_TYPE_HULL = 7;         // Convex hull O(log n) exact addressing
}
```

### 3.7 MLProvenance

```protobuf
// MLProvenance tracks the origin of ML-induced statechart elements.
message MLProvenance {
  string model_id = 1;
  string architecture = 2;  // "transformer", "gnn", "rl_policy"
  string dataset_id = 3;
  int64 training_step = 4;
  double confidence = 5;    // [0,1]
  double loss = 6;
  bool verified = 7;
  string verification_notes = 8;
}
```

### 3.8 RecursiveDelegationConfig (RLM)

```protobuf
// RecursiveDelegationConfig enables Recursive Language Model (RLM)
// delegation patterns where agents spawn sub-agents for subtasks.
//
// DELEGATION SEMANTICS:
// An agent A in statechart mode q may delegate a subtask to child A':
//   A'.context = project(A.context, context_keys)
//   A'.statechart = child_statechart_ref
//   result = A'.run(A'.context)
//   A.context[result_key] = aggregate(result, ...)
//
// This prevents context rot by limiting what the child sees,
// and enables parallel hypothesis exploration via AND-state children.
//
// STATECHART INTEGRATION:
// Delegation is modeled as a transition action. The parent enters a
// "waiting" state; on child completion, a synthetic RESULT event fires.
//
// RECURSION DEPTH:
// Bounded by max_depth to prevent unbounded spawning. The effective
// discount at depth d is gamma^d (multiplicative per §5.1).
//
// REFERENCES:
// [Beurer-Kellner 2025] RLMs: Recursive Language Models
// [Sutton 1999] Options framework for temporal abstraction
message RecursiveDelegationConfig {
  // Context keys projected to child (subset of parent Gamma).
  // Empty = project all keys (no isolation).
  repeated string context_keys = 1;

  // Maximum recursion depth across all delegation chains.
  // 0 = no delegation. Typical: 2-4.
  int32 max_depth = 2;

  // Key in parent context where aggregated child result is stored.
  string result_key = 3;

  // Reference to child statechart (label or URI).
  // If empty, child uses a copy of parent's statechart.
  string child_statechart_ref = 4;

  // Maximum steps for child execution (timeout).
  int32 max_child_steps = 5;

  // Maximum concurrent children from a single delegation point.
  int32 max_children = 6;

  // How to aggregate results from multiple children.
  DelegationAggregation aggregation = 7;

  // Context projection mode.
  ContextProjectionMode projection_mode = 8;

  // Delegation cost added to transition cost for parsimony pressure.
  // During RL training, this penalizes unnecessary delegation.
  double delegation_cost = 9;
}

// DelegationAggregation defines how child results are combined.
enum DelegationAggregation {
  DELEGATION_AGGREGATION_UNSPECIFIED = 0;

  // Take the single result (first to complete or highest confidence).
  DELEGATION_AGGREGATION_FIRST = 1;

  // Mean of child result embeddings.
  DELEGATION_AGGREGATION_MEAN = 2;

  // Learned attention over child results.
  DELEGATION_AGGREGATION_ATTENTION = 3;

  // Majority vote (for discrete action outputs).
  DELEGATION_AGGREGATION_VOTE = 4;

  // Mixture of experts: P(a) = sum_i g_i(x) * P_i(a|x).
  DELEGATION_AGGREGATION_MIXTURE_OF_EXPERTS = 5;

  // Concatenate all child results (for downstream processing).
  DELEGATION_AGGREGATION_CONCAT = 6;
}

// ContextProjectionMode controls how parent context is narrowed.
enum ContextProjectionMode {
  CONTEXT_PROJECTION_MODE_UNSPECIFIED = 0;

  // Key-based: only context_keys are passed.
  CONTEXT_PROJECTION_MODE_KEYS = 1;

  // Learned linear projection: child_ctx = W * parent_ctx.
  CONTEXT_PROJECTION_MODE_LINEAR = 2;

  // Gated projection: child_ctx = sigma(gate) * proj(parent_ctx).
  CONTEXT_PROJECTION_MODE_GATED = 3;

  // Attention-based: child attends over parent context entries.
  CONTEXT_PROJECTION_MODE_ATTENTION = 4;
}
```

### 3.9 DynamicTopologyConfig

```protobuf
// DynamicTopologyConfig enables runtime construction and mutation
// of statechart topology, covering four methodologies:
//
// 1. GRADIENT: Differentiable topology learning via soft adjacency
//    matrices and Gumbel-Softmax (§3.1-3.2). The topology is a
//    learnable parameter optimized end-to-end.
//
// 2. SAE_EXTRACTION: Mechanistic extraction via Sparse Autoencoders.
//    SAE features map to states; co-activation patterns induce
//    hierarchy (superstate/substate); activation thresholds become
//    guards.
//
// 3. EVOLUTIONARY: NSGA-II multi-objective evolution treating the
//    statechart as an evolvable genome. Structural mutations
//    (add_state, reparent, toggle_type, add_transition) co-evolve
//    with guards, priorities, and actions. Fitness balances
//    correctness, minimality, and interpretability.
//
// 4. META_LLM: LLM/RLM-based induction. Models emit special tokens
//    (SC:LOAD, SC:PUSH, SC:POP) to mutate the active statechart at
//    runtime. Offline trace-to-statechart induction reconstructs
//    topology from (state, event, action, next_state) logs.
//
// STORAGE: Pack into Statechart-level metadata (not per-state).
//
// REFERENCES:
// [Jang 2017] Gumbel-Softmax for gradient topology
// [Bricken 2023] Sparse Autoencoders for mechanistic interpretability
// [Deb 2002] NSGA-II multi-objective evolutionary algorithm
// [Beurer-Kellner 2025] RLM meta-constrained generation
message DynamicTopologyConfig {
  // Which methodology governs topology construction.
  TopologyConstructionMethod method = 1;

  // For GRADIENT: differentiable config reference.
  DifferentiableConfig differentiable = 2;

  // For SAE_EXTRACTION: SAE model reference and threshold.
  SAEExtractionConfig sae = 3;

  // For EVOLUTIONARY: evolution parameters.
  EvolutionaryTopologyConfig evolutionary = 4;

  // For META_LLM: LLM induction parameters.
  MetaLLMTopologyConfig meta_llm = 5;

  // Whether topology mutations are validated against well-formedness
  // rules before application. Default: true.
  bool validate_mutations = 6;

  // Maximum number of topology mutations per step.
  int32 max_mutations_per_step = 7;
}

enum TopologyConstructionMethod {
  TOPOLOGY_CONSTRUCTION_METHOD_UNSPECIFIED = 0;
  TOPOLOGY_CONSTRUCTION_METHOD_GRADIENT = 1;
  TOPOLOGY_CONSTRUCTION_METHOD_SAE_EXTRACTION = 2;
  TOPOLOGY_CONSTRUCTION_METHOD_EVOLUTIONARY = 3;
  TOPOLOGY_CONSTRUCTION_METHOD_META_LLM = 4;
}

// SAEExtractionConfig for mechanistic topology extraction.
message SAEExtractionConfig {
  // SAE model identifier.
  string sae_model_id = 1;

  // Activation threshold for state induction.
  // Features above this threshold become state activations.
  double activation_threshold = 2;

  // Minimum co-activation frequency to induce hierarchy.
  double hierarchy_threshold = 3;

  // Guard synthesis mode.
  GuardSynthesisMode guard_synthesis = 4;
}

enum GuardSynthesisMode {
  GUARD_SYNTHESIS_MODE_UNSPECIFIED = 0;

  // Boolean expressions evolved from activation patterns.
  GUARD_SYNTHESIS_MODE_BOOLEAN = 1;

  // Continuous neural predicates from feature scores.
  GUARD_SYNTHESIS_MODE_NEURAL = 2;

  // Threshold on single SAE feature activation.
  GUARD_SYNTHESIS_MODE_THRESHOLD = 3;
}

// EvolutionaryTopologyConfig for NSGA-II evolution.
message EvolutionaryTopologyConfig {
  // Population size.
  int32 population_size = 1;

  // Number of generations.
  int32 n_generations = 2;

  // Mutation rates for structural operators.
  double add_state_rate = 3;
  double remove_state_rate = 4;
  double reparent_rate = 5;
  double toggle_type_rate = 6;    // OR <-> AND
  double add_transition_rate = 7;
  double remove_transition_rate = 8;

  // Fitness objectives (Pareto front).
  repeated string objectives = 9;  // e.g., ["correctness", "minimality", "interpretability"]

  // Parsimony pressure coefficient.
  double parsimony_coefficient = 10;
}

// MetaLLMTopologyConfig for LLM/RLM topology induction.
message MetaLLMTopologyConfig {
  // LLM model identifier for topology generation.
  string model_id = 1;

  // Special tokens that trigger topology mutations.
  // Default: ["SC:LOAD", "SC:PUSH", "SC:POP", "SC:ADD_STATE", "SC:ADD_TRANS"]
  repeated string mutation_tokens = 2;

  // Whether to use offline trace induction.
  bool trace_induction = 3;

  // Maximum context window for trace analysis.
  int32 trace_window = 4;

  // Statechart stack depth limit for SC:PUSH/SC:POP.
  int32 max_stack_depth = 5;
}
```

### 3.10 Structural Mutation Operators + Live Patching

```protobuf
// StructuralMutation represents a single granular topology change.
//
// Used by both EVOLUTIONARY (batch of mutations per generation) and
// META_LLM (mutations emitted at runtime). Each mutation is validated
// against Harel well-formedness rules before application when
// DynamicTopologyConfig.validate_mutations is true.
//
// DESIGN PRINCIPLE:
// Mutations are fine-grained and composable. Complex topology changes
// are composed from sequences of atomic mutations. This ensures each
// mutation can be independently validated, rolled back, and attributed
// (via MLProvenance).
message StructuralMutation {
  // Which element and operation.
  MutationType type = 1;

  // Target element label (state, transition, or event label).
  string target_label = 2;

  // For state mutations.
  StateMutation state_mutation = 3;

  // For transition mutations.
  TransitionMutation transition_mutation = 4;

  // For guard mutations.
  GuardMutation guard_mutation = 5;

  // For action mutations.
  ActionMutation action_mutation = 6;

  // Provenance: who/what created this mutation.
  MLProvenance provenance = 7;
}

enum MutationType {
  MUTATION_TYPE_UNSPECIFIED = 0;

  // State topology mutations.
  MUTATION_TYPE_ADD_STATE = 1;         // Insert new state under parent
  MUTATION_TYPE_REMOVE_STATE = 2;      // Remove state (orphan handling required)
  MUTATION_TYPE_REPARENT_STATE = 3;    // Move state to new parent
  MUTATION_TYPE_TOGGLE_TYPE = 4;       // Switch OR <-> AND semantics
  MUTATION_TYPE_SET_HISTORY = 5;       // Change history type (none/shallow/deep)

  // Transition mutations.
  MUTATION_TYPE_ADD_TRANSITION = 6;    // Insert new transition
  MUTATION_TYPE_REMOVE_TRANSITION = 7; // Remove existing transition
  MUTATION_TYPE_RETARGET = 8;          // Change transition target
  MUTATION_TYPE_RESOURCE = 9;          // Change transition source

  // Guard mutations.
  MUTATION_TYPE_MUTATE_GUARD = 10;     // Modify guard expression AST
  MUTATION_TYPE_SET_GUARD = 11;        // Replace guard entirely

  // Action mutations.
  MUTATION_TYPE_ADD_ACTION = 12;       // Add action to transition/state
  MUTATION_TYPE_REMOVE_ACTION = 13;    // Remove action
  MUTATION_TYPE_MUTATE_ACTION = 14;    // Modify action parameters
}

// StateMutation for add/remove/reparent/toggle operations.
message StateMutation {
  // Parent label for ADD_STATE.
  string parent_label = 1;

  // New label for ADD_STATE.
  string new_label = 2;

  // State type for ADD_STATE (BASIC/NORMAL/PARALLEL).
  int32 state_type = 3;

  // Whether the new state is initial.
  bool is_initial = 4;

  // New parent label for REPARENT_STATE.
  string new_parent_label = 5;

  // History type for SET_HISTORY: "none", "shallow", "deep".
  string history_type = 6;
}

// TransitionMutation for add/remove/retarget operations.
message TransitionMutation {
  // Source state labels.
  repeated string from_labels = 1;

  // Target state labels.
  repeated string to_labels = 2;

  // Event label that triggers this transition.
  string event_label = 3;

  // New target for RETARGET.
  repeated string new_to_labels = 4;

  // New source for RESOURCE.
  repeated string new_from_labels = 5;

  // Priority for transition conflict resolution.
  int32 priority = 6;
}

// GuardMutation for AST-level guard evolution.
message GuardMutation {
  // AST operation type.
  GuardMutationOp op = 1;

  // New guard expression (for SET_GUARD or leaf replacement).
  string expression = 2;

  // AST node index for targeted mutations (0 = root).
  int32 ast_node_index = 3;
}

enum GuardMutationOp {
  GUARD_MUTATION_OP_UNSPECIFIED = 0;
  GUARD_MUTATION_OP_REPLACE_OPERATOR = 1;   // Change comparison op
  GUARD_MUTATION_OP_SWAP_OPERANDS = 2;      // Swap LHS/RHS
  GUARD_MUTATION_OP_GROW = 3;               // Add conjunction/disjunction
  GUARD_MUTATION_OP_SHRINK = 4;             // Remove subexpression
  GUARD_MUTATION_OP_NEGATE = 5;             // Wrap in NOT
  GUARD_MUTATION_OP_REPLACE_VARIABLE = 6;   // Change evaluated variable
  GUARD_MUTATION_OP_REPLACE_THRESHOLD = 7;  // Change comparison constant
}

// ActionMutation for runtime action modification.
message ActionMutation {
  // Action type: "SET", "INCREMENT", "DECREMENT", "TOGGLE", "EMIT".
  string action_type = 1;

  // Context variable targeted by the action.
  string variable = 2;

  // Value for SET or delta for INCREMENT.
  string value = 3;
}

// PatchStatechartRequest enables live patching of a running machine.
//
// An external controller or overseeing agent sends a sequence of
// structural mutations to a live machine instance. Mutations are
// applied atomically: all succeed or all roll back.
//
// USE CASES:
// - LLM agent dynamically adding substates during execution
// - Supervisor agent wiring new guard conditions into a child
// - Adaptive controller adding recovery transitions on fault detection
//
// SAFETY:
// - validate_mutations (default true) checks well-formedness
// - Mutations that would orphan the current configuration are rejected
// - A rollback log is maintained for undo
message PatchStatechartRequest {
  // Machine instance identifier.
  string machine_id = 1;

  // Ordered list of mutations to apply atomically.
  repeated StructuralMutation mutations = 2;

  // Whether to validate well-formedness before applying.
  bool validate = 3;

  // Whether to checkpoint before patching (for rollback).
  bool checkpoint = 4;

  // Reason for the patch (for audit trail).
  string reason = 5;
}

message PatchStatechartResponse {
  // Whether the patch was applied successfully.
  bool success = 1;

  // Error details if failed.
  string error = 2;

  // Number of mutations applied.
  int32 mutations_applied = 3;

  // Checkpoint ID for rollback (if checkpoint was requested).
  string checkpoint_id = 4;

  // Well-formedness violations (if any).
  repeated string validation_errors = 5;
}

// StatechartStackEntry represents one frame in the SC:PUSH/SC:POP stack.
//
// When a meta-LLM emits SC:PUSH, the current statechart is saved and
// a new one becomes active. SC:POP restores the previous frame.
// This enables temporary constraint contexts for subtask execution.
message StatechartStackEntry {
  // The saved statechart reference.
  string statechart_ref = 1;

  // The saved configuration at push time.
  repeated string active_states = 2;

  // The saved context snapshot.
  map<string, string> context_snapshot = 3;

  // Push timestamp (for timeout enforcement).
  int64 push_timestamp = 4;

  // Maximum duration before automatic pop (0 = no timeout).
  int64 timeout_ms = 5;
}
```

### 3.11 PolicyOptimizationConfig (GRPO/SDPO)

```protobuf
// PolicyOptimizationConfig specifies the RL training regime for
// statechart topology and policy learning.
//
// Extends the basic policy gradient (§4.3) with group-relative
// advantages and self-distillation for stable topology evolution.
//
// GRPO [Shao et al. 2024]:
// Computes advantages relative to a group of sampled trajectories:
//   A_t = (R_t - mean(R_group)) / std(R_group)
// Eliminates the need for a learned critic/value network.
//
// SDPO [Kim et al. 2024]:
// Augments GRPO with an EMA teacher for self-distillation:
//   L_SDPO = L_GRPO + alpha * D_KL(pi_theta || pi_EMA)
// Enables hindsight learning: failed executions reprompted as
// correct for synthetic tasks.
//
// STORAGE: Pack into Statechart-level metadata.
message PolicyOptimizationConfig {
  // Which RL algorithm to use.
  PolicyAlgorithm algorithm = 1;

  // PPO/GRPO clipping epsilon. Default: 0.2.
  double clip_epsilon = 2;

  // KL penalty coefficient against reference policy. Default: 0.01.
  double kl_coefficient = 3;

  // Group size for GRPO advantage estimation.
  int32 group_size = 4;

  // SDPO-specific: EMA teacher parameters.
  EMATeacherConfig ema_teacher = 5;

  // Hindsight relabeling config.
  HindsightConfig hindsight = 6;

  // Entropy bonus coefficient. Default: 0.01.
  double entropy_coefficient = 7;

  // Learning rate for policy updates.
  double learning_rate = 8;

  // Maximum gradient norm for clipping.
  double max_grad_norm = 9;
}

enum PolicyAlgorithm {
  POLICY_ALGORITHM_UNSPECIFIED = 0;
  POLICY_ALGORITHM_REINFORCE = 1;      // Vanilla policy gradient
  POLICY_ALGORITHM_PPO = 2;            // Proximal Policy Optimization
  POLICY_ALGORITHM_GRPO = 3;           // Group Relative Policy Optimization
  POLICY_ALGORITHM_SDPO = 4;           // Self-Distilled Policy Optimization
}

// EMATeacherConfig for self-distillation in SDPO.
message EMATeacherConfig {
  // EMA decay rate: theta_EMA = (1 - mu) * theta_EMA + mu * theta.
  // Default: 0.01 (slow teacher update).
  double ema_decay = 1;

  // KL divergence weight between student and EMA teacher.
  double distillation_weight = 2;

  // Update teacher every N training steps.
  int32 update_interval = 3;
}

// HindsightConfig for relabeling failed executions.
//
// HINDSIGHT LEARNING:
// When a statechart execution fails to achieve the target,
// the trajectory is reprompted: "given this execution trace,
// what task would this solve?" The relabeled (trace, synthetic_task)
// pair becomes a positive training example.
//
// This massively increases sample efficiency for topology learning
// without requiring external supervision.
message HindsightConfig {
  // Whether hindsight relabeling is enabled.
  bool enabled = 1;

  // Fraction of failed trajectories to relabel. Default: 0.5.
  double relabel_fraction = 2;

  // Maximum relabeled examples per batch.
  int32 max_relabeled_per_batch = 3;

  // Model used for relabeling (if different from main policy).
  string relabeler_model_id = 4;
}
```

### 3.12 HybridMemoryConfig (Hull + SSM)

```protobuf
// HybridMemoryConfig composes multiple memory backends for
// different aspects of statechart execution.
//
// DESIGN RATIONALE:
// Nested Hulls and SSMs serve complementary roles:
// - Hulls: O(log n) exact retrieval for deterministic topology
//   execution and Harel Deep History (H*) restoration. The hull
//   returns the exact argmax — no soft attention smearing.
// - SSMs: O(n log n) long-horizon temporal modeling. Mamba/S4
//   handle continuous-time event loops over thousands of steps
//   via block-diagonal A matrices encoding hierarchy independence.
//
// The hybrid approach uses Hulls for structural/topological memory
// (which states were visited, exact configurations) and SSMs for
// temporal/behavioral memory (patterns over long execution traces).
//
// REFERENCES:
// [Gu 2023] Mamba: Linear-Time Sequence Modeling
// [Barber 1996] Quickhull algorithm for convex hulls
message HybridMemoryConfig {
  // Hull backend for exact structural memory.
  NeuralMemoryConfig hull_memory = 1;

  // SSM backend for long-horizon temporal memory.
  NeuralMemoryConfig ssm_memory = 2;

  // How to combine outputs from both backends.
  MemoryCombination combination = 3;

  // Routing network: learns which queries go to hull vs. SSM.
  bool learned_routing = 4;

  // Routing threshold: queries above this go to hull (exact),
  // below go to SSM (approximate). Only used when learned_routing=false.
  double routing_threshold = 5;
}

enum MemoryCombination {
  MEMORY_COMBINATION_UNSPECIFIED = 0;
  MEMORY_COMBINATION_CONCAT = 1;     // Concatenate and project
  MEMORY_COMBINATION_ADD = 2;        // Additive fusion
  MEMORY_COMBINATION_GATE = 3;       // Learned gating: g * hull + (1-g) * ssm
  MEMORY_COMBINATION_ATTENTION = 4;  // Cross-attention between backends
}
```

---

## 4. Formal Semantics Reference

### 4.1 Statechart → Stochastic Model Correspondence

| Harel Concept | SPN | CTMC | MDP/RL | POMDP |
|---------------|-----|------|--------|-------|
| State s ∈ S | Place p | State s | State s | State s |
| Transition t ∈ δ | Timed transition | Rate λₜ | Action a | Action a |
| Configuration σ | Marking M | State vector | State s | Belief b |
| weight(t) | Weight wₜ | — | π(a\|s) prior | π(a\|b) prior |
| rate(t) | Firing rate | Q[s,s'] | — | — |
| reward(s,t) | — | — | r(s,a,s') | r(s,a,s') |
| discount(s) | — | — | γ | γ |
| dwell_dist | Sojourn dist | Exp(λ) | — | — |
| observation(s) | — | — | — | O(o\|s) |
| temperature | — | — | τ (softmax) | τ (softmax) |

### 4.2 Generator Matrix Construction (CTMC)

```
Q[sᵢ, sⱼ] = Σ { rate(t) | t ∈ δ, src(t) ∩ config(sᵢ) ≠ ∅, tgt(t) yields config(sⱼ) }
Q[sᵢ, sᵢ] = -Σⱼ≠ᵢ Q[sᵢ, sⱼ]
```

Stationary distribution π satisfies πQ = 0, Σπᵢ = 1.

### 4.3 MDP Value Function

```
V*(s) = reward(s) + γ(s) · max_a Σ_s' P(s'|s,a) · V*(s')
P(s'|s,a) = weight(t_{s,a,s'}) / Σ weight(t_{s,a,·})
```

### 4.4 Differentiable Execution (Gumbel-Softmax)

```
πᵢ = exp((log(αᵢ) + gᵢ) / τ) / Σⱼ exp((log(αⱼ) + gⱼ) / τ)
where gᵢ ~ Gumbel(0,1), τ is temperature
```

### 4.5 Hybrid Automaton Semantics

```
H = (Q, X, Init, f, Inv, E, G, R)
- Discrete transitions: σ →ᵉ σ'  (standard Harel)
- Continuous evolution: ẋ = f_q(x)  while x ∈ Inv(q)
- Mode switch: when x ∈ G(e), apply x' = R(e, x)
```

### 4.6 SMDP / Options Framework Correspondence

Statechart composite states correspond exactly to Sutton's "options" in the Semi-Markov Decision Process (SMDP) framework [Sutton, Precup & Singh 1999]:

```
Option ω = (I, π, β) maps to:
  I  = initiation set    → guard-enabled entry transitions to composite state
  π  = internal policy   → sub-statechart execution within composite
  β  = termination cond  → exit transitions from composite state (possibly guarded)
```

| SMDP Concept | Statechart Equivalent |
|---|---|
| Option ω | Composite state (NORMAL or PARALLEL) |
| Initiation set I | Guards on entry transitions |
| Internal policy π | Sub-statechart topology + guards |
| Termination β | Exit transitions (completion events) |
| Option duration τ | Dwell time under `TimingConstraints` |
| Intra-option discount | `StateSimConfig.discount` (multiplicative, §5.1) |

**Temporal abstraction:** A composite state acts as a temporally extended action. The agent reasons at the composite level (select option) while the sub-statechart handles fine-grained execution. This is the formal bridge between statechart engineering and hierarchical RL.

**Interruption semantics:** Unlike standard options which run to completion, Harel statecharts allow preemption via higher-priority transitions. This maps to the "interrupt" extension of options [Sutton et al. 1999, §5].

```
V_ω(s) = E[r_1 + γr_2 + ... + γ^(k-1)r_k + γ^k V(s') | s, ω]
        = reward(composite) + γ_effective · V(exit_state)

where k is option duration, γ_effective = Π{γ(p) | p ∈ ancestors}
```

### 4.7 GRPO / SDPO Policy Optimization

For RL training over statechart topologies, two policy optimization methods extend the basic policy gradient (§4.3):

**Group Relative Policy Optimization (GRPO):**

```
L_GRPO(θ) = -E_q[min(r_t(θ) · A_t, clip(r_t(θ), 1-ε, 1+ε) · A_t)]
             - β · D_KL(π_θ || π_ref)

where:
  r_t(θ) = π_θ(a_t|s_t) / π_old(a_t|s_t)   (importance ratio)
  A_t = (R_t - mean(R_group)) / std(R_group)  (group-relative advantage)
```

GRPO computes advantages relative to a group of sampled trajectories rather than a learned value function, eliminating the need for a critic network. This is efficient for statechart topology optimization where the "action" is a structural mutation or transition selection.

**Self-Distilled Policy Optimization (SDPO):**

```
L_SDPO(θ) = L_GRPO(θ) + α · D_KL(π_θ || π_EMA)

where π_EMA is an Exponential Moving Average of past policies:
  θ_EMA ← (1 - μ) · θ_EMA + μ · θ
```

SDPO augments GRPO with an EMA teacher model. The student minimizes KL divergence against soft targets from the teacher, enabling:
- **Hindsight learning:** Failed statechart executions are reprompted and relabeled as correct for synthetic tasks, massively increasing sample efficiency.
- **Stable topology evolution:** The EMA teacher prevents catastrophic forgetting of previously discovered topology structures.

---

## 5. Design Decisions (Resolved)

### 5.1 Hierarchical Discount — Multiplicative

γ_effective(s) = Π { γ(p) | p ∈ ancestors(s) ∪ {s} }

Follows the Options framework [Sutton, Precup & Singh 1999]. Confirmed by sc-ml differentiable LCA experiments.

### 5.2 Parallel Region PRNG — Independent Streams

seed(region_i) = hash(global_seed, region_i.label)

Prevents artificial correlation; preserves reproducibility via deterministic seeding.

### 5.3 Guard-Weight Renormalization — Mandatory

P(tᵢ | enabled(tᵢ)) = wᵢ / Σ { wⱼ | guard(tⱼ) = true }

Maintains Σ P(tᵢ) = 1 after guard shunting. Mathematically required.

### 5.4 CTMC/Discrete Mixing — Global SimulationMode

Requires `SimulationMode` enum on `SimulationScenario`. HYBRID mode uses sync protocol: rate > 0 transitions fire asynchronously; rate = 0 transitions batch at discrete sync points.

### 5.5 History + Stochastic Dwell — Resume

On history re-entry: remaining = T - t_exit (resume interrupted sojourn). On normal re-entry: fresh sample T' ~ dwell_distribution. Controlled by `DwellResumePolicy`.

### 5.6 Ensemble Topology — Independent with Channels

Members maintain independent state spaces linked by explicit `CommunicationChannel` messages. Confirmed by sc-ml cross-model steering transfer experiments.

### 5.7 SSM Memory — Included

`NEURAL_MEMORY_TYPE_SSM` (value 6). Validated by sc-ml exp_s4_statechart and exp_mamba_sc_executor.

### 5.8 Hull Memory — Included with HullMemoryConfig

`NEURAL_MEMORY_TYPE_HULL` (value 7) with `HullMemoryConfig` for nested hulls, top-k, and namespace isolation. See [`ml-memory-and-training.md`](ml-memory-and-training.md) for full Hull formal semantics.

### 5.9 RLM Delegation — Extension-only via Transition Actions

Delegation is modeled as a transition *action*, not a new transition type. The parent statechart enters a "waiting" state; child completion fires a synthetic RESULT event. This keeps delegation composable with all existing transition semantics (guards, weights, rates).

Context projection modes (KEYS, LINEAR, GATED, ATTENTION) trade off simplicity vs. expressiveness. GATED is the default — confirmed by exp_neurosymbolic_statecharts showing gated projection outperforms key-based subsetting for context rot prevention.

Delegation cost (field 9) adds to TransitionSimConfig.cost, creating RL parsimony pressure that the model learns to balance against delegation benefit.

### 5.10 Dynamic Topology — Four Methods, One Config

All four construction methods (gradient, SAE, evolutionary, meta-LLM) share a single `DynamicTopologyConfig` entry point rather than separate per-method extensions. This reflects the reality that a system may switch methods during its lifecycle (e.g., gradient-based during training, SAE-based for interpretability analysis, evolutionary for deployment optimization).

Validation of topology mutations (field 6, default true) ensures that runtime-constructed topologies satisfy Harel well-formedness rules before application. This prevents the system from generating invalid statecharts.

### 5.11 Structural Mutations — Atomic + Validated

Individual mutations are fine-grained (add_state, mutate_guard, retarget_transition) and composed into atomic batches via `PatchStatechartRequest`. This design:

1. **Composability:** Complex topology changes decompose into sequences of validated atomic operations.
2. **Safety:** Each mutation is validated against Harel well-formedness rules. Mutations that would orphan the current configuration (e.g., removing the active state) are rejected.
3. **Rollback:** Optional checkpointing before patch application enables undo on failure.
4. **Provenance:** Each `StructuralMutation` carries `MLProvenance` for attribution.

Guard mutation operates at the AST level (GROW, SHRINK, REPLACE_OPERATOR, NEGATE) rather than string replacement, enabling structured evolution of boolean expressions. This is validated by exp_topology_evolution's co-evolution of guard logic with topology structure.

The SC:PUSH/SC:POP stack enables temporary constraint contexts — an LLM agent can push a specialized sub-grammar for a subtask (e.g., JSON schema generation), operate within it, and pop back. Stack depth is bounded by `MetaLLMTopologyConfig.max_stack_depth`.

### 5.12 Remaining Open

1. Proto file organization: single `ml.proto` initially, split when complexity warrants.
2. Neural predicate training config: inference-only in proto; training hyperparameters in separate tooling.
3. Meta-LLM token vocabulary: the mutation_tokens field provides extensibility, but a standard set should be documented.
4. Delegation across language boundaries: when parent is Go and child is Python, context serialization protocol TBD.

---

## 6. Non-Goals

- Modifying `statecharts.proto` — all extensions use `google.protobuf.Any` packing
- Runtime/interpreter implementation — schema only
- Training infrastructure — proto captures model references, not training loops
- Specific neural architectures — proto references model IDs; details in model registries

---

## 7. Implementation Plan

### Phase 1: Stochastic (simulation.proto)
- Add `DistributionType` enum and `Distribution.distribution_type` field
- Add `TransitionSimConfig.temperature`
- Add `StateSimConfig.discount` and `StateSimConfig.observation`
- Add `ObservationModel`, `PayloadGenerator`, `PayloadGeneratorType`
- Add `FailureType`, `FailureTrigger`, `ArrivalProcess` enums
- Add `SimulationMode`, `StochasticSeedConfig`, `RegionSeedStrategy`, `DwellResumePolicy`
- Run `make generate` and verify wire compatibility

### Phase 2: Differentiable + RL (ml.proto)
- Define `DifferentiableConfig`, `NeuralPredicateConfig`, `AttentionTransitionConfig`
- Define `MLProvenance`
- Write examples showing packing into State/Transition extensions

### Phase 3: Ensemble + Hybrid + Memory (ml.proto)
- Define `EnsembleConfig` and supporting messages
- Define `HybridControlConfig` and `ContinuousVariable`
- Define `NeuralMemoryConfig`, `HullMemoryConfig`, `NeuralMemoryType`

### Phase 4: RLM Delegation + Dynamic Topology (ml.proto)
- Define `RecursiveDelegationConfig`, `DelegationAggregation`, `ContextProjectionMode`
- Define `DynamicTopologyConfig` and all construction method configs
- Define `SAEExtractionConfig`, `EvolutionaryTopologyConfig`, `MetaLLMTopologyConfig`
- Write examples showing delegation via transition actions

### Phase 4b: Policy Optimization + Hybrid Memory (ml.proto)
- Define `PolicyOptimizationConfig`, `PolicyAlgorithm`, `EMATeacherConfig`, `HindsightConfig`
- Define `HybridMemoryConfig`, `MemoryCombination`
- **Experiment code:** `exp_neurosymbolic_statecharts/training.py` (GRPOTrainer, SDPOTrainer)
- **Experiment code:** `exp_neurosymbolic_statecharts/hybrid_memory.py` (SSMMemory, HybridMemory)

### Phase 5: Integration Testing
- Python SDK: stochastic simulation with CTMC semantics
- Go semantics: extend `GuardEvaluator` to support neural predicates
- ML experiments: integrate with `exp_neurosymbolic_statecharts`
- Delegation tests: recursive child spawning with depth limits
- Dynamic topology: gradient-based topology discovery benchmarks
- GRPO/SDPO: verify group-relative advantage computation and EMA teacher tracking
- Hybrid memory: verify Hull + SSM combination modes (CONCAT/ADD/GATE/ATTENTION)

---

## 8. References

- [H87] D. Harel, "Statecharts: A visual formalism for complex systems," 1987
- [HN96] D. Harel and A. Naamad, "The STATEMATE semantics of statecharts," 1996
- [AD94] R. Alur and D. Dill, "A theory of timed automata," 1994
- [Marsan 1990] M. Ajmone Marsan, "Stochastic Petri Nets: An elementary introduction," 1990
- [Stewart 1994] W.J. Stewart, "Introduction to the Numerical Solution of Markov Chains," 1994
- [PRISM] M. Kwiatkowska et al., "PRISM 4.0: Verification of Probabilistic Real-time Systems," 2011
- [Avizienis 2004] A. Avizienis et al., "Basic Concepts and Taxonomy of Dependable and Secure Computing," IEEE TDSC, 2004
- [Sutton 1999] R. Sutton, D. Precup, S. Singh, "Between MDPs and Semi-MDPs: A Framework for Temporal Abstraction in RL," 1999
- [Jang 2017] E. Jang et al., "Categorical Reparameterization with Gumbel-Softmax," ICLR 2017
- [Maddison 2017] C. Maddison et al., "The Concrete Distribution," ICLR 2017
- [Bengio 2013] Y. Bengio et al., "Estimating or Propagating Gradients Through Stochastic Neurons," 2013
- [Henzinger 1996] T. Henzinger, "The Theory of Hybrid Automata," LICS 1996
- [Kuncheva 2003] L. Kuncheva and C. Whitaker, "Measures of Diversity in Classifier Ensembles," 2003
- [Puterman 1994] M. Puterman, "Markov Decision Processes," 1994
- [Kleinrock 1975] L. Kleinrock, "Queueing Systems, Volume I: Theory," 1975
- [Beurer-Kellner 2025] L. Beurer-Kellner et al., "Recursive Language Models," 2025
- [Deb 2002] K. Deb et al., "A Fast and Elitist Multiobjective Genetic Algorithm: NSGA-II," IEEE TEC, 2002
- [Bricken 2023] T. Bricken et al., "Towards Monosemanticity: Decomposing Language Models With Dictionary Learning," Anthropic, 2023
- [Shao 2024] Z. Shao et al., "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models," 2024 (GRPO)
- [Kim 2024] G. Kim et al., "SDPO: Don't Use Your Data All at Once," 2024 (Self-Distilled Policy Optimization)
- [Gu 2023] A. Gu and T. Dao, "Mamba: Linear-Time Sequence Modeling with Selective State Spaces," 2023
