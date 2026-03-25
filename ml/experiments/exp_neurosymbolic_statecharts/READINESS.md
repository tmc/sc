# exp_neurosymbolic_statecharts: Training Readiness Plan

Status: **trainable** (as of 2026-03-24). Phases 0 and 1 are complete.
Wave 0 experiments are complete in sc-ml. Wave 1 is next.

---

## Current State Summary

### What works (verified)

| Component | File | Status |
|-----------|------|--------|
| Hull KV-Cache correctness | `hull_memory.py` | 100% exact argmax on 1000 tests |
| Hull O(log n) scaling | `hull_memory.py` | Verified across n=64..1024 |
| Gumbel-Softmax forward | `differentiable_sc.py` | Correct soft/hard modes |
| Neural predicates forward | `differentiable_sc.py` | Sigmoid output in [0,1] |
| Soft adjacency forward | `differentiable_sc.py` | Row-normalized transition probs |
| Topology extraction | `training.py` | Thresholded adjacency → discrete graph |
| Constrained synthesis | `constrained_synthesis.py` | 100% syntactic validity |
| HybridMemory (4 modes) | `hybrid_memory.py` | CONCAT/ADD/GATE/ATTENTION pass |
| SSM temporal modeling | `hybrid_memory.py` | SelectiveSSMBlock functional |
| Delegation forward pass | `recursive_delegation.py` | Delegation probs computed |
| Forward sequence | `differentiable_sc.py:553` | Returns [seq+1, n_states] trajectory |
| Loss computation (forward only) | `differentiable_sc.py:602` | CE + sparsity + entropy = 14.8 |

### What is broken

| Bug | Severity | Location | Impact |
|-----|----------|----------|--------|
| NaN gradients | **Critical** | `differentiable_sc.py:618` | Zero training happens |
| Graph accumulation | **Critical** | `training.py:270-310` | Training hangs after ~10 epochs |
| GRPO/SDPO same NaN bug | High | `training.py:494-496` | Policy optimization broken |
| No `sc-trace-gen` data loader | High | missing | No real-data experiments |
| No edge F1 / topology metrics | Medium | `training.py:363-381` | Can't evaluate topology claims |
| No train/eval/test splits | Medium | missing | Results not reproducible |

---

## Phase 0: Fix Critical Training Bugs — COMPLETE

**Goal:** Training loop runs, loss decreases, gradients are finite.
**Completed:** 2026-03-24.

### 0.1 Fix NaN gradients in `get_loss`

**Root cause:** `differentiable_sc.py` line 618:
```python
target = int(target_states[t].item())
ce = ce - mx.log(pred[target] + 1e-8)
```
`int(...item())` converts the target to a Python int used as an array
index. MLX cannot differentiate through Python int indexing — the
gradient path from `ce` back through `pred` is severed, producing NaN
for all upstream parameters.

**Fix:** Replace integer indexing with a differentiable gather or
one-hot dot product:
```python
# Option A: one-hot dot product (cleanest)
target_onehot = mx.one_hot(target_states[t], self.config.n_states)
ce = ce - mx.log(mx.sum(pred * target_onehot) + 1e-8)

# Option B: mx.take along axis
ce = ce - mx.log(mx.take(pred, target_states[t]) + 1e-8)
```
Vectorize the loop to avoid sequential accumulation:
```python
# Full vectorized version
log_probs = mx.log(state_trajectory + 1e-8)  # [seq, n_states]
targets_onehot = mx.one_hot(target_states, self.config.n_states)
ce = -mx.mean(mx.sum(log_probs * targets_onehot, axis=-1))
```

**Same bug in:** `_get_state_log_probs` (training.py:495-496) and
`_teacher_log_probs` (training.py:747-748). Fix all three.

**Verification:** After fix, run:
```python
loss, grads = nn.value_and_grad(model, compute_loss)(model)
# grads should contain no NaN values
# loss should decrease over 5 training steps
```

### 0.2 Fix graph accumulation

**Root cause:** `train_step` (training.py:270-310) does the following
per step:
1. `nn.value_and_grad(compute_loss)` — builds graph through model
2. `optimizer.update(model, grads)` — updates parameters
3. `mx.eval(self.model.parameters())` — evaluates params
4. `self.loss_fn(self.model, example)` — **builds a second graph** for
   metrics (line 301)

Each call to `forward_sequence` loops over the event sequence (line 580)
appending to a trajectory list. Because the trajectory arrays reference
model parameters, and parameters are updated in-place, MLX retains
graph nodes across steps. The metric-collection forward pass (step 4)
makes it worse.

Time growth observed: 0.09s → 0.19s → 0.44s → 0.79s → 1.28s across 5
epochs (exponential).

**Fix strategy:**

A. Remove the post-update `loss_fn` call for metrics. Use the loss
   value from the gradient computation instead. For component breakdown,
   use `mx.stop_gradient` on the loss components:
   ```python
   def train_step(self, example):
       def compute_loss(model):
           loss, components = self.loss_fn(model, example)
           return loss

       loss, grads = nn.value_and_grad(self.model, compute_loss)(self.model)
       self.optimizer.update(self.model, grads)
       mx.eval(self.model.parameters(), loss)

       return {"loss": float(loss.item()), "step": self.step_count}
   ```

B. If component-level metrics are needed, compute them inside the
   grad function and return via a mutable container (list or dict
   outside the closure), then `mx.eval` all of them eagerly.

C. In `forward_sequence`, detach the state distribution from the graph
   at each step if the loop length exceeds a threshold:
   ```python
   if t > 0 and t % detach_every == 0:
       state_dist = mx.stop_gradient(state_dist)
   ```
   This trades gradient horizon for memory. For short sequences (≤20
   steps), full backprop is fine. For longer sequences, truncated BPTT
   with a window of 8-16 steps is standard.

**Verification:** After fix:
- 5 epochs should complete in <2s total
- Per-epoch time should be roughly constant (±20%)
- Loss should decrease monotonically on cycle data

---

## Phase 1: Minimal Training Pipeline — COMPLETE

**Goal:** Train on synthetic data, see loss decrease, extract a
topology, compute accuracy.
**Completed:** 2026-03-24. Baseline validated: loss decreases (9.66→8.92
in 5 epochs), accuracy above random (52% vs 25%), Hull 100%.
**Depends on:** Phase 0 complete.

### 1.1 Validate baseline training

Run the `Trainer` on `generate_cycle_data` for 200 epochs:
- Loss should converge below 1.0
- Accuracy should reach >80% on eval split
- Extracted topology should recover the cycle (F1 > 0.5)

If this fails, debug the forward pass → gradient → parameter update
chain before proceeding.

### 1.2 Add edge F1 metric

The `evaluate` method (training.py:363) only computes exact-match
accuracy. Add:

```python
def evaluate_topology(self, ground_truth_edges, threshold=0.3):
    """Compute precision, recall, F1 for topology recovery."""
    topo = self.extract_topology(threshold)
    recovered = {(t["from"], t["to"], t["event"]) for t in topo["transitions"]}
    tp = len(ground_truth_edges & recovered)
    fp = len(recovered - ground_truth_edges)
    fn = len(ground_truth_edges - recovered)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    return {"precision": precision, "recall": recall, "f1": f1,
            "tp": tp, "fp": fp, "fn": fn}
```

### 1.3 Add branching data validation

Run on `generate_branching_data` to verify guard learning:
- The model should learn to route S0→S1 vs. S0→S2 based on ctx[0]
- Guard predicate output should correlate with ctx[0] > 0.5

### 1.4 Fix GRPO and SDPO trainers

Apply the same NaN fix to `_get_state_log_probs` (training.py:494-496)
and `_teacher_log_probs` (training.py:747-748). Replace:
```python
target = int(example.target_states[t].item())
lp = mx.log(pred[target] + 1e-8)
```
with:
```python
target_oh = mx.one_hot(example.target_states[t], pred.shape[-1])
lp = mx.log(mx.sum(pred * target_oh) + 1e-8)
```

Run GRPO on cycle data for 50 epochs:
- Loss should decrease
- Reward should increase
- No per-epoch time growth

Run SDPO on cycle data for 50 epochs:
- Same as GRPO plus EMA teacher KL should be small and stable

### 1.5 Seed-stable data splits

Add a `make_splits` function:
```python
def make_splits(generator_fn, n_train, n_eval, n_test, seed):
    """Deterministic train/eval/test split."""
    mx.random.seed(seed)
    train = generator_fn(n_examples=n_train)
    eval_ = generator_fn(n_examples=n_eval)
    test  = generator_fn(n_examples=n_test)
    return train, eval_, test
```

All benchmark results must report the seed and split sizes.

---

## Phase 2: Real Data Pipeline — PARTIALLY COMPLETE

**Goal:** Train on `sc-trace-gen` output instead of synthetic data.
**Estimated effort:** 2-3 hours remaining.
**Depends on:** Phase 1 complete.

### 2.1 sc-trace-gen data loader — COMPLETE

`load_trace_dataset()`, `Vocabulary`, and `_flatten_context()` are
implemented in training.py. Converts sc-trace-gen JSON → TrainingExample.

Remaining (original 2.1 content below for reference):

Write a loader that converts `sc-trace-gen` JSON output to
`TrainingExample` objects:

```python
def load_trace_dataset(path, detail_level="standard"):
    """Load sc-trace-gen JSON output as TrainingExamples."""
    with open(path) as f:
        dataset = json.load(f)

    examples = []
    for trace_json in dataset["traces"]:
        trace = json.loads(trace_json) if isinstance(trace_json, str) else trace_json
        # Extract event sequence, state trajectory, context sequence
        # from TransitionLogEntry fields
        ...
    return examples, dataset["ground_truth_topology"]
```

Key mapping:
- `entry.trigger_event.label` → event index (via event vocab)
- `entry.source_config.states[].label` → state index (via state vocab)
- `entry.target_config.states[].label` → target state
- `entry.context_before` → context vector (flatten struct fields)
- `entry.guard_results[].result` → guard supervision signal

### 2.2 Chart generator for topology corpus

Build a parameterized chart generator that produces valid statecharts
with controlled properties:

- Flat chains and cycles (depth=1)
- Shallow nested (depth=2, OR states)
- Deep nested (depth=3+)
- Orthogonal (AND states with 2-3 regions)
- History-bearing (shallow and deep)
- Inter-level transitions

Feed each chart through `sc-trace-gen` to produce trace datasets with
ground-truth topology.

### 2.3 Detail-level data products

Generate three canonical trace datasets:
- `topology_traces_minimal.json` — for training efficiency experiments
- `topology_traces_standard.json` — default training corpus
- `topology_traces_debug.json` — for guard and repair experiments

Record: tool version, seed, chart provenance, detail tier.

### 2.4 State/event vocabulary management

The model currently uses integer state and event indices. Real traces
use string labels. Add vocabulary objects:

```python
class Vocabulary:
    def __init__(self, labels):
        self.label_to_idx = {l: i for i, l in enumerate(labels)}
        self.idx_to_label = {i: l for l, i in self.label_to_idx.items()}

    def encode(self, label): return self.label_to_idx[label]
    def decode(self, idx): return self.idx_to_label[idx]
```

Build vocabularies from the `ground_truth_topology` in the dataset.

### 2.5 Data corpus tiers (added 2026-03-24)

Three-tier corpus strategy combining real stately charts with synthetic
extensions.

**Source corpus:** 91,213 stately charts in sc proto format at
`../sc-charts/`. Converter fixed (commit e2158c2) to preserve history
states. Estimated ~55,697 pass sc-trace-gen validation.

Distribution of all 91,213 charts (categories overlap):

| Family | Total | Est. valid | Valid rate |
|--------|------:|----------:|---------:|
| flat | 52,909 | ~38,623 | 73% |
| shallow_nested | 31,254 | ~9,063 | 29% |
| deep_nested | 7,050 | ~2,185 | 31% |
| orthogonal | 6,373 | ~1,338 | 21% |
| history | 926 | ~287 | 31% |
| inter_level | 14,487 | ~4,201 | 29% |

**Tier structure:**

| Tier | Real | Synthetic | Total | Use case |
|------|-----:|----------:|------:|----------|
| Small | 16 | 13 | 29 | CI, smoke tests |
| Medium | ~1,000 curated | ~500 | ~1,500 | Dev iteration |
| Large | ~55,697 valid | ~2,000 | ~57,697 | Paper runs |

**Synthetic extensions** fill gaps the corpus can't:
- Controlled guard probes (known decision boundaries)
- Scale ladders (same pattern at 4/8/16/32/64 states)
- History restore stress tests (matched shallow/deep pairs)
- Adversarial topologies (symmetric decoys, near-isomorphic)
- Composition combos (history+parallel, deep+guards, etc.)
- Noise variants (clean + corrupted pairs for robustness)

Generator: `experiments/statechart_data/generate_synthetic.py`

---

## Phase 3: Evaluation Infrastructure

**Goal:** Compute all metrics needed for the experiment plan.
**Estimated effort:** 3-4 hours.
**Depends on:** Phase 1 complete; Phase 2 for real data.

### 3.1 Topology evaluation

Beyond edge F1, add:
- **Config match rate:** percentage of steps where argmax(predicted
  config) matches the target config
- **Topology exactness:** whether the extracted graph is isomorphic to
  the ground truth (binary)
- **Topology parsimony:** number of extracted edges vs. ground truth
  (fewer is better if F1 is maintained)

### 3.2 Guard evaluation

For charts with known guard structure:
- **Guard boundary accuracy:** how close the learned predicate's
  decision boundary is to the ground-truth threshold
- **Guard calibration:** ECE of guard output as a probability
- **Guard sharpness:** slope of the predicate at the boundary

### 3.3 Memory evaluation

For HybridMemory experiments:
- **Routing entropy:** H(routing_weights) — should be low if routing is
  query-dependent
- **Routing conditioned accuracy:** accuracy when routing favors hull
  vs. SSM
- **Restore accuracy:** exact match on history restoration queries

### 3.4 Training efficiency metrics

- **Sample efficiency curve:** accuracy vs. examples seen
- **Wall time curve:** accuracy vs. wall-clock seconds
- **Parameter count:** total trainable parameters per model variant
- **Graph memory:** peak MLX memory usage per epoch

### 3.5 Results format

Standardize benchmark output as JSON:
```json
{
  "experiment": "exp_neurosymbolic_statecharts",
  "variant": "cycle_baseline",
  "seed": 42,
  "n_epochs": 200,
  "metrics": {
    "train_loss_final": 0.45,
    "eval_accuracy": 0.87,
    "edge_f1": 0.75,
    "config_match_rate": 0.82,
    "wall_time_s": 34.2,
    "param_count": 12456
  },
  "topology": { ... },
  "history": [ ... ]
}
```

---

## Phase 4: Experiment Execution

**Goal:** Run the experiments from EXPERIMENT_PLAN.md.
**Depends on:** Phases 0-3 complete.

### Wave 0 — COMPLETE

All five experiments have results in `sc-ml`. See EXPERIMENT_PLAN.md for
detailed findings. Key takeaways:
- Semantics agreement: first divergence on history re-entry
- Trace detail: standard is cost-effective
- Temperature: edge F1=0.71, no collapse with exponential/cosine
- Guards: 96.9% accuracy, OOD generalization at 99%
- Routing: learned beats fixed on mixed regime (70% vs 64%)

### Wave 1 — EVIDENCE HARDENING IN PROGRESS

All six experiments have tensor-native scoring and multi-seed results.
Stately corpus (~48K valid charts + 926 history) being re-converted.

| # | Experiment | Key result | Status |
|---|-----------|-----------|--------|
| 6 | `ablation_ladder` | Rungs 0-4 tensor-native; flat dominates cycle, structured dominates branching | Re-run on full corpus |
| 7 | `hull_topology_gumbel` | fixed_soft F1=0.611, gumbel_hull F1=0.444 (16 charts) | Re-run on full corpus |
| 8 | `topology_policy_opt` | hybrid_search acc=0.470 F1=0.364 | Phase 1 flat-only |
| 9 | `grpo_sample_efficiency` | SDPO 3.7x faster than AdamW (240 vs 883 samples), same final acc (0.688) | **Done** (10 seeds) |
| 10 | `negative_edge_supervision` | plus_no_ops F1=0.600 vs positive-only 0.333 | Re-run on full corpus |
| 11 | `chart_family_generalization` | F1=0.634 (up from 0.285); deep/orthogonal/inter_level still fail | Re-run with stately charts |

**Key finding (exp 9):** SDPO direct and AdamW converge to identical
final accuracy (0.688) and edge F1 (1.000). SDPO is ~3.7x faster to
threshold. Both have high threshold-metric CV (~0.96) — this is
sensitivity to init, not method instability. GRPO/REINFORCE are broken
on this task (0.0 edge F1, ~31% accuracy).

**Remaining:** Re-run exps 6, 7, 11 on full stately corpus once
re-conversion completes. Diagnose deep/orthogonal family failure.

### Wave 2 and 3

See EXPERIMENT_PLAN.md. These depend on Wave 0-1 results and are
subject to negative-result contingencies.

---

## Phase 5: Paper Artifacts

**Goal:** Produce the tables, figures, and claims for the paper.
**Depends on:** Wave 0 and Wave 1 complete.

### Required tables

1. **Ablation ladder** (Wave 1, exp 6): The centerpiece. One row per
   rung, columns for accuracy, edge F1, loss, wall time, param count.

2. **Topology method comparison** (Wave 1, exps 7-8): Gumbel variants
   vs. fixed-soft vs. NSGA-II (from sc-ml). Edge F1, accuracy, wall
   time.

3. **Training algorithm comparison** (Wave 1, exp 9): AdamW vs. GRPO
   vs. SDPO. Sample efficiency, final accuracy, variance.

4. **Memory backend comparison** (Wave 2, exp 12): Hull vs. SSM vs.
   Hybrid vs. sc-ml baselines. Accuracy by sequence length, latency.

5. **Constrained decoder comparison** (Wave 2, exp 13): Flat executor
   vs. GrammarStatechart. Syntax validity, semantic validity, latency.

### Required figures

1. **Loss curves** per ablation rung
2. **Temperature schedule sensitivity** (Wave 0, exp 3)
3. **Guard decision boundary** visualization (Wave 0, exp 4)
4. **Routing weight distribution** (Wave 0, exp 5)
5. **Edge F1 vs. noise level** for topology variants
6. **Sample efficiency curves** for AdamW vs. GRPO vs. SDPO

---

## Dependency Graph

```
Phase 0 (fix NaN + graph accumulation)
  │
  ├── Phase 1 (minimal training pipeline)
  │     │
  │     ├── Wave 0: exps 3, 4, 5
  │     │
  │     ├── Phase 3 (evaluation infrastructure)
  │     │     │
  │     │     └── Wave 1: exps 6, 8, 9
  │     │
  │     └── Phase 2 (real data pipeline)
  │           │
  │           ├── Wave 0: exps 1, 2
  │           │
  │           └── Wave 1: exps 7, 10, 11
  │
  └── (nothing else can start until Phase 0 is done)
```

---

## Risk Register

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| NaN fix doesn't resolve all gradient issues | Medium | Critical | Test each component's gradient independently before integration |
| Graph accumulation fix insufficient for long sequences | High | High | Implement truncated BPTT with configurable window |
| MLX memory limits on large topology grids | Medium | Medium | Cap grid at 16 states, 8 events for initial experiments |
| Topology learning doesn't converge on real traces | Medium | High | Wave 0 diagnostics (schedule, guard, routing) detect this early |
| GRPO/SDPO overhead negates sample efficiency gains | Low | Medium | Measure wall-time efficiency, not just sample efficiency |
| sc-trace-gen charts don't cover needed motifs | Low | High | Build chart generator (Phase 2.2) with explicit coverage targets |

---

## Effort Estimate (updated 2026-03-24)

| Phase | Effort | Status |
|-------|--------|--------|
| Phase 0: Fix critical bugs | 2-3 hr | **DONE** |
| Phase 1: Minimal pipeline | 3-4 hr | **DONE** |
| Phase 2: Real data pipeline (remaining) | 2-3 hr | Larger corpus needed (>3 charts) |
| Phase 3: Eval infrastructure | 3-4 hr | Topology F1 done, guard/memory/efficiency pending |
| Wave 0 experiments | 4-6 hr | **DONE** (in sc-ml) |
| Wave 1 experiments (quick mode) | 8-12 hr | **DONE** — all 6 have initial results |
| Wave 1 tensor integration | 4-6 hr | **DONE** — rungs 0-4 tensor-native, 10-seed SDPO |
| Wave 1 evidence at scale | 2-4 hr | Waiting on stately corpus re-conversion |
| Wave 2 experiments | 8-12 hr | Next up (partially pre-existing in sc-ml) |
| Paper artifacts | 4-6 hr | Blocked on evidence quality |

Remaining: roughly 2-3 days from current state. The critical path is
now: stately corpus re-conversion → filter pipeline → re-run exps 6,
7, 11 at scale → paper artifacts. The 91K stately corpus (with
history fix, commit e2158c2) replaces the need for synthetic chart
generation for most families.
