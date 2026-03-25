# Sub-Experiment Plan for Unified Neuro-Symbolic Architecture

Experiments to flesh out in sc-ml to bolster the unified approach.
Ordered by priority. Each closes a specific gap between sc-ml's
existing evidence and what `exp_neurosymbolic_statecharts` claims.

The local `benchmark.py` is a useful harness seed, but it is not yet
strong evidence for the full architecture story:

- constrained generation is a standalone decoder benchmark, not a real
  LM loop
- delegation currently measures forward-pass delegation scores, not
  trained task completion
- topology recovery is limited to flat soft adjacency

Treat it as a scaffold for follow-on experiments, not as already
settled empirical support.

## Status Update (2026-03-24)

### Data infrastructure

| Component | Location | Status |
|-----------|----------|--------|
| sc-dataset-gen | `cmd/sc-dataset-gen/` | Committed on main. 8 products: traces, topology, graph, classification, vocabulary, history, guard_eval, mutation. Corpus batch mode with train/eval/test splits. |
| Stately corpus | `corpus/validated-manifest.json` | 49,100 valid charts from 91,213 scraped. |
| Synthetic generator | `statechart_data/generate_synthetic.py` | 11 family generators, 530 charts at medium tier. |
| sc-ml shared infra | `experiments/statechart_data/` | Build pipeline, splits, catalog, tracegen wrapper. Currently uses 17 charts. |
| sc-ml data dir | `data/statecharts/` | 17 charts, 48 topology rows, 768 execution rows. Needs rebuild on full corpus. |

### Stately corpus size distribution

| Family | Count | States (med/p95/max) | Transitions (med/max) |
|--------|------:|---------------------:|----------------------:|
| flat | 39,895 | 6 / 14 / 5,483 | 4 / 17,125 |
| shallow_nested | 7,276 | 8 / 19 / 234 | 7 / 228 |
| deep_nested | 1,929 | 13 / 29 / 69 | 10 / 81 |
| orthogonal | 1,905 | 11 / 26 / 69 | 8 / 94 |
| inter_level | 5,555 | 10 / 23 / 76 | 9 / 195 |
| history | 357 | 11 / 23 / 51 | 9 / 93 |

### Dataset tiers

| Tier | Charts | Families | Products | Use case |
|------|-------:|----------|----------|----------|
| CI | 30 | all 6 | traces, topology, classification, vocab | Fast CI smoke tests |
| Dev | 1,500 | all 6, balanced | all 8 | Development iteration |
| Full | 49,100 | natural distribution | all 8 | Paper runs |
| Synthetic | 530 | 11 families + guard/scale/adversarial | traces, topology | Controlled probes |

### Model size tiers

| Size | Parameters | n_states | hidden_dim | Heads | Hull layers | Use case |
|------|----------:|------:|----------:|------:|------:|----------|
| Tiny | ~5K | 8 | 32 | 2 | 1 | CI, smoke tests, debugging |
| Small | ~25K | 16 | 64 | 4 | 3 | Dev iteration, fast sweeps |
| Medium | ~100K | 32 | 128 | 4 | 4 | Main benchmark runs |
| Large | ~500K | 64 | 256 | 8 | 6 | Scaling analysis, paper |

Note: n_states is a maximum — charts with fewer states use the model
at reduced capacity. n_events and context_dim scale proportionally.

### Experiment status

| # | Experiment | Wave | Maturity | Dataset | Products | Model sizes | Gap |
|---|-----------|------|----------|---------|----------|-------------|-----|
| 1 | `exp_neurosymbolic_ablation_ladder` | 1 | synthetic | Dev, Full | topology, traces, vocab, class | Small, Medium | rungs 5-6 wrappers, tiny corpus |
| 2 | `exp_hull_topology_gumbel` | 1 | synthetic | Dev, Full | topology, traces, vocab | Small, Medium | small corpus, conservative gates |
| 3 | `exp_topology_policy_optimization` | 1 | synthetic | Dev, Full | topology, traces, vocab | Small, Medium | Phase 1 flat-only |
| 4 | `exp_hull_ssm_hybrid_memory` | 2 | synthetic | Dev | traces, history | Small, Medium | recall probe, not trace-backed |
| 5 | `exp_constrained_synthesis_lm` | 2 | synthetic | Dev | vocab, topology | Small | no real LM loop |
| 6 | `exp_mamba_hull_hybrid_executor` | 2 | synthetic | Dev | traces, history | Small, Medium | restore workload only |
| 7 | `exp_delegation_interactive` | 3 | scaffold | Dev | traces, topology, class | Small | demo harness only |
| 8 | `exp_typed_memory_hull_transfer` | 3 | scaffold | Dev | traces, history, mutation | Small | missing paired replay corpus |
| 9 | `exp_temperature_schedule_sensitivity` | 0 | paper-ready | Dev | topology, vocab | Tiny, Small | — |
| 10 | `exp_guard_predicate_interpretability` | 0 | paper-ready | Dev | guard_eval, traces | Tiny, Small | — |
| 11 | `exp_routing_analysis` | 0 | synthetic | Dev | traces, topology, class | Tiny, Small | heuristic proxy labels |
| 12 | `exp_grpo_sample_efficiency` | 1 | synthetic | Dev | topology, traces, vocab | Small, Medium | noisy threshold metric |
| 13 | `exp_smdp_option_boundaries` | 2 | scaffold | Dev | traces, topology, class | Small | proof-of-mechanism only |
| 14 | `exp_semantics_agreement_oracle` | 0 | mostly done | Dev | traces (debug) | — | empty deep-history divergence |
| 15 | `exp_trace_detail_ablation` | 0 | synthetic | Dev | traces (all levels) | — | — |
| 16 | `exp_negative_edge_supervision` | 1 | synthetic | Dev, Full | topology, traces | Small, Medium | local harness only |
| 17 | `exp_chart_family_generalization` | 1 | synthetic | Dev, Full | topology, class, graph | Small, Medium, Large | fails on held-out families |
| 18 | `exp_uncertainty_calibration` | 2 | not created | Dev | topology, traces, guard_eval | Small, Medium | — |
| 19 | `exp_synthetic_to_real_transfer` | 2 | not created | Synth+Dev | all | Small, Medium | — |
| 20 | `exp_observability_baseline` | 1 | not created | Dev, Full | topology, traces | — | needs sc-infer tool |

41 additional experiments with results exist in sc-ml outside this plan.

### Dataset × Model matrix (which combos to run)

**CI gate** (every PR): CI dataset × Tiny model. Must pass, <2 min.

**Dev iteration** (daily): Dev dataset × Small model. ~30 min.

**Benchmark runs** (weekly): Dev dataset × Small + Medium models.
Compare Small→Medium scaling. ~2 hours.

**Paper runs** (release): Full dataset × Small + Medium + Large models.
Full scaling curves. Overnight.

**Key scaling questions:**
- Does Medium beat Small on the same data? (architecture contributes)
- Does Full data beat Dev data at same model size? (data contributes)
- Does Large overfit on Dev but not on Full? (capacity/data balance)
- Do hard families (history, deep, ortho) need bigger models?

### Per-experiment metadata (`experiment.yaml`)

Each experiment directory now contains an `experiment.yaml` with static
metadata. Schema defined in `experiments/_infra/experiment_schema.py`.

```yaml
name: exp_hull_topology_gumbel
title: Hull-backed Gumbel-Softmax topology recovery
wave: 1
category: neurosymbolic
dataset_tiers:
  - Dev
  - Full
products:
  - topology
  - traces
  - vocabulary
model_sizes:
  - Small
  - Medium
maturity: synthetic
data_source: synthetic
gap: small corpus, conservative gates
```

Fields: `name`, `title`, `wave`, `category`, `dataset_tiers`, `products`,
`model_sizes`, `maturity` (paper-ready|synthetic|scaffold|not-created),
`data_source` (synthetic|corpus_v2|mixed), `gap`, `description`,
`depends_on`, `tags`.

### Normalized result reporting (`results.json`)

All experiments should write results.json in a common envelope:

```json
{
  "experiment": "exp_hull_topology_gumbel",
  "data_source": "corpus_v2",
  "dataset_tier": "Dev",
  "model_size": "Small",
  "timestamp": "2026-03-24T15:30:00",
  "seed": 42,
  "train_charts": 1050,
  "eval_charts": 225,
  "test_charts": 225,
  "epochs": 140,
  "wall_time_s": 342.5,
  "metrics": {
    "edge_f1": 0.87,
    "edge_precision": 0.91,
    "edge_recall": 0.83
  },
  "breakdowns": {
    "by_family": [...]
  },
  "case_results": [],
  "notes": ""
}
```

Use `experiments._infra.experiment_schema.wrap_results()` to build
envelopes and `save_results()` to write them.

Status report: `python3 -m experiments._infra.experiment_status`

---

## 1. `exp_neurosymbolic_ablation_ladder`

**Priority:** Highest. This is still the missing integration test.

**Current status in `sc-ml`:** Synthetic scaffold with results.
Tensor-native MLX forward passes for rungs 0-4, but rungs 5-6
(hybrid_memory, delegation) are still wrappers not full integrations.
Needs re-run on stately corpus.

**The gap:** The unified architecture claims that explicit topology,
guards, memory, and delegation help together. Right now we mostly have
component-level anchors. We do not yet have a clean "same task family,
add one component at a time" result.

**What to build:** A cumulative state-transition ablation ladder on the
same trace-prediction task family. Use the same optimizer, same train
/ eval split, same seed protocol, and report cycle and branching
regimes separately.

| Rung | Components | What it tests |
|------|-----------|---------------|
| 0 | Flat MLP baseline | Lower bound with no explicit SC bias |
| 1 | Differentiable statechart with soft adjacency | Does explicit SC structure help? |
| 1b | + Gumbel-Softmax | Does discrete selection pressure help on its own? |
| 2 | + neural predicates | Do learned guards add value beyond Gumbel selection? |
| 3 | + transition attention | Does richer transition resolution help? |
| 4 | + Hull-backed typed memory | Does exact sparse retrieval help? |
| 5 | + HybridMemory (Hull + SSM) | Does temporal memory add value beyond structural memory? |
| 6 | + learned delegation on compositional branching tasks | Does recursive delegation help when the task actually decomposes? |

**Important scope cut:** The constrained decoder is not part of this
ladder. It changes the problem from state-trajectory prediction to
token generation, so it belongs in `exp_constrained_synthesis_lm`.

**Task regimes:**
- Regime A: cycle + branching trace prediction
- Regime B: compositional branching tasks where delegation is actually
  meaningful

**Capacity control:** Keep the parameter budget fixed where practical.
When that is not possible, report parameter count and training FLOP
proxies per rung so gains are not mistaken for pure capacity effects.

**Metrics:** Test accuracy, edge F1, loss curve, wall time, parameter
count, and per-rung delta.

**Success bar:** Monotonic improvement is ideal. Failing that, we need
a clear regime-level story about which components help on which task
types.

**Why it matters:** This is the experiment that turns the work from "a
bundle of plausible mechanisms" into "an integrated architecture with
measured marginal gains."

---

## 2. `exp_hull_topology_gumbel`

**Priority:** High. This directly addresses the weakest topology
evidence gap.

**Current status in `sc-ml`:** Runnable synthetic pilot. The benchmark
now carries the staged seven-variant grid plus train/valid/test counts
and wall-time reporting, but it is still synthetic and still embeds
the split in the output artifact instead of writing a reusable
manifest.

**The gap:** sc-ml's topology learning experiments
(`exp_nested_hull_topology_learning`, `exp_nested_hull_topology_regime`)
use a small stdlib-only learner. Sparse projection behaves like a
regularizer, but it does not beat the fixed-soft baseline. Our local
model uses a richer topology learner, but we do not have the matching
ablation artifact in sc-ml.

**Implementation note:** The current `NeuroSymbolicStatechart` bundles
neural guards and transition attention together. To run a real ablation
grid, sc-ml will need either feature flags or thin wrapper variants.

**What to build:** Replace the toy adjacency learner with a staged
Gumbel-based learner and evaluate on the same trace-recovery task.

| Variant | Dense vs hull | Neural predicates | Transition attention |
|---------|---------------|-------------------|----------------------|
| fixed-soft | baseline | no | no |
| annealed-projected | baseline | no | no |
| gumbel-dense-minimal | dense | no | no |
| gumbel-dense-guarded | dense | yes | no |
| gumbel-dense-full | dense | yes | yes |
| gumbel-hull-guarded | hull | yes | no |
| gumbel-hull-full | hull | yes | yes |

Same data: synthetic traces from hidden topologies. Same noise sweep:
0.15, 0.35, 0.55.

**Compute note:** The full 7-variant grid across 3 noise levels and
multiple seeds is expensive if each run uses the full local model.
Prefer a topology-focused extraction of the learner first. Run a pilot
matrix to establish per-run cost before committing to the full sweep.

**Metrics:** Edge F1, test accuracy, entropy, topology recovery
exactness, and seed variance.

**Success bar:** At least one Gumbel variant should beat the fixed-soft
baseline on accuracy or F1. If none do, the "learnable topology"
argument remains provisional.

**Reference:** `exp_nested_hull_topology_learning/NOTES.md`,
`exp_nested_hull_topology_regime/NOTES.md`.

---

## 3. `exp_topology_policy_optimization`

**Priority:** High. This is the bridge from sc-ml's strongest topology
result to our training story.

**Current status in `sc-ml`:** Partly absorbed under
`exp_grpo_topology_evolution` and `exp_sdpo_hindsight_topology`.
Both are runnable flat synthetic TicTacToe benchmarks, but the
apples-to-apples Phase 1 parity with a flat-restricted NSGA-II
baseline is not finished yet.

**The gap:** sc-ml's NSGA-II result reaches 93.3% on TicTacToe. Our
GRPO and SDPO trainers optimize differentiable topologies, but the
current local model only parameterizes flat
`n_states x n_states x n_events` adjacency. That is a narrower search
space than sc-ml's full hierarchy / state-type / history encoding.

**What to build:** Stage the comparison explicitly.

**Phase 1: Flat-topology comparison only**

Restrict the search space so the methods are actually comparable:

- flatten the TicTacToe legality task to a flat transition graph, or
- constrain the NSGA-II baseline to the same flat topology class

Then compare:

| Method | Scope |
|--------|-------|
| NSGA-II evolution | flat restricted baseline |
| GRPO on soft adjacency | local gradient method |
| SDPO with EMA teacher + hindsight | local self-distilled method |
| NSGA-II coarse -> GRPO fine-tune | hybrid method |

**Dependency:** The hybrid row needs a serialization bridge from the
evolutionary genome encoding to the differentiable soft-adjacency
parameterization. That bridge is part of the experiment, not a trivial
adapter.

**Phase 2: Full topology comparison**

Only after Phase 1: extend the differentiable model to represent
hierarchy, state types, and history, then rerun against the full
upstream search space.

**Metrics:** Legal move accuracy, edge F1, wall time, sample
efficiency, and topology parsimony.

**Success bar:** In Phase 1, GRPO / SDPO should match or beat the flat
restricted evolutionary baseline, or the hybrid should dominate both.
If gradient methods lose, that is still a useful result, but it must be
reported as a flat-topology result, not as a full replacement for the
upstream search space.

**Why merged:** GRPO and SDPO belong in the same benchmark. Same task,
same harness, same scoring.

**Reference:** `exp_topology_evolution/NOTES.md`.

---

## 4. `exp_hull_ssm_hybrid_memory`

**Priority:** High. This validates HybridMemory against sc-ml's
established memory baselines.

**Current status in `sc-ml`:** Runnable and already useful as a local
mechanism benchmark. The strongest remaining gap is trace-backed
long-horizon data rather than benchmark shell work.

**The gap:** sc-ml benchmarked 5 memory mechanisms and HTM won the
short sequence-recall benchmark at 24%. We now have Hull, SSM, and
HybridMemory implementations, but no head-to-head comparison.

**What to build:** A two-regime benchmark by design.

**Regime A: short exact-history tasks**

- sequence recall, vocab=8, seq_len=6
- same sc-ml baselines: Attention, NTM, Recurrent, CMS, HTM
- new entries: HullKVCache, SSMMemory, HybridMemory in CONCAT / ADD /
  GATE / ATTENTION modes

**Regime B: long-horizon temporal tasks**

- sequence recall with seq_len=50+
- event-pattern recognition over 1000+ steps
- long-horizon control-like traces where SSM should matter

**Execution note:** Long-horizon MLX runs will likely need chunked
evaluation or another mitigation for graph accumulation. Treat this as
a harness requirement, not as an optional optimization.

**Metrics:** Accuracy, latency, parameter count, gradient flow,
routing entropy, and per-backend usage in the hybrid modes.

**Success bar:** HybridMemory should beat HTM on at least one regime.
If it only ties or loses everywhere, the "complementary backends"
thesis is weak.

**Reference:** `differentiable/NOTES.md`.

---

## 5. `exp_constrained_synthesis_lm`

**Priority:** High. This is the decoder-specific bridge experiment.

**Current status in `sc-ml`:** Runnable and strong as a local
mechanism result. The directory now includes a hard-eval builder, but
the decoder loop is still a small synthetic-logit harness rather than
the full Qwen/LoRA path.

**The gap:** sc-ml already has strong constrained decoding evidence,
but the executor in that path is flat. Our `GrammarStatechart` decoder
is richer, yet the current local benchmark only exercises it as a
standalone decoder. That is not the right comparison point.

**What to build:** Replace sc-ml's flat `SCExecutor` path with a
GrammarStatechart-based decoder inside the actual Qwen2.5-Coder LoRA
loop.

| Variant | Description |
|---------|-------------|
| flat SCExecutor | current sc-ml baseline |
| GrammarStatechart | parser-state statechart masking |
| GrammarStatechart + hull-backed guard memory | full decoder-side architecture |

Test hierarchical statechart outputs specifically, because that is
where the flat executor is known to under-model the semantics.

**Infrastructure fallback:** If the full Qwen2.5-Coder LoRA path is not
available early, stage this experiment with a smaller local model or a
learned-logit harness that preserves the masking behavior. The real LM
loop remains the target artifact.

**Metrics:** Syntax validity, semantic validity via `sc validate`,
generation latency, mask density, and failure taxonomy.

**Success bar:** GrammarStatechart should improve validity on the
hierarchical cases without blowing up latency.

**Reference:** `exp_schema_guided_sampling/REPORT.md`, `gap_analysis.md`.

---

## 6. `exp_mamba_hull_hybrid_executor`

**Priority:** High. This is the cleanest test of the "Hull for exact
restore, SSM for long horizon" claim.

**Current status in `sc-ml`:** Runnable and already strong as a local
mechanism result. The remaining gap is a trace-backed hierarchical
history-restore corpus.

**The gap:** sc-ml has an SSM executor and it has exact history-restore
experiments. We do not yet have a composed benchmark that uses the SSM
path for forward execution and Hull-backed retrieval for restore.

**What to build:** Compose the Mamba executor with hull-backed deep
history and evaluate on long sequences that repeatedly enter and exit
composite states.

| Variant | Forward execution | History restore |
|---------|------------------|-----------------|
| SSM only | Mamba executor | dense scan |
| Hull only | standard stepper | hull exact |
| Hybrid | Mamba executor | hull exact |

**Task-design requirement:** Use composite states nested at least 3-4
levels deep so restore error can actually accumulate in the SSM-only
baseline.

**Metrics:** Restore accuracy at depth, execution throughput, memory
overhead, and restore latency as sequence length grows.

**Success bar:** The hybrid should preserve exact restore while keeping
the SSM throughput advantage on long traces.

**Reference:** `exp_geometric_history_restore/NOTES.md`,
`exp_mamba_sc_executor`.

---

## 7. `exp_delegation_interactive`

**Priority:** Medium. Important, but it should start narrow.

**Current status in `sc-ml`:** Runnable synthetic control-loop
benchmark with richer artifact modes (`summary`, `episodes`, `full`),
but still not a corpus-backed learned delegation result.

**The gap:** sc-ml's delegation experiment is a Python scaffold. Our
local delegation module is neural, but the current local benchmark only
collects delegation probabilities from forward passes. That is not yet
a task-level result.

**What to build:** Start with a narrow task-conditioned benchmark, not
ARC or Go. Use sc-ml's existing hidden-rule / repair / trace task
family and replace the template library with learned delegation:

| Variant | Description |
|---------|-------------|
| Single-agent flat | baseline |
| Python scaffold delegation | current sc-ml delegation path |
| Neural delegation | learned decide / project / aggregate |

**Training requirement:** This experiment needs an actual task loss and
worker-policy learning loop. That includes:

- a trainable child policy
- a reward or loss that propagates through delegation decisions
- enough task variety that delegation is not just constant overhead

Pure delegation-probability telemetry is not enough. This may require
its own task-generation pipeline.

**Metrics:** Solve rate, iterations to solution, total actions,
delegation frequency, and depth distribution.

**Success bar:** Neural delegation should reduce iterations or actions
without reducing solve rate.

**Reference:** `exp_recursive_delegation_control/NOTES.md`.

---

## 8. `exp_typed_memory_hull_transfer`

**Priority:** Low. Defer until the replay corpus problem is solved.

**Current status in `sc-ml`:** Runnable synthetic benchmark. Evidence
gating rescues the transfer story, but ungated hull transfer still has
an explicit hurt regime.

**The gap:** sc-ml's transfer experiments are directionally useful, but
they still call out a real-input blocker: no local replay corpus with
paired transfer episodes and truthful confidence signals.

**Why deferred:** Swapping in Hull-backed typed memory on top of a
synthetic transfer benchmark does not strengthen the real claim. Wait
for the replay corpus, then test whether hull geometry changes the
transfer story.

**Reference:**
- `exp_runtime_memory_transfer/NOTES.md`
- `exp_runtime_negative_transfer/NOTES.md`
- `exp_runtime_transfer_gating/NOTES.md`

---

## 9. `exp_temperature_schedule_sensitivity`

**Priority:** Medium. Fast diagnostic, high leverage.

**The gap:** The differentiable statechart relies on temperature
annealing, but we only have one schedule family in hand. If topology
quality is highly schedule-sensitive, downstream results are fragile.

**What to build:** Fix the task to cycle / branching trace prediction
and sweep:

- exponential schedules with multiple decay rates
- cosine schedules with multiple periods
- linear schedules
- step schedules
- adaptive entropy-gated schedules

**Metrics:** Edge F1, final accuracy, topology sparsity, collapse rate,
and variance across seeds.

**Success bar:** Identify at least one schedule family that avoids
premature collapse across the tested noise levels.

**Why it matters:** If annealing choice dominates the result, topology
claims need to be framed as schedule-conditional.

---

## 10. `exp_guard_predicate_interpretability`

**Priority:** Medium. Mechanistic honesty for the guard story.

**The gap:** Neural predicates are currently treated as a capability,
but we do not inspect what they actually learn. If they collapse to
near-constant scores or memorize trivial patterns, the guard story is
weak.

**What to build:** Train on a task with known guard structure, then
probe the learned predicates directly:

- sweep key context dimensions through the ground-truth boundary
- plot guard output per transition
- measure boundary sharpness and crossing-point accuracy
- test generalization on held-out context ranges

**Metrics:** Decision-boundary accuracy, sharpness, calibration, and
OOD generalization.

**Success bar:** At least one learned predicate should recover the
ground-truth decision boundary cleanly on the branching benchmark.

---

## 11. `exp_routing_analysis`

**Priority:** Medium. Validates the HybridMemory routing claim.

**Current status in `sc-ml`:** Runnable synthetic routing probe. It is
useful for mechanism validation, but not yet trace-backed.

**The gap:** Learned routing is only interesting if it is query
dependent. A fixed mixture can match the interface without validating
the thesis.

**What to build:** Train HybridMemory on a mixed structural / temporal
task and analyze:

- routing-weight distribution
- routing conditioned on query type
- routing entropy over training
- learned routing versus fixed routing at 0.0, 0.5, and 1.0

**Metrics:** Routing bimodality, conditional routing accuracy,
task performance under learned versus fixed routing, and routing
stability across seeds.

**Success bar:** Learned routing should beat fixed routing and show a
meaningful query-conditioned split between Hull-heavy and SSM-heavy
queries.

---

## 12. `exp_grpo_sample_efficiency`

**Priority:** Medium-high. Direct validation of the GRPO / SDPO
training contribution.

**Current status in `sc-ml`:** Synthetic with results. 10-seed sweep
completed but synthetic-only and threshold metric is noisy.

**The gap:** We have functional trainer implementations, but not a
clear sample-efficiency comparison against simpler optimizers on the
same task and parameter budget.

**What to build:** Hold model and task fixed and compare:

- AdamW supervised training
- REINFORCE
- GRPO across multiple group sizes
- SDPO with and without hindsight relabeling
- NSGA-II as a reference curve when applicable

Measure accuracy against examples seen, not just epochs.

**Metrics:** Sample efficiency, final accuracy, wall time, and variance
across seeds.

**Success bar:** GRPO or SDPO should either reach a target accuracy
with materially fewer examples than AdamW, or surpass AdamW's final
accuracy at convergence.

---

## 13. `exp_smdp_option_boundaries`

**Priority:** Low-medium. Empirical support for the formal SMDP /
Options correspondence.

**Current status in `sc-ml`:** Scaffolded (benchmark.py, NOTES.md,
no results yet).

**The gap:** The proposal's "composite states = options" argument is
formally appealing, but it is not yet tied to measured behavior.

**What to build:** Construct a hierarchical task with explicit subtask
structure and compare flat versus hierarchical training:

- convergence speed
- value-function accuracy
- option-duration distribution
- empirical agreement with the effective-discount formulation

**Success bar:** Hierarchical training should converge faster than the
flat baseline, and the measured effective discount should match the
theoretical composition closely enough to support the formal section.

---

## Data Dependencies

The main data risk is not lack of files. It is lack of inputs in the
right shape for the claims we want to make.

### What Exists Today

- synthetic cycle and branching generators in
  `exp_neurosymbolic_statecharts/training.py`
- a flat prompt-to-statechart corpus in `sc-ml/data/train.jsonl` and
  `sc-ml/data/valid.jsonl`
- a small Go trace directory under `sc-ml/traces/`
- ARC reference data under `sc-ml/data/arc_reference_repo/`
- a synthetic execution-replay harness in `exp_execution_replay`
- a shared `sc-ml/experiments/statechart_data/` layer that wraps
  `sc-trace-gen`, builds manifests/splits, and derives execution /
  topology supervision rows

These are useful starting points, but several are explicitly scaffold
data sources, not benchmark-grade corpora. For example,
`exp_execution_replay` says it is still "trace-synthetic today," and
the current prompt corpus appears dominated by simple flat charts rather
than nested or history-rich machines. Likewise, the new
`statechart_data` layer solves build plumbing, but it does not by
itself create benchmark-grade corpora.

### Canonical Trace Builder: `cmd/sc-trace-gen`

Use `cmd/sc-trace-gen` as the canonical builder for raw execution-trace
datasets. Do not introduce a second ad hoc trace format if the existing
`ExecutionTrace` schema can carry the data.

`sc-ml` now has the companion wrapper/build layer for this under
`experiments/statechart_data/tracegen.py`,
`experiments/statechart_data/build.py`,
`experiments/statechart_data/derive_execution_labels.py`, and
`experiments/statechart_data/derive_topology_labels.py`.

**Current role:**
- load a statechart in proto JSON or textproto form
- run seeded random events through the Go semantics engine
- emit a JSON dataset wrapper with:
  - statechart id
  - source file
  - generation seed
  - sampled event vocabulary
  - ground-truth topology summary
  - one or more `ExecutionTrace` payloads

**Canonical trace contract:**
- per-trace payloads should stay aligned with
  `proto/statecharts/v1/execution.proto`
- if extra dataset metadata is needed, wrap the trace rather than
  forking the trace schema
- ground-truth topology should stay adjacent to the traces so topology
  recovery experiments do not need a second lookup path
- split manifests should be seed-stable and chart-family aware

**Intended detail tiers:**
- `minimal`: event, source / target configuration, fired transitions,
  and aggregate counters
- `standard`: `minimal` plus guard pass / fail records and context
  before / after
- `debug`: `standard` plus bound values, evaluation errors, no-op
  steps, and rejected candidate details where available

**Why this matters:** The same generator should feed multiple
experiments with different storage budgets:
- topology and sample-efficiency sweeps want compact traces
- guard-interpretability and repair tasks want rich guard metadata
- long-horizon executor work needs throughput-oriented traces with
  reproducible seeds and stable manifests

The rule should be simple: raw traces come from `sc-trace-gen`,
derived corpora add labels on top.

### Missing Sources by Experiment

#### 1. Real execution-trace corpus

**Needed for:**
- `exp_neurosymbolic_ablation_ladder`
- `exp_hull_topology_gumbel`
- `exp_grpo_sample_efficiency`
- `exp_topology_policy_optimization`
- follow-on offline / online comparisons

**Why missing:** The current trace-learning path is still synthetic.

**Status note:** The export/derivation plumbing now exists in
`sc-ml/experiments/statechart_data/`, but the corpus itself is still
largely unpopulated.

**Minimum shape:**
- chart id and chart definition or stable chart reference
- initial configuration and initial context
- per-step event
- source configuration and target configuration
- guard evaluation records
- action execution records
- context before and after
- episode metadata: domain, difficulty, seed, outcome

**Best acquisition path:**
- use `cmd/sc-trace-gen` as the primary export path for raw traces
- emit `ExecutionTrace`-aligned JSON or proto, not experiment-specific
  step records
- collect traces from real statechart runs, not only generated hidden
  topologies
- store train / eval splits by chart family, not by shuffled step

**Why it matters:** Without real traces, the topology and training
results stay in mechanism territory.

#### 2. Hierarchical topology corpus

**Needed for:**
- `exp_hull_topology_gumbel`
- `exp_topology_policy_optimization` Phase 2
- `exp_smdp_option_boundaries`
- stronger guard and schedule diagnostics

**Why missing:** The local generators only cover simple cycles and one
scalar-threshold branching case.

**Status note:** `statechart_data` can already derive topology
supervision from raw trace bundles once the chart family and raw traces
exist. The missing piece is the chart suite plus exported raw bundles.

**Minimum shape:**
- gold topology with explicit parent links
- state type per node: basic, OR, AND
- history flags and kind (`H` vs `H*`)
- transition set with guards and actions
- generated traces with configurable observation noise
- difficulty tags: flat, nested, orthogonal, inter-level, history-heavy

**Best acquisition path:**
- build a chart generator that samples from a constrained family of
  valid hierarchical machines
- feed those charts through `cmd/sc-trace-gen` so the resulting traces
  and topology summaries share one contract
- keep the ground-truth topology alongside each trace bundle
- stratify by mechanism so ablations can be attributed

**Why it matters:** A flat-only corpus cannot justify claims about
hierarchy, history, or options.

#### 3. Hierarchical constrained-generation corpus

**Needed for:**
- `exp_constrained_synthesis_lm`
- decoder-side semantic-validity evaluation

**Why missing:** The current prompt corpus is broad enough to train on
simple charts, but it does not appear to cover the hard cases in a
measurable way.

**Minimum shape:**
- prompt
- gold target program or JSON
- parsed statechart metadata: depth, parallel regions, history usage,
  number of guards, orthogonality
- semantic-validity label via `sc validate`
- negative examples or near-miss invalid outputs for evaluation

**Best acquisition path:**
- expand `train.jsonl` / `valid.jsonl` with prompts that explicitly ask
  for nested, orthogonal, and history-bearing charts
- generate hard evaluation subsets where flat executors are known to
  fail
- record semantic metadata once at data-build time so benchmark filters
  are cheap

**Why it matters:** If the eval set is mostly flat traffic lights and
linear chains, the GrammarStatechart result will be underpowered.

#### 4. Paired transfer replay corpus

**Needed for:**
- `exp_typed_memory_hull_transfer`
- any truthful evidence-gated transfer benchmark

**Why missing:** The transfer experiments already call out the blocker:
no replay source exposes paired helpful and harmful transfer episodes
with truthful confidence signals.

**Minimum shape:**
- source episode and target episode linked by task family
- reuse label: aligned, harmful, or neutral
- memory items available for transfer by tier
- task outcome under source and target semantics
- confidence / evidence signals derived from the replay, not synthetic
  placeholders

**Best acquisition path:**
- build paired episodes from a controller or repair workload with known
  reuse structure
- include alias-flip and stale-prior cases intentionally
- store transfer metadata with each episode pair

**Why it matters:** Without paired real episodes, transfer claims are
directional only.

#### 5. Replay-backed repair / patch corpus

**Needed for:**
- `exp_delegation_interactive`
- runtime patch-search follow-ons
- any repair-conditioned delegation or memory experiment

**Why missing:** The repo lacks a real task-suite JSON that binds
baseline charts, failure traces, and candidate repairs together.

**Minimum shape:**
- baseline chart
- failing trace
- observed failure type
- candidate patch set with dimension labels
  (`topology`, `guard`, `action`, `history`)
- patched chart outcome on held-out traces

**Best acquisition path:**
- harvest runtime failures from controller or validator runs
- normalize them into patch tasks
- keep accepted and rejected patches so search quality can be measured

**Why it matters:** Delegation and unified patch search both need
real subtask structure, not only hand-authored toy repairs.

#### 6. Task-conditioned delegation corpus

**Needed for:**
- `exp_delegation_interactive`

**Why missing:** The current delegation path measures probabilities,
not learned task decomposition on stored tasks.

**Minimum shape:**
- parent task state
- subtask boundary or candidate decomposition
- child context projection
- child action sequence or result
- solve outcome, cost, and depth
- supervision or reward signal for whether delegation helped

**Best acquisition path:**
- derive tasks from repair, hidden-rule, or execution-replay workloads
- record both delegated and non-delegated rollouts
- preserve enough variation that delegation is sometimes useful and
  sometimes wasteful

**Why it matters:** Without this, the model can only learn a generic
"delegate more / less" prior.

#### 7. Long-horizon history-restore corpus

**Needed for:**
- `exp_hull_ssm_hybrid_memory`
- `exp_mamba_hull_hybrid_executor`

**Why missing:** Current history-restore evidence is prototype-level
and exact-backend oriented.

**Status note:** The executor and memory benchmarks now exist, so this
is primarily a corpus gap rather than a benchmark-shell gap.

**Minimum shape:**
- long sequences with repeated entry / exit of composite states
- explicit shallow and deep history targets
- nested depth labels
- restore queries at multiple horizons
- throughput and latency annotations per run

**Best acquisition path:**
- generate hierarchical controller traces with forced re-entry patterns
- use `cmd/sc-trace-gen` as the raw trace emitter, then attach restore
  queries and labels as a derived layer
- include adversarial miss cases and repeated restore queries
- separate short exact-history tasks from long temporal tasks

**Why it matters:** The hybrid memory claim depends on workloads where
exact restore and long-horizon modeling both matter.

#### 8. ARC / controller replay corpus

**Needed for:**
- `exp_delegation_interactive` follow-ons
- transfer and typed-memory controller benchmarks
- broader systems validation after the core paper

**Why missing:** The ARC controller ablation still relies mostly on
synthetic tasks, even though a real mode exists.

**Minimum shape:**
- observation sequence
- controller actions
- planner decisions
- memory reads and writes
- reward or solve outcome
- task id and episode id

**Best acquisition path:**
- save real `arcagi3` or analogous controller rollouts as replay files
- normalize them into a stable schema independent of live API access

**Why it matters:** This is the bridge from mechanism experiments to a
systems story.

### Recommended Data Build Order

1. Real execution-trace corpus
2. Hierarchical topology corpus
3. Hierarchical constrained-generation eval corpus
4. Long-horizon history-restore corpus
5. Task-conditioned delegation corpus
6. Replay-backed repair / patch corpus
7. Paired transfer replay corpus
8. ARC / controller replay corpus

This order keeps Wave 0 and Wave 1 unblocked first, then unlocks the
memory and systems experiments.

### `sc-trace-gen` Data Products

Treat the trace tool as producing a small number of named source
artifacts rather than one generic dump:

- `execution_traces`: real or generated runs for training and replay
- `topology_traces`: trace bundles paired with gold topology metadata
- `history_restore_traces`: long-horizon traces intended for restore
  query derivation

Each artifact family should record:
- tool version or commit
- generation seed
- chart provenance
- detail tier used at export time
- split manifest or dataset partition label

This keeps downstream experiments honest about whether they were built
from minimal, standard, or debug-grade trace data.

### Canonical Storage Plan

To avoid ad hoc formats, use a small number of stable dataset roots:

- `sc-ml/data/execution_traces/`
- `sc-ml/data/topology_corpus/`
- `sc-ml/data/constrained_synthesis/`
- `sc-ml/data/history_restore/`
- `sc-ml/data/delegation_tasks/`
- `sc-ml/data/repair_tasks/`
- `sc-ml/data/transfer_pairs/`
- `sc-ml/data/controller_replays/`

Each root should contain:
- `README.md` with provenance and generation contract
- `train/`, `valid/`, `test/` splits
- `metadata.json` with schema version and build date
- `build.json` with the `sc-trace-gen` invocation or equivalent build
  recipe when the source is trace-derived
- task-family manifests so experiments can filter by mechanism

### Data Readiness Gates

Before promoting any source from scaffold to benchmark input:

- verify labels against a runtime or oracle when possible
- split by episode / chart family to avoid leakage
- measure coverage across flat, nested, parallel, and history-heavy
  cases
- record known blind spots explicitly in the experiment notes

If these gates are not met, the result should be described as a
mechanism benchmark, not as final evidence.

---

## 14. `exp_semantics_agreement_oracle`

**Priority:** Highest. Safety net for the entire ML pipeline.

**Current status in `sc-ml`:** Mostly implemented with results.
Trace-backed cross-check against sc-trace-gen, but has a checked-in
empty deep-history divergence that needs resolution.

**The gap:** No existing experiment tests cross-executor agreement.
`exp_executor_compliance` checks one executor against a fixed spec, and
`exp_sc_equivalence` compares chart-level bisimulation, but neither
measures whether a learned or hybrid executor produces the same
config, fired transitions, guard outcomes, and history restore as the
Go semantics engine behind `sc-trace-gen`.

**What to build:** Generate traces from hierarchical charts using
`sc-trace-gen -detail=debug`. Run the same event sequences through
learned / hybrid executors. Measure exact agreement on:

- configuration match per step
- fired transition set
- guard evaluation outcomes (leveraging the new `TransitionRef` on
  `GuardEvaluation`)
- history restore correctness on re-entry

**Task design:** Use charts with guards, history, AND states, and
inter-level transitions so agreement is non-trivial.

**Metrics:** Config match rate, transition agreement, guard agreement,
restore agreement, and first-divergence step.

**Success bar:** Any executor variant used downstream must achieve
≥99% config match on the test set. Anything lower means the model is
optimizing toward wrong semantics.

**Why it matters:** This is the experiment that catches semantic drift
before it contaminates all other results.

---

## 15. `exp_trace_detail_ablation`

**Priority:** High. Cheap diagnostic that should run early.

**Current status in `sc-ml`:** Synthetic with results.

**The gap:** We now emit three trace detail levels from `sc-trace-gen`
(minimal, standard, debug) but have no evidence for which level is
worth the storage and compute cost for downstream training.

**What to build:** Generate identical chart families at all three
detail tiers. Train the same topology learner on each. Compare:

- topology F1
- guard-boundary recovery (standard and debug only)
- training throughput (tokens/s, examples/s)
- dataset size

**Task design:** Use the same hidden-topology chart suite across all
tiers. Same seeds, same splits.

**Metrics:** Edge F1, guard accuracy, training throughput, dataset
bytes, and per-tier delta.

**Success bar:** Identify the minimum detail level that preserves F1
within 2% of the richest tier. If minimal matches standard, every
downstream experiment saves storage.

---

## 16. `exp_negative_edge_supervision`

**Priority:** High. Clean test of negative trace evidence.

**Current status in `sc-ml`:** Synthetic with results. `plus_no_ops`
best at F1=0.600.

**The gap:** No existing experiment explicitly uses negative-labeled
edges (transitions that were candidates but did not fire) as training
signals. `exp_08_adversarial` poisons a transformer with illegal
transitions, but that is about robustness, not about using no-op
events and rejected guards as supervision.

**What to build:** Train topology learners with and without explicit
negatives from:

- no-op events (event sent, no transition fired)
- rejected guard evaluations (guard evaluated false)
- non-fired candidate transitions (active source, event match, but
  outcompeted by conflict resolution)

These are already available at standard detail level from
`sc-trace-gen`.

**Variants:**

| Variant | Positive edges | Negative edges |
|---------|---------------|----------------|
| positive-only | fired transitions | none |
| +no-ops | fired | no-op events as non-edges |
| +guard-rejects | fired | guard-false as non-edges |
| +candidates | fired | all candidate non-edges |

**Metrics:** Edge F1, precision, recall, false positive rate, and
training sample efficiency.

**Success bar:** At least one negative variant should improve precision
without degrading recall, or improve F1 by ≥3%.

---

## 17. `exp_chart_family_generalization`

**Priority:** High. Tests whether the model learns semantics or
memorizes topology motifs.

**Current status in `sc-ml`:** Synthetic scaffold with results. F1
improved from 0.285 to 0.634 with shared-axis denoiser, but still
fails on held-out families (deep_nested, orthogonal, inter_level).

**The gap:** Existing transfer experiments (`exp_transfer_coverage`,
`exp_transfer_learning`, `exp_regex_transfer`) test domain-level
transfer (game→game, regex→regex). None test generalization across
parametric topology families: flat, nested OR, AND states, deep
history, inter-level transitions.

**What to build:** Define topology families by structural motif:

- flat chains and cycles
- shallow nested (depth ≤ 2)
- deep nested (depth ≥ 3)
- orthogonal (AND states)
- history-bearing (H and H*)
- inter-level transitions

Train on a subset of families, test on held-out families. Compare
against random splits of the same total data.

**Metrics:** Edge F1, config accuracy, and per-family breakdown on
held-out motifs.

**Success bar:** Family-split performance should degrade gracefully
(≤15% F1 drop on unseen families). If a specific family causes
catastrophic failure, that identifies the architectural gap.

---

## 18. `exp_uncertainty_calibration`

**Priority:** Medium-high. Zero coverage in sc-ml.

**Current status in `sc-ml`:** Not yet created.

**The gap:** No experiment measures whether model confidence scores
are well-calibrated. If routing weights, transfer gates, delegation
decisions, and guard outputs drive downstream behavior, confidence
quality matters as much as raw accuracy.

**What to build:** Evaluate calibration for:

- edge existence probabilities (topology learner)
- guard output confidences (neural predicates)
- routing weights (HybridMemory router)
- delegation scores (when to delegate)

**Metrics:** Expected Calibration Error (ECE), reliability diagrams,
Brier score, and calibration before vs. after temperature scaling.

**Success bar:** ECE < 0.05 after temperature scaling for at least
the edge and guard outputs. If routing or delegation scores are poorly
calibrated, downstream gating decisions will be unreliable.

---

## 19. `exp_synthetic_to_real_transfer`

**Priority:** Medium. Measures the gap that all mechanism experiments
are implicitly hoping is small.

**Current status in `sc-ml`:** Not yet created. Related:
`exp_runtime_negative_transfer` has results measuring negative
transfer but not the synthetic-to-real gap specifically.

**The gap:** The runtime trace experiments are moving toward real data
incidentally, but nobody has measured the synthetic-to-real
generalization gap as a controlled study. `exp_runtime_trace_topology`
uses real ARC traces, and `exp_runtime_patch_strategy_ablation`
accepts `--source real`, but neither measures the delta between
synthetic-only and real-finetuned performance.

**What to build:** Pretrain on `sc-trace-gen` output from the
topology suite. Finetune on real `ExecutionTrace`-aligned data from
controller runs or manually-authored charts. Measure:

- zero-shot transfer accuracy (synthetic model on real traces)
- finetuned accuracy vs. real-only baseline
- minimum real data needed to close the gap

**Metrics:** Edge F1, config accuracy, zero-shot vs. finetuned delta,
and data efficiency curve.

**Success bar:** Synthetic pretraining should provide ≥10% F1 gain
over training from scratch on the same amount of real data. If not,
synthetic data is not pulling its weight.

---

## 20. `exp_observability_baseline`

**Priority:** High. This is the missing non-neural baseline for every
topology learning experiment.

**Current status:** Not implemented. Requires `sc-infer` tool (prompt
written at `/tmp/prompt-sc-infer.md`, ready to build at `cmd/sc-infer/`).

**The gap:** Every topology experiment (exps 6, 7, 8, 11) compares
neural architectures against each other, but none compare against the
simplest possible baseline: counting observed transitions. Without
this, we can't distinguish "the neural model learned structure" from
"any algorithm that sees enough data would get this right."

**What to build:** Use `sc-infer` (a streaming topology inference tool
that reconstructs charts from transition observations via counting) as
a non-neural baseline. For each chart in the validated corpus:

1. Generate traces at increasing sample budgets:
   K = 10, 50, 200, 1000 traces × 20 steps each
2. Pipe each batch through `sc-infer`:
   `sc-trace-gen -chart X -traces K -steps 20 | sc-infer`
3. Compare inferred chart against ground truth using edge F1, config
   accuracy, and topology exactness
4. Record traces-to-convergence: the smallest K where F1=1.0

Aggregate results by chart family (flat, shallow_nested, deep_nested,
orthogonal, history, inter_level) to characterize per-family sample
complexity.

**What this gives us:**

| Output | How it's used |
|--------|---------------|
| Per-family traces-to-convergence | Training curriculum: sort easy→hard |
| Observability ceiling per chart | Upper bound — no model can learn what isn't in the data |
| Non-neural baseline curves | If neural model ≤ sc-infer, architecture isn't contributing |
| Unobservable edge fraction | Identifies charts where structure must be inferred, not observed |

**Comparison protocol:** Overlay sc-infer convergence curves with
neural model curves from exps 6, 7, 11. The neural model should either:
- Reach equivalent F1 with fewer traces (sample efficiency), or
- Exceed sc-infer's ceiling F1 (structural generalization)

If neither holds, the neural architecture is not adding value over
counting.

**Metrics:**
- Edge F1 vs. trace count (convergence curve)
- Per-family traces-to-F1=1.0 (sample complexity)
- Fraction of charts with unobservable structure (ceiling analysis)
- Neural model gap: (neural F1 - sc-infer F1) at matched trace budget

**Success bar:** sc-infer provides a meaningful baseline that the
neural model can beat. Specifically:
- sc-infer should converge on flat charts quickly (K≤50)
- sc-infer should struggle with history/deep/inter-level (K>200)
- The neural model should close the gap faster on hard families

If sc-infer achieves F1=1.0 on everything at K=10, the topology
learning problem is trivial and the neural approach needs a harder task.

**Dependencies:** `sc-infer` tool at `cmd/sc-infer/`, validated corpus
at `corpus/validated-manifest.json` (49,100 charts).

---

## Experiments evaluated and folded into existing work

The following proposed experiments overlap substantially with existing
sc-ml experiments and are better handled as extensions:

**`exp_partial_observability_belief_tracking`**: Near-duplicate of
`exp_partial_observability` (Bayesian belief updates via AND-state
regions) and `exp_runtime_trace_belief_probe` (discriminative probes).
The novel angle — comparing typed-memory vs. hybrid-memory under
masking — should be added as a regime in `exp_hull_ssm_hybrid_memory`.

**`exp_history_semantics_stress`**: History is the most thoroughly
covered area in sc-ml (5+ experiments including `exp_deep_history`,
`exp_history_states_comprehensive`, `exp_history_transfer`). The novel
stress cases (deep nesting >3 levels, parallel-region history corner
cases) should be added as test cases in `exp_mamba_hull_hybrid_executor`.

**`exp_patch_preservation`**: sc-ml has 6+ patch experiments. The novel
angle — formal behavioral equivalence pre/post-patch — should be added
as a metric to `exp_runtime_unified_patch_search` rather than a
standalone experiment.

## Experiments with narrower scope notes

**`exp_parallel_region_interleaving`**: `exp_parallel_region_coupling`
already tests independence vs. sync. The novel angle is step-ordering
semantics. If needed, scope to: does interleaving order affect
joint-config exactness? Add to Wave 2 only if AND-state coverage
proves insufficient in `exp_mamba_hull_hybrid_executor`.

**`exp_curriculum_scaling_topologies`**: `exp_scalability_benchmark`
covers performance scaling but not curriculum ordering. Worth adding to
Wave 2 as a contingency if topology learning proves unstable in Wave 1.
Compare easy→hard progression (flat→nested→orthogonal→inter-level) vs.
mixed training on sample efficiency.

**`exp_active_trace_selection`**: `exp_active_regex_learning` covers
active learning for synthesis (38-42% fewer examples). The novel angle
is active selection of which charts or traces to generate next for
training. Worth adding to Wave 3 if real trace collection becomes
expensive.

---

## Run Order

### Current Landing Status In `sc-ml`

**Updated 2026-03-24 (pass 4 — full scan, honest maturity assessment).**

All 17 existing experiments have benchmark.py and results.json, but
**none are paper-ready on real data yet.** Most results are on
synthetic probes or tiny corpora. The stately corpus (49K validated
charts) is now available to upgrade them.

- Wave 0 diagnostics: **COMPLETE.** All 5 have results. Two are
  paper-ready (`exp_temperature_schedule_sensitivity`,
  `exp_guard_predicate_interpretability`). One has an unresolved
  deep-history divergence (`exp_semantics_agreement_oracle`).
- Wave 1 architecture core: **ALL HAVE RESULTS, ALL SYNTHETIC.**
  Every experiment has results.json but on synthetic fixtures or
  tiny corpora. Key gaps: ablation rungs 5-6 are wrappers, topology
  optimization is Phase 1 flat-only, chart family generalization
  fails on held-out families, GRPO threshold metric is noisy.
  `exp_observability_baseline` is new (pending `sc-infer` tool).
- Wave 2 memory plus decoder bridge: **3/6 SYNTHETIC RESULTS.**
  Hull-SSM, constrained synthesis, and Mamba executor all have
  results but on synthetic workloads, not trace-backed benchmarks.
  `exp_smdp_option_boundaries` is proof-of-mechanism only.
  `exp_uncertainty_calibration`, `exp_synthetic_to_real_transfer`
  not yet created.
- Wave 3 systems plus extensions: **SCAFFOLD ONLY.**
  `exp_delegation_interactive` has a demo harness only.
  `exp_typed_memory_hull_transfer` is blocked on paired replay corpus.

### Training infrastructure status (2026-03-24)

Phase 0 and Phase 1 from READINESS.md are complete:
- NaN gradients fixed (one-hot via `mx.eye` replaces `int()` indexing)
- Gumbel-Softmax straight-through estimator fixed (was returning zeros)
- Graph accumulation fixed (removed duplicate `loss_fn` call in
  `train_step`)
- GRPO/SDPO NaN bugs fixed (same one-hot fix)
- `evaluate_topology()` with precision/recall/F1 added to Trainer
- `make_splits()` for deterministic seeding added
- `load_trace_dataset()` and `Vocabulary` for sc-trace-gen data added
- Baseline validated: loss decreases, accuracy above random, Hull 100%

### Wave 0 (diagnostics) — COMPLETE

1. `exp_semantics_agreement_oracle` — trace-backed, first divergence on
   history re-entry semantics (`history_restore_stress`)
2. `exp_trace_detail_ablation` — trace-backed across 4 fixture families
   (guarded, branching, history, parallel)
3. `exp_temperature_schedule_sensitivity` — edge F1=0.71, collapse
   detection working across schedule families
4. `exp_guard_predicate_interpretability` — guard accuracy=96.9%,
   boundary accuracy=91.2%, OOD accuracy=99%, crossing error=0.02
5. `exp_routing_analysis` — learned routing beats fixed baselines on
   mixed regime (~70% vs ~64%), query-conditioned split confirmed

**Exit condition MET:** Executors agree with Go semantics (with known
history re-entry divergence), standard detail is cost-effective, topology
learning is schedule-sensitive but does not collapse with exponential or
cosine schedules, learned guards recover ground-truth boundaries with
high fidelity, and HybridMemory routing is genuinely query-dependent.

### Wave 1 (architecture core) — ALL IMPLEMENTED, SYNTHETIC RESULTS

All experiments have benchmark.py + results.json in sc-ml, but all
results are on synthetic probes or tiny corpora (3-16 charts). None
are paper-ready. The 49K-chart stately corpus is now available to
upgrade them. Stately converter fixed to preserve history states
(926 charts, commit e2158c2). sc-trace-gen enhanced with inter-level
metadata and history activation telemetry (commit 92fb454).

**Updated 2026-03-24 (pass 3).**

6. `exp_neurosymbolic_ablation_ladder` — Tensor-native MLX forward
   passes for rungs 0-4 (linear baseline, SoftTopologyPolicy,
   Gumbel, guards, HullKVCache). Rungs 5-6 (SSM, delegation) still
   wrapper paths (Wave 2/3 components). Regime-specific results: flat
   dominates cycle (0.8), structured dominates branching (1.0).
   **Status:** Core story (rungs 0-4) is real. 16-chart corpus up
   from 3. **Needs:** Re-run on full stately corpus (~48K charts).
7. `exp_hull_topology_gumbel` — Corpus-backed, 16 charts / 48 rows.
   fixed_soft acc=0.917 F1=0.611, gumbel_dense_guarded acc=0.880
   F1=0.611, gumbel_hull_full acc=0.667 F1=0.444. **Needs:** Larger
   corpus with history/orthogonal diversity.
8. `exp_topology_policy_optimization` — Phase 1 flat comparison.
   Best: hybrid_search_then_finetune acc=0.470 F1=0.364. Hindsight
   rows 0.594-0.606 (not primary comparison). Flat-only scope.
9. `exp_grpo_sample_efficiency` — **10-seed sweep completed.**
   Corrected interpretation: SDPO direct is ~3.7x faster to threshold
   than AdamW (mean 240 vs 883 samples) with identical final accuracy
   (0.688) and edge F1 (1.000). Both have high absolute variance
   (CV≈0.96-0.98) due to threshold metric sensitivity, NOT method
   instability. GRPO/REINFORCE are broken: 0.0 edge F1, ~31% accuracy
   (near-random), only 60% convergence rate. The benchmark's reward
   signal may be too sparse for policy gradient methods.
   **Claim:** SDPO is faster to threshold than AdamW with same final
   quality. GRPO/REINFORCE fail on this task.
10. `exp_negative_edge_supervision` — `plus_no_ops` best at F1=0.600
    vs positive-only 0.333. Negative evidence clearly helps. **Needs:**
    Re-run on larger corpus.
11. `exp_chart_family_generalization` — Improved from F1=0.285 to
    F1=0.634 with shared-axis denoiser (old centroid approach was
    broken by design). Still fails on deep_nested, orthogonal,
    inter_level. **Status:** Partial negative result — shallow
    families transfer, deep/parallel do not. The stately corpus
    (~48K charts) may help by providing organic cross-family variation.

12. `exp_observability_baseline` — **NEW.** Non-neural topology
    inference baseline using `sc-infer` (see exp 20 below). Measures
    traces-to-convergence for counting-based reconstruction per chart
    family. Provides observability ceiling and non-neural baseline for
    all topology experiments. **Status:** Pending `sc-infer` tool.

**Exit condition:** NOT MET. All results are synthetic-only:
- Ablation ladder rungs 0-4 are tensor-native but on tiny corpus
- SDPO is faster than AdamW on synthetic task (noisy threshold)
- Negative evidence helps on synthetic (F1 doubles)
- Cross-family generalization fails on held-out families
- No non-neural baseline exists (exp 20 pending)
**To meet exit condition:** Re-run on stately corpus, build sc-infer
baseline, resolve held-out family failures.

**Remaining evidence gaps:**
1. Re-run exps 6, 7, 11 on the full stately corpus (~48K valid charts)
   once re-conversion completes
2. Diagnose exp 11 deep_nested/orthogonal/inter_level failures with
   real stately charts (organic variation may help vs synthetic)
3. Investigate GRPO/REINFORCE failure in exp 9 (sparse reward?
   insufficient budget? broken training loop?)
4. Build `sc-infer` and run exp 12 observability baseline — provides
   non-neural ceiling and per-family sample complexity curves

### Wave 2 (memory + decoder bridge)

12. `exp_hull_ssm_hybrid_memory`
13. `exp_constrained_synthesis_lm`
14. `exp_mamba_hull_hybrid_executor`
15. `exp_smdp_option_boundaries`
16. `exp_uncertainty_calibration`
17. `exp_synthetic_to_real_transfer`

**Exit condition:** Memory backends are validated against sc-ml
baselines, the GrammarStatechart is tested inside a real LM loop,
model confidences are calibrated, and the synthetic-to-real gap is
measured.

**Negative-result contingency:** If HybridMemory fails to beat simpler
memory baselines, keep Hull and SSM claims separate. If the decoder
only helps on toy or mocked LM paths, do not present it as a real-model
result. If synthetic-to-real gap is large, downstream claims need to
acknowledge it.

**Contingency experiments** (add if Wave 1 surfaces instability):
- `exp_curriculum_scaling_topologies`
- `exp_parallel_region_interleaving` (scoped to step-ordering)

### Wave 3 (systems + extensions)

18. `exp_delegation_interactive`
19. `exp_typed_memory_hull_transfer`

**Optional follow-on:** If Wave 1 looks good, extend the differentiable
topology model to hierarchy / history and rerun
`exp_topology_policy_optimization` as a full search-space comparison.

**Contingency experiment** (add if trace collection cost is high):
- `exp_active_trace_selection`

**Exit condition:** Delegation is measured on a real task family, and
the topology story is explicit about flat versus hierarchical scope.

**Negative-result contingency:** If the learned delegation policy does
not beat the scaffold, keep delegation out of the first-paper core and
treat it as a follow-on systems result.

---

## Paper Implications

If Wave 0 shows semantics agreement and stable diagnostics, we can
trust downstream results.

If Wave 1 succeeds, the paper can make a credible integrated-architecture
claim for the state-transition stack. The centerpiece table is the
ablation ladder, not a bundle of isolated component wins. Negative-edge
supervision and chart-family generalization strengthen the methodology
story.

If Wave 2 succeeds, the paper adds two important bridges: a real LM
decoder result and a concrete Hull + SSM composition result. Calibration
and synthetic-to-real results ground the confidence claims.

Wave 3 is follow-on. Delegation and transfer matter for the broader RLM
story, but they should not be used to over-claim the first paper.

Until Phase 2 of topology optimization is complete, the paper should be
careful to say "flat differentiable topology" rather than implying a
full replacement for hierarchical topology search.
