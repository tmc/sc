# exp_trm_sc_refinement: TRM-style Statechart Refinement

## Overview

Uses TinyRecursiveModels' H×L iteration dynamics to transform
partial/invalid statecharts into valid ones.

**Target:** Turn 60% valid initial SCs into 95%+ valid after refinement.

## Key Ideas

### TRM Iteration for Refinement

Apply TRM's two-level recursion to statechart repair:
- H_cycles (3): Global structure refinement (hierarchy, initial states)
- L_cycles (6): Local detail refinement (transitions, determinism)

Each iteration progressively fixes validity issues:
1. Encode current SC state
2. Apply attention between states and transitions
3. Predict corrections
4. Update embeddings for next iteration

### Validity Dimensions

Track multiple validity aspects:

| Dimension | Description | Fix Strategy |
|-----------|-------------|--------------|
| Hierarchy | Proper parent-child structure | Re-predict parent indices |
| Initial states | One per composite | Adjust is_initial predictions |
| Transitions | Valid source/target | Correct invalid references |
| Determinism | No conflicts | Remove/merge duplicates |
| Reachability | No orphan states | Add connecting transitions |

## Components

### sc_refiner.py
Core refinement model:
- `StateEncoder`: Embed state types, depths, labels
- `TransitionEncoder`: Embed source, target, events
- `RefinementBlock`: Self-attention + cross-attention
- `RefinementHead`: Predict corrections
- `SCRefiner`: Full H×L refinement

### refinement_loop.py
Manages the iteration dynamics:
- `RefinementStep`: Single iteration record
- `RefinementResult`: Full run results
- `RefinementLoop`: H×L iteration with early stopping
- Callbacks for monitoring

### validity_tracker.py
Track validity during refinement:
- `ValidityMetrics`: Comprehensive validity checks
- `ValidityHistory`: Track improvement over iterations
- `ValidityTracker`: Compute all validity dimensions

### benchmark.py
Evaluate refinement performance:
- Generate partially-valid SCs
- Run refinement
- Measure improvement
- Generate reports

## Usage

```bash
# Run full benchmark
python -m ml.experiments.exp_trm_sc_refinement.benchmark

# Test individual components
python -m ml.experiments.exp_trm_sc_refinement.sc_refiner
python -m ml.experiments.exp_trm_sc_refinement.refinement_loop
python -m ml.experiments.exp_trm_sc_refinement.validity_tracker
```

## Architecture

```
Input SC (60% valid)
       │
       ▼
┌──────────────────┐
│  State Encoder   │──┐
└──────────────────┘  │
                      │
┌──────────────────┐  │
│  Trans Encoder   │──┼──► Embeddings
└──────────────────┘  │
                      │
       ┌──────────────┘
       │
       ▼
┌──────────────────────────────────┐
│         H×L Refinement           │
│  ┌────────────────────────────┐  │
│  │  H-cycle (global context)  │  │
│  │  ┌──────────────────────┐  │  │
│  │  │  L-cycle (local fix) │×6│  │
│  │  └──────────────────────┘  │  │
│  └────────────────────────────┘×3│
└──────────────────────────────────┘
       │
       ▼
┌──────────────────┐
│ Refinement Head  │
└──────────────────┘
       │
       ▼
Output SC (95%+ valid)
```

## Key Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| Initial validity | % valid before refinement | ~60% |
| Final validity | % valid after refinement | ≥95% |
| Mean improvement | Avg validity increase | ≥35% |
| Convergence rate | % reaching target | ≥90% |
| Convergence iter | Avg iterations to converge | <12 |

## Expected Results

1. **Rapid early improvement**: First H-cycle fixes major issues
2. **Diminishing returns**: Later iterations for edge cases
3. **Hierarchy fixes first**: Root/parent issues resolved early
4. **Transition fixes later**: Fine-grained corrections in L-cycles

## Integration with SC Repo

```python
from ml.experiments.exp_trm_sc_refinement import (
    SCRefiner,
    SCRefinerConfig,
    run_refinement,
    run_benchmark,
)

# Load/encode a statechart
sc_data = encode_statechart(my_statechart)

# Refine
result = run_refinement(model, sc_data)
print(f"Validity: {result.input_validity:.2%} → {result.output_validity:.2%}")
```

## References

- TinyRecursiveModels: H×L iteration structure
- SC validation: `/validation/v1/`
- SC semantics: `/semantics/v1/`
