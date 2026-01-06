# exp_mlux_sc_circuits: Research Notes

## Goal

Find circuits in transformer models responsible for statechart validity.
Identify which components (attention heads, MLPs) affect which validity aspects.

## Approach

### 1. Validity Measurement

Break down SC validity into measurable components:

| Aspect | What it measures | Score 0-1 |
|--------|------------------|-----------|
| Structural | Valid JSON, required fields | Has root_state, etc |
| State Names | Consistent naming | No duplicates, valid refs |
| Transitions | Valid references | All from/to exist |
| Hierarchy | Proper nesting | Compound has children |
| Semantic | Initial states, reachability | Has initials |

### 2. Ablation Study

For each component:
1. Zero out component output
2. Generate statecharts
3. Measure validity degradation
4. Record which aspects affected

```python
# Ablation types
- ZERO: Set output to zeros
- MEAN: Replace with mean activation
- NOISE: Add Gaussian noise
- RESAMPLE: Use different input's activation
```

### 3. Circuit Identification

Components are assigned to circuits based on what breaks when ablated:

```
STATE_NAME_MEMORY: High state_name_drop when ablated
TRANSITION_VALIDITY: High transition_drop when ablated
HIERARCHY: High hierarchy_drop when ablated
STRUCTURAL: High overall drop but low specific drops
```

### 4. Circuit Graph

Build graph showing:
- Nodes: Critical components
- Edges: Information flow (layer adjacency + shared circuit)
- Colors: Circuit membership

## Implementation

### validity_measurer.py

Fine-grained validity scoring:

```python
class ValidityMeasurer:
    def measure(chart) -> ValidityMetrics:
        # Returns scores for each aspect

metrics = ValidityMetrics(
    structural_validity=0.9,
    state_name_consistency=0.85,
    transition_validity=0.95,
    hierarchy_validity=1.0,
    semantic_validity=0.8,
    total_score=0.88,
)
```

### ablation_runner.py

Systematic ablation:

```python
class AblationRunner:
    def enumerate_components() -> List[ComponentSpec]
    def run_ablation(component, type, prompts) -> ValidityMetrics
    def run_study(prompts, components) -> AblationStudy
```

### circuit_finder.py

Circuit discovery:

```python
class CircuitFinder:
    def find_circuits(study) -> CircuitGraph
    def analyze_layer_roles(graph) -> Dict
    def find_critical_paths(graph, circuit_type) -> List[Path]
```

### benchmark.py

Evaluation:

```python
class CircuitBenchmark:
    def run_benchmark(prompts, layers) -> BenchmarkResult
    def run_full_benchmark() -> BenchmarkSummary
```

## Expected Results

### Layer Roles

Based on transformer architecture:

| Layer Range | Expected Role |
|-------------|---------------|
| Early (0-8) | Structural patterns, JSON syntax |
| Middle (8-16) | State name memory, references |
| Late (16-24) | Transition logic, hierarchy |

### Circuit Patterns

Expected circuit structure:

```
STATE_NAME_MEMORY:
  - Middle layers: store state names
  - Late layers: retrieve for transitions

TRANSITION_VALIDITY:
  - Early: parse transition structure
  - Middle: resolve state references
  - Late: validate connections

HIERARCHY:
  - Early: detect parent-child patterns
  - Late: enforce constraints
```

## Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| Specificity | Components in few circuits | >70% |
| Coverage | Validity explained by circuits | >80% |
| Consistency | Same circuits across runs | >90% |

## Connections

| Experiment | Connection |
|------------|------------|
| exp_grammar_induction | SAE for circuit analysis |
| exp_priority_attention | Attention patterns in circuits |
| exp_qwen_repair | Circuits for error correction |

## Files

```
exp_mlux_sc_circuits/
├── __init__.py           # Package exports
├── validity_measurer.py  # Fine-grained validity metrics
├── ablation_runner.py    # Systematic ablation
├── circuit_finder.py     # Circuit identification
├── benchmark.py          # Evaluation framework
└── NOTES.md              # This file
```

## Usage

```python
from experiments.exp_mlux_sc_circuits import demo
demo()

# Or step by step:
from experiments.exp_mlux_sc_circuits import (
    AblationRunner,
    CircuitFinder,
    default_parser,
)

runner = AblationRunner(model=my_model)
study = runner.run_study(prompts, components)

finder = CircuitFinder(runner)
graph = finder.find_circuits(study)

print(summarize_circuits(graph))
```

## Status

- [x] Validity measurement
- [x] Ablation framework
- [x] Circuit identification
- [x] Graph visualization
- [x] Benchmark framework
- [ ] Real model testing (needs mlux)

## Date

2026-01-04
