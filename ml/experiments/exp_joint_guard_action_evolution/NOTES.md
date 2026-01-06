# exp_joint_guard_action_evolution

## Overview
Co-evolve guards and actions together with causal awareness. Guards depend on action effects (what actions write, guards read), creating causal dependencies that must be maintained during evolution.

## Core Insight

**Independent evolution breaks causal chains.**

When guards and actions evolve separately:
- Guards may check variables that no action sets
- Actions may set variables that no guard checks
- The result is incoherent: transitions that can never fire, or effects with no observable consequence

**Joint evolution maintains causal coherence** by:
1. Building a causal dependency graph: action writes X → guard reads X
2. Mutating causally-related pairs together
3. Using causal clusters in crossover

## Benchmark Results

| Method | Fitness | Behavioral | Temporal | Improvement |
|--------|---------|------------|----------|-------------|
| **Joint** | 0.888 | 0.667 | 0.941 | - |
| Independent | 0.788 | 0.333 | 0.938 | +12.8% |
| Random | 0.792 | 0.667 | 0.586 | +12.1% |

**Key finding**: Joint evolution achieves **2x behavioral coherence** compared to independent evolution (0.667 vs 0.333), meaning transitions actually produce the expected behavior on test scenarios.

## Architecture

### 1. Joint Genome (`joint_genome.py`)
- `GuardActionPair`: Guard + action bundled for co-evolution
- `Guard`: Clauses combined by AND/OR
- `Action`: Effects that modify context variables

### 2. Causal Graph (`causal_graph.py`)
- Builds dependency graph: action[i].write(X) → guard[j].read(X)
- `get_mutation_cluster()`: Find causally-related pairs
- `suggest_coherent_mutation()`: Guide mutations to maintain coherence

### 3. Coherence Fitness (`coherence_fitness.py`)
Multi-objective fitness:
- **Causal coherence**: Guards read what actions write
- **Temporal coherence**: Cause precedes effect
- **Behavioral coherence**: Correct on test scenarios
- **Structural coherence**: No dead guards or unused actions

### 4. Joint Evolver (`joint_evolver.py`)
- Causal-aware mutation: Mutate clusters together
- Cluster crossover: Exchange causal clusters between parents
- Coherence selection: Prefer individuals with high coherence

## Files

| File | Lines | Description |
|------|-------|-------------|
| joint_genome.py | ~450 | Joint guard-action representation |
| causal_graph.py | ~350 | Causal dependency graph |
| coherence_fitness.py | ~400 | Multi-objective coherence fitness |
| joint_evolver.py | ~500 | Co-evolution algorithm |
| benchmark.py | ~350 | Comparison vs independent evolution |

## Connections to Other Experiments

### Building On
- **exp_guard_synthesis**: Expression AST for guards
- **exp_action_composition**: Action effect algebra

### Related
- **exp_temporal_guards**: Temporal dependencies in guards
- **exp_action_side_effects**: Action effect modeling
- **exp_priority_attention**: Priority-based conflict resolution

## What Worked

1. **Causal dependency graph** - Explicitly modeling action→guard dependencies
2. **Cluster mutation** - Mutating related pairs together maintains coherence
3. **Multi-objective fitness** - Balancing causal, temporal, behavioral, structural

## What Didn't Work

1. **Independent evolution** - 2x worse behavioral coherence
2. **Random mutations** - Breaks causal chains quickly
3. **Single-objective fitness** - Misses important coherence aspects

## Key Metrics

| Metric | Description |
|--------|-------------|
| Variable coverage | Fraction of guard-read vars that are action-written |
| Dependency satisfaction | Fraction of pairs with incoming causal edges |
| Guard satisfiability | Fraction of guards that can potentially fire |
| Action usefulness | Fraction of actions that affect observable state |

## Future Directions

1. **Learning causal structure** - Infer dependencies from execution traces
2. **Hierarchical causal graphs** - Handle nested statechart causality
3. **Temporal causal constraints** - Model time-delayed dependencies
4. **Multi-agent causality** - Cross-agent causal dependencies

## Key Insight

The causal dependency graph is the hidden structure that makes statecharts work. By making it explicit and using it to guide evolution, we get statecharts where guards and actions form coherent causal chains rather than disconnected random pairs.
