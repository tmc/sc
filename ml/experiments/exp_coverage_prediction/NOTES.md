# exp_coverage_prediction: Research Notes

## Overview

This experiment investigates whether statechart-based representations of programs outperform flat representations for predicting code coverage. The core hypothesis: coverage prediction is equivalent to "which states will be visited?" - a natural fit for statechart simulation.

## Key Insights

### 1. Learned Structure Outperforms Hand-Designed Structure

The most significant finding: **evolved statecharts beat hardcoded CFG conversion**.

| Approach | F1 | States | Insight |
|----------|-----|--------|---------|
| Evolved | 0.703 | 5 | Learned optimal abstraction |
| Hardcoded CFG | 0.691 | ~10 | Over-engineers the structure |
| State Discovery | 0.674 | 30 | Too fine-grained |

The 4-5 state evolved model is simpler AND more accurate than the complex CFG-based approach. This suggests:
- CFG structure has unnecessary detail for coverage prediction
- The right abstraction level matters more than structural fidelity
- Evolution finds the "just right" granularity

### 2. Precision vs Recall Tradeoff

```
Hardcoded: Precision=0.559, Recall=0.977 (over-predicts)
Evolved:   Precision=0.698, Recall=0.774 (balanced)
```

Hardcoded CFG traces all reachable paths, leading to high recall but many false positives. Evolved models learn which paths are *actually taken* for given inputs.

### 3. State Discovery from Traces Works

Line clustering with >85% co-occurrence threshold finds meaningful state groupings:
- Lines that always execute together → basic blocks
- Lines that sometimes execute together → branch alternatives
- Significant line number jumps → phase boundaries

### 4. Guard Synthesis is Hard

Template-based guard synthesis had limited success. The search space is large and many programs have guards that don't fit simple templates like `x > 0` or `len(items) > 1`.

## What Worked

1. **Observation-seeded evolution**: Initializing population from observed coverage patterns dramatically improved convergence (F1 jumped from ~0.4 to ~0.6 in generation 0)

2. **Line co-occurrence clustering**: Simple but effective for discovering basic blocks

3. **Parsimony pressure**: Small penalty for number of states (0.01 * n_states) helped evolution find simpler models without hurting accuracy

4. **Coverage-based fitness**: Using F1 score directly as fitness aligned evolution with the actual goal

5. **Elitism**: Preserving top 5 genomes each generation prevented losing good solutions

## What Didn't Work

1. **Random initialization**: Randomly assigning lines to states produced genomes with ~0 F1 because line assignments didn't match actual programs

2. **Pure state discovery without evolution**: Discovered 30+ states, too fine-grained, lost the forest for the trees

3. **Complex guard templates**: Templates like `all(x > 0 for x in items)` rarely matched actual program conditions

4. **Single-program statecharts**: Each program has different structure; generic statecharts work better than per-program ones (counterintuitive but true for this dataset)

5. **High mutation rates**: Rates >0.5 destabilized good solutions; 0.3 worked well

## Future Directions

### Short-term

1. **Hierarchical state discovery**: Discover composite states (nested statecharts) for programs with nested control flow

2. **Guard learning with neural networks**: Replace template-based synthesis with learned guard predictors

3. **Cross-program transfer**: Can a statechart evolved on one program type generalize to others?

4. **History states**: Detect and model history-dependent behavior (e.g., initialization that only runs once)

### Medium-term

1. **Differentiable statecharts**: Make the entire statechart differentiable for end-to-end training (started in `DifferentiableStatechartPredictor` but not fully integrated)

2. **Attention over states**: Use transformer attention to learn which states are relevant for given inputs

3. **Active learning**: Generate inputs that maximize coverage uncertainty to improve training efficiency

### Long-term

1. **Statechart extraction from LLMs**: Can we extract program statecharts from LLM latent representations?

2. **Compositional statecharts**: Learn reusable statechart components (e.g., "loop pattern", "error handling pattern")

3. **Multi-modal coverage**: Extend to other coverage metrics (branch coverage, path coverage, MC/DC)

## Connections to Other Experiments

### exp_guard_synthesis
- This experiment could use learned guards from exp_guard_synthesis
- Guard synthesis is a key bottleneck; better guards → better evolved statecharts
- Shared abstraction: both use positive/negative examples for learning

### exp_topology_evolution
- Directly inspired the `statechart_evolver.py` implementation
- Key difference: here we evolve for coverage prediction fitness, not structural validity
- Could share genome representation and mutation operators

### exp_sae_statechart
- SAE features could identify monosemantic "states" in program behavior
- Potential hybrid: use SAE to discover state features, then cluster into discrete states
- SAE might find states that line clustering misses (semantic vs syntactic)

### exp_code_completion
- Coverage prediction could inform code completion (predict which branches are likely)
- Shared interest in program understanding
- Could combine: statechart-aware code completion

## Experimental Setup Notes

### Dataset
- 273 examples from 41 program templates
- 14 categories (SIMPLE_BRANCH, MULTI_BRANCH, LOOP, NESTED, etc.)
- Each example: (source, input, covered_lines, execution_trace)

### Evolution Parameters
- Population: 30
- Generations: 50
- Elite size: 5
- Mutation rate: 0.3
- Crossover rate: 0.7
- Tournament size: 3

### Key Files
```
coverage_collector.py   - Collect (program, input, coverage) triples
program_statechart.py   - CFG-to-statechart conversion
baseline_models.py      - Neural baselines (Seq, AST, GNN)
statechart_predictor.py - Symbolic execution predictor
benchmark.py            - Evaluation metrics
dataset.py              - Comprehensive dataset generator
state_discovery.py      - Discover states from traces
transition_learner.py   - Learn transitions and guards
statechart_evolver.py   - Evolutionary statechart discovery
discovery_benchmark.py  - Compare discovery approaches
```

## Reproducibility

```bash
cd ml
source .venv/bin/activate

# Run discovery benchmark
python -c "from experiments.exp_coverage_prediction.discovery_benchmark import demo; demo()"

# Run evolution comparison
python -c "from experiments.exp_coverage_prediction.statechart_evolver import demo; demo()"

# Full benchmark
python -c "from experiments.exp_coverage_prediction.benchmark import demo; demo()"
```

## References

- Harel, D. (1987). Statecharts: A visual formalism for complex systems
- Symbolic execution literature for coverage prediction
- Genetic programming for program synthesis
