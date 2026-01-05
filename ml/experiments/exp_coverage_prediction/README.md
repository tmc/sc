# exp_coverage_prediction: Statechart-Based Code Coverage Prediction

## Goal
Train a model to predict which lines of code will be covered given a program and input.
Compare statechart-based approach vs baseline neural methods.

## Hypothesis
Programs represented as statecharts (control flow = transitions) will outperform
flat sequence models for coverage prediction because:
1. State machines capture control flow structure explicitly
2. Guards model branch conditions naturally
3. Transitions track reachable code paths

## Approaches to Compare

### Baseline Methods
1. **Sequence Model**: Encode (program, input) → predict coverage vector
2. **AST-based**: Parse to AST, embed nodes, predict per-node coverage
3. **Graph Neural Network**: Build CFG, message passing, node classification

### Statechart Methods
1. **Statechart Traversal**:
   - Convert program to statechart (states = basic blocks)
   - Simulate with input to predict active states
   - Coverage = union of visited states

2. **Guard-Based Prediction**:
   - Learn guard expressions for each branch
   - Evaluate guards on input to predict taken branches
   - Use exp_guard_synthesis framework

3. **SAE Coverage Features**:
   - Train SAE on hidden states during execution
   - Discover features that correlate with line coverage
   - Use features as interpretable coverage predictors

## Evaluation

### Metrics
- Precision/Recall for covered lines
- Jaccard similarity between predicted and actual coverage
- F1 score per line, macro-averaged

### Datasets
- Python functions with test inputs + actual coverage (from pytest-cov)
- Starlark build rules with target inputs
- ARC programs with grid inputs

## Key Insight
Coverage prediction = "which states will be visited given this input?"
This is EXACTLY what statechart simulation does.

## Files to Implement
1. `coverage_collector.py` - Collect program+input+coverage triples
2. `program_statechart.py` - Convert Python/Starlark to statechart CFG
3. `baseline_models.py` - Sequence/AST/GNN baselines
4. `statechart_predictor.py` - Statechart traversal coverage model
5. `guard_coverage.py` - Guard synthesis for branch prediction
6. `benchmark.py` - Compare all approaches

## Integration with SOAR
Use SOAR's program archive as training data:
- Programs that solve different tasks have different coverage
- Hindsight relabeling creates (program, input, coverage) triples
- Test: Does coverage prediction improve program selection?

## Expected Results
| Method | Coverage F1 | Notes |
|--------|-------------|-------|
| Sequence | ~0.60 | Misses control flow |
| AST-based | ~0.70 | Better structure |
| GNN on CFG | ~0.75 | Explicit flow graph |
| **Statechart** | **~0.85+** | Natural representation |
| Guard-based | ~0.80 | Interpretable branches |
