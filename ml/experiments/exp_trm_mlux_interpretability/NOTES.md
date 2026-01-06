# exp_trm_mlux_interpretability: TRM Reasoning Analysis with mlux

## Overview

Analyzes TRM-style iterative reasoning in the Sudoku statechart model
using mlux-inspired hooks and activation caching.

## Key Questions

1. How do representations evolve during H×L iterations?
2. Which iterations are most critical for solving?
3. What distinguishes correct from incorrect solutions?
4. Can we identify early indicators of success/failure?

## Components

### iteration_analyzer.py
- Captures snapshots at each (H, L) iteration
- Tracks entropy, confidence, accuracy trajectories
- Finds convergence points
- Supports custom hooks

### activation_tracker.py
- mlux-style activation caching
- Tracks representation drift between iterations
- Computes similarity evolution
- Sparsity analysis

### solution_correlator.py
- Compares correct vs incorrect solutions
- Finds statistically significant correlations
- Identifies differentiating features
- Computes effect sizes

### benchmark.py
- Full interpretability benchmark
- Generates comprehensive reports
- Saves JSON + Markdown outputs

## Usage

```bash
# Run full benchmark
python -m ml.experiments.exp_trm_mlux_interpretability.benchmark

# Test individual components
python -m ml.experiments.exp_trm_mlux_interpretability.iteration_analyzer
python -m ml.experiments.exp_trm_mlux_interpretability.activation_tracker
python -m ml.experiments.exp_trm_mlux_interpretability.solution_correlator
```

## Integration with exp_trm_sudoku_9x9

```python
from ml.experiments.exp_trm_sudoku_9x9.sudoku_statechart_9x9 import SudokuStatechart9x9
from ml.experiments.exp_trm_mlux_interpretability import (
    IterationAnalyzer,
    ActivationTracker,
    SolutionCorrelator,
    run_interpretability_benchmark,
)

# Load trained model
model = SudokuStatechart9x9(hidden_dim=128, H_cycles=3, L_cycles=6)
# ... load weights ...

# Run analysis
result = run_interpretability_benchmark(model, questions, answers)
print(result.findings)
```

## Key Metrics

| Metric | Description |
|--------|-------------|
| Entropy trajectory | How uncertainty decreases over iterations |
| Confidence trajectory | How max probability increases |
| Convergence iteration | When predictions stabilize |
| Representation drift | L2 distance between consecutive iterations |
| Effect size | Cohen's d for correct vs incorrect |

## Expected Findings

1. **Early divergence**: Correct and incorrect solutions diverge early (iteration ~6)
2. **Entropy monotonic**: Entropy should decrease monotonically for well-trained models
3. **Guard activation**: Higher guard activation correlates with correctness
4. **Convergence speed**: Correct solutions converge faster

## References

- [mlux library](https://github.com/batson/mlux) - MLX interpretability
- [MLX framework](https://github.com/ml-explore/mlx) - Apple's ML framework
- TinyRecursiveModels paper - Iterative refinement approach
