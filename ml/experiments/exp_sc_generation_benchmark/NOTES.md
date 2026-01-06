# exp_sc_generation_benchmark

## Goal
Standardized 100-task benchmark for evaluating SC generation models.

## Categories (20 tasks each)
| Category | Difficulty | Features |
|----------|-----------|----------|
| Simple | 1-2 | 3-5 states, basic transitions |
| Hierarchical | 2-3 | Nested composite states, 2-3 levels |
| Parallel | 3 | Orthogonal regions, concurrent states |
| Guards | 3-4 | Conditional transitions, boolean guards |
| Complex | 4-5 | All features combined |

## Evaluation Metrics
- **Structural (40%)**: State/transition overlap with gold standard
- **Semantic (20%)**: Name and event similarity
- **Feature (20%)**: Required features present (hierarchy, parallel, guards)
- **Validity (20%)**: Well-formed statechart structure

## Usage
```python
from experiments.exp_sc_generation_benchmark import run_benchmark

def my_generator(prompt: str) -> str:
    # Return JSON statechart
    return '{"root_state": {...}}'

result = run_benchmark(my_generator)
print(result.summary())
```

## Pass Criteria
- Overall score >= 50% to pass individual task
- Target: 70%+ pass rate for production-ready models

## Files
- `benchmark_tasks.json`: 100 tasks with prompts and metadata
- `gold_standards/`: Validated gold standard SCs
- `evaluator.py`: Comparison and scoring logic
