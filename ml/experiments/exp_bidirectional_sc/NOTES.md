# exp_bidirectional_sc

## Goal
Train model to BOTH generate SCs from descriptions AND explain SCs in natural language.

## Hypothesis
Bidirectional training improves both directions through shared representation learning.

## Directions
| Direction | Input | Output |
|-----------|-------|--------|
| Forward | Natural language description | SC JSON |
| Reverse | SC JSON | Natural language explanation |

## Metrics
- **gen_validity**: % of generated SCs that are valid JSON with required structure
- **explain_clarity**: % of explanations that coherently describe the SC

## Dataset
Uses exp_sc_generation_benchmark's 100 tasks as test set.
Each task provides a (description, gold_sc) pair for bidirectional evaluation.

## Usage
```python
from experiments.exp_bidirectional_sc import run_benchmark, format_report

result = run_benchmark()
print(format_report(result))
# BIDIRECTIONAL: gen_validity=X%, explain_clarity=X%
```

## Report Format
```
[C9F0]: BIDIRECTIONAL: gen_validity=X%, explain_clarity=X%
```
