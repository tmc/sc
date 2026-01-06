# exp_path_length_prediction

## Hypothesis
Given a statechart and initial context, a small LLM (1.5B) can predict the number of state transitions needed to reach a terminal (final) state.

**Key finding: Scratchpad prompting dramatically improves accuracy from 17% to 83%.**

## Approaches Tested

| Approach | Description | Accuracy |
|----------|-------------|----------|
| Baseline | Direct prediction | 17% |
| **Scratchpad** | Step-by-step tracing | **83%** |
| BFS Simulation | Teach BFS algorithm | 33% |
| ASCII Visualization | Graph structure in prompt | 33% |

## Scratchpad Results (2026-01-05)

| Test Case | Expected | Predicted | Result |
|-----------|----------|-----------|--------|
| Linear A→B→C→D | 3 | 3 | ✓ |
| Start→Middle→End | 2 | 2 | ✓ |
| Branch Check→Done | 2 | 2 | ✓ |
| Counter (target=3) | 5 | 4 | ✗ (off by 1) |
| Infinite loop A↔B | infinite | infinite | ✓ |
| Single step Off→On | 1 | 1 | ✓ |

**Overall: 83% exact, 100% within ±1**

Report sent to B90CCCD4:
```
[91C6]: PATH_SCRATCHPAD baseline=17%, scratchpad=83%, bfs_sim=33%, ascii_viz=33%
```

## Why Scratchpad Works

The scratchpad prompt forces explicit step enumeration:

```
Let me trace the path step by step:
- Start at A
- Step 1: A --e1--> B (now at B)
- Step 2: B --e2--> C (now at C)
- Step 3: C --e3--> D (now at D, terminal!)
Path length: 3
```

This helps because:
1. **Explicit counting** - Model counts steps as it traces
2. **State tracking** - "now at X" maintains position awareness
3. **Terminal detection** - "terminal!" signals when to stop
4. **Few-shot alignment** - Examples match output format exactly

## What Doesn't Work

### BFS Simulation (33%)
- Too algorithmic for text-based reasoning
- Model struggles with queue/visited tracking
- Often miscounts or gets stuck

### ASCII Visualization (33%)
- Visual structure doesn't translate to counting
- Model can see the graph but can't traverse it
- Sometimes guesses based on visual density

## Key Findings

1. **Scratchpad >> Everything else**: 83% vs 17-33% for other methods
2. **Within ±1 is achievable**: 100% accuracy within ±1 tolerance
3. **Loops remain hard**: Counter loops off-by-one (5 expected, 4 predicted)
4. **Infinite detection works**: Model correctly identifies unreachable states

## Files

| File | Purpose |
|------|---------|
| `scratchpad_predictor.py` | Explicit reasoning prompts |
| `path_visualizer.py` | ASCII graph generation |
| `benchmark_scratchpad.py` | Compare all approaches |
| `length_predictor.py` | Original baseline predictor |
| `benchmark.py` | Original benchmark |

## Usage

```bash
# Run scratchpad benchmark
cd /Volumes/tmc/go/src/github.com/tmc/sc/ml
.venv/bin/python3 -m experiments.exp_path_length_prediction.benchmark_scratchpad
```

```python
# Programmatic usage
from experiments.exp_path_length_prediction import (
    run_scratchpad_benchmark,
    format_scratchpad_report,
)

result = run_scratchpad_benchmark()
print(format_scratchpad_report(result))
```

## Next Steps

1. **Fix counter loops**: Improve loop iteration counting
2. **Test larger models**: 3B/7B for complex paths
3. **GRPO fine-tuning**: Train specifically on path tracing
4. **Combine with guard prediction**: Use guard outcomes for variable paths
