# exp_topology_inference

## Hypothesis

Can an LLM (Qwen2.5-Coder-1.5B) infer statechart topology from execution traces?

Given execution traces (sequences of states visited), the model should extract:
1. Set of unique states
2. Transitions (consecutive state pairs)
3. Initial state (first state of first trace)
4. Structural features (self-loops, branching, cycles)

## Results

**Overall: 82%** (exceeds 50% target)

| Metric | Enumeration | Simple |
|--------|-------------|--------|
| States | 100% | 100% |
| Transitions | 78% | 74% |
| Initial | 100% | 100% |
| **Overall** | **82%** | 81% |
| Pass rate (≥80%) | 62% (5/8) | 50% (4/8) |
| Avg time | 5.5s | 1.6s |

**By Category (enumeration):**
| Category | Accuracy | Pass |
|----------|----------|------|
| Self-loop | 90% | 1/1 |
| Edge case | 90% | 1/1 |
| Branch | 87% | 2/2 |
| Cycle | 80% | 1/2 |
| Linear | 70% | 0/1 |
| Complex | 70% | 0/1 |

## Key Findings

**What Worked:**
1. State extraction is perfect (100%) - LLM reliably identifies unique states
2. Initial state detection is perfect (100%) - "first state of first trace" rule learned
3. Chain-of-thought enumeration approach outperforms simple prompting
4. Branch patterns detected well (87%)
5. Self-loop detection works reliably (90%)

**What Didn't Work:**
1. Transition extraction adds spurious edges (hallucinated transitions)
2. Linear chains get extra transitions (model infers connectivity)
3. Complex patterns accumulate errors
4. Cycle detection inconsistent (sometimes misses C→A)

## Approach

Step-by-step enumeration with chain-of-thought prompting:

```
INPUT TRACES:
[A, B, C, A]
[A, C, B, A]
[A, B, B, C]

SCRATCHPAD:
Step 1 - Extract unique states:
  Union: {A, B, C}

Step 2 - Extract transitions:
  Trace 1: A→B, B→C, C→A
  Trace 2: A→C, C→B, B→A
  Trace 3: A→B, B→B, B→C
  Union: {A→B, A→C, B→A, B→B, B→C, C→A, C→B}

Step 3 - Identify patterns:
  B→B = self-loop
  A→B, A→C = branching from A
  C→A = cycle back to initial

Step 4 - Initial: A (first state of first trace)

OUTPUT:
States: [A, B, C]
Transitions: [A→B, A→C, B→A, B→B, B→C, C→A, C→B]
Initial: A
Features: [self-loop on B, branching from A, cycle]
```

## Recommendations

1. **Stricter transition parsing**: Only accept transitions explicitly observed in traces
2. **Verification step**: After generating, verify each transition appears in at least one trace
3. **Larger model**: 3B+ model may have better attention span for trace-by-trace analysis
4. **Fine-tuning**: Train on trace→topology pairs for consistent output format

## Report to Orchestrator

```
[9D1B]: TOPOLOGY_INFERENCE states=100%, transitions=78%, overall=82%, by_pattern=[linear:70%, cycle:80%, branch:87%, selfloop:90%]

Summary:
- States: Perfect (100%) - all unique states found
- Transitions: 78% - some hallucinated edges
- Initial: Perfect (100%) - first-of-first rule learned
- Overall: 82% - exceeds 50% target

By pattern:
- Self-loop: 90% - detected reliably
- Edge: 90% - single-state handled
- Branch: 87% - diamond patterns good
- Cycle: 80% - cycle detection mostly works
- Linear: 70% - adds spurious edges
- Complex: 70% - cumulative errors

Best method: enumeration (chain-of-thought)
Simple method: 3.5x faster, 1pp lower accuracy

Files created:
- experiments/exp_topology_inference/__init__.py (test cases, metrics)
- experiments/exp_topology_inference/topology_inferrer.py (LLM inference)
- experiments/exp_topology_inference/benchmark.py (evaluation)
```

## Files

- `__init__.py` - Test cases, evaluation metrics, ground truth computation
- `topology_inferrer.py` - LLM-based inference with enumeration prompting
- `benchmark.py` - Full benchmark with category breakdown
- `NOTES.md` - This documentation

## Usage

```bash
cd /Volumes/tmc/go/src/github.com/tmc/sc/ml
.venv/bin/python3 -m experiments.exp_topology_inference.benchmark
```
