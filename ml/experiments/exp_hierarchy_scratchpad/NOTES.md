# exp_hierarchy_scratchpad

## Hypothesis

Can the scratchpad technique improve hierarchy understanding from 27% to 50%+?
The same technique improved path length (17%→83%) and reachability steps (0%→62%).

**Result: 48% overall (+21% improvement), short of 50% target but significant improvement.**

## Results

### Overall Accuracy

| Metric | Baseline | Scratchpad | Target |
|--------|----------|------------|--------|
| Overall | 27% | **48%** | 50% |
| Improvement | - | +21% | - |

### Per-Task Breakdown

| Task | Accuracy | Notes |
|------|----------|-------|
| Containment | **62%** (5/8) | Parent chain tracing works well |
| Cascade Exit | 17% (1/6) | Recursive enumeration fails |
| Default Entry | 50% (2/4) | Initial markers partially work |
| Depth | **60%** (3/5) | Step counting works |

Report sent to B90CCCD4:
```
[91C6]: HIERARCHY_SCRATCHPAD baseline=27%, scratchpad=48%, by_task=[contain:62%, cascade:17%, entry:50%, depth:60%]
```

## Why Scratchpad Partially Works

### What Worked

1. **Containment (62%)** - Parent chain tracing is effective:
   ```
   Trace UP from B:
   - B's parent is A
   - A's parent is Root
   Chain: B → A → Root
   Is Root in chain? YES
   ```

2. **Depth (60%)** - Step counting from UP direction:
   ```
   Trace UP from B:
   - B's parent is A (count 1)
   - A's parent is Root (count 2)
   Depth = 2
   ```

3. **Default Entry (50%)** - Following * markers DOWN:
   ```
   Follow * DOWN:
   - Root's children: A*, D. Pick A* (has *)
   - A's children: B*, C. Pick B* (has *)
   Active: Root, A, B
   ```

### What Didn't Work

1. **Cascade Exit (17%)** - Model fails to recursively enumerate descendants:
   - Expected: A, B, C (A contains B and C)
   - Predicted: A (only the starting state)
   - The model understands immediate children but doesn't recurse

2. **Deep hierarchies** - All tasks fail on 4+ level hierarchies:
   - Depth off-by-one for deep trees
   - Entry only gets first 2 levels
   - Containment confused by long chains

3. **Direction confusion** - Model still confuses UP (parents) vs DOWN (children):
   - "A inside B?" answered YES (wrong - A is parent of B)
   - "Playing inside Menu?" answered YES (they're siblings)

## Scratchpad Technique Analysis

### Why It Works Better Than Baseline

| Aspect | Baseline | Scratchpad |
|--------|----------|------------|
| Direction | Implicit | Explicit UP/DOWN markers |
| Counting | Mental math | Step-by-step enumeration |
| Format | Free form | Structured "Answer: X" |

### Fundamental Limitations

1. **Recursive enumeration**: LLMs struggle to recursively traverse and collect all descendants
2. **Long chains**: Attention fades over 3+ parent/child links
3. **Sibling confusion**: Model sometimes confuses siblings with ancestors

## Key Findings

### Hierarchy Understanding is Fundamentally Hard

| Technique | Path Length | Reachability Steps | Hierarchy |
|-----------|-------------|-------------------|-----------|
| Baseline | 17% | 0% | 27% |
| Scratchpad | **83%** | **62%** | 48% |
| Improvement | +66% | +62% | +21% |

Hierarchy shows the smallest improvement because:
1. Requires multi-hop reasoning through tree structures
2. Cascade needs recursive collection (not just tracing)
3. Direction semantics (up vs down) are confusing

### Task Difficulty Ranking

1. **Depth** (60%) - Simple counting UP
2. **Containment** (62%) - Boolean check on parent chain
3. **Entry** (50%) - Follow markers DOWN (single path)
4. **Cascade** (17%) - Collect ALL descendants (hardest)

## Gaps Identified

| Gap | Example | Impact |
|-----|---------|--------|
| Recursive collection | Cascade only gets root | 17% cascade |
| Deep trees (4+ levels) | depth(D)=4, predicted=5 | Off-by-one |
| Sibling vs ancestor | A inside B? YES (wrong) | False positives |
| Initial child selection | Paused vs Playing* | Wrong entry |

### Specific Failure Modes

1. **Cascade stops at start**:
   ```
   Input: exit A (A contains B, C)
   Expected: A, B, C
   Predicted: A
   ```

2. **Deep hierarchy confusion**:
   ```
   Input: Root{A{B{C{D}}}}
   Q: depth(D)?
   Expected: 4
   Predicted: 5
   ```

3. **Sibling confusion**:
   ```
   Tree: Root{Menu{...}, Game{Playing, Paused}}
   Q: Is Playing inside Menu?
   Expected: NO (siblings)
   Predicted: YES
   ```

## Recommendations

### Short-term Fixes

1. **Add explicit "STOP at leaf" markers** for cascade
2. **Pre-compute tree depth** and include in prompt
3. **Add sibling disambiguation examples**

### Architectural Improvements

1. **Hybrid approach** - Use LLM for structure extraction, symbolic for traversal
2. **Tree-specialized attention** - Fine-tune on tree traversal tasks
3. **Multi-step prompting** - One step per hierarchy level

### Alternative Approaches

1. **External tree representation** - Build adjacency list, query symbolically
2. **Chunked processing** - Process subtrees independently
3. **Fine-tuning** - Train on (tree, task, answer) triples

## Files

| File | Purpose |
|------|---------|
| `hierarchy_scratchpad.py` | Prompts for 4 task types + parsing |
| `benchmark.py` | Run benchmark across all tasks |
| `__init__.py` | Module exports |
| `NOTES.md` | This documentation |

## Usage

```bash
# Run benchmark
cd /Volumes/tmc/go/src/github.com/tmc/sc/ml
.venv/bin/python3 -m experiments.exp_hierarchy_scratchpad.benchmark
```

```python
# Programmatic usage
from experiments.exp_hierarchy_scratchpad import (
    run_benchmark,
    create_prompt,
    HierarchyTask,
)

result = run_benchmark()
print(f"Overall: {result['overall_accuracy']:.0f}%")
```

## Test Cases

### Hierarchy 1: Simple 3-level
```
Root{A*{B*, C}, D}
```
- Containment: B inside Root? ✓, D inside A? ✓, C inside A? ✓
- Cascade: exit A ✗, exit B ✓
- Entry: Root ✓, A ✓
- Depth: B=2 ✓, Root=0 ✓

### Hierarchy 2: 4-level deep
```
Root{A*{B*{C*{D*}}}}
```
- Containment: D inside A? ✗
- Cascade: exit B ✗
- Entry: Root ✗
- Depth: D=4 ✗ (predicted 5)

### Hierarchy 3: Wide and shallow
```
Root{A*, B, C, D, E}
```
- Containment: C inside Root? ✓, A inside B? ✗
- Cascade: exit Root ✗
- Depth: E=1 ✗

### Hierarchy 4: Mixed (Menu/Game)
```
Root{Menu*{Settings{Audio*, Video}, Help}, Game{Playing*, Paused}}
```
- Containment: Audio inside Menu? ✓, Playing inside Menu? ✗
- Cascade: exit Settings ✗, exit Menu ✗
- Entry: Game ✗
- Depth: Video=3 ✓

## Next Steps

1. **exp_symbolic_hybrid** - Combine LLM parsing with symbolic tree traversal
2. **exp_hierarchy_finetuning** - Fine-tune on tree traversal tasks
3. **exp_chunked_hierarchy** - Process one level at a time
