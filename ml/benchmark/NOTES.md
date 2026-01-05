# Benchmark: Memory Mechanism Comparison

## Overview
Compares all 5 memory mechanisms on a sequence recall task.

## Components
- `memory_comparison.py`: Full benchmark implementation

## Memory Mechanisms Compared
1. **ATTENTION** (exp_b): Soft attention over history buffer
2. **NTM** (exp_b): Neural Turing Machine with content-based addressing
3. **RECURRENT** (exp_b): GRU-based baseline
4. **HTM** (exp_i): Hierarchical Temporal Memory with SDR encoding
5. **CMS** (exp_j): Continuum Memory System / TITANS HOPE

## Task: Sequence Recall
- Vocab size: 8
- Sequence length: 6
- Batch size: 16
- Goal: Remember N items, recall in order

## Results
```
Memory Type       Accuracy      Latency    Grad Norm     Params
----------------------------------------------------------------------
ATTENTION            11.7%       1.20ms       0.0920      2,680
NTM                  12.9%       2.97ms       0.1863      3,836
RECURRENT            10.0%       1.87ms       0.1533      8,904
HTM                  24.0%       2.04ms       0.2163     14,713
CMS                  12.9%       0.82ms       0.1197     39,467

Rankings:
- Fastest: CMS (0.82ms)
- Most accurate: HTM (24.0%)
- Smallest: ATTENTION (2,680 params)
```

## Key Findings
- HTM captures sequence patterns most effectively on this task
- CMS benefits from batch sequence processing (parallelism)
- All mechanisms maintain gradient flow (non-zero grad norms)
- Random baseline: 12.5% (1/8 vocab)

## Issues Fixed
- Raw memory classes (SoftAttentionMemory, etc.) lack `init_state`: Use UnifiedMemory wrapper
- UnifiedMemory doesn't accept `hidden_size` for recurrent: Use `d_memory` param instead

## Reproduction Commands
```bash
cd /Volumes/tmc/go/src/github.com/tmc/sc/ml
.venv/bin/python3 benchmark/memory_comparison.py
```

## Dependencies
- differentiable/exp_b_memory.py (UnifiedMemory, MemoryType)
- differentiable/exp_i_htm_memory.py (HTMMemory)
- experiments/exp_j_titans_hope/continuum_memory.py (ContinuumMemorySystem)
