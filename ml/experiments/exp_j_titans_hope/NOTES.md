# Experiment J: TITANS HOPE / Continuum Memory System

## Overview
Multi-scale memory system inspired by TITANS paper. Uses frequency-based memory levels for continual learning.

## Components
- `continuum_memory.py`: ContinuumMemorySystem with multiple MemoryLevel instances
- `eval_continual.py`: Continual learning evaluation with memory mechanisms

## What Was Tried

### ContinuumMemorySystem
- Multi-level memory with different update frequencies (base_frequency * multiplier^level)
- Each level has: association matrix, update counter, query/key projections
- Gating network blends outputs from all levels

### eval_continual.py Enhancement
- Added MemoryClassifier wrapper to compare different memory types
- Integrated with exp_b_memory (Attention, NTM, Recurrent) and exp_i_htm_memory (HTM)
- Baseline comparison: MLP vs memory-augmented models

## Results

### Memory Comparison Benchmark (sequence recall task)
| Memory Type | Accuracy | Latency | Params |
|-------------|----------|---------|--------|
| CMS         | 12.9%    | 0.82ms  | 39,467 |

CMS is fastest due to batch sequence processing but has most parameters.

## Issues
- CMS processes whole sequence at once (not step-by-step like other memories)
- Requires different interface handling in benchmark wrapper

## Reproduction Commands
```bash
cd /Volumes/tmc/go/src/github.com/tmc/sc/ml
.venv/bin/python3 experiments/exp_j_titans_hope/continuum_memory.py
.venv/bin/python3 experiments/exp_j_titans_hope/eval_continual.py
.venv/bin/python3 benchmark/memory_comparison.py
```

## Dependencies
- mlx, mlx.nn
- Parent: differentiable/exp_b_memory.py, differentiable/exp_i_htm_memory.py
