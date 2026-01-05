# Differentiable Statechart Components

## Overview
Core differentiable components for statechart-based neural networks.

## Memory Mechanisms

### exp_b_memory.py - Soft Memory for Statecharts
Three memory mechanisms with consistent interface via UnifiedMemory:

#### SoftAttentionMemory
- Query-key attention over sliding history buffer
- Maintains `history` and `write_ptr` state
- Output: weighted sum of history vectors

#### NTMMemory
- Neural Turing Machine style content-based addressing
- Fixed-size memory matrix with read/write heads
- Supports both content-based and location-based addressing

#### RecurrentMemory
- GRU baseline for comparison
- Hidden state as implicit memory
- Simpler but less interpretable

#### UnifiedMemory
- Wrapper providing consistent interface
- `init_state(batch_size) -> dict`
- `step(config, state) -> (retrieved, new_state, weights)`

### exp_i_htm_memory.py - HTM-Inspired Memory
Hierarchical Temporal Memory concepts adapted for differentiable use:

#### SDREncoder
- Sparse Distributed Representation encoding
- Top-k sparsity with temperature-controlled softmax

#### TemporalPooler
- Learns temporal sequences over SDRs
- Maintains column activations and predictions

#### SequenceMemory
- Stores and retrieves SDR sequences
- Temporal context for prediction

#### ColumnMemory
- HTM column-like structure
- Proximal and distal dendrite connections

#### HTMMemory (main class)
- Combines SDR encoding with temporal pooling
- Returns `info` dict with `current_sdr`, `prediction`, `confidence`
- Highest accuracy (24%) on sequence recall benchmark

## Results

### Gradient Flow Tests
All mechanisms pass:
```
SoftAttentionMemory: OK
NTMMemory: OK
RecurrentMemory: OK
HTMMemory: OK
```

### Benchmark Performance (sequence recall)
| Memory | Accuracy | Latency | Params |
|--------|----------|---------|--------|
| ATTENTION | 11.7% | 1.20ms | 2,680 |
| NTM | 12.9% | 2.97ms | 3,836 |
| RECURRENT | 10.0% | 1.87ms | 8,904 |
| HTM | 24.0% | 2.04ms | 14,713 |

## Issues Fixed
- `mx.softplus` doesn't exist: Use `mx.log(1 + mx.exp(x))`
- `mx.tree_flatten` doesn't exist: Use `nn.utils.tree_flatten`
- nn.Sequential indexing: Can't use `[]`, use individual layer attributes
- HTMMemory interface: Updated to return `info` dict for integration

## Reproduction Commands
```bash
cd /Volumes/tmc/go/src/github.com/tmc/sc/ml
.venv/bin/python3 differentiable/exp_b_memory.py
.venv/bin/python3 differentiable/exp_i_htm_memory.py
```

## Integration Points
- exp_d_integrated.py: Combines memory with SoftStateConfiguration
- exp_j_titans_hope/eval_continual.py: Continual learning evaluation
- benchmark/memory_comparison.py: Cross-mechanism comparison
