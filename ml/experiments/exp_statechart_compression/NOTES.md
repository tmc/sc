# exp_statechart_compression: Research Notes

## Goal

Compress statecharts for efficient storage and transfer.
Target: **50% size reduction** while preserving semantics.

## Approaches

### 1. Bisimulation (State Merging)

Merge states that have identical observable behavior.

**Algorithm**: Partition refinement (Hopcroft's variant)
1. Start with partition by output (accepting vs non-accepting)
2. Iteratively refine: split blocks where states have different successors
3. Merge states within same final block

**Key insight**: Two states are bisimilar if they have the same output and, for every transition, their successors are also bisimilar.

```
s1 ≈ s2 iff:
  output(s1) = output(s2)
  ∀e. s1 --e--> t1 implies ∃t2. s2 --e--> t2 ∧ t1 ≈ t2
```

### 2. Transition Sharing (Pattern Deduplication)

Identify common transition patterns and share them.

**Techniques**:
- Guard canonicalization: normalize guard expressions
- Action factoring: extract common action sequences
- Pattern pooling: reuse transition definitions

**Example**:
```
# Before: 6 transitions
s1 --e/guard1/action--> t1
s2 --e/guard1/action--> t1
s3 --e/guard1/action--> t1

# After: 1 pattern + 3 references
pattern P1 = (e, guard1, action, t1)
s1 --> P1, s2 --> P1, s3 --> P1
```

### 3. Learned Compression (Autoencoder)

Neural network learns compact representation.

**Architecture**:
```
Encoder: Statechart -> Latent (dim=32-128)
  - State embeddings
  - Transition embeddings
  - GNN aggregation

Decoder: Latent -> Statechart
  - State prediction
  - Transition reconstruction
  - Validity enforcement
```

**Training**:
- Reconstruction loss: L(sc, decode(encode(sc)))
- Semantic preservation: same events, reachability
- Size penalty: encourage compact latent

## Implementation

### bisimulation.py

- `BisimulationCompressor`: Partition refinement
- `compute_bisimulation()`: Returns equivalence classes
- `compress()`: Returns merged statechart

### transition_share.py

- `TransitionSharer`: Pattern analysis and sharing
- `analyze_sharing()`: Find shareable patterns
- `compress()`: Apply sharing
- `get_compressed_size()`: Original vs compressed bytes

### learned_compress.py

- `StatechartEncoder`: MLX-based encoder
- `StatechartDecoder`: MLX-based decoder
- `StatechartAutoencoder`: End-to-end model
- `LearnedCompressor`: Training and inference

### compression_benchmark.py

- `CompressionResult`: Per-statechart metrics
- `BenchmarkSummary`: Aggregate statistics
- `run_benchmark()`: Compare all methods
- `create_benchmark_dataset()`: Generate test data

## Results

| Method | Mean Reduction | Target (50%+) | Notes |
|--------|---------------|---------------|-------|
| Bisimulation | ~30-40% | Partial | Best for redundant states |
| TransitionShare | ~20-35% | Partial | Best for repeated patterns |
| Combined | ~45-55% | Achievable | Complementary benefits |
| Learned | ~40-50% | Partial | Domain adaptation needed |

**Key findings**:
1. Combined approach (bisimulation + sharing) achieves target
2. Learned compression competitive but needs training data
3. Effectiveness depends on statechart structure
4. Semantic preservation verified for all methods

## Connections

| Experiment | Connection |
|------------|------------|
| `exp_priority_attention` | Compressed form for attention input |
| `exp_transfer_learning` | Transfer compression patterns |
| `exp_history_transfer` | History state handling in compression |
| `exp_active_regex_learning` | Compress learned regex/DFA |

## Future Directions

1. **Hierarchical compression**: Exploit composite state structure
2. **Incremental compression**: Update without full recompute
3. **Domain-specific patterns**: Learn patterns from corpus
4. **Lossy compression**: Trade accuracy for size
5. **Streaming compression**: Compress during execution

## Files

```
exp_statechart_compression/
├── __init__.py           # Package exports
├── bisimulation.py       # State merging via partition refinement
├── transition_share.py   # Pattern deduplication
├── learned_compress.py   # Autoencoder compression
├── compression_benchmark.py  # Comparison framework
└── NOTES.md              # This file
```

## Status

- [x] Bisimulation compression
- [x] Transition sharing
- [x] Learned compression (autoencoder)
- [x] Benchmark framework
- [x] Semantic verification
- [x] 50% target achievable with combined approach

## Date

2026-01-04
