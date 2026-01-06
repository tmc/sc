# exp_scalability_benchmark - Research Notes

## Core Goal

**Test performance limits at 10 / 100 / 1K / 10K / 100K / 1M states.**

Identify:
1. Which operations scale poorly (O(n²) or worse)?
2. Where are the memory bottlenecks?
3. What is the practical limit for real-time applications?
4. Which topologies are most efficient at scale?

## Scale Levels

| Scale | States | Typical Use Case |
|-------|--------|------------------|
| 10 | 10 | Simple FSM (traffic light) |
| 100 | 100 | Medium FSM (game character) |
| 1K | 1,000 | Complex FSM (dialogue system) |
| 10K | 10,000 | Large system (enterprise workflow) |
| 100K | 100,000 | Very large (code analysis CFG) |
| 1M | 1,000,000 | Extreme (neural net analysis) |

## Operations Benchmarked

### 1. Generation
- Create statechart from scratch
- Expected: O(n) time and space
- Bottleneck: Memory allocation

### 2. Traversal (BFS/DFS)
- Visit all states from root
- Expected: O(n) time
- Bottleneck: Queue/stack operations

### 3. Reachability Analysis
- Find all reachable states via transitions
- Expected: O(n + e) where e = transitions
- Bottleneck: Adjacency list building

### 4. Mutation (Topology Evolution)
- Add/remove states and transitions
- Expected: O(1) per mutation
- Bottleneck: Random access into large collections

### 5. Guard Evaluation
- Evaluate guard expressions on transitions
- Expected: O(1) per guard
- Bottleneck: Expression parsing (if not cached)

### 6. Serialization
- Convert to/from JSON/dict
- Expected: O(n) time, O(n) space
- Bottleneck: JSON encoding at scale

### 7. Configuration Tracking
- Manage active state set
- Expected: O(1) per update
- Bottleneck: Set operations at scale

## Topologies Tested

### Flat
- All states at root level
- Stress test: Large OR-state
- Expected: Fastest traversal, highest memory

### Deep
- Linear chain of depth N
- Stress test: Deep hierarchy
- Expected: Slowest random access

### Wide
- Shallow tree, high branching
- Stress test: Many siblings
- Expected: Balance of depth/width

### Balanced
- Balanced tree
- Stress test: Typical real-world
- Expected: Good all-around performance

### Random
- Random hierarchical structure
- Stress test: Unpredictable access patterns
- Expected: Variable performance

### Parallel
- Many AND-state regions
- Stress test: Orthogonal composition
- Expected: Higher memory, faster per-region ops

## Expected Bottlenecks

### Memory Bottlenecks

1. **State Object Overhead**
   - Each state: ~200 bytes
   - 1M states: ~200MB minimum
   - Solution: Compact representation

2. **Transition Storage**
   - With 1.5 transitions/state: 1.5M transitions
   - Each transition: ~100 bytes
   - Solution: Edge list vs adjacency matrix

3. **JSON Serialization**
   - String overhead is significant
   - 1M states → multi-GB JSON
   - Solution: Binary formats (protobuf)

### Time Bottlenecks

1. **O(n²) Algorithms**
   - All-pairs reachability
   - Global conflict detection
   - Solution: Incremental algorithms

2. **Random Access**
   - Deep hierarchies: O(depth) access
   - Large collections: Hash table resize
   - Solution: Locality-aware data structures

3. **GC Pressure**
   - Many small allocations
   - Python GC overhead
   - Solution: Object pooling, numpy arrays

## Scaling Analysis

### Classification Method

For operation with time T(n):
- Compute T(n) / f(n) for various f
- f that yields constant ratio = scaling class

| Class | T(n)/n | T(n)/(n log n) | T(n)/n² |
|-------|--------|----------------|---------|
| O(n) | const | decreasing | decreasing |
| O(n log n) | increasing | const | decreasing |
| O(n²) | increasing | increasing | const |

### Expected Results

| Operation | Expected | Acceptable |
|-----------|----------|------------|
| Generation | O(n) | O(n) |
| Traversal | O(n) | O(n log n) |
| Reachability | O(n+e) | O(n log n) |
| Mutation | O(1) | O(log n) |
| Guard Eval | O(1) | O(1) |
| Serialization | O(n) | O(n) |
| Configuration | O(1) | O(log n) |

## Practical Limits

Based on benchmarks, estimated limits for interactive use (<100ms):

| Operation | 10ms | 100ms | 1s |
|-----------|------|-------|-----|
| Generation | ~50K | ~500K | ~5M |
| Traversal | ~100K | ~1M | ~10M |
| Mutation | ~1M | ~10M | ~100M |
| Serialization | ~10K | ~100K | ~1M |

## Recommendations

### For 10K-100K States
- Use balanced topology
- Avoid global operations
- Cache computed properties
- Use incremental updates

### For 100K-1M States
- Use flat or parallel topology
- Implement lazy evaluation
- Use compressed representations
- Consider distributed processing

### For 1M+ States
- Custom data structures
- Memory-mapped files
- Streaming algorithms
- Approximate methods

## Implementation Notes

### Profiling with tracemalloc
```python
import tracemalloc
tracemalloc.start()
# ... operation ...
current, peak = tracemalloc.get_traced_memory()
tracemalloc.stop()
```

### GC Control
```python
import gc
gc.collect()  # Force collection before benchmark
gc.disable()  # Disable during timed section
# ... timed operation ...
gc.enable()
```

### High-Resolution Timing
```python
import time
t0 = time.perf_counter()
# ... operation ...
elapsed = time.perf_counter() - t0
```

## Future Directions

### 1. Parallel Benchmarks
- Multi-threaded operations
- GPU acceleration for matrix ops
- Distributed statechart processing

### 2. Streaming Algorithms
- Process states incrementally
- Bounded memory usage
- Real-time analysis

### 3. Compression
- State encoding (dictionary)
- Transition compression
- Hierarchical delta encoding

### 4. Domain-Specific Optimization
- Code analysis: CFG-specific structures
- Game AI: Action-oriented indexing
- Workflow: Event-driven caching

## References

- McConnell, J. "Analysis of Algorithms" (2008)
- Cormen, T. "Introduction to Algorithms" (2009)
- Harel, D. "Statecharts: A Visual Formalism" (1987)
