# Memory Profiling and Optimization System

This package provides comprehensive memory profiling and optimization tools specifically designed for statechart-based applications. It helps developers identify memory usage patterns, detect leaks, and optimize performance for large-scale statechart deployments.

## Features

### 🔍 Memory Profiling
- **Basic Memory Profiling**: Track memory allocation and usage patterns
- **Statechart-Specific Profiling**: Profile validation, machine creation, and event processing
- **Lifecycle Profiling**: Track memory usage throughout a statechart's complete lifecycle
- **Continuous Profiling**: Monitor memory usage over extended periods

### 🚨 Leak Detection
- **Real-time Leak Detection**: Identify memory, object, and goroutine leaks as they occur
- **Statechart-Specific Detection**: Monitor machines for excessive history, error accumulation, and configuration growth
- **Configurable Thresholds**: Set custom thresholds for different environments
- **Automated Suggestions**: Get actionable recommendations for addressing detected leaks

### ⚡ Memory Optimization
- **Object Pooling**: Reuse frequently allocated objects to reduce GC pressure
- **Optimized Machines**: Automatic history compaction and memory-efficient operations
- **Large Statechart Optimization**: String interning, compact indexing, and efficient lookups
- **Concurrent Optimization**: Parallel processing for very large statecharts

### 📊 Memory Management
- **Budget Management**: Enforce memory limits with callbacks for warnings
- **Memory-Efficient Caching**: LRU cache with size and memory constraints
- **Pooled Machine Factory**: Create and manage machines within memory budgets
- **Size Estimation**: Predict memory requirements before allocation

## Quick Start

### Basic Memory Profiling

```go
import "github.com/tmc/sc/semantics/v1/profiling"

// Create a profiler
profiler := profiling.NewMemoryProfiler()

// Set baseline
profiler.SetBaseline("before_operation")

// Do some work...

// Take snapshot and compute delta
snapshot := profiler.TakeSnapshot("after_operation")
delta := profiling.ComputeDelta(*profiler.GetBaseline(), snapshot)
fmt.Printf("Memory used: %d bytes\n", delta.DeltaAlloc)
```

### Statechart Profiling

```go
// Profile statechart operations
profiler := profiling.NewStatechartProfiler()

// Profile validation
deltaFunc, _ := profiler.ProfileValidation(statechart)
err := statechart.Validate()
validationDelta := deltaFunc()

// Profile machine creation
deltaFunc, machine, _ := profiler.ProfileMachineCreation(statechart, "machine_id")
creationDelta := deltaFunc()

// Profile event processing
machine.Start()
deltaFunc, stepped, _ := profiler.ProfileEventProcessing(machine, "event")
eventDelta := deltaFunc()
```

### Leak Detection

```go
detector := profiling.NewLeakDetector()
detector.SetThresholds(profiling.LeakThresholds{
    MemoryGrowthRate:  1024 * 1024,    // 1MB/sec
    ObjectGrowthRate:  10000,           // 10k objects/sec
    MinSampleDuration: 30 * time.Second,
    MinGrowthAmount:   10 * 1024 * 1024, // 10MB
})

detector.AddCallback(func(leak profiling.DetectedLeak) {
    log.Printf("Leak detected: %s (severity: %s)", 
        leak.Description, leak.Severity)
})

detector.Start()
defer detector.Stop()
```

### Memory Optimization

```go
// Use object pooling
pool := profiling.NewObjectPool()
config := pool.GetConfiguration()
// Use config...
pool.PutConfiguration(config)

// Create optimized machine
machine, _ := profiling.NewOptimizedMachine(statechart, "id", pool)
machine.SetHistoryLimit(1000)
machine.SetCompactionFrequency(100)

// Optimize large statechart
optimizer := profiling.NewLargeStatechartMemoryOptimizer()
optimized, report := optimizer.OptimizeLargeStatechart(largeStatechart)
fmt.Printf("Size reduced by %.2f%%\n", report.ReductionPercent)
```

## Performance Guidelines

### For Large Statecharts (>100 states)
- Use `LargeStatechartOptimizer` for fast lookups
- Build indexes for O(1) state access
- Consider concurrent optimization for very large statecharts
- Use memory-efficient indexing structures

### For High-Concurrency Applications
- Share object pools across machines
- Use the pooled machine factory
- Pre-optimize statecharts before creating machines
- Monitor GC pressure and tune GOGC if needed

### For Long-Running Applications
- Set appropriate history limits
- Enable periodic compaction
- Use continuous profiling
- Monitor for gradual memory growth

### For Memory-Constrained Environments
- Use `MemoryEfficientMachine` instead of regular machines
- Implement aggressive history limits
- Use memory budget management
- Consider disabling non-essential features

## Benchmarking

Run the complete benchmark suite:

```bash
go test -bench=BenchmarkMemorySuite -benchmem ./semantics/v1/profiling
```

Run specific benchmarks:

```bash
# Statechart creation
go test -bench=BenchmarkStatechartCreation -benchmem ./semantics/v1/profiling

# Concurrent machines
go test -bench=BenchmarkConcurrentMachines -benchmem ./semantics/v1/profiling

# Memory optimizations comparison
go test -bench=BenchmarkMemoryOptimizations -benchmem ./semantics/v1/profiling
```

## Production Setup Example

```go
// 1. Set up leak detection
leakDetector := profiling.NewStatechartLeakDetector()
leakDetector.SetProductionThresholds()
leakDetector.Start()

// 2. Set up continuous profiling
continuousProfiler := profiling.NewContinuousProfiler(5 * time.Minute)
continuousProfiler.Start()

// 3. Set up memory budget
memoryBudget := profiling.NewMemoryBudgetManager(1024 * 1024 * 1024) // 1GB

// 4. Create pooled machine factory
factory := profiling.NewMemoryPooledMachineFactory(512 * 1024 * 1024) // 512MB

// 5. Set up periodic reporting
go func() {
    ticker := time.NewTicker(time.Hour)
    for range ticker.C {
        report := continuousProfiler.GetProfiler().GetMemoryProfiler().GenerateReport()
        log.Printf("Peak Memory: %d MB", report.PeakAlloc/(1024*1024))
        // ... log other metrics
    }
}()
```

## Memory Usage Patterns

### Good Patterns
```go
// Pre-allocate with known capacity
states := make([]*sc.StateRef, 0, expectedSize)

// Reuse objects via pooling
config := pool.GetConfiguration()
defer pool.PutConfiguration(config)

// Use ring buffers for bounded history
history := profiling.NewRingBuffer(1000)
```

### Patterns to Avoid
```go
// Unbounded slice growth
var history []*sc.Step // Can grow indefinitely

// Frequent small allocations in hot paths
for event := range events {
    config := &sc.Configuration{} // Allocates every iteration
}
```

## Troubleshooting

### High Memory Usage
1. Enable leak detection to identify growth patterns
2. Check step history sizes - implement limits if needed
3. Review context data for accumulating values
4. Use object pooling for frequently allocated objects
5. Consider statechart structure optimization

### Memory Leaks
1. Use leak detection with appropriate thresholds
2. Check for unbounded slice/map growth
3. Verify proper cleanup of event listeners
4. Review goroutine lifecycle management
5. Monitor long-running machine state accumulation

### Performance Degradation
1. Use benchmarking to identify regressions
2. Profile memory allocation patterns
3. Check for inefficient lookups in large statecharts
4. Review concurrency patterns and contention
5. Consider memory-efficient implementations

## API Reference

See the [package documentation](https://pkg.go.dev/github.com/tmc/sc/semantics/v1/profiling) for detailed API reference.

## Examples

Check the `example_test.go` file for comprehensive examples covering:
- Basic memory profiling
- Leak detection
- Object pooling
- Large statechart optimization
- High-concurrency scenarios
- Production monitoring setups