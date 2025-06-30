# Memory Optimization Guide for Large Statecharts

This guide provides comprehensive strategies and best practices for optimizing memory usage when working with large statecharts in the sc library.

## Table of Contents

1. [Understanding Memory Usage](#understanding-memory-usage)
2. [Profiling Memory](#profiling-memory)
3. [Optimization Strategies](#optimization-strategies)
4. [Large Statechart Patterns](#large-statechart-patterns)
5. [Memory Leak Prevention](#memory-leak-prevention)
6. [Performance Benchmarks](#performance-benchmarks)
7. [Troubleshooting](#troubleshooting)

## Understanding Memory Usage

### Memory Components in Statecharts

A statechart's memory footprint consists of:

1. **State Hierarchy**: Tree structure of states
2. **Transitions**: Connections between states with events and actions
3. **Machine Instance**: Runtime state including configuration and history
4. **Context Data**: Application-specific data stored in machines
5. **Event Queue**: Pending events for processing

### Typical Memory Usage Patterns

| Statechart Size | States | Transitions | Typical Memory |
|----------------|--------|-------------|----------------|
| Small          | <100   | <200        | <1 MB          |
| Medium         | <1,000 | <2,000      | 1-10 MB        |
| Large          | <10,000| <20,000     | 10-100 MB      |
| Extra Large    | 50,000+| 100,000+    | 100+ MB        |

## Profiling Memory

### Basic Memory Profiling

```go
import "github.com/tmc/sc/semantics/v1/profiling"

// Create a profiler
profiler := profiling.NewStatechartProfiler()

// Profile statechart creation
deltaFunc, err := profiler.ProfileValidation(statechart)
err = statechart.Validate()
delta := deltaFunc()

fmt.Printf("Validation used %d bytes (%.2f MB)\n", 
    delta.DeltaAlloc, float64(delta.DeltaAlloc)/(1024*1024))
```

### Continuous Memory Monitoring

```go
// Set up continuous monitoring
monitor := profiling.NewMemoryMonitor()
monitor.SetThreshold("high_memory", 100*1024*1024, func(snapshot profiling.MemorySnapshot) {
    log.Printf("High memory alert: %d MB", snapshot.Alloc/(1024*1024))
})
monitor.StartMonitoring(5 * time.Second)
```

### Memory Leak Detection

```go
// Configure leak detector
detector := profiling.NewLeakDetector()
detector.SetThresholds(profiling.LeakThresholds{
    MemoryGrowthRate:  1024 * 1024,    // 1MB/sec
    ObjectGrowthRate:  10000,          // 10k objects/sec
    MinSampleDuration: 30 * time.Second,
})

detector.AddCallback(func(leak profiling.DetectedLeak) {
    log.Printf("Memory leak detected: %s (severity: %s)", 
        leak.Description, leak.Severity)
})

detector.Start()
defer detector.Stop()
```

## Optimization Strategies

### 1. Use Object Pooling

Object pooling significantly reduces allocation pressure:

```go
// Create an object pool
pool := profiling.NewObjectPool()

// Use pooled objects
config := pool.GetConfiguration()
defer pool.PutConfiguration(config)

// Create optimized machine with pooling
machine, err := profiling.NewOptimizedMachine(statechart, "machine_id", pool)
machine.SetHistoryLimit(1000)
machine.SetCompactionFrequency(100)
```

### 2. Optimize Large Statecharts

For statecharts with thousands of states:

```go
// Use the large statechart optimizer
optimizer := profiling.NewLargeStatechartMemoryOptimizer()
optimized, report := optimizer.OptimizeLargeStatechart(largeStatechart)

fmt.Printf("Memory reduced by %.2f%% (%d bytes saved)\n", 
    report.ReductionPercent, report.SizeReduction)
```

### 3. Use Memory-Efficient Indexing

For fast lookups in large statecharts:

```go
// Build memory-efficient index
index := profiling.BuildMemoryEfficientIndex(statechart)

// Fast O(1) lookups
parent, depth, childCount, found := index.GetStateInfo("state_label")
transitions := index.GetTransitionsFrom("state_label")

fmt.Printf("Index uses %d bytes (%.2f bytes/state)\n",
    index.GetMemoryUsage(), 
    float64(index.GetMemoryUsage())/float64(stateCount))
```

### 4. Implement Memory Budgets

Control memory usage with budgets:

```go
// Create factory with 512MB budget
factory := profiling.NewMemoryPooledMachineFactory(512 * 1024 * 1024)

// Create machines within budget
machine, err := factory.CreateOptimizedMachine(statechart, "id", 1000)
if err != nil {
    log.Printf("Memory budget exceeded: %v", err)
}

// Check current usage
stats := factory.GetMemoryStats()
fmt.Printf("Memory usage: %d/%d MB (%.2f%%)\n",
    stats.CurrentUsage/(1024*1024),
    stats.MaxMemory/(1024*1024),
    float64(stats.CurrentUsage)/float64(stats.MaxMemory)*100)
```

### 5. Use Ring Buffers for History

Limit memory growth from step history:

```go
// Create memory-efficient machine with bounded history
machine := profiling.NewMemoryEfficientMachine(statechart, 500) // 500 step history

// History automatically managed in ring buffer
err := machine.Step("event")
history := machine.GetHistory() // Max 500 entries
```

## Large Statechart Patterns

### Pattern 1: Hierarchical Decomposition

Break large flat statecharts into hierarchical structures:

```go
// Instead of 10,000 flat states
// Use hierarchical organization
rootState := &sc.State{
    Label: "System",
    Type:  sc.StateTypeParallel,
    Children: []*sc.State{
        createSubsystem("Network", 1000),    // 1000 states
        createSubsystem("Processing", 2000), // 2000 states
        createSubsystem("Storage", 1000),    // 1000 states
        // ... more subsystems
    },
}
```

### Pattern 2: Lazy Loading

Load statechart sections on demand:

```go
type LazyStatechart struct {
    core      *sc.Statechart
    modules   map[string]func() *sc.State
    loaded    map[string]*sc.State
    mu        sync.RWMutex
}

func (ls *LazyStatechart) LoadModule(name string) error {
    ls.mu.Lock()
    defer ls.mu.Unlock()
    
    if _, loaded := ls.loaded[name]; loaded {
        return nil
    }
    
    if loader, exists := ls.modules[name]; exists {
        state := loader()
        // Integrate state into core statechart
        ls.loaded[name] = state
        return nil
    }
    
    return fmt.Errorf("module %s not found", name)
}
```

### Pattern 3: Shared State References

Use shared references for common patterns:

```go
// Create shared states
sharedErrorState := &sc.State{
    Label: "Error",
    Type:  sc.StateTypeBasic,
}

// Reference in multiple places
for i := 0; i < 100; i++ {
    subsystem := &sc.State{
        Label: fmt.Sprintf("Subsystem_%d", i),
        Children: []*sc.State{
            {Label: "Active"},
            {Label: "Inactive"},
            sharedErrorState, // Shared reference
        },
    }
}
```

## Memory Leak Prevention

### Common Leak Sources

1. **Unbounded History**
   ```go
   // BAD: Unlimited history
   machine := semantics.NewMachine(statechart, "id", nil)
   
   // GOOD: Limited history
   machine := profiling.NewOptimizedMachine(statechart, "id", pool)
   machine.SetHistoryLimit(1000)
   ```

2. **Accumulated Context Data**
   ```go
   // BAD: Context grows indefinitely
   machine.UpdateContext("logs", append(currentLogs, newLog))
   
   // GOOD: Bounded context
   logs := machine.GetContext("logs").([]string)
   if len(logs) > 1000 {
       logs = logs[len(logs)-1000:] // Keep last 1000
   }
   machine.UpdateContext("logs", append(logs, newLog))
   ```

3. **Event Listener Accumulation**
   ```go
   // BAD: Listeners never removed
   machine.OnTransition(func(t *sc.Transition) {
       // process
   })
   
   // GOOD: Track and remove listeners
   listenerID := machine.OnTransition(handler)
   defer machine.RemoveListener(listenerID)
   ```

### Leak Detection in Production

```go
// Set up production leak monitoring
detector := profiling.NewStatechartLeakDetector()
detector.SetProductionThresholds() // Less aggressive than development

// Register machines
detector.RegisterMachine("service_1", machine1)
detector.RegisterMachine("service_2", machine2)

// Periodic checks
go func() {
    ticker := time.NewTicker(5 * time.Minute)
    for range ticker.C {
        reports := detector.CheckMachineLeaks()
        for _, report := range reports {
            if report.Severity >= profiling.High {
                alertOps(report)
            }
        }
    }
}()
```

## Performance Benchmarks

### Memory Optimization Results

Based on extensive benchmarking, here are typical optimization results:

| Optimization Technique | Memory Reduction | Performance Impact |
|-----------------------|------------------|-------------------|
| String Interning      | 20-40%          | <1% slower        |
| Object Pooling        | 30-50%          | 5-10% faster      |
| Slice Optimization    | 10-20%          | Negligible        |
| Index Building        | N/A             | 10-100x faster lookups |
| History Limiting      | 50-90%          | None              |

### Benchmark Your Statecharts

```go
func BenchmarkYourStatechart(b *testing.B) {
    suite := profiling.NewLargeStatechartBenchmarkSuite()
    
    // Add your statechart
    suite.AddStatechart("MyChart", createMyStatechart)
    
    // Run comprehensive benchmarks
    suite.BenchmarkCreation(b)
    suite.BenchmarkValidation(b)
    suite.BenchmarkMachineLifecycle(b)
    suite.BenchmarkConcurrency(b)
}
```

## Troubleshooting

### High Memory Usage Checklist

1. **Check History Size**
   ```go
   history := machine.GetStepHistory()
   fmt.Printf("History size: %d steps\n", len(history))
   ```

2. **Analyze State Distribution**
   ```go
   analyzer := profiling.NewMemoryAnalyzer()
   analysis := analyzer.AnalyzeStatechart(statechart)
   fmt.Printf("State depth: max=%d, avg=%.2f\n", 
       analysis.MaxDepth, analysis.AvgDepth)
   ```

3. **Monitor GC Pressure**
   ```go
   var m runtime.MemStats
   runtime.ReadMemStats(&m)
   fmt.Printf("GC runs: %d, Pause: %v\n", m.NumGC, m.PauseNs)
   ```

4. **Profile Allocations**
   ```go
   import _ "net/http/pprof"
   // Visit http://localhost:6060/debug/pprof/heap
   ```

### Memory Optimization Decision Tree

```
High Memory Usage?
├─> Yes
│   ├─> Many States (>10k)?
│   │   ├─> Yes: Use LargeStatechartOptimizer
│   │   └─> No: Continue
│   ├─> Long Running?
│   │   ├─> Yes: Implement history limits
│   │   └─> No: Continue
│   ├─> Many Machines?
│   │   ├─> Yes: Use object pooling
│   │   └─> No: Check for leaks
│   └─> Check context data growth
└─> No: Normal operation
```

### Getting Help

For additional assistance with memory optimization:

1. Run the built-in diagnostics:
   ```go
   diagnostics := profiling.RunMemoryDiagnostics(statechart, machine)
   fmt.Println(diagnostics.GetReport())
   ```

2. Enable verbose profiling:
   ```go
   profiling.EnableVerboseMode()
   ```

3. Check the examples in `/semantics/v1/profiling/example_test.go`

## Best Practices Summary

1. **Always set history limits** for long-running machines
2. **Use object pooling** for high-frequency operations
3. **Profile before optimizing** to identify actual bottlenecks
4. **Monitor production systems** with leak detection
5. **Design hierarchically** for very large statecharts
6. **Batch operations** when possible to reduce overhead
7. **Use memory budgets** to prevent runaway growth
8. **Test at scale** to catch issues early

Remember: The goal is to find the right balance between memory usage and performance for your specific use case. Not all optimizations are necessary for every application.