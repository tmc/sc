# Tools and Benchmarks

This directory contains documentation for running benchmarks and profiling tools for the statecharts project.

## Running Benchmarks

The project includes comprehensive benchmarks for measuring performance of various statechart operations. Benchmarks are located throughout the codebase in `*_test.go` files.

### Quick Start

Run all benchmarks in the project:
```bash
go test -bench=. ./...
```

Run benchmarks with memory allocation stats:
```bash
go test -bench=. -benchmem ./...
```

Run benchmarks for a specific duration:
```bash
go test -bench=. -benchtime=10s ./...
```

### Benchmark Categories

#### 1. Semantics Benchmarks (`./semantics/v1/`)

Core statechart operation benchmarks:

```bash
# Run all semantics benchmarks
go test -bench=. -benchmem ./semantics/v1/

# Specific benchmark suites:
go test -bench=BenchmarkStatechartValidation -benchmem ./semantics/v1/
go test -bench=BenchmarkInitialConfiguration -benchmem ./semantics/v1/
go test -bench=BenchmarkTransition -benchmem ./semantics/v1/
go test -bench=BenchmarkGuard -benchmem ./semantics/v1/
```

Available benchmarks:
- `BenchmarkStatechartValidation` - Statechart validation performance
- `BenchmarkInitialConfiguration` - Initial configuration computation
- `BenchmarkTransitionExecution` - Transition execution
- `BenchmarkEventProcessing` - Event processing
- `BenchmarkGuardEvaluation` - Guard condition evaluation

#### 2. Profiling Benchmarks (`./semantics/v1/profiling/`)

Specialized profiling and memory benchmarks:

```bash
# Memory benchmarks
go test -bench=BenchmarkMemory -benchmem ./semantics/v1/profiling/

# CPU profiling with pprof
go test -bench=. -cpuprofile=cpu.prof ./semantics/v1/profiling/
go tool pprof cpu.prof

# Memory profiling
go test -bench=. -memprofile=mem.prof ./semantics/v1/profiling/
go tool pprof mem.prof
```

Available benchmark suites:
- `BenchmarkMemorySuite` - Complete memory benchmark suite
- `BenchmarkStatechartCreation` - Memory usage during creation
- `BenchmarkMachineLifecycle` - Complete machine lifecycle
- `BenchmarkConcurrentMachines` - Concurrent machine memory usage
- `BenchmarkMemoryOptimizations` - Optimized vs unoptimized operations
- `BenchmarkLongRunningMachines` - Extended runtime behavior (use `-benchtime`)

#### 3. Integration Benchmarks (`./integration/`)

End-to-end performance tests:

```bash
go test -bench=. -benchmem ./integration/
```

#### 4. Bridge Benchmarks (`./bridges/`)

Protocol buffer bridge performance:

```bash
go test -bench=. -benchmem ./bridges/
```

#### 5. Validation Benchmarks (`./validation/v1/`)

Validation rule performance:

```bash
go test -bench=. -benchmem ./validation/v1/
```

#### 6. Web Visualizer Benchmarks (`./cmd/web-visualizer/`)

Search index and API performance:

```bash
go test -bench=. -benchmem ./cmd/web-visualizer/
```

### Advanced Profiling

#### CPU Profiling

Generate and analyze CPU profiles:

```bash
# Run benchmarks with CPU profiling
go test -bench=BenchmarkTransition -cpuprofile=cpu.prof ./semantics/v1/

# Analyze with pprof (interactive)
go tool pprof cpu.prof

# Generate profile graph (requires graphviz)
go tool pprof -png cpu.prof > cpu_profile.png
```

#### Memory Profiling

Generate and analyze memory profiles:

```bash
# Run benchmarks with memory profiling
go test -bench=BenchmarkMachine -memprofile=mem.prof ./semantics/v1/

# Analyze allocations
go tool pprof -alloc_space mem.prof

# Analyze in-use memory
go tool pprof -inuse_space mem.prof

# Generate visual graph
go tool pprof -png mem.prof > mem_profile.png
```

#### Block Profiling

Analyze blocking operations:

```bash
go test -bench=BenchmarkConcurrent -blockprofile=block.prof ./semantics/v1/profiling/
go tool pprof block.prof
```

#### Mutex Profiling

Analyze mutex contention:

```bash
go test -bench=BenchmarkConcurrent -mutexprofile=mutex.prof ./semantics/v1/profiling/
go tool pprof mutex.prof
```

### Benchmark Comparison

Compare benchmark results over time using benchstat:

```bash
# Install benchstat
go install golang.org/x/perf/cmd/benchstat@latest

# Run benchmarks and save results
go test -bench=. -benchmem ./semantics/v1/ > old.txt

# Make changes...

# Run benchmarks again
go test -bench=. -benchmem ./semantics/v1/ > new.txt

# Compare results
benchstat old.txt new.txt
```

### Continuous Benchmarking

Run benchmarks on every commit:

```bash
# Simple benchmark check
make bench

# With comparison against main branch
git checkout main
go test -bench=. -benchmem ./... > main.txt
git checkout your-branch
go test -bench=. -benchmem ./... > branch.txt
benchstat main.txt branch.txt
```

### Skipping Long-Running Benchmarks

Some benchmarks are marked as long-running. Skip them with the `-short` flag:

```bash
go test -bench=. -short ./semantics/v1/profiling/
```

### Benchmark Best Practices

1. **Stabilize the environment**: Close other applications and disable CPU frequency scaling when possible
2. **Run multiple iterations**: Use `-benchtime=10s` or `-count=10` for more stable results
3. **Warm up the cache**: Results from the first run may differ from subsequent runs
4. **Compare carefully**: Use `benchstat` for statistical comparison between runs
5. **Profile before optimizing**: Always profile to identify actual bottlenecks

### Example Workflow

```bash
# 1. Run all benchmarks to establish a baseline
go test -bench=. -benchmem ./... > baseline.txt

# 2. Run specific benchmark with profiling
go test -bench=BenchmarkTransition -cpuprofile=cpu.prof -memprofile=mem.prof ./semantics/v1/

# 3. Analyze the profiles
go tool pprof -top cpu.prof
go tool pprof -top mem.prof

# 4. Make optimizations based on profile data

# 5. Re-run benchmarks and compare
go test -bench=. -benchmem ./... > optimized.txt
benchstat baseline.txt optimized.txt
```

## Performance Targets

Current performance characteristics (as of v0.1.0):

- Simple statechart validation: < 1µs
- Hierarchical statechart validation: < 5µs
- Transition execution: < 500ns
- Event processing: < 1µs
- Guard evaluation: < 100ns

## Reporting Performance Issues

When reporting performance issues, please include:

1. Benchmark results showing the regression
2. CPU and/or memory profiles
3. System information (OS, CPU, Go version)
4. Statechart size and complexity metrics

## Additional Resources

- [Go Benchmark Documentation](https://pkg.go.dev/testing#hdr-Benchmarks)
- [pprof Documentation](https://github.com/google/pprof/blob/main/doc/README.md)
- [Go Performance Tuning](https://github.com/dgryski/go-perfbook)
