package profiling

import (
	"context"
	"fmt"
	"runtime"
	"sync"
	"testing"
	"time"

	"github.com/tmc/sc"
	"github.com/tmc/sc/semantics/v1"
	testutil "github.com/tmc/sc/testing"
)

// MemoryBenchmark provides standardized memory benchmarking for statechart operations.
type MemoryBenchmark struct {
	profiler   *StatechartProfiler
	optimizer  *MemoryOptimizer
	detector   *LeakDetector
	results    []BenchmarkResult
	mu         sync.Mutex
}

// BenchmarkResult captures the results of a memory benchmark.
type BenchmarkResult struct {
	Name            string
	Duration        time.Duration
	Iterations      int
	AllocatedBytes  int64
	AllocatedObjects int64
	PeakMemory      uint64
	MemoryEfficiency float64 // operations per MB
	GCCycles        uint32
	Recommendations []string
}

// NewMemoryBenchmark creates a new memory benchmark suite.
func NewMemoryBenchmark() *MemoryBenchmark {
	return &MemoryBenchmark{
		profiler:  NewStatechartProfiler(),
		optimizer: NewMemoryOptimizer(),
		detector:  NewLeakDetector(),
		results:   make([]BenchmarkResult, 0, 100),
	}
}

// BenchmarkStatechartCreation benchmarks memory usage during statechart creation.
func (mb *MemoryBenchmark) BenchmarkStatechartCreation(b *testing.B) {
	tests := []struct {
		name       string
		createFunc func() *sc.Statechart
	}{
		{"Simple", testutil.CreateSimpleStatechart},
		{"Hierarchical", testutil.CreateHierarchicalStatechart},
		{"Orthogonal", testutil.CreateOrthogonalStatechart},
		{"Large_50_100", func() *sc.Statechart { return testutil.CreateLargeStatechart(50, 100) }},
		{"Large_100_200", func() *sc.Statechart { return testutil.CreateLargeStatechart(100, 200) }},
		{"Large_500_1000", func() *sc.Statechart { return testutil.CreateLargeStatechart(500, 1000) }},
		{"XLarge_1000_2000", func() *sc.Statechart { return testutil.CreateLargeStatechart(1000, 2000) }},
	}

	for _, tt := range tests {
		b.Run(tt.name, func(b *testing.B) {
			mb.runStatechartCreationBenchmark(b, tt.name, tt.createFunc)
		})
	}
}

// runStatechartCreationBenchmark runs a single statechart creation benchmark.
func (mb *MemoryBenchmark) runStatechartCreationBenchmark(b *testing.B, name string, createFunc func() *sc.Statechart) {
	var before, after runtime.MemStats
	
	// Force GC and get baseline
	runtime.GC()
	runtime.ReadMemStats(&before)
	
	start := time.Now()
	
	b.ResetTimer()
	b.ReportAllocs()
	
	var peak uint64
	
	for i := 0; i < b.N; i++ {
		statechart := semantics.NewStatechart(createFunc())
		_ = statechart.Validate() // Include validation in the benchmark
		
		// Track peak memory usage
		var current runtime.MemStats
		runtime.ReadMemStats(&current)
		if current.Alloc > peak {
			peak = current.Alloc
		}
	}
	
	b.StopTimer()
	duration := time.Since(start)
	
	// Force GC and measure final state
	runtime.GC()
	runtime.ReadMemStats(&after)
	
	result := BenchmarkResult{
		Name:             fmt.Sprintf("StatechartCreation_%s", name),
		Duration:         duration,
		Iterations:       b.N,
		AllocatedBytes:   int64(after.TotalAlloc - before.TotalAlloc),
		AllocatedObjects: int64(after.Mallocs - before.Mallocs),
		PeakMemory:       peak,
		GCCycles:         after.NumGC - before.NumGC,
	}
	
	// Calculate memory efficiency (operations per MB)
	if result.AllocatedBytes > 0 {
		result.MemoryEfficiency = float64(b.N) / (float64(result.AllocatedBytes) / (1024 * 1024))
	}
	
	result.Recommendations = mb.generateCreationRecommendations(result)
	
	mb.recordResult(result)
	
	// Report custom metrics
	b.ReportMetric(float64(result.AllocatedBytes)/float64(b.N), "bytes/op")
	b.ReportMetric(float64(result.PeakMemory)/(1024*1024), "peak-MB")
	b.ReportMetric(result.MemoryEfficiency, "ops/MB")
}

// BenchmarkMachineLifecycle benchmarks the complete machine lifecycle.
func (mb *MemoryBenchmark) BenchmarkMachineLifecycle(b *testing.B) {
	tests := []struct {
		name       string
		createFunc func() *sc.Statechart
		events     []string
	}{
		{
			"Simple", 
			testutil.CreateSimpleStatechart,
			[]string{"go", "go", "go"},
		},
		{
			"Hierarchical",
			testutil.CreateHierarchicalStatechart,
			[]string{"start", "finish", "start", "stop"},
		},
		{
			"Large_Complex",
			func() *sc.Statechart { return testutil.CreateLargeStatechart(100, 200) },
			[]string{"event_0", "event_1", "event_2", "event_3", "event_4"},
		},
	}

	for _, tt := range tests {
		b.Run(tt.name, func(b *testing.B) {
			mb.runMachineLifecycleBenchmark(b, tt.name, tt.createFunc, tt.events)
		})
	}
}

// runMachineLifecycleBenchmark runs a machine lifecycle benchmark.
func (mb *MemoryBenchmark) runMachineLifecycleBenchmark(b *testing.B, name string, createFunc func() *sc.Statechart, events []string) {
	var before, after runtime.MemStats
	
	runtime.GC()
	runtime.ReadMemStats(&before)
	
	start := time.Now()
	
	b.ResetTimer()
	b.ReportAllocs()
	
	var peak uint64
	
	for i := 0; i < b.N; i++ {
		// Create statechart and machine
		statechart := semantics.NewStatechart(createFunc())
		machine, _ := semantics.NewMachine(statechart, fmt.Sprintf("bench-machine-%d", i), nil)
		
		// Start machine
		machine.Start()
		
		// Process events
		for _, event := range events {
			machine.Step(event)
		}
		
		// Stop machine
		machine.Stop()
		
		// Track peak memory
		var current runtime.MemStats
		runtime.ReadMemStats(&current)
		if current.Alloc > peak {
			peak = current.Alloc
		}
	}
	
	b.StopTimer()
	duration := time.Since(start)
	
	runtime.GC()
	runtime.ReadMemStats(&after)
	
	result := BenchmarkResult{
		Name:             fmt.Sprintf("MachineLifecycle_%s", name),
		Duration:         duration,
		Iterations:       b.N,
		AllocatedBytes:   int64(after.TotalAlloc - before.TotalAlloc),
		AllocatedObjects: int64(after.Mallocs - before.Mallocs),
		PeakMemory:       peak,
		GCCycles:         after.NumGC - before.NumGC,
	}
	
	if result.AllocatedBytes > 0 {
		result.MemoryEfficiency = float64(b.N) / (float64(result.AllocatedBytes) / (1024 * 1024))
	}
	
	result.Recommendations = mb.generateLifecycleRecommendations(result)
	
	mb.recordResult(result)
	
	b.ReportMetric(float64(result.AllocatedBytes)/float64(b.N), "bytes/op")
	b.ReportMetric(float64(result.PeakMemory)/(1024*1024), "peak-MB")
	b.ReportMetric(result.MemoryEfficiency, "ops/MB")
}

// BenchmarkConcurrentMachines benchmarks memory usage with concurrent machines.
func (mb *MemoryBenchmark) BenchmarkConcurrentMachines(b *testing.B) {
	concurrencyLevels := []int{1, 10, 50, 100, 500}
	
	for _, concurrency := range concurrencyLevels {
		b.Run(fmt.Sprintf("Concurrency_%d", concurrency), func(b *testing.B) {
			mb.runConcurrentMachinesBenchmark(b, concurrency)
		})
	}
}

// runConcurrentMachinesBenchmark runs concurrent machines benchmark.
func (mb *MemoryBenchmark) runConcurrentMachinesBenchmark(b *testing.B, concurrency int) {
	var before, after runtime.MemStats
	
	runtime.GC()
	runtime.ReadMemStats(&before)
	
	start := time.Now()
	
	b.ResetTimer()
	b.ReportAllocs()
	
	var peak uint64
	
	for i := 0; i < b.N; i++ {
		var wg sync.WaitGroup
		machines := make([]*semantics.MachineWrapper, concurrency)
		
		// Create machines
		for j := 0; j < concurrency; j++ {
			statechart := semantics.NewStatechart(testutil.CreateSimpleStatechart())
			machine, _ := semantics.NewMachine(statechart, fmt.Sprintf("machine-%d-%d", i, j), nil)
			machines[j] = machine
		}
		
		// Start and run machines concurrently
		for j := 0; j < concurrency; j++ {
			wg.Add(1)
			go func(machine *semantics.MachineWrapper) {
				defer wg.Done()
				machine.Start()
				machine.Step("go")
				machine.Stop()
			}(machines[j])
		}
		
		wg.Wait()
		
		// Track peak memory
		var current runtime.MemStats
		runtime.ReadMemStats(&current)
		if current.Alloc > peak {
			peak = current.Alloc
		}
	}
	
	b.StopTimer()
	duration := time.Since(start)
	
	runtime.GC()
	runtime.ReadMemStats(&after)
	
	result := BenchmarkResult{
		Name:             fmt.Sprintf("ConcurrentMachines_%d", concurrency),
		Duration:         duration,
		Iterations:       b.N,
		AllocatedBytes:   int64(after.TotalAlloc - before.TotalAlloc),
		AllocatedObjects: int64(after.Mallocs - before.Mallocs),
		PeakMemory:       peak,
		GCCycles:         after.NumGC - before.NumGC,
	}
	
	if result.AllocatedBytes > 0 {
		result.MemoryEfficiency = float64(b.N * concurrency) / (float64(result.AllocatedBytes) / (1024 * 1024))
	}
	
	result.Recommendations = mb.generateConcurrencyRecommendations(result, concurrency)
	
	mb.recordResult(result)
	
	b.ReportMetric(float64(result.AllocatedBytes)/float64(b.N), "bytes/op")
	b.ReportMetric(float64(result.PeakMemory)/(1024*1024), "peak-MB")
	b.ReportMetric(result.MemoryEfficiency, "ops/MB")
}

// BenchmarkMemoryOptimizations benchmarks optimized vs unoptimized operations.
func (mb *MemoryBenchmark) BenchmarkMemoryOptimizations(b *testing.B) {
	b.Run("Unoptimized", func(b *testing.B) {
		mb.runUnoptimizedBenchmark(b)
	})
	
	b.Run("ObjectPooling", func(b *testing.B) {
		mb.runObjectPoolBenchmark(b)
	})
	
	b.Run("OptimizedStatechart", func(b *testing.B) {
		mb.runOptimizedStatechartBenchmark(b)
	})
	
	b.Run("MemoryEfficientMachine", func(b *testing.B) {
		mb.runMemoryEfficientMachineBenchmark(b)
	})
}

// runUnoptimizedBenchmark runs benchmark without optimizations.
func (mb *MemoryBenchmark) runUnoptimizedBenchmark(b *testing.B) {
	var before, after runtime.MemStats
	
	runtime.GC()
	runtime.ReadMemStats(&before)
	
	b.ResetTimer()
	b.ReportAllocs()
	
	for i := 0; i < b.N; i++ {
		statechart := semantics.NewStatechart(testutil.CreateLargeStatechart(50, 100))
		machine, _ := semantics.NewMachine(statechart, fmt.Sprintf("unopt-machine-%d", i), nil)
		machine.Start()
		
		for j := 0; j < 10; j++ {
			machine.Step(fmt.Sprintf("event_%d", j%5))
		}
		
		machine.Stop()
	}
	
	b.StopTimer()
	
	runtime.GC()
	runtime.ReadMemStats(&after)
	
	result := BenchmarkResult{
		Name:             "Unoptimized",
		Iterations:       b.N,
		AllocatedBytes:   int64(after.TotalAlloc - before.TotalAlloc),
		AllocatedObjects: int64(after.Mallocs - before.Mallocs),
		GCCycles:         after.NumGC - before.NumGC,
	}
	
	mb.recordResult(result)
	
	b.ReportMetric(float64(result.AllocatedBytes)/float64(b.N), "bytes/op")
}

// runObjectPoolBenchmark runs benchmark with object pooling.
func (mb *MemoryBenchmark) runObjectPoolBenchmark(b *testing.B) {
	pool := NewObjectPool()
	var before, after runtime.MemStats
	
	runtime.GC()
	runtime.ReadMemStats(&before)
	
	b.ResetTimer()
	b.ReportAllocs()
	
	for i := 0; i < b.N; i++ {
		statechart := semantics.NewStatechart(testutil.CreateLargeStatechart(50, 100))
		machine, _ := NewOptimizedMachine(statechart, fmt.Sprintf("opt-machine-%d", i), pool)
		machine.Start()
		
		for j := 0; j < 10; j++ {
			machine.OptimizedStep(fmt.Sprintf("event_%d", j%5))
		}
		
		machine.Stop()
	}
	
	b.StopTimer()
	
	runtime.GC()
	runtime.ReadMemStats(&after)
	
	result := BenchmarkResult{
		Name:             "ObjectPooling",
		Iterations:       b.N,
		AllocatedBytes:   int64(after.TotalAlloc - before.TotalAlloc),
		AllocatedObjects: int64(after.Mallocs - before.Mallocs),
		GCCycles:         after.NumGC - before.NumGC,
	}
	
	mb.recordResult(result)
	
	b.ReportMetric(float64(result.AllocatedBytes)/float64(b.N), "bytes/op")
}

// runOptimizedStatechartBenchmark runs benchmark with optimized statecharts.
func (mb *MemoryBenchmark) runOptimizedStatechartBenchmark(b *testing.B) {
	optimizer := NewMemoryOptimizer()
	var before, after runtime.MemStats
	
	runtime.GC()
	runtime.ReadMemStats(&before)
	
	b.ResetTimer()
	b.ReportAllocs()
	
	for i := 0; i < b.N; i++ {
		originalStatechart := testutil.CreateLargeStatechart(50, 100)
		optimizedStatechart := optimizer.OptimizeStatechart(originalStatechart)
		statechart := semantics.NewStatechart(optimizedStatechart)
		machine, _ := semantics.NewMachine(statechart, fmt.Sprintf("opt-sc-machine-%d", i), nil)
		machine.Start()
		
		for j := 0; j < 10; j++ {
			machine.Step(fmt.Sprintf("event_%d", j%5))
		}
		
		machine.Stop()
	}
	
	b.StopTimer()
	
	runtime.GC()
	runtime.ReadMemStats(&after)
	
	result := BenchmarkResult{
		Name:             "OptimizedStatechart",
		Iterations:       b.N,
		AllocatedBytes:   int64(after.TotalAlloc - before.TotalAlloc),
		AllocatedObjects: int64(after.Mallocs - before.Mallocs),
		GCCycles:         after.NumGC - before.NumGC,
	}
	
	mb.recordResult(result)
	
	b.ReportMetric(float64(result.AllocatedBytes)/float64(b.N), "bytes/op")
}

// runMemoryEfficientMachineBenchmark runs benchmark with memory-efficient machine.
func (mb *MemoryBenchmark) runMemoryEfficientMachineBenchmark(b *testing.B) {
	var before, after runtime.MemStats
	
	runtime.GC()
	runtime.ReadMemStats(&before)
	
	b.ResetTimer()
	b.ReportAllocs()
	
	for i := 0; i < b.N; i++ {
		statechart := testutil.CreateLargeStatechart(50, 100)
		machine := NewMemoryEfficientMachine(statechart, 100) // Limit history to 100 steps
		
		for j := 0; j < 10; j++ {
			machine.Step(fmt.Sprintf("event_%d", j%5))
		}
	}
	
	b.StopTimer()
	
	runtime.GC()
	runtime.ReadMemStats(&after)
	
	result := BenchmarkResult{
		Name:             "MemoryEfficientMachine",
		Iterations:       b.N,
		AllocatedBytes:   int64(after.TotalAlloc - before.TotalAlloc),
		AllocatedObjects: int64(after.Mallocs - before.Mallocs),
		GCCycles:         after.NumGC - before.NumGC,
	}
	
	mb.recordResult(result)
	
	b.ReportMetric(float64(result.AllocatedBytes)/float64(b.N), "bytes/op")
}

// BenchmarkLongRunningMachines benchmarks memory behavior over extended periods.
func (mb *MemoryBenchmark) BenchmarkLongRunningMachines(b *testing.B) {
	durations := []time.Duration{
		1 * time.Second,
		10 * time.Second,
		30 * time.Second,
	}
	
	for _, duration := range durations {
		b.Run(fmt.Sprintf("Duration_%v", duration), func(b *testing.B) {
			mb.runLongRunningBenchmark(b, duration)
		})
	}
}

// runLongRunningBenchmark runs long-running machine benchmark.
func (mb *MemoryBenchmark) runLongRunningBenchmark(b *testing.B, duration time.Duration) {
	var before, after runtime.MemStats
	
	runtime.GC()
	runtime.ReadMemStats(&before)
	
	detector := NewLeakDetector()
	detector.Start()
	defer detector.Stop()
	
	b.ResetTimer()
	
	ctx, cancel := context.WithTimeout(context.Background(), duration)
	defer cancel()
	
	// Create a machine and run it for the specified duration
	statechart := semantics.NewStatechart(testutil.CreateHierarchicalStatechart())
	machine, _ := semantics.NewMachine(statechart, "long-running-machine", nil)
	machine.Start()
	
	eventCount := 0
	events := []string{"start", "finish", "start", "finish"}
	
	// Run until context is done
	ticker := time.NewTicker(10 * time.Millisecond)
	defer ticker.Stop()
	
	for {
		select {
		case <-ctx.Done():
			goto done
		case <-ticker.C:
			machine.Step(events[eventCount%len(events)])
			eventCount++
			detector.TakeSample(fmt.Sprintf("event_%d", eventCount))
		}
	}
	
done:
	machine.Stop()
	b.StopTimer()
	
	runtime.GC()
	runtime.ReadMemStats(&after)
	
	// Generate leak detection report
	leakReport := detector.GenerateReport()
	
	result := BenchmarkResult{
		Name:             fmt.Sprintf("LongRunning_%v", duration),
		Duration:         duration,
		Iterations:       eventCount,
		AllocatedBytes:   int64(after.TotalAlloc - before.TotalAlloc),
		AllocatedObjects: int64(after.Mallocs - before.Mallocs),
		GCCycles:         after.NumGC - before.NumGC,
	}
	
	if result.AllocatedBytes > 0 {
		result.MemoryEfficiency = float64(eventCount) / (float64(result.AllocatedBytes) / (1024 * 1024))
	}
	
	result.Recommendations = mb.generateLongRunningRecommendations(result, leakReport)
	
	mb.recordResult(result)
	
	b.ReportMetric(float64(result.AllocatedBytes)/float64(eventCount), "bytes/event")
	b.ReportMetric(leakReport.MemoryGrowthRate, "bytes/sec")
	b.ReportMetric(float64(eventCount)/duration.Seconds(), "events/sec")
}

// recordResult stores a benchmark result.
func (mb *MemoryBenchmark) recordResult(result BenchmarkResult) {
	mb.mu.Lock()
	defer mb.mu.Unlock()
	mb.results = append(mb.results, result)
}

// GetResults returns all recorded benchmark results.
func (mb *MemoryBenchmark) GetResults() []BenchmarkResult {
	mb.mu.Lock()
	defer mb.mu.Unlock()
	
	results := make([]BenchmarkResult, len(mb.results))
	copy(results, mb.results)
	return results
}

// GenerateReport creates a comprehensive benchmark report.
func (mb *MemoryBenchmark) GenerateReport() BenchmarkReport {
	mb.mu.Lock()
	defer mb.mu.Unlock()
	
	if len(mb.results) == 0 {
		return BenchmarkReport{
			Summary: "No benchmark results available",
		}
	}
	
	report := BenchmarkReport{
		ResultCount: len(mb.results),
		Results:     make([]BenchmarkResult, len(mb.results)),
	}
	copy(report.Results, mb.results)
	
	// Find best and worst performers
	var bestEfficiency, worstEfficiency float64
	var bestName, worstName string
	
	for i, result := range mb.results {
		if i == 0 || result.MemoryEfficiency > bestEfficiency {
			bestEfficiency = result.MemoryEfficiency
			bestName = result.Name
		}
		if i == 0 || result.MemoryEfficiency < worstEfficiency {
			worstEfficiency = result.MemoryEfficiency
			worstName = result.Name
		}
	}
	
	report.BestPerformer = bestName
	report.WorstPerformer = worstName
	report.EfficiencyRange = fmt.Sprintf("%.2f - %.2f ops/MB", worstEfficiency, bestEfficiency)
	
	// Generate overall recommendations
	report.OverallRecommendations = mb.generateOverallRecommendations()
	
	report.Summary = fmt.Sprintf("Benchmarked %d operations. Best: %s (%.2f ops/MB), Worst: %s (%.2f ops/MB)",
		len(mb.results), bestName, bestEfficiency, worstName, worstEfficiency)
	
	return report
}

// BenchmarkReport provides a comprehensive analysis of benchmark results.
type BenchmarkReport struct {
	ResultCount             int
	Results                 []BenchmarkResult
	BestPerformer          string
	WorstPerformer         string
	EfficiencyRange        string
	OverallRecommendations []string
	Summary                string
}

// Recommendation generation methods
func (mb *MemoryBenchmark) generateCreationRecommendations(result BenchmarkResult) []string {
	var recommendations []string
	
	bytesPerOp := float64(result.AllocatedBytes) / float64(result.Iterations)
	
	if bytesPerOp > 1024*1024 { // > 1MB per operation
		recommendations = append(recommendations, "Consider object pooling for large statechart creation")
		recommendations = append(recommendations, "Review statechart structure for optimization opportunities")
	}
	
	if result.GCCycles > uint32(result.Iterations/10) { // More than 1 GC per 10 operations
		recommendations = append(recommendations, "High GC pressure detected - consider reducing allocations")
	}
	
	if result.MemoryEfficiency < 10 { // Less than 10 operations per MB
		recommendations = append(recommendations, "Low memory efficiency - consider optimizations")
	}
	
	return recommendations
}

func (mb *MemoryBenchmark) generateLifecycleRecommendations(result BenchmarkResult) []string {
	var recommendations []string
	
	bytesPerOp := float64(result.AllocatedBytes) / float64(result.Iterations)
	
	if bytesPerOp > 512*1024 { // > 512KB per lifecycle
		recommendations = append(recommendations, "Consider implementing machine pooling")
		recommendations = append(recommendations, "Review step history management")
	}
	
	if result.MemoryEfficiency < 50 { // Less than 50 operations per MB
		recommendations = append(recommendations, "Consider memory-efficient machine implementation")
	}
	
	return recommendations
}

func (mb *MemoryBenchmark) generateConcurrencyRecommendations(result BenchmarkResult, concurrency int) []string {
	var recommendations []string
	
	bytesPerMachine := float64(result.AllocatedBytes) / float64(result.Iterations * concurrency)
	
	if bytesPerMachine > 100*1024 { // > 100KB per machine
		recommendations = append(recommendations, "High per-machine memory usage in concurrent scenarios")
		recommendations = append(recommendations, "Consider shared object pools across machines")
	}
	
	if concurrency > 100 && result.GCCycles > uint32(result.Iterations) {
		recommendations = append(recommendations, "High GC pressure with many concurrent machines")
		recommendations = append(recommendations, "Consider batching operations to reduce allocation frequency")
	}
	
	return recommendations
}

func (mb *MemoryBenchmark) generateLongRunningRecommendations(result BenchmarkResult, leakReport LeakDetectionReport) []string {
	var recommendations []string
	
	if leakReport.MemoryGrowthRate > 1024*1024 { // > 1MB/sec growth
		recommendations = append(recommendations, "Significant memory growth detected in long-running test")
		recommendations = append(recommendations, "Implement periodic cleanup or compaction")
	}
	
	if result.GCCycles > 0 {
		gcFreq := float64(result.GCCycles) / result.Duration.Seconds()
		if gcFreq > 1 { // > 1 GC per second
			recommendations = append(recommendations, "High GC frequency - consider reducing allocation rate")
		}
	}
	
	return recommendations
}

func (mb *MemoryBenchmark) generateOverallRecommendations() []string {
	// Analyze all results to provide overall recommendations
	var totalBytes, totalOperations int64
	var totalGC uint32
	
	for _, result := range mb.results {
		totalBytes += result.AllocatedBytes
		totalOperations += int64(result.Iterations)
		totalGC += result.GCCycles
	}
	
	var recommendations []string
	
	if totalBytes > 0 {
		avgBytesPerOp := float64(totalBytes) / float64(totalOperations)
		if avgBytesPerOp > 512*1024 { // > 512KB average
			recommendations = append(recommendations, "Overall high memory usage - implement comprehensive optimizations")
		}
	}
	
	if totalGC > uint32(totalOperations/100) { // More than 1 GC per 100 operations
		recommendations = append(recommendations, "High overall GC pressure - focus on allocation reduction")
	}
	
	if len(recommendations) == 0 {
		recommendations = append(recommendations, "Memory usage patterns are within acceptable ranges")
	}
	
	return recommendations
}

// RunAllBenchmarks runs the complete benchmark suite.
func (mb *MemoryBenchmark) RunAllBenchmarks(b *testing.B) {
	b.Run("StatechartCreation", mb.BenchmarkStatechartCreation)
	b.Run("MachineLifecycle", mb.BenchmarkMachineLifecycle) 
	b.Run("ConcurrentMachines", mb.BenchmarkConcurrentMachines)
	b.Run("MemoryOptimizations", mb.BenchmarkMemoryOptimizations)
	
	// Only run long-running benchmarks if requested
	if testing.Short() {
		b.Skip("Skipping long-running benchmarks in short mode")
	}
	b.Run("LongRunningMachines", mb.BenchmarkLongRunningMachines)
}