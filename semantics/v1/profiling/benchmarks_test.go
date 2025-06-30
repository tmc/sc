package profiling_test

import (
	"testing"

	"github.com/tmc/sc/semantics/v1/profiling"
)

// BenchmarkMemorySuite runs the complete memory benchmark suite
func BenchmarkMemorySuite(b *testing.B) {
	benchmark := profiling.NewMemoryBenchmark()
	benchmark.RunAllBenchmarks(b)
}

// Individual benchmarks for specific scenarios

// BenchmarkStatechartCreation benchmarks memory usage during statechart creation
func BenchmarkStatechartCreation(b *testing.B) {
	benchmark := profiling.NewMemoryBenchmark()
	benchmark.BenchmarkStatechartCreation(b)
}

// BenchmarkMachineLifecycle benchmarks complete machine lifecycle memory usage
func BenchmarkMachineLifecycle(b *testing.B) {
	benchmark := profiling.NewMemoryBenchmark()
	benchmark.BenchmarkMachineLifecycle(b)
}

// BenchmarkConcurrentMachines benchmarks memory usage with concurrent machines
func BenchmarkConcurrentMachines(b *testing.B) {
	benchmark := profiling.NewMemoryBenchmark()
	benchmark.BenchmarkConcurrentMachines(b)
}

// BenchmarkMemoryOptimizations compares optimized vs unoptimized operations
func BenchmarkMemoryOptimizations(b *testing.B) {
	benchmark := profiling.NewMemoryBenchmark()
	benchmark.BenchmarkMemoryOptimizations(b)
}

// BenchmarkLongRunningMachines benchmarks memory behavior over extended periods
func BenchmarkLongRunningMachines(b *testing.B) {
	if testing.Short() {
		b.Skip("Skipping long-running benchmarks in short mode")
	}
	benchmark := profiling.NewMemoryBenchmark()
	benchmark.BenchmarkLongRunningMachines(b)
}