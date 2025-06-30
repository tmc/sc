package profiling_test

import (
	"fmt"
	"testing"
	"time"

	"github.com/tmc/sc/semantics/v1/profiling"
	testutil "github.com/tmc/sc/testing"
	"github.com/tmc/sc/semantics/v1"
)

// TestMemoryProfilingDemo demonstrates the comprehensive memory profiling system
func TestMemoryProfilingDemo(t *testing.T) {
	fmt.Println("\n=== Memory Profiling and Optimization System Demo ===")
	
	// 1. Basic Memory Profiling
	fmt.Println("\n1. Basic Memory Profiling:")
	profiler := profiling.NewMemoryProfiler()
	profiler.SetBaseline("demo_start")
	
	// Allocate some memory
	data := make([]byte, 1024*1024) // 1MB
	_ = data
	
	snapshot := profiler.TakeSnapshot("after_allocation")
	baseline := profiler.GetBaseline()
	if baseline != nil {
		delta := profiling.ComputeDelta(*baseline, snapshot)
		fmt.Printf("   Memory allocated: %d bytes\n", delta.DeltaAlloc)
	}
	
	// 2. Statechart-Specific Profiling
	fmt.Println("\n2. Statechart-Specific Profiling:")
	scProfiler := profiling.NewStatechartProfiler()
	statechart := semantics.NewStatechart(testutil.CreateSimpleStatechart())
	
	// Profile validation
	deltaFunc, _ := scProfiler.ProfileValidation(statechart)
	_ = statechart.Validate()
	validationDelta := deltaFunc()
	fmt.Printf("   Validation used %d bytes\n", validationDelta.DeltaAlloc)
	
	// Profile machine creation
	deltaFunc, machine, _ := scProfiler.ProfileMachineCreation(statechart, "demo_machine")
	creationDelta := deltaFunc()
	fmt.Printf("   Machine creation used %d bytes\n", creationDelta.DeltaAlloc)
	
	// Profile event processing
	machine.Start()
	deltaFunc, stepped, _ := scProfiler.ProfileEventProcessing(machine, "go")
	eventDelta := deltaFunc()
	fmt.Printf("   Event processing used %d bytes (stepped: %v)\n", eventDelta.DeltaAlloc, stepped)
	machine.Stop()
	
	// 3. Object Pooling
	fmt.Println("\n3. Object Pooling:")
	pool := profiling.NewObjectPool()
	
	before := scProfiler.GetMemoryProfiler().TakeSnapshot("before_pooling")
	for i := 0; i < 100; i++ {
		config := pool.GetConfiguration()
		// Use config...
		pool.PutConfiguration(config)
	}
	after := scProfiler.GetMemoryProfiler().TakeSnapshot("after_pooling")
	poolingDelta := profiling.ComputeDelta(before, after)
	fmt.Printf("   100 pooled operations used %d bytes\n", poolingDelta.DeltaAlloc)
	
	// 4. Large Statechart Optimization
	fmt.Println("\n4. Large Statechart Optimization:")
	largeStatechart := testutil.CreateLargeStatechart(100, 200)
	optimizer := profiling.NewLargeStatechartMemoryOptimizer()
	
	optimized, report := optimizer.OptimizeLargeStatechart(largeStatechart)
	fmt.Printf("   Optimized statechart from %d to %d bytes (%.2f%% reduction)\n", 
		report.OriginalSize, report.OptimizedSize, report.ReductionPercent)
	fmt.Printf("   Strings interned: %d, States optimized: %d\n", 
		report.Stats.StringsInterned, report.Stats.StatesOptimized)
	_ = optimized
	
	// 5. Memory-Efficient Indexing
	fmt.Println("\n5. Memory-Efficient Indexing:")
	index := profiling.BuildMemoryEfficientIndex(largeStatechart)
	
	parentLabel, depth, childCount, found := index.GetStateInfo("State10")
	if found {
		fmt.Printf("   State10: parent=%s, depth=%d, children=%d\n", parentLabel, depth, childCount)
	}
	
	transitions := index.GetTransitionsFrom("State0")
	fmt.Printf("   State0 has %d outgoing transitions\n", len(transitions))
	fmt.Printf("   Index uses %d bytes of memory\n", index.GetMemoryUsage())
	
	// 6. Memory Budget Management
	fmt.Println("\n6. Memory Budget Management:")
	budget := profiling.NewMemoryBudgetManager(10 * 1024 * 1024) // 10MB
	
	if ok, available := budget.CheckBudget(5 * 1024 * 1024); ok {
		fmt.Printf("   Can allocate 5MB (available: %d bytes)\n", available)
		budget.AllocateMemory(5 * 1024 * 1024)
		fmt.Printf("   Current usage: %d bytes\n", budget.GetCurrentUsage())
	}
	
	// 7. Leak Detection Setup
	fmt.Println("\n7. Leak Detection:")
	detector := profiling.NewLeakDetector()
	detector.SetThresholds(profiling.LeakThresholds{
		MemoryGrowthRate:  1024 * 1024,
		ObjectGrowthRate:  10000,
		MinSampleDuration: time.Second,
		MinGrowthAmount:   1024,
	})
	
	detector.AddCallback(func(leak profiling.DetectedLeak) {
		fmt.Printf("   Leak detected: %s (severity: %s)\n", leak.Description, leak.Severity)
	})
	
	detector.Start()
	defer detector.Stop()
	
	// Simulate some memory usage
	var leakyData [][]byte
	for i := 0; i < 5; i++ {
		leakyData = append(leakyData, make([]byte, 512))
		detector.TakeSample(fmt.Sprintf("sample_%d", i))
		time.Sleep(10 * time.Millisecond)
	}
	
	// Generate comprehensive report
	report2 := detector.GenerateReport()
	fmt.Printf("   Monitoring summary: %s\n", report2.Summary)
	
	// 8. Memory Pattern Analysis
	fmt.Println("\n8. Memory Pattern Analysis:")
	analyzer := profiling.NewMemoryAnalyzer()
	
	// Create a sample profile
	profile, _ := scProfiler.ProfileStatechartLifecycle(
		testutil.CreateHierarchicalStatechart,
		[]string{"start", "finish"},
	)
	analyzer.AddProfile(*profile)
	
	analysis := analyzer.AnalyzePatterns()
	fmt.Printf("   Analyzed %d profiles\n", analysis.ProfileCount)
	for op, stats := range analysis.Operations {
		fmt.Printf("     %s: avg %d bytes, max %d bytes\n", op, stats.AvgAlloc, stats.MaxAlloc)
	}
	
	// Keep reference to prevent GC optimization
	_ = leakyData
	
	fmt.Println("\n=== Demo Complete ===")
	fmt.Println("\nThe memory profiling system provides:")
	fmt.Println("• Comprehensive memory tracking and profiling")
	fmt.Println("• Real-time leak detection with customizable thresholds")
	fmt.Println("• Object pooling for reduced GC pressure")
	fmt.Println("• Large statechart optimization with string interning")
	fmt.Println("• Memory-efficient indexing for O(1) lookups")
	fmt.Println("• Budget management with callback notifications")
	fmt.Println("• Pattern analysis across multiple profiling sessions")
	fmt.Println("• Production-ready monitoring and reporting")
}