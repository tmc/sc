package profiling_test

import (
	"context"
	"fmt"
	"log"
	"sync"
	"time"

	"github.com/tmc/sc"
	"github.com/tmc/sc/semantics/v1"
	"github.com/tmc/sc/semantics/v1/profiling"
	testutil "github.com/tmc/sc/testing"
)

// Example demonstrates basic memory profiling usage
func ExampleMemoryProfiler() {
	profiler := profiling.NewMemoryProfiler()
	
	// Set a baseline before operations
	profiler.SetBaseline("before_operations")
	
	// Perform some operations
	data := make([]byte, 1024*1024) // Allocate 1MB
	_ = data
	
	// Take a snapshot after operations
	snapshot := profiler.TakeSnapshot("after_allocation")
	
	// Compute memory delta
	baseline := profiler.GetBaseline()
	if baseline != nil {
		delta := profiling.ComputeDelta(*baseline, snapshot)
		fmt.Printf("Memory allocated: %d bytes\n", delta.DeltaAlloc)
	}
	
	// Output:
	// Memory allocated: 1048576 bytes
}

// ExampleStatechartProfiler demonstrates statechart-specific profiling
func ExampleStatechartProfiler() {
	profiler := profiling.NewStatechartProfiler()
	statechart := semantics.NewStatechart(testutil.CreateSimpleStatechart())
	
	// Profile validation
	deltaFunc, _ := profiler.ProfileValidation(statechart)
	_ = statechart.Validate()
	validationDelta := deltaFunc()
	
	// Profile machine creation and operation
	deltaFunc, machine, _ := profiler.ProfileMachineCreation(statechart, "example_machine")
	creationDelta := deltaFunc()
	
	// Profile event processing
	machine.Start()
	deltaFunc, _, _ = profiler.ProfileEventProcessing(machine, "go")
	eventDelta := deltaFunc()
	machine.Stop()
	
	fmt.Printf("Validation memory: %d bytes\n", validationDelta.DeltaAlloc)
	fmt.Printf("Machine creation memory: %d bytes\n", creationDelta.DeltaAlloc)
	fmt.Printf("Event processing memory: %d bytes\n", eventDelta.DeltaAlloc)
}

// ExampleLeakDetector demonstrates memory leak detection
func ExampleLeakDetector() {
	detector := profiling.NewLeakDetector()
	
	// Configure thresholds
	detector.SetThresholds(profiling.LeakThresholds{
		MemoryGrowthRate:  1024 * 1024,    // 1MB/sec
		ObjectGrowthRate:  10000,          // 10k objects/sec
		MinSampleDuration: 30 * time.Second,
		MinGrowthAmount:   10 * 1024 * 1024, // 10MB
	})
	
	// Add callback for leak notifications
	detector.AddCallback(func(leak profiling.DetectedLeak) {
		log.Printf("Memory leak detected: %s (severity: %s)", 
			leak.Description, leak.Severity)
		for _, suggestion := range leak.Suggestions {
			log.Printf("  Suggestion: %s", suggestion)
		}
	})
	
	// Start monitoring
	detector.Start()
	defer detector.Stop()
	
	// Your application code here...
}

// ExampleObjectPool demonstrates object pooling for efficiency
func ExampleObjectPool() {
	pool := profiling.NewObjectPool()
	
	// Reuse configuration objects
	for i := 0; i < 1000; i++ {
		config := pool.GetConfiguration()
		
		// Use the configuration
		config.States = append(config.States, &sc.StateRef{
			Label: fmt.Sprintf("state_%d", i),
		})
		
		// Return to pool when done
		pool.PutConfiguration(config)
	}
	
	fmt.Println("Processed 1000 configurations with object pooling")
	// Output:
	// Processed 1000 configurations with object pooling
}

// ExampleOptimizedMachine demonstrates memory-optimized machine usage
func Example_optimizedMachine() {
	pool := profiling.NewObjectPool()
	statechart := semantics.NewStatechart(testutil.CreateHierarchicalStatechart())
	
	// Create optimized machine
	machine, err := profiling.NewOptimizedMachine(statechart, "optimized_example", pool)
	if err != nil {
		log.Fatal(err)
	}
	
	// Configure memory limits
	machine.SetHistoryLimit(1000)        // Keep only last 1000 steps
	machine.SetCompactionFrequency(100)  // Compact every 100 steps
	
	// Use the machine
	machine.Start()
	for i := 0; i < 500; i++ {
		machine.OptimizedStep("start")
		machine.OptimizedStep("finish")
	}
	machine.Stop()
	
	// History is automatically compacted
	history := machine.GetStepHistory()
	fmt.Printf("History size after 1000 steps: %d\n", len(history))
}

// ExampleLargeStatechartOptimizer demonstrates optimization for large statecharts
func ExampleLargeStatechartOptimizer() {
	// Create a large statechart
	largeStatechart := testutil.CreateLargeStatechart(1000, 2000)
	
	// Build memory-efficient index for fast lookups
	index := profiling.BuildMemoryEfficientIndex(largeStatechart)
	
	// Use the index for fast lookups
	parentLabel, depth, childCount, found := index.GetStateInfo("State500")
	if found {
		fmt.Printf("Found State500: parent=%s, depth=%d, children=%d\n", parentLabel, depth, childCount)
	}
	
	// Get transitions efficiently
	transitions := index.GetTransitionsFrom("State100")
	fmt.Printf("State100 has %d outgoing transitions\n", len(transitions))
	
	// Use the optimizer for the statechart
	optimizer := profiling.NewMemoryOptimizer()
	optimizedSC := optimizer.OptimizeStatechart(largeStatechart)
	_ = optimizedSC
}

// ExampleMemoryBudgetManager demonstrates memory budget enforcement
func ExampleMemoryBudgetManager() {
	// Set a 100MB memory budget
	budget := profiling.NewMemoryBudgetManager(100 * 1024 * 1024)
	
	// Add callback for budget warnings
	budget.AddCallback(func(current, max uint64, exceeded bool) {
		if exceeded {
			log.Printf("Warning: Memory usage at %.1f%% of budget", 
				float64(current)/float64(max)*100)
		}
	})
	
	// Check before allocation
	estimatedSize := uint64(20 * 1024 * 1024) // 20MB
	if ok, available := budget.CheckBudget(estimatedSize); !ok {
		log.Printf("Cannot allocate %d bytes, only %d available", 
			estimatedSize, available)
		return
	}
	
	// Allocate memory
	if err := budget.AllocateMemory(estimatedSize); err != nil {
		log.Printf("Allocation failed: %v", err)
		return
	}
	
	// Do work...
	
	// Release when done
	budget.ReleaseMemory(estimatedSize)
}

// ExampleStatechartLeakDetector demonstrates statechart-specific leak detection
func ExampleStatechartLeakDetector() {
	detector := profiling.NewStatechartLeakDetector()
	detector.Start()
	defer detector.Stop()
	
	// Create statechart and machine
	statechart := semantics.NewStatechart(testutil.CreateHierarchicalStatechart())
	machine, _ := semantics.NewMachine(statechart, "leak_example", nil)
	
	// Register for monitoring
	detector.RegisterMachine("leak_example", machine)
	
	// Simulate usage
	machine.Start()
	for i := 0; i < 10000; i++ {
		machine.Step("start")
		machine.Step("finish")
	}
	
	// Check for leaks
	reports := detector.CheckMachineLeaks()
	for _, report := range reports {
		if report.HasIssues() {
			fmt.Printf("Machine %s has %d issues\n", 
				report.MachineID, len(report.Issues))
		}
	}
	
	// Cleanup
	machine.Stop()
	detector.UnregisterMachine("leak_example")
}

// example_concurrentOptimization demonstrates concurrent statechart optimization
func example_concurrentOptimization() {
	// Create optimizer with 4 workers
	optimizer := profiling.NewConcurrentStatechartOptimizer(4)
	
	// Create a very large statechart
	veryLargeStatechart := testutil.CreateLargeStatechart(5000, 10000)
	
	// Optimize concurrently
	optimized, report := optimizer.OptimizeConcurrently(veryLargeStatechart)
	
	fmt.Printf("Optimization completed in %v\n", report.Duration)
	fmt.Printf("States optimized: %d\n", report.Stats.StatesOptimized)
	fmt.Printf("Transitions optimized: %d\n", report.Stats.TransitionsOptimized)
	
	// Use the optimized statechart
	_ = optimized
}

// example_memoryEfficientCache demonstrates the memory-efficient cache
func example_memoryEfficientCache() {
	// Create cache with size and memory limits
	cache := profiling.NewMemoryEfficientStatechartCache(
		100,                    // Max 100 entries
		50*1024*1024,          // Max 50MB memory
	)
	
	// Cache statecharts
	for i := 0; i < 10; i++ {
		sc := testutil.CreateLargeStatechart(50+i*10, 100+i*20)
		cache.Put(fmt.Sprintf("statechart_%d", i), sc)
	}
	
	// Retrieve from cache
	if sc, found := cache.Get("statechart_5"); found {
		fmt.Println("Retrieved statechart_5 from cache")
		_ = sc
	}
	
	// Check cache statistics
	stats := cache.GetStats()
	fmt.Printf("Cache: %d entries, %.1f%% memory utilization\n", 
		stats.Size, stats.UtilizationPct)
}

// example_productionSetup demonstrates a production-ready memory monitoring setup
func example_productionSetup() {
	// Create a comprehensive monitoring setup
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	
	// 1. Set up leak detection
	leakDetector := profiling.NewStatechartLeakDetector()
	leakDetector.SetProductionThresholds()
	leakDetector.Start()
	defer leakDetector.Stop()
	
	// 2. Set up continuous profiling
	continuousProfiler := profiling.NewContinuousProfiler(5 * time.Minute)
	continuousProfiler.Start()
	defer continuousProfiler.Stop()
	
	// 3. Set up memory budget
	memoryBudget := profiling.NewMemoryBudgetManager(1024 * 1024 * 1024) // 1GB
	_ = memoryBudget // Use budget for allocations
	
	// 4. Create pooled machine factory
	factory := profiling.NewMemoryPooledMachineFactory(512 * 1024 * 1024) // 512MB for machines
	_ = factory // Use factory for machine creation
	
	// 5. Set up periodic reporting
	go func() {
		ticker := time.NewTicker(time.Hour)
		defer ticker.Stop()
		
		for {
			select {
			case <-ctx.Done():
				return
			case <-ticker.C:
				// Generate memory report
				profiler := continuousProfiler.GetProfiler()
				report := profiler.GetMemoryProfiler().GenerateReport()
				
				log.Printf("Hourly Memory Report:")
				log.Printf("  Peak Memory: %d MB", report.PeakAlloc/(1024*1024))
				log.Printf("  Total Allocated: %d MB", report.TotalAllocated/(1024*1024))
				log.Printf("  GC Cycles: %d", report.GCCount)
				
				// Check for machine leaks
				machineReports := leakDetector.CheckMachineLeaks()
				for _, mr := range machineReports {
					if mr.HasIssues() {
						log.Printf("  Machine %s has %d issues", 
							mr.MachineID, len(mr.Issues))
					}
				}
			}
		}
	}()
	
	// Application runs here...
	fmt.Println("Production memory monitoring setup complete")
}

// example_highConcurrencyOptimization demonstrates optimization for high-concurrency scenarios
func example_highConcurrencyOptimization() {
	// Shared resources for all machines
	pool := profiling.NewObjectPool()
	_ = pool // Pool would be used in machine creation
	lsOptimizer := profiling.NewLargeStatechartMemoryOptimizer()
	factory := profiling.NewMemoryPooledMachineFactory(500 * 1024 * 1024)
	
	// Pre-optimize the statechart
	statechart := testutil.CreateLargeStatechart(100, 200)
	optimizedSC, _ := lsOptimizer.OptimizeLargeStatechart(statechart)
	
	// Create many concurrent machines
	var wg sync.WaitGroup
	machineCount := 100
	
	for i := 0; i < machineCount; i++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			
			// Create optimized machine from factory
			machine, err := factory.CreateOptimizedMachine(
				optimizedSC, 
				fmt.Sprintf("concurrent_%d", id),
				500, // Limited history
			)
			if err != nil {
				log.Printf("Failed to create machine %d: %v", id, err)
				return
			}
			
			// Run machine
			machine.Start()
			for j := 0; j < 100; j++ {
				machine.Step(fmt.Sprintf("event_%d", j%10))
			}
			machine.Stop()
			
			// Return resources
			factory.ReleaseMachine(machine, 5*1024*1024) // Estimate 5MB
		}(i)
	}
	
	wg.Wait()
	
	// Report factory statistics
	stats := factory.GetMemoryStats()
	fmt.Printf("Processed %d concurrent machines\n", machineCount)
	fmt.Printf("Memory usage: %d/%d MB (%.1f%%)\n",
		stats.CurrentUsage/(1024*1024),
		stats.MaxMemory/(1024*1024),
		float64(stats.CurrentUsage)/float64(stats.MaxMemory)*100)
}