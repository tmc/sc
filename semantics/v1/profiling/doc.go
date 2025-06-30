// Package profiling provides comprehensive memory profiling and optimization tools for statechart operations.
//
// # Overview
//
// This package implements memory profiling, optimization, and leak detection capabilities
// specifically designed for statechart-based applications. It provides tools to:
//
//   - Profile memory usage during statechart operations
//   - Optimize memory usage for large statecharts and high-concurrency scenarios
//   - Detect and prevent memory leaks in long-running machines
//   - Benchmark memory performance and track improvements
//
// # Memory Profiling
//
// The memory profiling system captures detailed memory usage information during statechart
// operations. Use MemoryProfiler for general profiling and StatechartProfiler for
// statechart-specific operations:
//
//	profiler := profiling.NewStatechartProfiler()
//	
//	// Profile statechart validation
//	deltaFunc, err := profiler.ProfileValidation(statechart)
//	err = statechart.Validate()
//	delta := deltaFunc()
//	fmt.Printf("Validation used %d bytes\n", delta.DeltaAlloc)
//	
//	// Profile machine lifecycle
//	profile, err := profiler.ProfileStatechartLifecycle(
//		func() *sc.Statechart { return createStatechart() },
//		[]string{"event1", "event2", "event3"},
//	)
//
// # Memory Optimization
//
// Several optimization strategies are available for reducing memory usage:
//
// ## Object Pooling
//
// Use ObjectPool to reuse frequently allocated objects:
//
//	pool := profiling.NewObjectPool()
//	config := pool.GetConfiguration()
//	// Use configuration...
//	pool.PutConfiguration(config)
//
// ## Optimized Machines
//
// Use OptimizedMachineWrapper for automatic memory optimizations:
//
//	pool := profiling.NewObjectPool()
//	machine, err := profiling.NewOptimizedMachine(statechart, "machine_id", pool)
//	machine.SetHistoryLimit(1000)        // Limit step history
//	machine.SetCompactionFrequency(100)  // Compact every 100 steps
//
// ## Memory-Efficient Machines
//
// For extreme memory efficiency, use MemoryEfficientMachine:
//
//	machine := profiling.NewMemoryEfficientMachine(statechart, 500) // 500 step history
//	err := machine.Step("event")
//
// ## Large Statechart Optimization
//
// Use LargeStatechartOptimizer for statecharts with many states:
//
//	optimizer := profiling.NewLargeStatechartOptimizer()
//	index := optimizer.BuildIndex(statechart)
//	state, depth, children := optimizer.FastLookup(statechart, "state_label")
//	transitions := optimizer.GetTransitionsForState(statechart, "state_label")
//
// # Memory Leak Detection
//
// The leak detection system monitors memory usage patterns and identifies potential leaks:
//
//	detector := profiling.NewLeakDetector()
//	detector.SetThresholds(profiling.LeakThresholds{
//		MemoryGrowthRate:    1024 * 1024,    // 1MB/sec
//		ObjectGrowthRate:    10000,          // 10k objects/sec
//		MinSampleDuration:   30 * time.Second,
//		MinGrowthAmount:     10 * 1024 * 1024, // 10MB
//	})
//	
//	detector.AddCallback(func(leak profiling.DetectedLeak) {
//		fmt.Printf("Leak detected: %s (%s)\n", leak.Description, leak.Severity)
//		for _, suggestion := range leak.Suggestions {
//			fmt.Printf("  - %s\n", suggestion)
//		}
//	})
//	
//	detector.Start()
//	defer detector.Stop()
//
// ## Statechart-Specific Leak Detection
//
// Use StatechartLeakDetector for machine-specific monitoring:
//
//	detector := profiling.NewStatechartLeakDetector()
//	detector.RegisterMachine("machine1", machine)
//	
//	// Periodically check for leaks
//	reports := detector.CheckMachineLeaks()
//	for _, report := range reports {
//		if report.HasIssues() {
//			fmt.Printf("Machine %s has issues: %v\n", report.MachineID, report.Issues)
//		}
//	}
//
// # Memory Benchmarking
//
// Use MemoryBenchmark for comprehensive performance testing:
//
//	func BenchmarkMyStatechart(b *testing.B) {
//		benchmark := profiling.NewMemoryBenchmark()
//		benchmark.BenchmarkStatechartCreation(b)
//		benchmark.BenchmarkMachineLifecycle(b)
//		benchmark.BenchmarkConcurrentMachines(b)
//		
//		report := benchmark.GenerateReport()
//		fmt.Printf("Best performer: %s\n", report.BestPerformer)
//		fmt.Printf("Efficiency range: %s\n", report.EfficiencyRange)
//	}
//
// # Memory Monitoring
//
// For real-time monitoring, use MemoryMonitor:
//
//	monitor := profiling.NewMemoryMonitor()
//	monitor.SetThreshold("high_memory", 100*1024*1024, func(snapshot profiling.MemorySnapshot) {
//		fmt.Printf("High memory usage: %d bytes\n", snapshot.Alloc)
//	})
//	monitor.StartMonitoring(5 * time.Second)
//
// # Best Practices
//
// ## For Large Statecharts (>100 states)
//
//   - Use LargeStatechartOptimizer for fast lookups
//   - Implement statechart optimization with MemoryOptimizer
//   - Consider breaking very large statecharts into smaller, composable units
//   - Use indexed access patterns instead of linear searches
//
// ## For High-Concurrency Applications
//
//   - Use object pooling to reduce allocation pressure
//   - Implement machine pooling for frequently created/destroyed machines
//   - Monitor GC pressure and tune GOGC if necessary
//   - Consider batching operations to reduce per-operation overhead
//
// ## For Long-Running Applications
//
//   - Set appropriate history limits to prevent unbounded growth
//   - Implement periodic compaction of machine state
//   - Use leak detection to identify gradual memory growth
//   - Monitor and clear accumulated errors periodically
//
// ## For Memory-Constrained Environments
//
//   - Use MemoryEfficientMachine instead of regular machines
//   - Implement aggressive history compaction
//   - Consider disabling detailed tracing and logging
//   - Use minimal context data structures
//
// # Performance Guidelines
//
// ## Memory Allocation Patterns
//
// Prefer these patterns for better memory efficiency:
//
//	// Good: Pre-allocate with known capacity
//	states := make([]*sc.StateRef, 0, expectedSize)
//	
//	// Good: Reuse objects via pooling
//	config := pool.GetConfiguration()
//	defer pool.PutConfiguration(config)
//	
//	// Good: Use ring buffers for bounded history
//	history := profiling.NewRingBuffer(1000)
//	
//	// Avoid: Unbounded slice growth
//	var history []*sc.Step // Can grow indefinitely
//	
//	// Avoid: Frequent small allocations in hot paths
//	for event := range events {
//		config := &sc.Configuration{} // Allocates every iteration
//	}
//
// ## Memory Management
//
//	// Set explicit limits
//	machine.SetHistoryLimit(1000)
//	machine.SetCompactionFrequency(100)
//	
//	// Use constrained profiling for validation
//	constraints := []profiling.MemoryConstraint{
//		{Name: "memory_limit", MaxAllocation: 10 * 1024 * 1024}, // 10MB
//		{Name: "time_limit", MaxDuration: time.Second},
//	}
//	profiler := profiling.NewConstrainedProfiler(constraints)
//	
//	// Monitor continuously in production
//	detector := profiling.NewLeakDetector()
//	detector.Start()
//
// # Integration Examples
//
// ## With HTTP Services
//
//	func handleStatechartRequest(w http.ResponseWriter, r *http.Request) {
//		profiler := profiling.NewStatechartProfiler()
//		
//		deltaFunc, machine, err := profiler.ProfileMachineCreation(statechart, "http_machine")
//		if err != nil {
//			http.Error(w, err.Error(), 500)
//			return
//		}
//		defer func() {
//			delta := deltaFunc()
//			log.Printf("Machine creation used %d bytes", delta.DeltaAlloc)
//		}()
//		
//		// Process request with machine...
//	}
//
// ## With Background Workers
//
//	func worker() {
//		detector := profiling.NewStatechartLeakDetector()
//		detector.Start()
//		defer detector.Stop()
//		
//		pool := profiling.NewObjectPool()
//		
//		for task := range tasks {
//			machine, _ := profiling.NewOptimizedMachine(statechart, task.ID, pool)
//			machine.Start()
//			
//			// Process task...
//			
//			machine.Stop()
//			detector.RegisterMachine(task.ID, machine)
//		}
//		
//		// Periodic leak checking
//		ticker := time.NewTicker(time.Minute)
//		go func() {
//			for range ticker.C {
//				reports := detector.CheckMachineLeaks()
//				for _, report := range reports {
//					if report.HasIssues() {
//						log.Printf("Leak detected in machine %s", report.MachineID)
//					}
//				}
//			}
//		}()
//	}
//
// ## With Testing
//
//	func TestStatechartMemoryUsage(t *testing.T) {
//		profiler := profiling.NewStatechartProfiler()
//		
//		// Test with constraints
//		constraints := []profiling.MemoryConstraint{
//			{Name: "test_limit", MaxAllocation: 1024 * 1024}, // 1MB
//		}
//		constrained := profiling.NewConstrainedProfiler(constraints)
//		
//		_, err := constrained.ProfileWithConstraints("test_operation", func() error {
//			machine, _ := semantics.NewMachine(statechart, "test", nil)
//			machine.Start()
//			machine.Step("test_event")
//			machine.Stop()
//			return nil
//		})
//		
//		if err != nil {
//			t.Errorf("Memory constraint violated: %v", err)
//		}
//	}
//
// # Troubleshooting
//
// ## High Memory Usage
//
//   1. Enable leak detection to identify growth patterns
//   2. Check step history sizes - implement limits if needed
//   3. Review context data for accumulating values
//   4. Use object pooling for frequently allocated objects
//   5. Consider statechart structure optimization
//
// ## High GC Pressure
//
//   1. Use memory profiling to identify allocation hot spots
//   2. Implement object pooling for high-frequency allocations
//   3. Reduce temporary object creation in event processing
//   4. Use pre-allocated slices with known capacity
//   5. Consider adjusting GOGC environment variable
//
// ## Memory Leaks
//
//   1. Use leak detection with appropriate thresholds
//   2. Check for unbounded slice/map growth
//   3. Verify proper cleanup of event listeners
//   4. Review goroutine lifecycle management
//   5. Monitor long-running machine state accumulation
//
// ## Performance Degradation
//
//   1. Use benchmarking to identify performance regressions
//   2. Profile memory allocation patterns
//   3. Check for inefficient lookups in large statecharts
//   4. Review concurrency patterns and contention
//   5. Consider memory-efficient machine implementations
//
// This package provides the foundation for building memory-efficient, scalable
// statechart applications that can handle large statecharts, high concurrency,
// and long-running operations while maintaining optimal memory usage patterns.
package profiling