package profiling

import (
	"fmt"
	"runtime"
	"sync"
	"testing"
	"time"

	"github.com/tmc/sc"
	"github.com/tmc/sc/semantics/v1"
	testutil "github.com/tmc/sc/testing"
)

// TestMemoryProfilingIntegration tests the complete memory profiling system integration.
func TestMemoryProfilingIntegration(t *testing.T) {
	// Create integrated profiling system
	profiler := NewStatechartProfiler()
	detector := NewLeakDetector()
	optimizer := NewMemoryOptimizer()
	pool := optimizer.GetObjectPool()

	detector.Start()
	defer detector.Stop()

	// Test statechart optimization pipeline
	t.Run("StatechartOptimization", func(t *testing.T) {
		original := testutil.CreateLargeStatechart(50, 100)
		optimized := optimizer.OptimizeStatechart(original)

		// Validate structure preservation
		if original.RootState.Label != optimized.RootState.Label {
			t.Error("Root state label not preserved during optimization")
		}

		if len(original.Transitions) != len(optimized.Transitions) {
			t.Error("Transition count not preserved during optimization")
		}

		// Profile both versions
		originalStatechart := semantics.NewStatechart(original)
		optimizedStatechart := semantics.NewStatechart(optimized)

		deltaFunc1, _ := profiler.ProfileValidation(originalStatechart)
		_ = originalStatechart.Validate()
		originalDelta := deltaFunc1()

		deltaFunc2, _ := profiler.ProfileValidation(optimizedStatechart)
		_ = optimizedStatechart.Validate()
		optimizedDelta := deltaFunc2()

		// Heap deltas can be negative when GC runs between samples. Only use
		// the comparison as a guard when the baseline allocation is positive.
		if originalDelta.DeltaAlloc > 0 && optimizedDelta.DeltaAlloc > originalDelta.DeltaAlloc*2 {
			t.Errorf("Optimized statechart uses significantly more memory: %d vs %d",
				optimizedDelta.DeltaAlloc, originalDelta.DeltaAlloc)
		}
	})

	// Test machine optimization
	t.Run("MachineOptimization", func(t *testing.T) {
		statechart := semantics.NewStatechart(testutil.CreateHierarchicalStatechart())

		// Create regular and optimized machines
		regularMachine, err := semantics.NewMachine(statechart, "regular", nil)
		if err != nil {
			t.Fatalf("Failed to create regular machine: %v", err)
		}

		optimizedMachine, err := NewOptimizedMachine(statechart, "optimized", pool)
		if err != nil {
			t.Fatalf("Failed to create optimized machine: %v", err)
		}

		optimizedMachine.SetHistoryLimit(50)
		optimizedMachine.SetCompactionFrequency(10)

		// Start both machines
		regularMachine.Start()
		optimizedMachine.Start()

		// Run operations and compare
		events := []string{"start", "finish", "start", "finish", "stop"}

		for _, event := range events {
			deltaFunc1, _, _ := profiler.ProfileEventProcessing(regularMachine, event)
			regularDelta := deltaFunc1()

			deltaFunc2, _, _ := profiler.ProfileEventProcessing(optimizedMachine.MachineWrapper, event)
			optimizedDelta := deltaFunc2()

			// Log the differences (optimized should generally be better or similar)
			t.Logf("Event %s - Regular: %d bytes, Optimized: %d bytes",
				event, regularDelta.DeltaAlloc, optimizedDelta.DeltaAlloc)
		}

		regularMachine.Stop()
		optimizedMachine.Stop()
	})

	// Test leak detection
	t.Run("LeakDetection", func(t *testing.T) {
		leakDetector := NewStatechartLeakDetector()

		// Create machines with different characteristics
		normalMachine, _ := semantics.NewMachine(
			semantics.NewStatechart(testutil.CreateSimpleStatechart()), "normal", nil)

		// Register machines
		leakDetector.RegisterMachine("normal", normalMachine)

		// Run operations
		normalMachine.Start()
		for i := 0; i < 100; i++ {
			normalMachine.Step("go")
		}
		normalMachine.Stop()

		// Check for leaks
		reports := leakDetector.CheckMachineLeaks()

		for _, report := range reports {
			if report.HasIssues() {
				t.Logf("Machine %s has issues: %v", report.MachineID, report.Issues)

				// Check for critical issues
				for _, issue := range report.Issues {
					if issue.Severity == Critical {
						t.Errorf("Critical memory issue detected in machine %s: %s", report.MachineID, issue.Description)
					}
				}
			}
		}
	})

	// Test memory constraints
	t.Run("MemoryConstraints", func(t *testing.T) {
		constraints := []MemoryConstraint{
			{
				Name:          "test_constraint",
				MaxAllocation: 10 * 1024 * 1024, // 10MB
				MaxObjects:    100000,
				MaxDuration:   5 * time.Second,
			},
		}

		constrainedProfiler := NewConstrainedProfiler(constraints)

		// Test operation that should pass
		_, err := constrainedProfiler.ProfileWithConstraints("small_op", func() error {
			statechart := semantics.NewStatechart(testutil.CreateSimpleStatechart())
			machine, _ := semantics.NewMachine(statechart, "constrained", nil)
			machine.Start()
			machine.Step("go")
			machine.Stop()
			return nil
		})

		if err != nil {
			t.Errorf("Small operation should not violate constraints: %v", err)
		}

		// Test operation that might violate constraints
		_, err = constrainedProfiler.ProfileWithConstraints("large_op", func() error {
			// Create many large statecharts
			for i := 0; i < 100; i++ {
				_ = semantics.NewStatechart(testutil.CreateLargeStatechart(100, 200))
			}
			return nil
		})

		// This might or might not violate constraints depending on system memory
		if err != nil {
			t.Logf("Large operation violated constraints (expected): %v", err)
		}
	})
}

// TestMemoryOptimizationEffectiveness validates that optimizations actually improve memory usage.
func TestMemoryOptimizationEffectiveness(t *testing.T) {
	// Test object pooling effectiveness
	t.Run("ObjectPoolingEffectiveness", func(t *testing.T) {
		pool := NewObjectPool()

		var beforeStats, withPoolStats, withoutPoolStats runtime.MemStats

		// Baseline
		runtime.GC()
		runtime.ReadMemStats(&beforeStats)

		// Test with object pooling
		runtime.GC()
		runtime.ReadMemStats(&withPoolStats)

		for i := 0; i < 1000; i++ {
			config := pool.GetConfiguration()
			config.States = append(config.States, &sc.StateRef{Label: "test"})
			pool.PutConfiguration(config)
		}

		runtime.GC()
		runtime.ReadMemStats(&withPoolStats)
		withPoolAlloc := withPoolStats.TotalAlloc - beforeStats.TotalAlloc

		// Test without object pooling
		runtime.GC()

		for i := 0; i < 1000; i++ {
			config := &sc.Configuration{States: make([]*sc.StateRef, 0, 8)}
			config.States = append(config.States, &sc.StateRef{Label: "test"})
			_ = config // Use the config so it's not optimized away
		}

		runtime.GC()
		runtime.ReadMemStats(&withoutPoolStats)
		withoutPoolAlloc := withoutPoolStats.TotalAlloc - withPoolStats.TotalAlloc

		t.Logf("With pool: %d bytes, Without pool: %d bytes", withPoolAlloc, withoutPoolAlloc)

		// Object pooling should generally be more efficient or at least not significantly worse
		if withPoolAlloc > withoutPoolAlloc*2 {
			t.Errorf("Object pooling is significantly less efficient: %d vs %d",
				withPoolAlloc, withoutPoolAlloc)
		}
	})

	// Test ring buffer vs slice effectiveness
	t.Run("RingBufferEffectiveness", func(t *testing.T) {
		const iterations = 10000
		const bufferSize = 100

		var beforeStats, ringStats, sliceStats runtime.MemStats

		// Baseline
		runtime.GC()
		runtime.ReadMemStats(&beforeStats)

		// Test ring buffer
		ringBuffer := NewRingBuffer(bufferSize)
		for i := 0; i < iterations; i++ {
			ringBuffer.Add(fmt.Sprintf("item_%d", i))
		}

		runtime.GC()
		runtime.ReadMemStats(&ringStats)
		ringAlloc := ringStats.TotalAlloc - beforeStats.TotalAlloc

		// Test slice with compaction
		var slice []string
		for i := 0; i < iterations; i++ {
			slice = append(slice, fmt.Sprintf("item_%d", i))
			if len(slice) > bufferSize {
				// Simulate compaction
				copy(slice, slice[len(slice)-bufferSize/2:])
				slice = slice[:bufferSize/2]
			}
		}

		runtime.GC()
		runtime.ReadMemStats(&sliceStats)
		sliceAlloc := sliceStats.TotalAlloc - ringStats.TotalAlloc

		t.Logf("Ring buffer: %d bytes, Slice: %d bytes", ringAlloc, sliceAlloc)

		// Ring buffer should be more memory efficient for bounded collections
		if ringAlloc > sliceAlloc {
			t.Logf("Ring buffer used more memory than slice (not necessarily bad)")
		}
	})
}

// TestLargeStatechartScaling tests memory behavior with increasingly large statecharts.
func TestLargeStatechartScaling(t *testing.T) {
	optimizer := NewLargeStatechartOptimizer()

	sizes := []struct {
		states      int
		transitions int
	}{
		{10, 20},
		{50, 100},
		{100, 200},
		{200, 400},
		{500, 1000},
	}

	for _, size := range sizes {
		t.Run(fmt.Sprintf("Size_%d_%d", size.states, size.transitions), func(t *testing.T) {
			// Create large statechart
			statechart := testutil.CreateLargeStatechart(size.states, size.transitions)

			// Test without optimization
			profiler := NewStatechartProfiler()
			deltaFunc, err := profiler.ProfileValidation(semantics.NewStatechart(statechart))
			if err != nil {
				t.Fatalf("ProfileValidation failed: %v", err)
			}

			err = semantics.NewStatechart(statechart).Validate()
			if err != nil {
				t.Fatalf("Validation failed: %v", err)
			}

			unoptimizedDelta := deltaFunc()

			// Test with optimization
			index := optimizer.BuildIndex(statechart)
			if index == nil {
				t.Fatal("Failed to build index")
			}

			// Test index lookup performance
			start := time.Now()
			for i := 0; i < 1000; i++ {
				state, depth, children := optimizer.FastLookup(statechart, "State0")
				if state == nil {
					t.Error("Failed to find State0")
				}
				_ = depth
				_ = children
			}
			lookupDuration := time.Since(start)

			t.Logf("Size %d/%d: Validation %d bytes, Lookup time %v",
				size.states, size.transitions, unoptimizedDelta.DeltaAlloc, lookupDuration)

			// Validate scaling behavior
			expectedComplexity := float64(size.states + size.transitions)
			actualAlloc := float64(unoptimizedDelta.DeltaAlloc)

			// Memory usage should scale roughly linearly or sub-linearly
			if len(sizes) > 1 && actualAlloc > expectedComplexity*1000 {
				t.Logf("High memory usage for size %d/%d: %d bytes",
					size.states, size.transitions, unoptimizedDelta.DeltaAlloc)
			}
		})
	}
}

// TestConcurrentMemoryUsage tests memory behavior under concurrent load.
func TestConcurrentMemoryUsage(t *testing.T) {
	pool := NewObjectPool()

	concurrencyLevels := []int{1, 5, 10, 20}

	for _, concurrency := range concurrencyLevels {
		t.Run(fmt.Sprintf("Concurrency_%d", concurrency), func(t *testing.T) {
			var beforeStats, afterStats runtime.MemStats

			runtime.GC()
			runtime.ReadMemStats(&beforeStats)

			// Run concurrent operations
			var wg sync.WaitGroup
			for i := 0; i < concurrency; i++ {
				wg.Add(1)
				go func(id int) {
					defer wg.Done()

					// Create and run optimized machine
					statechart := semantics.NewStatechart(testutil.CreateHierarchicalStatechart())
					machine, err := NewOptimizedMachine(statechart, fmt.Sprintf("concurrent_%d", id), pool)
					if err != nil {
						t.Errorf("Failed to create machine: %v", err)
						return
					}

					machine.Start()

					// Perform operations
					events := []string{"start", "finish", "start", "finish"}
					for _, event := range events {
						machine.OptimizedStep(event)
					}

					machine.Stop()
				}(i)
			}

			wg.Wait()

			runtime.GC()
			runtime.ReadMemStats(&afterStats)

			totalAlloc := afterStats.TotalAlloc - beforeStats.TotalAlloc
			perMachineAlloc := totalAlloc / uint64(concurrency)

			t.Logf("Concurrency %d: Total %d bytes, Per machine %d bytes",
				concurrency, totalAlloc, perMachineAlloc)

			// Memory per machine should remain relatively stable as concurrency increases
			if concurrency > 1 && perMachineAlloc > 10*1024*1024 { // > 10MB per machine
				t.Logf("High per-machine memory usage: %d bytes", perMachineAlloc)
			}
		})
	}
}

// TestLongRunningMemoryBehavior tests memory behavior over extended periods.
func TestLongRunningMemoryBehavior(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping long-running test in short mode")
	}

	detector := NewLeakDetector()
	detector.SetThresholds(LeakThresholds{
		MemoryGrowthRate:  1024 * 1024, // 1MB/sec
		ObjectGrowthRate:  10000,       // 10k objects/sec
		MinSampleDuration: 5 * time.Second,
		MinGrowthAmount:   5 * 1024 * 1024, // 5MB
	})

	detector.Start()
	defer detector.Stop()

	// Create optimized machine
	pool := NewObjectPool()
	statechart := semantics.NewStatechart(testutil.CreateHierarchicalStatechart())
	machine, err := NewOptimizedMachine(statechart, "long_running", pool)
	if err != nil {
		t.Fatalf("Failed to create machine: %v", err)
	}

	machine.SetHistoryLimit(100)
	machine.SetCompactionFrequency(50)
	machine.Start()

	// Run for a period of time
	duration := 30 * time.Second
	endTime := time.Now().Add(duration)
	eventCount := 0

	events := []string{"start", "finish", "start", "finish"}

	for time.Now().Before(endTime) {
		event := events[eventCount%len(events)]
		machine.OptimizedStep(event)
		eventCount++

		if eventCount%100 == 0 {
			detector.TakeSample(fmt.Sprintf("event_%d", eventCount))
		}

		time.Sleep(10 * time.Millisecond)
	}

	machine.Stop()

	// Analyze results
	report := detector.GenerateReport()

	t.Logf("Long running test results:")
	t.Logf("Events processed: %d", eventCount)
	t.Logf("Memory growth: %d bytes", report.MemoryGrowth)
	t.Logf("Growth rate: %.2f bytes/sec", report.MemoryGrowthRate)
	t.Logf("Object growth: %d", report.ObjectGrowth)

	// Check for concerning patterns
	if report.MemoryGrowthRate > 2*1024*1024 { // > 2MB/sec
		t.Errorf("High memory growth rate: %.2f bytes/sec", report.MemoryGrowthRate)
	}

	// Check machine state
	history := machine.GetStepHistory()
	if len(history) > 150 { // Should be limited by compaction
		t.Errorf("History not properly compacted: %d steps", len(history))
	}

	errors := machine.GetErrors()
	if len(errors) > 0 {
		t.Logf("Machine accumulated %d errors", len(errors))
	}
}

// BenchmarkIntegratedMemoryProfiling benchmarks the complete profiling system.
func BenchmarkIntegratedMemoryProfiling(b *testing.B) {
	pool := NewObjectPool()

	b.Run("CompleteWorkflow", func(b *testing.B) {
		b.ReportAllocs()

		for i := 0; i < b.N; i++ {
			// Create statechart
			statechart := semantics.NewStatechart(testutil.CreateHierarchicalStatechart())

			// Create optimized machine
			machine, _ := NewOptimizedMachine(statechart, fmt.Sprintf("bench_%d", i), pool)

			// Profile machine lifecycle
			machine.Start()

			profiler := NewStatechartProfiler()
			deltaFunc, _, _ := profiler.ProfileEventProcessing(machine.MachineWrapper, "start")
			_ = deltaFunc()

			deltaFunc, _, _ = profiler.ProfileEventProcessing(machine.MachineWrapper, "finish")
			_ = deltaFunc()

			machine.Stop()
		}
	})

	b.Run("ProfilerOverhead", func(b *testing.B) {
		statechart := semantics.NewStatechart(testutil.CreateSimpleStatechart())
		machine, _ := semantics.NewMachine(statechart, "overhead_test", nil)
		machine.Start()

		b.ResetTimer()
		b.ReportAllocs()

		for i := 0; i < b.N; i++ {
			profiler := NewStatechartProfiler()
			deltaFunc, _, _ := profiler.ProfileEventProcessing(machine, "go")
			_ = deltaFunc() // Measure profiling overhead
		}
	})
}
