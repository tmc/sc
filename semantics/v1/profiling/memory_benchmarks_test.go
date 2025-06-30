package profiling_test

import (
	"fmt"
	"runtime"
	"testing"
	"time"

	"github.com/tmc/sc"
	"github.com/tmc/sc/semantics/v1"
	"github.com/tmc/sc/semantics/v1/profiling"
	testutil "github.com/tmc/sc/testing"
)

// BenchmarkLargeStatechartMemory benchmarks memory usage for large statecharts.
func BenchmarkLargeStatechartMemory(b *testing.B) {
	sizes := []struct {
		name            string
		states          int
		transitions     int
		depth           int
		parallelRegions int
	}{
		{"Small_10_20", 10, 20, 3, 0},
		{"Medium_100_200", 100, 200, 5, 2},
		{"Large_1000_2000", 1000, 2000, 7, 4},
		{"XLarge_5000_10000", 5000, 10000, 10, 8},
		{"XXLarge_10000_20000", 10000, 20000, 12, 10},
		{"Massive_50000_100000", 50000, 100000, 15, 20},
	}

	for _, size := range sizes {
		b.Run(size.name, func(b *testing.B) {
			benchmarkStatechartMemory(b, size.states, size.transitions, size.depth, size.parallelRegions)
		})
	}
}

func benchmarkStatechartMemory(b *testing.B, states, transitions, depth, parallelRegions int) {
	// Create profiler
	profiler := profiling.NewStatechartProfiler()
	
	// Force GC and get baseline
	runtime.GC()
	var m0 runtime.MemStats
	runtime.ReadMemStats(&m0)
	
	b.ResetTimer()
	b.ReportAllocs()
	
	var totalAlloc uint64
	var peakAlloc uint64
	
	for i := 0; i < b.N; i++ {
		// Create large statechart
		statechart := createComplexStatechart(states, transitions, depth, parallelRegions)
		
		// Wrap with semantics
		wrapped := semantics.NewStatechart(statechart)
		
		// Validate
		err := wrapped.Validate()
		if err != nil {
			b.Fatalf("Validation failed: %v", err)
		}
		
		// Track memory
		var m runtime.MemStats
		runtime.ReadMemStats(&m)
		
		if m.Alloc > peakAlloc {
			peakAlloc = m.Alloc
		}
		totalAlloc += m.TotalAlloc - m0.TotalAlloc
		
		// Force cleanup
		statechart = nil
		wrapped = nil
	}
	
	b.StopTimer()
	
	// Calculate and report metrics
	avgAlloc := totalAlloc / uint64(b.N)
	b.ReportMetric(float64(avgAlloc), "bytes/op")
	b.ReportMetric(float64(peakAlloc)/(1024*1024), "peak-MB")
	b.ReportMetric(float64(avgAlloc)/float64(states), "bytes/state")
	b.ReportMetric(float64(avgAlloc)/float64(transitions), "bytes/transition")
	
	// Generate memory report
	report := profiler.GetMemoryProfiler().GenerateReport()
	if len(report.Recommendations) > 0 {
		b.Logf("Memory recommendations: %v", report.Recommendations)
	}
}

// BenchmarkStatechartOptimization benchmarks the effectiveness of memory optimization.
func BenchmarkStatechartOptimization(b *testing.B) {
	optimizer := profiling.NewLargeStatechartMemoryOptimizer()
	
	sizes := []struct {
		name   string
		states int
		trans  int
	}{
		{"Small", 100, 200},
		{"Medium", 1000, 2000},
		{"Large", 5000, 10000},
		{"XLarge", 10000, 20000},
	}
	
	for _, size := range sizes {
		b.Run(size.name, func(b *testing.B) {
			// Create statechart
			statechart := testutil.CreateLargeStatechart(size.states, size.trans)
			
			b.ResetTimer()
			b.ReportAllocs()
			
			var totalReduction int64
			var totalDuration time.Duration
			
			for i := 0; i < b.N; i++ {
				optimized, report := optimizer.OptimizeLargeStatechart(statechart)
				_ = optimized
				
				totalReduction += report.SizeReduction
				totalDuration += report.Duration
			}
			
			b.StopTimer()
			
			avgReduction := totalReduction / int64(b.N)
			avgDuration := totalDuration / time.Duration(b.N)
			
			b.ReportMetric(float64(avgReduction), "bytes-saved/op")
			b.ReportMetric(float64(avgDuration.Microseconds()), "μs/op")
			b.ReportMetric(float64(avgReduction)/float64(size.states), "bytes-saved/state")
		})
	}
}

// BenchmarkMachineMemoryScaling benchmarks memory usage as machines scale.
func BenchmarkMachineMemoryScaling(b *testing.B) {
	machineCounts := []int{1, 10, 50, 100, 500, 1000}
	
	for _, count := range machineCounts {
		b.Run(fmt.Sprintf("Machines_%d", count), func(b *testing.B) {
			benchmarkMachineScaling(b, count)
		})
	}
}

func benchmarkMachineScaling(b *testing.B, machineCount int) {
	// Create a medium-sized statechart
	statechart := semantics.NewStatechart(testutil.CreateLargeStatechart(100, 200))
	
	pool := profiling.NewObjectPool()
	
	b.ResetTimer()
	b.ReportAllocs()
	
	var peakMemory uint64
	
	for i := 0; i < b.N; i++ {
		machines := make([]*semantics.MachineWrapper, 0, machineCount)
		
		// Create machines
		for j := 0; j < machineCount; j++ {
			machine, err := profiling.NewOptimizedMachine(statechart, fmt.Sprintf("machine_%d", j), pool)
			if err != nil {
				b.Fatalf("Failed to create machine: %v", err)
			}
			machines = append(machines, machine.MachineWrapper)
		}
		
		// Start all machines
		for _, machine := range machines {
			machine.Start()
		}
		
		// Simulate some steps
		events := []string{"event_0", "event_1", "event_2"}
		for _, event := range events {
			for _, machine := range machines {
				machine.Step(event)
			}
		}
		
		// Check memory
		var m runtime.MemStats
		runtime.ReadMemStats(&m)
		if m.Alloc > peakMemory {
			peakMemory = m.Alloc
		}
		
		// Stop all machines
		for _, machine := range machines {
			machine.Stop()
		}
		
		// Clear for next iteration
		machines = nil
	}
	
	b.StopTimer()
	
	b.ReportMetric(float64(peakMemory)/(1024*1024), "peak-MB")
	b.ReportMetric(float64(peakMemory)/float64(machineCount), "bytes/machine")
}

// BenchmarkMemoryPoolEffectiveness benchmarks the effectiveness of object pooling.
func BenchmarkMemoryPoolEffectiveness(b *testing.B) {
	scenarios := []struct {
		name     string
		pooled   bool
		states   int
		events   int
	}{
		{"NoPool_Small", false, 50, 100},
		{"Pool_Small", true, 50, 100},
		{"NoPool_Medium", false, 200, 500},
		{"Pool_Medium", true, 200, 500},
		{"NoPool_Large", false, 1000, 2000},
		{"Pool_Large", true, 1000, 2000},
	}
	
	for _, scenario := range scenarios {
		b.Run(scenario.name, func(b *testing.B) {
			benchmarkPoolEffectiveness(b, scenario.pooled, scenario.states, scenario.events)
		})
	}
}

func benchmarkPoolEffectiveness(b *testing.B, usePool bool, states, events int) {
	statechart := semantics.NewStatechart(testutil.CreateLargeStatechart(states, states*2))
	
	var pool *profiling.ObjectPool
	if usePool {
		pool = profiling.NewObjectPool()
	}
	
	b.ResetTimer()
	b.ReportAllocs()
	
	for i := 0; i < b.N; i++ {
		// Create machine
		var machine *semantics.MachineWrapper
		var err error
		
		if usePool {
			optimized, err := profiling.NewOptimizedMachine(statechart, fmt.Sprintf("machine_%d", i), pool)
			if err != nil {
				b.Fatalf("Failed to create optimized machine: %v", err)
			}
			machine = optimized.MachineWrapper
		} else {
			machine, err = semantics.NewMachine(statechart, fmt.Sprintf("machine_%d", i), nil)
			if err != nil {
				b.Fatalf("Failed to create machine: %v", err)
			}
		}
		
		// Run machine through events
		machine.Start()
		for j := 0; j < events; j++ {
			machine.Step(fmt.Sprintf("event_%d", j%10))
		}
		machine.Stop()
	}
	
	b.StopTimer()
}

// BenchmarkStatechartIndexing benchmarks memory-efficient indexing for large statecharts.
func BenchmarkStatechartIndexing(b *testing.B) {
	sizes := []struct {
		name   string
		states int
	}{
		{"Small_100", 100},
		{"Medium_1000", 1000},
		{"Large_5000", 5000},
		{"XLarge_10000", 10000},
	}
	
	for _, size := range sizes {
		b.Run(size.name, func(b *testing.B) {
			statechart := testutil.CreateLargeStatechart(size.states, size.states*2)
			
			b.ResetTimer()
			b.ReportAllocs()
			
			var totalIndexMemory int64
			
			for i := 0; i < b.N; i++ {
				index := profiling.BuildMemoryEfficientIndex(statechart)
				totalIndexMemory += index.GetMemoryUsage()
				
				// Perform some lookups to ensure index is used
				for j := 0; j < 10; j++ {
					label := fmt.Sprintf("State%d", j%size.states)
					_, _, _, _ = index.GetStateInfo(label)
					_ = index.GetTransitionsFrom(label)
				}
			}
			
			b.StopTimer()
			
			avgIndexMemory := totalIndexMemory / int64(b.N)
			b.ReportMetric(float64(avgIndexMemory), "index-bytes/op")
			b.ReportMetric(float64(avgIndexMemory)/float64(size.states), "index-bytes/state")
		})
	}
}

// BenchmarkConcurrentMachineMemory benchmarks memory usage with concurrent machines.
func BenchmarkConcurrentMachineMemory(b *testing.B) {
	concurrencyLevels := []int{1, 10, 50, 100, 500}
	
	for _, level := range concurrencyLevels {
		b.Run(fmt.Sprintf("Concurrent_%d", level), func(b *testing.B) {
			benchmarkConcurrentMemory(b, level)
		})
	}
}

func benchmarkConcurrentMemory(b *testing.B, concurrency int) {
	statechart := semantics.NewStatechart(testutil.CreateLargeStatechart(100, 200))
	factory := profiling.NewMemoryPooledMachineFactory(1024 * 1024 * 1024) // 1GB budget
	
	b.ResetTimer()
	b.ReportAllocs()
	
	for i := 0; i < b.N; i++ {
		// Run concurrent machines
		done := make(chan bool, concurrency)
		
		for j := 0; j < concurrency; j++ {
			go func(id int) {
				defer func() { done <- true }()
				
				machine, err := factory.CreateOptimizedMachine(statechart.Statechart, fmt.Sprintf("concurrent_%d", id), 100)
				if err != nil {
					b.Errorf("Failed to create machine: %v", err)
					return
				}
				
				machine.Start()
				for k := 0; k < 10; k++ {
					machine.Step(fmt.Sprintf("event_%d", k))
				}
				machine.Stop()
			}(j)
		}
		
		// Wait for completion
		for j := 0; j < concurrency; j++ {
			<-done
		}
	}
	
	b.StopTimer()
	
	stats := factory.GetMemoryStats()
	b.ReportMetric(float64(stats.CurrentUsage)/(1024*1024), "memory-MB")
	b.ReportMetric(float64(stats.Optimizations.StringsInterned), "strings-interned")
}

// BenchmarkMemoryLeakDetection benchmarks the overhead of leak detection.
func BenchmarkMemoryLeakDetection(b *testing.B) {
	scenarios := []struct {
		name         string
		enableDetect bool
		sampleRate   time.Duration
	}{
		{"NoDetection", false, 0},
		{"Detection_Fast", true, 100 * time.Millisecond},
		{"Detection_Normal", true, time.Second},
		{"Detection_Slow", true, 5 * time.Second},
	}
	
	for _, scenario := range scenarios {
		b.Run(scenario.name, func(b *testing.B) {
			benchmarkLeakDetection(b, scenario.enableDetect, scenario.sampleRate)
		})
	}
}

func benchmarkLeakDetection(b *testing.B, enableDetection bool, sampleRate time.Duration) {
	statechart := semantics.NewStatechart(testutil.CreateLargeStatechart(500, 1000))
	
	var detector *profiling.LeakDetector
	if enableDetection {
		detector = profiling.NewLeakDetector()
		detector.SetThresholds(profiling.LeakThresholds{
			MemoryGrowthRate:  10 * 1024 * 1024, // 10MB/sec
			ObjectGrowthRate:  100000,            // 100k objects/sec
			MinSampleDuration: sampleRate,
			MinGrowthAmount:   50 * 1024 * 1024, // 50MB
		})
		detector.Start()
		defer detector.Stop()
	}
	
	b.ResetTimer()
	b.ReportAllocs()
	
	for i := 0; i < b.N; i++ {
		// Create and run machine
		machine, _ := semantics.NewMachine(statechart, fmt.Sprintf("leak_test_%d", i), nil)
		machine.Start()
		
		// Simulate work
		for j := 0; j < 100; j++ {
			machine.Step(fmt.Sprintf("event_%d", j))
			
			if enableDetection && j%10 == 0 {
				detector.TakeSample(fmt.Sprintf("step_%d", j))
			}
		}
		
		machine.Stop()
	}
	
	b.StopTimer()
}

// BenchmarkStatechartCache benchmarks the memory-efficient cache.
func BenchmarkStatechartCache(b *testing.B) {
	cacheSizes := []struct {
		name      string
		maxItems  int
		maxMemory uint64
	}{
		{"Small_10_10MB", 10, 10 * 1024 * 1024},
		{"Medium_100_100MB", 100, 100 * 1024 * 1024},
		{"Large_1000_1GB", 1000, 1024 * 1024 * 1024},
	}
	
	for _, size := range cacheSizes {
		b.Run(size.name, func(b *testing.B) {
			benchmarkCache(b, size.maxItems, size.maxMemory)
		})
	}
}

func benchmarkCache(b *testing.B, maxItems int, maxMemory uint64) {
	cache := profiling.NewMemoryEfficientStatechartCache(maxItems, maxMemory)
	
	// Pre-populate cache with some statecharts
	for i := 0; i < maxItems/2; i++ {
		statechart := testutil.CreateLargeStatechart(50+i, 100+i*2)
		cache.Put(fmt.Sprintf("chart_%d", i), statechart)
	}
	
	b.ResetTimer()
	b.ReportAllocs()
	
	hits := 0
	misses := 0
	
	for i := 0; i < b.N; i++ {
		// Mix of hits and misses
		key := fmt.Sprintf("chart_%d", i%(maxItems))
		
		if sc, found := cache.Get(key); found {
			hits++
			_ = sc
		} else {
			misses++
			// Add new item
			statechart := testutil.CreateLargeStatechart(50, 100)
			cache.Put(key, statechart)
		}
	}
	
	b.StopTimer()
	
	hitRate := float64(hits) / float64(hits+misses) * 100
	stats := cache.GetStats()
	
	b.ReportMetric(hitRate, "hit-rate-%")
	b.ReportMetric(float64(stats.MemoryUsage)/(1024*1024), "cache-MB")
	b.ReportMetric(stats.UtilizationPct, "utilization-%")
}

// Helper function to create complex statecharts with various features
func createComplexStatechart(states, transitions, depth, parallelRegions int) *sc.Statechart {
	builder := testutil.NewStatechartBuilder()
	
	// Create root with parallel regions if specified
	rootBuilder := testutil.NewStateBuilder("__root__")
	if parallelRegions > 0 {
		rootBuilder.WithType(sc.StateTypeParallel)
		
		// Create parallel regions
		regions := make([]*sc.State, parallelRegions)
		statesPerRegion := states / parallelRegions
		
		for i := 0; i < parallelRegions; i++ {
			regionBuilder := testutil.NewStateBuilder(fmt.Sprintf("Region_%d", i)).
				WithType(sc.StateTypeNormal)
			
			// Add states to region
			regionStates := make([]*sc.State, 0, statesPerRegion)
			for j := 0; j < statesPerRegion; j++ {
				stateBuilder := testutil.NewStateBuilder(fmt.Sprintf("Region_%d_State_%d", i, j))
				if j == 0 {
					stateBuilder.AsInitial()
				}
				regionStates = append(regionStates, stateBuilder.Build())
			}
			
			regionBuilder.WithChildren(regionStates...)
			regions[i] = regionBuilder.Build()
		}
		
		rootBuilder.WithChildren(regions...)
	} else {
		// Create hierarchical structure
		rootBuilder.WithType(sc.StateTypeNormal)
		
		// Create states
		stateList := make([]*sc.State, states)
		for i := 0; i < states; i++ {
			stateBuilder := testutil.NewStateBuilder(fmt.Sprintf("State_%d", i))
			if i == 0 {
				stateBuilder.AsInitial()
			}
			stateList[i] = stateBuilder.Build()
		}
		
		rootBuilder.WithChildren(stateList...)
	}
	
	builder.WithRootState(rootBuilder.Build())
	
	// Add transitions
	for i := 0; i < transitions; i++ {
		fromIdx := i % states
		toIdx := (i + 1) % states
		
		transition := testutil.NewTransitionBuilder(fmt.Sprintf("Trans_%d", i)).
			From(fmt.Sprintf("State_%d", fromIdx)).
			To(fmt.Sprintf("State_%d", toIdx)).
			OnEvent(fmt.Sprintf("event_%d", i%100)).
			Build()
		
		builder.WithTransition(transition)
		builder.WithEvent(&sc.Event{Label: fmt.Sprintf("event_%d", i%100)})
	}
	
	return builder.Build()
}

// BenchmarkRealWorldScenarios benchmarks realistic statechart usage patterns.
func BenchmarkRealWorldScenarios(b *testing.B) {
	scenarios := []struct {
		name        string
		createFunc  func() *sc.Statechart
		eventCount  int
		machineCount int
	}{
		{
			name:         "IoTDevice",
			createFunc:   createIoTDeviceStatechart,
			eventCount:   1000,
			machineCount: 100,
		},
		{
			name:         "WorkflowEngine",
			createFunc:   createWorkflowStatechart,
			eventCount:   500,
			machineCount: 50,
		},
		{
			name:         "GameStateMachine",
			createFunc:   createGameStatechart,
			eventCount:   2000,
			machineCount: 200,
		},
	}
	
	for _, scenario := range scenarios {
		b.Run(scenario.name, func(b *testing.B) {
			benchmarkRealWorldScenario(b, scenario.createFunc, scenario.eventCount, scenario.machineCount)
		})
	}
}

func benchmarkRealWorldScenario(b *testing.B, createFunc func() *sc.Statechart, eventCount, machineCount int) {
	// Create optimized statechart
	optimizer := profiling.NewLargeStatechartMemoryOptimizer()
	originalChart := createFunc()
	optimizedChart, report := optimizer.OptimizeLargeStatechart(originalChart)
	
	b.Logf("Optimization report: %v", report)
	
	// Create factory with memory budget
	factory := profiling.NewMemoryPooledMachineFactory(512 * 1024 * 1024) // 512MB budget
	
	b.ResetTimer()
	b.ReportAllocs()
	
	for i := 0; i < b.N; i++ {
		// Simulate multiple machine instances
		machines := make([]*semantics.MachineWrapper, 0, machineCount)
		
		for j := 0; j < machineCount; j++ {
			machine, err := factory.CreateOptimizedMachine(optimizedChart, fmt.Sprintf("instance_%d", j), 1000)
			if err != nil {
				b.Fatalf("Failed to create machine: %v", err)
			}
			machines = append(machines, machine)
			machine.Start()
		}
		
		// Simulate event processing
		for k := 0; k < eventCount; k++ {
			eventName := generateRealisticEvent(k)
			machineIdx := k % machineCount
			machines[machineIdx].Step(eventName)
		}
		
		// Cleanup
		for _, machine := range machines {
			machine.Stop()
		}
	}
	
	b.StopTimer()
	
	stats := factory.GetMemoryStats()
	b.ReportMetric(float64(stats.CurrentUsage)/(1024*1024), "memory-MB")
	b.ReportMetric(float64(stats.CachedCharts), "cached-charts")
}

// Helper functions to create realistic statecharts

func createIoTDeviceStatechart() *sc.Statechart {
	return testutil.NewStatechartBuilder().
		WithRootState(
			testutil.NewStateBuilder("Device").
				WithType(sc.StateTypeParallel).
				WithChildren(
					// Connection region
					testutil.NewStateBuilder("Connection").
						WithType(sc.StateTypeNormal).
						WithChildren(
							testutil.NewStateBuilder("Disconnected").AsInitial().Build(),
							testutil.NewStateBuilder("Connecting").Build(),
							testutil.NewStateBuilder("Connected").Build(),
							testutil.NewStateBuilder("Error").Build(),
						).Build(),
					// Operation region
					testutil.NewStateBuilder("Operation").
						WithType(sc.StateTypeNormal).
						WithChildren(
							testutil.NewStateBuilder("Idle").AsInitial().Build(),
							testutil.NewStateBuilder("Processing").Build(),
							testutil.NewStateBuilder("Sending").Build(),
							testutil.NewStateBuilder("Receiving").Build(),
						).Build(),
					// Power region
					testutil.NewStateBuilder("Power").
						WithType(sc.StateTypeNormal).
						WithChildren(
							testutil.NewStateBuilder("Normal").AsInitial().Build(),
							testutil.NewStateBuilder("LowPower").Build(),
							testutil.NewStateBuilder("Sleep").Build(),
						).Build(),
				).Build(),
		).
		Build()
}

func createWorkflowStatechart() *sc.Statechart {
	return testutil.NewStatechartBuilder().
		WithRootState(
			testutil.NewStateBuilder("Workflow").
				WithType(sc.StateTypeNormal).
				WithChildren(
					testutil.NewStateBuilder("Draft").AsInitial().Build(),
					testutil.NewStateBuilder("Review").
						WithType(sc.StateTypeNormal).
						WithChildren(
							testutil.NewStateBuilder("PendingReview").AsInitial().Build(),
							testutil.NewStateBuilder("InReview").Build(),
							testutil.NewStateBuilder("ChangesRequested").Build(),
						).Build(),
					testutil.NewStateBuilder("Approved").Build(),
					testutil.NewStateBuilder("Published").Build(),
					testutil.NewStateBuilder("Archived").Build(),
				).Build(),
		).
		Build()
}

func createGameStatechart() *sc.Statechart {
	return testutil.NewStatechartBuilder().
		WithRootState(
			testutil.NewStateBuilder("Game").
				WithType(sc.StateTypeNormal).
				WithChildren(
					testutil.NewStateBuilder("Menu").AsInitial().Build(),
					testutil.NewStateBuilder("Playing").
						WithType(sc.StateTypeParallel).
						WithChildren(
							testutil.NewStateBuilder("Character").
								WithType(sc.StateTypeNormal).
								WithChildren(
									testutil.NewStateBuilder("Standing").AsInitial().Build(),
									testutil.NewStateBuilder("Walking").Build(),
									testutil.NewStateBuilder("Running").Build(),
									testutil.NewStateBuilder("Jumping").Build(),
									testutil.NewStateBuilder("Attacking").Build(),
								).Build(),
							testutil.NewStateBuilder("Health").
								WithType(sc.StateTypeNormal).
								WithChildren(
									testutil.NewStateBuilder("Healthy").AsInitial().Build(),
									testutil.NewStateBuilder("Damaged").Build(),
									testutil.NewStateBuilder("Critical").Build(),
									testutil.NewStateBuilder("Dead").Build(),
								).Build(),
							testutil.NewStateBuilder("Inventory").
								WithType(sc.StateTypeNormal).
								WithChildren(
									testutil.NewStateBuilder("Empty").AsInitial().Build(),
									testutil.NewStateBuilder("HasItems").Build(),
									testutil.NewStateBuilder("Full").Build(),
								).Build(),
						).Build(),
					testutil.NewStateBuilder("Paused").Build(),
					testutil.NewStateBuilder("GameOver").Build(),
				).Build(),
		).
		Build()
}

func generateRealisticEvent(index int) string {
	events := []string{
		"connect", "disconnect", "data_received", "error",
		"start_processing", "complete", "timeout", "retry",
		"low_battery", "charge", "sleep", "wake",
		"user_input", "system_update", "sync", "backup",
	}
	return events[index%len(events)]
}