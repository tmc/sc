package profiling_test

import (
	"context"
	"fmt"
	"runtime"
	"sync"
	"testing"
	"time"

	"github.com/tmc/sc"
	"github.com/tmc/sc/semantics/v1"
	"github.com/tmc/sc/semantics/v1/profiling"
	testutil "github.com/tmc/sc/testing"
)

// TestMemoryProfiler tests basic memory profiling functionality
func TestMemoryProfiler(t *testing.T) {
	profiler := profiling.NewMemoryProfiler()
	
	// Test baseline setting
	profiler.SetBaseline("test_baseline")
	baseline := profiler.GetBaseline()
	if baseline == nil {
		t.Fatal("Expected baseline to be set")
	}
	
	// Test snapshot
	snapshot := profiler.TakeSnapshot("test_snapshot")
	if snapshot.Label != "test_snapshot" {
		t.Errorf("Expected snapshot label 'test_snapshot', got '%s'", snapshot.Label)
	}
	
	// Test snapshots retrieval
	snapshots := profiler.GetSnapshots()
	if len(snapshots) != 1 {
		t.Errorf("Expected 1 snapshot, got %d", len(snapshots))
	}
	
	// Test delta computation
	time.Sleep(10 * time.Millisecond)
	snapshot2 := profiler.TakeSnapshot("test_snapshot_2")
	delta := profiling.ComputeDelta(snapshot, snapshot2)
	
	if delta.Duration <= 0 {
		t.Error("Expected positive duration in delta")
	}
}

// TestStatechartProfiler tests statechart-specific profiling
func TestStatechartProfiler(t *testing.T) {
	tests := []struct {
		name          string
		createFunc    func() *sc.Statechart
		expectSuccess bool
	}{
		{
			name:          "Simple statechart",
			createFunc:    testutil.CreateSimpleStatechart,
			expectSuccess: true,
		},
		{
			name:          "Hierarchical statechart",
			createFunc:    testutil.CreateHierarchicalStatechart,
			expectSuccess: true,
		},
		{
			name:          "Orthogonal statechart",
			createFunc:    testutil.CreateOrthogonalStatechart,
			expectSuccess: true,
		},
	}
	
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			profiler := profiling.NewStatechartProfiler()
			statechart := semantics.NewStatechart(tt.createFunc())
			
			// Profile validation
			deltaFunc, err := profiler.ProfileValidation(statechart)
			if err != nil {
				t.Fatalf("Failed to start validation profiling: %v", err)
			}
			
			validationErr := statechart.Validate()
			if tt.expectSuccess && validationErr != nil {
				t.Errorf("Validation failed: %v", validationErr)
			}
			
			delta := deltaFunc()
			t.Logf("Validation memory delta: %s", delta)
			
			// Profile machine creation
			deltaFunc, machine, err := profiler.ProfileMachineCreation(statechart, "test_machine")
			if err != nil {
				t.Fatalf("Failed to create machine: %v", err)
			}
			
			delta = deltaFunc()
			t.Logf("Machine creation memory delta: %s", delta)
			
			// Profile event processing
			machine.Start()
			events := []string{"go", "go", "go"} // Generic events for testing
			for _, event := range events {
				deltaFunc, stepped, err := profiler.ProfileEventProcessing(machine, event)
				if err != nil {
					t.Errorf("Event processing failed: %v", err)
				}
				
				delta = deltaFunc()
				t.Logf("Event '%s' processing memory delta: %s (stepped: %v)", event, delta, stepped)
			}
			machine.Stop()
		})
	}
}

// TestMemoryLeakDetection tests the leak detection system
func TestMemoryLeakDetection(t *testing.T) {
	detector := profiling.NewLeakDetector()
	detector.SetThresholds(profiling.LeakThresholds{
		MemoryGrowthRate:    1024,        // 1KB/sec for testing
		ObjectGrowthRate:    100,         // 100 objects/sec
		MinSampleDuration:   time.Second, // Short duration for testing
		MinGrowthAmount:     1024,        // 1KB minimum
	})
	
	// Track detected leaks
	detectedLeaks := make([]profiling.DetectedLeak, 0)
	var mu sync.Mutex
	
	detector.AddCallback(func(leak profiling.DetectedLeak) {
		mu.Lock()
		detectedLeaks = append(detectedLeaks, leak)
		mu.Unlock()
		t.Logf("Leak detected: %s (severity: %s)", leak.Description, leak.Severity)
	})
	
	detector.Start()
	defer detector.Stop()
	
	// Simulate memory growth
	var leakySlice [][]byte
	for i := 0; i < 10; i++ {
		// Allocate 1KB per iteration
		leakySlice = append(leakySlice, make([]byte, 1024))
		detector.TakeSample(fmt.Sprintf("iteration_%d", i))
		time.Sleep(150 * time.Millisecond)
	}
	
	// Wait for detection
	time.Sleep(500 * time.Millisecond)
	
	// Generate report
	report := detector.GenerateReport()
	t.Logf("Leak detection report: %s", report.Summary)
	
	// Keep reference to prevent GC
	_ = leakySlice
}

// TestObjectPooling tests object pool efficiency
func TestObjectPooling(t *testing.T) {
	pool := profiling.NewObjectPool()
	
	// Test configuration pooling
	configs := make([]*sc.Configuration, 100)
	for i := 0; i < 100; i++ {
		configs[i] = pool.GetConfiguration()
	}
	
	// Return half to pool
	for i := 0; i < 50; i++ {
		pool.PutConfiguration(configs[i])
	}
	
	// Get again - should reuse pooled objects
	var m1, m2 runtime.MemStats
	runtime.GC()
	runtime.ReadMemStats(&m1)
	
	for i := 0; i < 50; i++ {
		config := pool.GetConfiguration()
		// Simulate usage
		config.States = append(config.States, &sc.StateRef{Label: fmt.Sprintf("state_%d", i)})
		pool.PutConfiguration(config)
	}
	
	runtime.GC()
	runtime.ReadMemStats(&m2)
	
	allocDiff := m2.TotalAlloc - m1.TotalAlloc
	t.Logf("Allocations with pooling: %d bytes", allocDiff)
}

// TestOptimizedMachine tests memory-optimized machine operations
func TestOptimizedMachine(t *testing.T) {
	pool := profiling.NewObjectPool()
	statechart := semantics.NewStatechart(testutil.CreateLargeStatechart(50, 100))
	
	machine, err := profiling.NewOptimizedMachine(statechart, "test_optimized", pool)
	if err != nil {
		t.Fatalf("Failed to create optimized machine: %v", err)
	}
	
	// Configure optimization
	machine.SetHistoryLimit(100)
	machine.SetCompactionFrequency(10)
	
	// Start machine
	err = machine.Start()
	if err != nil {
		t.Fatalf("Failed to start machine: %v", err)
	}
	
	// Process many events to trigger compaction
	for i := 0; i < 150; i++ {
		event := fmt.Sprintf("event_%d", i%5)
		stepped, err := machine.OptimizedStep(event)
		if err != nil {
			t.Errorf("Step %d failed: %v", i, err)
		}
		_ = stepped
	}
	
	// Check history is limited
	history := machine.GetStepHistory()
	if len(history) > 100 {
		t.Errorf("Expected history to be limited to 100, got %d", len(history))
	}
	
	machine.Stop()
}

// TestLargeStatechartOptimization tests optimization for large statecharts
func TestLargeStatechartOptimization(t *testing.T) {
	optimizer := profiling.NewLargeStatechartMemoryOptimizer()
	
	// Create a large statechart
	originalSC := testutil.CreateLargeStatechart(100, 200)
	
	// Optimize it
	optimized, report := optimizer.OptimizeLargeStatechart(originalSC)
	
	t.Logf("Optimization report:\n%s", report)
	
	// Verify optimization
	if report.SizeReduction <= 0 {
		t.Log("Warning: No size reduction achieved")
	}
	
	// Test that optimized statechart still works
	statechart := semantics.NewStatechart(optimized)
	err := statechart.Validate()
	if err != nil {
		t.Errorf("Optimized statechart validation failed: %v", err)
	}
}

// TestMemoryEfficientStatechartIndex tests the compact indexing system
func TestMemoryEfficientStatechartIndex(t *testing.T) {
	statechart := testutil.CreateLargeStatechart(50, 100)
	
	// Build index
	index := profiling.BuildMemoryEfficientIndex(statechart)
	
	// Test lookups
	parentLabel, depth, childCount, found := index.GetStateInfo("State10")
	if !found {
		t.Error("Expected to find State10 in index")
	}
	t.Logf("State10: parent=%s, depth=%d, children=%d", parentLabel, depth, childCount)
	
	// Test transition lookups
	transitions := index.GetTransitionsFrom("State0")
	t.Logf("Transitions from State0: %+v", transitions)
	
	// Check memory usage
	memUsage := index.GetMemoryUsage()
	t.Logf("Index memory usage: %d bytes", memUsage)
}

// TestConcurrentStatechartOptimizer tests concurrent optimization
func TestConcurrentStatechartOptimizer(t *testing.T) {
	optimizer := profiling.NewConcurrentStatechartOptimizer(4)
	
	// Create a very large statechart
	largeSC := testutil.CreateLargeStatechart(500, 1000)
	
	// Optimize concurrently
	optimized, report := optimizer.OptimizeConcurrently(largeSC)
	
	t.Logf("Concurrent optimization report:\n%s", report)
	
	// Verify optimization
	statechart := semantics.NewStatechart(optimized)
	err := statechart.Validate()
	if err != nil {
		t.Errorf("Optimized statechart validation failed: %v", err)
	}
}

// TestMemoryBudgetManager tests memory budget enforcement
func TestMemoryBudgetManager(t *testing.T) {
	maxMemory := uint64(10 * 1024 * 1024) // 10MB budget
	manager := profiling.NewMemoryBudgetManager(maxMemory)
	
	// Track budget events
	var exceeded bool
	manager.AddCallback(func(current, max uint64, isExceeded bool) {
		if isExceeded {
			exceeded = true
			t.Logf("Memory budget warning: %d/%d bytes (%.2f%%)", 
				current, max, float64(current)/float64(max)*100)
		}
	})
	
	// Allocate memory
	err := manager.AllocateMemory(5 * 1024 * 1024) // 5MB
	if err != nil {
		t.Errorf("Failed to allocate 5MB: %v", err)
	}
	
	// This should succeed
	err = manager.AllocateMemory(3 * 1024 * 1024) // 3MB
	if err != nil {
		t.Errorf("Failed to allocate 3MB: %v", err)
	}
	
	// This should trigger warning (over 80%)
	if !exceeded {
		t.Error("Expected memory budget warning to be triggered")
	}
	
	// This should fail
	err = manager.AllocateMemory(3 * 1024 * 1024) // 3MB more
	if err == nil {
		t.Error("Expected allocation to fail due to budget limit")
	}
	
	// Release some memory
	manager.ReleaseMemory(5 * 1024 * 1024)
	
	// Now it should succeed
	err = manager.AllocateMemory(3 * 1024 * 1024)
	if err != nil {
		t.Errorf("Failed to allocate after release: %v", err)
	}
}

// TestMemoryPooledMachineFactory tests the pooled machine factory
func TestMemoryPooledMachineFactory(t *testing.T) {
	maxMemory := uint64(50 * 1024 * 1024) // 50MB budget
	factory := profiling.NewMemoryPooledMachineFactory(maxMemory)
	
	statechart := testutil.CreateLargeStatechart(100, 200)
	
	// Create multiple machines
	machines := make([]*semantics.MachineWrapper, 0)
	for i := 0; i < 5; i++ {
		machine, err := factory.CreateOptimizedMachine(statechart, fmt.Sprintf("machine_%d", i), 100)
		if err != nil {
			t.Errorf("Failed to create machine %d: %v", i, err)
			break
		}
		machines = append(machines, machine)
	}
	
	// Get memory stats
	stats := factory.GetMemoryStats()
	t.Logf("Factory stats: current=%d, max=%d, cached=%d", 
		stats.CurrentUsage, stats.MaxMemory, stats.CachedCharts)
	
	// Release a machine
	if len(machines) > 0 {
		factory.ReleaseMachine(machines[0], 5*1024*1024) // Estimate 5MB
	}
}

// TestStatechartLeakDetector tests statechart-specific leak detection
func TestStatechartLeakDetector(t *testing.T) {
	detector := profiling.NewStatechartLeakDetector()
	detector.Start()
	defer detector.Stop()
	
	// Create and register machines
	statechart := semantics.NewStatechart(testutil.CreateHierarchicalStatechart())
	
	for i := 0; i < 3; i++ {
		machine, err := semantics.NewMachine(statechart, fmt.Sprintf("leak_test_%d", i), nil)
		if err != nil {
			t.Fatalf("Failed to create machine: %v", err)
		}
		
		machine.Start()
		detector.RegisterMachine(fmt.Sprintf("leak_test_%d", i), machine)
		
		// Simulate excessive history accumulation
		for j := 0; j < 15000; j++ {
			machine.Step("start")
			machine.Step("finish")
		}
	}
	
	// Check for leaks
	reports := detector.CheckMachineLeaks()
	for _, report := range reports {
		if report.HasIssues() {
			t.Logf("Machine %s has issues:", report.MachineID)
			for _, issue := range report.Issues {
				t.Logf("  - %s (severity: %s)", issue.Description, issue.Severity)
			}
		}
	}
	
	// Unregister machines
	for i := 0; i < 3; i++ {
		detector.UnregisterMachine(fmt.Sprintf("leak_test_%d", i))
	}
}

// TestMemoryEfficientStatechartCache tests the memory-efficient cache
func TestMemoryEfficientStatechartCache(t *testing.T) {
	maxSize := 10
	maxMemory := uint64(10 * 1024 * 1024) // 10MB
	cache := profiling.NewMemoryEfficientStatechartCache(maxSize, maxMemory)
	
	// Add statecharts to cache
	for i := 0; i < 15; i++ {
		sc := testutil.CreateLargeStatechart(10+i, 20+i)
		cache.Put(fmt.Sprintf("statechart_%d", i), sc)
	}
	
	// Test retrieval
	sc, found := cache.Get("statechart_5")
	if !found {
		t.Error("Expected to find statechart_5 in cache")
	}
	if sc == nil {
		t.Error("Retrieved nil statechart")
	}
	
	// Get cache stats
	stats := cache.GetStats()
	t.Logf("Cache stats: size=%d, memory=%d/%d (%.2f%% utilization)", 
		stats.Size, stats.MemoryUsage, stats.MaxMemory, stats.UtilizationPct)
	
	// Verify LRU eviction
	if stats.Size > maxSize {
		t.Errorf("Cache size %d exceeds max size %d", stats.Size, maxSize)
	}
}

// TestLifecycleProfile tests complete lifecycle profiling
func TestLifecycleProfile(t *testing.T) {
	profiler := profiling.NewStatechartProfiler()
	
	profile, err := profiler.ProfileStatechartLifecycle(
		testutil.CreateHierarchicalStatechart,
		[]string{"start", "finish", "stop"},
	)
	
	if err != nil {
		t.Fatalf("Lifecycle profiling failed: %v", err)
	}
	
	t.Logf("Lifecycle profile duration: %v", profile.TotalDuration)
	for op, delta := range profile.Operations {
		t.Logf("  %s: %s", op, delta)
	}
	
	// Generate comprehensive report
	report := profiler.GetMemoryProfiler().GenerateReport()
	t.Logf("\nMemory Report:")
	t.Logf("  Peak allocation: %d bytes (%s)", report.PeakAlloc, report.PeakAllocLabel)
	t.Logf("  Total allocated: %d bytes", report.TotalAllocated)
	t.Logf("  GC cycles: %d", report.GCCount)
	t.Logf("  Recommendations:")
	for _, rec := range report.Recommendations {
		t.Logf("    - %s", rec)
	}
}

// TestContinuousProfiler tests continuous profiling
func TestContinuousProfiler(t *testing.T) {
	profiler := profiling.NewContinuousProfiler(100 * time.Millisecond)
	
	profiler.Start()
	defer profiler.Stop()
	
	// Simulate some activity
	ctx, cancel := context.WithTimeout(context.Background(), 500*time.Millisecond)
	defer cancel()
	
	// Create some memory churn
	go func() {
		for {
			select {
			case <-ctx.Done():
				return
			default:
				data := make([]byte, 1024*1024) // 1MB allocations
				_ = data
				time.Sleep(50 * time.Millisecond)
			}
		}
	}()
	
	<-ctx.Done()
	
	// Check snapshots were taken
	snapshots := profiler.GetProfiler().GetMemoryProfiler().GetSnapshots()
	if len(snapshots) < 3 {
		t.Errorf("Expected at least 3 snapshots, got %d", len(snapshots))
	}
	
	t.Logf("Continuous profiling captured %d snapshots", len(snapshots))
}

// TestMemoryConstraints tests memory constraint validation
func TestMemoryConstraints(t *testing.T) {
	constraints := []profiling.MemoryConstraint{
		{
			Name:          "test_memory_limit",
			MaxAllocation: 1024 * 1024, // 1MB
			MaxObjects:    10000,
			MaxDuration:   time.Second,
		},
	}
	
	profiler := profiling.NewConstrainedProfiler(constraints)
	
	// Operation that should pass constraints
	delta, err := profiler.ProfileWithConstraints("small_operation", func() error {
		// Small allocation
		data := make([]byte, 512*1024) // 512KB
		_ = data
		return nil
	})
	
	if err != nil {
		t.Errorf("Small operation failed constraints: %v", err)
	}
	t.Logf("Small operation delta: %s", delta)
	
	// Operation that should fail constraints
	delta, err = profiler.ProfileWithConstraints("large_operation", func() error {
		// Large allocation
		data := make([]byte, 2*1024*1024) // 2MB
		_ = data
		return nil
	})
	
	if err == nil {
		t.Error("Expected large operation to fail constraints")
	} else {
		t.Logf("Large operation correctly failed: %v", err)
	}
}

// TestBatchProfiler tests batch profiling operations
func TestBatchProfiler(t *testing.T) {
	profiler := profiling.NewBatchProfiler()
	
	// Define operations
	operations := map[string]func() error{
		"create_statechart": func() error {
			sc := testutil.CreateLargeStatechart(50, 100)
			_ = sc
			return nil
		},
		"validate_statechart": func() error {
			sc := semantics.NewStatechart(testutil.CreateSimpleStatechart())
			return sc.Validate()
		},
		"create_machine": func() error {
			sc := semantics.NewStatechart(testutil.CreateSimpleStatechart())
			machine, err := semantics.NewMachine(sc, "batch_test", nil)
			if err != nil {
				return err
			}
			return machine.Start()
		},
	}
	
	// Profile sequentially
	seqResults := profiler.ProfileSequential(operations)
	t.Log("Sequential profiling results:")
	for _, result := range seqResults {
		t.Logf("  %s: %s (duration: %v)", result.Name, result.Delta, result.Duration)
	}
	
	// Profile in parallel
	parResults := profiler.ProfileParallel(operations)
	t.Log("Parallel profiling results:")
	for _, result := range parResults {
		t.Logf("  %s: %s (duration: %v)", result.Name, result.Delta, result.Duration)
	}
}

// TestMemoryPatternAnalysis tests memory pattern analysis
func TestMemoryPatternAnalysis(t *testing.T) {
	analyzer := profiling.NewMemoryAnalyzer()
	profiler := profiling.NewStatechartProfiler()
	
	// Collect multiple profiles
	for i := 0; i < 5; i++ {
		profile, err := profiler.ProfileStatechartLifecycle(
			func() *sc.Statechart { 
				return testutil.CreateLargeStatechart(20+i*10, 40+i*20) 
			},
			[]string{"event_0", "event_1"},
		)
		if err != nil {
			t.Errorf("Profile %d failed: %v", i, err)
			continue
		}
		analyzer.AddProfile(*profile)
	}
	
	// Analyze patterns
	analysis := analyzer.AnalyzePatterns()
	t.Logf("Pattern analysis (profiles: %d):", analysis.ProfileCount)
	for op, stats := range analysis.Operations {
		t.Logf("  %s: avg=%d bytes, max=%d bytes, count=%d", 
			op, stats.AvgAlloc, stats.MaxAlloc, stats.Count)
	}
	t.Log("Issues:")
	for _, issue := range analysis.Issues {
		t.Logf("  - %s", issue)
	}
}