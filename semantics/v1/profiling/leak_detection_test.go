package profiling_test

import (
	"fmt"
	"runtime"
	"sync"
	"testing"
	"time"

	"github.com/tmc/sc/semantics/v1"
	"github.com/tmc/sc/semantics/v1/profiling"
	testutil "github.com/tmc/sc/testing"
)

// TestLeakDetectorBasic tests basic leak detection functionality.
func TestLeakDetectorBasic(t *testing.T) {
	detector := profiling.NewLeakDetector()

	// Set aggressive thresholds for testing
	detector.SetThresholds(profiling.LeakThresholds{
		MemoryGrowthRate:    500 * 1024, // 500KB/sec
		ObjectGrowthRate:    1000,       // 1k objects/sec
		GoroutineGrowthRate: 5,          // 5 goroutines/sec
		MinSampleDuration:   1 * time.Second,
		MinGrowthAmount:     512 * 1024, // 512KB
	})

	leakDetected := make(chan profiling.DetectedLeak, 1)
	detector.AddCallback(func(leak profiling.DetectedLeak) {
		select {
		case leakDetected <- leak:
		default:
		}
	})

	detector.Start()
	defer detector.Stop()

	// Simulate memory growth
	var allocations [][]byte

	// Take initial sample
	detector.TakeSample("initial")
	time.Sleep(100 * time.Millisecond)

	for i := 0; i < 10; i++ {
		// Allocate 512KB
		data := make([]byte, 512*1024)
		allocations = append(allocations, data)

		detector.TakeSample(fmt.Sprintf("alloc_%d", i))
		time.Sleep(200 * time.Millisecond)
	}

	// Wait for detection
	select {
	case leak := <-leakDetected:
		t.Logf("Leak detected: %s (severity: %s)", leak.Description, leak.Severity)
		if leak.Type != profiling.MemoryLeak {
			t.Errorf("Expected MemoryLeak type, got %v", leak.Type)
		}
	case <-time.After(5 * time.Second):
		t.Error("Expected leak detection but none occurred")
	}

	// Keep allocations alive to prevent GC
	_ = allocations
}

// TestStatechartLeakDetection tests leak detection in statechart operations.
func TestStatechartLeakDetection(t *testing.T) {
	detector := profiling.NewStatechartLeakDetector()

	// Set thresholds
	detector.SetThresholds(profiling.StatechartLeakThresholds{
		MaxHistorySize:      1000,
		MaxContextSize:      10 * 1024, // 10KB
		MaxErrorCount:       100,
		GrowthRateThreshold: 100, // items/sec
	})

	detector.Start()
	defer detector.Stop()

	// Create a statechart
	statechart := semantics.NewStatechart(testutil.CreateLargeStatechart(100, 200))

	// Create machines that will leak memory
	leakyMachines := make([]*LeakyMachine, 10)
	for i := 0; i < 10; i++ {
		machine, _ := semantics.NewMachine(statechart, fmt.Sprintf("leaky_%d", i), nil)
		leaky := &LeakyMachine{
			MachineWrapper: machine,
			leakData:       make([][]byte, 0),
		}
		leakyMachines[i] = leaky
		detector.RegisterMachine(fmt.Sprintf("leaky_%d", i), machine)
	}

	// Simulate operations that leak
	for i := 0; i < 100; i++ {
		for _, machine := range leakyMachines {
			machine.LeakyStep(fmt.Sprintf("event_%d", i))
		}

		if i%10 == 0 {
			// Check for leaks periodically
			reports := detector.CheckMachineLeaks()
			for _, report := range reports {
				if report.HasIssues() {
					t.Logf("Machine %s has issues: %v", report.MachineID, report.Issues)
				}
			}
		}

		time.Sleep(10 * time.Millisecond)
	}

	// Final leak check
	finalReports := detector.CheckMachineLeaks()
	leaksFound := false
	for _, report := range finalReports {
		if report.HasIssues() {
			leaksFound = true
			t.Logf("Final check - Machine %s has issues: %v", report.MachineID, report.Issues)
		}
	}

	if !leaksFound {
		t.Error("Expected to detect leaks in machines but none were found")
	}
}

// LeakyMachine simulates a machine that leaks memory.
type LeakyMachine struct {
	*semantics.MachineWrapper
	leakData [][]byte
	mu       sync.Mutex
}

func (lm *LeakyMachine) LeakyStep(event string) {
	lm.mu.Lock()
	defer lm.mu.Unlock()

	// Regular step
	lm.MachineWrapper.Step(event)

	// Leak memory by accumulating data
	lm.leakData = append(lm.leakData, make([]byte, 1024)) // 1KB per step
}

// TestGoroutineLeakDetection tests detection of goroutine leaks.
func TestGoroutineLeakDetection(t *testing.T) {
	detector := profiling.NewLeakDetector()

	detector.SetThresholds(profiling.LeakThresholds{
		GoroutineGrowthRate: 2, // 2 goroutines/sec
		MinSampleDuration:   2 * time.Second,
		MinGrowthAmount:     5, // at least 5 goroutines
	})

	leakDetected := make(chan profiling.DetectedLeak, 1)
	detector.AddCallback(func(leak profiling.DetectedLeak) {
		if leak.Type == profiling.GoroutineLeak {
			select {
			case leakDetected <- leak:
			default:
			}
		}
	})

	detector.Start()
	defer detector.Stop()

	// Get initial goroutine count
	initialCount := runtime.NumGoroutine()

	// Start leaking goroutines
	stopChan := make(chan struct{})
	defer close(stopChan)

	for i := 0; i < 20; i++ {
		go func() {
			<-stopChan // This will block until test ends
		}()

		detector.TakeSample(fmt.Sprintf("goroutine_%d", i))
		time.Sleep(200 * time.Millisecond)
	}

	// Wait for detection
	select {
	case leak := <-leakDetected:
		t.Logf("Goroutine leak detected: %s", leak.Description)
		if leak.Type != profiling.GoroutineLeak {
			t.Errorf("Expected GoroutineLeak type, got %v", leak.Type)
		}
	case <-time.After(5 * time.Second):
		currentCount := runtime.NumGoroutine()
		t.Errorf("Expected goroutine leak detection but none occurred (initial: %d, current: %d)",
			initialCount, currentCount)
	}
}

// TestHistoryLeakDetection tests detection of unbounded history growth.
func TestHistoryLeakDetection(t *testing.T) {
	detector := profiling.NewStatechartLeakDetector()
	detector.Start()
	defer detector.Stop()

	// Create a machine with no history limit
	statechart := semantics.NewStatechart(testutil.CreateSimpleStatechart())
	machine, _ := semantics.NewMachine(statechart, "history_leak", nil)

	detector.RegisterMachine("history_leak", machine)

	// Generate many steps
	machine.Start()
	for i := 0; i < 5000; i++ {
		machine.Step("go")
	}

	// Check for history leak
	reports := detector.CheckMachineLeaks()
	historyLeakFound := false

	for _, report := range reports {
		if report.MachineID == "history_leak" && report.HasIssues() {
			for _, issue := range report.Issues {
				if issue.Type == profiling.HistoryLeak {
					historyLeakFound = true
					t.Logf("History leak detected: %s", issue.Description)
				}
			}
		}
	}

	if !historyLeakFound {
		t.Error("Expected to detect history leak but none was found")
	}
}

// TestMemoryBudgetEnforcement tests memory budget management.
func TestMemoryBudgetEnforcement(t *testing.T) {
	maxMemory := uint64(10 * 1024 * 1024) // 10MB budget
	manager := profiling.NewMemoryBudgetManager(maxMemory)

	warningReceived := false
	manager.AddCallback(func(current, max uint64, exceeded bool) {
		if exceeded {
			warningReceived = true
			t.Logf("Memory warning: %d/%d bytes (%.2f%%)",
				current, max, float64(current)/float64(max)*100)
		}
	})

	// Allocate within budget
	err := manager.AllocateMemory(5 * 1024 * 1024) // 5MB
	if err != nil {
		t.Errorf("Failed to allocate within budget: %v", err)
	}

	// Allocate more to trigger warning (80% threshold)
	err = manager.AllocateMemory(4 * 1024 * 1024) // 4MB more (total 9MB = 90%)
	if err != nil {
		t.Errorf("Failed to allocate within budget: %v", err)
	}

	if !warningReceived {
		t.Error("Expected memory warning but none was received")
	}

	// Try to exceed budget
	err = manager.AllocateMemory(2 * 1024 * 1024) // 2MB more would exceed
	if err == nil {
		t.Error("Expected allocation to fail due to budget exceeded")
	}

	// Release memory
	manager.ReleaseMemory(5 * 1024 * 1024)

	// Now allocation should succeed
	err = manager.AllocateMemory(2 * 1024 * 1024)
	if err != nil {
		t.Errorf("Failed to allocate after releasing memory: %v", err)
	}
}

// TestLongRunningMachineLeaks tests leak detection in long-running scenarios.
func TestLongRunningMachineLeaks(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping long-running leak test in short mode")
	}

	detector := profiling.NewLeakDetector()
	detector.SetThresholds(profiling.LeakThresholds{
		MemoryGrowthRate:  500 * 1024, // 500KB/sec
		MinSampleDuration: 5 * time.Second,
		MinGrowthAmount:   2 * 1024 * 1024, // 2MB
	})

	detector.Start()
	defer detector.Stop()

	// Create a complex statechart
	statechart := semantics.NewStatechart(createComplexStatechart(500, 1000, 5, 4))

	// Run multiple machines concurrently
	var wg sync.WaitGroup
	machineCount := 10

	for i := 0; i < machineCount; i++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()

			machine, _ := semantics.NewMachine(statechart, fmt.Sprintf("long_running_%d", id), nil)
			machine.Start()

			// Simulate long-running operation
			for j := 0; j < 1000; j++ {
				event := fmt.Sprintf("event_%d", j%50)
				machine.Step(event)

				if j%100 == 0 {
					detector.TakeSample(fmt.Sprintf("machine_%d_step_%d", id, j))
				}

				time.Sleep(5 * time.Millisecond)
			}

			machine.Stop()
		}(i)
	}

	// Monitor memory while machines run
	monitorDone := make(chan struct{})
	go func() {
		defer close(monitorDone)

		ticker := time.NewTicker(time.Second)
		defer ticker.Stop()
		timeout := time.After(10 * time.Second)

		for {
			select {
			case <-ticker.C:
				var m runtime.MemStats
				runtime.ReadMemStats(&m)
				t.Logf("Memory: Alloc=%dMB, TotalAlloc=%dMB, Sys=%dMB, NumGC=%d",
					m.Alloc/(1024*1024), m.TotalAlloc/(1024*1024),
					m.Sys/(1024*1024), m.NumGC)
			case <-timeout:
				return
			}
		}
	}()

	wg.Wait()
	<-monitorDone
}

// TestOptimizationEffectiveness tests that optimization reduces memory usage.
func TestOptimizationEffectiveness(t *testing.T) {
	sizes := []struct {
		states      int
		transitions int
	}{
		{100, 200},
		{500, 1000},
		{1000, 2000},
	}

	optimizer := profiling.NewLargeStatechartMemoryOptimizer()

	for _, size := range sizes {
		t.Run(fmt.Sprintf("States_%d", size.states), func(t *testing.T) {
			// Create statechart
			original := testutil.CreateLargeStatechart(size.states, size.transitions)

			// Optimize
			optimized, report := optimizer.OptimizeLargeStatechart(original)

			// Verify optimization
			if report.OptimizedSize >= report.OriginalSize {
				t.Errorf("Optimization failed to reduce size: original=%d, optimized=%d",
					report.OriginalSize, report.OptimizedSize)
			}

			t.Logf("Optimization results: %v", report)

			// Verify functionality preserved
			originalSem := semantics.NewStatechart(original)
			optimizedSem := semantics.NewStatechart(optimized)

			if err := originalSem.Validate(); err != nil {
				t.Errorf("Original validation failed: %v", err)
			}

			if err := optimizedSem.Validate(); err != nil {
				t.Errorf("Optimized validation failed: %v", err)
			}
		})
	}
}

// TestConcurrentLeakDetection tests leak detection with concurrent operations.
func TestConcurrentLeakDetection(t *testing.T) {
	detector := profiling.NewLeakDetector()
	detector.Start()
	defer detector.Stop()

	// Track detected leaks
	var detectedLeaks []profiling.DetectedLeak
	var leakMu sync.Mutex

	detector.AddCallback(func(leak profiling.DetectedLeak) {
		leakMu.Lock()
		detectedLeaks = append(detectedLeaks, leak)
		leakMu.Unlock()
	})

	// Run concurrent operations
	var wg sync.WaitGroup
	workerCount := 5

	for i := 0; i < workerCount; i++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()

			// Each worker creates its own leaky behavior
			var data [][]byte
			for j := 0; j < 50; j++ {
				// Allocate memory
				chunk := make([]byte, 100*1024) // 100KB
				data = append(data, chunk)

				// Take sample
				detector.TakeSample(fmt.Sprintf("worker_%d_alloc_%d", id, j))

				time.Sleep(50 * time.Millisecond)
			}

			// Keep data alive
			_ = data
		}(i)
	}

	wg.Wait()
	time.Sleep(time.Second) // Give detector time to process

	// Check results
	leakMu.Lock()
	leakCount := len(detectedLeaks)
	leakMu.Unlock()

	if leakCount == 0 {
		t.Error("Expected to detect leaks from concurrent operations but none were found")
	} else {
		t.Logf("Detected %d leaks from concurrent operations", leakCount)
	}
}

// TestMemoryPoolRecycling tests that object pools effectively recycle memory.
func TestMemoryPoolRecycling(t *testing.T) {
	pool := profiling.NewObjectPool()

	// Track allocations
	runtime.GC()
	var m0 runtime.MemStats
	runtime.ReadMemStats(&m0)

	// Use pool many times
	iterations := 10000
	for i := 0; i < iterations; i++ {
		// Get objects from pool
		config := pool.GetConfiguration()
		ctx := pool.GetContext()
		trans := pool.GetTransition()
		state := pool.GetStateRef()
		step := pool.GetStep()
		event := pool.GetEvent()
		action := pool.GetAction()

		// Use objects (simulate work)
		config.States = append(config.States, state)
		trans.From = append(trans.From, "state1")
		trans.To = append(trans.To, "state2")
		step.Events = append(step.Events, event)
		trans.Actions = append(trans.Actions, action)

		// Return to pool
		pool.PutConfiguration(config)
		pool.PutContext(ctx)
		pool.PutTransition(trans)
		pool.PutStateRef(state)
		pool.PutStep(step)
		pool.PutEvent(event)
		pool.PutAction(action)
	}

	// Check memory usage
	runtime.GC()
	var m1 runtime.MemStats
	runtime.ReadMemStats(&m1)

	allocatedBytes := m1.TotalAlloc - m0.TotalAlloc
	allocatedObjects := m1.Mallocs - m0.Mallocs

	// Calculate per-iteration cost
	bytesPerIteration := allocatedBytes / uint64(iterations)
	objectsPerIteration := allocatedObjects / uint64(iterations)

	t.Logf("Pool efficiency: %d bytes/iteration, %d objects/iteration",
		bytesPerIteration, objectsPerIteration)

	// Verify pool is efficient (these are rough thresholds)
	if bytesPerIteration > 1000 { // More than 1KB per iteration suggests poor pooling
		t.Errorf("Pool appears inefficient: %d bytes per iteration", bytesPerIteration)
	}

	if objectsPerIteration > 10 { // More than 10 allocations per iteration
		t.Errorf("Pool has too many allocations: %d objects per iteration", objectsPerIteration)
	}
}

// TestStatechartIndexMemoryEfficiency tests memory efficiency of the state index.
func TestStatechartIndexMemoryEfficiency(t *testing.T) {
	sizes := []int{100, 500, 1000, 5000}

	for _, size := range sizes {
		t.Run(fmt.Sprintf("States_%d", size), func(t *testing.T) {
			// Create large statechart
			statechart := testutil.CreateLargeStatechart(size, size*2)

			// Build index
			index := profiling.BuildMemoryEfficientIndex(statechart)

			// Check memory usage
			indexMemory := index.GetMemoryUsage()
			memoryPerState := float64(indexMemory) / float64(size)

			t.Logf("Index memory: %d bytes total, %.2f bytes/state", indexMemory, memoryPerState)

			// Verify index works correctly
			for i := 0; i < min(10, size); i++ {
				label := fmt.Sprintf("State%d", i)
				parent, depth, childCount, found := index.GetStateInfo(label)

				if !found {
					t.Errorf("State %s not found in index", label)
				} else {
					t.Logf("State %s: parent=%s, depth=%d, children=%d",
						label, parent, depth, childCount)
				}

				// Check transitions
				transitions := index.GetTransitionsFrom(label)
				if len(transitions) > 0 {
					t.Logf("State %s has %d transitions", label, len(transitions))
				}
			}

			// Memory efficiency check - rough estimate
			expectedBytesPerState := 100.0 // Rough estimate
			if memoryPerState > expectedBytesPerState {
				t.Logf("Warning: Index uses %.2f bytes/state, expected < %.2f",
					memoryPerState, expectedBytesPerState)
			}
		})
	}
}

// Helper function
func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
