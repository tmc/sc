package integration

import (
	"fmt"
	"runtime"
	"sync"
	"testing"
	"time"

	"github.com/tmc/sc"
	"github.com/tmc/sc/semantics/v1"
	testutil "github.com/tmc/sc/testing"
	"google.golang.org/protobuf/types/known/structpb"
)

// TestPerformanceIntegration tests performance characteristics of integrated workflows
func TestPerformanceIntegration(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping performance tests in short mode")
	}

	t.Run("ValidationPerformance", func(t *testing.T) {
		testValidationPerformance(t)
	})

	t.Run("MachineCreationPerformance", func(t *testing.T) {
		testMachineCreationPerformance(t)
	})

	t.Run("EventProcessingPerformance", func(t *testing.T) {
		testEventProcessingPerformance(t)
	})

	t.Run("ConcurrentMachinePerformance", func(t *testing.T) {
		testConcurrentMachinePerformance(t)
	})

	t.Run("MemoryUsagePerformance", func(t *testing.T) {
		testMemoryUsagePerformance(t)
	})

	t.Run("ScalabilityPerformance", func(t *testing.T) {
		testScalabilityPerformance(t)
	})
}

// testValidationPerformance tests validation performance across different statechart sizes
func testValidationPerformance(t *testing.T) {
	testCases := []struct {
		name         string
		numStates    int
		numTransitions int
		maxDuration  time.Duration
	}{
		{"Small", 10, 15, 10 * time.Millisecond},
		{"Medium", 50, 100, 50 * time.Millisecond},
		{"Large", 200, 500, 200 * time.Millisecond},
		{"ExtraLarge", 500, 1000, 500 * time.Millisecond},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			// Create statechart of specified size
			statechart := testutil.CreateLargeStatechart(tc.numStates, tc.numTransitions)
			
			// Measure validation performance
			start := time.Now()
			
			// Test semantic validation
			wrapper := semantics.NewStatechart(statechart)
			err := wrapper.Validate()
			
			duration := time.Since(start)
			
			if err != nil {
				t.Logf("Validation failed (this may be expected for large statecharts): %v", err)
			}
			
			t.Logf("Semantic validation took %v for %d states, %d transitions", 
				duration, tc.numStates, tc.numTransitions)
			
			if duration > tc.maxDuration {
				t.Errorf("Validation took too long: %v > %v", duration, tc.maxDuration)
			}
			
			// Test performance comparison with a second validation run
			start = time.Now()
			
			// Run validation again to test consistency
			wrapper2 := semantics.NewStatechart(statechart)
			err2 := wrapper2.Validate()
			
			repeatDuration := time.Since(start)
			
			if err2 != nil {
				t.Logf("Second validation error: %v", err2)
			}
			
			t.Logf("Repeat validation took %v for %d states, %d transitions", 
				repeatDuration, tc.numStates, tc.numTransitions)
			
			// Performance should be consistent
			if repeatDuration > duration*2 {
				t.Logf("Note: repeat validation took significantly longer (%v vs %v)", repeatDuration, duration)
			}
		})
	}
}

// testMachineCreationPerformance tests machine creation and lifecycle performance
func testMachineCreationPerformance(t *testing.T) {
	testCases := []struct {
		name           string
		createChart    func() *sc.Statechart
		numMachines    int
		maxTimePerMachine time.Duration
	}{
		{
			name:        "SimpleChart",
			createChart: testutil.CreateSimpleStatechart,
			numMachines: 100,
			maxTimePerMachine: time.Millisecond,
		},
		{
			name:        "HierarchicalChart",
			createChart: testutil.CreateHierarchicalStatechart,
			numMachines: 50,
			maxTimePerMachine: 2 * time.Millisecond,
		},
		{
			name:        "OrthogonalChart",
			createChart: testutil.CreateOrthogonalStatechart,
			numMachines: 30,
			maxTimePerMachine: 3 * time.Millisecond,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			statechart := tc.createChart()
			wrapper := semantics.NewStatechart(statechart)
			
			// Measure machine creation performance
			start := time.Now()
			machines := make([]*semantics.MachineWrapper, tc.numMachines)
			
			for i := 0; i < tc.numMachines; i++ {
				machine, err := semantics.NewMachine(wrapper, fmt.Sprintf("perf-test-%d", i), nil)
				if err != nil {
					t.Fatalf("Failed to create machine %d: %v", i, err)
				}
				machines[i] = machine
			}
			
			creationDuration := time.Since(start)
			avgCreationTime := creationDuration / time.Duration(tc.numMachines)
			
			t.Logf("Created %d machines in %v (avg: %v per machine)", 
				tc.numMachines, creationDuration, avgCreationTime)
			
			if avgCreationTime > tc.maxTimePerMachine {
				t.Errorf("Machine creation too slow: %v > %v", avgCreationTime, tc.maxTimePerMachine)
			}
			
			// Measure start/stop performance
			start = time.Now()
			for i, machine := range machines {
				if err := machine.Start(); err != nil {
					t.Errorf("Failed to start machine %d: %v", i, err)
				}
			}
			startDuration := time.Since(start)
			
			start = time.Now()
			for i, machine := range machines {
				if err := machine.Stop(); err != nil {
					t.Errorf("Failed to stop machine %d: %v", i, err)
				}
			}
			stopDuration := time.Since(start)
			
			t.Logf("Started %d machines in %v, stopped in %v", 
				tc.numMachines, startDuration, stopDuration)
			
			avgStartTime := startDuration / time.Duration(tc.numMachines)
			avgStopTime := stopDuration / time.Duration(tc.numMachines)
			
			if avgStartTime > tc.maxTimePerMachine {
				t.Errorf("Machine start too slow: %v > %v", avgStartTime, tc.maxTimePerMachine)
			}
			if avgStopTime > tc.maxTimePerMachine {
				t.Errorf("Machine stop too slow: %v > %v", avgStopTime, tc.maxTimePerMachine)
			}
		})
	}
}

// testEventProcessingPerformance tests event processing performance
func testEventProcessingPerformance(t *testing.T) {
	statechart := &sc.Statechart{
		RootState: &sc.State{
			Label: "__root__",
			Type:  sc.StateTypeNormal,
			Children: []*sc.State{
				{Label: "State1", Type: sc.StateTypeBasic, IsInitial: true},
				{Label: "State2", Type: sc.StateTypeBasic},
				{Label: "State3", Type: sc.StateTypeBasic},
				{Label: "State4", Type: sc.StateTypeBasic},
			},
		},
		Transitions: []*sc.Transition{
			{Label: "1to2", From: []string{"State1"}, To: []string{"State2"}, Event: "NEXT"},
			{Label: "2to3", From: []string{"State2"}, To: []string{"State3"}, Event: "NEXT"},
			{Label: "3to4", From: []string{"State3"}, To: []string{"State4"}, Event: "NEXT"},
			{Label: "4to1", From: []string{"State4"}, To: []string{"State1"}, Event: "NEXT"},
		},
		Events: []*sc.Event{
			{Label: "NEXT"},
		},
	}

	wrapper := semantics.NewStatechart(statechart)
	machine, err := semantics.NewMachine(wrapper, "event-perf-test", nil)
	if err != nil {
		t.Fatalf("Failed to create machine: %v", err)
	}

	if err := machine.Start(); err != nil {
		t.Fatalf("Failed to start machine: %v", err)
	}
	defer machine.Stop()

	// Test event processing throughput
	numEvents := 10000
	start := time.Now()
	
	for i := 0; i < numEvents; i++ {
		triggered, err := machine.Step("NEXT")
		if err != nil {
			t.Errorf("Event processing failed at event %d: %v", i, err)
			break
		}
		if !triggered {
			t.Errorf("Event %d should have triggered a transition", i)
		}
	}
	
	duration := time.Since(start)
	eventsPerSecond := float64(numEvents) / duration.Seconds()
	
	t.Logf("Processed %d events in %v (%.0f events/sec)", numEvents, duration, eventsPerSecond)
	
	// Should process at least 10,000 events per second
	if eventsPerSecond < 10000 {
		t.Errorf("Event processing too slow: %.0f events/sec < 10000 events/sec", eventsPerSecond)
	}
	
	// Verify final state is consistent
	if err := machine.Validate(); err != nil {
		t.Errorf("Machine validation failed after event processing: %v", err)
	}
	
	// Test event processing with actions
	statechartWithActions := &sc.Statechart{
		RootState: statechart.RootState,
		Transitions: []*sc.Transition{
			{
				Label: "1to2WithAction",
				From:  []string{"State1"},
				To:    []string{"State2"},
				Event: "NEXT_WITH_ACTION",
				Actions: []*sc.Action{
					{Label: "increment_count"},
				},
			},
		},
		Events: []*sc.Event{
			{Label: "NEXT_WITH_ACTION"},
		},
	}

	wrapperWithActions := semantics.NewStatechart(statechartWithActions)
	machineWithActions, err := semantics.NewMachine(wrapperWithActions, "action-perf-test", &structpb.Struct{
		Fields: map[string]*structpb.Value{
			"count": structpb.NewNumberValue(0),
		},
	})
	if err != nil {
		t.Fatalf("Failed to create machine with actions: %v", err)
	}

	if err := machineWithActions.Start(); err != nil {
		t.Fatalf("Failed to start machine with actions: %v", err)
	}
	defer machineWithActions.Stop()

	// Test performance with actions
	numEventsWithActions := 1000
	start = time.Now()
	
	for i := 0; i < numEventsWithActions; i++ {
		machineWithActions.Step("NEXT_WITH_ACTION")
		machineWithActions.Reset() // Reset to initial state for next iteration
		machineWithActions.Start()
	}
	
	durationWithActions := time.Since(start)
	eventsPerSecondWithActions := float64(numEventsWithActions) / durationWithActions.Seconds()
	
	t.Logf("Processed %d events with actions in %v (%.0f events/sec)", 
		numEventsWithActions, durationWithActions, eventsPerSecondWithActions)
}

// testConcurrentMachinePerformance tests performance under concurrent load
func testConcurrentMachinePerformance(t *testing.T) {
	statechart := testutil.CreateSimpleStatechart()
	wrapper := semantics.NewStatechart(statechart)
	
	numMachines := 50
	eventsPerMachine := 100
	
	// Create machines
	machines := make([]*semantics.MachineWrapper, numMachines)
	for i := 0; i < numMachines; i++ {
		machine, err := semantics.NewMachine(wrapper, fmt.Sprintf("concurrent-perf-%d", i), nil)
		if err != nil {
			t.Fatalf("Failed to create machine %d: %v", i, err)
		}
		machines[i] = machine
		machine.Start()
	}
	defer func() {
		for _, machine := range machines {
			machine.Stop()
		}
	}()

	// Test concurrent event processing
	start := time.Now()
	var wg sync.WaitGroup
	
	for i, machine := range machines {
		wg.Add(1)
		go func(machineID int, m *semantics.MachineWrapper) {
			defer wg.Done()
			for j := 0; j < eventsPerMachine; j++ {
				m.Step("go")
			}
		}(i, machine)
	}
	
	wg.Wait()
	duration := time.Since(start)
	
	totalEvents := numMachines * eventsPerMachine
	eventsPerSecond := float64(totalEvents) / duration.Seconds()
	
	t.Logf("Processed %d events across %d machines in %v (%.0f events/sec)", 
		totalEvents, numMachines, duration, eventsPerSecond)
	
	// Should handle concurrent load efficiently
	if eventsPerSecond < 5000 {
		t.Errorf("Concurrent event processing too slow: %.0f events/sec < 5000 events/sec", eventsPerSecond)
	}
	
	// Verify all machines are still valid
	for i, machine := range machines {
		if err := machine.Validate(); err != nil {
			t.Errorf("Machine %d validation failed: %v", i, err)
		}
	}
}

// testMemoryUsagePerformance tests memory usage characteristics
func testMemoryUsagePerformance(t *testing.T) {
	// Get initial memory stats
	var m1 runtime.MemStats
	runtime.GC()
	runtime.ReadMemStats(&m1)
	
	// Create many machines and process events
	statechart := testutil.CreateSimpleStatechart()
	wrapper := semantics.NewStatechart(statechart)
	
	numMachines := 100
	machines := make([]*semantics.MachineWrapper, numMachines)
	
	for i := 0; i < numMachines; i++ {
		machine, err := semantics.NewMachine(wrapper, fmt.Sprintf("memory-test-%d", i), nil)
		if err != nil {
			t.Fatalf("Failed to create machine %d: %v", i, err)
		}
		machines[i] = machine
		machine.Start()
		
		// Process some events to build up history
		for j := 0; j < 10; j++ {
			machine.Step("go")
		}
	}
	
	// Get memory stats after creation
	var m2 runtime.MemStats
	runtime.GC()
	runtime.ReadMemStats(&m2)
	
	memoryUsed := m2.Alloc - m1.Alloc
	memoryPerMachine := memoryUsed / uint64(numMachines)
	
	t.Logf("Created %d machines using %d bytes total (%.1f KB per machine)", 
		numMachines, memoryUsed, float64(memoryPerMachine)/1024)
	
	// Clean up
	for _, machine := range machines {
		machine.Stop()
	}
	
	// Force garbage collection and check memory cleanup
	runtime.GC()
	runtime.GC() // Run twice to ensure cleanup
	
	var m3 runtime.MemStats
	runtime.ReadMemStats(&m3)
	
	memoryAfterCleanup := m3.Alloc
	memoryReclaimed := m2.Alloc - memoryAfterCleanup
	
	t.Logf("Memory after cleanup: %d bytes (reclaimed: %d bytes, %.1f%%)", 
		memoryAfterCleanup, memoryReclaimed, float64(memoryReclaimed)/float64(m2.Alloc)*100)
	
	// Memory per machine should be reasonable
	if memoryPerMachine > 10*1024 { // 10KB per machine seems reasonable
		t.Errorf("Memory usage per machine too high: %.1f KB > 10 KB", float64(memoryPerMachine)/1024)
	}
	
	// Should reclaim at least 50% of memory
	if float64(memoryReclaimed)/float64(m2.Alloc) < 0.5 {
		t.Errorf("Poor memory cleanup: only %.1f%% reclaimed", float64(memoryReclaimed)/float64(m2.Alloc)*100)
	}
}

// testScalabilityPerformance tests scalability characteristics
func testScalabilityPerformance(t *testing.T) {
	scales := []int{10, 50, 100, 200}
	
	for _, scale := range scales {
		t.Run(fmt.Sprintf("Scale%d", scale), func(t *testing.T) {
			// Create statechart with specified scale
			statechart := testutil.CreateLargeStatechart(scale, scale*2)
			
			// Measure validation time
			start := time.Now()
			wrapper := semantics.NewStatechart(statechart)
			err := wrapper.Validate()
			validationDuration := time.Since(start)
			
			if err != nil {
				t.Logf("Validation failed for scale %d: %v", scale, err)
			}
			
			// Measure machine creation time
			start = time.Now()
			machine, err := semantics.NewMachine(wrapper, fmt.Sprintf("scale-test-%d", scale), nil)
			creationDuration := time.Since(start)
			
			if err != nil {
				t.Logf("Machine creation failed for scale %d: %v", scale, err)
				return
			}
			
			// Measure startup time
			start = time.Now()
			err = machine.Start()
			startupDuration := time.Since(start)
			
			if err != nil {
				t.Logf("Machine startup failed for scale %d: %v", scale, err)
				machine.Stop()
				return
			}
			
			// Measure event processing time
			start = time.Now()
			numEvents := 10
			for i := 0; i < numEvents; i++ {
				machine.Step(fmt.Sprintf("event%d", i%scale))
			}
			eventDuration := time.Since(start)
			
			machine.Stop()
			
			t.Logf("Scale %d: validation=%v, creation=%v, startup=%v, events=%v", 
				scale, validationDuration, creationDuration, startupDuration, eventDuration)
			
			// Performance should not degrade exponentially
			maxValidationTime := time.Duration(scale) * time.Millisecond
			if validationDuration > maxValidationTime {
				t.Errorf("Validation time scales poorly: %v > %v for scale %d", 
					validationDuration, maxValidationTime, scale)
			}
			
			maxCreationTime := time.Duration(scale/10+1) * time.Millisecond
			if creationDuration > maxCreationTime {
				t.Errorf("Creation time scales poorly: %v > %v for scale %d", 
					creationDuration, maxCreationTime, scale)
			}
		})
	}
}

// BenchmarkIntegrationWorkflows provides benchmark tests for integration workflows
func BenchmarkIntegrationWorkflows(b *testing.B) {
	b.Run("SimpleStatechartLifecycle", func(b *testing.B) {
		statechart := testutil.CreateSimpleStatechart()
		wrapper := semantics.NewStatechart(statechart)
		
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			machine, err := semantics.NewMachine(wrapper, "bench-test", nil)
			if err != nil {
				b.Fatal(err)
			}
			machine.Start()
			machine.Step("go")
			machine.Stop()
		}
	})
	
	b.Run("HierarchicalStatechartLifecycle", func(b *testing.B) {
		statechart := testutil.CreateHierarchicalStatechart()
		wrapper := semantics.NewStatechart(statechart)
		
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			machine, err := semantics.NewMachine(wrapper, "bench-test", nil)
			if err != nil {
				b.Fatal(err)
			}
			machine.Start()
			machine.Step("start")
			machine.Step("finish")
			machine.Step("stop")
			machine.Stop()
		}
	})
	
	b.Run("SemanticValidation", func(b *testing.B) {
		statechart := testutil.CreateSimpleStatechart()
		
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			wrapper := semantics.NewStatechart(statechart)
			wrapper.Validate()
		}
	})
	
	b.Run("EventProcessing", func(b *testing.B) {
		statechart := testutil.CreateSimpleStatechart()
		wrapper := semantics.NewStatechart(statechart)
		machine, _ := semantics.NewMachine(wrapper, "bench-test", nil)
		machine.Start()
		
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			machine.Step("go")
		}
		
		machine.Stop()
	})
	
	b.Run("ConcurrentMachines", func(b *testing.B) {
		statechart := testutil.CreateSimpleStatechart()
		wrapper := semantics.NewStatechart(statechart)
		
		b.RunParallel(func(pb *testing.PB) {
			for pb.Next() {
				machine, err := semantics.NewMachine(wrapper, "bench-test", nil)
				if err != nil {
					b.Fatal(err)
				}
				machine.Start()
				machine.Step("go")
				machine.Stop()
			}
		})
	})
}

// TestPerformanceRegression tests for performance regressions
func TestPerformanceRegression(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping performance regression tests in short mode")
	}

	// Baseline performance expectations
	baselines := map[string]struct {
		operation    func() time.Duration
		maxDuration  time.Duration
		description  string
	}{
		"SimpleValidation": {
			operation: func() time.Duration {
				start := time.Now()
				statechart := testutil.CreateSimpleStatechart()
				wrapper := semantics.NewStatechart(statechart)
				wrapper.Validate()
				return time.Since(start)
			},
			maxDuration: 5 * time.Millisecond,
			description: "Simple statechart validation",
		},
		"MachineCreation": {
			operation: func() time.Duration {
				start := time.Now()
				statechart := testutil.CreateSimpleStatechart()
				wrapper := semantics.NewStatechart(statechart)
				machine, _ := semantics.NewMachine(wrapper, "regression-test", nil)
				machine.Start()
				machine.Stop()
				return time.Since(start)
			},
			maxDuration: 2 * time.Millisecond,
			description: "Machine creation and basic lifecycle",
		},
		"EventProcessing": {
			operation: func() time.Duration {
				statechart := testutil.CreateSimpleStatechart()
				wrapper := semantics.NewStatechart(statechart)
				machine, _ := semantics.NewMachine(wrapper, "regression-test", nil)
				machine.Start()
				
				start := time.Now()
				for i := 0; i < 100; i++ {
					machine.Step("go")
				}
				duration := time.Since(start)
				
				machine.Stop()
				return duration
			},
			maxDuration: 10 * time.Millisecond,
			description: "100 event processing operations",
		},
	}

	for name, baseline := range baselines {
		t.Run(name, func(t *testing.T) {
			// Run operation multiple times and take average
			numRuns := 5
			totalDuration := time.Duration(0)
			
			for i := 0; i < numRuns; i++ {
				duration := baseline.operation()
				totalDuration += duration
			}
			
			avgDuration := totalDuration / time.Duration(numRuns)
			
			t.Logf("%s took %v (avg of %d runs)", baseline.description, avgDuration, numRuns)
			
			if avgDuration > baseline.maxDuration {
				t.Errorf("Performance regression detected: %s took %v > %v", 
					baseline.description, avgDuration, baseline.maxDuration)
			}
		})
	}
}