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

// TestConcurrentMachineOperations tests concurrent access to machine operations
func TestConcurrentMachineOperations(t *testing.T) {
	// Create a statechart suitable for concurrent testing
	statechart := &sc.Statechart{
		RootState: &sc.State{
			Label: "__root__",
			Type:  sc.StateTypeNormal,
			Children: []*sc.State{
				{
					Label:     "Idle",
					Type:      sc.StateTypeBasic,
					IsInitial: true,
				},
				{
					Label: "Processing",
					Type:  sc.StateTypeBasic,
				},
				{
					Label: "Complete",
					Type:  sc.StateTypeBasic,
				},
			},
		},
		Transitions: []*sc.Transition{
			{
				Label: "StartProcessing",
				From:  []string{"Idle"},
				To:    []string{"Processing"},
				Event: "START",
				Actions: []*sc.Action{
					{Label: "increment_count"},
				},
			},
			{
				Label: "FinishProcessing",
				From:  []string{"Processing"},
				To:    []string{"Complete"},
				Event: "FINISH",
				Actions: []*sc.Action{
					{Label: "increment_count"},
				},
			},
			{
				Label: "Reset",
				From:  []string{"Complete"},
				To:    []string{"Idle"},
				Event: "RESET",
				Actions: []*sc.Action{
					{Label: "reset_count"},
				},
			},
		},
		Events: []*sc.Event{
			{Label: "START"},
			{Label: "FINISH"},
			{Label: "RESET"},
		},
	}

	t.Run("ConcurrentStepExecution", func(t *testing.T) {
		wrapper := semantics.NewStatechart(statechart)
		machine, err := semantics.NewMachine(wrapper, "concurrent-test", &structpb.Struct{
			Fields: map[string]*structpb.Value{
				"count": structpb.NewNumberValue(0),
			},
		})
		if err != nil {
			t.Fatalf("Failed to create machine: %v", err)
		}

		if err := machine.Start(); err != nil {
			t.Fatalf("Failed to start machine: %v", err)
		}
		defer machine.Stop()

		// Test concurrent step operations
		numGoroutines := 10
		numStepsPerGoroutine := 5
		var wg sync.WaitGroup

		// Channel to collect results
		results := make(chan bool, numGoroutines*numStepsPerGoroutine)
		errors := make(chan error, numGoroutines*numStepsPerGoroutine)

		events := []string{"START", "FINISH", "RESET"}

		for i := 0; i < numGoroutines; i++ {
			wg.Add(1)
			go func(goroutineID int) {
				defer wg.Done()
				for j := 0; j < numStepsPerGoroutine; j++ {
					event := events[j%len(events)]
					triggered, err := machine.Step(event)
					if err != nil {
						errors <- err
					} else {
						results <- triggered
					}
					// Small delay to increase chance of concurrency
					time.Sleep(time.Millisecond)
				}
			}(i)
		}

		// Wait for all goroutines to complete
		wg.Wait()
		close(results)
		close(errors)

		// Check for errors
		errorCount := 0
		for err := range errors {
			errorCount++
			t.Logf("Step error: %v", err)
		}

		if errorCount > 0 {
			t.Errorf("Got %d errors during concurrent execution", errorCount)
		}

		// Verify machine is still in a valid state
		if err := machine.Validate(); err != nil {
			t.Errorf("Machine validation failed after concurrent operations: %v", err)
		}

		// Verify step history is reasonable
		history := machine.GetStepHistory()
		expectedMinSteps := 1 // At least the start step
		if len(history) < expectedMinSteps {
			t.Errorf("Expected at least %d steps, got %d", expectedMinSteps, len(history))
		}
	})

	t.Run("ConcurrentStateQueries", func(t *testing.T) {
		wrapper := semantics.NewStatechart(statechart)
		machine, err := semantics.NewMachine(wrapper, "query-test", nil)
		if err != nil {
			t.Fatalf("Failed to create machine: %v", err)
		}

		if err := machine.Start(); err != nil {
			t.Fatalf("Failed to start machine: %v", err)
		}
		defer machine.Stop()

		// Test concurrent read operations
		numReaders := 20
		readDuration := 100 * time.Millisecond
		var wg sync.WaitGroup

		// Start concurrent readers
		for i := 0; i < numReaders; i++ {
			wg.Add(1)
			go func(readerID int) {
				defer wg.Done()
				start := time.Now()
				for time.Since(start) < readDuration {
					// These operations should be safe to call concurrently
					_ = machine.GetCurrentConfiguration()
					_ = machine.GetContext()
					_ = machine.IsRunning()
					_ = machine.IsStopped()
					_ = machine.GetStepHistory()
					_ = machine.GetErrors()

					// Small yield to other goroutines
					runtime.Gosched()
				}
			}(i)
		}

		// Start a writer that occasionally updates state
		wg.Add(1)
		go func() {
			defer wg.Done()
			events := []string{"START", "FINISH", "RESET"}
			for i := 0; i < 10; i++ {
				event := events[i%len(events)]
				machine.Step(event)
				time.Sleep(10 * time.Millisecond)
			}
		}()

		wg.Wait()

		// Verify machine is still in a valid state
		if err := machine.Validate(); err != nil {
			t.Errorf("Machine validation failed after concurrent queries: %v", err)
		}
	})

	t.Run("ConcurrentMachineLifecycle", func(t *testing.T) {
		// Test concurrent start/stop operations
		numMachines := 10
		var wg sync.WaitGroup

		startErrors := make(chan error, numMachines)
		stopErrors := make(chan error, numMachines)

		for i := 0; i < numMachines; i++ {
			wg.Add(1)
			go func(machineID int) {
				defer wg.Done()

				wrapper := semantics.NewStatechart(statechart)
				machine, err := semantics.NewMachine(wrapper, "lifecycle-test", nil)
				if err != nil {
					startErrors <- err
					return
				}

				// Start machine
				if err := machine.Start(); err != nil {
					startErrors <- err
					return
				}

				// Do some work
				machine.Step("START")
				machine.Step("FINISH")

				// Stop machine
				if err := machine.Stop(); err != nil {
					stopErrors <- err
					return
				}

				// Verify final state
				if err := machine.Validate(); err != nil {
					stopErrors <- err
				}
			}(i)
		}

		wg.Wait()
		close(startErrors)
		close(stopErrors)

		// Check for errors
		startErrorCount := 0
		for err := range startErrors {
			startErrorCount++
			t.Logf("Start error: %v", err)
		}

		stopErrorCount := 0
		for err := range stopErrors {
			stopErrorCount++
			t.Logf("Stop error: %v", err)
		}

		if startErrorCount > 0 {
			t.Errorf("Got %d start errors", startErrorCount)
		}
		if stopErrorCount > 0 {
			t.Errorf("Got %d stop errors", stopErrorCount)
		}
	})
}

// TestConcurrentEventProcessing tests concurrent event processing with the event processor
func TestConcurrentEventProcessing(t *testing.T) {
	// Create a test machine
	machine := &sc.Machine{
		Id:    "concurrent-event-test",
		State: sc.MachineStateRunning,
		Context: &structpb.Struct{
			Fields: map[string]*structpb.Value{
				"count": structpb.NewNumberValue(0),
			},
		},
		Statechart: &sc.Statechart{
			RootState: &sc.State{
				Label: "__root__",
				Children: []*sc.State{
					{Label: "StateA", Type: sc.StateTypeBasic, IsInitial: true},
					{Label: "StateB", Type: sc.StateTypeBasic},
					{Label: "StateC", Type: sc.StateTypeBasic},
				},
			},
			Transitions: []*sc.Transition{
				{
					Label: "AtoB",
					From:  []string{"StateA"},
					To:    []string{"StateB"},
					Event: "GO_B",
				},
				{
					Label: "BtoC",
					From:  []string{"StateB"},
					To:    []string{"StateC"},
					Event: "GO_C",
				},
				{
					Label: "CtoA",
					From:  []string{"StateC"},
					To:    []string{"StateA"},
					Event: "GO_A",
				},
			},
			Events: []*sc.Event{
				{Label: "GO_B"},
				{Label: "GO_C"},
				{Label: "GO_A"},
			},
		},
		Configuration: &sc.Configuration{
			States: []*sc.StateRef{{Label: "StateA"}},
		},
	}

	// Create event processor
	processor := semantics.NewEventProcessor(machine)
	processor.EnableTracing()

	// Start processor
	processor.Start()
	defer processor.Stop()

	t.Run("ConcurrentEventSending", func(t *testing.T) {
		numSenders := 10
		eventsPerSender := 20
		var wg sync.WaitGroup

		events := []string{"GO_B", "GO_C", "GO_A"}
		errors := make(chan error, numSenders*eventsPerSender)

		for i := 0; i < numSenders; i++ {
			wg.Add(1)
			go func(senderID int) {
				defer wg.Done()
				for j := 0; j < eventsPerSender; j++ {
					event := events[j%len(events)]
					if err := processor.SendEvent(event, nil); err != nil {
						errors <- err
					}
					// Small delay to avoid overwhelming the processor
					time.Sleep(time.Millisecond)
				}
			}(i)
		}

		wg.Wait()

		// Wait for processing to complete
		time.Sleep(100 * time.Millisecond)

		close(errors)

		// Check for errors
		errorCount := 0
		for err := range errors {
			errorCount++
			t.Logf("Event sending error: %v", err)
		}

		if errorCount > 0 {
			t.Errorf("Got %d errors during concurrent event sending", errorCount)
		}

		// Verify trace has entries
		trace := processor.GetTrace()
		if len(trace) == 0 {
			t.Error("Expected trace entries from event processing")
		}
	})

	t.Run("PriorityEventHandling", func(t *testing.T) {
		processor.ClearTrace()

		numEvents := 30
		var wg sync.WaitGroup

		priorities := []semantics.EventPriority{
			semantics.PriorityLow,
			semantics.PriorityNormal,
			semantics.PriorityHigh,
		}

		// Send events with different priorities concurrently
		for i := 0; i < numEvents; i++ {
			wg.Add(1)
			go func(eventID int) {
				defer wg.Done()
				priority := priorities[eventID%len(priorities)]
				event := "GO_B"
				if err := processor.SendEventWithPriority(event, priority, nil); err != nil {
					t.Logf("Failed to send priority event: %v", err)
				}
			}(i)
		}

		wg.Wait()

		// Wait for processing
		time.Sleep(100 * time.Millisecond)

		// Verify events were processed
		trace := processor.GetTrace()
		processedCount := 0
		for _, entry := range trace {
			if entry.Action == "processed" {
				processedCount++
			}
		}

		if processedCount == 0 {
			t.Error("Expected some events to be processed")
		}
	})
}

// TestConcurrentValidation tests concurrent validation operations
func TestConcurrentValidation(t *testing.T) {
	testCases := []struct {
		name        string
		createChart func() *sc.Statechart
	}{
		{
			name: "SimpleChart",
			createChart: func() *sc.Statechart {
				return testutil.CreateSimpleStatechart()
			},
		},
		{
			name: "HierarchicalChart",
			createChart: func() *sc.Statechart {
				return testutil.CreateHierarchicalStatechart()
			},
		},
		{
			name: "OrthogonalChart",
			createChart: func() *sc.Statechart {
				return testutil.CreateOrthogonalStatechart()
			},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			statechart := tc.createChart()
			numValidators := 20
			var wg sync.WaitGroup

			errors := make(chan error, numValidators)

			// Run concurrent validations
			for i := 0; i < numValidators; i++ {
				wg.Add(1)
				go func(validatorID int) {
					defer wg.Done()

					// Validate using semantic wrapper
					wrapper := semantics.NewStatechart(statechart)
					if err := wrapper.Validate(); err != nil {
						errors <- err
					}

					// Also test machine validation
					machine, err := semantics.NewMachine(wrapper, "validation-test", nil)
					if err != nil {
						errors <- err
						return
					}

					if err := machine.Validate(); err != nil {
						errors <- err
					}
				}(i)
			}

			wg.Wait()
			close(errors)

			// Check for errors
			errorCount := 0
			for err := range errors {
				errorCount++
				t.Logf("Validation error: %v", err)
			}

			if errorCount > 0 {
				t.Errorf("Got %d validation errors during concurrent execution", errorCount)
			}
		})
	}
}

// TestConcurrentConfigurationAccess tests concurrent access to machine configuration
func TestConcurrentConfigurationAccess(t *testing.T) {
	statechart := testutil.CreateHierarchicalStatechart()
	wrapper := semantics.NewStatechart(statechart)
	machine, err := semantics.NewMachine(wrapper, "config-test", nil)
	if err != nil {
		t.Fatalf("Failed to create machine: %v", err)
	}

	if err := machine.Start(); err != nil {
		t.Fatalf("Failed to start machine: %v", err)
	}
	defer machine.Stop()

	numReaders := 15
	numWriters := 5
	duration := 200 * time.Millisecond
	var wg sync.WaitGroup

	// Start configuration readers
	configErrors := make(chan error, numReaders)
	for i := 0; i < numReaders; i++ {
		wg.Add(1)
		go func(readerID int) {
			defer wg.Done()
			start := time.Now()
			for time.Since(start) < duration {
				config := machine.GetCurrentConfiguration()
				if config == nil {
					configErrors <- fmt.Errorf("reader %d: got nil configuration", readerID)
					return
				}

				// Verify configuration has states
				if len(config.States) == 0 {
					configErrors <- fmt.Errorf("reader %d: configuration has no states", readerID)
					return
				}

				// Small delay
				time.Sleep(time.Millisecond)
			}
		}(i)
	}

	// Start state transition writers
	stepErrors := make(chan error, numWriters)
	events := []string{"start", "finish", "stop"}
	for i := 0; i < numWriters; i++ {
		wg.Add(1)
		go func(writerID int) {
			defer wg.Done()
			start := time.Now()
			eventIndex := 0
			for time.Since(start) < duration {
				event := events[eventIndex%len(events)]
				_, err := machine.Step(event)
				if err != nil {
					stepErrors <- fmt.Errorf("writer %d: step failed: %v", writerID, err)
					return
				}
				eventIndex++
				time.Sleep(10 * time.Millisecond)
			}
		}(i)
	}

	wg.Wait()
	close(configErrors)
	close(stepErrors)

	// Check for errors
	configErrorCount := 0
	for err := range configErrors {
		configErrorCount++
		t.Logf("Config error: %v", err)
	}

	stepErrorCount := 0
	for err := range stepErrors {
		stepErrorCount++
		t.Logf("Step error: %v", err)
	}

	if configErrorCount > 0 {
		t.Errorf("Got %d configuration access errors", configErrorCount)
	}
	if stepErrorCount > 0 {
		t.Errorf("Got %d step execution errors", stepErrorCount)
	}

	// Final validation
	if err := machine.Validate(); err != nil {
		t.Errorf("Machine validation failed after concurrent access: %v", err)
	}
}

// TestRaceConditionDetection tests for common race conditions
func TestRaceConditionDetection(t *testing.T) {
	// This test is designed to run with -race flag to detect race conditions
	if !testing.Short() {
		t.Log("Running race condition detection test (use -race flag)")
	}

	statechart := &sc.Statechart{
		RootState: &sc.State{
			Label: "__root__",
			Type:  sc.StateTypeNormal,
			Children: []*sc.State{
				{Label: "State1", Type: sc.StateTypeBasic, IsInitial: true},
				{Label: "State2", Type: sc.StateTypeBasic},
			},
		},
		Transitions: []*sc.Transition{
			{
				Label: "Toggle",
				From:  []string{"State1"},
				To:    []string{"State2"},
				Event: "TOGGLE",
				Actions: []*sc.Action{
					{Label: "increment_count"},
				},
			},
			{
				Label: "ToggleBack",
				From:  []string{"State2"},
				To:    []string{"State1"},
				Event: "TOGGLE",
				Actions: []*sc.Action{
					{Label: "increment_count"},
				},
			},
		},
		Events: []*sc.Event{
			{Label: "TOGGLE"},
		},
	}

	wrapper := semantics.NewStatechart(statechart)
	machine, err := semantics.NewMachine(wrapper, "race-test", &structpb.Struct{
		Fields: map[string]*structpb.Value{
			"count": structpb.NewNumberValue(0),
		},
	})
	if err != nil {
		t.Fatalf("Failed to create machine: %v", err)
	}

	if err := machine.Start(); err != nil {
		t.Fatalf("Failed to start machine: %v", err)
	}
	defer machine.Stop()

	// Run many concurrent operations to try to trigger race conditions
	numGoroutines := 50
	operationsPerGoroutine := 20
	var wg sync.WaitGroup

	for i := 0; i < numGoroutines; i++ {
		wg.Add(1)
		go func(goroutineID int) {
			defer wg.Done()
			for j := 0; j < operationsPerGoroutine; j++ {
				switch j % 4 {
				case 0:
					machine.Step("TOGGLE")
				case 1:
					machine.GetCurrentConfiguration()
				case 2:
					machine.GetContext()
				case 3:
					machine.Validate()
				}
			}
		}(i)
	}

	wg.Wait()

	// Final checks
	if err := machine.Validate(); err != nil {
		t.Errorf("Machine validation failed: %v", err)
	}

	config := machine.GetCurrentConfiguration()
	activeToggleStates := 0
	for _, state := range config.States {
		if state.Label == "State1" || state.Label == "State2" {
			activeToggleStates++
		}
	}
	if activeToggleStates != 1 {
		t.Errorf("Expected exactly 1 toggle state in final configuration, got %v", config.States)
	}

	context := machine.GetContext()
	if context == nil {
		t.Error("Context should not be nil")
	}
}

// Note: fmt.Errorf is already imported from the fmt package
