package integration

import (
	"testing"

	"github.com/tmc/sc"
	"github.com/tmc/sc/semantics/v1"
	testutil "github.com/tmc/sc/testing"
	"google.golang.org/protobuf/types/known/structpb"
)

// TestMachineLifecycleIntegration tests the complete machine lifecycle from creation to cleanup
func TestMachineLifecycleIntegration(t *testing.T) {
	testCases := []struct {
		name         string
		description  string
		setupChart   func() *sc.Statechart
		setupContext func() *structpb.Struct
		events       []string
		validate     func(t *testing.T, machine *semantics.MachineWrapper, history []*sc.Step)
	}{
		{
			name:        "SimpleLifecycle",
			description: "Basic machine lifecycle with simple statechart",
			setupChart: func() *sc.Statechart {
				return testutil.CreateSimpleStatechart()
			},
			setupContext: func() *structpb.Struct {
				return &structpb.Struct{
					Fields: map[string]*structpb.Value{
						"initialized": structpb.NewBoolValue(true),
					},
				}
			},
			events: []string{"go"},
			validate: func(t *testing.T, machine *semantics.MachineWrapper, history []*sc.Step) {
				// Verify final configuration contains target state
				config := machine.GetCurrentConfiguration()
				foundB := false
				for _, state := range config.States {
					if state.Label == "B" {
						foundB = true
						break
					}
				}
				if !foundB {
					t.Errorf("Expected final configuration to contain state 'B', got %v", config.States)
				}

				// Verify step history contains start, event, and stop
				if len(history) < 3 {
					t.Errorf("Expected at least 3 steps (start, event, stop), got %d", len(history))
				}
			},
		},
		{
			name:        "HierarchicalLifecycle",
			description: "Machine lifecycle with hierarchical statechart",
			setupChart: func() *sc.Statechart {
				return testutil.CreateHierarchicalStatechart()
			},
			setupContext: func() *structpb.Struct {
				return &structpb.Struct{Fields: make(map[string]*structpb.Value)}
			},
			events: []string{"start", "finish", "stop"},
			validate: func(t *testing.T, machine *semantics.MachineWrapper, history []*sc.Step) {
				// Verify final state is Inactive
				config := machine.GetCurrentConfiguration()
				found := false
				for _, state := range config.States {
					if state.Label == "Inactive" {
						found = true
						break
					}
				}
				if !found {
					t.Errorf("Expected final configuration to contain 'Inactive' state")
				}
			},
		},
		{
			name:        "OrthogonalLifecycle",
			description: "Machine lifecycle with orthogonal statechart",
			setupChart: func() *sc.Statechart {
				return testutil.CreateOrthogonalStatechart()
			},
			setupContext: func() *structpb.Struct {
				return &structpb.Struct{Fields: make(map[string]*structpb.Value)}
			},
			events: []string{"switch_a", "switch_b"},
			validate: func(t *testing.T, machine *semantics.MachineWrapper, history []*sc.Step) {
				// For orthogonal statecharts, we verify that the machine executed the transitions
				// The final state might be affected by semantic implementation details
				config := machine.GetCurrentConfiguration()
				hasRegionA, hasRegionB := false, false
				for _, state := range config.States {
					if state.Label == "RegionA" {
						hasRegionA = true
					}
					if state.Label == "RegionB" {
						hasRegionB = true
					}
				}
				if !hasRegionA || !hasRegionB {
					t.Errorf("Expected both RegionA and RegionB in final configuration, got %v", config.States)
				}

				// Verify that step history shows transitions occurred
				if len(history) < 3 { // start + 2 events
					t.Errorf("Expected at least 3 steps, got %d", len(history))
				}
			},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			// Phase 1: Creation and Validation
			statechart := tc.setupChart()
			if statechart == nil {
				t.Fatal("Failed to create statechart")
			}

			// Validate statechart using semantic validation
			wrapper := semantics.NewStatechart(statechart)
			if err := wrapper.Validate(); err != nil {
				t.Fatalf("Semantic validation failed: %v", err)
			}

			// Phase 2: Machine Creation
			machine, err := semantics.NewMachine(wrapper, "lifecycle-test", tc.setupContext())
			if err != nil {
				t.Fatalf("Failed to create machine: %v", err)
			}

			// Phase 3: Machine Startup
			if !machine.IsStopped() {
				t.Error("Machine should be stopped initially")
			}

			if err := machine.Start(); err != nil {
				t.Fatalf("Failed to start machine: %v", err)
			}

			if !machine.IsRunning() {
				t.Error("Machine should be running after start")
			}

			// Phase 4: Event Processing
			for _, event := range tc.events {
				triggered, err := machine.Step(event)
				if err != nil {
					t.Fatalf("Failed to process event '%s': %v", event, err)
				}
				if !triggered {
					t.Logf("Event '%s' did not trigger any transitions", event)
				}
			}

			// Phase 5: Validation During Execution
			if err := machine.Validate(); err != nil {
				t.Errorf("Machine validation failed during execution: %v", err)
			}

			// Phase 6: Machine Shutdown
			if err := machine.Stop(); err != nil {
				t.Fatalf("Failed to stop machine: %v", err)
			}

			if !machine.IsStopped() {
				t.Error("Machine should be stopped after stop")
			}

			// Phase 7: Post-execution Validation
			history := machine.GetStepHistory()
			tc.validate(t, machine, history)

			// Phase 8: Error Checking
			errors := machine.GetErrors()
			if len(errors) > 0 {
				t.Logf("Machine accumulated %d errors during execution:", len(errors))
				for i, err := range errors {
					t.Logf("  Error %d: %v", i+1, err)
				}
			}

			// Phase 9: Reset and Reuse
			if err := machine.Reset(); err != nil {
				t.Fatalf("Failed to reset machine: %v", err)
			}

			if !machine.IsStopped() {
				t.Error("Machine should be stopped after reset")
			}

			resetHistory := machine.GetStepHistory()
			if len(resetHistory) != 0 {
				t.Errorf("Expected empty history after reset, got %d steps", len(resetHistory))
			}
		})
	}
}

// TestMachineLifecycleWithActions tests machine lifecycle with context-modifying actions
func TestMachineLifecycleWithActions(t *testing.T) {
	// Create a statechart with actions that modify context
	statechart := &sc.Statechart{
		RootState: &sc.State{
			Label: "__root__",
			Type:  sc.StateTypeNormal,
			Children: []*sc.State{
				{
					Label:     "Counter",
					Type:      sc.StateTypeBasic,
					IsInitial: true,
				},
			},
		},
		Transitions: []*sc.Transition{
			{
				Label: "IncrementTransition",
				From:  []string{"Counter"},
				To:    []string{"Counter"},
				Event: "INCREMENT",
				Actions: []*sc.Action{
					{Label: "increment_count"},
				},
			},
			{
				Label: "ResetTransition",
				From:  []string{"Counter"},
				To:    []string{"Counter"},
				Event: "RESET",
				Actions: []*sc.Action{
					{Label: "reset_count"},
				},
			},
		},
		Events: []*sc.Event{
			{Label: "INCREMENT"},
			{Label: "RESET"},
		},
	}

	// Create machine with initial context
	wrapper := semantics.NewStatechart(statechart)
	machine, err := semantics.NewMachine(wrapper, "action-test", &structpb.Struct{
		Fields: map[string]*structpb.Value{
			"count": structpb.NewNumberValue(0),
		},
	})
	if err != nil {
		t.Fatalf("Failed to create machine: %v", err)
	}

	// Start machine
	if err := machine.Start(); err != nil {
		t.Fatalf("Failed to start machine: %v", err)
	}

	// Test increment action
	triggered, err := machine.Step("INCREMENT")
	if err != nil {
		t.Fatalf("Failed to process INCREMENT: %v", err)
	}
	if !triggered {
		t.Error("Expected INCREMENT to trigger transition")
	}

	// Verify context was modified
	context := machine.GetContext()
	if count, ok := context.Fields["count"]; ok {
		if numValue, ok := count.GetKind().(*structpb.Value_NumberValue); ok {
			if numValue.NumberValue != 1 {
				t.Errorf("Expected count to be 1, got %f", numValue.NumberValue)
			}
		} else {
			t.Error("Count field is not a number")
		}
	} else {
		t.Error("Count field not found in context")
	}

	// Test multiple increments
	for i := 0; i < 3; i++ {
		machine.Step("INCREMENT")
	}

	context = machine.GetContext()
	if count, ok := context.Fields["count"]; ok {
		if numValue, ok := count.GetKind().(*structpb.Value_NumberValue); ok {
			if numValue.NumberValue != 4 {
				t.Errorf("Expected count to be 4, got %f", numValue.NumberValue)
			}
		}
	}

	// Test reset action
	triggered, err = machine.Step("RESET")
	if err != nil {
		t.Fatalf("Failed to process RESET: %v", err)
	}
	if !triggered {
		t.Error("Expected RESET to trigger transition")
	}

	context = machine.GetContext()
	if count, ok := context.Fields["count"]; ok {
		if numValue, ok := count.GetKind().(*structpb.Value_NumberValue); ok {
			if numValue.NumberValue != 0 {
				t.Errorf("Expected count to be 0 after reset, got %f", numValue.NumberValue)
			}
		}
	}

	// Stop and verify final state
	if err := machine.Stop(); err != nil {
		t.Fatalf("Failed to stop machine: %v", err)
	}

	// Verify step history captured all operations
	history := machine.GetStepHistory()
	if len(history) < 6 { // start + 4 increments + 1 reset + stop
		t.Errorf("Expected at least 6 steps in history, got %d", len(history))
	}
}

// TestMachineLifecycleErrorRecovery tests error handling and recovery during machine lifecycle
func TestMachineLifecycleErrorRecovery(t *testing.T) {
	// Create a statechart with guard conditions that can fail
	statechart := &sc.Statechart{
		RootState: &sc.State{
			Label: "__root__",
			Type:  sc.StateTypeNormal,
			Children: []*sc.State{
				{
					Label:     "Start",
					Type:      sc.StateTypeBasic,
					IsInitial: true,
				},
				{
					Label: "Guarded",
					Type:  sc.StateTypeBasic,
				},
			},
		},
		Transitions: []*sc.Transition{
			{
				Label: "ConditionalTransition",
				From:  []string{"Start"},
				To:    []string{"Guarded"},
				Event: "CONDITIONAL_EVENT",
				Guard: &sc.Guard{Expression: "context.count >= 5"},
			},
		},
		Events: []*sc.Event{
			{Label: "CONDITIONAL_EVENT"},
		},
	}

	// Test with context that should fail guard
	wrapper := semantics.NewStatechart(statechart)
	machine, err := semantics.NewMachine(wrapper, "guard-test", &structpb.Struct{
		Fields: map[string]*structpb.Value{
			"count": structpb.NewNumberValue(3), // Less than 5
		},
	})
	if err != nil {
		t.Fatalf("Failed to create machine: %v", err)
	}

	if err := machine.Start(); err != nil {
		t.Fatalf("Failed to start machine: %v", err)
	}

	// This should not trigger transition due to guard
	triggered, err := machine.Step("CONDITIONAL_EVENT")
	if err != nil {
		t.Fatalf("Failed to process event: %v", err)
	}
	if triggered {
		t.Error("Expected guard to prevent transition")
	}

	// Verify still in Start state
	config := machine.GetCurrentConfiguration()
	hasStart := false
	for _, state := range config.States {
		if state.Label == "Start" {
			hasStart = true
		}
	}
	if !hasStart {
		t.Errorf("Expected to remain in Start state, got %v", config.States)
	}

	// Clear any accumulated errors
	machine.ClearErrors()
	errors := machine.GetErrors()
	if len(errors) != 0 {
		t.Errorf("Expected no errors after clear, got %d", len(errors))
	}

	if err := machine.Stop(); err != nil {
		t.Fatalf("Failed to stop machine: %v", err)
	}

	// Test recovery after reset
	if err := machine.Reset(); err != nil {
		t.Fatalf("Failed to reset machine: %v", err)
	}

	// Verify machine is back to initial state
	if !machine.IsStopped() {
		t.Error("Machine should be stopped after reset")
	}

	errors = machine.GetErrors()
	if len(errors) != 0 {
		t.Errorf("Expected no errors after reset, got %d", len(errors))
	}
}
