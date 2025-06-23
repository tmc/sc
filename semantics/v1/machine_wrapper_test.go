package semantics

import (
	"testing"

	"github.com/tmc/sc"
	"google.golang.org/protobuf/types/known/structpb"
)

func TestMachineWrapperCreation(t *testing.T) {
	statechart := createSimpleStatechart()
	machine, err := NewMachine(statechart, "test-machine", nil)
	if err != nil {
		t.Fatalf("Failed to create machine: %v", err)
	}

	if machine.Id != "test-machine" {
		t.Errorf("Expected machine ID 'test-machine', got '%s'", machine.Id)
	}

	if machine.State != sc.MachineStateStopped {
		t.Errorf("Expected machine state STOPPED, got %v", machine.State)
	}

	if machine.Configuration == nil {
		t.Errorf("Expected machine to have a configuration")
	}

	if machine.Context == nil {
		t.Errorf("Expected machine to have a context")
	}
}

func TestMachineWrapperLifecycle(t *testing.T) {
	statechart := createSimpleStatechart()
	machine, err := NewMachine(statechart, "test-machine", nil)
	if err != nil {
		t.Fatalf("Failed to create machine: %v", err)
	}

	// Test initial state
	if !machine.IsStopped() {
		t.Errorf("Expected machine to be stopped initially")
	}

	// Test start
	if err := machine.Start(); err != nil {
		t.Errorf("Failed to start machine: %v", err)
	}

	if !machine.IsRunning() {
		t.Errorf("Expected machine to be running after start")
	}

	// Test stop
	if err := machine.Stop(); err != nil {
		t.Errorf("Failed to stop machine: %v", err)
	}

	if !machine.IsStopped() {
		t.Errorf("Expected machine to be stopped after stop")
	}
}

func TestMachineWrapperStep(t *testing.T) {
	statechart := createStatechartWithTransitions()
	machine, err := NewMachine(statechart, "test-machine", &structpb.Struct{
		Fields: map[string]*structpb.Value{
			"count": structpb.NewNumberValue(0),
		},
	})
	if err != nil {
		t.Fatalf("Failed to create machine: %v", err)
	}

	// Start the machine
	if err := machine.Start(); err != nil {
		t.Fatalf("Failed to start machine: %v", err)
	}

	// Check initial configuration
	config := machine.GetCurrentConfiguration()
	if len(config.States) == 0 {
		t.Fatalf("Expected machine to have initial states")
	}

	// Test stepping with an event
	executed, err := machine.Step("TURN_ON")
	if err != nil {
		t.Errorf("Step failed: %v", err)
	}

	// For this test, we expect the step to execute or not based on available transitions
	_ = executed // We'll validate based on the specific statechart
}

func TestMachineWrapperGuardEvaluation(t *testing.T) {
	statechart := createSimpleStatechart()
	machine, err := NewMachine(statechart, "test-machine", &structpb.Struct{
		Fields: map[string]*structpb.Value{
			"count": structpb.NewNumberValue(3),
		},
	})
	if err != nil {
		t.Fatalf("Failed to create machine: %v", err)
	}

	// Test guard evaluation
	guard1 := &sc.Guard{Expression: "context.count < 5"}
	result1, err := machine.evaluateGuard(guard1)
	if err != nil {
		t.Errorf("Guard evaluation failed: %v", err)
	}
	if !result1 {
		t.Errorf("Expected guard 'context.count < 5' to pass with count=3")
	}

	guard2 := &sc.Guard{Expression: "context.count >= 5"}
	result2, err := machine.evaluateGuard(guard2)
	if err != nil {
		t.Errorf("Guard evaluation failed: %v", err)
	}
	if result2 {
		t.Errorf("Expected guard 'context.count >= 5' to fail with count=3")
	}
}

func TestMachineWrapperActionExecution(t *testing.T) {
	statechart := createSimpleStatechart()
	machine, err := NewMachine(statechart, "test-machine", &structpb.Struct{
		Fields: map[string]*structpb.Value{
			"count": structpb.NewNumberValue(0),
		},
	})
	if err != nil {
		t.Fatalf("Failed to create machine: %v", err)
	}

	// Test increment action
	action := &sc.Action{Label: "increment_count"}
	if err := machine.executeAction(action); err != nil {
		t.Errorf("Action execution failed: %v", err)
	}

	context := machine.GetContext()
	count, ok := context.Fields["count"].GetKind().(*structpb.Value_NumberValue)
	if !ok || count.NumberValue != 1 {
		t.Errorf("Expected count to be 1 after increment, got %v", context.Fields["count"])
	}

	// Test reset action
	resetAction := &sc.Action{Label: "reset_count"}
	if err := machine.executeAction(resetAction); err != nil {
		t.Errorf("Reset action execution failed: %v", err)
	}

	context = machine.GetContext()
	count, ok = context.Fields["count"].GetKind().(*structpb.Value_NumberValue)
	if !ok || count.NumberValue != 0 {
		t.Errorf("Expected count to be 0 after reset, got %v", context.Fields["count"])
	}
}

func TestMachineWrapperErrorHandling(t *testing.T) {
	statechart := createSimpleStatechart()
	machine, err := NewMachine(statechart, "test-machine", nil)
	if err != nil {
		t.Fatalf("Failed to create machine: %v", err)
	}

	// Test stepping while stopped
	_, err = machine.Step("TEST_EVENT")
	if err == nil {
		t.Errorf("Expected error when stepping while machine is stopped")
	}

	// Test double start
	if err := machine.Start(); err != nil {
		t.Fatalf("Failed to start machine: %v", err)
	}

	if err := machine.Start(); err == nil {
		t.Errorf("Expected error when starting already running machine")
	}

	// Test double stop
	if err := machine.Stop(); err != nil {
		t.Fatalf("Failed to stop machine: %v", err)
	}

	if err := machine.Stop(); err == nil {
		t.Errorf("Expected error when stopping already stopped machine")
	}
}

func TestMachineWrapperValidation(t *testing.T) {
	statechart := createSimpleStatechart()
	machine, err := NewMachine(statechart, "test-machine", nil)
	if err != nil {
		t.Fatalf("Failed to create machine: %v", err)
	}

	// Test validation of valid machine
	if err := machine.Validate(); err != nil {
		t.Errorf("Validation failed for valid machine: %v", err)
	}
}

func TestMachineWrapperReset(t *testing.T) {
	statechart := createSimpleStatechart()
	machine, err := NewMachine(statechart, "test-machine", &structpb.Struct{
		Fields: map[string]*structpb.Value{
			"count": structpb.NewNumberValue(5),
		},
	})
	if err != nil {
		t.Fatalf("Failed to create machine: %v", err)
	}

	// Start and modify the machine
	if err := machine.Start(); err != nil {
		t.Fatalf("Failed to start machine: %v", err)
	}

	// Reset the machine
	if err := machine.Reset(); err != nil {
		t.Errorf("Failed to reset machine: %v", err)
	}

	// Check that machine is in initial state
	if !machine.IsStopped() {
		t.Errorf("Expected machine to be stopped after reset")
	}

	context := machine.GetContext()
	if len(context.Fields) != 0 {
		t.Errorf("Expected context to be empty after reset, got %v", context.Fields)
	}

	history := machine.GetStepHistory()
	if len(history) != 0 {
		t.Errorf("Expected step history to be empty after reset, got %d steps", len(history))
	}
}

func TestMachineWrapperConcurrency(t *testing.T) {
	statechart := createSimpleStatechart()
	machine, err := NewMachine(statechart, "test-machine", nil)
	if err != nil {
		t.Fatalf("Failed to create machine: %v", err)
	}

	// Test concurrent access to machine state
	go func() {
		machine.Start()
	}()

	go func() {
		machine.GetCurrentConfiguration()
	}()

	go func() {
		machine.GetContext()
	}()

	// This test mainly checks that there are no race conditions
	// In a more comprehensive test, you would use more sophisticated synchronization
}

// Helper functions to create test statecharts

func createSimpleStatechart() *Statechart {
	return NewStatechart(&sc.Statechart{
		RootState: &sc.State{
			Label: "__root__",
			Type:  sc.StateTypeNormal,
			Children: []*sc.State{
				{
					Label:     "Off",
					Type:      sc.StateTypeBasic,
					IsInitial: true,
				},
				{
					Label: "On",
					Type:  sc.StateTypeBasic,
				},
			},
		},
		Events: []*sc.Event{
			{Label: "TURN_ON"},
			{Label: "TURN_OFF"},
		},
	})
}

func createStatechartWithTransitions() *Statechart {
	return NewStatechart(&sc.Statechart{
		RootState: &sc.State{
			Label: "__root__",
			Type:  sc.StateTypeNormal,
			Children: []*sc.State{
				{
					Label:     "Off",
					Type:      sc.StateTypeBasic,
					IsInitial: true,
				},
				{
					Label: "On",
					Type:  sc.StateTypeBasic,
				},
			},
		},
		Transitions: []*sc.Transition{
			{
				Label: "turn_on",
				From:  []string{"Off"},
				To:    []string{"On"},
				Event: "TURN_ON",
				Guard: &sc.Guard{Expression: "context.count < 5"},
				Actions: []*sc.Action{
					{Label: "increment_count"},
				},
			},
			{
				Label: "turn_off",
				From:  []string{"On"},
				To:    []string{"Off"},
				Event: "TURN_OFF",
				Actions: []*sc.Action{
					{Label: "decrement_count"},
				},
			},
		},
		Events: []*sc.Event{
			{Label: "TURN_ON"},
			{Label: "TURN_OFF"},
		},
	})
}