package semantics

import (
	"testing"

	"github.com/tmc/sc"
	"google.golang.org/protobuf/types/known/structpb"
)

func TestMachineIntegration(t *testing.T) {
	// Create a simple statechart for testing
	statechart := &sc.Statechart{
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
				Label: "PowerOn",
				From:  []string{"Off"},
				To:    []string{"On"},
				Event: "POWER_ON",
			},
			{
				Label: "PowerOff",
				From:  []string{"On"},
				To:    []string{"Off"},
				Event: "POWER_OFF",
			},
		},
		Events: []*sc.Event{
			{Label: "POWER_ON"},
			{Label: "POWER_OFF"},
		},
	}

	// Create the statechart wrapper
	wrapper := NewStatechart(statechart)

	// Create a machine
	machine, err := NewMachine(wrapper, "test-machine", nil)
	if err != nil {
		t.Fatalf("Failed to create machine: %v", err)
	}

	// Test initial state
	if machine.IsStopped() != true {
		t.Errorf("Expected machine to be stopped initially")
	}

	// Start the machine
	if err := machine.Start(); err != nil {
		t.Fatalf("Failed to start machine: %v", err)
	}

	if !machine.IsRunning() {
		t.Errorf("Expected machine to be running after start")
	}

	// Check initial configuration
	config := machine.GetCurrentConfiguration()
	if len(config.States) < 1 {
		t.Fatalf("Expected at least 1 state in initial configuration, got %d", len(config.States))
	}

	// Send POWER_ON event
	triggered, err := machine.Step("POWER_ON")
	if err != nil {
		t.Fatalf("Failed to process POWER_ON event: %v", err)
	}

	if !triggered {
		t.Errorf("Expected POWER_ON event to trigger a transition")
	}

	// Send POWER_OFF event
	triggered, err = machine.Step("POWER_OFF")
	if err != nil {
		t.Fatalf("Failed to process POWER_OFF event: %v", err)
	}

	if !triggered {
		t.Errorf("Expected POWER_OFF event to trigger a transition")
	}

	// Send unknown event
	triggered, err = machine.Step("UNKNOWN")
	if err != nil {
		t.Fatalf("Failed to process UNKNOWN event: %v", err)
	}

	if triggered {
		t.Errorf("Expected UNKNOWN event to not trigger any transition")
	}

	// Stop the machine
	if err := machine.Stop(); err != nil {
		t.Fatalf("Failed to stop machine: %v", err)
	}

	if !machine.IsStopped() {
		t.Errorf("Expected machine to be stopped after stop")
	}

	// Verify we have step history
	history := machine.GetStepHistory()
	if len(history) < 3 { // START, POWER_ON, POWER_OFF, STOP
		t.Errorf("Expected at least 3 steps in history, got %d", len(history))
	}
}

func TestMachineWithContext(t *testing.T) {
	// Create a statechart with actions that modify context
	statechart := &sc.Statechart{
		RootState: &sc.State{
			Label: "__root__",
			Type:  sc.StateTypeNormal,
			Children: []*sc.State{
				{
					Label:     "CounterState",
					Type:      sc.StateTypeBasic,
					IsInitial: true,
				},
			},
		},
		Transitions: []*sc.Transition{
			{
				Label: "Increment",
				From:  []string{"CounterState"},
				To:    []string{"CounterState"},
				Event: "INCREMENT",
				Actions: []*sc.Action{
					{Label: "increment_count"},
				},
			},
			{
				Label: "Decrement",
				From:  []string{"CounterState"},
				To:    []string{"CounterState"},
				Event: "DECREMENT",
				Actions: []*sc.Action{
					{Label: "decrement_count"},
				},
			},
		},
		Events: []*sc.Event{
			{Label: "INCREMENT"},
			{Label: "DECREMENT"},
		},
	}

	// Create initial context
	initialContext := &structpb.Struct{
		Fields: map[string]*structpb.Value{
			"count": structpb.NewNumberValue(0),
		},
	}

	// Create the statechart wrapper and machine
	wrapper := NewStatechart(statechart)
	machine, err := NewMachine(wrapper, "counter-machine", initialContext)
	if err != nil {
		t.Fatalf("Failed to create machine: %v", err)
	}

	// Start the machine
	if err := machine.Start(); err != nil {
		t.Fatalf("Failed to start machine: %v", err)
	}

	// Check initial count
	context := machine.GetContext()
	count, ok := context.Fields["count"].GetKind().(*structpb.Value_NumberValue)
	if !ok || count.NumberValue != 0 {
		t.Errorf("Expected initial count to be 0, got %v", context.Fields["count"])
	}

	// Increment count
	_, err = machine.Step("INCREMENT")
	if err != nil {
		t.Fatalf("Failed to process INCREMENT event: %v", err)
	}

	// Check count after increment
	context = machine.GetContext()
	count, ok = context.Fields["count"].GetKind().(*structpb.Value_NumberValue)
	if !ok || count.NumberValue != 1 {
		t.Errorf("Expected count to be 1 after increment, got %v", context.Fields["count"])
	}

	// Decrement count
	_, err = machine.Step("DECREMENT")
	if err != nil {
		t.Fatalf("Failed to process DECREMENT event: %v", err)
	}

	// Check count after decrement
	context = machine.GetContext()
	count, ok = context.Fields["count"].GetKind().(*structpb.Value_NumberValue)
	if !ok || count.NumberValue != 0 {
		t.Errorf("Expected count to be 0 after decrement, got %v", context.Fields["count"])
	}
}

func TestMachineReset(t *testing.T) {
	// Create a simple statechart
	statechart := &sc.Statechart{
		RootState: &sc.State{
			Label: "__root__",
			Type:  sc.StateTypeNormal,
			Children: []*sc.State{
				{
					Label:     "Initial",
					Type:      sc.StateTypeBasic,
					IsInitial: true,
				},
				{
					Label: "Other",
					Type:  sc.StateTypeBasic,
				},
			},
		},
		Transitions: []*sc.Transition{
			{
				Label: "Transition",
				From:  []string{"Initial"},
				To:    []string{"Other"},
				Event: "MOVE",
			},
		},
		Events: []*sc.Event{
			{Label: "MOVE"},
		},
	}

	// Create machine and perform some operations
	wrapper := NewStatechart(statechart)
	machine, err := NewMachine(wrapper, "reset-test", nil)
	if err != nil {
		t.Fatalf("Failed to create machine: %v", err)
	}

	// Start and perform some operations
	machine.Start()
	machine.Step("MOVE")

	// Verify we have some history
	history := machine.GetStepHistory()
	if len(history) < 2 {
		t.Errorf("Expected at least 2 steps in history before reset, got %d", len(history))
	}

	// Reset the machine
	if err := machine.Reset(); err != nil {
		t.Fatalf("Failed to reset machine: %v", err)
	}

	// Verify reset state
	if !machine.IsStopped() {
		t.Errorf("Expected machine to be stopped after reset")
	}

	history = machine.GetStepHistory()
	if len(history) != 0 {
		t.Errorf("Expected empty history after reset, got %d steps", len(history))
	}

	errors := machine.GetErrors()
	if len(errors) != 0 {
		t.Errorf("Expected no errors after reset, got %d errors", len(errors))
	}
}

func TestMachineErrorHandling(t *testing.T) {
	// Test various error conditions
	
	t.Run("nil statechart", func(t *testing.T) {
		_, err := NewMachine(nil, "test", nil)
		if err == nil {
			t.Error("Expected error when creating machine with nil statechart")
		}
	})

	t.Run("empty machine id", func(t *testing.T) {
		statechart := &sc.Statechart{
			RootState: &sc.State{
				Label: "__root__",
				Type:  sc.StateTypeBasic,
			},
		}
		wrapper := NewStatechart(statechart)
		_, err := NewMachine(wrapper, "", nil)
		if err == nil {
			t.Error("Expected error when creating machine with empty ID")
		}
	})

	t.Run("step when stopped", func(t *testing.T) {
		statechart := &sc.Statechart{
			RootState: &sc.State{
				Label: "__root__",
				Type:  sc.StateTypeBasic,
			},
		}
		wrapper := NewStatechart(statechart)
		machine, _ := NewMachine(wrapper, "test", nil)
		
		// Don't start the machine, try to step
		_, err := machine.Step("EVENT")
		if err == nil {
			t.Error("Expected error when stepping a stopped machine")
		}
	})

	t.Run("double start", func(t *testing.T) {
		statechart := &sc.Statechart{
			RootState: &sc.State{
				Label: "__root__",
				Type:  sc.StateTypeBasic,
			},
		}
		wrapper := NewStatechart(statechart)
		machine, _ := NewMachine(wrapper, "test", nil)
		
		machine.Start()
		err := machine.Start()
		if err == nil {
			t.Error("Expected error when starting an already running machine")
		}
	})

	t.Run("double stop", func(t *testing.T) {
		statechart := &sc.Statechart{
			RootState: &sc.State{
				Label: "__root__",
				Type:  sc.StateTypeBasic,
			},
		}
		wrapper := NewStatechart(statechart)
		machine, _ := NewMachine(wrapper, "test", nil)
		
		// Don't start, just try to stop
		err := machine.Stop()
		if err == nil {
			t.Error("Expected error when stopping an already stopped machine")
		}
	})
}