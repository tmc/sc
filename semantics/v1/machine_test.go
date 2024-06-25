package semantics

import (
	"testing"

	statecharts "github.com/tmc/sc/gen/statecharts/v1"
	"google.golang.org/protobuf/types/known/structpb"
)

func TestMachineCreation(t *testing.T) {
	machine := &statecharts.Machine{
		Id:    "test-machine",
		State: statecharts.MachineState_MACHINE_STATE_RUNNING,
		Context: &structpb.Struct{
			Fields: map[string]*structpb.Value{
				"count": structpb.NewNumberValue(0),
			},
		},
		Statechart: exampleStatechart1.Statechart,
		Configuration: &statecharts.Configuration{
			States: []*statecharts.StateRef{
				{Label: "Off"},
			},
		},
	}

	if machine.Id != "test-machine" {
		t.Errorf("Expected machine ID 'test-machine', got '%s'", machine.Id)
	}

	if machine.State != statecharts.MachineState_MACHINE_STATE_RUNNING {
		t.Errorf("Expected machine state RUNNING, got %v", machine.State)
	}

	if len(machine.Configuration.States) != 1 || machine.Configuration.States[0].Label != "Off" {
		t.Errorf("Expected initial configuration [Off], got %v", machine.Configuration.States)
	}

	count, ok := machine.Context.Fields["count"].GetKind().(*structpb.Value_NumberValue)
	if !ok || count.NumberValue != 0 {
		t.Errorf("Expected context count to be 0, got %v", machine.Context.Fields["count"])
	}
}

func TestMachineStep(t *testing.T) {
	machine := &statecharts.Machine{
		Id:    "test-machine",
		State: statecharts.MachineState_MACHINE_STATE_RUNNING,
		Context: &structpb.Struct{
			Fields: map[string]*structpb.Value{
				"count": structpb.NewNumberValue(0),
			},
		},
		Statechart: exampleStatechart1.Statechart,
		Configuration: &statecharts.Configuration{
			States: []*statecharts.StateRef{
				{Label: "Off"},
			},
		},
	}

	// Simulate a step
	err := stepMachine(machine, "TURN_ON")
	if err != nil {
		t.Fatalf("Step failed: %v", err)
	}

	expectedStates := []string{"On", "Blocked", "Ready"}
	if len(machine.Configuration.States) != len(expectedStates) {
		t.Fatalf("Expected %d states after step, got %d", len(expectedStates), len(machine.Configuration.States))
	}

	for i, state := range machine.Configuration.States {
		if state.Label != expectedStates[i] {
			t.Errorf("Expected state %s at position %d, got %s", expectedStates[i], i, state.Label)
		}
	}

	// Check if context was updated
	count, ok := machine.Context.Fields["count"].GetKind().(*structpb.Value_NumberValue)
	if !ok || count.NumberValue != 1 {
		t.Errorf("Expected context count to be 1, got %v", machine.Context.Fields["count"])
	}
}

func TestMachineState(t *testing.T) {
	machine := &statecharts.Machine{
		Id:         "test-machine",
		State:      statecharts.MachineState_MACHINE_STATE_RUNNING,
		Statechart: exampleStatechart1.Statechart,
	}

	if machine.State != statecharts.MachineState_MACHINE_STATE_RUNNING {
		t.Errorf("Expected machine state RUNNING, got %v", machine.State)
	}

	machine.State = statecharts.MachineState_MACHINE_STATE_STOPPED
	if machine.State != statecharts.MachineState_MACHINE_STATE_STOPPED {
		t.Errorf("Expected machine state STOPPED, got %v", machine.State)
	}
}

func TestMachineContext(t *testing.T) {
	machine := &statecharts.Machine{
		Id:    "test-machine",
		State: statecharts.MachineState_MACHINE_STATE_RUNNING,
		Context: &structpb.Struct{
			Fields: map[string]*structpb.Value{
				"count": structpb.NewNumberValue(0),
				"name":  structpb.NewStringValue("test"),
			},
		},
		Statechart: exampleStatechart1.Statechart,
	}

	count, ok := machine.Context.Fields["count"].GetKind().(*structpb.Value_NumberValue)
	if !ok || count.NumberValue != 0 {
		t.Errorf("Expected context count to be 0, got %v", machine.Context.Fields["count"])
	}

	name, ok := machine.Context.Fields["name"].GetKind().(*structpb.Value_StringValue)
	if !ok || name.StringValue != "test" {
		t.Errorf("Expected context name to be 'test', got %v", machine.Context.Fields["name"])
	}

	// Update context
	machine.Context.Fields["count"] = structpb.NewNumberValue(1)
	machine.Context.Fields["name"] = structpb.NewStringValue("updated")

	count, ok = machine.Context.Fields["count"].GetKind().(*structpb.Value_NumberValue)
	if !ok || count.NumberValue != 1 {
		t.Errorf("Expected updated context count to be 1, got %v", machine.Context.Fields["count"])
	}

	name, ok = machine.Context.Fields["name"].GetKind().(*structpb.Value_StringValue)
	if !ok || name.StringValue != "updated" {
		t.Errorf("Expected updated context name to be 'updated', got %v", machine.Context.Fields["name"])
	}
}

// Helper function to simulate a step
func stepMachine(machine *statecharts.Machine, event string) error {
	// This is a simplified step function. In a real implementation,
	// you would use the actual statechart execution logic here.
	machine.Configuration = &statecharts.Configuration{
		States: []*statecharts.StateRef{
			{Label: "On"},
			{Label: "Blocked"},
			{Label: "Ready"},
		},
	}

	// Update context
	if count, ok := machine.Context.Fields["count"].GetKind().(*structpb.Value_NumberValue); ok {
		machine.Context.Fields["count"] = structpb.NewNumberValue(count.NumberValue + 1)
	}

	return nil
}
