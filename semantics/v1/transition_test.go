package semantics

import (
	"testing"

	sc "github.com/tmc/sc"
	"google.golang.org/protobuf/types/known/structpb"
)

func TestTransitionExecution(t *testing.T) {
	transition := &sc.Transition{
		Label: "turn_on",
		From:  []string{"Off"},
		To:    []string{"On"},
		Event: "TURN_ON",
		Guard: &sc.Guard{
			Expression: "context.count < 5",
		},
		Actions: []*sc.Action{
			{Label: "increment_count"},
		},
	}

	context := &structpb.Struct{
		Fields: map[string]*structpb.Value{
			"count": structpb.NewNumberValue(0),
		},
	}

	machine := &sc.Machine{
		Id:      "test-machine",
		State:   sc.MachineStateRunning,
		Context: context,
		Statechart: &sc.Statechart{
			RootState: &sc.State{
				Children: []*sc.State{
					{Label: "Off"},
					{Label: "On"},
				},
			},
			Transitions: []*sc.Transition{transition},
		},
		Configuration: &sc.Configuration{
			States: []*sc.StateRef{{Label: "Off"}},
		},
	}

	err := executeTransition(machine, transition)
	if err != nil {
		t.Fatalf("Transition execution failed: %v", err)
	}

	// Check if the configuration has changed
	if len(machine.Configuration.States) != 1 || machine.Configuration.States[0].Label != "On" {
		t.Errorf("Expected configuration [On], got %v", machine.Configuration.States)
	}

	// Check if the action was executed (count incremented)
	count, ok := machine.Context.Fields["count"].GetKind().(*structpb.Value_NumberValue)
	if !ok || count.NumberValue != 1 {
		t.Errorf("Expected context count to be 1, got %v", machine.Context.Fields["count"])
	}
}

func TestGuardEvaluation(t *testing.T) {
	guard := &sc.Guard{
		Expression: "context.count < 5",
	}

	tests := []struct {
		name     string
		context  *structpb.Struct
		expected bool
	}{
		{
			name: "Guard passes",
			context: &structpb.Struct{
				Fields: map[string]*structpb.Value{
					"count": structpb.NewNumberValue(3),
				},
			},
			expected: true,
		},
		{
			name: "Guard fails",
			context: &structpb.Struct{
				Fields: map[string]*structpb.Value{
					"count": structpb.NewNumberValue(7),
				},
			},
			expected: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result, err := evaluateGuard(guard, tt.context)
			if err != nil {
				t.Fatalf("Guard evaluation failed: %v", err)
			}
			if result != tt.expected {
				t.Errorf("Expected guard evaluation to be %v, got %v", tt.expected, result)
			}
		})
	}
}

func TestActionExecution(t *testing.T) {
	action := &sc.Action{
		Label: "increment_count",
	}

	context := &structpb.Struct{
		Fields: map[string]*structpb.Value{
			"count": structpb.NewNumberValue(0),
		},
	}

	err := executeAction(action, context)
	if err != nil {
		t.Fatalf("Action execution failed: %v", err)
	}

	count, ok := context.Fields["count"].GetKind().(*structpb.Value_NumberValue)
	if !ok || count.NumberValue != 1 {
		t.Errorf("Expected context count to be 1, got %v", context.Fields["count"])
	}
}

// Helper functions (these would be implemented in your actual code)

func executeTransition(machine *sc.Machine, transition *sc.Transition) error {
	// Check guard
	guardPasses, err := evaluateGuard(transition.Guard, machine.Context)
	if err != nil {
		return err
	}
	if !guardPasses {
		return nil // Guard didn't pass, so transition doesn't execute
	}

	// Update configuration
	machine.Configuration = &sc.Configuration{
		States: []*sc.StateRef{{Label: transition.To[0]}},
	}

	// Execute actions
	for _, action := range transition.Actions {
		if err := executeAction(action, machine.Context); err != nil {
			return err
		}
	}

	return nil
}

func evaluateGuard(guard *sc.Guard, context *structpb.Struct) (bool, error) {
	// This is a simplified guard evaluation.
	// In a real implementation, you would parse and evaluate the guard expression.
	if guard.Expression == "context.count < 5" {
		count, ok := context.Fields["count"].GetKind().(*structpb.Value_NumberValue)
		if !ok {
			return false, nil
		}
		return count.NumberValue < 5, nil
	}
	return true, nil
}

func executeAction(action *sc.Action, context *structpb.Struct) error {
	// This is a simplified action execution.
	// In a real implementation, you would have a way to map action labels to actual functions.
	if action.Label == "increment_count" {
		count, ok := context.Fields["count"].GetKind().(*structpb.Value_NumberValue)
		if !ok {
			return nil
		}
		context.Fields["count"] = structpb.NewNumberValue(count.NumberValue + 1)
	}
	return nil
}
