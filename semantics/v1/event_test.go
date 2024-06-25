package semantics

import (
	"fmt"
	"testing"

	"github.com/tmc/sc"
	"golang.org/x/exp/slices"
	"google.golang.org/protobuf/types/known/structpb"
)

func TestEventHandling(t *testing.T) {
	machine := &sc.Machine{
		Id:    "test-machine",
		State: sc.MachineStateRunning,
		Context: &structpb.Struct{
			Fields: map[string]*structpb.Value{
				"count": structpb.NewNumberValue(0),
			},
		},
		Statechart: &sc.Statechart{
			RootState: &sc.State{
				Children: []*sc.State{
					{Label: "Off"},
					{Label: "On"},
				},
			},
			Transitions: []*sc.Transition{
				{
					Label: "turn_on",
					From:  []string{"Off"},
					To:    []string{"On"},
					Event: "TURN_ON",
				},
				{
					Label: "turn_off",
					From:  []string{"On"},
					To:    []string{"Off"},
					Event: "TURN_OFF",
				},
			},
		},
		Configuration: &sc.Configuration{
			States: []*sc.StateRef{{Label: "Off"}},
		},
	}

	tests := []struct {
		name             string
		event            string
		expectedState    string
		expectedCount    float64
		expectTransition bool
	}{
		{
			name:             "Turn On",
			event:            "TURN_ON",
			expectedState:    "On",
			expectedCount:    1,
			expectTransition: true,
		},
		{
			name:             "Already On",
			event:            "TURN_ON",
			expectedState:    "On",
			expectedCount:    1,
			expectTransition: false,
		},
		{
			name:             "Turn Off",
			event:            "TURN_OFF",
			expectedState:    "Off",
			expectedCount:    2,
			expectTransition: true,
		},
		{
			name:             "Unhandled Event",
			event:            "UNKNOWN_EVENT",
			expectedState:    "Off",
			expectedCount:    2,
			expectTransition: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			transitioned, err := HandleEvent(machine, tt.event)
			if err != nil {
				t.Fatalf("Event handling failed: %v", err)
			}

			if transitioned != tt.expectTransition {
				t.Errorf("Expected transition: %v, got: %v", tt.expectTransition, transitioned)
			}

			if machine.Configuration.States[0].Label != tt.expectedState {
				t.Errorf("Expected state %s, got %s", tt.expectedState, machine.Configuration.States[0].Label)
			}

			count, ok := machine.Context.Fields["count"].GetKind().(*structpb.Value_NumberValue)
			if !ok || count.NumberValue != tt.expectedCount {
				t.Errorf("Expected count to be %f, got %v", tt.expectedCount, machine.Context.Fields["count"])
			}
		})
	}
}

func TestEventPriority(t *testing.T) {
	machine := &sc.Machine{
		Id:    "test-machine",
		State: sc.MachineStateRunning,
		Statechart: &sc.Statechart{
			RootState: &sc.State{
				Children: []*sc.State{
					{Label: "S1"},
					{Label: "S2"},
					{Label: "S3"},
				},
			},
			Transitions: []*sc.Transition{
				{
					Label: "t1",
					From:  []string{"S1"},
					To:    []string{"S2"},
					Event: "E",
				},
				{
					Label: "t2",
					From:  []string{"S1"},
					To:    []string{"S3"},
					Event: "E",
				},
			},
		},
		Configuration: &sc.Configuration{
			States: []*sc.StateRef{{Label: "S1"}},
		},
	}

	transitioned, err := handleEvent(machine, "E")
	if err != nil {
		t.Fatalf("Event handling failed: %v", err)
	}

	if !transitioned {
		t.Errorf("Expected a transition to occur")
	}

	if machine.Configuration.States[0].Label != "S2" {
		t.Errorf("Expected state S2 (first matching transition), got %s", machine.Configuration.States[0].Label)
	}
}

// Helper function (this would be implemented in your actual code)
func handleEvent(machine *sc.Machine, event string) (bool, error) {
	if machine == nil {
		return false, fmt.Errorf("machine is nil")
	}
	if machine.Statechart == nil {
		return false, fmt.Errorf("machine.Statechart is nil")
	}
	if machine.Statechart.Transitions == nil {
		return false, fmt.Errorf("machine.Statechart.Transitions is nil")
	}
	if machine.Configuration == nil {
		return false, fmt.Errorf("machine.Configuration is nil")
	}
	if len(machine.Configuration.States) == 0 {
		return false, fmt.Errorf("machine.Configuration.States is empty")
	}

	for _, transition := range machine.Statechart.Transitions {
		if transition.Event == event && slices.Contains(transition.From, machine.Configuration.States[0].Label) {
			// Execute transition
			machine.Configuration.States[0].Label = transition.To[0]

			// Increment count only for handled events
			if machine.Context != nil && machine.Context.Fields != nil {
				if countValue, exists := machine.Context.Fields["count"]; exists {
					if count, ok := countValue.GetKind().(*structpb.Value_NumberValue); ok {
						machine.Context.Fields["count"] = structpb.NewNumberValue(count.NumberValue + 1)
					}
				}
			}

			return true, nil
		}
	}
	return false, nil
}
