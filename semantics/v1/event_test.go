package semantics

import (
	"testing"

	sc "github.com/tmc/sc"
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
		expectedCount    int
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
			expectedCount:    1,
			expectTransition: true,
		},
		{
			name:             "Unhandled Event",
			event:            "UNKNOWN_EVENT",
			expectedState:    "Off",
			expectedCount:    1,
			expectTransition: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			transitioned, err := handleEvent(machine, tt.event)
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
			if !ok || count.NumberValue != float64(tt.expectedCount) {
				t.Errorf("Expected count to be %d, got %v", tt.expectedCount, machine.Context.Fields["count"])
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
	for _, transition := range machine.Statechart.Transitions {
		if transition.Event == event && contains(transition.From, machine.Configuration.States[0].Label) {
			// Execute transition
			machine.Configuration.States[0].Label = transition.To[0]

			// Increment count (simplified action execution)
			if count, ok := machine.Context.Fields["count"].GetKind().(*structpb.Value_NumberValue); ok {
				machine.Context.Fields["count"] = structpb.NewNumberValue(count.NumberValue + 1)
			}

			return true, nil
		}
	}
	return false, nil
}

// Helper function to check if a slice contains a string
func contains(slice []string, str string) bool {
	for _, v := range slice {
		if v == str {
			return true
		}
	}
	return false
}
