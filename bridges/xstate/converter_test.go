package xstate

import (
	"testing"

	"github.com/tmc/sc"
)

func TestConverter_Import(t *testing.T) {
	converter := NewConverter()

	// Test simple XState machine
	machine := Machine{
		ID:      "light",
		Initial: "green",
		States: map[string]*State{
			"green": {
				Type: "atomic",
				On: map[string]*Transition{
					"TIMER": {Target: "yellow"},
				},
			},
			"yellow": {
				Type: "atomic",
				On: map[string]*Transition{
					"TIMER": {Target: "red"},
				},
			},
			"red": {
				Type: "atomic",
				On: map[string]*Transition{
					"TIMER": {Target: "green"},
				},
			},
		},
	}

	statechart, err := converter.Import(machine)
	if err != nil {
		t.Fatalf("Import failed: %v", err)
	}

	if statechart.RootState == nil {
		t.Fatal("Expected root state")
	}

	if len(statechart.RootState.Children) != 3 {
		t.Errorf("Expected 3 states, got %d", len(statechart.RootState.Children))
	}

	if len(statechart.Transitions) != 3 {
		t.Errorf("Expected 3 transitions, got %d", len(statechart.Transitions))
	}

	if len(statechart.Events) != 1 {
		t.Errorf("Expected 1 event, got %d", len(statechart.Events))
	}
}

func TestConverter_Export(t *testing.T) {
	converter := NewConverter()

	// Create simple Harel statechart
	statechart := &sc.Statechart{
		RootState: &sc.State{
			Label: "__root__",
			Type:  sc.StateTypeNormal,
			Children: []*sc.State{
				{
					Label:     "green",
					Type:      sc.StateTypeBasic,
					IsInitial: true,
				},
				{
					Label: "yellow",
					Type:  sc.StateTypeBasic,
				},
			},
		},
		Transitions: []*sc.Transition{
			{
				Label: "green_to_yellow",
				From:  []string{"green"},
				To:    []string{"yellow"},
				Event: "TIMER",
			},
		},
		Events: []*sc.Event{
			{Label: "TIMER"},
		},
	}

	machine, err := converter.Export(statechart)
	if err != nil {
		t.Fatalf("Export failed: %v", err)
	}

	if machine.ID == "" {
		t.Error("Expected machine ID")
	}

	if machine.Initial != "green" {
		t.Errorf("Expected initial state 'green', got '%s'", machine.Initial)
	}

	if len(machine.States) != 2 {
		t.Errorf("Expected 2 states, got %d", len(machine.States))
	}
}

func TestConverter_Validate(t *testing.T) {
	converter := NewConverter()

	// Valid machine
	validMachine := Machine{
		ID: "test",
		States: map[string]*State{
			"state1": {Type: "atomic"},
		},
	}

	err := converter.Validate(validMachine)
	if err != nil {
		t.Errorf("Valid machine should pass validation: %v", err)
	}

	// Invalid machine (no ID)
	invalidMachine := Machine{
		States: map[string]*State{
			"state1": {Type: "atomic"},
		},
	}

	err = converter.Validate(invalidMachine)
	if err == nil {
		t.Error("Invalid machine should fail validation")
	}
}

func TestParseJSON(t *testing.T) {
	jsonStr := `{
		"id": "test",
		"initial": "idle",
		"states": {
			"idle": {
				"type": "atomic",
				"on": {
					"START": {
						"target": "running"
					}
				}
			},
			"running": {
				"type": "atomic"
			}
		}
	}`

	machine, err := ParseJSON(jsonStr)
	if err != nil {
		t.Fatalf("ParseJSON failed: %v", err)
	}

	if machine.ID != "test" {
		t.Errorf("Expected ID 'test', got '%s'", machine.ID)
	}

	if machine.Initial != "idle" {
		t.Errorf("Expected initial 'idle', got '%s'", machine.Initial)
	}
}

func TestToJSON(t *testing.T) {
	machine := Machine{
		ID:      "test",
		Initial: "idle",
		States: map[string]*State{
			"idle": {
				Type: "atomic",
			},
		},
	}

	jsonStr, err := ToJSON(machine)
	if err != nil {
		t.Fatalf("ToJSON failed: %v", err)
	}

	if jsonStr == "" {
		t.Error("Expected non-empty JSON string")
	}

	// Verify round-trip
	parsedMachine, err := ParseJSON(jsonStr)
	if err != nil {
		t.Fatalf("Round-trip failed: %v", err)
	}

	if parsedMachine.ID != machine.ID {
		t.Errorf("Round-trip ID mismatch: expected '%s', got '%s'", machine.ID, parsedMachine.ID)
	}
}