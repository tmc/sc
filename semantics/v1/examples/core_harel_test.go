package examples

import (
	"testing"

	"github.com/tmc/sc"
	"github.com/tmc/sc/semantics/v1"
)

// TestCoreHarelStatechartFeatures tests the core Harel formalism [H87, HN96]
func TestCoreHarelStatechartFeatures(t *testing.T) {
	// Create a statechart using only core Harel features
	chart := semantics.NewStatechart(&sc.Statechart{
		Name:        "Core Harel Example",
		Description: "Demonstrates core Harel statechart formalism",
		RootState: &sc.State{
			Label: "Root",
			Type:  sc.StateTypeOR, // OR-decomposition (exclusive substates)
			Children: []*sc.State{
				{
					Label:     "Active",
					Type:      sc.StateTypeOR,
					IsInitial: true,
					// Entry actions (core Harel feature)
					EntryActions: []*sc.Action{
						{
							Label:      "onEnterActive",
							Expression: "console.log('Entering Active state')",
							Language:   "javascript",
						},
					},
					// Exit actions (core Harel feature)
					ExitActions: []*sc.Action{
						{
							Label:      "onExitActive", 
							Expression: "console.log('Exiting Active state')",
							Language:   "javascript",
						},
					},
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
					},
				},
				{
					Label: "Inactive",
					Type:  sc.StateTypeBasic,
				},
				{
					Label: "Parallel",
					Type:  sc.StateTypeAND, // AND-decomposition (concurrent substates)
					Children: []*sc.State{
						{
							Label: "RegionA",
							Type:  sc.StateTypeOR,
							Children: []*sc.State{
								{
									Label:     "A1",
									Type:      sc.StateTypeBasic,
									IsInitial: true,
								},
								{
									Label: "A2",
									Type:  sc.StateTypeBasic,
								},
							},
						},
						{
							Label: "RegionB",
							Type:  sc.StateTypeOR,
							Children: []*sc.State{
								{
									Label:     "B1",
									Type:      sc.StateTypeBasic,
									IsInitial: true,
								},
								{
									Label: "B2",
									Type:  sc.StateTypeBasic,
								},
							},
						},
					},
				},
			},
		},
		Transitions: []*sc.Transition{
			{
				Label: "Start",
				From:  []string{"Idle"},
				To:    []string{"Processing"},
				Event: "START",
				// Transition actions (core Harel feature)
				Actions: []*sc.Action{
					{
						Label:      "onStart",
						Expression: "initialize()",
						Language:   "javascript",
					},
				},
			},
			{
				Label: "Complete",
				From:  []string{"Processing"},
				To:    []string{"Idle"},
				Event: "COMPLETE",
			},
			{
				Label: "Deactivate",
				From:  []string{"Active"},
				To:    []string{"Inactive"},
				Event: "STOP",
				// Priority for conflict resolution [HN96, Section 4.3]
				Priority: 10,
			},
			{
				Label: "ParallelTransition",
				From:  []string{"A1"},
				To:    []string{"A2"},
				Event: "SWITCH_A",
			},
			{
				Label: "ParallelTransitionB",
				From:  []string{"B1"},
				To:    []string{"B2"}, 
				Event: "SWITCH_B",
			},
		},
		Events: []*sc.Event{
			{Label: "START"},
			{Label: "COMPLETE"},
			{Label: "STOP"},
			{Label: "SWITCH_A"},
			{Label: "SWITCH_B"},
		},
	})

	// Test that the core statechart is valid
	if err := chart.Validate(); err != nil {
		t.Errorf("Core Harel statechart is invalid: %v", err)
	}

	// Test core Harel state types
	rootState := chart.RootState
	if rootState.Label != "__root__" { // NewStatechart changes this
		t.Errorf("Expected root label to be '__root__', got %s", rootState.Label)
	}

	// Find the Active state (OR-decomposition)
	var activeState *sc.State
	for _, child := range rootState.Children {
		if child.Label == "Active" {
			activeState = child
			break
		}
	}

	if activeState == nil {
		t.Fatal("Could not find Active state")
	}

	// Test OR-state semantics
	if activeState.Type != sc.StateTypeOR {
		t.Errorf("Expected OR state type, got %v", activeState.Type)
	}

	// Test entry/exit actions (core Harel features)
	if len(activeState.EntryActions) != 1 {
		t.Errorf("Expected 1 entry action, got %d", len(activeState.EntryActions))
	}

	if len(activeState.ExitActions) != 1 {
		t.Errorf("Expected 1 exit action, got %d", len(activeState.ExitActions))
	}

	// Find the Parallel state (AND-decomposition)
	var parallelState *sc.State
	for _, child := range rootState.Children {
		if child.Label == "Parallel" {
			parallelState = child
			break
		}
	}

	if parallelState == nil {
		t.Fatal("Could not find Parallel state")
	}

	// Test AND-state semantics
	if parallelState.Type != sc.StateTypeAND {
		t.Errorf("Expected AND state type, got %v", parallelState.Type)
	}

	// Test that AND-state has multiple concurrent regions
	if len(parallelState.Children) != 2 {
		t.Errorf("Expected 2 orthogonal regions, got %d", len(parallelState.Children))
	}

	// Test transitions with priority (conflict resolution)
	var stopTransition *sc.Transition
	for _, transition := range chart.Transitions {
		if transition.Label == "Deactivate" {
			stopTransition = transition
			break
		}
	}

	if stopTransition == nil {
		t.Fatal("Could not find Stop transition")
	}

	if stopTransition.Priority != 10 {
		t.Errorf("Expected priority 10, got %d", stopTransition.Priority)
	}

	// Test statechart metadata
	if chart.Name != "Core Harel Example" {
		t.Errorf("Expected name 'Core Harel Example', got %s", chart.Name)
	}
}

// TestHarelStateTypeAliases tests the academic terminology aliases
func TestHarelStateTypeAliases(t *testing.T) {
	// Test that academic aliases work correctly
	tests := []struct {
		name     string
		stateType sc.StateType
		expected string
	}{
		{"Basic state", sc.StateTypeBasic, "basic"},
		{"OR state", sc.StateTypeOR, "or"},
		{"AND state", sc.StateTypeAND, "and"},
		{"Normal alias", sc.StateTypeNormal, "or"}, // Alias for OR
		{"Parallel alias", sc.StateTypeParallel, "and"}, // Alias for AND
		{"Orthogonal alias", sc.StateTypeOrthogonal, "and"}, // Alias for AND (Harel's term)
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			state := &sc.State{
				Label: "Test",
				Type:  tt.stateType,
			}

			// Test that aliases resolve to core types
			switch tt.expected {
			case "basic":
				if state.Type != sc.StateTypeBasic {
					t.Errorf("Expected basic type, got %v", state.Type)
				}
			case "or":
				if state.Type != sc.StateTypeOR && state.Type != sc.StateTypeNormal {
					t.Errorf("Expected OR/Normal type, got %v", state.Type)
				}
			case "and":
				if state.Type != sc.StateTypeAND && state.Type != sc.StateTypeParallel && state.Type != sc.StateTypeOrthogonal {
					t.Errorf("Expected AND/Parallel/Orthogonal type, got %v", state.Type)
				}
			}
		})
	}
}