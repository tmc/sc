package bridges

import (
	"fmt"
	"reflect"
	"testing"

	"github.com/tmc/sc"
	"github.com/tmc/sc/bridges/scxml"
	"github.com/tmc/sc/bridges/xstate"
)

// RoundTripTest defines a test case for round-trip conversion testing.
type RoundTripTest struct {
	Name        string
	Description string
	Statechart  *sc.Statechart
	// Tolerance levels for different aspects
	AllowLabelChanges     bool // Allow state/transition label normalization
	AllowSemanticMapping  bool // Allow semantic differences between formats
	AllowStructuralChange bool // Allow minor structural reorganization
}

// SemanticEquivalence checks if two statecharts are semantically equivalent.
type SemanticEquivalence struct {
	StatesMatch      bool
	TransitionsMatch bool
	EventsMatch      bool
	StructureMatch   bool
	Details          []string
}

// TestHarelXStateHarelRoundTrip tests Harel → XState → Harel conversion.
func TestHarelXStateHarelRoundTrip(t *testing.T) {
	tests := []RoundTripTest{
		{
			Name:        "Simple State Machine",
			Description: "Basic three-state traffic light",
			Statechart:  createTrafficLightStatechart(),
		},
		{
			Name:        "Hierarchical State Machine", 
			Description: "Nested states with guards and actions",
			Statechart:  createHierarchicalStatechart(),
		},
		{
			Name:        "Parallel State Machine",
			Description: "Orthogonal regions with concurrent execution",
			Statechart:  createParallelStatechart(),
		},
	}

	for _, tt := range tests {
		t.Run(tt.Name, func(t *testing.T) {
			original := tt.Statechart

			// Harel → XState
			xstateConverter := xstate.NewConverter()
			xstateMachine, err := xstateConverter.Export(original)
			if err != nil {
				t.Fatalf("Harel → XState conversion failed: %v", err)
			}

			// XState → Harel
			restored, err := xstateConverter.Import(xstateMachine)
			if err != nil {
				t.Fatalf("XState → Harel conversion failed: %v", err)
			}

			// Validate semantic equivalence
			equiv := checkSemanticEquivalence(original, restored, tt)
			if !equiv.StatesMatch || !equiv.TransitionsMatch || !equiv.EventsMatch {
				t.Errorf("Round-trip failed for %s:", tt.Name)
				for _, detail := range equiv.Details {
					t.Errorf("  - %s", detail)
				}
			}
		})
	}
}

// TestHarelSCXMLHarelRoundTrip tests Harel → SCXML → Harel conversion.
func TestHarelSCXMLHarelRoundTrip(t *testing.T) {
	tests := []RoundTripTest{
		{
			Name:        "Simple State Machine",
			Description: "Basic traffic light with SCXML semantics",
			Statechart:  createTrafficLightStatechart(),
		},
		{
			Name:        "Compound State Machine",
			Description: "Hierarchical states with SCXML transitions",
			Statechart:  createHierarchicalStatechart(),
		},
		{
			Name:        "Parallel State Machine",
			Description: "SCXML parallel regions",
			Statechart:  createParallelStatechart(),
			AllowSemanticMapping: true, // SCXML handles parallel differently
		},
	}

	for _, tt := range tests {
		t.Run(tt.Name, func(t *testing.T) {
			original := tt.Statechart

			// Harel → SCXML
			scxmlConverter := scxml.NewConverter()
			scxmlDoc, err := scxmlConverter.Export(original)
			if err != nil {
				t.Fatalf("Harel → SCXML conversion failed: %v", err)
			}

			// SCXML → Harel
			restored, err := scxmlConverter.Import(scxmlDoc)
			if err != nil {
				t.Fatalf("SCXML → Harel conversion failed: %v", err)
			}

			// Validate semantic equivalence
			equiv := checkSemanticEquivalence(original, restored, tt)
			if !equiv.StatesMatch || !equiv.TransitionsMatch || !equiv.EventsMatch {
				t.Errorf("Round-trip failed for %s:", tt.Name)
				for _, detail := range equiv.Details {
					t.Errorf("  - %s", detail)
				}
			}
		})
	}
}

// TestXStateSCXMLCrossFormat tests XState ↔ SCXML cross-format conversion.
func TestXStateSCXMLCrossFormat(t *testing.T) {
	tests := []struct {
		Name        string
		XStateMachine xstate.Machine
		Description string
	}{
		{
			Name: "Simple XState to SCXML",
			XStateMachine: xstate.Machine{
				ID:      "simple",
				Initial: "idle",
				States: map[string]*xstate.State{
					"idle": {
						Type: "atomic",
						On: map[string]*xstate.Transition{
							"START": {Target: "running"},
						},
					},
					"running": {
						Type: "atomic",
						On: map[string]*xstate.Transition{
							"STOP": {Target: "idle"},
						},
					},
				},
			},
			Description: "Basic two-state machine",
		},
	}

	for _, tt := range tests {
		t.Run(tt.Name, func(t *testing.T) {
			xstateConverter := xstate.NewConverter()
			scxmlConverter := scxml.NewConverter()

			// XState → Harel
			harel1, err := xstateConverter.Import(tt.XStateMachine)
			if err != nil {
				t.Fatalf("XState → Harel failed: %v", err)
			}

			// Harel → SCXML
			scxmlDoc, err := scxmlConverter.Export(harel1)
			if err != nil {
				t.Fatalf("Harel → SCXML failed: %v", err)
			}

			// SCXML → Harel
			harel2, err := scxmlConverter.Import(scxmlDoc)
			if err != nil {
				t.Fatalf("SCXML → Harel failed: %v", err)
			}

			// Harel → XState
			xstateMachine2, err := xstateConverter.Export(harel2)
			if err != nil {
				t.Fatalf("Harel → XState failed: %v", err)
			}

			// Validate cross-format equivalence
			if xstateMachine2.ID == "" {
				t.Error("Cross-format conversion lost machine ID")
			}

			if len(xstateMachine2.States) != len(tt.XStateMachine.States) {
				t.Errorf("State count mismatch: original %d, converted %d", 
					len(tt.XStateMachine.States), len(xstateMachine2.States))
			}
		})
	}
}

// TestSemanticPreservation validates that core Harel semantics are preserved.
func TestSemanticPreservation(t *testing.T) {
	testCases := []struct {
		Name       string
		Statechart *sc.Statechart
		Semantics  []string // List of semantic properties to verify
	}{
		{
			Name:       "Hierarchical Semantics",
			Statechart: createHierarchicalStatechart(),
			Semantics:  []string{"parent-child", "initial-states", "transitions"},
		},
		{
			Name:       "Parallel Semantics", 
			Statechart: createParallelStatechart(),
			Semantics:  []string{"orthogonal-regions", "concurrent-execution"},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.Name, func(t *testing.T) {
			original := tc.Statechart

			// Test semantic preservation through multiple conversions
			formats := []string{"xstate", "scxml"}
			
			for _, format := range formats {
				t.Run(format, func(t *testing.T) {
					converted := roundTripThroughFormat(t, original, format)
					
					// Verify core semantics are preserved
					for _, semantic := range tc.Semantics {
						if !verifySemanticProperty(original, converted, semantic) {
							t.Errorf("Semantic property '%s' not preserved through %s conversion", semantic, format)
						}
					}
				})
			}
		})
	}
}

// Helper Functions

func createTrafficLightStatechart() *sc.Statechart {
	return &sc.Statechart{
		RootState: &sc.State{
			Label: "__root__",
			Type:  sc.StateTypeNormal,
			Children: []*sc.State{
				{Label: "green", Type: sc.StateTypeBasic, IsInitial: true},
				{Label: "yellow", Type: sc.StateTypeBasic},
				{Label: "red", Type: sc.StateTypeBasic},
			},
		},
		Transitions: []*sc.Transition{
			{Label: "green_to_yellow", From: []string{"green"}, To: []string{"yellow"}, Event: "TIMER"},
			{Label: "yellow_to_red", From: []string{"yellow"}, To: []string{"red"}, Event: "TIMER"},
			{Label: "red_to_green", From: []string{"red"}, To: []string{"green"}, Event: "TIMER"},
		},
		Events: []*sc.Event{
			{Label: "TIMER"},
		},
	}
}

func createHierarchicalStatechart() *sc.Statechart {
	return &sc.Statechart{
		RootState: &sc.State{
			Label: "__root__",
			Type:  sc.StateTypeNormal,
			Children: []*sc.State{
				{
					Label: "operating",
					Type:  sc.StateTypeNormal,
					IsInitial: true,
					Children: []*sc.State{
						{Label: "idle", Type: sc.StateTypeBasic, IsInitial: true},
						{Label: "working", Type: sc.StateTypeBasic},
					},
				},
				{Label: "error", Type: sc.StateTypeBasic},
			},
		},
		Transitions: []*sc.Transition{
			{Label: "start_work", From: []string{"idle"}, To: []string{"working"}, Event: "START"},
			{Label: "finish_work", From: []string{"working"}, To: []string{"idle"}, Event: "FINISH"},
			{Label: "error_from_any", From: []string{"operating"}, To: []string{"error"}, Event: "ERROR"},
			{Label: "reset", From: []string{"error"}, To: []string{"idle"}, Event: "RESET"},
		},
		Events: []*sc.Event{
			{Label: "START"}, {Label: "FINISH"}, {Label: "ERROR"}, {Label: "RESET"},
		},
	}
}

func createParallelStatechart() *sc.Statechart {
	return &sc.Statechart{
		RootState: &sc.State{
			Label: "__root__",
			Type:  sc.StateTypeNormal,
			Children: []*sc.State{
				{
					Label: "parallel_system",
					Type:  sc.StateTypeParallel,
					IsInitial: true,
					Children: []*sc.State{
						{
							Label: "region_a",
							Type:  sc.StateTypeNormal,
							Children: []*sc.State{
								{Label: "a1", Type: sc.StateTypeBasic, IsInitial: true},
								{Label: "a2", Type: sc.StateTypeBasic},
							},
						},
						{
							Label: "region_b", 
							Type:  sc.StateTypeNormal,
							Children: []*sc.State{
								{Label: "b1", Type: sc.StateTypeBasic, IsInitial: true},
								{Label: "b2", Type: sc.StateTypeBasic},
							},
						},
					},
				},
			},
		},
		Transitions: []*sc.Transition{
			{Label: "a1_to_a2", From: []string{"a1"}, To: []string{"a2"}, Event: "A_EVENT"},
			{Label: "b1_to_b2", From: []string{"b1"}, To: []string{"b2"}, Event: "B_EVENT"},
		},
		Events: []*sc.Event{
			{Label: "A_EVENT"}, {Label: "B_EVENT"},
		},
	}
}

func checkSemanticEquivalence(original, restored *sc.Statechart, test RoundTripTest) SemanticEquivalence {
	equiv := SemanticEquivalence{
		StatesMatch:      true,
		TransitionsMatch: true, 
		EventsMatch:      true,
		StructureMatch:   true,
	}

	// Check state count and structure
	originalStates := collectAllStates(original.RootState)
	restoredStates := collectAllStates(restored.RootState)

	if len(originalStates) != len(restoredStates) {
		equiv.StatesMatch = false
		equiv.Details = append(equiv.Details, 
			fmt.Sprintf("State count mismatch: original %d, restored %d", len(originalStates), len(restoredStates)))
	}

	// Check state types and hierarchy
	for label, originalState := range originalStates {
		if restoredState, exists := restoredStates[label]; exists {
			if originalState.Type != restoredState.Type && !test.AllowSemanticMapping {
				equiv.StatesMatch = false
				equiv.Details = append(equiv.Details, 
					fmt.Sprintf("State type mismatch for '%s': original %v, restored %v", 
					label, originalState.Type, restoredState.Type))
			}
		} else {
			equiv.StatesMatch = false
			equiv.Details = append(equiv.Details, fmt.Sprintf("Missing state: %s", label))
		}
	}

	// Check transition count and semantics
	if len(original.Transitions) != len(restored.Transitions) && !test.AllowStructuralChange {
		equiv.TransitionsMatch = false
		equiv.Details = append(equiv.Details, 
			fmt.Sprintf("Transition count mismatch: original %d, restored %d", 
			len(original.Transitions), len(restored.Transitions)))
	}

	// Check event preservation
	originalEvents := make(map[string]bool)
	for _, event := range original.Events {
		originalEvents[event.Label] = true
	}

	restoredEvents := make(map[string]bool)
	for _, event := range restored.Events {
		restoredEvents[event.Label] = true
	}

	if !reflect.DeepEqual(originalEvents, restoredEvents) {
		equiv.EventsMatch = false
		equiv.Details = append(equiv.Details, "Event sets don't match")
	}

	return equiv
}

func collectAllStates(root *sc.State) map[string]*sc.State {
	states := make(map[string]*sc.State)
	var visit func(*sc.State)
	visit = func(state *sc.State) {
		states[state.Label] = state
		for _, child := range state.Children {
			visit(child)
		}
	}
	visit(root)
	return states
}

func roundTripThroughFormat(t *testing.T, original *sc.Statechart, format string) *sc.Statechart {
	switch format {
	case "xstate":
		converter := xstate.NewConverter()
		machine, err := converter.Export(original)
		if err != nil {
			t.Fatalf("Export to XState failed: %v", err)
		}
		restored, err := converter.Import(machine)
		if err != nil {
			t.Fatalf("Import from XState failed: %v", err)
		}
		return restored

	case "scxml":
		converter := scxml.NewConverter()
		doc, err := converter.Export(original)
		if err != nil {
			t.Fatalf("Export to SCXML failed: %v", err)
		}
		restored, err := converter.Import(doc)
		if err != nil {
			t.Fatalf("Import from SCXML failed: %v", err)
		}
		return restored

	default:
		t.Fatalf("Unknown format: %s", format)
		return nil
	}
}

func verifySemanticProperty(original, converted *sc.Statechart, property string) bool {
	switch property {
	case "parent-child":
		return verifyParentChildRelationships(original, converted)
	case "initial-states":
		return verifyInitialStates(original, converted)
	case "transitions":
		return verifyTransitionSemantics(original, converted)
	case "orthogonal-regions":
		return verifyOrthogonalRegions(original, converted)
	case "concurrent-execution":
		return verifyConcurrentExecution(original, converted)
	default:
		return true // Unknown property, assume preserved
	}
}

func verifyParentChildRelationships(original, converted *sc.Statechart) bool {
	originalHierarchy := buildHierarchy(original.RootState)
	convertedHierarchy := buildHierarchy(converted.RootState)
	
	return reflect.DeepEqual(originalHierarchy, convertedHierarchy)
}

func verifyInitialStates(original, converted *sc.Statechart) bool {
	originalInitial := findInitialStates(original.RootState)
	convertedInitial := findInitialStates(converted.RootState)
	
	return reflect.DeepEqual(originalInitial, convertedInitial)
}

func verifyTransitionSemantics(original, converted *sc.Statechart) bool {
	// Check that essential transition structure is preserved
	originalTransMap := make(map[string][]string)
	for _, t := range original.Transitions {
		key := t.Event + ":" + t.From[0]
		originalTransMap[key] = t.To
	}
	
	convertedTransMap := make(map[string][]string)
	for _, t := range converted.Transitions {
		key := t.Event + ":" + t.From[0]
		convertedTransMap[key] = t.To
	}
	
	return reflect.DeepEqual(originalTransMap, convertedTransMap)
}

func verifyOrthogonalRegions(original, converted *sc.Statechart) bool {
	originalParallel := findParallelStates(original.RootState)
	convertedParallel := findParallelStates(converted.RootState)
	
	return len(originalParallel) == len(convertedParallel)
}

func verifyConcurrentExecution(original, converted *sc.Statechart) bool {
	// For now, just verify parallel states are preserved
	return verifyOrthogonalRegions(original, converted)
}

func buildHierarchy(root *sc.State) map[string][]string {
	hierarchy := make(map[string][]string)
	var visit func(*sc.State)
	visit = func(state *sc.State) {
		for _, child := range state.Children {
			hierarchy[state.Label] = append(hierarchy[state.Label], child.Label)
			visit(child)
		}
	}
	visit(root)
	return hierarchy
}

func findInitialStates(root *sc.State) []string {
	var initial []string
	var visit func(*sc.State)
	visit = func(state *sc.State) {
		if state.IsInitial {
			initial = append(initial, state.Label)
		}
		for _, child := range state.Children {
			visit(child)
		}
	}
	visit(root)
	return initial
}

func findParallelStates(root *sc.State) []string {
	var parallel []string
	var visit func(*sc.State)
	visit = func(state *sc.State) {
		if state.Type == sc.StateTypeParallel {
			parallel = append(parallel, state.Label)
		}
		for _, child := range state.Children {
			visit(child)
		}
	}
	visit(root)
	return parallel
}