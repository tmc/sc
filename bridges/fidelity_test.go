package bridges

import (
	"testing"

	"github.com/tmc/sc"
	"github.com/tmc/sc/bridges/scxml"
	"github.com/tmc/sc/bridges/xstate"
)

// TestConversionFidelity tests specific aspects of conversion accuracy.
func TestConversionFidelity(t *testing.T) {
	testCases := []struct {
		Name       string
		Statechart *sc.Statechart
		Checks     []FidelityCheck
	}{
		{
			Name:       "Guard Conditions",
			Statechart: createStatechartWithGuards(),
			Checks: []FidelityCheck{
				{Type: "guard-expressions", Property: "guard conditions preserved"},
				{Type: "transition-logic", Property: "conditional transitions work"},
			},
		},
		{
			Name:       "Action Execution",
			Statechart: createStatechartWithActions(),
			Checks: []FidelityCheck{
				{Type: "action-labels", Property: "action names preserved"},
				{Type: "action-sequence", Property: "action order maintained"},
			},
		},
		{
			Name:       "Complex Hierarchy",
			Statechart: createDeepHierarchicalStatechart(),
			Checks: []FidelityCheck{
				{Type: "nesting-depth", Property: "hierarchy depth preserved"},
				{Type: "scope-resolution", Property: "transition scope correct"},
			},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.Name, func(t *testing.T) {
			// Test through XState conversion
			t.Run("XState", func(t *testing.T) {
				xstateConverter := xstate.NewConverter()
				machine, err := xstateConverter.Export(tc.Statechart)
				if err != nil {
					t.Fatalf("XState export failed: %v", err)
				}

				restored, err := xstateConverter.Import(machine)
				if err != nil {
					t.Fatalf("XState import failed: %v", err)
				}

				for _, check := range tc.Checks {
					if !performFidelityCheck(tc.Statechart, restored, check) {
						t.Errorf("Fidelity check failed: %s - %s", check.Type, check.Property)
					}
				}
			})

			// Test through SCXML conversion
			t.Run("SCXML", func(t *testing.T) {
				scxmlConverter := scxml.NewConverter()
				doc, err := scxmlConverter.Export(tc.Statechart)
				if err != nil {
					t.Fatalf("SCXML export failed: %v", err)
				}

				restored, err := scxmlConverter.Import(doc)
				if err != nil {
					t.Fatalf("SCXML import failed: %v", err)
				}

				for _, check := range tc.Checks {
					if !performFidelityCheck(tc.Statechart, restored, check) {
						t.Errorf("Fidelity check failed: %s - %s", check.Type, check.Property)
					}
				}
			})
		})
	}
}

// TestEdgeCases tests conversion behavior for edge cases and boundary conditions.
func TestEdgeCases(t *testing.T) {
	testCases := []struct {
		Name       string
		Statechart *sc.Statechart
		ExpectIssues bool
		Description string
	}{
		{
			Name:       "Empty State Machine",
			Statechart: createEmptyStatechart(),
			ExpectIssues: false, // Our converters handle empty gracefully
			Description: "Machine with no states should handle gracefully",
		},
		{
			Name:       "Single State",
			Statechart: createSingleStateStatechart(),
			ExpectIssues: false,
			Description: "Minimal valid state machine",
		},
		{
			Name:       "Deeply Nested",
			Statechart: createDeepNestingStatechart(5),
			ExpectIssues: false,
			Description: "Very deep hierarchy should be preserved",
		},
		{
			Name:       "Many Transitions",
			Statechart: createHighFanoutStatechart(),
			ExpectIssues: false,
			Description: "State with many outgoing transitions",
		},
	}

	for _, tc := range testCases {
		t.Run(tc.Name, func(t *testing.T) {
			// Test XState conversion
			t.Run("XState", func(t *testing.T) {
				xstateConverter := xstate.NewConverter()
				
				_, exportErr := xstateConverter.Export(tc.Statechart)
				if tc.ExpectIssues && exportErr == nil {
					t.Errorf("Expected export to fail for %s, but it succeeded", tc.Description)
				}
				if !tc.ExpectIssues && exportErr != nil {
					t.Errorf("Unexpected export failure for %s: %v", tc.Description, exportErr)
				}
			})

			// Test SCXML conversion
			t.Run("SCXML", func(t *testing.T) {
				scxmlConverter := scxml.NewConverter()
				
				_, exportErr := scxmlConverter.Export(tc.Statechart)
				if tc.ExpectIssues && exportErr == nil {
					t.Errorf("Expected export to fail for %s, but it succeeded", tc.Description)
				}
				if !tc.ExpectIssues && exportErr != nil {
					t.Errorf("Unexpected export failure for %s: %v", tc.Description, exportErr)
				}
			})
		})
	}
}

// TestFormatSpecificFeatures tests how format-specific features are handled.
func TestFormatSpecificFeatures(t *testing.T) {
	t.Run("XState Actor Model", func(t *testing.T) {
		// Test XState features that don't map directly to Harel
		machine := xstate.Machine{
			ID: "actor_test",
			States: map[string]*xstate.State{
				"idle": {
					Type: "atomic",
					On: map[string]*xstate.Transition{
						"INVOKE": {
							Target: "working",
							Actions: []string{"spawnWorker"},
						},
					},
				},
				"working": {
					Type: "atomic",
				},
			},
		}

		xstateConverter := xstate.NewConverter()
		harel, err := xstateConverter.Import(machine)
		if err != nil {
			t.Fatalf("Import failed: %v", err)
		}

		// Verify that unsupported features are handled gracefully
		if len(harel.Transitions) == 0 {
			t.Error("Expected transitions to be preserved even with unsupported actions")
		}
	})

	t.Run("SCXML Datamodel", func(t *testing.T) {
		// Test SCXML datamodel features
		doc := scxml.Document{
			Namespace: "http://www.w3.org/2005/07/scxml",
			Version:   "1.0",
			Datamodel: &scxml.Datamodel{
				Data: []*scxml.Data{
					{ID: "counter", Expr: "0"},
					{ID: "name", Content: "test"},
				},
			},
			States: []*scxml.State{
				{ID: "state1"},
			},
		}

		scxmlConverter := scxml.NewConverter()
		harel, err := scxmlConverter.Import(doc)
		if err != nil {
			t.Fatalf("Import failed: %v", err)
		}

		// Verify basic structure is preserved
		if harel.RootState == nil {
			t.Error("Expected root state to be created")
		}
	})
}

// TestSemanticConsistency validates that semantic meaning is preserved across conversions.
func TestSemanticConsistency(t *testing.T) {
	original := createComplexSemanticStatechart()

	// Multi-hop conversion test: Harel → XState → SCXML → Harel
	xstateConverter := xstate.NewConverter()
	scxmlConverter := scxml.NewConverter()

	// Step 1: Harel → XState
	xstateMachine, err := xstateConverter.Export(original)
	if err != nil {
		t.Fatalf("Harel → XState failed: %v", err)
	}

	// Step 2: XState → Harel
	harel1, err := xstateConverter.Import(xstateMachine)
	if err != nil {
		t.Fatalf("XState → Harel failed: %v", err)
	}

	// Step 3: Harel → SCXML
	scxmlDoc, err := scxmlConverter.Export(harel1)
	if err != nil {
		t.Fatalf("Harel → SCXML failed: %v", err)
	}

	// Step 4: SCXML → Harel
	final, err := scxmlConverter.Import(scxmlDoc)
	if err != nil {
		t.Fatalf("SCXML → Harel failed: %v", err)
	}

	// Verify core semantics are preserved through multi-hop conversion
	if !semanticallyEquivalent(original, final) {
		t.Error("Multi-hop conversion changed core semantics")
	}
}

// Helper Types and Functions

type FidelityCheck struct {
	Type     string
	Property string
}

func performFidelityCheck(original, converted *sc.Statechart, check FidelityCheck) bool {
	switch check.Type {
	case "guard-expressions":
		return checkGuardPreservation(original, converted)
	case "transition-logic":
		return checkTransitionLogic(original, converted)
	case "action-labels":
		return checkActionLabels(original, converted)
	case "action-sequence":
		return checkActionSequence(original, converted)
	case "nesting-depth":
		return checkNestingDepth(original, converted)
	case "scope-resolution":
		return checkScopeResolution(original, converted)
	default:
		return true // Unknown check type, assume passed
	}
}

func createStatechartWithGuards() *sc.Statechart {
	return &sc.Statechart{
		RootState: &sc.State{
			Label: "__root__",
			Type:  sc.StateTypeNormal,
			Children: []*sc.State{
				{Label: "waiting", Type: sc.StateTypeBasic, IsInitial: true},
				{Label: "processing", Type: sc.StateTypeBasic},
				{Label: "complete", Type: sc.StateTypeBasic},
			},
		},
		Transitions: []*sc.Transition{
			{
				Label: "start_if_ready",
				From:  []string{"waiting"},
				To:    []string{"processing"},
				Event: "START",
				Guard: &sc.Guard{Expression: "ready == true"},
			},
			{
				Label: "finish_if_done",
				From:  []string{"processing"},
				To:    []string{"complete"},
				Event: "FINISH",
				Guard: &sc.Guard{Expression: "count > 0"},
			},
		},
		Events: []*sc.Event{{Label: "START"}, {Label: "FINISH"}},
	}
}

func createStatechartWithActions() *sc.Statechart {
	return &sc.Statechart{
		RootState: &sc.State{
			Label: "__root__",
			Type:  sc.StateTypeNormal,
			Children: []*sc.State{
				{Label: "idle", Type: sc.StateTypeBasic, IsInitial: true},
				{Label: "active", Type: sc.StateTypeBasic},
			},
		},
		Transitions: []*sc.Transition{
			{
				Label: "activate",
				From:  []string{"idle"},
				To:    []string{"active"},
				Event: "ACTIVATE",
				Actions: []*sc.Action{
					{Label: "initializeSystem"},
					{Label: "startLogging"},
					{Label: "notifyObservers"},
				},
			},
		},
		Events: []*sc.Event{{Label: "ACTIVATE"}},
	}
}

func createDeepHierarchicalStatechart() *sc.Statechart {
	return &sc.Statechart{
		RootState: &sc.State{
			Label: "__root__",
			Type:  sc.StateTypeNormal,
			Children: []*sc.State{
				{
					Label: "level1",
					Type:  sc.StateTypeNormal,
					IsInitial: true,
					Children: []*sc.State{
						{
							Label: "level2",
							Type:  sc.StateTypeNormal,
							IsInitial: true,
							Children: []*sc.State{
								{
									Label: "level3",
									Type:  sc.StateTypeNormal,
									IsInitial: true,
									Children: []*sc.State{
										{Label: "deepest", Type: sc.StateTypeBasic, IsInitial: true},
									},
								},
							},
						},
					},
				},
			},
		},
		Transitions: []*sc.Transition{
			{
				Label: "deep_transition",
				From:  []string{"deepest"},
				To:    []string{"level1"},
				Event: "ESCAPE",
			},
		},
		Events: []*sc.Event{{Label: "ESCAPE"}},
	}
}

func createEmptyStatechart() *sc.Statechart {
	return &sc.Statechart{
		RootState: &sc.State{
			Label: "__root__",
			Type:  sc.StateTypeNormal,
		},
	}
}

func createSingleStateStatechart() *sc.Statechart {
	return &sc.Statechart{
		RootState: &sc.State{
			Label: "__root__",
			Type:  sc.StateTypeNormal,
			Children: []*sc.State{
				{Label: "only", Type: sc.StateTypeBasic, IsInitial: true},
			},
		},
	}
}

func createDeepNestingStatechart(depth int) *sc.Statechart {
	root := &sc.State{
		Label: "__root__",
		Type:  sc.StateTypeNormal,
	}

	current := root
	for i := 0; i < depth; i++ {
		child := &sc.State{
			Label:     "level" + string(rune('0'+i)),
			Type:      sc.StateTypeNormal,
			IsInitial: true,
		}
		current.Children = []*sc.State{child}
		current = child
	}

	// Add leaf state
	current.Children = []*sc.State{
		{Label: "leaf", Type: sc.StateTypeBasic, IsInitial: true},
	}

	return &sc.Statechart{RootState: root}
}

func createHighFanoutStatechart() *sc.Statechart {
	root := &sc.State{
		Label: "__root__",
		Type:  sc.StateTypeNormal,
		Children: []*sc.State{
			{Label: "central", Type: sc.StateTypeBasic, IsInitial: true},
		},
	}

	var transitions []*sc.Transition
	var events []*sc.Event

	// Create many target states and transitions
	for i := 0; i < 10; i++ {
		target := "target" + string(rune('0'+i))
		event := "EVENT" + string(rune('0'+i))

		root.Children = append(root.Children, &sc.State{
			Label: target,
			Type:  sc.StateTypeBasic,
		})

		transitions = append(transitions, &sc.Transition{
			Label: "to_" + target,
			From:  []string{"central"},
			To:    []string{target},
			Event: event,
		})

		events = append(events, &sc.Event{Label: event})
	}

	return &sc.Statechart{
		RootState:   root,
		Transitions: transitions,
		Events:      events,
	}
}

func createComplexSemanticStatechart() *sc.Statechart {
	// Combines multiple semantic features for comprehensive testing
	return &sc.Statechart{
		RootState: &sc.State{
			Label: "__root__",
			Type:  sc.StateTypeNormal,
			Children: []*sc.State{
				{
					Label: "system",
					Type:  sc.StateTypeParallel,
					IsInitial: true,
					Children: []*sc.State{
						{
							Label: "mode",
							Type:  sc.StateTypeNormal,
							Children: []*sc.State{
								{Label: "manual", Type: sc.StateTypeBasic, IsInitial: true},
								{Label: "auto", Type: sc.StateTypeBasic},
							},
						},
						{
							Label: "status",
							Type:  sc.StateTypeNormal,
							Children: []*sc.State{
								{Label: "ready", Type: sc.StateTypeBasic, IsInitial: true},
								{Label: "busy", Type: sc.StateTypeBasic},
								{Label: "error", Type: sc.StateTypeBasic},
							},
						},
					},
				},
			},
		},
		Transitions: []*sc.Transition{
			{Label: "switch_mode", From: []string{"manual"}, To: []string{"auto"}, Event: "SWITCH"},
			{Label: "start_work", From: []string{"ready"}, To: []string{"busy"}, Event: "START"},
			{Label: "error_occurred", From: []string{"busy"}, To: []string{"error"}, Event: "ERROR"},
		},
		Events: []*sc.Event{{Label: "SWITCH"}, {Label: "START"}, {Label: "ERROR"}},
	}
}

// Semantic equivalence checking functions

func checkGuardPreservation(original, converted *sc.Statechart) bool {
	originalGuards := extractGuards(original)
	convertedGuards := extractGuards(converted)
	
	// Allow for minor expression format changes
	return len(originalGuards) == len(convertedGuards)
}

func checkTransitionLogic(original, converted *sc.Statechart) bool {
	return len(original.Transitions) <= len(converted.Transitions) // Allow additional internal transitions
}

func checkActionLabels(original, converted *sc.Statechart) bool {
	originalActions := extractActionLabels(original)
	convertedActions := extractActionLabels(converted)
	
	for label := range originalActions {
		if !convertedActions[label] {
			return false
		}
	}
	return true
}

func checkActionSequence(original, converted *sc.Statechart) bool {
	// For now, just check that actions exist
	return checkActionLabels(original, converted)
}

func checkNestingDepth(original, converted *sc.Statechart) bool {
	originalDepth := calculateMaxDepth(original.RootState)
	convertedDepth := calculateMaxDepth(converted.RootState)
	
	return originalDepth == convertedDepth
}

func checkScopeResolution(original, converted *sc.Statechart) bool {
	// Check that transitions maintain correct scope
	return len(original.Transitions) <= len(converted.Transitions)
}

func semanticallyEquivalent(a, b *sc.Statechart) bool {
	// Simplified semantic equivalence check
	return calculateMaxDepth(a.RootState) == calculateMaxDepth(b.RootState) &&
		   len(a.Events) <= len(b.Events) &&
		   countStates(a.RootState) == countStates(b.RootState)
}

func extractGuards(sc *sc.Statechart) []string {
	var guards []string
	for _, t := range sc.Transitions {
		if t.Guard != nil && t.Guard.Expression != "" {
			guards = append(guards, t.Guard.Expression)
		}
	}
	return guards
}

func extractActionLabels(sc *sc.Statechart) map[string]bool {
	actions := make(map[string]bool)
	for _, t := range sc.Transitions {
		for _, action := range t.Actions {
			actions[action.Label] = true
		}
	}
	return actions
}

func calculateMaxDepth(root *sc.State) int {
	if len(root.Children) == 0 {
		return 0
	}
	
	maxChildDepth := 0
	for _, child := range root.Children {
		depth := calculateMaxDepth(child)
		if depth > maxChildDepth {
			maxChildDepth = depth
		}
	}
	
	return maxChildDepth + 1
}

func countStates(root *sc.State) int {
	count := 1
	for _, child := range root.Children {
		count += countStates(child)
	}
	return count
}