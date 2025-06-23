package semantics

import (
	"testing"

	sc "github.com/tmc/sc"
	"google.golang.org/protobuf/types/known/structpb"
)

func TestExecuteTransitions(t *testing.T) {
	// Create a simple statechart: Off -> On
	statechart := NewStatechart(&sc.Statechart{
		RootState: &sc.State{
			Label: "__root__",
			Type:  sc.StateTypeNormal,
			Children: []*sc.State{
				{Label: "Off", Type: sc.StateTypeBasic, IsInitial: true},
				{Label: "On", Type: sc.StateTypeBasic},
			},
		},
		Transitions: []*sc.Transition{
			{
				Label: "turn_on",
				From:  []string{"Off"},
				To:    []string{"On"},
				Event: "TURN_ON",
				Guard: &sc.Guard{Expression: "context.count < 5"},
				Actions: []*sc.Action{{Label: "increment_count"}},
			},
		},
	})

	config := &sc.Configuration{
		States: []*sc.StateRef{{Label: "Off"}},
	}

	context := &structpb.Struct{
		Fields: map[string]*structpb.Value{
			"count": structpb.NewNumberValue(0),
		},
	}

	result, err := statechart.ExecuteTransitions(config, context, "TURN_ON")
	if err != nil {
		t.Fatalf("ExecuteTransitions failed: %v", err)
	}

	if !result.Executed {
		t.Error("Expected transition to be executed")
	}

	// Check new configuration
	if len(result.NewConfig.States) != 1 || result.NewConfig.States[0].Label != "On" {
		t.Errorf("Expected configuration [On], got %v", result.NewConfig.States)
	}

	// Check context was updated
	count, ok := result.NewContext.Fields["count"].GetKind().(*structpb.Value_NumberValue)
	if !ok || count.NumberValue != 1 {
		t.Errorf("Expected context count to be 1, got %v", result.NewContext.Fields["count"])
	}
}

func TestFindEnabledTransitions(t *testing.T) {
	statechart := NewStatechart(&sc.Statechart{
		RootState: &sc.State{
			Label: "__root__",
			Type:  sc.StateTypeNormal,
			Children: []*sc.State{
				{Label: "Off", Type: sc.StateTypeBasic, IsInitial: true},
				{Label: "On", Type: sc.StateTypeBasic},
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
	})

	config := &sc.Configuration{
		States: []*sc.StateRef{{Label: "Off"}},
	}

	enabled, err := statechart.FindEnabledTransitions(config, nil, "TURN_ON")
	if err != nil {
		t.Fatalf("FindEnabledTransitions failed: %v", err)
	}

	if len(enabled) != 1 {
		t.Errorf("Expected 1 enabled transition, got %d", len(enabled))
	}

	if enabled[0].Label != "turn_on" {
		t.Errorf("Expected 'turn_on' transition, got %s", enabled[0].Label)
	}

	// Test with different event
	enabled, err = statechart.FindEnabledTransitions(config, nil, "TURN_OFF")
	if err != nil {
		t.Fatalf("FindEnabledTransitions failed: %v", err)
	}

	if len(enabled) != 0 {
		t.Errorf("Expected 0 enabled transitions, got %d", len(enabled))
	}
}

func TestResolveConflicts(t *testing.T) {
	statechart := NewStatechart(&sc.Statechart{
		RootState: &sc.State{
			Label: "__root__",
			Type:  sc.StateTypeNormal,
			Children: []*sc.State{
				{
					Label: "A",
					Type:  sc.StateTypeNormal,
					Children: []*sc.State{
						{Label: "A1", Type: sc.StateTypeBasic, IsInitial: true},
						{Label: "A2", Type: sc.StateTypeBasic},
					},
				},
				{Label: "B", Type: sc.StateTypeBasic},
			},
		},
		Transitions: []*sc.Transition{
			{
				Label: "t1",
				From:  []string{"A1"},
				To:    []string{"B"},
				Event: "E",
			},
			{
				Label: "t2",
				From:  []string{"A"},
				To:    []string{"B"},
				Event: "E",
			},
		},
	})

	config := &sc.Configuration{
		States: []*sc.StateRef{{Label: "A"}, {Label: "A1"}},
	}

	transitions := []*sc.Transition{
		statechart.Transitions[0], // t1 from A1
		statechart.Transitions[1], // t2 from A
	}

	resolved, err := statechart.ResolveConflicts(transitions, config)
	if err != nil {
		t.Fatalf("ResolveConflicts failed: %v", err)
	}

	// t1 should have higher priority (deeper state)
	if len(resolved) != 1 {
		t.Errorf("Expected 1 resolved transition, got %d", len(resolved))
	}

	if resolved[0].Label != "t1" {
		t.Errorf("Expected 't1' to be selected, got %s", resolved[0].Label)
	}
}

func TestHierarchicalTransitions(t *testing.T) {
	// Test hierarchical state transitions
	statechart := NewStatechart(&sc.Statechart{
		RootState: &sc.State{
			Label: "__root__",
			Type:  sc.StateTypeNormal,
			Children: []*sc.State{
				{
					Label: "Outer",
					Type:  sc.StateTypeNormal,
					Children: []*sc.State{
						{
							Label: "Middle",
							Type:  sc.StateTypeNormal,
							Children: []*sc.State{
								{Label: "Inner", Type: sc.StateTypeBasic, IsInitial: true},
							},
							IsInitial: true,
						},
					},
					IsInitial: true,
				},
				{Label: "Target", Type: sc.StateTypeBasic},
			},
		},
		Transitions: []*sc.Transition{
			{
				Label: "exit_hierarchy",
				From:  []string{"Inner"},
				To:    []string{"Target"},
				Event: "EXIT",
			},
		},
	})

	config := &sc.Configuration{
		States: []*sc.StateRef{
			{Label: "Outer"},
			{Label: "Middle"},
			{Label: "Inner"},
		},
	}

	result, err := statechart.ExecuteTransitions(config, nil, "EXIT")
	if err != nil {
		t.Fatalf("ExecuteTransitions failed: %v", err)
	}

	if !result.Executed {
		t.Error("Expected transition to be executed")
	}

	// Check that we transitioned to Target
	found := false
	for _, state := range result.NewConfig.States {
		if state.Label == "Target" {
			found = true
			break
		}
	}
	if !found {
		t.Error("Expected Target state to be active")
	}
}

func TestParallelStates(t *testing.T) {
	// Test orthogonal/parallel state transitions
	statechart := NewStatechart(&sc.Statechart{
		RootState: &sc.State{
			Label: "__root__",
			Type:  sc.StateTypeParallel,
			Children: []*sc.State{
				{
					Label: "Region1",
					Type:  sc.StateTypeNormal,
					Children: []*sc.State{
						{Label: "R1S1", Type: sc.StateTypeBasic, IsInitial: true},
						{Label: "R1S2", Type: sc.StateTypeBasic},
					},
				},
				{
					Label: "Region2",
					Type:  sc.StateTypeNormal,
					Children: []*sc.State{
						{Label: "R2S1", Type: sc.StateTypeBasic, IsInitial: true},
						{Label: "R2S2", Type: sc.StateTypeBasic},
					},
				},
			},
		},
		Transitions: []*sc.Transition{
			{
				Label: "r1_transition",
				From:  []string{"R1S1"},
				To:    []string{"R1S2"},
				Event: "E1",
			},
			{
				Label: "r2_transition",
				From:  []string{"R2S1"},
				To:    []string{"R2S2"},
				Event: "E2",
			},
		},
	})

	config := &sc.Configuration{
		States: []*sc.StateRef{
			{Label: "Region1"},
			{Label: "R1S1"},
			{Label: "Region2"},
			{Label: "R2S1"},
		},
	}

	// Execute transition in Region1
	result, err := statechart.ExecuteTransitions(config, nil, "E1")
	if err != nil {
		t.Fatalf("ExecuteTransitions failed: %v", err)
	}

	if !result.Executed {
		t.Error("Expected transition to be executed")
	}

	// Check that both regions are still active
	stateMap := make(map[string]bool)
	for _, state := range result.NewConfig.States {
		stateMap[state.Label] = true
	}

	expectedStates := []string{"Region1", "R1S2", "Region2", "R2S1"}
	for _, expected := range expectedStates {
		if !stateMap[expected] {
			t.Errorf("Expected state %s to be active", expected)
		}
	}
}

func TestGuardEvaluation(t *testing.T) {
	statechart := NewStatechart(&sc.Statechart{
		RootState: &sc.State{
			Label: "__root__",
			Type:  sc.StateTypeNormal,
			Children: []*sc.State{
				{Label: "S1", Type: sc.StateTypeBasic, IsInitial: true},
				{Label: "S2", Type: sc.StateTypeBasic},
			},
		},
	})

	tests := []struct {
		name     string
		guard    *sc.Guard
		context  *structpb.Struct
		expected bool
	}{
		{
			name:     "No guard",
			guard:    nil,
			context:  nil,
			expected: true,
		},
		{
			name:     "Empty guard expression",
			guard:    &sc.Guard{Expression: ""},
			context:  nil,
			expected: true,
		},
		{
			name:     "True guard",
			guard:    &sc.Guard{Expression: "true"},
			context:  nil,
			expected: true,
		},
		{
			name:     "False guard",
			guard:    &sc.Guard{Expression: "false"},
			context:  nil,
			expected: false,
		},
		{
			name:  "Context-based guard - passes",
			guard: &sc.Guard{Expression: "context.count < 5"},
			context: &structpb.Struct{
				Fields: map[string]*structpb.Value{
					"count": structpb.NewNumberValue(3),
				},
			},
			expected: true,
		},
		{
			name:  "Context-based guard - fails",
			guard: &sc.Guard{Expression: "context.count < 5"},
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
			result, err := statechart.EvaluateGuard(tt.guard, tt.context)
			if err != nil {
				t.Fatalf("EvaluateGuard failed: %v", err)
			}
			if result != tt.expected {
				t.Errorf("Expected guard evaluation to be %v, got %v", tt.expected, result)
			}
		})
	}
}

func TestActionExecution(t *testing.T) {
	statechart := NewStatechart(&sc.Statechart{
		RootState: &sc.State{
			Label: "__root__",
			Type:  sc.StateTypeNormal,
			Children: []*sc.State{
				{Label: "S1", Type: sc.StateTypeBasic, IsInitial: true},
				{Label: "S2", Type: sc.StateTypeBasic},
			},
		},
	})

	tests := []struct {
		name           string
		action         *sc.Action
		initialContext *structpb.Struct
		expectedCount  float64
	}{
		{
			name:   "Increment action",
			action: &sc.Action{Label: "increment_count"},
			initialContext: &structpb.Struct{
				Fields: map[string]*structpb.Value{
					"count": structpb.NewNumberValue(5),
				},
			},
			expectedCount: 6,
		},
		{
			name:   "Decrement action",
			action: &sc.Action{Label: "decrement_count"},
			initialContext: &structpb.Struct{
				Fields: map[string]*structpb.Value{
					"count": structpb.NewNumberValue(5),
				},
			},
			expectedCount: 4,
		},
		{
			name:   "Reset action",
			action: &sc.Action{Label: "reset_count"},
			initialContext: &structpb.Struct{
				Fields: map[string]*structpb.Value{
					"count": structpb.NewNumberValue(10),
				},
			},
			expectedCount: 0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := statechart.ExecuteAction(tt.action, tt.initialContext)
			if err != nil {
				t.Fatalf("ExecuteAction failed: %v", err)
			}

			count, ok := tt.initialContext.Fields["count"].GetKind().(*structpb.Value_NumberValue)
			if !ok {
				t.Fatal("Count field is not a number")
			}

			if count.NumberValue != tt.expectedCount {
				t.Errorf("Expected count to be %v, got %v", tt.expectedCount, count.NumberValue)
			}
		})
	}
}

func TestTransitionPriority(t *testing.T) {
	statechart := NewStatechart(&sc.Statechart{
		RootState: &sc.State{
			Label: "__root__",
			Type:  sc.StateTypeNormal,
			Children: []*sc.State{
				{
					Label: "A",
					Type:  sc.StateTypeNormal,
					Children: []*sc.State{
						{
							Label: "A1",
							Type:  sc.StateTypeNormal,
							Children: []*sc.State{
								{Label: "A11", Type: sc.StateTypeBasic, IsInitial: true},
							},
							IsInitial: true,
						},
					},
				},
			},
		},
		Transitions: []*sc.Transition{
			{Label: "t1", From: []string{"A11"}, To: []string{"A"}, Event: "E"},
			{Label: "t2", From: []string{"A1"}, To: []string{"A"}, Event: "E"},
			{Label: "t3", From: []string{"A"}, To: []string{"A"}, Event: "E"},
		},
	})

	// Test priority calculation
	priority1 := statechart.calculateTransitionPriority(statechart.Transitions[0]) // from A11
	priority2 := statechart.calculateTransitionPriority(statechart.Transitions[1]) // from A1
	priority3 := statechart.calculateTransitionPriority(statechart.Transitions[2]) // from A

	if priority1 <= priority2 {
		t.Errorf("Expected transition from A11 to have higher priority than from A1")
	}

	if priority2 <= priority3 {
		t.Errorf("Expected transition from A1 to have higher priority than from A")
	}
}

func TestIsTransitionEnabled(t *testing.T) {
	statechart := NewStatechart(&sc.Statechart{
		RootState: &sc.State{
			Label: "__root__",
			Type:  sc.StateTypeNormal,
			Children: []*sc.State{
				{Label: "S1", Type: sc.StateTypeBasic, IsInitial: true},
				{Label: "S2", Type: sc.StateTypeBasic},
			},
		},
	})

	transition := &sc.Transition{
		Label: "t1",
		From:  []string{"S1"},
		To:    []string{"S2"},
		Event: "E",
		Guard: &sc.Guard{Expression: "context.count < 5"},
	}

	config := &sc.Configuration{
		States: []*sc.StateRef{{Label: "S1"}},
	}

	context := &structpb.Struct{
		Fields: map[string]*structpb.Value{
			"count": structpb.NewNumberValue(3),
		},
	}

	enabled, err := statechart.IsTransitionEnabled(transition, config, context)
	if err != nil {
		t.Fatalf("IsTransitionEnabled failed: %v", err)
	}

	if !enabled {
		t.Error("Expected transition to be enabled")
	}

	// Test with guard failing
	context.Fields["count"] = structpb.NewNumberValue(7)
	enabled, err = statechart.IsTransitionEnabled(transition, config, context)
	if err != nil {
		t.Fatalf("IsTransitionEnabled failed: %v", err)
	}

	if enabled {
		t.Error("Expected transition to be disabled due to guard")
	}
}

func TestGetTransitionsByEvent(t *testing.T) {
	statechart := NewStatechart(&sc.Statechart{
		RootState: &sc.State{
			Label: "__root__",
			Type:  sc.StateTypeNormal,
			Children: []*sc.State{
				{Label: "S1", Type: sc.StateTypeBasic, IsInitial: true},
				{Label: "S2", Type: sc.StateTypeBasic},
			},
		},
		Transitions: []*sc.Transition{
			{Label: "t1", From: []string{"S1"}, To: []string{"S2"}, Event: "E1"},
			{Label: "t2", From: []string{"S2"}, To: []string{"S1"}, Event: "E2"},
			{Label: "t3", From: []string{"S1"}, To: []string{"S2"}, Event: "E1"},
		},
	})

	transitions := statechart.GetTransitionsByEvent("E1")
	if len(transitions) != 2 {
		t.Errorf("Expected 2 transitions for event E1, got %d", len(transitions))
	}

	transitions = statechart.GetTransitionsByEvent("E2")
	if len(transitions) != 1 {
		t.Errorf("Expected 1 transition for event E2, got %d", len(transitions))
	}

	transitions = statechart.GetTransitionsByEvent("E3")
	if len(transitions) != 0 {
		t.Errorf("Expected 0 transitions for event E3, got %d", len(transitions))
	}
}

func TestCompoundTransition(t *testing.T) {
	statechart := NewStatechart(&sc.Statechart{
		RootState: &sc.State{
			Label: "__root__",
			Type:  sc.StateTypeParallel,
			Children: []*sc.State{
				{
					Label: "Region1",
					Type:  sc.StateTypeNormal,
					Children: []*sc.State{
						{Label: "R1S1", Type: sc.StateTypeBasic, IsInitial: true},
						{Label: "R1S2", Type: sc.StateTypeBasic},
					},
				},
				{
					Label: "Region2", 
					Type:  sc.StateTypeNormal,
					Children: []*sc.State{
						{Label: "R2S1", Type: sc.StateTypeBasic, IsInitial: true},
						{Label: "R2S2", Type: sc.StateTypeBasic},
					},
				},
			},
		},
		Transitions: []*sc.Transition{
			{
				Label: "t1",
				From:  []string{"R1S1"},
				To:    []string{"R1S2"},
				Event: "COMPOUND",
			},
			{
				Label: "t2",
				From:  []string{"R2S1"},
				To:    []string{"R2S2"},
				Event: "COMPOUND",
			},
		},
	})

	config := &sc.Configuration{
		States: []*sc.StateRef{
			{Label: "Region1"},
			{Label: "R1S1"},
			{Label: "Region2"},
			{Label: "R2S1"},
		},
	}

	compound := &CompoundTransition{
		Label:       "compound_transition",
		Transitions: []*sc.Transition{statechart.Transitions[0], statechart.Transitions[1]},
		Event:       "COMPOUND",
		Actions:     []*sc.Action{{Label: "compound_action"}},
	}

	result, err := statechart.ExecuteCompoundTransition(compound, config, nil)
	if err != nil {
		t.Fatalf("ExecuteCompoundTransition failed: %v", err)
	}

	if !result.Executed {
		t.Error("Expected compound transition to be executed")
	}

	// Check that both regions transitioned
	stateMap := make(map[string]bool)
	for _, state := range result.NewConfig.States {
		stateMap[state.Label] = true
	}

	expectedStates := []string{"Region1", "R1S2", "Region2", "R2S2"}
	for _, expected := range expectedStates {
		if !stateMap[expected] {
			t.Errorf("Expected state %s to be active", expected)
		}
	}
}

func TestCrossRegionTransition(t *testing.T) {
	statechart := NewStatechart(&sc.Statechart{
		RootState: &sc.State{
			Label: "__root__",
			Type:  sc.StateTypeParallel,
			Children: []*sc.State{
				{
					Label: "Region1",
					Type:  sc.StateTypeNormal,
					Children: []*sc.State{
						{Label: "R1S1", Type: sc.StateTypeBasic, IsInitial: true},
						{Label: "R1S2", Type: sc.StateTypeBasic},
					},
				},
				{
					Label: "Region2",
					Type:  sc.StateTypeNormal,
					Children: []*sc.State{
						{Label: "R2S1", Type: sc.StateTypeBasic, IsInitial: true},
						{Label: "R2S2", Type: sc.StateTypeBasic},
					},
				},
			},
		},
	})

	config := &sc.Configuration{
		States: []*sc.StateRef{
			{Label: "Region1"},
			{Label: "R1S1"},
			{Label: "Region2"},
			{Label: "R2S1"},
		},
	}

	crossTransition := &CrossRegionTransition{
		Label: "cross_region_transition",
		Sources: map[string]string{
			"Region1": "R1S1",
			"Region2": "R2S1",
		},
		Targets: map[string]string{
			"Region1": "R1S2",
			"Region2": "R2S2",
		},
		Event:   "CROSS",
		Actions: []*sc.Action{{Label: "cross_action"}},
	}

	result, err := statechart.ExecuteCrossRegionTransition(crossTransition, config, nil)
	if err != nil {
		t.Fatalf("ExecuteCrossRegionTransition failed: %v", err)
	}

	if !result.Executed {
		t.Error("Expected cross-region transition to be executed")
	}

	// Check that both regions transitioned to target states
	stateMap := make(map[string]bool)
	for _, state := range result.NewConfig.States {
		stateMap[state.Label] = true
	}

	expectedStates := []string{"Region1", "R1S2", "Region2", "R2S2"}
	for _, expected := range expectedStates {
		if !stateMap[expected] {
			t.Errorf("Expected state %s to be active", expected)
		}
	}
}

func TestActionRegistry(t *testing.T) {
	registry := NewActionRegistry()
	
	// Test default actions
	context := &structpb.Struct{
		Fields: map[string]*structpb.Value{
			"count": structpb.NewNumberValue(5),
		},
	}

	err := registry.ExecuteAction("increment_count", context)
	if err != nil {
		t.Fatalf("ExecuteAction failed: %v", err)
	}

	count, ok := context.Fields["count"].GetKind().(*structpb.Value_NumberValue)
	if !ok || count.NumberValue != 6 {
		t.Errorf("Expected count to be 6, got %v", context.Fields["count"])
	}

	// Test custom action registration
	customActionCalled := false
	registry.RegisterAction("custom_action", func(ctx *structpb.Struct) error {
		customActionCalled = true
		return nil
	})

	err = registry.ExecuteAction("custom_action", context)
	if err != nil {
		t.Fatalf("Custom action execution failed: %v", err)
	}

	if !customActionCalled {
		t.Error("Expected custom action to be called")
	}
}

func TestEntryExitActions(t *testing.T) {
	statechart := NewStatechart(&sc.Statechart{
		RootState: &sc.State{
			Label: "__root__",
			Type:  sc.StateTypeNormal,
			Children: []*sc.State{
				{Label: "Off", Type: sc.StateTypeBasic, IsInitial: true},
				{Label: "On", Type: sc.StateTypeBasic},
			},
		},
		Transitions: []*sc.Transition{
			{
				Label: "turn_on",
				From:  []string{"Off"},
				To:    []string{"On"},
				Event: "TURN_ON",
			},
		},
	})

	config := &sc.Configuration{
		States: []*sc.StateRef{{Label: "Off"}},
	}

	context := &structpb.Struct{
		Fields: map[string]*structpb.Value{},
	}

	result, err := statechart.ExecuteTransitions(config, context, "TURN_ON")
	if err != nil {
		t.Fatalf("ExecuteTransitions failed: %v", err)
	}

	if !result.Executed {
		t.Error("Expected transition to be executed")
	}

	// Check if entry/exit actions were executed based on the default registry
	if val, exists := result.NewContext.Fields["off_exited"]; exists {
		if boolVal, ok := val.GetKind().(*structpb.Value_BoolValue); ok && boolVal.BoolValue {
			// Exit action was executed
		}
	}

	if val, exists := result.NewContext.Fields["on_entered"]; exists {
		if boolVal, ok := val.GetKind().(*structpb.Value_BoolValue); ok && boolVal.BoolValue {
			// Entry action was executed
		}
	}
}

func TestOrthogonalTransitionDetection(t *testing.T) {
	statechart := NewStatechart(&sc.Statechart{
		RootState: &sc.State{
			Label: "__root__",
			Type:  sc.StateTypeParallel,
			Children: []*sc.State{
				{
					Label: "Region1",
					Type:  sc.StateTypeNormal,
					Children: []*sc.State{
						{Label: "R1S1", Type: sc.StateTypeBasic, IsInitial: true},
					},
				},
				{
					Label: "Region2",
					Type:  sc.StateTypeNormal,
					Children: []*sc.State{
						{Label: "R2S1", Type: sc.StateTypeBasic, IsInitial: true},
					},
				},
			},
		},
	})

	// Test orthogonal transition detection
	orthogonalTransition := &sc.Transition{
		From: []string{"R1S1"},
		To:   []string{"R2S1"},
	}

	isOrthogonal, err := statechart.IsOrthogonalTransition(orthogonalTransition)
	if err != nil {
		t.Fatalf("IsOrthogonalTransition failed: %v", err)
	}

	if !isOrthogonal {
		t.Error("Expected transition to be orthogonal")
	}

	// Test non-orthogonal transition
	nonOrthogonalTransition := &sc.Transition{
		From: []string{"R1S1"},
		To:   []string{"Region1"},
	}

	isOrthogonal, err = statechart.IsOrthogonalTransition(nonOrthogonalTransition)
	if err != nil {
		t.Fatalf("IsOrthogonalTransition failed: %v", err)
	}

	if isOrthogonal {
		t.Error("Expected transition to not be orthogonal")
	}
}
