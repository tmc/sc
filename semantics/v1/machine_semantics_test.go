package semantics

import (
	"testing"

	"github.com/tmc/sc"
	"google.golang.org/protobuf/types/known/structpb"
)

func newValue(v interface{}) *structpb.Value {
	val, _ := structpb.NewValue(v)
	return val
}

func TestMachine_HistorySemantics(t *testing.T) {
	// Statechart with History
	// Root
	//  |-- Off
	//  |-- On (Composite)
	//       |-- H (Shallow History)
	//       |-- A (Initial)
	//       |-- B
	//
	// Flow:
	// 1. Start -> Off
	// 2. Off -> On (enters A by default)
	// 3. On.A -> On.B
	// 4. On -> Off (should save history: B)
	// 5. Off -> On.H (should restore B)

	scDef := &sc.Statechart{
		RootState: &sc.State{
			Label: "Root",
			Children: []*sc.State{
				{Label: "Off", IsInitial: true},
				{
					Label: "On",
					Children: []*sc.State{
						{Label: "H", HistoryType: sc.HistoryType_HISTORY_TYPE_SHALLOW, IsHistory: true},
						{Label: "A", IsInitial: true},
						{Label: "B"},
					},
				},
			},
		},
		Transitions: []*sc.Transition{
			{From: []string{"Off"}, To: []string{"On"}, Event: "TURN_ON"},
			{From: []string{"A"}, To: []string{"B"}, Event: "NEXT"},
			{From: []string{"On"}, To: []string{"Off"}, Event: "TURN_OFF"},
			{From: []string{"Off"}, To: []string{"H"}, Event: "RESUME"},
		},
	}

	semChart := NewStatechart(scDef)
	normalized, err := semChart.Normalize()
	if err != nil {
		t.Fatalf("Failed to normalize statechart: %v", err)
	}

	m, err := NewMachine(normalized, "test_hist", nil)
	if err != nil {
		t.Fatalf("Failed to create machine: %v", err)
	}

	// 1. Start setup
	if err := m.Start(); err != nil {
		t.Fatalf("Start() failed: %v", err)
	}
	checkActive(t, m, "Off")

	// 2. Turn On -> A
	if _, err := m.Step("TURN_ON"); err != nil {
		t.Fatalf("Step(TURN_ON) failed: %v", err)
	}
	checkActive(t, m, "A", "On")

	// 3. A -> B
	if _, err := m.Step("NEXT"); err != nil {
		t.Fatalf("Step(NEXT) failed: %v", err)
	}
	checkActive(t, m, "B", "On")

	// 4. Turn Off (Should save B to history of On)
	if _, err := m.Step("TURN_OFF"); err != nil {
		t.Fatalf("Step(TURN_OFF) failed: %v", err)
	}
	checkActive(t, m, "Off")

	// Verify history is saved
	hist, ok := m.Configuration.History["On"]
	if !ok {
		t.Fatalf("History for 'On' not found")
	}
	if len(hist.States) != 1 || hist.States[0].Label != "B" {
		t.Errorf("Saved history = %v, want [B]", hist.States)
	}

	// 5. Resume -> H -> B
	if _, err := m.Step("RESUME"); err != nil {
		t.Fatalf("Step(RESUME) failed: %v", err)
	}
	checkActive(t, m, "B", "On")
}

func checkActive(t *testing.T, m *MachineWrapper, expected ...string) {
	t.Helper()
	active := make(map[string]bool)
	for _, s := range m.Configuration.States {
		active[s.Label] = true
	}
	for _, e := range expected {
		if !active[e] {
			t.Errorf("Expected state %s to be active. Active: %v", e, m.Configuration.States)
		}
	}
}

func TestMachine_HistoryDefaultFallback(t *testing.T) {
	// When a history pseudostate is targeted but no history has been saved,
	// the engine should fall back to the initial state of the parent
	// composite state (not produce an empty configuration).
	scDef := &sc.Statechart{
		RootState: &sc.State{
			Label: "Root",
			Children: []*sc.State{
				{Label: "Off", IsInitial: true},
				{
					Label: "On",
					Children: []*sc.State{
						{Label: "H", HistoryType: sc.HistoryType_HISTORY_TYPE_SHALLOW, IsHistory: true},
						{Label: "A", IsInitial: true},
						{Label: "B"},
					},
				},
			},
		},
		Transitions: []*sc.Transition{
			{From: []string{"Off"}, To: []string{"H"}, Event: "RESUME"},
			{From: []string{"A"}, To: []string{"B"}, Event: "NEXT"},
			{From: []string{"On"}, To: []string{"Off"}, Event: "STOP"},
		},
	}

	semChart := NewStatechart(scDef)
	normalized, err := semChart.Normalize()
	if err != nil {
		t.Fatalf("Normalize: %v", err)
	}

	m, err := NewMachine(normalized, "test_hist_default", nil)
	if err != nil {
		t.Fatalf("NewMachine: %v", err)
	}
	if err := m.Start(); err != nil {
		t.Fatalf("Start: %v", err)
	}
	checkActive(t, m, "Off")

	// RESUME targets H with no saved history -> should enter A (initial).
	if _, err := m.Step("RESUME"); err != nil {
		t.Fatalf("Step(RESUME): %v", err)
	}
	checkActive(t, m, "A", "On")

	// Move to B, stop, resume -> should restore B.
	if _, err := m.Step("NEXT"); err != nil {
		t.Fatalf("Step(NEXT): %v", err)
	}
	checkActive(t, m, "B", "On")

	if _, err := m.Step("STOP"); err != nil {
		t.Fatalf("Step(STOP): %v", err)
	}
	checkActive(t, m, "Off")

	if _, err := m.Step("RESUME"); err != nil {
		t.Fatalf("Step(RESUME) after history: %v", err)
	}
	checkActive(t, m, "B", "On")
}

func TestMachine_DeepHistoryRestore(t *testing.T) {
	// Deep history should restore the full nested configuration.
	scDef := &sc.Statechart{
		RootState: &sc.State{
			Label: "Root",
			Children: []*sc.State{
				{Label: "Off", IsInitial: true},
				{
					Label: "On",
					Children: []*sc.State{
						{Label: "DH", HistoryType: sc.HistoryType_HISTORY_TYPE_DEEP, IsHistory: true},
						{Label: "Idle", IsInitial: true},
						{
							Label: "Working",
							Children: []*sc.State{
								{Label: "SubA", IsInitial: true},
								{Label: "SubB"},
							},
						},
					},
				},
			},
		},
		Transitions: []*sc.Transition{
			{From: []string{"Off"}, To: []string{"On"}, Event: "START"},
			{From: []string{"Idle"}, To: []string{"Working"}, Event: "WORK"},
			{From: []string{"SubA"}, To: []string{"SubB"}, Event: "ADVANCE"},
			{From: []string{"On"}, To: []string{"Off"}, Event: "STOP"},
			{From: []string{"Off"}, To: []string{"DH"}, Event: "RESUME"},
		},
	}

	semChart := NewStatechart(scDef)
	normalized, err := semChart.Normalize()
	if err != nil {
		t.Fatalf("Normalize: %v", err)
	}

	m, err := NewMachine(normalized, "test_deep_hist", nil)
	if err != nil {
		t.Fatalf("NewMachine: %v", err)
	}
	if err := m.Start(); err != nil {
		t.Fatalf("Start: %v", err)
	}

	// Navigate to On > Working > SubB.
	for _, event := range []string{"START", "WORK", "ADVANCE"} {
		if _, err := m.Step(event); err != nil {
			t.Fatalf("Step(%s): %v", event, err)
		}
	}
	checkActive(t, m, "On", "Working", "SubB")

	// Stop (saves deep history for On).
	if _, err := m.Step("STOP"); err != nil {
		t.Fatalf("Step(STOP): %v", err)
	}
	checkActive(t, m, "Off")

	// Resume via deep history -> should restore On > Working > SubB.
	if _, err := m.Step("RESUME"); err != nil {
		t.Fatalf("Step(RESUME): %v", err)
	}
	checkActive(t, m, "On", "Working", "SubB")
}

func TestMachine_ExpressionGuards(t *testing.T) {
	tests := []struct {
		name    string
		guard   *sc.Guard
		context map[string]interface{}
		event   interface{}
		want    bool
		wantErr bool
	}{
		{
			name: "CEL Basic Comparison",
			guard: &sc.Guard{
				Condition: &sc.Expression{
					Type:   sc.ExpressionTypeCEL,
					Source: "context.value > 10",
				},
			},
			context: map[string]interface{}{"value": 15},
			want:    true,
		},
		{
			name: "CEL Basic Comparison False",
			guard: &sc.Guard{
				Condition: &sc.Expression{
					Type:   sc.ExpressionTypeCEL,
					Source: "context.value > 20",
				},
			},
			context: map[string]interface{}{"value": 15},
			want:    false,
		},
		{
			name: "Starlark Comparison",
			guard: &sc.Guard{
				Condition: &sc.Expression{
					Type:   sc.ExpressionTypeStarlark,
					Source: "1 + 1 == 2",
				},
			},
			want: true,
		},
	}

	evaluator := NewGuardEvaluator()

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ctx := &EvaluationContext{
				Variables: &structpb.Struct{Fields: make(map[string]*structpb.Value)},
			}
			if tt.context != nil {
				for k, v := range tt.context {
					ctx.Variables.Fields[k] = newValue(v)
				}
			}

			result, err := evaluator.EvaluateGuard(tt.guard, ctx)
			if (err != nil) != tt.wantErr {
				t.Errorf("EvaluateGuard() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if result != nil && result.Value != tt.want {
				t.Errorf("EvaluateGuard() result = %v, want %v", result.Value, tt.want)
			}
		})
	}
}

func TestMachine_FinalState(t *testing.T) {
	// Test that transitioning to a Final state stops the machine
	scDef := &sc.Statechart{
		RootState: &sc.State{
			Label: "Root",
			Children: []*sc.State{
				{Label: "A", IsInitial: true},
				{Label: "B", IsFinal: true},
			},
		},
		Transitions: []*sc.Transition{
			{From: []string{"A"}, To: []string{"B"}, Event: "FINISH"},
		},
	}

	semChart := NewStatechart(scDef)
	normalized, err := semChart.Normalize()
	if err != nil {
		t.Fatalf("Failed to normalize: %v", err)
	}

	m, err := NewMachine(normalized, "test_final", nil)
	if err != nil {
		t.Fatalf("Failed to create machine: %v", err)
	}

	if err := m.Start(); err != nil {
		t.Fatalf("Start() failed: %v", err)
	}
	checkActive(t, m, "A")

	if _, err := m.Step("FINISH"); err != nil {
		t.Fatalf("Step(FINISH) failed: %v", err)
	}
	// Machine should be stopped now
	if !m.IsStopped() {
		t.Errorf("Machine should be stopped after reaching Final state")
	}
	if !m.IsFinal() {
		t.Errorf("Machine should be in Final configuration")
	}
}
