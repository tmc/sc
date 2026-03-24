package validation

import (
	"context"
	"testing"

	pb "github.com/tmc/sc/gen/statecharts/v1"
	validationv1 "github.com/tmc/sc/gen/validation/v1"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
	"google.golang.org/protobuf/types/known/structpb"
)

func TestValidateChart(t *testing.T) {
	validator := NewSemanticValidator()

	tests := []struct {
		name           string
		chart          *pb.Statechart
		ignoreRules    []validationv1.RuleId
		wantViolations int
		wantCode       codes.Code
	}{
		{
			name: "Valid statechart",
			chart: &pb.Statechart{
				RootState: &pb.State{
					Label: "__root__",
					Type:  pb.StateType_STATE_TYPE_NORMAL,
					Children: []*pb.State{
						{
							Label:     "A",
							Type:      pb.StateType_STATE_TYPE_BASIC,
							IsInitial: true,
						},
						{
							Label: "B",
							Type:  pb.StateType_STATE_TYPE_BASIC,
						},
					},
				},
				Transitions: []*pb.Transition{
					{
						Label: "t1",
						From:  []string{"A"},
						To:    []string{"B"},
						Event: "e1",
					},
				},
				Events: []*pb.Event{
					{Label: "e1"},
				},
			},
			wantViolations: 0,
			wantCode:       codes.OK,
		},
		{
			name: "Invalid statechart - duplicate state labels",
			chart: &pb.Statechart{
				RootState: &pb.State{
					Label: "__root__",
					Type:  pb.StateType_STATE_TYPE_NORMAL,
					Children: []*pb.State{
						{
							Label:     "A",
							Type:      pb.StateType_STATE_TYPE_BASIC,
							IsInitial: true,
						},
						{
							Label: "A", // Duplicate label
							Type:  pb.StateType_STATE_TYPE_BASIC,
						},
					},
				},
			},
			wantViolations: 2,
			wantCode:       codes.FailedPrecondition,
		},
		{
			name: "Invalid statechart - basic state with children",
			chart: &pb.Statechart{
				RootState: &pb.State{
					Label: "__root__",
					Type:  pb.StateType_STATE_TYPE_NORMAL,
					Children: []*pb.State{
						{
							Label:     "A",
							Type:      pb.StateType_STATE_TYPE_BASIC,
							IsInitial: true,
							Children: []*pb.State{ // Basic state shouldn't have children
								{
									Label: "A1",
									Type:  pb.StateType_STATE_TYPE_BASIC,
								},
							},
						},
					},
				},
			},
			wantViolations: 2,
			wantCode:       codes.FailedPrecondition,
		},
		{
			name: "Invalid statechart - compound state without children",
			chart: &pb.Statechart{
				RootState: &pb.State{
					Label: "__root__",
					Type:  pb.StateType_STATE_TYPE_NORMAL,
					Children: []*pb.State{
						{
							Label:     "A",
							Type:      pb.StateType_STATE_TYPE_NORMAL, // Compound state without children
							IsInitial: true,
						},
					},
				},
			},
			wantViolations: 3,
			wantCode:       codes.FailedPrecondition,
		},
		{
			name: "Invalid statechart - multiple default states",
			chart: &pb.Statechart{
				RootState: &pb.State{
					Label: "__root__",
					Type:  pb.StateType_STATE_TYPE_NORMAL,
					Children: []*pb.State{
						{
							Label:     "A",
							Type:      pb.StateType_STATE_TYPE_BASIC,
							IsInitial: true,
						},
						{
							Label:     "B",
							Type:      pb.StateType_STATE_TYPE_BASIC,
							IsInitial: true, // Second default state
						},
					},
				},
			},
			wantViolations: 2,
			wantCode:       codes.FailedPrecondition,
		},
		{
			name: "Ignored rule",
			chart: &pb.Statechart{
				RootState: &pb.State{
					Label: "__root__",
					Type:  pb.StateType_STATE_TYPE_NORMAL,
					Children: []*pb.State{
						{
							Label:     "A",
							Type:      pb.StateType_STATE_TYPE_BASIC,
							IsInitial: true,
						},
						{
							Label:     "B",
							Type:      pb.StateType_STATE_TYPE_BASIC,
							IsInitial: true, // Second default state
						},
					},
				},
			},
			ignoreRules: []validationv1.RuleId{
				validationv1.RuleId_SINGLE_DEFAULT_CHILD,
				validationv1.RuleId_RULE_UNSPECIFIED,
				validationv1.RuleId_COMPLETION_TRANSITIONS_VALID, // ConfigurationConsistency
				validationv1.RuleId_HISTORY_DEFAULTS_VALID,       // InitialStateExists
			},
			wantViolations: 0,
			wantCode:       codes.OK,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			req := &validationv1.ValidateChartRequest{
				Chart:       tt.chart,
				IgnoreRules: tt.ignoreRules,
			}

			resp, err := validator.ValidateChart(context.Background(), req)
			if err != nil {
				t.Fatalf("ValidateChart() error = %v", err)
			}

			if len(resp.Violations) != tt.wantViolations {
				t.Errorf("ValidateChart() got %d violations, want %d", len(resp.Violations), tt.wantViolations)
			}

			s := status.FromProto(resp.Status)
			if s.Code() != tt.wantCode {
				t.Errorf("ValidateChart() got status code %v, want %v", s.Code(), tt.wantCode)
			}
		})
	}
}

func TestValidateTrace(t *testing.T) {
	validator := NewSemanticValidator()

	// Create a valid chart for testing
	validChart := &pb.Statechart{
		RootState: &pb.State{
			Label: "__root__",
			Type:  pb.StateType_STATE_TYPE_NORMAL,
			Children: []*pb.State{
				{
					Label:     "A",
					Type:      pb.StateType_STATE_TYPE_BASIC,
					IsInitial: true,
				},
				{
					Label: "B",
					Type:  pb.StateType_STATE_TYPE_BASIC,
				},
			},
		},
		Transitions: []*pb.Transition{
			{
				Label: "t1",
				From:  []string{"A"},
				To:    []string{"B"},
				Event: "e1",
			},
		},
		Events: []*pb.Event{
			{Label: "e1"},
		},
	}

	// Create test trace
	trace := []*pb.Machine{
		{
			Id:    "m1",
			State: pb.MachineState_MACHINE_STATE_RUNNING,
			StepHistory: []*pb.Step{
				{
					Transitions: []*pb.Transition{
						{
							Label: "t1",
							From:  []string{"A"},
							To:    []string{"B"},
							Event: "e1",
						},
					},
				},
			},
		},
	}

	req := &validationv1.ValidateTraceRequest{
		Chart: validChart,
		Trace: trace,
	}

	resp, err := validator.ValidateTrace(context.Background(), req)
	if err != nil {
		t.Fatalf("ValidateTrace() error = %v", err)
	}

	// For now, we're just validating the chart, so we expect the same
	// results as a chart validation
	if len(resp.Violations) != 0 {
		t.Errorf("ValidateTrace() got %d violations, want 0", len(resp.Violations))
	}

	s := status.FromProto(resp.Status)
	if s.Code() != codes.OK {
		t.Errorf("ValidateTrace() got status code %v, want %v", s.Code(), codes.OK)
	}
}

func TestValidateTrace_RejectsUndeclaredEventInStep(t *testing.T) {
	validator := NewSemanticValidator()

	chart := &pb.Statechart{
		RootState: &pb.State{
			Label: "__root__",
			Type:  pb.StateType_STATE_TYPE_NORMAL,
			Children: []*pb.State{
				{Label: "A", Type: pb.StateType_STATE_TYPE_BASIC, IsInitial: true},
				{Label: "B", Type: pb.StateType_STATE_TYPE_BASIC},
			},
		},
		Transitions: []*pb.Transition{
			{Label: "t1", From: []string{"A"}, To: []string{"B"}, Event: "e1"},
		},
		Events: []*pb.Event{
			{Label: "e1"},
		},
	}

	trace := []*pb.Machine{
		{
			StepHistory: []*pb.Step{
				{
					Events: []*pb.Event{
						{Label: "unknown"},
					},
				},
			},
		},
	}

	resp, err := validator.ValidateTrace(context.Background(), &validationv1.ValidateTraceRequest{
		Chart: chart,
		Trace: trace,
	})
	if err != nil {
		t.Fatalf("ValidateTrace() error = %v", err)
	}

	if len(resp.Violations) == 0 {
		t.Fatal("ValidateTrace() got 0 violations, want > 0")
	}

	s := status.FromProto(resp.Status)
	if s.Code() != codes.FailedPrecondition {
		t.Fatalf("ValidateTrace() status code = %v, want %v", s.Code(), codes.FailedPrecondition)
	}
}

func TestValidateTrace_RejectsUnknownStateInConfiguration(t *testing.T) {
	validator := NewSemanticValidator()

	chart := &pb.Statechart{
		RootState: &pb.State{
			Label: "__root__",
			Type:  pb.StateType_STATE_TYPE_NORMAL,
			Children: []*pb.State{
				{Label: "A", Type: pb.StateType_STATE_TYPE_BASIC, IsInitial: true},
			},
		},
	}

	trace := []*pb.Machine{
		{
			Configuration: &pb.Configuration{
				States: []*pb.StateRef{
					{Label: "UNKNOWN"},
				},
			},
		},
	}

	resp, err := validator.ValidateTrace(context.Background(), &validationv1.ValidateTraceRequest{
		Chart: chart,
		Trace: trace,
	})
	if err != nil {
		t.Fatalf("ValidateTrace() error = %v", err)
	}

	if len(resp.Violations) == 0 {
		t.Fatal("ValidateTrace() got 0 violations, want > 0")
	}

	s := status.FromProto(resp.Status)
	if s.Code() != codes.FailedPrecondition {
		t.Fatalf("ValidateTrace() status code = %v, want %v", s.Code(), codes.FailedPrecondition)
	}
}

func TestValidateTrace_RejectsTransitionNotInChart(t *testing.T) {
	validator := NewSemanticValidator()

	chart := &pb.Statechart{
		RootState: &pb.State{
			Label: "__root__",
			Type:  pb.StateType_STATE_TYPE_NORMAL,
			Children: []*pb.State{
				{Label: "A", Type: pb.StateType_STATE_TYPE_BASIC, IsInitial: true},
				{Label: "B", Type: pb.StateType_STATE_TYPE_BASIC},
			},
		},
		Transitions: []*pb.Transition{
			{Label: "t1", From: []string{"A"}, To: []string{"B"}, Event: "e1"},
		},
		Events: []*pb.Event{
			{Label: "e1"},
			{Label: "e2"},
		},
	}

	trace := []*pb.Machine{
		{
			StepHistory: []*pb.Step{
				{
					Transitions: []*pb.Transition{
						{
							From:  []string{"A"},
							To:    []string{"B"},
							Event: "e2",
						},
					},
				},
			},
		},
	}

	resp, err := validator.ValidateTrace(context.Background(), &validationv1.ValidateTraceRequest{
		Chart: chart,
		Trace: trace,
	})
	if err != nil {
		t.Fatalf("ValidateTrace() error = %v", err)
	}
	if len(resp.Violations) == 0 {
		t.Fatal("ValidateTrace() got 0 violations, want > 0")
	}
	s := status.FromProto(resp.Status)
	if s.Code() != codes.FailedPrecondition {
		t.Fatalf("ValidateTrace() status code = %v, want %v", s.Code(), codes.FailedPrecondition)
	}
}

func TestValidateTrace_RejectsMismatchedTransitionLabel(t *testing.T) {
	validator := NewSemanticValidator()

	chart := &pb.Statechart{
		RootState: &pb.State{
			Label: "__root__",
			Type:  pb.StateType_STATE_TYPE_NORMAL,
			Children: []*pb.State{
				{Label: "A", Type: pb.StateType_STATE_TYPE_BASIC, IsInitial: true},
				{Label: "B", Type: pb.StateType_STATE_TYPE_BASIC},
			},
		},
		Transitions: []*pb.Transition{
			{Label: "t1", From: []string{"A"}, To: []string{"B"}, Event: "e1"},
		},
		Events: []*pb.Event{
			{Label: "e1"},
		},
	}

	trace := []*pb.Machine{
		{
			StepHistory: []*pb.Step{
				{
					Transitions: []*pb.Transition{
						{
							Label: "t2",
							From:  []string{"A"},
							To:    []string{"B"},
							Event: "e1",
						},
					},
				},
			},
		},
	}

	resp, err := validator.ValidateTrace(context.Background(), &validationv1.ValidateTraceRequest{
		Chart: chart,
		Trace: trace,
	})
	if err != nil {
		t.Fatalf("ValidateTrace() error = %v", err)
	}
	if len(resp.Violations) == 0 {
		t.Fatal("ValidateTrace() got 0 violations, want > 0")
	}
	s := status.FromProto(resp.Status)
	if s.Code() != codes.FailedPrecondition {
		t.Fatalf("ValidateTrace() status code = %v, want %v", s.Code(), codes.FailedPrecondition)
	}
}

func TestValidateChart_MapsEventConsistencyRuleID(t *testing.T) {
	validator := NewSemanticValidator()

	chart := &pb.Statechart{
		RootState: &pb.State{
			Label: "__root__",
			Type:  pb.StateType_STATE_TYPE_NORMAL,
			Children: []*pb.State{
				{Label: "A", Type: pb.StateType_STATE_TYPE_BASIC, IsInitial: true},
				{Label: "B", Type: pb.StateType_STATE_TYPE_BASIC},
			},
		},
		Transitions: []*pb.Transition{
			{Label: "t1", From: []string{"A"}, To: []string{"B"}, Event: "UNKNOWN"},
		},
		Events: []*pb.Event{
			{Label: "KNOWN"},
		},
	}

	resp, err := validator.ValidateChart(context.Background(), &validationv1.ValidateChartRequest{
		Chart: chart,
	})
	if err != nil {
		t.Fatalf("ValidateChart() error = %v", err)
	}

	found := false
	for _, v := range resp.Violations {
		if v.Rule == validationv1.RuleId_EVENT_PARAMETERS_CONSISTENT {
			found = true
			break
		}
	}
	if !found {
		t.Fatalf("expected at least one violation with rule %v, got %+v", validationv1.RuleId_EVENT_PARAMETERS_CONSISTENT, resp.Violations)
	}
}

func TestConvertProtoToStatechartPreservesFields(t *testing.T) {
	variables, err := structpb.NewStruct(map[string]interface{}{
		"answer": 42,
	})
	if err != nil {
		t.Fatalf("structpb.NewStruct() error = %v", err)
	}

	chart := &pb.Statechart{
		Name:        "history-chart",
		Description: "preserve all modeled fields",
		Variables:   variables,
		RootState: &pb.State{
			Label: "root",
			Type:  pb.StateType_STATE_TYPE_OR,
			Children: []*pb.State{
				{
					Label:       "H",
					Type:        pb.StateType_STATE_TYPE_BASIC,
					IsHistory:   true,
					HistoryType: pb.HistoryType_HISTORY_TYPE_SHALLOW,
				},
				{
					Label:     "A",
					Type:      pb.StateType_STATE_TYPE_BASIC,
					IsInitial: true,
				},
			},
		},
		Transitions: []*pb.Transition{
			{
				Label:    "t1",
				From:     []string{"A"},
				To:       []string{"H"},
				Event:    "resume",
				Priority: 9,
			},
		},
		Events: []*pb.Event{
			{
				Label: "resume",
			},
		},
	}

	got := convertProtoToStatechart(chart)
	if got == nil {
		t.Fatal("convertProtoToStatechart() = nil, want non-nil")
	}
	if got.Name != "history-chart" || got.Description != "preserve all modeled fields" {
		t.Fatalf("convertProtoToStatechart() lost chart metadata: %+v", got)
	}
	if got.Variables == nil || got.Variables.Fields["answer"].GetNumberValue() != 42 {
		t.Fatalf("convertProtoToStatechart() variables = %#v, want answer=42", got.Variables)
	}
	if len(got.RootState.Children) != 2 || !got.RootState.Children[0].IsHistory {
		t.Fatalf("convertProtoToStatechart() root children = %#v, want history child preserved", got.RootState.Children)
	}
	if got.Transitions[0].Priority != 9 || got.Transitions[0].To[0] != "H" {
		t.Fatalf("convertProtoToStatechart() transition = %#v, want priority and target preserved", got.Transitions[0])
	}
}

func TestValidateChart_ReconciledConstraintWarning(t *testing.T) {
	validator := NewSemanticValidator()

	chart := &pb.Statechart{
		RootState: &pb.State{
			Label: "__root__",
			Type:  pb.StateType_STATE_TYPE_NORMAL,
			Children: []*pb.State{
				{Label: "A", Type: pb.StateType_STATE_TYPE_BASIC, IsInitial: true},
				{Label: "B", Type: pb.StateType_STATE_TYPE_BASIC},
			},
		},
		Transitions: []*pb.Transition{
			{
				Label: "t1",
				From:  []string{"A"},
				To:    []string{"B"},
				Event: "e",
				Actions: []*pb.Action{
					{Label: "raise:i"},
					{Label: "raise:j"},
				},
			},
		},
		Events: []*pb.Event{
			{Label: "e"},
		},
	}

	resp, err := validator.ValidateChart(context.Background(), &validationv1.ValidateChartRequest{
		Chart: chart,
	})
	if err != nil {
		t.Fatalf("ValidateChart() error = %v", err)
	}

	found := false
	for _, violation := range resp.Violations {
		if violation.Rule == validationv1.RuleId_RECONCILING_C14_SINGLE_GENERATED_EVENT {
			found = true
			if violation.Severity != validationv1.Severity_WARNING {
				t.Fatalf("reconciled warning severity = %v, want %v", violation.Severity, validationv1.Severity_WARNING)
			}
		}
	}
	if !found {
		t.Fatalf("expected reconciled C14 warning, got %+v", resp.Violations)
	}

	if code := status.FromProto(resp.Status).Code(); code != codes.OK {
		t.Fatalf("ValidateChart() status code = %v, want %v", code, codes.OK)
	}
}

func TestValidateChart_IgnoreReconciledConstraint(t *testing.T) {
	validator := NewSemanticValidator()

	chart := &pb.Statechart{
		RootState: &pb.State{
			Label: "__root__",
			Type:  pb.StateType_STATE_TYPE_NORMAL,
			Children: []*pb.State{
				{Label: "A", Type: pb.StateType_STATE_TYPE_BASIC, IsInitial: true},
				{Label: "B", Type: pb.StateType_STATE_TYPE_BASIC},
			},
		},
		Transitions: []*pb.Transition{
			{
				Label: "t1",
				From:  []string{"A"},
				To:    []string{"B"},
				Event: "e",
				Actions: []*pb.Action{
					{Label: "raise:i"},
					{Label: "raise:j"},
				},
			},
		},
		Events: []*pb.Event{
			{Label: "e"},
		},
	}

	resp, err := validator.ValidateChart(context.Background(), &validationv1.ValidateChartRequest{
		Chart:       chart,
		IgnoreRules: []validationv1.RuleId{validationv1.RuleId_RECONCILING_C14_SINGLE_GENERATED_EVENT},
	})
	if err != nil {
		t.Fatalf("ValidateChart() error = %v", err)
	}

	for _, violation := range resp.Violations {
		if violation.Rule == validationv1.RuleId_RECONCILING_C14_SINGLE_GENERATED_EVENT {
			t.Fatalf("unexpected reconciled C14 warning after ignore: %+v", resp.Violations)
		}
	}
}

func TestValidateChart_HistoryStateValidationRuleID(t *testing.T) {
	validator := NewSemanticValidator()

	chart := &pb.Statechart{
		RootState: &pb.State{
			Label: "__root__",
			Type:  pb.StateType_STATE_TYPE_NORMAL,
			Children: []*pb.State{
				{Label: "Off", Type: pb.StateType_STATE_TYPE_BASIC, IsInitial: true},
				{
					Label: "On",
					Type:  pb.StateType_STATE_TYPE_NORMAL,
					Children: []*pb.State{
						{
							Label:       "H",
							Type:        pb.StateType_STATE_TYPE_BASIC,
							IsHistory:   true,
							HistoryType: pb.HistoryType_HISTORY_TYPE_DEEP,
						},
						{
							Label:     "Idle",
							Type:      pb.StateType_STATE_TYPE_BASIC,
							IsInitial: true,
						},
					},
				},
			},
		},
	}

	resp, err := validator.ValidateChart(context.Background(), &validationv1.ValidateChartRequest{
		Chart: chart,
	})
	if err != nil {
		t.Fatalf("ValidateChart() error = %v", err)
	}

	found := false
	for _, violation := range resp.Violations {
		if violation.Rule == validationv1.RuleId_HISTORY_STATES_WELL_FORMED {
			found = true
			break
		}
	}
	if !found {
		t.Fatalf("expected history validation rule, got %+v", resp.Violations)
	}
}

func TestValidateChart_EmitsMultipleViolationsForSameRule(t *testing.T) {
	validator := NewSemanticValidator()

	chart := &pb.Statechart{
		RootState: &pb.State{
			Label: "__root__",
			Type:  pb.StateType_STATE_TYPE_NORMAL,
			Children: []*pb.State{
				{
					Label:     "A",
					Type:      pb.StateType_STATE_TYPE_BASIC,
					IsInitial: true,
					Children: []*pb.State{
						{Label: "A1", Type: pb.StateType_STATE_TYPE_BASIC},
					},
				},
				{
					Label: "B",
					Type:  pb.StateType_STATE_TYPE_BASIC,
					Children: []*pb.State{
						{Label: "B1", Type: pb.StateType_STATE_TYPE_BASIC},
					},
				},
			},
		},
	}

	resp, err := validator.ValidateChart(context.Background(), &validationv1.ValidateChartRequest{
		Chart: chart,
	})
	if err != nil {
		t.Fatalf("ValidateChart() error = %v", err)
	}

	var got []*validationv1.Violation
	for _, violation := range resp.Violations {
		if violation.Rule == validationv1.RuleId_BASIC_HAS_NO_CHILDREN {
			got = append(got, violation)
		}
	}
	if len(got) != 2 {
		t.Fatalf("BASIC_HAS_NO_CHILDREN violations = %d, want 2: %+v", len(got), resp.Violations)
	}
	for _, violation := range got {
		if len(violation.Xpath) == 0 {
			t.Fatalf("violation %+v missing xpath details", violation)
		}
	}
}

func TestValidateChart_DeterministicAndBroadcastRules(t *testing.T) {
	validator := NewSemanticValidator()

	chart := &pb.Statechart{
		RootState: &pb.State{
			Label: "__root__",
			Type:  pb.StateType_STATE_TYPE_NORMAL,
			Children: []*pb.State{
				{Label: "A", Type: pb.StateType_STATE_TYPE_BASIC, IsInitial: true},
				{Label: "B", Type: pb.StateType_STATE_TYPE_BASIC},
				{Label: "C", Type: pb.StateType_STATE_TYPE_BASIC},
			},
		},
		Transitions: []*pb.Transition{
			{Label: "t1", From: []string{"A"}, To: []string{"B"}, Event: "e", Actions: []*pb.Action{{Label: "raise:f"}}},
			{Label: "t2", From: []string{"A"}, To: []string{"C"}, Event: "e"},
			{Label: "t3", From: []string{"B"}, To: []string{"A"}, Event: "f", Actions: []*pb.Action{{Label: "raise:e"}}},
		},
		Events: []*pb.Event{
			{Label: "e"},
			{Label: "f"},
		},
	}

	resp, err := validator.ValidateChart(context.Background(), &validationv1.ValidateChartRequest{
		Chart: chart,
	})
	if err != nil {
		t.Fatalf("ValidateChart() error = %v", err)
	}

	foundDeterministic := false
	foundBroadcast := false
	for _, violation := range resp.Violations {
		switch violation.Rule {
		case validationv1.RuleId_DETERMINISTIC_TRANSITION_SELECTION:
			foundDeterministic = true
			if len(violation.Xpath) == 0 {
				t.Fatalf("deterministic violation missing xpath: %+v", violation)
			}
		case validationv1.RuleId_NO_EVENT_BROADCAST_CYCLES:
			foundBroadcast = true
			if len(violation.Xpath) == 0 {
				t.Fatalf("broadcast-cycle violation missing xpath: %+v", violation)
			}
		}
	}
	if !foundDeterministic {
		t.Fatalf("expected deterministic-transition violation, got %+v", resp.Violations)
	}
	if !foundBroadcast {
		t.Fatalf("expected event-broadcast-cycle violation, got %+v", resp.Violations)
	}
}

func TestValidateChart_TimeoutEventsUnique(t *testing.T) {
	validator := NewSemanticValidator()

	chart := &pb.Statechart{
		RootState: &pb.State{
			Label: "__root__",
			Type:  pb.StateType_STATE_TYPE_NORMAL,
			Children: []*pb.State{
				{Label: "A", Type: pb.StateType_STATE_TYPE_BASIC, IsInitial: true},
			},
		},
		Events: []*pb.Event{
			{Label: "after:500ms"},
			{Label: "after:500ms"},
			{Label: "after:1s"},
		},
	}

	resp, err := validator.ValidateChart(context.Background(), &validationv1.ValidateChartRequest{
		Chart: chart,
	})
	if err != nil {
		t.Fatalf("ValidateChart() error = %v", err)
	}

	found := false
	for _, violation := range resp.Violations {
		if violation.Rule == validationv1.RuleId_TIMEOUT_EVENTS_UNIQUE {
			found = true
			break
		}
	}
	if !found {
		t.Fatalf("expected TIMEOUT_EVENTS_UNIQUE violation, got %+v", resp.Violations)
	}
}

func TestValidateChart_StructuredGuardDoesNotRequireLegacyExpression(t *testing.T) {
	validator := NewSemanticValidator()

	chart := &pb.Statechart{
		RootState: &pb.State{
			Label: "__root__",
			Type:  pb.StateType_STATE_TYPE_NORMAL,
			Children: []*pb.State{
				{Label: "A", Type: pb.StateType_STATE_TYPE_BASIC, IsInitial: true},
				{Label: "B", Type: pb.StateType_STATE_TYPE_BASIC},
			},
		},
		Transitions: []*pb.Transition{
			{
				Label: "t1",
				From:  []string{"A"},
				To:    []string{"B"},
				Event: "e1",
				Guard: &pb.Guard{
					Condition: &pb.Expression{
						Type:   pb.ExpressionType_EXPRESSION_TYPE_RAW,
						Source: "context.ready",
					},
				},
			},
		},
		Events: []*pb.Event{
			{Label: "e1"},
		},
	}

	resp, err := validator.ValidateChart(context.Background(), &validationv1.ValidateChartRequest{
		Chart: chart,
	})
	if err != nil {
		t.Fatalf("ValidateChart() error = %v", err)
	}

	for _, violation := range resp.Violations {
		if violation.Rule == validationv1.RuleId_GUARD_EXPRESSIONS_VALID {
			t.Fatalf("unexpected guard-expression violation for structured guard: %+v", resp.Violations)
		}
	}
}
