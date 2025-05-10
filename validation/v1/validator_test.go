package validation

import (
	"context"
	"testing"

	pb "github.com/tmc/sc/gen/statecharts/v1"
	validationv1 "github.com/tmc/sc/gen/validation/v1"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
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
			wantViolations: 1,
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
			wantViolations: 1,
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
			wantViolations: 2, // Updated to expect 2 violations (root state needs initial state too)
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
			wantViolations: 1,
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
			ignoreRules:    []validationv1.RuleId{validationv1.RuleId_SINGLE_DEFAULT_CHILD},
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