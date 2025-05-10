package semantics

import (
	"context"
	"fmt"

	"github.com/tmc/sc"
	pb "github.com/tmc/sc/gen/statecharts/v1"
	validationv1 "github.com/tmc/sc/gen/validation/v1"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

// ValidatorClient wraps a connection to the SemanticValidator service.
type ValidatorClient struct {
	client validationv1.SemanticValidatorClient
	conn   *grpc.ClientConn
}

// NewValidatorClient creates a new client connection to the SemanticValidator service.
func NewValidatorClient(target string) (*ValidatorClient, error) {
	conn, err := grpc.Dial(target, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		return nil, fmt.Errorf("failed to connect to validator: %w", err)
	}

	return &ValidatorClient{
		client: validationv1.NewSemanticValidatorClient(conn),
		conn:   conn,
	}, nil
}

// Close closes the connection to the SemanticValidator service.
func (c *ValidatorClient) Close() error {
	if c.conn != nil {
		return c.conn.Close()
	}
	return nil
}

// ValidateStatechart validates a statechart using the SemanticValidator service.
func (c *ValidatorClient) ValidateStatechart(ctx context.Context, statechart *Statechart) error {
	// Convert to proto statechart
	protoStatechart := &pb.Statechart{
		RootState:   convertStateToProto(statechart.RootState),
		Transitions: make([]*pb.Transition, 0, len(statechart.Transitions)),
		Events:      make([]*pb.Event, 0, len(statechart.Events)),
	}

	for _, t := range statechart.Transitions {
		protoStatechart.Transitions = append(protoStatechart.Transitions, convertTransitionToProto(t))
	}

	for _, e := range statechart.Events {
		protoStatechart.Events = append(protoStatechart.Events, convertEventToProto(e))
	}

	// Call the validator
	resp, err := c.client.ValidateChart(ctx, &validationv1.ValidateChartRequest{
		Chart: protoStatechart,
	})
	if err != nil {
		return fmt.Errorf("failed to validate chart: %w", err)
	}

	// Check for errors
	if len(resp.Violations) > 0 {
		errorMsg := "validation failed:"
		for _, v := range resp.Violations {
			if v.Severity == validationv1.Severity_ERROR {
				errorMsg += fmt.Sprintf("\n  - %s: %s", v.Rule, v.Message)
			}
		}
		return fmt.Errorf(errorMsg)
	}

	return nil
}

// ValidateTrace validates a statechart trace using the SemanticValidator service.
func (c *ValidatorClient) ValidateTrace(ctx context.Context, statechart *Statechart, machines []*sc.Machine) error {
	// Convert to proto statechart
	protoStatechart := &pb.Statechart{
		RootState:   convertStateToProto(statechart.RootState),
		Transitions: make([]*pb.Transition, 0, len(statechart.Transitions)),
		Events:      make([]*pb.Event, 0, len(statechart.Events)),
	}

	for _, t := range statechart.Transitions {
		protoStatechart.Transitions = append(protoStatechart.Transitions, convertTransitionToProto(t))
	}

	for _, e := range statechart.Events {
		protoStatechart.Events = append(protoStatechart.Events, convertEventToProto(e))
	}

	// Convert machines to proto machines
	protoMachines := make([]*pb.Machine, 0, len(machines))
	for _, m := range machines {
		protoMachines = append(protoMachines, convertMachineToProto(m))
	}

	// Call the validator
	resp, err := c.client.ValidateTrace(ctx, &validationv1.ValidateTraceRequest{
		Chart: protoStatechart,
		Trace: protoMachines,
	})
	if err != nil {
		return fmt.Errorf("failed to validate trace: %w", err)
	}

	// Check for errors
	if len(resp.Violations) > 0 {
		errorMsg := "validation failed:"
		for _, v := range resp.Violations {
			if v.Severity == validationv1.Severity_ERROR {
				errorMsg += fmt.Sprintf("\n  - %s: %s", v.Rule, v.Message)
			}
		}
		return fmt.Errorf(errorMsg)
	}

	return nil
}

// Helper functions to convert between proto and regular types

func convertStateToProto(state *sc.State) *pb.State {
	if state == nil {
		return nil
	}

	result := &pb.State{
		Label:     state.Label,
		Type:      pb.StateType(state.Type),
		IsInitial: state.IsInitial,
		IsFinal:   state.IsFinal,
		Children:  make([]*pb.State, 0, len(state.Children)),
	}

	for _, child := range state.Children {
		result.Children = append(result.Children, convertStateToProto(child))
	}

	return result
}

func convertTransitionToProto(transition *sc.Transition) *pb.Transition {
	if transition == nil {
		return nil
	}

	result := &pb.Transition{
		Label: transition.Label,
		From:  transition.From,
		To:    transition.To,
		Event: transition.Event,
	}

	if transition.Guard != nil {
		result.Guard = &pb.Guard{
			Expression: transition.Guard.Expression,
		}
	}

	if len(transition.Actions) > 0 {
		result.Actions = make([]*pb.Action, 0, len(transition.Actions))
		for _, action := range transition.Actions {
			result.Actions = append(result.Actions, &pb.Action{
				Label: action.Label,
			})
		}
	}

	return result
}

func convertEventToProto(event *sc.Event) *pb.Event {
	if event == nil {
		return nil
	}

	return &pb.Event{
		Label: event.Label,
	}
}

func convertMachineToProto(machine *sc.Machine) *pb.Machine {
	if machine == nil {
		return nil
	}

	// This is a simplified conversion - a full implementation would convert
	// all fields including step history, etc.
	result := &pb.Machine{
		Id:    machine.Id,
		State: pb.MachineState(machine.State),
	}

	return result
}

// Validate calls the validator service to validate this statechart.
// This method can replace the current Validate method in the Statechart struct
// once the validator service is deployed.
func (s *Statechart) ValidateWithService(ctx context.Context, validatorClient *ValidatorClient) error {
	return validatorClient.ValidateStatechart(ctx, s)
}