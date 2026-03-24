package semantics

import (
	"context"
	"fmt"

	"github.com/tmc/sc"
	pb "github.com/tmc/sc/gen/statecharts/v1"
	validationv1 "github.com/tmc/sc/gen/validation/v1"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/protobuf/proto"
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
		err := c.conn.Close()
		c.conn = nil // Ensure subsequent closes are safe
		return err
	}
	return nil
}

// ValidateStatechart validates a statechart using the SemanticValidator service.
func (c *ValidatorClient) ValidateStatechart(ctx context.Context, statechart *Statechart) error {
	if c.client == nil {
		return fmt.Errorf("validator client is not initialized")
	}
	protoStatechart := convertStatechartToProto(statechart.Statechart)

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
		return fmt.Errorf("%s", errorMsg)
	}

	return nil
}

// ValidateTrace validates a statechart trace using the SemanticValidator service.
func (c *ValidatorClient) ValidateTrace(ctx context.Context, statechart *Statechart, machines []*sc.Machine) error {
	if c.client == nil {
		return fmt.Errorf("validator client is not initialized")
	}
	protoStatechart := convertStatechartToProto(statechart.Statechart)

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
		return fmt.Errorf("%s", errorMsg)
	}

	return nil
}

// Helper functions to convert between proto and regular types

func convertStatechartToProto(statechart *sc.Statechart) *pb.Statechart {
	if statechart == nil {
		return nil
	}
	return proto.Clone(statechart).(*pb.Statechart)
}

func convertStateToProto(state *sc.State) *pb.State {
	if state == nil {
		return nil
	}
	return proto.Clone(state).(*pb.State)
}

func convertTransitionToProto(transition *sc.Transition) *pb.Transition {
	if transition == nil {
		return nil
	}
	return proto.Clone(transition).(*pb.Transition)
}

func convertEventToProto(event *sc.Event) *pb.Event {
	if event == nil {
		return nil
	}
	return proto.Clone(event).(*pb.Event)
}

func convertMachineToProto(machine *sc.Machine) *pb.Machine {
	if machine == nil {
		return nil
	}
	return proto.Clone(machine).(*pb.Machine)
}

// Validate calls the validator service to validate this statechart.
// This method can replace the current Validate method in the Statechart struct
// once the validator service is deployed.
func (s *Statechart) ValidateWithService(ctx context.Context, validatorClient *ValidatorClient) error {
	return validatorClient.ValidateStatechart(ctx, s)
}
