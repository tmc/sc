// Package validation implements the SemanticValidator service.
package validation

import (
	"context"

	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"

	"github.com/tmc/sc"
	pb "github.com/tmc/sc/gen/statecharts/v1"
	validationv1 "github.com/tmc/sc/gen/validation/v1"
)

// NewSemanticValidator creates a new SemanticValidator service.
func NewSemanticValidator() *SemanticValidator {
	return &SemanticValidator{}
}

// SemanticValidator implements the SemanticValidator service.
type SemanticValidator struct {
	// This would normally include validationv1.UnimplementedSemanticValidatorServer
	// but we'll implement directly for now
}

// ValidateChart validates a statechart.
func (s *SemanticValidator) ValidateChart(ctx context.Context, req *validationv1.ValidateChartRequest) (*validationv1.ValidateChartResponse, error) {
	chart := req.GetChart()
	if chart == nil {
		return nil, status.Error(codes.InvalidArgument, "chart is required")
	}

	// Convert to native statechart
	statechart := convertProtoToStatechart(chart)

	// Ignore rules from request
	ignoreRules := make(map[validationv1.RuleId]bool)
	for _, rule := range req.GetIgnoreRules() {
		ignoreRules[rule] = true
	}

	// Run validation rules
	violations := s.validateChart(statechart, ignoreRules)

	// Convert response
	resp := &validationv1.ValidateChartResponse{
		Violations: violations,
	}

	// Set status based on violations
	if len(violations) > 0 {
		for _, v := range violations {
			if v.Severity == validationv1.Severity_ERROR {
				resp.Status = status.New(codes.FailedPrecondition, "validation failed").Proto()
				return resp, nil
			}
		}
		resp.Status = status.New(codes.OK, "validation passed with warnings").Proto()
	} else {
		resp.Status = status.New(codes.OK, "validation passed").Proto()
	}

	return resp, nil
}

// ValidateTrace validates a statechart trace.
func (s *SemanticValidator) ValidateTrace(ctx context.Context, req *validationv1.ValidateTraceRequest) (*validationv1.ValidateTraceResponse, error) {
	chart := req.GetChart()
	if chart == nil {
		return nil, status.Error(codes.InvalidArgument, "chart is required")
	}

	// Convert to native statechart
	statechart := convertProtoToStatechart(chart)

	// Ignore rules from request
	ignoreRules := make(map[validationv1.RuleId]bool)
	for _, rule := range req.GetIgnoreRules() {
		ignoreRules[rule] = true
	}

	// Run validation rules
	violations := s.validateChart(statechart, ignoreRules)

	// Additional validation for the trace would go here
	// For now we just validate the chart

	// Convert response
	resp := &validationv1.ValidateTraceResponse{
		Violations: violations,
	}

	// Set status based on violations
	if len(violations) > 0 {
		for _, v := range violations {
			if v.Severity == validationv1.Severity_ERROR {
				resp.Status = status.New(codes.FailedPrecondition, "validation failed").Proto()
				return resp, nil
			}
		}
		resp.Status = status.New(codes.OK, "validation passed with warnings").Proto()
	} else {
		resp.Status = status.New(codes.OK, "validation passed").Proto()
	}

	return resp, nil
}

// validateChart applies all validation rules to a statechart.
func (s *SemanticValidator) validateChart(statechart *sc.Statechart, ignoreRules map[validationv1.RuleId]bool) []*validationv1.Violation {
	var violations []*validationv1.Violation

	// Apply each rule if not ignored
	if !ignoreRules[validationv1.RuleId_UNIQUE_STATE_LABELS] {
		if err := validateUniqueStateLabels(statechart); err != nil {
			violations = append(violations, &validationv1.Violation{
				Rule:     validationv1.RuleId_UNIQUE_STATE_LABELS,
				Severity: validationv1.Severity_ERROR,
				Message:  err.Error(),
			})
		}
	}

	if !ignoreRules[validationv1.RuleId_SINGLE_DEFAULT_CHILD] {
		if err := validateSingleDefaultChild(statechart); err != nil {
			violations = append(violations, &validationv1.Violation{
				Rule:     validationv1.RuleId_SINGLE_DEFAULT_CHILD,
				Severity: validationv1.Severity_ERROR,
				Message:  err.Error(),
			})
		}
	}

	if !ignoreRules[validationv1.RuleId_BASIC_HAS_NO_CHILDREN] {
		if err := validateBasicHasNoChildren(statechart); err != nil {
			violations = append(violations, &validationv1.Violation{
				Rule:     validationv1.RuleId_BASIC_HAS_NO_CHILDREN,
				Severity: validationv1.Severity_ERROR,
				Message:  err.Error(),
			})
		}
	}

	if !ignoreRules[validationv1.RuleId_COMPOUND_HAS_CHILDREN] {
		if err := validateCompoundHasChildren(statechart); err != nil {
			violations = append(violations, &validationv1.Violation{
				Rule:     validationv1.RuleId_COMPOUND_HAS_CHILDREN,
				Severity: validationv1.Severity_ERROR,
				Message:  err.Error(),
			})
		}
	}

	// Add more rules as needed

	return violations
}

// convertProtoToStatechart converts a proto statechart to a native statechart.
// This is a simplified conversion for validation purposes.
func convertProtoToStatechart(protoChart *pb.Statechart) *sc.Statechart {
	if protoChart == nil {
		return nil
	}

	statechart := &sc.Statechart{
		RootState:   convertState(protoChart.RootState),
		Transitions: make([]*sc.Transition, 0, len(protoChart.Transitions)),
		Events:      make([]*sc.Event, 0, len(protoChart.Events)),
	}

	for _, t := range protoChart.Transitions {
		statechart.Transitions = append(statechart.Transitions, convertTransition(t))
	}

	for _, e := range protoChart.Events {
		statechart.Events = append(statechart.Events, convertEvent(e))
	}

	return statechart
}

func convertState(protoState *pb.State) *sc.State {
	if protoState == nil {
		return nil
	}

	state := &sc.State{
		Label:     protoState.Label,
		Type:      sc.StateType(protoState.Type),
		IsInitial: protoState.IsInitial,
		IsFinal:   protoState.IsFinal,
		Children:  make([]*sc.State, 0, len(protoState.Children)),
	}

	for _, child := range protoState.Children {
		state.Children = append(state.Children, convertState(child))
	}

	return state
}

func convertTransition(protoTransition *pb.Transition) *sc.Transition {
	if protoTransition == nil {
		return nil
	}

	transition := &sc.Transition{
		Label: protoTransition.Label,
		From:  protoTransition.From,
		To:    protoTransition.To,
		Event: protoTransition.Event,
	}

	if protoTransition.Guard != nil {
		transition.Guard = &sc.Guard{
			Expression: protoTransition.Guard.Expression,
		}
	}

	for _, a := range protoTransition.Actions {
		transition.Actions = append(transition.Actions, &sc.Action{
			Label: a.Label,
		})
	}

	return transition
}

func convertEvent(protoEvent *pb.Event) *sc.Event {
	if protoEvent == nil {
		return nil
	}

	return &sc.Event{
		Label: protoEvent.Label,
	}
}