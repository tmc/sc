// Package validation implements the SemanticValidator service.
package validation

import (
	"context"
	"fmt"
	"sort"
	"strings"

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

	// Validate trace semantics against the chart.
	traceViolations := s.validateTrace(req.GetTrace(), statechart, ignoreRules)
	violations = append(violations, traceViolations...)

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

func (s *SemanticValidator) validateTrace(trace []*sc.Machine, statechart *sc.Statechart, ignoreRules map[validationv1.RuleId]bool) []*validationv1.Violation {
	var violations []*validationv1.Violation

	// Backward compatibility: allow callers to skip unspecified extra checks.
	if ignoreRules[validationv1.RuleId_RULE_UNSPECIFIED] {
		return violations
	}

	declaredEvents := make(map[string]bool)
	for _, e := range statechart.Events {
		if e == nil || e.Label == "" {
			continue
		}
		declaredEvents[e.Label] = true
	}

	stateNames := make(map[string]bool)
	var collectStates func(*sc.State)
	collectStates = func(state *sc.State) {
		if state == nil {
			return
		}
		stateNames[state.Label] = true
		for _, child := range state.Children {
			collectStates(child)
		}
	}
	collectStates(statechart.RootState)

	validTransitionsByStructure := make(map[string]bool)
	validTransitionsByLabel := make(map[string]bool)
	for _, transition := range statechart.Transitions {
		if transition == nil {
			continue
		}
		validTransitionsByStructure[transitionStructureKey(transition)] = true
		if transition.Label != "" {
			validTransitionsByLabel[transitionLabelKey(transition)] = true
		}
	}

	validateConfig := func(machineIndex int, where string, cfg *sc.Configuration) {
		if cfg == nil {
			return
		}
		for _, ref := range cfg.States {
			if ref == nil || ref.Label == "" {
				continue
			}
			if !stateNames[ref.Label] {
				violations = append(violations, &validationv1.Violation{
					Rule:     validationv1.RuleId_RULE_UNSPECIFIED,
					Severity: validationv1.Severity_ERROR,
					Message:  fmt.Sprintf("trace machine %d %s references unknown state %q", machineIndex, where, ref.Label),
				})
			}
		}
	}

	for machineIndex, machine := range trace {
		if machine == nil {
			continue
		}

		validateConfig(machineIndex, "configuration", machine.Configuration)

		for stepIndex, step := range machine.StepHistory {
			if step == nil {
				continue
			}

			for _, event := range step.Events {
				if event == nil || event.Label == "" {
					continue
				}
				if !declaredEvents[event.Label] {
					violations = append(violations, &validationv1.Violation{
						Rule:     validationv1.RuleId_RULE_UNSPECIFIED,
						Severity: validationv1.Severity_ERROR,
						Message:  fmt.Sprintf("trace machine %d step %d uses undeclared event %q", machineIndex, stepIndex, event.Label),
					})
				}
			}

			validateConfig(machineIndex, fmt.Sprintf("step %d starting_configuration", stepIndex), step.StartingConfiguration)
			validateConfig(machineIndex, fmt.Sprintf("step %d resulting_configuration", stepIndex), step.ResultingConfiguration)

			for _, transition := range step.Transitions {
				if transition == nil {
					continue
				}
				if transition.Event != "" && !declaredEvents[transition.Event] {
					violations = append(violations, &validationv1.Violation{
						Rule:     validationv1.RuleId_RULE_UNSPECIFIED,
						Severity: validationv1.Severity_ERROR,
						Message:  fmt.Sprintf("trace machine %d step %d transition uses undeclared event %q", machineIndex, stepIndex, transition.Event),
					})
				}
				for _, from := range transition.From {
					if from != "" && !stateNames[from] {
						violations = append(violations, &validationv1.Violation{
							Rule:     validationv1.RuleId_RULE_UNSPECIFIED,
							Severity: validationv1.Severity_ERROR,
							Message:  fmt.Sprintf("trace machine %d step %d transition source references unknown state %q", machineIndex, stepIndex, from),
						})
					}
				}
				for _, to := range transition.To {
					if to != "" && !stateNames[to] {
						violations = append(violations, &validationv1.Violation{
							Rule:     validationv1.RuleId_RULE_UNSPECIFIED,
							Severity: validationv1.Severity_ERROR,
							Message:  fmt.Sprintf("trace machine %d step %d transition target references unknown state %q", machineIndex, stepIndex, to),
						})
					}
				}

				if transition.Label != "" {
					if !validTransitionsByLabel[transitionLabelKey(transition)] {
						violations = append(violations, &validationv1.Violation{
							Rule:     validationv1.RuleId_RULE_UNSPECIFIED,
							Severity: validationv1.Severity_ERROR,
							Message:  fmt.Sprintf("trace machine %d step %d transition %q does not exist in chart", machineIndex, stepIndex, transition.Label),
						})
					}
					continue
				}
				if !validTransitionsByStructure[transitionStructureKey(transition)] {
					violations = append(violations, &validationv1.Violation{
						Rule:     validationv1.RuleId_RULE_UNSPECIFIED,
						Severity: validationv1.Severity_ERROR,
						Message:  fmt.Sprintf("trace machine %d step %d transition does not exist in chart (event=%q from=%v to=%v)", machineIndex, stepIndex, transition.Event, transition.From, transition.To),
					})
				}
			}
		}
	}

	return violations
}

func transitionStructureKey(t *sc.Transition) string {
	return strings.Join(sortedCopy(t.From), ",") + "->" +
		strings.Join(sortedCopy(t.To), ",") + "[" + t.Event + "]"
}

func transitionLabelKey(t *sc.Transition) string {
	return t.Label + "|" + transitionStructureKey(t)
}

func sortedCopy(items []string) []string {
	out := append([]string(nil), items...)
	sort.Strings(out)
	return out
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

	// Run comprehensive Harel validation rules
	harelViolations := s.validateHarelRules(statechart, ignoreRules)
	violations = append(violations, harelViolations...)

	return violations
}

// validateHarelRules applies comprehensive formal validation rules based on Harel's semantics
func (s *SemanticValidator) validateHarelRules(statechart *sc.Statechart, ignoreRules map[validationv1.RuleId]bool) []*validationv1.Violation {
	var violations []*validationv1.Violation

	// Skip Harel rules if RULE_UNSPECIFIED is being ignored (for backward compatibility)
	if ignoreRules[validationv1.RuleId_RULE_UNSPECIFIED] {
		return violations
	}

	// Get selected Harel validation rules that don't duplicate existing functionality
	harelRules := GetHarelValidationRules()

	// Filter rules to avoid duplication with existing basic rules
	rulesToApply := []HarelValidationRule{}
	for _, rule := range harelRules {
		switch rule.Name {
		case "NoStateNameConflicts":
			// Skip - duplicates UNIQUE_STATE_LABELS
			continue
		case "StateHierarchyWellFormed":
			// Skip - overlaps with UNIQUE_STATE_LABELS for duplicate detection
			continue
		case "ConfigurationConsistency":
			// Skip parts that duplicate BASIC_HAS_NO_CHILDREN, COMPOUND_HAS_CHILDREN, SINGLE_DEFAULT_CHILD
			continue
		default:
			// Apply advanced rules that add new validation capabilities
			rulesToApply = append(rulesToApply, rule)
		}
	}

	// Apply filtered rules
	for _, rule := range rulesToApply {
		if err := rule.Validator(statechart); err != nil {
			ruleID := harelRuleToRuleID(rule.Name)
			violations = append(violations, &validationv1.Violation{
				Rule:     ruleID,
				Severity: validationv1.Severity_ERROR,
				Message:  fmt.Sprintf("Harel Rule %s: %s", rule.Name, err.Error()),
			})
		}
	}

	return violations
}

func harelRuleToRuleID(name string) validationv1.RuleId {
	switch name {
	case "EventConsistency":
		return validationv1.RuleId_EVENT_PARAMETERS_CONSISTENT
	case "HistoryStateProperties":
		return validationv1.RuleId_HISTORY_STATES_WELL_FORMED
	case "TransitionWellFormedness":
		return validationv1.RuleId_GUARD_EXPRESSIONS_VALID
	default:
		return validationv1.RuleId_RULE_UNSPECIFIED
	}
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
