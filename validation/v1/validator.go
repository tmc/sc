// Package validation implements the SemanticValidator service.
package validation

import (
	"context"
	"fmt"
	"sort"
	"strings"

	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
	"google.golang.org/protobuf/proto"

	"github.com/tmc/sc"
	pb "github.com/tmc/sc/gen/statecharts/v1"
	validationv1 "github.com/tmc/sc/gen/validation/v1"
	semantics "github.com/tmc/sc/semantics/v1"
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

	for _, rule := range []struct {
		id      validationv1.RuleId
		collect func(*sc.Statechart) []validationIssue
	}{
		{id: validationv1.RuleId_UNIQUE_STATE_LABELS, collect: collectUniqueStateLabelIssues},
		{id: validationv1.RuleId_SINGLE_DEFAULT_CHILD, collect: collectSingleDefaultChildIssues},
		{id: validationv1.RuleId_BASIC_HAS_NO_CHILDREN, collect: collectBasicHasNoChildrenIssues},
		{id: validationv1.RuleId_COMPOUND_HAS_CHILDREN, collect: collectCompoundHasChildrenIssues},
		{id: validationv1.RuleId_DETERMINISTIC_TRANSITION_SELECTION, collect: collectDeterministicTransitionSelectionIssues},
		{id: validationv1.RuleId_NO_EVENT_BROADCAST_CYCLES, collect: collectNoEventBroadcastCycleIssues},
		{id: validationv1.RuleId_TIMEOUT_EVENTS_UNIQUE, collect: collectTimeoutEventsUniqueIssues},
	} {
		violations = append(violations, collectRuleViolations(statechart, ignoreRules, rule.id, validationv1.Severity_ERROR, "", rule.collect)...)
	}

	// Run comprehensive Harel validation rules
	harelViolations := s.validateHarelRules(statechart, ignoreRules)
	violations = append(violations, harelViolations...)

	reconciledViolations := s.validateReconciledConstraints(statechart, ignoreRules)
	violations = append(violations, reconciledViolations...)

	sortViolations(violations)
	return violations
}

// validateHarelRules applies comprehensive formal validation rules based on Harel's semantics
func (s *SemanticValidator) validateHarelRules(statechart *sc.Statechart, ignoreRules map[validationv1.RuleId]bool) []*validationv1.Violation {
	var violations []*validationv1.Violation

	for _, rule := range []struct {
		name    string
		ruleID  validationv1.RuleId
		collect func(*sc.Statechart) []validationIssue
	}{
		{name: "StateHierarchyWellFormed", ruleID: harelRuleToRuleID("StateHierarchyWellFormed"), collect: collectStateHierarchyWellFormedIssues},
		{name: "OrthogonalStatesDisjoint", ruleID: harelRuleToRuleID("OrthogonalStatesDisjoint"), collect: collectOrthogonalStatesDisjointIssues},
		{name: "TransitionSourceTargetValidity", ruleID: harelRuleToRuleID("TransitionSourceTargetValidity"), collect: collectTransitionSourceTargetValidityIssues},
		{name: "EventConsistency", ruleID: harelRuleToRuleID("EventConsistency"), collect: collectEventConsistencyIssues},
		{name: "ConfigurationConsistency", ruleID: harelRuleToRuleID("ConfigurationConsistency"), collect: collectConfigurationConsistencyIssues},
		{name: "TransitionWellFormedness", ruleID: harelRuleToRuleID("TransitionWellFormedness"), collect: collectTransitionWellFormednessIssues},
		{name: "InitialStateExists", ruleID: harelRuleToRuleID("InitialStateExists"), collect: collectInitialStateExistsIssues},
		{name: "NoSelfContainment", ruleID: harelRuleToRuleID("NoSelfContainment"), collect: collectNoSelfContainmentIssues},
		{name: "FinalStateProperties", ruleID: harelRuleToRuleID("FinalStateProperties"), collect: collectFinalStatePropertiesIssues},
		{name: "HistoryStateProperties", ruleID: harelRuleToRuleID("HistoryStateProperties"), collect: collectHistoryStatePropertyIssues},
		{name: "ParallelStateSemantics", ruleID: harelRuleToRuleID("ParallelStateSemantics"), collect: collectParallelStateSemanticsIssues},
	} {
		violations = append(violations,
			collectRuleViolations(statechart, ignoreRules, rule.ruleID, validationv1.Severity_ERROR,
				fmt.Sprintf("Harel Rule %s: ", rule.name), rule.collect)...)
	}

	return violations
}

func collectRuleViolations(statechart *sc.Statechart, ignoreRules map[validationv1.RuleId]bool, ruleID validationv1.RuleId, severity validationv1.Severity, prefix string, collect func(*sc.Statechart) []validationIssue) []*validationv1.Violation {
	if collect == nil {
		return nil
	}
	if ignoreRules[ruleID] {
		return nil
	}
	if ruleID == validationv1.RuleId_RULE_UNSPECIFIED && ignoreRules[validationv1.RuleId_RULE_UNSPECIFIED] {
		return nil
	}
	return issuesToViolations(ruleID, severity, prefix, collect(statechart))
}

func harelRuleToRuleID(name string) validationv1.RuleId {
	switch name {
	case "StateHierarchyWellFormed":
		return validationv1.RuleId_PSEUDO_STATES_WELL_FORMED
	case "OrthogonalStatesDisjoint":
		return validationv1.RuleId_FORK_JOIN_BALANCED
	case "TransitionSourceTargetValidity":
		return validationv1.RuleId_INTERNAL_TRANSITIONS_VALID
	case "EventConsistency":
		return validationv1.RuleId_EVENT_PARAMETERS_CONSISTENT
	case "ConfigurationConsistency":
		return validationv1.RuleId_COMPLETION_TRANSITIONS_VALID
	case "TransitionWellFormedness":
		return validationv1.RuleId_GUARD_EXPRESSIONS_VALID
	case "InitialStateExists":
		return validationv1.RuleId_HISTORY_DEFAULTS_VALID
	case "NoSelfContainment":
		return validationv1.RuleId_INVARIANTS_SATISFIABLE
	case "FinalStateProperties":
		return validationv1.RuleId_ACTION_EXPRESSIONS_VALID
	case "HistoryStateProperties":
		return validationv1.RuleId_HISTORY_STATES_WELL_FORMED
	case "ParallelStateSemantics":
		return validationv1.RuleId_CHOICE_GUARDS_COMPLETE
	default:
		return validationv1.RuleId_RULE_UNSPECIFIED
	}
}

func sortViolations(violations []*validationv1.Violation) {
	sort.SliceStable(violations, func(i, j int) bool {
		if violations[i].Rule != violations[j].Rule {
			return violations[i].Rule < violations[j].Rule
		}
		if violations[i].Severity != violations[j].Severity {
			return violations[i].Severity < violations[j].Severity
		}
		if violations[i].Message != violations[j].Message {
			return violations[i].Message < violations[j].Message
		}
		return strings.Join(violations[i].Xpath, "\x00") < strings.Join(violations[j].Xpath, "\x00")
	})
}

func (s *SemanticValidator) validateReconciledConstraints(statechart *sc.Statechart, ignoreRules map[validationv1.RuleId]bool) []*validationv1.Violation {
	var violations []*validationv1.Violation

	reconciled := semantics.NewStatechart(statechart)
	reconciledViolations, err := reconciled.CheckReconciledConstraints(semantics.ReconciledOptions{MaxSteps: 256})
	if err != nil {
		if ignoreRules[validationv1.RuleId_RULE_UNSPECIFIED] {
			return violations
		}
		return []*validationv1.Violation{{
			Rule:     validationv1.RuleId_RULE_UNSPECIFIED,
			Severity: validationv1.Severity_WARNING,
			Message:  fmt.Sprintf("reconciled semantics constraints not checked: %v", err),
		}}
	}

	for _, violation := range reconciledViolations {
		ruleID := reconciledConstraintToRuleID(violation.Constraint)
		if ignoreRules[ruleID] {
			continue
		}
		violations = append(violations, &validationv1.Violation{
			Rule:     ruleID,
			Severity: validationv1.Severity_WARNING,
			Message:  fmt.Sprintf("reconciled %s: %s", violation.Constraint, violation.Message),
		})
	}

	return violations
}

func reconciledConstraintToRuleID(id semantics.ReconciledConstraintID) validationv1.RuleId {
	switch id {
	case semantics.ReconciledConstraintC1:
		return validationv1.RuleId_RECONCILING_C1_NO_COMPLETION_TRANSITIONS
	case semantics.ReconciledConstraintC2:
		return validationv1.RuleId_RECONCILING_C2_ACYCLIC_TRIGGERING
	case semantics.ReconciledConstraintC3:
		return validationv1.RuleId_RECONCILING_C3_NO_EXTERNAL_INTERNAL_CONFLICT
	case semantics.ReconciledConstraintC4:
		return validationv1.RuleId_RECONCILING_C4_TRIGGERS_ARE_CONSISTENT
	case semantics.ReconciledConstraintC5:
		return validationv1.RuleId_RECONCILING_C5_TOUCHED_INTERNAL_PRECONDITION
	case semantics.ReconciledConstraintC6:
		return validationv1.RuleId_RECONCILING_C6_CONSISTENT_TRIGGERS_STAY_CONSISTENT
	case semantics.ReconciledConstraintC7:
		return validationv1.RuleId_RECONCILING_C7_NO_COMPLETION_CYCLES
	case semantics.ReconciledConstraintC8:
		return validationv1.RuleId_RECONCILING_C8_NO_COMPLETION_TOUCHES_INTERNAL
	case semantics.ReconciledConstraintC9:
		return validationv1.RuleId_RECONCILING_C9_NO_EXTERNAL_COMPLETION_CONFLICT
	case semantics.ReconciledConstraintC10:
		return validationv1.RuleId_RECONCILING_C10_NO_COMPLETION_INTERNAL_CONFLICT
	case semantics.ReconciledConstraintC11:
		return validationv1.RuleId_RECONCILING_C11_CONFLICTING_COMPLETIONS_SHARE_SOURCES
	case semantics.ReconciledConstraintC12:
		return validationv1.RuleId_RECONCILING_C12_ACYCLIC_PREC_RELATION
	case semantics.ReconciledConstraintC13:
		return validationv1.RuleId_RECONCILING_C13_EQUAL_PRIORITY_SHAPE
	case semantics.ReconciledConstraintC14:
		return validationv1.RuleId_RECONCILING_C14_SINGLE_GENERATED_EVENT
	case semantics.ReconciledConstraintC15:
		return validationv1.RuleId_RECONCILING_C15_CONSISTENT_SAME_TRIGGER_SAME_OUTPUT
	case semantics.ReconciledConstraintC16:
		return validationv1.RuleId_RECONCILING_C16_COMPLETION_NOT_CONSISTENT_WITH_INTERNAL
	case semantics.ReconciledConstraintC17:
		return validationv1.RuleId_RECONCILING_C17_UML_INTERNAL_PRIORITY
	default:
		return validationv1.RuleId_RULE_UNSPECIFIED
	}
}

// convertProtoToStatechart converts a proto statechart to a native statechart.
// The native types in package sc are aliases of the generated protobuf types,
// so a protobuf clone preserves the full model without field-by-field loss.
func convertProtoToStatechart(protoChart *pb.Statechart) *sc.Statechart {
	if protoChart == nil {
		return nil
	}
	return proto.Clone(protoChart).(*pb.Statechart)
}
