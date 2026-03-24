package validation

import (
	"strings"

	"github.com/tmc/sc"
	"github.com/tmc/sc/semantics/v1"
)

// HarelValidationRule represents a formal validation rule based on Harel's semantics
type HarelValidationRule struct {
	Name        string
	Description string
	Validator   func(*sc.Statechart) error
}

// GetHarelValidationRules returns all formal Harel validation rules
func GetHarelValidationRules() []HarelValidationRule {
	return []HarelValidationRule{
		{
			Name:        "StateHierarchyWellFormed",
			Description: "State hierarchy must be well-formed with proper parent-child relationships",
			Validator:   validateStateHierarchyWellFormed,
		},
		{
			Name:        "OrthogonalStatesDisjoint",
			Description: "Orthogonal states must be disjoint (no overlapping substates)",
			Validator:   validateOrthogonalStatesDisjoint,
		},
		{
			Name:        "TransitionSourceTargetValidity",
			Description: "All transition sources and targets must reference valid states",
			Validator:   validateTransitionSourceTargetValidity,
		},
		{
			Name:        "EventConsistency",
			Description: "Events referenced in transitions must be declared",
			Validator:   validateEventConsistency,
		},
		{
			Name:        "ConfigurationConsistency",
			Description: "State configurations must respect XOR/AND semantics",
			Validator:   validateConfigurationConsistency,
		},
		{
			Name:        "NoStateNameConflicts",
			Description: "State names must be unique within their scope",
			Validator:   validateNoStateNameConflicts,
		},
		{
			Name:        "TransitionWellFormedness",
			Description: "Transitions must be well-formed with valid guards and actions",
			Validator:   validateTransitionWellFormedness,
		},
		{
			Name:        "InitialStateExists",
			Description: "Every compound state must have exactly one initial state",
			Validator:   validateInitialStateExists,
		},
		{
			Name:        "NoSelfContainment",
			Description: "States cannot contain themselves directly or indirectly",
			Validator:   validateNoSelfContainment,
		},
		{
			Name:        "FinalStateProperties",
			Description: "Final states must have correct properties (no outgoing transitions)",
			Validator:   validateFinalStateProperties,
		},
		{
			Name:        "HistoryStateProperties",
			Description: "History states must have correct properties and parent relationships",
			Validator:   validateHistoryStateProperties,
		},
		{
			Name:        "ParallelStateSemantics",
			Description: "Parallel states must follow AND-decomposition semantics",
			Validator:   validateParallelStateSemantics,
		},
	}
}

// validateStateHierarchyWellFormed checks that the state hierarchy forms a proper tree
func validateStateHierarchyWellFormed(statechart *sc.Statechart) error {
	return firstIssueError(collectStateHierarchyWellFormedIssues(statechart))
}

// validateOrthogonalStatesDisjoint checks that orthogonal regions don't overlap
func validateOrthogonalStatesDisjoint(statechart *sc.Statechart) error {
	return firstIssueError(collectOrthogonalStatesDisjointIssues(statechart))
}

// validateTransitionSourceTargetValidity checks that all transition endpoints are valid
func validateTransitionSourceTargetValidity(statechart *sc.Statechart) error {
	return firstIssueError(collectTransitionSourceTargetValidityIssues(statechart))
}

// validateEventConsistency checks that all events used in transitions are declared
func validateEventConsistency(statechart *sc.Statechart) error {
	return firstIssueError(collectEventConsistencyIssues(statechart))
}

// validateConfigurationConsistency checks XOR/AND configuration semantics
func validateConfigurationConsistency(statechart *sc.Statechart) error {
	return firstIssueError(collectConfigurationConsistencyIssues(statechart))
}

// validateNoStateNameConflicts checks for state name uniqueness
func validateNoStateNameConflicts(statechart *sc.Statechart) error {
	return firstIssueError(collectUniqueStateLabelIssues(statechart))
}

// validateTransitionWellFormedness checks transition structure
func validateTransitionWellFormedness(statechart *sc.Statechart) error {
	return firstIssueError(collectTransitionWellFormednessIssues(statechart))
}

// validateGuardExpression performs basic validation of guard expressions
func validateGuardExpression(expression string) error {
	if expression == "" {
		return semantics.ValidationErrorf(semantics.ErrCodeGuardInvalidExpression, "guard expression is empty")
	}

	// Basic syntax check - could be extended with proper parsing
	if strings.Contains(expression, ";;") {
		return semantics.NewErrorBuilder(semantics.ErrCodeGuardInvalidExpression, semantics.CategoryValidation,
			"guard expression contains invalid syntax").
			WithContext("expression", expression).
			WithContext("invalid_token", ";;").
			Build()
	}

	return nil
}

// validateInitialStateExists checks initial state requirements
func validateInitialStateExists(statechart *sc.Statechart) error {
	return firstIssueError(collectInitialStateExistsIssues(statechart))
}

// validateNoSelfContainment checks for circular containment
func validateNoSelfContainment(statechart *sc.Statechart) error {
	return firstIssueError(collectNoSelfContainmentIssues(statechart))
}

// validateFinalStateProperties checks final state constraints
func validateFinalStateProperties(statechart *sc.Statechart) error {
	return firstIssueError(collectFinalStatePropertiesIssues(statechart))
}

// validateHistoryStateProperties checks history state constraints
func validateHistoryStateProperties(statechart *sc.Statechart) error {
	return firstIssueError(collectHistoryStatePropertyIssues(statechart))
}

func historyTypeName(historyType sc.HistoryType) string {
	switch historyType {
	case sc.HistoryType_HISTORY_TYPE_DEEP:
		return "deep"
	default:
		return "shallow"
	}
}

// validateParallelStateSemantics checks parallel state AND semantics
func validateParallelStateSemantics(statechart *sc.Statechart) error {
	return firstIssueError(collectParallelStateSemanticsIssues(statechart))
}
