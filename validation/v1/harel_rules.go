package validation

import (
	"fmt"
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
	if statechart.RootState == nil {
		return semantics.ValidationErrorf(semantics.ErrCodeStatechartMissingRoot, "statechart has no root state")
	}

	// Track all states and their parents
	stateParents := make(map[string]string)
	var visitState func(*sc.State, string) error

	visitState = func(state *sc.State, parentLabel string) error {
		if state.Label == "" {
			return semantics.ValidationErrorf(semantics.ErrCodeStateEmpty, "state has empty label")
		}

		// Check for duplicate state in hierarchy
		if existingParent, exists := stateParents[state.Label]; exists {
			return semantics.NewErrorBuilder(semantics.ErrCodeStateDuplicate, semantics.CategoryValidation, 
				"state appears multiple times in hierarchy").
				WithContext("state_label", state.Label).
				WithContext("existing_parent", existingParent).
				WithContext("new_parent", parentLabel).
				Build()
		}

		stateParents[state.Label] = parentLabel

		// Recursively check children
		for _, child := range state.Children {
			if err := visitState(child, state.Label); err != nil {
				return err
			}
		}

		return nil
	}

	return visitState(statechart.RootState, "")
}

// validateOrthogonalStatesDisjoint checks that orthogonal regions don't overlap
func validateOrthogonalStatesDisjoint(statechart *sc.Statechart) error {
	var checkOrthogonality func(*sc.State) error

	checkOrthogonality = func(state *sc.State) error {
		if state.Type == sc.StateTypeParallel {
			// Collect all descendant states for each orthogonal region
			regionStates := make([]map[string]bool, len(state.Children))
			
			for i, child := range state.Children {
				regionStates[i] = make(map[string]bool)
				collectDescendants(child, regionStates[i])
			}

			// Check for overlaps between regions
			for i := 0; i < len(regionStates); i++ {
				for j := i + 1; j < len(regionStates); j++ {
					for stateName := range regionStates[i] {
						if regionStates[j][stateName] {
							return semantics.NewErrorBuilder(semantics.ErrCodeOrthogonalStatesOverlap, semantics.CategoryValidation,
								"orthogonal regions overlap").
								WithContext("parent_state", state.Label).
								WithContext("overlapping_state", stateName).
								WithContext("region_1", i).
								WithContext("region_2", j).
								Build()
						}
					}
				}
			}
		}

		// Recursively check children
		for _, child := range state.Children {
			if err := checkOrthogonality(child); err != nil {
				return err
			}
		}

		return nil
	}

	return checkOrthogonality(statechart.RootState)
}

// collectDescendants recursively collects all descendant state names
func collectDescendants(state *sc.State, descendants map[string]bool) {
	descendants[state.Label] = true
	for _, child := range state.Children {
		collectDescendants(child, descendants)
	}
}

// validateTransitionSourceTargetValidity checks that all transition endpoints are valid
func validateTransitionSourceTargetValidity(statechart *sc.Statechart) error {
	// Build map of all valid state names
	stateNames := make(map[string]bool)
	var collectStates func(*sc.State)
	
	collectStates = func(state *sc.State) {
		stateNames[state.Label] = true
		for _, child := range state.Children {
			collectStates(child)
		}
	}
	
	if statechart.RootState != nil {
		collectStates(statechart.RootState)
	}

	// Validate all transitions
	for i, transition := range statechart.Transitions {
		if transition.Label == "" {
			return semantics.NewErrorBuilder(semantics.ErrCodeTransitionMissingLabel, semantics.CategoryValidation,
				"transition has empty label").
				WithContext("transition_index", i).
				Build()
		}

		// Check source states
		for _, source := range transition.From {
			if !stateNames[source] {
				return semantics.NewErrorBuilder(semantics.ErrCodeTransitionInvalidSource, semantics.CategoryValidation,
					"transition references invalid source state").
					WithContext("transition_label", transition.Label).
					WithContext("invalid_source", source).
					Build()
			}
		}

		// Check target states
		for _, target := range transition.To {
			if !stateNames[target] {
				return semantics.NewErrorBuilder(semantics.ErrCodeTransitionInvalidTarget, semantics.CategoryValidation,
					"transition references invalid target state").
					WithContext("transition_label", transition.Label).
					WithContext("invalid_target", target).
					Build()
			}
		}
	}

	return nil
}

// validateEventConsistency checks that all events used in transitions are declared
func validateEventConsistency(statechart *sc.Statechart) error {
	// Build map of declared events
	declaredEvents := make(map[string]bool)
	for _, event := range statechart.Events {
		if event.Label == "" {
			return semantics.ValidationErrorf(semantics.ErrCodeEventInvalidLabel, "event has empty label")
		}
		declaredEvents[event.Label] = true
	}

	// Check that all transition events are declared
	for _, transition := range statechart.Transitions {
		if transition.Event != "" && !declaredEvents[transition.Event] {
			return semantics.NewErrorBuilder(semantics.ErrCodeEventUndeclared, semantics.CategoryValidation,
				"transition references undeclared event").
				WithContext("transition_label", transition.Label).
				WithContext("undeclared_event", transition.Event).
				Build()
		}
	}

	return nil
}

// validateConfigurationConsistency checks XOR/AND configuration semantics
func validateConfigurationConsistency(statechart *sc.Statechart) error {
	// This validates that state types are consistent with their children
	var validateConsistency func(*sc.State) error

	validateConsistency = func(state *sc.State) error {
		switch state.Type {
		case sc.StateTypeBasic:
			if len(state.Children) > 0 {
				return semantics.NewErrorBuilder(semantics.ErrCodeStateInvalidChildren, semantics.CategoryValidation,
					"basic state cannot have children").
					WithContext("state_label", state.Label).
					WithContext("child_count", len(state.Children)).
					Build()
			}

		case sc.StateTypeNormal:
			if len(state.Children) == 0 {
				return semantics.NewErrorBuilder(semantics.ErrCodeStateInvalidChildren, semantics.CategoryValidation,
					"normal (XOR) state must have children").
					WithContext("state_label", state.Label).
					Build()
			}
			
			// Check for exactly one initial state
			initialCount := 0
			for _, child := range state.Children {
				if child.IsInitial {
					initialCount++
				}
			}
			if initialCount != 1 {
				return semantics.NewErrorBuilder(semantics.ErrCodeStateInvalidInitial, semantics.CategoryValidation,
					"normal state must have exactly one initial child").
					WithContext("state_label", state.Label).
					WithContext("initial_count", initialCount).
					Build()
			}

		case sc.StateTypeParallel:
			if len(state.Children) < 2 {
				return semantics.NewErrorBuilder(semantics.ErrCodeParallelStateInsufficientRegions, semantics.CategoryValidation,
					"parallel (AND) state must have at least 2 children").
					WithContext("state_label", state.Label).
					WithContext("child_count", len(state.Children)).
					Build()
			}
			
			// All children of parallel states are implicitly active
			for _, child := range state.Children {
				if child.IsInitial {
					return semantics.NewErrorBuilder(semantics.ErrCodeStateInvalidInitial, semantics.CategoryValidation,
						"children of parallel state cannot be marked as initial").
						WithContext("parent_state", state.Label).
						WithContext("child_state", child.Label).
						Build()
				}
			}
		}

		// Recursively validate children
		for _, child := range state.Children {
			if err := validateConsistency(child); err != nil {
				return err
			}
		}

		return nil
	}

	if statechart.RootState == nil {
		return semantics.ValidationErrorf(semantics.ErrCodeStatechartMissingRoot, "statechart has no root state")
	}

	return validateConsistency(statechart.RootState)
}

// validateNoStateNameConflicts checks for state name uniqueness
func validateNoStateNameConflicts(statechart *sc.Statechart) error {
	stateNames := make(map[string]string) // name -> first occurrence path
	var checkNames func(*sc.State, string) error

	checkNames = func(state *sc.State, path string) error {
		currentPath := path + "/" + state.Label
		
		if existingPath, exists := stateNames[state.Label]; exists {
			return semantics.NewErrorBuilder(semantics.ErrCodeStateDuplicate, semantics.CategoryValidation,
				"state name conflict").
				WithContext("state_label", state.Label).
				WithContext("existing_path", existingPath).
				WithContext("current_path", currentPath).
				Build()
		}
		
		stateNames[state.Label] = currentPath

		for _, child := range state.Children {
			if err := checkNames(child, currentPath); err != nil {
				return err
			}
		}

		return nil
	}

	if statechart.RootState == nil {
		return semantics.ValidationErrorf(semantics.ErrCodeStatechartMissingRoot, "statechart has no root state")
	}

	return checkNames(statechart.RootState, "")
}

// validateTransitionWellFormedness checks transition structure
func validateTransitionWellFormedness(statechart *sc.Statechart) error {
	for i, transition := range statechart.Transitions {
		if transition.Label == "" {
			return semantics.NewErrorBuilder(semantics.ErrCodeTransitionMissingLabel, semantics.CategoryValidation,
				"transition has empty label").
				WithContext("transition_index", i).
				Build()
		}

		if len(transition.From) == 0 {
			return semantics.NewErrorBuilder(semantics.ErrCodeTransitionMalformed, semantics.CategoryValidation,
				"transition has no source states").
				WithContext("transition_label", transition.Label).
				Build()
		}

		if len(transition.To) == 0 {
			return semantics.NewErrorBuilder(semantics.ErrCodeTransitionMalformed, semantics.CategoryValidation,
				"transition has no target states").
				WithContext("transition_label", transition.Label).
				Build()
		}

		// Validate guard expression if present
		if transition.Guard != nil {
			if err := validateGuardExpression(transition.Guard.Expression); err != nil {
				return semantics.WrapErrorf(semantics.ErrCodeGuardInvalidExpression, err,
					"transition %s has invalid guard", transition.Label)
			}
		}

		// Validate actions if present
		for j, action := range transition.Actions {
			if action.Label == "" {
				return semantics.NewErrorBuilder(semantics.ErrCodeActionInvalidLabel, semantics.CategoryValidation,
					"transition action has empty label").
					WithContext("transition_label", transition.Label).
					WithContext("action_index", j).
					Build()
			}
		}
	}

	return nil
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
	var checkInitial func(*sc.State) error

	checkInitial = func(state *sc.State) error {
		if state.Type == sc.StateTypeNormal && len(state.Children) > 0 {
			hasInitial := false
			for _, child := range state.Children {
				if child.IsInitial {
					hasInitial = true
					break
				}
			}
			
			if !hasInitial {
				return semantics.NewErrorBuilder(semantics.ErrCodeStateInvalidInitial, semantics.CategoryValidation,
					"compound state has no initial state").
					WithContext("state_label", state.Label).
					Build()
			}
		}

		for _, child := range state.Children {
			if err := checkInitial(child); err != nil {
				return err
			}
		}

		return nil
	}

	if statechart.RootState == nil {
		return semantics.ValidationErrorf(semantics.ErrCodeStatechartMissingRoot, "statechart has no root state")
	}

	return checkInitial(statechart.RootState)
}

// validateNoSelfContainment checks for circular containment
func validateNoSelfContainment(statechart *sc.Statechart) error {
	var checkContainment func(*sc.State, map[string]bool) error

	checkContainment = func(state *sc.State, ancestors map[string]bool) error {
		if ancestors[state.Label] {
			return semantics.NewErrorBuilder(semantics.ErrCodeStateCircularContainment, semantics.CategoryValidation,
				"circular containment detected").
				WithContext("state_label", state.Label).
				Build()
		}

		newAncestors := make(map[string]bool)
		for k, v := range ancestors {
			newAncestors[k] = v
		}
		newAncestors[state.Label] = true

		for _, child := range state.Children {
			if err := checkContainment(child, newAncestors); err != nil {
				return err
			}
		}

		return nil
	}

	if statechart.RootState == nil {
		return fmt.Errorf("statechart has no root state")
	}

	return checkContainment(statechart.RootState, make(map[string]bool))
}

// validateFinalStateProperties checks final state constraints
func validateFinalStateProperties(statechart *sc.Statechart) error {
	// Collect final states
	finalStates := make(map[string]bool)
	var collectFinalStates func(*sc.State)
	
	collectFinalStates = func(state *sc.State) {
		if state.IsFinal {
			finalStates[state.Label] = true
		}
		for _, child := range state.Children {
			collectFinalStates(child)
		}
	}

	if statechart.RootState != nil {
		collectFinalStates(statechart.RootState)
	}

	// Check that final states have no outgoing transitions
	for _, transition := range statechart.Transitions {
		for _, source := range transition.From {
			if finalStates[source] {
				return fmt.Errorf("final state %s cannot have outgoing transitions", source)
			}
		}
	}

	return nil
}

// validateHistoryStateProperties checks history state constraints
func validateHistoryStateProperties(statechart *sc.Statechart) error {
	// Note: This is a placeholder as the current protobuf schema doesn't include history states
	// In a full implementation, this would validate:
	// - History states are properly contained in compound states
	// - History states have appropriate default transitions
	// - Shallow vs deep history semantics are correct
	return nil
}

// validateParallelStateSemantics checks parallel state AND semantics
func validateParallelStateSemantics(statechart *sc.Statechart) error {
	var checkParallel func(*sc.State) error

	checkParallel = func(state *sc.State) error {
		if state.Type == sc.StateTypeParallel {
			// Parallel states must have at least 2 regions
			if len(state.Children) < 2 {
				return fmt.Errorf("parallel state %s must have at least 2 orthogonal regions, found %d", 
					state.Label, len(state.Children))
			}

			// Each region should be a compound state (usually)
			for _, region := range state.Children {
				if region.Type == sc.StateTypeBasic {
					// While not strictly invalid, basic regions in parallel states are unusual
					// This could be a warning in a more sophisticated validator
				}
			}
		}

		for _, child := range state.Children {
			if err := checkParallel(child); err != nil {
				return err
			}
		}

		return nil
	}

	if statechart.RootState == nil {
		return fmt.Errorf("statechart has no root state")
	}

	return checkParallel(statechart.RootState)
}