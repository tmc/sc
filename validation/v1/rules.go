package validation

import (
	"fmt"

	"github.com/tmc/sc"
)

// validateUniqueStateLabels checks that all state labels are unique.
func validateUniqueStateLabels(statechart *sc.Statechart) error {
	if statechart.RootState == nil {
		return fmt.Errorf("root state is nil")
	}
	
	labels := make(map[string]bool)
	var checkLabels func(*sc.State) error
	
	checkLabels = func(state *sc.State) error {
		if labels[state.Label] {
			return fmt.Errorf("duplicate state label: %s", state.Label)
		}
		
		labels[state.Label] = true
		
		for _, child := range state.Children {
			if err := checkLabels(child); err != nil {
				return err
			}
		}
		
		return nil
	}
	
	return checkLabels(statechart.RootState)
}

// validateSingleDefaultChild ensures that XOR composite states have exactly one default child.
func validateSingleDefaultChild(statechart *sc.Statechart) error {
	if statechart.RootState == nil {
		return fmt.Errorf("root state is nil")
	}
	
	var checkDefaults func(*sc.State) error
	
	checkDefaults = func(state *sc.State) error {
		if state.Type == sc.StateTypeNormal {
			defaultCount := 0
			
			for _, child := range state.Children {
				if child.IsInitial {
					defaultCount++
				}
			}
			
			if defaultCount != 1 {
				return fmt.Errorf("state %s has %d default states, should have exactly 1", state.Label, defaultCount)
			}
		}
		
		for _, child := range state.Children {
			if err := checkDefaults(child); err != nil {
				return err
			}
		}
		
		return nil
	}
	
	return checkDefaults(statechart.RootState)
}

// validateBasicHasNoChildren ensures that basic states have no children.
func validateBasicHasNoChildren(statechart *sc.Statechart) error {
	if statechart.RootState == nil {
		return fmt.Errorf("root state is nil")
	}
	
	var checkBasicStates func(*sc.State) error
	
	checkBasicStates = func(state *sc.State) error {
		if state.Type == sc.StateTypeBasic && len(state.Children) > 0 {
			return fmt.Errorf("basic state %s has children", state.Label)
		}
		
		for _, child := range state.Children {
			if err := checkBasicStates(child); err != nil {
				return err
			}
		}
		
		return nil
	}
	
	return checkBasicStates(statechart.RootState)
}

// validateCompoundHasChildren ensures that compound states have children.
func validateCompoundHasChildren(statechart *sc.Statechart) error {
	if statechart.RootState == nil {
		return fmt.Errorf("root state is nil")
	}
	
	var checkCompoundStates func(*sc.State) error
	
	checkCompoundStates = func(state *sc.State) error {
		if (state.Type == sc.StateTypeNormal || state.Type == sc.StateTypeParallel) && len(state.Children) == 0 {
			return fmt.Errorf("compound state %s has no children", state.Label)
		}
		
		for _, child := range state.Children {
			if err := checkCompoundStates(child); err != nil {
				return err
			}
		}
		
		return nil
	}
	
	return checkCompoundStates(statechart.RootState)
}

// validateRootState ensures that the root state exists and has the correct label.
func validateRootState(statechart *sc.Statechart) error {
	if statechart.RootState == nil {
		return fmt.Errorf("root state is nil")
	}
	
	if statechart.RootState.Label != "__root__" {
		return fmt.Errorf("root state has an unexpected label of '%s' (expected '__root__')", statechart.RootState.Label)
	}
	
	return nil
}

// validateDeterministicTransitionSelection ensures that transitions are deterministic.
// This is a simplified implementation - a full implementation would need to analyze
// guards and potential conflicts.
func validateDeterministicTransitionSelection(statechart *sc.Statechart) error {
	eventTransitions := make(map[string]map[string]bool)
	
	for _, transition := range statechart.Transitions {
		event := transition.Event
		
		if event == "" {
			continue // Empty event transitions are not considered here
		}
		
		for _, source := range transition.From {
			if eventTransitions[event] == nil {
				eventTransitions[event] = make(map[string]bool)
			}
			
			if eventTransitions[event][source] {
				return fmt.Errorf("non-deterministic transitions: multiple transitions from state '%s' on event '%s'", source, event)
			}
			
			eventTransitions[event][source] = true
		}
	}
	
	return nil
}

// validateNoEventBroadcastCycles ensures there are no cycles in event broadcasts.
// This is a stub implementation. A complete implementation would need to analyze
// action-to-event relationships and detect cycles.
func validateNoEventBroadcastCycles(statechart *sc.Statechart) error {
	// In a real implementation, this would detect cycles in the event broadcast graph
	// For now, we'll return nil (no validation)
	return nil
}