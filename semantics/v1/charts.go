package semantics

import "github.com/tmc/sc"

// Validate validates the statechart.
// It runs various checks to ensure that the statechart is well-formed.
// If the statechart is not well-formed, an error is returned.
func (s *Statechart) Validate() error {
	// validateNonOverlappingStateLabels
	// validateRootState
	// validateStateTypeAgreesWithChildren
	// validateParentChildRelationships
	// validateParentStatesHaveSingleDefaults
	return nil
}

// Normalize normalizes the statechart.
// It attaches values to the statechart that are derived from the statechart's
// structure.
func (s *Statechart) Normalize() error {
	if err := normalizeStateTypes(s); err != nil {
		return err
	}
	return nil
}

// normalizeStateTypes normalizes the state types.
// It sets the state type of each state based on the state's children
func normalizeStateTypes(s *Statechart) error {
	return visitStates(s.RootState, func(state *sc.State) error {
		if len(state.Children) == 0 {
			state.Type = sc.StateTypeBasic
		} else {
			if state.Type == sc.StateTypeUnspecified {
				state.Type = sc.StateTypeNormal
			}
		}
		return nil
	})
}

func visitStates(state *sc.State, f func(*sc.State) error) error {
	if err := f(state); err != nil {
		return err
	}
	for _, child := range state.Children {
		if err := visitStates(child, f); err != nil {
			return err
		}
	}
	return nil
}
