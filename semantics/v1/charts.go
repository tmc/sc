package semantics

import (
	"fmt"
	
	"github.com/tmc/sc"
)

// Normalize normalizes the statechart.
// Normalize returns a new normalized Statechart.
func (s *Statechart) Normalize() (*Statechart, error) {
	newInternal := s.Statechart // Create a shallow copy
	if err := normalizeStateTypes(newInternal); err != nil {
		return nil, err
	}
	return NewStatechart(newInternal), nil
}

// normalizeStateTypes normalizes the state types.
// It sets the state type of each state based on the state's children
func normalizeStateTypes(s *sc.Statechart) error {
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

// Validate validates the statechart structure and semantics.
func (s *Statechart) Validate() error {
	if s.Statechart == nil {
		return fmt.Errorf("statechart is nil")
	}
	if s.RootState == nil {
		return fmt.Errorf("root state is nil")
	}
	
	// Validate state labels are unique
	stateLabels := make(map[string]bool)
	err := visitStates(s.RootState, func(state *sc.State) error {
		if state.Label == "" {
			return fmt.Errorf("state has empty label")
		}
		if stateLabels[state.Label] {
			return fmt.Errorf("duplicate state label: %s", state.Label)
		}
		stateLabels[state.Label] = true
		return nil
	})
	if err != nil {
		return err
	}
	
	// Validate each state has consistent type and children
	err = visitStates(s.RootState, func(state *sc.State) error {
		if state.Type == sc.StateTypeBasic && len(state.Children) > 0 {
			return fmt.Errorf("basic state %s cannot have children", state.Label)
		}
		if (state.Type == sc.StateTypeNormal || state.Type == sc.StateTypeParallel) && len(state.Children) == 0 {
			return fmt.Errorf("compound state %s must have children", state.Label)
		}
		
		// Check for exactly one initial child in normal states
		if state.Type == sc.StateTypeNormal && len(state.Children) > 0 {
			initialCount := 0
			for _, child := range state.Children {
				if child.IsInitial {
					initialCount++
				}
			}
			if initialCount != 1 {
				return fmt.Errorf("normal state %s must have exactly one initial child, found %d", state.Label, initialCount)
			}
		}
		
		return nil
	})
	
	return err
}

// InitialConfiguration computes the default initial configuration of the statechart.
func (s *Statechart) InitialConfiguration() (*sc.Configuration, error) {
	if s.RootState == nil {
		return nil, fmt.Errorf("root state is nil")
	}
	
	// Start with just the root state
	config := &sc.Configuration{
		States: []*sc.StateRef{{Label: s.RootState.Label}},
	}
	
	// Compute default completion to get all initial states
	return DefaultCompletion(s, config)
}

// GetDepth returns the depth of a state in the hierarchy (root = 0).
func (s *Statechart) GetDepth(label StateLabel) (int, error) {
	state, err := s.findState(label)
	if err != nil {
		return 0, err
	}

	depth := 0
	current := state
	for current != s.RootState {
		parent, err := s.GetParent(StateLabel(current.Label))
		if err != nil {
			return 0, err
		}
		if parent == nil {
			break
		}
		current = parent
		depth++
	}

	return depth, nil
}
