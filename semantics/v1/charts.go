package semantics

import "github.com/tmc/sc"

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
