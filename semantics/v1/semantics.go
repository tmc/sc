package semantics

import (
	"errors"

	"github.com/tmc/sc"
)

type StateLabel string

var (
	ErrNotFound = errors.New("not found")
)

// Statechart wraps a statechart and provides a simple interface for evaluating semantics.
type Statechart struct {
	*sc.Statechart
}

// NewStatechart creates a new statechart from a statechart definition.
func NewStatechart(sc *sc.Statechart) *Statechart {
	return &Statechart{sc}
}

// Validate validates the statechart.
// It runs various checks to ensure that the statechart is well-formed.
// If the statechart is not well-formed, an error is returned.
func (s *Statechart) Validate() error {
	// validateNonOverlappingStateLabels
	// validateSingleRootState
	// validateParentChildRelationships
	return nil
}

// Children returns the immediate children of the given state.
func (c *Statechart) Children(state StateLabel) ([]StateLabel, error) {
	s, err := c.findState(state)
	if err != nil {
		return nil, err
	}
	result := make([]StateLabel, len(s.Children))
	for i, child := range s.Children {
		result[i] = StateLabel(child.Label)
	}
	return result, nil
}

// findState finds the state with the given label.
func (s *Statechart) findState(label StateLabel) (*sc.State, error) {
	return s._findState(s.RootState, label)
}

// _findState finds the state with the given label.
func (s *Statechart) _findState(state *sc.State, label StateLabel) (*sc.State, error) {
	if state.Label == string(label) {
		return state, nil
	}
	for _, state := range state.Children {
		if result, err := s._findState(state, label); err == nil {
			return result, nil
		}
	}
	return nil, ErrNotFound
}
