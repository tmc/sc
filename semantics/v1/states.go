package semantics

import (
	"github.com/tmc/sc"
	"golang.org/x/exp/slices"
)

// statesContains returns true if the given slice of states contains the given state.
func statesContains(states []StateLabel, state StateLabel) bool {
	return slices.Contains(states, state)
}

// Children returns the immediate children of the given state.
func (c *Statechart) Children(state StateLabel) ([]StateLabel, error) {
	s, err := c.findState(state)
	if err != nil {
		return nil, err
	}
	var result []StateLabel
	for _, child := range s.Children {
		result = append(result, StateLabel(child.Label))
	}
	return result, nil
}

// ChildrenPlus returns the transitive closure of the children of the given state.
func (c *Statechart) ChildrenPlus(state StateLabel) ([]StateLabel, error) {
	s, err := c.findState(state)
	if err != nil {
		return nil, err
	}
	return c.childrenPlus(s)
}

// ChildrenStar returns the reflexive-transitive closure of the children of the given state.
func (c *Statechart) ChildrenStar(state StateLabel) ([]StateLabel, error) {
	s, err := c.findState(state)
	if err != nil {
		return nil, err
	}
	result := []StateLabel{state}
	children, err := c.childrenPlus(s)
	if err != nil {
		return nil, err
	}
	result = append(result, children...)
	return result, nil
}

// Descendant returns true if the given state is a descendant of the given potential ancestor.
func (c *Statechart) Descendant(state StateLabel, potentialAncestor StateLabel) (bool, error) {
	rtClosure, err := c.ChildrenStar(potentialAncestor)
	if err != nil {
		return false, err
	}
	return statesContains(rtClosure, state), nil
}

// Ancestor returns true if the given state is an ancestor of the given potential descendant.
func (c *Statechart) Ancestor(state StateLabel, potentialDescendant StateLabel) (bool, error) {
	rtClosure, err := c.ChildrenStar(state)
	if err != nil {
		return false, err
	}
	return statesContains(rtClosure, potentialDescendant), nil
}

// AncesterallyRelated returns true if the given states are ancestorally related.
func (c *Statechart) AncestrallyRelated(state1 StateLabel, state2 StateLabel) (bool, error) {
	ancestor, err := c.Ancestor(state1, state2)
	if err != nil {
		return false, err
	}
	if ancestor {
		return true, nil
	}
	descendant, err := c.Descendant(state2, state1)
	if err != nil {
		return false, err
	}
	return descendant, nil
}

// LeastCommonAncestor returns the least common ancestor of the given states.
func (c *Statechart) LeastCommonAncestor(states ...StateLabel) StateLabel {

	// return root
}

func (s *Statechart) childrenPlus(state *sc.State) ([]StateLabel, error) {
	result := make([]StateLabel, len(state.Children))
	for i, child := range state.Children {
		result[i] = StateLabel(child.Label)
		children, err := s.childrenPlus(child)
		if err != nil {
			return nil, err
		}
		result = append(result, children...)
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
