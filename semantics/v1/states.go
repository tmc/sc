package semantics

import (
	"errors"
	"fmt"

	"github.com/tmc/sc"
	"golang.org/x/exp/slices"
)

var (
	// ErrInconsistent is returned if a Statechart is inconsistent.
	ErrInconsistent = errors.New("inconsistent")
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
func (c *Statechart) LeastCommonAncestor(states ...StateLabel) (StateLabel, error) {
	if len(states) == 0 {
		return "", errors.New("no states provided")
	}
	if len(states) == 1 {
		_, err := c.findState(states[0])
		if err != nil {
			return "", err
		}
		return states[0], nil
	}
	ancestors := make([][]StateLabel, len(states))
	for i, state := range states {
		stateAncestors, err := c.findAncestors(state)
		if err != nil {
			return "", fmt.Errorf("failed to find ancestors of %s: %v", state, err)
		}
		ancestors[i] = stateAncestors
	}
	// Find the first ancestor that is common to all states.
	var lca StateLabel
	minAncestorsLength := len(ancestors[0])
	minAncestors := ancestors[0]
	for _, otherAncestors := range ancestors[1:] {
		if len(otherAncestors) < minAncestorsLength {
			minAncestorsLength = len(otherAncestors)
			minAncestors = otherAncestors
		}
	}

	for _, ancestor := range minAncestors {
		allContain := true
		for _, otherAncestors := range ancestors {
			if !statesContains(otherAncestors, ancestor) {
				allContain = false
				break
			}
		}
		if allContain {
			lca = ancestor
			break
		}
	}
	return lca, nil
}

// GetParent returns the parent state of the given state.
func (c *Statechart) GetParent(state StateLabel) (*sc.State, error) {
	s, err := c.findState(state)
	if err != nil {
		return nil, err
	}
	// recurse down the tree until we find the root state.
	return c.getParent(s, c.RootState)
}

// getParent returns the parent of the given state.
func (c *Statechart) getParent(needle *sc.State, haystack *sc.State) (*sc.State, error) {
	if haystack == nil {
		return nil, fmt.Errorf("nil haystack")
	}
	for _, child := range haystack.Children {
		if child == needle {
			return haystack, nil
		}
		parent, err := c.getParent(needle, child)
		if err == nil {
			return parent, nil
		}
	}
	return nil, errors.New("no parent found")
}

// findAncestors returns the ancestors of the given state.
func (c *Statechart) findAncestors(state StateLabel) ([]StateLabel, error) {
	ancestors := []StateLabel{state}
	currentState := state

	for {
		parent, err := c.GetParent(currentState)
		if err != nil {
			return nil, err
		}
		ancestors = append(ancestors, StateLabel(parent.Label))
		currentState = StateLabel(parent.Label)
		if parent.Label == c.RootState.Label {
			break
		}
	}

	return ancestors, nil
}

// childrenPlus returns the transitive closure of the children of the given state.
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

// Default returns the default state of the given state.
func (s *Statechart) Default(state StateLabel) (StateLabel, error) {
	stateObj, err := s.findState(state)
	if err != nil {
		return "", err
	}
	for _, child := range stateObj.Children {
		if child.IsInitial {
			return StateLabel(child.Label), nil
		}
	}
	return "", errors.New("no default state found")
}

// Orthogonal returns true if the given state is orthogonal.
// Two states x, y, are orthogonal, written x⊥y, if x and y are not ancestrally related, and their lca is an AND state.
func (s *Statechart) Orthogonal(state1, state2 StateLabel) (bool, error) {
	state1Obj, err := s.findState(state1)
	if err != nil {
		return false, err
	}
	state2Obj, err := s.findState(state2)
	if err != nil {
		return false, err
	}
	if state1Obj == state2Obj {
		return false, nil
	}
	lca, err := s.LeastCommonAncestor(state1, state2)
	if err != nil {
		return false, err
	}
	lcaObj, err := s.findState(lca)
	if err != nil {
		return false, err
	}
	return lcaObj.Type == sc.StateTypeParallel, nil
}

// Consistent returns true if the given set of states is consistent.
// A set X of states is consistent if for every x, y ∈ X, either x and y are ancestrally related or x⊥y (orthogonal).
func (s *Statechart) Consistent(states ...StateLabel) (bool, error) {
	for i, state1 := range states {
		// cehck that states are valid/present.
		if _, err := s.findState(state1); err != nil {
			return false, err
		}
		for _, state2 := range states[i+1:] {
			ancestrallyRelated, err := s.AncestrallyRelated(state1, state2)
			if err != nil {
				return false, err
			}
			if !ancestrallyRelated {
				orthogonal, err := s.Orthogonal(state1, state2)
				if err != nil {
					return false, err
				}
				if !orthogonal {
					return false, nil
				}
			}
		}
	}
	return true, nil
}

// DefaultCompletion returns the default completion of the given state.
func (s *Statechart) DefaultCompletion(states ...StateLabel) ([]StateLabel, error) {
	// First check if the input states are consistent.
	c, err := s.Consistent(states...)
	if err != nil {
		return nil, err
	}
	if !c {
		return nil, ErrInconsistent
	}

	return s.defaultCompletion(states...)
}

// defaultCompletion
//
// Given a consistent set X of nodes, the default completion dcomp(X) is the smallest set D such that:
// • X ⊆ D
// • if s ∈ D and type(s) = AND then children(s) ⊆ D
// • if s ∈ D and type(s) = OR and children(s) ∩ X = ∅ then default(s) ∈ D
// • if s ∈ D and s != root then parent(s) ∈ D.
func (s *Statechart) defaultCompletion(states ...StateLabel) ([]StateLabel, error) {
	// Initialize the result to the input states.
	result := make([]StateLabel, len(states))
	copy(result, states)

	// Calculate the default completion for each OR state in the input states.
	// Add it if and only if children of a candidate state are not already present.
	for _, state := range states {
		if st, err := s.getState(state); err == nil && st.Type == sc.StateTypeOr {
			children := st.Children
			if s.Contains(children, s.Intersect(children, states)) {
				defState, err := s.DefaultState(state)
				if err != nil {
					return nil, err
				}
				result = append(result, defState)
			}
		}
	}

	// Add the ancestors of the input states to the result, unless they're already present.
	for _, state := range states {
		for _, ancestor := range s.GetAncestors(state) {
			if !s.Contains(result, ancestor) {
				result = append(result, ancestor)
			}
		}
	}

	// Add the parent of each input state to the result, unless they're already present.
	for _, state := range states {
		parent, _, err := s.GetParent(state)
		if err != nil {
			return nil, err
		}
		if !s.Contains(result, parent) {
			result = append(result, parent)
		}
	}

	// Add the root state to the result.
	if !s.Contains(result, StateLabel(s.RootState.Label)) {
		result = append(result, StateLabel(s.RootState.Label))
	}

	return result, nil
}
