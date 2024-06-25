package semantics

import (
	"errors"
	"fmt"

	"github.com/tmc/sc"
	"golang.org/x/exp/slices"
)

var (
	// ErrInconsistent is returned if a Statechart is inconsistent.
	ErrInconsistent = errors.New("inconsistent statechart")
	// ErrNotFound is returned when a state is not found.
	ErrNotFound = errors.New("state not found")
)

// statesContains returns true if the given slice of states contains the given state.
func statesContains(states []StateLabel, state StateLabel) bool {
	return slices.Contains(states, state)
}

// Children returns the immediate children of the given state.
func (c *Statechart) Children(state StateLabel) ([]StateLabel, error) {
	s, err := c.findState(state)
	if err != nil {
		return nil, fmt.Errorf("failed to find state: %w", err)
	}
	var result []StateLabel
	for _, child := range s.Children {
		result = append(result, StateLabel(child.Label))
	}
	return result, nil
}

// ChildrenStar returns the reflexive-transitive closure of the children of the given state.
func (c *Statechart) ChildrenStar(state StateLabel) ([]StateLabel, error) {
	s, err := c.findState(state)
	if err != nil {
		return nil, fmt.Errorf("failed to find state: %w", err)
	}
	result := []StateLabel{state}
	children, err := c.childrenPlus(s)
	if err != nil {
		return nil, fmt.Errorf("failed to get children plus: %w", err)
	}
	result = append(result, children...)
	return result, nil
}

// ChildrenPlus returns the transitive closure of the children of the given state.
func (c *Statechart) ChildrenPlus(state StateLabel) ([]StateLabel, error) {
	s, err := c.findState(state)
	if err != nil {
		return nil, fmt.Errorf("failed to find state: %w", err)
	}
	return c.childrenPlus(s)
}

// childrenPlus is a helper function that does the actual work of ChildrenPlus
func (s *Statechart) childrenPlus(state *sc.State) ([]StateLabel, error) {
	var result []StateLabel
	for _, child := range state.Children {
		result = append(result, StateLabel(child.Label))
		children, err := s.childrenPlus(child)
		if err != nil {
			return nil, fmt.Errorf("failed to get children plus for state %s: %w", child.Label, err)
		}
		result = append(result, children...)
	}
	return result, nil
}

// Descendant returns true if the given state is a descendant of the given potential ancestor.
func (c *Statechart) Descendant(state StateLabel, potentialAncestor StateLabel) (bool, error) {
	_, err := c.findState(state)
	if err != nil {
		return false, err
	}
	_, err = c.findState(potentialAncestor)
	if err != nil {
		return false, err
	}
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
		return false, fmt.Errorf("failed to get children star: %w", err)
	}
	return statesContains(rtClosure, potentialDescendant), nil
}

// AncestrallyRelated returns true if the given states are ancestrally related.
func (c *Statechart) AncestrallyRelated(state1 StateLabel, state2 StateLabel) (bool, error) {
	ancestor, err := c.Ancestor(state1, state2)
	if err != nil {
		return false, err
	}
	if ancestor {
		return true, nil
	}
	return c.Ancestor(state2, state1)
}

// LeastCommonAncestor returns the least common ancestor of the given states.
func (c *Statechart) LeastCommonAncestor(states ...StateLabel) (StateLabel, error) {
	if len(states) == 0 {
		return "", errors.New("no states provided")
	}
	if len(states) == 1 {
		_, err := c.findState(states[0])
		if err != nil {
			return "", fmt.Errorf("failed to find state: %w", err)
		}
		return states[0], nil
	}
	ancestors := make([][]StateLabel, len(states))
	for i, state := range states {
		stateAncestors, err := c.findAncestors(state)
		if err != nil {
			return "", fmt.Errorf("failed to find ancestors of %s: %w", state, err)
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
func (s *Statechart) GetParent(state StateLabel) (*sc.State, error) {
	if state == RootState {
		return nil, nil // Root has no parent
	}

	queue := []*sc.State{s.RootState}
	for len(queue) > 0 {
		current := queue[0]
		queue = queue[1:]

		for _, child := range current.Children {
			if StateLabel(child.Label) == state {
				return current, nil
			}
			queue = append(queue, child)
		}
	}

	return nil, fmt.Errorf("parent not found for state %s", state)
}

// findAncestors returns the ancestors of the given state.
func (c *Statechart) findAncestors(state StateLabel) ([]StateLabel, error) {
	ancestors := []StateLabel{state}
	currentState := state

	for {
		parent, err := c.GetParent(currentState)
		if err != nil {
			return nil, fmt.Errorf("failed to get parent of %s: %w", currentState, err)
		}
		ancestors = append(ancestors, StateLabel(parent.Label))
		currentState = StateLabel(parent.Label)
		if parent.Label == c.RootState.Label {
			break
		}
	}

	return ancestors, nil
}

// findState finds the state with the given label.
func (s *Statechart) findState(label StateLabel) (*sc.State, error) {
	queue := []*sc.State{s.RootState}
	for len(queue) > 0 {
		state := queue[0]
		queue = queue[1:]

		if state.Label == string(label) {
			return state, nil
		}

		queue = append(queue, state.Children...)
	}
	return nil, fmt.Errorf("state %s not found", label)
}

// Default returns the default state of the given state.
func (s *Statechart) Default(state StateLabel) (StateLabel, error) {
	stateObj, err := s.findState(state)
	if err != nil {
		return "", fmt.Errorf("failed to find state %s: %w", state, err)
	}
	for _, child := range stateObj.Children {
		if child.IsInitial {
			return StateLabel(child.Label), nil
		}
	}
	return "", fmt.Errorf("no default state found for %s", state)
}

// Orthogonal returns true if the given state is orthogonal.
// Two states x, y, are orthogonal, written x⊥y, if x and y are not ancestrally related, and their lca is an AND state.
func (s *Statechart) Orthogonal(state1, state2 StateLabel) (bool, error) {
	state1Obj, err := s.findState(state1)
	if err != nil {
		return false, fmt.Errorf("failed to find state %s: %w", state1, err)
	}
	state2Obj, err := s.findState(state2)
	if err != nil {
		return false, fmt.Errorf("failed to find state %s: %w", state2, err)
	}
	if state1Obj == state2Obj {
		return false, nil
	}
	lca, err := s.LeastCommonAncestor(state1, state2)
	if err != nil {
		return false, fmt.Errorf("failed to find least common ancestor: %w", err)
	}
	lcaObj, err := s.findState(lca)
	if err != nil {
		return false, fmt.Errorf("failed to find LCA state %s: %w", lca, err)
	}
	return lcaObj.Type == sc.StateTypeParallel, nil
}

// Consistent returns true if the given set of states is consistent.
// A set X of states is consistent if for every x, y ∈ X, either x and y are ancestrally related or x⊥y (orthogonal).
func (s *Statechart) Consistent(states ...StateLabel) (bool, error) {
	for i, state1 := range states {
		// check that states are valid/present.
		if _, err := s.findState(state1); err != nil {
			return false, fmt.Errorf("failed to find state %s: %w", state1, err)
		}
		for _, state2 := range states[i+1:] {
			ancestrallyRelated, err := s.AncestrallyRelated(state1, state2)
			if err != nil {
				return false, fmt.Errorf("failed to check ancestral relation: %w", err)
			}
			if !ancestrallyRelated {
				orthogonal, err := s.Orthogonal(state1, state2)
				if err != nil {
					return false, fmt.Errorf("failed to check orthogonality: %w", err)
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
		return nil, fmt.Errorf("failed to check consistency: %w", err)
	}
	if !c {
		return nil, ErrInconsistent
	}
	return s.defaultCompletion(states...)
}

// defaultCompletion returns the default completion of the given state.
//
// The default completion of a state x is the set of states that are active when x is entered.
//
// From the paper:
// • states ⊆ D
// • if s ∈ D and type(s) = AND then children(s) ⊆ D
// • if s ∈ D and type(s) = OR and children(s) ∩ states = ∅ then default(s) ∈ D
// • if s ∈ D and s != root then parent(s) ∈ D.
func (s *Statechart) defaultCompletion(states ...StateLabel) ([]StateLabel, error) {
	var activeStates []StateLabel
	var defaultCompletion func(state *sc.State) error
	// First check if the input states are consistent.
	if consistent, err := s.Consistent(states...); err != nil || !consistent {
		return nil, ErrInconsistent
	}
	defaultCompletion = func(state *sc.State) error {
		// if state already in activeStates, skip.
		for _, activeState := range activeStates {
			if StateLabel(state.Label) == activeState {
				return nil
			}
		}
		activeStates = append(activeStates, StateLabel(state.Label))
		if state.Type == sc.StateTypeNormal {
			addDefault := true
			for _, child := range state.Children {
				for _, activeState := range activeStates {
					if StateLabel(child.Label) == activeState {
						addDefault = false
						break
					}
				}
			}
			if addDefault {
				defaultState, err := s.Default(StateLabel(state.Label))
				if err != nil {
					return fmt.Errorf("failed to get default state: %w", err)
				}
				activeStates = append(activeStates, defaultState)
			}
		} else if state.Type == sc.StateTypeParallel {
			for _, child := range state.Children {
				if err := defaultCompletion(child); err != nil {
					return fmt.Errorf("failed to complete default for child %s: %w", child.Label, err)
				}
			}
		}
		if state.Label != s.RootState.Label {
			parent, err := s.GetParent(StateLabel(state.Label))
			if err != nil {
				return fmt.Errorf("failed to get parent: %w", err)
			}
			if err := defaultCompletion(parent); err != nil {
				return fmt.Errorf("failed to complete default for parent %s: %w", parent.Label, err)
			}
		}

		return nil
	}

	for _, stateLabel := range states {
		state, err := s.findState(stateLabel)
		if err != nil {
			return nil, fmt.Errorf("failed to find state %s: %w", stateLabel, err)
		}
		err = defaultCompletion(state)
		if err != nil {
			return nil, fmt.Errorf("failed to complete default for state %s: %w", stateLabel, err)
		}
	}

	// Filter out the root state from the result
	var filteredStates []StateLabel
	for _, state := range activeStates {
		if state != RootState {
			filteredStates = append(filteredStates, state)
		}
	}

	return filteredStates, nil
}
