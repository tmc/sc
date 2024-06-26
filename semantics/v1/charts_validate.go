package semantics

import (
	"fmt"

	"github.com/tmc/sc"
)

func (s *Statechart) Validate() error {
	if err := s.validateRootState(); err != nil {
		return fmt.Errorf("invalid root state: %w", err)
	}
	if err := s.validateParentChildRelationships(); err != nil {
		return fmt.Errorf("invalid parent-child relationship: %w", err)
	}
	if err := s.validateNonOverlappingStateLabels(); err != nil {
		return fmt.Errorf("overlapping state labels: %w", err)
	}
	if err := s.validateStateTypeAgreesWithChildren(); err != nil {
		return fmt.Errorf("state type mismatch: %w", err)
	}
	if err := s.validateParentStatesHaveSingleDefaults(); err != nil {
		return fmt.Errorf("multiple default states: %w", err)
	}
	return nil
}

func (s *Statechart) validateNonOverlappingStateLabels() error {
	if s.RootState == nil {
		return nil // This will be caught by validateRootState
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
	return checkLabels(s.RootState)
}

func (s *Statechart) validateRootState() error {
	if s.RootState == nil {
		return fmt.Errorf("root state is nil")
	}
	if s.RootState.Label != RootState.String() {
		return fmt.Errorf("root state has an unexpected label of '%s' (expected '%s')", s.RootState.Label, RootState.String())
	}
	return nil
}

func (s *Statechart) validateStateTypeAgreesWithChildren() error {
	var checkType func(*sc.State) error
	checkType = func(state *sc.State) error {
		switch state.Type {
		case sc.StateTypeBasic:
			if len(state.Children) > 0 {
				return fmt.Errorf("basic state %s has children", state.Label)
			}
		case sc.StateTypeNormal, sc.StateTypeParallel:
			if len(state.Children) == 0 {
				return fmt.Errorf("compound state %s has no children", state.Label)
			}
		}
		for _, child := range state.Children {
			if err := checkType(child); err != nil {
				return err
			}
		}
		return nil
	}
	return checkType(s.RootState)
}

func (s *Statechart) validateParentChildRelationships() error {
	var checkRelationships func(*sc.State) error
	checkRelationships = func(state *sc.State) error {
		for _, child := range state.Children {
			parent, err := s.GetParent(StateLabel(child.Label))
			if err != nil {
				return fmt.Errorf("failed to get parent of %s: %w", child.Label, err)
			}
			if parent != state {
				return fmt.Errorf("inconsistent parent-child relationship for %s", child.Label)
			}
			if err := checkRelationships(child); err != nil {
				return err
			}
		}
		return nil
	}
	return checkRelationships(s.RootState)
}

func (s *Statechart) validateParentStatesHaveSingleDefaults() error {
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
	return checkDefaults(s.RootState)
}
