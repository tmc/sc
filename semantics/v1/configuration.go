package semantics

import (
	"fmt"
	"sort"

	"github.com/tmc/sc"
)

// ValidateConfiguration checks if a configuration is valid according to the following rules:
// 1. For each AND-state (parallel state) in C, all its children are in C.
// 2. For each OR-state (normal state) in C, exactly one of its children is in C.
// 3. If a state is in C, then its parent is in C (except for the root).
func ValidateConfiguration(statechart *Statechart, config *sc.Configuration) error {
	if statechart == nil || statechart.RootState == nil || config == nil {
		return fmt.Errorf("invalid input: statechart or configuration is nil")
	}

	stateMap := make(map[string]bool)
	for _, state := range config.States {
		if state != nil {
			stateMap[state.Label] = true
		}
	}

	var validateState func(*sc.State) error
	validateState = func(state *sc.State) error {
		if state == nil {
			return fmt.Errorf("encountered nil state")
		}

		if !stateMap[state.Label] {
			return nil // State not in configuration, which is valid
		}

		switch state.Type {
		case sc.StateTypeParallel:
			for _, child := range state.Children {
				if child == nil {
					return fmt.Errorf("encountered nil child state in %s", state.Label)
				}
				if !stateMap[child.Label] {
					return fmt.Errorf("child %s of AND-state %s is not in the configuration", child.Label, state.Label)
				}
				if err := validateState(child); err != nil {
					return err
				}
			}
		case sc.StateTypeNormal:
			activeChildren := 0
			for _, child := range state.Children {
				if child == nil {
					return fmt.Errorf("encountered nil child state in %s", state.Label)
				}
				if stateMap[child.Label] {
					activeChildren++
					if err := validateState(child); err != nil {
						return err
					}
				}
			}
			if len(state.Children) > 0 && activeChildren != 1 {
				return fmt.Errorf("OR-state %s has %d active children, expected exactly 1", state.Label, activeChildren)
			}
		}

		return nil
	}

	if err := validateState(statechart.RootState); err != nil {
		return err
	}

	// Check parent-child relationships
	for label := range stateMap {
		state, err := statechart.findState(StateLabel(label))
		if err != nil {
			return fmt.Errorf("state %s not found in statechart", label)
		}
		if state != statechart.RootState {
			parent, err := statechart.GetParent(StateLabel(state.Label))
			if err != nil {
				return fmt.Errorf("failed to get parent of %s: %v", state.Label, err)
			}
			if parent != nil && !stateMap[parent.Label] {
				return fmt.Errorf("state %s is in configuration but its parent %s is not", state.Label, parent.Label)
			}
		}
	}

	return nil
}

// IsConsistentConfiguration checks if a configuration is consistent according to the following rules:
// 1. The configuration is valid (satisfies all rules of ValidateConfiguration).
// 2. The configuration is closed under default completion (includes all default states down to the leaves).
func IsConsistentConfiguration(statechart *Statechart, config *sc.Configuration) (bool, error) {
	// Rule 1: The configuration must be valid
	if err := ValidateConfiguration(statechart, config); err != nil {
		return false, err
	}

	// Rule 2: The configuration must be closed under default completion
	completedConfig, err := DefaultCompletion(statechart, config)
	if err != nil {
		return false, fmt.Errorf("issue computing the default completion: %w", err)
	}
	if completedConfig == nil {
		return false, fmt.Errorf("failed to compute default completion")
	}

	// Compare original and completed configurations
	originalSet := make(map[string]bool)
	for _, state := range config.States {
		if state != nil {
			originalSet[state.Label] = true
		}
	}

	completedSet := make(map[string]bool)
	for _, state := range completedConfig.States {
		if state != nil {
			completedSet[state.Label] = true
		}
	}

	// Check if the completed configuration includes all states from the original configuration
	for label := range originalSet {
		if !completedSet[label] {
			return false, nil
		}
	}

	// Check if the completed configuration has any additional states
	for label := range completedSet {
		if !originalSet[label] {
			return false, nil
		}
	}

	return true, nil
}

// DefaultCompletion computes the default completion of a configuration.
// It adds default states to the configuration until no more default states can be added.
//
// A default completion D of a set of states X is the smallest set of states such that:
// 1. X ⊆ D (all states in X are in D)
// 2. If s ∈ D and type(s) = AND then children(s) ⊆ D (all children of AND-states in D are in D)
// 3. If s ∈ D and type(s) = OR and children(s) ∩ D = ∅ then default(s) ∈ D (if an OR-state in D has no children in D, its default child is in D)
// 4. If s ∈ D and s ≠ root then parent(s) ∈ D (if a state is in D, its parent is in D, except for the root)
//
// The resulting configuration is guaranteed to be consistent and include all necessary states
// according to the rules of default completion. The states in the output configuration are sorted
// by label for consistency.
func DefaultCompletion(statechart *Statechart, config *sc.Configuration) (*sc.Configuration, error) {
	if statechart == nil || statechart.RootState == nil || config == nil {
		return nil, fmt.Errorf("invalid input: statechart or configuration is nil")
	}

	completed := make(map[string]bool)
	for _, state := range config.States {
		if state != nil {
			completed[state.Label] = true
		}
	}

	var complete func(*sc.State) error
	complete = func(state *sc.State) error {
		if state == nil {
			return fmt.Errorf("encountered nil state")
		}
		if completed[state.Label] {
		} else {
			completed[state.Label] = true
		}

		switch state.Type {
		case sc.StateTypeParallel:
			for _, child := range state.Children {
				if err := complete(child); err != nil {
					return err
				}
			}
		case sc.StateTypeNormal:
			if len(state.Children) > 0 {
				hasActiveChild := false
				for _, child := range state.Children {
					if completed[child.Label] {
						hasActiveChild = true
						if err := complete(child); err != nil {
							return err
						}
						break
					}
				}
				if !hasActiveChild {
					defaultChild := findDefaultChild(state)
					if defaultChild == nil {
						return fmt.Errorf("OR-state %s has no default child", state.Label)
					}
					if err := complete(defaultChild); err != nil {
						return err
					}
				}
			}
		}

		return nil
	}

	// Complete all states in the initial configuration and their descendants
	for _, stateRef := range config.States {
		if stateRef != nil {
			state, err := statechart.findState(StateLabel(stateRef.Label))
			if err != nil {
				return nil, fmt.Errorf("failed to find state %s: %w", stateRef.Label, err)
			}
			if err := complete(state); err != nil {
				return nil, err
			}
		}
	}

	// Ensure all parents are completed
	for label := range completed {
		state, err := statechart.findState(StateLabel(label))
		if err != nil {
			return nil, fmt.Errorf("failed to find state %s: %w", label, err)
		}
		for state != statechart.RootState {
			parent, err := statechart.GetParent(StateLabel(state.Label))
			if err != nil {
				return nil, fmt.Errorf("failed to get parent of %s: %w", state.Label, err)
			}
			if err := complete(parent); err != nil {
				return nil, err
			}
			state = parent
		}
	}

	var completedStates []*sc.StateRef
	for label := range completed {
		completedStates = append(completedStates, &sc.StateRef{Label: label})
	}

	sortedStates, err := TopologicalSort(statechart, completedStates)
	if err != nil {
		return nil, err
	}
	return &sc.Configuration{States: sortedStates}, nil
}

// findDefaultChild returns the default (initial) child of a given state.
// It returns nil if the state has no children or no default child.
func findDefaultChild(state *sc.State) *sc.State {
	for _, child := range state.Children {
		if child.IsInitial {
			return child
		}
	}
	return nil
}

func TopologicalSort(statechart *Statechart, states []*sc.StateRef) ([]*sc.StateRef, error) {
	stateMap := make(map[string]*sc.StateRef)
	for _, state := range states {
		stateMap[state.Label] = state
	}

	var sorted []*sc.StateRef
	visited := make(map[string]bool)

	var visit func(string) error
	visit = func(label string) error {
		if visited[label] {
			return nil
		}
		visited[label] = true

		state, err := statechart.findState(StateLabel(label))
		if err != nil {
			return err
		}

		// Visit parent first
		if state != statechart.RootState {
			parent, err := statechart.GetParent(StateLabel(label))
			if err != nil {
				return err
			}
			if parent != nil && stateMap[parent.Label] != nil {
				if err := visit(parent.Label); err != nil {
					return err
				}
			}
		}

		// Add current state to sorted list
		if stateMap[label] != nil {
			sorted = append(sorted, stateMap[label])
		}

		// Visit children in a deterministic order

		childLabels := make([]string, 0, len(state.Children))
		for _, child := range state.Children {
			if stateMap[child.Label] != nil {
				childLabels = append(childLabels, child.Label)
			}
		}
		sort.Strings(childLabels)
		for _, childLabel := range childLabels {
			if err := visit(childLabel); err != nil {
				return err
			}
		}

		return nil
	}

	// Start with the root state
	if err := visit(statechart.RootState.Label); err != nil {
		return nil, err
	}

	// Visit any remaining states that weren't reached from the root
	remainingLabels := make([]string, 0)
	for label := range stateMap {
		if !visited[label] {
			remainingLabels = append(remainingLabels, label)
		}
	}
	sort.Strings(remainingLabels)
	for _, label := range remainingLabels {
		if err := visit(label); err != nil {
			return nil, err
		}
	}

	return sorted, nil
}
