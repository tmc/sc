package semantics

import (
	"fmt"
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/tmc/sc"
)

func TestConfigurationValidity(t *testing.T) {
	statechart := &sc.Statechart{
		RootState: &sc.State{
			Label: "Root",
			Children: []*sc.State{
				{
					Label: "A",
					Children: []*sc.State{
						{Label: "A1"},
						{Label: "A2"},
					},
				},
				{
					Label: "B",
					Type:  sc.StateTypeParallel,
					Children: []*sc.State{
						{
							Label: "B1",
							Children: []*sc.State{
								{Label: "B1a"},
								{Label: "B1b"},
							},
						},
						{
							Label: "B2",
							Children: []*sc.State{
								{Label: "B2a"},
								{Label: "B2b"},
							},
						},
					},
				},
			},
		},
	}

	tests := []struct {
		name      string
		config    *sc.Configuration
		wantValid bool
	}{
		{
			name: "Valid configuration",
			config: &sc.Configuration{
				States: []*sc.StateRef{
					{Label: "A"},
					{Label: "A1"},
				},
			},
			wantValid: true,
		},
		{
			name: "Valid parallel configuration",
			config: &sc.Configuration{
				States: []*sc.StateRef{
					{Label: "B"},
					{Label: "B1"},
					{Label: "B1a"},
					{Label: "B2"},
					{Label: "B2b"},
				},
			},
			wantValid: true,
		},
		{
			name: "Invalid: multiple children of XOR state",
			config: &sc.Configuration{
				States: []*sc.StateRef{
					{Label: "A"},
					{Label: "A1"},
					{Label: "A2"},
				},
			},
			wantValid: false,
		},
		{
			name: "Invalid: incomplete parallel state",
			config: &sc.Configuration{
				States: []*sc.StateRef{
					{Label: "B"},
					{Label: "B1"},
					{Label: "B1a"},
				},
			},
			wantValid: false,
		},
		{
			name: "Invalid: nonexistent state",
			config: &sc.Configuration{
				States: []*sc.StateRef{
					{Label: "NonexistentState"},
				},
			},
			wantValid: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			valid := isValidConfiguration(statechart, tt.config)
			if valid != tt.wantValid {
				t.Errorf("isValidConfiguration() = %v, want %v", valid, tt.wantValid)
			}
		})
	}
}

func TestConfigurationTransitions(t *testing.T) {
	statechart := &sc.Statechart{
		RootState: &sc.State{
			Label: "Root",
			Children: []*sc.State{
				{Label: "A"},
				{Label: "B"},
				{Label: "C"},
			},
		},
		Transitions: []*sc.Transition{
			{From: []string{"A"}, To: []string{"B"}, Event: "AB"},
			{From: []string{"B"}, To: []string{"C"}, Event: "BC"},
			{From: []string{"C"}, To: []string{"A"}, Event: "CA"},
		},
	}

	tests := []struct {
		name           string
		initialConfig  *sc.Configuration
		event          string
		wantNewConfig  *sc.Configuration
		wantTransition bool
	}{
		{
			name: "A to B",
			initialConfig: &sc.Configuration{
				States: []*sc.StateRef{{Label: "A"}},
			},
			event: "AB",
			wantNewConfig: &sc.Configuration{
				States: []*sc.StateRef{{Label: "B"}},
			},
			wantTransition: true,
		},
		{
			name: "B to C",
			initialConfig: &sc.Configuration{
				States: []*sc.StateRef{{Label: "B"}},
			},
			event: "BC",
			wantNewConfig: &sc.Configuration{
				States: []*sc.StateRef{{Label: "C"}},
			},
			wantTransition: true,
		},
		{
			name: "No transition",
			initialConfig: &sc.Configuration{
				States: []*sc.StateRef{{Label: "A"}},
			},
			event: "BC",
			wantNewConfig: &sc.Configuration{
				States: []*sc.StateRef{{Label: "A"}},
			},
			wantTransition: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			newConfig, transitioned := transitionConfiguration(statechart, tt.initialConfig, tt.event)
			if transitioned != tt.wantTransition {
				t.Errorf("transitionConfiguration() transitioned = %v, want %v", transitioned, tt.wantTransition)
			}
			if !cmp.Equal(newConfig, tt.wantNewConfig) {
				t.Errorf("transitionConfiguration() newConfig diff (-want +got):\n%s", cmp.Diff(tt.wantNewConfig, newConfig))
			}
		})
	}
}

// Helper functions (these would be implemented in your actual code)

func isValidConfiguration(statechart *sc.Statechart, config *sc.Configuration) bool {
	// This is a simplified validity check. A real implementation would be more complex.
	stateMap := make(map[string]bool)
	for _, state := range config.States {
		stateMap[state.Label] = true
	}

	var dfs func(*sc.State) bool
	dfs = func(state *sc.State) bool {
		if stateMap[state.Label] {
			if state.Type == sc.StateTypeParallel {
				for _, child := range state.Children {
					if !dfs(child) {
						return false
					}
				}
			} else {
				childCount := 0
				for _, child := range state.Children {
					if dfs(child) {
						childCount++
					}
				}
				if childCount > 1 {
					return false
				}
			}
			return true
		}
		return false
	}

	return dfs(statechart.RootState)
}

func transitionConfiguration(statechart *sc.Statechart, config *sc.Configuration, event string) (*sc.Configuration, bool) {
	for _, transition := range statechart.Transitions {
		if transition.Event == event && contains(transition.From, config.States[0].Label) {
			return &sc.Configuration{
				States: []*sc.StateRef{{Label: transition.To[0]}},
			}, true
		}
	}
	return config, false
}

func TestConfigurationConsistency(t *testing.T) {
	statechart := &sc.Statechart{
		RootState: &sc.State{
			Label: "Root",
			Children: []*sc.State{
				{
					Label: "A",
					Children: []*sc.State{
						{Label: "A1"},
						{Label: "A2"},
					},
				},
				{
					Label: "B",
					Type:  sc.StateTypeParallel,
					Children: []*sc.State{
						{
							Label: "B1",
							Children: []*sc.State{
								{Label: "B1a"},
								{Label: "B1b"},
							},
						},
						{
							Label: "B2",
							Children: []*sc.State{
								{Label: "B2a"},
								{Label: "B2b"},
							},
						},
					},
				},
			},
		},
	}

	tests := []struct {
		name           string
		config         *sc.Configuration
		wantConsistent bool
	}{
		{
			name: "Consistent configuration",
			config: &sc.Configuration{
				States: []*sc.StateRef{
					{Label: "A"},
					{Label: "A1"},
				},
			},
			wantConsistent: true,
		},
		{
			name: "Consistent parallel configuration",
			config: &sc.Configuration{
				States: []*sc.StateRef{
					{Label: "B"},
					{Label: "B1"},
					{Label: "B1a"},
					{Label: "B2"},
					{Label: "B2b"},
				},
			},
			wantConsistent: true,
		},
		{
			name: "Inconsistent: multiple children of XOR state",
			config: &sc.Configuration{
				States: []*sc.StateRef{
					{Label: "A"},
					{Label: "A1"},
					{Label: "A2"},
				},
			},
			wantConsistent: false,
		},
		{
			name: "Inconsistent: incomplete parallel state",
			config: &sc.Configuration{
				States: []*sc.StateRef{
					{Label: "B"},
					{Label: "B1"},
					{Label: "B1a"},
				},
			},
			wantConsistent: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			consistent := isConsistentConfiguration(statechart, tt.config)
			if consistent != tt.wantConsistent {
				t.Errorf("isConsistentConfiguration() = %v, want %v", consistent, tt.wantConsistent)
			}
		})
	}
}

func TestConfigurationCompletion(t *testing.T) {
	statechart := &sc.Statechart{
		RootState: &sc.State{
			Label: "Root",
			Children: []*sc.State{
				{
					Label: "A",
					Children: []*sc.State{
						{Label: "A1", IsInitial: true},
						{Label: "A2"},
					},
				},
				{
					Label: "B",
					Type:  sc.StateTypeParallel,
					Children: []*sc.State{
						{
							Label: "B1",
							Children: []*sc.State{
								{Label: "B1a", IsInitial: true},
								{Label: "B1b"},
							},
						},
						{
							Label: "B2",
							Children: []*sc.State{
								{Label: "B2a", IsInitial: true},
								{Label: "B2b"},
							},
						},
					},
				},
			},
		},
	}

	tests := []struct {
		name          string
		initialConfig *sc.Configuration
		wantConfig    *sc.Configuration
	}{
		{
			name: "Complete A",
			initialConfig: &sc.Configuration{
				States: []*sc.StateRef{{Label: "A"}},
			},
			wantConfig: &sc.Configuration{
				States: []*sc.StateRef{
					{Label: "A"},
					{Label: "A1"},
				},
			},
		},
		{
			name: "Complete B",
			initialConfig: &sc.Configuration{
				States: []*sc.StateRef{{Label: "B"}},
			},
			wantConfig: &sc.Configuration{
				States: []*sc.StateRef{
					{Label: "B"},
					{Label: "B1"},
					{Label: "B1a"},
					{Label: "B2"},
					{Label: "B2a"},
				},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			completedConfig := completeConfiguration(statechart, tt.initialConfig)
			if !cmp.Equal(completedConfig, tt.wantConfig) {
				t.Errorf("completeConfiguration() diff (-want +got):\n%s", cmp.Diff(tt.wantConfig, completedConfig))
			}
		})
	}
}

// Helper functions (these would be implemented in your actual code)

func isConsistentConfiguration(statechart *sc.Statechart, config *sc.Configuration) bool {
	// This is a simplified consistency check. A real implementation would be more complex.
	stateMap := make(map[string]bool)
	for _, state := range config.States {
		stateMap[state.Label] = true
	}

	var dfs func(*sc.State) bool
	dfs = func(state *sc.State) bool {
		if stateMap[state.Label] {
			if state.Type == sc.StateTypeParallel {
				for _, child := range state.Children {
					if !dfs(child) {
						return false
					}
				}
			} else {
				childCount := 0
				for _, child := range state.Children {
					if dfs(child) {
						childCount++
					}
				}
				if childCount != 1 {
					return false
				}
			}
			return true
		}
		return false
	}

	return dfs(statechart.RootState)
}
func completeConfiguration(statechart *sc.Statechart, config *sc.Configuration) *sc.Configuration {
	completedStates := make([]*sc.StateRef, 0)
	stateMap := make(map[string]bool)
	for _, state := range config.States {
		stateMap[state.Label] = true
		completedStates = append(completedStates, state)
	}

	var complete func(*sc.State)
	complete = func(state *sc.State) {
		if stateMap[state.Label] {
			if state.Type == sc.StateTypeParallel {
				for _, child := range state.Children {
					complete(child)
				}
			} else {
				for _, child := range state.Children {
					if child.IsInitial && !stateMap[child.Label] {
						stateMap[child.Label] = true
						completedStates = append(completedStates, &sc.StateRef{Label: child.Label})
						complete(child)
						break
					}
				}
			}
		}
	}

	complete(statechart.RootState)

	return &sc.Configuration{
		States: completedStates,
	}
}

func TestConfigurationLeastCommonAncestor(t *testing.T) {
	statechart := &sc.Statechart{
		RootState: &sc.State{
			Label: "Root",
			Children: []*sc.State{
				{
					Label: "A",
					Children: []*sc.State{
						{Label: "A1"},
						{Label: "A2"},
					},
				},
				{
					Label: "B",
					Children: []*sc.State{
						{Label: "B1"},
						{Label: "B2"},
					},
				},
			},
		},
	}

	tests := []struct {
		name      string
		states    []string
		wantLCA   string
		wantError bool
	}{
		{
			name:      "LCA of A1 and A2",
			states:    []string{"A1", "A2"},
			wantLCA:   "A",
			wantError: false,
		},
		{
			name:      "LCA of A1 and B1",
			states:    []string{"A1", "B1"},
			wantLCA:   "Root",
			wantError: false,
		},
		{
			name:      "LCA of A and B",
			states:    []string{"A", "B"},
			wantLCA:   "Root",
			wantError: false,
		},
		{
			name:      "LCA of A1 and nonexistent state",
			states:    []string{"A1", "C"},
			wantLCA:   "",
			wantError: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			lca, err := leastCommonAncestor(statechart, tt.states...)
			if (err != nil) != tt.wantError {
				t.Errorf("leastCommonAncestor() error = %v, wantError %v", err, tt.wantError)
				return
			}
			if lca != tt.wantLCA {
				t.Errorf("leastCommonAncestor() = %v, want %v", lca, tt.wantLCA)
			}
		})
	}
}

func TestConfigurationOrthogonality(t *testing.T) {
	statechart := &sc.Statechart{
		RootState: &sc.State{
			Label: "Root",
			Children: []*sc.State{
				{
					Label: "A",
					Type:  sc.StateTypeParallel,
					Children: []*sc.State{
						{Label: "A1"},
						{Label: "A2"},
					},
				},
				{
					Label: "B",
					Children: []*sc.State{
						{Label: "B1"},
						{Label: "B2"},
					},
				},
			},
		},
	}

	tests := []struct {
		name           string
		state1         string
		state2         string
		wantOrthogonal bool
		wantError      bool
	}{
		{
			name:           "A1 and A2 are orthogonal",
			state1:         "A1",
			state2:         "A2",
			wantOrthogonal: true,
			wantError:      false,
		},
		{
			name:           "A1 and B1 are not orthogonal",
			state1:         "A1",
			state2:         "B1",
			wantOrthogonal: false,
			wantError:      false,
		},
		{
			name:           "B1 and B2 are not orthogonal",
			state1:         "B1",
			state2:         "B2",
			wantOrthogonal: false,
			wantError:      false,
		},
		{
			name:           "A1 and nonexistent state",
			state1:         "A1",
			state2:         "C",
			wantOrthogonal: false,
			wantError:      true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			orthogonal, err := areOrthogonal(statechart, tt.state1, tt.state2)
			if (err != nil) != tt.wantError {
				t.Errorf("areOrthogonal() error = %v, wantError %v", err, tt.wantError)
				return
			}
			if orthogonal != tt.wantOrthogonal {
				t.Errorf("areOrthogonal() = %v, want %v", orthogonal, tt.wantOrthogonal)
			}
		})
	}
}

// Helper functions (these would be implemented in your actual code)

func leastCommonAncestor(statechart *sc.Statechart, states ...string) (string, error) {
	var findPath func(*sc.State, string) ([]string, bool)
	findPath = func(state *sc.State, target string) ([]string, bool) {
		if state.Label == target {
			return []string{state.Label}, true
		}
		for _, child := range state.Children {
			if path, found := findPath(child, target); found {
				return append([]string{state.Label}, path...), true
			}
		}
		return nil, false
	}

	var paths [][]string
	for _, state := range states {
		path, found := findPath(statechart.RootState, state)
		if !found {
			return "", fmt.Errorf("state not found: %s", state)
		}
		paths = append(paths, path)
	}

	minLen := len(paths[0])
	for _, path := range paths[1:] {
		if len(path) < minLen {
			minLen = len(path)
		}
	}

	for i := 0; i < minLen; i++ {
		for j := 1; j < len(paths); j++ {
			if paths[j][i] != paths[0][i] {
				if i == 0 {
					return "", fmt.Errorf("no common ancestor")
				}
				return paths[0][i-1], nil
			}
		}
	}

	return paths[0][minLen-1], nil
}

func areOrthogonal(statechart *sc.Statechart, state1, state2 string) (bool, error) {
	lca, err := leastCommonAncestor(statechart, state1, state2)
	if err != nil {
		return false, err
	}

	var findState func(*sc.State, string) *sc.State
	findState = func(state *sc.State, target string) *sc.State {
		if state.Label == target {
			return state
		}
		for _, child := range state.Children {
			if found := findState(child, target); found != nil {
				return found
			}
		}
		return nil
	}

	lcaState := findState(statechart.RootState, lca)
	return lcaState != nil && lcaState.Type == sc.StateTypeParallel, nil
}
