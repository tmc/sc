package semantics

import (
	"fmt"
	"testing"

	"github.com/google/go-cmp/cmp"
	statecharts "github.com/tmc/sc/gen/statecharts/v1"
)

func TestConfigurationValidity(t *testing.T) {
	statechart := &statecharts.Statechart{
		RootState: &statecharts.State{
			Label: "Root",
			Children: []*statecharts.State{
				{
					Label: "A",
					Children: []*statecharts.State{
						{Label: "A1"},
						{Label: "A2"},
					},
				},
				{
					Label: "B",
					Type:  statecharts.StateType_STATE_TYPE_PARALLEL,
					Children: []*statecharts.State{
						{
							Label: "B1",
							Children: []*statecharts.State{
								{Label: "B1a"},
								{Label: "B1b"},
							},
						},
						{
							Label: "B2",
							Children: []*statecharts.State{
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
		config    *statecharts.Configuration
		wantValid bool
	}{
		{
			name: "Valid configuration",
			config: &statecharts.Configuration{
				States: []*statecharts.StateRef{
					{Label: "A"},
					{Label: "A1"},
				},
			},
			wantValid: true,
		},
		{
			name: "Valid parallel configuration",
			config: &statecharts.Configuration{
				States: []*statecharts.StateRef{
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
			config: &statecharts.Configuration{
				States: []*statecharts.StateRef{
					{Label: "A"},
					{Label: "A1"},
					{Label: "A2"},
				},
			},
			wantValid: false,
		},
		{
			name: "Invalid: incomplete parallel state",
			config: &statecharts.Configuration{
				States: []*statecharts.StateRef{
					{Label: "B"},
					{Label: "B1"},
					{Label: "B1a"},
				},
			},
			wantValid: false,
		},
		{
			name: "Invalid: nonexistent state",
			config: &statecharts.Configuration{
				States: []*statecharts.StateRef{
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
	statechart := &statecharts.Statechart{
		RootState: &statecharts.State{
			Label: "Root",
			Children: []*statecharts.State{
				{Label: "A"},
				{Label: "B"},
				{Label: "C"},
			},
		},
		Transitions: []*statecharts.Transition{
			{From: []string{"A"}, To: []string{"B"}, Event: "AB"},
			{From: []string{"B"}, To: []string{"C"}, Event: "BC"},
			{From: []string{"C"}, To: []string{"A"}, Event: "CA"},
		},
	}

	tests := []struct {
		name           string
		initialConfig  *statecharts.Configuration
		event          string
		wantNewConfig  *statecharts.Configuration
		wantTransition bool
	}{
		{
			name: "A to B",
			initialConfig: &statecharts.Configuration{
				States: []*statecharts.StateRef{{Label: "A"}},
			},
			event: "AB",
			wantNewConfig: &statecharts.Configuration{
				States: []*statecharts.StateRef{{Label: "B"}},
			},
			wantTransition: true,
		},
		{
			name: "B to C",
			initialConfig: &statecharts.Configuration{
				States: []*statecharts.StateRef{{Label: "B"}},
			},
			event: "BC",
			wantNewConfig: &statecharts.Configuration{
				States: []*statecharts.StateRef{{Label: "C"}},
			},
			wantTransition: true,
		},
		{
			name: "No transition",
			initialConfig: &statecharts.Configuration{
				States: []*statecharts.StateRef{{Label: "A"}},
			},
			event: "BC",
			wantNewConfig: &statecharts.Configuration{
				States: []*statecharts.StateRef{{Label: "A"}},
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

func isValidConfiguration(statechart *statecharts.Statechart, config *statecharts.Configuration) bool {
	// This is a simplified validity check. A real implementation would be more complex.
	stateMap := make(map[string]bool)
	for _, state := range config.States {
		stateMap[state.Label] = true
	}

	var dfs func(*statecharts.State) bool
	dfs = func(state *statecharts.State) bool {
		if stateMap[state.Label] {
			if state.Type == statecharts.StateType_STATE_TYPE_PARALLEL {
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

func transitionConfiguration(statechart *statecharts.Statechart, config *statecharts.Configuration, event string) (*statecharts.Configuration, bool) {
	for _, transition := range statechart.Transitions {
		if transition.Event == event && contains(transition.From, config.States[0].Label) {
			return &statecharts.Configuration{
				States: []*statecharts.StateRef{{Label: transition.To[0]}},
			}, true
		}
	}
	return config, false
}

func TestConfigurationConsistency(t *testing.T) {
	statechart := &statecharts.Statechart{
		RootState: &statecharts.State{
			Label: "Root",
			Children: []*statecharts.State{
				{
					Label: "A",
					Children: []*statecharts.State{
						{Label: "A1"},
						{Label: "A2"},
					},
				},
				{
					Label: "B",
					Type:  statecharts.StateType_STATE_TYPE_PARALLEL,
					Children: []*statecharts.State{
						{
							Label: "B1",
							Children: []*statecharts.State{
								{Label: "B1a"},
								{Label: "B1b"},
							},
						},
						{
							Label: "B2",
							Children: []*statecharts.State{
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
		config         *statecharts.Configuration
		wantConsistent bool
	}{
		{
			name: "Consistent configuration",
			config: &statecharts.Configuration{
				States: []*statecharts.StateRef{
					{Label: "A"},
					{Label: "A1"},
				},
			},
			wantConsistent: true,
		},
		{
			name: "Consistent parallel configuration",
			config: &statecharts.Configuration{
				States: []*statecharts.StateRef{
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
			config: &statecharts.Configuration{
				States: []*statecharts.StateRef{
					{Label: "A"},
					{Label: "A1"},
					{Label: "A2"},
				},
			},
			wantConsistent: false,
		},
		{
			name: "Inconsistent: incomplete parallel state",
			config: &statecharts.Configuration{
				States: []*statecharts.StateRef{
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
	statechart := &statecharts.Statechart{
		RootState: &statecharts.State{
			Label: "Root",
			Children: []*statecharts.State{
				{
					Label: "A",
					Children: []*statecharts.State{
						{Label: "A1", IsInitial: true},
						{Label: "A2"},
					},
				},
				{
					Label: "B",
					Type:  statecharts.StateType_STATE_TYPE_PARALLEL,
					Children: []*statecharts.State{
						{
							Label: "B1",
							Children: []*statecharts.State{
								{Label: "B1a", IsInitial: true},
								{Label: "B1b"},
							},
						},
						{
							Label: "B2",
							Children: []*statecharts.State{
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
		initialConfig *statecharts.Configuration
		wantConfig    *statecharts.Configuration
	}{
		{
			name: "Complete A",
			initialConfig: &statecharts.Configuration{
				States: []*statecharts.StateRef{{Label: "A"}},
			},
			wantConfig: &statecharts.Configuration{
				States: []*statecharts.StateRef{
					{Label: "A"},
					{Label: "A1"},
				},
			},
		},
		{
			name: "Complete B",
			initialConfig: &statecharts.Configuration{
				States: []*statecharts.StateRef{{Label: "B"}},
			},
			wantConfig: &statecharts.Configuration{
				States: []*statecharts.StateRef{
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

func isConsistentConfiguration(statechart *statecharts.Statechart, config *statecharts.Configuration) bool {
	// This is a simplified consistency check. A real implementation would be more complex.
	stateMap := make(map[string]bool)
	for _, state := range config.States {
		stateMap[state.Label] = true
	}

	var dfs func(*statecharts.State) bool
	dfs = func(state *statecharts.State) bool {
		if stateMap[state.Label] {
			if state.Type == statecharts.StateType_STATE_TYPE_PARALLEL {
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
func completeConfiguration(statechart *statecharts.Statechart, config *statecharts.Configuration) *statecharts.Configuration {
	completedStates := make([]*statecharts.StateRef, 0)
	stateMap := make(map[string]bool)
	for _, state := range config.States {
		stateMap[state.Label] = true
		completedStates = append(completedStates, state)
	}

	var complete func(*statecharts.State)
	complete = func(state *statecharts.State) {
		if stateMap[state.Label] {
			if state.Type == statecharts.StateType_STATE_TYPE_PARALLEL {
				for _, child := range state.Children {
					complete(child)
				}
			} else {
				for _, child := range state.Children {
					if child.IsInitial && !stateMap[child.Label] {
						stateMap[child.Label] = true
						completedStates = append(completedStates, &statecharts.StateRef{Label: child.Label})
						complete(child)
						break
					}
				}
			}
		}
	}

	complete(statechart.RootState)

	return &statecharts.Configuration{
		States: completedStates,
	}
}

func TestConfigurationLeastCommonAncestor(t *testing.T) {
	statechart := &statecharts.Statechart{
		RootState: &statecharts.State{
			Label: "Root",
			Children: []*statecharts.State{
				{
					Label: "A",
					Children: []*statecharts.State{
						{Label: "A1"},
						{Label: "A2"},
					},
				},
				{
					Label: "B",
					Children: []*statecharts.State{
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
	statechart := &statecharts.Statechart{
		RootState: &statecharts.State{
			Label: "Root",
			Children: []*statecharts.State{
				{
					Label: "A",
					Type:  statecharts.StateType_STATE_TYPE_PARALLEL,
					Children: []*statecharts.State{
						{Label: "A1"},
						{Label: "A2"},
					},
				},
				{
					Label: "B",
					Children: []*statecharts.State{
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

func leastCommonAncestor(statechart *statecharts.Statechart, states ...string) (string, error) {
	var findPath func(*statecharts.State, string) ([]string, bool)
	findPath = func(state *statecharts.State, target string) ([]string, bool) {
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

func areOrthogonal(statechart *statecharts.Statechart, state1, state2 string) (bool, error) {
	lca, err := leastCommonAncestor(statechart, state1, state2)
	if err != nil {
		return false, err
	}

	var findState func(*statecharts.State, string) *statecharts.State
	findState = func(state *statecharts.State, target string) *statecharts.State {
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
	return lcaState != nil && lcaState.Type == statecharts.StateType_STATE_TYPE_PARALLEL, nil
}
