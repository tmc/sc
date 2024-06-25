package semantics

import (
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/tmc/sc"
)

func TestNormalize(t *testing.T) {
	tests := []struct {
		name       string
		statechart *Statechart
		wantErr    bool
	}{
		{
			name:       "Normalize valid statechart",
			statechart: exampleStatechart1,
			wantErr:    false,
		},
		// Add more test cases here
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := tt.statechart.Normalize()
			if (err != nil) != tt.wantErr {
				t.Errorf("Normalize() error = %v, wantErr %v", err, tt.wantErr)
			}
			// Add assertions to check if the statechart was normalized correctly
		})
	}
}

func TestNormalizeStateTypes(t *testing.T) {
	tests := []struct {
		name       string
		statechart *Statechart
		wantErr    bool
	}{
		{
			name: "Normalize state types",
			statechart: NewStatechart(&sc.Statechart{
				RootState: &sc.State{
					Label: "Root",
					Children: []*sc.State{
						{Label: "A"},
						{
							Label: "B",
							Children: []*sc.State{
								{Label: "B1"},
								{Label: "B2"},
							},
						},
					},
				},
			}),
			wantErr: false,
		},
		// Add more test cases here
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := normalizeStateTypes(tt.statechart)
			if (err != nil) != tt.wantErr {
				t.Errorf("normalizeStateTypes() error = %v, wantErr %v", err, tt.wantErr)
			}

			// Check if state types were normalized correctly
			if tt.statechart.RootState.Type != sc.StateTypeNormal {
				t.Errorf("Root state type not normalized, got %v, want %v", tt.statechart.RootState.Type, sc.StateTypeNormal)
			}
			for _, child := range tt.statechart.RootState.Children {
				if child.Label == "A" && child.Type != sc.StateTypeBasic {
					t.Errorf("State A type not normalized, got %v, want %v", child.Type, sc.StateTypeBasic)
				}
				if child.Label == "B" && child.Type != sc.StateTypeNormal {
					t.Errorf("State B type not normalized, got %v, want %v", child.Type, sc.StateTypeNormal)
				}
			}
		})
	}
}

func TestVisitStates(t *testing.T) {
	statechart := NewStatechart(&sc.Statechart{
		RootState: &sc.State{
			Label: "Root",
			Children: []*sc.State{
				{Label: "A"},
				{
					Label: "B",
					Children: []*sc.State{
						{Label: "B1"},
						{Label: "B2"},
					},
				},
			},
		},
	})

	visited := make(map[string]bool)
	err := visitStates(statechart.RootState, func(state *sc.State) error {
		visited[state.Label] = true
		return nil
	})

	if err != nil {
		t.Errorf("visitStates() returned unexpected error: %v", err)
	}

	expectedVisited := []string{"Root", "A", "B", "B1", "B2"}
	for _, label := range expectedVisited {
		if !visited[label] {
			t.Errorf("State %s was not visited", label)
		}
	}

	if len(visited) != len(expectedVisited) {
		t.Errorf("Unexpected number of visited states, got %d, want %d", len(visited), len(expectedVisited))
	}
}

func TestDefaultCompletion(t *testing.T) {
	tests := []struct {
		name    string
		states  []StateLabel
		want    []StateLabel
		wantErr bool
	}{
		{"Default completion of On", []StateLabel{"On"}, []StateLabel{"On", "Turnstile Control", "Blocked", "Card Reader Control", "Ready"}, false},
		{"Default completion of Off", []StateLabel{"Off"}, []StateLabel{"Off"}, false},
		{"Default completion of inconsistent states", []StateLabel{"On", "Off"}, nil, true},
		{"Non-existent state", []StateLabel{"NonExistent"}, nil, true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := exampleStatechart1.DefaultCompletion(tt.states...)
			if (err != nil) != tt.wantErr {
				t.Errorf("DefaultCompletion() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if diff := cmp.Diff(tt.want, got); diff != "" {
				t.Errorf("DefaultCompletion() mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func TestStatechart_findState(t *testing.T) {
	tests := []struct {
		name    string
		label   StateLabel
		wantErr bool
	}{
		{"Find existing state", "Blocked", false},
		{"Find root state", "", false},
		{"Non-existent state", "NonExistent", true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := exampleStatechart1.findState(tt.label)
			if (err != nil) != tt.wantErr {
				t.Errorf("Statechart.findState() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

func TestStatechart_childrenPlus(t *testing.T) {
	tests := []struct {
		name    string
		state   *sc.State
		want    []StateLabel
		wantErr bool
	}{
		{
			name:  "Children plus of On",
			state: exampleStatechart1.RootState.Children[1], // Assuming On is the second child
			want: []StateLabel{
				"Turnstile Control",
				"Blocked",
				"Unblocked",
				"Card Reader Control",
				"Ready",
				"Card Entered",
				"Turnstile Unblocked",
			},
			wantErr: false,
		},
		{
			name:    "Children plus of leaf state",
			state:   &sc.State{Label: "Leaf"},
			want:    nil,
			wantErr: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := exampleStatechart1.childrenPlus(tt.state)
			if (err != nil) != tt.wantErr {
				t.Errorf("Statechart.childrenPlus() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if diff := cmp.Diff(tt.want, got); diff != "" {
				t.Errorf("Statechart.childrenPlus(): %s", diff)
			}
		})
	}
}

func TestStatechart_getParent(t *testing.T) {
	tests := []struct {
		name     string
		needle   *sc.State
		haystack *sc.State
		want     string
		wantErr  bool
	}{
		{
			name:     "Find parent of Blocked",
			needle:   &sc.State{Label: "Blocked"},
			haystack: exampleStatechart1.RootState,
			want:     "Turnstile Control",
			wantErr:  false,
		},
		{
			name:     "Find parent of non-existent state",
			needle:   &sc.State{Label: "NonExistent"},
			haystack: exampleStatechart1.RootState,
			want:     "",
			wantErr:  true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := exampleStatechart1.GetParent(StateLabel(tt.needle.Label))
			if (err != nil) != tt.wantErr {
				t.Errorf("Statechart.getParent() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if got != nil && got.Label != tt.want {
				t.Errorf("Statechart.getParent() = %v, want %v", got.Label, tt.want)
			}
		})
	}
}

func TestStatechart_defaultCompletion(t *testing.T) {
	tests := []struct {
		name    string
		states  []StateLabel
		want    []StateLabel
		wantErr bool
	}{
		{
			name:    "Default completion of On",
			states:  []StateLabel{"On"},
			want:    []StateLabel{"On", "Turnstile Control", "Blocked", "Card Reader Control", "Ready"},
			wantErr: false,
		},
		{
			name:    "Default completion of inconsistent states",
			states:  []StateLabel{"On", "Off"},
			want:    nil,
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := exampleStatechart1.defaultCompletion(tt.states...)
			if (err != nil) != tt.wantErr {
				t.Errorf("Statechart.defaultCompletion() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if diff := cmp.Diff(tt.want, got); diff != "" {
				t.Errorf("Statechart.defaultCompletion() mismatch (-want +got):\n%s", diff)
			}
		})
	}
}
