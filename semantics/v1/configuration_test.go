package semantics

import (
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/tmc/sc"
	"google.golang.org/protobuf/testing/protocmp"
)

func TestValidateConfiguration(t *testing.T) {
	statechart := NewStatechart(&sc.Statechart{
		RootState: &sc.State{
			Label: "__root__",
			Type:  sc.StateTypeNormal,
			Children: []*sc.State{
				{
					Label: "A",
					Type:  sc.StateTypeNormal,
					Children: []*sc.State{
						{Label: "A1", Type: sc.StateTypeBasic},
						{Label: "A2", Type: sc.StateTypeBasic},
					},
				},
				{
					Label: "B",
					Type:  sc.StateTypeParallel,
					Children: []*sc.State{
						{
							Label: "B1",
							Type:  sc.StateTypeNormal,
							Children: []*sc.State{
								{Label: "B1a", Type: sc.StateTypeBasic},
								{Label: "B1b", Type: sc.StateTypeBasic},
							},
						},
						{
							Label: "B2",
							Type:  sc.StateTypeNormal,
							Children: []*sc.State{
								{Label: "B2a", Type: sc.StateTypeBasic},
								{Label: "B2b", Type: sc.StateTypeBasic},
							},
						},
					},
				},
			},
		},
	})

	tests := []struct {
		name    string
		config  *sc.Configuration
		wantErr bool
	}{
		{
			name: "Valid configuration - OR-state",
			config: &sc.Configuration{
				States: []*sc.StateRef{
					{Label: "__root__"},
					{Label: "A"},
					{Label: "A1"},
				},
			},
			wantErr: false,
		},
		{
			name: "Valid configuration - AND-state",
			config: &sc.Configuration{
				States: []*sc.StateRef{
					{Label: "__root__"},
					{Label: "B"},
					{Label: "B1"},
					{Label: "B1a"},
					{Label: "B2"},
					{Label: "B2a"},
				},
			},
			wantErr: false,
		},
		{
			name: "Invalid - multiple children of OR-state",
			config: &sc.Configuration{
				States: []*sc.StateRef{
					{Label: "__root__"},
					{Label: "A"},
					{Label: "A1"},
					{Label: "A2"},
				},
			},
			wantErr: true,
		},
		{
			name: "Invalid - incomplete AND-state",
			config: &sc.Configuration{
				States: []*sc.StateRef{
					{Label: "__root__"},
					{Label: "B"},
					{Label: "B1"},
					{Label: "B1a"},
				},
			},
			wantErr: true,
		},
		{
			name: "Invalid - missing parent",
			config: &sc.Configuration{
				States: []*sc.StateRef{
					{Label: "A1"},
				},
			},
			wantErr: true,
		},
		{
			name: "Invalid - nonexistent state",
			config: &sc.Configuration{
				States: []*sc.StateRef{
					{Label: "__root__"},
					{Label: "NonexistentState"},
				},
			},
			wantErr: true,
		},
		{
			name: "Invalid - incomplete parallel state (missing child of substate)",
			config: &sc.Configuration{
				States: []*sc.StateRef{
					{Label: "__root__"},
					{Label: "B"},
					{Label: "B1"},
					{Label: "B2"},
					// Missing B1a and B2a
				},
			},
			wantErr: true,
		},
		{
			name: "Invalid - incomplete parallel state (partial substate)",
			config: &sc.Configuration{
				States: []*sc.StateRef{
					{Label: "__root__"},
					{Label: "B"},
					{Label: "B1"},
					{Label: "B1a"},
					{Label: "B2"},
					// Missing B2a
				},
			},
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := ValidateConfiguration(statechart, tt.config)
			if (err != nil) != tt.wantErr {
				t.Errorf("ValidateConfiguration() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
		})
	}
}

func TestDefaultCompletionToplevel(t *testing.T) {
	statechart := NewStatechart(&sc.Statechart{
		RootState: &sc.State{
			Label: "__root__",
			Type:  sc.StateTypeNormal,
			Children: []*sc.State{
				{
					Label: "A",
					Type:  sc.StateTypeNormal,
					Children: []*sc.State{
						{Label: "A1", Type: sc.StateTypeBasic, IsInitial: true},
						{Label: "A2", Type: sc.StateTypeBasic},
					},
				},
				{
					Label: "B",
					Type:  sc.StateTypeParallel,
					Children: []*sc.State{
						{
							Label: "B1",
							Type:  sc.StateTypeNormal,
							Children: []*sc.State{
								{Label: "B1a", Type: sc.StateTypeBasic, IsInitial: true},
								{Label: "B1b", Type: sc.StateTypeBasic},
							},
						},
						{
							Label: "B2",
							Type:  sc.StateTypeNormal,
							Children: []*sc.State{
								{Label: "B2a", Type: sc.StateTypeBasic, IsInitial: true},
								{Label: "B2b", Type: sc.StateTypeBasic},
							},
						},
					},
				},
				{
					Label: "C",
					Type:  sc.StateTypeNormal,
					Children: []*sc.State{
						{Label: "C1", Type: sc.StateTypeBasic, IsInitial: true},
						{
							Label: "C2",
							Type:  sc.StateTypeNormal,
							Children: []*sc.State{
								{Label: "C2a", Type: sc.StateTypeBasic, IsInitial: true},
								{Label: "C2b", Type: sc.StateTypeBasic},
							},
						},
					},
				},
			},
		},
	})

	tests := []struct {
		name     string
		config   *sc.Configuration
		expected *sc.Configuration
	}{
		{
			name: "Complete OR-state",
			config: &sc.Configuration{
				States: []*sc.StateRef{{Label: "A"}},
			},
			expected: &sc.Configuration{
				States: []*sc.StateRef{
					{Label: "__root__"},
					{Label: "A"},
					{Label: "A1"},
				},
			},
		},
		{
			name: "Complete AND-state",
			config: &sc.Configuration{
				States: []*sc.StateRef{{Label: "B"}},
			},
			expected: &sc.Configuration{
				States: []*sc.StateRef{
					{Label: "__root__"},
					{Label: "B"},
					{Label: "B1"},
					{Label: "B1a"},
					{Label: "B2"},
					{Label: "B2a"},
				},
			},
		},
		{
			name: "Already complete configuration",
			config: &sc.Configuration{
				States: []*sc.StateRef{
					{Label: "__root__"},
					{Label: "A"},
					{Label: "A1"},
				},
			},
			expected: &sc.Configuration{
				States: []*sc.StateRef{
					{Label: "__root__"},
					{Label: "A"},
					{Label: "A1"},
				},
			},
		},
		{
			name: "Nested OR-state",
			config: &sc.Configuration{
				States: []*sc.StateRef{{Label: "C"}},
			},
			expected: &sc.Configuration{
				States: []*sc.StateRef{
					{Label: "__root__"},
					{Label: "C"},
					{Label: "C1"},
				},
			},
		},
		{
			name: "Multiple states",
			config: &sc.Configuration{
				States: []*sc.StateRef{
					{Label: "A"},
					{Label: "B"},
				},
			},
			expected: &sc.Configuration{
				States: []*sc.StateRef{
					{Label: "__root__"},
					{Label: "A"},
					{Label: "A1"},
					{Label: "B"},
					{Label: "B1"},
					{Label: "B1a"},
					{Label: "B2"},
					{Label: "B2a"},
				},
			},
		},
		{
			name: "Deep nested state",
			config: &sc.Configuration{
				States: []*sc.StateRef{{Label: "C2"}},
			},
			expected: &sc.Configuration{
				States: []*sc.StateRef{
					{Label: "__root__"},
					{Label: "C"},
					{Label: "C2"},
					{Label: "C2a"},
				},
			},
		},
		{
			name: "Empty configuration",
			config: &sc.Configuration{
				States: []*sc.StateRef{},
			},
			expected: &sc.Configuration{
				States: []*sc.StateRef{},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result, err := DefaultCompletion(statechart, tt.config)
			if err != nil {
				t.Fatalf("DefaultCompletion() error = %v", err)
			}
			if diff := cmp.Diff(tt.expected, result, protocmp.Transform()); diff != "" {
				t.Errorf("DefaultCompletion() mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func TestIsConsistentConfiguration(t *testing.T) {
	statechart := NewStatechart(&sc.Statechart{
		RootState: &sc.State{
			Label: "__root__",
			Type:  sc.StateTypeNormal,
			Children: []*sc.State{
				{
					Label: "A",
					Type:  sc.StateTypeNormal,
					Children: []*sc.State{
						{Label: "A1", Type: sc.StateTypeBasic, IsInitial: true},
						{Label: "A2", Type: sc.StateTypeBasic},
					},
				},
				{
					Label: "B",
					Type:  sc.StateTypeParallel,
					Children: []*sc.State{
						{
							Label: "B1",
							Type:  sc.StateTypeNormal,
							Children: []*sc.State{
								{Label: "B1a", Type: sc.StateTypeBasic, IsInitial: true},
								{Label: "B1b", Type: sc.StateTypeBasic},
							},
						},
						{
							Label: "B2",
							Type:  sc.StateTypeNormal,
							Children: []*sc.State{
								{Label: "B2a", Type: sc.StateTypeBasic, IsInitial: true},
								{Label: "B2b", Type: sc.StateTypeBasic},
							},
						},
					},
				},
			},
		},
	})

	tests := []struct {
		name    string
		config  *sc.Configuration
		want    bool
		wantErr bool
	}{
		{
			name: "Consistent configuration - OR-state",
			config: &sc.Configuration{
				States: []*sc.StateRef{
					{Label: "__root__"},
					{Label: "A"},
					{Label: "A1"},
				},
			},
			want:    true,
			wantErr: false,
		},
		{
			name: "Consistent configuration - AND-state",
			config: &sc.Configuration{
				States: []*sc.StateRef{
					{Label: "__root__"},
					{Label: "B"},
					{Label: "B1"},
					{Label: "B1a"},
					{Label: "B2"},
					{Label: "B2a"},
				},
			},
			want:    true,
			wantErr: false,
		},
		{
			name: "Inconsistent - incomplete default completion",
			config: &sc.Configuration{
				States: []*sc.StateRef{
					{Label: "__root__"},
					{Label: "A"},
				},
			},
			want:    false,
			wantErr: true,
		},
		{
			name: "Inconsistent - multiple children of OR-state",
			config: &sc.Configuration{
				States: []*sc.StateRef{
					{Label: "__root__"},
					{Label: "A"},
					{Label: "A1"},
					{Label: "A2"},
				},
			},
			want:    false,
			wantErr: true,
		},
		{
			name: "Inconsistent - incomplete AND-state",
			config: &sc.Configuration{
				States: []*sc.StateRef{
					{Label: "__root__"},
					{Label: "B"},
					{Label: "B1"},
					{Label: "B1a"},
				},
			},
			want:    false,
			wantErr: true,
		},
		{
			name: "Inconsistent - missing parent",
			config: &sc.Configuration{
				States: []*sc.StateRef{
					{Label: "A1"},
				},
			},
			want:    false,
			wantErr: true,
		},
		{
			name: "Inconsistent - nonexistent state",
			config: &sc.Configuration{
				States: []*sc.StateRef{
					{Label: "__root__"},
					{Label: "NonexistentState"},
				},
			},
			want:    false,
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := IsConsistentConfiguration(statechart, tt.config)
			if (err != nil) != tt.wantErr {
				t.Errorf("IsConsistentConfiguration() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if got != tt.want {
				t.Errorf("IsConsistentConfiguration() = %v, want %v", got, tt.want)
			}
		})
	}
}
