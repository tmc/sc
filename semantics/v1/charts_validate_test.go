package semantics

import (
	"testing"

	"github.com/tmc/sc"
)

func TestValidate(t *testing.T) {
	tests := []struct {
		name       string
		statechart *Statechart
		wantErr    bool
		errMsg     string
	}{
		{
			name:       "Valid statechart",
			statechart: exampleStatechart1,
			wantErr:    false,
		},
		{
			name: "Invalid statechart - duplicate state labels",
			statechart: NewStatechart(&sc.Statechart{
				RootState: &sc.State{
					Children: []*sc.State{
						{Label: "A"},
						{Label: "A"}, // Duplicate label
					},
				},
			}),
			wantErr: true,
			errMsg:  "duplicate state label: A",
		},
		{
			name: "Invalid statechart - missing initial state",
			statechart: NewStatechart(&sc.Statechart{
				RootState: &sc.State{
					Type: sc.StateTypeNormal,
					Children: []*sc.State{
						{Label: "A"},
						{Label: "B"},
					},
				},
			}),
			wantErr: true,
			errMsg:  "normal state __root__ must have exactly one initial child, found 0",
		},
		{
			name: "Invalid statechart - basic state with children",
			statechart: NewStatechart(&sc.Statechart{
				RootState: &sc.State{
					Children: []*sc.State{
						{
							Label: "A",
							Type:  sc.StateTypeBasic,
							Children: []*sc.State{
								{Label: "A1"},
							},
						},
					},
				},
			}),
			wantErr: true,
			errMsg:  "basic state A cannot have children",
		},
		{
			name: "Invalid statechart - compound state without children",
			statechart: NewStatechart(&sc.Statechart{
				RootState: &sc.State{
					Children: []*sc.State{
						{
							Label: "A",
							Type:  sc.StateTypeNormal,
						},
					},
				},
			}),
			wantErr: true,
			errMsg:  "compound state A must have children",
		},
		{
			name: "Invalid statechart - inconsistent parent-child relationship",
			statechart: NewStatechart(&sc.Statechart{
				RootState: &sc.State{
					Children: []*sc.State{
						{
							Label: "A",
							Children: []*sc.State{
								{Label: "B"},
							},
						},
						{Label: "B"}, // B appears twice in different places
					},
				},
			}),
			wantErr: true,
			errMsg:  "duplicate state label: B",
		},
		{
			name: "Invalid statechart - multiple default states",
			statechart: NewStatechart(&sc.Statechart{
				RootState: &sc.State{
					Type: sc.StateTypeNormal,
					Children: []*sc.State{
						{Label: "A", IsInitial: true},
						{Label: "B", IsInitial: true},
					},
				},
			}),
			wantErr: true,
			errMsg:  "normal state __root__ must have exactly one initial child, found 2",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := tt.statechart.Validate()
			if (err != nil) != tt.wantErr {
				t.Errorf("Validate() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if tt.wantErr && err.Error() != tt.errMsg {
				t.Errorf("Validate() error message = %v, want %v", err.Error(), tt.errMsg)
			}
		})
	}
}

func TestValidateNonOverlappingStateLabels(t *testing.T) {
	tests := []struct {
		name       string
		statechart *Statechart
		wantErr    bool
	}{
		{
			name:       "Valid non-overlapping labels",
			statechart: exampleStatechart1,
			wantErr:    false,
		},
		{
			name: "Overlapping labels",
			statechart: NewStatechart(&sc.Statechart{
				RootState: &sc.State{
					Children: []*sc.State{
						{Label: "A"},
						{Label: "B", Children: []*sc.State{{Label: "A"}}},
					},
				},
			}),
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := tt.statechart.validateNonOverlappingStateLabels()
			if (err != nil) != tt.wantErr {
				t.Errorf("validateNonOverlappingStateLabels() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

func TestValidateRootState(t *testing.T) {
	tests := []struct {
		name       string
		statechart *Statechart
		wantErr    bool
	}{
		{
			name:       "Valid root state",
			statechart: exampleStatechart1,
			wantErr:    false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := tt.statechart.validateRootState()
			if (err != nil) != tt.wantErr {
				t.Errorf("validateRootState() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

func TestValidateStateTypeAgreesWithChildren(t *testing.T) {
	tests := []struct {
		name       string
		statechart *Statechart
		wantErr    bool
	}{
		{
			name:       "Valid state types",
			statechart: exampleStatechart1,
			wantErr:    false,
		},
		{
			name: "Basic state with children",
			statechart: NewStatechart(&sc.Statechart{
				RootState: &sc.State{
					Children: []*sc.State{
						{Label: "A", Type: sc.StateTypeBasic, Children: []*sc.State{{Label: "A1"}}},
					},
				},
			}),
			wantErr: true,
		},
		{
			name: "Compound state without children",
			statechart: NewStatechart(&sc.Statechart{
				RootState: &sc.State{
					Children: []*sc.State{
						{Label: "A", Type: sc.StateTypeNormal},
					},
				},
			}),
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := tt.statechart.validateStateTypeAgreesWithChildren()
			if (err != nil) != tt.wantErr {
				t.Errorf("validateStateTypeAgreesWithChildren() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

func TestValidateParentChildRelationships(t *testing.T) {
	tests := []struct {
		name       string
		statechart *Statechart
		wantErr    bool
	}{
		{
			name:       "Valid parent-child relationships",
			statechart: exampleStatechart1,
			wantErr:    false,
		},
		{
			name: "Inconsistent parent-child relationship",
			statechart: NewStatechart(&sc.Statechart{
				RootState: &sc.State{
					Children: []*sc.State{
						{Label: "A", Children: []*sc.State{{Label: "B"}}},
						{Label: "B"},
					},
				},
			}),
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := tt.statechart.validateParentChildRelationships()
			if (err != nil) != tt.wantErr {
				t.Errorf("validateParentChildRelationships() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

func TestValidateParentStatesHaveSingleDefaults(t *testing.T) {
	tests := []struct {
		name       string
		statechart *Statechart
		wantErr    bool
	}{
		{
			name:       "Valid default states",
			statechart: exampleStatechart1,
			wantErr:    false,
		},
		{
			name: "Multiple default states",
			statechart: NewStatechart(&sc.Statechart{
				RootState: &sc.State{
					Type: sc.StateTypeNormal,
					Children: []*sc.State{
						{Label: "A", IsInitial: true},
						{Label: "B", IsInitial: true},
					},
				},
			}),
			wantErr: true,
		},
		{
			name: "No default state",
			statechart: NewStatechart(&sc.Statechart{
				RootState: &sc.State{
					Type: sc.StateTypeNormal,
					Children: []*sc.State{
						{Label: "A", IsInitial: false},
						{Label: "B", IsInitial: false},
					},
				},
			}),
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := tt.statechart.validateParentStatesHaveSingleDefaults()
			if (err != nil) != tt.wantErr {
				t.Errorf("validateParentStatesHaveSingleDefaults() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}
