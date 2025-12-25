package main

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/tmc/sc"
)

func TestParseChart(t *testing.T) {
	tests := []struct {
		name    string
		input   string
		wantErr bool
	}{
		{
			name: "valid statechart",
			input: `{
				"root_state": {
					"label": "__root__",
					"type": 2,
					"children": [
						{"label": "A", "type": 1, "is_initial": true},
						{"label": "B", "type": 1}
					]
				}
			}`,
			wantErr: false,
		},
		{
			name: "valid machine",
			input: `{
				"id": "test-machine",
				"statechart": {
					"root_state": {
						"label": "__root__",
						"type": 2,
						"children": [
							{"label": "A", "type": 1, "is_initial": true}
						]
					}
				}
			}`,
			wantErr: false,
		},
		{
			name:    "invalid json",
			input:   `{not valid}`,
			wantErr: true,
		},
		{
			name:    "empty object",
			input:   `{}`,
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, chart, err := parseChart(tt.input)
			if (err != nil) != tt.wantErr {
				t.Errorf("parseChart() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if !tt.wantErr && chart == nil {
				t.Error("parseChart() returned nil chart without error")
			}
		})
	}
}

func TestCountStates(t *testing.T) {
	chart := &sc.Statechart{
		RootState: &sc.State{
			Label: "__root__",
			Type:  sc.StateTypeOR,
			Children: []*sc.State{
				{Label: "A", Type: sc.StateTypeBasic, IsInitial: true},
				{Label: "B", Type: sc.StateTypeBasic},
				{
					Label: "C",
					Type:  sc.StateTypeAND,
					Children: []*sc.State{
						{Label: "C1", Type: sc.StateTypeBasic, IsInitial: true},
						{Label: "C2", Type: sc.StateTypeBasic},
					},
				},
			},
		},
	}

	total, leaf, parallel := countStates(chart.RootState)

	if total != 6 {
		t.Errorf("countStates() total = %d, want 6", total)
	}
	if leaf != 4 {
		t.Errorf("countStates() leaf = %d, want 4", leaf)
	}
	if parallel != 1 {
		t.Errorf("countStates() parallel = %d, want 1", parallel)
	}
}

func TestCollectStates(t *testing.T) {
	chart := &sc.Statechart{
		RootState: &sc.State{
			Label: "__root__",
			Type:  sc.StateTypeOR,
			Children: []*sc.State{
				{Label: "A", Type: sc.StateTypeBasic, IsInitial: true},
				{Label: "B", Type: sc.StateTypeBasic},
				{
					Label: "C",
					Type:  sc.StateTypeOR,
					Children: []*sc.State{
						{Label: "C1", Type: sc.StateTypeBasic, IsInitial: true},
						{Label: "C2", Type: sc.StateTypeBasic},
					},
				},
			},
		},
	}

	tests := []struct {
		name     string
		leafOnly bool
		showType bool
		want     []string
	}{
		{
			name:     "all states",
			leafOnly: false,
			showType: false,
			want:     []string{"__root__", "A", "B", "C", "C1", "C2"},
		},
		{
			name:     "leaf only",
			leafOnly: true,
			showType: false,
			want:     []string{"A", "B", "C1", "C2"},
		},
		{
			name:     "with types",
			leafOnly: false,
			showType: true,
			want:     []string{"__root__\tor", "A\tbasic"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			states := collectStates(chart.RootState, tt.leafOnly, tt.showType)
			result := strings.Join(states, "\n")

			for _, want := range tt.want {
				if !strings.Contains(result, want) {
					t.Errorf("collectStates() missing %q in result %q", want, result)
				}
			}
		})
	}
}

func TestCollectEvents(t *testing.T) {
	chart := &sc.Statechart{
		Events: []*sc.Event{
			{Label: "GO"},
			{Label: "STOP"},
		},
		Transitions: []*sc.Transition{
			{Event: "GO"},
			{Event: "RESET"}, // Not in events list
		},
	}

	events := collectEvents(chart)

	want := []string{"GO", "STOP", "RESET"}
	for _, w := range want {
		found := false
		for _, e := range events {
			if e == w {
				found = true
				break
			}
		}
		if !found {
			t.Errorf("collectEvents() missing event %q", w)
		}
	}
}

func TestGenerateMermaid(t *testing.T) {
	chart := &sc.Statechart{
		RootState: &sc.State{
			Label: "__root__",
			Type:  sc.StateTypeOR,
			Children: []*sc.State{
				{Label: "Red", Type: sc.StateTypeBasic, IsInitial: true},
				{Label: "Green", Type: sc.StateTypeBasic},
			},
		},
		Transitions: []*sc.Transition{
			{From: []string{"Red"}, To: []string{"Green"}, Event: "NEXT"},
		},
	}

	mermaid := generateMermaid(chart, "TB")

	if !strings.Contains(mermaid, "stateDiagram-v2") {
		t.Error("generateMermaid() missing stateDiagram-v2")
	}
	if !strings.Contains(mermaid, "direction TB") {
		t.Error("generateMermaid() missing direction")
	}
	if !strings.Contains(mermaid, "Red --> Green") {
		t.Error("generateMermaid() missing transition")
	}
}

func TestFormatTransitions(t *testing.T) {
	chart := &sc.Statechart{
		Transitions: []*sc.Transition{
			{
				Label: "t1",
				From:  []string{"A"},
				To:    []string{"B"},
				Event: "GO",
				Guard: &sc.Guard{Expression: "x > 0"},
			},
		},
	}

	result := formatTransitions(chart)

	var transitions []map[string]any
	if err := json.Unmarshal([]byte(result), &transitions); err != nil {
		t.Fatalf("formatTransitions() returned invalid JSON: %v", err)
	}

	if len(transitions) != 1 {
		t.Errorf("formatTransitions() returned %d transitions, want 1", len(transitions))
	}

	if transitions[0]["event"] != "GO" {
		t.Errorf("formatTransitions() event = %v, want GO", transitions[0]["event"])
	}
	if transitions[0]["guard"] != "x > 0" {
		t.Errorf("formatTransitions() guard = %v, want 'x > 0'", transitions[0]["guard"])
	}
}

func TestStateTypeString(t *testing.T) {
	tests := []struct {
		input sc.StateType
		want  string
	}{
		{sc.StateTypeBasic, "basic"},
		{sc.StateTypeOR, "or"},
		{sc.StateTypeAND, "parallel"},
		{sc.StateType(99), "unknown"},
	}

	for _, tt := range tests {
		got := stateTypeString(tt.input)
		if got != tt.want {
			t.Errorf("stateTypeString(%v) = %q, want %q", tt.input, got, tt.want)
		}
	}
}

func TestSanitizeID(t *testing.T) {
	tests := []struct {
		input string
		want  string
	}{
		{"simple", "simple"},
		{"with space", "with_space"},
		{"with-dash", "with_dash"},
		{"mixed-case ID", "mixed_case_ID"},
	}

	for _, tt := range tests {
		got := sanitizeID(tt.input)
		if got != tt.want {
			t.Errorf("sanitizeID(%q) = %q, want %q", tt.input, got, tt.want)
		}
	}
}

func TestGenerateDot(t *testing.T) {
	chart := &sc.Statechart{
		RootState: &sc.State{
			Label: "__root__",
			Type:  sc.StateTypeOR,
			Children: []*sc.State{
				{Label: "Red", Type: sc.StateTypeBasic, IsInitial: true},
				{Label: "Green", Type: sc.StateTypeBasic},
			},
		},
		Transitions: []*sc.Transition{
			{From: []string{"Red"}, To: []string{"Green"}, Event: "NEXT"},
		},
	}

	dot := generateDot(chart, "TB")

	if !strings.Contains(dot, "digraph statechart") {
		t.Error("generateDot() missing digraph declaration")
	}
	if !strings.Contains(dot, "rankdir=TB") {
		t.Error("generateDot() missing rankdir")
	}
	if !strings.Contains(dot, "Red -> Green") {
		t.Error("generateDot() missing transition edge")
	}
	if !strings.Contains(dot, `label="NEXT"`) {
		t.Error("generateDot() missing transition label")
	}
}
