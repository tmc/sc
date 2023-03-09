package semantics

import (
	"testing"

	"github.com/google/go-cmp/cmp"
)

func labels(l ...string) []StateLabel {
	var labels []StateLabel
	for _, s := range l {
		labels = append(labels, StateLabel(s))
	}
	return labels
}

type stateSemanticsTestCase struct {
	name    string
	chart   *Statechart
	state   StateLabel
	want    []StateLabel
	wantErr bool
}

func TestStatechart_Children(t *testing.T) {
	tests := []stateSemanticsTestCase{
		{"invalid", exampleStatechart1, StateLabel("this state does not exist"), nil, true},
		{"valid but not toplevel", exampleStatechart1, StateLabel("Turnstile Control"), nil, false},
		{"Off", exampleStatechart1, StateLabel("Off"), nil, false},
		{"On", exampleStatechart1, StateLabel("On"), labels("Turnstile Control", "Card Reader Control"), false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			c := tt.chart
			got, err := c.Children(tt.state)
			if (err != nil) != tt.wantErr {
				t.Errorf("Statechart.Children() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if !cmp.Equal(tt.want, got) {
				t.Errorf("(-want +got):\n%s", cmp.Diff(tt.want, got))
			}
		})
	}
}

func TestStatechart_ChildrenStar(t *testing.T) {
	tests := []stateSemanticsTestCase{
		{"invalid", exampleStatechart1, StateLabel("this state does not exist"), nil, true},
		{"valid but not toplevel", exampleStatechart1, StateLabel("Turnstile Control"),
			labels("Turnstile Control"), false},
		{"Off", exampleStatechart1, StateLabel("Off"),
			labels("Off"), false},
		{"On", exampleStatechart1, StateLabel("On"),
			labels("On", "Turnstile Control", "Card Reader Control", "Ready"), false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			c := tt.chart
			got, err := c.ChildrenStar(tt.state)
			if (err != nil) != tt.wantErr {
				t.Errorf("Statechart.Children() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if !cmp.Equal(tt.want, got) {
				t.Errorf("(-want +got):\n%s", cmp.Diff(tt.want, got))
			}
		})
	}
}

func TestStatechart_ChildrenPlus(t *testing.T) {
	tests := []stateSemanticsTestCase{
		{"On", exampleStatechart1, StateLabel("On"),
			labels("Turnstile Control", "Card Reader Control", "Ready"), false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			c := tt.chart
			got, err := c.ChildrenPlus(tt.state)
			if (err != nil) != tt.wantErr {
				t.Errorf("Statechart.Children() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if !cmp.Equal(tt.want, got) {
				t.Errorf("(-want +got):\n%s", cmp.Diff(tt.want, got))
			}
		})
	}
}

func TestStatechart_AncestorallyRelated(t *testing.T) {
	tests := []struct {
		name           string
		chart          *Statechart
		state1, state2 StateLabel
		want           bool
		wantErr        bool
	}{
		{"invalid", exampleStatechart1, StateLabel("this state does not exist"), StateLabel("On"), false, true},
		{"not related", exampleStatechart1, StateLabel("On"), StateLabel("Off"), false, false},
		{"related (self)", exampleStatechart1, StateLabel("On"), StateLabel("On"), true, false},
		{"related", exampleStatechart1, StateLabel("On"), StateLabel("Ready"), true, false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			c := tt.chart
			got, err := c.AncestrallyRelated(tt.state1, tt.state2)
			if (err != nil) != tt.wantErr {
				t.Errorf("Statechart.AncestorallyRelated() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if got != tt.want {
				t.Errorf("Statechart.AncestorallyRelated() = %v, want %v", got, tt.want)
			}
		})
	}
}
