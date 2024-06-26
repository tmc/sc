package semantics

import (
	"testing"

	"github.com/google/go-cmp/cmp"
)

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
		{"valid but not toplevel", exampleStatechart1, StateLabel("Turnstile Control"), []StateLabel{"Blocked", "Unblocked"}, false},
		{"Off", exampleStatechart1, StateLabel("Off"), nil, false},
		{"On", exampleStatechart1, StateLabel("On"), CreateStateLabels("Turnstile Control", "Card Reader Control"), false},
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
			CreateStateLabels("Turnstile Control", "Blocked", "Unblocked"), false},
		{"Off", exampleStatechart1, StateLabel("Off"),
			CreateStateLabels("Off"), false},
		// {"On", exampleStatechart1, StateLabel("On"),
		// 	labels(
		// 		"On", "Turnstile Control",
		// 		"Blocked", "Unblocked",
		// 		"Card Reader Control",
		// 		"Ready",
		// 		"Card Entered", "Turnstile Unblocked"), false},
		{"On", exampleStatechart1, StateLabel("On"),
			CreateStateLabels("On", "Turnstile Control", "Blocked", "Unblocked", "Card Reader Control", "Ready", "Card Entered", "Turnstile Unblocked"), false},
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
			CreateStateLabels(
				"Turnstile Control",
				"Blocked",
				"Unblocked",
				"Card Reader Control",
				"Ready",
				"Card Entered",
				"Turnstile Unblocked"), false},
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
func TestStatechart_LeastCommonAncestor(t *testing.T) {
	tests := []struct {
		name    string
		chart   *Statechart
		states  []StateLabel
		want    StateLabel
		wantErr bool
	}{
		{
			name:  "invalid",
			chart: exampleStatechart1,
			states: []StateLabel{
				StateLabel("this state does not exist"),
			},
			want:    "",
			wantErr: true,
		},
		{
			name:  "one state",
			chart: exampleStatechart1,
			states: []StateLabel{
				StateLabel("Off"),
			},
			want:    StateLabel("Off"),
			wantErr: false,
		},
		{
			name:  "two unrelated states",
			chart: exampleStatechart1,
			states: []StateLabel{
				StateLabel("Off"),
				StateLabel("On"),
			},
			want:    RootState,
			wantErr: false,
		},
		{
			name:  "two related states",
			chart: exampleStatechart1,
			states: []StateLabel{
				StateLabel("On"),
				StateLabel("Ready"),
			},
			want:    StateLabel("On"),
			wantErr: false,
		},
		{
			name:  "multiple related states",
			chart: exampleStatechart1,
			states: []StateLabel{
				StateLabel("On"),
				StateLabel("Ready"),
				StateLabel("Card Entered"),
			},
			want:    StateLabel("On"),
			wantErr: false,
		},
		{
			name:  "multiple unrelated states",
			chart: exampleStatechart1,
			states: []StateLabel{
				StateLabel("Off"),
				StateLabel("On"),
				StateLabel("Ready"),
			},
			want:    RootState,
			wantErr: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			c := tt.chart
			got, err := c.LeastCommonAncestor(tt.states...)
			if (err != nil) != tt.wantErr {
				t.Errorf("Statechart.LeastCommonAncestor() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if got != tt.want {
				t.Errorf("Statechart.LeastCommonAncestor() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestDefault(t *testing.T) {
	tests := []struct {
		name    string
		chart   *Statechart
		state   StateLabel
		want    StateLabel
		wantErr bool
	}{
		{"invalid", exampleStatechart1, StateLabel("this state does not exist"), "", true},
		{"default", exampleStatechart1, RootState, StateLabel("Off"), false},
		{"no default", exampleStatechart1, StateLabel("Off"), "", true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			c := tt.chart
			got, err := c.Default(tt.state)
			if (err != nil) != tt.wantErr {
				t.Errorf("Statechart.Default() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if got != tt.want {
				t.Errorf("Statechart.Default() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestOrthogonal(t *testing.T) {
	tests := []struct {
		name    string
		chart   *Statechart
		state1  StateLabel
		state2  StateLabel
		want    bool
		wantErr bool
	}{
		{"invalid", exampleStatechart1, StateLabel("this state does not exist"), StateLabel("On"), false, true},
		{"not orthogonal", exampleStatechart1, StateLabel("On"), StateLabel("Off"), false, false},
		{"orthogonal", exampleStatechart1, StateLabel("Blocked"), StateLabel("Card Reader Control"), true, false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			c := tt.chart
			got, err := c.Orthogonal(tt.state1, tt.state2)
			if (err != nil) != tt.wantErr {
				t.Errorf("Statechart.Orthogonal() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if !cmp.Equal(tt.want, got) {
				t.Errorf("(-want +got):\n%s", cmp.Diff(tt.want, got))
			}
		})
	}
}

func TestConsistent(t *testing.T) {
	tests := []struct {
		name    string
		chart   *Statechart
		states  []StateLabel
		want    bool
		wantErr bool
	}{
		{"invalid", exampleStatechart1, []StateLabel{StateLabel("this state does not exist")}, false, true},
		{"consistent", exampleStatechart1, []StateLabel{StateLabel("On"), StateLabel("Ready")}, true, false},
		{"inconsistent", exampleStatechart1, []StateLabel{StateLabel("On"), StateLabel("Off")}, false, false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			c := tt.chart
			got, err := c.Consistent(tt.states...)
			if (err != nil) != tt.wantErr {
				t.Errorf("Statechart.Consistent() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if !cmp.Equal(tt.want, got) {
				t.Errorf("(-want +got):\n%s", cmp.Diff(tt.want, got))
			}
		})
	}
}

// TODO: meld
// func TestDefaultCompletion(t *testing.T) {
// 	tests := []struct {
// 		name    string
// 		chart   *Statechart
// 		state   []StateLabel
// 		want    []StateLabel
// 		wantErr bool
// 	}{
// 		{"invalid", exampleStatechart1, []StateLabel{StateLabel("this state does not exist")}, nil, true},
// 		{"off", exampleStatechart1, []StateLabel{StateLabel("Off")}, labels("Off", ""), false},
// 		{"unblocked", exampleStatechart1, []StateLabel{StateLabel("Unblocked")}, labels(
// 			"Unblocked",
// 			"Turnstile Control",
// 			"On",
// 			"Card Reader Control",
// 			"Ready",
// 			"",
// 		), false},
// 	}
// 	for _, tt := range tests {
// 		t.Run(tt.name, func(t *testing.T) {
// 			c := tt.chart
// 			if err := c.Normalize(); err != nil {
// 				t.Errorf("Statechart.Normalize() error = %v", err)
// 				return
// 			}
// 			got, err := c.DefaultCompletion(tt.state...)
// 			if (err != nil) != tt.wantErr {
// 				t.Errorf("Statechart.DefaultCompletion() error = %v, wantErr %v", err, tt.wantErr)
// 				return
// 			}
// 			if !cmp.Equal(tt.want, got) {
// 				t.Errorf("(-want +got):\n%s", cmp.Diff(tt.want, got))
// 			}
// 		})
// 	}
// }
