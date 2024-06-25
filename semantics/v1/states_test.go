package semantics

import (
	"testing"

	"github.com/google/go-cmp/cmp"
)

func TestChildren(t *testing.T) {
	tests := []struct {
		name    string
		state   StateLabel
		want    []StateLabel
		wantErr bool
	}{
		{"Root children", "", []StateLabel{"Off", "On"}, false},
		{"On children", "On", []StateLabel{"Turnstile Control", "Card Reader Control"}, false},
		{"Off children", "Off", nil, false},
		{"Non-existent state", "NonExistent", nil, true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := exampleStatechart1.Children(tt.state)
			if (err != nil) != tt.wantErr {
				t.Errorf("Children() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if diff := cmp.Diff(got, tt.want); diff != "" {
				t.Errorf("Children() mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func TestChildrenPlus(t *testing.T) {
	tests := []struct {
		name    string
		state   StateLabel
		want    []StateLabel
		wantErr bool
	}{
		{"On children plus", "On", []StateLabel{
			"Turnstile Control",
			"Blocked", "Unblocked",
			"Card Reader Control",
			"Ready",
			"Card Entered", "Turnstile Unblocked"}, false},
		{"Off children plus", "Off", nil, false},
		{"Non-existent state", "NonExistent", nil, true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := exampleStatechart1.ChildrenPlus(tt.state)
			if (err != nil) != tt.wantErr {
				t.Errorf("ChildrenPlus() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if diff := cmp.Diff(got, tt.want); diff != "" {
				t.Errorf("ChildrenPlus() mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func TestChildrenStar(t *testing.T) {
	tests := []struct {
		name    string
		state   StateLabel
		want    []StateLabel
		wantErr bool
	}{
		{"On children star", "On", []StateLabel{"On", "Turnstile Control",
			"Blocked", "Unblocked",
			"Card Reader Control",
			"Ready", "Card Entered", "Turnstile Unblocked"}, false},
		{"Off children star", "Off", []StateLabel{"Off"}, false},
		{"Non-existent state", "NonExistent", nil, true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := exampleStatechart1.ChildrenStar(tt.state)
			if (err != nil) != tt.wantErr {
				t.Errorf("ChildrenStar() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if diff := cmp.Diff(tt.want, got); diff != "" {
				t.Errorf("ChildrenStar() mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func TestDescendant(t *testing.T) {
	tests := []struct {
		name              string
		state             StateLabel
		potentialAncestor StateLabel
		want              bool
		wantErr           bool
	}{
		{"Blocked is descendant of On", "Blocked", "On", true, false},
		{"On is not descendant of Blocked", "On", "Blocked", false, false},
		{"Off is not descendant of On", "Off", "On", false, false},
		{"Non-existent state", "NonExistent", "On", false, true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := exampleStatechart1.Descendant(tt.state, tt.potentialAncestor)
			if (err != nil) != tt.wantErr {
				t.Errorf("Descendant() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if got != tt.want {
				t.Errorf("Descendant() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestAncestor(t *testing.T) {
	tests := []struct {
		name                string
		state               StateLabel
		potentialDescendant StateLabel
		want                bool
		wantErr             bool
	}{
		{"On is ancestor of Blocked", "On", "Blocked", true, false},
		{"Blocked is not ancestor of On", "Blocked", "On", false, false},
		{"On is not ancestor of Off", "On", "Off", false, false},
		{"Non-existent state", "NonExistent", "On", false, true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := exampleStatechart1.Ancestor(tt.state, tt.potentialDescendant)
			if (err != nil) != tt.wantErr {
				t.Errorf("Ancestor() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if got != tt.want {
				t.Errorf("Ancestor() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestAncestrallyRelated(t *testing.T) {
	tests := []struct {
		name    string
		state1  StateLabel
		state2  StateLabel
		want    bool
		wantErr bool
	}{
		{"On and Blocked are ancestrally related", "On", "Blocked", true, false},
		{"Blocked and On are ancestrally related", "Blocked", "On", true, false},
		{"On and Off are not ancestrally related", "On", "Off", false, false},
		{"Non-existent state", "NonExistent", "On", false, true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := exampleStatechart1.AncestrallyRelated(tt.state1, tt.state2)
			if (err != nil) != tt.wantErr {
				t.Errorf("AncestrallyRelated() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if got != tt.want {
				t.Errorf("AncestrallyRelated() = %v, want %v", got, tt.want)
			}
		})
	}
}
