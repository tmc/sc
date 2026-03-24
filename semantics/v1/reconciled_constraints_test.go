package semantics

import (
	"testing"

	"github.com/tmc/sc"
)

func TestCheckReconciledConstraints_FindsEachConstraint(t *testing.T) {
	tests := []struct {
		name string
		id   ReconciledConstraintID
		opts ReconciledOptions
		make func() *Statechart
	}{
		{
			name: "C1",
			id:   ReconciledConstraintC1,
			opts: ReconciledOptions{MaxSteps: 256, UMLInternalPriority: true},
			make: chartC1,
		},
		{
			name: "C2",
			id:   ReconciledConstraintC2,
			opts: ReconciledOptions{MaxSteps: 256, UMLInternalPriority: true},
			make: chartC2,
		},
		{
			name: "C3",
			id:   ReconciledConstraintC3,
			opts: ReconciledOptions{MaxSteps: 256, UMLInternalPriority: true},
			make: chartC3,
		},
		{
			name: "C4",
			id:   ReconciledConstraintC4,
			opts: ReconciledOptions{MaxSteps: 256, UMLInternalPriority: true},
			make: chartC4,
		},
		{
			name: "C5",
			id:   ReconciledConstraintC5,
			opts: ReconciledOptions{MaxSteps: 256, UMLInternalPriority: true},
			make: chartC5,
		},
		{
			name: "C6",
			id:   ReconciledConstraintC6,
			opts: ReconciledOptions{MaxSteps: 256, UMLInternalPriority: true},
			make: chartC6,
		},
		{
			name: "C7",
			id:   ReconciledConstraintC7,
			opts: ReconciledOptions{MaxSteps: 256, UMLInternalPriority: true},
			make: chartC7,
		},
		{
			name: "C8",
			id:   ReconciledConstraintC8,
			opts: ReconciledOptions{MaxSteps: 256, UMLInternalPriority: true},
			make: chartC8,
		},
		{
			name: "C9",
			id:   ReconciledConstraintC9,
			opts: ReconciledOptions{MaxSteps: 256, UMLInternalPriority: true},
			make: chartC9,
		},
		{
			name: "C10",
			id:   ReconciledConstraintC10,
			opts: ReconciledOptions{MaxSteps: 256, UMLInternalPriority: true},
			make: chartC10,
		},
		{
			name: "C11",
			id:   ReconciledConstraintC11,
			opts: ReconciledOptions{MaxSteps: 256, UMLInternalPriority: true},
			make: chartC11,
		},
		{
			name: "C12",
			id:   ReconciledConstraintC12,
			opts: ReconciledOptions{MaxSteps: 256, UMLInternalPriority: true},
			make: chartC12,
		},
		{
			name: "C13",
			id:   ReconciledConstraintC13,
			opts: ReconciledOptions{MaxSteps: 256, UMLInternalPriority: true},
			make: chartC13,
		},
		{
			name: "C14",
			id:   ReconciledConstraintC14,
			opts: ReconciledOptions{MaxSteps: 256, UMLInternalPriority: true},
			make: chartC14,
		},
		{
			name: "C15",
			id:   ReconciledConstraintC15,
			opts: ReconciledOptions{MaxSteps: 256, UMLInternalPriority: true},
			make: chartC15,
		},
		{
			name: "C16",
			id:   ReconciledConstraintC16,
			opts: ReconciledOptions{MaxSteps: 256, UMLInternalPriority: true},
			make: chartC16,
		},
		{
			name: "C17",
			id:   ReconciledConstraintC17,
			opts: ReconciledOptions{MaxSteps: 256},
			make: chartC3,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			violations, err := tt.make().CheckReconciledConstraints(tt.opts)
			if err != nil {
				t.Fatalf("CheckReconciledConstraints() error = %v", err)
			}
			if !hasConstraint(violations, tt.id) {
				t.Fatalf("CheckReconciledConstraints() missing %s in %+v", tt.id, violations)
			}
		})
	}
}

func hasConstraint(violations []ReconciledConstraintViolation, want ReconciledConstraintID) bool {
	for _, violation := range violations {
		if violation.Constraint == want {
			return true
		}
	}
	return false
}

func chartC1() *Statechart {
	return newChart(
		orState("__root__", true, leaf("A", true), leaf("B", false)),
		&sc.Transition{Label: "complete", From: []string{"A"}, To: []string{"B"}},
	)
}

func chartC2() *Statechart {
	return newChart(
		andState("__root__",
			orState("left", true, leaf("A", true), leaf("B", false)),
			orState("right", false, leaf("C", true), leaf("D", false)),
		),
		&sc.Transition{Label: "t1", From: []string{"A"}, To: []string{"B"}, Event: "e", Actions: []*sc.Action{{Label: "raise:i"}}},
		&sc.Transition{Label: "t2", From: []string{"C"}, To: []string{"D"}, Event: "i", Actions: []*sc.Action{{Label: "raise:e"}}},
	)
}

func chartC3() *Statechart {
	return newChart(
		andState("__root__",
			orState("left", true, leaf("A", true), leaf("B", false), leaf("C", false)),
			orState("right", false, leaf("D", true), leaf("E", false)),
		),
		&sc.Transition{Label: "external", From: []string{"A"}, To: []string{"B"}, Event: "e"},
		&sc.Transition{Label: "internal", From: []string{"A"}, To: []string{"C"}, Event: "i"},
		&sc.Transition{Label: "generate", From: []string{"D"}, To: []string{"E"}, Event: "f", Actions: []*sc.Action{{Label: "raise:i"}}},
	)
}

func chartC4() *Statechart {
	return newChart(
		orState("__root__", true, leaf("A", true), leaf("B", false), leaf("C", false)),
		&sc.Transition{Label: "trigger", From: []string{"A"}, To: []string{"B"}, Event: "e", Actions: []*sc.Action{{Label: "raise:i"}}},
		&sc.Transition{Label: "inconsistent", From: []string{"B"}, To: []string{"C"}, Event: "i"},
	)
}

func chartC5() *Statechart {
	return newChart(
		andState("__root__",
			orState("left", true, leaf("A", true), leaf("B", false), leaf("C", false)),
			orState("right", false, leaf("D", true), leaf("E", false)),
		),
		&sc.Transition{Label: "external", From: []string{"A"}, To: []string{"B"}, Event: "f"},
		&sc.Transition{Label: "internal", From: []string{"B"}, To: []string{"C"}, Event: "i"},
		&sc.Transition{Label: "trigger", From: []string{"D"}, To: []string{"E"}, Event: "e", Actions: []*sc.Action{{Label: "raise:i"}}},
	)
}

func chartC6() *Statechart {
	return newChart(
		andState("__root__",
			orState("left", true, leaf("A", true), leaf("B", false)),
			orState("middle", false, leaf("D", true), leaf("E", false)),
			orState("right", false, leaf("X", true), leaf("Y", false), leaf("Z", false)),
		),
		&sc.Transition{Label: "left_e", From: []string{"A"}, To: []string{"B"}, Event: "e", Actions: []*sc.Action{{Label: "raise:i"}}},
		&sc.Transition{Label: "middle_f", From: []string{"D"}, To: []string{"E"}, Event: "f", Actions: []*sc.Action{{Label: "raise:j"}}},
		&sc.Transition{Label: "right_i", From: []string{"X"}, To: []string{"Y"}, Event: "i"},
		&sc.Transition{Label: "right_j", From: []string{"X"}, To: []string{"Z"}, Event: "j"},
	)
}

func chartC7() *Statechart {
	return newChart(
		orState("__root__", true, leaf("A", true), leaf("B", false)),
		&sc.Transition{Label: "ab", From: []string{"A"}, To: []string{"B"}},
		&sc.Transition{Label: "ba", From: []string{"B"}, To: []string{"A"}},
	)
}

func chartC8() *Statechart {
	return newChart(
		andState("__root__",
			orState("left", true, leaf("A", true), leaf("B", false), leaf("C", false)),
			orState("right", false, leaf("D", true), leaf("E", false)),
		),
		&sc.Transition{Label: "complete", From: []string{"A"}, To: []string{"B"}},
		&sc.Transition{Label: "internal", From: []string{"B"}, To: []string{"C"}, Event: "i"},
		&sc.Transition{Label: "generate", From: []string{"D"}, To: []string{"E"}, Event: "e", Actions: []*sc.Action{{Label: "raise:i"}}},
	)
}

func chartC9() *Statechart {
	return newChart(
		orState("__root__", true, leaf("A", true), leaf("B", false), leaf("C", false)),
		&sc.Transition{Label: "external", From: []string{"A"}, To: []string{"B"}, Event: "e"},
		&sc.Transition{Label: "complete", From: []string{"A"}, To: []string{"C"}},
	)
}

func chartC10() *Statechart {
	return newChart(
		andState("__root__",
			orState("left", true, leaf("A", true), leaf("B", false), leaf("C", false)),
			orState("right", false, leaf("D", true), leaf("E", false)),
		),
		&sc.Transition{Label: "complete", From: []string{"A"}, To: []string{"B"}},
		&sc.Transition{Label: "internal", From: []string{"A"}, To: []string{"C"}, Event: "i"},
		&sc.Transition{Label: "generate", From: []string{"D"}, To: []string{"E"}, Event: "e", Actions: []*sc.Action{{Label: "raise:i"}}},
	)
}

func chartC11() *Statechart {
	return newChart(
		orState("__root__", true,
			orState("P", true, leaf("A", true), leaf("B", false)),
			leaf("C", false),
		),
		&sc.Transition{Label: "ab", From: []string{"A"}, To: []string{"B"}},
		&sc.Transition{Label: "pc", From: []string{"P"}, To: []string{"C"}},
	)
}

func chartC12() *Statechart {
	return newChart(
		orState("__root__", true, leaf("A", true), leaf("B", false)),
		&sc.Transition{Label: "ab", From: []string{"A"}, To: []string{"B"}, Event: "e"},
		&sc.Transition{Label: "ba", From: []string{"B"}, To: []string{"A"}, Event: "f"},
	)
}

func chartC13() *Statechart {
	return newChart(
		orState("__root__", true,
			orState("P", true, leaf("A", true), leaf("B", false)),
			leaf("C", false),
		),
		&sc.Transition{Label: "ab", From: []string{"A"}, To: []string{"B"}, Event: "e"},
		&sc.Transition{Label: "pc", From: []string{"P"}, To: []string{"C"}, Event: "e"},
	)
}

func chartC14() *Statechart {
	return newChart(
		orState("__root__", true, leaf("A", true), leaf("B", false)),
		&sc.Transition{
			Label:   "generate-many",
			From:    []string{"A"},
			To:      []string{"B"},
			Event:   "e",
			Actions: []*sc.Action{{Label: "raise:i"}, {Label: "raise:j"}},
		},
	)
}

func chartC15() *Statechart {
	return newChart(
		andState("__root__",
			orState("left", true, leaf("A", true), leaf("B", false)),
			orState("right", false, leaf("D", true), leaf("E", false)),
		),
		&sc.Transition{Label: "left_e", From: []string{"A"}, To: []string{"B"}, Event: "e", Actions: []*sc.Action{{Label: "raise:i"}}},
		&sc.Transition{Label: "right_e", From: []string{"D"}, To: []string{"E"}, Event: "e", Actions: []*sc.Action{{Label: "raise:j"}}},
	)
}

func chartC16() *Statechart {
	return newChart(
		andState("__root__",
			orState("left", true, leaf("A", true), leaf("B", false)),
			orState("middle", false, leaf("D", true), leaf("E", false)),
			orState("right", false, leaf("F", true), leaf("G", false)),
		),
		&sc.Transition{Label: "complete", From: []string{"A"}, To: []string{"B"}},
		&sc.Transition{Label: "internal", From: []string{"D"}, To: []string{"E"}, Event: "i"},
		&sc.Transition{Label: "generate", From: []string{"F"}, To: []string{"G"}, Event: "e", Actions: []*sc.Action{{Label: "raise:i"}}},
	)
}

func newChart(root *sc.State, transitions ...*sc.Transition) *Statechart {
	return NewStatechart(&sc.Statechart{
		RootState:   root,
		Transitions: transitions,
	})
}

func leaf(label string, initial bool) *sc.State {
	return &sc.State{
		Label:     label,
		Type:      sc.StateTypeBasic,
		IsInitial: initial,
	}
}

func orState(label string, initial bool, children ...*sc.State) *sc.State {
	return &sc.State{
		Label:     label,
		Type:      sc.StateTypeNormal,
		IsInitial: initial,
		Children:  children,
	}
}

func andState(label string, children ...*sc.State) *sc.State {
	return &sc.State{
		Label:    label,
		Type:     sc.StateTypeParallel,
		Children: children,
	}
}
