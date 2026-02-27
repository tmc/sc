package analysis

import (
	"testing"

	sc "github.com/tmc/sc"
)

func TestDiff_DoesNotCollapseParallelGuardedTransitions(t *testing.T) {
	chart1 := chartWithTransitions([]*sc.Transition{
		{From: []string{"A"}, To: []string{"B"}, Event: "GO", Guard: &sc.Guard{Expression: "x"}},
		{From: []string{"A"}, To: []string{"B"}, Event: "GO", Guard: &sc.Guard{Expression: "y"}},
	})
	chart2 := chartWithTransitions([]*sc.Transition{
		{From: []string{"A"}, To: []string{"B"}, Event: "GO", Guard: &sc.Guard{Expression: "x"}},
	})

	diff := Diff(chart1, chart2)

	if got, want := len(diff.TransitionsRemoved), 1; got != want {
		t.Fatalf("len(TransitionsRemoved) = %d, want %d", got, want)
	}
	if diff.TransitionsRemoved[0].Guard != "y" {
		t.Fatalf("removed transition guard = %q, want %q", diff.TransitionsRemoved[0].Guard, "y")
	}
	if got := len(diff.TransitionsModified); got != 0 {
		t.Fatalf("len(TransitionsModified) = %d, want 0", got)
	}
	if diff.IsCompatible {
		t.Fatal("IsCompatible = true, want false")
	}
}

func TestDiff_OneToOneGuardChangeIsModified(t *testing.T) {
	chart1 := chartWithTransitions([]*sc.Transition{
		{From: []string{"A"}, To: []string{"B"}, Event: "GO", Guard: &sc.Guard{Expression: "old"}},
	})
	chart2 := chartWithTransitions([]*sc.Transition{
		{From: []string{"A"}, To: []string{"B"}, Event: "GO", Guard: &sc.Guard{Expression: "new"}},
	})

	diff := Diff(chart1, chart2)

	if got := len(diff.TransitionsRemoved); got != 0 {
		t.Fatalf("len(TransitionsRemoved) = %d, want 0", got)
	}
	if got := len(diff.TransitionsAdded); got != 0 {
		t.Fatalf("len(TransitionsAdded) = %d, want 0", got)
	}
	if got, want := len(diff.TransitionsModified), 1; got != want {
		t.Fatalf("len(TransitionsModified) = %d, want %d", got, want)
	}
	if got, want := diff.TransitionsModified[0].OldGuard, "old"; got != want {
		t.Fatalf("OldGuard = %q, want %q", got, want)
	}
	if got, want := diff.TransitionsModified[0].NewGuard, "new"; got != want {
		t.Fatalf("NewGuard = %q, want %q", got, want)
	}
}

func chartWithTransitions(transitions []*sc.Transition) *sc.Statechart {
	return &sc.Statechart{
		Name: "chart",
		RootState: &sc.State{
			Label: "root",
			Type:  sc.StateTypeOR,
			Children: []*sc.State{
				{Label: "A", Type: sc.StateTypeBasic, IsInitial: true},
				{Label: "B", Type: sc.StateTypeBasic},
			},
		},
		Transitions: transitions,
	}
}
