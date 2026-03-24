package semantics

import (
	"sort"
	"testing"

	"github.com/tmc/sc"
)

func TestReconciledFigure1Semantics(t *testing.T) {
	chart := NewStatechart(&sc.Statechart{
		RootState: &sc.State{
			Label: "__root__",
			Type:  sc.StateTypeParallel,
			Children: []*sc.State{
				{
					Label: "left",
					Type:  sc.StateTypeNormal,
					Children: []*sc.State{
						{Label: "s1", IsInitial: true},
						{Label: "s2"},
					},
				},
				{
					Label: "middle",
					Type:  sc.StateTypeNormal,
					Children: []*sc.State{
						{Label: "s3", IsInitial: true},
						{Label: "s4"},
					},
				},
				{
					Label: "right",
					Type:  sc.StateTypeNormal,
					Children: []*sc.State{
						{Label: "s5", IsInitial: true},
						{Label: "s6"},
					},
				},
			},
		},
		Transitions: []*sc.Transition{
			{
				Label: "left_e",
				From:  []string{"s1"},
				To:    []string{"s2"},
				Event: "e",
				Actions: []*sc.Action{
					{Label: "raise:i"},
				},
			},
			{
				Label: "middle_i",
				From:  []string{"s3"},
				To:    []string{"s4"},
				Event: "i",
			},
			{
				Label: "right_f",
				From:  []string{"s5"},
				To:    []string{"s6"},
				Event: "f",
			},
		},
	})

	runtime, err := chart.normalizedRuntime(nil)
	if err != nil {
		t.Fatalf("normalizedRuntime() error = %v", err)
	}

	fixpoint, err := chart.Reactions(SemanticsFixpoint, runtime, "e", "f")
	if err != nil {
		t.Fatalf("fixpoint reactions error = %v", err)
	}
	if len(fixpoint) != 1 {
		t.Fatalf("fixpoint reactions = %d, want 1", len(fixpoint))
	}
	assertTransitionLabels(t, fixpoint[0].Steps[0], "left_e", "middle_i", "right_f")
	assertActiveLeafStates(t, fixpoint[0].Final.Configuration, "s2", "s4", "s6")

	statemate, err := chart.Reactions(SemanticsStatemate, runtime, "e", "f")
	if err != nil {
		t.Fatalf("statemate reactions error = %v", err)
	}
	if len(statemate) != 1 {
		t.Fatalf("statemate reactions = %d, want 1", len(statemate))
	}
	if len(statemate[0].Steps) != 2 {
		t.Fatalf("statemate steps = %d, want 2", len(statemate[0].Steps))
	}
	assertTransitionLabels(t, statemate[0].Steps[0], "left_e", "right_f")
	assertTransitionLabels(t, statemate[0].Steps[1], "middle_i")
	assertActiveLeafStates(t, statemate[0].Final.Configuration, "s2", "s4", "s6")

	seStatemate, err := chart.Reactions(SemanticsSingleEventStatemate, runtime, "e", "f")
	if err != nil {
		t.Fatalf("single-event statemate reactions error = %v", err)
	}
	if len(seStatemate) != 2 {
		t.Fatalf("single-event statemate reactions = %d, want 2", len(seStatemate))
	}
	for _, reaction := range seStatemate {
		assertActiveLeafStates(t, reaction.Final.Configuration, "s2", "s4", "s6")
	}

	uml, err := chart.Reactions(SemanticsUML, runtime, "e", "f")
	if err != nil {
		t.Fatalf("uml reactions error = %v", err)
	}
	if len(uml) != 2 {
		t.Fatalf("uml reactions = %d, want 2", len(uml))
	}

	firstSteps := make(map[string]bool)
	for _, reaction := range uml {
		if len(reaction.Steps) != 3 {
			t.Fatalf("uml steps = %d, want 3", len(reaction.Steps))
		}
		assertActiveLeafStates(t, reaction.Final.Configuration, "s2", "s4", "s6")
		firstSteps[transitionSetKey(reaction.Steps[0].Transitions)] = true
	}
	if !firstSteps["left_e"] || !firstSteps["right_f"] || len(firstSteps) != 2 {
		t.Fatalf("uml first steps = %v, want {left_e,right_f}", firstSteps)
	}
}

func assertTransitionLabels(t *testing.T, step *sc.Step, want ...string) {
	t.Helper()
	if step == nil {
		t.Fatal("step is nil")
	}
	got := make([]string, 0, len(step.Transitions))
	for _, transition := range step.Transitions {
		got = append(got, transition.Label)
	}
	sort.Strings(got)
	sort.Strings(want)
	if len(got) != len(want) {
		t.Fatalf("transitions = %v, want %v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("transitions = %v, want %v", got, want)
		}
	}
}

func assertActiveLeafStates(t *testing.T, config *sc.Configuration, want ...string) {
	t.Helper()
	gotSet := make(map[string]bool)
	for _, state := range config.States {
		gotSet[state.Label] = true
	}
	for _, label := range want {
		if !gotSet[label] {
			t.Fatalf("configuration %v does not contain %s", labels(config), label)
		}
	}
}

func labels(config *sc.Configuration) []string {
	var out []string
	for _, state := range config.States {
		out = append(out, state.Label)
	}
	sort.Strings(out)
	return out
}
