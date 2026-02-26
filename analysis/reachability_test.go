package analysis

import (
	"reflect"
	"testing"

	sc "github.com/tmc/sc"
)

func TestReachableFromInitial_OrDefaultChildOnly(t *testing.T) {
	chart := &sc.Statechart{
		RootState: &sc.State{
			Label: "root",
			Type:  sc.StateTypeOR,
			Children: []*sc.State{
				{Label: "A", Type: sc.StateTypeBasic, IsInitial: true},
				{
					Label: "B",
					Type:  sc.StateTypeOR,
					Children: []*sc.State{
						{Label: "B1", Type: sc.StateTypeBasic, IsInitial: true},
						{Label: "B2", Type: sc.StateTypeBasic},
					},
				},
			},
		},
	}

	got := BuildGraph(chart).ReachableFromInitial()
	want := []string{"A"}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("ReachableFromInitial() = %v, want %v", got, want)
	}
}

func TestReachableFromInitial_DoesNotIncludeUnknownTargets(t *testing.T) {
	chart := &sc.Statechart{
		RootState: &sc.State{
			Label: "root",
			Type:  sc.StateTypeOR,
			Children: []*sc.State{
				{Label: "A", Type: sc.StateTypeBasic, IsInitial: true},
			},
		},
		Transitions: []*sc.Transition{
			{From: []string{"A"}, To: []string{"X"}, Event: "GO"},
		},
	}

	got := BuildGraph(chart).ReachableFromInitial()
	want := []string{"A"}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("ReachableFromInitial() = %v, want %v", got, want)
	}
}

func TestReachableFromInitial_AndEntersAllRegions(t *testing.T) {
	chart := &sc.Statechart{
		RootState: &sc.State{
			Label: "root",
			Type:  sc.StateTypeOR,
			Children: []*sc.State{
				{
					Label:     "P",
					Type:      sc.StateTypeAND,
					IsInitial: true,
					Children: []*sc.State{
						{Label: "C1", Type: sc.StateTypeBasic},
						{Label: "C2", Type: sc.StateTypeBasic},
					},
				},
			},
		},
	}

	got := BuildGraph(chart).ReachableFromInitial()
	want := []string{"C1", "C2", "P"}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("ReachableFromInitial() = %v, want %v", got, want)
	}
}
