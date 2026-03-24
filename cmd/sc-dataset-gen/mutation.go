package main

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"math/rand"

	"github.com/tmc/sc"
	"github.com/tmc/sc/analysis"
	"google.golang.org/protobuf/encoding/protojson"
	"google.golang.org/protobuf/proto"
)

type mutationRow struct {
	ChartID          string           `json:"chart_id"`
	CleanChartHash   string           `json:"clean_chart_hash"`
	MutatedChartHash string           `json:"mutated_chart_hash"`
	MutationFamily   string           `json:"mutation_family"`
	MutationOp       string           `json:"mutation_op"`
	AffectedLabels   []string         `json:"affected_labels"`
	BehavioralDelta  *behavioralDelta `json:"behavioral_delta,omitempty"`
}

type behavioralDelta struct {
	NewReachable     []string `json:"new_reachable,omitempty"`
	LostReachable    []string `json:"lost_reachable,omitempty"`
	NewDeadEnds      []string `json:"new_dead_ends,omitempty"`
	ResolvedDeadEnds []string `json:"resolved_dead_ends,omitempty"`
}

// generateMutations creates mutated variants of a chart and pairs them with the original.
func generateMutations(chart *sc.Statechart, chartID string, seed int64, count int) []mutationRow {
	rng := rand.New(rand.NewSource(seed))
	cleanHash := chartHash(chart)

	// Build clean graph for behavioral comparison.
	cleanGraph := analysis.BuildGraph(chart)
	cleanReachable := toSet(cleanGraph.ReachableFromInitial())
	cleanDeadEnds := toSet(cleanGraph.DeadEnds())

	mutators := []struct {
		family string
		fn     func(*sc.Statechart, *rand.Rand) (string, []string, bool)
	}{
		{"remove_transition", mutateRemoveTransition},
		{"change_target", mutateChangeTarget},
		{"remove_guard", mutateRemoveGuard},
	}

	var rows []mutationRow
	for i := 0; i < count; i++ {
		mutated := proto.Clone(chart).(*sc.Statechart)
		m := mutators[rng.Intn(len(mutators))]
		op, affected, ok := m.fn(mutated, rng)
		if !ok {
			continue
		}

		mutHash := chartHash(mutated)

		// Compute behavioral delta.
		mutGraph := analysis.BuildGraph(mutated)
		mutReachable := toSet(mutGraph.ReachableFromInitial())
		mutDeadEnds := toSet(mutGraph.DeadEnds())

		delta := &behavioralDelta{
			NewReachable:     setDiff(mutReachable, cleanReachable),
			LostReachable:    setDiff(cleanReachable, mutReachable),
			NewDeadEnds:      setDiff(mutDeadEnds, cleanDeadEnds),
			ResolvedDeadEnds: setDiff(cleanDeadEnds, mutDeadEnds),
		}

		rows = append(rows, mutationRow{
			ChartID:          chartID,
			CleanChartHash:   cleanHash,
			MutatedChartHash: mutHash,
			MutationFamily:   m.family,
			MutationOp:       op,
			AffectedLabels:   affected,
			BehavioralDelta:  delta,
		})
	}
	return rows
}

func mutateRemoveTransition(chart *sc.Statechart, rng *rand.Rand) (string, []string, bool) {
	if len(chart.Transitions) == 0 {
		return "", nil, false
	}
	idx := rng.Intn(len(chart.Transitions))
	t := chart.Transitions[idx]
	label := t.Label
	if label == "" {
		label = fmt.Sprintf("%v->%v", t.From, t.To)
	}
	affected := append(append([]string(nil), t.From...), t.To...)
	chart.Transitions = append(chart.Transitions[:idx], chart.Transitions[idx+1:]...)
	return fmt.Sprintf("remove transition %s", label), affected, true
}

func mutateChangeTarget(chart *sc.Statechart, rng *rand.Rand) (string, []string, bool) {
	if len(chart.Transitions) == 0 {
		return "", nil, false
	}

	// Collect all state labels.
	var labels []string
	var walk func(s *sc.State)
	walk = func(s *sc.State) {
		if s == nil {
			return
		}
		if s.Label != "__root__" {
			labels = append(labels, s.Label)
		}
		for _, c := range s.Children {
			walk(c)
		}
	}
	walk(chart.RootState)
	if len(labels) < 2 {
		return "", nil, false
	}

	idx := rng.Intn(len(chart.Transitions))
	t := chart.Transitions[idx]
	oldTo := append([]string(nil), t.To...)
	newTarget := labels[rng.Intn(len(labels))]
	t.To = []string{newTarget}
	affected := append(oldTo, newTarget)
	return fmt.Sprintf("retarget transition to %s", newTarget), affected, true
}

func mutateRemoveGuard(chart *sc.Statechart, rng *rand.Rand) (string, []string, bool) {
	// Find guarded transitions.
	var guarded []int
	for i, t := range chart.Transitions {
		if t != nil && t.Guard != nil {
			guarded = append(guarded, i)
		}
	}
	if len(guarded) == 0 {
		return "", nil, false
	}
	idx := guarded[rng.Intn(len(guarded))]
	t := chart.Transitions[idx]
	label := t.Label
	if label == "" {
		label = fmt.Sprintf("%v->%v", t.From, t.To)
	}
	t.Guard = nil
	affected := append(append([]string(nil), t.From...), t.To...)
	return fmt.Sprintf("remove guard from %s", label), affected, true
}

func chartHash(chart *sc.Statechart) string {
	b, err := protojson.MarshalOptions{
		UseProtoNames:   true,
		EmitUnpopulated: false,
	}.Marshal(chart)
	if err != nil {
		b = []byte(chart.Name)
	}
	sum := sha256.Sum256(b)
	return hex.EncodeToString(sum[:])
}

func toSet(labels []string) map[string]bool {
	s := make(map[string]bool, len(labels))
	for _, l := range labels {
		s[l] = true
	}
	return s
}

func setDiff(a, b map[string]bool) []string {
	var diff []string
	for k := range a {
		if !b[k] {
			diff = append(diff, k)
		}
	}
	return diff
}
