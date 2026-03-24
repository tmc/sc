package main

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"math/rand"
	"sort"

	"github.com/tmc/sc"
	validationv1 "github.com/tmc/sc/gen/validation/v1"
	validation "github.com/tmc/sc/validation/v1"
	"google.golang.org/protobuf/encoding/protojson"
	"google.golang.org/protobuf/proto"
)

// validationRow records the result of running the full validator against a chart.
type validationRow struct {
	ChartID         string           `json:"chart_id"`
	ChartHash       string           `json:"chart_hash"`
	IsMutated       bool             `json:"is_mutated"`
	MutationFamily  string           `json:"mutation_family,omitempty"`
	MutationOp      string           `json:"mutation_op,omitempty"`
	NStates         int              `json:"n_states"`
	NTransitions    int              `json:"n_transitions"`
	Families        []string         `json:"families"`
	Violations      []violationEntry `json:"violations"`
	ViolatedRuleIDs []int            `json:"violated_rule_ids"`
	NViolations     int              `json:"n_violations"`
	IsWellFormed    bool             `json:"is_well_formed"`
}

type violationEntry struct {
	RuleID   int    `json:"rule_id"`
	RuleName string `json:"rule_name"`
	Severity string `json:"severity"`
	Message  string `json:"message"`
}

// buildValidationRows runs the validator against the clean chart, targeted
// mutations, and synthetic charts designed to trigger all 35 rule IDs.
func buildValidationRows(chart *sc.Statechart, chartID string, families []string, seed int64, mutationCount int) []validationRow {
	var rows []validationRow

	// Clean chart row.
	rows = append(rows, runValidation(chart, chartID, families, false, "", ""))

	// Targeted mutations that exercise specific rule violations.
	rng := rand.New(rand.NewSource(seed))
	mutators := targetedMutators()
	for i := 0; i < mutationCount; i++ {
		m := mutators[rng.Intn(len(mutators))]
		mutated := proto.Clone(chart).(*sc.Statechart)
		op, ok := m.fn(mutated, rng)
		if !ok {
			continue
		}
		rows = append(rows, runValidation(mutated, chartID, families, true, m.family, op))
	}

	// Synthetic charts that exercise reconciled constraints (C1-C17) and
	// other rules that require specific structural patterns.
	for _, sc := range syntheticViolationCharts() {
		id := fmt.Sprintf("%s__synthetic_%s", chartID, sc.family)
		rows = append(rows, runValidation(sc.chart, id, families, true, sc.family, sc.op))
	}

	return rows
}

func runValidation(chart *sc.Statechart, chartID string, families []string, isMutated bool, mutFamily, mutOp string) validationRow {
	v := validation.NewSemanticValidator()
	resp, err := v.ValidateChart(context.Background(), &validationv1.ValidateChartRequest{Chart: chart})

	row := validationRow{
		ChartID:        chartID,
		ChartHash:      validationChartHash(chart),
		IsMutated:      isMutated,
		MutationFamily: mutFamily,
		MutationOp:     mutOp,
		NStates:        countStates(chart.RootState),
		NTransitions:   len(chart.Transitions),
		Families:       families,
		IsWellFormed:   true,
	}

	if err == nil && resp != nil {
		seen := map[int]bool{}
		for _, viol := range resp.Violations {
			id := int(viol.Rule)
			entry := violationEntry{
				RuleID:   id,
				RuleName: viol.Rule.String(),
				Severity: viol.Severity.String(),
				Message:  viol.Message,
			}
			row.Violations = append(row.Violations, entry)
			if !seen[id] {
				seen[id] = true
				row.ViolatedRuleIDs = append(row.ViolatedRuleIDs, id)
			}
			if viol.Severity == validationv1.Severity_ERROR {
				row.IsWellFormed = false
			}
		}
		sort.Ints(row.ViolatedRuleIDs)
	}
	row.NViolations = len(row.Violations)
	return row
}

func validationChartHash(chart *sc.Statechart) string {
	b, err := protojson.MarshalOptions{UseProtoNames: true}.Marshal(chart)
	if err != nil {
		b = []byte(chart.Name)
	}
	sum := sha256.Sum256(b)
	return hex.EncodeToString(sum[:])
}

func countStates(s *sc.State) int {
	if s == nil {
		return 0
	}
	n := 1
	for _, c := range s.Children {
		n += countStates(c)
	}
	return n
}

// targetedMutator describes a mutation operator designed to trigger specific
// validation rule violations.
type targetedMutator struct {
	family string
	fn     func(*sc.Statechart, *rand.Rand) (string, bool)
}

func targetedMutators() []targetedMutator {
	return []targetedMutator{
		// Rule 1: UNIQUE_STATE_LABELS
		{"duplicate_label", mutateDuplicateLabel},
		// Rule 2: SINGLE_DEFAULT_CHILD — add extra initial
		{"extra_initial", mutateExtraInitial},
		// Rule 2: SINGLE_DEFAULT_CHILD — remove initial
		{"remove_initial", mutateRemoveInitial},
		// Rule 3: BASIC_HAS_NO_CHILDREN
		{"basic_with_children", mutateBasicWithChildren},
		// Rule 4: COMPOUND_HAS_CHILDREN
		{"compound_no_children", mutateCompoundNoChildren},
		// Rule 5: DETERMINISTIC_TRANSITION_SELECTION
		{"nondeterministic_transition", mutateNondeterministicTransition},
		// Rule 6: NO_EVENT_BROADCAST_CYCLES
		{"event_broadcast_cycle", mutateEventBroadcastCycle},
		// Rule 7: HISTORY_STATES_WELL_FORMED
		{"invalid_history", mutateInvalidHistory},
		// Rule 10: CHOICE_GUARDS_COMPLETE — remove transition leaving dead end
		{"create_dead_end", mutateCreateDeadEnd},
		// Rule 11: TIMEOUT_EVENTS_UNIQUE
		{"duplicate_timeout_event", mutateDuplicateTimeoutEvent},
		// Rule 13: GUARD_EXPRESSIONS_VALID
		{"invalid_guard", mutateInvalidGuard},
		// Rule 14: EVENT_PARAMETERS_CONSISTENT
		{"undeclared_event", mutateUndeclaredEvent},
		// Rule 15: INTERNAL_TRANSITIONS_VALID — cross-boundary source
		{"cross_boundary_source", mutateCrossBoundarySource},
		// Structural: remove all transitions
		{"remove_all_transitions", mutateRemoveAllTransitions},
		// Structural: add self-loop on final state (Rule 12: ACTION_EXPRESSIONS_VALID)
		{"final_with_outgoing", mutateFinalWithOutgoing},
		// Structural: change state type without adjusting children
		{"wrong_state_type", mutateWrongStateType},
		// Rule 9: FORK_JOIN_BALANCED — overlapping orthogonal regions
		{"overlapping_regions", mutateOverlappingRegions},
	}
}

// --- Mutation operators ---

func mutateDuplicateLabel(chart *sc.Statechart, rng *rand.Rand) (string, bool) {
	states := collectAllStates(chart.RootState)
	if len(states) < 3 {
		return "", false
	}
	nonRoot := states[1:]
	if len(nonRoot) < 2 {
		return "", false
	}
	a := nonRoot[rng.Intn(len(nonRoot))]
	b := nonRoot[rng.Intn(len(nonRoot))]
	for b == a {
		b = nonRoot[rng.Intn(len(nonRoot))]
	}
	old := b.Label
	b.Label = a.Label
	for _, t := range chart.Transitions {
		for i, f := range t.From {
			if f == old {
				t.From[i] = a.Label
			}
		}
		for i, to := range t.To {
			if to == old {
				t.To[i] = a.Label
			}
		}
	}
	return "duplicate label " + a.Label, true
}

func mutateExtraInitial(chart *sc.Statechart, rng *rand.Rand) (string, bool) {
	composites := collectCompositeStates(chart.RootState, sc.StateTypeNormal)
	if len(composites) == 0 {
		return "", false
	}
	parent := composites[rng.Intn(len(composites))]
	for _, c := range parent.Children {
		if !c.IsInitial && !c.IsHistory {
			c.IsInitial = true
			return "extra initial in " + parent.Label, true
		}
	}
	return "", false
}

func mutateRemoveInitial(chart *sc.Statechart, rng *rand.Rand) (string, bool) {
	composites := collectCompositeStates(chart.RootState, sc.StateTypeNormal)
	if len(composites) == 0 {
		return "", false
	}
	parent := composites[rng.Intn(len(composites))]
	for _, c := range parent.Children {
		if c.IsInitial {
			c.IsInitial = false
			return "remove initial from " + parent.Label, true
		}
	}
	return "", false
}

func mutateBasicWithChildren(chart *sc.Statechart, rng *rand.Rand) (string, bool) {
	basics := collectBasicStates(chart.RootState)
	if len(basics) == 0 {
		return "", false
	}
	s := basics[rng.Intn(len(basics))]
	s.Children = []*sc.State{
		{Label: s.Label + "_illegal_child", Type: sc.StateTypeBasic, IsInitial: true},
	}
	return "add child to basic " + s.Label, true
}

func mutateCompoundNoChildren(chart *sc.Statechart, rng *rand.Rand) (string, bool) {
	composites := collectCompositeStates(chart.RootState, sc.StateTypeNormal)
	composites = append(composites, collectCompositeStates(chart.RootState, sc.StateTypeParallel)...)
	if len(composites) == 0 {
		return "", false
	}
	s := composites[rng.Intn(len(composites))]
	s.Children = nil
	return "remove children from " + s.Label, true
}

func mutateNondeterministicTransition(chart *sc.Statechart, rng *rand.Rand) (string, bool) {
	if len(chart.Transitions) == 0 {
		return "", false
	}
	idx := rng.Intn(len(chart.Transitions))
	t := chart.Transitions[idx]
	dup := proto.Clone(t).(*sc.Transition)
	states := collectAllStates(chart.RootState)
	nonRoot := filterNonRoot(states)
	if len(nonRoot) < 2 {
		return "", false
	}
	dup.To = []string{nonRoot[rng.Intn(len(nonRoot))].Label}
	dup.Label = t.Label + "_dup"
	chart.Transitions = append(chart.Transitions, dup)
	return "duplicate transition " + t.Label, true
}

func mutateEventBroadcastCycle(chart *sc.Statechart, rng *rand.Rand) (string, bool) {
	if len(chart.Transitions) < 2 {
		return "", false
	}
	// Pick two transitions and add raise: actions creating a cycle.
	i := rng.Intn(len(chart.Transitions))
	j := rng.Intn(len(chart.Transitions))
	for j == i {
		j = rng.Intn(len(chart.Transitions))
	}
	evA := chart.Transitions[i].Event
	evB := chart.Transitions[j].Event
	if evA == "" || evB == "" || evA == evB {
		return "", false
	}
	chart.Transitions[i].Actions = append(chart.Transitions[i].Actions, &sc.Action{Label: "raise:" + evB})
	chart.Transitions[j].Actions = append(chart.Transitions[j].Actions, &sc.Action{Label: "raise:" + evA})
	return fmt.Sprintf("broadcast cycle %s<->%s", evA, evB), true
}

func mutateInvalidHistory(chart *sc.Statechart, rng *rand.Rand) (string, bool) {
	basics := collectBasicStates(chart.RootState)
	if len(basics) == 0 {
		return "", false
	}
	s := basics[rng.Intn(len(basics))]
	s.IsHistory = true
	s.HistoryType = sc.HistoryType_HISTORY_TYPE_SHALLOW
	return "invalid history on " + s.Label, true
}

func mutateCreateDeadEnd(chart *sc.Statechart, rng *rand.Rand) (string, bool) {
	states := collectAllStates(chart.RootState)
	nonRoot := filterNonRoot(states)
	if len(nonRoot) == 0 {
		return "", false
	}
	target := nonRoot[rng.Intn(len(nonRoot))].Label
	kept := chart.Transitions[:0]
	removed := 0
	for _, t := range chart.Transitions {
		isFrom := false
		for _, f := range t.From {
			if f == target {
				isFrom = true
				break
			}
		}
		if isFrom {
			removed++
		} else {
			kept = append(kept, t)
		}
	}
	chart.Transitions = kept
	if removed == 0 {
		return "", false
	}
	return "create dead end at " + target, true
}

func mutateDuplicateTimeoutEvent(chart *sc.Statechart, _ *rand.Rand) (string, bool) {
	// Add two timeout events with the same label.
	chart.Events = append(chart.Events,
		&sc.Event{Label: "after:500ms"},
		&sc.Event{Label: "after:500ms"},
	)
	return "duplicate timeout event after:500ms", true
}

func mutateInvalidGuard(chart *sc.Statechart, rng *rand.Rand) (string, bool) {
	if len(chart.Transitions) == 0 {
		return "", false
	}
	idx := rng.Intn(len(chart.Transitions))
	t := chart.Transitions[idx]
	t.Guard = &sc.Guard{Expression: ";;invalid;;"}
	label := t.Label
	if label == "" {
		label = t.Event
	}
	return "invalid guard on " + label, true
}

func mutateUndeclaredEvent(chart *sc.Statechart, rng *rand.Rand) (string, bool) {
	if len(chart.Transitions) == 0 {
		return "", false
	}
	idx := rng.Intn(len(chart.Transitions))
	chart.Transitions[idx].Event = "__UNDECLARED_EVENT__"
	return "undeclared event on transition", true
}

func mutateCrossBoundarySource(chart *sc.Statechart, rng *rand.Rand) (string, bool) {
	if len(chart.Transitions) == 0 {
		return "", false
	}
	states := collectAllStates(chart.RootState)
	nonRoot := filterNonRoot(states)
	if len(nonRoot) < 2 {
		return "", false
	}
	idx := rng.Intn(len(chart.Transitions))
	t := chart.Transitions[idx]
	newSrc := nonRoot[rng.Intn(len(nonRoot))].Label
	t.From = []string{newSrc}
	return "cross boundary source " + newSrc, true
}

func mutateRemoveAllTransitions(chart *sc.Statechart, _ *rand.Rand) (string, bool) {
	if len(chart.Transitions) == 0 {
		return "", false
	}
	chart.Transitions = nil
	return "remove all transitions", true
}

func mutateFinalWithOutgoing(chart *sc.Statechart, rng *rand.Rand) (string, bool) {
	if len(chart.Transitions) == 0 {
		return "", false
	}
	t := chart.Transitions[rng.Intn(len(chart.Transitions))]
	if len(t.From) == 0 {
		return "", false
	}
	src := t.From[0]
	s := findState(chart.RootState, src)
	if s == nil {
		return "", false
	}
	s.IsFinal = true
	return "final with outgoing " + src, true
}

func mutateWrongStateType(chart *sc.Statechart, rng *rand.Rand) (string, bool) {
	composites := collectCompositeStates(chart.RootState, sc.StateTypeNormal)
	if len(composites) == 0 {
		basics := collectBasicStates(chart.RootState)
		if len(basics) == 0 {
			return "", false
		}
		s := basics[rng.Intn(len(basics))]
		s.Type = sc.StateTypeNormal
		return "basic->normal without children " + s.Label, true
	}
	s := composites[rng.Intn(len(composites))]
	s.Type = sc.StateTypeBasic
	return "normal->basic with children " + s.Label, true
}

func mutateOverlappingRegions(chart *sc.Statechart, _ *rand.Rand) (string, bool) {
	// Replace root with a parallel state with overlapping region labels.
	chart.RootState = &sc.State{
		Label: "__root__", Type: sc.StateTypeAND,
		Children: []*sc.State{
			{Label: "R1", Type: sc.StateTypeOR, Children: []*sc.State{
				{Label: "X", Type: sc.StateTypeBasic, IsInitial: true},
			}},
			{Label: "R2", Type: sc.StateTypeOR, Children: []*sc.State{
				{Label: "X", Type: sc.StateTypeBasic, IsInitial: true},
			}},
		},
	}
	chart.Transitions = nil
	return "overlapping regions with label X", true
}

// --- Synthetic charts for reconciled constraints (C1-C17) ---

type syntheticChart struct {
	family string
	op     string
	chart  *sc.Statechart
}

func syntheticViolationCharts() []syntheticChart {
	return []syntheticChart{
		// Rule 8: PSEUDO_STATES_WELL_FORMED — empty state label
		{
			family: "empty_label",
			op:     "state with empty label",
			chart: &sc.Statechart{
				Name: "empty_label",
				RootState: &sc.State{Label: "__root__", Type: sc.StateTypeOR, Children: []*sc.State{
					{Label: "", Type: sc.StateTypeBasic, IsInitial: true},
				}},
			},
		},
		// Rule 10: CHOICE_GUARDS_COMPLETE — parallel with <2 children
		{
			family: "parallel_one_child",
			op:     "parallel state with only one region",
			chart: &sc.Statechart{
				Name: "par1",
				RootState: &sc.State{Label: "__root__", Type: sc.StateTypeAND, Children: []*sc.State{
					{Label: "R1", Type: sc.StateTypeOR, Children: []*sc.State{
						{Label: "A", Type: sc.StateTypeBasic, IsInitial: true},
					}},
				}},
			},
		},
		// Rule 13: GUARD_EXPRESSIONS_VALID — invalid guard syntax
		{
			family: "invalid_guard_synth",
			op:     "guard with ;; syntax",
			chart: &sc.Statechart{
				Name: "bad_guard",
				RootState: &sc.State{Label: "__root__", Type: sc.StateTypeOR, Children: []*sc.State{
					{Label: "A", Type: sc.StateTypeBasic, IsInitial: true},
					{Label: "B", Type: sc.StateTypeBasic},
				}},
				Transitions: []*sc.Transition{
					{Label: "t1", From: []string{"A"}, To: []string{"B"}, Event: "E", Guard: &sc.Guard{Expression: ";;bad;;"}},
				},
				Events: []*sc.Event{{Label: "E"}},
			},
		},
		// Rule 14: EVENT_PARAMETERS_CONSISTENT — undeclared event
		{
			family: "undeclared_event_synth",
			op:     "transition uses undeclared event",
			chart: &sc.Statechart{
				Name: "undeclared",
				RootState: &sc.State{Label: "__root__", Type: sc.StateTypeOR, Children: []*sc.State{
					{Label: "A", Type: sc.StateTypeBasic, IsInitial: true},
					{Label: "B", Type: sc.StateTypeBasic},
				}},
				Transitions: []*sc.Transition{
					{Label: "t1", From: []string{"A"}, To: []string{"B"}, Event: "MISSING"},
				},
				Events: []*sc.Event{{Label: "KNOWN"}},
			},
		},
		// Rule 15: INTERNAL_TRANSITIONS_VALID — bad source state reference
		{
			family: "bad_source_synth",
			op:     "transition references nonexistent source",
			chart: &sc.Statechart{
				Name: "bad_src",
				RootState: &sc.State{Label: "__root__", Type: sc.StateTypeOR, Children: []*sc.State{
					{Label: "A", Type: sc.StateTypeBasic, IsInitial: true},
				}},
				Transitions: []*sc.Transition{
					{Label: "t1", From: []string{"NONEXISTENT"}, To: []string{"A"}, Event: "E"},
				},
				Events: []*sc.Event{{Label: "E"}},
			},
		},
		// Rule 18: HISTORY_DEFAULTS_VALID — compound with no initial
		{
			family: "no_initial_synth",
			op:     "compound state with no initial child",
			chart: &sc.Statechart{
				Name: "no_init",
				RootState: &sc.State{Label: "__root__", Type: sc.StateTypeOR, Children: []*sc.State{
					{Label: "A", Type: sc.StateTypeBasic},
					{Label: "B", Type: sc.StateTypeBasic},
				}},
			},
		},
		// Rule 11: TIMEOUT_EVENTS_UNIQUE — duplicate after: events
		{
			family: "duplicate_timeout",
			op:     "duplicate after:500ms timeout events",
			chart: &sc.Statechart{
				Name: "dup_timeout",
				RootState: &sc.State{Label: "__root__", Type: sc.StateTypeOR, Children: []*sc.State{
					{Label: "A", Type: sc.StateTypeBasic, IsInitial: true},
				}},
				Events: []*sc.Event{
					{Label: "after:500ms"},
					{Label: "after:500ms"},
				},
			},
		},
		// Rule 19: C1 — completion transition (no event)
		{
			family: "c1_completion",
			op:     "completion transition A->B",
			chart: &sc.Statechart{
				Name: "c1",
				RootState: &sc.State{Label: "__root__", Type: sc.StateTypeOR, Children: []*sc.State{
					{Label: "A", Type: sc.StateTypeBasic, IsInitial: true},
					{Label: "B", Type: sc.StateTypeBasic},
				}},
				Transitions: []*sc.Transition{
					{Label: "comp", From: []string{"A"}, To: []string{"B"}},
				},
			},
		},
		// Rule 20: C2 — self-triggering cycle via raise:
		{
			family: "c2_self_trigger",
			op:     "raise cycle X<->Y",
			chart: &sc.Statechart{
				Name: "c2",
				RootState: &sc.State{Label: "__root__", Type: sc.StateTypeOR, Children: []*sc.State{
					{Label: "A", Type: sc.StateTypeBasic, IsInitial: true},
					{Label: "B", Type: sc.StateTypeBasic},
				}},
				Transitions: []*sc.Transition{
					{Label: "t1", From: []string{"A"}, To: []string{"B"}, Event: "X", Actions: []*sc.Action{{Label: "raise:Y"}}},
					{Label: "t2", From: []string{"B"}, To: []string{"A"}, Event: "Y", Actions: []*sc.Action{{Label: "raise:X"}}},
				},
				Events: []*sc.Event{{Label: "X"}, {Label: "Y"}},
			},
		},
		// Rule 21: C3 — external conflicts with internal
		{
			family: "c3_ext_int_conflict",
			op:     "external and internal transitions conflict",
			chart: &sc.Statechart{
				Name: "c3",
				RootState: &sc.State{Label: "__root__", Type: sc.StateTypeAND, Children: []*sc.State{
					{Label: "R1", Type: sc.StateTypeOR, Children: []*sc.State{
						{Label: "A", Type: sc.StateTypeBasic, IsInitial: true},
						{Label: "B", Type: sc.StateTypeBasic},
					}},
					{Label: "R2", Type: sc.StateTypeOR, Children: []*sc.State{
						{Label: "C", Type: sc.StateTypeBasic, IsInitial: true},
						{Label: "D", Type: sc.StateTypeBasic},
					}},
				}},
				Transitions: []*sc.Transition{
					{Label: "ext", From: []string{"A"}, To: []string{"B"}, Event: "E", Actions: []*sc.Action{{Label: "raise:I"}}},
					{Label: "int", From: []string{"C"}, To: []string{"D"}, Event: "I"},
				},
				Events: []*sc.Event{{Label: "E"}, {Label: "I"}},
			},
		},
		// Rule 22: C4 — trigger inconsistent transition
		{
			family: "c4_inconsistent_trigger",
			op:     "trigger raises event for inconsistent transition",
			chart: &sc.Statechart{
				Name: "c4",
				RootState: &sc.State{Label: "__root__", Type: sc.StateTypeOR, Children: []*sc.State{
					{Label: "A", Type: sc.StateTypeBasic, IsInitial: true},
					{Label: "B", Type: sc.StateTypeBasic},
				}},
				Transitions: []*sc.Transition{
					{Label: "t1", From: []string{"A"}, To: []string{"B"}, Event: "X", Actions: []*sc.Action{{Label: "raise:Y"}}},
					{Label: "t2", From: []string{"B"}, To: []string{"A"}, Event: "Y", Actions: []*sc.Action{{Label: "raise:X"}}},
				},
				Events: []*sc.Event{{Label: "X"}, {Label: "Y"}},
			},
		},
		// Rule 23: C5 — touched internal precondition
		{
			family: "c5_touched_internal",
			op:     "external touches internal transition source",
			chart: &sc.Statechart{
				Name: "c5",
				RootState: &sc.State{Label: "__root__", Type: sc.StateTypeOR, Children: []*sc.State{
					{Label: "A", Type: sc.StateTypeBasic, IsInitial: true},
					{Label: "B", Type: sc.StateTypeBasic},
					{Label: "C", Type: sc.StateTypeBasic},
				}},
				Transitions: []*sc.Transition{
					{Label: "ext", From: []string{"A"}, To: []string{"B"}, Event: "E", Actions: []*sc.Action{{Label: "raise:I"}}},
					{Label: "int", From: []string{"B"}, To: []string{"C"}, Event: "I"},
				},
				Events: []*sc.Event{{Label: "E"}, {Label: "I"}},
			},
		},
		// Rule 24: C6 — consistent transitions trigger inconsistent ones
		{
			family: "c6_consistent_trigger_inconsistent",
			op:     "consistent t1,t2 trigger inconsistent i1,i2",
			chart: &sc.Statechart{
				Name: "c6",
				RootState: &sc.State{Label: "__root__", Type: sc.StateTypeAND, Children: []*sc.State{
					{Label: "R1", Type: sc.StateTypeOR, Children: []*sc.State{
						{Label: "A", Type: sc.StateTypeBasic, IsInitial: true},
						{Label: "B", Type: sc.StateTypeBasic},
					}},
					{Label: "R2", Type: sc.StateTypeOR, Children: []*sc.State{
						{Label: "C", Type: sc.StateTypeBasic, IsInitial: true},
						{Label: "D", Type: sc.StateTypeBasic},
					}},
					{Label: "R3", Type: sc.StateTypeOR, Children: []*sc.State{
						{Label: "X", Type: sc.StateTypeBasic, IsInitial: true},
						{Label: "Y", Type: sc.StateTypeBasic},
					}},
				}},
				Transitions: []*sc.Transition{
					{Label: "t1", From: []string{"A"}, To: []string{"B"}, Event: "E", Actions: []*sc.Action{{Label: "raise:I1"}}},
					{Label: "t2", From: []string{"C"}, To: []string{"D"}, Event: "E", Actions: []*sc.Action{{Label: "raise:I2"}}},
					{Label: "i1", From: []string{"X"}, To: []string{"Y"}, Event: "I1"},
					{Label: "i2", From: []string{"X"}, To: []string{"Y"}, Event: "I2"},
				},
				Events: []*sc.Event{{Label: "E"}, {Label: "I1"}, {Label: "I2"}},
			},
		},
		// Rule 25: C7 — completion cycle
		{
			family: "c7_completion_cycle",
			op:     "completion cycle A->B->A",
			chart: &sc.Statechart{
				Name: "c7",
				RootState: &sc.State{Label: "__root__", Type: sc.StateTypeOR, Children: []*sc.State{
					{Label: "A", Type: sc.StateTypeBasic, IsInitial: true},
					{Label: "B", Type: sc.StateTypeBasic},
				}},
				Transitions: []*sc.Transition{
					{Label: "c1", From: []string{"A"}, To: []string{"B"}},
					{Label: "c2", From: []string{"B"}, To: []string{"A"}},
				},
			},
		},
		// Rule 26: C8 — completion touches internal
		{
			family: "c8_completion_touches_internal",
			op:     "completion enters state that has internal transition",
			chart: &sc.Statechart{
				Name: "c8",
				RootState: &sc.State{Label: "__root__", Type: sc.StateTypeOR, Children: []*sc.State{
					{Label: "A", Type: sc.StateTypeBasic, IsInitial: true},
					{Label: "B", Type: sc.StateTypeBasic},
					{Label: "C", Type: sc.StateTypeBasic},
				}},
				Transitions: []*sc.Transition{
					{Label: "comp", From: []string{"A"}, To: []string{"B"}},
					{Label: "ext", From: []string{"A"}, To: []string{"C"}, Event: "E", Actions: []*sc.Action{{Label: "raise:I"}}},
					{Label: "int", From: []string{"B"}, To: []string{"C"}, Event: "I"},
				},
				Events: []*sc.Event{{Label: "E"}, {Label: "I"}},
			},
		},
		// Rule 27: C9 — external conflicts with completion
		{
			family: "c9_ext_comp_conflict",
			op:     "external and completion transitions conflict",
			chart: &sc.Statechart{
				Name: "c9",
				RootState: &sc.State{Label: "__root__", Type: sc.StateTypeOR, Children: []*sc.State{
					{Label: "A", Type: sc.StateTypeBasic, IsInitial: true},
					{Label: "B", Type: sc.StateTypeBasic},
					{Label: "C", Type: sc.StateTypeBasic},
				}},
				Transitions: []*sc.Transition{
					{Label: "comp", From: []string{"A"}, To: []string{"B"}},
					{Label: "ext", From: []string{"A"}, To: []string{"C"}, Event: "E"},
				},
				Events: []*sc.Event{{Label: "E"}},
			},
		},
		// Rule 28: C10 — completion conflicts with internal
		{
			family: "c10_comp_int_conflict",
			op:     "completion and internal transitions conflict",
			chart: &sc.Statechart{
				Name: "c10",
				RootState: &sc.State{Label: "__root__", Type: sc.StateTypeOR, Children: []*sc.State{
					{Label: "A", Type: sc.StateTypeBasic, IsInitial: true},
					{Label: "B", Type: sc.StateTypeBasic},
					{Label: "C", Type: sc.StateTypeBasic},
				}},
				Transitions: []*sc.Transition{
					{Label: "comp", From: []string{"A"}, To: []string{"B"}},
					{Label: "ext", From: []string{"A"}, To: []string{"C"}, Event: "E", Actions: []*sc.Action{{Label: "raise:I"}}},
					{Label: "int", From: []string{"A"}, To: []string{"C"}, Event: "I"},
				},
				Events: []*sc.Event{{Label: "E"}, {Label: "I"}},
			},
		},
		// Rule 29: C11 — conflicting completions with different sources
		{
			family: "c11_diff_source_completions",
			op:     "completions from different hierarchy levels conflict",
			chart: &sc.Statechart{
				Name: "c11",
				RootState: &sc.State{Label: "__root__", Type: sc.StateTypeOR, Children: []*sc.State{
					{Label: "P", Type: sc.StateTypeOR, IsInitial: true, Children: []*sc.State{
						{Label: "A", Type: sc.StateTypeBasic, IsInitial: true},
						{Label: "B", Type: sc.StateTypeBasic},
					}},
					{Label: "C", Type: sc.StateTypeBasic},
				}},
				Transitions: []*sc.Transition{
					{Label: "comp1", From: []string{"A"}, To: []string{"B"}},
					{Label: "comp2", From: []string{"P"}, To: []string{"C"}},
				},
			},
		},
		// Rule 30: C12 — cyclic prec relation
		{
			family: "c12_prec_cycle",
			op:     "precedence cycle between external events",
			chart: &sc.Statechart{
				Name: "c12",
				RootState: &sc.State{Label: "__root__", Type: sc.StateTypeAND, Children: []*sc.State{
					{Label: "R1", Type: sc.StateTypeOR, Children: []*sc.State{
						{Label: "A", Type: sc.StateTypeBasic, IsInitial: true},
						{Label: "B", Type: sc.StateTypeBasic},
					}},
					{Label: "R2", Type: sc.StateTypeOR, Children: []*sc.State{
						{Label: "C", Type: sc.StateTypeBasic, IsInitial: true},
						{Label: "D", Type: sc.StateTypeBasic},
					}},
				}},
				Transitions: []*sc.Transition{
					{Label: "t1", From: []string{"A"}, To: []string{"B"}, Event: "E1"},
					{Label: "t2", From: []string{"B"}, To: []string{"A"}, Event: "E2"},
					{Label: "t3", From: []string{"C"}, To: []string{"D"}, Event: "E2"},
					{Label: "t4", From: []string{"D"}, To: []string{"C"}, Event: "E1"},
				},
				Events: []*sc.Event{{Label: "E1"}, {Label: "E2"}},
			},
		},
		// Rule 31: C13 — conflicting same-trigger transitions differ in source/scope
		{
			family: "c13_diff_priority_shape",
			op:     "same trigger but different source hierarchy",
			chart: &sc.Statechart{
				Name: "c13",
				RootState: &sc.State{Label: "__root__", Type: sc.StateTypeOR, Children: []*sc.State{
					{Label: "P", Type: sc.StateTypeOR, IsInitial: true, Children: []*sc.State{
						{Label: "A", Type: sc.StateTypeBasic, IsInitial: true},
						{Label: "B", Type: sc.StateTypeBasic},
					}},
					{Label: "C", Type: sc.StateTypeBasic},
				}},
				Transitions: []*sc.Transition{
					{Label: "t1", From: []string{"A"}, To: []string{"B"}, Event: "E"},
					{Label: "t2", From: []string{"P"}, To: []string{"C"}, Event: "E"},
				},
				Events: []*sc.Event{{Label: "E"}},
			},
		},
		// Rule 32: C14 — multiple generated events
		{
			family: "c14_multi_gen",
			op:     "transition generates two internal events",
			chart: &sc.Statechart{
				Name: "c14",
				RootState: &sc.State{Label: "__root__", Type: sc.StateTypeOR, Children: []*sc.State{
					{Label: "A", Type: sc.StateTypeBasic, IsInitial: true},
					{Label: "B", Type: sc.StateTypeBasic},
				}},
				Transitions: []*sc.Transition{
					{Label: "t1", From: []string{"A"}, To: []string{"B"}, Event: "E", Actions: []*sc.Action{
						{Label: "raise:I1"},
						{Label: "raise:I2"},
					}},
				},
				Events: []*sc.Event{{Label: "E"}},
			},
		},
		// Rule 33: C15 — consistent transitions with same trigger generate different events
		{
			family: "c15_diff_output",
			op:     "consistent transitions same trigger different generated events",
			chart: &sc.Statechart{
				Name: "c15",
				RootState: &sc.State{Label: "__root__", Type: sc.StateTypeAND, Children: []*sc.State{
					{Label: "R1", Type: sc.StateTypeOR, Children: []*sc.State{
						{Label: "A", Type: sc.StateTypeBasic, IsInitial: true},
						{Label: "B", Type: sc.StateTypeBasic},
					}},
					{Label: "R2", Type: sc.StateTypeOR, Children: []*sc.State{
						{Label: "C", Type: sc.StateTypeBasic, IsInitial: true},
						{Label: "D", Type: sc.StateTypeBasic},
					}},
				}},
				Transitions: []*sc.Transition{
					{Label: "t1", From: []string{"A"}, To: []string{"B"}, Event: "E", Actions: []*sc.Action{{Label: "raise:I1"}}},
					{Label: "t2", From: []string{"C"}, To: []string{"D"}, Event: "E", Actions: []*sc.Action{{Label: "raise:I2"}}},
				},
				Events: []*sc.Event{{Label: "E"}},
			},
		},
		// Rule 34: C16 — completion consistent with internal
		{
			family: "c16_comp_consistent_int",
			op:     "completion transition consistent with internal",
			chart: &sc.Statechart{
				Name: "c16",
				RootState: &sc.State{Label: "__root__", Type: sc.StateTypeAND, Children: []*sc.State{
					{Label: "R1", Type: sc.StateTypeOR, Children: []*sc.State{
						{Label: "A", Type: sc.StateTypeBasic, IsInitial: true},
						{Label: "B", Type: sc.StateTypeBasic},
					}},
					{Label: "R2", Type: sc.StateTypeOR, Children: []*sc.State{
						{Label: "C", Type: sc.StateTypeBasic, IsInitial: true},
						{Label: "D", Type: sc.StateTypeBasic},
					}},
				}},
				Transitions: []*sc.Transition{
					{Label: "comp", From: []string{"A"}, To: []string{"B"}},
					{Label: "ext_raise", From: []string{"C"}, To: []string{"D"}, Event: "E", Actions: []*sc.Action{{Label: "raise:I"}}},
					{Label: "int", From: []string{"C"}, To: []string{"D"}, Event: "I"},
				},
				Events: []*sc.Event{{Label: "E"}, {Label: "I"}},
			},
		},
		// Rule 35: C17 — UML internal priority (both external and internal present)
		{
			family: "c17_mixed_events",
			op:     "chart with both external and internal events",
			chart: &sc.Statechart{
				Name: "c17",
				RootState: &sc.State{Label: "__root__", Type: sc.StateTypeOR, Children: []*sc.State{
					{Label: "A", Type: sc.StateTypeBasic, IsInitial: true},
					{Label: "B", Type: sc.StateTypeBasic},
					{Label: "C", Type: sc.StateTypeBasic},
				}},
				Transitions: []*sc.Transition{
					{Label: "ext", From: []string{"A"}, To: []string{"B"}, Event: "E", Actions: []*sc.Action{{Label: "raise:I"}}},
					{Label: "int", From: []string{"B"}, To: []string{"C"}, Event: "I"},
				},
				Events: []*sc.Event{{Label: "E"}, {Label: "I"}},
			},
		},
	}
}

// --- helpers ---

func collectAllStates(root *sc.State) []*sc.State {
	if root == nil {
		return nil
	}
	result := []*sc.State{root}
	for _, c := range root.Children {
		result = append(result, collectAllStates(c)...)
	}
	return result
}

func collectBasicStates(root *sc.State) []*sc.State {
	var result []*sc.State
	for _, s := range collectAllStates(root) {
		if s.Type == sc.StateTypeBasic && s.Label != "__root__" {
			result = append(result, s)
		}
	}
	return result
}

func collectCompositeStates(root *sc.State, typ sc.StateType) []*sc.State {
	var result []*sc.State
	for _, s := range collectAllStates(root) {
		if s.Type == typ && len(s.Children) > 0 && s.Label != "__root__" {
			result = append(result, s)
		}
	}
	return result
}

func filterNonRoot(states []*sc.State) []*sc.State {
	var result []*sc.State
	for _, s := range states {
		if s.Label != "__root__" {
			result = append(result, s)
		}
	}
	return result
}

func findState(root *sc.State, label string) *sc.State {
	if root == nil {
		return nil
	}
	if root.Label == label {
		return root
	}
	for _, c := range root.Children {
		if s := findState(c, label); s != nil {
			return s
		}
	}
	return nil
}
